import argparse
import os
import os.path as osp
import math
import time
import json
import platform
import psutil
import datetime
import numpy as np
import torch
import cv2
import warnings
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Suppress specific noisy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="mmcv.ops.sparse_structure")

# MM Libraries
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS, DATASETS
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.utils import register_all_modules 
import mmdet3d

# NuScenes & Geometry
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionBox

# Open3D (Optional)
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

# ==============================================================================
# 1. UTILS & TRANSFORMATIONS
# ==============================================================================

def setup_env():
    register_all_modules(init_default_scope=True)

def get_system_info():
    try: sys_mem = round(psutil.virtual_memory().total / (1024**3), 2)
    except: sys_mem = "N/A"
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "mmdet3d": mmdet3d.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "memory_gb": sys_mem
    }

def _quat_rot(q): return q.rotation_matrix.astype(np.float32)
def _Rt(R, t): 
    T = np.eye(4, dtype=np.float32); T[:3,:3]=R; T[:3,3]=t 
    return T

def _to_numpy_points(pts_item):
    if pts_item is None: return np.zeros((0, 5), dtype=np.float32)
    if hasattr(pts_item, 'tensor'): t = pts_item.tensor
    else: t = pts_item
    if isinstance(t, (list, tuple)) and len(t) > 0: t = t[0]
    if torch.is_tensor(t): return t.detach().cpu().numpy()
    return np.asarray(t)

# ==============================================================================
# 2. VISUALIZATION HELPERS
# ==============================================================================

def get_gt_boxes(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    sd = nusc.get('sample_data', lidar_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])
    boxes = []
    for ann in sample['anns']:
        a = nusc.get('sample_annotation', ann)
        box = Box(a['translation'], a['size'], Quaternion(a['rotation']))
        box.translate(-np.array(pose['translation']))
        box.rotate(Quaternion(pose['rotation']).inverse)
        box.translate(-np.array(cs['translation']))
        box.rotate(Quaternion(cs['rotation']).inverse)
        z_bot = box.center[2] - box.wlh[2]/2
        boxes.append([box.center[0], box.center[1], z_bot, box.wlh[1], box.wlh[0], box.wlh[2], box.orientation.yaw_pitch_roll[0]])
    return np.array(boxes, dtype=np.float32) if boxes else None

def project_points(pts, P):
    N = pts.shape[0]
    h = np.hstack([pts[:,:3], np.ones((N,1), dtype=np.float32)])
    c = (P @ h.T).T
    z = c[:, 2]
    mask = z > 0.1
    return c[mask, :2]/c[mask, 2:3], z[mask], mask

def boxes_to_lineset(boxes, color):
    if boxes is None or len(boxes) == 0: return None
    points, lines, colors = [], [], []
    for i, b in enumerate(boxes):
        x, y, z, dx, dy, dz, yaw = b[:7]
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        xc = [dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2]
        yc = [dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2]
        zc = [0, 0, 0, 0, dz, dz, dz, dz]
        corn = (R @ np.vstack([xc, yc, zc])).T + np.array([x, y, z])
        base = i * 8
        points.extend(corn.tolist())
        lines.extend([[base+u, base+v] for u,v in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]])
        colors.extend([color]*12)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def save_ply_files(out_dir, token, pts, pred_boxes, gt_boxes):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    o3d.io.write_point_cloud(osp.join(out_dir, f"{token}_points.ply"), pcd)
    ls_pred = boxes_to_lineset(pred_boxes, [0, 1, 0])
    if ls_pred: o3d.io.write_line_set(osp.join(out_dir, f"{token}_pred.ply"), ls_pred)
    ls_gt = boxes_to_lineset(gt_boxes, [1, 0, 0])
    if ls_gt: o3d.io.write_line_set(osp.join(out_dir, f"{token}_gt.ply"), ls_gt)

def run_open3d_viz(pts, pred_boxes, gt_boxes, window_name="3D Detection"):
    if not HAS_OPEN3D: return
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    colors = np.zeros((pts.shape[0], 3)); z = pts[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors[:, 1] = z_norm 
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)
    if pred_boxes is not None:
        ls = boxes_to_lineset(pred_boxes, [0, 1, 0])
        if ls: vis.add_geometry(ls)
    if gt_boxes is not None:
        ls = boxes_to_lineset(gt_boxes, [1, 0, 0])
        if ls: vis.add_geometry(ls)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    vis.run()
    vis.destroy_window()

def draw_2d_multiview(paths, pts, l2i_list, pred_boxes, gt_boxes, token, out_dir):
    if not paths: return
    canvases = []
    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None: continue
        h, w = img.shape[:2]
        
        # [VISUALIZATION LOGIC MATCHES LOADER]
        scale = 0.48
        nH, nW = int(h*scale), int(w*scale)
        img = cv2.resize(img, (nW, nH))
        sx, sy = (nW - 704)//2, (nH - 256)//2
        img = img[sy:sy+256, sx:sx+704].copy()
        
        P = l2i_list[i]
        
        uv, z, _ = project_points(pts, P)
        if len(uv) > 0:
            d_norm = np.clip(z, 0, 60)/60 * 255
            cols = cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_JET).reshape(-1,3)
            h_img, w_img = img.shape[:2]
            for j, (x,y) in enumerate(uv.astype(int)):
                if 0<=x<w_img and 0<=y<h_img: img[y,x] = cols[j]

        def draw_b(boxes, c):
            if boxes is None: return
            for b in boxes:
                if len(b)<7: continue
                x,y,zb,dx,dy,dz,yaw = b[:7]
                c_cos, s_sin = math.cos(yaw), math.sin(yaw)
                R = np.array([[c_cos,-s_sin,0],[s_sin,c_cos,0],[0,0,1]])
                xc = [dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2]
                yc = [dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2]
                zc = [0, 0, 0, 0, dz, dz, dz, dz]
                corn = (R @ np.vstack([xc,yc,zc])).T + np.array([x,y,zb])
                uv_b, z_b, _ = project_points(corn, P)
                if len(uv_b) < 8: continue
                edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                for u,v in edges:
                    p1, p2 = tuple(uv_b[u].astype(int)), tuple(uv_b[v].astype(int))
                    cv2.line(img, p1, p2, c, 2, cv2.LINE_AA)

        draw_b(gt_boxes, (0,0,255))
        draw_b(pred_boxes, (0,255,0))
        canvases.append(img)

    if len(canvases) == 6: final = np.vstack([np.hstack(canvases[:3]), np.hstack(canvases[3:])])
    elif len(canvases) > 0: final = np.hstack(canvases)
    else: return

    cv2.imwrite(osp.join(out_dir, f"{token}_multiview.jpg"), final)

# ==============================================================================
# 3. DATA LOADERS
# ==============================================================================

class BaseLoader(Dataset):
    def __len__(self): raise NotImplementedError
    def __getitem__(self, idx): raise NotImplementedError

def custom_collate(batch):
    item = batch[0]
    return item['token'], item['points'], item['imgs'], item['metainfo'], item['paths'], item['gt_boxes']

def _identity_collate(batch):
    return batch[0]

class NuScenesLoader(BaseLoader):
    """
    Fixed Loader: Bakes intrinsics, Identity Aug Matrix, No Pre-Normalization.
    """
    def __init__(self, dataroot, version, split='val', max_samples=-1, nsweeps=10, expects_bgr=True,
                 pc_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0), crop_policy="center"):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = dataroot
        self.expects_bgr = expects_bgr
        self.nsweeps = int(nsweeps)
        self.pc_range = list(pc_range)
        self.crop_policy = crop_policy

        if version == 'v1.0-mini': split_name = 'mini_val'
        elif split == 'val': split_name = 'val'
        else: split_name = 'train'

        scenes = create_splits_scenes().get(split_name, create_splits_scenes()['val'])
        self.samples = []
        scene_set = set(scenes)
        for scene in self.nusc.scene:
            if scene['name'] in scene_set:
                tok = scene['first_sample_token']
                while tok:
                    s = self.nusc.get('sample', tok)
                    self.samples.append(s)
                    tok = s['next']

        if max_samples != -1: self.samples = self.samples[:max_samples]

    def __len__(self): return len(self.samples)

    def get_sensor_transforms(self, sd_record):
        cs = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        l2e_t = np.array(cs['translation'], dtype=np.float32)
        l2e_r = _quat_rot(Quaternion(cs['rotation']))
        e2g_t = np.array(pose['translation'], dtype=np.float32)
        e2g_r = _quat_rot(Quaternion(pose['rotation']))
        return l2e_t, l2e_r, e2g_t, e2g_r

    def load_points(self, sample):
        lidar_token = sample['data']['LIDAR_TOP']
        ref_sd = self.nusc.get('sample_data', lidar_token)
        ref_l2e_t, ref_l2e_r, ref_e2g_t, ref_e2g_r = self.get_sensor_transforms(ref_sd)
        
        ref_g2e_r, ref_g2e_t = ref_e2g_r.T, -ref_e2g_r.T @ ref_e2g_t
        ref_e2l_r, ref_e2l_t = ref_l2e_r.T, -ref_l2e_r.T @ ref_l2e_t
        
        all_pts, curr_sd = [], ref_sd
        for i in range(self.nsweeps):
            path = osp.join(self.dataroot, curr_sd['filename'])
            pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]
            if i > 0:
                c_l2e_t, c_l2e_r, c_e2g_t, c_e2g_r = self.get_sensor_transforms(curr_sd)
                pts[:,:3] = (c_l2e_r @ pts[:,:3].T).T + c_l2e_t
                pts[:,:3] = (c_e2g_r @ pts[:,:3].T).T + c_e2g_t
                pts[:,:3] = (ref_g2e_r @ pts[:,:3].T).T + ref_g2e_t
                pts[:,:3] = (ref_e2l_r @ pts[:,:3].T).T + ref_e2l_t
            
            dt = (ref_sd['timestamp'] - curr_sd['timestamp']) / 1e6
            all_pts.append(np.hstack([pts, np.full((len(pts),1), dt, dtype=np.float32)]))
            if curr_sd['prev'] == '': break
            curr_sd = self.nusc.get('sample_data', curr_sd['prev'])

        pts = np.concatenate(all_pts, axis=0)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        xm, ym, zm, xM, yM, zM = self.pc_range
        mask = (x>=xm)&(x<=xM)&(y>=ym)&(y<=yM)&(z>=zm)&(z<=zM)
        return pts[mask]

    def load_imgs(self, sample):
        # [CRITICAL FIX] Logic matches original simple_infer_allv2b.py
        target_size = (256, 704)
        cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        tensors, paths, lidar2img, cam2img, lidar2cam = [], [], [], [], []
        
        lidar_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        l2e_t, l2e_r, e2g_t_L, e2g_r_L = self.get_sensor_transforms(lidar_sd)

        for c_name in cams:
            c_sd = self.nusc.get('sample_data', sample['data'][c_name])
            path = osp.join(self.dataroot, c_sd['filename']); paths.append(path)
            
            img = cv2.imread(path)
            if not self.expects_bgr: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            H, W = img.shape[:2]
            scale = 0.48
            nH, nW = int(H*scale), int(W*scale)
            img = cv2.resize(img, (nW, nH))
            
            sx, sy = (nW - target_size[1])//2, (nH - target_size[0])//2
            img = img[sy:sy+target_size[0], sx:sx+target_size[1]]
            
            # [CRITICAL] Do NOT Normalize here. Model preprocessor will do it.
            # But we must pass float tensor.
            tensors.append(torch.from_numpy(img).permute(2,0,1).float())
            
            c_cal = self.nusc.get('calibrated_sensor', c_sd['calibrated_sensor_token'])
            K = np.eye(4, dtype=np.float32); K[:3,:3] = c_cal['camera_intrinsic']
            
            # [CRITICAL] Bake Augmentations into K
            K[0]*=scale; K[1]*=scale; K[0,2]-=sx; K[1,2]-=sy
            cam2img.append(K)
            
            c2e_t, c2e_r, e2g_t_C, e2g_r_C = self.get_sensor_transforms(c_sd)
            T = np.linalg.inv(_Rt(c2e_r, c2e_t)) @ np.linalg.inv(_Rt(e2g_r_C, e2g_t_C)) @ _Rt(e2g_r_L, e2g_t_L) @ _Rt(l2e_r, l2e_t)
            lidar2cam.append(T)
            lidar2img.append(K @ T)

        # [CRITICAL] Identity Aug Matrix to prevent double-transform
        aug_matrix = np.tile(np.eye(4, dtype=np.float32)[None], (6, 1, 1))

        metainfo = {
            'lidar2img': np.stack(lidar2img), 
            'cam2img': np.stack(cam2img), 
            'lidar2cam': np.stack(lidar2cam), 
            'cam2lidar': np.linalg.inv(np.stack(lidar2cam)), 
            'img_aug_matrix': aug_matrix, 
            'img_shape': [target_size]*6, 
            'box_type_3d': LiDARInstance3DBoxes
        }
        
        # Return 4D tensor (N, C, H, W)
        return torch.stack(tensors), metainfo, paths

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pts = self.load_points(sample)
        imgs, meta, paths = self.load_imgs(sample)
        meta['token'] = sample['token']
        return {'token': sample['token'], 'points': pts, 'imgs': imgs, 'metainfo': meta, 'paths': paths, 'gt_boxes': None}

def cfg_iter(loader):
    """Adapter to yield uniform data from MMEngine DataLoader."""
    for sample in loader:
        if isinstance(sample, dict) and 'data_samples' in sample: 
            ds = sample['data_samples'][0] if isinstance(sample['data_samples'], list) else sample['data_samples']
            inp = sample['inputs']
        else: 
             ds = sample.data_samples; inp = sample.inputs
             
        meta = ds.metainfo
        token = str(meta.get('token', meta.get('sample_idx', '')))
        
        pts = _to_numpy_points(inp['points'][0] if isinstance(inp['points'], list) else inp['points'])
        imgs = inp['img'] 
        if isinstance(imgs, list): imgs = imgs[0] 
        # Ensure 4D
        if torch.is_tensor(imgs) and imgs.dim() == 5:
            imgs = imgs.squeeze(0)
        
        paths = meta.get('img_path', meta.get('img_paths', []))
        if isinstance(paths, str): paths = [paths]
        
        yield token, pts, imgs, meta, paths, None

def build_loader_pack(args, cfg):
    if args.data_source == "cfg":
        patch_cfg_paths(cfg, args.dataroot, args.ann_file)
        dataset = DATASETS.build(cfg.test_dataloader.dataset)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=_identity_collate, pin_memory=True)
        nusc = NuScenes(version=args.nus_version, dataroot=args.dataroot, verbose=False)
        return dict(loader=loader, iter_fn=cfg_iter, nusc=nusc)
    else:
        dataset = NuScenesLoader(args.dataroot, args.nus_version, max_samples=args.max_samples, 
                                 crop_policy=args.crop_policy, nsweeps=10) 
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=custom_collate)
        def custom_iter(dl):
            for t, p, i, m, pa, g in dl: yield t, p, i, m, pa, g
        return dict(loader=loader, iter_fn=custom_iter, nusc=dataset.nusc)

def patch_cfg_paths(cfg, dataroot, ann_file):
    def _patch(node):
        if isinstance(node, dict):
            if 'data_root' in node: node['data_root'] = dataroot
            if 'ann_file' in node and ann_file: node['ann_file'] = ann_file
            elif 'ann_file' in node: node['ann_file'] = osp.join(dataroot, node['ann_file']) if not osp.isabs(node['ann_file']) else node['ann_file']
            for k, v in node.items(): _patch(v)
        elif isinstance(node, list):
            for v in node: _patch(v)
    _patch(cfg.test_dataloader)

# ==============================================================================
# 4. BENCHMARK & INFERENCE LOGIC
# ==============================================================================

NUSCENES_ATTRIBUTES = ['cycle.with_rider', 'cycle.without_rider', 'pedestrian.moving', 'pedestrian.standing', 'pedestrian.sitting_lying_down', 'vehicle.moving', 'vehicle.parked', 'vehicle.stopped', 'None']

def _canon_nus_name(name):
    name = name.lower()
    mapping = {'ped': 'pedestrian', 'person': 'pedestrian', 'bike': 'bicycle', 'bus': 'bus', 
               'car': 'car', 'construction': 'construction_vehicle', 'trailer': 'trailer', 'truck': 'truck',
               'cone': 'traffic_cone', 'barrier': 'barrier', 'motor': 'motorcycle'}
    for k, v in mapping.items():
        if k in name: return v
    return 'car'

def get_default_attribute(label_name, velocity):
    v = np.linalg.norm(velocity[:2])
    if 'vehicle' in label_name or 'car' in label_name: return 'vehicle.moving' if v > 0.2 else 'vehicle.parked'
    if 'pedestrian' in label_name: return 'pedestrian.moving' if v > 0.2 else 'pedestrian.standing'
    if 'cycle' in label_name: return 'cycle.with_rider'
    return ''

def lidar_to_global_box(nusc, token, boxes, scores, labels, class_names, attrs=None, vels=None):
    box_list = []
    sd_rec = nusc.get('sample_data', nusc.get('sample', token)['data']['LIDAR_TOP'])
    
    cs = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    
    l2e_r = Quaternion(cs['rotation']); l2e_t = np.array(cs['translation'])
    e2g_r = Quaternion(pose['rotation']); e2g_t = np.array(pose['translation'])

    for i in range(len(boxes)):
        x, y, z, dx, dy, dz, yaw = boxes[i][:7]
        quat = Quaternion(axis=[0, 0, 1], radians=yaw)
        vx, vy = vels[i][:2] if vels is not None else (0.0, 0.0)
        v_lidar = np.array([vx, vy, 0.0])
        v_global = e2g_r.rotate(l2e_r.rotate(v_lidar))
        
        box = Box([x, y, z + dz/2], [dy, dx, dz], quat, label=int(labels[i]), score=float(scores[i]))
        box.rotate(l2e_r); box.translate(l2e_t); box.rotate(e2g_r); box.translate(e2g_t)

        cname = _canon_nus_name(class_names[box.label] if class_names else "car")
        aname = NUSCENES_ATTRIBUTES[int(attrs[i])] if attrs is not None and int(attrs[i]) < len(NUSCENES_ATTRIBUTES) else ""
        if not aname: aname = get_default_attribute(cname, v_global)

        box_list.append({
            "sample_token": token,
            "translation": box.center.tolist(), "size": box.wlh.tolist(),
            "rotation": box.orientation.elements.tolist(), "velocity": v_global[:2].tolist(),
            "detection_name": cname, "detection_score": box.score, "attribute_name": aname
        })
    return box_list

def run_manual_benchmark(model, pack, args, sys_info, class_names):
    print("\n" + "="*60 + "\n STARTING MANUAL BENCHMARK\n" + "="*60)
    res_path = osp.join(args.out_dir, "nuscenes_results.json")
    results_dict = {"meta": {"use_camera": True, "use_lidar": True, "use_radar": False, "use_map": False, "use_external": False}, "results": {}}
    metrics = []
    processed_tokens = []
    
    pbar = tqdm(pack['iter_fn'](pack['loader']), desc="Inference", total=len(pack['loader']))
    for token, pts, imgs, meta, _, _ in pbar:
        processed_tokens.append(token)
        inputs = {'points': [torch.from_numpy(pts).cuda()]}
        
        # [CRITICAL FIX] Pass LIST of 4D tensors to satisfy Preprocessor assertion
        if imgs is not None: 
            inputs['img'] = [imgs.cuda()]
            
        ds = Det3DDataSample(); ds.set_metainfo(meta)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            res = model.test_step(dict(inputs=inputs, data_samples=[ds]))
            
        torch.cuda.synchronize(); dt = (time.perf_counter()-t0)*1000.0
        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
        
        pred = res[0].pred_instances_3d; mask = pred.scores_3d > 0.05
        box = pred.bboxes_3d.tensor[mask].cpu().numpy()
        sc = pred.scores_3d[mask].cpu().numpy()
        lbl = pred.labels_3d[mask].cpu().numpy()
        vels = pred.velocities_3d[mask].cpu().numpy() if hasattr(pred, 'velocities_3d') else None
        if vels is None and box.shape[1] > 7: vels = box[:, 7:9]
        attrs = pred.attr_labels[mask].cpu().numpy() if hasattr(pred, 'attr_labels') else None

        results_dict["results"][token] = lidar_to_global_box(pack['nusc'], token, box, sc, lbl, class_names, attrs, vels)
        metrics.append({'lat': dt, 'mem': max_mem})
        pbar.set_postfix(lat=f"{dt:.1f}ms")

    with open(res_path, 'w') as f: json.dump(results_dict, f)
    print(f"Results saved to {res_path}")

    print("Running NuScenes Evaluator...")
    cfg = config_factory("detection_cvpr_2019")
    nusc_eval = NuScenesEval(pack['nusc'], config=cfg, result_path=res_path, eval_set="val", output_dir=osp.join(args.out_dir, "eval"), verbose=True)
    
    if args.max_samples != -1:
        from nuscenes.eval.common.loaders import load_prediction, load_gt
        nusc_eval.gt_boxes = load_gt(nusc_eval.nusc, nusc_eval.eval_set, DetectionBox, verbose=True)
        nusc_eval.pred_boxes, _ = load_prediction(res_path, nusc_eval.cfg.max_boxes_per_sample, DetectionBox, verbose=True)
        nusc_eval.gt_boxes.boxes = {k: v for k, v in nusc_eval.gt_boxes.boxes.items() if k in processed_tokens}
        nusc_eval.sample_tokens = processed_tokens

    metrics_summary, _ = nusc_eval.evaluate()
    print(f"\nNDS: {metrics_summary.nd_score:.4f} | mAP: {metrics_summary.mean_ap:.4f}")

    perf = {"latency_mean": np.mean([m['lat'] for m in metrics]), "mem_peak": np.max([m['mem'] for m in metrics])}
    with open(osp.join(args.out_dir, "benchmark_perf.json"), 'w') as f: json.dump(perf, f, indent=4)

def run_runner_benchmark(args):
    print("\n" + "="*60 + "\n STARTING RUNNER BENCHMARK\n" + "="*60)
    cfg = Config.fromfile(args.config)
    patch_cfg_paths(cfg, args.dataroot, args.ann_file)
    cfg.work_dir = args.out_dir
    cfg.load_from = args.checkpoint
    runner = Runner.from_cfg(cfg)
    runner.test()

def inference_loop(model, pack, args, metrics):
    print("Starting Visual Inference Loop...")
    pbar = tqdm(pack['iter_fn'](pack['loader']), desc="Visualizing", total=len(pack['loader']))
    
    for token, pts, imgs, meta, paths, _ in pbar:
        pts_t = torch.from_numpy(pts).float().to(args.device)
        inputs = dict(points=[pts_t])
        
        # [CRITICAL FIX] Pass LIST of 4D tensors to satisfy Preprocessor assertion
        if imgs is not None: 
            inputs['img'] = [imgs.to(args.device)]
            
        ds = Det3DDataSample(); ds.set_metainfo(meta)
        
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(args.device)
        t0 = time.perf_counter()
        
        with torch.no_grad():
            res = model.test_step(dict(inputs=inputs, data_samples=[ds]))
            
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000.0
        max_mem = torch.cuda.max_memory_allocated(args.device) / (1024**2)
        
        pred = res[0].pred_instances_3d
        sc = pred.scores_3d.cpu().numpy()
        box = pred.bboxes_3d.tensor.cpu().numpy()
        keep = sc > 0.25
        pred_boxes = box[keep, :7] if keep.any() else None
        max_conf = sc.max() if len(sc) > 0 else 0.0
        
        gt_boxes = get_gt_boxes(pack['nusc'], token) if pack['nusc'] else None
        
        metrics["samples"].append({'id': str(token), 'latency_ms': dt, 'peak_memory_mb': max_mem, 'max_conf': float(max_conf)})
        pbar.set_postfix({"Lat": f"{dt:.1f}ms"})
        
        if paths:
            draw_2d_multiview(
                paths, pts, meta['lidar2img'], pred_boxes, gt_boxes, str(token), args.out_dir
            )
            
        if HAS_OPEN3D:
            headless = os.environ.get('DISPLAY') is None
            if headless: 
                save_ply_files(args.out_dir, str(token), pts, pred_boxes, gt_boxes)
            else: 
                run_open3d_viz(pts, pred_boxes, gt_boxes, window_name=f"Sample {token}")

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    ap.add_argument("--dataroot", required=True, help="NuScenes dataroot")
    ap.add_argument("--out-dir", default="results")
    
    ap.add_argument("--benchmark-type", choices=['runner', 'manual'], default='manual', help="Type of benchmark")
    ap.add_argument("--data-source", choices=['cfg', 'custom'], default='cfg', help="Data loader source")
    ap.add_argument("--device", default="cuda")
    
    ap.add_argument("--nus-version", default="v1.0-trainval")
    ap.add_argument("--ann-file", default="", help="Path to nuscenes info pkl")
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval", action="store_true", help="Run NuScenes Eval (Benchmark Mode)")
    ap.add_argument("--crop-policy", default="center")
    args = ap.parse_args()

    setup_env()
    sys_info = get_system_info()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.benchmark_type == 'runner' and args.eval:
        run_runner_benchmark(args)
        return

    print(f"Loading Config & Model...")
    cfg = Config.fromfile(args.config)
    patch_cfg_paths(cfg, args.dataroot, args.ann_file)
    
    model = MODELS.build(cfg.model)
    if hasattr(cfg, 'test_cfg'): model.test_cfg = cfg.test_cfg
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device).eval()

    pack = build_loader_pack(args, cfg)
    
    if args.eval:
        args.max_samples = -1 
        run_manual_benchmark(model, pack, args, sys_info, cfg.class_names)
    else:
        metrics = {"system_info": sys_info, "samples": []}
        inference_loop(model, pack, args, metrics)
        with open(osp.join(args.out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Done. Results in {args.out_dir}")

if __name__ == "__main__":
    main()
"""
python simple_infer_allv2c.py \
    --config /data/rnd-liu/MyRepo/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    --checkpoint /data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth \
    --dataroot /data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes \
    --out-dir ./bevfusion_infer_results_v2c2 \
    --benchmark-type manual \
    --data-source custom \

"""