import argparse
import os
import os.path as osp
import math
import time
import json
import numpy as np
import torch
import cv2
from pathlib import Path

# MM Libraries
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.utils import register_all_modules 

# NuScenes & Geometry
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# Open3D (Optional)
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    print("Open3D not found. 3D visualization will be skipped.")
    HAS_OPEN3D = False

# ==============================================================================
# 1. ENVIRONMENT & UTILS
# ==============================================================================

def setup_env():
    register_all_modules(init_default_scope=True)

def _quat_rot(q): return q.rotation_matrix.astype(np.float32)
def _Rt(R, t): 
    T = np.eye(4, dtype=np.float32); T[:3,:3]=R; T[:3,3]=t 
    return T

def project_points(pts, P):
    """Project 3D points to 2D [N, 2] and Depth [N]."""
    N = pts.shape[0]
    h = np.hstack([pts[:,:3], np.ones((N,1), dtype=np.float32)])
    c = (P @ h.T).T
    z = c[:,2]
    mask = z > 0.1
    return c[mask, :2]/c[mask, 2:3], z[mask], mask

# ==============================================================================
# 2. DATASET LOADERS
# ==============================================================================

class BaseLoader:
    def __iter__(self): raise NotImplementedError

class NuScenesLoader(BaseLoader):
    def __init__(self, dataroot, version, max_samples=5):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.max_samples = max_samples
        self.dataroot = dataroot
        
    def get_sensor_transforms(self, sd_record):
        cs = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        l2e_t = np.array(cs['translation'], dtype=np.float32)
        l2e_r = _quat_rot(Quaternion(cs['rotation']))
        e2g_t = np.array(pose['translation'], dtype=np.float32)
        e2g_r = _quat_rot(Quaternion(pose['rotation']))
        return l2e_t, l2e_r, e2g_t, e2g_r

    def load_points(self, sample, nsweeps=5):
        lidar_token = sample['data']['LIDAR_TOP']
        ref_sd = self.nusc.get('sample_data', lidar_token)
        ref_l2e_t, ref_l2e_r, ref_e2g_t, ref_e2g_r = self.get_sensor_transforms(ref_sd)
        
        ref_g2e_r, ref_g2e_t = ref_e2g_r.T, -ref_e2g_r.T @ ref_e2g_t
        ref_e2l_r, ref_e2l_t = ref_l2e_r.T, -ref_l2e_r.T @ ref_l2e_t
        
        all_pts, curr_sd = [], ref_sd
        for i in range(nsweeps):
            path = osp.join(self.dataroot, curr_sd['filename'])
            pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]
            if i > 0:
                c_l2e_t, c_l2e_r, c_e2g_t, c_e2g_r = self.get_sensor_transforms(curr_sd)
                # Chain: L_i -> E_i -> G -> E_ref -> L_ref
                pts[:,:3] = (c_l2e_r @ pts[:,:3].T).T + c_l2e_t
                pts[:,:3] = (c_e2g_r @ pts[:,:3].T).T + c_e2g_t
                pts[:,:3] = (ref_g2e_r @ pts[:,:3].T).T + ref_g2e_t
                pts[:,:3] = (ref_e2l_r @ pts[:,:3].T).T + ref_e2l_t
            
            dt = (ref_sd['timestamp'] - curr_sd['timestamp']) / 1e6
            all_pts.append(np.hstack([pts, np.full((len(pts),1), dt, dtype=np.float32)]))
            if curr_sd['prev'] == '': break
            curr_sd = self.nusc.get('sample_data', curr_sd['prev'])
        return np.concatenate(all_pts)

    def load_imgs(self, sample, target_size=(256, 704)):
        cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)
        
        tensors, paths = [], []
        lidar2img, cam2img, lidar2cam = [], [], []
        
        lidar_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        l2e_t, l2e_r, e2g_t_L, e2g_r_L = self.get_sensor_transforms(lidar_sd)

        for c_name in cams:
            c_sd = self.nusc.get('sample_data', sample['data'][c_name])
            path = osp.join(self.dataroot, c_sd['filename'])
            paths.append(path)
            
            # Process Image
            img = cv2.imread(path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            scale = 0.48 # BEVFusion Standard
            nH, nW = int(H*scale), int(W*scale)
            img = cv2.resize(img, (nW, nH))
            sx, sy = (nW - target_size[1])//2, (nH - target_size[0])//2
            img = img[sy:sy+target_size[0], sx:sx+target_size[1]]
            tensors.append(torch.from_numpy(((img - mean)/std).transpose(2,0,1).astype(np.float32)))
            
            # Matrix K (Intrinsics)
            c_cal = self.nusc.get('calibrated_sensor', c_sd['calibrated_sensor_token'])
            K = np.eye(4, dtype=np.float32); K[:3,:3] = c_cal['camera_intrinsic']
            K[0]*=scale; K[1]*=scale; K[0,2]-=sx; K[1,2]-=sy
            cam2img.append(K)
            
            # Matrix T (Lidar2Cam)
            c2e_t, c2e_r, e2g_t_C, e2g_r_C = self.get_sensor_transforms(c_sd)
            T = np.linalg.inv(_Rt(c2e_r, c2e_t)) @ np.linalg.inv(_Rt(e2g_r_C, e2g_t_C)) @ _Rt(e2g_r_L, e2g_t_L) @ _Rt(l2e_r, l2e_t)
            lidar2cam.append(T)
            
            # Full Projection
            lidar2img.append(K @ T)

        return (torch.stack(tensors), 
                np.stack(lidar2img), 
                np.stack(cam2img), 
                np.stack(lidar2cam), 
                paths)
    
    def get_gt(self, sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd = self.nusc.get('sample_data', lidar_token)
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        pose = self.nusc.get('ego_pose', sd['ego_pose_token'])
        
        boxes = []
        for ann in sample['anns']:
            a = self.nusc.get('sample_annotation', ann)
            box = Box(a['translation'], a['size'], Quaternion(a['rotation']))
            box.translate(-np.array(pose['translation']))
            box.rotate(Quaternion(pose['rotation']).inverse)
            box.translate(-np.array(cs['translation']))
            box.rotate(Quaternion(cs['rotation']).inverse)
            z_bot = box.center[2] - box.wlh[2]/2
            boxes.append([box.center[0], box.center[1], z_bot, box.wlh[1], box.wlh[0], box.wlh[2], box.orientation.yaw_pitch_roll[0]])
        return np.array(boxes, dtype=np.float32) if boxes else None

    def __iter__(self):
        for i, sample in enumerate(self.nusc.sample):
            if i >= self.max_samples: break
            pts = self.load_points(sample)
            imgs, l2i, c2i, l2c, paths = self.load_imgs(sample)
            
            metainfo = {
                'lidar2img': l2i,
                'cam2img': c2i,
                'lidar2cam': l2c,
                'cam2lidar': np.linalg.inv(l2c),
                'img_aug_matrix': np.tile(np.eye(4, dtype=np.float32)[None], (6, 1, 1)),
                'lidar_aug_matrix': np.eye(4, dtype=np.float32),
                'img_shape': [(256, 704)]*6, 
                'box_type_3d': LiDARInstance3DBoxes
            }
            yield sample['token'], pts, imgs, metainfo, paths, self.get_gt(sample)

class KITTILoader(BaseLoader):
    def __init__(self, dataroot, max_samples=5):
        self.dataroot = Path(dataroot)
        self.ids = sorted([x.stem for x in (self.dataroot / "velodyne").glob("*.bin")])[:max_samples]
    
    def parse_calib(self, idx):
        with open(self.dataroot / "calib" / f"{idx}.txt", 'r') as f: lines = f.readlines()
        P2 = np.array([float(x) for x in lines[2].strip().split(' ')[1:]]).reshape(3, 4)
        R0 = np.array([float(x) for x in lines[4].strip().split(' ')[1:]]).reshape(3, 3)
        Tr = np.array([float(x) for x in lines[5].strip().split(' ')[1:]]).reshape(3, 4)
        P2_4x4, R0_4x4, Tr_4x4 = np.eye(4), np.eye(4), np.eye(4)
        P2_4x4[:3] = P2; R0_4x4[:3,:3] = R0; Tr_4x4[:3] = Tr
        return (P2_4x4 @ R0_4x4 @ Tr_4x4).astype(np.float32)

    def __iter__(self):
        for idx in self.ids:
            pts = np.fromfile(str(self.dataroot / "velodyne" / f"{idx}.bin"), dtype=np.float32).reshape(-1, 4)
            img_path = self.dataroot / "image_2" / f"{idx}.png"
            imgs, path_list, l2i_stack = [], [], []
            if img_path.exists():
                img = cv2.imread(str(img_path)); img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                mean, std = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])
                imgs.append(torch.from_numpy(((img_rgb - mean)/std).transpose(2,0,1)))
                path_list.append(str(img_path))
                l2i_stack.append(self.parse_calib(idx))
            
            imgs_t = torch.stack(imgs) if imgs else None
            l2i_t = np.stack(l2i_stack) if l2i_stack else np.eye(4)[None]
            
            # Fill missing keys with identity to prevent crash
            N = len(l2i_stack) if l2i_stack else 1
            eye_stack = np.tile(np.eye(4, dtype=np.float32)[None], (N,1,1))
            
            metainfo = {
                'lidar2img': l2i_t,
                'cam2img': eye_stack, # Dummy
                'lidar2cam': eye_stack, # Dummy
                'cam2lidar': eye_stack, # Dummy
                'img_aug_matrix': eye_stack,
                'lidar_aug_matrix': np.eye(4, dtype=np.float32),
                'box_type_3d': LiDARInstance3DBoxes,
                'sample_idx': idx
            }
            
            yield idx, pts, imgs_t, metainfo, path_list, None

class AnyLoader(BaseLoader):
    def __init__(self, dataroot, max_samples=5):
        self.dataroot = Path(dataroot)
        self.files = sorted(list(self.dataroot.glob("*.bin")) + list(self.dataroot.glob("*.pcd")))[:max_samples]
    def __iter__(self):
        for p in self.files:
            pts = np.fromfile(str(p), dtype=np.float32).reshape(-1, 4) if p.suffix == '.bin' else np.zeros((100,4))
            eye_stack = np.tile(np.eye(4, dtype=np.float32)[None], (1,1,1))
            
            metainfo = {
                'lidar2img': eye_stack,
                'cam2img': eye_stack,
                'lidar2cam': eye_stack,
                'cam2lidar': eye_stack,
                'img_aug_matrix': eye_stack,
                'lidar_aug_matrix': np.eye(4, dtype=np.float32),
                'box_type_3d': LiDARInstance3DBoxes
            }
            
            yield p.stem, pts, None, metainfo, [], None

# ==============================================================================
# 3. OPEN3D VISUALIZATION (HEADLESS SUPPORT)
# ==============================================================================

def boxes_to_lineset(boxes, color):
    """Converts [N, 7] boxes to Open3D LineSet."""
    if boxes is None or len(boxes) == 0: return None
    
    points = []
    lines = []
    colors = []
    
    for i, b in enumerate(boxes):
        x, y, z, dx, dy, dz, yaw = b[:7]
        
        # Create Rotation Matrix
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # 8 Corners relative to bottom-center
        x_corners = [dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2]
        y_corners = [dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2]
        z_corners = [0, 0, 0, 0, dz, dz, dz, dz]
        
        corners = np.vstack([x_corners, y_corners, z_corners]) # [3, 8]
        corners = (R @ corners).T + np.array([x, y, z]) # [8, 3]
        
        base_idx = i * 8
        points.extend(corners.tolist())
        
        # 12 Lines
        seg_indices = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        
        for u, v in seg_indices:
            lines.append([base_idx + u, base_idx + v])
            colors.append(color)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def save_ply_files(out_dir, token, pts, pred_boxes, gt_boxes):
    """Saves geometry as PLY files for headless visualization."""
    # 1. Save Points
    pcd = o3d.geometry.PointCloud()
    # Use only XYZ for visualization
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3]) 
    o3d.io.write_point_cloud(osp.join(out_dir, f"{token}_points.ply"), pcd)
    
    # 2. Save Pred Boxes
    ls_pred = boxes_to_lineset(pred_boxes, [0, 1, 0]) # Green
    if ls_pred:
        o3d.io.write_line_set(osp.join(out_dir, f"{token}_pred_boxes.ply"), ls_pred)
        
    # 3. Save GT Boxes
    ls_gt = boxes_to_lineset(gt_boxes, [1, 0, 0]) # Red
    if ls_gt:
        o3d.io.write_line_set(osp.join(out_dir, f"{token}_gt_boxes.ply"), ls_gt)
        
    print(f"    [Headless] Saved PLY files to {out_dir} (View with MeshLab/CloudCompare)")

# ==============================================================================
# 4. 2D VISUALIZATION (Dynamic Grid)
# ==============================================================================

def draw_2d_multiview(paths, pts, l2i_list, pred_boxes, gt_boxes, token, out_dir):
    if not paths: return
    
    canvases = []
    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None: continue
        
        # Resize to 704x256 to match P (Standard BEVFusion)
        h, w = img.shape[:2]
        scale = 0.48
        nH, nW = int(h*scale), int(w*scale)
        img = cv2.resize(img, (nW, nH))
        sx, sy = (nW - 704)//2, (nH - 256)//2
        img = img[sy:sy+256, sx:sx+704].copy()
        
        P = l2i_list[i]
        
        # 1. Points
        uv, z, _ = project_points(pts, P)
        if len(uv) > 0:
            d_norm = np.clip(z, 0, 60)/60 * 255
            cols = cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_JET).reshape(-1,3)
            h, w = img.shape[:2]
            for j, (x,y) in enumerate(uv.astype(int)):
                if 0<=x<w and 0<=y<h: img[y,x] = cols[j]

        # 2. Boxes
        def draw_b(boxes, c):
            if boxes is None: return
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for b in boxes:
                if len(b)<7: continue
                x,y,zb,dx,dy,dz,yaw = b[:7]
                c_cos, s_sin = math.cos(yaw), math.sin(yaw)
                R = np.array([[c_cos,-s_sin,0],[s_sin,c_cos,0],[0,0,1]])
                xc = np.array([dx,dx,-dx,-dx,dx,dx,-dx,-dx])/2
                yc = np.array([dy,-dy,-dy,dy,dy,-dy,-dy,dy])/2
                zc = np.array([0,0,0,0,dz,dz,dz,dz])
                corn = (R @ np.stack([xc,yc,zc])).T + np.array([x,y,zb])
                
                uv_b, z_b, _ = project_points(corn, P)
                if len(uv_b) < 8: continue
                for u,v in edges:
                    cv2.line(img, tuple(uv_b[u].astype(int)), tuple(uv_b[v].astype(int)), c, 2, cv2.LINE_AA)

        draw_b(gt_boxes, (0,0,255))
        draw_b(pred_boxes, (0,255,0))
        canvases.append(img)

    # Dynamic Grid
    N = len(canvases)
    if N == 6: # NuScenes
        final = np.vstack([np.hstack(canvases[:3]), np.hstack(canvases[3:])])
    elif N > 0:
        final = np.hstack(canvases)
    else: return

    # Legend Overlay
    overlay = final[0:60, 0:350].copy()
    cv2.rectangle(overlay, (0,0), (350,60), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.7, final[0:60, 0:350], 0.3, 0, final[0:60, 0:350])
    cv2.putText(final, f"ID: {token[:6]}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    cv2.circle(final, (20, 45), 5, (0,0,255), -1); cv2.putText(final, "GT", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.circle(final, (100, 45), 5, (0,255,0), -1); cv2.putText(final, "Pred", (115, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    
    cv2.imwrite(osp.join(out_dir, f"{token}_multiview.jpg"), final)

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    #ap.add_argument("--config", type=str, default="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py", help="Path to MMDet3D config .py")
    #ap.add_argument("--checkpoint", type=str, default="modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth", help="Path to model checkpoint .pth")
    ap.add_argument("--config", type=str, default="work_dirs/mybevfusion7_newv2/mybevfusion7_crossattnaux_paintingv2.py", help="Path to MMDet3D config .py")
    ap.add_argument("--checkpoint", type=str, default="work_dirs/mybevfusion7_newv2/epoch_2.pth", help="Path to model checkpoint .pth")
    
    ap.add_argument("--dataset", default="nuscenes", choices=["nuscenes", "kitti", "any"])
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--nus-version", default="v1.0-trainval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="infer_results_mybev")
    ap.add_argument("--max-samples", type=int, default=20)
    args = ap.parse_args()
    
    setup_env()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Build Model
    print(f"Building Model from {args.config}...")
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()
    
    # 2. Select Loader
    print(f"Initializing Loader for {args.dataset}...")
    if args.dataset == 'nuscenes':
        loader = NuScenesLoader(args.dataroot, args.nus_version, args.max_samples)
    elif args.dataset == 'kitti':
        loader = KITTILoader(args.dataroot, args.max_samples)
    else:
        loader = AnyLoader(args.dataroot, args.max_samples)

    # Check Display for Open3D 
    headless = os.environ.get('DISPLAY') is None
    if HAS_OPEN3D and headless:
        print("Running in HEADLESS mode. 3D visualization will be saved as .PLY files.")
    elif HAS_OPEN3D:
        print("Running in INTERACTIVE mode. A 3D window will open.")

    metrics = []
    
    print("Starting Inference...")
    for token, pts, imgs, meta, paths, gt_boxes in loader:
        print(f"Processing {token}...")
        
        # Prepare Inputs
        pts_t = torch.from_numpy(pts).float().to(args.device)
        inputs = dict(points=[pts_t])
        if imgs is not None:
            # 'img' key requires list of tensors for batching
            inputs['img'] = [imgs.to(args.device)]
            
        ds = Det3DDataSample()
        ds.set_metainfo(meta)
        
        # Inference & Metrics
        torch.cuda.synchronize()
        start_mem = torch.cuda.max_memory_allocated()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            res = model.test_step(dict(inputs=inputs, data_samples=[ds]))
            
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        max_mem = torch.cuda.max_memory_allocated()
        max_mem_mb = max_mem / (1024**2)
        mem_mb = (max_mem - start_mem) / (1024**2)
        
        # Parse
        pred = res[0].pred_instances_3d
        sc = pred.scores_3d.cpu().numpy()
        box = pred.bboxes_3d.tensor.cpu().numpy()
        
        keep = sc > 0.25
        pred_boxes = box[keep, :7] if keep.any() else None
        max_conf = sc.max() if len(sc) > 0 else 0.0
        
        print(f"   Lat: {dt:.1f}ms | Max_Mem: {max_mem_mb:.1f}MB | Max_conf: {max_conf:.3f}")
        
        metrics.append({'id': token, 'latency': dt, 'max_memory_mb': max_mem_mb, 'max_conf': float(max_conf)})
        
        # 2D Visualization
        if paths:
            draw_2d_multiview(paths, pts, meta['lidar2img'], pred_boxes, gt_boxes, str(token), args.out_dir)
            
        # 3D Visualization (Open3D)
        if HAS_OPEN3D:
            if headless:
                save_ply_files(args.out_dir, str(token), pts, pred_boxes, gt_boxes)
            else:
                # Optional: Interactive View
                # run_open3d_viz(pts, pred_boxes, gt_boxes)
                pass

    # Save Metrics
    with open(osp.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Done. Results in {args.out_dir}")

if __name__ == "__main__":
    main()