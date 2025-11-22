import argparse
import os
import os.path as osp
import math
import time
import numpy as np
import torch
import cv2

# Minimum MM imports required to run the model
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.utils import register_all_modules 

# NuScenes imports
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# ==============================================================================
# 1. ENVIRONMENT SETUP
# ==============================================================================
def setup_env():
    # Essential: Registers model layers so MODELS.build works
    register_all_modules(init_default_scope=True)

# ==============================================================================
# 2. MATH & TRANSFORM UTILS
# ==============================================================================
def _quat_rot(q): return q.rotation_matrix.astype(np.float32)
def _Rt(R, t): 
    T = np.eye(4, dtype=np.float32); T[:3,:3]=R; T[:3,3]=t 
    return T

def get_sensor_transforms(nusc, sd_record):
    cs = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd_record['ego_pose_token'])
    l2e_t = np.array(cs['translation'], dtype=np.float32)
    l2e_r = _quat_rot(Quaternion(cs['rotation']))
    e2g_t = np.array(pose['translation'], dtype=np.float32)
    e2g_r = _quat_rot(Quaternion(pose['rotation']))
    return l2e_t, l2e_r, e2g_t, e2g_r

# ==============================================================================
# 3. DATA LOADING (PURE PYTHON - NO REGISTRY)
# ==============================================================================

def load_points_multisweep(nusc, sample_record, lidar_name='LIDAR_TOP', nsweeps=5):
    """Load multi-sweep point cloud manually."""
    lidar_token = sample_record['data'][lidar_name]
    ref_sd = nusc.get('sample_data', lidar_token)
    
    ref_l2e_t, ref_l2e_r, ref_e2g_t, ref_e2g_r = get_sensor_transforms(nusc, ref_sd)
    
    # Inverse transforms for Reference Frame
    ref_g2e_r = ref_e2g_r.T
    ref_g2e_t = -ref_g2e_r @ ref_e2g_t
    ref_e2l_r = ref_l2e_r.T
    ref_e2l_t = -ref_e2l_r @ ref_l2e_t

    all_points = []
    curr_sd = ref_sd
    
    for i in range(nsweeps):
        lidar_path = osp.join(nusc.dataroot, curr_sd['filename'])
        pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        pts = pts[:, :4] # x,y,z,intensity
        
        if i > 0:
            # Transform past frame to current reference frame
            curr_l2e_t, curr_l2e_r, curr_e2g_t, curr_e2g_r = get_sensor_transforms(nusc, curr_sd)
            pts[:, :3] = (curr_l2e_r @ pts[:, :3].T).T + curr_l2e_t
            pts[:, :3] = (curr_e2g_r @ pts[:, :3].T).T + curr_e2g_t
            pts[:, :3] = (ref_g2e_r @ pts[:, :3].T).T + ref_g2e_t
            pts[:, :3] = (ref_e2l_r @ pts[:, :3].T).T + ref_e2l_t

        time_lag = (ref_sd['timestamp'] - curr_sd['timestamp']) / 1e6 
        pts = np.hstack([pts, np.full((pts.shape[0], 1), time_lag, dtype=np.float32)])
        all_points.append(pts)
        
        if curr_sd['prev'] == '': break
        curr_sd = nusc.get('sample_data', curr_sd['prev'])

    return np.concatenate(all_points, axis=0)

def load_imgs_manual(nusc, sample_record, dataroot):
    """Load, Resize, Crop, Normalize images manually."""
    cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # Model Config Settings (BEVFusion Standard)
    RESIZE_SCALE = 0.48
    FINAL_H, FINAL_W = 256, 704
    
    # ImageNet Normalization
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)
    
    img_tensors = []
    k_list, l2c_list, l2i_list, paths = [], [], [], []
    
    lidar_token = sample_record['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_token)
    l2e_t, l2e_r, e2g_t_L, e2g_r_L = get_sensor_transforms(nusc, lidar_sd)

    for c_name in cams:
        c_token = sample_record['data'][c_name]
        c_sd = nusc.get('sample_data', c_token)
        img_path = osp.join(dataroot, c_sd['filename'])
        paths.append(img_path)
        
        # 1. Load
        img = cv2.imread(img_path) # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        
        # 2. Resize
        new_W, new_H = int(W * RESIZE_SCALE), int(H * RESIZE_SCALE)
        img_resized = cv2.resize(img, (new_W, new_H))
        
        # 3. Crop (Center)
        start_x = (new_W - FINAL_W) // 2
        start_y = (new_H - FINAL_H) // 2 
        img_crop = img_resized[start_y : start_y+FINAL_H, start_x : start_x+FINAL_W]
        
        # 4. Normalize & Tensor
        img_norm = (img_crop.astype(np.float32) - mean) / std
        img_tensors.append(torch.from_numpy(img_norm.transpose(2, 0, 1)))
        
        # 5. Update Intrinsics for Resize/Crop
        c_cal = nusc.get('calibrated_sensor', c_sd['calibrated_sensor_token'])
        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = np.array(c_cal['camera_intrinsic'], dtype=np.float32)
        K[0] *= RESIZE_SCALE; K[1] *= RESIZE_SCALE # Scale
        K[0, 2] -= start_x; K[1, 2] -= start_y     # Translate
        k_list.append(K)
        
        # 6. Extrinsics
        c2e_t, c2e_r, e2g_t_C, e2g_r_C = get_sensor_transforms(nusc, c_sd)
        T_l2e = _Rt(l2e_r, l2e_t)
        T_eL2g = _Rt(e2g_r_L, e2g_t_L)
        T_g2eC = np.linalg.inv(_Rt(e2g_r_C, e2g_t_C))
        T_eC2c = np.linalg.inv(_Rt(c2e_r, c2e_t))
        l2c = T_eC2c @ T_g2eC @ T_eL2g @ T_l2e
        
        l2c_list.append(l2c)
        l2i_list.append(K @ l2c)

    # Stack [6, 3, H, W]
    imgs_stack = torch.stack(img_tensors)
    
    metainfo = {
        'cam2img': np.stack(k_list),
        'lidar2cam': np.stack(l2c_list),
        'lidar2img': np.stack(l2i_list),
        'cam2lidar': np.linalg.inv(np.stack(l2c_list)),
        'img_shape': [(FINAL_H, FINAL_W)] * 6,
        'ori_shape': [(H, W)] * 6,
        'pad_shape': [(FINAL_H, FINAL_W)] * 6,
        'scale_factor': 1.0,
        'box_type_3d': LiDARInstance3DBoxes,
        'img_aug_matrix': np.tile(np.eye(4, dtype=np.float32)[None], (6, 1, 1)),
        'lidar_aug_matrix': np.eye(4, dtype=np.float32)
    }
    return imgs_stack, metainfo, paths

# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================

def get_gt_boxes(nusc, sample_token):
    try:
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
    except: return None

def project_points(pts, P):
    N = pts.shape[0]
    h = np.hstack([pts[:,:3], np.ones((N,1), dtype=np.float32)])
    c = (P @ h.T).T
    z = c[:,2]
    mask = z > 0.1
    return c[mask, :2]/c[mask, 2:3], z[mask], mask

def draw_viz(img_path, pts, P, gt, pred, label=""):
    # Read & Match Preprocessing Crop/Resize for Viz
    img = cv2.imread(img_path)
    h_raw, w_raw = img.shape[:2]
    RESIZE_SCALE = 0.48
    FINAL_H, FINAL_W = 256, 704
    
    new_W, new_H = int(w_raw * RESIZE_SCALE), int(h_raw * RESIZE_SCALE)
    img = cv2.resize(img, (new_W, new_H))
    start_x = (new_W - FINAL_W) // 2
    start_y = (new_H - FINAL_H) // 2
    img = img[start_y : start_y+FINAL_H, start_x : start_x+FINAL_W].copy()
    h, w = img.shape[:2]

    # Draw Points
    uv, z, _ = project_points(pts, P)
    if len(uv) > 0:
        d = np.clip(z, 0, 60)/60 * 255
        cols = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_JET).reshape(-1,3)
        for i, (x,y) in enumerate(uv.astype(int)):
            if 0<=x<w and 0<=y<h: img[y,x] = cols[i]

    # Draw Boxes
    def draw_b(boxes, c):
        if boxes is None: return
        for b in boxes:
            if len(b)<7: continue
            x,y,zb,dx,dy,dz,yaw = b[:7]
            cos,sin = math.cos(yaw), math.sin(yaw)
            R = np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])
            xc = np.array([dx,dx,-dx,-dx,dx,dx,-dx,-dx])/2
            yc = np.array([dy,-dy,-dy,dy,dy,-dy,-dy,dy])/2
            zc = np.array([0,0,0,0,dz,dz,dz,dz])
            corn = (R @ np.stack([xc,yc,zc])).T + np.array([x,y,zb])
            uv_b, z_b, _ = project_points(corn, P)
            if len(uv_b) < 8: continue
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for u,v in edges:
                cv2.line(img, tuple(uv_b[u].astype(int)), tuple(uv_b[v].astype(int)), c, 2, cv2.LINE_AA)

    draw_b(gt, (0,0,255)) # Red
    draw_b(pred, (0,255,0)) # Green
    
    if label:
        cv2.rectangle(img, (0,0), (400,30), (0,0,0), -1)
        cv2.putText(img, label, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img

# ==============================================================================
# 5. MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    #ap.add_argument("--config", type=str, default="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py", help="Path to MMDet3D config .py")
    #ap.add_argument("--checkpoint", type=str, default="modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth", help="Path to model checkpoint .pth")
    ap.add_argument("--config", type=str, default="work_dirs/mybevfusion7_newv2/mybevfusion7_crossattnaux_paintingv2.py", help="Path to MMDet3D config .py")
    ap.add_argument("--checkpoint", type=str, default="work_dirs/mybevfusion7_newv2/epoch_2.pth", help="Path to model checkpoint .pth")
    
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--nus-version", default="v1.0-trainval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="infer_out2")
    ap.add_argument("--max-samples", type=int, default=5)
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    setup_env()
    
    print("Building Model...")
    cfg = Config.fromfile(args.config)
    # MODELS.build is the standard way, avoiding 'ImportError: build_detector'
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()
    
    print("Init NuScenes...")
    nusc = NuScenes(version=args.nus_version, dataroot=args.dataroot, verbose=False)
    
    print("Starting Loop...")
    for i, sample in enumerate(nusc.sample):
        if i >= args.max_samples: break
        token = sample['token']
        print(f"[{i}] {token[:6]}")
        
        # 1. Manual Load
        imgs_t, metainfo, img_paths = load_imgs_manual(nusc, sample, args.dataroot)
        imgs_t = imgs_t.to(args.device)
        
        pts_np = load_points_multisweep(nusc, sample, nsweeps=5)
        mask = (pts_np[:,0]>-54) & (pts_np[:,0]<54) & (pts_np[:,1]>-54) & (pts_np[:,1]<54) & (pts_np[:,2]>-5) & (pts_np[:,2]<3)
        pts_np = pts_np[mask]
        pts_t = torch.from_numpy(pts_np).float().to(args.device)
        
        # 2. Construct Inputs
        # KEY FIX: Input key must be 'img', NOT 'imgs'
        # KEY FIX: Wrap single sample in list [tensor] to denote batch size 1
        inputs = dict(img=[imgs_t], points=[pts_t])
        
        ds = Det3DDataSample()
        metainfo['sample_idx'] = token
        metainfo['token'] = token
        ds.set_metainfo(metainfo)
        
        # 3. Inference
        with torch.no_grad():
            t0 = time.time()
            res = model.test_step(dict(inputs=inputs, data_samples=[ds]))
            dt = (time.time()-t0)*1000
            
        pred = res[0].pred_instances_3d
        scores = pred.scores_3d.cpu().numpy()
        boxes = pred.bboxes_3d.tensor.cpu().numpy()
        
        # Filter
        keep = scores > 0.25
        pred_boxes = boxes[keep, :7] if keep.any() else None
        print(f"    Time: {dt:.1f}ms | Max Score: {scores.max() if len(scores)>0 else 0:.3f}")
        
        # 4. Visualize
        gt_boxes = get_gt_boxes(nusc, token)
        P = metainfo['lidar2img'][0] # Front Cam
        
        # Note: draw_viz applies same crop/resize logic to raw image to match P
        viz = draw_viz(img_paths[0], pts_np, P, gt_boxes, pred_boxes, label=f"P(G) GT(R) {token[:4]}")
        cv2.imwrite(osp.join(args.out_dir, f"{token}_final.jpg"), viz)

if __name__ == "__main__":
    main()