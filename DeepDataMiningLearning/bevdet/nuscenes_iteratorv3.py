import os
import os.path as osp
import math
import time
import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Optional, Union, Iterator
from PIL import Image, ImageDraw

# Matplotlib for high-quality depth rendering (non-interactive backend)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NuScenes DevKit Imports
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud, Box
    from pyquaternion import Quaternion
except ImportError:
    print("Warning: nuscenes-devkit not installed. (pip install nuscenes-devkit)")

# ==============================================================================
# SECTION 1: GEOMETRY & CALIBRATION HELPERS (Robust Ego-Motion Handling)
# ==============================================================================

def _quat_rot(q: Quaternion) -> np.ndarray:
    """Convert Quaternion to 3x3 rotation matrix (float32)."""
    return q.rotation_matrix.astype(np.float32)

def _Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build 4x4 homogeneous matrix from R (3x3) and t (3,).
    Returns: [4, 4] float32 matrix.
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t.astype(np.float32)
    return T

def _build_lidar2cam_chain(nusc: 'NuScenes', 
                           sample: dict, 
                           cam_token: str, 
                           lidar_sd: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes precise transforms accounting for ego-motion between LiDAR and Camera timestamps.
    
    Chain: LiDAR -> Ego(t_L) -> Global -> Ego(t_C) -> Camera -> Image
    
    Returns:
        lidar2cam: [4, 4] Transform from LiDAR frame to Camera frame.
        cam2lidar: [4, 4] Inverse of above.
        lidar2img: [4, 4] Full projection (Intrinsics @ lidar2cam).
        K4:        [4, 4] Camera intrinsics (padded with 0,0,0,1).
    """
    # 1. LiDAR -> Ego (at LiDAR timestamp)
    lidar_calib = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_ego = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    
    T_lidar_to_egoL = _Rt(_quat_rot(Quaternion(lidar_calib['rotation'])), np.array(lidar_calib['translation']))
    T_egoL_to_global = _Rt(_quat_rot(Quaternion(lidar_ego['rotation'])), np.array(lidar_ego['translation']))

    # 2. Global -> Ego (at Camera timestamp) -> Camera
    cam_sd = nusc.get('sample_data', cam_token)
    cam_calib = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    cam_ego = nusc.get('ego_pose', cam_sd['ego_pose_token'])

    # Invert to go Global -> EgoC
    T_global_to_egoC = np.linalg.inv(_Rt(_quat_rot(Quaternion(cam_ego['rotation'])), np.array(cam_ego['translation'])))
    # Invert to go EgoC -> Cam
    T_egoC_to_cam = np.linalg.inv(_Rt(_quat_rot(Quaternion(cam_calib['rotation'])), np.array(cam_calib['translation'])))

    # 3. Chain Transformations
    lidar2cam = T_egoC_to_cam @ T_global_to_egoC @ T_egoL_to_global @ T_lidar_to_ego
    cam2lidar = np.linalg.inv(lidar2cam)

    # 4. Intrinsics
    K = np.array(cam_calib['camera_intrinsic'], dtype=np.float32)
    K4 = np.eye(4, dtype=np.float32)
    K4[:3, :3] = K
    
    # 5. Projection Matrix
    lidar2img = K4 @ lidar2cam

    return lidar2cam.astype(np.float32), cam2lidar.astype(np.float32), lidar2img.astype(np.float32), K4.astype(np.float32)

# ==============================================================================
# SECTION 2: VISUALIZATION & RENDERING (Points & Boxes)
# ==============================================================================

def project_points(points_xyz: np.ndarray, P_4x4: np.ndarray):
    """
    Project 3D points to 2D image plane.
    Args:
        points_xyz: [N, 3] float32
        P_4x4:      [4, 4] float32 (lidar2img)
    Returns:
        uv:   [M, 2] Pixel coords
        zc:   [M]    Depth in camera frame
        mask: [N]    Boolean mask of valid points (in front of cam)
    """
    N = points_xyz.shape[0]
    homo = np.concatenate([points_xyz[:,:3], np.ones((N,1), dtype=np.float32)], axis=1) # [N,4]
    cam = (P_4x4 @ homo.T).T # [N,4]
    
    z = cam[:, 2].copy()
    mask = z > 1e-3 # Keep points in front of camera
    
    cam = cam[mask]
    uv = cam[:, :2] / cam[:, 2:3] # Perspective divide
    zc = z[mask]
    return uv, zc, mask

def overlay_points_depth(img: Image.Image, 
                         uv: np.ndarray, 
                         zc: np.ndarray,
                         point_size_px: Optional[int] = None,
                         cmap='turbo', 
                         alpha=0.95, 
                         clip_range=(1.0, 60.0)) -> Image.Image:
    """
    Draws depth-colored LiDAR points using Matplotlib for high-quality rendering.
    """
    w, h = img.size
    if zc.size == 0: return img

    # Normalize depth for color map
    zmin, zmax = clip_range
    norm = np.clip(zc, zmin, zmax)
    norm = (norm - zmin) / (zmax - zmin + 1e-6)

    # Sort: Far -> Near (so near points draw on top)
    idx = np.argsort(norm)[::-1] 
    uv = uv[idx]
    norm = norm[idx]

    if point_size_px is None:
        point_size_px = max(1, int(0.0015 * max(w, h)))

    # Plot
    fig = plt.figure(figsize=(w/100., h/100.), dpi=100)
    ax = plt.axes([0,0,1,1])
    ax.imshow(img)
    ax.scatter(uv[:, 0], uv[:, 1], s=point_size_px, c=norm, cmap=cmap, linewidths=0, alpha=alpha)
    ax.set_axis_off()
    
    # Render to buffer
    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(vis)

def box_corners_3d(bxyzwhl: np.ndarray) -> np.ndarray:
    """
    Convert [x, y, z, dx, dy, dz, yaw] -> [8, 3] corners.
    Assumes MMDet3D/NuScenes convention: (x,y,z) is bottom-center.
    """
    x, y, z, dx, dy, dz, yaw = bxyzwhl.tolist()
    
    # Rotation
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    
    # Box corners relative to center (z goes from 0 to h)
    l, w, h = dx, dy, dz
    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2], dtype=np.float32)
    y_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2], dtype=np.float32)
    z_c = np.array([ 0,    0,    0,    0,    h,    h,    h,    h  ], dtype=np.float32)
    
    pts = np.stack([x_c, y_c, z_c], axis=1)
    pts = (R @ pts.T).T + np.array([x,y,z], dtype=np.float32)
    return pts

def draw_boxes_on_image(img: Image.Image,
                        boxes: np.ndarray,
                        P_lidar2img: np.ndarray,
                        color=(0,255,0), 
                        width=2) -> Image.Image:
    """
    Draw 3D bounding boxes on 2D image.
    boxes: [N, 7] (x, y, z, dx, dy, dz, yaw)
    """
    img = img.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    
    if boxes is None or len(boxes) == 0:
        return img

    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]

    for b in boxes:
        if len(b) < 7: continue
        corners = box_corners_3d(b[:7]) # [8,3]
        uv, zc, mask = project_points(corners, P_lidar2img)
        
        # Skip if any corner is behind camera (simple clipping)
        if uv.shape[0] != 8: continue 
        
        # Draw lines
        for i, j in edges:
            x1, y1 = uv[i]
            x2, y2 = uv[j]
            # Check if roughly in view
            if (max(x1, x2) < 0 or min(x1, x2) >= w or max(y1, y2) < 0 or min(y1, y2) >= h):
                continue
            d.line((x1, y1, x2, y2), fill=color, width=width)
    return img

def overlay_points_and_boxes(img: Image.Image,
                             pts_xyz: np.ndarray,
                             P_l2i: np.ndarray,
                             pred_boxes: Optional[np.ndarray] = None,
                             gt_boxes: Optional[np.ndarray] = None,
                             depth_coloring=True) -> Image.Image:
    """
    Unified visualizer: Lidar Depth + GT Boxes (Red) + Pred Boxes (Green)
    """
    vis = img
    # 1. Lidar Depth
    if pts_xyz is not None and len(pts_xyz) > 0:
        uv, zc, mask = project_points(pts_xyz, P_l2i)
        w, h = img.size
        valid = (uv[:,0]>=0) & (uv[:,0]<w) & (uv[:,1]>=0) & (uv[:,1]<h)
        vis = overlay_points_depth(vis, uv[valid], zc[valid], cmap='turbo')
    
    # 2. GT Boxes (Red)
    if gt_boxes is not None and len(gt_boxes) > 0:
        vis = draw_boxes_on_image(vis, gt_boxes, P_l2i, color=(255,0,0), width=2)
        
    # 3. Pred Boxes (Green)
    if pred_boxes is not None and len(pred_boxes) > 0:
        vis = draw_boxes_on_image(vis, pred_boxes, P_l2i, color=(0,255,0), width=2)
        
    return vis

# ==============================================================================
# SECTION 3: DATA LOADING UTILITIES
# ==============================================================================

def load_nus_lidar_points(lidar_path: str) -> np.ndarray:
    """
    Load LiDAR points from .bin file.
    Format: [x, y, z, intensity, ring_index]
    Returns: [N, 5] numpy array
    """
    # nuScenes lidar binaries are float32, 5 values per point
    scan = np.fromfile(lidar_path, dtype=np.float32)
    points = scan.reshape((-1, 5))
    return points

def load_and_stack_images(img_paths: List[str], size_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Load list of images, resize, and stack into Tensor.
    Returns: [N_cams, 3, H, W] tensor, float32, normalized 0-1 (if required) or 0-255.
    NOTE: Adjust normalization (mean/std) here if your model expects it.
    """
    imgs = []
    H, W = size_hw
    for p in img_paths:
        # Open and resize
        im = Image.open(p).convert('RGB')
        im = im.resize((W, H), Image.BILINEAR)
        arr = np.array(im).astype(np.float32) # [H, W, 3]
        
        # Normalize to 0-1? Or keep 0-255?
        # Most MMDetection3D backbones expect normalized tensors.
        # Here we leave as 0-255 float32 for consistency with typical mmdet3d LoadImageFromFile.
        # If your config applies Normalize, do it there. If not, do it here.
        # For now: Transpose to [3, H, W]
        arr = arr.transpose(2, 0, 1)
        imgs.append(torch.from_numpy(arr))
        
    return torch.stack(imgs) # [N, 3, H, W]

def enrich_meta_for_predict(metainfo: dict, img_paths: List[str]):
    """
    Helper to populate standard fields MMDetection3D expects in metainfo.
    """
    metainfo['filename'] = img_paths
    metainfo['ori_shape'] = (900, 1600) # Original nuScenes size
    metainfo['img_shape'] = (900, 1600)
    metainfo['scale_factor'] = 1.0

# ==============================================================================
# SECTION 4: MAIN DATASET ITERATOR
# ==============================================================================

def iter_nuscenes_samples(dataroot: str, 
                          version: str = "v1.0-trainval", 
                          max_count: int = -1):
    """
    Generator that yields samples for inference.
    Automatically builds the precise calibration matrices.
    
    Yields:
        lidar_path (str): Path to .bin file
        img_paths (List[str]): Paths to 6 camera images
        metainfo (dict): Dictionary containing transformation matrices:
                         - lidar2img: [6, 4, 4]
                         - cam2img:   [6, 4, 4] (Intrinsics)
                         - lidar2cam: [6, 4, 4]
                         - cam2lidar: [6, 4, 4]
        basename (str): Sample identifier (usually sample_token or timestamp)
    """
    print(f"[Loader] Initializing NuScenes ({version})...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    
    # Standard camera order
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    count = 0
    for sample in nusc.sample:
        if max_count > 0 and count >= max_count:
            break
            
        # 1. Get LiDAR Path
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_sd = nusc.get('sample_data', lidar_token)
        lidar_path = osp.join(dataroot, lidar_sd['filename'])
        
        # 2. Get Image Paths & Calibration
        img_paths = []
        l2i_list, c2i_list, l2c_list, c2l_list = [], [], [], []
        
        for cam in cam_names:
            cam_token = sample['data'][cam]
            cam_sd = nusc.get('sample_data', cam_token)
            img_path = osp.join(dataroot, cam_sd['filename'])
            img_paths.append(img_path)
            
            # Compute Robust Transforms (Ego-Motion Compensated)
            l2c, c2l, l2i, K4 = _build_lidar2cam_chain(nusc, sample, cam_token, lidar_sd)
            
            l2i_list.append(l2i)
            c2i_list.append(K4) # Intrinsics
            l2c_list.append(l2c)
            c2l_list.append(c2l)

        # 3. Build Metainfo
        metainfo = {
            'lidar2img': np.stack(l2i_list), # [6, 4, 4]
            'cam2img':   np.stack(c2i_list), # [6, 4, 4]
            'lidar2cam': np.stack(l2c_list), # [6, 4, 4]
            'cam2lidar': np.stack(c2l_list), # [6, 4, 4]
            'sample_token': sample['token'],
            'box_type_3d': 'LiDAR'
        }
        
        # Use sample token or timestamp as basename
        basename = sample['token']
        
        yield lidar_path, img_paths, metainfo, basename
        count += 1