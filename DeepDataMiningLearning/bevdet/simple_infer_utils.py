"""
bevfusion_infer_utils.py

Utility functions for mmdetection3d / BEVFusion-style inference and
NuScenes evaluation.

This module is intentionally verbose and heavily commented so that the
*main* script can stay short and easy to modify (or hand to an AI).

Key features:
- Environment setup (MMDet3D registry + Torch inference settings)
- NuScenes data loading (raw lidar sweeps + 6 camera views)
- Projection & visualization helpers (2D multiview, Open3D)
- Inference / benchmarking loops (manual + runner-based)
- NuScenes-format JSON export + evaluation wrapper
"""

import os
import os.path as osp
import math
import time
import json
import datetime
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Sequence,
    Mapping,
    Iterable,
)
import numpy as np
import torch
import cv2
import psutil
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Suppress noisy warnings from some backends
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="mmcv.ops.sparse_structure")

# MMEngine / MMDet3D
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS, DATASETS
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.utils import register_all_modules
import mmdet3d  # for version & sanity

# NuScenes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.data_classes import DetectionBox
from pyquaternion import Quaternion

# Open3D (optional)
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

# -----------------------------------------------------------------------------
# Global configuration constants
# -----------------------------------------------------------------------------

# Score thresholds for different use cases
DEFAULT_SCORE_THRESH_MANUAL = 0.05  # for exported JSON / evaluation
DEFAULT_SCORE_THRESH_VIZ = 0.25     # for visualization-only loop

# Default point cloud range for NuScenes (meters)
PC_RANGE_DEFAULT = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)

# Standard NuScenes camera order
NUSC_CAMS = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]

# NuScenes attribute list (canonical ordering)
NUSCENES_ATTRIBUTES = [
    'cycle.with_rider', 'cycle.without_rider',
    'pedestrian.moving', 'pedestrian.standing', 'pedestrian.sitting_lying_down',
    'vehicle.moving', 'vehicle.parked', 'vehicle.stopped', 'None'
]

# -----------------------------------------------------------------------------
# 1. ENVIRONMENT & SYSTEM INFO
# -----------------------------------------------------------------------------

def setup_env(init_scope: bool = True) -> None:
    """
    Initialize MMDet3D / MMEngine environment.

    Args:
        init_scope: if True, calls `register_all_modules(init_default_scope=True)`
                    to ensure all APIs are registered and default scope is set.
    """
    if init_scope:
        register_all_modules(init_default_scope=True)


def configure_torch_for_inference() -> None:
    """
    Configure PyTorch global settings for inference-only workloads.

    - Enables cuDNN benchmark to auto-tune convolutions.
    - Disables gradient computation globally.
    """
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)


def get_system_info() -> Dict[str, Any]:
    """
    Collect basic system information for logging and benchmark metadata.

    Returns:
        dict with keys: timestamp, mmdet3d, gpu, memory_gb
    """
    try:
        sys_mem = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        sys_mem = "N/A"

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "mmdet3d": mmdet3d.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "memory_gb": sys_mem
    }

# -----------------------------------------------------------------------------
# 2. GEOMETRY HELPERS
# -----------------------------------------------------------------------------

def _quat_rot(q: Quaternion) -> np.ndarray:
    """Convert a Quaternion to a 3x3 rotation matrix (float32)."""
    return q.rotation_matrix.astype(np.float32)


def _Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform from rotation & translation.

    Args:
        R: (3, 3) rotation matrix
        t: (3,) translation vector

    Returns:
        (4, 4) float32 transform matrix
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _to_numpy_points(pts_item: Any) -> np.ndarray:
    """
    Convert a variety of container types used by MMEngine/MMDet3D
    into a NumPy array of shape (N, C).

    Handles:
      - Tensor-like objects with `.tensor`
      - Lists/tuples wrapping a single tensor/array
      - Plain PyTorch tensors
      - NumPy arrays

    Returns:
      np.ndarray of float32, or empty (0, 5) if input is None.

    NOTE: the channel count (C) is not enforced here; downstream code
    must handle (N, 4) vs (N, 5) etc.
    """
    if pts_item is None:
        return np.zeros((0, 5), dtype=np.float32)

    if hasattr(pts_item, 'tensor'):
        t = pts_item.tensor
    else:
        t = pts_item

    if isinstance(t, (list, tuple)) and len(t) > 0:
        t = t[0]

    if torch.is_tensor(t):
        return t.detach().cpu().numpy().astype(np.float32, copy=False)

    return np.asarray(t, dtype=np.float32)


def project_points(pts: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D lidar points into image plane using a 4x4 projection matrix.

    Args:
        pts: (N, C>=3) lidar points [x, y, z, ...]
        P:   (3 or 4, 4) lidar2img or camera projection

    Returns:
        uv:   (M, 2) projected pixel coordinates for points with z > 0.1
        z:    (M,) depth values corresponding to uv
        mask: (N,) boolean mask indicating which original points are valid
    """
    N = pts.shape[0]
    h = np.hstack([pts[:, :3], np.ones((N, 1), dtype=np.float32)])
    c = (P @ h.T).T  # (N, 4) or (N, 3)
    z = c[:, 2]
    mask = z > 0.1
    uv = c[mask, :2] / c[mask, 2:3]
    return uv, z[mask], mask

# -----------------------------------------------------------------------------
# 3. VISUALIZATION HELPERS (NuScenes GT, Open3D, 2D multiview)
# -----------------------------------------------------------------------------

def get_gt_boxes(nusc: NuScenes, sample_token: str) -> Optional[np.ndarray]:
    """
    Get ground-truth 3D boxes in lidar coordinate frame for a sample.

    The boxes are returned in BEV-style format:
        [x_center, y_center, z_bottom, dx, dy, dz, yaw]

    Args:
        nusc:         NuScenes instance
        sample_token: token of the sample

    Returns:
        (N, 7) float32 array or None if no boxes.
    """
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    sd = nusc.get('sample_data', lidar_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])

    boxes_out = []

    for ann in sample['anns']:
        a = nusc.get('sample_annotation', ann)
        box = Box(a['translation'], a['size'], Quaternion(a['rotation']))

        # Global -> ego
        box.translate(-np.array(pose['translation']))
        box.rotate(Quaternion(pose['rotation']).inverse)

        # Ego -> lidar
        box.translate(-np.array(cs['translation']))
        box.rotate(Quaternion(cs['rotation']).inverse)

        z_bot = box.center[2] - box.wlh[2] / 2.0
        yaw = box.orientation.yaw_pitch_roll[0]
        boxes_out.append([
            box.center[0], box.center[1], z_bot,
            box.wlh[1], box.wlh[0], box.wlh[2], yaw
        ])

    return np.array(boxes_out, dtype=np.float32) if boxes_out else None


def boxes_to_lineset(boxes: Optional[np.ndarray],
                     color: List[float]) -> Optional["o3d.geometry.LineSet"]:
    """
    Convert BEV-style 3D boxes into an Open3D LineSet for visualization.

    Args:
        boxes: (N, 7) array [x, y, z_bottom, dx, dy, dz, yaw]
        color: [r, g, b] in [0, 1] or [0, 255]

    Returns:
        Open3D LineSet instance or None if no boxes / Open3D disabled.
    """
    if not HAS_OPEN3D:
        return None
    if boxes is None or len(boxes) == 0:
        return None

    points, lines, colors = [], [], []
    for i, b in enumerate(boxes):
        x, y, z, dx, dy, dz, yaw = b[:7]
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]], dtype=np.float32)

        # 8 corners in local box frame
        xc = [dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2]
        yc = [dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2]
        zc = [0, 0, 0, 0, dz, dz, dz, dz]

        corn = (R @ np.vstack([xc, yc, zc])).T + np.array([x, y, z], dtype=np.float32)
        base = i * 8
        points.extend(corn.tolist())
        lines.extend([[base+u, base+v] for u, v in
                      [(0,1), (1,2), (2,3), (3,0),
                       (4,5), (5,6), (6,7), (7,4),
                       (0,4), (1,5), (2,6), (3,7)]])
        colors.extend([color] * 12)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def save_ply_files(out_dir: str,
                   token: str,
                   pts: np.ndarray,
                   pred_boxes: Optional[np.ndarray],
                   gt_boxes: Optional[np.ndarray]) -> None:
    """
    Save point cloud and predicted / GT box wireframes as PLY files.

    Useful for headless 3D inspection with Open3D or Meshlab.
    """
    if not HAS_OPEN3D:
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    o3d.io.write_point_cloud(osp.join(out_dir, f"{token}_points.ply"), pcd)

    ls_pred = boxes_to_lineset(pred_boxes, [0, 1, 0])
    if ls_pred is not None:
        o3d.io.write_line_set(osp.join(out_dir, f"{token}_pred.ply"), ls_pred)

    ls_gt = boxes_to_lineset(gt_boxes, [1, 0, 0])
    if ls_gt is not None:
        o3d.io.write_line_set(osp.join(out_dir, f"{token}_gt.ply"), ls_gt)


def run_open3d_viz(pts: np.ndarray,
                   pred_boxes: Optional[np.ndarray],
                   gt_boxes: Optional[np.ndarray],
                   window_name: str = "3D Detection") -> None:
    """
    Interactive Open3D visualization (requires DISPLAY + OpenGL).

    - Points are colored by height.
    - Predicted boxes are green, GT boxes are red.
    """
    if not HAS_OPEN3D:
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    colors = np.zeros((pts.shape[0], 3), dtype=np.float32)

    z = pts[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors[:, 1] = z_norm
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis.add_geometry(pcd)

    if pred_boxes is not None:
        ls = boxes_to_lineset(pred_boxes, [0, 1, 0])
        if ls is not None:
            vis.add_geometry(ls)
    if gt_boxes is not None:
        ls = boxes_to_lineset(gt_boxes, [1, 0, 0])
        if ls is not None:
            vis.add_geometry(ls)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0

    vis.run()
    vis.destroy_window()


def draw_2d_multiview(paths: List[str],
                      pts: np.ndarray,
                      l2i_list: np.ndarray,
                      pred_boxes: Optional[np.ndarray],
                      gt_boxes: Optional[np.ndarray],
                      token: str,
                      out_dir: str,
                      target_size: Tuple[int, int] = (256, 704),
                      scale: float = 0.48) -> None:
    """
    Render multi-view 2D images with lidar point projections and 3D boxes.

    Args:
        paths:      list of camera image paths (length 6 by default)
        pts:        (N, C) lidar points
        l2i_list:   (6, 4, 4) lidar2img matrices
        pred_boxes: (M, 7) predicted boxes in lidar frame
        gt_boxes:   (K, 7) ground-truth boxes in lidar frame
        token:      sample token (used in output filename)
        out_dir:    directory to save the resulting image
        target_size: (H, W) after crop (must match what model expects)
        scale:      resize scale factor applied before crop
    """
    if not paths:
        return

    canvases = []
    Ht, Wt = target_size

    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        nH, nW = int(h * scale), int(w * scale)
        img = cv2.resize(img, (nW, nH))

        sx = (nW - Wt) // 2
        sy = (nH - Ht) // 2
        img = img[sy:sy + Ht, sx:sx + Wt].copy()

        P = l2i_list[i]
        uv, z, _ = project_points(pts, P)

        # Colorize lidar points by depth
        if len(uv) > 0:
            d_norm = np.clip(z, 0, 60) / 60 * 255
            cols = cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_JET).reshape(-1, 3)
            h_img, w_img = img.shape[:2]
            uv_int = uv.astype(int)

            for j, (x, y) in enumerate(uv_int):
                if 0 <= x < w_img and 0 <= y < h_img:
                    img[y, x] = cols[j]

        def draw_boxes(boxes: Optional[np.ndarray], color: Tuple[int, int, int]):
            if boxes is None:
                return
            for b in boxes:
                if len(b) < 7:
                    continue
                x, y, zb, dx, dy, dz, yaw = b[:7]
                c_cos, s_sin = math.cos(yaw), math.sin(yaw)
                R = np.array([[c_cos, -s_sin, 0],
                              [s_sin,  c_cos, 0],
                              [0,      0,     1]], dtype=np.float32)
                xc = [dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2]
                yc = [dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2]
                zc = [0, 0, 0, 0, dz, dz, dz, dz]
                corn = (R @ np.vstack([xc, yc, zc])).T + np.array([x, y, zb], dtype=np.float32)

                uv_b, _, _ = project_points(corn, P)
                if len(uv_b) < 8:
                    continue
                uv_b = uv_b.astype(int)
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)
                ]
                for u, v in edges:
                    p1 = tuple(uv_b[u])
                    p2 = tuple(uv_b[v])
                    cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)

        draw_boxes(gt_boxes, (0, 0, 255))   # red
        draw_boxes(pred_boxes, (0, 255, 0)) # green
        canvases.append(img)

    if len(canvases) == 6:
        final = np.vstack([np.hstack(canvases[:3]), np.hstack(canvases[3:])])
    elif len(canvases) > 0:
        final = np.hstack(canvases)
    else:
        return

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(osp.join(out_dir, f"{token}_multiview.jpg"), final)

def draw_2d_multiview_from_tensor(
    imgs_tensor,
    meta,
    pts,
    l2i_list,
    pred_boxes,
    gt_boxes,
    token,
    out_dir,
):
    """
    Draw multiview projection using the **already preprocessed** images from
    the cfg-based dataloader.

    This avoids re-reading files and re-applying hard-coded crops / scales.
    We simply:
      1) De-normalize imgs using meta['img_norm_cfg']
      2) Convert back to BGR for cv2 if needed
      3) Project pts & boxes with meta['lidar2img'] (which matches imgs)

    Args:
        imgs_tensor: (N, 3, H, W) torch.Tensor used for inference.
        meta:        metainfo dict from Det3DDataSample.
                     Must contain 'img_norm_cfg' and 'lidar2img'.
        pts:         (M, C) numpy array in lidar coordinates.
        l2i_list:    (N, 4, 4) numpy array of lidar2img matrices.
        pred_boxes:  (K, 7) predicted boxes in lidar frame or None.
        gt_boxes:    (G, 7) GT boxes in lidar frame or None.
        token:       string id for saving.
        out_dir:     directory to write <token>_multiview.jpg.
    """
    if imgs_tensor is None:
        return
    if 'img_norm_cfg' not in meta:
        # Fallback: you can call the old draw_2d_multiview with raw paths
        return

    os.makedirs(out_dir, exist_ok=True)

    imgs = imgs_tensor.detach().cpu().numpy()  # (N, C, H, W)
    img_norm_cfg = meta['img_norm_cfg']
    mean = np.array(img_norm_cfg['mean'], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(img_norm_cfg['std'], dtype=np.float32).reshape(1, 3, 1, 1)
    to_rgb = img_norm_cfg.get('to_rgb', False)

    # de-normalize: x = x * std + mean
    imgs = imgs * std + mean
    imgs = np.clip(imgs, 0, 255).astype(np.uint8)

    canvases = []
    num_views = imgs.shape[0]

    for i in range(num_views):
        # CHW -> HWC
        img = imgs[i].transpose(1, 2, 0)  # (H, W, 3)

        # If the pipeline converted BGR->RGB (to_rgb=True), convert back to BGR
        # for cv2 visualization.
        if to_rgb:
            img = img[..., ::-1]  # RGB -> BGR

        P = l2i_list[i]

        # Project lidar points
        uv, z, _ = project_points(pts, P)
        if len(uv) > 0:
            d_norm = np.clip(z, 0, 60) / 60.0 * 255
            cols = cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_JET).reshape(-1, 3)
            h_img, w_img = img.shape[:2]
            for j, (x, y) in enumerate(uv.astype(int)):
                if 0 <= x < w_img and 0 <= y < h_img:
                    img[y, x] = cols[j]

        # Inner helper to draw 3D boxes in this view
        def draw_b(boxes, c):
            if boxes is None:
                return
            for b in boxes:
                if len(b) < 7:
                    continue
                x, y, zb, dx, dy, dz, yaw = b[:7]
                c_cos, s_sin = math.cos(yaw), math.sin(yaw)
                R = np.array([[c_cos, -s_sin, 0],
                              [s_sin,  c_cos, 0],
                              [0,      0,     1]])
                xc = [dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2]
                yc = [dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2]
                zc = [0, 0, 0, 0, dz, dz, dz, dz]
                corn = (R @ np.vstack([xc, yc, zc])).T + np.array([x, y, zb])
                uv_b, z_b, _ = project_points(corn, P)
                if len(uv_b) < 8:
                    continue
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)
                ]
                for u, v in edges:
                    p1 = tuple(uv_b[u].astype(int))
                    p2 = tuple(uv_b[v].astype(int))
                    cv2.line(img, p1, p2, c, 2, cv2.LINE_AA)

        # GT: red, Pred: green
        draw_b(gt_boxes, (0, 0, 255))
        draw_b(pred_boxes, (0, 255, 0))

        canvases.append(img)

    if not canvases:
        return

    if len(canvases) == 6:
        final = np.vstack([np.hstack(canvases[:3]),
                           np.hstack(canvases[3:])])
    else:
        final = np.hstack(canvases)

    cv2.imwrite(osp.join(out_dir, f"{token}_multiview.jpg"), final)
    
# -----------------------------------------------------------------------------
# 4. DATA LOADERS (BaseLoader, NuScenesLoader, cfg_iter, build_loader_pack)
# -----------------------------------------------------------------------------

class BaseLoader(Dataset):
    """Abstract base dataset class to unify typing."""
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


def custom_collate(batch: List[Dict[str, Any]]) -> Tuple[Any, ...]:
    """
    Collate function for NuScenesLoader to return a simple tuple.

    Returns:
        token, points, imgs, metainfo, paths, gt_boxes
    """
    item = batch[0]
    return (
        item['token'],
        item['points'],
        item['imgs'],
        item['metainfo'],
        item['paths'],
        item['gt_boxes'],
    )


def _identity_collate(batch: List[Any]) -> Any:
    """Collate that just returns the first element (batch_size == 1)."""
    return batch[0]


class NuScenesLoader(BaseLoader):
    """
    Standalone NuScenes inference loader.

    Responsibilities:
      - Reads raw lidar sweeps and transforms them into the reference
        LIDAR_TOP frame.
      - Reads and crops camera images to a fixed target size, applying
        the same resize/crop as the training pipeline.
      - Bakes lidar2img / lidar2cam extrinsics & intrinsics into
        metainfo expected by BEVFusion-style models.

    Configuration knobs:
      - nsweeps:     number of past lidar sweeps
      - pc_range:    axis-aligned crop (xm, ym, zm, xM, yM, zM)
      - expects_bgr: if False, convert BGR->RGB
      - crop_policy: currently only "center" crop is supported
    """

    def __init__(self,
                 dataroot: str,
                 version: str,
                 split: str = 'val',
                 max_samples: int = -1,
                 nsweeps: int = 10,
                 expects_bgr: bool = True,
                 pc_range: Tuple[float, float, float, float, float, float] = PC_RANGE_DEFAULT,
                 crop_policy: str = "center") -> None:
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = dataroot
        self.expects_bgr = expects_bgr
        self.nsweeps = int(nsweeps)
        self.pc_range = list(pc_range)
        self.crop_policy = crop_policy

        # Map requested split to NuScenes split name
        if version == 'v1.0-mini':
            split_name = 'mini_val'
        elif split == 'val':
            split_name = 'val'
        else:
            split_name = 'train'

        all_splits = create_splits_scenes()
        scenes = all_splits.get(split_name, all_splits['val'])

        self.samples = []
        scene_set = set(scenes)
        for scene in self.nusc.scene:
            if scene['name'] not in scene_set:
                continue
            tok = scene['first_sample_token']
            while tok:
                s = self.nusc.get('sample', tok)
                self.samples.append(s)
                tok = s['next']

        if max_samples != -1:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    # --- Transform helpers ----------------------------------------------------

    def get_sensor_transforms(self, sd_record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get lidar/camera-to-ego and ego-to-global transforms
        for a given sample_data record.
        """
        cs = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])

        l2e_t = np.array(cs['translation'], dtype=np.float32)
        l2e_r = _quat_rot(Quaternion(cs['rotation']))
        e2g_t = np.array(pose['translation'], dtype=np.float32)
        e2g_r = _quat_rot(Quaternion(pose['rotation']))
        return l2e_t, l2e_r, e2g_t, e2g_r

    # --- Lidar loading --------------------------------------------------------

    def load_points(self, sample: Dict[str, Any]) -> np.ndarray:
        """
        Load nsweeps of lidar points and transform them into the
        reference LIDAR_TOP frame. Adds a time-lag channel `dt`.

        Output shape: (N, 5) with [x, y, z, intensity, dt]
        """
        lidar_token = sample['data']['LIDAR_TOP']
        ref_sd = self.nusc.get('sample_data', lidar_token)
        ref_l2e_t, ref_l2e_r, ref_e2g_t, ref_e2g_r = self.get_sensor_transforms(ref_sd)

        # Precompute inverse transforms: global->ego and ego->lidar
        ref_g2e_r, ref_g2e_t = ref_e2g_r.T, -ref_e2g_r.T @ ref_e2g_t
        ref_e2l_r, ref_e2l_t = ref_l2e_r.T, -ref_l2e_r.T @ ref_l2e_t

        all_pts = []
        curr_sd = ref_sd
        for i in range(self.nsweeps):
            path = osp.join(self.dataroot, curr_sd['filename'])
            pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]  # keep intensity

            if i > 0:
                c_l2e_t, c_l2e_r, c_e2g_t, c_e2g_r = self.get_sensor_transforms(curr_sd)
                pts_xyz = pts[:, :3]
                # Lidar -> ego (current)
                pts_xyz = (c_l2e_r @ pts_xyz.T).T + c_l2e_t
                # Ego -> global
                pts_xyz = (c_e2g_r @ pts_xyz.T).T + c_e2g_t
                # Global -> ego (ref)
                pts_xyz = (ref_g2e_r @ pts_xyz.T).T + ref_g2e_t
                # Ego (ref) -> lidar (ref)
                pts_xyz = (ref_e2l_r @ pts_xyz.T).T + ref_e2l_t
                pts[:, :3] = pts_xyz

            dt = (ref_sd['timestamp'] - curr_sd['timestamp']) / 1e6
            dt_col = np.full((len(pts), 1), dt, dtype=np.float32)
            all_pts.append(np.hstack([pts, dt_col]))

            if curr_sd['prev'] == '':
                break
            curr_sd = self.nusc.get('sample_data', curr_sd['prev'])

        pts_all = np.concatenate(all_pts, axis=0)

        # Axis-aligned crop
        x, y, z = pts_all[:, 0], pts_all[:, 1], pts_all[:, 2]
        xm, ym, zm, xM, yM, zM = self.pc_range
        mask = (x >= xm) & (x <= xM) & (y >= ym) & (y <= yM) & (z >= zm) & (z <= zM)
        return pts_all[mask]

    # --- Image loading --------------------------------------------------------

    def load_imgs(self, sample: Dict[str, Any],
                  target_size: Tuple[int, int] = (256, 704),
                  scale: float = 0.48) -> Tuple[torch.Tensor, Dict[str, Any], List[str]]:
        """
        Load and preprocess all 6 nuScenes cameras:

        - Resize by fixed scale factor.
        - Center crop to target size.
        - Apply same geometric transform to intrinsics K.
        - Compute lidar2cam / lidar2img extrinsics.

        Args:
            sample:       NuScenes sample dict
            target_size:  (H, W) after crop
            scale:        resize scale factor

        Returns:
            imgs:      (6, 3, H, W) float32 tensor
            metainfo: dict for Det3DDataSample
            paths:    list of image paths
        """
        Ht, Wt = target_size

        tensors, paths = [], []
        lidar2img, cam2img, lidar2cam = [], [], []

        lidar_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        l2e_t, l2e_r, e2g_t_L, e2g_r_L = self.get_sensor_transforms(lidar_sd)

        for c_name in NUSC_CAMS:
            c_sd = self.nusc.get('sample_data', sample['data'][c_name])
            path = osp.join(self.dataroot, c_sd['filename'])
            paths.append(path)

            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {path}")

            if not self.expects_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            nH, nW = int(h * scale), int(w * scale)
            img = cv2.resize(img, (nW, nH))

            # Center crop
            sx = (nW - Wt) // 2
            sy = (nH - Ht) // 2
            img = img[sy:sy + Ht, sx:sx + Wt].copy()

            tensors.append(torch.from_numpy(img).permute(2, 0, 1).float())

            # Camera intrinsics
            c_cal = self.nusc.get('calibrated_sensor', c_sd['calibrated_sensor_token'])
            K = np.eye(4, dtype=np.float32)
            K[:3, :3] = c_cal['camera_intrinsic']

            # Bake scale and crop into K
            K[0] *= scale
            K[1] *= scale
            K[0, 2] -= sx
            K[1, 2] -= sy
            cam2img.append(K)

            # Camera extrinsics: lidar -> cam
            c2e_t, c2e_r, e2g_t_C, e2g_r_C = self.get_sensor_transforms(c_sd)
            T = (np.linalg.inv(_Rt(c2e_r, c2e_t)) @
                 np.linalg.inv(_Rt(e2g_r_C, e2g_t_C)) @
                 _Rt(e2g_r_L, e2g_t_L) @
                 _Rt(l2e_r, l2e_t))
            lidar2cam.append(T)
            lidar2img.append(K @ T)

        aug_matrix = np.tile(np.eye(4, dtype=np.float32)[None], (len(NUSC_CAMS), 1, 1))

        metainfo = {
            'lidar2img': np.stack(lidar2img),
            'cam2img': np.stack(cam2img),
            'lidar2cam': np.stack(lidar2cam),
            'cam2lidar': np.linalg.inv(np.stack(lidar2cam)),
            'img_aug_matrix': aug_matrix,
            'img_shape': [target_size] * len(NUSC_CAMS),
            'box_type_3d': LiDARInstance3DBoxes,
        }

        return torch.stack(tensors), metainfo, paths

    # --- Item -----------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        pts = self.load_points(sample)
        imgs, meta, paths = self.load_imgs(sample)
        meta['token'] = sample['token']
        return {
            'token': sample['token'],
            'points': pts,
            'imgs': imgs,
            'metainfo': meta,
            'paths': paths,
            'gt_boxes': None
        }

# -----------------------------------------------------------------------------
# 5. CFG-BASED LOADER ADAPTERS
# -----------------------------------------------------------------------------

def cfg_iter(loader: Iterable[Any]) -> Iterable[Tuple[Any, ...]]:
    """
    Adapter to yield a unified tuple signature from an MMEngine-style
    DataLoader.

    Yields:
      token:       generic ID used for filenames/logging (string)
      pts:         (N, C) numpy array of lidar points
      imgs:        (6, 3, H, W) tensor or None
      meta:        metainfo dict used for evaluation / viz
      paths:       list of image paths (if available)
      gt_boxes:    currently None (can be extended)

    NOTE:
      - For NuScenes GT lookup we should use meta['token'] (NuScenes sample_token)
      - `token` here may be a simple index (sample_idx) and SHOULD NOT be used
        for nusc.get('sample', token).
    """
    for sample in loader:
        if isinstance(sample, dict) and 'data_samples' in sample:
            ds = sample['data_samples'][0] if isinstance(sample['data_samples'], list) else sample['data_samples']
            inp = sample['inputs']
        else:
            ds = sample.data_samples
            inp = sample.inputs

        meta = ds.metainfo

        # NuScenes sample_token (if provided by the dataset)
        nus_token = meta.get('token', None)

        # Generic ID for logging / filenames:
        #   - use NuScenes token if available,
        #   - else fall back to sample_idx.
        token = str(nus_token if nus_token is not None else meta.get('sample_idx', ''))

        pts_raw = inp['points'][0] if isinstance(inp['points'], list) else inp['points']
        pts = _to_numpy_points(pts_raw)

        imgs = inp.get('img', None)
        if isinstance(imgs, list):
            imgs = imgs[0]
        if torch.is_tensor(imgs) and imgs.dim() == 5:
            imgs = imgs.squeeze(0)

        paths = meta.get('img_path', meta.get('img_paths', []))
        if isinstance(paths, str):
            paths = [paths]

        yield token, pts, imgs, meta, paths, None


def patch_cfg_paths(cfg: Config,
                    dataroot: str,
                    ann_file: str = "") -> None:
    """
    Recursively patch config's test_dataloader to override:
      - data_root
      - ann_file

    This allows using the same config with different data locations.
    """
    def _patch(node: Any) -> None:
        if isinstance(node, dict):
            if 'data_root' in node:
                node['data_root'] = dataroot
            if 'ann_file' in node:
                if ann_file:
                    node['ann_file'] = ann_file
                else:
                    val = node['ann_file']
                    node['ann_file'] = (osp.join(dataroot, val)
                                        if not osp.isabs(val) else val)
            for v in node.values():
                _patch(v)
        elif isinstance(node, list):
            for v in node:
                _patch(v)

    _patch(cfg.test_dataloader)

def build_loader_pack(
    data_source: str,
    cfg: Config,
    dataroot: str,
    nus_version: str = "v1.0-trainval",
    ann_file: str = "",
    max_samples: int = -1,
    crop_policy: str = "center",
    workers: int = 4,
    dataset: str = "nuscenes",
) -> Dict[str, Any]:
    """
    Build loader+iterator+dataset-wrapper for manual/visual loops.

    Args:
        data_source: "cfg" or "custom"
        cfg:         Config object
        dataroot:    dataset root path
        nus_version: NuScenes version (NuScenes-only)
        ann_file:    info pkl path (cfg mode)
        max_samples: max samples for custom loader
        crop_policy: crop policy for custom loader
        workers:     DataLoader workers
        dataset:     "nuscenes" or "kitti"

    Returns:
        dict(
          loader=...,
          iter_fn=...,
          nusc=NuScenes_or_None,
          sample_tokens=Optional[List[str]],   # <-- NEW for NuScenes+cfg
          dataset=dataset_obj                  # <-- optional, but handy
        )
    """
    # ------------------------------------------------------------------
    # PATH A: Use dataset defined in config (NuScenesDataset / KittiDataset, etc.)
    # ------------------------------------------------------------------
    if data_source == "cfg":
        # Build dataset from cfg.test_dataloader.dataset
        dataset_obj = DATASETS.build(cfg.test_dataloader.dataset)
        loader = DataLoader(
            dataset_obj,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            collate_fn=_identity_collate,
            pin_memory=True,
        )

        nusc = None
        sample_tokens = None  # <-- NEW

        if dataset.lower() == "nuscenes":
            # NuScenes API instance (only needed for evaluation)
            nusc = NuScenes(version=nus_version, dataroot=dataroot, verbose=False)

            # ------------------------------------------------------------------
            # NEW: Build ordered list of *true* NuScenes sample tokens.
            #      Order MUST match dataset_obj's indexing.
            #      NuScenesDataset from MMDet3D exposes data_infos with 'token'.
            # ------------------------------------------------------------------
            sample_tokens = []
            data_infos = getattr(dataset_obj, "data_infos", [])
            for idx, info in enumerate(data_infos):
                tok = info.get("token", None)
                if tok is None:
                    # Fallback – shouldn't happen for standard nuscenes_infos_*.pkl
                    tok = str(idx)
                sample_tokens.append(tok)

        # pack is same as before, but with extra fields
        return dict(
            loader=loader,
            iter_fn=cfg_iter,       # your existing iterator
            nusc=nusc,
            sample_tokens=sample_tokens,  # <-- NEW
            dataset=dataset_obj,          # <-- optional, for debugging
        )

    # ------------------------------------------------------------------
    # PATH B: "custom" raw loader path: currently only implemented for NuScenes
    # ------------------------------------------------------------------
    if dataset != "nuscenes":
        raise NotImplementedError(
            "Custom raw loader is only implemented for NuScenes. "
            "Use --data-source cfg for KITTI."
        )

    dataset_obj = NuScenesLoader(
        dataroot=dataroot,
        version=nus_version,
        max_samples=max_samples,
        crop_policy=crop_policy,
        nsweeps=10,
    )
    loader = DataLoader(
        dataset_obj,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=custom_collate,
    )

    def custom_iter(dl):
        for t, p, i, m, pa, g in dl:
            yield t, p, i, m, pa, g

    # Note: here NuScenesLoader already yields true tokens `t`,
    # so we don't need sample_tokens.
    return dict(loader=loader, iter_fn=custom_iter, nusc=dataset_obj.nusc)
# -----------------------------------------------------------------------------
# 6. MODEL / RUNNER HELPERS
# -----------------------------------------------------------------------------

def load_model_from_cfg(config_path: str,
                        checkpoint_path: str,
                        device: str = "cuda",
                        dataroot: Optional[str] = None,
                        ann_file: str = "",
                        work_dir: Optional[str] = None) -> Tuple[torch.nn.Module, Config]:
    """
    Construct an MMDet3D model from config + checkpoint.

    Args:
        config_path:   path to config file (.py)
        checkpoint_path: path to checkpoint (.pth)
        device:        device string, e.g., "cuda", "cuda:0", or "cpu"
        dataroot:      optional override for data_root in cfg.test_dataloader
        ann_file:      optional override for ann_file
        work_dir:      optional work_dir to attach to cfg

    Returns:
        model:  eval-mode model on `device`
        cfg:    loaded and patched Config object
    """
    cfg = Config.fromfile(config_path)
    if dataroot is not None:
        patch_cfg_paths(cfg, dataroot, ann_file)
    if work_dir is not None:
        cfg.work_dir = work_dir

    model = MODELS.build(cfg.model)
    if hasattr(cfg, 'test_cfg'):
        model.test_cfg = cfg.test_cfg

    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(torch.device(device)).eval()
    return model, cfg


def run_runner_benchmark(config_path: str,
                         checkpoint_path: str,
                         dataroot: str,
                         ann_file: str = "",
                         out_dir: str = "results") -> None:
    """
    Use MMEngine Runner to execute the config's own test loop.

    - Useful as a baseline: "stock" eval without custom scripts.

    Args:
        config_path:   config file
        checkpoint_path: checkpoint to load
        dataroot:      NuScenes dataroot
        ann_file:      optional info pkl override
        out_dir:       working directory for Runner outputs
    """
    print("\n" + "=" * 60 + "\n STARTING RUNNER BENCHMARK\n" + "=" * 60)

    cfg = Config.fromfile(config_path)
    patch_cfg_paths(cfg, dataroot, ann_file)
    cfg.work_dir = out_dir
    cfg.load_from = checkpoint_path

    runner = Runner.from_cfg(cfg)
    runner.test()

from mmengine.hooks import Hook
import numpy as np  # make sure this import exists at top of file


class PerfHook(Hook):
    """
    Simple performance hook for MMEngine Runner test loop.

    - Measures per-iteration latency (ms).
    - Tracks peak GPU memory (MB) per iteration.

    Use `get_summary()` after Runner.test() to retrieve metrics.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        if torch.cuda.is_available() and device.startswith("cuda"):
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self.latencies_ms: List[float] = []
        self.peak_mems_mb: List[float] = []
        self._t0: float = 0.0

    def before_test_iter(self, runner, batch_idx: int, data_batch=None):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._t0 = time.perf_counter()

    def after_test_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        else:
            peak_mem = 0.0

        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self.latencies_ms.append(dt_ms)
        self.peak_mems_mb.append(peak_mem)

    def get_summary(self) -> Dict[str, float]:
        """Return dict with mean latency/std and max peak memory."""
        if not self.latencies_ms:
            return {}
        return {
            "mean_latency_ms": float(np.mean(self.latencies_ms)),
            "std_latency_ms": float(np.std(self.latencies_ms)),
            "max_peak_mem_mb": float(max(self.peak_mems_mb)),
        }


def run_benchmark_evaluation(args, sys_info: Dict[str, Any]) -> None:
    """
    Runs the full MMEngine Runner test loop using the config.
    Used when --eval is set with --eval-backend runner.

    This path relies on the config's own test_dataloader/evaluator,
    so you get the 'official' metrics exactly as MMDet3D expects.

    DATASETS SUPPORTED:
      - nuscenes
      - kitti

    For both datasets we:
      * Patch cfg.test_dataloader.dataset.{data_root, ann_file}
      * Patch cfg.test_evaluator.{data_root, ann_file} so the metric
        can re-load the correct info pkl from args.dataroot.
    """
    print("\n" + "=" * 60)
    print(" STARTING BENCHMARK EVALUATION (Runner backend)")
    print("=" * 60)

    # 1. Config Setup
    cfg = Config.fromfile(args.config)
    cfg.setdefault('default_scope', 'mmdet3d')

    # Override Config with Args
    cfg.work_dir = args.out_dir
    cfg.load_from = args.checkpoint

    # ------------------------------------------------------------------
    # Patch dataset paths for test_dataloader
    # ------------------------------------------------------------------
    if hasattr(cfg, "test_dataloader") and hasattr(cfg.test_dataloader, "dataset"):
        ds = cfg.test_dataloader.dataset

        # data_root override
        if hasattr(ds, 'data_root'):
            print(f"[Runner Eval] Overriding dataset.data_root -> {args.dataroot}")
            ds.data_root = args.dataroot

        # ann_file override
        if args.ann_file:
            # Explicit override from CLI
            print(f"[Runner Eval] Overriding dataset.ann_file -> {args.ann_file}")
            ds.ann_file = args.ann_file
        else:
            # Choose default ann_file by dataset type if none given
            if args.dataset == 'nuscenes':
                ds.ann_file = 'nuscenes_infos_val.pkl'
            elif args.dataset == 'kitti':
                ds.ann_file = 'kitti_infos_val.pkl'
            # For other datasets, leave as in config

    # ------------------------------------------------------------------
    # Patch evaluator / metric paths (NuScenesMetric, KittiMetric, ...)
    # ------------------------------------------------------------------
    def _patch_metric(metric_cfg: Dict[str, Any]) -> None:
        """
        Patch a single metric config dict so data_root / ann_file are
        consistent with args.dataroot and args.ann_file.

        Works for both:
          - NuScenesMetric
          - KittiMetric
        because they both use these fields to reload info pkls.
        """
        if not isinstance(metric_cfg, dict):
            return

        # data_root override
        if 'data_root' in metric_cfg:
            metric_cfg['data_root'] = args.dataroot

        # ann_file override
        if args.ann_file:
            metric_cfg['ann_file'] = args.ann_file
        else:
            if 'ann_file' in metric_cfg and not osp.isabs(metric_cfg['ann_file']):
                metric_cfg['ann_file'] = osp.join(args.dataroot, metric_cfg['ann_file'])

    # test_evaluator is used by Runner.test()
    if hasattr(cfg, "test_evaluator"):
        if isinstance(cfg.test_evaluator, list):
            for metric_cfg in cfg.test_evaluator:
                _patch_metric(metric_cfg)
        else:
            _patch_metric(cfg.test_evaluator)

    # (Optional) If you ever use Runner.val(), patch val_evaluator too
    if hasattr(cfg, "val_evaluator"):
        if isinstance(cfg.val_evaluator, list):
            for metric_cfg in cfg.val_evaluator:
                _patch_metric(metric_cfg)
        else:
            _patch_metric(cfg.val_evaluator)

    # 2. Build Runner
    runner = Runner.from_cfg(cfg)

    # 3. Register Perf Hook
    perf_hook = PerfHook(device=args.device)
    runner.register_hook(perf_hook, priority='LOW')

    # 4. Run Test
    results = runner.test()

    # 5. Extract metrics
    final_metrics: Dict[str, Any] = {}
    if isinstance(results, dict):
        for key, value in results.items():
            final_metrics[key] = value
    else:
        final_metrics["raw_results"] = results

    # 6. Merge performance stats
    perf_stats = perf_hook.get_summary()

    output = {
        "system_info": sys_info,
        "accuracy_metrics": final_metrics,
        "performance_metrics": perf_stats
    }

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = osp.join(args.out_dir, "benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=4)

    print("\n" + "=" * 60)
    print(" Benchmark Complete.")
    print(f" Latency (Mean): {perf_stats.get('mean_latency_ms', 0):.2f} ms")
    print(f" Memory  (Max):  {perf_stats.get('max_peak_mem_mb', 0):.2f} MB")
    print(f" Results saved to: {json_path}")
    print("=" * 60 + "\n")

# -----------------------------------------------------------------------------
# 7. NUSCENES PREDICTION CONVERSION
# -----------------------------------------------------------------------------

def _canon_nus_name(name: str) -> str:
    """
    Map arbitrary class name strings to canonical NuScenes class names.
    """
    name = name.lower()
    mapping = {
        'ped': 'pedestrian',
        'person': 'pedestrian',
        'bike': 'bicycle',
        'bus': 'bus',
        'car': 'car',
        'construction': 'construction_vehicle',
        'trailer': 'trailer',
        'truck': 'truck',
        'cone': 'traffic_cone',
        'barrier': 'barrier',
        'motor': 'motorcycle',
    }
    for k, v in mapping.items():
        if k in name:
            return v
    return 'car'


def get_default_attribute(label_name: str,
                          velocity: np.ndarray) -> str:
    """
    Heuristic attribute assignment when model does not output attributes.
    """
    v = np.linalg.norm(velocity[:2])
    if 'vehicle' in label_name or 'car' in label_name:
        return 'vehicle.moving' if v > 0.2 else 'vehicle.parked'
    if 'pedestrian' in label_name:
        return 'pedestrian.moving' if v > 0.2 else 'pedestrian.standing'
    if 'cycle' in label_name:
        return 'cycle.with_rider'
    return ''


def lidar_to_global_box(nusc: NuScenes,
                        token: str,
                        boxes: np.ndarray,
                        scores: np.ndarray,
                        labels: np.ndarray,
                        class_names: List[str],
                        attrs: Optional[np.ndarray] = None,
                        vels: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """
    Convert model outputs in lidar coordinates into NuScenes global frame
    and pack them into detection dicts ready for NuScenes evaluation.

    Args:
        nusc:        NuScenes instance
        token:       sample token
        boxes:       (N, 7+?) array [x, y, z, dx, dy, dz, yaw, ...]
        scores:      (N,) detection scores
        labels:      (N,) class indices
        class_names: list mapping label index -> class name
        attrs:       optional (N,) attribute indices
        vels:        optional (N, 2 or 3) velocities in lidar frame

    Returns:
        list of dicts following NuScenes detection JSON schema.
    """
    box_list = []
    sd_rec = nusc.get('sample_data', nusc.get('sample', token)['data']['LIDAR_TOP'])

    cs = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd_rec['ego_pose_token'])

    l2e_r = Quaternion(cs['rotation'])
    l2e_t = np.array(cs['translation'])
    e2g_r = Quaternion(pose['rotation'])
    e2g_t = np.array(pose['translation'])

    for i in range(len(boxes)):
        x, y, z, dx, dy, dz, yaw = boxes[i][:7]
        quat = Quaternion(axis=[0, 0, 1], radians=yaw)

        vx, vy = (vels[i][:2] if vels is not None else (0.0, 0.0))
        v_lidar = np.array([vx, vy, 0.0])
        v_global = e2g_r.rotate(l2e_r.rotate(v_lidar))

        # Note: Box takes center with z at middle, size [w, l, h]
        box = Box(
            center=[x, y, z + dz / 2.0],
            size=[dy, dx, dz],
            orientation=quat,
            label=int(labels[i]),
            score=float(scores[i])
        )
        box.rotate(l2e_r)
        box.translate(l2e_t)
        box.rotate(e2g_r)
        box.translate(e2g_t)

        cname = _canon_nus_name(class_names[box.label] if class_names else "car")
        aname = ""
        if attrs is not None and int(attrs[i]) < len(NUSCENES_ATTRIBUTES):
            aname = NUSCENES_ATTRIBUTES[int(attrs[i])]
        if not aname:
            aname = get_default_attribute(cname, v_global)

        box_list.append({
            "sample_token": token,
            "translation": box.center.tolist(),
            "size": box.wlh.tolist(),
            "rotation": box.orientation.elements.tolist(),
            "velocity": v_global[:2].tolist(),
            "detection_name": cname,
            "detection_score": box.score,
            "attribute_name": aname
        })

    return box_list

# -----------------------------------------------------------------------------
# 8. INFERENCE / BENCHMARK LOOPS
# -----------------------------------------------------------------------------
from mmengine.structures import InstanceData
import os
import os.path as osp
import json
import time
import warnings
from typing import Any, Dict, Sequence, Union

import numpy as np
import torch
from tqdm import tqdm

from mmengine.structures import InstanceData
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes

from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox

def _to_device(x, device):
    """Small helper: convert numpy/CPU tensors to given device."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x

def _resolve_sample_token(
    loader_token: Any,
    meta: Dict[str, Any],
    pack: Dict[str, Any],
) -> Union[str, None]:
    """
    Best-effort resolution of a NuScenes sample token.

    Priority:
      1) meta['token'] if present.
      2) meta['sample_idx'] + dataset.data_infos[idx]['token'].
      3) loader_token interpreted as integer index into dataset.data_infos.
      4) loader_token used directly as a NuScenes sample token (only if valid).

    If nothing works, returns None and the sample will be skipped for detection
    metrics (but still used for perf stats).
    """
    # 1) Direct token from metainfo
    tok = meta.get('token', None)
    if isinstance(tok, str) and len(tok) > 0:
        return tok

    ds = pack.get('dataset', None)

    # 2) Use sample_idx from meta to index into dataset.data_infos
    sample_idx = meta.get('sample_idx', None)
    if ds is not None and sample_idx is not None:
        try:
            idx = int(sample_idx)
            if hasattr(ds, 'data_infos') and 0 <= idx < len(ds.data_infos):
                info = ds.data_infos[idx]
                if isinstance(info.get('token', None), str):
                    return info['token']
        except Exception:
            pass

    # 3) Treat loader_token as dataset index
    if ds is not None and hasattr(ds, 'data_infos'):
        try:
            idx = int(loader_token)
            if 0 <= idx < len(ds.data_infos):
                info = ds.data_infos[idx]
                if isinstance(info.get('token', None), str):
                    return info['token']
        except Exception:
            # loader_token not an int index, fall back below
            pass

    # 4) Last resort: loader_token is already a NuScenes sample token
    nusc = pack.get('nusc', None)
    if isinstance(loader_token, str) and nusc is not None:
        try:
            _ = nusc.get('sample', loader_token)
            return loader_token
        except Exception:
            pass

    warnings.warn(
        f"[run_manual_benchmark] Could not resolve a valid NuScenes sample token "
        f"for loader_token={loader_token!r}. This sample will be skipped for "
        f"detection metrics, but still used for perf stats."
    )
    return None

def _resolve_sample_tokenv1(
    loader_token: Any,
    meta: Dict[str, Any],
    pack: Dict[str, Any],
) -> Union[str, None]:
    """
    Best-effort resolution of a NuScenes sample token.

    Priority:
      1) meta['token'] if present.
      2) meta['sample_idx'] + dataset.data_infos[idx]['token'].
      3) loader_token interpreted as integer index into dataset.data_infos.
      4) loader_token used directly as a NuScenes sample token (only if valid).

    If nothing works, returns None and the sample will be skipped for detection
    metrics (but still used for perf stats).
    """
    # 1) Direct token from metainfo
    tok = meta.get('token', None)
    if isinstance(tok, str) and len(tok) > 0:
        return tok

    ds = pack.get('dataset', None)

    # 2) Use sample_idx from meta to index into dataset.data_infos
    sample_idx = meta.get('sample_idx', None)
    if ds is not None and sample_idx is not None:
        try:
            idx = int(sample_idx)
            if hasattr(ds, 'data_infos') and 0 <= idx < len(ds.data_infos):
                info = ds.data_infos[idx]
                if isinstance(info.get('token', None), str):
                    return info['token']
        except Exception:
            pass

    # 3) Treat loader_token as dataset index
    if ds is not None and hasattr(ds, 'data_infos'):
        try:
            idx = int(loader_token)
            if 0 <= idx < len(ds.data_infos):
                info = ds.data_infos[idx]
                if isinstance(info.get('token', None), str):
                    return info['token']
        except Exception:
            # loader_token not an int index, fall back below
            pass

    # 4) Last resort: loader_token is already a NuScenes sample token
    nusc = pack.get('nusc', None)
    if isinstance(loader_token, str) and nusc is not None:
        try:
            _ = nusc.get('sample', loader_token)
            return loader_token
        except Exception:
            pass

    warnings.warn(
        f"[run_manual_benchmark] Could not resolve a valid NuScenes sample token "
        f"for loader_token={loader_token!r}. This sample will be skipped for "
        f"detection metrics, but still used for perf stats."
    )
    return None


def _maybe_save_multiview(
    dataset: str,
    out_dir: str,
    token: str,
    inputs: Dict[str, Any],
    meta: Dict[str, Any],
    pts_np: np.ndarray,
    pred_boxes_np: np.ndarray,
    gt_boxes_np: np.ndarray,
) -> None:
    """
    Save multiview images (lidar projection + GT/pred boxes) if possible.

    Works for both NuScenes and KITTI as long as:
      - inputs contains 'img' tensor: (N, 3, H, W)
      - meta contains 'lidar2img' and 'img_norm_cfg'
    """
    if 'img' not in inputs or inputs['img'] is None:
        return
    if not isinstance(inputs['img'][0], torch.Tensor):
        return
    if 'lidar2img' not in meta or 'img_norm_cfg' not in meta:
        return

    imgs_tensor = inputs['img'][0]        # (N, 3, H, W)
    l2i_list = np.asarray(meta['lidar2img'], dtype=np.float32)
    out_vis_dir = osp.join(out_dir, f"vis_{dataset.lower()}")
    os.makedirs(out_vis_dir, exist_ok=True)

    try:
        draw_2d_multiview_from_tensor(
            imgs_tensor=imgs_tensor,
            meta=meta,
            pts=pts_np[:, :3],         # use xyz only
            l2i_list=l2i_list,
            pred_boxes=pred_boxes_np,
            gt_boxes=gt_boxes_np,
            token=str(token),
            out_dir=out_vis_dir,
        )
    except Exception as e:
        warnings.warn(f"[run_manual_benchmark] draw_2d_multiview_from_tensor failed: {e}")


def run_manual_benchmark(
    model: torch.nn.Module,
    pack: Dict[str, Any],
    class_names: Sequence[str],
    out_dir: str,
    device: Union[str, torch.device],
    eval_set: str,
    detection_cfg_name: str,
    score_thresh: float,
    max_samples: int,
    sys_info: Dict[str, Any],
    dataset: str = "nuscenes",
) -> None:
    """
    Manual benchmark loop shared by training script and standalone inference.

    Parameters match the original inference call:

        run_manual_benchmark(
            model=model,
            pack=pack,
            class_names=cfg.class_names,
            out_dir=args.out_dir,
            device=args.device,
            eval_set=args.eval_set,
            detection_cfg_name="detection_cvpr_2019",
            score_thresh=args.score_thresh,
            max_samples=args.max_samples,
            sys_info=sys_info,
            dataset=args.dataset,
        )

    Supports:
      - NuScenes: full detection metrics via NuScenesEval + multiview images.
      - KITTI:    perf stats + per-class detection counts + multiview images.
                  (official KITTI AP is still best via Runner backend).
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # ----------------------------------------------------------------------
    # KITTI PATH: perf + per-class detection counts + visualization
    # ----------------------------------------------------------------------
    if dataset.lower() == "kitti":
        print("\n" + "=" * 60)
        print(" MANUAL BENCHMARK (KITTI)")
        print("  - Running inference + perf stats + per-class counts.")
        print("  - For official KITTI AP, use Runner-based evaluation.")
        print("=" * 60)

        loader = pack['loader']
        iter_fn = pack['iter_fn']

        metrics = []
        per_class = {name: 0 for name in class_names}

        pbar = tqdm(iter_fn(loader), desc="Inference (KITTI)", total=len(loader))
        num_done = 0

        for token, pts, imgs, meta, gt_boxes, gt_labels in pbar:
            if max_samples > 0 and num_done >= max_samples:
                break

            # Prepare inputs dict for test_step
            pts_tensor = _to_device(pts, device)
            inputs = {'points': [pts_tensor]}

            if imgs is not None:
                imgs_tensor = _to_device(imgs, device)
                inputs['img'] = [imgs_tensor]

            # Build data sample (meta + optional GT info)
            ds = Det3DDataSample()
            ds.set_metainfo(meta)

            gt_boxes_np = gt_boxes if gt_boxes is not None else None
            gt_labels_np = gt_labels if gt_labels is not None else None
            if gt_boxes_np is not None and gt_labels_np is not None:
                gt_inst = InstanceData()
                gt_inst.bboxes_3d = LiDARInstance3DBoxes(gt_boxes_np)
                gt_inst.labels_3d = torch.as_tensor(gt_labels_np, dtype=torch.long, device=device)
                ds.gt_instances_3d = gt_inst

            # Forward pass + timing
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                res = model.test_step(dict(inputs=inputs, data_samples=[ds]))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            metrics.append({'lat': dt, 'mem': max_mem})
            pbar.set_postfix(lat=f"{dt:.1f}ms")
            num_done += 1

            # Collect per-class detection counts
            pred = res[0].pred_instances_3d
            if hasattr(pred, 'scores_3d'):
                mask = pred.scores_3d > score_thresh
            else:
                mask = torch.ones_like(pred.labels_3d, dtype=torch.bool)
            det_labels = pred.labels_3d[mask].detach().cpu().numpy()
            for lab in det_labels:
                idx = int(lab)
                if 0 <= idx < len(class_names):
                    cname = class_names[idx]
                    per_class[cname] = per_class.get(cname, 0) + 1

            # Optional multiview visualization
            pts_np = pts_tensor.detach().cpu().numpy()
            pred_boxes_np = (
                pred.bboxes_3d.tensor[mask].detach().cpu().numpy()
                if hasattr(pred, 'bboxes_3d') else None
            )
            _maybe_save_multiview(
                dataset="kitti",
                out_dir=out_dir,
                token=str(token),
                inputs=inputs,
                meta=meta,
                pts_np=pts_np,
                pred_boxes_np=pred_boxes_np,
                gt_boxes_np=gt_boxes_np,
            )

        # Aggregate performance stats
        lats = np.array([m['lat'] for m in metrics], dtype=np.float32)
        mems = np.array([m['mem'] for m in metrics], dtype=np.float32) if metrics else np.array([0.0])

        perf = {
            "latency_mean": float(lats.mean()),
            "latency_std": float(lats.std()),
            "latency_min": float(lats.min()),
            "latency_max": float(lats.max()),
            "mem_peak": float(mems.max()),
            "samples": int(num_done),
            "score_thresh": float(score_thresh),
            "system_info": sys_info,
        }

        result_obj = {
            "dataset": "kitti",
            "eval_set": eval_set,
            "perf": perf,
            "per_class_detections": per_class,
        }

        save_path = osp.join(out_dir, "benchmark_kitti.json")
        with open(save_path, "w") as f:
            json.dump(result_obj, f, indent=4)

        # Print a nice summary similar to MMDet3D runner logs
        print("\n+------------+--------+")
        print("| {:10s} | {:6s} |".format("Class", "Count"))
        print("+------------+--------+")
        for cname in class_names:
            print("| {:10s} | {:6d} |".format(cname, per_class.get(cname, 0)))
        print("+------------+--------+\n")

        print("KITTI manual benchmark complete.")
        print(f"  Mean latency: {perf['latency_mean']:.2f} ms")
        print(f"  Peak memory:  {perf['mem_peak']:.2f} MB")
        print(f"  Samples:      {perf['samples']}")
        print(f"  Results saved to: {save_path}\n")
        return

    # ----------------------------------------------------------------------
    # NUSCENES PATH: full detection eval + multiview visualization
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STARTING MANUAL BENCHMARK (NuScenes)")
    print("=" * 60)

    loader = pack['loader']
    iter_fn = pack['iter_fn']
    nusc = pack['nusc']

    results_dict = {
        "meta": {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": {},
    }

    metrics = []
    processed_tokens = []

    res_path = osp.join(out_dir, "nuscenes_results.json")
    pbar = tqdm(iter_fn(loader), desc="Inference (NuScenes)", total=len(loader))
    num_done = 0

    for loader_token, pts, imgs, meta, gt_boxes, gt_labels in pbar:
        if max_samples > 0 and num_done >= max_samples:
            break

        # Resolve sample_token that matches NuScenes GT
        sample_token = _resolve_sample_token(loader_token, meta, pack)
        if sample_token is None:
            # Use perf stats only; skip this for detection JSON
            # but still run inference to measure latency/memory.
            sample_token_for_det = None
        else:
            sample_token_for_det = sample_token
            processed_tokens.append(sample_token)

        # Prepare model inputs
        pts_tensor = _to_device(pts, device)
        inputs = {'points': [pts_tensor]}

        if imgs is not None:
            imgs_tensor = _to_device(imgs, device)
            inputs['img'] = [imgs_tensor]

        ds = Det3DDataSample()
        ds.set_metainfo(meta)

        gt_boxes_np = gt_boxes if gt_boxes is not None else None
        gt_labels_np = gt_labels if gt_labels is not None else None
        if gt_boxes_np is not None and gt_labels_np is not None:
            gt_inst = InstanceData()
            gt_inst.bboxes_3d = LiDARInstance3DBoxes(gt_boxes_np)
            gt_inst.labels_3d = torch.as_tensor(gt_labels_np, dtype=torch.long, device=device)
            ds.gt_instances_3d = gt_inst

        # Forward pass + timing
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            res = model.test_step(dict(inputs=inputs, data_samples=[ds]))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) * 1000.0
        max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        metrics.append({'lat': dt, 'mem': max_mem})
        pbar.set_postfix(lat=f"{dt:.1f}ms")
        num_done += 1

        pred = res[0].pred_instances_3d
        if hasattr(pred, 'scores_3d'):
            mask = pred.scores_3d > score_thresh
        else:
            mask = torch.ones_like(pred.labels_3d, dtype=torch.bool)

        # Multi-view visualization (optional)
        pts_np = pts_tensor.detach().cpu().numpy()
        pred_boxes_np = (
            pred.bboxes_3d.tensor[mask].detach().cpu().numpy()
            if hasattr(pred, 'bboxes_3d') else None
        )
        _maybe_save_multiview(
            dataset="nuscenes",
            out_dir=out_dir,
            token=sample_token or str(loader_token),
            inputs=inputs,
            meta=meta,
            pts_np=pts_np,
            pred_boxes_np=pred_boxes_np,
            gt_boxes_np=gt_boxes_np,
        )

        # Only add to detection JSON if we have a valid sample_token
        if sample_token_for_det is not None:
            box = pred.bboxes_3d.tensor[mask].detach().cpu().numpy()
            sc = pred.scores_3d[mask].detach().cpu().numpy()
            lbl = pred.labels_3d[mask].detach().cpu().numpy()
            vels = (
                pred.velocities_3d[mask].detach().cpu().numpy()
                if hasattr(pred, 'velocities_3d') else None
            )
            if vels is None and box.shape[1] > 7:
                vels = box[:, 7:9]
            attrs = (
                pred.attr_labels[mask].detach().cpu().numpy()
                if hasattr(pred, 'attr_labels') else None
            )

            results_dict["results"][sample_token_for_det] = lidar_to_global_box(
                nusc, sample_token_for_det, box, sc, lbl, class_names, attrs, vels
            )

    with open(res_path, "w") as f:
        json.dump(results_dict, f)
    print(f"Results saved to {res_path}")

    # ---------------- NuScenes evaluation ----------------
    print("Running NuScenes Evaluator...")
    cfg_eval = config_factory(detection_cfg_name)
    nusc_eval = NuScenesEval(
        nusc,
        config=cfg_eval,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=osp.join(out_dir, "eval"),
        verbose=True,
    )

    # If max_samples < 0 -> evaluate on full split normally.
    # If max_samples > 0 -> restrict GT/preds to processed_tokens.
    if max_samples > 0:
        from nuscenes.eval.common.loaders import load_prediction, load_gt

        nusc_eval.gt_boxes = load_gt(
            nusc_eval.nusc, nusc_eval.eval_set, DetectionBox, verbose=True
        )
        nusc_eval.pred_boxes, _ = load_prediction(
            res_path, nusc_eval.cfg.max_boxes_per_sample, DetectionBox, verbose=True
        )

        filtered_tokens = set(processed_tokens)
        nusc_eval.gt_boxes.boxes = {
            k: v for k, v in nusc_eval.gt_boxes.boxes.items()
            if k in filtered_tokens
        }
        nusc_eval.sample_tokens = [t for t in nusc_eval.sample_tokens if t in filtered_tokens]

    metrics_summary, _ = nusc_eval.evaluate()
    print(
        f"\nNuScenes detection metrics: "
        f"NDS={metrics_summary.nd_score:.4f}, mAP={metrics_summary.mean_ap:.4f}"
    )

    # Save perf summary
    lats = np.array([m['lat'] for m in metrics], dtype=np.float32)
    mems = np.array([m['mem'] for m in metrics], dtype=np.float32) if metrics else np.array([0.0])

    perf = {
        "latency_mean": float(lats.mean()),
        "latency_std": float(lats.std()),
        "latency_min": float(lats.min()),
        "latency_max": float(lats.max()),
        "mem_peak": float(mems.max()),
        "samples": int(num_done),
        "score_thresh": float(score_thresh),
        "system_info": sys_info,
        "NDS": float(metrics_summary.nd_score),
        "mAP": float(metrics_summary.mean_ap),
    }

    with open(osp.join(out_dir, "benchmark_nuscenes.json"), "w") as f:
        json.dump(perf, f, indent=4)

    print(f"\nNuScenes manual benchmark complete. Results saved to {out_dir}\n")


# simple_infer_utils.py (or wherever run_manual_benchmark lives)
def run_manual_benchmark_v1(
    model: torch.nn.Module,
    pack: Dict[str, Any],
    class_names: Sequence[str],
    out_dir: str,
    device: torch.device,
    eval_set: str = "val",
    detection_cfg_name: str = "detection_cvpr_2019",
    score_thresh: float = 0.05,
    max_samples: int = -1,
    sys_info: Optional[Dict[str, Any]] = None,
    dataset: str = "nuscenes",
) -> None:
    """
    Manual benchmark loop used by simple_infer_main.py.

    This matches the *original* inference footprint:

        run_manual_benchmark(
            model=model,
            pack=pack,
            class_names=cfg.class_names,
            out_dir=args.out_dir,
            device=args.device,
            eval_set="val",
            detection_cfg_name="detection_cvpr_2019",
            score_thresh=0.05,
            max_samples=args.max_samples,
            sys_info=sys_info,
            dataset=args.dataset
        )

    Behavior
    --------
    KITTI:
      - Iterates over the loader sample-by-sample.
      - Runs model.test_step().
      - Records latency & GPU mem.
      - Saves per-class detection counts.
      - Saves multi-view images (lidar projection + 3D GT & pred boxes)
        using draw_2d_multiview_from_tensor.
      - Writes benchmark_perf_kitti.json and prints a summary table.

    NuScenes:
      - Iterates over the loader sample-by-sample.
      - Runs model.test_step().
      - Writes NuScenes-format JSON via lidar_to_global_box.
      - Runs NuScenesEval (NDS / mAP).
      - Records latency & GPU mem.
      - Saves multi-view images using draw_2d_multiview_from_tensor.
      - Writes benchmark_perf.json and prints summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    dataset = dataset.lower()

    # ----------------------------------------------------------------------
    # KITTI path
    # ----------------------------------------------------------------------
    if dataset == "kitti":
        print("\n" + "=" * 60)
        print(" MANUAL BENCHMARK (KITTI)")
        print("  - Running inference + perf stats + multi-view vis.")
        print("  - For official KITTI AP, use Runner-based evaluation.")
        print("=" * 60)

        loader = pack["loader"]
        iter_fn = pack["iter_fn"]

        metrics: List[Dict[str, float]] = []
        per_class_detections: Dict[str, int] = {c: 0 for c in class_names}

        vis_dir = osp.join(out_dir, "kitti_multiview")
        os.makedirs(vis_dir, exist_ok=True)

        pbar = tqdm(iter_fn(loader), desc="Inference (KITTI)", total=len(loader))
        for i, (token, pts, imgs, meta, gt_boxes, extra) in enumerate(pbar):
            if max_samples != -1 and i >= max_samples:
                break

            # --------------------------------------------------------------
            # Build mmdet3d-style inputs / data_samples
            # --------------------------------------------------------------
            inputs: Dict[str, Any] = {
                "points": [torch.from_numpy(pts).to(device)]
            }
            if imgs is not None:
                # imgs: (N_views, 3, H, W) tensor from the dataloader
                inputs["img"] = [imgs.to(device)]

            ds = Det3DDataSample()
            ds.set_metainfo(meta)

            # --------------------------------------------------------------
            # Inference + perf stats
            # --------------------------------------------------------------
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                res = model.test_step(dict(inputs=inputs, data_samples=[ds]))
            torch.cuda.synchronize(device)

            dt = (time.perf_counter() - t0) * 1000.0  # ms
            max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats(device)

            metrics.append({"lat": dt, "mem": max_mem})
            pbar.set_postfix(lat=f"{dt:.1f}ms")

            # --------------------------------------------------------------
            # Decode predictions for stats + visualization
            # --------------------------------------------------------------
            pred_sample: Det3DDataSample = res[0]
            pred = pred_sample.pred_instances_3d

            # Per-class detection counts above score_thresh
            if pred is not None and pred.scores_3d.numel() > 0:
                keep = pred.scores_3d > score_thresh
                labels_np = pred.labels_3d[keep].cpu().numpy()
                for cid in labels_np.tolist():
                    if 0 <= cid < len(class_names):
                        per_class_detections[class_names[cid]] += 1

            # Multi-view visualization (KITTI)
            if imgs is not None and "lidar2img" in meta:
                l2i_list = np.asarray(meta["lidar2img"], dtype=np.float32)
                pred_boxes_np = None
                if pred is not None and pred.bboxes_3d is not None:
                    if pred.scores_3d.numel() > 0:
                        keep = pred.scores_3d > score_thresh
                        pred_boxes_np = (
                            pred.bboxes_3d.tensor[keep].detach().cpu().numpy()
                        )

                gt_boxes_np = None
                if gt_boxes is not None:
                    # whatever gt_boxes format your iter_fn uses; most likely
                    # (G, 7) in LiDAR frame already
                    gt_boxes_np = gt_boxes

                draw_2d_multiview_from_tensor(
                    imgs_tensor=imgs,
                    meta=meta,
                    pts=pts,
                    l2i_list=l2i_list,
                    pred_boxes=pred_boxes_np,
                    gt_boxes=gt_boxes_np,
                    token=str(token),
                    out_dir=vis_dir,
                )

        # ------------------------------------------------------------------
        # Aggregate metrics & save JSON
        # ------------------------------------------------------------------
        if len(metrics) > 0:
            lat_vals = np.array([m["lat"] for m in metrics], dtype=np.float32)
            mem_vals = np.array([m["mem"] for m in metrics], dtype=np.float32)
        else:
            lat_vals = np.array([0.0], dtype=np.float32)
            mem_vals = np.array([0.0], dtype=np.float32)

        perf = {
            "latency_mean": float(lat_vals.mean()),
            "latency_std": float(lat_vals.std()),
            "latency_min": float(lat_vals.min()),
            "latency_max": float(lat_vals.max()),
            "mem_peak": float(mem_vals.max()),
            "samples": len(metrics),
            "score_thresh": float(score_thresh),
            "system_info": sys_info or {},
        }

        result_dict = {
            "dataset": "kitti",
            "eval_set": eval_set,
            "perf": perf,
            "per_class_detections": per_class_detections,
        }

        out_json = osp.join(out_dir, "benchmark_perf_kitti.json")
        with open(out_json, "w") as f:
            json.dump(result_dict, f, indent=4)

        # Pretty print like mmdetection3d runner
        print("\n+------------+--------+")
        for cname, cnt in per_class_detections.items():
            print(f"| {cname:<10} | {cnt:6d} |")
        print("+------------+--------+\n")

        print("KITTI manual benchmark complete.")
        print(f"  Mean latency: {perf['latency_mean']:.2f} ms")
        print(f"  Peak memory:  {perf['mem_peak']:.2f} MB")
        print(f"  Samples:      {perf['samples']}")
        print(f"  Multi-view images: saved under {vis_dir}\n")
        return

    # ----------------------------------------------------------------------
    # NuScenes path
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STARTING MANUAL BENCHMARK (NuScenes)")
    print("  - Inference + NuScenesEval + multi-view vis")
    print("=" * 60)

    res_path = osp.join(out_dir, "nuscenes_results.json")
    vis_dir = osp.join(out_dir, "nuscenes_multiview")
    os.makedirs(vis_dir, exist_ok=True)

    results_dict: Dict[str, Any] = {
        "meta": {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": {},
    }

    metrics: List[Dict[str, float]] = []
    processed_tokens: List[str] = []

    loader = pack["loader"]
    iter_fn = pack["iter_fn"]
    nusc = pack["nusc"]

    pbar = tqdm(iter_fn(loader), desc="Inference (NuScenes)", total=len(loader))
    for i, (token, pts, imgs, meta, gt_boxes, extra) in enumerate(pbar):
        if max_samples != -1 and i >= max_samples:
            break

        sample_token = meta.get("token", token)
        processed_tokens.append(sample_token)

        inputs: Dict[str, Any] = {
            "points": [torch.from_numpy(pts).to(device)]
        }
        if imgs is not None:
            inputs["img"] = [imgs.to(device)]

        ds = Det3DDataSample()
        ds.set_metainfo(meta)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            res = model.test_step(dict(inputs=inputs, data_samples=[ds]))
        torch.cuda.synchronize(device)

        dt = (time.perf_counter() - t0) * 1000.0
        max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)
        metrics.append({"lat": dt, "mem": max_mem})
        pbar.set_postfix(lat=f"{dt:.1f}ms")

        pred_sample: Det3DDataSample = res[0]
        pred = pred_sample.pred_instances_3d

        if pred is not None and pred.scores_3d.numel() > 0:
            mask = pred.scores_3d > score_thresh
            box = pred.bboxes_3d.tensor[mask].detach().cpu().numpy()
            sc = pred.scores_3d[mask].detach().cpu().numpy()
            lbl = pred.labels_3d[mask].detach().cpu().numpy()
            vels = (
                pred.velocities_3d[mask].detach().cpu().numpy()
                if hasattr(pred, "velocities_3d")
                else None
            )
            if vels is None and box.shape[1] > 7:
                vels = box[:, 7:9]
            attrs = (
                pred.attr_labels[mask].detach().cpu().numpy()
                if hasattr(pred, "attr_labels")
                else None
            )
        else:
            box = np.zeros((0, 7), dtype=np.float32)
            sc = np.zeros((0,), dtype=np.float32)
            lbl = np.zeros((0,), dtype=np.int64)
            vels = None
            attrs = None

        results_dict["results"][sample_token] = lidar_to_global_box(
            nusc, sample_token, box, sc, lbl, class_names, attrs, vels
        )

        # Multi-view visualization (NuScenes)
        if imgs is not None and "lidar2img" in meta:
            l2i_list = np.asarray(meta["lidar2img"], dtype=np.float32)
            pred_boxes_np = box
            gt_boxes_np = gt_boxes  # expected (G, 7) in LiDAR frame, if provided

            draw_2d_multiview_from_tensor(
                imgs_tensor=imgs,
                meta=meta,
                pts=pts,
                l2i_list=l2i_list,
                pred_boxes=pred_boxes_np,
                gt_boxes=gt_boxes_np,
                token=str(sample_token),
                out_dir=vis_dir,
            )

    # Save detection results
    with open(res_path, "w") as f:
        json.dump(results_dict, f)
    print(f"Detection results saved to {res_path}")

    print("Running NuScenes Evaluator...")
    cfg_eval = config_factory(detection_cfg_name)
    nusc_eval = NuScenesEval(
        nusc,
        config=cfg_eval,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=osp.join(out_dir, "eval"),
        verbose=True,
    )

    if max_samples != -1:
        from nuscenes.eval.common.loaders import load_prediction, load_gt

        nusc_eval.gt_boxes = load_gt(
            nusc_eval.nusc, nusc_eval.eval_set, DetectionBox, verbose=True
        )
        nusc_eval.pred_boxes, _ = load_prediction(
            res_path, nusc_eval.cfg.max_boxes_per_sample, DetectionBox, verbose=True
        )
        nusc_eval.gt_boxes.boxes = {
            k: v for k, v in nusc_eval.gt_boxes.boxes.items()
            if k in processed_tokens
        }
        nusc_eval.sample_tokens = processed_tokens

    metrics_summary, _ = nusc_eval.evaluate()
    print(f"\nNDS: {metrics_summary.nd_score:.4f} | mAP: {metrics_summary.mean_ap:.4f}")

    # Perf stats JSON (NuScenes)
    if len(metrics) > 0:
        lat_vals = np.array([m["lat"] for m in metrics], dtype=np.float32)
        mem_vals = np.array([m["mem"] for m in metrics], dtype=np.float32)
    else:
        lat_vals = np.array([0.0], dtype=np.float32)
        mem_vals = np.array([0.0], dtype=np.float32)

    perf = {
        "latency_mean": float(lat_vals.mean()),
        "latency_std": float(lat_vals.std()),
        "latency_min": float(lat_vals.min()),
        "latency_max": float(lat_vals.max()),
        "mem_peak": float(mem_vals.max()),
    }
    with open(osp.join(out_dir, "benchmark_perf.json"), "w") as f:
        json.dump(perf, f, indent=4)

    print(f"\nMulti-view images saved under: {vis_dir}")

def inference_loop(model: torch.nn.Module,
                   pack: Dict[str, Any],
                   out_dir: str,
                   device: str = "cuda",
                   score_thresh: float = DEFAULT_SCORE_THRESH_VIZ,
                   metrics: Optional[Dict[str, Any]] = None,
                   save_images: bool = True,
                   save_ply_if_headless: bool = True,
                   show_open3d: bool = True,
                   max_samples: int = -1) -> Dict[str, Any]:
    """
    Visual inference loop for qualitative inspection.

    - Runs inference sample-by-sample.
    - Optionally saves multi-view 2D images and 3D PLYs / Open3D windows.
    - Records per-sample latency and peak memory.

    Args:
        model:       eval-mode model
        pack:        dict(loader, iter_fn, nusc)
        out_dir:     where to write visualization outputs
        device:      device string ("cuda" / "cpu")
        score_thresh: filtering for visualization
        metrics:     existing metrics dict to append, or None to create new
        save_images: if True, save multiview JPGs
        save_ply_if_headless: if True, save PLY when DISPLAY not set
        show_open3d: if True and DISPLAY set, open interactive viewer
        max_samples: if > 0, stop after processing this many samples
                     (useful with cfg-based datasets that have many samples)

    Returns:
        metrics dict with per-sample entries.
    """
    print("Starting Visual Inference Loop...")
    os.makedirs(out_dir, exist_ok=True)

    device_obj = torch.device(device)

    loader = pack['loader']
    iter_fn = pack['iter_fn']
    nusc = pack['nusc']

    if metrics is None:
        metrics = {"samples": []}

    # Respect max_samples for any data_source (cfg or custom)
    if max_samples is not None and max_samples > 0:
        total = min(len(loader), max_samples)
    else:
        total = len(loader)

    pbar = tqdm(enumerate(iter_fn(loader)), desc="Visualizing", total=total)
    for idx, (token, pts, imgs, meta, paths, _) in pbar:
        if max_samples is not None and max_samples > 0 and idx >= max_samples:
            break
        pts_t = torch.from_numpy(pts).float().to(device_obj)
        inputs = dict(points=[pts_t])

        if imgs is not None:
            inputs['img'] = [imgs.to(device_obj)]

        ds = Det3DDataSample()
        ds.set_metainfo(meta)

        torch.cuda.synchronize(device_obj)
        torch.cuda.reset_peak_memory_stats(device_obj)
        t0 = time.perf_counter()

        with torch.no_grad():
            res = model.test_step(dict(inputs=inputs, data_samples=[ds]))

        torch.cuda.synchronize(device_obj)
        dt = (time.perf_counter() - t0) * 1000.0
        max_mem = torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)

        pred = res[0].pred_instances_3d
        scores = pred.scores_3d.detach().cpu().numpy()
        boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()

        keep = scores > score_thresh
        pred_boxes = boxes[keep, :7] if keep.any() else None
        max_conf = float(scores.max()) if scores.size > 0 else 0.0

        # Use true NuScenes sample_token (if available) for GT lookup
        sample_token = meta.get('token', None)
        if nusc is not None and sample_token is not None:
            gt_boxes = get_gt_boxes(nusc, sample_token)
        else:
            gt_boxes = None

        metrics["samples"].append({
            'id': str(token),
            'latency_ms': dt,
            'peak_memory_mb': max_mem,
            'max_conf': max_conf
        })
        pbar.set_postfix({"Lat": f"{dt:.1f}ms"})

        # if save_images and paths:
        #     draw_2d_multiview(
        #         paths, pts, meta['lidar2img'], pred_boxes, gt_boxes,
        #         str(token), out_dir
        #     )
        if save_images:
            if 'img_norm_cfg' in meta and torch.is_tensor(imgs):
                # cfg-based dataloader: use tensor + meta directly
                draw_2d_multiview_from_tensor(
                    imgs_tensor=imgs,
                    meta=meta,
                    pts=pts,
                    l2i_list=meta['lidar2img'],
                    pred_boxes=pred_boxes,
                    gt_boxes=gt_boxes,
                    token=str(token),
                    out_dir=out_dir,
                )
            elif paths:
                # custom loader path: fall back to disk-based drawer
                draw_2d_multiview(
                    paths, pts, meta['lidar2img'],
                    pred_boxes, gt_boxes, str(token), out_dir
                )

        if HAS_OPEN3D and (save_ply_if_headless or show_open3d):
            headless = os.environ.get('DISPLAY') is None
            if headless and save_ply_if_headless:
                save_ply_files(out_dir, str(token), pts, pred_boxes, gt_boxes)
            elif (not headless) and show_open3d:
                run_open3d_viz(pts, pred_boxes, gt_boxes, window_name=f"Sample {token}")
                
    return metrics