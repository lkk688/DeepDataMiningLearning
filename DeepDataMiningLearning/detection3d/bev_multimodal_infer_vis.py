#!/usr/bin/env python3
"""
Unified BEV Multimodal Inference & Visualization Script
- Supports BEVFusion and TransFusion models from MMDetection3D
- Works with nuScenes and Waymo Open Dataset (v1/v2 converted for MMDet3D)
- Provides multimodal visualization: camera overlays, BEV 3D boxes, occupancy heatmap

Usage examples
--------------
# nuScenes + BEVFusion
python bev_multimodal_infer_vis.py \
  --model bevfusion \
  --config configs/bevfusion/bevfusion_lidar_camera_512x1408.py \
  --ckpt   work_dirs/bevfusion/epoch_24.pth \
  --dataset nuscenes \
  --data-root data/nuscenes \
  --save-dir out_vis --num-samples 8

# Waymo + TransFusion
python bev_multimodal_infer_vis.py \
  --model transfusion \
  --config configs/transfusion/transfusion_lidar_camera_704x256.py \
  --ckpt   work_dirs/transfusion/epoch_24.pth \
  --dataset waymo \
  --data-root data/waymo \
  --save-dir out_vis --num-samples 8

Notes
-----
1) Assumes you have prepared datasets following MMDetection3D's conventions.
2) Occupancy visualization will be drawn when the model outputs occupancy logits (e.g., BEVFusion variants with occupancy head). Otherwise a fallback rasterization from boxes is used for demo.
3) Camera overlay needs per-view lidar2img matrices provided by the dataset's pipeline (commonly under data_sample.metainfo["lidar2img"] or "ori_lidar2img").
"""
from __future__ import annotations
import os
import argparse
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from mmcv import Config
from mmengine.config import ConfigDict
from mmdet3d.apis import init_model
from mmdet3d.datasets import build_dataset
from mmdet3d.structures import LiDARInstance3DBoxes

# Optional Open3D (interactive 3D view)
try:
    import open3d as o3d
except Exception:
    o3d = None

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def tensor_to_numpy(x):
    if x is None:
        return None
    if hasattr(x, 'to'):
        x = x.detach().cpu()
    return np.array(x)


def get_box_corners_3d(box: np.ndarray) -> np.ndarray:
    """Return 8 corners (in lidar coordinate) for box = [x,y,z,dx,dy,dz,yaw].
    Convention: dx along x, dy along y, dz height; yaw around z.
    Output shape: (8, 3)
    """
    x, y, z, dx, dy, dz, yaw = box.tolist()
    # local corners around origin
    corners = np.array([
        [ dx/2,  dy/2,  dz/2],
        [ dx/2, -dy/2,  dz/2],
        [-dx/2, -dy/2,  dz/2],
        [-dx/2,  dy/2,  dz/2],
        [ dx/2,  dy/2, -dz/2],
        [ dx/2, -dy/2, -dz/2],
        [-dx/2, -dy/2, -dz/2],
        [-dx/2,  dy/2, -dz/2],
    ])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = corners @ R.T
    corners = corners + np.array([x, y, z])
    return corners


def project_lidar_to_img(pts_lidar: np.ndarray, lidar2img: np.ndarray) -> np.ndarray:
    """Project lidar xyz (N,3) to image plane using 4x4 matrix (lidar2img)."""
    pts_h = np.concatenate([pts_lidar, np.ones((pts_lidar.shape[0], 1))], axis=1)  # (N,4)
    pts_img = pts_h @ lidar2img.T  # (N,4)
    # avoid division by zero
    eps = 1e-6
    pts_img[:, 0] /= np.maximum(pts_img[:, 2], eps)
    pts_img[:, 1] /= np.maximum(pts_img[:, 2], eps)
    return pts_img[:, :3]  # u, v, depth


def draw_3d_box_on_image(img: np.ndarray, corners: np.ndarray, color=(0, 255, 0), thickness=2):
    """corners are in image pixel coords (8,2)."""
    corners = corners.astype(int)
    # 12 edges of a cuboid
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # top
        (4,5),(5,6),(6,7),(7,4),  # bottom
        (0,4),(1,5),(2,6),(3,7)   # verticals
    ]
    for i,j in edges:
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness, lineType=cv2.LINE_AA)


def visualize_camera_overlays(data_sample, result, save_dir: Path):
    """Overlay 3D boxes on each camera image using lidar2img matrices.
    data_sample: Det3DDataSample
    result: same-structured prediction for that sample
    """
    metas = data_sample.metainfo
    imgs = metas.get('img', None)
    img_paths = metas.get('img_path', None)
    lidar2img = metas.get('lidar2img', metas.get('ori_lidar2img', None))

    if img_paths is None and imgs is None:
        return

    # Unpack predictions
    pred = result.pred_instances_3d
    if pred is None or pred.bboxes_3d is None:
        return
    bboxes3d: LiDARInstance3DBoxes = pred.bboxes_3d
    boxes_np = bboxes3d.tensor.detach().cpu().numpy()

    num_views = len(img_paths) if img_paths is not None else len(imgs)
    for vid in range(num_views):
        if img_paths is not None:
            img = cv2.imread(img_paths[vid])
        else:
            # imgs could be a Tensor; convert to HxWxC BGR for OpenCV
            img_t = data_sample.inputs['img'][vid]
            img = img_t.detach().cpu().numpy().transpose(1,2,0)[:, :, ::-1]
        if img is None:
            continue
        L2I = np.array(lidar2img[vid])
        H, W = img.shape[:2]
        for box in boxes_np:
            corners_lidar = get_box_corners_3d(box)
            corners_img3d = project_lidar_to_img(corners_lidar, L2I)
            depth = corners_img3d[:, 2]
            # Only draw if at least 4 corners are in front of camera
            if (depth > 0).sum() >= 4:
                uv = corners_img3d[:, :2]
                # clip inside
                if (np.isfinite(uv).all()):
                    draw_3d_box_on_image(img, uv)
        out_path = save_dir / f"cam{vid}.jpg"
        cv2.imwrite(str(out_path), img)


def rasterize_boxes_to_bev(boxes: np.ndarray, bev_shape=(800, 800), scale: float = 2.0, color=(0,255,0)) -> np.ndarray:
    """Simple BEV raster (x,y) assuming forward x, left y.
    scale: pixels per meter. Center of BEV at image center.
    """
    H, W = bev_shape
    bev = np.zeros((H, W, 3), dtype=np.uint8)
    cx, cy = W//2, H//2
    for b in boxes:
        x, y, z, dx, dy, dz, yaw = b
        c, s = math.cos(yaw), math.sin(yaw)
        rect = np.array([
            [ dx/2,  dy/2],
            [ dx/2, -dy/2],
            [-dx/2, -dy/2],
            [-dx/2,  dy/2],
        ])
        R = np.array([[c, -s],[s, c]])
        rect = rect @ R.T + np.array([x, y])
        pts = np.round(rect * scale + np.array([cx, cy])).astype(np.int32)
        cv2.polylines(bev, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    return bev


def visualize_bev_and_occupancy(result, save_dir: Path, bev_shape=(800,800), scale=2.0):
    save_dir.mkdir(parents=True, exist_ok=True)

    pred = result.pred_instances_3d
    boxes_np = None
    if pred is not None and pred.bboxes_3d is not None:
        boxes_np = pred.bboxes_3d.tensor.detach().cpu().numpy()

    # Draw boxes BEV
    bev_img = rasterize_boxes_to_bev(boxes_np if boxes_np is not None else np.zeros((0,7)), bev_shape, scale)

    # Occupancy heatmap (prefer model output if available)
    occ = None
    if hasattr(result, 'pred_pts_seg') and result.pred_pts_seg is not None:
        occ = result.pred_pts_seg.pts_semantic_mask
    if hasattr(result, 'pred_occupancy'):
        occ = result.pred_occupancy
    if occ is not None:
        # Expect occ as (Hb, Wb) or logits; convert to color map
        occ_np = tensor_to_numpy(occ)
        if occ_np.ndim == 3 and occ_np.shape[-1] > 1:
            occ_np = occ_np.argmax(-1)
        occ_norm = (occ_np - occ_np.min()) / (occ_np.ptp() + 1e-6)
        occ_color = (plt.cm.inferno(occ_norm)[..., :3] * 255).astype(np.uint8)
        occ_color = cv2.resize(occ_color, (bev_shape[1], bev_shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(save_dir / 'bev_occupancy.png'), occ_color)
    else:
        # Fallback: box density heatmap
        heat = np.zeros((bev_shape[0], bev_shape[1]), dtype=np.float32)
        if boxes_np is not None:
            cx, cy = bev_shape[1]//2, bev_shape[0]//2
            for b in boxes_np:
                x, y = b[0], b[1]
                px, py = int(x*scale + cx), int(y*scale + cy)
                if 0<=px<bev_shape[1] and 0<=py<bev_shape[0]:
                    heat[py, px] = min(heat[py, px] + 1.0, 5.0)
        heat = heat / (heat.max() + 1e-6)
        occ_color = (plt.cm.magma(heat)[..., :3]*255).astype(np.uint8)
        cv2.imwrite(str(save_dir / 'bev_occupancy_fallback.png'), occ_color)

    cv2.imwrite(str(save_dir / 'bev_boxes.png'), bev_img)


def _boxes_to_o3d_linesets(boxes: np.ndarray, color=(0,1,0)):
    """Create Open3D LineSet(s) for given boxes array [N,7]."""
    if o3d is None or boxes is None or len(boxes) == 0:
        return []
    line_indices = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ]
    linesets = []
    for b in boxes:
        corners = get_box_corners_3d(b)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines  = o3d.utility.Vector2iVector(np.array(line_indices, dtype=np.int32))
        col = np.tile(np.array(color, dtype=np.float64)[None, :], (len(line_indices), 1))
        ls.colors = o3d.utility.Vector3dVector(col)
        linesets.append(ls)
    return linesets


def _load_lidar_points_from_meta(metas: Dict[str,Any], max_points: int = 200000) -> Optional[np.ndarray]:
    """Try to load lidar points from dataset metainfo. Supports nuScenes/Waymo default converters.
    Returns (N,3) or (N,4/5) if available; subsampled to max_points.
    """
    # Preferred: lidar path (nuScenes .bin, Waymo .bin)
    lidar_path = metas.get('lidar_path', None)
    if isinstance(lidar_path, (list, tuple)):
        lidar_path = lidar_path[0]
    pts = None
    if lidar_path and os.path.isfile(lidar_path):
        arr = np.fromfile(lidar_path, dtype=np.float32)
        # Guess dim: prefer 5 if divisible, else 4
        dim = 5 if (arr.size % 5 == 0) else 4
        pts = arr.reshape(-1, dim)
    else:
        # Sometimes points are attached in data_sample.inputs or metas
        if 'points' in metas:
            pts = metas['points']  # may be Tensor
            pts = tensor_to_numpy(pts)
        elif hasattr(metas, 'inputs') and 'points' in metas.inputs:
            pts = metas.inputs['points']
            pts = tensor_to_numpy(pts)
    if pts is None:
        return None
    if max_points and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    return pts


def visualize_open3d_scene(data_sample, result, max_points: int = 200000):
    """Interactive Open3D view: point cloud + predicted boxes.
    - Loads lidar from meta path (bin) or attached tensor.
    - Builds point cloud with intensity-based grayscale color if available.
    - Adds 3D boxes as LineSets.
    """
    if o3d is None:
        print('[Warn] Open3D is not installed. Skipping 3D visualization.')
        return
    metas = data_sample.metainfo
    pts = _load_lidar_points_from_meta(metas, max_points=max_points)
    if pts is None:
        print('[Warn] No lidar points found in meta; skip Open3D.')
        return

    # Prepare point cloud geometry
    if pts.shape[1] >= 3:
        xyz = pts[:, :3]
    else:
        print('[Warn] Lidar points malformed (dim < 3)')
        return
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    if pts.shape[1] >= 4:
        inten = pts[:, 3]
        inten = (inten - inten.min()) / (inten.ptp() + 1e-6)
        colors = np.stack([inten, inten, inten], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    geoms = [pcd]

    # Boxes
    pred = result.pred_instances_3d
    if pred is not None and pred.bboxes_3d is not None:
        boxes_np = pred.bboxes_3d.tensor.detach().cpu().numpy()
        linesets = _boxes_to_o3d_linesets(boxes_np, color=(0,1,0))
        geoms.extend(linesets)

    o3d.visualization.draw_geometries(geoms)

    pred = result.pred_instances_3d
    boxes_np = None
    if pred is not None and pred.bboxes_3d is not None:
        boxes_np = pred.bboxes_3d.tensor.detach().cpu().numpy()

    # Draw boxes BEV
    bev_img = rasterize_boxes_to_bev(boxes_np if boxes_np is not None else np.zeros((0,7)), bev_shape, scale)

    # Occupancy heatmap (prefer model output if available)
    occ = None
    if hasattr(result, 'pred_pts_seg') and result.pred_pts_seg is not None:
        occ = result.pred_pts_seg.pts_semantic_mask
    if hasattr(result, 'pred_occupancy'):
        occ = result.pred_occupancy
    if occ is not None:
        # Expect occ as (Hb, Wb) or logits; convert to color map
        occ_np = tensor_to_numpy(occ)
        if occ_np.ndim == 3 and occ_np.shape[-1] > 1:
            occ_np = occ_np.argmax(-1)
        occ_norm = (occ_np - occ_np.min()) / (occ_np.ptp() + 1e-6)
        occ_color = (plt.cm.inferno(occ_norm)[..., :3] * 255).astype(np.uint8)
        occ_color = cv2.resize(occ_color, (bev_shape[1], bev_shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(save_dir / 'bev_occupancy.png'), occ_color)
    else:
        # Fallback: box density heatmap
        heat = np.zeros((bev_shape[0], bev_shape[1]), dtype=np.float32)
        if boxes_np is not None:
            cx, cy = bev_shape[1]//2, bev_shape[0]//2
            for b in boxes_np:
                x, y = b[0], b[1]
                px, py = int(x*scale + cx), int(y*scale + cy)
                if 0<=px<bev_shape[1] and 0<=py<bev_shape[0]:
                    heat[py, px] = min(heat[py, px] + 1.0, 5.0)
        heat = heat / (heat.max() + 1e-6)
        occ_color = (plt.cm.magma(heat)[..., :3]*255).astype(np.uint8)
        cv2.imwrite(str(save_dir / 'bev_occupancy_fallback.png'), occ_color)

    cv2.imwrite(str(save_dir / 'bev_boxes.png'), bev_img)


# -----------------------------
# Dataset & Model Wrappers
# -----------------------------

class MMDet3DRunner:
    def __init__(self, cfg_path: str, ckpt_path: Optional[str], device='cuda:0', data_root: Optional[str]=None, test_mode=True):
        self.cfg = Config.fromfile(cfg_path)
        if data_root is not None:
            # try to override data_root & pipelines for test
            if 'data_root' in self.cfg:
                self.cfg.data_root = data_root
            if 'test_dataloader' in self.cfg and 'dataset' in self.cfg.test_dataloader:
                self.cfg.test_dataloader.dataset.data_root = data_root
        self.model = init_model(self.cfg, ckpt_path, device=device)
        self.model.eval()
        # Build dataset (test)
        if hasattr(self.cfg, 'test_dataloader'):
            self.dataset = build_dataset(self.cfg.test_dataloader.dataset)
        else:
            # legacy
            self.dataset = build_dataset(self.cfg.data.test)

    def num_samples(self):
        return len(self.dataset)

    @torch.no_grad()
    def infer_index(self, idx: int):
        sample = self.dataset[idx]
        # MMDet3D model expects list of data dicts
        data_list = [sample]
        data = self.model.data_preprocessor(data_list, training=False)
        results = self.model.test_step(data)
        # results is a list with one Det3DDataSample
        return sample, results[0]


# -----------------------------
# Main
# -----------------------------

MODELS = {
    'bevfusion': 'configs/bevfusion/bevfusion_lidar_camera_512x1408.py',
    'transfusion': 'configs/transfusion/transfusion_lidar_camera_704x256.py',
}


def parse_args():
    ap = argparse.ArgumentParser(description='Unified BEV Multimodal Inference & Visualization')
    ap.add_argument('--model', type=str, default='bevfusion', choices=list(MODELS.keys()))
    ap.add_argument('--config', type=str, default=None, help='path to config; overrides default by --model')
    ap.add_argument('--ckpt', type=str, default=None, help='checkpoint path (.pth)')
    ap.add_argument('--dataset', type=str, default='nuscenes', choices=['nuscenes','waymo'])
    ap.add_argument('--data-root', type=str, default=None)
    ap.add_argument('--save-dir', type=str, default='out_vis')
    ap.add_argument('--num-samples', type=int, default=4)
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--open3d', action='store_true', help='enable interactive Open3D visualization')
    ap.add_argument('--max-points', type=int, default=200000, help='max points to visualize in Open3D')
    return ap.parse_args()


def main():
    args = parse_args()

    cfg_path = args.config if args.config else MODELS[args.model]
    save_root = Path(args.save_dir)
    ensure_dir(save_root)

    runner = MMDet3DRunner(cfg_path, args.ckpt, device='cuda:0' if torch.cuda.is_available() else 'cpu', data_root=args.data_root)

    n = min(args.num_samples, runner.num_samples() - args.start_index)
    print(f"Loaded dataset with {runner.num_samples()} samples; visualizing {n} samples starting at {args.start_index}...")

    for i in range(args.start_index, args.start_index + n):
        data_sample, pred = runner.infer_index(i)
        case_dir = save_root / f"case_{i:06d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        # Per-camera overlays
        try:
            visualize_camera_overlays(data_sample, pred, case_dir / 'cams')
        except Exception as e:
            print(f"[Warn] camera overlay failed for sample {i}: {e}")

        # BEV boxes + occupancy
        try:
            visualize_bev_and_occupancy(pred, case_dir / 'bev')
        except Exception as e:
            print(f"[Warn] BEV visualization failed for sample {i}: {e}")

        # Open3D interactive 3D view
        if args.open3d:
            try:
                visualize_open3d_scene(data_sample, pred, max_points=args.max_points)
            except Exception as e:
                print(f"[Warn] Open3D visualization failed for sample {i}: {e}")

        print(f"Saved visualizations to: {case_dir}")


if __name__ == '__main__':
    main()
