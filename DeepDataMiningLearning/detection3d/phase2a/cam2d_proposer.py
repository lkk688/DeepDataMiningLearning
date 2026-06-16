"""
Validator B — 2D camera detector + LiDAR-depth 3D lift.

Uses torchvision's COCO-pretrained Faster R-CNN (no extra deps) to detect
2D boxes on each surround camera, then "lifts" each 2D box to a coarse
3D proposal using LiDAR depth inside the box.

Pipeline:
  1. Faster R-CNN forward → list of (xyxy_2d, coco_class_id, score)
  2. Map COCO class → transfer class {Vehicle, Pedestrian, Cyclist}.
     COCO IDs: 1=person, 2=bicycle, 3=car, 4=motorcycle, 6=bus, 8=truck
  3. For each 2D box:
       a. Project LiDAR (vehicle frame) to image via lidar2img.
       b. Take points inside the box; reject if <3 points (too sparse to
          trust depth).
       c. depth = median over inside-box LiDAR depths (in camera frame).
       d. 3D center = back-project (u_c, v_c, depth) using lidar2img^-1.
       e. 3D box size: a class-conditional mean prior (cheap).
       f. yaw: best heuristic = head toward / away from ego if vehicle.

This is "Validator B" — the camera-side validator. Knows semantic class,
but depth comes from LiDAR (so this validator implicitly requires LiDAR).
Where LiDAR is dense, B's 3D box is accurate; where LiDAR is sparse, the
depth is noisier — that's exactly where A is also weak, so we degrade
gracefully.

Returns proposals in the same schema as cluster_proposer.ClusterProposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch

from .cluster_proposer import ClusterProposal, TRANSFER_CLASSES


# COCO id → transfer class
COCO_TO_TRANSFER = {
    1: 'Pedestrian',   # person
    2: 'Cyclist',      # bicycle
    3: 'Vehicle',      # car
    4: 'Cyclist',      # motorcycle
    6: 'Vehicle',      # bus
    8: 'Vehicle',      # truck
}

# Class-conditional mean size priors (l, w, h) in meters.
# Used when lifting a 2D box: we know depth and class, so we synthesize a
# physically-plausible 3D box of average size and let downstream A∩B
# fusion decide whether to keep it.
SIZE_PRIOR_MEAN = {
    'Vehicle':    (4.5, 1.9, 1.7),
    'Pedestrian': (0.6, 0.6, 1.75),
    'Cyclist':    (1.7, 0.6, 1.6),
}


@dataclass
class Cam2DProposal:
    box: np.ndarray         # (7,) [x, y, z, l, w, h, yaw] in vehicle frame
    cls: str
    score: float            # 2D detector confidence
    cam_slot: int           # which of 6 cameras the 2D box came from
    depth_med: float        # median LiDAR depth inside the 2D box
    n_lidar_inside: int     # how many LiDAR points supported the depth
    box_2d: np.ndarray      # (4,) xyxy in original image space

    def to_dict(self) -> Dict:
        return dict(box=self.box.tolist(), cls=self.cls,
                    score=float(self.score), cam_slot=int(self.cam_slot),
                    depth_med=float(self.depth_med),
                    n_lidar_inside=int(self.n_lidar_inside),
                    box_2d=self.box_2d.tolist())


class Cam2DProposer:
    """Wraps torchvision Faster R-CNN. Single model instance is reused
    across frames. Move to GPU once."""

    def __init__(self,
                 device: str = 'cuda',
                 score_thresh: float = 0.30,
                 min_lidar_pts_for_depth: int = 3):
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights)
        self.device = torch.device(device)
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=weights, box_score_thresh=score_thresh
        ).to(self.device).eval()
        self.score_thresh = float(score_thresh)
        self.min_lidar_pts = int(min_lidar_pts_for_depth)

    @torch.inference_mode()
    def detect_2d(self, image_rgb_uint8: np.ndarray) -> Dict:
        """Run Faster R-CNN on one image. Returns numpy arrays."""
        if image_rgb_uint8.dtype != np.uint8:
            image_rgb_uint8 = image_rgb_uint8.astype(np.uint8)
        x = torch.from_numpy(image_rgb_uint8).to(self.device)
        x = x.permute(2, 0, 1).float() / 255.0
        out = self.model([x])[0]
        return {
            'boxes':  out['boxes'].detach().cpu().numpy(),
            'labels': out['labels'].detach().cpu().numpy(),
            'scores': out['scores'].detach().cpu().numpy(),
        }

    def lift_to_3d(self,
                   det2d: Dict,
                   lidar_xyz_vehicle: np.ndarray,
                   lidar2img_4x4: np.ndarray,
                   cam_slot: int,
                   image_hw: tuple) -> List[Cam2DProposal]:
        """For each 2D box, lift to a 3D proposal using LiDAR-derived depth."""
        out: List[Cam2DProposal] = []
        if det2d['boxes'].shape[0] == 0 or lidar_xyz_vehicle.shape[0] == 0:
            return out

        H, W = image_hw
        # Project LiDAR (homog) → image
        pts_h = np.concatenate([
            lidar_xyz_vehicle,
            np.ones((lidar_xyz_vehicle.shape[0], 1), dtype=np.float32)
        ], axis=1)
        proj = pts_h @ lidar2img_4x4.T   # (N, 4)
        z = proj[:, 2]
        valid = z > 0.5
        if valid.sum() == 0:
            return out
        u = proj[valid, 0] / z[valid]
        v = proj[valid, 1] / z[valid]
        depth = z[valid]
        xyz_valid = lidar_xyz_vehicle[valid]

        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if in_img.sum() == 0:
            return out
        u = u[in_img]; v = v[in_img]
        depth = depth[in_img]
        xyz_valid = xyz_valid[in_img]

        for k in range(det2d['boxes'].shape[0]):
            x1, y1, x2, y2 = det2d['boxes'][k]
            coco = int(det2d['labels'][k])
            score = float(det2d['scores'][k])
            cls = COCO_TO_TRANSFER.get(coco)
            if cls is None or score < self.score_thresh:
                continue
            inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            n_in = int(inside.sum())
            if n_in < self.min_lidar_pts:
                continue
            d_med = float(np.median(depth[inside]))
            # 3D center in vehicle frame = mean of LiDAR points inside the
            # box at roughly the median depth (filter outliers along ray).
            pts_in = xyz_valid[inside]
            d_pts = depth[inside]
            mask = (d_pts > d_med - 2.0) & (d_pts < d_med + 2.0)
            if mask.sum() < self.min_lidar_pts:
                continue
            center = pts_in[mask].mean(0)

            # Coarse size from prior
            l, w, h = SIZE_PRIOR_MEAN[cls]
            # Yaw: best cheap guess — atan2(y, x) makes the box face ego.
            # For Phase 2a this is good enough; the GRPO/refinement stage
            # would correct it.
            yaw = float(np.arctan2(center[1], center[0]))

            out.append(Cam2DProposal(
                box=np.array([center[0], center[1], center[2],
                              l, w, h, yaw], dtype=np.float32),
                cls=cls, score=score, cam_slot=cam_slot,
                depth_med=d_med, n_lidar_inside=n_in,
                box_2d=np.array([x1, y1, x2, y2], dtype=np.float32),
            ))
        return out
