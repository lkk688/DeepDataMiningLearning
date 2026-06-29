"""
ngperception.occupancy.predictors.depth_lift
============================================

The **depth -> occupancy baseline**: for each of the 6 surround cameras, run a metric
depth model, back-project every pixel to a 3D point in the ego frame, optionally tag it
with a 2D semantic-segmentation label, and splat into the Occ3D voxel grid. No occupancy
training at all — this measures how far a depth foundation model + 2D segmentation get you
toward learned occupancy (the ViPOcc-style "vision priors" question).

Semantics come from a Cityscapes SegFormer, mapped to the 18 Occ3D classes. Sky and
far/zero depth are dropped (they are not occupied surfaces).
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from ..geom import sample_cameras, backproject, voxelize, FREE, GRID_SIZE

# Cityscapes trainId (0..18) -> Occ3D class id. Sky(10) -> 255 (drop).
CS2OCC = np.array([
    11,  # 0 road        -> driveable_surface
    13,  # 1 sidewalk    -> sidewalk
    15,  # 2 building    -> manmade
    15,  # 3 wall        -> manmade
    15,  # 4 fence       -> manmade
    15,  # 5 pole        -> manmade
    15,  # 6 traffic light-> manmade
    15,  # 7 traffic sign-> manmade
    16,  # 8 vegetation  -> vegetation
    14,  # 9 terrain     -> terrain
    255, # 10 sky        -> drop
    7,   # 11 person     -> pedestrian
    7,   # 12 rider      -> pedestrian
    4,   # 13 car        -> car
    10,  # 14 truck      -> truck
    3,   # 15 bus        -> bus
    0,   # 16 train      -> others
    6,   # 17 motorcycle -> motorcycle
    2,   # 18 bicycle    -> bicycle
], dtype=np.int16)


class DepthLiftOccupancy:
    """Lift 6-camera metric depth (+ optional 2D semantics) into the Occ3D grid."""

    def __init__(self, nusc, depth_spec: str, seg_model: Optional[str] = None,
                 device="cuda", stride: int = 2):
        from ...depth.estimators.base import build_estimator
        self.nusc = nusc
        self.device = device
        self.stride = stride
        self.depth = build_estimator(depth_spec, device=device)
        if not self.depth.is_metric:
            print("[depth_lift] WARNING: depth model is not metric; geometry will be wrong.")
        self.seg = None
        if seg_model:
            import torch
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            self.torch = torch
            self.seg_proc = AutoImageProcessor.from_pretrained(seg_model)
            self.seg = AutoModelForSemanticSegmentation.from_pretrained(
                seg_model, torch_dtype=torch.float16).to(device).eval()

    def _seg_labels(self, image):
        """Cityscapes trainId map at the image resolution (H,W) int16."""
        import torch
        w, h = image.size
        inp = self.seg_proc(images=image, return_tensors="pt").to(self.device)
        inp = {k: v.half() if v.is_floating_point() else v for k, v in inp.items()}
        with torch.no_grad():
            logits = self.seg(**inp).logits
        up = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear",
                                             align_corners=False)
        return up.argmax(1)[0].cpu().numpy().astype(np.int16)

    def predict(self, token: str) -> np.ndarray:
        from PIL import Image
        pts_all, lab_all = [], []
        for cam in sample_cameras(self.nusc, token):
            image = Image.open(cam["path"]).convert("RGB")
            depth = self.depth.predict(image).depth
            pts_cam, valid = backproject(depth, cam["K"], self.stride)
            if pts_cam.shape[0] == 0:
                continue
            pts_ego = pts_cam @ cam["R"].T + cam["t"]               # ego frame
            pts_all.append(pts_ego)
            if self.seg is not None:
                cs = self._seg_labels(image)[::self.stride, ::self.stride].reshape(-1)[valid]
                lab_all.append(CS2OCC[np.clip(cs, 0, 18)])
        if not pts_all:
            return np.full(tuple(GRID_SIZE), FREE, np.uint8)
        pts = np.concatenate(pts_all)
        if self.seg is not None:
            lab = np.concatenate(lab_all)
            keep = lab != 255                                       # drop sky
            return voxelize(pts[keep], lab[keep])
        return voxelize(pts)                                        # geometry only (class 0)
