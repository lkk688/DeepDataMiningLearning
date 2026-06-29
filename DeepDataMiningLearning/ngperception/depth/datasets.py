"""
ngperception.depth.datasets
===========================

Depth datasets that yield `(PIL image, sparse GT depth map)` pairs in a model-agnostic
form. The first dataset is **KITTI**, whose ground-truth depth we build by **projecting
the Velodyne LiDAR point cloud into the left color camera** using the per-frame calib —
this is exactly how the KITTI depth benchmark defines GT (sparse, ~5% of pixels).

Projection (standard KITTI):
    X_cam = R0_rect @ Tr_velo_to_cam @ X_velo            # 3D in the rectified cam frame
    [u v w]^T = P2 @ [X_cam; 1];   u/=w, v/=w            # pixel
    depth = X_cam.z                                      # metric depth (meters)

Only points in front of the camera (z>0) and inside the image are kept; on pixel
collisions we keep the nearest (smallest depth).
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

DEFAULT_KITTI_ROOT = "/mnt/e/Shared/Dataset/Kitti/"


@dataclass
class DepthSample:
    image: "Image.Image"     # PIL RGB
    depth_gt: np.ndarray     # HxW float32, meters; 0 = no LiDAR return (invalid)
    sample_id: str

    def valid_mask(self, min_depth=1e-3, max_depth=80.0) -> np.ndarray:
        return (self.depth_gt > min_depth) & (self.depth_gt < max_depth)


def _read_calib(path: str) -> dict:
    """Parse a KITTI calib .txt into {key: float-array}."""
    out = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = np.array([float(x) for x in v.split()], dtype=np.float64)
    return out


def _project_velo_to_depth(velo: np.ndarray, calib: dict, h: int, w: int) -> np.ndarray:
    """Project an (N,4) Velodyne cloud to a sparse HxW depth map (meters)."""
    P2 = calib["P2"].reshape(3, 4)
    R0 = np.eye(4); R0[:3, :3] = calib["R0_rect"].reshape(3, 3)
    Tr = np.eye(4); Tr[:3, :4] = calib["Tr_velo_to_cam"].reshape(3, 4)

    pts = velo[:, :3]
    pts = pts[pts[:, 0] > 0]                               # drop points behind the LiDAR
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])   # (N,4)
    cam = (R0 @ Tr @ pts_h.T).T                            # (N,4) rectified cam frame
    depth = cam[:, 2]
    front = depth > 0
    cam, depth = cam[front], depth[front]

    img = (P2 @ np.hstack([cam[:, :3], np.ones((cam.shape[0], 1))]).T).T
    u = np.round(img[:, 0] / img[:, 2]).astype(int)
    v = np.round(img[:, 1] / img[:, 2]).astype(int)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, depth = u[inb], v[inb], depth[inb]

    dmap = np.zeros((h, w), dtype=np.float32)
    order = np.argsort(-depth)                             # far first so near overwrites
    dmap[v[order], u[order]] = depth[order]
    return dmap


class KITTIDepthDataset:
    """KITTI left-color images + LiDAR-projected sparse GT depth.

    Parameters
    ----------
    root : str
        KITTI dir containing `training/{image_2,velodyne,calib}`.
    max_images, offset, stride : int
        Subsample the 7481 training frames for quick comparison runs.
    """

    def __init__(self, root: str = DEFAULT_KITTI_ROOT, split: str = "training",
                 max_images: Optional[int] = None, offset: int = 0, stride: int = 1):
        self.base = os.path.join(root, split)
        self.img_dir = os.path.join(self.base, "image_2")
        self.velo_dir = os.path.join(self.base, "velodyne")
        self.calib_dir = os.path.join(self.base, "calib")
        ids = sorted(f[:-4] for f in os.listdir(self.img_dir) if f.endswith(".png"))
        ids = ids[offset::stride]
        if max_images:
            ids = ids[:max_images]
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int) -> DepthSample:
        sid = self.ids[i]
        image = Image.open(os.path.join(self.img_dir, sid + ".png")).convert("RGB")
        w, h = image.size
        velo = np.fromfile(os.path.join(self.velo_dir, sid + ".bin"),
                           dtype=np.float32).reshape(-1, 4)
        calib = _read_calib(os.path.join(self.calib_dir, sid + ".txt"))
        depth_gt = _project_velo_to_depth(velo, calib, h, w)
        return DepthSample(image=image, depth_gt=depth_gt, sample_id=sid)


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.depth.datasets
# Prints GT coverage stats for a few KITTI frames (no model needed).
# ===========================================================================
if __name__ == "__main__":
    ds = KITTIDepthDataset(max_images=3)
    print(f"KITTI depth: {len(ds)} frames sampled")
    for i in range(len(ds)):
        s = ds[i]
        m = s.valid_mask()
        print(f"  {s.sample_id}: img={s.image.size} valid_px={m.sum()} "
              f"({100*m.mean():.1f}%) depth[min/max]={s.depth_gt[m].min():.1f}/{s.depth_gt[m].max():.1f}m")
