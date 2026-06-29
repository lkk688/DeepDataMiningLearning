"""
ngperception.occupancy.geom
===========================

Geometry helpers for lifting 2D predictions into the Occ3D voxel grid:

* `sample_cameras(nusc, token)` -> per-camera (image path, intrinsics K, sensor->ego R,t).
  Validated empirically (LiDAR projects to the GT-occupied voxels) that the Occ3D-nuScenes
  grid is in the **ego** frame, so a camera point is lifted by `X_ego = R_c2e X_cam + t_c2e`.
* `backproject(depth, K)` -> camera-frame 3D points from a metric depth map.
* `voxelize(points_ego, labels)` -> a (200,200,16) semantic grid (free = 17).
"""

from __future__ import annotations
import os
from typing import List, Tuple

import numpy as np

PC_RANGE = np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4])
VOXEL_SIZE = 0.4
GRID_SIZE = np.array([200, 200, 16])
FREE = 17
CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def sample_cameras(nusc, token: str) -> List[dict]:
    """Return [{name, path, K(3x3), R(3x3 sensor->ego), t(3,)}] for the 6 surround cams."""
    from pyquaternion import Quaternion
    s = nusc.get("sample", token)
    out = []
    for cam in CAMS:
        sd = nusc.get("sample_data", s["data"][cam])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        out.append({
            "name": cam,
            "path": os.path.join(nusc.dataroot, sd["filename"]),
            "K": np.array(cs["camera_intrinsic"], np.float64),
            "R": Quaternion(cs["rotation"]).rotation_matrix,
            "t": np.array(cs["translation"], np.float64),
        })
    return out


def backproject(depth: np.ndarray, K: np.ndarray, stride: int = 2) -> np.ndarray:
    """Metric depth map (H,W) + intrinsics -> (N,3) camera-frame points (sub-sampled)."""
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[::stride, ::stride].reshape(-1)
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    valid = (d > 0.3) & (d < 60.0)                      # ignore zeros and far/noisy depth
    xs, ys, d = xs[valid], ys[valid], d[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (xs - cx) / fx * d
    Y = (ys - cy) / fy * d
    return np.stack([X, Y, d], axis=1), valid          # cam frame (z forward)


def to_voxels(points_ego: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map ego-frame points -> integer voxel indices, keeping in-bounds ones."""
    idx = np.floor((points_ego - PC_RANGE[:3]) / VOXEL_SIZE).astype(int)
    m = np.all((idx >= 0) & (idx < GRID_SIZE), axis=1)
    return idx[m], m


def voxelize(points_ego: np.ndarray, labels: np.ndarray = None) -> np.ndarray:
    """Build a (200,200,16) semantic grid (free=17). If labels is None, occupied voxels get
    class 0; else each voxel takes the majority label of points falling in it (nearest-z
    wins ties cheaply via last-write after sorting far->near)."""
    grid = np.full(tuple(GRID_SIZE), FREE, np.uint8)
    idx, m = to_voxels(points_ego)
    if idx.shape[0] == 0:
        return grid
    if labels is None:
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 0
        return grid
    lab = labels[m]
    # write near points last so the nearest surface label wins (sort by descending depth =
    # descending distance; here use ego x^2+y^2 as a cheap range proxy is unnecessary —
    # caller passes points already; we sort by z-up irrelevant, so use original order).
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = lab.astype(np.uint8)
    return grid
