"""
gaussian4d.teachers.semantics
=============================
LiDAR geometry + frozen-FM 2D semantics = the label-free supervision source shared by **all**
teachers. We load LiDAR points (single- or multi-sweep) in the ego frame and assign each point a
semantic class by projecting it into the surround cameras and sampling the cached `labelgen`
segmentation (SegFormer + Grounded-SAM, already in the Occ3D class space). No human 3D labels.

`labelgen` cache: `<token>.npz` with `sem (6,H,W)` uint8 in the labelgen camera order below.
"""
from __future__ import annotations
import os
import numpy as np

# labelgen NuScenesSource camera order (must match how the cache was written)
LABELGEN_CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                 "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
SEM_HW = (252, 700)                        # labelgen image_hw used for the cache
FREE = 17


def load_lidar_ego(nusc, token, sweeps=1):
    """LiDAR point cloud in the **ego** frame (of the keyframe). Returns (N,3) xyz + (N,) range.
    sweeps>1 aggregates temporally-nearby sweeps (denser geometry, the strong voxel-teacher baseline)."""
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
    sample = nusc.get("sample", token)
    lsd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    if sweeps > 1:
        pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample, "LIDAR_TOP", "LIDAR_TOP", nsweeps=sweeps)
    else:
        pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lsd["filename"]))
    cs = nusc.get("calibrated_sensor", lsd["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs["rotation"]).rotation_matrix)      # sensor -> ego
    pc.translate(np.array(cs["translation"]))
    pts = pc.points[:3].T.astype(np.float32)
    return pts, np.linalg.norm(pts, axis=1)


def assign_semantics(nusc, token, points_ego, labelgen_cache):
    """Project ego points into each camera, sample the labelgen segmentation -> per-point class.
    A point is labelled by the first camera it lands in (front cameras iterated first). Points that
    fall in no camera / on sky(free) get FREE and are dropped by the caller. Returns (N,) int labels."""
    from pyquaternion import Quaternion
    npz = np.load(os.path.join(labelgen_cache, token + ".npz"))
    sem = npz["sem"]                                            # (6,H,W) labelgen cam order
    H, W = SEM_HW
    sample = nusc.get("sample", token)
    labels = np.full(len(points_ego), -1, np.int64)            # -1 = unassigned
    for ci, cam in enumerate(LABELGEN_CAMS):
        sd = nusc.get("sample_data", sample["data"][cam])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        R = Quaternion(cs["rotation"]).rotation_matrix.astype(np.float32)   # cam->ego
        t = np.array(cs["translation"], np.float32)
        K = np.array(cs["camera_intrinsic"], np.float32).copy()
        ow, oh = sd["width"], sd["height"]
        K[0] *= W / ow; K[1] *= H / oh                          # K at labelgen resolution
        cam_pts = (points_ego - t) @ R                          # ego -> cam (R^T @ (p-t))
        z = cam_pts[:, 2]
        uvw = cam_pts @ K.T
        u = np.round(uvw[:, 0] / np.clip(z, 1e-3, None)).astype(np.int64)
        v = np.round(uvw[:, 1] / np.clip(z, 1e-3, None)).astype(np.int64)
        vis = (z > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (labels < 0)
        cls = sem[ci][v[vis], u[vis]]
        labels[np.where(vis)[0]] = cls
    return labels


def lidar_points_and_labels(nusc, token, labelgen_cache, sweeps=1):
    """(N,3) ego points + (N,) Occ3D-class labels, keeping only camera-labelled non-free points."""
    pts, _ = load_lidar_ego(nusc, token, sweeps)
    lab = assign_semantics(nusc, token, pts, labelgen_cache)
    keep = (lab >= 0) & (lab != FREE)
    return pts[keep], lab[keep]
