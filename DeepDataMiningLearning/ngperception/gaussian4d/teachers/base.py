"""
gaussian4d.teachers.base
========================
The shared contract for every occupancy **teacher** (Phase 1). A teacher turns raw sensor data
(LiDAR geometry + frozen-FM 2D semantics) into a dense supervision target for a *camera-only*
student — with **no human 3D/box labels**. Every teacher (voxel, Gaussian, …) returns the same
`TeacherTarget`, so the student trains identically against any of them and the Gaussian-vs-voxel
comparison (the Phase-1 gate) is apples-to-apples: only the geometric representation changes.

Grid = Occ3D-nuScenes: x,y ∈ [-40,40], z ∈ [-1,5.4], 0.4 m → (200,200,16), class 17 = free.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ...occupancy.geom import PC_RANGE, VOXEL_SIZE, GRID_SIZE   # [-40..5.4], 0.4, (200,200,16)

FREE = 17
NUM_CLASSES = 18                          # 0..16 semantic + 17 free
# Occ3D-nuScenes class names (index -> name); tail = rare/thin classes we care about most.
CLASS_NAMES = ["others", "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
               "pedestrian", "traffic_cone", "trailer", "truck", "driveable_surface", "other_flat",
               "sidewalk", "terrain", "manmade", "vegetation", "free"]
TAIL_CLASSES = [1, 2, 6, 7, 8, 9]         # barrier, bicycle, motorcycle, pedestrian, cone, trailer


@dataclass
class TeacherTarget:
    """Dense supervision target in the Occ3D voxel grid, produced by any teacher.

    semantics : (200,200,16) uint8   — per-voxel class (17 = free/empty).
    weight    : (200,200,16) float32 — per-voxel supervision confidence in [0,1] (the *uncertainty*
                signal). 1.0 = certain; 0.0 = do not supervise. Hard-voxel teachers use 1.0 on
                occupied voxels; the Gaussian teacher fills this with a soft, density/range-aware
                confidence so the student is down-weighted where the teacher is unsure.
    """
    semantics: np.ndarray
    weight: np.ndarray

    def __post_init__(self):
        assert self.semantics.shape == tuple(GRID_SIZE), self.semantics.shape
        assert self.weight.shape == tuple(GRID_SIZE), self.weight.shape

    @property
    def occupied(self) -> int:
        return int((self.semantics != FREE).sum())

    def save(self, path):
        np.savez_compressed(path, semantics=self.semantics.astype(np.uint8),
                            weight=self.weight.astype(np.float16))

    @staticmethod
    def load(path) -> "TeacherTarget":
        d = np.load(path)
        return TeacherTarget(d["semantics"].astype(np.uint8), d["weight"].astype(np.float32))


def voxel_indices(points_ego: np.ndarray):
    """(N,3) ego points -> integer voxel indices + in-bounds mask (shared by all teachers)."""
    idx = np.floor((points_ego - np.asarray(PC_RANGE[:3])) / VOXEL_SIZE).astype(np.int64)
    m = np.all((idx >= 0) & (idx < np.asarray(GRID_SIZE)), axis=1)
    return idx, m


def voxel_centers():
    """(200,200,16,3) ego-frame centre of every voxel — for Gaussian-to-voxel splatting."""
    gx, gy, gz = GRID_SIZE
    xs = PC_RANGE[0] + (np.arange(gx) + 0.5) * VOXEL_SIZE
    ys = PC_RANGE[1] + (np.arange(gy) + 0.5) * VOXEL_SIZE
    zs = PC_RANGE[2] + (np.arange(gz) + 0.5) * VOXEL_SIZE
    return np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).astype(np.float32)
