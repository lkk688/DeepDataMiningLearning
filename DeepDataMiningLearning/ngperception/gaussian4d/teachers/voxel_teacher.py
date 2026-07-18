"""
gaussian4d.teachers.voxel_teacher
================================
The **baseline teacher the Gaussian teacher must beat** (Phase-1 gate). Hard-voxelize the
semantically-labelled LiDAR points: each occupied voxel takes the majority class of the points that
fall in it, with confidence 1.0 (no uncertainty, hard quantization). `sweeps=1` = single-sweep;
`sweeps=10` = the strong multi-sweep baseline.

    teacher = VoxelTeacher(nusc, labelgen_cache, sweeps=10)
    target  = teacher(token)          # -> TeacherTarget
"""
from __future__ import annotations
import numpy as np

from .base import TeacherTarget, voxel_indices, FREE, NUM_CLASSES
from .semantics import lidar_points_and_labels
from ...occupancy.geom import GRID_SIZE


class VoxelTeacher:
    def __init__(self, nusc, labelgen_cache, sweeps=1):
        self.nusc, self.cache, self.sweeps = nusc, labelgen_cache, sweeps

    def __call__(self, token) -> TeacherTarget:
        pts, lab = lidar_points_and_labels(self.nusc, token, self.cache, self.sweeps)
        sem = np.full(tuple(GRID_SIZE), FREE, np.uint8)
        weight = np.zeros(tuple(GRID_SIZE), np.float32)
        if len(pts) == 0:
            return TeacherTarget(sem, weight)
        idx, m = voxel_indices(pts)
        idx, lab = idx[m], lab[m]
        # majority class per voxel: accumulate per-(voxel,class) counts, then argmax
        flat = (idx[:, 0] * GRID_SIZE[1] + idx[:, 1]) * GRID_SIZE[2] + idx[:, 2]
        counts = np.zeros((int(GRID_SIZE.prod()), NUM_CLASSES - 1), np.int32)   # classes 0..16
        np.add.at(counts, (flat, lab), 1)
        occ = counts.sum(1) > 0
        best = counts.argmax(1)
        sem_flat = sem.reshape(-1); w_flat = weight.reshape(-1)
        sem_flat[occ] = best[occ].astype(np.uint8)
        w_flat[occ] = 1.0                                       # hard: fully certain where occupied
        return TeacherTarget(sem_flat.reshape(GRID_SIZE), w_flat.reshape(GRID_SIZE))
