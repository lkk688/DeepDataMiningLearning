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
from .semantics import lidar_scene
from .raycast import free_space_mask
from ...occupancy.geom import GRID_SIZE


class VoxelTeacher:
    def __init__(self, nusc, labelgen_cache, sweeps=1, free_w=0.02):
        self.nusc, self.cache, self.sweeps, self.free_w = nusc, labelgen_cache, sweeps, free_w

    def __call__(self, token) -> TeacherTarget:
        pts, lab, origin = lidar_scene(self.nusc, token, self.cache, self.sweeps)
        sem = np.full(tuple(GRID_SIZE), FREE, np.uint8)
        weight = np.zeros(tuple(GRID_SIZE), np.float32)
        if len(pts) == 0:
            return TeacherTarget(sem, weight)
        # --- occupied: majority labelled-class per voxel ---
        keep = (lab >= 0) & (lab != FREE)
        opts, olab = pts[keep], lab[keep]
        if len(opts):
            idx, m = voxel_indices(opts)
            idx, olab = idx[m], olab[m]
            flat = (idx[:, 0] * GRID_SIZE[1] + idx[:, 1]) * GRID_SIZE[2] + idx[:, 2]
            counts = np.zeros((int(GRID_SIZE.prod()), NUM_CLASSES - 1), np.int32)
            np.add.at(counts, (flat, olab), 1)
            occ = counts.sum(1) > 0
            sem.reshape(-1)[occ] = counts.argmax(1)[occ].astype(np.uint8)
            weight.reshape(-1)[occ] = 1.0                       # hard: certain where occupied
        # --- ray-verified free (weight free_w) vs unknown (weight 0) ---
        free = free_space_mask(pts, origin) & (sem == FREE)     # free only where not occupied
        weight[free] = self.free_w
        return TeacherTarget(sem, weight)
