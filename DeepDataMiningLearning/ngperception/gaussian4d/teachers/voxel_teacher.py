"""
gaussian4d.teachers.voxel_teacher
================================
The **baseline teacher the Gaussian teacher must beat** (Phase-1 gate). Hard-voxelize the
semantically-labelled LiDAR points: each occupied voxel takes the majority class of the points that
fall in it, with confidence 1.0 (no uncertainty, hard quantization). `sweeps=1` = single-sweep;
`sweeps=10` = the strong multi-sweep baseline.

**hard** (`soft_cache=None`): per-voxel majority argmax class. **soft** (`soft_cache` set): each point
carries a top-K FM class distribution; accumulate soft class mass per voxel -> per-voxel top-K
distribution target. Same voxelization for both, so voxel-soft isolates the semantic-distillation
gain from the Gaussian-geometry gain in the {voxel,gaussian}×{hard,soft} 2×2.

    teacher = VoxelTeacher(nusc, labelgen_cache, sweeps=10)
    target  = teacher(token)          # -> TeacherTarget
"""
from __future__ import annotations
import numpy as np

from .base import TeacherTarget, voxel_indices, topk_from_mass, FREE, NUM_CLASSES
from .semantics import lidar_scene, lidar_scene_soft
from .raycast import free_space_mask
from ...occupancy.geom import GRID_SIZE


class VoxelTeacher:
    def __init__(self, nusc, labelgen_cache, sweeps=1, free_w=0.02, soft_cache=None):
        self.nusc, self.cache, self.sweeps, self.free_w = nusc, labelgen_cache, sweeps, free_w
        self.soft_cache = soft_cache                            # None = hard argmax; else top-K soft

    def __call__(self, token) -> TeacherTarget:
        sem = np.full(tuple(GRID_SIZE), FREE, np.uint8)
        weight = np.zeros(tuple(GRID_SIZE), np.float32)
        V = int(GRID_SIZE.prod())
        soft_idx = soft_prob = None
        if self.soft_cache:                                    # SOFT: per-point top-K distribution
            pts, sidx, sprob, origin = lidar_scene_soft(self.nusc, token, self.soft_cache, self.sweeps)
            keep = (sidx[:, 0] >= 0) & (sidx[:, 0] != FREE)
            opts, olab, oprob = pts[keep], sidx[keep], sprob[keep]
            if len(opts):
                idx, m = voxel_indices(opts)
                idx, olab, oprob = idx[m], olab[m], oprob[m]
                flat = (idx[:, 0] * GRID_SIZE[1] + idx[:, 1]) * GRID_SIZE[2] + idx[:, 2]
                cls_mass = np.zeros((V, NUM_CLASSES - 1), np.float32)
                for kk in range(olab.shape[1]):
                    ck = olab[:, kk]; val = (ck >= 0) & (ck < FREE)   # drop free (17) from mass
                    np.add.at(cls_mass, (flat[val], ck[val]), oprob[:, kk][val])
                occ = cls_mass.sum(1) > 0
                sem.reshape(-1)[occ] = cls_mass[occ].argmax(1).astype(np.uint8)
                weight.reshape(-1)[occ] = 1.0
                si, sp = topk_from_mass(cls_mass, occ, K=olab.shape[1])
                soft_idx = si.reshape(tuple(GRID_SIZE) + (-1,)); soft_prob = sp.reshape(tuple(GRID_SIZE) + (-1,))
        else:                                                  # HARD: majority argmax class
            pts, lab, origin = lidar_scene(self.nusc, token, self.cache, self.sweeps)
            keep = (lab >= 0) & (lab != FREE)
            opts, olab = pts[keep], lab[keep]
            if len(opts):
                idx, m = voxel_indices(opts)
                idx, olab = idx[m], olab[m]
                flat = (idx[:, 0] * GRID_SIZE[1] + idx[:, 1]) * GRID_SIZE[2] + idx[:, 2]
                counts = np.zeros((V, NUM_CLASSES - 1), np.int32)
                np.add.at(counts, (flat, olab), 1)
                occ = counts.sum(1) > 0
                sem.reshape(-1)[occ] = counts.argmax(1)[occ].astype(np.uint8)
                weight.reshape(-1)[occ] = 1.0                   # hard: certain where occupied
        # --- ray-verified free (weight free_w) vs unknown (weight 0) ---
        free = free_space_mask(pts, origin) & (sem == FREE)     # free only where not occupied
        weight[free] = self.free_w
        return TeacherTarget(sem, weight, soft_idx, soft_prob)
