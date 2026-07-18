"""
gaussian4d.teachers.gaussian_teacher
====================================
The **novel teacher** (Phase-1 gate: must beat `VoxelTeacher`). Each semantically-labelled LiDAR
point becomes an isotropic 3D Gaussian; the Gaussians are **soft-splatted** into the Occ3D grid.
This buys the three advantages a hard voxel grid cannot (each must be *demonstrated*, not asserted):

1. **Less hard quantization** — a Gaussian with local-density scale spreads sub-voxel mass into
   neighbouring voxels, so thin/tail structures (few points) still fill their footprint.
2. **Uncertainty** — `weight` = accumulated-mass confidence × range discount, so the student is
   down-weighted where the teacher is sparse/far (an *uncertainty* signal with a downstream use).
3. **Continuous representation** — Gaussians carry (position, scale); adding per-Gaussian
   velocity/deformation extends the *same* teacher to 4D forecasting (Phase 3). This file is the
   static (single-time) case; the Gaussian list is exposed via `gaussians()` for the 4D extension.

    teacher = GaussianTeacher(nusc, labelgen_cache, sweeps=1)
    target  = teacher(token)          # -> TeacherTarget (same contract as VoxelTeacher)
"""
from __future__ import annotations
import numpy as np

from .base import TeacherTarget, voxel_indices, FREE, NUM_CLASSES
from .semantics import lidar_points_and_labels
from ...occupancy.geom import PC_RANGE, VOXEL_SIZE, GRID_SIZE

# 3x3x3 voxel neighbourhood offsets — where each Gaussian deposits sub-voxel mass.
_OFFS = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], np.int64)


class GaussianTeacher:
    def __init__(self, nusc, labelgen_cache, sweeps=1, scale_k=1.0, mass_tau=1.0,
                 range_full=25.0, range_far=55.0, min_conf=0.3):
        self.nusc, self.cache, self.sweeps = nusc, labelgen_cache, sweeps
        self.scale_k = scale_k            # Gaussian sigma = scale_k * (nearest-neighbour spacing)
        self.mass_tau = mass_tau          # accumulated mass -> confidence saturation
        self.range_full, self.range_far, self.min_conf = range_full, range_far, min_conf

    def _sigma(self, pts):
        """Per-point isotropic sigma from local spacing (sparse regions -> larger, fills more)."""
        try:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(pts).query(pts, k=2)
            nn = d[:, 1]
        except Exception:
            nn = np.full(len(pts), VOXEL_SIZE, np.float32)      # fallback: one voxel
        return np.clip(self.scale_k * nn, 0.5 * VOXEL_SIZE, 2.0 * VOXEL_SIZE).astype(np.float32)

    def gaussians(self, token):
        """The raw Gaussian list (pos, sigma, class) — exposed for the 4D extension / query loss."""
        pts, lab = lidar_points_and_labels(self.nusc, token, self.cache, self.sweeps)
        return pts, (self._sigma(pts) if len(pts) else np.zeros(0, np.float32)), lab

    def __call__(self, token) -> TeacherTarget:
        pts, sig, lab = self.gaussians(token)
        gx, gy, gz = [int(v) for v in GRID_SIZE]
        cls_mass = np.zeros((gx * gy * gz, NUM_CLASSES - 1), np.float32)   # per-voxel per-class mass
        tot_mass = np.zeros(gx * gy * gz, np.float32)
        if len(pts):
            v0, _ = voxel_indices(pts)                          # each point's home voxel
            origin = np.asarray(PC_RANGE[:3], np.float32)
            inv2s2 = 1.0 / (2.0 * sig * sig)
            for off in _OFFS:
                vi = v0 + off                                   # neighbour voxel indices
                ok = np.all((vi >= 0) & (vi < np.asarray(GRID_SIZE)), 1)
                if not ok.any():
                    continue
                vok, pok, lok, s2 = vi[ok], pts[ok], lab[ok], inv2s2[ok]
                centres = origin + (vok + 0.5) * VOXEL_SIZE     # neighbour voxel centres
                d2 = ((pok - centres) ** 2).sum(1)
                g = np.exp(-d2 * s2).astype(np.float32)         # Gaussian mass into this voxel
                flat = (vok[:, 0] * gy + vok[:, 1]) * gz + vok[:, 2]
                np.add.at(cls_mass, (flat, lok), g)
                np.add.at(tot_mass, flat, g)
        sem = np.full(gx * gy * gz, FREE, np.uint8)
        occ = tot_mass > 0.1                                    # a voxel is occupied once it gathers mass
        sem[occ] = cls_mass[occ].argmax(1).astype(np.uint8)
        # uncertainty weight = mass-confidence x range-discount (in [min_conf,1] on occupied voxels)
        conf = 1.0 - np.exp(-tot_mass / self.mass_tau)
        centres_all = (np.asarray(PC_RANGE[:3], np.float32)
                       + (np.stack(np.unravel_index(np.arange(gx * gy * gz), (gx, gy, gz)), 1) + 0.5) * VOXEL_SIZE)
        rng = np.linalg.norm(centres_all[:, :2], axis=1)
        rdisc = np.clip(1.0 - (rng - self.range_full) / (self.range_far - self.range_full), self.min_conf, 1.0)
        weight = np.where(occ, np.clip(conf * rdisc, 0.0, 1.0), 0.0).astype(np.float32)
        return TeacherTarget(sem.reshape(GRID_SIZE), weight.reshape(GRID_SIZE))
