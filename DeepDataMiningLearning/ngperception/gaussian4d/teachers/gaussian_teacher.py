"""
gaussian4d.teachers.gaussian_teacher
====================================
The **novel teacher** (Phase-1 gate: must beat `VoxelTeacher`). Each semantically-labelled LiDAR
point becomes a 3D Gaussian; the Gaussians are **soft-splatted** into the Occ3D grid. Advantages a
hard voxel grid cannot have (each must be *demonstrated*):

1. **Less quantization / surface-aligned** — with an **anisotropic** covariance from local-surface
   PCA, a Gaussian is a *flat disk on the surface* (thin along the normal, wide in the tangent
   plane): it fills a thin/tail structure's footprint **without bleeding perpendicular into free
   space** (the failure mode of an isotropic blob). `anisotropic=False` = the old isotropic version.
2. **Uncertainty** — `weight` = accumulated-mass confidence × range discount → student down-weighted
   where the teacher is sparse/far.
3. **Continuous representation** — Gaussians carry (μ, Σ); per-Gaussian velocity extends the *same*
   teacher to 4D (Phase 3). `gaussians()` exposes the list.

Uses ray-verified free space (`raycast`) so the target is occupied / free / **unknown**.
"""
from __future__ import annotations
import numpy as np

from .base import TeacherTarget, voxel_indices, FREE, NUM_CLASSES
from .semantics import lidar_scene
from .raycast import free_space_mask
from ...occupancy.geom import PC_RANGE, VOXEL_SIZE, GRID_SIZE


def _offsets(r):
    a = range(-r, r + 1)
    return np.array([[i, j, k] for i in a for j in a for k in a], np.int64)


class GaussianTeacher:
    def __init__(self, nusc, labelgen_cache, sweeps=1, scale_k=1.0, mass_tau=1.0,
                 range_full=25.0, range_far=55.0, min_conf=0.3, free_w=0.02,
                 anisotropic=True, k_pca=12, sigma_normal=0.35):
        self.nusc, self.cache, self.sweeps = nusc, labelgen_cache, sweeps
        self.scale_k, self.mass_tau = scale_k, mass_tau
        self.range_full, self.range_far, self.min_conf, self.free_w = range_full, range_far, min_conf, free_w
        self.anisotropic, self.k_pca = anisotropic, k_pca
        self.sigma_normal = sigma_normal * VOXEL_SIZE          # thin extent along the surface normal
        self.offsets = _offsets(2 if anisotropic else 1)       # anisotropic needs the tangent reach

    def _sigma(self, pts):
        """Isotropic sigma from local spacing (fallback / anisotropic=False)."""
        try:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(pts).query(pts, k=2)
            nn = d[:, 1]
        except Exception:
            nn = np.full(len(pts), VOXEL_SIZE, np.float32)
        return np.clip(self.scale_k * nn, 0.5 * VOXEL_SIZE, 2.0 * VOXEL_SIZE).astype(np.float32)

    def _aniso(self, pts):
        """Surface-aligned covariance via local PCA. Returns eigenvectors V (N,3,3) and per-axis
        inverse-variances (N,3): thin (1/sigma_normal^2) along the least-spread eigenvector (surface
        normal), wide (1/sigma_tangent^2) in the two tangent directions. -> a flat disk on the surface."""
        from scipy.spatial import cKDTree
        k = min(self.k_pca, len(pts))
        d, idx = cKDTree(pts).query(pts, k=k)
        nbrs = pts[idx]                                        # (N,k,3)
        cen = nbrs - nbrs.mean(1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", cen, cen) / max(k, 1)
        _, V = np.linalg.eigh(cov)                            # ascending; V[:,:,0]=normal, 1,2=tangent
        nn = d[:, 1] if k > 1 else np.full(len(pts), VOXEL_SIZE)
        sig_t = np.clip(self.scale_k * nn, 0.5 * VOXEL_SIZE, 2.0 * VOXEL_SIZE)
        inv_n = np.full(len(pts), 1.0 / self.sigma_normal ** 2, np.float32)
        inv_t = (1.0 / sig_t ** 2).astype(np.float32)
        return V.astype(np.float32), np.stack([inv_n, inv_t, inv_t], 1)   # (N,3,3),(N,3)

    def gaussians(self, token):
        """Labelled surface Gaussians (pos, class) + all points + sensor origin — for 4D / query loss."""
        pts_all, lab, origin = lidar_scene(self.nusc, token, self.cache, self.sweeps)
        keep = (lab >= 0) & (lab != FREE)
        return pts_all[keep], lab[keep], pts_all, origin

    def __call__(self, token) -> TeacherTarget:
        pts, lab, pts_all, origin = self.gaussians(token)
        gx, gy, gz = [int(v) for v in GRID_SIZE]
        cls_mass = np.zeros((gx * gy * gz, NUM_CLASSES - 1), np.float32)
        tot_mass = np.zeros(gx * gy * gz, np.float32)
        grid0 = np.asarray(PC_RANGE[:3], np.float32)           # GRID corner (for voxel centres)
        if len(pts):
            v0, _ = voxel_indices(pts)
            if self.anisotropic:
                V, inv_var = self._aniso(pts)
            else:
                inv2s2 = 1.0 / (2.0 * self._sigma(pts) ** 2)
            for off in self.offsets:
                vi = v0 + off
                ok = np.all((vi >= 0) & (vi < np.asarray(GRID_SIZE)), 1)
                if not ok.any():
                    continue
                vok, pok, lok = vi[ok], pts[ok], lab[ok]
                delta = pok - (grid0 + (vok + 0.5) * VOXEL_SIZE)         # point -> voxel centre
                if self.anisotropic:
                    proj = np.einsum("mj,mji->mi", delta, V[ok])         # coords in surface eigenbasis
                    g = np.exp(-0.5 * (proj ** 2 * inv_var[ok]).sum(1)).astype(np.float32)
                else:
                    g = np.exp(-(delta ** 2).sum(1) * inv2s2[ok]).astype(np.float32)
                flat = (vok[:, 0] * gy + vok[:, 1]) * gz + vok[:, 2]
                np.add.at(cls_mass, (flat, lok), g)
                np.add.at(tot_mass, flat, g)
        sem = np.full(gx * gy * gz, FREE, np.uint8)
        occ = tot_mass > 0.1
        sem[occ] = cls_mass[occ].argmax(1).astype(np.uint8)
        conf = 1.0 - np.exp(-tot_mass / self.mass_tau)
        centres = (grid0 + (np.stack(np.unravel_index(np.arange(gx * gy * gz), (gx, gy, gz)), 1) + 0.5) * VOXEL_SIZE)
        rng = np.linalg.norm(centres[:, :2], axis=1)
        rdisc = np.clip(1.0 - (rng - self.range_full) / (self.range_far - self.range_full), self.min_conf, 1.0)
        weight = np.where(occ, np.clip(conf * rdisc, 0.0, 1.0), 0.0).astype(np.float32)
        sem = sem.reshape(GRID_SIZE); weight = weight.reshape(GRID_SIZE)
        free = free_space_mask(pts_all, origin) & (sem == FREE)   # origin = SENSOR origin (bug fixed)
        weight[free] = np.maximum(weight[free], self.free_w)
        return TeacherTarget(sem, weight)
