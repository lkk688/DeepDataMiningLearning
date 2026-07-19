"""
gaussian4d.teachers.gaussian_teacher
====================================
The **novel teacher**. LiDAR points -> anisotropic surface-aligned 3D Gaussians -> soft-splatted into
the Occ3D grid. Advantages over a hard voxel grid (each must be *demonstrated*):

1. **Less quantization / surface-aligned** — anisotropic Σ (local-PCA flat disk: thin along the
   surface normal, wide in the tangent plane) fills a structure's footprint without bleeding
   perpendicular into free space. (`anisotropic=False` = isotropic blob.)
2. **Uncertainty** — `weight` = mass-confidence × range discount.
3. **Continuous representation** — (μ, Σ, v) → the same teacher extends to 4D (Phase 3).

Semantics: **hard** (each point's argmax class) or **soft** (`soft_cache` set → each point carries a
top-K FM class *distribution*, splatted as soft mass → a per-voxel top-K distribution target). The
{voxel,gaussian}×{hard,soft} 2×2 separates semantic-distillation gain from Gaussian-geometry gain.
Ray-verified free space via `raycast` (occupied / free / unknown).
"""
from __future__ import annotations
import numpy as np

from .base import TeacherTarget, voxel_indices, topk_from_mass, FREE, NUM_CLASSES
from .semantics import lidar_scene, lidar_scene_soft
from .raycast import free_space_mask
from ...occupancy.geom import PC_RANGE, VOXEL_SIZE, GRID_SIZE


def _offsets(r):
    a = range(-r, r + 1)
    return np.array([[i, j, k] for i in a for j in a for k in a], np.int64)


class GaussianTeacher:
    def __init__(self, nusc, labelgen_cache, sweeps=1, scale_k=1.0, mass_tau=1.0,
                 range_full=25.0, range_far=55.0, min_conf=0.3, free_w=0.02,
                 anisotropic=True, k_pca=12, sigma_normal=0.35, soft_cache=None):
        self.nusc, self.cache, self.sweeps = nusc, labelgen_cache, sweeps
        self.scale_k, self.mass_tau = scale_k, mass_tau
        self.range_full, self.range_far, self.min_conf, self.free_w = range_full, range_far, min_conf, free_w
        self.anisotropic, self.k_pca = anisotropic, k_pca
        self.sigma_normal = sigma_normal * VOXEL_SIZE
        self.soft_cache = soft_cache                          # None = hard argmax; else top-K soft
        self.offsets = _offsets(2 if anisotropic else 1)

    def _sigma(self, pts):
        try:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(pts).query(pts, k=2); nn = d[:, 1]
        except Exception:
            nn = np.full(len(pts), VOXEL_SIZE, np.float32)
        return np.clip(self.scale_k * nn, 0.5 * VOXEL_SIZE, 2.0 * VOXEL_SIZE).astype(np.float32)

    def _aniso(self, pts):
        """Surface-aligned covariance via local PCA -> eigenvectors V (N,3,3) + per-axis inv-var (N,3)
        (thin along the surface normal, wide in tangent)."""
        from scipy.spatial import cKDTree
        k = min(self.k_pca, len(pts))
        d, idx = cKDTree(pts).query(pts, k=k)
        cen = pts[idx] - pts[idx].mean(1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", cen, cen) / max(k, 1)
        _, V = np.linalg.eigh(cov)
        nn = d[:, 1] if k > 1 else np.full(len(pts), VOXEL_SIZE)
        sig_t = np.clip(self.scale_k * nn, 0.5 * VOXEL_SIZE, 2.0 * VOXEL_SIZE)
        inv_n = np.full(len(pts), 1.0 / self.sigma_normal ** 2, np.float32)
        return V.astype(np.float32), np.stack([inv_n, (1 / sig_t ** 2).astype(np.float32),
                                               (1 / sig_t ** 2).astype(np.float32)], 1)

    def gaussians(self, token):
        """Hard labelled surface Gaussians (pos, class) + all points + sensor origin (for 4D / query)."""
        pts_all, lab, origin = lidar_scene(self.nusc, token, self.cache, self.sweeps)
        keep = (lab >= 0) & (lab != FREE)
        return pts_all[keep], lab[keep], pts_all, origin

    def __call__(self, token) -> TeacherTarget:
        gx, gy, gz = [int(v) for v in GRID_SIZE]
        cls_mass = np.zeros((gx * gy * gz, NUM_CLASSES - 1), np.float32)
        tot_mass = np.zeros(gx * gy * gz, np.float32)
        grid0 = np.asarray(PC_RANGE[:3], np.float32)
        if self.soft_cache:                                   # SOFT: per-point top-K distribution
            pts_all, sidx, sprob, origin = lidar_scene_soft(self.nusc, token, self.soft_cache, self.sweeps)
            keep = (sidx[:, 0] >= 0) & (sidx[:, 0] != FREE)
            pts, plab, pprob = pts_all[keep], sidx[keep], sprob[keep]      # plab (N,K), pprob (N,K)
        else:                                                 # HARD: argmax class
            pts_all, lab, origin = lidar_scene(self.nusc, token, self.cache, self.sweeps)
            keep = (lab >= 0) & (lab != FREE)
            pts, plab, pprob = pts_all[keep], lab[keep], None
        if len(pts):
            v0, _ = voxel_indices(pts)
            V, inv_var = self._aniso(pts) if self.anisotropic else (None, None)
            inv2s2 = None if self.anisotropic else 1.0 / (2.0 * self._sigma(pts) ** 2)
            for off in self.offsets:
                vi = v0 + off
                ok = np.all((vi >= 0) & (vi < np.asarray(GRID_SIZE)), 1)
                if not ok.any():
                    continue
                vok, pok = vi[ok], pts[ok]
                delta = pok - (grid0 + (vok + 0.5) * VOXEL_SIZE)
                if self.anisotropic:
                    proj = np.einsum("mj,mji->mi", delta, V[ok])
                    g = np.exp(-0.5 * (proj ** 2 * inv_var[ok]).sum(1)).astype(np.float32)
                else:
                    g = np.exp(-(delta ** 2).sum(1) * inv2s2[ok]).astype(np.float32)
                flat = (vok[:, 0] * gy + vok[:, 1]) * gz + vok[:, 2]
                if self.soft_cache:                           # splat the point's class distribution
                    for kk in range(plab.shape[1]):
                        ck = plab[ok, kk]; val = (ck >= 0) & (ck < FREE)   # drop free (17) from mass
                        np.add.at(cls_mass, (flat[val], ck[val]), g[val] * pprob[ok, kk][val])
                    np.add.at(tot_mass, flat, g)
                else:
                    np.add.at(cls_mass, (flat, plab[ok]), g)
                    np.add.at(tot_mass, flat, g)
        sem = np.full(gx * gy * gz, FREE, np.uint8)
        occ = tot_mass > 0.1
        sem[occ] = cls_mass[occ].argmax(1).astype(np.uint8)
        conf = 1.0 - np.exp(-tot_mass / self.mass_tau)
        centres = (grid0 + (np.stack(np.unravel_index(np.arange(gx * gy * gz), (gx, gy, gz)), 1) + 0.5) * VOXEL_SIZE)
        rng = np.linalg.norm(centres[:, :2], axis=1)
        rdisc = np.clip(1.0 - (rng - self.range_full) / (self.range_far - self.range_full), self.min_conf, 1.0)
        weight = np.where(occ, np.clip(conf * rdisc, 0.0, 1.0), 0.0).astype(np.float32)
        soft_idx = soft_prob = None
        if self.soft_cache:
            si, sp = topk_from_mass(cls_mass, occ, K=plab.shape[1] if len(pts) else 3)
            soft_idx = si.reshape(gx, gy, gz, -1); soft_prob = sp.reshape(gx, gy, gz, -1)
        sem = sem.reshape(GRID_SIZE); weight = weight.reshape(GRID_SIZE)
        free = free_space_mask(pts_all, origin) & (sem == FREE)
        weight[free] = np.maximum(weight[free], self.free_w)
        return TeacherTarget(sem, weight, soft_idx, soft_prob)
