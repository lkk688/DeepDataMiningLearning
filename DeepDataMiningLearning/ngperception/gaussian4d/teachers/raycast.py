"""
gaussian4d.teachers.raycast
==========================
Ray-aware free-space carving (staged build #1). A LiDAR beam that ends at a surface **verifies every
voxel it passed through as free** — this is real free-space supervision, and it also lets us mark
never-hit voxels as **unknown** (unobserved) so the student is *not* penalised there. Shared by
both teachers so the Gaussian-vs-voxel gate stays a clean representation comparison.

Occupancy state per voxel:  occupied (a surface point lands in it) · free (a ray passes through) ·
unknown (neither). The teacher supervises occupied + free, ignores unknown (weight 0).
"""
from __future__ import annotations
import numpy as np

from .base import voxel_indices
from ...occupancy.geom import VOXEL_SIZE, GRID_SIZE


def free_space_mask(points_ego, origin, max_range=60.0):
    """Voxels a LiDAR ray traverses before its hit = ray-verified free. -> (X,Y,Z) bool.
    `origin` = LiDAR sensor position in the ego frame; we march each ray in voxel-size steps and
    stop half a voxel short of the endpoint (the hit voxel is occupied, not free)."""
    gx, gy, gz = [int(v) for v in GRID_SIZE]
    free = np.zeros(gx * gy * gz, bool)
    if len(points_ego) == 0:
        return free.reshape((gx, gy, gz))
    d = points_ego - origin[None]
    dist = np.linalg.norm(d, axis=1)
    ok = dist > VOXEL_SIZE
    d, dist = d[ok], dist[ok]
    if len(dist) == 0:
        return free.reshape((gx, gy, gz))
    unit = d / dist[:, None]
    maxd = float(dist.max())
    for si in range(1, int(max_range / VOXEL_SIZE)):
        s = si * VOXEL_SIZE
        if s >= maxd:
            break
        active = s < (dist - 0.5 * VOXEL_SIZE)                  # rays still short of their hit
        if not active.any():
            continue
        idx, m = voxel_indices(origin[None] + unit[active] * s)
        if m.any():
            fl = (idx[m, 0] * gy + idx[m, 1]) * gz + idx[m, 2]
            free.flat[fl] = True
    return free.reshape((gx, gy, gz))
