"""
Validator A — LiDAR DBSCAN cluster proposer.

Takes a LiDAR sweep (N, 5: x, y, z, intensity, time) in vehicle frame and
returns 3D box candidates with a coarse class guess from cluster geometry.

Pipeline:
  1. Ego/ROI filter: drop points behind the car or beyond max range
  2. Ground removal: simple z-cutoff per (x,y) tile (RANSAC-plane optional)
  3. DBSCAN on remaining points (xy-only distance — z is implicit via tile)
  4. For each cluster: fit oriented bounding box (PCA on xy → yaw,
     centroid xyz, lwh from min/max along principal axes)
  5. Classify by size (height/length aspect): Vehicle / Pedestrian / Cyclist
  6. Reject by min-points, min-volume, plausible aspect

Outputs a list of proposals: each is a dict with keys
  box     (7,)  float32  [x, y, z, l, w, h, yaw] in vehicle frame
  cls     str           'Vehicle' / 'Pedestrian' / 'Cyclist'
  n_pts   int           number of points in cluster
  score   float         heuristic confidence (n_pts × density factor)

This is "Validator A" — the LiDAR-side validator. High recall on anything
with mass; no idea about semantic class except from geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
from sklearn.cluster import DBSCAN

TRANSFER_CLASSES = ('Vehicle', 'Pedestrian', 'Cyclist')


@dataclass
class ClusterProposal:
    box: np.ndarray         # (7,) [x, y, z, l, w, h, yaw]
    cls: str
    n_pts: int
    score: float            # 0-1 heuristic
    rng: float              # distance from ego in xy plane

    def to_dict(self) -> Dict:
        return dict(box=self.box.tolist(), cls=self.cls,
                    n_pts=int(self.n_pts), score=float(self.score),
                    rng=float(self.rng))


# ---- Size priors used for cluster -> class classification ------------------
# (l, w, h) ranges; mostly to reject noise blobs / huge ground patches.
# These are *generous* — the goal is recall, the VLM/2D detector clears noise.
SIZE_PRIORS = {
    'Vehicle':    dict(l=(2.0, 25.0), w=(0.6, 4.0), h=(0.8, 5.5)),
    'Pedestrian': dict(l=(0.15, 1.5), w=(0.15, 1.5), h=(0.7, 2.4)),
    'Cyclist':    dict(l=(0.7, 2.5), w=(0.2, 1.5), h=(0.7, 2.4)),
}


def _classify_cluster(l: float, w: float, h: float) -> Optional[str]:
    """Coarse rule-based class from cluster dimensions.

    Priorities (matching how Waymo distinguishes them):
      - Pedestrian: tall+thin, h>1.4 and max(l,w)<1.2
      - Cyclist:    intermediate, 1.2<max(l,w)<2.5 and 1.2<h<2.2
      - Vehicle:    everything else with l>2 or w>1.4 within size priors
    Returns class name or None if cluster matches no plausible class.
    """
    L = max(l, w)
    S = min(l, w)
    # Pedestrian first (most constrained)
    if 0.7 <= h <= 2.4 and L <= 1.2 and S <= 1.2:
        return 'Pedestrian'
    if 0.7 <= h <= 2.2 and 0.8 <= L <= 2.5 and S <= 1.5:
        return 'Cyclist'
    if (2.0 <= L <= 25.0 and 0.6 <= S <= 4.0 and 0.8 <= h <= 5.5):
        return 'Vehicle'
    return None


def _ground_mask_tile(points_xyz: np.ndarray,
                      tile_size: float = 5.0,
                      ground_quantile: float = 0.05,
                      above_ground: float = 0.20) -> np.ndarray:
    """Tile the xy plane; mark points within ``above_ground`` of the
    per-tile ``ground_quantile``-th percentile z as ground.

    Returns a boolean mask the size of ``points_xyz`` (True = ground).
    """
    if points_xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    x = points_xyz[:, 0]; y = points_xyz[:, 1]; z = points_xyz[:, 2]
    ix = np.floor(x / tile_size).astype(np.int64)
    iy = np.floor(y / tile_size).astype(np.int64)
    # Pack (ix,iy) into a single key
    key = (ix - ix.min()) * (iy.max() - iy.min() + 1) + (iy - iy.min())
    is_ground = np.zeros_like(z, dtype=bool)
    # Per-tile ground z = ground_quantile percentile
    uniq, inverse = np.unique(key, return_inverse=True)
    for tile_idx in range(uniq.size):
        m = inverse == tile_idx
        if m.sum() < 5:
            continue
        z_t = z[m]
        z_g = np.quantile(z_t, ground_quantile)
        is_ground[m] = z[m] < (z_g + above_ground)
    return is_ground


def propose_clusters(points: np.ndarray,
                     ego_radius: float = 1.5,
                     max_range: float = 60.0,
                     min_points: int = 8,
                     eps: float = 0.6,
                     z_max: float = 5.5,
                     z_min: float = -2.0) -> List[ClusterProposal]:
    """Return a list of ``ClusterProposal`` for the given LiDAR sweep.

    Args:
      points: (N, >=3) array in vehicle frame. Only first 3 cols used.
      ego_radius: drop points with ``sqrt(x^2+y^2) < ego_radius`` (the
        car itself / its mirrors).
      max_range: drop points with ``sqrt(x^2+y^2) > max_range``. We use
        50 m by default to match the nuScenes annotation range.
      min_points: minimum points per DBSCAN cluster.
      eps: DBSCAN epsilon in meters.
      z_max / z_min: clip extreme z (overhead structures / underbody noise).
    """
    if points is None or len(points) == 0:
        return []

    xyz = np.asarray(points, dtype=np.float32)[:, :3]

    # 1) Ego + range + z-band filter
    r = np.linalg.norm(xyz[:, :2], axis=1)
    keep = (r > ego_radius) & (r < max_range) & \
           (xyz[:, 2] > z_min) & (xyz[:, 2] < z_max)
    xyz = xyz[keep]
    if xyz.shape[0] < min_points:
        return []

    # 2) Ground removal (tile-based)
    ground_mask = _ground_mask_tile(xyz)
    xyz_obj = xyz[~ground_mask]
    if xyz_obj.shape[0] < min_points:
        return []

    # 3) DBSCAN in xy with z-extent considered via a 3D-like distance.
    # Cheap trick: cluster on 3D coords directly (eps in meters works
    # for the typical object scale).
    db = DBSCAN(eps=eps, min_samples=max(3, min_points // 3))
    labels = db.fit_predict(xyz_obj)

    proposals: List[ClusterProposal] = []
    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        m = labels == lbl
        pts = xyz_obj[m]
        if pts.shape[0] < min_points:
            continue

        # 4) Oriented box via PCA on xy
        xy = pts[:, :2]
        c_xy = xy.mean(0)
        xy_c = xy - c_xy
        cov = xy_c.T @ xy_c / max(1, xy_c.shape[0] - 1)
        try:
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        # Largest eigenvector = principal axis
        v = evecs[:, np.argmax(evals)]
        yaw = float(np.arctan2(v[1], v[0]))

        # Project points onto principal/orthogonal axes
        R = np.array([[np.cos(yaw),  np.sin(yaw)],
                      [-np.sin(yaw), np.cos(yaw)]], dtype=np.float32)
        xy_local = xy_c @ R.T
        l = float(xy_local[:, 0].max() - xy_local[:, 0].min())
        w = float(xy_local[:, 1].max() - xy_local[:, 1].min())

        z_min_pt = float(pts[:, 2].min())
        z_max_pt = float(pts[:, 2].max())
        h = z_max_pt - z_min_pt
        z_center = (z_min_pt + z_max_pt) * 0.5

        cls = _classify_cluster(l, w, h)
        if cls is None:
            continue

        # Heuristic score: tied to point count + matches a tight size band
        n = int(pts.shape[0])
        score = float(min(1.0, n / 30.0))   # 30+ pts → saturate
        rng = float(np.linalg.norm(c_xy))

        proposals.append(ClusterProposal(
            box=np.array([c_xy[0], c_xy[1], z_center, l, w, h, yaw],
                         dtype=np.float32),
            cls=cls, n_pts=n, score=score, rng=rng,
        ))

    return proposals


def proposals_to_dict_list(props: List[ClusterProposal]) -> List[Dict]:
    return [p.to_dict() for p in props]
