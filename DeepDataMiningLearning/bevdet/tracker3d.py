"""
Hungarian 3D Multi-Object Tracker for nuScenes.

Consumes per-frame 3D detections (produced by our B10c / unified inference
loop) and produces nuScenes tracking-format JSON ready for the official
TrackingEval (AMOTA / AMOTP).

Pure NumPy + scipy — no torch, no GPU. Tracking is a post-processing
layer on top of detections, run once over the saved JSONs in
``/tmp/unified_b10c/``.

Design notes
------------
* **Center-distance matching with per-class thresholds.** Matches the
  nuScenes tracking-eval metric (which uses 0.5/1.0/2.0/4.0 m
  center-distance), so the assignment cost is directly compatible with
  the metric's notion of "match".
* **Velocity-aware prediction**. Each tracklet carries a per-axis
  velocity in the **global** frame. Before matching, we propagate the
  last bbox forward by ``v · dt``. This is exactly the per-cell
  velocity our FG-TCA flow head learns to predict; the bbox head's
  velocity (last 2 dims of the 9-d bbox in LiDAR frame) is what we read
  from the detection JSON. **Toggle ``use_velocity=False`` to ablate**.
* **Birth / death**. A new detection that doesn't match any tracklet
  spawns a new tracklet *only* if its score >= ``birth_score``. A
  tracklet missing for >= ``max_age`` frames is killed.
* **No class switching**. Class is fixed at birth; cross-class matches
  are forbidden (per nuScenes-tracking convention).

The 10-class detection taxonomy → 7-class tracking taxonomy mapping
matches the official nuScenes tracking benchmark:
    car, truck, bus, trailer, pedestrian, motorcycle, bicycle
We DROP construction_vehicle, traffic_cone, barrier (not tracked).
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# -----------------------------------------------------------------------------
# Taxonomies and per-class match thresholds
# -----------------------------------------------------------------------------

# nuScenes 10 detection classes (canonical order)
DET_CLASSES_10 = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# nuScenes 7 tracking classes
TRACKING_CLASSES_7 = [
    'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian',
]

# Detection -> tracking-class name (None = skip — not tracked)
DET_TO_TRACK = {
    'car':                'car',
    'truck':              'truck',
    'construction_vehicle': None,
    'bus':                'bus',
    'trailer':            'trailer',
    'barrier':            None,
    'motorcycle':         'motorcycle',
    'bicycle':            'bicycle',
    'pedestrian':         'pedestrian',
    'traffic_cone':       None,
}

# Per-class center-distance threshold (meters) for tracklet↔detection
# matching. Chosen to be slightly tighter than nuScenes' loosest eval
# threshold (4 m) so we don't grossly over-match, but loose enough that
# 1 frame of missed detection doesn't kill the tracklet.
PER_CLASS_DIST_THRESH = {
    'car':        2.0,
    'truck':      2.5,
    'bus':        3.0,
    'trailer':    3.0,
    'motorcycle': 1.5,
    'bicycle':    1.5,
    'pedestrian': 1.0,
}


# -----------------------------------------------------------------------------
# Tracklet
# -----------------------------------------------------------------------------

@dataclass
class Tracklet:
    track_id: int
    class_name: str                 # 7-class tracking name
    last_box: np.ndarray            # [x, y, z, w, l, h, yaw] in GLOBAL frame
    velocity: np.ndarray            # [vx, vy] in GLOBAL frame, m/s
    last_score: float
    age: int = 0                    # frames since last successful match
    # History for nuScenes output:
    # list of (sample_token, np.ndarray[9] global, score) tuples.
    history: List[Tuple[str, np.ndarray, float]] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Tracker
# -----------------------------------------------------------------------------

class HungarianTracker3D:
    """
    Per-class Hungarian matching tracker. Stateful: one instance per scene.

    Usage::

        tracker = HungarianTracker3D(use_velocity=True)
        for sample_token, dets in scene_iter:
            tracker.update(dets, dt, sample_token)
        out = tracker.finalize_history()
    """

    def __init__(self,
                 use_velocity: bool = True,
                 max_age: int = 4,
                 birth_score: float = 0.20,
                 retention_score: float = 0.10):
        """
        Args:
            use_velocity:    if True, propagate tracklet position by v·dt
                             between frames. Set False for an IoU-only
                             ablation that doesn't use velocity GT
                             (the natural ablation for our FG-TCA flow head).
            max_age:         tracklet is killed after this many missed frames.
            birth_score:     new tracklets spawn only if det score >= this.
            retention_score: existing tracklets are kept alive even on missed
                             frames if their last_score >= this.
        """
        self.use_velocity = use_velocity
        self.max_age = int(max_age)
        self.birth_score = float(birth_score)
        self.retention_score = float(retention_score)

        self.tracklets: List[Tracklet] = []
        self._dead_list: List[Tracklet] = []   # killed tracklets, kept for output
        self.next_id: int = 0

    # ------------------------------------------------------------------
    # Main entry point: feed one frame of detections.
    # ------------------------------------------------------------------

    def update(self,
               detections: List[Dict],
               dt_seconds: float,
               sample_token: str) -> None:
        """
        Args:
            detections: list of dicts, each with keys:
                * 'box_global'  — np.ndarray [9] (x,y,z,w,l,h,yaw,vx,vy in GLOBAL frame)
                * 'score'       — float
                * 'class_track' — tracking-class name (must be in TRACKING_CLASSES_7)
            dt_seconds: seconds since last update (typically 0.5 for nuScenes).
            sample_token: nuScenes sample token for this frame.
        """
        # 1) Predict tracklets forward by dt.
        if self.use_velocity:
            for t in self.tracklets:
                t.last_box[0] += float(t.velocity[0]) * dt_seconds
                t.last_box[1] += float(t.velocity[1]) * dt_seconds
        for t in self.tracklets:
            t.age += 1

        # 2) Per-class Hungarian matching.
        matched, unmatched_det = self._match_per_class(detections)

        # 3) Update matched tracklets.
        for det_idx, trk_idx in matched:
            d = detections[det_idx]
            t = self.tracklets[trk_idx]
            t.last_box = d['box_global'].copy()
            # update velocity from detection (last 2 dims) — bbox head's
            # learned per-object velocity prediction
            t.velocity = d['box_global'][7:9].copy().astype(np.float64)
            t.last_score = float(d['score'])
            t.age = 0
            t.history.append((sample_token, d['box_global'].copy(), float(d['score'])))

        # 4) Birth new tracklets for unmatched, high-confidence detections.
        for det_idx in unmatched_det:
            d = detections[det_idx]
            if d['score'] < self.birth_score:
                continue
            t = Tracklet(
                track_id=self.next_id,
                class_name=d['class_track'],
                last_box=d['box_global'].copy(),
                velocity=d['box_global'][7:9].copy().astype(np.float64),
                last_score=float(d['score']),
                age=0,
            )
            t.history.append((sample_token, d['box_global'].copy(), float(d['score'])))
            self.tracklets.append(t)
            self.next_id += 1

        # 5) Kill tracklets that have been missing too long, unless they
        #    were very confident at last sighting (keep until max_age).
        #    Killed tracklets are moved to _dead_list so their history
        #    survives finalize_history().
        survivors: List[Tracklet] = []
        for t in self.tracklets:
            if t.age == 0:
                survivors.append(t)
            elif t.age <= self.max_age and t.last_score >= self.retention_score:
                survivors.append(t)
            else:
                self._dead_list.append(t)
        self.tracklets = survivors

    # ------------------------------------------------------------------
    # Per-class Hungarian assignment.
    # ------------------------------------------------------------------

    def _match_per_class(self,
                          detections: List[Dict]
                          ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Returns (matched_pairs, unmatched_detection_indices)."""
        matched: List[Tuple[int, int]] = []
        used_det = set()

        for cls_name in TRACKING_CLASSES_7:
            det_idxs = [i for i, d in enumerate(detections)
                        if d['class_track'] == cls_name]
            trk_idxs = [j for j, t in enumerate(self.tracklets)
                        if t.class_name == cls_name]
            if not det_idxs or not trk_idxs:
                continue

            dist_thresh = PER_CLASS_DIST_THRESH[cls_name]
            # Build cost matrix (large value = "no match"). We pad with
            # 1e6 outside the threshold so the Hungarian solver still
            # finds a feasible assignment.
            cost = np.full((len(det_idxs), len(trk_idxs)),
                           fill_value=1e6, dtype=np.float64)
            for i, di in enumerate(det_idxs):
                for j, tj in enumerate(trk_idxs):
                    d_xy = detections[di]['box_global'][:2]
                    t_xy = self.tracklets[tj].last_box[:2]
                    dist = float(np.linalg.norm(d_xy - t_xy))
                    if dist < dist_thresh:
                        cost[i, j] = dist
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < dist_thresh:
                    matched.append((det_idxs[r], trk_idxs[c]))
                    used_det.add(det_idxs[r])

        unmatched_det = [i for i in range(len(detections)) if i not in used_det]
        return matched, unmatched_det

    # ------------------------------------------------------------------
    # Final output: walk every tracklet's history, emit nuScenes
    # tracking-format records (per sample_token).
    # ------------------------------------------------------------------

    def finalize_history(self) -> Dict[str, List[Dict]]:
        """
        Returns a dict mapping ``sample_token -> list of nuScenes-tracking
        records``. Each record::

            {'sample_token': str,
             'translation':  [x, y, z],   # global frame
             'size':         [w, l, h],
             'rotation':     [w, x, y, z], # quaternion (yaw-only here)
             'velocity':     [vx, vy],
             'tracking_id':  str,
             'tracking_name': str,
             'tracking_score': float}
        """
        by_token: Dict[str, List[Dict]] = {}
        for t in self.tracklets + self._dead:    # include killed history too
            tid = f't{t.track_id}'
            for (tok, box, sc) in t.history:
                yaw = float(box[6])
                rec = {
                    'sample_token': tok,
                    'translation':  [float(box[0]), float(box[1]), float(box[2])],
                    'size':         [float(box[3]), float(box[4]), float(box[5])],
                    'rotation':     _yaw_to_quat_wxyz(yaw),
                    'velocity':     [float(box[7]), float(box[8])],
                    'tracking_id':  tid,
                    'tracking_name': t.class_name,
                    'tracking_score': float(sc),
                }
                by_token.setdefault(tok, []).append(rec)
        return by_token

    # Internal storage for killed tracklets so finalize_history sees them.
    @property
    def _dead(self):
        return getattr(self, '_dead_list', [])


def _yaw_to_quat_wxyz(yaw: float) -> List[float]:
    """Yaw (around +Z) → quaternion (w, x, y, z)."""
    half = 0.5 * yaw
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


# -----------------------------------------------------------------------------
# Geometry helpers — LiDAR-frame box → global-frame box.
# -----------------------------------------------------------------------------

def lidar_box_to_global(
    box_lidar: np.ndarray,
    lidar2ego_R: np.ndarray, lidar2ego_t: np.ndarray,
    ego2global_R: np.ndarray, ego2global_t: np.ndarray,
) -> np.ndarray:
    """
    Transform a single [9] LiDAR-frame box (x,y,z,w,l,h,yaw,vx,vy) into the
    GLOBAL frame using the per-frame extrinsics from nuScenes
    ``calibrated_sensor`` (lidar2ego) and ``ego_pose`` (ego2global).

    All rotations are 3x3 matrices; translations are 3-vectors.

    Position: ego2global_R @ (lidar2ego_R @ p_lidar + lidar2ego_t) + ego2global_t
    Yaw:      yaw_global = yaw_lidar + yaw(ego2global) + yaw(lidar2ego)
              (compose around +Z; we extract atan2 of the composed R)
    Velocity: rotate (vx,vy,0) by ego2global_R @ lidar2ego_R, keep XY.

    Output box: [x, y, z, w, l, h, yaw_global, vx_g, vy_g]
    """
    p_lidar = box_lidar[:3]
    p_ego = lidar2ego_R @ p_lidar + lidar2ego_t
    p_global = ego2global_R @ p_ego + ego2global_t

    R_l2g = ego2global_R @ lidar2ego_R              # [3, 3]
    # Local +x in LiDAR → global direction → atan2 gives composed yaw.
    yaw_lidar = float(box_lidar[6])
    cos_y, sin_y = float(np.cos(yaw_lidar)), float(np.sin(yaw_lidar))
    local_forward_global = R_l2g @ np.array([cos_y, sin_y, 0.0])
    yaw_global = float(np.arctan2(local_forward_global[1], local_forward_global[0]))

    v_lidar = np.array([float(box_lidar[7]), float(box_lidar[8]), 0.0])
    v_global = R_l2g @ v_lidar

    return np.array([
        float(p_global[0]), float(p_global[1]), float(p_global[2]),
        float(box_lidar[3]), float(box_lidar[4]), float(box_lidar[5]),
        yaw_global,
        float(v_global[0]), float(v_global[1]),
    ], dtype=np.float64)


def boxes_lidar_to_global(boxes_lidar: np.ndarray,
                          lidar2ego_R, lidar2ego_t,
                          ego2global_R, ego2global_t) -> np.ndarray:
    """Vectorized version of ``lidar_box_to_global`` for an [N, 9] array."""
    if boxes_lidar.shape[0] == 0:
        return boxes_lidar
    out = np.zeros_like(boxes_lidar)
    for i in range(boxes_lidar.shape[0]):
        out[i] = lidar_box_to_global(
            boxes_lidar[i],
            lidar2ego_R, lidar2ego_t,
            ego2global_R, ego2global_t,
        )
    return out
