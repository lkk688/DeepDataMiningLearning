"""
gaussian4d.teachers.dynamic_teacher
==================================
Label-free occ pseudo-label with the KEY Occ3D ingredient our old voxel-soft lacked:
**dynamic/static separation**. Occ3D accumulates multi-frame LiDAR (dense geometry) but uses human 3D
boxes to pull *moving* objects out and place them sharply (else accumulation smears them into
streaks). Our voxel-soft had multi-sweep + FM semantics but NO dynamic handling → smeared foreground
→ null detection transfer. This teacher adds it:

  static  = multi-sweep points NOT in any box  -> FM semantics (labelgen)     -> dense static occ
  dynamic = current-sweep points IN each box   -> the box's class (sharp)     -> overrides static
  free    = ray-cast from the sensor origin

`boxes="gt"` uses nuScenes 3D boxes (upper bound of the recipe — isolates the dynamic/static effect
from the box-source); the label-free version swaps in pseudo-tracks (zero-shot detector) with the same
code path. Output = the standard `TeacherTarget` (reuses the occ trainer/eval unchanged).
"""
from __future__ import annotations
import numpy as np

from .base import TeacherTarget, voxel_indices, FREE, NUM_CLASSES
from .semantics import load_lidar_ego, assign_semantics
from .raycast import free_space_mask
from ...occupancy.geom import GRID_SIZE

# nuScenes det-10 (car,truck,constr,bus,trailer,barrier,moto,bike,ped,cone) -> Occ3D semantic id
DET10_TO_OCC3D = {0: 4, 1: 10, 2: 5, 3: 3, 4: 9, 5: 1, 6: 6, 7: 2, 8: 7, 9: 8}


def ego_boxes(nusc, token, det_class_map):
    """3D boxes LiDAR->ego (the occ grid frame): (M,8) [x,y,z,l,w,h,yaw, det10_label]."""
    from pyquaternion import Quaternion
    from ...detection.nuscenes_dataset import NUSC_10CLASS
    s = nusc.get("sample", token)
    lsd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
    _, boxes, _ = nusc.get_sample_data(lsd["token"])            # boxes in LiDAR frame
    lcs = nusc.get("calibrated_sensor", lsd["calibrated_sensor_token"])
    Rl, tl = Quaternion(lcs["rotation"]), np.array(lcs["translation"])
    out = []
    for b in boxes:
        cls = None                                             # nuScenes category -> det-10 id
        for pref, idx in det_class_map.items():
            if b.name.startswith(pref) or pref in b.name:
                cls = idx; break
        if cls is None:
            continue
        b.rotate(Rl); b.translate(tl)                          # LiDAR -> ego
        x, y, z = b.center; w, l, h = b.wlh
        v = b.orientation.rotation_matrix[:, 0]
        out.append([x, y, z, l, w, h, float(np.arctan2(v[1], v[0])), float(cls)])
    return np.array(out, np.float32).reshape(-1, 8)


def points_in_box(pts, box, margin=0.1):
    """Boolean mask of pts (N,3) inside an oriented box [x,y,z,l,w,h,yaw,...] (+margin)."""
    c = box[:3]; l, w, h = box[3:6]; yaw = box[6]
    d = pts - c
    ca, sa = np.cos(-yaw), np.sin(-yaw)
    px = ca * d[:, 0] - sa * d[:, 1]                           # rotate into box frame
    py = sa * d[:, 0] + ca * d[:, 1]
    return (np.abs(px) <= l / 2 + margin) & (np.abs(py) <= w / 2 + margin) & \
           (np.abs(d[:, 2]) <= h / 2 + margin)


class DynamicOccTeacher:
    def __init__(self, nusc, labelgen_cache, sweeps=10, boxes="gt", free_w=0.02):
        self.nusc, self.cache, self.sweeps, self.free_w = nusc, labelgen_cache, sweeps, free_w
        self.boxes = boxes
        from ...detection.nuscenes_dataset import NUSC_10CLASS
        self.det_map = NUSC_10CLASS

    def __call__(self, token) -> TeacherTarget:
        gx, gy, gz = [int(v) for v in GRID_SIZE]; V = gx * gy * gz
        sem = np.full(V, FREE, np.uint8); weight = np.zeros(V, np.float32)
        # --- geometry: multi-sweep (static) + single-sweep (dynamic, sharp) ---
        pts_multi, _, origin = load_lidar_ego(self.nusc, token, self.sweeps)
        pts_single, _, _ = load_lidar_ego(self.nusc, token, 1)
        bx = ego_boxes(self.nusc, token, self.det_map) if self.boxes == "gt" else \
            np.zeros((0, 8), np.float32)                       # pseudo-track hook (same path)
        in_box_multi = np.zeros(len(pts_multi), bool)
        for b in bx:
            in_box_multi |= points_in_box(pts_multi, b)
        static_pts = pts_multi[~in_box_multi]                  # drop smeared dynamic points
        # --- static semantics from the FM (labelgen), majority class per voxel ---
        lab = assign_semantics(self.nusc, token, static_pts, self.cache)
        keep = (lab >= 0) & (lab != FREE)
        spts, slab = static_pts[keep], lab[keep]
        counts = np.zeros((V, NUM_CLASSES - 1), np.int32)
        if len(spts):
            idx, m = voxel_indices(spts); idx, slab = idx[m], slab[m]
            flat = (idx[:, 0] * gy + idx[:, 1]) * gz + idx[:, 2]
            np.add.at(counts, (flat, slab), 1)
            occ = counts.sum(1) > 0
            sem[occ] = counts[occ].argmax(1).astype(np.uint8); weight[occ] = 1.0
        # --- dynamic: current-sweep points inside each box -> box class (sharp, overrides) ---
        for b in bx:
            occ_cls = DET10_TO_OCC3D[int(b[7])]
            inb = points_in_box(pts_single, b)
            if not inb.any():
                continue
            idx, m = voxel_indices(pts_single[inb])
            if not m.any():
                continue
            flat = (idx[m, 0] * gy + idx[m, 1]) * gz + idx[m, 2]
            sem[flat] = occ_cls; weight[flat] = 1.0            # dynamic overrides static
        sem = sem.reshape(GRID_SIZE); weight = weight.reshape(GRID_SIZE)
        free = free_space_mask(pts_multi, origin) & (sem == FREE)
        weight[free] = np.maximum(weight[free], self.free_w)
        return TeacherTarget(sem, weight)
