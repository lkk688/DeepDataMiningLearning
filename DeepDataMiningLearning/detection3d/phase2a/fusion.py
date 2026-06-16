"""
Fusion: match Validator-A (LiDAR clusters) ↔ Validator-B (Cam-2D-lift)
proposals, then escalate disagreements to the VLM.

Outputs a final list of pseudo-labels with per-instance confidence weight
suitable for downstream supervised fine-tuning.

Routing logic (matches the architecture table from the design):

  A says X, B says X (A∩B)          → keep, weight=1.0,  cls=X
  A says X, B silent                → MEDIUM, call VLM. VLM confirms X
                                       → keep, weight=0.5, cls=X
  A silent, B says X                → MEDIUM, call VLM. VLM confirms X
                                       → keep, weight=0.5, cls=X
  A says X, B says Y (conflict)     → call VLM, take VLM's class,
                                       → keep, weight=0.3
  Neither + no VLM signal           → drop
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cluster_proposer import ClusterProposal
from .cam2d_proposer import Cam2DProposal


@dataclass
class PseudoLabel:
    """Final pseudo-GT instance with bookkeeping."""
    box: np.ndarray          # (7,) vehicle frame
    cls: str
    weight: float            # 0-1; loss weight for supervised fine-tune
    source: str              # 'A∩B' | 'A+VLM' | 'B+VLM' | 'VLM-resolved'
    vlm_called: bool = False
    vlm_cls: Optional[str] = None
    vlm_conf: float = 0.0
    n_lidar: int = 0
    cam_slot: int = -1

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['box'] = self.box.tolist()
        return d


# ---------------------------------------------------------------- matching
def _match_3d_boxes(a_props: List[ClusterProposal],
                    b_props: List[Cam2DProposal],
                    dist_thresh: float = 2.5) -> List[Tuple[int, int]]:
    """Greedy 1-1 matching by center-distance. Returns pairs (a_idx, b_idx).

    We also require the class to match (otherwise it's not "A∩B agreement",
    it's a conflict — handled separately).
    """
    if not a_props or not b_props:
        return []
    A = np.array([p.box[:3] for p in a_props])
    B = np.array([p.box[:3] for p in b_props])
    dist = np.linalg.norm(A[:, None, :2] - B[None, :, :2], axis=-1)
    pairs = []
    used_a = set(); used_b = set()
    flat = np.argsort(dist.flatten())
    for f in flat:
        ai = int(f // dist.shape[1])
        bi = int(f %  dist.shape[1])
        if ai in used_a or bi in used_b:
            continue
        if dist[ai, bi] > dist_thresh:
            break
        # same class? must agree for A∩B; conflicting class = separate.
        if a_props[ai].cls != b_props[bi].cls:
            continue
        pairs.append((ai, bi))
        used_a.add(ai); used_b.add(bi)
    return pairs


def _conflict_pairs(a_props: List[ClusterProposal],
                    b_props: List[Cam2DProposal],
                    used_a: set, used_b: set,
                    dist_thresh: float = 2.5) -> List[Tuple[int, int]]:
    """Find class-conflict A↔B pairs that are spatially close but disagree."""
    if not a_props or not b_props:
        return []
    out = []
    A = np.array([p.box[:3] for p in a_props])
    B = np.array([p.box[:3] for p in b_props])
    dist = np.linalg.norm(A[:, None, :2] - B[None, :, :2], axis=-1)
    for ai in range(len(a_props)):
        if ai in used_a:
            continue
        bi = int(np.argmin(dist[ai]))
        if bi in used_b:
            continue
        if dist[ai, bi] <= dist_thresh and a_props[ai].cls != b_props[bi].cls:
            out.append((ai, bi))
    return out


# ------------------------------------------------------------------ project
def _project_box_to_image(box_3d: np.ndarray,
                          lidar2img_4x4: np.ndarray,
                          image_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Project a 3D box center → 2D crop box around it. Returns xyxy or None."""
    H, W = image_hw
    x, y, z, l, w, h, _yaw = box_3d
    # Sample 8 corners (axis-aligned for simplicity — we only need a
    # tight-ish image crop, not pixel-perfect).
    dx, dy, dz = l/2, w/2, h/2
    corners = np.array([
        [x+dx, y+dy, z+dz, 1], [x+dx, y+dy, z-dz, 1],
        [x+dx, y-dy, z+dz, 1], [x+dx, y-dy, z-dz, 1],
        [x-dx, y+dy, z+dz, 1], [x-dx, y+dy, z-dz, 1],
        [x-dx, y-dy, z+dz, 1], [x-dx, y-dy, z-dz, 1],
    ], dtype=np.float64)
    proj = corners @ lidar2img_4x4.T
    z_p = proj[:, 2]
    if (z_p <= 0.5).any():
        return None
    u = proj[:, 0] / z_p; v = proj[:, 1] / z_p
    if (u < -200).all() or (u > W + 200).all() or \
       (v < -200).all() or (v > H + 200).all():
        return None
    x1 = max(0, int(np.floor(u.min())))
    y1 = max(0, int(np.floor(v.min())))
    x2 = min(W, int(np.ceil(u.max())))
    y2 = min(H, int(np.ceil(v.max())))
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ------------------------------------------------------------ main entrypoint
def _dedupe_nms(labels: List['PseudoLabel'],
                dist_thresh: float = 2.0) -> List['PseudoLabel']:
    """Greedy NMS in BEV: keep the higher-weight label, drop overlapping ones.

    Weight tie-broken by source priority (A∩B > A+VLM/B+VLM > VLM-resolved).
    Within the same class, two labels closer than dist_thresh are merged
    (we keep one, drop the other).
    """
    if not labels:
        return []
    src_pri = {'A∩B': 3, 'A+VLM': 2, 'B+VLM': 2, 'VLM-resolved': 1}
    keys = [(-l.weight, -src_pri.get(l.source, 0), i) for i, l in enumerate(labels)]
    order = [i for _, _, i in sorted(keys)]
    kept: List['PseudoLabel'] = []
    for idx in order:
        l = labels[idx]
        dup = False
        for k in kept:
            if k.cls != l.cls:
                continue
            d = float(np.linalg.norm(l.box[:2] - k.box[:2]))
            if d < dist_thresh:
                dup = True
                break
        if not dup:
            kept.append(l)
    return kept


def fuse(a_props: List[ClusterProposal],
         b_props: List[Cam2DProposal],
         vlm_voter=None,
         images_by_slot: Optional[Dict[int, np.ndarray]] = None,
         lidar2img_by_slot: Optional[Dict[int, np.ndarray]] = None,
         dist_thresh: float = 2.5,
         vlm_min_conf: float = 0.7,
         primary_cam_slot: int = 0,
         nms_thresh: float = 2.0,
         range_max: float = 50.0,
         skip_a_only: bool = False,
         debug: bool = False) -> List[PseudoLabel]:
    """Fuse A and B proposals into a final pseudo-label set.

    Args:
      a_props: LiDAR cluster proposals (Validator A)
      b_props: Cam2D-lift proposals (Validator B)
      vlm_voter: optional VLMVoter for tier-2 voting on disagreement.
      images_by_slot: dict cam_slot → (H,W,3) uint8 image; needed for VLM.
      lidar2img_by_slot: dict cam_slot → 4x4 matrix for projection.
        Used to find a crop around an A-only proposal for VLM voting.
      vlm_min_conf: VLM must return at least this confidence to accept its
        positive vote. Below this, the proposal is dropped.
      primary_cam_slot: which cam to query when A says X but B silent
        (default = FRONT).
    """
    out: List[PseudoLabel] = []
    a_used = set(); b_used = set()

    # 1) A ∩ B agreement (same class, close in xy)
    pairs = _match_3d_boxes(a_props, b_props, dist_thresh=dist_thresh)
    for ai, bi in pairs:
        a = a_props[ai]; b = b_props[bi]
        # Box: average centroid; size from B (camera-prior); yaw from A (PCA)
        cx = (a.box[0] + b.box[0]) / 2
        cy = (a.box[1] + b.box[1]) / 2
        cz = (a.box[2] + b.box[2]) / 2
        # Prefer A's geometry (it sees actual extent) when n_pts >= 15
        if a.n_pts >= 15:
            l, w, h = a.box[3], a.box[4], a.box[5]
        else:
            l, w, h = b.box[3], b.box[4], b.box[5]
        yaw = float(a.box[6])
        out.append(PseudoLabel(
            box=np.array([cx, cy, cz, l, w, h, yaw], dtype=np.float32),
            cls=a.cls, weight=1.0, source='A∩B',
            n_lidar=a.n_pts, cam_slot=int(b.cam_slot),
        ))
        a_used.add(ai); b_used.add(bi)

    # 2) Class conflicts (A says X, B says Y, but spatially co-located)
    conflicts = _conflict_pairs(a_props, b_props, a_used, b_used,
                                dist_thresh=dist_thresh)
    for ai, bi in conflicts:
        a = a_props[ai]; b = b_props[bi]
        a_used.add(ai); b_used.add(bi)
        if vlm_voter is None or images_by_slot is None:
            continue
        img = images_by_slot.get(int(b.cam_slot))
        if img is None:
            continue
        cls_vlm, conf = vlm_voter.vote(img, b.box_2d)
        if cls_vlm in ('Vehicle', 'Pedestrian', 'Cyclist') and conf >= vlm_min_conf:
            # Use B's geometry (camera-anchored), VLM's class
            out.append(PseudoLabel(
                box=b.box.copy(), cls=cls_vlm, weight=0.3,
                source='VLM-resolved', vlm_called=True,
                vlm_cls=cls_vlm, vlm_conf=conf,
                n_lidar=b.n_lidar_inside, cam_slot=int(b.cam_slot),
            ))

    # 3) A-only (LiDAR cluster, no 2D-detector hit) — ask VLM
    for ai, a in enumerate(a_props):
        if ai in a_used:
            continue
        if skip_a_only:
            continue
        if vlm_voter is None or images_by_slot is None or \
           lidar2img_by_slot is None:
            continue
        # Find a camera that contains this 3D box. Prefer primary cam slot.
        slot_order = [primary_cam_slot] + [s for s in images_by_slot
                                            if s != primary_cam_slot]
        for slot in slot_order:
            img = images_by_slot.get(slot)
            l2i = lidar2img_by_slot.get(slot)
            if img is None or l2i is None:
                continue
            crop_xyxy = _project_box_to_image(a.box, l2i, img.shape[:2])
            if crop_xyxy is None:
                continue
            cls_vlm, conf = vlm_voter.vote(img, crop_xyxy)
            if cls_vlm == a.cls and conf >= vlm_min_conf:
                out.append(PseudoLabel(
                    box=a.box.copy(), cls=a.cls, weight=0.5,
                    source='A+VLM', vlm_called=True,
                    vlm_cls=cls_vlm, vlm_conf=conf,
                    n_lidar=a.n_pts, cam_slot=int(slot),
                ))
                break
            elif cls_vlm in ('Vehicle', 'Pedestrian', 'Cyclist') and \
                 conf >= vlm_min_conf:
                # VLM disagrees with A's class but confirms an object exists.
                # Take VLM's class with A's geometry.
                out.append(PseudoLabel(
                    box=a.box.copy(), cls=cls_vlm, weight=0.3,
                    source='VLM-resolved', vlm_called=True,
                    vlm_cls=cls_vlm, vlm_conf=conf,
                    n_lidar=a.n_pts, cam_slot=int(slot),
                ))
                break
            else:
                # VLM says None — drop. Don't try other cameras.
                break

    # 4) B-only (2D detector hit, no LiDAR cluster) — ask VLM if very small
    # cluster might have been missed; otherwise B's confidence often
    # already came from a strong COCO prior.
    for bi, b in enumerate(b_props):
        if bi in b_used:
            continue
        if vlm_voter is None or images_by_slot is None:
            continue
        img = images_by_slot.get(int(b.cam_slot))
        if img is None:
            continue
        cls_vlm, conf = vlm_voter.vote(img, b.box_2d)
        if cls_vlm == b.cls and conf >= vlm_min_conf:
            out.append(PseudoLabel(
                box=b.box.copy(), cls=b.cls, weight=0.5,
                source='B+VLM', vlm_called=True,
                vlm_cls=cls_vlm, vlm_conf=conf,
                n_lidar=b.n_lidar_inside, cam_slot=int(b.cam_slot),
            ))

    # 5) Range filter (drop labels beyond range_max so we stay in the
    # training distribution of the source model — nuScenes annotates to 50m).
    out = [p for p in out
           if float(np.linalg.norm(p.box[:2])) <= range_max]

    # 6) NMS dedup (two close pseudo-labels of same class → keep best)
    out = _dedupe_nms(out, dist_thresh=nms_thresh)

    if debug:
        from collections import Counter
        src_counts = Counter(p.source for p in out)
        cls_counts = Counter(p.cls for p in out)
        print(f'[fusion] {len(out)} pseudo-labels  sources={dict(src_counts)}  classes={dict(cls_counts)}')

    return out
