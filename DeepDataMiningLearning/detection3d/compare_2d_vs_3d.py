"""
compare_2d_vs_3d.py
===================

Per-sample comparison of:
  * The 2D detections emitted by B11's ``image2d_head`` (one per camera).
  * The 2D boxes you get by **projecting the 3D detections** through
    ``lidar2img`` and taking the min/max of the 8 projected corners.

Goal: understand whether the 2D-aux head actually trained well in B11
*as a 2D detector*, given that the headline NDS went DOWN by 0.0044.

Two diagnostic questions:

  Q1. Does the 2D head detect objects that the 3D head missed?
      → "2D-only hits" — count + sample IDs.
      → if many: image features ARE more discriminative for some objects,
        but the FPN→BEV projection isn't carrying that signal.

  Q2. Does the 2D head miss objects that the 3D head finds?
      → "3D-only hits" — count.
      → if many: 2D head undertrained / weighted too low / class-imbalance.

Outputs:
  * Per-sample CSV under ``<out_root>/per_sample.csv``.
  * Overall summary printed to stdout + saved as ``summary.json``.

Usage:
    python compare_2d_vs_3d.py \\
        --runs-dir-3d /tmp/unified_b10c   \\
        --runs-dir-b11 /tmp/unified_b11   \\
        --out-dir /tmp/compare_2d_vs_3d   \\
        --iou-match 0.30                  \\
        --score-thr 0.20
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from glob import glob
from typing import Dict, List, Tuple

import numpy as np


# nuScenes 10-class label mapping (matches our configs).
CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two [N,4] / [M,4] xyxy arrays → [N,M]."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = (x2 - x1).clip(min=0)
    ih = (y2 - y1).clip(min=0)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]).clip(min=0) * (a[:, 3] - a[:, 1]).clip(min=0)
    bb = (b[:, 2] - b[:, 0]).clip(min=0) * (b[:, 3] - b[:, 1]).clip(min=0)
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def project_3d_box_to_camera(box_9d: np.ndarray,
                             lidar2img: np.ndarray,
                             img_aug: np.ndarray,
                             img_size: Tuple[int, int]) -> np.ndarray:
    """
    Project a single 3D box [x,y,z,w,l,h,yaw,vx,vy] to a 2D xyxy box in
    the **augmented image** (256x704). Returns ``[x1, y1, x2, y2]`` or
    ``None`` if the box is fully behind the camera.

    Convention: same as mmdet3d LiDARInstance3DBoxes.tensor:
      bottom-center coordinates, yaw around +Z.
    """
    x, y, z, w, l, h, yaw = box_9d[:7]
    cx, sy_ = np.cos(yaw), np.sin(yaw)
    # 8 corners in the box's local frame (BL = bottom-left bottom, etc.)
    # Box-local: x-axis = l (forward), y-axis = w (left), z-axis = h (up).
    dx = np.array([ l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2])
    dy = np.array([ w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2])
    dz = np.array([ 0,   0,   0,   0,   h,   h,   h,   h  ])
    # Rotate around Z, then translate to (x, y, z).
    R = np.array([[cx, -sy_, 0], [sy_, cx, 0], [0, 0, 1]])
    corners_local = np.stack([dx, dy, dz], axis=0)              # [3, 8]
    corners_world = R @ corners_local + np.array([[x], [y], [z]])  # [3, 8]
    corners_h = np.vstack([corners_world, np.ones((1, 8))])       # [4, 8]

    # lidar→image
    proj = lidar2img @ corners_h                                   # [4, 8]
    z_cam = proj[2, :]
    if (z_cam <= 0.1).all():
        return None
    valid = z_cam > 0.1
    pix = proj[:2, :] / np.maximum(z_cam[None, :], 1e-3)            # [2, 8]
    # Apply img_aug_matrix (256x704 augmentation transform). The matrix
    # is 4x4; only the 2x2 + translation top-left block is relevant for
    # 2D pixel coords.
    R = img_aug[:2, :2]               # [2, 2]
    t = img_aug[:2, 3:4]              # [2, 1]
    pix_aug = R @ pix + t              # [2, 8]
    # Clip + bbox
    x1 = float(pix_aug[0, valid].min().clip(0, img_size[1] - 1))
    y1 = float(pix_aug[1, valid].min().clip(0, img_size[0] - 1))
    x2 = float(pix_aug[0, valid].max().clip(0, img_size[1] - 1))
    y2 = float(pix_aug[1, valid].max().clip(0, img_size[0] - 1))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _load_meta(meta_dir: str, token: str) -> Dict:
    path = os.path.join(meta_dir, f'{token}_meta.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_3d(dir_path: str, token: str):
    path = os.path.join(dir_path, f'{token}_3d_dets.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_2d(dir_path: str, token: str):
    path = os.path.join(dir_path, f'{token}_2d_dets.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def project_all_3d_to_cameras(
    boxes_3d_lidar: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    lidar2img_per_cam: List[np.ndarray],
    img_aug_per_cam: List[np.ndarray],
    img_size: Tuple[int, int],
) -> List[Dict]:
    """For each camera, project all 3D boxes; return per-cam dict of
    {boxes, scores, labels}. A box appears in every camera where it's
    visible (in front of the camera + intersects the image plane)."""
    out = []
    for c in range(len(lidar2img_per_cam)):
        L2I = np.asarray(lidar2img_per_cam[c], dtype=np.float64)
        IAUG = np.asarray(img_aug_per_cam[c], dtype=np.float64)
        boxes_2d_proj = []
        scores_c = []
        labels_c = []
        for n in range(boxes_3d_lidar.shape[0]):
            b = project_3d_box_to_camera(
                boxes_3d_lidar[n], L2I, IAUG, img_size,
            )
            if b is None:
                continue
            boxes_2d_proj.append(b)
            scores_c.append(float(scores[n]))
            labels_c.append(int(labels[n]))
        out.append({
            'boxes': np.asarray(boxes_2d_proj, dtype=np.float64) if boxes_2d_proj
                     else np.zeros((0, 4), dtype=np.float64),
            'scores': np.asarray(scores_c, dtype=np.float64),
            'labels': np.asarray(labels_c, dtype=np.int64),
        })
    return out


def compare_one_sample(
    token: str,
    dir_with_2d: str,        # B11 dir (has 2D head outputs)
    dir_with_3d_baseline: str,  # B10c dir (3D-only baseline)
    iou_match: float,
    score_thr_2d: float,
    score_thr_3d: float,
    img_size: Tuple[int, int] = (256, 704),
) -> Dict:
    """Pair 2D dets (from B11's head) with projected-2D-from-3D dets
    (B11's own 3D head and B10c's 3D head) and report agreement stats."""

    # --- Load ----
    d3d_b11 = _load_3d(dir_with_2d,           token)
    d2d_b11 = _load_2d(dir_with_2d,           token)
    d3d_b10c = _load_3d(dir_with_3d_baseline, token)
    meta = _load_meta(dir_with_2d, token)
    if d3d_b11 is None or d2d_b11 is None:
        return None
    lidar2img_pc = meta.get('lidar2img')
    img_aug_pc   = meta.get('img_aug_matrix')
    if lidar2img_pc is None or img_aug_pc is None:
        return None
    # Some meta dumps may have these as nested lists; flatten one level
    # only if length matches the camera count.
    if len(lidar2img_pc) == 1 and isinstance(lidar2img_pc[0], list):
        lidar2img_pc = lidar2img_pc[0]
    if len(img_aug_pc) == 1 and isinstance(img_aug_pc[0], list):
        img_aug_pc = img_aug_pc[0]
    lidar2img_pc = [np.asarray(m, dtype=np.float64) for m in lidar2img_pc]
    img_aug_pc   = [np.asarray(m, dtype=np.float64) for m in img_aug_pc]

    # --- 3D dets → per-cam projected 2D ----
    boxes_lidar_b11 = np.asarray(d3d_b11['boxes_lidar'], dtype=np.float64) if d3d_b11['boxes_lidar'] else np.zeros((0, 9))
    sc_b11 = np.asarray(d3d_b11['scores'], dtype=np.float64) if d3d_b11['scores'] else np.zeros((0,))
    lb_b11 = np.asarray(d3d_b11['labels'], dtype=np.int64) if d3d_b11['labels'] else np.zeros((0,), dtype=np.int64)
    keep = sc_b11 >= score_thr_3d
    boxes_lidar_b11, sc_b11, lb_b11 = boxes_lidar_b11[keep], sc_b11[keep], lb_b11[keep]
    proj_b11 = project_all_3d_to_cameras(
        boxes_lidar_b11, lb_b11, sc_b11,
        lidar2img_pc, img_aug_pc, img_size,
    )

    if d3d_b10c is not None:
        boxes_lidar_b10c = np.asarray(d3d_b10c['boxes_lidar'], dtype=np.float64) if d3d_b10c['boxes_lidar'] else np.zeros((0, 9))
        sc_b10c = np.asarray(d3d_b10c['scores'], dtype=np.float64) if d3d_b10c['scores'] else np.zeros((0,))
        lb_b10c = np.asarray(d3d_b10c['labels'], dtype=np.int64) if d3d_b10c['labels'] else np.zeros((0,), dtype=np.int64)
        keep = sc_b10c >= score_thr_3d
        boxes_lidar_b10c, sc_b10c, lb_b10c = boxes_lidar_b10c[keep], sc_b10c[keep], lb_b10c[keep]
        proj_b10c = project_all_3d_to_cameras(
            boxes_lidar_b10c, lb_b10c, sc_b10c,
            lidar2img_pc, img_aug_pc, img_size,
        )
    else:
        proj_b10c = None

    # --- Pair 2D-head boxes with B11's projected-3D ----
    stats = {
        'token': token,
        'num_3d_b11': int(boxes_lidar_b11.shape[0]),
        'num_3d_b10c': int(boxes_lidar_b10c.shape[0]) if d3d_b10c else None,
        'num_2d_b11_total': 0,
        'paired': 0,
        '2d_only': 0,           # 2D head fired, no matching projected 3D
        '3d_only': 0,           # projected 3D exists, no matching 2D
        'class_agree': 0,
        'class_disagree': 0,
    }

    per_class_pair = defaultdict(int)
    per_class_2d_only = defaultdict(int)
    per_class_3d_only = defaultdict(int)

    for c_idx, cam_dets in enumerate(d2d_b11['cameras']):
        b2d_box = np.asarray(cam_dets['boxes'], dtype=np.float64) if cam_dets['boxes'] else np.zeros((0, 4))
        b2d_lab = np.asarray(cam_dets['labels'], dtype=np.int64) if cam_dets['labels'] else np.zeros((0,), dtype=np.int64)
        b2d_sc  = np.asarray(cam_dets['scores'], dtype=np.float64) if cam_dets['scores'] else np.zeros((0,))
        keep = b2d_sc >= score_thr_2d
        b2d_box, b2d_lab, b2d_sc = b2d_box[keep], b2d_lab[keep], b2d_sc[keep]

        proj_box = proj_b11[c_idx]['boxes']
        proj_lab = proj_b11[c_idx]['labels']

        stats['num_2d_b11_total'] += int(b2d_box.shape[0])

        if b2d_box.shape[0] == 0 and proj_box.shape[0] == 0:
            continue

        if b2d_box.shape[0] > 0 and proj_box.shape[0] > 0:
            ious = iou_xyxy(b2d_box, proj_box)
            matched_2d = np.zeros(b2d_box.shape[0], dtype=bool)
            matched_3d = np.zeros(proj_box.shape[0], dtype=bool)
            # Greedy match by descending IoU.
            flat = [(ious[i, j], i, j)
                    for i in range(b2d_box.shape[0])
                    for j in range(proj_box.shape[0])]
            flat.sort(key=lambda t: -t[0])
            for iou_val, i, j in flat:
                if iou_val < iou_match:
                    break
                if matched_2d[i] or matched_3d[j]:
                    continue
                matched_2d[i] = True
                matched_3d[j] = True
                stats['paired'] += 1
                if int(b2d_lab[i]) == int(proj_lab[j]):
                    stats['class_agree'] += 1
                    per_class_pair[int(b2d_lab[i])] += 1
                else:
                    stats['class_disagree'] += 1
            # 2D-only / 3D-only
            for i, m in enumerate(matched_2d):
                if not m:
                    stats['2d_only'] += 1
                    per_class_2d_only[int(b2d_lab[i])] += 1
            for j, m in enumerate(matched_3d):
                if not m:
                    stats['3d_only'] += 1
                    per_class_3d_only[int(proj_lab[j])] += 1
        else:
            stats['2d_only'] += int(b2d_box.shape[0])
            stats['3d_only'] += int(proj_box.shape[0])
            for lab in b2d_lab:
                per_class_2d_only[int(lab)] += 1
            for lab in proj_lab:
                per_class_3d_only[int(lab)] += 1

    stats['per_class_pair']    = dict(per_class_pair)
    stats['per_class_2d_only'] = dict(per_class_2d_only)
    stats['per_class_3d_only'] = dict(per_class_3d_only)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir-b11', required=True,
                    help='Unified-inference output dir for B11 (has 2D + 3D outputs).')
    ap.add_argument('--runs-dir-b10c', required=True,
                    help='Unified-inference output dir for B10c (3D-only baseline).')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--iou-match', type=float, default=0.30,
                    help='IoU threshold for pairing 2D-head detections to projected-3D boxes.')
    ap.add_argument('--score-thr-2d', type=float, default=0.20)
    ap.add_argument('--score-thr-3d', type=float, default=0.20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Discover sample tokens from B11 dir (it has both 2D + 3D files).
    sample_dirs = [d for d in glob(os.path.join(args.runs_dir_b11, '*')) if os.path.isdir(d)]
    tokens = sorted(os.path.basename(d) for d in sample_dirs)
    print(f'[compare] {len(tokens)} samples discovered in {args.runs_dir_b11}')

    rows = []
    cum = defaultdict(int)
    cum_pc_pair = defaultdict(int)
    cum_pc_2do  = defaultdict(int)
    cum_pc_3do  = defaultdict(int)
    for t in tokens:
        stats = compare_one_sample(
            t,
            dir_with_2d=os.path.join(args.runs_dir_b11, t),
            dir_with_3d_baseline=os.path.join(args.runs_dir_b10c, t),
            iou_match=args.iou_match,
            score_thr_2d=args.score_thr_2d,
            score_thr_3d=args.score_thr_3d,
        )
        if stats is None:
            print(f'[compare] {t}: missing input files; skipped')
            continue
        rows.append(stats)
        for k in ['num_3d_b11', 'num_2d_b11_total', 'paired',
                  '2d_only', '3d_only', 'class_agree', 'class_disagree']:
            cum[k] += stats.get(k, 0) or 0
        for k, v in stats.get('per_class_pair', {}).items():
            cum_pc_pair[k] += v
        for k, v in stats.get('per_class_2d_only', {}).items():
            cum_pc_2do[k] += v
        for k, v in stats.get('per_class_3d_only', {}).items():
            cum_pc_3do[k] += v

    # --- Per-sample CSV ----
    csv_path = os.path.join(args.out_dir, 'per_sample.csv')
    with open(csv_path, 'w', newline='') as f:
        cw = csv.DictWriter(
            f, fieldnames=['token', 'num_3d_b11', 'num_3d_b10c',
                           'num_2d_b11_total', 'paired',
                           '2d_only', '3d_only',
                           'class_agree', 'class_disagree'])
        cw.writeheader()
        for r in rows:
            cw.writerow({k: r.get(k, '') for k in cw.fieldnames})
    print(f'[compare] per-sample CSV → {csv_path}')

    # --- Aggregate summary ----
    print()
    print('=' * 60)
    print(f'AGGREGATE OVER {len(rows)} SAMPLES')
    print('=' * 60)
    n_pair  = cum['paired']
    n_2do   = cum['2d_only']
    n_3do   = cum['3d_only']
    n_total = n_pair + n_2do + n_3do
    if n_total == 0:
        print('No detections at all. Bad run.')
        return
    print(f'  Paired (2D head ↔ projected-3D):  {n_pair:5d}   '
          f'({n_pair/n_total*100:5.1f}%)')
    print(f'  2D-only  (head fires, 3D missed):  {n_2do:5d}   '
          f'({n_2do/n_total*100:5.1f}%)')
    print(f'  3D-only  (proj exists, no 2D):     {n_3do:5d}   '
          f'({n_3do/n_total*100:5.1f}%)')
    print(f'  Class agreement (where paired):    {cum["class_agree"]}/{n_pair}'
          f'   ({(cum["class_agree"]/max(1,n_pair))*100:.1f}%)')
    print()
    print('Per-class breakdown (paired / 2D-only / 3D-only):')
    for c in range(10):
        if cum_pc_pair[c] + cum_pc_2do[c] + cum_pc_3do[c] == 0:
            continue
        print(f'  {CLASSES[c]:22s}  paired={cum_pc_pair[c]:4d}   '
              f'2D-only={cum_pc_2do[c]:4d}   3D-only={cum_pc_3do[c]:4d}')

    summary = {
        'num_samples': len(rows),
        'cum': dict(cum),
        'per_class_paired': dict(cum_pc_pair),
        'per_class_2d_only': dict(cum_pc_2do),
        'per_class_3d_only': dict(cum_pc_3do),
        'iou_match': args.iou_match,
        'score_thr_2d': args.score_thr_2d,
        'score_thr_3d': args.score_thr_3d,
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[compare] summary → {os.path.join(args.out_dir, "summary.json")}')


if __name__ == '__main__':
    main()
