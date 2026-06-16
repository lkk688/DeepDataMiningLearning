"""S1: Quantify pseudo-label quality against Waymo v1.4.3 GT.

For each frame in pseudo_labels.jsonl whose extracted .npz has GT
boxes, compute per-class TP/FP/FN at 2 m center-distance match (same
metric as the downstream eval). Reports precision/recall/F1 and
per-class counts.

Waymo v1.4.3 box-type mapping (from extract_waymo_v1.py):
  1 = TYPE_VEHICLE       → 'Vehicle'
  2 = TYPE_PEDESTRIAN    → 'Pedestrian'
  3 = TYPE_SIGN          → (dropped)
  4 = TYPE_CYCLIST       → 'Cyclist'
"""
from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from collections import defaultdict

import numpy as np

WAYMO_TYPE_TO_TRANSFER = {
    1: 'Vehicle',
    2: 'Pedestrian',
    4: 'Cyclist',
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl',
        default='/tmp/waymo_pseudo_train_v1/pseudo_labels.jsonl')
    ap.add_argument('--extract-root',
        default='/tmp/waymo_v1_train_extract')
    ap.add_argument('--match-radius', type=float, default=2.0,
        help='BEV center-distance threshold for TP match (m).')
    ap.add_argument('--max-range', type=float, default=50.0,
        help='Only consider GT/pseudo within this BEV range (m).')
    return ap.parse_args()


def load_gt_boxes(npz_path):
    """Return list of (cls, [cx,cy,cz,lx,ly,lz,heading]) for one frame."""
    d = np.load(npz_path)
    out = []
    for row in d['boxes']:
        cx, cy, cz, lx, ly, lz, heading, wtype = (
            float(row[0]), float(row[1]), float(row[2]),
            float(row[3]), float(row[4]), float(row[5]),
            float(row[6]), int(row[7]))
        if wtype not in WAYMO_TYPE_TO_TRANSFER:
            continue
        cls = WAYMO_TYPE_TO_TRANSFER[wtype]
        out.append((cls, [cx, cy, cz, lx, ly, lz, heading]))
    return out


def match_per_class(pred_boxes, gt_boxes, match_radius, max_range):
    """Greedy 2-m BEV match per class. Returns dict cls -> (TP, FP, FN,
    n_pred_in_range, n_gt_in_range).
    """
    by_cls_pred = defaultdict(list)
    by_cls_gt   = defaultdict(list)

    def in_range(b):
        return (b[0]**2 + b[1]**2) ** 0.5 <= max_range

    for cls, b in pred_boxes:
        if in_range(b):
            by_cls_pred[cls].append(b)
    for cls, b in gt_boxes:
        if in_range(b):
            by_cls_gt[cls].append(b)

    out = {}
    for cls in set(list(by_cls_pred.keys()) + list(by_cls_gt.keys())):
        preds = by_cls_pred.get(cls, [])
        gts   = by_cls_gt.get(cls,   [])
        # Greedy nearest-first matching
        used_gt = [False] * len(gts)
        tp = 0
        for p in preds:
            best_d = match_radius
            best_j = -1
            for j, g in enumerate(gts):
                if used_gt[j]:
                    continue
                d = ((p[0]-g[0])**2 + (p[1]-g[1])**2) ** 0.5
                if d <= best_d:
                    best_d = d; best_j = j
            if best_j >= 0:
                used_gt[best_j] = True
                tp += 1
        fp = len(preds) - tp
        fn = sum(1 for u in used_gt if not u)
        out[cls] = (tp, fp, fn, len(preds), len(gts))
    return out


def main():
    args = parse_args()
    # Aggregate counts across all frames
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])  # [tp, fp, fn, n_pred, n_gt]
    n_frames_with_gt = 0
    n_frames_without_gt = 0

    with open(args.jsonl) as f:
        for line in f:
            rec = json.loads(line.strip() or '{}')
            if not rec:
                continue
            seg = rec['segment']
            fidx = int(rec['frame_idx'])
            npz = osp.join(args.extract_root, seg, f'f_{fidx:04d}.npz')
            if not osp.isfile(npz):
                n_frames_without_gt += 1
                continue
            n_frames_with_gt += 1

            gt = load_gt_boxes(npz)
            preds = [(p['cls'], p['box']) for p in rec['pseudo']
                     if p['cls'] in WAYMO_TYPE_TO_TRANSFER.values()]
            per_cls = match_per_class(preds, gt, args.match_radius,
                                       args.max_range)
            for cls, (tp, fp, fn, np_, ng) in per_cls.items():
                agg[cls][0] += tp
                agg[cls][1] += fp
                agg[cls][2] += fn
                agg[cls][3] += np_
                agg[cls][4] += ng

    print(f'[eval] matched {n_frames_with_gt} frames; skipped '
          f'{n_frames_without_gt} (no extracted GT)')
    print(f'[eval] match radius = {args.match_radius} m, '
          f'range ≤ {args.max_range} m\n')

    header = ('Class', 'Pred', 'GT', 'TP', 'FP', 'FN',
              'Precision', 'Recall', 'F1')
    print('  '.join(f'{h:>10s}' for h in header))
    print('-' * 96)
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    n_classes = 0
    for cls in ('Vehicle', 'Pedestrian', 'Cyclist'):
        tp, fp, fn, npred, ngt = agg.get(cls, (0, 0, 0, 0, 0))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        macro_p += p; macro_r += r; macro_f1 += f1
        n_classes += 1
        print(f'  {cls:>10s}  {npred:>10d}  {ngt:>10d}  {tp:>10d}  '
              f'{fp:>10d}  {fn:>10d}  {p:>10.3f}  {r:>10.3f}  {f1:>10.3f}')
    print('-' * 96)
    macro_p /= n_classes; macro_r /= n_classes; macro_f1 /= n_classes
    print(f'  {"Macro":>10s}                                                          '
          f'{macro_p:>10.3f}  {macro_r:>10.3f}  {macro_f1:>10.3f}')


if __name__ == '__main__':
    main()
