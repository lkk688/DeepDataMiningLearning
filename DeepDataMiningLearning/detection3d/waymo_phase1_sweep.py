"""
Phase-1 free diagnostic on the saved Waymo per-frame dets.

Question: how much of the zero-shot gap is *calibration* (just the wrong
score threshold for this domain) vs *real representation gap*?

Re-scores the saved dets at multiple (score_thresh, range_max) combos
and reports per-class AP@2m.

  python waymo_phase1_sweep.py /tmp/waymo_zs_b10c_wide/waymo_per_frame_dets.json
"""

import argparse
import json
import sys
from typing import List, Dict

import numpy as np


def _match_one_frame(dets, gt, dist_thresh: float = 2.0,
                     score_thresh: float = 0.0,
                     range_max: float = 80.0):
    """Return per-class lists: (is_tp 0/1 sorted by det score) and n_gt."""
    classes = ['Vehicle', 'Pedestrian', 'Cyclist']
    out = {c: {'tp_flags': [], 'scores': [], 'n_gt': 0} for c in classes}

    # Filter GT by range
    gt_boxes = np.asarray(gt['boxes'], dtype=np.float32).reshape(-1, 7)
    gt_cls = list(gt['transfer_cls'])
    if gt_boxes.shape[0] > 0:
        d = np.linalg.norm(gt_boxes[:, :2], axis=1)
        mask = d <= range_max
        gt_boxes = gt_boxes[mask]
        gt_cls = [gt_cls[i] for i in range(len(gt_cls)) if mask[i]]

    for c in classes:
        gt_idx_c = [i for i, x in enumerate(gt_cls) if x == c]
        out[c]['n_gt'] = len(gt_idx_c)

    # Filter dets by score + range, then sort by score
    det_boxes = np.asarray(dets['boxes'], dtype=np.float32).reshape(-1, 9 if
        (len(dets['boxes']) and len(dets['boxes'][0]) == 9) else 7)
    if det_boxes.size == 0:
        return out
    det_scores = np.asarray(dets['scores'], dtype=np.float32)
    det_cls = list(dets['transfer_cls'])

    keep = det_scores >= score_thresh
    if range_max < 1e6:
        d = np.linalg.norm(det_boxes[:, :2], axis=1)
        keep &= d <= range_max
    if not keep.any():
        return out
    det_boxes = det_boxes[keep]
    det_scores = det_scores[keep]
    det_cls = [det_cls[i] for i in range(len(det_cls)) if keep[i]]

    # Greedy match per class
    for c in classes:
        gt_idx_c = [i for i, x in enumerate(gt_cls) if x == c]
        det_idx_c = [i for i, x in enumerate(det_cls) if x == c]
        if not det_idx_c:
            continue
        # Sort dets by score (descending)
        det_idx_c.sort(key=lambda i: -det_scores[i])
        used_gt = set()
        for di in det_idx_c:
            s = det_scores[di]
            if not gt_idx_c:
                out[c]['tp_flags'].append(0)
                out[c]['scores'].append(float(s))
                continue
            gt_xy = gt_boxes[gt_idx_c][:, :2]
            det_xy = det_boxes[di:di+1, :2]
            dist = np.linalg.norm(gt_xy - det_xy, axis=1)
            for j in np.argsort(dist):
                gj = gt_idx_c[j]
                if gj in used_gt:
                    continue
                if dist[j] <= dist_thresh:
                    used_gt.add(gj)
                    out[c]['tp_flags'].append(1)
                    out[c]['scores'].append(float(s))
                    break
            else:
                out[c]['tp_flags'].append(0)
                out[c]['scores'].append(float(s))
                continue
            # mark TP if loop above didn't `else` (Python loop-else trick)
            if out[c]['tp_flags'][-1] != 1:
                out[c]['tp_flags'].append(0)
                out[c]['scores'].append(float(s))
    return out


def _accumulate(records, **kw):
    classes = ['Vehicle', 'Pedestrian', 'Cyclist']
    agg = {c: {'tp_flags': [], 'scores': [], 'n_gt': 0} for c in classes}
    for r in records:
        f = _match_one_frame(r['dets'], r['gt'], **kw)
        for c in classes:
            agg[c]['tp_flags'].extend(f[c]['tp_flags'])
            agg[c]['scores'].extend(f[c]['scores'])
            agg[c]['n_gt'] += f[c]['n_gt']
    return agg


def _ap_11pt(tp_flags, scores, n_gt):
    if not tp_flags or n_gt == 0:
        return 0.0, 0, 0, 0.0, 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp_flags)[order]
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    prec = tp_cum / np.maximum(1, tp_cum + fp_cum)
    rec = tp_cum / max(1, n_gt)
    ap = 0.0
    for r_thr in np.linspace(0, 1, 11):
        m = rec >= r_thr
        ap += (prec[m].max() if m.any() else 0.0)
    ap /= 11.0
    return float(ap), int(tp.sum()), int(fp.sum()), float(prec[-1]), float(rec[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json_path')
    ap.add_argument('--dist-thresh', type=float, default=2.0)
    args = ap.parse_args()

    print(f'[phase1] loading {args.json_path} ...')
    with open(args.json_path) as f:
        records = json.load(f)
    print(f'[phase1] {len(records)} frames loaded')

    score_grid = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    range_grid = [50.0, 1e6]

    header = (f'  {"score":>5s}  {"range":>5s}  '
              f'{"Vehicle":>20s}  {"Pedestrian":>20s}  {"Cyclist":>20s}  '
              f'{"macro":>6s}')
    print('=' * 95)
    print(header)
    print('=' * 95)
    for rng in range_grid:
        for s in score_grid:
            agg = _accumulate(records, dist_thresh=args.dist_thresh,
                              score_thresh=s, range_max=rng)
            row = f'  {s:>5.2f}  {("≤50m" if rng < 1e6 else "all"):>5s}  '
            aps = []
            for c in ['Vehicle', 'Pedestrian', 'Cyclist']:
                d = agg[c]
                ap_v, tp, fp, _, recall = _ap_11pt(
                    d['tp_flags'], d['scores'], d['n_gt'])
                aps.append(ap_v)
                row += (f'AP={ap_v:.3f} R={recall:.3f}'.ljust(20) + '  ')
            macro = float(np.mean(aps))
            row += f'{macro:>6.3f}'
            print(row)
        print('-' * 95)


if __name__ == '__main__':
    main()
