"""C1: Bootstrap 95% CIs for AP@2m on the 300-frame Waymo val subset.

For each model (Base / Mixed / Mixed+CBal / Mixed+PL), resample the
300 frames with replacement, compute AP per class, repeat $B$ times,
report median + 95% percentile interval. This establishes the noise
floor that the paper currently hand-waves as "$\\pm0.006$".

Output is a markdown table copy-paste-ready for the paper.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np


def load(path):
    return json.load(open(path))


def _ap_11pt(rec, prec):
    """11-point interpolated AP from a sorted-by-decreasing-score list."""
    out = 0.0
    for t in np.linspace(0, 1, 11):
        cap = np.max(prec[rec >= t]) if (rec >= t).any() else 0.0
        out += cap
    return float(out / 11.0)


def per_class_ap(frames, cls, match_radius=2.0, score_thresh=0.10,
                  range_within_50=False):
    """Compute AP@match_radius for one class across `frames`.

    Matches eval_waymo_zeroshot.py: greedy per-frame matching, then
    score-sorted PR curve across all frames, 11-point interpolation.
    Uses enumeration index as frame key so resampled (duplicated)
    frames each get their own GT-used array.
    """
    all_scores = []
    is_tp = []
    n_gt = 0
    for fr_id, fr in enumerate(frames):
        # GT for this class in this frame
        gt_centers = []
        for b, c in zip(fr['gt']['boxes'], fr['gt']['transfer_cls']):
            if c != cls:
                continue
            if range_within_50 and (b[0]**2 + b[1]**2) ** 0.5 > 50.0:
                continue
            gt_centers.append((b[0], b[1]))
        n_gt += len(gt_centers)
        gt_used = [False] * len(gt_centers)

        # Predictions for this class, sorted by descending score
        dets = []
        for b, s, pc in zip(fr['dets']['boxes'],
                              fr['dets'].get('scores',
                                  [1.0]*len(fr['dets']['boxes'])),
                              fr['dets']['transfer_cls']):
            if pc != cls:
                continue
            sm = max(s) if isinstance(s, (list, tuple)) else float(s)
            if sm < score_thresh:
                continue
            if range_within_50 and (b[0]**2 + b[1]**2) ** 0.5 > 50.0:
                continue
            dets.append((sm, b[0], b[1]))
        dets.sort(key=lambda x: -x[0])

        # Greedy match this frame's predictions to its own GTs
        for s, px, py in dets:
            best_d = match_radius
            best_j = -1
            for j, (gx, gy) in enumerate(gt_centers):
                if gt_used[j]:
                    continue
                d = ((px-gx)**2 + (py-gy)**2) ** 0.5
                if d <= best_d:
                    best_d = d; best_j = j
            all_scores.append(s)
            if best_j >= 0:
                gt_used[best_j] = True
                is_tp.append(1)
            else:
                is_tp.append(0)

    if not all_scores or n_gt == 0:
        return 0.0
    # Score-sorted PR curve
    order = np.argsort(-np.array(all_scores))
    is_tp = np.array(is_tp, dtype=np.int32)[order]
    tp_cum = np.cumsum(is_tp)
    fp_cum = np.cumsum(1 - is_tp)
    prec = tp_cum / np.maximum(1, tp_cum + fp_cum)
    rec = tp_cum / max(1, n_gt)
    return _ap_11pt(rec, prec)


def bootstrap(frames, B=1000, seed=42, **kwargs):
    """Bootstrap AP per class across `B` resamples. Returns dict
    cls -> (median, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    classes = ('Vehicle', 'Pedestrian', 'Cyclist')
    samples = {c: [] for c in classes}
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        sub = [frames[i] for i in idx]
        for c in classes:
            samples[c].append(per_class_ap(sub, c, **kwargs))
    out = {}
    for c in classes:
        arr = np.array(samples[c])
        median = float(np.median(arr))
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        out[c] = (median, lo, hi)
    # Macro mAP per resample
    macro = []
    for b in range(B):
        macro.append(np.mean([samples[c][b] for c in classes]))
    arr = np.array(macro)
    out['Macro'] = (float(np.median(arr)), float(np.percentile(arr, 2.5)),
                     float(np.percentile(arr, 97.5)))
    return out


def main():
    paths = {
        'Base':       '/tmp/waymo_zs_b10c_v6/waymo_per_frame_dets.json',
        'Mixed':      '/tmp/waymo_zs_b14_mixed/waymo_per_frame_dets.json',
        'Mixed+CBal': '/tmp/waymo_zs_b17_curated/waymo_per_frame_dets.json',
        'Mixed+PL':   '/tmp/waymo_zs_b18_pseudo/waymo_per_frame_dets.json',
    }
    data = {k: load(p) for k, p in paths.items()}

    # Two calibration regimes: Default (score=0.10) and Phase-1 best
    # (score=0.01 + ≤50m)
    regimes = [
        ('Default', dict(score_thresh=0.10, range_within_50=False)),
        ('Phase-1', dict(score_thresh=0.01, range_within_50=True)),
    ]
    B = 1000

    for regime_name, kwargs in regimes:
        print(f'\n## {regime_name}  (bootstrap B={B})')
        print(f'{"Variant":<14s} {"Veh.":>22s} {"Ped.":>22s} '
              f'{"Cyc.":>22s} {"Macro":>22s}')
        print('-' * 110)
        for key, frames in data.items():
            ci = bootstrap(frames, B=B, **kwargs)
            row = []
            for cls in ('Vehicle', 'Pedestrian', 'Cyclist', 'Macro'):
                m, lo, hi = ci[cls]
                row.append(f'{m:.3f} [{lo:.3f},{hi:.3f}]')
            print(f'{key:<14s} ' + ' '.join(f'{r:>22s}' for r in row))


if __name__ == '__main__':
    main()
