"""Q-Mix v0 — build a Waymo-v1-style info.pkl with calibrated
per-instance reliability scores baked into the ``pseudo_weight`` field.

  r_i = α_cls · P_cls[cls_i] + α_geom · p_geom(box_i) + α_match · p_match(source_i)

where the per-class precision ``P_cls`` is taken directly from the
S1 audit table on the scaled set (eval_pseudo_vs_gt.py output):

    Vehicle    P = 0.806
    Pedestrian P = 0.810
    Cyclist    P = 0.392

The per-box geometric quality ``p_geom`` rewards boxes with many
LiDAR points, and ``p_match`` rewards the strict-agreement A∩B branch
relative to A-only / B-only / class-conflict cases. Initial weights
α_cls = 0.5, α_geom = 0.25, α_match = 0.25.

Output: ``waymo_v1_infos_train_pseudo_qmix.pkl`` with
``instances[i]['pseudo_weight'] = r_i ∈ [0, 1]``. Downstream
QMixTransFusionHead reads this field and scales the per-positive loss
weights accordingly.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import sys
from collections import Counter, OrderedDict
from typing import Dict

import numpy as np

sys.path.insert(0, '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning')
from DeepDataMiningLearning.detection3d.class_map_waymo_to_nus import (
    NUS_DET_CLASSES_10)

PSEUDO_CLS_TO_NUS = {
    'Vehicle':    NUS_DET_CLASSES_10.index('car'),
    'Pedestrian': NUS_DET_CLASSES_10.index('pedestrian'),
    'Cyclist':    NUS_DET_CLASSES_10.index('bicycle'),
}

# Per-class precision from the S1 audit on the 3,409-frame scaled set
# (eval_pseudo_vs_gt.py). These are the directly-calibrated reliability
# anchors; everything else is a multiplicative refinement.
P_CLS_SCALED = {
    'Vehicle':    0.806,
    'Pedestrian': 0.810,
    'Cyclist':    0.392,
}

# Source-of-evidence reliability prior (A∩B = strict agreement).
# In the current run, every kept PL is A∩B; we still include the
# branch so future runs with VLM rescue can be properly weighted.
P_SOURCE = {
    'A∩B':              1.00,
    'A-only-VLM':       0.50,
    'B-only-VLM':       0.50,
    'conflict-VLM':     0.30,
    'unknown':          1.00,   # legacy fallback
}

NUS_CAM_ORDER = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
)


def compute_reliability(p, alpha_cls=0.5, alpha_geom=0.25, alpha_match=0.25,
                          geom_n_target=50.0, class_multiplier=None,
                          max_clip=None):
    """r_i for a single pseudo-label record from the JSONL.

    Default: r_i ∈ [0, 1]. With ``class_multiplier`` (e.g.
    {'Cyclist': 1.5}) the per-class component is multiplicatively
    boosted, allowing r_i > 1 to up-weight rare-class labels.
    ``max_clip`` caps the final value (default no cap).
    """
    cls = p['cls']
    p_cls = P_CLS_SCALED.get(cls, 0.5)
    if class_multiplier:
        p_cls = p_cls * class_multiplier.get(cls, 1.0)
    n_lidar = float(p.get('n_lidar', 0))
    p_geom = min(1.0, n_lidar / geom_n_target)
    source = str(p.get('source', 'unknown'))
    p_match = P_SOURCE.get(source, 1.0)
    r = float(alpha_cls * p_cls + alpha_geom * p_geom + alpha_match * p_match)
    if class_multiplier:
        # When up-weighting, also bump the WHOLE score for that class so the
        # increase is visible end-to-end (otherwise alpha_cls dampens it).
        mult = class_multiplier.get(cls, 1.0)
        if mult > 1.0:
            r = r * mult
    if max_clip is not None:
        r = min(r, max_clip)
    return r


def _frame_to_info(rec, global_idx, alpha_cls, alpha_geom, alpha_match,
                    class_multiplier=None, max_clip=None,
                    min_reliability=0.0):
    seg = rec['segment']
    fidx = int(rec['frame_idx'])
    ts = int(rec['timestamp'])

    instances = []
    for p in rec.get('pseudo', []):
        if p['cls'] not in PSEUDO_CLS_TO_NUS:
            continue
        nus_lab = PSEUDO_CLS_TO_NUS[p['cls']]
        cx, cy, cz, lx, ly, lz, heading = (float(x) for x in p['box'][:7])
        bbox_3d = [cx, -cy, cz, lx, ly, lz, -heading]
        r_i = compute_reliability(p,
                                    alpha_cls=alpha_cls,
                                    alpha_geom=alpha_geom,
                                    alpha_match=alpha_match,
                                    class_multiplier=class_multiplier,
                                    max_clip=max_clip)
        # Threshold filter: drop instances below min_reliability
        if r_i < min_reliability:
            continue
        instances.append({
            'bbox_label': nus_lab,
            'bbox_3d': bbox_3d,
            'bbox_3d_isvalid': True,
            'bbox_label_3d': nus_lab,
            'num_lidar_pts': int(p.get('n_lidar', 50)),
            'num_radar_pts': 0,
            'velocity': [0.0, 0.0],
            'pseudo_weight': r_i,
            'pseudo_source': str(p.get('source', 'unknown')),
        })

    lidar_path = f'waymo_v1://{seg}/{fidx}'
    images = OrderedDict()
    for cam_name in NUS_CAM_ORDER:
        images[cam_name] = dict(
            img_path=f'{lidar_path}/{cam_name}',
            cam2img=np.eye(4).tolist(),
            lidar2cam=np.eye(4).tolist(),
            lidar2img=np.eye(4).tolist(),
            height=256, width=704,
        )

    return dict(
        sample_idx=global_idx,
        token=f'{seg}__{ts}__qmix',
        timestamp=ts,
        ego2global=np.eye(4).tolist(),
        images=images,
        lidar_points=dict(num_pts_feats=5, lidar_path=lidar_path,
                          lidar2ego=np.eye(4).tolist()),
        instances=instances,
        lidar_sweeps=[],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl',
        default='/tmp/waymo_pseudo_train_v1/pseudo_labels.jsonl')
    ap.add_argument('--out-pkl',
        default='/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
                'data/waymo_finetune/waymo_v1_infos_train_pseudo_qmix.pkl')
    ap.add_argument('--require-extracted-root',
        default='/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
                'data/waymo_v1_extracted')
    ap.add_argument('--min-instances', type=int, default=1)
    ap.add_argument('--alpha-cls', type=float, default=0.5)
    ap.add_argument('--alpha-geom', type=float, default=0.25)
    ap.add_argument('--alpha-match', type=float, default=0.25)
    # Q-Mix v0 iteration knobs
    ap.add_argument('--min-reliability', type=float, default=0.0,
        help='v0a: drop PL instances with r_i below this threshold')
    ap.add_argument('--cyclist-multiplier', type=float, default=1.0,
        help='v0c: multiplicative boost for cyclist class weights')
    ap.add_argument('--max-clip', type=float, default=None,
        help='cap r_i at this value (None = no cap)')
    args = ap.parse_args()

    class_multiplier = None
    if args.cyclist_multiplier != 1.0:
        class_multiplier = {'Cyclist': args.cyclist_multiplier}

    infos = []
    n_total = n_kept = 0
    cls_counter = Counter()
    weight_buckets = Counter()
    weight_sum_by_cls = Counter(); weight_n_by_cls = Counter()
    skip_no_npz = skip_empty = 0
    with open(args.jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_total += 1
            if len(rec.get('pseudo', [])) < args.min_instances:
                skip_empty += 1
                continue
            if args.require_extracted_root:
                npz = osp.join(args.require_extracted_root,
                                rec['segment'],
                                f'f_{rec["frame_idx"]:04d}.npz')
                if not osp.isfile(npz):
                    skip_no_npz += 1
                    continue
            info = _frame_to_info(rec, n_kept,
                                    args.alpha_cls, args.alpha_geom,
                                    args.alpha_match,
                                    class_multiplier=class_multiplier,
                                    max_clip=args.max_clip,
                                    min_reliability=args.min_reliability)
            if len(info['instances']) == 0:
                skip_empty += 1
                continue
            for inst in info['instances']:
                cls_counter[inst['bbox_label']] += 1
                w = inst['pseudo_weight']
                bucket = int(round(w * 10)) / 10.0   # 0.0, 0.1, ...
                weight_buckets[bucket] += 1
                cls_name = NUS_DET_CLASSES_10[inst['bbox_label']]
                weight_sum_by_cls[cls_name] += w
                weight_n_by_cls[cls_name] += 1
            infos.append(info)
            n_kept += 1

    metainfo = {
        'categories': {n: i for i, n in enumerate(NUS_DET_CLASSES_10)},
        'dataset': 'waymo_v1_pseudo_labels_qmix',
        'version': 'v1.4.3-pseudo-qmix',
        'info_version': '1.0',
        'source_jsonl': osp.abspath(args.jsonl),
        'qmix_alphas': dict(cls=args.alpha_cls,
                              geom=args.alpha_geom,
                              match=args.alpha_match),
        'qmix_P_cls': P_CLS_SCALED,
    }
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(dict(metainfo=metainfo, data_list=infos), f)

    print(f'[qmix-info] read   {n_total} frames from {args.jsonl}')
    print(f'[qmix-info] kept   {n_kept} frames  '
          f'(skipped empty={skip_empty}, missing-npz={skip_no_npz})')
    print(f'[qmix-info] wrote  {args.out_pkl}  '
          f'({osp.getsize(args.out_pkl)/1e6:.1f} MB)')
    print(f'[qmix-info] per-class weight statistics:')
    for c in ('car', 'pedestrian', 'bicycle'):
        if weight_n_by_cls[c] > 0:
            mean_w = weight_sum_by_cls[c] / weight_n_by_cls[c]
            print(f'    {c:14s}  n={weight_n_by_cls[c]:5d}  mean_w={mean_w:.3f}')
    print(f'[qmix-info] reliability histogram (rounded to 0.1):')
    for k in sorted(weight_buckets.keys()):
        bar = '#' * int(80 * weight_buckets[k] / max(weight_buckets.values()))
        print(f'    {k:.1f}: {weight_buckets[k]:5d}  {bar}')


if __name__ == '__main__':
    main()
