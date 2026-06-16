"""Build a Waymo-v1-style info.pkl from a pseudo_labels.jsonl produced
by bulk_pseudo_label_v1.py.

Schema matches build_waymo_finetune_infos_v1.py so the same
WaymoFineTuneDataset + LoadWaymoFrameFromInfo pipeline can consume it
without modification. Per-instance loss-weighting is exposed through
``pseudo_weight`` on each instance dict (downstream optional).

Pseudo-label class → nuScenes-10 mapping (matches the GT v1 builder):

  'Vehicle'    → idx 0  (car)
  'Pedestrian' → idx 8
  'Cyclist'    → idx 7  (bicycle)
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import sys
from collections import Counter, OrderedDict
from typing import Dict, List

import numpy as np

sys.path.insert(0, '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning')
from DeepDataMiningLearning.detection3d.class_map_waymo_to_nus import (
    NUS_DET_CLASSES_10)

PSEUDO_CLS_TO_NUS = {
    'Vehicle':    NUS_DET_CLASSES_10.index('car'),
    'Pedestrian': NUS_DET_CLASSES_10.index('pedestrian'),
    'Cyclist':    NUS_DET_CLASSES_10.index('bicycle'),
}

NUS_CAM_ORDER = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
)


def _frame_to_info(rec: Dict, global_idx: int) -> Dict:
    seg = rec['segment']
    fidx = int(rec['frame_idx'])
    ts = int(rec['timestamp'])

    instances = []
    for p in rec.get('pseudo', []):
        if p['cls'] not in PSEUDO_CLS_TO_NUS:
            continue
        nus_lab = PSEUDO_CLS_TO_NUS[p['cls']]
        # box from fusion is [cx, cy, cz, lx, ly, lz, heading] in vehicle frame.
        cx, cy, cz, lx, ly, lz, heading = (float(x) for x in p['box'][:7])
        # Apply y-flip to match the model's training frame (same convention
        # as build_waymo_finetune_infos_v1.py).
        bbox_3d = [cx, -cy, cz, lx, ly, lz, -heading]
        instances.append({
            'bbox_label': nus_lab,
            'bbox_3d': bbox_3d,
            'bbox_3d_isvalid': True,
            'bbox_label_3d': nus_lab,
            'num_lidar_pts': int(p.get('n_lidar', 50)),  # rough; >0 keeps it
            'num_radar_pts': 0,
            'velocity': [0.0, 0.0],
            # Pseudo-label provenance (consumed by future weighted-loss
            # training; ignored by default training).
            'pseudo_weight': float(p.get('weight', 1.0)),
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
        token=f'{seg}__{ts}__pseudo',
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
    ap.add_argument('--jsonl', required=True,
                    help='Path to pseudo_labels.jsonl from '
                         'bulk_pseudo_label_v1.py')
    ap.add_argument('--out-pkl', required=True,
                    help='Output info.pkl path.')
    ap.add_argument('--require-extracted-root', default=None,
                    help='If set, drop pseudo-labeled frames whose '
                         '{root}/{seg}/f_{idx:04d}.npz does not exist.')
    ap.add_argument('--min-instances', type=int, default=1,
                    help='Drop frames with fewer than N pseudo-labels.')
    args = ap.parse_args()

    infos = []
    n_total = n_kept = 0
    cls_counter = Counter()
    skip_no_npz = 0
    skip_empty = 0
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
            info = _frame_to_info(rec, n_kept)
            for inst in info['instances']:
                cls_counter[inst['bbox_label']] += 1
            infos.append(info)
            n_kept += 1

    metainfo = {
        'categories': {n: i for i, n in enumerate(NUS_DET_CLASSES_10)},
        'dataset': 'waymo_v1_pseudo_labels',
        'version': 'v1.4.3-pseudo',
        'info_version': '1.0',
        'source_jsonl': osp.abspath(args.jsonl),
    }
    out = dict(metainfo=metainfo, data_list=infos)
    os.makedirs(osp.dirname(osp.abspath(args.out_pkl)) or '.', exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(out, f)

    print(f'[pseudo-info] read   {n_total} frames from {args.jsonl}')
    print(f'[pseudo-info] kept   {n_kept} frames  '
          f'(skipped empty={skip_empty}, missing-npz={skip_no_npz})')
    print(f'[pseudo-info] wrote  {args.out_pkl} '
          f'({osp.getsize(args.out_pkl)/1e6:.1f} MB)')
    for k, v in cls_counter.most_common():
        print(f'    {NUS_DET_CLASSES_10[k]:14s}  {v:6d}')


if __name__ == '__main__':
    main()
