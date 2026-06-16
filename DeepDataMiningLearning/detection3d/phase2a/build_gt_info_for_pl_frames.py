"""P0b: Build a Waymo-v1-style info.pkl with REAL Waymo GT for exactly
the same frame indices that the pseudo-label pipeline kept (the 868 +
2,541 frames in pseudo_labels.jsonl). This is the "Mixed-MoreGT"
matched-budget control:

  Mixed                      = 23 segments GT  + nuScenes
  Mixed+PL-Scaled            = 23 GT + 3,409 PL frames (auto-labels) + nuScenes
  Mixed-MoreGT (this script) = 23 GT + 3,409 GT  frames (real labels) + nuScenes

By matching the frame selection exactly, the only variable between
Mixed-MoreGT and Mixed+PL-Scaled is the label source (Waymo GT vs our
geometric-fusion PL). The gap quantifies "what's the price of using
auto-labels instead of GT?" If the gap is small, our PL is close to
the ceiling at this scale. If large, PL recall is the bottleneck.
"""
from __future__ import annotations

import argparse
import json
import os.path as osp
import pickle
import sys
from collections import Counter, OrderedDict
from typing import Dict

import numpy as np

sys.path.insert(0, '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning')
from DeepDataMiningLearning.detection3d.class_map_waymo_to_nus import (
    NUS_DET_CLASSES_10)

# Waymo v1.4.3 type id -> nuScenes-10 label idx (matches existing builder).
WAYMO_TO_NUS_LABEL = {
    1: NUS_DET_CLASSES_10.index('car'),
    2: NUS_DET_CLASSES_10.index('pedestrian'),
    4: NUS_DET_CLASSES_10.index('bicycle'),
}

NUS_CAM_ORDER = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
)


def build_one_info(seg: str, fidx: int, global_idx: int, v1_root: str):
    """Same schema as build_waymo_finetune_infos_v1.py but for one
    (segment, frame_idx) using GT from the extracted .npz."""
    npz = osp.join(v1_root, seg, f'f_{fidx:04d}.npz')
    d = np.load(npz)
    boxes = d['boxes']
    ts = int(d['ts'])
    instances = []
    for row in boxes:
        cx, cy, cz, lx, ly, lz, heading, wtyp = (
            float(row[0]), float(row[1]), float(row[2]),
            float(row[3]), float(row[4]), float(row[5]),
            float(row[6]), int(row[7]))
        if wtyp not in WAYMO_TO_NUS_LABEL:
            continue
        nus_lab = WAYMO_TO_NUS_LABEL[wtyp]
        bbox_3d = [cx, -cy, cz, lx, ly, lz, -heading]   # y-flip
        instances.append({
            'bbox_label': nus_lab,
            'bbox_3d': bbox_3d,
            'bbox_3d_isvalid': True,
            'bbox_label_3d': nus_lab,
            'num_lidar_pts': 99,
            'num_radar_pts': 0,
            'velocity': [0.0, 0.0],
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
        token=f'{seg}__{ts}__gtmatch',
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
        default='/tmp/waymo_pseudo_train_v1/pseudo_labels.jsonl',
        help='Source of (segment, frame_idx) selection.')
    ap.add_argument('--v1-root',
        default='/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
                'data/waymo_v1_extracted',
        help='Where the extracted .npz files live (with symlinks for new segs).')
    ap.add_argument('--out-pkl',
        default='/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
                'data/waymo_finetune/waymo_v1_infos_train_pl_frames_gt.pkl')
    ap.add_argument('--require-extracted', action='store_true',
        help='Skip frames whose npz is missing (recommended).')
    args = ap.parse_args()

    infos = []
    n_read = 0; n_kept = 0; n_skip_missing = 0; n_skip_empty = 0
    cls_counter = Counter()
    with open(args.jsonl) as f:
        for line in f:
            rec = json.loads(line.strip() or '{}')
            if not rec:
                continue
            n_read += 1
            seg = rec['segment']
            fidx = int(rec['frame_idx'])
            npz = osp.join(args.v1_root, seg, f'f_{fidx:04d}.npz')
            if not osp.isfile(npz):
                n_skip_missing += 1
                continue
            info = build_one_info(seg, fidx, n_kept, args.v1_root)
            if not info['instances']:
                n_skip_empty += 1
                continue
            for inst in info['instances']:
                cls_counter[inst['bbox_label']] += 1
            infos.append(info)
            n_kept += 1

    metainfo = {
        'categories': {n: i for i, n in enumerate(NUS_DET_CLASSES_10)},
        'dataset': 'waymo_v1_pl_frames_real_gt',
        'version': 'v1.4.3-pl-frames-gt',
        'info_version': '1.0',
        'source_jsonl': osp.abspath(args.jsonl),
    }
    out = dict(metainfo=metainfo, data_list=infos)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(out, f)

    print(f'[gt-info] read   {n_read} frames from {args.jsonl}')
    print(f'[gt-info] kept   {n_kept} frames  '
          f'(missing-npz={n_skip_missing}, empty-GT={n_skip_empty})')
    print(f'[gt-info] wrote  {args.out_pkl}  '
          f'({osp.getsize(args.out_pkl)/1e6:.1f} MB)')
    for k, v in cls_counter.most_common():
        print(f'    {NUS_DET_CLASSES_10[k]:14s}  {v:6d}')


if __name__ == '__main__':
    main()
