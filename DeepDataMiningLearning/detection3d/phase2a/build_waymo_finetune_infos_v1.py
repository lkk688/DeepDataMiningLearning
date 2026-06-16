"""
Build info.pkl from Waymo v1 extracted data (see extract_waymo_v1.py).
Mirrors build_waymo_finetune_infos_fast.py but reads .npz files from a
local folder layout instead of v2 parquets.

Y-flip on GT boxes is applied (cy → -cy, yaw → -yaw) so the model sees
the same frame as for the v2 training pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import random
import sys
from collections import Counter, OrderedDict
from typing import Dict, List

import numpy as np

sys.path.insert(0, '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning')
from DeepDataMiningLearning.detection3d.class_map_waymo_to_nus import (
    NUS_DET_CLASSES_10)

WAYMO_TO_NUS_LABEL = {
    1: NUS_DET_CLASSES_10.index('car'),
    2: NUS_DET_CLASSES_10.index('pedestrian'),
    4: NUS_DET_CLASSES_10.index('bicycle'),
}

NUS_CAM_ORDER = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
)


def build_info(v1_root: str, seg: str, frame_idx: int, global_idx: int):
    seg_dir = osp.join(v1_root, seg)
    npz_path = osp.join(seg_dir, f'f_{frame_idx:04d}.npz')
    d = np.load(npz_path)
    boxes = d['boxes']           # (M, 8) [cx,cy,cz,lx,ly,lz,heading,type]
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

    lidar_path = f'waymo_v1://{seg}/{frame_idx}'
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
        token=f'{seg}__{ts}',
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
    ap.add_argument('--v1-root',
        default='/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_v1_extracted')
    ap.add_argument('--train-frac', type=float, default=0.85)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    segments = sorted(d for d in os.listdir(args.v1_root)
                      if osp.isdir(osp.join(args.v1_root, d)))
    print(f'[v1-info] {len(segments)} extracted segments at {args.v1_root}')

    # Segment-level split (no leakage).
    random.seed(args.seed)
    shuffled = list(segments)
    random.shuffle(shuffled)
    n_train = int(round(len(shuffled) * args.train_frac))
    train_set = set(shuffled[:n_train])
    val_set   = set(shuffled[n_train:])
    print(f'[v1-info] segment split: {len(train_set)} train / {len(val_set)} val')

    train_infos: List[Dict] = []
    val_infos: List[Dict] = []
    cls_counter = Counter()
    global_idx = 0
    for s_i, seg in enumerate(segments):
        seg_dir = osp.join(args.v1_root, seg)
        frame_files = sorted(f for f in os.listdir(seg_dir)
                             if f.startswith('f_') and f.endswith('.npz')
                             and 'cam' not in f)
        is_train = seg in train_set
        target = train_infos if is_train else val_infos
        for ff in frame_files:
            try:
                frame_idx = int(ff[2:6])
            except ValueError:
                continue
            info = build_info(args.v1_root, seg, frame_idx, global_idx)
            global_idx += 1
            target.append(info)
            for inst in info['instances']:
                cls_counter[inst['bbox_label']] += 1
        if (s_i + 1) % 5 == 0:
            print(f'[v1-info] {s_i+1}/{len(segments)} segments  '
                  f'train={len(train_infos)} val={len(val_infos)}')

    metainfo = {
        'categories': {n: i for i, n in enumerate(NUS_DET_CLASSES_10)},
        'dataset': 'waymo_v1_finetune_yflipped',
        'version': 'v1.4.3',
        'info_version': '1.0',
    }
    for name, infos in [('train', train_infos), ('val', val_infos)]:
        out_path = osp.join(args.out_dir, f'waymo_v1_infos_{name}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(dict(metainfo=metainfo, data_list=infos), f)
        print(f'[v1-info] wrote {out_path}  ({len(infos)} samples)')

    with open(osp.join(args.out_dir, 'v1_segment_split.json'), 'w') as f:
        json.dump({
            'train_segments': sorted(train_set),
            'val_segments': sorted(val_set),
            'n_train_frames': len(train_infos),
            'n_val_frames': len(val_infos),
            'class_count_train': {NUS_DET_CLASSES_10[k]: v for k, v in cls_counter.items()},
        }, f, indent=2)

    print()
    print(f'[v1-info] DONE — train={len(train_infos)}  val={len(val_infos)}')
    for k, v in cls_counter.most_common():
        print(f'    {NUS_DET_CLASSES_10[k]:14s}  {v:6d}')


if __name__ == '__main__':
    main()
