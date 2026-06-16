"""
FAST Waymo→nuScenes info.pkl builder. Reads box parquets directly (no
LiDAR or image decoding), assigning frames to train/val by segment.

Performance: ~9000 frames in ~2 minutes (vs ~10h for the v1 builder
that called the full frame loader).
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import random
import sys
from collections import Counter, OrderedDict, defaultdict
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning')
from DeepDataMiningLearning.detection3d.class_map_waymo_to_nus import (
    NUS_DET_CLASSES_10)

# Waymo class id → nuScenes class index in NUS_DET_CLASSES_10
WAYMO_TO_NUS_LABEL = {
    1: NUS_DET_CLASSES_10.index('car'),         # Vehicle → car (collapse)
    2: NUS_DET_CLASSES_10.index('pedestrian'),
    4: NUS_DET_CLASSES_10.index('bicycle'),     # Cyclist → bicycle
    # 3 = Sign → dropped
}

NUS_CAM_ORDER = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
)


def _read_segment_frames(split_dir: str, fname: str):
    """Return list of (timestamp, [(box_7, type, n_pts), ...]) for one segment."""
    box_path = osp.join(split_dir, 'lidar_box', fname)
    df = pq.read_table(box_path).to_pandas()

    cols = ['[LiDARBoxComponent].box.center.x',
            '[LiDARBoxComponent].box.center.y',
            '[LiDARBoxComponent].box.center.z',
            '[LiDARBoxComponent].box.size.x',
            '[LiDARBoxComponent].box.size.y',
            '[LiDARBoxComponent].box.size.z',
            '[LiDARBoxComponent].box.heading',
            '[LiDARBoxComponent].type',
            '[LiDARBoxComponent].num_lidar_points_in_box',
            'key.frame_timestamp_micros']
    df_sub = df[cols]

    out = defaultdict(list)
    for row in df_sub.itertuples(index=False):
        ts = int(row[9])
        cx, cy, cz = float(row[0]), float(row[1]), float(row[2])
        lx, ly, lz = float(row[3]), float(row[4]), float(row[5])
        heading   = float(row[6])
        wtyp      = int(row[7])
        n_pts     = int(row[8])
        out[ts].append((cx, cy, cz, lx, ly, lz, heading, wtyp, n_pts))
    return dict(out)


def _build_info(seg: str, ts: int, frame_idx: int, boxes: list):
    """Build one nuScenes-style info dict, y-flipping GT boxes inline."""
    instances = []
    for cx, cy, cz, lx, ly, lz, heading, wtyp, n_pts in boxes:
        if wtyp not in WAYMO_TO_NUS_LABEL:
            continue
        nus_lab = WAYMO_TO_NUS_LABEL[wtyp]
        # Y-FLIP (cy → -cy, heading → -heading)
        bbox_3d = [cx, -cy, cz, lx, ly, lz, -heading]
        instances.append({
            'bbox_label': nus_lab,
            'bbox_3d': bbox_3d,
            'bbox_3d_isvalid': True,
            'bbox_label_3d': nus_lab,
            'num_lidar_pts': int(n_pts),
            'num_radar_pts': 0,
            'velocity': [0.0, 0.0],
        })

    lidar_path = f'waymo://{seg}/{ts}/{frame_idx}'
    images = OrderedDict()
    for slot, cam_name in enumerate(NUS_CAM_ORDER):
        images[cam_name] = dict(
            img_path=f'waymo://{seg}/{ts}/{frame_idx}/cam_{slot}',
            cam2img=np.eye(4).tolist(),
            lidar2cam=np.eye(4).tolist(),
            lidar2img=np.eye(4).tolist(),
            height=256, width=704,
        )

    return dict(
        sample_idx=frame_idx,
        token=f'{seg}__{ts}',
        timestamp=ts,
        ego2global=np.eye(4).tolist(),         # populated lazily by transform
        images=images,
        lidar_points=dict(num_pts_feats=5, lidar_path=lidar_path,
                          lidar2ego=np.eye(4).tolist()),
        instances=instances,
        lidar_sweeps=[],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--waymo-root',
        default='/fs/atipa/data/rnd-liu/Datasets/waymo201')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--train-frac', type=float, default=0.79)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    split_dir = osp.join(args.waymo_root, args.split)
    box_dir = osp.join(split_dir, 'lidar_box')
    # Restrict to segments that have ALL required parquets (lidar,
    # lidar_box, camera_image, vehicle_pose, lidar_calibration). Many
    # waymo201/validation segments are missing camera_image (~165/202),
    # and Waymo3DDataset drops those at load time → we'd produce info
    # entries with no matching loader frame_index.
    needed_dirs = ['lidar', 'lidar_box', 'camera_image',
                   'vehicle_pose', 'lidar_calibration']
    parquets_all = sorted(f for f in os.listdir(box_dir)
                          if f.endswith('.parquet'))
    parquets = []
    for f in parquets_all:
        if all(osp.exists(osp.join(split_dir, d, f)) for d in needed_dirs):
            parquets.append(f)
    print(f'[fast] {len(parquets)} segments with ALL required parquets '
          f'(of {len(parquets_all)} total in {box_dir})')

    # Segment-level split (use parquet filename as segment id).
    random.seed(args.seed)
    shuffled = list(parquets)
    random.shuffle(shuffled)
    n_train = int(round(len(shuffled) * args.train_frac))
    train_set = set(shuffled[:n_train])
    val_set   = set(shuffled[n_train:])
    print(f'[fast] segment split: {len(train_set)} train / {len(val_set)} val')

    # CRITICAL: Waymo3DDataset only reads row_group(0) of each lidar
    # parquet for its frame_index. So we MUST restrict info entries to
    # the (seg, ts) pairs the loader can actually serve. Pre-build the
    # same lookup the loader builds and use it as our filter.
    print('[fast] building loader (seg, ts) lookup for filtering ...')
    sys.path.insert(0, '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning')
    from DeepDataMiningLearning.detection3d.dataset_waymo_mmdet3d import (
        WaymoMMDet3DZeroShot)
    loader = WaymoMMDet3DZeroShot(root_dir=args.waymo_root,
                                  split=args.split, max_frames=None)
    loader_keys = {(seg, int(ts)) for (_, ts, seg) in loader.base.frame_index}
    print(f'[fast] loader has {len(loader_keys)} (seg, ts) pairs accessible')

    train_infos: List[Dict] = []
    val_infos: List[Dict] = []
    cls_counter = Counter()
    global_idx = 0
    dropped_unreachable = 0
    for s_i, fname in enumerate(parquets):
        seg = fname.replace('.parquet', '')
        by_ts = _read_segment_frames(split_dir, fname)
        sorted_ts = sorted(by_ts)
        is_train = fname in train_set
        target = train_infos if is_train else val_infos
        for ts in sorted_ts:
            if (seg, ts) not in loader_keys:
                dropped_unreachable += 1
                continue
            info = _build_info(seg, ts, global_idx, by_ts[ts])
            global_idx += 1
            target.append(info)
            for inst in info['instances']:
                cls_counter[inst['bbox_label']] += 1
        if (s_i + 1) % 10 == 0:
            print(f'[fast] {s_i+1}/{len(parquets)} segments  '
                  f'train_frames={len(train_infos)} val_frames={len(val_infos)} '
                  f'dropped={dropped_unreachable}')
    print(f'[fast] total dropped (unreachable by loader): {dropped_unreachable}')

    metainfo = {
        'categories': {n: i for i, n in enumerate(NUS_DET_CLASSES_10)},
        'dataset': 'waymo_v201_finetune_yflipped',
        'version': 'v2.0.1',
        'info_version': '1.1',
    }
    for name, infos in [('train', train_infos), ('val', val_infos)]:
        out_path = osp.join(args.out_dir, f'waymo_infos_{name}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(dict(metainfo=metainfo, data_list=infos), f)
        print(f'[fast] wrote {out_path}  ({len(infos)} samples)')

    with open(osp.join(args.out_dir, 'segment_split.json'), 'w') as f:
        json.dump({
            'train_segments': sorted(train_set),
            'val_segments': sorted(val_set),
            'n_train_frames': len(train_infos),
            'n_val_frames': len(val_infos),
            'class_count_train': {NUS_DET_CLASSES_10[k]: v for k, v in cls_counter.items()},
        }, f, indent=2)

    print()
    print(f'[fast] DONE — train={len(train_infos)}  val={len(val_infos)}')
    for k, v in cls_counter.most_common():
        print(f'    {NUS_DET_CLASSES_10[k]:14s}  {v:6d}')


if __name__ == '__main__':
    main()
