"""
Build a Waymo v2 → nuScenes-compatible info.pkl for B10c fine-tuning.

Splits the 202 waymo201/validation segments into 160 train / 42 held-out
val. Generates ``waymo_infos_train.pkl`` and ``waymo_infos_val.pkl`` whose
per-sample dicts mirror the nuScenes info-pkl structure that
``NuScenesDataset`` expects, so the existing B10c training config can be
reused with minimal changes.

Key structural notes:
  * Per-sample ``lidar_path`` points to a deterministic
    ``{segment}/{timestamp}.parquet#{frame_idx}`` reference; the
    ``WaymoFineTuneDataset`` knows how to decode that.
  * ``images`` dict has 6 keys (CAM_FRONT, CAM_FRONT_RIGHT, ...) following
    the slot mapping in dataset_waymo_mmdet3d.py.
  * ``instances`` are filtered to the 3 transfer classes (Vehicle,
    Pedestrian, Cyclist) with nuScenes-compatible labels (0-9 in the
    NUS_DET_CLASSES_10 mapping); BOXES are y-flipped here so the model
    sees the y-flipped frame consistently during fine-tune training (the
    same frame in which it produces inferences — see V6 in the pipeline
    doc).

Usage:
  python build_waymo_finetune_infos.py \
      --waymo-root /fs/atipa/data/rnd-liu/Datasets/waymo201 \
      --split validation \
      --train-frac 0.79 \
      --out-dir /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_finetune
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
import torch

sys.path.insert(0, '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning')
from DeepDataMiningLearning.detection3d.dataset_waymo_mmdet3d import (
    WaymoMMDet3DZeroShot, _load_camera_calibration, SLOT_TO_WAYMO_CAM,
    NUS_CAM_ORDER, NUM_CAMS)
from DeepDataMiningLearning.detection3d.class_map_waymo_to_nus import (
    NUS_DET_CLASSES_10)


# Waymo class id (1=Vehicle, 2=Ped, 3=Sign, 4=Cyclist)
#   → nuScenes class index (in NUS_DET_CLASSES_10)
# We assign Waymo Vehicle to the largest nuScenes vehicle category (car).
# Multi-class fine-tuning could use car/truck/bus heuristics from box size,
# but for the first pass we collapse to one nuScenes class per Waymo class.
WAYMO_TO_NUS_LABEL = {
    1: NUS_DET_CLASSES_10.index('car'),         # 0
    2: NUS_DET_CLASSES_10.index('pedestrian'),  # 8
    4: NUS_DET_CLASSES_10.index('bicycle'),     # 7 (closest to Cyclist)
    # 3 (Sign) is intentionally dropped — no nuScenes equivalent.
}


def build_one_info(ds: WaymoMMDet3DZeroShot, idx: int) -> Dict:
    """Build a single nuScenes-style info dict for one Waymo frame.

    GT boxes are y-flipped so the model sees the y-flipped frame
    consistently during training (matches V6 eval convention).
    """
    lidar_t, target = ds.base[idx]
    fname, ts, seg = ds.base.frame_index[idx]
    cam_calib = _load_camera_calibration(osp.join(ds.cam_calib_dir, fname))

    sample_idx = int(idx)
    token = f'{seg}__{ts}'

    # ego2global from base loader (we removed M_reflect there)
    T_wv = target.get('world_from_vehicle')
    if T_wv is None:
        T_wv_np = np.eye(4, dtype=np.float32)
    else:
        T_wv_np = T_wv.cpu().numpy() if torch.is_tensor(T_wv) else np.asarray(T_wv)

    # GT: in Waymo vehicle frame; y-flip to match model's training-time frame
    gt_boxes = target['boxes_3d']
    gt_labels = target['labels']
    if torch.is_tensor(gt_boxes): gt_boxes = gt_boxes.cpu().numpy()
    if torch.is_tensor(gt_labels): gt_labels = gt_labels.cpu().numpy()

    instances = []
    for b, lab in zip(gt_boxes, gt_labels):
        wlab = int(lab)
        if wlab not in WAYMO_TO_NUS_LABEL:
            continue
        nus_lab = WAYMO_TO_NUS_LABEL[wlab]
        # Y-FLIP: cy → -cy, yaw → -yaw. Velocity y also flips (we don't have it from GT).
        bbox_3d = [
            float(b[0]),
            float(-b[1]),                     # y-flip
            float(b[2]),
            float(b[3]),                      # l
            float(b[4]),                      # w
            float(b[5]),                      # h
            float(-b[6]),                     # yaw negate
        ]
        instances.append({
            'bbox_label': nus_lab,
            'bbox_3d': bbox_3d,
            'bbox_3d_isvalid': True,
            'bbox_label_3d': nus_lab,
            'num_lidar_pts': 99,                # placeholder (Waymo ensures >=9)
            'num_radar_pts': 0,
            'velocity': [0.0, 0.0],             # Waymo v2 has speed.x/.y; not loaded
        })

    # LiDAR path: a custom URI parsed by WaymoFineTuneDataset
    # Format:  waymo://{seg}/{ts}/{frame_idx}
    lidar_path = f'waymo://{seg}/{ts}/{idx}'

    # Images dict — populate with cam2img + lidar2cam in the y-flipped frame
    # (post-multiply by M_yflip if we end up rebuilding projections on read)
    images = OrderedDict()
    for slot in range(NUM_CAMS):
        cam_name = NUS_CAM_ORDER[slot]
        waymo_id = SLOT_TO_WAYMO_CAM[slot]
        if waymo_id is None or waymo_id not in cam_calib:
            # BACK cam is dummy (Waymo has no rear camera)
            images[cam_name] = dict(
                img_path=f'waymo://{seg}/{ts}/{idx}/cam_{slot}_dummy',
                cam2img=np.eye(4).tolist(),
                lidar2cam=np.eye(4).tolist(),
                lidar2img=np.eye(4).tolist(),
                height=256, width=704,
            )
            continue
        # We do NOT pre-resolve K @ T_oc_wc here — the dataset wrapper
        # rebuilds it per-frame so re-runs benefit from any future calibration
        # fixes. Just store the raw parquet path so the loader can fetch.
        images[cam_name] = dict(
            img_path=f'waymo://{seg}/{ts}/{idx}/cam_{slot}',
            cam2img=np.eye(4).tolist(),
            lidar2cam=np.eye(4).tolist(),
            lidar2img=np.eye(4).tolist(),
            height=256, width=704,
        )

    info = dict(
        sample_idx=sample_idx,
        token=token,
        timestamp=int(ts),
        ego2global=T_wv_np.astype(np.float32).tolist(),
        images=images,
        lidar_points=dict(
            num_pts_feats=5,
            lidar_path=lidar_path,
            # For nuScenes the lidar2ego rotates LIDAR → ego (90° z). For
            # Waymo we treat the y-flipped vehicle frame AS the model's
            # "LIDAR" frame, so lidar2ego is identity (the y-flip is baked
            # into the points + boxes already).
            lidar2ego=np.eye(4).tolist(),
        ),
        instances=instances,
        lidar_sweeps=[],
    )
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--waymo-root',
        default='/fs/atipa/data/rnd-liu/Datasets/waymo201')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--train-frac', type=float, default=0.79,
        help='Fraction of segments assigned to training (rest = held-out val)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'[infos] loading Waymo {args.split} from {args.waymo_root}')
    ds = WaymoMMDet3DZeroShot(root_dir=args.waymo_root, split=args.split,
                              max_frames=None)
    all_segments = sorted(set(t[2] for t in ds.base.frame_index))
    print(f'[infos] {len(all_segments)} unique segments, '
          f'{len(ds.base.frame_index)} total frames')

    random.seed(args.seed)
    shuffled = list(all_segments)
    random.shuffle(shuffled)
    n_train = int(round(len(shuffled) * args.train_frac))
    train_segs = set(shuffled[:n_train])
    val_segs   = set(shuffled[n_train:])
    print(f'[infos] segment split: {len(train_segs)} train / {len(val_segs)} val')

    train_infos: List[Dict] = []
    val_infos: List[Dict] = []
    cls_counter = Counter()
    for i in range(len(ds.base.frame_index)):
        _, _, seg = ds.base.frame_index[i]
        try:
            info = build_one_info(ds, i)
        except Exception as e:
            print(f'[infos] frame {i} skipped: {type(e).__name__}: {e}')
            continue
        for inst in info['instances']:
            cls_counter[inst['bbox_label']] += 1
        if seg in train_segs:
            train_infos.append(info)
        else:
            val_infos.append(info)
        if (i + 1) % 200 == 0:
            print(f'[infos]   {i+1}/{len(ds.base.frame_index)} frames processed  '
                  f'train={len(train_infos)} val={len(val_infos)}')

    # Save
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
        print(f'[infos] wrote {out_path}  ({len(infos)} samples)')

    # Save split metadata
    with open(osp.join(args.out_dir, 'segment_split.json'), 'w') as f:
        json.dump({
            'train_segments': sorted(train_segs),
            'val_segments': sorted(val_segs),
            'n_train_frames': len(train_infos),
            'n_val_frames': len(val_infos),
            'class_count_train': dict(cls_counter),
        }, f, indent=2)

    print()
    print('[infos] DONE')
    print(f'  train: {len(train_infos)} frames from {len(train_segs)} segments')
    print(f'  val:   {len(val_infos)} frames from {len(val_segs)} segments')
    print(f'  class instance counts (over all frames):')
    for k, v in cls_counter.most_common():
        print(f'     {NUS_DET_CLASSES_10[k]:14s}  {v:6d}')


if __name__ == '__main__':
    main()
