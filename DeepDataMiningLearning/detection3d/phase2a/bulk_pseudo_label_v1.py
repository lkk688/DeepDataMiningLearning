"""Bulk pseudo-label generation on Waymo v1.4.3 training tfrecords.

Streams tfrecords frame-by-frame (no intermediate npz dump), runs the
same Validator A (LiDAR DBSCAN) + Validator B (Faster R-CNN + LiDAR
depth lift) + fusion pipeline as bulk_pseudo_label.py, and writes one
JSONL line per frame to ``{out_dir}/pseudo_labels.jsonl``.

Output schema is IDENTICAL to bulk_pseudo_label.py so downstream tools
(info.pkl builders, training configs) work unchanged.

Usage:
  python bulk_pseudo_label_v1.py \\
      --tfrec-dir /tmp/waymo143_train_tfrec \\
      --frame-stride 5 --max-frames 1500 \\
      --out-dir /tmp/waymo_pseudo_train_v1
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from collections import Counter
from typing import Dict, List

import cv2
import numpy as np
import torch

sys.path.insert(0, '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning')

# Importing extract_waymo_v1 installs the protobuf-bytes patch for the
# range-image parser.
from DeepDataMiningLearning.detection3d.phase2a.extract_waymo_v1 import (
    _frame_to_points, _frame_to_camera_calib,
)
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from waymo_open_dataset import dataset_pb2

from DeepDataMiningLearning.detection3d.phase2a.cluster_proposer import (
    propose_clusters)
from DeepDataMiningLearning.detection3d.phase2a.cam2d_proposer import (
    Cam2DProposer)
from DeepDataMiningLearning.detection3d.phase2a.fusion import fuse
from DeepDataMiningLearning.detection3d.phase2a.vlm_voter import VLMVoter

# OpenCV/MMCV (Standard Camera) — same convention bulk_pseudo_label.py uses.
R_OC_WC = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
T_OC_WC = np.eye(4); T_OC_WC[:3, :3] = R_OC_WC

NUM_CAMS = 6
SLOT_TO_WAYMO_CAM = {0: 1, 1: 3, 2: 2, 3: None, 4: 4, 5: 5}


def _build_lidar2img_orig(cam_calib_for_id: Dict) -> np.ndarray:
    K = cam_calib_for_id['intrinsic_4x4']
    c2v = cam_calib_for_id['cam2vehicle']
    return K @ T_OC_WC @ np.linalg.inv(c2v)


def _process_frame(frame, frame_idx, seg, cam_proposer, vlm, args):
    """Run the pseudo-label pipeline on a single decoded Waymo Frame."""
    lidar_np = _frame_to_points(frame).astype(np.float32)  # (N,5) vehicle frame
    cam_calib = _frame_to_camera_calib(frame)              # {cam_id: dict}

    # ---- Validator A — LiDAR clusters
    a_props = propose_clusters(lidar_np, max_range=55.0)

    # ---- Validator B — per-camera 2D detect + LiDAR-depth lift
    b_props = []
    images_by_slot, lidar2img_by_slot = {}, {}
    cid_to_jpg = {img.name: img.image for img in frame.images}

    for slot in range(NUM_CAMS):
        waymo_id = SLOT_TO_WAYMO_CAM[slot]
        if waymo_id is None or waymo_id not in cid_to_jpg:
            continue
        # Decode JPEG bytes → BGR uint8 → RGB uint8 (Faster R-CNN expects RGB)
        jpg = np.frombuffer(cid_to_jpg[waymo_id], dtype=np.uint8)
        bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images_by_slot[slot] = img
        lidar2img_by_slot[slot] = _build_lidar2img_orig(cam_calib[waymo_id])
        det2d = cam_proposer.detect_2d(img)
        lifts = cam_proposer.lift_to_3d(
            det2d, np.asarray(lidar_np[:, :3]),
            lidar2img_by_slot[slot], cam_slot=slot,
            image_hw=img.shape[:2])
        b_props.extend(lifts)

    # ---- Fuse
    pseudo = fuse(a_props, b_props,
                  vlm_voter=vlm,
                  images_by_slot=images_by_slot if vlm else None,
                  lidar2img_by_slot=lidar2img_by_slot if vlm else None,
                  vlm_min_conf=args.vlm_conf,
                  skip_a_only=args.skip_a_only,
                  debug=False)

    return dict(
        frame_idx=int(frame_idx),
        segment=str(seg),
        timestamp=int(frame.timestamp_micros),
        n_a=int(len(a_props)),
        n_b=int(len(b_props)),
        n_pseudo=int(len(pseudo)),
        pseudo=[p.to_dict() for p in pseudo],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tfrec-dir', required=True,
                    help='Directory containing extracted .tfrecord files.')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-frames', type=int, default=-1,
                    help='Max frames to label total (after stride). -1 = all.')
    ap.add_argument('--frame-stride', type=int, default=5,
                    help='Within each segment, keep every N-th frame.')
    ap.add_argument('--use-vlm', action='store_true')
    ap.add_argument('--frcnn-thresh', type=float, default=0.50)
    ap.add_argument('--vlm-conf', type=float, default=0.80)
    ap.add_argument('--skip-a-only', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--progress-every', type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    tfrecs = sorted(f for f in os.listdir(args.tfrec_dir)
                    if f.endswith('.tfrecord'))
    print(f'[bulk-v1] {len(tfrecs)} tfrecords in {args.tfrec_dir}')

    print(f'[bulk-v1] loading Faster R-CNN ...')
    cam_proposer = Cam2DProposer(device=args.device,
                                  score_thresh=args.frcnn_thresh)
    vlm = None
    if args.use_vlm:
        cache = osp.join(args.out_dir, 'vlm_cache.json')
        print(f'[bulk-v1] enabling VLM (cache={cache})')
        vlm = VLMVoter(cache_path=cache, max_calls_per_minute=180)

    jsonl_path = osp.join(args.out_dir, 'pseudo_labels.jsonl')
    counts = Counter(); total_pseudo = 0; total_frames = 0; t0 = time.time()
    stop = False
    with open(jsonl_path, 'w') as jf:
        for tfi, fn in enumerate(tfrecs):
            if stop: break
            ds = tf.data.TFRecordDataset(osp.join(args.tfrec_dir, fn),
                                          compression_type='')
            seg = None
            t_tf0 = time.time()
            for i, data in enumerate(ds):
                if i % args.frame_stride != 0:
                    continue
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytes(data.numpy()))
                if seg is None:
                    seg = frame.context.name
                try:
                    rec = _process_frame(frame, i, seg, cam_proposer, vlm, args)
                except Exception as e:
                    print(f'[bulk-v1]   frame {fn}:{i} failed: '
                          f'{type(e).__name__}: {e}')
                    continue
                jf.write(json.dumps(rec) + '\n')
                jf.flush()
                for p in rec['pseudo']:
                    counts[p['cls']] += 1
                total_pseudo += rec['n_pseudo']
                total_frames += 1
                if total_frames % args.progress_every == 0:
                    elapsed = time.time() - t0
                    rate = total_frames / elapsed
                    print(f'[bulk-v1]   {total_frames} frames  '
                          f'rate={rate:.2f} fr/s  '
                          f'elapsed={elapsed/60:.1f}min  '
                          f'pseudo/frame={total_pseudo/total_frames:.1f}  '
                          f'by_cls={dict(counts)}')
                if args.max_frames > 0 and total_frames >= args.max_frames:
                    stop = True
                    break
            print(f'[bulk-v1] [{tfi+1}/{len(tfrecs)}] {fn}: '
                  f'cumulative {total_frames} frames, '
                  f'tfrec took {time.time()-t_tf0:.1f}s')

    if vlm is not None:
        vlm._save_cache()
        print(f'[bulk-v1] VLM final stats: {vlm.stats()}')

    print()
    print(f'[bulk-v1] DONE. {total_frames} frames, {total_pseudo} pseudo-labels')
    print(f'[bulk-v1] by class: {dict(counts)}')
    print(f'[bulk-v1] output → {jsonl_path}')


if __name__ == '__main__':
    main()
