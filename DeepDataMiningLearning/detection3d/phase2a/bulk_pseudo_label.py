"""
Bulk pseudo-label generation on Waymo v2.0.1 validation split.

Produces a single JSON ``pseudo_labels.jsonl`` (one line per frame) with
all pseudo-labels, suitable for downstream fine-tune dataset conversion.

Usage:
  python bulk_pseudo_label.py \
      --max-frames 2000 --frame-stride 3 \
      --use-vlm \
      --out-dir /tmp/waymo_pseudo_val_2k
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

import numpy as np
import torch

sys.path.insert(0, '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning')

from DeepDataMiningLearning.detection3d.phase2a.cluster_proposer import (
    propose_clusters)
from DeepDataMiningLearning.detection3d.phase2a.cam2d_proposer import (
    Cam2DProposer)
from DeepDataMiningLearning.detection3d.phase2a.fusion import fuse
from DeepDataMiningLearning.detection3d.phase2a.vlm_voter import VLMVoter
from DeepDataMiningLearning.detection3d.dataset_waymo_mmdet3d import (
    WaymoMMDet3DZeroShot, _load_camera_calibration, SLOT_TO_WAYMO_CAM,
    NUM_CAMS)


R_OC_WC = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
T_OC_WC = np.eye(4); T_OC_WC[:3, :3] = R_OC_WC


def _build_lidar2img_orig(cam_calib_for_id: Dict) -> np.ndarray:
    K = cam_calib_for_id['intrinsic_4x4']
    c2v = cam_calib_for_id['cam2vehicle']
    return K @ T_OC_WC @ np.linalg.inv(c2v)


def process_frame(ds, idx, cam_proposer, vlm, args):
    lidar, target = ds.base[idx]
    fname, ts, seg = ds.base.frame_index[idx]
    cam_calib = _load_camera_calibration(osp.join(ds.cam_calib_dir, fname))
    if torch.is_tensor(lidar):
        lidar_np = lidar.cpu().numpy()
    else:
        lidar_np = np.asarray(lidar)

    # --- Validator A: LiDAR clusters
    a_props = propose_clusters(lidar_np, max_range=55.0)

    # --- Validator B: per-camera 2D detect + 3D lift
    b_props = []
    images_by_slot, lidar2img_by_slot = {}, {}
    surround = target.get('surround_views', []) or []
    cid_to_img = {int(d.get('camera_id', -1)): d['image']
                   for d in surround if 'image' in d}

    for slot in range(NUM_CAMS):
        waymo_id = SLOT_TO_WAYMO_CAM[slot]
        if waymo_id is None or waymo_id not in cid_to_img:
            continue
        img = cid_to_img[waymo_id]
        if torch.is_tensor(img): img = img.cpu().numpy()
        img = np.asarray(img, dtype=np.uint8)
        images_by_slot[slot] = img
        lidar2img_by_slot[slot] = _build_lidar2img_orig(cam_calib[waymo_id])
        det2d = cam_proposer.detect_2d(img)
        lifts = cam_proposer.lift_to_3d(
            det2d, np.asarray(lidar_np[:, :3]),
            lidar2img_by_slot[slot], cam_slot=slot,
            image_hw=img.shape[:2])
        b_props.extend(lifts)

    # --- Fuse
    pseudo = fuse(a_props, b_props,
                  vlm_voter=vlm,
                  images_by_slot=images_by_slot if vlm else None,
                  lidar2img_by_slot=lidar2img_by_slot if vlm else None,
                  vlm_min_conf=args.vlm_conf,
                  skip_a_only=args.skip_a_only,
                  debug=False)

    return dict(
        frame_idx=int(idx),
        segment=str(seg),
        timestamp=int(ts),
        n_a=int(len(a_props)),
        n_b=int(len(b_props)),
        n_pseudo=int(len(pseudo)),
        pseudo=[p.to_dict() for p in pseudo],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--waymo-root',
        default='/fs/atipa/data/rnd-liu/Datasets/waymo201')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--max-frames', type=int, default=2000,
        help='Number of frames to label (after stride).')
    ap.add_argument('--frame-stride', type=int, default=3,
        help='Stride over the source frame list (3 covers ~5.5K frames).')
    ap.add_argument('--use-vlm', action='store_true')
    ap.add_argument('--frcnn-thresh', type=float, default=0.50)
    ap.add_argument('--vlm-conf', type=float, default=0.80)
    ap.add_argument('--skip-a-only', action='store_true',
        help='Skip A-only branch (saves ~half VLM calls; lower recall).')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--progress-every', type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg_path = osp.join(args.out_dir, 'config.json')
    json.dump(vars(args), open(cfg_path, 'w'), indent=2)

    print(f'[bulk] building Waymo iterator: {args.waymo_root}/{args.split}')
    ds = WaymoMMDet3DZeroShot(
        root_dir=args.waymo_root, split=args.split, max_frames=None,
    )
    indices = list(range(0, len(ds), max(1, args.frame_stride)))[:args.max_frames]
    print(f'[bulk] {len(indices)} frames (stride={args.frame_stride}, '
          f'source pool={len(ds)})')

    print(f'[bulk] loading Faster R-CNN ...')
    cam_proposer = Cam2DProposer(device=args.device,
                                  score_thresh=args.frcnn_thresh)
    vlm = None
    if args.use_vlm:
        cache = osp.join(args.out_dir, 'vlm_cache.json')
        print(f'[bulk] enabling VLM (cache={cache})')
        vlm = VLMVoter(cache_path=cache, max_calls_per_minute=180)

    jsonl_path = osp.join(args.out_dir, 'pseudo_labels.jsonl')
    counts = Counter(); total_pseudo = 0; t0 = time.time()
    with open(jsonl_path, 'w') as jf:
        for k, i in enumerate(indices):
            try:
                rec = process_frame(ds, i, cam_proposer, vlm, args)
            except Exception as e:
                print(f'[bulk] frame {i} failed: {type(e).__name__}: {e}')
                continue
            jf.write(json.dumps(rec) + '\n')
            jf.flush()
            for p in rec['pseudo']:
                counts[p['cls']] += 1
            total_pseudo += rec['n_pseudo']
            if (k + 1) % args.progress_every == 0:
                elapsed = time.time() - t0
                eta = elapsed / (k + 1) * (len(indices) - k - 1)
                rate = (k + 1) / elapsed
                print(f'[bulk]   {k+1}/{len(indices)}  '
                      f'rate={rate:.2f} fr/s  '
                      f'elapsed={elapsed/60:.1f}min  '
                      f'eta={eta/60:.1f}min  '
                      f'pseudo/frame={total_pseudo/(k+1):.1f}')
                if vlm is not None and (k + 1) % (args.progress_every * 4) == 0:
                    vlm._save_cache()
                    print(f'[bulk]     VLM stats: {vlm.stats()}')

    if vlm is not None:
        vlm._save_cache()
        print(f'[bulk] VLM final stats: {vlm.stats()}')
    print()
    print(f'[bulk] DONE. {len(indices)} frames, {total_pseudo} pseudo-labels')
    print(f'[bulk] by class: {dict(counts)}')
    print(f'[bulk] output → {jsonl_path}')


if __name__ == '__main__':
    main()
