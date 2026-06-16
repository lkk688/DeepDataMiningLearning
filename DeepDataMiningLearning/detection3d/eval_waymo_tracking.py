"""
Run our HungarianTracker3D on Waymo val detections and compute basic
tracking metrics. Picks N segments, iterates all consecutive frames in
each, runs B14/B15 inference, transforms predictions to global frame,
feeds the tracker.

Outputs:
  * per_track.json: every tracklet (global trajectory, class, score)
  * tracking_metrics.json: per-class MOTA, ID switches, recall, etc.

Usage:
  python eval_waymo_tracking.py --config <cfg> --checkpoint <ckpt> \
      --num-segments 5 --out-dir /tmp/waymo_tracking_v9
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning')

from DeepDataMiningLearning.detection3d.simple_infer_utils import load_multitask_model
from DeepDataMiningLearning.detection3d.dataset_waymo_mmdet3d import WaymoMMDet3DZeroShot
from DeepDataMiningLearning.detection3d.class_map_waymo_to_nus import (
    nus_label_to_transfer, waymo_type_to_transfer, NUS_DET_CLASSES_10,
)
from DeepDataMiningLearning.bevdet.tracker3d import HungarianTracker3D


# nuScenes detection class index → Waymo tracking class name
# (Drop signs/cones/barriers/construction_vehicle that don't track in Waymo)
NUS_TO_WAYMO_TRACK = {
    0: 'car',         # car
    1: 'truck',       # truck (collapse to truck for tracking)
    3: 'bus',         # bus
    4: 'trailer',     # trailer
    6: 'motorcycle',  # motorcycle
    7: 'bicycle',     # bicycle
    8: 'pedestrian',  # pedestrian
}
# Waymo type (1=Vehicle, 2=Ped, 4=Cyclist) → bucket name
WAYMO_TYPE_TO_BUCKET = {1: 'Vehicle', 2: 'Pedestrian', 4: 'Cyclist'}
# Our predictions need to map to those same 3 buckets.
NUS_TO_BUCKET = {
    0: 'Vehicle', 1: 'Vehicle', 2: 'Vehicle', 3: 'Vehicle', 4: 'Vehicle',
    6: 'Cyclist', 7: 'Cyclist', 8: 'Pedestrian',
}
# Make HungarianTracker3D accept these names too — but its TRACKING_CLASSES_7
# uses car/truck/.../bicycle. So pass through NUS_TO_WAYMO_TRACK to get a
# tracker-acceptable name and bucket separately for eval.


def _run_inference_one(model, frame: Dict, device: str) -> Dict:
    """Run model on one frame; apply V6/V9 y-flip post-processing."""
    inputs = frame['inputs']
    inputs['points'] = [p.to(device) for p in inputs['points']]
    inputs['img']    = [im.to(device) for im in inputs['img']]
    frame['inputs'] = inputs
    with torch.no_grad():
        out = model.test_step(frame)
    pred = out[0].pred_instances_3d
    boxes  = pred.bboxes_3d.tensor.detach().cpu().numpy()
    scores = pred.scores_3d.detach().cpu().numpy()
    labels = pred.labels_3d.detach().cpu().numpy()
    # y-flip predictions (model output frame → Waymo vehicle frame)
    if boxes.shape[0] > 0:
        boxes[:, 1] = -boxes[:, 1]
        if boxes.shape[1] >= 7:
            boxes[:, 6] = -boxes[:, 6]
        if boxes.shape[1] == 9:
            boxes[:, 8] = -boxes[:, 8]
    return dict(boxes=boxes, scores=scores, labels=labels)


def _vehicle_to_global(boxes_v: np.ndarray, T_wv: np.ndarray) -> np.ndarray:
    """Boxes in vehicle frame → global frame using world_from_vehicle."""
    if boxes_v.shape[0] == 0:
        return boxes_v.copy()
    out = boxes_v.copy()
    # xyz position
    xyz_h = np.concatenate([boxes_v[:, :3], np.ones((boxes_v.shape[0], 1),
                                                     dtype=np.float32)], axis=1)
    out[:, :3] = (xyz_h @ T_wv.T)[:, :3]
    # yaw (heading): add rotation around z of T_wv
    R = T_wv[:3, :3]
    delta = float(np.arctan2(R[1, 0], R[0, 0]))
    out[:, 6] = boxes_v[:, 6] + delta
    # velocity (last 2 dims): rotate
    if out.shape[1] == 9:
        vx, vy = boxes_v[:, 7], boxes_v[:, 8]
        out[:, 7] = R[0, 0] * vx + R[0, 1] * vy
        out[:, 8] = R[1, 0] * vx + R[1, 1] * vy
    return out


def _gt_to_global(gt_boxes_v: np.ndarray, gt_types: np.ndarray,
                  gt_ids: List[str], T_wv: np.ndarray) -> List[Dict]:
    """GT boxes (vehicle frame) + Waymo object IDs → global tracks."""
    out = []
    if gt_boxes_v.shape[0] == 0:
        return out
    R = T_wv[:3, :3]
    dyaw = float(np.arctan2(R[1, 0], R[0, 0]))
    xyz_h = np.concatenate([gt_boxes_v[:, :3], np.ones((gt_boxes_v.shape[0], 1),
                                                        dtype=np.float32)], axis=1)
    xyz_g = (xyz_h @ T_wv.T)[:, :3]
    for i in range(gt_boxes_v.shape[0]):
        wtype = int(gt_types[i])
        bucket = WAYMO_TYPE_TO_BUCKET.get(wtype)
        if bucket is None:
            continue
        out.append(dict(
            obj_id=str(gt_ids[i]) if i < len(gt_ids) else f'unk_{i}',
            bucket=bucket,
            xy=(float(xyz_g[i, 0]), float(xyz_g[i, 1])),
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--ckpt-multi', default='')
    ap.add_argument('--waymo-root',
        default='/fs/atipa/data/rnd-liu/Datasets/waymo201')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--num-segments', type=int, default=5)
    ap.add_argument('--num-sweeps', type=int, default=1)
    ap.add_argument('--score-thresh', type=float, default=0.01)
    ap.add_argument('--use-velocity', action='store_true', default=True)
    ap.add_argument('--match-radius-m', type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'[track] loading model from {args.ckpt_multi or args.checkpoint}')
    model, _ = load_multitask_model(
        config_path=args.config,
        checkpoint_path=args.ckpt_multi or args.checkpoint,
        device=args.device,
    )

    print(f'[track] building Waymo iterator from {args.waymo_root}/{args.split}')
    ds = WaymoMMDet3DZeroShot(
        root_dir=args.waymo_root, split=args.split, max_frames=None,
        num_sweeps=args.num_sweeps,
    )

    # Group frame indices by segment in chronological order.
    seg_frames = defaultdict(list)
    for i, (_, ts, seg) in enumerate(ds.base.frame_index):
        seg_frames[seg].append((int(ts), i))
    for seg in seg_frames:
        seg_frames[seg].sort()    # by ts

    segs = sorted(seg_frames.keys())[:args.num_segments]
    print(f'[track] running on {len(segs)} segments')

    # Aggregate stats
    all_tracks = []          # list of finalized tracklets across segments
    eval_per_class = defaultdict(lambda: dict(
        tp=0, fp=0, fn=0, ids=0, gt_total=0))

    t0 = time.time()
    for seg_idx, seg in enumerate(segs):
        frame_list = seg_frames[seg]
        print(f'[track] [{seg_idx+1}/{len(segs)}] {seg}  ({len(frame_list)} frames)')

        tracker = HungarianTracker3D(use_velocity=args.use_velocity)
        prev_ts = None
        # Track-id → bucket → set of (frame_idx, det_xy) for IDS counting
        track_history_buckets = defaultdict(list)
        # Per-frame: GT IDs and pred tracklets (for IDS / TP / FP)
        per_frame_pairs = []   # list of (frame_idx, gt_list, pred_list)

        for ts_pos, (ts, frame_idx) in enumerate(frame_list):
            try:
                frame = ds.get_frame(frame_idx)
            except Exception as e:
                print(f'  [{ts_pos}] skip: {e}')
                continue
            meta = frame['data_samples'][0].metainfo
            T_wv = ds.base[frame_idx][1].get('world_from_vehicle')
            if T_wv is None:
                T_wv = np.eye(4, dtype=np.float32)
            else:
                T_wv = T_wv.cpu().numpy() if torch.is_tensor(T_wv) else np.asarray(T_wv)

            # Run inference
            try:
                dets = _run_inference_one(model, frame, args.device)
            except Exception as e:
                print(f'  [{ts_pos}] infer fail: {e}')
                continue
            boxes_v = dets['boxes']
            scores  = dets['scores']
            labels  = dets['labels']

            # Filter by score + class bucket
            keep = []
            buckets = []
            tracker_class_names = []
            for k in range(boxes_v.shape[0]):
                if scores[k] < args.score_thresh:
                    continue
                bucket = NUS_TO_BUCKET.get(int(labels[k]))
                trk_name = NUS_TO_WAYMO_TRACK.get(int(labels[k]))
                if bucket is None or trk_name is None:
                    continue
                keep.append(k); buckets.append(bucket); tracker_class_names.append(trk_name)
            if not keep:
                boxes_g = np.zeros((0, 9), dtype=np.float32)
            else:
                keep = np.asarray(keep, dtype=np.int64)
                boxes_g = _vehicle_to_global(boxes_v[keep], T_wv)

            # Feed tracker
            dt = 0.1 if prev_ts is None else max(0.05,
                (ts - prev_ts) * 1e-6)
            tracker_dets = [
                dict(box_global=boxes_g[i],
                     score=float(scores[keep[i]]),
                     class_track=tracker_class_names[i])
                for i in range(len(keep))
            ]
            tracker.update(tracker_dets, dt_seconds=dt,
                           sample_token=f'{seg}_{frame_idx}')
            prev_ts = ts

            # Collect per-frame predicted tracks + GT for evaluation
            pred_pos_by_bucket = defaultdict(list)
            for t in tracker.tracklets:
                if t.age != 0:
                    continue
                # Map back from tracker class to bucket
                # (NUS_TO_WAYMO_TRACK output 'car/truck/bus/trailer' → Vehicle,
                #  'motorcycle/bicycle' → Cyclist, 'pedestrian' → Pedestrian)
                if t.class_name in ('car', 'truck', 'bus', 'trailer'):
                    b = 'Vehicle'
                elif t.class_name in ('motorcycle', 'bicycle'):
                    b = 'Cyclist'
                elif t.class_name == 'pedestrian':
                    b = 'Pedestrian'
                else:
                    continue
                pred_pos_by_bucket[b].append((t.track_id, t.last_box[0], t.last_box[1]))

            # GT
            _, target = ds.base[frame_idx]
            gt_boxes = target.get('boxes_3d')
            gt_types = target.get('labels')
            if torch.is_tensor(gt_boxes): gt_boxes = gt_boxes.cpu().numpy()
            if torch.is_tensor(gt_types): gt_types = gt_types.cpu().numpy()
            gt_ids = [f'{seg}_obj{j}' for j in range(len(gt_boxes))]
            gt_g = _gt_to_global(np.asarray(gt_boxes), np.asarray(gt_types),
                                  gt_ids, T_wv)
            per_frame_pairs.append((frame_idx, gt_g, pred_pos_by_bucket))

        # ------------------ Per-segment tracking metrics ------------------
        # Simple MOTA-style: for each frame, match GT to PRED by 2m center
        # distance per class bucket. TP=matched, FP=unmatched pred,
        # FN=unmatched GT. IDS=GT->different track than last time.
        seg_track_per_gt_obj = {}    # gt obj_id → list[(frame, track_id)]
        for frame_idx, gt_list, pred_by_bucket in per_frame_pairs:
            for bucket in ('Vehicle', 'Pedestrian', 'Cyclist'):
                gts_b = [g for g in gt_list if g['bucket'] == bucket]
                preds_b = pred_by_bucket.get(bucket, [])
                eval_per_class[bucket]['gt_total'] += len(gts_b)
                # Greedy match
                used_pred = set()
                for g in gts_b:
                    best = -1; best_d = args.match_radius_m
                    for pi, (pid, px, py) in enumerate(preds_b):
                        if pi in used_pred: continue
                        d = ((g['xy'][0]-px)**2 + (g['xy'][1]-py)**2)**0.5
                        if d < best_d:
                            best = pi; best_d = d
                    if best >= 0:
                        used_pred.add(best)
                        eval_per_class[bucket]['tp'] += 1
                        pred_pid = preds_b[best][0]
                        prev = seg_track_per_gt_obj.get(g['obj_id'])
                        if prev is not None and prev != pred_pid:
                            eval_per_class[bucket]['ids'] += 1
                        seg_track_per_gt_obj[g['obj_id']] = pred_pid
                    else:
                        eval_per_class[bucket]['fn'] += 1
                # Unmatched preds = false positives
                for pi in range(len(preds_b)):
                    if pi not in used_pred:
                        eval_per_class[bucket]['fp'] += 1

        # Collect finalized tracklets for this segment
        finalized = tracker.finalize_history()
        all_tracks.append({'segment': seg,
                           'n_frames': len(frame_list),
                           'n_tracks': len(finalized.get('tracks', []))
                                       if isinstance(finalized, dict) else len(finalized)})

    # ------------------ Aggregate metrics ------------------
    print()
    print('=' * 60)
    print(f'  WAYMO TRACKING  ({len(segs)} segments)')
    print('=' * 60)
    summary = {}
    for bucket in ('Vehicle', 'Pedestrian', 'Cyclist'):
        m = eval_per_class[bucket]
        tp, fp, fn, ids, gt = m['tp'], m['fp'], m['fn'], m['ids'], m['gt_total']
        recall = tp / max(1, gt)
        prec   = tp / max(1, tp + fp)
        # MOTA = 1 - (FN + FP + IDS) / GT, lower bounded at 0
        mota = max(0.0, 1.0 - (fn + fp + ids) / max(1, gt))
        summary[bucket] = dict(gt=gt, tp=tp, fp=fp, fn=fn, ids=ids,
                                recall=recall, precision=prec, mota=mota)
        print(f'  {bucket:12s}  GT={gt:5d}  TP={tp:4d}  FP={fp:4d}  '
              f'FN={fn:4d}  IDS={ids:3d}  '
              f'R={recall:.3f}  P={prec:.3f}  MOTA={mota:.3f}')
    # Macro MOTA across classes
    macro_mota = float(np.mean([summary[b]['mota'] for b in summary]))
    print(f'  Macro MOTA: {macro_mota:.3f}')
    print('=' * 60)

    json.dump(summary, open(osp.join(args.out_dir, 'tracking_metrics.json'), 'w'),
              indent=2)
    json.dump(all_tracks, open(osp.join(args.out_dir, 'per_segment_tracks.json'), 'w'),
              indent=2)
    print(f'[track] saved → {args.out_dir}')
    print(f'[track] elapsed: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
