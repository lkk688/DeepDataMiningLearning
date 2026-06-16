"""
Zero-shot evaluation: run a B10c-style model (trained on nuScenes) on the
Waymo v2.0.1 validation set and report cross-dataset detection metrics.

Pipeline
--------
1. Load the multitask wrapper (B10c arch + B10c epoch_3 weights) — uses
   ``load_multitask_model`` from simple_infer_utils.
2. For each Waymo val frame, build the mmdet3d-style ``(inputs, data_samples)``
   dict via ``WaymoMMDet3DZeroShot.get_frame(i)``.
3. Run ``model.test_step(data)``; get per-frame 3D detections in
   **Waymo vehicle frame** (which matches our nuScenes lidar frame convention).
4. Filter detections to the 3-class transfer taxonomy via
   ``class_map_waymo_to_nus``.
5. Per-frame center-distance match against the Waymo GT (also filtered
   to the 3-class taxonomy). Threshold: 2.0 m (matches the standard
   nuScenes detection match).
6. Aggregate per-class precision / recall / mAP@2m.

This is **NOT** the official Waymo metric (which is mAP/mAPH at LEVEL_1/LEVEL_2
difficulty with IoU-based matching). It's a quick cross-dataset transfer
proxy that lets us see whether B10c localizes objects correctly on Waymo
without the licensing detour. For a paper-grade Waymo number we'd need
to convert outputs to Waymo's submission format and run their devkit.

Usage::

    python eval_waymo_zeroshot.py \\
        --config       /path/to/B10c_flow_guided_warmstart_fixed.py \\
        --checkpoint   /path/to/B10c/epoch_3.pth \\
        --ckpt-multi   /path/to/B10c/epoch_3_multitask.pth \\
        --waymo-root   /fs/atipa/data/rnd-liu/Datasets/waymo201 \\
        --split        validation \\
        --out-dir      /tmp/waymo_zs_b10c \\
        --max-frames   50            # for the smoke run; -1 for full

Outputs (in ``--out-dir``):
    waymo_per_frame_dets.json   # raw per-frame outputs (vehicle frame)
    transfer_metrics.json        # per-class precision/recall/mAP@2m
"""
from __future__ import annotations
import argparse
import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Make DeepDataMiningLearning importable.
_HERE = osp.dirname(osp.abspath(__file__))
_DDML_ROOT = osp.dirname(osp.dirname(_HERE))
if _DDML_ROOT not in sys.path:
    sys.path.insert(0, _DDML_ROOT)

from DeepDataMiningLearning.detection3d.simple_infer_utils import (
    load_multitask_model,
)
from DeepDataMiningLearning.detection3d.dataset_waymo_mmdet3d import (
    WaymoMMDet3DZeroShot,
)
from DeepDataMiningLearning.detection3d.class_map_waymo_to_nus import (
    NUS_DET_CLASSES_10,
    TRANSFER_CLASSES,
    nus_label_to_transfer,
    waymo_type_to_transfer,
)


def _run_inference_one_frame(model, frame: Dict[str, Any], device: str
                              ) -> Dict[str, np.ndarray]:
    """Run model.test_step on one Waymo frame and return per-class
    transfer-taxonomy detections in vehicle frame."""
    # Move tensors to device.
    inputs = frame['inputs']
    inputs['points'] = [p.to(device) for p in inputs['points']]
    inputs['img']    = [im.to(device) for im in inputs['img']]
    frame['inputs'] = inputs

    with torch.no_grad():
        out = model.test_step(frame)

    ds = out[0]
    pred = ds.pred_instances_3d
    boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()    # [N, 9]
    scores = pred.scores_3d.detach().cpu().numpy()
    labels = pred.labels_3d.detach().cpu().numpy()
    # Map nuScenes labels → transfer-class names; drop the rest.
    keep_idx, transfer_cls = [], []
    for i, lab in enumerate(labels):
        tcls = nus_label_to_transfer(int(lab))
        if tcls is None:
            continue
        keep_idx.append(i)
        transfer_cls.append(tcls)
    if not keep_idx:
        return dict(boxes=np.zeros((0, 9), dtype=np.float32),
                    scores=np.zeros((0,), dtype=np.float32),
                    transfer_cls=[])
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    boxes_model = boxes[keep_idx].astype(np.float32)
    # The wrapper feeds the model y-flipped LiDAR points (because the
    # nuScenes-trained model expects that handedness). Predictions come
    # out in the same y-flipped frame; un-flip them so they match the
    # un-mirrored Waymo GT in the eval matcher below.
    boxes_veh = boxes_model.copy()
    boxes_veh[:, 1] = -boxes_model[:, 1]            # y → -y
    if boxes_veh.shape[1] >= 7:
        boxes_veh[:, 6] = -boxes_model[:, 6]        # yaw reflects too
    if boxes_veh.shape[1] == 9:
        boxes_veh[:, 8] = -boxes_model[:, 8]        # v_y reflects (v_x unchanged)
    return dict(
        boxes=boxes_veh,
        scores=scores[keep_idx].astype(np.float32),
        transfer_cls=transfer_cls,
    )


def _gt_to_transfer(meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Filter Waymo GT to the transfer-class taxonomy."""
    boxes = meta.get('waymo_gt_boxes_vehicle', np.zeros((0, 7)))
    types = meta.get('waymo_gt_types', np.zeros((0,), dtype=np.int64))
    if boxes.shape[0] == 0:
        return dict(boxes=boxes, transfer_cls=[])
    keep, transfer_cls = [], []
    for i, t in enumerate(types):
        tcls = waymo_type_to_transfer(int(t))
        if tcls is None:
            continue
        keep.append(i)
        transfer_cls.append(tcls)
    if not keep:
        return dict(boxes=boxes[:0], transfer_cls=[])
    keep = np.asarray(keep, dtype=np.int64)
    return dict(boxes=boxes[keep].astype(np.float32), transfer_cls=transfer_cls)


def _match_and_score(per_frame_records: List[Dict[str, Any]],
                      dist_thresh: float = 2.0,
                      score_thresh: float = 0.10) -> Dict[str, Dict]:
    """
    Per-class precision / recall / mAP@center-dist-{dist_thresh} m.

    Greedy match by ascending distance; only matches within ``dist_thresh``
    and same class count as TP. Each detection above ``score_thresh`` and
    each GT counted exactly once.
    """
    per_class_tp_scores: Dict[str, List[float]] = {c: [] for c in TRANSFER_CLASSES}
    per_class_fp_scores: Dict[str, List[float]] = {c: [] for c in TRANSFER_CLASSES}
    per_class_n_gt:      Dict[str, int]         = {c: 0   for c in TRANSFER_CLASSES}

    for r in per_frame_records:
        dets = r['dets']
        gt = r['gt']
        gt_classes = gt['transfer_cls']
        gt_centers = gt['boxes'][:, :2] if gt['boxes'].shape[0] else np.zeros((0, 2))
        gt_used = np.zeros(len(gt_classes), dtype=bool)
        for c in gt_classes:
            per_class_n_gt[c] += 1
        det_boxes = dets['boxes']
        det_scores = dets['scores']
        det_cls = dets['transfer_cls']
        if det_boxes.shape[0] == 0:
            continue
        # Score-threshold filter first.
        keep = det_scores >= score_thresh
        det_boxes, det_scores = det_boxes[keep], det_scores[keep]
        det_cls = [det_cls[i] for i, k in enumerate(keep) if k]
        if det_boxes.shape[0] == 0:
            continue
        # Greedy by descending score.
        order = np.argsort(-det_scores)
        det_centers = det_boxes[:, :2]
        for i in order:
            sc = float(det_scores[i])
            dc = det_cls[i]
            if not gt_classes:
                per_class_fp_scores[dc].append(sc)
                continue
            d = np.linalg.norm(gt_centers - det_centers[i], axis=1)
            # Only consider matching GT of same class:
            same_cls = np.array([g == dc for g in gt_classes])
            d_masked = np.where(same_cls & ~gt_used, d, np.inf)
            j = int(np.argmin(d_masked)) if d_masked.size else -1
            if j >= 0 and d_masked[j] < dist_thresh:
                per_class_tp_scores[dc].append(sc)
                gt_used[j] = True
            else:
                per_class_fp_scores[dc].append(sc)

    out: Dict[str, Dict] = {}
    for c in TRANSFER_CLASSES:
        tp = np.asarray(per_class_tp_scores[c], dtype=np.float32)
        fp = np.asarray(per_class_fp_scores[c], dtype=np.float32)
        n_gt = per_class_n_gt[c]
        n_tp = int(tp.size)
        n_fp = int(fp.size)
        prec = n_tp / max(1, n_tp + n_fp)
        rec  = n_tp / max(1, n_gt)
        # Crude AP: rank all dets by score, sweep PR.
        all_scores = np.concatenate([tp, fp]) if (tp.size or fp.size) else np.zeros((0,))
        is_tp = np.concatenate([
            np.ones_like(tp, dtype=np.int32),
            np.zeros_like(fp, dtype=np.int32),
        ]) if all_scores.size else np.zeros((0,), dtype=np.int32)
        if all_scores.size > 0:
            order = np.argsort(-all_scores)
            is_tp = is_tp[order]
            tp_cum = np.cumsum(is_tp)
            fp_cum = np.cumsum(1 - is_tp)
            precisions = tp_cum / np.maximum(1, tp_cum + fp_cum)
            recalls = tp_cum / max(1, n_gt)
            # 11-point interpolation
            ap = 0.0
            for r_thr in np.linspace(0, 1, 11):
                mask = recalls >= r_thr
                ap += (precisions[mask].max() if mask.any() else 0.0)
            ap /= 11.0
        else:
            ap = 0.0
        out[c] = dict(
            n_gt=n_gt, tp=n_tp, fp=n_fp,
            precision=prec, recall=rec,
            ap_11pt_at_2m=float(ap),
        )

    macro_ap = float(np.mean([v['ap_11pt_at_2m'] for v in out.values()]))
    return {'per_class': out, 'mAP_macro_at_2m': macro_ap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--ckpt-multi', default='')
    ap.add_argument('--waymo-root', required=True)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-frames', type=int, default=-1,
                    help='Limit number of Waymo val frames (smoke run); -1=all.')
    ap.add_argument('--frame-stride', type=int, default=1,
                    help='Stride over the val list (e.g. 30 = every 30th '
                         'frame). Use together with --max-frames to get '
                         'cross-segment coverage cheaply.')
    ap.add_argument('--dist-thresh', type=float, default=2.0)
    ap.add_argument('--score-thresh', type=float, default=0.10)
    ap.add_argument('--num-sweeps', type=int, default=1,
                    help='Multi-sweep input for the model. Use 5 for '
                         'B15+ checkpoints trained with sweeps.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'[zs] Loading model from {args.ckpt_multi or args.checkpoint} ...')
    model, _ = load_multitask_model(
        config_path=args.config,
        checkpoint_path=args.ckpt_multi or args.checkpoint,
        device=args.device,
    )

    print(f'[zs] Building Waymo iterator from {args.waymo_root}/{args.split} ...')
    # Build the full base, then sub-sample with stride to cover many segments
    # without paying full inference cost. We still cap at max_frames *after*
    # striding (so --frame-stride=30 --max-frames=200 → 200 strided samples
    # spread across ~6000 source frames).
    ds = WaymoMMDet3DZeroShot(
        root_dir=args.waymo_root,
        split=args.split,
        max_frames=None,
        num_sweeps=args.num_sweeps,
    )
    indices = list(range(0, len(ds), max(1, args.frame_stride)))
    if args.max_frames > 0:
        indices = indices[:args.max_frames]
    n = len(indices)
    print(f'[zs] {n} strided frames to process '
          f'(stride={args.frame_stride}, source pool={len(ds)})')

    records: List[Dict] = []
    t0 = time.time()
    for k, i in enumerate(indices):
        try:
            frame = ds.get_frame(i)
        except Exception as e:
            print(f'[zs] frame {i} skipped: {type(e).__name__}: {e}')
            continue
        meta = frame['data_samples'][0].metainfo
        try:
            dets = _run_inference_one_frame(model, frame, args.device)
        except Exception as e:
            print(f'[zs] frame {i} inference failed: {type(e).__name__}: {e}')
            continue
        gt = _gt_to_transfer(meta)
        records.append(dict(
            sample_idx=int(meta['sample_idx']),
            segment=str(meta.get('waymo_segment', '')),
            timestamp=int(meta.get('waymo_timestamp', 0)),
            dets=dets,
            gt=gt,
        ))
        if (k + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (n - k - 1)
            print(f'[zs]   {k+1}/{n}  elapsed={elapsed:.0f}s  eta={eta:.0f}s')

    print(f'[zs] inference done. Computing transfer metrics ...')
    metrics = _match_and_score(records, dist_thresh=args.dist_thresh,
                               score_thresh=args.score_thresh)
    with open(osp.join(args.out_dir, 'transfer_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print()
    print('=' * 60)
    print(f'  ZERO-SHOT TRANSFER  (B10c trained on nuScenes → Waymo val)')
    print('=' * 60)
    print(f'  match radius: {args.dist_thresh} m   score_thresh: {args.score_thresh}')
    print()
    for c, v in metrics['per_class'].items():
        print(f'  {c:12s}  n_gt={v["n_gt"]:5d}  TP={v["tp"]:5d}  FP={v["fp"]:5d}'
              f'   P={v["precision"]:.3f}  R={v["recall"]:.3f}  AP@2m={v["ap_11pt_at_2m"]:.3f}')
    print()
    print(f'  Macro mAP @ 2 m: {metrics["mAP_macro_at_2m"]:.3f}')
    print('=' * 60)

    # Save raw per-frame dets too (compact JSON).
    raw = []
    for r in records:
        raw.append(dict(
            sample_idx=r['sample_idx'], segment=r['segment'],
            timestamp=r['timestamp'],
            dets=dict(
                boxes=r['dets']['boxes'].tolist(),
                scores=r['dets']['scores'].tolist(),
                transfer_cls=r['dets']['transfer_cls'],
            ),
            gt=dict(
                boxes=r['gt']['boxes'].tolist(),
                transfer_cls=r['gt']['transfer_cls'],
            ),
        ))
    with open(osp.join(args.out_dir, 'waymo_per_frame_dets.json'), 'w') as f:
        json.dump(raw, f)
    print(f'[zs] per-frame dets saved → {args.out_dir}/waymo_per_frame_dets.json')


if __name__ == '__main__':
    main()
