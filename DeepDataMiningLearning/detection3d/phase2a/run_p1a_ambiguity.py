"""P1a — VLM ambiguity ablation.

Inputs:
  * ``ambiguous_candidates.jsonl`` from gen_ambiguous_candidates.py
  * Local vLLM Qwen-2.5-VL server (default) OR NIM Gemma-3n (--backend nim)
  * Waymo v1.4.3 GT (loaded from the extracted .npz files)

For each candidate (A-only / B-only / class-conflict), we:
  1. Decode the candidate's camera image from the original tfrecord
     (cached per (segment, frame_idx)).
  2. Crop + draw a red rectangle around the candidate's 2D pixel box.
  3. Send to the VLM with the red-box prompt.
  4. Match the candidate's 3D box against Waymo GT at 2 m
     center-distance to record whether the candidate is a true object.

Output: ``p1a_results.json`` containing aggregate stats:
  * for each (source ∈ {A-only, B-only, conflict}, class ∈ {V,P,C}):
      n_total, n_vlm_kept_high_conf, n_gt_match,
      precision_kept_vs_gt, precision_drop_vs_gt
  * VLM cost: api_calls, latency_avg
"""
from __future__ import annotations

import argparse
import io
import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning')

# We re-decode the tfrecord directly to get the candidate's source image.
from DeepDataMiningLearning.detection3d.phase2a.extract_waymo_v1 import (
    _frame_to_camera_calib,
)
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from waymo_open_dataset import dataset_pb2


SLOT_TO_WAYMO_CAM = {0: 1, 1: 3, 2: 2, 3: None, 4: 4, 5: 5}

# Type id (Waymo v1.4.3) -> transfer class
WAYMO_TYPE_TO_TRANSFER = {1: 'Vehicle', 2: 'Pedestrian', 4: 'Cyclist'}


def _load_voter(backend: str, **kwargs):
    if backend == 'local':
        from DeepDataMiningLearning.detection3d.phase2a.vlm_voter_local \
            import LocalVLMVoter
        return LocalVLMVoter(**kwargs)
    elif backend == 'nim':
        from DeepDataMiningLearning.detection3d.phase2a.vlm_voter \
            import VLMVoter
        return VLMVoter(**kwargs)
    else:
        raise ValueError(f'unknown backend: {backend}')


def _gt_centers_for_frame(npz_path):
    """Return list of (cls, [cx,cy,cz]) for each GT object in the frame."""
    d = np.load(npz_path)
    out = []
    for row in d['boxes']:
        wt = int(row[7])
        if wt not in WAYMO_TYPE_TO_TRANSFER:
            continue
        out.append((WAYMO_TYPE_TO_TRANSFER[wt],
                    np.array([row[0], row[1], row[2]], dtype=np.float64)))
    return out


def _match_gt(cand_box, gt_list, radius=2.0):
    """Return matched GT class within 2 m, or None."""
    best = None; best_d = radius
    cx, cy = cand_box[0], cand_box[1]
    for gcls, gctr in gt_list:
        d = ((gctr[0]-cx)**2 + (gctr[1]-cy)**2) ** 0.5
        if d <= best_d:
            best = gcls; best_d = d
    return best


def _decode_frame_camera_image(tfrec_path, frame_idx, cam_slot):
    """Return RGB image array for (tfrec, frame_idx, cam_slot) or None."""
    waymo_id = SLOT_TO_WAYMO_CAM.get(cam_slot)
    if waymo_id is None:
        return None
    ds = tf.data.TFRecordDataset(tfrec_path, compression_type='')
    for i, data in enumerate(ds):
        if i != frame_idx:
            continue
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(data.numpy()))
        for img in frame.images:
            if img.name == waymo_id:
                jpg = np.frombuffer(img.image, dtype=np.uint8)
                bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates',
                    default='/tmp/p1a_ambig/ambiguous_candidates.jsonl')
    ap.add_argument('--out-json',
                    default='/tmp/p1a_ambig/p1a_results.json')
    ap.add_argument('--cache-path', default='/tmp/p1a_ambig/vlm_cache.json')
    ap.add_argument('--extract-root',
                    default='/fs/atipa/data/rnd-liu/MyRepo/'
                            'DeepDataMiningLearning/data/waymo_v1_extracted')
    ap.add_argument('--tfrec-dir',
                    default='/tmp/waymo143_train_tfrec_round2',
                    help='Directory with the .tfrecord files matching segments '
                         'referenced in the candidates jsonl.')
    ap.add_argument('--backend', choices=('local', 'nim'), default='local')
    ap.add_argument('--server', default='http://localhost:8000/v1')
    ap.add_argument('--vlm-conf', type=float, default=0.7,
                    help='Confidence threshold for "VLM kept".')
    ap.add_argument('--max', type=int, default=2000,
                    help='Cap on candidates processed.')
    args = ap.parse_args()

    os.makedirs(osp.dirname(args.out_json) or '.', exist_ok=True)

    # Build voter
    if args.backend == 'local':
        voter = _load_voter('local', server_url=args.server,
                              cache_path=args.cache_path)
    else:
        voter = _load_voter('nim', cache_path=args.cache_path)

    # Index candidates by (segment, frame_idx) to amortize the tfrecord
    # decode (loading each segment's tfrecord is expensive).
    candidates = []
    with open(args.candidates) as f:
        for line in f:
            rec = json.loads(line)
            candidates.append(rec)
            if len(candidates) >= args.max:
                break
    by_seg = defaultdict(list)
    for c in candidates:
        by_seg[c['segment']].append(c)
    print(f'[p1a] {len(candidates)} candidates across {len(by_seg)} segments')

    # Stats accumulator
    agg = defaultdict(lambda: dict(
        n=0, n_vlm_kept=0, n_gt_match=0,
        n_kept_gt_match=0, n_drop_gt_match=0))
    t0 = time.time()

    # For each segment we find its tfrecord file, decode each candidate's
    # source frame once.
    for seg, cands in by_seg.items():
        tfrec = osp.join(args.tfrec_dir, f'segment-{seg}_with_camera_labels.tfrecord')
        if not osp.isfile(tfrec):
            # try fallback patterns
            matches = [f for f in os.listdir(args.tfrec_dir)
                       if seg in f and f.endswith('.tfrecord')]
            if not matches:
                print(f'[p1a] no tfrecord for {seg[:18]}..., skipping '
                      f'{len(cands)} cands')
                continue
            tfrec = osp.join(args.tfrec_dir, matches[0])

        gt_npz_cache = {}
        # Group by frame_idx to decode each frame only once
        by_fidx = defaultdict(list)
        for c in cands:
            by_fidx[int(c['frame_idx'])].append(c)

        for fidx, frame_cands in by_fidx.items():
            # Read GT once for this frame
            gt_npz = osp.join(args.extract_root, seg, f'f_{fidx:04d}.npz')
            gt_list = _gt_centers_for_frame(gt_npz) if osp.isfile(gt_npz) else []
            # Decode each camera image used by candidates in this frame
            cam_slots_needed = {int(c['cam_slot']) for c in frame_cands}
            cam_imgs = {}
            for slot in cam_slots_needed:
                img = _decode_frame_camera_image(tfrec, fidx, slot)
                if img is not None:
                    cam_imgs[slot] = img

            for c in frame_cands:
                slot = int(c['cam_slot'])
                if slot not in cam_imgs:
                    continue
                img = cam_imgs[slot]
                box_xyxy = np.array(c['box_2d_xyxy'], dtype=np.float32)
                cls_pred, conf = voter.vote(img, box_xyxy)
                kept = (cls_pred != 'None') and (conf >= args.vlm_conf)
                gt_cls = _match_gt(np.array(c['box_3d']), gt_list)
                cand_cls = c['cls_B'] or c['cls_A']
                key = (c['source'], cand_cls)
                d = agg[key]
                d['n'] += 1
                d['n_vlm_kept'] += int(kept)
                d['n_gt_match'] += int(gt_cls is not None)
                if kept:
                    d['n_kept_gt_match'] += int(gt_cls is not None)
                else:
                    d['n_drop_gt_match'] += int(gt_cls is not None)
        # Periodic checkpoint
        voter._save_cache()

    # Final report
    voter._save_cache()
    out = dict(
        config=vars(args),
        elapsed_s=time.time() - t0,
        voter_stats=voter.stats(),
        per_bucket={f'{src}|{cls}': v for (src, cls), v in agg.items()},
    )
    # Derive precision numbers
    summary = []
    for (src, cls), d in agg.items():
        if d['n'] == 0:
            continue
        p_kept = d['n_kept_gt_match'] / max(1, d['n_vlm_kept'])
        p_drop = d['n_drop_gt_match'] / max(1, d['n'] - d['n_vlm_kept'])
        recall_orig = d['n_gt_match'] / max(1, d['n'])
        summary.append(dict(
            source=src, cls=cls, n=d['n'],
            n_vlm_kept=d['n_vlm_kept'],
            n_kept_gt_match=d['n_kept_gt_match'],
            n_drop_gt_match=d['n_drop_gt_match'],
            precision_kept=round(p_kept, 3),
            precision_drop=round(p_drop, 3),
            orig_precision=round(recall_orig, 3),
        ))
    out['summary'] = summary
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[p1a] wrote {args.out_json}')
    print(f'[p1a] elapsed {out["elapsed_s"]/60:.1f} min, '
          f'VLM stats: {out["voter_stats"]}')
    for s in summary:
        print(f'  {s["source"]:<10} {s["cls"]:<11}  '
              f'n={s["n"]:4d}  kept={s["n_vlm_kept"]:4d}  '
              f'P_kept={s["precision_kept"]:.3f}  '
              f'P_drop={s["precision_drop"]:.3f}  '
              f'orig_P={s["orig_precision"]:.3f}')


if __name__ == '__main__':
    main()
