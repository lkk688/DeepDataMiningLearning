"""
Phase 2a end-to-end demo on N Waymo frames.

For each frame:
  1) Run Validator A (LiDAR DBSCAN)
  2) Run Validator B (Faster R-CNN on all 5 cams + LiDAR-depth 3D lift)
  3) Run fusion with optional VLM voting on disagreement
  4) Compare against the *real* Waymo GT (for sanity)
  5) Dump a BEV plot + the pseudo-labels JSON

Usage:
  python demo_single_frame.py --max-frames 5 --use-vlm \
      --out-dir /tmp/phase2a_demo
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
from DeepDataMiningLearning.detection3d.class_map_waymo_to_nus import (
    waymo_type_to_transfer)


R_OC_WC = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
T_OC_WC = np.eye(4); T_OC_WC[:3, :3] = R_OC_WC


def _build_lidar2img_orig(cam_calib_for_id: Dict) -> np.ndarray:
    """Original-resolution lidar2img (no img_aug). Used by 2D detector
    and for projecting cluster boxes to image crops for the VLM."""
    K = cam_calib_for_id['intrinsic_4x4']
    c2v = cam_calib_for_id['cam2vehicle']
    lidar2cam = np.linalg.inv(c2v)
    cam2img = K @ T_OC_WC
    return cam2img @ lidar2cam


def _bev_plot(pseudo_labels: List, gt_boxes: np.ndarray,
              gt_types: List[str], cluster_props, cam2d_props,
              save_path: str, title: str = ''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 10))
    cls_color = {'Vehicle': '#ff7f00', 'Pedestrian': '#1f77b4',
                 'Cyclist': '#2ca02c'}

    def _rect(box, c, alpha, ls):
        x, y, _, l, w, _, yaw = box[:7]
        cos, sin = np.cos(yaw), np.sin(yaw)
        # box corners (BEV)
        dx, dy = l/2, w/2
        local = np.array([[ dx,  dy], [ dx, -dy], [-dx, -dy], [-dx,  dy]])
        R = np.array([[cos, -sin], [sin, cos]])
        world = local @ R.T + np.array([x, y])
        ax.add_patch(plt.Polygon(world, closed=True, fill=False,
                                  edgecolor=c, linewidth=1.5,
                                  alpha=alpha, linestyle=ls))

    # GT in solid black
    for b, t in zip(gt_boxes, gt_types):
        if t is None:
            continue
        _rect(b, 'k', 0.8, '-')

    # Cluster proposals (thin gray dashed — show what A produced)
    for p in cluster_props:
        _rect(p.box, '#bbbbbb', 0.4, ':')

    # Cam2D proposals (thin colored dashed)
    for p in cam2d_props:
        _rect(p.box, cls_color.get(p.cls, '#777777'), 0.5, '--')

    # Pseudo-labels (final, colored, thicker)
    for pl in pseudo_labels:
        _rect(pl.box, cls_color.get(pl.cls, '#777777'),
              alpha=min(1.0, 0.5 + 0.5 * pl.weight), ls='-')

    ax.scatter(0, 0, c='red', marker='^', s=80, label='ego')
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    legend_lines = [
        plt.Line2D([], [], color='k', lw=1.5, label='GT (Waymo, transfer)'),
        plt.Line2D([], [], color='#bbbbbb', lw=1.5, ls=':',
                   label='A: LiDAR clusters'),
        plt.Line2D([], [], color='#777777', lw=1.5, ls='--',
                   label='B: Cam2D lift'),
        plt.Line2D([], [], color=cls_color['Vehicle'], lw=2,
                   label='Pseudo: Vehicle'),
        plt.Line2D([], [], color=cls_color['Pedestrian'], lw=2,
                   label='Pseudo: Pedestrian'),
        plt.Line2D([], [], color=cls_color['Cyclist'], lw=2,
                   label='Pseudo: Cyclist'),
    ]
    ax.legend(handles=legend_lines, loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def _eval_pseudo_vs_gt(pseudo_labels, gt_boxes, gt_types,
                       dist_thresh: float = 2.0):
    """Match pseudo-labels to GT; report TP/FP per class."""
    classes = ['Vehicle', 'Pedestrian', 'Cyclist']
    result = {c: {'n_gt': 0, 'tp': 0, 'fp': 0} for c in classes}
    # Filter GT to transfer classes
    gt_keep = [(b, t) for b, t in zip(gt_boxes, gt_types) if t is not None]
    for c in classes:
        result[c]['n_gt'] = sum(1 for _, t in gt_keep if t == c)
    used_gt = set()
    for pl in pseudo_labels:
        c = pl.cls
        # Find closest unused GT of same class
        best = -1; best_d = 1e9
        for gi, (b, t) in enumerate(gt_keep):
            if gi in used_gt or t != c:
                continue
            d = np.linalg.norm(b[:2] - pl.box[:2])
            if d < best_d:
                best_d = d; best = gi
        if best >= 0 and best_d <= dist_thresh:
            used_gt.add(best)
            result[c]['tp'] += 1
        else:
            result[c]['fp'] += 1
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--waymo-root',
        default='/fs/atipa/data/rnd-liu/Datasets/waymo201')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--max-frames', type=int, default=5)
    ap.add_argument('--use-vlm', action='store_true')
    ap.add_argument('--vlm-cache', default='',
        help='Defaults to <out-dir>/vlm_cache.json. Empty = no persistence.')
    ap.add_argument('--out-dir', default='/tmp/phase2a_demo')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--frcnn-thresh', type=float, default=0.50)
    ap.add_argument('--vlm-conf', type=float, default=0.80)
    ap.add_argument('--skip-a-only', action='store_true',
        help='Skip the A-only branch (saves ~half the VLM calls).')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('[phase2a] building Waymo iterator ...')
    ds = WaymoMMDet3DZeroShot(
        root_dir=args.waymo_root,
        split=args.split,
        max_frames=args.max_frames * 30,  # span ~30 frames per smoke index
    )
    # Stride to span multiple segments
    indices = list(range(0, len(ds), 30))[:args.max_frames]
    print(f'[phase2a] frames to process: {indices}')

    print(f'[phase2a] loading Faster R-CNN (score_thresh={args.frcnn_thresh}) ...')
    cam_proposer = Cam2DProposer(device=args.device,
                                  score_thresh=args.frcnn_thresh)

    vlm = None
    if args.use_vlm:
        cache_path = args.vlm_cache or osp.join(args.out_dir, 'vlm_cache.json')
        print(f'[phase2a] enabling VLM voter (cache={cache_path}) ...')
        vlm = VLMVoter(cache_path=cache_path)

    summary = []
    for f_idx in indices:
        print(f'\n========== frame {f_idx} ==========')
        t0 = time.time()
        lidar, target = ds.base[f_idx]
        fname, ts, seg = ds.base.frame_index[f_idx]
        cam_calib = _load_camera_calibration(
            osp.join(ds.cam_calib_dir, fname))

        # GT
        gt_boxes = target['boxes_3d']
        gt_labels = target['labels']
        if torch.is_tensor(gt_boxes): gt_boxes = gt_boxes.cpu().numpy()
        if torch.is_tensor(gt_labels): gt_labels = gt_labels.cpu().numpy()
        gt_types = [waymo_type_to_transfer(int(x)) for x in gt_labels]

        # ---- Validator A: LiDAR clusters
        if torch.is_tensor(lidar): lidar = lidar.cpu().numpy()
        t = time.time()
        a_props = propose_clusters(np.asarray(lidar), max_range=55.0)
        a_dt = time.time() - t

        # ---- Validator B: Cam2D + 3D lift (run all 5 cams, batch loop)
        b_props = []
        images_by_slot: Dict[int, np.ndarray] = {}
        lidar2img_by_slot: Dict[int, np.ndarray] = {}
        surround = target['surround_views']
        cid_to_img = {int(d.get('camera_id', -1)): d['image']
                       for d in surround if 'image' in d}
        t = time.time()
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
                det2d, np.asarray(lidar[:, :3]),
                lidar2img_by_slot[slot], cam_slot=slot,
                image_hw=img.shape[:2])
            b_props.extend(lifts)
        b_dt = time.time() - t

        # ---- Fusion
        t = time.time()
        pseudo = fuse(a_props, b_props,
                      vlm_voter=vlm,
                      images_by_slot=images_by_slot,
                      lidar2img_by_slot=lidar2img_by_slot,
                      vlm_min_conf=args.vlm_conf,
                      skip_a_only=args.skip_a_only,
                      debug=True)
        fuse_dt = time.time() - t

        # ---- Eval against GT
        result = _eval_pseudo_vs_gt(pseudo, gt_boxes, gt_types)
        total = time.time() - t0

        print(f'  A: {len(a_props)} clusters ({a_dt:.2f}s)')
        print(f'  B: {len(b_props)} cam2d lifts ({b_dt:.2f}s)')
        print(f'  Fuse: {len(pseudo)} pseudo-labels ({fuse_dt:.2f}s)')
        print(f'  Eval vs GT (dist≤2m):')
        for c in ['Vehicle', 'Pedestrian', 'Cyclist']:
            r = result[c]
            rec = r['tp'] / max(1, r['n_gt'])
            prec = r['tp'] / max(1, r['tp'] + r['fp'])
            print(f'    {c:12s} n_gt={r["n_gt"]:3d}  TP={r["tp"]:3d}  '
                  f'FP={r["fp"]:3d}  P={prec:.2f}  R={rec:.2f}')
        print(f'  total {total:.2f}s')

        # BEV plot
        out_png = osp.join(args.out_dir, f'frame_{f_idx:04d}_bev.png')
        _bev_plot(pseudo, gt_boxes, gt_types, a_props, b_props,
                  save_path=out_png,
                  title=f'frame {f_idx}  A={len(a_props)} B={len(b_props)} '
                        f'PL={len(pseudo)}')
        summary.append({
            'frame_idx': int(f_idx),
            'segment': seg, 'timestamp': int(ts),
            'a_count': len(a_props), 'b_count': len(b_props),
            'pseudo_count': len(pseudo),
            'eval': result, 'png': out_png,
        })

    # Save summary
    with open(osp.join(args.out_dir, 'demo_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    if vlm is not None:
        with open(osp.join(args.out_dir, 'vlm_stats.json'), 'w') as f:
            json.dump(vlm.stats(), f, indent=2)
        vlm._save_cache()
    print(f'\n[phase2a] done. Out → {args.out_dir}')
    if vlm is not None:
        print(f'[phase2a] VLM: {vlm.stats()}')


if __name__ == '__main__':
    main()
