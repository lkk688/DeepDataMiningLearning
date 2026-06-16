"""
Visual GT-sanity tool.

For N frames spanning multiple Waymo segments, render each surround
camera with 4 overlays:

  GREEN  3D-GT we currently load (projected to image) — what dataset_waymo3dv201 gives us
  YELLOW Waymo's per-camera 2D-GT (from box_2d parquet) — independent annotation
  RED    Our pseudo-labels from Phase 2a fusion (projected to image)
  CYAN   Our LiDAR-cluster proposals (Validator A) for context

If many YELLOW boxes appear without a GREEN box overlapping → our 3D-GT
parsing is dropping objects (or Waymo's 2D-GT is more permissive than
3D-GT). Either is useful to know.

Also writes a BEV-style top-down PNG per frame combining all overlays.

Usage:
  python viz_gt_check.py --max-frames 5 --frame-stride 50 \
      --out-dir /tmp/phase2a_viz
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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
    waymo_type_to_transfer, WAYMO_CLASSES)


R_OC_WC = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
T_OC_WC = np.eye(4); T_OC_WC[:3, :3] = R_OC_WC

# Slot index for each Waymo camera id (used to know which image to draw on)
WAYMO_CAM_NAMES = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT',
                   4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}


def _build_lidar2img_orig(cam_calib_for_id: Dict) -> np.ndarray:
    K = cam_calib_for_id['intrinsic_4x4']
    c2v = cam_calib_for_id['cam2vehicle']
    lidar2cam = np.linalg.inv(c2v)
    cam2img = K @ T_OC_WC
    return cam2img @ lidar2cam


def _project_3d_box_to_image(box_7d: np.ndarray, l2i_4x4: np.ndarray,
                              W: int, H: int,
                              min_depth: float = 0.5) -> Optional[List[Tuple]]:
    """Project the 8 corners of a 3D box. Returns 8 (u, v, depth) tuples
    ONLY if all 8 corners are in front of the camera (depth > min_depth).

    Partial occlusion (some corners behind the camera plane) makes the
    perspective projection geometrically degenerate — those corners produce
    garbage pixel coords. Drawing a wireframe across the resulting mixed
    set looks like a smear, NOT a box. So we just skip such boxes.

    Additionally, we require at least one corner to lie inside (or near)
    the image, otherwise the box is entirely off-screen.
    """
    x, y, z, lx, ly, lz, yaw = box_7d[:7]
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    dx, dy, dz = lx/2, ly/2, lz/2
    # 8 corners in box-local frame
    local = np.array([
        [ dx,  dy,  dz], [ dx,  dy, -dz],
        [ dx, -dy,  dz], [ dx, -dy, -dz],
        [-dx,  dy,  dz], [-dx,  dy, -dz],
        [-dx, -dy,  dz], [-dx, -dy, -dz],
    ])
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = local @ R.T + np.array([x, y, z])
    h = np.concatenate([corners, np.ones((8, 1))], axis=1)
    proj = h @ l2i_4x4.T
    depth = proj[:, 2]
    # STRICT: all 8 corners must be in front of the camera.
    if (depth <= min_depth).any():
        return None
    u = proj[:, 0] / depth
    v = proj[:, 1] / depth
    # At least one corner must lie inside the image (allow a small margin
    # so boxes partially extending off-edge are still drawn).
    margin = 100
    if (u.max() < -margin or u.min() > W + margin or
        v.max() < -margin or v.min() > H + margin):
        return None
    return list(zip(u.tolist(), v.tolist(), depth.tolist()))


def _draw_3d_box(draw: ImageDraw.ImageDraw, corners_uv: List[Tuple],
                 color: str, width: int = 2, label: str = '',
                 heading_arrow: bool = True) -> None:
    """Draw a wireframe box. Assumes all 8 corners are in front of the
    camera (caller guarantees this via _project_3d_box_to_image).

    Edges follow the order set up in _project_3d_box_to_image:
      0..3 = front face  (x = +dx in heading frame; the heading-facing side)
      4..7 = back face   (x = -dx)
    """
    if corners_uv is None or len(corners_uv) != 8:
        return
    edges = [
        (0, 1), (2, 3), (0, 2), (1, 3),     # front face
        (4, 5), (6, 7), (4, 6), (5, 7),     # back face
        (0, 4), (1, 5), (2, 6), (3, 7),     # front↔back
    ]
    pts = [(c[0], c[1]) for c in corners_uv]
    for a, b in edges:
        draw.line([pts[a], pts[b]], fill=color, width=width)
    # Heading arrow: from back-face center to front-face center
    if heading_arrow:
        fx = sum(pts[i][0] for i in (0, 1, 2, 3)) / 4
        fy = sum(pts[i][1] for i in (0, 1, 2, 3)) / 4
        bx = sum(pts[i][0] for i in (4, 5, 6, 7)) / 4
        by = sum(pts[i][1] for i in (4, 5, 6, 7)) / 4
        draw.line([(bx, by), (fx, fy)], fill=color, width=width)
    if label:
        # Place label at the projected geometric center (now all 8 corners
        # are valid so the mean is meaningful).
        cx = sum(p[0] for p in pts) / 8
        cy = sum(p[1] for p in pts) / 8
        draw.text((cx + 4, cy - 8), label, fill=color)


def _draw_2d_box(draw: ImageDraw.ImageDraw, xyxy: np.ndarray, color: str,
                 width: int = 2, label: str = '') -> None:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        draw.text((x1 + 2, y1 + 2), label, fill=color)


def _make_bev(pseudo, gt_3d, gt_types, cluster_props, save_path, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    def _rect(box, c, alpha, ls, lw):
        x, y, _, l, w, _, yaw = box[:7]
        cs, sn = np.cos(yaw), np.sin(yaw)
        dx, dy = l/2, w/2
        local = np.array([[ dx,  dy], [ dx, -dy], [-dx, -dy], [-dx,  dy]])
        R = np.array([[cs, -sn], [sn, cs]])
        world = local @ R.T + np.array([x, y])
        ax.add_patch(plt.Polygon(world, closed=True, fill=False,
                                  edgecolor=c, linewidth=lw, alpha=alpha,
                                  linestyle=ls))
    for b, t in zip(gt_3d, gt_types):
        if t is None:
            continue
        _rect(b, 'green', 0.9, '-', 2.0)
    for p in cluster_props:
        _rect(p.box, '#888888', 0.4, ':', 1.0)
    cls_color = {'Vehicle': '#ff7f00', 'Pedestrian': '#1f77b4',
                 'Cyclist': '#2ca02c'}
    for pl in pseudo:
        _rect(pl.box, cls_color.get(pl.cls, 'red'),
              0.4 + 0.6 * pl.weight, '-', 2.0)
    ax.scatter(0, 0, c='red', marker='^', s=60)
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
    ax.set_aspect('equal'); ax.set_title(title)
    ax.grid(alpha=0.3)
    legend = [
        plt.Line2D([], [], color='green', lw=2, label='3D-GT (our parse)'),
        plt.Line2D([], [], color='#888888', lw=1, ls=':',
                   label='LiDAR clusters (A)'),
        plt.Line2D([], [], color=cls_color['Vehicle'], lw=2,
                   label='Pseudo: Vehicle'),
        plt.Line2D([], [], color=cls_color['Pedestrian'], lw=2,
                   label='Pseudo: Pedestrian'),
        plt.Line2D([], [], color=cls_color['Cyclist'], lw=2,
                   label='Pseudo: Cyclist'),
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close(fig)


def viz_one_frame(ds, idx, cam_proposer, vlm, out_dir, vlm_conf=0.80):
    lidar, target = ds.base[idx]
    fname, ts, seg = ds.base.frame_index[idx]
    cam_calib = _load_camera_calibration(osp.join(ds.cam_calib_dir, fname))

    # GT
    gt_boxes = target['boxes_3d']
    gt_labels = target['labels']
    if torch.is_tensor(gt_boxes): gt_boxes = gt_boxes.cpu().numpy()
    if torch.is_tensor(gt_labels): gt_labels = gt_labels.cpu().numpy()
    gt_types = [waymo_type_to_transfer(int(x)) for x in gt_labels]
    gt_raw_types = [WAYMO_CLASSES.get(int(x), '?') for x in gt_labels]

    # LiDAR clusters (A)
    if torch.is_tensor(lidar): lidar = lidar.cpu().numpy()
    a_props = propose_clusters(np.asarray(lidar), max_range=55.0)

    # Cam2D + lift (B), all 5 cams
    b_props = []
    images_by_slot, lidar2img_by_slot = {}, {}
    surround = target['surround_views']
    cid_to_img = {int(d.get('camera_id', -1)): d['image']
                   for d in surround if 'image' in d}
    cid_to_2dgt = {}  # cam_id -> list of (xyxy, waymo_type_int)
    for d in surround:
        cid = int(d.get('camera_id', -1))
        if cid < 0:
            continue
        b2d = d.get('boxes_2d')
        l2d = d.get('labels_2d')
        if torch.is_tensor(b2d): b2d = b2d.cpu().numpy()
        if torch.is_tensor(l2d): l2d = l2d.cpu().numpy()
        if b2d is not None and len(b2d) > 0:
            cid_to_2dgt[cid] = (np.asarray(b2d), np.asarray(l2d))

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

    # Fuse
    pseudo = fuse(a_props, b_props, vlm_voter=vlm,
                  images_by_slot=images_by_slot,
                  lidar2img_by_slot=lidar2img_by_slot,
                  vlm_min_conf=vlm_conf)

    # Per-camera overlay
    frame_dir = osp.join(out_dir, f'frame_{idx:05d}')
    os.makedirs(frame_dir, exist_ok=True)
    for slot, img_np in images_by_slot.items():
        waymo_id = SLOT_TO_WAYMO_CAM[slot]
        cam_name = WAYMO_CAM_NAMES.get(waymo_id, f'CAM{waymo_id}')
        pil = Image.fromarray(img_np.copy())
        draw = ImageDraw.Draw(pil)
        H, W = img_np.shape[:2]
        l2i = lidar2img_by_slot[slot]

        # GREEN: 3D-GT (from our parse)
        for bi, (b, t_raw) in enumerate(zip(gt_boxes, gt_raw_types)):
            corners = _project_3d_box_to_image(b, l2i, W, H)
            if corners is None:
                continue
            _draw_3d_box(draw, corners, 'lime', 2, f'3DGT:{t_raw}')

        # YELLOW: Waymo 2D-GT (from box_2d parquet)
        if waymo_id in cid_to_2dgt:
            b2d, l2d = cid_to_2dgt[waymo_id]
            for xyxy, lab in zip(b2d, l2d):
                wname = WAYMO_CLASSES.get(int(lab), '?')
                _draw_2d_box(draw, xyxy, 'yellow', 2, f'2DGT:{wname}')

        # RED: Our pseudo-labels
        for pl in pseudo:
            corners = _project_3d_box_to_image(pl.box, l2i, W, H)
            if corners is None:
                continue
            us = [c[0] for c in corners]; vs = [c[1] for c in corners]
            if max(us) < 0 or min(us) > W or max(vs) < 0 or min(vs) > H:
                continue
            _draw_3d_box(draw, corners,
                         'red' if pl.cls == 'Vehicle' else
                         'magenta' if pl.cls == 'Pedestrian' else 'cyan',
                         2, f'PL:{pl.cls[:3]}')

        # Header
        hdr_text = (f'seg={seg[:24]}  frame_idx={idx}  cam={cam_name}\n'
                    f'green=3D-GT  yellow=Waymo-2D-GT  '
                    f'red=Pseudo-V  magenta=Pseudo-P  cyan=Pseudo-C')
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.rectangle([0, 0, W, 50], fill='black')
        draw.text((8, 4), hdr_text, fill='white', font=font)

        out_path = osp.join(frame_dir,
                            f'{slot}_{cam_name}_overlay.jpg')
        pil.save(out_path, quality=88)

    # BEV plot
    _make_bev(pseudo, gt_boxes, gt_types, a_props,
              save_path=osp.join(frame_dir, 'bev.png'),
              title=f'frame {idx}  A={len(a_props)} B={len(b_props)} '
                    f'PL={len(pseudo)} 3DGT={len([t for t in gt_types if t])}')

    # Per-frame summary
    summary = {
        'frame_idx': int(idx),
        'segment': seg,
        'timestamp': int(ts),
        'a_clusters': len(a_props),
        'b_lifts': len(b_props),
        'pseudo_count': len(pseudo),
        'gt_3d_total': int(len(gt_boxes)),
        'gt_3d_transfer': int(sum(1 for t in gt_types if t)),
        'gt_3d_by_waymo_class': {str(k): int((gt_labels == k).sum())
                                   for k in np.unique(gt_labels)},
        'gt_2d_by_cam': {WAYMO_CAM_NAMES.get(cid, str(cid)): int(len(b2d))
                          for cid, (b2d, _l) in cid_to_2dgt.items()},
    }
    with open(osp.join(frame_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[viz] frame {idx} → {frame_dir}')
    print(f'      3D-GT count={summary["gt_3d_total"]} (by class: {summary["gt_3d_by_waymo_class"]})')
    print(f'      2D-GT per cam: {summary["gt_2d_by_cam"]}')
    print(f'      pseudo={summary["pseudo_count"]}')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--waymo-root',
        default='/fs/atipa/data/rnd-liu/Datasets/waymo201')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--max-frames', type=int, default=5,
        help='Number of frames to visualize.')
    ap.add_argument('--frame-stride', type=int, default=200,
        help='Stride over the val set so frames span multiple segments.')
    ap.add_argument('--use-vlm', action='store_true')
    ap.add_argument('--frcnn-thresh', type=float, default=0.50)
    ap.add_argument('--vlm-conf', type=float, default=0.80)
    ap.add_argument('--out-dir', default='/tmp/phase2a_viz')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'[viz] building Waymo iterator from {args.waymo_root}/{args.split} ...')
    ds = WaymoMMDet3DZeroShot(
        root_dir=args.waymo_root, split=args.split, max_frames=None,
    )
    indices = list(range(0, len(ds), args.frame_stride))[:args.max_frames]
    print(f'[viz] frames: {indices}')

    print(f'[viz] loading Faster R-CNN ...')
    cam_proposer = Cam2DProposer(device=args.device,
                                  score_thresh=args.frcnn_thresh)
    vlm = None
    if args.use_vlm:
        cache_path = osp.join(args.out_dir, 'vlm_cache.json')
        print(f'[viz] enabling VLM (cache={cache_path}) ...')
        vlm = VLMVoter(cache_path=cache_path)

    summaries = []
    for idx in indices:
        s = viz_one_frame(ds, idx, cam_proposer, vlm, args.out_dir,
                          vlm_conf=args.vlm_conf)
        summaries.append(s)

    with open(osp.join(args.out_dir, 'all_frames_summary.json'), 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f'\n[viz] done — open {args.out_dir} to inspect.')
    if vlm is not None:
        vlm._save_cache()
        print(f'[viz] VLM stats: {vlm.stats()}')


if __name__ == '__main__':
    main()
