"""P1a Step 1 — generate ambiguous A/B candidates (the A∪B − A∩B set).

Re-runs validators A (LiDAR DBSCAN) and B (Faster R-CNN + LiDAR depth
lift) on a small frame subset and emits per-candidate records of the
form:

    {
      'frame_idx': int, 'segment': str, 'timestamp': int,
      'source': 'A-only' | 'B-only' | 'conflict',
      'cls_A': str|None, 'cls_B': str|None,
      'box_3d': [cx,cy,cz,dx,dy,dz,yaw],
      'cam_slot': int,         # which camera the box was visible in
      'box_2d_xyxy': [x1,y1,x2,y2],  # for the red-box crop
      'cam_image_path': str,   # absolute jpg path
    }

Output: ``{out_dir}/ambiguous_candidates.jsonl`` plus a metadata
config.json. The runner in ``run_p1a_ambiguity.py`` consumes this and
calls the (local or NIM) VLMVoter on every candidate.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning')

# Import extract_waymo_v1 to install the protobuf patch + helpers.
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
from DeepDataMiningLearning.detection3d.phase2a.fusion import _match_3d_boxes

R_OC_WC = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
T_OC_WC = np.eye(4); T_OC_WC[:3, :3] = R_OC_WC

NUM_CAMS = 6
SLOT_TO_WAYMO_CAM = {0: 1, 1: 3, 2: 2, 3: None, 4: 4, 5: 5}


def _build_lidar2img(K, cam2vehicle):
    return K @ T_OC_WC @ np.linalg.inv(cam2vehicle)


def _project_box_2d(box_3d_v, lidar2img_to_pixel, image_hw):
    """Project an oriented 3D box into 2D pixel xyxy. Returns None if
    box is mostly behind/off-camera."""
    cx, cy, cz, dx, dy, dz, yaw = box_3d_v[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    lc = np.array([
        [+dx/2, +dy/2, -dz/2], [+dx/2, -dy/2, -dz/2],
        [-dx/2, -dy/2, -dz/2], [-dx/2, +dy/2, -dz/2],
        [+dx/2, +dy/2, +dz/2], [+dx/2, -dy/2, +dz/2],
        [-dx/2, -dy/2, +dz/2], [-dx/2, +dy/2, +dz/2],
    ])
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = lc @ R.T + np.array([cx, cy, cz])
    corners_h = np.hstack([corners, np.ones((8, 1))])
    proj = lidar2img_to_pixel @ corners_h.T
    z = proj[2, :]
    if (z > 0.5).sum() < 3:
        return None
    uv = proj[:2, :] / np.where(np.abs(z) > 1e-6, z, 1.0)
    u_min, u_max = float(uv[0].min()), float(uv[0].max())
    v_min, v_max = float(uv[1].min()), float(uv[1].max())
    H, W = image_hw
    u_min = max(0.0, u_min); v_min = max(0.0, v_min)
    u_max = min(W - 1.0, u_max); v_max = min(H - 1.0, v_max)
    if u_max <= u_min + 4 or v_max <= v_min + 4:
        return None
    return [u_min, v_min, u_max, v_max]


def process_tfrecord(tfrec_path, args, out_jf, n_so_far):
    ds = tf.data.TFRecordDataset(tfrec_path, compression_type='')
    cam_proposer = args._cam_proposer
    seg = None
    n_added = 0
    for i, data in enumerate(ds):
        if i % args.frame_stride != 0:
            continue
        if args.max_frames > 0 and n_so_far + n_added >= args.max_frames:
            break
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(data.numpy()))
        if seg is None:
            seg = frame.context.name
        try:
            lidar = _frame_to_points(frame).astype(np.float32)
            cam_calib = _frame_to_camera_calib(frame)
        except Exception as e:
            continue

        a_props = propose_clusters(lidar, max_range=55.0)

        b_props = []
        slot_lidar2img = {}; slot_imgshape = {}
        cid_to_jpg = {img.name: img.image for img in frame.images}
        for slot in range(NUM_CAMS):
            waymo_id = SLOT_TO_WAYMO_CAM[slot]
            if waymo_id is None or waymo_id not in cid_to_jpg:
                continue
            jpg = np.frombuffer(cid_to_jpg[waymo_id], dtype=np.uint8)
            bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            K = cam_calib[waymo_id]['intrinsic_4x4']
            c2v = cam_calib[waymo_id]['cam2vehicle']
            l2i = _build_lidar2img(K, c2v)
            slot_lidar2img[slot] = l2i
            slot_imgshape[slot] = img_rgb.shape[:2]
            det2d = cam_proposer.detect_2d(img_rgb)
            lifts = cam_proposer.lift_to_3d(
                det2d, np.asarray(lidar[:, :3]), l2i, cam_slot=slot,
                image_hw=img_rgb.shape[:2])
            b_props.extend(lifts)

        # Match A↔B; emit ambiguous (A-only, B-only, conflict) candidates.
        matches, a_unmatched, b_unmatched = _match_3d_boxes(
            a_props, b_props)

        # Class-conflict (same A,B paired but different cls)
        for a_idx, b_idx in matches:
            ap, bp = a_props[a_idx], b_props[b_idx]
            if ap.cls != bp.cls:
                # use A's geometry (LiDAR-grounded) but record both
                _emit_candidate(ap, 'conflict', ap.cls, bp.cls,
                                 slot_lidar2img, slot_imgshape, seg,
                                 frame.timestamp_micros, i,
                                 args.out_dir, out_jf)
                n_added += 1
        for a_idx in a_unmatched:
            _emit_candidate(a_props[a_idx], 'A-only', a_props[a_idx].cls,
                             None, slot_lidar2img, slot_imgshape, seg,
                             frame.timestamp_micros, i,
                             args.out_dir, out_jf)
            n_added += 1
        for b_idx in b_unmatched:
            _emit_candidate(b_props[b_idx], 'B-only', None,
                             b_props[b_idx].cls,
                             slot_lidar2img, slot_imgshape, seg,
                             frame.timestamp_micros, i,
                             args.out_dir, out_jf)
            n_added += 1
        if (n_added + n_so_far) >= args.max_candidates:
            break
    return n_added


def _emit_candidate(prop, source, cls_a, cls_b, slot_lidar2img,
                     slot_imgshape, seg, ts, fidx, out_dir, out_jf):
    # Pick the best camera slot to project into
    chosen_slot = None; best_area = 0.0; chosen_xyxy = None
    cls_label = (cls_b if cls_b else cls_a)
    if cls_label not in ('Pedestrian', 'Cyclist', 'Vehicle'):
        return
    for slot, l2i in slot_lidar2img.items():
        xyxy = _project_box_2d(prop.box, l2i, slot_imgshape[slot])
        if xyxy is None:
            continue
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        if area > best_area:
            best_area = area; chosen_slot = slot; chosen_xyxy = xyxy
    if chosen_slot is None:
        return
    # Save a thumbnail of the camera image (lazy: just store seg + fidx
    # so the runner can re-decode the jpg by frame index).
    rec = {
        'frame_idx': int(fidx),
        'segment': str(seg),
        'timestamp': int(ts),
        'source': source,
        'cls_A': cls_a,
        'cls_B': cls_b,
        'box_3d': [float(v) for v in prop.box[:7]],
        'cam_slot': int(chosen_slot),
        'box_2d_xyxy': [float(v) for v in chosen_xyxy],
    }
    out_jf.write(json.dumps(rec) + '\n')
    out_jf.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tfrec-dir', required=True,
                    help='Directory with extracted *.tfrecord files.')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-frames', type=int, default=200,
                    help='Frames to scan total.')
    ap.add_argument('--frame-stride', type=int, default=5)
    ap.add_argument('--max-candidates', type=int, default=2000,
                    help='Stop early after this many ambiguous candidates.')
    ap.add_argument('--frcnn-thresh', type=float, default=0.50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    args._cam_proposer = Cam2DProposer(device='cuda',
                                         score_thresh=args.frcnn_thresh)
    tfrecs = sorted(f for f in os.listdir(args.tfrec_dir)
                     if f.endswith('.tfrecord'))
    out_path = osp.join(args.out_dir, 'ambiguous_candidates.jsonl')
    n_total = 0
    with open(out_path, 'w') as jf:
        for fn in tfrecs:
            if n_total >= args.max_candidates:
                break
            n_added = process_tfrecord(osp.join(args.tfrec_dir, fn), args,
                                         jf, n_total)
            n_total += n_added
            print(f'[ambig] {fn}: +{n_added} candidates  '
                  f'(running total {n_total})')
            if n_total >= args.max_candidates:
                break
    print(f'\n[ambig] DONE — {n_total} ambiguous candidates → {out_path}')


if __name__ == '__main__':
    main()
