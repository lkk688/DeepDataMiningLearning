"""
Extract Waymo v1.4.3 tfrecords to a per-frame folder layout that our
v2 pipeline already understands.

Output layout (one folder per segment):
    {out_dir}/{segment}/
        f_{idx}.npz         # lidar (N,5), 3D boxes, calib mats
        f_{idx}_cam_{slot}.jpg   # raw RGB JPEGs (one per Waymo camera)

The info.pkl builder (build_waymo_finetune_infos_v1.py — see sibling)
then walks these folders and emits the same nuScenes-style info dicts
our WaymoFineTuneDataset reads.

LiDAR convention: right-handed Waymo vehicle frame (+x forward, +y left,
+z up). NO y-flip applied here — the info.pkl builder applies the y-flip
to GT boxes (cy → -cy, yaw → -yaw) to match the model's training-time
frame, same convention used for v2.
"""

from __future__ import annotations

import argparse
import io
import os
import os.path as osp
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.utils import frame_utils

# Patch SDK incompatibility: frame_utils.parse_range_image_and_camera_projection
# calls ri.ParseFromString(bytearray(...)) but modern protobuf rejects
# bytearray, only accepts bytes. Wrap with a tolerant version.
import zlib
def _patched_parse(frame):
    range_images = {}
    camera_projections = {}
    range_image_top_pose = None
    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_compressed) > 0:
            from waymo_open_dataset import dataset_pb2 as _pb
            ri_str = zlib.decompress(laser.ri_return1.range_image_compressed)
            ri = _pb.MatrixFloat()
            ri.ParseFromString(bytes(ri_str))   # bytes, not bytearray
            range_images.setdefault(laser.name, []).append(ri)
            if laser.name == 1 and len(laser.ri_return1.range_image_pose_compressed) > 0:
                rip_str = zlib.decompress(laser.ri_return1.range_image_pose_compressed)
                range_image_top_pose = _pb.MatrixFloat()
                range_image_top_pose.ParseFromString(bytes(rip_str))
            if len(laser.ri_return1.camera_projection_compressed) > 0:
                cp_str = zlib.decompress(laser.ri_return1.camera_projection_compressed)
                cp = _pb.MatrixInt32()
                cp.ParseFromString(bytes(cp_str))
                camera_projections.setdefault(laser.name, []).append(cp)
        if len(laser.ri_return2.range_image_compressed) > 0:
            from waymo_open_dataset import dataset_pb2 as _pb
            ri_str = zlib.decompress(laser.ri_return2.range_image_compressed)
            ri = _pb.MatrixFloat()
            ri.ParseFromString(bytes(ri_str))
            range_images.setdefault(laser.name, []).append(ri)
            if len(laser.ri_return2.camera_projection_compressed) > 0:
                cp_str = zlib.decompress(laser.ri_return2.camera_projection_compressed)
                cp = _pb.MatrixInt32()
                cp.ParseFromString(bytes(cp_str))
                camera_projections.setdefault(laser.name, []).append(cp)
    return range_images, camera_projections, None, range_image_top_pose

frame_utils.parse_range_image_and_camera_projection = _patched_parse


WAYMO_CAM_NAMES = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT',
                   4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}


def _intensity_from_range_image(range_image_tensor, beam_inclinations,
                                 extrinsic):
    """Decode (N, 5) [x, y, z, intensity, time] using Waymo SDK primitives."""
    # Frame is parsed externally. Here we just need to read intensity from
    # the second channel of the range image.
    # Simpler path: convert_range_image_to_point_cloud already returns
    # geometry; we use keep_polar_features to also get intensity.
    pass   # Handled inline below.


def _frame_to_points(frame):
    """Get all-laser (N, 5) points in vehicle frame: x,y,z,intensity,timestamp_offset.

    The intensity channel comes from range_image.values[..., 1].
    """
    (range_images, camera_projections, _, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)
    # Geometry: returns list of (N_i, 3) per laser
    points_xyz, cp_pts = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose,
    )
    # All TOP=1, FRONT=2, etc. up to 5 lasers. Concatenate.
    xyz_all = np.concatenate(points_xyz, axis=0).astype(np.float32)

    # Intensity: extract from the range image directly.
    # range_images[laser_id] = [r1, r2]; each is a RangeImage proto with
    # .values (HxWxC), .shape (HxWxC). Channel 1 = intensity.
    intens_all = []
    for laser_id in sorted(range_images.keys()):
        ri_list = range_images[laser_id]
        for ri in ri_list:
            vals = np.array(ri.data, dtype=np.float32).reshape(ri.shape.dims)
            rng = vals[..., 0]
            ints = vals[..., 1] if vals.shape[-1] > 1 else np.zeros_like(rng)
            mask = rng > 0
            intens_all.append(ints[mask].astype(np.float32))
    intens = np.concatenate(intens_all, axis=0)

    # Pad/truncate to match xyz length (in case of off-by-one):
    if len(intens) != len(xyz_all):
        if len(intens) > len(xyz_all):
            intens = intens[:len(xyz_all)]
        else:
            intens = np.pad(intens, (0, len(xyz_all) - len(intens)),
                            constant_values=0.0)

    # 5th column: timestamp offset (0.0 since all current-frame)
    ts_col = np.zeros(len(xyz_all), dtype=np.float32)
    return np.column_stack([xyz_all, intens, ts_col]).astype(np.float32)


def _frame_to_gt_boxes(frame):
    """Return (N, 8) array [cx, cy, cz, lx, ly, lz, heading, waymo_type].
    Vehicle frame, right-handed. NO y-flip."""
    rows = []
    for lbl in frame.laser_labels:
        b = lbl.box
        rows.append([b.center_x, b.center_y, b.center_z,
                     b.length, b.width, b.height,
                     b.heading, int(lbl.type)])
    if not rows:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _frame_to_camera_calib(frame):
    """Return dict cam_id → {intrinsic_4x4, extrinsic_4x4_vehicle_from_camera,
    width, height}."""
    out = {}
    for cc in frame.context.camera_calibrations:
        cam_id = cc.name
        intr = np.array(cc.intrinsic, dtype=np.float64)  # [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
        K = np.eye(4, dtype=np.float64)
        K[0, 0] = intr[0]; K[1, 1] = intr[1]
        K[0, 2] = intr[2]; K[1, 2] = intr[3]
        extr = np.array(cc.extrinsic.transform, dtype=np.float64).reshape(4, 4)
        out[cam_id] = dict(
            intrinsic_4x4=K,
            cam2vehicle=extr,
            width=int(cc.width),
            height=int(cc.height),
        )
    return out


def _frame_to_lidar_calib(frame):
    """Return TOP LiDAR extrinsic (4x4) and beam_inclinations."""
    for lc in frame.context.laser_calibrations:
        if lc.name == 1:   # TOP
            T = np.array(lc.extrinsic.transform, dtype=np.float64).reshape(4, 4)
            incl = np.array(lc.beam_inclinations, dtype=np.float64)
            return T, incl
    return np.eye(4), np.array([])


def _frame_to_pose(frame):
    """Return 4x4 vehicle pose (world from vehicle)."""
    return np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)


def process_tfrecord(tfrecord_path: str, out_dir: str,
                     max_frames: int = -1) -> int:
    """Extract one tfrecord. Returns number of frames written."""
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    seg_name = None
    n_written = 0
    seg_dir = None
    seg_meta = []   # list of per-frame meta dicts (for the info.pkl builder)

    t0 = time.time()
    for i, data in enumerate(dataset):
        if max_frames > 0 and i >= max_frames:
            break
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(data.numpy()))

        if seg_name is None:
            seg_name = frame.context.name
            seg_dir = osp.join(out_dir, seg_name)
            os.makedirs(seg_dir, exist_ok=True)

        ts = int(frame.timestamp_micros)

        # Points: (N, 5)
        try:
            pts = _frame_to_points(frame)
        except Exception as e:
            print(f'    [{i}] points decode failed: {type(e).__name__}: {e}')
            continue

        # Boxes (vehicle frame, RH)
        boxes = _frame_to_gt_boxes(frame)

        # Calibrations
        cam_calib = _frame_to_camera_calib(frame)
        lidar_T, lidar_incl = _frame_to_lidar_calib(frame)
        pose = _frame_to_pose(frame)

        # Save lidar + boxes + calib as compressed npz
        npz_path = osp.join(seg_dir, f'f_{i:04d}.npz')
        np.savez_compressed(
            npz_path,
            lidar=pts.astype(np.float32),
            boxes=boxes,
            lidar_T_vehicle_from_lidar=lidar_T,
            lidar_beam_incl=lidar_incl,
            pose_world_from_vehicle=pose,
            ts=np.int64(ts),
            cam_ids=np.array(sorted(cam_calib.keys()), dtype=np.int32),
        )

        # Camera intrinsics + extrinsics (saved alongside)
        for cid, cal in cam_calib.items():
            np.savez_compressed(
                osp.join(seg_dir, f'f_{i:04d}_cam_{cid}_calib.npz'),
                intrinsic=cal['intrinsic_4x4'].astype(np.float32),
                cam2vehicle=cal['cam2vehicle'].astype(np.float32),
                width=np.int32(cal['width']),
                height=np.int32(cal['height']),
            )

        # Camera JPEGs (one file per camera, raw bytes from tfrecord)
        for img in frame.images:
            cid = img.name
            jpg_path = osp.join(seg_dir, f'f_{i:04d}_cam_{cid}.jpg')
            with open(jpg_path, 'wb') as f:
                f.write(img.image)

        seg_meta.append(dict(
            frame_idx=i,
            ts=ts,
            n_points=int(pts.shape[0]),
            n_boxes=int(boxes.shape[0]),
            cam_ids=sorted(cam_calib.keys()),
        ))
        n_written += 1

    # Save segment metadata
    if seg_dir is not None:
        import json
        with open(osp.join(seg_dir, 'meta.json'), 'w') as f:
            json.dump({'segment': seg_name, 'frames': seg_meta}, f, indent=2)
        print(f'  → {seg_name}: {n_written} frames in {time.time()-t0:.1f}s')
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tfrec-dir', default='/tmp/waymo143_extract',
        help='Directory containing extracted *.tfrecord files.')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-tfrecords', type=int, default=-1)
    ap.add_argument('--max-frames-per-tfrec', type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tfrecs = sorted(f for f in os.listdir(args.tfrec_dir)
                    if f.endswith('.tfrecord'))
    if args.max_tfrecords > 0:
        tfrecs = tfrecs[:args.max_tfrecords]
    print(f'[v1ext] {len(tfrecs)} tfrecords to extract → {args.out_dir}')

    total = 0
    t_all = time.time()
    for i, fn in enumerate(tfrecs):
        path = osp.join(args.tfrec_dir, fn)
        print(f'[v1ext] [{i+1}/{len(tfrecs)}] {fn}  ({osp.getsize(path)//(1024*1024)} MB)')
        n = process_tfrecord(path, args.out_dir, args.max_frames_per_tfrec)
        total += n
        elapsed = time.time() - t_all
        eta = elapsed / (i + 1) * (len(tfrecs) - i - 1)
        print(f'[v1ext]   running total = {total} frames  '
              f'elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min')

    print(f'\n[v1ext] DONE — {total} frames extracted from '
          f'{len(tfrecs)} segments → {args.out_dir}')


if __name__ == '__main__':
    main()
