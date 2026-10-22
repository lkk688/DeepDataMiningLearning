"""
ngperception.viz.waymo_v2_views
===============================

Waymo Open Dataset **v2 (parquet)** teaching figures, read with plain pyarrow — no `waymo_open_dataset` package needed
(TUTORIAL §0.5 / §7):

  waymo_range_image.png    the TOP LiDAR's native range image (64 x 2650, first return)
  waymo_lidar_on_cam.png   TOP-LiDAR returns on the FRONT camera, using the dataset's own `lidar_camera_projection`
  waymo_bev_boxes.png      range image -> 3D points (our own conversion) + the `lidar_box` labels, top-down

The BEV figure is the self-check of the conversion: if the beam inclinations, azimuth convention or extrinsic were
wrong, the points would not sit inside the labelled boxes.

Waymo v2 conventions used here:
  range image [64, 2650, 4] = (range, intensity, elongation, no-label-zone); range <= 0 = no return;
  row 0 = HIGHEST beam (calibration inclinations are listed lowest-first, so reverse them);
  column c -> azimuth (W - c - 0.5) / W * 2pi - pi, then subtract the extrinsic yaw (column 0 = behind, running
  through left, front, right);
  lidar_camera_projection [64, 2650, 6] = (cam1, x1, y1, cam2, x2, y2): a return can land in two cameras;
  camera names 1 FRONT ... 5 SIDE_RIGHT; laser names 1 TOP, 2-5 short-range.
Ignored for teaching: the per-pixel `lidar_pose` motion compensation of the TOP LiDAR (a few cm at city speed).

    export PATH=/data/cmpe249-fa26/envs/mmdet3d-cu118/bin:$PATH      # pyarrow + pandas are in the course env
    python -m DeepDataMiningLearning.ngperception.viz.waymo_v2_views \
        --root /data/cmpe249-fa26/datasets/waymo/v2.0/validation \
        --segment 10203656353524179475_7625_000_7645_000 --frame 100 --out figures
"""
from __future__ import annotations
import argparse, io, json, os

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

INK = '#0F172A'
TYPE_COLOR = {1: '#2563EB', 2: '#EF4444', 3: '#F59E0B', 4: '#10B981'}      # vehicle, pedestrian, sign, cyclist
TYPE_NAME = {1: 'vehicle', 2: 'pedestrian', 3: 'sign', 4: 'cyclist'}


def read(root, comp, seg, **eq):
    t = pq.read_table(os.path.join(root, comp, seg + '.parquet')).to_pandas()
    for k, v in eq.items():
        t = t[t[k] == v]
    return t


def range_image_to_points(rng, incl_low_first, extrinsic):
    """rng (H, W) metres (<=0 = none) -> (N, 3) points in the VEHICLE frame + the (row, col) of each point."""
    H, W = rng.shape
    incl = np.asarray(incl_low_first, np.float64)[::-1]              # row 0 = highest beam
    az_corr = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])           # yaw of the sensor in the vehicle frame
    az = (W - np.arange(W) - 0.5) / W * 2 * np.pi - np.pi - az_corr
    inc, azm = np.meshgrid(incl, az, indexing='ij')                  # (H, W)
    x = rng * np.cos(inc) * np.cos(azm); y = rng * np.cos(inc) * np.sin(azm); z = rng * np.sin(inc)
    valid = rng > 0
    p = np.stack([x[valid], y[valid], z[valid], np.ones(valid.sum())], 0)
    return (extrinsic @ p)[:3].T, np.argwhere(valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data/cmpe249-fa26/datasets/waymo/v2.0/validation')
    ap.add_argument('--segment', default='10203656353524179475_7625_000_7645_000'); ap.add_argument('--frame', type=int, default=100)
    ap.add_argument('--out', default='figures'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    K = '[LiDARComponent].range_image_return1.'; KP = '[LiDARCameraProjectionComponent].range_image_return1.'
    li = read(a.root, 'lidar', a.segment, **{'key.laser_name': 1})
    ts = sorted(li['key.frame_timestamp_micros'].unique())[a.frame]
    r = li[li['key.frame_timestamp_micros'] == ts].iloc[0]
    RI = np.asarray(r[K + 'values'], np.float32).reshape(r[K + 'shape']); rng = RI[..., 0]; valid = rng > 0

    # 1. native range image
    fig, ax = plt.subplots(figsize=(13, 3.0))
    ax.imshow(np.where(valid, np.clip(rng, 0, 75), np.nan), aspect='auto', cmap='turbo', interpolation='nearest')
    ax.set_yticks([0, RI.shape[0] - 1]); ax.set_yticklabels(['top beam', 'bottom beam'])
    ax.set_xticks(np.linspace(0, RI.shape[1], 5)); ax.set_xticklabels(['rear', 'left', 'front', 'right', 'rear'])
    ax.set_title(f'Waymo TOP LiDAR native range image {RI.shape[0]} × {RI.shape[1]} ({int(valid.sum()):,} returns)', color=INK)
    fig.savefig(os.path.join(a.out, 'waymo_range_image.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)

    # 2. dataset-provided LiDAR -> FRONT camera projection
    pr = read(a.root, 'lidar_camera_projection', a.segment, **{'key.laser_name': 1, 'key.frame_timestamp_micros': ts}).iloc[0]
    PJ = np.asarray(pr[KP + 'values'], np.float32).reshape(pr[KP + 'shape'])
    im = read(a.root, 'camera_image', a.segment, **{'key.camera_name': 1, 'key.frame_timestamp_micros': ts}).iloc[0]
    img = Image.open(io.BytesIO(im['[CameraImageComponent].image'])); Wi, Hi = img.size
    uv, dep = [], []
    for c, x, y in ((0, 1, 2), (3, 4, 5)):
        m = valid & (PJ[..., c] == 1)
        uv.append(np.stack([PJ[..., x][m], PJ[..., y][m]], 1)); dep.append(rng[m])
    uv = np.concatenate(uv); dep = np.concatenate(dep)
    fig, ax = plt.subplots(figsize=(12, 8.0)); ax.imshow(img)
    sc = ax.scatter(uv[:, 0], uv[:, 1], c=np.clip(dep, 0, 75), s=1.2, cmap='turbo'); ax.set_xlim(0, Wi); ax.set_ylim(Hi, 0); ax.axis('off')
    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01).set_label('LiDAR range (m)')
    ax.set_title(f'{len(dep):,} TOP-LiDAR returns in the FRONT camera ({Wi}×{Hi}) — projection shipped with the dataset', color=INK)
    fig.savefig(os.path.join(a.out, 'waymo_lidar_on_cam.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)

    # 3. our range-image -> points conversion, checked against the 3D labels
    cal = read(a.root, 'lidar_calibration', a.segment, **{'key.laser_name': 1}).iloc[0]
    ext = np.asarray(cal['[LiDARCalibrationComponent].extrinsic.transform'], np.float64).reshape(4, 4)
    pts, _ = range_image_to_points(rng.astype(np.float64), cal['[LiDARCalibrationComponent].beam_inclination.values'], ext)
    bx = read(a.root, 'lidar_box', a.segment, **{'key.frame_timestamp_micros': ts})
    B = '[LiDARBoxComponent].box.'
    R = 60.0; m = (np.abs(pts[:, 0]) < R) & (np.abs(pts[:, 1]) < R)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pts[m, 0], pts[m, 1], s=0.1, c=np.clip(pts[m, 2], -1, 4), cmap='viridis')
    inside = 0
    for _, b in bx.iterrows():
        cx, cy, cz = b[B + 'center.x'], b[B + 'center.y'], b[B + 'center.z']; l, w, h = b[B + 'size.x'], b[B + 'size.y'], b[B + 'size.z']
        yaw = b[B + 'heading']; c, s = np.cos(yaw), np.sin(yaw)
        corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]]) @ np.array([[c, s], [-s, c]]) + [cx, cy]
        col = TYPE_COLOR.get(int(b['[LiDARBoxComponent].type']), 'gray')
        ax.add_patch(Polygon(corners, closed=True, fill=False, ec=col, lw=1.2))
        d = pts - [cx, cy, cz]; lx = d[:, 0] * c + d[:, 1] * s; ly = -d[:, 0] * s + d[:, 1] * c   # points in the box frame
        inside += int(((np.abs(lx) < l / 2) & (np.abs(ly) < w / 2) & (np.abs(d[:, 2]) < h / 2)).sum())
    for k, v in TYPE_COLOR.items():
        ax.plot([], [], color=v, lw=3, label=TYPE_NAME[k])
    ax.legend(loc='upper right', fontsize=9); ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect('equal')
    ax.set_xlabel('x forward (m)'); ax.set_ylabel('y left (m)')
    ax.set_title(f'range image → {len(pts):,} points (vehicle frame) + {len(bx)} lidar_box labels', color=INK)
    fig.savefig(os.path.join(a.out, 'waymo_bev_boxes.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
    n_label = int(bx['[LiDARBoxComponent].num_top_lidar_points_in_box'].sum())
    meta = dict(segment=a.segment, timestamp=int(ts), range_image=list(RI.shape), n_returns=int(valid.sum()), n_front=int(len(dep)),
                img=[Wi, Hi], n_boxes=int(len(bx)), points_in_boxes_ours=inside, points_in_boxes_label=n_label)
    json.dump(meta, open(os.path.join(a.out, 'waymo_views_meta.json'), 'w'), indent=1); print(meta)


if __name__ == '__main__':
    main()
