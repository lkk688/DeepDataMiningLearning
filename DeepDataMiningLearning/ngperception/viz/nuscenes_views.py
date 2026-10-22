"""
ngperception.viz.nuscenes_views
===============================

Teaching figures from ONE nuScenes keyframe (TUTORIAL §0.4 / §7):

  nusc_bev_boxes.png      LIDAR_TOP sweep, top-down, coloured by height, with the GT boxes
  nusc_lidar_on_cam.png   the sweep projected into CAM_FRONT, coloured by depth (the BEVDepth supervision)
  nusc_range_image.png    the sweep as a 32 x 1080 range image -- in the SENSOR frame, ego-body returns removed
  nusc_surround.png       the 6 surround cameras with projected GT boxes

Two mistakes this file is written to avoid (both happened to us):
  * range images must be built in the LiDAR SENSOR frame. The ego frame's origin is the rear axle, 0.94 m behind and
    1.84 m below the LiDAR, which bends near returns into curves;
  * ~1/4 of a raw nuScenes sweep hits the ego car itself (< 2.5 m). Drop it (the devkit's remove_close does the same).

    # on the course HPC (CPU node is enough):
    export PATH=/data/cmpe249-fa26/envs/mmdet3d-cu118/bin:$PATH
    python -m DeepDataMiningLearning.ngperception.viz.nuscenes_views \
        --dataroot /data/cmpe249-fa26/datasets/nuscenes/v1.0-trainval \
        --sample 2ff5689a089844f0a2097b1883916003 --out ngperception/docs/figures
"""
from __future__ import annotations
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
from pyquaternion import Quaternion

INK, SUB = '#0F172A', '#64748B'
CLS_COLOR = {'vehicle': '#2563EB', 'human': '#EF4444', 'movable_object': '#F59E0B', 'static_object': SUB, 'animal': '#10B981'}
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)


def load_sweep(nusc, lidar_sd):
    """Raw sweep (x, y, z, intensity, ring) in the LIDAR_TOP frame + its calibrated_sensor / ego_pose records."""
    sd = nusc.get('sample_data', lidar_sd)
    raw = np.fromfile(os.path.join(nusc.dataroot, sd['filename']), dtype=np.float32).reshape(-1, 5)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token']); pose = nusc.get('ego_pose', sd['ego_pose_token'])
    return raw, cs, pose


def sensor_to_ego(pts, cs):
    return pts @ Quaternion(cs['rotation']).rotation_matrix.T + np.array(cs['translation'])


def boxes_in_ego(nusc, lidar_sd):
    sd = nusc.get('sample_data', lidar_sd); pose = nusc.get('ego_pose', sd['ego_pose_token'])
    out = []
    for b in nusc.get_boxes(lidar_sd):                       # global frame -> ego frame at the LiDAR timestamp
        b.translate(-np.array(pose['translation'])); b.rotate(Quaternion(pose['rotation']).inverse); out.append(b)
    return out


def ego_to_cam(pts_ego, pose_l, pose_c, cs_c):
    """ego(t_lidar) -> global -> ego(t_cam) -> camera. Skipping the global step smears points at speed."""
    p = pts_ego @ Quaternion(pose_l['rotation']).rotation_matrix.T + np.array(pose_l['translation'])
    p = (p - np.array(pose_c['translation'])) @ Quaternion(pose_c['rotation']).rotation_matrix
    return (p - np.array(cs_c['translation'])) @ Quaternion(cs_c['rotation']).rotation_matrix


def project(p_cam, K):
    uvw = p_cam @ np.asarray(K).T
    return uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None), p_cam[:, 2]


def fig_bev(P, boxes, out, R=50.0):
    m = (np.abs(P[:, 0]) < R) & (np.abs(P[:, 1]) < R)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(P[m, 0], P[m, 1], s=0.15, c=np.clip(P[m, 2], -2, 3), cmap='viridis')
    for b in boxes:
        c = b.corners()[:2, [0, 1, 5, 4]].T; col = CLS_COLOR.get(b.name.split('.')[0], SUB)
        ax.add_patch(Polygon(c, closed=True, fill=False, ec=col, lw=1.4))
        f = (b.corners()[:2, 0] + b.corners()[:2, 1]) / 2; ax.plot([b.center[0], f[0]], [b.center[1], f[1]], color=col, lw=1.4)
    ax.add_patch(Rectangle((-2.3, -1.0), 4.6, 2.0, color=INK)); ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect('equal')
    ax.set_xlabel('x forward (m)'); ax.set_ylabel('y left (m)')
    for k, v in CLS_COLOR.items():
        ax.plot([], [], color=v, lw=3, label=k)
    ax.legend(loc='upper right', fontsize=9); ax.set_title(f'LIDAR_TOP, ego frame: {len(P):,} points, {len(boxes)} GT boxes', color=INK)
    save(fig, out, 'nusc_bev_boxes.png')


def fig_lidar_on_cam(nusc, sample, P_ego, pose_l, out):
    sd_c = nusc.get('sample_data', sample['data']['CAM_FRONT']); cs_c = nusc.get('calibrated_sensor', sd_c['calibrated_sensor_token'])
    pose_c = nusc.get('ego_pose', sd_c['ego_pose_token'])
    uv, d = project(ego_to_cam(P_ego, pose_l, pose_c, cs_c), cs_c['camera_intrinsic'])
    img = Image.open(os.path.join(nusc.dataroot, sd_c['filename'])); W, H = img.size
    k = (d > 1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    fig, ax = plt.subplots(figsize=(12, 6.75)); ax.imshow(img)
    sc = ax.scatter(uv[k, 0], uv[k, 1], c=np.clip(d[k], 0, 60), s=2.0, cmap='turbo'); ax.axis('off')
    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01).set_label('LiDAR depth (m)')
    dt = (sd_c['timestamp'] - nusc.get('sample_data', sample['data']['LIDAR_TOP'])['timestamp']) / 1e3
    ax.set_title(f'{k.sum():,} of {len(P_ego):,} points land in CAM_FRONT (camera {dt:+.0f} ms vs LiDAR keyframe)', color=INK)
    save(fig, out, 'nusc_lidar_on_cam.png')
    return int(k.sum()), float(dt)


def fig_range_image(raw, out, ncol=1080, min_range=2.5):
    xyz, ring = raw[:, :3], raw[:, 4].astype(int)            # sensor frame; nuScenes ring 0 = lowest beam (-30.7 deg)
    r = np.linalg.norm(xyz, axis=1); keep = r > min_range
    # LIDAR_TOP axes are x right, y forward: azimuth measured from the forward axis, positive to the left
    az = np.arctan2(-xyz[:, 0], xyz[:, 1])
    col = ((np.pi - az) / (2 * np.pi) * ncol).astype(int).clip(0, ncol - 1)   # rear -> left -> front -> right -> rear
    RI = np.zeros((32, ncol)); RI[31 - ring[keep].clip(0, 31), col[keep]] = r[keep]
    fig, ax = plt.subplots(figsize=(13, 2.7))
    ax.imshow(np.where(RI > 0, np.clip(RI, 0, 60), np.nan), aspect='auto', cmap='turbo', interpolation='nearest')
    ax.set_yticks([0, 31]); ax.set_yticklabels(['top beam +10.7°', 'bottom beam −30.7°'])
    ax.set_xticks(np.linspace(0, ncol, 5)); ax.set_xticklabels(['rear', 'left', 'front', 'right', 'rear'])
    ax.set_title(f'nuScenes range image 32 × {ncol} (sensor frame; {int((~keep).sum()):,} ego-body returns < {min_range} m removed)', color=INK)
    save(fig, out, 'nusc_range_image.png')
    return int((~keep).sum())


def fig_surround(nusc, sample, out):
    fig, axs = plt.subplots(2, 3, figsize=(18, 7.2), gridspec_kw=dict(wspace=0.02, hspace=0.08))
    for ax, cam in zip(axs.ravel(), CAMS):
        sd_c = nusc.get('sample_data', sample['data'][cam]); cs_c = nusc.get('calibrated_sensor', sd_c['calibrated_sensor_token'])
        img = Image.open(os.path.join(nusc.dataroot, sd_c['filename'])); W, H = img.size; ax.imshow(img); ax.axis('off')
        _, boxes, K = nusc.get_sample_data(sample['data'][cam])        # boxes already in this camera's frame
        for b in boxes:
            c = b.corners()                                             # (3, 8) camera frame
            if (c[2] <= 0.5).any():
                continue
            uv = (np.asarray(K) @ c)[:2] / c[2]
            col = CLS_COLOR.get(b.name.split('.')[0], SUB)
            for i, j in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
                ax.plot(uv[0, [i, j]], uv[1, [i, j]], color=col, lw=1.0)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_title(cam, fontsize=11, color=INK)
    save(fig, out, 'nusc_surround.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/data/cmpe249-fa26/datasets/nuscenes/v1.0-trainval')
    ap.add_argument('--version', default='v1.0-trainval'); ap.add_argument('--sample', default='2ff5689a089844f0a2097b1883916003')
    ap.add_argument('--out', default='figures'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(a.version, a.dataroot, verbose=False); s = nusc.get('sample', a.sample)
    raw, cs_l, pose_l = load_sweep(nusc, s['data']['LIDAR_TOP']); P_ego = sensor_to_ego(raw[:, :3], cs_l)
    fig_bev(P_ego, boxes_in_ego(nusc, s['data']['LIDAR_TOP']), a.out)
    n_cam, dt = fig_lidar_on_cam(nusc, s, P_ego, pose_l, a.out)
    n_ego = fig_range_image(raw, a.out); fig_surround(nusc, s, a.out)
    meta = dict(sample=a.sample, scene=nusc.get('scene', s['scene_token'])['name'], n_points=len(raw), n_cam_front=n_cam,
                cam_minus_lidar_ms=dt, n_ego_body=n_ego)
    json.dump(meta, open(os.path.join(a.out, 'nusc_views_meta.json'), 'w'), indent=1); print(meta)


if __name__ == '__main__':
    main()
