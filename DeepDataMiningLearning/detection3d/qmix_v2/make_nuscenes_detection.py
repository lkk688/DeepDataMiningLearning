"""nuScenes multi-class detection showcase (source domain). For a frame
with diverse classes including bicycle/motorcycle (cyclist analogs), draw
the model's 3D boxes on the FRONT_LEFT|FRONT|FRONT_RIGHT camera panorama
and on an ego-aligned BEV, colored by class. Complements the Waymo
qualitative figures (which rarely contain the rare Cyclist class).
CPU-only: reuses the tracker's per-class predictions + val calibration.
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.lines import Line2D

import make_tracking_overlay as M  # box_corners_global (+dz/2), quat_to_rot

HERE = os.path.dirname(os.path.abspath(__file__))
NUS_ROOT = M.NUS_ROOT
TRACK_JSON = M.TRACK_JSON
FRONT_CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']

CLASS_COLOR = {
    'car': '#1f77b4', 'truck': '#17becf', 'bus': '#2ca02c',
    'trailer': '#9467bd', 'construction_vehicle': '#8c564b',
    'pedestrian': '#2ca02c', 'bicycle': '#d62728',
    'motorcycle': '#ff7f0e', 'traffic_cone': '#e377c2', 'barrier': '#7f7f7f',
}
# pedestrian shares green with bus? give pedestrian its own
CLASS_COLOR['pedestrian'] = '#33a02c'
CLASS_COLOR['bus'] = '#1b9e77'


def project_cam(corners_g, sample, cam_name):
    cam = sample['images'][cam_name]
    ego2global = np.array(sample['ego2global'])
    lidar2ego = np.array(sample['lidar_points']['lidar2ego'])
    lidar2cam = np.array(cam['lidar2cam'])
    cam2img = np.array(cam['cam2img'])
    hom = np.vstack([corners_g, np.ones((1, corners_g.shape[1]))])
    pc = lidar2cam @ (np.linalg.inv(lidar2ego) @
                      (np.linalg.inv(ego2global) @ hom))
    depth = pc[2]
    px = cam2img @ pc[:3]
    px = px[:2] / np.clip(px[2], 1e-3, None)
    return px, depth


def draw_boxes_on_cam(ax, sample, dets, cam_name, W=1600, H=900):
    img_path = os.path.join(NUS_ROOT, 'samples', cam_name,
                            sample['images'][cam_name]['img_path'])
    if os.path.isfile(img_path):
        ax.imshow(mpimg.imread(img_path))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    for b in dets:
        cg = M.box_corners_global(b['translation'], b['size'], b['rotation'])
        px, depth = project_cam(cg, sample, cam_name)
        if not (depth > 1.0).all():
            continue
        cu, cv = px[0].mean(), px[1].mean()
        if not (0 <= cu <= W and 0 <= cv <= H):
            continue
        col = CLASS_COLOR.get(b['tracking_name'], 'yellow')
        for a, c in M.BOX_EDGES:
            ax.plot([px[0][a], px[0][c]], [px[1][a], px[1][c]], '-',
                    color=col, lw=1.4, alpha=0.95)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(cam_name.replace('CAM_', '').replace('_', '\\_'),
                 fontsize=8)


def main():
    results = json.load(open(TRACK_JSON))['results']
    by_tok, rows = M.load_samples()

    # Score frames: cyclist(s) + class diversity, all visible in front cams.
    def frame_score(tok):
        s = by_tok[tok]
        cls = set(); ncyc = 0; n = 0
        for b in results.get(tok, []):
            if b['tracking_score'] < 0.35:
                continue
            vis = False
            for cam in FRONT_CAMS:
                cg = M.box_corners_global(b['translation'], b['size'],
                                          b['rotation'])
                px, depth = project_cam(cg, s, cam)
                if (depth > 1.0).all() and 0 <= px[0].mean() <= 1600 and \
                        0 <= px[1].mean() <= 900:
                    vis = True; break
            if not vis:
                continue
            n += 1; cls.add(b['tracking_name'])
            if b['tracking_name'] in ('bicycle', 'motorcycle'):
                ncyc += 1
        return (ncyc >= 1) * (len(cls) * 10 + n) if ncyc >= 1 else -1

    ranked = sorted((t for t, _, _ in rows), key=frame_score, reverse=True)
    # distinct scenes
    chosen, seen = [], set()
    for tok in ranked:
        if frame_score(tok) < 0:
            break
        log = os.path.basename(
            by_tok[tok]['lidar_points']['lidar_path']).split('__')[0]
        if log in seen:
            continue
        seen.add(log); chosen.append(tok)
        if len(chosen) >= 2:
            break

    for ci, tok in enumerate(chosen):
        render_frame(tok, by_tok, results, ci)


def render_frame(tok, by_tok, results, case_idx):
    sample = by_tok[tok]
    dets = [b for b in results.get(tok, []) if b['tracking_score'] >= 0.35]

    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.12,
                          wspace=0.04)
    for j, cam in enumerate(FRONT_CAMS):
        ax = fig.add_subplot(gs[0, j])
        draw_boxes_on_cam(ax, sample, dets, cam)

    # BEV, ego-aligned (forward=up, right=image-right), colored by class.
    axb = fig.add_subplot(gs[1, :])
    g2e = np.linalg.inv(np.array(sample['ego2global']))
    present = set()
    xs, ys = [], []
    for b in dets:
        pe = g2e @ np.array([b['translation'][0], b['translation'][1],
                             b['translation'][2] + b['size'][2]/2, 1.0])
        rx, fy = -pe[1], pe[0]
        if fy < -8 or abs(rx) > 55 or fy > 80:
            continue
        # box footprint in BEV: use length/width + heading
        dx, dy, _ = b['size']
        # heading in ego frame
        q = b['rotation']
        Rg = M.quat_to_rot(q)
        fwd_g = Rg @ np.array([1, 0, 0])
        fwd_e = g2e[:3, :3] @ fwd_g
        yaw = np.arctan2(-fwd_e[1], fwd_e[0])  # plot frame (right,forward)
        col = CLASS_COLOR.get(b['tracking_name'], 'yellow')
        present.add(b['tracking_name'])
        # rectangle corners
        c, s_ = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s_], [s_, c]])
        rect = (R @ np.array([[dx/2, dx/2, -dx/2, -dx/2],
                              [dy/2, -dy/2, -dy/2, dy/2]])).T + [rx, fy]
        rect = np.vstack([rect, rect[0]])
        axb.plot(rect[:, 0], rect[:, 1], '-', color=col, lw=1.3)
        # heading tick
        hd = R @ np.array([dx/2, 0])
        axb.plot([rx, rx+hd[0]], [fy, fy+hd[1]], '-', color=col, lw=1.0)
        xs.append(rx); ys.append(fy)

    from matplotlib.patches import Wedge
    rmax = max(20, (max(np.abs(xs)) if xs else 20),
               (max(ys) if ys else 20)) * 1.05
    axb.add_patch(Wedge((0, 0), rmax, 55, 125, color='gold', alpha=0.10,
                        zorder=0))
    axb.scatter([0], [0], marker='^', s=90, color='k', zorder=7)
    axb.text(0, -2.2, 'ego', fontsize=7, ha='center', va='top')
    axb.set_xlim(-rmax, rmax); axb.set_ylim(-8, rmax)
    axb.set_aspect('equal'); axb.grid(True, ls=':', alpha=0.4)
    axb.set_xlabel('right (m)'); axb.set_ylabel('forward (m)')
    axb.set_title('BEV detections (ego-aligned, colored by class)',
                  fontsize=9)

    handles = [Line2D([0], [0], color=CLASS_COLOR[c], lw=2, label=c)
               for c in sorted(present)]
    axb.legend(handles=handles, loc='upper right', fontsize=7, ncol=2,
               framealpha=0.9, title='class')

    fig.suptitle('nuScenes val multi-class 3D detection (source domain, '
                 'NDS 0.679): front camera panorama with projected boxes '
                 '(top) and ego-aligned BEV (bottom). Bicycle/motorcycle '
                 '= the cyclist-analog classes.', fontsize=9.5, y=0.99)
    suffix = '' if case_idx == 0 else f'_{case_idx+1}'
    out = os.path.join(HERE, f'nuscenes_detection{suffix}.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f'[nus-det] wrote {out} ({len(dets)} dets, classes={sorted(present)})')
    plt.close(fig)


if __name__ == '__main__':
    main()
