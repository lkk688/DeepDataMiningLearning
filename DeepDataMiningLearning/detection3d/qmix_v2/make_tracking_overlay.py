"""Tracking visualization with camera context: a 3-panel figure showing
[start-frame CAM_FRONT + projected 3D track boxes] | [BEV trajectories] |
[end-frame CAM_FRONT + projected 3D track boxes]. Track colors are
consistent across all three panels so a box in the start/end photo can
be matched to its BEV trajectory. CPU-only: uses the AMOTA-0.604 tracker
output + nuScenes val calibration; no GPU inference.
"""
from __future__ import annotations
import json
import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

plt.rcParams.update({
    'font.size': 8, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK_JSON = '/tmp/track_b10c_velocity/tracking_results.json'
NUS_ROOT = '/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes'
VAL_PKL = os.path.join(NUS_ROOT, 'nuscenes_infos_val.pkl')

CLASS_MARKER = {
    'car': 'o', 'truck': 's', 'bus': 'P', 'trailer': 'X',
    'pedestrian': '^', 'bicycle': 'D', 'motorcycle': 'v',
}
# 3D box edges (corner index pairs) for nuScenes devkit corner order.
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),      # top face
             (4, 5), (5, 6), (6, 7), (7, 4),      # bottom face
             (0, 4), (1, 5), (2, 6), (3, 7)]      # verticals


def quat_to_rot(q):
    w, x, y, z = q
    n = (w*w + x*x + y*y + z*z) ** 0.5
    if n < 1e-8:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


# The tracker reports box-center z at the box BOTTOM (a convention the
# nuScenes 2D-distance AMOTA metric is blind to), so global z is ~dz/2 too
# low. Lift by dz/2 to recover the true 3D center for image projection.
def box_corners_global(translation, size, rotation):
    # size = [dx, dy, dz] = [length(along heading +x), width(y), height(z)],
    # matching the mmdet3d LiDAR box order the tracker passes through.
    dx, dy, dz = size
    xc = dx/2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    yc = dy/2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    zc = dz/2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack([xc, yc, zc])                  # 3x8
    corners = quat_to_rot(rotation) @ corners
    center = np.array([translation[0], translation[1],
                       translation[2] + dz/2])
    corners += center.reshape(3, 1)
    return corners                                     # 3x8 global


def project_global_to_image(corners_g, sample):
    """global 3x8 -> pixel 2x8 (+ visible mask) for CAM_FRONT."""
    cam = sample['images']['CAM_FRONT']
    ego2global = np.array(sample['ego2global'])
    lidar2ego = np.array(sample['lidar_points']['lidar2ego'])
    lidar2cam = np.array(cam['lidar2cam'])
    cam2img = np.array(cam['cam2img'])

    hom = np.vstack([corners_g, np.ones((1, 8))])      # 4x8
    pts_ego = np.linalg.inv(ego2global) @ hom
    pts_lidar = np.linalg.inv(lidar2ego) @ pts_ego
    pts_cam = lidar2cam @ pts_lidar                    # 4x8
    depth = pts_cam[2]
    in_front = depth > 0.2
    px = cam2img @ pts_cam[:3]
    px = px[:2] / np.clip(px[2], 1e-3, None)
    return px, in_front, depth


def load_samples():
    with open(VAL_PKL, 'rb') as f:
        dl = pickle.load(f)['data_list']
    by_token = {s['token']: s for s in dl}
    rows = []
    for s in dl:
        lp = s['lidar_points']['lidar_path']
        log = os.path.basename(lp).split('__')[0]
        rows.append((s['token'], float(s['timestamp']), log))
    return by_token, rows


def segment_windows(rows, gap_s=2.0, max_frames=40):
    scenes, cur, last_ts, last_log = [], [], None, None
    for tok, ts, log in rows:
        if cur and (log != last_log or (ts - last_ts) > gap_s):
            scenes.append(cur); cur = []
        cur.append((tok, ts)); last_ts, last_log = ts, log
    if cur:
        scenes.append(cur)
    windows = []
    for sc in scenes:
        for st in range(0, len(sc), max_frames):
            windows.append(sc[st:st + max_frames])
    return windows


def project_point_global(xyz, sample):
    """Single global point -> (u, v, depth) in CAM_FRONT."""
    cam = sample['images']['CAM_FRONT']
    ego2global = np.array(sample['ego2global'])
    lidar2ego = np.array(sample['lidar_points']['lidar2ego'])
    lidar2cam = np.array(cam['lidar2cam'])
    cam2img = np.array(cam['cam2img'])
    hom = np.array([xyz[0], xyz[1], xyz[2], 1.0])
    p = lidar2cam @ (np.linalg.inv(lidar2ego) @
                     (np.linalg.inv(ego2global) @ hom))
    depth = p[2]
    px = cam2img @ p[:3]
    if abs(px[2]) < 1e-3:
        return None
    return px[0]/px[2], px[1]/px[2], depth


def box_visible(b, sample, score_min=0.4):
    """Return (corners_px 2x8, center_uv) if ALL corners are well in front
    and the box center projects inside the image, else None."""
    if b['tracking_score'] < score_min:
        return None
    cg = box_corners_global(b['translation'], b['size'], b['rotation'])
    px, infront, depth = project_global_to_image(cg, sample)
    if not infront.all() or depth.min() < 1.0:
        return None
    c = project_point_global(b['translation'], sample)
    if c is None or c[2] < 1.0:
        return None
    cu, cv = c[0], c[1]
    if not (0 <= cu <= 1600 and 0 <= cv <= 900):
        return None
    return px, (cu, cv)


def main():
    with open(TRACK_JSON) as f:
        results = json.load(f)['results']
    by_token, rows = load_samples()
    # Shorter windows (~8 s) so start/end frames are comparable and motion
    # is legible rather than a whole-scene change.
    windows = segment_windows(rows, max_frames=16)

    def n_clean_visible(tok):
        sample = by_token.get(tok)
        if sample is None:
            return 0
        return sum(1 for b in results.get(tok, [])
                   if box_visible(b, sample) is not None)

    def n_clean_vehicles(tok):
        sample = by_token.get(tok)
        if sample is None:
            return 0
        return sum(1 for b in results.get(tok, [])
                   if b['tracking_name'] in ('car', 'truck', 'bus')
                   and box_visible(b, sample) is not None)

    # Prefer well-populated scenes (~5 cleanly visible boxes at both ends)
    # with several detected vehicles, so the figure looks complete rather
    # than sparse. Targeting ~5 reduces the chance of an obvious empty car.
    def window_score(w):
        if len(w) < 10:
            return -1e9
        ns, ne = n_clean_visible(w[0][0]), n_clean_visible(w[-1][0])
        if ns < 2 or ne < 2:
            return -1e9
        veh = min(n_clean_vehicles(w[0][0]), n_clean_vehicles(w[-1][0]))
        richness = -(abs(ns - 5) + abs(ne - 5))   # sweet spot ~5 boxes
        return richness * 3 + min(ns, ne) + veh * 2
    # Rank windows, then take the best from DISTINCT scenes (avoid repeats).
    ranked = sorted(windows, key=window_score, reverse=True)

    def scene_key(w):
        s = by_token.get(w[0][0])
        return os.path.basename(
            s['lidar_points']['lidar_path']).split('__')[0] if s else id(w)

    chosen, seen = [], set()
    for w in ranked:
        if window_score(w) < -1e8:
            continue
        k = scene_key(w)
        if k in seen:
            continue
        seen.add(k)
        chosen.append(w)
        if len(chosen) >= 3:
            break

    for ci, window in enumerate(chosen):
        render_window(window, by_token, results, ci)


def render_window(window, by_token, results, case_idx):
    # Collect per-track trajectory (global x,y,z,height,name) + colors.
    traj = defaultdict(list)
    for tok, ts in window:
        for b in results.get(tok, []):
            if b['tracking_score'] < 0.4:
                continue
            traj[b['tracking_id']].append(
                (b['translation'][0], b['translation'][1],
                 b['translation'][2], b['size'][2], b['tracking_name']))
    traj = {k: v for k, v in traj.items() if len(v) >= 3}
    if not traj:
        return
    ids_sorted = sorted(traj.keys())
    cmap = plt.cm.get_cmap('turbo', max(len(ids_sorted), 1))
    color = {tid: cmap(i) for i, tid in enumerate(ids_sorted)}

    fig = plt.figure(figsize=(15.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.25], wspace=0.08)
    ax_s, ax_t, ax_e = (fig.add_subplot(gs[i]) for i in range(3))

    def draw_camera(ax, tok, title):
        sample = by_token[tok]
        img_path = os.path.join(NUS_ROOT, 'samples', 'CAM_FRONT',
                                sample['images']['CAM_FRONT']['img_path'])
        if os.path.isfile(img_path):
            ax.imshow(mpimg.imread(img_path))
        ax.set_xlim(0, 1600); ax.set_ylim(900, 0)

        # (1) Project each track's FULL trajectory into THIS frame's camera
        #     (lifted to object center height) so the path is on the photo.
        #     Strict bounds (depth>3 m, strictly inside the image) avoid
        #     wild projections of far/edge points; the polyline is broken at
        #     out-of-frame gaps so no segment is drawn across the sky.
        for tid, pts in traj.items():
            seg, segments = [], []
            for (gx, gy, gz, hh, nm) in pts:
                pr = project_point_global((gx, gy, gz + hh/2), sample)
                if pr is not None and pr[2] > 3.0 and \
                        0 <= pr[0] <= 1600 and 0 <= pr[1] <= 900:
                    seg.append((pr[0], pr[1]))
                else:
                    if len(seg) >= 2:
                        segments.append(seg)
                    seg = []
            if len(seg) >= 2:
                segments.append(seg)
            for seg in segments:
                a = np.array(seg)
                ax.plot(a[:, 0], a[:, 1], '-', color=color[tid], lw=1.6,
                        alpha=0.7, zorder=5)
                ax.scatter(a[:, 0], a[:, 1], s=5, color=color[tid],
                           alpha=0.6, zorder=5)

        # (2) Draw the current-frame 3D boxes (only fully-visible ones).
        for b in results.get(tok, []):
            if b['tracking_id'] not in color:
                continue
            vis = box_visible(b, sample)
            if vis is None:
                continue
            px, (cu, cv) = vis
            col = color[b['tracking_id']]
            for a, c in BOX_EDGES:
                ax.plot([px[0][a], px[0][c]], [px[1][a], px[1][c]],
                        '-', color=col, lw=1.6, alpha=0.95, zorder=6)
            ax.text(cu, max(px[1].min()-6, 8), b['tracking_id'],
                    color='white', fontsize=6, ha='center', va='bottom',
                    bbox=dict(fc=col, ec='none', alpha=0.85, pad=0.6),
                    zorder=7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    draw_camera(ax_s, window[0][0],
                '(a) Start frame: boxes + forward trajectory')
    draw_camera(ax_e, window[-1][0],
                '(c) End frame: boxes + trajectory so far')

    # Middle: BEV trajectories in the START-frame EGO frame so the panel's
    # orientation matches the camera view: up = forward (into the photo),
    # right = image-right. (ego x=forward, y=left -> plot x=-y, plot y=x.)
    g2e = np.linalg.inv(np.array(by_token[window[0][0]]['ego2global']))

    def to_bev(gx, gy, gz):
        pe = g2e @ np.array([gx, gy, gz, 1.0])
        return np.array([-pe[1], pe[0]])     # [right, forward]

    xs, ys = [], []
    for tid, pts in traj.items():
        arr = np.array([to_bev(p[0], p[1], p[2] + p[3]/2) for p in pts])
        nm = pts[-1][4]
        col = color[tid]
        disp = float(np.linalg.norm(arr[-1] - arr[0]))
        mk = CLASS_MARKER.get(nm, 'o')
        # Keep only tracks within the CAM_FRONT field of view, so every BEV
        # trajectory corresponds to a box visible in the start/end photos.
        ang = np.abs(np.arctan2(arr[:, 0], np.maximum(arr[:, 1], 0.01)))
        if not np.any((arr[:, 1] > 0.5) & (ang < np.radians(40))):
            continue
        xs.extend(arr[:, 0]); ys.extend(arr[:, 1])
        if disp >= 2.0:
            ax_t.plot(arr[:, 0], arr[:, 1], '-', color=col, lw=1.8,
                      alpha=0.9, zorder=3)
            ax_t.scatter(arr[:, 0], arr[:, 1], marker=mk, s=14, color=col,
                         edgecolors='k', linewidths=0.3, zorder=4)
            ax_t.scatter(arr[0, 0], arr[0, 1], marker='*', s=55, color=col,
                         edgecolors='k', linewidths=0.4, zorder=5)
        else:
            ax_t.scatter(arr[:, 0], arr[:, 1], marker=mk, s=12, color=col,
                         edgecolors='k', linewidths=0.2, alpha=0.8, zorder=3)
        # track-ID label (matches the IDs printed on the photo boxes)
        ax_t.text(arr[-1, 0], arr[-1, 1], tid, fontsize=5, color='white',
                  ha='center', va='center', zorder=6,
                  bbox=dict(fc=col, ec='none', alpha=0.85, pad=0.4))

    # Ego vehicle + CAM_FRONT field-of-view wedge (forward = up).
    from matplotlib.patches import Wedge
    rmax = (max(max(np.abs(xs)) if xs else 20,
                max(np.abs(ys)) if ys else 20)) * 1.05
    ax_t.add_patch(Wedge((0, 0), rmax, 55, 125, color='gold', alpha=0.12,
                         zorder=0))
    ax_t.scatter([0], [0], marker='^', s=90, color='k', zorder=7)
    ax_t.text(0, -rmax*0.06, 'ego', fontsize=6, ha='center', va='top')
    ax_t.set_xlim(-rmax, rmax); ax_t.set_ylim(-rmax*0.15, rmax)
    ax_t.set_aspect('equal'); ax_t.grid(True, ls=':', alpha=0.4)
    ax_t.set_xlabel('right (m)'); ax_t.set_ylabel('forward (m)')
    ax_t.set_title('(b) BEV tracklets, ego-aligned (AMOTA 0.604)',
                   fontsize=10)

    fig.suptitle('Velocity-aware Hungarian tracking on nuScenes val: 3D '
                 'track boxes AND projected motion trajectories on the '
                 'start/end CAM\\_FRONT images (a,c), with matching BEV '
                 'tracklets (b). Colors are consistent per track ID.',
                 fontsize=9.5, y=1.04)
    suffix = '' if case_idx == 0 else f'_{case_idx+1}'
    out = os.path.join(HERE, f'tracking_overlay{suffix}.pdf')
    fig.savefig(out)
    print(f'[overlay] wrote {out} ({len(traj)} tracklets)')
    plt.close(fig)


if __name__ == '__main__':
    main()
