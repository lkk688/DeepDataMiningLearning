"""
ngperception.viz.occ_video
==========================

Occupancy videos from a trained camera-only occupancy model (models/lss_occ.py) on nuScenes (TUTORIAL §7).

Three modes:

  keyframes  the 2 Hz annotated keyframes of a scene: 6 cameras | top-down map | 3D prediction next to the Occ3D
             ground truth. Both are restricted to the camera-visible voxels (what the metric scores), so the
             comparison is fair. The 3D camera sways gently between keyframes so the video is not a slideshow.
  sweeps     every 12 Hz camera sweep of the scene (Occ3D has no labels between keyframes): 6 cameras | top-down
             map | a wide 3D chase view of the prediction. Each camera is placed in the ego frame of CAM_FRONT
             with the ego poses, because the six cameras fire at slightly different times. Drawn: the camera-visible
             voxels of the nearest keyframe (Occ3D mask, warped by ego motion), or --sweep-mask raycast for unlabeled
             data (visibility by ray casting the prediction to its first hit).
  orbit      one keyframe; the 3D view circles the car 360 degrees: prediction | ground truth.

    python -m DeepDataMiningLearning.ngperception.viz.occ_video --mode sweeps --scene scene-0107 \\
        --gts /data/cmpe249-fa26/derived/nuscenes/occ3d/gts --nusc /data/cmpe249-fa26/datasets/nuscenes/v1.0-trainval \\
        --ckpt <lss_occ.pth> --out occ_scene-0107_sweeps.mp4

Two stages: the model runs once and caches each frame (prediction, visibility mask, camera mosaic) under <out>_cache/;
then a separate spawned process renders the 3D views (keeping VTK out of the process that holds CUDA + the
nuScenes tables, which stalled the renderer for ~20 s per frame). Rendering is headless (pyvista / VTK picks EGL on a
GPU node, OSMesa otherwise); the mp4 writer needs `pip install imageio-ffmpeg`. Use a scene from the official val split if the checkpoint was trained with --split train.
"""
from __future__ import annotations
import argparse, bisect, os, time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
import pyvista as pv  # noqa: E402

from ..occupancy.geom import CAMS, PC_RANGE, VOXEL_SIZE, FREE  # noqa: E402
from ..occupancy.visualize import OCC3D_PALETTE, CLASS_NAMES, bev_semantic  # noqa: E402

pv.OFF_SCREEN = True
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
GRID = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'], ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]
PAL8 = (OCC3D_PALETTE * 255).astype(np.uint8)
PAL8[0] = (90, 90, 90)                                           # "others": dark grey instead of black
PAL8[6] = (200, 180, 0)                                          # motorcycle: distinct from barrier (255, 120, 50)
X0, Y0, Z0 = PC_RANGE[:3]
BG, INK, SUB = (248, 250, 252), (15, 23, 42), (100, 116, 139)


def font(size, bold=False):
    import matplotlib
    f = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
    return ImageFont.truetype(f, size)


# ---------------------------------------------------------------- inputs
def _mat(rec):
    from pyquaternion import Quaternion
    T = np.eye(4); T[:3, :3] = Quaternion(rec['rotation']).rotation_matrix; T[:3, 3] = rec['translation']
    return T


def camera_inputs(nusc, sd_tokens, image_hw, ref_sd=None):
    """6 sample_data tokens -> model inputs (+ the raw images). With ref_sd, each camera is expressed in the ego frame of
    that sample_data (cam -> ego_i -> global -> ego_ref); without, in its own ego frame (as in training)."""
    H, W = image_hw
    E_ref_inv = np.linalg.inv(_mat(nusc.get('ego_pose', nusc.get('sample_data', ref_sd)['ego_pose_token']))) if ref_sd else None
    imgs, rots, trans, Ks, raw = [], [], [], [], {}
    for cam in CAMS:
        sd = nusc.get('sample_data', sd_tokens[cam]); cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        img = Image.open(os.path.join(nusc.dataroot, sd['filename'])).convert('RGB'); raw[cam] = img
        ow, oh = img.size
        arr = (np.asarray(img.resize((W, H)), np.float32) / 255.0 - MEAN) / STD
        imgs.append(arr.transpose(2, 0, 1))
        Ks.append(np.diag([W / ow, H / oh, 1.0]) @ np.array(cs['camera_intrinsic']))
        T = _mat(cs)
        if E_ref_inv is not None:
            T = E_ref_inv @ _mat(nusc.get('ego_pose', sd['ego_pose_token'])) @ T
        rots.append(T[:3, :3]); trans.append(T[:3, 3])
    f = lambda x: torch.from_numpy(np.stack(x).astype(np.float32)).unsqueeze(0)
    return (f(imgs), f(rots), f(trans), f(Ks)), raw


@torch.no_grad()
def visible_mask(pred, rots, trans, Ks, image_hw, stride=4, step=0.2, max_depth=60.0, behind=0.8):
    """Voxels a camera can see according to the PREDICTION: march rays from every stride-th pixel and keep the samples up
    to `behind` metres past the first occupied voxel. Same idea as Occ3D's mask_camera (which ray-casts the GT); use it
    when no GT mask exists (between keyframes). The model is only supervised on visible voxels, so the rest is noise."""
    dev = rots.device
    H, W = image_hw
    occ = torch.from_numpy(pred != FREE).to(dev); shape = torch.tensor(occ.shape, device=dev)
    lo = torch.tensor(PC_RANGE[:3], dtype=torch.float32, device=dev)
    vs, us = torch.meshgrid(torch.arange(stride / 2, H, stride, device=dev), torch.arange(stride / 2, W, stride, device=dev), indexing='ij')
    pix = torch.stack([us.flatten(), vs.flatten(), torch.ones_like(us.flatten())], -1)          # (P, 3)
    ds = torch.arange(step, max_depth, step, device=dev)                                         # (S,)
    vis = torch.zeros(occ.numel(), dtype=torch.bool, device=dev)
    for R, t, K in zip(rots[0], trans[0], Ks[0]):
        d = (torch.linalg.inv(K) @ pix.T).T; d = d / d.norm(dim=-1, keepdim=True)
        pts = t + (d @ R.T)[:, None, :] * ds[None, :, None]                                      # (P, S, 3) ego
        idx = ((pts - lo) / VOXEL_SIZE).floor().long()
        inside = ((idx >= 0) & (idx < shape)).all(-1)
        idx = idx.clamp(min=torch.zeros_like(shape), max=shape - 1)
        flat = (idx[..., 0] * shape[1] + idx[..., 1]) * shape[2] + idx[..., 2]
        hit = occ.flatten()[flat] & inside
        first = torch.where(hit.any(1), ds[hit.float().argmax(1)], torch.full_like(ds[:1].expand(len(hit)), float('inf')))
        keep = inside & (ds[None, :] <= first[:, None] + behind)
        vis[flat[keep]] = True
    return vis.view(*occ.shape).cpu().numpy()


def warp_mask(mask_key, T_key_from_cur):
    """Occ3D camera mask of a keyframe, resampled into the ego frame of a nearby sweep (rigid ego motion only)."""
    shape = np.array(mask_key.shape)
    g = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), -1).reshape(-1, 3)
    p = PC_RANGE[:3] + (g + 0.5) * VOXEL_SIZE
    q = p @ T_key_from_cur[:3, :3].T + T_key_from_cur[:3, 3]
    idx = np.floor((q - PC_RANGE[:3]) / VOXEL_SIZE).astype(int)
    ok = ((idx >= 0) & (idx < shape)).all(1)
    out = np.zeros(len(p), bool); out[ok] = mask_key[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return out.reshape(mask_key.shape)


def load_model(ckpt, dev):
    from ..occupancy.models.lss_occ import LSSOccupancy
    m = LSSOccupancy(backbone='dinov2_base', decoder_layers=4, decoder_hidden=96, refine_iters=2).to(dev).eval()
    sd = torch.load(ckpt, map_location=dev); m.load_state_dict(sd.get('state_dict', sd), strict=False)
    return m


@torch.no_grad()
def predict(model, inputs, dev):
    return model(*[x.to(dev) for x in inputs])[0].argmax(1)[0].cpu().numpy().astype(np.uint8)


def sweep_groups(nusc, scene):
    """For every CAM_FRONT sweep of the scene, the time-nearest sweep of each other camera."""
    first = nusc.get('sample', scene['first_sample_token'])
    chains = {}
    for cam in CAMS:
        ts, tok = [], first['data'][cam]
        while tok:
            sd = nusc.get('sample_data', tok); ts.append((sd['timestamp'], tok)); tok = sd['next']
        chains[cam] = ts
    out = []
    for t, tok in chains['CAM_FRONT']:
        g = {'CAM_FRONT': tok}
        for cam in CAMS[1:]:
            tt = [x[0] for x in chains[cam]]; k = bisect.bisect_left(tt, t)
            best = min([j for j in (k - 1, k) if 0 <= j < len(tt)], key=lambda j: abs(tt[j] - t))
            g[cam] = chains[cam][best][1]
        out.append((t, g))
    return out


# ---------------------------------------------------------------- rendering
def clear_ego(sem):
    s = sem.copy()
    ix = slice(int((-2.4 - X0) / VOXEL_SIZE), int((4.8 - X0) / VOXEL_SIZE)); iy = slice(int((-1.8 - Y0) / VOXEL_SIZE), int((1.8 - Y0) / VOXEL_SIZE))
    s[ix, iy, :] = FREE
    return s


class VoxelRenderer:
    """Colored cubes with thin edges, soft lighting and a gradient sky; one persistent off-screen plotter."""

    def __init__(self, size, edges=True):
        self.size = size; self.edges = edges
        self.p = pv.Plotter(off_screen=True, window_size=size, lighting='three lights')
        self.p.set_background('#f8fafc', top='#cbd5e1')
        try:
            self.p.enable_anti_aliasing('ssaa')
        except Exception:
            pass
        nx, ny, nz = ((PC_RANGE[3:] - PC_RANGE[:3]) / VOXEL_SIZE).round().astype(int)
        self.grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=(VOXEL_SIZE,) * 3, origin=tuple(PC_RANGE[:3]))
        self.ego = [pv.Box(bounds=(-1.0, 3.7, -0.95, 0.95, 0.2, 1.2)), pv.Box(bounds=(-0.4, 2.6, -0.85, 0.85, 1.2, 1.75)),
                    pv.Cone(center=(5.2, 0, 0.7), direction=(1, 0, 0), height=1.6, radius=0.55, resolution=24)]
        self.ground = pv.Plane(center=(0, 0, Z0 + 0.01), direction=(0, 0, 1), i_size=80, j_size=80)

    def render(self, sem, view, key=None):
        p = self.p
        if key is not None and key == getattr(self, '_key', None):  # same volume (orbit): only move the camera
            p.camera_position = view
            return np.asarray(p.screenshot(return_img=True))[..., :3]
        self._key = key; p.clear_actors()
        keep = np.flatnonzero((sem != FREE).ravel(order='F'))
        if len(keep):
            cells = self.grid.extract_cells(keep)
            cells.cell_data['rgb'] = PAL8[sem.ravel(order='F')[keep]]
            p.add_mesh(cells, scalars='rgb', rgb=True, show_edges=self.edges, edge_color='#334155', line_width=0.35,
                       ambient=0.35, diffuse=0.75, specular=0.15, specular_power=12)
        p.add_mesh(self.ground, color='#e2e8f0', ambient=0.6, diffuse=0.4)
        p.add_mesh(self.ego[0], color='#1e293b', specular=0.4); p.add_mesh(self.ego[1], color='#334155', specular=0.4)
        p.add_mesh(self.ego[2], color='#facc15', ambient=0.5)
        p.camera_position = view
        return np.asarray(p.screenshot(return_img=True))[..., :3]

    def close(self):
        self.p.close()


def chase_view(sway_deg=0.0, dist=30.0, height=22.0, look=12.0):
    a = np.deg2rad(180 + sway_deg)
    return [(dist * np.cos(a), dist * np.sin(a), height), (look, 0.0, 0.0), (0, 0, 1)]


def orbit_view(az_deg, radius=48.0, height=30.0):
    a = np.deg2rad(az_deg)
    return [(radius * np.cos(a), radius * np.sin(a), height), (0.0, 0.0, 0.0), (0, 0, 1)]


def bev_image(sem, mask, size):
    """Top-down map, x forward = up, y left = left; grey = never visible, light = free."""
    top = bev_semantic(np.where(mask, sem, FREE))
    rgb = PAL8[top].copy(); rgb[top == FREE] = (255, 255, 255); rgb[~mask.any(2)] = (176, 186, 200)
    img = Image.fromarray(rgb[::-1, ::-1]).resize((size, size), Image.NEAREST)
    d = ImageDraw.Draw(img); c = size / 2; k = size / 80.0
    d.rectangle((c - 0.95 * k, c - 3.7 * k, c + 0.95 * k, c + 1.0 * k), fill=(15, 23, 42))
    return img


def label(d, xy, text, f, fg=(255, 255, 255), bg=(15, 23, 42, 170), pad=6):
    x, y = xy; b = d.textbbox((x, y), text, font=f)
    d.rectangle((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad), fill=bg); d.text((x, y), text, font=f, fill=fg)


def camera_grid(raw, w, h):
    """2 x 3 surround grid; front row FL F FR, back row BL B BR."""
    canvas = Image.new('RGB', (3 * w, 2 * h)); d = ImageDraw.Draw(canvas, 'RGBA'); f = font(15)
    for r, row in enumerate(GRID):
        for c, cam in enumerate(row):
            canvas.paste(raw[cam].resize((w, h)), (c * w, r * h))
            label(d, (c * w + 10, r * h + 8), cam.replace('CAM_', ''), f)
    return canvas


def legend_strip(width, height=30):
    img = Image.new('RGB', (width, height), (255, 255, 255)); d = ImageDraw.Draw(img); f = font(14)
    items = list(range(1, 17)) + [0]
    step = width / len(items)
    for i, c in enumerate(items):
        x = int(i * step) + 8
        d.rectangle((x, 8, x + 14, 22), fill=tuple(int(v) for v in PAL8[c]), outline=(100, 116, 139))
        d.text((x + 19, 7), CLASS_NAMES[c].replace('_', ' '), font=f, fill=INK)
    return img


def compose(cams, bev, panels, header, legend):
    """1920 x 1080: camera mosaic (1440x540) | top-down map (480x540) over 3D panels (1920x510) and a legend (30)."""
    W, H = 1920, 1080
    canvas = Image.new('RGB', (W, H), BG)
    canvas.paste(cams, (0, 0))
    d = ImageDraw.Draw(canvas, 'RGBA')
    d.text((1440 + 16, 10), 'top-down (camera-visible)', font=font(17, True), fill=INK)
    canvas.paste(bev, (1440 + 20, 44))
    d.text((1440 + 16, 44 + bev.height + 8), header, font=font(15), fill=SUB)
    x = 0
    for img, title in panels:
        canvas.paste(Image.fromarray(img), (x, 540))
        label(d, (x + 16, 552), title, font(18, True), fg=INK, bg=(255, 255, 255, 215))
        x += img.shape[1]
    canvas.paste(legend, (0, H - legend.height))
    return np.asarray(canvas)


def writer(path, fps, crf=26):
    import imageio.v2 as imageio
    return imageio.get_writer(path, fps=fps, codec='libx264', quality=None, macro_block_size=8, pixelformat='yuv420p',
                              output_params=['-crf', str(crf), '-preset', 'slow'])


# ---------------------------------------------------------------- stage 2: rendering workers
_R = {}


def _renderer(size):
    if size not in _R:
        _R[size] = VoxelRenderer(size)
    return _R[size]


def render_job(job):
    """Worker: one cached frame + a list of 3D views -> composed 1920x1080 frames. Runs in a spawned process that holds
    neither CUDA state nor the nuScenes tables (sharing a process with them stalled VTK rendering)."""
    z = np.load(job['npz'])
    pred, mask = z['pred'], np.unpackbits(z['mask'])[:z['pred'].size].reshape(z['pred'].shape).astype(bool)
    shown = {'pred': clear_ego(np.where(mask, pred, FREE))}
    if 'gt' in z.files:
        shown['gt'] = clear_ego(np.where(mask, z['gt'], FREE))
    cams = Image.open(job['cams']); bev = bev_image(pred, mask, 440); legend = legend_strip(1920)
    frames = []
    for v in job['views']:
        panels = [(_renderer(tuple(size)).render(shown[k], v, key=(job['npz'], k)), title) for k, title, size in job['panels']]
        frames.append(compose(cams, bev, panels, job['header'], legend))
    return frames


# ---------------------------------------------------------------- stage 1: predict + cache, then render
def run(a):
    from nuscenes import NuScenes
    import multiprocessing as mp
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = a.cache or os.path.splitext(a.out)[0] + '_cache'
    os.makedirs(cache, exist_ok=True); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    nusc = NuScenes('v1.0-trainval', a.nusc, verbose=False)
    scene = next(s for s in nusc.scene if s['name'] == a.scene)
    model = load_model(a.ckpt, dev); hw = model.image_hw
    name = 'prediction · camera-only DINOv2-B lift-splat'

    def save(i, pred, mask, raw, gt=None):
        f = os.path.join(cache, f'f{i:04d}.npz'); extra = {'gt': gt} if gt is not None else {}
        np.savez_compressed(f, pred=pred, mask=np.packbits(mask.ravel()), **extra)
        c = os.path.join(cache, f'f{i:04d}_cams.jpg'); camera_grid(raw, 480, 270).save(c, quality=92)
        return f, c

    jobs = []
    if a.mode == 'sweeps':
        groups = sweep_groups(nusc, scene)[::a.stride][:a.max_frames]; tstart = groups[0][0]
        keys, tok = [], scene['first_sample_token']
        while tok:
            s = nusc.get('sample', tok); keys.append(s); tok = s['next']
        pose = lambda sd_tok: _mat(nusc.get('ego_pose', nusc.get('sample_data', sd_tok)['ego_pose_token']))
        kts = np.array([k['timestamp'] for k in keys]); kmask = {}
        for i, (ts, g) in enumerate(groups):
            inp, raw = camera_inputs(nusc, g, hw, ref_sd=g['CAM_FRONT'])
            pred = predict(model, inp, dev)
            if a.sweep_mask == 'raycast':
                mask = visible_mask(pred, *[x.to(dev) for x in inp[1:]], hw)
            else:                                                 # nearest keyframe's Occ3D camera mask, ego-motion warped
                k = keys[int(np.abs(kts - ts).argmin())]
                if k['token'] not in kmask:
                    kmask[k['token']] = np.load(os.path.join(a.gts, a.scene, k['token'], 'labels.npz'))['mask_camera'].astype(bool)
                T = np.linalg.inv(pose(k['data']['LIDAR_TOP'])) @ pose(g['CAM_FRONT'])
                mask = warp_mask(kmask[k['token']], T)
            f, c = save(i, pred, mask, raw)
            jobs.append(dict(npz=f, cams=c, header=f'{a.scene} · t = {(ts - tstart) / 1e6:5.1f} s · frame {i}',
                             views=[chase_view(sway_deg=8 * np.sin(i / len(groups) * 2 * np.pi), dist=34, height=20, look=14)],
                             panels=[('pred', name + '  · 12 Hz camera sweeps · ' + ('visibility by ray casting the prediction' if a.sweep_mask == 'raycast' else 'camera-visible voxels (nearest keyframe mask, ego-motion warped)'), (1920, 510))]))
    else:
        samples, tok = [], scene['first_sample_token']
        while tok:
            s = nusc.get('sample', tok); samples.append(s); tok = s['next']
        if a.mode == 'orbit':
            samples = [samples[min(a.still_at, len(samples) - 1)]]
        samples = samples[:a.max_frames]
        panels = [('pred', name, (960, 510)), ('gt', 'Occ3D ground truth', (960, 510))]
        for i, s in enumerate(samples):
            inp, raw = camera_inputs(nusc, {c: s['data'][c] for c in CAMS}, hw)
            z = np.load(os.path.join(a.gts, a.scene, s['token'], 'labels.npz'))
            f, c = save(i, predict(model, inp, dev), z['mask_camera'].astype(bool), raw, gt=z['semantics'])
            hdr = f'{a.scene} · keyframe {i} · t = {(s["timestamp"] - samples[0]["timestamp"]) / 1e6:4.1f} s'
            if a.mode == 'orbit':
                views = [orbit_view(az) for az in np.linspace(0, 360, a.orbit_frames, endpoint=False)]
                jobs += [dict(npz=f, cams=c, header=hdr, views=views[k:k + 24], panels=panels) for k in range(0, len(views), 24)]
            else:
                views = [chase_view(sway_deg=10 * np.sin((i * a.hold + k) / (len(samples) * a.hold) * 2 * np.pi)) for k in range(a.hold)]
                jobs.append(dict(npz=f, cams=c, header=hdr, views=views, panels=panels))
    print(f'[1/2] predicted {len(jobs)} jobs into {cache} ({time.time() - t0:.0f}s)', flush=True)
    model = nusc = None                                          # free GPU + RAM before rendering
    if dev == 'cuda':
        torch.cuda.empty_cache()

    still = os.path.splitext(a.out)[0] + '_still.png'; n = 0
    still_at = 0 if a.mode == 'orbit' else min(a.still_at, len(jobs) - 1) * len(jobs[0]['views'])
    w = writer(a.out, a.fps, a.crf)
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for frames in pool.imap(render_job, jobs):
            for fr in frames:
                if n == still_at:
                    Image.fromarray(fr).save(still)
                w.append_data(fr); n += 1
            if n % 60 < len(frames):
                print(f'  [2/2] {n} frames  {time.time() - t0:.0f}s', flush=True)
    w.close()
    print(f'wrote {a.out} ({n} frames, {os.path.getsize(a.out) / 1e6:.1f} MB) and {still} in {time.time() - t0:.0f}s')


def main():
    ap = argparse.ArgumentParser(description='Occupancy videos (keyframes vs GT, 12 Hz sweeps, or a 360-degree orbit).')
    ap.add_argument('--mode', choices=['keyframes', 'sweeps', 'orbit'], default='keyframes')
    ap.add_argument('--scene', default='scene-0107'); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--gts', required=True); ap.add_argument('--nusc', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=None, help='default: keyframes 6, sweeps 12, orbit 30')
    ap.add_argument('--hold', type=int, default=3, help='keyframes: rendered frames per keyframe (camera sways between)')
    ap.add_argument('--stride', type=int, default=1, help='sweeps: use every k-th sweep')
    ap.add_argument('--sweep-mask', choices=['keyframe', 'raycast'], default='keyframe',
                    help='sweeps: which voxels to draw. keyframe = Occ3D camera mask of the nearest keyframe warped by ego motion '
                         '(visibility only, no GT classes); raycast = visibility from the prediction itself (works without labels)')
    ap.add_argument('--max-frames', type=int, default=10 ** 6)
    ap.add_argument('--crf', type=int, default=26, help='x264 quality: lower = better / larger (18 near-lossless, 28 small)')
    ap.add_argument('--orbit-frames', type=int, default=240)
    ap.add_argument('--still-at', type=int, default=10, help='keyframe / sweep saved as <out>_still.png (orbit: the keyframe)')
    ap.add_argument('--workers', type=int, default=1, help='rendering processes (>1 only helps with several GPUs: concurrent\n'
                    'EGL contexts on one GPU were ~70x slower than one on our nodes)')
    ap.add_argument('--cache', default=None, help='where predictions are cached (default <out>_cache/)')
    a = ap.parse_args()
    a.fps = a.fps or {'keyframes': 6, 'sweeps': 12, 'orbit': 30}[a.mode]
    run(a)


if __name__ == '__main__':
    main()
