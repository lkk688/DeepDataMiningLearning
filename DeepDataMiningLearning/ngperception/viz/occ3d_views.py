"""
ngperception.viz.occ3d_views
============================

Occ3D-nuScenes ground-truth figures (TUTORIAL §2.0 / §7):

  occ3d_gt_bev.png       GT top-down (top-most occupied voxel per column) | only camera-visible voxels | visibility map
  occ3d_classfreq.png    class frequency inside mask_camera over N random frames (log scale): why the loss matters

    python -m DeepDataMiningLearning.ngperception.viz.occ3d_views --gts <Occ3D gts/> \
        --scene scene-0018 --token 2ff5689a089844f0a2097b1883916003 --out figures
`gts/<scene>/<sample_token>/labels.npz` holds semantics (200,200,16; 17 = free), mask_camera, mask_lidar.
"""
from __future__ import annotations
import argparse, glob, json, os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..occupancy.visualize import OCC3D_PALETTE as PAL, CLASS_NAMES as NAMES, bev_semantic

INK, SUB = '#0F172A', '#64748B'


def gt_bev(gts, scene, token, out):
    z = np.load(os.path.join(gts, scene, token, 'labels.npz'))
    sem, mc = z['semantics'], z['mask_camera'].astype(bool)
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.0), gridspec_kw=dict(wspace=0.28))
    panels = ((PAL[bev_semantic(sem)], 'Occ3D GT (top-most occupied voxel)'),
              (PAL[bev_semantic(np.where(mc, sem, 17))], 'camera-visible voxels only\n(= what mIoU scores)'))
    for ax, (img, t) in zip(axs[:2], panels):
        ax.imshow(img.transpose(1, 0, 2), origin='lower', extent=(-40, 40, -40, 40)); ax.set_title(t, color=INK)
    im = axs[2].imshow(mc.mean(-1).T, origin='lower', cmap='Blues', extent=(-40, 40, -40, 40))
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.02); axs[2].set_title('fraction of each column\nvisible to the cameras', color=INK)
    for ax in axs:
        ax.add_patch(plt.Rectangle((-2.3, -1), 4.6, 2, color=INK)); ax.set_xlabel('x forward (m)'); ax.set_ylabel('y left (m)')
    fig.savefig(os.path.join(out, 'occ3d_gt_bev.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
    occ = sem != 17
    return dict(occupied=int(occ.sum()), occupied_in_camera_mask=int((occ & mc).sum()), camera_mask_frac=float(mc.mean()))


def class_freq(gts, out, n=120, seed=0):
    files = sorted(glob.glob(os.path.join(gts, '*', '*', 'labels.npz'))); rng = np.random.default_rng(seed)
    cnt = np.zeros(18)
    for f in rng.choice(files, min(n, len(files)), replace=False):
        z = np.load(f); cnt += np.bincount(z['semantics'][z['mask_camera'].astype(bool)].ravel(), minlength=18)[:18]
    frac = cnt / cnt.sum(); order = np.argsort(-frac)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.bar(range(18), frac[order] * 100, color=[PAL[i] if i != 17 else (0.85, 0.88, 0.92) for i in order], edgecolor=SUB)
    ax.set_yscale('log'); ax.set_xticks(range(18)); ax.set_xticklabels([NAMES[i] for i in order], rotation=40, ha='right')
    for k, i in enumerate(order):
        ax.text(k, frac[i] * 115, f'{frac[i] * 100:.2g}', ha='center', va='bottom', fontsize=9, color=INK)
    ax.set_ylabel('% of camera-visible voxels (log)')
    ax.set_title(f'Occ3D-nuScenes class frequency inside mask_camera ({n} random frames)', color=INK)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(out, 'occ3d_classfreq.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
    return {NAMES[i]: round(float(frac[i]), 6) for i in range(18)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gts', required=True)
    ap.add_argument('--scene', default='scene-0018'); ap.add_argument('--token', default='2ff5689a089844f0a2097b1883916003')
    ap.add_argument('--n', type=int, default=120); ap.add_argument('--out', default='figures'); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = gt_bev(a.gts, a.scene, a.token, a.out); meta['class_freq'] = class_freq(a.gts, a.out, a.n)
    json.dump(meta, open(os.path.join(a.out, 'occ3d_views_meta.json'), 'w'), indent=1)
    print({k: v for k, v in meta.items() if k != 'class_freq'})


if __name__ == '__main__':
    main()
