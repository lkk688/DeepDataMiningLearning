"""
Example 1 — walk through the lift-splat occupancy model, stage by stage, on ONE official-val nuScenes frame.

It re-runs `LSSOccupancy.forward` by hand (encoder -> depth softmax -> frustum geometry -> lift -> splat -> 3D decoder ->
render-and-refine -> lift again), prints every tensor shape, checks that the manual pass equals `model(...)`, and draws:

  (a) CAM_FRONT with the cells whose depth distributions are plotted
  (b) those depth distributions: pass 1, after render-and-refine, and the LiDAR depth bin
  (c) expected depth of CAM_FRONT (pass 1) vs the projected LiDAR depth
  (d) the splatted context volume seen from above: camera rays are visible as "fans"
  (e) prediction and (f) Occ3D GT, both restricted to camera-visible voxels (what the metric scores)

    python -m DeepDataMiningLearning.ngperception.occupancy.examples.ex1_lift_splat_walkthrough \
        --gts /data/cmpe249-fa26/derived/nuscenes/occ3d/gts --nusc /data/cmpe249-fa26/datasets/nuscenes/v1.0-trainval \
        --ckpt <lss_occ.pth> --sample-idx 120 --out ex1.png
"""
from __future__ import annotations
import argparse, json, os, time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..datasets_train import NuScenesOccTrainDataset
from ..evaluator import OccupancyEvaluator
from ..geom import CAMS
from ..models.lss_occ import LSSOccupancy
from ..visualize import OCC3D_PALETTE, bev_semantic

MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])


def val_dataset(gts, nusc, model, n):
    from nuscenes.utils import splits
    scenes = sorted(set(splits.val) & set(os.listdir(gts)))
    return NuScenesOccTrainDataset(gts, nusc, image_hw=model.image_hw, downsample=model.downsample, scenes=scenes, max_samples=n)


def bev_rgb(sem, mask):
    rgb = OCC3D_PALETTE[bev_semantic(np.where(mask, sem, 17))]
    rgb[~mask.any(2)] = 0.82
    return rgb.transpose(1, 0, 2)[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gts', required=True); ap.add_argument('--nusc', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--sample-idx', type=int, default=120); ap.add_argument('--out', default='ex1_lift_splat.png')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); a = ap.parse_args()
    from nuscenes import NuScenes
    dev = a.device
    model = LSSOccupancy(backbone='dinov2_base', decoder_layers=4, decoder_hidden=96, refine_iters=2).to(dev).eval()
    sd = torch.load(a.ckpt, map_location=dev); model.load_state_dict(sd.get('state_dict', sd), strict=False)
    nusc = NuScenes('v1.0-trainval', a.nusc, verbose=False)
    ds = val_dataset(a.gts, nusc, model, a.sample_idx + 1); b = ds[a.sample_idx]
    imgs, rots, trans, K = (b[k].unsqueeze(0).to(dev) for k in ('imgs', 'rots', 'trans', 'intrins'))
    B, N = imgs.shape[:2]
    log = {}
    with torch.no_grad():
        t0 = time.time()
        ctx, depth = model.encoder(imgs.flatten(0, 1))                          # backbone + DepthNet
        C, D, h, w = ctx.shape[1], depth.shape[1], ctx.shape[2], ctx.shape[3]
        ctx = ctx.view(B, N, C, h, w); depth = depth.view(B, N, D, h, w)
        geom = model.get_geometry(rots, trans, K)                               # frustum -> ego xyz
        vox1 = model.voxel_pool(geom, depth.unsqueeze(2) * ctx.unsqueeze(3))    # lift (outer product) + splat
        occ1 = model.decoder(vox1)                                              # 3D CNN -> 18 logits / voxel
        depth2 = model._refine_depth(occ1, depth, geom)                         # render occupancy along rays
        vox2 = model.voxel_pool(geom, depth2.unsqueeze(2) * ctx.unsqueeze(3))
        occ2 = model.decoder(vox2)
        if dev == 'cuda':
            torch.cuda.synchronize()
        log['manual_pass_s'] = round(time.time() - t0, 3)
        ref = model(imgs, rots, trans, K)[0]
        # index_add_ on CUDA sums with float atomics, so two identical forward calls differ in the last bits; compare
        # the decisions (argmax) and the relative size of the difference rather than exact equality.
        log['argmax_agreement_vs_model_forward'] = float((ref.argmax(1) == occ2.argmax(1)).float().mean())
        log['max_rel_diff_vs_model_forward'] = float((ref - occ2).abs().max() / ref.abs().max())
    shapes = {'imgs': imgs.shape, 'context': ctx.shape, 'depth (softmax over bins)': depth.shape, 'frustum geometry (ego xyz)': geom.shape,
              'lifted = depth x context': (B, N, C, D, h, w), 'voxel volume (after splat)': vox1.shape, 'occupancy logits': occ2.shape}
    for k, v in shapes.items():
        print(f'{k:30s} {tuple(v)}')
    log['shapes'] = {k: list(v) for k, v in shapes.items()}

    # ---- per-frame score (pass 1 vs refined) on camera-visible voxels
    sem = b['semantics'].numpy(); mask = b['mask_camera'].numpy().astype(bool)
    for tag, o in (('pass1', occ1), ('refined', occ2)):
        ev = OccupancyEvaluator(); ev.add(o.argmax(1)[0].cpu().numpy(), sem, mask); r = ev.summarize(verbose=False)
        present = [c for c in range(17) if (sem[mask] == c).any()]
        log[f'{tag}_geo_IoU'] = round(r['geo_IoU'], 3)
        log[f'{tag}_mIoU_over_present_classes'] = round(float(np.mean([list(r['per_class'].values())[c] for c in present])), 3)
    print(json.dumps({k: v for k, v in log.items() if k != 'shapes'}, indent=1))

    # ---- figure
    cam = CAMS.index('CAM_FRONT'); dgt = b['depth_gt'][cam].numpy()                 # (h, w) LiDAR bin, -1 = none
    centers = model.dlo + model.dstep * np.arange(D)
    p1 = depth[0, cam].cpu().numpy(); p2 = depth2[0, cam].cpu().numpy()
    vv, uu = np.where(dgt >= 0)
    pick = []                                                                        # a near, a mid and a far cell
    for lo, hi in ((5, 12), (15, 25), (30, 50)):
        k = np.where((centers[dgt[vv, uu]] > lo) & (centers[dgt[vv, uu]] < hi))[0]
        if len(k):
            pick.append((vv[k[len(k) // 2]], uu[k[len(k) // 2]]))
    img = (imgs[0, cam].permute(1, 2, 0).cpu().numpy() * STD + MEAN).clip(0, 1)
    fig = plt.figure(figsize=(17, 9.4)); gs = fig.add_gridspec(2, 3, wspace=0.22, hspace=0.28)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(img); ax.axis('off'); ax.set_title('(a) CAM_FRONT, 252×700 → 18×50 feature cells')
    cols = ['#2563EB', '#10B981', '#EF4444']; s = model.downsample
    for (v, u), c in zip(pick, cols):
        ax.add_patch(plt.Rectangle((u * s, v * s), s, s, fill=False, ec=c, lw=2.5))
    ax = fig.add_subplot(gs[0, 1])
    for (v, u), c in zip(pick, cols):
        ax.plot(centers, p1[:, v, u], color=c, lw=1.4, alpha=0.55)
        ax.plot(centers, p2[:, v, u], color=c, lw=2.2)
        ax.axvline(centers[dgt[v, u]], color=c, ls='--', lw=1.2)
    ax.set_xlabel('depth (m), 112 bins of 0.5 m'); ax.set_ylabel('probability')
    ax.set_title('(b) depth softmax: thin = pass 1, thick = refined,\n dashed = LiDAR depth')
    ax = fig.add_subplot(gs[0, 2])
    exp_d = (p1 * centers[:, None, None]).sum(0)
    im = ax.imshow(exp_d, cmap='turbo', vmin=0, vmax=58); ax.scatter(uu, vv, c=centers[dgt[vv, uu]], cmap='turbo', vmin=0, vmax=58, s=10, edgecolors='k', linewidths=0.3)
    ax.set_title('(c) expected depth (pass 1)\n+ LiDAR depth dots'); ax.axis('off'); fig.colorbar(im, ax=ax, fraction=0.03)
    ax = fig.add_subplot(gs[1, 0])
    mag = vox1[0].abs().sum((0, 3)).cpu().numpy()                                     # (nx, ny)
    ax.imshow(np.log1p(mag).T[::-1], cmap='magma'); ax.set_title('(d) splatted context volume, top view:\nfeatures spread along camera rays'); ax.axis('off')
    ax = fig.add_subplot(gs[1, 1]); ax.imshow(bev_rgb(occ2.argmax(1)[0].cpu().numpy(), mask)); ax.axis('off')
    ax.set_title(f'(e) prediction (camera-visible voxels)\nframe geo-IoU {log["refined_geo_IoU"]}')
    ax = fig.add_subplot(gs[1, 2]); ax.imshow(bev_rgb(sem, mask)); ax.axis('off'); ax.set_title('(f) Occ3D GT (grey = not scored)')
    fig.savefig(a.out, dpi=130, bbox_inches='tight', facecolor='white'); plt.close(fig)
    json.dump(log, open(os.path.splitext(a.out)[0] + '.json', 'w'), indent=1); print('wrote', a.out)


if __name__ == '__main__':
    main()
