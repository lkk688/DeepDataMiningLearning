"""
Example 2 — evaluate an occupancy checkpoint on official-val frames the right way, and see three things a single
number hides:

  1. the official mIoU accumulates TP / FP / FN over ALL frames first, then takes IoU per class; averaging per-frame
     IoUs instead gives a different number (rare classes are absent from most frames);
  2. per-class IoU: the mean hides a 0.0 - 0.8 spread between object classes and surfaces;
  3. render-and-refine: pass 1 vs the refined pass of the same model on the same frames.

Frames are spread over all 150 val scenes (every k-th frame), so this is a fair, cheap estimate of the full-val number.

    python -m DeepDataMiningLearning.ngperception.occupancy.examples.ex2_evaluate_clean_model \
        --gts /data/cmpe249-fa26/derived/nuscenes/occ3d/gts --nusc /data/cmpe249-fa26/datasets/nuscenes/v1.0-trainval \
        --ckpt <lss_occ.pth> --n 300 --out ex2_eval
"""
from __future__ import annotations
import argparse, json, os, time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..datasets_train import NuScenesOccTrainDataset
from ..evaluator import OccupancyEvaluator, OCC3D_CLASSES
from ..models.lss_occ import LSSOccupancy
from ..visualize import OCC3D_PALETTE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gts', required=True); ap.add_argument('--nusc', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=300); ap.add_argument('--out', default='ex2_eval')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); a = ap.parse_args()
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    dev = a.device
    model = LSSOccupancy(backbone='dinov2_base', decoder_layers=4, decoder_hidden=96, refine_iters=2).to(dev).eval()
    sd = torch.load(a.ckpt, map_location=dev); model.load_state_dict(sd.get('state_dict', sd), strict=False)
    nusc = NuScenes('v1.0-trainval', a.nusc, verbose=False)
    scenes = sorted(set(splits.val) & set(os.listdir(a.gts)))
    full = NuScenesOccTrainDataset(a.gts, nusc, image_hw=model.image_hw, downsample=model.downsample, scenes=scenes)
    stride = max(1, len(full) // a.n)
    ds = NuScenesOccTrainDataset(a.gts, nusc, image_hw=model.image_hw, downsample=model.downsample, scenes=scenes,
                                 max_samples=a.n, stride=stride)
    print(f'{len(ds)} frames = every {stride}th of the {len(full)} official-val frames ({len(scenes)} scenes)')
    ev = {k: OccupancyEvaluator() for k in ('pass1', 'refined')}
    per_frame = {k: [] for k in ev}
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            x = [b[k].unsqueeze(0).to(dev) for k in ('imgs', 'rots', 'trans', 'intrins')]
            occ, _, aux = model(*x)
            sem = b['semantics'].numpy(); mask = b['mask_camera'].numpy().astype(bool)
            for k, o in (('pass1', aux['occ'][0]), ('refined', occ)):
                pred = o.argmax(1)[0].cpu().numpy(); ev[k].add(pred, sem, mask)
                one = OccupancyEvaluator(); one.add(pred, sem, mask); r = one.summarize(verbose=False)
                present = [c for c in range(17) if (sem[mask] == c).any()]       # classes present in this frame
                per_frame[k].append(np.mean([r['per_class'][OCC3D_CLASSES[c]] for c in present]))
    res = {'n_frames': len(ds), 'stride': stride, 'seconds': round(time.time() - t0, 1)}
    for k in ev:
        r = ev[k].summarize(verbose=False)
        res[k] = dict(mIoU_official_accumulated=round(r['mIoU'], 4), geo_IoU=round(r['geo_IoU'], 4),
                      mean_of_per_frame_mIoU=round(float(np.mean(per_frame[k])), 4),
                      per_class={c: round(v, 4) for c, v in r['per_class'].items()})
    print(json.dumps({k: (v if k not in ev else {kk: vv for kk, vv in v.items() if kk != 'per_class'}) for k, v in res.items()}, indent=1))
    json.dump(res, open(a.out + '.json', 'w'), indent=1)

    pc = res['refined']['per_class']; names = list(pc); vals = np.array([pc[n] for n in names]); order = np.argsort(-vals)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.bar(range(17), vals[order], color=[OCC3D_PALETTE[OCC3D_CLASSES.index(names[i])] for i in order], edgecolor='#64748B')
    for k, i in enumerate(order):
        ax.text(k, vals[i] + 0.01, f'{vals[i]:.2f}', ha='center', fontsize=9)
    ax.axhline(res['refined']['mIoU_official_accumulated'], color='#0F172A', ls='--', lw=1.2)
    ax.text(16.4, res['refined']['mIoU_official_accumulated'] + 0.015, f"mIoU {res['refined']['mIoU_official_accumulated']:.3f}", ha='right')
    ax.set_xticks(range(17)); ax.set_xticklabels([names[i] for i in order], rotation=40, ha='right'); ax.set_ylabel('IoU (camera-visible voxels)')
    ax.set_title(f'Leakage-free LSS (DINOv2-B) on {len(ds)} official-val frames: per-class IoU', fontsize=12.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.savefig(a.out + '_per_class.png', dpi=140, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print('wrote', a.out + '.json', a.out + '_per_class.png')


if __name__ == '__main__':
    main()
