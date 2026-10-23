"""
Example 3 — the two ways to get pixels into voxels, side by side, with the SAME frozen backbone and decoder:

  PUSH  models/lss_occ.py   each pixel predicts a depth distribution and pushes its feature along its ray (lift-splat)
  PULL  models/proj_occ.py  each voxel centre projects into every camera and samples the feature there (no depth)

Measures trainable parameters, forward time (median of 10 calls), peak GPU memory and the voxel volume each lift produces (how many voxels
receive any feature) on one real official-val frame. These are UNTRAINED heads: this compares cost and coverage,
not accuracy.

    python -m DeepDataMiningLearning.ngperception.occupancy.examples.ex3_push_vs_pull \
        --gts /data/cmpe249-fa26/derived/nuscenes/occ3d/gts --nusc /data/cmpe249-fa26/datasets/nuscenes/v1.0-trainval
"""
from __future__ import annotations
import argparse, json, os, time

import numpy as np
import torch

from ..datasets_train import NuScenesOccTrainDataset
from ..models.lss_occ import LSSOccupancy
from ..models.proj_occ import ProjOccupancy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gts', required=True); ap.add_argument('--nusc', required=True)
    ap.add_argument('--backbone', default='dinov2_base'); ap.add_argument('--out', default='ex3_push_vs_pull.json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); a = ap.parse_args()
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    dev = a.device
    push = LSSOccupancy(backbone=a.backbone, decoder_layers=4, decoder_hidden=96, refine_iters=1).to(dev).eval()
    pull = ProjOccupancy(backbone=a.backbone, decoder_layers=4, decoder_hidden=96).to(dev).eval()
    nusc = NuScenes('v1.0-trainval', a.nusc, verbose=False)
    ds = NuScenesOccTrainDataset(a.gts, nusc, image_hw=push.image_hw, downsample=push.downsample,
                                 scenes=sorted(set(splits.val) & set(os.listdir(a.gts))), max_samples=1)
    b = ds[0]; x = [b[k].unsqueeze(0).to(dev) for k in ('imgs', 'rots', 'trans', 'intrins')]
    res = {}
    for name, m in (('PUSH lss_occ', push), ('PULL proj_occ', pull)):
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
        feats = {}
        hook = m.decoder.register_forward_hook(lambda mod, inp, out: feats.__setitem__('vox', inp[0].detach()))
        with torch.no_grad():
            m(*x)                                                         # warm-up
            if dev == 'cuda':
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            times = []                                                    # median of 10 synchronised calls: robust
            for _ in range(10):                                           # to other jobs on a shared GPU
                t0 = time.time(); m(*x)
                if dev == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - t0)
            ms = float(np.median(times)) * 1e3
        hook.remove()
        vox = feats['vox'][0]                                             # decoder input (C, 200, 200, 16)
        img = vox[:m.C]                                                   # image-feature channels only (PULL appends
        cov = float((img.abs().sum(0) > 0).float().mean())                # Fourier position + visibility count)
        res[name] = dict(trainable_M=round(trainable, 2), forward_ms=round(ms, 1),
                         peak_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == 'cuda' else None,
                         decoder_input_channels=int(vox.shape[0]), voxels_with_an_image_feature=round(cov, 3))
    print(json.dumps(res, indent=1)); json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
