"""
ngperception.occupancy.cache_vggt_depth
========================================
Precompute **frozen VGGT depth** for the LSS occupancy lift (ablation #2, the trained
VGGT-backbone A/B). VGGT-1B is 1.26B params and a 6-camera forward is ~0.8 s — far too slow
to run every training step — but it is *frozen*, so we cache its output once and train fast.

For each nuScenes sample we run `facebook/VGGT-1B` on the 6 surround cameras at the SAME
252x700 the DINOv2 lift uses (both ÷14 -> 18x50 patch grid), take its dense per-camera depth,
and **block-min downsample** to the (18,50) feature grid (nearest visible surface per block).
Saved as `<sample_token>.npy` (6, fH, fW) float16. `train_lss.py --vggt-depth-cache DIR` then
blends this as a metric-depth prior (a learned scalar recovers VGGT's up-to-scale factor).

    python -m DeepDataMiningLearning.ngperception.occupancy.cache_vggt_depth \
        --gts <gts> --nusc <nuscenes> --out <cache_dir> --cap 2100 \
        --vggt-path /data/rnd-liu/Others/VGGT-Det-CVPR2026
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch

from .geom import CAMS
from .datasets import Occ3DNuScenesDataset


def main():
    ap = argparse.ArgumentParser(description="Precompute frozen-VGGT depth for the occupancy lift.")
    ap.add_argument("--gts", required=True)
    ap.add_argument("--nusc", required=True)
    ap.add_argument("--out", required=True, help="cache dir for <token>.npy (6,fH,fW)")
    ap.add_argument("--cap", type=int, default=2100, help="cache the first N occ items (covers "
                    "train max_samples + val stride subset)")
    ap.add_argument("--H", type=int, default=252)
    ap.add_argument("--W", type=int, default=700)
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--vggt-path", default="/data/rnd-liu/Others/VGGT-Det-CVPR2026")
    args = ap.parse_args()

    from PIL import Image
    from nuscenes import NuScenes
    fH, fW = args.H // args.patch, args.W // args.patch
    os.makedirs(args.out, exist_ok=True)

    sys.path.insert(0, args.vggt_path)
    from vggt.models.vggt import VGGT
    print("[cache] loading facebook/VGGT-1B ...", flush=True)
    vggt = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
    for p in vggt.parameters():
        p.requires_grad = False

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    occ = Occ3DNuScenesDataset(args.gts, scenes=None)          # all scenes (same order train_lss sees)
    n = min(args.cap, len(occ))
    print(f"[cache] {n} samples -> {args.out} | VGGT input {args.W}x{args.H} -> grid {fH}x{fW}",
          flush=True)

    done = skip = 0
    for i in range(n):
        s = occ[i]
        outp = os.path.join(args.out, s.sample_token + ".npy")
        if os.path.isfile(outp):
            skip += 1
            continue
        sample = nusc.get("sample", s.sample_token)
        imgs = []
        for cam in CAMS:
            sd = nusc.get("sample_data", sample["data"][cam])
            img = Image.open(os.path.join(nusc.dataroot, sd["filename"])).convert("RGB")
            imgs.append(np.asarray(img.resize((args.W, args.H)), np.float32) / 255.0)
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)[None].cuda()   # [1,6,3,H,W] in [0,1]
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            depth = vggt(x)["depth"][0, ..., 0].float()                          # (6,H,W)
        # block-min downsample to the (fH,fW) feature grid = nearest visible surface per block
        d = depth.view(6, fH, args.patch, fW, args.patch).amin(dim=(2, 4))       # (6,fH,fW)
        np.save(outp, d.cpu().numpy().astype(np.float16))
        done += 1
        if done % 100 == 0:
            print(f"  [{i+1}/{n}] cached={done} skip={skip}", flush=True)
    print(f"[cache] done: wrote {done}, skipped {skip} existing -> {args.out}")


if __name__ == "__main__":
    main()
