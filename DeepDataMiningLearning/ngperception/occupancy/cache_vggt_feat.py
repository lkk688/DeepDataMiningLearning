"""
ngperception.occupancy.cache_vggt_feat
=======================================
Precompute **frozen VGGT patch features** (the *features* backbone lever, doc §7) — distinct from
`cache_vggt_depth.py` (which cached VGGT's *depth*). Ablations #2/#3 showed the VGGT depth *prior*
is a dead end; this tests a *different* VGGT signal: its 2048-d 3-D-aware patch tokens as a drop-in
replacement for the DINOv2 backbone in the LSS lift (a learned depth head still predicts depth).

For each sample we run `facebook/VGGT-1B` on the 6 cameras at 252x700 (÷14 -> 18x50 patch grid,
identical to the DINOv2 grid), take the last-layer aggregated patch tokens (2048-d), and save
`<token>.npy` (6, 2048, 18, 50) float16. `train_lss.py --vggt-feat-cache DIR --backbone vggt` then
feeds these to the DepthNet in place of DINOv2 features.

    python -m DeepDataMiningLearning.ngperception.occupancy.cache_vggt_feat \
        --gts <gts> --nusc <nuscenes> --out <cache_dir> --cap 2100
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
    ap = argparse.ArgumentParser(description="Precompute frozen-VGGT patch features for the LSS lift.")
    ap.add_argument("--gts", required=True)
    ap.add_argument("--nusc", required=True)
    ap.add_argument("--out", required=True, help="cache dir for <token>.npy (6,2048,fH,fW)")
    ap.add_argument("--cap", type=int, default=2100)
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
    print("[cache-feat] loading facebook/VGGT-1B ...", flush=True)
    vggt = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
    for p in vggt.parameters():
        p.requires_grad = False

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    occ = Occ3DNuScenesDataset(args.gts, scenes=None)
    n = min(args.cap, len(occ))
    print(f"[cache-feat] {n} samples -> {args.out} | grid {fH}x{fW} x2048", flush=True)

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
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)[None].cuda()   # [1,6,3,H,W]
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            toks, psi = vggt.aggregator(x)                    # list; last layer (1,6,P,2048)
        patch = toks[-1][0, :, psi:, :].float()               # (6, fH*fW, 2048)
        feat = patch.transpose(1, 2).reshape(6, 2048, fH, fW) # (6,2048,fH,fW)
        np.save(outp, feat.cpu().numpy().astype(np.float16))
        done += 1
        if done % 100 == 0:
            print(f"  [{i+1}/{n}] cached={done} skip={skip}", flush=True)
    print(f"[cache-feat] done: wrote {done}, skipped {skip} -> {args.out}")


if __name__ == "__main__":
    main()
