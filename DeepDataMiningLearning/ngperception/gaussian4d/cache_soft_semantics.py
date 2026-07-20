"""
gaussian4d.cache_soft_semantics
===============================
Cache per-pixel **top-k Occ3D class distributions** (SegFormer stuff + Grounded-SAM things, blended
by detection confidence) for the soft-semantics 2x2. Mirrors `ngdet/labelgen/run.py --save-npz` but
stores a distribution, not an argmax: `<token>.npz` with `idx (6,H,W,K)` uint8 + `prob (6,H,W,K)`
f16, in the labelgen camera order. Only re-run for tokens that already have a (hard) labelgen cache.

    python -m DeepDataMiningLearning.ngperception.gaussian4d.cache_soft_semantics \
        --nusc <nuscenes> --hard-cache <labelgen_cache> --out <labelgen_soft_cache> --topk 3
"""
from __future__ import annotations
import argparse
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusc", required=True)
    ap.add_argument("--hard-cache", required=True, help="only process tokens present here")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--num", type=int, default=2200, help="scan the first N keyframes (want-tokens are early)")
    ap.add_argument("--image-h", type=int, default=252); ap.add_argument("--image-w", type=int, default=700)
    ap.add_argument("--depth-ckpt", default="depth-anything/Depth-Anything-V2-Small-hf")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from ...ngdet.labelgen.labeler import GroundedLabeler, NUSC_TAXONOMY
    from ...ngdet.labelgen.sources import NuScenesSource
    lab = GroundedLabeler(taxonomy=NUSC_TAXONOMY, depth_ckpt=args.depth_ckpt)
    want = set(f[:-4] for f in os.listdir(args.hard_cache) if f.endswith(".npz"))
    src = NuScenesSource(args.nusc, image_hw=(args.image_h, args.image_w), num=args.num)
    print(f"[soft] {len(want)} tokens to cover | topk={args.topk} -> {args.out}", flush=True)

    done = skip = 0
    for i, (token, cams) in enumerate(src):
        if token not in want:
            continue
        outp = os.path.join(args.out, token + ".npz")
        if os.path.isfile(outp):
            skip += 1; continue
        idxs, probs = [], []
        for name, pil, uvz in cams:
            idx, pk = lab.semantic_soft(pil, topk=args.topk)      # (H,W,K)
            idxs.append(idx); probs.append(pk)
        np.savez_compressed(outp, idx=np.stack(idxs), prob=np.stack(probs))   # (6,H,W,K)
        done += 1
        if done % 100 == 0:
            print(f"  cached={done} skip={skip}", flush=True)
    print(f"[soft] done: wrote {done}, skipped {skip}", flush=True)


if __name__ == "__main__":
    main()
