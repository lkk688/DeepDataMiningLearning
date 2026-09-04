"""
ngperception.occupancy.eval_depth_prior
=======================================
Measure a cached **depth-FM prior** directly against LiDAR — no training required.

This answers the first half of the depth-FM arm: *is the prior any good in our setting, and
how does it degrade as the LiDAR gets sparser?*  (The second half — does a good prior actually
help the trained lift — needs the occ A/B, because our VGGT result showed a good prior can
still be redundant with depth supervision.)

Compares each cache dir on the SAME feature grid the lift uses, over pixels where LiDAR has a
return, reporting standard depth metrics: AbsRel, RMSE, delta<1.25.

    python -m DeepDataMiningLearning.ngperception.occupancy.eval_depth_prior \
        --gts <gts> --nusc <nusc> --n 200 \
        --cache full32=<dir> beam8=<dir> monoonly=<dir>
"""
from __future__ import annotations
import argparse
import os
import numpy as np

from .geom import CAMS
from .datasets import Occ3DNuScenesDataset
from .cache_ldcm_depth import sparse_lidar_depth


def block_min_nonzero(dm, fH, fW, patch):
    """Sparse (H,W) depth -> (fH,fW) nearest-return per patch (0 where the patch has none)."""
    b = dm.reshape(fH, patch, fW, patch)
    b = np.where(b > 0, b, np.inf).min(axis=(1, 3))
    return np.where(np.isfinite(b), b, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gts", required=True)
    ap.add_argument("--nusc", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--H", type=int, default=252)
    ap.add_argument("--W", type=int, default=700)
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--dmax", type=float, default=58.0, help="lift's far plane; ignore beyond")
    ap.add_argument("--holdout-beams", type=int, default=0,
                    help="evaluate ONLY where the 32-beam GT has a return but this N-ring subset "
                         "does NOT. Without it, a prior that CONSUMES the 32-beam sparse depth is "
                         "scored on its own input pixels (self-reconstruction). Set this to the "
                         "sparsest arm's input (e.g. 8) so every arm is scored on the SAME "
                         "genuinely-held-out pixels.")
    ap.add_argument("--median-align", action="store_true",
                    help="per-sample median-scale each prior to the GT before scoring. Relative-depth "
                         "priors (MoGe alone) are up-to-scale, so raw metric scoring says only 'not "
                         "metric'. Aligned scoring isolates depth *structure* quality. Report both: "
                         "raw = metric grounding, aligned = shape.")
    ap.add_argument("--cache", nargs="+", required=True, help="name=dir pairs")
    args = ap.parse_args()

    from nuscenes import NuScenes
    from pyquaternion import Quaternion as Q
    fH, fW = args.H // args.patch, args.W // args.patch
    caches = dict(c.split("=", 1) for c in args.cache)
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    occ = Occ3DNuScenesDataset(args.gts, scenes=None)

    acc = {k: {"absrel": [], "rmse": [], "d125": [], "npix": 0} for k in caches}
    used = 0
    for i in range(min(args.n, len(occ))):
        s = occ[i]
        if not all(os.path.isfile(os.path.join(d, s.sample_token + ".npy")) for d in caches.values()):
            continue
        sample = nusc.get("sample", s.sample_token)
        # GT: full-32-beam LiDAR, the densest reference we have
        gts = []
        for cam in CAMS:
            sd = nusc.get("sample_data", sample["data"][cam])
            dm = sparse_lidar_depth(nusc, sample, sd, Q, args.H, args.W, beams=0)
            gts.append(block_min_nonzero(dm, fH, fW, args.patch))
        gt = np.stack(gts)                                            # (6,fH,fW)
        m = (gt > 1.0) & (gt < args.dmax)
        if args.holdout_beams:                                        # drop pixels the input saw
            seen = []
            for cam in CAMS:
                sd = nusc.get("sample_data", sample["data"][cam])
                dm = sparse_lidar_depth(nusc, sample, sd, Q, args.H, args.W, beams=args.holdout_beams)
                seen.append(block_min_nonzero(dm, fH, fW, args.patch))
            m &= ~(np.stack(seen) > 0)
        if m.sum() == 0:
            continue
        for k, d in caches.items():
            pr = np.load(os.path.join(d, s.sample_token + ".npy")).astype(np.float32)
            p, g = pr[m], gt[m]
            p = np.clip(p, 1e-3, None)
            if args.median_align:                                 # up-to-scale -> fit one scalar
                p = p * (np.median(g) / max(np.median(p), 1e-6))
            acc[k]["absrel"].append(np.abs(p - g) / g)
            acc[k]["rmse"].append((p - g) ** 2)
            acc[k]["d125"].append(np.maximum(p / g, g / p) < 1.25)
            acc[k]["npix"] += int(m.sum())
        used += 1
    ho = (f", HELD OUT from the {args.holdout_beams}-ring input" if args.holdout_beams
          else " (NOTE: includes pixels a 32-beam-conditioned prior saw as input)")
    print(f"[depth-prior] evaluated {used} samples on the {fH}x{fW} lift grid, "
          f"vs 32-beam LiDAR returns (<{args.dmax:g} m){ho}\n")
    print(f"{'prior':<12} {'AbsRel↓':>9} {'RMSE(m)↓':>10} {'δ<1.25↑':>9} {'pixels':>10}")
    for k in caches:
        a = np.concatenate(acc[k]["absrel"]); r = np.concatenate(acc[k]["rmse"])
        d = np.concatenate(acc[k]["d125"])
        print(f"{k:<12} {a.mean():>9.4f} {np.sqrt(r.mean()):>10.3f} {d.mean():>9.4f} {acc[k]['npix']:>10d}")


if __name__ == "__main__":
    main()
