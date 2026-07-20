"""Audit the hard-vs-soft Gaussian teacher for occupancy/semantic COUPLING confounds.
Answers: (1) are occupied/free/unknown masks identical hard vs soft? (2) is the soft per-voxel
distribution normalised? (3) does soft change tot_mass / free mask / occupancy weight?"""
from __future__ import annotations
import argparse, os
import numpy as np
from .teachers.gaussian_teacher import GaussianTeacher
from .teachers.base import FREE


def masks(t):
    occ = (t.semantics != FREE)
    free = (t.semantics == FREE) & (t.weight > 0)
    unk = (t.weight == 0) & (t.semantics == FREE)
    return occ, free, unk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusc", required=True); ap.add_argument("--gts", required=True)
    ap.add_argument("--labelgen-cache", required=True); ap.add_argument("--soft-cache", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    from nuscenes import NuScenes
    from ..occupancy.datasets import Occ3DNuScenesDataset
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    occ = Occ3DNuScenesDataset(args.gts, scenes=None)
    toks = [tok for _, tok, _ in occ.items
            if os.path.isfile(os.path.join(args.labelgen_cache, tok + ".npz"))
            and os.path.isfile(os.path.join(args.soft_cache, tok + ".npz"))][: args.n]
    hard = GaussianTeacher(nusc, args.labelgen_cache, sweeps=10, soft_cache=None)
    soft = GaussianTeacher(nusc, args.labelgen_cache, sweeps=10, soft_cache=args.soft_cache)

    d_occ = d_free = d_unk = 0; n_h = n_s = 0; wdiff = 0.0; nvox = 0
    minprob = 1e9; maxprob = -1e9; badnorm = 0
    for tok in toks:
        th, ts = hard(tok), soft(tok)
        oh, fh, uh = masks(th); os_, fs, us = masks(ts)
        d_occ += int((oh != os_).sum()); d_free += int((fh != fs).sum()); d_unk += int((uh != us).sum())
        n_h += int(oh.sum()); n_s += int(os_.sum()); nvox += oh.size
        wdiff += float(np.abs(th.weight - ts.weight).sum())
        # soft distribution normalisation on occupied voxels
        sp = ts.soft_prob[os_]                    # (n_occ, K)
        s = sp.sum(1)
        minprob = min(minprob, float(s.min())); maxprob = max(maxprob, float(s.max()))
        badnorm += int((np.abs(s - 1.0) > 1e-3).sum())
    print(f"== Gaussian hard-vs-soft audit ({len(toks)} frames) ==")
    print(f"occupied voxels: hard={n_h} soft={n_s}  (soft/hard={n_s/max(n_h,1):.3f})")
    print(f"MASK DIFFERENCES hard vs soft: occ={d_occ} free={d_free} unknown={d_unk}  "
          f"(should be 0 if occupancy is geometry-only)")
    print(f"occupancy WEIGHT L1 diff (hard vs soft): {wdiff:.1f} over {nvox} voxels  "
          f"(should be 0 if weight is geometry-only)")
    print(f"soft top-K prob sum on occupied: min={minprob:.4f} max={maxprob:.4f}  "
          f"non-normalised voxels={badnorm} (should be ~1.0 / 0)")


if __name__ == "__main__":
    main()
