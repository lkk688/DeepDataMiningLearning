"""
gaussian4d.build_dynamic
=======================
Cache DynamicOccTeacher pseudo-label targets (labels.npz) for occ pretraining — the Step-0 arm that
adds dynamic/static separation to the label-free pipeline. Same tokens as the voxel-soft student
(2044) so the det-transfer comparison is apples-to-apples. Resumable.

    python -m DeepDataMiningLearning.ngperception.gaussian4d.build_dynamic \
        --nusc <nusc>/v1.0-trainval --gts <nusc>/v1.0-trainval/gts \
        --labelgen-cache <labelgen_cache> --boxes gt --out-dir <teacher_cache_dynamic> --n 2044
"""
from __future__ import annotations
import argparse, os
import numpy as np

from .teachers.dynamic_teacher import DynamicOccTeacher
from ..occupancy.datasets import Occ3DNuScenesDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusc", required=True); ap.add_argument("--gts", required=True)
    ap.add_argument("--labelgen-cache", required=True)
    ap.add_argument("--boxes", default="gt", choices=["gt"])   # label-free pseudo-track hook: later
    ap.add_argument("--sweeps", type=int, default=10)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--n", type=int, default=2044)
    args = ap.parse_args()
    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    teacher = DynamicOccTeacher(nusc, args.labelgen_cache, sweeps=args.sweeps, boxes=args.boxes)
    occ = Occ3DNuScenesDataset(args.gts, scenes=None)
    items = [(sc, tok) for sc, tok, _ in occ.items
             if os.path.isfile(os.path.join(args.labelgen_cache, tok + ".npz"))][: args.n]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[dynamic] {len(items)} frames (boxes={args.boxes}, sweeps={args.sweeps}) -> {args.out_dir}", flush=True)
    done = skip = 0
    for i, (sc, tok) in enumerate(items):
        outp = os.path.join(args.out_dir, tok + ".npz")
        if os.path.isfile(outp):
            skip += 1; continue
        teacher(tok).save(outp); done += 1
        if done % 100 == 0:
            print(f"  cached={done} skip={skip} ({i+1}/{len(items)})", flush=True)
    print(f"[dynamic] done: wrote {done}, skipped {skip}", flush=True)


if __name__ == "__main__":
    main()
