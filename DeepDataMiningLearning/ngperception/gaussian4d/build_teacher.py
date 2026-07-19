"""
gaussian4d.build_teacher
========================
Build (and cache) label-free occupancy **teacher targets** for the Phase-1 comparison. One CLI, one
`--teacher` flag to swap representations — the whole point is that voxel vs Gaussian is a clean swap.

    # sanity-check a teacher vs Occ3D GT on a few frames (agreement mIoU / geo-IoU):
    python -m DeepDataMiningLearning.ngperception.gaussian4d.build_teacher \
        --nusc <nuscenes> --gts <gts> --labelgen-cache <labelgen_cache> \
        --teacher gaussian --stats --n 20

    # cache targets for student training (any teacher):
    python -m ...gaussian4d.build_teacher --nusc <nuscenes> --gts <gts> \
        --labelgen-cache <cache> --teacher voxel10 --out-dir <teacher_cache/voxel10> --n 2000
"""
from __future__ import annotations
import argparse
import os
import numpy as np

from .teachers import build_teacher
from .teachers.base import CLASS_NAMES, TAIL_CLASSES, FREE
from ..occupancy.datasets import Occ3DNuScenesDataset
from ..occupancy.evaluator import OccupancyEvaluator


def main():
    ap = argparse.ArgumentParser(description="Build/cache label-free occupancy teacher targets.")
    ap.add_argument("--nusc", required=True); ap.add_argument("--gts", required=True)
    ap.add_argument("--labelgen-cache", required=True)
    ap.add_argument("--teacher", required=True, choices=["voxel1", "voxel10", "gaussian", "gaussian10"])
    ap.add_argument("--out-dir", default=None, help="cache <token>.npz targets here (skip = stats only)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--stats", action="store_true", help="report agreement vs Occ3D GT (no student needed)")
    args = ap.parse_args()

    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    teacher = build_teacher(args.teacher, nusc, args.labelgen_cache)
    occ = Occ3DNuScenesDataset(args.gts, scenes=None)
    # only tokens that have a labelgen cache
    items = [(sc, tok) for sc, tok, _ in occ.items
             if os.path.isfile(os.path.join(args.labelgen_cache, tok + ".npz"))][: args.n]
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    print(f"[teacher] {args.teacher} on {len(items)} frames"
          f"{' | caching -> ' + args.out_dir if args.out_dir else ''}"
          f"{' | vs Occ3D GT' if args.stats else ''}", flush=True)

    ev = OccupancyEvaluator() if args.stats else None
    gt_by_tok = {tok: (sc, tok, lp) for sc, tok, lp in occ.items}
    for i, (sc, tok) in enumerate(items):
        tgt = teacher(tok)
        if args.out_dir:
            tgt.save(os.path.join(args.out_dir, tok + ".npz"))
        if args.stats:
            g = np.load(gt_by_tok[tok][2])
            ev.add(tgt.semantics, g["semantics"].astype(np.uint8), g["mask_camera"].astype(bool))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(items)}] occupied~{tgt.occupied}", flush=True)

    if args.stats:
        s = ev.summarize(verbose=False)
        print(f"\n=== {args.teacher} vs Occ3D GT (camera-visible voxels) ===")
        print(f"  agreement mIoU = {s['mIoU']:.3f}   geo-IoU = {s['geo_IoU']:.3f}")
        tail = [s['per_class'][CLASS_NAMES[c]] for c in TAIL_CLASSES]
        print(f"  tail-class IoU (barrier/bike/moto/ped/cone/trailer) = {np.mean(tail):.3f}  "
              + " ".join(f"{CLASS_NAMES[c]}={s['per_class'][CLASS_NAMES[c]]:.2f}" for c in TAIL_CLASSES))
        print("  (this is teacher-vs-GT agreement, an upper bound on how well a student trained on it "
              "could score; the Phase-1 question is which teacher gives the best STUDENT.)")


if __name__ == "__main__":
    main()
