"""
ngperception.occupancy.run_eval
===============================

Run the depth->occupancy baseline over an Occ3D-nuScenes subset and score it (mIoU +
geometric IoU). Optionally compare against a **LiDAR oracle** (project the real LiDAR
sweep) — the geometric ceiling a *single-shot* lift can reach against the densified GT.

Example
-------
python -m DeepDataMiningLearning.ngperception.occupancy.run_eval \
    --gts /path/to/extracted/gts \
    --nusc-root /mnt/e/Shared/Dataset/NuScenes/v1.0-trainval \
    --depth hf_depth:depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf \
    --seg nvidia/segformer-b2-finetuned-cityscapes-1024-1024 \
    --max-samples 30 --oracle
"""

from __future__ import annotations
import argparse
import os

import numpy as np

from .datasets import Occ3DNuScenesDataset
from .evaluator import OccupancyEvaluator
from .geom import voxelize, FREE, GRID_SIZE


def _lidar_grid(nusc, token):
    """Geometric oracle: project the real LIDAR_TOP sweep into the grid (ego frame)."""
    from pyquaternion import Quaternion
    s = nusc.get("sample", token)
    sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
    pts = np.fromfile(os.path.join(nusc.dataroot, sd["filename"]),
                      dtype=np.float32).reshape(-1, 5)[:, :3]
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    R = Quaternion(cs["rotation"]).rotation_matrix
    t = np.array(cs["translation"])
    return voxelize(pts @ R.T + t)             # geometry only (class 0)


def main():
    ap = argparse.ArgumentParser(description="Occ3D depth->occupancy baseline eval.")
    ap.add_argument("--gts", required=True, help="extracted Occ3D gts/ directory")
    ap.add_argument("--nusc-root", default="/mnt/e/Shared/Dataset/NuScenes/v1.0-trainval")
    ap.add_argument("--nusc-version", default="v1.0-trainval")
    ap.add_argument("--depth", default="hf_depth:depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    ap.add_argument("--seg", default=None,
                    help="HF semantic-seg model (Cityscapes) for mIoU; omit for geometry-only")
    ap.add_argument("--max-samples", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2, help="pixel subsample for back-projection")
    ap.add_argument("--oracle", action="store_true", help="also score the LiDAR geometric oracle")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from nuscenes import NuScenes
    from .predictors.depth_lift import DepthLiftOccupancy

    ds = Occ3DNuScenesDataset(args.gts, max_samples=args.max_samples)
    print(f"[occ] {len(ds)} Occ3D samples")
    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_root, verbose=False)
    pred = DepthLiftOccupancy(nusc, args.depth, seg_model=args.seg,
                              device=args.device, stride=args.stride)

    ev = OccupancyEvaluator()
    ev_or = OccupancyEvaluator() if args.oracle else None
    for i in range(len(ds)):
        s = ds[i]
        ev.add(pred.predict(s.sample_token), s.semantics, s.mask_camera)
        if ev_or is not None:
            ev_or.add(_lidar_grid(nusc, s.sample_token), s.semantics, s.mask_camera)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(ds)}")

    tag = "depth+seg" if args.seg else "depth (geometry)"
    print(f"\n=== depth->occupancy baseline [{tag}] ===")
    ev.summarize()
    if ev_or is not None:
        print("=== LiDAR oracle (single-sweep geometry ceiling) ===")
        ev_or.summarize()
    print("=== SOTA target (learned occupancy nets) ===")
    from .evaluator import print_sota_reference
    print_sota_reference()


if __name__ == "__main__":
    main()
