"""
Create a scene-stratified subsample of the nuScenes train infos for fast
ablation experiments.

Why scene-stratified:
- nuScenes is divided into ~700 train scenes with ~40 samples each
- Random per-sample subsampling breaks within-scene temporal context
- Per-scene subsampling preserves full sequences and dataset diversity

Usage:
    cd /data/rnd-liu/MyRepo/mmdetection3d
    conda run -n py310 python projects/bevdet/subsample_nuscenes.py \
        --input  data/nuscenes/nuscenes_infos_train.pkl \
        --output data/nuscenes/nuscenes_infos_train_25pct.pkl \
        --fraction 0.25 \
        --seed 42

Result: a new .pkl file with the same schema, containing samples from
~25% of train scenes (selected uniformly at random).
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import re
from collections import defaultdict


# nuScenes lidar paths look like:
#   n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin
# The scene tag is everything before the second "__" (i.e., the recording session).
SCENE_RE = re.compile(r"^([^/]+)__LIDAR_TOP__")


def scene_tag(lidar_path: str) -> str:
    """Extract a scene identifier from a lidar_path; falls back to dir name."""
    base = os.path.basename(lidar_path or "")
    m = SCENE_RE.match(base)
    if m:
        return m.group(1)
    # Fallback: use dir name + first 30 chars of basename
    return os.path.dirname(lidar_path or "") + "/" + base[:30]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Input nuscenes_infos_train.pkl")
    p.add_argument("--output", required=True, help="Output subsampled pkl")
    p.add_argument("--fraction", type=float, default=0.25,
                   help="Fraction of scenes to keep (default 0.25)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"[subsample] Loading {args.input} ...")
    with open(args.input, "rb") as f:
        d = pickle.load(f)
    if not (isinstance(d, dict) and "data_list" in d):
        raise ValueError("Expected dict with 'data_list' key (mmdet3d-style).")

    samples = d["data_list"]
    print(f"[subsample] Loaded {len(samples)} samples.")

    # Group by scene
    scenes = defaultdict(list)
    for i, s in enumerate(samples):
        lp = s.get("lidar_points", {}).get("lidar_path", "")
        scenes[scene_tag(lp)].append(i)

    n_scenes = len(scenes)
    print(f"[subsample] Found {n_scenes} unique scenes.")

    # Sample scenes deterministically
    rng = random.Random(args.seed)
    keep_n = max(1, int(round(n_scenes * args.fraction)))
    all_scene_ids = sorted(scenes.keys())
    rng.shuffle(all_scene_ids)
    keep_scenes = set(all_scene_ids[:keep_n])

    # Collect kept sample indices, in original order
    kept_indices = []
    for sc in sorted(keep_scenes):
        kept_indices.extend(scenes[sc])
    kept_indices.sort()

    new_samples = [samples[i] for i in kept_indices]
    out = dict(d)
    out["data_list"] = new_samples
    # Reset metainfo dataset_size if present
    mi = out.get("metainfo", {})
    if isinstance(mi, dict):
        mi.pop("dataset_size", None)

    print(f"[subsample] Keeping {keep_n}/{n_scenes} scenes "
          f"({100.0 * keep_n / n_scenes:.1f}%) "
          f"= {len(new_samples)}/{len(samples)} samples "
          f"({100.0 * len(new_samples) / len(samples):.1f}%)")

    print(f"[subsample] Writing {args.output} ...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[subsample] Done. Output size: "
          f"{os.path.getsize(args.output) / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
