"""
Per-range nuScenes AP analyzer for our ablation chain.

Hypothesis: 3D occupancy supervision (B6 = + Q-Occ) most directly helps
near-range object detection where 3D structure matters; mAP averaged over
all ranges (NDS) hides this signal.

Method:
  1. For each row, run nuScenes' DetectionEval with custom `class_range` to
     bucket evaluation by ego distance:
       - Near    : class_range = 15  (only objects ≤15 m from ego)
       - Mid     : class_range = 30  (only objects ≤30 m)
       - Default : class_range = nuScenes standard (max ~50 m)
  2. Compare (B5 vs B6) across these buckets per class. If Q-Occ helps near-
     range, B6's near-range AP > B5's; far-range AP is similar.

Inputs (per row): predictions JSON written by run_eval.sh's eval_runner with
                  the persistent jsonfile_prefix patch.

Usage:
    cd /data/rnd-liu/MyRepo/mmdetection3d
    conda run -n py310 python projects/bevdet/per_range_eval.py \\
        --pred  work_dirs/ablation_B5/eval_epoch_3/predictions/results_nusc.json \\
        --pred  work_dirs/ablation_B6/eval_epoch_3/predictions/results_nusc.json \\
        --names B5 B6 \\
        --data-root data/nuscenes \\
        --version v1.0-trainval \\
        --eval-set val
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

# nuScenes devkit
try:
    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.detection.config import config_factory
except ImportError as e:
    print("[per_range_eval] nuScenes devkit not installed: pip install nuscenes-devkit")
    raise


CLASS_NAMES = (
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)

# ------------------------------------------------------------------
# Custom DetectionConfig: override class_range for range-bucket eval
# ------------------------------------------------------------------

def make_config_for_range(max_range: float):
    """
    Returns a DetectionConfig where every class has class_range = max_range.
    Setting class_range causes nuScenes eval to ignore both predictions and GTs
    farther than max_range from the ego, effectively bucketing by range.
    """
    cfg = config_factory("detection_cvpr_2019")
    # Replace class_range for every class
    new_class_range = {c: float(max_range) for c in CLASS_NAMES}
    # The DetectionConfig dataclass exposes class_range as dict — mutate in place.
    cfg.class_range = new_class_range
    return cfg


# ------------------------------------------------------------------
# Run a single per-range eval
# ------------------------------------------------------------------

def run_one_eval(
    pred_path: str,
    nusc: NuScenes,
    eval_set: str,
    output_dir: str,
    max_range: float,
) -> Dict:
    cfg = make_config_for_range(max_range)
    os.makedirs(output_dir, exist_ok=True)
    nusc_eval = DetectionEval(
        nusc,
        config=cfg,
        result_path=pred_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=False,
    )
    metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)
    # metrics_summary is a dict with mean_ap, nd_score, mean_dist_aps, label_aps, etc.
    return metrics_summary


# ------------------------------------------------------------------
# Pretty-print comparison table
# ------------------------------------------------------------------

def _ap(metrics, key="mean_ap"):
    return metrics.get(key, float("nan"))


def _per_class_ap(metrics, dist_th=2.0):
    """Returns {class: AP at given dist threshold}"""
    label_aps = metrics.get("label_aps", {})
    out = {}
    for c in CLASS_NAMES:
        per_th = label_aps.get(c, {})
        # per_th maps str(distance) -> AP
        if isinstance(per_th, dict):
            v = per_th.get(str(dist_th), per_th.get(dist_th, None))
            if v is None and per_th:
                # average across thresholds if specific not found
                v = sum(per_th.values()) / len(per_th)
            out[c] = v if v is not None else float("nan")
        else:
            out[c] = float("nan")
    return out


def print_compare_table(results: Dict[str, Dict[float, Dict]], names: List[str], ranges: List[float]):
    """
    results[name][max_range] = nuScenes metrics_summary dict
    """
    print("\n" + "=" * 92)
    print(f"{'Per-range comparison (NDS / mAP per max_range)':<92}")
    print("=" * 92)
    header = f"{'Variant':<8}" + "".join(f"  {'≤'+str(int(r))+'m':<14}" for r in ranges)
    print(header)
    print("-" * 92)
    for name in names:
        row = f"{name:<8}"
        for r in ranges:
            m = results[name].get(r)
            if m is None:
                row += f"  {'(missing)':<14}"
            else:
                nds = m.get("nd_score", float("nan"))
                mAP = m.get("mean_ap", float("nan"))
                row += f"  NDS={nds:.4f} mAP={mAP:.4f}".ljust(16)
        print(row)
    print()

    print("=" * 92)
    print("Per-class AP (dist=2.0 m) by range bucket — useful for B6 / Q-Occ near-range hypothesis")
    print("=" * 92)
    for r in ranges:
        print(f"\n--- max_range = {r} m ---")
        hdr = f"{'class':<22}" + "".join(f"  {n:<10}" for n in names) + (f"  {'Δ('+names[-1]+'-'+names[0]+')':<14}" if len(names) >= 2 else "")
        print(hdr)
        print("-" * len(hdr))
        for c in CLASS_NAMES:
            row = f"{c:<22}"
            aps = []
            for n in names:
                ap_dict = _per_class_ap(results[n].get(r, {}))
                ap = ap_dict.get(c, float("nan"))
                aps.append(ap)
                row += f"  {ap:<10.4f}"
            if len(names) >= 2:
                delta = aps[-1] - aps[0]
                row += f"  {delta:+.4f}"
            print(row)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", action="append", required=True,
                   help="Path to results_nusc.json. Can repeat for multiple runs.")
    p.add_argument("--names", nargs="+", required=True,
                   help="Display name for each --pred (same order).")
    p.add_argument("--data-root", default="data/nuscenes",
                   help="nuScenes dataset root.")
    p.add_argument("--version", default="v1.0-trainval",
                   help="nuScenes version.")
    p.add_argument("--eval-set", default="val", choices=["train", "val", "test"])
    p.add_argument("--ranges", type=float, nargs="+", default=[15.0, 30.0, 50.0],
                   help="Per-class max ranges (meters) to evaluate at.")
    p.add_argument("--out-dir", default="work_dirs/per_range_eval",
                   help="Where to write per-range eval outputs.")
    args = p.parse_args()

    if len(args.pred) != len(args.names):
        raise ValueError("Number of --pred and --names must match.")

    print(f"[per_range_eval] Loading nuScenes {args.version} from {args.data_root} ...")
    t0 = time.time()
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    print(f"[per_range_eval] Loaded in {time.time()-t0:.1f}s.")

    results: Dict[str, Dict[float, Dict]] = {n: {} for n in args.names}
    for pred_path, name in zip(args.pred, args.names):
        print(f"\n[per_range_eval] === {name} === pred={pred_path}")
        if not os.path.isfile(pred_path):
            print(f"  [Warn] missing predictions, skipping")
            continue
        for r in args.ranges:
            sub_out = os.path.join(args.out_dir, name, f"range_{int(r)}m")
            print(f"  [{name}] eval @ ≤{r}m → {sub_out}")
            t0 = time.time()
            try:
                m = run_one_eval(pred_path, nusc, args.eval_set, sub_out, r)
                results[name][r] = m
                print(f"    NDS={m.get('nd_score', float('nan')):.4f}  mAP={m.get('mean_ap', float('nan')):.4f}  ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [Error] {type(e).__name__}: {e}")
                results[name][r] = {}

    print_compare_table(results, args.names, args.ranges)

    # Save consolidated results
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({n: {str(r): results[n][r] for r in args.ranges if r in results[n]}
                   for n in args.names}, f, indent=2, default=str)
    print(f"\n[per_range_eval] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
