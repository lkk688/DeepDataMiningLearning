"""
Parameter count analyzer for BEVFusion variants.

Builds each model on CPU (no GPU conflict with running training) and prints
per-component and total parameter counts. Used to verify that our memory
optimizations don't come at the cost of higher model complexity.

Usage:
    cd /data/rnd-liu/MyRepo/mmdetection3d
    conda run -n py310 python projects/bevdet/analyze_params.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# Bootstrap import of project modules so MODELS registry is populated
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECTS = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _PROJECTS not in sys.path:
    sys.path.insert(0, _PROJECTS)

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.utils import register_all_modules
from mmdet3d.registry import MODELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_params(module: nn.Module) -> Tuple[int, int]:
    """Return (total, trainable) parameter counts for a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def fmt_M(n: int) -> str:
    return f"{n / 1e6:.2f} M"


def analyze_model(name: str, cfg_path: str) -> Dict:
    """Build a model from config (on CPU) and break down its parameter count."""
    print(f"\n{'='*78}\n  {name}\n  config: {cfg_path}\n{'='*78}")

    cfg = Config.fromfile(cfg_path)
    cfg.setdefault("default_scope", "mmdet3d")

    # Build the inner detector only (skip data_preprocessor's voxelizer side
    # effects on CUDA by keeping everything on CPU).
    register_all_modules(init_default_scope=False)
    init_default_scope("mmdet3d")
    model = MODELS.build(cfg.model)
    model.eval()  # disables dropout etc. – not strictly needed for counting

    # Top-level breakdown: walk named children
    rows: List[Tuple[str, int, int]] = []
    seen_total = 0
    seen_trainable = 0
    for child_name, child in model.named_children():
        t, tr = count_params(child)
        rows.append((child_name, t, tr))
        seen_total += t
        seen_trainable += tr

    # Account for any params not under named_children (rare)
    all_total, all_trainable = count_params(model)
    other_total = all_total - seen_total
    other_trainable = all_trainable - seen_trainable
    if other_total > 0:
        rows.append(("(other / direct)", other_total, other_trainable))

    # Pretty print
    print(f"{'component':<28} {'#params':>14} {'#trainable':>14} {'%':>7}")
    print("-" * 72)
    for nm, t, tr in rows:
        pct = 100.0 * t / max(all_total, 1)
        print(f"{nm:<28} {fmt_M(t):>14} {fmt_M(tr):>14} {pct:>6.1f}%")
    print("-" * 72)
    print(f"{'TOTAL':<28} {fmt_M(all_total):>14} {fmt_M(all_trainable):>14}  100.0%")

    return {
        "name": name,
        "config": cfg_path,
        "total": all_total,
        "trainable": all_trainable,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

def print_comparison(results: List[Dict]) -> None:
    """Side-by-side total params across all variants, rounded to millions."""
    print("\n\n" + "=" * 78)
    print(" SUMMARY (all numbers in millions of parameters)")
    print("=" * 78)
    header = f"{'component':<28}" + "".join(f"{r['name']:>16}" for r in results)
    print(header)
    print("-" * len(header))

    # Aggregate per-component across results
    component_names: List[str] = []
    seen = set()
    for r in results:
        for nm, _, _ in r["rows"]:
            if nm not in seen:
                seen.add(nm)
                component_names.append(nm)

    for nm in component_names:
        row = f"{nm:<28}"
        for r in results:
            val = "—"
            for n2, t, _ in r["rows"]:
                if n2 == nm:
                    val = f"{t / 1e6:.2f}"
                    break
            row += f"{val:>16}"
        print(row)

    print("-" * len(header))
    row = f"{'TOTAL':<28}"
    for r in results:
        row += f"{r['total'] / 1e6:>16.2f}"
    print(row)
    print()

    # Relative to first variant
    base = results[0]["total"]
    print(f"\nRelative to {results[0]['name']}:")
    for r in results:
        delta = (r["total"] - base) / base * 100.0
        sign = "+" if delta >= 0 else ""
        print(f"  {r['name']:<35}  {fmt_M(r['total']):>10}  ({sign}{delta:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONFIGS = [
    ("Original BEVFusion (LSS+SECFPN)",
     "projects/bevdet/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"),
    ("Phase 1a (mybevfusion12v2)",
     "projects/bevdet/configs/mybevfusion12v2.py"),
    ("Phase 1b/1c (mybevfusion_phase1b)",
     "projects/bevdet/configs/mybevfusion_phase1b.py"),
]


def main():
    # mmdetection3d/ root: _HERE is .../mmdetection3d/projects/bevdet/
    mmdet3d_root = os.path.normpath(os.path.join(_HERE, "..", ".."))
    os.chdir(mmdet3d_root)

    # Single-config mode: pass a config path as argv[1] (and optional name as argv[2])
    if len(sys.argv) >= 2:
        cfg_path = sys.argv[1]
        name = sys.argv[2] if len(sys.argv) >= 3 else os.path.basename(cfg_path)
        print(f"[analyze_params] cwd={mmdet3d_root}")
        analyze_model(name, cfg_path)
        return

    # Multi-config mode (only safe if configs do not have conflicting registrations).
    print(f"[analyze_params] cwd={mmdet3d_root}")
    print("[analyze_params] Building models on CPU (no GPU conflict).")
    results = []
    for name, cfg_path in CONFIGS:
        if not os.path.isfile(cfg_path):
            print(f"\n[Warn] Skipping {name}: config not found at {cfg_path}")
            continue
        try:
            results.append(analyze_model(name, cfg_path))
        except Exception as e:
            print(f"\n[Error] Failed to build {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    if results:
        print_comparison(results)


if __name__ == "__main__":
    main()
