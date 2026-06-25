"""
ngdet.report
===========

Turn a list of per-(model, dataset) metrics dicts (as produced by run_eval) into
human-friendly artifacts:

    * a Markdown report   -> report.md
        - a model x dataset summary matrix (mAP / AP50)
        - a generalization-gap section (same-domain COCO minus cross-domain)
        - the FULL 12-row COCO table for every run
    * a heatmap PNG       -> heatmap_mAP.png
        - rows = models, cols = datasets, cell = mAP (the "generalization heatmap")

Kept separate from evaluation so it has no torch/cuda dependency -- only
matplotlib -- and can be re-run on a saved metrics.json.
"""

from __future__ import annotations
from typing import Dict, List
import json
import os

from .evaluator import format_coco_table


def _matrix(all_metrics: List[Dict], key: str):
    """Build ordered (models, datasets, value-grid) for a given metric key."""
    models, datasets = [], []
    for m in all_metrics:
        mo, ds = m["_meta"]["model"], m["_meta"]["dataset"]
        if mo not in models:
            models.append(mo)
        if ds not in datasets:
            datasets.append(ds)
    grid = {(m["_meta"]["model"], m["_meta"]["dataset"]): m.get(key, float("nan"))
            for m in all_metrics}
    return models, datasets, grid


def write_markdown_report(all_metrics: List[Dict], out_path: str, taxonomy) -> str:
    """Write report.md and return its path."""
    models, datasets, grid = _matrix(all_metrics, "mAP")
    _, _, grid50 = _matrix(all_metrics, "AP50")

    L = []
    L.append(f"# ngdet evaluation report\n")
    L.append(f"Unified taxonomy: **{taxonomy.name}** = `{taxonomy.classes}`\n")

    # --- summary matrix (mAP) ---
    L.append("## Summary matrix — mAP @[.5:.95]\n")
    L.append("| model \\ dataset | " + " | ".join(datasets) + " |")
    L.append("|---" * (len(datasets) + 1) + "|")
    for mo in models:
        row = [f"{grid.get((mo, ds), float('nan')):.3f}" for ds in datasets]
        L.append(f"| {mo} | " + " | ".join(row) + " |")
    L.append("")

    # --- summary matrix (AP50) ---
    L.append("## Summary matrix — AP50\n")
    L.append("| model \\ dataset | " + " | ".join(datasets) + " |")
    L.append("|---" * (len(datasets) + 1) + "|")
    for mo in models:
        row = [f"{grid50.get((mo, ds), float('nan')):.3f}" for ds in datasets]
        L.append(f"| {mo} | " + " | ".join(row) + " |")
    L.append("")

    # --- generalization gap (if COCO present as same-domain reference) ---
    if "coco" in datasets:
        L.append("## Generalization gap (COCO mAP − driving-dataset mAP)\n")
        L.append("Positive = drop when moving from the COCO training domain to a "
                 "driving domain (higher = worse generalization).\n")
        L.append("| model | " + " | ".join(d for d in datasets if d != "coco") + " |")
        L.append("|---" * (len([d for d in datasets if d != 'coco']) + 1) + "|")
        for mo in models:
            base = grid.get((mo, "coco"), float("nan"))
            row = [f"{base - grid.get((mo, ds), float('nan')):+.3f}"
                   for ds in datasets if ds != "coco"]
            L.append(f"| {mo} | " + " | ".join(row) + " |")
        L.append("")

    # --- full COCO table per run ---
    L.append("## Full COCO tables (per run)\n")
    for m in all_metrics:
        meta = m["_meta"]
        L.append(f"### {meta['model']}  ×  {meta['dataset']}  "
                 f"({meta['num_images']} imgs, {meta['seconds']}s, "
                 f"open_vocab={meta['open_vocab']})\n")
        L.append(format_coco_table(m))
        L.append("")

    text = "\n".join(L)
    with open(out_path, "w") as f:
        f.write(text)
    return out_path


def plot_heatmap(all_metrics: List[Dict], out_path: str, key: str = "mAP",
                 title: str = None) -> str:
    """Render a model x dataset heatmap PNG for the given metric and return path."""
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import numpy as np

    models, datasets, grid = _matrix(all_metrics, key)
    mat = np.array([[grid.get((mo, ds), np.nan) for ds in datasets]
                    for mo in models], dtype=float)

    fig, ax = plt.subplots(figsize=(1.6 * len(datasets) + 3, 0.9 * len(models) + 2))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0,
                   vmax=float(np.nanmax(mat)) if np.isfinite(mat).any() else 1.0)
    ax.set_xticks(range(len(datasets)), labels=datasets)
    ax.set_yticks(range(len(models)), labels=models)
    ax.set_xlabel("dataset (test domain)")
    ax.set_ylabel("model")
    ax.set_title(title or f"Zero-shot generalization heatmap ({key})")
    # annotate each cell with its value
    for i in range(len(models)):
        for j in range(len(datasets)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < 0.5 * np.nanmax(mat) else "black",
                        fontsize=9)
    fig.colorbar(im, ax=ax, label=key)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# Regenerate report + heatmap from a saved metrics.json (no models/GPU needed):
#
#   python -m DeepDataMiningLearning.ngdet.report --metrics output/ngdet/metrics.json
#
# Expected: writes report.md and heatmap_mAP.png next to the metrics file.
# ===========================================================================
if __name__ == "__main__":
    import argparse
    from .taxonomy import Taxonomy

    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="path to metrics.json")
    ap.add_argument("--taxonomy", default="driving3")
    a = ap.parse_args()

    with open(a.metrics) as f:
        all_metrics = json.load(f)
    out_dir = os.path.dirname(os.path.abspath(a.metrics))
    tax = Taxonomy.from_preset(a.taxonomy)
    rp = write_markdown_report(all_metrics, os.path.join(out_dir, "report.md"), tax)
    hp = plot_heatmap(all_metrics, os.path.join(out_dir, "heatmap_mAP.png"))
    print(f"wrote {rp}\nwrote {hp}")
