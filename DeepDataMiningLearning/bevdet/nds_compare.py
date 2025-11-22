# -*- coding: utf-8 -*-
import os
import os.path as osp
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Small utils
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _normalize_value(v: Any, default: float = 0.0) -> float:
    """Return float; if looks like percentage (>1.5), convert to [0,1]."""
    try:
        f = float(v)
        if f > 1.5:  # heuristically consider it percentage
            f = f / 100.0
        return f
    except Exception:
        return default

def _get_metric(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """
    Robust metric getter for common NuScenes keys.
    Accepts either exact keys (mAP, NDS, mATE, ...) or common alternates.
    """
    # direct hit
    if key in d:
        return _normalize_value(d[key], default)

    k_low = key.lower()
    dl = {str(k).lower(): v for k, v in d.items()}

    # common alternates
    synonyms = {
        "map": ["map", "mean_ap", "meanap"],
        "nds": ["nds", "nd_score", "ndscore"],
        "mate": ["mate", "mean_ate", "trans_err", "mean_trans_err"],
        "mase": ["mase", "mean_ase", "scale_err", "mean_scale_err"],
        "maoe": ["maoe", "mean_aoe", "orient_err", "mean_orient_err", "rot_err"],
        "mave": ["mave", "mean_ave", "vel_err", "mean_vel_err"],
        "maae": ["maae", "mean_aae", "attr_err", "mean_attr_err"],
    }
    root = k_low[1:] if k_low.startswith("m") and len(k_low) > 1 else k_low
    cand = synonyms.get(root, [])
    for c in [k_low, root] + cand:
        if c in dl:
            return _normalize_value(dl[c], default)

    # nested dicts (some writers do {"metrics": {...}})
    for nested_key in ("metrics", "results", "summary", "overall", "bbox", "eval"):
        nested = d.get(nested_key)
        if isinstance(nested, dict):
            v = _get_metric(nested, key, default=None)  # type: ignore
            if v is not None:
                return v
    return default


# -----------------------------
# JSON loading & parsing
# -----------------------------
def load_nuscenes_metrics_from_json(json_path: str) -> Dict[str, Any]:
    """
    Load a NuScenes evaluation JSON and return a flat metrics dict with keys:
      mAP, NDS, mATE, mASE, mAOE, mAVE, mAAE
    The parser is lenient to key variants.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    metrics = {
        "mAP": _get_metric(data, "mAP", 0.0),
        "NDS": _get_metric(data, "NDS", 0.0),
        "mATE": _get_metric(data, "mATE", 0.0),
        "mASE": _get_metric(data, "mASE", 0.0),
        "mAOE": _get_metric(data, "mAOE", 0.0),
        "mAVE": _get_metric(data, "mAVE", 0.0),
        "mAAE": _get_metric(data, "mAAE", 0.0),
    }
    return metrics


def load_multiple_runs(json_files: List[str],
                       method_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Load several NuScenes result JSONs. Returns:
      [{"name": <method>, "path": <file>, "metrics": {...}}, ...]
    """
    if method_names is None:
        method_names = [osp.splitext(osp.basename(p))[0] for p in json_files]
    assert len(method_names) == len(json_files)

    runs = []
    for name, jp in zip(method_names, json_files):
        m = load_nuscenes_metrics_from_json(jp)
        runs.append({"name": name, "path": jp, "metrics": m})
    return runs


# -----------------------------
# Plotting helpers (CVPR-friendly)
# -----------------------------
def _annotate_bars(ax, bars, fmt="{:.3f}", fontsize=9):
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h,
                fmt.format(h), ha="center", va="bottom", fontsize=fontsize)

def save_comprehensive_metrics_bar(acc: dict, title: str, out_pdf: str):
    """
    One figure summarizing the 7 major NuScenes metrics for ONE run.
    Saves vector PDF.
    """
    mAP = _get_metric(acc, "mAP", 0.0)
    NDS = _get_metric(acc, "NDS", 0.0)
    mATE = _get_metric(acc, "mATE", 0.0)
    mASE = _get_metric(acc, "mASE", 0.0)
    mAOE = _get_metric(acc, "mAOE", 0.0)
    mAVE = _get_metric(acc, "mAVE", 0.0)
    mAAE = _get_metric(acc, "mAAE", 0.0)

    labels = ["mAP (↑)", "NDS (↑)", "mATE (↓)", "mASE (↓)", "mAOE (↓)", "mAVE (↓)", "mAAE (↓)"]
    values = [mAP, NDS, mATE, mASE, mAOE, mAVE, mAAE]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    x = np.arange(len(labels))
    bars = ax.bar(x, values)  # use default colors (journal-friendly)
    _annotate_bars(ax, bars, fmt="{:.3f}", fontsize=9)

    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(title)
    fig.tight_layout()
    ensure_dir(osp.dirname(out_pdf))
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_metric_comparison_bar(runs: List[Dict[str, Any]], metric_key: str,
                               ylabel: str, title: str, out_pdf: str):
    """
    One figure comparing ONE metric across all methods.
    e.g., compare mAP across several JSONs.
    """
    names = [r["name"] for r in runs]
    vals = [_get_metric(r["metrics"], metric_key, 0.0) for r in runs]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    x = np.arange(len(names))
    bars = ax.bar(x, vals)
    _annotate_bars(ax, bars, fmt="{:.3f}", fontsize=9)

    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    ensure_dir(osp.dirname(out_pdf))
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_grouped_summary_bar(runs: List[Dict[str, Any]], title: str, out_pdf: str):
    """
    Grouped bars: x-axis = 7 metrics, bars = methods.
    Generates a compact “comparison at a glance” figure.
    """
    labels = ["mAP (↑)", "NDS (↑)", "mATE (↓)", "mASE (↓)", "mAOE (↓)", "mAVE (↓)", "mAAE (↓)"]
    keys   = ["mAP",       "NDS",     "mATE",     "mASE",     "mAOE",     "mAVE",     "mAAE"]
    x = np.arange(len(labels))
    W = 0.82  # total cluster width
    k = len(runs)
    if k == 0:
        return
    bw = W / k

    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=200)
    for i, r in enumerate(runs):
        vals = [_get_metric(r["metrics"], mk, 0.0) for mk in keys]
        bars = ax.bar(x + (i - (k-1)/2)*bw, vals, width=bw, label=r["name"])
        # annotate only the best bars to reduce clutter (optional)
        # Here we keep the figure clean (CVPR-style), omit per-bar labels.

    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(ncols=min(3, k), fontsize=9, frameon=False)
    fig.tight_layout()
    ensure_dir(osp.dirname(out_pdf))
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Orchestrator
# -----------------------------
def build_plots_and_tsv_from_json(json_files: List[str],
                                  method_names: Optional[List[str]],
                                  out_dir: str) -> None:
    """
    Load several NuScenes result JSONs and produce:
      - One per-method summary bar (7 metrics)
      - Seven across-method metric bars (one per metric)
      - One grouped summary bar (metrics on x, bars per method)
      - A TSV summarizing all methods
    """
    ensure_dir(out_dir)
    runs = load_multiple_runs(json_files, method_names)
    if not runs:
        print("[WARN] No runs to plot.")
        return

    # 1) Per-method comprehensive bar
    for r in runs:
        title = f"NuScenes summary metrics ({r['name']})"
        pdf = osp.join(out_dir, f"summary_{r['name']}.pdf")
        save_comprehensive_metrics_bar(r["metrics"], title, pdf)
        print(f"[OK] Per-method summary → {pdf}")

    # 2) Across-method comparisons (one figure per metric)
    cmp_specs: List[Tuple[str, str, str]] = [
        ("mAP", "mAP", "nuScenes mAP (higher is better)"),
        ("NDS", "NDS", "nuScenes NDS (higher is better)"),
        ("mATE", "mATE", "nuScenes mATE (lower is better)"),
        ("mASE", "mASE", "nuScenes mASE (lower is better)"),
        ("mAOE", "mAOE", "nuScenes mAOE (lower is better)"),
        ("mAVE", "mAVE", "nuScenes mAVE (lower is better)"),
        ("mAAE", "mAAE", "nuScenes mAAE (lower is better)"),
    ]
    for key, ylabel, title in cmp_specs:
        pdf = osp.join(out_dir, f"compare_{key}.pdf")
        save_metric_comparison_bar(runs, key, ylabel, title, pdf)
        print(f"[OK] Comparison ({key}) → {pdf}")

    # 3) Grouped summary figure (compact comparison)
    grouped_pdf = osp.join(out_dir, "compare_grouped_summary.pdf")
    save_grouped_summary_bar(runs, "nuScenes — Summary Comparison", grouped_pdf)
    print(f"[OK] Grouped summary → {grouped_pdf}")

    # 4) TSV export
    import csv
    tsv_path = osp.join(out_dir, "nuscenes_summary.tsv")
    with open(tsv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["method", "mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE", "json_path"])
        for r in runs:
            m = r["metrics"]
            w.writerow([
                r["name"],
                _get_metric(m, "mAP", 0.0),
                _get_metric(m, "NDS", 0.0),
                _get_metric(m, "mATE", 0.0),
                _get_metric(m, "mASE", 0.0),
                _get_metric(m, "mAOE", 0.0),
                _get_metric(m, "mAVE", 0.0),
                _get_metric(m, "mAAE", 0.0),
                r["path"],
            ])
    print(f"[OK] Wrote {tsv_path}")

if __name__ == "__main__":
    json_files = [
        "/path/to/methodA_eval.json",
        "/path/to/methodB_eval.json",
        "/path/to/methodC_eval.json",
    ]
    method_names = ["MethodA", "MethodB", "MethodC"]  # or None to infer from filenames
    out_dir = "/path/to/plots"

    build_plots_and_tsv_from_json(json_files, method_names, out_dir)