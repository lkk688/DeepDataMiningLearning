"""Aggregate saved runs into tables + plots, with seed statistics.

    python -m nanolm.analyze outputs/nanolm/sweeps/golf_ablation
    python -m nanolm.analyze outputs/nanolm/sweeps/golf_ablation --baseline baseline --metric quant_val_bpb
    python -m nanolm.analyze outputs/nanolm            # every run under a directory

Writes into the analyzed directory:
    summary.csv   one row per run (all metrics)
    summary.md    per-experiment mean +- std over seeds, delta vs. baseline, paired t-test p-value
    curves.png    validation bpb vs. tokens seen (mean over seeds)
    bars.png      final metric per experiment with seed error bars

Why the statistics matter: in parameter golf the seed-to-seed std of val_bpb was ~0.0005-0.001
while many "improvements" were ~0.001-0.003.  One seed cannot tell those apart; the paired test
(same seeds for baseline and variant) is the cheapest honest answer.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_runs(roots: list[str]) -> pd.DataFrame:
    rows = []
    for root in roots:
        for mf in sorted(Path(root).rglob("metrics.json")):
            m = json.loads(mf.read_text())
            m["run_dir"] = str(mf.parent)
            rows.append(m)
    if not rows:
        raise SystemExit(f"no metrics.json found under {roots}")
    return pd.DataFrame(rows)


def load_curves(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, r in df.iterrows():
        f = Path(r["run_dir"]) / "eval_log.jsonl"
        if f.exists():
            c = pd.read_json(f, lines=True)
            if len(c):
                c["name"], c["seed"] = r["name"], r["seed"]
                parts.append(c)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def paired_test(df: pd.DataFrame, metric: str, base: str, name: str):
    a = df[df.name == base].set_index("seed")[metric]
    b = df[df.name == name].set_index("seed")[metric]
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return None, None, len(common)
    d = (b.loc[common] - a.loc[common]).values
    sd = d.std(ddof=1)
    if sd == 0:
        return float(d.mean()), 0.0 if d.mean() != 0 else 1.0, len(common)
    t = d.mean() / (sd / math.sqrt(len(d)))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=len(d) - 1))
    except ImportError:
        p = float("nan")
    return float(d.mean()), p, len(common)


def summarize(df: pd.DataFrame, metric: str, baseline: str | None) -> pd.DataFrame:
    g = df.groupby("name", sort=False)
    out = pd.DataFrame({
        "n_seeds": g.size(),
        f"{metric}_mean": g[metric].mean(),
        f"{metric}_std": g[metric].std(ddof=1),
        "params_M": g["params"].first() / 1e6,
        "train_min": g["train_time_sec"].mean() / 60,
    })
    if "artifact_bytes" in df:
        out["artifact_MB"] = g["artifact_bytes"].mean() / 1e6
    if baseline and baseline in out.index:
        deltas, pvals = [], []
        for name in out.index:
            if name == baseline:
                deltas.append(0.0)
                pvals.append(np.nan)
                continue
            d, p, n = paired_test(df, metric, baseline, name)
            if d is None:   # no matched seeds -> unpaired difference of means
                d = out.loc[name, f"{metric}_mean"] - out.loc[baseline, f"{metric}_mean"]
            deltas.append(d)
            pvals.append(p if p is not None else np.nan)
        out[f"delta_vs_{baseline}"] = deltas
        out["paired_p"] = pvals
        out = out.sort_values(f"{metric}_mean")
    return out


def plot(df: pd.DataFrame, curves: pd.DataFrame, summary: pd.DataFrame, metric: str, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(curves):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, c in curves.groupby("name", sort=False):
            m = c.groupby("train_tokens")["bpb"].agg(["mean", "std"]).reset_index()
            ax.plot(m["train_tokens"] / 1e6, m["mean"], label=name)
            if m["std"].notna().any():
                ax.fill_between(m["train_tokens"] / 1e6, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.15)
        ax.set_xlabel("tokens seen (M)")
        ax.set_ylabel("val bits-per-byte (periodic estimate)")
        lo = curves["bpb"].quantile(0.0)
        hi = curves.groupby("name")["bpb"].apply(lambda s: s.iloc[len(s) // 4:].max()).max()
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.set_ylim(lo - 0.02 * (hi - lo), hi)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "curves.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(summary)), 4.5))
    mean, std = summary[f"{metric}_mean"], summary[f"{metric}_std"].fillna(0)
    ax.bar(range(len(summary)), mean, yerr=std, capsize=3, color="#4C78A8")
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary.index, rotation=40, ha="right", fontsize=8)
    pad = max(float(std.max()), 1e-3) * 3 + 0.1 * float(mean.max() - mean.min())
    ax.set_ylim(float(mean.min()) - pad, float(mean.max()) + pad)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bars.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="nanolm results analysis")
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--metric", default="auto", help="val_bpb | sliding_bpb | quant_val_bpb | auto")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--out", default=None, help="output dir (default: first root)")
    args = ap.parse_args(argv)

    df = load_runs(args.roots)
    metric = args.metric
    if metric == "auto":
        metric = next((m for m in ("quant_val_bpb", "sliding_bpb", "val_bpb") if m in df and df[m].notna().all()),
                      "val_bpb")
    out_dir = Path(args.out or args.roots[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    summary = summarize(df, metric, args.baseline if args.baseline in set(df.name) else None)
    pd.set_option("display.width", 200)
    print(f"\nmetric = {metric}  (lower is better)\n")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    within = df.groupby("name")[metric].std(ddof=1).dropna()
    if len(within):
        print(f"\nseed noise floor: median within-experiment std = {within.median():.4f} bpb "
              f"(differences below ~2x this are not trustworthy without more seeds)")
    with open(out_dir / "summary.md", "w") as f:
        f.write(f"metric: `{metric}` (lower is better)\n\n")
        try:
            f.write(summary.to_markdown(floatfmt=".4f"))
        except ImportError:                      # `tabulate` not installed
            f.write("```\n" + summary.to_string(float_format=lambda v: f"{v:.4f}") + "\n```")
        f.write("\n")
    plot(df, load_curves(df), summary, metric, out_dir)
    print(f"\nwrote {out_dir}/summary.csv, summary.md, curves.png, bars.png")


if __name__ == "__main__":
    main()
