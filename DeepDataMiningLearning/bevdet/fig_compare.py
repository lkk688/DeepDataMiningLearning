import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

def plot_bevfusion_latency_memory(out_dir="figs",
                                  filename="bevfusion_latency_memory",
                                  dpi=200):
    os.makedirs(out_dir, exist_ok=True)

    # --- Data ---
    methods = ["Original\nBEVFusion",
               "Ours\nOption1",
               "Ours\nOption2"]

    latency_ms = np.array([66.2, 55.1, 59.4], dtype=float)  # lower is better
    peak_mb    = np.array([1981.5, 852.7, 795.9], dtype=float)  # lower is better

    # Percent drop vs Original
    def pct_drop(old, new):
        return (old - new) / old * 100.0
    lat_drop = [0.0,
                pct_drop(latency_ms[0], latency_ms[1]),
                pct_drop(latency_ms[0], latency_ms[2])]
    mem_drop = [0.0,
                pct_drop(peak_mb[0], peak_mb[1]),
                pct_drop(peak_mb[0], peak_mb[2])]

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=dpi)  # CVPR-friendly width

    # Common x positions
    x = np.arange(len(methods))

    # Hatching: emphasize "Ours"
    hatches = ["", "//", "xx"]  # original has no hatch

    # --- Subplot (a): Latency ---
    ax = axes[0]
    bars = ax.bar(x, latency_ms)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)

    # Annotate values (ms) and percent drop for Ours
    for i, (b, v, d) in enumerate(zip(bars, latency_ms, lat_drop)):
        ax.text(b.get_x() + b.get_width()/2.0, v,
                f"{v:.1f} ms" + (f"\n(-{d:.1f}%)" if i != 0 else ""),
                ha="center", va="bottom", fontsize=9)

    ax.set_title("Latency (ms) ↓", fontsize=11)
    ax.set_xticks(x, methods, rotation=0)
    ax.set_ylabel("Latency (ms)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.15)

    # --- Subplot (b): Peak GPU memory ---
    ax = axes[1]
    bars = ax.bar(x, peak_mb)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)

    for i, (b, v, d) in enumerate(zip(bars, peak_mb, mem_drop)):
        ax.text(b.get_x() + b.get_width()/2.0, v,
                f"{v:.1f} MB" + (f"\n(-{d:.1f}%)" if i != 0 else ""),
                ha="center", va="bottom", fontsize=9)

    ax.set_title("Peak GPU Memory (MB) ↓", fontsize=11)
    ax.set_xticks(x, methods, rotation=0)
    ax.set_ylabel("Memory (MB)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.15)

    # Overall layout
    fig.tight_layout()

    pdf_path = osp.join(out_dir, f"{filename}.pdf")
    png_path = osp.join(out_dir, f"{filename}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {pdf_path}")
    print(f"[OK] Saved {png_path}")

def plot_bevfusion_latency_memory_singlecol(out_dir="figs",
                                            filename="bevfusion_latency_memory_singlecol",
                                            dpi=200):
    os.makedirs(out_dir, exist_ok=True)

    # Data
    methods = ["Original\nBEVFusion", "Ours\nOption1", "Ours\nOption2"]
    latency_ms = np.array([66.2, 55.1, 59.4], dtype=float)   # lower is better
    peak_mb    = np.array([1981.5, 852.7, 795.9], dtype=float)  # lower is better

    # Percent reductions vs Original
    def pct_drop(old, new): return (old - new) / old * 100.0
    lat_drop = [0.0, pct_drop(latency_ms[0], latency_ms[1]), pct_drop(latency_ms[0], latency_ms[2])]
    mem_drop = [0.0, pct_drop(peak_mb[0],    peak_mb[1]),    pct_drop(peak_mb[0],    peak_mb[2])]

    # Single-column figure: two vertical subplots
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.2), dpi=dpi, constrained_layout=True)
    x = np.arange(len(methods))
    hatches = ["", "//", "xx"]  # emphasize ours, grayscale-friendly

    # (a) Latency
    ax = axes[0]
    bars = ax.bar(x, latency_ms)
    for b, h in zip(bars, hatches): b.set_hatch(h)
    for i, (b, v, d) in enumerate(zip(bars, latency_ms, lat_drop)):
        ax.text(b.get_x() + b.get_width()/2.0, v, f"{v:.1f} ms" + (f"\n(-{d:.1f}%)" if i else ""),
                ha="center", va="bottom", fontsize=8)
    ax.set_title("Latency (ms) ↓", fontsize=10)
    ax.set_xticks(x, methods)
    ax.set_ylabel("Latency (ms)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.margins(y=0.15)

    # (b) Peak GPU Memory
    ax = axes[1]
    bars = ax.bar(x, peak_mb)
    for b, h in zip(bars, hatches): b.set_hatch(h)
    for i, (b, v, d) in enumerate(zip(bars, peak_mb, mem_drop)):
        ax.text(b.get_x() + b.get_width()/2.0, v, f"{v:.1f} MB" + (f"\n(-{d:.1f}%)" if i else ""),
                ha="center", va="bottom", fontsize=8)
    ax.set_title("Peak GPU Memory (MB) ↓", fontsize=10)
    ax.set_xticks(x, methods)
    ax.set_ylabel("Memory (MB)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.margins(y=0.15)

    pdf = osp.join(out_dir, f"{filename}.pdf")
    png = osp.join(out_dir, f"{filename}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {pdf}\n[OK] Saved {png}")

def plot_nuscenes_metrics_singlecol(out_dir="figs",
                                    filename="bevfusion_nuscenes_metrics_singlecol",
                                    dpi=200, setfigsize=(3.5, 4.8)):
    os.makedirs(out_dir, exist_ok=True)

    # --------- Input metrics (None/np.nan handled gracefully) ----------
    metrics_by_method = {
        "Original BEVFusion": {
            "mAP": 0.6841, "NDS": 0.7117,
            "mATE": 0.2796, "mASE": 0.2546, "mAOE": 0.3005, "mAVE": 0.2806, "mAAE": 0.1879,
        },
        "Ours Option1": {
            "mAP": None,  "NDS": 0.6811,
            "mATE": 0.2834, "mASE": 0.2560, "mAOE": 0.3277, "mAVE": 0.3005, "mAAE": 0.1897,
        },
        "Ours Option2": {
            "mAP": 0.6006, "NDS": 0.6515,
            "mATE": 0.3205, "mASE": 0.2617, "mAOE": 0.3812, "mAVE": 0.3292, "mAAE": 0.1953,
        },
    }

    # Define which metrics are higher/lower better
    higher_metrics = [("mAP", "mAP (↑)"), ("NDS", "NDS (↑)")]
    lower_metrics  = [("mATE", "mATE (↓)"), ("mASE", "mASE (↓)"),
                      ("mAOE", "mAOE (↓)"), ("mAVE", "mAVE (↓)"), ("mAAE", "mAAE (↓)")]

    methods = list(metrics_by_method.keys())
    k_h = len(higher_metrics)
    k_l = len(lower_metrics)

    # Single-column (two stacked subplots)
    fig, axes = plt.subplots(2, 1, figsize=setfigsize, dpi=dpi, constrained_layout=True)

    # Hatching to emphasize ours (grayscale-friendly)
    hatches = ["", "//", "xx"]  # Original, Option1, Option2

    # --------- Subplot 1: Higher is better (mAP, NDS) ----------
    ax = axes[0]
    x = np.arange(k_h)
    W = 0.82
    bw = W / len(methods)

    for i, mname in enumerate(methods):
        vals = []
        for key, _lbl in higher_metrics:
            v = metrics_by_method[mname].get(key, None)
            vals.append(np.nan if v is None else float(v))
        vals = np.array(vals, dtype=float)

        bars = ax.bar(x + (i - (len(methods)-1)/2)*bw, np.nan_to_num(vals, nan=0.0),
                      width=bw, label=mname, hatch=hatches[i])
        # annotate only finite values
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width()/2.0, v, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, [lbl for _, lbl in higher_metrics], rotation=0)
    ax.set_ylabel("Score")
    ax.set_title("NuScenes: mAP / NDS (higher is better)", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.margins(y=0.15)
    ax.legend(fontsize=8, frameon=False, ncols=1, loc="lower right")

    # --------- Subplot 2: Lower is better (error metrics) ----------
    ax = axes[1]
    x = np.arange(k_l)

    for i, mname in enumerate(methods):
        vals = []
        for key, _lbl in lower_metrics:
            v = metrics_by_method[mname].get(key, None)
            vals.append(np.nan if v is None else float(v))
        vals = np.array(vals, dtype=float)

        bars = ax.bar(x + (i - (len(methods)-1)/2)*bw, np.nan_to_num(vals, nan=0.0),
                      width=bw, label=mname, hatch=hatches[i])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width()/2.0, v, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, [lbl for _, lbl in lower_metrics], rotation=15, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("NuScenes: ATE / ASE / AOE / AVE / AAE (lower is better)", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.margins(y=0.15)

    # Save
    out_pdf = osp.join(out_dir, f"{filename}.pdf")
    out_png = osp.join(out_dir, f"{filename}.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_pdf}\n[OK] Saved {out_png}")


from matplotlib.gridspec import GridSpec

def plot_nuscenes_metrics_singlecol_sidelegend(
    out_dir="figs",
    filename="bevfusion_nuscenes_metrics_singlecol_sidelegend",
    dpi=200
):
    os.makedirs(out_dir, exist_ok=True)

    # --------- Input metrics ----------
    metrics_by_method = {
        "Original BEVFusion": {
            "mAP": 0.6841, "NDS": 0.7117,
            "mATE": 0.2796, "mASE": 0.2546, "mAOE": 0.3005, "mAVE": 0.2806, "mAAE": 0.1879,
        },
        "Ours Option1": {
            "mAP": 0.6397,  "NDS": 0.6811,
            "mATE": 0.2834, "mASE": 0.2560, "mAOE": 0.3277, "mAVE": 0.3005, "mAAE": 0.1897,
        },
        "Ours Option2": {
            "mAP": 0.6006, "NDS": 0.6515,
            "mATE": 0.3205, "mASE": 0.2617, "mAOE": 0.3812, "mAVE": 0.3292, "mAAE": 0.1953,
        },
    }
    higher_metrics = [("mAP", "mAP (↑)"), ("NDS", "NDS (↑)")]
    lower_metrics  = [("mATE", "mATE (↓)"), ("mASE", "mASE (↓)"),
                      ("mAOE", "mAOE (↓)"), ("mAVE", "mAVE (↓)"), ("mAAE", "mAAE (↓)")]
    methods = list(metrics_by_method.keys())
    hatches = ["", "//", "xx"]  # Original, Option1, Option2

    # --- Figure layout: 2 rows x 2 cols; legend gets its own right column in the top row
    # Left column holds plots; right column (narrower) hosts legend for the top plot only.
    #(3.5, 5.0)
    fig = plt.figure(figsize=(6, 6), dpi=dpi, constrained_layout=True)
    gs = GridSpec(nrows=2, ncols=2, figure=fig,
                  width_ratios=[1.0, 0.55], height_ratios=[1.0, 1.0])

    ax_top   = fig.add_subplot(gs[0, 0])
    ax_legen = fig.add_subplot(gs[0, 1])  # legend-only axis
    ax_bot   = fig.add_subplot(gs[1, :])  # span both columns

    # Hide legend axis frame/ticks
    ax_legen.axis("off")

    # ------------------- Top subplot: mAP/NDS (higher is better), narrower bars -------------------
    xh = np.arange(len(higher_metrics))
    W1 = 0.62  # narrower total cluster width for first subplot
    bw1 = W1 / len(methods)

    handles, labels = [], []
    for i, mname in enumerate(methods):
        vals = [metrics_by_method[mname].get(k, None) for k, _ in higher_metrics]
        vals = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)

        bars = ax_top.bar(xh + (i - (len(methods)-1)/2)*bw1,
                          np.nan_to_num(vals, nan=0.0),
                          width=bw1, label=mname, hatch=hatches[i])
        # keep one handle per method for legend
        handles.append(bars[0]); labels.append(mname)

        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax_top.text(b.get_x() + b.get_width()/2.0, v, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=8)

    ax_top.set_xticks(xh, [lbl for _, lbl in higher_metrics], rotation=0)
    ax_top.set_ylabel("Score")
    ax_top.set_title("NuScenes: mAP / NDS (higher is better)", fontsize=10)
    ax_top.spines["top"].set_visible(False); ax_top.spines["right"].set_visible(False)
    ax_top.margins(y=0.15)

    # Legend placed in dedicated side panel (more space, no overlap)
    ax_legen.legend(handles, labels, loc="center", frameon=False, fontsize=8)

    # ------------------- Bottom subplot: error metrics (lower is better), normal width -------------------
    xl = np.arange(len(lower_metrics))
    W2 = 0.82
    bw2 = W2 / len(methods)

    for i, mname in enumerate(methods):
        vals = [metrics_by_method[mname].get(k, None) for k, _ in lower_metrics]
        vals = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)

        bars = ax_bot.bar(xl + (i - (len(methods)-1)/2)*bw2,
                          np.nan_to_num(vals, nan=0.0),
                          width=bw2, label=mname, hatch=hatches[i])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax_bot.text(b.get_x() + b.get_width()/2.0, v, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=8)

    ax_bot.set_xticks(xl, [lbl for _, lbl in lower_metrics], rotation=15, ha="right")
    ax_bot.set_ylabel("Error")
    ax_bot.set_title("NuScenes: ATE / ASE / AOE / AVE / AAE (lower is better)", fontsize=10)
    ax_bot.spines["top"].set_visible(False); ax_bot.spines["right"].set_visible(False)
    ax_bot.margins(y=0.15)

    # --- Save
    out_pdf = osp.join(out_dir, f"{filename}.pdf")
    out_png = osp.join(out_dir, f"{filename}.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_pdf}\n[OK] Saved {out_png}")


OURS_RAW = """
car,0.861,0.175,0.153,0.100,0.306,0.190
truck,0.568,0.347,0.187,0.092,0.269,0.226
bus,0.731,0.318,0.190,0.048,0.542,0.248
trailer,0.449,0.532,0.220,0.570,0.183,0.161
construction_vehicle,0.242,0.649,0.410,0.877,0.126,0.314
pedestrian,0.860,0.129,0.285,0.378,0.230,0.093
motorcycle,0.666,0.196,0.238,0.235,0.593,0.247
bicycle,0.523,0.169,0.249,0.465,0.230,0.010
traffic_cone,0.722,0.121,0.329,nan,nan,nan
barrier,0.685,0.200,0.283,0.059,nan,nan
""".strip()

ORIG_RAW = """
car,0.894,0.169,0.150,0.063,0.285,0.185
truck,0.640,0.319,0.181,0.083,0.257,0.225
bus,0.769,0.332,0.185,0.058,0.472,0.257
trailer,0.481,0.507,0.210,0.628,0.189,0.169
construction_vehicle,0.288,0.693,0.422,0.848,0.129,0.312
pedestrian,0.879,0.130,0.292,0.391,0.227,0.103
motorcycle,0.758,0.185,0.238,0.256,0.478,0.243
bicycle,0.625,0.155,0.257,0.319,0.208,0.009
traffic_cone,0.790,0.122,0.327,nan,nan,nan
barrier,0.716,0.184,0.285,0.058,nan,nan
""".strip()

def _parse_csv_block(block: str):
    """
    CSV lines of: class, AP, ATE, ASE, AOE, AVE, AAE
    -> dict: {class: {'AP':..,'ATE':.., ...}}
    """
    out = {}
    for line in block.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        cls, ap, ate, ase, aoe, ave, aae = parts
        def f(x):
            try:
                return float(x)
            except Exception:
                return float("nan")
        out[cls] = {
            "AP":  f(ap),
            "ATE": f(ate),
            "ASE": f(ase),
            "AOE": f(aoe),
            "AVE": f(ave),
            "AAE": f(aae),
        }
    return out

def plot_per_class_comparison(
    ours_dict,
    orig_dict,
    out_dir="figs",
    filename="per_class_comparison",
    order=None,               # custom class order; default = sorted union preserving orig order if available
    figsize=(7.0, 9.0),       # for single-column CVPR, try (3.5, 9.0) and include at \columnwidth
    dpi=200,
    annotate=True
):
    """
    One figure with six subplots (AP ↑, then ATE/ASE/AOE/AVE/AAE ↓),
    each showing side-by-side bars for 'Original BEVFusion' and 'Ours'.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Canonical class order: use `order` if provided, else the order appearing in ORIG then Ours.
    if order is None:
        union = list(orig_dict.keys()) + [c for c in ours_dict.keys() if c not in orig_dict]
        order = union

    metrics = [
        ("AP",  "AP (↑)"),
        ("ATE", "ATE (↓)"),
        ("ASE", "ASE (↓)"),
        ("AOE", "AOE (↓)"),
        ("AVE", "AVE (↓)"),
        ("AAE", "AAE (↓)"),
    ]

    # Build aligned arrays
    def get_arr(d, key):
        return np.array([d.get(cls, {}).get(key, float("nan")) for cls in order], dtype=float)

    # Figure
    fig, axes = plt.subplots(6, 1, figsize=figsize, dpi=dpi, constrained_layout=True, sharex=True)

    x = np.arange(len(order))
    W = 0.80               # cluster width
    bw = W / 2.0           # bar width (two methods)
    offsets = (-bw/2, bw/2)

    # Two hatch styles to distinguish methods (no explicit colors)
    methods = [
        ("Original BEVFusion", orig_dict, ""),     # no hatch
        ("Ours",                ours_dict, "//"),  # hatched
    ]

    # Collect handles for a single legend (place beside first subplot)
    legend_handles = []
    legend_labels  = []

    for ax, (mkey, mlabel) in zip(axes, metrics):
        # Plot each method side-by-side
        for i, (name, dct, hatch) in enumerate(methods):
            y = get_arr(dct, mkey)
            xpos = x + offsets[i]
            bars = ax.bar(xpos, np.nan_to_num(y, nan=0.0), width=bw, hatch=hatch, label=name)

            # Save one handle per method for legend
            if len(legend_handles) < len(methods):
                legend_handles.append(bars[0])
                legend_labels.append(name)

            if annotate:
                for bx, v in zip(bars, y):
                    if np.isfinite(v):
                        ax.text(bx.get_x() + bx.get_width()/2.0, v, f"{v:.3f}",
                                ha="center", va="bottom", fontsize=7)
                    else:
                        ax.text(bx.get_x() + bx.get_width()/2.0, 0,
                                "N/A", ha="center", va="bottom", fontsize=6)

        ax.set_ylabel(mlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(y=0.15)

    # Bottom x tick labels = classes
    axes[-1].set_xticks(x, order, rotation=25, ha="right")

    # Title and side legend (next to first subplot for space)
    axes[0].set_title("Per-class Comparison: Original BEVFusion vs Ours", fontsize=11)
    # Place legend outside to the right of the top subplot
    axes[0].legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                   frameon=False, fontsize=8)

    # Save
    pdf = osp.join(out_dir, f"{filename}.pdf")
    png = osp.join(out_dir, f"{filename}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {pdf}\n[OK] Saved {png}")
    
if __name__ == "__main__":
    #plot_bevfusion_latency_memory(out_dir="figs")
    #plot_bevfusion_latency_memory_singlecol(out_dir="figs")
    #plot_nuscenes_metrics_singlecol(setfigsize=(5.5, 5.8))
    #plot_nuscenes_metrics_singlecol()
    plot_nuscenes_metrics_singlecol_sidelegend()
    
    ours = _parse_csv_block(OURS_RAW)
    orig = _parse_csv_block(ORIG_RAW)
    # Keep the semantic order used in the tables
    class_order = [
        "car","truck","bus","trailer","construction_vehicle",
        "pedestrian","motorcycle","bicycle","traffic_cone","barrier"
    ]
    plot_per_class_comparison(ours, orig,
                              out_dir="figs",
                              filename="per_class_comparison",
                              order=class_order,
                              figsize=(6.8, 9.0),  # for single-column, use (3.5, 9.0)
                              dpi=200,
                              annotate=True)