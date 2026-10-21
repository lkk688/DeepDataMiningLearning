"""Figures for TUTORIAL_mac_inference.md, from the raw measurement files in this directory.

python mac_inference/make_figures.py      # -> docs/mac_inference_figs/*.png
"""
import collections, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs/mac_inference_figs"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # categorical slots 1-5, fixed order
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def rows(name):
    return [json.loads(l) for l in open(HERE / name) if l.strip()]


def memory():
    """Stacked memory per context length: weights + KV cache + Gated DeltaNet state + prefill scratch."""
    m = {r["size"]: r for r in rows("results_m5pro_qwen3.8-27b_memory.jsonl")}
    w = m[0]["weights_gb"]; gdn = 0.154; kv_tok = 65536 / 1e9
    sizes = [1024, 4096, 16384, 32768, 131072, 262144]
    labels = ["1K", "4K", "16K", "32K", "128K*", "256K*"]
    kv = [n * kv_tok for n in sizes]
    scratch = [m[n]["peak_gb"] - m[n]["active_gb"] if n in m else 2.7 for n in sizes]   # *: projected
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    bottom = [0.0] * len(sizes)
    for vals, name, color in [([w] * len(sizes), "weights (affine 4-bit)", SERIES[0]), (kv, "KV cache (16 attention layers)", SERIES[1]),
                              ([gdn] * len(sizes), "Gated DeltaNet state (48 layers)", SERIES[2]), (scratch, "prefill scratch", SERIES[3])]:
        ax.bar(labels, vals, bottom=bottom, color=color, width=0.62, label=name, edgecolor=SURF, linewidth=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, t in enumerate(bottom):
        ax.text(i, t + 0.5, f"{t:.1f} GB", ha="center", color=INK, fontsize=9)
    for y, name in [(24 * 0.74, "24 GB Mac:\nGPU limit ≈ 17.8 GB"), (51.8, "64 GB Mac:\nGPU limit 51.8 GB")]:
        ax.axhline(y, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(1.01, y, name, transform=ax.get_yaxis_transform(), ha="left", va="center", color=INK2, fontsize=8)
    ax.set_ylabel("GB"); ax.set_ylim(0, 58); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.set_title("Qwen3.8-27B (4-bit) memory vs context length, MLX on M5 Pro", loc="left", color=INK, fontsize=11)
    ax.set_xlabel("context (tokens)   * projected from the measured 64 KiB/token")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=8.5)
    fig.tight_layout(); fig.subplots_adjust(right=0.80); fig.savefig(OUT / "memory_vs_context.png", dpi=150); plt.close(fig)


def decode():
    """Decode tok/s per engine configuration, grouped by prompt."""
    d = collections.defaultdict(list)
    for r in rows("results_m5pro_qwen3.8-27b.jsonl") + rows("results_m5pro_splash.jsonl"):
        if "decode_tok_s" in r:
            d[(r["label"], r["prompt"])].append(r["decode_tok_s"])
    configs = [("mlx_4bit", "MLX-LM"), ("mlxvlm_4bit_mtp", "mlx-vlm + MTP"), ("omlx_Qwen3.8-27B-oQ4e-mtp", "oMLX oQ4e + MTP"),
               ("lithos_4bit_nodraft", "lithos-metal"), ("lithos_4bit_dspark", "lithos-metal + DSpark"),
               ("splash_27b", "Splash + DFlash2")]
    prompts = [("code", "code"), ("explain", "prose"), ("math", "math")]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    h = 0.26
    for j, (p, pname) in enumerate(prompts):
        ys = [i + (j - 1) * h for i in range(len(configs))]
        vals = [sum(d[(c, p)]) / len(d[(c, p)]) for c, _ in configs]
        ax.barh(ys, vals, height=h - 0.03, color=SERIES[j], label=pname)
        for y, v in zip(ys, vals):
            ax.text(v + 0.6, y, f"{v:.0f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(configs)), [n for _, n in configs]); ax.invert_yaxis()
    ax.axvline(19, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(19.5, len(configs) - 0.45, "bandwidth limit for one token per pass ≈ 19 tok/s", fontsize=8, color=INK2)
    ax.set_xlabel("decode tokens/s (greedy, 512 tokens; higher is better)"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    ax.set_title("Qwen3.8-27B decode speed on M5 Pro", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "decode_by_engine.png", dpi=150); plt.close(fig)


def ttft():
    """Cold time to first token vs prompt length."""
    c = collections.defaultdict(list)
    for r in rows("results_m5pro_qwen3.8-27b_prefill.jsonl") + rows("results_m5pro_splash_prefill.jsonl") \
            + rows("results_thor_qwen3.8.jsonl"):
        if r.get("test") == "cold":
            c[(r["label"], r["size"])].append(r["ttft_s"])
    configs = [("pf_mlx_lm", "MLX-LM"), ("pf_mlxvlm_mtp", "mlx-vlm"), ("pf_splash_27b", "Splash"),
               ("pf_omlx_Qwen3.8-27B-oQ4e-mtp", "oMLX oQ4e"), ("pf_lithos_dspark", "lithos-metal")]
    sizes = [1024, 4096, 16384, 32768]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ends = []
    for k, (lab, name) in enumerate(configs):
        ys = [sum(c[(lab, n)]) / len(c[(lab, n)]) for n in sizes]
        ax.plot(sizes, ys, color=SERIES[k], linewidth=2, marker="o", markersize=5, label=name)
        ends.append([ys[-1], ys[-1], name])
    ys = [sum(c[("pf_thor_vllm_27b_nvfp4_mtp", n)]) / len(c[("pf_thor_vllm_27b_nvfp4_mtp", n)]) for n in sizes]
    ax.plot(sizes, ys, color=INK2, linewidth=1.5, linestyle=(0, (4, 3)), marker="o", markersize=4,
            label="Jetson Thor, vLLM NVFP4 (reference)")
    ends.append([ys[-1], ys[-1], "Thor vLLM"])
    ends.sort()                                                  # dodge end labels: at least x1.32 apart (log axis)
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] * 1.32)
    for v, y, name in ends:
        ax.text(sizes[-1] * 1.08, y, f"{name} {v:.0f} s", va="center", fontsize=8, color=INK)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(sizes, ["1K", "4K", "16K", "32K"]); ax.set_xlim(800, 32768 * 3.2)
    ax.set_xlabel("prompt tokens (cold, no cache)"); ax.set_ylabel("time to first token (s, log)")
    ax.grid(color=GRID, which="major"); ax.set_axisbelow(True)
    ax.set_title("Qwen3.8-27B prefill on M5 Pro (Jetson Thor dashed): what an agent waits for", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "ttft_vs_prompt.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    memory(); decode(); ttft()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
