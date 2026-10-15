"""Figures for the tutorial -- every PNG in docs/nanolm_figs/ is regenerated from saved runs by this file.

    python -m nanolm.viz all                      # everything that has its inputs available
    python -m nanolm.viz attention --run outputs/nanolm/laptop_ref/L2_char_gpt --text "ROMEO: But soft, what light"
    python -m nanolm.viz list                     # names of the figures

Students: point --run at your own run directory to see YOUR model's attention / sampling distribution.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

FIG_DIR = Path("docs/nanolm_figs")
# validated categorical palette (light surface), fixed order -- never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
        "font.family": "sans-serif",
    })


def save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {path}")


def blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)


def load_run(run_dir: str):
    from .generate import load_run as _lr
    return _lr(run_dir, device="cpu")


# =========================================================================== Part 1
def fig_data_window(run_dir="outputs/nanolm/laptop_ref/L2_char_gpt"):
    """How one training example looks: text -> token ids -> (input x, target y = x shifted by one)."""
    _, _, tok, _, _ = load_run(run_dir)
    text = "ROMEO:\nBut soft"
    ids = tok.encode(text)
    n = len(ids)
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * n, 2.6))
    ax.set_xlim(-1.6, n)
    ax.set_ylim(-0.4, 3.2)
    ax.axis("off")
    vis = {" ": "␣", "\n": "⏎"}
    rows = [("character", [vis.get(c, c) for c in text], 2.6),
            ("token id", [str(i) for i in ids], 1.8),
            ("input x", [str(i) for i in ids[:-1]] + [""], 1.0),
            ("target y", [str(i) for i in ids[1:]] + [""], 0.2)]
    for label, cells, y in rows:
        ax.text(-1.5, y, label, ha="left", va="center", color=INK2, fontsize=9)
        for j, c in enumerate(cells):
            if c == "":
                continue
            fc = "#cde2fb" if label == "target y" else ("#f0efec" if label != "input x" else "#fdebe2")
            ax.add_patch(plt.Rectangle((j - 0.42, y - 0.28), 0.84, 0.56, fc=fc, ec=AXIS, lw=0.6))
            ax.text(j, y, c, ha="center", va="center", fontsize=9, family="monospace", color=INK)
    ax.set_title("One training window (␣ = space, ⏎ = newline): the target row is the input row shifted left by one",
                 loc="left")
    save(fig, "p1_data_window")


def token_pieces(tok, text):
    """[(label, n_bytes)] for each token of `text`; byte-fallback tokens are shown as <hex>."""
    from .data import ByteTokenizer, CharTokenizer, SentencePieceTokenizer, TiktokenTokenizer, WordTokenizer
    if tok is None or isinstance(tok, CharTokenizer):
        return [(c, len(c.encode("utf-8"))) for c in text]
    if isinstance(tok, ByteTokenizer):
        return [(f"{b:02x}", 1) for b in text.encode("utf-8")]
    out = []
    for i in tok.encode(text):
        if isinstance(tok, WordTokenizer):
            b = bytes([i]) if i < 256 else tok.pieces[i - 256].encode("utf-8")
        elif isinstance(tok, TiktokenTokenizer):
            b = tok.enc.decode_single_token_bytes(i)
        elif isinstance(tok, SentencePieceTokenizer):
            piece = tok.sp.id_to_piece(i)
            b = bytes([int(piece[3:5], 16)]) if tok.sp.is_byte(i) else piece.replace("▁", " ").encode("utf-8")
        else:
            b = tok.decode([i]).encode("utf-8")
        try:
            label = b.decode("utf-8")
        except UnicodeDecodeError:
            label = "<" + b.hex() + ">"
        out.append((label, len(b)))
    return out


def fig_tokenization(text="The quick brown fox jumps over the lazy dog's 1,234 tokens — naïvely!"):
    """The same sentence cut by different tokenizers (box width = UTF-8 bytes the token covers)."""
    from .data import ByteTokenizer, SentencePieceTokenizer, TiktokenTokenizer, WordTokenizer
    corpus = Path("data/nanolm/raw/tinyshakespeare.txt").read_text()[:900_000]
    toks = [("character", None), ("byte (UTF-8)", ByteTokenizer()),
            ("word, V=2048 (trained on Shakespeare;\nunknown words fall back to bytes)", WordTokenizer.train(corpus, 2048))]
    sp = Path("data/golf/tokenizers/fineweb_1024_bpe.model")
    if sp.exists():
        toks.append(("BPE, V=1024 (FineWeb, Parameter Golf)", SentencePieceTokenizer(str(sp))))
    toks.append(("BPE, V=50257 (GPT-2)", TiktokenTokenizer("gpt2")))
    nbytes = len(text.encode("utf-8"))
    fig, ax = plt.subplots(figsize=(12, 0.8 * len(toks) + 0.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(toks))
    x0, span = 0.25, 0.745
    for r, (name, tok) in enumerate(toks):
        y = len(toks) - r - 0.5
        pieces = token_pieces(tok, text)
        ax.text(0.0, y, f"{name}\n{len(pieces)} tokens", ha="left", va="center", fontsize=8, color=INK2)
        x = x0
        for k, (lab, nb) in enumerate(pieces):
            w = span * nb / nbytes
            ax.add_patch(plt.Rectangle((x, y - 0.3), w, 0.6, fc=("#cde2fb" if k % 2 == 0 else "#9ec5f4"),
                                       ec=SURFACE, lw=1.0))
            lab = lab.replace(" ", "␣")
            if w > 0.008 * max(len(lab), 1):
                ax.text(x + w / 2, y, lab, ha="center", va="center", fontsize=7, family="monospace", color=INK)
            x += w
    ax.set_title("The same sentence, five tokenizers (box width = UTF-8 bytes covered; ␣ = space)", loc="left")
    save(fig, "p1_tokenization")


@torch.no_grad()
def attention_maps(model, idx):
    """Per-layer, per-head causal attention weights softmax(q k^T / sqrt(d)) of a nanolm GPT."""
    import torch.nn.functional as F
    from .models.gpt import apply_rope, rms
    caps = []
    hooks = [blk.attn.register_forward_hook(lambda m, inp, out, store=caps: store.append((m, inp[0])))
             for blk in model.blocks]
    model(idx)
    for h in hooks:
        h.remove()
    maps = []
    for m, x in caps:
        B, T, _ = x.shape
        q = m.c_q(x).view(B, T, m.H, m.hd).transpose(1, 2)
        k = m.c_k(x).view(B, T, m.KV, m.hd).transpose(1, 2)
        if m.qk_norm:
            q, k = rms(q), rms(k)
        if m.rope is not None:
            cos, sin = m.rope(T, x.device)
            q, k = apply_rope(q, cos, sin, m.rope.rot_dim), apply_rope(k, cos, sin, m.rope.rot_dim)
        if m.q_gain is not None:
            q = q * m.q_gain[None, :, None, None]
        if m.KV != m.H:
            k = k.repeat_interleave(m.H // m.KV, dim=1)
        s = q @ k.transpose(-1, -2) / math.sqrt(m.hd)
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1), -float("inf"))
        maps.append(F.softmax(s, -1)[0].cpu().numpy())
    return maps


def fig_attention(run_dir="outputs/nanolm/laptop_ref/L2_char_gpt", text="ROMEO:\nBut soft! what light through"):
    cfg, model, tok, dinfo, _ = load_run(run_dir)
    idx = torch.tensor([tok.encode(text)])
    maps = attention_maps(model, idx)
    labels = [repr(c)[1:-1].replace("\\n", "⏎") for c in text]
    L, H = len(maps), maps[0].shape[0]
    fig, axes = plt.subplots(L, H, figsize=(2.6 * H, 2.6 * L))
    for l in range(L):
        for h in range(H):
            ax = axes[l, h]
            ax.imshow(maps[l][h], cmap=blue_cmap(), vmin=0, vmax=1)
            ax.grid(False)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=5, family="monospace")
            ax.set_yticklabels(labels, fontsize=5, family="monospace")
            ax.tick_params(length=0)
            ax.set_title(f"layer {l} · head {h}", fontsize=8, fontweight="normal", color=INK2)
            if h == 0:
                ax.set_ylabel("query (predicting position)", fontsize=6)
            if l == L - 1:
                ax.set_xlabel("key (attended position)", fontsize=6)
    fig.suptitle(f"Attention weights of the trained character Transformer ({Path(run_dir).name}); "
                 "row t = where position t looks (only ≤ t: causal mask)", x=0.01, y=1.04, ha="left", fontsize=10)
    save(fig, "p1_attention_maps")


@torch.no_grad()
def fig_temperature(run_dir="outputs/nanolm/laptop_ref/L2_char_gpt", prompt="ROMEO:\nBut soft! what light through yonder wi"):
    from .evaluate import sample_next  # noqa: F401  (the same function used for sampling)
    cfg, model, tok, dinfo, _ = load_run(run_dir)
    logits = model(torch.tensor([tok.encode(prompt)]))[0, -1].float()
    temps = [0.5, 1.0, 1.5]
    top = torch.topk(logits, 12).indices
    names = [repr(tok.decode([i]))[1:-1].replace(" ", "␣").replace("\\n", "⏎") for i in top.tolist()]
    fig, axes = plt.subplots(1, len(temps), figsize=(11, 2.8), sharey=True)
    for ax, t in zip(axes, temps):
        p = torch.softmax(logits / t, -1)[top].numpy()
        ax.bar(range(len(top)), p, color=SERIES[0], width=0.7)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(names, family="monospace")
        ax.set_title(f"temperature τ = {t}", loc="left")
        ax.grid(axis="x", visible=False)
        for j in range(3):
            ax.text(j, p[j] + 0.01, f"{p[j]:.2f}", ha="center", fontsize=7.5, color=INK2)
    axes[0].set_ylabel("p(next character)")
    fig.suptitle(f"Next-character distribution after “…{prompt[-12:].replace(chr(10), '⏎')}” (top 12 characters)",
                 x=0.01, y=1.04, ha="left", fontsize=10)
    save(fig, "p1_temperature")


def _curves(sweep_dir, names=None, title="", out="", ylim=None):
    rows = []
    for d in sorted(Path(sweep_dir).iterdir()):
        f = d / "eval_log.jsonl"
        if not f.exists():
            continue
        m = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
        for line in f.read_text().splitlines():
            e = json.loads(line)
            rows.append((m.get("name", d.name), e["train_tokens"], e["bpb"]))
    import pandas as pd
    df = pd.DataFrame(rows, columns=["name", "tokens", "bpb"]).groupby(["name", "tokens"])["bpb"].mean().reset_index()
    order = names or sorted(df.name.unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, n in enumerate(order):
        d = df[df.name == n]
        ax.plot(d.tokens / 1e6, d.bpb, color=SERIES[i % len(SERIES)], label=n)
        ax.text(d.tokens.iloc[-1] / 1e6 * 1.01, d.bpb.iloc[-1], n, fontsize=7.5, color=INK2, va="center")
    ax.set_xlabel("training tokens seen (millions)")
    ax.set_ylabel("validation bits-per-byte (lower is better)")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title, loc="left")
    save(fig, out)


def fig_char_models():
    _curves("outputs/nanolm/sweeps/char_models",
            ["gpt_modern", "lstm", "gru", "lstm_512", "gpt_textbook", "rnn"],
            "RNN family vs Transformers on Tiny Shakespeare (mean of 2 seeds)", "p1_char_models", (2.05, 2.9))


def fig_tokenizer_curves():
    _curves("outputs/nanolm/sweeps/laptop_tokenizers",
            ["byte", "char", "word_2048", "sp_bpe_512", "sp_bpe_2048", "sp_unigram_2048", "word_8192", "gpt2_bpe"],
            "Same model, eight tokenizers: big vocabularies peak early, then overfit", "p1_tokenizers", (2.0, 3.0))


def fig_lr_schedules():
    from .config import TrainConfig
    from .optim import lr_multiplier
    steps = np.arange(0, 1000)
    specs = [("cosine (warmup 50, floor 0.1)", TrainConfig(schedule="cosine", warmup_steps=50, min_lr_frac=0.1)),
             ("WSD / warmdown (last 30% → 0)", TrainConfig(schedule="wsd", warmup_steps=50, warmdown_frac=0.3, min_lr_frac=0.0)),
             ("constant (warmup 50)", TrainConfig(schedule="constant", warmup_steps=50))]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, (n, tc) in enumerate(specs):
        ax.plot(steps, [lr_multiplier(tc, int(s), s / 1000) for s in steps], color=SERIES[i], label=n)
    ax.set_xlabel("training step (of 1000)")
    ax.set_ylabel("LR multiplier")
    ax.legend(fontsize=8)
    ax.set_title("Learning-rate schedules in nanolm/optim.py", loc="left")
    save(fig, "p2_lr_schedules")


def fig_muon():
    """Singular values of a real gradient matrix before / after Muon's Newton-Schulz orthogonalisation."""
    from .optim import zeropower_via_newtonschulz5
    cfg, model, tok, dinfo, meta = load_run("outputs/nanolm/laptop_ref/L4q_gpt_fineweb_quick")
    from .data import TokenData
    data = TokenData(meta, sampler="random", seed=0)
    x, y = data.train_batch(8, 256, torch.device("cpu"))
    model.train()
    logits = model(x)
    torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1)).backward()
    G = model.blocks[1].mlp.fc.weight.grad.float()
    s_raw = torch.linalg.svdvals(G / G.norm())
    s_ns = torch.linalg.svdvals(zeropower_via_newtonschulz5(G, 5).float())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].semilogy(s_raw.numpy(), color=SERIES[0])
    axes[0].set_title("gradient G (normalised): singular values", loc="left")
    axes[0].set_xlabel("singular value index")
    axes[1].plot(s_ns.numpy(), color=SERIES[1])
    axes[1].set_ylim(0, 1.4)
    axes[1].set_title("after 5 Newton–Schulz steps ≈ U Vᵀ", loc="left")
    axes[1].set_xlabel("singular value index")
    fig.suptitle(f"Muon equalises update directions (MLP weight of layer 1, {G.shape[0]}×{G.shape[1]})",
                 x=0.01, y=1.04, ha="left", fontsize=10)
    save(fig, "p2_muon_spectrum")


def fig_recipe():
    _curves("outputs/nanolm/sweeps/transformer_recipe",
            ["gpt2_textbook", "plus_rmsnorm_rope", "plus_swiglu", "plus_wsd_adamw", "plus_muon"],
            "From the 2019 recipe to Muon + WSD on FineWeb (mean of 2 seeds)", "p2_recipe", (1.6, 2.3))



def fig_positions():
    """Three ways to tell a Transformer where a token is: sinusoidal (2017), learned (GPT-2), RoPE."""
    from matplotlib.colors import LinearSegmentedColormap
    div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    # ---- figure 1: sinusoidal encoding + why sin/cos come in pairs
    d, P = 64, 128
    pos = np.arange(P)[:, None]
    i = np.arange(d // 2)[None, :]
    w = 1.0 / 10000 ** (2 * i / d)
    pe = np.zeros((P, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(pos * w), np.cos(pos * w)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    im = ax.imshow(pe.T, aspect="auto", cmap=div, vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xlabel("position p")
    ax.set_ylabel("dimension (even = sin, odd = cos)")
    ax.set_title("Sinusoidal encoding PE(p): fast-rotating pairs at the top, slow at the bottom", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax = axes[1]
    ax.set_aspect("equal")
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=AXIS, lw=1)
    for k, (wk, col, lab) in enumerate([(w[0, 0], SERIES[0], "pair 0 (ω = 1)"), (w[0, 6], SERIES[1], "pair 6 (ω ≈ 0.075)")]):
        pp = np.arange(0, 9)
        r = 1 - 0.12 * k
        xs, ys = r * np.cos(pp * wk), r * np.sin(pp * wk)
        ax.plot(xs, ys, "o-", color=col, ms=5, lw=1, label=lab)
        for j in ((0, 1, 2, 3) if k == 0 else (8,)):
            ax.annotate(f"p={j}", (xs[j], ys[j]), fontsize=7, color=col, xytext=(5, 3 if k == 0 else -10),
                        textcoords="offset points")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("each (sin, cos) pair is a point on a circle;\n+k positions = rotation by kω", loc="left", fontsize=9.5)
    save(fig, "p1_positions_sinusoidal")

    # ---- figure 2: GPT-2's learned table and RoPE's relative-distance profile
    from transformers import AutoModelForCausalLM
    wpe = AutoModelForCausalLM.from_pretrained("gpt2").transformer.wpe.weight.detach().numpy()   # [1024, 768]
    wn = wpe / np.linalg.norm(wpe, axis=1, keepdims=True)
    sim = wn @ wn.T
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.45})
    ax = axes[0]
    im = ax.imshow(sim, cmap=div, vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xlabel("position")
    ax.set_ylabel("position")
    ax.set_title("GPT-2's LEARNED position table (1024 × 768):\ncosine similarity between positions", loc="left", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax = axes[1]
    dh = 128
    delta = np.arange(0, 20001)
    for k, (base, lab) in enumerate([(1e4, "base 10,000 (RoFormer, Llama 1/2)"), (5e5, "base 500,000 (Llama 3)"),
                                      (1e7, "base 10,000,000 (Qwen3.5)")]):
        om = base ** (-2 * np.arange(dh // 2) / dh)
        f = np.cos(np.outer(delta, om)).mean(1)        # <q, R_delta k> for q = k = all-ones, normalised
        ax.plot(delta, f, color=SERIES[k], lw=1.2, label=lab)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("relative distance Δ = n − m (tokens, log scale)")
    ax.set_ylabel("score of identical q, k at distance Δ")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("RoPE: the score depends only on Δ and decays with\ndistance; a larger base decays more slowly", loc="left", fontsize=9.5)
    save(fig, "p1_positions_learned_rope")


# =========================================================================== Part 3
def _component_params(model):
    groups = {"embedding / output": 0, "attention (softmax)": 0, "linear attention": 0, "MLP / experts": 0, "norms & other": 0}
    for n, p in model.named_parameters():
        k = p.numel()
        if "embed" in n or "lm_head" in n or "wte" in n or "wpe" in n:
            groups["embedding / output"] += k
        elif "linear_attn" in n:
            groups["linear attention"] += k
        elif "attn" in n or "self_attn" in n:
            groups["attention (softmax)"] += k
        elif "mlp" in n or "experts" in n:
            groups["MLP / experts"] += k
        else:
            groups["norms & other"] += k
    return groups


def fig_model_anatomy(models=("gpt2", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B")):
    """Parameter breakdown of real small LLMs, built on the 'meta' device from config only (no weights)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    rows = []
    for name in models:
        cfg = AutoConfig.from_pretrained(name)
        cfg = getattr(cfg, "text_config", cfg)          # multimodal configs (Qwen3.5) keep the LM in text_config
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(cfg)
        g = _component_params(m)
        tied = getattr(getattr(cfg, "text_config", cfg), "tie_word_embeddings", False)
        rows.append((name.split("/")[-1], g, tied))
    keys = list(rows[0][1])
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for r, (name, g, tied) in enumerate(rows):
        left = 0
        for i, k in enumerate(keys):
            v = g[k] / 1e6
            if v <= 0:
                continue
            ax.barh(r, v, left=left, color=SERIES[i], height=0.6, edgecolor=SURFACE, linewidth=2,
                    label=k if r == 0 or k == "linear attention" else None)
            left += v
        ax.text(left + 8, r, f"{left:.0f}M{'  (tied emb.)' if tied else ''}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("parameters (millions)")
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, ncol=5, loc="upper left", bbox_to_anchor=(0.0, -0.18))
    ax.grid(axis="y", visible=False)
    ax.set_title("Where the parameters live in small LLMs (built from config.json on the 'meta' device)", loc="left")
    save(fig, "p3_model_anatomy")
    return rows


def fig_layer_pattern(name="Qwen/Qwen3.5-0.8B"):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(name)
    tc = getattr(cfg, "text_config", cfg)
    types = tc.layer_types
    fig, ax = plt.subplots(figsize=(10, 1.4))
    for i, t in enumerate(types):
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, fc=SERIES[0] if t == "linear_attention" else SERIES[1], ec=SURFACE))
        ax.text(i + 0.46, -0.35, str(i), ha="center", fontsize=7, color=MUTED)
    ax.set_xlim(-0.2, len(types))
    ax.set_ylim(-0.6, 1.1)
    ax.axis("off")
    n_lin = sum(t == "linear_attention" for t in types)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label=f"Gated DeltaNet (linear attention) × {n_lin}"),
                       Patch(color=SERIES[1], label=f"gated full softmax attention × {len(types) - n_lin}")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    ax.set_title(f"{name}: {len(types)} layers, layer_types from config.json", loc="left", pad=26)
    save(fig, "p3_qwen35_layer_pattern")


def fig_sft_dpo(sft="outputs/nanolm/part2_ref/P2_sft_smollm2_360m", dpo="outputs/nanolm/part2_ref/P3_dpo_qwen05b_instruct"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    tr = [json.loads(l) for l in (Path(sft) / "train_log.jsonl").read_text().splitlines()]
    ev = [json.loads(l) for l in (Path(sft) / "eval_log.jsonl").read_text().splitlines()]
    m = json.loads((Path(sft) / "metrics.json").read_text())
    ax = axes[0]
    ax.plot([d["step"] for d in tr], [d["loss"] for d in tr], color=SERIES[0], label="train answer loss")
    ax.plot([0] + [d["step"] for d in ev], [m["before_val_response_nll"]] + [d["val_response_nll"] for d in ev],
            color=SERIES[1], marker="o", markersize=5, label="held-out answer loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("nats / answer token")
    ax.legend(fontsize=8)
    ax.set_title("SFT (LoRA) of SmolLM2-360M on Alpaca", loc="left")
    tr = [json.loads(l) for l in (Path(dpo) / "train_log.jsonl").read_text().splitlines()]
    ev = [json.loads(l) for l in (Path(dpo) / "eval_log.jsonl").read_text().splitlines()]
    m = json.loads((Path(dpo) / "metrics.json").read_text())
    ax = axes[1]
    ax.plot([d["step"] for d in tr], [d["reward_acc"] for d in tr], color=SERIES[0], label="train-batch reward accuracy")
    ax.plot([0] + [d["step"] for d in ev], [m["before_val_reward_acc"]] + [d["val_reward_acc"] for d in ev],
            color=SERIES[1], marker="o", markersize=5, label="held-out reward accuracy")
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.text(1, 0.505, "chance", fontsize=7.5, color=MUTED)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("optimizer step")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("DPO (LoRA) of Qwen2.5-0.5B-Instruct on UltraFeedback", loc="left")
    save(fig, "p3_sft_dpo_curves")


CHANCE = {"mmlu": 0.25, "arc_easy": 0.25, "arc_challenge": 0.25, "hellaswag": 0.25, "winogrande": 0.5,
          "piqa": 0.5, "gpqa_diamond_zeroshot": 0.25, "gsm8k": 0.0, "ifeval": 0.0}


def fig_benchmarks(root="outputs/nanolm/benchmarks/small_models"):
    import pandas as pd
    from .benchmarks import collect
    df = pd.DataFrame(collect(Path(root)))
    pick = {"mmlu": "acc", "arc_challenge": "acc_norm", "hellaswag": "acc_norm", "gpqa_diamond_zeroshot": "acc_norm",
            "gsm8k": "exact_match_flexible-extract", "ifeval": "prompt_level_strict_acc"}
    df = df[df.apply(lambda r: pick.get(r.task) == r.metric, axis=1)]
    tasks = [t for t in pick if t in set(df.task)]
    models = [m for m in ["gpt2-124M", "SmolLM2-360M", "SmolLM2-360M+ourSFT", "SmolLM2-360M-Instruct", "Qwen2.5-0.5B",
                          "Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Inst+ourDPO", "Qwen3-0.6B-Base", "Qwen3-0.6B",
                          "Qwen3.5-0.8B-Base", "Qwen3.5-0.8B"] if m in set(df.model)]
    ncol = 3
    nrow = math.ceil(len(tasks) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.6 * nrow), squeeze=False)
    for k, t in enumerate(tasks):
        ax = axes[k // ncol][k % ncol]
        d = df[df.task == t].set_index("model")
        vals = [d.value.get(m, np.nan) for m in models]
        ax.barh(range(len(models)), vals, color=SERIES[0], height=0.65)
        for j, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(v + 0.01, j, f"{100 * v:.1f}", va="center", fontsize=7, color=INK2)
        ch = CHANCE.get(t, 0)
        if ch:
            ax.axvline(ch, color=SERIES[7], ls="--", lw=1)
            ax.text(ch, len(models) - 0.3, " chance", color=SERIES[7], fontsize=7)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models if k % ncol == 0 else [""] * len(models), fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{t}  ({pick[t]})", loc="left", fontsize=9.5)
    for k in range(len(tasks), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Small open models on release-style benchmarks (lm-evaluation-harness, our runs)", x=0.01,
                 ha="left", fontsize=11)
    save(fig, "p3_benchmarks")


# =========================================================================== Part 4
def fig_golf_ablation():
    import pandas as pd
    rows = []
    for d in Path("outputs/nanolm/sweeps/golf_ablation").iterdir():
        f = d / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            rows.append((m["name"], m["seed"], m["val_bpb"]))
    df = pd.DataFrame(rows, columns=["name", "seed", "bpb"])
    base = df[df.name == "baseline"].set_index("seed").bpb
    g = []
    for n, d in df.groupby("name"):
        if n == "baseline":
            continue
        delta = d.set_index("seed").bpb - base
        g.append((n, delta.mean(), delta.min(), delta.max()))
    g.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, (n, mu, lo, hi) in enumerate(g):
        col = SERIES[0] if mu < -0.0034 else (SERIES[7] if mu > 0.0034 else AXIS)
        ax.barh(j, mu, color=col, height=0.6)
        ax.plot([lo, hi], [j, j], color=INK2, lw=1)
        ax.text(min(mu, 0) - 0.001, j, f"{mu:+.4f}", ha="right", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.axvspan(-0.0034, 0.0034, color="#f0efec", zorder=0)
    ax.text(0.0, len(g) - 0.2, "±2× seed noise", ha="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([r[0] for r in g], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Δ val bpb vs. golf baseline (negative = better); whisker = the two seeds")
    ax.grid(axis="y", visible=False)
    ax.set_title("Parameter-golf techniques, one at a time (lab scale, 131M tokens)", loc="left")
    save(fig, "p4_golf_ablation")


FIGS = {"positions": fig_positions, "data_window": fig_data_window, "tokenization": fig_tokenization, "attention": fig_attention,
        "temperature": fig_temperature, "char_models": fig_char_models, "tokenizer_curves": fig_tokenizer_curves,
        "lr_schedules": fig_lr_schedules, "muon": fig_muon, "recipe": fig_recipe, "model_anatomy": fig_model_anatomy,
        "layer_pattern": fig_layer_pattern, "sft_dpo": fig_sft_dpo, "benchmarks": fig_benchmarks,
        "golf_ablation": fig_golf_ablation}


def main(argv=None):
    ap = argparse.ArgumentParser(description="regenerate tutorial figures")
    ap.add_argument("figure", help="figure name, 'all', or 'list'")
    ap.add_argument("--run", default=None, help="run dir (attention / temperature / data_window)")
    ap.add_argument("--text", default=None)
    args = ap.parse_args(argv)
    style()
    torch.manual_seed(0)
    if args.figure == "list":
        print("\n".join(FIGS))
        return
    names = list(FIGS) if args.figure == "all" else [args.figure]
    for n in names:
        kw = {}
        if args.run and n in ("attention", "temperature", "data_window"):
            kw["run_dir"] = args.run
        if args.text and n == "attention":
            kw["text"] = args.text
        try:
            FIGS[n](**kw)
        except Exception as e:                  # one missing input must not stop the others
            print(f"[viz] skipped {n}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
