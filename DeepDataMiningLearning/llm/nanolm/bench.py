"""How long will this config take on MY machine, and will it fit in memory?

    python -m nanolm.bench -c nanolm/configs/laptop/L4_gpt_fineweb.yaml
    python -m nanolm.bench -c nanolm/configs/laptop/L4_gpt_fineweb.yaml train.batch_size=8   # try a smaller batch

Runs a few real training steps (forward + backward + optimizer) on random tokens, then reports
tokens/second, peak memory, and the projected wall-clock time of the full config. No data needed.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from .config import load_config
from .device import autocast_ctx, get_device, memory_mb, synchronize
from .models import build_model, count_params
from .optim import build_optimizers


def flops_per_token(model, cfg) -> float:
    """~6 x params (forward+backward matmuls) + attention scores; a rough but standard estimate."""
    n = count_params(model, exclude_embeddings=True)
    attn = 12 * cfg.model.n_layers * cfg.model.dim * cfg.train.seq_len if cfg.model.arch == "gpt" else 0
    return 6 * n + attn


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--steps", type=int, default=10, help="timed steps (after 3 warm-up steps)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if cfg.model.vocab_size <= 0:
        cfg.model.vocab_size = 1024 if cfg.data.source == "shards" else 65
    dinfo = get_device(cfg.device, cfg.precision)
    if dinfo.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dinfo.device)
    model = build_model(cfg.model).to(dinfo.device)
    model.recur_active = bool(cfg.model.recur_layers)        # worst case: recurrence on
    opts = build_optimizers(model, cfg.train, dinfo)
    fwd = torch.compile(model) if (cfg.train.compile and dinfo.compile_ok) else model
    tc = cfg.train
    B, T, A = tc.batch_size, tc.seq_len, tc.grad_accum
    V = cfg.model.vocab_size

    def step():
        for _ in range(A):
            x = torch.randint(0, V, (B, T), device=dinfo.device)
            with autocast_ctx(dinfo):
                logits = fwd(x)
            F.cross_entropy(logits.float().view(-1, V), x.view(-1)).div(A).backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)

    print(f"[bench] {cfg.name} on {dinfo}: {count_params(model) / 1e6:.2f}M params, "
          f"{B}x{T}x{A} = {B * T * A:,} tokens/step")
    for _ in range(3):
        step()
    synchronize(dinfo)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    synchronize(dinfo)
    dt = (time.perf_counter() - t0) / args.steps
    tps = B * T * A / dt
    fpt = flops_per_token(model, cfg)
    total = tc.max_steps * dt
    print(f"[bench] {dt * 1000:.0f} ms/step | {tps / 1e3:.1f}k tokens/s | ~{tps * fpt / 1e12:.2f} TFLOP/s achieved")
    mem = memory_mb(dinfo)
    if dinfo.type == "cuda":
        print(f"[bench] peak GPU memory {mem / 1024:.2f} GB")
    elif dinfo.type == "mps":
        print(f"[bench] MPS memory (Metal driver, incl. cache) {mem / 1024:.2f} GB")
    print(f"[bench] projected training time for max_steps={tc.max_steps}: {total / 60:.1f} min "
          f"(+ periodic eval)" + (f"; capped at max_time_sec={tc.max_time_sec:.0f}s" if tc.max_time_sec else ""))
    if tc.max_time_sec and total > tc.max_time_sec:
        print(f"[bench] -> on this machine the time cap ends training at ~step "
              f"{int(tc.max_time_sec / dt)} of {tc.max_steps}; the LR schedule adapts automatically.")


if __name__ == "__main__":
    main()
