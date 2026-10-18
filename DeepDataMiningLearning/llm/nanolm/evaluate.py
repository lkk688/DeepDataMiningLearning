"""Evaluation: loss, bits-per-byte, sliding-window eval, and text sampling.

bits-per-byte (bpb) = (sum of -log p(target) in nats) / ln 2 / (number of UTF-8 bytes predicted)
It answers "how many bits does this model need to compress the validation text?" and, unlike
per-token loss or perplexity, it is comparable across tokenizers (char vs. BPE vs. bytes).
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .data import count_bytes
from .device import autocast_ctx


@torch.no_grad()
def evaluate(model, val_tokens, seq_len: int, batch_size: int, luts, dinfo,
             stride: int = 0, rank: int = 0, world: int = 1) -> dict:
    """Score every target token of `val_tokens` exactly once.

    stride == 0      : non-overlapping windows; token t is predicted from the context since the
                       start of its window (0 .. seq_len-1 tokens, avg ~seq_len/2).
    0 < stride < T   : sliding windows; after the first window every scored token has at least
                       seq_len - stride tokens of context.  Costs ~seq_len/stride x more compute.
    """
    was_training = model.training
    model.eval()
    dev = dinfo.device
    toks = torch.as_tensor(val_tokens.astype("int64") if hasattr(val_tokens, "astype") else val_tokens)
    N = toks.numel() - 1                       # number of (input, target) pairs
    T = seq_len
    # Each job = (window_start, first_scored_offset_within_window)
    if stride <= 0 or stride >= T:
        jobs = [(s, 0) for s in range(0, N - T + 1, T)]
    else:
        jobs = [(0, 0)]
        s = stride
        while s + T <= N:
            jobs.append((s, T - stride))
            s += stride
        last_end = jobs[-1][0] + T
        if last_end < N:                       # final partial window: score only the uncovered tail
            s = N - T
            jobs.append((s, last_end - s))
    jobs = jobs[rank::world]

    # Accumulate in float64 for exact totals. MPS has no float64, so on a Mac the per-batch values
    # are moved to the CPU first (a few hundred KB per batch, negligible next to the forward pass).
    acc_dev = torch.device("cpu") if dev.type == "mps" else dev
    sum64 = lambda t: t.to(acc_dev).double().sum()
    nats = torch.zeros((), device=acc_dev, dtype=torch.float64)
    n_tok = torch.zeros((), device=acc_dev, dtype=torch.float64)
    n_byte = torch.zeros((), device=acc_dev, dtype=torch.float64)
    n_correct = torch.zeros((), device=acc_dev, dtype=torch.float64)
    for b in range(0, len(jobs), batch_size):
        chunk = jobs[b: b + batch_size]
        win = torch.stack([toks[s: s + T + 1] for s, _ in chunk]).to(dev)
        x, y = win[:, :-1], win[:, 1:]
        with autocast_ctx(dinfo):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                               reduction="none").view_as(y)
        mask = torch.zeros_like(y, dtype=torch.bool)
        for r, (_, off) in enumerate(chunk):
            mask[r, off:] = True
        nats += sum64(loss[mask])
        n_tok += sum64(mask)
        n_byte += sum64(count_bytes(x, y, luts)[mask])
        n_correct += sum64((logits.argmax(-1) == y)[mask])
    if world > 1:
        for t in (nats, n_tok, n_byte, n_correct):
            dist.all_reduce(t)
    if was_training:
        model.train()
    loss = (nats / n_tok).item()
    return {
        "loss": loss,
        "bpb": (nats / math.log(2) / n_byte).item(),
        "ppl": math.exp(min(loss, 50)),
        "acc": (n_correct / n_tok).item(),
        "tokens": int(n_tok.item()),
        "bytes": int(n_byte.item()),
        "bytes_per_token": (n_byte / n_tok).item(),
    }


def sample_next(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 0, top_p: float = 1.0,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """Pick the next token from logits [B, V].

    temperature : < 1 sharpens the distribution, > 1 flattens it, 0 = greedy (argmax)
    top_k       : keep only the k most likely tokens
    top_p       : nucleus sampling -- keep the smallest set of tokens whose total probability >= p
    """
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
    if top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, -1).cumsum(-1)
        drop = cum - torch.softmax(sorted_logits, -1) >= top_p     # tokens entirely beyond the nucleus
        sorted_logits = sorted_logits.masked_fill(drop, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, idx, sorted_logits)
    probs = torch.softmax(logits, -1).cpu()
    return torch.multinomial(probs, 1, generator=generator).to(logits.device)


@torch.no_grad()
def generate(model, tok, dinfo, prompt: str, max_new: int, ctx_len: int,
             temperature: float = 0.8, top_k: int = 50, seed: int = 0, top_p: float = 1.0,
             stream=None) -> str:
    """Autoregressive sampling. The model only ever sees the last `ctx_len` tokens."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = tok.encode(prompt) if prompt else [tok.eot_id if tok.eot_id is not None else 0]
    idx = torch.tensor([ids], dtype=torch.long, device=dinfo.device)
    n0 = idx.size(1)
    for _ in range(max_new):
        with autocast_ctx(dinfo):
            logits = model(idx[:, -ctx_len:])[:, -1, :]
        nxt = sample_next(logits, temperature, top_k, top_p, g)
        idx = torch.cat([idx, nxt], dim=1)
        if stream is not None:
            stream(tok.decode(idx[0, n0:].tolist()))
    return tok.decode(idx[0].tolist())
