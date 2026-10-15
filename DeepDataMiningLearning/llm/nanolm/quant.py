"""Post-training quantization + compression: the "16 MB artifact" half of parameter golf.

Pipeline (same idea as the golf baseline, extended to k-bit):
  1. every large 2-D float tensor -> per-row symmetric int-k (k = 8, 6, 5, 4) with a per-row
     clip chosen by MSE search over a few percentiles; scales stored as fp16;
  2. small tensors (norm weights, gains, ...) -> fp16 passthrough;
  3. torch.save -> zlib / lzma compress -> count bytes;
  4. de-quantize and re-evaluate ("round-trip" bpb) -- the number that actually counts.

k-bit values are stored one per int8 byte; the entropy coder (lzma) then squeezes out the unused
high bits, which is why int6 + lzma lands close to 6 bits/weight without manual bit packing.
"""
from __future__ import annotations

import io
import lzma
import zlib

import torch

KEEP_FLOAT_MAX_NUMEL = 65_536
CLIP_PERCENTILES = (1.0, 0.99999, 0.9999, 0.999, 0.995)


def _quant_rows(w: torch.Tensor, bits: int):
    qmax = 2 ** (bits - 1) - 1
    w = w.float()
    absw = w.abs()
    best_q, best_s, best_err = None, None, None
    for pct in CLIP_PERCENTILES:
        if pct >= 1.0:
            clip = absw.amax(dim=1)
        else:  # quantile on a row sample keeps this fast for wide matrices
            sample = absw if absw.shape[1] <= 4096 else absw[:, torch.randperm(absw.shape[1])[:4096]]
            clip = torch.quantile(sample, pct, dim=1)
        scale = (clip / qmax).clamp_min(1e-8)
        q = torch.clamp(torch.round(w / scale[:, None]), -qmax, qmax)
        err = ((q * scale[:, None] - w) ** 2).sum(dim=1)
        if best_err is None:
            best_q, best_s, best_err = q, scale, err
        else:
            better = err < best_err
            best_q[better], best_s[better], best_err[better] = q[better], scale[better], err[better]
    return best_q.to(torch.int8), best_s.to(torch.float16)


def quantize_state_dict(sd: dict, bits: int) -> dict:
    q, scales, passthrough, dtypes, aliases = {}, {}, {}, {}, {}
    seen: dict[int, str] = {}
    for name, t in sd.items():
        if t.data_ptr() in seen and t.numel() > 0:      # tied weights: store once
            aliases[name] = seen[t.data_ptr()]
            continue
        seen[t.data_ptr()] = name
        t = t.detach().cpu()
        if not t.is_floating_point():
            passthrough[name] = t
        elif t.ndim == 2 and t.numel() > KEEP_FLOAT_MAX_NUMEL:
            q[name], scales[name] = _quant_rows(t, bits)
            dtypes[name] = str(t.dtype).removeprefix("torch.")
        else:
            passthrough[name] = t.to(torch.float16)
            dtypes[name] = str(t.dtype).removeprefix("torch.")
    return {"bits": bits, "q": q, "scales": scales, "passthrough": passthrough, "dtypes": dtypes,
            "aliases": aliases}


def dequantize_state_dict(obj: dict) -> dict:
    out = {}
    for name, q in obj["q"].items():
        out[name] = (q.float() * obj["scales"][name].float()[:, None]).to(getattr(torch, obj["dtypes"][name]))
    for name, t in obj["passthrough"].items():
        dt = obj["dtypes"].get(name)
        out[name] = t.to(getattr(torch, dt)) if dt else t
    for name, src in obj.get("aliases", {}).items():
        out[name] = out[src]
    return out


def compress(obj: dict, method: str = "lzma") -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    raw = buf.getvalue()
    if method == "zlib":
        return zlib.compress(raw, level=9)
    if method == "lzma":
        return lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    raise ValueError(f"unknown compressor {method!r}")


def decompress(blob: bytes, method: str = "lzma") -> dict:
    raw = zlib.decompress(blob) if method == "zlib" else lzma.decompress(blob)
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


def export_artifact(model, bits: int, method: str) -> tuple[bytes, dict]:
    """Quantize + compress the model. Returns (blob, de-quantized state_dict for round-trip eval).
    Tied weights appear once in state_dict() under both names; torch.save stores shared storage once."""
    sd = {k: v for k, v in model.state_dict().items()}
    obj = quantize_state_dict(sd, bits)
    blob = compress(obj, method)
    return blob, dequantize_state_dict(decompress(blob, method))
