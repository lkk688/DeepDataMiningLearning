"""Post-training analysis of a saved run -- no retraining needed.

    python -m nanolm.compress outputs/nanolm/sweeps/golf_ablation/baseline_s1337 \
        --bits 8 6 5 4 --strides 0 256 64 --eval-tokens 4194304

For every (quantization bits x compressor) it reports compressed size and round-trip bpb; for every
sliding-window stride it reports bpb and eval cost. Results go to <run_dir>/compress.json.
These are exactly the two "free" levers of parameter golf: how you EVALUATE and how you STORE.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .config import from_dict, RunConfig
from .data import TokenData, byte_lut_tensors
from .device import get_device
from .evaluate import evaluate
from .models import build_model
from .quant import export_artifact


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--bits", type=int, nargs="*", default=[8, 6, 5, 4])
    ap.add_argument("--compressors", nargs="*", default=["zlib", "lzma"])
    ap.add_argument("--strides", type=int, nargs="*", default=[0, 256, 64])
    ap.add_argument("--eval-tokens", type=int, default=4_194_304)
    ap.add_argument("--seq-len", type=int, default=0, help="eval context (0 = training seq_len)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    cfg = from_dict(RunConfig, ckpt["config"])
    dinfo = get_device(args.device, cfg.precision)
    model = build_model(cfg.model).to(dinfo.device)
    model.recur_active = bool(cfg.model.recur_layers)
    model.load_state_dict(ckpt["model"])
    data = TokenData(ckpt["data_meta"])
    luts = byte_lut_tensors(ckpt["data_meta"], dinfo.device)
    T = args.seq_len or cfg.eval.eval_seq_len or cfg.train.seq_len
    val = data.val_windows(T, args.eval_tokens)
    bs = cfg.eval.batch_size
    out = {"run_dir": str(run), "eval_seq_len": T, "eval_tokens": args.eval_tokens, "stride": [], "quant": []}

    for s in args.strides:
        t0 = time.perf_counter()
        r = evaluate(model, val, T, bs, luts, dinfo, stride=s)
        out["stride"].append({"stride": s, "bpb": r["bpb"], "loss": r["loss"], "sec": time.perf_counter() - t0})
        print(f"stride {s:4d}: val_bpb {r['bpb']:.4f}  ({time.perf_counter() - t0:.1f}s)")

    ref = out["stride"][0]["bpb"] if out["stride"] else evaluate(model, val, T, bs, luts, dinfo)["bpb"]
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    n_params = sum(p.numel() for p in model.parameters())
    for bits in args.bits:
        for comp in args.compressors:
            blob, deq = export_artifact(model, bits, comp)
            model.load_state_dict({k: v.to(dinfo.device) for k, v in deq.items()})
            r = evaluate(model, val, T, bs, luts, dinfo)
            model.load_state_dict(sd)
            row = {"bits": bits, "compressor": comp, "bytes": len(blob), "bits_per_param": 8 * len(blob) / n_params,
                   "bpb": r["bpb"], "gap": r["bpb"] - ref}
            out["quant"].append(row)
            print(f"int{bits} + {comp:4s}: {len(blob) / 1e6:7.3f} MB ({row['bits_per_param']:.2f} bits/param) "
                  f"| round-trip bpb {r['bpb']:.4f} (gap {row['gap']:+.4f})")
    (run / "compress.json").write_text(json.dumps(out, indent=1))
    print(f"-> {run / 'compress.json'}")


if __name__ == "__main__":
    main()
