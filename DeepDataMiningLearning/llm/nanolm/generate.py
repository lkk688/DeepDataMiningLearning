"""Sample text from a trained nanolm checkpoint (replaces the old inference_utils.py).

    python -m nanolm.generate outputs/nanolm/<run_dir> --prompt "ROMEO:" --tokens 300
    python -m nanolm.generate <run_dir> --temperature 0 --prompt "To be"        # greedy
    python -m nanolm.generate <run_dir> --top-p 0.9 --top-k 0 --interactive    # type prompts
    python -m nanolm.generate <run_dir> --compare-temperatures                  # 0 / 0.5 / 1.0 / 1.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import RunConfig, from_dict
from .data import build_tokenizer
from .device import get_device
from .evaluate import generate
from .models import build_model


def load_run(run_dir: str, device: str = "auto"):
    ckpt = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    cfg = from_dict(RunConfig, ckpt["config"])
    dinfo = get_device(device, cfg.precision)
    model = build_model(cfg.model).to(dinfo.device)
    model.recur_active = bool(cfg.model.recur_layers)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = build_tokenizer(ckpt["data_meta"]["spec"], meta=ckpt["data_meta"])
    return cfg, model, tok, dinfo, ckpt["data_meta"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="sample from a nanolm checkpoint")
    ap.add_argument("run_dir")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--compare-temperatures", action="store_true")
    args = ap.parse_args(argv)
    cfg, model, tok, dinfo, _ = load_run(args.run_dir, args.device)
    ctx = cfg.eval.eval_seq_len or cfg.train.seq_len
    print(f"[generate] {cfg.name}: {cfg.model.arch}, tokenizer {tok.spec}, context {ctx}, device {dinfo}")

    def run(prompt, temperature):
        return generate(model, tok, dinfo, prompt, args.tokens, ctx, temperature, args.top_k, args.seed, args.top_p)

    if args.compare_temperatures:
        for t in (0.0, 0.5, 1.0, 1.5):
            print(f"\n===== temperature {t} =====\n{run(args.prompt, t)}")
    elif args.interactive:
        while True:
            try:
                prompt = input("\nprompt> ")
            except EOFError:
                break
            print(run(prompt, args.temperature))
    else:
        print(run(args.prompt, args.temperature))


if __name__ == "__main__":
    main()
