"""llmtrain entry point: one config file -> data, training with the chosen backend, the common evaluation.

    python -m llmtrain -c llmtrain/configs/sft_alpaca_trl.yaml                 # train
    python -m llmtrain -c llmtrain/configs/sft_alpaca_trl.yaml backend=native  # same experiment, other backend
    python -m llmtrain -c ... --dry-run          # write data + backend config only (AutoModel: the recipe YAML)
    python -m llmtrain --list                    # which method runs on which backend; data presets

Every run directory gets config.yaml (the resolved spec), meta.json (git commit, seed, library versions,
GPU, command), data/{train,val}.jsonl (the exact examples), the backend's own config/logs, and metrics.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from datetime import datetime
from importlib import metadata

from . import data as D
from .config import BACKENDS, METHODS, load_spec, save_config

SUPPORT = {"native": ("sft", "dpo"), "trl": ("sft", "dpo", "grpo", "opd"), "automodel": ("sft", "kd", "recipe"),
           "nemo_rl": ("grpo", "opd", "recipe")}
NOTES = {"native": "nanolm's hand-written loops (reference implementation)",
         "trl": "Hugging Face TRL trainers",
         "automodel": "NVIDIA NeMo AutoModel recipes (FSDP2/TP/CP; official recipes via automodel.recipe_yaml)",
         "nemo_rl": "NVIDIA NeMo RL (Ray + vLLM rollouts; own uv env in nemo_rl.dir; NeMo Gym environments)"}


def validate(spec) -> None:
    if spec.method not in METHODS:
        raise ValueError(f"method {spec.method!r} not in {METHODS}")
    if spec.backend not in BACKENDS:
        raise ValueError(f"backend {spec.backend!r} not in {BACKENDS}")
    if spec.method not in SUPPORT[spec.backend]:
        ok = [b for b, ms in SUPPORT.items() if spec.method in ms]
        raise ValueError(f"method {spec.method!r} is not available on backend {spec.backend!r}; use backend in {ok}")
    if spec.model.peft not in ("none", "lora"):
        raise ValueError("model.peft must be none or lora")
    if spec.method in ("opd", "kd") and not spec.model.teacher:
        raise ValueError(f"method {spec.method} needs model.teacher")
    if spec.launch.target not in ("local", "dli"):
        raise ValueError("launch.target must be local or dli")
    if spec.launch.target == "dli" and (spec.backend, spec.method) not in (("automodel", "sft"), ("automodel", "recipe")):
        raise ValueError("launch.target=dli trains through the workshop's AutoModel container: backend=automodel, "
                         "method sft or recipe")


def versions() -> dict:
    out = {}
    for pkg in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "nemo-automodel"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def run(spec) -> dict:
    validate(spec)
    if spec.launch.target == "dli":                 # NVIDIA DLI workshop VM: no torch needed in this container
        from .dli import run_dli
        return run_dli(spec)
    import torch
    from nanolm.device import seed_everything
    from nanolm.train import git_info, make_run_dir
    torch.backends.cuda.enable_cudnn_sdp(spec.train.cudnn_sdpa)       # see TrainSpec.cudnn_sdpa
    seed_everything(spec.seed)
    run_dir = make_run_dir(spec)
    save_config(spec, run_dir / "config.yaml")
    (run_dir / "meta.json").write_text(json.dumps({
        "name": spec.name, "seed": spec.seed, "method": spec.method, "backend": spec.backend,
        "date": datetime.now().isoformat(timespec="seconds"), "git": git_info(), "host": socket.gethostname(),
        "platform": platform.platform(), "python": platform.python_version(), "gpu": gpu_name(),
        "versions": versions(), "command": " ".join(sys.argv)}, indent=1))
    print(f"[run] {run_dir}\n[run] method={spec.method} backend={spec.backend} model={spec.model.name}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.model.name)
    use_our_data = not ((spec.method == "recipe" and (spec.backend == "nemo_rl" or not spec.automodel.use_our_data))
                        or (spec.backend == "nemo_rl" and spec.nemo_rl.gym_env))    # Gym environments bring their own tasks
    kind = "recipe"
    if use_our_data:
        train, val, kind = D.build(spec.data, tok)
        if kind not in D.METHOD_KINDS[spec.method]:
            raise ValueError(f"method {spec.method} cannot train on {kind!r} records (preset {spec.data.preset})")
        D.write_jsonl(train, run_dir / "data/train.jsonl")
        D.write_jsonl(val, run_dir / "data/val.jsonl")
        print(f"[data] {spec.data.preset}: {len(train)} train / {len(val)} val ({kind}, max_len {spec.data.max_len})")

    metrics = {"name": spec.name, "seed": spec.seed, "method": spec.method, "backend": spec.backend,
               "model": spec.model.name, "teacher": spec.model.teacher, "peft": spec.model.peft, "data_kind": kind}
    if spec.launch.dry_run and spec.backend not in ("automodel", "nemo_rl"):
        print("[dry-run] data written; not training")
        (run_dir / "metrics.json").write_text(json.dumps({**metrics, "dry_run": True}, indent=1))
        return metrics

    if spec.backend == "native":
        from .backends import native as backend
    elif spec.backend == "trl":
        from .backends import trl_backend as backend
    elif spec.backend == "automodel":
        from .backends import automodel_backend as backend
    else:
        from .backends import nemo_rl_backend as backend
    t0 = datetime.now()
    result = backend.train(spec, run_dir, kind)
    metrics.update({k: v for k, v in result.items() if k != "backend_metrics"})
    metrics["backend_metrics"] = result.get("backend_metrics", {})
    metrics["train_wall_sec"] = round((datetime.now() - t0).total_seconds(), 1)

    if spec.eval.enabled and use_our_data and result.get("final_model"):
        from .evaluate import evaluate
        metrics.update(evaluate(spec, run_dir, kind, result["final_model"], result.get("is_adapter", False)))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items() if k.startswith(("before_", "after_"))}
    print(f"[done] {run_dir}\n[eval] {shown}")
    return metrics


def list_support() -> None:
    print("method x backend:")
    for b, ms in SUPPORT.items():
        print(f"  {b:10s} {', '.join(ms):24s} {NOTES[b]}")
    print("data presets (kind, source):")
    for p, (kind, hf, cfg, tr, va) in D.PRESETS.items():
        print(f"  {p:14s} {kind:5s} {hf or '(local jsonl: data.path, data.columns)'} {cfg}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="llmtrain: one config -> SFT / DPO / GRPO / OPD / KD with native, TRL or NeMo AutoModel")
    ap.add_argument("--config", "-c")
    ap.add_argument("--dry-run", action="store_true", help="write data and the backend config, do not train")
    ap.add_argument("--list", action="store_true", help="print the method x backend table and data presets")
    ap.add_argument("overrides", nargs="*", help="key.sub=value overrides, e.g. backend=native train.steps=50")
    args = ap.parse_args(argv)
    if args.list:
        list_support()
        return
    if not args.config:
        ap.error("--config is required (or --list)")
    sys.stdout.reconfigure(line_buffering=True)
    spec = load_spec(args.config, args.overrides + (["launch.dry_run=true"] if args.dry_run else []))
    run(spec)


if __name__ == "__main__":
    main()
