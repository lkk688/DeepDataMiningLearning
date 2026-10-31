"""launch.target=dli: run an llmtrain experiment on an NVIDIA DLI workshop VM.

On that VM the notebook container (JupyterLab) has no direct GPU access for training: a long-lived orchestrator starts
the workshop's own containers (`nemo-train` for NeMo AutoModel, `vllm-serve` for inference) when the `workshop` helper
package asks it to. So llmtrain does not launch AutoModel itself there; it does what the workshop notebooks do, from
one llmtrain config:

    1. data       our presets / your JSONL -> chat records {"messages": [...]} (the workshop's ChatDataset format),
                  written under dli.root (/scratch: shared with the training container; /workspace is read-only)
    2. config     the workshop's own AutoModel YAML (known to work on that VM: model path, LoRA, batch, FSDP) with our
                  data, steps, learning rate, seed and checkpoint directory; anything else via backend_overrides
    3. train      env.stop_vllm(); workshop.train.train(config_path, run_id)        (vLLM holds the GPUs otherwise)
    4. evaluate   env.start_vllm(); env.load_adapter(run_id, path=<adapter>); the base model and the adapter answer
                  the same held-out prompts through the vLLM server: qa accuracy (dli.match), exact match for sft,
                  and the sample prompts. Response NLL (the local evaluation) needs the weights in this process: not here.

    python -m llmtrain -c llmtrain/configs/dli_sft_lora.yaml data.path=/scratch/my_data.jsonl
    python dli_probe.py                                   # first: what the VM provides (llmtrain/dli_probe.py)

Needs only the standard library + PyYAML in the notebook container (transformers, if present, enables the
data.max_len filter). Run directory = <dli.root>/<date>_<name>, with the usual config.yaml / meta.json / data /
metrics.json, plus automodel_config.yaml, train_summary.json and the before / after generations.
"""
from __future__ import annotations

import inspect
import json
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from nanolm.config import deep_merge

from . import data as D
from . import serve
from .config import save_config

WORKSHOP_CONFIGS = Path("/workspace/configs")
WORKSHOP_SCRATCH = Path("/scratch")
CHAT_T = "nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset"


def git_info() -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain"))}


def base_yaml(spec) -> Path:
    """The workshop YAML to start from: dli.base_yaml, else automodel.recipe_yaml, else the first sft_lora*.yaml."""
    src = spec.dli.base_yaml or spec.automodel.recipe_yaml
    if src:
        return Path(src)
    found = sorted(WORKSHOP_CONFIGS.glob("sft_lora*.yaml"))
    if not found:
        raise FileNotFoundError(f"no dli.base_yaml given and no {WORKSHOP_CONFIGS}/sft_lora*.yaml; run dli_probe.py "
                                "and set dli.base_yaml to one of the workshop's AutoModel configs")
    return found[0]


def resolve_model(spec, base: dict) -> str:
    """model.name 'dli:base' = the model of the workshop YAML (e.g. a checkpoint under /workspace/checkpoints)."""
    if spec.model.name == "dli:base":
        return base.get("model", {}).get("pretrained_model_name_or_path", "")
    return spec.model.name


def tokenizer(model: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model)
    except Exception as e:                       # no transformers in the notebook container, or the path is not visible
        print(f"[dli] no tokenizer for {model!r} here ({type(e).__name__}): data.max_len is not enforced while "
              "selecting examples; the training YAML's seq_length truncates instead")
        return None


def chat_record(rec: dict, system: str) -> dict:
    msgs = [{"role": "system", "content": system}] if system else []
    answer = rec.get("response", rec.get("answer", ""))
    return {"messages": msgs + [{"role": "user", "content": rec["prompt"]}, {"role": "assistant", "content": answer}]}


def write_chat_data(run_dir: Path, system: str) -> dict:
    paths = {}
    for split in ("train", "val"):
        rows = [chat_record(r, system) for r in D.read_jsonl(run_dir / f"data/{split}.jsonl")]
        p = run_dir / f"data/chat_{split}.jsonl"
        D.write_jsonl(rows, p)
        paths[split] = str(p.resolve())
    return paths


def _with_path(ds: dict | None, path: str) -> dict:
    """Point a dataset block at our file; a non-chat dataset block is replaced by the workshop's ChatDataset."""
    if ds and ds.get("_target_") == CHAT_T:
        return {**ds, "path_or_dataset_id": path}
    return {"_target_": CHAT_T, "path_or_dataset_id": path}


def build_config(spec, base: dict, run_dir: Path, paths: dict | None) -> dict:
    """Workshop YAML  <  our data / steps / lr / seed / checkpoint dir  <  backend_overrides."""
    cfg = json.loads(json.dumps(base))                                    # deep copy
    t = spec.train
    y: dict = {"checkpoint": {"checkpoint_dir": str((run_dir / "checkpoints").resolve())}}
    if spec.method == "sft":
        if t.steps > 0:                       # train.steps=0 keeps the workshop's own schedule (epochs)
            y["step_scheduler"] = {"max_steps": t.steps, "ckpt_every_steps": t.steps, "num_epochs": 1000}
        y["optimizer"] = {"lr": t.lr}
        if "rng" in cfg:
            y["rng"] = {"seed": spec.seed}
        if spec.model.name != "dli:base":
            y["model"] = {"pretrained_model_name_or_path": spec.model.name}
        if spec.model.peft == "none":
            cfg.pop("peft", None)
    if paths:
        cfg["dataset"] = _with_path(cfg.get("dataset"), paths["train"])
        if "validation_dataset" in cfg:
            cfg["validation_dataset"] = _with_path(cfg["validation_dataset"], paths["val"])
    return deep_merge(deep_merge(cfg, y), dict(spec.backend_overrides))


def find_adapter(run_dir: Path, run_id: str, summary: dict) -> tuple[str, bool]:
    """-> (path, is_adapter) of the newest checkpoint, wherever the workshop put it."""
    roots = [run_dir / "checkpoints", WORKSHOP_SCRATCH / run_id / "checkpoints"]
    for v in (summary or {}).values():                    # directories the workshop's summary mentions
        if isinstance(v, str) and v.startswith("/"):
            roots += [Path(v), Path(v) / "checkpoints"]
    ckpts = [p for r in roots if r.is_dir() for p in r.glob("epoch_*_step_*")]
    if not ckpts:
        return "", False
    mdir = max(ckpts, key=lambda p: int(p.name.rsplit("_", 1)[1])) / "model"
    if (mdir / "adapter_config.json").exists():
        return str(mdir), True
    cons = mdir / "consolidated"
    return (str(cons) if cons.exists() else str(mdir)), False


def train_curve(run_dir: Path, run_id: str) -> dict:
    for p in (run_dir / "checkpoints/training.jsonl", WORKSHOP_SCRATCH / run_id / "checkpoints/training.jsonl"):
        if p.exists():
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            loss = [r["loss"] for r in rows if "loss" in r]
            return {"train_log": str(p), "first_loss": loss[0] if loss else None, "last_loss": loss[-1] if loss else None,
                    "logged_steps": len(rows)}
    return {}


def served_base(url: str) -> str:
    ms = serve.models(url)
    roots = [m["id"] for m in ms if not m.get("parent")]
    if not roots:
        raise RuntimeError(f"vLLM at {url} serves no base model: {ms}")
    return roots[0]


def evaluate(spec, run_dir: Path, kind: str, run_id: str, adapter: str, is_adapter: bool) -> dict:
    from workshop import env
    if spec.dli.restart_vllm:
        env.start_vllm()
    url = env.vllm_url()
    base = served_base(url)

    def after() -> str:                        # LoRA: hot-load the adapter; full fine-tuning: serve the new weights
        if is_adapter:
            env.load_adapter(run_id, path=adapter)
            return run_id
        env.swap_vllm_to(adapter)
        return served_base(url)
    val = D.read_jsonl(run_dir / "data/val.jsonl")[: spec.eval.n_val]
    sysm, gen = spec.dli.system, spec.eval.gen_tokens
    out = {"eval_n_val": len(val), "eval_base_model": base}
    for tag, get_model in (("before", lambda: base), ("after", after)):
        model = get_model()
        if kind == "qa":
            res = serve.eval_qa(val, url, model, max_tokens=gen, workers=spec.dli.eval_workers, how=spec.dli.match, system=sysm)
            out[f"{tag}_val_answer_acc"] = res["val_answer_acc"]
            out[f"{tag}_eval_errors"] = res["errors"]
        rows = serve.generate(val, url, model, workers=spec.dli.eval_workers, system=sysm, temperature=0.0, max_tokens=gen)
        D.write_jsonl(rows, run_dir / f"val_generations_{tag}.jsonl")
        if kind == "sft":                      # meaningful when the response is a label or a short canonical answer
            ok = [serve.match((r["completions"] or [""])[0], r["response"], "exact") for r in rows]
            out[f"{tag}_val_exact_match"] = sum(ok) / max(len(ok), 1)
            out[f"{tag}_eval_errors"] = sum("error" in r for r in rows)
        samples = serve.generate([{"prompt": p} for p in spec.eval.prompts], url, model, system=sysm, max_tokens=gen)
        (run_dir / f"samples_{tag}.txt").write_text(
            "\n".join(f"### {r['prompt']}\n{(r['completions'] or [r.get('error', '')])[0]}\n" for r in samples))
    return out


def run_dli(spec) -> dict:
    random.seed(spec.seed)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(spec.run_dir or Path(spec.dli.root) / f"{stamp}_{spec.name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    save_config(spec, run_dir / "config.yaml")
    src = base_yaml(spec)
    base = yaml.safe_load(src.read_text()) or {}
    model = resolve_model(spec, base)
    (run_dir / "meta.json").write_text(json.dumps({
        "name": spec.name, "seed": spec.seed, "method": spec.method, "backend": spec.backend, "target": "dli",
        "date": datetime.now().isoformat(timespec="seconds"), "git": git_info(), "host": socket.gethostname(),
        "platform": platform.platform(), "python": platform.python_version(), "base_yaml": str(src), "model": model,
        "run_id": run_id, "command": " ".join(sys.argv)}, indent=1))
    print(f"[run] {run_dir}\n[run] target=dli method={spec.method} model={model} base_yaml={src}")

    kind, paths = "recipe", None
    if spec.method == "sft" or spec.automodel.use_our_data:
        train, val, kind = D.build(spec.data, tokenizer(model) if model else None)
        if kind not in ("sft", "qa"):
            raise ValueError(f"the DLI path trains SFT on sft / qa records, got {kind!r} (preset {spec.data.preset})")
        D.write_jsonl(train, run_dir / "data/train.jsonl")
        D.write_jsonl(val, run_dir / "data/val.jsonl")
        paths = write_chat_data(run_dir, spec.dli.system)
        print(f"[data] {spec.data.preset}: {len(train)} train / {len(val)} val ({kind})")
    cfg = build_config(spec, base, run_dir, paths)
    cfg_path = run_dir / "automodel_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    metrics = {"name": spec.name, "seed": spec.seed, "method": spec.method, "backend": spec.backend, "target": "dli",
               "model": model, "base_yaml": str(src), "run_id": run_id, "data_kind": kind}
    if spec.launch.dry_run:
        print(f"[dry-run] wrote {cfg_path}; not training")
        (run_dir / "metrics.json").write_text(json.dumps({**metrics, "dry_run": True}, indent=1))
        return metrics

    from workshop import env
    from workshop import train as wtrain
    env.stop_vllm()                                     # the training container needs the GPUs vLLM holds
    kw = {}
    params = inspect.signature(wtrain.train).parameters
    for name in ("nproc_per_node", "nproc"):
        if name in params and spec.launch.nproc_per_node > 1:
            kw[name] = spec.launch.nproc_per_node
    t0 = time.time()
    summary = wtrain.train(config_path=str(cfg_path.resolve()), run_id=run_id, **kw)
    summary = summary if isinstance(summary, dict) else {"returned": repr(summary)}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    adapter, is_adapter = find_adapter(run_dir, run_id, summary)
    metrics.update({"final_model": adapter, "is_adapter": is_adapter, "train_wall_sec": round(time.time() - t0, 1),
                    "backend_metrics": {**{k: v for k, v in summary.items() if isinstance(v, (int, float, str))},
                                        **train_curve(run_dir, run_id)}})
    print(f"[train] {summary}\n[train] checkpoint: {adapter or 'NOT FOUND'}")
    if spec.eval.enabled and adapter and kind in ("sft", "qa"):
        metrics.update(evaluate(spec, run_dir, kind, run_id, adapter, is_adapter))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items() if k.startswith(("before_", "after_"))}
    print(f"[done] {run_dir}\n[eval] {shown}")
    return metrics
