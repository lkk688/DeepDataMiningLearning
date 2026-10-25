"""automodel backend: NVIDIA NeMo AutoModel recipes (FSDP2 / TP / CP, Hugging Face checkpoints in and out).

We do not re-implement AutoModel. We write a recipe YAML -- either from scratch from the llmtrain spec, or
starting from an official NVIDIA recipe (`automodel.recipe_yaml`, a path or an https URL into the
NVIDIA-NeMo/Automodel repo) -- and launch the `automodel` CLI on it:

    automodel <run_dir>/automodel_config.yaml --nproc-per-node N

    sft    -> TrainFinetuneRecipeForNextTokenPrediction          (full or LoRA, answer-only loss)
    kd     -> KnowledgeDistillationRecipeForNextTokenPrediction  (teacher logits + CE, kd_ratio)
    recipe -> the official recipe as is, with our run directory, seed, steps and (optionally) our data

Precedence: official recipe  <  values derived from the spec  <  backend_overrides (deep-merged last).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

from nanolm.config import deep_merge

from ..data import read_jsonl, write_jsonl

RECIPES = {"sft": "TrainFinetuneRecipeForNextTokenPrediction", "kd": "KnowledgeDistillationRecipeForNextTokenPrediction"}
MODEL_T = "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"
COLMAP_T = "nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset"
LOADER_T = "torchdata.stateful_dataloader.StatefulDataLoader"
COLLATE = "nemo_automodel.components.datasets.utils.default_collater"


def load_recipe(src: str, run_dir: Path) -> dict:
    if not src:
        return {}
    if src.startswith(("http://", "https://")):
        text = urllib.request.urlopen(src, timeout=60).read().decode()
        (run_dir / "automodel_recipe_source.yaml").write_text(text)          # provenance: the exact file used
    else:
        text = Path(src).read_text()
    return yaml.safe_load(text) or {}


def write_data(run_dir: Path, chat: bool) -> dict:
    """AutoModel reads JSONL with a question/answer column mapping. Chat models: their own template
    (use_hf_chat_template); base models: the same '### Instruction' rendering as the other backends."""
    paths = {}
    for split in ("train", "val"):
        recs = read_jsonl(run_dir / f"data/{split}.jsonl")
        rows = [{"question": r["prompt"] if chat else f"### Instruction:\n{r['prompt']}\n\n### Response:\n",
                 "answer": r.get("response", r.get("answer", ""))} for r in recs]
        p = run_dir / f"data/automodel_{split}.jsonl"
        write_jsonl(rows, p)
        paths[split] = str(p.resolve())
    return paths


def colmap(path: str, chat: bool, max_len: int, limit: int | None = None) -> dict:
    d = {"_target_": COLMAP_T, "path_or_dataset_id": path, "column_mapping": {"question": "question", "answer": "answer"},
         "answer_only_loss_mask": True, "use_hf_chat_template": chat, "seq_length": max_len, "truncation": True}
    if limit:
        d["limit_dataset_samples"] = limit
    return d


def build_config(spec, run_dir: Path, chat: bool) -> dict:
    m, t, a = spec.model, spec.train, spec.automodel
    base = load_recipe(a.recipe_yaml, run_dir)
    if spec.method == "recipe" and not base:
        raise ValueError("method=recipe needs automodel.recipe_yaml (a path or URL of an official AutoModel recipe)")
    ckpt = str((run_dir / "automodel" / "checkpoints").resolve())
    nproc = spec.launch.nproc_per_node
    y: dict = {
        "rng": {"_target_": "nemo_automodel.components.training.rng.StatefulRNG", "seed": spec.seed, "ranked": True},
        "step_scheduler": {"max_steps": t.steps, "ckpt_every_steps": t.steps, "num_epochs": 1000},
        "checkpoint": {"enabled": True, "checkpoint_dir": ckpt, "model_save_format": "safetensors", "save_consolidated": True},
    }
    if spec.method != "recipe":
        y["recipe"] = RECIPES[spec.method]
        y["step_scheduler"].update({"global_batch_size": t.batch_size * t.grad_accum * nproc, "local_batch_size": t.batch_size,
                                    "val_every_steps": t.eval_every or t.steps})
        y["dist_env"] = {"backend": "nccl", "timeout_minutes": 10}
        y["model"] = {"_target_": MODEL_T, "pretrained_model_name_or_path": m.name, "torch_dtype": m.dtype}
        y["distributed"] = {"strategy": a.strategy, "dp_size": None, "tp_size": a.tp_size, "cp_size": a.cp_size,
                            "sequence_parallel": False}
        y["loss_fn"] = {"_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"}
        y["optimizer"] = {"_target_": "torch.optim.AdamW", "lr": t.lr, "weight_decay": t.weight_decay, "betas": [0.9, 0.999]}
        y["lr_scheduler"] = {"lr_decay_style": t.schedule, "lr_warmup_steps": t.warmup_steps, "min_lr": t.lr * 0.1}
        y["clip_grad_norm"] = {"max_norm": t.max_grad_norm}
        if m.peft == "lora":
            y["peft"] = {"_target_": "nemo_automodel.components._peft.lora.PeftConfig", "target_modules": [t if ("*" in t or "." in t) else f"*.{t}" for t in m.lora_targets],  # AutoModel matches full names
                         "dim": m.lora_r, "alpha": m.lora_alpha, "dropout": m.lora_dropout}
        if spec.method == "kd":
            if not m.teacher:
                raise ValueError("kd needs model.teacher")
            y["teacher_model"] = {"_target_": MODEL_T, "pretrained_model_name_or_path": m.teacher, "torch_dtype": m.dtype}
            y["kd_ratio"] = spec.kd.ratio
            y["kd_loss_fn"] = {"_target_": "nemo_automodel.components.loss.kd_loss.KDLoss", "ignore_index": -100,
                               "temperature": spec.kd.temperature, "fp32_upcast": True}
    if spec.method != "recipe" or a.use_our_data:
        paths = write_data(run_dir, chat)
        y["dataset"] = colmap(paths["train"], chat, spec.data.max_len)
        y["validation_dataset"] = colmap(paths["val"], chat, spec.data.max_len, limit=spec.eval.n_val)
        y["dataloader"] = {"_target_": LOADER_T, "collate_fn": COLLATE, "shuffle": True}
        y["validation_dataloader"] = {"_target_": LOADER_T, "collate_fn": COLLATE}
        for k in ("dataset", "validation_dataset", "packed_sequence"):     # a recipe's own dataset keys do not merge
            base.pop(k, None)
    cfg = deep_merge(deep_merge(base, y), dict(spec.backend_overrides))
    cfg.pop("ci", None)
    return cfg


def find_final(run_dir: Path) -> tuple[str, bool]:
    """-> (path, is_adapter) of the last checkpoint: consolidated HF weights, or a PEFT adapter dir."""
    ckpts = sorted((run_dir / "automodel" / "checkpoints").glob("epoch_*_step_*"),
                   key=lambda p: int(p.name.rsplit("_", 1)[1]))
    if not ckpts:
        return "", False
    mdir = ckpts[-1] / "model"
    if (mdir / "adapter_config.json").exists():
        return str(mdir), True
    cons = mdir / "consolidated"
    return (str(cons), False) if cons.exists() else ("", False)


def train(spec, run_dir: Path, kind: str) -> dict:
    if spec.method not in ("sft", "kd", "recipe"):
        raise ValueError("automodel backend: sft | kd | recipe (DPO / GRPO live in NeMo RL, not AutoModel)")
    from transformers import AutoTokenizer
    chat = bool(AutoTokenizer.from_pretrained(spec.model.name).chat_template)
    cfg = build_config(spec, run_dir, chat)
    cfg_path = run_dir / "automodel_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    exe = spec.launch.automodel_cmd
    if not shutil.which(exe) and (Path(sys.executable).parent / exe).exists():   # env not activated: use this env's CLI
        exe = str(Path(sys.executable).parent / exe)
    cmd = [exe, str(cfg_path.resolve())]
    if spec.launch.nproc_per_node > 1:
        cmd += ["--nproc-per-node", str(spec.launch.nproc_per_node)]
    (run_dir / "automodel_command.txt").write_text(" ".join(cmd) + "\n")
    if spec.launch.dry_run:
        print("[automodel] dry run:", " ".join(cmd))
        return {"final_model": "", "is_adapter": False, "dry_run": True}
    print("[automodel]", " ".join(cmd))
    with open(run_dir / "automodel.log", "w") as log:
        env = dict(os.environ)
        if not spec.train.cudnn_sdpa:                        # carry the attention-kernel choice into the child processes
            env["LLMTRAIN_CUDNN_SDPA"] = "0"
            env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1] / "_site"), env.get("PYTHONPATH", "")])
        proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"automodel exited with code {rc}; see {run_dir / 'automodel.log'}")
    final, is_adapter = find_final(run_dir)
    return {"final_model": final, "is_adapter": is_adapter}
