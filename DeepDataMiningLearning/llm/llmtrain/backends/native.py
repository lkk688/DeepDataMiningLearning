"""native backend: nanolm's own ~300-line SFT / DPO loops (nanolm/finetune.py), no Trainer abstractions.

The point of this backend is readability: every loss is written out, so it is the reference the TRL and
AutoModel backends are checked against. It trains on the same <run_dir>/data/*.jsonl as every backend.
"""
from __future__ import annotations

import random
from pathlib import Path

from nanolm import finetune as ft
from nanolm.config import FTDataConfig, FTModelConfig, FTRunConfig, FTTrainConfig

from ..data import read_jsonl


def _examples(recs, kind, tok, max_len):
    out = []
    for r in recs:
        if kind == "pref":
            w = ft.chat_example(tok, r["prompt"], r["chosen"], max_len)
            l = ft.chat_example(tok, r["prompt"], r["rejected"], max_len)
            if w and l:
                out.append({"chosen": w, "rejected": l, "prompt": r["prompt"]})
        else:
            ex = ft.chat_example(tok, r["prompt"], r["response"], max_len)
            if ex:
                out.append(ex)
    return out


def train(spec, run_dir: Path, kind: str) -> dict:
    if spec.method not in ("sft", "dpo"):
        raise ValueError("native backend implements sft and dpo only (use backend=trl for grpo / opd)")
    if spec.method == "dpo" and spec.model.peft != "lora":
        raise ValueError("native DPO uses the LoRA-disabled model as the reference: set model.peft=lora")
    t = spec.train
    cfg = FTRunConfig(
        name=spec.name, seed=spec.seed, run_dir=str(run_dir / "native"),
        model=FTModelConfig(name=spec.model.name, dtype=spec.model.dtype, lora=spec.model.peft == "lora",
                            lora_r=spec.model.lora_r, lora_alpha=spec.model.lora_alpha, lora_dropout=spec.model.lora_dropout,
                            lora_targets=list(spec.model.lora_targets), grad_checkpointing=spec.model.grad_checkpointing),
        data=FTDataConfig(max_len=spec.data.max_len),
        train=FTTrainConfig(method=spec.method, steps=t.steps, batch_size=t.batch_size, grad_accum=t.grad_accum, lr=t.lr,
                            weight_decay=t.weight_decay, warmup_steps=t.warmup_steps, dpo_beta=spec.dpo.beta,
                            log_every=t.log_every, eval_every=t.eval_every or 10 ** 9, max_time_sec=t.max_time_sec),
        eval_prompts=list(spec.eval.prompts), gen_tokens=spec.eval.gen_tokens)
    train_recs, val_recs = read_jsonl(run_dir / "data/train.jsonl"), read_jsonl(run_dir / "data/val.jsonl")
    # nanolm.finetune.run loads its own data; hand it ours instead (same chat_example rendering)
    orig = ft.load_examples
    ft.load_examples = lambda c, tok: (_examples(train_recs, kind, tok, c.data.max_len),
                                       _examples(val_recs, kind, tok, c.data.max_len))
    try:
        random.seed(spec.seed)
        m = ft.run(cfg)
    finally:
        ft.load_examples = orig
    final = run_dir / "native" / "adapter"
    return {"final_model": str(final) if final.exists() else "", "is_adapter": final.exists(),
            "backend_metrics": {k: v for k, v in m.items() if k.startswith(("before_", "after_", "train_", "peak_"))}}
