"""trl backend: Hugging Face TRL trainers, configured from the llmtrain spec.

    sft  -> SFTTrainer   (prompt-completion data: loss on the completion only)
    dpo  -> DPOTrainer   (with LoRA the reference is the adapter-disabled model, no second copy)
    grpo -> GRPOTrainer  (verifiable rewards from llmtrain/rewards.py; DAPO-style clip-higher by default)
    opd  -> GKDTrainer   with lmbda = 1 (only student samples) and beta = 1 (reverse KL): on-policy distillation

Every key we set and every key in `backend_overrides` is checked against the installed TRL config dataclass,
so a TRL version change raises an error instead of silently ignoring a setting.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch

from .. import rewards
from ..data import read_jsonl

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def make_args(cls, values: dict, overrides: dict):
    fields = {f.name for f in dataclasses.fields(cls)}
    merged = {**values, **overrides}
    unknown = sorted(k for k in merged if k not in fields)
    if unknown:
        raise KeyError(f"{cls.__name__} (TRL) has no field(s) {unknown}; check backend_overrides / the TRL version")
    return cls(**merged)


def _chat(tok, prompt):
    return [{"role": "user", "content": prompt}] if tok.chat_template else f"### Instruction:\n{prompt}\n\n### Response:\n"


def _answer(tok, text):
    return [{"role": "assistant", "content": text}] if tok.chat_template else text + (tok.eos_token or "")


def to_trl(method: str, recs: list[dict], tok) -> list[dict]:
    if method == "sft":
        return [{"prompt": _chat(tok, r["prompt"]), "completion": _answer(tok, r["response"])} for r in recs]
    if method == "dpo":
        return [{"prompt": _chat(tok, r["prompt"]), "chosen": _answer(tok, r["chosen"]),
                 "rejected": _answer(tok, r["rejected"])} for r in recs]
    if method == "grpo":
        return [{"prompt": _chat(tok, r["prompt"]), "answer": r["answer"]} for r in recs]
    if method == "opd":
        if not tok.chat_template:
            raise ValueError("opd (TRL GKD) needs a chat model: its dataset is in 'messages' format")
        return [{"messages": [{"role": "user", "content": r["prompt"]},
                              {"role": "assistant", "content": r.get("response", r.get("answer", ""))}]} for r in recs]
    raise ValueError(f"trl backend: method {method!r} not supported (sft | dpo | grpo | opd)")


def train(spec, run_dir: Path, kind: str) -> dict:
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m, t = spec.model, spec.train
    tok = AutoTokenizer.from_pretrained(m.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = DTYPES[m.dtype]
    peft_config = None
    if m.peft == "lora":
        peft_config = LoraConfig(r=m.lora_r, lora_alpha=m.lora_alpha, lora_dropout=m.lora_dropout,
                                 target_modules=list(m.lora_targets), task_type="CAUSAL_LM")
    train_ds = Dataset.from_list(to_trl(spec.method, read_jsonl(run_dir / "data/train.jsonl"), tok))
    val_recs = read_jsonl(run_dir / "data/val.jsonl")
    eval_ds = Dataset.from_list(to_trl(spec.method, val_recs, tok)) if (t.eval_every and val_recs) else None
    out = run_dir / "trl"
    common = dict(output_dir=str(out), max_steps=t.steps, per_device_train_batch_size=t.batch_size,
                  gradient_accumulation_steps=t.grad_accum, learning_rate=t.lr, weight_decay=t.weight_decay,
                  warmup_steps=t.warmup_steps, lr_scheduler_type=t.schedule, max_grad_norm=t.max_grad_norm,
                  logging_steps=t.log_every, save_strategy="no", seed=spec.seed, data_seed=spec.seed, report_to="none",
                  bf16=t.precision == "bf16", fp16=t.precision == "fp16", gradient_checkpointing=m.grad_checkpointing)
    if eval_ds is not None:
        common.update(eval_strategy="steps", eval_steps=t.eval_every, per_device_eval_batch_size=t.batch_size)
    ov = dict(spec.backend_overrides)

    load_kw = {"dtype": dtype, **({"attn_implementation": m.attn_impl} if m.attn_impl else {})}
    model = AutoModelForCausalLM.from_pretrained(m.name, **load_kw)
    if spec.method == "sft":
        args = make_args(trl.SFTConfig, {**common, "max_length": spec.data.max_len, "completion_only_loss": True,
                                         "packing": False}, ov)
        trainer = trl.SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                                 processing_class=tok, peft_config=peft_config)
    elif spec.method == "dpo":
        args = make_args(trl.DPOConfig, {**common, "beta": spec.dpo.beta, "loss_type": spec.dpo.loss_type,
                                         "max_length": spec.data.max_len}, ov)
        trainer = trl.DPOTrainer(model=model, ref_model=None, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                                 processing_class=tok, peft_config=peft_config)
    elif spec.method == "grpo":
        g = spec.grpo
        args = make_args(trl.GRPOConfig, {**common, "num_generations": g.num_generations,
                                          "max_completion_length": g.max_new_tokens, "temperature": g.temperature,
                                          "beta": g.beta, "epsilon": g.epsilon, "epsilon_high": g.epsilon_high,
                                          "loss_type": g.loss_type, "scale_rewards": g.scale_rewards,
                                          "use_vllm": g.use_vllm}, ov)
        trainer = trl.GRPOTrainer(model=model, reward_funcs=rewards.get(list(g.rewards)), args=args,
                                  train_dataset=train_ds, processing_class=tok, peft_config=peft_config)
    elif spec.method == "opd":
        if not m.teacher:
            raise ValueError("opd needs model.teacher")
        o = spec.opd
        try:                                                    # TRL >= 1.x keeps GKD under trl.experimental
            from trl.experimental.gkd import GKDConfig, GKDTrainer
        except ImportError:
            from trl import GKDConfig, GKDTrainer
        args = make_args(GKDConfig, {**common, "lmbda": o.lmbda, "beta": o.beta, "temperature": o.temperature,
                                         "max_new_tokens": o.max_new_tokens, "max_length": spec.data.max_len}, ov)
        teacher = AutoModelForCausalLM.from_pretrained(m.teacher, **load_kw)
        trainer = GKDTrainer(model=model, teacher_model=teacher, args=args, train_dataset=train_ds,
                                 eval_dataset=eval_ds, processing_class=tok, peft_config=peft_config)
    else:
        raise ValueError(f"trl backend: method {spec.method!r} not supported (sft | dpo | grpo | opd)")

    result = trainer.train()
    final = run_dir / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    with open(run_dir / "train_log.jsonl", "w") as f:
        for row in trainer.state.log_history:
            f.write(json.dumps(row) + "\n")
    return {"final_model": str(final), "is_adapter": peft_config is not None, "trl_version": trl.__version__,
            "backend_metrics": {k: v for k, v in result.metrics.items() if isinstance(v, (int, float))}}
