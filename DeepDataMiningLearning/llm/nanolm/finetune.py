"""Part 2: working with PRETRAINED models -- evaluate, instruction-tune (SFT), preference-tune (DPO).

    python -m nanolm.finetune -c nanolm/configs/part2/P1_eval_bpb.yaml     # how good is Qwen2.5-0.5B, in bpb?
    python -m nanolm.finetune -c nanolm/configs/part2/P2_sft_alpaca.yaml   # base model -> instruction follower
    python -m nanolm.finetune -c nanolm/configs/part2/P3_dpo.yaml          # preference tuning

Everything is written out in ~300 lines (no TRL / Trainer magic) so the losses are readable:

  SFT  loss = mean over RESPONSE tokens of  -log p(token | prompt, previous tokens)
             (prompt tokens are masked out: we teach answering, not repeating the question)
  DPO  loss = -log sigmoid( beta * [ (log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x)) ] )
             y_w = preferred answer, y_l = rejected answer, ref = the model before tuning.
             With LoRA the reference model is just the same network with the adapter switched
             off, so DPO needs no second copy of the model -- important on a laptop.

Memory tricks for 8-16 GB laptops: LoRA (train ~1% of the weights), gradient checkpointing, and
computing the 150k-way softmax only at response positions, in chunks (`token_logprobs`).
Run directories, configs and metadata work exactly like nanolm.train.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import socket
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import FTRunConfig, load_config, save_config
from .device import get_device, memory_mb, seed_everything, synchronize
from .train import JsonlLogger, git_info, make_run_dir

IGNORE = -100


# ======================================================================= model + tokenizer
def load_model(cfg, dinfo):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if cfg.model.dtype == "auto":
        dtype = torch.bfloat16 if dinfo.type == "cuda" else torch.float32
    else:
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.model.dtype]
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=dtype).to(dinfo.device)
    if cfg.train.method != "eval_bpb":
        if cfg.model.grad_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        if cfg.model.lora:
            from peft import LoraConfig, get_peft_model
            model.enable_input_require_grads()
            model = get_peft_model(model, LoraConfig(
                r=cfg.model.lora_r, lora_alpha=cfg.model.lora_alpha, lora_dropout=cfg.model.lora_dropout,
                target_modules=list(cfg.model.lora_targets), task_type="CAUSAL_LM"))
    return model, tok


def _decoder_and_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.get_output_embeddings()


def token_logprobs(model, input_ids, attention_mask, labels, chunk: int = 256):
    """Sum of log p(label) over the labelled positions of each sequence -> [B].

    Only the hidden states at labelled positions go through the output layer, in chunks with
    activation checkpointing, so the [tokens x 150k vocab] logits never exist all at once."""
    decoder, head = _decoder_and_head(model)
    h = decoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state   # [B, T, D]
    tgt = labels[:, 1:]
    mask = tgt != IGNORE
    hs = h[:, :-1][mask]                         # [N, D]   hidden state that predicts each target
    ys = tgt[mask]                               # [N]
    rows = mask.nonzero()[:, 0]                  # which sequence each target belongs to

    def chunk_lp(hc, yc):
        logits = head(hc).float()
        return logits.gather(-1, yc[:, None]).squeeze(-1) - torch.logsumexp(logits, -1)

    parts = []
    for i in range(0, ys.numel(), chunk):
        hc, yc = hs[i: i + chunk], ys[i: i + chunk]
        parts.append(checkpoint(chunk_lp, hc, yc, use_reentrant=False) if torch.is_grad_enabled()
                     else chunk_lp(hc, yc))
    lp = torch.cat(parts) if parts else hs.new_zeros(0)
    out = torch.zeros(input_ids.size(0), device=lp.device, dtype=lp.dtype).index_add_(0, rows, lp)
    n = torch.zeros_like(out).index_add_(0, rows, torch.ones_like(lp))
    return out, n


# ================================================================================= data
def chat_example(tok, prompt: str, response: str, max_len: int):
    """Tokenize (prompt, response) with the model's chat template; label only the response."""
    msgs = [{"role": "user", "content": prompt}]
    if tok.chat_template:
        p_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": response}], tokenize=True)
        if hasattr(p_ids, "input_ids"):          # newer transformers return a BatchEncoding
            p_ids, full = p_ids["input_ids"], full["input_ids"]
        if full[: len(p_ids)] != p_ids:          # template not prefix-consistent: fall back
            full = p_ids + tok.encode(response, add_special_tokens=False) + [tok.eos_token_id]
    else:
        p_ids = tok.encode(f"### Instruction:\n{prompt}\n\n### Response:\n", add_special_tokens=False)
        full = p_ids + tok.encode(response, add_special_tokens=False) + [tok.eos_token_id]
    if len(full) > max_len or len(full) <= len(p_ids):
        return None
    labels = [IGNORE] * len(p_ids) + full[len(p_ids):]
    return {"input_ids": full, "labels": labels}


def load_examples(cfg, tok):
    from datasets import load_dataset
    d = cfg.data
    rng = random.Random(0)                       # data selection is fixed by the data, not the seed

    def take(split):
        ds = load_dataset(d.dataset, split=split)
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        return ds, idx

    def build(ds, idx, n):
        out = []
        for i in idx:
            r = ds[i]
            if d.format == "alpaca":
                prompt = r["instruction"] + (f"\n\n{r['input']}" if r.get("input") else "")
                ex = chat_example(tok, prompt, r["output"], d.max_len)
            elif d.format == "ultrafeedback":
                if float(r["score_chosen"]) - float(r["score_rejected"]) < d.min_score_gap:
                    continue
                prompt = r["chosen"][0]["content"]
                w = chat_example(tok, prompt, r["chosen"][-1]["content"], d.max_len)
                l = chat_example(tok, prompt, r["rejected"][-1]["content"], d.max_len)
                ex = {"chosen": w, "rejected": l, "prompt": prompt} if (w and l) else None
            else:
                raise ValueError(f"unknown data.format {d.format!r}")
            if ex is not None:
                out.append(ex)
            if len(out) >= n:
                break
        return out

    if d.val_split:
        ds, idx = take(d.split)
        train = build(ds, idx, d.n_train)
        vds, vidx = take(d.val_split)
        val = build(vds, vidx, d.n_val)
    else:                                         # carve validation out of the same split, disjoint
        ds, idx = take(d.split)
        val = build(ds, idx[: 5 * d.n_val], d.n_val)
        train = build(ds, idx[5 * d.n_val:], d.n_train)
    return train, val


def collate(exs, pad_id, device):
    T = max(len(e["input_ids"]) for e in exs)
    ids = torch.full((len(exs), T), pad_id, dtype=torch.long)
    lab = torch.full((len(exs), T), IGNORE, dtype=torch.long)
    att = torch.zeros((len(exs), T), dtype=torch.long)
    for i, e in enumerate(exs):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        lab[i, :n] = torch.tensor(e["labels"])
        att[i, :n] = 1
    return ids.to(device), att.to(device), lab.to(device)


# ================================================================================ losses
def sft_loss(model, batch):
    lp, n = token_logprobs(model, *batch)
    return -(lp.sum() / n.sum()), n.sum()


def dpo_terms(model, pair_batch, beta):
    """Returns (loss, reward accuracy, mean reward margin) for a batch of preference pairs."""
    w, l = pair_batch
    pw, _ = token_logprobs(model, *w)
    pl, _ = token_logprobs(model, *l)
    with torch.no_grad():
        ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else None
        if ctx is None:
            raise ValueError("DPO here uses the LoRA-disabled model as reference: set model.lora=true")
        with ctx:
            rw, _ = token_logprobs(model, *w)
            rl, _ = token_logprobs(model, *l)
    margin = beta * ((pw - rw) - (pl - rl))
    # ties (|margin| ~ 0, e.g. before training when policy == reference) count as half right
    acc = torch.where(margin.abs() < 1e-3, torch.full_like(margin, 0.5), (margin > 0).float())
    return -F.logsigmoid(margin).mean(), acc.mean(), margin.mean()


@torch.no_grad()
def evaluate_examples(model, cfg, val, tok, dinfo):
    model.eval()
    bs = cfg.train.batch_size
    if cfg.train.method == "sft":
        nll, ntok = 0.0, 0.0
        for i in range(0, len(val), bs):
            lp, n = token_logprobs(model, *collate(val[i: i + bs], tok.pad_token_id, dinfo.device))
            nll -= lp.sum().item()
            ntok += n.sum().item()
        model.train()
        return {"val_response_nll": nll / ntok, "val_response_ppl": math.exp(nll / ntok)}
    accs, margins, pref = [], [], []
    for i in range(0, len(val), bs):
        chunk = val[i: i + bs]
        w = collate([e["chosen"] for e in chunk], tok.pad_token_id, dinfo.device)
        l = collate([e["rejected"] for e in chunk], tok.pad_token_id, dinfo.device)
        _, acc, m = dpo_terms(model, (w, l), cfg.train.dpo_beta)
        accs.append(acc.item())
        margins.append(m.item())
        pw, nw = token_logprobs(model, *w)          # does the POLICY itself prefer the chosen answer?
        pl, nl = token_logprobs(model, *l)
        pref.append(((pw / nw) > (pl / nl)).float().mean().item())
    model.train()
    return {"val_reward_acc": sum(accs) / len(accs), "val_reward_margin": sum(margins) / len(margins),
            "val_policy_prefers_chosen": sum(pref) / len(pref)}


@torch.no_grad()
def sample_answers(model, tok, cfg, dinfo) -> str:
    model.eval()
    out = []
    for p in cfg.eval_prompts:
        msgs = [{"role": "user", "content": p}]
        if tok.chat_template:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
        else:
            ids = torch.tensor([tok.encode(f"### Instruction:\n{p}\n\n### Response:\n")])
        ids = ids.to(dinfo.device)
        gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=cfg.gen_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        out.append(f"### {p}\n{tok.decode(gen[0, ids.size(1):], skip_special_tokens=True).strip()}\n")
    model.train()
    return "\n".join(out)


# =========================================================================== bpb of HF model
@torch.no_grad()
def eval_bpb(model, tok, cfg, dinfo) -> dict:
    """Bits-per-byte of a pretrained model on the SAME FineWeb validation documents that the
    golf models are scored on (decoded from the golf shards), i.e. directly comparable numbers."""
    from .data import prepare_dataset, val_documents
    from .config import DataConfig
    meta = prepare_dataset(DataConfig(source="shards", shard_dir=cfg.data.shard_dir,
                                      tokenizer=cfg.data.shard_tokenizer))
    docs = val_documents(meta, cfg.data.eval_docs)
    T = cfg.train.ctx_len
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    nats, nbytes, ntok = 0.0, 0, 0
    model.eval()
    for doc in docs:
        ids = [start] + tok.encode(doc, add_special_tokens=False)
        nbytes += len(doc.encode("utf-8"))
        for s in range(0, len(ids) - 1, T):               # non-overlapping windows, like the golf baseline eval
            chunk = torch.tensor([ids[s: s + T + 1]], device=dinfo.device)
            logits = model(chunk[:, :-1]).logits.float()
            nats += F.cross_entropy(logits[0], chunk[0, 1:], reduction="sum").item()
            ntok += chunk.size(1) - 1
    return {"val_bpb": nats / math.log(2) / nbytes, "val_loss_per_token": nats / ntok,
            "eval_docs": len(docs), "eval_bytes": nbytes, "eval_tokens": ntok,
            "bytes_per_token": nbytes / ntok}


# ================================================================================= driver
def run(cfg: FTRunConfig) -> dict:
    dinfo = get_device(cfg.device, cfg.precision)
    seed_everything(cfg.seed)
    run_dir = make_run_dir(cfg)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "meta.json").write_text(json.dumps({
        "name": cfg.name, "seed": cfg.seed, "date": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(), "host": socket.gethostname(), "platform": platform.platform(),
        "torch": torch.__version__, "device": str(dinfo), "command": " ".join(sys.argv)}, indent=1))
    print(f"[run] {run_dir}\n[run] {cfg.train.method} | {cfg.model.name} | device {dinfo}")
    model, tok = load_model(cfg, dinfo)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics = {"name": cfg.name, "seed": cfg.seed, "method": cfg.train.method, "model": cfg.model.name,
               "params": n_all, "trainable_params": n_train, "device": str(dinfo),
               "dtype": str(next(model.parameters()).dtype).replace("torch.", "")}

    if cfg.train.method == "eval_bpb":
        t0 = time.perf_counter()
        metrics.update(eval_bpb(model, tok, cfg, dinfo))
        metrics["eval_time_sec"] = round(time.perf_counter() - t0, 1)
        print(f"[eval_bpb] {cfg.model.name}: val_bpb {metrics['val_bpb']:.4f} on {metrics['eval_docs']} docs "
              f"({metrics['bytes_per_token']:.2f} bytes/token)")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
        return metrics

    print(f"[model] {n_all / 1e6:.0f}M params, {n_train / 1e6:.2f}M trainable "
          f"({100 * n_train / n_all:.2f}%{' LoRA' if cfg.model.lora else ''})")
    train, val = load_examples(cfg, tok)
    print(f"[data] {len(train)} train / {len(val)} val examples ({cfg.data.format}, max_len {cfg.data.max_len})")
    (run_dir / "samples_before.txt").write_text(sample_answers(model, tok, cfg, dinfo))
    before = evaluate_examples(model, cfg, val, tok, dinfo)
    metrics.update({f"before_{k}": v for k, v in before.items()})
    print("[eval] before:", {k: round(v, 4) for k, v in before.items()})

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay, betas=(0.9, 0.999))
    log = JsonlLogger(run_dir / "train_log.jsonl", True)
    evlog = JsonlLogger(run_dir / "eval_log.jsonl", True)
    tc = cfg.train
    rng = random.Random(cfg.seed)
    order: list[int] = []
    step, t_train, t_last, acc = 0, 0.0, time.perf_counter(), []
    model.train()
    while step < tc.steps and not (tc.max_time_sec and t_train >= tc.max_time_sec):
        prog = step / tc.steps
        mult = (step + 1) / tc.warmup_steps if step < tc.warmup_steps else 0.5 * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = tc.lr * mult
        for _ in range(tc.grad_accum):
            if len(order) < tc.batch_size:
                order += rng.sample(range(len(train)), len(train))
            exs = [train[order.pop()] for _ in range(tc.batch_size)]
            if tc.method == "sft":
                loss, _ = sft_loss(model, collate(exs, tok.pad_token_id, dinfo.device))
                extra = {}
            else:
                w = collate([e["chosen"] for e in exs], tok.pad_token_id, dinfo.device)
                l = collate([e["rejected"] for e in exs], tok.pad_token_id, dinfo.device)
                loss, racc, margin = dpo_terms(model, (w, l), tc.dpo_beta)
                extra = {"reward_acc": racc.item(), "margin": margin.item()}
            (loss / tc.grad_accum).backward()
            acc.append({"loss": loss.item(), **extra})
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % tc.log_every == 0 or step == tc.steps:
            synchronize(dinfo)
            now = time.perf_counter()
            t_train += now - t_last
            t_last = now
            avg = {k: sum(a[k] for a in acc) / len(acc) for k in acc[0]}
            acc = []
            log.log(step=step, lr=opt.param_groups[0]["lr"], time=round(t_train, 1), **avg)
            print(f"step {step:5d} | " + " | ".join(f"{k} {v:.4f}" for k, v in avg.items())
                  + f" | {t_train:.0f}s | mem {memory_mb(dinfo) / 1024:.1f} GB")
        if tc.eval_every and step % tc.eval_every == 0 and step < tc.steps:
            r = evaluate_examples(model, cfg, val, tok, dinfo)
            evlog.log(step=step, **r)
            print("  eval:", {k: round(v, 4) for k, v in r.items()})
            t_last = time.perf_counter()

    after = evaluate_examples(model, cfg, val, tok, dinfo)
    evlog.log(step=step, **after)
    metrics.update({f"after_{k}": v for k, v in after.items()})
    metrics.update({"steps": step, "train_time_sec": round(t_train, 1), "peak_mem_mb": round(memory_mb(dinfo), 1),
                    "n_train_examples": len(train), "n_val_examples": len(val)})
    print("[eval] after:", {k: round(v, 4) for k, v in after.items()})
    (run_dir / "samples_after.txt").write_text(sample_answers(model, tok, cfg, dinfo))
    if cfg.model.lora:
        model.save_pretrained(run_dir / "adapter")      # only the LoRA weights (a few MB)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"[done] {run_dir}  (compare samples_before.txt vs samples_after.txt)")
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description="nanolm Part 2: evaluate / SFT / DPO a pretrained model")
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    run(load_config(args.config, args.overrides, cls=FTRunConfig))


if __name__ == "__main__":
    main()
