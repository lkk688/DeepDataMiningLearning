"""Backend-independent evaluation: the same held-out records, the same rendering, before and after training.

This is what makes runs comparable across backends -- a TRL run and an AutoModel run of the same spec are
scored by this code, not by each library's own logging (which differ in averaging, masking and padding).

    kind sft / qa : held-out response NLL (nats / response token) and perplexity
    kind pref     : how often the policy itself prefers the chosen answer (per-token log-prob)
    kind qa       : greedy-decoding answer accuracy with the verifiable checker (rewards.answer_correct)
    all           : sample answers to eval.prompts, written to samples_{before,after}.txt
"""
from __future__ import annotations

import gc
import math
from pathlib import Path

import torch

from nanolm.finetune import chat_example, collate, token_logprobs

from . import rewards
from .data import read_jsonl
from .backends.trl_backend import DTYPES


def load_model(spec, path: str, is_adapter: bool, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = DTYPES[spec.model.dtype]
    tok_src = path if (path and (Path(path) / "tokenizer_config.json").exists()) else spec.model.name
    tok = AutoTokenizer.from_pretrained(tok_src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not path:
        model = AutoModelForCausalLM.from_pretrained(spec.model.name, dtype=dtype)
    elif is_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(spec.model.name, dtype=dtype), path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    return model.to(device).eval(), tok


@torch.no_grad()
def response_nll(model, tok, recs, max_len, device, bs=8) -> dict:
    exs = [e for e in (chat_example(tok, r["prompt"], r["response"], max_len) for r in recs if "response" in r) if e]
    nll, n = 0.0, 0.0
    for i in range(0, len(exs), bs):
        lp, cnt = token_logprobs(model, *collate(exs[i: i + bs], tok.pad_token_id, device))
        nll -= lp.sum().item()
        n += cnt.sum().item()
    return {"val_response_nll": nll / n, "val_response_ppl": math.exp(nll / n)} if n else {}


@torch.no_grad()
def pair_logprobs(model, tok, recs, max_len, device) -> list[tuple]:
    """(sum log p(chosen), n tokens, sum log p(rejected), n tokens) per preference pair (None if too long)."""
    out = []
    for r in recs:
        w, l = chat_example(tok, r["prompt"], r["chosen"], max_len), chat_example(tok, r["prompt"], r["rejected"], max_len)
        if not (w and l):
            out.append(None)
            continue
        pw, nw = token_logprobs(model, *collate([w], tok.pad_token_id, device))
        pl, nl = token_logprobs(model, *collate([l], tok.pad_token_id, device))
        out.append((pw.item(), nw.item(), pl.item(), nl.item()))
    return out


def pref_metrics(cur, ref=None, beta=0.1) -> dict:
    """policy_prefers_chosen: the model alone ranks chosen above rejected (per-token log-prob).
    implicit_reward_acc: DPO's own metric, beta*[(log pi - log ref)(y_w) - (log pi - log ref)(y_l)] > 0 (ties = 1/2)."""
    pairs = [c for c in cur if c]
    out = {"val_policy_prefers_chosen": sum(c[0] / c[1] > c[2] / c[3] for c in pairs) / len(pairs)} if pairs else {}
    if ref:
        m = [beta * ((c[0] - r[0]) - (c[2] - r[2])) for c, r in zip(cur, ref) if c and r]
        out["val_implicit_reward_acc"] = sum(0.5 if abs(x) < 1e-3 else float(x > 0) for x in m) / len(m)
        out["val_implicit_reward_margin"] = sum(m) / len(m)
    return out


def _prompt_ids(tok, prompt):
    if tok.chat_template:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    return tok.encode(f"### Instruction:\n{prompt}\n\n### Response:\n")


@torch.no_grad()
def generate(model, tok, prompts, max_new, device, bs=16) -> list[str]:
    outs = []
    pad = tok.pad_token_id
    for i in range(0, len(prompts), bs):
        batch = [_prompt_ids(tok, p) for p in prompts[i: i + bs]]
        T = max(len(b) for b in batch)                        # left padding for batched generation
        ids = torch.tensor([[pad] * (T - len(b)) + b for b in batch], device=device)
        att = torch.tensor([[0] * (T - len(b)) + [1] * len(b) for b in batch], device=device)
        gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False, pad_token_id=pad)
        outs += [tok.decode(g[T:], skip_special_tokens=True).strip() for g in gen]
    return outs


def qa_accuracy(model, tok, recs, max_new, device) -> dict:
    outs = generate(model, tok, [r["prompt"] for r in recs], max_new, device)
    acc = rewards.answer_correct(None, outs, [r["answer"] for r in recs])
    fmt = rewards.answer_format(None, outs)
    return {"val_answer_acc": sum(acc) / len(acc), "val_answer_format": sum(f > 0 for f in fmt) / len(fmt)}


def score(spec, model, tok, kind, val, device, ref=None) -> dict:
    out = {}
    if kind in ("sft", "qa"):
        out.update(response_nll(model, tok, val, spec.data.max_len, device))
    if kind == "pref":
        cur = pair_logprobs(model, tok, val, spec.data.max_len, device)
        out.update(pref_metrics(cur, ref, spec.dpo.beta))
        out["_pair_logprobs"] = cur
    if kind == "qa":
        out.update(qa_accuracy(model, tok, val, spec.eval.gen_tokens, device))
    return out


def evaluate(spec, run_dir: Path, kind: str, final_model: str, is_adapter: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = read_jsonl(run_dir / "data/val.jsonl")[: spec.eval.n_val]
    metrics, ref = {}, None
    for tag, path, adapter in (("before", "", False), ("after", final_model, is_adapter)):
        model, tok = load_model(spec, path, adapter, device)
        res = score(spec, model, tok, kind, val, device, ref)
        ref = res.pop("_pair_logprobs", None) if tag == "before" else ref      # the base model is the DPO reference
        res.pop("_pair_logprobs", None)
        for k, v in res.items():
            metrics[f"{tag}_{k}"] = v
        texts = generate(model, tok, list(spec.eval.prompts), spec.eval.gen_tokens, device)
        (run_dir / f"samples_{tag}.txt").write_text("\n".join(f"### {p}\n{t}\n" for p, t in zip(spec.eval.prompts, texts)))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metrics["eval_n_val"] = len(val)
    return metrics


def main(argv=None):
    """Re-run the common evaluation on finished runs:  python -m llmtrain.evaluate <run_dir> [<run_dir> ...]"""
    import argparse
    import json
    from .config import load_spec
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    for rd in map(Path, ap.parse_args(argv).run_dirs):
        spec = load_spec(str(rd / "config.yaml"))
        torch.backends.cuda.enable_cudnn_sdp(spec.train.cudnn_sdpa)
        m = json.loads((rd / "metrics.json").read_text())
        m = {k: v for k, v in m.items() if not k.startswith(("before_", "after_"))}
        m.update(evaluate(spec, rd, m["data_kind"], m["final_model"], m.get("is_adapter", False)))
        (rd / "metrics.json").write_text(json.dumps(m, indent=1))
        print(rd, {k: round(v, 4) for k, v in m.items() if k.startswith(("before_", "after_"))})


if __name__ == "__main__":
    main()
