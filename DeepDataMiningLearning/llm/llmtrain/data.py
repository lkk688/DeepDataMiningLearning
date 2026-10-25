"""Datasets -> a few canonical record kinds, written to <run_dir>/data/{train,val}.jsonl.

Every backend reads the same JSONL files, so a method compared across backends sees identical examples:

    kind "sft"  : {"prompt": str, "response": str}
    kind "pref" : {"prompt": str, "chosen": str, "rejected": str}
    kind "qa"   : {"prompt": str, "answer": str, "response": str}   (verifiable answer; response = reference solution)

Selection is fixed by the data, not the seed (shuffle with Random(0)), and train / val never overlap.
Examples whose chat-templated length exceeds data.max_len tokens are dropped, the same rule for every backend.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

GSM8K_INSTRUCTION = "Solve the problem step by step. Put the final answer after '####'."

# preset -> (kind, HF id, config, train split, val split)
PRESETS = {
    "alpaca": ("sft", "yahma/alpaca-cleaned", "", "train", "none"),
    "ultrafeedback": ("pref", "trl-lib/ultrafeedback_binarized", "", "train", "test"),
    "gsm8k": ("qa", "openai/gsm8k", "main", "train", "test"),
    "squad": ("sft", "rajpurkar/squad", "", "train", "validation"),
    "jsonl": ("auto", "", "", "train", "none"),
}
# which record kinds each method can train on
METHOD_KINDS = {"sft": {"sft", "qa"}, "dpo": {"pref"}, "grpo": {"qa"}, "opd": {"sft", "qa"}, "kd": {"sft", "qa"},
                "recipe": {"sft", "qa", "pref"}}


def gsm8k_final(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def to_record(preset: str, r: dict, d) -> dict | None:
    """One raw dataset row -> one canonical record (or None to skip)."""
    if preset == "alpaca":
        prompt = r["instruction"] + (f"\n\n{r['input']}" if r.get("input") else "")
        return {"prompt": prompt, "response": r["output"]}
    if preset == "ultrafeedback":
        if float(r["score_chosen"]) - float(r["score_rejected"]) < d.min_score_gap:
            return None
        return {"prompt": r["chosen"][0]["content"], "chosen": r["chosen"][-1]["content"],
                "rejected": r["rejected"][-1]["content"]}
    if preset == "gsm8k":
        sol = re.sub(r"<<[^>]*>>", "", r["answer"])                # drop calculator annotations: "= <<40-4=36>>36" -> "= 36"
        return {"prompt": f"{r['question']}\n\n{GSM8K_INSTRUCTION}", "answer": gsm8k_final(r["answer"]), "response": sol}
    if preset == "squad":
        return {"prompt": f"Context: {r['context']}\n\nQuestion: {r['question']}\nAnswer briefly.",
                "response": r["answers"]["text"][0]}
    if preset == "jsonl":
        c = {"prompt": "prompt", "response": "response", "chosen": "chosen", "rejected": "rejected", "answer": "answer"}
        c.update(d.columns)
        out = {k: r[v] for k, v in c.items() if v in r}
        return out if "prompt" in out else None
    raise ValueError(f"unknown data.preset {preset!r}; choose from {sorted(PRESETS)}")


def kind_of(rec: dict) -> str:
    if "chosen" in rec and "rejected" in rec:
        return "pref"
    if "answer" in rec:
        return "qa"
    if "response" in rec:
        return "sft"
    raise ValueError(f"record has no response / chosen+rejected / answer: keys {sorted(rec)}")


def render_len(tok, prompt: str, response: str) -> int:
    """Token length after the chat template (the same rendering nanolm.finetune.chat_example uses)."""
    if tok.chat_template:
        msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        ids = tok.apply_chat_template(msgs, tokenize=True)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        return len(ids)
    return len(tok.encode(f"### Instruction:\n{prompt}\n\n### Response:\n{response}")) + 1


def fits(tok, rec: dict, max_len: int) -> bool:
    if tok is None:
        return True
    if "chosen" in rec:
        return max(render_len(tok, rec["prompt"], rec["chosen"]), render_len(tok, rec["prompt"], rec["rejected"])) <= max_len
    return render_len(tok, rec["prompt"], rec.get("response", rec.get("answer", ""))) <= max_len


def _rows(d, preset: str, split: str):
    if preset == "jsonl":
        path = d.path
        if not path:
            raise ValueError("data.preset=jsonl needs data.path (a local .jsonl file)")
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    from datasets import load_dataset
    _, hf_id, cfg_name, _, _ = PRESETS[preset]
    return load_dataset(d.path or hf_id, d.config_name or cfg_name or None, split=split)


def build(d, tok=None) -> tuple[list[dict], list[dict], str]:
    """-> (train records, val records, kind)."""
    if d.preset not in PRESETS:
        raise ValueError(f"unknown data.preset {d.preset!r}; choose from {sorted(PRESETS)}")
    _, _, _, tr_split, va_split = PRESETS[d.preset]
    tr_split = d.split or tr_split
    va_split = d.val_split or va_split
    def shuffled(rows, seed):
        idx = list(range(len(rows)))
        random.Random(seed).shuffle(idx)
        return idx

    def collect(rows, order, n):
        out = []
        for i in order:
            rec = to_record(d.preset, rows[i], d)
            if rec is not None and fits(tok, rec, d.max_len):
                out.append(rec)
            if len(out) >= n:
                break
        return out

    rows = _rows(d, d.preset, tr_split)
    idx = shuffled(rows, 0)
    if va_split == "none":            # carve val out of the train split: the first 5*n_val shuffled rows are reserved
        val = collect(rows, idx[: 5 * d.n_val], d.n_val)
        train = collect(rows, idx[5 * d.n_val:], d.n_train)
    else:
        train = collect(rows, idx, d.n_train)
        vrows = _rows(d, d.preset, va_split)
        val = collect(vrows, shuffled(vrows, 1), d.n_val)
    if not train:
        raise ValueError("no training examples survived (check data.max_len and the dataset)")
    return train, val, kind_of(train[0])


def write_jsonl(recs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
