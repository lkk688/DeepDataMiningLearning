"""Run the benchmarks that model releases report (MMLU, GPQA, GSM8K, IFEval, ...) on small models.

    python -m nanolm.benchmarks run -c nanolm/configs/part3/B1_small_models.yaml            # all models x tasks
    python -m nanolm.benchmarks run -c nanolm/configs/part3/B2_laptop.yaml --slots 1         # laptop subset
    python -m nanolm.benchmarks summary outputs/nanolm/benchmarks/small_models              # table + figure

Two layers, on purpose:
  * `run` drives EleutherAI's lm-evaluation-harness (`lm_eval`), the tool behind most published small-model
    numbers, so results are comparable with model cards. Each (model, task-group) is one subprocess with its
    own output directory; finished ones are skipped, so a crashed run resumes.
  * `mc_accuracy` / `gsm8k_accuracy` below re-implement the two scoring rules in ~60 lines so students
    can see exactly what "accuracy on a multiple-choice benchmark" means -- and we cross-check them
    against lm_eval on the same model (tutorial §3.4).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


# ======================================================================= lm_eval driver
def load_bench_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("name", "models", "task_groups"):
        if key not in cfg:
            raise KeyError(f"benchmark config needs `{key}`")
    for m in cfg["models"]:
        unknown = set(m) - {"name", "hf", "peft", "chat", "dtype", "skip_groups", "extra_args"}
        if unknown:
            raise KeyError(f"unknown model key(s) {unknown} in {m}")
    return cfg


def jobs_for(cfg: dict) -> list[dict]:
    out_root = Path(cfg.get("out_dir", "outputs/nanolm/benchmarks")) / cfg["name"]
    jobs = []
    for m in cfg["models"]:
        for gname, g in cfg["task_groups"].items():
            if gname in (m.get("skip_groups") or []):
                continue
            if g.get("chat_only") and not m.get("chat"):
                continue
            margs = f"pretrained={m['hf']},dtype={m.get('dtype', 'bfloat16')}"
            if m.get("peft"):
                margs += f",peft={m['peft']}"
            if m.get("extra_args"):                     # e.g. enable_thinking=False for Qwen3/3.5
                margs += "," + m["extra_args"]
            cmd = [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args", margs,
                   "--tasks", ",".join(g["tasks"]),
                   "--batch_size", str(g.get("batch_size", cfg.get("batch_size", "auto"))),
                   "--output_path", str(out_root / m["name"] / gname), "--seed", "1234"]
            if g.get("num_fewshot") is not None:
                cmd += ["--num_fewshot", str(g["num_fewshot"])]
            if g.get("limit"):
                cmd += ["--limit", str(g["limit"])]
            if m.get("chat") and g.get("use_chat_template", False):
                cmd += ["--apply_chat_template"] + (["--fewshot_as_multiturn"] if g.get("num_fewshot") else [])
            if g.get("gen_kwargs"):
                cmd += ["--gen_kwargs", g["gen_kwargs"]]
            dev = cfg.get("device", "auto")
            if dev == "auto":                            # lm_eval defaults to cuda; pick what this machine has
                import torch
                dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            cmd += ["--device", dev]
            jobs.append({"model": m["name"], "group": gname, "dir": out_root / m["name"] / gname, "cmd": cmd})
    return jobs


def run(cfg_path: str, slots: int, dry: bool):
    cfg = load_bench_config(cfg_path)
    jobs = jobs_for(cfg)
    for j in jobs:                           # lm_eval writes results_*.json only on success
        if not (j["dir"] / "DONE").exists() and any(j["dir"].rglob("results_*.json")):
            (j["dir"] / "DONE").write_text("results file present\n")
    todo = [j for j in jobs if not (j["dir"] / "DONE").exists()]
    print(f"[bench] {cfg['name']}: {len(jobs)} jobs ({len(jobs) - len(todo)} done)")
    for j in todo:
        print("   ", j["model"], j["group"], " ".join(j["cmd"][3:9]))
    if dry or not todo:
        return
    root = Path(cfg.get("out_dir", "outputs/nanolm/benchmarks")) / cfg["name"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(Path(cfg_path).read_text())
    running, queue = [], list(todo)
    while queue or running:
        while queue and len(running) < slots:
            j = queue.pop(0)
            j["dir"].mkdir(parents=True, exist_ok=True)
            log = open(j["dir"] / "stdout.log", "w")
            env = dict(os.environ, TOKENIZERS_PARALLELISM="false")
            if j["cmd"][-1] == "mps":        # transformers 5.x threaded weight loading segfaults on MPS
                env.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
            j["proc"], j["t0"] = subprocess.Popen(j["cmd"], stdout=log, stderr=subprocess.STDOUT, env=env), time.time()
            running.append(j)
            print(f"[bench] start {j['model']} / {j['group']}")
        time.sleep(5)
        for j in list(running):
            if j["proc"].poll() is not None:
                running.remove(j)
                ok = j["proc"].returncode == 0
                if ok:
                    (j["dir"] / "DONE").write_text(f"{time.time() - j['t0']:.0f}s\n")
                print(f"[bench] {'done' if ok else 'FAIL'} {j['model']} / {j['group']} "
                      f"({(time.time() - j['t0']) / 60:.1f} min){'' if ok else ' -> ' + str(j['dir'] / 'stdout.log')}")


# ===================================================================== result summary
PRIMARY = {"acc_norm,none", "acc,none", "exact_match,flexible-extract", "exact_match,strict-match",
           "prompt_level_strict_acc,none", "inst_level_strict_acc,none"}


def collect(root: Path) -> list[dict]:
    rows = []
    for f in sorted(root.rglob("results_*.json")):
        model = f.relative_to(root).parts[0]
        res = json.loads(f.read_text())
        for task, metrics in res["results"].items():
            for k, v in metrics.items():
                if k in PRIMARY and isinstance(v, (int, float)):
                    rows.append({"model": model, "task": task, "metric": k.split(",")[0] + ("" if k.endswith("none") else "_" + k.split(",")[1]),
                                 "value": v, "stderr": metrics.get(k.replace(",", "_stderr,"), None),
                                 "n_shot": res.get("n-shot", {}).get(task)})
    return rows


def summary(root: str):
    import pandas as pd
    rows = collect(Path(root))
    if not rows:
        raise SystemExit(f"no lm_eval results under {root}")
    df = pd.DataFrame(rows)
    df.to_csv(Path(root) / "summary.csv", index=False)
    df = df[~df.task.str.startswith("mmlu_")]           # keep the MMLU average, not its 57 subjects
    pt = df.pivot_table(index="model", columns=["task", "metric"], values="value")
    pd.set_option("display.width", 250)
    print(pt.round(3).to_string())


# ============================================================== our own minimal scorers
def mc_accuracy(model, tok, items, device) -> dict:
    """Multiple-choice by log-likelihood (how lm_eval scores ARC/HellaSwag/MMLU/GPQA-zeroshot).

    For question q and choices c_1..c_K:   score_i = sum_t log p(c_i[t] | q, c_i[<t])
      acc      = mean[ argmax_i score_i == label ]
      acc_norm = mean[ argmax_i score_i / len(c_i) == label ]   (removes the bias toward short answers)
    len(c_i) counts the CHARACTERS of the choice text without the separating space, exactly as
    lm-evaluation-harness does; counting that one extra space changes ~2% of ARC-Easy answers.
    `items` = [{"context": str, "choices": [str, ...], "label": int}, ...]
    """
    import torch
    hits, hits_norm = 0, 0
    model.eval()
    for it in items:
        scores, norms = [], []
        for ch in it["choices"]:
            # tokenize context+choice together (as lm_eval does), then score only the choice tokens
            full = tok(it["context"] + ch, add_special_tokens=False)["input_ids"]
            n_ctx = len(tok(it["context"], add_special_tokens=False)["input_ids"])
            ids = torch.tensor([full], device=device)
            with torch.no_grad():
                logp = model(ids).logits.float().log_softmax(-1)[0, n_ctx - 1:-1]
            s = logp.gather(-1, ids[0, n_ctx:, None]).sum().item()
            scores.append(s)
            norms.append(s / len(ch.lstrip(" ")))
        hits += int(max(range(len(scores)), key=scores.__getitem__) == it["label"])
        hits_norm += int(max(range(len(norms)), key=norms.__getitem__) == it["label"])
    return {"acc": hits / len(items), "acc_norm": hits_norm / len(items), "n": len(items)}


def arc_items(split="test", subset="ARC-Easy") -> list[dict]:
    """ARC in exactly lm_eval's prompt format: 'Question: ...\nAnswer:' + ' <choice text>'."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", subset, split=split)
    out = []
    for r in ds:
        labels = r["choices"]["label"]
        out.append({"context": f"Question: {r['question']}\nAnswer:",
                    "choices": [" " + c for c in r["choices"]["text"]],
                    "label": labels.index(r["answerKey"])})
    return out


def crosscheck(model_name: str, limit: int = 0, dtype: str = "fp32"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dt).to(dev)
    items = arc_items()
    if limit:
        items = items[:limit]
    t0 = time.time()
    r = mc_accuracy(model, tok, items, dev)
    print(f"[crosscheck] {model_name} ({dtype}) ARC-Easy test (n={r['n']}): acc {r['acc']:.4f}  acc_norm {r['acc_norm']:.4f}  "
          f"({time.time() - t0:.0f}s, our 60-line scorer)")
    return r


def gsm8k_accuracy(model, tok, items, device, fewshot: str = "", max_new: int = 256) -> float:
    """Generative exact match (how GSM8K is scored): generate greedily, take the last number in the
    answer, compare with the reference number after '####'."""
    import torch
    correct = 0
    for it in items:
        prompt = fewshot + f"Question: {it['question']}\nAnswer:"
        ids = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).split("Question:")[0]
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        pred = nums[-1].replace(",", "").rstrip(".") if nums else None
        gold = it["answer"].split("####")[-1].strip().replace(",", "")
        correct += int(pred is not None and float(pred) == float(gold))
    return correct / len(items)


def main(argv=None):
    ap = argparse.ArgumentParser(description="benchmarks for small LLMs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", "-c", required=True)
    r.add_argument("--slots", type=int, default=1, help="parallel lm_eval processes (GPU memory permitting)")
    r.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("root")
    c = sub.add_parser("crosscheck", help="our own MC scorer on ARC-Easy, to compare with lm_eval")
    c.add_argument("model")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    if args.cmd == "run":
        run(args.config, args.slots, args.dry_run)
    elif args.cmd == "crosscheck":
        crosscheck(args.model, args.limit, args.dtype)
    else:
        summary(args.root)


if __name__ == "__main__":
    main()
