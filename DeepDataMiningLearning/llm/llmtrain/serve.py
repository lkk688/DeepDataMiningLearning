"""Inference and data generation against any OpenAI-compatible server (vLLM): the NVIDIA DLI workshop's `vllm-serve`,
a local `vllm serve`, or a hosted endpoint. Standard library only, so it also runs in the workshop's JupyterLab.

    # list what the server serves
    python -m llmtrain.serve models --url http://vllm-serve:8000
    # generate: one output row per input row; prompts from a JSONL field or a template over its fields
    python -m llmtrain.serve generate --url http://vllm-serve:8000 --model <id> --in prompts.jsonl --out gen.jsonl \
        --field prompt --n 4 --temperature 0.7 --max-tokens 512 --workers 16
    # score a qa JSONL (prompt / answer) the way llmtrain/evaluate.py does, but through the server;
    # --match exact | contains | json:<field> for label answers (classification) instead of numbers
    python -m llmtrain.serve eval-qa --url ... --model ... --in val.jsonl

On the DLI workshop VM, `--url dli` is the workshop's vLLM server (workshop.env.vllm_url()) and `--url dli:teacher`
the teacher model behind the workshop's proxy (workshop.env.teacher_chat_completion; no API key) -- the latter is
how to generate synthetic data there.

Output rows keep every input field and add `completions` (n strings) and `usage`.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _post(url: str, body: dict, timeout: float = 600, retries: int = 3) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            if e.code < 500 or attempt == retries - 1:       # 4xx (e.g. context too long) is not retried
                raise RuntimeError(f"HTTP {e.code} from {url}: {msg[:500]}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def resolve(url: str) -> str:
    """'dli' -> the workshop's vLLM URL; anything else unchanged ('dli:teacher' is handled in chat())."""
    if url == "dli":
        from workshop import env
        return env.vllm_url()
    return url


def models(url: str) -> list[dict]:
    with urllib.request.urlopen(resolve(url).rstrip("/") + "/v1/models", timeout=30) as r:
        return json.loads(r.read()).get("data", [])


def chat(url: str, model: str, messages: list[dict], n: int = 1, temperature: float = 0.0, max_tokens: int = 512,
         **extra) -> dict:
    """One /v1/chat/completions call -> {"completions": [...], "usage": {...}}. `extra` goes into the body as is
    (top_p, top_k, seed, chat_template_kwargs={"enable_thinking": False}, ...)."""
    if url == "dli:teacher":                        # the workshop's teacher proxy: one sample per call
        from workshop import env
        js = [env.teacher_chat_completion(messages, temperature=temperature, max_tokens=max_tokens, timeout=180,
                                          **({"model": model} if model else {})) for _ in range(n)]
        # a reasoning teacher may put the answer in reasoning_content and the trace in content, or the reverse
        texts = [m.get("content") or m.get("reasoning_content") or "" for m in (j["choices"][0]["message"] for j in js)]
        return {"completions": texts, "usage": js[-1].get("usage")}
    body = {"model": model, "messages": messages, "n": n, "temperature": temperature, "max_tokens": max_tokens, **extra}
    j = _post(resolve(url).rstrip("/") + "/v1/chat/completions", body)
    return {"completions": [c["message"].get("content") or "" for c in j["choices"]], "usage": j.get("usage")}


def to_messages(row: dict, field: str = "prompt", template: str = "", system: str = "") -> list[dict]:
    if "messages" in row and not template:                          # chat rows: everything before the last answer
        msgs = list(row["messages"])
        while msgs and msgs[-1].get("role") == "assistant":
            msgs.pop()
        return msgs
    text = template.format(**row) if template else row[field]
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]


def generate(rows: list[dict], url: str, model: str, workers: int = 16, field: str = "prompt", template: str = "",
             system: str = "", **sampling) -> list[dict]:
    """Batch generation with a thread pool (the server batches concurrent requests itself)."""
    def one(row):
        try:
            return row | chat(url, model, to_messages(row, field, template, system), **sampling)
        except Exception as e:                                      # keep going; record the failure on the row
            return row | {"completions": [], "error": str(e)[:300]}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, rows))


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().strip(".").lower()


def match(text: str, answer, how: str = "number") -> bool:
    """number: llmtrain's numeric rule (rewards.answer_correct); exact: the whole output equals the label (case and
    surrounding whitespace ignored); contains: the label occurs in the output; json:<field>: the first JSON object in
    the output has field == label."""
    if how == "number":
        from .rewards import answer_correct
        return answer_correct(None, [text], [answer])[0] > 0
    if how == "exact":
        return _norm(text) == _norm(answer)
    if how == "contains":
        return _norm(answer) in _norm(text)
    if how.startswith("json:"):
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return m is not None and _norm(json.loads(m.group(0)).get(how[5:], "")) == _norm(answer)
        except (json.JSONDecodeError, AttributeError):
            return False
    raise ValueError(f"unknown match {how!r}: number | exact | contains | json:<field>")


def eval_qa(rows: list[dict], url: str, model: str, max_tokens: int = 320, workers: int = 16, how: str = "number",
            system: str = "", **extra) -> dict:
    """Greedy answers for qa records (prompt / answer), scored with `match` (default: the numeric rule of evaluate.py)."""
    out = generate(rows, url, model, workers=workers, max_tokens=max_tokens, temperature=0.0, system=system, **extra)
    texts = [(r["completions"] or [""])[0] for r in out]
    acc = [match(t, r["answer"], how) for t, r in zip(texts, rows)]
    return {"val_answer_acc": sum(acc) / max(len(acc), 1), "n": len(rows), "errors": sum("error" in r for r in out),
            "samples": [{"prompt": r["prompt"][:200], "answer": r["answer"], "output": t[:400]} for r, t in zip(rows[:3], texts[:3])]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("models"); m.add_argument("--url", required=True)
    for name in ("generate", "eval-qa"):
        g = sub.add_parser(name)
        g.add_argument("--url", required=True); g.add_argument("--model", required=True)
        g.add_argument("--in", dest="inp", required=True); g.add_argument("--out", default="")
        g.add_argument("--field", default="prompt"); g.add_argument("--template", default=""); g.add_argument("--system", default="")
        g.add_argument("--n", type=int, default=1); g.add_argument("--temperature", type=float, default=0.0)
        g.add_argument("--top-p", type=float, default=1.0); g.add_argument("--max-tokens", type=int, default=512)
        g.add_argument("--workers", type=int, default=16); g.add_argument("--limit", type=int, default=0)
        g.add_argument("--no-thinking", action="store_true", help="chat_template_kwargs enable_thinking=False (Qwen3)")
        g.add_argument("--match", default="number", help="eval-qa: number | exact | contains | json:<field>")
    a = ap.parse_args(argv)
    if a.cmd == "models":
        print(json.dumps([{k: d.get(k) for k in ("id", "root", "max_model_len", "parent")} for d in models(a.url)], indent=1))
        return
    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    rows = rows[: a.limit] if a.limit else rows
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if a.no_thinking else {}
    t0 = time.time()
    if a.cmd == "eval-qa":
        res = eval_qa(rows, a.url, a.model, max_tokens=a.max_tokens, workers=a.workers, how=a.match, system=a.system, **extra)
        res["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(res, indent=1, ensure_ascii=False))
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
        return
    out = generate(rows, a.url, a.model, workers=a.workers, field=a.field, template=a.template, system=a.system,
                   n=a.n, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, **extra)
    dst = a.out or str(Path(a.inp).with_suffix(".gen.jsonl"))
    Path(dst).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    toks = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in out)
    print(f"[serve] {len(out)} rows, {sum('error' in r for r in out)} errors, {toks} completion tokens, "
          f"{time.time() - t0:.1f}s ({toks / max(time.time() - t0, 1e-9):.0f} tok/s) -> {dst}")


if __name__ == "__main__":
    main()
