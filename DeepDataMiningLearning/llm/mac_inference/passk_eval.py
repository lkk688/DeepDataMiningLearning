"""pass@k evaluation against any OpenAI-compatible server: k sampled rollouts per task, checked by a verifier.

Tasks with automatic checks:
  gsm8k     : grade-school math (openai/gsm8k test split); the final number must equal the reference.
  humaneval : Python functions (openai/openai_humaneval); the code must pass the unit tests (run in a subprocess).
The k samples of a task are sent together, as pass@k rollouts are, and up to --conc requests run at once.

Reported per suite: pass@1 and pass@k (unbiased estimator, Chen et al. 2021), verified-correct samples per minute
(the harvest rate), aggregate tokens/s, per-request decode tokens/s and TTFT.

determinism : greedy answers to the first --det-tasks GSM8K tasks, sent (a) one at a time, (b) one at a time again,
              (c) all at once. A pass rate is only meaningful if (a) = (b), and batching should not change (a) = (c).

python passk_eval.py --model <id> --label <label> --out passk.jsonl [--suites gsm8k humaneval determinism] [--k 8]
"""
import argparse, glob, json, math, os, re, statistics, subprocess, sys, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HF = Path.home() / ".cache/huggingface/hub"
SAMPLING = {"temperature": 0.7, "top_p": 0.8}     # Qwen3.8's non-thinking sampling, minus top_k (not every server takes it)


def parquet(pattern):
    import pyarrow.parquet as pq
    hits = sorted(glob.glob(str(HF / pattern)))
    if not hits:
        raise SystemExit(f"not in the HF cache: {pattern}; run: hf download <dataset> --repo-type dataset")
    return pq.read_table(hits[-1]).to_pylist()


def gsm8k(n):
    rows = parquet("datasets--openai--gsm8k/snapshots/*/main/test-*.parquet")[:n]
    return [{"id": f"gsm8k/{i}", "suite": "gsm8k",
             "prompt": r["question"] + "\n\nSolve it step by step, briefly. End with a last line of the form 'Answer: <number>'.",
             "gold": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(rows)]


def humaneval(n):
    rows = parquet("datasets--openai--openai_humaneval/snapshots/*/openai_humaneval/test-*.parquet")[:n]
    return [{"id": r["task_id"], "suite": "humaneval",
             "prompt": "Complete this Python function. Reply with the whole function in one ```python code block, "
                       "no explanation.\n\n```python\n" + r["prompt"] + "```",
             "header": r["prompt"], "test": r["test"], "entry": r["entry_point"]} for r in rows]


def check(task, reply):
    if task["suite"] == "gsm8k":
        m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", reply) or re.findall(r"-?[\d,]*\.?\d+", reply)
        try:
            return bool(m) and abs(float(m[-1].replace(",", "")) - float(task["gold"])) < 1e-6
        except ValueError:
            return False
    blocks = re.findall(r"```(?:python)?\n(.*?)```", reply, re.S)
    code = blocks[0] if blocks else reply
    if f"def {task['entry']}" not in code:                         # only a body came back: put it under the header
        code = task["header"] + code
    imports = "\n".join(l for l in task["header"].splitlines() if l.startswith(("import ", "from ")))
    prog = f"{imports}\n{code}\n\n{task['test']}\n\ncheck({task['entry']})\n"
    with tempfile.TemporaryDirectory() as d:
        Path(d, "t.py").write_text(prog)
        try:
            return subprocess.run([sys.executable, "-I", "t.py"], cwd=d, capture_output=True, timeout=15).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def chat(url, body):
    """One streamed request: reply text, TTFT, decode speed, completion tokens. A failed request counts as an error."""
    try:
        return _chat(url, body)
    except Exception as e:                               # HTTP error, dropped stream: a wrong sample, flagged
        return {"text": "", "tokens": 0, "ttft_s": 0.0, "t_end": time.perf_counter(), "decode_tok_s": None,
                "error": f"{type(e).__name__}: {e}"[:200]}


def _chat(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; n = 0; usage = None; out = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:]); usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                piece = ch.get("delta", {}).get("content") or ""
                if piece:
                    last = time.perf_counter(); first = first or last; n += 1; out.append(piece)
    toks = (usage or {}).get("completion_tokens") or n
    return {"text": "".join(out), "tokens": toks, "ttft_s": (first or t0) - t0, "t_end": last or time.perf_counter(),
            "decode_tok_s": (toks - 1) / (last - first) if first and last and last > first else None}


def run_jobs(url, bodies, conc):
    """Send the bodies in order with at most conc in flight; returns results in order and the wall time."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(lambda b: chat(url, b), bodies))
    return res, max(r["t_end"] for r in res) - t0


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def passk(url, base, tasks, k, conc, max_tokens):
    bodies = [{**base, **SAMPLING, "seed": 1000 + j, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": t["prompt"]}]} for t in tasks for j in range(k)]
    res, wall = run_jobs(url, bodies, conc)
    ok = [check(t, res[i * k + j]["text"]) for i, t in enumerate(tasks) for j in range(k)]
    per_task = [sum(ok[i * k:(i + 1) * k]) for i in range(len(tasks))]
    toks = sum(r["tokens"] for r in res); dec = [r["decode_tok_s"] for r in res if r["decode_tok_s"]]
    ttft = sorted(r["ttft_s"] for r in res)
    return {"tasks": len(tasks), "k": k, "conc": conc, "max_tokens": max_tokens,
            "pass@1": round(statistics.mean(pass_at_k(k, c, 1) for c in per_task), 4),
            f"pass@{k}": round(statistics.mean(pass_at_k(k, c, k) for c in per_task), 4),
            "correct": sum(ok), "samples": len(ok), "correct_per_min": round(60 * sum(ok) / wall, 1),
            "wall_s": round(wall, 1), "tokens": toks, "aggregate_tok_s": round(toks / wall, 1),
            "decode_tok_s_median": round(statistics.median(dec), 1) if dec else None,
            "ttft_s_p50": round(ttft[len(ttft) // 2], 2), "ttft_s_p95": round(ttft[int(0.95 * (len(ttft) - 1))], 2),
            "truncated": sum(r["tokens"] >= max_tokens for r in res), "errors": sum("error" in r for r in res)}


def determinism(url, base, tasks, max_tokens):
    bodies = [{**base, "temperature": 0.0, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": t["prompt"]}]} for t in tasks]
    alone, _ = run_jobs(url, bodies, 1)
    again, _ = run_jobs(url, bodies, 1)
    batched, _ = run_jobs(url, bodies, len(bodies))
    same = lambda xs, ys: sum(x["text"] == y["text"] for x, y in zip(xs, ys))
    # where two answers part: share of the first answer's characters before the first difference (1.0 = identical)
    agree = lambda xs, ys: round(statistics.median(len(os.path.commonprefix([x["text"], y["text"]])) / max(len(x["text"]), 1)
                                                   for x, y in zip(xs, ys)), 2)
    return {"tasks": len(tasks), "repeat_identical": same(alone, again), "batched_identical": same(alone, batched),
            "repeat_prefix_share": agree(alone, again), "batched_prefix_share": agree(alone, batched),
            "correct_alone": sum(check(t, r["text"]) for t, r in zip(tasks, alone)),
            "correct_batched": sum(check(t, r["text"]) for t, r in zip(tasks, batched))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--suites", nargs="+", default=["determinism", "gsm8k", "humaneval"])
    ap.add_argument("--tasks", type=int, default=40); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--conc", type=int, default=8); ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--det-tasks", type=int, default=8)
    ap.add_argument("--extra", default=os.environ.get("EXTRA_JSON", "{}"))
    a = ap.parse_args()
    base = {"model": a.model, "stream": True, "stream_options": {"include_usage": True}, **json.loads(a.extra)}
    log = lambda r: (print(json.dumps(r), flush=True), open(a.out, "a").write(json.dumps(r) + "\n"))
    chat(a.url, {**base, "max_tokens": 8, "messages": [{"role": "user", "content": "Say hi."}]})       # warm-up
    for s in a.suites:
        t0 = time.time()
        if s == "determinism":
            r = determinism(a.url, base, gsm8k(a.det_tasks), a.max_tokens)
        else:
            r = passk(a.url, base, {"gsm8k": gsm8k, "humaneval": humaneval}[s](a.tasks), a.k, a.conc, a.max_tokens)
        log({"label": a.label, "suite": s, **r, "elapsed_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
