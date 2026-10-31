"""Concurrency: N requests at once against an OpenAI-compatible server, the way pass@k evaluation runs rollouts.

independent : N different short prompts (a batch of unrelated tasks).
rollouts    : N samples (temperature 0.7, different seeds) of the same long agent context. The context is sent once
              first, so a prefix cache holds it, as it would after the agent's earlier turns.
For each N it records every request's TTFT and decode speed, and the aggregate throughput:
all generated tokens / wall time from the first send to the last token.

python rollout_client.py --model <name> --label <label> --out rollouts.jsonl [--n 1 2 4 8] [--prefix 8192]
"""
import argparse, json, os, random, threading, time, urllib.request

from prefill_client import TOK, text_of

PROMPTS = ["Write a Python function that merges two sorted lists, with a docstring and three tests.",
           "Explain how a hash map handles collisions, in plain prose.",
           "A tank fills in 6 hours and drains in 9. Starting empty with both open, when is it full? Show the steps.",
           "Write a bash script that finds the ten largest files under a directory.",
           "Describe the causes of the French Revolution in three paragraphs.",
           "Implement binary search in C and explain each line.",
           "What is the difference between TCP and UDP? Give examples of when to use each.",
           "Write a SQL query for the top three salaries per department, and explain it."]


def one(url, body, res, i):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:]); usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    last = time.perf_counter(); first = first or last; n += 1
    toks = (usage or {}).get("completion_tokens") or n
    res[i] = {"t_start": t0, "t_first": first, "t_end": last or time.perf_counter(), "tokens": toks,
              "ttft_s": round((first or t0) - t0, 3),
              "decode_tok_s": round((toks - 1) / (last - first), 1) if first and last and last > first else None}


def run(url, bodies):
    res = [None] * len(bodies)
    th = [threading.Thread(target=one, args=(url, b, res, i)) for i, b in enumerate(bodies)]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = max(r["t_end"] for r in res) - t0
    return res, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--prefix", type=int, default=8192); ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--extra", default=os.environ.get("EXTRA_JSON", "{}"))
    a = ap.parse_args(); extra = json.loads(a.extra)
    base = {"model": a.model, "max_tokens": a.max_tokens, "stream": True, "stream_options": {"include_usage": True}, **extra}
    log = lambda r: (print(json.dumps(r), flush=True), open(a.out, "a").write(json.dumps(r) + "\n"))
    run(a.url, [{**base, "max_tokens": 8, "messages": [{"role": "user", "content": "Say hi."}]}])          # warm-up
    system = (f"Session {random.Random(1).randrange(10**12)}. You are a coding agent. Reference material:\n\n"
              + text_of(a.prefix, 11))
    task = "Using the reference material, propose one concrete improvement to the training recipe and justify it."
    ctx = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    for test in ["independent", "rollouts"]:
        if test == "rollouts":
            run(a.url, [{**base, "max_tokens": 8, "temperature": 0.0, "messages": ctx}])                  # cache the context
        for n in a.n:
            if test == "independent":
                bodies = [{**base, "temperature": 0.0, "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]}
                          for i in range(n)]
            else:
                bodies = [{**base, "temperature": 0.7, "seed": 1000 + i, "messages": ctx} for i in range(n)]
            res, wall = run(a.url, bodies)
            toks = sum(r["tokens"] for r in res)
            log({"label": a.label, "test": test, "n": n, "prefix": a.prefix if test == "rollouts" else 0,
                 "wall_s": round(wall, 2), "tokens": toks, "aggregate_tok_s": round(toks / wall, 1),
                 "ttft_s": [r["ttft_s"] for r in res], "decode_tok_s": [r["decode_tok_s"] for r in res]})


if __name__ == "__main__":
    main()
