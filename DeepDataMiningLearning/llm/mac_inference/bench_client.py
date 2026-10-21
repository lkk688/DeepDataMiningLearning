"""Same streaming client for any OpenAI-compatible server: TTFT, decode tok/s, total tok/s.

python bench_client.py --url http://127.0.0.1:8000 --model <name> --label lithos_dspark --out results.jsonl
"""
import argparse, json, time, urllib.request

PROMPTS = {
    "code": "Write a Python function that parses an ISO-8601 date string without using any libraries, "
            "with docstring, type hints and at least five unit tests using pytest.",
    "explain": "Explain to a first-year computer science student how a GPU executes a matrix multiplication, "
               "covering threads, tiles, shared memory and memory bandwidth.",
    "math": "A train leaves city A at 9:00 at 80 km/h, another leaves city B, 300 km away, at 10:00 at 100 km/h "
            "towards A. When and where do they meet? Show every step.",
}


def run(url, model, prompt, max_tokens, extra):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}, **extra}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = t_last = None; text = []; usage = None; chunks = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    now = time.perf_counter()
                    t_first = t_first or now; t_last = now; chunks += 1; text.append(piece)
    n = (usage or {}).get("completion_tokens")
    if not n and text:                       # server sent no usage: count with the model's own tokenizer
        import glob, os
        from tokenizers import Tokenizer
        tj = os.environ.get("TOKENIZER_JSON") or sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--*Qwen3.8*/snapshots/*/tokenizer.json")))[0]
        n = len(Tokenizer.from_file(tj).encode("".join(text), add_special_tokens=False).ids)
    return {"ttft_s": round(t_first - t0, 3), "total_s": round(t_last - t0, 3), "completion_tokens": n,
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "chunks": chunks,
            "decode_tok_s": round((n - 1) / (t_last - t_first), 2) if n and t_last > t_first else None,
            "e2e_tok_s": round(n / (t_last - t0), 2) if n else None, "text_head": "".join(text)[:300]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512); ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--extra", default=__import__("os").environ.get("EXTRA_JSON", "{}"), help="extra JSON fields for the request body (default: $EXTRA_JSON)")
    a = ap.parse_args()
    run(a.url, a.model, "Say hi.", 16, json.loads(a.extra))                     # warm-up
    for name, p in PROMPTS.items():
        for rep in range(a.repeats):
            r = {"label": a.label, "prompt": name, "rep": rep, "max_tokens": a.max_tokens,
                 **run(a.url, a.model, p, a.max_tokens, json.loads(a.extra))}
            print(json.dumps({k: v for k, v in r.items() if k != "text_head"}), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
