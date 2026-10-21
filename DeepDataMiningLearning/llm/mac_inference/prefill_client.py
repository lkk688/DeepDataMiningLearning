"""Prefill speed and time-to-first-token (TTFT) of an OpenAI-compatible server, for long prompts.

cold  : a unique prompt of N tokens (a random session id first, so no prefix cache can help);
        prefill tok/s = prompt_tokens / TTFT.
agent : the coding-agent pattern. A long system prompt (N tokens: "tools + instructions + files"), then
        3 user turns, each request = the whole conversation so far (append-only). Turn 1 pays the
        full prefill; turns 2-3 show whether the engine's prefix cache reuses it.

python prefill_client.py --model <name> --label <label> --out prefill.jsonl [--sizes 1024 4096 16384 32768]
"""
import argparse, glob, json, os, random, time, urllib.request
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
def _tokenizer_path():
    """TOKENIZER_JSON, else any Qwen3.8 checkpoint's tokenizer in the Hugging Face cache (Mac or Jetson)."""
    if os.environ.get("TOKENIZER_JSON"):
        return os.environ["TOKENIZER_JSON"]
    hits = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--*Qwen3.8*/snapshots/*/tokenizer.json")))
    if not hits:
        raise SystemExit("no Qwen3.8 tokenizer.json found; set TOKENIZER_JSON")
    return hits[0]


TOK = Tokenizer.from_file(_tokenizer_path())
CORPUS_IDS = TOK.encode("\n\n".join(p.read_text() for p in sorted(HERE.parent.glob("*.md"))) * 4,
                        add_special_tokens=False).ids        # this course's own docs, repeated as needed


def text_of(n, offset=0):
    return TOK.decode(CORPUS_IDS[offset: offset + n])


EXTRA = {}


def request(url, model, messages, max_tokens=16):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}, **EXTRA}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; usage = None; out = []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                piece = (ch.get("delta", {}).get("content") or "")
                if piece:
                    ttft = ttft or time.perf_counter() - t0
                    out.append(piece)
    reply = "".join(out)
    n_prompt = (usage or {}).get("prompt_tokens") or len(TOK.encode(
        "".join(m["content"] for m in messages), add_special_tokens=False).ids)
    return {"ttft_s": round(ttft, 3), "prompt_tokens": n_prompt, "prefill_tok_s": round(n_prompt / ttft, 1),
            "total_s": round(time.perf_counter() - t0, 3)}, reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 4096, 16384, 32768])
    ap.add_argument("--agent-size", type=int, default=16384)
    ap.add_argument("--extra", default=os.environ.get("EXTRA_JSON", "{}"), help="extra JSON fields for every request body (default: $EXTRA_JSON)")
    a = ap.parse_args()
    EXTRA.update(json.loads(a.extra))
    rng = random.Random(0)
    log = lambda r: (print(json.dumps(r), flush=True), open(a.out, "a").write(json.dumps(r) + "\n"))
    request(a.url, a.model, [{"role": "user", "content": "Say hi."}])                     # warm-up
    for n in a.sizes:                                                                      # cold prefill
        for rep in range(2 if n <= 16384 else 1):
            sid = f"Session {rng.randrange(10**12)}.\n\n"
            msg = [{"role": "user", "content": sid + text_of(n, offset=rng.randrange(0, 50_000))
                    + "\n\nIn one sentence: what is the text above about?"}]
            r, _ = request(a.url, a.model, msg)
            log({"label": a.label, "test": "cold", "size": n, "rep": rep, **r})
    system = f"Session {rng.randrange(10**12)}. You are a coding agent. Reference material:\n\n" + text_of(a.agent_size, 7)
    conv = [{"role": "system", "content": system}]
    for turn, q in enumerate(["Which file explains bits-per-byte? Answer in one line.",
                              "Name one optimizer used in Part 2. One line.",
                              "What is the laptop golf budget in MB? One line."], 1):
        conv.append({"role": "user", "content": q})
        r, reply = request(a.url, a.model, conv, max_tokens=24)
        log({"label": a.label, "test": "agent", "size": a.agent_size, "turn": turn, **r})
        conv.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
