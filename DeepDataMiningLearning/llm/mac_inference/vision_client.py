"""Vision check for an OpenAI-compatible server: one figure from this course, sent as a base64 image.

Each request is sent twice; the second shows whether the server caches the image / prefix.
python vision_client.py --model <name> --label <label> --out vision.jsonl
"""
import argparse, base64, json, time, urllib.request
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "docs/nanolm_figs/laptop_golf_ablation_bars.png"
QUESTION = ("This is a bar chart from a machine-learning experiment. Which bar is the lowest (best), "
            "and roughly what is its value? Answer in two sentences.")


def ask(url, model, image_b64, max_tokens=80):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
                {"type": "text", "text": QUESTION}]}]}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; usage = None; out = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                piece = ch.get("delta", {}).get("content") or ""
                if piece:
                    ttft = ttft or time.perf_counter() - t0
                    out.append(piece)
    return {"ttft_s": round(ttft, 3) if ttft else None, "total_s": round(time.perf_counter() - t0, 3),
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": (usage or {}).get("completion_tokens"),
            "answer": "".join(out).strip()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    img = base64.b64encode(FIG.read_bytes()).decode()
    for rep in range(2):
        r = {"label": a.label, "image": FIG.name, "rep": rep, **ask(a.url, a.model, img)}
        print(json.dumps(r), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
