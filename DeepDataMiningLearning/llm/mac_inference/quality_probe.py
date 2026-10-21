"""A quick sanity check of answer quality: 8 questions with checkable answers, greedy, thinking off.

Not a benchmark; it catches a quantization that is badly broken (for example a 2-bit checkpoint).
python quality_probe.py --model <name> --label <label> --out quality.jsonl
"""
import argparse, json, os, re, urllib.request

QA = [
    ("A train leaves at 9:40 and arrives at 13:05. How many minutes is the trip? Answer with a number only.", r"\b205\b"),
    ("What is 17 * 23 + 144 / 12? Answer with a number only.", r"\b403\b"),
    ("A shirt costs $40. It is discounted 25%, then the discounted price is raised 10%. Final price in dollars? Number only.", r"\b33(\.0+)?\b"),
    ("How many prime numbers are there between 1 and 50? Number only.", r"\b15\b"),
    ("What is the chemical symbol for tungsten? Answer in one word.", r"\bW\b"),
    ("Which planet has the moon Titan? One word.", r"Saturn"),
    ("Write a Python expression that reverses the string s. Code only, no explanation.", r"s\[::-1\]|reversed\(s\)"),
    ("In Python, what does len({1, 2, 2, 3}) return? Number only.", r"\b3\b"),
]


def ask(url, model, q, extra):
    body = {"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": 64, "temperature": 0.0, **extra}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default=os.environ.get("EXTRA_JSON", "{}"))
    a = ap.parse_args()
    extra = json.loads(a.extra); ok = 0
    for i, (q, pat) in enumerate(QA):
        ans = ask(a.url, a.model, q, extra).strip()
        hit = bool(re.search(pat, ans)); ok += hit
        r = {"label": a.label, "q": i, "correct": hit, "answer": ans[:120]}
        print(json.dumps(r), flush=True); open(a.out, "a").write(json.dumps(r) + "\n")
    print(f"{a.label}: {ok}/{len(QA)} correct")


if __name__ == "__main__":
    main()
