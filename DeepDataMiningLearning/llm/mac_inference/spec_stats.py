"""Speculative-decoding statistics from server logs: tokens per target pass and draft acceptance, greedy vs sampled.

IronMLX logs one "DFlash2 request completed" line per request (batch width, windows, drafted/accepted tokens);
TensorFold logs one "done" line (rounds, accepted/drafted, seed). Requests without a seed (TensorFold) or with
"sampled": false (IronMLX) are greedy.

python spec_stats.py logs/ironmlx_27b.server.log logs/tensorfold_27b.server.log
"""
import json, re, statistics, sys
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def ironmlx(line):
    if "DFlash2 request completed" not in line:
        return None
    m = json.loads(line.split("metrics=", 1)[1].split(" ", 1)[0] if False else line.split("metrics=", 1)[1].strip())
    width = int(re.search(r"batch_width=(\d+)", line).group(1))
    return {"mode": "sampled" if m["sampled"] else "greedy", "width": width, "tokens": m["generated_tokens"],
            "passes": m["windows"], "drafted": m["drafted_tokens"], "accepted": m["accepted_draft_tokens"]}


def tensorfold(line):
    m = re.search(r"\[tensorfold\] done .* tokens=(\d+) .* rounds=(\d+) accepted=(\d+)/(\d+)", line)
    if not m:
        return None
    tokens, rounds, acc, drafted = map(int, m.groups())
    return {"mode": "sampled" if "seed=" in line else "greedy", "width": None, "tokens": tokens, "passes": rounds,
            "drafted": drafted, "accepted": acc}


def summarize(path):
    groups = defaultdict(list)
    for raw in open(path, errors="replace"):
        line = ANSI.sub("", raw)
        try:
            r = ironmlx(line) or tensorfold(line)
        except (ValueError, KeyError, AttributeError):
            r = None
        if r and r["passes"]:
            groups[r["mode"]].append(r)
    for mode, rs in sorted(groups.items()):
        tok, passes = sum(r["tokens"] for r in rs), sum(r["passes"] for r in rs)
        acc, dr = sum(r["accepted"] for r in rs), sum(r["drafted"] for r in rs)
        widths = [r["width"] for r in rs if r["width"]]
        print(json.dumps({"log": path, "mode": mode, "requests": len(rs), "tokens_per_pass": round(tok / passes, 2),
                          "draft_acceptance": round(acc / dr, 3) if dr else None,
                          "mean_batch_width": round(statistics.mean(widths), 2) if widths else None}))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
