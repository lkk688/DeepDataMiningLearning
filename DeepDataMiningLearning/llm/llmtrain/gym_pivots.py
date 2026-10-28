"""Select PivotRL training tasks: keep the Gym tasks the POLICY BEING TRAINED solves sometimes.

A pivot is only useful to GRPO if the policy's samples disagree (0 < successes < G); otherwise every advantage is zero.
Pass rates shipped with a dataset were measured on someone else's model, so re-measure them on yours:

    # 1) measure: score candidate tasks G times each with the real NeMo RL + Gym pipeline (validation only)
    python -m llmtrain -c llmtrain/configs/gym_swe_pivot_nemo_rl.yaml nemo_rl.val_generations=4 train.steps=1 \
        nemo_rl.gym_val=<candidates.jsonl> nemo_rl.gym_val_limit=1024 "backend_overrides={++grpo.val_at_end: false}"
    # 2) select
    python -m llmtrain.gym_pivots outputs/llmtrain/<measure run> --out data/gym_swe_pivot/pivots_<model>.jsonl
    # 3) train on them: nemo_rl.gym_train=<that file>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import read_jsonl


def measured(run_dir: Path) -> tuple[list[dict], list[list[float]]]:
    """-> (the validation tasks of a measurement run, their G rewards each, from NeMo RL's step-0 validation dump)."""
    run_dir = Path(run_dir)
    tasks = read_jsonl(next(run_dir.glob("data/gym_*_val*.jsonl")))
    dump = read_jsonl(next(run_dir.glob("nemo_rl/logs/exp_*/val_data_step0.jsonl")))
    rewards = [float(x["rewards"][0] if isinstance(x["rewards"], list) else x["rewards"]) for x in dump]
    if len(rewards) % len(tasks):
        raise ValueError(f"{len(rewards)} rewards for {len(tasks)} tasks: not a whole number of samples per task")
    g = len(rewards) // len(tasks)                       # NeMo RL repeats each task g times, consecutively
    return tasks, [rewards[i * g:(i + 1) * g] for i in range(len(tasks))]


def select(run_dir: Path, out: Path, lo: float = 0.0, hi: float = 1.0) -> dict:
    """Keep tasks with lo < (successes / G) < hi; record the measured rate as `pass_rate_policy`."""
    tasks, groups = measured(run_dir)
    kept = []
    for task, r in zip(tasks, groups):
        p = sum(x > 0 for x in r) / len(r)
        if lo < p < hi:
            kept.append(task | {"pass_rate_policy": p})
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in kept))
    hist = {}
    for r in groups:
        k = int(sum(x > 0 for x in r)); hist[k] = hist.get(k, 0) + 1
    return {"tasks": len(tasks), "samples_per_task": len(groups[0]), "kept": len(kept), "successes_histogram": dict(sorted(hist.items()))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir"); ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=float, default=0.0); ap.add_argument("--hi", type=float, default=1.0)
    a = ap.parse_args(argv)
    print(json.dumps(select(Path(a.run_dir), Path(a.out), a.lo, a.hi)))


if __name__ == "__main__":
    main()
