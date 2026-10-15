"""Run a set of experiments x seeds from ONE sweep file -- the code never changes.

Sweep YAML:
    name: golf_ablation            # results go to <out_dir>/sweeps/<name>/<run>_s<seed>/
    base: ../04_golf_lab.yaml      # experiment config every run starts from (relative to this file)
    seeds: [1337, 42, 7]
    common:                        # overrides applied to every run
      train.max_steps: 1500
    runs:
      - name: baseline             # no overrides = the base config
      - name: leaky_relu2
        overrides: {model.mlp: leaky_relu2}
    grid:                          # optional cartesian product, expanded into extra runs
      model.n_layers: [6, 9]

    python -m nanolm.sweep --sweep nanolm/configs/sweeps/golf_ablation.yaml            # sequential
    python -m nanolm.sweep --sweep ... --gpus 0,1,2,3                                 # 1 run per GPU
    python -m nanolm.sweep --sweep ... --only baseline leaky_relu2 --dry-run

Runs whose metrics.json already exists are skipped, so a crashed/interrupted sweep is resumed by
simply running the same command again. Each run is a separate process (clean GPU memory).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .config import load_config


def expand_runs(spec: dict) -> list[dict]:
    runs = [dict(r) for r in spec.get("runs", [])] or [{"name": "base"}]
    grid = spec.get("grid") or {}
    if grid:
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            ov = dict(zip(keys, combo))
            name = "_".join(f"{k.split('.')[-1]}{v}" for k, v in ov.items())
            runs.append({"name": name, "overrides": ov})
    return runs


def plan(sweep_path: str, only: list[str] | None = None, extra: list[str] | None = None,
         seeds: list[int] | None = None) -> tuple[dict, list[dict]]:
    sweep_path = Path(sweep_path)
    spec = yaml.safe_load(sweep_path.read_text())
    base = str((sweep_path.parent / spec["base"]).resolve())
    common = spec.get("common") or {}
    seeds = seeds or spec.get("seeds") or [1337]
    probe = load_config(base)
    root = (Path(spec.get("out_dir") or probe.out_dir) / "sweeps" / spec["name"]).resolve()
    jobs = []
    for run in expand_runs(spec):
        if only and run["name"] not in only:
            continue
        for seed in seeds:
            ov = {**common, **(run.get("overrides") or {})}
            run_dir = root / f"{run['name']}_s{seed}"
            args = [f"{k}={json.dumps(v)}" for k, v in ov.items()] + list(extra or [])
            args += [f"name={run['name']}", f"seed={seed}", f"run_dir={run_dir}",
                     f"tags={json.dumps([spec['name']])}"]
            load_config(base, args)          # validate every config BEFORE launching anything
            jobs.append({"name": run["name"], "seed": seed, "run_dir": run_dir, "config": base, "args": args})
    root.mkdir(parents=True, exist_ok=True)
    (root / "sweep.yaml").write_text(sweep_path.read_text())
    return {"root": root, "spec": spec}, jobs


def launch(job: dict, gpu: str | None) -> subprocess.Popen:
    env = dict(os.environ)
    # make the top-level package importable while keeping the caller's cwd (relative paths in
    # configs such as data/ and outputs/ then mean the same thing for every run)
    import_root = Path(__file__).resolve().parents[len(__package__.split("."))]
    env["PYTHONPATH"] = os.pathsep.join([str(import_root), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")   # less fragmentation / OOM
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    job["run_dir"].mkdir(parents=True, exist_ok=True)
    log = open(job["run_dir"] / "stdout.log", "w")
    cmd = [sys.executable, "-m", f"{__package__}.train", "--config", job["config"], *job["args"]]
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def claim(job: dict) -> bool:
    """Atomically claim a run (O_EXCL create of run_dir/CLAIM) so that several sweep processes --
    e.g. one per machine on a shared filesystem -- can work through the same sweep together."""
    job["run_dir"].mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(job["run_dir"] / "CLAIM", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{socket.gethostname()} pid={os.getpid()} {time.ctime()}\n".encode())
    os.close(fd)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="nanolm sweep runner")
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--gpus", default="", help="comma list, e.g. 0,1,2,3 (one run per GPU at a time)")
    ap.add_argument("--only", nargs="*", default=None, help="run only these run names")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="extra overrides for every run, e.g. --set device=cpu precision=fp32")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="override the sweep file's seeds")
    ap.add_argument("--reclaim", action="store_true",
                    help="ignore CLAIM files left by a killed sweep (only if no other sweep is running!)")
    args = ap.parse_args(argv)

    sys.stdout.reconfigure(line_buffering=True)     # progress visible in `nohup ... > log`
    info, jobs = plan(args.sweep, args.only, args.set, args.seeds)
    todo = [j for j in jobs if not (j["run_dir"] / "metrics.json").exists()]
    if args.reclaim and not args.dry_run:
        for j in todo:
            (j["run_dir"] / "CLAIM").unlink(missing_ok=True)
    print(f"[sweep] {info['spec']['name']}: {len(jobs)} runs ({len(jobs) - len(todo)} already done) -> {info['root']}")
    for j in todo:
        print(f"   {j['name']:<28s} seed={j['seed']:<6d} {' '.join(j['args'][:-4 - len(args.set)])}")
    if args.dry_run or not todo:
        return
    # a GPU id may be listed twice ("0,0") to run two small jobs on one GPU
    slots = list(enumerate(args.gpus.split(","))) if args.gpus else [(0, None)]
    running: dict = {}
    failed = []
    queue = list(todo)
    t0 = time.time()
    try:
        while queue or running:
            for slot in slots:
                while slot not in running and queue:
                    job = queue.pop(0)
                    if (job["run_dir"] / "metrics.json").exists() or not claim(job):
                        continue                  # finished or taken by another sweep process
                    running[slot] = (job, launch(job, slot[1]))
                    print(f"[sweep] start {job['name']} s{job['seed']} on "
                          f"{'gpu ' + slot[1] if slot[1] is not None else 'default device'}")
            time.sleep(2)
            for slot, (job, proc) in list(running.items()):
                if proc.poll() is not None:
                    del running[slot]
                    ok = proc.returncode == 0 and (job["run_dir"] / "metrics.json").exists()
                    if ok:
                        m = json.loads((job["run_dir"] / "metrics.json").read_text())
                        extra = f" | quant_bpb {m['quant_val_bpb']:.4f}" if "quant_val_bpb" in m else ""
                        print(f"[sweep] done  {job['name']} s{job['seed']} | val_bpb {m['val_bpb']:.4f}{extra} "
                              f"| {(time.time() - t0) / 60:.1f} min elapsed")
                    else:
                        failed.append(job)
                        (job["run_dir"] / "CLAIM").unlink(missing_ok=True)   # let a later resume retry it
                        print(f"[sweep] FAIL  {job['name']} s{job['seed']} (rc={proc.returncode}) -> "
                              f"{job['run_dir'] / 'stdout.log'}")
    except KeyboardInterrupt:                 # Ctrl-C: stop children and release their claims
        for job, proc in running.values():
            proc.terminate()
            (job["run_dir"] / "CLAIM").unlink(missing_ok=True)
        print("\n[sweep] interrupted -- run the same command again to resume")
        raise SystemExit(130)
    print(f"[sweep] finished ({len(failed)} failed in this process). Analyze with:\n"
          f"   python -m {__package__}.analyze {info['root']}")


if __name__ == "__main__":
    main()
