"""Run evaluation suites against one or more backends (backends.py), one server at a time.

For each backend: start its server, run each suite's client against it, sample system memory while the clients run,
stop the server. Every client appends JSON lines to --out; bench.py adds one "server" row per backend.

suites:
  passk     : passk_eval.py, pass@k rollouts on GSM8K and HumanEval + the determinism check
  rollouts  : rollout_client.py, aggregate tok/s at 1/2/4/8 concurrent requests (independent and same-context)
  quality   : quality_probe.py, 8 greedy questions with known answers
  decode    : bench_client.py, single-stream decode speed on three prompts

python bench.py ironmlx_27b tensorfold_27b mtplx_27b splash_27b --suites passk rollouts --out results_m5pro_passk.jsonl
python bench.py --list
"""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path

from backends import BACKENDS, Server, used_gb

HERE = Path(__file__).resolve().parent
CLIENTS = {"passk": "passk_eval.py", "rollouts": "rollout_client.py", "quality": "quality_probe.py",
           "decode": "bench_client.py"}


class MemorySampler(threading.Thread):
    """Peak system-wide used memory while the clients run (sampled every second)."""

    def __init__(self):
        super().__init__(daemon=True); self.peak = 0.0; self.stop = False

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, used_gb()); time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backends", nargs="*"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--suites", nargs="+", default=["passk"], choices=list(CLIENTS))
    ap.add_argument("--out", default="results_passk.jsonl"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--client-args", default="", help="extra arguments for every client, e.g. '--tasks 20 --k 4'")
    a = ap.parse_args()
    if a.list or not a.backends:
        for b in BACKENDS.values():
            print(f"{b.name:18s} {'installed' if b.installed() else 'missing  '}  {b.notes}")
        return
    subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])   # no idle sleep while benchmarking
    for name in a.backends:
        b = BACKENDS[name]
        idle = used_gb()
        try:
            with Server(b, a.port, log=HERE / f"logs/{name}.server.log") as srv:
                loaded = used_gb(); mem = MemorySampler(); mem.start()
                print(f"{name}: ready after {srv.ready_s}s, model id {srv.model}", flush=True)
                for s in a.suites:
                    cmd = [sys.executable, HERE / CLIENTS[s], "--url", srv.url, "--model", srv.model,
                           "--label", f"{name}", "--out", a.out, "--extra", json.dumps(b.extra), *a.client_args.split()]
                    if s != "passk":                       # the older clients take only the common arguments
                        cmd = cmd[:12]
                    subprocess.run([str(c) for c in cmd], cwd=HERE)
                mem.stop = True; mem.join()
                row = {"label": name, "suite": "server", "notes": b.notes, "ready_s": srv.ready_s,
                       "idle_gb": round(idle, 1), "loaded_gb": round(loaded - idle, 1), "peak_gb": round(mem.peak - idle, 1)}
        except RuntimeError as e:
            row = {"label": name, "suite": "server", "error": str(e)}
        print(json.dumps(row), flush=True)
        with open(HERE / a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
