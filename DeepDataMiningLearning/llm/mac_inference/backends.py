"""Inference backends as data: how to start each engine's OpenAI-compatible server, what to add to every request, and
how to stop it. bench.py starts one backend, runs evaluation suites against it, and stops it.

Each backend serves one model on 127.0.0.1:<port>. The clients only speak /v1/chat/completions, so adding an engine
means adding one entry here: its command line, the request fields that turn thinking off, and where it is installed.

python backends.py            # list the backends and whether each one is installed
"""
import glob, json, os, signal, subprocess, time, urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()
SPLASH = HOME / "opt/splash-1.4.0-arm64-macos26"
IRONMLX_APP = HOME / "opt/ironmlx-0.2.0/IronMLX-0.2.0/IronMLX.app/Contents"
IRONMLX = IRONMLX_APP / "Helpers/ironmlx"
TENSORFOLD = HOME / "opt/tensorfold-1.2.0/tensorfold-1.2.0-macos-arm64/bin/tensorfold-native"
MTPLX = HOME / "venvs/mtplx/bin/mtplx"
STRATA = Path(os.environ.get("STRATA_DIR", HOME / "opt/strata"))
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def hf_snapshot(repo):
    """Local snapshot directory of a Hugging Face repo in the cache (engines that take a path, not a repo id)."""
    hits = sorted(glob.glob(str(HOME / f".cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*/")))
    return hits[-1].rstrip("/") if hits else f"<{repo} not in the HF cache>"


@dataclass
class Backend:
    name: str
    cmd: list                      # server command; "{port}" is filled in
    exe: Path                      # what must exist for the backend to count as installed
    extra: dict = field(default_factory=dict)   # merged into every request body
    model: str = ""                # served model id; empty = first id from /v1/models
    env: dict = field(default_factory=dict)
    notes: str = ""

    def command(self, port):
        return [str(c).replace("{port}", str(port)) for c in self.cmd]

    def installed(self):
        return Path(self.exe).exists()


Q27 = "mlx-community/Qwen3.8-27B-4bit"
DRAFT = "z-lab/Qwen3.8-27B-DFlash2"

BACKENDS = {b.name: b for b in [
    Backend("splash_27b",
            [SPLASH / "python/bin/python3", SPLASH / "install/launcher.py", "serve", "--port", "{port}",
             "--default-reasoning-effort", "none", "--idle-release", "off", "--model", Q27],
            SPLASH / "install/launcher.py", {"reasoning_effort": "none", **NO_THINK},
            notes="Splash 1.4.0, its own DFlash2 drafter (incoai), continuous batching"),
    Backend("ironmlx_27b",
            [IRONMLX, "--mlx-metallib", IRONMLX_APP / "Resources/mlx.metallib", "serve", "--model", hf_snapshot(Q27), "--model-id", "qwen3.8-27b", "--port", "{port}",
             "--dflash2-model-dir", hf_snapshot(DRAFT), "--dflash2-draft-bits", "4",
             "--max-sequences", "8", "--max-cache-cap", "16384"],
            IRONMLX, NO_THINK, "qwen3.8-27b",
            notes="IronMLX 0.2.0, DFlash2 draft tree (B1), cross-request tensor batching (default width 4)"),
    Backend("ironmlx_27b_w8",
            [IRONMLX, "--mlx-metallib", IRONMLX_APP / "Resources/mlx.metallib", "serve", "--model", hf_snapshot(Q27), "--model-id", "qwen3.8-27b", "--port", "{port}",
             "--dflash2-model-dir", hf_snapshot(DRAFT), "--dflash2-draft-bits", "4",
             "--max-sequences", "8", "--max-cache-cap", "16384", "--dflash2-tensor-batch-max-width", "8"],
            IRONMLX, NO_THINK, "qwen3.8-27b", notes="IronMLX with 8-wide DFlash2 verification groups"),
    Backend("ironmlx_27b_pc",
            [IRONMLX, "--mlx-metallib", IRONMLX_APP / "Resources/mlx.metallib", "serve", "--model", hf_snapshot(Q27), "--model-id", "qwen3.8-27b", "--port", "{port}",
             "--dflash2-model-dir", hf_snapshot(DRAFT), "--dflash2-draft-bits", "4",
             "--max-sequences", "8", "--max-cache-cap", "16384", "--prefix-lru-cache-max-bytes", str(4 << 30)],
            IRONMLX, NO_THINK, "qwen3.8-27b", notes="IronMLX with its cross-request prefix cache on (4 GiB; off by default)"),
    Backend("tensorfold_27b",
            [TENSORFOLD, "serve", Q27, "--drafter", DRAFT, "--drafter-bits", "4", "--parallel", "8",
             "--context", "16384", "--no-thinking", "--name", "qwen3.8-27b", "--port", "{port}", "--no-update-check"],
            TENSORFOLD, NO_THINK, "qwen3.8-27b", notes="TensorFold 1.2.0, DFlash2 drafter, 8 parallel streams"),
    # MTPLX's batched MTP lane (mtp_batch) exists only for Qwen3.6-35B-A3B; for the 27B it offers these two
    Backend("mtplx_27b",
            [MTPLX, "serve", "--model", "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed", "--model-id", "qwen3.8-27b",
             "--port", "{port}", "--no-auth", "--yes", "--reasoning", "off", "--scheduler-mode", "cooperative"],
            MTPLX, NO_THINK, "qwen3.8-27b",
            notes="MTPLX 2.12.2, flat 4-bit pack, native MTP, requests interleaved (cooperative)"),
    Backend("mtplx_27b_arbatch",
            [MTPLX, "serve", "--model", "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed", "--model-id", "qwen3.8-27b",
             "--port", "{port}", "--no-auth", "--yes", "--reasoning", "off", "--scheduler-mode", "ar_batch",
             "--max-active-requests", "8", "--decode-batch-max", "8"],
            MTPLX, NO_THINK, "qwen3.8-27b", notes="MTPLX batched decoding without MTP (ar_batch, 8 rows)"),
    Backend("mtplx_27b_serial",
            [MTPLX, "serve", "--model", "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed", "--model-id", "qwen3.8-27b",
             "--port", "{port}", "--no-auth", "--yes", "--reasoning", "off"],
            MTPLX, NO_THINK, "qwen3.8-27b", notes="MTPLX default: one request at a time, MTP depth 3"),
    Backend("mlx_lm_27b",
            [HOME / "venvs/mlxlm/bin/mlx_lm.server", "--model", Q27, "--port", "{port}",
             "--chat-template-args", '{"enable_thinking": false}'],
            HOME / "venvs/mlxlm/bin/mlx_lm.server", notes="reference: no drafter, sequential"),
    Backend("strata_fn",
            [STRATA / ".venv/bin/python", STRATA / "serve/server.py", "--engine", "strata",
             "--config", STRATA / "strata-q2_0.json", "--port", "{port}"],
            STRATA / "serve/server.py", {"reasoning_effort": "none", **NO_THINK},
            notes="Strata, Qwen3.8-Flash-Next Q2_0 (2-bit; speed reference only)"),
]}


def free_port(port):
    """Kill whatever still listens on the port (some launchers fork a detached server)."""
    pids = subprocess.run(["lsof", "-t", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in pids:
            try:
                os.kill(int(p), sig)
            except ProcessLookupError:
                pass
        time.sleep(3)


def served_models(url):
    with urllib.request.urlopen(url + "/v1/models", timeout=5) as r:
        return [m["id"] for m in json.loads(r.read())["data"]]


class Server:
    """Context manager: start a backend, wait until /v1/models answers, stop the whole process group on exit."""

    def __init__(self, backend, port=8000, log=None, timeout=1800):
        self.b, self.port, self.timeout = backend, port, timeout
        self.url = f"http://127.0.0.1:{port}"
        self.log = Path(log or f"logs/{backend.name}.server.log")

    def __enter__(self):
        free_port(self.port)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.proc = subprocess.Popen(self.b.command(self.port), stdout=open(self.log, "w"), stderr=subprocess.STDOUT,
                                     env={**os.environ, **self.b.env}, start_new_session=True)
        while True:
            try:
                ids = served_models(self.url)
                break
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"{self.b.name}: server exited; see {self.log}")
                if time.time() - self.t0 > self.timeout:
                    self.__exit__(None, None, None)
                    raise RuntimeError(f"{self.b.name}: not ready after {self.timeout}s")
                time.sleep(2)
        self.model = self.b.model or ids[0]
        self.ready_s = round(time.time() - self.t0, 1)
        return self

    def __exit__(self, *exc):
        for sig, wait in ((signal.SIGINT, 5), (signal.SIGTERM, 5), (signal.SIGKILL, 1)):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.proc.wait(wait)
                break
            except subprocess.TimeoutExpired:
                pass
        free_port(self.port)
        time.sleep(5)                                    # let the GPU memory go back before the next engine


def used_gb():
    """System-wide wired + active + compressed memory (GB), as sysmem.sh: counts mmapped Metal buffers too."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    ps = int(out.split("page size of")[1].split()[0])
    val = lambda key: int(next(l for l in out.splitlines() if key in l).split()[-1].rstrip("."))
    return (val("Pages wired down") + val("Pages active") + val("occupied by compressor")) * ps / 1e9


if __name__ == "__main__":
    for b in BACKENDS.values():
        print(f"{b.name:18s} {'installed' if b.installed() else 'missing  '}  {b.notes}")
