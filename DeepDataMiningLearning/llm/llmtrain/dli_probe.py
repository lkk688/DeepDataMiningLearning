"""Probe an NVIDIA DLI workshop VM: what this container sees, and how its GPUs are reached.

Standalone: only the Python standard library, no llmtrain imports, so the single file can be uploaded to the workshop's
JupyterLab and run there. Read-only by default.

    python dli_probe.py                      # report -> /scratch/dli_probe/ (or ./dli_probe/ if /scratch is missing)
    python dli_probe.py --vllm-smoke         # + one chat request to the running vLLM server (starts it if needed)
    python dli_probe.py --train-smoke        # + a 5-step LoRA SFT through workshop.train.train (stops vLLM first!)

What it records (report.json + report.md):
  identity      host, user, container hints, OS, CPU, RAM
  storage       /scratch, /workspace, /home, /tmp: size, free, writable; relevant mounts
  gpus          nvidia-smi, /dev/nvidia*, CUDA_VISIBLE_DEVICES, torch.cuda if importable (from THIS container)
  packages      versions of torch / transformers / vllm / nemo_* / trl / peft / datasets / openai ...; full pip list
  env           environment variables, values of secrets redacted
  workshop      the `workshop` helper package: modules, public functions + signatures, full source (saved)
  services      DNS + HTTP: vLLM (/v1/models, max_model_len), the orchestrator (/openapi.json route list), others
  workspace     configs (model path, dataset, batch), scripts, data, model directories (config.json) and sizes
  internet      Hugging Face, PyPI, NVIDIA API reachability
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import inspect
import json
import os
import pkgutil
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE", re.I)
PACKAGES = ["torch", "transformers", "vllm", "nemo_automodel", "nemo-automodel", "nemo_rl", "nemo-rl", "nemo_toolkit",
            "nemo-curator", "nemo_curator", "data-designer", "nemo-microservices", "trl", "peft", "accelerate", "datasets",
            "huggingface_hub", "openai", "requests", "httpx", "pyyaml", "flash_attn", "flash-attn", "deepspeed",
            "megatron-core", "transformer_engine", "ray", "mlflow", "pandas", "jupyterlab"]
HOSTS = ["vllm-serve", "orchestrator", "workshop-orchestrator", "curator", "nemo-curator", "nemo-train", "nemo-rl",
         "jupyter", "guide", "localhost"]
PORTS = [8000, 8080, 8888, 9000, 5000, 7000, 8001]


def sh(cmd: list[str], timeout: int = 20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:                                      # tool missing, timeout, ...
        return f"<unavailable: {type(e).__name__}: {e}>"


def http(url: str, timeout: float = 5.0, method: str = "GET", body: dict | None = None) -> dict:
    try:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(2_000_000).decode("utf-8", "replace")
            out = {"status": r.status, "seconds": round(time.time() - t0, 3)}
            try:
                out["json"] = json.loads(raw)
            except ValueError:
                out["text"] = raw[:2000]
            return out
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": e.read(500).decode("utf-8", "replace")}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def identity() -> dict:
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = v.strip()
    except OSError:
        pass
    cg = ""
    try:
        cg = Path("/proc/1/cgroup").read_text()[:500]
    except OSError:
        pass
    osr = {}
    try:
        osr = dict(l.split("=", 1) for l in Path("/etc/os-release").read_text().splitlines() if "=" in l)
    except OSError:
        pass
    return {"hostname": socket.gethostname(), "user": os.environ.get("USER") or sh(["whoami"]), "uid": os.getuid(),
            "cwd": os.getcwd(), "home": str(Path.home()), "in_docker": Path("/.dockerenv").exists(),
            "cgroup_hint": cg.splitlines()[:3], "os": osr.get("PRETTY_NAME", "").strip('"'), "kernel": platform.release(),
            "python": sys.version.split()[0], "python_exe": sys.executable, "cpus": os.cpu_count(), "memory": mem}


def storage() -> dict:
    out = {}
    for p in ["/scratch", "/workspace", str(Path.home()), "/tmp", "/models", "/data"]:
        if not os.path.exists(p):
            continue
        u = shutil.disk_usage(p)
        writable = False
        try:
            t = Path(p) / f".dli_probe_write_test_{os.getpid()}"
            t.write_text("x"); t.unlink(); writable = True
        except OSError:
            pass
        out[p] = {"total_GB": round(u.total / 1e9, 1), "free_GB": round(u.free / 1e9, 1), "writable": writable}
    mounts = []
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            if re.search(r"scratch|workspace|models|nvidia|/data|/home", line):
                mounts.append(" ".join(line.split()[:4]))
    except OSError:
        pass
    return {"paths": out, "mounts": mounts[:40]}


def gpus() -> dict:
    out = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
           "dev_nvidia": sorted(str(p) for p in Path("/dev").glob("nvidia*"))}
    if shutil.which("nvidia-smi"):
        out["nvidia_smi"] = sh(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version", "--format=csv,noheader"])
        out["nvidia_smi_apps"] = sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
    else:
        out["nvidia_smi"] = "<not found in this container>"
    try:
        import torch
        out["torch_cuda"] = {"available": torch.cuda.is_available(), "count": torch.cuda.device_count(),
                             "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                             "torch_cuda_version": torch.version.cuda}
    except Exception as e:
        out["torch_cuda"] = f"<torch not usable: {type(e).__name__}>"
    return out


def packages(out_dir: Path) -> dict:
    vers = {}
    for p in PACKAGES:
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    allp = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions() if d.metadata["Name"])
    (out_dir / "pip_list.txt").write_text("\n".join(allp) + "\n")
    return {"selected": vers, "n_installed": len(allp), "full_list": str(out_dir / "pip_list.txt")}


def environment() -> dict:
    env = {k: ("<redacted>" if SECRET.search(k) else v) for k, v in sorted(os.environ.items())}
    keys = [k for k in env if re.search(r"VLLM|ORCH|NEMO|HF_|HUGGING|NVIDIA|SDG|CUDA|WORKSHOP|OPENAI|API_BASE|MODEL|SCRATCH|URL|HOST|PORT", k, re.I)]
    return {"relevant": {k: env[k] for k in keys}, "all_names": list(env)}


def workshop_pkg(out_dir: Path) -> dict:
    try:
        ws = importlib.import_module("workshop")
    except Exception as e:
        return {"importable": False, "error": f"{type(e).__name__}: {e}", "hint": "run inside the workshop JupyterLab kernel"}
    info = {"importable": True, "location": getattr(ws, "__file__", None), "modules": {}, "urls_in_source": [], "constants": {}}
    src_dir = out_dir / "workshop_src"; src_dir.mkdir(exist_ok=True)
    names = ["workshop"] + [f"workshop.{m.name}" for m in pkgutil.iter_modules(ws.__path__)]
    for name in names:
        try:
            mod = importlib.import_module(name)
        except Exception as e:
            info["modules"][name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        funcs = {}
        for attr, obj in vars(mod).items():
            if attr.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
                continue
            if inspect.isfunction(obj) or inspect.isclass(obj):
                try:
                    funcs[attr] = str(inspect.signature(obj))
                except (TypeError, ValueError):
                    funcs[attr] = "?"
            elif isinstance(obj, (str, int, float)) and attr.isupper():
                info["constants"][f"{name}.{attr}"] = "<redacted>" if SECRET.search(attr) else obj
        info["modules"][name] = funcs
        try:
            src = inspect.getsource(mod)
            (src_dir / f"{name}.py").write_text(src)
            info["urls_in_source"] += re.findall(r"https?://[\w.\-]+(?::\d+)?[\w/\-.]*", src)
        except (OSError, TypeError):
            pass
    info["urls_in_source"] = sorted(set(info["urls_in_source"]))
    info["source_saved_to"] = str(src_dir)
    for call in ("status", "models", "vllm_url"):                 # read-only helpers, if present
        fn = getattr(getattr(ws, "env", None), call, None) if hasattr(ws, "env") else None
        if fn is None:
            try:
                fn = getattr(importlib.import_module("workshop.env"), call)
            except Exception:
                fn = None
        if fn:
            try:
                info[f"env.{call}()"] = repr(fn())[:3000]
            except Exception as e:
                info[f"env.{call}()"] = f"<{type(e).__name__}: {e}>"
    return info


def services(ws_info: dict) -> dict:
    hosts = list(HOSTS)
    for u in ws_info.get("urls_in_source", []) + [v for v in ws_info.get("constants", {}).values() if isinstance(v, str)]:
        m = re.match(r"https?://([\w.\-]+)", str(u))
        if m and m.group(1) not in hosts:
            hosts.append(m.group(1))
    dns = {}
    for h in hosts:
        try:
            dns[h] = socket.gethostbyname(h)
        except OSError:
            pass
    probes = {}
    bases = set(re.match(r"(https?://[\w.\-]+(?::\d+)?)", u).group(1) for u in ws_info.get("urls_in_source", []) if re.match(r"https?://", u))
    for h in dns:
        for p in PORTS:
            try:
                with socket.create_connection((h, p), timeout=0.5):
                    bases.add(f"http://{h}:{p}")
            except OSError:
                pass
    for b in sorted(bases):
        if any(x in b for x in ("nvidia.com", "huggingface", "github", "pypi", "arxiv")):
            continue
        r = {}
        for path in ("/v1/models", "/health", "/status", "/openapi.json", "/"):
            res = http(b + path, timeout=3)
            if "status" in res:
                if path == "/openapi.json" and isinstance(res.get("json"), dict):
                    res = {"status": res["status"], "routes": sorted(res["json"].get("paths", {}).keys())}
                elif path == "/v1/models" and isinstance(res.get("json"), dict):
                    res = {"status": res["status"], "models": [{k: m.get(k) for k in ("id", "root", "max_model_len", "parent")} for m in res["json"].get("data", [])]}
                r[path] = res
        if r:
            probes[b] = r
    return {"dns": dns, "http": probes}


def workspace() -> dict:
    out = {"configs": {}, "scripts": [], "data_top": [], "model_dirs": []}
    ws = Path("/workspace")
    for y in sorted(ws.glob("configs/*.y*ml")) if ws.exists() else []:
        txt = y.read_text(errors="replace")
        keys = {}
        for k in ("pretrained_model_name_or_path", "model_name", "path_or_dataset_id", "global_batch_size", "local_batch_size",
                  "max_steps", "num_epochs", "nproc", "gpus_per_node", "checkpoint_dir", "lr", "dim", "tensor_parallel_size"):
            m = re.findall(rf"^\s*{k}\s*:\s*(.+)$", txt, re.M)
            if m:
                keys[k] = m[:3]
        out["configs"][y.name] = keys
    out["scripts"] = sorted(p.name for p in ws.glob("scripts/*")) if ws.exists() else []
    out["data_top"] = sorted(p.name for p in ws.glob("data/*")) if ws.exists() else []
    seen = 0
    for root in ("/workspace", "/models", "/scratch", "/data", str(Path.home())):
        for cfg in Path(root).glob("**/config.json") if Path(root).exists() else []:
            if seen >= 60 or len(cfg.parts) > 9:
                break
            d = cfg.parent
            if any(d.glob("*.safetensors")) or any(d.glob("adapter_config.json")) or any(d.glob("*.bin")):
                size = sum(f.stat().st_size for f in d.glob("*") if f.is_file())
                out["model_dirs"].append({"path": str(d), "GB": round(size / 1e9, 2), "adapter": (d / "adapter_config.json").exists()})
                seen += 1
    return out


def internet() -> dict:
    return {u: http(u, timeout=5).get("status", "unreachable") for u in
            ("https://huggingface.co/api/models?limit=1", "https://pypi.org/simple/pip/", "https://integrate.api.nvidia.com/v1/models")}


def vllm_smoke() -> dict:
    env = importlib.import_module("workshop.env")
    env.start_vllm()
    url = env.vllm_url()
    models = http(url + "/v1/models").get("json", {}).get("data", [])
    model = models[0]["id"] if models else None
    t0 = time.time()
    r = http(url + "/v1/chat/completions", timeout=120, method="POST",
             body={"model": model, "messages": [{"role": "user", "content": "In one sentence: what is a GPU?"}], "max_tokens": 64, "temperature": 0})
    j = r.get("json", {})
    return {"url": url, "model": model, "seconds": round(time.time() - t0, 2), "usage": j.get("usage"),
            "reply": (j.get("choices") or [{}])[0].get("message", {}).get("content", r.get("error"))}


def train_smoke(out_dir: Path) -> dict:
    env = importlib.import_module("workshop.env")
    train = importlib.import_module("workshop.train").train
    base = next(iter(sorted(Path("/workspace/configs").glob("sft_lora*.yaml"))), None)
    if base is None:
        return {"error": "no /workspace/configs/sft_lora*.yaml to start from"}
    data = out_dir / "smoke_sft.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"What is {i} + {i}?"}, {"role": "assistant", "content": str(2 * i)}]} for i in range(64)]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    txt = base.read_text()
    txt = re.sub(r"(path_or_dataset_id\s*:\s*).+", rf"\g<1>{data}", txt)
    txt = re.sub(r"(max_steps\s*:\s*).+", r"\g<1>5", txt)
    cfg = out_dir / "smoke_sft.yaml"; cfg.write_text(txt)
    sig = str(inspect.signature(train))
    env.stop_vllm()
    t0 = time.time()
    try:
        summary = train(config_path=str(cfg), run_id="dli_probe_smoke")
    finally:
        pass
    return {"base_yaml": str(base), "train_signature": sig, "seconds": round(time.time() - t0, 1),
            "summary": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else repr(v)[:300]) for k, v in dict(summary or {}).items()},
            "checkpoints": sorted(str(p) for p in Path("/scratch/dli_probe_smoke").glob("**/epoch_*_step_*"))[:10]}


def to_md(rep: dict) -> str:
    lines = ["# DLI workshop VM probe", ""]
    for sec, val in rep.items():
        lines += [f"## {sec}", "```json", json.dumps(val, indent=1, default=str)[:20000], "```", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="/scratch/dli_probe" if os.access("/scratch", os.W_OK) else "dli_probe")
    ap.add_argument("--vllm-smoke", action="store_true"); ap.add_argument("--train-smoke", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = {"probe_time": time.strftime("%Y-%m-%d %H:%M:%S")}
    steps = [("identity", identity), ("storage", storage), ("gpus", gpus), ("packages", lambda: packages(out)),
             ("env", environment), ("workshop", lambda: workshop_pkg(out)), ("workspace", workspace), ("internet", internet)]
    for name, fn in steps:
        try:
            rep[name] = fn()
        except Exception as e:
            rep[name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[probe] {name}: done", flush=True)
    rep["services"] = services(rep.get("workshop", {}) if isinstance(rep.get("workshop"), dict) else {})
    print("[probe] services: done", flush=True)
    for flag, name, fn in ((a.vllm_smoke, "vllm_smoke", vllm_smoke), (a.train_smoke, "train_smoke", lambda: train_smoke(out))):
        if flag:
            try:
                rep[name] = fn()
            except Exception as e:
                rep[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[probe] {name}: done", flush=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    (out / "report.md").write_text(to_md(rep))
    g = rep.get("gpus", {}); w = rep.get("workshop", {}); s = rep.get("services", {})
    print("\n== summary ==")
    print("GPUs visible here :", g.get("nvidia_smi"), "| torch:", g.get("torch_cuda"))
    print("workshop package  :", w.get("location") if w.get("importable") else w.get("error"))
    print("services          :", {b: list(r) for b, r in s.get("http", {}).items()})
    print("internet          :", rep.get("internet"))
    print("report            :", out / "report.md")


if __name__ == "__main__":
    main()
