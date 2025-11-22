import json, os, platform, subprocess, time
from datetime import datetime
from typing import Any, Dict, List

# Optional: pip install psutil pynvml codecarbon
try:
    import psutil
except Exception:
    psutil = None
try:
    import pynvml
except Exception:
    pynvml = None
try:
    import torch
except Exception:
    torch = None

def _sh(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return ""

def gpu_info() -> List[Dict[str, Any]]:
    out = []
    if pynvml is None:
        return out
    try:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h).decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h).total // (1024**2)  # MB
            pl  = None
            try:
                pl = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0  # Watts
            except Exception:
                pass
            cc = None
            if torch and torch.cuda.is_available():
                try:
                    cc = ".".join(map(str, torch.cuda.get_device_capability(i)))
                except Exception:
                    pass
            out.append(dict(index=i, name=name, memory_mb=mem, power_limit_w=pl, compute_capability=cc))
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return out

def cpu_info() -> Dict[str, Any]:
    info = {"machine": platform.machine(), "processor": platform.processor()}
    info["platform"] = platform.platform()
    if psutil:
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    info["lscpu"] = _sh(["bash","-lc","lscpu"])
    return info

def storage_info() -> Dict[str, Any]:
    return {
        "lsblk": _sh(["bash","-lc","lsblk -d -o NAME,MODEL,ROTA,SIZE,TYPE"]),
        "df_pwd": _sh(["bash","-lc","df -h ."])
    }

def software_info() -> Dict[str, Any]:
    py = _sh(["bash","-lc","python -c \"import sys; print(sys.version.replace('\\n',' '))\""])
    torch_v = cuda_v = cudnn_v = ""
    if torch:
        try:
            torch_v = torch.__version__
            cuda_v  = getattr(torch.version, "cuda", "")
            cudnn_v = str(torch.backends.cudnn.version())
        except Exception:
            pass
    return {
        "python": py,
        "torch": torch_v,
        "cuda_toolkit_nvcc": _sh(["bash","-lc","nvcc --version || echo 'nvcc not found'"]),
        "torch_cuda": cuda_v,
        "cudnn": cudnn_v,
        "os": _sh(["bash","-lc","cat /etc/os-release || true"]),
        "kernel": _sh(["bash","-lc","uname -a"]),
        "nvidia_smi": _sh(["bash","-lc","nvidia-smi -q -x | head -n 60 || nvidia-smi || true"])
    }

def slurm_info() -> Dict[str, Any]:
    keys = ["SLURM_JOB_ID","SLURM_JOB_NAME","SLURM_NTASKS","SLURM_NNODES","SLURM_NODELIST","SLURM_GPUS"]
    return {k: os.environ.get(k) for k in keys if k in os.environ}

class TrainingMonitor:
    """
    Wrap training to measure wall time and peak GPU memory per GPU.
    Usage:
        mon = TrainingMonitor("exp_name")
        with mon.track():
            train()  # your training
        rpt = mon.finalize(extra={"epochs":4,"batch_size":16})
    """
    def __init__(self, exp_name="experiment"):
        self.exp_name = exp_name
        self.t0 = None
        self.t1 = None
        self.peak_mem_mb = []

    def track(self):
        class _Ctx:
            def __init__(self, outer): self.o=outer
            def __enter__(self):
                self.o.t0 = time.time()
                if torch and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                return self
            def __exit__(self, et, ev, tb):
                if torch and torch.cuda.is_available():
                    # Capture per-GPU peak reserved memory (more robust than allocated)
                    peaks = []
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            p = torch.cuda.max_memory_reserved() / (1024**2)
                            peaks.append(round(p, 1))
                    self.o.peak_mem_mb = peaks
                self.o.t1 = time.time()
        return _Ctx(self)

    def finalize(self, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        dur_s = (self.t1 - self.t0) if (self.t0 and self.t1) else None
        gpus = torch.cuda.device_count() if (torch and torch.cuda.is_available()) else 0
        gpu_hours = (dur_s/3600.0)*gpus if dur_s else None
        rep = {
            "exp_name": self.exp_name,
            "started_at": datetime.fromtimestamp(self.t0).isoformat() if self.t0 else None,
            "ended_at": datetime.fromtimestamp(self.t1).isoformat() if self.t1 else None,
            "duration_sec": round(dur_s, 2) if dur_s else None,
            "num_gpus": gpus,
            "peak_gpu_mem_mb_per_device": self.peak_mem_mb,
            "estimated_gpu_hours": round(gpu_hours, 3) if gpu_hours else None
        }
        if extra: rep.update(extra)
        return rep

def snapshot_compute_report(out_path="compute_report.json", extra: Dict[str,Any]=None) -> Dict[str,Any]:
    report = {
        "timestamp": datetime.now().isoformat(),
        "hardware": {"gpus": gpu_info(), "cpu": cpu_info(), "storage": storage_info()},
        "software": software_info(),
        "scheduler": slurm_info()
    }
    if extra: report.update(extra)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Wrote {out_path}")
    return report

if __name__ == "__main__":
    snapshot_compute_report()