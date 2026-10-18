"""Device / precision helpers so the same code runs on CUDA, Apple-Silicon (MPS) and CPU.

The rule of thumb used everywhere in nanolm:
  * never write ``.cuda()`` or ``device="cuda"`` -- always ask ``get_device()``;
  * never hard-code ``torch.autocast(device_type="cuda")`` -- use ``autocast_ctx()``;
  * anything CUDA-only (torch.compile, bf16 Newton-Schulz, fused AdamW, pin_memory)
    is switched on through ``DeviceInfo`` flags, never assumed.
"""
from __future__ import annotations

import contextlib
import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DeviceInfo:
    device: torch.device
    type: str                 # "cuda" | "mps" | "cpu"
    amp_dtype: torch.dtype | None   # dtype used inside autocast, None = full fp32
    compile_ok: bool          # torch.compile is reliable on this backend
    fused_adam: bool          # AdamW(fused=True) is available

    def __str__(self) -> str:
        amp = str(self.amp_dtype).replace("torch.", "") if self.amp_dtype else "fp32"
        return f"{self.type} (precision={amp}, compile={'yes' if self.compile_ok else 'no'})"


def get_device(requested: str = "auto", precision: str = "auto") -> DeviceInfo:
    """Pick the best available backend.

    requested: "auto" | "cuda" | "cuda:1" | "mps" | "cpu"
    precision: "auto" | "fp32" | "bf16" | "fp16"
      auto -> bf16 on CUDA GPUs that support it, fp32 on MPS/CPU
      (bf16 autocast on MPS needs macOS 14+ and recent PyTorch; opt in with precision=bf16).
    """
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    # torchrun sets LOCAL_RANK; each process owns one GPU.
    if requested == "cuda":
        requested = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    device = torch.device(requested)
    dtype_map = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if precision == "auto":
            amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            amp = dtype_map[precision]
        return DeviceInfo(device, "cuda", amp, compile_ok=True, fused_adam=True)
    if device.type == "mps":
        amp = None if precision == "auto" else dtype_map[precision]
        return DeviceInfo(device, "mps", amp, compile_ok=False, fused_adam=False)
    amp = None if precision in ("auto", "fp16") else dtype_map[precision]  # fp16 on CPU is slow/unsupported
    return DeviceInfo(device, "cpu", amp, compile_ok=False, fused_adam=False)


def autocast_ctx(info: DeviceInfo):
    """Mixed-precision context for the forward pass (no-op when amp_dtype is None)."""
    if info.amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=info.type, dtype=info.amp_dtype)


def synchronize(info: DeviceInfo) -> None:
    """Wait for queued kernels -- needed for honest wall-clock timing on GPUs."""
    if info.type == "cuda":
        torch.cuda.synchronize(info.device)
    elif info.type == "mps":
        torch.mps.synchronize()


def memory_mb(info: DeviceInfo) -> float:
    if info.type == "cuda":
        return torch.cuda.max_memory_allocated(info.device) / 2**20
    if info.type == "mps":
        # MPS has no peak counter. current_allocated_memory() is ~0 once a step's activations are
        # freed; the driver total also counts the allocator's cached blocks, so it tracks the peak.
        return torch.mps.driver_allocated_memory() / 2**20
    return 0.0


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """One call fixes every RNG we use (Python, NumPy, PyTorch CPU/CUDA/MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # also seeds CUDA and MPS generators
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
