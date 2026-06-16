"""
Shared utilities: seeding, meters, formatting, Lovász softmax loss.
No mmdet3d dependency.
"""

import math
import random
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Meters
# ---------------------------------------------------------------------------

class AverageMeter:
    """Exponential moving average for scalars (loss, timing)."""

    def __init__(self, momentum: float = 0.98):
        self.momentum = momentum
        self.val: float = 0.0
        self.avg: float = 0.0
        self._init = False

    def update(self, v: float) -> None:
        self.val = float(v)
        if not self._init:
            self.avg = self.val
            self._init = True
        else:
            self.avg = self.momentum * self.avg + (1.0 - self.momentum) * self.val


class ScalarDict:
    """Running sum of a dict of scalar losses (for per-epoch averages)."""

    def __init__(self):
        self._sums: dict = {}
        self._counts: dict = {}

    def update(self, d: dict) -> None:
        for k, v in d.items():
            val = float(v) if not isinstance(v, float) else v
            self._sums[k] = self._sums.get(k, 0.0) + val
            self._counts[k] = self._counts.get(k, 0) + 1

    def mean(self) -> dict:
        return {k: self._sums[k] / max(1, self._counts[k]) for k in self._sums}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    d, r = divmod(seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return f"{d}d {h}:{m:02d}:{s:02d}" if d else f"{h}:{m:02d}:{s:02d}"


def gpu_mem_mb() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return 0


def grad_norm(parameters, norm_type: float = 2.0) -> float:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    device = params[0].grad.device
    if norm_type == float("inf"):
        return float(max(p.grad.detach().abs().max().to(device) for p in params))
    norms = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in params])
    return float(torch.norm(norms, norm_type))


# ---------------------------------------------------------------------------
# LR helpers
# ---------------------------------------------------------------------------

def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def base_lr(optimizer) -> float:
    pg = optimizer.param_groups[0]
    return float(pg.get("initial_lr", pg["lr"]))


# ---------------------------------------------------------------------------
# Lovász softmax loss  (Berman et al. 2018, used for occupancy)
# Pure PyTorch, no external dependency.
# ---------------------------------------------------------------------------

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Lovász extension for a single class sorted by prediction error."""
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
    return jaccard


def lovasz_softmax_flat(
    probas: torch.Tensor,   # [N, C] – after softmax
    labels: torch.Tensor,   # [N]    – integer class labels
    ignore_index: int = -1,
    classes: str = "present",
) -> torch.Tensor:
    """
    Multi-class Lovász-softmax on flat (N, C) predictions.
    classes: 'all' | 'present' (skip classes absent from GT batch)
    """
    valid = labels != ignore_index
    probas = probas[valid]
    labels = labels[valid]
    if probas.numel() == 0:
        return probas.sum() * 0.0

    C = probas.size(1)
    losses = []
    class_to_sum = range(C) if classes == "all" else range(C)
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))

    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(
    logits: torch.Tensor,   # [B, C, Z, H, W] or [N, C]
    labels: torch.Tensor,   # [B, Z, H, W]   or [N]
    ignore_index: int = -1,
    classes: str = "present",
) -> torch.Tensor:
    """
    Convenience wrapper: accepts volume or flat tensors.
    Applies softmax internally.
    """
    if logits.dim() == 5:
        B, C, Z, H, W = logits.shape
        # permute → [B*Z*H*W, C]
        probas = F.softmax(logits.permute(0, 2, 3, 4, 1).reshape(-1, C), dim=1)
        labels_flat = labels.reshape(-1)
    elif logits.dim() == 2:
        probas = F.softmax(logits, dim=1)
        labels_flat = labels
    else:
        raise ValueError(f"Expected logits dim 2 or 5, got {logits.dim()}")

    return lovasz_softmax_flat(probas, labels_flat, ignore_index=ignore_index, classes=classes)
