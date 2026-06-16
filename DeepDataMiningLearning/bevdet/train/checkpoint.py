"""
Checkpoint utilities: EMA, save/load full training state, weight-only load.
No mmdet3d dependency.
"""

import copy
import os
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------

class EMA:
    """
    Shadow-parameter EMA for the model weights.

    Usage:
        ema = EMA(model, decay=0.9998)
        # after each optimizer step:
        ema.update(model)
        # for evaluation:
        with ema.apply(model):
            metrics = evaluate(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.shadow: dict = {}
        self.backup: dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply(self, model: nn.Module):
        """Context manager: swaps in shadow weights, restores on exit."""
        return _EMAContext(self, model)

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state.get("decay", self.decay)
        self.shadow = state.get("shadow", {})


class _EMAContext:
    def __init__(self, ema: EMA, model: nn.Module):
        self.ema = ema
        self.model = model

    def __enter__(self):
        for name, param in self.model.named_parameters():
            if name in self.ema.shadow:
                self.ema.backup[name] = param.data.clone()
                param.data.copy_(self.ema.shadow[name])
        return self.model

    def __exit__(self, *_):
        for name, param in self.model.named_parameters():
            if name in self.ema.backup:
                param.data.copy_(self.ema.backup[name])
        self.ema.backup.clear()


# ---------------------------------------------------------------------------
# Save / load full training state (for resume)
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    meta: Optional[dict] = None,
    ema: Optional[EMA] = None,
) -> None:
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "ema": ema.state_dict() if ema else None,
        "epoch": epoch,
        "meta": meta or {},
    }
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema: Optional[EMA] = None,
    map_location: str = "cpu",
) -> int:
    """Returns start_epoch (= saved_epoch + 1)."""
    state = torch.load(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing:
        print(f"[Checkpoint] Missing keys ({len(missing)}): {missing[:5]}{'…' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected)>5 else ''}")

    if optimizer and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if ema and state.get("ema"):
        ema.load_state_dict(state["ema"])

    start_epoch = int(state.get("epoch", 0)) + 1
    print(f"[Checkpoint] Resumed from {path} → start_epoch={start_epoch}")
    return start_epoch


# ---------------------------------------------------------------------------
# Weights-only save/load (for eval with Runner)
# ---------------------------------------------------------------------------

def save_weights(path: str, model: nn.Module) -> None:
    """Save state_dict only (no optimizer state). Atomic write."""
    tmp = path + ".tmp"
    torch.save({"state_dict": model.state_dict()}, tmp)
    os.replace(tmp, path)


def load_weights(
    path: str,
    model: nn.Module,
    strict: bool = False,
    prefix_strip: str = "",
    prefix_add: str = "",
    auto_prefix: bool = True,
) -> None:
    """
    Load weights-only checkpoint into model.

    Args:
        prefix_strip : remove a leading prefix from saved keys, e.g. 'det_model.'
                       when the checkpoint was saved from MultiTaskBEVFusion but
                       loaded into BEVFusionCA.
        prefix_add   : add a leading prefix to saved keys, e.g. 'det_model.' when
                       loading a flat BEVFusionCA checkpoint into a wrapper class
                       like TemporalBEVFusionCA whose registered module is `det_model`.
        auto_prefix  : when no manual prefix-strip/add is given but the saved keys
                       and model keys don't match, automatically detect and apply
                       the most likely prefix transformation. Default True.
    """
    obj = torch.load(path, map_location="cpu")
    sd = obj.get("state_dict", obj)  # handle both formats

    if prefix_strip:
        sd = {
            (k[len(prefix_strip):] if k.startswith(prefix_strip) else k): v
            for k, v in sd.items()
        }
    if prefix_add:
        sd = {prefix_add + k: v for k, v in sd.items()}

    # Auto-detect missing prefix mismatch (e.g. flat ckpt → det_model.* model).
    if auto_prefix and not prefix_strip and not prefix_add:
        model_keys = set(model.state_dict().keys())
        saved_keys = list(sd.keys())[:8]
        if model_keys and saved_keys and not any(k in model_keys for k in saved_keys):
            for candidate_prefix in ("det_model.",):
                prefixed = [candidate_prefix + k for k in saved_keys]
                hits = sum(1 for k in prefixed if k in model_keys)
                if hits >= max(1, len(saved_keys) // 2):
                    sd = {candidate_prefix + k: v for k, v in sd.items()}
                    print(f"[load_weights] auto-prepended '{candidate_prefix}' to ckpt keys "
                          f"to match wrapper model")
                    break

    # Filter out shape-mismatched tensors (strict=False only skips missing/extra keys).
    model_sd = model.state_dict()
    shape_skipped = []
    filtered = {}
    for k, v in sd.items():
        if k in model_sd and isinstance(v, torch.Tensor) and v.shape != model_sd[k].shape:
            shape_skipped.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(model_sd[k].shape)}")
        else:
            filtered[k] = v
    if shape_skipped:
        print(f"[load_weights] Skipping {len(shape_skipped)} shape-mismatched tensors:")
        for s in shape_skipped:
            print(f"  {s}")

    missing, unexpected = model.load_state_dict(filtered, strict=strict)
    n_loaded = len(filtered) - len(unexpected)
    print(f"[load_weights] {path}: {n_loaded}/{len(sd)} keys loaded "
          f"(missing={len(missing)}, unexpected={len(unexpected)}, shape_skip={len(shape_skipped)})")
