"""
Trainer – clean PyTorch training loop for MultiTaskBEVFusion.

Features
--------
• AdamW with per-module LR multipliers (mirrors your mmdet3d config).
• BF16 AMP via torch.amp (H100-optimized).
• Warmup + cosine LR scheduler, by iteration.
• Gradient clipping.
• EMA (optional).
• Per-epoch checkpoint + weights-only file for eval.
• W&B logging (optional).
• No mmdet3d Runner in the training path.

Evaluation
----------
Evaluation is NOT done here – always use eval_runner.py after saving
a weights checkpoint.  This keeps training and eval fully decoupled.
The Runner.test() path gives correct NDS; do not reimplement it here.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import amp

from .checkpoint import EMA, save_checkpoint, save_weights
from .misc import (
    AverageMeter,
    ScalarDict,
    base_lr,
    current_lr,
    format_eta,
    gpu_mem_mb,
    grad_norm,
    set_seed,
)


# ---------------------------------------------------------------------------
# Training config dataclass  (replaces scattered argparse fields)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Paths
    work_dir: str = "work_dirs/phase1"
    # Optimizer
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.99)
    clip_grad_norm: float = 20.0
    # Scheduler
    warmup_iters: int = 500
    total_epochs: int = 10
    # AMP
    dtype: str = "bfloat16"   # "bfloat16" for H100, "float16" for older GPUs
    # Gradient accumulation
    accum_steps: int = 1
    # EMA
    use_ema: bool = False
    ema_decay: float = 0.9998
    # Checkpointing
    save_interval: int = 1    # save every N epochs
    keep_last_n: int = 3      # keep N most-recent epoch checkpoints
    # Logging
    log_interval: int = 50    # iterations
    debug_iters: int = 0      # stop epoch early (0=full); for smoke-testing
    # W&B
    use_wandb: bool = False
    wandb_project: str = "bevfusion-multitask"
    wandb_run_name: str = ""
    # Per-module LR multipliers  key=module_name_substr, val=multiplier
    lr_multipliers: Dict[str, float] = field(default_factory=dict)
    # Seed
    seed: int = 42


# ---------------------------------------------------------------------------
# Optimizer / scheduler builders
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    cfg: TrainConfig,
) -> torch.optim.AdamW:
    """
    Build AdamW with per-module LR multipliers.

    cfg.lr_multipliers example:
        {
            "img_backbone": 0.0,        # frozen
            "view_transform": 0.5,
            "fusion_layer": 0.7,
            "bbox_head": 0.7,
            "occ_head": 1.5,            # new head → higher LR
        }
    All other parameters use base LR.
    """
    mults = cfg.lr_multipliers or {}

    # Bucket parameters by (lr_mult, no_decay)
    buckets: Dict[str, list] = {}

    def _mult_for(name: str) -> float:
        for key, mult in mults.items():
            if key in name:
                return float(mult)
        return 1.0

    def _no_decay(name: str) -> bool:
        return (
            name.endswith(".bias")
            or any(x in name for x in [
                "norm", "bn", "ln", "gn",
                "absolute_pos_embed", "relative_position_bias_table",
            ])
        )

    param_groups = []
    seen = set()
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        mult = _mult_for(name)
        if mult == 0.0:
            param.requires_grad_(False)
            continue
        lr_i = cfg.lr * mult
        wd_i = 0.0 if _no_decay(name) else cfg.weight_decay
        key = f"lr{lr_i:.2e}_wd{wd_i:.2e}"
        if key not in buckets:
            buckets[key] = {"params": [], "lr": lr_i, "weight_decay": wd_i}
        buckets[key]["params"].append(param)

    param_groups = list(buckets.values())
    return torch.optim.AdamW(
        param_groups,
        lr=cfg.lr,
        betas=cfg.betas,
        fused=torch.cuda.is_available(),
    )


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup then cosine decay to 0, iterated per step."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base in self.base_lrs:
            if step <= self.warmup_steps:
                lrs.append(base * step / self.warmup_steps)
            else:
                t = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lrs.append(0.5 * base * (1.0 + math.cos(math.pi * t)))
        return lrs


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Usage:
        trainer = Trainer(model, train_loader, cfg, device)
        trainer.run()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        cfg: TrainConfig,
        device: torch.device,
        start_epoch: int = 0,
        resume_path: Optional[str] = None,
        telemetry=None,                 # optional TelemetryRecorder; logs internal stats
    ):
        self.model = model
        self.loader = train_loader
        self.cfg = cfg
        self.device = device
        self.telemetry = telemetry

        set_seed(cfg.seed)
        os.makedirs(cfg.work_dir, exist_ok=True)

        self.steps_per_epoch = len(train_loader)
        total_steps = self.steps_per_epoch * cfg.total_epochs

        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, cfg.warmup_iters, total_steps
        )
        self.scaler = amp.GradScaler(device="cuda", enabled=(cfg.dtype == "float16"))
        self.ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None
        # Epochs are 1-indexed: fresh start begins at epoch 1
        self.start_epoch = max(1, start_epoch)

        # Recover AMP dtype flag
        self._amp_enabled = cfg.dtype in ("bfloat16", "float16")
        self._amp_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16

        # Resume: load_checkpoint returns saved_epoch + 1 (next epoch to run)
        if resume_path and os.path.isfile(resume_path):
            from .checkpoint import load_checkpoint
            self.start_epoch = load_checkpoint(
                resume_path, model, self.optimizer, self.scheduler,
                self.scaler, self.ema, map_location=str(device),
            )
            # Align scheduler iteration counter with resumed epoch
            completed_steps = (self.start_epoch - 1) * self.steps_per_epoch
            self.scheduler.last_epoch = completed_steps - 1
            self.scheduler.step()  # advance to correct LR

        # W&B
        self._wandb = None
        if cfg.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        try:
            import wandb
            run_name = self.cfg.wandb_run_name or os.path.basename(self.cfg.work_dir)
            self._wandb = wandb.init(
                project=self.cfg.wandb_project,
                name=run_name,
                config=vars(self.cfg),
                resume="allow",
            )
        except Exception as e:
            print(f"[W&B] init failed: {e}")

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def _forward_loss(self, batch: Any):
        """
        Run data preprocessor → model.loss() inside AMP context.
        Returns (loss_dict, log_dict).
        """
        with amp.autocast(
            device_type="cuda",
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            # mmdet3d data_preprocessor: moves tensors to device, normalises, etc.
            proc = self.model.data_preprocessor(batch, training=True)
            batch_inputs = proc["inputs"]
            data_samples = proc["data_samples"]

            loss_dict = self.model.loss(batch_inputs, data_samples)

        if not isinstance(loss_dict, dict):
            raise RuntimeError(f"model.loss() returned {type(loss_dict)}, expected dict")

        total = torch.zeros([], device=self.device)
        logs: Dict[str, float] = {}
        for name, val in loss_dict.items():
            if isinstance(val, torch.Tensor):
                v = val.mean()
            elif isinstance(val, (list, tuple)):
                tensors = [x for x in val if isinstance(x, torch.Tensor)]
                v = sum(x.mean() for x in tensors) if tensors else None
            else:
                continue
            if v is None:
                continue
            if "loss" in name:
                total = total + v
            logs[f"losses/{name}"] = float(v.detach())

        logs["loss"] = float(total.detach())
        return total, logs

    # ------------------------------------------------------------------
    # One epoch
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch: int, global_step: int):
        """Returns (epoch_mean_loss, updated_global_step)."""
        self.model.train()
        cfg = self.cfg
        scalar_agg = ScalarDict()
        iter_meter = AverageMeter()
        t0 = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        max_iters = cfg.debug_iters if cfg.debug_iters > 0 else self.steps_per_epoch
        for it, batch in enumerate(self.loader):
            if it >= max_iters:
                print(f"[Debug] Stopped at iter {it} (--debug-iters={cfg.debug_iters})")
                break
            total_loss, logs = self._forward_loss(batch)
            scalar_agg.update(logs)

            # Backward
            scaled = total_loss / max(1, cfg.accum_steps)
            if cfg.dtype == "float16":
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

            do_step = (it + 1) % cfg.accum_steps == 0

            if do_step:
                if cfg.dtype == "float16":
                    self.scaler.unscale_(self.optimizer)

                # ---- Telemetry: capture per-module gradient norms BEFORE clipping ----
                if self.telemetry is not None:
                    from .telemetry import record_grad_norms, record_losses
                    record_grad_norms(self.telemetry, self.model)
                    record_losses(self.telemetry, {k.split("/")[-1]: torch.tensor(v)
                                                   for k, v in logs.items()
                                                   if isinstance(v, (int, float))})
                    self.telemetry.step()

                if cfg.clip_grad_norm and cfg.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.clip_grad_norm
                    )
                gn = grad_norm(self.model.parameters())

                if cfg.dtype == "float16":
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

                if self.ema is not None:
                    self.ema.update(self.model)

                global_step += 1
            else:
                gn = None

            iter_meter.update(time.time() - t0)
            t0 = time.time()

            # Logging
            if (it + 1) % cfg.log_interval == 0 or (it + 1) == self.steps_per_epoch:
                done = (epoch - 1) * self.steps_per_epoch + (it + 1)
                total_iters = cfg.total_epochs * self.steps_per_epoch
                eta = max(total_iters - done, 0) * iter_meter.avg
                mem = gpu_mem_mb()
                lr_now = current_lr(self.optimizer)
                loss_val = logs["loss"]

                loss_terms = ", ".join(
                    f"{k.split('/')[-1]}={v:.4f}"
                    for k, v in logs.items()
                    if k != "loss" and "loss" in k
                )
                print(
                    f"Epoch(train) [{epoch}][{it+1:5d}/{self.steps_per_epoch}]  "
                    f"lr: {lr_now:.4e}  eta: {format_eta(eta)}  "
                    f"time: {iter_meter.val:.3f}  mem: {mem}  "
                    f"loss: {loss_val:.4f}"
                    + (f"  [{loss_terms}]" if loss_terms else "")
                    + (f"  grad_norm: {gn:.3f}" if gn is not None else "")
                )

                if self._wandb is not None:
                    try:
                        self._wandb.log(
                            {**{f"train/{k}": v for k, v in logs.items()},
                             "train/lr": lr_now,
                             "train/mem_mb": mem,
                             "global_step": global_step},
                            step=global_step,
                        )
                    except Exception:
                        pass

        return scalar_agg.mean().get("loss", 0.0), global_step

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg = self.cfg
        global_step = (self.start_epoch - 1) * self.steps_per_epoch
        saved_paths: List[str] = []

        for epoch in range(self.start_epoch, cfg.total_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{cfg.total_epochs}  |  work_dir: {cfg.work_dir}")
            print(f"{'='*70}")

            epoch_loss, global_step = self.train_one_epoch(epoch, global_step)
            print(f"[Epoch {epoch}] mean_loss={epoch_loss:.4f}")

            # Save full resume state every epoch
            resume_path = os.path.join(cfg.work_dir, "last_state.pth")
            save_checkpoint(
                resume_path,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                epoch,
                meta={"epoch": epoch, "loss": epoch_loss},
                ema=self.ema,
            )

            # Save weights-only checkpoint for evaluation
            if epoch % cfg.save_interval == 0:
                w_path = os.path.join(cfg.work_dir, f"epoch_{epoch}.pth")
                # Save det-only weights so eval Runner can load without wrapper
                det_sd = (
                    self.model.det_state_dict()
                    if hasattr(self.model, "det_state_dict")
                    else self.model.state_dict()
                )
                torch.save({"state_dict": det_sd}, w_path)
                print(f"[Checkpoint] Saved det weights → {w_path}")

                # Also save full multitask weights separately
                mt_path = os.path.join(cfg.work_dir, f"epoch_{epoch}_multitask.pth")
                save_weights(mt_path, self.model)
                print(f"[Checkpoint] Saved full multitask weights → {mt_path}")

                saved_paths.append(w_path)
                # Prune old checkpoints
                if cfg.keep_last_n > 0 and len(saved_paths) > cfg.keep_last_n:
                    old = saved_paths.pop(0)
                    old_mt = old.replace(".pth", "_multitask.pth")
                    for p in (old, old_mt):
                        if os.path.isfile(p):
                            os.remove(p)

        print(f"\n[Done] Training finished.  Checkpoints in {cfg.work_dir}")
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
