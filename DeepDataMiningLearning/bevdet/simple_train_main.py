#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified MMDet3D training script.

Features
========
1) Two training backends:
   - manual : custom PyTorch loop (AdamW + warmup+cosine LR, AMP, grad clip).
   - runner : uses MMEngine Runner defined in the MMDet3D config.

2) Evaluation after each epoch (manual backend):
   - Uses the evaluation utilities from simple_infer_utils.py:
       * run_benchmark_evaluation(...)  # Runner-based eval
       * run_manual_benchmark(...)      # Manual NuScenes/KITTI eval
   - Evaluation is run on a freshly-built model/config, using the
     checkpoint saved from the current epoch.

3) Datasets:
   - Designed for detection datasets supported in your configs
     (e.g., NuScenes, KITTI). For the evaluation step you specify
     --dataset {nuscenes,kitti} and --ann-file.

Usage example
=============

Manual training + Runner evaluation after each epoch:

    python simple_train_main.py \
        --train-backend manual \
        --eval-backend runner \
        --model-config /path/to/nuscenes_config.py \
        --model-checkpoint /path/to/init_weights.pth \
        --data-root /data/nuscenes \
        --dataset nuscenes \
        --ann-file /data/nuscenes/nuscenes_infos_val.pkl \
        --epochs 4 \
        --batch-size 2 \
        --work-dir ./outputs/my_exp

Runner training (uses config’s own training+eval):

    python simple_train_main.py \
        --train-backend runner \
        --model-config /path/to/nuscenes_config.py \
        --model-checkpoint /path/to/init_weights.pth \
        --data-root /data/nuscenes \
        --dataset nuscenes \
        --ann-file /data/nuscenes/nuscenes_infos_val.pkl \
        --work-dir ./outputs/my_exp_runner
"""

import os
import math
import argparse
from pathlib import Path
from typing import Dict, Any
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import amp

from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

from mmdet3d.registry import MODELS

# Project-local modules
from mmdet3d_models import build_model_wrapper, SaveLoadState
from mmdet3d_datasets import build_dataloaders

from simple_infer_utils import (
    get_system_info,
    run_benchmark_evaluation,
    #run_manual_benchmark,
    build_loader_pack,
    patch_cfg_paths,
    load_model_from_cfg
)

import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Sequence,
    Mapping,
    Iterable,
)
from tqdm import tqdm
# MMEngine / MMDet3D
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS, DATASETS
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.utils import register_all_modules
import mmdet3d  # for version & sanity

# NuScenes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.data_classes import DetectionBox
from pyquaternion import Quaternion
import json
import collections
def run_manual_benchmark(
    model: torch.nn.Module,
    pack: Dict[str, Any],
    class_names: Sequence[str],
    out_dir: str,
    device: Union[str, torch.device],
    eval_set: str = "val",
    detection_cfg_name: str = "detection_cvpr_2019",
    score_thresh: float = 0.05,
    max_samples: int = -1,
    sys_info: Optional[Dict[str, Any]] = None,
    dataset: str = "nuscenes",
) -> None:
    """
    Manual benchmark loop, matching the original simple_infer_main behavior.

    NuScenes:
      - Runs test_step sample-by-sample.
      - Uses meta['token'] as the *only* NuScenes sample token.
      - Writes NuScenes-format JSON.
      - Calls NuScenesEval for NDS/mAP.
      - Records latency & peak memory.

    KITTI:
      - Runs test_step sample-by-sample.
      - Records latency & peak memory.
      - Writes a small JSON with perf stats + per-class detection counts.
      - Official AP still recommended via Runner backend.

    Args:
        model:       eval-mode model.
        pack:        dict(loader, iter_fn, nusc) from build_loader_pack().
        class_names: model's class names in NuScenes order.
        out_dir:     directory for JSON + eval outputs.
        device:      torch.device or str ("cuda", "cpu").
        eval_set:    NuScenes eval split ("val", "test", etc.).
        detection_cfg_name: NuScenes config_factory name.
        score_thresh: score threshold for detections.
        max_samples:  limit number of samples for debugging (<=0 => all).
        sys_info:     system info dict (optional; stored in perf JSON).
        dataset:      "nuscenes" or "kitti".
    """

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    # ------------------------------------------------------------------
    # KITTI path
    # ------------------------------------------------------------------
    if dataset.lower() == "kitti":
        print("\n" + "=" * 60)
        print(" MANUAL BENCHMARK (KITTI)")
        print("  - Running inference + perf stats only.")
        print("  - For official KITTI AP, use Runner-based evaluation.")
        print("=" * 60)

        loader = pack["loader"]
        iter_fn = pack["iter_fn"]

        metrics = []
        per_class_counts = collections.Counter()

        pbar = tqdm(
            iter_fn(loader),
            desc="Inference (KITTI)",
            total=len(loader),
        )

        for idx, (token, pts, imgs, meta, _, _) in enumerate(pbar):
            if max_samples > 0 and idx >= max_samples:
                break

            # Prepare inputs
            inputs = {
                "points": [torch.from_numpy(pts).to(device)],
            }
            if imgs is not None:
                # imgs is already a tensor in cfg-based path
                inputs["img"] = [imgs.to(device)]

            ds = Det3DDataSample()
            ds.set_metainfo(meta)

            # Timing & memory
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                res = model.test_step(dict(inputs=inputs, data_samples=[ds]))
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)

            dt = (time.perf_counter() - t0) * 1000.0
            max_mem = (
                torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                if torch.cuda.is_available()
                else 0.0
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            metrics.append({"lat": dt, "mem": max_mem})
            pbar.set_postfix(lat=f"{dt:.1f}ms")

            # Count detections per class (for debugging / sanity)
            if res and hasattr(res[0], "pred_instances_3d"):
                pred = res[0].pred_instances_3d
                mask = pred.scores_3d > score_thresh
                lbl = pred.labels_3d[mask].cpu().numpy()
                for l in lbl:
                    if 0 <= int(l) < len(class_names):
                        per_class_counts[class_names[int(l)]] += 1

        # Aggregate perf stats
        if len(metrics) > 0:
            latencies = [m["lat"] for m in metrics]
            mems = [m["mem"] for m in metrics]
            perf = {
                "latency_mean": float(np.mean(latencies)),
                "latency_std": float(np.std(latencies)),
                "latency_min": float(np.min(latencies)),
                "latency_max": float(np.max(latencies)),
                "mem_peak": float(np.max(mems)),
                "samples": len(metrics),
                "score_thresh": float(score_thresh),
                "system_info": sys_info or {},
            }
        else:
            perf = {
                "latency_mean": 0.0,
                "latency_std": 0.0,
                "latency_min": 0.0,
                "latency_max": 0.0,
                "mem_peak": 0.0,
                "samples": 0,
                "score_thresh": float(score_thresh),
                "system_info": sys_info or {},
            }

        # Pretty print counts (like mmdet3d runner)
        if per_class_counts:
            print("\n+------------+--------+")
            print("| {:<10} | {:>6} |".format("Class", "Count"))
            print("+------------+--------+")
            for name in ["Pedestrian", "Cyclist", "Car"]:
                val = per_class_counts.get(name, 0)
                print("| {:<10} | {:>6} |".format(name, val))
            print("+------------+--------+\n")

        # Save JSON
        out_json = {
            "dataset": "kitti",
            "eval_set": eval_set,
            "perf": perf,
            "per_class_detections": dict(per_class_counts),
        }
        out_path = osp.join(out_dir, "benchmark_perf_kitti.json")
        with open(out_path, "w") as f:
            json.dump(out_json, f, indent=4)

        print("KITTI manual benchmark complete.")
        print(f"  Mean latency: {perf['latency_mean']:.2f} ms")
        print(f"  Peak memory:  {perf['mem_peak']:.2f} MB")
        print(f"  Samples:      {perf['samples']}")
        print(f"  JSON saved to {out_path}\n")
        return

    # ------------------------------------------------------------------
    # NuScenes path  (this is what needs to exactly match your old behavior)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STARTING MANUAL BENCHMARK (NuScenes)")
    print("=" * 60)

    loader = pack["loader"]
    iter_fn = pack["iter_fn"]
    nusc = pack["nusc"]

    # NuScenes JSON skeleton
    res_path = osp.join(out_dir, "nuscenes_results.json")
    results_dict = {
        "meta": {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": {},
    }

    metrics: List[Dict[str, float]] = []
    processed_tokens: List[str] = []

    pbar = tqdm(
        iter_fn(loader),
        desc="Inference (NuScenes)",
        total=len(loader),
    )

    for idx, (loader_token, pts, imgs, meta, _, _) in enumerate(pbar):
        # Respect max_samples if set (for debugging)
        if max_samples > 0 and idx >= max_samples:
            break

        # ---- CRITICAL: use meta['token'] as the ONLY NuScenes sample_token ----
        sample_token = meta.get("token", None)
        if sample_token is None:
            # Fallback: skip this sample for metrics, but still record perf.
            # This mirrors the spirit of the old code, but in practice
            # NuScenesDataset *does* provide meta['token'] for every sample.
            # We do NOT attempt any loader_token -> NuScenes token remap here.
            # Just skip for detection metrics if something is wrong.
            # (You should never see this in normal NuScenes val.)
            # We still run inference so latency stats are accurate.
            sample_token = None

        # Prepare inputs dict
        inputs = {
            "points": [torch.from_numpy(pts).to(device)],
        }
        if imgs is not None:
            inputs["img"] = [imgs.to(device)]

        ds = Det3DDataSample()
        ds.set_metainfo(meta)

        # Timing & memory
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            res = model.test_step(dict(inputs=inputs, data_samples=[ds]))
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        dt = (time.perf_counter() - t0) * 1000.0
        max_mem = (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            if torch.cuda.is_available()
            else 0.0
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        metrics.append({"lat": dt, "mem": max_mem})
        pbar.set_postfix(lat=f"{dt:.1f}ms")

        # If we couldn't get a real NuScenes token, skip detection export
        if sample_token is None:
            continue

        processed_tokens.append(sample_token)

        # Extract predictions
        pred = res[0].pred_instances_3d
        scores = pred.scores_3d
        labels = pred.labels_3d
        boxes = pred.bboxes_3d.tensor
        vels = (
            pred.velocities_3d if hasattr(pred, "velocities_3d") else None
        )
        attrs = (
            pred.attr_labels if hasattr(pred, "attr_labels") else None
        )

        mask = scores > score_thresh
        if mask.sum() == 0:
            # No boxes survive threshold; still keep an empty list to maintain
            # one entry per sample_token in results JSON.
            results_dict["results"][sample_token] = []
            continue

        box = boxes[mask].cpu().numpy()
        sc = scores[mask].cpu().numpy()
        lbl = labels[mask].cpu().numpy()

        if vels is not None:
            vels_np = vels[mask].cpu().numpy()
        elif box.shape[1] > 7:
            vels_np = box[:, 7:9]
        else:
            vels_np = None

        attrs_np = attrs[mask].cpu().numpy() if attrs is not None else None

        # Convert from LiDAR frame to NuScenes global boxes
        results_dict["results"][sample_token] = lidar_to_global_box(
            nusc=nusc,
            sample_token=sample_token,
            boxes=box,
            scores=sc,
            labels=lbl,
            class_names=class_names,
            attrs=attrs_np,
            velocities=vels_np,
        )

    # Save NuScenes-format results JSON
    with open(res_path, "w") as f:
        json.dump(results_dict, f)
    print(f"Results saved to {res_path}")

    # ------------------------------------------------------------------
    # Run NuScenes official evaluator
    # ------------------------------------------------------------------
    print("Running NuScenes Evaluator...")
    cfg_eval = config_factory(detection_cfg_name)
    nusc_eval = NuScenesEval(
        nusc,
        config=cfg_eval,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=osp.join(out_dir, "eval"),
        verbose=True,
    )

    # If you ever use max_samples > 0, we must restrict GT/pred boxes
    if max_samples > 0 and len(processed_tokens) > 0:
        from nuscenes.eval.common.loaders import load_prediction, load_gt

        nusc_eval.gt_boxes = load_gt(
            nusc_eval.nusc, nusc_eval.eval_set, DetectionBox, verbose=True
        )
        nusc_eval.pred_boxes, _ = load_prediction(
            res_path, nusc_eval.cfg.max_boxes_per_sample, DetectionBox, verbose=True
        )

        # Keep only processed tokens
        allowed = set(processed_tokens)
        nusc_eval.gt_boxes.boxes = {
            k: v for k, v in nusc_eval.gt_boxes.boxes.items() if k in allowed
        }
        nusc_eval.sample_tokens = processed_tokens

    metrics_summary, _ = nusc_eval.evaluate()
    print(
        f"\nNDS: {metrics_summary.nd_score:.4f} | "
        f"mAP: {metrics_summary.mean_ap:.4f}"
    )

    # Perf summary
    if len(metrics) > 0:
        latencies = [m["lat"] for m in metrics]
        mems = [m["mem"] for m in metrics]
        perf = {
            "latency_mean": float(np.mean(latencies)),
            "latency_std": float(np.std(latencies)),
            "latency_min": float(np.min(latencies)),
            "latency_max": float(np.max(latencies)),
            "mem_peak": float(np.max(mems)),
            "score_thresh": float(score_thresh),
            "samples": len(metrics),
        }
    else:
        perf = {
            "latency_mean": 0.0,
            "latency_std": 0.0,
            "latency_min": 0.0,
            "latency_max": 0.0,
            "mem_peak": 0.0,
            "score_thresh": float(score_thresh),
            "samples": 0,
        }

    with open(osp.join(out_dir, "benchmark_perf.json"), "w") as f:
        json.dump(perf, f, indent=4)

    print(
        f"NuScenes manual benchmark complete.\n"
        f"  NDS:           {metrics_summary.nd_score:.4f}\n"
        f"  mAP:           {metrics_summary.mean_ap:.4f}\n"
        f"  Mean latency:  {perf['latency_mean']:.2f} ms\n"
        f"  Peak memory:   {perf['mem_peak']:.2f} MB\n"
        f"  Samples:       {perf['samples']}\n"
    )


# =============================================================================
# Small utilities
# =============================================================================

class AverageMeter:
    """Momentum-based running average tracker (for timing, etc.)."""
    def __init__(self, momentum=0.98):
        self.momentum = momentum
        self.val = 0.0
        self.avg = 0.0
        self.initialized = False

    def update(self, v: float) -> None:
        self.val = float(v)
        if not self.initialized:
            self.avg = self.val
            self.initialized = True
        else:
            self.avg = self.momentum * self.avg + (1.0 - self.momentum) * self.val


def set_seed(seed: int) -> None:
    """Reproducible seeds for Python / NumPy / PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _format_eta(seconds: float) -> str:
    """Return ETA in 'D day, H:MM:SS' format."""
    seconds = int(max(0, seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d > 0:
        return f"{d} day, {h:01d}:{m:02d}:{s:02d}"
    return f"{h:01d}:{m:02d}:{s:02d}"


def _get_base_lr(optimizer, scheduler) -> float:
    """Base LR from scheduler.base_lrs or optimizer param-group."""
    try:
        if hasattr(scheduler, "base_lrs"):
            return float(scheduler.base_lrs[0])
    except Exception:
        pass
    pg0 = optimizer.param_groups[0]
    if "initial_lr" in pg0:
        return float(pg0["initial_lr"])
    return float(pg0["lr"])


def _current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _gpu_memory_mb() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return 0


def _grad_total_norm(parameters, norm_type=2.0) -> float:
    """Gradient norm. Call AFTER scaler.unscale_(optimizer)."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return 0.0
    device = params[0].grad.device
    if norm_type == float("inf"):
        total = max(p.grad.detach().abs().max().to(device) for p in params)
        return float(total.item())
    norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in params]),
        norm_type,
    )
    return float(norm.item())


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup + cosine decay to 0 (by iteration)."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * step / self.warmup_steps
            else:
                t = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * t))
            lrs.append(lr)
        return lrs


def get_param_groups(model: nn.Module, weight_decay: float):
    """Split parameters into weight-decay and no-decay groups."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".bias") or any(x in n for x in ["norm", "bn", "ln", "gn"]):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def save_state(
    path: str,
    model_wrapper: "SaveLoadState",
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best: float,
) -> None:
    """Save full training state (for resume)."""
    obj = {
        "model": model_wrapper.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_val_loss": best,
    }
    torch.save(obj, path)
    print(f"[Info] Saved state to {path}")


def load_state(
    path: str,
    model_wrapper: "SaveLoadState",
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
):
    """Resume training state."""
    state = torch.load(path, map_location=map_location)
    model_wrapper.load_state_dict(state["model"], strict=False)
    if optimizer and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    start_epoch = int(state.get("epoch", 0)) + 1
    best_val_loss = float(state.get("best_val_loss", float("inf")))
    print(
        f"[Info] Resume from {path}: start_epoch={start_epoch}, "
        f"best_val_loss={best_val_loss:.4f}"
    )
    return start_epoch, best_val_loss



def parse_kv_list(kv_list):
    out = {}
    for kv in kv_list:
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


# =============================================================================
# Evaluation helper: call simple_infer_utils after each epoch
# =============================================================================
import sys
import subprocess
def evaluate_epoch_with_simple_infer(
    args,
    epoch_idx: int,
    ckpt_path: str,
) -> Dict[str, Any]:
    """
    Run evaluation after a training epoch by DELEGATING to simple_infer_main.py.

    This avoids re-implementing the evaluation logic and guarantees we use the
    exact same behavior that already works for your standalone inference:

        python simple_infer_main.py \
            --config ... \
            --checkpoint ... \
            --dataroot ... \
            --out-dir ... \
            --data-source cfg \
            --ann-file ... \
            --dataset ... \
            --eval \
            --eval-backend {manual,runner}

    We DO NOT touch run_manual_benchmark / build_loader_pack here;
    we simply shell out to simple_infer_main.py and let it handle everything.

    Args:
        args:       original training args.
        epoch_idx:  0-based epoch index.
        ckpt_path:  path to checkpoint containing current model weights
                    (this is passed as --checkpoint to simple_infer_main.py).
    """
    print(
        f"\n[Eval] Epoch {epoch_idx + 1}: running {args.eval_backend} "
        f"evaluation using checkpoint: {ckpt_path}"
    )

    if args.eval_backend == "none":
        print("[Eval] eval-backend == none -> skipping evaluation.")
        return {}

    # ------------------------------------------------------------------
    # Fix / normalize ann_file for evaluation (directory -> *.pkl)
    # ------------------------------------------------------------------
    raw_ann_file = args.ann_file
    ann_file = raw_ann_file
    ds_name = (args.dataset or "nuscenes").lower()

    if ann_file and os.path.isdir(ann_file):
        # User passed a directory like ".../data/nuscenes/" instead of a .pkl
        if ds_name == "nuscenes":
            candidate = os.path.join(ann_file, "nuscenes_infos_val.pkl")
        elif ds_name == "kitti":
            candidate = os.path.join(ann_file, "kitti_infos_val.pkl")
        else:
            candidate = None

        if candidate is not None and os.path.exists(candidate):
            print(
                "[Eval] ann_file is a directory; "
                f"auto-resolved to annotation file: {candidate}"
            )
            ann_file = candidate
        else:
            print(
                "[Eval][WARN] ann_file is a directory and no default *.pkl "
                "was found. Please pass an explicit --ann-file path.\n"
                f"  ann_file={raw_ann_file}\n"
                f"  dataset={ds_name}"
            )

    # ------------------------------------------------------------------
    # Prepare output dir for this epoch
    # ------------------------------------------------------------------
    eval_out_dir = os.path.join(args.work_dir, f"eval_epoch_{epoch_idx + 1}")
    os.makedirs(eval_out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build the command to call simple_infer_main.py
    # ------------------------------------------------------------------
    infer_script = Path(__file__).with_name("simple_infer_main.py")
    if not infer_script.is_file():
        print(
            f"[Eval][ERROR] simple_infer_main.py not found at {infer_script}. "
            "Cannot run epoch evaluation."
        )
        return {}

    cmd = [
        sys.executable,
        str(infer_script),

        # Core config / weights
        "--config", args.model_config,
        "--checkpoint", ckpt_path,

        # Data paths
        "--dataroot", args.data_root,
        "--out-dir", eval_out_dir,
        "--data-source", "cfg",
        "--dataset", args.dataset,
        "--nus-version", args.nus_version,
        "--workers", str(args.num_workers),

        # Evaluation mode
        "--eval",
        "--eval-backend", args.eval_backend,
    ]

    # Only pass ann_file if we have one resolved
    if ann_file:
        cmd += ["--ann-file", ann_file]

    # If you want to limit samples for faster epoch-eval, map your
    # training arg (e.g., max_eval_iters) into simple_infer_main's
    # --max-samples. If you don't have such an arg, you can drop this.
    max_samples = getattr(args, "max_eval_iters", 0)
    if max_samples and max_samples > 0:
        cmd += ["--max-samples", str(max_samples)]

    print("[Eval] Launching simple_infer_main.py as subprocess:")
    print("       " + " ".join(cmd))

    # ------------------------------------------------------------------
    # Run subprocess and capture output
    # ------------------------------------------------------------------
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print("\n[Eval] simple_infer_main.py output (epoch {}):\n".format(epoch_idx + 1))
    print(result.stdout)

    if result.returncode != 0:
        print(
            f"[Eval][ERROR] simple_infer_main.py exited with code {result.returncode}. "
            "See log above for details."
        )
        return {}

    # simple_infer_main.py already writes metrics / benchmark JSON into eval_out_dir.
    # If in the future you want to parse a "best" metric (e.g., NDS, mAP),
    # you can read the JSON here and return a dict. For now, just return {}.
    return {}

def evaluate_epoch_with_simple_inferv1(
    args,
    epoch_idx: int,
    ckpt_path: str,
) -> Dict[str, Any]:
    """
    Run evaluation after a training epoch using the same model + dataset
    setup as simple_infer_main.py.

    For eval_backend == 'runner':
        -> run_benchmark_evaluation(eval_args, sys_info)

    For eval_backend == 'manual':
        -> load_model_from_cfg(...)
        -> build_loader_pack(...)
        -> run_manual_benchmark(...)

    Args:
        args:       original training args.
        epoch_idx:  0-based epoch index.
        ckpt_path:  checkpoint to evaluate (usually epoch_{E:03d}_weights.pth).
    """
    print(
        f"\n[Eval] Epoch {epoch_idx + 1}: running {args.eval_backend} "
        f"evaluation using checkpoint: {ckpt_path}"
    )

    if args.eval_backend == "none":
        print("[Eval] eval-backend == none -> skipping evaluation.")
        return {}

    # ------------------------------------------------------------------
    # Normalize ann_file (directory -> *infos_val.pkl)
    # ------------------------------------------------------------------
    raw_ann_file = args.ann_file
    ann_file = raw_ann_file
    ds_name = (args.dataset or "nuscenes").lower()

    if ann_file and os.path.isdir(ann_file):
        # e.g.  /.../data/nuscenes/  or  /.../data/kitti/
        if ds_name == "nuscenes":
            candidate = os.path.join(ann_file, "nuscenes_infos_val.pkl")
        elif ds_name == "kitti":
            candidate = os.path.join(ann_file, "kitti_infos_val.pkl")
        else:
            candidate = None

        if candidate is not None and os.path.exists(candidate):
            print(
                "[Eval] ann_file is a directory; "
                f"auto-resolved to annotation file: {candidate}"
            )
            ann_file = candidate
        else:
            print(
                "[Eval][WARN] ann_file is a directory and no default *.pkl "
                "was found. Please pass an explicit --ann-file path.\n"
                f"  ann_file={raw_ann_file}\n"
                f"  dataset={ds_name}"
            )

    # System info (for metrics JSON)
    sys_info = get_system_info()

    # Epoch-specific output directory, like simple_infer_main's out_dir
    eval_out_dir = os.path.join(args.work_dir, f"eval_epoch_{epoch_idx + 1}")
    os.makedirs(eval_out_dir, exist_ok=True)

    # Construct a "mini args" container that looks like simple_infer_main's args
    # (this mirrors what you pasted from the working inference main())
    max_samples = -1
    if getattr(args, "max_eval_iters", 0) and args.max_eval_iters > 0:
        max_samples = int(args.max_eval_iters)

    eval_args = SimpleNamespace(
        config=args.model_config,
        checkpoint=ckpt_path,
        dataroot=args.data_root,
        out_dir=eval_out_dir,
        dataset=args.dataset,
        ann_file=ann_file,
        device="cuda" if torch.cuda.is_available() else "cpu",
        nus_version=args.nus_version,
        max_samples=max_samples,     # -1 = full eval, >0 = subsample
        workers=args.num_workers,
        data_source="cfg",
        benchmark_type="manual",
        eval=True,
        eval_backend=args.eval_backend,
        crop_policy="center",
        eval_set="val",
        score_thresh=0.05,
    )

    # ------------------------------------------------------------------
    # PATH 1: Runner-based evaluation (unchanged)
    # ------------------------------------------------------------------
    if args.eval_backend == "runner":
        print(
            f"[Eval] Using Runner-based evaluation with ann_file={eval_args.ann_file}"
        )
        run_benchmark_evaluation(eval_args, sys_info)
        return {}

    # ------------------------------------------------------------------
    # PATH 2: Manual backend (NuScenesEval / KITTI perf) – match simple_infer_main
    # ------------------------------------------------------------------
    print(
        f"[Eval] Using MANUAL evaluation backend with ann_file={eval_args.ann_file}"
    )

    # 1) Build model + cfg EXACTLY like simple_infer_main.main()
    #    - This function already:
    #        * loads the config
    #        * optionally patches paths based on dataroot / ann_file
    #        * builds the model
    #        * loads `checkpoint_path` weights
    #    - We pass checkpoint_path=ckpt_path (current epoch weights).
    model, cfg = load_model_from_cfg(
        config_path=eval_args.config,
        checkpoint_path=eval_args.checkpoint,
        device=eval_args.device,
        # For data_source == 'cfg', the loader pack uses cfg.test_dataloader,
        # so we let utils patch them with this dataroot:
        dataroot=eval_args.dataroot if eval_args.data_source == "cfg" else None,
        ann_file=eval_args.ann_file,
        work_dir=eval_args.out_dir,
    )

    # 2) Build loader pack EXACTLY as in simple_infer_main.main()
    pack = build_loader_pack(
        data_source=eval_args.data_source,
        cfg=cfg,
        dataroot=eval_args.dataroot,
        nus_version=eval_args.nus_version,
        ann_file=eval_args.ann_file,
        max_samples=eval_args.max_samples,
        crop_policy=eval_args.crop_policy,
        workers=eval_args.workers,
        dataset=eval_args.dataset,
    )

    # 3) Call run_manual_benchmark with the SAME signature as inference
    run_manual_benchmark(
        model=model,
        pack=pack,
        class_names=cfg.class_names,
        out_dir=eval_args.out_dir,
        device=eval_args.device,
        eval_set=eval_args.eval_set,
        detection_cfg_name="detection_cvpr_2019",
        score_thresh=eval_args.score_thresh,
        max_samples=eval_args.max_samples,
        sys_info=sys_info,
        dataset=eval_args.dataset,
    )

    # Metrics are written by run_manual_benchmark into JSON files,
    # so the training loop doesn't need anything specific here.
    return {}
# =============================================================================
# Training backend: Runner
# =============================================================================

def run_training_with_runner(args) -> None:
    """
    Use MMEngine Runner to train, as defined in the MMDet3D config.

    Evaluation during training is handled by the config:
      - val_dataloader / val_evaluator / val_cfg
      - default_hooks.checkpoint, logger, etc.

    We patch:
      - work_dir
      - load_from (initial weights)
      - train/val/test dataloader dataset.data_root
      - ann_file for val/test datasets (using args.ann_file)
      - max_epochs (if present) and val interval (if available).
    """
    print("\n[Runner Training] Building Runner from config...")
    cfg = Config.fromfile(args.model_config)
    cfg.setdefault("default_scope", "mmdet3d")

    cfg.work_dir = args.work_dir
    if args.model_checkpoint:
        cfg.load_from = args.model_checkpoint

    # Patch data_root & ann_file for train/val/test datasets
    def _patch_dl(dl_name: str):
        if not hasattr(cfg, dl_name):
            return
        dl = getattr(cfg, dl_name)
        if not isinstance(dl, dict):
            return
        ds = dl.get("dataset", None)
        if isinstance(ds, dict):
            if args.data_root and "data_root" in ds:
                ds["data_root"] = args.data_root
            if args.ann_file and "ann_file" in ds:
                ds["ann_file"] = args.ann_file

    for dn in ["train_dataloader", "val_dataloader", "test_dataloader"]:
        _patch_dl(dn)

    # Epoch count
    if hasattr(cfg, "train_cfg") and hasattr(cfg.train_cfg, "max_epochs"):
        cfg.train_cfg.max_epochs = args.epochs
    elif "max_epochs" in cfg:
        cfg["max_epochs"] = args.epochs

    # Try to ensure validation runs every epoch (if val_cfg exists)
    if hasattr(cfg, "val_cfg") and hasattr(cfg.val_cfg, "interval"):
        cfg.val_cfg.interval = 1

    runner = Runner.from_cfg(cfg)
    print("[Runner Training] Starting training...")
    runner.train()
    print("[Runner Training] Finished.")


# =============================================================================
# Training backend: Manual loop
# =============================================================================

def manual_training_loop(args) -> None:
    """
    Manual training loop.

    - Uses build_model_wrapper / build_dataloaders to construct model+data.
    - Optimizer: AdamW on param-groups with weight decay.
    - LR: warmup+cosine over iterations.
    - AMP: optional (torch.amp GradScaler).
    - After each epoch, saves:
        * last_state.pth  (for resume)
        * epoch_{E:03d}_weights.pth (state_dict only, for eval)
      and optionally runs evaluation via evaluate_epoch_with_simple_infer().
    """
    os.makedirs(args.work_dir, exist_ok=True)
    set_seed(args.seed)

    extra = parse_kv_list(args.extra)
    if args.data_root:
        extra["data_root"] = args.data_root

    # --- Build model wrapper ---
    print("[Manual Training] Building model wrapper...")
    model_wrapper = build_model_wrapper(
        backend=args.model_backend,
        model_config=args.model_config,
        checkpoint=args.model_checkpoint,
        extra=extra,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper.to(device)
    model = model_wrapper.unwrap()  # raw MMDet3D detector (e.g., BEVFusion)

    # Move data_preprocessor to the same device as the model
    dp = getattr(model, "data_preprocessor", None)
    if dp is not None:
        dp.to(device)
        print(f"[DEBUG] data_preprocessor moved to device={device}")

    # --- Build dataloaders ---
    print("[Manual Training] Building dataloaders...")
    data_cfg = args.data_config or args.model_config
    train_loader, val_loader = build_dataloaders(
        backend=args.data_backend,
        data_config=data_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        extra=extra,
    )

    print(
        f"[build_dataloaders] Built train_dataloader: "
        f"len={len(train_loader)}, batch_size={train_loader.batch_size}, "
        f"num_workers={train_loader.num_workers}"
    )
    if val_loader is not None:
        print(
            f"[build_dataloaders] Built val_dataloader: "
            f"len={len(val_loader)}, batch_size={val_loader.batch_size}, "
            f"num_workers={val_loader.num_workers}"
        )

    if args.inspect_dataset > 0:
        print(
            f"[Note] Dataset inspection requested for first {args.inspect_dataset} "
            "batches (stub; add your own debug prints here if needed)."
        )

    # --- Optional Weights & Biases init ---
    wandb_run = None
    if getattr(args, "wandb", False):
        try:
            import wandb  # type: ignore

            run_name = args.wandb_run_name or Path(args.work_dir).name
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "warmup_epochs": args.warmup_epochs,
                    "model_config": args.model_config,
                    "data_root": args.data_root,
                },
            )
            print(f"[W&B] Logging enabled: project={args.wandb_project}, run={run_name}")
        except Exception as e:
            print(f"[W&B] Failed to init Weights & Biases: {e}")
            wandb_run = None

    # --- Optimizer / Scheduler / AMP ---
    optim_params = get_param_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler = amp.GradScaler(device="cuda", enabled=args.fp16)

    # --- Resume (optional) ---
    start_epoch, best_val_loss = 0, float("inf")
    if args.resume_from and os.path.isfile(args.resume_from):
        start_epoch, best_val_loss = load_state(
            args.resume_from,
            model_wrapper,
            optimizer,
            scheduler,
            scaler,
            map_location=device,
        )
        scheduler.last_epoch = start_epoch * steps_per_epoch

    # Evaluation
    print("Start initial evaluation...")
    eval_results = evaluate_epoch_with_simple_infer(
                args=args,    
                epoch_idx=0,
                ckpt_path=args.model_checkpoint,
            )
    print("Initial Evaluation results:", eval_results)
    
    # --- Training loop ---
    global_step = start_epoch * steps_per_epoch
    best_metric_map: Dict[str, float] = {}
    sys_info = get_system_info()

    log_interval = getattr(args, "log_interval", 50)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n[Epoch {epoch + 1}/{args.epochs}] Starting training...")
        model_wrapper.train()
        optimizer.zero_grad(set_to_none=True)

        iter_time_meter = AverageMeter()
        data_time_meter = AverageMeter()
        last_iter_end = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for it, batch in enumerate(train_loader):
            now = time.time()
            data_time_meter.update(now - last_iter_end)

            # For mmdet3d backend, pass CPU batch directly into data_preprocessor
            if args.model_backend == "mmdet3d":
                batch_for_dp = batch
            else:
                batch_for_dp = model_wrapper.move_batch_to_device(batch, device)

            with amp.autocast(device_type="cuda", enabled=args.fp16):
                # Preprocess (move to device, normalize, collate)
                proc = model.data_preprocessor(batch_for_dp, training=True)
                batch_inputs = proc["inputs"]       # dict: 'points', 'img'/'imgs'
                data_samples = proc["data_samples"] # list[Det3DDataSample]

                # Loss dict from MMDet3D model
                loss_dict = model.loss(batch_inputs, data_samples)
                if not isinstance(loss_dict, dict):
                    raise RuntimeError(
                        f"Expected dict from model.loss(...), got {type(loss_dict)}"
                    )

                # Aggregate total_loss, collect per-term logs
                total_loss = torch.zeros([], device=device)
                logs: Dict[str, float] = {}
                for name, val in loss_dict.items():
                    if isinstance(val, torch.Tensor):
                        loss_val = val.mean()
                    elif isinstance(val, (list, tuple)):
                        tensors = [x for x in val if isinstance(x, torch.Tensor)]
                        if not tensors:
                            continue
                        loss_val = sum(x.mean() for x in tensors)
                    else:
                        continue

                    if "loss" in name:
                        total_loss = total_loss + loss_val
                    logs[f"losses/{name}"] = float(loss_val.detach().cpu())

                logs["loss"] = float(total_loss.detach().cpu())

            # Backward with optional grad accumulation
            loss_scaled = total_loss / max(1, args.accum_steps)
            if args.fp16:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            grad_norm_value = None
            if (it + 1) % args.accum_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                if args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_norm
                    )
                try:
                    grad_norm_value = _grad_total_norm(model.parameters())
                except Exception:
                    grad_norm_value = None

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            global_step += 1

            iter_time = time.time() - now
            iter_time_meter.update(iter_time)
            last_iter_end = time.time()

            # ETA / logging
            iter_done = epoch * steps_per_epoch + (it + 1)
            iters_total = args.epochs * steps_per_epoch
            eta_seconds = max(iters_total - iter_done, 0) * max(
                iter_time_meter.avg, 1e-6
            )

            mem_mb = _gpu_memory_mb()
            base_lr = _get_base_lr(optimizer, scheduler)
            lr_now = _current_lr(optimizer)
            total_loss_val = float(total_loss.detach().cpu())

            # ---- Console printing (with individual loss terms) ----
            if (it + 1) % log_interval == 0 or (it + 1) == steps_per_epoch:
                head = f"Epoch(train) [{epoch + 1}][{it + 1:5d}/{steps_per_epoch}]"
                fields = [
                    f"base_lr: {base_lr:.4e}",
                    f"lr: {lr_now:.4e}",
                    f"eta: {_format_eta(int(eta_seconds))}",
                    f"time: {iter_time_meter.val:.4f}",
                    f"data_time: {data_time_meter.val:.4f}",
                    f"memory: {mem_mb}",
                    f"loss: {total_loss_val:.4f}",
                ]
                if grad_norm_value is not None:
                    fields.append(f"grad_norm: {grad_norm_value:.4f}")

                # Pretty-print individual loss components (strip "losses/" prefix)
                loss_terms_str = ", ".join(
                    f"{k.split('/')[-1]}={v:.4f}"
                    for k, v in logs.items()
                    if k != "loss"
                )
                if loss_terms_str:
                    fields.append(f"[{loss_terms_str}]")

                print(head, "  ", "  ".join(fields))

            # ---- Optional Weights & Biases logging ----
            if wandb_run is not None and ((it + 1) % log_interval == 0):
                try:
                    import wandb  # type: ignore

                    wb_dict = {
                        "train/loss": total_loss_val,
                        "train/lr": lr_now,
                        "train/memory_mb": mem_mb,
                        "train/epoch": epoch + 1,
                        "train/iter_in_epoch": it + 1,
                        "global_step": global_step,
                    }
                    for k, v in logs.items():
                        wb_dict[f"train/{k}"] = v
                    if grad_norm_value is not None:
                        wb_dict["train/grad_norm"] = grad_norm_value

                    wandb.log(wb_dict, step=global_step)
                except Exception as e:
                    print(f"[W&B] log failed: {e}")

        # --- Save resume state ---
        save_state(
            os.path.join(args.work_dir, "last_state.pth"),
            model_wrapper,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_loss,
        )

        # --- Save weights-only checkpoint for evaluation ---
        epoch_ckpt = os.path.join(
            args.work_dir, f"epoch_{epoch + 1:03d}_weights.pth"
        )
        torch.save(
            {"state_dict": model.state_dict()},
            epoch_ckpt,
        )
        print(f"[Info] Saved epoch weights to {epoch_ckpt}")

        # --- Optional per-epoch evaluation ---
        if args.eval_interval > 0 and ((epoch + 1) % args.eval_interval == 0):
            eval_results = evaluate_epoch_with_simple_infer(
                args=args,
                epoch_idx=epoch,
                ckpt_path=epoch_ckpt,
            )

            best_key = args.save_best_metric
            if best_key and best_key in eval_results:
                cur = float(eval_results[best_key])
                prev = best_metric_map.get(best_key, None)
                if prev is None:
                    better = True
                else:
                    better = cur < prev if "loss" in best_key.lower() else cur > prev
                if better:
                    best_metric_map[best_key] = cur
                    best_path = os.path.join(
                        args.work_dir, f"best_by_{best_key}.pth"
                    )
                    torch.save(
                        {"state_dict": model.state_dict()},
                        best_path,
                    )
                    print(
                        f"[Best] Updated {best_key}: {prev} -> {cur}, "
                        f"saved to {best_path}"
                    )

    final_path = os.path.join(args.work_dir, "final_model_only.pth")
    torch.save(model_wrapper.state_dict(), final_path)
    print(f"[Done] Training completed. Final model saved to {final_path}")

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

# =============================================================================
# Main
# =============================================================================
# =============================================================================
# Argparse
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser("Unified MMDet3D training (manual / Runner)")

    # Core experiment settings
    p.add_argument("--work-dir", default="outputs/mm3dtrain", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--epochs", default=4, type=int)
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--num-workers", default=0, type=int)
    p.add_argument("--accum-steps", default=1, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--weight-decay", default=0.01, type=float)
    p.add_argument("--clip-grad-norm", default=None, type=float)
    p.add_argument("--warmup-epochs", default=0.0, type=float)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--no-pin-memory", action="store_true")
        # Logging / W&B
    p.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print/log every N iterations.",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="mmdet3d-training",
        help="W&B project name.",
    )
    p.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name (defaults to work_dir name).",
    )

    # Backends
    p.add_argument(
        "--train-backend",
        choices=["manual", "runner"],
        default="manual",
        help="manual: custom loop; runner: use MMEngine Runner.",
    )
    p.add_argument(
        "--eval-backend",
        choices=["runner", "manual", "none"],
        default="manual",
        help=(
            "Evaluation backend for manual training:\n"
            "  runner: run_benchmark_evaluation (Runner.test())\n"
            "  manual: run_manual_benchmark (NuScenes/KITTI)\n"
            "  none:   skip evaluation after epochs\n"
            "NOTE: For train-backend=runner, evaluation is handled by the config."
        ),
    )

    # Model / data config for MMDet3D
    p.add_argument(
        "--model-config",
        type=str,
        #default="/data/rnd-liu/MyRepo/mmdetection3d/projects/bevdet/configs/mybevfusion9_new2.py",
        default="/data/rnd-liu/MyRepo/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
        help="Path to MMDet3D config (used for both training and eval).",
    )
    p.add_argument(
        "--model-checkpoint",
        type=str,
        #default="/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/mybevfusion9_new2/epoch_6.pth",
        default="/data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth",
        help="Initial weights to load (pretrained checkpoint).",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="/data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes",
        help="Dataset root. Used to override cfg.*_dataloader.dataset.data_root.",
    )

    # Dataset meta for evaluation
    p.add_argument(
        "--dataset",
        type=str,
        default="nuscenes",
        choices=["nuscenes", "kitti"],
        help="Dataset name (used in evaluation functions).",
    )
    p.add_argument(
        "--ann-file",
        type=str,
        default="/data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes/",
        help="Path to val/test info pkl (e.g., nuscenes_infos_val.pkl, kitti_infos_val.pkl).",
    )
    p.add_argument(
        "--nus-version",
        type=str,
        default="v1.0-trainval",
        help="NuScenes version (if dataset=nuscenes).",
    )

    # Evaluate frequency (manual backend only)
    p.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluate every N epochs (manual backend only; 0 disables).",
    )
    p.add_argument(
        "--max-eval-iters",
        type=int,
        default=0,
        help="Limit validation iters for quick eval (0=all) in manual eval backend.",
    )

    # Choose which metric to track as "best"
    p.add_argument(
        "--save-best-metric",
        type=str,
        default=None,
        help="Metric key to track for best checkpoint, e.g., 'NDS' or 'mAP_3d'.",
    )

    # Generic backends for the model wrapper / dataloaders (to keep your old infra)
    p.add_argument(
        "--model-backend",
        choices=["mmdet3d", "openpcdet", "custom"],
        default="mmdet3d",
    )
    p.add_argument(
        "--data-backend",
        choices=["mmdet3d", "custom"],
        default="mmdet3d",
    )
    p.add_argument(
        "--data-config",
        type=str,
        default="",
        help="Path to dataset config (backend-specific). If empty, uses model-config.",
    )

    p.add_argument(
        "--inspect-dataset",
        type=int,
        default=0,
        help="If >0, print that we would inspect the first N batches (stub).",
    )

    # Simple extra key=val pairs (optional)
    p.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Extra key=val pairs passed to build_model_wrapper / build_dataloaders.",
    )

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    print(f"[Main] work_dir = {args.work_dir}")
    print(f"[Main] train_backend = {args.train_backend}, eval_backend = {args.eval_backend}")

    if args.train_backend == "runner":
        run_training_with_runner(args)
    else:
        manual_training_loop(args)


if __name__ == "__main__":
    main()