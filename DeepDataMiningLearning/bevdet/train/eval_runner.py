"""
eval_runner.py – correct NDS/mAP evaluation via mmdet3d Runner.test().

WHY this module exists
----------------------
The manual evaluation loop (custom JSON → NuScenesEval) in simple_infer_main.py
gives LOWER NDS than Runner.test() because of subtle differences in:
  • box coordinate handling during NuScenes export
  • velocity / attribute post-processing
  • NuScenes API call order

Runner.test() is the reference path that matches tools/test.py exactly.
Always use this module for final NDS numbers.

Usage (from training script after each epoch)
---------------------------------------------
    from projects.bevdet.train.eval_runner import RunnerEvaluator

    evaluator = RunnerEvaluator(
        config_path="projects/bevdet/configs/mybevfusion12v2.py",
        data_root="/data/nuscenes",
        work_dir="work_dirs/phase1",
    )
    # after each epoch:
    metrics = evaluator.evaluate(ckpt_path="work_dirs/phase1/epoch_3.pth")
    print(f"NDS={metrics['NDS']:.4f}  mAP={metrics['mAP']:.4f}")

Standalone CLI
--------------
    python -m projects.bevdet.train.eval_runner \\
        --config projects/bevdet/configs/mybevfusion12v2.py \\
        --checkpoint work_dirs/phase1/epoch_3.pth \\
        --data-root /data/nuscenes \\
        --out-dir work_dirs/phase1/eval_epoch_3
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from mmengine.config import Config
from mmengine.runner import Runner


# ---------------------------------------------------------------------------
# Lightweight hook to record latency & GPU memory during Runner.test()
# ---------------------------------------------------------------------------

class _PerfHook:
    """Attaches to runner.test_loop to track per-batch timing."""

    def __init__(self):
        import time
        self._time = time
        self.latencies = []
        self._t0 = None

    def before_test_iter(self, runner, batch_idx, data_batch):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = self._time.perf_counter()

    def after_test_iter(self, runner, batch_idx, data_batch, outputs):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._t0 is not None:
            self.latencies.append((self._time.perf_counter() - self._t0) * 1000.0)

    def perf_stats(self) -> Dict[str, float]:
        import numpy as np
        if not self.latencies:
            return {}
        lats = self.latencies
        return {
            "lat_mean_ms": float(np.mean(lats)),
            "lat_p50_ms": float(np.median(lats)),
            "lat_p95_ms": float(np.percentile(lats, 95)),
            "lat_max_ms": float(np.max(lats)),
            "n_samples": len(lats),
        }


# ---------------------------------------------------------------------------
# RunnerEvaluator
# ---------------------------------------------------------------------------

class RunnerEvaluator:
    """
    Wraps mmdet3d Runner.test() for clean epoch-by-epoch evaluation.

    Args:
        config_path:  path to mmdet3d config (.py).
                      This should be the DETECTION-only config (e.g. mybevfusion12v2.py)
                      because the Runner only evaluates detection.
        data_root:    override cfg.test_dataloader.dataset.data_root.
        work_dir:     base work dir; eval outputs go to <work_dir>/eval_<tag>.
    """

    def __init__(
        self,
        config_path: str,
        data_root: str,
        work_dir: str,
    ):
        self.config_path = config_path
        self.data_root = data_root
        self.work_dir = work_dir

    def _build_cfg(self, ckpt_path: str, out_dir: str) -> Config:
        cfg = Config.fromfile(self.config_path)
        cfg.setdefault("default_scope", "mmdet3d")
        cfg.work_dir = out_dir
        cfg.load_from = ckpt_path

        # Patch data_root in test dataloader
        for dl_name in ("test_dataloader", "val_dataloader"):
            if not hasattr(cfg, dl_name):
                continue
            dl = getattr(cfg, dl_name)
            if not isinstance(dl, dict):
                continue
            ds = dl.get("dataset", {})
            if isinstance(ds, dict) and self.data_root:
                ds["data_root"] = self.data_root

        # Ensure test pipeline is used (not train)
        cfg.train_dataloader = cfg.get("test_dataloader", cfg.get("val_dataloader", {}))

        # Strip EMAHook (and any other train-time hooks) from custom_hooks before
        # eval. Our custom Trainer manages EMA itself; the Runner's EMAHook expects
        # an EMA shadow stored in the checkpoint and crashes otherwise:
        #   RuntimeError: ... must match the size of tensor b ... at non-singleton dimension 2
        # See ema_hook.py:_swap_ema_parameters in mmengine.
        train_only_hook_types = {"EMAHook", "FreezeExceptHook"}
        if hasattr(cfg, "custom_hooks") and isinstance(cfg.custom_hooks, list):
            cfg.custom_hooks = [
                h for h in cfg.custom_hooks
                if not (isinstance(h, dict) and h.get("type") in train_only_hook_types)
            ]

        # Persist nuScenes predictions JSON to a non-temp location so post-hoc
        # per-range / per-class analysis can be run without re-eval.
        # By default mmdet3d's NuScenesMetric writes to a tempfile.mkdtemp() path
        # that gets cleaned up; we override jsonfile_prefix to <out_dir>/predictions.
        pred_prefix = os.path.join(out_dir, "predictions")
        os.makedirs(pred_prefix, exist_ok=True)
        for ev_name in ("val_evaluator", "test_evaluator"):
            ev = getattr(cfg, ev_name, None)
            if isinstance(ev, dict):
                ev["jsonfile_prefix"] = pred_prefix

        return cfg

    def evaluate(
        self,
        ckpt_path: str,
        tag: str = "",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run Runner.test() and return metrics dict.

        The ckpt_path should point to an epoch_N.pth saved by Trainer,
        which contains only the BEVFusionCA state dict (no occ_head keys)
        so strict=False loading is not required.

        Returns:
            dict with 'NDS', 'mAP' and per-class AP keys, plus 'perf' sub-dict.
        """
        if not tag:
            tag = os.path.splitext(os.path.basename(ckpt_path))[0]
        out_dir = os.path.join(self.work_dir, f"eval_{tag}")
        os.makedirs(out_dir, exist_ok=True)

        cfg = self._build_cfg(ckpt_path, out_dir)

        runner = Runner.from_cfg(cfg)

        if verbose:
            print(f"\n[Eval] Running Runner.test()  ckpt={ckpt_path}")

        metrics_raw = runner.test()

        # Flatten nested dicts (mmdet3d sometimes nests metrics)
        metrics: Dict[str, Any] = {}
        if isinstance(metrics_raw, dict):
            for k, v in metrics_raw.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        metrics[f"{k}/{kk}"] = vv
                else:
                    metrics[k] = v

        # Write JSON
        result_path = os.path.join(out_dir, "eval_results.json")
        with open(result_path, "w") as f:
            json.dump({k: (float(v) if isinstance(v, (int, float)) else v)
                       for k, v in metrics.items()}, f, indent=2)

        if verbose:
            nds = metrics.get("NDS", metrics.get("nds", "n/a"))
            mAP = metrics.get("mAP", metrics.get("map", "n/a"))
            print(f"[Eval] NDS={nds}  mAP={mAP}  →  {result_path}")

        return metrics


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser("RunnerEvaluator standalone")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", default="")
    p.add_argument("--tag", default="")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    work_dir = args.out_dir or os.path.dirname(args.checkpoint)
    ev = RunnerEvaluator(
        config_path=args.config,
        data_root=args.data_root,
        work_dir=work_dir,
    )
    ev.evaluate(ckpt_path=args.checkpoint, tag=args.tag)
