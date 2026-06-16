"""
train.py – clean entry point for MultiTaskBEVFusion training.

Replaces mmdetection3d/tools/train.py for our custom training loop.
mmdet3d is still used for:
  • data loading     (cfg.train_dataloader, same as before)
  • model construction (MODELS.build from config)
  • evaluation        (Runner.test() via eval_runner.py)

Everything else is plain PyTorch.

Quick start
-----------
Phase 1 – detection only (exactly like tools/train.py but with our loop):

    conda activate py310 && cd /data/rnd-liu/MyRepo/mmdetection3d
    python projects/bevdet/train/train.py \\
        --config projects/bevdet/configs/mybevfusion12v2.py \\
        --work-dir work_dirs/phase1_det \\
        --epochs 10 \\
        --eval-interval 2

Phase 1 – detection + occupancy (binary, no Occ3D dataset):

    python projects/bevdet/train/train.py \\
        --config projects/bevdet/configs/mybevfusion_phase1.py \\
        --work-dir work_dirs/phase1_det_occ \\
        --epochs 10 \\
        --occ-classes 2 \\
        --occ-num-z 16 \\
        --occ-loss-weight 0.5

Resume interrupted training:

    python projects/bevdet/train/train.py \\
        --config projects/bevdet/configs/mybevfusion12v2.py \\
        --work-dir work_dirs/phase1_det \\
        --resume

Evaluate only (runs Runner.test() on a saved checkpoint):

    python projects/bevdet/train/train.py \\
        --config projects/bevdet/configs/mybevfusion12v2.py \\
        --eval-only \\
        --load-from work_dirs/phase1_det/epoch_10.pth \\
        --data-root /data/nuscenes \\
        --work-dir work_dirs/phase1_det

Notes
-----
• The config's train_dataloader is used unchanged – same augmentations
  and pipelines as when running tools/train.py directly.
• Evaluation always uses Runner.test() (correct NDS, matches mmdet3d).
• The per-epoch checkpoint epoch_N.pth contains ONLY the BEVFusionCA
  weights so the Runner can load them without the occ_head wrapper.
• epoch_N_multitask.pth contains all weights including occ_head.
"""

from __future__ import annotations

import argparse
import os
import sys

# ---- Package bootstrap -------------------------------------------------------
# Allow running as a plain script:  python projects/bevdet/train/train.py ...
# Without this, relative imports (.checkpoint, .trainer, …) fail because Python
# treats __main__ as having no parent package.
if __name__ == "__main__" and __package__ is None:
    import importlib
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 3 levels: train/ → bevdet/ → projects/ → mmdetection3d/
    _pkg_root = os.path.normpath(os.path.join(_script_dir, "..", "..", ".."))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    __package__ = "projects.bevdet.train"
    importlib.import_module(__package__)  # triggers __init__.py
# ------------------------------------------------------------------------------

import torch
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules
from mmengine.config import Config
from mmengine.registry import init_default_scope


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean multi-task training for BEVFusion",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ---- Paths ----
    p.add_argument("--config", required=True,
                   help="Path to mmdet3d config (.py)")
    p.add_argument("--work-dir", default="work_dirs/phase1",
                   help="Directory for checkpoints and logs")
    p.add_argument("--load-from", default="",
                   help="Initial weights (pre-trained checkpoint)")
    p.add_argument("--data-root", default="",
                   help="Override dataset data_root in config")

    # ---- Training schedule ----
    p.add_argument("--epochs", type=int, default=0,
                   help="Total epochs (0 = use config's max_epochs)")
    p.add_argument("--lr", type=float, default=0.0,
                   help="Base learning rate (0 = use config value)")
    p.add_argument("--batch-size", type=int, default=0,
                   help="Per-GPU batch size (0 = use config value)")
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--clip-grad-norm", type=float, default=20.0)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="AMP dtype: bfloat16 for H100, float16 for older GPUs")

    # ---- Occupancy head ----
    p.add_argument("--occ-classes", type=int, default=0,
                   help="Number of occ classes (0=no occ head). "
                        "2=binary LiDAR pseudo-occ, 17=Occ3D-nuScenes")
    p.add_argument("--occ-num-z", type=int, default=16,
                   help="Number of Z slices in occupancy voxel grid")
    p.add_argument("--occ-loss-weight", type=float, default=1.0,
                   help="Weight multiplied on ALL occ loss terms")
    p.add_argument("--occ-head-type", choices=["dense", "query"], default="dense",
                   help="'dense' = BEVOccHead (3D-conv voxel logits); "
                        "'query' = QueryOccHead (UNet world tensor + MLP queries, P2.11)")
    # ---- Dense BEVOccHead-only options ----
    p.add_argument("--occ-hidden", type=int, default=128,
                   help="[dense] Hidden channels in BEVOccHead")
    p.add_argument("--occ-lovasz-weight", type=float, default=1.0,
                   help="[dense] Weight on Lovász loss term (0 to disable)")
    # ---- QueryOccHead-only options ----
    p.add_argument("--occ-world-channels", type=int, default=64,
                   help="[query] World tensor channels (compressed BEV)")
    p.add_argument("--occ-mlp-hidden", type=int, default=128,
                   help="[query] MLP hidden size")
    p.add_argument("--occ-pos-freqs", type=int, default=6,
                   help="[query] Sinusoidal positional encoding frequencies")
    p.add_argument("--occ-train-max-queries", type=int, default=50000,
                   help="[query] Cap on queries per batch sample during training "
                        "(controls Q-Occ memory savings)")

    # ---- Depth head (BEVDepth-style) ----
    p.add_argument("--depth-head", action="store_true",
                   help="Enable BEVDepth-style depth distribution supervision "
                        "on the camera FPN P3 features")
    p.add_argument("--depth-bins", type=int, default=59,
                   help="Number of depth bins. With dbound=(1,60,1) → 59 bins")
    p.add_argument("--depth-hidden", type=int, default=128,
                   help="Hidden channels in DepthHead")
    p.add_argument("--depth-loss-weight", type=float, default=1.0,
                   help="Weight on the depth loss term")

    # ---- Dedicated camera-only detection head (modality-agnostic backbone) ----
    p.add_argument("--cam-head", action="store_true",
                   help="Add a 2nd detection head (clone of cfg.model.bbox_head) "
                        "trained on the camera-only BEV. Pairs with --depth-head.")
    p.add_argument("--cam-loss-weight", type=float, default=1.0,
                   help="Weight on the camera-head detection loss terms")

    # ---- EMA / checkpointing ----
    p.add_argument("--ema", action="store_true", help="Enable EMA")
    p.add_argument("--ema-decay", type=float, default=0.9998)
    p.add_argument("--save-interval", type=int, default=1)
    p.add_argument("--keep-last-n", type=int, default=3)

    # ---- Evaluation ----
    p.add_argument("--eval-interval", type=int, default=0,
                   help="Evaluate every N epochs (0 = disable during training)")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; just run Runner.test() on --load-from")

    # ---- Resume ----
    p.add_argument("--resume", action="store_true",
                   help="Resume from <work-dir>/last_state.pth")
    p.add_argument("--resume-from", default="",
                   help="Resume from explicit path")

    # ---- Logging ----
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="bevfusion-multitask")
    p.add_argument("--wandb-run-name", default="")

    # ---- Telemetry (model-internals diagnostics) ----
    p.add_argument("--telemetry", action="store_true",
                   help="Enable internal telemetry hooks: BEV feature stats, "
                        "cross-modal alignment, per-module grad norms, per-loss "
                        "values. Writes JSONL to <work-dir>/telemetry.jsonl. "
                        "Cheap (~ms/iter); useful for understanding what the "
                        "model is doing internally.")
    p.add_argument("--telemetry-every", type=int, default=100,
                   help="Flush a telemetry summary every N iters (default 100)")

    # ---- Misc ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=-1,
                   help="Override num_workers (-1 = use config value, 0 = single-process)")
    p.add_argument("--debug-iters", type=int, default=0,
                   help="Stop each epoch after N iters (0=full). Use for quick smoke-test.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Setup mmdet3d environment
# ---------------------------------------------------------------------------

def setup_env():
    """Register all mmdet3d modules and set default scope."""
    register_all_modules(init_default_scope=False)
    init_default_scope("mmdet3d")


# ---------------------------------------------------------------------------
# Config patching helpers
# ---------------------------------------------------------------------------

def _patch_data_root(cfg: Config, data_root: str) -> None:
    if not data_root:
        return
    for dl_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if not hasattr(cfg, dl_name):
            continue
        dl = getattr(cfg, dl_name)
        if not isinstance(dl, dict):
            continue
        for ds_key in ("dataset",):
            ds = dl.get(ds_key, {})
            if isinstance(ds, dict):
                if "data_root" in ds:
                    ds["data_root"] = data_root
                # Nested (e.g. ConcatDataset / RepeatDataset)
                inner = ds.get("dataset", {})
                if isinstance(inner, dict) and "data_root" in inner:
                    inner["data_root"] = data_root


def _patch_batch_size(cfg: Config, batch_size: int) -> None:
    if batch_size <= 0:
        return
    if hasattr(cfg, "train_dataloader"):
        cfg.train_dataloader["batch_size"] = batch_size


def _patch_workers(cfg: Config, workers: int) -> None:
    if workers < 0:  # -1 = don't override
        return
    for dl_name in ("train_dataloader", "val_dataloader"):
        if not hasattr(cfg, dl_name):
            continue
        dl = getattr(cfg, dl_name)
        dl["num_workers"] = workers
        if workers == 0:
            # prefetch_factor requires num_workers > 0
            dl.pop("prefetch_factor", None)


def _get_max_epochs(cfg: Config) -> int:
    if hasattr(cfg, "train_cfg") and isinstance(cfg.train_cfg, dict):
        return int(cfg.train_cfg.get("max_epochs", 10))
    return int(getattr(cfg, "max_epochs", 10))


def _get_base_lr(cfg: Config) -> float:
    ow = getattr(cfg, "optim_wrapper", {})
    if isinstance(ow, dict):
        opt = ow.get("optimizer", {})
        if isinstance(opt, dict) and "lr" in opt:
            return float(opt["lr"])
    return 2e-4


# ---------------------------------------------------------------------------
# Build train DataLoader from config
# ---------------------------------------------------------------------------

def build_train_loader(cfg: Config) -> DataLoader:
    """
    Build the training DataLoader exactly as mmdet3d Runner would.
    Uses mmengine Runner.build_dataloader (same path as tools/train.py).
    Signature: build_dataloader(dataloader_cfg, seed=None, diff_rank_seed=False)
    """
    from mmengine.runner import Runner
    return Runner.build_dataloader(cfg.train_dataloader, seed=42)


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(
    cfg: Config,
    load_from: str,
    occ_classes: int,
    occ_num_z: int,
    occ_hidden: int,
    occ_lovasz_weight: float,
    occ_loss_weight: float,
    device: torch.device,
    depth_head_enabled: bool = False,
    depth_bins: int = 59,
    depth_hidden: int = 128,
    depth_loss_weight: float = 1.0,
    cam_head_enabled: bool = False,
    cam_loss_weight: float = 1.0,
    occ_head_type: str = "dense",
    occ_world_channels: int = 64,
    occ_mlp_hidden: int = 128,
    occ_pos_freqs: int = 6,
    occ_train_max_queries: int = 50000,
) -> torch.nn.Module:
    """
    1. Build BEVFusionCA (or whatever model the config specifies).
    2. Load pretrained weights (strict=False to allow architecture changes).
    3. Optionally wrap in MultiTaskBEVFusion when any auxiliary head
       (occ / depth) is requested.
    """
    # Build base detector from config
    det_model = MODELS.build(cfg.model)

    # Load pretrained / resume weights
    if load_from and os.path.isfile(load_from):
        from .checkpoint import load_weights
        load_weights(load_from, det_model, strict=False)
    elif cfg.get("load_from"):
        from mmengine.runner import load_checkpoint
        load_checkpoint(det_model, cfg.load_from, map_location="cpu")

    # Read camera input/feature sizes + dbound from the config's view_transform
    vt_cfg = cfg.model.get("view_transform", {}) if isinstance(cfg.model, dict) else {}
    image_size   = tuple(vt_cfg.get("image_size",   (256, 704)))
    feature_size = tuple(vt_cfg.get("feature_size", (32, 88)))
    dbound       = tuple(vt_cfg.get("dbound",       (1.0, 60.0, 1.0)))
    cam_in_ch    = int(vt_cfg.get("in_channels",    256))

    # Build auxiliary heads (lazy imports keep this file standalone)
    occ_head = None
    if occ_classes > 0:
        if occ_head_type == "query":
            from .query_occ_head import QueryOccHead
            # H, W = camera-BEV grid (typically 180×180 from xbound/ybound + downsample)
            xb = vt_cfg.get("xbound", (-54.0, 54.0, 0.3))
            yb = vt_cfg.get("ybound", (-54.0, 54.0, 0.3))
            ds = int(vt_cfg.get("downsample", 2))
            grid_w = int(round((xb[1] - xb[0]) / (xb[2] * ds)))
            grid_h = int(round((yb[1] - yb[0]) / (yb[2] * ds)))
            occ_head = QueryOccHead(
                in_channels=256,                   # ConvFuser output
                world_channels=occ_world_channels,
                mlp_hidden=occ_mlp_hidden,
                num_classes=occ_classes,
                pos_freqs=occ_pos_freqs,
                grid_size=(occ_num_z, grid_h, grid_w),
                train_max_queries=occ_train_max_queries,
            )
        else:
            from .occ_head import BEVOccHead
            occ_head = BEVOccHead(
                in_channels=256,
                num_classes=occ_classes,
                num_z=occ_num_z,
                hidden_channels=occ_hidden,
                lovasz_weight=occ_lovasz_weight,
            )

    depth_head = None
    if depth_head_enabled:
        from .depth_head import DepthHead
        depth_head = DepthHead(
            in_channels=cam_in_ch,
            hidden_channels=depth_hidden,
            d_bins=depth_bins,
            image_size=image_size,
            feature_size=feature_size,
            dbound=dbound,
        )

    cam_bbox_head_cfg = None
    if cam_head_enabled:
        from copy import deepcopy as _dc
        if not (isinstance(cfg.model, dict) and cfg.model.get("bbox_head")):
            raise ValueError("--cam-head requires cfg.model.bbox_head to clone from")
        cam_bbox_head_cfg = _dc(dict(cfg.model["bbox_head"]))

    # Auto-detect BEVDepth-style depth lifting in the view transform. If on, the
    # VT predicts+uses its own depth distribution, so we supervise THAT depthnet
    # (depth_from_vt) and do NOT build a redundant separate DepthHead.
    _vt = getattr(det_model, "view_transform", None)
    vt_depth_lift = bool(getattr(_vt, "depth_lift", False))
    if vt_depth_lift and depth_head is not None:
        print("[Model] view_transform has depth_lift=True → using its depthnet; "
              "ignoring separate --depth-head")
        depth_head = None

    if (occ_head is not None or depth_head is not None
            or cam_bbox_head_cfg is not None or vt_depth_lift):
        from .multitask_bev import MultiTaskBEVFusion
        model = MultiTaskBEVFusion(
            det_model=det_model,
            occ_head=occ_head,
            occ_loss_weight=occ_loss_weight,
            depth_head=depth_head,
            depth_loss_weight=depth_loss_weight,
            cam_bbox_head=cam_bbox_head_cfg,
            cam_loss_weight=cam_loss_weight,
            depth_from_vt=vt_depth_lift,
        )
        tags = []
        if occ_head is not None:
            head_kind = type(occ_head).__name__
            tags.append(f"occ-{head_kind} ({occ_classes} cls, {occ_num_z} Z)")
        if depth_head is not None:
            tags.append(f"depth ({depth_bins} bins, dbound={dbound})")
        if vt_depth_lift:
            tags.append(f"vt-depth-lift (D={_vt.D_bins}, w={depth_loss_weight})")
        if cam_bbox_head_cfg is not None:
            tags.append(f"cam-head (w={cam_loss_weight})")
        print(f"[Model] MultiTaskBEVFusion: det + " + " + ".join(tags))
    else:
        model = det_model
        print("[Model] Detection only (no auxiliary heads)")

    model.to(device)
    # data_preprocessor may be reached only via a Python property (not in _modules)
    # in some mmengine versions, so its mean/std buffers may not be moved by .to().
    # Explicitly move it here to guarantee mean/std land on the right device.
    dp = getattr(model, 'data_preprocessor', None)
    if dp is not None:
        dp.to(device)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train.py] device={device}  work_dir={args.work_dir}")

    # ---- Load config ----
    cfg = Config.fromfile(args.config)
    cfg.setdefault("default_scope", "mmdet3d")

    _patch_data_root(cfg, args.data_root)
    _patch_batch_size(cfg, args.batch_size)
    _patch_workers(cfg, args.workers)

    total_epochs = args.epochs if args.epochs > 0 else _get_max_epochs(cfg)
    base_lr_val = args.lr if args.lr > 0.0 else _get_base_lr(cfg)

    # ---- Eval-only mode ----
    if args.eval_only:
        if not args.load_from:
            raise ValueError("--eval-only requires --load-from <checkpoint>")
        from .eval_runner import RunnerEvaluator
        ev = RunnerEvaluator(
            config_path=args.config,
            data_root=args.data_root or cfg.get("data_root", ""),
            work_dir=args.work_dir,
        )
        ev.evaluate(ckpt_path=args.load_from)
        return

    # ---- Build train dataloader ----
    print("[train.py] Building train dataloader (via Runner)...")
    train_loader = build_train_loader(cfg)
    print(f"[train.py] train_dataloader: {len(train_loader)} iters/epoch, "
          f"batch_size={train_loader.batch_size}")

    # ---- Build model ----
    model = build_model(
        cfg=cfg,
        load_from=args.load_from,
        occ_classes=args.occ_classes,
        occ_num_z=args.occ_num_z,
        occ_hidden=args.occ_hidden,
        occ_lovasz_weight=args.occ_lovasz_weight,
        occ_loss_weight=args.occ_loss_weight,
        device=device,
        depth_head_enabled=args.depth_head,
        depth_bins=args.depth_bins,
        depth_hidden=args.depth_hidden,
        depth_loss_weight=args.depth_loss_weight,
        cam_head_enabled=args.cam_head,
        cam_loss_weight=args.cam_loss_weight,
        occ_head_type=args.occ_head_type,
        occ_world_channels=args.occ_world_channels,
        occ_mlp_hidden=args.occ_mlp_hidden,
        occ_pos_freqs=args.occ_pos_freqs,
        occ_train_max_queries=args.occ_train_max_queries,
    )

    # ---- LR multipliers from config (if optim_wrapper.paramwise_cfg present) ----
    lr_multipliers = {}
    ow = getattr(cfg, "optim_wrapper", {})
    if isinstance(ow, dict):
        pw = ow.get("paramwise_cfg", {})
        if isinstance(pw, dict):
            for key, kv in pw.get("custom_keys", {}).items():
                if isinstance(kv, dict) and "lr_mult" in kv:
                    lr_multipliers[key] = float(kv["lr_mult"])

    # ---- Build TrainConfig ----
    from .trainer import TrainConfig, Trainer
    tcfg = TrainConfig(
        work_dir=args.work_dir,
        lr=base_lr_val,
        weight_decay=0.01,
        betas=(0.9, 0.99),
        clip_grad_norm=args.clip_grad_norm,
        warmup_iters=args.warmup_iters,
        total_epochs=total_epochs,
        dtype=args.dtype,
        accum_steps=args.accum_steps,
        use_ema=args.ema,
        ema_decay=args.ema_decay,
        save_interval=args.save_interval,
        keep_last_n=args.keep_last_n,
        log_interval=args.log_interval,
        debug_iters=args.debug_iters,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        lr_multipliers=lr_multipliers,
        seed=args.seed,
    )

    # ---- Resolve resume path ----
    resume_path = ""
    if args.resume_from and os.path.isfile(args.resume_from):
        resume_path = args.resume_from
    elif args.resume:
        default = os.path.join(args.work_dir, "last_state.pth")
        if os.path.isfile(default):
            resume_path = default
        else:
            print(f"[Warn] --resume set but no last_state.pth found in {args.work_dir}")

    # ---- Optional internal-telemetry hooks ----
    if args.telemetry:
        from .telemetry import TelemetryRecorder, attach_basic_hooks, attach_attn_hooks
        telemetry = TelemetryRecorder(
            save_path=os.path.join(args.work_dir, "telemetry.jsonl"),
            log_every=args.telemetry_every,
        )
        attach_basic_hooks(model, telemetry)
        attach_attn_hooks(model, telemetry)
        print(f"[Telemetry] Enabled. Writing to {args.work_dir}/telemetry.jsonl every {args.telemetry_every} iters.")
    else:
        telemetry = None

    # ---- Run training ----
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        cfg=tcfg,
        device=device,
        resume_path=resume_path or None,
        telemetry=telemetry,
    )
    trainer.run()

    # ---- Post-training evaluation ----
    if args.eval_interval > 0 or args.eval_only:
        print("\n[train.py] Running final evaluation with Runner.test()...")
        from .eval_runner import RunnerEvaluator
        ev = RunnerEvaluator(
            config_path=args.config,
            data_root=args.data_root or "",
            work_dir=args.work_dir,
        )
        final_ckpt = os.path.join(args.work_dir, f"epoch_{total_epochs}.pth")
        if os.path.isfile(final_ckpt):
            ev.evaluate(ckpt_path=final_ckpt, tag=f"epoch_{total_epochs}_final")
        else:
            print(f"[Warn] Final checkpoint not found: {final_ckpt}")


if __name__ == "__main__":
    main()
