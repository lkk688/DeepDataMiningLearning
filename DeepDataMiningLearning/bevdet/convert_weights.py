#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch

def safe_load_checkpoint(path: Path):
    """
    Try to load a checkpoint under PyTorch 2.6 'safe' unpickler.
    If that fails, fall back to weights_only=False (UNSAFE; trust the source).
    """
    # 1) Try safe_globals allowlist (PyTorch 2.6+)
    try:
        from torch.serialization import safe_globals  # type: ignore
        # Common mmengine classes that appear in runners/checkpoints
        allow = []
        try:
            from mmengine.logging.history_buffer import HistoryBuffer  # type: ignore
            allow.append(HistoryBuffer)
        except Exception:
            pass
        try:
            from mmengine.logging.message_hub import MessageHub  # type: ignore
            allow.append(MessageHub)
        except Exception:
            pass

        if allow:
            print(f"[INFO] Using safe_globals allowlist: {[c.__name__ for c in allow]}")
        else:
            print("[INFO] No mmengine classes found for allowlist; attempting safe load anyway.")

        with safe_globals(allow):
            return torch.load(path, map_location="cpu")  # default weights_only=True on PyTorch 2.6
    except Exception as e:
        print(f"[WARN] Safe load failed: {repr(e)}")
        print("[WARN] Falling back to torch.load(weights_only=False). ONLY do this if you trust the checkpoint source.")
        # 2) Unsafe fallback (can execute arbitrary code if checkpoint is malicious)
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(ckpt: dict, prefer_ema: bool = False):
    """
    Extract a state_dict from various checkpoint formats.
    Preference order:
    - if prefer_ema and 'ema_state_dict' exists -> use it
    - 'state_dict'
    - 'model'
    - raw dict (assume it's already a state_dict)
    """
    # Some training pipelines save both model and ema under different keys
    if prefer_ema and isinstance(ckpt, dict):
        for k in ["ema_state_dict", "state_dict_ema", "model_ema"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                print(f"[INFO] Using EMA weights from key: '{k}'")
                return ckpt[k]

    if isinstance(ckpt, dict):
        for k in ["state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                print(f"[INFO] Using weights from key: '{k}'")
                return ckpt[k]

    # If it looks like a plain state_dict already
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        print("[INFO] Input looks like a plain state_dict.")
        return ckpt

    # Otherwise, try common nesting
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                print(f"[INFO] Heuristically using nested dict under key: '{k}' as state_dict.")
                return v

    raise ValueError("Could not find a state_dict in the checkpoint.")


def validate_weights_only(path: Path):
    """
    Ensure the converted file can be loaded with the default (weights_only=True) loader.
    """
    try:
        obj = torch.load(path, map_location="cpu")  # PyTorch 2.6 default: weights_only=True
        if not (isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict)):
            raise RuntimeError("Converted file does not contain {'state_dict': ...}.")
        n_params = sum(p.numel() for p in obj["state_dict"].values() if isinstance(p, torch.Tensor))
        print(f"[OK] Validation passed. state_dict params: {n_params}")
        return True
    except Exception as e:
        print(f"[ERROR] Validation failed: {repr(e)}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Convert a checkpoint to weights-only format with backup.")
    ap.add_argument("--ckpt", default="/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/mybevfusion12v2/epoch_8.pth", type=str, help="Path to checkpoint (e.g., epoch_1.pth)")
    ap.add_argument("--prefer-ema", action="store_true", help="Prefer EMA weights if present.")
    ap.add_argument("--backup-suffix", type=str, default=None,
                    help="Backup suffix (default: .backup-YYYYmmdd-HHMMSS)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        print(f"[FATAL] File not found: {ckpt_path}")
        sys.exit(1)

    # 1) Make a backup
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup_suffix = args.backup_suffix or f".backup-{ts}"
    backup_path = ckpt_path.with_name(ckpt_path.name + backup_suffix)
    print(f"[INFO] Backing up '{ckpt_path.name}' -> '{backup_path.name}'")
    shutil.copy2(ckpt_path, backup_path)

    # 2) Load original (safe first, then unsafe fallback if needed)
    print(f"[INFO] Loading original checkpoint: {ckpt_path}")
    ckpt = safe_load_checkpoint(ckpt_path)

    # 3) Extract state_dict
    state = extract_state_dict(ckpt, prefer_ema=args.prefer_ema)

    # 4) Overwrite original file with weights-only
    print(f"[INFO] Writing weights-only to original path: {ckpt_path}")
    try:
        torch.save({"state_dict": state}, ckpt_path)
    except Exception as e:
        print(f"[FATAL] Failed to save weights-only file: {repr(e)}")
        print("[INFO] Restoring from backup...")
        shutil.copy2(backup_path, ckpt_path)
        sys.exit(2)

    # 5) Validate loadability under PyTorch 2.6 default
    if not validate_weights_only(ckpt_path):
        print("[INFO] Restoring from backup due to failed validation...")
        shutil.copy2(backup_path, ckpt_path)
        sys.exit(3)

    print(f"[DONE] Converted to weights-only and kept backup at: {backup_path}")

if __name__ == "__main__":
    main()