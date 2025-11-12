#!/usr/bin/env python3
"""
inspect_bevfusion_ckpt.py

Prints major I/O shapes and key parameters of a BEVFusion-style model.
- Figures out Depth LSS vt_channels (from module attr or checkpoint conv shapes)
- Reports which FPN level the VT consumes (stride heuristic)
- Confirms head input channels (256 vs 512)
- Shows LiDAR neck (SECFPN) channel mapping from ckpt (if available)

Usage:
  python inspect_bevfusion_ckpt.py --config <cfg.py> [--weights <model.pth>]
"""
import argparse
from pprint import pprint

import torch
import torch.nn as nn

from mmengine import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS


def _find_in_pipeline(cfg, key):
    """Find first dict in any pipeline list containing `key`."""
    candidates = []

    def scan(node):
        if isinstance(node, dict):
            if key in node:
                candidates.append(node)
            for v in node.values():
                scan(v)
        elif isinstance(node, (list, tuple)):
            for x in node:
                scan(x)

    # try top-level, loaders, datasets
    for k in ("train_pipeline", "test_pipeline", "val_pipeline",
              "train_dataloader", "val_dataloader", "test_dataloader"):
        if k in cfg:
            scan(cfg[k])
    return candidates[0] if candidates else None


def _safe_getattr(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _vt_channels_from_module(vt):
    # Best-case: module has explicit `out_channels`
    ch = _safe_getattr(vt, "out_channels", None)
    if isinstance(ch, int):
        return ch

    # Heuristic: scan Conv2d layers in VT; pick the most common output channels
    counts = {}
    for n, m in vt.named_modules():
        if isinstance(m, nn.Conv2d) and hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
            o = m.weight.shape[0]
            counts[o] = counts.get(o, 0) + 1
    if counts:
        # pick the channel number that appears most often
        return max(counts.items(), key=lambda kv: kv[1])[0]
    return None


def _vt_channels_from_state(state_dict, vt_prefix="view_transform"):
    # Last resort: scan state_dict conv weights under VT prefix
    counts = {}
    for k, v in state_dict.items():
        if not k.startswith(vt_prefix):
            continue
        if k.endswith("weight") and v.dim() == 4:
            o = v.shape[0]
            counts[o] = counts.get(o, 0) + 1
    if counts:
        return max(counts.items(), key=lambda kv: kv[1])[0]
    return None


def _first_conv_in(module, name_contains=None):
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            if name_contains is None or name_contains in n:
                return n, m
    return None, None

# --- add this helper near the top of your script ---
import torch

def load_state_dict_robust(path: str):
    """
    Loads a checkpoint state_dict on PyTorch 2.6+ where weights_only=True is default.
    1) Try safe allowlist for mmengine HistoryBuffer (weights_only=True).
    2) Fallback to weights_only=False (trusted file).
    Returns a plain state_dict (dict[str, Tensor]).
    """
    # Try the safe allowlist approach
    try:
        from torch.serialization import safe_globals
        from mmengine.logging.history_buffer import HistoryBuffer  # the blocked type
        try:
            with safe_globals([HistoryBuffer]):
                raw = torch.load(path, map_location='cpu')  # default weights_only=True
        except TypeError:
            # Older PyTorch without safe_globals; just try normal load first
            raw = torch.load(path, map_location='cpu')
        # Normalize to a state_dict
        if isinstance(raw, dict) and 'state_dict' in raw:
            return raw['state_dict']
        return raw
    except Exception as e:
        print(f"[load_state_dict_robust] allowlist path failed: {e}\n"
              f"-> falling back to weights_only=False (trusted checkpoint).")
        raw = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(raw, dict) and 'state_dict' in raw:
            return raw['state_dict']
        return raw
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
    ap.add_argument("--weights", default="modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.fixed_spconvv2.pth", help="(optional) checkpoint to load")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    model = MODELS.build(cfg.model)
    model.eval()
    model.cpu()
    ckpt_state = None
    if args.weights:
            print(f"\nLoading checkpoint (robust): {args.weights}")
            # Build the model from cfg as you already do...
            # model = MODELS.build(cfg.model)
            # model.eval().cpu()
            # Instead of load_checkpoint(...), use our robust loader:
            state_dict = load_state_dict_robust(args.weights)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded into model. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            ckpt_state = state_dict  # keep for vt_channels/head inference

    print("\n=== IMAGE (backbone/FPN → VT) ===")
    img_backbone = _safe_getattr(model, "img_backbone")
    img_neck = _safe_getattr(model, "img_neck")
    vt = _safe_getattr(model, "view_transform")

    if img_backbone is not None:
        print(f"img_backbone: {img_backbone.__class__.__name__}")
        print(f"  out_indices: {_safe_getattr(img_backbone, 'out_indices', None)}")
    if img_neck is not None:
        print(f"img_neck: {img_neck.__class__.__name__}")
        print(f"  out_channels: {_safe_getattr(img_neck, 'out_channels', None)}")
        print(f"  num_outs: {_safe_getattr(img_neck, 'num_outs', None)}")
    if vt is not None:
        print(f"view_transform: {vt.__class__.__name__}")
        vt_ch_attr = _safe_getattr(vt, "out_channels", None)
        vt_ch_guess = _vt_channels_from_module(vt)
        if ckpt_state is not None and vt_ch_attr is None:
            vt_ch_ckpt = _vt_channels_from_state(ckpt_state, vt_prefix="view_transform")
        else:
            vt_ch_ckpt = None
        print(f"  VT out_channels (attr): {vt_ch_attr}")
        print(f"  VT out_channels (scan module): {vt_ch_guess}")
        print(f"  VT out_channels (scan ckpt): {vt_ch_ckpt}")
        vt_channels = vt_ch_attr or vt_ch_guess or vt_ch_ckpt
        print(f"→ vt_channels (final): {vt_channels}")

        feat_size = _safe_getattr(vt, "feature_size", None)
        # Try to recover final_dim from pipeline
        final_dim = None
        p = _find_in_pipeline(cfg, "final_dim")
        if p:
            final_dim = p.get("final_dim", None)
        print(f"  VT feature_size: {feat_size}")
        print(f"  Image final_dim (pipeline): {final_dim}")
        if feat_size and final_dim:
            try:
                stride_h = int(round(final_dim[0] / feat_size[0]))
                stride_w = int(round(final_dim[1] / feat_size[1]))
                print(f"  → Deduced VT consumes FPN level with stride ~{stride_h}x{stride_w}")
            except Exception:
                pass

    fusion = _safe_getattr(model, "fusion_layer")
    if fusion is not None:
        print(f"fusion_layer: {fusion.__class__.__name__}")
        print(f"  in_channels: {_safe_getattr(fusion, 'in_channels', None)}")
        print(f"  out_channels: {_safe_getattr(fusion, 'out_channels', None)}")

    print("\n=== LiDAR (SECOND → Neck) ===")
    pts_backbone = _safe_getattr(model, "pts_backbone")
    pts_neck = _safe_getattr(model, "pts_neck")
    if pts_backbone is not None:
        print(f"pts_backbone: {pts_backbone.__class__.__name__}")
        print(f"  out_channels: {_safe_getattr(pts_backbone, 'out_channels', None)}")
    if pts_neck is not None:
        print(f"pts_neck: {pts_neck.__class__.__name__}")
        print(f"  in_channels: {_safe_getattr(pts_neck, 'in_channels', None)}")
        print(f"  out_channels: {_safe_getattr(pts_neck, 'out_channels', None)}")
        print(f"  num_outs: {_safe_getattr(pts_neck, 'num_outs', None)}")
        print(f"  upsample_strides: {_safe_getattr(pts_neck, 'upsample_strides', None)}")

    # Peek a couple of deblock weights (SECFPN) or align-conv (custom neck) from ckpt
    if ckpt_state is not None:
        print("\n[Checkpoint sanity: a few LiDAR neck weights]")
        shown = 0
        for key in ckpt_state.keys():
            if key.startswith("pts_neck.") and key.endswith(".weight"):
                w = ckpt_state[key]
                if isinstance(w, torch.Tensor):
                    print(f"  {key} {tuple(w.shape)}")
                    shown += 1
                    if shown >= 4:
                        break
        if shown == 0:
            print("  (no pts_neck.*.weight tensors found)")

    print("\n=== Detection Head ===")
    head = _safe_getattr(model, "bbox_head")
    if head is not None:
        print(f"bbox_head: {head.__class__.__name__}")
        in_ch = _safe_getattr(head, "in_channels", None)
        print(f"  in_channels (attr): {in_ch}")
        # Also infer from first conv weight in the head
        conv_name, conv_mod = _first_conv_in(head, "shared_conv")
        if conv_mod is None:
            conv_name, conv_mod = _first_conv_in(head, None)
        if conv_mod is not None and hasattr(conv_mod, "weight"):
            w = conv_mod.weight
            cin = w.shape[1]
            cout = w.shape[0]
            print(f"  first conv: {conv_name} weight {tuple(w.shape)} → inferred in_channels={cin}, out_channels={cout}")
        else:
            print("  (could not find a Conv2d in bbox_head to infer channels)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()