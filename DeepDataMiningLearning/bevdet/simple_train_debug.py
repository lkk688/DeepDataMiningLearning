#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from typing import Any, Dict, List
import torch
from torch.utils.data._utils.collate import default_collate as torch_default_collate


# ---------------------------------------------------------------------
# Minimal smart_det3d_collate (same one you just verified)
# ---------------------------------------------------------------------
def debug_smart_det3d_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate mmdet3d samples into:
        {"inputs": {"points": [...], "img": [...]}, "data_samples": [...]}
    so Det3DDataPreprocessor can handle them.
    """
    if not batch:
        raise RuntimeError("Empty batch in collate")

    if not isinstance(batch[0], dict):
        return torch_default_collate(batch)  # fallback

    merged_inputs: Dict[str, List[Any]] = {}
    data_samples: List[Any] = []

    for item in batch:
        ins = item.get("inputs", {})
        ds = item.get("data_samples", None)

        if isinstance(ins, dict):
            for k, v in ins.items():
                merged_inputs.setdefault(k, []).append(v)
        else:
            merged_inputs.setdefault("inputs", []).append(ins)

        if isinstance(ds, (list, tuple)):
            data_samples.extend(list(ds))
        elif ds is not None:
            data_samples.append(ds)

    return {"inputs": merged_inputs, "data_samples": data_samples}


# ---------------------------------------------------------------------
# Small helpers: inspect batch + hook fusion_layer
# ---------------------------------------------------------------------
def inspect_batch(batch: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("[DEBUG] Train batch structure")
    print("=" * 80)

    print(f"type(batch) = {type(batch)}")
    if not isinstance(batch, dict):
        print(batch)
        return

    print(f"batch keys = {list(batch.keys())}")

    # inputs
    inputs = batch.get("inputs", None)
    print("\n[inputs]")
    print(f"  type(inputs) = {type(inputs)}")
    if isinstance(inputs, dict):
        print(f"  keys = {list(inputs.keys())}")
        for k, v in inputs.items():
            print(f"    - {k}: type={type(v)}")
            if isinstance(v, list) and v:
                elem = v[0]
                if torch.is_tensor(elem):
                    print(f"        list len={len(v)}, elem.shape={tuple(elem.shape)}, device={elem.device}")
                else:
                    print(f"        list len={len(v)}, elem.type={type(elem)}")
            elif torch.is_tensor(v):
                print(f"        tensor shape={tuple(v.shape)}, device={v.device}")
    # data_samples
    ds = batch.get("data_samples", None)
    print("\n[data_samples]")
    print(f"  type(data_samples) = {type(ds)}")
    if isinstance(ds, list):
        print(f"  len(data_samples) = {len(ds)}")
        if ds:
            s0 = ds[0]
            print(f"    sample[0] type={type(s0)}")
            if hasattr(s0, "metainfo"):
                print(f"    sample[0].metainfo keys = {list(s0.metainfo.keys())[:10]} ...")


def hook_fusion_layer(model: torch.nn.Module) -> None:
    fusion = getattr(model, "fusion_layer", None)
    if fusion is None or not hasattr(fusion, "forward"):
        print("[DEBUG] fusion_layer not found; skipping fusion hook.")
        return

    orig_forward = fusion.forward

    def wrapped_forward(inputs):
        print("\n[DEBUG] fusion_layer.forward called")
        print(f"  type(inputs) = {type(inputs)}")
        if isinstance(inputs, (list, tuple)):
            print(f"  len(inputs) = {len(inputs)}")
            for i, x in enumerate(inputs):
                if torch.is_tensor(x):
                    print(f"    inputs[{i}] shape={tuple(x.shape)}, device={x.device}")
                else:
                    print(f"    inputs[{i}] type={type(x)}")
        elif torch.is_tensor(inputs):
            print(f"  inputs.shape = {tuple(inputs.shape)}, device={inputs.device}")

        out = orig_forward(inputs)

        if torch.is_tensor(out):
            print(f"  fusion_layer output shape={tuple(out.shape)}, device={out.device}")
        else:
            print(f"  fusion_layer output type={type(out)}")
        print("[DEBUG] fusion_layer.forward end\n")
        return out

    fusion.forward = wrapped_forward  # type: ignore
    print("[DEBUG] fusion_layer.forward wrapped for shape logging.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Debug BEVFusion training pipeline")
    parser.add_argument("--config", default="/data/rnd-liu/MyRepo/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py", help="Path to mmdet3d config (.py)")
    parser.add_argument("--checkpoint", default="/data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth", help="Model checkpoint (.pth) for BEVFusion")
    parser.add_argument("--data-root", default="/data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes",
                        help="Override dataset data_root (e.g., /data/.../mmdetection3d/data/nuscenes)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for the debug loader")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device = {device}")

    import mmdet3d_datasets as ds_mod
    import mmdet3d_models as models_mod

    # Monkey-patch collate to our verified version
    print("[DEBUG] Monkey-patching smart_det3d_collate...")
    ds_mod.smart_det3d_collate = debug_smart_det3d_collate  # type: ignore

    # Build model wrapper
    print("[DEBUG] Building model via build_model_wrapper...")
    extra_model: Dict[str, Any] = {}
    model_wrapper = models_mod.build_model_wrapper(
        backend="mmdet3d",
        model_config=args.config,
        checkpoint=args.checkpoint,
        extra=extra_model,
    )
    model = model_wrapper.unwrap().to(device)  # type: ignore
    model.eval()
    print(f"[DEBUG] Model type = {type(model)}")

    hook_fusion_layer(model)

    # Build dataloaders
    print("[DEBUG] Building dataloaders via build_dataloaders...")
    extra_data: Dict[str, Any] = {}
    if args.data_root:
        extra_data["data_root"] = args.data_root
    train_loader, val_loader = ds_mod.build_dataloaders(
        backend="mmdet3d",
        data_config=args.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        extra=extra_data,
    )
    print(f"[DEBUG] train_loader.batch_size = {train_loader.batch_size}")
    if val_loader is not None:
        print(f"[DEBUG] val_loader.batch_size = {val_loader.batch_size}")

    # One batch
    print("\n[DEBUG] Fetching one batch from train_loader...")
    batch = next(iter(train_loader))
    inspect_batch(batch)

    # Single forward_loss (uses model._run_forward internally now)
    print("\n" + "=" * 80)
    print("[DEBUG] Running a single forward_loss(...)")
    print("=" * 80)

    with torch.no_grad():
        total_loss, logs = model_wrapper.forward_loss(batch, amp=False, scaler=None)

    print("\n[DEBUG] Loss output:")
    print(f"  total_loss = {float(total_loss.detach().cpu()):.6f}")
    print("  logs keys:", list(logs.keys()))
    for k in sorted(logs.keys()):
        if isinstance(logs[k], (int, float)):
            print(f"    {k}: {logs[k]:.6f}")
        else:
            print(f"    {k}: {logs[k]}")

    print("\n[DEBUG] Done.")


if __name__ == "__main__":
    main()
    
  