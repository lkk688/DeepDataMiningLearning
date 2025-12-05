# -*- coding: utf-8 -*-
"""
Model backend wrappers used by the training / evaluation scripts.

Supported backends
==================
- mmdet3d
    Uses MMEngine / MMDet3D registries to build the model.
    Loss is computed via model(..., mode="loss").
    Prediction is via model(..., mode="predict").

- openpcdet
    Uses OpenPCDet's model construction and training APIs.
    Assumes the model exposes a callable that returns loss dicts
    (e.g., get_training_loss or forward(batch) -> loss dict).

- custom
    Uses a user-defined nn.Module imported dynamically.
    The model must return either:
        * a scalar loss tensor, or
        * a dict whose keys containing 'loss' are summed as total loss.

This file deliberately isolates 3rd-party dependencies so the training
script only depends on these wrappers and does not need to import
MMDet3D / OpenPCDet directly.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from mmengine.runner.checkpoint import load_checkpoint
from pickle import UnpicklingError
import inspect

# =============================================================================
# Common helper utilities
# =============================================================================

class SaveLoadState:
    """Small interface to unify saving / loading of model state."""

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict: bool = True):
        raise NotImplementedError


def _move_tensors_to_device(obj, device):
    """Recursively move all tensors inside obj to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_move_tensors_to_device(v, device) for v in obj]
        return type(obj)(out) if not isinstance(obj, list) else out
    return obj


def _tensorize_to_device(x, device, dtype=None):
    """Convert x to a tensor on device, optional dtype cast."""
    t = torch.as_tensor(x, device=device)
    if dtype is not None:
        t = t.to(dtype)
    return t


def _normalize_data_samples_on_device(samples, device):
    """
    Ensure labels / boxes inside Det3DDataSample are on the same device
    as the model and that labels are long type.

    This is a safety layer on top of Det3DDataSample.to(), which already
    moves many tensors, but we explicitly:
      - enforce labels_{3d} to be long
      - enforce bboxes_{3d} to be on device
    """
    if not isinstance(samples, (list, tuple)):
        it = [samples]
    else:
        it = samples

    for s in it:
        if not hasattr(s, "to"):
            continue

        # First, move the whole sample (most nested tensors) to device.
        s.to(device)

        # Then, enforce type / device for labels and boxes.
        for gi_name in ("gt_instances_3d", "gt_instances"):
            gi = getattr(s, gi_name, None)
            if gi is None:
                continue

            # Labels (2D or 3D)
            for lab_name in ("labels_3d", "labels"):
                if hasattr(gi, lab_name):
                    lab = getattr(gi, lab_name)
                    if isinstance(lab, torch.Tensor):
                        if lab.device != device or lab.dtype != torch.long:
                            setattr(
                                gi,
                                lab_name,
                                lab.to(device=device, dtype=torch.long, non_blocking=True),
                            )
                    else:
                        setattr(
                            gi,
                            lab_name,
                            _tensorize_to_device(lab, device, torch.long),
                        )

            # Bounding boxes (2D or 3D)
            for bb_name in ("bboxes_3d", "bboxes"):
                if hasattr(gi, bb_name):
                    bb = getattr(gi, bb_name)
                    if hasattr(bb, "to"):
                        try:
                            setattr(gi, bb_name, bb.to(device))
                        except Exception:
                            # Some box types might not support .to; ignore.
                            pass


def _move_data_samples_to_device(samples, device):
    """
    Move Det3DDataSample objects to device using .to(device).
    This is lighter than _normalize_data_samples_on_device, and meant
    for prediction mode (no labels / gt required).
    """
    if isinstance(samples, (list, tuple)):
        for s in samples:
            if hasattr(s, "to"):
                s.to(device)
    else:
        if hasattr(samples, "to"):
            samples.to(device)


def _first_attr_no_bool(obj, names):
    """
    Safely get the first existing, non-None attribute from `names` list
    without triggering boolean conversions (some tensors throw on bool()).
    """
    for n in names:
        if hasattr(obj, n):
            val = getattr(obj, n)
            if val is not None:
                return val
    return None


# =============================================================================
# Base wrapper
# =============================================================================

class BaseModelWrapper(nn.Module, SaveLoadState):
    """
    Base abstraction for all backends (MMDet3D, OpenPCDet, custom).

    Subclasses MUST implement:
        - unwrap()
        - forward_loss(batch, amp, scaler=None)

    Optional methods:
        - move_batch_to_device(batch, device)
        - predict_step(...)
    """

    def __init__(self):
        super().__init__()

    # ---- public API ----

    def unwrap(self) -> nn.Module:
        """Return the underlying nn.Module (for optimizer, debugging, etc.)."""
        return self

    # def move_batch_to_device(self, batch, device):
    #     """
    #     Default implementation: recursively move tensors in batch to device.
    #     Subclasses can override to handle special container types.
    #     """
    #     def to_dev(x):
    #         if isinstance(x, torch.Tensor):
    #             return x.to(device, non_blocking=True)
    #         if isinstance(x, list):
    #             return [to_dev(xx) for xx in x]
    #         if isinstance(x, dict):
    #             return {k: to_dev(v) for k, v in x.items()}
    #         return x

    #     return to_dev(batch)
    
    def move_batch_to_device(self, batch, device):
        """For MMDet3D we let model.data_preprocessor handle device moves.

        The DataLoader outputs dicts of the form:
          {"inputs": {...}, "data_samples": [...]}

        Passing this directly into model._run_forward(..., mode="loss")
        triggers the standard MMDet3D pipeline (including data_preprocessor),
        which we already verified to work in simple_train_debug.py.
        """
        return batch

    # def forward_loss(self, batch, amp: bool, scaler=None) -> Tuple[torch.Tensor, Dict[str, float]]:
    #     """
    #     Compute training loss given a batch.

    #     Returns:
    #         total_loss: torch.Tensor
    #         logs:       dict with scalar loss components
    #     """
    #     raise NotImplementedError
    
    def forward_loss(self, batch, amp: bool, scaler=None):
        """
        Compute loss for MMDet3D models using the *standard* forward path:

        1) Normalize batch into a single dict:
               {"inputs": ..., "data_samples": [...]}
        2) Call model._run_forward(data, mode="loss"), which internally:
               - applies model.data_preprocessor(...)
               - calls model.loss(...)
        3) Reduce the returned loss dict into (total_loss, logs).

        This matches the behavior we validated in simple_train_debug.py and
        avoids manual device juggling / custom data_preprocessor calls.
        """

        def _merge_list_batch(list_batch):
            """Merge list[dict] batches into a single dict for MMDet3D.

            Each item is expected to look like:
              {"inputs": {...}, "data_samples": Det3DDataSample or list[Det3DDataSample]}
            """
            merged_inputs = {}
            merged_samples = []
            for item in list_batch:
                if not isinstance(item, dict):
                    raise RuntimeError(
                        f"Each batch item must be a dict, got {type(item)}"
                    )

                ins = item.get("inputs", None)
                samp = item.get("data_samples", None)
                if ins is None or samp is None:
                    raise RuntimeError(
                        "Each batch item must contain 'inputs' and 'data_samples'."
                    )

                # inputs is usually a dict like:
                #   {'points': list[Tensor], 'img': list[Tensor], ...}
                if isinstance(ins, dict):
                    for k, v in ins.items():
                        merged_inputs.setdefault(k, []).append(v)
                else:
                    # Rare fallback: if inputs is tensor/list, we put under "inputs"
                    merged_inputs.setdefault("inputs", []).append(ins)

                # data_samples → flatten into a single list
                if isinstance(samp, (list, tuple)):
                    merged_samples.extend(list(samp))
                else:
                    merged_samples.append(samp)

            return merged_inputs, merged_samples

        # 1) Normalize inputs / data_samples into a single dict
        if isinstance(batch, dict):
            inputs = batch.get("inputs", None)
            data_samples = batch.get("data_samples", None)

            # Some collates may produce list[dict] under "inputs", handle that.
            if isinstance(inputs, list) and inputs and isinstance(inputs[0], dict):
                inputs, data_samples = _merge_list_batch(
                    [{"inputs": i, "data_samples": s}
                     for i, s in zip(inputs, data_samples)]
                )
        elif isinstance(batch, (list, tuple)):
            # pseudo_collate / smart_collate often produce list[dict]
            inputs, data_samples = _merge_list_batch(batch)
        else:
            raise RuntimeError(f"Unsupported batch type: {type(batch)}")

        if inputs is None or data_samples is None:
            raise RuntimeError("Missing 'inputs' or 'data_samples' after collation/merge.")

        data = {"inputs": inputs, "data_samples": data_samples}

        # 2) Standard MMDet3D path: data_preprocessor + loss, via _run_forward
        if amp:
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                losses = self.model._run_forward(data, mode="loss")
        else:
            losses = self.model._run_forward(data, mode="loss")

        # 3) Reduce loss dict → (total_loss, logs)
        total, logs = self._scalarize_losses(losses)
        logs["loss"] = float(total.detach().cpu())
        return total, logs

    # ---- SaveLoadState passthrough ----

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, strict=strict)


# =============================================================================
# MMDetection3D wrapper
# =============================================================================
class MMDet3DWrapper(BaseModelWrapper):
    """
    Wrapper around an MMDet3D model built from a config.

    Key points:
    - We do NOT call model.data_preprocessor() here.
      The DataLoader pipeline + our manual device moves are enough.
    - Expects batches of the form:
        {
            "inputs": {
                "points": list[Tensor],  # len = batch_size
                "img":    list[Tensor],  # len = batch_size
            },
            "data_samples": list[Det3DDataSample],
        }
    - Handles AMP outside, via the `amp` flag.
    """

    def __init__(
        self,
        model_config: str,
        checkpoint: str = None,
        extra: Dict[str, Any] = None,
    ):
        super().__init__()
        extra = extra or {}

        # Lazy imports so train.py doesn't hard-depend on mmdet3d
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmdet3d.registry import MODELS
        from mmengine.runner.checkpoint import load_checkpoint

        cfg = Config.fromfile(model_config)
        if isinstance(extra.get("cfg_options"), dict):
            cfg.merge_from_dict(extra["cfg_options"])

        # Default scope
        default_scope = cfg.get("default_scope", "mmdet3d")
        init_default_scope(default_scope)

        # Build the MMDet3D model (no weights yet)
        self.model = MODELS.build(cfg.model)
        self.cfg = cfg

        # Robust checkpoint loading (handles PyTorch 2.6 weights_only change)
        if checkpoint:
            try:
                print(f"Loads checkpoint by local backend from path: {checkpoint}")
                load_checkpoint(
                    self.model,
                    checkpoint,
                    map_location="cpu",
                    strict=False,
                )
            except Exception as e:
                # Typical path when hitting PyTorch 2.6 "weights_only" behavior
                print(
                    "[WARN] mmengine.load_checkpoint failed, "
                    "falling back to manual torch.load:\n"
                    f"        {repr(e)}"
                )
                try:
                    # Try explicit weights_only=False if available
                    state = torch.load(
                        checkpoint,
                        map_location="cpu",
                        weights_only=False,  # PyTorch >= 2.6
                    )
                except TypeError:
                    # For older torch: no weights_only arg
                    state = torch.load(checkpoint, map_location="cpu")

                # Many OpenMMLab checkpoints store weights under "state_dict"
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]

                missing, unexpected = self.model.load_state_dict(
                    state, strict=False
                )
                print(
                    "[INFO] Manual load done. "
                    f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
                )

        # One-time debug flag for device/dtype checks
        self._dev_checked = False

    def unwrap(self) -> nn.Module:
        """Return the underlying MMDet3D model."""
        return self.model

    def move_batch_to_device(self, batch, device):
        """
        Move common container types to the specified device.

        - Tensors in batch["inputs"] are moved via _move_tensors_to_device.
        - Det3DDataSample objects in batch["data_samples"] are moved
          later in forward_loss via _normalize_data_samples_on_device.
        """
        if not isinstance(batch, dict):
            return batch

        out = {}
        for k, v in batch.items():
            if k == "inputs":
                out[k] = _move_tensors_to_device(v, device)
            else:
                # data_samples (and other metadata) stay as-is here;
                # Det3DDataSample.to(device) is called in forward_loss.
                out[k] = v
        return out

    @staticmethod
    def _scalarize_losses(losses: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Sum all entries whose key contains 'loss' (case-insensitive),
        and build a flat logs dict (floats).
        """
        total = None
        logs: Dict[str, float] = {}

        for k, v in losses.items():
            key = str(k)
            if isinstance(v, torch.Tensor):
                val = v.mean()
                logs[f"losses/{key}"] = float(val.detach().cpu())
                if "loss" in key.lower():
                    total = val if total is None else total + val
            elif isinstance(v, (int, float)):
                logs[f"losses/{key}"] = float(v)
            elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
                s = torch.stack([t.mean() for t in v]).sum()
                logs[f"losses/{key}"] = float(s.detach().cpu())
                if "loss" in key.lower():
                    total = s if total is None else total + s

        if total is None:
            total = torch.tensor(0.0, device=next(self.parameters()).device)

        logs["loss"] = float(total.detach().cpu())
        return total, logs

    def forward_loss(self, batch, amp: bool, scaler=None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for a batch.

        Expected batch format (from mmdet3d_datasets.build_dataloaders):

            batch = {
                "inputs": {
                    "points": list[Tensor],
                    "img":    list[Tensor],
                },
                "data_samples": list[Det3DDataSample],
            }

        We:
          1) Move all tensors in 'inputs' to model.device.
          2) Move / normalize labels & boxes inside data_samples to model.device.
          3) Call self.model(proc_inputs, proc_samples, mode="loss") directly.
        """
        if not isinstance(batch, dict):
            raise RuntimeError(f"MMDet3DWrapper.forward_loss expects a dict batch, got {type(batch)}")

        inputs = batch.get("inputs", None)
        data_samples = batch.get("data_samples", None)

        if inputs is None or data_samples is None:
            raise RuntimeError(
                "Batch must contain 'inputs' and 'data_samples' keys. "
                f"Got keys: {list(batch.keys())}"
            )

        device = next(self.model.parameters()).device

        # 1) Move inputs (points/img) tensors to device
        proc_inputs = _move_tensors_to_device(inputs, device)

        # 2) Move / normalize Det3DDataSample to device
        _normalize_data_samples_on_device(data_samples, device)

        # One-time debug print to verify device/dtype alignment
        if not self._dev_checked:
            gi = _first_attr_no_bool(data_samples[0], ["gt_instances_3d", "gt_instances"])
            lbl = _first_attr_no_bool(gi, ["labels_3d", "labels"]) if gi is not None else None

            if lbl is not None and hasattr(lbl, "device"):
                print("[DBG] labels device/dtype:", lbl.device, getattr(lbl, "dtype", None))
            else:
                print("[DBG] labels missing or has no device attr:", type(lbl))

            vox = proc_inputs.get("voxels", None) if isinstance(proc_inputs, dict) else None
            if isinstance(vox, dict):
                any_vox = next((v for v in vox.values() if hasattr(v, "device")), None)
                print("[DBG] vox device:", getattr(any_vox, "device", None))

            self._dev_checked = True

        # 3) Forward pass (no call to data_preprocessor here)
        if amp:
            from torch.amp import autocast
            with autocast(dtype=torch.float16):
                losses = self.model(proc_inputs, data_samples, mode="loss")
        else:
            losses = self.model(proc_inputs, data_samples, mode="loss")

        # 4) Scalarize loss dict
        total, logs = self._scalarize_losses(losses)
        return total, logs

    @torch.no_grad()
    def predict_step(self, batch, amp: bool = False):
        """
        Minimal predict interface that mirrors forward_loss:

            preds = wrapper.predict_step(batch)

        Here we:
          - Move tensors in batch["inputs"] to device,
          - Move Det3DDataSample objects to device,
          - Call self.model(proc_inputs, proc_samples, mode="predict").

        This is enough for debugging / simple inference.
        """
        if not isinstance(batch, dict):
            raise RuntimeError(f"MMDet3DWrapper.predict_step expects a dict batch, got {type(batch)}")

        inputs = batch.get("inputs", None)
        data_samples = batch.get("data_samples", None)
        if inputs is None or data_samples is None:
            raise RuntimeError(
                "Batch must contain 'inputs' and 'data_samples' keys. "
                f"Got keys: {list(batch.keys())}"
            )

        device = next(self.model.parameters()).device
        proc_inputs = _move_tensors_to_device(inputs, device)
        _normalize_data_samples_on_device(data_samples, device)

        if amp:
            from torch.amp import autocast
            with autocast(dtype=torch.float16):
                preds = self.model(proc_inputs, data_samples, mode="predict")
        else:
            preds = self.model(proc_inputs, data_samples, mode="predict")

        return preds
        
class MMDet3DWrapper_old(BaseModelWrapper):
    """
    Wrapper around an MMDet3D model.

    Key responsibilities:
      - Build the model from a config file via MODELS.build().
      - (Optionally) load a checkpoint.
      - Normalize various batch formats into the shape MMDet3D expects:
            dict{"inputs": ..., "data_samples": ...}
        or
            list[dict{"inputs":..., "data_samples":...}]
      - Call data_preprocessor for both training and prediction.
      - Ensure labels / boxes are on the correct device and dtype.
      - Sum loss terms into a single scalar for optimization.
    """

    def __init__(self, model_config: str, checkpoint: str = None, extra: Dict[str, Any] = None):
        super().__init__()
        extra = extra or {}

        # Lazy imports so that the training script does not depend on MMDet3D directly.
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmdet3d.registry import MODELS
        from mmengine.runner.checkpoint import load_checkpoint

        cfg = Config.fromfile(model_config)

        # Optional config overrides (e.g., cfg_options from CLI)
        if "cfg_options" in extra and isinstance(extra["cfg_options"], dict):
            cfg.merge_from_dict(extra["cfg_options"])

        # Default scope is usually 'mmdet3d'
        default_scope = cfg.get("default_scope", "mmdet3d")
        init_default_scope(default_scope)

        # Build model without weights first
        self.model = MODELS.build(cfg.model)

        # Optional weights
        if checkpoint:
            try:
                # Normal path: let mmengine handle state_dict extraction
                load_checkpoint(self.model, checkpoint, map_location="cpu", strict=False)
            except Exception as e:
                print("[WARN] mmengine.load_checkpoint failed, falling back to manual torch.load:")
                print("       ", repr(e))

                # Fallback: load raw checkpoint with weights_only=False (PyTorch 2.6+)
                load_kwargs = {"map_location": "cpu"}
                sig = inspect.signature(torch.load)
                if "weights_only" in sig.parameters:
                    load_kwargs["weights_only"] = False  # important for PyTorch 2.6+

                raw = torch.load(checkpoint, **load_kwargs)

                # Common mmengine checkpoint formats: 'state_dict', 'model', or raw dict
                if isinstance(raw, dict):
                    if "state_dict" in raw:
                        state_dict = raw["state_dict"]
                    elif "model" in raw:
                        state_dict = raw["model"]
                    else:
                        # Interpret the dict itself as a state_dict
                        state_dict = raw
                else:
                    # Last-resort: treat as state_dict directly
                    state_dict = raw

                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                print(f"[INFO] Manual load done. missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")

        self.cfg = cfg
        self._dev_debugged = False  # internal flag to print device info once

    # ---- helpers ----

    def unwrap(self) -> nn.Module:
        return self.model

    def move_batch_to_device(self, batch, device):
        """
        MMDet3D batches typically look like:

            dict{
                "inputs": {...},         # points / images / voxels, etc.
                "data_samples": [...],   # list[Det3DDataSample]
            }

        DataSamples are NOT fully moved here (the preprocessor & wrapper
        handle them separately). We only move pure tensors to device.
        """

        def to_dev(x):
            if isinstance(x, torch.Tensor):
                return x.to(device, non_blocking=True)
            if isinstance(x, list):
                # Keep DataSamples as-is; they are handled elsewhere.
                if len(x) > 0 and x[0].__class__.__name__.endswith("DataSample"):
                    return x
                return [to_dev(xx) for xx in x]
            if isinstance(x, dict):
                return {k: to_dev(v) for k, v in x.items()}
            return x

        return to_dev(batch)

    @staticmethod
    def _merge_list_batch(list_batch):
        """
        Merge a list[dict{"inputs":..., "data_samples":...}] into a single
        MMDet3D-style batch dict: {"inputs": merged_inputs, "data_samples": merged_samples}.
        """
        merged_inputs: Dict[str, Any] = {}
        merged_samples = []

        for item in list_batch:
            if not isinstance(item, dict):
                raise RuntimeError("Each element of list batch must be a dict.")
            ins = item.get("inputs", None)
            samp = item.get("data_samples", None)
            if ins is None or samp is None:
                raise RuntimeError("Each batch item must have 'inputs' and 'data_samples' keys.")

            # Merge inputs (usually dict fields are lists concatenated per batch)
            if isinstance(ins, dict):
                for k, v in ins.items():
                    merged_inputs.setdefault(k, []).append(v)
            else:
                merged_inputs.setdefault("inputs", []).append(ins)

            # Ensure we have a flat list of Det3DDataSample
            if isinstance(samp, (list, tuple)):
                merged_samples.extend(list(samp))
            else:
                merged_samples.append(samp)

        return merged_inputs, merged_samples

    @staticmethod
    def _scalarize_loss_dict(losses: Dict[str, Any], device) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Convert the loss dict returned by MMDet3D into:
          - total loss tensor
          - flat scalar logs dict
        """
        total = None
        logs: Dict[str, float] = {}

        for k, v in losses.items():
            name = f"losses/{k}"
            if isinstance(v, torch.Tensor):
                val = v.mean()
                logs[name] = float(val.detach().cpu())
                if "loss" in k.lower():
                    total = val if total is None else total + val
            elif isinstance(v, (int, float)):
                logs[name] = float(v)
            elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
                s = torch.stack([t.mean() for t in v]).sum()
                logs[name] = float(s.detach().cpu())
                if "loss" in k.lower():
                    total = s if total is None else total + s

        if total is None:
            # Defensive fallback to avoid None bubbling up.
            total = torch.tensor(0.0, device=device)

        logs["loss"] = float(total.detach().cpu())
        return total, logs

    # ---- prediction ----

    def predict_step(self, batch, amp: bool = False):
        """
        Run MMDet3D predict pipeline and return a list[Det3DDataSample]
        with pred_instances_3d fields populated.

        This normalizes batch formats:
          - dict{"inputs":..., "data_samples":[...]}
          - list[dict{"inputs":..., "data_samples":...}]
        """
        # Normalize batch to merged inputs + samples
        if isinstance(batch, dict):
            inputs = batch.get("inputs", None)
            data_samples = batch.get("data_samples", None)

            # Rare case: inputs is list[dict] (multi-batch per item)
            if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict):
                inputs, data_samples = self._merge_list_batch(
                    [{"inputs": i, "data_samples": s} for i, s in zip(inputs, data_samples)]
                )
        elif isinstance(batch, (list, tuple)):
            inputs, data_samples = self._merge_list_batch(batch)
        else:
            raise RuntimeError(f"Unsupported batch type for predict_step: {type(batch)}")

        if inputs is None or data_samples is None:
            raise RuntimeError("Missing 'inputs' or 'data_samples' in predict_step batch.")

        data_for_proc = {"inputs": inputs, "data_samples": data_samples}
        proc = self.model.data_preprocessor(data_for_proc, training=False)
        proc_inputs = proc.get("inputs", proc)
        proc_samples = proc.get("data_samples", data_samples)

        device = next(self.model.parameters()).device
        proc_inputs = _move_tensors_to_device(proc_inputs, device)
        _move_data_samples_to_device(proc_samples, device)

        if amp:
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                preds = self.model(proc_inputs, proc_samples, mode="predict")
        else:
            preds = self.model(proc_inputs, proc_samples, mode="predict")

        return preds

    # ---- loss / training ----

    def forward_loss(self, batch, amp: bool, scaler=None):
        """
        Normalize various batch types into a single MMDet3D-style batch,
        run the data preprocessor, and call model(..., mode="loss").

        Supported batch shapes:
          A) dict{"inputs": ..., "data_samples": [...]}
          B) list[dict{"inputs": ..., "data_samples": ...}]
             (e.g., from pseudo_collate / smart_collate)
        """
        # 1) Merge list batch if needed
        if isinstance(batch, dict):
            inputs = batch.get("inputs", None)
            data_samples = batch.get("data_samples", None)

            # Rare: inputs is list[dict]
            if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict):
                inputs, data_samples = self._merge_list_batch(
                    [{"inputs": i, "data_samples": s} for i, s in zip(inputs, data_samples)]
                )
        elif isinstance(batch, (list, tuple)):
            inputs, data_samples = self._merge_list_batch(batch)
        else:
            raise RuntimeError(f"Unsupported batch type for forward_loss: {type(batch)}")

        if inputs is None or data_samples is None:
            raise RuntimeError("Missing 'inputs' or 'data_samples' after collation/merge.")

        # 2) Run data_preprocessor
        try:
            data_for_proc = {"inputs": inputs, "data_samples": data_samples}
            proc = self.model.data_preprocessor(data_for_proc, training=self.training)

            proc_inputs = proc.get("inputs", proc)
            proc_samples = proc.get("data_samples", data_samples)
        except Exception as e:
            # Fallback: skip preprocessing but still log error.
            print("[DEBUG] MMDet3D data_preprocessor failed:", repr(e))
            proc_inputs, proc_samples = inputs, data_samples

        device = next(self.model.parameters()).device
        proc_inputs = _move_tensors_to_device(proc_inputs, device)
        _normalize_data_samples_on_device(proc_samples, device)

        # One-time debug print to check device/dtype alignment
        if not self._dev_debugged and isinstance(proc_samples, (list, tuple)) and len(proc_samples) > 0:
            gi = _first_attr_no_bool(proc_samples[0], ["gt_instances_3d", "gt_instances"])
            lbl = _first_attr_no_bool(gi, ["labels_3d", "labels"]) if gi is not None else None

            if lbl is not None and hasattr(lbl, "device"):
                print("[DBG] labels device/dtype:", lbl.device, getattr(lbl, "dtype", None))
            else:
                print("[DBG] labels missing or has no device attr:", type(lbl))

            vox = proc_inputs.get("voxels", None) if isinstance(proc_inputs, dict) else None
            if isinstance(vox, dict):
                any_vox = next((v for v in vox.values() if hasattr(v, "device")), None)
                print("[DBG] vox device:", getattr(any_vox, "device", None))

            self._dev_debugged = True

        # 3) Forward with loss
        if amp:
            from torch.amp import autocast
            with autocast(dtype=torch.float16):
                losses = self.model(proc_inputs, proc_samples, mode="loss")
        else:
            losses = self.model(proc_inputs, proc_samples, mode="loss")

        # 4) Aggregate loss
        total, logs = self._scalarize_loss_dict(losses, device=device)
        return total, logs


# =============================================================================
# OpenPCDet wrapper (example / template)
# =============================================================================

class OpenPCDetWrapper(BaseModelWrapper):
    """
    Wrapper for an OpenPCDet model.

    NOTE: OpenPCDet APIs differ slightly by version/branch. This wrapper
    assumes that calling `self.model(batch)` returns (ret_dict, tb_dict, disp_dict)
    and that at least one of those contains "loss" fields.

    Adapt this class to match your specific OpenPCDet version if needed.
    """

    def __init__(self, model_config: str, checkpoint: str = None, extra: Dict[str, Any] = None):
        super().__init__()
        extra = extra or {}

        try:
            from pcdet.config import cfg, cfg_from_yaml_file  # typical OpenPCDet interface
            from pcdet.models import build_network
        except Exception as e:
            raise ImportError(
                "OpenPCDet is not installed or cannot be imported. "
                "Ensure your PYTHONPATH includes the OpenPCDet repo."
            ) from e

        cfg_from_yaml_file(model_config, cfg)
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)

        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            # Most OpenPCDet checkpoints store weights under "model_state"
            self.model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)

        self.cfg = cfg

    def unwrap(self) -> nn.Module:
        return self.model

    def forward_loss(self, batch, amp: bool, scaler=None):
        """
        Run the OpenPCDet model on the given batch and extract total loss.

        Assumes:
          self.model(batch) -> (ret_dict, tb_dict, disp_dict)
        and at least one of these dicts contains losses.
        """
        if amp:
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                ret_dict, tb_dict, disp_dict = self.model(batch)
        else:
            ret_dict, tb_dict, disp_dict = self.model(batch)

        # Total loss: look for 'loss' in ret_dict, fallback to tb_dict, or sum all loss-like keys.
        if isinstance(ret_dict, dict) and "loss" in ret_dict:
            total = ret_dict["loss"]
        elif isinstance(tb_dict, dict) and "loss" in tb_dict:
            total = tb_dict["loss"]
        else:
            total = sum(
                v.mean()
                for d in (ret_dict, tb_dict, disp_dict)
                if isinstance(d, dict)
                for k, v in d.items()
                if "loss" in k.lower() and isinstance(v, torch.Tensor)
            )

        logs: Dict[str, float] = {}
        for d in (ret_dict, tb_dict, disp_dict):
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    logs[f"losses/{k}"] = float(v.mean().detach().cpu())
                elif isinstance(v, (int, float)):
                    logs[f"losses/{k}"] = float(v)

        logs["loss"] = float(total.detach().cpu())
        return total, logs


# =============================================================================
# Custom (generic) PyTorch model wrapper
# =============================================================================

class CustomModelWrapper(BaseModelWrapper):
    """
    Generic wrapper for an arbitrary nn.Module.

    Requirements for the underlying model:
      - It must be importable via extra["module"] and extra["class"].
      - It must accept a single `batch` argument and return:

          a) a scalar torch.Tensor loss, or
          b) a dict of loss components, where keys containing 'loss'
             are summed to form the total loss.

    Optional:
      - extra["model_kwargs"] dict passed to the model constructor.
      - If `checkpoint` is provided, the file can contain either:
          * {"model": <state_dict>}   or
          * a raw state_dict
    """

    def __init__(self, model_config: str = None, checkpoint: str = None, extra: Dict[str, Any] = None):
        super().__init__()
        extra = extra or {}

        import importlib

        module_path = extra.get("module", None)  # e.g. "myproj.models.my_net"
        class_name = extra.get("class", None)    # e.g. "MyNet"
        model_kwargs = extra.get("model_kwargs", {})

        assert module_path and class_name, "For backend 'custom', provide extra.module and extra.class"

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        self.model = cls(**(model_kwargs if isinstance(model_kwargs, dict) else {}))

        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            # Allow either a full dict with "model" key or a plain state_dict
            self.model.load_state_dict(state.get("model", state), strict=False)

    def unwrap(self) -> nn.Module:
        return self.model

    def forward_loss(self, batch, amp: bool, scaler=None):
        """
        Forward through the custom model and aggregate loss(s).

        Returns:
            total_loss: scalar tensor
            logs:       flat dict of loss-related scalars
        """
        if amp:
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                out = self.model(batch)
        else:
            out = self.model(batch)

        device = next(self.parameters()).device

        if isinstance(out, dict):
            total = None
            logs: Dict[str, float] = {}
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    val = v.mean()
                    logs[f"losses/{k}"] = float(val.detach().cpu())
                    if "loss" in k.lower():
                        total = val if total is None else total + val
                elif isinstance(v, (int, float)):
                    logs[f"losses/{k}"] = float(v)

            if total is None:
                total = torch.tensor(0.0, device=device)
        elif torch.is_tensor(out):
            total = out.mean()
            logs = {}
        else:
            raise RuntimeError("Custom model must return a loss tensor or a dict of losses.")

        logs["loss"] = float(total.detach().cpu())
        return total, logs


# =============================================================================
# Factory
# =============================================================================

def build_model_wrapper(
    backend: str,
    model_config: str = None,
    checkpoint: str = None,
    extra: Dict[str, Any] = None,
) -> BaseModelWrapper:
    """
    Factory to build the appropriate wrapper for the requested backend.

    Args:
        backend:       one of {"mmdet3d", "openpcdet", "custom"} (case-insensitive).
        model_config:  path to config file (MMDet3D/OpenPCDet) or unused for some custom models.
        checkpoint:    optional initial weights.
        extra:         extra configuration per backend:
                       - mmdet3d: {"cfg_options": {...}} to merge into Config.
                       - openpcdet: can pass anything you want (currently unused).
                       - custom: {"module": "...", "class": "...", "model_kwargs": {...}}.

    Returns:
        An instance of BaseModelWrapper (subclass).
    """
    backend = backend.lower()
    if backend == "mmdet3d":
        return MMDet3DWrapper(model_config=model_config, checkpoint=checkpoint, extra=extra)
    elif backend == "openpcdet":
        return OpenPCDetWrapper(model_config=model_config, checkpoint=checkpoint, extra=extra)
    elif backend == "custom":
        return CustomModelWrapper(model_config=model_config, checkpoint=checkpoint, extra=extra)
    else:
        raise ValueError(f"Unknown model backend: {backend}")