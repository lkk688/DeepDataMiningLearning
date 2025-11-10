#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
from pathlib import Path
from typing import Dict, Any
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import amp


from mmdet3d_models import build_model_wrapper, SaveLoadState  
from mmdet3d_datasets import build_dataloaders, build_evaluator

import time

class AverageMeter:
    def __init__(self, momentum=0.98):
        self.momentum = momentum
        self.val = 0.0
        self.avg = 0.0
        self.initialized = False

    def update(self, v):
        self.val = float(v)
        if not self.initialized:
            self.avg = self.val
            self.initialized = True
        else:
            self.avg = self.momentum * self.avg + (1 - self.momentum) * self.val

def _format_eta(seconds):
    # 返回 "1 day, 4:46:30" 的格式
    seconds = int(max(0, seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d > 0:
        return f"{d} day, {h:01d}:{m:02d}:{s:02d}"
    return f"{h:01d}:{m:02d}:{s:02d}"

def _get_base_lr(optimizer, scheduler):
    # base_lr 来自 scheduler.base_lrs 或 optimizer 初始 lr
    try:
        if hasattr(scheduler, "base_lrs"):
            return float(scheduler.base_lrs[0])
    except Exception:
        pass
    return float(optimizer.param_groups[0]["initial_lr"]) if "initial_lr" in optimizer.param_groups[0] else float(optimizer.param_groups[0]["lr"])

def _current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])

def _gpu_memory_mb():
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return 0

def _grad_total_norm(parameters, norm_type=2.0):
    # 统计梯度范数：在 scaler.unscale_(optimizer) 之后调用才准确
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return 0.0
    device = params[0].grad.device
    if norm_type == float("inf"):
        total = max(p.grad.detach().abs().max().to(device) for p in params)
        return float(total.item())
    norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in params]), norm_type)
    return float(norm.item())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
                lr = 0.5 * base_lr * (1 + math.cos(math.pi * t))
            lrs.append(lr)
        return lrs

def _typen(x):
    return x.__class__.__name__

def _shape(x):
    try:
        import numpy as _np
        if hasattr(x, "shape"):
            return tuple(x.shape)
        if isinstance(x, (list, tuple)) and len(x) > 0 and hasattr(x[0], "shape"):
            # list of tensors/arrays
            return [tuple(getattr(t, "shape", ())) for t in x]
    except Exception:
        pass
    return None

def _summarize_tensor(t):
    try:
        return {"type": _typen(t), "dtype": getattr(t, "dtype", None).__str__() if hasattr(t, "dtype") else None,
                "shape": _shape(t), "device": str(getattr(t, "device", "cpu"))}
    except Exception:
        return {"type": _typen(t)}

def _print_kv(indent, key, val):
    pad = " " * indent
    print(f"{pad}- {key}: {_typen(val)}", end="")
    shp = _shape(val)
    if shp is not None:
        print(f", shape={shp}")
    else:
        if isinstance(val, (list, tuple)):
            print(f", len={len(val)}")
        elif isinstance(val, dict):
            print(f", keys={list(val.keys())[:8]}{'...' if len(val)>8 else ''}")
        else:
            print()

def _describe_inputs(inputs, indent=2):
    pad = " " * indent
    if not isinstance(inputs, (dict, list, tuple)):
        _print_kv(indent, "inputs", inputs)
        return
    if isinstance(inputs, dict):
        print(f"{pad}inputs: dict keys={list(inputs.keys())}")
        # 常见键的摘要
        for k in inputs.keys():
            v = inputs[k]
            if k in ("points", "imgs", "img", "images"):
                if isinstance(v, (list, tuple)):
                    print(f"{pad}  {k}: list len={len(v)}, elem_type={_typen(v[0]) if len(v)>0 else None}, elem_shape={_shape(v[0]) if len(v)>0 else None}")
                else:
                    _print_kv(indent+2, k, v)
            else:
                _print_kv(indent+2, k, v)
    else:  # list/tuple
        print(f"{pad}inputs: {_typen(inputs)}, len={len(inputs)}")
        if len(inputs) > 0:
            _print_kv(indent+2, "inputs[0]", inputs[0])

def _first_attr(obj, names):
    """Return the first existing attribute in names without boolean coercion."""
    for n in names:
        if hasattr(obj, n):
            try:
                return getattr(obj, n)
            except Exception:
                pass
    return None

def _describe_data_samples(samples, indent=2):
    pad = " " * indent

    def _safe_len(x):
        try:
            if hasattr(x, "numel"):
                return int(x.numel())
            return len(x)  # may still raise for objects without __len__
        except Exception:
            return None

    if isinstance(samples, (list, tuple)):
        n = len(samples)
        print(f"{pad}data_samples: list len={n} elem_type={_typen(samples[0]) if n>0 else None}")

        if n == 0:
            return

        s0 = samples[0]
        # 列出常见字段（尽量不触发重运算）
        candidate_fields = [
            "gt_instances_3d", "gt_instances", "pred_instances_3d",
            "metainfo", "img_shape", "lidar_points", "cam2img"
        ]
        present = [f for f in candidate_fields if hasattr(s0, f)]
        print(f"{pad}  sample[0] has fields: {present}")

        gi = _first_attr(s0, ["gt_instances_3d", "gt_instances"])
        if gi is not None:
            bboxes = _first_attr(gi, ["bboxes_3d", "bboxes"])
            labels = _first_attr(gi, ["labels_3d", "labels"])

            if bboxes is not None:
                print(f"{pad}  gt_instances.bboxes*: type={_typen(bboxes)}")

            if labels is not None:
                linfo = {"type": _typen(labels)}
                l_len = _safe_len(labels)
                if l_len is not None:
                    linfo["len"] = l_len
                try:
                    if hasattr(labels, "dtype"):
                        linfo["dtype"] = str(labels.dtype)
                except Exception:
                    pass
                print(f"{pad}  gt_instances.labels*: {linfo}")
    else:
        print(f"{pad}data_samples: {_typen(samples)}")

def _summarize_batch(batch, idx=0):
    print(f"\n=== Inspect batch #{idx} ===")
    # 形态 A：dict{"inputs":..., "data_samples":...}
    if isinstance(batch, dict):
        keys = list(batch.keys())
        print(f"batch: dict keys={keys}")
        if "inputs" in batch: _describe_inputs(batch["inputs"])
        if "data_samples" in batch: _describe_data_samples(batch["data_samples"])
        # 其他 key 简略打印
        for k in keys:
            if k not in ("inputs", "data_samples"):
                _print_kv(2, k, batch[k])
        return

    # 形态 B：list[dict]（pseudo_collate）
    if isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], dict):
        print(f"batch: list[dict], len={len(batch)}")
        b0 = batch[0]
        print(" first element structure:")
        _summarize_batch(b0, idx="first-of-list")
        return

    # 其他：直接打印类型与形状
    _print_kv(0, "batch", batch)

def inspect_dataloader(dataloader, max_batches=2):
    """
    拉取前 max_batches 个 batch，打印结构；若 DataLoader worker 抛错，给出指引。
    """
    print(f"\n[Dataset Inspect] Will fetch {max_batches} batch(es) from train_loader...")
    got = 0
    try:
        for i, batch in enumerate(dataloader):
            _summarize_batch(batch, idx=i)
            got += 1
            if got >= max_batches:
                break
    except Exception as e:
        print("\n[Inspect ERROR] Exception occurred while iterating DataLoader:")
        import traceback; traceback.print_exc()
        print("\nHints:")
        print("  1) 尝试把 --num-workers 设为 0 再跑一次自检，方便看到原始异常；")
        print("  2) 若是 shape 不一致导致 default_collate 报错，改用 pseudo/smart collate；")
        print("  3) 检查数据根路径与文件是否存在（可用 --data-root 或自动探测）；")
        print("  4) 多模态/多相机任务返回 list[dict] 很正常，模型前向会处理。")
        raise e
    else:
        print("\n[Dataset Inspect] Done.")

def _wb_sanitize(v):
    import numpy as np
    import torch
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v.detach().cpu().numpy().tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (float, int, str, bool)) or v is None:
        return v
    # last-resort cast
    try:
        return float(v)
    except Exception:
        return str(v)

from typing import List
import torch

def _extract_gt_samples_strict(batch) -> List:
    """从 val batch 取 GT（Det3DDataSample 列表），绝不返回字符串。"""
    if isinstance(batch, dict):
        if "data_samples" not in batch:
            raise RuntimeError("val batch missing key 'data_samples'")
        ds = batch["data_samples"]
        return ds if isinstance(ds, list) else [ds]
    if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], dict):
        out = []
        for i, item in enumerate(batch):
            if "data_samples" not in item:
                raise RuntimeError(f"item[{i}] missing key 'data_samples'")
            ds = item["data_samples"]
            out.extend(ds if isinstance(ds, list) else [ds])
        return out
    raise RuntimeError(f"Unsupported val batch type: {type(batch)}")

def _empty_instancedata(device):
    from mmengine.structures import InstanceData
    idt = InstanceData()
    # 2D可为空；3D至少给 scores/labels 空张量，bboxes_3d 可由上游填或置空
    idt.scores_3d = torch.empty(0, device=device)
    idt.labels_3d = torch.empty(0, dtype=torch.long, device=device)
    return idt

def instancedata_to_plain_dict(x):
    """
    Convert mmengine.structures.InstanceData into a plain dict that
    NuScenesMetric.process() expects to index & .to('cpu') per field.

    Keeps standard keys if present:
      - 'bboxes_3d' (LiDARInstance3DBoxes / CameraInstance3DBoxes)
      - 'scores_3d' (Tensor)
      - 'labels_3d' (Tensor)
      - 'attr_labels' (optional, Tensor)
    If x is already a dict, return a shallow copy.
    If x is None, return an empty dict.
    """
    from mmengine.structures import InstanceData
    if x is None:
        return {}
    if isinstance(x, dict):
        # ensure a shallow copy so we don't mutate caller's object
        return {k: v for k, v in x.items()}
    if isinstance(x, InstanceData):
        out = {}
        # Only pick the fields NuScenesMetric will know how to handle
        for k in ("bboxes_3d", "scores_3d", "labels_3d", "attr_labels"):
            if hasattr(x, k):
                out[k] = getattr(x, k)
        return out
    # Fallback: unknown type -> empty dict (avoids __getitem__ assertions)
    return {}

def _build_nuscenes_pred_records(
    self, preds: Any, gt_samples: List[Any], device: torch.device
) -> List[Dict[str, Any]]:
    # 1) normalize container
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds if isinstance(preds, (list, tuple)) else [preds]
    preds = list(preds)
    if len(preds) == 1 and isinstance(preds[0], (list, tuple)):
        preds = list(preds[0])

    records: List[Dict[str, Any]] = []
    for i, p in enumerate(preds):
        # 2) fetch 3D preds (Det3DDataSample or dict)
        pred3d = getattr(p, "pred_instances_3d", None)
        if pred3d is None and isinstance(p, dict):
            pred3d = p.get("pred_instances_3d", None)
        pred3d = instancedata_to_plain_dict(pred3d)  # <<< KEY CHANGE

        # 3) optional 2D preds
        pred2d = getattr(p, "pred_instances", None)
        if pred2d is None and isinstance(p, dict):
            pred2d = p.get("pred_instances", None)
        pred2d = instancedata_to_plain_dict(pred2d)  # <<< also dict

        # 4) resolve sample_idx
        sample_idx = None
        meta = getattr(p, "metainfo", None)
        if isinstance(meta, dict):
            sample_idx = meta.get("sample_idx", None)
        if sample_idx is None:
            gi = gt_samples[min(i, len(gt_samples) - 1)]
            if hasattr(gi, "metainfo") and isinstance(gi.metainfo, dict):
                sample_idx = gi.metainfo.get("sample_idx", None)
            if sample_idx is None:
                sample_idx = getattr(gi, "sample_idx", None)
        if sample_idx is None:
            raise RuntimeError(f"Cannot resolve sample_idx for pred #{i}")

        records.append(
            {
                "pred_instances_3d": pred3d,   # dict (NOT InstanceData)
                "pred_instances": pred2d,      # dict (or empty)
                "sample_idx": int(sample_idx),
            }
        )

    if not records or not isinstance(records[0], dict):
        raise TypeError("pred_records must be list[dict].")
    return records

def safe_process_nuscenes(evaluator, val_batch, preds, device):
    """唯一正确的调用：GT 放 data_batch，PRED 列表放第二参。"""
    gt_samples   = _extract_gt_samples_strict(val_batch)             # list[Det3DDataSample]
    pred_records = _build_nuscenes_pred_records(preds, gt_samples, device)  # list[dict]

    # —— 强力防呆：禁止字符串、禁止默认回退成'data_samples' —— #
    if isinstance(pred_records, (str, bytes)):
        raise TypeError(f"BUG: pred_records is {type(pred_records)}")
    if not isinstance(pred_records, list) or not pred_records or not isinstance(pred_records[0], dict):
        raise TypeError(f"BUG: pred_records must be list[dict], got {type(pred_records)}")

    # 可打印一条检查，确认不会再传字符串
    # print("[DEBUG] process with", type(pred_records), type(pred_records[0]), "first keys:", list(pred_records[0].keys()))

    evaluator.process({"data_samples": gt_samples}, pred_records)    # ✅ 正确

def _norm_pred_to_mapping_list(preds) -> List[dict]:
    """
    将预测规范化为 List[dict]，每个元素至少包含 'pred_instances_3d'，
    以便 NuScenesMetric 做 data_sample['pred_instances_3d'] 访问。
    """
    if isinstance(preds, tuple):  # e.g. (preds, aux)
        preds = preds[0]

    if not isinstance(preds, (list, tuple)):
        preds = [preds]
    else:
        preds = list(preds)

    if len(preds) == 1 and isinstance(preds[0], (list, tuple)):
        preds = list(preds[0])

    out, bad = [], []
    for i, p in enumerate(preds):
        if isinstance(p, (str, bytes)):
            bad.append(f"idx {i}: got {type(p).__name__}"); continue

        # 已是 dict 且有 pred_instances_3d
        if isinstance(p, dict) and "pred_instances_3d" in p:
            out.append(p); continue

        # Det3DDataSample 的属性访问
        inst = getattr(p, "pred_instances_3d", None)
        if inst is not None:
            meta = getattr(p, "metainfo", None)
            item = {"pred_instances_3d": inst}
            if isinstance(meta, dict):
                item["metainfo"] = meta
            out.append(item); continue

        # 类 dict 的 .get 访问
        try:
            if hasattr(p, "get") and p.get("pred_instances_3d", None) is not None:
                out.append({"pred_instances_3d": p.get("pred_instances_3d")}); continue
        except Exception:
            pass

        bad.append(f"idx {i}: {type(p).__name__}")

    if bad:
        raise TypeError("predictions missing 'pred_instances_3d':\n  " + "\n  ".join(bad))
    return out



class EvalRunner:
    """Run val/test evaluation with robust batch/pred normalization."""
    def __init__(
        self,
        model_wrapper,
        val_loader,
        evaluator,
        device,
        use_wandb: bool = False,
        fp16: bool = False,
        max_eval_iters: int = 0,
        label: str = "val",
    ):
        self.model_wrapper = model_wrapper
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.device = device
        self.use_wandb = use_wandb
        self.fp16 = fp16
        self.max_eval_iters = max_eval_iters
        self.label = label

    @staticmethod
    def _flatten_gt_from_batch(batch):
        """Return list[Det3DDataSample] ground-truths for evaluator.process(data_batch, preds)."""
        # mmdet3d expects data_batch to be {"data_samples": list[Det3DDataSample]}
        if isinstance(batch, dict):
            ds = batch.get("data_samples", None)
            if ds is None:
                # pseudo_collate may return {"inputs":..., "data_samples":...}; if not, try to reconstruct
                raise RuntimeError("Batch dict has no 'data_samples'. Cannot evaluate.")
            return ds if isinstance(ds, list) else [ds]

        if isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], dict):
            # list[dict] → extract each one's data_samples (usually a single sample)
            out = []
            for item in batch:
                ds = item.get("data_samples", None)
                if ds is None:
                    raise RuntimeError("Item in list batch has no 'data_samples'.")
                if isinstance(ds, list):
                    out.extend(ds)
                else:
                    out.append(ds)
            return out

        raise RuntimeError(f"Unsupported batch type for GT extraction: {type(batch)}")

    @staticmethod
    def _has_pred3d(x) -> bool:
        try:
            _ = x['pred_instances_3d']  # Det3DDataSample supports __getitem__
            return True
        except Exception:
            return False

    @staticmethod
    def _normalize_preds(preds):
        """
        Normalize predictions so each item is a Mapping with key 'pred_instances_3d'.
        Accept:
          - Det3DDataSample (attr-only)  -> wrap to {'pred_instances_3d': ds.pred_instances_3d, 'metainfo': ds.metainfo}
          - dict with 'pred_instances_3d' -> keep
          - list/tuple of above           -> flatten to list
          - (preds, aux)                  -> take preds
        """
        if isinstance(preds, tuple):
            preds = preds[0]

        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        else:
            preds = list(preds)

        if len(preds) == 1 and isinstance(preds[0], (list, tuple)):
            preds = list(preds[0])

        out, bad = [], []
        for i, p in enumerate(preds):
            if isinstance(p, (str, bytes)):
                bad.append(f"idx {i}: str/bytes"); continue

            if isinstance(p, dict) and ("pred_instances_3d" in p):
                out.append(p); continue

            pred_inst = getattr(p, "pred_instances_3d", None)
            if pred_inst is not None:
                meta = getattr(p, "metainfo", None)
                item = {"pred_instances_3d": pred_inst}
                if isinstance(meta, dict):
                    item["metainfo"] = meta
                out.append(item); continue

            try:
                if hasattr(p, "get") and p.get("pred_instances_3d", None) is not None:
                    out.append({"pred_instances_3d": p.get("pred_instances_3d")})
                    continue
            except Exception:
                pass

            bad.append(f"idx {i}: {type(p).__name__}")

        if bad:
            raise TypeError(
                "EvalRunner: predictions missing 'pred_instances_3d' after normalization:\n  "
                + "\n  ".join(bad)
            )
        return out

    @torch.no_grad()
    def run(self, epoch_idx: int, global_step: int):
        if self.val_loader is None or self.evaluator is None:
            return {}

        self.model_wrapper.eval()
        n_samples = 0

        for vi, vb in enumerate(self.val_loader):
            if self.max_eval_iters and vi >= self.max_eval_iters:
                break

            # 1) 移到 device（保持你原来的封装）
            vb = self.model_wrapper.move_batch_to_device(vb, self.device)

            # 2) 预测
            preds = self.model_wrapper.predict_step(vb, amp=self.fp16)

            # 3) 严格提取 GT（list[Det3DDataSample]），并把预测规范成 NuScenes 需要的 list[dict]
            gt_samples   = _extract_gt_samples_strict(vb)
            pred_records = _build_nuscenes_pred_records(preds, gt_samples, device=self.device)

            # 4) 防呆：再也不允许把字符串传进去
            if isinstance(pred_records, (str, bytes)):
                raise TypeError("BUG: pred_records is a string; never pass 'data_samples' literal.")
            if not isinstance(pred_records, list) or not pred_records or not isinstance(pred_records[0], dict):
                raise TypeError(f"BUG: pred_records must be list[dict], got {type(pred_records)}")

            # 5) 正确调用：GT 放 data_batch，预测放第二个参数
            self.evaluator.process({"data_samples": gt_samples}, pred_records)
            n_samples += len(gt_samples)

        results = self.evaluator.evaluate(n_samples)

        # ===== 打印 + W&B =====
        line = "  ".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in results.items()])
        print(f"[Eval:{self.label}][epoch {epoch_idx}] {line}")

        if self.use_wandb:
            import wandb, numpy as np, torch as _t
            def _wb(v):
                if isinstance(v, _t.Tensor): return v.item() if v.numel()==1 else v.detach().cpu().numpy().tolist()
                if isinstance(v, (np.floating, np.integer)): return v.item()
                if isinstance(v, np.ndarray): return v.tolist()
                if isinstance(v, (float,int,str,bool)) or v is None: return v
                try: return float(v)
                except Exception: return str(v)
            payload = {f"{self.label}/{k}": _wb(v) for k, v in results.items()}
            payload[f"{self.label}/epoch"] = int(epoch_idx)
            wandb.log(payload, step=global_step)

        return results

#~/Developer/mmdetection3d/modelzoo$ mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .
"""
['3dssd_4x4_kitti-3d-car', 'centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_voxel01_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_pillar02_second_secfpn_head-dcn_8xb4-cyclic-20e_nus-3d', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area1.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area2.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area3.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area4.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area5.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area6.py', 'dv_second_secfpn_6x8_80e_kitti-3d-car', 'dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class', 'dv_pointpillars_secfpn_6x8_160e_kitti-3d-car', 'fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune', 'pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-400mf_fpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-400mf_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d', 'hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d', 'pointpillars_hv_regnet-1.6gf_fpn_head-free-anchor_sbn-all_8xb4-strong-aug-3x_nus-3d', 'pointpillars_hv_regnet-3.2gf_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-3.2gf_fpn_head-free-anchor_sbn-all_8xb4-strong-aug-3x_nus-3d', 'groupfree3d_head-L6-O256_4xb8_scannet-seg.py', 'groupfree3d_head-L12-O256_4xb8_scannet-seg.py', 'groupfree3d_w2x-head-L12-O256_4xb8_scannet-seg.py', 'groupfree3d_w2x-head-L12-O512_4xb8_scannet-seg.py', 'h3dnet_3x8_scannet-3d-18class', 'imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class', 'imvotenet_stage2_16x8_sunrgbd-3d-10class', 'imvoxelnet_kitti-3d-car', 'monoflex_dla34_pytorch_dlaneck_gn-all_2x4_6x_kitti-mono3d', 'dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class', 'mask-rcnn_r50_fpn_1x_nuim', 'mask-rcnn_r50_fpn_coco-2x_1x_nuim', 'mask-rcnn_r50_caffe_fpn_1x_nuim', 'mask-rcnn_r50_caffe_fpn_coco-3x_1x_nuim', 'mask-rcnn_r50_caffe_fpn_coco-3x_20e_nuim', 'mask-rcnn_r101_fpn_1x_nuim', 'mask-rcnn_x101_32x4d_fpn_1x_nuim', 'cascade-mask-rcnn_r50_fpn_1x_nuim', 'cascade-mask-rcnn_r50_fpn_coco-20e_1x_nuim', 'cascade-mask-rcnn_r50_fpn_coco-20e_20e_nuim', 'cascade-mask-rcnn_r101_fpn_1x_nuim', 'cascade-mask-rcnn_x101_32x4d_fpn_1x_nuim', 'htc_r50_fpn_coco-20e_1x_nuim', 'htc_r50_fpn_coco-20e_20e_nuim', 'htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim', 'paconv_ssg_8xb8-cosine-150e_s3dis-seg.py', 'paconv_ssg-cuda_8xb8-cosine-200e_s3dis-seg', 'parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class', 'parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-car', 'pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d', 'pgd_r101-caffe_fpn_head-gn_16xb2-1x_nus-mono3d', 'pgd_r101-caffe_fpn_head-gn_16xb2-1x_nus-mono3d_finetune', 'pgd_r101-caffe_fpn_head-gn_16xb2-2x_nus-mono3d', 'pgd_r101-caffe_fpn_head-gn_16xb2-2x_nus-mono3d_finetune', 'point-rcnn_8xb2_kitti-3d-3class', 'pointnet2_ssg_2xb16-cosine-200e_scannet-seg-xyz-only', 'pointnet2_ssg_2xb16-cosine-200e_scannet-seg', 'pointnet2_msg_2xb16-cosine-250e_scannet-seg-xyz-only', 'pointnet2_msg_2xb16-cosine-250e_scannet-seg', 'pointnet2_ssg_2xb16-cosine-50e_s3dis-seg', 'pointnet2_msg_2xb16-cosine-80e_s3dis-seg', 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car', 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class', 'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_secfpn_sbn-all_8xb4-amp-2x_nus-3d', 'pointpillars_hv_fpn_sbn-all_8xb4-amp-2x_nus-3d', 'pointpillars_hv_secfpn_sbn-all_8xb2-2x_lyft-3d', 'pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymoD5-3d-car', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymoD5-3d-3class', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymo-3d-car', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymo-3d-3class', 'pointpillars_hv_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d', 'pointpillars_hv_regnet-1.6gf_fpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d', 'pointpillars_hv_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d', 'second_hv_secfpn_8xb6-80e_kitti-3d-car', 'second_hv_secfpn_8xb6-80e_kitti-3d-3class', 'second_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class', 'second_hv_secfpn_8xb6-amp-80e_kitti-3d-car', 'second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class', 'smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d', 'hv_ssn_secfpn_sbn-all_16xb2-2x_nus-3d', 'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d', 'hv_ssn_secfpn_sbn-all_16xb2-2x_lyft-3d', 'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb1-2x_lyft-3d', 'votenet_8xb16_sunrgbd-3d.py', 'votenet_8xb8_scannet-3d.py', 'votenet_iouloss_8x8_scannet-3d-18class', 'minkunet18_w16_torchsparse_8xb2-amp-15e_semantickitti', 'minkunet18_w20_torchsparse_8xb2-amp-15e_semantickitti', 'minkunet18_w32_torchsparse_8xb2-amp-15e_semantickitti', 'minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_spconv_8xb2-amp-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_spconv_8xb2-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_torchsparse_8xb2-laser-polar-mix-3x_semantickitti', 'minkunet34v2_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti', 'cylinder3d_4xb4-3x_semantickitti', 'cylinder3d_8xb2-laser-polar-mix-3x_semantickitti', 'pv_rcnn_8xb2-80e_kitti-3d-3class', 'fcaf3d_2xb8_scannet-3d-18class', 'fcaf3d_2xb8_sunrgbd-3d-10class', 'fcaf3d_2xb8_s3dis-3d-5class', 'spvcnn_w16_8xb2-amp-15e_semantickitti', 'spvcnn_w20_8xb2-amp-15e_semantickitti', 'spvcnn_w32_8xb2-amp-15e_semantickitti', 'spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti']
"""
def parse_args():
    p = argparse.ArgumentParser("Universal PyTorch training loop (MMDet3D/OpenPCDet/Custom)")
    # 通用
    p.add_argument("--work-dir", default="outputs/mm3dtrain", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--epochs", default=4, type=int)
    p.add_argument("--batch-size", default=2, type=int)
    p.add_argument("--num-workers", default=0, type=int)
    p.add_argument("--accum-steps", default=1, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--weight-decay", default=0.01, type=float)
    p.add_argument("--clip-grad-norm", default=None, type=float)
    p.add_argument("--warmup-epochs", default=0.0, type=float)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--no-pin-memory", action="store_true")

    # model and data related
    p.add_argument("--model-backend", choices=["mmdet3d", "openpcdet", "custom"], default="mmdet3d")
    p.add_argument("--model-config", type=str, default="/home/lkk688/Developer/mmdetection3d/modelzoo/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py", help="Path to model config (backend-specific)")
    p.add_argument("--model-checkpoint", type=str, default="/home/lkk688/Developer/mmdetection3d/modelzoo/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth", help="Initial weights to load")
    p.add_argument("--data-backend", choices=["mmdet3d", "custom"], default="mmdet3d")
    p.add_argument("--data-config", type=str, default="", help="Path to dataset config (backend-specific)")
    p.add_argument("--data-root", type=str, default="/home/lkk688/Developer/mmdetection3d/data/nuscenes", help="Global dataset root. If set, we will use <data-root>/<auto-dataset-subdir> as data_root.")
    p.add_argument("--inspect-dataset", type=int, default=1, help="If >0, print structure of the first N batches from train_loader and exit.")
    
    # W&B
    p.add_argument("--wandb-project", type=str, default="mmdet3d_mytrain")
    p.add_argument("--wandb-name", type=str, default="pointpillars_nus_test1")
    p.add_argument("--wandb-offline", action="store_true")
    
    # Evaluate
    p.add_argument("--eval-interval", type=int, default=1, help="Evaluate every N epochs (0 to disable).")
    p.add_argument("--max-eval-iters", type=int, default=0, help="Limit validation iters for quick eval (0=all).")
    p.add_argument("--save-best-metric", type=str, default=None, help="Metric key to track for best checkpoint, e.g., 'NDS' or 'mAP_3d'.")

    # 自定义补充参数（可选，透传给后端）
    p.add_argument("--extra", nargs="*", default=[], help="Extra key=val pairs passed to backends")
    return p.parse_args()


def parse_kv_list(kv_list):
    out = {}
    for kv in kv_list:
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def get_param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".bias") or any(x in n for x in ["norm", "bn", "ln", "gn"]):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def save_state(path: str, model_wrapper: SaveLoadState, optimizer, scheduler, scaler, epoch: int, best: float):
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


def load_state(path: str, model_wrapper: SaveLoadState, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
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
    print(f"[Info] Resume from {path}: start_epoch={start_epoch}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss

def _resolve_nuscenes_meta_from_cfg(config_path: str, extra: dict):
    """Return (data_root, ann_file, classes, version) for NuScenes eval.

    Rule: if user provided extra['data_root'], use it as-is.
          else default to <mmdetection3d>/data/nuscenes (fallback: ./data/nuscenes).
    """
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    import os

    cfg = Config.fromfile(config_path)
    if isinstance(extra.get("cfg_options"), dict):
        cfg.merge_from_dict(extra["cfg_options"])
    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    # 1) pick val dataset dict (兼容新旧写法)
    if "val_dataloader" in cfg and "dataset" in cfg.val_dataloader:
        ds_val = cfg.val_dataloader["dataset"]
    else:
        ds_val = getattr(cfg, "data", {}).get("val", None)
    assert isinstance(ds_val, dict), "Config missing val dataset dict."

    # 2) data_root: prefer user → default mmdet3d/data/nuscenes → ./data/nuscenes
    data_root = extra.get("data_root")
    if not data_root:
        data_root = _default_mmdet3d_nus_root()
        if not os.path.isdir(data_root):
            data_root = os.path.join("data", "nuscenes")

    # 3) ann_file from cfg
    ann_file = ds_val.get("ann_file") or ds_val.get("data_prefix", {}).get("ann_file")
    if not ann_file:
        raise RuntimeError("Cannot resolve ann_file for val split from config.")
    if not os.path.isabs(ann_file):
        ann_file = os.path.join(data_root, ann_file)

    # 4) classes & version
    metainfo = ds_val.get("metainfo", {}) or {}
    classes = metainfo.get("classes") or ds_val.get("classes")
    if classes is None:
        raise RuntimeError("Cannot resolve class names (metainfo.classes).")
    version = ds_val.get("version") or metainfo.get("version") or "v1.0-trainval"

    return data_root, ann_file, list(classes), str(version)


def _default_mmdet3d_nus_root():
    """Return <mmdetection3d>/data/nuscenes if possible, else <mmdetection3d>/data."""
    import os, mmdet3d
    mm_root = os.path.dirname(os.path.abspath(mmdet3d.__file__))          # .../mmdet3d
    repo_root = os.path.dirname(mm_root)                                   # repo root
    cand = os.path.join(repo_root, "data", "nuscenes")
    return cand if os.path.isdir(cand) else os.path.join(repo_root, "data")

def build_nuscenes_evaluator(args, extra, val_loader, model_wrapper, device):
    """Factory that returns a ready-to-use SimpleNuScenesEvaluator, or None if val_loader is None."""
    if val_loader is None:
        return None

    # infer data_root / ann_file / classes / version from your config (or args.data_config fallback)
    data_cfg = args.data_config or args.model_config
    data_root, ann_file, classes, version = _resolve_nuscenes_meta_from_cfg(data_cfg, extra)

    from mmdet3d_evaluator import SimpleNuScenesEvaluator
    evaluator = SimpleNuScenesEvaluator(
        data_root=data_root,
        ann_file=ann_file,
        dataset_meta={"classes": classes, "version": version},
        jsonfile_prefix=os.path.join(args.work_dir, "nusc_eval_outputs"),
        metric="bbox",
        eval_version="detection_cvpr_2019",
        collect_device="cpu",
        format_only=False,
        backend_args=None,
    )
    return evaluator


def run_one_full_eval(evaluator, val_loader, model_wrapper, device, amp=False, max_eval_iters=0):
    """A tiny, explicit evaluation loop (no extra runner needed)."""
    if evaluator is None or val_loader is None:
        return {}

    evaluator.reset()
    model_wrapper.eval()
    with torch.no_grad():
        for vi, vb in enumerate(val_loader):
            if max_eval_iters and vi >= max_eval_iters:
                break
            vb = model_wrapper.move_batch_to_device(vb, device)
            preds = model_wrapper.predict_step(vb, amp=amp)
            evaluator.process_batch(vb, preds, device=device)
    return evaluator.compute()

def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    set_seed(args.seed)
    extra = parse_kv_list(args.extra)
    if args.data_root:
        extra["data_root"] = args.data_root

    # 1) Model wrapper
    model_wrapper = build_model_wrapper(
        backend=args.model_backend,
        model_config=args.model_config,
        checkpoint=args.model_checkpoint,
        extra=extra,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper.to(device)

    # 2) Dataloaders
    train_loader, val_loader = build_dataloaders(
        backend=args.data_backend,
        data_config=args.data_config or args.model_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        extra=extra,
    )

    # 2.1 (optional) dataset self-inspect
    if args.inspect_dataset and args.inspect_dataset > 0:
        if args.num_workers != 0:
            print("[Note] For clearer errors during inspect, consider re-running with --num-workers 0")
        inspect_dataloader(train_loader, max_batches=args.inspect_dataset)

    # 3) Optimizer / Scheduler / AMP
    optimizer = torch.optim.AdamW(get_param_groups(model_wrapper.unwrap(), args.weight_decay),
                                  lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler = amp.GradScaler(device="cuda", enabled=args.fp16)

    # 4) W&B
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import os as _os
        if args.wandb_offline:
            _os.environ["WANDB_MODE"] = "offline"
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, config={
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "accum_steps": args.accum_steps,
            "warmup_epochs": args.warmup_epochs, "fp16": args.fp16,
            "model_backend": args.model_backend, "data_backend": args.data_backend,
            "model_config": args.model_config, "data_config": args.data_config,
        })

    # 5) Resume (optional)
    start_epoch, best_val = 0, float("inf")
    if args.resume_from and os.path.isfile(args.resume_from):
        start_epoch, best_val = load_state(args.resume_from, model_wrapper, optimizer, scheduler, scaler, map_location=device)
        scheduler.last_epoch = start_epoch * steps_per_epoch

    # 6) Build evaluator (modular) + pre-eval
    evaluator = build_nuscenes_evaluator(args, extra, val_loader, model_wrapper, device)
    if evaluator is not None:
        pre_metrics = run_one_full_eval(evaluator, val_loader, model_wrapper, device, amp=False, max_eval_iters=getattr(args, "max_eval_iters", 0))
        print("[Pre-Eval]", pre_metrics)
        if use_wandb:
            import wandb
            wandb.log({f"val/{k}": (float(v) if isinstance(v, (int,float)) else v) for k,v in pre_metrics.items()},
                      step=0)

    # 7) Training loop
    global_step = start_epoch * steps_per_epoch
    best_metric_map = {}
    for epoch in range(start_epoch, args.epochs):
        model_wrapper.train()
        optimizer.zero_grad(set_to_none=True)

        iter_time_meter = AverageMeter()
        data_time_meter = AverageMeter()
        last_iter_end = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for it, batch in enumerate(train_loader):
            now = time.time()
            data_time = now - last_iter_end
            data_time_meter.update(data_time)

            batch = model_wrapper.move_batch_to_device(batch, device)
            total_loss, logs = model_wrapper.forward_loss(batch, amp=args.fp16, scaler=scaler)

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
                    torch.nn.utils.clip_grad_norm_(model_wrapper.unwrap().parameters(), args.clip_grad_norm)
                try:
                    grad_norm_value = _grad_total_norm(model_wrapper.unwrap().parameters(), norm_type=2.0)
                except Exception:
                    grad_norm_value = None

                if args.fp16:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            global_step += 1

            iter_time = time.time() - now
            iter_time_meter.update(iter_time)

            # ETA
            iter_done = epoch * steps_per_epoch + (it + 1)
            iters_total = args.epochs * steps_per_epoch
            eta_seconds = max(iters_total - iter_done, 0) * max(iter_time_meter.avg, 1e-6)

            # GPU mem
            mem_mb = _gpu_memory_mb()

            base_lr = _get_base_lr(optimizer, scheduler)
            lr_now = _current_lr(optimizer)

            total_loss_val = float(total_loss.detach().cpu())
            flat_logs = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in logs.items()}

            if (it + 1) % 50 == 0 or (it + 1) == steps_per_epoch:
                head = f"Epoch(train) [{epoch+1}][{it+1:5d}/{steps_per_epoch}]"
                fields = [
                    f"base_lr: {base_lr:.4e}",
                    f"lr: {lr_now:.4e}",
                    f"eta: {_format_eta(int(eta_seconds))}",
                    f"time: {iter_time_meter.val:.4f}",
                    f"data_time: {data_time_meter.val:.4f}",
                    f"memory: {mem_mb}",
                ]
                if grad_norm_value is not None:
                    fields.append(f"grad_norm: {grad_norm_value:.4f}")
                fields.append(f"loss: {total_loss_val:.4f}")
                for key in ("losses/loss_heatmap", "losses/loss_cls", "losses/loss_bbox",
                            "losses/matched_ious", "losses/loss_aux_img_bev"):
                    if key in flat_logs:
                        fields.append(f"{key.split('/')[-1]}: {float(flat_logs[key]):.4f}")
                print(head, "  ", "  ".join(fields))

            if use_wandb and (it % 10 == 0):
                import wandb
                wb = {
                    "train/iter_loss": total_loss_val,
                    "train/iter_time": iter_time_meter.val,
                    "train/data_time": data_time_meter.val,
                    "train/iter_time_avg": iter_time_meter.avg,
                    "train/data_time_avg": data_time_meter.avg,
                    "train/memory_mb": mem_mb,
                    "train/eta_sec": eta_seconds,
                    "lr": lr_now,
                    "base_lr": base_lr,
                    "epoch": epoch + (it + 1) / steps_per_epoch
                }
                if grad_norm_value is not None:
                    wb["train/grad_norm"] = grad_norm_value
                for k, v in flat_logs.items():
                    if isinstance(v, (int, float)):
                        wb[f"train/{k}"] = float(v)
                wandb.log(wb, step=global_step)

            last_iter_end = time.time()

        # save last
        save_state(os.path.join(args.work_dir, "last_state.pth"),
                   model_wrapper, optimizer, scheduler, scaler, epoch, best_val)

        if use_wandb and torch.cuda.is_available():
            import wandb
            wandb.log({"epoch/end_memory_mb": _gpu_memory_mb(), "epoch/idx": epoch + 1}, step=global_step)

        # 8) Per-epoch evaluation (direct, modular)
        if evaluator is not None and args.eval_interval > 0 and ((epoch + 1) % args.eval_interval == 0):
            eval_results = run_one_full_eval(
                evaluator, val_loader, model_wrapper, device, amp=args.fp16, max_eval_iters=getattr(args, "max_eval_iters", 0)
            )
            print(f"[Eval][epoch {epoch+1}] {eval_results}")

            # log to wandb
            if use_wandb:
                import wandb
                wandb.log({f"val/{k}": (float(v) if isinstance(v, (int,float)) else v) for k,v in eval_results.items()},
                          step=global_step)

            # optional: save best by a metric key
            best_key = getattr(args, "save_best_metric", None)
            if best_key and isinstance(eval_results.get(best_key, None), (int, float)):
                prev = best_metric_map.get(best_key, None)
                cur = float(eval_results[best_key])
                better = cur > prev if prev is not None else True
                if best_key.lower().startswith("loss"):
                    better = cur < prev if prev is not None else True
                if better:
                    best_metric_map[best_key] = cur
                    save_state(os.path.join(args.work_dir, f"best_by_{best_key}.pth"),
                               model_wrapper, optimizer, scheduler, scaler, epoch, cur)

    torch.save(model_wrapper.state_dict(), os.path.join(args.work_dir, "final_model_only.pth"))
    print("[Done] Training completed.")


if __name__ == "__main__":
    main()