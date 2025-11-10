# -*- coding: utf-8 -*-
"""
后端可选：
- mmdet3d：使用 mmengine/mmdet3d Registry 构建，forward(mode='loss')
- openpcdet：使用其模型，要求实现 get_training_loss 或类似 API（按项目适配）
- custom：用户自定义 nn.Module，需实现 forward(batch) 或 loss(batch)
本文件中完成第三方依赖的按需导入，train.py 不感知这些库。
"""
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass


# ---- 保存/加载状态接口约束 ----
class SaveLoadState:
    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        raise NotImplementedError

def _move_tensors_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_move_tensors_to_device(v, device) for v in obj]
        return type(obj)(out) if not isinstance(obj, list) else out
    return obj

def _tensorize_to_device(x, device, dtype=None):
    t = torch.as_tensor(x, device=device)
    if dtype is not None:
        t = t.to(dtype)
    return t

def _normalize_data_samples_on_device(samples, device):
    """确保 data_samples 中的 labels/bboxes 与模型设备一致；labels 为 long。"""
    if isinstance(samples, (list, tuple)):
        it = samples
    else:
        it = [samples]
    for s in it:
        if not hasattr(s, "to"):
            continue
        # 先整体 .to(device)（会递归多数张量）
        s.to(device)

        # 保险：确保 gt_instances* 里的 labels 是 long 且在 device；bboxes_3d 也迁设备
        for gi_name in ("gt_instances_3d", "gt_instances"):
            gi = getattr(s, gi_name, None)
            if gi is None:
                continue
            # labels
            for lab_name in ("labels_3d", "labels"):
                if hasattr(gi, lab_name):
                    lab = getattr(gi, lab_name)
                    if isinstance(lab, torch.Tensor):
                        if lab.device != device or lab.dtype != torch.long:
                            setattr(gi, lab_name, lab.to(device=device, dtype=torch.long, non_blocking=True))
                    else:
                        setattr(gi, lab_name, _tensorize_to_device(lab, device, torch.long))
            # bboxes_3d
            for bb_name in ("bboxes_3d", "bboxes"):
                if hasattr(gi, bb_name):
                    bb = getattr(gi, bb_name)
                    if hasattr(bb, "to"):
                        try:
                            setattr(gi, bb_name, bb.to(device))
                        except Exception:
                            pass
                        
def _move_data_samples_to_device(samples, device):
    # Det3DDataSample 实现了 .to()，可就地移动内部张量
    if isinstance(samples, (list, tuple)):
        for s in samples:
            if hasattr(s, "to"):
                s.to(device)
    else:
        if hasattr(samples, "to"):
            samples.to(device)

def _first_attr_no_bool(obj, names):
    """Return the first existing, non-None attribute from names without boolean coercion."""
    for n in names:
        if hasattr(obj, n):
            val = getattr(obj, n)
            # 允许 0 长度 / 全零 tensor，不能用 if val 判断
            if val is not None:
                return val
    return None

# ---- 通用包装器基类 ----
class BaseModelWrapper(nn.Module, SaveLoadState):
    def __init__(self):
        super().__init__()

    def unwrap(self) -> nn.Module:
        """返回底层真实模型以供优化器取参数或调试。"""
        return self

    def move_batch_to_device(self, batch, device):
        """把常见的 Tensor/Dict/List 迁移到 device。第三方特殊类型可在子类覆写。"""
        def to_dev(x):
            if isinstance(x, torch.Tensor):
                return x.to(device, non_blocking=True)
            if isinstance(x, list):
                return [to_dev(xx) for xx in x]
            if isinstance(x, dict):
                return {k: to_dev(v) for k, v in x.items()}
            return x
        return to_dev(batch)

    def forward_loss(self, batch, amp: bool, scaler=None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """返回 (total_loss, logs)。子类必须实现。"""
        raise NotImplementedError

    # SaveLoadState
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)


# =========================
# MMDetection3D 包装器
# =========================
class MMDet3DWrapper(BaseModelWrapper):
    def __init__(self, model_config: str, checkpoint: str = None, extra: Dict[str, Any] = None):
        super().__init__()
        extra = extra or {}

        # 延迟导入，避免 train.py 依赖
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmdet3d.registry import MODELS
        from mmengine.runner.checkpoint import load_checkpoint

        cfg = Config.fromfile(model_config)
        # 可选覆盖（从 extra 里拿）
        if "cfg_options" in extra and isinstance(extra["cfg_options"], dict):
            cfg.merge_from_dict(extra["cfg_options"])

        # scope
        default_scope = cfg.get("default_scope", "mmdet3d")
        init_default_scope(default_scope)

        # 构建模型，不加载权重
        self.model = MODELS.build(cfg.model)

        # 加权重（与构建分离）
        if checkpoint:
            load_checkpoint(self.model, checkpoint, map_location="cpu", strict=False)

        self.cfg = cfg

    def unwrap(self) -> nn.Module:
        return self.model

    def move_batch_to_device(self, batch, device):
        # mmdet3d 的 batch 通常是 dict，内含 inputs(可以是 list[Tensor]) 和 data_samples(List[Det3DDataSample])
        # data_samples 里对象不可直接 to(device)，模型内部会处理；只把 Tensor/列表 Tensor 放到 device。
        def to_dev(x):
            if isinstance(x, torch.Tensor):
                return x.to(device, non_blocking=True)
            if isinstance(x, list):
                # 保持 Det3DDataSample 原样
                if len(x) > 0 and x[0].__class__.__name__.endswith("DataSample"):
                    return x
                return [to_dev(xx) for xx in x]
            if isinstance(x, dict):
                return {k: to_dev(v) for k, v in x.items()}
            return x
        return to_dev(batch)

    @staticmethod
    def _scalarize_losses(losses: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = None
        logs = {}
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                val = v.mean()
                logs[k] = float(val.detach().cpu())
                if "loss" in k.lower():
                    total = val if total is None else total + val
            elif isinstance(v, (int, float)):
                logs[k] = float(v)
            elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
                s = torch.stack([t.mean() for t in v]).sum()
                logs[k] = float(s.detach().cpu())
                if "loss" in k.lower():
                    total = s if total is None else total + s
        if total is None:
            # 回退
            total = torch.tensor(0.0, device=next(self.parameters()).device)
        return total, logs
    
    def predict_step(self, batch, amp: bool = False):
        """返回 list[Det3DDataSample]，其中每个元素含 pred_instances_3d 等。"""
        # 兼容 dict / list[dict] 批次（与 forward_loss 同样的合并逻辑）
        def _merge_list_batch(list_batch):
            merged_inputs, merged_samples = {}, []
            for item in list_batch:
                ins = item.get("inputs", None); samp = item.get("data_samples", None)
                if ins is None or samp is None:
                    raise RuntimeError("Each item must contain 'inputs' and 'data_samples'.")
                if isinstance(ins, dict):
                    for k, v in ins.items():
                        merged_inputs.setdefault(k, []).append(v)
                else:
                    merged_inputs.setdefault("inputs", []).append(ins)
                merged_samples.extend(samp if isinstance(samp, (list, tuple)) else [samp])
            return merged_inputs, merged_samples

        if isinstance(batch, dict):
            inputs, data_samples = batch.get("inputs", None), batch.get("data_samples", None)
            if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict):
                inputs, data_samples = _merge_list_batch(
                    [{"inputs": i, "data_samples": s} for i, s in zip(inputs, data_samples)]
                )
        elif isinstance(batch, (list, tuple)):
            inputs, data_samples = _merge_list_batch(batch)
        else:
            raise RuntimeError(f"Unsupported batch type: {type(batch)}")

        # 预处理
        data_for_proc = {"inputs": inputs, "data_samples": data_samples}
        proc = self.model.data_preprocessor(data_for_proc, training=False)
        proc_inputs = proc.get("inputs", proc)
        proc_samples = proc.get("data_samples", data_samples)

        # 设备对齐（张量与 DataSample）
        device = next(self.model.parameters()).device

        def _move_tensors_to_device(obj, device):
            if isinstance(obj, torch.Tensor): return obj.to(device, non_blocking=True)
            if isinstance(obj, dict): return {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                xs = [_move_tensors_to_device(v, device) for v in obj]
                return type(obj)(xs) if not isinstance(obj, list) else xs
            return obj

        def _move_data_samples_to_device(samples, device):
            it = samples if isinstance(samples, (list, tuple)) else [samples]
            for s in it:
                if hasattr(s, "to"): s.to(device)

        proc_inputs = _move_tensors_to_device(proc_inputs, device)
        _move_data_samples_to_device(proc_samples, device)

        # 预测
        if amp:
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                preds = self.model(proc_inputs, proc_samples, mode="predict")
        else:
            preds = self.model(proc_inputs, proc_samples, mode="predict")
        # mmdet3d 返回 list[Det3DDataSample]
        return preds

    def forward_loss(self, batch, amp: bool, scaler=None):
        """
        支持两种 batch 形态：
        A) dict: {"inputs": {...}, "data_samples": [...]}
        B) list[dict], 每个元素类似 {"inputs": {...}, "data_samples": <Det3DDataSample>}
            —— 这是 pseudo_collate / smart_collate 常见输出
        我们把 B 聚合为 A 再喂给 mmdet3d 模型。
        """
        def _merge_list_batch(list_batch):
            merged_inputs = {}
            merged_samples = []
            for item in list_batch:
                # 每个 item 必须是 dict，含 "inputs" 和 "data_samples"
                ins = item.get("inputs", None)
                samp = item.get("data_samples", None)
                if ins is None or samp is None:
                    raise RuntimeError("Each item in the batch must contain 'inputs' and 'data_samples'.")
                # inputs 一般是 dict，比如 {'points': BasePoints 或 list[BasePoints], 'imgs': list[Tensor], ...}
                if isinstance(ins, dict):
                    for k, v in ins.items():
                        merged_inputs.setdefault(k, []).append(v)
                else:
                    # 若是张量/列表，统一放在 'inputs' 这个 key 下
                    merged_inputs.setdefault("inputs", []).append(ins)
                # data_samples：保证是 list[Det3DDataSample]
                if isinstance(samp, (list, tuple)):
                    merged_samples.extend(list(samp))
                else:
                    merged_samples.append(samp)
            return merged_inputs, merged_samples

        # 1) 整理 inputs / data_samples
        if isinstance(batch, dict):
            inputs = batch.get("inputs", None)
            data_samples = batch.get("data_samples", None)
            # 某些管道把 inputs 弄成 list[dict]（极少见），也做一次聚合
            if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict):
                inputs, data_samples = _merge_list_batch([{"inputs": i, "data_samples": s}
                                                        for i, s in zip(inputs, data_samples)])
        elif isinstance(batch, (list, tuple)):
            # pseudo_collate/smart_collate 常见形态：list[dict]
            inputs, data_samples = _merge_list_batch(batch)
        else:
            raise RuntimeError(f"Unsupported batch type: {type(batch)}")

        if inputs is None or data_samples is None:
            raise RuntimeError("Missing 'inputs' or 'data_samples' after collation/merge.")


        try:
            data_for_proc = {"inputs": inputs, "data_samples": data_samples}
            proc = self.model.data_preprocessor(data_for_proc, training=self.training)
            
            # mmdet3d 的 preprocessor 返回 dict，包含 'inputs' 和 'data_samples'
            proc_inputs = proc.get("inputs", proc)
            proc_samples = proc.get("data_samples", data_samples)
        except Exception as _e:
            print("[DEBUG] data_preprocessor pass failed:", repr(_e))
        
        device = next(self.model.parameters()).device
        proc_inputs = _move_tensors_to_device(proc_inputs, device)
        #_move_data_samples_to_device(proc_samples, device)
        _normalize_data_samples_on_device(proc_samples, device)
        
        if not hasattr(self, "_dev_checked"):
            gi = _first_attr_no_bool(proc_samples[0], ["gt_instances_3d", "gt_instances"])
            lbl = _first_attr_no_bool(gi, ["labels_3d", "labels"]) if gi is not None else None

            # 打印时也要避免触发布尔转换
            if lbl is not None and hasattr(lbl, "device"):
                print("[DBG] labels device/dtype:", lbl.device, getattr(lbl, "dtype", None))
            else:
                print("[DBG] labels missing or has no device attr:", type(lbl))

            vox = proc_inputs.get("voxels", None)
            if isinstance(vox, dict):
                any_vox = next((v for v in vox.values() if hasattr(v, "device")), None)
                print("[DBG] vox device:", getattr(any_vox, "device", None))
            self._dev_checked = True

        # 2) 前向 + loss
        if amp:
            from torch.amp import autocast
            with autocast(dtype=torch.float16):
                losses = self.model(proc_inputs, proc_samples, mode="loss")
        else:
            losses = self.model(proc_inputs, proc_samples, mode="loss")

        # 3) 汇总 loss
        total = None
        logs = {}
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                val = v.mean()
                logs[f"losses/{k}"] = float(val.detach().cpu())
                if "loss" in k.lower():
                    total = val if total is None else total + val
            elif isinstance(v, (int, float)):
                logs[f"losses/{k}"] = float(v)
            elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
                s = torch.stack([t.mean() for t in v]).sum()
                logs[f"losses/{k}"] = float(s.detach().cpu())
                if "loss" in k.lower():
                    total = s if total is None else total + s

        if total is None:
            # 异常兜底，避免 None 继续向下
            total = torch.tensor(0.0, device=next(self.parameters()).device)

        logs["loss"] = float(total.detach().cpu())
        return total, logs


# =========================
# OpenPCDet 包装器（示例适配）
# =========================
class OpenPCDetWrapper(BaseModelWrapper):
    """
    说明：OpenPCDet 项目中的训练接口可能随分支不同而略有差异。
    这里假设模型暴露 get_training_loss() 返回 loss_dict，或 forward(batch) 返回 loss_dict。
    可根据你的实际仓库改动该包装器内部几行即可。
    """
    def __init__(self, model_config: str, checkpoint: str = None, extra: Dict[str, Any] = None):
        super().__init__()
        extra = extra or {}

        try:
            from pcdet.config import cfg, cfg_from_yaml_file  # 典型 OpenPCDet 接口
            from pcdet.models import build_network
        except Exception as e:
            raise ImportError(
                "OpenPCDet is not installed or import failed. "
                "Please ensure your PYTHONPATH includes OpenPCDet."
            ) from e

        cfg_from_yaml_file(model_config, cfg)
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)

        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(ckpt["model_state"], strict=False)

        self.cfg = cfg

    def unwrap(self) -> nn.Module:
        return self.model

    def forward_loss(self, batch, amp: bool, scaler=None):
        # 假设 batch 已经是 OpenPCDet 风格的 dict，包含点云/标签等
        if amp:
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                ret_dict, tb_dict, disp_dict = self.model(batch)  # 有些分支返回这些
        else:
            ret_dict, tb_dict, disp_dict = self.model(batch)

        # 统一拿 loss
        if "loss" in ret_dict:
            total = ret_dict["loss"]
        elif "loss" in tb_dict:
            total = tb_dict["loss"]
        else:
            # 若需自己汇总，示例：
            total = sum(v.mean() for k, v in ret_dict.items() if "loss" in k.lower())

        logs = {}
        for d in (ret_dict, tb_dict, disp_dict):
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, torch.Tensor):
                        logs[f"losses/{k}"] = float(v.mean().detach().cpu())
                    elif isinstance(v, (int, float)):
                        logs[f"losses/{k}"] = float(v)
        logs["loss"] = float(total.detach().cpu())
        return total, logs


# =========================
# 自定义 PyTorch 模型包装器
# =========================
class CustomModelWrapper(BaseModelWrapper):
    """
    要求：
    - 你的模型（nn.Module）需能处理一个 batch（dict 或 tensor），返回：
      a) dict（含多个 loss* 项），或
      b) 一个标量 loss 张量。
    - 如果返回 dict，本包装器会将包含 'loss' 的键求和为总损失。
    extra 可传入 import 路径或构造 kwargs。
    """
    def __init__(self, model_config: str = None, checkpoint: str = None, extra: Dict[str, Any] = None):
        super().__init__()
        extra = extra or {}
        # 示例：通过模块路径 + 类名动态导入
        import importlib

        module_path = extra.get("module", None)  # e.g. myproj.models.my_net
        class_name = extra.get("class", None)    # e.g. MyNet
        model_kwargs = extra.get("model_kwargs", {})

        assert module_path and class_name, "For custom backend, provide extra.module and extra.class"
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        self.model = cls(**(model_kwargs if isinstance(model_kwargs, dict) else {}))

        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            # 允许 user 传完整 state 或仅权重
            self.model.load_state_dict(state.get("model", state), strict=False)

    def unwrap(self) -> nn.Module:
        return self.model

    def forward_loss(self, batch, amp: bool, scaler=None):
        if amp:
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                out = self.model(batch)
        else:
            out = self.model(batch)

        if isinstance(out, dict):
            total = None
            logs = {}
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    val = v.mean()
                    logs[f"losses/{k}"] = float(val.detach().cpu())
                    if "loss" in k.lower():
                        total = val if total is None else total + val
                elif isinstance(v, (int, float)):
                    logs[f"losses/{k}"] = float(v)
            if total is None:
                total = torch.tensor(0.0, device=next(self.parameters()).device)
        elif torch.is_tensor(out):
            total = out.mean()
            logs = {}
        else:
            raise RuntimeError("Custom model must return a loss tensor or a dict of losses.")

        logs["loss"] = float(total.detach().cpu())
        return total, logs


# ---- 构建工厂 ----
def build_model_wrapper(backend: str, model_config: str = None, checkpoint: str = None, extra: Dict[str, Any] = None) -> BaseModelWrapper:
    backend = backend.lower()
    if backend == "mmdet3d":
        return MMDet3DWrapper(model_config=model_config, checkpoint=checkpoint, extra=extra)
    elif backend == "openpcdet":
        return OpenPCDetWrapper(model_config=model_config, checkpoint=checkpoint, extra=extra)
    elif backend == "custom":
        return CustomModelWrapper(model_config=model_config, checkpoint=checkpoint, extra=extra)
    else:
        raise ValueError(f"Unknown model backend: {backend}")