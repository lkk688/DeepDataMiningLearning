# -*- coding: utf-8 -*-
"""
- mmdet3d：从 config 里读取 train/val dataloader.dataset 等并构建，再返回标准 torch DataLoader
- custom：用户自定义 Dataset（通过模块与类名动态导入）
"""
from typing import Dict, Any, Tuple, Optional
import torch
from torch.utils.data import DataLoader
import os
from copy import deepcopy

def _choose_collate_fn(name: Optional[str]):
    from mmengine.dataset import default_collate, pseudo_collate
    if not name:
        return default_collate
    name = name.lower()
    if name == "pseudo":
        return pseudo_collate
    return default_collate


def _to_int(v, default):
    try:
        return int(v)
    except Exception:
        return default


def _try_get_mmdet3d_data_root_from_install() -> str | None:
    """推断 mmdet3d 安装根下的 data 目录，例如 /path/to/mmdetection3d/data"""
    try:
        import mmdet3d
        pkg_dir = os.path.dirname(mmdet3d.__file__)
        # 常见结构：<repo>/mmdet3d/__init__.py → ROOT = dirname(pkg_dir)
        repo_root = os.path.abspath(os.path.join(pkg_dir, os.pardir))
        data_dir = os.path.join(repo_root, "data")
        return data_dir if os.path.isdir(data_dir) else None
    except Exception:
        return None


def _guess_dataset_subdir(ds_type: str) -> str | None:
    """根据数据集类型猜测默认子目录名。可按需扩展。"""
    t = (ds_type or "").lower()
    # 常见关键字匹配
    if "nuscene" in t:
        return "nuscenes"
    if "kitti" in t:
        return "kitti"
    if "waymo" in t:
        return "waymo"
    if "lyft" in t:
        return "lyft"
    if "scannet" in t or "scan" in t:
        return "scannet"
    if "sunrgbd" in t:
        return "sunrgbd"
    if "argo" in t:
        # 你的目录结构可能是 data/argo2/ 或 data/argo2/argo2
        return "argo2"
    if "s3dis" in t:
        return "s3dis"
    return None


def _patch_dataset_cfg_data_root(ds_cfg: dict, global_root: str | None, mmdet3d_data_root: str | None):
    """
    递归修改 dataset cfg 的 data_root 字段：
    优先使用用户传入的 global_root；否则尝试 mmdet3d 安装下的 data 目录。
    当子 cfg 没 data_root 或是相对/缺省时才覆盖；已有绝对路径则尊重不改。
    """
    if ds_cfg is None or not isinstance(ds_cfg, dict):
        return

    ds_type = ds_cfg.get("type", "")
    # 处理 RepeatDataset / ConcatDataset
    if ds_type in ("RepeatDataset", "ConcatDataset"):
        sub = ds_cfg.get("dataset") if ds_type == "RepeatDataset" else ds_cfg.get("datasets")
        if isinstance(sub, dict):
            _patch_dataset_cfg_data_root(sub, global_root, mmdet3d_data_root)
        elif isinstance(sub, (list, tuple)):
            for s in sub:
                _patch_dataset_cfg_data_root(s, global_root, mmdet3d_data_root)
        return

    # 普通单一数据集
    cur_root = ds_cfg.get("data_root", None)
    # 若已经是绝对路径，直接跳过
    if isinstance(cur_root, str) and os.path.isabs(cur_root) and os.path.isdir(cur_root):
        return

    # 选择基底根目录：用户 > 自动探测
    base_root = None
    if global_root and os.path.isdir(global_root):
        base_root = global_root
    elif mmdet3d_data_root and os.path.isdir(mmdet3d_data_root):
        base_root = mmdet3d_data_root

    if base_root is None:
        return  # 没找到可用根目录，保持原样

    # 推断子目录名
    subdir = _guess_dataset_subdir(ds_type)
    candidate = os.path.join(base_root, subdir) if subdir else base_root

    # 如果 config 里原本就写了相对路径（例如 'data/nuscenes'），也能兜底到绝对路径
    if isinstance(cur_root, str) and cur_root:
        # 相对路径 → 拼接到 base_root
        if not os.path.isabs(cur_root):
            cand2 = os.path.join(base_root, cur_root)
            if os.path.isdir(cand2):
                ds_cfg["data_root"] = cand2
                return

    # 否则试试 base_root/subdir
    if os.path.isdir(candidate):
        ds_cfg["data_root"] = candidate

# =========================
# MMDetection3D 数据加载
# =========================
def _default_mmdet3d_nus_root():
    """默认返回 <mmdetection3d repo>/data/nuscenes，如果不存在则返回 <repo>/data。"""
    import mmdet3d
    repo_dir = os.path.dirname(os.path.abspath(mmdet3d.__file__))  # .../mmdet3d
    repo_root = os.path.dirname(repo_dir)                          # one level up
    cand = os.path.join(repo_root, "data", "nuscenes")
    return cand if os.path.isdir(cand) else os.path.join(repo_root, "data")

from typing import Any, Dict, List
import torch
from torch.utils.data._utils.collate import default_collate as torch_default_collate

def smart_det3d_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Smart collate for mmdet3d batches:
    - Keep 'data_samples' as a flat list of Det3DDataSample.
    - For other keys, try torch default_collate; if it fails, keep as list.
    Works for batch item like: {"inputs": {...}, "data_samples": Det3DDataSample or list[Det3DDataSample]}
    """
    if not isinstance(batch, list) or not batch:
        return batch  # unusual

    if not isinstance(batch[0], dict):
        # fallback to torch's behavior
        try:
            return torch_default_collate(batch)
        except Exception:
            return batch

    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [item[k] for item in batch]

        if k == "data_samples":
            flat = []
            for v in vals:
                if isinstance(v, list):
                    flat.extend(v)
                else:
                    flat.append(v)
            out[k] = flat
            continue

        # Try to collate tensors/arrays/numbers/dicts/lists as usual
        try:
            out[k] = torch_default_collate(vals)
        except Exception:
            # Some nested dicts or custom objects may still fail → keep as list
            out[k] = vals

    return out

def _build_mmdet3d_loaders(
    data_config: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    extra: Dict[str, Any],
):
    # 延迟导入，避免硬依赖
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet3d.registry import DATASETS

    cfg = Config.fromfile(data_config)
    if isinstance(extra.get("cfg_options"), dict):
        cfg.merge_from_dict(extra["cfg_options"])
    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    # === 1) 解析“优先级最高”的 data_root：CLI/extra > cfg.dataset > 自动探测 ===
    user_root = extra.get("data_root", None)
    auto_pkg_root = _default_mmdet3d_nus_root()     # <mmdet3d repo>/data/nuscenes (或 /data)
    # 如果用户给到了上级 data/，自动补全到 data/nuscenes
    if user_root and os.path.basename(user_root.rstrip("/")) == "data":
        candidate = os.path.join(user_root, "nuscenes")
        if os.path.isdir(candidate):
            user_root = candidate

    # === 2) 覆盖 train/val/test 的 dataset.data_root & 相对路径字段 ===
    def _patch_dataset_cfg(ds_cfg: Dict[str, Any]):
        if not isinstance(ds_cfg, dict):
            return
        # 优先使用 user_root，否则 cfg 中已有的 data_root，再否则自动推断
        eff_root = user_root or ds_cfg.get("data_root") or auto_pkg_root
        ds_cfg["data_root"] = eff_root

        # 兼容 mmdet3d 新旧风格：有些相对路径放在 data_prefix 或 ann_file
        # - ann_file 是相对路径时，拼到 data_root 下
        if "ann_file" in ds_cfg and isinstance(ds_cfg["ann_file"], str):
            if not os.path.isabs(ds_cfg["ann_file"]):
                ds_cfg["ann_file"] = os.path.join(eff_root, ds_cfg["ann_file"])

        dp = ds_cfg.get("data_prefix")
        if isinstance(dp, dict):
            for k, v in list(dp.items()):
                if isinstance(v, str) and not os.path.isabs(v):
                    dp[k] = os.path.join(eff_root, v)
            ds_cfg["data_prefix"] = dp

        # 老配置里常见的 img_prefix / pts_prefix 等相对路径，也拼接一下
        for key in ("img_prefix", "pts_prefix", "seg_prefix"):
            if isinstance(ds_cfg.get(key), str) and not os.path.isabs(ds_cfg[key]):
                ds_cfg[key] = os.path.join(eff_root, ds_cfg[key])

        # 嵌套 dataset（如 ConcatDataset、RepeatDataset）
        if "datasets" in ds_cfg and isinstance(ds_cfg["datasets"], (list, tuple)):
            for sub in ds_cfg["datasets"]:
                _patch_dataset_cfg(sub)
        if "dataset" in ds_cfg and isinstance(ds_cfg["dataset"], dict):
            _patch_dataset_cfg(ds_cfg["dataset"])

    def _get_ds_cfg(split: str) -> Dict[str, Any]:
        if f"{split}_dataloader" in cfg and "dataset" in cfg[f"{split}_dataloader"]:
            return cfg[f"{split}_dataloader"]["dataset"]
        elif hasattr(cfg, "data") and split in cfg.data:
            return cfg.data[split]
        else:
            return {}

    for split in ("train", "val", "test"):
        ds_cfg = _get_ds_cfg(split)
        _patch_dataset_cfg(ds_cfg)

    # === 3) 实例化 dataset & dataloader ===
    def _choose_collate_fn(name_or_callable):
        # 自己的 collate 选择器；默认用 torch 的
        if callable(name_or_callable):
            return name_or_callable
        # 可根据字符串名切换不同 collate 策略
        return None  # 让 DataLoader 用默认 collate

    def build_ds(split: str):
        ds_cfg = _get_ds_cfg(split)
        assert ds_cfg, f"No dataset cfg for split={split}"
        return DATASETS.build(ds_cfg)

    def build_dl(dataset, split: str):
        dl_cfg = cfg.get(f"{split}_dataloader", {})
        # 用户若在 config 里显式指定了 collate_fn，我们尊重；否则用 smart
        collate_name = dl_cfg.get("collate_fn", None)

        if callable(collate_name):
            collate_fn = collate_name
        elif isinstance(collate_name, str) and collate_name.lower() == "default":
            from mmengine.dataset.utils import default_collate as mm_default_collate
            collate_fn = mm_default_collate  # 也能处理部分结构，但可能仍拼不动 DataSample
        elif isinstance(collate_name, str) and collate_name.lower() == "pseudo":
            from mmengine.dataset.utils import pseudo_collate as mm_pseudo_collate
            collate_fn = mm_pseudo_collate    # 直接把 batch 包成 list（最保守）
        else:
            # 默认：用我们定制的 smart collate，兼顾易用与鲁棒
            collate_fn = smart_det3d_collate

        shuffle = (split == "train")
        drop_last = dl_cfg.get("drop_last", shuffle)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
        )

    train_ds = build_ds("train")
    val_loader = None
    try:
        val_ds = build_ds("val")
        val_loader = build_dl(val_ds, "val")
    except Exception:
        val_loader = None

    train_loader = build_dl(train_ds, "train")
    return train_loader, val_loader


# =========================
# 自定义数据集加载
# =========================
def _build_custom_loaders(data_config: str, batch_size: int, num_workers: int, pin_memory: bool, extra: Dict[str, Any]):
    """
    要求 extra 里提供：
    - dataset_module: 例如 myproj.data.my_dataset
    - train_class / val_class: 类名（可同一个）
    - dataset_kwargs_train / dataset_kwargs_val（可选 dict）
    - collate (可选): "default" / "pseudo"；若不使用 mmengine 的 pseudo，请提供自定义 collate callable
    """
    import importlib
    from mmengine.dataset import default_collate, pseudo_collate  # 若不想依赖，可自己写个简单 pseudo
    mod = importlib.import_module(extra.get("dataset_module"))
    train_cls = getattr(mod, extra.get("train_class"))
    val_cls_name = extra.get("val_class", extra.get("train_class"))
    val_cls = getattr(mod, val_cls_name)

    ds_train = train_cls(**(extra.get("dataset_kwargs_train", {}) or {}))
    dl_collate = extra.get("collate", "default")
    if dl_collate == "pseudo":
        collate_fn = pseudo_collate
    elif callable(dl_collate):
        collate_fn = dl_collate
    else:
        collate_fn = default_collate

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
    )

    val_loader = None
    if extra.get("have_val", True):
        ds_val = val_cls(**(extra.get("dataset_kwargs_val", {}) or {}))
        val_loader = DataLoader(
            ds_val,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
        )

    return train_loader, val_loader


# ---- 工厂函数 ----
def build_dataloaders(
    backend: str,
    data_config: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    extra: Dict[str, Any],
):
    backend = backend.lower()
    if backend == "mmdet3d":
        return _build_mmdet3d_loaders(data_config, batch_size, num_workers, pin_memory, extra)
    elif backend == "custom":
        return _build_custom_loaders(data_config, batch_size, num_workers, pin_memory, extra)
    else:
        raise ValueError(f"Unknown dataset backend: {backend}")


def build_evaluator(backend: str, data_config: str, split: str, extra: Dict[str, Any]):
    """
    返回 (evaluator, meta)，其中 evaluator 具有 .process(data_batch, data_samples) 与 .evaluate(size)。
    对 mmdet3d：从 config 里取 {split}_evaluator 或 test_evaluator。
    """
    backend = backend.lower()
    if backend != "mmdet3d":
        return None, {}

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.evaluator import Evaluator
    from mmdet3d.registry import METRICS

    cfg = Config.fromfile(data_config)
    if "cfg_options" in extra and isinstance(extra["cfg_options"], dict):
        cfg.merge_from_dict(extra["cfg_options"])
    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    ev_cfg = None
    key = f"{split}_evaluator"
    if key in cfg:
        ev_cfg = cfg[key]
    elif "test_evaluator" in cfg and split in ("val", "test"):
        ev_cfg = cfg["test_evaluator"]
    elif "val_evaluator" in cfg and split == "val":
        ev_cfg = cfg["val_evaluator"]
    assert ev_cfg is not None, f"No evaluator cfg found for split={split}"

    # 允许传列表或单个 metric
    metric_cfgs = ev_cfg if isinstance(ev_cfg, (list, tuple)) else [ev_cfg]
    metrics = [METRICS.build(mc) for mc in metric_cfgs]
    evaluator = Evaluator(metrics=metrics)
    return evaluator, {"metric_names": [m.__class__.__name__ for m in metrics]}