from __future__ import annotations
"""
Dataset + DataLoader builders for MMDetection3D with robust path handling.
- Works outside the mmdetection3d repo root.
- Normalizes data_root, ann_file, data_prefix for common 3D datasets.
- Avoids version-fragile mmengine build_dataloader; uses torch DataLoader + DATA_SAMPLERS.
- Provides a smart collate that preserves Det3DDataSample lists.
- Exposes build_dataloaders(...) and build_evaluator(...) for convenience.
"""
from typing import Any, Dict, List, Optional, Tuple
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate as torch_default_collate

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.registry import DATA_SAMPLERS
from mmengine.evaluator import Evaluator

from mmdet3d.registry import DATASETS, METRICS, MODELS


# ------------------------------
# Registry priming (transforms/pipelines) — comments expanded
# ------------------------------

def _import_openmmlab_modules() -> None:
    """Eager-import common OpenMMLab modules so their classes register with
    the global registries. This prevents errors like:
    KeyError: 'LoadPointsFromFile' is not in the ...::transform registry.
    """
    try:
        import mmdet3d  # noqa: F401
        try:
            import mmdet3d.datasets.transforms  # noqa: F401
        except Exception:
            try:
                import mmdet3d.datasets.pipelines  # noqa: F401
            except Exception:
                pass
        # Some configs also reference 2D mmdet transforms
        try:
            import mmdet.datasets.transforms  # noqa: F401
        except Exception:
            try:
                import mmdet.datasets.pipelines  # noqa: F401
            except Exception:
                pass
    except Exception:
        # If mmdet3d itself isn't importable, the subsequent build will fail
        # with more context. We don't raise here to keep the root cause visible.
        pass


# ------------------------------
# Path helpers
# ------------------------------

def _patch_db_sampler_paths(db_sampler: Dict[str, Any], eff_root: str, ds_type: str | None) -> None:
    """Normalize DBSampler's paths (used by ObjectSample-like transforms).

    - db_sampler.data_root: join under dataset root if relative.
    - db_sampler.info_path: str | list[str], join safely under dataset root.
    """
    if not isinstance(db_sampler, dict) or not eff_root:
        return
    # data_root inside sampler
    dr = db_sampler.get('data_root')
    if isinstance(dr, str) and dr:
        db_sampler['data_root'] = _safe_join_under(eff_root, dr, ds_type)
    # info_path string or list
    ip = db_sampler.get('info_path')
    if isinstance(ip, str):
        db_sampler['info_path'] = _safe_join_under(eff_root, ip, ds_type)
    elif isinstance(ip, (list, tuple)):
        db_sampler['info_path'] = [
            _safe_join_under(eff_root, x, ds_type) if isinstance(x, str) else x
            for x in ip
        ]


def _patch_pipeline_paths(pipeline: Any, eff_root: str, ds_type: str | None) -> None:
    """Traverse a pipeline list and normalize any transform-level paths.

    Currently handles ObjectSample-like transforms that embed a ``db_sampler``
    with its own data_root/info_path. Safe no-op for unrelated transforms.
    """
    if not isinstance(pipeline, (list, tuple)):
        return
    for t in pipeline:
        if not isinstance(t, dict):
            continue
        # Common object sampling transforms in mmdet3d
        ttype = (t.get('type') or '').lower()
        if 'objectsample' in ttype or 'objectpaste' in ttype:
            _patch_db_sampler_paths(t.get('db_sampler', {}), eff_root, ds_type)


def _guess_dataset_subdir(ds_type: str | None) -> str | None:
    t = (ds_type or "").lower()
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
        return "argo2"
    if "s3dis" in t:
        return "s3dis"
    return None


def _default_mmdet3d_root_for(ds_type: str | None) -> str | None:
    """Return <mmdet3d_repo>/data/<subdir> if it exists, else <repo>/data, else None."""
    try:
        import mmdet3d
        repo_dir = os.path.dirname(os.path.abspath(mmdet3d.__file__))
        repo_root = os.path.dirname(repo_dir)
        data_root = os.path.join(repo_root, "data")
        if not os.path.isdir(data_root):
            return None
        sub = _guess_dataset_subdir(ds_type)
        if sub and os.path.isdir(os.path.join(data_root, sub)):
            return os.path.join(data_root, sub)
        return data_root
    except Exception:
        return None


def _safe_join_under(base_dir: str, maybe_rel: str, ds_type: str | None) -> str:
    """Join a possibly-relative path under base_dir without duplicating prefixes.
    - Strips leading 'data/' if present.
    - Avoids base_dir/<ds>/<ds>/... when maybe_rel already starts with the ds name.
    """
    if os.path.isabs(maybe_rel):
        return maybe_rel

    rel = maybe_rel.replace("\\", "/")
    if rel.startswith("data/"):
        rel = rel[5:]

    ds_key = _guess_dataset_subdir(ds_type or "") or ""
    if ds_key and rel.startswith(ds_key + "/") and os.path.basename(base_dir.rstrip("/")) == ds_key:
        rel = rel[len(ds_key) + 1:]

    return os.path.normpath(os.path.join(base_dir, rel))


# ------------------------------
# Collate helpers
# ------------------------------

def smart_det3d_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Smart collate for mmdet3d batches.
    - Keep 'data_samples' as a flat list of Det3DDataSample.
    - For other keys, try torch default_collate; if it fails, keep as list.
    Works for batch item like: {"inputs": {...}, "data_samples": Det3DDataSample or list[Det3DDataSample]}
    """
    if not isinstance(batch, list) or not batch:
        return batch  # unusual

    if not isinstance(batch[0], dict):
        try:
            return torch_default_collate(batch)
        except Exception:
            return batch

    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [item[k] for item in batch]

        if k == "data_samples":
            flat: List[Any] = []
            for v in vals:
                if isinstance(v, list):
                    flat.extend(v)
                else:
                    flat.append(v)
            out[k] = flat
            continue

        try:
            out[k] = torch_default_collate(vals)
        except Exception:
            out[k] = vals

    return out


# ------------------------------
# Internal config helpers
# ------------------------------

def _pick_loader_cfg(cfg: Config, split: str) -> Dict[str, Any]:
    # OpenMMLab 3.x style dataloader cfgs
    key = f"{split}_dataloader"
    if key in cfg:
        dl = cfg[key]
        if isinstance(dl, dict):
            return dl
    # Older style
    if hasattr(cfg, "data") and isinstance(cfg.data, dict) and split in cfg.data:
        return {"dataset": cfg.data[split], "batch_size": 1, "num_workers": 0}
    raise KeyError(f"No dataloader config for split={split}")


def _get_ds_cfg(cfg: Config, split: str) -> Dict[str, Any]:
    dl = _pick_loader_cfg(cfg, split)
    if "dataset" in dl:
        return dl["dataset"]
    return dl


# ------------------------------
# Public builders
# ------------------------------

def build_dataloaders(
    # Build train/val loaders from a config using the selected backend.

    backend: str,
    data_config: str,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train/val (or train/test) dataloaders.

    Args:
        backend: currently only 'mmdet3d' and 'custom' supported.
        data_config: path to a mmdet3d config file, or a custom module spec.
        batch_size, num_workers, pin_memory: DataLoader params.
        extra: backend-specific extras.
            For 'mmdet3d': { 'data_root': "/abs/path/to/dataset",
                             'cfg_options': {...} }
            For 'custom' : { 'dataset_module': 'my.package',
                             'train_class': 'MyTrainDS', 'val_class': 'MyValDS',
                             'dataset_kwargs_train': {...}, 'dataset_kwargs_val': {...},
                             'collate': 'default'|'pseudo'|callable }
    Returns:
        (train_loader, val_loader_or_None)
    """
    backend = (backend or "mmdet3d").lower()
    extra = extra or {}

    if backend == "custom":
        return _build_custom_loaders(data_config, batch_size, num_workers, pin_memory, extra)
    if backend != "mmdet3d":
        raise ValueError(f"Unknown dataset backend: {backend}")

    # ---- mmdet3d path ----
    cfg = Config.fromfile(data_config)
    if isinstance(extra.get("cfg_options"), dict):
        cfg.merge_from_dict(extra["cfg_options"])  # runtime overrides

    init_default_scope(cfg.get("default_scope", "mmdet3d"))
    _import_openmmlab_modules()

    # Normalize dataset roots and relative fields across splits
    def _patch_dataset_cfg(ds_cfg: Dict[str, Any]):
        """Normalize dataset config paths in-place.

        - Recurse into wrapper datasets (Repeat/Concat/ClassBalanced) and patch only their children.
        - For regular datasets, compute an effective data_root and normalize
          ann_file/data_prefix and **pipeline-embedded** db_sampler paths.
        """
        if not isinstance(ds_cfg, dict):
            return

        ds_type = ds_cfg.get('type', '')

        # Wrapper datasets: patch inner dataset(s) only
        if ds_type in ('RepeatDataset', 'ConcatDataset', 'ClassBalancedDataset'):
            sub = ds_cfg.get('dataset') if ds_type != 'ConcatDataset' else ds_cfg.get('datasets')
            if isinstance(sub, dict):
                _patch_dataset_cfg(sub)
            elif isinstance(sub, (list, tuple)):
                for s in sub:
                    _patch_dataset_cfg(s)
            return

        # Regular dataset
        user_root = extra.get('data_root')
        cur_root = ds_cfg.get('data_root')
        if isinstance(cur_root, str) and os.path.isabs(cur_root) and os.path.isdir(cur_root):
            eff_root = cur_root
        else:
            auto_root = _default_mmdet3d_root_for(ds_type)
            eff_root = user_root or auto_root or cur_root or ''
        if eff_root:
            ds_cfg['data_root'] = eff_root

        # ann_file
        if 'ann_file' in ds_cfg and eff_root:
            ann = ds_cfg['ann_file']
            if isinstance(ann, str):
                ds_cfg['ann_file'] = _safe_join_under(eff_root, ann, ds_type)
            elif isinstance(ann, (list, tuple)):
                ds_cfg['ann_file'] = [
                    _safe_join_under(eff_root, a, ds_type) if isinstance(a, str) else a
                    for a in ann
                ]

        # data_prefix
        dp = ds_cfg.get('data_prefix')
        if isinstance(dp, dict) and eff_root:
            new_dp = {}
            for k, v in dp.items():
                if isinstance(v, str) and not os.path.isabs(v):
                    new_dp[k] = _safe_join_under(eff_root, v, ds_type)
                else:
                    new_dp[k] = v
            ds_cfg['data_prefix'] = new_dp

        # legacy prefixes
        for key in ('img_prefix', 'pts_prefix', 'seg_prefix'):
            v = ds_cfg.get(key)
            if isinstance(v, str) and eff_root and not os.path.isabs(v):
                ds_cfg[key] = _safe_join_under(eff_root, v, ds_type)

        # pipeline-level (train/eval) transforms that embed their own paths
        if 'pipeline' in ds_cfg and eff_root:
            _patch_pipeline_paths(ds_cfg['pipeline'], eff_root, ds_type)

    for split in ('train', 'val', 'test'):
        try:
            ds_cfg = _get_ds_cfg(cfg, split)
        except KeyError:
            continue
        _patch_dataset_cfg(ds_cfg)

    # Build datasets and loaders
    def _resolve_collate_fn(dl_cfg: Dict[str, Any]):
        from mmengine.dataset.utils import default_collate as mm_default_collate
        from mmengine.dataset.utils import pseudo_collate as mm_pseudo_collate
        collate_opt = dl_cfg.get("collate_fn")
        if callable(collate_opt):
            return collate_opt
        if isinstance(collate_opt, str):
            t = collate_opt.lower()
            if t in ("default", "default_collate"):
                return mm_default_collate
            if t in ("pseudo", "pseudo_collate"):
                return mm_pseudo_collate
        return smart_det3d_collate

    def build_one(split: str) -> Tuple[Any, Optional[DataLoader]]:
        try:
            dl_cfg = _pick_loader_cfg(cfg, split)
        except KeyError:
            return None, None

        dataset = DATASETS.build(_get_ds_cfg(cfg, split))

        # Sampler
        sampler_cfg = dict(dl_cfg.get("sampler", {}))
        if "type" not in sampler_cfg:
            sampler_cfg["type"] = "DefaultSampler"
        if "shuffle" not in sampler_cfg:
            sampler_cfg["shuffle"] = (split == "train")
        sampler = DATA_SAMPLERS.build({**sampler_cfg, "dataset": dataset})

        # DataLoader kwargs
        bs = int(dl_cfg.get("batch_size", 1))
        nw = int(dl_cfg.get("num_workers", 0))
        pin = bool(dl_cfg.get("pin_memory", False))
        drop = bool(dl_cfg.get("drop_last", split == "train"))
        persistent_workers = (nw > 0) and bool(dl_cfg.get("persistent_workers", False))
        prefetch_factor = dl_cfg.get("prefetch_factor", None)
        collate_fn = _resolve_collate_fn(dl_cfg)

        dl_kwargs = dict(
            dataset=dataset,
            batch_size=bs,
            num_workers=nw,
            sampler=sampler,
            pin_memory=pin,
            drop_last=drop,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor

        return dataset, DataLoader(**dl_kwargs)

    train_ds, train_loader = build_one("train")
    _, val_loader = build_one("val")
    if val_loader is None:
        _, val_loader = build_one("test")

    assert train_loader is not None, "Failed to build train dataloader"
    return train_loader, val_loader


# ------------------------------
# Custom backend (optional)
# ------------------------------

def _build_custom_loaders(
    data_config: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    extra: Dict[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build loaders for user-defined datasets via import strings.
    extra must contain:
      - dataset_module: e.g., 'myproj.data.my_dataset'
      - train_class / val_class: class names
      - dataset_kwargs_train / dataset_kwargs_val (optional dicts)
      - collate: 'default'|'pseudo'|callable (optional)
    """
    import importlib
    from mmengine.dataset.utils import default_collate as mm_default_collate
    from mmengine.dataset.utils import pseudo_collate as mm_pseudo_collate

    mod = importlib.import_module(extra.get("dataset_module"))
    train_cls = getattr(mod, extra.get("train_class"))
    val_cls_name = extra.get("val_class", extra.get("train_class"))
    val_cls = getattr(mod, val_cls_name)

    ds_train = train_cls(**(extra.get("dataset_kwargs_train", {}) or {}))

    dl_collate = extra.get("collate", "default")
    if dl_collate == "pseudo":
        collate_fn = mm_pseudo_collate
    elif callable(dl_collate):
        collate_fn = dl_collate
    else:
        collate_fn = mm_default_collate

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


# ------------------------------
# Evaluator (from config)
# ------------------------------

def build_evaluator(
    # Build an mmengine Evaluator from the config for the requested split.
backend: str, data_config: str, split: str, extra: Optional[Dict[str, Any]] = None) -> Tuple[Evaluator, Dict[str, Any]]:
    """Build an mmengine.Evaluator from the config for the given split.

    Returns (evaluator, meta) where meta includes the metric class names used.
    """
    backend = (backend or "mmdet3d").lower()
    if backend != "mmdet3d":
        return None, {}

    cfg = Config.fromfile(data_config)
    if extra and "cfg_options" in extra and isinstance(extra["cfg_options"], dict):
        cfg.merge_from_dict(extra["cfg_options"])  # runtime overrides

    init_default_scope(cfg.get("default_scope", "mmdet3d"))
    _import_openmmlab_modules()

    ev_cfg = None
    key = f"{split}_evaluator"
    if key in cfg:
        ev_cfg = cfg[key]
    elif "test_evaluator" in cfg and split in ("val", "test"):
        ev_cfg = cfg["test_evaluator"]
    elif "val_evaluator" in cfg and split == "val":
        ev_cfg = cfg["val_evaluator"]
    assert ev_cfg is not None, f"No evaluator cfg found for split={split}"

    metric_cfgs = ev_cfg if isinstance(ev_cfg, (list, tuple)) else [ev_cfg]
    metrics = [METRICS.build(mc) for mc in metric_cfgs]
    evaluator = Evaluator(metrics=metrics)
    return evaluator, {"metric_names": [m.__class__.__name__ for m in metrics]}


# ------------------------------
# Simple one-shot evaluation runner (NuScenes/KITTI)
# ------------------------------
# ---- Prediction normalization helpers for mmdet3d metrics ----
from mmengine.structures import InstanceData
from typing import List, Any, Dict, Optional
import torch

def _extract_gt_samples_strict(batch: Any) -> List[Any]:
    """Extract ground-truth Det3DDataSample list from a val/test batch."""
    if isinstance(batch, dict):
        if "data_samples" not in batch:
            raise RuntimeError("val/test batch missing key 'data_samples'")
        ds = batch["data_samples"]
        return ds if isinstance(ds, list) else [ds]
    if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], dict):
        out: List[Any] = []
        for i, item in enumerate(batch):
            if "data_samples" not in item:
                raise RuntimeError(f"val/test batch item[{i}] missing key 'data_samples'")
            ds = item["data_samples"]
            out.extend(ds if isinstance(ds, list) else [ds])
        return out
    raise RuntimeError(f"Unsupported val/test batch type for GT extraction: {type(batch)}")

def _empty_instancedata(device: torch.device) -> InstanceData:
    """Create an empty InstanceData with minimal 3D fields."""
    idata = InstanceData()
    idata.scores_3d = torch.empty(0, device=device)
    idata.labels_3d = torch.empty(0, dtype=torch.long, device=device)
    return idata

def _resolve_sample_idx(src: Any, fallback: Any = None) -> Optional[int]:
    """Try several common fields to resolve integer sample index."""
    def _from_meta(meta: Dict[str, Any]) -> Optional[int]:
        for k in ("sample_idx", "image_id", "img_id", "frame_id", "sample_id"):
            if k in meta and meta[k] is not None:
                try:
                    return int(meta[k])
                except Exception:
                    pass
        return None
    meta = getattr(src, "metainfo", None)
    if isinstance(meta, dict):
        si = _from_meta(meta)
        if si is not None:
            return si
    for k in ("sample_idx", "image_id", "img_id", "frame_id", "sample_id"):
        if hasattr(src, k):
            try:
                return int(getattr(src, k))
            except Exception:
                pass
    if fallback is not None:
        return _resolve_sample_idx(fallback, None)
    return None

# --- Instantiate metric with only the kwargs its __init__ accepts ---
import inspect

def _safe_instantiate_metric(metric_cls, **kwargs):
    """
    Create metric_cls(...) but only pass keyword args that appear in
    the class' __init__ signature. This makes the code robust across
    MMDet3D versions where init params differ (e.g., KITTI).
    """
    sig = inspect.signature(metric_cls.__init__)
    allowed = {k for k in sig.parameters.keys() if k != "self"}
    filtered = {k: v for k, v in kwargs.items() if (k in allowed and v is not None)}
    return metric_cls(**filtered)

def _build_pred_records(
    preds: Any,
    gt_samples: List[Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Normalize model outputs into list[dict] items required by mmdet3d metrics.
    Each dict contains:
      - 'pred_instances_3d': InstanceData with bboxes_3d/scores_3d/labels_3d (or empty)
      - 'pred_instances':    InstanceData for 2D (optional; empty if absent)
      - 'sample_idx':        int (taken from pred metainfo or GT fallback)
    """
    # Unwrap tuple outputs like (preds, aux)
    if isinstance(preds, tuple):
        preds = preds[0]
    # Ensure a flat list
    preds = list(preds) if isinstance(preds, (list, tuple)) else [preds]
    if len(preds) == 1 and isinstance(preds[0], (list, tuple)):
        preds = list(preds[0])

    records: List[Dict[str, Any]] = []
    for i, p in enumerate(preds):
        # 3D preds: attribute on DataSample or dict key
        pred3d = getattr(p, "pred_instances_3d", None)
        if pred3d is None and isinstance(p, dict):
            pred3d = p.get("pred_instances_3d", None)
        if pred3d is None:
            pred3d = _empty_instancedata(device)

        # Optional 2D preds
        pred2d = getattr(p, "pred_instances", None)
        if pred2d is None and isinstance(p, dict):
            pred2d = p.get("pred_instances", None)
        if pred2d is None:
            pred2d = InstanceData()

        # sample_idx from pred or fallback to aligned GT
        sample_idx = None
        meta = getattr(p, "metainfo", None)
        if isinstance(meta, dict):
            sample_idx = meta.get("sample_idx", None)
        if sample_idx is None:
            gi = gt_samples[min(i, len(gt_samples) - 1)]
            sample_idx = _resolve_sample_idx(p, fallback=gi)
        if sample_idx is None:
            raise RuntimeError(f"Cannot resolve sample_idx for pred #{i}")

        records.append(
            {
                "pred_instances_3d": pred3d,
                "pred_instances": pred2d,
                "sample_idx": int(sample_idx),
                "img_id": int(sample_idx),   # <-- some KITTI configs look for img_id
            }
        )
        # records.append(
        #     {
        #         "pred_instances_3d": pred3d,
        #         "pred_instances": pred2d,
        #         "sample_idx": int(sample_idx),
        #     }
        # )

    if not records or not isinstance(records[0], dict):
        raise TypeError("BUG: pred_records must be non-empty list[dict].")
    return records

def run_evaluation(
    *,
    cfg_path: str,
    checkpoint: Optional[str] = None,
    split: str = "val",
    device: str | torch.device = "cuda",
    backend: str = "mmdet3d",
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    extra: Optional[Dict[str, Any]] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Run evaluation using the config-defined metric (NuScenes/KITTI/etc.).

    This function wires together **existing logic** in this module without
    altering it: we build dataloaders via :func:`build_dataloaders`, build an
    mmengine :class:`Evaluator` via :func:`build_evaluator`, run the model on
    the selected split, call ``evaluator.process(...)`` per batch, and finally
    return ``evaluator.evaluate(n_samples)``.

    Args:
        cfg_path: Path to the mmdet3d config.
        checkpoint: Optional checkpoint to load into the model.
        split: Which split to evaluate: "val" (default) or "test".
        device: Torch device for model execution.
        backend: Dataset backend, defaults to "mmdet3d".
        batch_size: Optional override for dataloader batch size (used only when
            the config doesn't specify). To strictly keep config behavior,
            leave at 1 (default) and configure in the cfg file instead.
        num_workers: Same note as ``batch_size``.
        pin_memory: Same note as ``batch_size``.
        extra: Extra dict forwarded to builders. For mmdet3d you can pass
            ``{"data_root": "/abs/path/to/dataset", "cfg_options": {...}}``.
        max_batches: Stop after this many batches (quick smoke test).

    Returns:
        A dictionary of computed metrics (e.g., NDS/mAP for nuScenes, AP for KITTI).
    """
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint

    extra = extra or {}

    # Prime registries and default scope **without changing your build logic**.
    init_default_scope("mmdet3d")
    _import_openmmlab_modules()

    # Load full config, then build train/val loaders per your existing builders.
    cfg = Config.fromfile(cfg_path)
    if isinstance(extra.get("cfg_options"), dict):
        cfg.merge_from_dict(extra["cfg_options"])  # runtime overrides

    # Respect user's config for loader hyperparams by default.
    # These params are only used by our builder if the cfg lacks them.
    train_loader, val_loader = build_dataloaders(
        backend=backend,
        data_config=cfg_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        extra=extra,
    )

    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    target_loader = val_loader if val_loader is not None else train_loader

    evaluator, meta = build_evaluator(
        backend=backend,
        data_config=cfg_path,
        split=split,
        extra=extra,
    )

    # Build model from cfg and (optionally) load checkpoint.
    model = MODELS.build(cfg.model)
    model.to(device)  # type: ignore
    model.eval()
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location="cpu")

    # Iterate over the loader and feed evaluator
    seen = 0
    with torch.inference_mode():
        # choose device for empty InstanceData creation
        dev = torch.device(device if isinstance(device, str) else device)
        for batch in target_loader:
            # In MMDet3D, test_step returns list[Det3DDataSample] predictions.
            out = model.test_step(batch)
            
            # 1) Normalize the raw model output to a flat list
            if isinstance(out, tuple):
                out = out[0]
            preds = list(out) if isinstance(out, (list, tuple)) else [out]
            if len(preds) == 1 and isinstance(preds[0], (list, tuple)):
                preds = list(preds[0])
            
            # Extract GT list from data_batch for the metric
            gt_samples = _extract_gt_samples_strict(batch)

            # 3) Decide the format based on metric type(s)
            metric_names = [m.__class__.__name__.lower() for m in evaluator.metrics]
            #expects_nuscenes_style = any("nuscene" in n for n in metric_names)
            # Both NuScenes and KITTI want dict-style in mmdet3d’s current metrics
            expects_dict_style = any(x in n for n in metric_names for x in ("nuscene", "kitti"))
            
            
            if expects_dict_style:
                pred_records = _build_pred_records(preds, gt_samples, dev)  # dev = torch.device(...)
                evaluator.process({"data_samples": gt_samples}, pred_records)
            else:
                evaluator.process(batch, preds)

            seen += 1
            if max_batches is not None and seen >= max_batches:
                break

    # Return computed metrics.
    return evaluator.evaluate(seen)

# ==============================
# Direct KITTI / nuScenes eval (bypass mmengine.Evaluator)
# ==============================
from typing import Optional, Dict, Any, List, Union
import torch

def run_evaluation_direct(
    *,
    cfg_path: str,
    checkpoint: Optional[str] = None,
    split: str = "val",                      # "val" or "test"
    device: Union[str, torch.device] = "cuda",
    backend: str = "mmdet3d",
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    dataset_hint: str = "auto",              # "auto" | "kitti" | "nuscenes"
    extra: Optional[Dict[str, Any]] = None,  # e.g., {"data_root": "/abs/path/to/data/kitti", "cfg_options": {...}}
    jsonfile_prefix: Optional[str] = None,   # used by NuScenesMetric/KittiMetric if you want result dumps
    eval_version: str = "detection_cvpr_2019"  # nuScenes eval version
) -> Dict[str, Any]:
    """
    Evaluate KITTI/nuScenes by calling the metric's .process(...) and .evaluate(...)
    **directly** (no mmengine.Evaluator in the loop).

    Reuses your existing builders; does not change their logic.
    """

    # --- local lazy imports to avoid hard deps at module import time ---
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmdet3d.registry import MODELS, DATASETS

    # Try to import metrics (whichever you have installed)
    try:
        from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric  # type: ignore
    except Exception:
        KittiMetric = None  # type: ignore
    try:
        from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric  # type: ignore
    except Exception:
        NuScenesMetric = None  # type: ignore

    # ---- prime scope/registries and build config ----
    extra = extra or {}
    init_default_scope("mmdet3d")
    cfg = Config.fromfile(cfg_path)
    if isinstance(extra.get("cfg_options"), dict):
        cfg.merge_from_dict(extra["cfg_options"])

    # ---- build loaders via your existing logic (unchanged) ----
    train_loader, val_loader = build_dataloaders(
        backend=backend,
        data_config=cfg_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        extra=extra,
    )
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    data_loader = val_loader if val_loader is not None else train_loader
    if data_loader is None:
        raise RuntimeError(f"No '{split}' loader available from config.")

    # Grab the underlying dataset to infer meta/paths
    dataset_obj = getattr(data_loader, "dataset", None)
    dataset_meta = {}
    if hasattr(dataset_obj, "metainfo") and isinstance(dataset_obj.metainfo, dict):
        dataset_meta.update(dataset_obj.metainfo)
    if not dataset_meta.get("classes") and hasattr(dataset_obj, "CLASSES"):
        dataset_meta["classes"] = list(getattr(dataset_obj, "CLASSES"))

    data_root = getattr(dataset_obj, "data_root", dataset_meta.get("data_root", ""))
    ann_file = getattr(dataset_obj, "ann_file", dataset_meta.get("ann_file", ""))
    if isinstance(ann_file, (list, tuple)) and ann_file:
        ann_file = ann_file[0]

    # ---- determine dataset type ----
    ds_name = type(dataset_obj).__name__.lower()
    if dataset_hint == "auto":
        if "kitti" in ds_name:
            dataset_hint = "kitti"
        elif "nuscene" in ds_name:
            dataset_hint = "nuscenes"
        else:
            raise RuntimeError(f"Cannot auto-detect dataset from type '{type(dataset_obj).__name__}'.")

    # ---- instantiate the metric directly ----
    if dataset_hint == "kitti":
        if KittiMetric is None:
            raise ImportError("KittiMetric not available in this environment.")
        # Some versions accept 'jsonfile_prefix', some accept 'outfile_prefix';
        # _safe_instantiate_metric will only pass the supported ones.
        metric = _safe_instantiate_metric(
            KittiMetric,
            ann_file=ann_file,
            metric="bbox",
            jsonfile_prefix=jsonfile_prefix,
            outfile_prefix=jsonfile_prefix,  # alt name in some versions
            collect_device="cpu",
        )
    elif dataset_hint == "nuscenes":
        if NuScenesMetric is None:
            raise ImportError("NuScenesMetric not available in this environment.")
        metric = _safe_instantiate_metric(
            NuScenesMetric,
            data_root=data_root,                    # accepted in most versions
            ann_file=ann_file,
            metric="bbox",
            modality=dict(use_camera=False, use_lidar=True),
            jsonfile_prefix=jsonfile_prefix,
            eval_version=eval_version,
            collect_device="cpu",
        )
    else:
        raise RuntimeError(f"Unsupported dataset_hint='{dataset_hint}'")

    # attach dataset meta (classes, version, etc.)
    metric.dataset_meta = dataset_meta

    # ---- build model and (optionally) load checkpoint ----
    model = MODELS.build(cfg.model)
    model.to(device)  # type: ignore
    model.eval()
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location="cpu")

    # ---- helpers copied from your previous snippets (used as-is) ----
    # If you already have these in file, just reuse them; otherwise paste them above.
    # _extract_gt_samples_strict, _build_pred_records

    # ---- evaluation loop (DIRECT metric calls) ----
    n_seen = 0
    dev = torch.device(device if isinstance(device, str) else device)
    with torch.inference_mode():
        for batch in data_loader:
            preds = model.test_step(batch)  # raw model output

            # Ground-truth samples for this batch
            gt_samples = _extract_gt_samples_strict(batch)

            # Build exactly what the mmdet3d metrics expect
            pred_records = _build_pred_records(preds, gt_samples, dev)

            # Call metric.process(...) directly (bypass mmengine.Evaluator)
            metric.process({"data_samples": gt_samples}, pred_records)

            n_seen += 1

    # Final scores (NDS/mAP for nuScenes; AP/APH for KITTI, etc.)
    return metric.evaluate(n_seen)

if __name__ == "__main__":
    metrics = run_evaluation_direct(
        cfg_path="/data/rnd-liu/MyRepo/mmdetection3d/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py",
        checkpoint="/data/rnd-liu/MyRepo/mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth",
        split="val",
        device="cuda",
        backend="mmdet3d",
        dataset_hint="kitti",
        extra={"data_root": "/data/rnd-liu/MyRepo/mmdetection3d/data/kitti"},  # IMPORTANT: dataset root
    )
    print(metrics)
    # metrics = run_evaluation(
    #     cfg_path="/data/rnd-liu/MyRepo/mmdetection3d/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py",
    #     checkpoint="/data/rnd-liu/MyRepo/mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth",   # or None to eval current weights
    #     split="val",                              # or "test"
    #     device="cuda",                            # or "cpu"
    #     backend="mmdet3d",
    #     extra={
    #         # Optional: absolute dataset root to avoid relative-path issues
    #         "data_root": "/data/rnd-liu/MyRepo/mmdetection3d/data/kitti", # ""/data/rnd-liu/MyRepo/mmdetection3d/",
    #         # Optional: runtime cfg overrides
    #         # "cfg_options": {...}
    #     },
    #     # Optional quick smoke test limit
    #     # max_batches=50,
    # )
    # print(metrics)