#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, os.path as osp, time, json, csv, glob, argparse
from copy import deepcopy
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from mmengine.config import Config
from mmengine.runner import Runner
#from mmengine.dataset import build_dataset, build_dataloader

# --- replace previous dataset imports ---
# from mmengine.dataset import build_dataset, build_dataloader
from torch.utils.data import DataLoader
from mmengine.registry import build_from_cfg

# ---- registries: only DATASETS is guaranteed; DATALOADERS may not exist ----
try:
    from mmdet3d.registry import DATASETS
except Exception:
    # some stacks re-export via mmdet
    from mmdet.registry import DATASETS

# MMEngine utilities for sampler + collate
try:
    from mmengine.dataset import DefaultSampler, default_collate
except Exception:
    # very old mmengine
    from mmengine.dataset.samplers import DefaultSampler
    from mmengine.dataset.utils import default_collate

from copy import deepcopy
#from mmdet.datasets.utils import pseudo_collate
from mmengine.dataset.utils import pseudo_collate
def make_vis_loader_from_cfg(cfg):
    """
    Build a visualization dataloader that preserves multiview images as a list[CHW]
    per sample using pseudo_collate. Batch size = 1 for stable vis.
    """
    try:
        from mmdet3d.registry import DATASETS
    except Exception:
        from mmdet.registry import DATASETS

    from mmengine.registry import build_from_cfg
    from mmengine.dataset import DefaultSampler
    from torch.utils.data import DataLoader

    # prefer val_dataloader; fallback to test_dataloader
    if hasattr(cfg, 'val_dataloader'):
        dcfg = deepcopy(cfg.val_dataloader)
    elif hasattr(cfg, 'test_dataloader'):
        dcfg = deepcopy(cfg.test_dataloader)
    else:
        raise RuntimeError("Config has neither val_dataloader nor test_dataloader.")

    # unwrap dataset sub-config if present (MMDet3D often nests it)
    dataset_cfg = dcfg.get('dataset', dcfg)
    if 'dataset' in dataset_cfg:  # some configs double-wrap (ConcatDataset, etc.)
        dataset_cfg = dataset_cfg['dataset']

    dataset = build_from_cfg(dataset_cfg, DATASETS)

    # deterministic sampler for val/test
    sampler = DefaultSampler(dataset, shuffle=False)

    # force small workers + BS=1 for vis
    batch_size = 1
    num_workers = min(int(dcfg.get('num_workers', 2)), 2)
    pin_memory = bool(dcfg.get('pin_memory', True))
    persistent_workers = bool(dcfg.get('persistent_workers', False))
    drop_last = False

    # IMPORTANT: keep multiview lists
    collate_fn = pseudo_collate

    return DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )
    
# (optional) resolve collate fn from a cfg dict like {'type':'default_collate'}
def _resolve_collate_fn(collate_cfg):
    if collate_cfg is None:
        return default_collate
    if isinstance(collate_cfg, dict):
        t = collate_cfg.get('type', 'default_collate')
        if t in ('default_collate', 'DefaultFormatBundle'):  # be lenient
            return default_collate
    # fallback
    return default_collate

def build_dataset_from_cfg(dataset_cfg: dict):
    """Build dataset safely via registry (works across versions)."""
    return build_from_cfg(dataset_cfg, DATASETS)


def make_loader(dcfg: dict):
    from copy import deepcopy
    from torch.utils.data import DataLoader
    from mmengine.registry import build_from_cfg
    try:
        from mmdet3d.registry import DATASETS
    except Exception:
        from mmdet.registry import DATASETS

    _dcfg = deepcopy(dcfg)
    dataset_cfg = _dcfg.pop('dataset')
    dataset = build_from_cfg(dataset_cfg, DATASETS)

    # MMEngine sampler (shuffle False for val/test)
    from mmengine.dataset import DefaultSampler
    sampler_cfg = _dcfg.pop('sampler', {}) or {}
    shuffle = bool(sampler_cfg.get('shuffle', False))
    sampler = DefaultSampler(dataset, shuffle=shuffle)

    batch_size = int(_dcfg.pop('batch_size', 1))
    num_workers = int(_dcfg.pop('num_workers', 2))
    pin_memory = bool(_dcfg.pop('pin_memory', False))
    persistent_workers = bool(_dcfg.pop('persistent_workers', False))
    drop_last = bool(_dcfg.pop('drop_last', False))

    # >>> key change: collate keeps lists (multiview) per sample
    collate_fn = pseudo_collate

    return DataLoader(
        dataset=dataset, sampler=sampler,
        batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=pin_memory,
        persistent_workers=persistent_workers, drop_last=drop_last
    )

# --------------------------- filesystem utils ---------------------------
def ensure_dir(d: str): os.makedirs(d, exist_ok=True)
def save_json(obj: dict, path: str):
    with open(path, "w") as f: json.dump(obj, f, indent=2)
    print(f"[OK] Wrote {path}")

# ------------------------------- plotting -------------------------------
def save_pdf_plot(xs, ys, xlabel, ylabel, title, out_pdf):
    plt.figure(figsize=(4.8, 3.2), dpi=200)
    plt.plot(xs, ys, marker='o')
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.35); plt.tight_layout(); plt.savefig(out_pdf); plt.close()
    print(f"[OK] Plot saved to {out_pdf}")

# --------------------------- dataloader helper --------------------------
# def make_loader(dcfg: dict):
#     """Version-safe builder for test dataloader dict from cfg.test_dataloader."""
#     dcfg = deepcopy(dcfg)
#     dataset_cfg = dcfg.pop('dataset')
#     dataset = build_dataset(dataset_cfg)
#     # MMEngine build_dataloader expects: dataset=..., sampler=..., batch_size=..., num_workers=...
#     loader = build_dataloader(dataset=dataset, **dcfg)
#     return loader

# --------------------------- attn_chunk helpers -------------------------
def override_attn_chunk_in_cfg(cfg: Config, chunk: Optional[int]) -> bool:
    if chunk is None: return False
    vt = cfg.model.get('view_transform', None) if isinstance(cfg.model, dict) else None
    if isinstance(vt, dict) and ('attn_chunk' in vt):
        cfg.model['view_transform']['attn_chunk'] = int(chunk)
        return True
    return False

def override_attn_chunk_in_model(model: torch.nn.Module, chunk: Optional[int]) -> bool:
    if chunk is None: return False
    vt = getattr(model, 'view_transform', None)
    if vt is not None and hasattr(vt, 'attn_chunk'):
        try: setattr(vt, 'attn_chunk', int(chunk)); return True
        except Exception: return False
    return False

def read_attn_chunk_from_model(model: torch.nn.Module) -> Optional[int]:
    vt = getattr(model, 'view_transform', None)
    if vt is not None and hasattr(vt, 'attn_chunk'):
        try: return int(getattr(vt, 'attn_chunk'))
        except Exception: return None
    return None

# ----------------------------- qualitative ------------------------------
def _to_bgr_uint8(img_tensor):
    x = img_tensor.detach().cpu().float().clamp(0, 255)
    if x.max() <= 1.0: x = x * 255.0
    x = x.byte().numpy()
    x = np.transpose(x, (1, 2, 0))  # CHW->HWC RGB
    x = x[:, :, ::-1]               # RGB->BGR
    return x

def _project_pts3d_to_img(K, lidar2cam, pts_xyz):
    if lidar2cam.shape == (3, 4):
        T = np.vstack([lidar2cam, np.array([0,0,0,1], dtype=lidar2cam.dtype)])
    else:
        T = lidar2cam
    homo = np.c_[pts_xyz, np.ones((pts_xyz.shape[0], 1), dtype=pts_xyz.dtype)]
    cam = (T @ homo.T).T
    Z = cam[:, 2:3]; valid = (Z[:, 0] > 1e-3)
    uvw = (K @ cam[:, :3].T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    return uv, valid

def draw_3d_boxes_on_image(img_bgr, boxes_3d_xyzwhlr, K, lidar2cam, color=(0,255,0), thickness=2):
    H, W = img_bgr.shape[:2]
    for box in boxes_3d_xyzwhlr:
        x, y, z, w, l, h, yaw = map(float, box)  # simple axis-aligned approx
        dx, dy, dz = w/2, l/2, h/2
        corners = np.array([
            [x-dx,y-dy,z-dz],[x-dx,y+dy,z-dz],[x+dx,y+dy,z-dz],[x+dx,y-dy,z-dz],
            [x-dx,y-dy,z+dz],[x-dx,y+dy,z+dz],[x+dx,y+dy,z+dz],[x+dx,y-dy,z+dz],
        ], dtype=np.float32)
        uv, valid = _project_pts3d_to_img(K, lidar2cam, corners); uv = uv.astype(np.int32)
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for a,b in edges:
            if valid[a] and valid[b]:
                ua,va = uv[a]; ub,vb = uv[b]
                if 0<=ua<W and 0<=va<H and 0<=ub<W and 0<=vb<H:
                    cv2.line(img_bgr, (ua,va), (ub,vb), color, thickness)
    return img_bgr

def bev_visualization(pred_boxes_xyz, bev_range, img_size=800, color=(0,0,255)):
    xmin, ymin, _, xmax, ymax, _ = bev_range
    Wm, Hm = xmax-xmin, ymax-ymin
    scale = img_size / max(Wm, Hm)
    canvas = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
    for x,y,_ in pred_boxes_xyz:
        px = int((x - xmin) * scale); py = int((ymax - y) * scale)
        if 0 <= px < img_size and 0 <= py < img_size:
            cv2.circle(canvas, (px, py), 2, color, -1)
    return canvas

@torch.no_grad()
def save_qualitative_samples(cfg, model, loader, out_dir,
                             vis_samples=6,
                             bev_range=None,
                             device='cuda'):
    model = model.module if hasattr(model, 'module') else model
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for batch in loader:
        # With pseudo_collate, loader yields a list of dicts (len==1 if BS=1)
        if not isinstance(batch, list):
            batch = [batch]

        # Move inputs lists to device (keep list-of-CHW!)
        inputs = batch[0].get('inputs', batch[0].get('img'))
        if isinstance(inputs, list):
            batch[0]['inputs'] = [img.to(device, non_blocking=True) for img in inputs]
        elif torch.is_tensor(inputs) and inputs.ndim == 5 and inputs.size(0) == 1:
            # Extremely rare: rewrap [1, N, C, H, W] → list[CHW]
            mv = inputs[0].to(device, non_blocking=True)
            batch[0]['inputs'] = [mv[i] for i in range(mv.size(0))]
        else:
            # Let model/data_preprocessor handle other cases
            pass

        preds = model.test_step(batch)  # list of one Det3DDataSample
        pred = preds[0]
        data_sample = preds

        # multi-view image tensor expected as [1,N,3,H,W]
        imgs = batch['inputs']['img']
        if isinstance(imgs, list):
            mv = [imgs[i].squeeze(0) for i in range(len(imgs))]
        else:
            assert imgs.ndim == 5 and imgs.shape[0] == 1, f"Expect [1,N,3,H,W], got {tuple(imgs.shape)}"
            mv = [imgs[0, i] for i in range(imgs.shape[1])]

        metas = data_sample.metainfo
        cam2img_list  = metas.get('cam2img', None)
        lidar2cam_list= metas.get('lidar2cam', None)

        pred3d = getattr(data_sample, 'pred_instances_3d', None)
        boxes = None
        if pred3d is not None:
            if hasattr(pred3d, 'bboxes_3d'):
                boxes = pred3d.bboxes_3d.tensor.detach().cpu().numpy()
            elif hasattr(pred3d, 'boxes_3d'):
                boxes = pred3d.boxes_3d.tensor.detach().cpu().numpy()

        out_cam_dir = osp.join(out_dir, f"sample_{saved:03d}_cams"); os.makedirs(out_cam_dir, exist_ok=True)
        if boxes is not None and cam2img_list is not None and lidar2cam_list is not None:
            for i, img_chw in enumerate(mv):
                img_bgr = _to_bgr_uint8(img_chw)
                K  = np.array(cam2img_list[i], dtype=np.float32)[:3, :3]
                L2C= np.array(lidar2cam_list[i], dtype=np.float32)
                img_drawn = draw_3d_boxes_on_image(img_bgr, boxes, K, L2C, color=(0,255,0), thickness=2)
                cv2.imwrite(osp.join(out_cam_dir, f"cam{i}.png"), img_drawn)

        centers = boxes[:, :3] if boxes is not None else np.zeros((0,3))
        bev_img = bev_visualization(centers, bev_range, img_size=800, color=(0,0,255))
        cv2.imwrite(osp.join(out_dir, f"sample_{saved:03d}_bev.png"), bev_img)
        saved += 1
        if saved >= vis_samples:
            break
    print(f"[OK] Saved {saved} qualitative samples to {out_dir}")

# ------------------------------- evaluation -----------------------------
def cuda_sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def build_runner(cfg_path: str, work_dir: str, checkpoint: str):
    cfg = Config.fromfile(cfg_path); cfg = deepcopy(cfg)
    cfg.work_dir = work_dir; cfg.load_from = checkpoint; cfg.resume = False
    runner = Runner.from_cfg(cfg)
    return cfg, runner

def perf_loop(cfg: Config, runner: Runner, warmup_iters=10, max_samples=0):
    perf = {"iters":0,"latency_ms_all":[], "latency_ms_mean":None,"latency_ms_p50":None,
            "latency_ms_p90":None,"latency_ms_p95":None,"latency_ms_p99":None,"peak_mem_mb":None}
    test_loader = make_loader(cfg.test_dataloader)
    model = runner.model; model.eval()

    # warmup
    it = 0
    for batch in test_loader:
        it += 1
        if it > warmup_iters: break
        with torch.inference_mode():
            cuda_sync(); _ = model.test_step(batch); cuda_sync()

    # measure
    lat, max_mem, it = [], 0.0, 0
    for batch in test_loader:
        if max_samples and it >= max_samples: break
        with torch.inference_mode():
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            t0 = time.time(); _ = model.test_step(batch); cuda_sync()
            lat.append( (time.time() - t0) * 1000.0 )
            if torch.cuda.is_available():
                max_mem = max(max_mem, torch.cuda.max_memory_allocated()/(1024**2))
        it += 1

    perf["iters"] = it; perf["latency_ms_all"] = lat
    if len(lat):
        arr = np.array(lat, dtype=np.float64)
        perf["latency_ms_mean"] = float(arr.mean())
        for p in [50,90,95,99]:
            perf[f"latency_ms_p{p}"] = float(np.percentile(arr, p))
    perf["peak_mem_mb"] = float(max_mem)
    return perf

def evaluate_nuscenes(cfg: Config, runner: Runner):
    metrics = runner.test()
    if isinstance(metrics, list) and len(metrics) == 1: metrics = metrics[0]
    return metrics


# -------------------------- summary / plots-only ------------------------
def load_summary_from_json(path: str) -> List[dict]:
    with open(path, "r") as f: data = json.load(f)
    return data if isinstance(data, list) else [data]

import os, os.path as osp, json, glob, re, math
from typing import List, Dict, Any, Optional

ATTN_DIR_RE = re.compile(r"(?:^|[/\\])attn[_\-]?(\d+)(?:$|[/\\])", re.IGNORECASE)

def _safe_float(x, default=None):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def _infer_attn_from_path(path: str) -> Optional[int]:
    """Infer attn_chunk from any 'attn_XXXX' or 'attn-XXXX' segment in the path."""
    m = ATTN_DIR_RE.search(path)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def _json_load_relaxed(fp: str) -> Any:
    """Load JSON; if it fails due to NaN, try replacing bare NaN/Inf."""
    try:
        with open(fp, "r") as f:
            return json.load(f)
    except Exception:
        # Relaxed fallback: replace bare NaN/Inf with null before parsing
        try:
            with open(fp, "r") as f:
                s = f.read()
            s = s.replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
            return json.loads(s)
        except Exception:
            raise

def _normalize_run(rec: Dict[str, Any], work_dir: str) -> Dict[str, Any]:
    """Ensure required fields exist and infer attn_chunk if missing."""
    rec = dict(rec) if isinstance(rec, dict) else {"raw": rec}
    rec.setdefault("work_dir", work_dir)

    # Prefer explicit field, else infer from path, else keep None (baseline)
    chunk = rec.get("attn_chunk", None)
    if chunk in ("", "None"):  # cleanup common stringy cases
        chunk = None
    if chunk is None:
        inf = _infer_attn_from_path(work_dir)
        if inf is not None:
            chunk = inf
    # Try to coerce to int if it looks numeric
    if isinstance(chunk, str) and chunk.isdigit():
        chunk = int(chunk)
    rec["attn_chunk"] = chunk

    # Optional: ensure numeric perf fields are floats (if present)
    perf = rec.get("perf", {})
    if isinstance(perf, dict):
        for k in ["latency_ms_p50", "latency_ms_mean", "peak_mem_mb"]:
            if k in perf:
                perf[k] = _safe_float(perf[k], perf[k])
        rec["perf"] = perf

    return rec

def _gather_json_candidates(root: str) -> List[str]:
    """Find plausible result files under root (deep)."""
    patterns = [
        "**/result.json",
        "**/results.json",
        "**/metrics.json",
        # also accept per-run summaries; we’ll tag them by their folder
        "**/summary.json",
    ]
    files = set()
    for pat in patterns:
        for fp in glob.glob(osp.join(root, pat), recursive=True):
            if osp.isfile(fp):
                files.add(osp.normpath(fp))
    return sorted(files)

def collect_runs_from_dir(root: str) -> List[dict]:
    """
    Scan a results root and return a unique list of run dicts.
    Dedup key: (work_dir, attn_chunk); newest mtime wins.
    """
    root = osp.abspath(root)
    files = _gather_json_candidates(root)

    # Load all JSON blobs
    loaded: List[Dict[str, Any]] = []
    for fp in files:
        try:
            blob = _json_load_relaxed(fp)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
            continue

        # Two common shapes: a single run dict, or a list of runs (summary.json)
        work_dir = osp.dirname(fp)
        if isinstance(blob, list):
            for item in blob:
                if isinstance(item, dict):
                    loaded.append(_normalize_run(item, work_dir))
        elif isinstance(blob, dict):
            loaded.append(_normalize_run(blob, work_dir))
        else:
            # Unknown shape; wrap it minimally
            loaded.append(_normalize_run({"raw": blob}, work_dir))

    if not loaded:
        return []

    # Deduplicate: pick newest file per (work_dir, attn_chunk)
    # We need a map from that key to best record + mtime
    best: Dict[tuple, tuple] = {}
    for rec in loaded:
        wd = osp.normpath(str(rec.get("work_dir", "")))
        chunk = rec.get("attn_chunk", None)
        # Find a representative file mtime from that work_dir
        # Prefer result.json; fallback to any json inside
        candidate = None
        for name in ("result.json", "results.json", "metrics.json", "summary.json"):
            cand = osp.join(wd, name)
            if osp.isfile(cand):
                candidate = cand
                break
        if candidate is None:
            # fallback: the directory mtime
            mt = os.path.getmtime(wd) if osp.isdir(wd) else 0
        else:
            mt = os.path.getmtime(candidate)

        key = (wd, chunk)
        prev = best.get(key)
        if prev is None or mt > prev[0]:
            best[key] = (mt, rec)

    runs = [t[1] for t in best.values()]
    # Sort for stable presentation: baseline first, then increasing chunk, then path
    def _sort_key(r):
        c = r.get("attn_chunk")
        c_val = -1 if c is None else int(c)
        return (c_val, r.get("work_dir", ""))
    runs.sort(key=_sort_key)
    return runs


import csv, math
import json
import os, os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import os, json, random, time
from typing import Any, Dict
import numpy as np
import torch

# MMEngine minimal imports you still need
from mmengine.config import Config
from mmengine.runner import Runner

# ----------------- tiny, robust local helpers -----------------

def make_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory '{path}': {e}")

def set_determinism(seed: int = 0, deterministic: bool = False):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

def override_attn_chunk(cfg: Config, chunk: int):
    """Set model.view_transform.attn_chunk if present; otherwise no-op."""
    try:
        mv = cfg.get('model', {})
        vt = mv.get('view_transform', None)
        if isinstance(vt, dict) and 'attn_chunk' in vt:
            vt['attn_chunk'] = int(chunk)
            print(f"[INFO] Set model.view_transform.attn_chunk = {chunk}")
        else:
            print("[INFO] Model has no 'attn_chunk' (skipping override).")
    except Exception as e:
        print(f"[WARN] Could not set attn_chunk: {e}")

def pick_test_loader_cfg(cfg: Config):
    if hasattr(cfg, 'test_dataloader'):
        return cfg.test_dataloader
    if hasattr(cfg, 'val_dataloader'):
        return cfg.val_dataloader
    raise RuntimeError("No test/val dataloader found in cfg—please add one.")

def pick_evaluator_cfg(cfg: Config):
    if hasattr(cfg, 'test_evaluator'):
        return cfg.test_evaluator
    if hasattr(cfg, 'val_evaluator'):
        return cfg.val_evaluator
    raise RuntimeError("No test/val evaluator found in cfg—please add one.")

def _flatten_accuracy(acc_like: Any) -> Dict[str, Any]:
    flat = {}
    if isinstance(acc_like, dict):
        flat.update(acc_like)
    elif isinstance(acc_like, (list, tuple)):
        for r in acc_like:
            if isinstance(r, dict):
                flat.update(r)
    return flat

def _extract_scalar(d: Dict[str, Any], keys, default=0.0):
    for k in keys:
        v = d.get(k, None)
        if v is None:
            continue
        try:
            fv = float(v)
            if not np.isnan(fv):
                return fv
        except Exception:
            continue
    return float(default)

# ----------------- simple perf hook (no external deps) -----------------

from mmengine.hooks import Hook

class PerfHook(Hook):
    """Measure latency per test/val iter and report GPU peak memory."""
    def __init__(self, warmup_iters=10, max_samples=0):
        self.warmup_iters = int(max(0, warmup_iters))
        self.max_samples = int(max(0, max_samples))
        self._times = []
        self._count = 0
        self._tic = None

    # ---- loop-level entrypoints ----
    def before_test(self, runner):
        self._times.clear()
        self._count = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def before_val(self, runner):
        # in case the repo uses val-loop for evaluation
        self.before_test(runner)

    # ---- iter-level timing (test loop) ----
    def before_test_iter(self, runner, batch_idx, data_batch=None):
        self._tic = time.perf_counter()

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        toc = time.perf_counter()
        dt_ms = (toc - self._tic) * 1000.0
        self._count += 1
        if self._count > self.warmup_iters:
            self._times.append(dt_ms)

        # Best-effort early stop (optional). If you prefer full eval, remove.
        if self.max_samples > 0 and len(self._times) >= self.max_samples:
            # Tell the loop to finish; different MMEngine versions expose different attrs.
            # Try common fields (ignore failures silently).
            try:
                loop = runner.test_loop
            except Exception:
                loop = getattr(runner, 'val_loop', None)
            try:
                if loop is not None and hasattr(loop, 'dataloader'):
                    # Consume remaining dataloader items by marking max iters reached
                    loop._max_iters = loop._iter + 1
            except Exception:
                pass

    # ---- iter-level timing (val loop) ----
    def before_val_iter(self, runner, batch_idx, data_batch=None):
        self.before_test_iter(runner, batch_idx, data_batch)

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.after_test_iter(runner, batch_idx, data_batch, outputs)

    # ---- result aggregation ----
    def summary(self):
        if not self._times:
            peak_b = int(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
            return dict(
                num_measured=0,
                latency_ms_mean=0.0,
                latency_ms_p50=0.0,
                latency_ms_p90=0.0,
                latency_ms_p95=0.0,
                latency_ms_p99=0.0,
                latency_ms_min=0.0,
                latency_ms_max=0.0,
                peak_mem_bytes=peak_b,
                peak_mem_mb=float(peak_b / (1024**2)),
            )
        arr = np.asarray(self._times, dtype=np.float64)
        p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
        peak_b = int(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
        return dict(
            num_measured=int(arr.size),
            latency_ms_mean=float(arr.mean()),
            latency_ms_p50=float(p50),
            latency_ms_p90=float(p90),
            latency_ms_p95=float(p95),
            latency_ms_p99=float(p99),
            latency_ms_min=float(arr.min()),
            latency_ms_max=float(arr.max()),
            peak_mem_bytes=peak_b,
            peak_mem_mb=float(peak_b / (1024**2)),
        )
        
# ----------------- your simple, robust eval -----------------

def one_eval(cfg_path: str, checkpoint: str, work_dir: str, attn_chunk: int,
             warmup_iters: int, max_samples: int, vis_samples: int = 0, seed: int = 0) -> Dict[str, Any]:
    """
    - seeds
    - sets attn_chunk if present
    - builds Runner and runs test
    - saves {accuracy, perf} JSON
    """
    set_determinism(seed)

    cfg = Config.fromfile(cfg_path)
    # Ensure default_scope is set; Runner will pick it up.
    cfg.setdefault('default_scope', 'mmdet3d')

    make_dir(work_dir)
    override_attn_chunk(cfg, attn_chunk)

    cfg.resume = False
    cfg.load_from = checkpoint
    cfg.work_dir = work_dir

    cfg.test_dataloader = pick_test_loader_cfg(cfg)
    cfg.test_evaluator = pick_evaluator_cfg(cfg)
    if not hasattr(cfg, 'test_cfg'):
        cfg.test_cfg = dict()

    runner = Runner.from_cfg(cfg)
    perf_hook = PerfHook(warmup_iters=warmup_iters, max_samples=max_samples)
    runner.register_hook(perf_hook, priority='LOW')
    
    results = runner.test()

    # normalize metrics: flatten & expose NDS/mAP top-level
    acc_raw = _flatten_accuracy(results)
    nds = _extract_scalar(
        acc_raw,
        keys=("NDS",
              "pred_instances_3d_NuScenes/NDS",
              "NuScenes metric/pred_instances_3d_NuScenes/NDS"),
        default=np.nan
    )
    map_ = _extract_scalar(
        acc_raw,
        keys=("mAP",
              "pred_instances_3d_NuScenes/mAP",
              "NuScenes metric/pred_instances_3d_NuScenes/mAP"),
        default=np.nan
    )
    acc = dict(**acc_raw, NDS=nds, mAP=map_)

    perf = perf_hook.summary()

    out_json = os.path.join(work_dir, f"results_attn{attn_chunk}.json")
    with open(out_json, 'w') as f:
        json.dump(dict(attn_chunk=attn_chunk, accuracy=acc, perf=perf, work_dir=work_dir), f, indent=2)
    print(f"[OK] Saved results to {out_json}")

    return dict(attn_chunk=attn_chunk, accuracy=acc, perf=perf, work_dir=work_dir)


def one_eval2(cfg_path: str,
             checkpoint: str,
             work_dir: str,
             attn_chunk: Optional[int],
             warmup_iters: int,
             max_samples: int,
             vis_samples: int = 0,
             seed: int = 0) -> Dict[str, Any]:
    """
    Build Runner -> hijack internals for a controlled eval loop:
      - set & verify model.view_transform.attn_chunk (if present)
      - accurate latency + GPU peak memory (reserved) with sync/reset
      - feed predictions to the existing evaluator to compute NDS/mAP
      - save JSON {accuracy, perf, attn_chunk_applied, work_dir}
    """
    # ---- Determinism (no mmengine helpers) -----------------------------
    set_determinism(seed)

    # ---- Load & tweak cfg ----------------------------------------------
    cfg = Config.fromfile(cfg_path)
    cfg.setdefault('default_scope', 'mmdet3d')
    make_dir(work_dir)

    # Let the cfg carry the chunk (for logging), but we will also set it after build.
    attn_in_cfg = override_attn_chunk(cfg, attn_chunk)

    cfg.resume = False
    cfg.load_from = checkpoint
    cfg.work_dir = work_dir

    # Ensure test pieces exist (fallbacks are your own helpers)
    cfg.test_dataloader = pick_test_loader_cfg(cfg)
    cfg.test_evaluator = pick_evaluator_cfg(cfg)
    cfg.setdefault('test_cfg', dict())

    # ---- Build runner (so it constructs model/dataloader/evaluator) ----
    runner = Runner.from_cfg(cfg)

    # Access internals
    model = runner.model
    test_loop = getattr(runner, 'test_loop', None)
    if test_loop is None:
        raise RuntimeError("Runner has no test_loop; check your mmengine/mmdet3d version.")
    dataloader = getattr(test_loop, 'dataloader', None)
    evaluator = getattr(test_loop, 'evaluator', None)
    if dataloader is None or evaluator is None:
        raise RuntimeError("Could not access test_loop.dataloader/evaluator from Runner.")

    # ---- Apply attn_chunk directly on the built model (authoritative) --
    attn_applied = False
    actual_attn = None
    vt = getattr(model, 'view_transform', None)
    if vt is not None and hasattr(vt, 'attn_chunk'):
        try:
            if attn_chunk is not None:
                setattr(vt, 'attn_chunk', int(attn_chunk))
            actual_attn = int(getattr(vt, 'attn_chunk'))
            attn_applied = (attn_chunk is None) or (actual_attn == int(attn_chunk))
        except Exception:
            actual_attn = getattr(vt, 'attn_chunk', None)

    # ---- Move model to device & set eval mode --------------------------
    device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # ---- Accurate perf accounting --------------------------------------
    # We’ll track the global peak (across all iterations after warmup)
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    torch.cuda.synchronize(device) if device.type == 'cuda' else None
    measured = 0
    lat_ms = []

    global_peak_reserved = 0
    global_peak_alloc = 0

    # Evaluator reset (some versions need this)
    if hasattr(evaluator, 'dataset_meta'):
        # Typically set by the loop; if missing we try to infer
        evaluator.dataset_meta = getattr(dataloader.dataset, 'metainfo', None) or {}

    # ---- Warmup --------------------------------------------------------
    with torch.no_grad():
        it = iter(dataloader)
        for i in range(warmup_iters):
            try:
                data_batch = next(it)
            except StopIteration:
                break
            _ = model.test_step(data_batch)  # don’t record metrics or perf

    # ---- Measure loop --------------------------------------------------
    # Reset peaks after warmup, before the first measured iter
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Continue from current iterator; if exhausted, rebuild
    if 'it' not in locals():
        it = iter(dataloader)

    with torch.no_grad():
        for data_batch in it:
            t0 = time.perf_counter()
            preds = model.test_step(data_batch)  # list[Det3DDataSample]
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            lat_ms.append(dt_ms)

            # Feed evaluator
            # Many MMDet3D evaluators expect {"data_samples": gt, "preds": preds},
            # but the unified API is evaluator.process(data_batch, preds)
            try:
                evaluator.process(data_batch, preds)
            except TypeError:
                # Older signatures: evaluator.process({'data_samples': ...}, preds)
                evaluator.process({"data_samples": data_batch.get('data_samples', None)}, preds)

            # Track peaks
            if device.type == 'cuda':
                cur_reserved = torch.cuda.max_memory_reserved(device)
                cur_alloc = torch.cuda.max_memory_allocated(device)
                global_peak_reserved = max(global_peak_reserved, cur_reserved)
                global_peak_alloc = max(global_peak_alloc, cur_alloc)

            measured += 1
            if max_samples and measured >= max_samples:
                break

    # ---- Evaluate metrics ----------------------------------------------
    # Most evaluators accept size of dataset (or None)
    try:
        size = len(getattr(dataloader, 'dataset', []))
        metrics = evaluator.evaluate(size)
    except TypeError:
        metrics = evaluator.evaluate()

    # ---- Flatten + extract NDS/mAP (robust) ----------------------------
    acc_raw = _flatten_accuracy(metrics)
    nds = _extract_scalar(
        acc_raw,
        keys=("NDS",
              "pred_instances_3d_NuScenes/NDS",
              "NuScenes metric/pred_instances_3d_NuScenes/NDS"),
        default=np.nan
    )
    map_ = _extract_scalar(
        acc_raw,
        keys=("mAP",
              "pred_instances_3d_NuScenes/mAP",
              "NuScenes metric/pred_instances_3d_NuScenes/mAP"),
        default=np.nan
    )
    accuracy = dict(**acc_raw, NDS=nds, mAP=map_)

    # ---- Perf summary ---------------------------------------------------
    perf = {}
    if len(lat_ms):
        lat_arr = np.array(lat_ms, dtype=np.float64)
        perf.update(dict(
            num_measured=int(measured),
            latency_ms_mean=float(lat_arr.mean()),
            latency_ms_p50=float(np.percentile(lat_arr, 50)),
            latency_ms_p90=float(np.percentile(lat_arr, 90)),
            latency_ms_p95=float(np.percentile(lat_arr, 95)),
            latency_ms_p99=float(np.percentile(lat_arr, 99)),
            latency_ms_min=float(lat_arr.min()),
            latency_ms_max=float(lat_arr.max()),
        ))
    if device.type == 'cuda':
        perf.update(dict(
            peak_mem_bytes=int(global_peak_reserved),
            peak_mem_mb=float(global_peak_reserved / (1024 ** 2)),
            peak_mem_alloc_bytes=int(global_peak_alloc),
            peak_mem_alloc_mb=float(global_peak_alloc / (1024 ** 2)),
        ))

    # ---- Persist JSON ---------------------------------------------------
    out_json = os.path.join(work_dir, f"results_attn{attn_chunk}.json")
    payload = dict(
        attn_chunk=attn_chunk,
        attn_chunk_in_cfg=attn_in_cfg,
        attn_chunk_on_model=actual_attn,
        attn_chunk_applied=bool(attn_applied),
        accuracy=accuracy,
        perf=perf,
        work_dir=work_dir
    )
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Saved results to {out_json}")

    return payload
# -------- robust metric getters (reuse in both places) ----------
def _to_float(x, default=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def _get_metric(acc: dict, name: str, default=0.0):
    """Fetch a metric like 'NDS' or 'mAP' or 'mATE' from flexible NuScenes keys."""
    if not isinstance(acc, dict):
        return default
    # direct
    if name in acc:
        return _to_float(acc[name], default)
    # common variants
    candidates = [
        f"pred_instances_3d_NuScenes/{name}",
        f"NuScenes metric/pred_instances_3d_NuScenes/{name}",
        f"NuScenes metric/{name}",
    ]
    for k in candidates:
        if k in acc:
            return _to_float(acc[k], default)
    # suffix match
    suffix = f"/{name}"
    suffix_keys = [k for k in acc.keys() if isinstance(k, str) and k.endswith(suffix)]
    if suffix_keys:
        best = max(suffix_keys, key=len)
        return _to_float(acc[best], default)
    # case-insensitive suffix
    suffix_keys_ci = [k for k in acc.keys()
                      if isinstance(k, str) and k.lower().endswith(suffix.lower())]
    if suffix_keys_ci:
        best = max(suffix_keys_ci, key=len)
        return _to_float(acc[best], default)
    return default

def save_comprehensive_metrics_bar(acc: dict, title: str, out_pdf: str):
    """
    Make a single figure that summarizes major NuScenes metrics for ONE run:
      - mAP (higher better), NDS (higher better),
      - mATE/mASE/mAOE/mAVE/mAAE (lower better)
    Saves to a vector PDF for paper use.
    """
    # Pull metrics (robust to different key names)
    mAP = _get_metric(acc, "mAP", 0.0)
    NDS = _get_metric(acc, "NDS", 0.0)
    mATE = _get_metric(acc, "mATE", 0.0)
    mASE = _get_metric(acc, "mASE", 0.0)
    mAOE = _get_metric(acc, "mAOE", 0.0)
    mAVE = _get_metric(acc, "mAVE", 0.0)
    mAAE = _get_metric(acc, "mAAE", 0.0)

    labels = ["mAP (↑)", "NDS (↑)", "mATE (↓)", "mASE (↓)", "mAOE (↓)", "mAVE (↓)", "mAAE (↓)"]
    values = [mAP, NDS, mATE, mASE, mAOE, mAVE, mAAE]

    plt.figure(figsize=(7.2, 3.6), dpi=200)
    x = np.arange(len(labels))
    bars = plt.bar(x, values)  # no explicit colors (journal-friendly default)

    # Value labels on top of bars
    for rect, val in zip(bars, values):
        plt.text(rect.get_x() + rect.get_width()/2.0,
                 rect.get_height(),
                 f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9)

    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Value")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(osp.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    
def build_plots_and_tsv(runs: list, out_dir: str):
    ensure_dir(out_dir)
    if not runs:
        print("[WARN] No runs to plot.")
        return

    # --- NEW: save one comprehensive bar chart per run ---
    for i, r in enumerate(runs):
        acc = r.get("accuracy", {}) or {}
        # Create a readable title: include chunk (or 'baseline')
        tag = r.get("attn_chunk")
        tag_str = "baseline" if tag in (None, -1) else f"attn_chunk={tag}"
        title = f"NuScenes summary metrics ({tag_str})"
        pdf_path = osp.join(out_dir, f"summary_metrics_run{i}_{tag_str.replace('=','_')}.pdf")
        save_comprehensive_metrics_bar(acc, title, pdf_path)
        print(f"[OK] Comprehensive summary figure → {pdf_path}")

    # --- Existing across-chunk comparison plots ---
    chunks, mAPs, NDSs, lat, mem = [], [], [], [], []
    for r in runs:
        chunks.append(r.get("attn_chunk"))
        acc = r.get("accuracy", {}) or {}

        mAPs.append(_get_metric(acc, "mAP", 0.0))
        NDSs.append(_get_metric(acc, "NDS", 0.0))

        perf = r.get("perf", {}) or {}
        lat.append(_to_float(perf.get("latency_ms_p50", perf.get("latency_ms_mean", 0.0)), 0.0))
        mem.append(_to_float(perf.get("peak_mem_mb", 0.0), 0.0))

    # Sort by chunk; treat None as -1 to keep them first
    sort_keys = np.array([(-1 if c is None else int(c)) for c in chunks], dtype=np.int64)
    order = np.argsort(sort_keys)

    chunks = list(np.array(chunks, dtype=object)[order])
    mAPs   = list(np.array(mAPs)[order])
    NDSs   = list(np.array(NDSs)[order])
    lat    = list(np.array(lat)[order])
    mem    = list(np.array(mem)[order])

    xlabels = [("baseline" if c in (None, -1) else str(c)) for c in chunks]

    save_pdf_plot(xlabels, mAPs, "attn_chunk", "mAP",
                  "nuScenes mAP vs attn_chunk",
                  osp.join(out_dir, "mAP_vs_attn_chunk.pdf"))
    save_pdf_plot(xlabels, NDSs, "attn_chunk", "NDS",
                  "nuScenes NDS vs attn_chunk",
                  osp.join(out_dir, "NDS_vs_attn_chunk.pdf"))
    save_pdf_plot(xlabels, lat, "attn_chunk", "Latency (ms, p50)",
                  "Latency vs attn_chunk",
                  osp.join(out_dir, "latency_vs_attn_chunk.pdf"))
    save_pdf_plot(xlabels, mem, "attn_chunk", "Peak GPU Mem (MB)",
                  "Memory vs attn_chunk",
                  osp.join(out_dir, "memory_vs_attn_chunk.pdf"))

    # TSV export (use the string labels so 'baseline' is preserved)
    import csv
    tsv_path = osp.join(out_dir, "summary.tsv")
    with open(tsv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(["attn_chunk", "mAP", "NDS", "latency_ms_p50", "peak_mem_mb"])
        for c_label, a, n, l, m in zip(xlabels, mAPs, NDSs, lat, mem):
            w.writerow([c_label, a, n, l, m])
    print(f"[OK] Wrote {tsv_path}")

def probe_attn_chunk_support(cfg_path: str) -> bool:
    from mmengine.config import Config
    cfg = Config.fromfile(cfg_path)
    vt = (cfg.get('model', {}) or {}).get('view_transform', {}) or {}
    # Support if the view transform exposes an attn_chunk param or looks like Cross-Attn VT
    names = str(vt.get('type', '')).lower()
    keys  = vt.keys()
    return ('attn_chunk' in keys) or ('crossattn' in names) or ('cross_attn' in names)

# --------------------------------- main ---------------------------------
#ap.add_argument("--config", type=str, default="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py", help="Path to MMDet3D config .py")
#    ap.add_argument("--checkpoint", type=str, default="modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth", help="Path to model checkpoint .pth")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="work_dirs/mybevfusion7_newv3/mybevfusion7_crossattnaux_paintingv3.py", help="Path to MMDet3D config .py")
    parser.add_argument("--checkpoint", type=str, default="work_dirs/mybevfusion7_newv3/epoch_3.pth", help="Path to model checkpoint .pth")
    parser.add_argument("--work-dir", default="work_dirs/mybevfusion7_newv3", help="Output dir")
    parser.add_argument("--eval", action="store_true", default=True, help="Run evaluation (accuracy + perf)")
    parser.add_argument("--viz-only", action="store_true", default=True, help="Run qualitative visualization only")
    parser.add_argument("--plots-only", action="store_true", default=True, help="Build plots/TSV from saved JSON")
    parser.add_argument("--attn-chunks", type=int, nargs="+", default=None,
                        help="List of attn_chunk values to sweep; ignored if model lacks attr. [2048, 4096, 8192, 16384], ")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0, help="Measured samples after warmup; 0 = full split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vis-samples", type=int, default=6, help="How many qualitative frames to dump")
    parser.add_argument("--qual-subdir", type=str, default="qual_viz", help="Output subdir for qualitative PNGs")
    parser.add_argument("--summary-json", type=str, default="eval_nuscenes_results/attn_4096/results_attn4096.json", help="Path to a summary.json (optional)")
    args = parser.parse_args()

    ensure_dir(args.work_dir)
    if args.eval:
        summary = []
    
        # Decide the list of chunks to try
        supports = probe_attn_chunk_support(args.config)
        chunks_to_run = (args.attn_chunks if (args.attn_chunks and supports) else [None])

        for chunk in chunks_to_run:
            print("="*80)
            print(f"[RUN] attn_chunk={chunk}")
            tag = f"attn_{chunk}" if chunk is not None else "attn_none"
            run_dir = osp.join(args.work_dir, tag)
            ensure_dir(run_dir)
            

            # res = one_eval2(args.config, args.checkpoint, run_dir, chunk,
            #             args.warmup_iters, args.max_samples, args.seed)
            res = one_eval(args.config, args.checkpoint, run_dir, chunk,
                        args.warmup_iters, args.max_samples, args.seed)
            # Log but do NOT break; this allows you to sweep even if the first chunk fails to apply
            if chunk is not None and not res.get("attn_chunk", False):
                print(f"[INFO] Chunk {chunk} could not be applied to this model. "
                    f"Run tagged '{tag}' will reflect default behavior.")

            summary.append(res)

        save_json(summary, osp.join(args.work_dir, "summary.json"))

    # if args.eval:
    #     summary = []
    #     for chunk in (args.attn_chunks if len(args.attn_chunks) else [None]):
    #         print("="*80); print(f"[RUN] attn_chunk={chunk}")
    #         tag = f"attn_{chunk}" if chunk is not None else "attn_none"
    #         run_dir = osp.join(args.work_dir, tag); ensure_dir(run_dir)
    #         res = one_eval(args.config, args.checkpoint, run_dir, chunk,
    #                        args.warmup_iters, args.max_samples, args.seed)
    #         summary.append(res)
    #         if not res.get("attn_chunk_supported", False):
    #             print("[INFO] Model does not expose 'attn_chunk'. Stopping chunk sweep early.")
    #             break
    #     save_json(summary, osp.join(args.work_dir, "summary.json"))

    # if args.viz_only:
    #     print("="*80)
    #     print("[RUN] Qualitative visualization only")

    #     cfg, runner = build_runner(args.config, args.work_dir, args.checkpoint)
    #     model = runner.model
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     model = model.to(device)

    #     # Build a special vis loader that preserves multiview as list[CHW]
    #     vis_loader = make_vis_loader_from_cfg(cfg)

    #     bev_range = getattr(cfg, 'point_cloud_range', [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])
    #     out_dir = os.path.join(args.work_dir, args.qual_subdir)

    #     try:
    #         save_qualitative_samples(
    #             cfg=cfg,
    #             model=model,
    #             loader=vis_loader,
    #             out_dir=out_dir,
    #             vis_samples=args.vis_samples,
    #             bev_range=bev_range,
    #             device=device,
    #         )
    #         print(f"[OK] Saved qualitative samples to: {out_dir}")
    #     except Exception as e:
    #         print(f"[WARN] Qualitative visualization failed: {e}")
    #     sys.exit(0)

    if args.plots_only:
        print("="*80); print("[RUN] Plots/TSV only from saved JSON")
        if args.summary_json and osp.isfile(args.summary_json):
            runs = load_summary_from_json(args.summary_json)
        else:
            runs = collect_runs_from_dir(args.work_dir)
        build_plots_and_tsv(runs, args.work_dir)

if __name__ == "__main__":
    main()