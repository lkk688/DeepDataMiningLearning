#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# eval_nuscenes_crossattn.py
# Author: Your Name
#
# Evaluate an MMDetection3D BEVFusion variant with CrossAttn LSS on nuScenes.
# Features:
# - Sweep over multiple `model.view_transform.attn_chunk` values (post-load override)
# - Accurate latency (per frame) with CUDA events
# - Accurate peak GPU memory (per setting) using torch.cuda.max_memory_allocated
# - Save raw JSON results per setting + a combined summary JSON/CSV
# - Generate paper-ready PDF plots (accuracy vs chunk, latency vs chunk, memory vs chunk)
# - Export a few qualitative visualizations:
#     * 3D boxes projected on a chosen camera image
#     * BEV map with 2D boxes footprints
#
# This script is designed to run inside your MMDet3D environment.
#
# Example:
#   python eval_nuscenes_crossattn.py \
#     --config configs/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
#     --checkpoint work_dirs/your_run/epoch_6.pth \
#     --work-dir work_dirs/eval_crossattn_sweep \
#     --attn-chunks 2048 4096 8192 16384 \
#     --max-samples 1500 \
#     --vis-samples 8
#
# Tips:
# - Set --max-samples small for quick probes; set to 0 to evaluate full val split.
# - If your config already has CrossAttnLSSTransform, this script just overrides attn_chunk.
# - For Depth LSS baselines, script still evaluates but the chunk sweep has no effect.
#
# -----------------------------------------------------------------------------

import os
import json
import time
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mmengine import Config, mkdir_or_exist
from mmengine.registry import init_default_scope, RUNNERS, MODELS, DATASETS #, EVALUATORS
from mmengine.runner import Runner

try:
    from mmdet3d.structures import LiDARInstance3DBoxes
except Exception:
    LiDARInstance3DBoxes = None

def set_determinism(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def override_attn_chunk(cfg: Config, chunk: int) -> None:
    try:
        if 'model' in cfg and 'view_transform' in cfg.model and isinstance(cfg.model['view_transform'], dict):
            vt = cfg.model['view_transform']
            vt_type = vt.get('type', '')
            if vt_type in ['CrossAttnLSSTransform', 'CrossAttnLSS', 'CA_LSS', 'LSSCrossAttn']:
                vt['attn_chunk'] = int(chunk)
                print(f"[cfg] Set view_transform.attn_chunk = {chunk}")
    except Exception as e:
        print(f"[WARN] Failed to set attn_chunk to {chunk}: {e}")

def pick_evaluator_cfg(cfg: Config) -> Dict[str, Any]:
    if hasattr(cfg, 'val_evaluator'):
        return cfg.val_evaluator
    if hasattr(cfg, 'test_evaluator'):
        return cfg.test_evaluator
    return dict(type='NuScenesMetric', metric='bbox')

def pick_test_loader_cfg(cfg: Config) -> Dict[str, Any]:
    if hasattr(cfg, 'val_dataloader'):
        return cfg.val_dataloader
    if hasattr(cfg, 'test_dataloader'):
        return cfg.test_dataloader
    raise RuntimeError("No val_dataloader or test_dataloader found in config.")

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def now():
    return time.strftime('%Y%m%d_%H%M%S')

# --- helpers for setting attn_chunk both in config and on the built model ---

def set_nested(cfg_obj, dotted_key, value):
    """Set cfg['a']['b']['c']=value given 'a.b.c' (handles Config/ConfigDict)."""
    cur = cfg_obj
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        if isinstance(cur, dict):
            cur = cur[p]
        else:
            cur = getattr(cur, p)
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)

def set_attn_chunk_on_config(cfg, chunk: int):
    """
    Try common locations where CrossAttnLSSTransform might live in configs.
    Adjust/add more keys here if your repo uses different naming.
    """
    keys_to_try = [
        "model.view_transform.attn_chunk",
        "model.img_view_transform.attn_chunk",
        "model.view_transforms.attn_chunk",  # some forks use plural
    ]
    ok = False
    for k in keys_to_try:
        try:
            set_nested(cfg, k, int(chunk))
            ok = True
        except Exception:
            pass
    return ok

def unwrap_model(m):
    # works for DDP, MMDistributedDataParallel, DataParallel, etc.
    try:
        from mmengine.model import is_model_wrapper
        return m.module if is_model_wrapper(m) else m
    except Exception:
        return getattr(m, "module", m)

def set_attn_chunk_on_model(model, chunk: int):
    """
    Set attn_chunk on the actual module instance after build/load.
    """
    m = unwrap_model(model)

    # Try the usual attribute
    if hasattr(m, "view_transform") and hasattr(m.view_transform, "attn_chunk"):
        m.view_transform.attn_chunk = int(chunk)
        return True

    # Some repos name it differently:
    for name in ["img_view_transform", "view_transforms"]:
        vt = getattr(m, name, None)
        if vt is not None and hasattr(vt, "attn_chunk"):
            vt.attn_chunk = int(chunk)
            return True

    # As a last resort, search modules by type/attr
    for sub in m.modules():
        if hasattr(sub, "attn_chunk"):
            sub.attn_chunk = int(chunk)
            return True

    return False

from mmengine.hooks import Hook

class EvalPerfHook(Hook):
    def __init__(self, warmup_iters: int = 10, max_samples: int = 0):
        super().__init__()
        self.warmup_iters = warmup_iters
        self.max_samples = max_samples
        self.latencies_ms = []
        self.timestamps = []
        self.num_seen = 0
        self._using_cuda = torch.cuda.is_available()
        self.peak_mem_bytes = 0
        self._start_event = None
        self._end_event = None

    def before_test(self, runner):
        self.latencies_ms.clear()
        self.timestamps.clear()
        self.num_seen = 0
        if self._using_cuda:
            torch.cuda.reset_peak_memory_stats()
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
        print("[EvalPerfHook] Start test. Warmup:", self.warmup_iters, "Max samples:", self.max_samples)

    def before_test_iter(self, runner, batch_idx: int, data_batch=None):
        if self._using_cuda:
            torch.cuda.synchronize()
            self._start_event.record()

    def after_test_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        if self._using_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            elapsed_ms = 0.0

        self.num_seen += 1
        if self.num_seen > self.warmup_iters:
            self.latencies_ms.append(float(elapsed_ms))
            self.timestamps.append(time.time())

        if self._using_cuda:
            self.peak_mem_bytes = max(self.peak_mem_bytes, torch.cuda.max_memory_allocated())

        if self.max_samples > 0 and self.num_seen >= (self.warmup_iters + self.max_samples):
            runner.test_loop._epoch_length = runner.test_loop.iter

    def after_test(self, runner):
        if self._using_cuda:
            torch.cuda.synchronize()
            self.peak_mem_bytes = max(self.peak_mem_bytes, torch.cuda.max_memory_allocated())
        print(f"[EvalPerfHook] Done. Used {len(self.latencies_ms)} measured samples. "
              f"Peak mem: {self.peak_mem_bytes/1024/1024:.2f} MB")

    def summary(self) -> Dict[str, Any]:
        lat = np.array(self.latencies_ms, dtype=np.float64) if self.latencies_ms else np.array([0.0])
        out = dict(
            num_measured=int(lat.size),
            latency_ms_mean=float(lat.mean()) if lat.size else 0.0,
            latency_ms_p50=float(np.percentile(lat, 50)) if lat.size else 0.0,
            latency_ms_p90=float(np.percentile(lat, 90)) if lat.size else 0.0,
            latency_ms_p95=float(np.percentile(lat, 95)) if lat.size else 0.0,
            latency_ms_p99=float(np.percentile(lat, 99)) if lat.size else 0.0,
            latency_ms_min=float(lat.min()) if lat.size else 0.0,
            latency_ms_max=float(lat.max()) if lat.size else 0.0,
            peak_mem_bytes=int(self.peak_mem_bytes),
            peak_mem_mb=float(self.peak_mem_bytes / (1024.0 * 1024.0))
        )
        return out

def draw_bev_boxes_png(save_path: str, bev_size_hw: tuple, boxes_xywh: np.ndarray, title: str = "BEV boxes"):
    ensure_dir(os.path.dirname(save_path) or ".")
    H, W = bev_size_hw
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    for (x, y, w, h) in boxes_xywh:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5)
        ax.add_patch(rect)
    ax.set_xlabel("BEV X (cells)")
    ax.set_ylabel("BEV Y (cells)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def project_lidar_box_to_image(box_3d: np.ndarray, lidar2img: np.ndarray, img_w: int, img_h: int) -> Optional[np.ndarray]:
    pts = np.concatenate([box_3d, np.ones((box_3d.shape[0], 1))], axis=1)
    proj = (lidar2img @ pts.T).T
    zs = proj[:, 2]
    valid = zs > 1e-6
    if not np.any(valid):
        return None
    u = proj[:, 0] / (proj[:, 2] + 1e-6)
    v = proj[:, 1] / (proj[:, 2] + 1e-6)
    uv = np.stack([u, v], axis=1)
    uv[:, 0] = np.clip(uv[:, 0], 0, img_w - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, img_h - 1)
    return uv

def box3d_corners_xyzhwl_yaw(box_center: np.ndarray) -> np.ndarray:
    x, y, z, h, w, l, yaw = box_center
    xs = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2 ])
    ys = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2 ])
    zs = np.array([ 0, 0, 0, 0,  -h, -h, -h, -h ]) + z
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    xr = x + xs * cos_y - ys * sin_y
    yr = y + xs * sin_y + ys * cos_y
    return np.stack([xr, yr, zs], axis=1)

def overlay_boxes_on_image(img: np.ndarray, boxes_center: np.ndarray, lidar2img: np.ndarray) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    img_out = img.copy()
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for b in boxes_center:
        corners = box3d_corners_xyzhwl_yaw(b)
        uv = project_lidar_box_to_image(corners, lidar2img, w, h)
        if uv is None:
            continue
        uv = uv.astype(np.int32)
        for (i,j) in edges:
            p1 = tuple(uv[i])
            p2 = tuple(uv[j])
            cv2.line(img_out, p1, p2, (0,255,0), 2)
    return img_out

def save_pdf_plot(x, y, xlabel, ylabel, title, save_path):
    ensure_dir(os.path.dirname(save_path) or ".")
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(x, y, marker='o', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_bar_pdf(labels, values, xlabel, ylabel, title, save_path):
    ensure_dir(os.path.dirname(save_path) or ".")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(labels, values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def one_eval(cfg_path: str, checkpoint: str, work_dir: str, attn_chunk: int,
             warmup_iters: int, max_samples: int, vis_samples: int = 0, seed: int = 0) -> Dict[str, Any]:
    set_determinism(seed)
    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    mkdir_or_exist(work_dir)
    override_attn_chunk(cfg, attn_chunk)
    cfg.resume = False
    cfg.load_from = checkpoint
    cfg.work_dir = work_dir
    if not hasattr(cfg, 'test_dataloader'):
        cfg.test_dataloader = pick_test_loader_cfg(cfg)
    if not hasattr(cfg, 'test_evaluator'):
        cfg.test_evaluator = pick_evaluator_cfg(cfg)
    if not hasattr(cfg, 'test_cfg'):
        cfg.test_cfg = dict()
    runner = Runner.from_cfg(cfg)
    perf_hook = EvalPerfHook(warmup_iters=warmup_iters, max_samples=max_samples)
    runner.register_hook(perf_hook, priority='LOW')
    results = runner.test()
    acc = {}
    if isinstance(results, dict):
        acc.update(results)
    elif isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                acc.update(r)
    perf = perf_hook.summary()
    out_json = os.path.join(work_dir, f"results_attn{attn_chunk}.json")
    with open(out_json, 'w') as f:
        json.dump(dict(attn_chunk=attn_chunk, accuracy=acc, perf=perf), f, indent=2)
    print(f"[OK] Saved results to {out_json}")
    if vis_samples > 0:
        try:
            save_qualitative_samples(cfg, runner, work_dir, vis_samples=vis_samples)
        except Exception as e:
            print(f"[WARN] Qualitative visualization failed: {e}")
    return dict(attn_chunk=attn_chunk, accuracy=acc, perf=perf, json_path=out_json)

def save_qualitative_samples(cfg: Config, runner: Runner, work_dir: str, vis_samples: int = 4):
    try:
        from mmengine.dataset import build_dataloader as mm_build_dataloader
    except Exception:
        mm_build_dataloader = None
    dataset = DATASETS.build(pick_test_loader_cfg(cfg)['dataset'])
    if mm_build_dataloader is not None:
        dl = mm_build_dataloader(pick_test_loader_cfg(cfg))
    else:
        from mmengine.dataset import default_collate
        dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         num_workers=2, collate_fn=default_collate)
    model = runner.model
    model.eval()
    count = 0
    for data in dl:
        with torch.no_grad():
            if isinstance(data, dict):
                outputs = model.test_step(data)
            else:
                outputs = model.test_step({'inputs': data})
        samples = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        sample = samples[0]
        metas = getattr(sample, 'metainfo', None)
        if hasattr(sample, 'pred_instances_3d') and hasattr(sample.pred_instances_3d, 'bboxes_3d'):
            b3d = sample.pred_instances_3d.bboxes_3d
            if hasattr(b3d, 'tensor'):
                boxes_np = b3d.tensor.detach().cpu().numpy()
            else:
                boxes_np = np.asarray(b3d)
        else:
            boxes_np = np.zeros((0, 7), dtype=np.float32)
        try:
            img_path = None
            lidar2img = None
            if metas is not None:
                img_p = metas.get('img_path', None)
                if isinstance(img_p, (list, tuple)):
                    img_path = img_p[0]
                else:
                    img_path = img_p
                lid2img = metas.get('lidar2img', None)
                if isinstance(lid2img, (list, tuple)):
                    lidar2img = np.array(lid2img[0])
                elif lid2img is not None:
                    lidar2img = np.array(lid2img)
            if img_path is not None and lidar2img is not None:
                import cv2
                img = cv2.imread(img_path)
                if img is not None:
                    if boxes_np.shape[1] == 7:
                        boxes_c = np.stack([
                            boxes_np[:,0], boxes_np[:,1], boxes_np[:,2],
                            boxes_np[:,5], boxes_np[:,4], boxes_np[:,3], boxes_np[:,6]
                        ], axis=1)
                    else:
                        boxes_c = np.zeros((0,7), dtype=np.float32)
                    over = overlay_boxes_on_image(img[..., ::-1], boxes_c, lidar2img)
                    out_path = os.path.join(work_dir, "qual", f"img_overlay_{count:04d}.png")
                    ensure_dir(os.path.dirname(out_path))
                    import cv2
                    cv2.imwrite(out_path, over[..., ::-1])
                    print(f"[qual] Wrote {out_path}")
        except Exception as e:
            print(f"[qual] Image overlay failed: {e}")
        try:
            bev_h = 180
            bev_w = 180
            m2c = 1.0 / 0.3
            xywh = []
            for bb in boxes_np:
                x, y, z, l, w, h, yaw = bb
                cx = bev_w/2 + x * m2c
                cy = bev_h/2 - y * m2c
                ww = max(1.0, w * m2c)
                hh = max(1.0, l * m2c)
                xywh.append([cx - ww/2, cy - hh/2, ww, hh])
            out_bev = os.path.join(work_dir, "qual", f"bev_{count:04d}.png")
            draw_bev_boxes_png(out_bev, (bev_h, bev_w), np.array(xywh), title="BEV footprints")
            print(f"[qual] Wrote {out_bev}")
        except Exception as e:
            print(f"[qual] BEV draw failed: {e}")
        count += 1
        if count >= vis_samples:
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="work_dirs/mybevfusion7_new/mybevfusion7_crossattnaux_painting.py", help="Path to MMDet3D config .py")
    parser.add_argument("--checkpoint", type=str, default="work_dirs/mybevfusion7_new/epoch_4.pth", help="Path to model checkpoint .pth")
    parser.add_argument("--work-dir", default="eval_nuscenes_results", help="Output dir")
    parser.add_argument("--attn-chunks", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0, help="Measured samples after warmup; 0 = full split")
    parser.add_argument("--vis-samples", type=int, default=6, help="How many qualitative frames to dump per run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensure_dir(args.work_dir)
    summary = []
    for chunk in args.attn_chunks:
        print("="*80)
        print(f"[RUN] attn_chunk={chunk}")
        run_dir = os.path.join(args.work_dir, f"attn_{chunk}")
        ensure_dir(run_dir)
        res = one_eval(
            cfg_path=args.config,
            checkpoint=args.checkpoint,
            work_dir=run_dir,
            attn_chunk=chunk,
            warmup_iters=args.warmup_iters,
            max_samples=args.max_samples,
            vis_samples=args.vis_samples,
            seed=args.seed
        )
        summary.append(res)

    summary_path = os.path.join(args.work_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Wrote {summary_path}")

    chunks = []
    mAPs = []
    NDSs = []
    lat = []
    mem = []
    for item in summary:
        chunks.append(item["attn_chunk"])
        acc = item.get("accuracy", {})
        mAPs.append(float(acc.get("mAP", acc.get("pred_instances_3d_NuScenes/mAP", 0.0))))
        NDSs.append(float(acc.get("NDS", acc.get("pred_instances_3d_NuScenes/NDS", 0.0))))
        perf = item.get("perf", {})
        lat.append(perf.get("latency_ms_p50", perf.get("latency_ms_mean", 0.0)))
        mem.append(perf.get("peak_mem_mb", 0.0))

    order = np.argsort(np.array(chunks))
    chunks = list(np.array(chunks)[order])
    mAPs = list(np.array(mAPs)[order])
    NDSs = list(np.array(NDSs)[order])
    lat = list(np.array(lat)[order])
    mem = list(np.array(mem)[order])

    save_pdf_plot(chunks, mAPs, "attn_chunk", "mAP", "nuScenes mAP vs attn_chunk",
                  os.path.join(args.work_dir, "mAP_vs_attn_chunk.pdf"))
    save_pdf_plot(chunks, NDSs, "attn_chunk", "NDS", "nuScenes NDS vs attn_chunk",
                  os.path.join(args.work_dir, "NDS_vs_attn_chunk.pdf"))
    save_pdf_plot(chunks, lat, "attn_chunk", "Latency (ms, p50)", "Latency vs attn_chunk",
                  os.path.join(args.work_dir, "latency_vs_attn_chunk.pdf"))
    save_pdf_plot(chunks, mem, "attn_chunk", "Peak GPU Mem (MB)", "Memory vs attn_chunk",
                  os.path.join(args.work_dir, "memory_vs_attn_chunk.pdf"))

    import csv
    tsv_path = os.path.join(args.work_dir, "summary.tsv")
    with open(tsv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(["attn_chunk", "mAP", "NDS", "latency_ms_p50", "peak_mem_mb"])
        for c, a, n, l, m in zip(chunks, mAPs, NDSs, lat, mem):
            w.writerow([c, a, n, l, m])
    print(f"[OK] Wrote {tsv_path}")

if __name__ == "__main__":
    main()
