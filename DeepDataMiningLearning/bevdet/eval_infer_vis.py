# eval_infer_vis.py
# Minimal, evaluation-style inference over arbitrary LiDAR+Image folders using MMDet3D inferencers.
# - Supports attn_chunk sweep if model exposes it (e.g., Cross-Attn LSS)
# - Collects latency + peak GPU memory
# - Saves JSON per-run and summary
# - Optional qualitative visualizations via vis_open3d_utils.py
#
# Requires: torch, numpy, mmdet3d>=1.1 (for Inferencers), open3d (optional), pillow (optional)

from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

# Select inferencer class based on modality
try:
    from mmdet3d.apis import (
        LidarDet3DInferencer,
        MonoDet3DInferencer,
        MultiModalityDet3DInferencer
    )
except Exception as e:
    print("[FATAL] MMDet3D inferencers not found. Please install compatible mmdet3d.", e)
    sys.exit(1)

from vis_open3d_utils import (
    visualize_with_open3d,
)

# ----------------------------
# Basic utils
# ----------------------------

def ensure_dir(d: str) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)

def save_json(obj: Any, fp: str) -> None:
    with open(fp, "w") as f:
        json.dump(obj, f, indent=2)

def set_determinism(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Dataset file pairing
# ----------------------------

LIDAR_EXTS = {".bin", ".npy", ".npz", ".pcd", ".ply"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
CALIB_EXTS = {".json", ".npz", ".npy"}

def collect_inputs(
    lidar_dir: Optional[str],
    image_dir: Optional[str],
    calib_dir: Optional[str],
    max_count: int = 0
) -> List[Dict[str, str]]:
    """
    Return a list of dicts with paths for 'points', 'img', 'calib' if available.
    Pairs by basename if dirs provided; otherwise emits what's available.
    """
    pairs: List[Dict[str, str]] = []

    def list_files(root: Optional[str], exts: set) -> Dict[str, str]:
        if not root:
            return {}
        rootp = Path(root)
        out: Dict[str, str] = {}
        for p in sorted(rootp.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                out[p.stem] = str(p)
        return out

    lid = list_files(lidar_dir, LIDAR_EXTS)
    img = list_files(image_dir, IMG_EXTS)
    cal = list_files(calib_dir, CALIB_EXTS)

    # If LiDAR exists, use it as primary; else try images
    if lid:
        for stem, lpath in lid.items():
            entry = {"points": lpath}
            if stem in img:
                entry["img"] = img[stem]
            if stem in cal:
                entry["calib"] = cal[stem]
            pairs.append(entry)
    elif img:
        for stem, ipath in img.items():
            entry = {"img": ipath}
            if stem in cal:
                entry["calib"] = cal[stem]
            pairs.append(entry)
    else:
        print("[WARN] No LiDAR or images found.")
        return []

    if max_count > 0:
        pairs = pairs[:max_count]
    return pairs


# ----------------------------
# Model helpers
# ----------------------------

def build_inferencer(modality: str, config: str, checkpoint: str, device: str):
    if modality == "lidar":
        return LidarDet3DInferencer(config, checkpoint, device=device)
    if modality == "mono":
        return MonoDet3DInferencer(config, checkpoint, device=device)
    if modality in ["multi-modal", "multimodal", "camera-lidar"]:
        return MultiModalityDet3DInferencer(config, checkpoint, device=device)
    raise ValueError(f"Unknown modality: {modality}")

def get_model_from_inferencer(inferencer) -> torch.nn.Module:
    # Inferencers keep .model (may be wrapped by DataParallel/DistributedDataParallel)
    model = getattr(inferencer, "model", None)
    if model is None:
        raise RuntimeError("Inferencer has no .model attribute.")
    if hasattr(model, "module"):
        return model.module
    return model

def set_attn_chunk_if_available(model: torch.nn.Module, attn_chunk: Optional[int]) -> bool:
    """
    Try to set model.view_transform.attn_chunk (or any submodule exposing this attr).
    Returns True if attribute is found and set.
    """
    if attn_chunk is None:
        return False
    # direct path
    vt = getattr(model, "view_transform", None)
    if vt is not None and hasattr(vt, "attn_chunk"):
        setattr(vt, "attn_chunk", int(attn_chunk))
        return True
    # scan submodules
    for name, m in model.named_modules():
        if hasattr(m, "attn_chunk"):
            try:
                setattr(m, "attn_chunk", int(attn_chunk))
                return True
            except Exception:
                pass
    return False


# ----------------------------
# Prediction parsing
# ----------------------------

def parse_predictions(predictions: Any) -> Dict[str, Any]:
    """
    Normalize one-sample prediction output to a dict with bboxes_3d, scores_3d, labels_3d.
    Works with MMDet3D inferencers (returns dict) or Det3DDataSample-like structures.
    """
    if isinstance(predictions, dict):
        # Inferencers usually return {'predictions':[...], 'visualization':...}
        preds_list = predictions.get("predictions", None)
        if isinstance(preds_list, list) and len(preds_list) > 0:
            pred = preds_list[0]
            # Expect keys: 'bboxes_3d', 'scores_3d', 'labels_3d'
            out = {
                "bboxes_3d": np.array(pred.get("bboxes_3d", []), dtype=np.float32),
                "scores_3d": np.array(pred.get("scores_3d", []), dtype=np.float32),
                "labels_3d": np.array(pred.get("labels_3d", []), dtype=np.int32),
                "metainfo": pred.get("metainfo", {})
            }
            return out
        # Or a single-sample dict
        out = {
            "bboxes_3d": np.array(predictions.get("bboxes_3d", []), dtype=np.float32),
            "scores_3d": np.array(predictions.get("scores_3d", []), dtype=np.float32),
            "labels_3d": np.array(predictions.get("labels_3d", []), dtype=np.int32),
            "metainfo": predictions.get("metainfo", {})
        }
        return out

    # Try Det3DDataSample-like: predictions.pred_instances_3d
    inst = getattr(predictions, "pred_instances_3d", None)
    if inst is not None:
        b = getattr(inst, "bboxes_3d", None)
        s = getattr(inst, "scores_3d", None)
        l = getattr(inst, "labels_3d", None)
        def to_np(x):
            if x is None:
                return np.zeros((0,), dtype=np.float32)
            # boxes may be a BaseInstance3DBoxes; try .tensor or .gravity_center+dims+yaw
            if hasattr(x, "tensor"):
                arr = x.tensor
            else:
                arr = x
            if hasattr(arr, "cpu"):
                arr = arr.cpu().numpy()
            return np.asarray(arr)
        return {
            "bboxes_3d": to_np(b),
            "scores_3d": to_np(s),
            "labels_3d": to_np(l),
            "metainfo": getattr(predictions, "metainfo", {})
        }

    # Fallback empty
    return {"bboxes_3d": np.zeros((0, 7), dtype=np.float32), "scores_3d": np.zeros((0,), dtype=np.float32), "labels_3d": np.zeros((0,), dtype=np.int32)}


# ----------------------------
# Inference loop (perf + viz)
# ----------------------------

def run_inference_once(
    inferencer,
    single_input: Dict[str, str],
    score_thr: float = 0.0
) -> Dict[str, Any]:
    """
    Call inferencer on one sample. Input dict may include:
      - 'points': LiDAR file path
      - 'img'   : image file path (multi-modal)
      - other keys ignored by inferencer
    Returns raw inferencer result dict.
    """
    # Pass full dict to let inferencer choose relevant keys
    res = inferencer(single_input, out_dir=None, show=False, pred_score_thr=score_thr)
    return res


def measure_latency_and_mem(
    fn, warmup_iters: int, measure_iters: int
) -> Dict[str, float]:
    """
    Measure latency and GPU peak memory during repeated calls to fn().
    """
    lat = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    # Warmup
    for _ in range(max(0, warmup_iters)):
        _ = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    # Measure
    for _ in range(max(1, measure_iters)):
        t0 = time.perf_counter()
        _ = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000.0)

    stats = {
        "latency_ms_mean": float(np.mean(lat)),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p90": float(np.percentile(lat, 90)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "latency_ms_p99": float(np.percentile(lat, 99)),
        "latency_ms_min": float(np.min(lat)),
        "latency_ms_max": float(np.max(lat)),
        "num_measured": int(len(lat)),
        "peak_mem_bytes": 0,
        "peak_mem_mb": 0.0
    }
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        stats["peak_mem_bytes"] = int(peak)
        stats["peak_mem_mb"] = float(peak / (1024.0 * 1024.0))
    return stats


def run_eval_for_chunk(
    config: str,
    checkpoint: str,
    modality: str,
    device: str,
    inputs: List[Dict[str, str]],
    out_dir: str,
    attn_chunk: Optional[int],
    warmup_iters: int,
    max_samples: int,
    viz: bool,
    max_viz: int,
    headless: bool,
    score_thr: float
) -> Dict[str, Any]:
    """
    Build inferencer, set chunk if available, run latency/memory on N samples, save per-run JSON.
    Also dump qualitative visualizations.
    """
    tag = f"attn_{attn_chunk}" if attn_chunk is not None else "attn_none"
    run_dir = Path(out_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    inferencer = build_inferencer(modality, config, checkpoint, device)
    model = get_model_from_inferencer(inferencer)
    chunk_supported = set_attn_chunk_if_available(model, attn_chunk)

    # If measuring, define callable that cycles through samples
    sample_count = len(inputs) if max_samples <= 0 else min(max_samples, len(inputs))
    # choose a fixed subset deterministically
    idxs = list(range(sample_count))

    it = {"i": 0}

    def _call():
        i = it["i"]
        single_input = inputs[i % sample_count]
        it["i"] += 1
        return run_inference_once(inferencer, single_input, score_thr=score_thr)

    perf = measure_latency_and_mem(
        _call,
        warmup_iters=warmup_iters,
        measure_iters=max(1, sample_count)
    )

    # Save first result's prediction for reference (non-quantitative)
    sample_pred = {}
    try:
        first = inputs[0]
        res = run_inference_once(inferencer, first, score_thr=score_thr)
        parsed = parse_predictions(res)
        # keep top-k (optional)
        sample_pred = {
            "file": first.get("points", first.get("img", "")),
            "num_preds": int(parsed["bboxes_3d"].shape[0])
        }
    except Exception as e:
        sample_pred = {"error": str(e)}

    out = {
        "attn_chunk": attn_chunk if chunk_supported else None,
        "attn_chunk_supported": bool(chunk_supported),
        "perf": perf,
        "sample_pred": sample_pred,
        "work_dir": str(run_dir)
    }
    save_json(out, str(run_dir / "result.json"))
    print(f"[OK] Saved: {run_dir/'result.json'}")

    # Qualitative visualization
    if viz:
        qual_dir = run_dir / "qualitative"
        qual_dir.mkdir(parents=True, exist_ok=True)
        vis_count = min(max_viz, len(inputs))
        for j in range(vis_count):
            single = inputs[j]
            try:
                res = run_inference_once(inferencer, single, score_thr=score_thr)
                pred = parse_predictions(res)
                # Visualization expects LiDAR file. If missing, skip 3D view.
                lidar_file = single.get("points", None)
                img_file = single.get("img", None)
                calib_file = single.get("calib", None)
                gt_bboxes = []  # user can extend to pass GT if available

                base = Path(lidar_file if lidar_file else img_file).stem
                visualize_with_open3d(
                    lidar_file=lidar_file if lidar_file else img_file,  # if only image, still proceed (2D only)
                    predictions_dict=pred,
                    gt_bboxes=gt_bboxes,
                    out_dir=str(qual_dir),
                    basename=base,
                    headless=headless,
                    img_file=img_file,
                    calib_file=calib_file
                )
            except Exception as e:
                print(f"[WARN] Visualization failed for sample {j}: {e}")

    return out


# ----------------------------
# Summary aggregation
# ----------------------------

def collect_runs_from_dir(root: str) -> List[dict]:
    """
    Walk 'root' and gather any */result.json. If a 'summary.json' exists, merge it as well.
    Deduplicate by (work_dir, attn_chunk).
    """
    rootp = Path(root)
    runs: List[dict] = []
    for rp in rootp.rglob("result.json"):
        try:
            with open(rp, "r") as f:
                runs.append(json.load(f))
        except Exception as e:
            print(f"[WARN] Failed to read {rp}: {e}")

    summ = rootp / "summary.json"
    if summ.is_file():
        try:
            with open(summ, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                runs.extend(data)
            elif isinstance(data, dict):
                runs.append(data)
        except Exception as e:
            print(f"[WARN] Failed to read {summ}: {e}")

    # dedup
    uniq = {}
    for r in runs:
        key = (r.get("work_dir"), str(r.get("attn_chunk")))
        uniq[key] = r
    return list(uniq.values())

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


# ----------------------------
# Main
# ----------------------------
from dataset_resolver import make_iterator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="work_dirs/mybevfusion7_new/mybevfusion7_crossattnaux_painting.py", help="Path to MMDet3D config .py")
    ap.add_argument("--checkpoint", type=str, default="work_dirs/mybevfusion7_new/epoch_4.pth", help="Path to model checkpoint .pth")
    ap.add_argument("--modality", type=str, default="multi-modal",
                    choices=["lidar", "mono", "multi-modal", "multimodal", "camera-lidar"])
    ap.add_argument("--device", type=str, default="cuda:0")
    # Dataset mode
    ap.add_argument('--dataset', type=str, default='nuscenes',
                choices=['any','kitti','waymokitti','nuscenes'])
    # Common path
    ap.add_argument('--input-path', type=str, default='data/nuscenes')  # for kitti/waymokitti: dataset root; for nuscenes: dataroot

    # Frame selection
    ap.add_argument('--frame-number', type=str, default='-1', help="KITTI stem like '000008'; '-1' for all (kitti/waymokitti)")
    ap.add_argument('--limit', type=int, default=10, help='Limit number of samples (any/nuscenes)')

    # ANY mode folders
    ap.add_argument('--lidar-dir', type=str, default=None)
    ap.add_argument('--image-dir', type=str, default=None)
    ap.add_argument('--calib-dir', type=str, default=None)

    # nuScenes extras
    ap.add_argument('--nus-version', type=str, default='v1.0-trainval') #'v1.0-mini'
    ap.add_argument('--nus-camera', type=str, default='CAM_FRONT')
    ap.add_argument('--nus-tokens-file', type=str, default=None)
    
    ap.add_argument("--attn-chunks", type=int, nargs="*", default=[4096, 8192, 16384])
    ap.add_argument("--warmup-iters", type=int, default=5)
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default="eval_vis_outputs")
    ap.add_argument("--eval", action="store_true", default=True, help="Run latency/mem measurement.")
    ap.add_argument("--viz", action="store_true", default=True, help="Save qualitative visualizations.")
    ap.add_argument("--max-viz", type=int, default=8, help="Number of qualitative frames to dump per run.")
    ap.add_argument("--headless", action="store_true", default=True, help="Save PLYs instead of opening Open3D window.")
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-plot", action="store_true", help="Skip eval; just aggregate existing results.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    set_determinism(args.seed)

    # Decide how many samples to materialize
    # - eval mode: respect --max-samples if >0 else use --limit (or all)
    # - viz/non-eval: respect --max-viz
    iter_limit = None
    if args.eval:
        iter_limit = args.max_samples if args.max_samples > 0 else args.limit
    else:
        iter_limit = args.max_viz if args.max_viz > 0 else args.limit

    it = make_iterator(
        dataset=args.dataset,                 # {'any','kitti','waymokitti','nuscenes'}
        input_path=args.input_path,           # dataset root (kitti/waymokitti/nuscenes) or ignored for 'any'
        frame_number=getattr(args, "frame_number", "-1"),
        # ANY-mode folders (used only when dataset='any')
        lidar_dir=args.lidar_dir,
        image_dir=args.image_dir,
        calib_dir=args.calib_dir,
        # nuScenes specifics
        nus_version=getattr(args, "nus_version", "v1.0-mini"),
        nus_camera=getattr(args, "nus_camera", "CAM_FRONT"),
        nus_limit=iter_limit if (iter_limit is not None and iter_limit > 0) else -1,
        nus_tokens_file=getattr(args, "nus_tokens_file", None),
        # KITTI-like specifics
        split=getattr(args, "kitti_split", "training"),
        use_cam=getattr(args, "kitti_cam", "image_2"),
    )

    inputs: List[Dict[str, Any]] = []
    for li, im, calib, base in it:
        inputs.append({
            "lidar": li,         # required
            "img": im,           # may be None
            "calib": calib,      # dict with 'lidar2img' (3x4) or None
            "basename": base
        })
        # Hard cap here as a secondary guard
        if args.eval and args.max_samples > 0 and len(inputs) >= args.max_samples:
            break
        if (not args.eval) and args.max_viz > 0 and len(inputs) >= args.max_viz:
            break

    if not inputs:
        print("[ERROR] No inputs collected. "
            "For dataset='any', set --lidar-dir (and optional --image-dir/--calib-dir). "
            "For 'kitti'/'waymokitti'/'nuscenes', set --input-path correctly.")
        sys.exit(1)

    summary: List[Dict[str, Any]] = []

    # --- Evaluation mode (runs all requested chunks) ---
    if args.eval and not args.only_plot:
        chunks = args.attn_chunks if (args.attn_chunks and len(args.attn_chunks) > 0) else [None]
        for chunk in chunks:
            print("=" * 80)
            print(f"[RUN] attn_chunk={chunk}")
            out = run_eval_for_chunk(
                config=args.config,
                checkpoint=args.checkpoint,
                modality=args.modality,
                device=args.device,
                inputs=inputs if args.max_samples <= 0 else inputs[:args.max_samples],
                out_dir=args.out_dir,
                attn_chunk=chunk,
                warmup_iters=args.warmup_iters,
                max_samples=args.max_samples,
                viz=args.viz,
                max_viz=args.max_viz,
                headless=args.headless,
                score_thr=args.score_thr
            )
            summary.append(out)

        save_json(summary, str(Path(args.out_dir) / "summary.json"))
        print(f"[OK] Wrote {Path(args.out_dir) / 'summary.json'}")

    # # --- Optional: plots/TSV only (re-use saved JSON without re-evaluating) ---
    # if args.only_plot:
    #     # This will scan args.out_dir for subfolders like attn_4096/result.json
    #     runs = collect_runs_from_dir(args.out_dir)
    #     if len(runs) == 0:
    #         print("[WARN] No existing results found for plotting. "
    #             "Run with --eval first, or place result.json files under subfolders (e.g., out/attn_4096/result.json).")
    #     else:
    #         # 1) comprehensive metrics chart for this experiment
    #         save_comprehensive_metrics_bar(runs, args.out_dir)
    #         # 2) per-chunk comparisons
    #         build_plots_and_tsv(runs, args.out_dir)
    #         print("[OK] Plots and TSV generated.")


if __name__ == "__main__":
    main()