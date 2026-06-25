"""
ngdet.latency
============

Latency / throughput benchmark for the detectors, comparing the **native
PyTorch / HuggingFace / Ultralytics** path against available **accelerated**
backends (FP16, torch.compile, TensorRT, ONNX).

What it measures
----------------
End-to-end `detector.predict(image)` wall-clock latency (image in -> unified
detections out), with proper CUDA synchronization, after warmup. We report
mean / p50 / p90 latency (ms), throughput (FPS), and the speedup vs the model's
own FP32 baseline. End-to-end is what a user feels (it includes pre/post-proc,
not just the GPU forward) -- the honest number to compare.

Acceleration support per family (modes a model can't do are skipped):
  * fp32      : native baseline (all)
  * fp16      : half precision (torchvision, hf_detr, yolo)
  * compile   : torch.compile reduce-overhead (torchvision, hf_detr)
  * tensorrt  : Ultralytics native .engine export (yolo)
  * onnx      : Ultralytics native .onnx export (yolo)

DETR / RT-DETR `onnx` and `tensorrt` modes use **onnxruntime-gpu** (ONNX export +
CUDA or TensorRT Execution Provider). They need the GPU EP libraries on the loader
path; onnxruntime-gpu finds torch's bundled CUDA/cuDNN libs if you export:

    SP=$(python -c "import site; print(site.getsitepackages()[0])")
    export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH

before running. Without it, the CUDA EP can't load (libcudnn_adv.so.9 missing) and
those rows are skipped.

Not wired (documented): torch-tensorrt (not installed), OpenVINO (not installed),
and vLLM serving for the LocateAnything VLM (vLLM lacks this custom arch; it would
be the serving path).
"""

from __future__ import annotations
import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np

from .taxonomy import Taxonomy
from .detectors.base import build_detector


def _load_image(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def time_predict(detector, image, warmup: int = 5, iters: int = 30) -> Dict:
    """Time detector.predict(image): warmup, then `iters` synchronized runs."""
    import torch
    cuda = torch.cuda.is_available()

    for _ in range(warmup):              # warmup (also triggers compile/engine build)
        detector.predict(image)
    if cuda:
        torch.cuda.synchronize()

    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        detector.predict(image)
        if cuda:
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)   # ms
    a = np.array(lat, dtype=float)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "std_ms": float(a.std()),
        "fps": float(1000.0 / a.mean()),
    }


def benchmark(model_specs: List[str], accels: List[str], image, taxonomy,
              device: str, warmup: int, iters: int) -> List[Dict]:
    """Benchmark every (model, accel) pair; fp32 first so we can compute speedups."""
    rows = []
    fp32_fps = {}
    accels = sorted(set(accels), key=lambda x: (x != "fp32", x))  # fp32 first
    for spec in model_specs:
        for accel in accels:
            label = f"{spec.split(':',1)[0]}:{spec.split(':',1)[-1].split('/')[-1]}"
            try:
                det = build_detector(spec, taxonomy, device=device,
                                     score_thr=0.3, accel=accel)
                stats = time_predict(det, image, warmup, iters)
            except Exception as e:  # noqa: BLE001 - unsupported accel/model -> skip
                print(f"  [skip] {label} [{accel}]: {str(e).splitlines()[0][:80]}")
                continue
            if accel == "fp32":
                fp32_fps[spec] = stats["fps"]
            stats["speedup"] = (stats["fps"] / fp32_fps[spec]
                                if spec in fp32_fps else float("nan"))
            stats.update({"model": label, "spec": spec, "accel": accel})
            rows.append(stats)
            print(f"  {label:40s} [{accel:8s}] "
                  f"{stats['mean_ms']:7.1f} ms  {stats['fps']:7.1f} FPS  "
                  f"x{stats['speedup']:.2f}")
            # free GPU memory between variants
            del det
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    return rows


def write_report(rows: List[Dict], out_dir: str, device_name: str):
    """Write latency.json + latency_report.md + latency_fps.png."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "latency.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # --- markdown table ---
    L = [f"# ngdet latency benchmark\n",
         f"Device: **{device_name}**. End-to-end `predict()` latency "
         f"(includes pre/post-processing).\n",
         "| model | accel | mean ms | p50 ms | p90 ms | FPS | speedup |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['accel']} | {r['mean_ms']:.1f} | "
                 f"{r['p50_ms']:.1f} | {r['p90_ms']:.1f} | {r['fps']:.1f} | "
                 f"x{r['speedup']:.2f} |")
    md = os.path.join(out_dir, "latency_report.md")
    with open(md, "w") as f:
        f.write("\n".join(L))

    # --- grouped bar chart: FPS per model, bars = accel modes ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    accels = []
    for r in rows:
        if r["accel"] not in accels:
            accels.append(r["accel"])
    fps = {(r["model"], r["accel"]): r["fps"] for r in rows}

    x = np.arange(len(models))
    w = 0.8 / max(1, len(accels))
    fig, ax = plt.subplots(figsize=(1.5 * len(models) + 4, 5))
    for k, accel in enumerate(accels):
        vals = [fps.get((m, accel), 0.0) for m in models]
        ax.bar(x + k * w, vals, w, label=accel)
    ax.set_xticks(x + 0.4 - w / 2, models, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Throughput (FPS, higher=better)")
    ax.set_title(f"Native vs accelerated inference ({device_name})")
    ax.legend(title="backend")
    fig.tight_layout()
    png = os.path.join(out_dir, "latency_fps.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return md, png


def main():
    ap = argparse.ArgumentParser(description="Detector latency / throughput benchmark.")
    ap.add_argument("--models", nargs="+", required=True,
                    help="detector specs, e.g. yolo:yolo11x.pt hf_detr:facebook/detr-resnet-50")
    ap.add_argument("--accel", nargs="+", default=["fp32", "fp16"],
                    help="accel modes: fp32 fp16 compile tensorrt onnx")
    ap.add_argument("--image", default="/mnt/e/Shared/Dataset/Kitti/training/image_2/000008.png",
                    help="image to benchmark on")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--taxonomy", default="driving3")
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "output"))
    a = ap.parse_args()

    import torch
    dev_name = (torch.cuda.get_device_name(0)
                if (a.device.startswith("cuda") and torch.cuda.is_available()) else "CPU")
    tax = Taxonomy.from_preset(a.taxonomy)
    image = _load_image(a.image)
    print(f"Benchmarking on {dev_name}, image {image.size}, "
          f"warmup={a.warmup} iters={a.iters}")
    rows = benchmark(a.models, a.accel, image, tax, a.device, a.warmup, a.iters)
    if rows:
        md, png = write_report(rows, a.out_dir, dev_name)
        print(f"\nSaved {md}\nSaved {png}")


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# Compare native vs FP16 vs TensorRT for YOLO, and native vs FP16 vs compile for
# DETR / Faster R-CNN, on a GPU:
#
#   python -m DeepDataMiningLearning.ngdet.latency \
#       --models yolo:yolo11x.pt hf_detr:facebook/detr-resnet-50 \
#                torchvision:fasterrcnn_resnet50_fpn_v2 \
#       --accel fp32 fp16 compile tensorrt --iters 30
#
# Writes latency_report.md + latency_fps.png + latency.json to ngdet/output/.
# (TensorRT/ONNX only apply to yolo; unsupported (model,accel) pairs are skipped.)
# ===========================================================================
if __name__ == "__main__":
    main()
