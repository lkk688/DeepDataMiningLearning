"""
ngdet.run_eval
=============

THE single unified entry point. Evaluate one or more detectors on one or more
datasets, zero-shot (no training), producing:

    * a COCO mAP table (unified taxonomy) printed to console + saved as JSON
    * one annotated mp4 per (model, dataset) pair (predictions + GT)
    * a per-image predictions JSON for offline analysis

This is the teaching/baseline driver: students plug in a model spec and a dataset
name and immediately see how a pretrained detector generalizes across domains.

Run `python -m DeepDataMiningLearning.ngdet.run_eval --help` for all options.
"""

from __future__ import annotations
import argparse
import json
import os
import time
from typing import Dict, List

from .taxonomy import Taxonomy
from .datasets import EvalDataset
from .evaluator import COCOUnifiedEvaluator
from .detectors.base import build_detector
from .report import write_markdown_report, plot_heatmap

# Default output dir lives inside the package (ngdet/output, git-ignored).
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Default dataset roots on this machine; override on the CLI with --roots.
DEFAULT_ROOTS = {
    "kitti": "/mnt/e/Shared/Dataset/Kitti/",
    "waymo": "/mnt/e/Shared/Dataset/waymodata",
    "nuscenes": "/mnt/e/Shared/Dataset/NuScenes/v1.0-trainval",
    "nuimages": "/mnt/e/Shared/Dataset/NuScenes/nuimages",
    "coco": "/mnt/e/Shared/Dataset/coco2017/val2017",
}


def evaluate_one(model_spec: str, dataset_name: str, root: str, taxonomy: Taxonomy,
                 device: str, score_thr: float, max_images, make_video: bool,
                 out_dir: str, ds_kwargs: dict, video_max: int = 0) -> Dict:
    """Run one (model x dataset) pair and return its metrics dict.

    video_max: if >0 and make_video, only the first `video_max` frames are written
    to the mp4 (a small human-eval clip) while ALL images still count toward mAP.
    """
    print(f"\n{'='*70}\n[{model_spec}]  x  [{dataset_name}]\n{'='*70}")
    t0 = time.time()

    detector = build_detector(model_spec, taxonomy, device=device, score_thr=score_thr)
    dataset = EvalDataset(dataset_name, root, taxonomy, max_images=max_images,
                          **ds_kwargs)
    evaluator = COCOUnifiedEvaluator(taxonomy)

    # Clean label for tables/filenames (drop any "@thr" threshold override).
    label = model_spec.split("@")[0]

    vw = None
    if make_video:
        from .video import EvalVideoWriter
        tag = label.replace(":", "_").replace("/", "_")
        vpath = os.path.join(out_dir, f"video_{tag}__{dataset_name}.mp4")
        vw = EvalVideoWriter(vpath, taxonomy, fps=5, draw_gt=True)

    per_image = []
    for i, sample in enumerate(dataset):
        det = detector.predict(sample.image)
        evaluator.add(sample, det)          # ALL images count toward mAP
        if vw is not None and (video_max <= 0 or i < video_max):
            vw.add(sample, det)             # video may cover only a subset
        per_image.append({
            "image_id": sample.image_id,
            "num_pred": len(det), "num_gt": len(sample.gt_boxes),
        })
    if vw is not None:
        vw.release()
        print(f"  video -> {vpath}")

    metrics = evaluator.summarize()
    metrics["_meta"] = {
        "model": label, "dataset": dataset_name,
        "num_images": len(dataset), "open_vocab": detector.is_open_vocab,
        "score_thr": detector.score_thr,
        "seconds": round(time.time() - t0, 1),
    }

    # Save per-image counts for analysis.
    tag = label.replace(":", "_").replace("/", "_")
    with open(os.path.join(out_dir, f"pred_{tag}__{dataset_name}.json"), "w") as f:
        json.dump(per_image, f)
    return metrics


def print_table(all_metrics: List[Dict], taxonomy: Taxonomy):
    """Pretty-print the model x dataset mAP matrix."""
    print(f"\n{'#'*70}\n# RESULTS (unified taxonomy: {taxonomy.classes})\n{'#'*70}")
    header = f"{'model':<42}{'dataset':<11}{'mAP':>7}{'AP50':>7}{'AP75':>7}"
    print(header + "  per-class AP50:95")
    print("-" * len(header))
    for m in all_metrics:
        meta = m["_meta"]
        pc = "  ".join(f"{k}={v:.3f}" for k, v in m["per_class"].items())
        print(f"{meta['model']:<42}{meta['dataset']:<11}"
              f"{m['mAP']:>7.3f}{m['AP50']:>7.3f}{m['AP75']:>7.3f}  {pc}")


def main():
    ap = argparse.ArgumentParser(
        description="Unified zero-shot 2D detection evaluation across datasets.")
    ap.add_argument("--models", nargs="+", required=True,
                    help="detector specs 'backend:checkpoint', e.g. "
                         "hf_detr:facebook/detr-resnet-50 yolo:yolo11x")
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="dataset names: kitti waymo nuscenes coco")
    ap.add_argument("--taxonomy", default="driving3",
                    help="unified taxonomy preset (driving3 / vehicle_person / driving5)")
    ap.add_argument("--max-images", type=int, default=200,
                    help="cap images per dataset for quick runs (0 = all)")
    ap.add_argument("--waymo-stride", type=int, default=1,
                    help="subsample stride for Waymo so the sample spans many "
                         "segments (frames are grouped by segment; stride>1 is "
                         "needed to see pedestrians/cyclists, not just the "
                         "vehicle-heavy opening segment)")
    ap.add_argument("--score-thr", type=float, default=0.3,
                    help="shared confidence threshold (keep identical across models!)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video", action="store_true", help="also write annotated mp4s")
    ap.add_argument("--video-max", type=int, default=0,
                    help="if >0, only write the first N frames to each video "
                         "(human-eval clip); all images still count toward mAP")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="output dir (default: ngdet/output, git-ignored)")
    ap.add_argument("--coco-ann", default=None,
                    help="path to instances_val2017.json (required for coco)")
    ap.add_argument("--nuimages-version", default="v1.0-mini",
                    help="nuImages annotation split: v1.0-mini / v1.0-val / v1.0-train")
    ap.add_argument("--roots", nargs="*", default=[],
                    help="override dataset roots as name=path pairs")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    roots = dict(DEFAULT_ROOTS)
    for kv in a.roots:
        k, v = kv.split("=", 1)
        roots[k] = v

    taxonomy = Taxonomy.from_preset(a.taxonomy)
    max_images = a.max_images or None

    all_metrics = []
    for model_spec in a.models:
        for ds_name in a.datasets:
            ds_kwargs = {}
            if ds_name == "coco":
                if not a.coco_ann:
                    print("  [skip] coco needs --coco-ann instances_val2017.json")
                    continue
                ds_kwargs["ann_file"] = a.coco_ann
            if ds_name == "waymo" and a.waymo_stride > 1:
                ds_kwargs["stride"] = a.waymo_stride
            if ds_name == "nuimages":
                ds_kwargs["version"] = a.nuimages_version
            try:
                m = evaluate_one(model_spec, ds_name, roots[ds_name], taxonomy,
                                 a.device, a.score_thr, max_images, a.video,
                                 a.out_dir, ds_kwargs, video_max=a.video_max)
                all_metrics.append(m)
            except Exception as e:  # noqa: BLE001 - one pair failing shouldn't abort all
                import traceback
                print(f"  [error] {model_spec} x {ds_name}: {e}")
                traceback.print_exc()

    if all_metrics:
        print_table(all_metrics, taxonomy)
        with open(os.path.join(a.out_dir, "metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        # Generate the Markdown report (full COCO tables + gap) and the heatmap.
        rp = write_markdown_report(all_metrics, os.path.join(a.out_dir, "report.md"),
                                   taxonomy)
        hp = plot_heatmap(all_metrics, os.path.join(a.out_dir, "heatmap_mAP.png"))
        print(f"\nSaved metrics -> {os.path.join(a.out_dir, 'metrics.json')}")
        print(f"Saved report  -> {rp}")
        print(f"Saved heatmap -> {hp}")


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# Phase-1 minimal run: DETR + YOLO on KITTI, 50 images, with video.
# (Run from the repo root; first run downloads the two checkpoints.)
#
#   python -m DeepDataMiningLearning.ngdet.run_eval \
#       --models hf_detr:facebook/detr-resnet-50 yolo:yolo11x \
#       --datasets kitti --max-images 50 --score-thr 0.3 --video \
#       --out-dir output/ngdet
#
# Add COCO (same-domain reference) once you have val2017 + annotations:
#   ... --datasets kitti coco \
#       --coco-ann /mnt/e/Shared/Dataset/coco2017/annotations/instances_val2017.json
#
# Add open-vocab LocateAnything / Waymo / NuScenes in Phase 2:
#   --models locate_anything:nvidia/LocateAnything-3B --datasets waymo nuscenes
# ===========================================================================
if __name__ == "__main__":
    main()
