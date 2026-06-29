"""
ngperception.depth.run_eval
===========================

The single CLI for the depth task: run one or more depth models over KITTI, score them
with LiDAR GT, and write a comparison table + chart. Optionally fuse with an ngdet
detector to render **per-object distance** images.

Examples
--------
# Compare a relative model vs a metric model on 200 KITTI frames:
python -m DeepDataMiningLearning.ngperception.depth.run_eval \
    --models hf_depth:depth-anything/Depth-Anything-V2-Small-hf \
             hf_depth:Intel/zoedepth-kitti \
    --max-images 200 --out-dir DeepDataMiningLearning/ngperception/output/depth

# Detection + depth fusion visualization (boxes labeled with metres):
python -m DeepDataMiningLearning.ngperception.depth.run_eval \
    --models hf_depth:Intel/zoedepth-kitti --max-images 5 --viz \
    --detector hf_detr:facebook/detr-resnet-50
"""

from __future__ import annotations
import argparse
import json
import os

import numpy as np

from .datasets import KITTIDepthDataset, DEFAULT_KITTI_ROOT
from .evaluator import DepthEvaluator
from .estimators.base import build_estimator


def _bar_chart(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [r["model"].split(":")[-1].split("/")[-1] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.barh(labels, [r["AbsRel"] for r in rows], color="#d1495b")
    a1.set_xlabel("AbsRel (lower better)"); a1.invert_yaxis()
    a2.barh(labels, [r["delta1"] for r in rows], color="#2e8b57")
    a2.set_xlabel("δ<1.25 (higher better)"); a2.invert_yaxis()
    fig.suptitle("KITTI monocular depth (LiDAR GT)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _write_report(rows, path):
    lines = ["# ngperception/depth — KITTI comparison\n",
             "**Aligned** = median scale-invariant (fair across relative & metric). "
             "**Metric** = true metres, no alignment (only meaningful for metric models).\n",
             "| model | type | aligned AbsRel↓ | aligned RMSE↓ | aligned δ1↑ "
             "| metric AbsRel↓ | metric RMSE↓ | metric δ1↑ |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = "metric" if r["metric"] else "relative"
        def mc(k):
            return f"{r['m_'+k]:.3f}" if ("m_" + k) in r else "—"
        lines.append(
            f"| {r['model']} | {t} | {r['AbsRel']:.3f} | {r['RMSE']:.2f} | {r['delta1']:.3f} "
            f"| {mc('AbsRel')} | {mc('RMSE')} | {mc('delta1')} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _build_det(det_spec, device):
    from DeepDataMiningLearning.ngdet.taxonomy import Taxonomy
    from DeepDataMiningLearning.ngdet.detectors.base import build_detector
    return build_detector(det_spec, Taxonomy.from_preset("driving3"),
                          device=device, score_thr=0.3)


def _compose_frame(estimator, det, sample):
    """One fused frame: distance-annotated RGB (top) + depth colormap (bottom), as RGB uint8."""
    import matplotlib.cm as cm
    from .fusion import distances_from_detection, draw_distances
    dr = estimator.predict(sample.image)
    detection = det.predict(sample.image)
    objs = distances_from_detection(detection, dr)
    annotated = np.asarray(draw_distances(sample.image, objs))
    dn = np.clip(dr.depth, 0, 80) / 80.0
    colored = (cm.magma(1 - dn)[:, :, :3] * 255).astype(np.uint8)   # bright = near
    return np.vstack([annotated, colored])


def _viz(estimator, det_spec, ds, out_dir, n=5, tag=""):
    """Render per-object distance images by fusing a detector with the depth map."""
    import matplotlib
    matplotlib.use("Agg")
    from PIL import Image
    det = _build_det(det_spec, estimator.device)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(min(n, len(ds))):
        s = ds[i]
        Image.fromarray(_compose_frame(estimator, det, s)).save(
            os.path.join(out_dir, f"fuse_{tag}_{s.sample_id}.png"))
    print(f"[viz] wrote {min(n, len(ds))} fusion images ({tag}) -> {out_dir}")


def _video(estimator, det_spec, ds, out_path, n=30, fps=5):
    """Render an mp4 of detection+distance+depth over the first `n` frames."""
    import matplotlib
    matplotlib.use("Agg")
    import cv2
    det = _build_det(det_spec, estimator.device)
    n = min(n, len(ds))
    H = W = None
    writer = None
    for i in range(n):
        frame = _compose_frame(estimator, det, ds[i])             # RGB uint8, 2H x W
        if writer is None:
            H, W = frame.shape[:2]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (W, H))
        if frame.shape[:2] != (H, W):                             # KITTI sizes vary slightly
            frame = cv2.resize(frame, (W, H))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if writer is not None:
        writer.release()
        print(f"[video] wrote {n} frames @ {fps}fps -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="ngperception depth comparison.")
    ap.add_argument("--models", nargs="+", required=True,
                    help="depth specs, e.g. hf_depth:Intel/zoedepth-kitti")
    ap.add_argument("--dataset", default="kitti")
    ap.add_argument("--root", default=DEFAULT_KITTI_ROOT)
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-garg-crop", action="store_true",
                    help="disable the standard KITTI center crop")
    ap.add_argument("--out-dir", default="DeepDataMiningLearning/ngperception/output/depth")
    ap.add_argument("--viz", action="store_true", help="also render fusion PNGs")
    ap.add_argument("--video", action="store_true",
                    help="also render an mp4 of detection+distance+depth")
    ap.add_argument("--video-frames", type=int, default=30)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--detector", default="hf_detr:facebook/detr-resnet-50",
                    help="ngdet detector spec used for --viz/--video fusion")
    args = ap.parse_args()

    if args.dataset != "kitti":
        raise SystemExit("only 'kitti' is wired so far")
    ds = KITTIDepthDataset(args.root, max_images=args.max_images,
                           offset=args.offset, stride=args.stride)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[depth] {len(ds)} KITTI frames | models: {args.models}")

    rows = []
    for spec in args.models:
        est = build_estimator(spec, device=args.device)
        ev = DepthEvaluator(is_metric=est.is_metric, garg_crop=not args.no_garg_crop)
        for i in range(len(ds)):
            s = ds[i]
            ev.add(est.predict(s.image).depth, s.depth_gt)
        m = ev.summarize(verbose=False)
        m["model"] = spec
        rows.append(m)
        t = "metric" if est.is_metric else "relative"
        extra = (f" | metric AbsRel={m['m_AbsRel']:.3f} d1={m['m_delta1']:.3f}"
                 if "m_AbsRel" in m else "")
        print(f"  {spec:52s} [{t:8s}] aligned AbsRel={m['AbsRel']:.3f} "
              f"d1={m['delta1']:.3f}{extra}")
        # fusion viz/video only for metric models (per-object distance needs true metres)
        if est.is_metric:
            tag = spec.split(":")[-1].split("/")[-1]
            if args.viz:
                _viz(est, args.detector, ds, os.path.join(args.out_dir, "viz"), n=5, tag=tag)
            if args.video:
                _video(est, args.detector, ds,
                       os.path.join(args.out_dir, f"depth_fusion_{tag}.mp4"),
                       n=args.video_frames, fps=args.fps)
        del est
        import torch; torch.cuda.empty_cache()

    json.dump(rows, open(os.path.join(args.out_dir, "depth_metrics.json"), "w"), indent=2)
    _write_report(rows, os.path.join(args.out_dir, "depth_report.md"))
    _bar_chart(rows, os.path.join(args.out_dir, "depth_bars.png"))
    print(f"[depth] wrote report + chart -> {args.out_dir}")


if __name__ == "__main__":
    main()
