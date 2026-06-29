# ngperception — downstream perception on top of ngdet

A pluggable, **teaching-oriented** suite of *downstream* perception heads that run on the
same images (and optionally the same `ngdet.Detection` boxes) and add per-object/scene
attributes. Each task mirrors ngdet's three-layer design — **estimators / datasets /
evaluator** — so students can compare many models (**basic → SOTA**) with one command,
see the accuracy trade-offs, and then **fine-tune**.

| task | folder | status | models (basic → SOTA) | metric |
|---|---|---|---|---|
| **Depth / distance** | [depth/](depth/) | ✅ built | MiDaS/DPT → Depth-Anything-V2 → ZoeDepth (metric) | AbsRel / RMSE / δ1 |
| **3D occupancy** | [occupancy/](occupancy/) | ✅ baseline | depth→voxel lift (ViPOcc-style) → learned occ nets (planned) | mIoU / geo-IoU (Occ3D) |
| Segmentation | `segmentation/` | planned | DeepLabv3 → SegFormer → Mask2Former → OneFormer/SAM2 | mIoU / PQ |
| Tracking (MOT) | `tracking/` | planned | SORT → DeepSORT → ByteTrack → OC-SORT/BoT-SORT | MOTA / IDF1 / HOTA |
| Lane detection | `lane/` | planned | UFLD → CLRNet → CLRerNet | F1 (TuSimple/CULane) |

It reuses ngdet's datasets/taxonomy/detectors for **composable fusion** (e.g. detection →
per-object distance), rather than duplicating them.

## Task 1 — Monocular depth & per-object distance

### Why two model *types*

- **Relative** models (MiDaS/DPT, Depth-Anything-V2) predict depth only **up to scale**
  (and as inverse depth). They are aligned to GT scale before scoring — cheap, general,
  but not directly metric.
- **Metric** models (ZoeDepth, Depth-Anything-V2-Metric) predict absolute **meters** —
  what you need for "the car is 18 m away".

The adapter normalizes both to *larger = farther*; the evaluator median-aligns relative
models (Monodepth2 protocol). See [TUTORIAL.md](TUTORIAL.md) §1.

### Ground truth

KITTI GT depth is built by **projecting the Velodyne LiDAR into the left color camera**
with the per-frame calib (`X_cam = R0_rect · Tr_velo_to_cam · X_velo`; `depth = X_cam.z`).
This is sparse (~4–5 % of pixels) — exactly the KITTI depth-benchmark definition.

### Commands

```bash
# Compare a basic relative model, a SOTA relative model, and a metric model:
python -m DeepDataMiningLearning.ngperception.depth.run_eval \
    --models hf_depth:Intel/dpt-hybrid-midas \
             hf_depth:depth-anything/Depth-Anything-V2-Small-hf \
             hf_depth:Intel/zoedepth-kitti \
    --max-images 200 --out-dir DeepDataMiningLearning/ngperception/output/depth

# Detection + depth fusion -> boxes labelled with metres (composable add-on):
#   --viz   : PNG stills      --video : an mp4 of detection+distance+depth
python -m DeepDataMiningLearning.ngperception.depth.run_eval \
    --models hf_depth:depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf \
    --max-images 30 --stride 1 --viz --video --video-frames 30 --fps 5 \
    --detector hf_detr:facebook/detr-resnet-50
```

Writes `depth_report.md` (per-model AbsRel/SqRel/RMSE/RMSElog/δ1-3), `depth_bars.png`,
`depth_metrics.json`, and — for **metric** models — `viz/fuse_*.png` (with `--viz`) and
`depth_fusion_<model>.mp4` (with `--video`) to the out-dir. Fusion viz/video are restricted
to metric models, since per-object distance needs true metres.

### Add a model (≈30 lines)

Most HF depth models work through the existing adapter — just pass the hub id:
`--models hf_depth:<org/model>`. Metric models are auto-detected by name (`zoedepth`,
`metric`, `-kitti`, `-nyu`) or forced with the adapter's `metric=` kwarg. For a non-HF
model, subclass `BaseDepthEstimator`, implement `predict() -> DepthResult`, and
`@register("your_key")` it (see `estimators/base.py`).

### Per-file self-tests

```bash
python -m DeepDataMiningLearning.ngperception.depth.datasets        # KITTI GT coverage
python -m DeepDataMiningLearning.ngperception.depth.evaluator       # synthetic sanity
python -m DeepDataMiningLearning.ngperception.depth.fusion          # nearer box = smaller dist
python -m DeepDataMiningLearning.ngperception.depth.estimators.base # list backends
```

## Setup

Uses the same `py312` env as ngdet (`torch transformers pillow numpy matplotlib`). Depth
models download from the HF Hub on first use. KITTI root defaults to
`/mnt/e/Shared/Dataset/Kitti/` (override with `--root`); needs `training/{image_2,
velodyne,calib}` extracted.
