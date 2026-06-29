# ngperception/occupancy — 3D semantic occupancy (Occ3D-nuScenes)

Predict a dense voxel grid of semantics around the ego car ($200\times200\times16$ @ 0.4 m,
17 classes + free). We start with a **depth→occupancy baseline** — lift 6-camera metric
depth (+ 2D segmentation) into the voxel grid, *no occupancy training* — to measure how
far vision foundation models alone get you (the ViPOcc question), and to expose the
headroom a learned occupancy net must earn. See [TUTORIAL.md](../TUTORIAL.md) §2.

## Data

The Occ3D-nuScenes GT (`gts/scene-XXXX/<token>/labels.npz`: `semantics`, `mask_camera`,
`mask_lidar`) from [Tsinghua-MARS-Lab/Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D),
plus the nuScenes `v1.0-trainval` metadata + images (for the 6-camera calibration). Extract
a subset of the `gts` tar to a working dir and point `--gts` at it.

## Run

```bash
# depth->occupancy baseline + LiDAR geometric-ceiling oracle:
python -m DeepDataMiningLearning.ngperception.occupancy.run_eval \
    --gts /path/to/extracted/gts \
    --nusc-root /mnt/e/Shared/Dataset/NuScenes/v1.0-trainval \
    --depth hf_depth:depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf \
    --seg nvidia/segformer-b2-finetuned-cityscapes-1024-1024 \
    --max-samples 40 --oracle
```

Reports **mIoU** (17 classes, on `mask_camera` voxels) + class-agnostic **geometric IoU**.
Omit `--seg` for a geometry-only (geo-IoU) run.

## Layout

```
occupancy/
├── datasets.py            # Occ3DNuScenesDataset (labels.npz loader)
├── evaluator.py           # OccupancyEvaluator: mIoU + geo-IoU on camera-visible voxels
├── geom.py                # nuScenes calib, back-projection, voxelization (ego frame)
├── predictors/
│   └── depth_lift.py      # 6-cam metric depth + Cityscapes seg -> voxel grid
└── run_eval.py            # CLI: baseline + LiDAR oracle
```

## Result (official val split, 30 samples)

| method | mIoU | geo-IoU | runnable |
|---|---|---|---|
| LiDAR oracle (single sweep) | — | 0.167 | ✅ |
| depth→occ, DA-V2-Metric-Small + seg | 0.014 | 0.093 | ✅ |
| depth→occ, DA-V2-Metric-Base + seg | 0.011 | 0.085 | ✅ |
| CTF-Occ / FlashOcc / **Dr.Occ (depth-guided)** / EFFOcc | 28.5 / 32 / **43.4** / 50.5 | — | ✗ mmdet3d |

`run_eval` prints the cited SOTA table after each run. Key finding from the sweep: a
**bigger depth model doesn't help** (Base < Small) — the single-shot lift, not depth
quality, is the bottleneck. The ~20–35× mIoU gap to learned nets is the research
opportunity; that the strongest closers (Dr.Occ) are **depth-guided** confirms depth is a
valuable signal. See TUTORIAL §2.4 for next steps (depth-supervised learned occupancy,
temporal aggregation, cross-camera metric depth).

## Setup

```bash
pip install nuscenes-devkit          # calibration + LiDAR/camera resolution
```
Reuses `ngperception.depth` for metric depth and `transformers` for the seg model.
