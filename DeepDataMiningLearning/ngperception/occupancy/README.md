# ngperception/occupancy — 3D semantic occupancy (Occ3D-nuScenes)

Predict a dense voxel grid of semantics around the ego car ($200\times200\times16$ @ 0.4 m,
17 classes + free). We start with a **depth→occupancy baseline** — lift 6-camera metric
depth (+ 2D segmentation) into the voxel grid, *no occupancy training* — to measure how
far vision foundation models alone get you (the ViPOcc question), and to expose the
headroom a learned occupancy net must earn. See [TUTORIAL.md](../TUTORIAL.md) §2.

## Data

Two sources, both used:

1. **Occ3D-nuScenes GT** (`gts/scene-XXXX/<token>/labels.npz` → `semantics` 200×200×16,
   `mask_camera`, `mask_lidar`) from
   [Tsinghua-MARS-Lab/Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D) — the **dense
   occupancy GT** for training (occupancy CE) and the mIoU eval. It is itself built by
   aggregating LiDAR over the whole sequence + nuScenes-lidarseg labels + occlusion masks
   (see TUTORIAL §2.7.1).
2. **nuScenes `v1.0-trainval`** images + metadata — the 6 surround images + per-camera
   calibration (intrinsics/extrinsics), and the `LIDAR_TOP` sweeps used to make the
   auxiliary **depth supervision** (projected into each camera). LiDAR is training-only;
   inference is camera-only.

On this machine:

```bash
# Occ3D gts archive lives here; extract it once to a local SSD working dir:
ls /mnt/e/Shared/Dataset/gts/gts-002.tar.gz
mkdir -p /home/lkk688/Developer/occ3d_data
tar -xzf /mnt/e/Shared/Dataset/gts/gts-002.tar.gz -C /home/lkk688/Developer/occ3d_data
# -> /home/lkk688/Developer/occ3d_data/gts  (34,149 labels.npz across 850 scenes)

# nuScenes trainval (images + calib + LiDAR):
ls /mnt/e/Shared/Dataset/NuScenes/v1.0-trainval   # samples/ sweeps/ maps/ v1.0-trainval/
```

`--gts /home/lkk688/Developer/occ3d_data/gts` and
`--nusc /mnt/e/Shared/Dataset/NuScenes/v1.0-trainval` are the paths used below.

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

## Learned model — depth-supervised LSS occupancy (pure PyTorch)

The in-house, **GT-supervised** occupancy net (no mmcv, runs in the main torch env): image
encoder (ResNet-18 or frozen **DINOv2**) + a **depth-distribution head** + the **LSS lift**
(frustum→ego→voxel scatter) + a 3D voxel decoder. **Inference is camera-only**; trained with
**occupancy CE (vs dense Occ3D GT) + auxiliary LiDAR depth CE** (BEVDepth-style). See
TUTORIAL §2.7.1 for the supervision details.

```bash
# baseline (ResNet-18, ~20 min) -> mIoU 0.092
python -m DeepDataMiningLearning.ngperception.occupancy.train_lss \
    --gts /home/lkk688/Developer/occ3d_data/gts \
    --nusc /mnt/e/Shared/Dataset/NuScenes/v1.0-trainval \
    --max-samples 500 --epochs 8 --depth-weight 1.0

# strongest (DINOv2-base + deeper decoder + cosine + AMP + Lovász & class-balanced CE)
python -m DeepDataMiningLearning.ngperception.occupancy.train_lss \
    --gts /home/lkk688/Developer/occ3d_data/gts \
    --nusc /mnt/e/Shared/Dataset/NuScenes/v1.0-trainval \
    --backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --cosine --amp \
    --max-samples 3000 --epochs 12 --depth-weight 1.0 --refine-iters 2 \
    --occ-lovasz 1.0 --occ-class-balance --occ-cb-power 0.25 \
    --occ-cb-cache /home/lkk688/Developer/occ3d_data/classw_3000.npy \
    --out-dir DeepDataMiningLearning/ngperception/output/lss_occ_dinobase
```

Results (Occ3D val, camera-mask mIoU), scaling on a single RTX 3090:

| config | mIoU | geo-IoU |
|---|---|---|
| ResNet-18, 500 samples / 8 ep | 0.092 | 0.547 |
| DINOv2-small, 1500 / 10 ep, AMP | 0.152 | 0.626 |
| DINOv2-base + deeper dec + cosine, 3000 / 12 ep | 0.216 | 0.681 |
| + Lovász + class-balanced CE (`--occ-lovasz --occ-class-balance`) | 0.284 | 0.701 |
| **+ iterative render-and-refine lift** (`--refine-iters 2`) | **0.298** | **0.710** |

**The loss is the biggest lever per unit effort.** Plain occupancy CE over ~85 %-free voxels
lets dominant classes swamp the rare-class gradient, and mIoU averages over the 17 non-free
classes — so rare-class IoU≈0 tanks it. Adding **Lovász-softmax** (a direct mIoU surrogate) +
**inverse-frequency class weights** roughly **doubles** mIoU at a fixed cheap setting
(DINOv2-small / 1k / 8ep: **0.139 → 0.226**, +7σ over 3 seeds, no geo-IoU loss) — a one-file
change that *beats the 3× bigger/longer 0.216 run above* — and on the strong config lifts it
**0.216 → 0.284** (geo-IoU 0.701). See TUTORIAL §2.8.1 (ported from the
[GaussianFormer3D study](../docs/GaussianFormer3D_study.md)).

On top of that, an **iterative render-and-refine lift** (`--refine-iters 2`, TUTORIAL §2.8.2)
adds a further **+0.014 → 0.298 mIoU / 0.710 geo-IoU** — sampling the decoded occupancy back
along each ray (first-hit transmittance) to sharpen the depth and re-lift, the pure-PyTorch
analogue of GaussianFormer3D's iterative deformable refinement. **0.298 mIoU is past CTF-Occ
(28.5)** and above BEVFormer (27) / OccFormer (~21) — camera-only, on ~10 % of the train
split, a frozen backbone, pure PyTorch, no mmcv. Visualizations (open3d) in `output/lss_occ/`:
`lss_occ_surround_demo.mp4` (6 cams + global occ + ego marker), `lss_occ_camview.mp4`
(camera-aligned + global), `lss_occ_vs_gt.mp4` (pred vs GT). See TUTORIAL §2.7–2.8.

## Layout

```
occupancy/
├── datasets.py            # Occ3DNuScenesDataset (labels.npz loader)
├── datasets_train.py      # NuScenesOccTrainDataset: 6-cam + calib + GT + LiDAR depth
├── evaluator.py           # OccupancyEvaluator: mIoU + geo-IoU on camera-visible voxels
├── geom.py                # nuScenes calib, back-projection, voxelization (ego frame)
├── models/lss_occ.py      # depth-supervised LSS occupancy network (pure PyTorch)
├── train_lss.py           # train loop: occ CE + depth CE + Occ3D eval
├── predictors/
│   └── depth_lift.py      # 6-cam metric depth + Cityscapes seg -> voxel grid (baseline)
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
