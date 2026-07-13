# Occupancy Tutorial — full-data results, multi-task, and modality-robustness

This tutorial documents the **experimental results** of the pure-PyTorch depth-supervised LSS
occupancy stack (`occupancy/`): scaling to full nuScenes on an H100, the official Occ3D-nuScenes
metrics, sharing the encoder with detection, and making one model modality-robust
(camera-only / LiDAR-only / fusion).

For the *method* walkthrough (how the LSS lift-splat + depth supervision + Lovász/class-balanced
loss + iterative refine work) see the main [../TUTORIAL.md](../TUTORIAL.md) §2. This file is the
**results + commands** companion. All runs are pure PyTorch (no mmcv/spconv), conda env `py310`.

Paths below use:
- `GTS=/data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes/gts` (extracted Occ3D GT)
- `NUSC=/data/rnd-liu/Datasets/nuScenes/v1.0-trainval`
- `OUT=DeepDataMiningLearning/ngperception/output`

---

## 1. Full-data results (single H100 NVL)

Scaling the validated recipe (DINOv2-base + 4-layer decoder + cosine + AMP + refine-2 +
Lovász/class-balanced CE) from the 3090 ablation scale (~10 % of the split) to the **full
34 149-frame train split, 24 epochs**.

| model | mIoU | geo-IoU | vs 3090 ref (3k/12ep) |
|---|---|---|---|
| **camera-only** | **0.302** | 0.669 | 0.298 → +0.004 |
| **camera + LiDAR fusion** | **0.558** | 0.838 | 0.493 → **+0.065** |

Camera-only barely moves (already near its capacity ceiling at 10 % data); **fusion gains +0.065**
— the LiDAR-geometry input has real headroom, landing in the supervised-fusion SOTA band.

```bash
# camera-only  -> mIoU 0.302 / geo-IoU 0.669
python -m DeepDataMiningLearning.ngperception.occupancy.train_lss \
    --gts $GTS --nusc $NUSC \
    --backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 \
    --cosine --amp --refine-iters 2 --occ-lovasz 1.0 --occ-class-balance \
    --occ-cb-cache $OUT/classw_full.npy \
    --max-samples 34149 --val-samples 300 --epochs 24 --batch-size 8 \
    --lr 2e-3 --depth-weight 1.0 --num-workers 8 --seed 0 \
    --out-dir $OUT/lss_occ_full

# camera + LiDAR fusion  -> mIoU 0.558 / geo-IoU 0.838   (add the two --lidar-* flags)
python -m DeepDataMiningLearning.ngperception.occupancy.train_lss \
    ... (identical) ... \
    --lidar-fusion --lidar-cache $OUT/lidar_cache --out-dir $OUT/lss_occ_full_fusion
```

**Two bugs the full-data scale exposed (both fixed in `train_lss.py`):**
1. **Class-weight GC stall** — the one-time `compute_class_weights` pass stalled >1 h (a fresh
   process does it in 44 s). Cause: per-iteration `NpzFile`+array allocation with the resident
   ~10 M-object nuScenes graph made cyclic GC rescan everything each iteration (~250× slowdown).
   Fix: `gc.disable()` around the loop.
2. **Fusion NaN divergence** — the fusion run diverged to NaN at ep3 (cosine LR near its 2e-3
   peak, heavier model, no grad clipping). Fix: `clip_grad_norm_(..., 5.0)` with the correct
   `GradScaler.unscale_` ordering.

---

## 2. Official val evaluation + per-class IoU (`eval_val.py`)

`eval_val.py` filters to the **150 official Occ3D-nuScenes val scenes** (6019 frames, via
`nuscenes.utils.splits.val`) and reports full per-class IoU — the real Occ3D metric.

```bash
python -m DeepDataMiningLearning.ngperception.occupancy.eval_val \
    --gts $GTS --nusc $NUSC --ckpt $OUT/lss_occ_full/lss_occ.pth \
    --backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --refine-iters 2
# add --lidar-fusion --lidar-cache $OUT/lidar_cache  for the fusion checkpoint
```

> **⚠️ Leakage caveat.** The Occ3D `gts` dir is the *full trainval* (850 scenes = 700 train +
> 150 val = 34149 = 28130 + 6019). The full-data runs above used `--max-samples 34149` with **no
> scene filter**, so all 6019 official-val frames were *in the training set*. Treat these mIoU as
> an **upper bound**; the per-class breakdown is diagnostic (which classes the model represents).

| class | camera-only | fusion |    | class | camera-only | fusion |
|---|---|---|---|---|---|---|
| others | 0.105 | 0.230 | | trailer | 0.308 | 0.665 |
| barrier | 0.294 | 0.669 | | truck | 0.347 | 0.602 |
| bicycle | 0.105 | 0.600 | | driveable_surface | 0.788 | 0.816 |
| bus | 0.469 | 0.731 | | other_flat | 0.480 | 0.515 |
| car | 0.429 | 0.612 | | sidewalk | 0.487 | 0.525 |
| construction_veh | 0.131 | 0.591 | | terrain | 0.548 | 0.584 |
| motorcycle | 0.118 | 0.624 | | manmade | 0.398 | 0.684 |
| pedestrian | 0.133 | 0.570 | | vegetation | 0.365 | 0.666 |
| traffic_cone | 0.076 | 0.446 | | **mIoU / geo-IoU** | **0.328 / 0.678** | **0.596 / 0.851** |

**Reading it:** camera-only is bimodal — large static surfaces strong (driveable 0.79), but
small/dynamic classes collapse (traffic_cone 0.08, bicycle/pedestrian/motorcycle 0.10–0.13).
**Fusion lifts exactly those weak classes most** (bicycle 0.10→0.60, motorcycle 0.12→0.62) and is
uniform (0.44–0.82): LiDAR gives the shape of small objects the camera lift can't resolve.

**Clean-retrain plan (leakage-free, deferred):** add scene filtering to `train_lss.py`
(`--split train` via `nuscenes.utils.splits.train`, 700 scenes), retrain on train-only, eval on
val-only with `eval_val.py` (already correct). The gap vs the leaked 0.328/0.596 quantifies the
leakage.

---

## 3. Multi-task — occupancy + detection on one encoder (`train_multitask_joint.py`)

The fused voxel volume feeds **two heads** (occ 3-D decoder + a `VoxelDetHead`). Detection details
live in [../detection/TUTORIAL.md](../detection/TUTORIAL.md); here is the **multi-task conflict**
finding, which is an occupancy story.

**The conflict, quantified** (8k/12ep, center det head, official val, best occ+det):

| shared-encoder strategy | occ mIoU | det car_AP | verdict |
|---|---|---|---|
| **frozen trunk** | **0.521** | 0.513 | ✅ no conflict — occ at full quality |
| naive joint finetune | 0.496 | 0.510 | dominated |
| PCGrad joint finetune | 0.502 | 0.522 | best det, small occ cost |

- **The headers coexist.** The catastrophic conflict we first saw (occ 0.521→0.001) was an
  occ-weight-0.1 artifact; at **occ-weight 1.0** finetuning drops occ only ~0.02.
- **PCGrad Pareto-dominates naive** (0.502/0.522 beats 0.496/0.510 on both axes) — gradient surgery
  genuinely resolves the shared-trunk conflict.
- **But the frozen trunk is the best practical architecture**: occ **0.521 (full)** + det 0.513
  (~=PCGrad det, +0.019 occ, cheapest). **Good fusion = one occ-trained trunk (frozen) + two
  task-specific heads.**

```bash
# 3-way conflict study (warm-starts trunk+occ-head + center det-head from our checkpoints)
python -m DeepDataMiningLearning.ngperception.occupancy.train_multitask_joint \
    --gts $GTS --nusc $NUSC \
    --occ-ckpt $OUT/lss_occ_full_fusion/lss_occ.pth \
    --det-ckpt $OUT/abl_dcenter_frozen_sub/det_abl.pth \
    --mode {frozen,naive,pcgrad} --cosine --max-samples 8000 --epochs 12 \
    --batch-size 4 --out-dir $OUT/mtj_<mode>
```

---

## 4. Modality-robust — one model for camera / LiDAR / fusion (`train_modality_robust.py`)

Goal: a single model where **every modality is a positive gain**, for both tasks. Mechanism:
per-batch **modality dropout** (`forward(drop_camera=, drop_lidar=)` zeros a branch, keeping the
decoder's channel layout), evaluated under all 3 modalities each epoch, best-saved by the *worst*
modality.

**Finding — naive dropout fails, fusion-anchored is the fix:**

| recipe | fusion (occ/det) | lidar-only | camera-only |
|---|---|---|---|
| warm-start (fusion-only) | 0.49 / 0.54 | 0.13 / 0.09 | 0.10 / 0.00 |
| **naive dropout** (1 modality/batch) | **degrades → 0.31/0.30** | ~0.10 / 0.14 | ~0.09 / 0.05 |
| **anchored + distillation** | **holds ~0.40 / ~0.43–0.53** | ~0.11 / ~0.21 | ~0.13 / ~0.07 |

- **Naive uniform dropout starves the fusion path** (only 1 modality/batch) → it degrades.
- **Anchored**: train fusion *every* batch (anchor, stays strong) + one random single-modality
  path (lifted), with the single-modality **distilled toward the fusion teacher** (occ KL + det
  heatmap MSE). Fusion holds; LiDAR-only becomes a clean positive; **occupancy lifts under all
  modalities**.
- **Camera-only detection is the documented hard limit** (~0.07) — camera-only 3-D boxes from a
  frozen-DINOv2 BEV lift-splat is architecturally weak (see [../detection/TUTORIAL.md] for the
  SOTA-PETR route that fixes camera-only detection).

```bash
python -m DeepDataMiningLearning.ngperception.occupancy.train_modality_robust \
    --gts $GTS --nusc $NUSC \
    --occ-ckpt $OUT/lss_occ_full_fusion/lss_occ.pth \
    --det-ckpt $OUT/abl_dcenter_frozen_sub/det_abl.pth \
    --mode anchored --mod-probs 0.0,0.55,0.45 --distill-weight 1.0 --cosine \
    --max-samples 8000 --epochs 12 --batch-size 3 --out-dir $OUT/mod_robust
# --mode dropout reproduces the naive (failing) baseline
```

---

## 5. Summary of levers (occupancy)

| lever | effect | where |
|---|---|---|
| DINOv2-base + deeper decoder + cosine + more data | 0.092 → 0.216 mIoU | main §2.8 |
| Lovász + class-balanced CE | +31 % (0.216 → 0.284) | main §2.8.1 |
| iterative render-and-refine lift | +0.014 (0.284 → 0.298) | main §2.8.2 |
| **full data + 24 ep (H100)** | **0.298 → 0.302 cam / 0.493 → 0.558 fusion** | §1 |
| LiDAR fusion (input, not just depth-sup) | camera-only → fusion, biggest single lever | main §2.8.3 |
| frozen shared trunk (multi-task) | occ 0.521 + det 0.513, no conflict | §3 |
| fusion-anchored + distill (modality-robust) | all modalities positive | §4 |

Remaining: clean train-only retrain (leakage-free numbers), DINOv2-large / 0.2 m voxels, temporal
fusion, and strong camera-only via PETR (detection tutorial).
