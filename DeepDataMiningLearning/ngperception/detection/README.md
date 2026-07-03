# ngperception/detection — pure-PyTorch 3D object detection (PointPillars)

> **Teaching walkthrough: [TUTORIAL.md](TUTORIAL.md)** — how the pure-PyTorch PointPillars works
> component by component, the results/ablations, and copy-paste reproduction commands.

A LiDAR 3D detector with **no spconv, no mmcv, no compiled CUDA ops** — the same
"runs-in-the-main-torch-env" philosophy as [`occupancy/`](../occupancy/). The algorithm is
**harvested from [`2D3DFusion/mydetector3d`](/home/lkk688/Developer/2D3DFusion/mydetector3d)**
(an OpenPCDet fork); the one CUDA piece there (`ops/iou3d_nms`) is replaced by a pure-torch
IoU/NMS. This is the **M0 skeleton** (see [../PLAN.md](../PLAN.md) for the roadmap).

## Why PointPillars first

It is the one classic LiDAR detector that needs **no sparse convolution**: pillar VFE
(per-pillar PointNet) → scatter to a BEV pseudo-image → 2-D CNN → anchor head. Every step is
plain `torch.nn`, so it drops straight into our env — and it is the natural sibling of the
occupancy LSS lift (both: *encode → BEV/voxel features → head*), which is why occ and det will
share one encoder (PLAN.md M3–M5, giving camera/LiDAR/fusion ablation for free).

## Layout

```
detection/
├── pointpillars.py     # anchor-based model + BaseBEV/ResBEV backbones (pillarize,VFE,scatter,head)
├── centerpoint.py      # center-based model: same front-end + CenterHead (Gaussian heatmap)
├── bev_transformer.py  # attention/DETR model: queries + BEV cross-attn + Hungarian matching
├── box_utils.py        # ResidualCoder + pure-torch 3D IoU (aligned + vectorised rotated) + NMS
├── losses.py           # focal cls loss + smooth-L1 box loss (harvested, device-agnostic)
├── kitti_dataset.py    # KITTI Car loader (+ cam->LiDAR transform, reused by Waymo)
├── nuscenes_dataset.py # nuScenes loader (devkit; boxes already LiDAR-frame)
├── waymo_dataset.py    # Waymo (KITTI-export) loader
├── train_kitti.py      # train+eval; --model {pointpillars,centerpoint,bev_transformer} --backbone {base,res}
├── eval3d.py           # self-contained BEV AP (rotated-IoU, R40) — runs locally, no numba
└── kitti_eval/         # harvested official KITTI eval (numba) — HPC-only, see caveat below
```

**Three model paradigms** (all pure-torch, sharing the pillar front-end): anchor-based
**PointPillars**, center-based **CenterPoint**, attention/DETR **BEV-transformer** — **× two
backbones** (`base` SSD-style, `res` PillarNeXt-style ResNet+FPN) **× three datasets** in one
interface (KITTI / nuScenes / Waymo). Key findings (overfit mAP@0.5): the **res backbone is a
large win** (PointPillars 0.17→0.63, CenterPoint 0.93→1.00); the BEV-transformer needs
**deformable attention** to converge — vanilla full-attention is stuck at 0.00, and
deformable-attn + iterative refinement + aux losses (all pure-torch, `grid_sample`) reaches
**0.925**. See [TUTORIAL.md](TUTORIAL.md) §9–10. (SECOND/PV-RCNN/VoxelNeXt need spconv — out of
scope.)

## What's harvested vs written fresh

| harvested (pure torch, adapted to explicit args) | written fresh (pure torch) |
|---|---|
| PillarVFE / PFNLayer, PointPillarScatter, BaseBEVBackbone | `pillarize` (points→pillars, vectorised numpy) |
| AnchorGenerator, AnchorHead, ResidualCoder | rotated + axis-aligned BEV IoU, 3D IoU, NMS (`box_utils`) |
| SigmoidFocal + SmoothL1 losses; KITTI eval | `eval3d` BEV-AP (local, numba-free) |

Skipped (need spconv): SECOND, VoxelNeXt, BEVFusion — see [../PLAN.md](../PLAN.md).

## Run / verify

```bash
P=/home/lkk688/miniconda/envs/py312/bin/python
# M0 — forward + loss + backward + predict on random points:
$P -m DeepDataMiningLearning.ngperception.detection.pointpillars
# M0 — BEV-AP evaluator self-test:
$P -m DeepDataMiningLearning.ngperception.detection.eval3d
# M1 — KITTI loader sanity (real data):
$P -m DeepDataMiningLearning.ngperception.detection.kitti_dataset --root /mnt/e/Shared/Dataset/Kitti
# M1 — overfit sanity (pipeline learns) + short real train:
$P -m DeepDataMiningLearning.ngperception.detection.train_kitti --root /mnt/e/Shared/Dataset/Kitti --overfit --max-frames 12 --epochs 80
$P -m DeepDataMiningLearning.ngperception.detection.train_kitti --root /mnt/e/Shared/Dataset/Kitti --max-frames 500 --val-frames 150 --epochs 30
```

**M0 verified**: box-coder round-trips to 0 error; axis-aligned IoU (identical=1.0,
half-shift=0.333) and exact rotated IoU (identical=1.0, 45°=0.707) correct; NMS de-duplicates;
the model does a full forward → loss → backward → NMS'd prediction (4.8 M params, 107 k
anchors); BEV-AP is 1.0 on perfect predictions and degrades correctly on FPs / misses.

**M1 verified** (real KITTI Car, `kitti_dataset.py` + `train_kitti.py`): loader confirmed —
camera→LiDAR box transform gives correct Car boxes (size ~4.15×1.73×1.57 m). Full pipeline
runs data → train → eval. Two checks:
- **overfit** (6 frames, loss→0.011): predicts **exactly 18 detections for 18 GT cars**,
  **mAP@0.5 = 0.45** — localization correct.
- **generalization** (train 500 / **val 150 held-out**, 30 ep): val **mAP@0.5 climbs
  0 → 0.324** (ep15 0.125 → ep20 0.261 → ep29 0.324), so it learns a real detector, not just
  memorises.

Both ceilings are **heading quality**: mAP@0.7 stays ~0.036. The gap to competitive KITTI
numbers is M2 + full-schedule training on H100 (ours is 6.7 % of the split, 30 ep, from scratch).

**M2 — rotated-IoU assignment + direction classifier (opt-in, ablated).** Two standard
improvements, added behind flags (`--rotated-assign`, `--use-dir`) and A/B'd on the overfit
(6 frames / 18 cars, 3 seeds):

| config | mAP@0.5 (3-seed) | note |
|---|---|---|
| baseline (axis-aligned assign) | 0.225 ± 0.097 | over-assigns (num_pos 276) |
| **`--rotated-assign`** | **0.323 ± 0.046** | cleaner assign (num_pos 123): higher mean, **½ the variance** |
| `--rotated-assign --use-dir` | 0.141 (1 seed) | direction classifier **hurt** — see below |

Honest read: **rotated-IoU assignment helps modestly and stabilises training** (+0.098 mean,
~1.6σ at n=3, but variance halves — the cleaner target set is the mechanism, exactly what the
per-box diagnosis predicted: axis-aligned assignment gave muddy regression targets on close
cars). The **direction classifier hurt** here, because our raw angle regression *already*
resolves heading well (per-box mod-π error 0.02–0.09); the 2-bin inference correction only
perturbs already-good headings. So `--use-dir` stays **off by default**.

**M2b — vectorised rotated IoU (done).** The rotated assignment originally used a numpy
Sutherland–Hodgman IoU (~2× iter time). M2b replaces it with a **fully vectorised torch**
convex-quad intersection (points-inside + edge-intersections + angular-sort shoelace,
`rotated_iou_bev_paired_torch`) that runs on GPU: it **matches the numpy version to 1e-4** and
is **~65× faster per pair** (5100 pairs in 14.7 ms). Rotated assignment is now ~free (46 s vs
96 s per 250-iter overfit, same as baseline's 44 s) — i.e. **practical for full-split training
on H100**, which is what turns the overfit gain into real val-AP.

## Status & caveats (honest)

- **M0/M1 = correct pipeline, not yet a competitive KITTI number.** The default fast IoU path
  is **axis-aligned** (exact for yaw≈0). `--rotated-assign` (M2) improves and stabilises it,
  and after **M2b** (vectorised torch rotated IoU) it runs at ~baseline speed on GPU — usable
  for full-split training on H100, which is what lifts real val-AP.
- **`kitti_eval/` (official metric) needs `numba` + a working CUDA** — its `rotate_iou.py` uses
  `numba.cuda`, which **segfaults under this WSL2 setup**. Use `eval3d.py` locally; run
  `kitti_eval/` on the H100/HPC for official-KITTI parity (**M1**).
- Real KITTI data loading + training to real mAP is **M1** (next), and fusing the head onto the
  shared occupancy encoder for the camera/LiDAR/fusion ablation is **M3**.
