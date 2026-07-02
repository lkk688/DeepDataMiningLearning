# ngperception/detection — pure-PyTorch 3D object detection (PointPillars)

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
├── pointpillars.py   # pillarize + PillarVFE + scatter + BEV backbone + anchor head + model
├── box_utils.py      # ResidualCoder + pure-torch 3D IoU (aligned + exact rotated) + NMS
├── losses.py         # focal cls loss + smooth-L1 box loss (harvested, device-agnostic)
├── eval3d.py         # self-contained BEV AP (rotated-IoU, R40) — runs locally, no numba
└── kitti_eval/       # harvested official KITTI eval (numba) — HPC-only, see caveat below
```

## What's harvested vs written fresh

| harvested (pure torch, adapted to explicit args) | written fresh (pure torch) |
|---|---|
| PillarVFE / PFNLayer, PointPillarScatter, BaseBEVBackbone | `pillarize` (points→pillars, vectorised numpy) |
| AnchorGenerator, AnchorHead, ResidualCoder | rotated + axis-aligned BEV IoU, 3D IoU, NMS (`box_utils`) |
| SigmoidFocal + SmoothL1 losses; KITTI eval | `eval3d` BEV-AP (local, numba-free) |

Skipped (need spconv): SECOND, VoxelNeXt, BEVFusion — see [../PLAN.md](../PLAN.md).

## Run / verify (M0)

```bash
P=/home/lkk688/miniconda/envs/py312/bin/python
# forward + loss + backward + predict on random points:
$P -m DeepDataMiningLearning.ngperception.detection.pointpillars
# BEV-AP evaluator self-test:
$P -m DeepDataMiningLearning.ngperception.detection.eval3d
```

Verified: box-coder round-trips to 0 error; axis-aligned IoU (identical=1.0, half-shift=0.333)
and exact rotated IoU (identical=1.0, 45°=0.707) correct; NMS de-duplicates; the model does a
full forward → loss → backward → NMS'd prediction (4.8 M params, 107 k anchors); BEV-AP is 1.0
on perfect predictions and degrades correctly on FPs / misses.

## Status & caveats (honest)

- **M0 = skeleton, not a KITTI number.** The fast IoU path (target assignment + NMS) is
  **axis-aligned** (exact for yaw≈0, an approximation under rotation). Exact rotated IoU exists
  (`box_utils.rotated_iou_bev`) for eval; vectorising it for the training path is **M2**.
- **`kitti_eval/` (official metric) needs `numba` + a working CUDA** — its `rotate_iou.py` uses
  `numba.cuda`, which **segfaults under this WSL2 setup**. Use `eval3d.py` locally; run
  `kitti_eval/` on the H100/HPC for official-KITTI parity (**M1**).
- Real KITTI data loading + training to real mAP is **M1** (next), and fusing the head onto the
  shared occupancy encoder for the camera/LiDAR/fusion ablation is **M3**.
