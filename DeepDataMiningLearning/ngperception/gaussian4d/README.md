# gaussian4d — label-free Gaussian occupancy teacher → camera-only student

Phase-1 code for the direction in
[../docs/NEXT_PAPER_4D_GAUSSIAN_TEACHER.md](../docs/NEXT_PAPER_4D_GAUSSIAN_TEACHER.md): build an
occupancy **teacher** from raw LiDAR (geometry) + frozen-FM 2D semantics (`ngdet/labelgen`) with
**no human 3D/box labels**, and use it to supervise a **camera-only student**. The whole package is
organised so that the Phase-1 gate — *does a Gaussian teacher beat a hard-voxel LiDAR teacher?* — is
a clean one-flag swap.

## Layout

```
gaussian4d/
  teachers/
    base.py            # TeacherTarget (semantics + per-voxel uncertainty weight) + grid helpers
    semantics.py       # LiDAR points -> Occ3D-class labels by projecting into labelgen 2D semantics
    voxel_teacher.py   # BASELINE: hard voxelization (single / multi-sweep), weight=1.0
    gaussian_teacher.py# NOVEL: LiDAR-init Gaussians soft-splatted -> sub-voxel mass + uncertainty
    __init__.py        # build_teacher(kind, ...) factory: voxel1 | voxel10 | gaussian | gaussian10
  build_teacher.py     # CLI: cache teacher targets, or --stats to sanity-check vs Occ3D GT
  README.md
```

Every teacher returns the **same `TeacherTarget`** (a dense Occ3D-grid `semantics` + a `weight`
uncertainty map), so the student trains identically against any of them — only the geometric
representation changes.

## Why Gaussian, not voxel BCE (the reviewer's question)

The Gaussian teacher must *demonstrate* three advantages a voxel grid cannot (see the design doc):
1. **less hard quantization** — sub-voxel splatting fills thin/tail structures;
2. **uncertainty** — `weight` down-weights the student where the teacher is sparse/far;
3. **continuous representation** — Gaussians carry (pos, scale) → the *same* teacher extends to 4D
   forecasting (Phase 3) via per-Gaussian motion (`GaussianTeacher.gaussians()` exposes the list).

## Usage

```bash
GTS=<…/gts>; NUSC=<…/v1.0-trainval>; LGC=<…/labelgen_cache>   # labelgen npz cache (SegFormer+SAM sem)

# 1) sanity-check a teacher vs Occ3D GT (no student needed) — agreement mIoU / geo-IoU / tail-IoU:
python -m DeepDataMiningLearning.ngperception.gaussian4d.build_teacher \
    --nusc $NUSC --gts $GTS --labelgen-cache $LGC --teacher gaussian --stats --n 20

# 2) cache teacher targets for student training (any --teacher):
python -m ...gaussian4d.build_teacher --nusc $NUSC --gts $GTS --labelgen-cache $LGC \
    --teacher voxel10 --out-dir <teacher_cache/voxel10> --n 2000
```

First sanity signal (teacher-vs-GT agreement, 20 frames): Gaussian(1-sweep) mIoU 0.154 / geo-IoU
0.443 / 22.5k occupied vs Voxel(10-sweep) 0.128 / 0.359 / 10.2k — the less-quantization advantage
shows up as denser, higher-agreement targets. (Agreement is an *upper bound*; the gate is the
**student**.)

## Phase-1 plan (the gate)

Cache each teacher's targets → train a **camera-only student** on each → eval camera-only inference
on **mIoU + RayIoU + geo-IoU + tail-class IoU**. Arms: `camera baseline` · `render2d` (aux only) ·
`voxel1` · `voxel10` · `gaussian` · `gaussian10` · `Occ3D-GT` (ceiling). **If the Gaussian teacher
does not stably beat `voxel10` (esp. tail classes), stop — the story would be plain LiDAR
distillation.** Student trainer + eval reuse the occupancy harness (`train_lss.py` + a
`--teacher-target-cache` target) and add RayIoU + tail-IoU. TODO: `train_student.py`, `eval_student.py`.
