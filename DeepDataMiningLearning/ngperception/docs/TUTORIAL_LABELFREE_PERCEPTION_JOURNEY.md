# Tutorial & Log: Label-Free / Label-Efficient AD Perception — every solution we tried

A single reference for the whole investigation: goal, datasets, environments, each approach (how it
works, code, command, weights, result), the positive and negative findings, and where we stand.
Companion docs: `PLAN_CROSSDATASET_OCC_PRETRAIN.md`, `PLAN_FLASHOCC_MIGRATION.md`,
`RESULTS_LABELFREE_OCC_2x2_TRANSFER.md`, `REFS_2026_OCC_GAUSSIAN_LABELFREE.md`.

## 0. Goal & north-star metric
Publishable **label-free / label-efficient** 3D perception (occupancy + detection) on nuScenes.
The metric that ended up mattering: **occ pretraining → low-label 3D-detection label-efficiency**
(official nuScenes mAP/NDS at 2k/4k/8k labels). Repo: `DeepDataMiningLearning/ngperception`.

## 1. Datasets & key cache paths
| data | path |
|---|---|
| nuScenes v1.0-trainval | `/data/rnd-liu/Datasets/nuScenes/v1.0-trainval` |
| Occ3D-nuScenes GT | `.../v1.0-trainval/gts` (labels.npz: semantics 200×200×16, mask_camera) |
| FM semantics cache (SegFormer+GSAM→Occ3D) | `.../nuScenes/labelgen_cache` (hard), `.../labelgen_soft_cache` (top-K) |
| teacher target caches | `.../nuScenes/teacher_cache_{ra,ra2,2x2soft,dynamic}` |
| Waymo v2.0.1 | `/fs/atipa/data/rnd-liu/Datasets/waymo201/validation` (loader `detection3d/dataset_waymo3dv201.py`) |
| Waymo v1.4.3 (raw tfrecords) | `/fs/atipa/data/cmpe249-fa25/waymov143_individuals/training` |
| Argoverse2 sensor | `/data/cmpe258-sp24/.../thesis-nurec/data/argoverse2_raw/sensor/{train,val}` |
| PhysicalAI-AV | `/fs/atipa/data/rnd-liu/Datasets/PhysicalAI-AV` + `.../thesis-nurec/data/pai/ncore` |

## 2. Environments
- **py310** — main (torch 2.9+cu126, mmdet3d 1.4). Everything in `ngperception` runs here.
- **py311** — GaussianOcc repro only.
- **py312** — PhysicalAI/AV2 ingestion (pandas/pyarrow/DracoPy).
- **CUDA toolkit for `bev_pool_v2`** (FlashOcc): isolated `conda create -p /data/rnd-liu/cudatk -c nvidia
  cuda-toolkit=12.6`; merged home `/data/rnd-liu/cuda_home2` (bin→cudatk/bin,
  include→cudatk/targets/x86_64-linux/include, lib64→cudatk/lib). Export `CUDA_HOME=/data/rnd-liu/cuda_home2`,
  PATH/LD_LIBRARY_PATH, `TORCH_CUDA_ARCH_LIST=9.0` for any FlashOcc run.
- Common preamble: `export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning`.
- **fp32 mandatory for detection** (amp corrupts BN running stats → eval 0.000). amp OK for occ.

## 3. The label-free occupancy TEACHER (shared supervision)
Geometry from LiDAR + semantics from frozen 2D FMs, no human 3D labels. `gaussian4d/teachers/`:
`base.py` (TeacherTarget: semantics/weight/soft), `semantics.py` (LiDAR→ego + FM projection),
`raycast.py` (ray-cast free space), `voxel_teacher.py`, `gaussian_teacher.py`, `dynamic_teacher.py`.
Build a cache: `python -m ...gaussian4d.build_teacher --teacher voxel10 --out-dir <cache> --n 2044`.

---

## 4. Approaches tried — how, code, command, result

### 4.1 Soft-FM semantic distillation (voxel teacher)   ✅ POSITIVE (occ mIoU)
- **How:** per-point top-K FM class *distribution* (not argmax) → per-voxel soft target; student
  distilled with soft-CE + entropy/confidence weighting. Preserves car/truck, bike/moto confusion.
- **Code:** `gaussian4d/teachers/voxel_teacher.py` (soft mode), `train_student.py`
  (`teacher_loss_soft`). Cache `teacher_cache_2x2soft/voxel10`.
- **Result:** camera-only occ **mIoU .092 → .104 (+13%)**, best of the fair 2×2.
- **Weights:** `output/student_soft_voxel10/student.pth`.

### 4.2 4D-Gaussian occupancy teacher   ❌ NEGATIVE (stopped)
- **How:** anisotropic surface-aligned 3D Gaussians (local-PCA disk) splatted into the grid vs hard
  voxels; uncertainty weighting; meant to extend to 4D.
- **Code:** `gaussian_teacher.py`, `run_2x2.sh`, `audit_soft.py`, `run_rescue.sh` (factorized loss).
- **Result:** fair 2×2 + audit + factorized rescue → **Gaussian earns NO advantage** once the
  occupancy/semantic coupling is removed (the apparent +9.7% was an artifact). Stopped 4D/VGGT/query.
- **Weights:** `output/student_ra2_gaussian`, `output/student_fac_{voxel10,gaussian}`.
- **Ref:** we found (RESULTS_LABELFREE_OCC_2x2_TRANSFER.md) geo-IoU collapse was implementation
  coupling (audit) not physics.

### 4.3 FlashOcc / BEVDet-Occ external validation   ✅ ceiling reproduced
- **How:** port FlashOcc to modern torch (mmcv stripped, torchvision R50, reuse compiled
  `bev_pool_v2`); load official checkpoint; train label-free from voxel-soft teacher.
- **Code:** `flashocc/{model.py, model_stereo.py, data_stereo.py, train_flashocc.py, eval_flashocc.py,
  eval_stereo.py, train_det_stereo.py, eval_det_stereo.py}`.
- **Command (ceiling):** `python -m ...flashocc.eval_stereo --ckpt Others/FlashOCC/ckpts/flashocc-r50-4d-stereo.pth`
- **Result:** 4D-stereo supervised **mIoU 0.3809 (published 0.3784)** — exact reproduction (found+fixed
  a BGR-normalization bug worth ~.04). Single-frame R50 **label-free voxel-soft = mIoU 0.093** vs
  supervised 0.3195. Det head on FROZEN 4D-stereo backbone = **NDS 0.070** (weak linear probe).
- **Weights:** `Others/FlashOCC/ckpts/flashocc-r50-4d-stereo.pth`, `output/flashocc_r50_voxelsoft/flashocc.pth`,
  `output/det_on_stereo_frozen/det_head.pth`.

### 4.4 Occupancy → detection LABEL-EFFICIENCY transfer   ✅ POSITIVE (the key result) / ❌ label-free
- **How:** pretrain an occ encoder, finetune into a center-head 3D detector at small label budgets;
  official nuScenes DetectionEval. Arms swap the `--pretrained` occ checkpoint.
- **Code:** `occupancy/train_det_ablation.py` (`--pretrained --seed --max-samples`),
  `eval_det_ablation_official.py`, grids `occupancy/run_le_{occ3d,dynamic}.sh` +
  `run_label_efficiency.sh`, plot `occupancy/plot_label_efficiency.py`.
- **Command:** `bash occupancy/run_le_occ3d.sh` (fp32, camera-only, center head, seeds).
- **Results (official mAP, seed-mean; scratch=DINOv2 no-occ):**
  | budget | scratch | **occ3d-GT (manual)** | voxel-soft (label-free) | DynamicOcc (label-free) |
  |---|---|---|---|---|
  | 2k | 0.121 | **0.160 (+32%)** | 0.115 (null) | 0.087 (−28%) |
  | 4k | 0.153 | **0.183 (+20%)** | 0.140 | — |
  | 8k | 0.177 | **0.197 (+11%)** | — | — |
- **Finding:** occ pretraining IS label-efficient for detection **but only with dense manual-quality
  occ labels (Occ3D-GT)**. Every label-free pretext is null-to-negative.
- **Weights:** occ3d pretext `output/lss_occ_full/lss_occ.pth`; label-free `output/student_soft_voxel10`,
  `output/student_dynamicocc`.

### 4.5 DynamicOcc pseudo-label (Occ3D-learned dynamic/static)   ❌ NEGATIVE (clarifying)
- **How:** Occ3D's trick label-free — multi-sweep points NOT in boxes = static (FM semantics);
  current-sweep points IN boxes = dynamic, sharp, box class. Fixes the "smeared moving objects" of
  voxel-soft. (`boxes=gt` here = recipe upper bound; label-free would use pseudo-tracks.)
- **Code:** `gaussian4d/teachers/dynamic_teacher.py`, `build_dynamic.py`, `test_dynamic_teacher.py`.
- **Command:** `python -m ...gaussian4d.build_dynamic --boxes gt --out-dir <cache> --n 2044`
  → `train_student` → `run_le_dynamic.sh`.
- **Result:** pseudo-label GT-agreement UP (**foreground +60%**, mIoU .127→.162) but det transfer DOWN
  (**mAP .087 @2k, −28% vs scratch**). *Better labels, worse transfer.*
- **Weights:** cache `teacher_cache_dynamic`, `output/student_dynamicocc/student.pth`.
- **Interpretation:** the detection benefit is a **dense camera-predictable SCENE representation**
  effect, not a foreground-object effect. Sharp GT-box objects a camera can't predict → ill-posed
  pretext → corrupts the encoder. (occ-mIoU-of-student diagnostic: see §6.)

### 4.6 Cross-dataset labeled-occ pool (Waymo/AV2/PhysicalAI)   ⏸ scoped, gated
- **How:** unify LiDAR-geometry + 3D-box foreground into the Occ3D grid across datasets. Inventory
  done (loaders/paths in `PLAN_CROSSDATASET_OCC_PRETRAIN.md`). **Blocker:** none of the three has
  LiDAR semseg → only foreground+geometry cheaply → gated on Step-0 (which came back negative).
- **Reusable:** Waymo `detection3d/dataset_waymo3dv201.py`; PhysicalAI
  `PhysicalAI-Drive/.../pseudolabel_physicalai/*`; AV2 plain `pd.read_feather`.

### 4.7 Earlier explorations (from project memory)
- **GaussianOcc** self-sup occ repro: mIoU **11.26** (matches paper). Reference only.
- **Camera-only BEV lift-splat**: proven dead (5 experiments) — pivoted to camera-primary/LiDAR-aux.
- **Modality-robust BEVFusion** (L/C/LC one model): C-only branch collapses.
- **VGGT geometry backbone** for the dead camera path: negative study.

---

## 5. Positive vs negative — scoreboard
**Positive**
- Soft-FM semantic distillation: +13% occ mIoU (label-free).
- **occ pretraining → label-efficient detection: +32% mAP @2k — but needs manual Occ3D-GT labels.**
- FlashOcc ceiling reproduced (0.3809) — rigorous external validation; found a real BGR bug.
- occ→det works finetuned (NDS 0.206 full-data).

**Negative (each a clean, controlled result)**
- Gaussian occ teacher: no advantage over voxel (audit + factorized rescue).
- Label-free occ → detection transfer: **null (voxel-soft) to harmful (DynamicOcc)** — teacher-quality
  bounded; the benefit needs dense camera-predictable semantics label-free can't reach.
- Frozen strong-occ backbone → detection: weak (NDS 0.070).
- Camera-only BEV lift, modality-robust C-only, VGGT: negative.

## 6. Diagnostic (DynamicOcc student occ mIoU)  — <FILL>
Eval: `python -m ...gaussian4d.eval_student --ckpt output/student_dynamicocc/student.pth`.
Expectation: LOW mIoU (sharp GT-box labels unpredictable from camera → weak occ student → bad det
init), confirming the "label agreement ≠ transferable representation" mechanism.

## 7. Where we stand / open direction
Label-free occ *prediction* is now strong in the literature (TT-Occ, CVPR'26) but label-free occ **as a
detection pretext** keeps failing on the same wall: the transfer benefit lives in dense,
camera-predictable, manual-quality semantics. Candidate pivots (see REFS doc): (a) better *background*
FM semantics via multi-view consensus + confidence (OnlinePG) and 2D+3D refinement (PanDA) — target the
part that actually helps; (b) reframe around the **positive** label-efficiency result with a modest
label budget rather than fully label-free; (c) accept the negatives as a controlled study of *what
transfers* (dense scene semantics, not object sharpness; manual vs FM label quality).
