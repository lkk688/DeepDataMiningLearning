# Plan: migrate to FlashOcc / BEVDet-Occ (external validation, no new backbone)

**Decision (2026-07-20, user):** stop developing our own occupancy backbone. Use public
**FlashOcc/BEVDet-Occ** as a standard external backbone. Public **supervised** checkpoint = ceiling;
a **same-architecture, voxel-soft (label-free)** checkpoint = our core result. Our LSS numbers
(`RESULTS_LABELFREE_OCC_2x2_TRANSFER.md`) = the first backbone set.

## Environment finding (the fork)

- **Official FlashOcc** (`Others/FlashOCC`, cloned) installs torch **1.10+cu111**, mmcv-full **1.5.3**,
  mmdet **2.25.1** (old mmlab 1.x). **This cannot run on our H100 (sm_90 needs cu11.8+/torch 2.x).**
- **py310 is modern and H100-ready**: torch **2.9+cu126**, and — crucially — the BEVDet LSS CUDA op
  is already built here (`bev-pool` editable install; `DeepDataMiningLearning/bevdet/setup.py` builds
  `bev_pool_ext` + `voxel_layer`). So a **modern-stack BEVDet LSS runs on H100 today**.
- **FlashOcc's contribution is architecturally tiny**: BEVDet (LSS) → BEV feature (C×H×W) →
  *channel-to-height* occ head (reshape C → Dz × n_class, replacing a heavy 3D decoder). Trivial to
  add on top of the modern LSS we already have.

⟹ **Path B (port the FlashOcc head into the modern py310 BEVDet)** is the feasible route; Path A
(native old-mmlab FlashOcc on H100) is blocked.

## The one open decision — how to get the supervised ceiling

The public checkpoints are in **old-mmlab format**. To use them as a ceiling *in our codebase* we
must remap weights into the modern port. Options:

- **(A) Remap official R50 checkpoint** (mIoU 31.95, single-frame) into the modern port and eval it
  ourselves. Most faithful within-codebase ceiling; backbone (R50) remaps trivially, LSS
  view-transformer + BEV encoder + channel2height head need key-mapping (architecture matches, so
  doable) — moderate effort + verification risk.
- **(B) Use the published supervised mIoU (31.95 R50 / 43.52 Swin-B) as the ceiling number**, and only
  train/eval OUR label-free same-arch model in-codebase. Fastest, standard in papers, no remap risk.
- **(C) Separate old-mmlab env** (CPU-only eval of the checkpoint, since H100 GPU is blocked). Highest
  fidelity for the ceiling number, slowest, env-fragile.

## Work plan (once ceiling approach is chosen)

1. **Modern FlashOcc backbone** (`ngperception/flashocc/`): wrap the existing modern LSS (bev_pool) +
   R50 image backbone + BEV encoder + **channel2height occ head** (port from
   `Others/FlashOCC/projects/mmdet3d_plugin/.../FlashOCC`). Verify a forward pass + occ shape
   (200×200×16×18) on H100.
2. **Wire the voxel-soft teacher**: reuse the existing label-free target
   (`teacher_cache_2x2soft/voxel10` — soft `weight` + `soft_idx/soft_prob`) and the factorized/soft
   distillation loss (`gaussian4d/train_student.py`). Train the FlashOcc-arch student label-free.
3. **Ceiling**: per the chosen option (A/B/C). Report our label-free mIoU vs the supervised ceiling.
4. **Detection transfer, official CenterHead**: attach mmdet3d CenterHead to the FlashOcc BEV feature;
   **low-label transfer** (2k/4k/8k labels; from-scratch control vs voxel-soft-pretrained; ≥3 seeds)
   → closes the causal chain (better label-free occ → better low-label detection).
5. Report: FlashOcc (label-free voxel-soft) vs FlashOcc (supervised ceiling) on occ; and the low-label
   detection-transfer curve. Our LSS results stand as backbone set #1; FlashOcc = external validation.

## Next paper (not now): 4D label-free forecasting pretraining

Reuse the voxel-soft teacher to generate **temporal** occupied/free/semantic distributions; predict
future occupancy with an existing forecasting architecture (**OccProphet / OPUS-V2 / OccWorld**).
Research question = **forecasting *pretraining* without human 3D/4D labels** — not another generic
world model.

## Env unblock (2026-07-20): bev_pool_v2 builds on H100 — port route is GO

Official FlashOcc torch1.10/cu111 is blocked on H100, but the only hard native dep (`bev_pool_v2`
CUDA op) **compiles and loads on H100** (sm_90) with the modern stack + an isolated CUDA toolkit:

- torch 2.9+cu126 (py310); no system nvcc, so installed an **isolated CUDA 12.6 toolkit**:
  `conda create -p /data/rnd-liu/cudatk -c nvidia cuda-toolkit=12.6` (does NOT touch py310's torch).
- conda's `targets/x86_64-linux/` layout ≠ what torch's cpp_extension expects, so a **merged
  CUDA_HOME** (`/data/rnd-liu/cuda_home2`) via symlinks: `bin→cudatk/bin`,
  `include→cudatk/targets/x86_64-linux/include`, `lib64→cudatk/lib`, `nvvm→cudatk/nvvm`.
- Build env: `CUDA_HOME=/data/rnd-liu/cuda_home2`, prepend `$CUDA_HOME/bin` to PATH,
  `$CUDA_HOME/lib64` to LD_LIBRARY_PATH, `TORCH_CUDA_ARCH_LIST=9.0`.
- Then `torch.utils.cpp_extension.load("bev_pool_v2_ext", ["src/bev_pool.cpp","src/bev_pool_cuda.cu"])`
  from `Others/FlashOCC/projects/mmdet3d_plugin/ops/bev_pool_v2/` → **COMPILE_OK**, extension loads &
  its forward is callable on the H100.

⟹ The FlashOcc port is now a **pure-torch** job (strip mmcv `ConvModule`/`BaseModule`/`force_fp32` +
registry decorators from the ~1289-line 3-file source; R50 from torchvision), reusing this compiled
`bev_pool_v2`. Next: port the 3 modules → load the 562-param checkpoint (near-1:1 key match) →
reproduce ~37.84 mIoU on Occ3D val (the de-risking milestone).
