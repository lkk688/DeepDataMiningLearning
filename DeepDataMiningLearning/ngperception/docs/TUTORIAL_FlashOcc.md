# Tutorial: FlashOcc / BEVDet-Occ as an external backbone (label-free voxel-soft)

This tutorial sets up **FlashOcc/BEVDet-Occ** on our modern H100 stack and trains it **label-free**
from our **voxel-soft teacher** (no human 3D labels), with the public **supervised** checkpoint as the
upper bound. It is the external-validation backbone for the label-free occupancy result (see
`RESULTS_LABELFREE_OCC_2x2_TRANSFER.md`); our own LSS model is backbone set #1.

Rationale and decisions are in `PLAN_FLASHOCC_MIGRATION.md`. Two backbones:
- **Single-frame `BEVDetOCC-R50`** (supervised ceiling mIoU 31.95) — current-frame only, reuses our
  existing camera+calib data pipeline. **This is where we start training** (simplest, fastest).
- **`BEVStereo4DOCC-R50-4D-stereo`** (supervised ceiling mIoU 37.84) — temporal + stereo; the chosen
  config for the checkpoint-remap ceiling. Added after single-frame is validated (needs adjacent-frame
  + stereo data).

---

## 0. Why not run official FlashOcc directly?

Official FlashOcc installs **torch 1.10 + cu111 + mmcv-full 1.5.3 + mmdet 2.25** (old mmlab 1.x). That
torch/CUDA **cannot run on an H100 (sm_90 needs cu11.8+/torch 2.x)**. So we **port** the model into our
modern stack (torch 2.9 + cu126, py310) and reuse only its one CUDA op. The grid matches Occ3D exactly
(`x,y∈[-40,40]@0.4`, `z∈[-1,5.4]` → 200×200×16), so the occ output aligns 1:1 with our teacher target.

## 1. Clone + checkpoint

```bash
cd <repo>
git clone --depth 1 https://github.com/Yzichen/FlashOCC.git Others/FlashOCC   # gitignored
# R50-4D-stereo supervised checkpoint (ceiling weights + port spec; 235 MB):
python -m gdown 12WYaCdoZA8-A6_oh6vdLgOmqyEc3PNCe -O Others/FlashOCC/ckpts/flashocc-r50-4d-stereo.pth
```

Checkpoint = 562 params; prefixes `img_backbone / img_neck / img_view_transformer /
img_bev_encoder_backbone / img_bev_encoder_neck / pre_process_net / occ_head`. The occ head is the
"channel-to-height" trick: `final_conv` (256→256 conv3×3) → MLP `predicter` (256→512→**288 = 16×18**)
→ reshape to (H, W, Dz=16, classes=18).

## 2. CUDA op (`bev_pool_v2`) — the one native dependency

No system nvcc, so install an **isolated** CUDA 12.6 toolkit (does NOT touch py310's torch):

```bash
conda create -y -p /data/rnd-liu/cudatk -c nvidia cuda-toolkit=12.6
# conda uses a targets/ layout; assemble a merged CUDA_HOME torch's cpp_extension understands:
CH=/data/rnd-liu/cuda_home2; TK=/data/rnd-liu/cudatk
mkdir -p $CH
ln -sf $TK/bin $CH/bin
ln -sf $TK/nvvm $CH/nvvm
ln -sf $TK/targets/x86_64-linux/include $CH/include
ln -sf $TK/lib $CH/lib64
```

Build/load env (needed for every run that touches the op):

```bash
export CUDA_HOME=/data/rnd-liu/cuda_home2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST=9.0            # H100 sm_90
```

The op is JIT-loaded on first import via `torch.utils.cpp_extension.load` from
`Others/FlashOCC/projects/mmdet3d_plugin/ops/bev_pool_v2/src/{bev_pool.cpp,bev_pool_cuda.cu}`
(**verified: COMPILE_OK on H100**). Note the old prebuilt `bevdet/…/bev_pool_ext.so` is the v1 API and
is **not** compatible with v2 — don't use it.

## 3. The ported model

`ngperception/flashocc/model.py` — standalone modern-torch port (mmcv `ConvModule`/`BaseModule`/
`force_fp32` stripped, registries removed; ResNet50 from torchvision), reusing the compiled
`bev_pool_v2`:

```
6 imgs (B,6,3,256,704) ─ ResNet50(layer3,layer4) ─ CustomFPN → 256ch
   → LSSViewTransformer (DepthNet → depth dist + context; bev_pool_v2 splat) → BEV (200×200×64)
   → CustomResNet BEV encoder + FPN_LSS → 256
   → BEVOCCHead2D (channel2height) → occ logits (B,18,200,200,16)
```

Inputs come from the existing `gaussian4d/train_student.py:StudentDataset`: `imgs, intrins (K@256×704),
rots (cam→ego), trans (cam→ego)`; `post_rots=I, post_trans=0, ego2global=I, bda=I`.

Smoke test: `python -m DeepDataMiningLearning.ngperception.flashocc.smoke_model` (with the CUDA env
above) → prints occ shape 200×200×16×18 and finite values.

## 4. Label-free training from the voxel-soft teacher

Reuse the label-free target and loss we already built:
- teacher target cache: `teacher_cache_2x2soft/voxel10` (soft `weight` + top-K `soft_idx/soft_prob`).
- loss: `teacher_loss_factorized` (or `teacher_loss_soft`) in `gaussian4d/train_student.py` — geometry
  (binary occ/free) + 17-way conditional-on-occupied soft-CE, uncertainty on semantics only.

```bash
# (CUDA env from §2 exported)
python -m DeepDataMiningLearning.ngperception.flashocc.train_flashocc \
    --nusc  <nuscenes>/v1.0-trainval --gts <nuscenes>/v1.0-trainval/gts \
    --teacher-cache <nuscenes>/teacher_cache_2x2soft/voxel10 \
    --model bevdet_r50 --epochs 24 --batch-size 4 --amp --factorized \
    --out-dir output/flashocc_r50_voxelsoft
```

Eval on Occ3D val (our evaluator, camera-visible voxels): `flashocc/eval_flashocc.py` → mIoU / geo /
tail. Report label-free FlashOcc-R50 vs the supervised ceiling.

## 5. Ceiling (checkpoint remap) — R50-4D-stereo

Port the stereo view transformer + temporal alignment (`LSSViewTransformerBEVStereo`, `pre_process_net`,
BEVDet4D BEV shift) to match the 562 checkpoint keys, load `flashocc-r50-4d-stereo.pth` (near-1:1 key
match), and eval on Occ3D val → should reproduce **mIoU ≈ 37.84**. This validates the full port and is
the supervised upper bound for the same-architecture label-free number.

## 6. Detection transfer (low-label, official CenterHead)

Attach mmdet3d **CenterHead** to the FlashOcc BEV feature; **low-label** transfer (2k/4k/8k labels;
from-scratch control vs voxel-soft-pretrained; ≥3 seeds; fp32 — amp corrupts BN → eval 0.000). Closes
the causal chain: better label-free occ pretraining → better low-label detection.

## Results (label-free, Occ3D val, camera-only)

Same voxel-soft teacher + same 2044 frames as the LSS backbone set #1 — apples-to-apples across backbones:

| model (label-free)              | mIoU   | geo-IoU | tail-IoU |
|---------------------------------|--------|---------|----------|
| LSS voxel-soft (backbone #1)    | 0.1040 | 0.396   | 0.0367   |
| **FlashOcc-R50 voxel-soft**     | 0.0926 | **0.4367** | 0.0223 |
| FlashOcc-R50 supervised (ceiling) | 0.3195 | —     | —        |

External validation: the label-free teacher transfers to a different backbone (BEVDet/LSS+R50) at
comparable mIoU and **better geometry** (explicit depth lift). Label-free gap to ceiling (~3.4×) is
expected — this run = 2044 frames, no depth supervision, single-frame, no aug. Levers to close it:
full teacher cache, LiDAR-depth supervision on the lift, image aug, 4D-stereo.

## Status

- [x] FlashOcc cloned, R50-4D-stereo checkpoint downloaded + mapped (562 params).
- [x] `bev_pool_v2` CUDA op builds + loads on H100 (isolated cu126 toolkit).
- [x] Single-frame `BEVDetOCC-R50` port + smoke test (forward+backward OK).
- [x] Label-free training from voxel-soft teacher → mIoU 0.0926 / geo 0.4367 (24 ep, 2044 frames).
- [ ] Scale label-free run: full teacher cache + LiDAR-depth supervision on the lift + aug.
- [ ] R50-4D-stereo port + checkpoint remap → reproduce 37.84 (ceiling).
- [ ] Low-label CenterHead detection transfer.
