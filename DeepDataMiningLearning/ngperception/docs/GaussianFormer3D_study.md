# GaussianFormer3D — a study, and what it means for our occupancy net

*LiDAR-camera fusion 3D semantic occupancy with 3D Gaussians (ICRA 2026).*
Paper: [arXiv:2505.10685](https://arxiv.org/abs/2505.10685) ·
Project: <https://lunarlab-gatech.github.io/GaussianFormer3D/> ·
Code (local clone): `/home/lkk688/Developer/GaussianFormer3D`

This doc is a code-verified walkthrough of how GaussianFormer3D (GF3D) works, how to run
it, and an honest comparison with our own [depth-supervised LSS occupancy](../occupancy/README.md)
— ending in a ranked list of things **we** can borrow. File:line references point into the
GF3D clone above.

---

## 1. TL;DR

GF3D predicts a **200×200×16 semantic occupancy grid** (17 classes + free) from **6 cameras
+ a LiDAR sweep stack**. Its one idea: don't carry a dense voxel feature volume — carry a
**set of 25,600 3D Gaussians** (each a position + scale + rotation + opacity + 17 semantic
logits), refine them for 4 blocks with deformable attention over fused image+LiDAR features,
then **splat** them into the voxel grid at the very end. LiDAR shows up three times:
(a) as a **geometry prior that initializes the Gaussian positions**, (b) as **GT depth
supervision** on the image features (BEVDepth-style), and (c) as a **fused feature volume**
the deformable attention samples from.

Headline result (their released weights):

| Benchmark | Modality | IoU | mIoU |
|---|---|---|---|
| nuScenes-**SurroundOcc** | **LiDAR + Camera** | **43.3** | **27.1** |
| RELLIS3D-WildOcc (off-road) | LiDAR + Camera | 33.9 | 13.1 |

For reference, the camera-only parent **GaussianFormer** reports ≈19 mIoU on SurroundOcc —
so most of GF3D's gain is **adding LiDAR** (as input, prior, and supervision), not the
Gaussian representation per se (which the parent already had).

---

## 2. The benchmark matters — SurroundOcc ≠ Occ3D

Before any comparison: **GF3D is scored on SurroundOcc, our net on Occ3D.** These are
different benchmarks and the numbers are *not* directly comparable.

| | SurroundOcc (GF3D) | Occ3D-nuScenes (ours) |
|---|---|---|
| grid | 200×200×16 @ **0.5 m** | 200×200×16 @ **0.4 m** |
| range | [-50,50]×[-50,50]×[-5,3] | [-40,40]×[-40,40]×[-1,5.4] |
| eval mask | **all annotated voxels** (no visibility mask) | **camera-visibility mask only** (`mask_camera`) |
| classes | 17 + free | 17 + free |

The eval-mask difference is the big one: Occ3D only scores voxels the cameras can see, which
is *easier* per-voxel but the two mIoUs live on different scales. Treat cross-benchmark mIoU
as "same league," never head-to-head.

---

## 3. Architecture, end to end

Config: `config/nuscenes_surroundocc_gs25600.py` (+ `_base_/model.py`,
`_base_/surroundocc_pcd_dfa3d.py`). Segmentor: `model/segmentor/bev_segmentor_lidar_3d.py`.

```
6 imgs ──▶ ResNet101-DCN + FPN ─────────────┐
   │                                         ├─▶ GT-depth head (BEVDepth) ─┐
   └────────────────────────────────────────┘                             │
LiDAR (10 sweeps) ─▶ voxelize 0.075m ─▶ HardSimpleVFE ─┐                   │
                                                       ▼                   ▼
                              voxel-to-Gaussian init (25,600 Gaussians) ──▶ Encoder ×4
                                                                          │  { DFA3D deformable attn
                                                                          │    + SparseConv3D
                                                                          │    + FFN/Norm/Refine }
                                                                          ▼
                                              25,600 refined Gaussians ──▶ LocalAggregator (CUDA splat)
                                                                          ▼
                                                          occupancy logits (B,18,200,200,16)
```

### 3.1 Image branch
- **ResNet101 + DCNv2** on stages 3–4, **FCOS3D-pretrained** (`r101_dcn_fcos3d_pretrain.pth`),
  frozen stage-1, caffe style — `config …gs25600.py:109-121`.
- **FPN neck** `start_level=1` → multi-scale features at `embed_dims=128`.

### 3.2 LiDAR branch
- **10 aggregated sweeps** (`LoadPointsFromMultiSweeps sweeps_num=10`), 5-dim points
  `[x,y,z,intensity,Δt]` — `_base_/surroundocc_pcd_dfa3d.py`.
- Voxelized at **0.075×0.075×0.2 m**, `max_voxels=[120k,160k]`, encoded by **HardSimpleVFE**
  (per-voxel mean) — `…gs25600.py:100-107`.

### 3.3 GT-depth head (BEVDepth-style) — note the overlap with our study
- `DepthHead_GTDpt`, `d_bound=[2.0, 58, 0.5]`, `loss_weight=0.5`, **`max_tol=2`** —
  `…gs25600.py:88-99`.
- `d_bound` is **identical to our `DBOUND=(2,58,0.5)`**, and `max_tol=2` is a **±2-bin
  tolerance window** — the exact "tolerant window" idea we A/B-tested. Difference in *role*:
  here the depth head produces the depth features that DFA3D samples along, so the tolerance
  softens a signal feeding attention, not a direct voxel-CE term.

### 3.4 The Gaussian representation
Each of the 25,600 Gaussians is a **28-dim** vector (`model/lifter/gaussian_lifter.py:209`):

| slice | meaning |
|---|---|
| `[0:3]` | xyz mean (position) |
| `[3:6]` | scale (log-space) |
| `[6:10]` | rotation quaternion |
| `[10:11]` | opacity |
| `[11:28]` | 17 semantic logits |

### 3.5 Voxel-to-Gaussian initialization (contribution #1)
`GaussianLifterLiDAR` (`model/lifter/gaussian_lifter.py:167-263`): take the **occupied LiDAR
voxels**, normalize their centres to `[0,1]³`, `inverse_sigmoid` into the position space,
and use them as the initial Gaussian **means** — so Gaussians *start on real geometry*
instead of a uniform grid. If there are more Gaussians than occupied voxels they're sampled
with replacement; fewer, subsampled. LiDAR **intensity** seeds one channel; semantics start
random. This is the "geometry prior" the paper sells.

### 3.6 Encoder — LiDAR-guided 3D deformable attention (contribution #2)
`num_decoder=4` blocks (1 single-frame + 3 with SparseConv), operation order
`…gs25600.py:209-221`:
```
block 0:            deformable → ffn → norm → refine
blocks 1..3: spconv → norm → deformable → ffn → norm → refine
```
- **DFA3D** (`model/encoder/gaussian_encoder/deformable_module_3d.py`): each Gaussian emits
  **9 keypoints** = 7 fixed offsets (`0`, `±0.45 m` on each axis) + **2 learnable**. Keypoints
  project into the images, and the attention samples **image features *and* a depth
  dimension** (learned `sampling_offsets` **and** `sampling_offsets_depth`, normalized by
  `[W,H,D]`) — i.e. sampling in a *lifted 3D* feature space, not just 2D. Runs through the
  `MultiScale3DDeformableAttnFunction` CUDA kernel (from the DFA3D project).
- **SparseConv3D** between blocks lets nearby Gaussians talk (spatial diffusion) at grid 0.5 m.
- **Refinement** (`SparseGaussian3DRefinementModule`, `restrict_xyz=True, unit_xyz=[4,4,1]`)
  updates each Gaussian's mean/scale/rot/sem/opacity per block — the "deform" step.

### 3.7 Gaussian-to-voxel splatting
`GaussianHead` → `LocalAggregator` CUDA op (`model/head/localagg/…`): for each of the
640,000 voxel centres, sum over nearby Gaussians of `opacity · exp(−½ Mahalanobis) ·
semantic_logits`, giving `(200,200,16,18)` logits. A per-Gaussian **radius** bounds the
neighbourhood (`ceil(max_scale·3 / 0.5)`). One giant **"empty" Gaussian** (`scale=[100,100,8]`)
supplies the free-class baseline. Class count is **compiled into the CUDA op** (18 for
nuScenes, 9 for WildOcc → recompile).

### 3.8 Losses
`OccupancyLoss` (`loss/occupancy_loss.py`, config `…gs25600.py:38-59`):
- **voxel cross-entropy ×10**, class-balanced (`manual_class_weight`, 18 values in [0.5,1.31]);
- **Lovász-softmax ×1** (directly optimizes IoU), ignoring free;
- **depth loss ×0.5** (the GT-depth head);
- `apply_loss_type='random_1'` — supervise one randomly chosen decoder block per step
  (cheap deep supervision).

### 3.9 Training
AdamW `lr=1e-4` (backbone ×0.1), `wd=0.01`, grad-clip 35, **24 epochs, batch 1 × 8 GPUs**
(`…gs25600.py:26-36`).

---

## 4. How to install & run

**Heads-up:** GF3D is a heavy mmdet3d stack with **three custom CUDA builds**. Standing it up
is comparable to the GaussianOcc effort we already went through — plan for a dedicated conda
env and a GPU that can compile against your CUDA toolkit.

### Environment (`docs/installation.md`)
```bash
conda create -n gf3d python=3.8.16 && conda activate gf3d
# CUDA 12.1 line:
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install openmim && mim install mmcv==2.1.0 mmdet==3.2.0 mmsegmentation==1.2.1 mmdet3d==1.3.0
pip install spconv-cu117 timm
# CUDA ops (build all three):
cd model/encoder/gaussian_encoder/ops && pip install -e . && cd -   # 2D deformable + splat helpers
cd model/head/localagg && pip install -e . && cd -                  # Gaussian→voxel splat (NUM_CHANNELS=18)
git clone https://github.com/IDEA-Research/3D-deformable-attention.git && cd 3D-deformable-attention/DFA3D && bash setup.sh 0
```

### Data
- nuScenes v1.0 full + their `depth_gt/` + `nuscenes_infos_gf3d_{train,val}.pkl` (provided);
- SurroundOcc occupancy GT (`data/surroundocc/samples/*.pcd.bin.npy`);
- image backbone `ckpts/r101_dcn_fcos3d_pretrain.pth`.
- We already have nuScenes trainval at `/mnt/e/Shared/Dataset/NuScenes/v1.0-trainval`, but
  **SurroundOcc GT ≠ our Occ3D `gts/`** — GF3D needs the SurroundOcc `.npy` labels, a separate
  download.

### Run (their released weights)
```bash
# eval on SurroundOcc val:
python eval.py --py-config config/nuscenes_surroundocc_gs25600.py \
  --work-dir out/nuscenes_surroundocc_gs25600/ \
  --resume-from out/…/surroundocc_release.pth
# visualize occ + the Gaussians themselves:
CUDA_VISIBLE_DEVICES=0 python visualize.py --py-config config/nuscenes_surroundocc_gs25600.py \
  --work-dir out/… --resume-from out/…/surroundocc_release.pth --vis-occ --vis-gaussian --num-samples 3
# train (8 GPUs):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --py-config config/nuscenes_surroundocc_gs25600.py --work-dir out/…
```

For *learning*, the payoff is porting ideas into our pure-PyTorch net (§6), not reproducing
their 8-GPU run.

---

## 5. GF3D vs our depth-supervised LSS occupancy

| axis | **GaussianFormer3D** | **Our LSS occupancy** ([README](../occupancy/README.md)) |
|---|---|---|
| inference modality | **LiDAR + camera** (fusion) | **camera-only** (LiDAR = train-time depth CE) |
| scene representation | **25,600 sparse 3D Gaussians** → splat | **dense 200×200×16 voxel** features + 3D CNN |
| lift | iterative: **4× deformable-attention refine** | **single-shot** LSS depth-distribution lift |
| geometry prior | LiDAR **voxel-to-Gaussian init** | none (uniform frustum) |
| depth supervision | BEVDepth GT depth, **`max_tol=2`**, `d_bound=[2,58,0.5]` | LiDAR depth CE, `DBOUND=(2,58,0.5)` (same bins) |
| occ loss | **CE×10 + Lovász×1 + class-balanced** + deep sup. | **plain CE** on `mask_camera` |
| backbone | ResNet101-DCN (FCOS3D) | frozen **DINOv2-base** / ResNet18 |
| deps / infra | mmdet3d + 3 CUDA builds, 8 GPUs, 24 ep | **pure PyTorch**, 1× RTX 3090 |
| benchmark / score | SurroundOcc, **mIoU 27.1 / IoU 43.3** | Occ3D (cam-mask), **mIoU 0.216 / geo-IoU 0.681** |

**Reading it fairly.** GF3D is a mature fusion system; ours is a compact camera-only teaching
baseline. Their headline number rides largely on **LiDAR at inference** — a modality we
deliberately don't use at test time. Where the comparison is genuinely instructive is the
**three levers they have that we don't**, two of which our own experiments already pointed at:

1. Our README/TUTORIAL already concluded **"the single-shot lift is the bottleneck."** GF3D's
   answer is exactly **iterative refinement** (4 deformable-attention passes). Strong external
   confirmation of our own diagnosis.
2. Our depth-supervision study found the **tolerant window** neutral *as a direct voxel term*.
   GF3D keeps a `max_tol=2` window but in a **different role** (softening depth features that
   feed attention) — consistent with "it's not useless, it just wasn't the lever where we put
   it."
3. They optimize **Lovász + class-balanced CE**; we optimize plain CE. Lovász directly targets
   IoU and is **free to add** to our loss with no new dependency.

---

## 6. What we can do next (ranked by value ÷ effort for *our* codebase)

All of these are portable into our pure-PyTorch `train_lss.py` / `lss_occ.py` — no mmcv, no
CUDA builds. Per our house rule, anything that looks like a win gets a **multi-seed** check
before we believe it (we already have `--seed` + caching).

1. **Lovász-softmax + class-balanced CE** *(cheapest, do first).* Add a Lovász term and
   nuScenes class weights to `occ_loss` in `train_lss.py`. Directly optimizes mIoU; pure
   PyTorch; a few lines. Best effort-to-upside ratio on the board.

2. **Iterative refinement decoder** *(likely biggest camera-only gain).* Our lift is one shot;
   add 2–4 refine blocks that re-attend to image features from the current voxel/BEV state
   (a lightweight deformable or even a plain 3D-conv "refine + residual" loop). This attacks
   the bottleneck **we ourselves identified**, and doesn't require LiDAR at inference.

3. **Optional LiDAR-fusion inference mode** *(biggest absolute jump, changes framing).* Add a
   LiDAR voxel branch (voxelize → simple VFE → concat into our voxel features before the
   decoder). Keep camera-only as the default; expose `--fusion lidar`. This is the single
   largest lever, at the cost of no longer being camera-only.

4. **Geometry-prior query/lift weighting.** Even camera-only, weight the lift by the predicted
   **depth confidence** so mass concentrates on likely surfaces (a soft analogue of their
   voxel-to-Gaussian init). In fusion mode, initialize directly from LiDAR occupancy like they do.

5. **Deep supervision** (`random_1`-style): once we have a multi-block decoder (#2), apply the
   occ loss on an intermediate block too. Nearly free.

6. **Sparse / Gaussian representation for memory** *(high effort, high teaching value).* The
   full GF3D idea — carry Gaussians, splat at the end. Needs custom CUDA (localagg) to be
   efficient; a pure-PyTorch toy version (few-thousand Gaussians, differentiable splat) would
   be a great *tutorial* even if slower. Lowest priority for a metric win, highest for
   pedagogy.

**Suggested first experiment:** #1 (Lovász + class weights) on the existing DINOv2-small / 1k
baseline, seed-swept, since it's a one-file change and directly targets the metric. If it
clears noise, layer in #2 (refinement) next.

---

*See also:* [../occupancy/README.md](../occupancy/README.md) (our LSS net + results),
[../TUTORIAL.md](../TUTORIAL.md) §2.7.1 (our depth-supervision study, incl. the tolerant-window
and resolution ablations referenced above).
