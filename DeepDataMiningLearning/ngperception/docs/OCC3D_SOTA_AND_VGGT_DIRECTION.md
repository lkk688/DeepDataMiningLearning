# Occ3D-nuScenes SOTA, our standing, and the VGGT direction

Where our occupancy results sit vs the current Occ3D-nuScenes state of the art, and why a
**geometric-foundation camera backbone (VGGT)** is the right next lever — especially for our
weakest link (the camera-only path). Companion to [../occupancy/TUTORIAL.md](../occupancy/TUTORIAL.md)
and [RESULTS_FULLDATA_H100.md](RESULTS_FULLDATA_H100.md).

---

## 1. Occ3D-nuScenes SOTA (metric: mIoU over 17 classes on camera-visible voxels)

**Camera-only:**

| method | mIoU | note |
|---|---|---|
| MonoScene | 6.1 | early |
| CTF-Occ (Occ3D paper baseline, 2023) | 28.5 | the benchmark's own baseline |
| PanoOcc | 42.1 | temporal |
| DHD | 41.5 | +1 history frame |
| **COTR / SHTOcc** | **~44.5** | current camera-only SOTA (2024–25) |
| FMOcc / GaussRender (2025) | 30–40 | 2-frame / Gaussian-render |

**LiDAR-camera fusion:**

| method | mIoU |
|---|---|
| EFFOcc | 51.5 |
| DAOcc | 53.8 |
| **FusionOcc (SOTA)** | **56.6** (with visibility mask) |

**Recent trends (2024–26):** (a) **temporal multi-frame fusion** — every strong camera method uses
history frames; (b) **3D-Gaussian occupancy** (GaussianFormer, GaussRender, VG3T); (c) **sparse
voxels** (SparseOcc, SHTOcc); (d) **weak/self-supervision** (GaussianOcc, GS-Occ3D); (e) the
**RayIoU** metric (SparseOcc) as a thickness-unbiased complement to mIoU; (f) **VGGT
geometric-foundation backbones** (§3).

## 2. How our results compare — honestly

| ours (⚠️ leaked — trained on the 6019 val frames) | vs SOTA |
|---|---|
| **camera-only 0.302 (30.2 mIoU)** | ~14–16 **below** camera SOTA (COTR 44.5); in the CTF-Occ (28.5) tier |
| **fusion 0.558 (55.8 mIoU)** | the **leak inflates it to ≈ fusion SOTA** (FusionOcc 56.6) — not real |

**The fusion number *matching* SOTA is itself the tell that the leakage inflates it.** A clean
train-only retrain would land camera ~25–28 and fusion ~45–50 — competitive mid-tier, not SOTA.

**Why we are below SOTA** (all architectural, not tuning): we are **single-frame** (SOTA is
temporal), **frozen DINOv2** (self-supervised, not 3-D-aware), and **0.4 m grid** (SOTA uses finer).
Our value is not the absolute mIoU — it is the **occupancy→detection transfer** finding
(TUTORIAL §13, 2.5–3.15× over from-scratch), which is orthogonal to the leaderboard.

## 3. The VGGT direction — the right next lever

**VGGT** (Visual Geometry Grounded Transformer, CVPR 2025) is a camera **geometry foundation
model**: from multi-view RGB it predicts dense depth, point maps, camera poses and tracks in one
forward, **without camera calibration**. It is 3-D-aware in a way our frozen DINOv2 lift is not.

- **VGGT-Det** (`github.com/yangcaoai/VGGT-Det-CVPR2026`, cloned at
  `/data/rnd-liu/Others/VGGT-Det-CVPR2026`) is **indoor** 3-D detection (ScanNet/ARKitScenes) —
  *not* our domain. Read it for the **methodology**: sensor-geometry-free, attention-guided 3-D
  queries mined from VGGT's learned geometric priors.
- **The on-point driving references (CVPR 2026, all on nuScenes)** are exploding right now:
  **VG3T** (VGGT as the occupancy backbone, **+1.7 mIoU**, 46 % fewer Gaussians, 16 % faster),
  **DVGT** (Driving Visual Geometry Transformer), **DriveVGGT** (calibration-constrained VGGT).

**Why it fits us:** it attacks our **#1 weakness — the dead camera path** (BEV lift-splat proven
weak: camera-only occ ~30, camera-only det ~0). Swapping the camera encoder from DINOv2 to VGGT
gives a **3-D-aware, calibration-free** backbone. It also matches our own camera-primary pivot and
the LiAuto-GeoX dense-geometry-backbone direction.

**Risk / differentiation:** "VGGT → occupancy" is **crowding fast** (VG3T/DVGT already did it), so
do **not** ship plain VGGT-occ — you'd be scooped. **Differentiate by combining the VGGT backbone
with our contribution:** *label-efficient occupancy pretraining that transfers to detection*. That
fuses a frontier backbone with a claim the VGGT-occ papers don't make, and fixes the camera
bottleneck at the same time.

## 4. Ablation #1 — frozen VGGT geometry, training-free (DONE)

Before any trained integration we ran the cheapest, most decisive probe: **is VGGT's camera
*geometry* actually better than our current camera geometry?** We run frozen `facebook/VGGT-1B`
on the 6 surround cameras, take its dense per-camera depth, back-project with the **known**
nuScenes intrinsics/extrinsics into the ego voxel grid, and score class-agnostic **geometric IoU**
(occupied-vs-free) against the Occ3D GT — **no training, no semantics**. This isolates geometry so
it is directly comparable to the other geometry-only baselines we already had.
Script: `occupancy/vggt_lift_eval.py`. Measured on 50 official-val frames:

| geometry-only method (training-free) | geo-IoU | vs prior camera |
|---|---|---|
| Depth-Anything mono depth-lift (our previous camera geometry) | 0.093 | — |
| **frozen VGGT depth-lift (scale-aligned)** | **0.140** | **+51 %** |
| LiDAR single-sweep oracle | 0.167 | VGGT reaches **84 %** of LiDAR |
| DINOv2 LSS, **trained** (leaked, + learned semantics) — upper bound | 0.669 | trained ceiling |

**Read:** a *frozen, untrained* camera model reaches **84 % of a LiDAR sweep's** occupancy geometry
and beats our prior camera geometry by half — strong confirmation that VGGT is the right camera-path
backbone. **The one caveat is scale:** VGGT is calibration-free, so its depth is **not metric**
(raw geo-IoU ≈ 0.000; a single global scale ≈ **18.8×** recovers the 0.140). A metric mechanism is
therefore required for a real system — either feed VGGT the known camera baselines/extrinsics, or
learn a small scale head. This matches the VGGT-driving literature (DriveVGGT is explicitly
"calibration-constrained" for exactly this reason).

**Command:**
```bash
python -m DeepDataMiningLearning.ngperception.occupancy.vggt_lift_eval \
    --gts <gts> --nusc <nuscenes> --n 50 \
    --vggt-path /data/rnd-liu/Others/VGGT-Det-CVPR2026
```

## 5. Ablation #2 — trained VGGT-depth lift (DONE — negative result)

The probe justified a trained test. We wrapped frozen VGGT depth as a **metric-depth prior** in the
LSS lift (`cache_vggt_depth.py` precomputes VGGT-1B depth → (6,18,50) per token;
`lss_occ.forward(vggt_depth=…)` soft-bins it over the 112 lift bins and log-blends it into the
learned depth; a learned scalar recovers VGGT's ~18.8× scale). Clean A/B — **identical DINOv2
context, identical everything except the VGGT depth prior** — so it isolates *learned depth vs
frozen-VGGT depth geometry*. 2000 train / 300 val frames, 15 epochs, camera-only, LiDAR depth
supervision (`--depth-source lidar`):

| arm | best val mIoU | final (ep14) | best geo-IoU |
|---|---|---|---|
| baseline (DINOv2 + learned depth) | **0.293** | **0.293** | 0.677 |
| DINOv2 + frozen-VGGT depth prior | 0.287 | 0.268 | 0.651 |

**The VGGT depth prior did not help — it tracked slightly *below* the baseline the whole run.** Not
a bug: the VGGT arm's depth loss fell 10→2.4 (the scale/blend *did* reconcile with LiDAR), yet occ
mIoU never beat baseline. **Why the training-free win (§4, 0.140 geo-IoU = 84 % of LiDAR) does not
transfer:**
1. The trained lift with **LiDAR depth supervision already learns better depth** than frozen VGGT
   (learned geo-IoU 0.677 ≫ VGGT training-free 0.140). VGGT's prior competes with an already-strong
   signal and, being scale-ambiguous, adds placement noise rather than information.
2. **Global scale can't fix per-frame scale drift** (16–19× across frames) — one learned scalar
   mis-scales most frames.
3. Occ3D GT + LiDAR depth already teach the decoder geometry, so **camera depth is not the
   bottleneck** in this regime — exactly where §4 (a *no-training, no-LiDAR* probe) made VGGT look
   essential.

**Honest takeaway:** frozen-VGGT depth is a great *label-free* geometry source (§4) but is **redundant
once LiDAR supervision is present** (§5). The naive "inject VGGT depth prior" integration is a dead
end for the LiDAR-supervised setting.

## 6. Where VGGT could still win (next levers, not yet run)

The §5 null result rules out one integration, not the direction. Two settings target VGGT's actual
edge — dense geometry *where LiDAR is absent*:

- **Camera-only *without* LiDAR depth supervision** (`--depth-source occ`, or none). Here VGGT's
  dense prior is the *only* geometry signal; the baseline learned-depth has no LiDAR crutch. This is
  the regime §4 actually measured, and the fair home for VGGT. **Highest-value next run.**
- **VGGT *features* as the backbone** (the original `--backbone vggt`), not just its depth — a
  2048-d 3-D-aware context replacing DINOv2, with a learned depth head. Heavier (cache 2048-d feats
  ≈ 22 MB/frame, or in-loop 3.8 s/6-cam) but tests a different VGGT contribution.
- **Per-frame metric scale** (feed known extrinsic baselines / a learned per-image scale head) —
  removes the §5 failure mode #2 before re-testing the prior.

Whichever wins, the actual contribution remains the **label-efficient occupancy→detection transfer**
(§3), not the occupancy leaderboard. Reproduce §5:
```bash
python -m ...occupancy.cache_vggt_depth --gts <gts> --nusc <nuscenes> --out <cache> --cap 2100
# baseline:  ...occupancy.train_lss --backbone dinov2_base --depth-source lidar --max-samples 2000 --val-samples 300 --epochs 15 --batch-size 4 --occ-lovasz 0.1 --amp
# vggt arm:  (same) + --vggt-depth-cache <cache>
```

## Sources
Occ3D (tsinghua-mars-lab.github.io/Occ3D); Occ3D-nuScenes benchmark (emergentmind); GaussRender
(ICCV'25); EFFOcc (arXiv 2406.07042); DAOcc (arXiv 2409.19972); FusionOcc (ACM MM'24); VGGT-Det
(CVPR'26 repo); VG3T (arXiv 2512.05988); DVGT (CVPR'26); DriveVGGT (arXiv 2511.22264).
