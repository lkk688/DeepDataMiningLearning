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

## 5. Ablation #2 — trained VGGT backbone (next)

The geometry probe justifies the heavier build: wrap VGGT as a **frozen `--backbone vggt`** encoder
in `occupancy/models/lss_occ.py` (parallel to the DINOv2 `CamEncoder`) and A/B it on **camera-only
occupancy mIoU** (the weakest, highest-headroom path):

| arm | camera encoder | expect |
|---|---|---|
| baseline | frozen DINOv2 + LSS depth lift | camera-only occ ~0.30 (our current) |
| **VGGT-frozen** | frozen VGGT features + its depth → lift to voxels | ↑ camera-only occ (3-D-aware) |
| VGGT + our loss/refine | VGGT + Lovász/class-bal + refine + metric-scale head | best camera-only |

Implementation notes from the probe: (a) VGGT ingests any H,W divisible by 14 (we lifted at
518×294, ~16:9) and runs 6 cameras in **3.8 s / 9 GB** frozen — cache its features/depth once to
keep training fast; (b) the trained head must **solve scale** (metric depth head or extrinsic
baselines), since raw VGGT depth is up-to-scale. Success = camera-only occ mIoU clearly above the
DINOv2 baseline (toward the 40s SOTA band), then carry the winner into the label-efficient
occupancy→detection transfer that is our actual contribution (§3).

## Sources
Occ3D (tsinghua-mars-lab.github.io/Occ3D); Occ3D-nuScenes benchmark (emergentmind); GaussRender
(ICCV'25); EFFOcc (arXiv 2406.07042); DAOcc (arXiv 2409.19972); FusionOcc (ACM MM'24); VGGT-Det
(CVPR'26 repo); VG3T (arXiv 2512.05988); DVGT (CVPR'26); DriveVGGT (arXiv 2511.22264).
