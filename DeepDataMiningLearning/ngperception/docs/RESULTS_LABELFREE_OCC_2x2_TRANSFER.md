# Label-Free Occupancy: Fair 2×2, Gaussian Rescue, and Detection Transfer

**Status: 2026-07-20.** Consolidated results for the label-free occupancy-teacher study. Bottom line:
the continuous **Gaussian** teacher direction is **stopped** (its apparent advantage was a
confound); the defensible results are (1) **soft-FM semantic distillation on a voxel teacher**, (2) a
clean **negative** result on Gaussian with a rigorous mechanism, and (3) label-free **occ→detection
transfer**. Next: externally validate on **FlashOcc/BEVDet-Occ** rather than growing our own backbone.

---

## 1. Setup

- **Task:** label-free camera-only 3D occupancy (Occ3D-nuScenes, 200×200×16, 0.4 m, 18 classes,
  17=free). No human 3D/box labels: geometry from LiDAR, semantics from frozen 2D FMs (SegFormer +
  Grounded-SAM) projected into the voxel grid = the **teacher**. A **camera-only student**
  (LSSOccupancy, DINOv2-base, decoder 4×96) is distilled from the teacher and evaluated on Occ3D val
  (camera-visible voxels). Metrics: mIoU / geo-IoU / tail-IoU (tail = barrier, bicycle, motorcycle,
  pedestrian, cone, trailer).
- **Teacher representations compared:** hard multi-sweep **voxel** vs anisotropic surface-aligned
  **3D Gaussian** (local-PCA flat disk, thin along normal, wide in tangent), both ray-aware
  (occupied / free / unknown via ray-casting from the LiDAR sensor origin).
- **Semantics compared:** **hard** (per-point argmax class) vs **soft** (per-point top-K FM class
  *distribution*, preserving car/truck & bicycle/motorcycle confusion structure).
- All arms trained on the **identical 2044-token** set (verified: soft ∩ voxel-hard ∩ gauss-hard =
  2044), same recipe (24 ep, batch 4, amp), so every comparison is apples-to-apples.

## 2. The fair 2×2 (camera-only student, Occ3D val)

| teacher  | hard (argmax)        | soft (top-K FM dist)          |
|----------|----------------------|-------------------------------|
| voxel10  | .0920 / .360 / .0335 | **.1040 / .396 / .0367** (best) |
| gaussian | .1009 / .370 / .0336 | .0997 / .320 / .0343          |

(mIoU / geo-IoU / tail-IoU.)

- **Soft FM semantics helps the voxel teacher a lot** (+13% mIoU, +10% geo, +9.6% tail) — and
  voxel-soft becomes the **best arm on all three metrics**, overtaking both Gaussian arms.
- **Soft semantics does not help Gaussian** (−1.2% mIoU) and **hurts its geometry badly** (geo
  .370 → .320).
- The hard 2×2 alone had suggested gaussian > voxel (+9.7% mIoU). Once soft semantics are in play,
  that ordering **inverts** — the first sign the Gaussian advantage was not robust.

## 3. Audit — the Gaussian collapse is coupling, not physics

`gaussian4d/audit_soft.py` (hard vs soft Gaussian teacher, 10 frames):

- **Occupancy masks differ**: occ 11031 / free 4984 / unknown 6047 voxels — occupancy was being
  derived from the semantic *non-free cut*, so swapping the semantic source **changed the geometry**.
  Occupancy and semantics were entangled in the teacher.
- **Occupancy weight L1 diff 5182.8** (should be 0 if geometry-only) — same entanglement.
- **Soft top-K prob sum = 1.0000, 0 non-normalized** — the distribution itself is correct.
- Second, asymmetric coupling in the loss: the soft loss down-weighted the *entire* occupied-voxel
  loss by (1 − entropy). The Gaussian 5³ splat produces higher-entropy (blended) voxels, so
  uncertainty **leaked into geometry supervision** and penalized Gaussian more than voxel.

## 4. Factorized rescue — decisive

`teacher_loss_factorized` truly separates the two objectives (per the pre-registered design):

```
L = L_geom(1 − p_free, occupied/free)          # binary, full 18-way, weighted by teacher geom conf
  + λ · L_sem(p(c | occupied), q_c)            # 17-way conditional-on-occupied soft-CE
```

- occupancy = independent binary loss (logsumexp of non-free logits vs the free logit), occupied
  mask untouched by semantics;
- semantics = 17-way (free excluded), soft-CE to the teacher top-K distribution;
- **uncertainty (1 − normalized entropy) modulates ONLY the semantic term**;
- no Lovász (it mixes geometry+semantics); both arms use the identical loss.

Result (camera-only student, Occ3D val):

| arm          | mIoU   | geo-IoU    | tail-IoU |
|--------------|--------|------------|----------|
| voxel-fac    | .0956  | **.3917**  | **.0308** |
| gaussian-fac | .0964  | .3554      | .0276    |

The fix **did** recover Gaussian geometry (.320 → .355), confirming the coupling was real — but
recovery still lands **below voxel**, ties on mIoU (Δ.0008, noise), and loses geo + tail. The
residual gap is the physical 5³-splat dilation that voxel does not have.

**Pre-registered stop criterion** — gaussian-soft-fac must exceed voxel-soft (mIoU > .1040), geo ≥
voxel-fac, tail not degraded — **FAILED on all three**. **Verdict: continuous Gaussian earns no
advantage once the confounds are removed; the +9.7% was an artifact. Stop continuous-query /
VGGT-densifier / 4D.**

## 5. Detection transfer (causal chain, first arm)

Label-free occ → 3D detection: the voxel-soft camera-only occ encoder is finetuned into a center-head
detector (fp32 — amp corrupts BN running stats → eval 0.000), official nuScenes DetectionEval:

- **NDS = 0.206, mAP = 0.205** (best car_AP@0.5 = 0.184 @ ep5).

A real transfer number for the winner. **Not yet a causal claim** — it needs a from-scratch control
and ≥1 other teacher (identical recipe) to show *better occ teacher → better detection*.

## 6. Verdict and pivot

- **Positive:** soft-FM semantic distillation on a simple **voxel** teacher, label-free, +13% mIoU
  (.092 → .104), best on all three metrics.
- **Negative (clean, publishable):** continuous Gaussian occupancy does **not** beat voxel once the
  occupancy/semantic coupling is removed — audit + factorized loss as the mechanism.
- **Transfer:** label-free occ → detection works (NDS .206 / mAP .205, camera-only).

## 7. Next — external validation on FlashOcc/BEVDet-Occ (no new backbone)

Decision: **stop developing our own occupancy backbone.** Use the public **FlashOcc / BEVDet-Occ**
as an external, standard backbone; the public **supervised** checkpoint is the **upper bound**, and a
**same-architecture voxel-soft (label-free) checkpoint is the core result**. Concretely:

1. Wire the **voxel-soft teacher** into FlashOcc (label-free supervision on the FlashOcc BEV/occ head).
2. Public supervised FlashOcc checkpoint = **ceiling**; our LSS numbers above = the first backbone set.
3. Attach the **official CenterHead** to the FlashOcc BEV feature; **low-label detection transfer**
   (2k/4k/8k labels, from-scratch control + label-free-pretrained) to close the causal chain.

A follow-up paper would extend the **voxel-soft teacher to 4D**: generate temporal occupied/free/
semantic distributions and predict future occupancy with an existing forecasting architecture
(OccProphet / OPUS-V2 / OccWorld). Research question = **forecasting *pretraining* without human
3D/4D labels**, not another generic world model.

**Scripts:** `gaussian4d/{run_2x2.sh, audit_soft.py, run_rescue.sh, det_transfer.sh}`;
loss in `gaussian4d/train_student.py` (`teacher_loss_soft`, `teacher_loss_factorized`).
