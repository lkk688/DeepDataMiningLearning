# Uncertainty-aware 4D Gaussian occupancy teacher → camera-only student

**Thesis.** Build — *offline, with no human 3D occupancy or box labels* — an **uncertainty-aware 4D
Gaussian occupancy teacher** from raw LiDAR (geometry) + image foundation models (semantics), and
distill it into a **camera-only student**. At inference the student uses **multi-camera only**. The
representation (a) improves *current-frame* occupancy, (b) is a label-efficient pretext for
long-tail 3D **detection**, and (c) extends to **future occupancy forecasting** (4D). This upgrades
our earlier line ("label-free occupancy pretraining helps detection",
[LABEL_FREE_OCC_TO_DETECTION.md](LABEL_FREE_OCC_TO_DETECTION.md)) into a world-model program.

## 1. Why this framing (and why the obvious version isn't enough)

- **The structural fix.** Feature-level fusion lets the strong LiDAR modality dominate and the
  camera branch collapses (we measured camera-only → ~0 in a fused net). So **move fusion out of the
  inference network and into offline supervision / data generation**: LiDAR = geometry teacher,
  camera-FM = semantics teacher, student = camera-only. Fusion becomes a *label engine*, not a
  runtime dependency.
- **But "LiDAR-train, camera-infer" alone is not novel** (privileged distillation is old). The
  novelty must sit in **(i) a high-quality *Gaussian* 4D teacher, (ii) dynamic handling, (iii)
  uncertainty supervision, (iv) downstream transfer.**

### The reviewer's question — "why not simple voxelized-LiDAR BCE?"

A hard-voxel multi-sweep LiDAR teacher is cheap and strong; the Gaussian teacher must earn its place
on **three advantages that voxels cannot match — each demonstrated, not asserted:**

1. **Less hard quantization.** Continuous Gaussians place mass sub-voxel → measurable wins on
   **thin / tail structures** (poles, cones, pedestrians, barriers) at a fixed grid resolution.
   *Evidence:* tail-class IoU and thin-structure IoU vs the voxel teacher at equal grid size.
2. **Measurement + semantic uncertainty.** Each Gaussian carries opacity/scale (geometric
   confidence, from LiDAR density/range) and a semantic distribution (from FM confidence). This is
   only useful if it has a **downstream use** — uncertainty-weighted student loss and/or
   uncertainty-gated pseudo-labels (don't supervise where the teacher is unsure). *Evidence:*
   calibration + ablation "uncertainty weighting on/off".
3. **Continuous time links 3D↔4D.** A Gaussian's position/scale/rotation are functions of time, so
   the *same* representation forecasts future occupancy by advecting/deforming Gaussians — the one
   thing a voxel grid **structurally cannot do**. This is the least-attackable novelty; lead with it.

**Go/no-go gate (Phase 1):** if the Gaussian teacher does not *stably* beat a strong multi-sweep
hard-voxel LiDAR teacher (esp. on tail/thin classes), **do not proceed to 4D** — the story would
collapse to plain LiDAR distillation.

## 2. Architecture

**Offline teacher (per clip, no human labels):**
- **Geometry** ← accumulated multi-sweep LiDAR → fit/initialise 3D Gaussians (position from points,
  scale/opacity from local density; dynamic objects get per-Gaussian velocity/deformation).
- **Semantics** ← `ngdet/labelgen` (SegFormer + Grounded-SAM, already in the Occ3D class space) +
  our metric-depth fusion → paint each Gaussian's semantic distribution.
- **Uncertainty** ← geometric (LiDAR sparsity/range/occlusion) + semantic (FM agreement) → per-
  Gaussian confidence.
- **4D** ← static/dynamic decomposition; dynamic Gaussians carry motion so the field is queryable at
  **arbitrary time t**.

**Camera-only student** (our LSS/DINOv2 occ net, or a query-based head) trained by:
- **Teacher-query loss (MAIN, native-3D):** sample the Gaussian field at 3D query points → supervise
  the student's predicted occupancy/semantics/uncertainty there. Directly geometric; **no ray
  bleeding.**
- **render2d loss (AUX, camera-space consistency):** the existing `render_2d` (volume-render the
  student field to 2D vs FM pseudo-labels) — demoted from primary supervision to a consistency
  regulariser (it is prone to ray-bleeding, hence auxiliary only).
- **Uncertainty-weighted** everywhere: down-weight/omit supervision where the teacher is unsure.

## 3. Three-phase plan

**Phase 1 — validate the core hypothesis (current-frame 3D occupancy only).**
Camera-only *inference* for all arms; report **mIoU, RayIoU, geo-IoU, and tail-class IoU**:
`camera-only baseline` · `render2d` · `raw single-sweep LiDAR-voxel sup` · `multi-sweep LiDAR-voxel
sup` · `3D Gaussian sup` · `Occ3D-GT ceiling`. **Gate:** Gaussian must stably > multi-sweep voxel.

**Phase 2 — representation transfer (the label-efficiency payoff).**
Freeze/finetune the *same student encoder* → detection at **2k/4k/8k** budgets, **tail mAP**,
**≥3 seeds**, vs render2d-pretrain / Occ3D-GT-pretrain / scratch. **Target: stably beat scratch
across *most* budgets** (not a single 4k win — the honest weakness of the current render2d result:
partial, +35% only at 4k). Reuse the harness in LABEL_FREE_OCC_TO_DETECTION.md (fp32 mandatory;
official mAP, not car_AP@0.5).

**Phase 3 — 4D world model (only if Phases 1–2 hold).**
Past cameras → future occupancy at **1s/2s/3s**; static/dynamic decomposition; **arbitrary-time
Gaussian query**; ego-action/trajectory conditioning. **Forecasting first**; wire occupancy cost
into a planner only after forecasting clearly helps.

## 4. What plugs in from existing work

- **render2d** ([train_occ_render2d.py], `render_2d`) → demote to the student aux loss. ✓ built
- **labelgen** (ngdet/labelgen) → the FM-semantics half of the teacher, already in Occ3D classes. ✓
- **LiDAR voxel/depth infra** + **occ→det transfer harness** (Phase 2) with the learned gotchas. ✓
- **occupancy-world-model-teacher** (memory) → the LiDAR+detections pseudo-GT + future-occ generator
  is the seed of the offline teacher. **GaussianOcc** = a *reference* (self-sup Gaussian occ), not
  frozen features to engineer around. **VGGT negatives** = motivation (camera geometry needs a
  teacher). Refs: [refs-mta-sensor2sensor] (Sensor2Sensor = 4DGS data engine), LiAuto-GeoX, SurroundNEXO.

## 5. Risks / de-risking (front-loaded)

| risk | why it matters | de-risk |
|---|---|---|
| **Gaussian ≯ voxel teacher** | kills the "why Gaussian" story | Phase-1 gate first; measure on tail/thin classes at equal grid |
| **Dynamic Gaussian instability** | movers are where Gaussian wins *and* breaks | start static-only 3D; add dynamics only in Phase 3 |
| **Uncertainty = decoration** | reviewers reject unused uncertainty | tie to a concrete loss (weight/gate) + calibration ablation |
| **scope creep (thesis in one paper)** | dilutes each result | **Paper 1 = Phase 1+2**; Paper 2 = Phase 3 (4D) |

## 6. Positioning

Distinct from: GaussianOcc / GaussianFormer / GaussRender (Gaussian occ, but *not* an offline
uncertainty-aware teacher distilled to a camera-only student for downstream transfer), VG3T/DVGT
(VGGT backbones), Sensor2Sensor (generative sensor synthesis), plain privileged distillation
(no Gaussian / no 4D / no transfer study). The defensible combination is **{4D Gaussian teacher +
uncertainty + camera-only student + label-efficient downstream transfer}**, each shown empirically.
