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

## 6b. Full target architecture (multi-modal teacher) + VGGT as a teacher densifier

```
 training data ─┬─ LiDAR rays ──── metric geometry + ray-verified FREE space
                ├─ VGGT ────────── dense points + temporal tracks (LiDAR-anchored, filtered)
                └─ 2D VFM ──────── semantic pseudo-label distributions
                              │
                              ▼
             uncertainty-aware 3D/4D Gaussian teacher
             (per Gaussian: μ metric pos, Σ surface covariance, α occupancy conf,
              s semantic distribution, v temporal velocity)
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
    continuous 3D query loss          render2d camera-consistency loss (aux)
             └────────────────┬────────────────┘
                              ▼
              multi-camera temporal student  →  camera-only occupancy / detection / forecasting
```

**VGGT as a teacher *densifier* (the narrative flip).** VGGT failed as a *student* prior (§5 / the
VGGT ablations: depth-prior null, features −33%) because it is **dense but scale-drifting**. That
same property makes it valuable *offline, LiDAR-anchored*: LiDAR fixes metric scale + gives reliable
surfaces; VGGT fills the space *between* LiDAR beams (range / thin structures); **keep only VGGT
points that are multi-view-consistent AND close to a LiDAR point, inside the camera frustum**, each
with a **confidence α from its LiDAR-agreement × VGGT-confidence** (never uniform weight). So VGGT is
useless as an unanchored inference prior but fills the teacher's real weakness (LiDAR sparsity). Same
model, opposite role — a defensible, non-obvious use.

## 6c. Current status vs the target (the running gate is a *lower bound*)

The Phase-1 code (`gaussian4d/`) implements a **minimal** Gaussian teacher; several pieces that
*create* the Gaussian advantage are not in yet, so a *tie* in the current gate is **not** a clean
no-go:

| target component | current `gaussian_teacher.py` | status |
|---|---|---|
| ray-aware free/occupied/**unknown** | occupied only; rest = "free" (no ray-cast) | **TODO #1** |
| Σ surface covariance | isotropic scalar σ (kNN spacing) | partial |
| s semantic **distribution** | hard argmax (cache only stores argmax) | partial |
| v temporal velocity (4D) | none (static) | Phase 3 |
| continuous 3D **query** loss | voxel-CE only | TODO #3 |
| render2d as student aux | not wired | TODO #3 |
| VGGT densifier | none | TODO #4 |

**Staged build (each step must improve the *student*, else stop):**
1. **ray-aware free/occupied/unknown** — cheap, high value (real free-space supervision + stop
   penalising unobserved voxels). Add to **both** teachers.
2. **anisotropic Σ + FM semantic distribution** (needs the labelgen cache to also store per-class
   confidence, not just argmax).
3. **continuous 3D query loss**; re-wire render2d as the camera-consistency aux.
4. **VGGT densifier** (LiDAR-anchored, frustum-limited, confidence-weighted).
5. **temporal v → 4D forecasting** (Phase 3).

**Reviewer-defense rule (applies throughout):** any improvement that is *not the Gaussian
representation itself* (ray-free-space, VGGT densify, better semantics) must be applied to the
**voxel baseline too**, so the Gaussian-vs-voxel gate stays a clean representation comparison.

## 6d. Phase-1 gate results (so far)

Camera-only student on Occ3D-nuScenes val; the *only* difference between arms is the teacher.
2044-frame label-free pretraining (mixed scenes — the *delta* between teachers is the clean signal;
absolute is low because the teacher is noisy + camera-only, Occ3D-GT camera ceiling ≈ 0.30).

| version | teacher | mIoU | geo-IoU | tail-IoU |
|---|---|---|---|---|
| **minimal** (isotropic σ, hard argmax sem, no ray-free, voxel-CE) | voxel-10sweep | 0.0913 | 0.2938 | 0.0467 |
| | **gaussian-1sweep** | **0.0947** | 0.2869 | **0.0486** |
| **#1 ray-aware** (free/occupied/**unknown**) | voxel-10sweep | **0.0920** | **0.3597** | **0.0335** |
| | gaussian-1sweep | 0.0852 | 0.2799 | 0.0305 |

**The ray-aware upgrade FLIPPED the gate** — voxel10 now wins all three metrics, reversing the
minimal Gaussian edge. **Mechanism (diagnostic):** the isotropic Gaussian **over-fills** (30.9k
occupied voxels vs voxel's 12.1k). With no free-space penalty (minimal) that over-fill *looked* like
a recall advantage; once ray-casting supplies **verified free space**, the over-filled voxels
**conflict** with it — voxel10's geo-IoU jumps 0.294→0.360 while Gaussian's stays flat
(0.287→0.280). So the minimal Gaussian "win" was an artifact of not penalizing over-fill. (Tail-IoU
also dropped for both — the `free_w` balance needs tuning.)

**Verdict:** the current **isotropic** Gaussian teacher does *not* beat the voxel teacher in a fair
comparison — which points precisely at the untested core of the "less-quantization" claim: an
**anisotropic, surface-aligned Σ** (a flat disk on a surface) would not bleed perpendicular into
free space the way an isotropic blob does. That is the make-or-break next test (§6c step 2). Per the
gate discipline: if a surface-aligned Gaussian does not beat voxel10 on tail-IoU, **do not proceed to
4D** — the story is plain LiDAR distillation.

## 6. Positioning

Distinct from: GaussianOcc / GaussianFormer / GaussRender (Gaussian occ, but *not* an offline
uncertainty-aware teacher distilled to a camera-only student for downstream transfer), VG3T/DVGT
(VGGT backbones), Sensor2Sensor (generative sensor synthesis), plain privileged distillation
(no Gaussian / no 4D / no transfer study). The defensible combination is **{4D Gaussian teacher +
uncertainty + camera-only student + label-efficient downstream transfer}**, each shown empirically.
