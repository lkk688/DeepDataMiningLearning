# Closed-Loop Occupancy → Planning: turning the negative open-loop result into a positive one

## 0. Why this project
Our backbone-transfer study (ngperception/docs/PAPER_DRAFT.md) produced strong but **negative/decoupling**
findings: occupancy is a good *detection* pretext yet a **null open-loop-planning input** (occ+ego ≈ ego on
nuScenes L2; §4.7). That is a *metric* limitation, not evidence against occ→planning — nuScenes open-loop L2
is ego-status-(constant-velocity)-dominated (BEV-Planner/AD-MLP), and Tesla-style 4D-occ→planning value shows
up only in **closed-loop, reactive, long-tail** driving that open-loop L2 cannot see.

**Goal (the positive result we need to publish):** show, in a *closed-loop reactive* simulator, that a
**label-free occupancy-conditioned policy improves driving over a matched non-occ (image/ego-only) policy** —
lower at-fault collisions / better progress / better lane-keeping. This converts our negative open-loop result
into a positive closed-loop contribution and keeps the label-free thread ([[label-efficiency-positive]],
[[srsd-next-paper]]).

## 1. Hypothesis
> H1: In closed loop, conditioning the planner on (label-free) occupancy reduces at-fault collisions and
> off-road/lane violations vs. the same planner without occupancy, **especially in reactive / long-tail
> scenarios** where the constant-velocity prior fails. Δ(occ − no-occ) > 0 is the contribution.

The controlled comparison mirrors §4.7's rigor: **same policy architecture, ablate only the occ input**, but
now scored **closed-loop** instead of open-loop L2.

## 2. Simulator = NVIDIA AlpaSim (copied here, code-only)
`closedloop_occ/alpasim/` = the NVIDIA AlpaSim closed-loop stack copied from the thesis
(`/data/cmpe258-sp24/010892622/thesis/e2e/diffusion/thesis-nurec/alpasim`), **code only** (67 MB; excluded
`runs/ outputs/ data/ .venv .git *.sif *.usdz`). Microservices: `driver / renderer(runtime) / physics /
controller / eval / wizard`. Containers (`nre-ga.sif`, `alpagym_toolchain.sif`, `pytorch-25.10-py3.sif`) stay
referenced in place at the thesis `containers/` — multi-GB, not copied.

**Driver is pluggable (the key enabler).** Drivers run as an external service that AlpaSim connects to;
`alpasim/plugins/transfuser_driver/` is a complete worked example of a *custom* camera+LiDAR→trajectory
planner (TransFuser) wired in, targeting CARLA/NAVSIM/WAYMO. Our occ policy = a new plugin cloned from it.

**Scoped-OUT per direction:** HD-map injection, Argoverse2 data, lane-marking-legality reward, GRPO/RL. We use
**only the base sim + closed-loop scoring** (collision / progress / lane-following scorers in `src/eval`).

## 3. Data = NVIDIA PhysicalAI-AV (local, no AV2, no map)
`/data/rnd-liu/Datasets/PhysicalAI-AV` = raw sensor logs (camera ring + lidar + radar + calibration + labels)
— the right ingredients for NuRec, but **no pre-built reconstructions**. Reconstruction path (adapt the AV2
pipeline, which itself derives from NVIDIA's PandaSet recipe in `alpamayo-recipes/recipes`):
PhysicalAI logs → nCore converter → NuRec 3DGUT (≈2.5 h/scene, 1×H100, `nre-ga.sif`) → `last.usdz` →
AlpaSim drives inside. **No map injection.**

## 4. Policy = occupancy-conditioned planner (the ablation)
Clone `transfuser_driver` → `occ_driver`:
- **Backbone / occ provider (two candidates):**
  - **(a) Our DINOv2-LSS occ** (ngperception, the study's winner) — supervised-pretext occ.
  - **(b) GaussianOcc** (`/fs/atipa/data/rnd-liu/Others/GaussianOcc`, ckpts present) — **label-free** self-sup
    occ (mIoU 11.26 repro; [[gaussianocc-repro]]). Preferred for the label-free story **if** camera coverage
    works out (see risks).
- **Planner head:** the CV-residual transformer from `ngperception/occupancy/train_planning.py` (already
  debugged: CV prior + residual, ego-status token), re-used so occ enters as extra tokens.
- **Ablation arms (closed-loop):** `ego/image-only` vs `+occ(DINOv2)` vs `+occ(GaussianOcc, label-free)`.

## 5. Milestones (de-risked order)
- **M0 — sim smoke test:** bring AlpaSim up under Apptainer host-venv on one already-reconstructed thesis
  scene (borrow one `last.usdz`); run the `manual` or `transfuser` driver end-to-end; read `src/eval` scores.
  *De-risks the stack before any Physical-AI reconstruction.*
- **M0.5 — (fallback) NAVSIM path:** the transfuser driver already targets NAVSIM (navtest/PDMS, log-replay,
  no neural recon). If NuRec/Physical-AI recon proves too heavy, run the occ-vs-no-occ ablation on NAVSIM PDMS
  first — a faster, standard semi-closed-loop positive result.
- **M1 — occ_driver plugin:** clone transfuser plugin; stub occ=zeros → confirm it drives; then wire real occ.
- **M2 — reconstruct Physical-AI scenes:** Physical-AI→nCore converter + 3DGUT on ~10–20 scenes (no map).
- **M3 — closed-loop ablation:** ego-only vs +occ(DINOv2) vs +occ(GaussianOcc) on held-out scenes → scores.
- **M4 — result + paper:** if Δ(occ − no-occ) > 0 (esp. collisions/reactive), that's the positive contribution.

## 6. Risks / open decisions
- **Camera coverage.** The thesis driving loop used **front-camera-only** (Alpamayo). GaussianOcc/DINOv2-LSS
  are **surround** (6-cam). Must confirm AlpaSim can render the Physical-AI multi-cam rig to the driver, else
  occ quality collapses. *Decision:* verify multi-cam rendering in M0; if front-only, retrain a front-cam occ
  or use LiDAR-occ.
- **Domain gap.** GaussianOcc/DINOv2-LSS are nuScenes-trained; Physical-AI + NuRec-rendered views differ.
  May need light fine-tune/adapt of the occ provider on Physical-AI.
- **Reconstruction cost/format.** Physical-AI→nCore converter does not exist yet (AV2 one does); ~2.5 h/scene.
  M0.5 NAVSIM fallback exists if this blocks.
- **Cross-repo/container deps.** `ncore` is private in the thesis (`drwx------`); containers are multi-GB and
  referenced in place. Needs the thesis owner's containers to remain accessible (coordinate).
- **Reactivity of the metric.** Even closed-loop, if scenes are "easy" (wide roads, on-route start) the effect
  can be small (the thesis saw `offroad=0` degenerate). Target **reactive/long-tail** scene subsets.

## 7. Status
- [x] AlpaSim code copied here (§2). GaussianOcc + PhysicalAI-AV located. Driver contract understood.
- [x] **M0 PASSED (2026-07-26)**: full AlpaSim closed-loop runs on OUR H100, OUR copy, NO SLURM. GT-replay
      reproduced the gold-standard scores (lane_centering 0.24m, offroad 0, collisions 0, correct_lane 1.0)
      + NuRec mp4. No-SLURM backend + on-disk /tmp bind + bind-source fixes all in run_local_gtreplay.sh /
      deployment/singularity.py. (Details in [[closedloop-occ-alpasim]] memory.)
- [~] M1a (IN PROGRESS): occ_driver plugin built + installed (`plugins/occ_driver`, entry-point `occ`,
      registry-discoverable) + `driver=occ` config. `OccModel(BaseTrajectoryModel)` implements the full
      contract; STUB = constant-velocity-from-ego-speed = the **ego-only / no-occ ablation arm** (use_occ=false).
      NEXT: closed-loop drive test on an AV2 recon scene (`driver=occ +cameras=av2_1cam +exp/sim=av2_recon`,
      NO force_gt) -> first ego-only closed-loop score.
- [x] **M1b DONE**: occ provider = DA3 metric depth (front-cam). OccModel caps the CV trajectory at the
      nearest obstacle (DA3 p5 depth in the forward corridor). DA3 installed non-editable into the alpasim
      venv (+ small deps + export-path patch); DA3 loads and runs per-step inside the closed-loop container.
- [x] **M3/M4 POSITIVE RESULT (2026-07-26, 1 AV2 scene)**: ego-only (no perception) collision_at_fault
      **1.00 -> 0.00** with +occ; lane_centering 0.53 -> 0.15. Perception (DA3 occ/depth) eliminates the
      at-fault collision the ego-status CV prior drives into. (collision_any stays 1.0 = rear/blameless from
      pure-braking; at-fault is the meaningful metric.) THE positive result the open-loop null couldn't show.
- [ ] Next: multi-scene ablation (stats over the AV2 val split) + add steering-avoidance to also cut rear
      collisions; then M2 (PhysicalAI-AV recon) for cross-dataset, and semantic occ (DA3 depth + 2D seg).
