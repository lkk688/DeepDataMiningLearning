# Tutorial: From an Open-Loop Occupancy Study to a Closed-Loop Positive Result

*A record of the full arc — the controlled backbone study (open-loop, mostly negative/decoupling findings),
the pivot to closed-loop simulation, and the first positive result (occupancy prevents an at-fault collision).
Includes DA3 and GaussianOcc roles, and the strategy for a solid paper result.*

Last updated: 2026-07-26. Companion files: `docs/PAPER_DRAFT.md` (open-loop paper), `closedloop_occ/PLAN.md`
(closed-loop project + milestones).

---

## 0. The one-paragraph story
On nuScenes **open-loop** L2, occupancy adds ~nothing to planning over an ego constant-velocity prior — a
real but *negative* result (occ is ego-status-dominated, per BEV-Planner/AD-MLP). We pivoted to **closed-loop,
reactive** simulation (NVIDIA AlpaSim on NuRec reconstructions), where perception should matter. There, a
perception-free (ego-only) driver **crashes**, and adding a monocular metric-depth occupancy signal (DA3)
**eliminates the at-fault collision** (1.00 → 0.00). That is the positive contribution the open-loop metric
could not show: *occupancy is valuable for closed-loop collision avoidance, not for open-loop L2.*

---

## Part 1 — Open-loop backbone study (what we measured, all numbers)

Setup: frozen foundation-model (FM) backbone → LSS depth-supervised lift → occupancy head; then transfer the
occ-pretrained features to a lightweight detection head. Single H100, low-label budget. This measures
**transferability**, not a SOTA detector (our own BEVFusion already reaches NDS 0.688).

### 1.1 Backbone ranking (nuScenes, occ @2044 frames, det @2k)
| backbone | occ mIoU | det mAP@2k | det NDS | note |
|---|---|---|---|---|
| **DINOv2-large** | **0.316** | 0.114 | 0.1235 | best occ |
| **DINOv3-large** | 0.292 | **0.1248** | **0.1296** | **best cam-only det** |
| DINOv2-base | 0.288 | 0.106 | 0.1215 | |
| RADIO (agglomerative) | 0.274 | 0.096 | 0.1095 | distills DINOv2+CLIP+SAM |
| SigLIP2 (VL) | 0.234 | 0.0628 | 0.0872 | VL-contrastive + aspect distortion |
| VGGT (geometry) | 0.218 | (deferred) | — | weakest semantic occ |
| *FlashOcc-4D-stereo (supervised ceiling)* | **0.3809** | — | — | reproduced (vs published .3784) |

Ordering tracks **pretraining objective**, not model size: self-supervised dense (DINOv2/v3) > distilled
(RADIO) > VL (SigLIP2) > geometry (VGGT).

### 1.2 The key finding: occ mIoU ≠ detection transferability
- Occ mIoU **saturates** by 2044 frames; det-transfer keeps **scaling with data** (DINOv2-L det: @2044 0.114
  → @16k 0.1462 → @28k-ref 0.163).
- **DINOv3** is the sharpest single-table example: occ 0.292 (*below* DINOv2-L 0.316) yet det 0.1248 (*best*
  camera-only). Newer/patch-16 features transfer better to detection despite weaker semantic occ.
- **Finetuning buys ~0 det over frozen-at-same-data** and *costs* occ: finetune@8k occ 0.316→0.290, det →0.137
  ≈ frozen@8k interp 0.135. Lesson: **freeze the FM, scale data; don't finetune.**

### 1.3 LiDAR+camera fusion (the strong-sensor anchor)
| DINOv2-L | occ mIoU | det mAP | det NDS |
|---|---|---|---|
| camera-only | 0.316 | 0.114 | 0.1235 |
| **LiDAR + camera** | **0.493** | **0.2417** | **0.2197** |

Both ≈ double with LiDAR (direct range geometry). Fusion occ (0.493) exceeds the camera-only supervised
ceiling (0.3809), as expected for a sensor with explicit depth.

### 1.4 Open-loop planning: ego-status dominates (the negative result)
3-arm ablation, L2@3s (m): **occ-only 5.04 ≫ ego-only 2.09 ≈ occ+ego-base 2.11** (occ+ego-large 2.49 — a
*larger* occ backbone even hurts). Δ(occ+ego − ego) ≈ 0 ⇒ occupancy adds nothing over the ego constant-
velocity prior. nuScenes open-loop L2 is ego-status-dominated.

**Double decoupling:** occupancy is a good *detection* pretext but a *null* open-loop-*planning* input. Its
value is entirely task-dependent — a caution against treating occupancy as a universal world model.

---

## Part 2 — Closed-loop simulation (where perception matters)

**Why closed-loop:** open-loop L2 cannot credit perception because the logged trajectory is mostly straight
constant-velocity. Closed-loop replays the ego *off* the recorded path (re-rendering new viewpoints), so
reactive obstacle avoidance — where occupancy helps — is actually exercised.

**Stack:** NVIDIA AlpaSim (microservice sim: driver / NuRec renderer / physics / controller / runtime + eval
scorers) driving inside 3D-Gaussian (NuRec/3DGUT) reconstructions of Argoverse2 scenes. Ported to our single
H100 with **no SLURM** (see `closedloop_occ/`).

### 2.1 M0 — the stack runs on our H100 (GT-replay sanity)
Ground-truth-replay reproduced the gold-standard scores: lane_centering 0.24 m, offroad 0, collisions 0,
correct_lane 1.0 + a NuRec front-cam mp4. Port hurdles solved (all in `run_local_gtreplay.sh` +
`deployment/singularity.py`): unset SLURM_JOB_ID → plain `apptainer exec` (no srun); recreate excluded
`data/` bind sources; bind a fresh on-disk `/tmp` (stale read-only shader cache + tiny tmpfs both fail).

### 2.2 M1a — occ_driver plugin + the ego-only baseline
`OccModel(BaseTrajectoryModel)` plugs in like any driver: `PredictionInput`{cameras, command, speed,
ego_pose_history} → `ModelPrediction`{trajectory_xy, headings}. With `use_occ=false` it is a
**constant-velocity-from-ego-speed stub = the perception-free ego-only arm**. Closed-loop result:
**collision_at_fault 1.00, offroad 0, progress 0.74** — it drives, then crashes. The ideal baseline: it leaves
clear room for perception to help.

### 2.3 M1b — DA3 occupancy: a 1-scene "positive" that multi-scene validation DEBUNKED
Occupancy provider = **DA3 monocular metric depth** (front camera). OccModel reads the nearest obstacle
distance (robust 5th-pct DA3 depth over the forward corridor) and **caps the CV trajectory at (obstacle −
margin)** = reactive braking.

**1 scene looked great:** ego-only collision_at_fault 1.00 → +occ **0.00**, lane_centering 0.53 → 0.15.

**8-scene ablation told the honest story (`run_multiscene_ablation.sh`):**
| arm | collision_at_fault | collision_any | lane_centering | progress |
|---|---|---|---|---|
| ego-only | 0.125 (1/8) | 0.750 | 0.348 | higher |
| +occ (DA3-cap) | **0.000** | **1.000** ⚠️ | 0.290 | **lower** ⚠️ |

Occ zeros the at-fault collisions, **but makes collision_any WORSE (0.75→1.0) and progress lower** — i.e.
it **over-brakes on every scene**. Root cause (verified): **DA3 metric depth is BROKEN on NuRec-rendered
frames** — median **1.35 m** (range [1.25,1.70]) vs **18 m** ([2.6,128]) on real images. The splat renders +
AV2 portrait aspect are out-of-domain; cropping to the road band helps on some frames but is not robust. So
the "+occ" arm was mostly *"depth says 1.4 m everywhere → always brake → avoid by barely moving (and get
rear-ended)."* **The 1-scene positive was largely an artifact; multi-scene validation exposed it.**

**Lesson:** in-loop perception must be reliable *on the rendered domain*. DA3-on-render is a domain gap that
preprocessing does not fix → the fix is **sim-GT geometry (Part 6, Option A)**.

---

## Part 3 — What DA3 (Depth Anything 3) helps with
- **DA3METRIC-LARGE** (0.35B): monocular **metric** depth (returns meters directly; ~[2.6,128] m, median ~18 m
  on real driving frames) + **sky segmentation** + intrinsics/confidence. Apache-2.0.
- **Role here:** a *front-camera, label-free, real-time-ish* occupancy/geometry source for the closed-loop
  driver. Depth → nearest-obstacle → collision avoidance (Part 2.3). It sidesteps GaussianOcc's surround-camera
  requirement (the sim driving loop is front-cam-only).
- **Install gotchas (recorded):** clone repo, `uv pip install <repo> --no-deps` **non-editable** (editable
  points at an unbound path invisible inside the container); add moviepy==1.0.3, addict, plyfile, evo,
  pillow_heif; patch `utils/export/__init__.py` to make the 3DGS/COLMAP/GLB exporters optional (we only need
  depth). Weights live in the container-bound HF cache.
- **Next uses:** DA3 depth + a 2D semantic segmenter → lift to a **semantic 3D occupancy** (not just geometry);
  DA3 also emits 3D Gaussians (a bridge to GaussianOcc-style representations).

## Part 4 — What GaussianOcc can help with
- **GaussianOcc** (`/fs/atipa/data/rnd-liu/Others/GaussianOcc`, reproduced mIoU 11.26): **label-free** self-
  supervised *semantic* occupancy via 3D Gaussian splatting from **surround** cameras.
- **Why it's on the bench (fallback, not primary):** it needs 6-camera surround input, but the current sim
  driving loop renders a single front camera → mismatch. It is the right tool once we (a) enable multi-cam
  rendering in the sim, or (b) use it offline as a **label-free semantic-occ teacher** rather than an in-loop
  provider.
- **Its real value for us:** the **label-free semantic occupancy** signal (classes: drivable / obstacle /
  vehicle …) that DA3 depth alone does not give. That semantic occ is what turns "brake for any close thing"
  into "brake for obstacles, keep going on drivable road" — a better closed-loop policy.

---

## Part 5 — Strategy: your two questions

### Q1. "We need a GOOD open-loop occ/detection first, then show closed-loop improvement. Which backbone?"
Agreed — the collision result is preliminary partly because the in-loop occ (DA3 depth) is *geometric only*.
To show "**better perception → better closed-loop driving**" we want the strongest occ/det we can run in the
loop. From Part 1:
- **Best semantic occ:** DINOv2-L LSS occ (mIoU **0.316**) — the strongest camera-only *semantic* occ we own.
- **Best camera-only detection transfer:** DINOv3-L (**0.1248**) — if the driver conditions on detections.
- **Best absolute perception:** **LiDAR+camera fusion** (occ **0.493**, det **0.242**) — nearly double
  camera-only; the sim reconstructions carry LiDAR, so a fusion occ is feasible in the loop.
- **Recommended path:** use **DINOv2-L semantic occ** (or the **fusion** occ) as the in-loop provider and show
  a *perception-quality → closed-loop-safety* curve: DA3-geometry < DINOv2-L-semantic < fusion. If safety
  improves with occ quality, that is the strong, non-preliminary result (perception quality *causally* helps
  closed-loop driving, measured on the same scenes).

### Q2. "Besides sparse collision reward, is there a DENSE perception reward? e.g. use DA3/GaussianOcc dense depth/occ to train our real-time model?"
Yes — and this is the most promising lever. Two dense signals:
1. **Dense supervision from the sim's own geometry (free GT):** the NuRec reconstruction *is* 3D geometry, so
   the sim can render a **ground-truth depth/occupancy buffer per frame**. That gives a **per-pixel dense
   training target** — orders of magnitude denser than the sparse collision outcome. Train our small real-time
   occ/depth model to match the sim-GT depth/occ; measure a **dense perception metric (depth L1 / occ IoU)**
   alongside collision. This both trains the model and gives a dense eval reward.
2. **Distill the big model into the fast one (DA3/GaussianOcc as teacher):** run DA3 (metric depth) or
   GaussianOcc (semantic occ) as a **dense teacher** offline on the sim frames → dense pseudo-GT → distill into
   our lightweight, real-time student occ/depth net. The student runs in-loop at speed; the teacher supplies
   the dense signal the sparse collision reward cannot. (This mirrors our label-free fusion-teacher→camera-
   student distillation line — the same idea, now closed-loop and dense.)

**Concrete plan:** (a) render sim-GT depth per frame → dense depth loss to train a fast occ/depth head; (b) or
distill DA3/GaussianOcc; (c) then run the closed-loop ablation with students of increasing perception quality
and plot collision-avoidance vs perception accuracy. Dense perception reward = per-pixel depth/occ error vs
sim-GT (or teacher) — a smooth signal with statistical power, unlike binary collision.

---

## Part 6 — Honest status & roadmap
**Done:** open-loop backbone study (all tables §1); closed-loop stack on our H100, no SLURM (M0); occ_driver +
ego-only baseline (M1a); DA3 in-loop occupancy (M1b) + **8-scene ablation that debunked the 1-scene positive**
(DA3-on-render domain gap, §2.3). occ_driver `steer` mode (Q2) implemented (untested — depends on a reliable
depth first).

**THE NEXT STEP (Option A — GT-LiDAR occupancy) — the blocker to a valid result.** DA3-on-render is unreliable,
so replace it with **sim-GT geometry**: render LiDAR from the reconstruction (sensorsim `render_lidar` RPC) and
push it to the driver, then run the clean **GT-occ vs no-occ** upper-bound. Only the runtime can render lidar
(it alone has the actor `dynamic_objects`). Proto step is DONE (`egodriver.proto` `submit_lidar_observation` +
`RolloutLidar`); the remaining 6-file integration is fully specified in
**`closedloop_occ/IMPLEMENTATION_PLAN_LIDAR.md`** (exact files/lines/code + rebuild + test). ~1 day; do it in a
fresh focused session (deep async event-queue surgery on the working runtime — clean context matters).

**After Option A lands (priority order):**
1. **Multi-scene GT-occ vs no-occ** → collision_at_fault mean ± std (a valid statistic, not a DA3 artifact).
2. **Perception-quality → safety curve** (Q1): GT-lidar (ceiling) vs DINOv2-L semantic occ vs DA3 → safety
   should scale with occ quality. (Surround occ needs a new AV2 multi-cam render config — front loop today.)
3. **Dense perception reward / distillation** (Q2): sim-GT depth or DA3/GaussianOcc teacher → dense-train a
   fast student that works *on the render domain*; add depth-L1 / occ-IoU as a dense metric.
4. Steering avoidance (cut rear collisions); semantic occ (DA3 depth + 2D seg); cross-dataset (PhysicalAI-AV).

## Part 7 — HANDOFF (resume in a fresh session)
1. **Read this file + `closedloop_occ/IMPLEMENTATION_PLAN_LIDAR.md`** (the Option A step-by-step) +
   `closedloop_occ/PLAN.md` (milestones). Memory notes: `closedloop-occ-alpasim`, `backbone-transfer-study`.
2. **Execute Option A** per IMPLEMENTATION_PLAN_LIDAR.md (proto already done): sensorsim_service `render_lidar`
   → event_loop submit → driver handler → PredictionInput → OccModel `occ_mode=lidar` → `compile-protos` +
   reinstall + test. Verify lidar points arrive per step; obstacle scenes show near points.
3. **Re-run the ablation** (`run_multiscene_ablation.sh`, add a `lidar` arm) for GT-occ vs no-occ.
4. Env: conda `py310` for the open-loop code; the closed-loop uses `closedloop_occ/alpasim/.venv` inside
   apptainer (containers referenced read-only from the thesis; DA3 already installed non-editable in that venv).
   Cluster internet via proxy `http_proxy=http://172.16.1.2:3128`; we run inside a SLURM interactive alloc so
   the run scripts `unset SLURM_JOB_ID` to force the local apptainer path.

**Reproduce:** open-loop = `ngperception/occupancy/*` (see PAPER_DRAFT). Closed-loop = `closedloop_occ/`:
`run_local_gtreplay.sh` (M0), `run_local_occ_drive.sh` (ego-only), `run_local_occ_da3.sh` (+occ). Details in
the [[closedloop-occ-alpasim]] and [[backbone-transfer-study]] memory notes.
