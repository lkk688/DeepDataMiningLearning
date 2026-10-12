# Strategy: the role of Occupancy (and detection) in end-to-end driving

**2026-08-27.** Consolidates our own results into a position on (1) which occ model to use, (2) whether
E2E driving needs explicit occ/detection, (3) how E2E is trained and what data it needs, and (4) what is
publishable. Grounded in *our* experiments, not opinion.

---

## 1. Which occupancy model is best (our cross-domain study, TUTORIAL §4.3)

nuScenes-trained occ → AV2 NuRec renders, camera-only unless noted (FG_REC = foreground-object recall):

| model | in-domain | AV2 render | retention |
|---|---|---|---|
| **LSS (DINOv2)** | 0.645 | **0.382** | 59 % |
| GaussianOcc (self-sup) | 0.297 | 0.185 | **62 %** |
| FlashOcc (R50 stereo) | 0.525 | 0.105 | 20 % |
| OPUSv2-L (R50 sparse-query) | 0.618 | 0.036 | 6 % |
| FusionOcc (cam+LiDAR) | **0.812** | (needs LiDAR) | — |

- **For renders / deployment (camera-only): DINOv2 LSS** — best absolute cross-domain, robust, and
  fine-tunes to **0.706** on renders label-free.
- **In-domain with LiDAR: FusionOcc.**
- **Key finding:** the split is by **pretraining (self-supervised transfers; supervised-ResNet50
  collapses)**, NOT architecture — the fanciest model (OPUS sparse-query) collapsed *worst* cross-domain.

## 2. Does E2E need explicit occ/detection? — our evidence says: NOT as a planning input

Three of our own results converge:
- **Double-decoupling** (`backbone-transfer-study`): occ is a **good detection *pretext*** but a **NULL
  planning input**; occ mIoU ≠ det-transfer; **ego-status dominates open-loop planning** (the planner
  barely reads occ).
- **Reactivity tax** (closed-loop, `closedloop_occ`): an occ reflex that *acts* sometimes **induces**
  collisions — explicit occ→action can *hurt*.
- **Label-efficiency positive**: occ pretraining → detection is **label-efficient (mAP 1.4–2.1×,
  rare-object-driven)** — occ as *representation* is valuable.

**Position:** don't wire explicit occ/detection as a hard intermediate the planner reads (null/negative in
our data). DO use them as (a) **pretext / representation learning** and (b) an **interpretable, auditable
safety decode** (occ as a *view* of a latent world-state, plus a general-obstacle reflex). Action should
come from a **learned latent world-state**, not occ→action. → this is the `closedloop_occ/worldmodel_occ`
direction (a recurrent latent that *decodes* to occ for audit/safety, not a forced planning layer).

## 3. How E2E is actually trained, and why trajectory-BC under-supervises perception

Most E2E driving = **imitation / behavior cloning (BC)** on human-driving logs (action = f(sensors) from
expert trajectories; UniAD/VAD, Tesla fleet, etc.), sometimes + auxiliary perception heads + closed-loop
RL/sim fine-tuning.

**The problem you flagged is real and well-documented:** pure trajectory-BC *under-supervises perception*.
The waypoint gradient doesn't force the model to robustly perceive every object — the model takes
**shortcuts** (follow the road, copy typical behavior, lean on ego-status/inertia — exactly our
"ego-status dominates" result = causal confusion). Rare/safety-critical objects that rarely alter the
trajectory get missed.

**What data/supervision is needed on top of trajectory-BC:**
1. **Explicit perception labels (det/occ/track) as auxiliary supervision** — forces objects into the
   representation even when the trajectory doesn't. (This is UniAD's argument and matches our
   occ-as-pretext result.)
2. **Long-tail / rare-object data** — targeted mining (our label-free long-tail mining) or simulated rare
   events; the expert trajectory won't cover them.
3. **Closed-loop / counterfactual data** — BC on expert logs never sees the *consequences of mistakes* or
   recovery from off-distribution states; need DAgger/RL/sim (our AlpaSim closed-loop, the reactivity-tax
   protocol).
4. **General/unknown-obstacle supervision** — the long tail includes *unclassifiable* objects; class-based
   detection misses them, which is exactly what a general occupancy captures.

## 4. Is Tesla-style Occupancy "useless" now? — No; its role shifted

Tesla's Occupancy Network (AI-Day 2022) predicts **dense general 3D occupancy** — its durable value is
**detecting ANY obstacle, including unlabeled/unknown ones** (the long tail), plus free-space, as a
**geometric safety layer + planner constraint**, not as the sole planning signal.
- Not useless: **general/unknown-obstacle geometry is the part that survives** — you must not hit the thing
  you can't classify. (Our closed-loop reflex used exactly this: *general actor-box occupancy*, not dense
  semantics.)
- But: Tesla itself moved toward more end-to-end (v12 FSD, "photons→controls"), reducing explicit
  intermediate modules — while (plausibly) keeping occupancy as a **safety/interpretability layer**.
- So the durable role of occ = **(1) general/unknown-obstacle detection, (2) pretext, (3) interpretable
  safety decode** — NOT the dense-semantic road/building classes, and NOT the explicit planning input.
  This is why our line's takeaway is **"robust + general-obstacle occ" beats "pixel-perfect semantic occ"**
  (and why the camera-only pixel-clean-occ chase hit diminishing returns — see PLAN_LABELFREE_SIM_OCC §4).

## 5. Is "our occ is best cross-dataset" publishable?

Enhancing it *can* be — but **not as "we win a benchmark."** It needs a **claim**, **scale**, and ideally
the **closed-loop tie-in** we uniquely have:
- **Publishable claim (not crowded):** *domain robustness of occ is determined by pretraining, not
  architecture* — self-supervised/foundation backbones transfer to the render/sim domain while supervised
  ones collapse; + a **label-free render-domain adaptation** that nearly doubles obstacle recall. Most occ
  papers are in-domain mIoU; a **cross-domain robustness + label-free adaptation** story is differentiated.
- **Needed to be credible:** more scenes + a *second* target domain (e.g. nuScenes→AV2→Waymo), a
  mechanistic analysis (why supervised-R50 collapses: intrinsics/appearance overfit), and standard mIoU
  alongside FG_REC.
- **The strongest, most novel framing uses our closed-loop asset:** *does domain-robust occ improve
  closed-loop driving?* — BUT honesty requires threading our own double-decoupling/reactivity-tax finding:
  the value of robust occ is **perception robustness + general-obstacle safety + pretext**, NOT "better occ
  → better planner." A paper claiming occ→driving would contradict our own null result; frame it as a
  **perception/safety** contribution.
- **Cost/benefit (pushback):** cross-domain occ robustness is moderately (not fully) crowded. Without the
  closed-loop/safety framing it is an incremental analysis paper. The differentiator is the **closed-loop
  + label-free + general-obstacle** angle — that is where to invest, not another mIoU point.

## 6. Bottom line

- Use **DINOv2 LSS** (robust, fine-tunes to 0.706); don't chase pixel-clean camera-only occ (monocular
  geometry wall, PLAN_LABELFREE_SIM_OCC §4).
- Treat occ/detection as **pretext + interpretable general-obstacle safety layer**, not an explicit
  planning input (our data: null/negative there).
- E2E ≠ trajectory-BC alone: it needs **auxiliary perception + long-tail mining + closed-loop/counterfactual
  data** — all assets we have.
- Publish the **pretraining-determines-robustness + label-free adaptation + closed-loop-safety** story,
  framed as a *perception/safety* contribution, not occ→planning.
