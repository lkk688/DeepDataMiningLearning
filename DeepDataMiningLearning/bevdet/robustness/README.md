# Fusion-robustness probe

**The question.** nuScenes val ranks LiDAR-camera fusion detectors by NDS on clean
data. Does that ranking survive when the sensors degrade? If it does, the "which
fusion representation" question is already settled by the leaderboard and we have
nothing to add. If it inverts, the leaderboard is measuring the wrong thing, and
the inversion *is* the result.

This is a **go/no-go probe**, not a paper. It exists to kill or confirm the
direction cheaply before anything is built on it.

> ⚠ **Read this first (2026-09-08).** A literature check done *after* these experiments found that
> three of the four things this probe was built to show are already published — see
> `closedloop_occ/RESEARCH_DIRECTIONS.md` (private, gitignored) §4.1. In short: `MoME` (arXiv 2503.19776, 2025)
> Table 1 already shows the clean ranking failing to predict the degraded ranking (UniTR 73.3 clean →
> 1.5 NDS under LiDAR drop), already evaluates spatially-localized LiDAR loss (*Limited FOV*
> [-60°, 60°]), and already shows separate-modality architectures fixing much of it
> (MetaBEV 47.0 → CMT 54.0 → MoME 58.3 NDS there). MoME also beats MetaBEV on **clean** (73.6 vs 71.5).
>
> What is **not** in the literature, and is what this harness is actually for: the **two-ablation
> audit** (input corruption vs feature ablation), the **"constant prior"** failure mode it exposes,
> and the **matched-floor decomposition** (what each modality is *worth* under each condition). Use
> the probe for those. Do not use it to argue novelty for the ranking-inversion observation.

## Why this and not "beat GaussianFusion"

GaussianFusion (ICLR 2026, arXiv 2607.00746) reports 74.0 NDS vs BEVFusion's 71.4
on nuScenes val, using 8xA800 for 20 epochs with CBGS. Two reasons we do not chase
that number:

1. **We have never matched the BEVFusion baseline in our own setup** — our best
   variant is 0.6881 NDS against the official 0.7117, and our 21%-subset /
   3-epoch ablation protocol has a noise floor of about +-0.003 NDS (see
   `../paper/EXPERIMENTS.md` §5.6). A 0.9-NDS effect is barely resolvable and a
   0.3-NDS one is not resolvable at all.
2. **Their own ablation undercuts the headline.** Table 8: randomly initialised
   Gaussians score 71.2 NDS — *below* BEVFusion. The +2.6 comes from the
   forward-projection (LSS depth-distribution) initialisation, not from the
   continuous representation. Against EA-LSS (73.1 NDS, same backbone and
   resolution, no Gaussians at all, just better depth supervision) the honest
   delta is +0.9 NDS.

What GaussianFusion never measures is the thing it puts in its motivation:
the claim that Gaussian covariance "enables adaptive modeling of uncertainty".
There is no degradation, cross-domain or closed-loop experiment in the paper.
That is the flank this probe opens.

## What it measures

Every cell is one `(checkpoint, sensor condition)` pair scored on the **full
official nuScenes val split (6019 samples) with the official metric**, so the
`clean` row must reproduce published anchors — that is the correctness check on
the whole harness.

Corruptions are applied at the **sensor level**, immediately after the loading
transforms and before `ImageAug3D` / range filtering / voxelisation, so a
condition means the same physical thing for every model regardless of what its
own pipeline does downstream. Randomness is seeded from
`(seed, condition, sample_token)`, so a condition produces **byte-identical**
corrupted input for every model — without that, the ranking comparison would be
confounded by corruption noise.

Conditions in v1: camera loss (all / front), blur, low light, fog, sensor noise;
LiDAR beam reduction (32→16, 32→8), 50 % return dropout, 180° field-of-view loss;
and a 1°/5 cm camera-extrinsic miscalibration.

## Usage

```bash
cd /data/rnd-liu/MyRepo/mmdetection3d

# what would run
python projects/bevdet/robustness/probe.py \
    --config projects/bevdet/robustness/configs/gonogo_v1.yaml --dry-run

# run the sweep (each cell is its own subprocess; safe to interrupt)
python projects/bevdet/robustness/probe.py \
    --config projects/bevdet/robustness/configs/gonogo_v1.yaml

# continue an interrupted sweep -- finished cells are skipped
python projects/bevdet/robustness/probe.py \
    --config projects/bevdet/robustness/configs/gonogo_v1.yaml \
    --resume <output_root>/2026-09-04_..._gonogo_v1

# one cell only
python projects/bevdet/robustness/probe.py --config ... --only ours_calss__cam_fog
```

Each run writes `outputs/<timestamp>_<name>/` containing `meta.json` (git commit,
seed, host, GPU, date), the exact `config.yaml` used, one directory per cell with
its own log and `result.json`, plus `summary.csv` and `summary_NDS.md` rebuilt
after every cell.

## Measured anchors (2026-09-05, run `outputs/2026-09-05_18-08-29_gonogo_v1`)

The `clean` row is the harness's correctness proof. Both checkpoints with a prior
reference reproduce it:

| model | NDS | mAP | prior reference | Δ NDS |
|---|---|---|---|---|
| `bevfusion_lc_official` | 0.7118 | 0.6844 | 0.71166 / 0.68373 (2025-11-13) | 1e-4 |
| `bevfusion_l_official` | 0.6922 | 0.6434 | mmdet3d model zoo 69.2 NDS | — |
| `ours_calss` | 0.6881 | 0.6397 | 0.6881 / 0.6397 (`work_dirs/.../summary.tsv`) | <1e-4 |

`clean` runs the corruption transform as a no-op (`if not self.ops: return results`),
so this also proves the pipeline surgery introduces no bias. ~40 min per cell.

**Two facts from the anchor row that set up the whole probe:**

1. **The camera branch is worth only +0.0196 NDS on clean data** for official BEVFusion
   (0.6922 LiDAR-only → 0.7118 fused). Whatever fusion buys, it is small in-domain —
   which is exactly why an in-domain leaderboard is a poor instrument for comparing
   fusion designs.
2. **`ours_calss` (0.6881) does not beat the LiDAR-only baseline (0.6922).** Our
   cross-attention lifting is a net negative on clean data. State this plainly in any
   write-up; it also makes the LiDAR-only row a hard floor:

> **Floor rule — camera-side conditions only.** Under a *camera* degradation the LiDAR input is
> untouched, so the clean LiDAR-only score (0.6922) is the right reference: a fused model scoring
> below it has a camera branch that is not merely unhelpful but *actively harmful* there — the model
> would do better ignoring its cameras.
>
> **This line does not transfer to LiDAR-side conditions.** Under `lidar_beams_*`, `lidar_dropout_*`
> or `lidar_fov_*` the LiDAR-only model degrades too, so the correct reference is
> `bevfusion_l_official` **under the matched condition**, not its clean score. Comparing a fused
> model's 16-beam score against the clean 0.6922 would credit the degradation to the camera branch.
> The sweep runs the L-only arm on every LiDAR condition precisely so this matched floor exists;
> build the LiDAR-side table only once those cells have landed.

## Results — frozen 2026-09-07

Run `outputs/2026-09-05_18-08-29_gonogo_v1/`. 29 cells; the 30th
(`bevfusion_l_official × calib_rot1deg`) is not a valid cell — `calib_rot1deg` perturbs *camera*
extrinsics and a LiDAR-only pipeline has no `lidar2cam`, so it is now in that model's
`skip_conditions`.

### Camera-side (floor = clean LiDAR-only, 0.6922)

| condition | LC official | vs floor | ours_calss | \|Δ vs its clean\| |
|---|---|---|---|---|
| clean | 0.7118 | +0.0196 | 0.6881 | — |
| calib_rot1deg (1°, 5 cm) | 0.7096 | +0.0174 | 0.6881 | 1.5e-06 |
| cam_blur (σ = 3 px) | 0.7044 | +0.0122 | 0.6882 | 1.9e-05 |
| cam_noise (σ = 25) | 0.7014 | +0.0092 | 0.6881 | 1.7e-05 |
| cam_fog (t = 0.6) | 0.6974 | +0.0052 | 0.6881 | 1.5e-05 |
| cam_drop_front | 0.6972 | +0.0051 | 0.6881 | 3.6e-05 |
| **cam_dark** (gain .25, γ 1.5) | **0.6855** | **−0.0067** | 0.6881 | 7.2e-06 |
| **cam_drop_all** | **0.6695** | **−0.0226** | 0.6881 | 1.8e-05 |

### LiDAR-side (matched floor = LiDAR-only under the same condition)

Ordered by how much the condition damages the LiDAR-only model:

| condition | L-only | LC official | camera value | × clean | L-only damage |
|---|---|---|---|---|---|
| clean | 0.6922 | 0.7118 | +0.0196 | 1.00× | — |
| lidar_dropout_p50 | 0.6656 | 0.6938 | +0.0282 | 1.44× | −0.027 |
| lidar_beams_16 | 0.5683 | 0.6182 | +0.0499 | 2.55× | −0.124 |
| lidar_dropout_p85 | 0.5382 | 0.6054 | +0.0672 | 3.43× | −0.154 |
| **lidar_fov_front180** | 0.5153 | 0.5270 | **+0.0117** | **0.60×** | −0.177 |
| lidar_beams_8 | 0.3852 | 0.4482 | +0.0630 | 3.22× | −0.307 |
| lidar_dropout_p95 | 0.3785 | 0.4683 | +0.0898 | 4.58× | −0.314 |

The p85 / p95 rows were added 2026-09-08 as **severity-matched controls**: p=0.95 damages the
LiDAR-only model by −0.314 against 8-beam's −0.307, and p=0.85 by −0.154 against 16-beam's −0.124.
They exist to separate "how badly the LiDAR is hurt" from "in what way", and they overturned the
first reading of this table — see below.

### Findings

**F1 — in-domain NDS cannot distinguish working fusion from decorative fusion.** The camera branch is
worth **+0.0196 NDS** clean. A model whose camera branch does not respond to its own input at all sits
0.024 below the working one — the same order. One `cam_drop_all` separates them by **1179×**.

**F2 — `ours_calss` emits a weak input-independent prior; it is not a perception branch.**
Both ablations, run on the same checkpoint:

| ablation | Δ NDS | what it tests |
|---|---|---|
| input corruption (7 camera conditions, worst case) | **1.8e-05** | does the branch respond to its input? |
| feature ablation (camera BEV zeroed before fusion) | **9.3e-04** | does the head use the branch at all? |
| *positive control: LiDAR BEV zeroed instead* | *0.6881 → **0.0*** | *the ablation demonstrably works* |

So the branch **is** read by the head — removing it costs 9.3e-04 NDS — but **none of that is image
dependent**: the worst of seven mechanistically different input perturbations moves the output 50×
less than removing the feature does. `ours_calss` sits in **row 2, constant prior**, the same category
as `moddrop` but with a prior 12× weaker (9.3e-04 vs 1.12e-02).

The LiDAR-zeroed control returning **exactly 0.0** matters twice over: it proves the hook fires (a
null from an unverified intervention is worthless, §7.8), and it independently confirms this model is
entirely LiDAR-driven, consistent with `../paper/MODALITY_ROBUST_FINDINGS.md`'s camera-only collapse.

Consequence for `../paper/EXPERIMENTS.md`: the B1–B6 ablations (CA-LSS, voxel painting, multi-scale,
GQA, Q-Occ) were tuning a branch whose entire contribution is a 9.3e-04 constant. Their "all within
±0.003 noise" conclusion has a stronger explanation than noise.

**F3 — the ranking inverts.** Clean: LC +0.024 over ours. Camera blackout: ours +0.019 over LC. Ours
does not win by being robust; it wins by having nothing to lose.

**F4 — the camera refines LiDAR evidence, it does not substitute for it.** Order the LiDAR conditions
by damage and the camera's value does not follow: `lidar_fov_front180` is the *second most damaging*
condition but has the *lowest* camera value (0.60× of clean). What separates them is the kind of loss:

| LiDAR degradation | L-only damage | camera value | × clean |
|---|---|---|---|
| clean | — | +0.0196 | 1.00× |
| random dropout p=0.50 | −0.027 | +0.0282 | 1.44× |
| 16-beam | −0.124 | +0.0499 | 2.55× |
| random dropout p=0.85 | −0.154 | +0.0672 | 3.43× |
| **FOV cut (front 180°)** | **−0.177** | **+0.0117** | **0.60×** |
| 8-beam | −0.307 | +0.0630 | 3.22× |
| random dropout p=0.95 | −0.314 | +0.0898 | 4.58× |

**Camera value grows monotonically with LiDAR damage — with exactly one exception, the FOV cut.**
Two severity-matched pairs settle what the variable is:

```
p=0.95 (damage −0.314) vs 8-beam  (−0.307):  +0.0898  vs  +0.0630
p=0.85 (damage −0.154) vs 16-beam (−0.124):  +0.0672  vs  +0.0499
```

At matched (or worse) damage, **random thinning gives the camera MORE value than beam reduction, not
less.** So the distinction is **not** "sparse vs coarse vs random" — dropout, 16-beam and 8-beam all
behave the same way. It is **"degraded everywhere" vs "absent in a region"**: the FOV cut is the only
condition where LiDAR coverage vanishes over a contiguous volume, and it is the only one where the
camera's value falls *below* its clean level (0.60×), against a trend line that predicts ~+0.07 at
that damage — a 6× shortfall.

> **The camera compensates for uniformly degraded LiDAR. It does not cover where LiDAR is spatially
> absent** — even though three rear cameras see that region.

⚠ **Retracted 2026-09-08.** An earlier revision of this table claimed a three-way split — "thinner
(dropout) 1.4×, coarser (beam reduction) 2.6–3.2×, absent 0.6×" — and read a *structure* effect into
the dropout-vs-beam gap. That gap was **severity**: p=0.50 barely damages the LiDAR-only model
(−0.027) while 16-beam damages it −0.124. The severity-matched arms above reverse the claimed
ordering. Only the FOV exception survives.

### Why F4 happens — and the correction that matters

It is tempting to blame the detection head: TransFusion picks queries from a heatmap, so "no LiDAR →
no peaks → no queries". **We read the code and that is wrong.** The heatmap is computed on the
**fused** feature (`transfusion_head.py:228`) and queries are gathered from the fused map, so the
architecture *permits* camera-only proposals.

There is also **no cross-modal attention** anywhere: fusion is `ConvFuser` =
`cat([cam_bev, lidar_bev])` → 3×3 conv → BN → ReLU (`transfusion_head.py:29-42`), and the decoder
cross-attends only to that single fused map (`key=fusion_feat_flatten`).

Two consequences:

1. **The LiDAR dependence is learned, not structural.** The model trained on data where LiDAR is
   always present, so it never learned to raise a proposal from camera evidence alone. That makes it a
   *training* target, not a reason to redesign — and modality-dropout training already fixed the
   LiDAR-only direction (L-only ≈ LC, −0.011 NDS).
2. **The fuser has no mechanism for modality-adaptive weighting.** A fixed conv applies the same
   linear combination whether or not the camera channels are informative, so blacked-out cameras are
   mixed in at full weight as an out-of-distribution constant. This is the mechanism behind the two
   floor crossings, and both are **photometric** (dark, blackout) while every *geometric* perturbation
   (blur, fog, noise, dropped view, miscalibration) stays above the floor.

**Testable prediction.** GaussianFusion (ICLR 2026, arXiv 2607.00746) pools Gaussians back to a voxel
grid and uses the same TransFusion head; we predict it inherits the same behaviour. Its paper contains
no degradation, cross-domain or closed-loop experiment.

## Reading the result

The probe answers one question: **is the clean ranking predictive?**

- *Ranking holds across all conditions* → no-go. The leaderboard already tells
  you what you need; drop the direction.
- *Ranking inverts* → go. Report which condition inverts it and what property of
  the model predicts the inversion (our prior, from the cross-domain occupancy
  study, is that it is the camera branch's depth quality — not the representation
  container).

The LiDAR-only row is the control that keeps this honest: if a fusion model's
degraded score never falls below the LiDAR-only model's, then its "robustness" is
just the LiDAR still being there, not anything the fusion design contributes.

## Limitations to state with any number produced here

- Corruptions are synthetic, in the manner of nuScenes-C / RoboBEV; they are not
  measured adverse-weather data.
- Beam decimation bins by elevation over the accumulated 9-sweep cloud, whose
  earlier sweeps are ego-motion compensated — it approximates a lower-beam
  sensor rather than exactly resampling one.
- `cam_fog` is a uniform airlight blend with no depth dependence.
- All models are evaluated **as trained**; none was trained under degradation.
  The probe measures the robustness of existing checkpoints, not the ceiling of
  each architecture.
