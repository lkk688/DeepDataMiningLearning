# Label-efficient / label-free AD perception — what we have, and the publication plan

A consolidation of the occupancy + detection experiments in this repo, and a concrete plan for a
publishable paper. Companion to [OCC3D_SOTA_AND_VGGT_DIRECTION.md](OCC3D_SOTA_AND_VGGT_DIRECTION.md),
[RESULTS_FULLDATA_H100.md](RESULTS_FULLDATA_H100.md), and the occupancy/detection tutorials.

---

## 1. Everything we measured (one table)

**Occupancy, Occ3D-nuScenes val, mIoU** (our runs are the 2000-frame / 15-epoch ablation harness
unless noted; ⚠️ = trained-on-val leakage in the full-data numbers):

| setting | supervision | mIoU | note |
|---|---|---|---|
| camera SOTA (COTR/SHTOcc) | full labels | ~44.5 | temporal, published |
| our LSS full-data ⚠️ | full labels | 0.302 | leaked; ~CTF-Occ tier |
| our LSS 2k-frame baseline (DINOv2, LiDAR-dep sup) | full labels | **0.293** | the A/B reference |
| **GaussianOcc (reproduced)** | **NONE (self-sup)** | **11.26** | no labels, no GT poses |

**Geometry-only, training-free (class-agnostic geo-IoU):**

| method | geo-IoU | |
|---|---|---|
| Depth-Anything mono lift | 0.093 | prior camera geometry |
| **frozen VGGT depth-lift** | **0.140** | +51%, = 84% of LiDAR |
| LiDAR single-sweep oracle | 0.167 | |

**VGGT integration ablations (trained, 2k/15ep, vs the 0.293 / 0.262 baselines):**

| # | what | regime | baseline | VGGT | verdict |
|---|---|---|---|---|---|
| #2 | depth **prior** | with LiDAR sup | 0.293 | 0.287 | no gain |
| #3 | depth **prior** | no LiDAR sup (occ-dep) | 0.262 | 0.263 | tied (noise) |
| #4 | **features** backbone | with LiDAR sup | 0.293 | 0.196 | **−33% (worse)** |

**Detection (from earlier work, [RESULTS_FULLDATA_H100.md]):** occupancy-pretrained backbone →
detection transfer gives **2.5–3.15× carAP over from-scratch**; PETR/BEVFusion multi-expert pipeline
built and evaluated on official nuScenes metrics.

## 2. The two findings

**Finding A — a geometry foundation model's label-free geometry does NOT transfer to trained
occupancy.** VGGT (CVPR'25) gives genuinely strong *label-free* geometry — a frozen, untrained
depth-lift reaches **84% of a LiDAR sweep** and +51% over the prior camera lift. Yet plugged into a
*trained* occupancy net it helps in **none of 4 integration ablations**: the **depth prior** is a
wash with *and* without LiDAR supervision (#2/#3 — redundant once a learned depth head + depth
supervision are present), and its **features backbone is 33% *worse* than DINOv2** (#4 — VGGT's
tokens are geometry-specialized, the wrong bias for a *semantic* occupancy head). Mechanism, not
just a null: **a geometry FM helps only where geometry is the bottleneck (the label-free /
no-supervision regime, i.e. GaussianOcc-style self-sup), not the supervised regime where semantics
and supervision dominate.** "Strong zero-shot geometry ≠ useful trained prior."

**Finding B — the label-free occupancy ceiling is ~11 mIoU.** We reproduced GaussianOcc
(ICCV'25, fully self-supervised, no labels/poses) *exactly* (11.26) and visualized it. That fixes a
concrete anchor: label-free 11 vs supervised ~30 (ours) vs SOTA ~44.5. The ~19-point gap is
structured — flat/large classes decent, rare/thin classes ≈0.

## 3. Publication options

**Option A — Analysis / negative-results paper.** *"When does geometry-foundation transfer help
3D occupancy? A systematic study."* Contributions: (1) Finding A with the redundancy mechanism +
the training-free-vs-trained gap; (2) Finding B as the label-free anchor; (3) the reusable harness.
Fit: CVPR/ICCV workshops, WACV, or the analysis/negative-results tracks. Honest, self-contained,
already ~80% done.

**Option B — Positive paper with these as ablations.** *Label-efficient occupancy pretraining that
transfers to detection* (the proven 2.5–3.15× core) — with Findings A/B as the "what doesn't work"
ablations that motivate the design. Stronger venue potential, but needs the clean train-only
occ→det data-efficiency curve (not yet run) as the headline positive.

**Recommendation:** aim for **B**, but **A is the fallback and is nearly publishable now**. The
VGGT/GaussianOcc negative+analysis results are the differentiator either way — the VGGT-occ space is
crowded (VG3T/DVGT/DriveVGGT), so *"we show the obvious VGGT-occ integration doesn't actually work,
and why"* is a contribution those papers don't make.

## 4. What's needed to get to submission

- [x] Ablation #4 (VGGT features backbone) — done, −33%; Finding A closed across 4 ablations.
- [ ] **Clean train-only retrain** (leakage-free occ numbers) — required for any headline occ claim.
- [ ] **Data-efficiency curve** for occ→detection transfer (10/25/50/100% labels) — the Option-B
      positive headline.
- [ ] One more VGGT lever if we want Finding A airtight: per-frame metric-scale head (removes the
      scale-drift confound before declaring the prior dead).
- [ ] Scale the GaussianOcc analysis: per-class label-free-vs-supervised gap → which classes are
      "learnable without labels" (flat/large) vs not (rare/thin) — a clean sub-result.
