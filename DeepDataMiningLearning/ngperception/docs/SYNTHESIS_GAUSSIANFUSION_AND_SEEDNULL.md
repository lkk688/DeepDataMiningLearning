# GaussianFusion (2607.00746) read against our three projects

**2026-09-04.** Cross-project note. Connects the ICLR'26 submission *GaussianFusion:
Unified 3D Gaussian Representation for Multi-Modal Fusion Perception* to (a) the
frozen-FM backbone study (`RESEARCH_DIRECTIONS.md`), (b) the closed-loop / occ-in-E2E
position (`STRATEGY_OCC_IN_E2E.md`), and (c) the ICRA'27 certification work
(`physicalai_autolabel/paper_icra27/`).

---

## 1. What the paper actually reports

Zhao, Liu, Zhu, Zhang, Song, Luo, Guo, Su. Replaces the BEV grid with a shared,
continuous **3D Gaussian space** as the fusion substrate: forward-projection
Gaussian init from both modalities, then a **shared cross-modal encoder** that
iteratively updates Gaussian properties by attention. Task-agnostic head interface.

| nuScenes detection | modality | NDS | mAP |
|---|---|---|---|
| BEVFusion (M) | C+L | 71.4 | 68.5 |
| **GaussianFusion** | C+L | **74.0** | **71.7** |

**+2.6 NDS / +3.2 mAP.**

| Occ3D-nuScenes, camera-only | mIoU | #Gaussians | latency |
|---|---|---|---|
| GaussianFormer | 19.10 | 140,000 | 475 ms |
| **GaussianFusion-C** | **20.65** | **43,296** | **105 ms** |

Swin-T image backbone, VoxelNet LiDAR, 20 epochs, AdamW, one-cycle 2e-4.
Ablations: Gaussian init strategy (random / backward / LiDAR / forward projection);
encoder design (shared vs separate, deformable attention, PE, offset prediction).

**No error bars, no variance, no multiple seeds anywhere in the paper.** Every
number is one training run.

---

## 2. Three readings only this lab can make

### 2.1 The occupancy claim is intra-family, and we have the number that shows it

GaussianFusion-C's 20.65 mIoU beats GaussianFormer's 19.10 — but our own
reproduced **FlashOcc-4D-stereo sits at 38.09 mIoU** (`RESEARCH_DIRECTIONS.md` §4.5,
vs published 37.84), and our **fusion column reaches 0.493** (§4.6).

So the occupancy result is a **within-sparse-Gaussian-family** improvement at
roughly **half** the dense-occ state of the art we have already reproduced on this
machine. The real contribution in that table is the **4.5× latency and 30% of the
Gaussians**, not the mIoU.

⚠ *Verify before using this in writing:* GaussianFormer's 19.10 may be evaluated
under a different Occ3D protocol (camera-visible mask, class set) than FlashOcc's
37.84. The gap is too large to attribute to method alone without checking. This is
a one-afternoon check and it is worth doing — if the protocols match, we have a
publishable observation about how the Gaussian-occ literature reports itself.

### 2.2 It refines, rather than contradicts, our "Gaussians earn no advantage"

`RESEARCH_DIRECTIONS.md` §4.4 found voxel-soft beats aniso-Gaussian once an
occupancy/semantic-coupling confound is removed, and §5 concludes *"Gaussians
belong in prediction/reconstruction, not labels."*

GaussianFusion uses Gaussians as **neither** — it uses them as the *internal
fusion substrate*, with task heads on top. That is a third role, and it wins. Our
taxonomy should be sharpened to three, which makes our negative result stronger,
not weaker:

| role of the Gaussian | evidence | verdict |
|---|---|---|
| as occupancy **label / teacher** | our §4.4 2×2 | ✗ no advantage over voxels |
| as **reconstruction / prediction** | VGOcc, ADGaussian, IDESplat | ✓ established |
| as **fusion substrate** | **GaussianFusion** | ✓ +2.6 NDS over BEVFusion |

Cite it as support for the distinction. Our claim was never "Gaussians don't
work"; it was "Gaussians don't help *as labels*", and this is the paper that makes
that precision necessary.

### 2.3 It is a candidate teacher for the ICRA loop — and our own result predicts it will not matter

The ICRA'27 auto-label loop is BEVFusion-derived (B14 / FlowGuidedTemporalBEVFusion).
A +2.6 NDS teacher is the obvious upgrade. But **R15 measured the seed-only range
of one pseudo-label round at 7.1 recall points [6.2, 8.0]**
(`RESULTS_ICRA27.md` §8.26). A +2.6 NDS teacher improvement entering a loop whose
round-to-round noise is 7.1 points is a **falsifiable prediction: it will be
invisible.** That is a genuinely interesting experiment either way, and it costs
one teacher swap plus the reseeds we now have tooling for.

---

## 3. The convergence: all three projects share one gap, and we just measured it

| project | headline effect | runs per arm | seed null run? |
|---|---|---|---|
| GaussianFusion (theirs) | +2.6 NDS | 1 | ✗ |
| Backbone study (`RESEARCH_DIRECTIONS.md` §4.1) | DINOv2-L 0.316 vs DINOv3-L 0.292 vs DINOv2-B 0.288 | 1 | ✗ |
| ICRA'27 round curve | −2.0 recall | 1 | ✅ **and the effect lost** |

The ICRA project ran the null and the headline did not survive it: seed-only range
**7.1** against a **−1.4 [−2.4, −0.5]** decline on the same clips — **4.9× the
effect** (§8.26). Two single runs of the same recipe differ with SD ≈ 3.85, so any
single-run round-to-round difference carries roughly **±7.5**.

**The same ruler has not been laid against the other two projects, and both have
differences far smaller than the one that failed.** In `RESEARCH_DIRECTIONS.md` §4.1, the
DINOv3-L vs DINOv2-B occ gap is **0.004** and the whole occ↓/det↑ decoupling claim
rests on **0.024 occ / 0.011 mAP** between two single runs.

**Honest calibration, which matters:** our 7.1 comes from a *1,743-frame
pseudo-label* recipe. Seed variance falls with training-set size, so full-nuScenes
20-epoch training will be far tighter — plausibly SD 0.2–0.4 NDS, which would make
GaussianFusion's +2.6 a comfortable 6–13σ and entirely safe. **Do not claim +2.6 is
noise.** The target is not that headline; it is the *fine structure* — the
0.3–0.5 NDS increments that separate adjacent entries on the fusion leaderboard,
and the 0.004–0.024 gaps inside our own backbone table.

---

## 4. What to do, ranked

### ✅ P1 — DONE 2026-09-07. See `FINDING_BACKBONE_TABLE_ESTIMATOR.md` + both addenda.
**Outcome: the occ↓/det↑ decoupling is CONFIRMED** at 4 runs per arm (det mAP
p=0.018, det NDS p=0.014), the occ ranking is a trend (p=0.064), the "≈ DINOv2-B"
claim is dropped, and the table's mixed max/final estimator must be fixed.
(A claimed bonus finding — "det transfer is more reproducible than occ mIoU" — was
**retracted the same day**: wrong comparison (absolute sd across different scales)
and untestable at n=4 anyway, F=2.0–5.5 against a critical 15.4.)

### P1 (original scoping note) — Reseed our own backbone table first *(cheap, and it is our exposure, not theirs)*
The frozen-FM probe trains only the lift/decoder/heads — the backbone is frozen, so
a reseed is a fraction of a full training run. **4 seeds × the top 3 backbones**
gives error bars on §4.1 and tells us whether the DINOv3-L decoupling claim
survives. This is the single highest value-per-GPU-hour item in either project, and
it protects the paper's key claim rather than attacking someone else's.
`scripts/train_round0.py --seed` (added 2026-09-02) is the pattern to copy.

### P2 — The seed null for fusion detection *(a paper nobody has written)*
We have **official BEVFusion reproduced at NDS 0.712 / mAP 0.683** — within 0.2 of
the 71.4 / 68.5 this paper uses as its baseline. That is the rarest asset here:
an exactly-reproduced leaderboard baseline plus the H100 to reseed it.

Reseed BEVFusion ×5 on nuScenes → the run-to-run SD of NDS and mAP for 3D fusion
detection. Then: *what fraction of published increments on this leaderboard are
inside it?* Combined with the ICRA seed-variance result on a second, very different
regime (small-data pseudo-label rounds), it becomes a two-point curve of **seed
variance vs training-set size** — which is a much better paper than either point
alone, and directly extends §4.2's data-scaling finding.

### P3 — GaussianFusion as an ICRA teacher, as a pre-registered prediction
Swap the teacher, run the loop, and pre-register the prediction from §2.3: a +2.6
NDS teacher will not move the loop, because the loop's own seed noise is 7.1. Write
the prediction down *before* running it — that is the protocol discipline the ICRA
paper is arguing for, applied to ourselves.

### P4 — Gaussians as a native refutation handle *(speculative, and check the baseline first)*
GaussianFusion's Gaussians carry explicit position, scale, rotation and **opacity** —
an intrinsic geometric confidence a BEV grid does not expose. That is a candidate
label-free instrument for the ICRA certification framework, which currently relies
on *external* instruments (radar Doppler, future observation, V2X).

**But §8.24 is binding here:** we built an elaborate label-free statistic (ASYM),
controlled it, replicated it across pools, and then found it **dominated by the mean
detection confidence we already had**. Rule 16 now stands: *a new instrument is not
a result until it has been compared against the cheapest baseline that could replace
it.* So Gaussian opacity gets benchmarked against mean confidence **on day one**, not
after four sections.

---

## 5. One line

GaussianFusion is a real result and a good citation for our Gaussian taxonomy, but
its occupancy claim is smaller than it looks against numbers we have already
reproduced — and it shares with **our own backbone table** the gap the ICRA project
just measured and failed: **one training run per arm, and no seed null.** Fixing
that in our table is P1; measuring it for the fusion leaderboard is the paper.
