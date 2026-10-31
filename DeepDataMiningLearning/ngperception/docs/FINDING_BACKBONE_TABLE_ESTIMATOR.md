# The §4.1 backbone table needs a fixed estimator before it needs seeds

**2026-09-04.** Found while setting up the seed sweep (`run_backbone_seeds.sh`,
P1 of `SYNTHESIS_GAUSSIANFUSION_AND_SEEDNULL.md`). **No new training was required
— this is all in the July logs.** The seed sweep is still running; this is what the
existing runs already say.

---

## 1. Two different estimators are mixed in the published table

`run_backbone_bench.sh` records `... | sort -g | tail -1` — the **maximum val mIoU
over all 24 epochs**. But `occ_results.csv` does not consistently contain that:

| backbone | final epoch | max over epochs | **CSV / §4.1 reports** | which estimator |
|---|---|---|---|---|
| dinov2_large | 0.316 | 0.316 | 0.316 | (identical) |
| dinov3 | 0.292 | 0.292 | 0.292 | (identical) |
| **dinov2_base** | **0.288** | **0.308** | **0.288** | **final** |
| radio | 0.274 | 0.274 | 0.274 | (identical) |
| siglip2 | 0.234 | 0.234 | 0.234 | (identical) |
| **vggt** | **0.187** | **0.218** | **0.218** | **max** |

Each log is a single clean 24-epoch run (24 `val mIoU=` lines, one `epoch 0:`), so
this is not concatenated runs. **`dinov2_base` is reported at its final epoch and
`vggt` at its maximum** — the two rows where final and max differ are the two rows
that disagree about which to use.

Separately: `max` is selected on the **same 300 val samples** that then report the
number, so it carries a selection bias whose size scales with the run's noise. The
two rows with the largest max−final gaps (dinov2_base +0.020, vggt +0.031) are the
two noisiest runs. That is the selection effect, not a property of the backbone.

## 2. Within-run epoch noise is larger than the gaps the table's claims rest on

Over the last 5 epochs of each single run:

| backbone | final | max | **mean(last 5)** | **sd(last 5)** | range(last 5) |
|---|---|---|---|---|---|
| dinov2_large | 0.316 | 0.316 | **0.306** | 0.009 | 0.025 |
| dinov2_base | 0.288 | 0.308 | **0.283** | **0.023** | **0.062** |
| dinov3 | 0.292 | 0.292 | **0.282** | 0.007 | 0.019 |
| radio | 0.274 | 0.274 | 0.253 | 0.017 | 0.046 |
| siglip2 | 0.234 | 0.234 | 0.217 | 0.013 | 0.035 |
| vggt | 0.187 | 0.218 | 0.210 | 0.013 | 0.031 |

The training curve has **not plateaued at epoch 24** — DINOv2-L runs
…0.307, 0.308, 0.291, 0.307, **0.316** — so the final epoch is one draw from a
still-moving, still-noisy sequence.

**dinov2_base fluctuates by 0.062 across its own last five epochs.** The
DINOv3-L vs DINOv2-B gap the table asserts is **0.004**.

## 3. What this does to the two claims in §4.1

| claim | by final | by max | by mean(last 5) | verdict |
|---|---|---|---|---|
| DINOv2-L − DINOv3-L | **+0.024** | **+0.024** | **+0.024** | ✅ **estimator-robust** |
| DINOv3-L − DINOv2-B | +0.004 | **−0.016** | −0.001 | ❌ **sign flips** |

- **"DINOv2-L is the best occ backbone" survives.** The +0.024 margin over DINOv3-L
  is identical under all three estimators. Whether it survives *seeds* is what the
  running sweep will say — it is 1–3× the within-run epoch sd, so this is not yet
  settled either.
- **"DINOv3-L ≈ DINOv2-B" is not established.** The ordering reverses depending on
  the estimator, and DINOv2-B's own epoch-to-epoch sd (0.023) is **six times** the
  claimed 0.004 gap. Under a consistent estimator the two are indistinguishable
  (mean-last-5: 0.282 vs 0.283).

The occ↓/det↑ decoupling headline is **not** damaged by this — it is carried by
DINOv2-L vs DINOv3-L (+0.024 occ, −0.011 det), the comparison that is
estimator-robust. What must be dropped is the incidental "≈ DINOv2-B" ranking and
any fine-structure reading of the table's middle rows.

## 4. Fix

1. **One estimator, stated.** Recommend **mean ± sd over the last 5 epochs** — it
   uses the plateau rather than one draw from it, and it reports its own noise. Max
   over epochs should not be used at all: it is selected on the reporting split.
2. **Re-issue the §4.1 table** with mean(last 5) ± sd. VGGT drops to 0.210 and
   DINOv2-B to 0.283, which does not change the top of the ranking.
3. **Add the seed column** when `run_backbone_seeds.sh` finishes (4 seeds ×
   DINOv2-L, then DINOv3-L if needed) so the table carries **run-level** as well as
   epoch-level variance. The det arm is already single-seed by construction —
   `det_results.csv` has a `seed` column and every row reads `1`.
4. **Train longer or stop earlier on a criterion.** The curve is still rising at
   epoch 24; "24 epochs" is currently an arbitrary stopping point that interacts
   with the noise.

## 5. Why this was worth finding before the seeds finished

This is the same failure the ICRA'27 project logged four times
(`RESULTS_ICRA27.md` §8.20, §8.22, §8.23, §8.24): **a comparison built without
checking what its two arms actually differed in.** Here the arms differ in which
epoch's number got recorded.

It also cost nothing — the evidence was in logs written on 2026-07-23. Per Rule 15
(*grep the record for the comparison you are about to make*), the check should
precede the experiment, and in this case it would have changed which experiment to
run: **fixing the estimator is free and settles one of the two claims; the seed
sweep is 18 GPU-hours and settles the other.**

---

# ADDENDUM 2026-09-06 — the seed sweep finished

4 seeds × DINOv2-L and DINOv3-L, identical recipe, final-epoch estimator:

| backbone | seeds | mean | **sd** | range | published (n=1) |
|---|---|---|---|---|---|
| DINOv2-L | 0.278, 0.298, 0.299, 0.312 | **0.2968** | 0.0140 | **0.034** | 0.316 |
| DINOv3-L | 0.267, 0.280, 0.281, 0.282 | **0.2775** | 0.0070 | 0.015 | 0.292 |

**1. Both published numbers sit above all four of their own reseeds.** 0.316 > max
seed 0.312; 0.292 > max seed 0.282. Two independent runs each landing above their
own 4-sample range has probability ≈0.04 if the reseeds are exchangeable with the
originals. Either the originals were lucky draws, or something differs between the
unseeded July runs and these (the recipe is byte-identical apart from `--seed`; the
only other change is that these three ran concurrently). **Worth one diagnostic
before the table is re-issued** — a 5th unseeded run would settle it.

**2. The gap shrinks and loses significance.**

| | value |
|---|---|
| published single-run gap | **+0.024** |
| reseeded difference of means | **+0.019 [+0.001, +0.037]** |
| Welch | t=2.45, df=4.4, **p=0.064** |
| DINOv2-L seed range alone | **0.034 — larger than the gap** |

So **"DINOv2-L is the best occupancy backbone" is directionally reproduced but not
significant at 4 seeds per arm** (p=0.064). It is not refuted — the sign is
consistent and the CI barely excludes zero — but it cannot be stated as a fact from
n=1, and at n=4 it is a trend.

**3. Combined with the estimator finding above, §4.1's status is:**

| claim | verdict |
|---|---|
| DINOv2-L > DINOv3-L on occ | **trend, p=0.064 at 4 seeds.** Report as mean ± sd over seeds, not as a ranking |
| DINOv3-L ≈ DINOv2-B on occ | **not established** — sign flips with estimator; DINOv2-B's within-run epoch sd (0.023) is 6× the 0.004 gap |
| occ↓ / det↑ decoupling (DINOv2-L vs DINOv3-L) | **the occ half is a 0.019 ± 0.008 trend.** The det half (0.1140 vs 0.1248) is still n=1 per arm and is now the weakest link |

**4. What to run next, in order.**
- **det seeds for DINOv2-L and DINOv3-L** (4 each). The decoupling claim needs both
  halves; only the occ half now has error bars, and the det gap (+0.011) is smaller
  in absolute terms than the occ gap that came out at p=0.064.
- one unseeded rerun of DINOv2-L, to check finding 1.
- DINOv2-B seeds only if the "≈" claim is to be kept at all — recommend dropping it.

**5. The honest headline for §4.1.** Not "DINOv2-L wins" but: *under a fixed
finetune budget, occupancy mIoU differences between frozen foundation backbones are
0.019–0.024, and seed-only variance on the same recipe is 0.015–0.034. The backbone
ranking in this regime is at the edge of what a single run can resolve.* That is a
more useful statement for the field than a leaderboard, and it is the same finding
the ICRA'27 project reached independently on a completely different task
(`RESULTS_ICRA27.md` §8.26–§8.27).

---

# ADDENDUM 2 — 2026-09-07: det seeds + the unseeded control. **The decoupling survives.**

## Task 2 first — the anomaly was a lucky draw

An unseeded DINOv2-L rerun (everything else identical) gave **final 0.302**, which
sits **inside** the seeded range [0.278, 0.312]. So there is **no systematic
seeded/unseeded offset**; the reseeds are exchangeable with the originals and the
published 0.316 was simply a high draw.

DINOv2-L, all five runs: **0.278, 0.298, 0.299, 0.302, 0.312** —
mean **0.2978**, sd **0.0124**, range **0.034**.

## Task 1 — det transfer, occ-pretrain seed varied, det seed fixed at 1

| | DINOv2-L | DINOv3-L | difference | |
|---|---|---|---|---|
| **det mAP@2k** | 0.1247, 0.1170, 0.1193, 0.1156<br>mean **0.1192** ± 0.0040 | 0.1270, 0.1284, 0.1247, 0.1294<br>mean **0.1274** ± 0.0020 | **+0.0082 [+0.0020, +0.0145]** | t=3.66, df=4.5, **p=0.018** ✅ |
| **det NDS** | mean **0.1289** ± 0.0026 | mean **0.1346** ± 0.0019 | **+0.0057 [+0.0012, +0.0102]** | t=3.55, df=5.5, **p=0.014** ✅ |
| **occ mIoU** | mean **0.2968** ± 0.0140 | mean **0.2775** ± 0.0070 | **+0.0192 [−0.0025, +0.0410]** | t=2.45, df=4.4, p=0.064 ⚠ |

## The verdict: §4.1's headline claim is CONFIRMED, with error bars

**DINOv3-L transfers better to detection (p=0.014–0.018) while being no better —
and probably worse — at occupancy (p=0.064).** That is precisely the occ≠det
decoupling the paper asserts, and it now rests on 4 runs per arm instead of 1.

**This corrects a prediction made in Addendum 1.** That addendum called the det
half "the weakest link" because its published gap (+0.011) was smaller in absolute
terms than the occ gap (+0.024). **The opposite is true.** Absolute gap size was the
wrong thing to look at; what matters is the gap against *its own* noise:

| | gap | sd (pooled-ish) | gap / sd | p |
|---|---|---|---|---|
| occ mIoU | 0.0192 | ~0.011 | **1.7** | 0.064 |
| det mAP | 0.0082 | ~0.003 | **2.7** | 0.018 |
| det NDS | 0.0057 | ~0.002 | **2.5** | 0.014 |

~~**Detection transfer is a 3–5× more reproducible measurement than occupancy mIoU
under this recipe.**~~ **RETRACTED, same day — see "Correction" below.**

## Final status of §4.1

| claim | status |
|---|---|
| **occ↓ / det↑ decoupling (DINOv2-L vs DINOv3-L)** | ✅ **confirmed**, both halves, 4 runs per arm |
| ~~det transfer more reproducible than occ mIoU~~ | ❌ **retracted** — see Correction below |
| "DINOv2-L is the best occupancy backbone" | ⚠ **trend only**, p=0.064. Report mean ± sd, not a ranking |
| "DINOv3-L ≈ DINOv2-B" | ❌ **drop it.** Sign flips with estimator; DINOv2-B's within-run epoch sd (0.023) is 6× the 0.004 gap |
| table estimator | ❌ **fix**: `dinov2_base` recorded final, `vggt` recorded max. Use mean ± sd over the last 5 epochs everywhere |

## Recommended §4.1 replacement

| backbone | occ mIoU (mean ± sd over seeds) | det mAP@2k | det NDS |
|---|---|---|---|
| DINOv2-large | **0.298 ± 0.012** (n=5) | 0.1192 ± 0.0040 (n=4) | 0.1289 ± 0.0026 |
| DINOv3-large | **0.278 ± 0.007** (n=4) | **0.1274 ± 0.0020** (n=4) | **0.1346 ± 0.0019** |
| DINOv2-base | 0.283 (n=1, last-5 mean) | 0.1061 (n=1) | 0.1215 (n=1) |
| RADIO | 0.253 (n=1, last-5 mean) | 0.0964 (n=1) | 0.1095 (n=1) |
| SigLIP2 | 0.217 (n=1, last-5 mean) | 0.0628 (n=1) | 0.0872 (n=1) |
| VGGT | 0.210 (n=1, last-5 mean) | — | — |

with the caption stating which rows carry replicates and which do not.

## Why this matters beyond this table

The same null was applied to two projects in the same week. In the ICRA'27
auto-label study it **killed the headline**: a −1.4 recall "decline" became
+0.8 [−2.5, +4.2], p=0.58 (`RESULTS_ICRA27.md` §8.27). Here it **confirmed the
headline** and added a new supporting finding.

That contrast is the argument for running seed nulls as routine. They are not a
nihilism machine — they separate the claims that hold from the claims that were
draw ordering, and **you cannot tell which is which by looking at effect size
alone.** The occ gap here is 2.3× larger than the det gap and it is the one that
failed.


---

## Correction (same day) — the "det is more reproducible" finding does not hold

Written above and withdrawn within the hour. Two errors, and the second is the one
that matters.

**Error 1 — wrong comparison.** I compared **absolute** standard deviations across
metrics on different scales: occ mIoU ≈ 0.30, det mAP ≈ 0.12. Of course the sd of
the smaller-magnitude quantity is smaller. The scale-free comparison is the
coefficient of variation:

| metric / arm | mean | sd | **CV** |
|---|---|---|---|
| occ, DINOv2-L | 0.2968 | 0.0140 | **4.73%** |
| occ, DINOv3-L | 0.2775 | 0.0070 | **2.54%** |
| det mAP, DINOv2-L | 0.1192 | 0.0040 | 3.36% |
| det mAP, DINOv3-L | 0.1274 | 0.0020 | 1.60% |
| det NDS, DINOv2-L | 0.1289 | 0.0026 | 2.02% |
| det NDS, DINOv3-L | 0.1346 | 0.0019 | 1.40% |

The real ratios are **1.4×–2.3×**, not 3–5×.

**Error 2 — untestable at this n, which is fatal.** Comparing two sds estimated
from 4 runs each is an F-test with (3,3) degrees of freedom, whose two-tailed 5%
critical value is **≈15.4**. Our variance ratios:

| comparison | F |
|---|---|
| occ vs det mAP, DINOv2-L | 1.98 |
| occ vs det mAP, DINOv3-L | 2.52 |
| occ vs det NDS, DINOv2-L | 5.47 |
| occ vs det NDS, DINOv3-L | 3.30 |

**None is close.** At n=4 an sd estimate carries a 95% interval spanning roughly
0.57×–3.7× its own value; these samples cannot distinguish the two noise levels.

**This is the third instance of the same error in this project** — asserting a
difference without measuring it against its own noise — and it was committed
*inside the document arguing that differences must be measured against their own
noise*, one section after doing exactly that correctly for the means. The seed
discipline was applied to the quantity under test and not to the claim made about
the test's own variance.

> **Rule 18 — a statement about variance needs a variance test, not an eyeball.**
> Ratios of standard deviations estimated at n=4 are nearly uninformative; say so
> or do not make the claim. If a reproducibility comparison matters, it needs
> n≈20 per arm, which is a different experiment from the one that establishes a
> mean difference.

**What stands after the correction:** the det-vs-occ *mean* comparisons (p=0.018,
p=0.014, p=0.064) are unaffected — they are tests on means, correctly powered and
correctly computed. Only the claim about which metric is *noisier* is withdrawn.
The point estimates hint that det may be the tighter measurement, and that remains
a reasonable thing to check properly; it is not a result.
