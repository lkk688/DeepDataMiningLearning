# What does 3D actually contribute to a driving VLA? — an audit

**Status:** the measurements are done and they are consistent; the positioning is decided;
the writing is not started. This document is the synthesis — our experiments, the literature
survey, and the case for what to claim. Engineering detail lives in
[`QWEN_DRIVE_LIDAR_TOKENS.md`](QWEN_DRIVE_LIDAR_TOKENS.md); the project-wide index is
[`RESEARCH_DIRECTIONS.md`](../../../closedloop_occ/RESEARCH_DIRECTIONS.md) (private, gitignored) (§4.6, §4.7).

---

## 0. The thesis, in one paragraph

Driving VLAs report gains of **1–2 %** from adding a 3D modality (LiDAR input, or occupancy
supervision) and attribute them to 3D understanding. We measure that a channel of exactly
that shape — trainable tokens at a frozen VLM's input, read by a frozen 3D head — yields
**+0.12 fwIoU from an input that carries no information at all**, a *fixed* token block that
merely recalibrates the frozen head to the evaluation's ground-truth convention. Across six
papers that inject 3D into a language model, **not one** runs a wrong-input, shuffled-input
or constant-input control; four expert reviewers at confidence 5 did not ask for one either.
The claim is therefore not "LiDAR does not help" — it demonstrably does, by a lot, when fused
at BEV level — but:

> **A modality-injection gain in a frozen VLM + head can be produced in full by an
> information-free token block. Without a wrong-input or constant-input control, such a gain
> is indistinguishable from recalibration, and the field currently reports it without one.**

---

## 1. What we measured

Every number below is from our own runs. Where a claim was retracted, the retraction is kept.

### 1.1 The apparatus — and two defects in the released model

Host: **Qwen-Drive-1.0** (Qwen3.5-4B VLM + BEVFormer head: det 7-class/900q, occ
200×200×16/10-class, map 200×400/6-class). Two things had to be fixed before any science:

| defect | measurement | fix |
|---|---|---|
| The released BEV head **does not backpropagate** — `ms_deform_attn_bf16_forward` calls the JIT extension directly with no `torch.autograd.Function`, so every deformable-attention output has `grad_fn=None` | all head outputs `requires_grad=False`; `adaptor`/`vit_neck` kept the graph, localising the break | route through the release's **own** pure-torch fallback; **99.983 %** occ-argmax parity with the kernel |
| **99 % of the forward is one layer.** `Qwen3_5VisionPatchEmbed` is an `nn.Conv3d(3,1024,k=[2,16,16],s=[2,16,16])` — the kernel covers its entire input, i.e. a matmul in convolution clothing | steady state `infer()` 92.76 s, of which `patch_embed` **91.79 s**; vision blocks 0.00 s; checkpointing on vs off 91.85 vs 92.86 (not the cause) | equivalent matmul: isolated **91.49 s → 0.0046 s**, `max abs diff = 0.0` (bit-identical); end to end **92.76 s → 1.02 s, 91×** |

The 91× is what made everything else affordable: a nuScenes-train epoch went from ~780 to
~42 GPU-hours, a training step from ~100 s to 5.4 s.

**A third defect is in the *data*, not the code.** Comparing Qwen-Drive's shipped GT against
our Occ3D-derived mapping of the **same sample token** (n = 2, the only nuScenes frames the
release ships): occupied IoU **0.60**, and ~90 % of the voxels they call occupied and Occ3D
calls free lie **outside `mask_camera`**. Restricted to Occ3D's observed region the same
comparison is **0.84 / 0.87**. At the model level the released Qwen-Drive on 120 packed val
frames scores fwIoU **0.184 unmasked → 0.324 inside `mask_camera` (+76 %)**. So 0.324 is the
headroom number, and **0.84–0.87 is the ceiling any model trained to their convention can
reach against our GT** — never read it against 1.0.

### 1.2 Token injection into the frozen VLM — the main experiment

Frozen VLM, frozen BEV head, train **only** a projector writing K tokens into the LLM's input
embeddings. Arms `none` / `zero` / `const` / `lidar` / `shuffle`; two tokenisers; a 2×2 of
objectives. Two design points that are not optional:

* **Tokens must be inserted BEFORE the image tokens.** Appending them at the end (which
  preserves every image token's rope position, and looks strictly better for that reason)
  gave an **exactly zero** gradient: the decoder is causal and the BEV head reads only
  image-token positions.
* **`zero` is the correct baseline, not `none`.** Injecting 64 *null* tokens costs
  **−0.0122 mIoU** on its own — more than the whole effect measured at that point. Against
  `none` a result reads as a loss; against `zero`, as a gain. Same number, opposite sign.

**Channel bandwidth** (one frame, untrained projector, only the point cloud changes):

| pair | histogram K=64, d=19 | **frozen detector queries K=200, d=128** |
|---|---|---|
| `self` vs `other` — **information transfer** | 0.157 % | **0.142 %** |
| `random` vs `zero` | 0.259 % | 0.307 % |
| `none` vs `zero` — the K extra positions alone | 0.208 % | 0.258 % |

A real NDS-0.6922 LiDAR detector's object queries carry **no more** into the BEV head than a
hand-made 8×8 height histogram. More tokens buy **more perturbation and no more information**.
This pre-answers the obvious objection ("your encoder was a histogram") with a measurement.

**The objective 2×2** (24 real nuScenes frames, 20 epochs, detector queries, scored inside
`mask_camera`). Trained against pooled unmasked CE the projector *destroys* objects:

| config | trained loss | support | eval CE @20 | **fwIoU @20** | **ΔfwIoU** |
|---|---|---|---|---|---|
| **A** | CE | all voxels | 0.2159 | 0.2740 | **−0.0378** |
| **B** | CE | `mask_camera` | 0.3363 | 0.4216 | **+0.1097** |
| **C** | CE + Lovász | all voxels | 0.2567 | 0.3418 | +0.0299 |
| **D** | CE + Lovász | `mask_camera` | 0.3527 | **0.4333** | **+0.1215** |

Under A the only classes that improve are the two largest (`empty` +0.002, `driveable`
+0.013) while every object class collapses (`bicycle` **−0.204**, `vehicle` −0.119,
`traffic_cone` −0.104). **The supervision region, not the class imbalance, was the dominant
cause** — masking alone recovers +0.1097 of the +0.1215. That was the opposite of our prior:
the in-house "Lovász is the best single change" evidence came from models trained *from
scratch* on Occ3D, where the supervision region was never in question. The tell is that the
reported plain unmasked CE **rises** in B and D while fwIoU rises — in this setup that loss is
**anti-correlated with quality**.

### 1.3 The decisive control

`const`: the projector's input is the **same vector for every frame**, so the tokens carry
zero information about anything; only `pos`, the MLP bias and `gate` can move.

| arm | train ΔfwIoU | **held-out ΔfwIoU** (24 frames never trained on) |
|---|---|---|
| **`const`** — zero information | +0.1242 | **+0.0881** |
| `lidar` s0 / s1 | +0.1215 / +0.1224 | — / +0.0778 |
| `shuffle` s0 / s1 | +0.1475 / +0.1326 | — / +0.0818 |

**`const` is the best arm on held-out data.** Seed spreads: `lidar` 0.0009, `shuffle` 0.0149
— so seed-0's apparent "shuffle beats lidar by 0.026" was mostly noise, and the honest
statement is that **`const`, `shuffle` and `lidar` are indistinguishable**. The gain
generalises to unseen frames, which is what identifies it as a **recalibration of the frozen
head** rather than per-frame memorisation. The injected modality contributes nothing
measurable.

### 1.4 The other topology: a VLM tap as the camera backbone

Frozen Qwen-Drive image tap vs DINOv2-L as the camera branch of our own LSS occupancy head,
`balanced` region weighting, 2 seeds each, identical otherwise. This also re-opened §4.6's
claim that the occluded-region plateau is "a property of the available information" — a claim
established by varying the **loss** (two weightings → 0.069 / 0.070) and never by varying the
**feature prior**.

| metric | DINOv2-L mean (spread) | qwendrive mean (spread) | Δ |
|---|---|---|---|
| `pred_mIoU` | 0.0685 (0.003) | **0.0685** (0.001) | **+0.0000** |
| `pred_geo` | 0.0665 (0.007) | 0.0560 (**0.028**) | −0.0105 |
| in-mask mIoU | 0.199 (0.014) | 0.1865 (**0.049**) | −0.0125 |
| `all_mIoU` | 0.1355 (0.001) | 0.1220 (0.002) | **−0.0135** |

`pred_mIoU` is **identical to four decimals**, confirming the pre-registered prediction (no
single-frame prior can supply labels that are observations from *other timestamps*). The VLM
tap is equal-to-worse and **3.5–4× less stable across seeds**. The occluded-space ceiling is
an **information** ceiling, now confirmed on a second independent axis.

**⚠ CORRECTION (2026-09-22) — the "worse and noisier" reading was the resolution confound.**
The tap's grid is 448 cells/camera against DINOv2-L's 900 — half the lift's ray density.
`--feat-upsample 2` (1792 cells) closes it:

| metric | DINOv2-L | qwendrive (native) | **qwendrive, upsampled** |
|---|---|---|---|
| `pred_mIoU` | 0.0685 | 0.0685 | **0.069** |
| `pred_geo` | 0.0665 | 0.056 | **0.0665** |
| in-mask mIoU | 0.199 | 0.1865 | **0.201** |
| `all_mIoU` | 0.1355 | 0.122 | **0.138** |
| seed spread, `pred_geo` | 0.007 | 0.028 | **0.003** |

**At matched ray density the VLM tap equals DINOv2-L on every metric, and its seed stability
recovers.** The deficit and the 4× instability were entirely resolution, not features. The
correct statement is *equal*, not *equal-to-worse*. The core conclusion is unchanged and now
confound-free: `pred_mIoU` does not move, so the occluded-space ceiling is informational.

### 1.5 Two inherited protocol flaws, found while reproducing the baseline

* **The val set is 97 % train data** — train is `items[:2044]`, val is `items[::7][:300]` from
  the same list with no scene split; **292 of 300** val samples are in train.
* **2044 frames span 53 scenes**, not a broad sample of 850.

Neither overturns §4.6: leakage *inflates* the measured number, so 0.070 on seen frames is an
**upper bound** and the "dead" reading is conservative. But absolute magnitudes from this
protocol are not trustworthy, and any future claim of *improvement* on it must be checked on
a disjoint split.

### 1.6 Everything we ran, in one table

| # | question | answer | where |
|---|---|---|---|
| 1 | Does the released head backpropagate? | No — raw CUDA op; fixed with the release's own fallback | §1.1 |
| 2 | Why is the forward 100 s/frame? | One `Conv3d`; 91× fix, bit-identical | §1.1 |
| 3 | Do the two GTs agree? | occupied IoU 0.60; 0.84–0.87 inside `mask_camera` | §1.1 |
| 4 | How wide is the injection channel? | 0.14 % of voxels change class on a full cloud swap | §1.2 |
| 5 | Is the tokeniser the limit? | No — detector queries ≈ height histogram | §1.2 |
| 6 | Is the objective the limit? | Partly; the **supervision region** dominates | §1.2 |
| 7 | Is the gain about LiDAR? | **No** — `const` reproduces it in full | §1.3 |
| 8 | Is a VLM tap a better camera prior? | No — **equal** to DINOv2-L at matched resolution (the earlier "worse, 4× noisier" was a resolution artefact) | §1.4 |
| 9 | Is the occluded ceiling informational? | **Yes**, on a second axis | §1.4 |

---

## 2. Survey — how 3D actually enters end-to-end and VLA models

Six mechanisms. The numbers are from each paper's own tables (full text read, not abstracts,
except where marked).

| # | mechanism | representative work | where 3D enters | reported effect |
|---|---|---|---|---|
| 1 | **Dense BEV fusion** | BEVFusion; [GMF-Drive](https://arxiv.org/pdf/2508.06113); DriveInsight | before any head | **ours: occ mIoU 0.302 → 0.558**, geo-IoU 0.669 → 0.838 |
| 2 | **Multi-scale transformer feature fusion** | [TransFuser](https://arxiv.org/pdf/2104.09224) | intermediate feature maps | the classic CARLA E2E recipe |
| 3 | **Sparse query fusion** | CMT, MetaBEV, MoME, DriveTransformer | object queries attend to both | Limited-FOV NDS: MetaBEV 47.0 → CMT 54.0 → **MoME 58.3** |
| 4 | **BEV tokens + cross-attention *beside* the LLM** | [2608.20890](https://arxiv.org/html/2608.20890) | fused BEV → planner | **LiDAR-only mAP 70.7 → 74.0 fused**; PDMS 91.0 → 92.2 |
| 5 | **Projector / Q-Former → LLM tokens** | [LVLDrive](https://arxiv.org/abs/2512.24331); [LiDAR-LLM](https://arxiv.org/pdf/2312.14074) | the LLM's input | LVLDrive **+0.01** BEV mIoU (0.18 → 0.19) |
| 6 | **3D as train-time supervision only** | [OccVLA](https://arxiv.org/pdf/2509.05578); [VLGA](https://arxiv.org/html/2606.12396) | discarded at inference | OccVLA **+1.5 %** meta-action; VLGA has no fusion baseline |
| (6b) | **Symbolic serialisation** — detections → text → prompt | [PLA / 2507.23540](https://arxiv.org/html/2507.23540v1) | the prompt | **no baselines, no ablations, single scenario** |

**The dividing line is sharp: 1–4 fuse *before or beside* the language model and report real
gains; 5–6 fuse *inside* it and report 1–2 %.** We tested 5. Our result is consistent with
everyone else's numbers; we added the control that says what the 1–2 % is made of.

### 2.1 LVLDrive, read in full — its own table is the best motivation for the audit

Base OmniDrive-Agent; EVA-02-L vision; LLaVA-v1.5 LLM; **FSDv2 pretrained on nuScenes** as
the point encoder; **nothing frozen** (joint fine-tune, LoRA on the LLM, 8×A100-40G × 30 h).
Their Table 3, independent Q-Formers, **each row a separately trained model**:

| modality | CR (%) ↓ | IR (%) ↓ | grounding BEV mIoU ↑ |
|---|---|---|---|
| camera only | 0.33 | 3.32 | **0.18** |
| **LiDAR only** | 0.39 | 3.26 | **0.17** |
| camera + LiDAR | 0.29 | 3.20 | **0.19** |

and the gate ablation at fixed C+L: no gate **0.17** → single 0.21 → multi-head 0.21.

1. A fine-tuned LiDAR **detector** backbone inside a VLM is **worse than the camera** at
   metric 3D localisation.
2. Adding LiDAR to camera buys **+0.01**; their headline 0.21 comes from the **gate**, not the
   modality.
3. It is a **retrained-arm** ablation, which cannot separate "the model reads the LiDAR" from
   "the extra parameters helped".

Their grounding metric also hands the model the **2D box in the question** — it is not on any
ruler a detector is measured on.

### 2.2 OccVLA (ICLR 2026, rejected) — and its own numbers

3D is a **training-time signal only**: occupancy is the LLM's *output*, camera-only at
inference. PaliGemma + adapters.

| ablation | result |
|---|---|
| occupancy supervision off → on | Avg 65.50 → 66.37, Overall 41.48 → **43.08** (≈ +1.5 %) |
| **ego trajectory absent → present** | L2 avg **0.48 → 0.28** (**−42 %**) |

**The ego's own past trajectory is worth ~25× the occupancy supervision.** This is the
ego-status dominance our project measured independently (`occupancy is a null open-loop
planning input`, commit `fa6bd4c`). They also argue that camera-only + occupancy-supervision
matches methods fed **ground-truth occupancy as input** — the same conclusion from the other
direction: the channel into the LLM carries little, even when the 3D is perfect.

**Why it was rejected.** AC: *"limited technical contribution ... and marginal experimental
improvement"*. Reviewers (all confidence 5): resemblance to Ross (ICLR 2025) undiscussed;
the EMMA baseline *"inaccurate and potentially misleading"*; the anti-scalability argument for
3D boxes outdated given auto-labelling; **"evaluation of the planning capability is limited to
open-loop metrics on nuScenes, lacking essential simulation or closed-loop experiments"**;
and *"the nuScenes planning benchmark is rapidly becoming outdated"* — WOD-E2E requested.

### 2.3 A published mechanism for our bandwidth result

[ScanReQA](https://arxiv.org/pdf/2504.04540) (Tsinghua + ByteDance) measures **attention
sink** in 3D LLMs: attention concentrates on fixed early token positions regardless of
content, and **masking those sink tokens does not reduce QA accuracy — it slightly improves
it on ScanQA**. Their QA-relevant object tokens do matter (>2 % drop when masked), so in a
*trained* 3D LLM the tokens are partly read. Ours was **frozen** — the harsher case. This is
an independent, published mechanism for why an injected 3D token budget is largely not read.
(Indoor ScanNet, and their headline is that point clouds *do* beat text-only LLMs — no
conflict with our claim.)

### 2.4 Not a channel: LiT

[LiT](https://yxlao.github.io/lit/) (NeurIPS 2024) is **LiDAR sensor-domain translation** —
"Languages" means scan patterns, not natural language. Neural implicit static background +
tracked dynamic objects + GPU ray-casting with the *target* sensor model. No language model.
Relevant to us twice over, on the **data** side rather than the channel side: it answers the
bottleneck LVLDrive itself names (*"the limited availability of large-scale, naturally paired
text–LiDAR data"*), and it would supply the sensor-faithful point clouds our
`xdomain-occ-transfer` line lacks (AV2 renders are xyz-only, no intensity).

---

## 3. The control nobody runs

Checked in each paper's own text, not from abstracts:

| work | 3D enters at | wrong / shuffled / constant input control? |
|---|---|---|
| LVLDrive | LLM input (Q-Former) | ❌ retrained arms only (C / L / C+L) |
| VLGA | train-time supervision | ❌ no fusion baseline at all |
| LiDAR-LLM | LLM input | ❌ staged curriculum, no such control |
| 2608.20890 | BEV tokens beside the LLM | ❌ component removal only |
| OccVLA | train-time supervision | ❌ on/off only |
| PLA / 2507.23540 | prompt text | ❌ no baselines whatsoever |

**6 / 6.** And in OccVLA's four reviews — all confidence 5 — the reviewers asked for more
datasets, closed-loop experiments, latency profiles, related work and justification of
claims, but **none asked whether the 3D modality was actually used**. The gap is in the
reviewing standard, not only in the papers.

---

## 4. What the field is missing that we already have

The OccVLA reviews name three deficiencies of this subfield, and we hold instruments for all
three:

| reviewer demand | what we have |
|---|---|
| "lacking essential simulation or **closed-loop** experiments" | NF-Bench (21 forcing scenes, contact-onset fault attribution) + AlpaSim port |
| "nuScenes planning benchmark is rapidly becoming **outdated**" | the two-ablation audit is benchmark-agnostic; WOD-E2E is a port, not a redesign |
| *(nobody asked)* modality-use control | `const` / `shuffle` / `zero` arms, measured three times |

Plus the ego-status result, which OccVLA's own Table 5 reproduces at 25× the size of its
headline contribution.

---

## 5. The paper

**Claim.** Not "we built a better fusion" — that is mechanism 1–4 and it is crowded and
solved. The claim is a **judgement criterion**:

> A modality-injection gain in a frozen VLM + head is reproducible in full by an
> information-free token block that recalibrates the frozen head to the evaluation's GT
> convention. Reported 1–2 % gains from injecting 3D into driving VLAs are, without a
> wrong-input or constant-input control, indistinguishable from that.

**Is it worth writing? A cost/benefit, stated plainly.** *For:* the `const` result is
striking, nobody has it, and six papers' own tables support the picture. *Against:* (1) this is
a methodological paper about a subfield barely a year old and moving fast — the specific
targets may be superseded within a review cycle; (2) **this subfield's reviewers do not
currently value modality-use controls** — OccVLA's four confidence-5 reviewers asked for
datasets, closed-loop and latency, so an audit must first argue that the question matters;
(3) negative/methodological contributions cap out at short-paper or workshop scale without a
second axis. **Recommendation: this is the methodology section of the `occ→det` label-efficiency
paper (p = .006/.0009), or a standalone short paper — it should not displace that positive as
the main thrust.**

**Shape.** The trainable stack is the **apparatus**, not the contribution — no released
artifact permits a feature-level ablation (LVLDrive and VLGA have no code; Qwen-Drive has no
training code: `grep -rlE "def loss|CrossEntropyLoss|HungarianAssigner|optimizer|AdamW"
src/ scripts/` returns **nothing**), so the instrument had to be built. Stack in §3 of the
paper, findings in §4–6.

**What makes it more than a case study.** One model is a bug report; a second injection design
makes it a property of the *channel*.

⚠ **Do NOT do this by re-implementing LVLDrive's Gradual Fusion Q-Former** — an earlier draft
of this document called for exactly that and it was a bad call. **A negative result about an
unreleased method you re-implemented yourself is the weakest kind of claim**: "you did not
reproduce it properly, that is why it failed" is unanswerable when there is no code and the
method is trained on their own unreleased SA-QA dataset. We could not even align to their
numbers to show the re-implementation was faithful.

**Instead: vary the injection design inside our own stack** — input-embedding tokens (done),
per-layer KV prefixes, and a resampler/Q-Former-style block — and run the same `const` /
`shuffle` / `zero` controls on each. The claim becomes *"across N injection designs in a
controlled, reproducible stack, the gain is indistinguishable from constant input."* Weaker
than testing their exact method, but defensible, cheaper, and it cannot be dismissed on
reproduction grounds.

---

## 6. What remains to be run

| | task | cost | why |
|---|---|---|---|
| **must** | a second **injection design in our own stack** (per-layer KV prefix, resampler) + the same controls | ~2 days | case study → a property of the channel. **Not** a re-implementation of an unreleased method (§5) |
| **must** | BEVFusion-LC on the same 120 packed tokens, same ruler | hours | the other half of the headroom number; packer + taxonomy are ready |
| ✅ | `--feat-upsample 2` control for §1.4 | done | closed the confound — and **corrected** the conclusion from "worse" to "equal" |
| should | a third seed on `pred_geo` | ~4.5 h | 2 seeds give a spread, not a σ |
| optional | WOD-E2E port | days | the reviewers' explicit demand |
| **no** | VLM tap + `--lidar-fusion` | 9 h | camera is worth +0.0196 NDS on top of LiDAR and this tap is *worse* and 4× noisier — a predictable null |
| **no** | more projector / tokeniser / channel tuning | — | it would be optimising a recalibration |
| ⭐ | **audit the brain→cerebellum interface** of a dual-system stack (§9) | weeks | the same instrument, applied to the architecture that actually won |

---

## 7. Limitations, stated before a reviewer states them

* Our token-injection measurements are on **one host model** (Qwen-Drive) with a **frozen**
  VLM and head. LVLDrive fine-tunes everything. A frozen stack is the harsher case, and that
  is an argument, not a proof.
* The overfit gates train on the frames they evaluate; the `const` result is the exception —
  it holds on **24 held-out frames**, which is the load-bearing part.
* 24–120 frames, 1–2 seeds. Directions and per-class patterns are unambiguous; magnitudes are
  not seed-certified.
* The occupancy baseline protocol leaks (§1.5). It biases *against* our conclusion there.
* **What would falsify us:** a driving VLA showing a modality gain that *survives* a
  constant-input and wrong-input control at the same magnitude. We have not found one, and
  no paper has run the test.

---

## 8. Reproduction

```bash
cd DeepDataMiningLearning/bevdet/lidar_tokens
PY=/home/010796032/miniconda3/envs/py310/bin/python
export CUDA_HOME=/data/rnd-liu/cudatk CPATH=$CUDA_HOME/targets/x86_64-linux/include
export PATH=$(dirname $PY):$PATH

$PY check_grad.py --stage ops        # which released CUDA ops lack a backward (no GPU)
$PY check_grad.py --stage parity     # kernel vs differentiable fallback (99.98 % argmax)
$PY check_grad.py --stage profile    # where the forward time goes (one Conv3d)
$PY qwen_drive_labels.py --self-test # the 18 -> 10 common ruler (13/13)
$PY train_projector.py --self-test   # the ported Lovász vs a direct Jaccard (5/5)

$PY pack_nuscenes.py --split val --n 120 --out <pack>        # projection control included
( cd <mmdet3d> && $PY .../extract.py --tokens <pack>/manifest.json --out <pack>/q.npz )
./run_step4.sh    # the 4-arm gate, detector queries, inside mask_camera
./run_step5.sh    # the objective 2x2
./run_step7.sh    # const + seeds + held-out    <- the decisive control
```

Every run writes `config.yaml`, `meta.json` (seed, git commit + dirty flag) and `log.json`
(per-epoch loss, fwIoU, mIoU, per-class IoU, per-class GT voxel counts, argmax drift).

---

## 9. The dual-system architecture — and why our negative result predicts it

**The proposal.** Vision + language in a slow "brain" (System 2, low rate); camera + LiDAR +
radar into a fast "cerebellum" (System 1, high rate); the brain steers the cerebellum. This is
not a speculation — it is the dominant architecture, in robotics and in driving, and **our
measurement is the explanation for why it won.**

| system | model | rate | what it consumes |
|---|---|---|---|
| **NVIDIA GR00T N1** System 2 | Eagle-2 VLM, features taken from the **12th layer** (2B model) | **10 Hz** | images 224², language |
| GR00T N1 System 1 | Diffusion Transformer, flow matching, **cross-attends to System 2's tokens**, embodiment-specific encoders/decoders | **120 Hz** | proprioception + action |
| **DriveVLM-Dual** slow | DriveVLM | async | camera, language |
| DriveVLM-Dual fast | the **traditional AD pipeline** — 3D detectors, occupancy nets, motion planner | high-frequency | **this is where LiDAR lives** |
| **Senna-2** | VLM + E2E policy, asynchronous, a **memory bank caches VLM features** | async | — |

Two structural points:

* **The coupling is brain → cerebellum, never sensor → brain.** GR00T's System 1 *cross-attends
  to System 2's tokens*; DriveVLM-Dual's planner "selectively receives a trajectory from the VLM
  branch as additional input". The slow system emits a **coarse intent** (trajectory,
  meta-action); it never ingests raw 3D.
* **That is exactly the dividing line we measured** (§2). Mechanisms that fuse *before or
  beside* the language model work; mechanisms that fuse *inside* it return 1–2 %. The
  dual-system architecture is the one that puts LiDAR where fusion demonstrably works and asks
  the language model only for what it is good at.

### The hardest single data point in this whole document

**NVIDIA's Alpamayo-R1 does not take LiDAR as a model input.** Inputs are multi-camera video,
calibration and ego-motion. And this is not for lack of data or hardware: their own open
dataset ships **7 cameras + a top-mounted 360° LiDAR for every clip** (radar for 163 850
clips), and the DRIVE Hyperion reference platform carries 14 cameras, 9 radars, a LiDAR and 12
ultrasonics. Alpamayo 1 is a 10 B CoT VLA on Cosmos-Reason; Alpamayo 2 Super scales it to 32 B
with 360° surround perception and a Meta-Action output (yield / lane change / stop).

**The best-resourced driving-VLA programme in the industry has the LiDAR sitting in its own
dataset and on its own reference vehicle, and keeps it out of the reasoning model — emitting
Meta-Actions instead.** Whatever else our negative result is, it is not idiosyncratic.

### ⭐ DONE — the instrument transferred, on NVIDIA's flagship open VLA

**Alpamayo 1.5 10B runs locally** (weights 20 GB, venv, PhysicalAI-AV data; 7.1 s and 21.6 GiB
per clip). It is the dual-system architecture in one model: Cosmos-Reason2-8B VLM as the
brain, a non-causal action expert with flow matching as the cerebellum, 64 waypoints over
6.4 s, and **the coupling is the VLM's KV cache** — the expert denoises with
`past_key_values = vlm_outputs.past_key_values`, so the generated reasoning tokens are
literally in the cache it reads. Full recipes in
[`TUTORIAL_E2E_DRIVING_MODELS.md`](TUTORIAL_E2E_DRIVING_MODELS.md).

**Audit 1 — which sensor inputs does it use?** 40 clips, only one input replaced at a time:

| arm | replaced | mean L2 | median ratio | beats `real` |
|---|---|---|---|---|
| `real` | — | **2.523** | 1.00× | 0/40 |
| `frozen` | ego history → constant | 19.836 | **8.47×** | 4/40 |
| `shuffled` | ego history → other clip | 15.072 | 6.61× | 3/40 |
| `blind` | cameras → mid-grey | 4.450 | **2.25×** | **13/40** |
| `camswap` | cameras → other clip | 8.717 | 4.91× | 4/40 |

**Ego history matters 3.8× more than the cameras**, and blinding every camera still beats the
unmodified model on a third of clips. Under `blind` the Chain-of-Causation collapses to one
verbatim constant ("Keep lane since the lane is clear ahead") while the trajectory barely
moves — the reasoning is perception-driven, the trajectory is not.

**Audit 2 — does the cerebellum read the brain?** Teacher-force the reasoning text: identical
images, identical ego history, **only the CoT differs**. 30 clips, 150 runs.

| arm | reasoning given to the expert | mean L2 | vs `real` | Wilcoxon |
|---|---|---|---|---|
| `real` | teacher-forced, the model's own text | 1.777 | 1.00× | — |
| `swap` | **another clip's** reasoning | 1.708 | 0.95× | p = 0.088 |
| `const` | "Keep lane since the lane is clear ahead" | 1.715 | 0.95× | p = 0.145 |
| `contradict` | **"Brake hard for the pedestrian ahead"** | 1.490 | 0.89× | p = 0.061 |

**No substitution degrades the trajectory** — not another clip's reasoning, not a constant
string, not one demanding emergency braking. Effect sizes:

```
removing the ego history   17.31 m
changing the CoT content    0.68 m      ->  26x
```

**The reasoning trace — the artefact that makes the system "interpretable and auditable" —
has ~26× less influence on the executed trajectory than the ego history.** Two honesty notes:
`contradict` trending *better* is unexplained and not significant, so this is "no
degradation", not "contradiction helps"; and sampling at temperature 0.6 alone (rollout 2.141
vs teacher-forced 1.777, differing on 27/30 clips) is a **larger** effect than the entire
reasoning content.

> ⚠ **CORRECTED 2026-09-23** (TUTORIAL_E2E_DRIVING_MODELS §3.9). "No substitution degrades"
> is true, but "the CoT is not read" is not. A signed, paired, 6-seed test on 60 clips,
> replicated on an independent seed set, shows that Alpamayo's expert follows the rationale
> **longitudinally**: "stop" −0.9 m, "accelerate" +0.7 m at 5 s, p < 1e-8. It does not follow
> it laterally. The effect is smaller than Alpamayo's own sampling spread, which is why the
> L2 and divergence tests missed it. Qwen-Drive obeys both axes, at ~10× its seed spread. The
> ratios (26×, 13.2×) stand as magnitudes, not as "zero".

**Four independent stacks now say the same thing.**

| stack | comparison | result |
|---|---|---|
| ours (`fa6bd4c`) | occupancy vs ego history | occ-only L2@3s 5.04, ego-only **2.09**, occ+ego **2.11** → **Δ ≈ 0**; a larger perception backbone makes it *worse* (2.49). Ego-only equals the analytic constant-velocity floor |
| OccVLA (their Table 5) | occupancy supervision vs ego trajectory | +1.5 % vs **−42 %** ≈ **25×** |
| Alpamayo 1.5 | ego history vs cameras | **8.47× vs 2.25×** |
| Alpamayo 1.5 | ego history vs reasoning content | **26×** |

Three are entirely independent codebases. This is no longer a case study.

### What this opens for us — the instrument transfers
### What this opens for us — the instrument transfers

If the architecture is settled, the open questions move to the **interface**, and they are
exactly the questions our instrument answers:

1. **Does the fast system actually read the brain?** DriveVLM-Dual's planner *"selectively
   receives"* a trajectory; Senna-2 caches VLM features in a memory bank. **Nobody runs a
   constant-output or shuffled-output control on the brain→cerebellum channel.** Replace the
   slow system's output with (a) a constant meta-action, (b) another scene's trajectory, and
   measure. This is the same audit, on the architecture that won, and it is *live* rather than
   retrospective.
2. **We already have the prior that makes it non-trivial.** This project measured that
   occupancy is a **null open-loop planning input** (commit `fa6bd4c`) and that **ego status
   dominates** — and OccVLA's own Table 5 reproduces it at 25× its headline (ego trajectory
   L2 0.48 → 0.28 vs occupancy +1.5 %). A coarse intent signal from a 10 Hz brain is *exactly*
   the shape of input that our own work says tends to be null.
3. **The rate question is ours too.** What happens between brain updates? Our occupancy
   degradation curve already found **recall binding and latency nearly free** — which predicts
   that a slow brain is affordable but a low-recall cerebellum is not. That is a testable,
   architecture-level claim.
4. **And the degradation question.** §2.6 F4: the camera compensates *uniformly degraded*
   LiDAR but **not spatially absent** LiDAR. In a dual-system stack the brain carries priors a
   BEV grid does not ("the road continues behind the truck"). Does brain guidance close the
   spatially-absent gap? That is the one place where a language-level prior has a mechanism to
   matter, and we have both the robustness harness and the closed-loop bench.

**Cost, honestly.** (1) is cheap if we can drive any dual-system stack; DriveVLM/Senna-2 code
availability is unverified and must be checked before planning. (3) and (4) need a trainable
dual-system stack, which is weeks, not days. (2) is free — it is re-reading results we already
have through this lens.

**What not to conclude.** That the dual-system architecture makes LiDAR-in-a-VLA work. It does
not: it makes the question disappear by never asking the language model to ingest LiDAR. That
is a good engineering answer and a poor research question — the research question is whether
the brain's output is used at all.

---
