# Paper Draft — Occupancy Pretraining Buys Label-Efficient 3D Detection — and Almost Nothing Else Does

*Working draft compiled from the full experimental log (2026-06 → 2026-09). All numbers are our own
runs on nuScenes / Occ3D-nuScenes unless marked as a published reference. Companion docs:
`TUTORIAL_RESULTS_AND_REPRODUCTION.md` (chapter-per-result, reproducible),
`TUTORIAL_LABELFREE_PERCEPTION_JOURNEY.md`, `RESULTS_LABELFREE_OCC_2x2_TRANSFER.md`,
`PLAN_FLASHOCC_MIGRATION.md`, `PLAN_FUSIONOCC.md`, `REFS_2026_OCC_GAUSSIAN_LABELFREE.md`.*

---

## Abstract (draft)
Labelled 3D boxes are the scarcest resource in autonomous driving. We show that **pretraining on
dense 3D occupancy makes 3D detection markedly more label-efficient**, and — more usefully — we
identify *why*: on nuScenes with only 2k detection labels, occupancy pretraining lifts official mAP
from **0.121 to 0.163 (+35%)**, and the gain is **not uniform** — it is concentrated almost entirely
on **rare, thin classes** (pedestrian / cone / barrier, 1.5–3×) while easy `car` is a wash. The
occupancy encoder learns dense per-voxel geometry exactly where detection-from-scratch is
data-starved, and transfers it.

The result only means something because we tried, under one controlled harness, essentially every
competing way to get that gain — and **none of them worked**. Frozen foundation models: plain
**DINOv2 beats agglomerative (RADIO) and geometry (VGGT)** FMs, and VGGT's tokens are **33% worse**
as an occupancy backbone. Label-free occupancy pretexts are **null-to-negative**, and a pretext with
**+60% better foreground pseudo-labels transfers *worse*** — better labels ≠ better transfer.
Extending the study to the two non-camera modality axes closes the same way: a **point FM**
(Sonata, teacher-free) *hurts* the LiDAR branch, and a **depth FM** (LDCM) appears to give +38%
until a beam ladder shows the gain is **the LiDAR the prior ingests, not the model** — at zero rings
it reproduces the VGGT null exactly. That last result generalises into a methodological rule we
argue the field needs: **whenever a "prior" consumes a sensor the baseline lacks, add a zero-sensor
rung before attributing the gain to the model.** We further show **occupancy mIoU does not predict
detection transferability**, so the community's default pretext metric is the wrong one to optimise.
We reproduce a supervised ceiling exactly (FlashOcc-4D-stereo, **0.3809** vs published 0.3784,
finding a normalisation bug worth ~0.04 en route) and release the harness, every negative arm, and a
chapter-per-result reproduction tutorial.

## 1. Contributions
1. **The positive result, with its mechanism.** Occupancy pretraining is label-efficient for 3D
   detection (**+35% official mAP at 2k labels**, holding at 4k/8k), and the gain is **rare-object
   transfer** — a mechanism, not a leaderboard delta. §4.3.
2. **Occ-mIoU ≠ transferability.** Detection transfer keeps scaling with pretraining data while occ
   mIoU saturates — the default pretext metric is the wrong objective. §4.2.
3. **Every alternative, actually run.** A battery of controlled negatives under one harness:
   agglomerative/geometry camera FMs, a **point FM** (Sonata), a **depth FM** (LDCM), Gaussian occ
   teachers, and label-free/pseudo-label pretexts — each with a mechanism, not just a null. §4.1,
   §4.4, §4.7.
4. **A methodological rule: the zero-sensor rung.** Our depth-FM ladder (32 / 8 / 0 rings) shows a
   "camera-only" model carrying a sparse-depth-conditioned prior is not camera-only, and we document
   two further evaluation traps we hit — scoring a sparse-conditioned prior on its own input pixels,
   and scoring an up-to-scale prior as metric. §4.7, §5.
5. **Exact reproductions + a runnable harness.** FlashOcc-4D-stereo (0.3809 vs 0.3784, incl. a
   BGR-normalisation bug worth ~0.04) and GaussianOcc (11.26), ported to a modern stack that runs on
   H100 where the originals cannot; plus a chapter-per-result reproduction tutorial. §4.5.

## 2. Related work (grouped, with our take)
**Occupancy datasets/label-gen.** Occ3D [Tian et al., CVPR'23] — the Occ3D-nuScenes/Waymo benchmark and
a *manual* label-gen pipeline (accumulated LiDAR + LiDAR-semseg + 3D boxes for dynamic/static). CVT-Occ —
followup occ predictor. *We adopt Occ3D's grid/eval; our label-free pretexts replace its two manual
dependencies (semseg→FM projection, boxes→pseudo-tracks) and we show why that transfer is bounded.*

**Label-free / test-time occupancy.** TT-Occ [Zhang et al., CVPR'26, 2503.08485] — test-time occ from
raw sensors + VFMs, no training, SOTA self-sup on Occ3D-nuScenes. GaussianOcc [ICCV'25] — self-sup occ
(we reproduce mIoU 11.26). *TT-Occ shows label-free occ *prediction* is near-solved; our contribution is
the *transfer* question, which none of these address.*

**Open-vocab / FM semantics in 3D.** OnlinePG [Zhai et al., CVPR'26, 2603.18510] — online open-vocab
panoptic mapping with 3DGS (multi-view consensus to denoise 2D-VLM priors). ExtrinSplat [2509.22225] —
decouple geometry/semantics, object-level VLM descriptions. *Both give recipes to denoise our FM-semantic
labels (multi-view consensus, object-level not per-point).*

**Multi-modal robustness / domain adaptation.** PanDA [Pan et al., CVPR'26, 2604.19379] — UDA for
multimodal 3D panoptic seg (modality-drop + 2D+3D pseudo-label refinement). *Directly relevant to
cross-dataset/cross-sensor robustness.*

**Gaussian representations.** VGOcc [Lin et al., 2607.18078] — visual-geometric Gaussians, SOTA vision
occ (foundation-model features + ray-depth + pose-aware fusion). ADGaussian — generalizable feed-forward
GS for AD *reconstruction* (depth-guided positional embedding). IDESplat [CVPR'26, 2601.03824] —
generalizable 3DGS with iterative depth. *These confirm Gaussians excel at *prediction/reconstruction*,
**not** as occupancy *labels* — consistent with our stop-decision; pose/ray-aware feature fusion is the
portable idea.*

**Fusion occupancy.** FusionOcc [Zhang et al., MM2024] — LiDAR-camera fusion occ, **56.6 mIoU
Occ3D-nuScenes** (2D dense-depth + 3D voxel-LiDAR fusion). BEVFusion — the fusion detector we build on.
*The recipe for our SOTA-competitive fusion column.*

**Detection / monocular geometry.** SPAN [Wang et al., CVPR'26, 2511.06702] — 3D↔2D projection-alignment
(a label-free geometric aux loss we can borrow). FlashOcc — efficient channel-to-height occ (our exact
supervised ceiling).

**Depth foundation models / sparse-depth completion.** **LDCM** [Yu et al., ICLR'26, 2605.30115] —
a *large depth-completion model*: a frozen monocular depth FM (DepthAnythingV2) supplies relative
depth, a **Poisson gradient-field alignment** fuses it with sparse LiDAR into a coherent *metric*
coarse depth, and a DINOv2 ViT-B dual-encoder (RGB + coarse depth, prompt-fused) regresses an
**intrinsic-free point map** (per-pixel 3D coords) instead of a depth map; ~2.7M samples / 11
datasets, zero-shot across 1–10% random, keypoint, and **64/32/16/8-beam LiDAR** sparsity (KITTI rel.
0.026 vs 0.042 prior best). *Directly relevant: our lift is depth-supervised and depth is its
bottleneck, yet we found a geometry FM (VGGT) to be a **wash** as a depth prior — precisely because a
learned depth head plus LiDAR depth supervision already carried that signal. LDCM differs **in kind**
(metric, **sparse-observation-conditioned**), so it is the right test of whether a depth FM helps
where VGGT could not: the sparse/low-beam and cross-rig regimes. Its intrinsic-free point map also
removes the per-dataset intrinsics coupling that complicates our cross-dataset occ pool.*

**3D / point-cloud foundation models.** **Sonata** [Wu et al., CVPR'25, 2503.16429] — self-supervised
point representations (PTv3, encoder-only, 108M) self-distilled over **140k point clouds**. It names
the **"geometric shortcut"**: in 3D, representations collapse onto low-level spatial cues (normals,
height) because position is handed directly to the operators — fixed by *obscuring* spatial
information and forcing reliance on input features. Outdoor: nuScenes **81.7** mIoU finetuned (80.4
scratch) / **66.1** linear-probe, Waymo 72.9, SemanticKITTI 72.6; extreme data-efficiency (1% of
ScanNet: 25.8→45.3). *Three things for us. (i) The geometric shortcut is an independently-named
mechanism for our own negative — VGGT's geometry-specialized tokens are the wrong bias for a
**semantic** occupancy head. (ii) Sonata reports **only segmentation** outdoors — **no detection** —
so it does not establish the transfer our §4.2 decoupling says must not be assumed. (iii) It is the
LiDAR-side counterpart our benchmark lacks, and an alternative route to label-free pretraining that
sidesteps the teacher-quality bound of §4.3 entirely, because it uses **no teacher at all**.*

**Policy / VLA (why we avoid them).** AutoVLA — VLM-based end-to-end driving (SFT+RL, camera→trajectory
tokens; unreleased, multi-GPU). SimScale [OpenDriveLab] — sim-real scaling for NAVSIM planning (8-GPU).
**Patch Policy [Zhou, Cui, …, LeCun, Pinto, 2607.18236]** — *frozen dense ViT patches + a lightweight
block-causal transformer policy* **beats fine-tuned OpenVLA-OFT by 18% at ~0.7% of parameters**, 6.5
GPU-h on one L40S. *This is the direct validation of our frozen-FM + lightweight-head thesis over heavy
VLAs, and the basis for our transformer planning head.* OccNet [OpenDriveLab] — occupancy as a unified
representation feeding planning (our occ→planning protocol lineage; ST-P3/UniAD metrics).

## 3. Method — the controlled study
**Backbone (frozen probe).** A frozen FM emits dense patch tokens → an LSS depth-supervised lift →
occupancy volume; only the lightweight lift/decoder/heads train. Backbones: **DINOv2-S/B/L** (patch-14),
**RADIO v2.5-b** (agglomerative, patch-16→interp), **VGGT** (geometry FM, cached), **SigLIP2-base** (VL
FM, patch-16, fixed-224→interp; aspect caveat), **DINOv3** (pending gated access). ResNet18 dropped.
**Phase 2 (light finetune).** Unfreeze the FM at a low LR (1e-5) + more data.
**Tasks / heads (all lightweight, single-GPU).** Occupancy (Occ3D CE, mIoU); detection (center head,
official nuScenes mAP/NDS); **planning** — `PlanHeadBC`: dense-token transformer over occ-BEV tokens +
command + query readout, block-causal interface for temporal windows (Patch-Policy-style), L2@1/2/3s +
collision. **Fusion (Dir 2 / FusionOcc-style):** LiDAR-voxel + camera-BEV → 3D occ encoder + head on the
Occ3D grid.
**Budget.** One H100, finetune-only (no from-scratch backbones, no VLA/RL training).

## 4. Results

**At a glance** (ordered by what the paper claims, not by section number):

| # | claim | evidence | § |
|---|---|---|---|
| 1 | **Occupancy pretraining buys detection label-efficiency** | 0.121 → **0.163** mAP @2k labels (+35%); holds @4k/8k | §4.3 |
| 2 | …and the gain is **rare-object transfer**, not a uniform lift | pedestrian/cone/barrier **1.5–3×**; `car` a wash | §4.3 |
| 3 | **Occ mIoU ≠ transferability** | occ mIoU saturates @2044; det transfer keeps scaling | §4.2 |
| 4 | Camera FMs: **plain DINOv2 wins**; geometry/agglomerative lose | VGGT backbone **−33%**; RADIO < DINOv2 | §4.1 |
| 5 | Label-free pretexts are **null-to-negative** | voxel-soft null; DynamicOcc **worse** w/ +60% better labels | §4.3, §4.4 |
| 6 | **Point FM** (teacher-free) does not clear the ceiling | Sonata **0.299 → 0.285** with *more* capacity | §4.7 |
| 7 | **Depth FM's** apparent +38% is a **LiDAR side-channel** | 32/8/**0** rings → +0.070 / +0.060 / **−0.010** | §4.7 |
| 8 | Supervised ceiling **reproduced exactly** | FlashOcc **0.3809** vs published 0.3784 | §4.5 |

Rows 1–2 are the paper. Rows 4–7 are why row 1 is non-obvious: every competing route to the same
gain was run under this harness and none of them worked.

### 4.1 Frozen-FM backbone ranking (nuScenes, @2044 frames)
| backbone | occ mIoU | det mAP@2k | note |
|---|---|---|---|
| **DINOv2-large** | **0.316** | **0.114** | best on both |
| DINOv2-base | 0.288 | 0.106 | |
| RADIO (agglomerative) | 0.274 | 0.096 | < DINOv2 (non-obvious) |
| VGGT (geometry FM) | 0.218 | (deferred) | weakest semantic occ |
| SigLIP2 (VL FM) | (running) | (running) | aspect-distortion caveat |
| *lss_occ_full (DINOv2, 28k ref)* | 0.302 | 0.163 | full-data reference |
| *FlashOcc-4D-stereo (supervised)* | **0.3809** | — | ceiling, reproduced |

Occ and det rankings **agree**; larger DINOv2 helps both; DINOv2-L @2044 beats the 28k reference on occ.

### 4.2 Occ mIoU ≠ detection transferability (the key decoupling)
Same trainer (`train_lss`), Occ3D-GT labels: occ mIoU **saturates by 2044** (0.288 vs 28k 0.302), but
**detection transfer scales with pretraining data** (DINOv2-base @2044 det 0.106 → full-data 0.163).
[Data-scale @16k point completing.]

### 4.3 Occupancy → detection label-efficiency (official mAP)
| pretext @budget | 2k | 4k | 8k |
|---|---|---|---|
| **occ3d-GT (manual)** | **0.163** | **0.183** | **0.197** |
| from-scratch (DINOv2) | 0.121 | 0.153 | 0.177 |
| voxel-soft (label-free) | 0.115 | 0.140 | — |
| DynamicOcc (label-free, dynamic-sep) | 0.087 | — | — |

Occ pretraining is label-efficient for detection **with good labels**; every label-free pretext is
null-to-negative. DynamicOcc has **+60% better foreground pseudo-label agreement** yet **worse** transfer
— *better labels ≠ better transfer*; the pretext must teach camera-inferable structure.

### 4.4 Fair 2×2 occ (label-free teacher representation)
{voxel, aniso-Gaussian} × {hard, soft-FM}: **voxel-soft best (mIoU 0.104)**; Gaussian earns no advantage
once an occupancy/semantic-coupling confound is removed (audit + factorized-loss rescue).

### 4.5 Supervised ceiling reproduced
FlashOcc-4D-stereo ported to modern torch (H100; original torch-1.10 can't run) → full-val **mIoU
0.3809 vs published 0.3784** (exact; found+fixed a BGR-normalization bug worth ~0.04).

### 4.6 Planning / fusion [in progress]
Lightweight transformer planning head (5.4M params) → L2@1/2/3s + collision on frozen backbones;
FusionOcc-style fusion occ head on our BEVFusion (NDS 0.688), target ~FusionOcc 0.566.

### 4.7 The two modality arms, measured (point-FM and depth-FM)

Local-scale controlled runs — 400 frames / 12 epochs / DINOv2-B / **single seed**, one RTX 3090:
a scaled-down §4.1 harness, so read the **deltas**, not the absolute mIoU.

**Depth-FM arm (LDCM).** A ladder over *how much LiDAR the prior ingests*:

| arm | depth prior | rings the prior sees | occ mIoU | Δ vs b0 |
|---|---|---|---|---|
| b0 | none | — | 0.183 | — |
| b3 | MoGe alone (up-to-scale) | **0** | 0.173 | −0.010 (wash) |
| b2 | LDCM | 8 | 0.243 | **+0.060** |
| b1 | LDCM | 32 | 0.253 | **+0.070** |

Prior accuracy on LiDAR returns **held out from the 8-ring input** (raw = metric grounding;
median-aligned = shape only, the fair protocol for an up-to-scale prior):

| prior | AbsRel raw | δ<1.25 raw | AbsRel aligned | δ<1.25 aligned |
|---|---|---|---|---|
| LDCM-32 | 0.069 | 0.923 | 0.088 | 0.934 |
| LDCM-8 | 0.099 | 0.885 | 0.114 | 0.891 |
| MoGe alone | 0.905 | **0.0002** | 0.329 | 0.456 |

![LDCM depth completion](ldcm_depth_completion.png)

The figure *is* the mechanism: MoGe alone (c) recovers plausible **structure** but no metric scale
(raw δ<1.25 = 0.0002); the metric surface appears only once the Poisson step ingests the sparse
returns (d, e).

![beam ladder](arms_beam_ladder.png)

**This corrects our own hypothesis.** We predicted LDCM would help *because* it is metric and
sparse-conditioned — i.e. succeed where the VGGT prior was a wash. It does help (+38%), but the
ladder shows the gain is **monotone in the LiDAR the prior ingests and vanishes at zero rings**:
with no sparse depth (b3) a strong monocular FM lands in exactly the same place as VGGT — a wash.
So the +38% is a **LiDAR side-channel, not better monocular reasoning**, and a "camera-only" model
carrying a sparse-depth-conditioned prior is *not camera-only*. The MoGe raw δ<1.25 = 0.0002
confirms the mechanism directly: the monocular prior is not metric at all; metric grounding is
injected by the Poisson step that consumes the sparse LiDAR.

**What survives is a different, practical result:** a depth-completion FM is an *efficient
channel* for routing sparse LiDAR into a camera lift — **8 rings recover 86% of the 32-ring gain**
(+0.060 of +0.070), roughly half what a dedicated 32-beam fusion branch buys (a0, +0.116).
That is a real option for low-beam / cheap-LiDAR rigs, and it is a claim about *plumbing*, not
about foundation-model perception.

**Point-FM arm (Sonata).** Frozen Sonata (PTv3, 108M) features, PCA-16, appended to the LiDAR
branch's 3 raw geometry channels:

| arm | LiDAR-branch input | occ mIoU | Δ |
|---|---|---|---|
| a0 | 3 raw geometry channels | **0.299** | — |
| a1 | + frozen Sonata (16 ch) | 0.285 | **−0.014** |

A clean negative: the FM arm has **more** capacity (19 vs 3 channels) and still loses, so capacity
is not the explanation. It is consistent with the domain gap — Sonata is pretrained on **indoor**
scans with `coord + colour`, while nuScenes LiDAR is outdoor and colourless (we feed intensity as
grey), and Sonata's published outdoor numbers come from **fine-tuning**, not frozen transfer. It
also answers the question the arm was built to decide: the label-free ceiling of §4.3 is **not
merely a teacher-quality bound** — a *teacher-free* point FM does not clear it either when
transferred out of domain.

*Caveat: single seed at reduced scale; the ±0.01 differences (b3, a1) are within run-to-run noise
and should be read as "no gain", not as a measured loss.*

## 5. Discussion / findings
- **Backbone capacity > backbone "type".** Larger DINOv2 wins; agglomerative (RADIO) and geometry (VGGT)
  FMs *underperform* plain DINOv2 for semantic occ+det — a caution against assuming "more teachers /
  more geometry = better features."
- **Occ mIoU is the wrong proxy for a detection/planning pretext.** Report transfer, not occ mIoU.
- **Label-free occ pretraining is bounded** by teacher quality *and* pretraining scale; naive label
  improvements can hurt. The lever is dense, camera-inferable, high-quality labels at scale.
- **Frozen FM + lightweight head is the resource-right paradigm** (Patch Policy corroborates over VLAs).
- **Gaussians belong in prediction/reconstruction, not labels** (VGOcc/ADGaussian corroborate).
- **Depth-FM priors do not beat the VGGT null — they smuggle in LiDAR.** The strongest available
  depth FM (LDCM: metric, sparse-conditioned, ICLR'26) helps the lift by +38% *only* in proportion
  to the LiDAR it ingests, and reproduces the VGGT wash at zero rings (§4.7). The lesson
  generalises: when a "prior" consumes a sensor the baseline lacks, attribute the gain to the
  sensor until a zero-sensor rung says otherwise. Our own draft claimed the opposite before we ran
  the ladder.
- **Both new modality axes land on the same conclusion as the camera axis.** A point FM (Sonata)
  and a depth FM (LDCM/MoGe) each fail to beat a plain, well-supervised baseline on *its own
  merits* — mirroring RADIO and VGGT losing to plain DINOv2. Across camera, point and depth FMs,
  **capacity and modality-specialisation lose to in-domain semantic capacity plus supervision.**
- **The "geometric shortcut" names our geometry-FM negative.** Sonata [25] identifies, in 3D SSL, a
  collapse onto low-level spatial cues instead of semantics. Our camera-side result rhymes: VGGT (and,
  more mildly, RADIO) lose to plain DINOv2 on *semantic* occupancy and its transfer. Geometry capacity
  is not the binding constraint here — semantic capacity is.
- **Segmentation-style metrics keep failing as transfer proxies.** Our occ-mIoU decoupling (§4.2) and
  Sonata's outdoor tables (large semseg gains, *no* detection numbers) point the same way: a strong
  dense-prediction score does not license a claim about detection transfer. Report transfer.

## 6. Limitations / future
Small label budgets (finetune-only); single dataset (nuScenes); SigLIP2 aspect distortion; DINOv3 gated;
fusion column and full multi-task (occ+det+planning) tables completing; cross-dataset (Waymo/AV2/
PhysicalAI) and pose/ray-aware feature fusion (VGOcc/ADGaussian) as next steps.

**Modality asymmetry.** The camera-side benchmark is now complemented by a **point-FM** and a
**depth-FM** arm (both run — §4.7), but only at reduced local scale and a single seed; re-running
them at the §4.1 budget (2044 frames / 24 epochs, multi-seed) and pushing them through the
*detection-transfer* protocol is the remaining work. The two arms were posed as:
- **(a) point-FM arm** — frozen **Sonata** [25] as the LiDAR branch: *answered* — no gain out of
  domain, and the label-free ceiling is not merely teacher-bounded (§4.7). Still open: an
  **in-domain** point FM (outdoor-pretrained), and the detection-transfer curve.
- **(b) depth-FM arm** — **LDCM** [26] as the lift's depth prior: *answered* — the gain tracks the
  ingested LiDAR, not the FM (§4.7). Still open: its **intrinsic-free point map** as the shared
  cross-rig representation for the Waymo/AV2/PhysicalAI pool, which the ladder does not touch.

## References
[1] Tian et al. **Occ3D**: A Large-Scale 3D Occupancy Prediction Benchmark. CVPR 2023. (Tsinghua-MARS-Lab)
[2] **CVT-Occ** — Cost-Volume Temporal occupancy (Tsinghua-MARS-Lab followup).
[3] Zhang et al. **TT-Occ**: Test-Time 3D Occupancy Prediction. CVPR 2026. arXiv:2503.08485.
[4] Zhai et al. **OnlinePG**: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting. CVPR 2026. arXiv:2603.18510.
[5] Pan et al. **PanDA**: Unsupervised Domain Adaptation for Multimodal 3D Panoptic Segmentation. CVPR 2026. arXiv:2604.19379.
[6] Ding et al. **ExtrinSplat**: Decoupling Geometry and Semantics for Open-Vocabulary 3DGS. arXiv:2509.22225.
[7] Wang et al. **SPAN**: Spatial-Projection Alignment for Monocular 3D Object Detection. CVPR 2026. arXiv:2511.06702.
[8] Long et al. **IDESplat**: Iterative Depth Probability Estimation for Generalizable 3DGS. CVPR 2026. arXiv:2601.03824.
[9] Lin et al. **VGOcc**: Learning Visual-Geometric Gaussians for Vision-Centric 3D Occupancy. arXiv:2607.18078.
[10] **ADGaussian**: Generalizable Gaussian Splatting for Autonomous Driving with Multi-modal Inputs.
[11] Zhang et al. **FusionOcc**: Multi-Modal Fusion for 3D Occupancy Prediction. ACM MM 2024. (56.6 mIoU)
[12] Zhou, Cui, Langford, Tan, **LeCun**, Pinto. **Patch Policy**: Efficient Embodied Control via Dense Visual Representations. arXiv:2607.18236.
[13] **AutoVLA**: A VLA Model for End-to-End Autonomous Driving with Adaptive Reasoning and RFT.
[14] **SimScale** (OpenDriveLab): sim-real scaling for end-to-end planning (NAVSIM).
[15] **OccNet** (OpenDriveLab): 3D Occupancy as a general representation. CVPR 2023 challenge.
[16] **FlashOcc**: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height.
[17] **BEVFusion**: Multi-Task Multi-Sensor Fusion with Unified BEV Representation.
[18] **GaussianOcc**: Fully Self-supervised 3D Occupancy Estimation via Gaussian Splatting. ICCV 2025.
[19] Oquab et al. **DINOv2**; **DINOv3** (Meta). [20] Ranzinger et al. **AM-RADIO/RADIO** (NVIDIA).
[21] **VGGT**: Visual Geometry Grounded Transformer. [22] **SigLIP 2** (Google). [23] **V-JEPA 2** (Meta).
[24] **ST-P3 / UniAD** — open-loop planning protocols.
[25] Wu, DeTone, Frost, Shen, Xie, Yang, Engel, Newcombe, Zhao, Straub. **Sonata**: Self-Supervised Learning of Reliable Point Representations. CVPR 2025. arXiv:2503.16429. (Pointcept × Meta)
[26] Yu, Zhao, Zhang, Qiu, Qiu, He, Zhu, Dong, Cao, Shen. **LDCM**: Large Depth Completion Model from Sparse Observations. ICLR 2026. arXiv:2605.30115. (Zhejiang Univ. × Tongyi Lab, Alibaba)

*Result values current as of the run log; Phase-2 finetune, data-scale, SigLIP2, fusion, and planning
numbers finalize as those jobs complete.*
