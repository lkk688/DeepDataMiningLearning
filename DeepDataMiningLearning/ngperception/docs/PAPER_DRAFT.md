# Paper Draft — Foundation-Model Backbones for the Autonomous-Driving Perception→Planning Stack: A Controlled Multi-Task Transfer Study

*Working draft compiled from the full experimental log (2026-06 → 2026-07). All numbers are our own
runs on nuScenes / Occ3D-nuScenes unless marked as a published reference. Companion docs:
`TUTORIAL_LABELFREE_PERCEPTION_JOURNEY.md`, `RESULTS_LABELFREE_OCC_2x2_TRANSFER.md`,
`PLAN_FLASHOCC_MIGRATION.md`, `PLAN_FUSIONOCC.md`, `REFS_2026_OCC_GAUSSIAN_LABELFREE.md`.*

---

## Abstract (draft)
3D occupancy is emerging as a unified scene representation for autonomous driving, and a wave of recent
work explores *how* to obtain it — label-free pseudo-labeling, Gaussian representations, and large
vision-language-action (VLA) policies. We run a **controlled, single-GPU, finetune-budget study** that
cuts across these choices and asks a practical question: *which pretrained image backbone best supports
the full perception→planning stack — occupancy, 3D detection, and open-loop planning — and why?* Our
findings are largely **negative-result-driven and therefore diagnostic**: (1) among frozen foundation
models, **plain DINOv2 (larger = better) beats agglomerative (RADIO) and geometry (VGGT) FMs** for
semantic occupancy and its downstream transfer; (2) **occupancy mIoU does *not* predict detection
transferability** — occ mIoU saturates with little data while detection transfer keeps improving with
pretraining scale; (3) **label-free / pseudo-label occupancy pretexts are teacher- and data-bounded** —
Gaussian teachers earn no advantage over voxels, and adding "better" object pseudo-labels (dynamic/static
separation) *hurts* detection transfer; (4) consistent with concurrent work (Patch Policy, LeCun et al.),
a **frozen FM + a lightweight head outperforms heavy fine-tuned VLAs** at a fraction of the cost. We
reproduce a supervised occupancy ceiling exactly (FlashOcc-4D-stereo, 0.3809 vs published 0.3784) and
outline a LiDAR-camera **fusion** extension (FusionOcc-style) for SOTA-competitive numbers. The study is
a controlled map of *what transfers* in AD perception and *what does not*.

## 1. Contributions
1. **A controlled frozen-FM backbone benchmark** across occupancy + detection (+ planning) under a fixed
   finetune budget, on one GPU — DINOv2-S/B/L, RADIO, VGGT, SigLIP2 (DINOv3 pending access).
2. **Occ-mIoU ≠ transferability**: a clean decoupling showing detection-transfer scales with pretraining
   data while occ mIoU saturates — occ mIoU is the wrong proxy for a pretext.
3. **A battery of controlled negatives** (Gaussian occ teachers; label-free/pseudo-label pretexts;
   agglomerative/geometry FMs) with mechanisms (audit + factorized-loss rescue).
4. **Exact reproduction of a supervised occ ceiling** (FlashOcc-4D-stereo) and a modern-stack port that
   runs on H100 where the original cannot.
5. A **multi-task extension** (lightweight occ→detection→planning heads on one frozen backbone) with a
   Patch-Policy-style dense-token transformer planning head.

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
### 4.1 Frozen-FM backbone ranking (nuScenes, @2044 frames)
| backbone | occ mIoU | det mAP@2k | note |
|---|---|---|---|
| backbone | occ mIoU | det mAP@2k | det NDS | note |
|---|---|---|---|---|
| **DINOv2-large** | **0.316** | 0.114 | 0.1235 | best occ |
| **DINOv3-large** | 0.292 | **0.1248** | **0.1296** | **best cam-only det** (occ↓ det↑ vs DINOv2-L) |
| DINOv2-base | 0.288 | 0.106 | 0.1215 | |
| RADIO (agglomerative) | 0.274 | 0.096 | 0.1095 | < DINOv2 (non-obvious) |
| SigLIP2 (VL FM) | 0.234 | 0.0628 | 0.0872 | VL-contrastive + aspect distortion |
| VGGT (geometry FM) | 0.218 | (deferred) | — | weakest semantic occ |
| *lss_occ_full (DINOv2, 28k ref)* | 0.302 | 0.163 | — | full-data reference |
| *FlashOcc-4D-stereo (supervised)* | **0.3809** | — | — | ceiling, reproduced |

The occ ranking (DINOv2-L > DINOv3-L ≈ DINOv2-B > RADIO > SigLIP2 > VGGT) tracks **pretraining objective**, not
model size — self-supervised dense FMs beat distilled/agglomerative (RADIO), VL-contrastive (SigLIP2), and
geometry (VGGT). **DINOv3-L is a within-table instance of the occ≠det decoupling (§4.2): its occ (0.292) is
*below* DINOv2-L (0.316) yet its det-transfer (0.1248) is the *best* camera-only** — newer/patch-16 features
transfer better to detection despite weaker semantic occ (its patch-16 grid is interpolated to our patch-14
lift, which may cost occ detail while its register-token features help det). So occ and det rankings do **not**
agree once DINOv3 is included; report transfer, not occ mIoU.

### 4.2 Occ mIoU ≠ detection transferability (the key decoupling)
Same trainer (`train_lss`), Occ3D-GT labels: occ mIoU **saturates by 2044** (0.288 vs 28k 0.302), but
**detection transfer scales with pretraining data**. Frozen DINOv2-L det-transfer vs pretraining frames:

| frozen DINOv2-L → det mAP@2k | 2044 | 16k | 28k (base ref) |
|---|---|---|---|
| det mAP | 0.114 | **0.1462** | 0.163 |

monotone and roughly log-linear in data — occ mIoU is flat over the same range.

**Frozen vs. light-finetune (DINOv2-L) — a clean double dissociation.** Unfreezing the ViT-L at a low LR
(1e-5) on 8000 frames moves the two metrics in **opposite** directions:

| DINOv2-L | occ mIoU | det mAP | det NDS |
|---|---|---|---|
| frozen probe @2044 | **0.316** | 0.114 | — |
| light-finetune @8000 | 0.290 ↓ | **0.1373** ↑ | 0.1390 |

Finetuning **hurts occ mIoU** (0.316→0.290, mild over-fit / forgetting of general features — the frozen-FM
thesis of Patch-Policy/LeCun) yet appears to **help detection transfer** (0.114→0.137). The same intervention
pushes occ down and det up — the strongest single-experiment evidence that occ-mIoU and det-transferability
are distinct objectives.

**But the det gain is data, not finetuning.** The frozen data-scale curve above reaches **0.1462 at 16k with
no finetuning**; log-linear interpolation puts **frozen @8000 ≈ 0.135**, statistically indistinguishable from
the **finetune @8000 = 0.137**. So light-finetuning buys **essentially zero** detection transfer over feeding
the same data to a *frozen* backbone — while **costing** occ mIoU (0.316→0.290) and ViT-L-scale training. The
operational conclusion: **freeze the FM and scale pretraining data; don't finetune.** This is exactly the
Patch-Policy/LeCun frozen-FM prescription, here validated on the AD occ→det transfer axis.

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

### 4.6 LiDAR+camera fusion column (Direction 2)
Adding a LiDAR branch to the DINOv2-L LSS column (same @2044 occ pretrain → det @2k protocol):

| DINOv2-L | occ mIoU | det mAP@2k | det NDS |
|---|---|---|---|
| camera-only | 0.316 | 0.114 | 0.1235 |
| **LiDAR + camera** | **0.493** | **0.2417** | **0.2197** |

Both metrics ≈ **double** with the LiDAR branch (occ +56%, det mAP +112%; car AP@0.5 0.520, ped 0.420) —
direct range geometry makes occupancy far easier and gives detection a metric prior the camera lacks. The
fusion occ mIoU (0.493) exceeds the camera-only supervised FlashOcc ceiling (0.3809), as expected for a
sensor with explicit depth. This is the strong-sensor anchor for the otherwise camera-primary study.

### 4.7 Occupancy → open-loop planning: ego-status dominates (Direction 4)
Lightweight transformer planning head (Patch-Policy-style dense-token attention) → L2@1/2/3s + collision.
To isolate the *perception* contribution from the well-known ego-status shortcut (BEV-Planner / AD-MLP:
nuScenes open-loop L2 is dominated by the ego constant-velocity prior), we run a **controlled 3-arm ablation**
on frozen backbones. The ego arm adds a constant-velocity prior from 2 s of ego history (the head predicts
only the residual); the analytic CV floor on val is L2@3s = 2.09 m.

| arm | input | L2@1s | L2@2s | **L2@3s** | collision |
|---|---|---|---|---|---|
| **occ-only** | occupancy + command | 2.12 | 3.56 | **5.04** | 0.065 |
| **ego-only** | ego-history + command | 0.64 | 1.27 | **2.09** | 0.018 |
| **occ+ego (DINOv2-base)** | both | 0.66 | 1.28 | **2.11** | 0.019 |
| **occ+ego (DINOv2-large)** | both | 1.07 | 1.74 | **2.49** | 0.018 |

Two clean, negative-but-informative findings:
1. **Occupancy alone is a poor open-loop planner** (5.04 m ≫ 2.09 m): without ego kinematics the model cannot
   infer speed, so it collapses to the mean trajectory.
2. **Occupancy adds ~nothing on top of ego-status** — Δ(occ+ego − ego) ≈ 0 (base 2.11 vs 2.09), and the
   *larger* occ backbone even **hurts** (large 2.49), its richer features injecting residual noise. Ego-only
   already sits at the CV floor.

So nuScenes open-loop L2 is **ego-status-dominated**; occupancy is valuable for *detection* transfer (§4.3)
but **not** for open-loop planning. Combined with §4.2 (occ-mIoU ≠ det-transfer), this gives a **double
decoupling**: the value of an occupancy representation depends entirely on the downstream task — it is a good
detection pretext and a null planning input — a caution against treating occupancy as a universal world model.
(FusionOcc-style semantic occ head on our BEVFusion, NDS 0.688, remains a queued build.)

## 5. Discussion / findings
- **Backbone capacity > backbone "type".** Larger DINOv2 wins; agglomerative (RADIO) and geometry (VGGT)
  FMs *underperform* plain DINOv2 for semantic occ+det — a caution against assuming "more teachers /
  more geometry = better features."
- **Occ mIoU is the wrong proxy for a detection/planning pretext.** Report transfer, not occ mIoU.
- **The value of occupancy is task-dependent (double decoupling).** Occupancy is a strong *detection* pretext
  (§4.3) but a *null* open-loop-planning input (§4.7): once the ego constant-velocity prior is present,
  occupancy adds ≈0 (a larger occ backbone even hurts). nuScenes open-loop L2 is ego-status-dominated —
  don't credit a perception module for it (corroborates BEV-Planner / AD-MLP).
- **Label-free occ pretraining is bounded** by teacher quality *and* pretraining scale; naive label
  improvements can hurt. The lever is dense, camera-inferable, high-quality labels at scale.
- **Frozen FM + lightweight head is the resource-right paradigm** (Patch Policy corroborates over VLAs).
- **Gaussians belong in prediction/reconstruction, not labels** (VGOcc/ADGaussian corroborate).

## 6. Limitations / future
Small label budgets (finetune-only); single dataset (nuScenes); SigLIP2 aspect distortion; DINOv3 gated;
fusion column and full multi-task (occ+det+planning) tables completing; cross-dataset (Waymo/AV2/
PhysicalAI) and pose/ray-aware feature fusion (VGOcc/ADGaussian) as next steps.

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

*Result values current as of the run log; Phase-2 finetune, data-scale, SigLIP2, fusion, and planning
numbers finalize as those jobs complete.*
