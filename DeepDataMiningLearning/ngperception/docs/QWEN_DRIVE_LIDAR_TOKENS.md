# Qwen-Drive-1.0 as the host for LiDAR-in-a-VLA

**Status:** **complete, as a negative result with an instrument.** Gradient gate passed
after fixing a non-differentiable released head; the forward made **91× faster** by replacing
one pathological `Conv3d`; the training objective diagnosed and fixed (+0.12 fwIoU); and then
the decisive control showed that **the entire gain is a learned constant prefix that
recalibrates the frozen head — the injected LiDAR contributes nothing measurable**, on
held-out frames as well as trained ones. Do not tune this further; write it up. A 64-token projector at a frozen VLM's input does reach the
BEV head (0.585 % of voxels change class, +0.0044 mIoU on the frame it trains on), but
**57.5 % of its loss gain survives being fed the wrong point cloud**, i.e. is a
frame-independent constant. No claim about LiDAR helping is supported yet. This document records what the model is, exactly how to run it here, what
does and does not backpropagate, and what the next student should and should not attempt.

**Read [`AUDIT_3D_IN_DRIVING_VLA.md`](AUDIT_3D_IN_DRIVING_VLA.md) first** — it is the synthesis
(all results, the six-mechanism survey of how 3D actually enters driving models, the 6/6
missing-control audit, and the paper's claim). This file is the engineering record behind it.

**Companion documents.** [`RESEARCH_DIRECTIONS.md`](../../../closedloop_occ/RESEARCH_DIRECTIONS.md) (private, gitignored) is the index of
what is go and no-go across the whole project; this file is one direction's engineering
record and §4.7 there is its index entry. Also relevant: §2.6 (the fusion-robustness probe
and the two-ablation audit — the instrument this direction inherits), §4.6
("visibility-conditioned fusion") and §7.8–7.11 (rules of practice).

**Code:** `DeepDataMiningLearning/bevdet/lidar_tokens/`

| file | what it is |
|---|---|
| `qwen_drive_grad.py` | the patch that makes the released head differentiable, plus `audit_ops()` |
| `check_grad.py` | the gate. `--stage ops\|parity\|profile\|llm\|head\|full` |
| `train_projector.py` | the projector + the 4-arm overfit gate (config-driven; writes `config.yaml` + `meta.json` + `log.json` per run) |
| `run_projector_gate.sh` | runs all four arms in order, resumable; **its header holds the pre-registered reading** |
| `configs/gate_demo6.yaml` | the gate's configuration |
| `extract.py` | dumps a frozen BEVFusion's TransFusion queries (the eventual LiDAR tokeniser) |

---

## 0. The idea in one paragraph, and why this model

There is no web-scale corpus of point clouds, so a LiDAR encoder trained from scratch
inside a VLA is data-starved next to a vision tower that has seen billions of images. The
cheap way out is not to learn to read point clouds at all: run a **frozen, already-trained
LiDAR detector**, and train only a small **projector** that turns its object queries into
tokens in the language model's input sequence. The problem becomes "describe what the
detector found", which is a language-shaped problem, on a frozen backbone, on one GPU.

Qwen-Drive-1.0 is the natural host because it already has the two things such an experiment
needs and that almost nothing else open has together: a strong native-multimodal VLM
(Qwen3.5-4B) **and** a real 3D perception head on top of it (detection + occupancy + map
segmentation), so an injected token's effect can be read out as a *3D* number rather than as
a text score. It also gives us a camera-only 3D baseline for free.

---

## 1. What the model actually is

Two separately released pieces, joined at run time by `model.attach(vlm, processor)`:

```
Qwen-Drive-1.0-4B/              the VLM (Qwen3_5ForConditionalGeneration + vision_config)
Qwen-Drive-1.0-4B/perception/   the BEV head (config.json + model.safetensors)
```

The head reads the VLM at **two** taps, and the distinction is load-bearing for everything
below:

| stream | tapped where | shape (8-camera nuPlan frame) | width |
|---|---|---|---|
| **LLM stream** (main) | decoder output at image-token positions, after `language_model.norm` | `(8, 16, 28, 2560)` | `llm_dim` 2560 |
| **ViT stream** | vision tower's **pre-merge** patches, after `visual.merger.norm` | `(8, 32, 56, 1024)` | `vit_dim` 1024 |

* LLM stream → `adaptor` (FPN, 4 scales) → BEVFormer encoder (temporal self-attn +
  spatial cross-attn, 6 layers, BEV 200×200) → `bev_embed` → detection decoder (6 layers,
  900 queries) / occupancy decoder / map decoder.
* ViT stream → `vit_neck` → `DepthNet` → `Uni3DVoxelPoolDepth` (an LSS-style lift) →
  added into the BEV queries and fused into the occupancy volume.

**The consequence that makes the whole plan clean:** the ViT runs *before* the language
model. Tokens injected at the LLM input therefore cannot change the ViT stream **at all**.
They can only act by changing the image tokens' representations through attention inside
the decoder. There is no plumbing shortcut by which an injected token reaches the BEV head
directly — any downstream change is genuine propagation through the frozen 4 B decoder.

### Label spaces (needed before any comparison with our stack)

| task | classes | grid |
|---|---|---|
| detection | 7 (`DET_CLASS_NAMES`) | 900 queries, 300 kept, `det_pc_range` ±51.2 m |
| occupancy | 10 = 7 det + `driveable`, `background`, `empty` (**index 9 = free**) | 200×200×16; nuScenes ±40 m @ 0.4 m, nuPlan ±50 m @ 0.5 m |
| map seg | 6 | 200×400 @ 0.15 m, ±30 × ±15 m |

Occ3D-nuScenes, which every other experiment in this project uses, has **18** classes and
`FREE = 17`. **Any comparison against our own occupancy models needs an explicit 18→10
mapping, and the mapping is lossy.** This is the main cost of task (b) (the headroom
measurement) and it has not been done.

---

## 2. Running it here — the reproducible recipe

Nothing needed installing. Five obstacles, all environmental:

```bash
cd /data/rnd-liu/Others/Qwen-Drive-1.0        # the --frames path below is repo-relative

# 1. the env: torch 2.9.0+cu126, transformers 5.12.1
PY=/home/010796032/miniconda3/envs/py310/bin/python

# 2. the ops JIT-compile on first use and need a matching CUDA toolkit
export CUDA_HOME=/data/rnd-liu/cudatk
# 3. ninja must be on PATH (it is already in the env's bin, but only if PATH includes it)
export PATH=/home/010796032/miniconda3/envs/py310/bin:$PATH
# 4. nvcc needs cuda_runtime_api.h, which lives in a targets/ subdir in this toolkit
export CPATH=$CUDA_HOME/targets/x86_64-linux/include
# 5. flash-attn is not installed -> the default attn_implementation fails
$PY scripts/run_perception.py --vlm /data/rnd-liu/Others/Qwen-Drive-1.0-4B \
    --model /data/rnd-liu/Others/Qwen-Drive-1.0-4B/perception \
    --attn-implementation sdpa --frames data/demo/perception --output /tmp/qd
```

Compiled extensions cache under `~/.cache/torch_extensions`, so only the first run pays the
couple of minutes. `src/` is added to `sys.path` by the scripts themselves; the package is
not installed, and `import qwen_drive` is what registers the custom `model_type` with
transformers — importing `qwen_drive_perception` alone is not enough.

**The demo set is 6 frames**: 4 nuPlan (8 cameras) and 2 nuScenes (6 cameras). Each frame
directory packs images, calibration, a LiDAR point cloud (`lidar.npy`, ~94 k points) and
**ground truth** — `occ` `(200,200,16) int8`, `map` `(200,400) int8`, `boxes` `(N,9)`. That
GT is why a gate is possible at all without building a dataset loader first. Six frames is
also the ceiling on what can be concluded (rule 7.10: consecutive frames are not samples,
and 6 packed frames are 6 scenes at best).

---

## 3. The gradient gate — what backpropagates and what does not

The plan needs gradient to travel: **occupancy loss → BEV head → tapped image tokens → 36
frozen decoder layers → input embeddings → projector.** The release is inference-only
(`QwenDrivePerception.infer` is `@torch.no_grad()`), so this had to be checked before any
projector was written. It was checked in halves, because the composed path OOMed at 57–60
GiB on the first attempt.

### 3.1 The LLM half — PASS as released

`check_grad.py --stage llm`

```
embedding-output grad: |g|_1 1845   max 0.365   finite True   nonzero_rows 256/256
VERDICT[llm]: PASS -- gradient crosses the frozen decoder to the input embeddings
peak 8.95 GiB
```

Two memory findings, both decision-relevant for training cost, not just for this test:

1. `output_hidden_states=True` — what the released `infer` uses — materialises all 36
   layers' hidden states, and with a graph attached they are all *retained*. We only ever
   read the last one, and `vlm.model(...)`'s `last_hidden_state` is that same tensor
   without the other 35.
2. **`gradient_checkpointing_enable()` is silently a no-op under `.eval()`.** HF gates it
   on `self.gradient_checkpointing and self.training`. The first two attempts printed
   "enabled" and saved 3.5 of 60 GiB for exactly this reason. Calling `vlm.train()` fixed
   it (60.5 → 8.95 GiB). The parameters are frozen either way; `train()` here only
   re-enables checkpointing, and Qwen3.5 has no dropout by default.

### 3.2 The head half — FAIL as released, and the reason is in the CUDA ops

`check_grad.py --stage head --attn kernel`

```
loss tensor: shape (1, 200, 200, 16, 10) dtype torch.bfloat16 grad_fn None
FAIL: the BEV head output carries no grad_fn -- the graph was cut inside the head.
peak 20.81 GiB
```

Every output of `bev_modeling` — `bev_embed`, `all_cls_scores`, `all_bbox_preds`,
`occ_pred`, `seg_preds` — came back `requires_grad=False, grad_fn=None`, despite grad mode
being on and `img_llm_feats.requires_grad=True`. The submodules on the way in (`adaptor`,
`vit_neck`) *did* keep the graph (`grad_fn=AddBackward0`), which localised the break to the
encoder.

`check_grad.py --stage ops` (no GPU needed) names it:

```json
{"ms_deform_attn":    {"is_autograd_function": false, "raw_extension_call": true,
                       "verdict": "cuts the graph (no autograd.Function)"},
 "voxel_pool_depth":  {"is_autograd_function": true,  "has_backward": false,
                       "verdict": "autograd.Function with no backward -> raises if reached"}}
```

`ops/__init__.py:ms_deform_attn_bf16_forward` calls the JIT extension's `.forward()`
**directly**, with no `torch.autograd.Function` wrapper. Its output is a fresh tensor with
no `grad_fn`, so **every deformable-attention call in the BEV head silently severs the
graph** — and the encoder is deformable attention all the way. `_VoxelPoolDepthCuda` is an
`autograd.Function` that defines only `forward`.

This is a real property of the release, not a bug in our code. It is also not documented.

### 3.3 The fix is one line, and it is the authors' own code

The release **also ships a pure-torch implementation of the same op** —
`layers.py:multi_scale_deformable_attn_pytorch`, `F.grid_sample`-based, autograd-native —
which the dispatcher already selects when the tensors are not on CUDA
(`attention.py:52-53`). So the fix is not "write a backward kernel"; it is to route the CUDA
path through the fallback the authors already wrote. All three call sites resolve the
dispatcher as a module global, so patching one attribute covers the head:

```python
from qwen_drive_grad import patch_deformable_attention
with patch_deformable_attention(math_dtype=torch.float32):
    outs = model.bev_modeling(img_vit_feats=vit.detach(), img_llm_feats=llm, ...)
    loss.backward()
```

The second op needs no fix **for this use case**, and why is an architectural fact worth
keeping: `voxel_pool_depth` lives on the **ViT stream**, which the injected tokens cannot
influence (§1). Keep that stream detached and the op receives no grad-requiring input,
creates no autograd node, and is never asked to go backward.

`check_grad.py --stage head --attn torch`

```
loss tensor: shape (1, 200, 200, 16, 10) grad_fn ViewBackward0
img_llm_feats grad: |g|_1 1.5462e+05  max 9.9609e-01  finite True
VERDICT[head]: PASS -- the BEV head backpropagates into its image features
peak 32.76 GiB
```

### 3.4 Fidelity of the swap — the control that makes it usable

A differentiable substitute is only useful if it computes the released model's function. The
two paths are not bit-identical (different accumulation order; the kernel is built with
`--use_fast_math`). `check_grad.py --stage parity` runs the head twice on the **real tapped
features** of a real frame, once through each path:

| tensor | shape | max abs diff | rel. L2 |
|---|---|---|---|
| `bev_embed` | (40000, 1, 256) | 3.98e-01 | 1.21e-02 |
| `all_cls_scores` | (6, 1, 900, 7) | 1.00e+00 | 9.91e-03 |
| `all_bbox_preds` | (6, 1, 900, 10) | 4.75e+00 | 1.11e-02 |
| `occ_pred` | (1, 200, 200, 16, 10) | 2.00e+00 | 3.05e-03 |
| `seg_preds` | (1, 6, 200, 400) | 3.75e-01 | 5.74e-03 |

**Occupancy argmax agreement 99.983 %** of 640 000 voxels; map argmax agreement 99.819 %.
Relative L2 of 0.3–1.2 % on bf16 logits, and the decision is the same one. The fallback is
a faithful stand-in.

### 3.5 The cost of differentiability

| configuration | peak GiB (incl. ~8.9 GiB of frozen weights) |
|---|---|
| head forward only, CUDA kernel, `no_grad` | 20.32 |
| head forward only, torch fallback, `no_grad` | 20.35 |
| head forward + backward, CUDA kernel | 20.81 (and no gradient) |
| head forward + backward, torch fallback | **32.76** |

The fallback's forward costs essentially nothing extra; the ~12 GiB is the retained
backward graph (the BEV encoder's `n_query` is `bev_h*bev_w` = 40 000, and the fallback
materialises `[bs·heads, C, n_query, levels·points]` per level per layer). Practical
consequence: **use the fallback for the training forward and the kernel for evaluation.**
`--attn-math bf16` roughly halves the fallback's activation memory if needed.

### 3.6 Verdict on gate (a)

| half | as released | with the patch |
|---|---|---|
| loss → BEV head → tapped image features | ❌ graph cut at every deformable attention | ✅ PASS |
| tapped features → 36 frozen layers → input embeddings | ✅ PASS | ✅ PASS |

The two halves were measured separately because the composed path OOMed on the first
attempt, and two passing halves only *imply* the composition. **The composition is now
confirmed directly**, not by `--stage full` but by the thing that matters: one real
training step in `train_projector.py` — occupancy cross-entropy at the BEV head, backward
through the frozen 4 B decoder, into the projector — gives a non-zero finite projector
gradient (‖g‖ = 2.84e-01) at a peak of 33.54 GiB. §4.

**The projector-at-the-LLM-input route is trainable.** The earlier reading of this
situation — "the BEV head can only be an evaluation instrument, train against our own probe
instead" — was wrong: it assumed the missing backward had to be written. Reading the
release's own fallback removed the need.

---

## 4. The gate that follows: can a projector at the LLM input move the BEV head?

`train_projector.py`. Freeze everything, train only a projector, and ask the *capacity*
question before the science question:

> Can a projector at the LLM input, with the VLM and the BEV head frozen, **overfit 6
> frames**?

A "no" kills the route — the channel from injected tokens to the BEV head is too narrow to
matter, and more data will not widen it. A "yes" says the channel is open. It says **nothing
about whether LiDAR helps**: overfitting 6 frames is what a working optimiser does.

**Four arms**, run by `run_projector_gate.sh`, whose header carries the pre-registered
reading so it cannot be rationalised afterwards:

| arm | what it is | what it isolates |
|---|---|---|
| `none` | the released path, no extra tokens | the number to beat |
| `zero` | K tokens whose embedding is exactly **zero**, no training | the cost of the K extra sequence positions, separately from their *content*. Without it, any change in a trained arm is confounded with "the prompt got K tokens longer" |
| `lidar` | K tokens from **this** frame's points, projector trained | the effect |
| `shuffle` | K tokens from a **different** frame's points, projector trained | **the required control.** With 6 frames and a trainable input, "it learned the frame index" is the default explanation; only this separates it from "it read the LiDAR" |

**PASS** = `lidar` beats `none` on **mIoU** with non-trivial argmax drift, *and* beats
`shuffle` on both. **FAIL-1** = `lidar` cannot beat `none` (the channel does not reach the
BEV head usefully). **FAIL-2** = `lidar` beats `none` but `shuffle` matches it (the
projector is free per-frame parameters, not a reading). **FAIL-3** = the loss falls but
argmax drift is ~0 (calibration, not information — see below).

**Design choices worth keeping.**

* **The tokens must go BEFORE the first image token, and this was learned the hard way.**
  The first version appended them at the end of the prompt, which keeps every image token's
  rope position identical to the released path and looked strictly better for that reason.
  The projector's gradient was then **exactly zero** and the loss bit-identical to the
  baseline: the decoder is **causal**, so a token placed after the images cannot change the
  images' hidden states — and those are the only thing the BEV head reads. They are now
  inserted at the first `<|vision_start|>` (index 8 of a 3718-token prompt), so every
  camera attends to them. The deliberate cost is that every image token's rope position
  shifts by K; the model was trained with variable-length text before the images so this is
  in-distribution, but it is not a no-op — which is exactly what the `zero` arm measures.
* **Injection by embedding hook, not `inputs_embeds=`.** HF's Qwen-VL forward locates image
  positions from `input_ids` and masked-scatters the vision features *after* embedding, so
  passing `inputs_embeds` risks silently dropping the images. Overwriting text positions
  inside the embedding hook leaves the image scatter and the image-token tap untouched.
* **The projector's output is normalised to the VLM's own embedding RMS** (measured: 0.650
  for this checkpoint). A freshly initialised projector otherwise emits vectors whose norm
  is nothing like a real token's, and the frozen decoder either ignores them or saturates —
  a failure of *scale* that looks exactly like a failure of the idea.
* **A `gate` scalar** on the whole output (`--gate-init`, default 1.0) so the injection
  strength is one interpretable number to watch, and so `--gate-init 0` can start training
  from the `zero` arm instead of from a random one.
* **Evaluation runs the released CUDA kernel, training runs the differentiable fallback**
  (`--eval-attn kernel`, the default), so every reported metric is the released model's own
  number and only the gradient path uses the substitute.
* **The LiDAR tokeniser is deliberately crude** — a parameter-free 8×8 BEV grid of height
  histograms (K = 64 tokens of dim 19; projector 1.59 M parameters). The eventual design is
  `extract.py`'s frozen TransFusion queries; the histogram stands in because the packed
  demo frames carry no nuScenes sample token to run our detector on, and it still carries
  the fact that matters ("where is there stuff, and how tall") which a single image answers
  badly.

### What is established so far

**The training step runs, and the gradient is real.** One end-to-end step — occupancy
cross-entropy at the BEV head, backward through the frozen 4 B decoder, into the projector:

```
prompt length 3718 tokens, LiDAR tokens inserted at index 8
embedding rms 0.6500   d_llm 2560   K 64 tokens of dim 19
projector: 1.59 M trainable parameters
  [eval@0] loss 0.5480  mIoU 0.2769
   [mem] after first train step   allocated 9.01 GiB   peak 33.54 GiB
   grad norm at step 1: 2.8444e-01   gate 9.9899e-01
```

So the whole path costs **~33.5 GiB peak** (≈ 25 GiB of activations over ~8.9 GiB of frozen
weights) for batch size 1 — one H100, comfortably, alongside another job.

**Rung 1 — 1 frame, 50 steps, lr 3e-3, seed 0 (complete).** The frame is *trained on*, so
these are train-set numbers; there is no held-out set at this rung and none is intended.

| epoch | occ loss | Δloss | mIoU | ΔmIoU |
|---|---|---|---|---|
| 0 (untrained projector) | 0.5481 | — | 0.2759 | — |
| 10 | 0.5365 | −0.0116 | 0.2773 | +0.0015 |
| 20 | 0.5324 | −0.0156 | 0.2741 | −0.0018 |
| 30 | 0.5307 | −0.0174 | 0.2756 | −0.0003 |
| 40 | 0.5276 | −0.0204 | 0.2779 | +0.0021 |
| **50** | **0.5241** | **−0.0239** | **0.2802** | **+0.0044** |

Final gate 1.1003; mean 98.8 s/step. Per-class IoU, epoch 0 → 50: `background` +0.0163,
class 4 +0.0087, `empty` **+0.0007**, class 0 −0.0054, class 7 −0.0022.

**Read.** The loss falls monotonically and the last three mIoU points rise monotonically
(0.2756 → 0.2779 → 0.2802), ending +0.0044 above baseline. But the mIoU series spans
0.0061 overall — it dipped *below* baseline at epoch 20 — so **the final gap is comparable
to the series' own spread**, and one frame at one seed cannot separate them. The curve had
not plateaued.

⚠ **A call was made too early here and is retracted.** At epochs 20–30, with loss falling
and mIoU flat-to-negative, this was written up as the "constant prior / calibration" failure
mode of `RESEARCH_DIRECTIONS.md` §2.6 — the objective being served by sharpening a class
that already dominates. The completed run does not support that: `empty` IoU moved by
+0.0007, essentially nothing, while the mIoU gains sit on `background` and class 4. The loss
reduction is not concentrated in the dominant class. **Two of five points on a monotone-loss
curve is not enough to name a mechanism**, and the mechanism named happened to be the one
this project had recently been burned by — which is precisely when to be more careful, not
less (§7.5: do not report the most favourable of several readings; the same applies to the
most *familiar* one).

**What the rung does establish.** The channel is open and the projector can use it, but
weakly: 1.59 M free parameters, 50 steps, and a *single training frame* bought +0.0044
mIoU. Easy overfitting — which is what a wide channel would produce — would look nothing
like this. So the live question is bandwidth, which is rung 0's job.

**Rung 0 — channel bandwidth (complete).** One frame (`4d0d1ccb`, nuPlan, 640 000 voxels),
one set of images, trains nothing; only the injected tokens vary. `trained` is rung 1's
saved projector, `trained_other` is that same projector fed a **different frame's** point
cloud. Its loss reproduced rung 1's final eval to the digit (0.5241), so the two runs
measure the same object.

| probe | occ loss | argmax vs `zero` |
|---|---|---|
| `none` (no extra tokens) | 0.5491 | 0.083 % |
| `zero` (64 null embeddings) | 0.5485 | — |
| `self` (this cloud, untrained projector) | 0.5481 | 0.084 % |
| `other` (other cloud, untrained) | 0.5487 | 0.094 % |
| `random` (norm-matched noise) | 0.5489 | 0.122 % |
| **`trained`** (this cloud, rung 1's weights) | **0.5241** | **0.575 %** |
| **`trained_other`** (other cloud, same weights) | **0.5343** | 0.436 % |

| pair | logit rel L2 | argmax diff | |
|---|---|---|---|
| `self` vs `other` | 1.26e-02 | **0.039 %** | bandwidth, untrained: only the cloud differs |
| `random` vs `zero` | 3.40e-02 | 0.122 % | a *typical* token direction |
| `none` vs `zero` | — | 0.083 % | the 64 extra positions alone |
| `trained` vs `self` | 7.13e-02 | **0.585 %** | what 50 steps achieved, in predictions |
| `trained` vs `trained_other` | 2.81e-02 | **0.171 %** | bandwidth, trained: only the cloud differs |

**Result 1 — the effect is real, and 57 % of it is a constant.** The wrong-cloud control
decomposes rung 1's gain exactly:

```
self  -> trained            0.5481 -> 0.5241     total gain 0.0239
self  -> trained_other      0.5481 -> 0.5343     survives a WRONG cloud   0.0138 = 57.5 %
trained_other -> trained    0.5343 -> 0.5241     cloud-specific            0.0102 = 42.5 %
```

**The majority of what the projector learned is a frame-independent constant** — a bias the
frozen head finds useful regardless of what the LiDAR says. That is the §2.6 "constant
prior", here for the first time *quantified* rather than diagnosed as present/absent, and
the remaining 42.5 % is genuine cloud-reading. Both components are real. **Any future number
from this line is meaningless without the wrong-cloud arm reported next to it**, because
more than half of the naive gain is not about the LiDAR.

**Result 2 — the channel is narrow but exploitable.** Every prediction change is well under
1 % of voxels. Before training, prompt *length* matters more than LiDAR content (0.083 % for
64 extra positions vs 0.039 % for swapping the entire point cloud) — which is exactly why
the `zero` arm is not optional. After training, a cloud swap moves 0.171 %, 4.4× the
untrained sensitivity, and training as a whole moved 0.585 % (3 744 voxels).

⚠ **Result 3 — a control of mine was mislabelled, and it would have given the wrong
remedy.** `random vs zero` was pre-registered as "the upper bound on what ANY token content
can do". **It is not an upper bound.** Training reached 0.585 %, **4.8× more** than random
content's 0.122 %. In a 2560-dimensional embedding space a random direction is nearly
orthogonal to the useful subspace, so a random-input control measures the *typical*
reachable effect, not the *maximum* one. The pre-registered remedy table's first row used
"`random` ≈ 0 and `self vs other` ≈ 0 → the channel itself is narrow → go widen it", and on
these numbers that row fires — yet the trained probe shows the channel supports 15× the
untrained content sensitivity, so "widen the channel" is not the diagnosis. **Generalised:
a random-input control lower-bounds what optimisation can reach; only an optimised probe
upper-bounds it.** The `trained` probe was added late, almost by accident (§4's kill
incident), and it is the reason this was caught.

**Verdict on rung 0 and rung 1 together.** Not a FAIL — pre-registered FAIL-3 required
argmax drift ≈ 0, and it is 0.585 % with 42.5 % of the loss gain cloud-dependent. Not a PASS
either: the whole effect is sub-1 % of voxels and +0.0044 mIoU on the frame being trained
on, with the majority a constant. **The route is alive and weak.** What that licenses is the
4-arm gate on 6 frames — where `shuffle` is the same control as `trained_other`, now known
to be the load-bearing one — and *not* any claim about LiDAR helping.

The argmax-drift instrument postdates rung 1, so its logs cannot supply the drift number.

**The zero-gradient episode is the finding worth carrying.** In the first version the
gradient was exactly `0.0` and the loss bit-identical across a step. That is not a negative
result about the idea; it is a plumbing bug (causal order — see the first design bullet),
and `train_projector.py` now **aborts** on a zero gradient with that message rather than
reporting it as a number. §7.8 in the other document says a null from an intervention is
worthless until the intervention is shown to do something; a zero gradient is the sharpest
possible version of that.

**Cost, measured.** One train step (fwd + bwd, batch 1) is **~100 s**, and one eval frame
**~95 s**. `--attn-math bf16` changes the *memory* (33.5 → 27.2 GiB peak) but **not the
time** (99.6 vs ~100 s/step) — so the deformable attention's arithmetic precision is not the
bottleneck, and fp32 is the right default because it also keeps the backward out of bf16.

⚠ These timings are **contended**: another job held 22–43 GiB and drove the GPU to 99 %
utilisation throughout. Treat them as an upper bound on an idle card.

**⚠ The forward cost, not the gradient, was the real blocker — and it is fixed.** Three
timings first said where it was *not*: eval with the released CUDA kernel under `no_grad`
(95–98 s/frame), training with the pure-torch fallback plus backward (~100 s), and the same
in bf16 (99.6 s) were all the same. The backward is nearly free, and the deformable-attention
implementation and its precision are both irrelevant. So it was neither the gradient nor the
attention — which is not an answer, so the stack got profiled instead of guessed at
(`check_grad.py --stage profile`, nested inclusive labels, `cuda.synchronize()` on both
sides, two passes so cold one-off costs are not charged to whoever triggered them).

**Steady state, one 8-camera frame, as released:**

```
pass 2 (steady): infer() total  92.76 s
vlm.visual (ViT)     91.85   99.0%      <-- everything is here
  patch_embed        91.79   98.9%      <-- and it is all in ONE layer
  blocks[0] of 24     0.00
  merger              0.00
vlm.language_model    0.57    0.6%
bev_head              0.33    0.4%      <-- the part this whole document is about
```

Two hypotheses died on the way. *Attention:* the vision blocks are 0.00 s. *Gradient
checkpointing* (`gradient_checkpointing_enable()` is global, and `Qwen3_5VisionBlock` is a
`GradientCheckpointingLayer`, so `vlm.train()` wraps all 24): measured on **and** off,
91.85 vs 92.86 s — no effect. `set_vit_checkpointing()` exists from that attempt and is
harmless, but it is not the fix.

**The cause.** `Qwen3_5VisionPatchEmbed` is

```python
self.proj = nn.Conv3d(3, 1024, kernel_size=[2, 16, 16], stride=[2, 16, 16], bias=True)
hidden_states.view(-1, 3, 2, 16, 16)      # the kernel covers the ENTIRE input
```

— a convolution whose kernel is its whole input and whose output is 1×1×1 per patch, i.e. a
matmul in convolution clothing. At 14 336 patches in bf16 it lands on a cudnn path four to
five orders of magnitude off the arithmetic. Isolated, with no model loaded, on exactly the
released shape:

```
conv3d     91.4920 s      out (14336, 1024)
matmul      0.0046 s      out (14336, 1024)
max|conv - matmul| = 0.0000e+00      rel L2 = 0.0000e+00
```

**Bit-identical**, not merely within bf16 tolerance — the contraction runs over
`(in_ch, kt, kh, kw)` in C order for both operands, so `x.reshape(N,1536) @
w.reshape(1024,1536).T + b` is the same computation, and the incoming tensor is *already*
`(N, 1536)`, so even the reshape is free.

**`set_vision_patch_embed(vlm, 'matmul')`** applies it (flag `--patch-embed`; default
`matmul` in `train_projector.py`, `conv` in `check_grad.py` so the released behaviour stays
reproducible). Every call self-verifies and prints the max abs diff. Result end to end:

| | as released | patch-embed as matmul |
|---|---|---|
| `infer()` steady state | **92.76 s** | **1.02 s** |
| ViT | 91.85 s (99.0 %) | 0.06 s (5.9 %) |
| language model | 0.57 s | 0.57 s (56.1 %) |
| BEV head | 0.33 s | 0.37 s (36.1 %) |
| one nuScenes-train epoch (28 130 frames), forward only | ~780 GPU-h | **~8 GPU-h** |

**91× on the full forward, for free and exactly.** The profile is now sane — the 4 B decoder
and the BEV head dominate, as they should — and step (c) went from arithmetically impossible
to routine. This also retires the "cache the frozen ViT per frame" idea: at 0.06 s there is
nothing left to cache.

⚠ Both earlier timing tables in this document (the 95–98 s eval, the ~100 s train step, the
fp32-vs-bf16 comparison) were measured with the Conv3d path and are **all ~99 % patch
embedding**. They are kept because they are what led here, but do not use them to compare
anything: they measure one pathological layer, not the model.

**Ladder, not one big run.** At ~100 s/step, the 4-arm gate on 6 frames is ~5 h, and a
FAIL there would be ambiguous between "the channel is too narrow" and "60 steps was not
enough training". So the question is asked in cheaper forms first:

0. **`--mode sensitivity`** (5 forwards, ~9 min, trains nothing). The prior question:
   *how wide is the channel at all?* With an **untrained** projector, feed the same frame
   the same images and vary only the injected tokens — `none` / `zero` / this frame's cloud
   (`self`) / another frame's cloud (`other`) / norm-matched `random` — and measure how far
   the occupancy output moves. **`self` vs `other` is the bandwidth**: same projector, same
   images, only the point cloud differs. If that is ≈ 0, no optimiser will find a signal in
   the channel. `random` bounds what *any* token content can do; `none` vs `zero` is the
   cost of the K extra positions alone.
1. **1-frame capacity probe** (`--max-frames 1 --epochs 50 --lr 3e-3`, ~85 min): can the
   projector actually *use* it? `--inject shuffle` is refused with one frame — the "other"
   cloud would be the same cloud, so it would not be a control.
2. only if that moves: the **4-arm gate** on all 6 frames (`run_projector_gate.sh`).

⚠ **Rung 0 was written after rung 1 had already been launched.** That is §7.9 being
violated and then obeyed inside the hour: 50 SGD steps *is* the perturbation, and its
effect size had not been measured. It is recorded because the same reflex produced the
FusionOcc FOV probe, which cost a week.

⚠ **And the attempt to stop rung 1 failed silently** — `kill` hit the launching shell, and
`nohup`'d Python was reparented to init and kept running with 41 GiB. The next run then
OOMed against it (`Process 1004665 has 41.21 GiB memory in use` — that process was mine).
**On this box, verify a kill by PID and by `nvidia-smi --query-compute-apps`, not by the
shell's exit**; two unrelated `alpasim` jobs also live on the same card, so a stale job of
your own is the difference between fitting and not. Rung 1 was left to finish and rung 0
chained behind it.

Numbers land in `outputs/<timestamp>_<name>_<arm>/log.json`, each next to the
`config.yaml`, seed and git commit that produced it.

### Rung 0 is a two-way diagnostic, and each way has a different remedy

The pair of rows `random vs zero` and `self vs other` separates two failure causes that
would otherwise look identical from the training curve:

| `random vs zero` | `self vs other` | diagnosis | remedy |
|---|---|---|---|
| ≈ 0 | ≈ 0 | **the channel itself is narrow** — 64 soft tokens at the input of a frozen 36-layer decoder cannot modulate 3584 image tokens enough to reach the BEV head | widen it: per-layer KV prefixes (prefix-tuning / P-Tuning v2) instead of one input-level prompt, or a residual written into the **image-token positions** themselves — 3584 positions instead of 64, which is how the vision features already enter |
| large | ≈ 0 | the channel is wide but **our content is not reaching it** — a projector/optimisation problem, not a bandwidth one | more tokens, a better LiDAR tokeniser (`extract.py`'s detector queries instead of height histograms), longer schedule, LR sweep |
| large | large | the channel is fine; a FAIL at rung 1/2 would then be about the *supervision*, not the plumbing | proceed to the 4-arm gate |

Writing this table before the numbers arrive is deliberate: each cell already names its
next action, so the result cannot be rationalised into whichever remedy is most appealing
afterwards. Note that "widen the channel" moves us toward what LVLDrive's Gradual Fusion
Q-Former already does (§5) — worth knowing *before* choosing it, not after.

⚠ **The table's first row turned out to rest on a bad premise** and must not be used as
written: it treats `random vs zero` as an upper bound on the channel, and it is not one
(Result 3 below). Read the row conditions as "*typical* content barely moves it" rather than
"*nothing* can move it", and require an optimised probe before concluding the channel is the
problem.

### Post-fix costs, and the `zero` arm earning its place

With `--patch-embed matmul` the whole apparatus is a different size:

| | before | after | |
|---|---|---|---|
| training step (fwd+bwd, batch 1) | ~100 s | **5.4 s** | 18× |
| eval frame | 95–98 s | **1.0 s** | ~95× |
| 4-arm gate, 6 frames × 40 epochs | ~5 h | **~50 min** | |
| one nuScenes-train epoch (28 130) | ~780 GPU-h | **~42 GPU-h** | training, not just forward |
| peak | 33.6 GiB | 33.6 GiB | unchanged |

In-model self-verification of the swap, on the real tensor: conv 26.51 s → matmul 0.0005 s
(**52 541×**), `max abs diff 0.00e+00`.

**Reference arms over all 6 demo frames** (released model, eval only):

| arm | occ loss | mIoU |
|---|---|---|
| `none` — released path | 0.2225 | **0.3735** |
| `zero` — +64 null-embedding tokens | 0.2217 | **0.3613** |

**Injecting 64 tokens with no content costs −0.0122 mIoU**, which is larger in magnitude
than rung 1's entire +0.0044 gain. So **the reference for the trained arms is `zero`, not
`none`** — and the arm that looked like bookkeeping is what prevents a sign error: a `lidar`
result of 0.3700 reads as "LiDAR hurts by 0.0035" against `none` and "**LiDAR helps by
0.0087**" against `zero`. Any result in this family quoted against a no-extra-token baseline
is measuring prompt length as much as LiDAR.

### The 4-arm gate — result, and a metric that had to be thrown away

6 demo frames, 40 epochs, lr 1e-3, seed 0, histogram tokeniser. Evaluated on the frames it
trains on; there is no held-out set at this rung.

| arm | mIoU@0 | mIoU@40 | ΔmIoU | fwIoU@0 | fwIoU@40 | **ΔfwIoU** | argmax drift |
|---|---|---|---|---|---|---|---|
| `none` | 0.3735 | — | — | 0.2865 | — | — | — |
| `zero` | 0.3613 | — | — | 0.2860 | — | — | — |
| `lidar` | 0.3719 | 0.2735 | **−0.0983** | 0.2844 | 0.3078 | **+0.0233** | 0.829 % |
| `shuffle` | 0.3719 | 0.2700 | **−0.1019** | 0.2847 | 0.3038 | **+0.0192** | 0.561 % |

**⚠ mIoU and frequency-weighted IoU disagree in sign, and mIoU is the one that is wrong.**
The −0.098 "collapse" is *one class*: `generic_object` goes 0.8750 → 0.0000 in both arms.
Every other class improves. That class has **14 voxels** in the entire 6-frame set:

| class | GT voxels (6 frames) |
|---|---|
| `empty` | 3 597 816 |
| `background` | 184 577 |
| `driveable` | 43 908 |
| `vehicle` | 11 892 |
| `pedestrian` | 1 664 |
| `bicycle` | **73** |
| `traffic_cone` | **55** |
| **`generic_object`** | **14** |
| `czone_sign` | **1** |
| `barrier` | **0** (excluded) |

Half the classes entering the unweighted mean are decided by 1–73 voxels, so mIoU here is a
mean over four real numbers and four coin flips; 14 voxels changing class moves it by
0.875/8 ≈ 0.109. This is `RESEARCH_DIRECTIONS.md` §7.3 (never quote a ratio with a near-zero
denominator) arriving through a metric rather than a ratio, and §7.1 (measure the noise floor
before interpreting a gap) — **the noise floor here is per class, and it was never asked
for.** The pre-registered read criterion for this gate named mIoU. That was a flaw in my own
instrument, found only by printing per-class IoU instead of accepting the summary line.
Frequency-weighted IoU (GT-voxel-weighted) is used from here on, with mIoU reported beside
it and never alone.

**The result, on the metric that works: FAIL-2, as pre-registered.**

```
lidar    fwIoU +0.0233     drift 0.829 %
shuffle  fwIoU +0.0192     drift 0.561 %
lidar - shuffle           +0.0041   =  18 % of the effect
```

**82 % of the gain survives feeding the projector a different frame's point cloud.** On one
frame at rung 0 the constant fraction was 57.5 %; over six frames and a full 40-epoch run it
is 82 %. The projector is overwhelmingly being used as free parameters that shift the frozen
head's output in a globally useful direction, not as a channel that carries this scene's
LiDAR. That is the §2.6 constant prior, third independent measurement, now with the
pre-registered control that names it.

**What this does and does not license.** It is FAIL-2 for *this* tokeniser: an 8×8 height
histogram. It is **not** a claim that LiDAR tokens cannot help — §5.5's prerequisite is
exactly this situation, and the queries-vs-histogram comparison is the next measurement. The
`zero` arm remains load-bearing: against `none` the trained arms look like a loss, against
`zero` like a gain, and only `shuffle` says how little of either is about LiDAR.

### ⚠ The common ruler needs a common *ground truth*, not just a common taxonomy

Both nuScenes demo frames are keyed by real 32-hex sample tokens, so Qwen-Drive's **shipped**
`gt.npz` can be compared voxel-by-voxel against our Occ3D-derived mapping of the *same*
frame. They do not agree.

| | `90162f90…` | `f41862469…` |
|---|---|---|
| overall voxel agreement | 97.72 % | 98.23 % |
| **occupied IoU** | **0.6043** | **0.6152** |
| occupied voxels, shipped | 32 250 | 27 392 |
| occupied voxels, ours (Occ3D) | 25 624 | 18 794 |
| theirs occupied, ours empty | 10 450 | 9 800 |
| ours occupied, theirs empty | 3 824 | 1 202 |

The overall 98 % is meaningless — 94 % of voxels are `empty`. The real number is **occupied
IoU 0.60**, and the asymmetry is one-sided: their GT calls ~10 000 voxels occupied that
Occ3D calls free, three times more than the reverse.

**Where those voxels are settles it:**

| | `90162f90…` | `f41862469…` |
|---|---|---|
| extra voxels inside `mask_camera` | **5.1 %** | **10.9 %** |
| restricted to `mask_camera`: agreement | 96.32 % | 96.44 % |
| restricted to `mask_camera`: **occupied IoU** | **0.8384** | **0.8697** |

**~90 % of the disagreement is in unobserved space.** Their annotation fills occupancy where
Occ3D declines to label; inside the region Occ3D itself considers observed the two agree at
0.84–0.87 occupied IoU. (`mask_camera` and `mask_lidar` give identical percentages here,
consistent with the nesting measured in `visibility.py`.)

**Consequence for task (b), and it is not a detail.** Scoring Qwen-Drive against raw Occ3D
charges it ~0.24 occupied IoU for predicting in regions our GT declines to label — a
systematic, *asymmetric* penalty that looks exactly like model error, and it points the
opposite way from the taxonomy-collapse bias (§5 `qwen_drive_labels`), so the two do not
cancel. The headroom measurement must be run **inside `mask_camera`**, or reported both ways
with this ceiling stated. `pack_nuscenes.py` now carries `mask_camera`/`mask_lidar` into every
packed frame and `train_projector.py --occ-mask camera` applies it to both the metric and the
argmax-drift support.

**And there is a ceiling worth quoting on its own:** 0.84–0.87 is what *their own GT* scores
against Occ3D inside the observed region. No model trained to their convention can exceed
that against our GT. Any headroom number must be read against that ceiling, not against 1.0.

*n = 2 frames* — the only nuScenes frames the release ships. The effect is large and
one-directional, but it is two frames, and the packer now makes it cheap to check on many.

This is what "there is no common ruler" actually means, and it is more than taxonomy: two
independent annotation pipelines disagree three times more about **unobserved** space than
about observed space. That connects straight to §4.6 and the occluded-region line — the
region where the two pipelines diverge is precisely the region that project has been arguing
is both excluded from evaluation and where camera+prior would have to do the work.

### The tokeniser is not the bottleneck — the channel is

§5.5 named a prerequisite: no negative claim about LiDAR utility survives an 8×8 height
histogram, because LVLDrive uses FSDv2 pretrained on nuScenes and the referee's first
objection writes itself. So the histogram was replaced by a **frozen LiDAR-only BEVFusion**'s
TransFusion object queries (`extract.py`, 200 queries × 128 ch, NDS 0.6922 checkpoint —
LiDAR-only *deliberately*: queries from BEVFusion-LC would carry camera features the VLM
already sees, and both of this line's questions would become unanswerable).

Rung 0, same frame (`02a9f747`, cloud swapped with `04917e6b`), same probe, same untrained
projector architecture — only the tokeniser differs:

| pair | histogram (K=64, d=19) | **queries (K=200, d=128)** | |
|---|---|---|---|
| `self` vs `other` | 0.157 % | **0.142 %** | **bandwidth: only the cloud differs** |
| `random` vs `zero` | 0.259 % | 0.307 % | a typical token direction |
| `none` vs `zero` | 0.208 % | 0.258 % | the K extra positions alone |
| loss, `self` → `other` | 0.3205 → 0.3200 | 0.3163 → 0.3184 | |

**A real detector's queries carry no more into the BEV head than a hand-made height
histogram** — 0.142 % against 0.157 %, if anything slightly less. 200 tokens of 128
learned channels from an NDS-0.6922 model move the frozen head's predictions by the same
fraction of a percent as 64 tokens of 19 arithmetic features.

Note what *does* scale: `none vs zero` grows 0.208 → 0.258 % and `random vs zero` 0.259 →
0.307 % as K goes 64 → 200. **More tokens buy more perturbation and no more information
transfer.** The loss does move the right way for queries (self 0.3163 < other 0.3184, where
the histogram gives nothing), so the queries are not *worse* — they are simply not the
limiting factor.

**This answers the prerequisite objection with a measurement rather than a promise:**
upgrading to the encoder family LVLDrive uses changes the bandwidth by nothing, so the limit
is the injection channel, not the LiDAR encoder. *Caveat: one frame, untrained projector —
this is a bandwidth probe.* The confirmatory run is the 4-arm gate with queries on real
frames (`run_step4.sh`), whose prediction — FAIL-2 repeats — is written into the script
before the numbers exist.

### Headroom, first half: the released model on real nuScenes

120 shuffled val frames, 74 distinct scenes, our packing, common taxonomy:

| scoring region | loss | fwIoU | mIoU |
|---|---|---|---|
| all voxels (raw Occ3D GT) | 0.2708 | 0.1840 | 0.1897 |
| **inside `mask_camera`** | 0.2708 | **0.3235** | **0.2817** |

`mask_camera` covers 13.4 % of voxels. Restricting to Occ3D's observed region raises fwIoU
by **+76 %** (0.184 → 0.324) — the same GT-convention penalty measured at the GT level
(occupied IoU 0.60 → 0.84/0.87) now reproduced at the model level. The identical loss in both
rows is the consistency check: the loss is deliberately unmasked, so the two runs are the
same forward pass and only the metric's support changed.

**So the number to carry forward is fwIoU 0.3235 inside `mask_camera`, not 0.184** — and it
must be read against the 0.84–0.87 ceiling that their own GT scores against ours, not
against 1.0. The other half of the headroom measurement (BEVFusion-LC on these same 120
tokens) has not been run.

### ⭐ The citable run: the LiDAR *is* read, and the **objective** is what destroys the result

The version whose negative result survives review: 24 real nuScenes frames (74-scene pool),
20 epochs, **frozen LiDAR-only detector queries**, scored **inside `mask_camera`**, with
fwIoU beside mIoU. The prediction written into `run_step4.sh` before it ran was "FAIL-2
repeats". **That prediction was wrong**, and the truth is more useful.

| arm | loss @0 → @20 | fwIoU @0 → @20 | ΔfwIoU | argmax drift |
|---|---|---|---|---|
| `zero` | 0.2900 | 0.3120 | — | — |
| `lidar` | 0.2896 → **0.2159** (−0.0737) | 0.3119 → 0.2740 | **−0.0378** | **6.08 %** |
| `shuffle` | 0.2895 → 0.2278 (−0.0617) | 0.3118 → 0.2861 | −0.0257 | 5.64 % |

Loss falls hard (−25 %) and **fwIoU falls with it**. Drift is now 6 %, not 0.8 %, so the
channel is carrying a great deal. And `lidar` drops the loss *more* than `shuffle`
(−0.0737 vs −0.0617) while damaging fwIoU *more* (−0.0378 vs −0.0257).

**Per-class, `lidar` arm, and this is the whole story:**

| class | GT voxels | IoU @0 → @20 | |
|---|---|---|---|
| `empty` | 1 697 686 | 0.8217 → **0.8239** | **+0.0022 ↑** |
| `driveable` | 152 540 | 0.4596 → **0.4722** | **+0.0126 ↑** |
| `background` | 304 677 | 0.2244 → 0.1689 | −0.0555 ↓ |
| `vehicle` | 28 124 | 0.4752 → 0.3563 | **−0.1189 ↓↓** |
| `pedestrian` | 1 253 | 0.2905 → 0.2595 | −0.0310 ↓ |
| `bicycle` | 449 | 0.2352 → 0.0311 | **−0.2041 ↓↓↓** |
| `traffic_cone` | 436 | 0.2794 → 0.1757 | −0.1037 ↓↓ |

**The only two classes that improve are the two largest.** Every object class collapses, and
`shuffle` degrades them *less* on exactly the classes LiDAR should help with most
(`vehicle` −0.077 vs −0.119, `bicycle` −0.134 vs −0.204).

**Read it plainly: the LiDAR content is being read, and reading it makes the prediction
worse, because the objective being minimised is not the objective we want.** Occupancy
cross-entropy over a grid that is 74 % `empty` even inside `mask_camera` is minimised by
sharpening the dominant classes; a 1.6 M-parameter projector at the LLM input finds that
direction within ten epochs, and better LiDAR information simply gives it more leverage to
do so. At `eval@5` — before that takes over — **both arms are above baseline** (fwIoU
0.3178 and 0.3194 vs 0.3120), which is the same story from the other side.

**This supersedes the earlier "constant prior" reading of this line, without contradicting
it.** On 6 demo frames with a histogram, 82 % of a small fwIoU *gain* survived a wrong cloud.
On 24 real frames with real queries the arms do not gain at all — they trade objects for
background, and the right cloud trades harder. Both are the same underlying fact: **nothing
in this setup ever asked the projector to preserve objects.**

**So the next thing to fix is the loss, not the tokeniser and not the channel.** That is the
opposite of where §5.5 pointed, and it is where the evidence points now:

* Pooled occupancy CE is the wrong training signal here. Our own occupancy work already
  measured this family: class re-weighting was **negative**, **Lovász-softmax was the best
  single change** (ConvNeXt-V2-L + 3D head + Lovász 0.1485, +70 % over the R50 baseline —
  `RESEARCH_DIRECTIONS.md` §2.1 / the Occ3D SOTA line). Lovász optimises an IoU surrogate
  directly, which is exactly the mismatch measured above.
* An object-restricted objective (or a detection-query probe instead of occupancy) removes
  the imbalance entirely and is closer to what the frozen detector's tokens actually encode.
* Early stopping is *not* the fix — it would report the `eval@5` bump as a result while the
  optimiser is demonstrably heading somewhere worse.

*Caveats: 24 frames, one seed, 20 epochs, and the frames are the ones evaluated (no held-out
split at this rung). The direction and the per-class pattern are unambiguous; the magnitudes
are not seed-checked.*

### ⭐⭐ Fixing the objective — and it was mostly *not* the class imbalance

Two orthogonal candidate causes, so a 2×2 on the `lidar` arm rather than one combined run;
everything else identical to the baseline (24 real nuScenes frames, 20 epochs, frozen
LiDAR-only detector queries, scored inside `mask_camera`).

| config | trained loss | loss support | eval loss @20 | **fwIoU @20** | **ΔfwIoU** | mIoU @20 | drift |
|---|---|---|---|---|---|---|---|
| **A** (baseline) | CE | all voxels | 0.2159 | 0.2740 | **−0.0378** | 0.2059 | 6.08 % |
| **B** | CE | `mask_camera` | 0.3363 | 0.4216 | **+0.1097** | 0.3273 | 6.88 % |
| **C** | CE + Lovász | all voxels | 0.2567 | 0.3418 | **+0.0299** | 0.3026 | 3.29 % |
| **D** | CE + Lovász | `mask_camera` | 0.3527 | **0.4333** | **+0.1215** | 0.3462 | 7.21 % |

**The supervision region was the dominant cause, not the class imbalance.** Masking the loss
alone recovers +0.1097 of the +0.1215; Lovász alone recovers +0.0299 and adds only ~+0.012
on top of the mask. That is the opposite of the prior stated one section earlier, where
Lovász was the indicated fix on in-house evidence — the in-house evidence was about a model
trained *from scratch* on Occ3D, where the supervision region was never in question. Here the
frozen head was trained to a **denser** GT convention, so an unmasked loss spends its
capacity teaching the projector to delete occupancy the head was built to emit.

**The tell is the eval loss, which is plain unmasked CE in every row.** It *rises* in B and D
(0.2896 → 0.3363 / 0.3527) while fwIoU rises sharply — the exact mirror of the baseline,
which drove that same number down 25 % and got worse. Optimising unmasked CE made the model
worse; declining to optimise it makes the model better. It is not a proxy for quality here;
it is anti-correlated with it.

**Per-class, ΔIoU from the untrained state — the object collapse is reversed:**

| class | GT voxels | A (baseline) | B | C | **D** |
|---|---|---|---|---|---|
| `vehicle` | 28 124 | −0.119 | +0.004 | −0.009 | **+0.016** |
| `bicycle` | 449 | −0.204 | +0.063 | +0.091 | **+0.099** |
| `pedestrian` | 1 253 | −0.031 | +0.021 | +0.043 | **+0.063** |
| `barrier` | 1 241 | −0.004 | +0.129 | +0.075 | **+0.153** |
| `traffic_cone` | 436 | −0.104 | −0.027 | −0.009 | **−0.017** |
| `driveable` | 152 540 | +0.013 | +0.202 | +0.059 | **+0.220** |
| `background` | 304 677 | −0.056 | +0.074 | +0.019 | **+0.083** |
| `empty` | 1 697 686 | +0.002 | +0.031 | +0.010 | **+0.033** |

Every class improves under D except `traffic_cone` (−0.017, and it was −0.104 under A) and
`generic_object`, which is never predicted by any arm. The classes that collapsed hardest
under pooled unmasked CE — `bicycle` −0.204, `vehicle` −0.119 — are now positive.

**So the diagnosis was right and the fix works: the optimiser no longer buys loss by
destroying objects.**

⚠ **What this does NOT yet show.** The `shuffle` control for config D has not returned.
Until it does, +0.1215 is a statement about the *optimiser*, not about the LiDAR: a fix that
works just as well with a different frame's point cloud has repaired training, not enabled
the sensor. The prior from the histogram gate is that **82 % of a gain survived a wrong
cloud**, and drift here is 7.2 %, so expect the worst and read the control before the number.
Note also that both training and scoring now use `mask_camera`; the supports match by design
(it is Occ3D's own protocol), and A rules out an artefact since A was *scored* on the same
mask and still got worse.

*24 frames, one seed, trained on the frames evaluated.*

### ⭐⭐⭐ The answer: the gain is a learned constant prefix, and the LiDAR contributes nothing

Config D (CE + Lovász, loss on `mask_camera`) on 24 real nuScenes frames, 20 epochs, frozen
LiDAR-only detector queries, scored inside `mask_camera`; **24 further frames held out and
never trained on**; `const` is the arm whose projector input is the **same vector for every
frame**, so its tokens carry zero information about anything.

| arm | train fwIoU @0 → @20 | Δ | **held-out** @0 → @20 | **Δ** |
|---|---|---|---|---|
| **`const`** s0 — *zero information* | 0.3122 → 0.4364 | **+0.1242** | 0.3434 → **0.4315** | **+0.0881** |
| `lidar` s0 | 0.3119 → 0.4333 | +0.1215 | — | — |
| `shuffle` s0 | 0.3118 → 0.4593 | +0.1475 | — | — |
| `lidar` s1 | 0.3121 → 0.4346 | +0.1224 | 0.3430 → 0.4207 | +0.0778 |
| `shuffle` s1 | 0.3125 → 0.4451 | +0.1326 | 0.3431 → 0.4249 | +0.0818 |

**`const` is the best arm on held-out data** (+0.0881) and the correct point cloud is the
worst (`lidar` s1, +0.0778). On the training frames all five land in +0.12 – +0.15. Seed
spread: `lidar` +0.1215/+0.1224 (0.0009), `shuffle` +0.1475/+0.1326 (0.0149) — so seed-0's
"shuffle beats lidar by 0.026" was largely seed noise, and the honest statement is that
**`const`, `shuffle` and `lidar` are indistinguishable.**

**Conclusion, and it is clean.** The +0.12 fwIoU produced by the fixed objective is a
**learned constant prefix**: a fixed block of soft tokens that recalibrates the frozen head
from Qwen-Drive's denser GT convention toward Occ3D's. It needs no input, it is not about
LiDAR, and it **generalises to frames never trained on** (+0.088), which is exactly what
identifies it as a recalibration rather than per-frame memorisation. The injected modality
contributes **nothing measurable**.

**The causal chain, end to end, each link measured:**

1. The released head does not backpropagate — a raw CUDA op with no autograd wrapper; fixed
   with the release's own pure-torch fallback (99.98 % argmax parity).
2. 99 % of the forward was one `Conv3d` that is a matmul in disguise; replacing it is
   bit-identical and **91× faster**, which is what made any of this affordable.
3. Trained against pooled unmasked CE, the projector *destroys* objects (`bicycle` −0.204,
   `vehicle` −0.119) while driving the loss down 25 %.
4. The cause is mostly **the supervision region**, not class imbalance: masking the loss to
   `mask_camera` recovers +0.1097 of the +0.1215; Lovász alone recovers +0.0299.
5. And the repaired gain is **input-independent**: `const` matches it.

**What this licenses as a claim.** Not "LiDAR tokens do not help" — that is a statement about
LiDAR we have not earned. The earned statement is sharper and more useful:

> A gain from injecting a modality into a frozen VLM + frozen head can be produced **in
> full** by a fixed, information-free token block that recalibrates the frozen head to the
> evaluation's GT convention. Without a wrong-input or constant-input control, such a gain is
> indistinguishable from modality fusion.

**And that lands directly on the published work.** LVLDrive's headline modality gain is
**+0.01 BEV mIoU** (camera 0.18 → camera+LiDAR 0.19) from a *retrained-arm* ablation with no
wrong-input control. We measure that a channel of this shape yields **+0.12 with no
information at all** — an order of magnitude more than the effect they attribute to LiDAR.
That does not prove their result is an artefact; it proves **their design cannot tell.**

**Stop tuning this; it is a result.** Further work on the projector, the tokeniser or the
channel would be optimising a recalibration. What remains is the write-up plus the two
measurements that make it a paper rather than a case study: BEVFusion-LC on these same 120
tokens under the same ruler (§7 task b — the packer and taxonomy are ready), and a second
architecture (a re-implementation of LVLDrive's Gradual Fusion Q-Former) so the claim is
about LLM-level injection and not about our own stack.

---

## 5. ⚠ Literature check — done **before** the experiment this time, and it moves the goal

Rule 7.11 exists because four "nobody does X" claims in this project were already published.
So this was searched before the projector was trained, and the finding matters:

**"Add LiDAR as an extra input modality to a pretrained driving VLM" is published.**

| work | how LiDAR enters | VLM | what they report | code |
|---|---|---|---|---|
| **LVLDrive** / *Spatial-aware VLM for AD*, [arXiv 2512.24331](https://arxiv.org/abs/2512.24331), CVPR-W 2026 (AUTOPILOT) | **Gradual Fusion Q-Former** that *incrementally* injects LiDAR features, explicitly to avoid "catastrophic disturbance" of the pretrained VLM; plus their own SA-QA dataset | pretrained, largely preserved | beats vision-only on scene understanding, metric spatial perception, driving decisions | none found |
| **VLGA**, [arXiv 2606.12396](https://arxiv.org/html/2606.12396) | LiDAR at **training time only** — a 5-layer decoder regresses 3D pointmaps from "geometry tokens", discarded at inference | **frozen** | nuScenes open-loop L2/collision, Bench2Drive closed-loop DS 79.08 | none found |
| **VLA-MP** ([Sensors 25(19):6163](https://www.mdpi.com/1424-8220/25/19/6163)) | RGB + LiDAR → cross-modal projector → LLM → trajectory | fine-tuned | planning | — |
| **LiDAR-LLM**, [arXiv 2312.14074](https://arxiv.org/pdf/2312.14074) | BEV features → LLM, staged curriculum | fine-tuned | 3D captioning / grounding / QA | — |
| **OccVLA**, [arXiv 2509.05578](https://arxiv.org/pdf/2509.05578) | no LiDAR; *implicit 3D occupancy supervision* inside a VLA | — | — | — |

### 5.1 LVLDrive, read properly — and its own numbers argue *for* the audit

Read in full (arXiv 2512.24331v2, 14 pp., Motional + U. Amsterdam), not from the abstract.

**What it actually is.** Base model **OmniDrive-Agent**; vision encoder EVA-02-L; LLM
**LLaVA v1.5**; point encoder **FSDv2 pretrained on nuScenes** — i.e. a real LiDAR detector
backbone, not a hand-rolled tokeniser. **Nothing is frozen**: joint fine-tune of the whole
thing, LoRA on the LLM, lr 2e-5 on the image encoder, point encoder *and* LLM, 4e-4 on the
Gradual Fusion Q-Former. 12 epochs, batch 16, **8×A100-40GB for ~30 h** (≈240 A100-hours).
No code release statement; CC-BY-NC-SA.

So it is **not** our plan. Our plan is frozen-VLM + frozen-detector + a projector; theirs is
a full joint fine-tune with a learned gated Q-Former. That difference is real but it is a
*method* difference, and method is the crowded part — do not sell it.

**What it reports.** Open-loop planning on nuScenes (L2, collision rate CR, intersection
rate IR), DriveLM language metrics (BLEU-4/ROUGE-L/CIDEr), and one 3D number they call
*grounding*: the VLM is asked a 3D grounding question **with the 2D box already supplied in
the prompt**, the answer's BEV box is parsed, IoU computed per object, averaged inside four
distance bins, then averaged again → **BEV mIoU**.

| | L2 avg ↓ | CR avg ↓ | IR avg ↓ |
|---|---|---|---|
| OmniDrive-Agent (their base) | 0.33 | 0.30 | 3.00 |
| **LVLDrive** | **0.29** | **0.25** | **2.59** |

**The ablation that matters (their Table 3, independent Q-Formers):**

| modality | CR (%) ↓ | IR (%) ↓ | grounding mIoU ↑ |
|---|---|---|---|
| camera only (C) | 0.33 | 3.32 | **0.18** |
| **LiDAR only (L)** | 0.39 | 3.26 | **0.17** |
| C + L | 0.29 | 3.20 | **0.19** |

and the gate ablation, unified Q-Former, all C+L: no gate **0.17** → single gate 0.21 →
multi-head 0.21 mIoU (CR 0.40 → 0.30 → 0.28).

**Three things follow, and they are the best motivation this direction has.**

1. **An FSDv2 detector backbone, fine-tuned, inside a VLM, is *worse than the camera* at
   metric 3D localisation (0.17 vs 0.18).** On a task that is nothing but metric 3D
   localisation. That is prima facie evidence that the LiDAR path is barely being read —
   the exact hypothesis our audit exists to test, now with a published data point behind it.
2. **Adding LiDAR to camera buys +0.01 BEV mIoU** (0.18 → 0.19). Their headline 0.21 comes
   from the **gate** (0.17 → 0.21 at fixed C+L modality) and the SA-QA dataset — i.e. most
   of the reported gain is attributable to training machinery, not to LiDAR content.
3. **Their modality ablation is a retrained-arm ablation.** Each row is a separately trained
   model. That design *cannot* separate "the model reads the LiDAR" from "the extra
   parameters and the regularisation helped", which is precisely the distinction §2.6's
   two-ablation audit was built for — and which we have now measured on our own stack at
   **57.5 % frame-independent constant**. Nobody has run the feature-level ablation on an
   LLM-level fusion.

Their own limitation section names the bottleneck as the absence of large-scale paired
text–LiDAR data. Our frozen-detector-queries framing sidesteps that ("describe what the
detector found" instead of "learn to read point clouds") — a legitimate difference in
positioning, still not a headline.

### 5.2 VLGA, for contrast

LiDAR appears **only at training time**: a 5-layer decoder regresses 3D pointmaps from
"geometry tokens" and is **discarded at inference**; the vision-language backbone stays
frozen. Evaluated on nuScenes open-loop (L2 0.50, collision 0.18 % at 3 s, best among
methods without ego status) and Bench2Drive closed-loop (Driving Score 79.08, +0.71 over the
prior best VLA). **No detection, no occupancy, no segmentation, and no fusion-detector
baseline of any kind.** No code found.

So the two nearest works bracket the design space — LVLDrive injects LiDAR and fine-tunes
everything; VLGA never sees LiDAR at inference — and **neither puts a 3D number on the same
ruler as a LiDAR-camera detector.**

**What this closes.** "Frozen VLM + LiDAR tokens + a small projector" is not a method
contribution. LVLDrive is the same idea with a more careful injector, and it is a year old.
Do **not** write a paper whose claim is "we added LiDAR to a VLA."

**What it does not close**, and what we are actually equipped to do:

1. **Nobody in that list audits whether the LLM uses the LiDAR or merely benefits from a
   constant.** This is precisely the instrument from `RESEARCH_DIRECTIONS.md` §2.6: the
   **two-ablation audit** (corrupt the input *and* ablate the feature) separates a working
   branch from one emitting a useful input-*independent* prior, and it found the latter in
   two of our own three fusion models. A LVLDrive-style gain is exactly the shape of result
   that audit exists to test. This is cheap, it is ours, and it is unasked.
2. **Nobody measures the 3D headroom against a real fusion detector.** VLGA has no
   BEVFusion comparison at all; LVLDrive reports QA and decision metrics. The question
   "how much of BEVFusion's LiDAR advantage does a token-level injection actually recover?"
   has no published answer, and we have BEVFusion-LC (NDS 0.7118), the LiDAR-only floor
   (0.6922) and a 33-cell corruption harness to ask it in.
3. **Nobody tests it on the failure mode we measured.** §2.6 F4: the camera compensates
   *uniformly degraded* LiDAR but **not spatially absent** LiDAR, and the ranking of fusion
   models inverts between clean and degraded conditions. Whether a language-model-level
   fusion behaves differently there is a real question with a real prior.

So the framing shifts from *method* to *audit + headroom*, which is where this project's
comparative advantage has been all along.

---

## 5.5 The paper: what is the claim, and what is only the apparatus

Proposed shape — **frozen Qwen3-VL tap + our own 3D head + LiDAR, trained in
`ngperception/`**, with the paper's endpoint being (i) the two-ablation audit, (ii) 3D
headroom against a real fusion detector, (iii) the spatially-absent-LiDAR failure mode
(§2.6 F4). Assessment, bluntly.

### The stack is an enabler, not a contribution

"We built an open, trainable LiDAR+VLM+3D-head pipeline" will not carry a paper. LVLDrive
trains end to end today on 8×A100 in 30 h; the field does not lack a trainable stack, *this
release* lacks one. A referee will answer "OmniDrive and LVLDrive already do this" and be
right. Artifact/systems contributions land in workshops, benchmark tracks or journals, not
as a main claim at a top venue.

**But it is the necessary apparatus, and that is a defensible framing.** The audit needs a
host whose LiDAR pathway can be zeroed *at the feature level* at inference. LVLDrive's code
is unreleased, VLGA's is unreleased, and Qwen-Drive cannot be trained (§6). So there is
**no existing artifact on which this measurement can be made** — we have to build one, and
the paper says "we built the instrument, here is what it measures", with the stack in
Section 3 and the finding in Sections 4–6. That is a normal, publishable shape. What is not
publishable is the stack as the headline.

### The three endpoints, ranked

**(1) The two-ablation audit — strongest, and now it has a positive prior.** Two independent
data points already point the same way: our own **57.5 % frame-independent constant**, and
LVLDrive's own Table 3 where a fine-tuned FSDv2 backbone scores **below the camera** at
metric 3D localisation. Every published modality ablation in this space is a *retrained-arm*
ablation, which cannot separate reading from capacity. The feature-level ablation is unrun.

*Risk, and it is the real one:* **one model is a case study, not a finding about the field.**
"Our model learned a constant" is a bug report. To claim "LLM-level LiDAR injection tends to
a constant prior" needs ≥2 architectures — ours plus a reimplementation of the Gradual
Fusion Q-Former, which their paper describes in enough detail to rebuild. Budget for that
explicitly; it is the difference between a workshop note and a paper.

**(2) 3D headroom vs a real fusion detector — open, and now cheap.** VLGA has no fusion
baseline at all; LVLDrive reports BEV mIoU on grounding *with the 2D box handed to the
model*, which is not on any ruler a detector is measured on. **There is no common ruler**,
and building one is the contribution — not the (entirely expected) conclusion that a VLM is
worse than BEVFusion.

*Risk:* the headline answer is unsurprising, so the paper must live in the decomposition —
per distance band, per class, per visibility (we have independent per-sensor masks,
`visibility.py`), and the implied ceiling for VLA planning. "How much metric 3D survives a
language-level representation?" is interesting; "is it worse?" is not.

*Do not* quote 0.21 against BEVFusion's 0.7118 NDS. Different metrics, different tasks. The
whole point is that the comparison does not currently exist.

**(3) Spatially-absent LiDAR (§2.6 F4) — most interesting scientifically, most exposed.**
Careful: MoME and MetaBEV already run Limited-FOV robustness (§4.1), so "we test spatial
absence" is taken. What is ours is **spatial absence against an LLM-level fusion**, where
there is a genuine hypothesis rather than a benchmark row: a language-level representation
carries priors that a BEV grid does not, so it may *predict* where LiDAR cannot *measure* —
"LiDAR measures, camera + prior predicts". That is the one claim here that could be a
finding rather than a measurement.

### One prerequisite that is not optional

**The current LiDAR tokeniser is an 8×8 height histogram, and it cannot support any negative
claim.** LVLDrive uses FSDv2 pretrained on nuScenes. If we report "LiDAR tokens do not
help", the first referee comment is "your encoder was a histogram" and they will be right.
Before any negative result about LiDAR utility, switch to `extract.py`'s frozen TransFusion
queries (or FSDv2). The audit's *positive* findings — the constant-prior fraction — are
robust to a weak tokeniser; conclusions about LiDAR's *value* are not.

---

## 6. "Is Qwen-Drive hard to train?" — yes, and the reason is not the gradient

The gradient problem is solved (§3). What remains is worse, and it is worth being precise
because it decides whether to invest here at all.

**There is no training code.** Verified, not assumed:

```bash
grep -rln "def loss|CrossEntropyLoss|FocalLoss|HungarianAssigner|def forward_train|optimizer|AdamW" src/ scripts/
# -> no matches
```

Zero loss functions, no assigner, no optimiser, no dataset class for anything but the six
packed demo frames. To train *anything* against this head at scale you must supply:

* a nuScenes/nuPlan dataset class producing the release's exact packed format (images at
  896×512, ChatML prompt layout, `lidar2img` with the image-scale applied, `lidar2ego`);
* an **18 → 10 occupancy label mapping** (and 10 → our detection classes), which is lossy
  and a judgement call;
* the losses themselves — for detection that means a Hungarian assigner matched to their
  `bbox_coder`, which is guesswork against a checkpoint you cannot see the recipe for.

That is 1–2 weeks of work whose *output* is a training stack for someone else's frozen
head. Set against it: our own repo already has occupancy training (`train_lss.py`), Occ3D
GT, detection heads, seeds, and the label-efficiency curve.

### What Qwen-Drive **is** good for

* a **frozen evaluation instrument** — it gives 3D read-outs (det + occ + map) from an
  LLM's own hidden states, which no other open release does;
* a **camera-only 3D reference** on frames we also have LiDAR for — task (b), the headroom
  measurement;
* the **gate host**: §4's overfit test needs exactly what it has and nothing it lacks.

### The alternatives, honestly

Assessed on the five things that actually cost time: (i) open training loop with losses,
(ii) a data pipeline for a dataset we already hold, (iii) a **3D** read-out so an injected
3D token's effect is measurable in 3D rather than in a text score, (iv) single-GPU with the
backbone frozen, (v) differentiable as released.

| candidate | training code | 3D read-out | verdict for us |
|---|---|---|---|
| **Qwen-Drive-1.0** | ❌ none (verified) | ✅ det + occ + map | frozen instrument, not a training host |
| **OpenDriveVLA** ([repo](https://github.com/DriveVLA/OpenDriveVLA), AAAI 2026) | ❌ "release training scripts" still on the TODO; inference + a 0.5 B checkpoint only | ✅ (mmdet3d-based, nuScenes) | watch it; mmcv/mmdet3d pinned to torch 2.1.2, so a port |
| **LVLDrive / VLGA / VLA-MP** | ❌ no code found | varies | read them, don't wait for them |
| **OpenVLA / π0 (openpi) / SmolVLA / Octo** | ✅ full, LoRA-friendly, single-GPU | ❌ manipulation, no 3D driving read-out | the right answer to "which VLA trains easily", the wrong domain. An `openpi_train` env already exists on this machine if a manipulation detour is ever wanted |
| **build the minimal host ourselves**: frozen Qwen3-VL tap + *our* light 3D head, trained in `ngperception/` | ✅ we write it | ✅ Occ3D, our classes, our seeds and noise floors | **recommended** |

The last row is the recommendation, and it is not a compromise. What makes Qwen-Drive hard
is the missing *training loop and data pipeline* — precisely the two things our own stack
already has and Qwen-Drive's frozen head would force us to rebuild in a foreign label
space. Tapping a frozen Qwen3-VL and attaching a head we already train is a **smaller**
change to code we own. Qwen-Drive then does what it is actually good at: an independent,
strong, frozen reference to measure against — which also keeps §7.9 satisfied (their head
tells us whether there is an effect worth building an apparatus for, before we build one).

---

## 7. Next steps, in order, with kill criteria

**(a) gate — DONE.** Gradient reaches the projector's position. See §3.

**(a′) capacity gate — §4, running.** Two rungs: the 1-frame probe (~85 min), then the
4-arm gate on 6 frames (~5 h).
*Kill:* the 1-frame probe's loss does not move; or the `lidar` arm cannot beat the `none`
baseline on the frames it trains on; or it does but the `shuffle` control matches it. Any of
those means the injection channel is not carrying LiDAR to the BEV head, and the route stops
here.

**(b) headroom.** Qwen-Drive camera-only 3D accuracy vs BEVFusion-LC on the *same* frames.
This is what says whether there is anything for LiDAR tokens to add. Needs the 18→10 label
mapping (§1) and a nuScenes frame packer. *Read it first* — §7.9: measure the size of the
effect before building the perturbation. If the gap is small, the whole direction is
uninteresting no matter how well the projector trains.

**(c) the audit, which is the part that is ours.** Apply §2.6's two-ablation audit to
whatever LiDAR-token model exists — ours or a re-implementation of LVLDrive's injector:
corrupt the point cloud *and* zero the projected tokens, separately. If zeroing the tokens
costs much more than corrupting the points, the model learned a constant, not a reading.
This is the contribution; (a′) and (b) are its preconditions.

**(c′) profile the forward — DONE, and it was one layer.** 99 % of a 92.76 s forward was
`Qwen3_5VisionPatchEmbed`, a Conv3d whose kernel covers its entire input. Replaced by the
bit-identical matmul: **92.76 s → 1.02 s**, so a nuScenes-train epoch's forward goes from
~780 to ~8 GPU-h (§4). Re-measure the training step before planning (c): every timing in
this document that predates the fix is ~99 % that one layer.

**Explicitly not next:** writing a backward for the two CUDA ops (§3.3 removed the need);
building a Qwen-Drive training pipeline (§6); claiming "LiDAR in a VLA" as a method (§5).

---

## 8. Rules this direction added

**Q1. A released "inference-only" model can be non-differentiable in ways nothing tells
you about.** No error is raised: outputs simply arrive with `grad_fn=None`, and a loss on
them fails much later with "element 0 of tensors does not require grad". Before planning
any training around a third-party head, run the one-line check — forward a
`requires_grad=True` tensor and print `out.grad_fn`. `check_grad.py --stage ops` is that
check, and it needs no GPU.

**Q2. When a graph is cut, bisect by submodule before theorising.** `adaptor` and
`vit_neck` returning `AddBackward0` while the head's outputs returned `None` localised the
break to the encoder in one pass, and pointed at the custom ops rather than at the
`@torch.no_grad()` decorators, which were the obvious suspects and were not the cause.

**Q3. Look for the reference implementation before writing the missing kernel.** The first
conclusion here — "the BEV head can only be an evaluation instrument" — was wrong, and it
was wrong because it assumed a missing backward has to be *written*. The release already
contained a pure-torch implementation of the same op, three lines above the dispatcher that
skips it. Cost of the assumption: nearly a pivot in research plan. Cost of checking: one
`grep`.

**Q4. A substituted implementation needs a parity control, at the decision level.** `--stage
parity`: 0.3–1.2 % relative L2 on the logits, **99.98 % argmax agreement**. Relative error
on logits alone would not have been convincing; agreement on the prediction is.

**Q5. `gradient_checkpointing_enable()` does nothing under `.eval()`.** HF gates it on
`self.training`. It reported "enabled" and saved 3.5 of 60 GiB. Frozen parameters do not
require `eval()`; call `train()` and freeze with `requires_grad_(False)`.

**Q7. Profile before optimising, and profile *inside* the module the profile blames.** The
first profile said "the ViT is 99 % of the forward", and the two natural readings of that —
attention, and the gradient checkpointing I had myself enabled on all 24 vision blocks —
were both **wrong**, testable, and tested (blocks: 0.00 s; checkpointing on vs off: 91.85 vs
92.86 s). One more level of labels found it: a single `Conv3d` layer, 91.79 of 92.76 s, fixed
exactly by a matmul for a 91× end-to-end speedup. Three cheap measurements had already
established what it *was not* (not the gradient, not the attention implementation, not its
precision) — and "not X, not Y, not Z" is not a diagnosis. **Never accept a module-level
attribution as a cause; descend until the leaf is arithmetic you can predict the cost of.**
22 GFLOP should not take 91 s, and noticing that gap is what closed this.

**Q0. A random-input control lower-bounds what optimisation can reach; it does not
upper-bound it.** `random vs zero` moved 0.122 % of voxels and was pre-registered as "the
upper bound on what ANY token content can do"; 50 optimisation steps reached 0.585 %, 4.8×
more. In a 2560-dimensional space a random direction is nearly orthogonal to the useful
subspace. **To bound what is reachable, the probe has to be optimised** — a noise probe
answers a different question ("is the model sensitive to arbitrary perturbation?") than the
one being asked ("is there a direction that matters?").

**Q6. In a causal decoder, an injected token can only affect what comes *after* it.**
Obvious in hindsight, and it still cost a run: the tokens were appended at the end of the
prompt precisely because that leaves the image tokens' positions untouched — the property
that also makes them causally invisible to the image tokens, which are the only thing the
downstream head reads. **When a soft prompt has zero gradient, check the ordering before
anything else**, and make the training script refuse to continue on a zero gradient rather
than log it.

---

## 9. Reproduction

```bash
cd DeepDataMiningLearning/bevdet/lidar_tokens
PY=/home/010796032/miniconda3/envs/py310/bin/python
export CUDA_HOME=/data/rnd-liu/cudatk CPATH=$CUDA_HOME/targets/x86_64-linux/include
export PATH=/home/010796032/miniconda3/envs/py310/bin:$PATH

$PY check_grad.py --stage ops                    # which ops lack a backward (no GPU)
$PY check_grad.py --stage llm                    # embedding <- 36 frozen layers   PASS
$PY check_grad.py --stage head --attn kernel     # as released                     FAIL
$PY check_grad.py --stage head --attn torch      # with the fallback               PASS
$PY check_grad.py --stage parity                 # kernel vs fallback fidelity
$PY check_grad.py --stage profile                # where the ~100 s/frame goes

./run_projector_gate.sh          # all four arms in order, resumable, reads the summary
# or one at a time:
# rung 0: how wide is the channel at all (trains nothing, ~9 min)
$PY train_projector.py --mode sensitivity --max-frames 2 --exp-name sensitivity
# rung 1: can a projector use it (one frame, ~85 min)
$PY train_projector.py --inject lidar --max-frames 1 --epochs 50 --lr 3e-3 \
    --exp-name capacity1f
# rung 2: the four arms
$PY train_projector.py --config configs/gate_demo6.yaml --inject none      # baseline
$PY train_projector.py --config configs/gate_demo6.yaml --inject zero      # K null tokens
$PY train_projector.py --config configs/gate_demo6.yaml --inject lidar
$PY train_projector.py --config configs/gate_demo6.yaml --inject shuffle   # the control
```

Every `train_projector.py` run writes `outputs/<timestamp>_<name>_<arm>/` containing
`config.yaml`, `meta.json` (seed, git commit + dirty flag, token geometry, embedding RMS)
and `log.json` (per-epoch loss, per-eval mIoU and per-class IoU).
