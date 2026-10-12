# Tutorial — Qwen-Drive-1.0: run it, plan with it, audit it

A hands-on guide to Alibaba's Qwen-Drive-1.0. It is one VLM (Qwen3.5-4B) with two heads: a
BEV perception head (detection + occupancy + map) and a flow-matching **Planning Expert**
that reads the VLM's attention cache. The guide covers inference in all three modes, how to
build planning scenes the release does not ship, and how to measure which inputs the planner
uses. Everything was run on this machine (H100).

**Companions.** [`TUTORIAL_ALPAMAYO.md`](TUTORIAL_ALPAMAYO.md) is the same walk-through for
NVIDIA's dual-system VLA, and the two share clips, t0 and metrics so they compare directly.
[`QWEN_DRIVE_LIDAR_TOKENS.md`](QWEN_DRIVE_LIDAR_TOKENS.md) is the engineering record for the
perception head (differentiability fix, LiDAR-token injection).
[`TUTORIAL_E2E_DRIVING_MODELS.md`](TUTORIAL_E2E_DRIVING_MODELS.md) §3.8–3.9 is the research log.

**Videos** (local files, see [§7](#7-videos)):
[counterfactual rationales, both models](videos/counterfactual_cot.mp4) ·
[replay: stop sign](videos/rollout_bb912f3d.mp4) ·
[replay: left turn](videos/rollout_6d8b64de.mp4) ·
[replay: pedestrian yield](videos/rollout_4f14b8f4.mp4)

---

## 1. What the model is

```
3 cameras (FRONT, FRONT-LEFT, FRONT-RIGHT) x 4 frames (t-1.5, -1.0, -0.5, 0 s)
   history frames ~320p, current frame ~720p
+ a text instruction: the 4 history poses printed as numbers + "Active navigation command: [...]"
            |
            v
   Qwen3.5-4B VLM  (unchanged architecture; also answers free-form VQA)
            |  REASONING mode: writes a one-sentence rationale first
            |  ("Stop for the stop sign at the intersection.")
            v
   KV cache of the full-attention layers  ----------------+
            |                                             |
   BEV perception head (BEVFormer-style)          Planning Expert (flow matching, 10 steps)
   det + occupancy + map                            reads: the KV cache
                                                         + history poses/vel/acc tensors
                                                         + nav one-hot (nav_mlp)
                                                         + ego_status [vx, vy, ax, ay, driving_command one-hot]
                                                    -> 50 waypoints @ 10 Hz (5 s), x fwd / y left
```

**The nav command reaches the planner by three routes**: the prompt text, the `nav_command`
tensor, and the `driving_command` one-hot inside `ego_status`. The ego history reaches it by
two: the numbers in the prompt, and the tensors. §5 separates these routes.

## 2. Environment and checkpoints

```bash
PY=~/miniconda3/envs/py310/bin/python        # py310 (the mmdet3d env), NOT the Alpamayo venv
REPO=/data/rnd-liu/Others/Qwen-Drive-1.0     # code:    src/qwen_drive/
CKPT=/data/rnd-liu/Others/Qwen-Drive-1.0-4B  # weights: VLM at root, heads in subfolders
```

| subfolder | size | use |
|---|---|---|
| (root) | 8.4 GB | Qwen3.5-4B VLM, shared by every task |
| `perception/` | 0.5 GB | BEV head |
| `planner-sft/` | 2.1 GB | imitation-trained Planning Expert: direct **and** reasoning planning |
| `planner-rl/` | 2.1 GB | reward-optimized; **use in reasoning mode only** (trained only on reasoning rollouts) |

The planner heads ship separately. We fetched them with
`hf download Qwen/Qwen-Drive-1.0-4B planner-sft/config.json planner-sft/model.safetensors planner-rl/config.json planner-rl/model.safetensors --local-dir $CKPT`.
Glob `--include` patterns failed with "File not found"; name the files explicitly.

## 3. First inference — the 91× fix first

```python
import sys, torch
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, "DeepDataMiningLearning/bevdet/lidar_tokens")      # for qwen_drive_grad
from qwen_drive.modeling_qwen_drive import QwenDriveForPlanning
from qwen_drive_grad import set_vision_patch_embed

model = QwenDriveForPlanning.from_pretrained(CKPT, planner=f"{CKPT}/planner-sft",
                                             dtype=torch.bfloat16).cuda().eval()
set_vision_patch_embed(model.vlm, "matmul")     # <- do not skip this
```

**Why `set_vision_patch_embed`.** The released vision patch embedding is a `Conv3d` whose
kernel covers its entire input: a matmul written as a convolution. In bf16 it lands on a
cuDNN path that took **34 s per prefill** here, against **0.4 ms** as a GEMM. That is the same
contraction; the helper verifies it (`max_abs_diff 0.0`). Without it, one audited clip
(~15 prefills) takes more than 20 minutes.

```python
from qwen_drive.scene import CameraFrame, DrivingScene, CAMERA_VIEWS

scene = DrivingScene(
    views={v: [CameraFrame(p) for p in paths[v]] for v in CAMERA_VIEWS},  # 4 frames per view, oldest first
    history=hist,                 # (16, 3) x, y, heading @ 10 Hz, last pose = (0, 0, 0)
    history_velocity=hv,          # (16, 2) body-frame
    history_acceleration=ha,      # (16, 2)
    ego_velocity=hv[-1], ego_acceleration=ha[-1],
    driving_command=[0, 1, 0, 0], # one-hot: left, straight, right, unknown
    nav_command=0,                # 0 straight, 1 left, 2 right  (NB: different order!)
    instruction_text=None)        # None -> the release synthesizes the prompt from the history
out = model.generate_trajectory(scene, mode="reasoning_planning", num_samples=6, seed=42)
out.trajectories                  # (6, 50, 3)
out.reasoning                     # "<think>\n\n</think>\n\nStop for the stop sign at the intersection."
```

Three modes (`InferenceMode`):

| mode | what happens |
|---|---|
| `direct_planning` | the assistant turn is closed empty; the expert plans from the prompt cache |
| `reasoning_planning` | greedy generation of a one-sentence rationale, then the turn is closed and the expert plans from that cache |
| `vqa` | `model.generate_text(images, question)`: the plain VLM, no expert |

⚠ **Two different nav encodings**: `nav_command` 0/1/2 = straight/left/right, but the
`driving_command` one-hot is ordered **left, straight, right**, unknown
(`docs/data.md`). Getting this wrong silently feeds a contradictory command.

## 4. Building PhysicalAI planning scenes (the release ships four)

Qwen-Drive reports PhysicalAI results (ADE@3s 0.37–0.42) but ships no PhysicalAI scene files.
`alpamayo1.5/qwendrive_bridge/build_scenes.py` rebuilds them from raw data, following
`docs/data.md`. It runs in the **Alpamayo venv**, because it needs `physical_ai_av`.

| item | rule |
|---|---|
| cameras | `front_wide_120fov` / `cross_left_120fov` / `cross_right_120fov` → the 3 views |
| frames | t0 −1.5 / −1.0 / −0.5 / 0 s; history resized to 576×320, current to 1280×720 (`CameraFrame.target_size` semantics) |
| history | 16 poses @ 10 Hz in the t0 ego frame, headings wrapped |
| vel / acc | from the egomotion *interpolator*, rotated into each step's body frame (not differentiated poses) |
| guards | reject yaw-rate > 1.2 rad/s or \|acc\| > 1 g |
| nav | ⚠ derived from the GT future (lateral offset at 5 s, ±2 m): the usual open-loop leak, undocumented upstream |

```bash
cd /data/rnd-liu/MyRepo/alpamayo1.5
./a1_5_venv/bin/python qwendrive_bridge/build_scenes.py --n-clips 60           # t0 = 5.1 s
./a1_5_venv/bin/python qwendrive_bridge/build_scenes.py --clips <id,...> \
      --t0-list 2.0,2.25,...,14.5 --out qwendrive_bridge/sweep                  # time sweep for videos
```

For a sweep, each clip × camera is decoded **once** for all timestamps. Decoding per scene
was ~1 scene/min.

**Faithfulness check** (always do this before auditing): on 60 clips at t0 = 5.1 s, ADE@3s
over 6 samples is **0.51** (SFT, reasoning), 0.52 (SFT, direct) and 0.54 (RL). The paper
reports 0.37 / 0.38 / 0.42. We are ~25 % worse, most likely from the nav rule and the t0
choice. Treat our numbers as near-faithful, not exact.

## 5. Auditing the planner

### 5.1 Teacher-forcing a rationale

In reasoning mode the cache is `[prompt | rationale | <|im_end|> \n]`. Teacher-forcing is
one prefill over exactly that sequence. Its last rotary position equals the release's
`prompt_anchor + len(closed_turn)`, so the expert sees your sentence where its own would be:

```python
inputs = model.processor(scene, with_reasoning=True)
ids = inputs["input_ids"][0].tolist() + think_prefix + tok.encode(sentence) \
      + [proc.im_end_id] + proc.newline_ids
forced = dict(inputs, input_ids=torch.tensor([ids]).cuda())
cache, anchor = model._prefill(forced)
traj = model._plan_from_cache(cache, anchor, forced, num_samples=6, num_steps=10, seed=42)
```

`think_prefix` is the empty `<think>\n\n</think>\n\n` block the model always emits. Keep it,
so the injected text sits in the model's own slot. **Parity check:** forcing the model's
*own* rationale reproduces `plan_from_inputs` at **0.016 m against a 0.136 m noise floor**
(0/60 clips above). If your parity is not far below the floor, your forcing is wrong.

### 5.2 KV dropping (mechanism)

The expert attends the whole cache with no mask, and the keys are post-rotary. So deleting
rows from the cache is an exact "cannot read this span" intervention:
`cache = [(k[:, keep], v[:, keep]) for k, v in cache]`.

### 5.3 Results (`run_audit.py`, 60 clips; × the seed-noise floor)

| intervention | × floor |
|---|---|
| ego history frozen (direct mode) | **79.9×** |
| image KV dropped | 23–24× |
| images → grey | 14.3× |
| nav swapped (all three routes) | 10.6×; prompt text only 4.4×, tensors only 6.5× |
| rationale → another clip's | **2.85×** (p 5e-6) |
| rationale → imperative "STOP…" | 3.4× |
| **rationale dropped from the KV** | **1.2× (p 0.53): no effect** |
| direct vs reasoning planning | 1.5× |

The RL planner reproduces every row within ~10 %. A nav swap rewrites the model's own
rationale on 87 % of clips.

### 5.4 Is the effect *semantic*? The signed test (`semantic_cot.py`)

Divergence has no sign. Force sentences in the model's *own* phrasing with a known direction
and measure signed Δ vs its own rationale (6 paired seeds, replicated on a second seed set):

| forced rationale | Δx @5 s | Δy @5 s |
|---|---|---|
| "Slow down to follow the stopped lead vehicle." | −0.88 m (replicate −0.90), 52/60 | 0 |
| "Stop for the stop sign at the intersection." | −1.18 m (−1.20), 57/60 | 0 |
| "Accelerate to target speed along the clear lane markings." | **+1.20 m** (+1.20), 52/60 | 0 |
| "Maintain speed and nudge **left**…" | — | **+7 cm**, 47/60, p 1e-4 |
| "Maintain speed and nudge **right**…" | — | **−5 cm**, 50/60, p 1e-5 |

**The expert obeys the meaning, in both directions, on both axes.** The effect sizes are
7–11× its own seed spread (~0.13 m at 5 s). Noise from unfamiliar text would push every arm
the same way; these arms go in opposite directions.

### 5.5 What this means

- **Obeyed, but not informative.** Dropping its own rationale changes nothing, and direct and
  reasoning planning nearly agree. The rationale is a **steering port**: text you put there is
  followed, but the text the model writes adds almost nothing beyond images, nav and ego state.
- **Versus Alpamayo** (same clips, same sentences): both obey speed text by similar absolute
  amounts; only Qwen-Drive obeys lateral text; Alpamayo's effect hides inside a seed spread
  ~8× larger. See the [counterfactual video](videos/counterfactual_cot.mp4).
- **Hierarchy**: ego ≫ images > nav ≫ rationale, as in Alpamayo.

## 6. Perception head, in one paragraph

The BEV head runs from the same VLM, but as released it is not differentiable: its
deformable-attention op has `grad_fn=None`. `qwen_drive_grad.patch_deformable_attention()`
routes it to the release's own pure-torch fallback. With that fix and ViT checkpointing
(`vlm.train()`, since checkpointing is a no-op under `eval()`), LiDAR tokens can be trained
into the frozen VLM. A **constant** token block reproduces the whole gain, so the VLM does
not read the LiDAR content. Full story, plus the fwIoU / `mask_camera` scoring pitfalls:
[`QWEN_DRIVE_LIDAR_TOKENS.md`](QWEN_DRIVE_LIDAR_TOKENS.md).

## 7. Videos

Generated by `alpamayo1.5/render_videos.py` (py310). They are local files under
`docs/videos/` and are **not** committed: the frames come from the NVIDIA PhysicalAI-AV
dataset.

| video | what it shows |
|---|---|
| [counterfactual_cot.mp4](videos/counterfactual_cot.mp4) ([still](videos/counterfactual_cot.png)) | per clip, then a 60-clip average: Δx(t), Δy(t) of Alpamayo vs Qwen-Drive under the same forced sentences; grey band = own 6-seed spread |
| [rollout_bb912f3d.mp4](videos/rollout_bb912f3d.mp4) ([still](videos/rollout_bb912f3d.png)) | open-loop replay at a stop sign (t0 2–14.5 s, every 0.25 s): camera, both 6-seed fans vs GT, both rationales, ADE@3s over time |
| [rollout_6d8b64de.mp4](videos/rollout_6d8b64de.mp4) ([still](videos/rollout_6d8b64de.png)) | unprotected left turn with a crossing vehicle |
| [rollout_4f14b8f4.mp4](videos/rollout_4f14b8f4.mp4) ([still](videos/rollout_4f14b8f4.png)) | yielding to a crossing pedestrian |

Mean ADE@3s over each replay (51 queries per clip, 6 seeds): stop sign Alpamayo 0.50 /
Qwen-Drive 0.64 m, left turn 0.34 / 0.56, pedestrian 0.35 / 0.31. **Three clips are an
illustration, not a ranking**; the 60-clip numbers in §5 are the measurement.

## 8. Reproduce everything

```bash
cd /data/rnd-liu/MyRepo/alpamayo1.5
PY=~/miniconda3/envs/py310/bin/python
./a1_5_venv/bin/python qwendrive_bridge/build_scenes.py --n-clips 60             # §4
$PY qwendrive_bridge/run_audit.py --n 60 --planner planner-sft                   # §5.3
$PY qwendrive_bridge/run_audit.py --n 60 --planner planner-rl --skip-direct
$PY qwendrive_bridge/analyze.py qwendrive_bridge/outputs/<run_dir>
$PY qwendrive_bridge/semantic_cot.py --n 60 --planner planner-sft                # §5.4
$PY qwendrive_bridge/semantic_cot.py --n 60 --planner planner-sft --seed 1042    # replicate
$PY qwendrive_bridge/sweep_qwendrive.py                                          # videos
$PY render_videos.py rollout --clip bb912f3d --out <docs>/videos/rollout_bb912f3d.mp4
```

Each run writes `qwendrive_bridge/outputs/<timestamp>_<name>/` with `meta.json` (config,
seed, upstream git hash), incremental `rows.jsonl`, `reasoning.json` and `summary.json`.

## 9. Pitfalls we hit

| symptom | cause | fix |
|---|---|---|
| one clip takes > 20 min | Conv3d `patch_embed` on a slow cuDNN path | `set_vision_patch_embed(vlm, "matmul")`, bit-identical |
| `hf download --include "planner-rl/*"` fails | glob not resolved | name the four files explicitly |
| planner ignores or contradicts nav | `nav_command` and `driving_command` use different orders | straight/left/right vs one-hot left/straight/right |
| injected rationale behaves oddly | dropped the `<think></think>` prefix | keep the model's prefix; check parity ≪ floor |
| swap arm reads ≈ 0 on some clips | donor rationale identical to own (common templates) | report the swap arm on differing donors only (59/60) |
| "no gradient" into the perception head | released deformable op has no autograd | `patch_deformable_attention()` |
| checkpointing "does nothing" | module in `eval()` | `vlm.train()` (no dropout in this VLM) |
