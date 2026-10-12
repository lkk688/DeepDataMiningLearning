# Tutorial — NVIDIA Alpamayo 1.5: run it, read it, audit it

A hands-on guide to a dual-system driving VLA: a reasoning VLM (the "brain") plus a
flow-matching action expert (the "cerebellum"). It takes you from a first inference to
*measuring which inputs the planner actually uses*. Everything here was run on this machine
(H100, 60 PhysicalAI-AV clips).

**Companions.** [`TUTORIAL_QWEN_DRIVE.md`](TUTORIAL_QWEN_DRIVE.md) is the same walk-through
for the other open dual-system stack. [`TUTORIAL_E2E_DRIVING_MODELS.md`](TUTORIAL_E2E_DRIVING_MODELS.md)
§3 is the research log with every number and every correction.

**Videos** (local files, see [§7](#7-videos)):
[counterfactual rationales, both models](videos/counterfactual_cot.mp4) ·
[replay: stop sign](videos/rollout_bb912f3d.mp4) ·
[replay: left turn](videos/rollout_6d8b64de.mp4) ·
[replay: pedestrian yield](videos/rollout_4f14b8f4.mp4)

---

## 1. What the model is

```
inputs     4 cameras x 4 timesteps (1080p)       ego history: 16 poses @ 10 Hz (xyz + rotation)
             |                                             |
             v                                             |
  VLM  Cosmos-Reason2-8B (Qwen3-VL family)   <- System 2   |
             |  generates a Chain-of-Causation (CoC) sentence,
             |  e.g. "Slow down due to the speed bump ahead"
             v                                             |
  KV cache of [prompt | images | CoC | <traj_future_start>]|
             |                                             |
             v                                             v
  action expert  (hidden 2048, 16 heads, NON-causal)   <- System 1
     flow matching (Euler), attends the VLM KV cache
             |
             v
  64 actions (UnicycleAccelCurvature, dt 0.1 s = 6.4 s)
             |  integrated from ego_history[:, -1]
             v
  trajectory xyz (64 x 3), t0 ego frame: x forward, y left
```

Three things to take from the diagram. They are exactly what the audit in §5 tests.

1. **The "brain → cerebellum" channel is the KV cache.** The expert does not receive the CoC
   as a string. It attends the VLM's keys and values, and the generated rationale is part of
   them.
2. **The ego history enters twice.** It enters as tokens fused into the prompt
   (`fuse_traj_tokens`), and as the **integration anchor** (actions are unicycle controls
   integrated from the last history pose).
3. **Sampling is stochastic twice.** The CoC is sampled (top-p 0.98, T 0.6) and the flow
   starts from Gaussian noise. §5.4 shows why this matters for any measurement.

## 2. Environment

```bash
cd /data/rnd-liu/MyRepo/alpamayo1.5
V=./a1_5_venv/bin/python        # py 3.12, torch 2.8.0+cu128, transformers 4.57.1
                                # do NOT use py310 here; the Qwen-Drive tutorial uses py310
```

| item | where |
|---|---|
| weights (20 GB, 5 shards) | `~/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B` |
| data | `/data/rnd-liu/Datasets/PhysicalAI-AV` (+ `clip2chunk.json`); streamed from HF if a clip is missing |
| demo clip list | `notebooks/clip_ids.parquet` (1181 clips) |
| cost | load ~84 s / 20.6 GiB; one inference ~7 s, 21.6 GiB peak |

## 3. First inference

```python
import torch
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5 import helper

model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16,
                                    attn_implementation="sdpa").to("cuda")   # (1)
proc = helper.get_processor(model.tokenizer)

data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)                   # (2)
msgs = helper.create_message(data["image_frames"].flatten(0, 1),
                             camera_indices=data["camera_indices"])
tok = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                               continue_final_message=True, return_dict=True,
                               return_tensors="pt")
inp = helper.to_device({"tokenized_data": tok,
                        "ego_history_xyz": data["ego_history_xyz"],
                        "ego_history_rot": data["ego_history_rot"]}, "cuda")

torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    xyz, rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=inp, top_p=0.98, temperature=0.6, num_traj_samples=1,
        max_generation_length=256, return_extra=True)
print(extra["cot"][0][0][0])          # e.g. "Nudge left due to the vehicle encroaching ..."
gt = data["ego_future_xyz"][0, 0]     # same frame as xyz
```

(1) The config requests `flash_attention_2`. Use **`sdpa`** unless flash-attn was built for
this toolkit.
(2) `t0_us` is the query time inside the ~20 s clip. The history covers t0−1.5 s … t0, and the
future to be predicted is t0 … t0+6.4 s.

Smoke test from the shell: `$V audit_ego.py --n-clips 4 --arms real`.

### 3.1 ⚠ One clip is not a measurement

Our first clip scored L2@3s **0.110 m**, an order of magnitude better than nuScenes SOTA.
Over 40 clips the same model scores **1.901 m** (p10–p90: 0.46–6.34 m). Always report a
distribution.

## 4. Driving it with a navigation command

`alpamayo1_5.nav_utils.compare_nav_conditions(model, processor, data, nav_text, ...)` runs
three conditions: with the nav text, without it, and with a counterfactual direction.
`notebooks/nav_demo_samples.json` carries maneuver-labelled examples. Swapping the direction
moves the trajectory **3.44×** the sampling-noise floor, and makes it *worse* against GT. So
nav is causally used.

## 5. How to audit a dual-system VLA — the method, step by step

The question is not "how good is it" but **which input does the planner actually read?**
Answer it by **replacing one input** with an uninformative or wrong one, keeping everything
else fixed, and measuring what moves.

### 5.1 Replace inputs (`audit_ego.py`)

| arm | replaced by | L2 vs GT (40 clips) |
|---|---|---|
| real | — | 2.52 m |
| ego frozen | constant pose | **19.84 m (8.5×)** |
| ego shuffled | another clip's history | 15.07 m |
| cameras blind | mid-grey images | 4.45 m (2.3×; still beats `real` on 13/40) |
| cameras swapped | another clip's cameras | 8.72 m |

The ego history dominates the cameras.

### 5.2 Teacher-force the reasoning (`audit_channels.py → traj_forced_cot`)

To ask "does the expert read the rationale?", bypass generation and prefill the VLM on
`[prompt | YOUR sentence | <traj_future_start>]`, then run the expert on that cache:

```python
full = torch.cat([ids, tokenizer(cot_text).input_ids, [[traj_future_start]]], dim=1)
cache = model.vlm(input_ids=full, use_cache=True, **td).past_key_values
pos, attn = model._build_expert_pos_ids_and_attn_mask(offset=..., kv_cache_seq_len=prefill, ...)
def step(x, t):                         # one flow-matching step of the expert
    o = model.expert(inputs_embeds=model.action_in_proj(x, t), position_ids=pos,
                     past_key_values=cache, attention_mask=attn, is_causal=False)
    cache.crop(prefill)                 # restore the cache: it is reusable across seeds
    return model.action_out_proj(o.last_hidden_state[:, -nd:])
act = model.diffusion.sample(batch_size=1, step_fn=step, device="cuda")
xyz, _ = model.action_space.action_to_traj(act, ego_xyz[:, -1], ego_rot[:, -1])
```

`semantic_cot_alpamayo.prefill_forced()` wraps this and returns a `sample(seed)` closure.
One prefill serves any number of seeds, and we verified the result is bit-identical to a
fresh prefill.

### 5.3 Block attention (`audit_mech.py`)

Put `-inf` into the expert's 4-D mask over a token span of the cache. That removes what the
expert can *read* without changing what the VLM computed. Result (25 clips): blocking all
image tokens gives **5.2×** the floor; blocking the CoC gives 0.21× (nothing); blocking the
history tokens gives 0.99× (nothing). The ego effect comes from the integration anchor, not
from attention.

⚠ Masking one camera out of four is **not** a valid positive control (0.003 m): the other
cameras carry the same scene. Mask all images.

### 5.4 Measure the right thing: signed, paired, multi-seed

This is the lesson that cost us a wrong conclusion.

- **L2 vs GT is the wrong metric** for "is X used?". It cannot distinguish "unchanged" from
  "changed but equally wrong". Use trajectory-to-trajectory divergence.
- **Divergence needs a noise floor.** Use the same input with a different seed. Alpamayo's
  floor is large: ~0.8 m mean, ~1 m at 5 s.
- **But unsigned divergence vs the floor is under-powered.** It asks "is the effect bigger
  than one sample's noise?". Our first CoT audit found every rationale swap *below* the floor
  (0.12–0.6×) and concluded "the CoT is not read". **That was wrong.**
- **The signed test** (`semantic_cot_alpamayo.py`): force sentences with a known direction,
  6 seeds per arm *paired with the same seeds* on the model's own rationale, and measure
  signed Δx (forward) / Δy (left) at 5 s. Then rerun on an **independent seed set**.

| forced rationale (same text given to Qwen-Drive) | Δx @5 s | Δy @5 s |
|---|---|---|
| "Stop for the stop sign at the intersection." | **−0.91 m** (replicate −0.95), 49/60 clips | 0 |
| "Slow down to follow the stopped lead vehicle." | −0.62 m (−0.66), 53/60 | 0 |
| "Accelerate to target speed along the clear lane markings." | **+0.69 m** (+0.62), 58/60 | 0 |
| "Accelerate since the lane is clear ahead" (its own register) | +0.78 m (+0.70), **60/60** | 0 |
| "…nudge left / right…" (both registers) | small | **none, p > 0.1** |

**Alpamayo's expert follows the rationale on the speed axis, but not laterally.** The effect
is real (p < 1e-8, replicated) and smaller than its own sampling spread. That is why the
unsigned test missed it.

**Trap — common random numbers.** A seed yields the same noise tensor on every clip, so a
*seed set* carries a shared bias. Our "different seed" arm drifted **+1.09 m on 55/60 clips**,
then **−0.97 m** when the seed set changed. It is fine as a magnitude floor, but invalid as a
signed test. Paired arms cancel the bias; only an independent-set replicate shows an effect
is not seed-specific.

## 6. What we learned about Alpamayo

| channel | effect on the plan |
|---|---|
| ego history | dominant (8.5× L2; acts through the integration anchor) |
| images | large (5.2× when all image KV is blocked) |
| nav command | causal, 3.4× floor, 13× the CoT on the same clips |
| its own CoC | **obeyed on speed** (±0.7–0.9 m), **ignored laterally**, and removing it changes nothing: a steering port, not an information source |

## 7. Videos

Generated by `alpamayo1.5/render_videos.py` (py310). They are local files under
`docs/videos/` and are **not** committed: the frames come from the NVIDIA PhysicalAI-AV
dataset, whose license does not allow redistribution in this public repo.

| video | what it shows |
|---|---|
| [counterfactual_cot.mp4](videos/counterfactual_cot.mp4) ([still](videos/counterfactual_cot.png)) | 12 clips, then a 60-clip average. For each, Δx(t) and Δy(t) of both models under the same forced sentences. Grey band = each model's own 6-seed spread. Alpamayo's arms stay inside its band; Qwen-Drive's leave it, and split laterally for nudge left/right |
| [rollout_bb912f3d.mp4](videos/rollout_bb912f3d.mp4) | open-loop replay at a stop sign, t0 = 2–14.5 s every 0.25 s. Both models' 6-seed fans vs GT, both rationales as captions, ADE@3s over time |
| [rollout_6d8b64de.mp4](videos/rollout_6d8b64de.mp4) | same, an unprotected left turn with a crossing vehicle |
| [rollout_4f14b8f4.mp4](videos/rollout_4f14b8f4.mp4) | same, yielding to a pedestrian |

Mean ADE@3s over each replay (51 queries per clip, 6 seeds): stop sign Alpamayo 0.50 /
Qwen-Drive 0.64 m, left turn 0.34 / 0.56, pedestrian 0.35 / 0.31. **Three clips are an
illustration, not a ranking**; the 60-clip numbers in §5 are the measurement.

Open-loop replay means each frame is an independent query on logged data. The car never
executes the plan. That is **not** closed-loop driving (see `closedloop_occ/` for that).

## 8. Reproduce everything

```bash
cd /data/rnd-liu/MyRepo/alpamayo1.5
V=./a1_5_venv/bin/python
$V audit_ego.py --n-clips 40 --arms real,frozen,shuffled --out audit_ego.json      # §5.1
$V audit_ego.py --n-clips 40 --arms real,blind,camswap  --out audit_cam.json
$V audit_channels.py --mode both                                                  # §4/§5.2
$V audit_mech.py                                                                  # §5.3
$V semantic_cot_alpamayo.py --n 60                                                # §5.4
$V semantic_cot_alpamayo.py --n 60 --seed 1042                                    # replicate
# videos (needs the Qwen-Drive runs too, see TUTORIAL_QWEN_DRIVE.md §8)
$V qwendrive_bridge/build_scenes.py --clips <id,...> --t0-list 2.0,2.25,...,14.5 --out qwendrive_bridge/sweep
$V sweep_alpamayo.py
P=~/miniconda3/envs/py310/bin/python
$P render_videos.py counterfactual --out <docs>/videos/counterfactual_cot.mp4
$P render_videos.py rollout --clip bb912f3d --out <docs>/videos/rollout_bb912f3d.mp4
```

Every script writes a timestamped directory under `outputs/` with `meta.json` (config,
seed) and incremental `rows.jsonl`, so a crash never loses a run.

## 9. Pitfalls we hit (so you do not)

| symptom | cause | fix |
|---|---|---|
| model load fails on attention | config asks for flash-attn-2 | `attn_implementation="sdpa"` |
| two arms identical to 3 decimals | donor clip `sel[i+1]` not yet generated, silent fallback | use `sel[i-1]`; treat identical numbers as a bug signal |
| "CoT not read" | unsigned divergence vs a 1 m floor is under-powered | signed paired multi-seed test plus seed-set replicate |
| noise arm has a significant sign | common random numbers across clips | use it as a magnitude only |
| one-camera mask does nothing | cameras are redundant | mask all image tokens |
| stunning single-clip result | an easy clip | ≥40 clips, report the distribution |
