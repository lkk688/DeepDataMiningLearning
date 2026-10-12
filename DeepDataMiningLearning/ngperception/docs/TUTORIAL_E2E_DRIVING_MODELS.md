# Running and auditing end-to-end driving models — a tutorial

**What this is.** A runnability audit plus recipes for the end-to-end / VLA driving models we
can actually execute on this machine, and one evaluation axis worth spending GPU on. It is
deliberately *not* a survey benchmark: most published systems in this space cannot be run at
all (see Tier 4), so "evaluate them all" is not an option, and pretending otherwise produces
a tutorial nobody can follow.

**Companions.** [`AUDIT_3D_IN_DRIVING_VLA.md`](AUDIT_3D_IN_DRIVING_VLA.md) — the synthesis and
the survey of *where* 3D enters. [`QWEN_DRIVE_LIDAR_TOKENS.md`](QWEN_DRIVE_LIDAR_TOKENS.md) —
the engineering record for Qwen-Drive.
**Step-by-step tutorials:** [`TUTORIAL_ALPAMAYO.md`](TUTORIAL_ALPAMAYO.md) ·
[`TUTORIAL_QWEN_DRIVE.md`](TUTORIAL_QWEN_DRIVE.md) · comparison videos under [`videos/`](videos/)
(local only; PhysicalAI-AV frames are not redistributable).

---

## 1. Runnability audit — verified on disk, not from papers

| tier | model | what we verified |
|---|---|---|
| **1 — runs today** | **Alpamayo 1.5 10B** (NVIDIA) | weights **20 GB complete** (5 shards, 0 incomplete blobs) at `~/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B`; venv `alpamayo1.5/a1_5_venv` (py 3.12.14, torch 2.8.0+cu128, tf 4.57.1); **PhysicalAI-AV data local**. Inference **7.1 s, 21.6 GiB** |
| | **Qwen-Drive-1.0** (Alibaba) | runs; det + occ + map; needed a differentiability fix and a **91× forward fix** |
| | **our own stack** | BEVFusion-LC 0.7118 / LiDAR-only 0.6922 NDS; occupancy 0.302 cam / **0.558** fused; NF-Bench + AlpaSim closed loop |
| **2 — cloned, unverified** | UniAD, StreamPETR, PanoOcc, OPUS, Diffusion-Planner, pluto, WA-JEPA, Cosmos3-Nano/Edge | present under `Others/`; weights and envs **not** checked — do that before planning anything on them |
| **3 — code exists, not local** | OmniDrive (NVlabs, + TensorRT on DRIVE AGX Thor), OpenDriveVLA (inference + 0.5 B ckpt; **training scripts still TODO**), Senna | a download/port, not a run |
| **4 — cannot be run** | **DriveVLM / DriveVLM-Dual** ("lacks accessible code or inference pipelines"), LVLDrive, VLGA, OccVLA | paper-only. Any claim about them is a claim about their tables |

**The Tier-4 row is the reason this is not a benchmark paper.** Four of the most-cited systems
in the 3D-in-VLA literature cannot be executed by anyone outside their labs.

---

## 2. Alpamayo 1.5 — the reference dual-system VLA

### 2.1 What it is

```
VLM backbone   qwenvl3 / nvidia/Cosmos-Reason2-8B          <- System 2, the "brain"
action expert  hidden 2048, 16 heads, non-causal attention <- System 1, the "cerebellum"
generator      FlowMatching (euler)
action space   UnicycleAccelCurvature, 64 waypoints, dt 0.1 -> 6.4 s
inputs         4 cameras x 4 timesteps (1080p) + camera_indices
               + ego_history_xyz / ego_history_rot (16 steps @ 10 Hz)
outputs        trajectory (64 waypoints) + a Chain-of-Causation text trace
```

Same family as GR00T N1 (Eagle-2 VLM at 10 Hz + diffusion-transformer action expert at
120 Hz, System 1 cross-attending to System 2's tokens) and π0. **No LiDAR input** — consistent
with Alpamayo-R1, whose inputs are multi-camera video, calibration and ego-motion even though
the same programme's open dataset ships a 360° LiDAR on every clip.

### 2.2 Running it

```bash
cd /data/rnd-liu/MyRepo/alpamayo1.5
V=./a1_5_venv/bin/python                    # py3.12 / torch 2.8; do NOT use py310 here
$V audit_ego.py --n-clips 4 --arms real     # smoke test, ~7 s/clip after load
```

Gotchas, all hit and fixed:

* the config asks for `flash_attention_2`; pass **`attn_implementation="sdpa"`** unless
  flash-attn has been built against this toolkit,
* model load is ~84 s and 20.6 GiB before any input,
* `load_physical_aiavdataset(clip_id)` streams from HF if a clip is not local; the local copy
  is `/data/rnd-liu/Datasets/PhysicalAI-AV` with `clip2chunk.json`,
* `notebooks/clip_ids.parquet` lists 1181 demo clips.

### 2.3 ⚠ One clip is not a measurement

The first clip we ran gave **L2 mean 0.375 m, @3s 0.110 m** — an order of magnitude better
than nuScenes open-loop SOTA, and we briefly wrote it up that way. Over **40 clips** the same
model scores **mean 2.523 m, @3s 1.901 m**, with a p10–p90 spread of **0.46–6.34 m**. Clip 774
was simply an easy one. Quote the distribution, never a clip.

---

## 3. The evaluation axis worth spending GPU on

Not "which model is best" — the models consume different sensors, predict different things
and are trained on different data, so a leaderboard between them would measure the datasets.
The axis that *is* comparable, and that nothing in the literature reports, is:

> **Which inputs does each model actually use?**

The method is the two-ablation audit: don't remove a component, **replace an input** with one
that carries no information (constant) or wrong information (another clip's), and keep
everything else fixed. Six papers that inject 3D into a language model run no such control,
and OccVLA's four ICLR-2026 reviewers (all confidence 5) did not ask for one.

### 3.1 Result — Alpamayo 1.5, ego-history audit (40 clips, 160 runs)

Only `ego_history_xyz` / `ego_history_rot` change; cameras, seed and decoding are fixed.

| arm | mean L2 | median | @1s | @2s | @3s | final |
|---|---|---|---|---|---|---|
| **`real`** | **2.523** | 1.252 | 0.153 | 0.812 | 1.901 | 6.856 |
| `shuffled` — another clip's ego history | 15.072 | 11.685 | 4.920 | 9.695 | 14.352 | 28.210 |
| `frozen` — constant pose, zero ego information | 19.836 | 18.417 | 7.080 | 13.364 | 19.040 | 36.058 |

**The ego-history input is genuinely used**: removing it degrades L2 by a median **8.47×**, and
it beats `real` on only 4 of 40 clips. Wrong ego history (15.07) is *better* than none (19.84)
— a wrong but kinematically plausible prior still beats a degenerate stationary one.

*Design note:* a fourth arm, `static` (hold the t0 pose), was **identical to `frozen` on 40/40
clips**, because the ego history is expressed relative to t0 and its last step already is
(0, identity). It is kept only as a self-check that two code paths agree; it is not an
independent arm.

### 3.2 Why this result matters more than it looks

**The same instrument returns opposite verdicts on two models, which is what makes it an
instrument rather than a machine that prints "not used".**

| model | channel audited | verdict |
|---|---|---|
| Qwen-Drive-1.0 | LiDAR tokens at the frozen VLM's input | **not used** — a constant, information-free token block reproduces the entire gain, and beats the real cloud on held-out frames |
| Alpamayo 1.5 | ego history into the action expert | **used** — 8.5× degradation without it |

A control that only ever says "the input is decorative" would be worthless. This one has a
positive control, from a different model, on the same principle.

### 3.3 The mirror question — are the cameras used?

Same design, 40 clips, only the images change:

| arm | what is replaced | mean L2 | @1s | @3s | median ratio | beats `real` |
|---|---|---|---|---|---|---|
| `real` | nothing | **2.523** | 0.153 | 1.901 | 1.00× | 0/40 |
| `frozen` | ego history → constant | 19.836 | 7.080 | 19.040 | **8.47×** | 4/40 |
| `shuffled` | ego history → other clip | 15.072 | 4.920 | 14.352 | 6.61× | 3/40 |
| `blind` | cameras → mid-grey | 4.450 | 0.317 | 3.063 | **2.25×** | **13/40** |
| `camswap` | cameras → other clip | 8.717 | 0.561 | 6.141 | 4.91× | 4/40 |

**The ego history matters 3.8× more than the cameras** (8.47× vs 2.25×), and **blinding every
camera still beats the unmodified model on 13 of 40 clips**.

A clean asymmetry worth keeping: *wrong* ego history is **better** than none (15.07 < 19.84),
while a *wrong* scene is **worse** than none (8.72 > 4.45). Zeroing the ego history
manufactures a degenerate "stationary vehicle" prior; a wrong-but-kinematically-plausible
history is still a usable one. A wrong scene, by contrast, actively steers the model for
something that is not there, whereas grey images let it fall back on the ego prior — which is
still correct.

Under `blind` the Chain-of-Causation collapses to one constant string, verbatim across clips:

```
blind : "Keep lane since the lane is clear ahead"   (every clip)
real  : "Nudge left due to the vehicle encroaching from the right"
        "Nudge left due to the open door of a stopped vehicle"
        "Accelerate to clear the intersection because of a traffic signal"
```

So the *reasoning trace* is genuinely perception-driven while the *trajectory* barely moves.
That motivated the next audit.

### 3.4 The brain → cerebellum channel, audited directly

**Where the channel is, in code.** `sample_trajectories_from_data_with_vlm_rollout` first lets
the VLM generate the CoT autoregressively, then denoises the action expert with
`past_key_values = vlm_outputs.past_key_values` — so the expert cross-attends to a KV cache
that *contains the generated reasoning tokens* — and finally integrates the trajectory from
`ego_history[:, -1]` through a unicycle action space. The reasoning trace is not side-channel
commentary; it is literally in the cache the controller reads.

**The experiment.** Teacher-force the reasoning text: identical images, identical ego history,
**only the CoT differs** (`audit_cot.py`). 30 clips, 150 runs.

| arm | the reasoning given to the expert | mean L2 | median | vs `real` | better | Wilcoxon vs `real` |
|---|---|---|---|---|---|---|
| `rollout` | the model's own, sampled | 2.141 | 1.137 | 1.11× | 13/30 | — |
| **`real`** | teacher-forced, same text | **1.777** | 1.525 | 1.00× | — | — |
| `swap` | **another clip's** reasoning | 1.708 | 1.509 | 0.95× | 18/30 | p = 0.088 |
| `const` | "Keep lane since the lane is clear ahead" | 1.715 | 1.509 | 0.95× | 18/30 | p = 0.145 |
| `contradict` | **"Brake hard for the pedestrian ahead"** | 1.490 | 1.295 | 0.89× | 19/30 | p = 0.061 |

**No substitution degrades the trajectory.** Not another clip's reasoning, not a constant
string, not one demanding emergency braking. All three trend *better* than the real CoT and
none is significant at 0.05.

**Effect size, the number to quote:**

```
removing the ego history   17.31 m   (mean L2 penalty)
changing the CoT content    0.68 m   (mean per-clip spread across the four arms)
                           ------
                              26x
```

**The reasoning trace — the artefact marketed as making decisions interpretable and
auditable — has ~26× less influence on the executed trajectory than the ego history does.**

Two honesty notes. (1) `contradict` trending *better* (p = 0.061) is unexplained; "brake
hard" likely yields shorter, more conservative trajectories that happen to score better, but
this is a guess and the result is not significant — it should be reported as "no degradation",
not as "contradiction improves driving". (2) `rollout` (2.141) is worse than `real` (1.777)
and differs on 27/30 clips purely because rollout samples at temperature 0.6 while
teacher-forcing is deterministic — **sampling variance alone is a larger effect than the
entire reasoning content.**

### 3.5 ⭐ Which channel does the controller actually read?

§3.4 measured each arm by L2 against ground truth, and that was **the wrong quantity**: it
cannot separate "the reasoning did not change the trajectory" from "it changed it, and both
are equally far from GT". The direct measurement is the **distance between the predicted
trajectories themselves**, and it needs a **sampling-noise floor** — the same input rerun at
a different seed — or a sub-metre divergence cannot be told from temperature-0.6 jitter.

Two channels, one ruler. CoT arms are teacher-forced (identical images, identical ego
history, only the text differs); nav arms use the release's own counterfactual utility
`nav_utils.compare_nav_conditions`, which ships a `swap_direction` helper.

| intervention | divergence (m) | median | in noise-floor units | below floor | Wilcoxon |
|---|---|---|---|---|---|
| **nav: direction swapped** | **2.755** | 2.109 | **3.44×** | — | 3.8e-06 |
| **nav: removed** | 2.027 | 1.451 | 2.53× | — | 3.8e-06 |
| *CoT: same text, different seed* — **THE FLOOR** | *0.801* | *0.730* | *1.00×* | — | — |
| CoT: imperative command | 0.479 | 0.348 | 0.60× | 31/40 | 1.4e-03 |
| CoT: contradictory rationale | 0.347 | 0.222 | 0.43× | 36/40 | 7.5e-06 |
| CoT: another clip's reasoning | 0.248 | 0.107 | 0.31× | 37/40 | 5.8e-07 |
| CoT: constant string | 0.185 | 0.089 | 0.23× | 39/40 | 1.6e-10 |

```
nav direction swapped  /  CoT replaced by another clip's  =  11.1x
```

**Every CoT intervention moves the trajectory less than simply re-running the sampler does**,
significantly so (`const`: 39 of 40 clips below the floor, p = 1.6e-10). The nav channel,
in the same model and the same forward pass, moves it **3.44×** the floor — and *causally*:
swapping the direction makes the trajectory worse against GT (2.223 → 2.727), i.e. the model
follows the wrong instruction.

**The falsification arm failed to falsify.** `imperative` ("STOP. Brake to zero immediately.")
was added to narrow the claim to "*declarative* reasoning is not consumed" — a direct order
should steer if anything does. It reaches 0.60× the floor. So the claim does not narrow:

> **Structured intent is consumed; the natural-language explanation is not.** No text in that
> slot — rationale or command — steers the trajectory as much as changing the random seed.

For a system whose stated value is "reasoning traces that make each decision interpretable
and auditable", this is safety-relevant: **the trace explains something other than the
decision that is executed.**

> ⚠ **CORRECTED 2026-09-23 by §3.9 — the quoted claim above is too strong.** A *signed*,
> paired test on 60 clips with 6 seeds, replicated on an independent seed set, shows that
> Alpamayo's expert **does** follow the rationale's meaning **longitudinally**: "stop"
> −0.9 m, "accelerate" +0.7 m at 5 s, p < 1e-8. It does **not** follow it laterally. The
> unsigned divergence-vs-floor test above had **too little power**: Alpamayo's own sampling
> spread (~1 m at 5 s) is larger than the CoT effect, so a real, systematic effect read as
> "below the floor". What survives from this section: the CoT effect is **smaller than
> Alpamayo's own sampling noise**, and far smaller than nav (13.2×) or ego history.

⚠ **A bug caught by the numbers, and worth keeping.** In the first run `swap` and `const`
came out **identical to three decimals** (0.185 / 0.089). That tell was real: the donor was
`sel[i+1]`, but clips are processed in order, so `real_cot` only ever held `0..i` and the
lookup silently fell back to the constant string on **39 of 40 clips**. Fixed to `sel[i-1]`
(now 24 distinct donor texts over 37/40 clips); the corrected arm is 0.248 m, and the
conclusion never depended on it since `const` / `contradict` / `imperative` were always
valid. **Two arms agreeing to three decimals is not a finding, it is a symptom.**

---

### 3.6 The same comparison, paired within clips

The 11.1× in §3.5 compared two channels measured on **different clip sets** — 19 nav-demo
samples for nav, 40 parquet clips for CoT. That is not a paired comparison, and a reviewer
would say so. `audit_channels.py --mode both` runs both channels on the **same** 19
maneuver-labelled clips:

| channel / intervention | divergence (m) | median | noise-floor units | vs floor |
|---|---|---|---|---|
| **nav: direction swapped** | **2.755** | 2.109 | **1.98×** | p = 1.1e-02 |
| nav: removed | 2.027 | 1.451 | 1.45× | p = 0.20 |
| *CoT: same text, new seed — the floor* | *1.394* | *1.318* | *1.00×* | — |
| CoT: contradictory rationale | 0.399 | 0.306 | 0.29× | p = 1.9e-05 |
| CoT: imperative command | 0.328 | 0.195 | 0.24× | p = 1.1e-05 |
| CoT: another clip's reasoning | 0.208 | 0.098 | 0.15× | p = 3.8e-06 |
| CoT: constant string | 0.168 | 0.083 | 0.12× | p = 3.8e-06 |

```
nav swapped / CoT swapped, SAME clips  =  13.2x
nav > CoT on 19/19 clips;  every CoT arm below the floor on 18/19
```

Cross-set 11.1× and paired 13.2× agree — the gap was closed and the number did not soften.
Note the floor here is **1.394 m** against 0.801 m on the generic slice: turns and yields
carry more sampling variance, and the CoT interventions are still only 0.12–0.29× of it.

A stratification of the 40-clip run by whether the model's own CoT names a reason to deviate
(29 "decision-critical" vs 11 routine) points the same way, and harder: on the
decision-critical clips the CoT interventions are **0.17–0.45×** the floor against 0.61–0.93×
on routine ones. *Caveats: the split is a keyword match on the model's own text, not a scenario
label, and constrained scenes (a lead vehicle, a speed bump) narrow the feasible trajectory set
for any conditioning — normalising by the per-clip floor controls for that only partly.*

### 3.7 Mechanism — what the action expert attends to (`audit_mech.py`)

§3.4–3.6 replaced *inputs*. This blocks *attention*: the action expert's 4-D mask gets `-inf`
on a token span of the VLM KV cache, so the expert cannot read that span while everything
else (including what the VLM computed from it) stays intact. 25 clips, same noise-floor
protocol (floor 0.680 m).

| blocked span | divergence (m) | × floor |
|---|---|---|
| **all image tokens** | **3.531** | **5.19×** |
| ego-history tokens | 0.674 | 0.99× |
| reasoning (CoT) tokens | 0.140 | 0.21× |

- The expert **reads the images directly**, not through the reasoning. Blocking the CoT is
  indistinguishable from resampling the noise.
- Blocking the history *tokens* does nothing, but freezing ego history (§3.1) is 8.47×. The
  ego effect therefore does not travel through attention. It travels through the
  **integration anchor**: the trajectory is integrated from `ego_history[:, -1]`.
- ⚠ The first positive control (mask one camera's image tokens, equal length) gave 0.003 m.
  That is not evidence of anything, because the other cameras carry the same scene. The
  valid positive control is `mask_all_img`. Assertions in the script now check that every
  masked span is non-empty.

### 3.8 ⭐ The second dual-system stack — Qwen-Drive obeys its reasoning (and see §3.9: so does Alpamayo, longitudinally)

Same instrument, same PhysicalAI clips, same t0 = 5.1 s (`alpamayo1.5/qwendrive_bridge/`).
The release ships no PhysicalAI scenes, so they are rebuilt from raw data by
`build_scenes.py`: 3 cameras × 4 frames at t−1.5…0 s, 320p history / 720p current, egomotion
vel/acc in the body frame, nav derived from the GT lateral offset at 5 s (±2 m). 60 clips
(45 straight / 8 left / 7 right). The prompt comes from the release's own
`DrivingScene.instruction()`. Six samples per arm (seeds 42…47, the paper's protocol), paired
sample-by-sample; floor = seeds 1042…1047.

**Faithfulness.** ADE@3s is **0.51** (SFT, reasoning), 0.52 (SFT, direct) and 0.54 (RL). The
paper reports 0.37 / 0.38 / 0.42 on the 644 split and 0.42 on its 700-frame subset. We are
~25 % worse, most likely from the undocumented nav rule and t0 choice, so these results are
*near*-faithful, not exact. Teacher-forcing parity: forcing the model's own rationale
reproduces the release's `plan_from_inputs` at **0.016 m vs a 0.136 m floor (0/60 above)**.

**Channel audit** (`run_audit.py`; SFT, with RL in brackets; × floor):

| intervention | reasoning mode | | intervention | direct mode |
|---|---|---|---|---|
| CoT → another clip's (donor differs) | **2.85×** (2.64) p 5e-6 | | nav: prompt text only | 4.39× |
| CoT → contradictory | 2.37× (2.20) | | nav: tensors only | 6.45× |
| CoT → imperative | 3.41× (3.11) | | nav: all three routes | 10.55× |
| CoT → constant "Keep lane…" | 1.57× (1.43) p 0.07 | | images → gray | 14.3× |
| **CoT tokens dropped from KV** | **1.21× (1.10) p 0.53** | | image KV dropped | 23.0× |
| image KV dropped | 24.3× (24.4) | | ego history frozen | **79.9×** |
| nav swapped, CoT held fixed | 9.72× (10.17) | | | |
| nav swapped, CoT regenerated | 11.60× (11.72) | | | |

nav / CoT on the same 59 clips = **3.46×** (RL 3.90×) against Alpamayo's 13.2×. A nav swap
rewrites the model's own rationale on 87 % of clips.

**Is the CoT effect semantic?** Divergence has no sign, so `semantic_cot.py` measures one.
It teacher-forces rationales in the model's *own* sentence templates and scores signed
progress against the model's own rationale:

| forced rationale | Δx @5 s, SFT (RL) | sign | Δy @5 s |
|---|---|---|---|
| "Slow down to follow the stopped lead vehicle." | −0.88 (−0.86) m | 52/60 neg, p 5e-8 | ~0 |
| "Stop for the stop sign at the intersection." | −1.18 (−1.15) m | 57/60 neg, p 2e-10 | ~0 |
| "Accelerate to target speed along the clear lane markings." | **+1.20 (+1.11) m** | 52/60 pos, p 5e-9 | ~0 |
| "…nudge **left** to increase clearance…" | +0.60 | n.s. | **+7 (+6) cm**, 47/60, p 1e-4 |
| "…nudge **right** to increase clearance…" | +0.56 | n.s. | **−5 (−5) cm**, 50/60 neg, p 1e-5 |
| Alpamayo-style "Keep lane since the lane is clear ahead" | +0.15 | p 0.49 | ~0 |

The effect is **bidirectional on both axes and matches the meaning of the text.** That rules
out an OOD-perturbation explanation, which would push every arm the same way. The
Alpamayo-register sentence, meaning "carry on", does nothing, as it should.

**What the two stacks together say.**

1. ~~The instrument discriminates: "not read" for Alpamayo, "read" for Qwen-Drive.~~
   **Withdrawn after §3.9.** The unsigned Alpamayo verdict was a power failure. Both experts
   obey longitudinal text. The real differences are **lateral obedience** (Qwen-Drive only)
   and **signal-to-noise** (Qwen-Drive's CoT effect is ~10× its seed spread; Alpamayo's is
   0.3–1×).
2. **"Read" ≠ "informative".** In Qwen-Drive, dropping the rationale from the cache is null
   (1.1–1.2×), and direct vs reasoning planning differ by only 1.5× the floor. The expert
   *obeys* a rationale, but its own rationale adds almost nothing it does not already get from
   images, nav and ego state. The CoT works as a **steering port**, not as an information
   source.
3. **The hierarchy is the same in both stacks:** ego ≫ images > nav ≫ CoT (80× / 14–24× /
   10× / ~3× in Qwen-Drive). The magnitudes are not comparable across models, since each has
   its own floor. A rationale moves the plan ~1 m over 5 s; a nav swap or a frozen ego moves it
   metres to tens of metres.
4. **Why the stacks differ is not established.** Candidates: Qwen-Drive's expert attends the
   full-attention-layer KV at the rationale's rotary positions, and its RL stage was trained on
   reasoning-conditioned rollouts (the RL planner's CoT sensitivity is the same as SFT's, so
   RL is not the cause). Alpamayo's expert anchors on `ego_history[:, -1]`. This is a
   hypothesis, not a result.

### 3.9 The signed test on Alpamayo — and what it overturns

`alpamayo1.5/semantic_cot_alpamayo.py`: the same signed test as §3.8, on **the same 60
clips at the same t0**. Six seeds, and one prefill per arm reused across seeds (verified
exact: repeat calls, reordered calls and a fresh prefill give bit-identical trajectories).
Two registers:
- `a_*`: Alpamayo's own Chain-of-Causation templates. `a_accel` is *constructed*, because no
  "accelerate" template occurred in 40 clips.
- `q_*`: Qwen-Drive's exact strings, the same text on both models.

Each cell is Δ vs the model's own CoT at 5 s. Seeds 42–47, with the independent replicate
at seeds 1042–1047 in brackets:

| forced rationale | Alpamayo Δx | Alpamayo Δy | Qwen-Drive Δx | Qwen-Drive Δy |
|---|---|---|---|---|
| q "Stop for the stop sign at the intersection." | **−0.91 (−0.95)** 49/60 neg | ~0 | −1.18 (−1.20) | ~0 |
| q "Slow down to follow the stopped lead vehicle." | −0.62 (−0.66) 53/60 | ~0 | −0.88 (−0.90) | ~0 |
| q "Accelerate to target speed…" | **+0.69 (+0.62)** 58/60 pos | ~0 | +1.20 (+1.20) | ~0 |
| a "Accelerate since the lane is clear ahead" | +0.78 (+0.70) **60/60** pos | ~0 | — | — |
| a "Slow down for the stop sign ahead…" | −0.54 (−0.62) 49/60 | ~0 | — | — |
| a/q nudge **left** | +0.1–0.3 | **−0.01…−0.03, p > 0.1** | +0.6 | **+0.07**, p 1e-4 |
| a/q nudge **right** | +0.1–0.3 | **~0, p > 0.4** | +0.56 | **−0.05**, p 1e-5 |
| seed floor, mean \|Δx\| | ~1.0 m | | ~0.13 m | |

**What this establishes.**
1. **Both experts read the CoT for its meaning along the speed axis.** The absolute effects
   are similar: Alpamayo about 70 % of Qwen-Drive's.
2. **Only Qwen-Drive reads it laterally.** Alpamayo ignores "nudge left/right" in both
   registers and both seed sets, though its own CoTs say "nudge left" on ~10 % of clips.
3. **Signal-to-noise is where the models differ.** Alpamayo's seed spread is ~8× larger, so
   on a single trajectory its CoT effect is buried in sampling noise (|Δ|/floor 0.3–1.0,
   against 7–11 for Qwen-Drive).
4. **In both, the model's own CoT carries little extra information.** Dropping it from the
   cache or masking it is null in both (Qwen-Drive 1.1–1.2×, Alpamayo 0.21×, §3.7). The CoT
   is an **obeyed steering port, not an information source**. That claim now rests on two
   stacks.

**Methodological lesson (worth a paragraph in any paper).** An unsigned divergence against
a seed floor tests "is the effect larger than one sample's noise?", not "is there an effect?".
For a high-variance sampler it has low power. A **signed, paired, multi-seed test with an
independent seed-set replicate** is the right instrument. Two traps met on the way:
- **Common random numbers.** A seed produces the same noise on every clip, so a *seed set*
  carries a shared bias. Our "noise" arm drifted +1.09 m on 55/60 clips, then −0.97 m on
  56/60 when the seed set changed. The arm is fine as a magnitude floor, but a signed test on
  it is invalid.
- **Seed-set dependence.** Paired arms cancel the shared bias. Only the independent-set
  replicate shows the effect is not seed-specific.

---

---

## 4. What we can do — ranked by cost and by whether it is ours

| | task | cost | why it is ours |
|---|---|---|---|
| ✅ | Alpamayo input audit (ego / cameras) | done | §3.1, §3.3 |
| ✅ | brain→cerebellum audit (CoT teacher-forcing) | done | §3.4 |
| ✅ | Qwen-Drive planning audit | done | §3.8 — 60 rebuilt PhysicalAI scenes; CoT **is** read, semantically (bidirectional Δx ±1 m, Δy ±6 cm), but its own CoT is uninformative (KV drop null) |
| ✅ | channel audit: nav vs CoT, with a sampling-noise floor | done | §3.5 — **11.1×**, and the falsification arm failed to falsify |
| ⭐ | per-camera and per-horizon decomposition of the Alpamayo audit | hours | which camera, and at which horizon, does perception start to matter? |
| ✅ | the same CoT audit on Qwen-Drive's `PlanningExpert` | done | §3.8; the signed test (§3.9) then overturned the Alpamayo "not read" verdict |
| | BEVFusion-LC on the 120 packed Qwen-Drive tokens, same ruler | hours | the other half of the headroom number; packer + taxonomy ready |
| | verify Tier 2 (weights + envs) so the tutorial can grow | ~a day | turns 8 cloned repos into runnable baselines |
| | closed-loop: the reviewers' explicit complaint about this whole subfield | weeks | **NF-Bench + AlpaSim already exist here**; nobody else in this literature has it |
| ✗ | a leaderboard across Tier 1–3 | weeks | measures datasets, not methods; Tier 4 cannot participate |

**The through-line — four independent stacks, one conclusion.**

| stack | what was compared | result |
|---|---|---|
| **ours** (occ → planning, `fa6bd4c`) | occupancy vs ego history | occ-only L2@3s **5.04**; ego-only **2.09**; occ+ego **2.11** → **Δ ≈ 0**, and a *larger* perception backbone makes it worse (2.49). Ego-only equals the analytic constant-velocity floor (2.09) |
| **OccVLA** (their own Table 5) | occupancy supervision vs ego trajectory | occupancy **+1.5 %**; ego trajectory L2 0.48 → 0.28 (**−42 %**) ≈ **25×** |
| **Alpamayo 1.5** (§3.1, §3.3) | ego history vs cameras | **8.47×** vs **2.25×** |
| **Alpamayo 1.5** (§3.4) | ego history vs reasoning content | **26×** |

Three of these are entirely independent codebases — ours, OccVLA's, NVIDIA's. **Open-loop
driving benchmarks are substantially measuring ego extrapolation, and this field has no
standard control that would reveal it.** That is the paper, and the closed loop — which
OccVLA's own reviewers demanded and which we already have in NF-Bench + AlpaSim — is where it
gets settled.

---

## 5. Reproduction

```bash
# Alpamayo 1.5 input audit
cd /data/rnd-liu/MyRepo/alpamayo1.5
./a1_5_venv/bin/python audit_ego.py --n-clips 40 \
    --arms real,frozen,shuffled --out audit_ego.json
./a1_5_venv/bin/python audit_ego.py --n-clips 40 \
    --arms real,blind,camswap  --out audit_cam.json

# Qwen-Drive planning audit (§3.8): Stage 1 in the Alpamayo venv, Stage 2 in py310
./a1_5_venv/bin/python qwendrive_bridge/build_scenes.py --n-clips 60
PY=~/miniconda3/envs/py310/bin/python
$PY qwendrive_bridge/run_audit.py --n 60 --planner planner-sft
$PY qwendrive_bridge/run_audit.py --n 60 --planner planner-rl --skip-direct
$PY qwendrive_bridge/analyze.py qwendrive_bridge/outputs/<run>
$PY qwendrive_bridge/semantic_cot.py --n 60 --planner planner-sft          # + --seed 1042 replicate
./a1_5_venv/bin/python semantic_cot_alpamayo.py --n 60                    # §3.9; + --seed 1042
# ⚠ run_audit/semantic_cot swap the released Conv3d patch_embed for its bit-identical matmul
#   (34 s -> 0.4 ms per prefill); without it one clip takes >20 min

# Qwen-Drive: the same principle, on injected LiDAR tokens
cd DeepDataMiningLearning/bevdet/lidar_tokens
./run_step7.sh          # const / lidar / shuffle + held-out
```

`audit_ego.py` writes one JSON row per (clip, arm) with the full L2 curve and the
Chain-of-Causation trace, incrementally, so a crash in the summary does not lose the run —
which is how the first 40-clip run survived a `df.clip` name collision with
`DataFrame.clip`.
