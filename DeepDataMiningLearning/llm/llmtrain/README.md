# llmtrain: post-training LLMs from one config file

**SFT · LoRA · DPO · GRPO · on-policy distillation · knowledge distillation, on three backends:
nanolm's own loops, Hugging Face TRL, and NVIDIA NeMo AutoModel.**

This tutorial teaches the post-training methods of the lecture deck *From GPT to Modern LLMs* (Part 4) by
running them. Each method is one YAML file, and the same file runs on different training libraries. The
evaluation code is the same for all of them, so you can compare libraries as well as methods.

```bash
cd DeepDataMiningLearning/llm
python -m llmtrain -c llmtrain/configs/sft_alpaca.yaml                     # Hugging Face TRL
python -m llmtrain -c llmtrain/configs/sft_alpaca.yaml backend=native      # the same experiment, nanolm's loop
python -m llmtrain -c llmtrain/configs/sft_alpaca.yaml backend=automodel   # the same experiment, NVIDIA AutoModel
```

## Contents

0. [Why a thin layer](#0-why-a-thin-layer)
1. [Install](#1-install)
2. [Your first run, step by step](#2-your-first-run-step-by-step)
3. [The config file](#3-the-config-file)
4. [Data](#4-data)
5. [The methods](#5-the-methods): SFT and LoRA · DPO · GRPO · on-policy distillation · knowledge distillation
6. [NVIDIA NeMo AutoModel in depth](#6-nvidia-nemo-automodel-in-depth)
7. [The common evaluation](#7-the-common-evaluation)
8. [Debug or negative result? A case study (NaN gradients)](#8-debug-or-negative-result-a-case-study)
9. [Extending llmtrain](#9-extending-llmtrain)
10. [Validation results](#10-validation-results)
11. [Exercises](#11-exercises)

---

## 0. Why a thin layer

Three libraries can post-train an LLM today, and each is good at something different:

| backend | what it is | methods in llmtrain | strength |
|---|---|---|---|
| `native` | nanolm's own ~300-line loops (`nanolm/finetune.py`) | sft, dpo | every loss is written out; the reference |
| `trl` | Hugging Face TRL (`SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, `GKDTrainer`) | sft, dpo, grpo, opd | the widest menu of RL / preference methods |
| `automodel` | NVIDIA NeMo AutoModel (FSDP2, tensor and context parallelism, Hugging Face checkpoints in and out) | sft, kd, recipe | scale, MoE models, NVIDIA's tested recipes |

A tempting design is one framework that wraps all three. It is usually a mistake. The libraries change every
month, and a big wrapper ends up as the lowest common denominator plus a large bug surface. NVIDIA's own BioNeMo
recipes make the same argument ("rather than offering a batteries-included training framework … KISS over DRY").

So `llmtrain` is a **thin translation layer**. It does four things and nothing else:

1. **One config schema** (`config.py`): what you want to run.
2. **One data layer** (`data.py`): the exact examples, written to the run directory and shared by every backend.
3. **Adapters** (`backends/`, about 150 lines each): each translates the config into the library's *own* config
   and calls the library's *own* entry point.
4. **One evaluation** (`evaluate.py`): before vs after, scored by the same code for every backend.

When a library changes, only its adapter changes. Anything the schema does not cover goes in
`backend_overrides` and is handed to the library unchanged. A real example from building this: in TRL 1.16, GKD
moved to `trl.experimental`, and the fix was 4 lines in one adapter.

---

## 1. Install

```bash
conda create -n llmtrain python=3.12 -y && conda activate llmtrain
pip install uv
uv pip install nemo-automodel trl peft datasets pyyaml pytest
nvidia-smi                                   # the driver must support CUDA 13 (AutoModel's PyTorch wheels are cu130)
python -m pytest llmtrain/tests -q           # 20 CPU tests, no downloads
python -m llmtrain --list                    # what runs where
```

The versions we tested with (October 2026) are torch 2.14.1+cu130, transformers 5.18, TRL 1.16, PEFT 0.21,
nemo-automodel 0.7.0. AutoModel, TRL and PEFT install fine into one environment.

Notes:
- AutoModel needs Python ≥ 3.11.
- For Nemotron (Mamba) models, NVIDIA recommends their container `nvcr.io/nvidia/nemo-automodel`.
- If AutoModel lives in a different environment, pass `launch.automodel_cmd=/path/to/env/bin/automodel`.
- No GPU? The `native` backend is nanolm's, which runs on CPU and Apple MPS. TRL also runs on MPS, slowly.

---

## 2. Your first run, step by step

```bash
python -m llmtrain -c llmtrain/configs/sft_alpaca.yaml
```

The config trains Qwen2.5-0.5B with LoRA on 2,000 Alpaca instructions for 120 steps. It takes about 3 minutes
on an H100. The console shows:

```
[run] outputs/llmtrain/2026-10-25_12-13-34_sft_alpaca_trl
[run] method=sft backend=trl model=Qwen/Qwen2.5-0.5B
[data] alpaca: 2000 train / 100 val (sft, max_len 512)
... TRL progress bar, loss every 10 steps ...
[eval] {'before_val_response_nll': 1.5648, 'after_val_response_nll': 1.3946, ...}
```

Everything about the run is in its directory:

```
outputs/llmtrain/<date>_<name>/
  config.yaml            the fully resolved config: re-run it with  python -m llmtrain -c <this>/config.yaml
  meta.json              git commit (and whether the tree was dirty), seed, library versions, GPU, the command
  data/train.jsonl       the exact training examples;  data/val.jsonl  the held-out ones
  final/                 (trl) the LoRA adapter + tokenizer
  train_log.jsonl        (trl) the trainer's log history
  samples_before.txt     greedy answers of the base model to eval.prompts
  samples_after.txt      the same prompts after training
  metrics.json           the common before/after evaluation + the backend's own numbers
```

Read the samples. Qwen2.5-0.5B *base* already answers instructions, because modern base models see
instruction-like text during pre-training. SFT therefore changes the **style** more than the ability:

```
BEFORE  Sure, here are three tips for staying healthy:
        1. **Eat a Balanced Diet**: A balanced diet is essential for ...
AFTER   1. Eat a balanced diet: A healthy diet should include a variety of fruits, vegetables, ...
```

After SFT there is no "Sure, here are …" and no markdown bold. That is the Alpaca house style. Held-out answer
loss fell from 1.565 to 1.395 nats/token. This is the lesson of TUTORIAL_nanolm §3.8 again: SFT teaches format
and behaviour.

Now run the same experiment on the other two backends and compare `metrics.json`:

```bash
python -m llmtrain -c llmtrain/configs/sft_alpaca.yaml backend=native
python -m llmtrain -c llmtrain/configs/sft_alpaca.yaml backend=automodel
```

We measured 1.393 (native), 1.395 (TRL) and 1.403 (AutoModel). §10 explains the small difference.

---

## 3. The config file

A config is a small YAML file that inherits shared defaults:

```yaml
# llmtrain/configs/sft_alpaca.yaml
defaults: [base.yaml]            # merged first; this file overrides it
name: sft_alpaca
method: sft                      # sft | dpo | grpo | opd | kd | recipe
backend: trl                     # native | trl | automodel
model: {name: Qwen/Qwen2.5-0.5B} # + peft: lora, lora_r, lora_alpha, ... from base.yaml
data: {preset: alpaca, n_train: 2000, n_val: 100, max_len: 512}
train: {steps: 120, lr: 2.0e-4}
```

Every key is documented in `llmtrain/config.py`. The sections:

| section | what it holds |
|---|---|
| `model` | `name`, `dtype`, `peft` (none / lora), LoRA rank / alpha / targets, `teacher` (opd, kd), `attn_impl` |
| `data` | `preset`, `path`, splits, `n_train`, `n_val`, `max_len`, `columns` (your own JSONL) |
| `train` | `steps`, `batch_size`, `grad_accum`, `lr`, `schedule`, `warmup_steps`, `precision`, `cudnn_sdpa`, … |
| `dpo` / `grpo` / `opd` / `kd` | method-specific settings (§5) |
| `eval` | the common evaluation: `n_val`, `gen_tokens`, `prompts` |
| `launch` | `nproc_per_node`, `automodel_cmd`, `dry_run` |
| `automodel` | `recipe_yaml` (an official recipe), `strategy`, `tp_size`, `cp_size` |
| `backend_overrides` | anything else, passed to the library unchanged |

**Overrides from the command line** use `section.key=value`, and lists and dicts in YAML syntax:

```bash
python -m llmtrain -c llmtrain/configs/grpo_gsm8k.yaml grpo.loss_type=grpo grpo.epsilon_high=0.2 name=grpo_vanilla
python -m llmtrain -c llmtrain/configs/sft_alpaca.yaml "model.lora_targets=[q_proj,v_proj]" model.lora_r=8
```

**A typo is an error**, not a silently ignored setting:

```
$ python -m llmtrain -c llmtrain/configs/sft_alpaca.yaml train.step=10
KeyError: unknown config key(s) ['step'] in section 'train'. Valid keys: [... 'steps' ...]
```

A silently ignored setting is the most common reason an ablation "did nothing".

**`backend_overrides`** reach every option of the underlying library:

```yaml
backend: trl
backend_overrides: {neftune_noise_alpha: 5, packing: true}       # any field of trl.SFTConfig; unknown fields -> error
---
backend: automodel
backend_overrides:                                               # deep-merged last into the AutoModel YAML
  distributed: {tp_size: 2}
  packed_sequence: {packed_sequence_size: 4096}
```

---

## 4. Data

### 4.1 Presets and canonical records

Each preset is reduced to one of three **canonical record kinds**. These are written to `data/train.jsonl` and
`data/val.jsonl`, and every backend reads only those files:

| kind | fields | used by |
|---|---|---|
| `sft` | `prompt`, `response` | sft, opd, kd |
| `pref` | `prompt`, `chosen`, `rejected` | dpo |
| `qa` | `prompt`, `answer` (verifiable), `response` (reference solution) | grpo, and also sft / opd / kd |

| preset | source | kind | notes |
|---|---|---|---|
| `alpaca` | `yahma/alpaca-cleaned` | sft | instruction (+ input) → output |
| `ultrafeedback` | `trl-lib/ultrafeedback_binarized` | pref | keeps pairs whose scores differ by ≥ `data.min_score_gap` |
| `gsm8k` | `openai/gsm8k` (main) | qa | answer = the number after `####`; calculator annotations `<<…>>` removed |
| `squad` | `rajpurkar/squad` | sft | context + question → first answer |
| `jsonl` | your file (`data.path`) | any | map your column names with `data.columns` |

A GSM8K record:

```json
{"prompt": "... What is Marion's score?\n\nSolve the problem step by step. Put the final answer after '####'.",
 "answer": "24",
 "response": "Ella's score is 40 items - 4 items = 36 items.\n...\n#### 24"}
```

### 4.2 Your own data

```bash
# my.jsonl:  {"question": "...", "solution": "..."}   one object per line
python -m llmtrain -c llmtrain/configs/sft_alpaca.yaml data.preset=jsonl data.path=my.jsonl \
       "data.columns={prompt: question, response: solution}"
```

Records with `chosen` and `rejected` become `pref`, records with `answer` become `qa`, and otherwise they are
`sft`. llmtrain checks that the kind fits the method: DPO on SFT data is an error.

### 4.3 Three rules that keep comparisons fair

- **Fixed selection.** The examples are chosen by `Random(0)`, not by the run's seed. Two seeds therefore train
  on the same data and differ only in initialization and ordering.
- **No leakage.** If a dataset has no validation split, the first 5 × `n_val` shuffled rows are reserved, and
  training never sees them.
- **One length rule.** An example is dropped if its length *after the chat template* exceeds `data.max_len`
  tokens. Each library truncates differently; dropping examples up front keeps every backend's data identical.

Chat models are trained through their own chat template. Base models without a template get the same
`### Instruction / ### Response` rendering in every backend.

---

## 5. The methods

Each subsection gives the idea, the loss, the config, and what we measured. The derivations are in the lecture
deck, Part 4.

### 5.1 SFT and LoRA

Cross-entropy on the answer tokens only. The prompt positions are masked out, and the end-of-turn token is
scored, which is how the model learns to stop:

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|\mathcal{A}|}\sum_{t\in\mathcal{A}} \log p_\theta(x_t \mid x_{<t})$$

LoRA trains a low-rank update $W = W_0 + \tfrac{\alpha}{r}BA$ with $B=0$ at the start. With `peft: lora`,
r = 16 on all projections, about 2 % of the weights are trained. Use `model.peft=none` for full fine-tuning,
with a ~10× smaller learning rate.

| backend | how the masking is done |
|---|---|
| native | `nanolm.finetune.chat_example`: label −100 on prompt tokens |
| trl | prompt-completion dataset, `completion_only_loss=True` |
| automodel | `ColumnMappedTextInstructionDataset(answer_only_loss_mask=True)` |

### 5.2 DPO

```bash
python -m llmtrain -c llmtrain/configs/dpo_ultrafeedback.yaml               # trl;  backend=native also works
```

DPO learns from comparisons without a reward model or sampling:

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\Big(\beta\Big[\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big]\Big)$$

With LoRA, the reference $\pi_{\text{ref}}$ is the same network with the adapter switched off, on both backends.
No second copy of the model is needed.

Implicit-reward accuracy went from 0.50 (policy = reference) to 0.66–0.67 on both backends. Yet the model's
own ranking of the two answers (per-token log-probability) stayed at 0.66–0.68. DPO moves probabilities
*relative to the reference*; it does not make the model a better judge. This is the same finding as
TUTORIAL_nanolm §3.9, now reproduced on two backends.

### 5.3 GRPO with verifiable rewards

```bash
python -m llmtrain -c llmtrain/configs/grpo_gsm8k.yaml
```

For each prompt, sample G answers and score each with a program. The advantage is the group-normalized reward,
so no critic network is needed:

$$\hat A_i = \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})}$$

The rewards live in `llmtrain/rewards.py`:

- `answer_correct`: 1 if the final answer, after `####`, else in `\boxed{}`, else the last number, equals the
  reference.
- `answer_format`: 0.1 if the model used the requested `#### <number>` format.

The GRPO family from the lecture is selected with config keys. TRL 1.16 `loss_type` values: `grpo`, `dr_grpo`,
`dapo`, `bnpo`, `cispo`, `sapo`, `luspo`, `vespo`.

```yaml
grpo:
  num_generations: 8          # group size G
  loss_type: dapo             # token-level loss over the whole batch (DAPO)
  epsilon: 0.2                # lower clip
  epsilon_high: 0.28          # "clip-higher" (DAPO): rare good tokens can grow
  beta: 0.0                   # no KL to the reference (as in DAPO and the Nemotron recipes)
  scale_rewards: group        # 'none' = Dr. GRPO (no std division)
  rewards: [answer_correct, answer_format]
  use_vllm: false             # true = generate with vLLM (strongly recommended beyond toy runs)
```

The validation run (60 steps × 4 prompts × 8 samples) taught two things:
- **0.38 → 0.42 accuracy is within noise.** The standard error at 200 questions is ≈ 0.035. Real GRPO runs use
  thousands of steps.
- **The format reward never fired**, not even once. Qwen answers with `\boxed{}`, so every sample got 0 from
  `answer_format`, and a reward that is constant within a group has zero advantage. **A shaping reward that never
  fires teaches nothing.** Look at the reward means in `train_log.jsonl` before you trust a reward design.

### 5.4 On-policy distillation (OPD)

```bash
python -m llmtrain -c llmtrain/configs/opd_gsm8k.yaml     # student Qwen2.5-0.5B-Instruct, teacher Qwen2.5-1.5B-Instruct
```

The student samples its own answer, and the teacher scores every token. The loss is the reverse KL on the
student's own trajectories: dense feedback (every token), on-policy (the student's mistakes):

$$\mathcal{L}_{\text{OPD}} = \mathbb{E}_{y\sim\pi_\theta}\Big[\sum_t \mathrm{KL}\big(\pi_\theta(\cdot|c_t)\,\Vert\,\pi_T(\cdot|c_t)\big)\Big]$$

In TRL this is the GKD trainer with `lmbda = 1` (only student samples) and `beta = 1` (reverse KL; 0 would be
forward KL). Teacher and student must share a tokenizer. In the validation run the reverse-KL loss fell from 0.66
to 0.55 in 60 steps, and accuracy went 0.38 → 0.40 (within noise).

**Cost:** 70 minutes for 60 steps, because every training sample is generated by the student with HF `generate`.
This is the price of being on-policy. Production systems (DeepSeek-V4, Nemotron MOPD) run generation on separate
inference engines (lecture Part 4).

### 5.5 Knowledge distillation with AutoModel (off-policy)

```bash
python -m llmtrain -c llmtrain/configs/kd_gsm8k_automodel.yaml
```

Same student, teacher and data as 5.4, but *off-policy*. The student is trained on the reference solutions with:

$$\mathcal{L} = (1-\lambda)\,\mathrm{CE}(\text{reference}) + \lambda\,\mathrm{KL}(\text{teacher}\,\Vert\,\text{student}), \qquad \lambda = \texttt{kd.ratio} = 0.5$$

Result: accuracy 0.38 → 0.45, and reference-solution NLL 0.883 → 0.708, in under 2 minutes.

**Compare 5.4 and 5.5 carefully before you conclude "KD beats OPD".** KD also trains on the reference solutions
(half of its loss is plain SFT), and both runs are 60 steps on one seed. Exercise 3 asks you to separate the two
effects.

---

## 6. NVIDIA NeMo AutoModel in depth

### 6.1 What the adapter writes

AutoModel recipes are YAML files whose `_target_` keys name the Python objects to build. llmtrain writes one to
`automodel_config.yaml` and runs `automodel <yaml> [--nproc-per-node N]`. Inspect it without training:

```bash
python -m llmtrain -c llmtrain/configs/sft_alpaca.yaml backend=automodel --dry-run
```

```yaml
recipe: TrainFinetuneRecipeForNextTokenPrediction
step_scheduler: {max_steps: 120, global_batch_size: 16, local_batch_size: 4, ...}
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen2.5-0.5B
distributed: {strategy: fsdp2, tp_size: 1, cp_size: 1}
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: ['*.q_proj', '*.k_proj', ...]     # AutoModel matches full module names, so llmtrain adds '*.'
  dim: 16
  alpha: 32
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: <run_dir>/data/automodel_train.jsonl
  column_mapping: {question: question, answer: answer}
  answer_only_loss_mask: true
  use_hf_chat_template: true
checkpoint: {model_save_format: safetensors, save_consolidated: true, checkpoint_dir: <run_dir>/automodel/checkpoints}
```

The `'*.'` prefix is the kind of detail a thin adapter exists for. With plain `q_proj`, AutoModel attached no
adapter and stopped with "optimizer received no trainable parameters".

### 6.2 Running NVIDIA's official recipes

`method: recipe` runs an official recipe from the NVIDIA-NeMo/Automodel repository, given as a path or a URL.
llmtrain adds only:
- a run directory;
- the seed;
- the step budget;
- Hugging Face–format checkpoints;
- optionally, our data (`automodel.use_our_data`).

```yaml
# llmtrain/configs/recipe_automodel_qwen3_hellaswag.yaml
method: recipe
backend: automodel
model: {name: Qwen/Qwen3-0.6B}
automodel:
  recipe_yaml: https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml
  use_our_data: false              # keep the recipe's own dataset
train: {steps: 50}
backend_overrides:
  step_scheduler: {global_batch_size: 32, local_batch_size: 8}
```

The downloaded recipe is archived as `automodel_recipe_source.yaml` in the run directory, so the run stays
reproducible even if the URL changes. Precedence, lowest to highest:

    official recipe  <  values derived from the llmtrain config  <  backend_overrides

The repository has recipes for hundreds of models (DeepSeek, Qwen, Llama, Gemma, gpt-oss, Nemotron, Kimi, …).
To run one, point `recipe_yaml` at it.

### 6.3 Scaling out

```bash
python -m llmtrain -c my.yaml backend=automodel launch.nproc_per_node=8                    # FSDP2 over 8 GPUs
python -m llmtrain -c my.yaml backend=automodel launch.nproc_per_node=8 automodel.tp_size=2 automodel.cp_size=2
```

`global_batch_size` is computed as `batch_size × grad_accum × nproc_per_node`, so the effective batch stays what
the config says. For multi-node SLURM jobs, use AutoModel's own launcher on the generated
`automodel_config.yaml`.

### 6.4 Where the result is

Full fine-tuning writes `automodel/checkpoints/epoch_*_step_*/model/consolidated/` (Hugging Face safetensors;
loads with `from_pretrained`, vLLM, lm-eval). LoRA writes an adapter directory. llmtrain finds either one and runs
the common evaluation on it.

---

## 7. The common evaluation

Every library logs differently: per-token or per-sequence averages, with or without padding, sometimes on a
moving window. Comparing logged losses across libraries is therefore not meaningful. `evaluate.py` scores the base
model and the trained model on the same `data/val.jsonl` with the same rendering:

| metric | for | definition |
|---|---|---|
| `val_response_nll` | sft, qa | mean −log p over held-out answer tokens (nats/token) |
| `val_policy_prefers_chosen` | pref | fraction of pairs where the model's per-token log-prob of `chosen` > `rejected` |
| `val_implicit_reward_acc` | pref | DPO's metric: β[(log π − log π_ref)(y_w) − (log π − log π_ref)(y_l)] > 0; the base model is π_ref |
| `val_answer_acc` | qa | greedy decoding + `rewards.answer_correct` |
| `val_answer_format` | qa | the answer used the requested `####` format |

```bash
python -m llmtrain.evaluate outputs/llmtrain/<run_dir>          # re-score a finished run (e.g. after changing evaluate.py)
```

**Check the evaluator itself.** The native backend also runs nanolm's own evaluation. For the SFT run nanolm
reported 1.3921 and `evaluate.py` 1.393. The DPO margins agree too (0.4317 vs 0.4327). Two independent
implementations agree, so we trust the number.

---

## 8. Debug or negative result? A case study

The first TRL run went wrong:

```
step 20 | loss 13.83 | grad_norm nan
step 30 | loss 0     | grad_norm nan
```

Is lr = 2e-4 too high, so this is a genuine divergence? Or is it a bug? The way to decide is to change one thing
at a time, with per-step logging:

| variant | step 22 grad norm | step 23 grad norm |
|---|---:|---:|
| bf16 (as configured) | 0.99 | **nan** |
| no gradient checkpointing | 0.98 | **nan** |
| fp32 weights, fp32 compute | 0.97 | 0.93 |
| fp32 weights, bf16 autocast | 0.95 | **nan** |
| bf16, eager attention | 0.96 | 0.94 |
| bf16, memory-efficient SDPA kernel disabled | 1.04 | **nan** |
| bf16, **cuDNN SDPA kernel disabled** | 1.00 | 0.93 |

What the table shows:
- The loss is smooth up to the failure (grad norm ≈ 1), so this is not a divergence.
- One batch produces a NaN *gradient* while its loss is finite.
- Only bf16 attention through the cuDNN kernel fails.

AutoModel later hit the same failure independently, at step 12 of its own run. That is two libraries and one
kernel. **This is an implementation bug in a dependency (cuDNN attention backward in bf16, torch 2.14.1+cu130 on
H100), not a property of the method.**

llmtrain disables that kernel by default (`train.cudnn_sdpa: false`). The switch also reaches AutoModel's
subprocesses through `llmtrain/_site/sitecustomize.py`. After the fix, all three backends trained cleanly and
agreed to 0.01.

General lessons:
- A finite loss with a NaN gradient points at the backward pass of a kernel, not at your hyperparameters.
- Controls have to separate the candidates. For example, "fp32 weights + bf16 autocast" separates *weight
  precision* from *compute precision*.
- When two independent libraries fail the same way, look at what they share.

---

## 9. Extending llmtrain

**A dataset preset.** Add an entry to `PRESETS` and a branch in `to_record` (`data.py`) that returns a canonical
record.

**A reward.** Write a function `f(prompts, completions, answer, **kw) -> list[float]` in `rewards.py` and register
it in `REWARDS`. Then use `grpo.rewards=[answer_correct, my_reward]`.

**A backend.** A backend is a module with one function:

```python
def train(spec, run_dir: Path, kind: str) -> dict:
    # read run_dir/"data/train.jsonl" and "data/val.jsonl", translate `spec` into the library's own config,
    # call the library's own entry point, and return
    return {"final_model": "<HF model dir or LoRA adapter dir>", "is_adapter": True, "backend_metrics": {...}}
```

Register it in `run.py` (`SUPPORT` and the dispatch). The common evaluation, run directory and metadata come for
free.

**Rule of thumb:** if an adapter grows past a few hundred lines, it is re-implementing the library. Use
`backend_overrides` instead.

---

## 10. Validation results

These are one-seed runs on a single H100 NVL that another job was also using (2026-10-25). The configs are used
as shipped. They show that every path works and that the backends agree. They are **not** tuned results.

| config | backend | metric (held-out) | before | after |
|---|---|---|---:|---:|
| `sft_alpaca` (Qwen2.5-0.5B, LoRA, 120 steps) | native | response NLL (nats/token) | 1.565 | **1.393** |
| | trl | | 1.565 | **1.395** |
| | automodel | | 1.565 | **1.403** |
| `dpo_ultrafeedback` (Qwen2.5-0.5B-Instruct, 100 steps) | native | implicit-reward accuracy | 0.50 (policy = reference) | 0.66 |
| | trl | | 0.50 | 0.67 |
| | both | policy prefers chosen | 0.68 | 0.67 / 0.66 |
| `grpo_gsm8k` (60 steps × 4 prompts × 8 samples) | trl | GSM8K answer accuracy (200) | 0.38 | 0.42 |
| `opd_gsm8k` (teacher Qwen2.5-1.5B-Instruct, 60 steps) | trl | GSM8K answer accuracy (200) | 0.38 | 0.40 |
| `kd_gsm8k_automodel` (same teacher, kd_ratio 0.5, 60 steps) | automodel | GSM8K answer accuracy (200) | 0.38 | 0.45 |
| `recipe_automodel_qwen3_hellaswag` (official recipe, 50 steps) | automodel | the recipe's own val loss | 3.32 (step 24) | 3.12 (step 49) |

Wall-clock time per run: SFT 2–3 min, DPO 3–6 min, KD 2 min, GRPO 20 min, OPD 70 min. The last two generate
with HF `generate`; use vLLM for real runs.

How to read it:
- **The SFT backends agree within 0.01 nats.** native and TRL compute the same loss and agree to 0.002.
  AutoModel's defaults differ slightly (xavier-initialized LoRA A, warm-up from 0.1·lr, a min-lr floor of 0.1·lr).
  Find out how much each contributes in Exercise 1.
- **DPO agrees across backends.** The model's own ranking did not move (§5.2).
- **GRPO and OPD (+0.02, +0.04) are within noise.** These are 60-step smoke runs, and the standard error is
  ≈ 0.035.
- **KD's +0.07 mixes distillation with SFT on reference solutions** (§5.5).

---

## 11. Exercises

1. **Why does AutoModel differ by 0.01?** Use `backend_overrides` to match AutoModel's LoRA initialization and
   LR schedule to TRL's. Can you close the gap? Run two seeds before you claim anything.
2. **The GRPO family.** On `grpo_gsm8k`, compare `loss_type` ∈ {grpo, dr_grpo, dapo} and `epsilon_high` ∈ {0.2,
   0.28}. Train ≥ 300 steps with `grpo.use_vllm=true` and use 2 seeds. Which differences exceed the noise?
3. **KD vs OPD, fairly.** Separate "distillation" from "training on reference solutions":
   - run KD with `kd.ratio=1.0` (no CE);
   - run plain SFT on the same GSM8K solutions;
   - run OPD.

   Use the same steps and seeds for all three. Which one helps, and why?
4. **A reward that fires.** Change the GSM8K instruction or the format reward so that the shaping term is
   non-zero for some samples. Plot `rewards/answer_format/mean` from `train_log.jsonl`. Does accuracy change?
5. **Your data.** Fine-tune on your own JSONL (§4.2) with `backend=automodel launch.nproc_per_node=2`, then
   evaluate the consolidated checkpoint with `nanolm.benchmarks` (TUTORIAL_nanolm §3.6).
6. **An official recipe.** Pick a recipe for a model you care about from the AutoModel repository, run it with
   `method: recipe` for 100 steps, and report its validation loss curve from `automodel.log`.

---

**Files.** `config.py` (the schema), `data.py`, `rewards.py`, `evaluate.py`, `run.py` (entry point),
`backends/{native,trl_backend,automodel_backend}.py`, `configs/*.yaml`, `tests/test_llmtrain.py`,
`_site/sitecustomize.py`.

**See also.** The lecture deck *From GPT to Modern LLMs* (Part 4: InstructGPT → PPO → DPO → the GRPO family →
agentic RL → OPD / MOPD → Nemotron), `TUTORIAL_nanolm.md` §3.7–3.10, and the NVIDIA NeMo AutoModel docs
(docs.nvidia.com/nemo/automodel).
