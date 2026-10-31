# llmtrain: post-training LLMs from one config file

**SFT · LoRA · DPO · GRPO · on-policy distillation · knowledge distillation · agent RL, on four backends:
nanolm's own loops, Hugging Face TRL, NVIDIA NeMo AutoModel, and NVIDIA NeMo RL (with NeMo Gym).**

This tutorial teaches the post-training methods of the lecture deck *From GPT to Modern LLMs* (Part 4) by
running them. Each method is one YAML file, and the same file runs on different training libraries. The
evaluation code is the same for all of them, so you can compare libraries as well as methods.

```bash
cd DeepDataMiningLearning/llm
python -m llmtrain -c llmtrain/configs/sft_alpaca.yaml                     # Hugging Face TRL
python -m llmtrain -c llmtrain/configs/sft_alpaca.yaml backend=native      # the same experiment, nanolm's loop
python -m llmtrain -c llmtrain/configs/sft_alpaca.yaml backend=automodel   # the same experiment, NVIDIA AutoModel
python -m llmtrain -c llmtrain/configs/grpo_gsm8k_nemo_rl.yaml              # GRPO with vLLM rollouts, NVIDIA NeMo RL
```

## Contents

0. [Why a thin layer](#0-why-a-thin-layer)
1. [Install](#1-install)
2. [Your first run, step by step](#2-your-first-run-step-by-step)
3. [The config file](#3-the-config-file)
4. [Data](#4-data)
5. [The methods](#5-the-methods): SFT and LoRA · DPO · GRPO · on-policy distillation · knowledge distillation
6. [NVIDIA NeMo AutoModel in depth](#6-nvidia-nemo-automodel-in-depth)
7. [NVIDIA NeMo RL and NeMo Gym](#7-nvidia-nemo-rl-and-nemo-gym): vLLM rollouts, agent RL, a negative result,
   running on an NVIDIA DLI workshop VM
8. [The common evaluation](#8-the-common-evaluation)
9. [Debug or negative result? A case study (NaN gradients)](#9-debug-or-negative-result-a-case-study)
10. [Extending llmtrain](#10-extending-llmtrain)
11. [Validation results](#11-validation-results)
12. [Exercises](#12-exercises)

---

## 0. Why a thin layer

Four libraries can post-train an LLM today, and each is good at something different:

| backend | what it is | methods in llmtrain | strength |
|---|---|---|---|
| `native` | nanolm's own ~300-line loops (`nanolm/finetune.py`) | sft, dpo | every loss is written out; the reference |
| `trl` | Hugging Face TRL (`SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, `GKDTrainer`) | sft, dpo, grpo, opd | the widest menu of RL / preference methods |
| `automodel` | NVIDIA NeMo AutoModel (FSDP2, tensor and context parallelism, Hugging Face checkpoints in and out) | sft, kd, recipe | scale, MoE models, NVIDIA's tested recipes |
| `nemo_rl` | NVIDIA NeMo RL (Ray + vLLM rollouts + an AutoModel / Megatron policy) and NeMo Gym environments | grpo, opd, recipe | fast RL, multi-step agents with tools, multi-node |

A tempting design is one framework that wraps all four. It is usually a mistake. The libraries change every
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
python -m pytest llmtrain/tests -q           # 28 CPU tests, no downloads
python -m llmtrain --list                    # what runs where
```

The versions we tested with (October 2026) are torch 2.14.1+cu130, transformers 5.18, TRL 1.16, PEFT 0.21,
nemo-automodel 0.7.0. AutoModel, TRL and PEFT install fine into one environment.

Notes:
- AutoModel needs Python ≥ 3.11.
- For Nemotron (Mamba) models, NVIDIA recommends their container `nvcr.io/nvidia/nemo-automodel`.
- If AutoModel lives in a different environment, pass `launch.automodel_cmd=/path/to/env/bin/automodel`.
- No GPU? The `native` backend is nanolm's, which runs on CPU and Apple MPS. TRL also runs on MPS, slowly.
- NeMo RL is **not** installed into this environment: it lives in its own checkout with its own environments (§7.2).

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

We measured 1.393 (native), 1.395 (TRL) and 1.403 (AutoModel). §11 explains the small difference.

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
| `nemo_rl` | `dir` (the checkout), `env`, `gym_env`, `val_generations`, shared-GPU knobs (§7) |
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

## 7. NVIDIA NeMo RL and NeMo Gym

### 7.1 What NeMo RL adds, and how llmtrain drives it

NeMo RL is NVIDIA's RL library. It runs a Ray cluster in which vLLM generates the rollouts and an AutoModel
(DTensor) or Megatron policy trains on them; after every step the new weights are copied into vLLM ("refit").
NeMo Gym is the environment side: servers that hold tools, databases and verifiers, and score *multi-step agents*.
Generation is where TRL spends its time with HF `generate` (OPD: 70 minutes, §5.4); here it is vLLM's job.

NeMo RL pins its own stack (Python 3.13 and its own torch, vLLM and Ray) and builds one uv environment per worker
type, so it cannot live in the llmtrain environment. The adapter therefore **never imports it**. It starts from one
of NeMo RL's *own* example configs, adds Hydra overrides derived from the llmtrain config, and runs NeMo RL's own
script inside its checkout:

| llmtrain | NeMo RL script | base config |
|---|---|---|
| `method: grpo` | `examples/run_grpo.py` | `examples/configs/grpo_math_1B.yaml` |
| `method: grpo` + `nemo_rl.gym_env` | `examples/nemo_gym/run_grpo_nemo_gym.py` | `examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml` |
| `method: opd` | `examples/run_distillation.py` | `examples/configs/distillation_math.yaml` |
| `method: recipe` | `nemo_rl.script` | `nemo_rl.config` (any NeMo RL recipe, as is) |

Precedence: NeMo RL's example config < overrides derived from the llmtrain config < `backend_overrides` (Hydra
keys, e.g. `{loss_fn.ratio_clip_max: 0.3}`; a key that the base config lacks needs a `++` prefix).

The run directory gets, besides the usual files:
- `nemo_rl_command.txt`: the exact command. To debug, run it by hand in the checkout.
- `nemo_rl_overrides.json` and `nemo_rl.log` (NeMo RL's full output).
- `nemo_rl/checkpoints/` (Hugging Face weights in `.../policy/weights/model/consolidated`) and `nemo_rl/logs/`.
- `metrics.json` → `backend_metrics`: training reward and loss per step, every validation pass, and
  `val_groups` (§7.4).

### 7.2 Setup (once)

```bash
git clone https://github.com/NVIDIA-NeMo/RL nemo-rl && cd nemo-rl
git submodule update --init --recursive        # Megatron-Bridge, Automodel, Gym
pip install uv                                  # NeMo RL builds its own environments with uv
export NEMO_RL_DIR=$PWD                         # or pass nemo_rl.dir=... to llmtrain
```

We tested commit `4d464d5f3` (2026-10-25) on one H100 NVL. The first run of each worker type builds its
environment (the asynchronous vLLM worker took 25 minutes); later runs start in about 2 minutes. Put
`UV_CACHE_DIR` on a large disk: the environments take tens of GB.

The builds compile CUDA extensions (DeepEP, TransformerEngine). Without a system CUDA 13 toolkit we hit:

| build error | fix |
|---|---|
| DeepEP: `CUDA_HOME` not set, missing CCCL / NVTX headers | `conda create -p $TK -c nvidia cuda-toolkit=13.0 cuda-nvtx-dev=13.0`, and a `CUDA_HOME` with `bin`, `include` (incl. `include/cccl`), `lib64`, `nvvm` |
| DeepEP: `infiniband/verbs.h` not found | `conda create -p $RDMA -c conda-forge rdma-core`; add `$RDMA/include` to `CPATH` |
| TransformerEngine: `nccl.h` not found | an NCCL `include/` and `lib/` (with a `libnccl.so` link) on `CPATH` / `LIBRARY_PATH` |

**Give these to NeMo RL only; never export them globally** (a global `CUDA_HOME` breaks other CUDA projects on the
same machine). `nemo_rl.env` is applied to the NeMo RL process alone, and its `CUDA_HOME/bin` goes first on that
process's `PATH`:

```bash
# nemo_env.args: one override per line
nemo_rl.dir=/path/to/nemo-rl
nemo_rl.env={CUDA_HOME: /path/to/cuda_home13, CPATH: "/path/rdma/include:/path/cuda_home13/include/cccl:/path/nccl/include", LIBRARY_PATH: "/path/rdma/lib:/path/nccl/lib", TORCH_CUDA_ARCH_LIST: "9.0", UV_CACHE_DIR: /big/disk/uv}

mapfile -t ARGS < nemo_env.args
python -m llmtrain -c llmtrain/configs/grpo_gsm8k_nemo_rl.yaml "${ARGS[@]}"
```

**On a shared GPU.** NVIDIA's configs assume the GPU is yours. llmtrain's defaults handle the problems we hit:

| symptom | cause | llmtrain setting |
|---|---|---|
| vLLM: "Free memory on device … is less than desired GPU memory utilization" | another job holds memory | `nemo_rl.gpu_mem_util: 0.5` |
| vLLM's memory-profiling assertion | the other job's memory changes while vLLM profiles | `nemo_rl.kv_cache_gb: 8` (a fixed KV-cache size; no profiling) |
| sequence packing fails | it needs flash-attn, absent from the policy worker | `nemo_rl.sequence_packing: false` |
| Ray "403 Forbidden" at step 1 | Ray's own gRPC traffic goes through `http_proxy` | llmtrain adds the host name and IP to `no_proxy` |

### 7.3 GRPO on GSM8K: the same experiment, two libraries

```bash
python -m llmtrain -c llmtrain/configs/grpo_gsm8k_nemo_rl.yaml "${ARGS[@]}"
```

`grpo_gsm8k_nemo_rl.yaml` inherits `grpo_gsm8k.yaml` (§5.3) and changes only `backend`. The data files and the
common evaluation are the same:

| backend | generation | GSM8K accuracy (200), before → after | wall-clock |
|---|---|---|---:|
| trl | HF `generate` | 0.38 → 0.425 | 18 min |
| nemo_rl | vLLM | 0.38 → 0.455 | 9.5 min (6.7 s/step + startup) |

- **The accuracy difference is within noise** (one seed, SE ≈ 0.035).
- **It is also not quite the same experiment.** NeMo RL's math environment checks answers with `math-verify`, not
  `rewards.answer_correct`, and gives no format bonus. Its baseline is the leave-one-out mean, not the group mean.
  "The same config" does not guarantee the same algorithm; read what each library actually computes.
- **vLLM is 1.9× faster even at this toy size.** The gap grows with longer generations.

### 7.4 Agent RL with NeMo Gym: Workplace Assistant, and a negative result

The environment: 27 tools over five simulated databases (email, calendar, CRM, analytics, project management). A
task is a request such as "move every consulting customer who did not answer the proposal in six weeks to lost",
which needs up to 6 tool calls. The reward is 1 if the databases end in the correct state.

```bash
hf download nvidia/Nemotron-RL-agent-workplace_assistant --repo-type dataset \
   --local-dir $NEMO_RL_DIR/3rdparty/Gym-workspace/Gym/data/workplace_assistant     # train.jsonl, validation.jsonl
python -m llmtrain -c llmtrain/configs/gym_workplace_assistant_nemo_rl.yaml "${ARGS[@]}"
```

NVIDIA's example is Nemotron Nano 9B on 8 GPUs with Megatron. The adapter makes it run with a small Qwen3 on one
GPU:
- Megatron off, AutoModel on, and a full `policy.optimizer` (the example has `optimizer: null`, because Megatron
  owns it).
- **The tool-call parser belongs to the policy model:** `nemo_rl.gym_tool_parser: hermes` for Qwen. The example's
  `nemotron_json` stops vLLM with `KeyError: Tool parser 'nemotron_json' not found`.
- **`nemo_rl.gym_max_seq_len: 8192`.** The 27 tool schemas alone make a 4.2k-token first prompt. With 4096, every
  request was rejected ("NeMo Gym returned a result with no generation data").
- The first `gym_val_limit` validation tasks are written to the run directory, because Gym validates on a whole
  file.

**The result: 20 steps (8 tasks × 4 rollouts each) moved nothing.** Validation accuracy on 64 tasks was 0.1875
before and 0.1875 after. Bug or negative result? We went through it like §9:

1. **The loss was exactly 0.0000 in 17 of 20 steps, while the reward was not.** A GRPO loss is zero only when
   every group's advantage is zero, that is, when all 4 rollouts of each task got the same reward. Every training
   reward was a multiple of 4/32, except the two steps that had a non-zero loss.
2. **Is sampling broken, giving 4 identical rollouts?** We sampled 8 answers per task at T = 1 with plain vLLM: 14
   of 16 tasks gave at least 7 distinct answers. Sampling works.
   (Our first attempt at this check compared NeMo RL's validation dumps from step 0 and step 20 and found 61/64
   "identical". But those files contain no text, only empty message shells; we were comparing turn counts. Look
   inside a file before you compare it.)
3. **Measure it inside NeMo RL.** `nemo_rl.val_generations=4` samples each validation task 4 times and reports the
   share of *informative* groups (mixed rewards, so a non-zero advantage):

   | policy | accuracy | pass@4 | informative groups |
   |---|---:|---:|---:|
   | Qwen3-0.6B | 0.180 | 0.188 | 1.6% (1 of 64) |
   | Qwen3-1.7B | 0.137 | 0.172 | 6.3% (4 of 64) |

**The code is correct. This is a genuine negative result for this model, task and group size.** Each task's
outcome is close to deterministic for Qwen3-0.6B: 52 tasks always fail and 11 always succeed, even though the text
varies. GRPO learns only from groups with mixed outcomes, and 20 steps × 8 groups contained about two of them. A
different learning rate or more steps would not change that.

What would help:
- **A policy that sometimes succeeds** on the hard tasks: a larger model (NVIDIA uses a 9B), or an SFT warm start on
  successful trajectories.
- **More tasks per step.** NVIDIA's recipe uses 64 tasks × 16 rollouts = 1024 rollouts per step; ours used 32. When
  outcomes are 0-or-1 per task, more rollouts per task do not help; more tasks do.
- **DAPO dynamic sampling** (`grpo.use_dynamic_sampling: true`, `grpo.batch_multiplier`) keeps only informative
  groups. At 1.6% it needs about 60 tasks per useful group.

**The lesson: measure the informative-group share before an RL run.** One validation pass with
`nemo_rl.val_generations=4 train.steps=1` takes about 5 minutes. A full run with almost no informative groups
cannot learn, whatever its hyperparameters.

### 7.5 Coding-agent RL with PivotRL (NeMo Gym SWE pivots)

NeMo Gym ships software-engineering environments (`swebench`, `swebench_pro`, `swe_gym`, `r2e_gym`, `swe_rebench`,
`deepswe`, `terminal_bench`, …), agent harnesses (OpenHands, mini-SWE-agent, Codex, Claude Code, OpenCode, …), and a
script that converts the task images into Apptainer containers (`examples/nemo_gym/download_swe_images.py`). NeMo RL
has a two-stage recipe for Qwen3-30B-A3B on 16 nodes:
- `swe1` is PivotRL, with no sandbox;
- `swe2` runs OpenHands in real sandboxes, rewarded by hidden tests.

Stage one runs on one GPU, so llmtrain has it as `gym_swe_pivot_nemo_rl.yaml`.

**What a pivot is.** NVIDIA's open dataset `nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1` (CC-BY-4.0, 5.1 GB, 50,661 rows)
is built from real coding-agent trajectories. Each row holds three things:
- the history up to one turn;
- the tools available at that turn;
- the action the successful agent took there.

The policy samples its own next tool call. The verifier
(`single_step_tool_use_with_argument_comparison`, with the recipe's settings) gives 1 if the sampled call matches the
expert's on three counts:
- the same tool;
- the same argument keys;
- matching non-string values (strings always pass at the recipe's threshold of 0).

Training happens only at pivots: turns where the policy's samples sometimes match and sometimes do not. These are the
only turns that give GRPO a non-zero advantage, and they avoid long-horizon credit assignment altogether.

```bash
hf download nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1 train.jsonl --repo-type dataset \
   --local-dir $NEMO_RL_DIR/3rdparty/Gym-workspace/Gym/resources_servers/single_step_tool_use_with_argument_comparison/data
python -m llmtrain -c llmtrain/configs/gym_swe_pivot_nemo_rl.yaml "${ARGS[@]}" nemo_rl.val_generations=4 train.steps=1   # measure
```

llmtrain streams the 5 GB file and prepares the subset:
- It keeps tasks that fit the context: `nemo_rl.gym_max_task_chars: 24000` is ≤ 6.2k Qwen3 tokens, measured at
  ≈ 4.3 characters per token.
- It re-stamps each row with the agent this config starts.
- It splits train and validation by `trajectory_id`, so pivots of one trajectory never straddle the split.

**The pivots must be re-measured on your policy.** NVIDIA pre-filtered every row to 0 < pass rate < 1, but for
NVIDIA's own model. We sampled 1,024 candidate pivots four times each with the real NeMo RL + Gym pipeline:

| policy | informative pivots (0 < successes < 4) | never solved | always solved |
|---|---:|---:|---:|
| Qwen3-1.7B | 27 (2.6 %) | 996 | 1 |
| Qwen3-4B (LoRA) | 45 (4.4 %) | 952 | 27 |

Before trusting such a low score, check whether it is a pipeline bug. We scored plain-vLLM samples with Gym's own
`ActionComparator` and got the same ≈ 5 %. Most failures pick the wrong tool, or the right tool with a different
argument value. So training on the full set would have hit the zero-variance trap of §7.4. `llmtrain.gym_pivots` turns
a measurement run into a PivotRL training set:

```bash
python -m llmtrain -c llmtrain/configs/gym_swe_pivot_nemo_rl.yaml "${ARGS[@]}" nemo_rl.val_generations=4 train.steps=1 \
    nemo_rl.gym_val=<candidates.jsonl> nemo_rl.gym_val_limit=1024 "backend_overrides={++grpo.val_at_end: false}"
python -m llmtrain.gym_pivots outputs/llmtrain/<that run> --out data/gym_swe_pivot/pivots_qwen3_1p7b.jsonl
```

**Training on the policy's own pivots.** 2,000 candidates gave 61 informative pivots from 22 trajectories. We split
them by trajectory into 47 for training and 14 held out, then trained Qwen3-1.7B for 20 steps of 8 pivots × 4 rollouts
(lr 2e-6, β = 0). Validation draws 4 fresh samples per held-out pivot:

| run (Qwen3-1.7B, 47 training pivots, one seed) | held-out step 0 | 5 | 10 | 15 | 20 | train reward, first → last 5 steps |
|---|---:|---:|---:|---:|---:|---|
| full-parameter, one epoch (5 steps, lr 2e-6) | 8.9 % | **17.9 %** | | | | 0.06 → 0.50 (single steps) |
| LoRA r = 16, 20 steps (lr 1e-5) | 5.4 % | **16.1 %** | 7.1 % | 1.8 % | 3.6 % | 0.32 → 0.58 |

- **Both runs show the same shape.** After one pass over the 47 pivots, held-out accuracy roughly doubles, about
  +10 points. At n = 56 samples that is roughly 2.5 standard errors.
- **More passes over-fit.** The training reward keeps rising while held-out accuracy falls below where it started:
  the policy learns 47 specific decisions, not the skill behind them.
- **Data is the bottleneck, not the method.** NVIDIA's recipe trains on about 10k pivots with a 30B model. With 47
  pivots, stop after one epoch and validate often.

How to read it:
- The held-out pivots start lower than when they were measured (25 %). They were selected because they were solved
  sometimes, and part of that was luck: regression to the mean.
- Only the comparison inside one run is fair.
- 14 held-out pivots × 4 samples is a tiny evaluation, and one seed.

Practical notes:
- On an 80 GB H100 shared with other jobs, use micro-batch 1 and `policy.logprob_chunk_size: 2048`. With a
  151k-token vocabulary, the 8k-token logits of two sequences need about 10 GB.
- NeMo RL stops at `grpo.max_num_epochs` (1 in the Gym base config). llmtrain sets it high, so `train.steps` decides
  the length of the run.


### 7.6 On-policy distillation, and why MOPD needs more than one machine

```bash
python -m llmtrain -c llmtrain/configs/opd_gsm8k_nemo_rl.yaml "${ARGS[@]}"           # §5.4, trained by NeMo RL
python -m llmtrain -c llmtrain/configs/recipe_nemo_rl_grpo_smoke.yaml "${ARGS[@]}"   # an official recipe, as is
```

`opd` runs NeMo RL's `run_distillation.py`: the student samples, and the teacher scores the student's tokens with its
top-k logits (`distillation.topk_logits_k`, default 64) under a reverse KL. It is the same experiment as §5.4
(student Qwen2.5-0.5B-Instruct with LoRA, teacher Qwen2.5-1.5B-Instruct, 60 steps):

| backend | GSM8K accuracy (200) | reference-solution NLL | wall-clock |
|---|---|---|---:|
| trl (HF `generate`) | 0.38 → 0.40 | 0.883 → 0.846 | 70 min |
| nemo_rl (vLLM) | 0.38 → 0.40 | 0.883 → 0.822 | **10.7 min** |

The two libraries agree on accuracy (within noise either way). NeMo RL is 6.6× faster, because the student's
samples come from vLLM instead of HF `generate`: being on-policy is cheap only with a fast inference engine. Do not
compare the two logged losses; TRL's is the full-vocabulary KL, NeMo RL's is the top-64 one.

The official recipe (`recipe_nemo_rl_grpo_smoke`, NeMo RL's own 10-step GSM8K smoke test) runs unchanged in
3 minutes, with llmtrain's run directory and metadata around it.

**MOPD** (multi-teacher OPD, lecture Part 4) is in NeMo RL as GRPO with an OPD advantage, Â_t = log π_teacher − log
π_student (`grpo.adv_estimator.name: opd`). A teacher is chosen per Gym agent
(`on_policy_distillation.teacher_model_by_agent_name`). Its recipe,
`examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack.yaml`, uses 3 nodes × 8 GPUs, and the teachers
must be *non-colocated*: each reserves whole nodes, and setup requires `cluster.num_nodes − teacher_nodes ≥ 1`. **It
cannot run on a single machine.** On one GPU, the closest experiment is `opd` with one teacher per data source,
run one after another.

### 7.7 Running on an NVIDIA DLI workshop VM

A DLI workshop VM (2× H100) is not a plain GPU machine. The notebook container (JupyterLab) does not train
itself: an orchestrator starts the workshop's own containers on request, one with NeMo AutoModel for training and
one with a vLLM server for inference, through a `workshop` helper package. `/scratch` is shared and writable;
`/workspace` (the workshop's configs, data and model checkpoints) is read-only. llmtrain therefore does not launch
AutoModel there. With `launch.target: dli` it drives the workshop's helpers instead (`llmtrain/dli.py`):

| step | what llmtrain does |
|---|---|
| data | presets or your JSONL → `data/{train,val}.jsonl` as usual, plus chat records `{"messages": [...]}` (the workshop's `ChatDataset` format, with `dli.system` as the system prompt), under `dli.root` on `/scratch` |
| config | the workshop's own AutoModel YAML (`dli.base_yaml`; default: the first `/workspace/configs/sft_lora*.yaml`), which is known to work on that VM, with our data, `train.steps`, `train.lr`, seed and checkpoint directory. Its model, LoRA, batch and parallelism settings are kept; change them with `backend_overrides` |
| train | stop vLLM (it holds the GPUs), then `workshop.train.train(config_path, run_id)` |
| evaluate | start vLLM, hot-load the LoRA adapter, and send the same held-out prompts to the base model and to the adapter |

```bash
# on the VM, in a JupyterLab terminal, from DeepDataMiningLearning/llm
python llmtrain/dli_probe.py                    # 1. what the VM provides -> /scratch/dli_probe/report.md
python -m llmtrain -c llmtrain/configs/dli_sft_lora.yaml data.path=/scratch/my_task.jsonl dli.system="..."
python -m llmtrain.serve models --url dli       # what vLLM serves now (base + adapters)
```

`model.name: "dli:base"` means the model of the workshop YAML. Any other path must be readable from the
training container, so it must be under `/scratch` or `/workspace`. `train.steps: 0` keeps the YAML's epoch schedule.

**Evaluation is through vLLM, not in-process.** The local evaluation (§8) loads the weights into the evaluating
process, which the notebook container cannot do. On the VM llmtrain reports the following instead:
- qa records: answer accuracy before and after training, with `dli.match`:
  - `number`: the GSM8K rule;
  - `exact`: the whole output equals the label;
  - `contains`: the label occurs in the output;
  - `json:<field>`: the field of the first JSON object in the output equals the label.
- sft records: exact match against the reference. It is meaningful for labels and short answers; for free text,
  read `val_generations_{before,after}.jsonl`.

Response NLL is not available on this path. The workshop's vLLM server has a short context (about 4k tokens), so
long prompts come back as errors, which are counted in `*_eval_errors`.

**Inference and data generation** use `llmtrain/serve.py`. It needs only the standard library, so it also runs in a
bare notebook container. It works against any OpenAI-compatible server: the workshop's (`--url dli`), a local
`vllm serve`, or the workshop's teacher model behind its proxy (`--url dli:teacher`), which is how to generate
synthetic data there:

```bash
python -m llmtrain.serve generate --url dli:teacher --model "" --in seeds.jsonl --template "Write a ... about {topic}" \
    --temperature 0.8 --max-tokens 1024 --workers 8 --out generated.jsonl
python -m llmtrain.serve eval-qa --url dli --model <adapter name> --in val.jsonl --match contains --system "..."
```

What was tested, and what was not:
- **Tested on our H100.** The whole path ran end to end against a stand-in `workshop` package with the same calls.
  That package served vLLM 0.17 with runtime LoRA loading, and trained with the AutoModel CLI on a workshop-style
  `ChatDataset` YAML (unit test: `test_dli_end_to_end_with_fake_workshop`).
- **Not yet tested on a real DLI VM.** Run `dli_probe.py` there first. If a helper's signature differs, its report
  shows it: the source of the `workshop` package is saved with the report.

---

## 8. The common evaluation

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

## 9. Debug or negative result? A case study

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

## 10. Extending llmtrain

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

## 11. Validation results

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
| `grpo_gsm8k_nemo_rl` (same as `grpo_gsm8k`) | nemo_rl | GSM8K answer accuracy (200) | 0.38 | 0.455 |
| `opd_gsm8k_nemo_rl` (same as `opd_gsm8k`) | nemo_rl | GSM8K answer accuracy (200) | 0.38 | 0.40 |
| `gym_workplace_assistant` (Qwen3-0.6B, 20 steps × 8 tasks × 4) | nemo_rl | the environment's verifier (64 tasks) | 0.1875 | 0.1875 (§7.4) |

Wall-clock time per run: SFT 2–3 min, DPO 3–6 min, KD 2 min, GRPO 18 min, OPD 70 min. The last two generate
with HF `generate`; with NeMo RL's vLLM rollouts the same GRPO and OPD runs take 9.5 and 10.7 min (§7).

How to read it:
- **The SFT backends agree within 0.01 nats.** native and TRL compute the same loss and agree to 0.002.
  AutoModel's defaults differ slightly (xavier-initialized LoRA A, warm-up from 0.1·lr, a min-lr floor of 0.1·lr).
  Find out how much each contributes in Exercise 1.
- **DPO agrees across backends.** The model's own ranking did not move (§5.2).
- **GRPO and OPD (+0.02, +0.04) are within noise.** These are 60-step smoke runs, and the standard error is
  ≈ 0.035.
- **KD's +0.07 mixes distillation with SFT on reference solutions** (§5.5).

---

## 12. Exercises

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
7. **Make the agent learn.** On `gym_workplace_assistant`, first measure the informative-group share
   (`nemo_rl.val_generations=4 train.steps=1`, §7.4). Then find a setting with at least 20%, for example a larger
   policy, dynamic sampling, or an SFT warm start. Train it and report validation accuracy with two seeds. Is the
   gain larger than the standard error (≈ 0.05 at 64 tasks)?
8. **Same config, different algorithm.** Make the TRL and NeMo RL GRPO runs of §7.3 compute the same thing: the
   same reward function, the same baseline (`grpo.use_leave_one_out_baseline` in NeMo RL), the same
   normalization. Does the gap in speed or accuracy change?

9. **PivotRL at scale.** Measure 5,000 candidate pivots with your policy (§7.5), train on all informative ones, and
   validate every 2 steps on held-out trajectories. Does the one-epoch gain grow with the number of pivots, and does the
   over-fitting go away?
---

**Files.** `config.py` (the schema), `data.py`, `rewards.py`, `evaluate.py`, `run.py` (entry point),
`backends/{native,trl_backend,automodel_backend,nemo_rl_backend}.py`, `gym_pivots.py`, `configs/*.yaml`, `tests/test_llmtrain.py`,
`_site/sitecustomize.py`.

**See also.** The lecture deck *From GPT to Modern LLMs* (Part 4: InstructGPT → PPO → DPO → the GRPO family →
agentic RL → OPD / MOPD → Nemotron), `TUTORIAL_nanolm.md` §3.7–3.10, the NVIDIA NeMo AutoModel docs
(docs.nvidia.com/nemo/automodel), and NeMo RL (github.com/NVIDIA-NeMo/RL) and NeMo Gym
(github.com/NVIDIA-NeMo/Gym).
