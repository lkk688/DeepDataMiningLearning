# Laptop Quickstart: train language models on your own Mac or laptop

This guide takes you from zero to a working language model, then to a parameter-golf model, and
then to an instruction-tuned LLM. It is meant for an **ordinary laptop**: an Apple-Silicon Mac
(M1 or newer, 8 GB of memory or more) or a laptop with a small NVIDIA GPU. Each step fits in
**1–2 hours** or less. Explanations of *why* are in [`TUTORIAL_nanolm.md`](TUTORIAL_nanolm.md); this page
is the *how*.

Every command, output and timing marked **"measured on Mac"** below was run from scratch on this
machine, following this page:

| | |
|---|---|
| machine | MacBook Pro, **Apple M5 Pro**, 18 CPU cores, **64 GB** unified memory |
| OS | macOS 26.5.2 (arm64) |
| Python | 3.11.17 (installed with `uv`, see Step 1) |
| PyTorch | 2.14.1, device `mps`, precision fp32 (the default) |
| date | 2026-10-18 |

| step | what you do | download | measured on M5 Pro | memory (measured†) |
|---|---|---:|---:|---:|
| 1 | install + sanity check | ~1 GB (PyTorch) | 2–3 min | — |
| 2 | measure your machine | — | 1 min | — |
| 3 | Part 1: char LSTM, Transformer, RNN on Shakespeare | 1 MB | 44 s / 2.7 min / 1.2 min | 0.5–2.7 GB |
| 4 | Part 1: tokenizers study (optional) | — | *not yet measured* (est. 20–40 min) | — |
| 5 | Part 2: GPT on real web text (FineWeb) | 309 MB | 18 s + 3.1 min + 26.7 min | 2.8–3.9 GB |
| 6 | Part 3: evaluate / instruction-tune / preference-tune a real LLM | 3.4 GB | (a) 35 s; (b)–(d) *not yet measured* | 4.9–8.7 GB |
| 7 | Part 4: Parameter Golf mini (baseline + verified stack) | — | L5 23.3 min; L6 *running* (~35 min) | 4.0 GB |
| 8 | Part 4: golf ablation (optional, overnight) | — | *not measured* (est. 3–4 h) | — |

† "peak memory footprint" of the whole process from `/usr/bin/time -l`. On Apple Silicon the CPU
and GPU share memory, so this includes the model, activations, the MPS allocator cache, and about
1–1.5 GB of Metal/PyTorch overhead. It is an upper bound; the same run on a 16 GB Mac caches less.

An M5 Pro is fast: it reached **2.4–3.0 TFLOP/s** in fp32 (Step 2). An M1/M2 base model is
roughly 3–6× slower. Run Step 2 on your own machine to get your numbers. Every config also has a
wall-clock cap, so a slow machine finishes on time with a slightly weaker model.

---

## Step 1. Install

### 1a. Python 3.10+ (a fresh Mac has only Python 3.9)

A new Mac ships only Apple's `/usr/bin/python3` (Python 3.9.6), and there is no `python` command.
nanolm needs **Python 3.10 or newer**. If you have no Homebrew or conda, the quickest way is
[`uv`](https://docs.astral.sh/uv/). It installs into your home folder and needs no admin rights:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv into ~/.local/bin
source $HOME/.local/bin/env                          # or open a new terminal
uv python install 3.11                               # a standalone arm64 CPython 3.11
$(uv python find 3.11) -m venv ~/venvs/nanolm        # a normal venv that includes pip
source ~/venvs/nanolm/bin/activate                   # now `python` and `pip` are 3.11
```

Other options that also work: `conda create -n nanolm python=3.11` (Miniforge), the installer from
python.org, or `brew install python@3.11`. Whatever you use, it must be a native **arm64** Python,
not one running under Rosetta. Check with `python -c "import platform; print(platform.machine())"`,
which must print `arm64`.

### 1b. Packages

```bash
pip install torch numpy pyyaml sentencepiece tiktoken pandas matplotlib scipy huggingface_hub datasets pytest
pip install transformers peft            # only needed for step 6
```

Measured on Mac: 83 s; the venv is 1.0 GB. pip resolved torch 2.14.1, datasets 5.1.0,
transformers 5.20.1, peft 0.21.2, huggingface_hub 2.2.0 and pyarrow 26.0.0.

> **Pitfall (hit on this Mac):** do not put everything into a single `uv pip install ...` command.
> uv's resolver then picked the very old **datasets 2.14.4**, which crashes at import with
> `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'` (step 6 fails, steps 1–5 still
> work). Plain `pip`, as above, gets the current versions. If you already hit the error, run
> `pip install -U datasets transformers peft huggingface_hub`.

### 1c. Get the code and check it

```bash
git clone https://github.com/lkk688/DeepDataMiningLearning.git
cd DeepDataMiningLearning/DeepDataMiningLearning/llm         # EVERY command below runs from here

python -m pytest nanolm/tests -q                               # ~15 s
python -m nanolm.train -c nanolm/configs/00_smoke.yaml         # ~5 s: trains a tiny model end to end
```

Output measured on Mac:

```text
30 passed, 1 skipped in 14.52s          # the skipped test needs the FineWeb data from Step 5; 31 passed after it
```

```text
[run] outputs/nanolm/2026-10-18_15-35-36_smoke
[run] device: mps (precision=fp32, compile=no) | world_size=1
[data] downloading Tiny Shakespeare -> data/nanolm/raw/tinyshakespeare.txt
[model] gpt: 102,784 params (98,624 non-embedding)
[train] 60 steps x 1,024 tokens/step (0.06 passes over train) | optimizer=adamw schedule=cosine
step     20 | loss 3.4596 | lr 2.39e-03 | 101.7k tok/s | 0s
  eval step     30 | val_loss 2.8183 | val_bpb 4.0659 | acc 25.0% | 0s
step     60 | loss 2.6748 | lr 3.02e-04 | 304.2k tok/s | 0s
[final] val_loss 2.6544 | val_bpb 3.8295 | ppl 14.22 | acc 27.22% | 32,768 tokens, 1.00 bytes/token
```

The second line is the one to check.
- On a Mac it must say `mps`. If it says `cpu`, your PyTorch build has no MPS support: upgrade
  PyTorch, and use an arm64 Python (see 1a).
- On an NVIDIA laptop it must say `cuda`.

> Fixed while testing on this Mac: earlier versions crashed at the first evaluation with
> `TypeError: Cannot convert a MPS Tensor to float64 dtype`, because MPS has no float64. The
> evaluator now accumulates its totals on the CPU when running on MPS. If you see this error,
> `git pull`.

## Step 2. Measure your machine

```bash
python -m nanolm.bench -c nanolm/configs/laptop/L2_char_gpt.yaml
python -m nanolm.bench -c nanolm/configs/laptop/L4_gpt_fineweb.yaml
```

`bench` runs 13 real training steps on random data and prints tokens/s, memory, and
`projected training time for max_steps=...`. Example output measured on Mac:

```text
[bench] L4_gpt_fineweb on mps (precision=fp32, compile=no): 11.02M params, 16x512x2 = 16,384 tokens/step
[bench] 429 ms/step | 38.2k tokens/s | ~2.98 TFLOP/s achieved
[bench] MPS memory (Metal driver, incl. cache) 3.13 GB
[bench] projected training time for max_steps=3000: 21.4 min (+ periodic eval); capped at max_time_sec=7200s
```

All laptop configs, measured on Mac (M5 Pro, fp32, and with `precision=bf16` for comparison):

| config | params | fp32 tok/s | fp32 TFLOP/s | projected | **actual** | bf16 tok/s |
|---|---:|---:|---:|---:|---:|---:|
| L1_char_lstm | 1.07M | 473k | 2.99 | 0.6 min | 0.7 min | — |
| L2_char_gpt | 3.16M | 112k | 2.45 | 2.4 min | 2.7 min | 146k (1.3×) |
| L4q_gpt_fineweb_quick | 3.41M | 97k | 2.44 | 2.8 min | 3.1 min | — |
| L4_gpt_fineweb | 11.0M | 38k | 2.98 | 21.4 min | 26.7 min | 52k (1.36×) |
| L5_golf_mini | 6.60M | 32k | 1.65 | 21.2 min | 23.3 min | — |
| L6_golf_mini_stack | 8.37M | 21k | 1.27 | 33.3 min | *running* | 23k (1.12×) |

The "actual" time is the whole command: data loading, periodic evals, the final eval, export.
It is 10–25 % above the projection.

**Tips for the Mac**
- `precision=bf16` (macOS 14+) is 1.1–1.4× faster on this Mac. The reference numbers on this page
  are fp32, so use fp32 when you want to compare with them.
- If the projection is much longer than you want, lower `train.max_steps`.
- If memory is too high, halve `train.batch_size` and double `train.grad_accum`. That gives the same
  result with half the activation memory.
- Any key can be changed on the command line:

```bash
python -m nanolm.bench -c nanolm/configs/laptop/L4_gpt_fineweb.yaml train.batch_size=8 train.grad_accum=4
```

## Step 3. Part 1: your first language models (Tiny Shakespeare, downloads itself)

```bash
python -m nanolm.train -c nanolm/configs/laptop/L1_char_lstm.yaml      # LSTM
python -m nanolm.train -c nanolm/configs/laptop/L2_char_gpt.yaml       # Transformer
python -m nanolm.train -c nanolm/configs/laptop/L1_char_lstm.yaml model.arch=rnn name=L1_char_rnn   # vanilla RNN
```

Each run creates `outputs/nanolm/<date>_<name>/` containing:
- `metrics.json`: the final numbers;
- `sample.txt`: generated text;
- `config.yaml`: the exact settings, so it can be re-run;
- the logs.

**Results measured on Mac** (val bits-per-byte, lower is better), next to the reference runs:

| run | params | wall time | peak memory† | val bpb (Mac) | val bpb (reference) |
|---|---:|---:|---:|---:|---:|
| L1 LSTM | 1.07M | 44 s | 1.8 GB | **2.1447** | 2.154 |
| L2 Transformer | 3.16M | 161 s | 2.7 GB | **2.0986** | 2.106 |
| vanilla RNN | 0.28M | 72 s | 0.5 GB | **2.4247** | 2.436 (2 seeds, server sweep) |

All three are within 0.012 of the reference. The Mac numbers are slightly *better* than the
reference; small differences like this come from different random-number streams on different devices.
The vanilla RNN runs its time loop step by step, so it uses mostly the CPU (68 s of CPU time).

The end of the L1 log:

```text
[train] 2000 steps x 8,192 tokens/step (16.32 passes over train) | optimizer=adamw schedule=cosine | time budget 1800s
[final] val_loss 1.4866 | val_bpb 2.1447 | ppl 4.42 | acc 55.86% | 111,488 tokens, 1.00 bytes/token
[sample]

Canst thou denied all root is ease to captain this.

First Gentleman:
That insterment seven that is confess behall
Than that I hope a grace, I am poes,
```

Then make the Transformer write:

```bash
python -m nanolm.generate outputs/nanolm/<your L2 run dir> --prompt "ROMEO:" --compare-temperatures
python -m nanolm.generate outputs/nanolm/<your L2 run dir> --interactive
```

Measured on Mac (shortened). Temperature 0 repeats itself, 0.5–1.0 is Shakespeare-like, and
1.5 starts to invent words:

```text
===== temperature 0.0 =====
ROMEO:
I will not stay to the state of the state.
And then the state of the state of the state,
And then the state of the state of the state,

===== temperature 1.0 =====
ROMEO:
Why must I swear? I would never brought to thee,
Is a with that here hath not so believe.

ROMEO:
March thee well death: my friend, no; my father?

===== temperature 1.5 =====
ROMEO:
Portchful hards:
Lestow a meaner void I tleep greeping years:
```

## Step 4. Part 1: which tokenizer? (optional)

```bash
python -m nanolm.sweep   --sweep nanolm/configs/sweeps/laptop_tokenizers.yaml
python -m nanolm.analyze outputs/nanolm/sweeps/laptop_tokenizers --baseline char --metric val_bpb
```

This trains the same model with 8 tokenizers: char, byte, two word vocabularies, BPE and unigram
vocabularies you train yourself, and GPT-2's BPE. `analyze` writes `summary.md`, `curves.png` and
`bars.png` into the sweep directory. Look at `curves.png`: the large vocabularies get *worse* after a
few hundred steps. Tutorial §1.8 explains why.

*Not yet measured on the Mac.* Each run is the size of L2 (2.7 min on the M5 Pro), so expect
about 20–40 min here and 1–2 h on an M1/M2.

## Step 5. Part 2: pre-train a GPT on real web text

```bash
python -m nanolm.prepare --golf-shards 1 --dest data/golf
```

This downloads the Parameter Golf FineWeb data once: 62M validation tokens, 100M training tokens,
and a 1024-token BPE tokenizer. On the Mac: **309 MB, 18 s**.

```bash
python -m nanolm.train -c nanolm/configs/laptop/L4q_gpt_fineweb_quick.yaml    # check everything works
python -m nanolm.train -c nanolm/configs/laptop/L4_gpt_fineweb.yaml           # the real run (capped at 2 h)
```

**Results measured on Mac:**

| run | params | wall time | peak memory† | val bpb (Mac) | val bpb (reference) |
|---|---:|---:|---:|---:|---:|
| L4-quick (4 × 256) | 3.4M | 3.1 min | 2.8 GB | **1.8842** | 1.879 |
| L4 (6 × 384, Muon, 49M tokens) | 11.0M | 26.7 min | 3.9 GB | **1.6476** | 1.647 |

The L4 log, measured on Mac. The bpb drops fastest at the end, in the WSD "warm-down" phase
(tutorial §2):

```text
[model] gpt: 11,015,040 params (10,621,824 non-embedding)
[train] 3000 steps x 16,384 tokens/step (0.49 passes over train) | optimizer=muon schedule=wsd | time budget 7200s
  eval step    500 | val_loss 3.4582 | val_bpb 2.0814 | acc 26.7% | 230s
  eval step   1000 | val_loss 3.1206 | val_bpb 1.8782 | acc 33.3% | 464s
  eval step   1500 | val_loss 2.9869 | val_bpb 1.7978 | acc 35.6% | 701s
  eval step   2000 | val_loss 2.9181 | val_bpb 1.7563 | acc 36.9% | 943s
  eval step   2500 | val_loss 2.8027 | val_bpb 1.6869 | acc 38.7% | 1181s
  eval step   3000 | val_loss 2.7187 | val_bpb 1.6364 | acc 40.3% | 1425s
[final] val_loss 2.7466 | val_bpb 1.6476 | ppl 15.59 | acc 39.93% | 2,097,152 tokens, 2.41 bytes/token
```

Its sample is grammatical web-style English with no real meaning:

```text
This article is a real public hip-hop driven by the victim of the Witches Hope. For more information
on the Twitter and we'll add your victims to the page, ...
```

For scale: GPT-2 small scores 1.075 on the same documents (step 6), and the Parameter Golf winner
1.057 (Parts 3–4).

## Step 6. Part 3: work with a real pretrained LLM

These download models from HuggingFace on first use (3.4 GB in about 1 min on the test Mac):
- SmolLM2-360M: 0.7 GB
- Qwen2.5-0.5B: 1 GB
- Qwen2.5-0.5B-Instruct: 1 GB

```bash
# (a) how good is a real LLM in our metric?  no training
python -m nanolm.finetune -c nanolm/configs/part2/P1_eval_bpb.yaml
python -m nanolm.finetune -c nanolm/configs/part2/P1_eval_bpb.yaml model.name=gpt2 name=P1_gpt2

# (b) instruction tuning (SFT) with LoRA: a base model learns to answer and stop   ~30-90 min
python -m nanolm.finetune -c nanolm/configs/part2/P2_sft_alpaca.yaml
#     8 GB Mac: add  model.name=HuggingFaceTB/SmolLM2-135M

# (c) preference tuning (DPO) with LoRA   ~30-90 min
python -m nanolm.finetune -c nanolm/configs/part2/P3_dpo.yaml

# (d) the benchmarks model releases report (ARC, HellaSwag, MMLU, GSM8K), subsampled   ~30-60 min
pip install lm-eval langdetect immutabledict
python -m nanolm.benchmarks run -c nanolm/configs/part3/B2_laptop.yaml
python -m nanolm.benchmarks summary outputs/nanolm/benchmarks/laptop
```

**(a) Measured on Mac:**

```text
[run] eval_bpb | Qwen/Qwen2.5-0.5B | device mps (precision=fp32, compile=no)
[eval_bpb] Qwen/Qwen2.5-0.5B: val_bpb 0.9511 on 200 docs (4.57 bytes/token)
[eval_bpb] gpt2: val_bpb 1.0750 on 200 docs (4.56 bytes/token)
```

| model | wall time (Mac) | peak memory† | val bpb (Mac) | val bpb (reference) |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B (494M) | 35 s | 8.7 GB | **0.9511** | 0.951 |
| GPT-2 small (124M) | 15 s | 4.9 GB | **1.0750** | 1.079 |

The memory is high because the model outputs fp32 logits over Qwen's 151,936-token vocabulary
for every 1,024-token window, and the MPS allocator keeps those buffers cached on a 64 GB machine.
On an 8–16 GB Mac it caches less. If it runs out of memory, add `train.ctx_len=512`.
The bpb will then be slightly higher, because each token sees less context.

**(b)–(d): not yet measured on the Mac** (these runs are queued). Until then, here are the reference results
(details in tutorial §3.5 and §3.8–§3.9). Open `samples_before.txt` and `samples_after.txt` in the run
directory: they show the same 5 prompts, before and after tuning. `metrics.json` has the measured
before/after numbers.

| run | reference result | peak memory (fp32) |
|---|---|---:|
| (b) SFT, SmolLM2-360M | held-out answer NLL 1.556 → 1.321; the "after" samples answer and stop | 2.0 GB |
| (c) DPO, Qwen2.5-0.5B-Instruct | held-out reward accuracy 0.50 → 0.68 | 2.9 GB |

The "after" samples are fluent but not always correct. For example, the SFT model puts Paris in
Normandy. SFT and DPO change *how* a model answers, not what it knows (tutorial §3.8).

## Step 7. Part 4: Parameter Golf on a laptop

```bash
python -m nanolm.train -c nanolm/configs/laptop/L5_golf_mini.yaml          # golf baseline architecture
python -m nanolm.train -c nanolm/configs/laptop/L6_golf_mini_stack.yaml    # + the verified technique stack
```

L5 is the golf challenge's baseline model (U-Net skips, QK-norm, relu², Muon, ...) at laptop size.
L6 adds the techniques that survived our ablation, exports in int6 (the "artifact" size is in
`metrics.json`), and evaluates with sliding windows.

**L5 measured on Mac:** 23.3 min, 4.0 GB peak memory†.

```text
[model] gpt: 6,601,764 params (6,208,548 non-embedding)
[train] 2500 steps x 16,384 tokens/step (0.41 passes over train) | optimizer=muon schedule=wsd | time budget 5400s
  eval step    500 | val_loss 3.3804 | val_bpb 2.0018 | acc 28.4% | 275s
  eval step   1500 | val_loss 2.9296 | val_bpb 1.7348 | acc 36.4% | 820s
  eval step   2500 | val_loss 2.7082 | val_bpb 1.6037 | acc 40.2% | 1358s
[final] val_loss 2.7233 | val_bpb 1.6336 | ppl 15.23 | acc 40.26% | 2,097,152 tokens, 2.41 bytes/token
[export] int8+lzma: 6.089 MB model + 22 KB code | round-trip val_bpb 1.6337 (gap +0.0001)
```

| run | val bpb (Mac) | val bpb (reference) | artifact size |
|---|---:|---:|---:|
| L5 | **1.6336** (int8 round-trip 1.6337) | 1.633 | 6.11 MB (int8) |
| L6 | *running* | 1.601, then 1.554 with sliding-window eval | 5.6 MB (int6, reference) |

Both fit the laptop "golf budget" of 6.5 MB. Tutorial §4.8 has the full ablation.

## Step 8. Your own ablation (optional, overnight)

```bash
python -m nanolm.sweep   --sweep nanolm/configs/sweeps/laptop_golf_ablation.yaml   # 7 runs; resumable
python -m nanolm.analyze outputs/nanolm/sweeps/laptop_golf_ablation --baseline baseline
```

If you press Ctrl-C, run the same `sweep` command again. Finished runs are skipped, and the
interrupted run restarts. If the process was killed hard (closed terminal, crash), add `--reclaim`
to the command so it clears the stale run claims. If the laptop just sleeps, the run continues when it wakes.
Not measured on the Mac. With 7 runs of 23–35 min each, expect 3–4 h on an M5 Pro.

**Long runs on a Mac:** a MacBook on battery may sleep, and the run then pauses. Keep it on power and
run `caffeinate -i python -m nanolm.sweep ...` to stop it from sleeping. Use `nohup ... &` if you want
to close the terminal.

---

## Troubleshooting

| problem | fix |
|---|---|
| `python: command not found`, or Python 3.9 | a fresh Mac has only `/usr/bin/python3` 3.9; install 3.11 (Step 1a) and activate the venv |
| `TypeError: Cannot convert a MPS Tensor to float64 dtype` | old code; `git pull` (fixed in the evaluator) |
| `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'` | `datasets` is too old: `pip install -U datasets transformers peft huggingface_hub` |
| `RuntimeError: MPS backend out of memory` / CUDA OOM | `train.batch_size=8 train.grad_accum=4` (same math, half the memory); for step 6 also `data.max_len=256` |
| `NotImplementedError: ... not currently implemented for the MPS device` | prefix the command with `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| a config is too slow | `python -m nanolm.bench ...`, then lower `train.max_steps` (or `train.max_time_sec=1800` for a hard 30-min budget); on macOS 14+ try `precision=bf16` |
| results differ from the reference by > 0.02 | check the run printed the full number of steps (not stopped by `max_time_sec`), and that `config.yaml` is what you intended |
| HuggingFace downloads fail or are slow | `hf auth login` (or set `HF_TOKEN`); in regions where huggingface.co is blocked, set `HF_ENDPOINT` to a mirror |
| an "unknown config key" error | you misspelled a key; the error lists the valid ones (this is a feature) |
| want to run on CPU only | add `device=cpu` (works everywhere, ~3–10× slower) |
| Windows + NVIDIA | keep `train.compile=false` (the laptop configs already do) |

Disk use for everything above: about 5 GB in total. That is 1.0 GB for the venv, 0.3 GB of FineWeb
data in `data/`, and 3.4 GB of HuggingFace models in `~/.cache/huggingface`.
