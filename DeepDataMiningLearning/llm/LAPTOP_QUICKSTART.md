# Laptop Quickstart: train language models on your own Mac or laptop

This guide takes you from zero to a working language model, then to a parameter-golf model, and
then to an instruction-tuned LLM. It is meant for an **ordinary laptop**: an Apple-Silicon Mac
(M1 or newer, 8 GB of memory or more) or a laptop with a small NVIDIA GPU. Each step fits in
**1–2 hours** or less. Explanations of *why* are in [`TUTORIAL_nanolm.md`](TUTORIAL_nanolm.md); this page
is the *how*.

| step | what you do | download | time on a Mac* | memory |
|---|---|---:|---:|---:|
| 1 | install + sanity check | ~1 GB (PyTorch) | 10 min | — |
| 2 | measure your machine | — | 1 min | — |
| 3 | Part 1: char LSTM and Transformer on Shakespeare | 1 MB | 5–30 min | < 1 GB |
| 4 | Part 1: tokenizers study (optional) | — | 1–2 h | < 1 GB (4 GB for GPT-2 BPE) |
| 5 | Part 2: GPT on real web text (FineWeb) | 324 MB | 10 min + 1–2 h | < 2 GB |
| 6 | Part 2: evaluate / instruction-tune / preference-tune a real LLM | 1–3 GB | 15 min + 2 × (0.5–1.5 h) | 2–4 GB |
| 7 | Part 3: Parameter Golf mini (baseline + verified stack) | — | 2 × (0.5–1.5 h) | < 2 GB |
| 8 | Part 3: golf ablation (optional, overnight) | — | 5–8 h | < 2 GB |

\* **Estimated** from each config's FLOP count, assuming 0.5–1.5 TFLOP/s on Apple-Silicon MPS. Step 2
gives you the real number for your machine. Every config also has a wall-clock cap, so a slow machine
finishes on time with a slightly weaker model.

---

## Step 1. Install

```bash
# Python 3.10+; a fresh environment is recommended
python -m venv ~/venvs/nanolm && source ~/venvs/nanolm/bin/activate      # or: conda create -n nanolm python=3.11
pip install torch numpy pyyaml sentencepiece tiktoken pandas matplotlib scipy huggingface_hub datasets pytest
pip install transformers peft            # only needed for step 6

git clone https://github.com/lkk688/DeepDataMiningLearning.git
cd DeepDataMiningLearning/DeepDataMiningLearning/llm         # EVERY command below runs from here

python -m pytest nanolm/tests -q                               # all tests should pass (~10 s)
python -m nanolm.train -c nanolm/configs/00_smoke.yaml         # 30 s: trains a tiny model end to end
```

The smoke run prints a line like `[run] device: mps (precision=fp32, compile=no)`.
- On a Mac it must say `mps`. If it says `cpu`, your PyTorch build has no MPS support: upgrade PyTorch, and use Python from an arm64 environment, not Rosetta.
- On an NVIDIA laptop it must say `cuda`.

## Step 2. Measure your machine

```bash
python -m nanolm.bench -c nanolm/configs/laptop/L2_char_gpt.yaml
python -m nanolm.bench -c nanolm/configs/laptop/L4_gpt_fineweb.yaml
```

`bench` runs 13 real training steps on random data and prints tokens/s, memory, and
`projected training time for max_steps=...`. If the projection is much longer than you want,
lower `train.max_steps`. If memory is too high, halve `train.batch_size` and double
`train.grad_accum`: that gives the same result with half the activation memory. Any key can be
changed on the command line:

```bash
python -m nanolm.bench -c nanolm/configs/laptop/L4_gpt_fineweb.yaml train.batch_size=8 train.grad_accum=4
```

## Step 3. Part 1: your first language models (Tiny Shakespeare, downloads itself)

```bash
python -m nanolm.train -c nanolm/configs/laptop/L1_char_lstm.yaml      # LSTM         ~1-3 min
python -m nanolm.train -c nanolm/configs/laptop/L2_char_gpt.yaml       # Transformer  ~4-12 min
python -m nanolm.train -c nanolm/configs/laptop/L1_char_lstm.yaml model.arch=rnn name=L1_char_rnn   # vanilla RNN
```

Each run creates `outputs/nanolm/<date>_<name>/` containing:
- `metrics.json`: the final numbers;
- `sample.txt`: generated text;
- `config.yaml`: the exact settings, so it can be re-run;
- the logs.

**What you should see** (val bits-per-byte, lower is better; a reference run of the same config):

| run | val bpb |
|---|---:|
| L1 LSTM | 2.154 |
| L2 Transformer | 2.106 |
| vanilla RNN (2 seeds, server sweep) | 2.436 |

Your number should be within about ±0.01. Then make it write:

```bash
python -m nanolm.generate outputs/nanolm/<your L2 run dir> --prompt "ROMEO:" --compare-temperatures
python -m nanolm.generate outputs/nanolm/<your L2 run dir> --interactive
```

## Step 4. Part 1: which tokenizer? (optional, ~1–2 h)

```bash
python -m nanolm.sweep   --sweep nanolm/configs/sweeps/laptop_tokenizers.yaml
python -m nanolm.analyze outputs/nanolm/sweeps/laptop_tokenizers --baseline char --metric val_bpb
```

This trains the same model with 8 tokenizers: char, byte, two word vocabularies, BPE and unigram
vocabularies you train yourself, and GPT-2's BPE. `analyze` writes `summary.md`, `curves.png` and
`bars.png` into the sweep directory. Look at `curves.png`: the large vocabularies get *worse* after a
few hundred steps. Tutorial §1.7.1 explains why.

## Step 5. Part 2: pre-train a GPT on real web text

```bash
python -m nanolm.prepare --golf-shards 1 --dest data/golf
```

This downloads 324 MB once: the Parameter Golf FineWeb data, which is 62M validation tokens, 100M
training tokens, and a 1024-token BPE tokenizer.

```bash
python -m nanolm.train -c nanolm/configs/laptop/L4q_gpt_fineweb_quick.yaml    # ~5-15 min, check everything works
python -m nanolm.train -c nanolm/configs/laptop/L4_gpt_fineweb.yaml           # ~45 min - 2 h (capped at 2 h)
```

Reference: L4-quick **1.879** bpb, L4 **1.647** bpb. For scale: GPT-2 small scores 1.079 on the same
documents, and the Parameter Golf winner 1.057 (step 6, and Part 3).

## Step 6. Part 2: work with a real pretrained LLM

These download models from HuggingFace on first use:
- SmolLM2-360M: 0.7 GB
- Qwen2.5-0.5B: 1 GB
- Qwen2.5-0.5B-Instruct: 1 GB

```bash
# (a) how good is a real LLM in our metric?  ~5-15 min, no training
python -m nanolm.finetune -c nanolm/configs/part2/P1_eval_bpb.yaml
python -m nanolm.finetune -c nanolm/configs/part2/P1_eval_bpb.yaml model.name=gpt2 name=P1_gpt2

# (b) instruction tuning (SFT) with LoRA: a base model learns to answer and stop   ~30-90 min
python -m nanolm.finetune -c nanolm/configs/part2/P2_sft_alpaca.yaml
#     8 GB Mac: add  model.name=HuggingFaceTB/SmolLM2-135M

# (c) preference tuning (DPO) with LoRA   ~30-90 min
python -m nanolm.finetune -c nanolm/configs/part2/P3_dpo.yaml
```

Open `samples_before.txt` and `samples_after.txt` in the run directory: they show the same 5 prompts,
before and after tuning. `metrics.json` has the measured before/after numbers.

**What you should see** (reference runs; details in tutorial §2.2–§2.4):

| run | reference result | peak memory (fp32) |
|---|---|---:|
| (a) bpb on FineWeb val, 200 docs | Qwen2.5-0.5B 0.951, SmolLM2-360M 0.928, GPT-2 1.079 | ~3 GB |
| (b) SFT, SmolLM2-360M | held-out answer NLL 1.556 → 1.321; the "after" samples answer and stop | 2.0 GB |
| (c) DPO, Qwen2.5-0.5B-Instruct | held-out reward accuracy 0.50 → 0.68 | 2.9 GB |

The "after" samples are fluent but not always correct. For example, the SFT model puts Paris in
Normandy. SFT and DPO change *how* a model answers, not what it knows (tutorial §2.3).

## Step 7. Part 3: Parameter Golf on a laptop

```bash
python -m nanolm.train -c nanolm/configs/laptop/L5_golf_mini.yaml          # golf baseline architecture
python -m nanolm.train -c nanolm/configs/laptop/L6_golf_mini_stack.yaml    # + the verified technique stack
```

L5 is the golf challenge's baseline model (U-Net skips, QK-norm, relu², Muon, ...) at laptop size.
L6 adds the techniques that survived our ablation, exports in int6 (the "artifact" size is in
`metrics.json`), and evaluates with sliding windows.

**What you should see:**
- L5: **1.633** bpb, 6.1 MB at int8.
- L6: **1.601** bpb, then **1.554** with sliding-window eval, 5.6 MB at int6.

Both fit the laptop "golf budget" of 6.5 MB. Tutorial §3.8 has the full ablation.

## Step 8. Your own ablation (optional, overnight)

```bash
python -m nanolm.sweep   --sweep nanolm/configs/sweeps/laptop_golf_ablation.yaml   # 7 runs; resumable
python -m nanolm.analyze outputs/nanolm/sweeps/laptop_golf_ablation --baseline baseline
```

If you press Ctrl-C, run the same `sweep` command again. Finished runs are skipped, and the
interrupted run restarts. If the process was killed hard (closed terminal, crash), add `--reclaim`
to the command so it clears the stale run claims. If the laptop just sleeps, the run continues when it wakes.

---

## Troubleshooting

| problem | fix |
|---|---|
| `RuntimeError: MPS backend out of memory` / CUDA OOM | `train.batch_size=8 train.grad_accum=4` (same math, half the memory); for step 6 also `data.max_len=256` |
| `NotImplementedError: ... not currently implemented for the MPS device` | prefix the command with `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| a config is too slow | `python -m nanolm.bench ...`, then lower `train.max_steps` (or `train.max_time_sec=1800` for a hard 30-min budget) |
| results differ from the reference by > 0.02 | check the run printed the full number of steps (not stopped by `max_time_sec`), and that `config.yaml` is what you intended |
| HuggingFace downloads fail or are slow | `huggingface-cli login`; in regions where huggingface.co is blocked, set `HF_ENDPOINT` to a mirror |
| an "unknown config key" error | you misspelled a key; the error lists the valid ones (this is a feature) |
| want to run on CPU only | add `device=cpu` (works everywhere, ~3–10× slower) |
| Windows + NVIDIA | keep `train.compile=false` (the laptop configs already do) |

Disk use for everything above: about 4 GB, mostly HuggingFace models in `~/.cache/huggingface`.
