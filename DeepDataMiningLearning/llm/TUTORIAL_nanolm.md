# From RNNs to Parameter Golf: a hands-on language-modeling course with `nanolm`

This course goes from a character-level RNN to the techniques behind the winning entries of
OpenAI's **Parameter Golf** challenge (2026). The same small codebase, `llm/nanolm/`, is used
throughout. Every experiment is a **YAML config** that you run with one command on a **Mac (Apple
Silicon)**, a laptop with an **NVIDIA GPU**, a **server GPU**, or a plain **CPU**. Every run saves
its config, metadata, logs and final metrics, so results can be analyzed, compared across seeds, and
reproduced.

**Students with a laptop: start with [`LAPTOP_QUICKSTART.md`](LAPTOP_QUICKSTART.md).** It is a
step-by-step guide (download, run, expected time, memory, and numbers) where every step fits in
1–2 hours on an ordinary Mac.

| Part | Topic | Laptop configs | Server-scale configs |
|---|---|---|---|
| 0 | Setup, the laptop track, the experiment workflow | `00_smoke.yaml` | |
| 1 | Foundations: the LM task, bits-per-byte, RNN → LSTM → Transformer, tokenizers, sampling | `laptop/L1–L3`, `sweeps/laptop_tokenizers.yaml` | `01_*`, `02_*`, `sweeps/01_*`, `sweeps/02_*` |
| 2 | Transformers for real: pre-training on web text; using, instruction-tuning (SFT) and preference-tuning (DPO) pretrained models | `laptop/L4*`, `part2/P1–P3` | `03_*`, `sweeps/03_*` |
| 3 | Parameter Golf: the challenge, the winners, ablations, compression | `laptop/L5–L6`, `sweeps/laptop_golf_ablation.yaml` | `04_*`, `05_*`, `sweeps/04_*` |
| 4 | Methodology: seeds, noise, debug vs. negative result, a post-mortem | `analyze.py` | |

> **Where the numbers come from.** Every result table in this tutorial was produced by the commands
> shown next to it, on NVIDIA H100 GPUs. For some runs another job was using most of the GPU at the
> same time, so wall-clock times are pessimistic. The raw run directories are under
> `outputs/nanolm/sweeps/`. Numbers about the public leaderboard are quoted from the official
> repository and linked. Nothing is copied from memory.
> Laptop-track reference numbers were produced with `precision=fp32`, the Mac default, so that
> memory and accuracy match what a Mac run reports.

---

# Part 0 — Getting started

## 0.1 Install

```bash
# any machine: Python >= 3.10, PyTorch >= 2.4
pip install torch numpy pyyaml sentencepiece tiktoken pandas matplotlib scipy huggingface_hub datasets pytest
pip install transformers peft                 # Part 2 (pretrained models) only
cd DeepDataMiningLearning/llm                 # all commands below run from this directory
python -m pytest nanolm/tests -q              # correctness tests, ~5 s on CPU
python -m nanolm.train -c nanolm/configs/00_smoke.yaml     # 30-second end-to-end run
```

`device: auto` chooses **CUDA → MPS (Apple Silicon) → CPU** in that order. Nothing in the code says
`.cuda()`. The device-specific choices all live in `nanolm/device.py`:

| | CUDA | MPS (Mac) | CPU |
|---|---|---|---|
| mixed precision (`precision: auto`) | bf16 autocast | fp32 (opt in with `precision=bf16`, needs macOS 14+) | fp32 |
| `torch.compile` (`train.compile`) | yes | ignored | ignored |
| fused AdamW | yes | no | no |
| Muon Newton-Schulz | bf16 | fp32 | fp32 |
| multi-GPU (`torchrun`) | NCCL | — | gloo |

The official challenge shipped a separate Apple-MLX script (`train_gpt_mlx.py`) next to its CUDA
script. That means two codebases that drift apart. We chose one PyTorch codebase with a device
abstraction instead: every feature works on every backend, and a student's Mac result and a GPU
result come from the same code.

> **Honest caveat.** The MPS path is written to avoid CUDA-only operations, and the CPU path
> (which exercises the same non-CUDA branches) is tested. It has **not** been benchmarked on a Mac by
> us. If something fails on MPS, run with `device=cpu` to check whether it is MPS-specific, and
> please report it.

## 0.2 The laptop track: what runs where

Everything in this course has a **laptop configuration** in `nanolm/configs/laptop/` (Parts 1–3) and
`nanolm/configs/part2/` (Part 2, pretrained models). Each one is sized for an ordinary laptop:

- **Compute.** Each config does at most about 4 PFLOP of training (FLOPs ≈ 6 × parameters × tokens).
  On an Apple-Silicon Mac this means minutes for Part 1 and roughly 1–2 hours for the biggest runs.
- **Time cap.** Each config also sets `train.max_time_sec`. If your machine is slower than expected,
  training stops at the cap, and the learning-rate schedule (which follows *progress* =
  max(step/max_steps, time/max_time_sec)) still decays properly. You get a slightly weaker model,
  not a crash at 3 a.m.
- **Memory.** Measured fp32 peaks are at most 2.9 GB, for both the from-scratch and the
  post-training configs, so an 8 GB Mac works. Close other heavy apps.

**Measure your own machine first.** Run `python -m nanolm.bench -c <config>`. It runs 13 real
training steps and prints tokens/s, memory, and the projected training time for that config.

| config | what it trains | FLOPs | est. time on a Mac* | peak memory (fp32) | reference val bpb |
|---|---|---:|---:|---:|---:|
| `00_smoke` | tiny GPT, 60 steps | — | < 1 min | < 0.2 GB | (sanity only) |
| `laptop/L1_char_lstm` | 2-layer LSTM, Shakespeare chars | 0.1 P | 1–3 min | 0.3 GB | 2.154 |
| `laptop/L2_char_gpt` | 4-layer modern Transformer, Shakespeare | 0.36 P | 4–12 min | 0.8 GB | 2.106 |
| `laptop/L3_tokenizer` | same, your own BPE-1024 tokenizer | 0.36 P | 4–12 min | 0.9 GB | 2.279 (best checkpoint lower, §1.7.1) |
| `laptop/L4q_gpt_fineweb_quick` | 4 × 256 GPT on FineWeb, 16M tokens | 0.4 P | 5–14 min | 0.9 GB | 1.879 |
| `laptop/L4_gpt_fineweb` | 6 × 384 GPT on FineWeb, 49M tokens, Muon | 3.8 P | 45–130 min (cap 2 h) | 1.8 GB | 1.647 |
| `laptop/L5_golf_mini` | golf baseline architecture, 6 × 384, 41M tokens | 2.1 P | 25–70 min | 1.8 GB | 1.633 |
| `laptop/L6_golf_mini_stack` | + the verified golf stack, int6, sliding eval | 3.1 P | 35–105 min | 2.9 GB | 1.601 (sliding: 1.554) |
| `part2/P1_eval_bpb` | score Qwen2.5-0.5B on FineWeb (no training) | — | 5–15 min | ~3 GB (est.) | 0.951 |
| `part2/P2_sft_alpaca` | LoRA instruction tuning of SmolLM2-360M | ~3 P | 30–90 min | 2.0 GB | answer NLL 1.556 → 1.321 |
| `part2/P3_dpo` | LoRA DPO of Qwen2.5-0.5B-Instruct | ~3 P | 30–90 min | 2.9 GB | reward acc 0.50 → 0.68 |

\* **Estimated.** The time range assumes 0.5–1.5 TFLOP/s achieved on MPS in fp32. These are not
measurements from a Mac. Your `bench` output is the real number. The reference bpb and memory
were measured on a GPU with `precision=fp32` (the Mac default), so your result should match the
reference to within about ±0.01 bpb. The exception is when the time cap stops your run early.

On a **laptop with an NVIDIA GPU** (4–8 GB of VRAM), the same configs run in bf16 and use roughly
half the memory shown. Keep `train.compile=false` on Windows.

## 0.3 Code map

```
llm/nanolm/
  config.py      typed dataclass configs; YAML + inheritance + key=value overrides; rejects typos
  device.py      cuda/mps/cpu selection, autocast, seed_everything(), synchronize()
  data.py        tokenizers (char, byte, word, gpt2-BPE, SentencePiece, train-your-own BPE, any HF)
                 -> token shards -> batches; per-token byte lookup tables for bits-per-byte
  models/rnn.py  RNN / GRU / LSTM (+ a hand-written LSTM cell)
  models/gpt.py  ONE GPT class; every parameter-golf trick is a config switch (default off)
  optim.py       AdamW groups, Muon (Newton-Schulz), cosine / WSD schedules, EMA
  evaluate.py    loss, bpb, sliding-window eval (each token scored exactly once), sampling
  quant.py       int8/6/5/4 per-row quantization + zlib/lzma -> "artifact" size + round-trip bpb
  train.py       Parts 1, 3: the single training entry point for from-scratch models
  finetune.py    Part 2: pretrained HF models -- bits-per-byte eval, LoRA SFT, LoRA DPO
  generate.py    sample from a checkpoint (temperature / top-k / top-p, interactive)
  bench.py       measure THIS machine: tokens/s, memory, projected time of a config
  sweep.py       experiments x seeds from one sweep file; resumable; several machines can share it
  analyze.py     tables (mean +- std over seeds, paired t-test vs. baseline) + plots
  compress.py    re-evaluate a saved model: quantization bits x compressor x eval stride
  prepare.py     download / tokenize data ahead of time
  configs/       one YAML per experiment: laptop/ (L1-L6), part2/ (P1-P3), server-scale 00-05,
                 sweeps/ one YAML per study
  tests/         correctness tests (causality of every feature, bpb byte counting, tokenizers, ...)
llm/legacy/      the previous course code, with a map from each old function to its replacement
```

## 0.4 The experiment workflow (the important part)

**Rule 1: an experiment is a config, never an edited line of code.** Configs inherit:

```yaml
# configs/02b_char_transformer_modern.yaml  -- the whole file
defaults: [02_char_transformer.yaml]
name: char_gpt_modern
model: {norm: rmsnorm, pos: rope, mlp: swiglu, bias: false}
```

Any key can also be overridden from the command line:

```bash
python -m nanolm.train -c nanolm/configs/02_char_transformer.yaml model.n_layers=6 train.lr=5e-4 seed=2
```

Every key is typed, and a misspelled key is an error. For example, `model.n_layer=6` stops with
`unknown config key(s) ['n_layer']`. In our own parameter-golf code, configuration went through
environment variables, and an unknown key only produced a warning. Several "ablations" there
silently changed nothing (§4.3).

**Rule 2: every run leaves a complete record.** `outputs/nanolm/<date>_<name>/` holds:

| file | content |
|---|---|
| `config.yaml` | the fully resolved config; rerun the run with `--config <this file>` |
| `meta.json` | git commit + dirty flag, seed, date, host, device, torch version, command line |
| `train_log.jsonl` | step, loss, lr, tokens/s every `log_every` steps |
| `eval_log.jsonl` | val loss / bpb every `eval_every` steps |
| `metrics.json` | final full-validation bpb, sliding-window bpb, quantized size and round-trip bpb, time, memory |
| `sample.txt`, `model.pt` | generated text, checkpoint |

**Rule 3: a study is a sweep file, and every claim needs seeds.**

```bash
python -m nanolm.sweep  --sweep nanolm/configs/sweeps/04_golf_ablation.yaml --gpus 0,1,2,3
python -m nanolm.analyze outputs/nanolm/sweeps/golf_ablation
```

The sweep validates every config before launching anything. It runs each (experiment, seed) in its
own process, and skips runs that already have a `metrics.json`, so rerunning the command resumes a
crashed sweep. `analyze.py` reports mean ± std over seeds and a paired t-test against the baseline,
and plots the curves. Part 4 explains why this is not optional.

**Rule 4: the code is tested.** The most important test is causality. Every model feature is
checked to make sure that changing *future* tokens cannot change *past* predictions. A model that
leaks the future gets a spectacular and meaningless score (§4.3 shows how this happened to us).

---

# Part 1 — Foundations: from characters to Transformers

## 1.1 The task

A language model assigns a probability to the next token given the past:
p(x_t | x_<t). Training minimizes the cross-entropy, i.e. the average −ln p(x_t | x_<t) over the
training tokens. In `nanolm`, every model maps `idx[B, T] → logits[B, T, V]`, and the loss in
`train.py` is one line:

```python
loss = F.cross_entropy(logits.float().view(-1, V), y.reshape(-1))     # y = idx shifted by one
```

## 1.2 Why bits-per-byte, not loss or perplexity

Per-token loss and perplexity depend on what a "token" is. A character model predicts about 1 byte
per step. A GPT-2 BPE model predicts about 4 bytes per step, so its per-token loss is naturally
larger. Comparing the two numbers is meaningless. The old tokenizer-comparison experiment in this
folder (now `legacy/experiments.py: run_tokenizer_comparison_experiment`) made exactly this mistake.

**Bits-per-byte** removes the unit. It is the number of bits the model needs to compress the
validation text, divided by the length of that text in UTF-8 bytes:

```
bpb = (sum over scored tokens of -ln p(token)) / ln 2 / (number of UTF-8 bytes those tokens cover)
```

It is the official Parameter Golf metric, and it is what `nanolm` reports as `val_bpb`.
Counting bytes per token is subtle for SentencePiece, where the leading "▁" stands for a space byte
that is not counted after a document boundary. `data.py` builds three lookup tables per tokenizer
that reproduce the challenge's rule exactly:
`bytes(y) = base_bytes[y] + (has_leading_space[y] & ~is_boundary[x_prev])`.
`tests/test_nanolm.py` checks that the counted bytes equal `len(text.encode("utf-8"))`.

Reference points: a uniform guess over 256 byte values costs 8 bpb. gzip on English text gets about
2.5–3 bpb. The Parameter Golf baseline (a 17M-parameter GPT, 10 min on 8xH100) gets 1.2244, and the
winning entry gets 1.0565.

## 1.3 The data pipeline (borrowed from Parameter Golf)

1. **Tokenize once.** Store all token ids as one flat `uint16` array in the challenge's `.bin` shard
   format (a 256-int header, then the tokens; `uint32` for vocabularies above 65,535), and memory-map it (`data.py: write_shard, read_shard`).
   A 100M-token shard costs no RAM until it is read.
2. **Sample windows.** Each training batch is `B` random windows of `T+1` tokens (`sampler: random`),
   or the next contiguous chunk of the stream (`sampler: sequential`, the golf loader, which never
   repeats data).
3. **Count steps and tokens, not epochs.** `train.max_steps × batch_size × seq_len × grad_accum` is
   the token budget. The log prints how many passes over the training set that is.
4. **Fix the validation split by the data, not by the run seed.** Use the contiguous tail of a
   corpus (Shakespeare), a seeded *document* shuffle (`split_by: doc`), or the challenge's fixed
   first-50k-documents split.

## 1.4 What was wrong with the old pipeline (worth discussing in class)

| Old behavior (`legacy/lm_dataset.py` / `legacy/train_lm.py`) | Consequence | nanolm |
|---|---|---|
| sliding windows with `stride=1`, stored as Python lists | 13.7M overlapping 64-token "sequences" for 13.7M characters; one "epoch" shows each character 64 times; about 12 min per epoch for a 2-layer RNN | windows sampled from a memory-mapped token array |
| validation = last 10% of an **alphabetically sorted** dictionary | validation is all words starting with late letters (distribution shift). The log shows val loss *rising* from epoch 1 while train loss falls | data-defined random doc split or contiguous tail |
| tokenizer comparison by per-token loss / perplexity | units differ, so the ranking is meaningless | bits-per-byte |
| device = `cuda` or `cpu` | Mac students fall back to CPU | cuda → mps → cpu |
| `FullTransformer` built `[layer for _ in range(n)]` | the same layer object was repeated, so all N layers shared weights | fixed (in `legacy/transformer.py`) |
| hyper-parameters spread over `Args` classes inside `legacy/experiments.py` | an experiment means editing code | YAML + overrides |

---

## 1.5 Recurrent language models: RNN → GRU → LSTM

Code: `nanolm/models/rnn.py`. Config: `configs/01_char_rnn.yaml` (laptop: `configs/laptop/L1_char_lstm.yaml`, the same run).

A recurrent LM reads one token at a time and carries a hidden state h_t:

- **Vanilla RNN:** h_t = tanh(W_x x_t + W_h h_{t-1}). Gradients flowing back through T steps are
  multiplied by W_h about T times, so they explode or vanish. Gradient clipping (`train.grad_clip`)
  fixes the explosion. Nothing simple fixes the vanishing.
- **LSTM:** adds a cell c_t that is updated *additively*, c_t = f ⊙ c_{t-1} + i ⊙ g. The forget gate f
  lets information (and gradient) flow across long spans. `LSTMScratch` writes the four gates out
  explicitly, and initializes the forget-gate bias to 1 so the cell remembers by default.
- **GRU:** a cheaper two-gate variant.

Training uses **truncated back-propagation through time**: each window of `seq_len` characters
starts from a zero state. RNNs and Transformers therefore share the data pipeline and the evaluation
exactly.

```bash
python -m nanolm.train -c nanolm/configs/01_char_rnn.yaml                      # LSTM
python -m nanolm.train -c nanolm/configs/01_char_rnn.yaml model.arch=gru name=char_gru
python -m nanolm.train -c nanolm/configs/01_char_rnn.yaml model.arch=lstm_scratch train.max_steps=200  # readable, slow
```

The results for all architectures are in the table in §1.6.3.

---

## 1.6 The character-level Transformer

Code: `nanolm/models/gpt.py`. Configs: `02_char_transformer.yaml` (textbook), `02b_char_transformer_modern.yaml`
(laptop: `laptop/L2_char_gpt.yaml`). Every run in this section takes under 15 minutes on a laptop.

### 1.6.1 The textbook (GPT-2) block

```
x = tok_emb(idx) + pos_emb(positions)
for each layer:   x = x + Attention(LayerNorm(x))         # tokens exchange information
                  x = x + MLP(LayerNorm(x))               # per-token computation (GELU, 4x width)
logits = LayerNorm(x) @ tok_emb.weight^T                  # tied input/output embedding
```

The attention is causal: `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, so position t
only sees positions ≤ t. This one flag is what makes it a *language model*. Remove it and the model
reads the answer.

### 1.6.2 The modern ("Llama") block

| Textbook | Modern | Why |
|---|---|---|
| LayerNorm | **RMSNorm** | no mean subtraction or bias; cheaper, equally stable |
| learned absolute positions | **RoPE** (rotary) | relative positions enter q·k directly; no position table |
| GELU MLP | **SwiGLU** | gated MLP; the 2/3 width keeps the parameter count equal |
| biases | none | fewer parameters, no loss in quality |

Each row is one config key (`model.norm`, `model.pos`, `model.mlp`, `model.bias`).

### 1.6.3 Results: RNN vs. Transformer (Tiny Shakespeare, characters)

```bash
python -m nanolm.sweep   --sweep nanolm/configs/sweeps/01_char_models.yaml
python -m nanolm.analyze outputs/nanolm/sweeps/char_models --baseline gpt_textbook --metric val_bpb
```

Setup: 1.0M training / 0.11M validation characters. 2000 steps × 32 × 256 = 16.4M training
characters (about 16 passes). Final evaluation on the full validation split. Two seeds.

| model | params | val bpb (mean ± std, 2 seeds) | Δ vs. textbook GPT | paired p | train time |
|---|---:|---:|---:|---:|---:|
| **GPT modern** (RMSNorm+RoPE+SwiGLU, 4L×256) | 3.16M | **2.105 ± 0.002** | −0.084 | 0.03 | 37 s |
| LSTM 2×256 | 1.07M | 2.134 ± 0.008 | −0.056 | 0.005 | 51 s |
| GRU 2×256 | 0.81M | 2.167 ± 0.007 | −0.022 | 0.26 | 48 s |
| LSTM 2×512 (parameter-matched) | 4.24M | 2.171 ± 0.013 | −0.019 | 0.13 | 58 s |
| GPT textbook (LayerNorm+learned pos+GELU) | 3.44M | 2.189 ± 0.007 | 0 | — | 26 s |
| vanilla RNN 2×256 | 0.28M | 2.436 ± 0.009 | +0.246 | 0.003 | 17 s |

Seed noise floor (median within-experiment std): **0.0076 bpb**. With only 2 seeds a paired t-test
has 1 degree of freedom, so treat the p-values as a rough guide.

![validation bpb vs tokens](docs/nanolm_figs/char_models_curves.png)

What to notice:
- **The vanilla RNN is far behind** (+0.25 bpb). The gated cells (GRU, LSTM) fix most of that.
- **The modern recipe matters more than "Transformer vs. LSTM".** The textbook GPT is *worse* than a
  1M-parameter LSTM at this budget. The same-size model with RMSNorm + RoPE + SwiGLU is the best
  model. (2.105 bpb = 1.46 nats/char, in line with the well-known nanoGPT Shakespeare-char result.)
- **The bigger LSTM overfits.** LSTM-512 is best at about 10M tokens and then gets *worse*: its final
  train loss is 1.02 nats against 1.51 on validation (LSTM-256: 1.28 vs. 1.48). At 1 MB of text,
  data is the bottleneck, not parameters. This is a regularization lesson, not an architecture lesson.
- These conclusions hold **at this budget** (16 passes over 1 MB). Exercise 1 asks how they change
  with context length; more data (Part 2) changes them more.

---

## 1.7 Tokenizers: what a "token" should be

```bash
python -m nanolm.sweep --sweep nanolm/configs/sweeps/02_tokenizers.yaml
python -m nanolm.analyze outputs/nanolm/sweeps/tokenizers --metric val_bpb
```

Same modern Transformer, same 2000 × 32 × 256 token budget, three tokenizers. Because the token
budget is fixed, the BPE model sees about 3–4× more *text*, but it must also learn a
50,257-row embedding table from 1 MB of data.

| tokenizer | vocab | params (non-emb.) | bytes/token | train passes | val loss (nats/token) | val ppl | **val bpb** |
|---|---:|---:|---:|---:|---:|---:|---:|
| char | 65 | 3.16M (3.15M) | 1.00 | 16 | 1.460 | 4.31 | **2.107** |
| byte | 256 | 3.21M (3.15M) | 1.00 | 16 | 1.450 | 4.26 | **2.092** |
| GPT-2 BPE | 50,257 | 16.0M (3.15M) | 3.09 | 54 | 5.998 | 402 | **2.798** (best checkpoint, step 250: 2.163) |

(One seed each. The char/byte difference of 0.015 is about 2× the seed noise floor measured in §1.6.3,
so do not over-read it.)

![tokenizer curves](docs/nanolm_figs/tokenizers_curves.png)

Three lessons in one table:
1. **Loss and perplexity are not comparable across tokenizers.** By perplexity, BPE looks 100× worse
   (402 vs. 4.3). In bpb it is 33% worse at the end and nearly equal at its best point. Only bpb
   measures the same thing for all three.
2. **A large vocabulary needs data.** Shakespeare has only 302k GPT-2 tokens. The fixed token budget
   becomes 54 passes, and the 12.9M-parameter embedding table memorizes: train loss 1.30 nats/token
   against 6.0 on validation. The BPE model was best after 250 steps (2.163 bpb) and got worse from
   then on. Early stopping, or more data, would rescue it.
3. On a small corpus, bytes are as good as characters and need no vocabulary building. This is the
   byte-level-model argument in miniature.

Parameter Golf turned this trade-off into one of its biggest levers. The vocabulary grew from
sp1024 → sp4096 → sp8192 over the competition, about −0.012 bpb per step, combined with other
changes (§3.2). A larger vocabulary means more bytes per token, which means more text per
step and per unit of attention context. But the embedding table must fit in the 16 MB budget,
which is why quantizing the embeddings (int7/int6 GPTQ) arrived together with the large vocabularies.

---

### 1.7.1 More tokenizers: words, and training your own BPE (laptop sweep)

Part 1's tokenizer menu, all trained on the **training split only** (a tokenizer trained on
validation text leaks it). Each one is lossless, so bpb is exact (`tests/test_nanolm.py` checks this):

| spec | what it is |
|---|---|
| `char`, `byte` | one token per character / per UTF-8 byte |
| `word:<V>` | the V−256 most frequent words, whitespace runs and punctuation; **out-of-vocabulary words fall back to bytes** instead of `<unk>` |
| `sp_train:bpe:<V>`, `sp_train:unigram:<V>` | train your own SentencePiece BPE or unigram vocabulary (lossless settings, byte fallback) |
| `gpt2`, `hf:<name>` | a pretrained tokenizer (GPT-2's 50k BPE, Qwen's 151k, ...) |

```bash
python -m nanolm.sweep --sweep nanolm/configs/sweeps/laptop_tokenizers.yaml     # 8 runs, ~1-2 h on a Mac
python -m nanolm.analyze outputs/nanolm/sweeps/laptop_tokenizers --baseline char --metric val_bpb
```

Same 4-layer Transformer, same 2000 × 8,192-token budget, one seed. "Passes" is how many times the
1 MB corpus is seen; a larger vocabulary packs more text into each token:

| tokenizer | vocab | bytes/token | passes | final val bpb | **best val bpb** (step) |
|---|---:|---:|---:|---:|---:|
| byte | 256 | 1.00 | 16 | 2.092 | 2.088 (1750) |
| char | 65 | 1.00 | 16 | 2.105 | 2.105 (1750) |
| **word, 2048** | 2048 | 1.62 | ~27 | **2.080** | **2.047** (1250) |
| SentencePiece BPE, 512 | 512 | 1.76 | 29 | 2.163 | 2.123 (1250) |
| SentencePiece BPE, 2048 | 2048 | 2.41 | 42 | 2.368 | 2.104 (500) |
| SentencePiece unigram, 2048 | 2048 | 2.36 | 43 | 2.340 | 2.113 (500) |
| word, 8192 | 8192 | 1.98 | ~32 | 2.318 | 2.113 (750) |
| GPT-2 BPE | 50257 | 3.09 | 54 | 2.825 | 2.171 (500) |

Every tokenizer with a larger vocabulary reaches its best score *early* and then overfits. At a fixed
token budget a larger vocabulary means more passes over a tiny corpus, plus a larger embedding table
to memorize with. Compared at their best checkpoints, word-2048 (2.047) and BPE-2048 (2.104) beat
characters (2.105).

The conclusion depends on the regime, which is the general lesson:
- **Data-limited** (1 MB of Shakespeare): small vocabularies and early stopping win.
- **Data-rich** (Parameter Golf, billions of FineWeb tokens): the vocabulary *grew* from 1024 to 8192
  over the competition and gained about 0.012 bpb at each step.

## 1.8 Sampling: temperature, top-k, top-p

A language model gives a distribution over the next token. *Decoding* turns that distribution into
text (`evaluate.py: sample_next`, CLI `nanolm/generate.py`, replacing the old `inference_utils.py`):

- **temperature** τ divides the logits. τ → 0 is greedy (always the argmax): repetitive and safe.
  τ = 1 samples the model's own distribution. τ > 1 flattens it toward noise.
- **top-k** keeps the k most likely tokens.
- **top-p (nucleus)** keeps the smallest set of tokens whose total probability is ≥ p, so the set
  adapts to how confident the model is.

```bash
python -m nanolm.generate outputs/nanolm/<your L2 run> --prompt "ROMEO:" --compare-temperatures
python -m nanolm.generate outputs/nanolm/<your L2 run> --prompt "JULIET:" --top-k 0 --top-p 0.9
python -m nanolm.generate outputs/nanolm/<your L2 run> --interactive
```

Real output of the char Transformer from §1.6 (`--compare-temperatures`, prompt `ROMEO:`):

```
τ=0.0  That the sun shall be so soon to the state, / And therefore shall be so soon t...   (loops)
τ=0.5  The more than self-made a merry brother, / The true of the house of Lancaster is not live.
τ=1.0  Three fairest self it one means! would help me Menenius, / Or now here hath not seen like...
τ=1.5  Threehold her! Therefore, meank! woud I tles, grey / further forwardre hath not seemed...
```

Decoding settings change how the text *looks*, but not the model's bpb. bpb is a property of the
distribution, not of one sample from it.

---

# Part 2 — Transformers for real: pre-training and post-training

## 2.1 Pre-training on web text: FineWeb, the Parameter Golf data

```bash
python -m nanolm.prepare --golf-shards 1 --dest data/golf    # val (62M tokens) + 1 train shard (100M) + sp1024 tokenizer
python -m nanolm.train -c nanolm/configs/03_gpt_fineweb.yaml
python -m nanolm.sweep --sweep nanolm/configs/sweeps/03_transformer_recipe.yaml
```

The data is the challenge's FineWeb export: web documents tokenized with a 1024-token SentencePiece
BPE (`sp1024`). Because we report bpb on the same validation documents, our numbers sit on the same
axis as the public leaderboard. Ours use about 50M training tokens, while leaderboard runs use about
7B tokens (8 GPUs × 10 minutes).

### 2.1.1 Muon

AdamW scales every coordinate of the update separately. **Muon** (Keller Jordan, used by every
golf record) treats each hidden weight matrix as a whole. It takes the momentum-averaged gradient
G = U S Vᵀ and applies **U Vᵀ** instead: every singular direction gets the same step size, so
directions with small gradients are not starved. U Vᵀ is approximated in 5 matrix-multiply-only
Newton–Schulz iterations (`optim.py: zeropower_via_newtonschulz5`). Embeddings, the output head and
all vectors/scalars stay on Adam (`build_optimizers`). A test checks that the iteration maps the
singular values into roughly [0.7, 1.2].

### 2.1.2 Schedules

- **Cosine** (textbook): warm up, then cosine decay to `min_lr_frac`.
- **WSD / "warmdown"** (golf): constant LR, then linear decay over the last `warmdown_frac` of
  training. Progress is `max(step/max_steps, time/max_time_sec)`, so the same schedule works under
  a wall-clock budget, which is how golf runs were limited.

### 2.1.3 Results: the recipe, one change at a time

Setup: 6 layers × 384, 11M parameters, 3000 steps × 16k tokens = 49M training tokens (half of one
shard, streamed once). Final evaluation on the first 4.2M validation tokens. Two seeds. "p" is a
paired t-test with 1 degree of freedom, so it is a rough guide only.

| step in the recipe | val bpb (2 seeds) | change vs. the previous row | paired p |
|---|---:|---:|---:|
| GPT-2 textbook (LayerNorm, learned pos, GELU, bias), AdamW + cosine | 1.790 ± 0.001 | — | — |
| + RMSNorm + RoPE, no bias | 1.744 ± 0.010 | **−0.046** (both seeds −0.04 to −0.05) | 0.10 |
| + SwiGLU (same parameter count) | 1.786 ± 0.013 | **+0.042** (worse on both seeds) | 0.23 |
| ↳ SwiGLU at half the LR (5e-4) | 1.814 ± 0.001 | +0.029 vs. SwiGLU | 0.20 |
| + WSD schedule instead of cosine (still AdamW) | 1.728 ± 0.010 | **−0.058** | 0.02 |
| + **Muon** (matrices) + Adam (rest), WSD | **1.647 ± 0.010** | **−0.081** | 0.001 |

![recipe curves](docs/nanolm_figs/recipe_curves.png)

Reading the table as a researcher:
- **The two biggest wins are about optimization, not architecture.** Holding the LR high and then
  decaying it linearly to 0 (WSD) beats cosine by 0.058. Muon adds another 0.081 on top of the same
  schedule. Together they are worth more than all the block changes combined. This is also what
  Parameter Golf found.
- **The `plus_muon` row is not a pure optimizer effect on its own.** That run also switched cosine →
  WSD. Only the `plus_wsd_adamw` control row lets us split the gain into schedule and optimizer.
  When a run changes two things, add the control.
- **SwiGLU did *not* help here.** It learned faster for the first ~1,500 steps (training loss up to
  0.18 lower), then crossed over and finished worse on both seeds. Halving the LR made it *worse*,
  which suggests this short, cosine-decayed run is under-trained rather than unstable. We report the
  result as it is: the published benefit of SwiGLU did not reproduce at 49M tokens with this LR. With
  only 2 seeds, p = 0.23, this is a weak negative result, and finding out why is a good student project.
- Seed noise here is about 0.01 bpb. That is larger than on Shakespeare, because a 50M-token run is
  still far from converged.

---

### 2.1.4 The laptop version

`laptop/L4_gpt_fineweb.yaml` is the best recipe from §2.1.3: RMSNorm, RoPE, SwiGLU, Muon and WSD,
with 6 layers × 384 width, 11M parameters, and 49M tokens. The only change is the micro-batch:
16 sequences × 2 accumulation steps instead of 32 × 1. That is the same math in half the
activation memory.

```bash
python -m nanolm.prepare --golf-shards 1 --dest data/golf          # ~330 MB, once
python -m nanolm.bench   -c nanolm/configs/laptop/L4q_gpt_fineweb_quick.yaml
python -m nanolm.train   -c nanolm/configs/laptop/L4q_gpt_fineweb_quick.yaml   # ~5-15 min
python -m nanolm.train   -c nanolm/configs/laptop/L4_gpt_fineweb.yaml          # ~1-2 h
```

Reference (fp32, as on a Mac): quick version **1.879** bpb (0.9 GB). Full L4 **1.647** bpb (1.8 GB).
The full L4 matches the server-scale `plus_muon` result (1.647 ± 0.010). Same model, same tokens,
same answer, at any batch split and any precision.

## 2.2 Using pretrained models: how good is a "real" LLM in bits-per-byte?

Part 2 switches from models we train from scratch to models someone else pre-trained.
`nanolm/finetune.py` loads any HuggingFace causal LM (this replaces the old `train_lm.py`/
`experiments.py` HF branch). The first thing to do with a new model is to measure it in **our**
metric, on **our** data:

```bash
python -m nanolm.finetune -c nanolm/configs/part2/P1_eval_bpb.yaml                                  # Qwen2.5-0.5B
python -m nanolm.finetune -c nanolm/configs/part2/P1_eval_bpb.yaml model.name=gpt2 name=P1_gpt2
python -m nanolm.finetune -c nanolm/configs/part2/P1_eval_bpb.yaml model.name=HuggingFaceTB/SmolLM2-360M name=P1_smol
```

How it stays comparable: `val_documents()` decodes the Parameter Golf validation shards back to text.
Each model tokenizes that text with its *own* tokenizer, and bpb divides by the same UTF-8 byte count
(570,218 bytes for the first 200 documents). Context is 1,024 tokens, non-overlapping windows, as in
the golf baseline eval.

| model | year | parameters | weights (bf16) | val bpb (200 docs) |
|---|---|---:|---:|---:|
| GPT-2 small | 2019 | 124M | 0.25 GB | 1.079 |
| *Parameter Golf winner (full val set)* | *2026* | *~35M (our estimate)* | *16 MB (int6)* | *1.057* |
| Qwen2.5-0.5B | 2024 | 494M | 1.0 GB | 0.951 |
| SmolLM2-360M | 2024 | 362M | 0.7 GB | **0.928** |
| *our L4 laptop model (4M tokens of val)* | | *11M* | *44 MB* | *1.647* |

What to take from it:
- **The golf winner beats GPT-2 small in about 1/15 of the bytes (16 MB vs. 250 MB).** Five years of better
  architectures, optimizers, quantization and evaluation tricks are worth more than a 2019 model with ~3.5× the parameters and 15× the bytes.
  (Caveat: the golf number is on the full 50k-document set and ours on the first 200, so
  read this as a comparison to about ±0.02.)
- **Training data matters as much as size.** SmolLM2-360M beats the larger Qwen2.5-0.5B on this
  benchmark. A plausible reason is that SmolLM2 was pre-trained largely on FineWeb-Edu, the same
  distribution as these documents. Benchmark numbers always carry a training-distribution footnote.
- Our laptop model (1.647) is far from all of them. It saw 49M tokens; Qwen2.5 saw about 18 trillion.

## 2.3 Instruction tuning (SFT) with LoRA

A *base* model only continues text. **Supervised fine-tuning (SFT)** on (instruction, answer) pairs
teaches it to *answer*: to follow the instruction, then stop.

```bash
python -m nanolm.finetune -c nanolm/configs/part2/P2_sft_alpaca.yaml
```

How `finetune.py` does it (this replaces the old `experiments.py: run_qwen_finetune_experiment`, which
unfroze the last two layers of a 3B model, and the TRL-based `rl_train_main.py`):

1. **Format.** Each Alpaca example becomes a chat. The model's own chat template wraps the user turn
   and the assistant turn (`chat_example`).
2. **Loss on the answer only.** Prompt positions get label −100. The model is trained to produce
   answers, not to reproduce questions.
3. **LoRA.** Every attention and MLP projection W gets a trainable low-rank update W + (α/r)·B·A,
   with A ∈ ℝ^{r×d} and B ∈ ℝ^{d×r}. B starts at 0, so training starts from the unchanged model.
   With r = 16 that is **8.7M trainable parameters out of 370M (2.3%)**. Optimizer state and
   gradients shrink by the same factor, which is what makes this fit on a laptop.
4. **Memory.** Gradient checkpointing, plus computing the vocabulary softmax (49k for SmolLM2, 151k for
   Qwen) only at answer positions, in chunks (`token_logprobs`). Peak memory: **1.97 GB in fp32**.

Reference run (SmolLM2-360M base, LoRA r=16, 2,000 Alpaca examples, 250 steps × 16 examples, fp32):

| | before SFT | after SFT |
|---|---:|---:|
| held-out answer NLL (nats/token, 100 examples) | 1.556 | **1.321** |
| held-out answer perplexity | 4.74 | 3.75 |

The samples (`samples_before.txt` / `samples_after.txt`, greedy decoding) say more than the numbers:

```
### Give three tips for staying healthy.
BEFORE: • Eat a balanced diet. • Exercise regularly. • Get enough sleep. ...
        ### Question 2:  Give three tips for staying healthy.  ### Response: • Eat a balanced diet. ...
        ### Question 3: ...                                  <- never stops: it continues a "document"
AFTER:  1. Eat a balanced diet: A healthy diet is essential for maintaining good health. It should
        include a variety of fruits, vegetables, whole grains, ... 2. Stay active: ... 3. Get enough sleep: ...

### Write a haiku about autumn.
AFTER:  Autumn's golden hues / Shimmer in the breeze, / Nature's silent symphony.

### What is the capital of France, and why is it famous?
AFTER:  The capital of France is Paris, which is located in the region of Normandy ...      <- wrong
### Translate 'good morning' into Spanish and Chinese.
AFTER:  Good morning in Spanish: ¡Hola!   Good morning in Chinese: 你好！                  <- wrong
```

**SFT teaches format and behaviour, not knowledge.** After two passes over 2,000 examples the model answers in the
right shape and stops. But it states falsehoods (Paris is not in Normandy) and mistranslates
("¡Hola!" and "你好" mean "hello", not "good morning") with the same fluent confidence. That is
hallucination in its simplest form. A 360M model does not know enough, and SFT cannot add knowledge
that pre-training did not put there. The haiku also does not count syllables: a sub-word model does
not "see" syllables.

## 2.4 Preference tuning (DPO)

SFT shows the model *what* an answer looks like. **Preference tuning** shows it which of two answers
people prefer. **DPO** (Direct Preference Optimization, Rafailov et al. 2023) does this without
training a separate reward model and without reinforcement learning. For a prompt x, a preferred
answer y_w and a rejected answer y_l:

```
loss = -log σ( β · [ (log π(y_w|x) − log π_ref(y_w|x)) − (log π(y_l|x) − log π_ref(y_l|x)) ] )
```

The model π is pushed to raise the probability of y_w relative to y_l. The ratio to the frozen
reference π_ref (the model before tuning) acts as an anchor: β controls how far the model may drift.
The bracket, times β, is the **implicit reward margin**. In `finetune.py` the reference is the same
network with the LoRA adapter switched off (`model.disable_adapter()`), so DPO needs **no second copy
of the model**.

```bash
python -m nanolm.finetune -c nanolm/configs/part2/P3_dpo.yaml
```

Reference run:
- Model: Qwen2.5-0.5B-Instruct with LoRA r=16 (8.8M trainable parameters).
- Data: 800 UltraFeedback pairs whose quality scores differ by at least 2 points.
- Training: 100 steps × 8 pairs, β = 0.1.
- Evaluation: 100 held-out pairs from UltraFeedback's test split.
- Peak memory: **2.85 GB in fp32**.

| | before DPO | after DPO |
|---|---:|---:|
| implicit-reward accuracy (margin > 0) | 0.50 (policy = reference: all ties) | **0.68** |
| mean implicit-reward margin | 0.00 | 0.63 |
| policy alone prefers y_w (per-token log-prob) | 0.61 | 0.63 |
| training-batch reward accuracy (last steps) | — | ~0.71–0.80 |

How to read it honestly:
- **0.68 on 100 pairs is real.** The standard error is about 0.047, so this is about 3.8 standard
  errors above chance. The gap to the ~0.75–0.8 seen on training batches is the usual
  small-data overfitting.
- **The model's *own* preference barely moved** (0.61 → 0.63, within noise). DPO raised the
  probability of preferred answers *relative to the reference*, a small, distributed change. It did
  not turn the model into a better judge of answer quality.
- **The samples changed in style more than in substance.** After DPO the model drops the opening
  "Certainly!" and the markdown bold, and reorders some content. A factual error in the Chinese
  answer survived unchanged: both versions give the pinyin of 早上好 as "Wǎng bù duì" (correct:
  zǎo shang hǎo). Preference tuning shapes *how* a model answers; it does not fix what it does not know.
- With one seed and 100 validation pairs, the tutorial claims only the reward-accuracy effect.
  Exercise 11 asks you to vary β.

DPO is one of several preference-tuning methods. PPO and GRPO generate answers during training,
which is far slower and impractical on a laptop. The TRL-based versions remain in
`legacy/rl_train_main.py`.

---

# Part 3 — Parameter Golf

## 3.1 The challenge

[OpenAI Parameter Golf](https://github.com/openai/parameter-golf) (March 18 – April 30, 2026, more than
1,000 participants): train the language model with the lowest FineWeb validation bpb, subject to:

- the **artifact** (compressed weights + training code) is at most **16 MB**;
- training takes at most **10 minutes on 8×H100**; evaluation has its own time limit;
- evaluation is **tokenizer-agnostic** (bpb) on a fixed set of 50k documents.

This is "L(N)" optimization: the best loss for a fixed parameter/byte budget. It complements the
NanoGPT *speedrun* (best time for a fixed loss). With a budget this tight, the winning ideas are
about **using every byte** (quantization, compression, weight sharing) and **using every FLOP of
evaluation** (context, test-time training), as much as about architecture.

## 3.2 Final leaderboard (10 min / 16 MB track)

From the [official README](https://github.com/openai/parameter-golf/blob/main/README.md) (final
audited update, May 2026). Scores are 3-seed means.

| bpb | Author | What the record adds (on top of the previous one) |
|---:|---|---|
| **1.0565** | codemath3000 | PR #2130 architecture rerun on clean CaseOps data; token-only n-gram tilt, AsymLogit, more GPTQ calibration batches |
| 1.0576 | simonbissonnette | progressive context growth to 3k; short-document score-first TTT |
| 1.0586 | andrewbaggio1 | 2560-token eval/TTT context; no-Q/V TTT mask; QK-gain 5.25 |
| 1.0594 | alertcat | AWQ-lite mixed GPTQ + AsymLogit |
| 1.0611 | codemath3000 | BOS-fixed SmearGate, LQER, SparseAttnGate, per-group lrzip |
| 1.0634 | nprime06 | Polar-Express Newton–Schulz, MIN_LR, SparseAttnGate, fused CE |
| 1.0655 | dexhunter | CaseOps + gated attention + quant gate + loop 4–5 + phased TTT |
| 1.0678 | romeerp | CaseOps lossless case-transform tokenizer, tapered WD |
| 1.0810 | bigbag | SP8192 + 3-layer recurrence + parallel residuals + QK-gain 5.25 + legal TTT |
| 1.0856 | Kevin Clark | SP8192 + GPTQ embeddings + depth recurrence + SDClip |
| 1.0979 | Kevin Clark | SP4096 + 4× MLP + high WD, *removing* hash embeddings / SmearGate / value residual |
| 1.1194 | abaybektursun | LeakyReLU² + legal score-first TTT + parallel Muon (late March) |
| 1.1925 | Matthew Li | sliding-window eval only (stride 64) |
| 1.2244 | baseline | 9 layers × 512, 1024 vocab, tied embeddings, 4 KV heads |

How to read this table:
1. It is a **lineage**: almost every record is "previous stack + 1–3 changes".
2. The top 5 are within 0.005 bpb of each other, and several were accepted at p = 0.01–0.06. Their
   order is close to the seed noise.
3. The audit **excluded** many submissions that leaked the target or counted bytes wrongly (n-gram
   channels that depend on the target token, unnormalized byte mixtures, wrong byte denominator).
   Metric hygiene was a real part of the competition (§4.3).

## 3.3 The techniques, grouped by lever

The impact numbers are quoted from the record write-ups. Most are single-seed and/or bundled with
other changes, so treat them as rough.

**A. Evaluation (free at training time)**

| technique | what it does | reported gain |
|---|---|---|
| sliding-window eval | score each token with nearly the full context instead of on average half (`eval.sliding_stride`) | **−0.032** (03-19) |
| legal score-first TTT | adapt (LoRA) weights on validation chunks *after* scoring them | −0.002 early … −0.013 late (phased) |
| longer eval context | 2k–3k tokens at eval | ~−0.001 |

**B. Bytes: quantization × compression × weight decay**

| technique | what it does | reported gain |
|---|---|---|
| int8 → int6 per-row + zstd/lzma | ~25% fewer bytes per weight, spent on a bigger model (3× MLP) | −0.029 together with MLP3×/SmearGate/BigramHash (03-19/20) |
| QAT (fake-quant STE) | train through the rounding (`train.qat_bits`) | small; removed later |
| GPTQ (Hessian-aware rounding), SDClip, AWQ-lite, LQER | better rounding / clipping / low-rank error correction | −0.0046 (full-Hessian GPTQ) … −0.0003 |
| high weight decay (0.09) | smaller weights compress better (weight RMS predicts compressed size, R² ≈ 0.99), so room for more parameters | part of −0.012 (04-01) |

**C. Tokenizer**: sp1024 → sp4096 → sp8192 (about −0.012 per step, bundled), CaseOps (−0.004).

**D. Architecture** (each is a key in `ModelConfig`):

| technique | key | what it does | reported gain |
|---|---|---|---|
| U-Net skips | `unet_skips` | encoder-half outputs feed mirrored decoder layers | baseline |
| relu² / LeakyReLU(0.5)² | `mlp` | squared activation; leaky keeps gradient for x<0 | −0.002 … −0.003 |
| XSA (exclusive self-attn) | `xsa_last_n` | subtract each head's projection onto the token's *own* value vector | −0.005 with EMA |
| partial RoPE + LN scale | `rope_dims`, `ln_scale` | rotate 16/64 dims; damp layer i's input by 1/√(i+1) | −0.0023 |
| QK-gain | `qk_gain_init` | learnable per-head temperature after QK-norm, initialized 4–5.25 | ~−0.003 (4→5) |
| SmearGate | `smear_gate` | blend each token's embedding with the previous token's | part of −0.029 |
| BigramHash | `bigram_buckets` | hashed (prev, cur) pair embedding | part of −0.029, later removed |
| value residual | `value_residual` | mix layer-0 values into later layers | later removed |
| depth recurrence | `recur_layers` | re-run layers 3–5 (or 4–5): more depth, zero extra bytes | −0.002 … −0.003 |
| parallel residuals | `parallel_from` | attention and MLP read the same input in late layers | −0.002 |
| gated attention | `attn_gate` | sigmoid gate per head on the attention output | ~−0.002 |

**E. Optimizer**: Muon everywhere; momentum 0.95 → 0.97–0.99; Muon weight decay; EMA of weights
(−0.0006 alone); long warmdown + LR floor; MuonEq-R and Polar-Express coefficients (~−0.001 each).

**What this tells a student.** The big, robust wins were *not* clever blocks. They were:
(1) evaluate with more context, (2) spend bytes better (quantize, compress, then buy parameters with
the savings), (3) a larger vocabulary, and (4) test-time training. Most architectural tweaks were
worth about 0.002 bpb, a size that only shows up inside an already-tuned stack and that is
comparable to the seed noise. Some (BigramHash, value residual) were later *removed*: they helped a
weak model and became useless once the model was stronger. An independent analysis of about 2,000
submissions (["Parameter Golf: What Really Works?", arXiv 2607.01517](https://arxiv.org/abs/2607.01517))
reaches a similar conclusion, although its correlational method is weak evidence.

## 3.4 The golf baseline in `nanolm`

`configs/04_golf_baseline.yaml` re-implements the challenge baseline with config switches:
9 layers, d = 512, 8 heads / 4 KV heads (GQA), relu² MLP ×2, tied embeddings (init std 0.005),
U-Net skips, x0 residual mixing, QK-norm with gain 1.5, logit softcap 30, zero-initialized output
projections, Muon (lr 0.04) + Adam (embeddings 0.05, scalars 0.04), constant LR then linear warmdown,
int8 + compression export. With `torchrun --nproc_per_node=8` and `max_time_sec: 600` this matches the
official budget.

On one GPU, `04_golf_lab.yaml` keeps the same model and shrinks the budget to 2000 steps × 65,536
tokens = 131M tokens. That is about 2% of an official run, so absolute numbers are higher than the
leaderboard. The *differences* between techniques are what we measure.

## 3.5 Our ablation: each technique alone, two seeds

```bash
python -m nanolm.sweep   --sweep nanolm/configs/sweeps/04_golf_ablation.yaml
python -m nanolm.analyze outputs/nanolm/sweeps/golf_ablation --baseline baseline --metric val_bpb
```

Setup: the golf baseline model (17.07M parameters), 2000 steps × 65,536 tokens = 131M FineWeb
tokens, the same data order for every run, and evaluation on the first 4.2M validation tokens
(non-overlapping windows). Each technique is switched on **alone**. Two seeds; "p" is a paired
t-test (1 degree of freedom). Seed noise floor: **0.0017 bpb**.

| technique (alone) | val bpb (2 seeds) | Δ vs. baseline | paired p | params | int8 size | verdict |
|---|---:|---:|---:|---:|---:|---|
| MLP 3× (instead of 2×) | 1.4167 ± 0.0003 | **−0.0121** | 0.10 | 21.8M | 19.8 MB ✗ | helps (both seeds), but needs int6 to fit |
| value residual | 1.4218 ± 0.0013 | **−0.0069** | 0.08 | 17.1M | 15.7 MB | helps (both seeds) |
| LeakyReLU(0.5)² | 1.4228 ± 0.0015 | **−0.0059** | 0.07 | 17.1M | 15.8 MB | helps (both seeds) |
| depth recurrence (layers 3–5 ×2 from 35%) | 1.4238 ± 0.0019 | **−0.0049** | 0.05 | 17.1M | 15.7 MB | helps, zero extra bytes |
| XSA, last 4 layers | 1.4265 ± 0.0006 | −0.0023 | 0.33 | 17.1M | 15.7 MB | small, ~1 noise floor |
| QK-gain init 4.0 | 1.4277 ± 0.0023 | −0.0010 | 0.05 | 17.1M | 15.7 MB | within noise |
| SmearGate + BigramHash(4096) | 1.4287 ± 0.0009 | −0.0001 | 0.99 | 17.7M | 16.3 MB | nothing (and costs 0.6 MB) |
| **baseline** | **1.4287 ± 0.0025** | 0 | — | 17.1M | 15.7 MB | |
| parallel residual from layer 6 | 1.4291 ± 0.0017 | +0.0003 | 0.65 | 17.1M | 15.7 MB | nothing |
| gated attention | 1.4297 ± 0.0021 | +0.0010 | 0.16 | 17.1M | 15.8 MB | nothing |
| EMA (0.997) | 1.4299 ± 0.0021 | +0.0012 | 0.14 | 17.1M | 15.7 MB | nothing at 2000 steps |
| partial RoPE (16/64) + LN-scale | 1.4361 ± 0.0029 | **+0.0073** | 0.03 | 17.1M | 15.7 MB | **hurts** (both seeds) |
| Muon weight decay 0.04 | 1.4367 ± 0.0012 | **+0.0080** | 0.07 | 17.1M | 15.7 MB | **hurts**, saves only 0.3% bytes |
| "record stack" (10 tricks from records, see `stack`) | 1.4300 ± 0.0020 | +0.0013 | 0.75 | 22.4M | 19.6 MB | **nothing**: the harmful ones cancel the good ones |
| **verified stack** (only the 5 that helped) + int6 | **1.3935 ± 0.0012** | **−0.0352** | 0.016 | 21.8M | **14.6 MB (int6)** ✓ | sum of the parts (−0.032 expected) |

![golf ablation](docs/nanolm_figs/golf_ablation_bars.png)

What this experiment teaches:
1. **Ablate one thing at a time, then stack only what survived.** The "record stack" (everything a
   record write-up recommended) is *no better than the baseline*. Two of its components hurt at our
   scale, and EMA and SmearGate/BigramHash did nothing. Stacking only the five techniques that helped
   on both seeds gives −0.035, almost exactly the sum of their separate effects (−0.032). At this scale
   the good effects are close to additive.
2. **Context matters for "what works".** Partial RoPE + LN-scale and Muon WD were real gains in the
   records: in an 11-layer model trained on about 7B tokens, with a size budget that WD helped meet.
   In a 9-layer model trained on 131M tokens they cost 0.007–0.008. Muon WD exists to make weights
   compress better, but with int8 at our scale it saved only 0.3% of the bytes. A technique is a
   claim about a regime, not a law.
3. **Bytes are a currency.** 3× MLP is the largest single gain, but it breaks the 16 MB budget at int8
   (19.8 MB). It only "counts" because int6 brings it back to 14.6 MB for a +0.0016 bpb rounding cost
   (§3.6). In golf, an architecture result without its artifact size is half a result, and that is
   why `metrics.json` always records both.
4. **Two seeds is a minimum, not a luxury.** The noise floor (0.0017) is about the size of the
   XSA and QK-gain effects. We call those "within noise", not "helps".

## 3.6 The free levers: evaluation stride and quantization (no retraining)

```bash
python -m nanolm.compress outputs/nanolm/sweeps/golf_ablation/baseline_s1337 --bits 8 6 5 4 --strides 0 256 64
```

Golf-lab baseline (seed 1337, 17.07M parameters), evaluated on the first 4.2M validation tokens
with a 1024-token context:

**Evaluation stride (same weights):**

| eval | val bpb | eval time |
|---|---:|---:|
| non-overlapping 1024-token windows (average context ≈ 512) | 1.4270 | 5.5 s |
| sliding, stride 256 (context ≥ 768) | 1.3915 (−0.0355) | 20 s |
| sliding, stride 64 (context ≥ 960) | **1.3904 (−0.0366)** | 81 s |

This is the single largest effect in the whole tutorial, and it comes with *zero* training change.
It reproduces the leaderboard record (−0.032). Most of the gain is already there at stride 256 for
a quarter of the cost. Nothing about the model changed: non-overlapping evaluation simply
under-reports what the model can do, because a token at the start of a window has almost no context.

**Quantization × compressor (round-trip = de-quantize and re-evaluate):**

| export | baseline: size | bits/param | round-trip bpb gap | stack (22.4M params): size | gap |
|---|---:|---:|---:|---:|---:|
| int8 + zlib | 15.68 MB | 7.35 | +0.0001 | 19.87 MB | +0.0001 |
| int8 + lzma | 15.72 MB | 7.37 | +0.0001 | 19.57 MB | +0.0001 |
| int6 + lzma | 11.52 MB | 5.40 | +0.0019 | **14.24 MB** | +0.0012 |
| int5 + lzma | 9.81 MB | 4.60 | +0.0075 | 11.90 MB | +0.0052 |
| int4 + lzma | 7.78 MB | 3.64 | +0.0302 | 9.77 MB | +0.0225 |

- int8 is essentially free (+0.0001) but cannot fit a model of more than about 17M parameters into 16 MB.
- **int6 costs about 0.002 bpb and saves 27% of the bytes.** That is how the 22M-parameter (MLP 3×)
  model fits into 14.2 MB, and 3× MLP alone was worth −0.012 (§3.5). Every golf record made this trade.
- int4 with simple rounding is too lossy (+0.03). The records that went below 6 bits used GPTQ
  (Hessian-aware rounding) or ternary/binary *training*, not post-hoc rounding.
- lzma squeezes the unused high bits of int6/int5 values stored in int8 bytes (5.40 bits/param at
  "6 bits"). At int8 there is nothing to squeeze, and lzma is even slightly larger than zlib.

## 3.7 Putting it together (lab scale, 131M training tokens, 2 seeds)

| | val bpb | artifact |
|---|---:|---:|
| golf baseline, int8 | 1.4287 | 15.7 MB |
| + verified architecture stack (3× MLP, value residual, LeakyReLU², depth recurrence, XSA) | 1.3935 | 14.6 MB at int6 |
| + sliding-window eval, stride 64 | **1.3558** | |
| int6 round-trip cost (measured at stride 0) | +0.0016 | |

That is about −0.073 bpb (−5%) from a 2% compute budget. On the leaderboard, baseline → winner was
−0.168 with about 50× more training tokens, a larger vocabulary, GPTQ and test-time training: the
techniques we did not implement, which make good exercises.

---

## 3.8 Parameter Golf on a laptop

Everything in this Part runs on a laptop at a smaller scale. `laptop/L5_golf_mini.yaml` is the golf
baseline architecture at 6 layers × 384 (U-Net skips, x0 mixing, QK-norm with gain, relu² MLP, logit
softcap, tied embeddings, Muon + Adam, warmdown), trained on 41M FineWeb tokens. It has a
laptop-sized "golf budget" of **6.5 MB**.

```bash
python -m nanolm.train -c nanolm/configs/laptop/L5_golf_mini.yaml          # baseline
python -m nanolm.train -c nanolm/configs/laptop/L6_golf_mini_stack.yaml    # verified stack + int6 + sliding eval
python -m nanolm.sweep --sweep nanolm/configs/sweeps/laptop_golf_ablation.yaml   # overnight: one technique at a time
```

Reference runs (fp32, as on a Mac):

| | params | val bpb | + sliding eval (stride 128) | artifact | fits 6.5 MB? | peak memory |
|---|---:|---:|---:|---:|:---:|---:|
| L5 golf-mini baseline (int8) | 6.6M | 1.633 | — | 6.11 MB | ✓ | 1.8 GB |
| L6 verified stack (int6) | 8.4M | 1.601 | **1.554** (int6 round-trip 1.555) | 5.59 MB | ✓ | 2.9 GB |

The laptop ablation (`sweeps/laptop_golf_ablation.yaml`) was run here with **two seeds**
(`--seeds 1337 42 --set precision=fp32`). On a laptop you would run one seed overnight. Noise
floor: 0.0018 bpb.

| technique (alone, on L5) | val bpb | Δ vs. baseline | paired p | same technique at lab scale (§3.5) |
|---|---:|---:|---:|---:|
| depth recurrence (layers 2–3) | 1.6284 ± 0.0018 | **−0.0053** | 0.001 | −0.0049 |
| LeakyReLU(0.5)² | 1.6285 ± 0.0022 | **−0.0053** | 0.04 | −0.0059 |
| value residual | 1.6305 ± 0.0024 | −0.0033 | 0.08 | −0.0069 |
| **baseline** | 1.6338 ± 0.0018 | 0 | — | 0 |
| MLP 3× | 1.6341 ± 0.0018 | **+0.0003** | 0.92 | **−0.0121** |
| partial RoPE + LN-scale | 1.6478 ± 0.0011 | **+0.0140** | 0.09 | +0.0073 |
| **verified stack** (all of the helpful ones + XSA, int6) | **1.6011 ± 0.0017** | **−0.0327** | 0.0006 | −0.0352 |

![laptop golf ablation](docs/nanolm_figs/laptop_golf_ablation_bars.png)

What carries over from the server-scale results, and what does not:
- **Carries over:** recurrence and LeakyReLU² help (both scales, about −0.005), and partial RoPE +
  LN-scale hurts (both scales). The verified stack gives about −0.033 at both scales.
- **Does not carry over: 3× MLP**, the largest single gain at lab scale, does nothing here. We have
  not tested *why*. One hypothesis: a 384-wide model with a 6-layer budget is limited by something
  other than MLP width at 41M tokens. Note that the tokens-per-parameter ratio (~5) is similar to the
  lab run (~6), so "too little data" alone does not explain it. Treat this as an open question.
- **The stack is more than the sum of its parts here.** The individually measured parts add up to about
  −0.014 (XSA was not tested alone), but together they give −0.033. The parts interact: for example,
  3× MLP may only pay off together with value residual and recurrence. The way to find out is a
  *leave-one-out* ablation, which is Exercise 13. Whether effects add up is itself something to
  measure, not assume.
- The whole laptop golf track (L5, L6 and the one-seed ablation) is about one night of laptop compute.
  It reproduces the main lessons of a multi-GPU study.

---

# Part 4 — Methodology: seeds, noise, and a post-mortem

## 4.1 One seed is an anecdote

Two runs that differ only in the random seed differ in val bpb. `analyze.py` prints the median
within-experiment standard deviation (the **noise floor**). On the leaderboard, the seed std was
about 0.0003–0.001 bpb, while many claimed improvements were 0.001–0.003. Rules of thumb:

- Use **the same seeds** for the baseline and the variant, and compare *paired* differences
  (`paired_p` in `summary.md`). The data order is fixed by the data, not the seed, so pairing removes
  most of the shared variance.
- A difference smaller than about 2× the noise floor needs more seeds, not more enthusiasm.
- Decide the metric *before* looking at the results. Report `val_bpb`, `sliding_bpb` and
  `quant_val_bpb` separately; they answer different questions.

## 4.2 Debug or negative result?

When a variant does not help, check these **before** concluding that the idea is dead:
1. Did the switch actually change the model? Compare `params` in `metrics.json`, and look for the
   "depth recurrence ON" / "QAT ON" lines in the log.
2. Is the run comparable? Same tokens seen, same data order, same eval tokens.
3. Is the effect larger than the noise floor at this scale? Many golf tricks only matter for a
   well-trained model. A 131M-token run is not a 7B-token run.

If all three check out, it is a genuine negative result at this scale. Write it down; it is a result.

## 4.3 Post-mortem of our own 2026 submission (why this codebase looks the way it does)

We entered Parameter Golf with a 3,400-line script (`mytrain_gpt_v6_1.py`) and a JSON →
environment-variable launcher. A careful audit after the challenge found:

| What happened | Effect | Fixed here by |
|---|---|---|
| A "PPM byte-mixture" eval built the network's byte probability *using the true byte* | reported 0.83–0.94 bpb; **invalid** (the real numbers were around 1.16–1.28), and the launcher ranked runs by this number | metric code separated from model code; causality tests; bpb defined in one place |
| The launcher's size regex did not match the new script's log line | artifact size was `None` for every run; our best model (1.1586 bpb) was **32 MB**, twice the limit, and nobody noticed | size computed and written to `metrics.json` by the trainer itself (`fits_budget`) |
| Unknown config keys only produced a warning; several flags were never read | ablations of features that were not running | typed configs that reject unknown keys |
| Late QAT was scheduled on the step limit while the run stopped on wall-clock | QAT never switched on in time-limited runs | progress-based switches (`progress = max(step, time)`) with a log line |
| Sliding-window eval was used for the in-training curve but not for the final number | curves and final numbers disagreed (1.238 vs 1.273) | one `evaluate()` with an explicit, logged stride |
| Decisions made from single seeds at the 0.002 level | noise was chased | sweeps with seeds + paired tests |

Our best *valid* submission scored about 1.277–1.281 bpb (3 seeds, 15.97 MB). The baseline was
1.2244 and the winner 1.0565. The lesson is not that the ideas were bad. Without a trustworthy
measurement pipeline we could not tell which ideas were good.

---

---

# Exercises

1. **Context length.** Train `02b` with `train.seq_len=64, 128, 256, 512`. Plot bpb against context
   for the Transformer and the LSTM. Where does the LSTM stop benefiting?
2. **Scaling law in one afternoon.** Sweep `model.dim` × `model.n_layers` on `03_gpt_fineweb.yaml`
   at a fixed token budget. Plot bpb against non-embedding parameters (`params_non_embedding`).
3. **AdamW vs. Muon at equal tuning effort.** Give each optimizer the same 4-point LR grid
   (use `grid:` in a sweep file). Is the gap larger or smaller than with default LRs?
4. **The byte budget.** With `compress.py`, find the smallest bits/weight that costs less than
   0.01 bpb. Then make the model wider until int6 + lzma reaches exactly 16 MB. Does it beat the int8
   model of the same size?
5. **Reproduce a record claim.** Pick one row of §3.3 and test it with 3 seeds in `04_golf_lab`.
   Report the mean, std and paired p-value. Does the claimed gain survive at our scale?
6. **Break it on purpose.** Remove `is_causal=True` from `Attention.forward` and run the tests.
   Then train for 100 steps and look at the bpb. Explain why the number is meaningless.
7. **Implement a new technique** as one config key in `models/gpt.py` (e.g. multi-token prediction,
   or SwiGLU vs. relu² at equal parameters), add a causality test case, and run an ablation sweep.

---
**Part 2 and laptop exercises**

8. **Reproduce a reference.** Run `laptop/L2_char_gpt` and `laptop/L4q_gpt_fineweb_quick` on your
   machine. Report your `bench` TFLOP/s, your wall-clock time, and your val bpb against the reference
   (§0.2). If you are off by more than 0.01, find out why.
9. **LoRA rank.** Run `part2/P2_sft_alpaca` with `model.lora_r=4`, `16` and `64`. Plot the after-SFT
   response NLL against trainable parameters. Where does a larger rank stop helping?
10. **LoRA vs. full fine-tuning.** Run P2 with `model.lora=false train.lr=2e-5`, using the 135M model
    on a laptop. Compare memory (`peak_mem_mb`), time and quality. Is full fine-tuning worth it here?
11. **How hard should DPO pull?** Sweep `train.dpo_beta` over 0.05, 0.1 and 0.5 in P3. Higher β keeps the
    policy closer to the reference. Plot reward accuracy against how much the samples change.
12. **The alignment tax.** Does instruction tuning make a model worse at plain web text? Extend
    `finetune.py` so that `eval_bpb` can load a LoRA adapter (`peft.PeftModel.from_pretrained`). Then
    measure FineWeb bpb before and after SFT.
13. **Leave-one-out.** In the laptop golf ablation the stack (−0.033) beat the sum of its parts
    (about −0.014). Add five runs to `sweeps/laptop_golf_ablation.yaml`, each equal to `stack_verified`
    minus one technique. Which technique matters most *inside* the stack? Is 3× MLP still useless
    there?

---

## Appendix A. Laptop and Mac reference

Start with [`LAPTOP_QUICKSTART.md`](LAPTOP_QUICKSTART.md). The command reference:

```bash
python -m nanolm.bench    -c <config> [overrides]            # tokens/s, memory, projected time on THIS machine
python -m nanolm.train    -c <config> [overrides]            # train from scratch (Parts 1, 2.1, 3)
python -m nanolm.finetune -c <part2 config> [overrides]      # pretrained models: eval_bpb / sft / dpo
python -m nanolm.generate <run_dir> --prompt "..." [--top-p 0.9] [--interactive]
python -m nanolm.sweep    --sweep <sweep.yaml> [--set device=cpu] [--only name ...]
python -m nanolm.analyze  <sweep dir or outputs dir>
python -m nanolm.compress <run_dir> --bits 8 6 4 --strides 0 64
```

Memory knobs, from most to least effective:

| knob | effect |
|---|---|
| `train.batch_size=8 train.grad_accum=4` (keep the product) | halves activation memory; identical math |
| `train.seq_len=256` | halves activation memory, but changes the experiment (shorter context) |
| `eval.batch_size=4` | lower memory at evaluation time |
| Part 2: `data.max_len=256`, `model.name=HuggingFaceTB/SmolLM2-135M`, `model.grad_checkpointing=true` | memory for pretrained models |
| `precision=bf16` (Mac, macOS 14+) | about half the activation memory, often faster; less tested than fp32 |

Device notes:
- **Apple MPS.** fp32 by default. `torch.compile` is skipped. If an operation is not implemented on
  MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **NVIDIA laptop GPUs (4–8 GB).** bf16 automatically (Ampere / RTX 30xx and newer). Turn on
  `train.compile=true` only on Linux.
- **CPU.** Works for every config, about 3–10× slower than a laptop GPU.

The MPS path has not yet been benchmarked by us on a real Mac. The time estimates in §0.2 come from
FLOP counts, and reports from students' machines are welcome.

## Appendix B. Multi-GPU

```bash
torchrun --standalone --nproc_per_node=8 -m nanolm.train -c nanolm/configs/04_golf_baseline.yaml
```
Each rank reads its own slice of the sequential stream. Muon orthogonalizes a slice of the matrices
on each rank and all-reduces the updates. Under `max_time_sec`, the ranks agree on when to stop.

> **Not yet tested.** Every result in this tutorial was produced on a single GPU. The DDP path follows
> the official baseline's design, but we have not run it on multiple GPUs. Run
> `torchrun --nproc_per_node=2 -m nanolm.train -c nanolm/configs/00_smoke.yaml` and check that it
> matches a single-GPU run before relying on it.

## Appendix C. Adding a new experiment

1. Add a key to `ModelConfig` (or `TrainConfig`) in `config.py`, with a default that means *off*.
2. Implement it in `models/gpt.py` behind that key.
3. Add it to `GOLF_FEATURES` in `tests/test_nanolm.py` and run the tests.
4. Add a run to a sweep file. Do not edit the base config.

## Appendix D. Where did the old code go?

The previous course files (`transformer.py`, `train_lm.py`, `lm_dataset.py`, `tokenizer_utils.py`,
`experiments.py`, `rl_*.py`, ...) now live in [`legacy/`](legacy/README.md). Its README maps every old
class and function to its `nanolm` replacement.
