# Serving a 27B LLM on a Mac: memory, speed and inference engines

This tutorial serves **Qwen3.8-27B**, a 27-billion-parameter vision-language model, on one Apple-Silicon laptop.
It compares four engines: **MLX-LM**, **mlx-vlm**, **oMLX** and **lithos-metal**. It works out where every gigabyte
of memory goes and why decoding and prompt reading (prefill) are limited by different hardware resources. It also
explains the optimization each engine uses: quantization, kernel fusion, speculative decoding, prefix caching,
GPU-resident decoding. Every number below was measured on the machine described next, and every measurement can
be reproduced with the scripts in [`mac_inference/`](mac_inference/).

It is the inference companion to [`TUTORIAL_nanolm.md`](TUTORIAL_nanolm.md), where we *train* models up to about 11M
parameters. Part 3 of that tutorial uses pretrained models of 0.1–0.5B. Here the model is 2,500× larger than our largest
trained one, and the question changes from "how do we learn the weights?" to "how do we move them through the
chip fast enough?".

| | |
|---|---|
| machine | MacBook Pro, **Apple M5 Pro** (20-core GPU), **64 GB** unified memory |
| OS | macOS 26.5.2 |
| engines | mlx-lm 0.32.2, mlx-vlm 0.7.8, oMLX 0.7.1rc1, lithos-metal 0.1.2 (`5485ad6`); MLX 0.32.4 |
| model | Qwen3.8-27B: affine 4-bit (`mlx-community/Qwen3.8-27B-4bit`), oQ4e (`Jundot/Qwen3.8-27B-oQ4e-mtp`), NVFP4 (`nvidia/Qwen3.8-27B-NVFP4`) |
| date | 2026-10-18/19 |

**Results in one table** (details in §6):

| | decode tok/s (code / prose / math) | TTFT, 16K-token prompt | agent follow-up turn | vision |
|---|---:|---:|---:|:---:|
| MLX-LM | 17 / 17 / 17 | 37 s | 0.5 s | — |
| mlx-vlm + MTP (`APC_ENABLED=1`) | 40 / 32 / 37 | 34–36 s | 0.7 s | ✓ |
| **oMLX**, oQ4e + MTP | 42 / **34** / 45 | 47 s | 0.6–0.8 s | ✓ |
| lithos-metal + DSpark | **52** / 30 / **52** | 74 s | 0.9 s | — |

---

## Contents

1. [How LLM inference uses the chip](#1-how-llm-inference-uses-the-chip)
2. [The model: Qwen3.8-27B](#2-the-model-qwen38-27b)
3. [Memory: where every gigabyte goes](#3-memory-where-every-gigabyte-goes)
4. [Optimization techniques](#4-optimization-techniques)
5. [The four engines and what each optimizes](#5-the-four-engines-and-what-each-optimizes)
6. [Measurements](#6-measurements)
7. [What the measurements teach](#7-what-the-measurements-teach)
8. [Choosing an engine](#8-choosing-an-engine)
9. [Reproduce it](#9-reproduce-it)
10. [Exercises](#10-exercises)

---

## 1. How LLM inference uses the chip

Serving a request has two phases with very different costs.

**Prefill** reads the prompt. All prompt tokens are known, so the model processes them in parallel, in chunks of
128–2,048 tokens. Each weight matrix is read from memory once per chunk and multiplied with many token vectors at
once. The work per byte of weights is high, so prefill is limited by **arithmetic throughput** (FLOP/s).

**Decode** generates the answer, one token at a time, because token *t+1* depends on token *t*. To produce one
token, *every* weight is read from memory once and multiplied with a single vector. The work per byte is tiny, so
decode is limited by **memory bandwidth** (GB/s).

A forward pass of a dense model with N parameters costs about 2N FLOPs per token. Each weight takes part in one
multiply and one add. For one decode step and one prefill chunk this gives:

```math
t_{\text{decode}} \approx \frac{\text{bytes of weights}}{\text{memory bandwidth}}, \qquad
t_{\text{prefill}}(n) \approx \frac{2 N \cdot n}{\text{achieved FLOP/s}} \quad (n \text{ prompt tokens}).
```

For Qwen3.8-27B in 4-bit on the M5 Pro:

| quantity | value | source |
|---|---:|---|
| weights read per decode step | 15.1 GB | measured (§3) |
| peak read bandwidth | about 290 GB/s | lithos-metal's probe of this chip (`m5_pro/config.json`); Apple's nominal figure is 307 GB/s |
| **decode upper bound** | 290 / 15.1 ≈ **19 tok/s** | |
| measured MLX-LM decode | 17.4 tok/s, so 17.4 × 15.1 ≈ 263 GB/s, **91 % of peak** | §6.1 |
| FLOPs per prompt token | 2 × 27B ≈ 54 GFLOP | |
| measured MLX prefill | 475 tok/s at 4K, so about **26 TFLOP/s** | §6.2 |

Two consequences shape everything that follows:

- **One token per weight read cannot get much faster than about 19 tok/s on this Mac**, whatever the engine.
  The only way past it is to produce *several* tokens per read of the weights. That is what speculative decoding does (§4.4).
- **Prefill is where the Mac is far behind an NVIDIA GPU.** Decode is limited by bandwidth: an H100 has
  about 3.35 TB/s, roughly 11× this Mac. Prefill is limited by arithmetic: dense BF16 on an H100 is about 990 TFLOP/s,
  roughly 38× the 26 TFLOP/s measured here. As a rough estimate, *not measured here*: at 40–50 % utilization an H100
  would read a 16K-token prompt of this model in about 2 s, against 37 s on the M5 Pro. For an agent that sends long
  prompts, the time to first token is therefore the Mac's weak point (§6.2).

**Unified memory.** The CPU and GPU of an Apple-Silicon chip share one pool of memory. There is no copy from "host"
to "device", and a model can use most of the machine's RAM. macOS still caps how much memory the GPU may use: the
Metal "recommended max working set". On this 64 GB machine it is **51.8 GB** (oMLX logs it at start-up); on a 24 GB
M5 Pro it is about **17.8 GB** (lithos-metal's chip profile). The cap can be raised with
`sudo sysctl iogpu.wired_limit_mb=<MB>`, which oMLX suggests in its log. This tutorial did not need that.

## 2. The model: Qwen3.8-27B

Qwen3.8-27B uses the Qwen3.5 architecture (`Qwen3_5ForConditionalGeneration`). Its configuration, from `config.json`:

| | value |
|---|---|
| layers | 64: **48 Gated DeltaNet** (linear attention) + **16 full attention**, in a repeating 3 : 1 pattern |
| hidden size / MLP size | 5,120 / 17,408 (SwiGLU) |
| full attention | 24 query heads, **4 KV heads** (grouped-query attention), head dim 256, RoPE on 25 % of each head, output gate |
| Gated DeltaNet | 16 key heads and 48 value heads of dim 128; a short causal convolution (kernel 4) |
| vocabulary | 248,320 tokens; input embedding and output head are **not** tied |
| context | 262,144 tokens natively |
| extras | a 27-layer vision encoder (hidden 1,152, 16-pixel patches); **one MTP layer** for multi-token prediction |

Two parts of this design matter most for serving:

- **Gated DeltaNet layers keep a fixed-size state, not a growing KV cache.** A full-attention layer must keep the
  key and value of every previous token. A DeltaNet layer folds the whole history into a matrix state S
  of shape 128 × 128 per head, and updates it with each new token. The update below is from the lithos-metal
  mixer design, simplified:

  ```math
  \bar S_t = \alpha_t S_{t-1}, \qquad
  \delta_t = \beta_t \,(v_t - \bar S_t^{\top} k_t), \qquad
  S_t = \bar S_t + k_t \delta_t^{\top}, \qquad
  o_t = S_t^{\top} q_t .
  ```

  Only 16 of the 64 layers therefore need a KV cache (§3.2). The catch is that the state *overwrites* the past.
  You can shorten a KV cache by cutting off its last tokens, but you cannot "un-apply" tokens from S. That is why
  prefix caching is harder for this model (§4.5).
- **The MTP layer** is a small extra transformer layer, trained together with the model, that predicts the token
  *after* the next one. It is a built-in draft model for speculative decoding (§4.4). Some conversions keep it and
  some drop it: `mlx-community/Qwen3.8-27B-4bit` has no MTP tensors, while `Jundot/Qwen3.8-27B-oQ4e-mtp` has 29.

## 3. Memory: where every gigabyte goes

[`mac_inference/memory_probe.py`](mac_inference/memory_probe.py) loads the model with mlx-lm and prefills 1K–32K
random tokens into a fresh cache. It records MLX's own counters (`mx.get_active_memory`, `mx.get_peak_memory`) and the
exact bytes held by every cache entry. Raw rows:
[`mac_inference/results_m5pro_qwen3.8-27b_memory.jsonl`](mac_inference/results_m5pro_qwen3.8-27b_memory.jsonl).

![memory vs context](docs/mac_inference_figs/memory_vs_context.png)

### 3.1 Weights

Bytes on disk of `mlx-community/Qwen3.8-27B-4bit`, by component (read from the safetensors headers):

| component | bytes | stored as |
|---|---:|---|
| 64 decoder layers | 13.70 GB | affine 4-bit, group size 64 |
| input embedding (248,320 × 5,120) | 0.72 GB | 4-bit |
| output head (248,320 × 5,120) | 0.72 GB | 4-bit |
| vision encoder | 0.92 GB | BF16 (not quantized) |
| **total** | **16.05 GB** | |

mlx-lm serves text only and **does not load the vision encoder**, so its weights take **15.13 GB** in memory, exactly
what `memory_probe.py` measures. mlx-vlm and oMLX load all 16.05 GB.

Why 4-bit is about 4.5 bits per weight. **Affine group quantization** stores each block of 64 weights as 4-bit
integers q plus one BF16 scale s and one BF16 bias b per block:

```math
w \approx s \cdot q + b, \qquad \text{bits per weight} = 4 + \frac{16 + 16}{64} = 4.5 .
```

At 4.5 bits, 27B parameters take 27 × 10⁹ × 4.5 / 8 ≈ 15.2 GB, which matches the measured size. The same model in
other formats:

| format | bits / weight | size | used by |
|---|---:|---:|---|
| BF16 (original) | 16 | about 54 GB (Unsloth lists 56 GB including vision) | — (would leave about 0 GB for context on a 64 GB Mac) |
| 8-bit affine | 8.5 | about 29 GB | (mlx-community also has this one) |
| **affine 4-bit, group 64** | 4.5 | **15.1 GB** text / 16.05 GB with vision | MLX-LM, mlx-vlm, oMLX, lithos-metal |
| **oQ4e** (oMLX mixed precision) | 4.89 average | 16.3 GB loaded, 17.0 GB on disk | oMLX |
| **NVFP4** (NVIDIA ModelOpt) | about 4.5 | 22 GB download, a 19 GB lithos-metal pack | lithos-metal |
| MTP head (separate, 4-bit) | — | 0.27 GB | mlx-vlm |
| DSpark draft head (NVFP4) | — | 1.1 GB | lithos-metal |

### 3.2 KV cache

Only the 16 full-attention layers store keys and values. Per token, in BF16 (2 bytes):

```math
\underbrace{16}_{\text{layers}} \times \underbrace{2}_{K,\,V} \times \underbrace{4}_{\text{KV heads}} \times \underbrace{256}_{\text{head dim}} \times 2\ \text{bytes} = 65{,}536\ \text{bytes} = 64\ \text{KiB per token}.
```

`memory_probe.py` measured **exactly 65,536 bytes per token** at every length. MLX grows the cache in blocks of 256
tokens.

| context | 1K | 4K | 16K | 32K | 128K | 256K (native max) |
|---|---:|---:|---:|---:|---:|---:|
| KV cache | 0.07 GB | 0.27 GB | 1.07 GB | 2.15 GB | 8.6 GB | 17.2 GB |
| if all 64 layers were full attention | 0.27 GB | 1.07 GB | 4.3 GB | 8.6 GB | 34 GB | **69 GB** |

The hybrid design is what lets a laptop hold the full 256K context. With full attention in all 64 layers, 256K tokens
would need 69 GB for the cache alone.

Grouped-query attention helps in the same way. The model has 24 query heads but only 4 KV heads, which shrinks the cache
6×. Both savings multiply.

### 3.3 Recurrent state

Each Gated DeltaNet layer keeps a recurrent state of 48 heads × 128 × 128 in FP32, plus a convolution window of 3
steps × 10,240 channels in BF16:

```math
48 \text{ layers} \times \left(48 \cdot 128 \cdot 128 \cdot 4 + 3 \cdot 10{,}240 \cdot 2\right)\ \text{bytes} = 0.154\ \text{GB},
```

measured: **0.1539 GB, constant** at every context length. It does not grow; that is the point of linear attention.

### 3.4 Prefill scratch and the total

While a chunk of 2,048 prompt tokens is processed, MLX needs temporary buffers for activations, attention scores and
MLP intermediates. The measured peak minus the resident memory is **1.7–2.7 GB**:

| context | weights | KV cache | DeltaNet state | resident after prefill | **peak during prefill** |
|---:|---:|---:|---:|---:|---:|
| 1K | 15.13 | 0.07 | 0.154 | 15.36 GB | **17.0 GB** |
| 4K | 15.13 | 0.27 | 0.154 | 15.56 GB | **18.3 GB** |
| 16K | 15.13 | 1.07 | 0.154 | 16.36 GB | **19.1 GB** |
| 32K | 15.13 | 2.15 | 0.154 | 17.44 GB | **20.1 GB** |

Resident memory = weights + KV + state, to the megabyte (15.13 + 2.15 + 0.154 = 17.44 GB at 32K). A smaller prefill
chunk (`--prefill-step-size`) lowers the scratch peak at the cost of more passes.

**What fits on which Mac** (4-bit, one request, MLX), from the table and the GPU limits in §1:

| Mac | GPU limit | usable context for Qwen3.8-27B 4-bit |
|---|---:|---|
| 24 GB | about 17.8 GB | only short prompts. 15.1 GB of weights + 2 GB of scratch already hits the limit, so use a smaller model or a 3-bit quantization. |
| 32 GB | about 24 GB | about 32K–64K tokens |
| 48 GB | about 38 GB | the full 256K (about 35 GB peak) |
| 64 GB (this machine) | 51.8 GB | 256K with room for prefix caches |

The 32 GB and 48 GB limits assume the same ~75 % ratio as the two measured machines.

### 3.5 What servers add on top

A server holds more than one request's state: prefix caches, allocator pools, sometimes a draft model. These are
the peak footprints of the serving process measured during our tests (`/usr/bin/time -l` and `footprint`):

| server | decode test (short prompts) | after the 1K–32K prefill tests | why |
|---|---:|---:|---|
| MLX-LM | 17.6 GB | **36.1 GB** | its prompt cache keeps up to 10 complete caches with no byte limit by default (`--prompt-cache-size 10`, `--prompt-cache-bytes` unset) |
| mlx-vlm + MTP | 18.4 GB | 28.7 GB | + vision encoder + MTP head; APC pool and disk tier |
| oMLX | 17–19 GB | 22–23 GB | paged hot cache sized as a fraction of RAM; cold blocks go to SSD |
| lithos-metal | 8.5 GB | 13–15 GB | weights are memory-mapped from its pack files, and file-backed pages are not counted, so this is *not* less memory |

Disk matters too. lithos-metal writes a converted weight "pack" per model and mode: **70 GB** for our four
configurations. mlx-vlm's APC disk tier is capped at 20 GB, and oMLX's SSD cache defaults to half the free disk.

## 4. Optimization techniques

### 4.1 Quantization

Decode time is proportional to the bytes of weights, so fewer bits per weight means proportionally faster decoding
and less memory. The formats used here:

- **Affine group quantization** (MLX's default; §3.1). Simple and fast to dequantize inside the matmul kernel. All
  layers get the same bits.
- **Mixed precision (oMLX's oQ).** Not all layers are equally sensitive. oQ measures each layer's sensitivity on
  calibration text (an "importance matrix") and gives more bits where errors cost the most. oQ4e averages 4.89
  bits/weight. The checkpoint's card reports a mean next-token KL divergence from BF16 of 0.012 on code and 0.060 on
  prose. The price is a more complex dequantization: oQ4e decodes at the same 17 tok/s as affine 4-bit, but its
  prefill is about 20 % slower (§6.2).
- **NVFP4** (NVIDIA's 4-bit float). Each value is a 4-bit float (E2M1), with an FP8 scale per block of 16 values and an
  FP32 scale per tensor. On NVIDIA Blackwell GPUs the tensor cores compute on it directly. On a Mac it is just a storage
  format that the kernel decodes: lithos-metal ran it slightly *faster* than affine 4-bit without drafting (13.7 vs 12.9
  tok/s), but slower with DSpark (40.9 vs 44.3).
- **KV-cache quantization** (mlx-lm `--kv-bits`, mlx-vlm TurboQuant, oMLX Affine4/Affine8). This shrinks §3.2's
  64 KiB/token. It matters at long context; we did not use it.

### 4.2 Kernel fusion and compilation

A transformer layer is dozens of small operations: norm, projections, RoPE, attention, gating, activation, residual.
Run naively, each is a separate GPU kernel that writes its result to memory, only for the next kernel to read it
back. Fusion merges such chains into one kernel that keeps intermediates in registers or on-chip memory.

- **MLX** has hand-written fused kernels (`mx.fast.scaled_dot_product_attention`, `rms_norm`, `rope`), quantized
  matmuls that dequantize on the fly, and `mx.compile` to fuse element-wise chains. The same idea made the official
  Parameter Golf MLX trainer 4.8× faster than PyTorch on MPS in our training tests (LAPTOP_QUICKSTART Step 7b).
- **oMLX** adds native Metal kernels for this model family (`qwen35_prefill`, `decode_fast`). Without them (a source
  install without the full Xcode) it falls back to a much slower path.
- **lithos-metal** goes furthest: it compiles a whole attention or DeltaNet *mixer* into one **megakernel**. The
  operators are split into tiles and tasks with dependencies, executed by persistent threadgroups ("workers") that
  keep shared intermediates local, following Mirage's Persistent Kernel design. A related trick, **input-normalization
  fusion**, moves RMSNorm into the next matmul's epilogue: it computes r(h)·((h·γ)Wᵀ) instead of (r(h)·h·γ)Wᵀ. That
  removes a kernel but changes BF16 rounding, so it can be switched off.

### 4.3 Keeping the CPU out of the loop

Python launching every kernel of every token costs time. lithos-metal encodes the whole decode step once, into a
Metal **indirect command buffer**, and replays it. The token position, accepted drafts, stop flags and the
"rewind" of the DeltaNet state all live in GPU memory, so the host only collects finished tokens. MLX reduces
host overhead differently: lazy evaluation builds a graph and submits it in large batches.

### 4.4 Speculative decoding

Decode reads all weights to produce one token. **Verifying** k proposed tokens in one pass costs almost the same
as producing one, because the weight reads dominate (§1). So:

1. a cheap **drafter** proposes k tokens;
2. the big **target** model scores all k positions in one pass;
3. the longest prefix the target agrees with is accepted, plus one token from the target itself.

With greedy decoding the output is the same as without speculation, up to rounding (§6.5). If each draft token is
accepted with probability α independently, the expected number of tokens per round is

```math
E[\text{tokens per round}] = \frac{1 - \alpha^{k+1}}{1 - \alpha}, \qquad
\text{speed-up} = E[\text{tokens}] \cdot \frac{t_{\text{plain step}}}{t_{\text{round}}} .
```

Measured for lithos-metal's DSpark drafter (k = 7):

| prompt | rounds for 512 tokens | tokens per round | implied α | round cost / plain step | speed-up vs its own plain decode |
|---|---:|---:|---:|---:|---:|
| code, math | 94 | 5.45 | 0.89 | 105 ms / 77 ms | 5.45 × 77 / 105 = **4.0×** (measured 51.6 / 12.9 = 4.0×) |
| prose | 163 | 3.14 | 0.70 | 105 ms / 77 ms | **2.3×** (measured 29.6 / 12.9 = 2.3×) |

The formula predicts the measurement exactly. Code and math are predictable: the drafter guesses syntax and
arithmetic right. Prose has many equally good continuations, so fewer guesses survive. Three kinds of drafters
appear in this tutorial:

| drafter | what it is | engines |
|---|---|---|
| **MTP head** | the model's own extra layer, trained with it (0.27 GB) | mlx-vlm (block of 3), oMLX "Lightning MTP" (draft depth chosen adaptively, up to 8) |
| **DSpark** | a separately trained drafter that reads the target's hidden states and proposes a block of 7, with a Markov head for dependencies between proposals (1.1 GB) | lithos-metal |
| **separate small model** | e.g. a 0.8B model with the same tokenizer (`--draft-model`) | mlx-lm (not tested here); oMLX also supports DFlash block-diffusion drafters |

### 4.5 Prefix caching

An agent sends the same long prefix (system prompt, tools, files, earlier turns) with every request. If the server
keeps the model state for that prefix, a follow-up request only needs to prefill the new tokens. On a Mac, where
prefill costs about 2.3 ms per token (§1), that is the difference between 37 s and 0.5 s.

How engines store it:

- **Whole-cache LRU (MLX-LM).** It keeps up to 10 complete caches, keyed by their token sequence in a trie. A new
  prompt reuses an exact match or the longest cached prefix. A KV cache could also be *trimmed* back to a shorter
  common prefix, but a DeltaNet state cannot (`is_trimmable()` returns False). For Qwen3.8, MLX-LM therefore reuses a
  cache only when the new prompt *extends* the cached one. That covers append-only agent turns, but not edited history.
- **Paged blocks + state checkpoints (mlx-vlm APC, oMLX, lithos-metal).** KV entries are stored in fixed blocks of
  16 tokens, each identified by a chained hash of all tokens up to it. That is the vLLM design; mlx-vlm's `apc.py`
  documents it. Identical prefixes share blocks, with copy-on-write in oMLX. For the recurrent layers they store
  **checkpoints** of the state at boundaries: mlx-vlm every 2,048 tokens by default, lithos-metal at chunk boundaries.
  A new prompt can then resume from the last checkpoint inside the shared prefix. Image content is hashed into the key,
  so the same tokens with a different image never collide.
- **Tiers.** oMLX writes cold blocks to SSD in safetensors format and restores them even after a restart. mlx-vlm also has
  a disk tier, and lithos-metal can persist checkpoints with `--prefix-cache-dir`.

**mlx-vlm's prefix cache is off by default.** Its server reads `APC_ENABLED` from the environment, default `0`.

### 4.6 Chunked prefill and batching

Prefill runs in chunks (MLX: 2,048 tokens; lithos-metal: 128 on this chip, 512 qualified only on the M5 Max).
Larger chunks reuse each weight read for more tokens but need more scratch (§3.4). **Continuous batching** (oMLX)
interleaves several requests' decode steps in one batch, which raises total throughput when more than one client is
active. Our tests send one request at a time.

## 5. The four engines and what each optimizes

| | **MLX-LM** | **mlx-vlm** | **oMLX** | **lithos-metal** |
|---|---|---|---|---|
| what it is | Apple's reference LLM runner on MLX | MLX runner for vision-language models | a server on top of mlx-lm + mlx-vlm, with a menu-bar app | its own engine that generates Metal megakernels |
| kernels | MLX quantized matmul, fused SDPA, `mx.compile` | MLX | MLX + native Qwen3.5-family prefill/decode kernels | generated per-layer megakernels, per-chip tuned recipes |
| weights | affine 2–8-bit | affine | affine, **oQ mixed precision** | affine 4-bit (`mlx-community`), **NVFP4** |
| speculative decoding | separate draft model only; **drops Qwen3.8's MTP weights** | **MTP**, EAGLE-3, DFlash drafters | **Lightning MTP** (adaptive depth), DFlash | **DSpark** (7 drafts + anchor) |
| prefix cache | LRU of whole caches (no trimming for DeltaNet) | APC: hashed 16-token blocks + state checkpoints, RAM + disk (**opt-in**) | paged blocks, copy-on-write, RAM + **SSD** tiers | exact-token prefix checkpoints, RAM (+ optional disk) |
| GPU-resident decode loop | — | — | — | **yes** (indirect command buffer) |
| batching | sequential | sequential | **continuous batching** | one generation at a time |
| vision | — (404 on image input) | **yes** | **yes** (images, video) | — |
| APIs | OpenAI | OpenAI | OpenAI + **Anthropic**; admin UI; integrations for OpenCode, Codex, … | OpenAI, Responses, Anthropic, Ollama |
| install | `pip install mlx-lm` | `pip install mlx-vlm` | release wheel or DMG (precompiled kernels) | build from source (22 s) or Homebrew |
| models | almost every MLX checkpoint | most VLMs | mlx-lm + mlx-vlm models | a few validated pairs (Qwen3.8-27B, Qwen3.6-35B-A3B) |

Other options exist but were not measured here:
- **llama.cpp** (GGUF, with the vision part in a separate `mmproj` file): reports say its MTP and vision do not yet
  combine well on Apple Silicon.
- **Ollama:** reportedly has a `qwen3.8:27b-mlx` tag.
- **LM Studio:** its MLX engine builds on mlx-lm and mlx-vlm.

## 6. Measurements

Every engine was measured with the same clients. Requests are streamed, greedy (`temperature 0`), with thinking off.
lithos-metal always disables thinking, so the MLX engines get `enable_thinking: false`. One server runs at a time, and
nothing else uses the GPU.

### 6.1 Decode

[`bench_client.py`](mac_inference/bench_client.py) sends 3 prompts twice each: a Python function with tests, a prose
explanation, and a word problem. Each request generates 512 tokens. Decode tok/s = (tokens − 1) / (time from first to
last token), using the server-reported token count. Each pair of runs agreed within 2 %.

![decode by engine](docs/mac_inference_figs/decode_by_engine.png)

| engine | weights | speculative | code | prose | math | **mean** | vs MLX-LM | start-up |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MLX-LM | 4-bit | — | 17.4 | 17.4 | 17.4 | **17.4** | 1.0× | 2 s |
| mlx-vlm | 4-bit | — | 17.8 | 17.6 | 17.6 | **17.6** | 1.0× | 4 s |
| mlx-vlm | 4-bit | MTP | 40.2 | 32.5 | 37.4 | **36.7** | 2.1× | 7 s |
| mlx-vlm, `APC_ENABLED=1` | 4-bit | MTP | 40.2 | 32.0 | 36.9 | **36.3** | 2.1× | 4 s |
| oMLX | 4-bit | — | 17.3 | 17.0 | 17.3 | **17.2** | 1.0× | 2 s† |
| oMLX | oQ4e | — | 17.2 | 17.1 | 17.0 | **17.1** | 1.0× | 2 s† |
| oMLX | oQ4e | Lightning MTP | 42.0 | **34.3** | 44.9 | **40.4** | 2.3× | 2 s† |
| lithos-metal | 4-bit | — | 13.0 | 12.9 | 12.9 | **12.9** | 0.74× | 40 s |
| lithos-metal | 4-bit | DSpark | **51.6** | 29.6 | **51.7** | **44.3** | **2.5×** | 37 s |
| lithos-metal | NVFP4 | — | 13.8 | 13.5 | 13.8 | **13.7** | 0.79× | 71 s |
| lithos-metal | NVFP4 | DSpark | 46.6 | 29.5 | 46.7 | **40.9** | 2.4× | 53 s |

† oMLX loads a model on its first request (about 3 s), not at start-up. lithos-metal's start-up includes building its
weight pack the first time, and compiling and warming up kernels every time.

### 6.2 Prefill and time to first token

[`prefill_client.py`](mac_inference/prefill_client.py) sends a fresh prompt of 1K–32K tokens (this course's own
Markdown files). It starts with a random session id, so no cache can help, and asks for a one-sentence answer.
TTFT is the time until the first content token arrives.

![TTFT vs prompt length](docs/mac_inference_figs/ttft_vs_prompt.png)

| prompt | MLX-LM | mlx-vlm | oMLX 4-bit | oMLX oQ4e | lithos-metal | lithos-metal, 512-token chunks |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 2.4 s (437 tok/s) | **2.2 s** (491) | 2.3 s (458) | 2.7 s (399) | 3.3 s (320) | 3.3 s (320) |
| 4K | 8.7 s (475) | **8.4 s** (495) | 8.9 s (467) | 11.7 s (354) | 13.8 s (300) | 13.4 s (308) |
| 16K | 37 s (441) | **36 s** (454) | 38 s (428) | 47 s (347) | 74 s (223) | 71 s (232) |
| 32K | 82 s (402) | **78 s** (419) | 85 s (387) | 103 s (319) | 204 s (161) | 198 s (166) |

The lithos-metal columns are with DSpark. Without a draft it is the same: 73 s at 16K and 202 s at 32K.

### 6.3 The agent pattern

The client sends a 16K-token system prompt, then three user turns. Each request carries the whole conversation
so far (append-only), as a coding agent does.

| engine | turn 1 (cold) | turn 2 | turn 3 | prefix reuse |
|---|---:|---:|---:|---|
| MLX-LM | 39.6 s | **0.48 s** | **0.49 s** | prompt cache |
| mlx-vlm + MTP, default | 36.8 s | 37.5 s | 38.0 s | **none: APC is off by default** |
| mlx-vlm + MTP, `APC_ENABLED=1` | **33.9 s** | 0.68 s | 0.75 s | APC |
| oMLX, 4-bit | 40.6 s | 0.78 s | 0.53 s | paged RAM/SSD cache |
| oMLX, oQ4e + MTP | 49.7 s | 0.81 s | 0.59 s | paged RAM/SSD cache |
| lithos-metal + DSpark | 77.8 s | 0.96 s | 0.92 s | prefix checkpoints (`cached_prompt_tokens: 16417`) |

### 6.4 Vision

[`vision_client.py`](mac_inference/vision_client.py) sends one of this course's figures
([`laptop_golf_ablation_bars.png`](docs/nanolm_figs/laptop_golf_ablation_bars.png); 477 prompt tokens with the image).
It asks which bar is lowest, and sends the same request twice.

| engine | TTFT, first | TTFT, repeat | answer |
|---|---:|---:|---|
| mlx-vlm (MTP, APC) | 2.6 s | 0.17 s | "stack_verified … approximately 1.60" ✓ |
| oMLX (oQ4e, MTP) | 6.7 s | 0.31 s | "stack_verified … approximately 1.60" ✓ |
| MLX-LM | — | — | HTTP 404: `mlx_lm.server` serves text only |
| lithos-metal | — | — | text-only |

Both answers are correct. Each misread the axis label slightly ("val_pbp", "val_ppl" instead of `val_bpb`).
Both servers cache the image; the repeat request skips the vision encoder.

### 6.5 Correctness

Greedy outputs from MLX-LM and lithos-metal without drafting were identical: the first 300 characters of all 3
prompts. The oMLX outputs began with the same text. With DSpark, 2 of 3 prompts matched. The word problem diverged
after 195 characters. That is expected: batched verification of 8 positions rounds BF16 differently from one-at-a-time
decoding, and both lithos-metal and mlx-vlm document this "batch non-invariance". Exact equality would need
batch-invariant kernels.

## 7. What the measurements teach

1. **Decode is a bandwidth problem, and MLX is already near the ceiling.** MLX-LM reaches 91 % of the chip's
   measured bandwidth. Kernel work can only take a single-token decode from 17 to at most 19 tok/s. lithos-metal's
   megakernels are *slower* here (12.9 tok/s, 77 vs 57 ms per token), because the project has no tuned 27B recipes for
   the M5 Pro: its log shows `recipe contexts=[]`. Its 27B kernels are tuned on a 40-core M5 Max.
2. **Speculative decoding is the only large decode win: 2–3×.** It reads the weights once for several tokens. Its benefit
   depends on the text: DSpark accepts 5.4 tokens per round on code but 3.1 on prose (α = 0.89 vs 0.70). Qwen's own MTP head
   in oMLX's Lightning MTP is the most even (34–45 tok/s), and the best on prose.
3. **Prefill is a compute problem, and that is the Mac's real limit for agents.** About 26 TFLOP/s means 37 s for a 16K-token
   prompt with the fastest engine. Neither kernel fusion nor speculation helps here:
   - lithos-metal is 2× slower (its prefill kernels are untuned on this chip). Larger chunks gained 3 %, and DSpark adds nothing.
   - oQ4e is 20 % slower than affine 4-bit, because of the more expensive dequantization.
   - oMLX's INT8-activation prefill for M5 (`qwen35_oq_a8_enabled`) is designed for exactly this, but its kernels were
     not in the pip wheel (`oq_a8_available()` returned `False`). oMLX ignored the setting silently, so we could not measure it.
4. **For agents, prefix caching matters more than raw prefill speed.** With the cache on, every engine answers a
   follow-up turn in under 1 s instead of 34–78 s. A cache that is *off by default* (mlx-vlm) makes a fast engine look slow.
5. **Hybrid models change memory and caching.** The DeltaNet layers make the KV cache 4× smaller (§3.2) and hold a
   constant 0.15 GB of state. In return, a cache cannot be shortened, so engines need state checkpoints, and MLX-LM can
   only reuse a cache when the prompt extends it.
6. **Read headline numbers carefully.** lithos-metal's README quotes about 168 tok/s for this model. That figure is for a
   40-core M5 Max with about twice the bandwidth. It is also a *projection*: it assumes 6 of 8 drafts are accepted and
   leaves out drafting time. Measured end to end on the M5 Pro we got 29–52 tok/s.

A rough per-turn budget for an agent, estimated from the measured rates (not measured end to end): append 2K tokens of
tool output, then write 300 tokens.

| engine | prefill 2K | decode 300 tokens | per turn |
|---|---:|---:|---:|
| MLX-LM | about 4.4 s | 17 s | **about 22 s** |
| mlx-vlm + MTP (APC) | about 4.1 s | 8–9 s | **about 13 s** |
| oMLX oQ4e + MTP | about 5.6 s | 7–9 s | **about 13–15 s** |
| lithos-metal + DSpark | about 6.6 s | 6–10 s | **about 13–17 s** |

## 8. Choosing an engine

| need | choice | why |
|---|---|---|
| text agent with many turns | **oMLX** (oQ4e + MTP) or **mlx-vlm** + MTP with `APC_ENABLED=1` | 2.1–2.3× decode, prefix reuse under 1 s, Anthropic API (oMLX) for Claude Code |
| vision (screenshots, figures, video) | **mlx-vlm** (+ APC) or **oMLX** | the only two here that serve images; both cache them |
| large new chunks every turn (a whole file) | **mlx-vlm**, **MLX-LM**, or oMLX with affine 4-bit | the fastest prefill (430–495 tok/s) |
| short, code-heavy turns | **lithos-metal** + DSpark | 52 tok/s on code, but slow prefill, text only, few models |
| learning, reference numbers, scripting | **MLX-LM** | simplest; the baseline every other engine builds on |

On this Mac the best all-round setup was **oMLX with the oQ4e-mtp checkpoint**: 40 tok/s, sub-second follow-up turns,
vision, and Anthropic/OpenAI APIs. **mlx-vlm with MTP and `APC_ENABLED=1`** was a close second, with the fastest prefill.

## 9. Reproduce it

Everything runs from `DeepDataMiningLearning/llm`. Each engine gets its own venv (Python 3.12 via `uv`, see
[`LAPTOP_QUICKSTART.md`](LAPTOP_QUICKSTART.md) Step 1a).

```bash
uv python install 3.12; PY=$(uv python find 3.12)
$PY -m venv ~/venvs/mlxlm  && ~/venvs/mlxlm/bin/pip install mlx-lm
$PY -m venv ~/venvs/mlxvlm && ~/venvs/mlxvlm/bin/pip install mlx-vlm
$PY -m venv ~/venvs/omlx   && ~/venvs/omlx/bin/pip install \
  https://github.com/jundot/omlx/releases/download/v0.7.1rc1/omlx-0.7.1rc1-cp312-cp312-macosx_15_0_universal2.whl
git clone https://github.com/lithos-ai/lithos-metal.git ~/src/lithos-metal
$PY -m venv ~/venvs/lithos && ~/venvs/lithos/bin/pip install "$HOME/src/lithos-metal[serve]"

hf download mlx-community/Qwen3.8-27B-4bit          # 16 GB: all engines
hf download mlx-community/Qwen3.8-27B-MTP-4bit      # 0.27 GB: MTP head for mlx-vlm
hf download Jundot/Qwen3.8-27B-oQ4e-mtp             # 17 GB: oMLX's oQ4e with MTP
hf download LithosAI/Qwen3.8-27B-DSpark-NVFP4       # 1.1 GB: DSpark head for lithos-metal
hf download nvidia/Qwen3.8-27B-NVFP4                # 22 GB: optional
```

Each download took about 4–5 min here.
- **oMLX:** install it from the release wheel or the DMG, which ship the native kernels precompiled. Check with
  `python -c "from omlx.custom_kernels import native_kernel_status; print(native_kernel_status())"`.
- **lithos-metal:** compiles its Metal shaders at run time, so the Command Line Tools are enough.

oMLX serves the model directories under `--model-dir`, with per-model options in `model_settings.json`. We used an
isolated base directory:

```bash
mkdir -p ~/omlx_bench/models
ln -s ~/.cache/huggingface/hub/models--Jundot--Qwen3.8-27B-oQ4e-mtp/snapshots/* ~/omlx_bench/models/Qwen3.8-27B-oQ4e-mtp
cat > ~/omlx_bench/model_settings.json <<'EOF'
{"version": 1, "models": {"Qwen3.8-27B-oQ4e-mtp":
  {"mtp_enabled": true, "chat_template_kwargs": {"enable_thinking": false}}}}
EOF
```

The runs ([`serve_bench.sh`](mac_inference/serve_bench.sh) starts a server, waits for it, runs a client, stops it and
records the serving process's peak memory):

```bash
cd mac_inference
M=mlx-community/Qwen3.8-27B-4bit
bash serve_bench.sh mlx_4bit ~/venvs/mlxlm $M -- \
  mlx_lm.server --model $M --port 8000 --chat-template-args '{"enable_thinking": false}'
APC_ENABLED=1 bash serve_bench.sh mlxvlm_mtp ~/venvs/mlxvlm $M -- \
  python -m mlx_vlm server --model $M --draft-model mlx-community/Qwen3.8-27B-MTP-4bit --draft-kind mtp --port 8000
bash serve_bench.sh lithos_dspark ~/venvs/lithos $M -- lithos-metal serve --model $M --max-context 40960
bash run_omlx.sh Qwen3.8-27B-oQ4e-mtp                  # decode + prefill for one oMLX model id

CLIENT=prefill_client.py OUT=prefill.jsonl bash serve_bench.sh ...   # prefill/agent test, same server commands
CLIENT=vision_client.py  OUT=vision.jsonl  bash serve_bench.sh ...   # vision test
~/venvs/mlxlm/bin/python memory_probe.py --sizes 1024 4096 16384 32768
~/venvs/nanolm-pip/bin/python make_figures.py          # the three figures above
```

[`run_prefill_all.sh`](mac_inference/run_prefill_all.sh) lists every prefill configuration we ran.

**Clean-up.** This uses about 36 GB of models in `~/.cache/huggingface`, plus 17 GB for oQ4e. lithos-metal's packs take
70 GB in `~/.cache/lithos-metal`, and mlx-vlm's APC disk tier up to 20 GB in `~/.cache/mlx-vlm/apc`. There are also four
venvs and `~/omlx_bench`.

**Gotchas we hit.**
- `mlx_lm.server` ignores Ctrl-C; stop it with SIGTERM.
- oMLX's launcher starts a detached `omlx-server` process. It renames its process title, and it keeps running when the
  launcher exits; `serve_bench.sh` stops whatever listens on port 8000.
- oMLX also auto-discovers every model in the Hugging Face cache.

## 10. Exercises

1. **Bandwidth ceiling.** Measure MLX-LM's decode with the 8-bit checkpoint (`mlx-community/Qwen3.8-27B-8bit`). Predict
   the speed from §1 before running it. How close is the result to bandwidth ÷ bytes?
2. **Acceptance rate.** Run the decode client against mlx-vlm with `--draft-block-size` 1, 2 and 3. From the measured
   speed-ups, estimate α with the formula of §4.4. Does α depend on the block size?
3. **Context capacity.** Extend `memory_probe.py` to 64K and 128K tokens. Check §3.2's projection, and find where peak
   memory crosses 24 GB and 32 GB.
4. **Cache breakers.** Change one word near the *start* of the 16K-token system prompt between agent turns. Which
   engines still answer in under 1 s, and why (§4.5)?
5. **Prefill chunk.** Vary MLX-LM's `--prefill-step-size` (512, 1024, 2048, 4096). Plot TTFT and peak memory against chunk
   size and explain the trade-off with §3.4.
6. **Quality vs speed.** Write 20 short questions with known answers, and send them through the same OpenAI API
   (adapt `bench_client.py`). Use oMLX serving the affine 4-bit checkpoint and the oQ4e checkpoint. Count how many
   greedy answers differ and how many are wrong. Is the gap consistent with the KL divergences on the oQ4e model card (§4.1)?

---

**Sources.**
- Engines: [mlx-lm](https://github.com/ml-explore/mlx-lm), [mlx-vlm](https://github.com/Blaizzy/mlx-vlm),
  [oMLX](https://github.com/jundot/omlx), [lithos-metal](https://github.com/lithos-ai/lithos-metal)
  (its design docs: `docs/design/*.md`).
- Model and checkpoints: [Qwen3.8 guide (Unsloth)](https://unsloth.ai/docs/models/qwen3.8),
  [`Jundot/Qwen3.8-27B-oQ4e-mtp`](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp),
  [`mlx-community/Qwen3.8-27B-MTP-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit).
- Megakernels: [Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219). DSpark: [arXiv 2607.05147](https://arxiv.org/abs/2607.05147).
- llama.cpp and Apple Silicon: [MTP on Metal issue](https://github.com/ggml-org/llama.cpp/issues/23752),
  [MTP and mmproj](https://github.com/unslothai/unsloth/issues/5534).
