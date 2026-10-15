# nanolm: config-driven language-model lab (Mac, CUDA, or CPU)

The full course is in [`../TUTORIAL_nanolm.md`](../TUTORIAL_nanolm.md); the step-by-step laptop guide is
[`../LAPTOP_QUICKSTART.md`](../LAPTOP_QUICKSTART.md). The course runs:
- Part 1: language models from scratch (loss, data, tokenizers, RNN → LSTM → Transformer, sampling);
- Part 2: GPT pre-training on FineWeb;
- Part 3: from GPT to modern LLMs (Qwen3.5 inside, benchmarks such as MMLU/GPQA, LoRA, SFT, DPO);
- Part 4: Parameter Golf;
- Part 5: research methodology.

Every part has measured results.

```bash
cd DeepDataMiningLearning/llm
pip install torch numpy pyyaml sentencepiece tiktoken pandas matplotlib scipy huggingface_hub datasets pytest
python -m pytest nanolm/tests -q                                   # correctness tests
python -m nanolm.train -c nanolm/configs/00_smoke.yaml             # 30 s, any device
python -m nanolm.train -c nanolm/configs/02b_char_transformer_modern.yaml model.n_layers=6   # override any key
python -m nanolm.prepare --golf-shards 1 --dest data/golf          # FineWeb sp1024 (Parameter Golf data)
python -m nanolm.sweep   --sweep nanolm/configs/sweeps/04_golf_ablation.yaml --gpus 0,1
python -m nanolm.analyze outputs/nanolm/sweeps/golf_ablation       # mean±std over seeds, paired tests, plots
python -m nanolm.compress <run_dir> --bits 8 6 4 --strides 0 64   # quantization / eval-stride study
python -m nanolm.bench    -c nanolm/configs/laptop/L4_gpt_fineweb.yaml   # will it fit / how long, on THIS machine
python -m nanolm.generate <run_dir> --prompt "ROMEO:" --top-p 0.9   # sampling
python -m nanolm.finetune -c nanolm/configs/part2/P2_sft_alpaca.yaml # LoRA SFT of a pretrained LLM (also eval_bpb, dpo)
python -m nanolm.benchmarks run -c nanolm/configs/part3/B2_laptop.yaml   # MMLU/ARC/GSM8K/... via lm-evaluation-harness
python -m nanolm.viz all                                           # regenerate every tutorial figure
```

| config | what it trains |
|---|---|
| `00_smoke.yaml` | a tiny GPT for 60 steps (sanity check) |
| `01_char_rnn.yaml` | RNN / GRU / LSTM / hand-written LSTM on Tiny Shakespeare |
| `02_char_transformer.yaml`, `02b_...modern.yaml` | textbook vs. modern (RMSNorm + RoPE + SwiGLU) character Transformer |
| `03_gpt_fineweb.yaml` | GPT pre-training on FineWeb with the golf sp1024 tokenizer |
| `04_golf_baseline.yaml` | the Parameter Golf baseline (8×H100, 600 s) |
| `04_golf_lab.yaml` | the same model with a single-GPU budget (the ablation baseline) |
| `05_golf_stack.yaml` | the ablation-verified golf stack + int6 export + sliding eval (1.356 bpb at lab scale) |
| `laptop/L1-L6*.yaml` | laptop-sized versions of every stage (≤ ~4 PFLOP, ≤ 2 GB, wall-clock capped) |
| `part2/P1-P3.yaml` | pretrained LLMs: bits-per-byte eval, LoRA SFT, LoRA DPO |
| `part3/B1-B3.yaml` | benchmarks for small open models (B2 = laptop subset, B3 = Qwen3.5) |
| `sweeps/*.yaml` | the studies behind every table in the tutorial (`laptop_*` ones run on a laptop) |

Each run writes `outputs/nanolm/<date>_<name>/` containing:
- `config.yaml`
- `meta.json` (git commit, seed, device, command)
- `train_log.jsonl`
- `eval_log.jsonl`
- `metrics.json`
- `sample.txt`
- `model.pt`
