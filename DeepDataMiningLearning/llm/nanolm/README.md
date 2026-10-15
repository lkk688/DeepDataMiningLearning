# nanolm: config-driven language-model lab (Mac, CUDA, or CPU)

The full course is in [`../TUTORIAL_nanolm.md`](../TUTORIAL_nanolm.md). It runs from an RNN to a
character-level Transformer, then to GPT pre-training on FineWeb, and finally to Parameter Golf
techniques, with measured results at each step.

```bash
cd DeepDataMiningLearning/llm
pip install torch numpy pyyaml sentencepiece tiktoken pandas matplotlib scipy huggingface_hub datasets pytest
python -m pytest nanolm/tests -q                                   # correctness tests
python -m nanolm.train -c nanolm/configs/00_smoke.yaml             # 30 s, any device
python -m nanolm.train -c nanolm/configs/02b_char_transformer_modern.yaml model.n_layers=6   # override any key
python -m nanolm.prepare --golf-shards 1 --dest data/golf          # FineWeb sp1024 (Parameter Golf data)
python -m nanolm.sweep   --sweep nanolm/configs/sweeps/04_golf_ablation.yaml --gpus 0,1
python -m nanolm.analyze outputs/nanolm/sweeps/golf_ablation       # mean±std over seeds, paired tests, plots
python -m nanolm.posttrain <run_dir> --bits 8 6 4 --strides 0 64   # quantization / eval-stride study
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
| `sweeps/*.yaml` | the studies behind every table in the tutorial |

Each run writes `outputs/nanolm/<date>_<name>/` containing:
- `config.yaml`
- `meta.json` (git commit, seed, device, command)
- `train_log.jsonl`
- `eval_log.jsonl`
- `metrics.json`
- `sample.txt`
- `model.pt`
