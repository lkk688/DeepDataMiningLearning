# Parameter Golf on Apple MLX

`train_gpt_mlx.py` is the official Apple-Silicon trainer from
[openai/parameter-golf](https://github.com/openai/parameter-golf/blob/main/train_gpt_mlx.py),
copied **unmodified** from upstream commit `6a08c9d` (2026-03-19), MIT License, Copyright (c) 2026 OpenAI.
It trains the challenge's baseline model with [MLX](https://github.com/ml-explore/mlx), Apple's own
array framework, instead of PyTorch. It runs in bf16 and compiles the training step with `mx.compile`.

The two other files are ours:
- `make_val_subset.py` makes a data dir whose val split is the first 2,097,152 targets of the official
  one. The MLX script always evaluates its whole val split (62M tokens). With the subset it scores
  exactly the tokens nanolm's laptop configs score, so the two val_bpb numbers can be compared directly.
- `run_mlx.sh` sets the script's environment variables. Preset `L5` mirrors
  `nanolm/configs/laptop/L5_golf_mini.yaml`: same model, batch, sequential data order, learning rates
  and warmdown. Preset `baseline` is the official 9 × 512 model at a laptop budget.

```bash
cd DeepDataMiningLearning/llm
pip install mlx
python -m nanolm.prepare --golf-shards 1 --dest data/golf       # if not done yet
python parameter_golf_mlx/make_val_subset.py                    # data/golf_val2m (symlinks + 4 MB)
bash parameter_golf_mlx/run_mlx.sh L5                           # logs + model in outputs/mlx/
```

Measured speed and results against nanolm (PyTorch MPS) are in
[`LAPTOP_QUICKSTART.md`](../LAPTOP_QUICKSTART.md), Step 7b.
