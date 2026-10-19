#!/usr/bin/env bash
# Run the official Parameter Golf MLX trainer (train_gpt_mlx.py, unmodified) on nanolm's data.
#
#   bash parameter_golf_mlx/run_mlx.sh L5          # MLX twin of nanolm/configs/laptop/L5_golf_mini.yaml
#   bash parameter_golf_mlx/run_mlx.sh baseline    # the official 9 x 512 baseline model, laptop budget
#   ITERATIONS=200 MLX_EAGER_EVAL=0 bash parameter_golf_mlx/run_mlx.sh L5    # any setting can be overridden
#
# Run from llm/. Needs `pip install mlx`, the data from `python -m nanolm.prepare --golf-shards 1 --dest
# data/golf`, and the 2M-token val subset from `python parameter_golf_mlx/make_val_subset.py`.
set -euo pipefail
cd "$(dirname "$0")/.."
preset=${1:-L5}

export DATA_PATH=${DATA_PATH:-data/golf_val2m/fineweb10B_sp1024}
export TOKENIZER_PATH=${TOKENIZER_PATH:-data/golf/tokenizers/fineweb_1024_bpe.model}
export OUT_DIR=${OUT_DIR:-outputs/mlx}
export VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-0}
export TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-100}
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-0}   # 0 = run all ITERATIONS, warmdown by step

case "$preset" in
  L5)   # same model, data order, batch, LRs and schedule as laptop/L5_golf_mini.yaml
    export RUN_ID=${RUN_ID:-L5_mlx}
    export NUM_LAYERS=${NUM_LAYERS:-6} MODEL_DIM=${MODEL_DIM:-384} NUM_HEADS=${NUM_HEADS:-6} NUM_KV_HEADS=${NUM_KV_HEADS:-3}
    export TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-512} TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-16384} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
    export ITERATIONS=${ITERATIONS:-2500} WARMDOWN_ITERS=${WARMDOWN_ITERS:-625}
    export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-131072}
    ;;
  baseline)   # official defaults (9 x 512, seq 1024), but 64k instead of 524k tokens per step
    export RUN_ID=${RUN_ID:-baseline_mlx}
    export TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-65536} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
    export ITERATIONS=${ITERATIONS:-1500} WARMDOWN_ITERS=${WARMDOWN_ITERS:-375}
    export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-524288}
    ;;
  *) echo "unknown preset '$preset' (L5 | baseline)"; exit 1 ;;
esac

exec python "${MLX_SCRIPT:-parameter_golf_mlx/train_gpt_mlx.py}"   # MLX_SCRIPT=<your edited copy> for experiments
