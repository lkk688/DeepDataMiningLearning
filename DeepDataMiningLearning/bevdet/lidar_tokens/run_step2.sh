#!/usr/bin/env bash
# 1. released model's occupancy on 120 real nuScenes val frames  (headroom, first half)
# 2. rung 0 with the height histogram, on real frames
# 3. rung 0 with the frozen LiDAR-only detector's queries, SAME frames
# The read for 2 vs 3 is `self vs other`: same VLM, same images, same untrained projector
# architecture, only the tokeniser differs.
set -u
cd "$(dirname "$0")"
PY=/home/010796032/miniconda3/envs/py310/bin/python
export CUDA_HOME=/data/rnd-liu/cudatk
export CPATH=$CUDA_HOME/targets/x86_64-linux/include
export PATH=$(dirname $PY):$PATH
export PYTHONUNBUFFERED=1
PACK=/data/rnd-liu/Datasets/nuScenes/packed_qd_val120
Q=$PACK/lidar_queries.npz

echo "===== 1. released model on 120 packed nuScenes val frames"
$PY train_projector.py --inject none --frames $PACK \
    --exp-name nusc120_released --out-root outputs > logs/nusc120_released.log 2>&1
echo "  exit=$?"; grep -E "frames:|\[eval@" logs/nusc120_released.log | tail -2

for T in pillar queries; do
  echo
  echo "===== rung 0, tokeniser = $T"
  EXTRA=""
  [ "$T" = queries ] && EXTRA="--queries $Q"
  $PY train_projector.py --mode sensitivity --frames $PACK --max-frames 2 \
      --tokenizer $T $EXTRA --exp-name sens_${T}_nusc --out-root outputs \
      > logs/sens_${T}_nusc.log 2>&1
  echo "  exit=$?"
  grep -E "^  \[|^pair|vs |^n = |queries:|K [0-9]" logs/sens_${T}_nusc.log
done
