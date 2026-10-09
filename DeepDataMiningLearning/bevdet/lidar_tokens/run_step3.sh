#!/usr/bin/env bash
# rung 0: height histogram vs frozen LiDAR-only detector queries, same frames, same probe.
# Plus the released model's occupancy INSIDE mask_camera (the GT-convention-corrected
# headroom number -- see QWEN_DRIVE_LIDAR_TOKENS.md, "common ruler needs a common GT").
set -u
cd "$(dirname "$0")"
PY=/home/010796032/miniconda3/envs/py310/bin/python
export CUDA_HOME=/data/rnd-liu/cudatk
export CPATH=$CUDA_HOME/targets/x86_64-linux/include
export PATH=$(dirname $PY):$PATH
export PYTHONUNBUFFERED=1
PACK=/data/rnd-liu/Datasets/nuScenes/packed_qd_val120
Q=$PACK/lidar_queries.npz

echo "===== rung 0, tokeniser = queries (200 x 128 frozen TransFusion queries)"
$PY train_projector.py --mode sensitivity --frames $PACK --max-frames 2 \
    --tokenizer queries --queries $Q --exp-name sens_queries_nusc --out-root outputs \
    > logs/sens_queries_nusc.log 2>&1
echo "  exit=$?"
grep -E "^  \[|^pair|vs |^n = |queries:|K [0-9]" logs/sens_queries_nusc.log

echo
echo "===== released model on 120 nuScenes frames, INSIDE mask_camera"
$PY train_projector.py --inject none --frames $PACK --occ-mask camera \
    --exp-name nusc120_masked --out-root outputs > logs/nusc120_masked.log 2>&1
echo "  exit=$?"
grep -E "\[eval@" logs/nusc120_masked.log
