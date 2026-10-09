#!/usr/bin/env bash
# Post-fix re-measurement + the full 4-arm gate. See QWEN_DRIVE_LIDAR_TOKENS.md §4.
set -u
cd "$(dirname "$0")"
PY=/home/010796032/miniconda3/envs/py310/bin/python
# one export per line: with `set -u`, $CUDA_HOME inside the same export statement that
# defines it is still unbound.
export CUDA_HOME=/data/rnd-liu/cudatk
export CPATH=$CUDA_HOME/targets/x86_64-linux/include
export PATH=$(dirname $PY):$PATH
export PYTHONUNBUFFERED=1
mkdir -p logs outputs

echo "===== step 1: re-measure the training step with the patch-embed fix"
$PY train_projector.py --inject lidar --max-frames 2 --epochs 3 --eval-every 3 \
    --exp-name steptime --out-root outputs > logs/steptime.log 2>&1
echo "  exit=$?"
grep -E "patch embed|\[eval@|^epoch|\[mem\]" logs/steptime.log

echo
echo "===== step 2: the 4-arm gate, 6 frames, 40 epochs"
EPOCHS=40 ./run_projector_gate.sh
