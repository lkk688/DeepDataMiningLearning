#!/usr/bin/env bash
# THE control for the objective fix. Config D (ce+lovasz, loss-mask camera) took the `lidar`
# arm from fwIoU 0.3119 -> 0.4333 (+0.1214). That number means nothing until the same config
# is run with the WRONG point cloud:
#   shuffle clearly worse than lidar  -> the fix enabled the LiDAR. The direction is real.
#   shuffle matches lidar             -> the fix repaired the OPTIMISER, not the LiDAR path;
#                                        +0.1214 is free parameters and this is FAIL-2 again.
# Prior from the histogram gate: 82 % of the gain survived a wrong cloud. Expect the worst.
set -u
cd "$(dirname "$0")"
PY=/home/010796032/miniconda3/envs/py310/bin/python
export CUDA_HOME=/data/rnd-liu/cudatk
export CPATH=$CUDA_HOME/targets/x86_64-linux/include
export PATH=$(dirname $PY):$PATH
export PYTHONUNBUFFERED=1
PACK=/data/rnd-liu/Datasets/nuScenes/packed_qd_val120
$PY train_projector.py --inject shuffle --frames $PACK --max-frames 24 --epochs 20 \
    --eval-every 5 --tokenizer queries --queries $PACK/lidar_queries.npz \
    --occ-mask camera --loss ce+lovasz --loss-mask camera \
    --exp-name D_both_shuffle --out-root outputs/obj > logs/obj_D_both_shuffle.log 2>&1
echo "exit=$?"
grep -E "\[eval@|READ" logs/obj_D_both_shuffle.log
