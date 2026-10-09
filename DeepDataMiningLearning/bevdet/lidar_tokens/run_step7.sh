#!/usr/bin/env bash
# The gain under the fixed objective (config D) is +0.1215 fwIoU for `lidar` and +0.1475 for
# `shuffle` -- the WRONG point cloud wins on every class. So the remaining question is not
# "how much does LiDAR add" but "does the token CONTENT matter at all".
#
#   const   : the projector's input is the same vector for EVERY frame -> zero information.
#             If it matches lidar/shuffle, the whole effect is a learned constant prefix
#             recalibrating the frozen head, and no injected modality is doing anything.
#   seed 1  : lidar and shuffle again. One seed cannot support "shuffle > lidar" (§7.1);
#             FAIL-2 holds either way, but the ordering needs a noise floor.
#   holdout : 24 frames never trained on, evaluated at every checkpoint. A global
#             recalibration should transfer; a memorised per-frame fix should not.
set -u
cd "$(dirname "$0")"
PY=/home/010796032/miniconda3/envs/py310/bin/python
export CUDA_HOME=/data/rnd-liu/cudatk
export CPATH=$CUDA_HOME/targets/x86_64-linux/include
export PATH=$(dirname $PY):$PATH
export PYTHONUNBUFFERED=1
PACK=/data/rnd-liu/Datasets/nuScenes/packed_qd_val120

run () {  # name arm seed
  echo "===== $1 (arm=$2 seed=$3)  $(date '+%F %T')"
  $PY train_projector.py --inject $2 --seed $3 --frames $PACK --max-frames 24 --holdout 24 \
      --epochs 20 --eval-every 5 --tokenizer queries --queries $PACK/lidar_queries.npz \
      --occ-mask camera --loss ce+lovasz --loss-mask camera \
      --exp-name $1 --out-root outputs/obj > logs/obj_$1.log 2>&1
  echo "  exit=$?"; grep -E "\[eval@20|\[hold@" logs/obj_$1.log | tail -6
}

run E_const_s0    const   0
run D_lidar_s1    lidar   1
run D_shuffle_s1  shuffle 1
