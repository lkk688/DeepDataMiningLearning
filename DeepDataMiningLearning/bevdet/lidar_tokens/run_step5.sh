#!/usr/bin/env bash
# Fixing the objective. Baseline (already measured, `lidar` arm, 24 real nuScenes frames,
# 20 epochs, frozen detector queries, scored inside mask_camera):
#     loss 0.2896 -> 0.2159 (-25 %)   fwIoU 0.3119 -> 0.2740 (-0.0378)
#     the only classes that improve are the two largest; every object class collapses.
#
# Two orthogonal candidate causes, so a 2x2 on the `lidar` arm rather than one combined run
# -- the question is WHICH fix matters, not whether the pair happens to work:
#     A  ce         / loss-mask none    <- the baseline above, already run
#     B  ce         / loss-mask camera  <- supervision trustworthiness alone
#     C  ce+lovasz  / loss-mask none    <- class imbalance alone
#     D  ce+lovasz  / loss-mask camera  <- both
# D runs first: if even D cannot stop the object collapse, the objective story is wrong and
# B/C are not worth 80 minutes.
#
# Then `shuffle` with whichever wins -- a fix that also works with the WRONG point cloud has
# fixed the optimiser, not enabled the LiDAR.
set -u
cd "$(dirname "$0")"
PY=/home/010796032/miniconda3/envs/py310/bin/python
export CUDA_HOME=/data/rnd-liu/cudatk
export CPATH=$CUDA_HOME/targets/x86_64-linux/include
export PATH=$(dirname $PY):$PATH
export PYTHONUNBUFFERED=1
PACK=/data/rnd-liu/Datasets/nuScenes/packed_qd_val120
Q=$PACK/lidar_queries.npz
N=24; E=20

run () {   # name arm loss lossmask
  echo "===== $1  (arm=$2 loss=$3 loss-mask=$4)  $(date '+%F %T')"
  $PY train_projector.py --inject $2 --frames $PACK --max-frames $N --epochs $E \
      --eval-every 5 --tokenizer queries --queries $Q --occ-mask camera \
      --loss $3 --loss-mask $4 --exp-name $1 --out-root outputs/obj \
      > logs/obj_$1.log 2>&1
  echo "  exit=$?"
  grep -E "\[eval@|READ" logs/obj_$1.log
}

run D_both_lidar        lidar   ce+lovasz camera
run B_mask_lidar        lidar   ce        camera
run C_lovasz_lidar      lidar   ce+lovasz none
