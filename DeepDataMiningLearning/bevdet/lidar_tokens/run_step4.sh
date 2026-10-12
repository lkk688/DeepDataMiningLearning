#!/usr/bin/env bash
# The 4-arm gate again, but on REAL nuScenes frames, with the FROZEN LiDAR-ONLY DETECTOR's
# queries as tokens, scored inside mask_camera. This is the version whose negative result is
# citable: rung 0 says the tokeniser does not change the channel (0.142% vs 0.157% argmax
# bandwidth), so the prediction is that FAIL-2 repeats -- and a prediction stated before the
# run is worth more than the same number afterwards.
#   PASS  : lidar beats zero on fwIoU AND beats shuffle clearly
#   FAIL-2: shuffle matches lidar  -> the projector is free parameters, not a LiDAR reading
set -u
cd "$(dirname "$0")"
PY=/home/010796032/miniconda3/envs/py310/bin/python
export CUDA_HOME=/data/rnd-liu/cudatk
export CPATH=$CUDA_HOME/targets/x86_64-linux/include
export PATH=$(dirname $PY):$PATH
export PYTHONUNBUFFERED=1
PACK=/data/rnd-liu/Datasets/nuScenes/packed_qd_val120
Q=$PACK/lidar_queries.npz
N=${N:-24}          # frames; 24 x 20 epochs x ~6 s = ~50 min per trained arm
E=${E:-20}

for ARM in zero lidar shuffle; do
  echo "===== $ARM  $(date '+%F %T')"
  $PY train_projector.py --inject $ARM --frames $PACK --max-frames $N --epochs $E \
      --eval-every 5 --tokenizer queries --queries $Q --occ-mask camera \
      --exp-name q24 --out-root outputs/q24 > logs/q24_$ARM.log 2>&1
  echo "  exit=$?"
  grep -E "\[eval@|READ" logs/q24_$ARM.log
done
