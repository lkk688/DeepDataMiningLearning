#!/usr/bin/env bash
# P1 -- seed null for the frozen-FM backbone table (RESEARCH_DIRECTIONS.md 4.1).
#
# The published table is ONE training run per backbone, unseeded
# (run_backbone_bench.sh passes no --seed).  Its load-bearing claim is the
# occ/det decoupling:
#
#     DINOv3-L occ 0.292 < DINOv2-L 0.316   but   det 0.1248 > 0.1140
#
# i.e. an occ gap of 0.024 and a det gap of 0.011 between two single runs.
# Gaps this small can be pure seed noise; this sweep measures the seed-only
# spread so the table can be read against it.
#
# Identical to run_backbone_bench.sh in every hyperparameter -- 2044 frames,
# 24 epochs, bs 2, lr 2e-3, 4x96 decoder, refine-iters 1 -- so the only new
# variable is --seed.  train_lss.py already seeds torch/cuda/numpy/random and
# the DataLoader generator (train_lss.py:262-266,295).
set -u
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/010796032/miniconda3/envs/py310/bin/python
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
OUT=DeepDataMiningLearning/ngperception/output/backbone_seeds
mkdir -p $OUT
CSV=$OUT/occ_seeds.csv
[ -f $CSV ] || echo "backbone,seed,occ_mIoU,geo_IoU" > $CSV

BACKBONES="${BACKBONES:-dinov2_large dinov3}"
SEEDS="${SEEDS:-1 2 3 4}"
MAXJOBS="${MAXJOBS:-3}"

run_one () {
  local bb=$1 s=$2
  local W=$OUT/occ_${bb}_s${s}
  local LG=$OUT/occ_${bb}_s${s}.log
  if grep -q "^${bb},${s}," $CSV 2>/dev/null; then echo "[skip] $bb seed $s"; return; fi
  echo "[start] $bb seed $s $(date +%H:%M:%S)"
  $PY -m DeepDataMiningLearning.ngperception.occupancy.train_lss \
      --nusc $NUSC --gts $GTS --max-samples 2044 --val-samples 300 \
      --epochs 24 --batch-size 2 --lr 2e-3 --backbone $bb \
      --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 \
      --seed $s --out-dir $W > $LG 2>&1
  local miou geo
  miou=$(grep -oE "val mIoU=[0-9.]+" $LG | tail -1 | grep -oE "[0-9.]+")
  geo=$(grep -oE "geo_IoU=[0-9.]+" $LG | tail -1 | grep -oE "[0-9.]+")
  echo "${bb},${s},${miou:-NA},${geo:-NA}" >> $CSV
  echo "[done ] $bb seed $s -> mIoU=${miou:-NA} $(date +%H:%M:%S)"
}

for bb in $BACKBONES; do
  for s in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 60; done
    run_one $bb $s &
    sleep 90            # stagger: avoid N processes hitting NuScenes index load at once
  done
done
wait
echo "=== P1 BACKBONE SEED SWEEP DONE $(date) ==="
column -s, -t $CSV
