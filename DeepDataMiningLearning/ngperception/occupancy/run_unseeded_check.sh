#!/usr/bin/env bash
# Task 2 -- is the published 0.316 reproducible without a seed?
#
# The four seeded DINOv2-L reruns gave 0.278-0.312; the published unseeded run gave
# 0.316, above all of them.  DINOv3-L shows the same pattern (0.292 vs a 0.267-0.282
# range).  Two originals each landing above their own 4-sample range is p~0.04 if
# the runs are exchangeable, so either they were lucky draws or the unseeded path
# differs from the seeded one.  This reruns DINOv2-L with NO --seed, everything
# else identical, to tell those apart.
set -u
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/010796032/miniconda3/envs/py310/bin/python
ROOT=/data/rnd-liu/Datasets/nuScenes
OUT=DeepDataMiningLearning/ngperception/output/backbone_seeds
LG=$OUT/occ_dinov2_large_unseeded.log
echo "[unseeded] start $(date)"
$PY -m DeepDataMiningLearning.ngperception.occupancy.train_lss \
    --nusc $ROOT/v1.0-trainval --gts $ROOT/v1.0-trainval/gts \
    --max-samples 2044 --val-samples 300 --epochs 24 --batch-size 2 --lr 2e-3 \
    --backbone dinov2_large --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 \
    --out-dir $OUT/occ_dinov2_large_unseeded > $LG 2>&1
echo "[unseeded] final=$(grep -oE 'val mIoU=[0-9.]+' $LG | tail -1)  max=$(grep -oE 'val mIoU=[0-9.]+' $LG | grep -oE '[0-9.]+' | sort -g | tail -1)  $(date)"
