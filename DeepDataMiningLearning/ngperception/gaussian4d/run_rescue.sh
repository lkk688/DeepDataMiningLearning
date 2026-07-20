#!/usr/bin/env bash
# Gaussian RESCUE: retrain voxel-soft and gaussian-soft with the FACTORIZED loss (geometry and
# semantics separated; semantic uncertainty can't leak into occupancy). Same soft teacher caches as
# the 2x2 (no re-cache needed — they already separate geom `weight` from `soft_idx/soft_prob`).
# Sequential to stay within GPU memory alongside the concurrent det-transfer run.
#
# Stopping criterion (decisive): gaussian-soft-fac must EXCEED voxel-soft (mIoU>0.1040), geo >=
# voxel-soft-fac, tail not degraded. Else Gaussian doesn't earn its complexity -> stop 4D/VGGT.
set -e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
source ~/.bashrc 2>/dev/null || true; conda activate py310 2>/dev/null || true
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
TC2=$ROOT/teacher_cache_2x2soft
OUT=DeepDataMiningLearning/ngperception/output
PY="python -m DeepDataMiningLearning.ngperception.gaussian4d"

for arm in voxel10 gaussian; do
  $PY.train_student --nusc $NUSC --gts $GTS --teacher-cache $TC2/$arm \
      --epochs 24 --batch-size 4 --num-workers 8 --amp --factorized \
      --out-dir $OUT/student_fac_$arm
done

echo "==================== RESCUE EVAL (factorized) ===================="
for arm in voxel10 gaussian; do
  echo "----- student_fac_$arm -----"
  $PY.eval_student --nusc $NUSC --gts $GTS --ckpt $OUT/student_fac_$arm/student.pth --num-workers 8 2>&1 | tail -6
done
echo "[rescue] DONE"
