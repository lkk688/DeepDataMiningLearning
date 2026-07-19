#!/usr/bin/env bash
# Fair {voxel10, anisotropic-gaussian} x {hard, soft} 2x2 for the soft-semantics arm (#2b).
# Hard arms already trained (student_ra_voxel10, student_ra2_gaussian) on the 2044-token set.
# This builds the 2 SOFT teacher caches on the SAME tokens, trains 2 soft students with the
# IDENTICAL recipe (epochs 24, batch 4, --amp), then evals all 4 on Occ3D val.
set -e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
source ~/.bashrc 2>/dev/null || true
conda activate py310 2>/dev/null || true

ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval
GTS=$ROOT/v1.0-trainval/gts
LG=$ROOT/labelgen_cache
SOFT=$ROOT/labelgen_soft_cache
TC2=$ROOT/teacher_cache_2x2soft
OUT=DeepDataMiningLearning/ngperception/output
PY="python -m DeepDataMiningLearning.ngperception.gaussian4d"

# 0) wait for the soft-label cache to finish
echo "[2x2] waiting for soft-label cache (PID 3874344) ..."
while kill -0 3874344 2>/dev/null; do sleep 60; done
echo "[2x2] soft cache complete: $(ls $SOFT | wc -l) tokens"

# token-set sanity vs the hard caches (should be identical 2044)
echo "[2x2] soft=$(ls $SOFT|wc -l)  voxel10-hard=$(ls $ROOT/teacher_cache_ra/voxel10|wc -l)  gauss-hard=$(ls $ROOT/teacher_cache_ra2/gaussian|wc -l)"
comm -12 <(ls $SOFT|sort) <(ls $ROOT/teacher_cache_ra2/gaussian|sort) | wc -l | xargs echo "[2x2] soft∩gauss-hard ="

# 1) build the 2 soft teacher caches on the soft tokens
$PY.build_teacher --nusc $NUSC --gts $GTS --labelgen-cache $LG --soft-cache $SOFT \
    --teacher voxel10  --out-dir $TC2/voxel10  --n 2200
$PY.build_teacher --nusc $NUSC --gts $GTS --labelgen-cache $LG --soft-cache $SOFT \
    --teacher gaussian --out-dir $TC2/gaussian --n 2200
echo "[2x2] soft caches: voxel10=$(ls $TC2/voxel10|wc -l)  gaussian=$(ls $TC2/gaussian|wc -l)"

# 2) train the 2 soft students (identical recipe to the hard arms)
$PY.train_student --nusc $NUSC --gts $GTS --teacher-cache $TC2/voxel10 \
    --epochs 24 --batch-size 4 --num-workers 8 --amp --out-dir $OUT/student_soft_voxel10
$PY.train_student --nusc $NUSC --gts $GTS --teacher-cache $TC2/gaussian \
    --epochs 24 --batch-size 4 --num-workers 8 --amp --out-dir $OUT/student_soft_gaussian

# 3) eval all 4 arms on Occ3D val
echo "======================= 2x2 EVAL ======================="
for name in student_ra_voxel10 student_ra2_gaussian student_soft_voxel10 student_soft_gaussian; do
  echo "----- $name -----"
  $PY.eval_student --nusc $NUSC --gts $GTS --ckpt $OUT/$name/student.pth --num-workers 8 2>&1 | tail -12
done
echo "[2x2] DONE"
