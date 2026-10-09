#!/usr/bin/env bash
# Decompose the occ->detection transfer benefit into BACKGROUND vs FOREGROUND. Train 3 occ pretexts
# from Occ3D-GT (all / bg-only / fg-only, same 2044 tokens, same recipe) -> det-transfer @2k,4k seed1.
# Tests: does the +32% come from dense background (bg~all, fg~null) as the DynamicOcc negative implied?
# Waits for the masked caches; resumable via the results CSV.
set -e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.bashrc 2>/dev/null || true; conda activate py310 2>/dev/null || true
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
OUT=DeepDataMiningLearning/ngperception/output
CSV=$OUT/label_eff/results_bgfg.csv
PY="python -m DeepDataMiningLearning.ngperception"
MCFG="--backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 --det-head center"
mkdir -p $OUT/label_eff
[ -f $CSV ] || echo "arm,budget,seed,mAP,NDS,ped_AP" > $CSV

# 0) wait for the 3 masked caches (2044 each)
for arm in all bg fg; do
  until [ "$(ls $ROOT/teacher_cache_occ3d_$arm 2>/dev/null | wc -l)" -ge 2044 ]; do sleep 30; done
done
echo "[bgfg] all masked caches ready"

for arm in all bg fg; do
  # 1) pretrain occ student on the masked Occ3D-GT
  ST=$OUT/student_occ3d_$arm
  [ -f $ST/student.pth ] || $PY.gaussian4d.train_student --nusc $NUSC --gts $GTS \
      --teacher-cache $ROOT/teacher_cache_occ3d_$arm \
      --epochs 24 --batch-size 4 --num-workers 8 --amp --out-dir $ST
  # 2) det-transfer @2k,4k seed1
  for budget in 2000 4000; do
    grep -q "^${arm},${budget},1," $CSV && continue
    tag=bgfg_${arm}_b${budget}
    $PY.occupancy.train_det_ablation --nusc $NUSC --gts $GTS --pretrained $ST/student.pth $MCFG \
        --max-samples $budget --val-samples 200 --epochs 12 --batch-size 8 --lr 2e-3 --cosine \
        --num-workers 8 --seed 1 --out-dir $OUT/label_eff/$tag
    $PY.occupancy.eval_det_ablation_official --nusc $NUSC --gts $GTS \
        --ckpt $OUT/label_eff/$tag/det_abl.pth --out-dir $OUT/label_eff/${tag}_eval > $OUT/label_eff/${tag}_eval.log 2>&1 || true
    mAP=$(grep -oE "mAP = [0-9.]+" $OUT/label_eff/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
    NDS=$(grep -oE "NDS = [0-9.]+" $OUT/label_eff/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
    ped=$(grep -E "pedestrian" $OUT/label_eff/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+" | tail -1)
    echo "${arm},${budget},1,${mAP:-NA},${NDS:-NA},${ped:-NA}" >> $CSV
    echo "[bgfg] $tag -> mAP=${mAP} NDS=${NDS} ped=${ped}"
    rm -f $OUT/label_eff/$tag/det_abl.pth
  done
done
echo "[bgfg] DONE -> $CSV"
