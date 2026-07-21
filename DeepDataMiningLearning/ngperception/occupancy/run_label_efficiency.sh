#!/usr/bin/env bash
# Label-efficiency figure (the paper's main positive): official nuScenes mAP/NDS vs detection label
# budget, LABEL-FREE occ-pretrained (voxel-soft, mIoU .104) vs from-scratch, 3 seeds for error bars.
# Camera-only, center head, fp32 (amp corrupts BN -> eval 0.000). Tests whether a BETTER label-free
# occ teacher yields a real detection label-efficiency gain (the old render2d pretext barely did).
# Resumable: each (arm,budget,seed) appends one row to results.csv; existing rows are skipped.
set -e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
source ~/.bashrc 2>/dev/null || true; conda activate py310 2>/dev/null || true
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
OUT=DeepDataMiningLearning/ngperception/output/label_eff
PRETRAIN=DeepDataMiningLearning/ngperception/output/student_soft_voxel10/student.pth
CSV=$OUT/results.csv
PY="python -m DeepDataMiningLearning.ngperception.occupancy"
MCFG="--backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 --det-head center"
mkdir -p $OUT
[ -f $CSV ] || echo "arm,budget,seed,mAP,NDS,ped_AP" > $CSV

for seed in 1 2 3; do
  for budget in 2000 4000 8000; do
    for arm in scratch voxelsoft; do
      tag=${arm}_b${budget}_s${seed}
      grep -q "^${arm},${budget},${seed}," $CSV && { echo "[le] skip $tag (done)"; continue; }
      pre=""; [ "$arm" = "voxelsoft" ] && pre="--pretrained $PRETRAIN"
      echo "===== [le] train $tag ====="
      $PY.train_det_ablation --nusc $NUSC --gts $GTS $pre $MCFG \
          --max-samples $budget --val-samples 200 --epochs 12 --batch-size 8 --lr 2e-3 --cosine \
          --num-workers 8 --seed $seed --out-dir $OUT/$tag
      echo "===== [le] eval $tag ====="
      $PY.eval_det_ablation_official --nusc $NUSC --gts $GTS \
          --ckpt $OUT/$tag/det_abl.pth --out-dir ${OUT}/${tag}_eval > ${OUT}/${tag}_eval.log 2>&1 || true
      mAP=$(grep -oE "mAP = [0-9.]+" ${OUT}/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
      NDS=$(grep -oE "NDS = [0-9.]+" ${OUT}/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
      ped=$(grep -E "pedestrian" ${OUT}/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+" | tail -1)
      echo "${arm},${budget},${seed},${mAP:-NA},${NDS:-NA},${ped:-NA}" >> $CSV
      echo "[le] $tag -> mAP=${mAP} NDS=${NDS} ped=${ped}"
      rm -f $OUT/$tag/det_abl.pth                       # free disk; keep eval logs + csv
    done
  done
done
echo "[le] DONE -> $CSV"
