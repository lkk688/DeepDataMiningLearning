#!/usr/bin/env bash
# 2026-10-18 (user-approved): 3rd seed of the CLEAN label-efficiency arm at 2k / 4k (scratch already has seeds 1-3).
# Identical to step 3 of run_clean_retrain_chain.sh. Waits for the AlpaSim chain (PID arg) so the two never share the GPU.
set -uo pipefail
WAIT_PID=${1:?pid to wait for}
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PYB=/home/010796032/miniconda3/envs/py310/bin/python
ROOT=/data/rnd-liu/Datasets/nuScenes; NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
NG=DeepDataMiningLearning/ngperception/output; OCC=$NG/lss_occ_train; LE=$NG/label_eff; CSV=$LE/results_occ3d_clean.csv
PY="$PYB -m DeepDataMiningLearning.ngperception.occupancy"
MCFG="--backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 --det-head center"
echo "$(date) start clean seed 3"
for budget in 2000 4000; do
  seed=3; tag=occ3dclean_b${budget}_s${seed}
  grep -q "^occ3dclean,${budget},${seed}," $CSV && { echo "skip $tag"; continue; }
  $PY.train_det_ablation --nusc $NUSC --gts $GTS --pretrained $OCC/lss_occ.pth $MCFG \
      --max-samples $budget --val-samples 200 --epochs 12 --batch-size 8 --lr 2e-3 --cosine \
      --num-workers 16 --seed $seed --out-dir $LE/$tag
  $PY.eval_det_ablation_official --nusc $NUSC --gts $GTS \
      --ckpt $LE/$tag/det_abl.pth --out-dir ${LE}/${tag}_eval > ${LE}/${tag}_eval.log 2>&1 || true
  mAP=$(grep -oE "mAP = [0-9.]+" ${LE}/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
  NDS=$(grep -oE "NDS = [0-9.]+" ${LE}/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
  ped=$(grep -E "pedestrian" ${LE}/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+" | tail -1)
  echo "occ3dclean,${budget},${seed},${mAP:-NA},${NDS:-NA},${ped:-NA}" >> $CSV
  echo "[le-clean] $tag -> mAP=${mAP} NDS=${NDS} ped=${ped}"
  rm -f $LE/$tag/det_abl.pth
done
echo "$(date) ALL_DONE"
