#!/usr/bin/env bash
# Leakage-free re-run of the label-efficiency positive (docs/RESULTS_FULLDATA_H100.md "Clean-retrain plan";
# lab notes). The occ3d arm of output/label_eff/results_occ3d.csv was pretrained from
# output/lss_occ_full, which trained on ALL 34149 trainval frames (incl. the 150 official val scenes) -> its
# detection transfer is evaluated on val images the encoder has seen.
#   1) retrain the SAME camera-only occupancy recipe (Run 1) on the 700 official train scenes only (--split train),
#      internal val = official val scenes; train-only class weights;
#   2) official-val occupancy eval of the clean checkpoint (quantifies the leakage: compare vs 0.328 leaked);
#   3) the occ3d label-efficiency arm again from the clean checkpoint (same MCFG / budgets / seeds as
#      run_le_occ3d.sh) into results_occ3d_clean.csv. The scratch arm does not depend on pretraining -> reused.
set -uo pipefail
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 2026-10-16: each dataloader worker otherwise spawns a full OpenMP/BLAS pool (8 workers x ~11 cores, 62% sys time
# on a 96-core node shared with AlpaSim). Cap threads per process; more workers instead. Numerics unchanged.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PYB=/home/010796032/miniconda3/envs/py310/bin/python
ROOT=/data/rnd-liu/Datasets/nuScenes; NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
NG=DeepDataMiningLearning/ngperception/output
OCC=$NG/lss_occ_train; LE=$NG/label_eff; CSV=$LE/results_occ3d_clean.csv
PY="$PYB -m DeepDataMiningLearning.ngperception.occupancy"
echo "$(date) clean-retrain chain on $(hostname)"

# 1) occupancy pretraining, train scenes only (resumable: re-launching this script continues the run)
if [ ! -f $OCC/lss_occ.pth ]; then
  $PY.train_lss --gts $GTS --nusc $NUSC \
    --backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 \
    --cosine --amp --refine-iters 2 --occ-lovasz 1.0 --occ-class-balance \
    --occ-cb-cache $NG/classw_train.npy \
    --max-samples 34149 --val-samples 300 --epochs 24 --batch-size 8 \
    --lr 2e-3 --depth-weight 1.0 --num-workers 16 --seed 0 \
    --split train --resume --out-dir $OCC || { echo "OCC TRAIN FAILED"; exit 1; }
fi
echo "$(date) BATCH_DONE occ_train"

# 2) official val (150 scenes / 6019 frames)
[ -f $OCC/eval_val.log ] || $PY.eval_val --gts $GTS --nusc $NUSC --ckpt $OCC/lss_occ.pth \
    --backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --refine-iters 2 > $OCC/eval_val.log 2>&1
grep -iE "miou|geo" $OCC/eval_val.log | tail -3
echo "$(date) BATCH_DONE occ_eval"

# 3) label-efficiency occ3d arm from the CLEAN encoder (identical to run_le_occ3d.sh except --pretrained / CSV)
MCFG="--backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 --det-head center"
[ -f $CSV ] || echo "arm,budget,seed,mAP,NDS,ped_AP" > $CSV
for seed in 1 2; do
  for budget in 2000 4000 8000; do
    tag=occ3dclean_b${budget}_s${seed}
    grep -q "^occ3dclean,${budget},${seed}," $CSV && { echo "[le-clean] skip $tag"; continue; }
    echo "===== [le-clean] train $tag $(date) ====="
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
done
echo "$(date) BATCH_DONE le_clean"; cat $CSV
echo ALL_DONE
