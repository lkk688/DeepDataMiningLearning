#!/usr/bin/env bash
# Seed null for the FROM-SCRATCH control of the label-efficiency curve
# (RESEARCH_DIRECTIONS.md §2.2 / §7.1, PUBLICATION_DIRECTION.md §1).
#
# Why this exists. Every det-side seed we have sits on the occ3d-*pretrained* arm
# (results_occ3d.csv: spread 0.0045 @2k, 0.0028 @4k). Pretraining is exactly what
# damps seed variance, so borrowing that sigma for the control is unjustified --
# and the control is the arm the whole claim rests on. `--seed` here also redraws
# the label-budget subset (train_det_ablation.py:53,91 `subset_seed`), which is
# the right nuisance factor: at 2000 of 28130 frames, *which* 2000 you draw is
# plausibly worth more than the init.
#
# It matters most for the rare classes that carry the mechanism result
# (Spearman rho -0.74 @2k): construction_vehicle scores 0.0073 on scratch @2k,
# and an AP that small is where seed jitter would do the most damage to rho.
#
# Identical to run_label_efficiency.sh in every hyperparameter -- same MCFG, same
# 12 epochs / bs 8 / lr 2e-3 / cosine / val-samples 200 -- so the only new
# variable is --seed. Appends to the SAME results.csv and skips rows already in
# it, so it is safe to re-run.

set -uo pipefail
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/010796032/miniconda3/envs/py310/bin/python
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
OUT=DeepDataMiningLearning/ngperception/output/label_eff
CSV=$OUT/results.csv
MOD=DeepDataMiningLearning.ngperception.occupancy
MCFG="--backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 --det-head center"

# fp32 only -- --amp collapses this trainer (fp16 BN running-stat corruption -> eval 0.000)
BUDGETS="${BUDGETS:-2000 4000}"
SEEDS="${SEEDS:-2 3}"

mkdir -p $OUT
[ -f $CSV ] || echo "arm,budget,seed,mAP,NDS,ped_AP" > $CSV

for budget in $BUDGETS; do
  for seed in $SEEDS; do
    tag=scratch_b${budget}_s${seed}
    if grep -q "^scratch,${budget},${seed}," $CSV 2>/dev/null; then
      echo "[ss] skip $tag (already in $CSV)"; continue
    fi
    echo "===== [ss] train $tag  $(date +%H:%M:%S) ====="
    $PY -m $MOD.train_det_ablation --nusc $NUSC --gts $GTS $MCFG \
        --max-samples $budget --val-samples 200 --epochs 12 --batch-size 8 \
        --lr 2e-3 --cosine --num-workers 8 --seed $seed \
        --out-dir $OUT/$tag > $OUT/${tag}_train.log 2>&1
    if [ ! -f "$OUT/$tag/det_abl.pth" ]; then
      echo "[ss] TRAIN FAILED $tag -- see $OUT/${tag}_train.log"; continue
    fi
    echo "===== [ss] eval  $tag  $(date +%H:%M:%S) ====="
    $PY -m $MOD.eval_det_ablation_official --nusc $NUSC --gts $GTS \
        --ckpt $OUT/$tag/det_abl.pth --out-dir ${OUT}/${tag}_eval \
        > ${OUT}/${tag}_eval.log 2>&1 || true
    mAP=$(grep -oE "mAP = [0-9.]+" ${OUT}/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
    NDS=$(grep -oE "NDS = [0-9.]+" ${OUT}/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
    ped=$(grep -E "pedestrian" ${OUT}/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+" | tail -1)
    echo "scratch,${budget},${seed},${mAP:-NA},${NDS:-NA},${ped:-NA}" >> $CSV
    echo "[ss] $tag -> mAP=${mAP:-NA} NDS=${NDS:-NA} ped=${ped:-NA}"
    rm -f $OUT/$tag/det_abl.pth      # free disk; the eval log keeps the per-class table
  done
done

echo "=== SCRATCH SEED NULL DONE $(date) ==="
column -s, -t $CSV
