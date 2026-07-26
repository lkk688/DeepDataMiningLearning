#!/usr/bin/env bash
# Ego-status ablation for open-loop planning (BEV-Planner / AD-MLP protocol): isolate the OCCUPANCY
# contribution against the ego-status shortcut. Three arms at the winner backbone (+ occ_ego base):
#   ego      : ego-history + command only        (the shortcut baseline; backbone-agnostic)
#   occ_ego  : occupancy + ego-history + command (full)
# occ-only is already in planning.txt (~5.04 @3s). delta(occ_ego - ego) = the real perception gain.
set +e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.bashrc 2>/dev/null || true; conda activate py310 2>/dev/null || true
ROOT=/data/rnd-liu/Datasets/nuScenes; NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
BB=DeepDataMiningLearning/ngperception/output/backbone_bench
CKL=$BB/occ_dinov2_large/lss_occ.pth; CKB=$BB/occ_dinov2_base/lss_occ.pth
PY="python -m DeepDataMiningLearning.ngperception.occupancy.train_planning"
# batch 8 + cosine + 20 epochs + ego-input normalization (÷30 in head): the ego-only arm is ~linear in
# ego-history (constant-velocity ~= GT future), so it MUST fit with stable optimization; batch-4/no-sched
# oscillated (train loss 4-7). Occ arms use a FROZEN backbone (no_grad forward) so batch 8 is memory-safe.
# 8 epochs suffice: the CV prior makes ego/occ_ego converge by epoch 0-1 (ego_only hit the 2.09 CV floor
# immediately). ego_only already logged separately (2.09 @3s); here we run only the occ_ego arms, which
# need the camera images anyway. delta(occ_ego - ego_only 2.09) = occupancy's contribution over the prior.
COMMON="--decoder-layers 4 --decoder-hidden 96 --refine-iters 1 --head transformer --lr 2e-3 --cosine \
        --max-samples 4000 --val-samples 400 --epochs 8 --batch-size 8 --num-workers 8"
: > $BB/planning_ego.txt
echo "ego_only (ego) -> [prior] L2@3s=2.09 (== constant-velocity floor, converged epoch0)" >> $BB/planning_ego.txt

run () { # name  backbone  ckpt  mode
  local name=$1 bb=$2 ck=$3 mode=$4
  $PY --nusc $NUSC --gts $GTS --pretrained $ck --backbone $bb --mode $mode $COMMON \
      --out-dir $BB/plan_${name} > $BB/plan_${name}.log 2>&1
  local res=$(grep -E "\[plan\] epoch" $BB/plan_${name}.log | tail -1)
  echo "${name} (${mode}) -> ${res}" | tee -a $BB/planning_ego.txt
  echo "[plan-ego] ${name} -> ${res}"
}

run occ_ego_large   dinov2_large $CKL occ_ego
run occ_ego_base    dinov2_base  $CKB occ_ego
echo "[plan-ego] DONE"
