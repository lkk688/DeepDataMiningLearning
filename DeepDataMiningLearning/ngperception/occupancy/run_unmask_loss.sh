#!/usr/bin/env bash
# RESEARCH_DIRECTIONS.md 4.6, experiment 2: does supervising the occluded region help?
#
# Occ3D's protocol computes both the loss and the metric on camera-visible voxels only
# (train_lss.py occ_loss, evaluator.py use_camera_mask). Measured over 120 frames that
# discards 37% of labelled occupied voxels, disproportionately vehicles (car 1.04x,
# truck 1.23x, trailer 1.58x more voxels outside the mask than inside) while the kept
# region is dominated by near-field flat ground. Those labels already exist -- Occ3D's
# GT is accumulated over the sequence -- so this costs no annotation.
#
# Two arms, identical apart from --occ-mask:
#   camera : the standard protocol (control)
#   all    : loss over every labelled voxel
#
# Both are evaluated stratified (train_lss.evaluate):
#   mIoU      camera-visible voxels  -- the standard number, comparable to §2.1
#   pred_mIoU voxels OUTSIDE mask_camera -- the region the protocol discards
#   all_mIoU  every labelled voxel
#
# WHAT WOULD MAKE THIS INTERESTING, decided before running:
#   pred_mIoU rises a lot AND mIoU does not fall  -> the discarded region is learnable
#     for free; the protocol was simply leaving it on the table.
#   pred_mIoU rises AND mIoU falls                -> a real trade-off; report the curve,
#     do not claim a free lunch.
#   pred_mIoU stays ~0                            -> the occluded region is not learnable
#     from a single frame at this scale; the direction needs temporal input (§4.5) or
#     dies. This is the kill criterion.
#
# Hyperparameters match run_backbone_bench.sh / run_backbone_seeds.sh exactly (2044
# frames, 24 epochs, bs 2, lr 2e-3, 4x96 decoder, refine-iters 1, dinov2_large, seed 1),
# so the `camera` arm should land near the §1.1 seed distribution (mean .2968, sd .014)
# -- if it does not, stop and find out why before reading the `all` arm.

set -uo pipefail
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/010796032/miniconda3/envs/py310/bin/python
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
OUT=DeepDataMiningLearning/ngperception/output/unmask_loss
CSV=$OUT/results.csv
MOD=DeepDataMiningLearning.ngperception.occupancy

ARMS="${ARMS:-camera all}"
SEED="${SEED:-1}"
MAXJOBS="${MAXJOBS:-2}"

mkdir -p $OUT
[ -f $CSV ] || echo "occ_mask,seed,mIoU,pred_mIoU,pred_geo,all_mIoU,geo_IoU" > $CSV

run_one () {
  local arm=$1
  local tag=${arm}_s${SEED}
  local LG=$OUT/${tag}.log
  if grep -q "^${arm},${SEED}," $CSV 2>/dev/null; then echo "[um] skip $tag"; return; fi
  echo "[um] start $tag $(date +%H:%M:%S)"
  $PY -m $MOD.train_lss \
      --nusc $NUSC --gts $GTS --max-samples 2044 --val-samples 300 \
      --epochs 24 --batch-size 2 --lr 2e-3 --backbone dinov2_large \
      --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 \
      --occ-mask $arm --seed $SEED --out-dir $OUT/$tag > $LG 2>&1
  # last epoch line: "... val mIoU=X geo_IoU=Y | pred_mIoU=Z pred_geo=W all_mIoU=V"
  local last=$(grep -oE "val mIoU=[0-9.]+ geo_IoU=[0-9.]+ \| pred_mIoU=[0-9.na]+ pred_geo=[0-9.na]+ all_mIoU=[0-9.na]+" $LG | tail -1)
  local g=($(echo "$last" | grep -oE "[0-9.]+" ))
  echo "${arm},${SEED},${g[0]:-NA},${g[2]:-NA},${g[3]:-NA},${g[4]:-NA},${g[1]:-NA}" >> $CSV
  echo "[um] done  $tag -> $last"
}

for arm in $ARMS; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 60; done
  run_one $arm &
  sleep 90                       # stagger the NuScenes index load
done
wait
echo "=== UNMASK-LOSS DONE $(date) ==="
column -s, -t $CSV
