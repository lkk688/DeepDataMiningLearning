#!/usr/bin/env bash
# Is the occluded-space ceiling a property of single-frame INFORMATION, or of VISION-ONLY
# FEATURES?  RESEARCH_DIRECTIONS.md §4.6 established pred_mIoU ~ 0.070 by varying the LOSS
# (occluded region at 84 % vs 50 % -> 0.069 / 0.070). Neither arm varied the feature prior.
#
# Identical to run_unmask_loss.sh's `balanced` arm in every respect except the backbone:
#   A  dinov2_large   measured: s1 pred_mIoU .070 pred_geo .070 | s2 .067 / .063
#   B  qwendrive      frozen driving-VLM image tap, fed from cache
#
# THE BAR, from the two-seed spread measured 2026-09-11:
#   pred_geo   spread 0.007  (mean 0.0665)   <- primary endpoint; needs ~>= 0.08 to matter
#   pred_mIoU  spread 0.003  (mean 0.0685)
#   in-mask    spread 0.014  (mean 0.199)    <- reproduces the known sigma ~ 0.014
#   geo_IoU    spread 0.045                  <- too noisy to lean on
# Two seeds give a spread, not a sigma. Read nothing that does not clear ~2x the spread.
#
# PRE-REGISTERED: pred_mIoU should NOT move (Occ3D's occluded labels come from other
# timestamps). pred_geo MAY move (a language-grounded prior encodes typical structure).
# If neither clears the bar, the information-ceiling conclusion is confirmed on a second,
# independent axis -- a negative, and it must be called one.
set -uo pipefail
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/010796032/miniconda3/envs/py310/bin/python
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
CACHE=$ROOT/vlm_cache_occ2052
OUT=DeepDataMiningLearning/ngperception/output/vlm_backbone
CSV=$OUT/results.csv
MOD=DeepDataMiningLearning.ngperception.occupancy
mkdir -p $OUT
[ -f $CSV ] || echo "backbone,occ_mask,seed,mIoU,pred_mIoU,pred_geo,all_mIoU,geo_IoU" > $CSV

# the cache writes meta.json only after the last frame
while [ ! -f "$CACHE/meta.json" ]; do sleep 60; done
echo "cache complete ($(ls $CACHE/*.npy | wc -l) frames) at $(date '+%F %T')"

UPS="${UPS:-1}"
for SEED in 1 2; do
  tag=qwendrive_balanced_s${SEED}
  [ "$UPS" != 1 ] && tag=qwendrive_u${UPS}_balanced_s${SEED}
  [ -f "$OUT/${tag}/lss_occ.pth" ] && { echo "skip $tag (done)"; continue; }
  echo "[vlm] start $tag $(date +%H:%M:%S)"
  $PY -m $MOD.train_lss \
      --nusc $NUSC --gts $GTS --max-samples 2044 --val-samples 300 \
      --epochs 24 --batch-size 2 --lr 2e-3 --backbone qwendrive --vlm-feat-cache $CACHE \
      --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 --feat-upsample $UPS \
      --occ-mask balanced --seed $SEED --out-dir $OUT/$tag > $OUT/${tag}.log 2>&1
  echo "  exit=$? -> $OUT/${tag}.log"
  tail -3 $OUT/${tag}.log
done
