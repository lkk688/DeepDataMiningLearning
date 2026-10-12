#!/usr/bin/env bash
# Task 1 -- error bars on the DET half of the occ/det decoupling claim.
#
# RESEARCH_DIRECTIONS.md 4.1 asserts DINOv3-L transfers better to detection than DINOv2-L
# (0.1248 vs 0.1140) despite worse occupancy.  Both numbers are one det run on one
# unseeded occ checkpoint.  The occ seed sweep measured the upstream variance and
# it is large (DINOv2-L sd 0.014, range 0.034), so the det arm inherits it.
#
# Design: hold the DET seed fixed at 1 -- exactly what the published runs used --
# and vary the OCC PRETRAIN checkpoint across the four seeds already trained.
# That propagates the variance we measured into the number the claim is about,
# and isolates it from det-training noise, which is a separate (smaller) source.
# Everything else is byte-identical to run_backbone_det.sh.
set -u
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/010796032/miniconda3/envs/py310/bin/python
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
SEEDDIR=DeepDataMiningLearning/ngperception/output/backbone_seeds
OUT=$SEEDDIR/det
CSV=$OUT/det_seeds.csv
DCFG="--decoder-layers 4 --decoder-hidden 96 --refine-iters 1 --det-head center"
mkdir -p $OUT
[ -f $CSV ] || echo "backbone,occ_seed,det_seed,mAP,NDS,ped_AP" > $CSV

for bb in dinov2_large dinov3; do
  for s in 1 2 3 4; do
    grep -q "^${bb},${s}," $CSV && { echo "[skip] $bb occ_seed $s"; continue; }
    ST=$SEEDDIR/occ_${bb}_s${s}/lss_occ.pth
    [ -f "$ST" ] || { echo "[miss] $ST"; continue; }
    tag=det_${bb}_s${s}
    echo "===== [det] $bb occ_seed=$s  $(date +%H:%M:%S)"
    $PY -m DeepDataMiningLearning.ngperception.occupancy.train_det_ablation \
        --nusc $NUSC --gts $GTS --pretrained $ST --backbone $bb $DCFG \
        --max-samples 2000 --val-samples 200 --epochs 12 --batch-size 8 --lr 2e-3 \
        --cosine --num-workers 6 --seed 1 --out-dir $OUT/$tag > $OUT/$tag.log 2>&1
    $PY -m DeepDataMiningLearning.ngperception.occupancy.eval_det_ablation_official \
        --nusc $NUSC --gts $GTS --ckpt $OUT/$tag/det_abl.pth \
        --out-dir $OUT/${tag}_eval > $OUT/${tag}_eval.log 2>&1
    mAP=$(grep -oE "mAP = [0-9.]+" $OUT/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
    NDS=$(grep -oE "NDS = [0-9.]+" $OUT/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
    ped=$(grep -E "pedestrian" $OUT/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+" | tail -1)
    echo "${bb},${s},1,${mAP:-NA},${NDS:-NA},${ped:-NA}" >> $CSV
    echo "[det] $bb s$s -> mAP=${mAP:-NA} NDS=${NDS:-NA}"
    rm -f $OUT/$tag/det_abl.pth
  done
done
echo "=== DET SEED SWEEP DONE $(date) ==="; column -s, -t $CSV
