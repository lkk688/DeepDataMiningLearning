#!/usr/bin/env bash
# Detection transfer for the causal chain: does the best occ teacher (voxel-soft) give a good
# occ->detection representation? Finetune (arm C) the camera-only occ encoder into a center-head
# 3D detector, then official nuScenes DetectionEval (center-distance mAP/NDS). fp32 mandatory
# (amp corrupts BN running stats -> eval 0.000).  $1 = student ckpt dir name, default student_soft_voxel10.
set -e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
source ~/.bashrc 2>/dev/null || true; conda activate py310 2>/dev/null || true
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
OUT=DeepDataMiningLearning/ngperception/output
ST=${1:-student_soft_voxel10}
DET=$OUT/det_${ST}
PY="python -m DeepDataMiningLearning.ngperception.occupancy"

echo "[det] transfer from $ST (fp32, center head, finetune)"
$PY.train_det_ablation --nusc $NUSC --gts $GTS \
    --pretrained $OUT/$ST/student.pth \
    --backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 \
    --det-head center --max-samples 28130 --val-samples 400 \
    --epochs 12 --batch-size 8 --lr 2e-3 --cosine --num-workers 8 \
    --out-dir $DET
echo "[det] official eval -> mAP/NDS"
$PY.eval_det_ablation_official --nusc $NUSC --gts $GTS \
    --ckpt $DET/det_abl.pth --out-dir ${DET}_eval 2>&1 | tail -25
echo "[det] DONE $ST"
