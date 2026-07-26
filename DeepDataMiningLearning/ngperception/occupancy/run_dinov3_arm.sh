#!/usr/bin/env bash
# Standalone DINOv3 arm of the backbone benchmark (the only backbone still missing from occ/det CSVs).
# Same frozen-probe protocol as run_backbone_bench.sh + run_backbone_det.sh: occ @2044 -> det @2k.
# Launched in parallel with the fusion column (DINOv3 frozen probe ~15GB, plenty of headroom).
set +e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.bashrc 2>/dev/null || true; conda activate py310 2>/dev/null || true
ROOT=/data/rnd-liu/Datasets/nuScenes; NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
OUT=DeepDataMiningLearning/ngperception/output/backbone_bench
OCSV=$OUT/occ_results.csv; DCSV=$OUT/det_results.csv
PY="python -m DeepDataMiningLearning.ngperception.occupancy"
CFG="--backbone dinov3 --decoder-layers 4 --decoder-hidden 96 --refine-iters 1"
mkdir -p $OUT

# 1) occ frozen probe @2044
ST=$OUT/occ_dinov3
if ! grep -q "^dinov3," $OCSV; then
  $PY.train_lss --nusc $NUSC --gts $GTS --max-samples 2044 --val-samples 300 --epochs 24 \
      --batch-size 2 --lr 2e-3 $CFG --out-dir $ST > $OUT/occ_dinov3.log 2>&1
  miou=$(grep -oE "mIoU[ =:]+[0-9.]+" $OUT/occ_dinov3.log | grep -oE "[0-9.]+" | sort -g | tail -1)
  geo=$(grep -oE "geo[_-]?IoU[ =:]+[0-9.]+" $OUT/occ_dinov3.log | grep -oE "[0-9.]+" | sort -g | tail -1)
  echo "dinov3,${miou:-NA},${geo:-NA}" >> $OCSV
  echo "[sig] dinov3 -> occ mIoU=${miou}"
fi

# 2) det-transfer @2k seed1
if ! grep -q "^dinov3," $DCSV; then
  [ -f $ST/lss_occ.pth ] || { echo "[sig] no occ ckpt, abort det"; exit 1; }
  sleep 20
  $PY.train_det_ablation --nusc $NUSC --gts $GTS --pretrained $ST/lss_occ.pth $CFG --det-head center \
      --max-samples 2000 --val-samples 200 --epochs 12 --batch-size 8 --lr 2e-3 --cosine \
      --num-workers 4 --seed 1 --out-dir $OUT/det_dinov3
  $PY.eval_det_ablation_official --nusc $NUSC --gts $GTS --ckpt $OUT/det_dinov3/det_abl.pth \
      --out-dir $OUT/det_dinov3_eval > $OUT/det_dinov3_eval.log 2>&1
  mAP=$(grep -oE "mAP = [0-9.]+" $OUT/det_dinov3_eval.log | tail -1 | grep -oE "[0-9.]+")
  NDS=$(grep -oE "NDS = [0-9.]+" $OUT/det_dinov3_eval.log | tail -1 | grep -oE "[0-9.]+")
  ped=$(grep -E "pedestrian" $OUT/det_dinov3_eval.log | tail -1 | grep -oE "[0-9.]+" | tail -1)
  echo "dinov3,2000,1,${mAP:-NA},${NDS:-NA},${ped:-NA}" >> $DCSV
  echo "[sig] dinov3 -> det mAP=${mAP} NDS=${NDS}"
  rm -f $OUT/det_dinov3/det_abl.pth
fi
echo "[sig] DINOv3 arm DONE"
