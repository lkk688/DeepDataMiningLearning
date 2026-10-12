#!/usr/bin/env bash
# Re-run the Direction-4 planning head after the NaN-crash fix (collision_rate NaN-safe + grad-clip +
# skip-diverged-step + lower LR). Frozen-probe occ backbones (DINOv2-L / -B). L2@1/2/3s + collision.
set +e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.bashrc 2>/dev/null || true; conda activate py310 2>/dev/null || true
ROOT=/data/rnd-liu/Datasets/nuScenes; NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
BB=DeepDataMiningLearning/ngperception/output/backbone_bench
PY="python -m DeepDataMiningLearning.ngperception.occupancy"
: > $BB/planning.txt   # reset (previous entries were crashed partials)

for entry in "dinov2_large:$BB/occ_dinov2_large/lss_occ.pth" "dinov2_base:$BB/occ_dinov2_base/lss_occ.pth"; do
  bb="${entry%%:*}"; ck="${entry##*:}"
  [ -f "$ck" ] || { echo "[plan] skip $bb (no ckpt)"; continue; }
  $PY.train_planning --nusc $NUSC --gts $GTS --pretrained $ck --backbone $bb \
      --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 --head transformer --lr 1e-3 \
      --max-samples 4000 --val-samples 400 --epochs 12 --batch-size 4 --num-workers 6 \
      --out-dir $BB/plan_${bb} > $BB/plan_${bb}.log 2>&1
  res=$(grep -E "\[plan\] epoch" $BB/plan_${bb}.log | tail -1)
  echo "${bb} PLANNING -> ${res}" >> $BB/planning.txt
  echo "[plan-rerun] $bb -> ${res}"
done
echo "[plan-rerun] DONE"
