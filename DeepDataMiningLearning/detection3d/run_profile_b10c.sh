#!/bin/bash
# Phase 1: per-stage GPU profile on B10c (the most representative ckpt for
# what our paper actually claims). Run after B11 finishes (GPU must be free).
#
# Outputs:
#   work_dirs/profile_b10c/profile.json   per-stage mean/p50/p95 in ms
#   stdout                                formatted ranking + end-to-end FPS

set -e

# Wait until the GPU has at least 30 GB free.
TARGET_FREE_MIB=30000
echo "[profile] Waiting for GPU memory free >= ${TARGET_FREE_MIB} MiB ..."
while true; do
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    free_mib=$((total_mib - used_mib))
    if [ "$free_mib" -ge "$TARGET_FREE_MIB" ]; then
        echo "[profile] GPU free: ${free_mib} MiB — starting."
        break
    fi
    sleep 60
done

cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/detection3d

CONFIG=/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/projects/bevdet/configs/ablation/B10c_flow_guided_warmstart_fixed.py
CKPT=/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/ablation_B10c/epoch_3.pth
CKPT_MULTI=/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/ablation_B10c/epoch_3_multitask.pth
DATAROOT=/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes
OUT_DIR=/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/profile_b10c

mkdir -p "$OUT_DIR"

conda run --no-capture-output -n py310 python -u simple_infer_main.py \
    --config       "$CONFIG" \
    --checkpoint   "$CKPT" \
    --unified-checkpoint-multitask "$CKPT_MULTI" \
    --dataroot     "$DATAROOT" \
    --out-dir      "$OUT_DIR" \
    --data-source  cfg \
    --ann-file     "$DATAROOT/nuscenes_infos_val_mkf30.pkl" \
    --dataset      nuscenes \
    --device       cuda \
    --workers      0 \
    --max-samples  25 \
    --profile-stages \
    --profile-num-samples 20 \
    --profile-warmup      3 \
    --occ-classes  2 \
    --occ-num-z    16

echo "[profile] Done. Report: $OUT_DIR/profile.json"
