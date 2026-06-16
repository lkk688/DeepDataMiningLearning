#!/bin/bash
# Run unified inference (3D + 2D + occupancy) on both B10c and B11 on the
# SAME 15 val samples so we can compare 2D-from-image-head outputs to
# 3D-projected boxes from the detection head.
#
# Outputs:
#   /tmp/unified_b10c/<sample>/  3D + occ JSONs   (no 2D — B10c has no image2d_head)
#   /tmp/unified_b11/<sample>/   3D + 2D + occ JSONs
#
# Then run compare_2d_vs_3d.py over these directories.

set -e

TARGET_FREE_MIB=20000
echo "[unified-test] Waiting for GPU ≥ ${TARGET_FREE_MIB} MiB free ..."
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    free=$((tot - used))
    [ "$free" -ge "$TARGET_FREE_MIB" ] && break
    sleep 60
done

cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/detection3d

DATAROOT=/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes
ANN_VAL="$DATAROOT/nuscenes_infos_val_mkf30.pkl"

run_one() {
    local TAG="$1"
    local CFG="$2"
    local CKPT="$3"
    local CKPT_MULTI="$4"
    local OUT="/tmp/unified_${TAG}"
    rm -rf "$OUT"
    mkdir -p "$OUT"
    echo "[unified-test] === ${TAG} ==="
    conda run --no-capture-output -n py310 python -u simple_infer_main.py \
        --config       "$CFG" \
        --checkpoint   "$CKPT" \
        --unified-checkpoint-multitask "$CKPT_MULTI" \
        --dataroot     "$DATAROOT" \
        --ann-file     "$ANN_VAL" \
        --out-dir      "$OUT" \
        --data-source  cfg \
        --dataset      nuscenes \
        --device       cuda \
        --workers      0 \
        --max-samples  15 \
        --unified-inference \
        --occ-classes  2 \
        --occ-num-z    16 \
        --unified-3d-score-thresh 0.10 \
        --unified-2d-score-thresh 0.10
    echo "[unified-test] ${TAG} → ${OUT}"
}

run_one "b10c" \
    /fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/projects/bevdet/configs/ablation/B10c_flow_guided_warmstart_fixed.py \
    /fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/ablation_B10c/epoch_3.pth \
    /fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/ablation_B10c/epoch_3_multitask.pth

run_one "b11" \
    /fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/projects/bevdet/configs/ablation/B11_image2d_aux.py \
    /fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/ablation_B11/epoch_3.pth \
    /fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/ablation_B11/epoch_3_multitask.pth

echo "[unified-test] Done — outputs at /tmp/unified_b10c/ and /tmp/unified_b11/"
