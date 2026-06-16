#!/bin/bash
# Auto-launch B32 (Q-Mix v2) once PL re-extraction is verified complete.
# Waits for the extraction done-flag + resolvable check, then trains + 300eval.
set -u
REPO=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
MMDET3D=/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d
EVAL=$REPO/DeepDataMiningLearning/detection3d/eval_waymo_zeroshot.py
WAYMO_ROOT=/fs/atipa/data/rnd-liu/Datasets/waymo201
CFG=projects/bevdet/configs/finetune/B32_qmix_v2.py
WORK_DIR=work_dirs/finetune_B32
CKPT=$MMDET3D/$WORK_DIR/epoch_1.pth
LOG=/tmp/b32_launch.log
echo "B32 launcher queued $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG"

# 1) wait for PL extraction to finish
until [ -f /tmp/reextract2_done.flag ]; do sleep 60; done
echo "## PL extraction done; verify: $(cat /tmp/pl_reextract_ok 2>/dev/null)" >> "$LOG"
if [ "$(cat /tmp/pl_reextract_ok 2>/dev/null)" != "3409,0" ]; then
  # tolerate a few missing but warn
  echo "## WARNING: PL resolvable != 3409,0 (got $(cat /tmp/pl_reextract_ok 2>/dev/null))" >> "$LOG"
fi

# 2) wait until GPU has room (B31 back-eval / occ-viz may still run; need >=45GB)
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 45000 ]; do
  sleep 60
done
echo "## GPU free; B32 train start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

cd "$MMDET3D"; rm -rf "$WORK_DIR"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=. \
  conda run --no-capture-output -n py310 python tools/train.py \
  --config "$CFG" --work-dir "$WORK_DIR" > /tmp/B32_train.log 2>&1
echo "## B32 train done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
if [ ! -f "$CKPT" ]; then echo "## B32 ABORT (no ckpt)" >> "$LOG"; exit 1; fi

echo "## B32 300-frame eval start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
rm -rf /tmp/waymo_zs_B32
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  conda run --no-capture-output -n py310 python "$EVAL" \
  --config "$CFG" --checkpoint "$CKPT" --ckpt-multi "$CKPT" \
  --waymo-root "$WAYMO_ROOT" --split validation \
  --out-dir /tmp/waymo_zs_B32 --frame-stride 20 --max-frames 300 --num-sweeps 1 \
  > /tmp/B32_quick.log 2>&1
echo "## B32 300-frame eval done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
grep -E "Vehicle |Pedestrian |Cyclist |Macro" /tmp/B32_quick.log | tail -5 >> "$LOG"
touch /tmp/b32_done.flag
echo "B32 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
