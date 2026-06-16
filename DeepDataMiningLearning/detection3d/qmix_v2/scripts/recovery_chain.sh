#!/bin/bash
# Recovery after HPC GPU session loss. Re-run the interrupted work:
#   1) B32 (Q-Mix v2) train + 300-frame Waymo eval  -- was killed at ~6%, no ckpt
#   2) B31 nuScenes back-eval  -- was killed at 4800/6019 (finish Pareto curve)
#   3) occupancy/velocity viz  -- never ran
set -u
WAYMO_ROOT=/fs/atipa/data/rnd-liu/Datasets/waymo201
EVAL=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/detection3d/eval_waymo_zeroshot.py
MMDET3D=/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d
LOG=/tmp/recovery_chain.log
echo "Recovery chain started at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG"

# ---------- 1) B32 Q-Mix v2 train + 300-frame eval ----------
CFG=projects/bevdet/configs/finetune/B32_qmix_v2.py
WORK_DIR=work_dirs/finetune_B32
CKPT=${MMDET3D}/${WORK_DIR}/epoch_1.pth
echo "## B32 train start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
cd "$MMDET3D"; rm -rf "$WORK_DIR"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=. \
  conda run --no-capture-output -n py310 python tools/train.py \
  --config "$CFG" --work-dir "$WORK_DIR" > /tmp/B32_train.log 2>&1
echo "## B32 train done: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
if [ -f "$CKPT" ]; then
  echo "## B32 300-frame eval start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  rm -rf /tmp/waymo_zs_B32
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    conda run --no-capture-output -n py310 python "$EVAL" \
    --config "$CFG" --checkpoint "$CKPT" --ckpt-multi "$CKPT" \
    --waymo-root "$WAYMO_ROOT" --split validation \
    --out-dir /tmp/waymo_zs_B32 --frame-stride 20 --max-frames 300 --num-sweeps 1 \
    > /tmp/B32_quick.log 2>&1
  echo "## B32 300-frame eval done: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  grep -E "Vehicle |Pedestrian |Cyclist |Macro" /tmp/B32_quick.log | tail -5 >> "$LOG"
else
  echo "## B32 ABORT (no ckpt)" >> "$LOG"
fi

# ---------- 2) B31 nuScenes back-eval (finish Pareto curve) ----------
B31_CKPT=${MMDET3D}/work_dirs/finetune_B31/epoch_1.pth
echo "## B31 nuScenes back-eval start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=. \
  conda run --no-capture-output -n py310 python tools/test.py \
  projects/bevdet/configs/finetune/B14_nuscenes_backeval.py "$B31_CKPT" \
  --work-dir work_dirs/B31_nuscenes_backeval > /tmp/B31_nuscenes.log 2>&1
echo "## B31 nuScenes back-eval done: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
grep -oE "NuScenes/NDS: [0-9.]+|NuScenes/mAP: [0-9.]+" /tmp/B31_nuscenes.log | tail -2 >> "$LOG"

# ---------- 3) occupancy/velocity viz ----------
echo "## occ/vel viz start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
B10C_CFG=$MMDET3D/projects/bevdet/configs/ablation/B10c_flow_guided_warmstart_fixed.py
B10C_CKPT=$MMDET3D/work_dirs/ablation_B10c/epoch_3_multitask.pth
VIS=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/detection3d/bev_multimodal_infer_vis.py
OUTDIR=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/paper_corl/figures/qualitative
mkdir -p "$OUTDIR"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=.:$MMDET3D \
  conda run --no-capture-output -n py310 python "$VIS" \
  --config "$B10C_CFG" --ckpt "$B10C_CKPT" --dataset nuscenes \
  --num-samples 6 --start-index 0 --save-dir "$OUTDIR/nuscenes" >> "$LOG" 2>&1
conda run --no-capture-output -n py310 python \
  /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/paper_corl/figures/assemble_occupancy_panel.py >> "$LOG" 2>&1
echo "## occ/vel viz done: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

touch /tmp/recovery_done.flag
echo "RECOVERY DONE: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
