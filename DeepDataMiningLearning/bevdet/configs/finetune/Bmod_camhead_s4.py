# =====================================================================
# Bmod (Stage 4) — ARCHITECTURE CHANGE for camera-only capability.
#
# Stage-2 (camera-supervised dropout) and Stage-3 (BEV-feature KD) both FAILED:
# camera-only stayed flat (~0.004 head bias, 0 boxes). Root causes were (1) the
# CA-LSS camera BEV is geometry-weak (only 4 implicit depth anchors, NO depth
# supervision) and (2) the shared detection head is LiDAR-anchored, so training
# camera signal through it never lifts off.
#
# This stage fixes BOTH (BEVDepth recipe), via train.py flags (no config change
# to the model graph — wiring is in MultiTaskBEVFusion + build_model):
#   --depth-head        : BEVDepth-style per-pixel depth supervision on the camera
#                         FPN features (GT = projected LiDAR). Grounds the camera BEV.
#   --cam-head          : a 2nd TransFusion head (clone of bbox_head) trained on the
#                         camera-ONLY deep BEV. Never sees LiDAR -> can't be anchored.
# Per batch: full-LC pass -> main head (preserves L/LC) + depth aux ; camera-only
# pass -> cam head. Inference routes by available sensors.
#
# Launch (Phase 1, 12 ep first test):
#   py310 projects/bevdet/train/train.py --config .../Bmod_camhead_s4.py \
#     --work-dir work_dirs/Bmod_camhead_s4 --cam-head --depth-head \
#     --cam-loss-weight 1.0 --depth-loss-weight 0.5 --log-interval 50
# =====================================================================

_base_ = ['./B22_qmix_v0.py']

# Warm-start from the modality-robust Stage-2 model (good L/LC + L-only).
load_from = '/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/Bmod_dropout_s2/epoch_4.pth'

# The cam head is randomly initialized (trained from scratch), so keep a real LR;
# warmup then hold constant (no decay-to-zero).
optim_wrapper = dict(optimizer=dict(lr=1.0e-4))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
]

# Phase 1: 12 epochs to test whether camera-only lifts off the floor; extend to
# 24 for the final model if there is signal.
train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=12)
