# =====================================================================
# Bmod (Stage 3) — LC->C DISTILLATION for true camera-only capability.
#
# Stage-2 modality-dropout (BEV_MOD_DROP) gave robust L-only (NDS 0.557 ~ LC
# 0.569) but camera-only STILL collapsed: score diag showed a flat ~0.0088
# head bias everywhere (0 boxes survive). Root cause is structural: the
# detection head is LiDAR-anchored; zeroing the LiDAR BEV collapses the
# classifier to its bias, and 25% C-only dropout over 4 epochs can't wake a
# camera->BEV->head path the LiDAR-trained init never used.
#
# Fix = online self-distillation (env BEV_KD). Per batch, the detector loss()
# runs TWO forwards: (1) full-LC anchor (supervised det loss, preserves L/LC)
# whose final BEV is the teacher; (2) camera-only student whose BEV is pulled
# toward the detached teacher BEV via MSE. The shared (good) head then yields
# boxes from the camera-only BEV. Launch WITHOUT BEV_MOD_DROP and WITH:
#   env BEV_KD="1.0"            -> pure feature KD (featW=1.0, camSupW=0)
#   env BEV_KD="1.0,0.5"        -> + camera-only supervised det loss (weight 0.5)
# =====================================================================

_base_ = ['./B22_qmix_v0.py']

# Warm-start from the modality-robust Stage-2 model (good L/LC to preserve).
load_from = '/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/Bmod_dropout_s2/epoch_4.pth'

# Same warmup-then-constant LR as Stage-2 (learns a new capability without decay).
optim_wrapper = dict(optimizer=dict(lr=1.0e-4))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
]

# KD needs more passes over the data than dropout; 2x forward/batch is slower.
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=6)
