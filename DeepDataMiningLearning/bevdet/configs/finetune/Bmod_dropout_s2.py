# =====================================================================
# Bmod (Stage 2) — Modality-robust backbone, FIXED LR.
#
# Stage-1 failed: inherited lr=5e-6 + CosineAnnealingLR(end=1) decayed LR
# to ~1e-12 by epoch 2, so the model never adapted (loss stuck ~11,
# matched_ious 0.03). A preserve-the-model LR can't TEACH camera-only
# detection — a capability the fused model entirely lacks (C-only NDS 0).
#
# Fix: warm-start fresh from B22, raise LR to 1e-4 with warmup-then-
# constant (no decay-to-zero), 4 epochs. Modality dropout via env
# BEV_MOD_DROP="0.5,0.25,0.25"; the 50% full-LC batches anchor the fused
# path against catastrophic forgetting while the 25% C-only batches teach
# the camera->BEV->head path.
# =====================================================================

_base_ = ['./B22_qmix_v0.py']

load_from = '/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/finetune_B22/epoch_1_weights.pth'

# Higher LR to learn a new capability; warmup then hold constant (the
# Stage-1 schedule collapsed LR to zero).
optim_wrapper = dict(optimizer=dict(lr=1.0e-4))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
]

train_cfg = dict(by_epoch=True, max_epochs=4, val_interval=4)
