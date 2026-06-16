# =====================================================================
# Bmod (Stage 5c) — FINAL bounded test: unfreeze the Swin img_backbone.
#
# s5b kept all fixes (img_aug applied, camera-aware depthnet, focal depth loss,
# depth-lift, unfrozen neck+VT) yet loss_depth still plateaued ~3.5 and camera-
# only stayed flat. The ONE variable untouched across all 5 attempts is the
# FROZEN Swin img_backbone — a frozen backbone can't surface metric-depth
# features, so the depthnet hits a ceiling regardless of conditioning/loss.
#
# This test unfreezes img_backbone (gentle lr_mult=0.1 to protect LC and the
# pretrained features). EARLY-ABORT: if loss_depth doesn't fall below ~2.5 by
# epoch 2-3 with a trainable backbone, the architecture genuinely can't do
# camera-only → accept L/LC (negative result airtight).
#
# Launch (bs4 — Swin-backward activations need headroom over s5b's bs6):
#   py310 train.py --config .../Bmod_lss_s5c.py --work-dir work_dirs/Bmod_lss_s5c
#     --cam-head --batch-size 4 --cam-loss-weight 1.0 --depth-loss-weight 1.0
#     --log-interval 50 --keep-last-n 12
# =====================================================================

_base_ = ['./Bmod_lss_s5b.py']

# Unfreeze the Swin backbone (was lr_mult=0.0). Gentle to preserve LC + features.
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=dict(
        img_backbone=dict(lr_mult=0.1),
        img_neck=dict(lr_mult=0.5),
        view_transform=dict(lr_mult=1.0),
    )),
)
