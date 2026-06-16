# =====================================================================
# Bmod (Stage 5) — BEVDepth-style DEPTH-LIFT rewrite + UNFREEZE camera branch.
#
# Stages 2/3/4 all failed (camera-only flat ~0.004-0.011). Two root causes,
# fixed here together:
#   (1) CA-LSS lifting ignored depth: it combined K implicit height anchors via
#       cross-attention. Stage-5 sets view_transform.depth_lift=True → the VT now
#       predicts a per-pixel depth distribution and weights each (cell,cam,z)
#       sample by p(its camera depth), i.e. BEVDepth lift-splat in the inverse/
#       sampling formulation. num_z raised 2→8 for vertical resolution. The VT's
#       depthnet is supervised on projected-LiDAR depth GT (depth_from_vt, auto-
#       detected in build_model → MultiTaskBEVFusion).
#   (2) The whole CAMERA BRANCH WAS FROZEN (img_backbone/img_neck/view_transform
#       lr_mult=0.0) in every prior stage — so the cam BEV could never adapt.
#       Unfreeze view_transform (1.0) + img_neck (0.5); keep Swin img_backbone
#       frozen (0.0) for stability/cost (revisit if signal is weak).
#
# Paired with --cam-head (dedicated head on the camera-only BEV, warm-started
# from the main head). Launch (Phase-2, 12 ep first signal check):
#   py310 train.py --config .../Bmod_lss_s5.py --work-dir work_dirs/Bmod_lss_s5 \
#     --cam-head --cam-loss-weight 1.0 --depth-loss-weight 1.0 --log-interval 50
# (no --depth-head: the VT's own depthnet is used + supervised instead.)
# =====================================================================

_base_ = ['./B22_qmix_v0.py']

load_from = '/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/Bmod_dropout_s2/epoch_4.pth'

# Enable depth-distribution lifting in the view transform (merges into the base
# CrossAttnLSSTransform dict — keeps type/in_channels/dbound/etc).
model = dict(
    view_transform=dict(
        depth_lift=True,
        depth_hidden=128,
        num_z=8,          # height anchors per BEV cell (was 2); depth dist picks the right one
        attn_chunk=2048,  # halve BEV-query chunk: depth grid_sample peak ~ chunk*K*D (mem)
    ),
)

# Unfreeze the camera branch so the depthnet + lifting + cam features can learn.
# (Base froze img_backbone/img_neck/view_transform at lr_mult=0.0.)
optim_wrapper = dict(
    optimizer=dict(lr=1.0e-4),
    paramwise_cfg=dict(custom_keys=dict(
        view_transform=dict(lr_mult=1.0),
        img_neck=dict(lr_mult=0.5),
        # img_backbone stays frozen (inherited lr_mult=0.0)
    )),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
]

train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=12)
