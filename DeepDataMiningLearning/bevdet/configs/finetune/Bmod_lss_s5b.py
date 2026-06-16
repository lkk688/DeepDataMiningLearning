# =====================================================================
# Bmod (Stage 5b) — paper-informed camera-only fix. Three changes over s5:
#   (1) FIX img_aug bug: view_transform now applies img_aug_matrix in the
#       projection (apply_img_aug=True). It was always ignored (read from empty
#       kwargs) → camera sampling geometrically scrambled (~2.3x scale off from
#       the ImageAug3D resize). LIKELY THE main root cause of camera-only collapse;
#       LiDAR-fusion masked it (camera only +3 NDS, C-only=0).
#   (2) Camera-aware depthnet (cam_aware_depth=True): condition the depth net on
#       (post-aug) camera intrinsics [fx,fy,cx,cy] + residual blocks (BEVDepth/
#       STUR3D). s5's plain 2-conv depthnet plateaued loss_depth at ~3.7.
#   (3) Focal depth loss (in MultiTaskBEVFusion depth_from_vt path).
# Plus the s5 setup: depth-distribution lifting, num_z=8, unfrozen neck+VT,
# dedicated warm-started cam head.
#
# EARLY-ABORT GATE: if loss_depth doesn't fall below ~2.5 by epoch 3-4, the depth
# (hence camera BEV) won't work → stop and accept L/LC.
#
# Launch: py310 train.py --config .../Bmod_lss_s5b.py --work-dir work_dirs/Bmod_lss_s5b
#   --cam-head --batch-size 6 --cam-loss-weight 1.0 --depth-loss-weight 1.0
#   --log-interval 50 --keep-last-n 12
# =====================================================================

_base_ = ['./B22_qmix_v0.py']

load_from = '/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/Bmod_dropout_s2/epoch_4.pth'

model = dict(
    view_transform=dict(
        depth_lift=True,
        cam_aware_depth=True,
        apply_img_aug=True,
        depth_hidden=128,
        num_z=8,
        attn_chunk=2048,
    ),
)

optim_wrapper = dict(
    optimizer=dict(lr=1.0e-4),
    paramwise_cfg=dict(custom_keys=dict(
        view_transform=dict(lr_mult=1.0),
        img_neck=dict(lr_mult=0.5),
    )),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
]

train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=12)
