# =====================================================================
# B0 — Modern BEVFusion baseline for fair ablation chain.
#
# This is the *reference* baseline our paper compares against. It mirrors
# the canonical BEVFusion (DepthLSSTransform + SECFPN + ConvFuser) but
# adds the standard auxiliary heads that any "modern" 2026 BEVFusion
# would already include:
#   • 3D occupancy auxiliary supervision (binary pseudo-occ from LiDAR)
#   • LiDAR depth distribution supervision on FPN P3
# Both aux heads are training-only (zero deployed parameters).
#
# Trained from scratch (no warm-start) on a 25% scene-stratified subsample
# of nuScenes-train, 6 epochs, single H100, BF16, batch_size=16.
#
# Subsequent rows in the ablation chain (B1..B4) replace LSS with CA-LSS
# and add components ONE AT A TIME so each delta is attributable.
# =====================================================================

_base_ = [
    '../bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# ---- 25% subset training data -----------------------------------------
train_dataloader = dict(
    batch_size=16, num_workers=8, persistent_workers=True,
    pin_memory=True, prefetch_factor=4,
    dataset=dict(
        # CBGSDataset wraps NuScenesDataset; replace ann_file in inner dataset
        dataset=dict(
            ann_file='nuscenes_infos_train_25pct.pkl'
        )
    )
)

# ---- 6-epoch schedule -------------------------------------------------
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=2)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333, by_epoch=False, begin=0, end=400),
    dict(type='CosineAnnealingLR', T_max=6, begin=0, end=6,
         by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', begin=0, end=2.4, eta_min=0.85/0.95,
         by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', begin=2.4, end=6, eta_min=1,
         by_epoch=True, convert_to_iter_based=True),
]

# ---- BF16 AMP, frozen Swin --------------------------------------------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    accumulative_counts=1,
    optimizer=dict(type='AdamW', lr=2e-4, betas=(0.9, 0.99),
                   weight_decay=0.01, fused=True),
    paramwise_cfg=dict(custom_keys={
        'img_backbone':                  dict(lr_mult=0.0, decay_mult=0.0),
        'absolute_pos_embed':            dict(decay_mult=0.0),
        'relative_position_bias_table':  dict(decay_mult=0.0),
        'norm':                          dict(decay_mult=0.0),
    }),
    clip_grad=dict(max_norm=20, norm_type=2),
)

auto_scale_lr = dict(enable=False, base_batch_size=32)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', interval=1),
)

# Disable any custom_hooks inherited from the base (e.g., FreezeExceptHook).
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=False, after_epoch=True),
]

# ---- No warm-start: train from scratch (Swin pretrained init only) ----
auto_resume = False
resume = False
