# =====================================================================
# B10 = B9 + flow-guided temporal attention + GT velocity supervision +
#         FROM-SCRATCH 6-epoch training on real multi-sweep data.
#
# Background
# ----------
# B5 (no temporal), B5R (no temporal, real multi-sweep), B8 V2
# (temporal attn, degenerate multi-sweep), B9 (temporal attn, real
# multi-sweep) all landed within ±0.005 NDS at the 3-epoch warm-start
# budget. The temporal cross-frame attention's correspondence problem
# is too data-hungry for that budget. B10 fixes this in two ways:
#
#   1. *Flow-guided warp.* A small per-cell BEV velocity head (~50k
#      params) predicts v(h,w). Past tokens are sampled with grid_sample
#      at the predicted offset BEFORE attention runs, so the attention
#      block only does residual refinement — not "discover correspondence
#      from scratch".
#
#   2. *GT velocity supervision.* nuScenes objects come annotated with
#      velocity. We rasterize each object's BEV footprint and paint its
#      GT velocity as a dense supervision target for the flow head
#      (SmoothL1 loss, mask-weighted). This bypasses the slow gradient-
#      only learning of correspondence.
#
# To remove the warm-start mismatch problem (B5R got -0.005 NDS vs B5
# *purely* from fine-tuning the degenerate-multi-sweep ckpt on real
# data), B10 also trains FROM SCRATCH (no `--load-from`), 6 epochs,
# B0-style schedule.
# =====================================================================

_base_ = ['./B9_multikeyframe_temporal.py']

custom_imports = dict(
    imports=[
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.bevfusion_ca',
        'projects.bevdet.multitoken_temporal_lidar',
        'projects.bevdet.flow_guided_temporal',
    ],
    allow_failed_imports=False,
)

# --- Switch the detector class. The data + pipeline + ann_file and all
# inherited model fields (data_preprocessor / pts_* / view_transform /
# fusion_layer / aux_cfg / etc.) are kept from the B5_gqa → B9 chain via
# mmengine's recursive dict merge. We only add fields the new detector
# needs and override `type`. (Do NOT use _delete_=True here — it would
# wipe data_preprocessor and all submodule configs.) ---
model = dict(
    type='FlowGuidedTemporalBEVFusion',
    # num_buckets / window_seconds / num_attn_heads / attn_dropout are
    # inherited from B9 (4 / 1.5 / 4 / 0.0). New fields below.
    flow_loss_weight=0.5,
    cell_size=0.6,
    pc_range=(-54.0, -54.0, 54.0, 54.0),
)

# --- From-scratch 6-epoch schedule (mirrors B0_modern_baseline) ---
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=2)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333, by_epoch=False, begin=0, end=400),
    dict(type='CosineAnnealingLR', T_max=6, begin=0, end=6,
         by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', begin=0, end=2.4, eta_min=0.85 / 0.95,
         by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', begin=2.4, end=6, eta_min=1,
         by_epoch=True, convert_to_iter_based=True),
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    accumulative_counts=1,
    optimizer=dict(type='AdamW', lr=2e-4, betas=(0.9, 0.99),
                   weight_decay=0.01, fused=True),
    paramwise_cfg=dict(custom_keys={
        # Keep Swin frozen — same as B0..B9 — so the only delta vs the
        # rest of the ablation chain is the LiDAR-side temporal block.
        'img_backbone':                 dict(lr_mult=0.0, decay_mult=0.0),
        'absolute_pos_embed':           dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0),
        'norm':                         dict(decay_mult=0.0),
    }),
    clip_grad=dict(max_norm=20, norm_type=2),
)

auto_scale_lr = dict(enable=False, base_batch_size=32)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', interval=1),
)
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=False, after_epoch=True),
]
auto_resume = False
resume = False
