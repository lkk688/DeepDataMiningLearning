# =====================================================================
# B15c — continue B15 (temporal multi-sweep) for 2 more epochs from
# B15 epoch_1.pth, keeping all other settings identical. Tests whether
# the conservative-shift seen in V10b heals with more training.
# =====================================================================

_base_ = ['./B15_waymo_nuscenes_mixed_temporal.py']

custom_imports = dict(
    imports=[
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.bevfusion_ca',
        'projects.bevdet.multitoken_temporal_lidar',
        'projects.bevdet.temporal_lidar',
        'projects.bevdet.flow_guided_temporal',
        'projects.bevdet.waymo_finetune_dataset',
    ],
    allow_failed_imports=False,
)

# Warm-start from B15's epoch_1, not B14 — pick up where we left off.
load_from = '/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/finetune_B15/epoch_1.pth'
resume = False

# 2 more epochs at slightly lower LR (since we're already partially adapted).
train_cfg = dict(by_epoch=True, max_epochs=2)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(type='CosineAnnealingLR', begin=0, T_max=2, end=2, by_epoch=True,
         eta_min_ratio=1e-3, convert_to_iter_based=True),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3,
                    save_best='auto'),
    logger=dict(type='LoggerHook', interval=100),
)
