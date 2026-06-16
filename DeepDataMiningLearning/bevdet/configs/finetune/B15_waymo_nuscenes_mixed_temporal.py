# =====================================================================
# B15 — Mixed nuScenes + Waymo training WITH temporal multi-sweep.
#
# Same as B14 except:
#   * Waymo branch loads num_sweeps=5 instead of 1
#     (matches B10c's training distribution; activates FG-TCA temporal block)
#   * Warm-start from B14 epoch_1.pth (already adapted to Waymo distribution)
#   * 1 more epoch of training with temporal info active
#
# Hypothesis: FG-TCA was sitting dormant during B14 because we fed it
# single-frame Waymo points (no historical sweeps in the past slice).
# Enabling sweeps gives the temporal block actual motion cues to fuse,
# which should help dynamic objects (Pedestrian, Cyclist) most.
# =====================================================================

_base_ = ['./B14_waymo_nuscenes_mixed.py']

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

waymo_v1_root = '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_v1_extracted'
NUM_SWEEPS = 5

# ---- Waymo-side pipeline now uses num_sweeps=5 ----
# Override LoadWaymoFrameFromInfo arg in the inherited pipeline.
from copy import deepcopy
waymo_train_pipeline = deepcopy(_base_.train_pipeline)
# First entry is LoadWaymoFrameFromInfo — override num_sweeps.
assert waymo_train_pipeline[0]['type'] == 'LoadWaymoFrameFromInfo'
waymo_train_pipeline[0]['num_sweeps'] = NUM_SWEEPS

# Re-build the Waymo dataset spec with the patched pipeline.
waymo_dataset = deepcopy(_base_.waymo_dataset)
waymo_dataset['pipeline'] = waymo_train_pipeline

train_dataloader = dict(
    _delete_=True,
    batch_size=_base_.train_dataloader.batch_size,
    num_workers=_base_.train_dataloader.num_workers,
    persistent_workers=True, pin_memory=True, prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[waymo_dataset, _base_.nus_dataset],
        ignore_keys=['version', 'dataset', 'categories', 'info_version',
                     'sample_idx_to_data_idx'],
    ),
)

train_cfg = dict(by_epoch=True, max_epochs=1)
val_cfg = None
test_cfg = None
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None

# Warm-start from B14 (V9) so we build on the mixed-training gains.
load_from = '/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/finetune_B14/epoch_1.pth'
resume = False

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2,
                    save_best='auto'),
    logger=dict(type='LoggerHook', interval=100),
)
