# =====================================================================
# B12 — Real Waymo v2 GT supervised fine-tune of B10c.
#
# Setup
# -----
# * Init: B10c epoch_3.pth (warm start with the working V6 baseline).
# * Data: 160-segment subset of waymo201/validation, segment-level split
#   (avoids data leakage). 31,639 train frames, 8,311 held-out val.
# * GT: real Waymo v2 3D-GT, **y-flipped** at info.pkl build time so it
#   lives in the same frame as the model's natural output (V6 finding).
# * Input: raw Waymo vehicle-frame points + raw lidar2img / lidar2cam.
#   We do NOT y-flip the input — the model already converts un-flipped
#   input to y-flipped output (V6); training with un-flipped input +
#   y-flipped GT reinforces that existing behavior.
# * After training: V6 eval continues to work — predictions still come
#   out in y-flipped frame, get post-flipped to vehicle for matching
#   against the un-flipped Waymo GT we eval on.
#
# Budget: 1 epoch on 31K frames at LR 1e-5 (10× lower than B10c's
# fine-tune LR). Pure adaptation, not large-scale retraining.
# =====================================================================

_base_ = ['../ablation/B10c_flow_guided_warmstart_fixed.py']

custom_imports = dict(
    imports=[
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.bevfusion_ca',
        'projects.bevdet.multitoken_temporal_lidar',
        'projects.bevdet.temporal_lidar',
        'projects.bevdet.flow_guided_temporal',
        'projects.bevdet.waymo_finetune_dataset',   # NEW: ours
    ],
    allow_failed_imports=False,
)

# ---- Data root + dataset class ---------------------------------------
data_root_waymo = '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_finetune'
waymo_v2_root = '/fs/atipa/data/rnd-liu/Datasets/waymo201'

# Reuse the same nuScenes class list (10 classes) so the head output layer
# matches. We map Waymo's {1,2,4} into {car, pedestrian, bicycle}.
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

input_modality = dict(use_lidar=True, use_camera=True)
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

backend_args = None

# ---- Custom pipeline that uses the Waymo loader ----------------------
# We replace the 3 standard loading transforms with one custom loader.
train_pipeline = [
    dict(type='LoadWaymoFrameFromInfo',
         waymo_root=waymo_v2_root, waymo_split='validation',
         y_flip=False),                     # see config docstring
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    # ImageAug3D SKIPPED — LoadWaymoFrameFromInfo already resizes images
    # to 256x704 and bakes img_aug into lidar2img / img_aug_matrix.
    dict(type='BEVFusionGlobalRotScaleTrans',
         scale_ratio_range=[0.9, 1.1],
         rot_range=[-0.78539816, 0.78539816],
         translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs',
         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=[
             'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
             'img_aug_matrix', 'lidar_aug_matrix',
             'box_type_3d', 'sample_idx', 'lidar_path', 'img_path',
             'transformation_3d_flow', 'pcd_rotation',
             'pcd_scale_factor', 'pcd_trans', 'num_pts_feats',
         ]),
]

test_pipeline = [
    dict(type='LoadWaymoFrameFromInfo',
         waymo_root=waymo_v2_root, waymo_split='validation',
         y_flip=False),
    # ImageAug3D skipped — see train_pipeline comment above.
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img',
                    'cam2lidar', 'img_aug_matrix',
                    'box_type_3d', 'sample_idx', 'lidar_path',
                    'img_path', 'num_pts_feats']),
]

# ---- Dataloaders -----------------------------------------------------
# Smaller batch than B10c (8 instead of 32) to leave headroom for the
# heavy Waymo loading. Reduce num_workers because each worker re-init's
# Waymo3DDataset (one-time scan of the parquet directory).
train_dataloader = dict(
    batch_size=8, num_workers=4, persistent_workers=True, pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='WaymoFineTuneDataset',
        data_root=data_root_waymo,
        ann_file='waymo_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=dict(classes=class_names),
        box_type_3d='LiDAR',
        waymo_root=waymo_v2_root,
        waymo_split='validation',
        y_flip_on_load=False,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='WaymoFineTuneDataset',
        data_root=data_root_waymo,
        ann_file='waymo_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=dict(classes=class_names),
        box_type_3d='LiDAR',
        waymo_root=waymo_v2_root,
        waymo_split='validation',
        y_flip_on_load=False,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

# ---- Schedule: 1 epoch fine-tune at LR=1e-5 --------------------------
train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
val_cfg = dict()
test_cfg = dict()

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=1, end=1, by_epoch=True,
         eta_min_ratio=1e-3, convert_to_iter_based=True),
]

# ---- Optimizer: 10× lower LR than B10c, freeze img backbone -----------
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    accumulative_counts=1,
    optimizer=dict(type='AdamW', lr=1e-5, betas=(0.9, 0.99),
                   weight_decay=0.01, fused=True),
    paramwise_cfg=dict(custom_keys={
        # Freeze image backbone entirely — domain shift is at the
        # geometry/fusion level, not at low-level visual features.
        'img_backbone': dict(lr_mult=0.0),
        # Camera neck still gets gradient (small).
        'img_neck': dict(lr_mult=0.5),
        'view_transform': dict(lr_mult=1.0),
        'fusion_layer': dict(lr_mult=1.5),
        'bbox_head': dict(lr_mult=2.0),

        # LiDAR side: heavier adaptation since LiDAR distribution
        # differs the most between nuScenes and Waymo.
        'pts_voxel_encoder': dict(lr_mult=2.0),
        'pts_middle_encoder': dict(lr_mult=1.5),
        'pts_neck': dict(lr_mult=2.0),

        'absolute_pos_embed': dict(decay_mult=0.0),
        'norm': dict(decay_mult=0.0),
    }),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)

# ---- Warm start from B10c epoch_3.pth --------------------------------
load_from = '/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/ablation_B10c/epoch_3.pth'
resume = False

# Tag the work directory so checkpoints/logs land in a B12-specific spot.
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2,
                    save_best='auto'),
    logger=dict(type='LoggerHook', interval=50),
)
