# =====================================================================
# B13 — Real Waymo v1 GT supervised fine-tune of B10c.
#
# Same architecture as B12 but with 3× more training data:
#   * 4453 train frames from 23 segments (Waymo v1.4.3, extracted)
#   * 793 held-out val frames from 4 segments
#   * vs B12's 1508 train / 416 val (Waymo v2 subset)
#
# Also softer hyperparameters to avoid the V7 catastrophic-forgetting
# trap:
#   * LR 5e-6 (½ of B12's 1e-5)
#   * Freeze img_neck + view_transform too
#   * Train only fusion + head
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
        'projects.bevdet.waymo_finetune_dataset',
    ],
    allow_failed_imports=False,
)

data_root_waymo = '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_finetune'
waymo_v1_root   = '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_v1_extracted'

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
input_modality = dict(use_lidar=True, use_camera=True)
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
backend_args = None

train_pipeline = [
    dict(type='LoadWaymoFrameFromInfo',
         waymo_v1_root=waymo_v1_root,
         y_flip=False),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
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
         waymo_v1_root=waymo_v1_root,
         y_flip=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img',
                    'cam2lidar', 'img_aug_matrix',
                    'box_type_3d', 'sample_idx', 'lidar_path',
                    'img_path', 'num_pts_feats']),
]

train_dataloader = dict(
    batch_size=8, num_workers=4, persistent_workers=True, pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='WaymoFineTuneDataset',
        data_root=data_root_waymo,
        ann_file='waymo_v1_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=dict(classes=class_names),
        box_type_3d='LiDAR',
        waymo_root='',                  # unused for v1
        waymo_split='',                 # unused for v1
        y_flip_on_load=False,
        backend_args=backend_args,
    ),
)
# val/test disabled — mmengine requires (val_dataloader, val_cfg, val_evaluator)
# to be all-None or all-set. We evaluate externally via the V6 pipeline.
val_dataloader = None
test_dataloader = None

train_cfg = dict(by_epoch=True, max_epochs=1)
val_cfg = None
test_cfg = None
val_evaluator = None
test_evaluator = None

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=1, end=1, by_epoch=True,
         eta_min_ratio=1e-3, convert_to_iter_based=True),
]

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    accumulative_counts=1,
    optimizer=dict(type='AdamW', lr=5e-6, betas=(0.9, 0.99),
                   weight_decay=0.01, fused=True),
    paramwise_cfg=dict(custom_keys={
        # Aggressive freezing — only train fusion + head to limit catastrophic
        # forgetting (V7 lesson).
        'img_backbone': dict(lr_mult=0.0),
        'img_neck': dict(lr_mult=0.0),
        'view_transform': dict(lr_mult=0.0),
        'pts_voxel_encoder': dict(lr_mult=0.5),
        'pts_middle_encoder': dict(lr_mult=0.5),
        'pts_neck': dict(lr_mult=1.0),
        'fusion_layer': dict(lr_mult=2.0),
        'bbox_head': dict(lr_mult=2.0),
        'absolute_pos_embed': dict(decay_mult=0.0),
        'norm': dict(decay_mult=0.0),
    }),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)

load_from = '/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/ablation_B10c/epoch_3.pth'
resume = False

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2,
                    save_best='auto'),
    logger=dict(type='LoggerHook', interval=50),
)
