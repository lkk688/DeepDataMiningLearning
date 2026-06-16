# =====================================================================
# B14 — Mixed nuScenes + Waymo training of B10c.
#
# Combines:
#   * Waymo v1 (4453 frames from 23 segments, real GT, y-flipped)
#     — same data as B13
#   * nuScenes train 25% mkf30 (~7K frames, same data B10c was trained on)
#     — keeps the model's original knowledge alive, prevents
#       catastrophic forgetting that ruined V7 and slightly degraded V8's
#       cyclist class.
#
# Goals:
#   1. Lock in V8's gains (esp. pedestrian +95%)
#   2. Recover cyclist (V6: 0.115 → V8: 0.069; we want +).
#   3. Push Vehicle past V6's 0.36 ceiling.
#
# Data balancing
# --------------
# Natural ratio: ~4.5K Waymo + ~7K nuScenes = ~14%/86% split (Waymo
# samples are 1.6× per epoch). That's reasonable — we want to PROTECT
# nuScenes representations (which V7 catastrophically forgot) while
# nudging the head toward Waymo.
#
# Each dataset has its own pipeline (Waymo uses LoadWaymoFrameFromInfo,
# nuScenes uses BEVLoadMultiViewImageFromFiles + LoadPointsFromFile
# etc.) — ConcatDataset stitches them; outputs are uniform after
# Pack3DDetInputs.
# =====================================================================

_base_ = ['./B13_waymo_v1_finetune.py']

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

# ============== nuScenes-side pipeline =================================
# Mirrors B10c's train_pipeline. Standard nuScenes loaders + augmentations.
nus_data_root = 'data/nuscenes/'
backend_args = None
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
# Palette mirrors NuScenesDataset.METAINFO ordering but for OUR class order.
# Colors are arbitrary — they're only used for visualization, not training.
class_palette = [
    (255, 158, 0),    # car
    (255, 99, 71),    # truck
    (233, 150, 70),   # construction_vehicle
    (255, 127, 80),   # bus
    (255, 140, 0),    # trailer
    (112, 128, 144),  # barrier
    (255, 61, 99),    # motorcycle
    (220, 20, 60),    # bicycle
    (0, 0, 230),      # pedestrian
    (47, 79, 79),     # traffic_cone
]
common_metainfo = dict(classes=class_names, palette=class_palette,
                        version='waymo_nuscenes_mixed')
input_modality = dict(use_lidar=True, use_camera=True)

nus_train_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True,
         color_type='color', backend_args=backend_args),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5,
         use_dim=5, backend_args=backend_args),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=5, load_dim=5,
         use_dim=5, pad_empty_sweeps=True, remove_close=True,
         backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False),
    dict(type='ImageAug3D', final_dim=[256, 704],
         resize_lim=[0.38, 0.55], bot_pct_lim=[0.0, 0.0],
         rot_lim=[-5.4, 5.4], rand_flip=True, is_train=True),
    dict(type='BEVFusionGlobalRotScaleTrans',
         scale_ratio_range=[0.9, 1.1],
         rot_range=[-0.78539816, 0.78539816], translation_std=0.5),
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

# ============== Waymo-side pipeline ====================================
# Inherited as B13's `train_pipeline` (set in B13_waymo_v1_finetune.py).
# Pull it explicitly so the ConcatDataset entries see a deep copy.
from copy import deepcopy
waymo_train_pipeline = deepcopy(_base_.train_pipeline)

# ============== ConcatDataset(Waymo + nuScenes) =========================
waymo_dataset = dict(
    type='WaymoFineTuneDataset',
    data_root=_base_.data_root_waymo,
    ann_file='waymo_v1_infos_train.pkl',
    pipeline=waymo_train_pipeline,
    modality=input_modality,
    test_mode=False,
    metainfo=common_metainfo,
    box_type_3d='LiDAR',
    waymo_root='', waymo_split='',
    y_flip_on_load=False,
    backend_args=backend_args,
)

nus_dataset = dict(
    type='NuScenesDataset',
    data_root=nus_data_root,
    ann_file='nuscenes_infos_train_25pct_mkf30.pkl',
    pipeline=nus_train_pipeline,
    modality=input_modality,
    test_mode=False,
    metainfo=common_metainfo,
    data_prefix=dict(
        pts='samples/LIDAR_TOP',
        CAM_FRONT='samples/CAM_FRONT',
        CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
        CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
        CAM_BACK='samples/CAM_BACK',
        CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
        CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
        sweeps='sweeps/LIDAR_TOP'),
    box_type_3d='LiDAR',
    backend_args=backend_args,
)

train_dataloader = dict(
    _delete_=True,
    batch_size=8, num_workers=4, persistent_workers=True, pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[waymo_dataset, nus_dataset],
        # ConcatDataset normally requires identical metainfo across all
        # sub-datasets. Our two pipelines naturally have slightly different
        # metainfo (NuScenesDataset adds its own keys; ours adds a few
        # routing fields). Both share the same `classes` + `palette` (the
        # only ones that matter for label semantics) — ignore the rest.
        ignore_keys=['version', 'dataset', 'categories', 'info_version',
                     'sample_idx_to_data_idx'],
    ),
)
val_dataloader = None
test_dataloader = None
val_cfg = None
test_cfg = None
val_evaluator = None
test_evaluator = None

train_cfg = dict(by_epoch=True, max_epochs=1)

# Schedule scaled for the larger combined dataset.
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=1, end=1, by_epoch=True,
         eta_min_ratio=1e-3, convert_to_iter_based=True),
]

# Same softer LR + aggressive freezing as B13.
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    accumulative_counts=1,
    optimizer=dict(type='AdamW', lr=5e-6, betas=(0.9, 0.99),
                   weight_decay=0.01, fused=True),
    paramwise_cfg=dict(custom_keys={
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
    logger=dict(type='LoggerHook', interval=100),
)
