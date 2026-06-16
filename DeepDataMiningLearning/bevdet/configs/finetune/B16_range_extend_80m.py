# =====================================================================
# B16 — Range-extend to ±80 m of B14's mixed (Waymo + nuScenes) recipe.
#
# Motivation: Waymo LiDAR returns objects out to ~75 m; B14/B10c were
# clipping at ±54 m and the Waymo eval recall on the 30-75m ring is
# strictly bounded by the data pipeline range filter. This config extends
# the point-cloud-range to ±80.4 m (chosen as 16·134·0.075 so the grid
# stays divisible by 16 — required by 4-level sparse-encoder downsample
# and SECOND backbone strides; paper-rounded "±80 m").
#
# Architecture changes vs B14 (all auto-derived from PCR + voxel_size):
#   - voxelize_cfg.point_cloud_range : ±54 → ±80.4
#   - voxelize_cfg.max_voxels        : [120k, 160k] → [240k, 320k]
#   - sparse_shape                   : [1440, 1440, 41] → [2144, 2144, 41]
#   - view_transform.x/ybound        : [-54, 54, 0.3] → [-80.4, 80.4, 0.3]
#   - bbox_head train_cfg/test_cfg/bbox_coder pc_range / grid_size
#   - post_center_range              : ±61.2 → ±88.4 (1.1× PCR_xy)
#   - pipeline PointsRangeFilter / ObjectRangeFilter PCR
#
# Memory: BEV feature map area grows 2.22× (180² → 268²). To stay within
# H100 80GB at bf16, batch_size 8 → 4 with accumulative_counts=2 to keep
# effective batch=8. ~1 epoch ≈ 6-8h.
#
# Warm-start: load_from B10c epoch_3 (PCR=±54). Conv weights are
# spatially-shared so all kernels load; the head was trained to predict
# objects only within ±54m — outer ring (54-80m) is initially un-supervised
# territory the model has to learn during this fine-tune.
# =====================================================================

_base_ = ['./B14_waymo_nuscenes_mixed.py']

# ---- New range / grid -----------------------------------------------------
# Chosen so grid_size = extent / voxel_size = 160.8 / 0.075 = 2144 = 16·134
# (must be divisible by 16 for clean 4-stage sparse-encoder downsample).
point_cloud_range = [-80.4, -80.4, -5.0, 80.4, 80.4, 3.0]
voxel_size = [0.075, 0.075, 0.2]
sparse_shape = [2144, 2144, 41]
post_center_range = [-88.4, -88.4, -10.0, 88.4, 88.4, 10.0]

# ---- Model overrides ------------------------------------------------------
model = dict(
    data_preprocessor=dict(
        voxelize_cfg=dict(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            # ~2× the voxel budget; LiDAR returns at 80m are sparser than at
            # 54m so this is conservative.
            max_voxels=[240000, 320000],
        ),
    ),
    pts_middle_encoder=dict(sparse_shape=sparse_shape),
    view_transform=dict(
        xbound=[-80.4, 80.4, 0.3],
        ybound=[-80.4, 80.4, 0.3],
        # zbound, dbound, image_size unchanged — only BEV target grows.
    ),
    bbox_head=dict(
        train_cfg=dict(
            point_cloud_range=point_cloud_range,
            grid_size=sparse_shape,
            voxel_size=voxel_size,
        ),
        test_cfg=dict(
            grid_size=sparse_shape,
            voxel_size=[voxel_size[0], voxel_size[1]],
            pc_range=[point_cloud_range[0], point_cloud_range[1]],
        ),
        bbox_coder=dict(
            pc_range=[point_cloud_range[0], point_cloud_range[1]],
            post_center_range=post_center_range,
            voxel_size=[voxel_size[0], voxel_size[1]],
        ),
    ),
)

# ---- Override data pipelines (Waymo + nuScenes) to use new PCR ------------
# B14 builds pipelines from `point_cloud_range` at config load time, so the
# inherited pipelines still reference the OLD ±54m value (closure-captured).
# We rebuild both pipelines explicitly.

backend_args = None
class_names = _base_.class_names
class_palette = _base_.class_palette
common_metainfo = _base_.common_metainfo
input_modality = _base_.input_modality

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

# Waymo pipeline rebuild — clone B13's structure with the new PCR.
# `_base_` here is B14, which inherits B13's `train_pipeline` and
# `data_root_waymo` via its own `_base_` chain — both are visible as
# direct attributes on the resolved ConfigDict.
from copy import deepcopy
waymo_train_pipeline = deepcopy(_base_.train_pipeline)
for t in waymo_train_pipeline:
    if t.get('type') in ('PointsRangeFilter', 'ObjectRangeFilter'):
        t['point_cloud_range'] = point_cloud_range

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
    data_root='data/nuscenes/',
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
    # Halve batch + 2× accum to keep effective batch=8 with ~2.2× BEV memory.
    batch_size=4, num_workers=4, persistent_workers=True, pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[waymo_dataset, nus_dataset],
        ignore_keys=['version', 'dataset', 'categories', 'info_version',
                     'sample_idx_to_data_idx'],
    ),
)

# Effective batch = 4 × 2 = 8 (matches B14).
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    accumulative_counts=2,
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

# 1 epoch — match B14's schedule. Decide whether to extend after eval.
train_cfg = dict(by_epoch=True, max_epochs=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=1, end=1, by_epoch=True,
         eta_min_ratio=1e-3, convert_to_iter_based=True),
]

load_from = '/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/ablation_B10c/epoch_3.pth'
resume = False

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2,
                    save_best='auto'),
    logger=dict(type='LoggerHook', interval=100),
)
