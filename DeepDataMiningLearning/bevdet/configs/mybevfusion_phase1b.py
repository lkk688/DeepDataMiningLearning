
# =====================================================================
# Phase 1b: BEVFusion + CrossAttn LSS + Occupancy auxiliary loss
#
# Changes from mybevfusion12v2.py:
#   1. frozen_stages 4 → 2  (unfreeze Swin stages 2 & 3; low LR 0.1x)
#   2. Camera BEV channels 160 → 256  (wider camera representation)
#   3. GridMask prob 0.0 → 0.7  (enable the augmentation properly)
#   4. max_epochs 10 → 20  (full cosine cycle)
#   5. EMAHook enabled  (momentum=0.0002)
#   6. FreezeExceptHook removed  (paramwise_cfg + frozen_stages is sufficient)
#   7. load_from → mybevfusion12v2/epoch_8.pth (warm-start from best v2 ckpt)
#
# Training command (with occupancy auxiliary head):
#   bash run_phase1b.sh
#
# Expected NDS improvement over v2:  +2~4 points → ~0.70-0.72
# =====================================================================

_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# ---- Feature toggles ---------------------------------------------------
voxel_painting_on = True
use_rpf_img       = True
use_rpf_lidar     = True

# ---- Registry ---------------------------------------------------------
default_scope = 'mmdet3d'
custom_imports = dict(
    imports=[
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.bevfusion_ca',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.freeze_utils',
        'projects.bevdet.fire_rpf_necks',
    ],
    allow_failed_imports=not (use_rpf_img or use_rpf_lidar)
)

# ---- Environment -------------------------------------------------------
env_cfg = dict(cudnn_benchmark=True)
backend_args = None

# ---- Geometry ---------------------------------------------------------
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)

# ---- Model ------------------------------------------------------------
model = dict(
    type='BEVFusionCA',
    voxel_painting_on=voxel_painting_on,
    use_two_scale_tokens=True,

    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True,
        voxelize_cfg=dict(
            voxelize_reduce=True,
            max_num_points=10,
            voxel_size=[0.075, 0.075, 0.2],
            point_cloud_range=point_cloud_range
        )
    ),

    # ---- IMAGE BRANCH -------------------------------------------------
    # Change 1: frozen_stages 4 → 2
    # Stages 0-1 stay frozen (low-level edges/textures, well-pretrained).
    # Stages 2-3 are unfrozen with lr_mult=0.1 in paramwise_cfg below.
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.3,
        patch_norm=True, out_indices=[1, 2, 3],
        with_cp=True, convert_weights=True,
        frozen_stages=2,   # ← was 4; unfreeze stages 2 & 3
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
        )
    ),

    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)
    ),

    # Change 2: camera BEV channels 160 → 256
    view_transform=dict(
        type='CrossAttnLSSTransform',
        in_channels=256,
        out_channels=256,          # ← was 160
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-5.0, 5.0, 10.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2,
        num_z=6,
        use_cam_embed=True,
        attn_chunk=16384,
        ms_extra_in_channels=[256],
        ms_fuse_mode="gated_add",
        ms_align_corners=False,
        debug=False
    ),

    # Change 2 (continued): fusion_layer input updated to match 256-ch cam BEV
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[256, 256],    # ← was [160, 256]
        out_channels=256
    ),
    bbox_head=dict(in_channels=256),

    aux_cfg=dict(loss_weight=0.15, radius_cells=2, loss_type='focal_dice'),
)

# ---- LiDAR voxelization -----------------------------------------------
pts_voxel_layer = dict(
    max_num_points=10,
    voxel_size=[0.075, 0.075, 0.2],
    point_cloud_range=point_cloud_range
)

if voxel_painting_on:
    pts_voxel_encoder = dict(
        type='PaintedWrapperVFE',
        base_vfe=dict(type='HardSimpleVFE', num_features=5),
        point_cloud_range=point_cloud_range,
        voxel_size=[0.075, 0.075, 0.2],
        image_size=[256, 704],
        feature_size=[32, 88],
        img_feat_level=0,
        cam_pool='avg',
        img_feat_out=96,
        fuse='gated',
        detach_img=True,
        align_corners=True,
        chunk_voxels=200000
    )
else:
    pts_voxel_encoder = dict(type='HardSimpleVFE', num_features=5)

model.update(dict(
    pts_voxel_layer=pts_voxel_layer,
    pts_voxel_encoder=pts_voxel_encoder,
))

if use_rpf_lidar:
    model.update(dict(
        pts_neck=dict(
            type='FireRPFNeck',
            in_channels=[128, 256],
            out_channels=256,
            num_outs=2,
            upsample_strides=[1, 2],
            blocks_per_stage=3,
            with_residual=True,
            use_cbam=True, ca_reduction=16, sa_kernel=7,
        )
    ))

if use_rpf_img:
    model.update(dict(
        img_rpf_neck=dict(
            type='FireRPF2DNeck',
            in_channels=[256, 256, 256],
            out_channels=256,
            num_outs=3,
            blocks_per_stage=1,
            with_residual=True,
            use_cbam=True, ca_reduction=16, sa_kernel=7,
        )
    ))

# ---- Pipelines --------------------------------------------------------
# Change 3: GridMask prob 0.0 → 0.7 (re-enable this augmentation)
train_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color', backend_args=backend_args),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True, backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ImageAug3D', final_dim=[256, 704], resize_lim=[0.38, 0.55], bot_pct_lim=[0.0, 0.0], rot_lim=[-5.4, 5.4], rand_flip=True, is_train=True),
    dict(type='BEVFusionGlobalRotScaleTrans', scale_ratio_range=[0.9, 1.1], rot_range=[-0.78539816, 0.78539816], translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']),
    dict(type='GridMask', use_h=True, use_w=True, max_epoch=20, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7, fixed_prob=True),  # ← was prob=0.0
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'box_type_3d', 'sample_idx', 'lidar_path', 'img_path',
            'transformation_3d_flow', 'pcd_rotation', 'pcd_scale_factor', 'pcd_trans',
            'img_aug_matrix', 'lidar_aug_matrix', 'num_pts_feats'
        ]
    )
]
test_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color', backend_args=backend_args),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True, backend_args=backend_args),
    dict(type='ImageAug3D', final_dim=[256, 704], resize_lim=[0.48, 0.48], bot_pct_lim=[0.0, 0.0], rot_lim=[0.0, 0.0], rand_flip=False, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar', 'img_aug_matrix',
                    'box_type_3d', 'sample_idx', 'lidar_path', 'img_path', 'num_pts_feats'])
]

train_dataloader = dict(
    batch_size=16, num_workers=16, persistent_workers=True, pin_memory=True, prefetch_factor=4,
    dataset=dict(dataset=dict(pipeline=train_pipeline, modality=input_modality))
)
val_dataloader = dict(
    batch_size=1, num_workers=16, persistent_workers=True, pin_memory=True, prefetch_factor=4,
    dataset=dict(pipeline=test_pipeline, modality=input_modality)
)
test_dataloader = val_dataloader

# ---- Scheduler (Change 4: 20 epochs) ----------------------------------
param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0, end=800),
    dict(type='CosineAnnealingLR', begin=0, T_max=20, end=20, by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=0.85/0.95, begin=0, end=4.8, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=1, begin=4.8, end=20, by_epoch=True, convert_to_iter_based=True),
]
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# ---- Optimizer (Change 1 continued: backbone stages 2/3 at lr_mult=0.1) ---
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    accumulative_counts=1,
    optimizer=dict(type='AdamW', lr=0.0002, betas=(0.9, 0.99), weight_decay=0.01, fused=True),
    paramwise_cfg=dict(custom_keys={
        # Camera side
        'view_transform':    dict(lr_mult=0.7),
        'img_aux_head':      dict(lr_mult=1.0),
        'fusion_layer':      dict(lr_mult=0.7),
        'bbox_head':         dict(lr_mult=0.7),
        # LiDAR side
        'pts_voxel_encoder': dict(lr_mult=0.7),
        'pts_middle_encoder.encoder_layers.0': dict(lr_mult=0.5),
        'pts_middle_encoder.encoder_layers.1': dict(lr_mult=0.5),
        # RPF necks
        'img_rpf_neck':      dict(lr_mult=1.5),
        'pts_neck':          dict(lr_mult=1.2),
        # Swin: frozen stages 0-1 (lr_mult=0); unfrozen stages 2-3 (lr_mult=0.1)
        'img_backbone.patch_embed':              dict(lr_mult=0.0, decay_mult=0.0),
        'img_backbone.stages.0':                 dict(lr_mult=0.0, decay_mult=0.0),
        'img_backbone.stages.1':                 dict(lr_mult=0.0, decay_mult=0.0),
        'img_backbone.stages.2':                 dict(lr_mult=0.1),   # ← unfrozen, fine-tune gently
        'img_backbone.stages.3':                 dict(lr_mult=0.1),   # ← unfrozen, fine-tune gently
        # Standard decay exceptions
        'absolute_pos_embed':                    dict(decay_mult=0.0),
        'relative_position_bias_table':          dict(decay_mult=0.0),
        'norm':                                  dict(decay_mult=0.0),
    }),
    clip_grad=dict(max_norm=20, norm_type=2)
)

auto_scale_lr = dict(enable=False, base_batch_size=32)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

# ---- Custom hooks (Change 4: EMA enabled; FreezeExceptHook removed) ---
try:
    del _base_.custom_hooks
except Exception:
    pass

custom_hooks = [
    dict(type='EMAHook', momentum=0.0002, update_buffers=True),  # ← enabled
    dict(type='EmptyCacheHook', after_iter=False, after_epoch=True),
]

# Warm-start from best mybevfusion12v2 checkpoint
load_from = 'work_dirs/mybevfusion12v2/epoch_8.pth'
load_cfg = dict(strict=False)
auto_resume = False
resume = False
