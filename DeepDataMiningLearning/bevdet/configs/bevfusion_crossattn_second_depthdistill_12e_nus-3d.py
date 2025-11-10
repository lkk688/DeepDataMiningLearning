# ===== base =====
# 你可以保留原 base 以继承数据/评测器/默认 hooks，也可以全部在此重写。
# 这里直接用你给的文件为参考，但我们在本文件中覆盖关键部件。
_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

custom_imports = dict(
    imports=[
        # original BEVFusion (registers 'BEVFusion')
        'projects.BEVFusion.bevfusion.bevfusion',

        # your custom view transformer (if any)
        'projects.bevdet.bevfusion.view_transformers.cross_attn',

        # your derived detector (registers 'BEVFusionAuxDepth')
        'projects.bevdet.bevfusion.bevfusion_aux',
    ],
    allow_failed_imports=False,
)

# ===== ranges & modality =====
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

# ===== model: BEVFusion with aux depth (wrap) =====
model = dict(
    type='BEVFusionAuxDepth',   # defined in bevfusion_aux.py
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False
    ),

    # --- image branch: Swin + FPN(4 levels) ---
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[0, 1, 2, 3],  # 4 levels
        with_cp=True,              # gradient checkpoint to save memory
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
        )
    ),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=4,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)
    ),

    # --- view transform: Cross-Attn + sparse depth distill ---
    # view_transform=dict(
    #     type='BEVCrossAttnTransform',  #DepthLSSTransform or BEVCrossAttnTransform projects.BEVFusion.bevfusion.view_transformers.cross_attn.BEVCrossAttnTransform
    #     in_channels=256,
    #     out_channels=256,
    #     xbound=[-54.0, 54.0, 0.3],
    #     ybound=[-54.0, 54.0, 0.3],
    #     zbound=[-10.0, 10.0, 4.0],   # ~5 anchors
    #     num_cams=6,
    #     num_levels=4,
    #     num_points=4,
    #     num_heads=8,
    #     num_layers=2,
    #     depth_distill_weight=0.1,    # 稀疏深度蒸馏损失权重
    #     depth_head_on_level=1,       # 在 stride≈8 的 level 上监督 (Hf≈H/8, Wf≈W/8)
    #     return_aux=True
    # ),
    # view_transform = dict(
    #     type='BEVCrossAttnTransform',
    #     expected_layout='BCNHW',  # 你的实现使用这个
    #     num_levels=3,             # 和 img_neck 的输出层数一致
    #     embed_dims=256,
    #     num_heads=8,
    #     num_points=8,
    #     bev_h=128, bev_w=128,
    #     out_channels=256,
    #     with_reference_2d=True,
    #     depth_distill_weight=0.0,
    #     return_aux=False,
    #     # 不要写 xbound/ybound/zbound/dbound/downsample（那是 LSS 的）
    # )
    view_transform = dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2,
    ),

    # --- LiDAR branch: keep SECOND + SECFPN as in your base ---
    #   这里不显式重写 pts_*，默认继承 base 的 voxel0075+SECOND+SECFPN 设置
    #   如需手动指定，可在此添加 pts_voxel_layer/encoder/middle/backbone/neck

    # --- fusion layer: 两路 256 -> 256 ---
    fusion_layer=dict(type='ConvFuser', in_channels=[256, 256], out_channels=256),
)

# ===== train/test pipelines =====
# 在原 pipeline 基础上，我们只修改 sweeps_num 与保证 meta keys 包含 'lidar2img' / 'img_shape'
train_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color', backend_args=backend_args),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=15, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True, backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ImageAug3D', final_dim=[256, 704], resize_lim=[0.38, 0.55], bot_pct_lim=[0.0, 0.0], rot_lim=[-5.4, 5.4], rand_flip=True, is_train=True),
    dict(type='BEVFusionGlobalRotScaleTrans', scale_ratio_range=[0.9, 1.1], rot_range=[-0.78539816, 0.78539816], translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter',
         classes=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']),
    dict(type='GridMask', use_h=True, use_w=True, max_epoch=6, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.0, fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points','img','gt_bboxes_3d','gt_labels_3d','gt_bboxes','gt_labels'],
        meta_keys=[
            'cam2img','ori_cam2img','lidar2cam','lidar2img','cam2lidar','ori_lidar2img',
            'img_aug_matrix','box_type_3d','sample_idx','lidar_path','img_path','transformation_3d_flow',
            'pcd_rotation','pcd_scale_factor','pcd_trans','lidar_aug_matrix','num_pts_feats','img_shape'
        ])
]
test_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color', backend_args=backend_args),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=15, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True, backend_args=backend_args),
    dict(type='ImageAug3D', final_dim=[256, 704], resize_lim=[0.48, 0.48], bot_pct_lim=[0.0, 0.0], rot_lim=[0.0, 0.0], rand_flip=False, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['img','points','gt_bboxes_3d','gt_labels_3d'],
        meta_keys=['cam2img','ori_cam2img','lidar2cam','lidar2img','cam2lidar','ori_lidar2img','img_aug_matrix','box_type_3d','sample_idx','lidar_path','img_path','num_pts_feats','img_shape']
    )
]

# 用 CBGS 包裹训练数据集（提升长尾）
train_dataloader = dict(
    num_workers=8, persistent_workers=True, pin_memory=True,
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(  # 原本的 train dataset 放在这里
            # 原配置里的 dataset/dataset 结构在不同分支略有不同，保持一致即可：
            # 如果你原来是 train_dataloader = dict(dataset=dict(dataset=dict(...)))
            # 这里就把最内层 dict 放进 dataset= 里
            pipeline=train_pipeline, modality=input_modality
        )
    )
)
val_dataloader = dict(num_workers=8, persistent_workers=True, pin_memory=True, dataset=dict(pipeline=test_pipeline, modality=input_modality))
test_dataloader = val_dataloader

# ===== OPT & SCHED =====
param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=12, end=12, by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=0.85/0.95, begin=0, end=4.8, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=1, begin=4.8, end=12, by_epoch=True, convert_to_iter_based=True),
]

# AMP + AdamW 参数组 & EMA
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(weight_decay=0.0),
        'relative_position_bias_table': dict(weight_decay=0.0),
        'norm': dict(weight_decay=0.0),
    })
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExponentialMovingAverage',
        momentum=0.0002,
        update_buffers=True
    )
]

# runtime
train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# 日志 & ckpt
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=2)
)

# env
env_cfg = dict(cudnn_benchmark=True)
randomness = dict(seed=42, deterministic=False)
auto_scale_lr = dict(enable=False, base_batch_size=32)