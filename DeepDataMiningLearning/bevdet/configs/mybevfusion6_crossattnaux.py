# ==== Base ====
_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# ==== Registry / Scope ====
default_scope = 'mmdet3d'
custom_imports = dict(
    imports=[
        'projects.bevdet.bevfusion',                 # 原 BEVFusion 注册
        'projects.bevdet.cross_attn_lss',     # 你的 CrossAttnLSSTransform
        'projects.bevdet.bevfusion_with_aux', # 带 AUX 的子类（内部用 hook 抓 img BEV）
        'projects.bevdet.freeze_utils'               # FreezeExceptHook
    ],
    allow_failed_imports=False
)

# ==== Env ====
env_cfg = dict(cudnn_benchmark=True)
backend_args = None

# ==== Common ====
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)

# ==== Model ====
model = dict(
    type='BEVFusionWithAux',   # 使用带 AUX 的子类
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=True,  # 省显存
        convert_weights=True,
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

    # --- LSS -> CrossAttnLSSTransform ---
    view_transform=dict(
        type='CrossAttnLSSTransform',
        in_channels=256,
        out_channels=64,            # 从80降到64：更省显存/带宽
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-5.0, 5.0, 10.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2,               # 输出 180x180
        num_z=2,
        use_cam_embed=True,
        attn_chunk=4096,            # 提速（显存允许可再升到 8192）
        debug=False
    ),

    # --- 融合层：同步更新输入通道 ---
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[64, 256],
        out_channels=256
    ),

    # --- AUX: 图像 BEV 辅助监督（权重略小，避免主损失过大） ---
    aux_cfg=dict(
        loss_weight=0.1,
        radius_cells=2
    )
)

# ==== Pipelines ====
train_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color', backend_args=backend_args),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=5,  # 7 -> 5：I/O 与 voxel 更快
        load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True, backend_args=backend_args
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True
    ),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5
    ),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
    ),
    dict(
        type='GridMask',
        use_h=True, use_w=True, max_epoch=6, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.0, fixed_prob=True
    ),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points','img','gt_bboxes_3d','gt_labels_3d','gt_bboxes','gt_labels'],
        meta_keys=[
            'cam2img','ori_cam2img','lidar2cam','lidar2img','cam2lidar',
            'img_aug_matrix','lidar_aug_matrix',
            'box_type_3d','sample_idx','lidar_path','img_path',
            'transformation_3d_flow','pcd_rotation','pcd_scale_factor','pcd_trans',
            'img_aug_matrix','lidar_aug_matrix','num_pts_feats'
        ]
    )
]

test_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color', backend_args=backend_args),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=5, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True, backend_args=backend_args
    ),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False
    ),
    dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='Pack3DDetInputs',
        keys=['img','points','gt_bboxes_3d','gt_labels_3d'],
        meta_keys=['cam2img','ori_cam2img','lidar2cam','lidar2img','cam2lidar','img_aug_matrix','box_type_3d','sample_idx','lidar_path','img_path','num_pts_feats']
    )
]

# ==== Dataloaders ====
train_dataloader = dict(
    batch_size=8,                 # ↑ batch，减少每 epoch 的 iter 数
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    dataset=dict(dataset=dict(pipeline=train_pipeline, modality=input_modality))
)
val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    dataset=dict(pipeline=test_pipeline, modality=input_modality)
)
test_dataloader = val_dataloader

# ==== Scheduler ====
param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0, end=800),
    dict(type='CosineAnnealingLR', begin=0, T_max=6, end=6, by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=0.85/0.95, begin=0, end=2.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=1, begin=2.4, end=6, by_epoch=True, convert_to_iter_based=True)
]

# ==== Runtime ====
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# ==== Optimizer / AMP(BF16) ====
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',          # H100 友好
    accumulative_counts=1,     # 增大 batch 后不再累积
    optimizer=dict(
        type='AdamW',
        lr=0.0002,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        fused=True
    ),
    # 解冻 fusion_layer + bbox_head，但设较小 LR；view_transform 适中；AUX 头正常 LR
    paramwise_cfg=dict(custom_keys={
        'view_transform': dict(lr_mult=0.5),
        'img_aux_head':   dict(lr_mult=1.0),
        'fusion_layer':   dict(lr_mult=0.5),
        'bbox_head':      dict(lr_mult=0.5),
        # 常见不权重衰减参数
        'absolute_pos_embed': dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0),
        'norm': dict(decay_mult=0.0),
    }),
    clip_grad=dict(max_norm=20, norm_type=2)  # 收紧，抑制 grad_norm 过大
)

auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

# ==== Hooks ====
# 移除 base 的自定义 hook，挂冻结 + EMA + 清缓存
del _base_.custom_hooks
custom_hooks = [
    # 只训练 CrossAttn + AUX 头 + 融合/检测头（其余全部冻结，含 BN/LN）
    dict(
        type='FreezeExceptHook',
        allowlist=('view_transform', 'img_aux_head', 'fusion_layer', 'bbox_head'),
        freeze_norm=True,
        verbose=True,
        use_regex=False
    ),
    dict(type='EMAHook', momentum=0.0002, update_buffers=True),
    dict(type='EmptyCacheHook', after_iter=False, after_epoch=True)
]