# ==== Base ====
_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# ==== Registry / Scope ====
default_scope = 'mmdet3d'
custom_imports = dict(
    imports=[
        'projects.bevdet.bevfusion',              # 注册 BEVFusion
        'projects.bevdet.cross_attn_lss',  # 你的 CrossAttnLSSTransform 源文件
        'projects.bevdet.bevfusion_with_aux',
        'projects.bevdet.freeze_utils'            # FreezeExceptHook
    ],
    allow_failed_imports=False
)

# ==== Env ====
env_cfg = dict(cudnn_benchmark=True)  # 更快；若要更低 reserved 显存可改 False
backend_args = None

# ==== Common ====
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)

# ==== Model ====
model = dict(
    type='BEVFusionWithAux', #'BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True    # 与 ImageNet 统计一致
    ),
    # --- Image backbone / neck (保持一致，仅做省显存) ---
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
        drop_path_rate=0.3,   # 小幅正则
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=True,         # 梯度检查点省显存
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

    # --- 替换 LSS -> CrossAttnLSSTransform（输出与接口保持一致） ---
    view_transform=dict(
        type='CrossAttnLSSTransform',
        in_channels=256,
        out_channels=64,           # 从80降到64：更省显存/带宽
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-5.0, 5.0, 10.0],  # 用于高度锚点范围
        dbound=[1.0, 60.0, 0.5],   # 仅作为K推导参考
        downsample=2,              # 与 LiDAR BEV 对齐，输出 180x180
        num_z=2,                   # K：越小越省显存（2~3常用）
        use_cam_embed=True,
        attn_chunk=2048,           # 显存/速度权衡；显存足可 4096/8192
        debug=False                # 关闭内部调试打印
    ),

    # --- 融合层：同步更新输入通道（从80改为64） ---
    fusion_layer=dict(
        type='ConvFuser', in_channels=[64, 256], out_channels=256
    ),
    
    # << 新增 AUX 配置
    aux_cfg=dict(
        loss_weight=0.2,     # AUX loss 权重
        radius_cells=2       # BEV 高斯半径（以 cell 为单位）
    )
)

# ==== Pipelines ====
train_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color', backend_args=backend_args),
    # 不加入 PhotoMetricDistortionMultiViewImage（避免注册缺失）；如后续有 MultiViewWrapper 再开启
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=7,   # 9 -> 7：I/O 与内存更友好
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
            'cam2img','ori_cam2img','lidar2cam','lidar2img','cam2lidar','ori_lidar2img',
            'img_aug_matrix','box_type_3d','sample_idx','lidar_path','img_path',
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
        sweeps_num=7, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True, backend_args=backend_args
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
        meta_keys=['cam2img','ori_cam2img','lidar2cam','lidar2img','cam2lidar','ori_lidar2img','img_aug_matrix','box_type_3d','sample_idx','lidar_path','img_path','num_pts_feats']
    )
]

# ==== Dataloaders ====
train_dataloader = dict(
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    dataset=dict(dataset=dict(pipeline=train_pipeline, modality=input_modality))
)
val_dataloader = dict(
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    dataset=dict(pipeline=test_pipeline, modality=input_modality)
)
test_dataloader = val_dataloader

# ==== Scheduler ====
param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0, end=800),  # warmup 略延长更稳
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
    dtype='bfloat16',           # H100 推荐
    accumulative_counts=2,      # 梯度累积：削峰显存
    optimizer=dict(
        type='AdamW',
        lr=0.0002,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        fused=True              # fused AdamW（PyTorch 2.x）
    ),
    # 对 Swin 常见项不做 wd
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0),
        'norm': dict(decay_mult=0.0),
    }),
    clip_grad=dict(max_norm=35, norm_type=2)
)

auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

# ==== Hooks ====
# 移除 base 中可能的自定义 hooks，再挂载冻结+EMA+空缓存
del _base_.custom_hooks
custom_hooks = [
    dict(
        type='FreezeExceptHook',
        allowlist=('view_transform', 'img_aux_head'),  # << 把 AUX 头也加入白名单
        freeze_norm=True,
        verbose=True,
        use_regex=False
    ),
    dict(type='EMAHook', momentum=0.0002, update_buffers=True),
    dict(type='EmptyCacheHook', after_iter=False, after_epoch=True)
]