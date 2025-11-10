_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# ================== Env / Global ==================
# H100: 保持高性能的 cuDNN 算法选择；若想进一步降低 nvidia-smi 占用，可改为 False
env_cfg = dict(cudnn_benchmark=True)
backend_args = None

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)

# ================== Model ==================
model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True  # 与 ImageNet 统计一致
    ),
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,   # 轻量正则，稳定训练
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=True,         # 梯度检查点：省显存
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
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2
    ),
    fusion_layer=dict(type='ConvFuser', in_channels=[80, 256], out_channels=256)
)

# ================== Pipelines ==================
train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),

    # 注意：你的环境未注册 PhotoMetricDistortionMultiViewImage，这里不加入色彩扰动
    # 若后续有 MultiViewWrapper 可安全添加，再行开启

    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(  # 保持原行为（prob=0）
        type='GridMask',
        use_h=True, use_w=True, max_epoch=6, rotate=1,
        offset=False, ratio=0.5, mode=1, prob=0.0, fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

# ================== Dataloaders（H100建议） ==================
# 仅通过配置提升输入吞吐：更多 worker、pin memory、常驻 worker。
train_dataloader = dict(
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(dataset=dict(pipeline=train_pipeline, modality=input_modality))
)
val_dataloader = dict(
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(pipeline=test_pipeline, modality=input_modality)
)
test_dataloader = val_dataloader

# ================== Schedulers ==================
param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0, T_max=6, end=6, by_epoch=True,
        eta_min_ratio=1e-4, convert_to_iter_based=True),
    dict( # momentum
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95, begin=0, end=2.4, by_epoch=True, convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1, begin=2.4, end=6, by_epoch=True, convert_to_iter_based=True)
]

# ================== Runtime ==================
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# ================== Optimizer / AMP(BF16) / Decay ==================
# H100 推荐 BF16（更稳，无需 loss scale），并开启 fused AdamW（PyTorch 2.x）
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',                # 关键：H100 用 BF16 AMP
    optimizer=dict(
        type='AdamW',
        lr=0.0002,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        fused=True                  # 关键：使用 fused AdamW
    ),
    # 对 Swin 常见权重禁用 wd
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# ================== Misc ==================
auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

# 基线中可能带有自定义 hooks；先删除再安全添加 EMA
del _base_.custom_hooks
custom_hooks = [
    dict(type='EMAHook', momentum=0.0002, update_buffers=True)
]

"""
11/08 16:14:51 - mmengine - INFO - Epoch(train) [1][   50/30895]  base_lr: 7.9760e-05 lr: 7.9760e-05  eta: 2 days, 12:24:06  time: 1.1734  data_time: 0.1148  memory: 14448  grad_norm: 3874.2796  loss: 657.5450  loss_heatmap: 635.5471  layer_-1_loss_cls: 9.8937  layer_-1_loss_bbox: 12.1043  matched_ious: 0.0043
11/08 16:15:39 - mmengine - INFO - Epoch(train) [1][  100/30895]  base_lr: 9.3120e-05 lr: 9.3120e-05  eta: 2 days, 6:48:10  time: 0.9564  data_time: 0.0202  memory: 11601  grad_norm: 114.3170  loss: 29.1531  loss_heatmap: 9.7790  layer_-1_loss_cls: 7.3424  layer_-1_loss_bbox: 12.0317  matched_ious: 0.0176
11/08 16:16:26 - mmengine - INFO - Epoch(train) [1][  150/30895]  base_lr: 1.0648e-04 lr: 1.0648e-04  eta: 2 days, 4:42:46  time: 0.9439  data_time: 0.0186  memory: 11219  grad_norm: 33.6501  loss: 15.3590  loss_heatmap: 3.1677  layer_-1_loss_cls: 5.4654  layer_-1_loss_bbox: 6.7259  matched_ious: 0.0345
11/08 16:17:18 - mmengine - INFO - Epoch(train) [1][  200/30895]  base_lr: 1.1984e-04 lr: 1.1984e-04  eta: 2 days, 4:49:51  time: 1.0348  data_time: 0.0241  memory: 11400  grad_norm: 48.6302  loss: 13.0587  loss_heatmap: 3.0497  layer_-1_loss_cls: 4.1779  layer_-1_loss_bbox: 5.8311  matched_ious: 0.0433
11/08 16:18:10 - mmengine - INFO - Epoch(train) [1][  250/30895]  base_lr: 1.3320e-04 lr: 1.3320e-04  eta: 2 days, 4:55:59  time: 1.0385  data_time: 0.0252  memory: 11384  grad_norm: 23.3042  loss: 10.1187  loss_heatmap: 2.7791  layer_-1_loss_cls: 2.6707  layer_-1_loss_bbox: 4.6690  matched_ious: 0.0487
11/08 16:18:57 - mmengine - INFO - Epoch(train) [1][  300/30895]  base_lr: 1.4656e-04 lr: 1.4656e-04  eta: 2 days, 4:05:45  time: 0.9333  data_time: 0.0180  memory: 11321  grad_norm: 27.2718  loss: 9.8010  loss_heatmap: 2.6955  layer_-1_loss_cls: 2.3535  layer_-1_loss_bbox: 4.7519  matched_ious: 0.0704
11/08 16:19:44 - mmengine - INFO - Epoch(train) [1][  350/30895]  base_lr: 1.5992e-04 lr: 1.5992e-04  eta: 2 days, 3:33:49  time: 0.9428  data_time: 0.0188  memory: 11194  grad_norm: 31.8102  loss: 8.8697  loss_heatmap: 2.6848  layer_-1_loss_cls: 1.9935  layer_-1_loss_bbox: 4.1914  matched_ious: 0.0868
11/08 16:20:32 - mmengine - INFO - Epoch(train) [1][  400/30895]  base_lr: 1.7328e-04 lr: 1.7328e-04  eta: 2 days, 3:19:08  time: 0.9674  data_time: 0.0199  memory: 11213  grad_norm: 25.7979  loss: 8.1235  loss_heatmap: 2.5964  layer_-1_loss_cls: 1.3321  layer_-1_loss_bbox: 4.1950  matched_ious: 0.0630
11/08 16:21:20 - mmengine - INFO - Epoch(train) [1][  450/30895]  base_lr: 1.8664e-04 lr: 1.8664e-04  eta: 2 days, 2:59:55  time: 0.9451  data_time: 0.0175  memory: 11442  grad_norm: 13.2402  loss: 7.6915  loss_heatmap: 2.5329  layer_-1_loss_cls: 0.9877  layer_-1_loss_bbox: 4.1709  matched_ious: 0.0618
11/08 16:22:06 - mmengine - INFO - Epoch(train) [1][  500/30895]  base_lr: 2.0000e-04 lr: 2.0000e-04  eta: 2 days, 2:41:35  time: 0.9360  data_time: 0.0169  memory: 11241  grad_norm: 8.1660  loss: 6.9481  loss_heatmap: 2.4541  layer_-1_loss_cls: 0.6887  layer_-1_loss_bbox: 3.8053  matched_ious: 0.0591
11/08 16:22:53 - mmengine - INFO - Epoch(train) [1][  550/30895]  base_lr: 2.0000e-04 lr: 2.0000e-04  eta: 2 days, 2:24:24  time: 0.9287  data_time: 0.0169  memory: 11768  grad_norm: 9.4336  loss: 6.5353  loss_heatmap: 2.4784  layer_-1_loss_cls: 0.6748  layer_-1_loss_bbox: 3.3822  matched_ious: 0.1052
11/08 16:23:39 - mmengine - INFO - Epoch(train) [1][  600/30895]  base_lr: 1.9999e-04 lr: 1.9999e-04  eta: 2 days, 2:06:36  time: 0.9157  data_time: 0.0160  memory: 11532  grad_norm: 8.3385  loss: 6.3760  loss_heatmap: 2.3995  layer_-1_loss_cls: 0.5629  layer_-1_loss_bbox: 3.4136  matched_ious: 0.0687
11/08 16:24:25 - mmengine - INFO - Epoch(train) [1][  650/30895]  base_lr: 1.9999e-04 lr: 1.9999e-04  eta: 2 days, 1:56:17  time: 0.9362  data_time: 0.0173  memory: 11313  grad_norm: 8.2241  loss: 6.3856  loss_heatmap: 2.3667  layer_-1_loss_cls: 0.4882  layer_-1_loss_bbox: 3.5307  matched_ious: 0.0713
11/08 16:25:13 - mmengine - INFO - Epoch(train) [1][  700/30895]  base_lr: 1.9999e-04 lr: 1.9999e-04  eta: 2 days, 1:49:20  time: 0.9453  data_time: 0.0225  memory: 11491  grad_norm: 8.3774  loss: 6.2872  loss_heatmap: 2.3456  layer_-1_loss_cls: 0.4912  layer_-1_loss_bbox: 3.4504  matched_ious: 0.1157
11/08 16:26:00 - mmengine - INFO - Epoch(train) [1][  750/30895]  base_lr: 1.9999e-04 lr: 1.9999e-04  eta: 2 days, 1:41:45  time: 0.9382  data_time: 0.0162  memory: 11370  grad_norm: 8.1959  loss: 6.1650  loss_heatmap: 2.2895  layer_-1_loss_cls: 0.4512  layer_-1_loss_bbox: 3.4243  matched_ious: 0.1068
11/08 16:26:46 - mmengine - INFO - Epoch(train) [1][  800/30895]  base_lr: 1.9999e-04 lr: 1.9999e-04  eta: 2 days, 1:32:08  time: 0.9232  data_time: 0.0154  memory: 11402  grad_norm: 9.1624  loss: 5.5875  loss_heatmap: 2.2093  layer_-1_loss_cls: 0.4301  layer_-1_loss_bbox: 2.9481  matched_ious: 0.1303

"""