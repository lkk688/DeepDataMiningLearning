_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# --------- Global / Env ---------
env_cfg = dict(cudnn_benchmark=True)
backend_args = None

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)

# --------- Model ---------
model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True   # 修正为RGB，匹配ImageNet均值方差
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
        drop_path_rate=0.3,   # 轻量正则
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

# --------- Pipelines ---------
train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),

    # ⚠️ 撤回会触发报错的 PhotoMetricDistortionMultiViewImage
    # 如需色彩扰动，等你环境里有 MultiView 包装器时再加（见下方备注）

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
        use_h=True,
        use_w=True,
        max_epoch=6,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
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

train_dataloader = dict(
    dataset=dict(dataset=dict(pipeline=train_pipeline, modality=input_modality)))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, modality=input_modality))
test_dataloader = val_dataloader

# --------- Schedulers ---------
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

# --------- Runtime ---------
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# --------- Optimizer / AMP / Decay Tuning ---------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=0.0002,
        betas=(0.9, 0.99),
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

# 移除base中的自定义hooks，再新增EMA
del _base_.custom_hooks
custom_hooks = [
    dict(type='EMAHook', momentum=0.0002, update_buffers=True)
]


"""

11/08 15:58:16 - mmengine - INFO - Epoch(train) [1][   50/30895]  base_lr: 7.9760e-05 lr: 7.9760e-05  eta: 2 days, 7:32:05  time: 1.0788  data_time: 0.0849  memory: 15079  grad_norm: nan  loss: 859.5978  loss_heatmap: 836.5007  layer_-1_loss_cls: 10.1055  layer_-1_loss_bbox: 12.9917  matched_ious: 0.0001
11/08 15:59:05 - mmengine - INFO - Epoch(train) [1][  100/30895]  base_lr: 9.3120e-05 lr: 9.3120e-05  eta: 2 days, 5:05:24  time: 0.9844  data_time: 0.0376  memory: 10971  grad_norm: 128.4062  loss: 31.9162  loss_heatmap: 11.3917  layer_-1_loss_cls: 7.3698  layer_-1_loss_bbox: 13.1547  matched_ious: 0.0112
11/08 15:59:53 - mmengine - INFO - Epoch(train) [1][  150/30895]  base_lr: 1.0648e-04 lr: 1.0648e-04  eta: 2 days, 3:56:07  time: 0.9651  data_time: 0.0376  memory: 10947  grad_norm: 29.8001  loss: 14.8121  loss_heatmap: 3.0683  layer_-1_loss_cls: 5.7256  layer_-1_loss_bbox: 6.0182  matched_ious: 0.0385
11/08 16:00:42 - mmengine - INFO - Epoch(train) [1][  200/30895]  base_lr: 1.1984e-04 lr: 1.1984e-04  eta: 2 days, 3:32:25  time: 0.9798  data_time: 0.0373  memory: 10927  grad_norm: 16.5184  loss: 12.1014  loss_heatmap: 2.8777  layer_-1_loss_cls: 4.0052  layer_-1_loss_bbox: 5.2185  matched_ious: 0.0355
11/08 16:01:31 - mmengine - INFO - Epoch(train) [1][  250/30895]  base_lr: 1.3320e-04 lr: 1.3320e-04  eta: 2 days, 3:12:52  time: 0.9717  data_time: 0.0371  memory: 10957  grad_norm: 17.6786  loss: 10.2420  loss_heatmap: 2.7602  layer_-1_loss_cls: 3.0032  layer_-1_loss_bbox: 4.4785  matched_ious: 0.0413
11/08 16:02:19 - mmengine - INFO - Epoch(train) [1][  300/30895]  base_lr: 1.4656e-04 lr: 1.4656e-04  eta: 2 days, 2:56:45  time: 0.9662  data_time: 0.0376  memory: 10986  grad_norm: 16.9440  loss: 9.7703  loss_heatmap: 2.6961  layer_-1_loss_cls: 2.3908  layer_-1_loss_bbox: 4.6834  matched_ious: 0.0520
11/08 16:03:08 - mmengine - INFO - Epoch(train) [1][  350/30895]  base_lr: 1.5992e-04 lr: 1.5992e-04  eta: 2 days, 2:51:48  time: 0.9817  data_time: 0.0377  memory: 10941  grad_norm: 14.5549  loss: 9.1276  loss_heatmap: 2.6166  layer_-1_loss_cls: 1.7355  layer_-1_loss_bbox: 4.7754  matched_ious: 0.0541
11/08 16:03:57 - mmengine - INFO - Epoch(train) [1][  400/30895]  base_lr: 1.7328e-04 lr: 1.7328e-04  eta: 2 days, 2:43:44  time: 0.9709  data_time: 0.0379  memory: 10982  grad_norm: 9.9027  loss: 7.5579  loss_heatmap: 2.6390  layer_-1_loss_cls: 1.2017  layer_-1_loss_bbox: 3.7173  matched_ious: 0.0975
11/08 16:04:45 - mmengine - INFO - Epoch(train) [1][  450/30895]  base_lr: 1.8664e-04 lr: 1.8664e-04  eta: 2 days, 2:35:20  time: 0.9652  data_time: 0.0379  memory: 10951  grad_norm: 10.3536  loss: 6.8737  loss_heatmap: 2.4743  layer_-1_loss_cls: 0.9390  layer_-1_loss_bbox: 3.4604  matched_ious: 0.0852
11/08 16:05:33 - mmengine - INFO - Epoch(train) [1][  500/30895]  base_lr: 2.0000e-04 lr: 2.0000e-04  eta: 2 days, 2:30:27  time: 0.9716  data_time: 0.0380  memory: 11011  grad_norm: 9.6390  loss: 6.5293  loss_heatmap: 2.4880  layer_-1_loss_cls: 0.7093  layer_-1_loss_bbox: 3.3321  matched_ious: 0.1163
11/08 16:06:22 - mmengine - INFO - Epoch(train) [1][  550/30895]  base_lr: 2.0000e-04 lr: 2.0000e-04  eta: 2 days, 2:25:38  time: 0.9693  data_time: 0.0363  memory: 11054  grad_norm: 8.3990  loss: 6.4755  loss_heatmap: 2.4236  layer_-1_loss_cls: 0.5970  layer_-1_loss_bbox: 3.4550  matched_ious: 0.0782
11/08 16:07:11 - mmengine - INFO - Epoch(train) [1][  600/30895]  base_lr: 1.9999e-04 lr: 1.9999e-04  eta: 2 days, 2:24:22  time: 0.9805  data_time: 0.0371  memory: 10998  grad_norm: 8.7161  loss: 6.3269  loss_heatmap: 2.3245  layer_-1_loss_cls: 0.5186  layer_-1_loss_bbox: 3.4838  matched_ious: 0.0897
11/08 16:07:59 - mmengine - INFO - Epoch(train) [1][  650/30895]  base_lr: 1.9999e-04 lr: 1.9999e-04  eta: 2 days, 2:19:41  time: 0.9658  data_time: 0.0370  memory: 11115  grad_norm: 8.3814  loss: 6.1764  loss_heatmap: 2.3287  layer_-1_loss_cls: 0.5102  layer_-1_loss_bbox: 3.3375  matched_ious: 0.0852
11/08 16:08:48 - mmengine - INFO - Epoch(train) [1][  700/30895]  base_lr: 1.9999e-04 lr: 1.9999e-04  eta: 2 days, 2:17:43  time: 0.9756  data_time: 0.0368  memory: 10957  grad_norm: 8.0454  loss: 5.6703  loss_heatmap: 2.2075  layer_-1_loss_cls: 0.4567  layer_-1_loss_bbox: 3.0061  matched_ious: 0.0606
11/08 16:09:37 - mmengine - INFO - Epoch(train) [1][  750/30895]  base_lr: 1.9999e-04 lr: 1.9999e-04  eta: 2 days, 2:17:23  time: 0.9828  data_time: 0.0369  memory: 10943  grad_norm: 8.7286  loss: 5.4142  loss_heatmap: 2.1000  layer_-1_loss_cls: 0.4265  layer_-1_loss_bbox: 2.8877  matched_ious: 0.0758
11/08 16:10:26 - mmengine - INFO - Epoch(train) [1][  800/30895]  base_lr: 1.9999e-04 lr: 1.9999e-04  eta: 2 days, 2:14:52  time: 0.9718  data_time: 0.0365  memory: 10999  grad_norm: 8.6331  loss: 5.5433  loss_heatmap: 2.1741  layer_-1_loss_cls: 0.4269  layer_-1_loss_bbox: 2.9423  matched_ious: 0.0707
"""