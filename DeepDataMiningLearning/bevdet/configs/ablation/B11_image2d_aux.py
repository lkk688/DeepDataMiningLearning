# =====================================================================
# B11 = B10c + per-camera 2D detection aux head.
#
# Motivation
# ----------
# Across B7..B10c every "temporal" intervention has landed within
# ±0.005 NDS of B5. EXPERIMENTS.md §5.9 identifies one untouched
# structural axis: the image branch is supervised *only* through the
# 3D-detection loss, which propagates back through a long chain
# (bbox_head → pts_neck → pts_backbone → fusion → CA-LSS → img_neck)
# before reaching the Swin features. The signal is heavily diluted by
# the LiDAR side.
#
# B11 adds an FPN-P3 2D detection head per camera, supervised by
# nuScenes' annotated 2D boxes (info['cam_instances']). The head is
# small (~70 k params), zero-deploy, gradient-only contribution at
# inference. Same "direct supervision channel" idea that finally let
# the temporal block train in B10c — applied to the image branch here.
#
# Inherits everything from B10c (FG-TCA + temporal aux + real-multi-sweep
# data) and only:
#   * Switches the detector class to ``Image2DAuxBEVFusion``
#     (a thin subclass of FlowGuidedTemporalBEVFusion).
#   * Rewrites train/test pipelines to insert ``LoadCamInstances2D``
#     after ``ImageAug3D`` and add ``cam_inst_2d`` to Pack3DDetInputs
#     meta_keys so the head can read 2D GT in ``loss()``.
# =====================================================================

_base_ = ['./B10c_flow_guided_warmstart_fixed.py']

custom_imports = dict(
    imports=[
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.bevfusion_ca',
        'projects.bevdet.multitoken_temporal_lidar',
        'projects.bevdet.flow_guided_temporal',
        'projects.bevdet.image2d_aux',
    ],
    allow_failed_imports=False,
)

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
backend_args = None
image_size = [256, 704]

# Detector class switch + 2D aux config.
model = dict(
    type='Image2DAuxBEVFusion',
    # FG-TCA fields inherited from B10c (num_buckets, window_seconds,
    # flow_loss_weight, temporal_aux_loss_weight, cell_size, pc_range).
    image2d_aux=dict(
        num_classes=10,
        loss_weight_hm=0.5,
        loss_weight_wh=0.1,
        hidden_channels=256,
        image_size=tuple(image_size),
        in_channels=256,
    ),
)

# ---- Pipelines: insert LoadCamInstances2D after ImageAug3D ----
# (Need to rewrite the whole list because mmengine doesn't merge list elts.)
train_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True,
         color_type='color', backend_args=backend_args),
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=5, use_dim=5, backend_args=backend_args),
    dict(type='LoadPointsFromMultiSweeps',
         sweeps_num=30, load_dim=5, use_dim=5,
         pad_empty_sweeps=True, remove_close=True,
         backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True,
         with_label_3d=True, with_attr_label=False),
    dict(type='ImageAug3D', final_dim=image_size,
         resize_lim=[0.38, 0.55], bot_pct_lim=[0.0, 0.0],
         rot_lim=[-5.4, 5.4], rand_flip=True, is_train=True),
    # NEW: project per-cam 2D boxes through img_aug_matrix before any
    # subsequent transforms (BEVFusionRandomFlip3D + GlobalRotScaleTrans
    # only affect LiDAR / 3D, so it's safe to compute 2D GT now).
    dict(type='LoadCamInstances2D', image_size=tuple(image_size)),
    dict(type='BEVFusionGlobalRotScaleTrans',
         scale_ratio_range=[0.9, 1.1],
         rot_range=[-0.78539816, 0.78539816], translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter',
         classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                  'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                  'traffic_cone']),
    dict(type='GridMask', use_h=True, use_w=True, max_epoch=6, rotate=1,
         offset=False, ratio=0.5, mode=1, prob=0.0, fixed_prob=True),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs',
         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d',
               'gt_bboxes', 'gt_labels'],
         meta_keys=['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img',
                    'cam2lidar', 'ori_lidar2img', 'img_aug_matrix',
                    'box_type_3d', 'sample_idx', 'lidar_path', 'img_path',
                    'transformation_3d_flow', 'pcd_rotation',
                    'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
                    'lidar_aug_matrix', 'num_pts_feats',
                    'cam_inst_2d']),                  # NEW
]

test_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True,
         color_type='color', backend_args=backend_args),
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=5, use_dim=5, backend_args=backend_args),
    dict(type='LoadPointsFromMultiSweeps',
         sweeps_num=30, load_dim=5, use_dim=5,
         pad_empty_sweeps=True, remove_close=True,
         backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ImageAug3D', final_dim=image_size,
         resize_lim=[0.48, 0.48], bot_pct_lim=[0.0, 0.0],
         rot_lim=[0.0, 0.0], rand_flip=False, is_train=False),
    dict(type='Pack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img',
                    'cam2lidar', 'img_aug_matrix', 'box_type_3d',
                    'sample_idx', 'lidar_path', 'img_path',
                    'num_pts_feats']),
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            ann_file='nuscenes_infos_train_25pct_mkf30.pkl',
            pipeline=train_pipeline,
        ),
    ),
)
val_dataloader = dict(
    dataset=dict(
        ann_file='nuscenes_infos_val_mkf30.pkl',
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(ann_file='data/nuscenes/nuscenes_infos_val_mkf30.pkl')
test_evaluator = val_evaluator
