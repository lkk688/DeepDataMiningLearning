# =====================================================================
# B11b = B10c + FCOS3D-lite per-camera aux head.
#
# Diagnostic findings from B11 (NDS regression of -0.0044 vs B10c):
#   • The 2D-only aux head trained fast (loss_2d_aux_hm: 1.0 → 0.7)
#     but learned an over-eager 2D detector:
#       - bicycle 2D-only count = 39,808 vs 3D-only = 354 (hallucinates)
#       - barrier 3D-only count = 9,396 vs 2D-only = 3,327 (misses)
#       - class agreement on spatially-paired matches: only 54%
#   • The gradient signal pulled FPN features toward 2D shape
#     discriminability, away from depth/geometric structure that CA-LSS
#     needs for BEV lifting.
#
# B11b replaces the 2D-only head with an FCOS3D-lite head that
# additionally supervises log-depth and 3D-center pixel offset from
# nuScenes' cam_instances (which already carry .depth and .center_2d
# for every annotated object). These two new objectives force the FPN
# to encode geometric structure (occlusion, parallax, scale-from-depth)
# — the right signal for CA-LSS.
#
# Inherits B10c (FG-TCA + temporal aux + real-multi-sweep mkf30 data).
# Replaces only:
#   * detector class: Image2DAuxBEVFusion → Image3DAuxBEVFusion
#   * pipeline transform: LoadCamInstances2D → LoadCamInstances3DAux
#   * Pack3DDetInputs meta key: cam_inst_2d → cam_inst_3d_aux
# =====================================================================

_base_ = ['./B10c_flow_guided_warmstart_fixed.py']

custom_imports = dict(
    imports=[
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.bevfusion_ca',
        'projects.bevdet.multitoken_temporal_lidar',
        'projects.bevdet.flow_guided_temporal',
        'projects.bevdet.image2d_aux',           # for shared helpers
        'projects.bevdet.image_3d_aux',          # NEW
    ],
    allow_failed_imports=False,
)

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
backend_args = None
image_size = [256, 704]

model = dict(
    type='Image3DAuxBEVFusion',
    image3d_aux=dict(
        num_classes=10,
        loss_weight_cls=0.3,      # down from B11's 0.5 — less FPN dominance
        loss_weight_wh=0.1,
        loss_weight_depth=0.2,    # NEW
        loss_weight_c3d=0.1,      # NEW
        hidden_channels=256,
        image_size=tuple(image_size),
        in_channels=256,
    ),
)

# ---- Pipelines: insert LoadCamInstances3DAux after ImageAug3D ----
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
    # NEW: project cam_instances (boxes + center_2d) through img_aug_matrix.
    dict(type='LoadCamInstances3DAux', image_size=tuple(image_size)),
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
                    'cam_inst_3d_aux']),       # NEW
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
