# =====================================================================
# B5R = B5 (GQA, no temporal) trained on REAL multi-sweep data.
#
# Control experiment for B9's interpretation.
#
# Background:
#   * B5..B8 all silently trained on degenerate multi-sweep (because
#     ``nuscenes_infos_train_25pct.pkl`` lacks the ``lidar_sweeps``
#     field, so ``LoadPointsFromMultiSweeps`` falls into the
#     ``pad_empty_sweeps`` branch that duplicates the current keyframe
#     N times with time=0). They were effectively single-keyframe runs.
#   * B9 swapped in the new ``..._mkf30.pkl`` files (real 30 sweeps,
#     ~1.5s span, proper per-sweep ego-warp) AND turned on the
#     K-token temporal cross-frame attention block. Result:
#     NDS 0.6811 / mAVE 0.2989 — basically tied with B8 V2 (degenerate
#     multi-sweep + temporal attn).
#
# What we still don't know:
#   How much of B9's number comes from the *real multi-sweep data
#   change* vs. the *temporal-attention block*? B5R isolates that.
#
# B5R = same B5_gqa architecture (no temporal attention) but trained
# on the real ``mkf30`` pkls with ``sweeps_num=30``. Same 3-epoch,
# 25%-subset warm-start fine-tune as the rest of the ablation chain.
# Compared against:
#   * B5  (degenerate multi-sweep, no temporal)  → NDS 0.6848
#   * B9  (real multi-sweep, temporal)            → NDS 0.6811
# The delta (B5R vs B9) is the contribution of the temporal block
# under matched data conditions. The delta (B5 vs B5R) is the
# contribution of the real multi-sweep data alone.
# =====================================================================

_base_ = ['./B5_gqa.py']

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
backend_args = None

# --- Real-multi-sweep pipelines (only sweeps_num differs from base) ---
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
    dict(type='ImageAug3D', final_dim=[256, 704],
         resize_lim=[0.38, 0.55], bot_pct_lim=[0.0, 0.0],
         rot_lim=[-5.4, 5.4], rand_flip=True, is_train=True),
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
                    'lidar_aug_matrix', 'num_pts_feats']),
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
    dict(type='ImageAug3D', final_dim=[256, 704],
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

# No model.type override — keeps the B5 BEVFusionCA detector (no
# temporal-attention block). That's the whole point of this control.
