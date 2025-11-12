# ===============================================================
# BEVFusion (Depth LSS) → Probe run: train ONLY FireRPF necks
# Loads original BEVFusion weights, freezes everything else.
# Target: quick 3–6 epoch check of img_rpf_neck + pts_neck impact.
# ===============================================================

# Inherit your known-good BEVFusion (Depth LSS) baseline.
# (Adjust the path to your repo’s baseline if needed.)
_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

default_scope = 'mmdet3d'
backend_args = None
env_cfg = dict(cudnn_benchmark=True)

# ----- Imports (our CBAM+RPF necks + your freeze hook) -------------------
custom_imports = dict(
    imports=[
            'projects.bevdet.bevfusion',
            'projects.bevdet.fire_rpf_necks',   # contains FireRPF2DNeck & FireRPFNeck
            'projects.bevdet.freeze_utils',     # FreezeExceptHook
        # (Your base config already imports bevfusion modules.)
    ],
    allow_failed_imports=False
)

# ----- Geometry / modality (match base) ---------------------------------
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)

# ----- Depth LSS channel setting (match your ORIGINAL BEVFusion ckpt) ----
# Most public BEVFusion ckpts use 80 or 96. Set this to YOUR ckpt’s VT out_channels.
vt_channels = 80  # ← change to 96 if your original ckpt used 96

# ----- Model -------------------------------------------------------------
# NOTE:
# - We assume your BEVFusion forward supports an optional `img_rpf_neck`
#   inserted between FPN and Depth LSS. If your code names differ, rename
#   the key below to what your forward expects.
model = dict(
    # If your baseline uses a subclass like BEVFusionWithAux, keep it.
    # Otherwise, set to 'BEVFusion'. Both work as long as the forward wires necks.
    type='BEVFusion',

    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.3,
        patch_norm=True, out_indices=[1, 2, 3],
        with_cp=True, convert_weights=True,
        frozen_stages=4,
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

    # --- Image backbone & FPN are inherited from base. We add an enhancer neck: ---
    # Shape-preserving, sits AFTER img_neck and BEFORE view_transform.
    img_rpf_neck=dict(
        type='FireRPF2DNeck',
        # From your inspect script: FPN emits 3 levels of 256ch each (P3,P4,P5).
        in_channels=[256, 256, 256],
        out_channels=256,
        num_outs=3,
        blocks_per_stage=1,
        with_residual=True,
        use_cbam=True, ca_reduction=16, sa_kernel=7
    ),

    # --- View transform: restore Depth LSS, matching your original ckpt ---
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,                # FPN/RPF outputs 256 per level
        out_channels=vt_channels,       # MUST match original ckpt (80 or 96 typically)
        image_size=[256, 704],
        feature_size=[32, 88],          # P3 (stride 8) from your FPN
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-5.0, 5.0, 10.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2
    ),

    # --- Fusion (camera BEV vt_channels + LiDAR BEV 256) → 256 ---
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[vt_channels, 256],
        out_channels=256
    ),

    # --- Replace SECFPN with our LiDAR FireRPF neck (SECONDFPN-compatible) ---
    pts_neck=dict(
        type='FireRPFNeck',
        # From your inspect script: SECOND → [128, 256] (2 scales)
        in_channels=[128, 256],
        out_channels=256,
        num_outs=2,
        upsample_strides=[1, 2],
        # Try learnable deconv first; swap to interp if you prefer
        upsample_cfg=dict(type='deconv', bias=False),
        blocks_per_stage=1,
        with_residual=True,
        use_cbam=True, ca_reduction=16, sa_kernel=7
    ),

    # Keep head as in the original BEVFusion. If your original forward concatenates
    # fused(256) + lidar(256) → 512 into the head, DO NOT change in_channels here.
    # If you had modified it to 256 earlier, revert to 512 for apples-to-apples.
    bbox_head=dict(in_channels=256),  # uncomment only if your baseline needs forcing
)

# ----- Train only the two new necks -------------------------------------
# 1) Hard freeze everything except the necks (and allow BN to update in necks)
del _base_.custom_hooks
custom_hooks = [
    dict(
        type='FreezeExceptHook',
        allowlist=('pts_neck', 'img_rpf_neck'),
        freeze_norm=True,   # BN in necks must keep updating
        verbose=True, use_regex=False
    ),
    dict(type='EmptyCacheHook', after_iter=False, after_epoch=True),
]

# 2) (Optional but safer) also zero LR on known modules; give LR budget to necks
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',#'bfloat16',            # H100-friendly
    accumulative_counts=1,
    optimizer=dict(type='AdamW', lr=2.0e-4, betas=(0.9, 0.99), weight_decay=0.01, fused=True),
    paramwise_cfg=dict(custom_keys={
        # Train these:
        'pts_neck': dict(lr_mult=1.0),
        'img_rpf_neck': dict(lr_mult=1.0),

        # Everything else effectively frozen:
        'img_backbone': dict(lr_mult=0.0, decay_mult=0.0),
        'img_neck': dict(lr_mult=0.0),
        'view_transform': dict(lr_mult=0.0),
        'fusion_layer': dict(lr_mult=0.0),
        'bbox_head': dict(lr_mult=1.0),
        'pts_backbone': dict(lr_mult=0.0),
        'pts_voxel_encoder': dict(lr_mult=0.0),
        'pts_middle_encoder': dict(lr_mult=0.0),

        # Typical decay exceptions (kept for completeness)
        'absolute_pos_embed': dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0),
        'norm': dict(decay_mult=0.0),
    }),
    clip_grad=dict(max_norm=20, norm_type=2)
)



# ---- Pipelines (same as your baseline, with small speed tweaks) --------
train_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color', backend_args=backend_args),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=5, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True, backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ImageAug3D', final_dim=[256, 704], resize_lim=[0.38, 0.55], bot_pct_lim=[0.0, 0.0], rot_lim=[-5.4, 5.4], rand_flip=True, is_train=True),
    dict(type='BEVFusionGlobalRotScaleTrans', scale_ratio_range=[0.9, 1.1], rot_range=[-0.78539816, 0.78539816], translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']),
    dict(type='GridMask', use_h=True, use_w=True, max_epoch=6, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.0, fixed_prob=True),
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
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=5, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True, backend_args=backend_args),
    dict(type='ImageAug3D', final_dim=[256, 704], resize_lim=[0.48, 0.48], bot_pct_lim=[0.0, 0.0], rot_lim=[0.0, 0.0], rand_flip=False, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(type='Pack3DDetInputs', keys=['img','points','gt_bboxes_3d','gt_labels_3d'],
         meta_keys=['cam2img','ori_cam2img','lidar2cam','lidar2img','cam2lidar','img_aug_matrix','box_type_3d','sample_idx','lidar_path','img_path','num_pts_feats'])
]

train_dataloader = dict(
    batch_size=32, num_workers=16, persistent_workers=True, pin_memory=True, prefetch_factor=4,
    dataset=dict(dataset=dict(pipeline=train_pipeline, modality=input_modality))
)
val_dataloader = dict(
    batch_size=1, num_workers=16, persistent_workers=True, pin_memory=True, prefetch_factor=4,
    dataset=dict(pipeline=test_pipeline, modality=input_modality)
)
test_dataloader = val_dataloader

# ----- Short probe schedule (3–6 epochs) --------------------------------
# If you want exactly 6, keep this. For even faster probe, set max_epochs=3.
param_scheduler = [
    dict(type='LinearLR', start_factor=1/3, by_epoch=False, begin=0, end=800),
    dict(type='CosineAnnealingLR', begin=0, T_max=6, end=6, by_epoch=True,
         eta_min_ratio=1e-4, convert_to_iter_based=True),
]
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

try:
    del _base_.custom_hooks
except Exception:
    pass

custom_hooks = [
    dict(
        type='FreezeExceptHook',
        allowlist=(
            'img_rpf_neck',   # 2D RPF enhancer
            'pts_neck',        # FireRPF neck (when replacing SECFPN)
            'bbox_head'
        ),
        freeze_norm=False, verbose=True, use_regex=False
    ),
    #dict(type='EMAHook', momentum=0.0002, update_buffers=True),
    dict(type='EmptyCacheHook', after_iter=False, after_epoch=True),
]
# ----- Dataloaders (use your baseline settings) -------------------------
# For a fair, fast probe, 5 sweeps is okay. If you can afford it, 10 sweeps
# improves mAVE/motion classes quickly. Uncomment to bump both train & val:
# for p in _base_.train_dataloader['dataset']['dataset']['pipeline']:
#     if p.get('type', '') == 'LoadPointsFromMultiSweeps': p['sweeps_num'] = 10
# for p in _base_.val_dataloader['dataset']['pipeline']:
#     if p.get('type', '') == 'LoadPointsFromMultiSweeps': p['sweeps_num'] = 10

# ----- Checkpoint loading: LOAD ORIGINAL BEVFUSION (Depth LSS) -----------
# IMPORTANT: point this to your original, full-model BEVFusion ckpt
load_from = '/data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth'  # ← EDIT ME
load_cfg = dict(strict=False)
# Make sure we don’t pick up an old run’s last_checkpoint
auto_resume = False
resume = False

# (Optional) send outputs to a fresh work_dir to avoid accidental resume
#work_dir = 'work_dirs/bevfusion_depthlss_rpf_probe'