# ===============================================================
# BEVFusion + CrossAttn LSS + (optional) VoxelPainting with Gating
# Target: H100-friendly training (BF16, larger attn_chunk),
#         minimal structural changes, switchable voxel painting.
# ===============================================================

# ---- Base (keep your original baseline for data/schedule/defaults) ----
_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# ---- Global switches ---------------------------------------------------
# Turn voxel painting ON/OFF here. When OFF, the model behaves as your original version.
voxel_painting_on = True

# ---- Registry / dynamic imports ---------------------------------------
default_scope = 'mmdet3d'
custom_imports = dict(
    imports=[
        'projects.bevdet.bevfusion',                   # your BEVFusion (unchanged)
        'projects.bevdet.cross_attn_lss',       # CrossAttnLSSTransform
        'projects.bevdet.bevfusion_with_aux_mvx',   # BEVFusionWithAux (we add a tiny switch only)
        'projects.bevdet.painting_context',        # NEW: thread-local ctx for painting
        'projects.bevdet.mvx_voxel_painting',      # NEW: PaintedWrapperVFE (supports fuse='gated')
        'projects.bevdet.freeze_utils',                 # FreezeExceptHook
        'projects.bevdet.painting_context',
        'projects.bevdet.mvx_voxel_painting',
    ],
    allow_failed_imports=False
)

# ---- Environment / IO --------------------------------------------------
env_cfg = dict(cudnn_benchmark=True)
backend_args = None

# ---- Geometry & modalities ---------------------------------------------
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)

# ---- Model --------------------------------------------------------------
model = dict(
    type='BEVFusionWithAux',     # subclass that adds AUX loss & a tiny switch for painting hooks
    # Pass the switch into the model; if False, hooks won't be installed.
    voxel_painting_on=voxel_painting_on,

    # Data preprocessor (RGB order for Swin/ConvNeXt/YOLO backbones)
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),

    # ---- Image backbone & FPN ----
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
        with_cp=True,  # save memory by checkpointing
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

    # ---- Camera branch: Cross-Attention LSS (drop-in replacement for Depth LSS) ----
    view_transform=dict(
        type='CrossAttnLSSTransform',
        in_channels=256,
        out_channels=64,            # 64 for lower memory/bandwidth
        image_size=[256, 704],
        feature_size=[32, 88],      # must match FPN feature fed to VT
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-5.0, 5.0, 10.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2,               # BEV 180 x 180
        num_z=2,
        use_cam_embed=True,
        attn_chunk=8192,            # larger chunk on H100; can go 16384 if memory allows
        debug=False
    ),

    # ---- Fusion layer (camera BEV + lidar BEV) ----
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[64, 256],      # [camera_out, lidar_out]
        out_channels=256
    ),

    # ---- LiDAR encoder --------------------------------------------------
    # SECOND family (sparse UNet) remains, but VFE is optionally wrapped by PaintedWrapperVFE.
    # If voxel_painting_on=False, we will set pts_voxel_encoder to your original VFE below.
)

# ---- LiDAR voxel settings ----------------------------------------------
pts_voxel_layer = dict(
    max_num_points=10,
    voxel_size=[0.075, 0.075, 0.2],
    point_cloud_range=point_cloud_range
)

# Build VFE depending on the switch
if voxel_painting_on:
    # PaintedWrapperVFE wraps your base VFE; it samples per-voxel image descriptors
    # from multi-view FPN features and fuses them back into voxel features.
    pts_voxel_encoder = dict(
        type='PaintedWrapperVFE',
        base_vfe=dict(             # your original VFE config goes here unchanged
            type='HardSimpleVFE',  # or 'DynamicVFE' depending on your baseline
            num_features=5,      # ← 
            #out_channels=64, #feat_channels=[64],
            #with_distance=False
        ),
        point_cloud_range=point_cloud_range,
        voxel_size=[0.075, 0.075, 0.2],
        image_size=[256, 704],     # final image size after ImageAug3D
        feature_size=[32, 88],     # FPN feature size provided to view_transform
        img_feat_level=0,          # choose the first level fed to VT
        cam_pool='avg',            # 'avg' (smoother) or 'max' (more robust)
        img_feat_out=32,           # dimension of per-voxel image descriptor
        fuse='gated',              # <<<<<< enable GATED fusion (channel-wise sigmoid gate)
        detach_img=True,           # only backprop into LiDAR branch
        align_corners=True,
        chunk_voxels=200000        # chunked sampling to avoid OOM
    )
else:
    pts_voxel_encoder = dict(
        type='HardSimpleVFE',      # original VFE untouched
        num_features=5,   # 
        #out_channels=64,  # 
        #with_distance=False
    )

# Attach voxel parts to model
model.update(dict(
    pts_voxel_layer=pts_voxel_layer,
    pts_voxel_encoder=pts_voxel_encoder,
    # Keep your SECOND sparse encoder & BEV neck unchanged from base
    # (they are inherited from the _base_ config)
))

# ---- AUX loss on camera BEV (already supported by your subclass) -------
model.update(dict(
    aux_cfg=dict(
        loss_weight=0.1,
        radius_cells=2
    )
))

# ---- Pipelines (minor speed tweaks: fewer sweeps) ----------------------
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
    batch_size=16, num_workers=16, persistent_workers=True, pin_memory=True, prefetch_factor=4,
    dataset=dict(dataset=dict(pipeline=train_pipeline, modality=input_modality))
)
val_dataloader = dict(
    batch_size=1, num_workers=16, persistent_workers=True, pin_memory=True, prefetch_factor=4,
    dataset=dict(pipeline=test_pipeline, modality=input_modality)
)
test_dataloader = val_dataloader

# ---- Schedulers / runtime ---------------------------------------------
param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0, end=800),
    dict(type='CosineAnnealingLR', begin=0, T_max=6, end=6, by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=0.85/0.95, begin=0, end=2.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=1, begin=2.4, end=6, by_epoch=True, convert_to_iter_based=True)
]
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# ---- Optimizer / AMP (BF16) -------------------------------------------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',        # H100-friendly
    accumulative_counts=1,
    optimizer=dict(type='AdamW', lr=0.0002, betas=(0.9, 0.99), weight_decay=0.01, fused=True),
    # param-wise: allow camera VT + fusion + bbox_head + (optional) painting VFE to adapt
    paramwise_cfg=dict(custom_keys={
        'view_transform': dict(lr_mult=0.5),
        'img_aux_head':   dict(lr_mult=1.0),
        'fusion_layer':   dict(lr_mult=0.7),
        'bbox_head':      dict(lr_mult=0.7),
        'pts_voxel_encoder': dict(lr_mult=1.0),
        'pts_middle_encoder': dict(lr_mult=1.0),
        # LiDAR-side small lr to adapt painting (when enabled)
        # typical decay exceptions
        'absolute_pos_embed': dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0),
        'norm': dict(decay_mult=1.0),
        # fully freeze image backbone by default (set to 0.1 if you want tiny finetune)
        #'img_backbone': dict(lr_mult=0.0),
        # Swin no training
        'img_backbone': dict(lr_mult=0.0, decay_mult=0.0),
    }),
    clip_grad=dict(max_norm=20, norm_type=2)
)

auto_scale_lr = dict(enable=False, base_batch_size=32)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

# ---- Custom hooks ------------------------------------------------------
# Remove base custom hooks, install Freeze + EMA + EmptyCache
del _base_.custom_hooks
custom_hooks = [
    dict(
        type='FreezeExceptHook',
        allowlist=(
            'view_transform', 'img_aux_head', 'fusion_layer', 'bbox_head',
            # LiDAR-side small adaptation
            'pts_voxel_encoder','pts_middle_encoder','pts_backbone','pts_neck'
        ),
        freeze_norm=False, verbose=True, use_regex=False
    ),
    #dict(type='EMAHook', momentum=0.0002, update_buffers=True),
    dict(type='EmptyCacheHook', after_iter=False, after_epoch=True),
]

load_from = '/data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth'  # ← EDIT ME

# Make sure we don’t pick up an old run’s last_checkpoint
auto_resume = False
resume = False

# (Optional) send outputs to a fresh work_dir to avoid accidental resume
work_dir = 'work_dirs/mybevfusion7_new'