# ===============================================================
# BEVFusion + CrossAttn LSS + (optional) VoxelPainting with Gating
# + Optional Image RPF neck (post-FPN enhancer)
# + Optional LiDAR FireRPF neck (replaces SECFPN, shape-compatible)
#
# Target: H100-friendly training (BF16, larger attn_chunk),
#         minimal structural changes, switchable voxel painting & RPF,
#         stable, shape-preserving interfaces.
# ===============================================================

# ---- Base (keep your original baseline for data/schedule/defaults) ----
_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# ---- Global feature toggles -------------------------------------------
voxel_painting_on = True      # turn voxel painting on/off
use_rpf_img = True            # OPTIONAL: add a 2D RPF enhancer after the image FPN (keeps P3/P4/P5 shapes)
use_rpf_lidar = True          # OPTIONAL: replace SECFPN with FireRPF neck on the LiDAR side (same #outs/channels)

# ---- Registry / dynamic imports ---------------------------------------
# If you haven't added FireRPF/RPF2D to your codebase, set the corresponding
# switches to False so imports aren't required.
default_scope = 'mmdet3d'
custom_imports = dict(
    imports=[
        # Your existing modules
        'projects.bevdet.bevfusion',               # BEVFusion core
        'projects.bevdet.cross_attn_lss',          # CrossAttnLSSTransform
        'projects.bevdet.bevfusion_with_aux_mvx',  # subclass with AUX loss & hooks
        'projects.bevdet.painting_context',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.freeze_utils',
        # OPTIONAL RPF modules (from extension branch). Only needed if switches are True.
        # Paths here match the extension experiment; adjust if you placed them elsewhere.
        'projects.bevdet.fire_rpf_necks',        # optional lightweight helpers (not strictly required)
    ],
    allow_failed_imports=not (use_rpf_img or use_rpf_lidar)
)

# ---- Environment / IO --------------------------------------------------
env_cfg = dict(cudnn_benchmark=True)
backend_args = None

# ---- Geometry & modalities ---------------------------------------------
# World coordinate range (meters): [xmin, ymin, zmin, xmax, ymax, zmax]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)

# ---- Model --------------------------------------------------------------
# Key I/O conventions used in comments below:
#   B  := batch size
#   V  := #voxels (dynamic) or Nx*Ny*Nz (static)
#   C  := channels
#   H,W: spatial dims
#   Ncam := #cameras (usually 6 for nuScenes)
#   BEV grid: xbound/ybound with step 0.3m → 108/0.3 = 360 cells, downsample=2 → 180
model = dict(
    type='BEVFusionWithAux',
    voxel_painting_on=voxel_painting_on,

    # Data preprocessor (RGB order; Swin expects RGB)
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),

    # ------------------------ IMAGE BRANCH -------------------------------
    # Backbone: Swin-Tiny (frozen by paramwise below)
    # INPUT:  images  [B, Ncam, 3, 256, 704]
    # OUTPUT: stages C3,C4,C5 (out_indices=[1,2,3]) → per-cam feature maps
    #         C3: [B*Ncam, 192, 32, 88]   (stride 8)
    #         C4: [B*Ncam, 384, 16, 44]   (stride 16)
    #         C5: [B*Ncam, 768,  8, 22]   (stride 32)
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

    # FPN (GeneralizedLSSFPN): make P3/P4/P5, all 256ch
    # INPUT:  [C3:192, C4:384, C5:768]
    # OUTPUT: [P3:256, P4:256, P5:256] with shapes
    #         P3: [B*Ncam, 256, 32, 88]  (stride 8)   <-- used by VT (see feature_size below)
    #         P4: [B*Ncam, 256, 16, 44]  (stride 16)
    #         P5: [B*Ncam, 256,  8, 22]  (stride 32)
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

    # OPTIONAL: RPF enhancer AFTER the FPN (shape-preserving).
    # INPUT:  list[P3,P4,P5] each [B*Ncam, 256, H/stride, W/stride]
    # OUTPUT: list[P3',P4',P5'] with SAME shapes/channels as inputs.
    # VT will keep consuming level 0 (P3') below.
    # If you don't have this module in your codebase, set use_rpf_img=False.
    # NOTE: We pass this as a named submodule; your BEVFusionWithAux should route FPN→RPF→VT.
    #       If your class uses a different arg name, rename 'img_rpf_neck' accordingly.
    # (We add this block below via `if use_rpf_img:` to avoid passing None.)
    # img_rpf_neck = <added below>

    # View transformer: Cross-Attention LSS (drop-in replacement for depth-LSS)
    # INPUT:  FPN level idx 0 (P3 or P3') → [B*Ncam, 256, 32, 88], image_size=[256,704]
    # OUTPUT: camera BEV tensor: [B, C_cam, H_bev, W_bev] = [B, 64, 180, 180]
    # Grid dims from bounds: 108/0.3 = 360; downsample=2 → 180.
    view_transform=dict(
        type='CrossAttnLSSTransform',
        in_channels=256,           # matches P3/P3' channel
        out_channels=64,           # camera BEV channels kept small for bandwidth
        image_size=[256, 704],
        feature_size=[32, 88],     # consumes P3/P3' (stride 8)
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-5.0, 5.0, 10.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2,              # BEV 180 x 180
        num_z=2,
        use_cam_embed=True,
        attn_chunk=8192,           # H100-friendly; 16384 if memory allows
        debug=False
    ),

    # ------------------------ FUSION LAYER -------------------------------
    # INPUT:  camera BEV [B, 64, 180, 180], LiDAR BEV [B, 256, 180, 180]
    # OUTPUT: fused BEV   [B, 256, 180, 180]
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[64, 256],
        out_channels=256
    ),
    bbox_head=dict(in_channels=256),

    # ------------------------ AUX HEAD (camera BEV) ----------------------
    # Small auxiliary loss on camera BEV for stabilizing training
    # INPUT:  camera BEV [B, 64, 180, 180]
    # OUTPUT: aux predictions (implementation-specific); does not affect shapes above
    aux_cfg=dict(loss_weight=0.1, radius_cells=2),

    # ------------------------ LiDAR BRANCH -------------------------------
    # VFE is optionally wrapped with per-voxel image painting (gated).
    # Voxel settings given below; SECOND backbone + (SECFPN or FireRPF neck).
    # pts_voxel_layer / pts_voxel_encoder are attached later via model.update(...).
    # pts_neck (LiDAR BEV neck) is replaced below when use_rpf_lidar=True.
)

# ---- LiDAR voxelization settings ---------------------------------------
# With voxel_size=[0.075,0.075,0.2] over 108×108 m XY:
#   Nx = 108/0.075 = 1440, Ny = 1440, Nz depends on z range / 0.2
# SECOND middle encoder typically downsamples by 8 → BEV 180×180 (matches camera BEV)
pts_voxel_layer = dict(
    max_num_points=10,
    voxel_size=[0.075, 0.075, 0.2],
    point_cloud_range=point_cloud_range
)

# ---- Build VFE depending on painting switch ----------------------------
if voxel_painting_on:
    # PaintedWrapperVFE: samples per-voxel image descriptors from FPN (level 0)
    # INPUT: voxels → base_vfe feats (e.g., 5 dims), + sampled img feats (P3/P3'), fused by a gated unit
    # OUTPUT: voxel features with img context (dimension controlled by base_vfe + img_feat_out)
    pts_voxel_encoder = dict(
        type='PaintedWrapperVFE',
        base_vfe=dict(
            type='HardSimpleVFE',    # or 'DynamicVFE' if that's your baseline
            num_features=5
        ),
        point_cloud_range=point_cloud_range,
        voxel_size=[0.075, 0.075, 0.2],
        image_size=[256, 704],
        feature_size=[32, 88],      # align with VT input level
        img_feat_level=0,           # use P3/P3' for painting
        cam_pool='avg',             # 'avg' (smoother) or 'max'
        img_feat_out=32,            # per-voxel image descriptor dim before gating
        fuse='gated',               # channel-wise sigmoid gate
        detach_img=True,            # backprop only through LiDAR branch
        align_corners=True,
        chunk_voxels=200000         # chunked sampling to avoid OOM
    )
else:
    pts_voxel_encoder = dict(
        type='HardSimpleVFE',
        num_features=5
    )

# Attach voxel parts to model
model.update(dict(
    pts_voxel_layer=pts_voxel_layer,
    pts_voxel_encoder=pts_voxel_encoder
))

# ---- OPTIONAL LiDAR NECK: FireRPF replaces SECFPN (shape-compatible) ---
# Pins from inspect:
#   SECFPN.in_channels = [128, 256]
#   SECFPN.out_channels = [256, 256]
#   #outs = 2 ; first deblock uses stride 1 (128->256), second uses stride 2 (256->256)
if use_rpf_lidar:
    model.update(dict(
        pts_neck=dict(
            type='FireRPFNeck',            # from extension branch
            in_channels=[128, 256],        # << pinned
            out_channels=256,
            num_outs=2,                    # << pinned
            upsample_strides=[1, 2],       # keep spatial scale behavior aligned with SECFPN
            blocks_per_stage=1,            # tiny, stable; increase if headroom
            with_residual=True,
            use_cbam=True, ca_reduction=16, sa_kernel=7,
        )
    ))
# else: keep SECFPN inherited from the _base_ config

# ---- OPTIONAL Image RPF neck: post-FPN enhancer (shape-preserving) -----
# Pins from your FPN: out_channels=256, num_outs=3 → P3,P4,P5 all 256ch.
if use_rpf_img:
    model.update(dict(
        img_rpf_neck=dict(
            type='FireRPF2DNeck',          # from extension branch (2D)
            in_channels=[256, 256, 256],   # P3,P4,P5
            out_channels=256,
            num_outs=3,                    # keep 3 scales
            blocks_per_stage=1,            # tiny & fast
            with_residual=True,
            use_cbam=True, ca_reduction=16, sa_kernel=7,
        )
    ))
# IMPORTANT: Your BEVFusionWithAux forward should route:
#   img_backbone → img_neck → (img_rpf_neck if exists) → view_transform
#   pts_voxel_encoder → pts_middle_encoder(SECOND) → (pts_neck=FireRPF or SECFPN) → LiDAR BEV

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
# Strategy:
#   • freeze image backbone; train VT + fusion + head + (first spconv stage) + VFE (+ RPF necks)
#   • slightly higher LR for the tiny RPF necks
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',        # H100-friendly
    accumulative_counts=1,
    optimizer=dict(type='AdamW', lr=0.0002, betas=(0.9, 0.99), weight_decay=0.01, fused=True),
    paramwise_cfg=dict(custom_keys={
        'view_transform': dict(lr_mult=0.5),
        'img_aux_head':   dict(lr_mult=1.0),
        'fusion_layer':   dict(lr_mult=0.7),
        'bbox_head':      dict(lr_mult=0.7),

        # LiDAR-side small adaptation (painting + first spconv encoder layer)
        'pts_voxel_encoder': dict(lr_mult=0.7),
        'pts_middle_encoder.encoder_layers.0': dict(lr_mult=0.5),
        'pts_middle_encoder.encoder_layers.1': dict(lr_mult=0.5),

        # NEW: allow training of the optional RPF necks (safe even if absent; ignored silently)
        'img_rpf_neck': dict(lr_mult=1.5),   # tiny/new → slightly higher LR
        'pts_neck':     dict(lr_mult=1.2),   # FireRPF replacing SECFPN

        # typical decay exceptions
        'absolute_pos_embed': dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0),
        'norm': dict(decay_mult=0.0),

        # fully freeze Swin backbone
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
# Freeze everything except the modules listed in allowlist (safe even if some are absent).
try:
    del _base_.custom_hooks
except Exception:
    pass

# custom_hooks = [
#     dict(
#         type='FreezeExceptHook',
#         allowlist=(
#             # camera side
#             'view_transform', 'img_aux_head', 'fusion_layer', 'bbox_head',
#             # lidar side
#             'pts_voxel_encoder',
#             'pts_middle_encoder.encoder_layers.0',
#             'pts_middle_encoder.encoder_layers.1',
#             # optional RPF modules
#             'img_rpf_neck',   # 2D RPF enhancer
#             'pts_neck'        # FireRPF neck (when replacing SECFPN)
#         ),
#         freeze_norm=False, verbose=True, use_regex=False
#     ),
#     #dict(type='EMAHook', momentum=0.0002, update_buffers=True),
#     dict(type='EmptyCacheHook', after_iter=False, after_epoch=True),
# ]
custom_hooks=[
    dict(type='EmptyCacheHook', after_iter=False, after_epoch=True),
]