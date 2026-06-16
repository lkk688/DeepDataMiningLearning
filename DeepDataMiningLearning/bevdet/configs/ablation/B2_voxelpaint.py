# =====================================================================
# B2 = B1 + Gated Voxel Painting.
#
# Adds early cross-modal fusion: for each LiDAR voxel center, sample the
# camera FPN P3 feature, project through a small MLP, and inject into the
# voxel feature via a learnable per-channel sigmoid gate. Shape-preserving
# (output channels unchanged), so all downstream sparse-conv code is
# unaffected. Initialized so the gate ≈ 0 (identity at init).
# =====================================================================

_base_ = ['./B1_calss.py']

custom_imports = dict(
    imports=[
        'projects.BEVFusion.bevfusion',           # original BEVFusion class (inherited type)
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.mvx_voxel_painting',     # PaintedWrapperVFE
    ],
    allow_failed_imports=False,
)

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

model = dict(
    pts_voxel_encoder=dict(
        _delete_=True,
        type='PaintedWrapperVFE',
        base_vfe=dict(type='HardSimpleVFE', num_features=5),
        point_cloud_range=point_cloud_range,
        voxel_size=[0.075, 0.075, 0.2],
        image_size=[256, 704],
        feature_size=[32, 88],
        img_feat_level=0,         # FPN P3
        cam_pool='max',           # max-pool across cameras (matches mybevfusion7_newv3)
        img_feat_out=64,
        fuse='gated',
        detach_img=True,
        align_corners=True,
        chunk_voxels=200000,
    ),
)
