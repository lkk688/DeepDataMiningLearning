# =====================================================================
# B1 = B0 + CA-LSS (Cross-Attention Lift-Splat replaces DepthLSS).
#
# This is the *single architectural change* that motivates the paper:
# replacing the dense depth-volume LSS with chunkable BEV-query
# cross-attention. Memory footprint drops dramatically; accuracy
# should remain comparable when other components are matched.
# =====================================================================

_base_ = ['./B0_modern_baseline.py']

custom_imports = dict(
    imports=[
        'projects.BEVFusion.bevfusion',             # original BEVFusion class (inherited type)
        'projects.bevdet.cross_attn_lss2',          # CrossAttnLSSTransform
    ],
    allow_failed_imports=False,
)

model = dict(
    view_transform=dict(
        _delete_=True,
        type='CrossAttnLSSTransform',
        in_channels=256,            # P3 from FPN
        out_channels=128,            # camera BEV channels (matches our best variant)
        image_size=[256, 704],
        feature_size=[32, 88],       # P3 (stride 8)
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-5.0, 5.0, 10.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2,
        num_z=2,
        use_cam_embed=True,
        attn_chunk=8192,
        # No multi-scale yet (P3 only) — added in B3.
    ),
    fusion_layer=dict(
        in_channels=[128, 256], out_channels=256
    ),
)
