# =====================================================================
# B8 = B5 + Multi-Token Temporal LiDAR Attention (Stage 1 of temporal).
#
# Replaces B7's naive now/past split with:
#   • K=4 time buckets across the existing 10-sweep window (~500 ms)
#   • Learnable per-bucket temporal positional encoding
#   • Per-BEV-cell cross-frame attention (queries from current,
#     K/V across all K buckets)
#   • Residual + zero-init out_proj + zero-init residual scalar so the
#     block is a *strict no-op at iteration 0* — warm-start safe.
#
# Same training recipe as B5/B6/B7: 3 epochs on the 25% nuScenes
# subset, warm-started from mybevfusion7_newv3/epoch_3.pth.
# =====================================================================

_base_ = ['./B5_gqa.py']

custom_imports = dict(
    imports=[
        # bevfusion_ca already loads the BEVFusion class via its import
        # chain; do NOT import projects.BEVFusion.bevfusion directly here.
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.bevfusion_ca',
        'projects.bevdet.multitoken_temporal_lidar',
    ],
    allow_failed_imports=False,
)

# Switch the detector class. All other fields (data_preprocessor /
# pts_* / img_* / view_transform / fusion_layer / bbox_head /
# use_two_scale_tokens / aux_cfg / etc.) are inherited from B5_gqa and
# forwarded as **bevfusion_ca_kwargs.
model = dict(
    type='MultiTokenTemporalBEVFusion',
    num_buckets=4,
    window_seconds=0.5,
    num_attn_heads=4,
    attn_dropout=0.0,
)
