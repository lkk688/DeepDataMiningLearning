# =====================================================================
# B7-B = B5 + Temporal LiDAR (real multi-frame on the LiDAR side).
#
# Splits the existing 10-sweep aggregated point cloud (~500 ms span) into
# "now" (latest, |time|≤τ) and "past" (older) branches inside the model.
# Past branch is no_grad (BEVDet4D-style). 1×1 conv combines the two
# LiDAR BEVs back to the channel count expected by ConvFuser.
# =====================================================================

_base_ = ['./B5_gqa.py']

custom_imports = dict(
    imports=[
        # Note: bevfusion_ca's import chain already loads the BEVFusion class
        # (via .bevfusion.ops → .bevfusion). Do NOT import projects.BEVFusion.bevfusion
        # here — would double-register.
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.bevfusion_ca',
        'projects.bevdet.temporal_lidar',
    ],
    allow_failed_imports=False,
)

# Switch the detector class to our TemporalBEVFusionCA wrapper.
# All existing model fields (data_preprocessor / pts_* / img_* / view_transform /
# fusion_layer / bbox_head / use_two_scale_tokens / aux_cfg / etc.) are
# inherited from B5_gqa via _base_ and forwarded to BEVFusionCA via **kwargs.
#
# We keep the inherited model dict and only override `type` and add the new
# `time_threshold` field. mmengine's lazy merge handles this fine — the
# child's `type` wins, and added fields propagate to TemporalBEVFusionCA's
# __init__ kwargs.
model = dict(
    type='TemporalBEVFusionCA',
    time_threshold=0.1,
)
