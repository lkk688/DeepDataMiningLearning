# =====================================================================
# B3 = B2 + multi-scale tokens (P3+P4) into CA-LSS.
#
# Instead of feeding only P3 to the cross-attention, feed both P3 and P4
# fused via a learnable gated_add. Gives the BEV queries access to two
# semantic scales without inflating the key/value count.
#
# Empirically (mybevfusion12 → mybevfusion12v2) this delivered +2.2 NDS.
# =====================================================================

_base_ = ['./B2_voxelpaint.py']

# Need BEVFusionCA-style detector to handle multi-scale list input,
# OR a model that wraps view_transform with multiscale-friendly call.
# Our CrossAttnLSSTransform itself accepts list inputs; the surrounding
# detector must pass a list. The original BEVFusion class only passes
# the level-0 tensor, so we use BEVFusionCA which supports two-scale
# tokens.
custom_imports = dict(
    imports=[
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.bevfusion_ca',          # BEVFusionCA detector
    ],
    allow_failed_imports=False,
)

model = dict(
    type='BEVFusionCA',
    use_two_scale_tokens=True,        # send [P3, P4] into view_transform
    view_transform=dict(
        ms_extra_in_channels=[256],   # P4 channels
        ms_fuse_mode='gated_add',
        ms_align_corners=False,
    ),
)
