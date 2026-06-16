# =====================================================================
# B10b = FG-TCA warm-started from B5R (real-multi-sweep baseline).
#
# This is the *clean control* for FG-TCA's marginal contribution.
#
# Why this row exists
# -------------------
# B10 (FG-TCA from scratch, 6 epochs) hit NDS 0.4153 — far below all
# warm-start rows. The model was severely undertrained (final loss
# 5.48 vs B9's 4.21 with warm-start in 3 epochs). The headline number
# was dominated by the training-budget gap, not by anything FG-TCA did
# or didn't do.
#
# B10b removes that confound:
#   * Same warm-start protocol as B9 (B5R epoch_3.pth, which IS a
#     model trained on real multi-sweep data — no warm-start / data
#     distribution mismatch).
#   * Same 3-epoch fine-tune budget.
#   * Same MultiTokenTemporalBEVFusion config — only flipped to
#     FlowGuidedTemporalBEVFusion to introduce the flow head + warp.
#
# The B10b vs B9 delta is therefore exactly the FG-TCA contribution
# under matched data + budget conditions.
#
# Loading note: B5R's checkpoint was saved from BEVFusionCA (no
# temporal block). When loaded into FlowGuidedTemporalBEVFusion via
# load_weights(..., strict=False), all backbone / VT / fusion /
# bbox_head weights transfer; the temporal_pe, temporal_attn.*, and
# temporal_alpha parameters are initialized fresh from their __init__
# (out_proj and flow_head final-conv are zero-init → strict no-op at
# iter 0, identical to the B9 warm-start invariant).
# =====================================================================

_base_ = ['./B9_multikeyframe_temporal.py']

custom_imports = dict(
    imports=[
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.bevfusion_ca',
        'projects.bevdet.multitoken_temporal_lidar',
        'projects.bevdet.flow_guided_temporal',
    ],
    allow_failed_imports=False,
)

# Same model spec as B10 (just no schedule override; run_ablation.sh
# controls --epochs and --load-from).
model = dict(
    type='FlowGuidedTemporalBEVFusion',
    # num_buckets / window_seconds / num_attn_heads / attn_dropout
    # inherited from B9 (4 / 1.5 / 4 / 0.0).
    flow_loss_weight=0.5,
    cell_size=0.6,
    pc_range=(-54.0, -54.0, 54.0, 54.0),
)
