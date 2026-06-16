# =====================================================================
# B10c = B10b + dead-pathway bug fix + temporal aux supervision.
#
# B10b confirmed (via post-hoc weight inspection) that every prior
# temporal experiment (B8 V2, B9, B10, B10b) had a silent
# dead-pathway bug:
#
#     fused = bev_main + temporal_alpha · delta
#     where temporal_alpha = 0 AND out_proj.weight = 0 at init.
#
# Both started at zero and stayed at exact zero through training,
# because:
#     d(scaled)/d(temporal_alpha) = delta = 0    → ∇temporal_alpha = 0
#     d(scaled)/d(delta)          = alpha = 0    → no signal to out_proj
#
# The temporal attention block computed correctly but its contribution
# was multiplied by zero before being added to bev_main. Every prior
# "temporal" run was effectively the non-temporal baseline + dead
# weights.
#
# B10c changes:
#   1. Drop ``temporal_alpha`` (the redundant scale gate). ``out_proj``
#      zero-init alone is sufficient to preserve the warm-start
#      invariant AND lets gradient flow.
#   2. Add ``temporal_aux_head`` — a 2-conv stack that predicts per-cell
#      velocity from the post-temporal fused BEV — and supervise it
#      against GT velocity (same painter as B10/B10b). This gives the
#      temporal block a strong direct gradient signal (out_proj,
#      q/k/v_proj, temporal_pe, past-slice encoders all get supervised
#      via the velocity-regression objective).
#
# Everything else identical to B10b: warm-start from B5R epoch_3.pth,
# 3 epochs, 25% subset, real multi-sweep mkf30 pkls.
# =====================================================================

_base_ = ['./B10b_flow_guided_warmstart.py']

# Only override the temporal-aux weight (new field) — everything else
# is inherited from B10b's chain.
model = dict(
    type='FlowGuidedTemporalBEVFusion',
    flow_loss_weight=0.5,
    temporal_aux_loss_weight=0.5,   # NEW: weight on the post-temporal
                                    # velocity regression that
                                    # supervises the temporal block.
    cell_size=0.6,
    pc_range=(-54.0, -54.0, 54.0, 54.0),
)
