# =====================================================================
# B4 = B3 + auxiliary BEV center heatmap supervision (training-only).
#
# A class-agnostic Gaussian heatmap on the camera BEV provides dense,
# well-shaped gradients during early training, especially helpful when
# the image backbone is frozen. Removed at inference; zero deployment cost.
# =====================================================================

_base_ = ['./B3_multiscale.py']

model = dict(
    aux_cfg=dict(
        loss_weight=0.15,
        radius_cells=2,
        loss_type='focal_dice',
    ),
)
