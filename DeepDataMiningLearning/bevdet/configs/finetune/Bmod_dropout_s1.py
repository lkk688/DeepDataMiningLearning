# =====================================================================
# Bmod (Stage 1) — Modality-robust backbone via modality dropout.
#
# Warm-starts from the trained Q-Mix model (B22) and fine-tunes on the
# SAME Waymo+nuScenes mixed corpus (with reliability weighting), but with
# stochastic modality dropout enabled in the BEV fusion:
#   each forward, with prob (pFull,pLidar,pCam) one BEV branch is zeroed,
#   so one set of weights handles L-only / C-only / LC.
#
# Dropout is driven entirely by the env var (read in
# BEVFusionCA._modality_mask) — NO config knowledge needed:
#   BEV_MOD_DROP="0.5,0.25,0.25"
# Eval forces a mode with BEV_MODALITY=full|lidar|camera.
#
# Stage 2 will add PhysicalAI pseudo as a forced-L-only contributor.
# =====================================================================

_base_ = ['./B22_qmix_v0.py']

# Adapt the already-good Q-Mix model to be modality-robust.
load_from = '/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/finetune_B22/epoch_1_weights.pth'

# Dropout makes the objective harder than the 1-epoch base fine-tunes,
# so give it 2 epochs over the mixed corpus.
train_cfg = dict(by_epoch=True, max_epochs=2, val_interval=2)
