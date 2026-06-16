# ============================================================
# Phase 1: Detection + Occupancy  (shared BEV encoder)
#
# Inherits everything from mybevfusion12v2.py (CrossAttnLSS +
# FireRPF necks + VoxelPainting + 10-epoch schedule) and adds:
#
#   1. MultiTaskBEVFusion wrapper that hooks into ConvFuser
#      to capture fused BEV features.
#   2. BEVOccHead for 3-D occupancy prediction.
#
# Occupancy GT options
# --------------------
# a) Binary LiDAR pseudo-labels (no extra dataset needed):
#       Set occ_classes=2 and occ_num_z=16 here.
#       The build_pseudo_occ_gt() helper in occ_head.py voxelizes
#       raw LiDAR points into [Z, H, W] binary occupancy on the fly.
#       Add a thin transform to the train_pipeline that calls it and
#       stores the result under key 'occ_gt' in the batch.
#
# b) Occ3D-nuScenes (17 semantic classes):
#       Download https://tsinghua-mars-lab.github.io/Occ3D/
#       Set occ_classes=17 and point ann_file to the occ pkl.
#       See occ_head.OCC3D_CLASS_WEIGHTS_17 for recommended weights.
#
# Training command
# ----------------
#   conda activate py310
#   cd /data/rnd-liu/MyRepo/mmdetection3d
#
#   # Detection only (test that new codebase matches original):
#   python projects/bevdet/train/train.py \
#       --config projects/bevdet/configs/mybevfusion_phase1.py \
#       --work-dir work_dirs/phase1_det \
#       --epochs 10
#
#   # Detection + binary occupancy:
#   python projects/bevdet/train/train.py \
#       --config projects/bevdet/configs/mybevfusion_phase1.py \
#       --work-dir work_dirs/phase1_det_occ \
#       --epochs 10 \
#       --occ-classes 2 \
#       --occ-num-z 16 \
#       --occ-loss-weight 0.5
#
#   # Evaluate only:
#   python projects/bevdet/train/train.py \
#       --config projects/bevdet/configs/mybevfusion_phase1.py \
#       --eval-only \
#       --load-from work_dirs/phase1_det/epoch_10.pth \
#       --data-root /data/nuscenes \
#       --work-dir work_dirs/phase1_det
# ============================================================

_base_ = ['./mybevfusion12v2.py']

# ---- import new modules via custom_imports -------------------------
# Add multitask_bev so MODELS.register_module() fires for
# MultiTaskBEVFusion.  The occ_head / misc modules are pure PyTorch
# and do not need MODELS registration.
custom_imports = dict(
    imports=[
        # Existing modules (from mybevfusion12v2.py base)
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.bevfusion_ca',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.freeze_utils',
        'projects.bevdet.fire_rpf_necks',
        # New training modules
        'projects.bevdet.train.multitask_bev',   # registers MultiTaskBEVFusion
    ],
    allow_failed_imports=False,
)

# ---- Top-level occupancy config (read by train.py if --occ-classes=0) ---
# These values serve as documentation and defaults when NOT using --occ-* flags.
# The train.py CLI args override these; the config values are used when running
# eval_runner.py standalone (no CLI args).
occ_head_cfg = dict(
    in_channels=256,         # must match ConvFuser out_channels
    num_classes=2,           # 2 = binary pseudo-occ; 17 = Occ3D-nuScenes
    num_z=16,                # Z slices:  [-5, 3] / 0.5 m = 16
    z_range=[-5.0, 3.0],
    hidden_channels=128,
    lovasz_weight=1.0,
    class_weights=None,      # set to OCC3D_CLASS_WEIGHTS_17 for 17-class
)
occ_loss_weight = 0.5

# NOTE:
# The train.py script reads occ_head_cfg and occ_loss_weight from this config
# if --occ-classes > 0 is NOT explicitly set on the CLI.
# If you want to always train with occ, just pass --occ-classes 2 on the CLI.

# ---- No changes to model / data / schedule below -------------------------
# Everything else (BEVFusionCA, CrossAttnLSS, pipelines, scheduler,
# optim_wrapper, custom_hooks) is inherited verbatim from mybevfusion12v2.py.
