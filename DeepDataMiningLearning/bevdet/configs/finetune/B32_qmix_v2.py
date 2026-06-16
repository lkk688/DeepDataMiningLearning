# =====================================================================
# B32 - Q-Mix v2 (Dense-Q-Mix, minimal): informativeness-densified,
# unbiased frame resampling.
#
# Same data as B20 (Mixed+PL-Scaled: Waymo GT + nuScenes 25% + PL), plain
# TransFusionHead (NO loss weighting -- v0's biasing mistake). The only
# change vs B20 is the train sampler: QMixWeightedSampler resamples frames
# with replacement, proportional to per-frame informativeness
#   w = clip( 1 + sum_inst rarity(cls) x sparsity(pts) x range , 1, w_max )
# with pseudo-label reliability as a hard gate (no loss scaling) and a
# w_max cap so rare-class frames cannot dominate (fixes v1b overshoot).
#
# Motivated by dense learning (Feng & Liu, Nat. Commun. 2026): overcome the
# Curse of Rarity by unbiased resampling toward informative samples rather
# than biasing the gradient (v0) or naively duplicating a class (v1).
# =====================================================================

_base_ = ['./B20_pseudo_scaled.py']

custom_imports = dict(
    imports=[
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.bevfusion_ca',
        'projects.bevdet.multitoken_temporal_lidar',
        'projects.bevdet.temporal_lidar',
        'projects.bevdet.flow_guided_temporal',
        'projects.bevdet.waymo_finetune_dataset',
        'projects.bevdet.qmix',
    ],
    allow_failed_imports=False,
)

# Swap ONLY the sampler; dataset composition is inherited verbatim from B20.
train_dataloader = dict(
    sampler=dict(
        _delete_=True,
        type='QMixWeightedSampler',
        # Cyclist-focused informativeness: only the rare bottleneck classes
        # get a large bonus so cyclist frames stand out under per-source
        # normalization. nuScenes-10 idx: motorcycle=6, bicycle=7,
        # pedestrian=8; car/truck/etc. = 0 (well-detected, not the bottleneck).
        rarity_bonus={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
                      6: 3.0, 7: 5.0, 8: 0.8, 9: 0.0},
        geom_n_target=50.0,     # sparsity: fewer LiDAR pts -> harder
        range_norm=50.0,        # range: farther -> harder
        reliability_tau=0.0,    # gate off (scaled PL already pre-filtered)
        w_max=6.0,              # cap prevents v1b-style overshoot
    ),
)
