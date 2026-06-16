# =====================================================================
# B17 — Curated class-balanced Waymo + nuScenes mixed training.
#
# Identical recipe to B14 (V9) except the Waymo ann_file is switched
# from `waymo_v1_infos_train.pkl` (4,453 frames, 62.8/36.4/0.8% class
# share) to `waymo_v1_infos_train_curated.pkl` (11,657 frames,
# 50.5/48.1/1.4% class share). Curation logic (see
# detection3d/phase2a/build_curated_waymo_train.py):
#
#   - Cyclist frames (>=1 cyclist):  5× replication  → +4,048 frames
#   - Ped-rich frames (>=5 peds):    2× replication  → +2,312 frames
#   - Other frames:                  1× (unchanged)
#
# Hypothesis: V9/V10/V12 plateau on Ped/Cyc because the per-batch class
# distribution mirrors the natural long-tail. Rebalancing the dataset
# (no synthetic data, no architecture change, no extra epochs) is the
# class-balance lever the data-wall thesis predicts will move Ped/Cyc
# while preserving Vehicle. Will be reported as V13 in the paper.
#
# Mixed-training preserved (nuScenes 25% mkf30 stream) to prevent
# the catastrophic forgetting that ruined V7.
# =====================================================================

_base_ = ['./B14_waymo_nuscenes_mixed.py']

# Single-attribute override: the curated info file. Everything else
# (model, pipeline, LR, freezing, mixing ratio, 1 epoch) is identical
# to B14 so any AP delta is attributable to the curated class balance.
waymo_dataset = dict(
    type='WaymoFineTuneDataset',
    data_root=_base_.data_root_waymo,
    ann_file='waymo_v1_infos_train_curated.pkl',  # <-- the only change
    pipeline=_base_.waymo_train_pipeline,
    modality=_base_.input_modality,
    test_mode=False,
    metainfo=_base_.common_metainfo,
    box_type_3d='LiDAR',
    waymo_root='', waymo_split='',
    y_flip_on_load=False,
    backend_args=None,
)

train_dataloader = dict(
    _delete_=True,
    batch_size=8, num_workers=4, persistent_workers=True, pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[waymo_dataset, _base_.nus_dataset],
        ignore_keys=['version', 'dataset', 'categories', 'info_version',
                     'sample_idx_to_data_idx'],
    ),
)

# Curated set is 2.62× bigger, so 1 epoch is now ~2.6× the gradient
# updates of B14. Schedule unchanged (cosine over 1 epoch) — the larger
# effective iter count is exactly what we want for the rare classes.
train_cfg = dict(by_epoch=True, max_epochs=1)
