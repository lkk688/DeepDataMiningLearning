# =====================================================================
# B30 — Mixed-source Pareto point: nuScenes 12.5% (half of B14's 25%)
#
# Identical to B14 (Waymo GT + nuScenes mixed, no PL) except the
# nuScenes anchor is halved (5,988 -> 2,881 samples). Used to trace the
# source-retention (nuScenes NDS) vs target-gain (Waymo Macro) Pareto
# front controlled by the mix ratio.
# =====================================================================

_base_ = ['./B14_waymo_nuscenes_mixed.py']

nus_dataset = dict(_base_.nus_dataset)
nus_dataset['ann_file'] = 'nuscenes_infos_train_12pct_mkf30.pkl'

train_dataloader = dict(
    _delete_=True,
    batch_size=8, num_workers=4, persistent_workers=True, pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[_base_.waymo_dataset, nus_dataset],
        ignore_keys=['version', 'dataset', 'categories', 'info_version',
                     'sample_idx_to_data_idx'],
    ),
)
