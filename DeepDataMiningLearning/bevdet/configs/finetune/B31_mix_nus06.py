# =====================================================================
# B31 — Mixed-source Pareto point: nuScenes 6.25% (quarter of B14's 25%)
#
# Identical to B14 except the nuScenes anchor is quartered (5,988 ->
# 1,741 samples). Lowest source-anchor Pareto point: expect the most
# Waymo-favoring behaviour and the largest nuScenes NDS drop.
# =====================================================================

_base_ = ['./B14_waymo_nuscenes_mixed.py']

nus_dataset = dict(_base_.nus_dataset)
nus_dataset['ann_file'] = 'nuscenes_infos_train_6pct_mkf30.pkl'

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
