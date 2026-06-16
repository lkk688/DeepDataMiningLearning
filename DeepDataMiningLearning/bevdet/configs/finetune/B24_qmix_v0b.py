# =====================================================================
# B24 — Q-Mix v0b (drop per-class component; geom+match only)
#
# v0 used α_cls=0.5 — a strong per-class precision prior — which
# discounted cyclist labels (P=0.392 → r ~0.35). v0b sets α_cls=0
# and re-balances the remaining weight to α_geom=0.5 + α_match=0.5.
# Mean weights become car 0.999, ped 0.926, bicycle 0.969, so all
# classes contribute roughly uniformly. Tests whether the per-class
# down-weighting was the actual failure mode of v0.
# =====================================================================

_base_ = ['./B22_qmix_v0.py']

PSEUDO_INFO_QMIX = (
    '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
    'data/waymo_finetune/waymo_v1_infos_train_pseudo_qmix_v0b.pkl')

pseudo_dataset_qmix = dict(_base_.pseudo_dataset_qmix)
pseudo_dataset_qmix['ann_file'] = PSEUDO_INFO_QMIX

train_dataloader = dict(
    _delete_=True,
    batch_size=8, num_workers=4, persistent_workers=True, pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[_base_.waymo_dataset_qmix,
                  _base_.nus_dataset_qmix,
                  pseudo_dataset_qmix],
        ignore_keys=['version', 'dataset', 'categories', 'info_version',
                     'sample_idx_to_data_idx', 'source_jsonl',
                     'qmix_alphas', 'qmix_P_cls'],
    ),
)
