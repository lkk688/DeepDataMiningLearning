# =====================================================================
# B28 — Q-Mix v0b + v1 combined: cyclist frame x3 oversample AND per-
# instance v0b loss weighting (uniform-class, α_cls=0, α_geom/match=0.5).
#
# v1 alone (B26) ties B20 (Macro -0.001) with a +3% cyclist edge.
# v0b alone (B24) loses Macro -0.011 but has +5pp Cyclist AP @ 2m vs v0.
# Combined: data-side rare-class amplification + per-instance quality
# control. Tests whether the two mechanisms compose constructively.
# =====================================================================

_base_ = ['./B22_qmix_v0.py']

# v0b with cyclist frames x3-replicated (3,409 -> 3,769 frames)
PSEUDO_INFO_QMIX = (
    '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
    'data/waymo_finetune/waymo_v1_infos_train_pseudo_qmix_v0b_cycle3x.pkl')

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
