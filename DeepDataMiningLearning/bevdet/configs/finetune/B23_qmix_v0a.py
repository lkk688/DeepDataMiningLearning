# =====================================================================
# B23 — Q-Mix v0a (threshold filter: drop PL with r_i < 0.7)
#
# v0 failed (Macro 0.158 vs B20 0.183) because per-class down-weighting
# reduces cyclist signal in proportion to noise. v0a takes the opposite
# approach: keep only PLs whose reliability passes a hard threshold,
# then weight survivors uniformly. With r>=0.7 the dataset still has
# 3,404 frames (vs 3,409 baseline) but is heavily concentrated at
# r=0.9 (29,077 of 31,451 instances).
#
# Note: all cyclist instances have r_i < 0.7 in the current calibration,
# so this variant intentionally tests whether removing low-confidence
# rare-class signal is recoverable from Waymo GT alone.
# =====================================================================

_base_ = ['./B22_qmix_v0.py']

PSEUDO_INFO_QMIX = (
    '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
    'data/waymo_finetune/waymo_v1_infos_train_pseudo_qmix_v0a.pkl')

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
