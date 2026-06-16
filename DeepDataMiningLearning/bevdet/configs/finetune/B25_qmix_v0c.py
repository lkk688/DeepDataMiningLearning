# =====================================================================
# B25 — Q-Mix v0c (up-weight rare-class PLs: cyclist x1.5)
#
# Counterintuitive but data-supported: cyclist labels are NOISY (P=0.39),
# but the model emits FEWER cyclist predictions in v0 than the unweighted
# B20 baseline (8 vs 14 → AP@2m 0.061 vs 0.136). The diagnosis is that
# down-weighting collapses the loss signal for an already-rare class.
# v0c keeps the per-class precision prior but multiplies cyclist weights
# by 1.5 (mean_w=1.167 vs 0.78 in v0). Tests whether boosting rare-class
# gradient is the right intervention.
# =====================================================================

_base_ = ['./B22_qmix_v0.py']

PSEUDO_INFO_QMIX = (
    '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
    'data/waymo_finetune/waymo_v1_infos_train_pseudo_qmix_v0c.pkl')

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
