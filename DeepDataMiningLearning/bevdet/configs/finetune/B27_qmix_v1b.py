# =====================================================================
# B27 — Q-Mix v1b (cyclist x5 frame oversampling, plain TF head)
#
# v1 (B26, x3 oversample) tied B20 macro with +3% cyclist edge on the
# 300-frame subset. v1b pushes the oversampling factor to x5 to test
# whether stronger rare-class emphasis converts the tie into a win.
# Cyclist instance count: 237 -> 1185 (5x); total PL frames 3,409 -> 4,129.
# =====================================================================

_base_ = ['./B18_pseudo_label_v14.py']

PSEUDO_INFO_PATH = (
    '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
    'data/waymo_finetune/waymo_v1_infos_train_pseudo_v1b_cycle5x.pkl')

pseudo_dataset = dict(_base_.pseudo_dataset)
pseudo_dataset['ann_file'] = PSEUDO_INFO_PATH

train_dataloader = dict(
    _delete_=True,
    batch_size=8, num_workers=4, persistent_workers=True, pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[_base_.waymo_dataset,
                  _base_.nus_dataset,
                  pseudo_dataset],
        ignore_keys=['version', 'dataset', 'categories', 'info_version',
                     'sample_idx_to_data_idx', 'source_jsonl'],
    ),
)
