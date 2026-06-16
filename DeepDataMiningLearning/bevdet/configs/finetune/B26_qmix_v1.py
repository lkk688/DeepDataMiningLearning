# =====================================================================
# B26 — Q-Mix v1 (Sampler approach: frame-level cyclist oversampling)
#
# v0 family (B22/B23/B24/B25) showed conclusively that per-instance
# *loss* reweighting hurts more than it helps — the per-class precision
# prior over-suppresses cyclist gradient. v1 attacks the same problem
# from the data side: physically replicate cyclist-bearing PL frames
# so the standard DefaultSampler naturally sees them more often per
# epoch, without touching the loss.
#
# Implementation: same B18/B20 multi-source training (Waymo GT + nuScenes
# 25% mkf30 + PL frames), but the PL info.pkl uses cyclist frames ×3.
# This grows the PL subset from 3,409 → 3,769 frames (cyclist instances
# 237 → 711). No QMixTransFusionHead — back to plain TransFusionHead
# (the unweighted-loss winner from B20).
# =====================================================================

_base_ = ['./B18_pseudo_label_v14.py']

# v1 cyclist-oversampled PL info (3409 + 360 cyclist-frame duplicates = 3769)
PSEUDO_INFO_PATH = (
    '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
    'data/waymo_finetune/waymo_v1_infos_train_pseudo_v1_cycle3x.pkl')

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
                     'sample_idx_to_data_idx', 'source_jsonl',
                     'qmix_v1_repeat', 'qmix_v1_orig_n_frames',
                     'qmix_v1_n_cyclist_frames'],
    ),
)
