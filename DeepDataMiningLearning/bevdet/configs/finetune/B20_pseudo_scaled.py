# =====================================================================
# B20 — Mixed+PL-Scaled. Identical recipe to B18 but with 3,409 PL
# frames (vs 868) by adding 27 more Waymo training tfrecords to the
# pseudo-label generation. Class counts in the PL stream:
#   Vehicle    22,574  (4.4× B18)
#   Pedestrian  8,978  (3.6× B18)
#   Cyclist       237  (11.3× B18)
# Tests whether the +0.013 Cyclist trend at small scale converts to
# a statistically significant gain when cyclist PL count scales up.
# =====================================================================

_base_ = ['./B18_pseudo_label_v14.py']

PSEUDO_INFO_SCALED = (
    '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
    'data/waymo_finetune/waymo_v1_infos_train_pseudo_scaled.pkl')

pseudo_dataset_scaled = dict(
    type='WaymoFineTuneDataset',
    data_root=_base_.data_root_waymo,
    ann_file=PSEUDO_INFO_SCALED,
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
        datasets=[_base_.waymo_dataset,
                  _base_.nus_dataset,
                  pseudo_dataset_scaled],
        ignore_keys=['version', 'dataset', 'categories', 'info_version',
                     'sample_idx_to_data_idx', 'source_jsonl'],
    ),
)
