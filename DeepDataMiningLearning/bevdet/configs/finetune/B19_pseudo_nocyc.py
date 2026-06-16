# =====================================================================
# B19 — Mixed+PL with all Cyclist pseudo-labels REMOVED. S3 ablation.
#
# If Cyclist AP at eval still beats Mixed (V9) despite zero cyclist
# instances in the pseudo-label stream, the +19% gain is from general
# scene-diversity exposure, not the 21 cyclist labels.
#
# Result will be reported as V14-NoCyc.
# =====================================================================

_base_ = ['./B18_pseudo_label_v14.py']

PSEUDO_INFO_PATH_NOCYC = (
    '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
    'data/waymo_finetune/waymo_v1_infos_train_pseudo_nocyc.pkl')

pseudo_dataset_nocyc = dict(
    type='WaymoFineTuneDataset',
    data_root=_base_.data_root_waymo,
    ann_file=PSEUDO_INFO_PATH_NOCYC,
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
                  pseudo_dataset_nocyc],
        ignore_keys=['version', 'dataset', 'categories', 'info_version',
                     'sample_idx_to_data_idx', 'source_jsonl'],
    ),
)
