# =====================================================================
# Track A2 — noise-vs-partiality factorial, cell (precision .080, recall .100).
#
# Identical to B18 (M_pl) in every respect EXCEPT the pseudo-label
# ann_file, which is a synthetic label set built from real Waymo GT at a
# controlled operating point by analysis/a2_make_synthetic_labels.py.
# Frames, images, LiDAR, pipeline, schedule and the nuScenes anchor are
# held fixed, so any difference between cells is attributable to the
# label statistics alone.
# =====================================================================

_base_ = ['./B14_waymo_nuscenes_mixed.py']

A2_INFO = (
    '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
    'data/waymo_finetune/waymo_v1_infos_train_a2_p080_r100.pkl')

pseudo_dataset = dict(
    type='WaymoFineTuneDataset',
    data_root=_base_.data_root_waymo,
    ann_file=A2_INFO,
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
        datasets=[_base_.waymo_dataset, _base_.nus_dataset, pseudo_dataset],
        ignore_keys=['version', 'dataset', 'categories', 'info_version',
                     'sample_idx_to_data_idx', 'source_jsonl'],
    ),
)

train_cfg = dict(by_epoch=True, max_epochs=1)
