# =====================================================================
# B21 — Mixed-MoreGT. Matched-budget control for Mixed+PL-Scaled (B20).
#
# Mixed (B14):         GT(23 segs, 4453 frames) + nuScenes
# Mixed+PL-Sc (B20):   GT(23 segs) + nuScenes + 3409 PL-labeled frames
#                                                from 37 new segments
# Mixed-MoreGT (B21):  GT(23 segs) + nuScenes + 3636 GT-labeled frames
#                                                from SAME 37 new segments
#
# B21 isolates a single variable from B20: the labels on the new 3,636
# frames are real Waymo GT (109K car/60K ped/1,123 cyc) instead of
# our geometric-fusion PL (22.6K car/9K ped/237 cyc). The eval gap
# (B21 - B20) quantifies "what's the price of using auto-labels".
# =====================================================================

_base_ = ['./B18_pseudo_label_v14.py']

MOREGT_INFO_PATH = (
    '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
    'data/waymo_finetune/waymo_v1_infos_train_pl_frames_gt.pkl')

moregt_dataset = dict(
    type='WaymoFineTuneDataset',
    data_root=_base_.data_root_waymo,
    ann_file=MOREGT_INFO_PATH,
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
        datasets=[_base_.waymo_dataset,    # Mixed's 23 segments GT
                  _base_.nus_dataset,      # nuScenes mkf30
                  moregt_dataset],         # NEW: 3,636 frames with real Waymo GT
        ignore_keys=['version', 'dataset', 'categories', 'info_version',
                     'sample_idx_to_data_idx', 'source_jsonl'],
    ),
)
