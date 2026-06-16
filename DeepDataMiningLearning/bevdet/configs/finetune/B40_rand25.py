# Hard-mining efficiency test: pseudo-ONLY finetune on a rand25 subset of the
# Waymo pseudo pool (no GT mix), from the B14 pretrained init. Eval on Waymo GT.
_base_ = ['./B18_pseudo_label_v14.py']
SUB = '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_finetune/waymo_v1_infos_train_pseudo_qmix_v0b_rand25.pkl'
pseudo_dataset = dict(_base_.pseudo_dataset)
pseudo_dataset['ann_file'] = SUB
train_dataloader = dict(
    _delete_=True, batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True), dataset=pseudo_dataset)
train_cfg = dict(by_epoch=True, max_epochs=3, val_interval=99)
load_from = '/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/finetune_B14/epoch_1_weights.pth'
work_dir = '/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/work_dirs/B40_rand25'
