# ====== Base (adjust path to where you keep the official file) ======
_base_ = ['/data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py']

# Keep registry under mmdet3d unless you have custom scopes
default_scope = 'mmdet3d'

# ========= Data root (set your actual path!) =========
# Example:
# data_root = '/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes/'
data_root = '/data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes/'

# ========= Tweaks for stability & speed on your stack =========
# - AMP via AmpOptimWrapper
# - EMA for better NDS
# - CBGS to fight long-tail classes
# - More sweeps (typical: 10~15) for better accuracy
# - Cosine schedule over 24 epochs

backend_args = None
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

# ----- Train / Test pipelines (override sweeps if you want) -----
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, load_dim=5, use_dim=5,
         pad_empty_sweeps=True, remove_close=True, backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectSample', db_sampler=dict(
        data_root=data_root,
        info_path=data_root + 'nuscenes_dbinfos_train.pkl',
        rate=1.0,
        prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(car=5, truck=5, bus=5, trailer=5,
                                                                          construction_vehicle=5, pedestrian=5,
                                                                          traffic_cone=5, barrier=5,
                                                                          motorcycle=5, bicycle=5)),
        classes=['car','truck','construction_vehicle','bus','trailer',
                 'barrier','motorcycle','bicycle','pedestrian','traffic_cone'],
        sample_groups=dict(car=2, truck=3, construction_vehicle=7, bus=4, trailer=6,
                           barrier=2, motorcycle=6, bicycle=6, pedestrian=2, traffic_cone=2),
        points_loader=dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5))),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05], translation_std=[0.2,0.2,0.2]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points','gt_bboxes_3d','gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, load_dim=5, use_dim=5,
         pad_empty_sweeps=True, remove_close=True, backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# ----- Wrap train dataset with CBGS -----
train_dataloader = dict(
    batch_size=4,  # 8x4 in the original name refers to 8 GPUs * 4 imgs/GPU; here adapt to your GPU count
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type='NuScenesDataset',
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=dict(classes=[
                'car','truck','construction_vehicle','bus','trailer',
                'barrier','motorcycle','bicycle','pedestrian','traffic_cone'
            ]),
            modality=dict(use_lidar=True, use_camera=False),
            test_mode=False,
            box_type_3d='LiDAR',
            backend_args=backend_args
        )
    )
)

val_dataloader = dict(
    batch_size=4, num_workers=8, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

# ----- Optimizer + AMP -----
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'norm': dict(weight_decay=0.0)})
)

# ----- Scheduler (24 epochs cosine) -----
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0/3.0, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=24, eta_min_ratio=1e-4, by_epoch=True, convert_to_iter_based=True),
]

# ----- Runtime -----
train_cfg = dict(by_epoch=True, max_epochs=24, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# ----- EMA -----
custom_hooks = [
    dict(type='EMAHook', ema_type='ExponentialMovingAverage', momentum=0.0002, update_buffers=True),
]

# ----- Checkpoint & logging -----
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=2)
)

# ----- Evaluator (nuScenes NDS / mAP) -----
val_evaluator = dict(type='NuScenesMetric', data_root=data_root, metric='bbox')
test_evaluator = val_evaluator

# ----- Misc -----
env_cfg = dict(cudnn_benchmark=True)
randomness = dict(seed=42, deterministic=False)
auto_scale_lr = dict(enable=False, base_batch_size=32)