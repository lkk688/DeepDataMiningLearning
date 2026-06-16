# =====================================================================
# B22 — Q-Mix v0 (Quality-Controlled Multi-Source Training)
#
# Same data composition as B20 (Mixed+PL-Scaled):
#   Waymo GT (23 segs, 4,453 frames) + nuScenes 25% mkf30
#                                     + 3,409 PL frames
#
# Difference vs B20: every PL instance carries a calibrated reliability
# weight r_i ∈ [0.5, 1.0] from build_qmix_info.py, and a custom head
# (QMixTransFusionHead) scales the per-positive classification and
# regression loss by that weight. GT instances default to weight=1.0,
# so behaviour on nuScenes and Waymo GT streams is unchanged.
#
# Q-Mix v0 isolates the "reliability weighting" component of the
# Q-Mix recipe; sampler (v1) and domain-aware heads (v2) are NOT yet
# enabled.
# =====================================================================

_base_ = ['./B18_pseudo_label_v14.py']

custom_imports = dict(
    imports=[
        'projects.bevdet.cross_attn_lss2',
        'projects.bevdet.mvx_voxel_painting',
        'projects.bevdet.bevfusion_ca',
        'projects.bevdet.multitoken_temporal_lidar',
        'projects.bevdet.temporal_lidar',
        'projects.bevdet.flow_guided_temporal',
        'projects.bevdet.waymo_finetune_dataset',
        'projects.bevdet.qmix',
    ],
    allow_failed_imports=False,
)

# ---- Q-Mix pseudo-label info (per-instance r_i baked in) ----
PSEUDO_INFO_QMIX = (
    '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
    'data/waymo_finetune/waymo_v1_infos_train_pseudo_qmix.pkl')

# ---- Rebuild both pipelines to use the Q-Mix transforms ----
# We swap LoadAnnotations3D → QMixLoadAnnotations3D and
# Pack3DDetInputs → QMixPack3DDetInputs while keeping every other
# pipeline step intact (so the only behavioural change is per-instance
# weight passthrough; geometry / augmentation are identical).

from copy import deepcopy

def _qmixify_pipeline(steps):
    """Swap the relevant transforms to QMix subclasses so the per-instance
    pseudo_weight is loaded, range/name-filtered in sync with the GT
    boxes, and packed into ``gt_instances_3d``."""
    out = []
    for s in steps:
        s = deepcopy(s)
        t = s.get('type', '')
        if t == 'LoadAnnotations3D':
            s['type'] = 'QMixLoadAnnotations3D'
        elif t == 'ObjectRangeFilter':
            s['type'] = 'QMixObjectRangeFilter'
        elif t == 'ObjectNameFilter':
            s['type'] = 'QMixObjectNameFilter'
        elif t == 'Pack3DDetInputs':
            s['type'] = 'QMixPack3DDetInputs'
            keys = list(s.get('keys', []))
            if 'gt_pseudo_weights' not in keys:
                keys.append('gt_pseudo_weights')
            s['keys'] = keys
        out.append(s)
    return out

waymo_train_pipeline_qmix = _qmixify_pipeline(_base_.waymo_train_pipeline)
nus_train_pipeline_qmix = _qmixify_pipeline(_base_.nus_train_pipeline)

# ---- Pseudo-label dataset (the QMix info, with per-instance r_i) ----
pseudo_dataset_qmix = dict(
    type='WaymoFineTuneDataset',
    data_root=_base_.data_root_waymo,
    ann_file=PSEUDO_INFO_QMIX,
    pipeline=waymo_train_pipeline_qmix,
    modality=_base_.input_modality,
    test_mode=False,
    metainfo=_base_.common_metainfo,
    box_type_3d='LiDAR',
    waymo_root='', waymo_split='',
    y_flip_on_load=False,
    backend_args=None,
)

# Wrap the existing waymo_dataset / nus_dataset with the Q-Mix pipelines
waymo_dataset_qmix = deepcopy(_base_.waymo_dataset)
waymo_dataset_qmix['pipeline'] = waymo_train_pipeline_qmix
nus_dataset_qmix = deepcopy(_base_.nus_dataset)
nus_dataset_qmix['pipeline'] = nus_train_pipeline_qmix

train_dataloader = dict(
    _delete_=True,
    batch_size=8, num_workers=4, persistent_workers=True, pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[waymo_dataset_qmix,    # Waymo GT (r=1.0)
                  nus_dataset_qmix,      # nuScenes GT (r=1.0)
                  pseudo_dataset_qmix],  # Waymo PL (r=calibrated)
        ignore_keys=['version', 'dataset', 'categories', 'info_version',
                     'sample_idx_to_data_idx', 'source_jsonl',
                     'qmix_alphas', 'qmix_P_cls'],
    ),
)

# ---- Swap the TransFusion head for the Q-Mix subclass ----
# The base config's model already specifies the head; we override its
# type to QMixTransFusionHead. All other model components unchanged.
model = dict(
    bbox_head=dict(
        type='QMixTransFusionHead',
    ),
)
