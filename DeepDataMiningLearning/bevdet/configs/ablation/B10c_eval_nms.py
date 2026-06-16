# =====================================================================
# Eval-only override for B10c with post-processing tweaks:
#   - Enable circular NMS at inference (was nms_type=None).
#   - Use the per-task radii from the original TransFusion paper:
#       * cars / trucks / buses / trailers / construction-vehicle /
#         barrier / motorcycle / bicycle (8 classes) → radius via nms_bev.
#       * pedestrian → circle_nms radius 0.175 m.
#       * traffic_cone → circle_nms radius 0.175 m.
#   - Raise score_threshold from 0.0 → 0.1 to drop low-confidence proposals
#     before NDS matching (extra noisy detections hurt mAVE / mAOE via
#     mismatched-pair selection).
#
# This is a pure POST-PROCESSING change on the existing B10c checkpoint.
# No retraining. Loading the same epoch_3.pth, evaluating with these
# overrides, and comparing to the baseline B10c eval (NDS 0.6792).
# =====================================================================

_base_ = ['./B10c_flow_guided_warmstart_fixed.py']

# Override the bbox_head's test_cfg + bbox_coder. Everything else
# inherited from B10c (model architecture, data pipeline, ann_file).
model = dict(
    bbox_head=dict(
        test_cfg=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 41],
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            pc_range=[-54.0, -54.0],
            # Enable circular NMS at inference.
            nms_type='circle',
            # Required by nms_bev path for the 8-class group (cars, etc.).
            # These bound how many proposals enter / leave the NMS step.
            pre_maxsize=1000,
            post_maxsize=83,
        ),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.1,       # ← was 0.0
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            code_size=10,
        ),
    ),
)
