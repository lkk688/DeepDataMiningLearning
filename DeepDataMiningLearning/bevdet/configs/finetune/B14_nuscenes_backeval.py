# =====================================================================
# P0a: nuScenes back-eval of Mixed (B14). Tests whether mixed-source
# training preserves source-domain performance — defuses the
# "Mixed trades nuScenes for Waymo" reviewer concern.
#
# Same architecture as B10c (B14 just has different fine-tuned weights);
# we inherit the B10c eval setup, which has the nuScenes val_dataloader
# and the official NuScenesMetric.
# =====================================================================
_base_ = ['../ablation/B10c_eval_nms.py']
