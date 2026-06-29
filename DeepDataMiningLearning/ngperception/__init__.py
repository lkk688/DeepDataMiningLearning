"""
ngperception — downstream perception tasks on top of ngdet's 2D detection.

A pluggable, teaching-oriented suite of *downstream* perception heads that consume the
same images (and optionally the same ngdet `Detection` boxes) and add per-object/scene
attributes. Each task mirrors ngdet's three-layer design — estimators/datasets/evaluator
— so you can compare many models (basic -> SOTA) with one command, then fine-tune.

Tasks (built incrementally):
    depth/        monocular depth & per-object distance   (first)
    segmentation/ semantic / instance / panoptic          (planned)
    tracking/     multi-object tracking over ngdet boxes   (planned)
    lane/         lane detection                           (planned)
"""
