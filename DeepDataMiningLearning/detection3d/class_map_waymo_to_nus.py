"""
Cross-taxonomy mapping between nuScenes (10 classes used by B10c) and
Waymo Perception (4 classes — but we only evaluate on 3 because Waymo's
"Sign" has no nuScenes counterpart).

Maps in both directions for the zero-shot transfer eval:

  ┌──────────────────────────────────┬──────────────────┐
  │ nuScenes (B10c outputs)          │ Waymo (GT)       │
  ├──────────────────────────────────┼──────────────────┤
  │ car, truck, bus, trailer,        │  1 Vehicle       │
  │ construction_vehicle             │                  │
  ├──────────────────────────────────┼──────────────────┤
  │ pedestrian                       │  2 Pedestrian    │
  ├──────────────────────────────────┼──────────────────┤
  │ motorcycle, bicycle              │  4 Cyclist       │
  ├──────────────────────────────────┼──────────────────┤
  │ barrier, traffic_cone            │  (no equivalent — Waymo's 'Sign' is
  │                                  │   pole-shaped, totally different)
  └──────────────────────────────────┴──────────────────┘

For zero-shot eval we collapse to **3 evaluation classes** matching
Waymo's annotations. nuScenes detections in the "no equivalent" buckets
(barrier, traffic_cone) are dropped; Waymo Sign GT (label 3) is also
dropped so we don't penalize the model for not predicting a class it
was never trained on.

This is the standard cross-dataset transfer protocol (e.g. used in
"Trans3D" and "Unidet3D" — generalize-to-Waymo papers).
"""

# nuScenes detection class ordering (matches DET_CLASSES_10 in tracker3d.py)
NUS_DET_CLASSES_10 = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# Waymo class IDs (1=Vehicle, 2=Pedestrian, 3=Sign, 4=Cyclist)
WAYMO_CLASSES = {
    1: 'Vehicle',
    2: 'Pedestrian',
    3: 'Sign',
    4: 'Cyclist',
}

# 3-class transfer taxonomy (the bucket each side maps to)
TRANSFER_CLASSES = ['Vehicle', 'Pedestrian', 'Cyclist']

# nuScenes class index -> transfer bucket name, or None to drop
NUS_TO_TRANSFER = {
    0: 'Vehicle',      # car
    1: 'Vehicle',      # truck
    2: 'Vehicle',      # construction_vehicle
    3: 'Vehicle',      # bus
    4: 'Vehicle',      # trailer
    5: None,           # barrier      — no Waymo equivalent
    6: 'Cyclist',      # motorcycle
    7: 'Cyclist',      # bicycle
    8: 'Pedestrian',   # pedestrian
    9: None,           # traffic_cone — no Waymo equivalent
}

# Waymo class ID -> transfer bucket name, or None to drop
WAYMO_TO_TRANSFER = {
    1: 'Vehicle',
    2: 'Pedestrian',
    3: None,           # Sign — no nuScenes equivalent (would be a false-FN)
    4: 'Cyclist',
}


def nus_label_to_transfer(label_idx: int) -> str:
    """nuScenes class index (0-9) → transfer-bucket name, or None."""
    return NUS_TO_TRANSFER.get(int(label_idx))


def waymo_type_to_transfer(waymo_type: int) -> str:
    """Waymo class ID (1-4) → transfer-bucket name, or None."""
    return WAYMO_TO_TRANSFER.get(int(waymo_type))


def filter_nus_dets_to_transfer(boxes, scores, labels):
    """
    Drop nuScenes detections whose class has no Waymo equivalent.
    Returns (boxes, scores, transfer_class_names) ready for transfer eval.
    """
    keep = []
    out_cls = []
    for i, lab in enumerate(labels):
        tcls = nus_label_to_transfer(int(lab))
        if tcls is None:
            continue
        keep.append(i)
        out_cls.append(tcls)
    if not keep:
        import numpy as np
        return (
            boxes[:0] if hasattr(boxes, '__getitem__') else boxes,
            scores[:0] if hasattr(scores, '__getitem__') else scores,
            [],
        )
    import numpy as np
    keep = np.asarray(keep, dtype=np.int64)
    return boxes[keep], scores[keep], out_cls
