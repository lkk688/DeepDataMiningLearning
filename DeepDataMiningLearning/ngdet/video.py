"""
ngdet.video
==========

Render an evaluation video: for each frame, draw the model's predictions (and
optionally the ground-truth boxes) and write to an mp4 so the detector's behavior
on a dataset can be inspected visually.

We REUSE the drawing/encoding helpers already written for dataset verification in
`detection/verify_datasets_video.py` (to_bgr_uint8 / draw_boxes / VideoWriter),
so there is a single place that knows how to turn images+boxes into video frames.
"""

from __future__ import annotations
from typing import Optional

import numpy as np

# Reuse the existing, tested helpers from the detection package.
from DeepDataMiningLearning.detection.verify_datasets_video import (
    to_bgr_uint8, draw_boxes, VideoWriter,
)

# GT boxes are drawn in a fixed contrasting color (green) so they are easy to
# tell apart from the per-class colored predictions.
import cv2
_GT_COLOR = (0, 200, 0)  # BGR


def _draw_gt(bgr, boxes, names):
    for b, n in zip(boxes, names):
        x1, y1, x2, y2 = [int(round(float(v))) for v in b[:4]]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), _GT_COLOR, 1)
        cv2.putText(bgr, f"GT:{n}", (x1, max(11, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _GT_COLOR, 1, cv2.LINE_AA)
    return bgr


class EvalVideoWriter:
    """Accumulate annotated frames into one mp4.

    Parameters
    ----------
    path : str           output .mp4 path
    taxonomy : Taxonomy  unified label space (for GT class names)
    fps : int
    draw_gt : bool       also overlay ground-truth boxes (green) for comparison
    """

    def __init__(self, path: str, taxonomy, fps: int = 5, draw_gt: bool = True):
        self.taxonomy = taxonomy
        self.draw_gt = draw_gt
        self.vw = VideoWriter(path, fps=fps)
        self.path = path

    def add(self, sample, detection):
        """Draw one frame: predictions (colored by class) + optional GT (green)."""
        bgr = to_bgr_uint8(sample.image)
        # Predictions: reuse the palette-colored draw_boxes.
        draw_boxes(bgr, detection.boxes, detection.labels, self.taxonomy.classes)
        if self.draw_gt and len(sample.gt_boxes):
            gt_names = [self.taxonomy.classes[int(l)] for l in sample.gt_labels]
            _draw_gt(bgr, sample.gt_boxes, gt_names)
        # Header banner so it is obvious which is which.
        cv2.putText(bgr, "pred=color  GT=green", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        self.vw.write(bgr)

    def release(self):
        self.vw.release()


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# Renders a 3-frame KITTI clip with GT boxes only (no model needed):
#
#   python -m DeepDataMiningLearning.ngdet.video
#
# Expected: writes /tmp/ngdet_video_test.mp4 with green GT boxes.
# ===========================================================================
if __name__ == "__main__":
    from .taxonomy import Taxonomy
    from .datasets import EvalDataset
    from .detectors.base import Detection

    tax = Taxonomy.from_preset("driving3")
    ds = EvalDataset("kitti", "/mnt/e/Shared/Dataset/Kitti/", tax, max_images=3)
    vw = EvalVideoWriter("/tmp/ngdet_video_test.mp4", tax, fps=2, draw_gt=True)
    for s in ds:
        vw.add(s, Detection())  # empty predictions -> shows GT only
    vw.release()
    print("wrote /tmp/ngdet_video_test.mp4")
