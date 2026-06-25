"""
ngdet.evaluator
==============

Standard COCO mAP evaluation on the UNIFIED taxonomy, using `pycocotools`
(the same engine the project's `detection/cocoevaluator.py` wraps). We build an
in-memory COCO ground-truth object and a COCO results list from unified-space
predictions, then run `COCOeval` -- so students get the exact, canonical COCO
metric (AP@[.5:.95], AP50, AP75, AP per class) for ANY model on ANY dataset.

Everything here operates purely on unified ids, so it is model- and
dataset-agnostic: feed it the GT samples and the per-image Detections and it
returns a metrics dict.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np


class COCOUnifiedEvaluator:
    """Accumulate predictions vs unified GT and compute COCO mAP.

    Usage
    -----
    ev = COCOUnifiedEvaluator(taxonomy)
    for sample in dataset:
        det = detector.predict(sample.image)
        ev.add(sample, det)
    metrics = ev.summarize()
    """

    def __init__(self, taxonomy):
        self.taxonomy = taxonomy
        # COCO category ids must be >=1; map unified id u -> coco cat id u+1.
        self._cats = [{"id": u + 1, "name": n}
                      for u, n in enumerate(taxonomy.classes)]
        self._images: List[dict] = []
        self._gt_anns: List[dict] = []
        self._dets: List[dict] = []
        self._ann_id = 1
        self._seen = set()

    @staticmethod
    def _xyxy_to_xywh(b):
        x1, y1, x2, y2 = [float(v) for v in b]
        return [x1, y1, x2 - x1, y2 - y1]

    def add(self, sample, detection):
        """Register one image's GT (from EvalSample) and predictions (Detection)."""
        img_id = int(sample.image_id)
        if img_id in self._seen:
            raise ValueError(f"Duplicate image_id {img_id}; ids must be unique.")
        self._seen.add(img_id)
        w, h = sample.image.size
        self._images.append({"id": img_id, "width": w, "height": h})

        # Ground-truth annotations.
        for box, lab in zip(sample.gt_boxes, sample.gt_labels):
            xywh = self._xyxy_to_xywh(box)
            self._gt_anns.append({
                "id": self._ann_id,
                "image_id": img_id,
                "category_id": int(lab) + 1,
                "bbox": xywh,
                "area": float(xywh[2] * xywh[3]),
                "iscrowd": 0,
            })
            self._ann_id += 1

        # Detections (COCO results format).
        for box, sc, lab in zip(detection.boxes, detection.scores, detection.labels):
            self._dets.append({
                "image_id": img_id,
                "category_id": int(lab) + 1,
                "bbox": self._xyxy_to_xywh(box),
                "score": float(sc),
            })

    def summarize(self, verbose: bool = True) -> Dict[str, float]:
        """Run COCOeval and return a metrics dict (mAP, AP50, AP75, per-class AP)."""
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if not self._dets:
            if verbose:
                print("  [evaluator] no detections passed the threshold -> mAP 0")
            return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0,
                    "per_class": {n: 0.0 for n in self.taxonomy.classes}}

        # Build an in-memory COCO GT object (bypass file I/O).
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": self._images,
            "annotations": self._gt_anns,
            "categories": self._cats,
        }
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(self._dets)

        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()  # prints the canonical 12-line COCO table

        # Per-class AP@[.5:.95]: precision dims = [T, R, K, A, M].
        prec = ev.eval["precision"]  # T iou x R recall x K class x A area x M maxDet
        per_class = {}
        for k, name in enumerate(self.taxonomy.classes):
            p = prec[:, :, k, 0, -1]      # all-area, max-det=100
            p = p[p > -1]
            per_class[name] = float(p.mean()) if p.size else 0.0

        # ev.stats is the canonical 12-value COCO vector (6 AP + 6 AR).
        s = [float(x) for x in ev.stats]
        metrics = {
            "mAP": s[0],          # AP @ [.5:.95]  | all     | maxDets=100
            "AP50": s[1],         # AP @ .50       | all     | maxDets=100
            "AP75": s[2],         # AP @ .75       | all     | maxDets=100
            "AP_small": s[3],
            "AP_medium": s[4],
            "AP_large": s[5],
            "AR_1": s[6],         # AR             | all     | maxDets=1
            "AR_10": s[7],        # AR             | all     | maxDets=10
            "AR_100": s[8],       # AR             | all     | maxDets=100
            "AR_small": s[9],
            "AR_medium": s[10],
            "AR_large": s[11],
            "coco_stats": s,      # full 12-vector, in canonical order
            "per_class": per_class,
        }
        return metrics


# Canonical COCO metric row labels, in ev.stats order (for pretty tables).
COCO_STAT_LABELS = [
    "AP  @[IoU=0.50:0.95 | area=   all | maxDets=100]",
    "AP  @[IoU=0.50      | area=   all | maxDets=100]",
    "AP  @[IoU=0.75      | area=   all | maxDets=100]",
    "AP  @[IoU=0.50:0.95 | area= small | maxDets=100]",
    "AP  @[IoU=0.50:0.95 | area=medium | maxDets=100]",
    "AP  @[IoU=0.50:0.95 | area= large | maxDets=100]",
    "AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1]",
    "AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10]",
    "AR  @[IoU=0.50:0.95 | area=   all | maxDets=100]",
    "AR  @[IoU=0.50:0.95 | area= small | maxDets=100]",
    "AR  @[IoU=0.50:0.95 | area=medium | maxDets=100]",
    "AR  @[IoU=0.50:0.95 | area= large | maxDets=100]",
]


def format_coco_table(metrics: Dict) -> str:
    """Render a metrics dict as the full 12-row COCO table in Markdown.

    This is the complete canonical COCO evaluation (all 6 AP + 6 AR rows),
    not just mAP -- the same numbers `COCOeval.summarize()` prints, plus a
    per-class AP@[.5:.95] block.
    """
    stats = metrics.get("coco_stats", [metrics.get("mAP", 0.0)] + [0.0] * 11)
    lines = ["| Metric | Value |", "|---|---|"]
    for label, val in zip(COCO_STAT_LABELS, stats):
        lines.append(f"| {label} | {val:.3f} |")
    lines.append("")
    lines.append("| Class | AP@[.5:.95] |")
    lines.append("|---|---|")
    for name, ap in metrics.get("per_class", {}).items():
        lines.append(f"| {name} | {ap:.3f} |")
    return "\n".join(lines)


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# Self-contained synthetic sanity check (no models/datasets/network needed):
#
#   python -m DeepDataMiningLearning.ngdet.evaluator
#
# Expected: feeds perfect predictions == GT for 2 toy images and prints mAP ~1.0.
# ===========================================================================
if __name__ == "__main__":
    from types import SimpleNamespace
    from PIL import Image
    from .taxonomy import Taxonomy
    from .detectors.base import Detection

    tax = Taxonomy.from_preset("driving3")
    ev = COCOUnifiedEvaluator(tax)
    for i in range(2):
        gt_boxes = np.array([[10, 10, 60, 80], [100, 50, 200, 160]], np.float32)
        gt_labels = np.array([0, 1], np.int64)         # vehicle, person
        sample = SimpleNamespace(image_id=i, image=Image.new("RGB", (320, 240)),
                                 gt_boxes=gt_boxes, gt_labels=gt_labels)
        # perfect predictions (boxes identical to GT, score 0.99)
        det = Detection(boxes=gt_boxes.copy(),
                        scores=np.array([0.99, 0.99], np.float32),
                        labels=gt_labels.copy(),
                        names=["vehicle", "person"])
        ev.add(sample, det)
    m = ev.summarize()
    print(f"\nsanity mAP={m['mAP']:.3f} (expected ~1.0), per_class={m['per_class']}")
