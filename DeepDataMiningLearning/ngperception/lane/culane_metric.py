"""
ngperception.lane.culane_metric
===============================

The **CULane F1** metric, pure numpy (no cv2, no external eval binary).

Official protocol: draw each lane as a thick curve (30 px wide on the 1640-px-wide
original), a predicted lane **matches** a GT lane if their masks' **IoU > 0.5**; then
greedily one-to-one match, count TP / FP / FN, and report

    precision = TP / (TP+FP),   recall = TP / (TP+FN),   F1 = 2PR/(P+R).

We rasterise in the (already resized) working resolution and scale the line width
accordingly (``width ≈ img_w/55`` ≈ the 30/1640 ratio). Lanes are point sets in the
same resized pixel space the model decodes to, so no coordinate juggling is needed.
"""
from __future__ import annotations
import numpy as np


def rasterize(pts, H, W, width):
    """Stamp a lane (K,2 array of (x,y)) as a thick polyline mask (H,W bool)."""
    mask = np.zeros((H, W), dtype=bool)
    pts = np.asarray(pts, dtype=np.float32)
    hw = max(1, int(round(width / 2)))
    for a, b in zip(pts[:-1], pts[1:]):
        steps = int(max(2, np.hypot(*(b - a))))
        xs = np.linspace(a[0], b[0], steps)
        ys = np.linspace(a[1], b[1], steps)
        for x, y in zip(xs, ys):
            xi, yi = int(round(x)), int(round(y))
            x0, x1 = max(0, xi - hw), min(W, xi + hw + 1)
            y0, y1 = max(0, yi - hw), min(H, yi + hw + 1)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = True
    return mask


def _match(preds, gts, H, W, width, iou_thresh):
    """Greedy IoU matching → (tp, fp, fn) for one image."""
    if len(gts) == 0:
        return 0, len(preds), 0
    if len(preds) == 0:
        return 0, 0, len(gts)
    pm = [rasterize(p, H, W, width) for p in preds]
    gm = [rasterize(g, H, W, width) for g in gts]
    iou = np.zeros((len(pm), len(gm)), dtype=np.float32)
    for i, a in enumerate(pm):
        asum = a.sum()
        for j, b in enumerate(gm):
            inter = np.logical_and(a, b).sum()
            union = asum + b.sum() - inter
            iou[i, j] = inter / union if union > 0 else 0.0
    tp = 0
    used_g = set()
    for i in np.argsort(-iou.max(axis=1)):          # preds by best-IoU first
        j = int(iou[i].argmax())
        if j not in used_g and iou[i, j] >= iou_thresh:
            tp += 1
            used_g.add(j)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


class CULaneF1:
    """Accumulate TP/FP/FN across images, then report precision / recall / F1."""

    def __init__(self, img_h=320, img_w=800, iou_thresh=0.5, width=None):
        self.H, self.W = img_h, img_w
        self.iou_thresh = iou_thresh
        self.width = width or max(4, round(img_w / 55))
        self.tp = self.fp = self.fn = 0

    def update(self, preds_batch, gts_batch):
        """preds_batch/gts_batch: list (len B) of lists of (K,2) point arrays (resized px)."""
        for preds, gts in zip(preds_batch, gts_batch):
            preds = [np.asarray(p) for p in preds]
            gts = [np.asarray(g) for g in gts]
            tp, fp, fn = _match(preds, gts, self.H, self.W, self.width, self.iou_thresh)
            self.tp += tp; self.fp += fp; self.fn += fn

    def compute(self):
        p = self.tp / (self.tp + self.fp + 1e-9)
        r = self.tp / (self.tp + self.fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        return {"precision": p, "recall": r, "f1": f1,
                "tp": self.tp, "fp": self.fp, "fn": self.fn}
