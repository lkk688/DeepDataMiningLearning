"""
ngperception.depth.evaluator
============================

Standard monocular-depth metrics (the Eigen et al. / KITTI set), computed on the LiDAR
GT pixels only. For **relative** models we first align the prediction to GT scale, since
their output is defined only up to scale.

Metrics (lower is better unless noted):
    AbsRel = mean(|d* - d| / d)
    SqRel  = mean((d* - d)^2 / d)
    RMSE   = sqrt(mean((d* - d)^2))                     [meters]
    RMSElog= sqrt(mean((log d* - log d)^2))
    delta1 = % of px with max(d*/d, d/d*) < 1.25        [higher better]
    delta2 = ... < 1.25^2 ;  delta3 = ... < 1.25^3      [higher better]

Alignment for relative models:
    median scaling (Monodepth2 protocol): d* <- d* * median(d_gt) / median(d*).
This removes the unknown global scale; it does NOT fix per-image shift, so relative
models are still at a slight disadvantage vs metric ones — which is the honest comparison.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np

_MIN_DEPTH, _MAX_DEPTH = 1e-3, 80.0      # KITTI evaluation cap


def _metrics_one(pred: np.ndarray, gt: np.ndarray, align: bool,
                 garg_crop: bool = True) -> Dict[str, float]:
    """Per-image metrics on valid GT pixels. Returns dict + 'n' (pixel count).

    `align=True` median-scales the prediction to GT (scale-invariant accuracy, fair across
    relative & metric models). `align=False` scores the raw prediction in its own units
    (only meaningful — i.e. true metres — for a metric model)."""
    h, w = gt.shape
    valid = (gt > _MIN_DEPTH) & (gt < _MAX_DEPTH)
    if garg_crop:                          # the standard KITTI 'Garg' center crop
        crop = np.zeros_like(valid)
        y1, y2 = int(0.40810811 * h), int(0.99189189 * h)
        x1, x2 = int(0.03594771 * w), int(0.96405229 * w)
        crop[y1:y2, x1:x2] = True
        valid &= crop
    if valid.sum() == 0:
        return {}

    p = pred[valid].astype(np.float64).copy()
    g = gt[valid].astype(np.float64)
    if align:                              # median scaling removes the global scale
        p *= np.median(g) / max(np.median(p), 1e-12)
    p = np.clip(p, _MIN_DEPTH, _MAX_DEPTH)

    thresh = np.maximum(p / g, g / p)
    rmse = np.sqrt(np.mean((p - g) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))
    return {
        "AbsRel": float(np.mean(np.abs(p - g) / g)),
        "SqRel": float(np.mean((p - g) ** 2 / g)),
        "RMSE": float(rmse),
        "RMSElog": float(rmse_log),
        "delta1": float(np.mean(thresh < 1.25)),
        "delta2": float(np.mean(thresh < 1.25 ** 2)),
        "delta3": float(np.mean(thresh < 1.25 ** 3)),
        "n": int(valid.sum()),
    }


class DepthEvaluator:
    """Accumulates per-image metrics and reports the dataset mean (image-averaged).

    Always reports **aligned** (scale-invariant) metrics so relative and metric models are
    compared fairly. For metric models it additionally reports **as-metric** metrics (no
    alignment) — the true-metres accuracy that only a correctly-calibrated metric model
    can achieve."""

    KEYS = ("AbsRel", "SqRel", "RMSE", "RMSElog", "delta1", "delta2", "delta3")

    def __init__(self, is_metric: bool, garg_crop: bool = True):
        self.is_metric = is_metric
        self.garg_crop = garg_crop
        self._aligned: List[Dict[str, float]] = []
        self._metric: List[Dict[str, float]] = []

    def add(self, pred_depth: np.ndarray, gt_depth: np.ndarray):
        a = _metrics_one(pred_depth, gt_depth, align=True, garg_crop=self.garg_crop)
        if a:
            self._aligned.append(a)
        if self.is_metric:
            m = _metrics_one(pred_depth, gt_depth, align=False, garg_crop=self.garg_crop)
            if m:
                self._metric.append(m)

    def summarize(self, verbose: bool = True) -> Dict[str, float]:
        if not self._aligned:
            return {k: float("nan") for k in self.KEYS}
        out = {k: float(np.mean([r[k] for r in self._aligned])) for k in self.KEYS}
        out["num_images"] = len(self._aligned)
        out["metric"] = self.is_metric
        if self._metric:                   # true-metres scores (metric models only)
            for k in self.KEYS:
                out["m_" + k] = float(np.mean([r[k] for r in self._metric]))
        if verbose:
            msg = ("  images={} aligned: ".format(out["num_images"])
                   + " ".join(f"{k}={out[k]:.3f}" for k in ("AbsRel", "RMSE", "delta1")))
            if self._metric:
                msg += "  | metric: " + " ".join(
                    f"{k}={out['m_'+k]:.3f}" for k in ("AbsRel", "RMSE", "delta1"))
            print(msg)
        return out


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.depth.evaluator
# Synthetic sanity check: a perfect prediction scores AbsRel~0, delta1~1.
# ===========================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gt = rng.uniform(5, 60, size=(370, 1224)).astype(np.float32)
    ev = DepthEvaluator(is_metric=True, garg_crop=False)
    ev.add(gt.copy(), gt)                                  # perfect
    print("perfect prediction:"); ev.summarize()
    ev2 = DepthEvaluator(is_metric=False, garg_crop=False)
    ev2.add(1.0 / gt, gt)                                  # relative (inverse), median-aligned
    print("inverse+median-scaled (should still be poor unless monotonic):"); ev2.summarize()
