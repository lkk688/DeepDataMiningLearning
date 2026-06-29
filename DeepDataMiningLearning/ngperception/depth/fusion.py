"""
ngperception.depth.fusion
=========================

The **composable add-on**: combine an ngdet `Detection` (2D boxes) with a `DepthResult`
to get **per-object distance** — "the car is 18 m away". This is what makes depth a
downstream head of detection rather than a separate pipeline.

For each box we take a robust depth over its *central inner region* (boxes include
background near their edges, so the center is a better object estimate) and reduce with
the median. With a **metric** depth model the distance is in meters; with a relative
model it is only an ordering (we say so).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ObjectDistance:
    box: np.ndarray          # xyxy pixels
    name: str                # class name (from the detector), or ""
    score: float
    distance: float          # meters if metric else arbitrary units
    metric: bool


def per_object_distance(boxes_xyxy: np.ndarray, depth: np.ndarray, metric: bool,
                        names: Optional[List[str]] = None,
                        scores: Optional[np.ndarray] = None,
                        inner: float = 0.6) -> List[ObjectDistance]:
    """Distance per box = median depth over the central `inner` fraction of the box.

    Parameters
    ----------
    boxes_xyxy : (N,4) array of pixel boxes (x1,y1,x2,y2).
    depth : H×W depth map (larger = farther), from a `DepthResult`.
    metric : whether `depth` is in meters.
    inner : keep the central fraction of each box (0.6 -> drop a 20% border each side).
    """
    h, w = depth.shape
    out: List[ObjectDistance] = []
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes_xyxy, float)):
        bw, bh = x2 - x1, y2 - y1
        mx, my = bw * (1 - inner) / 2, bh * (1 - inner) / 2
        cx1, cy1 = int(max(0, x1 + mx)), int(max(0, y1 + my))
        cx2, cy2 = int(min(w, x2 - mx)), int(min(h, y2 - my))
        if cx2 <= cx1 or cy2 <= cy1:
            dist = float("nan")
        else:
            patch = depth[cy1:cy2, cx1:cx2]
            patch = patch[np.isfinite(patch) & (patch > 0)]
            dist = float(np.median(patch)) if patch.size else float("nan")
        out.append(ObjectDistance(
            box=np.asarray([x1, y1, x2, y2], np.float32),
            name=names[i] if names is not None and i < len(names) else "",
            score=float(scores[i]) if scores is not None and i < len(scores) else 1.0,
            distance=dist, metric=metric))
    return out


def distances_from_detection(detection, depth_result, inner: float = 0.6):
    """Convenience: take an ngdet `Detection` + a `DepthResult` -> list[ObjectDistance]."""
    return per_object_distance(
        detection.boxes, depth_result.depth, depth_result.metric,
        names=list(detection.names), scores=detection.scores, inner=inner)


def draw_distances(image, objects: List[ObjectDistance]):
    """Return a PIL image with boxes + distance labels drawn (for quick visualization)."""
    from PIL import ImageDraw
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for o in objects:
        x1, y1, x2, y2 = o.box.tolist()
        d.rectangle([x1, y1, x2, y2], outline=(255, 80, 0), width=3)
        unit = "m" if o.metric else ""
        label = f"{o.name} {o.distance:.1f}{unit}" if np.isfinite(o.distance) else o.name
        d.text((x1 + 2, max(0, y1 - 11)), label, fill=(255, 255, 0))
    return im


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.depth.fusion
# Synthetic check: a depth ramp + two boxes -> nearer box reports smaller distance.
# ===========================================================================
if __name__ == "__main__":
    depth = np.tile(np.linspace(5, 50, 400), (200, 1)).astype(np.float32)  # left near, right far
    boxes = np.array([[10, 50, 90, 150], [300, 50, 380, 150]], float)
    for o in per_object_distance(boxes, depth, metric=True, names=["car", "car"]):
        print(f"  box x~{o.box[0]:.0f}: {o.distance:.1f}m")
