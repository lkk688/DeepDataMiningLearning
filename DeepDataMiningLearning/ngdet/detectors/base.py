"""
ngdet.detectors.base
=====================

The pluggable detector contract. Every detector backend (HuggingFace DETR,
Ultralytics YOLO, NVIDIA LocateAnything, ...) is wrapped in a small adapter that
subclasses `BaseDetector` and is registered into `DETECTOR_REGISTRY` via the
`@register("name")` decorator.

A detector's ONLY job is:
    image (+ optional text prompt)  ->  Detection in the UNIFIED taxonomy

i.e. it must (1) run its native model, (2) convert outputs to absolute-pixel xyxy
boxes, and (3) fold its native class ids into unified ids using the active
`Taxonomy`. After that, the evaluator and video writer are model-agnostic.

This decoupling is the whole point of the framework: students can drop in a new
backbone by writing ~50 lines (one adapter) without touching evaluation, datasets,
or visualization.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type

import numpy as np


@dataclass
class Detection:
    """Model output for ONE image, already projected to the unified taxonomy.

    All fields are aligned by index (the i-th box has the i-th score/label/name).
    Boxes are absolute pixel coordinates in xyxy (x_min, y_min, x_max, y_max),
    matching the original (un-resized) image so they overlay correctly on it.
    """
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    names: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.boxes)


class BaseDetector:
    """Abstract base for all detector adapters.

    Subclasses must implement `predict`. Heavy imports (torch, transformers,
    ultralytics) belong INSIDE `__init__` of the subclass, never at module top
    level, so that `import ngdet` stays cheap and works even when a given backend
    is not installed.

    Parameters
    ----------
    taxonomy : Taxonomy
        The active unified label space; used to fold native classes into unified ids.
    device : str
        e.g. "cuda", "cuda:0", "cpu".
    score_thr : float
        Confidence threshold applied before returning detections. Keep this the
        SAME across models when comparing, or your mAP/ablation is not apples-to-apples.
    """

    #: human-readable backend family name (set by subclass)
    family: str = "base"
    #: True for text-promptable open-vocabulary detectors (LocateAnything, YOLO-World)
    is_open_vocab: bool = False

    def __init__(self, taxonomy, device: str = "cuda", score_thr: float = 0.3,
                 accel: str = "fp32", **kwargs):
        self.taxonomy = taxonomy
        self.device = device
        self.score_thr = score_thr
        # Acceleration backend for the latency study. Adapters interpret this:
        #   "fp32" (native baseline), "fp16" (half precision), "compile"
        #   (torch.compile), "tensorrt"/"onnx" (YOLO via Ultralytics export).
        # Adapters silently ignore modes they do not support.
        self.accel = accel
        # A {native_class_id -> unified_id_or_None} LUT. Closed-set adapters build
        # this once from their model's id2label in `__init__`.
        self.id_lut: Dict[int, Optional[int]] = {}

    # -- the one method subclasses must provide ------------------------------
    def predict(self, image, prompt: Optional[List[str]] = None) -> Detection:
        """Run the model on a single image and return a unified `Detection`.

        `image` is a PIL.Image (RGB). `prompt` is the unified class-name list for
        open-vocab detectors (ignored by closed-set detectors).
        """
        raise NotImplementedError

    # -- shared helper for closed-set adapters -------------------------------
    def _fold_to_unified(self, boxes: np.ndarray, scores: np.ndarray,
                         native_label_ids: np.ndarray) -> Detection:
        """Apply score threshold + native->unified class folding.

        Detections whose native class maps to None (not in the taxonomy) are
        dropped. Used by closed-set adapters (DETR, YOLO) after they produce
        native xyxy boxes/scores/ids.
        """
        keep_boxes, keep_scores, keep_labels, keep_names = [], [], [], []
        for box, sc, nid in zip(boxes, scores, native_label_ids):
            if sc < self.score_thr:
                continue
            uid = self.id_lut.get(int(nid), None)
            if uid is None:
                continue  # class not part of the active taxonomy -> ignore
            keep_boxes.append(box)
            keep_scores.append(float(sc))
            keep_labels.append(int(uid))
            keep_names.append(self.taxonomy.classes[uid])
        if not keep_boxes:
            return Detection()
        return Detection(
            boxes=np.asarray(keep_boxes, np.float32).reshape(-1, 4),
            scores=np.asarray(keep_scores, np.float32),
            labels=np.asarray(keep_labels, np.int64),
            names=keep_names,
        )


# ---------------------------------------------------------------------------
# Registry: maps a short backend key -> adapter class, so the CLI can spell
# "--models hf_detr:facebook/detr-resnet-50 yolo:yolo11x" and we resolve the
# class by the part before the colon.
# ---------------------------------------------------------------------------
DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {}


def register(name: str) -> Callable[[Type[BaseDetector]], Type[BaseDetector]]:
    def deco(cls: Type[BaseDetector]) -> Type[BaseDetector]:
        DETECTOR_REGISTRY[name] = cls
        return cls
    return deco


def build_detector(spec: str, taxonomy, device: str = "cuda",
                   score_thr: float = 0.3, accel: str = "fp32",
                   **kwargs) -> BaseDetector:
    """Instantiate a detector from a "key:model_name" spec string.

    Examples
    --------
    build_detector("hf_detr:facebook/detr-resnet-50", tax)
    build_detector("yolo:yolo11x", tax)
    build_detector("locate_anything:nvidia/LocateAnything-3B", tax)
    """
    # Import adapters here (not at top) so registration happens on demand and a
    # missing optional dependency only breaks the backend that needs it.
    from . import (  # noqa: F401  (side-effect: register)
        hf_detr, yolo, torchvision_det, grounding_dino, locate_anything,
    )

    # Optional per-model threshold override: "backend:model@0.05" lets open-vocab
    # families (whose score scale differs) use their own threshold for a fair
    # comparison, while closed-set models keep the shared default.
    if "@" in spec:
        spec, thr = spec.rsplit("@", 1)
        score_thr = float(thr)

    if ":" in spec:
        key, model_name = spec.split(":", 1)
    else:
        key, model_name = spec, None
    if key not in DETECTOR_REGISTRY:
        raise KeyError(
            f"Unknown detector backend '{key}'. Registered: {list(DETECTOR_REGISTRY)}"
        )
    return DETECTOR_REGISTRY[key](
        taxonomy=taxonomy, device=device, score_thr=score_thr,
        model_name=model_name, accel=accel, **kwargs,
    )


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
#   python -m DeepDataMiningLearning.ngdet.detectors.base
# Expected: prints the registered detector backends (after importing adapters).
# ===========================================================================
if __name__ == "__main__":
    from . import hf_detr, yolo, locate_anything  # noqa: F401
    print("Registered detector backends:", list(DETECTOR_REGISTRY))
