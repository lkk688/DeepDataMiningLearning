"""
ngperception.depth.estimators.base
==================================

The pluggable depth-estimator contract — the depth analogue of `ngdet.detectors.base`.
Every backend (HuggingFace DPT / Depth-Anything / ZoeDepth, MiDaS, ...) is wrapped in a
small adapter that subclasses `BaseDepthEstimator` and registers itself with
`@register("name")`.

A depth estimator's ONLY job is:
    image (PIL RGB)  ->  DepthResult  (a per-pixel depth map at the ORIGINAL resolution)

Two flavors of model exist, and the difference matters for evaluation:

* **metric** models (ZoeDepth, Depth-Anything-V2-Metric) predict absolute depth in
  **meters**. They can be compared to LiDAR GT directly.
* **relative** models (MiDaS / DPT, Depth-Anything-V2 relative) predict depth only up to
  an unknown scale (and often as *inverse depth* / disparity). They must be aligned to GT
  (median or least-squares scaling) before metrics make sense (see `evaluator.py`).

The adapter normalizes every model to return a **depth-like map where larger = farther**
(meters for metric models; arbitrary-but-monotonic units for relative ones), so the rest
of the pipeline (evaluator, fusion, visualization) is model-agnostic.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type

import numpy as np


@dataclass
class DepthResult:
    """Per-pixel depth for ONE image, at the original (un-resized) image resolution.

    Attributes
    ----------
    depth : np.ndarray
        HxW float32. Larger = farther. Meters iff `metric` is True, else arbitrary scale.
    metric : bool
        True if `depth` is absolute metric depth (meters); False if scale-ambiguous.
    valid : np.ndarray | None
        Optional HxW bool mask of pixels with a usable prediction (None = all valid).
    """
    depth: np.ndarray
    metric: bool = False
    valid: Optional[np.ndarray] = None

    @property
    def shape(self):
        return self.depth.shape


class BaseDepthEstimator:
    """Abstract base for all depth-estimator adapters.

    Subclasses implement `predict`. Heavy imports (torch, transformers) belong INSIDE the
    subclass `__init__`, never at module top level, so importing the package stays cheap
    and a missing optional dependency only breaks the backend that needs it.
    """

    #: human-readable backend family name (set by subclass)
    family: str = "base"
    #: True if the model outputs absolute metric depth (meters)
    is_metric: bool = False

    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device

    def predict(self, image) -> DepthResult:
        """Run the model on a single PIL RGB image and return a `DepthResult` whose map
        is the same H×W as the input image."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry: short key -> adapter class, so a CLI can spell
# "--models hf_depth:depth-anything/Depth-Anything-V2-Small-hf" and we resolve the class
# by the part before the colon (identical convention to ngdet.detectors).
# ---------------------------------------------------------------------------
DEPTH_REGISTRY: Dict[str, Type[BaseDepthEstimator]] = {}


def register(name: str) -> Callable[[Type[BaseDepthEstimator]], Type[BaseDepthEstimator]]:
    def deco(cls: Type[BaseDepthEstimator]) -> Type[BaseDepthEstimator]:
        DEPTH_REGISTRY[name] = cls
        return cls
    return deco


def build_estimator(spec: str, device: str = "cuda", **kwargs) -> BaseDepthEstimator:
    """Instantiate a depth estimator from a "key:model_name" spec string.

    Examples
    --------
    build_estimator("hf_depth:Intel/dpt-large")
    build_estimator("hf_depth:depth-anything/Depth-Anything-V2-Small-hf")
    build_estimator("hf_depth:Intel/zoedepth-kitti")           # metric
    """
    from . import hf_depth  # noqa: F401  (side effect: register backends)

    if ":" in spec:
        key, model_name = spec.split(":", 1)
    else:
        key, model_name = spec, None
    if key not in DEPTH_REGISTRY:
        raise KeyError(
            f"Unknown depth backend '{key}'. Registered: {list(DEPTH_REGISTRY)}")
    return DEPTH_REGISTRY[key](device=device, model_name=model_name, **kwargs)


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.depth.estimators.base
# Expected: prints the registered depth backends (after importing adapters).
# ===========================================================================
if __name__ == "__main__":
    from . import hf_depth  # noqa: F401
    print("Registered depth backends:", list(DEPTH_REGISTRY))
