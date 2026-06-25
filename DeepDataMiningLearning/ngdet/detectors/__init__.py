"""
ngdet.detectors -- pluggable detector backends.

Each adapter registers itself into `DETECTOR_REGISTRY` via `@register("key")`.
Importing an adapter module triggers its registration (the registry is populated
lazily by `ngdet.detectors.base.build_detector`, which imports the adapters on
demand so a missing optional dependency only affects the backend that needs it).
"""

from .base import (
    BaseDetector, Detection, DETECTOR_REGISTRY, register, build_detector,
)

__all__ = [
    "BaseDetector", "Detection", "DETECTOR_REGISTRY", "register", "build_detector",
]
