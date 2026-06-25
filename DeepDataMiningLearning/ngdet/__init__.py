"""
ngdet -- Next-Generation Detection evaluation framework
=======================================================

A pluggable, teaching-oriented framework to evaluate (zero-shot) any 2D detector
on any driving/COCO dataset under one unified, configurable label taxonomy, and
to render annotated evaluation videos.

Three decoupled layers (see EVAL_FRAMEWORK_PLAN.md):
    * detectors/  -- model adapters (HF DETR, Ultralytics YOLO, NVIDIA LocateAnything)
    * datasets    -- EvalDataset wrappers over the project's existing loaders
    * evaluator   -- canonical COCO mAP (pycocotools) on the unified taxonomy

Top-level entry point:
    python -m DeepDataMiningLearning.ngdet.run_eval --help

Submodule imports are intentionally NOT eager here, so `import ngdet` stays light
and does not require transformers/ultralytics/pycocotools to be installed unless
the corresponding backend is actually used.
"""

from .taxonomy import Taxonomy, TAXONOMY_PRESETS

__all__ = ["Taxonomy", "TAXONOMY_PRESETS"]
__version__ = "0.1.0"
