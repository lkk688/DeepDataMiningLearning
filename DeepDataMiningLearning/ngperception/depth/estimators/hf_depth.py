"""
ngperception.depth.estimators.hf_depth
======================================

One adapter for the whole HuggingFace depth-estimation zoo via
`AutoModelForDepthEstimation` — covers the full basic->SOTA ladder with one class:

    Intel/dpt-large, Intel/dpt-hybrid-midas              MiDaS/DPT      (relative)
    depth-anything/Depth-Anything-V2-Small-hf  (+Base/Large)           (relative)
    depth-anything/Depth-Anything-V2-Metric-KITTI-Small-hf            (METRIC, meters)
    Intel/zoedepth-kitti, Intel/zoedepth-nyu-kitti                    (METRIC, meters)

Metric vs relative is auto-detected from the model name (or forced with `metric=`).
Relative models emit *inverse depth* (disparity, larger = closer); we invert them so the
returned `DepthResult.depth` is always **larger = farther** (see base.py).
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from .base import BaseDepthEstimator, DepthResult, register

# substrings that mark a model as predicting absolute metric depth (meters)
_METRIC_HINTS = ("zoedepth", "metric", "-kitti", "-nyu")
_EPS = 1e-6


@register("hf_depth")
class HFDepthEstimator(BaseDepthEstimator):
    family = "hf_depth"

    def __init__(self, device="cuda", model_name: Optional[str] = None,
                 metric: Optional[bool] = None, **kwargs):
        super().__init__(device=device)
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.model_name = model_name or "depth-anything/Depth-Anything-V2-Small-hf"
        cache_dir = kwargs.get("cache_dir", None)

        lower = self.model_name.lower()
        # explicit `metric=` wins; otherwise sniff the name
        self.is_metric = metric if metric is not None else any(h in lower for h in _METRIC_HINTS)

        self.processor = AutoImageProcessor.from_pretrained(self.model_name, cache_dir=cache_dir)
        dtype = torch.float16 if (device.startswith("cuda") and kwargs.get("fp16", True)) else torch.float32
        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.model_name, torch_dtype=dtype, cache_dir=cache_dir).to(device).eval()
        self.dtype = dtype

    def predict(self, image) -> DepthResult:
        w, h = image.size
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        inputs = {k: (v.to(self.dtype) if v.is_floating_point() else v) for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        # resize prediction back to the original image size
        try:    # newer transformers: handles resize + (for some models) metric scaling
            try:
                post = self.processor.post_process_depth_estimation(outputs, target_sizes=[(h, w)])
            except ValueError:   # ZoeDepth pads inputs and needs the pre-pad source size
                post = self.processor.post_process_depth_estimation(
                    outputs, target_sizes=[(h, w)], source_sizes=[(h, w)])
            pred = post[0]["predicted_depth"]
        except (AttributeError, KeyError, TypeError):
            pred = self.torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1).float(), size=(h, w),
                mode="bicubic", align_corners=False).squeeze()
        pred = pred.float().cpu().numpy()

        if self.is_metric:
            depth = np.clip(pred, _EPS, None)               # already meters, larger = farther
        else:
            # relative models output disparity (inverse depth, larger = closer);
            # invert to a depth-like map (larger = farther), scale still arbitrary.
            disp = np.clip(pred, _EPS, None)
            depth = 1.0 / disp
        return DepthResult(depth=depth.astype(np.float32), metric=self.is_metric)


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.depth.estimators.hf_depth
# Needs transformers + a GPU/network (downloads weights). Runs Depth-Anything-V2-Small
# on a sample image and prints the depth-map stats.
# ===========================================================================
if __name__ == "__main__":
    import requests
    from PIL import Image

    img = Image.open(requests.get(
        "https://ultralytics.com/images/bus.jpg", stream=True).raw).convert("RGB")
    for spec in ["depth-anything/Depth-Anything-V2-Small-hf"]:
        est = HFDepthEstimator(model_name=spec)
        r = est.predict(img)
        print(f"{spec}: metric={r.metric} shape={r.shape} "
              f"depth[min/med/max]={r.depth.min():.2f}/{np.median(r.depth):.2f}/{r.depth.max():.2f}")
