"""
ngdet.detectors.hf_detr
=======================

Adapter for HuggingFace `AutoModelForObjectDetection` checkpoints. This covers the
whole DETR family and its modern descendants that share the same processor /
post-processing API, e.g.:

    facebook/detr-resnet-50            (classic DETR, COCO-91 label index)
    facebook/detr-resnet-101
    SenseTime/deformable-detr          (Deformable DETR)
    PekingU/rtdetr_r50vd               (RT-DETR, real-time DETR)
    jozhang97/deta-resnet-50           (DETA)

These are CLOSED-SET COCO detectors: they emit COCO class ids, which we fold into
the active unified taxonomy by NAME via `model.config.id2label`. So the same
adapter works regardless of whether a checkpoint uses the 80- or 91-entry COCO index.

This logic is consolidated from the old `vision/hfvision_inference.py::object_detection`
and made model-agnostic + taxonomy-aware.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np

from .base import BaseDetector, Detection, register


@register("hf_detr")
class HFDetrDetector(BaseDetector):
    family = "hf_detr"
    is_open_vocab = False

    def __init__(self, taxonomy, device="cuda", score_thr=0.3,
                 model_name: Optional[str] = None, accel: str = "fp32", **kwargs):
        super().__init__(taxonomy, device=device, score_thr=score_thr, accel=accel)
        # Heavy imports kept local so `import ngdet` works without transformers.
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        if accel not in ("fp32", "fp16", "compile", "onnx", "tensorrt"):
            raise ValueError(f"hf_detr backend does not support accel='{accel}'")
        self.torch = torch
        self.model_name = model_name or "facebook/detr-resnet-50"
        cache_dir = kwargs.get("cache_dir", None)

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name, cache_dir=cache_dir)
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_name, cache_dir=cache_dir)
        self.model.to(device).eval()

        # Acceleration:
        #   fp16     -> half precision
        #   compile  -> torch.compile
        #   onnx     -> ONNX export + onnxruntime-gpu CUDA Execution Provider
        #   tensorrt -> ONNX export + onnxruntime-gpu TensorRT Execution Provider
        #               (falls back to CUDA EP for any op TRT can't build)
        self._half = (accel == "fp16")
        self._onnx = accel in ("onnx", "tensorrt")
        if self._half:
            self.model.half()
        elif accel == "compile":
            self.model = torch.compile(self.model, mode="reduce-overhead")
        elif self._onnx:
            self._setup_onnx(accel)

        # Build the native-id -> unified-id LUT once from the checkpoint's own
        # id2label map. This is what makes the adapter robust to 80-vs-91 indexing.
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.id_lut = self.taxonomy.build_id_lut(id2label)

    # ---- ONNX / TensorRT path (via onnxruntime-gpu) ------------------------
    def _setup_onnx(self, accel: str):
        """Export the DETR forward to ONNX (fixed input size) and open an
        onnxruntime-gpu session on the CUDA or TensorRT execution provider."""
        import os
        import numpy as np
        import onnxruntime as ort
        from torch import nn

        self._np = np
        # Fixed input size -> a static graph that TensorRT can build an engine for.
        # Boxes are normalized cxcywh, so post-processing with the ORIGINAL image
        # size rescales them correctly even though we resize (aspect-distort) here.
        self._in_h, self._in_w = 800, 1333
        self._mean = np.array(self.processor.image_mean, np.float32).reshape(3, 1, 1)
        self._std = np.array(self.processor.image_std, np.float32).reshape(3, 1, 1)

        safe = self.model_name.replace("/", "_")
        onnx_path = f"/tmp/ngdet_{safe}_{self._in_h}x{self._in_w}.onnx"
        if not os.path.exists(onnx_path):
            class _Wrap(nn.Module):                      # return plain tensors
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, pv):
                    o = self.m(pixel_values=pv)
                    return o.logits, o.pred_boxes
            w = _Wrap(self.model).to(self.device).eval()
            dummy = self.torch.randn(1, 3, self._in_h, self._in_w, device=self.device)
            self.torch.onnx.export(
                w, (dummy,), onnx_path, input_names=["pixel_values"],
                output_names=["logits", "pred_boxes"], opset_version=17,
                do_constant_folding=True)

        providers = (["TensorrtExecutionProvider", "CUDAExecutionProvider",
                      "CPUExecutionProvider"] if accel == "tensorrt"
                     else ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._sess = ort.InferenceSession(onnx_path, providers=providers)
        # the provider actually serving (TRT may fall back to CUDA on unsupported ops)
        self.accel = self._sess.get_providers()[0]

    def _predict_onnx(self, image) -> Detection:
        from types import SimpleNamespace
        np = self._np
        w0, h0 = image.size
        arr = np.asarray(image.resize((self._in_w, self._in_h)), np.float32)
        arr = arr.transpose(2, 0, 1) / 255.0
        arr = ((arr - self._mean) / self._std)[None].astype(np.float32)
        logits, boxes = self._sess.run(None, {"pixel_values": arr})
        outputs = SimpleNamespace(logits=self.torch.from_numpy(logits),
                                  pred_boxes=self.torch.from_numpy(boxes))
        res = self.processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=self.torch.tensor([[h0, w0]]))[0]
        return self._fold_to_unified(
            res["boxes"].numpy().reshape(-1, 4),
            res["scores"].numpy().reshape(-1),
            res["labels"].numpy().reshape(-1))

    def predict(self, image, prompt: Optional[List[str]] = None) -> Detection:
        if self._onnx:
            return self._predict_onnx(image)
        # `image` is a PIL.Image (RGB). DETR's processor handles resize/normalize.
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        if self._half:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        # post_process_object_detection maps the network outputs back to ABSOLUTE
        # pixel xyxy boxes in the ORIGINAL image size given via target_sizes.
        # target_sizes expects (height, width); PIL image.size is (width, height).
        w, h = image.size
        target_sizes = self.torch.tensor([[h, w]], device=self.device)
        # threshold=0.0 here: we apply our own unified score_thr in _fold_to_unified
        # so the threshold is identical across all backends.
        results = self.processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target_sizes)[0]

        boxes = results["boxes"].detach().cpu().numpy().reshape(-1, 4)
        scores = results["scores"].detach().cpu().numpy().reshape(-1)
        native_ids = results["labels"].detach().cpu().numpy().reshape(-1)
        return self._fold_to_unified(boxes, scores, native_ids)


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# Needs network access (downloads the checkpoint on first run) + `transformers`.
#
#   python -m DeepDataMiningLearning.ngdet.detectors.hf_detr
#
# Expected: downloads facebook/detr-resnet-50, runs it on a COCO sample image,
# and prints the unified-taxonomy detections (vehicle/person/cyclist).
# ===========================================================================
if __name__ == "__main__":
    import requests
    from PIL import Image
    from ..taxonomy import Taxonomy

    tax = Taxonomy.from_preset("driving3")
    det = HFDetrDetector(tax, device="cpu", score_thr=0.5,
                         model_name="facebook/detr-resnet-50")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    out = det.predict(img)
    print(f"{len(out)} unified detections on the sample image:")
    for b, s, n in zip(out.boxes, out.scores, out.names):
        print(f"  {n:8s} {s:.3f} box={[round(float(x), 1) for x in b]}")
