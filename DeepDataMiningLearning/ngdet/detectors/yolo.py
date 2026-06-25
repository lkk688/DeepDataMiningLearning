"""
ngdet.detectors.yolo
====================

Adapter for Ultralytics YOLO checkpoints (YOLOv8 / YOLO11 / YOLO12, and the
open-vocabulary YOLO-World / YOLOE variants).

Two modes:
  * CLOSED-SET (default): a COCO-pretrained `yolo11x.pt` emits 80 COCO classes,
    which we fold into the unified taxonomy by NAME via `model.names`.
  * OPEN-VOCAB: if the checkpoint supports `set_classes` (YOLO-World / YOLOE),
    we set its vocabulary to the unified class prompts, so it directly predicts
    unified classes (id i == unified class i). Enable with model_name like
    "yolov8s-world.pt".

Consolidated from the YOLO paths in `detection/myinference.py` and
`detection/test_class_mapping_simple.py` (the native->unified mapping idea).
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np

from .base import BaseDetector, Detection, register


@register("yolo")
class YOLODetector(BaseDetector):
    family = "yolo"

    def __init__(self, taxonomy, device="cuda", score_thr=0.3,
                 model_name: Optional[str] = None, accel: str = "fp32", **kwargs):
        super().__init__(taxonomy, device=device, score_thr=score_thr, accel=accel)
        import os
        from ultralytics import YOLO  # local import: optional dependency

        if accel not in ("fp32", "fp16", "tensorrt", "onnx"):
            raise ValueError(f"yolo backend does not support accel='{accel}'")
        self.model_name = model_name or "yolo11x.pt"
        self._half = False
        dev_idx = device.split(":")[-1] if ":" in device else (0 if "cuda" in device else "cpu")

        # Acceleration via Ultralytics' native export (cached on disk by stem).
        if accel == "tensorrt":
            engine = self.model_name.replace(".pt", ".engine")
            if not os.path.exists(engine):
                YOLO(self.model_name).export(format="engine", half=True, device=dev_idx)
            self.model = YOLO(engine)
            self._half = True
        elif accel == "onnx":
            onnxp = self.model_name.replace(".pt", ".onnx")
            if not os.path.exists(onnxp):
                YOLO(self.model_name).export(format="onnx", device=dev_idx)
            self.model = YOLO(onnxp)
        else:
            self.model = YOLO(self.model_name)
            self._half = (accel == "fp16")

        # Detect open-vocab capability (YOLO-World / YOLOE expose set_classes).
        self.is_open_vocab = hasattr(self.model, "set_classes")
        if self.is_open_vocab:
            # Prompt the model with our unified class names; it will then output
            # ids 0..K-1 aligned to the taxonomy -> identity LUT.
            self.model.set_classes(self.taxonomy.prompts())
            self.id_lut = {i: i for i in range(self.taxonomy.num_classes)}
        else:
            # Closed-set: fold the model's native COCO names into unified ids.
            # `model.names` is a {id: name} dict.
            self.id_lut = self.taxonomy.build_id_lut(dict(self.model.names))

    def predict(self, image, prompt: Optional[List[str]] = None) -> Detection:
        # Ultralytics accepts a PIL image directly. conf=0.0 because we apply the
        # shared unified score_thr ourselves for cross-model comparability.
        results = self.model.predict(
            image, device=self.device, conf=0.0, verbose=False, half=self._half)[0]
        boxes_xyxy = results.boxes.xyxy.detach().cpu().numpy().reshape(-1, 4)
        scores = results.boxes.conf.detach().cpu().numpy().reshape(-1)
        native_ids = results.boxes.cls.detach().cpu().numpy().astype(int).reshape(-1)
        return self._fold_to_unified(boxes_xyxy, scores, native_ids)


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# Needs `ultralytics` (downloads weights on first run).
#
#   python -m DeepDataMiningLearning.ngdet.detectors.yolo
#
# Expected: downloads yolo11x.pt, runs on the Ultralytics sample bus image,
# prints unified detections. Then (best-effort) tries the open-vocab YOLO-World.
# ===========================================================================
if __name__ == "__main__":
    from PIL import Image
    import requests
    from ..taxonomy import Taxonomy

    tax = Taxonomy.from_preset("driving3")
    url = "https://ultralytics.com/images/bus.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    det = YOLODetector(tax, device="cpu", score_thr=0.3, model_name="yolo11n.pt")
    out = det.predict(img)
    print(f"[closed-set yolo11n] {len(out)} unified detections:")
    for b, s, n in zip(out.boxes, out.scores, out.names):
        print(f"  {n:8s} {s:.3f} box={[round(float(x), 1) for x in b]}")
