"""
ngdet.detectors.torchvision_det
==============================

Adapter for the classic torchvision detection models -- the canonical academic
baselines every student should know:

    fasterrcnn_resnet50_fpn_v2   (two-stage, anchor-based)
    fasterrcnn_resnet50_fpn      (the original)
    retinanet_resnet50_fpn_v2    (one-stage, focal loss)
    fcos_resnet50_fpn            (one-stage, anchor-free)
    ssd300_vgg16 / ssdlite...    (also work via the same API)

All are COCO-pretrained closed-set detectors. torchvision exposes their class
names via `weights.meta["categories"]` (a 91-entry list, index == label id, with
"__background__" at 0 and "N/A" gaps), which we fold into the unified taxonomy by
NAME -- the same name-based bridge used by the DETR/YOLO adapters.

Input convention: torchvision detection models expect a list of CHW float tensors
in [0,1] and do their OWN normalization/resize internally (GeneralizedRCNNTransform),
so we pass `to_tensor(pil)` directly and get boxes back in original pixel coords.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np

from .base import BaseDetector, Detection, register


@register("torchvision")
class TorchvisionDetector(BaseDetector):
    family = "torchvision"
    is_open_vocab = False

    def __init__(self, taxonomy, device="cuda", score_thr=0.3,
                 model_name: Optional[str] = None, accel: str = "fp32", **kwargs):
        super().__init__(taxonomy, device=device, score_thr=score_thr, accel=accel)
        import torch
        from torchvision.models import get_model, get_model_weights
        from torchvision.transforms.functional import to_tensor

        if accel not in ("fp32", "fp16", "compile"):
            raise ValueError(f"torchvision backend does not support accel='{accel}'")
        self.torch = torch
        self._to_tensor = to_tensor
        self.model_name = model_name or "fasterrcnn_resnet50_fpn_v2"

        checkpoint = kwargs.get("checkpoint", None)
        if checkpoint:
            # Evaluate a FINE-TUNED model: head sized to the taxonomy (+1 background),
            # weights loaded from our training checkpoint. Native labels are 1..K
            # (0 = background), so they map to unified ids l -> l-1.
            self.model = get_model(self.model_name, weights=None,
                                   num_classes=self.taxonomy.num_classes + 1)
            sd = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) else sd)
            self.id_lut = {l: l - 1 for l in range(1, self.taxonomy.num_classes + 1)}
            self.id_lut[0] = None
            self.model.to(device).eval()
            self._fold_setup_done = True
        else:
            # Default: COCO-pretrained weights; map COCO names -> unified by name.
            weights = get_model_weights(self.model_name).DEFAULT
            self.model = get_model(self.model_name, weights=weights)
            self.model.to(device).eval()
            self._fold_setup_done = False

        # Acceleration: fp16 (half) or torch.compile.
        self._half = (accel == "fp16")
        if self._half:
            self.model.half()
        elif accel == "compile":
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # For COCO-pretrained models, map COCO names -> unified by name. (For a
        # fine-tuned checkpoint the LUT was already set in __init__ above.)
        if not self._fold_setup_done:
            cats = weights.meta["categories"]      # index == COCO label id
            id2name = {i: name for i, name in enumerate(cats)}
            self.id_lut = self.taxonomy.build_id_lut(id2name)

    def predict(self, image, prompt: Optional[List[str]] = None) -> Detection:
        # PIL RGB -> CHW float[0,1]; the model normalizes/resizes internally.
        img_t = self._to_tensor(image).to(self.device)
        if self._half:
            img_t = img_t.half()
        with self.torch.no_grad():
            out = self.model([img_t])[0]   # dict: boxes (xyxy abs px), labels, scores
        boxes = out["boxes"].detach().cpu().numpy().reshape(-1, 4)
        scores = out["scores"].detach().cpu().numpy().reshape(-1)
        native_ids = out["labels"].detach().cpu().numpy().reshape(-1)
        return self._fold_to_unified(boxes, scores, native_ids)


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# Needs torchvision (weights download on first run).
#
#   python -m DeepDataMiningLearning.ngdet.detectors.torchvision_det
#
# Expected: runs Faster R-CNN v2 on a COCO sample image and prints unified
# detections (vehicle/person/cyclist).
# ===========================================================================
if __name__ == "__main__":
    import requests
    from PIL import Image
    from ..taxonomy import Taxonomy

    tax = Taxonomy.from_preset("driving3")
    det = TorchvisionDetector(tax, device="cpu", score_thr=0.5,
                              model_name="fasterrcnn_resnet50_fpn_v2")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    out = det.predict(img)
    print(f"{len(out)} unified detections:")
    for b, s, n in zip(out.boxes, out.scores, out.names):
        print(f"  {n:8s} {s:.3f} box={[round(float(x), 1) for x in b]}")
