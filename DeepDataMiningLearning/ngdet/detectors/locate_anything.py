"""
ngdet.detectors.locate_anything
===============================

Adapter for NVIDIA **LocateAnything** (`nvidia/LocateAnything-3B`), a VLM-based
open-vocabulary grounding/detection model (Moon-ViT + Qwen2.5 + Parallel Box
Decoding; see ngdet/EVAL_FRAMEWORK_PLAN.md).

This adapter follows the official inference recipe from the model card:
  * custom `AutoModel` + `AutoTokenizer` + `AutoProcessor` (trust_remote_code),
  * a chat-message prompt with the image and a category description,
  * `model.generate(..., generation_mode="hybrid")` (the fast PBD/MTP path with
    autoregressive fallback),
  * output boxes in the text format `<box><x1><y1><x2><y2></box>` with integer
    coords in [0, 1000], rescaled to absolute pixels.

LABELING STRATEGY: the model's multi-category output does not cleanly tag each box
with its class, so for reliable labels we prompt ONCE PER UNIFIED CLASS using that
class's curated terms (joined by the model's "</c>" separator) and assign every
returned box that class's unified id. That is `num_classes` generate() calls per
image (3 for driving3) -- slower than a single pass, but gives correct labels.

STATUS: requires a GPU + network. The model has no calibrated per-box confidence,
so we emit score=1.0 (mAP then ranks its boxes equally -- a known limitation noted
in the report).
"""

from __future__ import annotations
import re
from typing import List, Optional

import numpy as np

from .base import BaseDetector, Detection, register

# Official output box pattern + coordinate range.
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
# The model tags each group of boxes with the class it matched:
#   <ref>person</ref><box>..</box><box>..</box><ref>car</ref><box>..</box>...
# so we can label boxes from ONE multi-class prompt (3x faster than per-class).
_REF_RE = re.compile(r"<ref>(.*?)</ref>")
_COORD_RANGE = 1000.0


@register("locate_anything")
class LocateAnythingDetector(BaseDetector):
    family = "locate_anything"
    is_open_vocab = True

    def __init__(self, taxonomy, device="cuda", score_thr=0.3,
                 model_name: Optional[str] = None, **kwargs):
        super().__init__(taxonomy, device=device, score_thr=score_thr)
        import torch
        from transformers import AutoModel, AutoTokenizer, AutoProcessor

        self.torch = torch
        self.model_name = model_name or "nvidia/LocateAnything-3B"
        cache_dir = kwargs.get("cache_dir", None)
        # Detection output is short; 768 is plenty and far faster than 2048.
        self.max_new_tokens = int(kwargs.get("max_new_tokens", 768))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, cache_dir=cache_dir)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            cache_dir=cache_dir).to(device).eval()

        # Curated concrete prompt terms (car/truck/bus/... ) for the single
        # multi-class prompt; returned boxes are labeled from the <ref> tags.
        self._term_list = [t for t, _ in self.taxonomy.open_vocab_terms()]

    def _generate(self, image, cats_str: str) -> str:
        """Run one generate() pass for a "</c>"-joined category string."""
        prompt = (f"Locate all the instances that matches the following "
                  f"description: {cats_str}.")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos,
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            response = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.torch.bfloat16),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                generation_mode="hybrid",
                use_cache=True,   # required by the LocateAnything decoder
            )
        return response[0] if isinstance(response, tuple) else response

    def _parse_ref_boxes(self, text, w: int, h: int):
        """Parse a multi-class answer into (box_xyxy, unified_id) pairs.

        The answer interleaves class tags and boxes:
            <ref>car</ref><box>None</box><ref>person</ref><box>..</box><box>..</box>
        We walk each <ref>NAME</ref> segment, map NAME -> unified id, and assign the
        boxes that follow it (until the next <ref>). The hybrid decoder sometimes
        loops the same box, so we de-duplicate per segment on the raw [0,1000] quad.
        """
        out = []
        refs = list(_REF_RE.finditer(str(text)))
        for i, m in enumerate(refs):
            uid = self.taxonomy.name_to_id(m.group(1))
            if uid is None:
                continue
            seg = text[m.end(): refs[i + 1].start() if i + 1 < len(refs) else len(text)]
            seen = set()
            for bm in _BOX_RE.finditer(seg):
                quad = tuple(int(g) for g in bm.groups())
                if quad in seen:
                    continue
                seen.add(quad)
                x1, y1, x2, y2 = [v / _COORD_RANGE for v in quad]
                bx = [x1 * w, y1 * h, x2 * w, y2 * h]
                if bx[2] > bx[0] and bx[3] > bx[1]:
                    out.append((bx, uid))
        return out

    def predict(self, image, prompt: Optional[List[str]] = None) -> Detection:
        w, h = image.size
        # ONE generate() with all curated terms; labels come from the <ref> tags.
        cats_str = "</c>".join(self._term_list)
        try:
            answer = self._generate(image, cats_str)
        except Exception as e:  # noqa: BLE001
            print(f"[locate_anything] generate failed: {e}")
            return Detection()

        pairs = self._parse_ref_boxes(answer, w, h)
        if not pairs:
            return Detection()
        boxes = np.asarray([p[0] for p in pairs], np.float32).reshape(-1, 4)
        labels = np.asarray([p[1] for p in pairs], np.int64)
        return Detection(
            boxes=boxes,
            scores=np.ones(len(boxes), np.float32),  # no calibrated confidence
            labels=labels,
            names=[self.taxonomy.classes[int(l)] for l in labels],
        )


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# Heavy (3B VLM): needs a GPU, `transformers`, and network to download
# `nvidia/LocateAnything-3B`.
#
#   python -m DeepDataMiningLearning.ngdet.detectors.locate_anything
#
# Expected: prompts the model per driving3 class on a sample image and prints
# parsed unified boxes. If the checkpoint API changes, only `_generate` /
# `_parse_boxes` need editing -- the rest of the framework is unaffected.
# ===========================================================================
if __name__ == "__main__":
    import requests
    from PIL import Image
    from ..taxonomy import Taxonomy

    tax = Taxonomy.from_preset("driving3")
    try:
        det = LocateAnythingDetector(tax, device="cuda", score_thr=0.3)
        url = "https://ultralytics.com/images/bus.jpg"
        img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        out = det.predict(img)
        print(f"{len(out)} unified detections from LocateAnything:")
        for b, s, n in zip(out.boxes, out.scores, out.names):
            print(f"  {n:8s} box={[round(float(x), 1) for x in b]}")
    except Exception as e:  # noqa: BLE001
        print(f"[locate_anything] adapter failed: {e}")
        print("Check the model card inference recipe and edit _generate/_parse_boxes.")
