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

Three adapter choices make this work well (the box *quality* is excellent — the
model predicts near-exact boxes — but naive use scores very low; see TUTORIAL §20):

  1. PROMPT ONCE PER UNIFIED CLASS (not one multi-class prompt). The hybrid decoder
     loops on the first class and eats the token budget, starving the rest, so a
     single prompt returns only "car". One generate() per class (3 for driving3)
     gives every class its own budget. The label is then known per prompt.
  2. ORDER-BASED CONFIDENCE. The model emits no calibrated score, but it outputs its
     most-confident detections FIRST, so we rank boxes by output order (loops / false
     positives land last -> low score). This is what lets COCO mAP rank correctly.
  3. DOWNSCALE LARGE INPUTS + `repetition_penalty`. A 1920x1280 frame makes a huge
     number of vision tokens (300+ s/image); capping the longest edge to `image_max`
     (boxes are normalized, so coords stay correct) brings it to ~7 s/image.

These took nuImages mAP from ~0.05 to ~0.30 and cut Waymo latency ~46x.

STATUS: requires a GPU + network. Still ~7 s/image (a 3B VLM, 3 generates) -- much
slower than a real-time detector, fine for evaluation.
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
        self.max_new_tokens = int(kwargs.get("max_new_tokens", 384))
        # >1 discourages the hybrid decoder's box-repeat loops (which otherwise eat
        # the token budget and starve later classes).
        self.repetition_penalty = float(kwargs.get("repetition_penalty", 1.05))
        # Downscale large inputs before generation. A 1920x1280 Waymo frame makes a
        # huge number of vision tokens (100s of seconds/image); capping the longest
        # edge cuts that dramatically. Boxes are normalized [0,1000], so we rescale
        # to the ORIGINAL size afterwards -> coordinates stay correct.
        self.image_max = int(kwargs.get("image_max", 1024))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, cache_dir=cache_dir)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            cache_dir=cache_dir).to(device).eval()

        # Curated prompt terms grouped per unified class. We prompt ONCE PER CLASS so
        # each class gets the full token budget (a single multi-class prompt loops on
        # the first class and starves the rest); the label is then known per prompt.
        self._term_list = [t for t, _ in self.taxonomy.open_vocab_terms()]
        self._terms_by_class = self.taxonomy.open_vocab_terms_by_class()

    def _generate(self, image, cats_str: str) -> str:
        """Run one generate() pass for a "</c>"-joined category string."""
        # downscale large images for speed (boxes are normalized -> still correct)
        w, h = image.size
        if max(w, h) > self.image_max:
            sc = self.image_max / max(w, h)
            image = image.resize((max(1, int(w * sc)), max(1, int(h * sc))))
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
                repetition_penalty=self.repetition_penalty,
            )
        return response[0] if isinstance(response, tuple) else response

    @staticmethod
    def _parse_boxes_ordered(text, w: int, h: int):
        """All boxes in OUTPUT ORDER, de-duplicated. For a single-class prompt every
        box belongs to that class, and the order is the model's confidence order."""
        seen, out = set(), []
        for bm in _BOX_RE.finditer(str(text)):
            quad = tuple(int(g) for g in bm.groups())
            if quad in seen:
                continue
            seen.add(quad)
            x1, y1, x2, y2 = [v / _COORD_RANGE for v in quad]
            bx = [x1 * w, y1 * h, x2 * w, y2 * h]
            if bx[2] > bx[0] and bx[3] > bx[1]:
                out.append(bx)
        return out

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
        boxes, scores, labels, names = [], [], [], []
        # One generate() per unified class -> full token budget per class.
        for uid, terms in self._terms_by_class.items():
            try:
                answer = self._generate(image, "</c>".join(terms))
            except Exception as e:  # noqa: BLE001 - keep other classes if one fails
                print(f"[locate_anything] generate failed (uid {uid}): {e}")
                continue
            bxs = self._parse_boxes_ordered(answer, w, h)
            n = len(bxs)
            for i, b in enumerate(bxs):
                boxes.append(b)
                # Order-based confidence: the model emits its most confident
                # detections first, so rank by output order (loops/false positives
                # come later -> lower score). This is what lets mAP rank correctly
                # despite the model emitting no calibrated scores.
                scores.append(1.0 - 0.7 * (i / max(1, n - 1)) if n > 1 else 1.0)
                labels.append(uid)
                names.append(self.taxonomy.classes[uid])
        if not boxes:
            return Detection()
        return Detection(
            boxes=np.asarray(boxes, np.float32).reshape(-1, 4),
            scores=np.asarray(scores, np.float32),
            labels=np.asarray(labels, np.int64),
            names=names,
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
