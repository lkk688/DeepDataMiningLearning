"""
ngdet.detectors.qwen_vl
======================

Adapter for **Qwen2.5-VL** (and Qwen2-VL) used as an open-vocabulary detector. Unlike
the LocateAnything VLM (custom Parallel-Box-Decoding internals — not trainable here),
Qwen2.5-VL is a *standard* HF model: it grounds objects by **generating JSON** and its
`forward` computes a normal LM loss, so it can be LoRA fine-tuned (see train.py
`--trainer qwen_lora`).

Detection prompt -> the model returns
    [{"bbox_2d": [x1,y1,x2,y2], "label": "car"}, ...]
in the coordinate space of the **smart-resized** image (grid_thw * patch_size), which
we rescale back to the original image. Qwen emits no confidence, so we score by output
order (the model lists its most confident detections first).
"""

from __future__ import annotations
import json
import re
from typing import List, Optional

import numpy as np

from .base import BaseDetector, Detection, register

_PATCH = 14  # Qwen2.x-VL vision patch size; grid_thw is in patch units


@register("qwen_vl")
class QwenVLDetector(BaseDetector):
    family = "qwen_vl"
    is_open_vocab = True

    def __init__(self, taxonomy, device="cuda", score_thr=0.3,
                 model_name: Optional[str] = None, **kwargs):
        super().__init__(taxonomy, device=device, score_thr=score_thr)
        import torch
        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info

        self.torch = torch
        self._pvi = process_vision_info
        self.model_name = model_name or "Qwen/Qwen2.5-VL-3B-Instruct"
        cache_dir = kwargs.get("cache_dir", None)
        self.max_new_tokens = int(kwargs.get("max_new_tokens", 1024))

        # A LoRA checkpoint dir (from train_qwen_lora) has adapter_config.json; load
        # the base model named inside it, then apply the LoRA adapter.
        import os, json as _json
        lora_dir = self.model_name if os.path.isfile(
            os.path.join(self.model_name, "adapter_config.json")) else None
        base_name = (_json.load(open(os.path.join(lora_dir, "adapter_config.json")))
                     ["base_model_name_or_path"]) if lora_dir else self.model_name

        if "2.5" in base_name or "2_5" in base_name:
            from transformers import Qwen2_5_VLForConditionalGeneration as VL
        else:
            from transformers import Qwen2VLForConditionalGeneration as VL
        # cap vision tokens for speed/memory (must match training, see train_qwen_lora)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name if lora_dir else base_name, cache_dir=cache_dir,
            max_pixels=int(kwargs.get("max_pixels", 1024 * 1024)))
        self.model = VL.from_pretrained(
            base_name, torch_dtype=torch.bfloat16,
            device_map=device, cache_dir=cache_dir).eval()
        if lora_dir:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_dir).eval()
            print(f"[qwen_vl] loaded LoRA adapter from {lora_dir}")

        # the prompt lists the curated class terms so the JSON labels fold cleanly
        self._terms = [t for t, _ in self.taxonomy.open_vocab_terms()]
        self.prompt = (
            "Detect all " + ", ".join(self._terms) + " in the image. Output ONLY a JSON "
            'list, each item {"bbox_2d":[x1,y1,x2,y2],"label":"<class>"}.')

    def _generate(self, image):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": self.prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = self._pvi(messages)
        inp = self.processor(text=[text], images=imgs, videos=vids, padding=True,
                             return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        gen = out[:, inp.input_ids.shape[1]:]
        ans = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        # resized image dims the model saw (for coordinate rescaling)
        grid = inp.get("image_grid_thw")
        rh = rw = None
        if grid is not None:
            g = grid[0].tolist()
            rh, rw = g[1] * _PATCH, g[2] * _PATCH
        return ans, rh, rw

    def predict(self, image, prompt: Optional[List[str]] = None) -> Detection:
        try:
            ans, rh, rw = self._generate(image)
        except Exception as e:  # noqa: BLE001
            print(f"[qwen_vl] generate failed: {e}")
            return Detection()
        w, h = image.size
        sx = w / rw if rw else 1.0          # resized-space -> original-space
        sy = h / rh if rh else 1.0

        # pull the JSON list out of the (possibly fenced) text
        items = []
        m = re.search(r"\[.*\]", ans, re.DOTALL)
        if m:
            try:
                items = json.loads(m.group(0))
            except json.JSONDecodeError:
                items = []
        kb, ks, kl, kn = [], [], [], []
        for i, it in enumerate(items):
            box = it.get("bbox_2d") or it.get("bbox")
            uid = self.taxonomy.name_to_id(str(it.get("label", "")))
            if not box or len(box) != 4 or uid is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy
            if x2 <= x1 or y2 <= y1:
                continue
            kb.append([x1, y1, x2, y2])
            # order-based confidence (Qwen lists confident detections first)
            ks.append(1.0 - 0.7 * (i / max(1, len(items) - 1)))
            kl.append(uid)
            kn.append(self.taxonomy.classes[uid])
        if not kb:
            return Detection()
        return Detection(np.asarray(kb, np.float32).reshape(-1, 4),
                         np.asarray(ks, np.float32), np.asarray(kl, np.int64), kn)


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# Needs `transformers`, `qwen-vl-utils`, a GPU + network (downloads ~7GB).
#
#   python -m DeepDataMiningLearning.ngdet.detectors.qwen_vl
#
# Expected: runs Qwen2.5-VL-3B on a COCO sample, prints unified detections.
# ===========================================================================
if __name__ == "__main__":
    import requests
    from PIL import Image
    from ..taxonomy import Taxonomy

    tax = Taxonomy.from_preset("driving3")
    det = QwenVLDetector(tax, device="cuda")
    url = "https://ultralytics.com/images/bus.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    out = det.predict(img)
    print(f"{len(out)} unified detections:")
    for b, s, n in zip(out.boxes, out.scores, out.names):
        print(f"  {n:8s} {s:.2f} {[round(float(x), 1) for x in b]}")
