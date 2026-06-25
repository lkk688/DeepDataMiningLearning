"""
ngdet.detectors.grounding_dino
=============================

Adapter for HuggingFace **open-vocabulary / zero-shot** detectors that share the
`AutoModelForZeroShotObjectDetection` + `post_process_grounded_object_detection`
API. This covers the Grounding DINO family and the OWL family:

    gdino  ->  IDEA-Research/grounding-dino-tiny
               IDEA-Research/grounding-dino-base
               (and MM-Grounding-DINO variants exported to this API)
    owlv2  ->  google/owlv2-base-patch16-ensemble
               google/owlv2-large-patch14-ensemble
               google/owlvit-base-patch32

These are TEXT-PROMPTED: we prompt them with the unified taxonomy class names, so
they predict our classes directly (no COCO->unified synonym table needed). Unlike
the LocateAnything VLM, these return real confidence scores, so mAP / PR curves are
meaningful.

Why one adapter for both: Grounding DINO and OWLv2 use the same HF auto-classes and
post-processing entry point; only the text-prompt format differs (Grounding DINO
wants a single ". "-joined caption; OWL wants a list of queries). We branch on that
and map returned phrase strings back to unified ids by NAME.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np

from .base import BaseDetector, Detection, register


class HFZeroShotDetector(BaseDetector):
    """Shared implementation for HF zero-shot OD models (Grounding DINO / OWL)."""

    is_open_vocab = True
    #: "caption" (Grounding DINO: one ". "-joined string) or
    #: "queries" (OWL: a list of class strings)
    prompt_style = "caption"

    def __init__(self, taxonomy, device="cuda", score_thr=0.3,
                 model_name: Optional[str] = None, default_model: str = None,
                 **kwargs):
        super().__init__(taxonomy, device=device, score_thr=score_thr)
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.torch = torch
        self.model_name = model_name or default_model
        cache_dir = kwargs.get("cache_dir", None)
        # text_threshold gates phrase<->box association for Grounding DINO.
        self.text_threshold = float(kwargs.get("text_threshold", 0.25))

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, cache_dir=cache_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_name, cache_dir=cache_dir)
        self.model.to(device).eval()

        # Build the text prompt from CURATED concrete terms (e.g. car/truck/bus
        # for "vehicle"), not the abstract unified names -- open-vocab recall is
        # far better with concrete words. We keep each term's unified id so the
        # returned boxes can be folded back.
        terms = self.taxonomy.open_vocab_terms()        # [(term, uid), ...]
        self._term_list = [t.lower() for t, _ in terms]
        self._term_uids = [uid for _, uid in terms]
        if self.prompt_style == "caption":
            # Grounding DINO convention: lowercase, each term ends with " . ".
            self.text_prompt = ". ".join(self._term_list) + " ."
        else:
            self.text_prompt = [self._term_list]  # OWL expects a batch of query lists

    def _label_to_unified(self, text_label) -> Optional[int]:
        """Map a returned phrase/label string to a unified class id by name."""
        if text_label is None:
            return None
        s = str(text_label).strip().lower()
        uid = self.taxonomy.name_to_id(s)
        if uid is not None:
            return uid
        # Grounding DINO may return multi-word or partial phrases; fall back to
        # substring matching against the unified class names.
        for u, name in enumerate(self.taxonomy.classes):
            if name.lower() in s or s in name.lower():
                return u
        return None

    def predict(self, image, prompt: Optional[List[str]] = None) -> Detection:
        w, h = image.size
        if self.prompt_style == "caption":
            inputs = self.processor(images=image, text=self.text_prompt,
                                    return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(images=image, text=self.text_prompt,
                                    return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [(h, w)]
        # The grounded post-processor signature has varied across transformers
        # versions; try the modern keyword form, then older positional forms.
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs, threshold=0.0, text_threshold=self.text_threshold,
                target_sizes=target_sizes)[0]
        except TypeError:
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=0.0,
                    text_threshold=self.text_threshold,
                    target_sizes=target_sizes)[0]
            except (TypeError, AttributeError):
                # OWL fallback: plain object-detection post-processing.
                results = self.processor.post_process_object_detection(
                    outputs, threshold=0.0, target_sizes=target_sizes)[0]

        boxes = results["boxes"].detach().cpu().numpy().reshape(-1, 4)
        scores = results["scores"].detach().cpu().numpy().reshape(-1)
        # Newer transformers returns "text_labels"; older returns "labels".
        text_labels = results.get("text_labels", results.get("labels"))
        if hasattr(text_labels, "tolist"):
            text_labels = text_labels.tolist()

        keep_b, keep_s, keep_l, keep_n = [], [], [], []
        for box, sc, lab in zip(boxes, scores, text_labels):
            if sc < self.score_thr:
                continue
            # OWL returns an integer index INTO THE QUERY-TERM LIST (len = #terms,
            # not #classes), so map it through term->unified id. Grounding DINO
            # returns the matched phrase string, mapped by name.
            if isinstance(lab, (int, np.integer)):
                li = int(lab)
                uid = self._term_uids[li] if 0 <= li < len(self._term_uids) else None
            else:
                uid = self._label_to_unified(lab)
            if uid is None:
                continue
            keep_b.append(box)
            keep_s.append(float(sc))
            keep_l.append(int(uid))
            keep_n.append(self.taxonomy.classes[uid])
        if not keep_b:
            return Detection()
        return Detection(
            boxes=np.asarray(keep_b, np.float32).reshape(-1, 4),
            scores=np.asarray(keep_s, np.float32),
            labels=np.asarray(keep_l, np.int64),
            names=keep_n,
        )


@register("gdino")
class GroundingDinoDetector(HFZeroShotDetector):
    family = "gdino"
    prompt_style = "caption"

    def __init__(self, taxonomy, device="cuda", score_thr=0.3, model_name=None, **kw):
        super().__init__(taxonomy, device, score_thr, model_name=model_name,
                         default_model="IDEA-Research/grounding-dino-tiny", **kw)


@register("owlv2")
class Owlv2Detector(HFZeroShotDetector):
    family = "owlv2"
    prompt_style = "queries"

    def __init__(self, taxonomy, device="cuda", score_thr=0.3, model_name=None, **kw):
        super().__init__(taxonomy, device, score_thr, model_name=model_name,
                         default_model="google/owlv2-base-patch16-ensemble", **kw)


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# Needs `transformers` + network (downloads the open-vocab checkpoint).
#
#   python -m DeepDataMiningLearning.ngdet.detectors.grounding_dino
#
# Expected: prompts Grounding DINO with the driving3 class names on a COCO sample
# image and prints unified detections (with real confidence scores).
# ===========================================================================
if __name__ == "__main__":
    import requests
    from PIL import Image
    from ..taxonomy import Taxonomy

    tax = Taxonomy.from_preset("driving3")
    det = GroundingDinoDetector(tax, device="cpu", score_thr=0.3,
                                model_name="IDEA-Research/grounding-dino-tiny")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    out = det.predict(img)
    print(f"{len(out)} unified detections from Grounding DINO:")
    for b, s, n in zip(out.boxes, out.scores, out.names):
        print(f"  {n:8s} {s:.3f} box={[round(float(x), 1) for x in b]}")
