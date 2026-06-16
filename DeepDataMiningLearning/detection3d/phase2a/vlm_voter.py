"""
Tier-2 tie-breaker — NVIDIA Gemma-3n VLM voting on disagreement cases.

We only call the VLM when validators A and B disagree (or when only one
fires). On agreement we trust the local validators. This keeps the API
call rate low (~10-30% of detections).

Cache: result keyed by (image_hash, crop_xyxy, candidate_classes). A
single segment's repeated 2D dets across frames produce near-identical
crops, so we hash crop pixels (small thumbnail) and cache aggressively.

Returns:
  cls : str    one of {'Vehicle','Pedestrian','Cyclist','None'}
  conf: float  parsed from VLM's confidence answer (0-1)
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import requests

API_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
DEFAULT_MODEL = 'google/gemma-3n-e4b-it'

PROMPT_TEMPLATE = (
    "You are verifying an autonomous-driving 3D detection proposal. "
    "The image has a RED rectangle marking a candidate object. "
    "Look CAREFULLY at WHAT IS INSIDE THE RED BOX. Ignore objects OUTSIDE "
    "the red box.\n\n"
    "Question: what is inside the red box?\n"
    "  V = Vehicle (car, truck, bus, trailer, van)\n"
    "  P = Pedestrian (person walking or standing, NOT inside a vehicle)\n"
    "  C = Cyclist (person on bicycle or motorcycle)\n"
    "  N = None of the above (background, sign, pole, vegetation, building, "
    "road, empty road, distant blur)\n\n"
    "Be strict: if the red box is mostly empty road, vegetation, or a sign, "
    "answer N. Only answer V/P/C if a clear instance is inside the red box.\n\n"
    "Respond with a SINGLE character (V/P/C/N) on the first line, and a "
    "confidence score 0.0-1.0 on the second line. Nothing else."
)


class VLMVoter:
    """Cached, rate-limited Gemma VLM votes for tie-breaking."""

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = DEFAULT_MODEL,
                 cache_path: Optional[str] = None,
                 max_calls_per_minute: int = 240,
                 timeout_s: float = 20.0):
        self.api_key = api_key or os.environ.get('NVAPI_KEY', '')
        if not self.api_key:
            raise RuntimeError(
                'No NVAPI_KEY env var set; pass api_key=... to VLMVoter.')
        self.model = model
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, Dict] = {}
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open() as f:
                self.cache = json.load(f)
        self.max_calls_per_minute = int(max_calls_per_minute)
        self.timeout = float(timeout_s)
        self._call_times: List[float] = []
        self._n_calls = 0
        self._n_cache_hits = 0

    # ------------------------------------------------------------------ cache
    def _crop_hash(self, image_rgb_uint8: np.ndarray) -> str:
        # Downsample to 32x32 grayscale → perceptual hash
        thumb = Image.fromarray(image_rgb_uint8).convert('L').resize(
            (32, 32), Image.BILINEAR)
        return hashlib.md5(np.asarray(thumb).tobytes()).hexdigest()

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix('.tmp')
        with tmp.open('w') as f:
            json.dump(self.cache, f)
        tmp.rename(self.cache_path)

    # --------------------------------------------------------------- rate lim
    def _throttle(self) -> None:
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 60]
        if len(self._call_times) >= self.max_calls_per_minute:
            sleep_s = 60 - (now - self._call_times[0]) + 0.1
            if sleep_s > 0:
                time.sleep(sleep_s)
        self._call_times.append(time.time())

    # --------------------------------------------------------------- API call
    def _ask(self, image_rgb_uint8: np.ndarray) -> Tuple[str, float]:
        pil = Image.fromarray(image_rgb_uint8)
        # Crop is small (~100-300 px on each side); send at 256 max edge.
        pil.thumbnail((256, 256), Image.BILINEAR)
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        self._throttle()
        try:
            r = requests.post(
                API_URL,
                json={
                    'model': self.model,
                    'messages': [{
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': PROMPT_TEMPLATE},
                            {'type': 'image_url',
                             'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                        ],
                    }],
                    'max_tokens': 16, 'temperature': 0.0,
                },
                headers={'Authorization': f'Bearer {self.api_key}'},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return 'N', 0.0
            text = r.json()['choices'][0]['message']['content'].strip()
            return self._parse_response(text)
        except Exception:
            return 'N', 0.0

    @staticmethod
    def _parse_response(text: str) -> Tuple[str, float]:
        lines = [s.strip() for s in text.replace('\r', '').split('\n')
                 if s.strip()]
        if not lines:
            return 'N', 0.0
        ch = lines[0].upper()[:1]
        if ch not in 'VPCN':
            return 'N', 0.0
        try:
            conf = float(lines[1])
            conf = max(0.0, min(1.0, conf))
        except (IndexError, ValueError):
            conf = 0.5
        return ch, conf

    # ------------------------------------------------------------------ vote
    @staticmethod
    def _annotated_crop(image_rgb_uint8: np.ndarray,
                        box_xyxy: np.ndarray,
                        context_ratio: float = 1.5) -> np.ndarray:
        """Crop region around ``box_xyxy``, with extra context, and draw a
        RED rectangle on top of the candidate box.

        The VLM needs the red rectangle so it can ground its answer to the
        right region; otherwise it just describes the whole scene.
        """
        H, W = image_rgb_uint8.shape[:2]
        x1, y1, x2, y2 = box_xyxy
        bw = max(8, x2 - x1); bh = max(8, y2 - y1)
        # Context box: 1.5x in each direction, clipped
        cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
        ctx_w = bw * (1 + context_ratio); ctx_h = bh * (1 + context_ratio)
        cx1 = int(max(0, cx - ctx_w / 2))
        cy1 = int(max(0, cy - ctx_h / 2))
        cx2 = int(min(W, cx + ctx_w / 2))
        cy2 = int(min(H, cy + ctx_h / 2))
        if cx2 <= cx1 + 8 or cy2 <= cy1 + 8:
            cx1, cy1, cx2, cy2 = 0, 0, min(64, W), min(64, H)
        crop = image_rgb_uint8[cy1:cy2, cx1:cx2].copy()

        # Coordinates of the candidate box inside the crop:
        rx1 = max(0, int(x1) - cx1); ry1 = max(0, int(y1) - cy1)
        rx2 = min(crop.shape[1] - 1, int(x2) - cx1)
        ry2 = min(crop.shape[0] - 1, int(y2) - cy1)
        if rx2 > rx1 + 2 and ry2 > ry1 + 2:
            # Draw thick red rectangle (3-px border)
            for t in range(3):
                if rx1 + t < crop.shape[1] and rx2 - t >= 0:
                    crop[ry1:ry2 + 1, rx1 + t] = [255, 0, 0]
                    crop[ry1:ry2 + 1, max(0, rx2 - t)] = [255, 0, 0]
                if ry1 + t < crop.shape[0] and ry2 - t >= 0:
                    crop[ry1 + t, rx1:rx2 + 1] = [255, 0, 0]
                    crop[max(0, ry2 - t), rx1:rx2 + 1] = [255, 0, 0]
        return crop

    CHAR_TO_CLS = {'V': 'Vehicle', 'P': 'Pedestrian',
                   'C': 'Cyclist', 'N': 'None'}

    def vote(self,
             image_rgb_uint8: np.ndarray,
             box_xyxy_image: np.ndarray) -> Tuple[str, float]:
        """Return (transfer_class_or_None, confidence)."""
        crop = self._annotated_crop(image_rgb_uint8, box_xyxy_image)
        if crop.size == 0:
            return 'None', 0.0
        h = self._crop_hash(crop)
        if h in self.cache:
            self._n_cache_hits += 1
            d = self.cache[h]
            return self.CHAR_TO_CLS.get(d['ch'], 'None'), d['conf']
        ch, conf = self._ask(crop)
        self._n_calls += 1
        self.cache[h] = {'ch': ch, 'conf': conf}
        return self.CHAR_TO_CLS.get(ch, 'None'), conf

    def stats(self) -> Dict:
        return dict(api_calls=self._n_calls, cache_hits=self._n_cache_hits,
                    cache_size=len(self.cache))
