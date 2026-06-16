"""Local Qwen2.5-VL drop-in replacement for :class:`VLMVoter`.

Talks to an OpenAI-compatible vLLM server (started by
``start_vllm_server.sh``). Same red-box visual prompt; same V/P/C/N
return signature; same cache + throttle so it can be swapped in
without touching the upstream pipeline (``bulk_pseudo_label_v1.py``,
``fusion.py``).

Usage:
  # Terminal 1:
  bash phase2a/start_vllm_server.sh
  # Terminal 2:
  from phase2a.vlm_voter_local import LocalVLMVoter
  vlm = LocalVLMVoter(server_url='http://localhost:8000/v1',
                      cache_path='/tmp/vlm_cache_local.json')
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
import requests
from PIL import Image

# Re-use the prompt + parser from the canonical VLMVoter so behaviour is
# identical except for the backend.
from DeepDataMiningLearning.detection3d.phase2a.vlm_voter import (
    PROMPT_TEMPLATE, VLMVoter,
)


# Match the --served-model-name in start_vllm_server.sh. The team's
# default is Qwen3.6-27B-FP8 served as "qwen-27b"; pass `model=gemma-4-31b`
# to talk to the Gemma container instead.
DEFAULT_LOCAL_MODEL = 'qwen-27b'

# HPC-specific: localhost endpoints must bypass the corporate proxy.
# `requests` honours these env vars via `trust_env=True`; we set them
# explicitly so users don't have to remember the `--noproxy "*"` flag.
_NO_PROXY = {
    'http': None, 'https': None,
}


class LocalVLMVoter:
    """vLLM-backed Qwen2.5-VL voter with the same interface as
    :class:`VLMVoter`."""

    def __init__(self,
                 server_url: str = 'http://localhost:8000/v1',
                 model: str = DEFAULT_LOCAL_MODEL,
                 cache_path: Optional[str] = None,
                 max_calls_per_minute: int = 600,
                 timeout_s: float = 60.0):
        self.server_url = server_url.rstrip('/')
        self.model = model
        self.cache_path = Path(cache_path) if cache_path else None
        # Bypass proxy for localhost requests on this HPC (matches the
        # `curl --noproxy "*"` invocation used to test the endpoint).
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.proxies = _NO_PROXY
        self.cache: Dict[str, Dict] = {}
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open() as f:
                self.cache = json.load(f)
        self.max_calls_per_minute = int(max_calls_per_minute)
        self.timeout = float(timeout_s)
        self._call_times: List[float] = []
        self._n_calls = 0
        self._n_cache_hits = 0

    # Re-use VLMVoter's helpers (composition over inheritance).
    _crop_hash = VLMVoter._crop_hash
    _annotated_crop = VLMVoter._annotated_crop
    _parse_response = VLMVoter._parse_response
    _save_cache = VLMVoter._save_cache
    _throttle = VLMVoter._throttle
    CHAR_TO_CLS = VLMVoter.CHAR_TO_CLS

    def _ask(self, image_rgb_uint8: np.ndarray) -> Tuple[str, float]:
        """OpenAI chat-completion call against local vLLM server."""
        pil = Image.fromarray(image_rgb_uint8)
        pil.thumbnail((384, 384), Image.BILINEAR)   # Qwen-VL likes 384+
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        self._throttle()
        try:
            r = self._session.post(
                f'{self.server_url}/chat/completions',
                json={
                    'model': self.model,
                    'messages': [{
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': PROMPT_TEMPLATE},
                            {'type': 'image_url',
                             'image_url': {
                                 'url': f'data:image/jpeg;base64,{b64}'}},
                        ],
                    }],
                    'max_tokens': 16,
                    'temperature': 0.0,
                },
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return 'N', 0.0
            text = r.json()['choices'][0]['message']['content'].strip()
            return self._parse_response(text)
        except Exception:
            return 'N', 0.0

    def vote(self, image_rgb_uint8: np.ndarray,
              box_xyxy_image: np.ndarray) -> Tuple[str, float]:
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
                    cache_size=len(self.cache),
                    backend='local-vLLM', model=self.model)


if __name__ == '__main__':
    # Quick smoketest with a dummy red-box crop.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://localhost:8000/v1')
    args = ap.parse_args()
    voter = LocalVLMVoter(server_url=args.server,
                            cache_path='/tmp/vlm_local_smoketest.json')
    # Make a synthetic crop with red box
    img = np.full((128, 128, 3), 60, dtype=np.uint8)
    img[40:90, 30:90] = (200, 30, 30)
    cls, conf = voter.vote(img, np.array([30, 40, 90, 90], dtype=np.float32))
    print(f'smoketest: predicted={cls} conf={conf:.3f}')
    print(f'stats: {voter.stats()}')
