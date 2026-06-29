"""
ngperception.occupancy.datasets
===============================

Loader for the **Occ3D-nuScenes** GT (the `gts/scene-XXXX/<sample_token>/labels.npz`
layout from Tsinghua-MARS-Lab/Occ3D). Each sample is a (200,200,16) semantic voxel grid
plus `mask_camera` / `mask_lidar` visibility masks.

For the depth->occupancy baseline (Phase 2) we also need the 6 surround images + their
calibration; those are keyed by the same nuScenes `sample_token` and resolved lazily via
the nuScenes devkit (see predictors/depth_lift.py), so this loader stays dependency-free.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class OccSample:
    semantics: np.ndarray    # (200,200,16) uint8
    mask_camera: np.ndarray  # (200,200,16) uint8
    mask_lidar: np.ndarray   # (200,200,16) uint8
    sample_token: str
    scene: str


class Occ3DNuScenesDataset:
    """Scans an extracted Occ3D `gts` directory for `labels.npz` files.

    Parameters
    ----------
    gts_root : str
        Directory containing `scene-XXXX/<token>/labels.npz` (i.e. the extracted `gts/`).
    scenes : list[str] | None
        Restrict to these scene names (e.g. a val split); None = all found.
    max_samples, stride : int
        Subsample for quick runs.
    """

    def __init__(self, gts_root: str, scenes: Optional[List[str]] = None,
                 max_samples: Optional[int] = None, stride: int = 1):
        self.gts_root = gts_root
        items = []
        scene_dirs = sorted(os.listdir(gts_root)) if scenes is None else scenes
        for sc in scene_dirs:
            sd = os.path.join(gts_root, sc)
            if not os.path.isdir(sd):
                continue
            for tok in sorted(os.listdir(sd)):
                lp = os.path.join(sd, tok, "labels.npz")
                if os.path.isfile(lp):
                    items.append((sc, tok, lp))
        items = items[::stride]
        if max_samples:
            items = items[:max_samples]
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> OccSample:
        sc, tok, lp = self.items[i]
        d = np.load(lp)
        return OccSample(semantics=d["semantics"], mask_camera=d["mask_camera"],
                         mask_lidar=d["mask_lidar"], sample_token=tok, scene=sc)


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.occupancy.datasets --gts <extracted_gts_dir>
# Prints per-sample occupied/visible voxel counts for a few frames.
# ===========================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gts", required=True, help="extracted Occ3D gts/ directory")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    ds = Occ3DNuScenesDataset(a.gts, max_samples=a.n)
    print(f"Occ3D: {len(ds)} samples found")
    for i in range(len(ds)):
        s = ds[i]
        occ = int((s.semantics != 17).sum())
        vis = int(s.mask_camera.sum())
        print(f"  {s.scene}/{s.sample_token[:12]}: occupied={occ} cam_visible={vis}")
