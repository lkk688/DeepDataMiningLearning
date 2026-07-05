"""
ngperception.lane.culane_dataset
================================

Two datasets with the **same interface** for the CLRNet trainer:

* :class:`CULaneDataset` — reads the **real CULane format**: a list file of image paths,
  each image ``foo.jpg`` paired with ``foo.lines.txt`` whose every line is one lane as
  ``x1 y1 x2 y2 …`` (pixels in the original 1640×590 image). Drop-in once CULane is staged.
* :class:`SyntheticLanes` — draws random smooth lanes on a road-like background with known
  ground truth, for **local de-risking** (no download): it exercises the whole pipeline —
  anchor assignment, LaneIoU, decode, F1 — under fully-controlled geometry. Clearly *not* a
  benchmark number, just a correctness harness (same role as the overfit-a-few-samples
  sanity check used for the detection module).

Both yield ``(image_tensor(3,H,W), lanes)`` where ``lanes`` is a list of ``(K,2)`` float
tensors of ``(x,y)`` points in **resized** pixel coords, plus a ``meta`` dict (orig H/W,
path) used by the evaluator to map predictions back to original resolution.
"""
from __future__ import annotations
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def _to_tensor(img_np, img_h, img_w):
    im = Image.fromarray(img_np).resize((img_w, img_h), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def _augment(img, lanes, img_w):
    """Geometrically-safe lane augmentation on a target-size (H,W,3) uint8 image + its
    lane point-lists (in target pixels). Horizontal flip (x → W−1−x, lanes stay valid) +
    photometric brightness/contrast jitter. Augmentation is *the* generalisation lever for
    lane detectors — CLRNet leans on random affine/flip/HSV; without it a model memorises
    the training drivers (we measured train-F1 0.89 vs unseen-driver 0.10)."""
    if np.random.rand() < 0.5:                       # horizontal flip
        img = img[:, ::-1, :].copy()
        lanes = [torch.stack([(img_w - 1) - l[:, 0], l[:, 1]], dim=1) for l in lanes]
    b = np.random.uniform(0.7, 1.3)                  # brightness
    c = np.random.uniform(0.8, 1.2)                  # contrast (around image mean)
    mean = float(img.mean())
    img = np.clip((img.astype(np.float32) - mean) * c + mean * b, 0, 255).astype(np.uint8)
    return img, lanes


def collate_lanes(batch):
    """Stack images; keep lanes + meta as lists (variable #lanes per image)."""
    imgs = torch.stack([b[0] for b in batch])
    lanes = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return imgs, lanes, metas


class CULaneDataset(Dataset):
    def __init__(self, root, list_file, img_h=320, img_w=800, max_samples=None, augment=False):
        self.root, self.img_h, self.img_w = root, img_h, img_w
        self.augment = augment
        with open(os.path.join(root, list_file)) as f:
            lines = [l.strip().split()[0] for l in f if l.strip()]
        self.items = lines[:max_samples] if max_samples else lines

    def __len__(self):
        return len(self.items)

    def _load_lanes(self, ann_path):
        lanes = []
        if not os.path.exists(ann_path):
            return lanes
        with open(ann_path) as f:
            for line in f:
                v = [float(x) for x in line.split()]
                if len(v) < 4:
                    continue
                pts = np.array(v, dtype=np.float32).reshape(-1, 2)
                pts = pts[pts[:, 0] >= 0]                      # CULane marks off-image x as -1..
                if len(pts) >= 2:
                    lanes.append(pts)
        return lanes

    def __getitem__(self, i):
        rel = self.items[i].lstrip("/")
        img_path = os.path.join(self.root, rel)
        img = np.asarray(Image.open(img_path).convert("RGB"))
        H0, W0 = img.shape[:2]
        ann = os.path.splitext(img_path)[0] + ".lines.txt"
        lanes0 = self._load_lanes(ann)
        sx, sy = self.img_w / W0, self.img_h / H0
        lanes = [torch.from_numpy(p * np.array([sx, sy], np.float32)) for p in lanes0]
        if self.augment:
            img_rs = np.asarray(Image.fromarray(img).resize((self.img_w, self.img_h), Image.BILINEAR))
            img_rs, lanes = _augment(img_rs, lanes, self.img_w)
            t = (torch.from_numpy(img_rs).permute(2, 0, 1).float() / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        else:
            t = _to_tensor(img, self.img_h, self.img_w)
        meta = {"path": img_path, "H0": H0, "W0": W0, "sx": sx, "sy": sy}
        return t, lanes, meta


class SyntheticLanes(Dataset):
    """Random smooth lanes (quadratic in y) on a textured background — controlled GT.

    Deterministic per index (seeded by index), so train/val splits are stable and the
    overfit sanity check is reproducible.
    """

    def __init__(self, n=256, img_h=320, img_w=800, min_lanes=2, max_lanes=4, seed=0):
        self.n, self.img_h, self.img_w = n, img_h, img_w
        self.min_lanes, self.max_lanes, self.seed = min_lanes, max_lanes, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.RandomState(self.seed * 100000 + i)
        H, W = self.img_h, self.img_w
        img = (rng.rand(H, W, 3) * 40 + 90).astype(np.uint8)      # grey road-ish noise
        n_lanes = rng.randint(self.min_lanes, self.max_lanes + 1)
        ys = np.linspace(H * 0.45, H - 1, 40)                     # lanes live in lower half
        lanes = []
        centers = np.sort(rng.uniform(W * 0.1, W * 0.9, n_lanes))
        for cx in centers:
            curv = rng.uniform(-1.2, 1.2) / (H * H)               # gentle curvature
            slope = rng.uniform(-0.6, 0.6)
            xs = cx + slope * (ys - H) + curv * (ys - H) ** 2 * W
            pts = np.stack([xs, ys], 1).astype(np.float32)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W)]
            if len(pts) >= 2:
                lanes.append(pts)
                # paint the lane (thin bright band) so a real backbone has signal
                for x, y in pts:
                    xi, yi = int(x), int(y)
                    img[max(0, yi - 1):yi + 2, max(0, xi - 2):xi + 3] = 230
        t = _to_tensor(img, H, W)
        lanes_t = [torch.from_numpy(p) for p in lanes]
        meta = {"path": f"synthetic/{i}", "H0": H, "W0": W, "sx": 1.0, "sy": 1.0}
        return t, lanes_t, meta
