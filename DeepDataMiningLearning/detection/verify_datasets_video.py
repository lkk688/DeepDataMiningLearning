"""
Verify 2D bounding-box export for Waymo / KITTI / NuScenes by rendering a video.

Each dataset loader is exercised through its real __getitem__, the returned
[xmin,ymin,xmax,ymax] boxes (pixel space) are drawn onto the image, and the
frames are written to an mp4 so the 2D labels can be checked visually before
training.

Usage (py312 env which has pyarrow):
    python -m DeepDataMiningLearning.detection.verify_datasets_video --dataset waymo  --num 150
    python -m DeepDataMiningLearning.detection.verify_datasets_video --dataset kitti  --num 150
    python -m DeepDataMiningLearning.detection.verify_datasets_video --dataset nuscenes --num 200 \
        --nuscenes-root /mnt/e/Shared/Dataset/NuScenes/v1.0-trainval

NuScenes note: the loader expects a "v1.0-trainval" sub-folder inside --nuscenes-root.
For the mini split, point --nuscenes-root at a folder whose "v1.0-trainval" symlinks
to the mini json folder (see build_nuscenes_mini_root()).
"""
import os
import argparse
import numpy as np
import cv2
import torch
from PIL import Image


# ---------------- drawing helpers ----------------

def to_bgr_uint8(img):
    """PIL.Image | np.ndarray(HWC) | torch.Tensor(CHW) -> BGR uint8."""
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"))
    elif isinstance(img, torch.Tensor):
        a = img.detach().cpu().float()
        if a.dim() == 3 and a.shape[0] in (1, 3):
            a = a.permute(1, 2, 0)
        a = a.numpy()
        if a.min() < 0:  # likely ImageNet-normalized
            a = a * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        if a.max() <= 1.0 + 1e-6:
            a = a * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
    else:
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


_PALETTE = [
    (245, 135, 66), (66, 66, 245), (105, 245, 66), (66, 209, 245),
    (245, 66, 188), (233, 245, 66), (66, 132, 245), (140, 140, 140),
    (167, 245, 66), (167, 66, 245),
]


def draw_boxes(bgr, boxes, labels, class_names=None):
    if boxes is None or len(boxes) == 0:
        return bgr
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    boxes = np.asarray(boxes).reshape(-1, 4)
    labels = np.asarray(labels).reshape(-1) if labels is not None and len(labels) else np.zeros(len(boxes), int)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(float(v))) for v in b[:4]]
        lid = int(labels[i]) if i < len(labels) else 0
        color = _PALETTE[lid % len(_PALETTE)]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        name = class_names[lid] if class_names and 0 <= lid < len(class_names) else str(lid)
        cv2.putText(bgr, name, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return bgr


class VideoWriter:
    def __init__(self, path, fps=5):
        self.path, self.fps, self.size, self.vw = str(path), fps, None, None

    def write(self, bgr):
        if self.vw is None:
            self.size = (bgr.shape[1], bgr.shape[0])
            self.vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        if (bgr.shape[1], bgr.shape[0]) != self.size:
            bgr = cv2.resize(bgr, self.size)
        self.vw.write(bgr)

    def release(self):
        if self.vw is not None:
            self.vw.release()


# ---------------- per-dataset runners ----------------

def run_waymo(base, out, num, fps):
    from DeepDataMiningLearning.detection.dataset_waymov3_1 import Waymo2DDataset, WAYMO_CLASSES
    names = [WAYMO_CLASSES.get(i, str(i)) for i in range(max(WAYMO_CLASSES) + 1)]
    ds = Waymo2DDataset(base, split="training", max_frames=num)
    return _render(ds, out, num, fps, names, torch_img=True)


def run_kitti(root, out, num, fps):
    from DeepDataMiningLearning.detection.dataset_kitti import KittiDataset
    ds = KittiDataset(root, train=True, split="train", output_format="torch")
    return _render(ds, out, num, fps, ds.INSTANCE_CATEGORY_NAMES, torch_img=False)


def run_nuscenes(root, out, num, fps):
    from DeepDataMiningLearning.detection.dataset_nuscenes import NuScenesDataset, CATEGORY_NAMES
    ds = NuScenesDataset(root_dir=root, split="train", camera_types=["CAM_FRONT"],
                         transform=None, validate_on_init=False)
    return _render(ds, out, num, fps, CATEGORY_NAMES, torch_img=False)


def _render(ds, out, num, fps, names, torch_img):
    n = min(num, len(ds))
    print(f"dataset frames: {len(ds)}, rendering {n}")
    vw, total, with_boxes = VideoWriter(out, fps=fps), 0, 0
    for i in range(n):
        img, tgt = ds[i]
        boxes, labels = tgt.get("boxes", []), tgt.get("labels", [])
        total += len(boxes)
        with_boxes += 1 if len(boxes) else 0
        bgr = to_bgr_uint8(img)
        draw_boxes(bgr, boxes, labels, names)
        vw.write(bgr)
    vw.release()
    print(f"DONE -> {out} | frames={n} frames_with_boxes={with_boxes} total_boxes={total}")
    return out


def build_nuscenes_mini_root(mini_dir, dst):
    """Create a root with a v1.0-trainval symlink so the mini split works with the loader."""
    os.makedirs(dst, exist_ok=True)
    for sub, target in [("samples", "samples"), ("sweeps", "sweeps"),
                        ("maps", "maps"), ("v1.0-trainval", "v1.0-mini")]:
        link = os.path.join(dst, sub)
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link) if os.path.islink(link) else None
        os.symlink(os.path.join(mini_dir, target), link)
    return dst


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["waymo", "kitti", "nuscenes"])
    p.add_argument("--num", type=int, default=150)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--out", default=None)
    p.add_argument("--waymo-base", default="/mnt/e/Shared/Dataset/waymodata")
    p.add_argument("--kitti-root", default="/mnt/e/Shared/Dataset/Kitti/")
    p.add_argument("--nuscenes-root", default="/mnt/e/Shared/Dataset/NuScenes/v1.0-trainval")
    a = p.parse_args()

    if a.dataset == "waymo":
        run_waymo(a.waymo_base, a.out or os.path.join(a.waymo_base, "verify_waymo_2d.mp4"), a.num, a.fps)
    elif a.dataset == "kitti":
        run_kitti(a.kitti_root, a.out or os.path.join(a.kitti_root, "verify_kitti_2d.mp4"), a.num, a.fps)
    else:
        run_nuscenes(a.nuscenes_root, a.out or "/mnt/e/Shared/Dataset/NuScenes/verify_nuscenes_2d.mp4", a.num, a.fps)
