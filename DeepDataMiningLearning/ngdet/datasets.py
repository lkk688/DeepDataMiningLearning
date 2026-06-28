"""
ngdet.datasets
==============

Thin wrappers that turn the project's existing dataset loaders into ONE common
evaluation interface, with ground-truth labels already folded into the active
unified taxonomy.

Each adapter yields per-image samples:

    EvalSample(
        image_id : int,            # unique, contiguous index
        image    : PIL.Image RGB,  # original resolution
        gt_boxes : np.ndarray [M,4] xyxy absolute pixels
        gt_labels: np.ndarray [M]  unified class ids
    )

We REUSE the existing loaders (no re-implementation):
    * KITTI    -> detection.dataset_kitti.KittiDataset
    * Waymo    -> detection.dataset_waymov3_1.Waymo2DDataset
    * NuScenes -> detection.dataset_nuscenes.NuScenesDataset
    * COCO     -> torchvision.datasets.CocoDetection (val2017)

Only the label-space projection (dataset-native id -> unified id) lives here, via
the id2name tables in ngdet.taxonomy.
"""

from __future__ import annotations
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

from .taxonomy import (
    Taxonomy, KITTI_ID2NAME, WAYMO_ID2NAME, NUSCENES_ID2NAME,
)


def nuimages_name_to_unified(dotted_name: str, taxonomy):
    """Map a nuImages dotted category (e.g. 'vehicle.car', 'human.pedestrian.adult',
    'vehicle.bicycle') to a unified id. We try the dot-separated tokens RIGHT TO
    LEFT (most specific first), so 'vehicle.bicycle' -> cyclist (via 'bicycle')
    rather than vehicle (via 'vehicle')."""
    for tok in reversed(str(dotted_name).split(".")):
        uid = taxonomy.name_to_id(tok)
        if uid is not None:
            return uid
    return None


@dataclass
class EvalSample:
    image_id: int
    image: Image.Image
    gt_boxes: np.ndarray   # [M,4] xyxy abs px
    gt_labels: np.ndarray  # [M] unified ids


def _to_pil(img) -> Image.Image:
    """Coerce a loader's image (PIL or CHW float/uint8 tensor) to a PIL RGB image."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    import torch
    if isinstance(img, torch.Tensor):
        a = img.detach().cpu().float()
        if a.dim() == 3 and a.shape[0] in (1, 3):
            a = a.permute(1, 2, 0)
        a = a.numpy()
        if a.max() <= 1.0 + 1e-6:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype("uint8")
        if a.shape[2] == 1:
            a = np.repeat(a, 3, axis=2)
        return Image.fromarray(a)
    return Image.fromarray(np.asarray(img)).convert("RGB")


class EvalDataset:
    """Common iterable over (image, unified GT) samples for one source dataset.

    Parameters
    ----------
    name : str           one of {"kitti","waymo","nuscenes","coco"}
    root : str           dataset root path
    taxonomy : Taxonomy  active unified label space
    max_images : int     cap for quick runs (None = all)
    **kw                 passed through to the underlying loader
    """

    def __init__(self, name: str, root: str, taxonomy: Taxonomy,
                 max_images: Optional[int] = None, stride: int = 1, **kw):
        self.name = name
        self.taxonomy = taxonomy
        self.max_images = max_images
        # stride>1 subsamples the underlying dataset (every stride-th item). For
        # Waymo this is essential: frames are grouped by segment, so a contiguous
        # slice covers only the first segment(s); striding spreads the sample
        # across many segments (and thus across pedestrians/cyclists, not just
        # the vehicle-heavy opening segment).
        self.stride = max(1, int(stride))
        builder = {
            "kitti": self._build_kitti,
            "waymo": self._build_waymo,
            "nuscenes": self._build_nuscenes,
            "nuimages": self._build_nuimages,
            "coco": self._build_coco,
            "mixed": self._build_mixed,
        }.get(name)
        if builder is None:
            raise KeyError(f"Unknown dataset '{name}'. Choose from "
                           f"kitti/waymo/nuscenes/nuimages/coco/mixed.")
        builder(root, **kw)
        # Build the (strided, capped) list of underlying indices we will read.
        idxs = list(range(0, len(self._ds), self.stride))
        if self.max_images:
            idxs = idxs[:self.max_images]
        self._indices = idxs

    # -- builders: set self._ds, self._gt_lut, self._get(i) ------------------
    def _build_kitti(self, root, split="train", **kw):
        from DeepDataMiningLearning.detection.dataset_kitti import KittiDataset
        self._ds = KittiDataset(root, train=True, split=split, output_format="torch")
        self._gt_lut = self.taxonomy.build_id_lut(KITTI_ID2NAME)
        self._mode = "torch_target"

    def _build_waymo(self, root, split="training", **kw):
        from DeepDataMiningLearning.detection.dataset_waymov3_1 import Waymo2DDataset
        # Build a frame index large enough that striding spans several segments.
        # (max_frames caps the time-ordered index; we then stride within it.)
        n_build = (self.max_images or 200) * self.stride if self.max_images else None
        self._ds = Waymo2DDataset(root, split=split, max_frames=n_build)
        self._gt_lut = self.taxonomy.build_id_lut(WAYMO_ID2NAME)
        self._mode = "torch_target"

    def _build_nuscenes(self, root, **kw):
        from DeepDataMiningLearning.detection.dataset_nuscenes import NuScenesDataset
        self._ds = NuScenesDataset(root_dir=root, split=kw.get("split", "train"),
                                   camera_types=["CAM_FRONT"], transform=None,
                                   validate_on_init=False)
        self._gt_lut = self.taxonomy.build_id_lut(NUSCENES_ID2NAME)
        self._mode = "torch_target"

    def _build_nuimages(self, root, version="v1.0-mini", **kw):
        """nuImages: REAL human-annotated 2D boxes (unlike nuscenes' projected 3D).

        We parse the JSON tables directly (no devkit dependency):
          * object_ann.json : 2D boxes -> bbox [x1,y1,x2,y2] (pixels) + category_token
          * sample_data.json: key-frame images (token -> filename, w, h)
          * category.json   : category_token -> dotted name
        and build an in-memory list of (image_path, [(bbox_xyxy, category_name), ...]).
        """
        import json
        ann_dir = os.path.join(root, version)
        cats = {c["token"]: c["name"]
                for c in json.load(open(os.path.join(ann_dir, "category.json")))}
        sdata = json.load(open(os.path.join(ann_dir, "sample_data.json")))
        oann = json.load(open(os.path.join(ann_dir, "object_ann.json")))

        # token -> sample_data record (key-frame camera images only)
        sd_by_tok = {s["token"]: s for s in sdata
                     if s.get("is_key_frame") and s.get("fileformat") in ("jpg", "png")}
        # group annotations by image
        anns_by_img = defaultdict(list)
        for o in oann:
            sdt = o["sample_data_token"]
            if sdt in sd_by_tok:
                anns_by_img[sdt].append((o["bbox"], cats[o["category_token"]]))

        # records: one per annotated key-frame image, sorted for determinism
        self._records = []
        for sdt in sorted(anns_by_img.keys()):
            sd = sd_by_tok[sdt]
            self._records.append((os.path.join(root, sd["filename"]),
                                  anns_by_img[sdt]))
        self._ds = self._records          # so __init__ stride/len logic works
        self._mode = "nuimages"

    def _build_mixed(self, root, split="train", **kw):
        """The mixed KITTI+Waymo+nuImages base built by ngdet.mixed_dataset — a
        COCO-format dir (`images/` + `train.json`/`val.json`) in unified categories."""
        import torchvision
        img_dir = os.path.join(root, "images")
        ann = os.path.join(root, f"{split}.json")
        self._ds = torchvision.datasets.CocoDetection(img_dir, ann)
        coco = self._ds.coco
        cocoid2name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
        self._gt_lut = self.taxonomy.build_id_lut(cocoid2name)
        self._mode = "coco"

    def _build_coco(self, root, ann_file=None, **kw):
        # root points to the val images dir; ann_file to instances_val2017.json.
        import torchvision
        if ann_file is None:
            raise ValueError("coco requires ann_file=<instances_val2017.json>")
        self._ds = torchvision.datasets.CocoDetection(root, ann_file)
        # COCO category id -> name, then fold names into unified ids.
        coco = self._ds.coco
        cocoid2name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
        self._gt_lut = self.taxonomy.build_id_lut(cocoid2name)
        self._mode = "coco"

    # -- common access -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i: int) -> EvalSample:
        # i is the contiguous eval index; map it to the underlying (strided) index.
        u = self._indices[i]
        if self._mode == "nuimages":
            # records are (image_path, [(bbox_xyxy, dotted_category_name), ...]).
            img_path, anns = self._ds[u]
            pil = Image.open(img_path).convert("RGB")
            keep_b, keep_l = [], []
            for bbox, cname in anns:
                uid = nuimages_name_to_unified(cname, self.taxonomy)
                if uid is None:
                    continue
                keep_b.append([float(v) for v in bbox])   # already xyxy pixels
                keep_l.append(uid)
            return EvalSample(
                image_id=i, image=pil,
                gt_boxes=np.asarray(keep_b, np.float32).reshape(-1, 4),
                gt_labels=np.asarray(keep_l, np.int64),
            )
        if self._mode == "torch_target":
            img, target = self._ds[u]
            pil = _to_pil(img)
            boxes = target.get("boxes")
            labels = target.get("labels")
            boxes = (boxes.cpu().numpy() if hasattr(boxes, "cpu")
                     else np.asarray(boxes)).reshape(-1, 4)
            labels = (labels.cpu().numpy() if hasattr(labels, "cpu")
                      else np.asarray(labels)).reshape(-1)
        else:  # coco: target is a list of annotation dicts with xywh bbox
            img, anns = self._ds[u]
            pil = img.convert("RGB")
            boxes, labels = [], []
            for a in anns:
                x, y, bw, bh = a["bbox"]                 # COCO xywh
                boxes.append([x, y, x + bw, y + bh])     # -> xyxy
                labels.append(a["category_id"])
            boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
            labels = np.asarray(labels).reshape(-1)

        # Fold native GT labels into unified ids; drop classes outside taxonomy.
        keep_b, keep_l = [], []
        for b, l in zip(boxes, labels):
            uid = self._gt_lut.get(int(l), None)
            if uid is None:
                continue
            keep_b.append(b)
            keep_l.append(uid)
        return EvalSample(
            image_id=i,
            image=pil,
            gt_boxes=np.asarray(keep_b, np.float32).reshape(-1, 4),
            gt_labels=np.asarray(keep_l, np.int64),
        )


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# Uses the locally-mounted datasets. KITTI is the quickest to verify:
#
#   python -m DeepDataMiningLearning.ngdet.datasets \
#       --name kitti --root /mnt/e/Shared/Dataset/Kitti/ --max-images 3
#
# Expected: prints image sizes and the unified GT label counts for a few frames.
# ===========================================================================
if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="kitti")
    ap.add_argument("--root", default="/mnt/e/Shared/Dataset/Kitti/")
    ap.add_argument("--max-images", type=int, default=3)
    ap.add_argument("--taxonomy", default="driving3")
    a = ap.parse_args()

    tax = Taxonomy.from_preset(a.taxonomy)
    ds = EvalDataset(a.name, a.root, tax, max_images=a.max_images)
    print(f"{a.name}: {len(ds)} images (capped), taxonomy={tax.classes}")
    counts = Counter()
    for s in ds:
        for l in s.gt_labels:
            counts[tax.classes[int(l)]] += 1
        print(f"  img {s.image_id}: size={s.image.size} gt_boxes={len(s.gt_boxes)} "
              f"labels={[tax.classes[int(l)] for l in s.gt_labels]}")
    print("unified GT class counts:", dict(counts))
