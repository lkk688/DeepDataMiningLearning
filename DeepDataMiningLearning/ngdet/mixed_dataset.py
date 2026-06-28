"""
ngdet.mixed_dataset
==================

Build a **mixed multi-dataset** training base by sampling KITTI + Waymo + nuImages,
projecting every box into the unified taxonomy, and writing ONE on-disk dataset in
**COCO format** (images + `train.json` / `val.json`). Because it is COCO format with
unified category ids, the SAME dataset trains all backends:
  * torchvision (Faster R-CNN) and HF DETR  → via `--dataset mixed`
  * YOLO (native)                           → exported to YOLO format on the fly

Why mix? A model trained on a single domain transfers poorly (see TUTORIAL §5a). A
mixed base is the simplest way to improve cross-domain robustness — and this script
is the data side of that experiment.

Layout written:
    <out>/
      images/         all jpgs, named "<source>_<idx>.jpg"
      train.json      COCO annotations (unified categories) for the train split
      val.json        COCO annotations for the val split
      manifest.json   per-source counts + config (for reproducibility)
"""

from __future__ import annotations
import argparse
import json
import os
import random
from typing import Dict, List

from .taxonomy import Taxonomy
from .datasets import EvalDataset

# Default per-source roots (override on the CLI with --roots name=path).
DEFAULT_ROOTS = {
    "kitti": "/mnt/e/Shared/Dataset/Kitti/",
    "waymo": "/mnt/e/Shared/Dataset/waymodata",
    "nuimages": "/mnt/e/Shared/Dataset/NuScenes/nuimages",
}


def _source_dataset(name: str, root: str, taxonomy: Taxonomy, per_source: int,
                    waymo_stride: int, nuimages_version: str) -> EvalDataset:
    kw = {}
    if name == "waymo":
        kw["stride"] = waymo_stride                 # span many segments
    if name == "nuimages":
        kw["version"] = nuimages_version
    return EvalDataset(name, root, taxonomy, max_images=per_source, **kw)


def build_mixed(out_dir: str, sources: List[str], per_source: int, taxonomy: Taxonomy,
                roots: Dict[str, str], test_per_source: int = 150, seed: int = 0,
                waymo_stride: int = 40, nuimages_version: str = "v1.0-train",
                skip_empty: bool = True) -> Dict:
    """Sample each source into a TRAIN block and a DISJOINT held-out TEST block, then
    write a unified-taxonomy COCO dataset with **leakage-free** splits:

        train.json          — training pool (all sources, block A)
        val.json            — held-out mixed (all sources, block B)  [for monitoring]
        test_<source>.json  — held-out per source (block B, one source)

    Because train (block A) and test (block B) images are disjoint by construction,
    evaluating a model trained on train.json against val/test_* is leakage-free.
    """
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    images, annotations = [], []           # records carry "source" and "split"
    ann_id, img_id = 1, 0
    per_src_counts = {}

    for src in sources:
        # Sample enough non-empty images for BOTH blocks; first `per_source` -> train,
        # next `test_per_source` -> the held-out test block (disjoint indices).
        need = per_source + test_per_source
        ds = _source_dataset(src, roots[src], taxonomy, need, waymo_stride, nuimages_version)
        kept = 0
        for i in range(len(ds)):
            if kept >= need:
                break
            s = ds[i]
            if skip_empty and len(s.gt_boxes) == 0:
                continue
            split = "train" if kept < per_source else "test"
            fname = f"{src}_{i:06d}.jpg"
            s.image.save(os.path.join(img_dir, fname), quality=92)
            w, h = s.image.size
            images.append({"id": img_id, "file_name": fname, "width": w, "height": h,
                           "source": src, "split": split})
            for (x1, y1, x2, y2), lab in zip(s.gt_boxes, s.gt_labels):
                bw, bh = float(x2 - x1), float(y2 - y1)
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": int(lab) + 1,
                    "bbox": [float(x1), float(y1), bw, bh], "area": bw * bh, "iscrowd": 0})
                ann_id += 1
            img_id += 1
            kept += 1
        per_src_counts[src] = kept
        print(f"  [{src}] kept {kept} ({min(kept, per_source)} train + "
              f"{max(0, kept - per_source)} test)")

    categories = [{"id": i + 1, "name": n} for i, n in enumerate(taxonomy.classes)]

    def _write(name, keep_imgs):
        ids = {im["id"] for im in keep_imgs}
        anns = [a for a in annotations if a["image_id"] in ids]
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump({"images": keep_imgs, "annotations": anns,
                       "categories": categories}, f)
        print(f"  wrote {name}.json: {len(keep_imgs)} imgs, {len(anns)} anns")
        return len(keep_imgs), len(anns)

    train_imgs = [im for im in images if im["split"] == "train"]
    test_imgs = [im for im in images if im["split"] == "test"]
    # Shuffle the train order so that taking the first --max-images during training
    # yields a balanced (cross-source) subset -- needed for clean data-size studies.
    random.Random(seed + 1).shuffle(train_imgs)
    splits = {"train": _write("train", train_imgs),
              "val": _write("val", test_imgs)}                 # held-out mixed
    for src in sources:                                        # held-out per source
        splits[f"test_{src}"] = _write(f"test_{src}",
                                       [im for im in test_imgs if im["source"] == src])

    manifest = {
        "taxonomy": taxonomy.name, "classes": taxonomy.classes, "sources": sources,
        "per_source": per_source, "test_per_source": test_per_source,
        "per_source_counts": per_src_counts, "seed": seed, "waymo_stride": waymo_stride,
        "nuimages_version": nuimages_version,
        "splits": {k: {"images": v[0], "annotations": v[1]} for k, v in splits.items()},
        "leakage_free": "train (block A) and val/test_* (block B) are disjoint by index",
        "total_images": len(images), "total_annotations": len(annotations),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build_mixed] DONE -> {out_dir} | {len(images)} imgs "
          f"({len(train_imgs)} train / {len(test_imgs)} held-out test), "
          f"classes={taxonomy.classes}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Build a mixed KITTI+Waymo+nuImages dataset.")
    ap.add_argument("--out-dir", required=True, help="output dataset directory")
    ap.add_argument("--sources", nargs="+", default=["kitti", "waymo", "nuimages"])
    ap.add_argument("--per-source", type=int, default=400,
                    help="TRAIN images sampled from each source (block A)")
    ap.add_argument("--test-per-source", type=int, default=150,
                    help="held-out TEST images per source (block B, disjoint from train)")
    ap.add_argument("--taxonomy", default="driving3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--waymo-stride", type=int, default=40)
    ap.add_argument("--nuimages-version", default="v1.0-train")
    ap.add_argument("--roots", nargs="*", default=[], help="name=path overrides")
    a = ap.parse_args()

    roots = dict(DEFAULT_ROOTS)
    for kv in a.roots:
        k, v = kv.split("=", 1)
        roots[k] = v
    taxonomy = Taxonomy.from_preset(a.taxonomy)
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[build_mixed] sources={a.sources} per_source={a.per_source} "
          f"taxonomy={taxonomy.classes} -> {a.out_dir}")
    build_mixed(a.out_dir, a.sources, a.per_source, taxonomy, roots,
                test_per_source=a.test_per_source, seed=a.seed,
                waymo_stride=a.waymo_stride, nuimages_version=a.nuimages_version)


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# Build a mixed base (300 imgs each from KITTI/Waymo/nuImages):
#
#   python -m DeepDataMiningLearning.ngdet.mixed_dataset \
#       --out-dir DeepDataMiningLearning/ngdet/output/mixed \
#       --sources kitti waymo nuimages --per-source 300
#
# Then train on it (--dataset mixed) and evaluate:
#   python -m DeepDataMiningLearning.ngdet.train --trainer pytorch \
#       --backend torchvision --dataset mixed --root <out-dir>
#   python -m DeepDataMiningLearning.ngdet.run_eval --datasets mixed --roots mixed=<out-dir> ...
# ===========================================================================
if __name__ == "__main__":
    main()
