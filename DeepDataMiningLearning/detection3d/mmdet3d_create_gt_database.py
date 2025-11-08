#!/usr/bin/env python3
"""
Minimal-ish Ground-Truth Database creator for MMDetection3D datasets
===================================================================

This file is a **lightweight** rework of the official
`tools/dataset_converters/create_gt_database.py` that aims to **reduce
runtime dependencies** on `mmcv`/`mmengine` while keeping the behavior familiar.

Key differences vs. the official script
---------------------------------------
- **No mmcv or mmengine imports** in the hot path:
  - Uses Python's `logging` and `os.makedirs` instead of `mmengine.print_log`
    and `mmengine.mkdir_or_exist`.
  - Uses simple loops / `concurrent.futures` instead of
    `mmengine.track_*_progress` utilities.
- **No `mmcv.ops.roi_align`** / `mmdet.evaluation.bbox_overlaps` hard deps.
  - A tiny NumPy IoU is provided (`bbox_overlaps_np`) for optional mask logic.
  - Image patch saving (for `with_mask=True`) uses **Pillow** if present; if not,
    it is gracefully skipped.
- **Still depends on MMDetection3D** for dataset loading and box ops:
  - `mmdet3d.registry.DATASETS` (builds datasets/pipelines).
  - `mmdet3d.structures.ops.box_np_ops.points_in_rbbox`.

What it produces
----------------
- `<data_path>/<info_prefix>_gt_database/` containing per-object point crops
  (`.bin`, float32) centered at GT box.
- `<data_path>/<info_prefix>_dbinfos_train.pkl` mapping class names to lists of
  object metadata used by GT sampling augmentation.

Supported datasets
------------------
- `KittiDataset`, `NuScenesDataset`, `WaymoDataset` (KITTI-format conversion).

Usage (Python API)
------------------
Create serially:

    from mmdet3d_create_gt_database import create_groundtruth_database
    create_groundtruth_database(
        dataset_class_name="KittiDataset",
        data_path="/path/to/kitti",
        info_prefix="kitti",
        info_path="/path/to/kitti/kitti_infos_train.pkl",
        relative_path=False,
    )

Parallel creator (threaded):

    GTDatabaseCreater(
        "KittiDataset", "/path/to/kitti", "kitti",
        info_path="/path/to/kitti/kitti_infos_train.pkl",
        num_worker=8,
    ).create()

CLI (optional, defaults are reasonable):

    python mmdet3d_create_gt_database.py \
      --dataset KittiDataset \
      --data-path /path/to/kitti \
      --info-prefix kitti \
      --info-path /path/to/kitti/kitti_infos_train.pkl

Notes
-----
- `with_mask=True` requires COCO-style 2D mask annotations and optional
  `Pillow` for patch saving. If missing, mask-patch saving is skipped.
- This file intentionally avoids importing `mmcv`/`mmengine`.

License
-------
This derivation follows the spirit of the upstream OpenMMLab tools. Please
respect the original project license when redistributing.
"""
from __future__ import annotations

import os
import sys
import math
import pickle
import logging
from os import path as osp
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Core MMDet3D deps we still rely on
from mmdet3d.registry import DATASETS  # brings mmengine internally, but we don't import mmengine here
from mmdet3d.structures.ops import box_np_ops as box_np_ops

# --- bootstrap registry so Compose can find mmdet3d transforms ---
def _bootstrap_mmdet3d_registry():
    # 1) set default scope to 'mmdet3d'
    try:
        from mmengine.registry import init_default_scope
        init_default_scope('mmdet3d')
    except Exception:
        pass

    # 2) import transform modules so their @register() runs
    for mod in ('mmdet3d.datasets.transforms', 'mmdet3d.datasets.pipelines'):
        try:
            __import__(mod)
        except Exception as e:
            print(f'Optional import failed: {mod}: {e}')

_bootstrap_mmdet3d_registry()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("gt_db_min")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Small utility helpers (no mmcv/mmengine)
# -----------------------------------------------------------------------------

def mkdir_or_exist(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def bbox_overlaps_np(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """A tiny IoU implementation for [x1,y1,x2,y2] boxes (used for with_mask path).
    Returns an array with shape (len(bboxes1), len(bboxes2)).
    """
    if bboxes1.size == 0 or bboxes2.size == 0:
        return np.zeros((bboxes1.shape[0], bboxes2.shape[0]), dtype=np.float32)

    b1 = bboxes1.astype(np.float32)
    b2 = bboxes2.astype(np.float32)

    area1 = (b1[:, 2] - b1[:, 0]).clip(min=0) * (b1[:, 3] - b1[:, 1]).clip(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clip(min=0) * (b2[:, 3] - b2[:, 1]).clip(min=0)

    ious = np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float32)
    for i in range(b1.shape[0]):
        xx1 = np.maximum(b1[i, 0], b2[:, 0])
        yy1 = np.maximum(b1[i, 1], b2[:, 1])
        xx2 = np.minimum(b1[i, 2], b2[:, 2])
        yy2 = np.minimum(b1[i, 3], b2[:, 3])
        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        union = area1[i] + area2 - inter
        ious[i] = np.where(union > 0, inter / union, 0.0)
    return ious


def _poly2mask(segm: Any, h: int, w: int) -> np.ndarray:
    """Decode a COCO-style segmentation `segm` to a binary mask.
    Lazily imports pycocotools; returns (H,W) uint8 mask in {0,1}.
    """
    from pycocotools import mask as maskUtils  # lazy import

    if isinstance(segm, list):  # polygon
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(segm, dict) and "counts" in segm:
        rle = segm
    else:
        raise TypeError("Unsupported segmentation format")
    m = maskUtils.decode(rle)
    if m.ndim == 3:  # sometimes returns (H,W,1)
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def _parse_coco_ann_info(anns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse COCO annotations into simple dict with 'bboxes' and 'masks'."""
    bboxes, masks = [], []
    for a in anns:
        x, y, w, h = a.get("bbox", [0, 0, 0, 0])
        bboxes.append([x, y, x + max(0.0, w), y + max(0.0, h)])
        seg = a.get("segmentation", None)
        if seg is not None:
            masks.append(seg)
    if not bboxes:
        bboxes = np.zeros((0, 4), dtype=np.float32)
    else:
        bboxes = np.asarray(bboxes, dtype=np.float32)
    return {"bboxes": bboxes, "masks": masks}


def _maybe_save_patch(img_arr: np.ndarray, mask_arr: np.ndarray, out_png: str, out_mask_png: str) -> None:
    """Optionally save an image patch and its mask with Pillow if available.
    If Pillow is missing, this function becomes a no-op.
    """
    try:
        from PIL import Image
        Image.fromarray(img_arr).save(out_png)
        Image.fromarray((mask_arr.astype(np.uint8) * 255)).save(out_mask_png)
    except Exception as e:  # Pillow missing or invalid array
        logger.debug(f"Skip saving patch due to: {e}")


# -----------------------------------------------------------------------------
# Core functions
# -----------------------------------------------------------------------------

def create_groundtruth_database(
    dataset_class_name: str,
    data_path: str,
    info_prefix: str,
    info_path: Optional[str] = None,
    mask_anno_path: Optional[str] = None,
    used_classes: Optional[List[str]] = None,
    database_save_path: Optional[str] = None,
    db_info_save_path: Optional[str] = None,
    relative_path: bool = True,
    add_rgb: bool = False,      # kept for compatibility, not used
    lidar_only: bool = False,   # kept for compatibility, not used
    bev_only: bool = False,     # kept for compatibility, not used
    coors_range: Optional[List[float]] = None,  # kept for compatibility, not used
    with_mask: bool = False,
) -> None:
    """Build GT database and dbinfos pickle for a dataset split.

    Parameters largely mirror the original API for drop-in usage.
    """
    logger.info(f"Create GT Database of {dataset_class_name}")

    # Dataset config (small, dependency-light variants)
    dataset_cfg: Dict[str, Any] = dict(
        type=dataset_class_name,
        data_root=data_path,
        ann_file=info_path,
        test_mode=False,
    )

    if dataset_class_name == "KittiDataset":
        pts_dir = 'training/velodyne_reduced'
        if not osp.exists(osp.join(data_path, pts_dir)) and osp.exists(osp.join(data_path, 'training/velodyne')):
            pts_dir = 'training/velodyne'
    
        dataset_cfg.update(
            modality=dict(use_lidar=True, use_camera=with_mask),
            data_prefix=dict(pts=pts_dir, img="training/image_2"),
            pipeline=[
                dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
                dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
            ],
        )

    elif dataset_class_name == "NuScenesDataset":
        dataset_cfg.update(
            use_valid_flag=True,
            data_prefix=dict(pts="samples/LIDAR_TOP", img="", sweeps="sweeps/LIDAR_TOP"),
            pipeline=[
                dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
                dict(type="LoadPointsFromMultiSweeps", sweeps_num=10, use_dim=[0,1,2,3,4],
                     pad_empty_sweeps=True, remove_close=True),
                dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
            ],
        )

    elif dataset_class_name == "WaymoDataset":
        dataset_cfg.update(
            modality=dict(use_lidar=True, use_camera=False, use_lidar_intensity=True, use_depth=False),
            data_prefix=dict(pts="training/velodyne", img="", sweeps="training/velodyne"),
            pipeline=[
                dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=6, use_dim=6),
                dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
            ],
        )

    else:
        raise ValueError(f"Unsupported dataset_class_name: {dataset_class_name}")

    dataset = DATASETS.build(dataset_cfg)

    # Resolve paths
    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")

    mkdir_or_exist(database_save_path)

    # Optional COCO mask setup
    coco = None
    file2id: Dict[str, int] = {}
    if with_mask and mask_anno_path:
        try:
            from pycocotools.coco import COCO  # type: ignore
            coco = COCO(osp.join(data_path, mask_anno_path))
            for img_id in coco.getImgIds():
                info = coco.loadImgs([img_id])[0]
                file2id[info["file_name"]] = img_id
        except Exception as e:
            logger.warning(f"with_mask requested but COCO load failed: {e}")
            coco = None

    all_db_infos: Dict[str, List[Dict[str, Any]]] = {}

    # Iterate samples
    total = len(dataset)
    for j in range(total):
        if j % 200 == 0:
            logger.info(f"Processing sample {j}/{total}")

        data_info = dataset.get_data_info(j)
        example = dataset.pipeline(data_info)
        annos = example["ann_info"]
        image_idx = example.get("sample_idx", j)
        points = example["points"].numpy().astype(np.float32)
        gt_boxes_3d = annos["gt_bboxes_3d"].numpy().astype(np.float32)
        names = [dataset.metainfo["classes"][i] for i in annos["gt_labels_3d"]]

        # Fallbacks
        group_ids = annos.get("group_ids", np.arange(gt_boxes_3d.shape[0], dtype=np.int64))
        difficulty = annos.get("difficulty", np.zeros(gt_boxes_3d.shape[0], dtype=np.int32))

        # Points mask for each GT box
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        # Optional mask mapping (no image I/O here; just 2D matching)
        if with_mask and coco is not None and "gt_bboxes" in annos:
            gt_boxes_2d = annos["gt_bboxes"]  # shape (N,4) in [x1,y1,x2,y2]
            img_info = example.get("img_info") or {}
            img_file = osp.split(img_info.get("filename", ""))[-1]
            valid_inds = np.ones(gt_boxes_2d.shape[0], dtype=bool)
            mask_inds = np.zeros(gt_boxes_2d.shape[0], dtype=np.int64)
            if img_file and img_file in file2id:
                img_id = file2id[img_file]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                coco_anns = coco.loadAnns(ann_ids)
                parsed = _parse_coco_ann_info(coco_anns)
                iou = bbox_overlaps_np(parsed["bboxes"], gt_boxes_2d)
                mask_inds = iou.argmax(axis=0) if iou.size else mask_inds
                valid_inds = (iou.max(axis=0) > 0.5) if iou.size else valid_inds
            # else: keep defaults (all valid)
        else:
            valid_inds = np.ones(gt_boxes_3d.shape[0], dtype=bool)
            mask_inds = np.zeros(gt_boxes_3d.shape[0], dtype=np.int64)

        # Create per-object entries
        group_counter = 0
        group_map: Dict[int, int] = {}

        for i in range(gt_boxes_3d.shape[0]):
            gt_mask = point_indices[:, i]
            gt_points = points[gt_mask]
            if gt_points.size == 0:
                continue

            # center to box origin
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            filename = f"{image_idx}_{names[i]}_{i}.bin"
            abs_path = osp.join(database_save_path, filename)
            rel_path = osp.join(f"{info_prefix}_gt_database", filename) if relative_path else abs_path

            # save binary points
            with open(abs_path, "wb") as f:
                gt_points.tofile(f)

            if (used_classes is None) or (names[i] in used_classes):
                local_gid = int(group_ids[i]) if i < len(group_ids) else i
                if local_gid not in group_map:
                    group_map[local_gid] = group_counter
                    group_counter += 1

                db_rec: Dict[str, Any] = {
                    "name": names[i],
                    "path": rel_path,
                    "image_idx": int(image_idx),
                    "gt_idx": int(i),
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": int(gt_points.shape[0]),
                    "difficulty": int(difficulty[i]) if i < len(difficulty) else 0,
                    "group_id": int(group_map[local_gid]),
                }
                # Optional 2D info if present
                if with_mask and "gt_bboxes" in annos and i < len(annos["gt_bboxes"]):
                    db_rec["box2d_camera"] = annos["gt_bboxes"][i]
                    db_rec["mask_valid"] = bool(valid_inds[i])
                    db_rec["mask_match_index"] = int(mask_inds[i])

                all_db_infos.setdefault(names[i], []).append(db_rec)

    # Summaries & save
    for cls_name, recs in all_db_infos.items():
        logger.info(f"Collected {len(recs)} objects for class '{cls_name}'")

    logger.info(f"Saving dbinfos to {db_info_save_path}")
    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)


class GTDatabaseCreater:
    """Parallel (threaded) GT database creator.

    When `num_worker > 0`, uses a ThreadPool; else falls back to serial.
    Note: Python threads are often I/O-bound friendly (writing many small files).
    """

    def __init__(
        self,
        dataset_class_name: str,
        data_path: str,
        info_prefix: str,
        info_path: Optional[str] = None,
        mask_anno_path: Optional[str] = None,
        used_classes: Optional[List[str]] = None,
        database_save_path: Optional[str] = None,
        db_info_save_path: Optional[str] = None,
        relative_path: bool = True,
        add_rgb: bool = False,
        lidar_only: bool = False,
        bev_only: bool = False,
        coors_range: Optional[List[float]] = None,
        with_mask: bool = False,
        num_worker: int = 8,
    ) -> None:
        self.dataset_class_name = dataset_class_name
        self.data_path = data_path
        self.info_prefix = info_prefix
        self.info_path = info_path
        self.mask_anno_path = mask_anno_path
        self.used_classes = used_classes
        self.database_save_path = database_save_path or osp.join(data_path, f"{info_prefix}_gt_database")
        self.db_info_save_path = db_info_save_path or osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
        self.relative_path = relative_path
        self.add_rgb = add_rgb
        self.lidar_only = lidar_only
        self.bev_only = bev_only
        self.coors_range = coors_range
        self.with_mask = with_mask
        self.num_worker = max(0, int(num_worker))

        # Build dataset once
        dataset_cfg: Dict[str, Any] = dict(type=dataset_class_name, data_root=data_path, ann_file=info_path, test_mode=False)
        if dataset_class_name == "KittiDataset":
            
            pts_dir = 'training/velodyne_reduced'
            if not osp.exists(osp.join(data_path, pts_dir)) and osp.exists(osp.join(data_path, 'training/velodyne')):
                pts_dir = 'training/velodyne'
            dataset_cfg.update(
                modality=dict(use_lidar=True, use_camera=with_mask),
                data_prefix=dict(pts=pts_dir, img="training/image_2"),
                pipeline=[
                    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
                    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
                ],
            )
        elif dataset_class_name == "NuScenesDataset":
            dataset_cfg.update(
                use_valid_flag=True,
                data_prefix=dict(pts="samples/LIDAR_TOP", img="", sweeps="sweeps/LIDAR_TOP"),
                pipeline=[
                    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
                    dict(type="LoadPointsFromMultiSweeps", sweeps_num=10, use_dim=[0,1,2,3,4], pad_empty_sweeps=True, remove_close=True),
                    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
                ],
            )
        elif dataset_class_name == "WaymoDataset":
            dataset_cfg.update(
                modality=dict(use_lidar=True, use_camera=False, use_lidar_intensity=True, use_depth=False),
                data_prefix=dict(pts="training/velodyne", img="", sweeps="training/velodyne"),
                pipeline=[
                    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=6, use_dim=6),
                    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
                ],
            )
        else:
            raise ValueError(f"Unsupported dataset_class_name: {dataset_class_name}")

        self.dataset = DATASETS.build(dataset_cfg)
        self.pipeline = self.dataset.pipeline

        mkdir_or_exist(self.database_save_path)

        # Optional COCO for mask mapping
        self.coco = None
        self.file2id: Dict[str, int] = {}
        if with_mask and mask_anno_path:
            try:
                from pycocotools.coco import COCO  # lazy
                self.coco = COCO(osp.join(self.data_path, self.mask_anno_path))
                for img_id in self.coco.getImgIds():
                    info = self.coco.loadImgs([img_id])[0]
                    self.file2id[info["file_name"]] = img_id
            except Exception as e:
                logger.warning(f"with_mask requested but COCO load failed: {e}")
                self.coco = None

    # Per-sample worker
    def _create_single(self, idx: int) -> Dict[str, List[Dict[str, Any]]]:
        single_db_infos: Dict[str, List[Dict[str, Any]]] = {}
        input_dict = self.dataset.get_data_info(idx)
        example = self.pipeline(input_dict)
        annos = example["ann_info"]
        image_idx = example.get("sample_idx", idx)
        points = example["points"].numpy().astype(np.float32)
        gt_boxes_3d = annos["gt_bboxes_3d"].numpy().astype(np.float32)
        names = [self.dataset.metainfo["classes"][i] for i in annos["gt_labels_3d"]]

        group_ids = annos.get("group_ids", np.arange(gt_boxes_3d.shape[0], dtype=np.int64))
        difficulty = annos.get("difficulty", np.zeros(gt_boxes_3d.shape[0], dtype=np.int32))

        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        # Optional 2D matching
        valid_inds = np.ones(gt_boxes_3d.shape[0], dtype=bool)
        mask_inds = np.zeros(gt_boxes_3d.shape[0], dtype=np.int64)
        if self.with_mask and self.coco is not None and "gt_bboxes" in annos:
            gt_boxes_2d = annos["gt_bboxes"]
            img_info = example.get("img_info") or {}
            img_file = osp.split(img_info.get("filename", ""))[-1]
            if img_file and img_file in self.file2id:
                img_id = self.file2id[img_file]
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                coco_anns = self.coco.loadAnns(ann_ids)
                parsed = _parse_coco_ann_info(coco_anns)
                iou = bbox_overlaps_np(parsed["bboxes"], gt_boxes_2d)
                if iou.size:
                    mask_inds = iou.argmax(axis=0)
                    valid_inds = (iou.max(axis=0) > 0.5)

        # Group-id mapping
        group_counter = 0
        group_map: Dict[int, int] = {}

        # Build per-object entries
        for i in range(gt_boxes_3d.shape[0]):
            gt_mask = point_indices[:, i]
            gt_points = points[gt_mask]
            if gt_points.size == 0:
                continue
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            filename = f"{image_idx}_{names[i]}_{i}.bin"
            abs_path = osp.join(self.database_save_path, filename)
            rel_path = osp.join(f"{self.info_prefix}_gt_database", filename) if self.relative_path else abs_path

            with open(abs_path, "wb") as f:
                gt_points.tofile(f)

            if (self.used_classes is None) or (names[i] in self.used_classes):
                local_gid = int(group_ids[i]) if i < len(group_ids) else i
                if local_gid not in group_map:
                    group_map[local_gid] = group_counter
                    group_counter += 1

                db_rec: Dict[str, Any] = {
                    "name": names[i],
                    "path": rel_path,
                    "image_idx": int(image_idx),
                    "gt_idx": int(i),
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": int(gt_points.shape[0]),
                    "difficulty": int(difficulty[i]) if i < len(difficulty) else 0,
                    "group_id": int(group_map[local_gid]),
                }
                if self.with_mask and "gt_bboxes" in annos and i < len(annos["gt_bboxes"]):
                    db_rec["box2d_camera"] = annos["gt_bboxes"][i]
                    db_rec["mask_valid"] = bool(valid_inds[i])
                    db_rec["mask_match_index"] = int(mask_inds[i])

                single_db_infos.setdefault(names[i], []).append(db_rec)

        return single_db_infos

    def create(self) -> None:
        logger.info(f"Create GT Database of {self.dataset_class_name}")
        total = len(self.dataset)

        all_db_infos: Dict[str, List[Dict[str, Any]]] = {}
        mkdir_or_exist(self.database_save_path)

        if self.num_worker <= 0:
            for i in range(total):
                if i % 200 == 0:
                    logger.info(f"Processing sample {i}/{total}")
                res = self._create_single(i)
                for k, v in res.items():
                    all_db_infos.setdefault(k, []).extend(v)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=self.num_worker) as ex:
                futures = {ex.submit(self._create_single, i): i for i in range(total)}
                for n, fut in enumerate(as_completed(futures)):
                    if n % 50 == 0:
                        logger.info(f"Finished {n}/{total} samples")
                    res = fut.result()
                    for k, v in res.items():
                        all_db_infos.setdefault(k, []).extend(v)

        # Normalize group ids globally
        logger.info("Make global unique group id")
        group_counter_offset = 0
        normalized: Dict[str, List[Dict[str, Any]]] = {}
        for name, items in all_db_infos.items():
            max_gid = -1
            for rec in items:
                gid = rec.get("group_id", -1)
                if gid > max_gid:
                    max_gid = gid
                rec["group_id"] = gid + group_counter_offset if gid >= 0 else gid
            group_counter_offset += (max_gid + 1)
            normalized[name] = items

        for k, v in normalized.items():
            logger.info(f"load {len(v)} {k} database infos")

        logger.info(f"Saving GT database infos into {self.db_info_save_path}")
        with open(self.db_info_save_path, "wb") as f:
            pickle.dump(normalized, f)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None):
    import argparse
    p = argparse.ArgumentParser("Minimal GT-DB creator (no mmcv/mmengine runtime deps)")
    p.add_argument("--dataset", default="KittiDataset", choices=["KittiDataset", "NuScenesDataset", "WaymoDataset"])
    p.add_argument("--data-path", required=True, help="Dataset root path")
    p.add_argument("--info-prefix", default="kitti")
    p.add_argument("--info-path", required=True, help="Path to *_infos_train.pkl (or split pkl)")
    p.add_argument("--db-save-path", default=None, help="Override GT-DB dir path (default: <data>/<prefix>_gt_database)")
    p.add_argument("--dbinfo-save-path", default=None, help="Override dbinfos pkl path (default: <data>/<prefix>_dbinfos_train.pkl)")
    p.add_argument("--relative-path", action="store_true", help="Store relative file paths in dbinfos")
    p.add_argument("--with-mask", action="store_true", help="Enable 2D mask metadata (requires COCO json)")
    p.add_argument("--mask-anno-path", default=None, help="COCO json path relative to data root")
    p.add_argument("--num-worker", type=int, default=0, help="Thread workers (0 = serial)")
    return p.parse_args(argv)


def _main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    if args.num_worker <= 0:
        create_groundtruth_database(
            dataset_class_name=args.dataset,
            data_path=args.data_path,
            info_prefix=args.info_prefix,
            info_path=args.info_path,
            database_save_path=args.db_save_path,
            db_info_save_path=args.dbinfo_save_path,
            relative_path=args.relative_path,
            with_mask=args.with_mask,
            mask_anno_path=args.mask_anno_path,
        )
    else:
        GTDatabaseCreater(
            dataset_class_name=args.dataset,
            data_path=args.data_path,
            info_prefix=args.info_prefix,
            info_path=args.info_path,
            database_save_path=args.db_save_path,
            db_info_save_path=args.dbinfo_save_path,
            relative_path=args.relative_path,
            with_mask=args.with_mask,
            mask_anno_path=args.mask_anno_path,
            num_worker=args.num_worker,
        ).create()


if __name__ == "__main__":
    _main()


# python mmdet3d_create_gt_database.py \
#   --dataset KittiDataset \
#   --data-path /path/to/kitti \
#   --info-prefix kitti \
#   --info-path /path/to/kitti/kitti_infos_train.pkl \
#   --relative-path