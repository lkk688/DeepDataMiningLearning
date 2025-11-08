#!/usr/bin/env python3
"""
create_data_resumable.py (universal, nuScenes + KITTI + Waymo 1.4.3)

A **resume-aware**, **anywhere-runnable** data preparation helper for
MMDetection3D. It supports **nuScenes**, **KITTI**, and **Waymo v1.4.3** and can
be executed from *any* directory by providing the mmdetection3d repository path.

Main goals
----------
1) Validate dataset roots and common pitfalls (case sensitivity, required
   subfolders, version folders).
2) Detect which artifacts already exist (info PKLs, updated PKLs, reduced point
   clouds for KITTI, GT databases) and **only run missing steps**.
3) Provide **clear logging** and **idempotent** operations. Re-running the same
   step is safe.

What gets produced (high-level)
-------------------------------
nuScenes
  • `<out_dir>/<prefix>_infos_{train,val}.pkl` (or `{test}.pkl`)
    - **Format**: Python pickle dict; contains sample-level metadata such as
      file paths, calibrations, annotations (train/val), timestamps, etc.
    - **Purpose**: Consumed by mmdet3d `Dataset` classes during training/inference
      to avoid re-parsing the raw devkit tables repeatedly.
  • (After update) PKLs are normalized to the latest **v2 schema**.
  • `<out_dir>/gt_database/` + `<prefix>_dbinfos_train.pkl`
    - **Format**: A folder of per-object point-cloud snippets and a pickle index.
    - **Purpose**: Used by detectors (PointPillars/SECOND/CenterPoint, etc.) to
      perform **class-balanced sampling** and **Object-GT augmentation**.

KITTI
  • `<out_dir>/<prefix>_infos_{train,val,trainval,test}.pkl`
    - **Format/Purpose**: Same idea as nuScenes; lists per-sample LiDAR/image
      files, calibration, labels (where available), etc.
  • `training/velodyne_reduced/` (and optionally `testing/velodyne_reduced/`)
    - **Format**: Binary point clouds (`.bin`) cropped to the camera FOV.
    - **Purpose**: Speed up certain pipelines that rely on reduced point clouds.
  • `<out_dir>/gt_database/` + `<prefix>_dbinfos_train.pkl`
    - **Purpose**: Same as nuScenes.

Waymo v1.4.3 (converted to KITTI-format)
  • `kitti_format/{training,testing}/...` directories created from Waymo TFRecords.
    - **Format**: Images, LiDAR `.bin`, KITTI-style labels/calibs; plus ImageSets.
  • `<out_dir>/kitti_format/<prefix>_infos_{train,val,test}.pkl`
    - **Purpose**: mmdet3d fast-loading metadata.
  • `<out_dir>/kitti_format/gt_database/` + `<prefix>_dbinfos_train.pkl`
    - **Purpose**: GT DB for augmentation.

Usage examples (run from *any* directory)
-----------------------------------------
nuScenes
    python create_data_resumable.py nuscenes \
        --repo-root /path/to/mmdetection3d \
        --root-path  /data/nuscenes \
        --out-dir    /data/nuscenes \
        --info-prefix nuscenes \
        --version v1.0-trainval \
        --max-sweeps 10 \
        --fix-case-symlink

KITTI
    python create_data_resumable.py kitti \
        --repo-root /path/to/mmdetection3d \
        --root-path  /data/kitti \
        --out-dir    /data/kitti \
        --info-prefix kitti \
        --with-plane

Waymo v1.4.3
    python create_data_resumable.py waymo \
        --repo-root /path/to/mmdetection3d \
        --root-path  /data/waymo \
        --out-dir    /data/waymo \
        --info-prefix waymo \
        --waymo-version v1.4.3 \
        --workers 8 --max-sweeps 10 \
        --save-sensor-data \
        --skip-cam-instances-infos   # omit camera instance infos if you want speed

Common flags
------------
--dry-run            Print what would run; do not execute.
--force              Rebuild/overwrite even if outputs already exist.
--repo-root          Path to **mmdetection3d** repository (so you can run from anywhere).
--fix-case-symlink   For nuScenes: if basename is not `nuscenes`, create lowercase alias.

"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
import time
import pickle
from typing import Dict, Any, Tuple, List

# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def _yes(x: bool) -> str:
    return "YES" if x else "NO"


# ---------------------------------------------------------------------------
# Repo import plumbing — run from *any* directory by injecting repo into sys.path
# ---------------------------------------------------------------------------

def add_repo_to_syspath(repo_root: str) -> None:
    """Ensure we can import `tools.*` modules by adding the repo root to sys.path.

    We call this **before** importing any mmdetection3d helpers.

    Modules imported from the repo (used by this script)
    ---------------------------------------------------
    • tools.dataset_converters.nuscenes_converter
    • tools.dataset_converters.kitti_converter
    • tools.dataset_converters.waymo_converter
       - waymo_converter.Waymo2KITTI
       - waymo_converter.GTDatabaseCreater
       - waymo_converter.create_ImageSets_img_ids
    • tools.dataset_converters.update_infos_to_v2.update_pkl_infos
    • tools.create_data.create_groundtruth_database (for nuScenes/KITTI GT DB)

    External runtime dependencies (install via pip/conda)
    -----------------------------------------------------
    Core:   mmengine, mmcv, mmdet, mmdet3d, numpy, pillow, pyquaternion, tqdm
    KITTI:  (none beyond core)
    nuScenes: (none beyond core; official devkit is a transitive dep of mmdet3d)
    Waymo 1.4.3:
      • TensorFlow 2.x (use a wheel matching your CUDA/CPU; e.g., tf>=2.11)
      • waymo-open-dataset 1.4.3 wheel matching your TensorFlow build,
        typically named like: `waymo-open-dataset-tf-2-11-0==1.4.3`
      • protobuf (the wheel pins this; ensure you don't override it)

    Tip: keep these binary packages from a **single source** (all conda-forge, or
    all pip) to avoid ABI clashes.
    """
    repo_root = osp.abspath(osp.expanduser(repo_root))
    if not osp.isdir(repo_root):
        raise FileNotFoundError(f"Repo root not found: {repo_root}")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    _log(f"[env] Using repo root: {repo_root}")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def exists_file(p: str) -> bool:
    try:
        return osp.isfile(p)
    except Exception:
        return False


def ensure_lowercase_alias(root_path: str, expect: str = "nuscenes", fix_symlink: bool = False) -> str:
    """For nuScenes, optionally create a lowercase alias beside an incorrectly
    cased directory (e.g., `/data/nuScenes` -> `/data/nuscenes`).
    """
    root_abs = osp.abspath(root_path)
    base = osp.basename(root_abs)
    if base == expect:
        return root_abs  # already canonical
    parent = osp.dirname(root_abs)
    canonical = osp.join(parent, expect)
    if base.lower() == expect and base != expect:
        if fix_symlink:
            _log(f"[case-fix] Creating lowercase alias: {canonical} -> {root_abs}")
            try:
                if osp.lexists(canonical):
                    os.unlink(canonical)
            except FileNotFoundError:
                pass
            os.symlink(root_abs, canonical)
            return canonical
        else:
            _log("[case-fix] Detected non-canonical basename. Re-run with --fix-case-symlink,\n"
            f"           or rename '{base}' to '{expect}'. Using given path as-is.")
    return root_abs


def try_peek_infos_head(pkl_path: str) -> Dict[str, Any]:
    """Lightweight look into an info PKL to display top-level keys.

    This is **not** used to decide whether to skip; updates are idempotent.
    """
    info: Dict[str, Any] = {}
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            info["keys"] = list(obj.keys())[:8]
    except Exception as e:
        info["error"] = repr(e)
    return info


def check_gt_database(out_dir: str, info_prefix: str) -> Tuple[bool, Dict[str, bool]]:
    """Detect whether the GT DB and its index pickle exist."""
    db_dir = osp.join(out_dir, "gt_database")
    dbinfo = osp.join(out_dir, f"{info_prefix}_dbinfos_train.pkl")
    flags = {
        "gt_database_dir": osp.isdir(db_dir) if hasattr(osp, "isdir") else osp.exists(db_dir),
        "dbinfos_pickle": exists_file(dbinfo),
    }
    ok = flags["gt_database_dir"] and flags["dbinfos_pickle"]
    return ok, flags


# ---------------------------------------------------------------------------
# nuScenes-specific checks and steps
# ---------------------------------------------------------------------------

def check_nuscenes_layout(root_path: str, version: str) -> Tuple[bool, Dict[str, bool]]:
    """Validate that `maps/`, `samples/`, `sweeps/`, and `<version>/` exist."""
    req = ["maps", "samples", "sweeps", version]
    flags = {name: osp.exists(osp.join(root_path, name)) for name in req}
    ok = all(flags.values())
    return ok, flags


def step_nusc_generate_infos(nuscenes_converter, root_path: str, info_prefix: str,
                             version: str, max_sweeps: int, out_dir: str,
                             dry_run: bool, force: bool) -> None:
    """Create `<prefix>_infos_{train,val}.pkl` (or `{test}.pkl`).

    • **Input**: nuScenes devkit tables under `root_path` (JSONs + samples/sweeps).
    • **Output**: PKLs in `out_dir` with per-sample metadata.
    • **Use**: Fast dataset loading for training/inference.
    """
    p_train = osp.join(out_dir, f"{info_prefix}_infos_train.pkl")
    p_val   = osp.join(out_dir, f"{info_prefix}_infos_val.pkl")
    if not force and exists_file(p_train) and exists_file(p_val):
        _log("[nusc/infos] Train & val PKLs exist — skipping (use --force to rebuild).")
        return
    if dry_run:
        _log("[dry-run] Would run nuscenes_converter.create_nuscenes_infos(...)")
        return
    _log("[nusc/infos] Generating nuScenes info PKLs (this may take a while)...")
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps
    )
    _log("[nusc/infos] Done.")


def step_nusc_update_infos(update_pkl_infos, out_dir: str, info_prefix: str,
                           version: str, dry_run: bool) -> None:
    """Normalize PKLs to v2 schema (safe to re-run)."""
    if version == "v1.0-test":
        p = osp.join(out_dir, f"{info_prefix}_infos_test.pkl")
        _log(f"[nusc/update] target: {p}")
        if dry_run:
            _log("[dry-run] Would update test PKL to v2 schema.")
            return
        update_pkl_infos("nuscenes", out_dir=out_dir, pkl_path=p)
        _log("[nusc/update] Test PKL updated.")
        return

    p_train = osp.join(out_dir, f"{info_prefix}_infos_train.pkl")
    p_val   = osp.join(out_dir, f"{info_prefix}_infos_val.pkl")
    _log(f"[nusc/update] train keys: {try_peek_infos_head(p_train)}")
    _log(f"[nusc/update] val   keys: {try_peek_infos_head(p_val)}")
    if dry_run:
        _log("[dry-run] Would update train/val PKLs to v2 schema.")
        return
    update_pkl_infos("nuscenes", out_dir=out_dir, pkl_path=p_train)
    update_pkl_infos("nuscenes", out_dir=out_dir, pkl_path=p_val)
    _log("[nusc/update] Done.")


def step_nusc_create_gtdb(dataset_name: str, root_path: str,
                          info_prefix: str, out_dir: str, info_name: str,
                          dry_run: bool, force: bool) -> None:
    """Build nuScenes GT-DB using our minimal creator."""
    ok, flags = check_gt_database(out_dir, info_prefix)
    _log(f"[nusc/gtdb] present? dir={_yes(flags['gt_database_dir'])} "
         f"dbinfo={_yes(flags['dbinfos_pickle'])}")
    if ok and not force:
        _log("[nusc/gtdb] Found existing GT DB — skipping (use --force to rebuild).")
        return
    if dry_run:
        _log("[dry-run] Would build nuScenes GT DB from info: " + info_name)
        return

    info_path = osp.join(out_dir, info_name)  # e.g. nuscenes_infos_train.pkl
    _log(f"[nusc/gtdb] Building GT DB from {info_path}")
    create_gt_db(
        dataset_class_name=dataset_name,         # "NuScenesDataset"
        data_path=root_path,                     # nuScenes root (has samples/sweeps/...)
        info_prefix=info_prefix,                 # "nuscenes"
        info_path=info_path,                     # PKL path
        relative_path=True,
        with_mask=False,                         # nuScenes 默认无 2D mask
    )
    _log("[nusc/gtdb] Done.")

# ---------------------------------------------------------------------------
# KITTI-specific checks and steps
# ---------------------------------------------------------------------------

def check_kitti_layout(root_path: str) -> Tuple[bool, Dict[str, bool]]:
    """Basic KITTI 3D detection layout validation.

    Expected structure under `root_path`:
      training/{image_2, calib, velodyne, label_2}
      testing/{image_2, calib, velodyne}
      ImageSets/{train.txt, val.txt, trainval.txt, test.txt}
    """
    req = {
        "training": osp.isdir(osp.join(root_path, "training")),
        "testing": osp.isdir(osp.join(root_path, "testing")),
        "ImageSets": osp.isdir(osp.join(root_path, "ImageSets")),
        # sub-folders in training
        "tr_image_2": osp.isdir(osp.join(root_path, "training", "image_2")),
        "tr_calib":   osp.isdir(osp.join(root_path, "training", "calib")),
        "tr_velo":    osp.isdir(osp.join(root_path, "training", "velodyne")),
        "tr_label_2": osp.isdir(osp.join(root_path, "training", "label_2")),
        # sub-folders in testing (labels not provided)
        "te_image_2": osp.isdir(osp.join(root_path, "testing", "image_2")),
        "te_calib":   osp.isdir(osp.join(root_path, "testing", "calib")),
        "te_velo":    osp.isdir(osp.join(root_path, "testing", "velodyne")),
    }
    ok = all(req.values())
    return ok, req


def step_kitti_generate_infos(kitti_converter, root_path: str, info_prefix: str,
                              out_dir: str, with_plane: bool, dry_run: bool,
                              force: bool) -> None:
    """Create KITTI info PKLs using mmdet3d's converter.

    • **Outputs**: `<prefix>_infos_{train,val,trainval,test}.pkl` in `out_dir`.
    • **Use**: Fast dataset loading.
    • **with_plane**: If plane files are present, include them to support some
      height-normalization tricks used by classic detectors.
    • **Also**: Build `velodyne_reduced/` folders with FOV-cropped point clouds.
    """
    targets = [
        f"{info_prefix}_infos_train.pkl",
        f"{info_prefix}_infos_val.pkl",
        f"{info_prefix}_infos_trainval.pkl",
        f"{info_prefix}_infos_test.pkl",
    ]
    have_all = all(exists_file(osp.join(out_dir, t)) for t in targets)
    _log(f"[kitti/infos] all PKLs present? {_yes(have_all)}")
    if have_all and not force:
        _log("[kitti/infos] Skipping generation (use --force to rebuild).")
        return

    if dry_run:
        _log("[dry-run] Would run kitti.create_kitti_info_file(...) and create_reduced_point_cloud(...)")
        return

    _log("[kitti/infos] Creating info PKLs...")
    kitti_converter.create_kitti_info_file(root_path, info_prefix, with_plane)
    _log("[kitti/infos] Creating reduced point clouds (cropped to image FOV)...")
    kitti_converter.create_reduced_point_cloud(root_path, info_prefix)
    _log("[kitti/infos] Done.")


import os, pickle as pkl

def _is_v2_infos(pkl_path: str) -> bool:
    try:
        with open(pkl_path, "rb") as f:
            obj = pkl.load(f)
        # v1: list[...]；v2: dict{ 'data_list': [...], 'metainfo': {...}, ... }
        return isinstance(obj, dict) and "data_list" in obj
    except Exception:
        return False

def step_kitti_update_infos(update_pkl_infos, out_dir, info_prefix, dry_run):
    for split in ("train", "val", "trainval", "test"):
        path = os.path.join(out_dir, f"{info_prefix}_infos_{split}.pkl")
        if not os.path.exists(path):
            continue
        if _is_v2_infos(path):
            _log(f"[kitti/update] skip {split}: already v2 schema")
            continue
        if dry_run:
            _log(f"[dry-run] would update {path}")
        else:
            _log(f"[kitti/update] updating {path}")
            update_pkl_infos("kitti", out_dir=out_dir, pkl_path=path)


def step_kitti_create_gtdb(root_path: str, info_prefix: str, out_dir: str,
                           version: str, dry_run: bool, force: bool) -> None:
    """Build KITTI GT-DB using our minimal creator."""
    ok, flags = check_gt_database(out_dir, info_prefix)
    _log(f"[kitti/gtdb] present? dir={_yes(flags['gt_database_dir'])} "
         f"dbinfo={_yes(flags['dbinfos_pickle'])}")
    if ok and not force:
        _log("[kitti/gtdb] Found existing GT DB — skipping (use --force to rebuild).")
        return
    if dry_run:
        _log("[dry-run] Would build KITTI GT DB.")
        return

    info_path = osp.join(out_dir, f"{info_prefix}_infos_train.pkl")
    _log(f"[kitti/gtdb] Building GT DB from {info_path}")
    create_gt_db(
        dataset_class_name="KittiDataset",
        data_path=root_path,                     # KITTI 根目录（有 training/testing/...）
        info_prefix=info_prefix,                 # "kitti"
        info_path=info_path,                     # kitti_infos_train.pkl
        relative_path=False,
        mask_anno_path="instances_train.json",
        with_mask=(version == "mask"),
    )
    _log("[kitti/gtdb] Done.")


# ---------------------------------------------------------------------------
# Waymo v1.4.3-specific checks and steps (conversion to KITTI-format)
# ---------------------------------------------------------------------------

def check_waymo_layout(root_path: str) -> Tuple[bool, Dict[str, bool]]:
    """Expect `waymo_format/{training,validation,testing}/` to exist under root.

    Each split should contain the raw `*.tfrecord` files downloaded from Waymo.
    """
    req = {
        "waymo_format": osp.isdir(osp.join(root_path, "waymo_format")),
        "training": osp.isdir(osp.join(root_path, "waymo_format", "training")),
        "validation": osp.isdir(osp.join(root_path, "waymo_format", "validation")),
        "testing": osp.isdir(osp.join(root_path, "waymo_format", "testing")),
    }
    ok = req["waymo_format"] and (req["training"] or req["validation"] or req["testing"])
    return ok, req


def waymo_version_to_splits(version: str) -> List[str]:
    """Map Waymo version label to the set of splits we will process.

    v1.4.3 and v1.4 share the same directory conventions; "mini" only uses
    training/validation for quick smoke-tests.
    """
    if version in ("v1.4.3", "v1.4"):
        return ["training", "validation", "testing", "testing_3d_camera_only_detection"]
    if version == "v1.4-mini":
        return ["training", "validation"]
    raise NotImplementedError(f"Unsupported Waymo version: {version}")


def step_waymo_convert_and_infos(waymo_converter, root_path: str, out_dir: str, info_prefix: str,
                                 splits: List[str], workers: int, max_sweeps: int,
                                 save_sensor_data: bool, skip_cam_instances_infos: bool,
                                 dry_run: bool, force: bool) -> None:
    """Steps 1 & 2: Convert Waymo TFRecords to KITTI-format + generate info PKLs.

    • **Input**: `root_path/waymo_format/<split>/*.tfrecord`
    • **Output**: `out_dir/kitti_format/{training,testing}/...` KITTI-style structure
                   and `<prefix>_infos_{train,val,test}.pkl` under `kitti_format/`.
    • **Notes**:
       - Validation is merged into training infos for trainval usage.
       - `save_sensor_data` follows the historical misspelling used by the converter
         as `save_senor_data` (we pass through accordingly).
    """
    kitti_out = osp.join(out_dir, "kitti_format")
    os.makedirs(kitti_out, exist_ok=True)

    # Check if all final info PKLs already exist
    targets = [osp.join(kitti_out, f"{info_prefix}_infos_{p}.pkl") for p in ("train", "val", "test")]
    have_all = all(exists_file(t) for t in targets)
    _log(f"[waymo/infos] all core PKLs present? {_yes(have_all)}")
    if have_all and not force:
        _log("[waymo/infos] Skipping conversion+infos (use --force to rebuild).")
        return

    if dry_run:
        _log("[dry-run] Would convert Waymo TFRecords to KITTI format and build infos.")
        return

    for i, split in enumerate(splits):
        load_dir = osp.join(root_path, "waymo_format", split)
        # Save validation into training folder to form a combined trainval set
        save_dir = osp.join(kitti_out, "training" if split == "validation" else split)
        os.makedirs(save_dir, exist_ok=True)

        converter = waymo_converter.Waymo2KITTI(
            load_dir,
            save_dir,
            prefix=str(i),
            workers=workers,
            test_mode=(split in ["testing", "testing_3d_camera_only_detection"]),
            info_prefix=info_prefix,
            max_sweeps=max_sweeps,
            split=split,
            # Historical API uses misspelled arg name `save_senor_data`
            save_senor_data=save_sensor_data,
            save_cam_instances=not skip_cam_instances_infos,
        )
        _log(f"[waymo/convert] Converting split={split} (threads={workers}) -> {save_dir}")
        converter.convert()
        if split == "validation":
            _log("[waymo/convert] Merging train+val infos into trainval...")
            converter.merge_trainval_infos()

    # ImageSets files with image ids for all splits
    waymo_converter.create_ImageSets_img_ids(kitti_out, splits)
    _log("[waymo/infos] Conversion + infos complete.")


def step_waymo_create_gtdb(out_dir: str, info_prefix: str,
                           workers: int, dry_run: bool, force: bool) -> None:
    """Build Waymo GT-DB on the converted KITTI-format tree."""
    kitti_out = osp.join(out_dir, "kitti_format")
    ok, flags = check_gt_database(kitti_out, info_prefix)
    _log(f"[waymo/gtdb] present? dir={_yes(flags['gt_database_dir'])} "
         f"dbinfo={_yes(flags['dbinfos_pickle'])}")
    if ok and not force:
        _log("[waymo/gtdb] GT DB exists — skipping (use --force to rebuild).")
        return
    if dry_run:
        _log("[dry-run] Would build Waymo GT DB.")
        return

    info_path = osp.join(kitti_out, f"{info_prefix}_infos_train.pkl")
    _log(f"[waymo/gtdb] Building GT DB from {info_path}")
    if workers and workers > 0:
        GTDBCreater(
            dataset_class_name="WaymoDataset",
            data_path=kitti_out,                 # 指向 kitti_format 根
            info_prefix=info_prefix,
            info_path=info_path,
            relative_path=False,
            with_mask=False,
            num_worker=workers,                  # 线程模式
        ).create()
    else:
        create_gt_db(
            dataset_class_name="WaymoDataset",
            data_path=kitti_out,
            info_prefix=info_prefix,
            info_path=info_path,
            relative_path=False,
            with_mask=False,
        )
    _log("[waymo/gtdb] Done.")

# ---------------------------------------------------------------------------
# Orchestration per dataset
# ---------------------------------------------------------------------------
from mmdet3d_create_gt_database import (
    create_groundtruth_database as create_gt_db,
    GTDatabaseCreater as GTDBCreater,
)

def run_nuscenes(repo_root: str, root_path: str, out_dir: str, info_prefix: str, version: str,
                 max_sweeps: int, dataset_name: str, fix_case_symlink: bool,
                 dry_run: bool, force: bool) -> None:
    # Inject repo and import lazily
    add_repo_to_syspath(repo_root)
    from tools.dataset_converters import nuscenes_converter  # type: ignore
    from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos  # type: ignore
    #from tools.create_data import create_groundtruth_database  # type: ignore

    root_path = ensure_lowercase_alias(root_path, expect="nuscenes", fix_symlink=fix_case_symlink)
    ok, flags = check_nuscenes_layout(root_path, version)
    _log(f"[nusc/check] layout ok? {_yes(ok)} | maps={_yes(flags['maps'])} "
         f"samples={_yes(flags['samples'])} sweeps={_yes(flags['sweeps'])} {version}={_yes(flags[version])}")
    if not ok and not dry_run:
        raise AssertionError(
            f"nuScenes root looks incomplete at '{root_path}'. Expected: maps/, samples/, sweeps/, {version}/")

    step_nusc_generate_infos(nuscenes_converter, root_path, info_prefix, version, max_sweeps, out_dir, dry_run, force)
    step_nusc_update_infos(update_pkl_infos, out_dir, info_prefix, version, dry_run)
    #info_name = f"{info_prefix}_infos_train.pkl" if version != "v1.0-test" else f"{info_prefix}_infos_test.pkl"
    #step_nusc_create_gtdb(create_groundtruth_database, dataset_name, root_path, info_prefix, out_dir, info_name, dry_run, force)
    info_name = f"{info_prefix}_infos_train.pkl" if version != "v1.0-test" else f"{info_prefix}_infos_test.pkl"
    step_nusc_create_gtdb(dataset_name, root_path, info_prefix, out_dir, info_name, dry_run, force)


    ok_db, db_flags = check_gt_database(out_dir, info_prefix)
    _log("[nusc/summary] info_prefix = %s | gt_db_dir=%s dbinfos=%s" % (
        info_prefix, _yes(db_flags['gt_database_dir']), _yes(db_flags['dbinfos_pickle'])))


def run_kitti(repo_root: str, root_path: str, out_dir: str, info_prefix: str,
              version: str, with_plane: bool, dry_run: bool, force: bool) -> None:
    add_repo_to_syspath(repo_root)
    from tools.dataset_converters import kitti_converter  # type: ignore
    from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos  # type: ignore
    #from tools.create_data import create_groundtruth_database  # type: ignore

    ok, req = check_kitti_layout(root_path)
    _log(f"[kitti/check] layout ok? {_yes(ok)} | req={req}")
    if not ok and not dry_run:
        raise AssertionError(
            "KITTI root looks incomplete at '%s'. Expected training/testing/ImageSets subfolders." % root_path)

    step_kitti_generate_infos(kitti_converter, root_path, info_prefix, out_dir, with_plane, dry_run, force) #create data/kitti/kitti_infos_train.pkl 
    step_kitti_update_infos(update_pkl_infos, out_dir, info_prefix, dry_run)
    #step_kitti_create_gtdb(create_groundtruth_database, root_path, info_prefix, out_dir, version, dry_run, force)

    step_kitti_create_gtdb(root_path, info_prefix, out_dir, version, dry_run, force) #Saving dbinfos to /home/lkk688/Developer/mmdetection3d/data/kitti/kitti_dbinfos_train.pkl
    ok_db, db_flags = check_gt_database(out_dir, info_prefix)
    _log("[kitti/summary] info_prefix = %s | gt_db_dir=%s dbinfos=%s" % (
        info_prefix, _yes(db_flags['gt_database_dir']), _yes(db_flags['dbinfos_pickle'])))


def run_waymo(repo_root: str, root_path: str, out_dir: str, info_prefix: str,
              waymo_version: str, workers: int, max_sweeps: int,
              only_gt_database: bool, save_sensor_data: bool,
              skip_cam_instances_infos: bool, dry_run: bool, force: bool) -> None:
    add_repo_to_syspath(repo_root)
    from tools.dataset_converters import waymo_converter  # type: ignore

    ok, req = check_waymo_layout(root_path)
    _log(f"[waymo/check] layout ok? {_yes(ok)} | req={req}")
    if not ok and not dry_run:
        raise AssertionError(
            f"Waymo root looks incomplete at '{root_path}'. Expected waymo_format/<split> folders.")

    splits = waymo_version_to_splits(waymo_version)
    kitti_out = osp.join(out_dir, "kitti_format")

    if not only_gt_database:
        step_waymo_convert_and_infos(
            waymo_converter,
            root_path=root_path,
            out_dir=out_dir,
            info_prefix=info_prefix,
            splits=splits,
            workers=workers,
            max_sweeps=max_sweeps,
            save_sensor_data=save_sensor_data,
            skip_cam_instances_infos=skip_cam_instances_infos,
            dry_run=dry_run,
            force=force,
        )
    else:
        _log("[waymo] --only-gt-database set: skipping conversion+infos.")

    # step_waymo_create_gtdb(
    #     waymo_converter,
    #     out_dir=out_dir,
    #     info_prefix=info_prefix,
    #     workers=workers,
    #     dry_run=dry_run,
    #     force=force,
    # )
    step_waymo_create_gtdb(out_dir=out_dir, info_prefix=info_prefix,
                       workers=workers, dry_run=dry_run, force=force)

    ok_db, db_flags = check_gt_database(kitti_out, info_prefix)
    _log("[waymo/summary] info_prefix = %s | kitti_out=%s | gt_db_dir=%s dbinfos=%s" % (
        info_prefix, kitti_out, _yes(db_flags['gt_database_dir']), _yes(db_flags['dbinfos_pickle'])))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#in Windows WSL: ln -sfn /mnt/e/Shared/Dataset/Kitti data/kitti
#ln -s /mnt/e/Shared/Dataset/NuScenes/v1.0-trainval /home/lkk688/Developer/mmdetection3d/data/nuscenus
#(py310) lkk688@newalienware:~/Developer/DeepDataMiningLearning/DeepDataMiningLearning/detection3d$ python mmdet3d_create_data.py --fix-case-symlink --dataset nuscenes --root-path /home/lkk688/Developer/mmdetection3d/data/nuscenus --out-dir /home/lkk688/Developer/mmdetection3d/data/nuscenus


#(py310) lkk688@newalienware:~/Developer/mmdetection3d$ python /home/lkk688/Developer/DeepDataMiningLearning/DeepDataMiningLearning/detection3d/mmdet3d_create_data.py --fix-case-symlink --dataset nuscenes --root-path /home/lkk688/Developer/mmdetection3d/data/nuscenes --out-dir /home/lkk688/Developer/mmdetection3d/data/nuscenes

import argparse

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable dataset preparation for MMDetection3D (nuScenes, KITTI, Waymo v1.4.3)"
    )

    parser.add_argument("--dataset", default="kitti",
                        choices=["nuscenes", "kitti", "waymo"],
                        help="Which dataset to prepare (default: kitti).")

    # mmdetection3d path
    parser.add_argument("--repo-root", default="/home/lkk688/Developer/mmdetection3d",
                        help="Path to the mmdetection3d repository root (run from anywhere).")

    # general parameters
    parser.add_argument("--root-path", default="/home/lkk688/Developer/mmdetection3d/data/kitti", type=str,
                        help="Dataset root path (e.g., /data/nuscenes or /data/kitti or /data/waymo)")
    parser.add_argument("--out-dir", default="/home/lkk688/Developer/mmdetection3d/data/kitti", type=str,
                        help="Output dir (often same as root).")
    parser.add_argument("--info-prefix", default=None, type=str,
                        help="Info file prefix; default is dataset name.")

    # nuScenes parameters
    parser.add_argument("--version", default="v1.0-trainval",
                        choices=["v1.0-trainval", "v1.0-test", "v1.0-mini"],
                        help="nuScenes split version (ignored for KITTI/Waymo).")
    parser.add_argument("--max-sweeps", default=10, type=int,
                        help="nuScenes/Waymo: number of input consecutive frames (default: 10).")
    parser.add_argument("--fix-case-symlink", action="store_true", default=False,
                        help="nuScenes: if basename is not 'nuscenes', create a lowercase alias symlink.")
    parser.add_argument("--dataset-name", default="NuScenesDataset", type=str,
                        help="nuScenes: Dataset class name for GT DB step.")

    # KITTI related
    parser.add_argument("--no-plane", dest="with_plane", action="store_false",
                        help="KITTI: disable using plane info.")
    parser.set_defaults(with_plane=False)

    # Waymo related
    parser.add_argument("--waymo-version", default="v1.4.3",
                        choices=["v1.4.3", "v1.4", "v1.4-mini"],
                        help="Waymo dataset version label.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Waymo: number of threads for conversion.")
    parser.add_argument("--only-gt-database", action="store_true", default=False,
                        help="Waymo: skip conversion+infos; only build GT DB (expects infos present).")
    parser.add_argument("--save-sensor-data", action="store_true", default=False,
                        help="Waymo: actually write images/LiDAR to KITTI tree.")
    parser.add_argument("--skip-cam-instances-infos", action="store_true", default=False,
                        help="Waymo: skip collecting camera instance infos for speed.")

    # general control
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print planned actions without executing.")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Force rebuild of steps (infos/conversion and GT DB).")

    return parser.parse_args(argv)


def main():
    args = parse_args()
    # 默认 info_prefix = dataset 名
    if args.info_prefix is None:
        args.info_prefix = args.dataset

    print(f"[DEBUG] dataset={args.dataset} root={args.root_path} out={args.out_dir}", flush=True)

    if args.dataset == "kitti":
        run_kitti(
            repo_root=args.repo_root,
            root_path=args.root_path,
            out_dir=args.out_dir,
            info_prefix=args.info_prefix,
            version=getattr(args, "version", ""),
            with_plane=args.with_plane,
            dry_run=args.dry_run,
            force=args.force,
        )

    elif args.dataset == "nuscenes":
        run_nuscenes(
            repo_root=args.repo_root,
            root_path=args.root_path,
            out_dir=args.out_dir,
            info_prefix=args.info_prefix,
            version=args.version,
            max_sweeps=args.max_sweeps,
            dataset_name=args.dataset_name,
            fix_case_symlink=args.fix_case_symlink,
            dry_run=args.dry_run,
            force=args.force,
        )

    elif args.dataset == "waymo":
        run_waymo(
            repo_root=args.repo_root,
            root_path=args.root_path,
            out_dir=args.out_dir,
            info_prefix=args.info_prefix,
            waymo_version=args.waymo_version,
            workers=args.workers,
            max_sweeps=args.max_sweeps,
            only_gt_database=args.only_gt_database,
            save_sensor_data=args.save_sensor_data,
            skip_cam_instances_infos=args.skip_cam_instances_infos,
            dry_run=args.dry_run,
            force=args.force,
        )

if __name__ == "__main__":
    main()