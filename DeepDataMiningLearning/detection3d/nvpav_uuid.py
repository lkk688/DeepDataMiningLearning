#!/usr/bin/env python3
r"""
Clip-centric packager (FLAT, no chunk folders)
=============================================

This script reorganizes NVIDIA PhysicalAI-AV into a **clip-centric view** where each
UUID folder directly contains top-level modality folders (no `chunk_XXXX` level):

clips_by_uuid/
  <uuid>/
    manifest.json
    camera/
      camera_front_wide_120fov/
        <uuid>.camera_front_wide_120fov.mp4 -> (symlink to original)
        <uuid>.camera_front_wide_120fov.blurred_boxes.parquet -> ...
        <uuid>.camera_front_wide_120fov.timestamps.parquet -> ...
    lidar/
      lidar_top_360fov/
        <uuid>.lidar_top_360fov.parquet -> ...
    radar/
      radar_front_center_mrr_2/
        <uuid>....parquet -> ...
    labels/
      ...
    ego_motion/
      <uuid>.ego_motion.parquet -> ...
    calibration/
      camera_intrinsics/camera_intrinsics.chunk_0004.parquet -> ...  # filename keeps chunk id
      sensor_extrinsics/sensor_extrinsics.chunk_0004.parquet -> ...

Key behavior
------------
- **No chunk directories** in the destination layout. We flatten intermediate
  `'<sensor>.chunk_XXXX'` directory levels (e.g., `camera_front_wide_120fov.chunk_0004/`).
- **Calibration** is linked per-clip by matching the chunk id(s) where this clip occurs
  (keeps filenames with chunk id). If no matching files found, we link the entire
  calibration subtree once.
- **Space friendly**: default is **symlink**; optionally use hardlink/copy.
- **Manifest** per UUID enumerates all linked files.

Usage
-----
python pav_clip_packager_flat.py \
  --data-root /data/rnd-liu/Datasets/PhysicalAI-AV \
  --dest-root /data/rnd-liu/Datasets/clips_by_uuid \
  --link-mode symlink \
  --include camera,lidar,radar,calibration,labels,ego_motion \
  --uuid-limit 200

Optionally fold derived assets (extracted frames / lidar PLY) into the clip view:
python pav_clip_packager_flat.py \
  --data-root /data/rnd-liu/Datasets/PhysicalAI-AV \
  --dest-root /data/rnd-liu/Datasets/clips_by_uuid \
  --link-mode symlink \
  --also-collect /data/rnd-liu/Datasets/pav_frames2d_fast:/camera/*/*/*/frames \
  --also-collect /data/exports/pav_lidar_ply:/lidar/*/*/ply

Requirements
------------
Python 3.9+. No external deps.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

UUID_RE = re.compile(r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})")
CHUNK_RE = re.compile(r"chunk_(\d{4})")

DEFAULT_INCLUDE = ["camera", "lidar", "radar", "calibration", "labels", "ego_motion", "ego"]

@dataclass
class Cfg:
    data_root: Path
    dest_root: Path
    include: List[str]
    link_mode: str  # symlink | hardlink | copy
    uuid_limit: Optional[int]
    reset: bool
    dry_run: bool
    also_collect: List[Tuple[Path, str]]  # (root, glob-like pattern to fold into clip tree)


def parse_args() -> Cfg:
    ap = argparse.ArgumentParser(description="Package PhysicalAI-AV into flat clip-centric folders (no chunk level)")
    ap.add_argument("--data-root", default="/data/rnd-liu/Datasets/PhysicalAI-AV", help="Original dataset root (repo-like tree)")
    ap.add_argument("--dest-root", default="/data/rnd-liu/Datasets/nvpav_clips_by_uuid", help="Output root for clip-centric view")
    ap.add_argument("--include", default=",".join(DEFAULT_INCLUDE), help="Comma-separated top-level dirs to include")
    ap.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="copy")
    ap.add_argument("--uuid-limit", type=int, default=1, help="Process at most N UUIDs (debug)")
    ap.add_argument("--reset", action="store_true", help="Ignore existing manifests and rebuild")
    ap.add_argument("--dry-run", action="store_true", help="Plan only; don't write")
    ap.add_argument("--also-collect", action="append", default=[], help=(
        "Extra roots to fold in, format: <root_path>:<pattern>. The <pattern> is a glob-like path under that root. "
        "The tool will guess the modality (camera/lidar/radar/labels/ego_motion/calibration) from the path; otherwise it uses 'derived'."))
    a = ap.parse_args()

    include = [s.strip() for s in a.include.split(',') if s.strip()]

    extras: List[Tuple[Path, str]] = []
    for spec in a.also_collect:
        try:
            root_str, patt = spec.split(":", 1)
            extras.append((Path(root_str), patt))
        except ValueError:
            raise SystemExit(f"Invalid --also-collect spec: {spec}")

    return Cfg(
        data_root=Path(a.data_root),
        dest_root=Path(a.dest_root),
        include=include,
        link_mode=a.link_mode,
        uuid_limit=a.uuid_limit,
        reset=a.reset,
        dry_run=a.dry_run,
        also_collect=extras,
    )


# --------------------------
# Small helpers
# --------------------------

def find_uuids_in_path(p: Path) -> List[str]:
    return UUID_RE.findall(str(p))


def extract_chunk(p: Path) -> Optional[str]:
    m = CHUNK_RE.search(str(p))
    return m.group(1) if m else None


def top_level(p: Path, root: Path) -> Optional[str]:
    try:
        rel = p.relative_to(root)
    except ValueError:
        return None
    return rel.parts[0] if len(rel.parts) > 0 else None


def ensure_dir(d: Path, dry: bool = False):
    if dry:
        return
    d.mkdir(parents=True, exist_ok=True)


def link_file(src: Path, dst: Path, mode: str, dry: bool = False):
    if dry:
        print(f"LINK [{mode}] {src} -> {dst}")
        return
    if dst.exists() or dst.is_symlink():
        return
    ensure_dir(dst.parent, dry=False)
    if mode == "symlink":
        try:
            dst.symlink_to(src)
            return
        except Exception:
            mode = "hardlink"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            mode = "copy"
    if mode == "copy":
        shutil.copy2(src, dst)


def write_manifest(d: Path, manifest: dict, dry: bool = False):
    if dry:
        return
    ensure_dir(d, dry=False)
    tmp = d / "manifest.json.tmp"
    dst = d / "manifest.json"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(dst)


# --------------------------
# Scan and group files by UUID
# --------------------------

def scan_by_uuid(cfg: Cfg) -> Dict[str, Dict[str, List[Path]]]:
    """Return mapping: uuid → { modality (top-level) → list[Path] } for files that contain UUID in name/path."""
    buckets: Dict[str, Dict[str, List[Path]]] = {}
    allowed = set(cfg.include)

    for p in cfg.data_root.rglob("*"):
        if not p.is_file():
            continue
        tl = top_level(p, cfg.data_root)
        if tl is None or tl not in allowed:
            continue
        uuids = find_uuids_in_path(p)
        if len(uuids) != 1:
            continue
        u = uuids[0]
        buckets.setdefault(u, {}).setdefault(tl, []).append(p)

    # Optionally fold in extra roots (e.g., extracted frames or PLYs)
    for root, patt in cfg.also_collect:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            uuids = find_uuids_in_path(p)
            if len(uuids) != 1:
                continue
            u = uuids[0]
            # Guess modality by path segment names
            tl_guess = None
            s = str(p)
            for m in ("camera","lidar","radar","labels","ego_motion","ego","calibration"):
                if f"/{m}/" in s:
                    tl_guess = m
                    break
            if tl_guess is None:
                tl_guess = "derived"
            buckets.setdefault(u, {}).setdefault(tl_guess, []).append(p)

    return buckets


# --------------------------
# Calibration helper (no dest chunk folder)
# --------------------------

def find_calibration_for_chunks(data_root: Path, chunk_ids: Set[str]) -> List[Path]:
    out: List[Path] = []
    calib_root = data_root / "calibration"
    if not calib_root.exists():
        return out
    for p in calib_root.rglob("*"):
        if not p.is_file():
            continue
        m = CHUNK_RE.search(str(p))
        if m and m.group(1) in chunk_ids:
            out.append(p)
    return out


# --------------------------
# Packaging (flat, no chunk)
# --------------------------

def flatten_rel_path(tl: str, p: Path, data_root: Path) -> Path:
    """Given a file path p under <data_root>/<tl>/..., drop any '<sensor>.chunk_XXXX' folder level.
    Example: camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0004/file
             → camera_front_wide_120fov/file
    """
    rel = p.relative_to(data_root / tl)
    parts = list(rel.parts)
    if len(parts) >= 2:
        sensor = parts[0]
        # pattern '<sensor>.chunk_XXXX'
        pat = re.compile(re.escape(sensor) + r"\.chunk_\d{4}$")
        if pat.match(parts[1]):
            parts = [sensor] + parts[2:]
    return Path(*parts)


def package_one_uuid(u: str, files: Dict[str, List[Path]], cfg: Cfg):
    # detect all chunks that contain this uuid (for calibration matching only)
    chunks: Set[str] = set()
    for lst in files.values():
        for p in lst:
            c = extract_chunk(p)
            if c:
                chunks.add(c)

    clip_dir = cfg.dest_root / u
    if cfg.reset and clip_dir.exists() and not cfg.dry_run:
        shutil.rmtree(clip_dir)

    manifest = {
        "uuid": u,
        "chunks_seen": sorted(chunks),  # informative only; not used for layout
        "modalities": {},
        "source_root": str(cfg.data_root),
        "link_mode": cfg.link_mode,
    }

    # Link per modality directly under UUID (no chunk folder)
    for tl, lst in files.items():
        for p in lst:
            sub_rel = flatten_rel_path(tl, p, cfg.data_root)
            dst = clip_dir / tl / sub_rel
            link_file(p, dst, cfg.link_mode, cfg.dry_run)
            manifest.setdefault("modalities", {}).setdefault(tl, []).append(str(dst))

    # Calibration: attach files matching any detected chunk ids; keep subfolders, no chunk dir
    calibs = find_calibration_for_chunks(cfg.data_root, chunks) if chunks else []
    calib_root = cfg.data_root / "calibration"
    if calibs:
        for p in calibs:
            dst = clip_dir / "calibration" / p.relative_to(calib_root)
            link_file(p, dst, cfg.link_mode, cfg.dry_run)
            manifest.setdefault("modalities", {}).setdefault("calibration", []).append(str(dst))
    else:
        if calib_root.exists():
            for p in calib_root.rglob("*"):
                if p.is_file():
                    dst = clip_dir / "calibration" / p.relative_to(calib_root)
                    link_file(p, dst, cfg.link_mode, cfg.dry_run)
                    manifest.setdefault("modalities", {}).setdefault("calibration", []).append(str(dst))

    write_manifest(clip_dir, manifest, cfg.dry_run)


# --------------------------
# Main
# --------------------------

def main():
    cfg = parse_args()
    cfg.dest_root.mkdir(parents=True, exist_ok=True)

    buckets = scan_by_uuid(cfg)
    uuids = sorted(buckets.keys())
    if cfg.uuid_limit:
        uuids = uuids[:cfg.uuid_limit]

    print(f"[INFO] Will package {len(uuids)} UUID(s) → {cfg.dest_root}")
    for i, u in enumerate(uuids, 1):
        print(f"[{i}/{len(uuids)}] {u}")
        package_one_uuid(u, buckets[u], cfg)

    print("[DONE] Flat clip-centric packaging complete.")


if __name__ == "__main__":
    main()
