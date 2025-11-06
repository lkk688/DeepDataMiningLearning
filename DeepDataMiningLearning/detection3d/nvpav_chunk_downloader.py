#!/usr/bin/env python3
r"""
Per-directory chunk grabber for NVIDIA PhysicalAI AV
====================================================

What you get
------------
Download **a fixed range of chunk numbers per folder** ("folder" = second-level directory, e.g.
- camera/camera_front_wide_120fov
- lidar/lidar_top_360fov
- radar/radar_corner_front_left_srr_0
- calibration/camera_intrinsics
- labels/...
- ego/ or ego_motion/...
)
that look like *...chunk_00xx.* (both .zip and .parquet/.parq allowed where relevant),
**preserving the original repo folder structure** under your destination.

This script does **not** use metadata. It lists files in the repo, groups by folder, then for **each
folder** attempts to fetch the chunk numbers in a fixed range: `[start, start+per_dir-1]`.
Default is `start=0`, so `per_dir=10` means `chunk_0000 .. chunk_0009` best-effort **in every folder**.
Files already recorded in the index are skipped.

Why this helps your workflow
----------------------------
You asked to later unzip + collate per-clip data. Pulling the **same numbers per folder** keeps modalities
aligned (e.g., all folders try to fetch 0000..0009), which makes per-clip merging straightforward.

Quick start
-----------
# Grab 10 per folder into /data/rnd-liu/Datasets (default repo)
python pav_grab_perdir.py --per-dir 10

# Preview only
python pav_grab_perdir.py --per-dir 10 --dry-run

# Only certain top-level dirs
python pav_grab_perdir.py --per-dir 10 --include-dirs "camera,lidar,radar,calibration,labels,ego_motion"

# Try to align chunk numbers across folders
python pav_grab_perdir.py --per-dir 10 --align-by-number

Requirements
------------
  pip install --upgrade huggingface_hub
  # optional acceleration
  pip install hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from huggingface_hub import list_repo_files, hf_hub_download

# Recognize basenames like:
#   chunk_0001.zip
#   camera_intrinsics.chunk_0001.parquet
CHUNK_NAME_RE = re.compile(r"(?:^|\.)chunk_(\d+)\.(zip|parq(uet)?)$", re.IGNORECASE)

# Per top-level dir, which extensions are valid
ALLOWED_EXTS_BY_TOP: Dict[str, Set[str]] = {
    "camera/": {".zip"},
    "lidar/": {".zip"},
    "radar/": {".zip"},
    "calibration/": {".zip", ".parquet", ".parq"},
    "labels/": {".zip", ".parquet", ".parq"},
    "ego_motion/": {".zip", ".parquet", ".parq"},
    "ego/": {".zip", ".parquet", ".parq"},
}

DEFAULT_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
DEFAULT_DEST = "/data/rnd-liu/Datasets/PhysicalAI-AV"


def top1(path: str) -> str:
    return path.split("/")[0] + "/" if "/" in path else ""


def folder_key(path: str) -> str:
    """Return the second-level folder key like 'camera/camera_front_wide_120fov/'.
    Falls back to top-level if no second component.
    """
    parts = path.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2]) + "/"
    return top1(path)


def is_allowed_file(path: str, include_toplevel: Optional[Set[str]]) -> bool:
    if path.startswith("metadata/"):
        return False
    t1 = top1(path)
    if include_toplevel and t1 not in include_toplevel:
        return False
    ext = Path(path).suffix.lower()
    allowed = ALLOWED_EXTS_BY_TOP.get(t1)
    if allowed is None:
        # Allow calibration/labels variants like 'calib/' or future dirs if they contain parquet/zip
        allowed = {".zip", ".parquet", ".parq"}
    return ext in allowed


def match_chunk(path: str) -> Optional[int]:
    """Return numeric suffix if the basename matches the chunk pattern, else None."""
    name = Path(path).name
    m = CHUNK_NAME_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def load_index(index_path: Path) -> dict:
    if index_path.exists():
        try:
            return json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"files": {}, "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z"}


def save_index(index_path: Path, data: dict) -> None:
    tmp = index_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(index_path)


def parse_args():
    ap = argparse.ArgumentParser(description="Download a fixed range of chunk numbers per folder (no metadata)")
    ap.add_argument("--per-dir", type=int, default=10, help="How many *numbers* to try per folder; e.g., 10 -> chunk_0000..0009 by default")
    ap.add_argument("--start", type=int, default=0, help="Start number for the range (default: 0 -> chunk_0000)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    ap.add_argument("--dest", default=DEFAULT_DEST, help="Destination directory (original folders preserved)")
    ap.add_argument("--include-dirs", default="", help="Comma-separated top-level dirs to include (e.g., 'camera,lidar,radar,calibration,labels,ego_motion')")
    ap.add_argument("--dry-run", action="store_true", help="Preview but do not download")
    ap.add_argument("--reset-index", action="store_true", help="Ignore existing index JSON")
    return ap.parse_args()


def main():
    args = parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    index_path = dest / "perdir_index.json"
    if args.reset_index and index_path.exists():
        index_path.unlink()
    index = load_index(index_path)

    include_toplevel = {d.strip() + "/" for d in args.include_dirs.split(",") if d.strip()} or None

    # 1) List files from repo
    files = list_repo_files(args.repo, repo_type="dataset")

    # 2) Filter eligible files and bucket by folder
    buckets: Dict[str, List[Tuple[int, str]]] = defaultdict(list)  # folder_key -> list of (num, path)
    for p in files:
        if not is_allowed_file(p, include_toplevel):
            continue
        num = match_chunk(p)
        if num is None:
            continue
        buckets[folder_key(p)].append((num, p))

    if not buckets:
        print("No eligible folders/files found. Check --include-dirs or repo contents.")
        return

    # Sort each bucket deterministically by number then path
    for k in buckets:
        buckets[k].sort(key=lambda x: (x[0], x[1]))

    # 3) Build selection per folder using a FIXED number range
    target_numbers = list(range(args.start, args.start + args.per_dir))
    # Determine already-downloaded files (present in index and on disk)
    already: Set[str] = set()
    for k2, v2 in index["files"].items():
        if v2.get("error"):
            continue
        lp = v2.get("local_path")
        if lp and Path(lp).exists():
            already.add(k2)
    selection: Dict[str, List[str]] = {}

    for k, items in buckets.items():
        # map desired number -> first pending path that has this number
        mp: Dict[int, str] = {}
        for num, path in items:
            if path in already:
                continue
            if num in target_numbers and num not in mp:
                mp[num] = path
        sel_k = [mp[n] for n in target_numbers if n in mp]
        if sel_k:
            selection[k] = sel_k

    # 4) Preview
    total = sum(len(v) for v in selection.values())
    print(f"Planned downloads: {total} file(s) across {len(selection)} folder(s).{' [dry-run]' if args.dry_run else ''}")
    for k in sorted(selection.keys()):
        print(f"[{k}] {len(selection[k])} file(s)")
        for p in selection[k][:10]:
            print("   ", p)
        if len(selection[k]) > 10:
            print(f"   ... and {len(selection[k]) - 10} more")

    if args.dry_run or total == 0:
        return

    # 5) Download, preserving original folder structure under dest
    downloaded = 0
    for k in sorted(selection.keys()):
        for rel in selection[k]:
            try:
                local = hf_hub_download(
                    args.repo,
                    filename=rel,
                    repo_type="dataset",
                    local_dir=str(dest),
                    local_dir_use_symlinks=False,
                )
                index["files"][rel] = {
                    "local_path": local,
                    "downloaded_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                    "size": Path(local).stat().st_size,
                }
                downloaded += 1
            except Exception as e:
                index["files"][rel] = {
                    "error": str(e),
                    "attempted_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                }
            finally:
                save_index(index_path, index)

    print(f"Done. Downloaded {downloaded} file(s). Index: {index_path}")


if __name__ == "__main__":
    main()
