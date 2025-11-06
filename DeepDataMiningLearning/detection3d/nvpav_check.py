#!/usr/bin/env python3
r"""
PAV: Unzip *in-place*, link labels, and visualize 2D boxes
=========================================================

What this script does
---------------------
1) **Unzip in-place**: Recursively find all `.zip` files under `--data-root` and extract each
   ZIP into **its original folder** as a sibling directory named after the ZIP stem.
   Example:
     camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0007.zip
     -> camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0007/
   The original repository folder structure is fully preserved.

2) **Update previous index**: Update your existing index JSON (from the downloader), marking each
   unzipped ZIP as `"unzipped": true`, `"unzipped_at": <ISO timestamp>`, and `"extracted_dir": <path>`.
   - Use `--index-path` to point to a specific JSON.
   - If not provided, the script auto-detects one of the following under `--data-root`:
       * `perdir_index.json` (created by pav_grab_perdir.py)
       * `simple_index.json` (created by pav_grab_simple.py)
       * else, it will create `pav_index.json`.

3) **Load labels and images, link them, and draw 2D boxes**:
   - Read likely 2D labels from Parquet files (under `labels/` or paths containing keywords `2d`, `bbox`, `box`).
   - Inventory camera images under the unzipped directories.
   - Try to link images with labels by `(clip_id, camera_name, frame_idx)` or nearest `ts_ms` within a tolerance.
   - Draw resulting 2D boxes and save **visualizations** under `--vis-root` (separate from the data root).

4) **Print helpful summaries** of the discovered structure (ZIP counts by top-level, camera image counts,
   label schema and unique values, linking stats), so you can quickly understand what's available.

Typical usage
-------------
# Unzip in-place, update index, and render at most 50 views
python pav_unpack_link_viz.py \
  --data-root /data/rnd-liu/Datasets/PhysicalAI-AV \
  --vis-root  /data/rnd-liu/Datasets/PAI-AV-2Dviz \
  --max-samples 50

# Restrict to specific cameras and chunk number range
python pav_unpack_link_viz.py \
  --data-root /data/rnd-liu/Datasets/PhysicalAI-AV \
  --vis-root  /data/rnd-liu/Datasets/PAI-AV-2Dviz \
  --cameras camera_front_wide_120fov,camera_front_tele_30fov \
  --chunk-range 0 19 \
  --max-samples 100

Dependencies
------------
  pip install pandas pyarrow pillow

Notes
-----
- The script is best-effort and schema-tolerant. If a column is missing (e.g., camera_name), it will try to infer.
- If labels do not contain 2D boxes, drawing will be skipped but summaries are still printed.
"""
from __future__ import annotations
import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ----------
# Regexes
# ----------
UUID_RE = re.compile(r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})")
CHUNK_NUM_RE = re.compile(r"chunk_(\d+)")
TS_ANY_RE = re.compile(r"(\d{13,16}|\d{9,10})")  # ms/us then seconds
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------
# Config
# ----------
@dataclass
class Cfg:
    data_root: Path
    vis_root: Path
    index_path: Path
    cameras: Optional[List[str]]
    chunk_range: Optional[Tuple[int, int]]
    max_samples: int
    tol_ms: int
    label_filter: Optional[str]
    filter_expr: Optional[str]


def parse_args() -> Cfg:
    ap = argparse.ArgumentParser(description="Unzip in-place, link labels, and draw 2D boxes (with index updates)")
    ap.add_argument("--data-root", default="/data/rnd-liu/Datasets/PhysicalAI-AV", help="Root of downloaded dataset (repository-like tree)")
    ap.add_argument("--vis-root", default="outputs/nvpav", help="Output folder for visualizations (separate from data root)")
    ap.add_argument("--index-path", default="", help="Path to downloader index JSON; auto-detected if empty")
    ap.add_argument("--cameras", default="", help="Comma-separated camera folder names to include")
    ap.add_argument("--chunk-range", nargs=2, type=int, default=None, help="Optional chunk number range: start end (inclusive)")
    ap.add_argument("--max-samples", type=int, default=10, help="Max number of visualizations to write")
    ap.add_argument("--tol-ms", type=int, default=25, help="Timestamp tolerance for linking (ms)")
    ap.add_argument("--label-filter", default="", help="Substring to select labels parquet files (path contains this)")
    ap.add_argument("--filter-expr", default="", help="pandas query string to pre-filter labels")

    a = ap.parse_args()

    data_root = Path(a.data_root)
    vis_root = Path(a.vis_root)

    # Auto-detect index path if not provided
    cand = [data_root / "perdir_index.json", data_root / "simple_index.json", data_root / "pav_index.json"]
    index_path = Path(a.index_path) if a.index_path else next((p for p in cand if p.exists()), cand[-1])

    cams = [c.strip() for c in a.cameras.split(",") if c.strip()] or None
    cr = (a.chunk_range[0], a.chunk_range[1]) if a.chunk_range else None

    return Cfg(
        data_root=data_root,
        vis_root=vis_root,
        index_path=index_path,
        cameras=cams,
        chunk_range=cr,
        max_samples=a.max_samples,
        tol_ms=a.tol_ms,
        label_filter=a.label_filter or None,
        filter_expr=a.filter_expr or None,
    )


# ----------
# Index I/O
# ----------

def load_index(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"files": {}, "created_at": datetime.utcnow().isoformat() + "Z"}


def save_index(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _mark_unzipped(index: dict, zip_path: Path, extracted_dir: Path) -> None:
    """Mark a local ZIP as unzipped in the index, creating/merging an entry if needed."""
    # Try to find an existing entry by matching local_path or key
    key = None
    for k, v in index.get("files", {}).items():
        lp = v.get("local_path")
        if lp and Path(lp) == zip_path:
            key = k
            break
    if key is None:
        key = str(zip_path)
        index.setdefault("files", {})[key] = {"local_path": str(zip_path)}

    entry = index["files"][key]
    entry["unzipped"] = True
    entry["unzipped_at"] = datetime.utcnow().isoformat() + "Z"
    entry["extracted_dir"] = str(extracted_dir)


# ----------
# Unzip in-place
# ----------

def unzip_in_place(data_root: Path, index: dict) -> List[Path]:
    """Unzip every `.zip` under `data_root` into `zip.parent / zip.stem`.
    - Robust to upper-case extensions (e.g., .ZIP)
    - If no zips are found at `data_root`, auto-detect a nested dataset root that
      contains expected top-level dirs (camera/lidar/radar/calibration/labels/ego*/).
    Returns a list of created/verified extracted directories.
    """
    def find_zips(root: Path) -> List[Path]:
        # rglob is case-sensitive on pattern; scan all and filter by suffix
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".zip"]

    # Try given root first
    zips = find_zips(data_root)

    # Auto-adjust root if nothing found (common when user points to parent folder)
    if not zips:
        candidates = [p for p in data_root.iterdir() if p.is_dir()]
        expected = {"camera", "lidar", "radar", "calibration", "labels", "ego", "ego_motion"}
        for cand in candidates:
            try:
                children = {q.name for q in cand.iterdir() if q.is_dir()}
            except Exception:
                continue
            score = len(children & expected)
            if score >= 2:  # likely dataset root
                adj = cand
                print(f"[INFO] No zips at {data_root}. Using nested root: {adj}")
                data_root = adj
                zips = find_zips(data_root)
                break

    # Print a quick summary by top-level folder for clarity
    by_top: Dict[str, int] = {}
    top_examples: Dict[str, str] = {}
    for z in zips:
        rel = z.relative_to(data_root)
        top = rel.parts[0]
        by_top[top] = by_top.get(top, 0) + 1
        top_examples.setdefault(top, str(rel))

    print("[STRUCT] ZIP counts by top-level:")
    if not by_top:
        print("  (none found under)", data_root)
    else:
        for t in sorted(by_top.keys()):
            print(f"  {t}: {by_top[t]}  e.g., {top_examples[t]}")

    created: List[Path] = []
    for zpath in zips:
        rel = zpath.relative_to(data_root)
        target = zpath.parent / zpath.stem
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(target)
                print(f"[UNZIP] {rel} -> {target.relative_to(data_root)}")
            except Exception as e:
                print(f"[WARN] Failed to unzip {rel}: {e}")
                continue
        else:
            print(f"[SKIP] Already extracted: {rel} -> {target.relative_to(data_root)}")
        created.append(target)
        _mark_unzipped(index, zpath, target)
        save_index(cfg.index_path, index)
    return created


# ----------
# Labels & images
# ----------

def _ensure_clip_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    for cand in ("clip_id", "clip_uuid", "uuid", "clip"):
        if cand in df.columns:
            if cand != "clip_id":
                df = df.rename(columns={cand: "clip_id"})
            df["clip_id"] = df["clip_id"].astype(str)
            return df
    # Try index
    try:
        idx = df.index
        if not isinstance(idx, pd.RangeIndex):
            ser = pd.Series(idx, name=idx.name or "clip_id")
            if ser.astype(str).str.match(UUID_RE).mean() > 0.5:
                df = df.reset_index().rename(columns={ser.name: "clip_id"})
                df["clip_id"] = df["clip_id"].astype(str)
                return df
    except Exception:
        pass
    return df


def load_labels_2d_tables(root: Path, label_filter: Optional[str], filter_expr: Optional[str]) -> pd.DataFrame:
    """Load likely 2D label Parquet(s) under `root`.
    Heuristics: paths containing 'labels' or filenames containing '2d', 'bbox', 'box'.
    Returns a DataFrame with normalized columns x1,y1,x2,y2, camera_name, clip_id, ts_ms (optional), frame_idx (optional).
    """
    cand_files: List[Path] = []
    for p in root.rglob("*.parquet"):
        s = str(p).lower()
        if "metadata/" in s:
            continue
        if label_filter and label_filter.lower() not in s:
            continue
        if ("labels" in s) or ("2d" in s) or ("bbox" in s) or ("box" in s):
            cand_files.append(p)
    if not cand_files:
        for p in (root / "labels").rglob("*.parquet"):
            cand_files.append(p)

    frames: List[pd.DataFrame] = []
    for fp in sorted(set(cand_files)):
        try:
            df = pd.read_parquet(fp)
            df = _ensure_clip_id_column(df)
            df["__src_file"] = str(fp)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read labels parquet {fp}: {e}")
    if not frames:
        print("[INFO] No labels parquet found.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Column normalization
    for cand in ("camera", "camera_name", "sensor", "sensor_name", "cam"):
        if cand in df.columns:
            df = df.rename(columns={cand: "camera_name"})
            break
    if "camera_name" not in df.columns:
        df["camera_name"] = None

    for cand in ("timestamp", "ts", "time_ms", "t_ms", "frame_timestamp", "image_timestamp"):
        if cand in df.columns:
            df = df.rename(columns={cand: "ts_ms"})
            break
    if "ts_ms" not in df.columns:
        df["ts_ms"] = pd.NA

    for cand in ("frame_index", "frame_idx", "idx", "image_index"):
        if cand in df.columns and "frame_idx" not in df.columns:
            df = df.rename(columns={cand: "frame_idx"})
            break
    if "frame_idx" not in df.columns:
        df["frame_idx"] = pd.NA

    df = _normalize_boxes_from_df(df)

    # Helpful schema print
    print("[STRUCT] Labels DataFrame:")
    print("  rows:", len(df))
    print("  columns:", list(df.columns))
    if "camera_name" in df.columns:
        print("  unique camera_name (sample):", list(pd.Series(df["camera_name"]).dropna().unique())[:10])
    print("  unique clip_id count:", df["clip_id"].nunique() if "clip_id" in df.columns else 0)

    if filter_expr:
        try:
            before = len(df)
            df = df.query(filter_expr)
            print(f"[FILTER] labels query: '{filter_expr}'  {before} -> {len(df)}")
        except Exception as e:
            print(f"[WARN] Filter expr failed: {e}")
    return df


def _normalize_boxes_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = df.columns.str.lower().tolist()
    def has_all(names: Sequence[str]) -> bool:
        return all(n in cols for n in names)

    mappings = [
        ("x_min", "y_min", "x_max", "y_max"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    ]
    normalized = False
    for a,b,c,d in mappings:
        if has_all([a,b,c,d]):
            df = df.rename(columns={a:"x1", b:"y1", c:"x2", d:"y2"})
            normalized = True
            break
    if not normalized:
        for a,b,c,d in [("x","y","w","h"),("cx","cy","w","h")]:
            if has_all([a,b,c,d]):
                df["x1"] = pd.to_numeric(df[a], errors="coerce")
                df["y1"] = pd.to_numeric(df[b], errors="coerce")
                df["x2"] = df["x1"] + pd.to_numeric(df[c], errors="coerce")
                df["y2"] = df["y1"] + pd.to_numeric(df[d], errors="coerce")
                normalized = True
                break
    if not normalized and "bbox" in df.columns:
        def _to_cols(v):
            try:
                x,y,w,h = v
                return pd.Series([x,y,x+w,y+h])
            except Exception:
                return pd.Series([pd.NA,pd.NA,pd.NA,pd.NA])
        xyxy = df["bbox"].apply(_to_cols)
        xyxy.columns = ["x1","y1","x2","y2"]
        df = pd.concat([df, xyxy], axis=1)
        normalized = True

    for c in ("x1","y1","x2","y2"):
        if c not in df.columns:
            df[c] = pd.NA
    return df


def inventory_images(data_root: Path, cameras: Optional[List[str]], chunk_range: Optional[Tuple[int,int]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for p in data_root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            rel = p.relative_to(data_root)
            cam_name = _infer_camera_name_from_path(rel)
            if cameras and cam_name not in cameras:
                continue
            chunk_num = _infer_chunk_num_from_path(rel)
            if chunk_range and chunk_num is not None:
                if not (chunk_range[0] <= chunk_num <= chunk_range[1]):
                    continue
            clip_id = _infer_clip_id_from_path(rel)
            ts_ms, frame_idx = _infer_ts_or_index_from_name(p.name)
            rows.append({
                "img_path": str(p),
                "clip_id": clip_id,
                "camera_name": cam_name,
                "chunk_num": chunk_num,
                "ts_ms": ts_ms,
                "frame_idx": frame_idx,
            })
    df = pd.DataFrame(rows)
    print("[STRUCT] Images:")
    print("  total:", len(df))
    if not df.empty:
        print("  cameras (counts):", df.groupby("camera_name").size().sort_values(ascending=False).to_dict())
        print("  example:", df.iloc[0].to_dict())
    return df


def _infer_camera_name_from_path(rel: Path) -> Optional[str]:
    parts = [str(x) for x in rel.parts]
    for i, part in enumerate(parts):
        if part == "camera" and i+1 < len(parts):
            return parts[i+1]
    m = re.search(r"(camera_[a-z_0-9]+)", str(rel), re.IGNORECASE)
    return m.group(1) if m else None


def _infer_chunk_num_from_path(rel: Path) -> Optional[int]:
    m = CHUNK_NUM_RE.search(str(rel))
    return int(m.group(1)) if m else None


def _infer_clip_id_from_path(rel: Path) -> Optional[str]:
    m = UUID_RE.search(str(rel))
    return m.group(1) if m else None


def _infer_ts_or_index_from_name(name: str) -> Tuple[Optional[int], Optional[int]]:
    m = TS_ANY_RE.search(name)
    if m:
        val = int(m.group(1))
        if len(m.group(1)) <= 10:
            val *= 1000
        return val, None
    m2 = re.search(r"(?:frame|idx|image)_?(\d+)", name)
    if m2:
        return None, int(m2.group(1))
    return None, None


def link_images_labels(imgs: pd.DataFrame, labels: pd.DataFrame, tol_ms: int) -> pd.DataFrame:
    if imgs.empty or labels.empty:
        return pd.DataFrame()

    L = labels.copy()
    L = _ensure_clip_id_column(L)
    if "camera_name" not in L.columns:
        L["camera_name"] = L.get("camera") or L.get("sensor_name") or None
    for c in ("x1","y1","x2","y2"):
        if c not in L.columns:
            L[c] = pd.NA

    # 1) frame_idx exact join
    A = imgs.dropna(subset=["clip_id","camera_name","frame_idx"]).copy()
    B = L.dropna(subset=["clip_id","camera_name","frame_idx"]).copy()
    merged_idx = pd.merge(A, B, on=["clip_id","camera_name","frame_idx"], how="inner", suffixes=("_img","_lab"))

    # 2) nearest timestamp
    left_ts = imgs.dropna(subset=["clip_id","camera_name","ts_ms"]).copy()
    right_ts = L.dropna(subset=["clip_id","camera_name","ts_ms"]).copy()
    left_ts = left_ts.sort_values(["clip_id","camera_name","ts_ms"]) 
    right_ts = right_ts.sort_values(["clip_id","camera_name","ts_ms"]) 
    merged_ts_parts: List[pd.DataFrame] = []
    for (cid, cam), g in left_ts.groupby(["clip_id","camera_name"], sort=False):
        r = right_ts[(right_ts["clip_id"]==cid) & (right_ts["camera_name"]==cam)]
        if r.empty:
            continue
        m = pd.merge_asof(g, r, on="ts_ms", direction="nearest", tolerance=pd.Timedelta(milliseconds=tol_ms))
        if not m.empty:
            merged_ts_parts.append(m)
    merged_ts = pd.concat(merged_ts_parts, ignore_index=True) if merged_ts_parts else pd.DataFrame()

    merged = pd.concat([merged_idx, merged_ts], ignore_index=True, sort=False)

    for c in ("x1","y1","x2","y2"):
        if c not in merged.columns:
            merged[c] = pd.NA

    keep = ["img_path","clip_id","camera_name","ts_ms","frame_idx","x1","y1","x2","y2"]
    if "class" in merged.columns: keep.append("class")
    if "category" in merged.columns and "class" not in keep: keep.append("category")

    merged = merged[keep].dropna(subset=["img_path","x1","y1","x2","y2"], how="any")
    print("[STRUCT] Linked pairs:")
    print("  total:", len(merged))
    if not merged.empty:
        print("  by camera:", merged.groupby("camera_name").size().sort_values(ascending=False).to_dict())
        print("  example:", merged.iloc[0].to_dict())
    return merged


def draw_boxes(merged: pd.DataFrame, vis_root: Path, max_samples: int = 50) -> int:
    vis_root.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for _, row in merged.head(max_samples).iterrows():
        try:
            img = Image.open(row["img_path"]).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")
            x1,y1,x2,y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]) 
            draw.rectangle([x1,y1,x2,y2], outline=(255,0,0,255), width=3)
            label = str(row.get("class") or row.get("category") or "").strip()
            if label and font:
                tw, th = draw.textsize(label, font=font)
                draw.rectangle([x1, max(0,y1-th-4), x1+tw+6, y1], fill=(255,0,0,160))
                draw.text((x1+3, y1-th-2), label, fill=(255,255,255,255), font=font)

            cam = row.get("camera_name") or "camera"
            cid = (row.get("clip_id") or "clip").replace("/","_")
            sub = vis_root / str(cam) / str(cid)
            sub.mkdir(parents=True, exist_ok=True)
            base = Path(row["img_path"]).stem + "_2dbox.jpg"
            save_path = sub / base
            img.save(save_path, quality=92)
            count += 1
        except Exception as e:
            print(f"[WARN] draw failed for {row.get('img_path')}: {e}")
    print(f"[DONE] wrote {count} visualization(s) under: {vis_root}")
    return count


# ----------
# Main
# ----------

def main() -> None:
    global cfg
    cfg = parse_args()

    cfg.vis_root.mkdir(parents=True, exist_ok=True)

    # Load or initialize index
    index = load_index(cfg.index_path)
    print(f"[INDEX] Using: {cfg.index_path}")

    # 1) Unzip in-place & update index
    extracted_dirs = unzip_in_place(cfg.data_root, index)
    print(f"[SUMMARY] Extracted dirs: {len(extracted_dirs)} (unique)")

    # # 2) Load labels
    # labels_df = load_labels_2d_tables(cfg.data_root, cfg.label_filter, cfg.filter_expr)
    # if labels_df.empty:
    #     print("[INFO] No labels found; stopping after unzip & index updates.")
    #     return

    # # 3) Inventory images (now present inside extracted directories)
    # imgs_df = inventory_images(cfg.data_root, cfg.cameras, cfg.chunk_range)
    # if imgs_df.empty:
    #     print("[INFO] No camera images found.")
    #     return

    # # 4) Link & draw
    # merged = link_images_labels(imgs_df, labels_df, cfg.tol_ms)
    # if merged.empty:
    #     print("[INFO] No linked (image,label) pairs; nothing to draw.")
    #     return

    # draw_boxes(merged, cfg.vis_root, max_samples=cfg.max_samples)


if __name__ == "__main__":
    main()
