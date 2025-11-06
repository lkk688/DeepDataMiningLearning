#!/usr/bin/env python3
r"""
PAV camera MP4 → frame extraction + 2D box overlay
==================================================

This script finds **MP4 camera videos** and their matching **2D box parquet** files (e.g.
`<clip_uuid>.<camera_name>.mp4` and `<clip_uuid>.<camera_name>.blurred_boxes.parquet`),
extracts frames, and draws the 2D boxes onto the frames.

It is schema-tolerant: it will normalize common 2D label formats and try to align by
`frame_idx` when available, else it estimates frame index from `ts_ms` using the
video FPS. It also copes with normalized boxes (0–1 range) or pixel coordinates.

Typical usage
-------------
# Extract every 5th frame and draw boxes (limit 200 frames per video)
python pav_video_extract_draw.py \
  --data-root /data/rnd-liu/Datasets/PhysicalAI-AV \
  --out-root  /data/rnd-liu/Datasets/pav_frames2d \
  --every-n 5 \
  --max-per-video 200

# Only specific camera(s) and chunk range
python pav_video_extract_draw.py \
  --data-root /data/rnd-liu/Datasets/PhysicalAI-AV \
  --out-root  /data/rnd-liu/Datasets/pav_frames2d \
  --cameras camera_front_wide_120fov,camera_front_tele_30fov \
  --chunk-range 0 9

Requirements
------------
Python ≥ 3.9 and these packages:
  pip install opencv-python-headless pandas pyarrow pillow

What the script prints
----------------------
• Which (clip, camera) pairs it found and processed
• Video fps, total frames, and a few sample frame indices with labels
• Counts of extracted frames and saved overlays per video
• Where outputs are saved (preserving a camera/clip folder layout)
"""
from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

UUID_RE = re.compile(r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})")
CHUNK_NUM_RE = re.compile(r"chunk_(\d+)")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

@dataclass
class Cfg:
    data_root: Path
    out_root: Path
    cameras: Optional[List[str]]
    chunk_range: Optional[Tuple[int,int]]
    every_n: int
    max_per_video: int
    tol_ms: int


def parse_args() -> Cfg:
    ap = argparse.ArgumentParser(description="Extract frames from camera MP4 and draw 2D boxes")
    ap.add_argument("--data-root", required=True, help="Dataset root that contains camera/*/*/*.mp4 and parquet")
    ap.add_argument("--out-root", required=True, help="Output root for frames/overlays")
    ap.add_argument("--cameras", default="", help="Comma-separated camera folder names to include")
    ap.add_argument("--chunk-range", nargs=2, type=int, default=None, help="Optional chunk number range: start end (inclusive)")
    ap.add_argument("--every-n", type=int, default=1, help="Extract every Nth frame (1 = every frame)")
    ap.add_argument("--max-per-video", type=int, default=200, help="Max number of frames to save per video (after every-N sampling)")
    ap.add_argument("--tol-ms", type=int, default=25, help="Timestamp tolerance when estimating frame index from ts_ms")
    a = ap.parse_args()
    return Cfg(
        data_root=Path(a.data_root),
        out_root=Path(a.out_root),
        cameras=[c.strip() for c in a.cameras.split(",") if c.strip()] or None,
        chunk_range=(a.chunk_range[0], a.chunk_range[1]) if a.chunk_range else None,
        every_n=max(1, a.every_n),
        max_per_video=max(1, a.max_per_video),
        tol_ms=a.tol_ms,
    )


# -------------------------------
# Utilities for discovery/parsing
# -------------------------------

def top1(p: Path) -> str:
    return p.parts[0] if len(p.parts) > 0 else ""


def folder2(p: Path) -> str:
    return "/".join(p.parts[:2]) if len(p.parts) >= 2 else str(p)


def infer_camera_folder(rel: Path) -> Optional[str]:
    # expects camera/<camera_name>/... structure; fallback: regex in path
    parts = list(rel.parts)
    for i, s in enumerate(parts):
        if s == "camera" and i+1 < len(parts):
            return parts[i+1]
    m = re.search(r"(camera_[a-z0-9_]+)", str(rel), re.IGNORECASE)
    return m.group(1) if m else None


def infer_chunk_num(rel: Path) -> Optional[int]:
    m = CHUNK_NUM_RE.search(str(rel))
    return int(m.group(1)) if m else None


def parse_clip_cam_from_name(name: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse `<clip_uuid>.<camera_name>.*` → (clip_uuid, camera_name)."""
    m_uuid = UUID_RE.search(name)
    m_cam = re.search(r"(camera_[a-z0-9_]+)", name, re.IGNORECASE)
    return (m_uuid.group(1) if m_uuid else None, m_cam.group(1) if m_cam else None)


# -------------------------------
# Parquet → box normalization
# -------------------------------

def ensure_clip_id(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    for cand in ("clip_id", "clip_uuid", "uuid", "clip"):
        if cand in df.columns:
            if cand != "clip_id":
                df = df.rename(columns={cand: "clip_id"})
            df["clip_id"] = df["clip_id"].astype(str)
            return df
    # try to populate later from path
    return df


def normalize_2d_boxes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    def col(*cands) -> Optional[str]:
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    # frame index
    if col("frame_idx"):
        df.rename(columns={col("frame_idx"): "frame_idx"}, inplace=True)
    elif col("frame_index", "idx", "image_index"):
        df.rename(columns={col("frame_index", "idx", "image_index"): "frame_idx"}, inplace=True)

    # timestamp (ms/us/seconds)
    ts_col = col("ts_ms", "timestamp", "time_ms", "t_ms", "frame_timestamp", "image_timestamp")
    if ts_col and ts_col != "ts_ms":
        df.rename(columns={ts_col: "ts_ms"}, inplace=True)

    # box forms → x1,y1,x2,y2
    m = col("x_min"), col("y_min"), col("x_max"), col("y_max")
    if all(m):
        df.rename(columns={m[0]: "x1", m[1]: "y1", m[2]: "x2", m[3]: "y2"}, inplace=True)
    else:
        m = col("xmin"), col("ymin"), col("xmax"), col("ymax")
        if all(m):
            df.rename(columns={m[0]: "x1", m[1]: "y1", m[2]: "x2", m[3]: "y2"}, inplace=True)
        else:
            m = col("left"), col("top"), col("right"), col("bottom")
            if all(m):
                df.rename(columns={m[0]: "x1", m[1]: "y1", m[2]: "x2", m[3]: "y2"}, inplace=True)
            else:
                # xywh
                xs, ys, ws, hs = col("x"), col("y"), col("w", "width"), col("h", "height")
                if xs and ys and ws and hs:
                    df["x1"] = pd.to_numeric(df[xs], errors="coerce")
                    df["y1"] = pd.to_numeric(df[ys], errors="coerce")
                    df["x2"] = df["x1"] + pd.to_numeric(df[ws], errors="coerce")
                    df["y2"] = df["y1"] + pd.to_numeric(df[hs], errors="coerce")
                elif "bbox" in df.columns:
                    # bbox can be list-like [x,y,w,h]
                    def _to_cols(v):
                        try:
                            x,y,w,h = v
                            return pd.Series([x, y, x+w, y+h])
                        except Exception:
                            return pd.Series([np.nan, np.nan, np.nan, np.nan])
                    xyxy = df["bbox"].apply(_to_cols)
                    xyxy.columns = ["x1","y1","x2","y2"]
                    df = pd.concat([df, xyxy], axis=1)

    # camera_name
    cam = col("camera_name", "camera", "sensor_name", "sensor", "cam")
    if cam and cam != "camera_name":
        df.rename(columns={cam: "camera_name"}, inplace=True)

    # class/category (optional)
    cls = col("class", "category", "label", "name", "type")
    if cls and cls != "class":
        df.rename(columns={cls: "class"}, inplace=True)

    # numeric coercion for boxes, ts
    for c in ("x1","y1","x2","y2","ts_ms","frame_idx"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def is_normalized_boxes(df: pd.DataFrame) -> bool:
    # Heuristic: if x2 or y2 <= 2 for 99% rows, assume 0..1 normalized
    if not set(["x1","y1","x2","y2"]).issubset(df.columns):
        return False
    q = np.nanquantile(df[["x2","y2"]].to_numpy(dtype=float), 0.99)
    return float(q) <= 2.0


# -------------------------------
# Core processing per video
# -------------------------------

def build_frame_index_from_df(df: pd.DataFrame, fps: float, tol_ms: int) -> Dict[int, List[Tuple[float,float,float,float,Optional[str]]]]:
    """Return a dict: frame_idx -> list of boxes (x1,y1,x2,y2,label or None).
    If df has frame_idx, use it. Else, estimate frame_idx from ts_ms via fps.
    """
    df = df.copy()
    has_idx = "frame_idx" in df.columns and df["frame_idx"].notna().any()
    has_ts = "ts_ms" in df.columns and df["ts_ms"].notna().any()

    if not has_idx and not has_ts:
        return {}

    if has_ts and not has_idx:
        # normalize possible seconds/us to ms
        ts = df["ts_ms"].to_numpy(dtype=float)
        # seconds if mostly <= 1e10; microseconds if >= 1e12
        scale = 1.0
        if np.nanmedian(ts) < 1e11:
            # likely seconds
            if np.nanmax(ts) < 1e6:
                scale = 1000.0
        elif np.nanmedian(ts) > 1e12:
            # likely microseconds
            scale = 0.001
        ts_ms = ts * scale
        frame_idx_est = np.rint(ts_ms * fps / 1000.0).astype(np.int64)
        df["frame_idx"] = frame_idx_est
        has_idx = True

    # collect
    out: Dict[int, List[Tuple[float,float,float,float,Optional[str]]]] = {}
    for _, r in df.iterrows():
        fi = int(r["frame_idx"]) if not pd.isna(r.get("frame_idx")) else None
        if fi is None:
            continue
        x1,y1,x2,y2 = r.get("x1"), r.get("y1"), r.get("x2"), r.get("y2")
        if any(pd.isna(v) for v in (x1,y1,x2,y2)):
            continue
        label = r.get("class") if "class" in r else None
        out.setdefault(fi, []).append((float(x1), float(y1), float(x2), float(y2), label))
    return out


def draw_on_frame(frame_bgr: np.ndarray, boxes: List[Tuple[float,float,float,float,Optional[str]]], normalized: bool) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for (x1,y1,x2,y2,label) in boxes:
        if normalized:
            x1,y1,x2,y2 = x1*w, y1*h, x2*w, y2*h
        # clamp
        x1,y1,x2,y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
        draw.rectangle([x1,y1,x2,y2], outline=(255,0,0,255), width=3)
        if label and font:
            tw, th = draw.textsize(label, font=font)
            draw.rectangle([x1, max(0,y1-th-4), x1+tw+6, y1], fill=(255,0,0,160))
            draw.text((x1+3, y1-th-2), label, fill=(255,255,255,255), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def process_one_pair(mp4_path: Path, parquet_path: Path, out_root: Path, cfg: Cfg) -> Tuple[int,int]:
    rel = mp4_path.relative_to(cfg.data_root)
    clip_id, cam = parse_clip_cam_from_name(mp4_path.name)
    if cfg.cameras and cam not in cfg.cameras:
        return (0,0)

    chunk = infer_chunk_num(rel)
    if cfg.chunk_range and chunk is not None:
        if not (cfg.chunk_range[0] <= chunk <= cfg.chunk_range[1]):
            return (0,0)

    # Prepare output folder: <out>/<camera>/<clip>/chunk_XXXX/
    chunk_str = f"chunk_{chunk:04d}" if chunk is not None else "chunk"
    out_dir = out_root / (cam or "camera") / (clip_id or "clip") / chunk_str
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PROC] video: {rel}")
    print(f"       labels: {parquet_path.relative_to(cfg.data_root)}")

    # Load labels
    df = pd.read_parquet(parquet_path)
    df = ensure_clip_id(df)
    df = normalize_2d_boxes(df)

    # If camera_name missing, fill from parsed camera
    if "camera_name" not in df.columns or df["camera_name"].isna().all():
        if cam:
            df["camera_name"] = cam

    # Filter by camera_name if present
    if "camera_name" in df.columns and cam:
        df = df[df["camera_name"].astype(str).str.lower() == cam.lower()].copy()

    # Determine if boxes are normalized
    normalized = is_normalized_boxes(df)

    # Build frame index → boxes
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print("[WARN] cannot open video:", mp4_path)
        return (0,0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"       fps={fps:.3f}, frames={total_frames}, normalized_boxes={normalized}")

    idx2boxes = build_frame_index_from_df(df, fps=fps, tol_ms=cfg.tol_ms)
    if not idx2boxes:
        print("[INFO] no usable 2D labels (no frame_idx/ts) → skipping overlays for this video")

    saved = 0
    extracted = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % cfg.every_n != 0:
            frame_idx += 1
            continue
        extracted += 1

        out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
        frame_to_write = frame

        if frame_idx in idx2boxes and len(idx2boxes[frame_idx]) > 0:
            frame_to_write = draw_on_frame(frame, idx2boxes[frame_idx], normalized)
        # Save image
        cv2.imwrite(str(out_path), frame_to_write, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved += 1

        if saved >= cfg.max_per_video:
            break
        frame_idx += 1

    cap.release()

    # Nice summary for this pair
    sample_keys = sorted(list(idx2boxes.keys()))[:10]
    print(f"       frames with labels (sample): {sample_keys}")
    print(f"       extracted_frames={extracted}, saved_images={saved} → {out_dir}")
    return extracted, saved


# -------------------------------
# Discovery of (video, label) pairs
# -------------------------------

def find_video_label_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Look for pairs in the **same directory** that share `<uuid>.<camera>` stem.
    E.g.
      01d3...-96db.camera_front_wide_120fov.mp4
      01d3...-96db.camera_front_wide_120fov.blurred_boxes.parquet
    Returns list of (mp4_path, parquet_path).
    """
    pairs: List[Tuple[Path, Path]] = []
    for dirpath in root.rglob("*"):
        if not dirpath.is_dir():
            continue
        mp4s = list(dirpath.glob("*.mp4"))
        if not mp4s:
            continue
        parqs = list(dirpath.glob("*.parquet"))
        if not parqs:
            continue
        # index parquet by stem without trailing qualifier
        cand_map: Dict[str, Path] = {}
        for pq in parqs:
            # keep everything; later we prefer files with 'blurred_boxes' in name
            cand_map[pq.stem] = pq
        for v in mp4s:
            stem = v.stem  # <uuid>.<camera_name>
            # prefer exact 'blurred_boxes' name
            pq = None
            for opq in parqs:
                if opq.stem.startswith(stem) and "blurred_boxes" in opq.name:
                    pq = opq
                    break
            if pq is None:
                # fallback: any parquet that starts with the same stem
                for opq in parqs:
                    if opq.stem.startswith(stem):
                        pq = opq
                        break
            if pq is not None:
                pairs.append((v, pq))
    return pairs


# -------------------------------
# Main
# -------------------------------

def main() -> None:
    cfg = parse_args()
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    pairs = find_video_label_pairs(cfg.data_root)
    if not pairs:
        print("[INFO] No (mp4, parquet) pairs found under:", cfg.data_root)
        return

    print(f"[INFO] Found {len(pairs)} (video,label) pair(s). Processing…")

    total_saved = 0
    for (mp4, pq) in pairs:
        extracted, saved = process_one_pair(mp4, pq, cfg.out_root, cfg)
        total_saved += saved

    print(f"[DONE] Total saved images: {total_saved} → {cfg.out_root}")


if __name__ == "__main__":
    main()
