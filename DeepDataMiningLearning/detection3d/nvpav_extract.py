#!/usr/bin/env python3
r"""
Pack one clip (UUID folder) into a compressed NumPy bundle (.npz) — with VERBOSE DEBUG
=====================================================================================

This is a **debug/diagnostic** build of the DracoPy-based packer. It adds:
- Rich logging (step-by-step) with timing for each major stage
- Safety flags to isolate bottlenecks: `--skip-video`, `--skip-radar`, `--skip-ego`,
  `--skip-lidar-decode`
- PyAV frame-extraction watchdog + automatic fallback to OpenCV if slow
- Progress prints every K spins (`--report-every`)

Typical triage usage
--------------------
# 1) 快速自检（不解码视频也不解码点云）
python pav_clip_to_npz_dracopy_debug.py \
  --clip-dir /path/to/<uuid> \
  --skip-video --skip-lidar-decode --report-every 1

# 2) 只测 LiDAR Draco 解码
python pav_clip_to_npz_dracopy_debug.py \
  --clip-dir /path/to/<uuid> \
  --skip-video --report-every 1

# 3) 只测视频抽帧（不做 LiDAR Draco 解码）
python pav_clip_to_npz_dracopy_debug.py \
  --clip-dir /path/to/<uuid> \
  --skip-lidar-decode --report-every 1

# 4) 正常打包，但每 5 帧报告一次进度，帧解码超 1.5s 自动换 OpenCV
python pav_clip_to_npz_dracopy_debug.py \
  --clip-dir /path/to/<uuid> \
  --report-every 5 --frame-decode-timeout 1.5

Notes
-----
- 仍按 LiDAR `reference_timestamp` (~10Hz) 做锚点；`--sample-step` 可降到 5Hz/2Hz 等。
- 相机帧选择根据 `<uuid>.<camera>.timestamps.parquet` 的最近时间（容差 `--tol-us`）。
- 输出 .npz 保存在 UUID 目录。
"""
from __future__ import annotations
import argparse
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import cv2
from PIL import Image

# Optional dependencies
try:
    import av  # PyAV
except Exception:
    av = None

try:
    import open3d as o3d
except Exception:
    o3d = None

# DracoPy backend
try:
    import DracoPy as _DracoPy
except Exception:
    _DracoPy = None

UUID_RE = re.compile(r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})")

# ---------------------------
# CLI & config
# ---------------------------
@dataclass
class Cfg:
    clip_dir: Path
    out_name: Optional[str]
    tol_us: int
    sample_step: int
    max_spins: Optional[int]
    max_points: Optional[int]
    voxel: Optional[float]
    cameras: Optional[List[str]]
    jpeg_quality: int
    verbose: bool
    report_every: int
    skip_video: bool
    skip_radar: bool
    skip_ego: bool
    skip_lidar_decode: bool
    frame_decode_timeout: float


def parse_args() -> Cfg:
    ap = argparse.ArgumentParser(description="Pack one UUID into .npz aligned by LiDAR (~10 Hz) — DEBUG build")
    ap.add_argument("--clip-dir", default="/data/rnd-liu/Datasets/nvpav_clips_by_uuid/002dec8e-3d95-4cc2-abbe-99b3a2e78618", help="UUID directory (flat layout, no chunk)")
    ap.add_argument("--out-name", default="", help="Optional output filename; default 'clip_<uuid>_multimodal_10hz.npz'")
    ap.add_argument("--tol-us", type=int, default=50_000, help="Timestamp tolerance (μs), default 50ms")
    ap.add_argument("--sample-step", type=int, default=1, help="Use every K-th LiDAR spin (1=10Hz; 2≈5Hz)")
    ap.add_argument("--max-spins", type=int, default=None, help="Process at most this many spins (debug)")
    ap.add_argument("--max-points", type=int, default=None, help="Cap points per LiDAR spin (random subsample)")
    ap.add_argument("--voxel", type=float, default=None, help="Voxel size in meters for LiDAR downsample (Open3D)")
    ap.add_argument("--cameras", default="", help="Comma-separated camera names to include (default: all found)")
    ap.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality for saved frames")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--report-every", type=int, default=5, help="Print a progress line every K spins (0=off)")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--skip-radar", action="store_true")
    ap.add_argument("--skip-ego", action="store_true")
    ap.add_argument("--skip-lidar-decode", action="store_true")
    ap.add_argument("--frame-decode-timeout", type=float, default=1.5, help="PyAV per-frame timeout seconds before falling back to OpenCV")
    a = ap.parse_args()
    return Cfg(
        clip_dir=Path(a.clip_dir),
        out_name=a.out_name or "",
        tol_us=a.tol_us,
        sample_step=max(1, a.sample_step),
        max_spins=a.max_spins,
        max_points=a.max_points,
        voxel=a.voxel,
        cameras=[s.strip() for s in a.cameras.split(',') if s.strip()] or None,
        jpeg_quality=max(10, min(100, a.jpeg_quality)),
        verbose=bool(a.verbose),
        report_every=max(0, a.report_every),
        skip_video=bool(a.skip_video),
        skip_radar=bool(a.skip_radar),
        skip_ego=bool(a.skip_ego),
        skip_lidar_decode=bool(a.skip_lidar_decode),
        frame_decode_timeout=float(a.frame_decode_timeout),
    )


# ---------------------------
# Logging helpers
# ---------------------------

def log(msg: str, *, important: bool=False):
    ts = time.strftime("%H:%M:%S")
    prefix = "[**]" if important else "[..]"
    print(f"{ts} {prefix} {msg}", flush=True)


def tsec(start: float) -> str:
    return f"{(time.perf_counter()-start):.3f}s"


# ---------------------------
# Calibration loader (best-effort)
# ---------------------------

def load_calibration(clip_dir: Path) -> Dict[str, object]:
    t0 = time.perf_counter()
    cal_root = clip_dir / "calibration"
    out: Dict[str, object] = {"camera_intrinsics": [], "sensor_extrinsics": [], "vehicle": {}}
    if not cal_root.exists():
        log(f"calibration/ not found under {clip_dir}")
        return out

    # camera intrinsic files
    ci = list(cal_root.rglob("*camera_intrinsics*.parquet"))
    se = list(cal_root.rglob("*sensor_extrinsics*.parquet"))
    vd = list(cal_root.rglob("*vehicle*dimension*.parquet"))
    log(f"Calibration files: intrinsics={len(ci)}, extrinsics={len(se)}, vehicle={len(vd)}")
    for p in ci:
        try:
            df = pd.read_parquet(p)
            out["camera_intrinsics"].extend(df.to_dict("records"))
        except Exception as e:
            log(f"WARN read camera_intrinsics failed: {p} → {e}")
    for p in se:
        try:
            df = pd.read_parquet(p)
            out["sensor_extrinsics"].extend(df.to_dict("records"))
        except Exception as e:
            log(f"WARN read sensor_extrinsics failed: {p} → {e}")
    for p in vd:
        try:
            df = pd.read_parquet(p)
            if len(df) > 0:
                out["vehicle"] = df.iloc[0].to_dict()
        except Exception as e:
            log(f"WARN read vehicle dimensions failed: {p} → {e}")
    log(f"Loaded calibration in {tsec(t0)}")
    return out


# ---------------------------
# LiDAR decoding (DracoPy)
# ---------------------------

def draco_decode_points(blob: bytes) -> Dict[str, np.ndarray]:
    if _DracoPy is None:
        raise RuntimeError("DracoPy is required: pip install DracoPy")
    # normalize to bytes
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    elif hasattr(blob, "to_pybytes"):
        blob = blob.to_pybytes()
    elif isinstance(blob, bytearray):
        blob = bytes(blob)
    # try point cloud API
    try:
        if hasattr(_DracoPy, "decode_buffer_to_point_cloud"):
            pc = _DracoPy.decode_buffer_to_point_cloud(blob)
            pts = np.asarray(pc.points, dtype=np.float32)
            out: Dict[str, np.ndarray] = {"points": pts}
            cols = getattr(pc, "colors", None)
            if cols is not None and len(cols) > 0:
                cols = np.asarray(cols, dtype=np.float32)
                if cols.max() > 1.0:
                    cols = cols / 255.0
                out["colors"] = cols
            return out
    except Exception as e:
        log(f"WARN DracoPy point_cloud decode failed, falling back: {e}")
    # generic decode
    mesh = _DracoPy.decode(blob)
    pts = np.asarray(getattr(mesh, "points"), dtype=np.float32)
    out: Dict[str, np.ndarray] = {"points": pts}
    cols = getattr(mesh, "colors", None)
    if cols is not None and len(cols) > 0:
        cols = np.asarray(cols, dtype=np.float32)
        if cols.max() > 1.0:
            cols = cols / 255.0
        out["colors"] = cols
    return out


# ---------------------------
# Camera helpers
# ---------------------------

def load_camera_index(cam_dir: Path) -> Optional[pd.DataFrame]:
    parqs = list(cam_dir.glob("*.timestamps.parquet")) or list(cam_dir.glob("*.parquet"))
    if not parqs:
        return None
    pq = None
    for p in parqs:
        if p.name.endswith(".timestamps.parquet"):
            pq = p; break
    if pq is None:
        pq = parqs[0]
    t0 = time.perf_counter()
    df = pd.read_parquet(pq)
    log(f"Loaded camera index {pq.name} rows={len(df)} in {tsec(t0)}")
    cols = {c.lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    if col("frame_idx"):
        df.rename(columns={col("frame_idx"): "frame_idx"}, inplace=True)
    elif col("frame_index","idx","image_index"):
        df.rename(columns={col("frame_index","idx","image_index"): "frame_idx"}, inplace=True)
    else:
        df["frame_idx"] = np.arange(len(df), dtype=np.int64)
    if col("timestamp_us","timestamp","ts_us","time_us"):
        df.rename(columns={col("timestamp_us","timestamp","ts_us","time_us"): "ts_us"}, inplace=True)
    elif col("ts_ms","time_ms"):
        df.rename(columns={col("ts_ms","time_ms"): "ts_ms"}, inplace=True)
        df["ts_us"] = (pd.to_numeric(df["ts_ms"], errors="coerce").astype("float64") * 1000.0).astype("int64")
    else:
        raise KeyError(f"Cannot find timestamp column in {pq}")
    df = df[["frame_idx","ts_us"]].dropna().astype({"frame_idx":"int64","ts_us":"int64"}).sort_values("ts_us")
    return df


def decode_frame_at(video_path: Path, frame_idx: int, timeout_s: float) -> Optional[np.ndarray]:
    t0 = time.perf_counter()
    if av is not None:
        try:
            container = av.open(str(video_path))
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                log(f"PyAV: no video stream in {video_path}")
                raise RuntimeError("no video stream")
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            tb = float(stream.time_base) if stream.time_base else 1.0 / fps
            target_pts = int(round(frame_idx / fps / tb))
            container.seek(target_pts, any_frame=False, stream=stream)
            for frame in container.decode(stream):
                if (time.perf_counter() - t0) > timeout_s:
                    raise TimeoutError(f"PyAV decode timeout > {timeout_s}s for frame {frame_idx}")
                idx = int(round(frame.pts * tb * fps)) if frame.pts is not None else None
                if idx is None:
                    continue
                if idx >= frame_idx:
                    img = frame.to_ndarray(format="bgr24")
                    container.close()
                    log(f"PyAV got frame {frame_idx} in {tsec(t0)}")
                    return img
            container.close()
            raise RuntimeError("PyAV decode loop ended without frame")
        except Exception as e:
            log(f"PyAV decode failed ({video_path.name} @ {frame_idx}): {e}; fallback to OpenCV")
    # OpenCV fallback
    t1 = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log(f"OpenCV cannot open {video_path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    cap.release()
    if ok:
        log(f"OpenCV got frame {frame_idx} in {tsec(t1)} (fallback)")
        return frame
    log(f"OpenCV failed to read frame {frame_idx}")
    return None


def jpeg_bytes(img_bgr: np.ndarray, quality: int = 92) -> Tuple[bytes, Tuple[int,int]]:
    h, w = img_bgr.shape[:2]
    ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return (buf.tobytes(), (int(h), int(w)))


# ---------------------------
# Radar / Ego helpers
# ---------------------------

def load_radar_tables(radar_root: Path) -> Dict[str, pd.DataFrame]:
    t0 = time.perf_counter()
    tables: Dict[str, pd.DataFrame] = {}
    if not radar_root.exists():
        log("radar/ not found — skipping")
        return tables
    for sensor_dir in sorted([d for d in radar_root.iterdir() if d.is_dir()]):
        parqs = list(sensor_dir.glob("*.parquet"))
        if not parqs:
            continue
        dfs = []
        for p in parqs:
            try:
                df = pd.read_parquet(p)
                dfs.append(df)
            except Exception as e:
                log(f"WARN read radar parquet failed: {p} → {e}")
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True, sort=False)
        cols = {c.lower(): c for c in df.columns}
        def col(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None
        if col("sensor_timestamp","timestamp_us","ts_us","time_us"):
            df.rename(columns={col("sensor_timestamp","timestamp_us","ts_us","time_us"): "ts_us"}, inplace=True)
        elif col("ts_ms","time_ms","timestamp"):
            df.rename(columns={col("ts_ms","time_ms","timestamp"): "ts_ms"}, inplace=True)
            df["ts_us"] = (pd.to_numeric(df["ts_ms"], errors="coerce").astype("float64") * 1000.0).astype("int64")
        else:
            log(f"WARN radar table lacks ts column: {sensor_dir}")
            continue
        df = df.dropna(subset=["ts_us"]).astype({"ts_us":"int64"}).sort_values("ts_us")
        tables[sensor_dir.name] = df
        log(f"Radar {sensor_dir.name}: rows={len(df)}")
    log(f"Loaded all radar in {tsec(t0)}")
    return tables


def load_ego_table(ego_root: Path) -> Optional[pd.DataFrame]:
    if not ego_root.exists():
        log("ego_motion/ not found — skipping")
        return None
    parqs = list(ego_root.glob("*.parquet"))
    if not parqs:
        log("ego_motion has no parquet — skipping")
        return None
    pq = max(parqs, key=lambda p: p.stat().st_size)
    t0 = time.perf_counter()
    df = pd.read_parquet(pq)
    log(f"Loaded ego {pq.name} rows={len(df)} in {tsec(t0)}")
    cols = {c.lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    if col("timestamp_us","timestamp","ts_us","time_us"):
        df.rename(columns={col("timestamp_us","timestamp","ts_us","time_us"): "ts_us"}, inplace=True)
    elif col("ts_ms","time_ms"):
        df.rename(columns={col("ts_ms","time_ms"): "ts_ms"}, inplace=True)
        df["ts_us"] = (pd.to_numeric(df["ts_ms"], errors="coerce").astype("float64") * 1000.0).astype("int64")
    else:
        raise KeyError("ego_motion parquet lacks timestamp column")
    df = df.dropna(subset=["ts_us"]).astype({"ts_us":"int64"}).sort_values("ts_us")
    return df


# ---------------------------
# LiDAR loader
# ---------------------------

# def load_lidar_parquet(lidar_root: Path) -> pd.DataFrame:
#     parqs = list(lidar_root.glob("*.parquet"))
#     if not parqs:
#         raise FileNotFoundError(f"No LiDAR parquet in {lidar_root}")
#     pq = parqs[0]
#     t0 = time.perf_counter()
#     df = pd.read_parquet(pq, columns=["spin_index","reference_timestamp","draco_encoded_pointcloud"])  # narrow read
#     log(f"Loaded LiDAR {pq.name} rows={len(df)} in {tsec(t0)}")
#     df = df.dropna(subset=["reference_timestamp"]).astype({"reference_timestamp":"int64"}).sort_values("reference_timestamp")
#     return df
def load_lidar_parquet(lidar_root: Path) -> pd.DataFrame:
    parqs = list(lidar_root.glob("*.parquet"))
    if not parqs:
        raise FileNotFoundError(f"No LiDAR parquet in {lidar_root}")
    pq = parqs[0]
    # 先尝试窄列读取；失败就全读再重命名
    want_cols = ["spin_index", "reference_timestamp", "draco_encoded_pointcloud"]
    try:
        df = pd.read_parquet(pq, columns=want_cols)
    except Exception:
        df = pd.read_parquet(pq)  # 读全表，再做归一化
    print("[DEBUG] LiDAR columns in file:", list(df.columns))

    # 统一列名：时间戳
    ts_map = ["reference_timestamp", "reference_timestamp_us", "timestamp_us", "ts_us", "sensor_timestamp"]
    if "reference_timestamp" not in df.columns:
        for c in ts_map:
            if c in df.columns:
                df = df.rename(columns={c: "reference_timestamp"})
                break
    if "reference_timestamp" not in df.columns:
        raise KeyError("LiDAR parquet lacks reference_timestamp-like column")

    # 统一列名：spin_index（没有就合成）
    if "spin_index" not in df.columns:
        for alt in ["spin", "spin_no", "spin_id", "frame_id", "index"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "spin_index"})
                break
        else:
            df = df.sort_values("reference_timestamp").reset_index(drop=True)
            df["spin_index"] = df.index.astype("int64")

    # 保留必需列并排序
    keep = ["spin_index", "reference_timestamp", "draco_encoded_pointcloud"]
    keep = [k for k in keep if k in df.columns]
    df = df[keep].dropna(subset=["reference_timestamp"]).astype({"reference_timestamp": "int64"})
    df = df.sort_values("reference_timestamp")
    return df


# ---------------------------
# Alignment helpers
# ---------------------------

def nearest_index(sorted_ts: np.ndarray, t: int) -> int:
    i = np.searchsorted(sorted_ts, t)
    if i <= 0:
        return 0
    if i >= len(sorted_ts):
        return len(sorted_ts) - 1
    return i if (sorted_ts[i] - t) < (t - sorted_ts[i-1]) else i-1


# ---------------------------
# Main pipeline for a clip
# ---------------------------

def process_clip(cfg: Cfg) -> Path:
    uuid = cfg.clip_dir.name
    out_path = cfg.clip_dir / (cfg.out_name or f"clip_{uuid}_multimodal_10hz.npz")

    log(f"==== PROCESS CLIP {uuid} ====", important=True)
    log(f"Options: tol_us={cfg.tol_us}, sample_step={cfg.sample_step}, max_spins={cfg.max_spins}, "
        f"voxel={cfg.voxel}, max_points={cfg.max_points}, cameras={cfg.cameras}, verbose={cfg.verbose}")

    # --- Calibration ---
    calib = load_calibration(cfg.clip_dir)

    # --- LiDAR (anchor timeline) ---
    lidar_root = cfg.clip_dir / "lidar" / "lidar_top_360fov"
    log(f"LiDAR root: {lidar_root}")
    spins = load_lidar_parquet(lidar_root)

    # Subsample spins by step and optional cap
    idxs = np.arange(len(spins), dtype=np.int64)[:: cfg.sample_step]
    if cfg.max_spins is not None:
        idxs = idxs[: cfg.max_spins]
    anchor_ts = spins.iloc[idxs]["reference_timestamp"].to_numpy(dtype=np.int64)
    anchor_spin_idx = spins.iloc[idxs]["spin_index"].to_numpy(dtype=np.int64)
    log(f"Anchor spins: total={len(spins)}, selected={len(idxs)}; ts range=[{anchor_ts.min()}..{anchor_ts.max()}]")

    # --- Cameras ---
    cam_root = cfg.clip_dir / "camera"
    cam_dirs = [d for d in cam_root.iterdir() if d.is_dir()] if cam_root.exists() else []
    if cfg.cameras:
        cam_dirs = [d for d in cam_dirs if d.name in cfg.cameras]
    camera_idxs: Dict[str, pd.DataFrame] = {}
    camera_videos: Dict[str, Path] = {}
    for d in cam_dirs:
        mp4s = list(d.glob("*.mp4"))
        mp4 = None
        for p in mp4s:
            if uuid in p.name:
                mp4 = p; break
        if mp4 is None and mp4s:
            mp4 = mp4s[0]
        idx = load_camera_index(d)
        if mp4 is not None and idx is not None:
            camera_videos[d.name] = mp4
            camera_idxs[d.name] = idx
            log(f"Camera {d.name}: mp4={mp4.name}, index rows={len(idx)}")
        else:
            log(f"Camera {d.name}: missing mp4 or index — skipped")
    camera_names = sorted(camera_videos.keys())
    log(f"Total cameras to pack: {len(camera_names)} → {camera_names}")

    # --- Radar ---
    radar_root = cfg.clip_dir / "radar"
    radar_tables = {} if cfg.skip_radar else load_radar_tables(radar_root)
    radar_names = sorted(radar_tables.keys())

    # --- Ego motion ---
    ego_root = cfg.clip_dir / "ego_motion"
    ego_df = None if cfg.skip_ego else (load_ego_table(ego_root) if ego_root.exists() else None)

    # Pre-cache camera ts arrays for speed
    cam_ts_cache = {k: v['ts_us'].to_numpy(dtype=np.int64) for k, v in camera_idxs.items()}

    # --- Iterate anchors and assemble samples ---
    t_loop = time.perf_counter()
    samples: List[dict] = []
    for kk, (ts_us, li_row) in enumerate(zip(anchor_ts, spins.iloc[idxs].itertuples())):
        t_iter = time.perf_counter()
        if cfg.report_every and (kk % cfg.report_every == 0):
            log(f"[progress] spin {kk+1}/{len(anchor_ts)} ts={int(ts_us)}")

        # LiDAR
        one = {'ts_us': int(ts_us)}
        if cfg.skip_lidar_decode:
            one['lidar'] = {'points': np.empty((0,3), dtype=np.float32)}
        else:
            blob = li_row.draco_encoded_pointcloud
            if isinstance(blob, (memoryview, bytearray)):
                blob = bytes(blob)
            t_dec = time.perf_counter()
            decoded = draco_decode_points(blob)
            pts = decoded["points"]
            cols = decoded.get("colors")
            log(f"  LiDAR decode: {pts.shape} in {tsec(t_dec)}")
            # Optional downsampling
            if cfg.voxel and o3d is not None and pts.shape[0] > 0:
                t_vox = time.perf_counter()
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                if cols is not None:
                    pc.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0).astype(np.float64))
                pc = pc.voxel_down_sample(cfg.voxel)
                pts = np.asarray(pc.points, dtype=np.float32)
                cols = np.asarray(pc.colors, dtype=np.float32) if pc.has_colors() else None
                log(f"  LiDAR voxel {cfg.voxel}: → {pts.shape} in {tsec(t_vox)}")
            if cfg.max_points is not None and pts.shape[0] > cfg.max_points:
                t_sub = time.perf_counter()
                sel = np.random.choice(pts.shape[0], cfg.max_points, replace=False)
                pts = pts[sel]
                if cols is not None:
                    cols = cols[sel]
                log(f"  LiDAR subsample {cfg.max_points}: in {tsec(t_sub)}")
            one['lidar'] = {'points': pts.astype(np.float32)}
            if cols is not None:
                one['lidar']['colors'] = cols.astype(np.float32)

        # Cameras
        if not cfg.skip_video and camera_names:
            cam_pack = {}
            for cam_name in camera_names:
                idx_df = camera_idxs[cam_name]
                ts_arr = cam_ts_cache[cam_name]
                j = np.searchsorted(ts_arr, int(ts_us))
                if j <= 0:
                    jj = 0
                elif j >= len(ts_arr):
                    jj = len(ts_arr)-1
                else:
                    jj = j if (ts_arr[j] - int(ts_us)) < (int(ts_us) - ts_arr[j-1]) else j-1
                ts_j = int(ts_arr[jj])
                if abs(ts_j - int(ts_us)) <= cfg.tol_us:
                    fidx = int(idx_df.iloc[jj]['frame_idx'])
                    t_cam = time.perf_counter()
                    img = decode_frame_at(camera_videos[cam_name], fidx, cfg.frame_decode_timeout)
                    if img is not None:
                        jpeg, (h,w) = jpeg_bytes(img, cfg.jpeg_quality)
                        cam_pack[cam_name] = {'frame_idx': fidx, 'jpeg': jpeg, 'size_hw': (h,w), 'ts_us': ts_j}
                        log(f"  Camera {cam_name}: frame {fidx} ({h}x{w}) in {tsec(t_cam)}")
                    else:
                        log(f"  Camera {cam_name}: FAILED to get frame {fidx}")
                else:
                    log(f"  Camera {cam_name}: no frame within tol ({cfg.tol_us}us); nearest Δ={(abs(ts_j-int(ts_us)))}us")
            if cam_pack:
                one['camera'] = cam_pack

        # Radar
        if radar_tables and not cfg.skip_radar:
            rad_pack = {}
            for rname, rdf in radar_tables.items():
                ts_arr = rdf['ts_us'].to_numpy(dtype=np.int64)
                if ts_arr.size == 0:
                    continue
                j = np.searchsorted(ts_arr, int(ts_us))
                if j <= 0:
                    jj = 0
                elif j >= len(ts_arr):
                    jj = len(ts_arr)-1
                else:
                    jj = j if (ts_arr[j] - int(ts_us)) < (int(ts_us) - ts_arr[j-1]) else j-1
                if abs(int(ts_arr[jj]) - int(ts_us)) <= cfg.tol_us:
                    rows = rdf[rdf['ts_us'] == int(ts_arr[jj])]
                    cols_keep = [c for c in ['azimuth','elevation','distance','radial_velocity','rcs','snr'] if c in rows.columns]
                    if cols_keep:
                        ra = rows[cols_keep].to_numpy(dtype=np.float32)
                        rad_pack[rname] = {'ts_us': int(ts_arr[jj]), 'data': ra, 'fields': cols_keep}
                        log(f"  Radar {rname}: hits={len(rows)}")
            if rad_pack:
                one['radar'] = rad_pack

        # Ego motion
        if ego_df is not None and not cfg.skip_ego and len(ego_df) > 0:
            ts_arr = ego_df['ts_us'].to_numpy(dtype=np.int64)
            j = np.searchsorted(ts_arr, int(ts_us))
            if j <= 0:
                jj = 0
            elif j >= len(ts_arr):
                jj = len(ts_arr)-1
            else:
                jj = j if (ts_arr[j] - int(ts_us)) < (int(ts_us) - ts_arr[j-1]) else j-1
            if abs(int(ts_arr[jj]) - int(ts_us)) <= cfg.tol_us:
                rowe = ego_df.iloc[jj]
                ego = {}
                for kf in ['x','y','z','vx','vy','vz','ax','ay','az','qx','qy','qz','qw','curvature']:
                    if kf in rowe.index:
                        try:
                            ego[kf] = float(rowe[kf])
                        except Exception:
                            pass
                ego['ts_us'] = int(ts_arr[jj])
                if ego:
                    one['ego'] = ego
                    log("  Ego: matched sample")

        samples.append(one)
        if cfg.verbose:
            log(f"Spin {kk} done in {tsec(t_iter)}")

    log(f"All spins processed in {tsec(t_loop)}; N={len(samples)}")

    # --- Save bundle ---
    t_save = time.perf_counter()
    calib_json = json.dumps(load_calibration(cfg.clip_dir), ensure_ascii=False)
    arr_samples = np.array(samples, dtype=object)
    np.savez_compressed(
        out_path,
        uuid=str(uuid),
        timestamps_us=anchor_ts.astype(np.int64),
        samples=arr_samples,
        calibration_json=np.array(calib_json),
        camera_names=np.array(sorted(list(camera_videos.keys())), dtype=object),
        radar_names=np.array(sorted(list(radar_tables.keys())), dtype=object) if radar_tables else np.array([], dtype=object),
    )
    log(f"WROTE {out_path} in {tsec(t_save)}", important=True)
    return out_path


# ---------------------------
# Main
# ---------------------------

def main():
    try:
        cfg = parse_args()
        log(f"PyAV={'YES' if av else 'NO'}, DracoPy={'YES' if _DracoPy else 'NO'}, Open3D={'YES' if o3d else 'NO'}")
        process_clip(cfg)
    except KeyboardInterrupt:
        log("Interrupted by user (Ctrl-C)", important=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
