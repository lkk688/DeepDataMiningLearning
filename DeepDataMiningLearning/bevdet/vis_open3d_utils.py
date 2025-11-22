# vis_open3d_utils.py
# Lightweight Open3D + 2D overlay utilities for 3D detection visualization.
# Works standalone (no MMDet dependencies). Requires: numpy, open3d, pillow, matplotlib (for colormap).

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    import open3d as o3d
except Exception as e:
    o3d = None
    print("[WARN] Open3D is not available. 3D visualizations will be skipped.", e)

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None
    print("[WARN] Pillow (PIL) is not available. 2D overlays will be skipped.")

try:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
except Exception:
    cm = None
    mcolors = None
    print("[WARN] matplotlib is not available. Height coloring will be basic.")


# -----------------------------
# I/O utilities
# -----------------------------

def load_lidar_file(fp: Union[str, Path]) -> np.ndarray:
    """
    Load LiDAR point cloud.
      - .bin : KITTI format float32, Nx4
      - .npy : numpy array
      - .npz : expects 'points' key or single array
      - .ply/.pcd : parse via Open3D (requires open3d)
    Returns Nx4 (x,y,z,reflectance) if available; if not, pads fourth column with zeros.
    """
    fp = str(fp)
    ext = Path(fp).suffix.lower()
    if ext == ".bin":
        pts = np.fromfile(fp, dtype=np.float32)
        pts = pts.reshape(-1, 4)
        return pts
    if ext == ".npy":
        arr = np.load(fp, allow_pickle=True)
        pts = np.array(arr)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 4)
        if pts.shape[1] == 3:
            pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
        return pts
    if ext == ".npz":
        data = np.load(fp, allow_pickle=True)
        if "points" in data:
            pts = data["points"]
        else:
            # fall back to first array
            key = list(data.keys())[0]
            pts = data[key]
        pts = np.array(pts)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 4)
        if pts.shape[1] == 3:
            pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
        return pts
    if ext in [".ply", ".pcd"]:
        if o3d is None:
            raise RuntimeError("Open3D not available to read PLY/PCD.")
        pcd = o3d.io.read_point_cloud(fp)
        pts3 = np.asarray(pcd.points, dtype=np.float32)
        if pts3.shape[1] == 3:
            r = np.zeros((pts3.shape[0], 1), dtype=np.float32)
            return np.concatenate([pts3, r], axis=1)
        return pts3
    raise ValueError(f"Unsupported LiDAR file type: {ext}")


# -----------------------------
# Coloring utilities
# -----------------------------

def color_points_by_height(points: np.ndarray, z_clip: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Map z (height) to RGB. If matplotlib is available, use 'turbo' colormap; else grayscale.
    points: Nx4
    returns Nx3 float in [0,1]
    """
    z = points[:, 2]
    if z_clip is None:
        zmin, zmax = np.percentile(z, 1), np.percentile(z, 99)
    else:
        zmin, zmax = z_clip
    z_norm = (z - zmin) / (max(1e-6, (zmax - zmin)))
    z_norm = np.clip(z_norm, 0.0, 1.0)
    if cm is not None:
        cmap = cm.get_cmap('turbo')  # vibrant rainbow-like
        colors = cmap(z_norm)[:, :3]
        return colors.astype(np.float32)
    # fallback: simple blue-to-red via two channels
    colors = np.stack([z_norm, np.zeros_like(z_norm), 1.0 - z_norm], axis=1)
    return colors.astype(np.float32)


# -----------------------------
# 3D Boxes and helpers
# -----------------------------

def box3d_to_corners_xyzhwlr(box: np.ndarray) -> np.ndarray:
    """
    Convert [x, y, z, dx, dy, dz, yaw] (nuScenes/KITTI yaw around Z) to 8 corners in LiDAR frame.
    Returns (8,3) corners in order suitable for a LineSet.
    """
    x, y, z, dx, dy, dz, yaw = box.tolist()
    # corners in box frame (centered)
    # front is +x, left is +y for LiDAR convention
    x_c = dx / 2.0
    y_c = dy / 2.0
    z_c = dz / 2.0
    corners = np.array([
        [ x_c,  y_c,  z_c],
        [ x_c, -y_c,  z_c],
        [-x_c, -y_c,  z_c],
        [-x_c,  y_c,  z_c],
        [ x_c,  y_c, -z_c],
        [ x_c, -y_c, -z_c],
        [-x_c, -y_c, -z_c],
        [-x_c,  y_c, -z_c],
    ], dtype=np.float32)

    # rotation around z
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)
    corners = (R @ corners.T).T
    corners += np.array([x, y, z], dtype=np.float32)
    return corners


def create_open3d_bbox(box: np.ndarray, color: List[float] = [0.0, 1.0, 0.0]) -> "o3d.geometry.LineSet":
    """
    Create an Open3D LineSet for a 3D box: box=[x,y,z,dx,dy,dz,yaw]
    """
    if o3d is None:
        raise RuntimeError("Open3D not available.")
    corners = box3d_to_corners_xyzhwlr(np.asarray(box, dtype=np.float32))
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # top face
        [4, 5], [5, 6], [6, 7], [7, 4],  # bottom face
        [0, 4], [1, 5], [2, 6], [3, 7]   # verticals
    ]
    colors = np.tile(np.asarray(color, dtype=np.float32)[None, :], (len(lines), 1))
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def get_bbox_center(box: np.ndarray) -> np.ndarray:
    return np.asarray(box[:3], dtype=np.float32)


def get_bbox_top_center(box: np.ndarray) -> np.ndarray:
    x, y, z, dx, dy, dz, yaw = box.tolist()
    return np.array([x, y, z + dz * 0.5], dtype=np.float32)


def create_sphere_marker(center: np.ndarray, radius: float = 0.08, color: List[float] = [1, 1, 1]) -> "o3d.geometry.TriangleMesh":
    """
    Small sphere marker at 'center'.
    """
    if o3d is None:
        raise RuntimeError("Open3D not available.")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.translate(center.astype(np.float64))
    mesh.paint_uniform_color(color)
    return mesh


def create_text_stroke_label(text: str, position: np.ndarray, color: List[float] = [1, 1, 1], scale: float = 0.6) -> "o3d.geometry.LineSet":
    """
    Open3D doesn't render text in 3D easily; emulate by a small cross marker + short vertical line.
    """
    if o3d is None:
        raise RuntimeError("Open3D not available.")
    p = position.astype(np.float64)
    s = 0.25 * scale
    pts = np.array([
        p + np.array([-s, 0, 0]),
        p + np.array([ s, 0, 0]),
        p + np.array([ 0,-s, 0]),
        p + np.array([ 0, s, 0]),
        p + np.array([ 0, 0,-s]),
        p + np.array([ 0, 0, s]),
    ], dtype=np.float64)
    lines = np.array([
        [0, 1], [2, 3], [4, 5]
    ], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (3, 1)))
    return ls


def combine_line_sets(line_sets: List["o3d.geometry.LineSet"], color: Optional[List[float]] = None) -> "o3d.geometry.LineSet":
    """
    Merge multiple LineSets into one. If 'color' is given, override all colors.
    """
    if o3d is None:
        raise RuntimeError("Open3D not available.")
    all_pts = []
    all_lines = []
    all_colors = []
    offset = 0
    for ls in line_sets:
        pts = np.asarray(ls.points)
        lines = np.asarray(ls.lines)
        cols = np.asarray(ls.colors) if ls.colors else np.ones((len(lines), 3), dtype=np.float32)
        all_pts.append(pts)
        all_lines.append(lines + offset)
        all_colors.append(cols)
        offset += pts.shape[0]
    all_pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
    all_lines = np.concatenate(all_lines, axis=0) if all_lines else np.zeros((0, 2), dtype=np.int32)
    all_colors = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3))
    if color is not None and all_lines.shape[0] > 0:
        all_colors = np.tile(np.asarray(color, dtype=np.float32), (all_lines.shape[0], 1))
    out = o3d.geometry.LineSet()
    out.points = o3d.utility.Vector3dVector(all_pts)
    out.lines = o3d.utility.Vector2iVector(all_lines.astype(np.int32))
    if all_colors.size:
        out.colors = o3d.utility.Vector3dVector(all_colors)
    return out


# -----------------------------
# 2D Overlay (projection)
# -----------------------------

def load_calib_matrix(calib_file: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load a 3x4 or 4x4 lidar->image projection matrix.
    Supports:
      - JSON: expects 'lidar2img' or {'P': [...], 'T': [...]} (then returns P @ T)
      - NPZ/NPY: either 'lidar2img' key or the first array
    Returns (3,4) or (4,4) numpy array; the drawing util expects (3,4) ultimately.
    """
    calib_path = Path(calib_file)
    ext = calib_path.suffix.lower()
    if ext == ".json":
        with open(calib_path, "r") as f:
            data = json.load(f)
        if "lidar2img" in data:
            M = np.array(data["lidar2img"], dtype=np.float64)
            return M
        # try P and T keys
        P = np.array(data.get("P", []), dtype=np.float64).reshape(-1, 4)
        T = np.array(data.get("T", []), dtype=np.float64).reshape(4, 4) if "T" in data else None
        if P.size and T is not None:
            # handle P (3x4) or (4x4)
            if P.shape == (4, 4):
                P = P[:3, :]
            M = P @ T
            return M
        return None
    if ext in [".npz", ".npy"]:
        calib = np.load(calib_path, allow_pickle=True)
        if isinstance(calib, np.ndarray):
            return calib
        # dict-like
        if "lidar2img" in calib:
            return calib["lidar2img"]
        # fall back to first item
        key = list(calib.keys())[0]
        return calib[key]
    return None


def project_points(M: np.ndarray, xyz: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Project xyz (N,3) using M (3x4 or 4x4).
    Returns uvz (N,3) where (u,v) are pixel coords, z is depth before divide (for masking).
    """
    if M.shape == (4, 4):
        P = M[:3, :]
    elif M.shape == (3, 4):
        P = M
    else:
        raise ValueError(f"Unexpected calib shape: {M.shape}")
    N = xyz.shape[0]
    homo = np.concatenate([xyz, np.ones((N, 1), dtype=xyz.dtype)], axis=1)  # (N,4)
    uvw = (P @ homo.T).T   # (N,3)
    z = uvw[:, 2] + eps
    u = uvw[:, 0] / z
    v = uvw[:, 1] / z
    return np.stack([u, v, z], axis=1)


def draw_projected_boxes_on_image(
    img_file: Union[str, Path],
    calib_file: Union[str, Path],
    pred_bboxes: np.ndarray,
    gt_bboxes: List[np.ndarray],
    out_path: Union[str, Path],
    pred_labels: Optional[Iterable[int]] = None,
    class_names: Optional[List[str]] = None,
    thickness: int = 2
) -> None:
    """
    Project 3D boxes to the image and save an overlay PNG.
    """
    if Image is None:
        print("[WARN] Pillow not available. Skipping 2D overlay.")
        return
    img = Image.open(str(img_file)).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    M = load_calib_matrix(calib_file)
    if M is None:
        print(f"[WARN] Could not load calib from {calib_file}. Skipping 2D overlay.")
        return

    def draw_box(bx: np.ndarray, color: Tuple[int, int, int]):
        corners = box3d_to_corners_xyzhwlr(bx)  # (8,3)
        uvz = project_points(M, corners)  # (8,3)
        # keep only positive depth
        mask = uvz[:, 2] > 0.1
        if not np.all(mask):
            # If some are behind camera, skip
            return
        uv = uvz[:, :2]
        # edges as in 3D lines
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for a, b in edges:
            ua, va = uv[a]
            ub, vb = uv[b]
            draw.line([(ua, va), (ub, vb)], fill=color, width=thickness)

    # Pred in green
    for i, bx in enumerate(np.asarray(pred_bboxes)):
        draw_box(bx, (40, 255, 40))

    # GT in red
    for bx in gt_bboxes:
        draw_box(np.asarray(bx), (255, 40, 40))

    img.save(str(out_path))


# -----------------------------
# Main visualization entry
# -----------------------------

def visualize_with_open3d(
    lidar_file: Union[str, Path],
    predictions_dict: Dict[str, Any],
    gt_bboxes: List[np.ndarray],
    out_dir: Union[str, Path],
    basename: str,
    headless: bool = False,
    img_file: Optional[Union[str, Path]] = None,
    calib_file: Optional[Union[str, Path]] = None
) -> None:
    """
    Visualize point cloud + predicted/gt boxes. Save PLYs in headless mode; otherwise open an interactive window.
    Optionally also draw 2D projected overlays to PNG if 'img_file' and 'calib_file' are provided.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load points
    points = load_lidar_file(lidar_file)
    if o3d is None:
        print("[WARN] Open3D not installed; skipping 3D visualization, but 2D overlay may still be produced.")
    else:
        # Create colored point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        colors = color_points_by_height(points)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        # Origin frame
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        # Predictions
        pred_bboxes = np.asarray(predictions_dict.get("bboxes_3d", []), dtype=np.float32)
        pred_labels = predictions_dict.get("labels_3d", [])
        pred_scores = predictions_dict.get("scores_3d", [])

        # Fallback class names
        class_names = None
        meta = predictions_dict.get("metainfo", {})
        if isinstance(meta, dict):
            class_names = meta.get("classes", None)
        if class_names is None:
            class_names = ["Car", "Pedestrian", "Cyclist"]

        geometries = [pcd, axes]
        pred_line_sets = []
        pred_text_line_sets = []

        for i, bx in enumerate(pred_bboxes):
            ls = create_open3d_bbox(bx, color=[0.0, 1.0, 0.0])
            geometries.append(ls)
            pred_line_sets.append(ls)
            # center + tiny text-stroke marker
            center = get_bbox_center(bx)
            geometries.append(create_sphere_marker(center, radius=0.06, color=[0.0, 1.0, 0.0]))

            topc = get_bbox_top_center(bx)
            # try to attach a label cross
            geometries.append(create_text_stroke_label("", topc, color=[1.0, 1.0, 1.0], scale=0.5))

        # Ground-truth
        gt_line_sets = []
        for bx in gt_bboxes:
            ls = create_open3d_bbox(np.asarray(bx), color=[1.0, 0.0, 0.0])
            geometries.append(ls)
            gt_line_sets.append(ls)
            center = get_bbox_center(bx)
            geometries.append(create_sphere_marker(center, radius=0.05, color=[1.0, 0.0, 0.0]))

        # 2D overlay if possible
        if img_file and calib_file:
            try:
                out2d = out_dir / f"{basename}_2d.png"
                draw_projected_boxes_on_image(
                    img_file=img_file,
                    calib_file=calib_file,
                    pred_bboxes=pred_bboxes,
                    gt_bboxes=gt_bboxes,
                    out_path=str(out2d),
                    pred_labels=pred_labels,
                    class_names=class_names
                )
                print(f"[2D] Saved overlay: {out2d}")
            except Exception as e:
                print(f"[WARN] 2D overlay failed: {e}")

        if headless:
            # Save PLYs
            pcd_path = out_dir / f"{basename}_points.ply"
            axes_path = out_dir / f"{basename}_axes.ply"
            o3d.io.write_point_cloud(str(pcd_path), pcd)
            o3d.io.write_triangle_mesh(str(axes_path), axes)
            if pred_line_sets:
                o3d.io.write_line_set(str(out_dir / f"{basename}_pred_bboxes.ply"), combine_line_sets(pred_line_sets))
            if gt_line_sets:
                o3d.io.write_line_set(str(out_dir / f"{basename}_gt_bboxes.ply"), combine_line_sets(gt_line_sets, color=[1, 0, 0]))
            print(f"[3D] Saved PLYs to {out_dir}")
        else:
            print("[3D] Opening interactive Open3D window. Close the window to continue...")
            o3d.visualization.draw_geometries(geometries, window_name=f"3D Viz: {basename}", width=1400, height=900)