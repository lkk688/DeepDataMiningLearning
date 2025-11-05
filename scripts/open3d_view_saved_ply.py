#!/usr/bin/env python3
"""
Simple Open3D viewer for PLY files saved by mmdet3d_inference2.py.

Loads any of the following if present:
- <basename>_points.ply         (PointCloud)
- <basename>_axes.ply           (TriangleMesh)
- <basename>_pred_bboxes.ply    (LineSet)
- <basename>_pred_labels.ply    (LineSet)
- <basename>_gt_bboxes.ply      (LineSet)

Usage:
  python scripts/open3d_view_saved_ply.py --dir /path/to/inference_preview --basename 000008

On macOS, install Open3D via:
  pip install open3d
"""

import argparse
import os
import sys

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is not installed. Install with `pip install open3d`.\n")
    sys.exit(1)


def load_if_exists(path: str, loader, name: str):
    """Load a geometry with the given loader if the path exists."""
    if os.path.exists(path):
        try:
            obj = loader(path)
            print(f"[LOAD] {name}: {path}")
            return obj
        except Exception as e:
            print(f"[WARN] Failed to load {name} ({path}): {e}")
    else:
        print(f"[SKIP] {name} not found: {path}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Open3D viewer for saved PLY outputs")
    parser.add_argument("--dir", default="/Volumes/Samsung_T3/inference_results",
                        help="Folder containing PLY files (default: inference_preview)")
    parser.add_argument("--basename", default="000008",
                        help="Base name, e.g. 000008")
    parser.add_argument("--width", type=int, default=1440,
                        help="Viewer window width (default: 1440)")
    parser.add_argument("--height", type=int, default=900,
                        help="Viewer window height (default: 900)")
    args = parser.parse_args()

    base_dir = os.path.expanduser(args.dir)
    base = args.basename

    points_path = os.path.join(base_dir, f"{base}_points.ply")
    axes_path = os.path.join(base_dir, f"{base}_axes.ply")
    pred_bbox_path = os.path.join(base_dir, f"{base}_pred_bboxes.ply")
    pred_label_path = os.path.join(base_dir, f"{base}_pred_labels.ply")
    gt_bbox_path = os.path.join(base_dir, f"{base}_gt_bboxes.ply")

    geoms = []

    pcd = load_if_exists(points_path, o3d.io.read_point_cloud, "Point cloud")
    if pcd is not None:
        geoms.append(pcd)

    axes = load_if_exists(axes_path, o3d.io.read_triangle_mesh, "Coordinate axes")
    if axes is not None:
        geoms.append(axes)

    pred_bboxes = load_if_exists(pred_bbox_path, o3d.io.read_line_set, "Predicted bboxes")
    if pred_bboxes is not None:
        geoms.append(pred_bboxes)

    pred_labels = load_if_exists(pred_label_path, o3d.io.read_line_set, "Predicted labels")
    if pred_labels is not None:
        geoms.append(pred_labels)

    gt_bboxes = load_if_exists(gt_bbox_path, o3d.io.read_line_set, "Ground truth bboxes")
    if gt_bboxes is not None:
        geoms.append(gt_bboxes)

    if not geoms:
        print("\nNo geometries loaded. Check --dir and --basename.")
        print(f"Tried paths:\n  {points_path}\n  {axes_path}\n  {pred_bbox_path}\n  {pred_label_path}\n  {gt_bbox_path}")
        return

    print("\n[INFO] Opening viewer. Controls: mouse to rotate, scroll to zoom, 'Q' to exit.")
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"PLY Viewer: {base}",
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()