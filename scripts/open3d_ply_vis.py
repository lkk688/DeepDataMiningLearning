# ==========================================================
# Local visualization using saved PLY files
# ==========================================================
def load_ply_and_visualize(ply_points_path: str, ply_boxes_path: str | None = None):
    """
    Load previously saved PLY files (points and boxes) for offline visualization.

    Parameters
    ----------
    ply_points_path : str
        Path to the saved LiDAR point cloud PLY file.
    ply_boxes_path : str | None
        Path to optional bounding box PLY file.

    Usage Example
    -------------
    >>> load_ply_and_visualize("frame_0000_points.ply", "frame_0000_boxes.ply")
    """

    import open3d as o3d
    import numpy as np
    import os

    if not os.path.exists(ply_points_path):
        raise FileNotFoundError(ply_points_path)

    pcd = o3d.io.read_point_cloud(ply_points_path)
    geoms = [pcd]
    print(f"[LOAD] Point cloud loaded: {ply_points_path} ({len(pcd.points)} points)")

    if ply_boxes_path and os.path.exists(ply_boxes_path):
        boxes = o3d.io.read_triangle_mesh(ply_boxes_path)
        boxes.paint_uniform_color([1.0, 0.3, 0.0])
        geoms.append(boxes)
        print(f"[LOAD] Boxes loaded: {ply_boxes_path}")

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    geoms.append(axis)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Loaded PLY Visualization",
        width=1440, height=810,
        point_show_normal=False
    )

import os
if __name__ == "__main__":
    import open3d as o3d
        #from visualize_open3d import load_ply_and_visualize
    dir_path="/Users/kaikailiu/Downloads"
    points_path=os.path.join(dir_path, "scene_vis_points.ply")
    boxes_path=os.path.join(dir_path, "scene_vis_boxes.ply")

    pcd = o3d.io.read_point_cloud(points_path)
    boxes = o3d.io.read_line_set(boxes_path)   # ← use read_line_set
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)

    o3d.visualization.draw_geometries([pcd, boxes, axis],
                                    window_name="Point cloud + boxes",
                                    width=1440, height=810)


    #load_ply_and_visualize(points_path, boxes_path)