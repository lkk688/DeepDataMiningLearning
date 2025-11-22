import os
import argparse
from pathlib import Path
import numpy as np

try:
    # Use mmdet3d's high-level inferencers
    from mmdet3d.apis import (
        LidarDet3DInferencer,
        MonoDet3DInferencer,
        MultiModalityDet3DInferencer
    )
except ImportError:
    print("Error: This script requires 'mmdetection3d' and its dependencies.")
    print("Could not import LidarDet3DInferencer, MonoDet3DInferencer, or MultiModalityDet3DInferencer.")
    print("Please follow the mmdet3d installation guide:")
    print("https://mmdetection3d.readthedocs.io/en/latest/get_started.html")
    exit()

try:
    import open3d as o3d
except ImportError:
    print("Error: This script requires 'open3d' for visualization.")
    print("Please install it: pip install open3d")
    exit()

try:
    import cv2
except ImportError:
    print("Error: This script requires 'opencv-python' for 2D visualization.")
    print("Please install it: pip install opencv-python-headless")
    exit()

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: This script requires 'matplotlib' for point cloud coloring.")
    print("Please install it: pip install matplotlib")
    exit()


def load_lidar_file(file_path):
    """
    Loads a LiDAR file (.bin, .ply, .pcd) and returns (N, C) points.
    For .bin, assumes (x, y, z, intensity).
    """
    ext = os.path.splitext(file_path)[-1]
    
    if ext == '.bin':
        # Assuming KITTI-style .bin (x, y, z, intensity)
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return points
    elif ext in ['.ply', '.pcd']:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        # Add a dummy intensity channel if it doesn't exist
        if points.shape[1] == 3:
            points = np.hstack((points, np.zeros((points.shape[0], 1))))
        return points
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def color_points_by_height(points):
    """
    Color points by height (Z) using a Mayavi-like, high-contrast colormap.
    Applies robust normalization (percentile clipping) and gamma to spread colors.

    Returns: np.ndarray of shape (N, 3) in RGB.
    """
    z_values = points[:, 2]

    # Robust normalization: clip extremes to avoid color compression
    try:
        z_low, z_high = np.percentile(z_values, [2, 98])
    except Exception:
        z_low, z_high = z_values.min(), z_values.max()

    if z_high > z_low:
        z_norm = (z_values - z_low) / (z_high - z_low)
    else:
        z_norm = np.zeros_like(z_values)

    # Clip to [0,1] and apply gamma for mid-range emphasis
    z_norm = np.clip(z_norm, 0.0, 1.0)
    gamma = 0.8
    z_norm = np.power(z_norm, gamma)

    # Use turbo if available (rainbow-like); otherwise viridis fallback
    try:
        cmap = plt.cm.get_cmap('turbo')
    except Exception:
        cmap = plt.cm.viridis

    colors = cmap(z_norm)[:, :3]
    return colors

def create_coordinate_axes_with_arrows(size=10.0):
    """
    Creates coordinate axes with arrow heads for 3D visualization.
    Returns a list of Open3D geometries for X, Y, Z axes with arrows.
    """
    geometries = []
    arrow_length = size * 0.15  # Arrow head length
    arrow_radius = size * 0.02  # Arrow head radius
    
    # X-axis (Red) - pointing in positive X direction
    x_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=size * 0.005, height=size - arrow_length)
    x_cylinder.translate([size/2 - arrow_length/2, 0, 0])
    x_cylinder.rotate(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), center=[0, 0, 0])
    x_cylinder.paint_uniform_color([1, 0, 0])  # Red
    
    x_arrow = o3d.geometry.TriangleMesh.create_cone(
        radius=arrow_radius, height=arrow_length)
    x_arrow.translate([size - arrow_length/2, 0, 0])
    x_arrow.rotate(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), center=[0, 0, 0])
    x_arrow.paint_uniform_color([1, 0, 0])  # Red
    
    geometries.extend([x_cylinder, x_arrow])
    
    # Y-axis (Green) - pointing in positive Y direction
    y_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=size * 0.005, height=size - arrow_length)
    y_cylinder.translate([0, size/2 - arrow_length/2, 0])
    y_cylinder.rotate(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), center=[0, 0, 0])
    y_cylinder.paint_uniform_color([0, 1, 0])  # Green
    
    y_arrow = o3d.geometry.TriangleMesh.create_cone(
        radius=arrow_radius, height=arrow_length)
    y_arrow.translate([0, size - arrow_length/2, 0])
    y_arrow.rotate(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), center=[0, 0, 0])
    y_arrow.paint_uniform_color([0, 1, 0])  # Green
    
    geometries.extend([y_cylinder, y_arrow])
    
    # Z-axis (Blue) - pointing in positive Z direction
    z_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=size * 0.005, height=size - arrow_length)
    z_cylinder.translate([0, 0, size/2 - arrow_length/2])
    z_cylinder.paint_uniform_color([0, 0, 1])  # Blue
    
    z_arrow = o3d.geometry.TriangleMesh.create_cone(
        radius=arrow_radius, height=arrow_length)
    z_arrow.translate([0, 0, size - arrow_length/2])
    z_arrow.paint_uniform_color([0, 0, 1])  # Blue
    
    geometries.extend([z_cylinder, z_arrow])
    
    return geometries

def get_3d_box_corners(bbox_tensor):
    """
    Converts a 7D mmdet3d bbox tensor (x, y, z, l, w, h, yaw)
    into its 8 corner points.
    (Used for 2D projection)
    """
    center = bbox_tensor[:3]
    extent = bbox_tensor[3:6] # l, w, h
    yaw = bbox_tensor[6]
    
    # In KITTI/mmdet3d, z often encodes the bottom center of the box in lidar coords.
    # Open3D OrientedBoundingBox expects the geometric center.
    # Shift z by +h/2 to convert bottom-center -> center.
    center = np.array(center, dtype=float)
    center[2] = float(center[2]) + float(extent[2]) / 2.0
    
    # Create an OrientedBoundingBox
    o3d_bbox = o3d.geometry.OrientedBoundingBox(
        center, 
        o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw)), 
        extent
    )
    
    # Get the 8 corners
    return np.asarray(o3d_bbox.get_box_points())

def create_open3d_bbox(bbox_tensor, color=[1, 0, 0]):
    """
    Converts a 7D mmdet3d bbox tensor (x, y, z, l, w, h, yaw)
    into an open3d.geometry.LineSet for visualization.
    (Using the robust create_from_oriented_bounding_box method)
    """
    center = bbox_tensor[:3]
    extent = bbox_tensor[3:6] # l, w, h
    yaw = bbox_tensor[6]
    
    # Open3D's rotation matrix is from z-axis
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
    
    # Create an OrientedBoundingBox
    # Shift z by +h/2 to convert bottom-center -> geometric center
    c = np.array(center, dtype=float)
    e = np.array(extent, dtype=float)
    c[2] = float(c[2]) + float(e[2]) / 2.0
    o3d_bbox = o3d.geometry.OrientedBoundingBox(c, R, e)
    
    # Create a LineSet from the bounding box
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_bbox)
    
    # Add the color
    line_set.paint_uniform_color(color)
    
    return line_set

def combine_line_sets(line_sets, color=None):
    """
    Combine multiple Open3D LineSet geometries into a single LineSet.
    This avoids relying on operator overloading and ensures correct aggregation.
    """
    import numpy as np
    combined = o3d.geometry.LineSet()
    if not line_sets:
        return combined

    points_accum = []
    lines_accum = []
    offset = 0
    for ls in line_sets:
        pts = np.asarray(ls.points)
        lns = np.asarray(ls.lines)
        if pts.size == 0 or lns.size == 0:
            continue
        points_accum.append(pts)
        lines_accum.append(lns + offset)
        offset += pts.shape[0]

    if not points_accum:
        return combined

    combined.points = o3d.utility.Vector3dVector(np.vstack(points_accum))
    combined.lines = o3d.utility.Vector2iVector(np.vstack(lines_accum))
    if color is not None:
        combined.paint_uniform_color(color)
    return combined

def create_text_label_3d(text, position, color=[1, 1, 1], size=0.5):
    """
    Creates a compact 3D marker (sphere). Intended for center markers.
    The 'text' argument is ignored here; use create_text_stroke_label for text.

    Args:
        text: Unused, kept for API compatibility
        position: 3D position [x, y, z]
        color: RGB color for the marker
        size: Overall scale (sphere radius is size * 0.3)

    Returns:
        Open3D TriangleMesh sphere translated to position.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(size * 0.3, 1e-3))
    sphere.translate(position)
    sphere.paint_uniform_color(color)
    return sphere

def create_text_stroke_label(text, position, color=[1, 1, 1], scale=0.4):
    """
    Creates a lightweight stroke-based text label as an Open3D LineSet.
    This avoids textures and works in both interactive and headless modes.

    Supported characters: A–Z (uppercase), 0–9, dash and space.
    For unknown characters, they are skipped. Text is rendered on the XY plane.

    Args:
        text: String to render (will be uppercased)
        position: 3D position [x, y, z] for the label baseline
        color: RGB color for the strokes
        scale: Overall scale of the rendered text

    Returns:
        Open3D LineSet geometry positioned at 'position'.
    """
    import numpy as np

    # Basic vector font (normalized to 1x1 box per glyph)
    # Each glyph is a list of line segments ((x1, y1), (x2, y2))
    glyphs = {
        'A': [((0,0), (0.5,1)), ((1,0), (0.5,1)), ((0.25,0.5), (0.75,0.5))],
        'B': [((0,0), (0,1)), ((0,1), (0.6,1)), ((0.6,1),(0.6,0.5)), ((0.6,0.5),(0,0.5)),
              ((0,0.5),(0.6,0)), ((0.6,0),(0,0))],
        'C': [((1,0),(0,0)), ((0,0),(0,1)), ((0,1),(1,1))],
        'D': [((0,0),(0,1)), ((0,1),(0.7,0.85)), ((0.7,0.85),(0.7,0.15)), ((0.7,0.15),(0,0))],
        'E': [((1,1),(0,1)), ((0,1),(0,0)), ((0,0),(1,0)), ((0,0.5),(0.6,0.5))],
        'I': [((0.5,0),(0.5,1))],
        'L': [((0,1),(0,0)), ((0,0),(1,0))],
        'N': [((0,0),(0,1)), ((0,1),(1,0)), ((1,0),(1,1))],
        'O': [((0,0),(1,0)), ((1,0),(1,1)), ((1,1),(0,1)), ((0,1),(0,0))],
        'P': [((0,0),(0,1)), ((0,1),(0.7,1)), ((0.7,1),(0.7,0.6)), ((0.7,0.6),(0,0.6))],
        'R': [((0,0),(0,1)), ((0,1),(0.7,1)), ((0.7,1),(0.7,0.6)), ((0.7,0.6),(0,0.6)),
              ((0,0.6),(0.9,0)),],
        'S': [((1,1),(0.2,1)), ((0.2,1),(0,0.8)), ((0,0.8),(0.8,0.6)), ((0.8,0.6),(1,0.4)),
              ((1,0.4),(0.2,0.2)), ((0.2,0.2),(0,0))],
        'T': [((0,1),(1,1)), ((0.5,1),(0.5,0))],
        'U': [((0,1),(0,0.2)), ((0,0.2),(1,0.2)), ((1,0.2),(1,1))],
        'V': [((0,1),(0.5,0)), ((0.5,0),(1,1))],
        'W': [((0,1),(0.25,0)), ((0.25,0),(0.5,0.5)), ((0.5,0.5),(0.75,0)), ((0.75,0),(1,1))],
        'X': [((0,0),(1,1)), ((1,0),(0,1))],
        'Y': [((0,1),(0.5,0.5)), ((1,1),(0.5,0.5)), ((0.5,0.5),(0.5,0))],
        'Z': [((0,1),(1,1)), ((1,1),(0,0)), ((0,0),(1,0))],
        '0': [((0,0),(1,0)), ((1,0),(1,1)), ((1,1),(0,1)), ((0,1),(0,0)), ((0,0),(1,1))],
        '1': [((0.5,0),(0.5,1)), ((0.3,0.2),(0.5,0))],
        '2': [((0,1),(1,1)), ((1,1),(0,0.5)), ((0,0.5),(1,0)), ((1,0),(0,0))],
        '3': [((0,1),(1,1)), ((1,1),(0.2,0.6)), ((0.2,0.6),(1,0.3)), ((1,0.3),(0,0))],
        '4': [((0,1),(0,0.4)), ((1,1),(0,0.4)), ((1,1),(1,0))],
        '5': [((1,1),(0,1)), ((0,1),(0,0.6)), ((0,0.6),(1,0.6)), ((1,0.6),(1,0)), ((1,0),(0,0))],
        '6': [((1,1),(0,1)), ((0,1),(0,0)), ((0,0),(1,0)), ((1,0),(1,0.6)), ((1,0.6),(0,0.6))],
        '7': [((0,1),(1,1)), ((1,1),(0,0))],
        '8': [((0,0),(1,0)), ((1,0),(1,1)), ((1,1),(0,1)), ((0,1),(0,0)), ((0,0.5),(1,0.5))],
        '9': [((1,0),(1,1)), ((1,1),(0,1)), ((0,1),(0,0.5)), ((0,0.5),(1,0.5))],
        '-': [((0,0.5),(1,0.5))],
        ' ': [],
    }

    text = (text or '').upper()
    points = []
    lines = []
    cursor_x = 0.0
    spacing = 0.25  # glyph spacing
    glyph_w = 1.0

    for ch in text:
        segments = glyphs.get(ch, [])
        start_idx = len(points)
        # Add points for this glyph
        for (x1, y1), (x2, y2) in segments:
            # Scale to glyph width and height
            p1 = [ (cursor_x + x1 * glyph_w) * scale, y1 * scale, 0 ]
            p2 = [ (cursor_x + x2 * glyph_w) * scale, y2 * scale, 0 ]
            points.append(p1)
            points.append(p2)
            # Each pair contributes one line between consecutive points
            lines.append([len(points)-2, len(points)-1])
        cursor_x += glyph_w + spacing

    # Convert to numpy arrays and translate to position
    if len(points) == 0:
        # Fallback: simple small sphere if text empty/unsupported
        return create_text_label_3d('', position, color=color, size=scale)

    pts = np.array(points)
    pts[:, 0] += position[0]
    pts[:, 1] += position[1]
    pts[:, 2] += position[2]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.paint_uniform_color(color)
    return ls

def get_bbox_top_center(bbox_tensor):
    """
    Gets the top center position of a 3D bounding box for label placement.
    
    Args:
        bbox_tensor: 7D bounding box [x, y, z, dx, dy, dz, heading]
    
    Returns:
        3D position [x, y, z] at the top center of the box
    """
    x, y, z, dx, dy, dz, heading = bbox_tensor
    # z in many outputs is bottom-center; top center is z + dz
    return [x, y, z + dz + 0.15]  # small offset above the box

def get_bbox_center(bbox_tensor):
    """
    Returns the center position of a 3D bounding box [x, y, z].
    Useful for placing compact markers without height bias.
    """
    x, y, z, dx, dy, dz, heading = bbox_tensor
    # Convert bottom-center z -> geometric center for marker placement
    return [x, y, z + dz/2.0]

def load_kitti_gt_labels(label_file):
    """
    Loads KITTI-style ground truth labels from a .txt file.
    Only loads 'Car', 'Van', 'Truck' and converts to 7D bbox format.
    (h, w, l, x, y, z, yaw) -> (x_cam, y_cam, z_cam, l, w, h, yaw_lidar)
    
    Returns: List of 7D bbox tensors [x, y, z, l, w, h, yaw] in LiDAR coords.
    """
    gt_bboxes = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            obj_type = parts[0]
            
            # We only care about objects for detection
            if obj_type not in ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']:
                continue
                
            # h, w, l (dimensions in camera_coords)
            h, w, l = map(float, parts[8:11])
            
            # x, y, z (center in camera_coords)
            x_cam, y_cam, z_cam = map(float, parts[11:14])
            
            # yaw (in camera_coords)
            yaw_cam = float(parts[14])
            
            # Convert camera-coord box to LiDAR-coord box
            # 1. (x, y, z, l, w, h, yaw)_cam
            # 2. (x, y, z, l, w, h, yaw)_lidar
            # In mmdet3d (for KITTI):
            #   - x_lidar = z_cam
            #   - y_lidar = -x_cam
            #   - z_lidar = -y_cam + h/2 (to get center)
            #   - l_lidar = l
            #   - w_lidar = w
            #   - h_lidar = h
            #   - yaw_lidar = -yaw_cam - pi/2
            
            x_lidar = z_cam
            y_lidar = -x_cam
            # KITTI 'y_cam' is the bottom-center in camera coords; using -y_cam aligns
            # GT centers with LiDAR z without adding h/2 offset, which can bias height.
            z_lidar = -y_cam
            yaw_lidar = -yaw_cam - (np.pi / 2.0)
            
            # We use [x, y, z, l, w, h, yaw]
            gt_bboxes.append(np.array([x_lidar, y_lidar, z_lidar, l, w, h, yaw_lidar]))
            
    return gt_bboxes

def read_kitti_calib(calib_file):
    """
    Reads a KITTI calibration file and returns the P2 (camera)
    and Velo-to-Cam (R0_rect, Tr_velo_to_cam) matrices.
    """
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.strip().split(' ')])

    # Get P2 (projection matrix for cam 2)
    P2 = calib['P2'].reshape(3, 4)
    
    # Get Velo-to-Cam0 transform
    Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    
    # Get R0_rect (rectification matrix)
    R0_rect = calib['R0_rect'].reshape(3, 3)
    R0_rect = np.hstack([R0_rect, np.zeros((3, 1))])
    R0_rect = np.vstack([R0_rect, [0, 0, 0, 1]])

    # Velo -> Cam0 -> CamRect -> Image
    # We need Velo -> CamRect
    Tr_velo_to_rect = R0_rect @ Tr_velo_to_cam
    
    return P2, Tr_velo_to_rect

def project_lidar_to_image(points_lidar, P2, Tr_velo_to_rect):
    """
    Projects 3D points from LiDAR coordinates to 2D image coordinates.
    """
    # Add homogeneous coordinate
    points_lidar_hom = np.hstack((points_lidar, np.ones((points_lidar.shape[0], 1))))
    
    # Transform to rectified camera coordinates
    points_cam_rect = (Tr_velo_to_rect @ points_lidar_hom.T).T
    
    # Filter points in front of camera
    in_front = points_cam_rect[:, 2] > 0
    points_cam_rect_in_front = points_cam_rect[in_front]
    
    # Project to image plane
    points_cam_rect_hom = np.hstack((points_cam_rect_in_front[:, :3], np.ones((points_cam_rect_in_front.shape[0], 1))))
    points_img_hom = (P2 @ points_cam_rect_hom.T).T
    
    # Normalize by z
    points_img = points_img_hom[:, :2] / points_img_hom[:, 2, np.newaxis]
    
    # Return only points that are in front
    return points_img, in_front

def draw_projected_boxes_on_image(image_path, calib_path, pred_bboxes_3d, gt_bboxes_3d, out_path,
                                  pred_labels=None, class_names=None):
    """
    Loads an image, reads calibration, projects 3D boxes (pred and gt),
    overlays predicted class labels, and saves the visualized image.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"  > Warning: Could not load image {image_path}. Skipping 2D vis.")
            return
        P2, Tr_velo_to_rect = read_kitti_calib(calib_path)
    except Exception as e:
        print(f"  > Warning: Could not read image or calib file. Skipping 2D vis. {e}")
        return

    def _draw_boxes(bboxes, color, labels=None, cls_names=None):
        for idx, bbox in enumerate(bboxes):
            center = np.array(bbox[:3], dtype=float)
            extent = np.array(bbox[3:6], dtype=float)
            yaw = float(bbox[6])
            center[2] = center[2] + extent[2] / 2.0
            R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
            o3d_bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)

            corners_3d = np.asarray(o3d_bbox.get_box_points())
            points_2d, mask = project_lidar_to_image(corners_3d, P2, Tr_velo_to_rect)

            ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_bbox)
            edges = np.asarray(ls.lines)

            # Map original 8-corner indices to filtered 2D indices
            true_idx = np.where(mask)[0]
            orig_to_vis = {orig: vis for vis, orig in enumerate(true_idx)}

            points_2d = points_2d.astype(np.int32)
            for i, j in edges:
                if mask[i] and mask[j]:
                    vi = orig_to_vis.get(i, None)
                    vj = orig_to_vis.get(j, None)
                    if vi is not None and vj is not None:
                        cv2.line(img, tuple(points_2d[vi]), tuple(points_2d[vj]), color, 2)

            # Overlay class label text near top-left of visible 2D bbox
            if labels is not None and cls_names is not None and isinstance(labels, (list, np.ndarray)):
                # points_2d already contains only visible points due to masking
                visible_pts = points_2d
                if visible_pts.shape[0] > 0 and idx < len(labels):
                    x_min = int(np.min(visible_pts[:, 0]))
                    y_min = int(np.min(visible_pts[:, 1]))
                    try:
                        lid = int(labels[idx])
                    except Exception:
                        lid = None
                    label_text = cls_names[lid] if (lid is not None and 0 <= lid < len(cls_names)) else 'OBJ'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    bg_color = (int(color[0]), int(color[1]), int(color[2]))
                    cv2.rectangle(img,
                                  (x_min, max(0, y_min - th - 6)),
                                  (x_min + tw + 6, max(0, y_min)),
                                  bg_color,
                                  -1)
                    cv2.putText(img, label_text,
                                (x_min + 3, max(0, y_min - 3)),
                                font, font_scale,
                                (255, 255, 255), thickness, cv2.LINE_AA)

    # Draw Ground Truth boxes (Green)
    _draw_boxes(gt_bboxes_3d, (0, 255, 0))

    # Draw Predicted boxes (Red) with labels
    _draw_boxes(pred_bboxes_3d, (0, 0, 255), labels=pred_labels, cls_names=class_names)

    cv2.imwrite(out_path, img)
    print(f"  > Saved 2D visualization: {out_path}")

def visualize_with_open3d(lidar_file, predictions_dict, gt_bboxes, out_dir, basename, 
                          headless=False, img_file=None, calib_file=None):
    """
    Visualizes the point cloud and predicted/gt boxes using Open3D with enhanced features.
    Saves to .ply in headless mode, otherwise shows an interactive window.
    
    Args:
        lidar_file: Path to LiDAR point cloud file
        predictions_dict: Dictionary containing prediction results
        gt_bboxes: List of ground truth 3D bounding boxes
        out_dir: Output directory for saving files
        basename: Base name for output files
        headless: Whether to run in headless mode
        img_file: Optional path to corresponding image file
        calib_file: Optional path to calibration file
    """
    # Load the point cloud (N, 4)
    points = load_lidar_file(lidar_file)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Color points by height with high contrast colors (blue to red)
    pcd_colors = color_points_by_height(points)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    
    # Get predicted boxes and labels
    pred_bboxes_list = predictions_dict['bboxes_3d']
    pred_bboxes_tensor = np.array(pred_bboxes_list)
    
    # Get predicted labels if available
    pred_labels = predictions_dict.get('labels_3d', [])
    pred_scores = predictions_dict.get('scores_3d', [])
    
    # Create geometries list starting with point cloud
    geometries = [pcd]
    
    # Add compact coordinate frame at origin (smaller to avoid overflow)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geometries.append(coordinate_frame)
    
    # Create geometries for predicted boxes (Green)
    pred_line_sets = []
    pred_text_line_sets = []
    # Resolve class names if provided in metainfo; fallback to KITTI classes
    metainfo = predictions_dict.get('metainfo', {}) if isinstance(predictions_dict, dict) else {}
    class_names = metainfo.get('classes', None)
    if class_names is None:
        class_names = ['Car', 'Pedestrian', 'Cyclist']

    for i, bbox in enumerate(pred_bboxes_tensor):
        bbox_lines = create_open3d_bbox(bbox, color=[0.0, 1.0, 0.0])  # Green
        pred_line_sets.append(bbox_lines)
        geometries.append(bbox_lines)
        # Center marker: single green dot for predictions
        center_pos = get_bbox_center(bbox)
        pred_center = create_text_label_3d('', center_pos, color=[0.0, 1.0, 0.0], size=0.14)
        geometries.append(pred_center)

        # Add class label text at top center of box
        cls_id = None
        if isinstance(pred_labels, (list, np.ndarray)) and i < len(pred_labels):
            try:
                cls_id = int(pred_labels[i])
            except Exception:
                cls_id = None
        cls_name = class_names[cls_id] if (cls_id is not None and 0 <= cls_id < len(class_names)) else 'OBJ'
        top_pos = get_bbox_top_center(bbox)
        text_ls = create_text_stroke_label(cls_name, top_pos, color=[1.0, 1.0, 1.0], scale=0.6)
        geometries.append(text_ls)
        pred_text_line_sets.append(text_ls)
    
    # Create geometries for ground truth boxes (Red)
    gt_line_sets = []
    for i, bbox in enumerate(gt_bboxes):
        bbox_lines = create_open3d_bbox(bbox, color=[1.0, 0.0, 0.0])  # Red
        gt_line_sets.append(bbox_lines)
        geometries.append(bbox_lines)
        # Center marker: single red dot for GT
        gt_center = get_bbox_center(bbox)
        gt_center_marker = create_text_label_3d('', gt_center, color=[1.0, 0.0, 0.0], size=0.12)
        geometries.append(gt_center_marker)

# Generate 2D visualization if image and calibration data are provided
    if img_file and calib_file:
        try:
            img_2d_vis_path = Path(out_dir) / f"{basename}_2d_vis.png"
            draw_projected_boxes_on_image(img_file, calib_file, pred_bboxes_tensor, gt_bboxes, str(img_2d_vis_path), pred_labels=pred_labels, class_names=class_names)
        except Exception as e:
            print(f"  > Warning: Could not generate 2D visualization. {e}")
    
    if headless:
        print(f"  > Headless mode. Saving to .ply files in {out_dir}")
        pcd_file = Path(out_dir) / f"{basename}_points.ply"
        axes_file = Path(out_dir) / f"{basename}_axes.ply"
        pred_bbox_file = Path(out_dir) / f"{basename}_pred_bboxes.ply"
        pred_label_file = Path(out_dir) / f"{basename}_pred_labels.ply"
        gt_bbox_file = Path(out_dir) / f"{basename}_gt_bboxes.ply"
        
        o3d.io.write_point_cloud(str(pcd_file), pcd)
        
        # Save coordinate frame mesh
        o3d.io.write_triangle_mesh(str(axes_file.with_suffix('.ply')), coordinate_frame)
        
        # Save bounding boxes (combine into single LineSet for each group)
        if len(pred_line_sets) > 0:
            combined_pred = combine_line_sets(pred_line_sets, color=[0.0, 1.0, 0.0])
            o3d.io.write_line_set(str(pred_bbox_file), combined_pred)
        if len(gt_line_sets) > 0:
            combined_gt = combine_line_sets(gt_line_sets, color=[1.0, 0.0, 0.0])
            o3d.io.write_line_set(str(gt_bbox_file), combined_gt)
            
        print(f"  > Saved points: {pcd_file}")
        print(f"  > Saved coordinate axes: {axes_file}")
        if len(pred_bboxes_tensor) > 0:
            print(f"  > Saved pred bboxes: {pred_bbox_file}")
        if len(gt_bboxes) > 0:
            print(f"  > Saved gt bboxes: {gt_bbox_file}")
        # Save predicted top text labels in headless mode
        if len(pred_text_line_sets) > 0:
            combined_text = combine_line_sets(pred_text_line_sets, color=[1.0, 1.0, 1.0])
            o3d.io.write_line_set(str(pred_label_file), combined_text)
    else:
        print(f"  > Displaying Open3D visualization for {basename}...")
        print(f"  > Point cloud colored with turbo colormap (rainbow-like, high contrast)")
        print(f"  > Coordinate axes with arrows: X=Red, Y=Green, Z=Blue")
        print(f"  > Predicted boxes: Green, Ground truth boxes: Red")
        print(f"  > Markers: Green pred center, Red GT center, White top text")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Enhanced 3D Visualization: {basename}",
            width=1400,
            height=900
        )

def find_matching_file(basename, directory, extensions):
    """
    Find a file with the given basename and one of the given extensions in the directory.
    
    Args:
        basename: Base filename without extension
        directory: Directory to search in (can be None)
        extensions: List of extensions to try (e.g., ['.png', '.jpg'])
    
    Returns:
        Full path to matching file, or None if not found
    """
    if not directory or not os.path.isdir(directory):
        return None
        
    for ext in extensions:
        candidate = os.path.join(directory, basename + ext)
        if os.path.isfile(candidate):
            return candidate
    return None

def build_kitti_input_list(base_folder, frame_number=None):
    """
    Build input list for KITTI dataset structure.
    
    Args:
        base_folder (str): Base KITTI dataset folder
        frame_number (str): Specific frame number (e.g., '000008') or '-1' for all frames
        
    Returns:
        list: List of input dictionaries with KITTI data paths
    """
    import glob
    
    # KITTI folder structure
    velodyne_dir = os.path.join(base_folder, 'velodyne')
    image_dir = os.path.join(base_folder, 'image_2')
    calib_dir = os.path.join(base_folder, 'calib')
    label_dir = os.path.join(base_folder, 'label_2')
    
    # Check if required directories exist
    if not os.path.exists(velodyne_dir):
        raise ValueError(f"KITTI velodyne directory not found: {velodyne_dir}")
    
    inputs_list = []
    
    if frame_number == '-1' or frame_number is None:
        # Get all frames
        velodyne_files = sorted(glob.glob(os.path.join(velodyne_dir, '*.bin')))
        frame_numbers = [os.path.splitext(os.path.basename(f))[0] for f in velodyne_files]
    else:
        # Get specific frame
        frame_numbers = [frame_number]
    
    for frame_num in frame_numbers:
        # Build paths for this frame
        velodyne_file = os.path.join(velodyne_dir, f'{frame_num}.bin')
        image_file = os.path.join(image_dir, f'{frame_num}.png')
        calib_file = os.path.join(calib_dir, f'{frame_num}.txt')
        label_file = os.path.join(label_dir, f'{frame_num}.txt')
        
        # Check if velodyne file exists (required)
        if not os.path.exists(velodyne_file):
            print(f"Warning: Velodyne file not found for frame {frame_num}: {velodyne_file}")
            continue
            
        input_dict = {
            'points': velodyne_file,
            'img': image_file if os.path.exists(image_file) else None,
            'calib': calib_file if os.path.exists(calib_file) else None,
            'gt_label': label_file if os.path.exists(label_file) else None,
            'frame_id': frame_num
        }
        
        inputs_list.append(input_dict)
        
    print(f"Found {len(inputs_list)} KITTI frames to process")
    return inputs_list


def build_waymokitti_input_list(base_folder, frame_number=None):
    """
    Build input list for WaymoKITTI dataset structure (generated by waymo2kitti.py).
    
    Args:
        base_folder (str): Base WaymoKITTI dataset folder
        frame_number (str): Specific frame number or '-1' for all frames
        
    Returns:
        list: List of input dictionaries with WaymoKITTI data paths
    """
    import glob
    
    # WaymoKITTI folder structure (similar to KITTI but may have different naming)
    velodyne_dir = os.path.join(base_folder, 'velodyne')
    image_dir = os.path.join(base_folder, 'image_0')  # WaymoKITTI often uses image_0
    calib_dir = os.path.join(base_folder, 'calib')
    label_dir = os.path.join(base_folder, 'label_0')  # WaymoKITTI often uses label_0
    
    # Fallback to standard KITTI naming if waymo-specific doesn't exist
    if not os.path.exists(image_dir):
        image_dir = os.path.join(base_folder, 'image_2')
    if not os.path.exists(label_dir):
        label_dir = os.path.join(base_folder, 'label_2')
    
    # Check if required directories exist
    if not os.path.exists(velodyne_dir):
        raise ValueError(f"WaymoKITTI velodyne directory not found: {velodyne_dir}")
    
    inputs_list = []
    
    if frame_number == '-1' or frame_number is None:
        # Get all frames
        velodyne_files = sorted(glob.glob(os.path.join(velodyne_dir, '*.bin')))
        frame_numbers = [os.path.splitext(os.path.basename(f))[0] for f in velodyne_files]
    else:
        # Get specific frame
        frame_numbers = [frame_number]
    
    for frame_num in frame_numbers:
        # Build paths for this frame
        velodyne_file = os.path.join(velodyne_dir, f'{frame_num}.bin')
        image_file = os.path.join(image_dir, f'{frame_num}.png')
        calib_file = os.path.join(calib_dir, f'{frame_num}.txt')
        label_file = os.path.join(label_dir, f'{frame_num}.txt')
        
        # Check if velodyne file exists (required)
        if not os.path.exists(velodyne_file):
            print(f"Warning: Velodyne file not found for frame {frame_num}: {velodyne_file}")
            continue
            
        input_dict = {
            'points': velodyne_file,
            'img': image_file if os.path.exists(image_file) else None,
            'calib': calib_file if os.path.exists(calib_file) else None,
            'gt_label': label_file if os.path.exists(label_file) else None,
            'frame_id': frame_num
        }
        
        inputs_list.append(input_dict)
        
    print(f"Found {len(inputs_list)} WaymoKITTI frames to process")
    return inputs_list


def build_input_dict(primary_file, modality, img_dir, calib_dir, gt_label_dir):
    """
    Build input dictionary for a single sample, finding matching files in provided directories.
    
    Args:
        primary_file: Path to the primary input file (LiDAR .bin or image)
        modality: 'lidar', 'mono', or 'multi-modal'
        img_dir: Directory containing image files (can be None)
        calib_dir: Directory containing calibration files (can be None)
        gt_label_dir: Directory containing ground truth label files (can be None)
    
    Returns:
        Dictionary with input file paths
    """
    basename = Path(primary_file).stem
    input_dict = {}

    if modality == 'mono':
        input_dict['img'] = str(primary_file)
    else:
        # lidar or multi-modal
        input_dict['points'] = str(primary_file)
        
    # --- 1. Find matching image file ---
    img_exts = ['.png', '.jpg', '.jpeg']
    if img_dir and os.path.isdir(img_dir):
        img_file = find_matching_file(basename, img_dir, img_exts)
        if img_file:
            input_dict['img'] = img_file
        elif modality == 'multi-modal':
            print(f"Warning: --img-dir provided, but no matching image for {basename} found.")
    elif img_dir and not os.path.isdir(img_dir):
        print(f"Warning: --img-dir '{img_dir}' is not a valid directory.")

    # --- 2. Find matching calibration file ---
    calib_exts = ['.txt']
    if calib_dir and os.path.isdir(calib_dir):
        calib_file = find_matching_file(basename, calib_dir, calib_exts)
        if calib_file:
            input_dict['calib'] = calib_file
        else:
            print(f"Warning: --calib-dir provided, but no matching calib file for {basename} found.")
    elif calib_dir and not os.path.isdir(calib_dir):
        print(f"Warning: --calib-dir '{calib_dir}' is not a valid directory.")

    # --- 3. Find matching ground truth label file ---
    gt_exts = ['.txt']
    if gt_label_dir and os.path.isdir(gt_label_dir):
        gt_file = find_matching_file(basename, gt_label_dir, gt_exts)
        if gt_file:
            input_dict['gt_label'] = gt_file
        else:
            print(f"Warning: --gt-label-dir provided, but no matching label for {basename} found.")
    elif gt_label_dir and not os.path.isdir(gt_label_dir):
        print(f"Warning: --gt-label-dir '{gt_label_dir}' is not a valid directory.")

    return input_dict

# --- Defaults ---
DEFAULT_MODEL = '/Developer/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
#DEFAULT_CHECKPOINT = 'https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_ktti-3d-car_20220331_134606-d42d15ed.pth'
DEFAULT_CHECKPOINT = '/Developer/mmdetection3d/modelzoos/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
#DEFAULT_INPUT = '/Developer/mmdetection3d/demo/data/kitti/000008.bin'
DEFAULT_INPUT = '/data/Datasets/kitti/training/velodyne/000008.bin'
DEFAULT_LABEL = '/data/Datasets/kitti/training/label_2/000008.txt'
DEFAULT_CALIB = '/data/Datasets/kitti/training/calib/000008.txt'
DEFAULT_IMG = '/data/Datasets/kitti/training/image_2/000008.png'
# --- End Defaults ---

def main(args):
    # --- 1. Initialize Model ---
    print(f"Initializing {args.modality} inferencer...")
    
    model_path = args.model
    checkpoint_path = args.checkpoint
    
    # --- New Logic for default checkpoint ---
    if args.model == DEFAULT_MODEL and args.checkpoint is None:
        print(f"Using default model, setting default checkpoint: {DEFAULT_CHECKPOINT}")
        checkpoint_path = DEFAULT_CHECKPOINT
    # --- End New Logic ---
    
    # Handle auto-download for model names
    if not os.path.isfile(args.model):
        if not checkpoint_path:
            checkpoint_path = None
        print(f"Loading model '{args.model}' with checkpoint: {checkpoint_path or 'auto-download'}")
    else:
        if not checkpoint_path:
            print(f"Error: --checkpoint is required when 'model' ({args.model}) is a config file and not the default.")
            exit()
        print(f"Loading local model from config: {args.model}")

    # Select the correct inferencer class
    if args.modality == 'lidar':
        InferencerClass = LidarDet3DInferencer
    elif args.modality == 'mono':
        InferencerClass = MonoDet3DInferencer
    elif args.modality == 'multi-modal':
        InferencerClass = MultiModalityDet3DInferencer
    else:
        print(f"Error: Unknown modality '{args.modality}'")
        exit()
        
    inferencer = InferencerClass(
        model_path,
        checkpoint_path,
        device=args.device
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    is_headless = args.headless or not os.environ.get('DISPLAY')
    if is_headless:
        print("Running in headless mode. Visualizations will be saved to files.")
    
    # --- 2. Gather all inputs based on dataset mode ---
    inputs_list = []
    
    if args.dataset == 'kitti':
        print(f"Using KITTI dataset mode with base folder: {args.input_path}")
        if not os.path.isdir(args.input_path):
            print(f"Error: KITTI base folder does not exist: {args.input_path}")
            return
        inputs_list = build_kitti_input_list(args.input_path, args.frame_number)
        
    elif args.dataset == 'waymokitti':
        print(f"Using WaymoKITTI dataset mode with base folder: {args.input_path}")
        if not os.path.isdir(args.input_path):
            print(f"Error: WaymoKITTI base folder does not exist: {args.input_path}")
            return
        inputs_list = build_waymokitti_input_list(args.input_path, args.frame_number)
        
    else:  # args.dataset == 'any'
        print("Using manual path mode (any dataset)")
        if os.path.isfile(args.input_path):
            inputs_list.append(
                build_input_dict(args.input_path, args.modality, args.img_dir, args.calib_dir, args.gt_label_dir)
            )
        elif os.path.isdir(args.input_path):
            if args.modality == 'mono':
                file_exts = ('.png', '.jpg', '.jpeg')
            else:
                file_exts = ('.bin', '.ply', '.pcd')
                
            print(f"Scanning folder: {args.input_path}")
            for fname in sorted(os.listdir(args.input_path)):
                if fname.lower().endswith(file_exts):
                    primary_file = os.path.join(args.input_path, fname)
                    inputs_list.append(
                        build_input_dict(primary_file, args.modality, args.img_dir, args.calib_dir, args.gt_label_dir)
                    )
        else:
            print(f"Error: Input path does not exist: {args.input_path}")
            return

    if not inputs_list:
        print("Error: No valid input files found.")
        return
        
    print(f"Found {len(inputs_list)} samples to infer.")

    # --- 3. Run Inference & Visualize ---
    for single_input in inputs_list:
        
        # Determine the basename and primary file based on dataset mode
        if args.dataset in ['kitti', 'waymokitti']:
            basename = single_input.get('frame_id', 'unknown_frame')
            primary_file = single_input['points']
        else:
            primary_input_key = 'img' if args.modality == 'mono' else 'points'
            primary_file = single_input[primary_input_key]
            basename = Path(primary_file).stem
            
        print(f"\nRunning inference on input: {basename}")

        # Load GT labels if available
        gt_bboxes_3d = []
        if 'gt_label' in single_input:
            try:
                gt_bboxes_3d = load_kitti_gt_labels(single_input['gt_label'])
            except Exception as e:
                print(f"  > Warning: Could not load GT labels. {e}")
        
        # Prepare input for inferencer based on dataset mode
        # Pass the full dict so inferencer can use all available info
        inferencer_input = single_input
        
        # Run inference
        results_dict = inferencer(
            inferencer_input,
            show=False,
            out_dir=args.out_dir,
            pred_score_thr=args.score_thr
        )
        
        pred_dict = results_dict['predictions'][0]
        pred_bboxes_3d = np.array(pred_dict['bboxes_3d'])

        # Save the raw predictions (JSON)
        pred_path = Path(args.out_dir) / f"{basename}_predictions.json"
        print(f"  > Saving raw predictions to {pred_path}")
        try:
            import json
            serializable_pred_data = {}
            for k, v in pred_dict.items():
                if isinstance(v, np.ndarray):
                    serializable_pred_data[k] = v.tolist()
                else:
                    serializable_pred_data[k] = v
            with open(pred_path, 'w') as f:
                json.dump(serializable_pred_data, f, indent=2)
        except Exception as e:
            print(f"  > Warning: Could not save prediction JSON. {e}")

        # --- Generate 2D Visualization (if img and calib are available) ---
        if 'img' in single_input and 'calib' in single_input:
            img_2d_vis_path = Path(args.out_dir) / f"{basename}_2d_vis.png"
            draw_projected_boxes_on_image(
                single_input['img'],
                single_input['calib'],
                pred_bboxes_3d,
                gt_bboxes_3d,
                str(img_2d_vis_path)
            )

        # --- Generate 3D Visualization ---
        if args.modality != 'mono':
            # Determine lidar file path based on dataset mode
            # Use the 'points' key across all modes
            lidar_file = single_input['points']
            
            # Pass image and calibration files if available for enhanced visualization
            img_file = single_input.get('img', None)
            calib_file = single_input.get('calib', None)
            
            visualize_with_open3d(
                lidar_file,
                pred_dict,
                gt_bboxes_3d,
                args.out_dir,
                basename,
                headless=is_headless,
                img_file=img_file,
                calib_file=calib_file
            )
        else:
            print("  > Monocular model. Skipping Open3D visualization.")
            
    print(f"\nInference complete. Results saved in {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMDetection3D Inference Script")
    
    # Dataset mode selection
    parser.add_argument('--dataset', type=str, default='kitti',
                        choices=['any', 'kitti', 'waymokitti'],
                        help="Dataset mode: 'any' (manual paths), 'kitti' (KITTI dataset structure), 'waymokitti' (Waymo2KITTI structure)")
    
    parser.add_argument('--model', type=str, 
                        default=DEFAULT_MODEL,
                        help="Model name (e.g., 'pointpillars_kitti') or path to config file.")
    
    # Dataset-specific arguments
    parser.add_argument('--input-path', type=str, 
                        default="/data/Datasets/kitti/training/",
                        help="Path to input. For 'any': LiDAR file/folder or image file/folder. For 'kitti'/'waymokitti': dataset base folder.")
    parser.add_argument('--frame-number', type=str, default='000008',
                        help="Frame number for KITTI/WaymoKITTI datasets (e.g., '000008'). Use -1 for all frames in dataset.")
    
    parser.add_argument('--out-dir', type=str, 
                        default='./inference_results',
                        help="Directory to save prediction results and visualizations.")
    
    parser.add_argument('--modality', type=str, default='lidar',
                        choices=['lidar', 'mono', 'multi-modal'],
                        help="Modality of the model (e.g., 'lidar', 'mono', 'multi-modal').")
    
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help="(Optional) Path or URL to checkpoint. If 'model' is a name, will auto-download if not provided."
                             f" Defaults to {DEFAULT_CHECKPOINT} if default model is used.")
    
    # Manual path args (used only with --dataset=any)
    parser.add_argument('--img-dir', type=str, default=None,
                        help="(Optional) Directory of camera images. Only used with --dataset=any.")
    parser.add_argument('--calib-dir', type=str, default=None,
                        help="(Optional) Directory of calibration files (e.g., KITTI-style .txt). Only used with --dataset=any.")
    parser.add_argument('--gt-label-dir', type=str, default=None,
                        help="(Optional) Directory of ground truth label files (e.g., KITTI-style .txt). Only used with --dataset=any.")
    
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help="Score threshold for filtering predictions.")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="Device to use for inference (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument('--headless', action='store_true',
                        help="Run in headless mode. Will save visualizations to .ply files "
                             "instead of opening an interactive window.")
    
    args = parser.parse_args()
    
    # Update default paths from relative to absolute
    # (Assuming your defaults are relative to a project root)
    # If your paths are already absolute, you can remove this block.
    if not Path(args.model).is_absolute() and args.model == DEFAULT_MODEL:
        args.model = str(Path.cwd() / args.model)
    if not Path(args.checkpoint).is_absolute() and args.checkpoint == DEFAULT_CHECKPOINT:
        args.checkpoint = str(Path.cwd() / args.checkpoint)
    if not Path(args.input_path).is_absolute() and args.input_path == DEFAULT_INPUT:
        args.input_path = str(Path.cwd() / args.input_path)
    
    main(args)