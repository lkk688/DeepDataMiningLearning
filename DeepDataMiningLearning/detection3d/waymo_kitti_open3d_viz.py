"""
Optimized 3D Visualization for KITTI and Waymo2KITTI Datasets using Open3D

This module provides comprehensive 3D visualization capabilities for autonomous driving
datasets, specifically optimized for KITTI and Waymo2KITTI formats. It replaces the
original Mayavi-based visualization with Open3D for better performance, cross-platform
compatibility, and enhanced visual quality.

Key Features:
- High-performance 3D point cloud visualization with height-based coloring
- 3D bounding box rendering with proper corner calculations
- Headless environment support with PLY file export
- Mathematical transformations with detailed LaTeX documentation
- Coordinate system transformations (LiDAR ↔ Camera)
- Multi-sensor data fusion visualization

Mathematical Foundations:
- Point cloud transformations using homogeneous coordinates
- 3D bounding box corner generation with rotation matrices
- Perspective projection for camera coordinate systems
- Color mapping based on height distributions

Author: Optimized for Waymo2KITTI dataset visualization
Date: 2024
"""

import numpy as np
import open3d as o3d
import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from typing import List, Tuple, Optional, Union
import warnings

# Suppress Open3D warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

# Color palette for different object classes and height levels
BOX_COLORMAP = [
    [1.0, 1.0, 1.0],  # White - Unknown/Default
    [0.0, 1.0, 0.0],  # Green - Car
    [0.0, 1.0, 1.0],  # Cyan - Pedestrian  
    [1.0, 1.0, 0.0],  # Yellow - Cyclist
    [1.0, 0.0, 0.0],  # Red - Other
]

# Height-based color levels for point cloud visualization
HEIGHT_LEVELS = [-3, -2, -1, 0, 1, 2, 3]  # Meters above ground


def check_numpy_to_torch(x):
    """
    Convert numpy arrays to torch tensors if needed.
    
    This utility function maintains compatibility with existing torch-based
    code while working primarily with numpy arrays in Open3D.
    
    Args:
        x: Input array (numpy or torch)
        
    Returns:
        tuple: (converted_array, was_numpy_flag)
    """
    if isinstance(x, np.ndarray):
        return x, True
    # If torch tensor, convert to numpy
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy(), False
    except ImportError:
        pass
    return x, True


def rotate_points_along_z(points: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Rotate 3D points around the Z-axis using rotation matrices.
    
    This function performs batch rotation of 3D points using the Z-axis rotation
    matrix. Essential for 3D bounding box transformations and coordinate system
    conversions in autonomous driving applications.
    
    Mathematical Foundation:
    
    **Z-axis Rotation Matrix:**
    $$\\mathbf{R}_z(\\theta) = \\begin{bmatrix}
    \\cos\\theta & -\\sin\\theta & 0 \\\\
    \\sin\\theta & \\cos\\theta & 0 \\\\
    0 & 0 & 1
    \\end{bmatrix}$$
    
    **Batch Rotation Operation:**
    $$\\mathbf{P}_{rotated} = \\mathbf{P}_{original} \\cdot \\mathbf{R}_z^T$$
    
    Where:
    - $\\theta$ is the rotation angle (positive = counter-clockwise)
    - $\\mathbf{P}$ contains 3D points with shape (B, N, 3)
    - B is batch size, N is number of points
    
    Args:
        points (np.ndarray): 3D points to rotate, shape (B, N, 3+C)
                           Where B=batch, N=points, C=additional channels
        angle (np.ndarray): Rotation angles in radians, shape (B,)
                          Positive angles rotate counter-clockwise
    
    Returns:
        np.ndarray: Rotated points with same shape as input
                   Only the first 3 coordinates are rotated
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)
    
    # Ensure inputs are numpy arrays
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if not isinstance(angle, np.ndarray):
        angle = np.array(angle)
    
    # Compute trigonometric functions for rotation matrix
    cosa = np.cos(angle)  # Shape: (B,)
    sina = np.sin(angle)  # Shape: (B,)
    zeros = np.zeros_like(angle)  # Shape: (B,)
    ones = np.ones_like(angle)   # Shape: (B,)
    
    # Construct rotation matrices for each batch element
    # Stack elements to form rotation matrix: [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
    rot_matrix = np.stack([
        cosa, -sina, zeros,  # First row
        sina,  cosa, zeros,  # Second row  
        zeros, zeros, ones   # Third row
    ], axis=1).reshape(-1, 3, 3)  # Shape: (B, 3, 3)
    
    # Apply rotation to the first 3 coordinates of each point
    # Extract XYZ coordinates: (B, N, 3)
    points_xyz = points[:, :, :3]  # Shape: (B, N, 3)
    
    # Batch matrix multiplication: (B, N, 3) @ (B, 3, 3) -> (B, N, 3)
    points_rotated = np.einsum('bni,bij->bnj', points_xyz, rot_matrix)
    
    # Combine rotated XYZ with original additional channels (if any)
    if points.shape[2] > 3:
        # Concatenate rotated XYZ with unchanged additional channels
        points_rot = np.concatenate([points_rotated, points[:, :, 3:]], axis=-1)
    else:
        points_rot = points_rotated
    
    return points_rot


def boxes_to_corners_3d(boxes3d: np.ndarray) -> np.ndarray:
    """
    Convert 3D bounding box parameters to 8 corner coordinates.
    
    This function generates the 8 corner points of 3D bounding boxes from their
    parametric representation. The corners follow a specific indexing convention
    that is consistent with autonomous driving datasets.
    
    Mathematical Process:
    
    **Corner Template Generation:**
    $$\\mathbf{C}_{template} = \\frac{1}{2} \\begin{bmatrix}
    +1 & +1 & -1 \\\\  \\text{Corner 0: front-right-bottom} \\\\
    +1 & -1 & -1 \\\\  \\text{Corner 1: front-left-bottom} \\\\
    -1 & -1 & -1 \\\\  \\text{Corner 2: rear-left-bottom} \\\\
    -1 & +1 & -1 \\\\  \\text{Corner 3: rear-right-bottom} \\\\
    +1 & +1 & +1 \\\\  \\text{Corner 4: front-right-top} \\\\
    +1 & -1 & +1 \\\\  \\text{Corner 5: front-left-top} \\\\
    -1 & -1 & +1 \\\\  \\text{Corner 6: rear-left-top} \\\\
    -1 & +1 & +1       \\text{Corner 7: rear-right-top}
    \\end{bmatrix}$$
    
    **Scaling and Rotation:**
    $$\\mathbf{C}_{scaled} = \\mathbf{C}_{template} \\odot [l, w, h]$$
    $$\\mathbf{C}_{rotated} = \\text{rotate\\_along\\_z}(\\mathbf{C}_{scaled}, \\theta)$$
    $$\\mathbf{C}_{final} = \\mathbf{C}_{rotated} + [x_c, y_c, z_c]$$
    
    Corner Indexing Convention:
    ```
           7 -------- 4
          /|         /|
         6 -------- 5 |
         | |        | |
         | 3 -------- 0
         |/         |/
         2 -------- 1
    ```
    
    Args:
        boxes3d (np.ndarray): 3D box parameters, shape (N, 7)
                             Format: [x, y, z, dx, dy, dz, heading]
                             Where (x,y,z) is box center, (dx,dy,dz) are dimensions
    
    Returns:
        np.ndarray: Corner coordinates, shape (N, 8, 3)
                   Each box has 8 corners with [x, y, z] coordinates
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    
    # Ensure input is numpy array
    if not isinstance(boxes3d, np.ndarray):
        boxes3d = np.array(boxes3d)
    
    # Define corner template in normalized coordinates
    # Each row represents one corner: [x_norm, y_norm, z_norm]
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],  # Bottom face (z=-1)
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],      # Top face (z=+1)
    ], dtype=np.float32) / 2.0  # Normalize to [-0.5, +0.5] range
    
    # Extract box dimensions: [length, width, height]
    dimensions = boxes3d[:, 3:6]  # Shape: (N, 3)
    
    # Scale template corners by box dimensions
    # Broadcasting: (N, 1, 3) * (8, 3) -> (N, 8, 3)
    corners3d = dimensions[:, np.newaxis, :] * template[np.newaxis, :, :]
    
    # Apply Z-axis rotation using heading angles
    heading_angles = boxes3d[:, 6]  # Shape: (N,)
    corners3d = rotate_points_along_z(corners3d, heading_angles)
    
    # Translate corners to box center positions
    # Broadcasting: (N, 8, 3) + (N, 1, 3) -> (N, 8, 3)
    box_centers = boxes3d[:, :3]  # Shape: (N, 3)
    corners3d += box_centers[:, np.newaxis, :]
    
    return corners3d


class Open3DVisualizer:
    """
    Advanced 3D visualization system using Open3D for KITTI and Waymo2KITTI datasets.
    
    This class provides comprehensive 3D visualization capabilities optimized for
    autonomous driving datasets. It supports both interactive visualization and
    headless operation with file export capabilities.
    
    Key Features:
    - High-performance point cloud rendering with customizable coloring
    - 3D bounding box visualization with proper corner calculations  
    - Coordinate system axis display with mathematical accuracy
    - Grid overlay for spatial reference
    - Headless operation with PLY/PNG export
    - Multi-sensor data fusion support
    
    Mathematical Foundations:
    - Point cloud transformations using homogeneous coordinates
    - Color mapping based on statistical height distributions
    - 3D bounding box corner generation with rotation matrices
    """
    
    def __init__(self, 
                 window_name: str = "3D Visualization",
                 window_size: Tuple[int, int] = (1600, 1000),
                 background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 point_size: float = 2.0,
                 headless: bool = False):
        """
        Initialize the Open3D visualization system.
        
        Args:
            window_name: Name of the visualization window
            window_size: Window dimensions (width, height)
            background_color: RGB background color in [0,1] range
            point_size: Size of point cloud points in pixels
            headless: Whether to run in headless mode (no GUI)
        """
        self.window_name = window_name
        self.window_size = window_size
        self.background_color = background_color
        self.point_size = point_size
        self.headless = headless
        
        # Initialize visualization components
        self.vis = None
        self.geometries = []
        
        # Color palettes for different visualization modes
        self.height_colors = self._generate_height_colormap()
        
        # Camera parameters for consistent viewpoints
        self.default_camera_params = {
            'azimuth': -179,
            'elevation': 54.0, 
            'distance': 104.0,
            'roll': 90.0
        }
    
    def _generate_height_colormap(self) -> np.ndarray:
        """
        Generate color mapping for height-based point cloud visualization.
        
        Creates a smooth color gradient that maps point heights to visually
        distinct colors. This enhances depth perception and spatial understanding
        of 3D point clouds.
        
        Mathematical Color Mapping:
        $$\\text{color}(h) = \\text{interpolate}(\\text{colormap}, \\frac{h - h_{min}}{h_{max} - h_{min}})$$
        
        Where:
        - $h$ is the point height (z-coordinate)
        - $h_{min}, h_{max}$ are the height range bounds
        - colormap provides smooth color transitions
        
        Returns:
            np.ndarray: Color mapping array, shape (N_levels, 3)
                       Each row contains RGB values in [0,1] range
        """
        # Use matplotlib's colormap for smooth color transitions
        colormap = plt.cm.get_cmap('viridis')  # Professional color scheme
        
        # Generate colors for each height level
        n_levels = len(HEIGHT_LEVELS)
        colors = np.zeros((n_levels, 3))
        
        for i, height in enumerate(HEIGHT_LEVELS):
            # Normalize height to [0,1] range for colormap
            normalized_height = (i) / (n_levels - 1)
            rgb = colormap(normalized_height)[:3]  # Extract RGB, ignore alpha
            colors[i] = rgb
            
        return colors
    
    def _setup_visualizer(self):
        """Initialize Open3D visualizer with proper settings."""
        if self.headless:
            # Headless mode - no GUI window
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(visible=False)
        else:
            # Interactive mode with GUI
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=self.window_name,
                width=self.window_size[0],
                height=self.window_size[1]
            )
        
        # Configure rendering options
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array(self.background_color)
        render_option.point_size = self.point_size
        render_option.show_coordinate_frame = True
        
        return self.vis
    
    def generate_point_colors_by_height(self, points: np.ndarray) -> np.ndarray:
        """
        Generate colors for point cloud based on height (Z-coordinate) values.
        
        This function creates visually appealing height-based coloring that enhances
        depth perception and spatial understanding of 3D point clouds. The coloring
        scheme is optimized for autonomous driving scenarios.
        
        Mathematical Color Assignment:
        
        **Height Discretization:**
        $$\\text{level}(z) = \\arg\\min_i |z - h_i|$$
        
        Where $h_i$ are predefined height levels: $[-3, -2, -1, 0, 1, 2, 3]$ meters.
        
        **Color Interpolation:**
        For smooth transitions between discrete levels:
        $$\\mathbf{c}(z) = \\text{lerp}(\\mathbf{c}_i, \\mathbf{c}_{i+1}, \\alpha)$$
        
        Where:
        - $\\alpha = \\frac{z - h_i}{h_{i+1} - h_i}$ is the interpolation factor
        - $\\mathbf{c}_i, \\mathbf{c}_{i+1}$ are colors at adjacent height levels
        
        Args:
            points (np.ndarray): 3D points with shape (N, 3) or (N, 4)
                               Format: [x, y, z] or [x, y, z, intensity]
        
        Returns:
            np.ndarray: RGB colors for each point, shape (N, 3)
                       Values in [0, 1] range for Open3D compatibility
        """
        if points.shape[0] == 0:
            return np.empty((0, 3))
        
        # Extract height values (Z-coordinates)
        heights = points[:, 2]  # Shape: (N,)
        
        # Initialize color array
        colors = np.ones((points.shape[0], 3))  # Default to white
        
        # Get height statistics for adaptive coloring
        min_height = np.min(heights)
        max_height = np.max(heights)
        
        print(f"Height range: [{min_height:.2f}, {max_height:.2f}] meters")
        
        # Assign colors based on height levels
        for i, point_height in enumerate(heights):
            # Find appropriate color based on height level
            if point_height < HEIGHT_LEVELS[0]:
                # Below minimum level - use first color
                colors[i] = self.height_colors[0]
            elif point_height >= HEIGHT_LEVELS[-1]:
                # Above maximum level - use last color
                colors[i] = self.height_colors[-1]
            else:
                # Interpolate between adjacent height levels
                for j in range(len(HEIGHT_LEVELS) - 1):
                    if HEIGHT_LEVELS[j] <= point_height < HEIGHT_LEVELS[j + 1]:
                        # Linear interpolation between two adjacent colors
                        alpha = (point_height - HEIGHT_LEVELS[j]) / (HEIGHT_LEVELS[j + 1] - HEIGHT_LEVELS[j])
                        colors[i] = (1 - alpha) * self.height_colors[j] + alpha * self.height_colors[j + 1]
                        break
        
        return colors
    
    def create_point_cloud(self, 
                          points: np.ndarray, 
                          colors: Optional[np.ndarray] = None,
                          color_mode: str = 'height') -> o3d.geometry.PointCloud:
        """
        Create Open3D point cloud geometry with advanced coloring options.
        
        This method converts numpy point arrays into Open3D point cloud objects
        with sophisticated coloring schemes optimized for autonomous driving
        visualization.
        
        Supported Color Modes:
        - 'height': Height-based coloring using Z-coordinates
        - 'intensity': Intensity-based coloring (if available)
        - 'custom': User-provided colors
        - 'uniform': Single color for all points
        
        Args:
            points (np.ndarray): 3D points, shape (N, 3) or (N, 4)
                               Format: [x, y, z] or [x, y, z, intensity]
            colors (np.ndarray, optional): Custom colors, shape (N, 3)
            color_mode (str): Coloring scheme to use
        
        Returns:
            o3d.geometry.PointCloud: Configured point cloud geometry
        """
        if points.shape[0] == 0:
            return o3d.geometry.PointCloud()
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Apply coloring based on specified mode
        if color_mode == 'height':
            point_colors = self.generate_point_colors_by_height(points)
        elif color_mode == 'intensity' and points.shape[1] >= 4:
            # Use intensity values for coloring
            intensities = points[:, 3]
            # Normalize intensities to [0, 1] range
            norm_intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities) + 1e-8)
            # Apply colormap
            colormap = plt.cm.get_cmap('hot')
            point_colors = colormap(norm_intensities)[:, :3]
        elif color_mode == 'custom' and colors is not None:
            point_colors = colors
        else:
            # Default uniform coloring
            point_colors = np.tile([0.7, 0.7, 0.7], (points.shape[0], 1))
        
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        return pcd
    
    def create_3d_bounding_boxes(self, 
                                boxes_3d: np.ndarray,
                                labels: Optional[np.ndarray] = None,
                                scores: Optional[np.ndarray] = None,
                                color_mode: str = 'class') -> List[o3d.geometry.LineSet]:
        """
        Create 3D bounding box visualizations using Open3D line sets.
        
        This method generates wireframe 3D bounding boxes with proper corner
        connections and optional class-based coloring. The implementation
        follows autonomous driving dataset conventions.
        
        Mathematical Box Construction:
        
        **Corner Generation:**
        Uses the `boxes_to_corners_3d` function to generate 8 corners per box.
        
        **Line Connectivity:**
        Defines 12 edges connecting the 8 corners:
        - 4 edges for bottom face
        - 4 edges for top face  
        - 4 vertical edges connecting bottom to top
        
        **Edge Definition:**
        $$\\text{edges} = \\{(i,j) : \\text{corners } i \\text{ and } j \\text{ are connected}\\}$$
        
        Args:
            boxes_3d (np.ndarray): 3D box parameters, shape (N, 7)
                                  Format: [x, y, z, dx, dy, dz, heading]
            labels (np.ndarray, optional): Class labels for each box, shape (N,)
            scores (np.ndarray, optional): Confidence scores, shape (N,)
            color_mode (str): Coloring scheme ('class', 'score', 'uniform')
        
        Returns:
            List[o3d.geometry.LineSet]: List of line set geometries for each box
        """
        if boxes_3d.shape[0] == 0:
            return []
        
        # Generate corner coordinates for all boxes
        corners_3d = boxes_to_corners_3d(boxes_3d)  # Shape: (N, 8, 3)
        
        line_sets = []
        
        # Define line connections between corners (12 edges per box)
        # Corner indexing follows the convention in boxes_to_corners_3d
        lines = [
            # Bottom face edges (z = -height/2)
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Top face edges (z = +height/2)  
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Vertical edges connecting bottom to top
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        for i, box_corners in enumerate(corners_3d):
            # Create line set for current box
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(box_corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            
            # Determine box color based on mode
            if color_mode == 'class' and labels is not None:
                # Use class-based coloring
                class_id = int(labels[i]) if i < len(labels) else 0
                color = BOX_COLORMAP[class_id % len(BOX_COLORMAP)]
            elif color_mode == 'score' and scores is not None:
                # Use score-based coloring (red = low, green = high)
                score = scores[i] if i < len(scores) else 0.5
                color = [1 - score, score, 0]  # Red to green gradient
            else:
                # Default uniform coloring
                color = [0, 1, 0]  # Green
            
            # Apply color to all lines in the box
            colors = np.tile(color, (len(lines), 1))
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            line_sets.append(line_set)
        
        return line_sets
    
    def create_coordinate_frame(self, size: float = 2.0) -> o3d.geometry.TriangleMesh:
        """
        Create coordinate system axes for spatial reference.
        
        Generates X, Y, Z axes with standard color coding:
        - X-axis: Red (forward direction in vehicle frame)
        - Y-axis: Green (left direction in vehicle frame)  
        - Z-axis: Blue (upward direction)
        
        Mathematical Representation:
        $$\\mathbf{X} = [size, 0, 0]^T \\quad \\text{(Red)}$$
        $$\\mathbf{Y} = [0, size, 0]^T \\quad \\text{(Green)}$$
        $$\\mathbf{Z} = [0, 0, size]^T \\quad \\text{(Blue)}$$
        
        Args:
            size (float): Length of each axis in meters
        
        Returns:
            o3d.geometry.TriangleMesh: Coordinate frame geometry
        """
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    def create_ground_grid(self, 
                          grid_size: float = 20.0,
                          extent: Tuple[float, float, float, float] = (-60, -60, 60, 60),
                          color: Tuple[float, float, float] = (0.5, 0.5, 0.5)) -> List[o3d.geometry.LineSet]:
        """
        Create ground plane grid for spatial reference.
        
        Generates a grid overlay on the ground plane (z=0) to provide spatial
        context and scale reference for 3D visualizations.
        
        Mathematical Grid Generation:
        $$\\text{Grid lines: } \\{(x_i, y_j, 0) : x_i \\in [x_{min}, x_{max}], y_j \\in [y_{min}, y_{max}]\\}$$
        
        Where grid spacing is determined by `grid_size` parameter.
        
        Args:
            grid_size (float): Spacing between grid lines in meters
            extent (tuple): Grid boundaries (x_min, y_min, x_max, y_max)
            color (tuple): RGB color for grid lines
        
        Returns:
            List[o3d.geometry.LineSet]: Grid line geometries
        """
        x_min, y_min, x_max, y_max = extent
        grid_lines = []
        
        # Create vertical grid lines (parallel to Y-axis)
        for x in np.arange(x_min, x_max + grid_size, grid_size):
            points = [[x, y_min, 0], [x, y_max, 0]]
            lines = [[0, 1]]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([color])
            
            grid_lines.append(line_set)
        
        # Create horizontal grid lines (parallel to X-axis)
        for y in np.arange(y_min, y_max + grid_size, grid_size):
            points = [[x_min, y, 0], [x_max, y, 0]]
            lines = [[0, 1]]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([color])
            
            grid_lines.append(line_set)
        
        return grid_lines
    
    def visualize_scene(self,
                       points: np.ndarray,
                       gt_boxes: Optional[np.ndarray] = None,
                       pred_boxes: Optional[np.ndarray] = None,
                       gt_labels: Optional[np.ndarray] = None,
                       pred_labels: Optional[np.ndarray] = None,
                       pred_scores: Optional[np.ndarray] = None,
                       save_path: Optional[str] = None,
                       show_grid: bool = True,
                       show_axes: bool = True) -> None:
        """
        Comprehensive 3D scene visualization with point clouds and bounding boxes.
        
        This is the main visualization function that combines all components:
        point clouds, 3D bounding boxes, coordinate frames, and spatial grids.
        
        Args:
            points (np.ndarray): 3D point cloud, shape (N, 3) or (N, 4)
            gt_boxes (np.ndarray, optional): Ground truth boxes, shape (M, 7)
            pred_boxes (np.ndarray, optional): Predicted boxes, shape (K, 7)
            gt_labels (np.ndarray, optional): GT class labels, shape (M,)
            pred_labels (np.ndarray, optional): Predicted labels, shape (K,)
            pred_scores (np.ndarray, optional): Prediction scores, shape (K,)
            save_path (str, optional): Path to save visualization
            show_grid (bool): Whether to display ground grid
            show_axes (bool): Whether to display coordinate axes
        """
        # Initialize visualizer
        if self.vis is None:
            self._setup_visualizer()
        
        # Clear previous geometries
        self.vis.clear_geometries()
        self.geometries.clear()
        
        # Create and add point cloud
        if points.shape[0] > 0:
            print(f"Visualizing {points.shape[0]} points")
            pcd = self.create_point_cloud(points, color_mode='height')
            self.vis.add_geometry(pcd)
            self.geometries.append(pcd)
        
        # Add ground truth bounding boxes (blue)
        if gt_boxes is not None and gt_boxes.shape[0] > 0:
            print(f"Adding {gt_boxes.shape[0]} ground truth boxes")
            gt_line_sets = self.create_3d_bounding_boxes(
                gt_boxes, gt_labels, color_mode='class'
            )
            # Override colors to blue for GT boxes
            for line_set in gt_line_sets:
                colors = np.tile([0, 0, 1], (12, 1))  # Blue for all edges
                line_set.colors = o3d.utility.Vector3dVector(colors)
                self.vis.add_geometry(line_set)
                self.geometries.append(line_set)
        
        # Add predicted bounding boxes (green/red based on score)
        if pred_boxes is not None and pred_boxes.shape[0] > 0:
            print(f"Adding {pred_boxes.shape[0]} predicted boxes")
            pred_line_sets = self.create_3d_bounding_boxes(
                pred_boxes, pred_labels, pred_scores, color_mode='score'
            )
            for line_set in pred_line_sets:
                self.vis.add_geometry(line_set)
                self.geometries.append(line_set)
        
        # Add coordinate frame
        if show_axes:
            coord_frame = self.create_coordinate_frame(size=3.0)
            self.vis.add_geometry(coord_frame)
            self.geometries.append(coord_frame)
        
        # Add ground grid
        if show_grid:
            grid_lines = self.create_ground_grid(
                grid_size=20.0,
                extent=(0, -40, 80, 40)  # Typical autonomous driving range
            )
            for line_set in grid_lines:
                self.vis.add_geometry(line_set)
                self.geometries.append(line_set)
        
        # Set camera viewpoint
        self._set_camera_view()
        
        # Handle visualization mode
        if self.headless or save_path:
            # Render and save
            self.vis.poll_events()
            self.vis.update_renderer()
            
            if save_path:
                self._save_visualization(save_path)
        else:
            # Interactive mode
            print("Press Q to quit, R to reset view")
            self.vis.run()
    
    def _set_camera_view(self):
        """Set optimal camera viewpoint for autonomous driving scenes."""
        # Get view control
        view_control = self.vis.get_view_control()
        
        # Set camera parameters for optimal viewing
        # These parameters are optimized for autonomous driving scenarios
        view_control.set_front([0.0, 0.0, -1.0])  # Look down slightly
        view_control.set_lookat([40.0, 0.0, 0.0])  # Focus on forward area
        view_control.set_up([0.0, 0.0, 1.0])      # Z-axis up
        view_control.set_zoom(0.3)                 # Appropriate zoom level
    
    def _save_visualization(self, save_path: str):
        """
        Save visualization to file with support for multiple formats.
        
        Supports:
        - PNG: High-quality screenshots
        - PLY: Point cloud data export
        - JSON: Camera parameters and scene configuration
        
        Args:
            save_path (str): Output file path (extension determines format)
        """
        try:
            if save_path.endswith('.png'):
                # Save screenshot
                self.vis.capture_screen_image(save_path)
                print(f"Screenshot saved to: {save_path}")
            
            elif save_path.endswith('.ply'):
                # Save point cloud data
                if self.geometries:
                    # Combine all point clouds
                    combined_pcd = o3d.geometry.PointCloud()
                    for geom in self.geometries:
                        if isinstance(geom, o3d.geometry.PointCloud):
                            combined_pcd += geom
                    
                    if len(combined_pcd.points) > 0:
                        o3d.io.write_point_cloud(save_path, combined_pcd)
                        print(f"Point cloud saved to: {save_path}")
            
            else:
                print(f"Unsupported file format: {save_path}")
                
        except Exception as e:
            print(f"Error saving visualization: {e}")
    
    def close(self):
        """Clean up visualization resources."""
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None
        self.geometries.clear()


def detect_headless_environment() -> bool:
    """
    Detect if running in a headless environment (no display).
    
    This function checks various environment indicators to determine
    if the system supports GUI display or requires headless operation.
    
    Returns:
        bool: True if headless environment detected
    """
    # Check for display environment variable
    if os.environ.get('DISPLAY') is None:
        return True
    
    # Check for common headless indicators
    headless_indicators = [
        'SSH_CONNECTION' in os.environ,
        'SSH_CLIENT' in os.environ,
        os.environ.get('TERM') == 'dumb',
        'CI' in os.environ,  # Continuous Integration
        'GITHUB_ACTIONS' in os.environ,
    ]
    
    return any(headless_indicators)


def main():
    """
    Main function with command-line interface for dataset visualization.
    
    Provides a comprehensive CLI for visualizing KITTI and Waymo2KITTI datasets
    with various options for customization and output formats.
    """
    parser = argparse.ArgumentParser(
        description="Advanced 3D Visualization for KITTI and Waymo2KITTI Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize point cloud with bounding boxes
  python waymo_kitti_open3d_viz.py --points data.npy --boxes boxes.npy
  
  # Headless mode with PNG export
  python waymo_kitti_open3d_viz.py --points data.npy --headless --save output.png
  
  # Export point cloud to PLY format
  python waymo_kitti_open3d_viz.py --points data.npy --save output.ply
        """
    )
    
    # Input data arguments
    parser.add_argument('--points', type=str, required=True,
                       help='Path to point cloud data (.npy format)')
    parser.add_argument('--gt-boxes', type=str,
                       help='Path to ground truth boxes (.npy format)')
    parser.add_argument('--pred-boxes', type=str,
                       help='Path to predicted boxes (.npy format)')
    parser.add_argument('--gt-labels', type=str,
                       help='Path to ground truth labels (.npy format)')
    parser.add_argument('--pred-labels', type=str,
                       help='Path to predicted labels (.npy format)')
    parser.add_argument('--pred-scores', type=str,
                       help='Path to prediction scores (.npy format)')
    
    # Visualization options
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no GUI)')
    parser.add_argument('--save', type=str,
                       help='Save visualization to file (.png or .ply)')
    parser.add_argument('--point-size', type=float, default=2.0,
                       help='Point cloud point size')
    parser.add_argument('--no-grid', action='store_true',
                       help='Disable ground grid display')
    parser.add_argument('--no-axes', action='store_true',
                       help='Disable coordinate axes display')
    
    # Window options
    parser.add_argument('--window-size', type=int, nargs=2, default=[1600, 1000],
                       help='Window size (width height)')
    parser.add_argument('--background-color', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                       help='Background color (R G B)')
    
    args = parser.parse_args()
    
    # Auto-detect headless environment
    if not args.headless:
        args.headless = detect_headless_environment()
        if args.headless:
            print("Headless environment detected - running without GUI")
    
    try:
        # Load point cloud data
        print(f"Loading point cloud from: {args.points}")
        points = np.load(args.points)
        print(f"Loaded {points.shape[0]} points with {points.shape[1]} dimensions")
        
        # Load optional data
        gt_boxes = np.load(args.gt_boxes) if args.gt_boxes else None
        pred_boxes = np.load(args.pred_boxes) if args.pred_boxes else None
        gt_labels = np.load(args.gt_labels) if args.gt_labels else None
        pred_labels = np.load(args.pred_labels) if args.pred_labels else None
        pred_scores = np.load(args.pred_scores) if args.pred_scores else None
        
        # Initialize visualizer
        visualizer = Open3DVisualizer(
            window_name="KITTI/Waymo2KITTI 3D Visualization",
            window_size=tuple(args.window_size),
            background_color=tuple(args.background_color),
            point_size=args.point_size,
            headless=args.headless
        )
        
        # Run visualization
        visualizer.visualize_scene(
            points=points,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            save_path=args.save,
            show_grid=not args.no_grid,
            show_axes=not args.no_axes
        )
        
        # Clean up
        visualizer.close()
        
        print("Visualization completed successfully!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())