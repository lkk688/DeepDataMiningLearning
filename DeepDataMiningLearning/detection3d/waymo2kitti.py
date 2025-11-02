try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError('Please run "pip install waymo-open-dataset-tf-2-6-0" '
                      '>1.4.5 to install the official devkit first.')

import copy
import os
import os.path as osp
from glob import glob
from io import BytesIO
from os.path import exists, join
import pickle
import concurrent.futures
import argparse
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image
from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.utils.frame_utils import \
    parse_range_image_and_camera_projection

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")
    
    # Fallback progress tracker
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable else 0)
            self.desc = desc or ""
            self.n = 0
            
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
                
        def update(self, n=1):
            self.n += n
            if self.n % max(1, self.total // 20) == 0:  # Show progress every 5%
                print(f"{self.desc}: {self.n}/{self.total} ({100*self.n/self.total:.1f}%)")
                
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass


class Box3DMode(object):
    """Simple Enum for 3D box modes."""
    LIDAR = 0  # 3D box in LiDAR coordinates
    CAM = 1  # 3D box in camera coordinates


def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project 3D points in camera coordinates to 2D image coordinates.
    
    Performs perspective projection using the camera's intrinsic and extrinsic
    parameters encoded in the projection matrix. This is fundamental for 
    computer vision tasks like object detection and depth estimation.
    
    Mathematical Foundation:
    
    **Perspective Projection Equation:**
    $$\\mathbf{p}_{img} = \\mathbf{K} \\cdot \\mathbf{P}_{cam}$$
    
    Where:
    $$\\mathbf{K} = \\begin{bmatrix} 
    f_x & 0 & c_x & 0 \\\\ 
    0 & f_y & c_y & 0 \\\\ 
    0 & 0 & 1 & 0 
    \\end{bmatrix} \\in \\mathbb{R}^{3 \\times 4}$$
    
    **Homogeneous Coordinate Transformation:**
    $$\\begin{bmatrix} x \\\\ y \\\\ z \\end{bmatrix}_{cam} \\rightarrow 
    \\begin{bmatrix} x \\\\ y \\\\ z \\\\ 1 \\end{bmatrix}_{hom}$$
    
    **Projection Process:**
    1. **Homogeneous Conversion**: Add homogeneous coordinate (w=1)
    2. **Matrix Multiplication**: Apply projection matrix
    3. **Perspective Division**: Normalize by depth (z-coordinate)
    
    **Normalized Image Coordinates:**
    $$u = \\frac{f_x \\cdot x + c_x \\cdot z}{z}, \\quad 
    v = \\frac{f_y \\cdot y + c_y \\cdot z}{z}$$
    
    Where:
    - $(f_x, f_y)$ are focal lengths in pixels
    - $(c_x, c_y)$ are principal point coordinates
    - $(x, y, z)$ are 3D camera coordinates
    - $(u, v)$ are 2D image pixel coordinates
    
    Args:
        points_3d (np.ndarray): 3D points in camera coordinates, shape (N, 3)
                               Each point is [x_cam, y_cam, z_cam]
        proj_mat (np.ndarray): Camera projection matrix, shape (3, 4)
                              Combines intrinsic and extrinsic parameters
        with_depth (bool): Whether to return depth values along with 2D coordinates
                          Default: False (returns only [u, v])
    
    Returns:
        np.ndarray: Projected 2D image coordinates
                   - If with_depth=False: shape (N, 2) with [u, v] pixel coordinates
                   - If with_depth=True: shape (N, 3) with [u, v, depth] values
    """
    # Preserve original shape for proper reshaping at the end
    points_shape = list(points_3d.shape)  # Store original shape: [..., 3]
    points_3d = points_3d.reshape(-1, 3)  # Flatten to (N, 3) for processing
    num_points = points_3d.shape[0]       # Number of points to project

    # ==================== HOMOGENEOUS COORDINATE CONVERSION ====================
    # Convert 3D points to homogeneous coordinates by adding w=1 column
    # [x, y, z] -> [x, y, z, 1]
    
    points_4d = np.hstack([points_3d, np.ones((num_points, 1))]).T  # Shape: (4, N)
    # Transpose for efficient matrix multiplication: (N, 4) -> (4, N)

    # ==================== PERSPECTIVE PROJECTION ====================
    # Apply projection matrix to transform 3D camera points to 2D image plane
    # Mathematical operation: P_img_hom = K @ P_cam_hom
    
    points_2d = proj_mat @ points_4d  # Shape: (3, 4) @ (4, N) = (3, N)
    # Result contains [u*z, v*z, z] for each point (before normalization)

    # ==================== PERSPECTIVE DIVISION ====================
    # Normalize by depth (z-coordinate) to get actual pixel coordinates
    # This step converts from homogeneous to Cartesian coordinates
    
    # Avoid division by zero: ensure z > 0 (points in front of camera)
    z_coords = points_2d[2, :]  # Extract depth values, shape: (N,)
    
    # Perform perspective division: [u*z, v*z, z] -> [u, v, z]
    points_2d[:2, :] /= z_coords  # Divide x and y by z, shape: (2, N)
    # Now points_2d[0,:] = u coordinates, points_2d[1,:] = v coordinates

    # ==================== OUTPUT FORMATTING ====================
    if with_depth:
        # Return [u, v, depth] for each point
        points_2d = points_2d.T  # Shape: (N, 3) - transpose back to row format
    else:
        # Return only [u, v] for each point (drop depth information)
        points_2d = points_2d[:2, :].T  # Shape: (N, 2) - take only x,y and transpose

    # ==================== SHAPE RESTORATION ====================
    # Restore original batch dimensions while updating the last dimension
    # Original: (..., 3) -> Output: (..., 2) or (..., 3)
    
    points_shape[:1] = [-1]  # Set first dimension to -1 for automatic inference
    points_shape[-1] = points_2d.shape[-1]  # Update last dim: 3->2 or 3->3
    
    return points_2d.reshape(points_shape)  # Restore original batch structure


def post_process_coords(corner_coords, imsize):
    """Get 2D bounding box from projected 3D box corners with image boundary clipping.
    
    Computes the axis-aligned 2D bounding box that encloses all projected 3D box
    corners, then clips it to image boundaries. This is essential for object
    detection in computer vision, ensuring bounding boxes are valid for training
    and inference.
    
    Mathematical Process:
    
    **Bounding Box Computation:**
    $$\\text{bbox}_{2D} = [\\min(u_i), \\min(v_i), \\max(u_i), \\max(v_i)]$$
    
    Where $(u_i, v_i)$ are the projected corner coordinates.
    
    **Boundary Clipping:**
    $$\\begin{align}
    u_{min} &= \\max(0, \\min(u_i)) \\\\
    v_{min} &= \\max(0, \\min(v_i)) \\\\
    u_{max} &= \\min(W-1, \\max(u_i)) \\\\
    v_{max} &= \\min(H-1, \\max(v_i))
    \\end{align}$$
    
    Where $(W, H)$ are image width and height.
    
    **Visibility Check:**
    A bounding box is considered visible if:
    $$u_{max} \\geq 0 \\land u_{min} < W \\land v_{max} \\geq 0 \\land v_{min} < H$$
    
    Args:
        corner_coords (list[list[float]]): List of projected corner coordinates
                                         Each corner is [u, v] in image pixels
                                         Typically 8 corners from a 3D bounding box
        imsize (tuple[int, int]): Image dimensions as (width, height)
                                 Used for boundary clipping validation
    
    Returns:
        tuple[float, float, float, float] or None: 2D bounding box coordinates
            - If visible: (min_x, min_y, max_x, max_y) in image pixel coordinates
            - If completely outside image: None (indicates invisible object)
            
    Note:
        The returned coordinates are clipped to [0, width-1] and [0, height-1]
        to ensure valid pixel indices for image processing operations.
    """
    im_width, im_height = imsize  # Extract image dimensions

    # ==================== INPUT VALIDATION ====================
    if not corner_coords:
        # No corner coordinates provided - cannot compute bounding box
        return None

    # ==================== BOUNDING BOX COMPUTATION ====================
    # Find axis-aligned bounding box that encloses all projected corners
    # Convert list to numpy array for efficient min/max operations
    
    coords = np.array(corner_coords)  # Shape: (N_corners, 2) where N_corners ≤ 8
    
    # Compute bounding box extrema from all corner projections
    min_x = np.min(coords[:, 0])  # Leftmost pixel coordinate
    min_y = np.min(coords[:, 1])  # Topmost pixel coordinate  
    max_x = np.max(coords[:, 0])  # Rightmost pixel coordinate
    max_y = np.max(coords[:, 1])  # Bottommost pixel coordinate

    # ==================== VISIBILITY CHECK ====================
    # Check if the bounding box is completely outside the image boundaries
    # If so, the object is not visible and should be filtered out
    
    # Check horizontal visibility: box must overlap with [0, width-1]
    if max_x < 0 or min_x > im_width - 1:
        return None  # Box is completely left or right of image
    
    # Check vertical visibility: box must overlap with [0, height-1]  
    if max_y < 0 or min_y > im_height - 1:
        return None  # Box is completely above or below image

    # ==================== BOUNDARY CLIPPING ====================
    # Clip bounding box coordinates to valid image pixel ranges
    # This ensures all coordinates are within [0, width-1] × [0, height-1]
    
    min_x = max(0.0, min_x)                    # Clip left boundary
    min_y = max(0.0, min_y)                    # Clip top boundary
    max_x = min(float(im_width - 1), max_x)    # Clip right boundary  
    max_y = min(float(im_height - 1), max_y)   # Clip bottom boundary

    # Return clipped 2D bounding box in KITTI format: [x_min, y_min, x_max, y_max]
    return min_x, min_y, max_x, max_y


class Converted3DBoxes(object):
    """Proxy class to hold 3D bounding box data converted from LiDAR to camera coordinates.
    
    This class performs coordinate transformation using homogeneous coordinates and 
    transformation matrices. The conversion includes both spatial transformation and
    yaw angle correction for proper camera coordinate representation.
    
    Mathematical Foundation:
    - Homogeneous transformation: P_cam = T_lidar2cam @ P_lidar_hom
    - Where T_lidar2cam is a 4x4 transformation matrix
    - P_lidar_hom = [x, y, z, 1]^T (homogeneous coordinates)
    """

    def __init__(self, lidar_boxes, lidar2cam, correct_yaw):
        """Initialize converted 3D boxes with coordinate transformation.
        
        Args:
            lidar_boxes (LiDARInstance3DBoxes): Input boxes in LiDAR coordinates
            lidar2cam (np.ndarray): Transformation matrix from LiDAR to camera coords (4, 4)
            correct_yaw (bool): Whether to recalculate yaw angles in camera coordinates
        """
        self.N = lidar_boxes.tensor.shape[0]  # Number of bounding boxes
        self.dims = lidar_boxes.dims  # Box dimensions (N, 3): [length, width, height]

        # ==================== CORNER TRANSFORMATION ====================
        # Transform all 8 corners of each bounding box from LiDAR to camera coordinates
        # Mathematical operation: P_cam = T_lidar2cam @ P_lidar_hom
        
        corners_lidar = lidar_boxes.corners  # Shape: (N, 8, 3) - 8 corners per box
        corners_lidar_flat = corners_lidar.reshape(self.N * 8, 3)  # Shape: (N*8, 3)
        
        # Convert to homogeneous coordinates by adding ones column
        # [x, y, z] -> [x, y, z, 1]
        corners_lidar_hom = np.hstack(
            [corners_lidar_flat, np.ones((self.N * 8, 1))])  # Shape: (N*8, 4)
        
        # Apply transformation matrix: P_cam_hom = P_lidar_hom @ T^T
        # Note: Using transpose because we're doing row-vector multiplication
        corners_cam_hom = corners_lidar_hom @ lidar2cam.T  # Shape: (N*8, 4)
        
        # Extract 3D coordinates (drop homogeneous coordinate) and reshape back
        self._corners = corners_cam_hom[:, :3].reshape(self.N, 8, 3)  # Shape: (N, 8, 3)

        # ==================== CENTER TRANSFORMATION ====================
        # Transform gravity centers from LiDAR to camera coordinates
        
        centers_lidar = lidar_boxes.gravity_center  # Shape: (N, 3) - box centers
        
        # Convert centers to homogeneous coordinates
        centers_lidar_hom = np.hstack(
            [centers_lidar, np.ones((self.N, 1))])  # Shape: (N, 4)
        
        # Apply transformation: center_cam = T_lidar2cam @ center_lidar_hom
        centers_cam_hom = centers_lidar_hom @ lidar2cam.T  # Shape: (N, 4)
        
        # Extract 3D coordinates
        self._center = centers_cam_hom[:, :3]  # Shape: (N, 3)

        # ==================== YAW ANGLE CORRECTION ====================
        # Recalculate yaw angles in camera coordinate system
        
        if correct_yaw:
            # Calculate yaw from transformed corner positions
            # Use vector from corner 0 to corner 1 to determine orientation
            # Mathematical formula: yaw = -arctan2(Δz, Δx)
            # where Δz and Δx are differences in camera coordinates
            
            v = self._corners[:, 0, :] - self._corners[:, 1, :]  # Shape: (N, 3)
            # Corner indexing: 0 and 1 are adjacent corners along length dimension
            
            # Calculate yaw angle: θ = -arctan2(v_z, v_x)
            # Negative sign accounts for coordinate system differences
            self.yaw = -np.arctan2(v[:, 2], v[:, 0])  # Shape: (N,)
        else:
            # Fallback: use original yaw angles (may be incorrect in camera coords)
            # This preserves original LiDAR yaw angles without transformation
            self.yaw = lidar_boxes.tensor[:, 6]  # Shape: (N,)

    @property
    def corners(self):
        """Get transformed corner coordinates in camera coordinate system.
        
        Returns:
            np.ndarray: Corner coordinates in camera coords, shape (N, 8, 3)
                       Each box has 8 corners: 4 bottom + 4 top vertices
        """
        return self._corners

    @property
    def gravity_center(self):
        """Get transformed gravity centers in camera coordinate system.
        
        Returns:
            np.ndarray: Gravity centers in camera coords, shape (N, 3)
                       Each center is [x_cam, y_cam, z_cam]
        """
        return self._center

    def numpy(self):
        """Return the complete 3D box representation in camera coordinates.
        
        Combines center coordinates, dimensions, and yaw angles into a single tensor.
        Box format: [x_center, y_center, z_center, length, width, height, yaw]
        
        Mathematical representation:
        $$\\mathbf{B}_{cam} = [x_c, y_c, z_c, l, w, h, \\theta]^T$$
        
        Where:
        - $(x_c, y_c, z_c)$ are the gravity center coordinates in camera frame
        - $(l, w, h)$ are the box dimensions (length, width, height)
        - $\\theta$ is the yaw angle in camera coordinate system
        
        Returns:
            np.ndarray: Complete box tensor in camera coordinates, shape (N, 7)
        """
        return np.hstack([self._center, self.dims, self.yaw[:, np.newaxis]])


class LiDARInstance3DBoxes(object):
    """3D bounding box representation optimized for Waymo to KITTI conversion.

    This class handles 3D bounding boxes in LiDAR coordinate system with specific
    conventions for autonomous driving datasets. The implementation focuses on
    efficient coordinate transformations and corner generation.

    Coordinate System Conventions:
    - LiDAR coordinate system: X-forward, Y-left, Z-up
    - Box origin: Bottom center (z=0 at bottom face)
    - Box format: [x, y, z, length, width, height, yaw]
    
    Mathematical Foundation:
    - Box tensor: $\\mathbf{B} = [x, y, z, l, w, h, \\theta] \\in \\mathbb{R}^7$
    - Origin offset: $(0.5, 0.5, 0.0)$ represents bottom-center anchoring
    - Rotation matrix: $\\mathbf{R}_z(\\theta) = \\begin{bmatrix} \\cos\\theta & -\\sin\\theta & 0 \\\\ \\sin\\theta & \\cos\\theta & 0 \\\\ 0 & 0 & 1 \\end{bmatrix}$
    """

    def __init__(self, tensor, box_dim=7, origin=(0.5, 0.5, 0.0)):
        """Initialize LiDAR 3D bounding boxes from tensor representation.
        
        Args:
            tensor (np.ndarray or list): Box parameters, shape (N, 7) or (7,)
                                       Format: [x, y, z, length, width, height, yaw]
            box_dim (int): Dimension of box representation (default: 7)
            origin (tuple): Box origin offset (default: (0.5, 0.5, 0.0) for bottom-center)
                          - (0.5, 0.5, 0.0): bottom center
                          - (0.5, 0.5, 0.5): geometric center
        """
        # Ensure tensor is numpy array with proper shape
        if not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor)
        if tensor.ndim == 1:
            tensor = tensor[np.newaxis, :]  # Convert single box to batch format

        self.tensor = tensor.astype(np.float32)  # Shape: (N, 7)
        self.origin = origin  # Box anchoring point: (0.5, 0.5, 0.0) -> bottom center
        self.dims = self.tensor[:, 3:6]  # Extract dimensions: (N, 3) -> [length, width, height]

    @property
    def gravity_center(self):
        """Compute gravity center coordinates from bottom-center box representation.
        
        Converts from bottom-center anchored boxes to gravity-center coordinates.
        This is essential for proper 3D object detection and coordinate transformations.
        
        Mathematical transformation:
        $$\\mathbf{c}_{gravity} = \\mathbf{c}_{bottom} + \\Delta z \\cdot \\hat{\\mathbf{z}}$$
        
        Where:
        - $\\mathbf{c}_{bottom} = [x, y, z]^T$ is the bottom-center position
        - $\\Delta z = h \\cdot (0.5 - origin_z)$ is the vertical offset
        - $h$ is the box height
        - $origin_z = 0.0$ for bottom-center anchoring
        
        For bottom-center boxes: $\\Delta z = h \\cdot 0.5 = \\frac{h}{2}$
        
        Returns:
            np.ndarray: Gravity center coordinates, shape (N, 3)
                       Format: [x_gravity, y_gravity, z_gravity]
        """
        center = self.tensor[:, :3].copy()  # Shape: (N, 3) - [x, y, z] bottom-center coords
        
        # Adjust z-coordinate from bottom center to gravity center
        # z_gravity = z_bottom + height * (0.5 - origin_z)
        # For origin_z = 0.0: z_gravity = z_bottom + height/2
        center[:, 2] += self.tensor[:, 5] * (0.5 - self.origin[2])  # Add height/2 to z
        
        return center  # Shape: (N, 3)

    @property
    def corners(self):
        """Generate 8 corner coordinates for each 3D bounding box.
        
        Computes the 3D coordinates of all 8 corners of each bounding box using
        rotation matrices and translation. This is fundamental for 3D object
        detection, visualization, and coordinate system transformations.
        
        Mathematical Process:
        1. **Template Generation**: Create normalized corner template
        2. **Rotation**: Apply yaw rotation around Z-axis
        3. **Translation**: Translate to box center position
        
        Corner Template (normalized, before rotation):
        $$\\mathbf{C}_{template} = \\begin{bmatrix}
        \\pm\\frac{l}{2} & \\pm\\frac{w}{2} & 0 \\\\ 
        \\pm\\frac{l}{2} & \\pm\\frac{w}{2} & h
        \\end{bmatrix}$$
        
        Rotation Matrix (Z-axis rotation):
        $$\\mathbf{R}_z(\\theta) = \\begin{bmatrix} 
        \\cos\\theta & -\\sin\\theta & 0 \\\\ 
        \\sin\\theta & \\cos\\theta & 0 \\\\ 
        0 & 0 & 1 
        \\end{bmatrix}$$
        
        Final Corner Transformation:
        $$\\mathbf{C}_{world} = \\mathbf{C}_{template} \\cdot \\mathbf{R}_z^T + \\mathbf{t}$$
        
        Where:
        - $\\mathbf{t} = [x, y, z]^T$ is the box center translation
        - $(l, w, h)$ are length, width, height dimensions
        - $\\theta$ is the yaw angle around Z-axis
        
        Corner Indexing Convention:
        - Corners 0-3: Bottom face (z = 0)
        - Corners 4-7: Top face (z = height)
        - Order: [front-right, front-left, rear-left, rear-right] for each face
        
        Returns:
            np.ndarray: Corner coordinates, shape (N, 8, 3)
                       Each box has 8 corners with [x, y, z] coordinates
        """
        N = self.tensor.shape[0]  # Number of boxes
        l, w, h = self.dims[:, 0], self.dims[:, 1], self.dims[:, 2]  # Dimensions (N,) each

        # ==================== CORNER TEMPLATE GENERATION ====================
        # Create 8-corner template in local coordinate system (before rotation)
        # Template assumes box center at origin, z=0 at bottom face
        
        # X-coordinates: ±length/2 for front/rear faces
        x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
        
        # Y-coordinates: ±width/2 for left/right sides  
        y_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
        
        # Z-coordinates: 0 for bottom face, height for top face
        z_corners = np.array([0, 0, 0, 0, h, h, h, h])  # Bottom: 0-3, Top: 4-7

        # Stack into corner template: Shape (8, 3) -> (N, 8, 3) via broadcasting
        corners_base = np.stack(
            [x_corners, y_corners, z_corners], axis=2).transpose(0, 2, 1)  # Shape: (N, 3, 8)

        # ==================== ROTATION MATRIX COMPUTATION ====================
        # Compute rotation matrices for yaw angles around Z-axis
        
        yaw = self.tensor[:, 6]  # Extract yaw angles, shape: (N,)
        rot_sin = np.sin(yaw)    # Shape: (N,)
        rot_cos = np.cos(yaw)    # Shape: (N,)

        # Create helper arrays for 3D rotation matrix
        zeros = np.zeros_like(rot_cos)  # Shape: (N,)
        ones = np.ones_like(rot_cos)    # Shape: (N,)
        
        # Construct rotation matrix transpose for efficient multiplication
        # R_z^T = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        rot_mat_T = np.array([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros],
                              [zeros, zeros, ones]]).transpose(2, 0, 1)  # Shape: (N, 3, 3)

        # ==================== CORNER TRANSFORMATION ====================
        # Apply rotation: corners_rotated = corners_base @ R_z^T
        # Einstein summation for efficient batch matrix multiplication
        corners_3d = np.einsum('nij,nkj->nik', corners_base, rot_mat_T)  # Shape: (N, 8, 3)
        
        # Apply translation: add box center coordinates
        corners_3d += self.tensor[:, np.newaxis, :3]  # Broadcast center (N, 1, 3) to (N, 8, 3)

        return corners_3d  # Shape: (N, 8, 3)

    def convert_to(self, dst, rt_mat, correct_yaw=True):
        """Convert 3D bounding boxes to target coordinate system.
        
        This method creates a proxy object that handles coordinate system transformation
        from LiDAR coordinates to camera coordinates using homogeneous transformation
        matrices. The conversion is essential for multi-sensor fusion in autonomous
        driving applications.
        
        Mathematical Foundation:
        
        **Homogeneous Transformation:**
        $$\\mathbf{P}_{cam} = \\mathbf{T}_{lidar2cam} \\cdot \\mathbf{P}_{lidar}$$
        
        Where:
        $$\\mathbf{T}_{lidar2cam} = \\begin{bmatrix} 
        \\mathbf{R} & \\mathbf{t} \\\\ 
        \\mathbf{0}^T & 1 
        \\end{bmatrix} \\in \\mathbb{R}^{4 \\times 4}$$
        
        **Coordinate System Transformation Process:**
        1. **Corner Transformation**: All 8 corners of each box are transformed
        2. **Center Transformation**: Gravity centers are transformed
        3. **Yaw Correction**: Orientation angles are recalculated in target frame
        
        **Yaw Angle Correction:**
        $$\\theta_{cam} = -\\arctan2(\\Delta z, \\Delta x)$$
        
        Where $\\Delta z$ and $\\Delta x$ are corner differences in camera coordinates.
        
        Args:
            dst (int): Destination coordinate system (Box3DMode.CAM for camera coords)
            rt_mat (np.ndarray): 4x4 transformation matrix from LiDAR to target coords
                               Contains rotation (3x3) and translation (3x1) components
            correct_yaw (bool): Whether to recalculate yaw angles in target coordinate system
                              Default: True for proper orientation alignment
        
        Returns:
            Converted3DBoxes: Proxy object containing transformed 3D boxes
                             Provides same interface as original boxes but in target coords
        
        Raises:
            NotImplementedError: If destination coordinate system is not supported
        """
        if dst == Box3DMode.CAM:
            # Create a proxy object with transformed data
            return Converted3DBoxes(self, rt_mat, correct_yaw)
        else:
            raise NotImplementedError


class Waymo2KITTI(object):
    """Waymo to KITTI converter. There are 2 steps as follows:

    Step 1. Extract camera images and lidar point clouds from waymo raw data in
        '*.tfreord' and save as kitti format.
    Step 2. Generate waymo train/val/test infos and save as pickle file.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
        test_mode (bool, optional): Whether in the test_mode.
            Defaults to False.
        save_senor_data (bool, optional): Whether to save image and lidar
            data. Defaults to True.
        save_cam_sync_instances (bool, optional): Whether to save cam sync
            instances. Defaults to True.
        save_cam_instances (bool, optional): Whether to save cam instances.
            Defaults to False.
        info_prefix (str, optional): Prefix of info filename.
            Defaults to 'waymo'.
        max_sweeps (int, optional): Max length of sweeps. Defaults to 10.
        split (str, optional): Split of the data. Defaults to 'training'.
        subsample_interval (int, optional): Rate to subsample frames.
            Defaults to 1 (process every frame).
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 prefix,
                 workers=64,
                 test_mode=False,
                 save_senor_data=True,
                 save_cam_sync_instances=True,
                 save_cam_instances=True,
                 info_prefix='waymo',
                 max_sweeps=10,
                 split='training',
                 subsample_interval=1):  # <-- MODIFIED: Added subsample
        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        # keep the order defined by the official protocol
        self.cam_list = [
            '_FRONT',
            '_FRONT_LEFT',
            '_FRONT_RIGHT',
            '_SIDE_LEFT',
            '_SIDE_RIGHT',
        ]
        self.lidar_list = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']
        self.type_list = [
            'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
        ]

        # MMDetection3D unified camera keys & class names
        self.camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_SIDE_LEFT',
            'CAM_SIDE_RIGHT',
        ]
        self.selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        self.info_map = {
            'training': '_infos_train.pkl',
            'validation': '_infos_val.pkl',
            'testing': '_infos_test.pkl',
            'testing_3d_camera_only_detection': '_infos_test_cam_only.pkl'
        }

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode
        self.save_senor_data = save_senor_data
        self.save_cam_sync_instances = save_cam_sync_instances
        self.save_cam_instances = save_cam_instances
        self.info_prefix = info_prefix
        self.max_sweeps = max_sweeps
        self.split = split
        self.subsample_interval = int(subsample_interval)  # <-- MODIFIED: Store
        assert self.subsample_interval >= 1, \
            'subsample_interval must be >= 1'

        # TODO: Discuss filter_empty_3dboxes and filter_no_label_zone_points
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True
        self.save_track_id = False

        self.tfrecord_pathnames = sorted(
            glob(join(self.load_dir, '*.tfrecord')))

        self.image_save_dir = f'{self.save_dir}/image_'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'

        # Create folder for saving KITTI format camera images and
        # lidar point clouds.
        if 'testing_3d_camera_only_detection' not in self.load_dir:
            # Replaced mmengine.mkdir_or_exist
            os.makedirs(self.point_cloud_save_dir, exist_ok=True)
        for i in range(5):
            # Replaced mmengine.mkdir_or_exist
            os.makedirs(f'{self.image_save_dir}{str(i)}', exist_ok=True)

    def convert(self):
        """Convert action with progress tracking and validation."""
        start_time = time.time()
        print(f'Start converting {self.split} dataset at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Found {len(self)} tfrecord files to process')
        print(f'Using {self.workers} workers (0 = sequential processing)')
        print(f'Subsample interval: {self.subsample_interval}')
        
        # Statistics tracking
        total_frames = 0
        total_images = 0
        total_point_clouds = 0
        failed_files = []
        
        if self.workers == 0:
            # Sequential processing with progress bar
            data_infos = []
            with tqdm(range(len(self)), desc="Converting files", unit="file") as pbar:
                for i in pbar:
                    try:
                        result = self.convert_one(i)
                        data_infos.append(result)
                        
                        # Update statistics
                        total_frames += len(result)
                        if self.save_senor_data:
                            total_images += len(result) * 5  # 5 cameras
                            total_point_clouds += len(result)
                            
                        pbar.set_postfix({
                            'frames': total_frames,
                            'images': total_images,
                            'point_clouds': total_point_clouds
                        })
                    except Exception as e:
                        failed_files.append((i, str(e)))
                        print(f"Error processing file {i}: {e}")
                        continue
        else:
            # Parallel processing with progress tracking
            data_infos_list = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
                # Submit all tasks
                future_to_idx = {executor.submit(self.convert_one, i): i for i in range(len(self))}
                
                # Process results with progress bar
                with tqdm(total=len(self), desc="Converting files", unit="file") as pbar:
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            result = future.result()
                            data_infos_list.append(result)
                            
                            # Update statistics
                            total_frames += len(result)
                            if self.save_senor_data:
                                total_images += len(result) * 5  # 5 cameras
                                total_point_clouds += len(result)
                                
                            pbar.set_postfix({
                                'frames': total_frames,
                                'images': total_images,
                                'point_clouds': total_point_clouds
                            })
                        except Exception as e:
                            failed_files.append((idx, str(e)))
                            print(f"Error processing file {idx}: {e}")
                        finally:
                            pbar.update(1)
            
            data_infos = data_infos_list

        # Flatten data_infos
        data_list = []
        for data_info in data_infos:
            if data_info:  # Skip None results from failed files
                data_list.extend(data_info)
        
        # Create metadata
        metainfo = dict()
        metainfo['dataset'] = 'waymo'
        metainfo['version'] = 'waymo_v1.4'
        metainfo['info_version'] = 'mmdet3d_v1.4_standalone'
        metainfo['conversion_time'] = datetime.now().isoformat()
        metainfo['subsample_interval'] = self.subsample_interval
        metainfo['total_files'] = len(self)
        metainfo['successful_files'] = len(self) - len(failed_files)
        metainfo['failed_files'] = len(failed_files)
        metainfo['total_frames'] = total_frames
        
        waymo_infos = dict(data_list=data_list, metainfo=metainfo)
        
        # Save pickle file
        filenames = osp.join(
            osp.dirname(self.save_dir),
            f'{self.info_prefix + self.info_map[self.split]}')
        print(f'\nSaving {self.split} dataset infos into {filenames}')
        
        with open(filenames, 'wb') as f:
            pickle.dump(waymo_infos, f)
        
        # Print conversion summary
        elapsed_time = time.time() - start_time
        self._print_conversion_summary(elapsed_time, total_frames, total_images, 
                                     total_point_clouds, failed_files, filenames)
        
        return waymo_infos

    def convert_one(self, file_idx):
        """Convert one '*.tfrecord' file to kitti format. Each file stores all
        the frames (about 200 frames) in current scene. We treat each frame as
        a sample, save their images and point clouds in kitti format, and then
        create info for all frames.

        Args:
            file_idx (int): Index of the file to be converted.

        Returns:
            List[dict]: Waymo infos for the subsampled frames in current file.
        """
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        # NOTE: all_frame_infos stores metadata for *all* frames
        # in the current file. This is necessary to correctly
        # gather sweep data, which looks at previous frames.
        all_frame_infos = []

        # NOTE: subsampled_frame_infos stores metadata for
        # only the frames we want to keep (e.g., every Kth frame).
        # This is the list that will be returned.
        subsampled_frame_infos = []

        for frame_idx, data in enumerate(dataset):

            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            # We must generate the info file for *every* frame
            # so that sweep lookups are correct.
            # create_waymo_info_file appends the info to all_frame_infos.
            self.create_waymo_info_file(frame, file_idx, frame_idx,
                                        all_frame_infos)

            # Now, check if this frame is one we should keep
            if frame_idx % self.subsample_interval == 0:
                # If so, save the sensor data (images/lidar)
                if self.save_senor_data:
                    self.save_image(frame, file_idx, frame_idx)
                    self.save_lidar(frame, file_idx, frame_idx)

                # And add the corresponding info (the last one added)
                # to the list we will return.
                subsampled_frame_infos.append(all_frame_infos[-1])

        return subsampled_frame_infos  # Return the subsampled list

    def _print_conversion_summary(self, elapsed_time, total_frames, total_images, 
                                total_point_clouds, failed_files, pickle_path):
        """Print detailed conversion summary."""
        print("\n" + "="*80)
        print("WAYMO TO KITTI CONVERSION SUMMARY")
        print("="*80)
        
        # Time information
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Conversion completed in: {int(hours):02d}h {int(minutes):02d}m {seconds:.1f}s")
        
        # File statistics
        print(f"\nFile Processing:")
        print(f"  Total tfrecord files: {len(self.tfrecord_pathnames)}")
        print(f"  Successfully processed: {len(self.tfrecord_pathnames) - len(failed_files)}")
        print(f"  Failed files: {len(failed_files)}")
        
        if failed_files:
            print(f"  Failed file indices: {[idx for idx, _ in failed_files]}")
        
        # Frame and data statistics
        print(f"\nData Statistics:")
        print(f"  Total frames processed: {total_frames}")
        print(f"  Subsample interval: {self.subsample_interval}")
        if self.save_senor_data:
            print(f"  Images saved: {total_images}")
            print(f"  Point clouds saved: {total_point_clouds}")
        
        # Output directory structure
        print(f"\nOutput Directory Structure:")
        print(f"  Root directory: {self.save_dir}")
        print(f"  Images: {self.image_save_dir}[0-4]/")
        if 'testing_3d_camera_only_detection' not in self.load_dir:
            print(f"  Point clouds: {self.point_cloud_save_dir}/")
        print(f"  Pickle file: {pickle_path}")
        
        # Validate output
        self._validate_output()
        
        print("="*80)

    def _validate_output(self):
        """Validate the converted output files."""
        print(f"\nValidation Results:")
        
        # Check image directories
        for i in range(5):
            img_dir = f'{self.image_save_dir}{i}'
            if os.path.exists(img_dir):
                img_count = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
                print(f"  Camera {i} images: {img_count} files")
            else:
                print(f"  Camera {i} images: Directory not found")
        
        # Check point cloud directory
        if 'testing_3d_camera_only_detection' not in self.load_dir:
            if os.path.exists(self.point_cloud_save_dir):
                pc_count = len([f for f in os.listdir(self.point_cloud_save_dir) if f.endswith('.bin')])
                print(f"  Point clouds: {pc_count} files")
            else:
                print(f"  Point clouds: Directory not found")
        
        # Check pickle file
        pickle_path = osp.join(
            osp.dirname(self.save_dir),
            f'{self.info_prefix + self.info_map[self.split]}')
        if os.path.exists(pickle_path):
            file_size = os.path.getsize(pickle_path) / (1024 * 1024)  # MB
            print(f"  Pickle file: {file_size:.2f} MB")
            
            # Try to load and validate pickle structure
            try:
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"  Pickle validation: ✓ Valid structure")
                print(f"    - Data entries: {len(data['data_list'])}")
                print(f"    - Metadata keys: {list(data['metainfo'].keys())}")
            except Exception as e:
                print(f"  Pickle validation: ✗ Error loading: {e}")
        else:
            print(f"  Pickle file: Not found")

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, file_idx, frame_idx):
        """Parse and save the images in jpg format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            img_path = f'{self.image_save_dir}{str(img.name - 1)}/' + \
                f'{self.prefix}{str(file_idx).zfill(3)}' + \
                f'{str(frame_idx).zfill(3)}.jpg'
            with open(img_path, 'wb') as fp:
                fp.write(img.image)

    def save_lidar(self, frame, file_idx, frame_idx):
        """Parse and save the lidar data in psd format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        range_images, camera_projections, seg_labels, range_image_top_pose = \
            parse_range_image_and_camera_projection(frame)

        if range_image_top_pose is None:
            # the camera only split doesn't contain lidar points.
            return
        # First return
        points_0, cp_points_0, intensity_0, elongation_0, mask_indices_0 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=0
            )
        points_0 = np.concatenate(points_0, axis=0)
        intensity_0 = np.concatenate(intensity_0, axis=0)
        elongation_0 = np.concatenate(elongation_0, axis=0)
        mask_indices_0 = np.concatenate(mask_indices_0, axis=0)

        # Second return
        points_1, cp_points_1, intensity_1, elongation_1, mask_indices_1 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=1
            )
        points_1 = np.concatenate(points_1, axis=0)
        intensity_1 = np.concatenate(intensity_1, axis=0)
        elongation_1 = np.concatenate(elongation_1, axis=0)
        mask_indices_1 = np.concatenate(mask_indices_1, axis=0)

        points = np.concatenate([points_0, points_1], axis=0)
        intensity = np.concatenate([intensity_0, intensity_1], axis=0)
        elongation = np.concatenate([elongation_0, elongation_1], axis=0)
        mask_indices = np.concatenate([mask_indices_0, mask_indices_1], axis=0)

        # timestamp = frame.timestamp_micros * np.ones_like(intensity)

        # concatenate x,y,z, intensity, elongation, timestamp (6-dim)
        point_cloud = np.column_stack(
            (points, intensity, elongation, mask_indices))

        pc_path = f'{self.point_cloud_save_dir}/{self.prefix}' + \
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.bin'
        point_cloud.astype(np.float32).tofile(pc_path)

    def convert_range_image_to_point_cloud(self,
                                             frame,
                                             range_images,
                                             camera_projections,
                                             range_image_top_pose,
                                             ri_index=0):
        """Convert range images to point cloud.

        Args:
            frame (:obj:`Frame`): Open dataset frame.
            range_images (dict): Mapping from laser_name to list of two
                range images corresponding with two returns.
            camera_projections (dict): Mapping from laser_name to list of two
                camera projections corresponding with two returns.
            range_image_top_pose (:obj:`Transform`): Range image pixel pose for
                top lidar.
            ri_index (int, optional): 0 for the first return,
                1 for the second return. Default: 0.

        Returns:
            tuple[list[np.ndarray]]: (List of points with shape [N, 3],
                camera projections of points with shape [N, 6], intensity
                with shape [N, 1], elongation with shape [N, 1], points'
                position in the depth map (element offset if points come from
                the main lidar otherwise -1) with shape[N, 1]). All the
                lists have the length of lidar numbers (5).
        """
        calibrations = sorted(
            frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []
        intensity = []
        elongation = []
        mask_indices = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = \
            transform_utils.get_rotation_matrix(
                range_image_top_pose_tensor[..., 0],
                range_image_top_pose_tensor[..., 1],
                range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = \
            range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant(
                        [c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data),
                range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0

            if self.filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask

            range_image_cartesian = \
                range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(
                        value=beam_inclinations), axis=0),
                    pixel_pose=pixel_pose_local,
                    frame_pose=frame_pose_local)

            mask_index = tf.where(range_image_mask)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian, mask_index)

            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(
                tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, mask_index)
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

            intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                            mask_index)
            intensity.append(intensity_tensor.numpy())

            elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                             mask_index)
            elongation.append(elongation_tensor.numpy())
            if c.name == 1:
                mask_index = (ri_index * range_image_mask.shape[0] +
                              mask_index[:, 0]
                              ) * range_image_mask.shape[1] + mask_index[:, 1]
                mask_index = mask_index.numpy().astype(elongation[-1].dtype)
            else:
                mask_index = np.full_like(elongation[-1], -1)

            mask_indices.append(mask_index)

        return points, cp_points, intensity, elongation, mask_indices

    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret

    def create_waymo_info_file(self, frame, file_idx, frame_idx, file_infos):
        r"""Generate waymo train/val/test infos.

        This function appends the generated info to the `file_infos` list.

        For more details about infos, please refer to:
        https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html
        """  # noqa: E501
        frame_infos = dict()

        # Gather frame infos
        sample_idx = \
            f'{self.prefix}{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}'
        frame_infos['sample_idx'] = int(sample_idx)
        frame_infos['timestamp'] = frame.timestamp_micros
        frame_infos['ego2global'] = np.array(frame.pose.transform).reshape(
            4, 4).astype(np.float32).tolist()
        frame_infos['context_name'] = frame.context.name

        # Gather camera infos
        frame_infos['images'] = dict()
        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        camera_calibs = []
        Tr_velo_to_cams = []
        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            Tr_velo_to_cams.append(Tr_velo_to_cam)

            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calibs.append(camera_calib)

        for i, (cam_key, camera_calib, Tr_velo_to_cam) in enumerate(
                zip(self.camera_types, camera_calibs, Tr_velo_to_cams)):
            cam_infos = dict()
            cam_infos['img_path'] = str(sample_idx) + '.jpg'
            # NOTE: frames.images order is different
            img_found = False
            for img in frame.images:
                if img.name == i + 1:
                    width, height = Image.open(BytesIO(img.image)).size
                    img_found = True
            
            if not img_found:
                # Handle cases where image might be missing (e.g. cam-only data)
                width, height = -1, -1 
                
            cam_infos['height'] = height
            cam_infos['width'] = width
            cam_infos['lidar2cam'] = Tr_velo_to_cam.astype(np.float32).tolist()
            cam_infos['cam2img'] = camera_calib.astype(np.float32).tolist()
            cam_infos['lidar2img'] = (camera_calib @ Tr_velo_to_cam).astype(
                np.float32).tolist()
            frame_infos['images'][cam_key] = cam_infos

        # Gather lidar infos
        lidar_infos = dict()
        lidar_infos['lidar_path'] = str(sample_idx) + '.bin'
        lidar_infos['num_pts_feats'] = 6
        frame_infos['lidar_points'] = lidar_infos

        # Gather lidar sweeps and camera sweeps infos
        # TODO: Add lidar2img in image sweeps infos when we need it.
        # TODO: Consider merging lidar sweeps infos and image sweeps infos.
        lidar_sweeps_infos, image_sweeps_infos = [], []
        # `file_infos` contains the history of all previous frames in
        # this tfrecord file.
        for prev_offset in range(-1, -self.max_sweeps - 1, -1):
            prev_lidar_infos = dict()
            prev_image_infos = dict()
            if frame_idx + prev_offset >= 0:
                prev_frame_infos = file_infos[prev_offset]
                prev_lidar_infos['timestamp'] = prev_frame_infos['timestamp']
                prev_lidar_infos['ego2global'] = prev_frame_infos['ego2global']
                prev_lidar_infos['lidar_points'] = dict()
                lidar_path = prev_frame_infos['lidar_points']['lidar_path']
                prev_lidar_infos['lidar_points']['lidar_path'] = lidar_path
                lidar_sweeps_infos.append(prev_lidar_infos)

                prev_image_infos['timestamp'] = prev_frame_infos['timestamp']
                prev_image_infos['ego2global'] = prev_frame_infos['ego2global']
                prev_image_infos['images'] = dict()
                for cam_key in self.camera_types:
                    prev_image_infos['images'][cam_key] = dict()
                    img_path = prev_frame_infos['images'][cam_key]['img_path']
                    prev_image_infos['images'][cam_key]['img_path'] = img_path
                image_sweeps_infos.append(prev_image_infos)
        if lidar_sweeps_infos:
            frame_infos['lidar_sweeps'] = lidar_sweeps_infos
        if image_sweeps_infos:
            frame_infos['image_sweeps'] = image_sweeps_infos

        if not self.test_mode:
            # Gather instances infos which is used for lidar-based 3D detection
            frame_infos['instances'] = self.gather_instance_info(frame)
            # Gather cam_sync_instances infos which is used for image-based
            # (multi-view) 3D detection.
            if self.save_cam_sync_instances:
                frame_infos['cam_sync_instances'] = self.gather_instance_info(
                    frame, cam_sync=True)
            # Gather cam_instances infos which is used for image-based
            # (monocular) 3D detection (optional).
            # TODO: Should we use cam_sync_instances to generate cam_instances?
            if self.save_cam_instances:
                frame_infos['cam_instances'] = self.gather_cam_instance_info(
                    copy.deepcopy(frame_infos['instances']),
                    frame_infos['images'])

        # Append the current frame's info to the list
        file_infos.append(frame_infos)

    def gather_instance_info(self, frame, cam_sync=False):
        """Generate instances and cam_sync_instances infos.

        For more details about infos, please refer to:
        https.mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html
        """  # noqa: E501
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                # TODO: need a workaround as bbox may not belong to front cam
                bbox = [
                    label.box.center_x - label.box.length / 2,
                    label.box.center_y - label.box.width / 2,
                    label.box.center_x + label.box.length / 2,
                    label.box.center_y + label.box.width / 2
                ]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        group_id = 0
        instance_infos = []
        for obj in frame.laser_labels:
            instance_info = dict()
            bounding_box = None
            name = None
            id = obj.id
            for proj_cam in self.cam_list:
                if id + proj_cam in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + proj_cam)
                    name = id_to_name.get(id + proj_cam)
                    break

            # NOTE: the 2D labels do not have strict correspondence with
            # the projected 2D lidar labels
            # e.g.: the projected 2D labels can be in camera 2
            # while the most_visible_camera can have id 4
            if cam_sync:
                if obj.most_visible_camera_name:
                    name = self.cam_list.index(
                        f'_{obj.most_visible_camera_name}')
                    box3d = obj.camera_synced_box
                else:
                    continue
            else:
                box3d = obj.box

            if bounding_box is None or name is None:
                name = 0
                bounding_box = [0.0, 0.0, 0.0, 0.0]

            my_type = self.type_list[obj.type]

            if my_type not in self.selected_waymo_classes:
                continue
            else:
                label = self.selected_waymo_classes.index(my_type)

            if self.filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue

            group_id += 1
            instance_info['group_id'] = group_id
            instance_info['camera_id'] = name
            instance_info['bbox'] = bounding_box
            instance_info['bbox_label'] = label

            height = box3d.height
            width = box3d.width
            length = box3d.length

            # NOTE: We save the bottom center of 3D bboxes.
            x = box3d.center_x
            y = box3d.center_y
            z = box3d.center_z - height / 2

            rotation_y = box3d.heading

            instance_info['bbox_3d'] = np.array(
                [x, y, z, length, width, height,
                 rotation_y]).astype(np.float32).tolist()
            instance_info['bbox_label_3d'] = label
            instance_info['num_lidar_pts'] = obj.num_lidar_points_in_box

            if self.save_track_id:
                instance_info['track_id'] = obj.id
            instance_infos.append(instance_info)
        return instance_infos

    def gather_cam_instance_info(self, instances: dict, images: dict):
        """Generate cam_instances infos.

        For more details about infos, please refer to:
        https.mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html
        """  # noqa: E501
        cam_instances = dict()
        for cam_type in self.camera_types:
            cam_instances[cam_type] = []
            
            # Skip if image data is missing
            if images[cam_type]['width'] == -1:
                continue
                
            lidar2cam = np.array(images[cam_type]['lidar2cam'])
            cam2img = np.array(images[cam_type]['cam2img'])

            for instance in instances:
                cam_instance = dict()
                gt_bboxes_3d = np.array(instance['bbox_3d'])
                # Convert lidar coordinates to camera coordinates
                gt_bboxes_3d_cam = LiDARInstance3DBoxes(
                    gt_bboxes_3d[None, :]).convert_to(
                        Box3DMode.CAM, lidar2cam, correct_yaw=True)
                
                corners_3d = gt_bboxes_3d_cam.corners.numpy()
                corners_3d = corners_3d[0].T  # (1, 8, 3) -> (3, 8)
                in_camera = np.argwhere(corners_3d[2, :] > 0).flatten()
                
                # Skip if all corners are behind the camera
                if len(in_camera) == 0:
                    continue
                    
                corners_3d_in_front = corners_3d[:, in_camera]

                # Project 3d box to 2d.
                # Use our self-contained points_cam2img
                corner_coords = points_cam2img(
                    corners_3d_in_front.T, cam2img,
                    with_depth=False).tolist()

                # Keep only corners that fall within the image.
                # TODO: imsize should be determined by the current image size
                # CAM_FRONT: (1920, 1280)
                # CAM_FRONT_LEFT: (1920, 1280)
                # CAM_SIDE_LEFT: (1920, 886)
                final_coords = post_process_coords(
                    corner_coords,
                    imsize=(images['CAM_FRONT']['width'],
                            images['CAM_FRONT']['height']))

                # Skip if the convex hull of the re-projected corners
                # does not intersect the image canvas.
                if final_coords is None:
                    continue
                else:
                    min_x, min_y, max_x, max_y = final_coords

                cam_instance['bbox'] = [min_x, min_y, max_x, max_y]
                cam_instance['bbox_label'] = instance['bbox_label']
                cam_instance['bbox_3d'] = gt_bboxes_3d_cam.numpy().squeeze(
                ).astype(np.float32).tolist()
                cam_instance['bbox_label_3d'] = instance['bbox_label_3d']

                center_3d = gt_bboxes_3d_cam.gravity_center.numpy()
                center_2d_with_depth = points_cam2img(
                    center_3d, cam2img, with_depth=True)
                center_2d_with_depth = center_2d_with_depth.squeeze().tolist()

                # normalized center2D + depth
                # if samples with depth < 0 will be removed
                if center_2d_with_depth[2] <= 0:
                    continue
                cam_instance['center_2d'] = center_2d_with_depth[:2]
                cam_instance['depth'] = center_2d_with_depth[2]

                # TODO: Discuss whether following info is necessary
                cam_instance['bbox_3d_isvalid'] = True
                cam_instance['velocity'] = -1
                cam_instances[cam_type].append(cam_instance)

        return cam_instances

    def merge_trainval_infos(self):
        """Merge training and validation infos into a single file."""
        train_infos_path = osp.join(
            osp.dirname(self.save_dir), f'{self.info_prefix}_infos_train.pkl')
        val_infos_path = osp.join(
            osp.dirname(self.save_dir), f'{self.info_prefix}_infos_val.pkl')
        
        # Replaced mmengine.load
        with open(train_infos_path, 'rb') as f:
            train_infos = pickle.load(f)
        with open(val_infos_path, 'rb') as f:
            val_infos = pickle.load(f)
            
        trainval_infos = dict(
            metainfo=train_infos['metainfo'],
            data_list=train_infos['data_list'] + val_infos['data_list'])
        
        # Replaced mmengine.dump
        with open(
            osp.join(
                osp.dirname(self.save_dir),
                f'{self.info_prefix}_infos_trainval.pkl'), 'wb') as f:
            pickle.dump(trainval_infos, f)


def create_ImageSets_img_ids(root_dir, splits):
    """Create txt files indicating what to collect in each split."""
    save_dir = join(root_dir, 'ImageSets/')
    if not exists(save_dir):
        os.makedirs(save_dir, exist_ok=True) # Use os.makedirs

    idx_all = [[] for _ in splits]
    for i, split in enumerate(splits):
        path = join(root_dir, split, 'image_0')
        if not exists(path):
            RawNames = []
        else:
            RawNames = os.listdir(path)

        for name in RawNames:
            if name.endswith('.jpg'):
                idx = name.replace('.jpg', '\n')
                idx_all[int(idx[0])].append(idx)
        idx_all[i].sort()

    open(save_dir + 'train.txt', 'w').writelines(idx_all[0])
    open(save_dir + 'val.txt', 'w').writelines(idx_all[1])
    open(save_dir + 'trainval.txt', 'w').writelines(idx_all[0] + idx_all[1])
    if len(idx_all) >= 3:
        open(save_dir + 'test.txt', 'w').writelines(idx_all[2])
    if len(idx_all) >= 4:
        open(save_dir + 'test_cam_only.txt', 'w').writelines(idx_all[3])
    print('created txt files indicating what to collect in ', splits)


def main():
    """Main function to run Waymo to KITTI conversion with progress tracking and validation."""
    parser = argparse.ArgumentParser(
        description='Waymo dataset processing pipeline: prepare, convert, test, or all',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        required=True
    )
    
    # Common arguments for all commands
    def add_common_args(subparser):
        subparser.add_argument(
            '--load_dir', 
            type=str, 
            required=True,
            help='Path to directory containing Waymo data (tar files for prepare, tfrecord files for convert/test)'
        )
        subparser.add_argument(
            '--save_dir', 
            type=str, 
            required=True,
            help='Output directory for processed dataset'
        )
    
    # Prepare command - untar downloaded files
    prepare_parser = subparsers.add_parser(
        'prepare',
        help='Extract tar files from downloaded Waymo dataset'
    )
    add_common_args(prepare_parser)
    prepare_parser.add_argument(
        '--extract_dir',
        type=str,
        help='Directory to extract tar files (defaults to load_dir if not specified)'
    )
    
    # Convert command - convert tfrecord to KITTI format
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert Waymo tfrecord files to KITTI format'
    )
    add_common_args(convert_parser)
    
    # Optional arguments for convert command
    convert_parser.add_argument(
        '--prefix', 
        type=str, 
        default='',
        help='Prefix for output files'
    )
    convert_parser.add_argument(
        '--num_proc', 
        type=int, 
        default=1,
        help='Number of parallel processes for conversion'
    )
    convert_parser.add_argument(
        '--only_gt_boxes_for_camera', 
        action='store_true',
        help='Only save ground truth boxes visible in camera images'
    )
    convert_parser.add_argument(
        '--sampled_interval', 
        type=int, 
        default=5,
        help='Sampling interval for frames (1 = all frames, 5 = every 5th frame)'
    )
    convert_parser.add_argument(
        '--save_track_id', 
        action='store_true',
        help='Save tracking IDs in the annotations'
    )
    convert_parser.add_argument(
        '--save_cam_sync_labels', 
        action='store_true',
        help='Save camera-synchronized labels'
    )
    convert_parser.add_argument(
        '--info_prefix', 
        type=str, 
        default='waymo',
        help='Prefix for info pickle files'
    )
    convert_parser.add_argument(
        '--max_sweeps', 
        type=int, 
        default=5,
        help='Maximum number of sweeps to include'
    )
    convert_parser.add_argument(
        '--split', 
        type=str, 
        choices=['training', 'validation', 'testing'],
        default='training',
        help='Dataset split to convert'
    )
    convert_parser.add_argument(
        '--downsample', 
        type=int, 
        default=1,
        help='Downsample factor: process every K samples in each segment (1 = all samples, 2 = every 2nd sample, etc.)'
    )
    convert_parser.add_argument(
        '--skip_validation', 
        action='store_true',
        help='Skip output validation after conversion'
    )
    
    # Test command - validate converted dataset
    test_parser = subparsers.add_parser(
        'test',
        help='Test and validate converted KITTI dataset'
    )
    add_common_args(test_parser)
    test_parser.add_argument(
        '--test_type',
        type=str,
        choices=['basic', 'integration', 'performance', 'all'],
        default='basic',
        help='Type of test to run'
    )
    
    # All command - run prepare, convert, and test in sequence
    all_parser = subparsers.add_parser(
        'all',
        help='Run complete pipeline: prepare, convert, and test'
    )
    add_common_args(all_parser)
    all_parser.add_argument(
        '--extract_dir',
        type=str,
        help='Directory to extract tar files (defaults to load_dir if not specified)'
    )
    # Add all convert arguments to 'all' command
    all_parser.add_argument(
        '--prefix', 
        type=str, 
        default='',
        help='Prefix for output files'
    )
    all_parser.add_argument(
        '--num_proc', 
        type=int, 
        default=1,
        help='Number of parallel processes for conversion'
    )
    all_parser.add_argument(
        '--only_gt_boxes_for_camera', 
        action='store_true',
        help='Only save ground truth boxes visible in camera images'
    )
    all_parser.add_argument(
        '--sampled_interval', 
        type=int, 
        default=5,
        help='Sampling interval for frames (1 = all frames, 5 = every 5th frame)'
    )
    all_parser.add_argument(
        '--save_track_id', 
        action='store_true',
        help='Save tracking IDs in the annotations'
    )
    all_parser.add_argument(
        '--save_cam_sync_labels', 
        action='store_true',
        help='Save camera-synchronized labels'
    )
    all_parser.add_argument(
        '--info_prefix', 
        type=str, 
        default='waymo',
        help='Prefix for info pickle files'
    )
    all_parser.add_argument(
        '--max_sweeps', 
        type=int, 
        default=5,
        help='Maximum number of sweeps to include'
    )
    all_parser.add_argument(
        '--split', 
        type=str, 
        choices=['training', 'validation', 'testing'],
        default='training',
        help='Dataset split to convert'
    )
    all_parser.add_argument(
        '--downsample', 
        type=int, 
        default=1,
        help='Downsample factor: process every K samples in each segment (1 = all samples, 2 = every 2nd sample, etc.)'
    )
    all_parser.add_argument(
        '--test_type',
        type=str,
        choices=['basic', 'integration', 'performance', 'all'],
        default='basic',
        help='Type of test to run'
    )
    
    args = parser.parse_args()
    
    # Execute based on command
    if args.command == 'prepare':
        prepare_dataset(args)
    elif args.command == 'convert':
        convert_dataset(args)
    elif args.command == 'test':
        test_dataset(args)
    elif args.command == 'all':
        print(f"\n{'='*60}")
        print(f"RUNNING COMPLETE WAYMO PROCESSING PIPELINE")
        print(f"{'='*60}\n")
        
        # Step 1: Prepare
        print("Step 1/3: Preparing dataset (extracting tar files)...")
        prepare_dataset(args)
        
        # Step 2: Convert
        print("\nStep 2/3: Converting to KITTI format...")
        convert_dataset(args)
        
        # Step 3: Test
        print("\nStep 3/3: Testing and validation...")
        test_dataset(args)
        
        print(f"\n{'='*60}")
        print(f"COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print(f"{'='*60}\n")


def prepare_dataset(args):
    """Extract tar files from downloaded Waymo dataset with robust error handling."""
    import tarfile
    import glob
    
    def validate_tar_file(tar_path):
        """Validate tar file integrity and format.
        
        Args:
            tar_path (str): Path to tar file
            
        Returns:
            tuple: (is_valid, error_message, file_size)
        """
        try:
            # Check if file exists and get size
            if not os.path.exists(tar_path):
                return False, "File does not exist", 0
            
            file_size = os.path.getsize(tar_path)
            if file_size == 0:
                return False, "File is empty", file_size
            
            # Check if file is too small to be a valid tar
            if file_size < 1024:  # Less than 1KB
                return False, f"File too small ({file_size} bytes)", file_size
            
            # Try to open and validate tar file structure
            with tarfile.open(tar_path, 'r') as tar:
                # Try to get member list to validate structure
                members = tar.getmembers()
                if not members:
                    return False, "Tar file contains no members", file_size
                
                # Check for expected file types (tfrecord files)
                tfrecord_count = sum(1 for m in members if m.name.endswith('.tfrecord'))
                if tfrecord_count == 0:
                    return False, f"No .tfrecord files found in archive (found {len(members)} files)", file_size
                
                return True, f"Valid tar file with {tfrecord_count} tfrecord files", file_size
                
        except tarfile.ReadError as e:
            return False, f"Invalid tar format: {str(e)}", file_size
        except Exception as e:
            return False, f"Validation error: {str(e)}", file_size
    
    def extract_tar_with_progress(tar_path, extract_dir, file_index, total_files):
        """Extract tar file with detailed progress and error handling.
        
        Creates individual directory for each tar file and extracts contents there.
        This follows the pattern: mkdir training_0000 && tar -xvf training_0000.tar -C training_0000
        
        Args:
            tar_path (str): Path to tar file
            extract_dir (str): Base extraction directory
            file_index (int): Current file index (1-based)
            total_files (int): Total number of files
            
        Returns:
            tuple: (success, extracted_count, error_message)
        """
        try:
            # Get tar filename without extension to create subdirectory
            tar_filename = os.path.basename(tar_path)
            tar_name = os.path.splitext(tar_filename)[0]  # Remove .tar extension
            
            # Create individual directory for this tar file
            tar_extract_dir = os.path.join(extract_dir, tar_name)
            os.makedirs(tar_extract_dir, exist_ok=True)
            
            print(f"  → Creating directory: {tar_name}")
            print(f"  → Extracting to: {tar_extract_dir}")
            
            with tarfile.open(tar_path, 'r') as tar:
                members = tar.getmembers()
                extracted_count = 0
                
                print(f"  → Extracting {len(members)} files...")
                
                for member in members:
                    try:
                        # Extract to the specific subdirectory (equivalent to -C option)
                        tar.extract(member, path=tar_extract_dir)
                        extracted_count += 1
                        
                        # Show progress every 10 files or for small archives
                        if extracted_count % max(1, len(members) // 10) == 0 or len(members) < 20:
                            progress = (extracted_count / len(members)) * 100
                            print(f"    Progress: {extracted_count}/{len(members)} ({progress:.1f}%)")
                            
                    except Exception as member_error:
                        print(f"    ⚠ Warning: Failed to extract {member.name}: {str(member_error)}")
                        continue
                
                return True, extracted_count, None
                
        except Exception as e:
            return False, 0, str(e)
    
    print(f"\n{'='*60}")
    print(f"WAYMO DATASET PREPARATION")
    print(f"{'='*60}")
    print(f"Input directory: {args.load_dir}")
    
    # Set extraction directory
    extract_dir = args.extract_dir if hasattr(args, 'extract_dir') and args.extract_dir else args.load_dir
    print(f"Extraction directory: {extract_dir}")
    
    # Find tar files
    tar_files = glob.glob(os.path.join(args.load_dir, '*.tar'))
    if not tar_files:
        raise ValueError(f"No .tar files found in {args.load_dir}")
    
    print(f"Found {len(tar_files)} tar files")
    print(f"{'='*60}\n")
    
    # Create extraction directory
    os.makedirs(extract_dir, exist_ok=True)
    
    # Track extraction statistics
    successful_extractions = 0
    failed_extractions = 0
    total_extracted_files = 0
    failed_files = []
    
    # Validate and extract each tar file
    for i, tar_file in enumerate(tar_files, 1):
        filename = os.path.basename(tar_file)
        print(f"Processing {i}/{len(tar_files)}: {filename}")
        
        # Step 1: Validate tar file
        is_valid, validation_msg, file_size = validate_tar_file(tar_file)
        print(f"  File size: {file_size:,} bytes")
        
        if not is_valid:
            print(f"  ✗ Validation failed: {validation_msg}")
            failed_extractions += 1
            failed_files.append((filename, validation_msg))
            print(f"  → Skipping corrupted file\n")
            continue
        
        print(f"  ✓ Validation passed: {validation_msg}")
        
        # Step 2: Extract tar file
        success, extracted_count, error_msg = extract_tar_with_progress(
            tar_file, extract_dir, i, len(tar_files)
        )
        
        if success:
            print(f"  ✓ Successfully extracted {extracted_count} files")
            successful_extractions += 1
            total_extracted_files += extracted_count
        else:
            print(f"  ✗ Extraction failed: {error_msg}")
            failed_extractions += 1
            failed_files.append((filename, error_msg))
        
        print()  # Add spacing between files
    
    # Final summary
    print(f"{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tar files processed: {len(tar_files)}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    print(f"Total files extracted: {total_extracted_files}")
    
    if failed_files:
        print(f"\nFailed files:")
        for filename, error in failed_files:
            print(f"  • {filename}: {error}")
    
    # Verify final extraction results
    tfrecord_files = glob.glob(os.path.join(extract_dir, '*.tfrecord'))
    print(f"\nFinal verification:")
    print(f"Found {len(tfrecord_files)} .tfrecord files in extraction directory")
    
    if successful_extractions == 0:
        raise RuntimeError("No tar files were successfully extracted. Please check file integrity.")
    elif failed_extractions > 0:
        print(f"\n⚠ Warning: {failed_extractions} files failed to extract. You may want to:")
        print("  1. Re-download the corrupted files")
        print("  2. Check file integrity with checksums")
        print("  3. Verify sufficient disk space")
    
    print(f"{'='*60}\n")


def convert_dataset(args):
    """Convert Waymo tfrecord files to KITTI format."""
    # Validate input arguments
    if not os.path.exists(args.load_dir):
        raise ValueError(f"Input directory does not exist: {args.load_dir}")
    
    # Find tfrecord files
    tfrecord_files = glob(os.path.join(args.load_dir, '*.tfrecord'))
    if not tfrecord_files:
        raise ValueError(f"No .tfrecord files found in {args.load_dir}")
    
    print(f"\n{'='*60}")
    print(f"WAYMO TO KITTI CONVERSION")
    print(f"{'='*60}")
    print(f"Input directory: {args.load_dir}")
    print(f"Output directory: {args.save_dir}")
    print(f"Found {len(tfrecord_files)} tfrecord files")
    print(f"Split: {args.split}")
    print(f"Sampling interval: {args.sampled_interval}")
    print(f"Downsample factor: {args.downsample}")
    print(f"Number of processes: {args.num_proc}")
    print(f"Only GT boxes for camera: {args.only_gt_boxes_for_camera}")
    print(f"Save track ID: {args.save_track_id}")
    print(f"Save cam sync labels: {args.save_cam_sync_labels}")
    print(f"Max sweeps: {args.max_sweeps}")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize converter
    converter = Waymo2KITTI(
        load_dir=args.load_dir,
        save_dir=args.save_dir,
        prefix=args.prefix,
        workers=args.num_proc,
        only_gt_boxes_for_camera=args.only_gt_boxes_for_camera,
        sampled_interval=args.sampled_interval,
        subsample_interval=args.downsample,
        save_track_id=args.save_track_id,
        save_cam_sync_labels=args.save_cam_sync_labels,
        info_prefix=args.info_prefix,
        max_sweeps=args.max_sweeps
    )
    
    try:
        # Run conversion
        print("Starting conversion...")
        start_time = time.time()
        
        converter.convert()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"CONVERSION COMPLETED SUCCESSFULLY!")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"{'='*60}")
        
        # Perform validation if not skipped
        if not hasattr(args, 'skip_validation') or not args.skip_validation:
            print("\nPerforming output validation...")
            converter._validate_output()
        
        # Print final summary
        converter._print_conversion_summary()
        
        print(f"\n{'='*60}")
        print(f"DATASET CONVERSION COMPLETE!")
        print(f"Output saved to: {args.save_dir}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"CONVERSION FAILED!")
        print(f"Error: {str(e)}")
        print(f"{'='*60}")
        raise


def test_dataset(args):
    """Test and validate converted KITTI dataset."""
    print(f"\n{'='*60}")
    print(f"WAYMO DATASET TESTING")
    print(f"{'='*60}")
    print(f"Dataset directory: {args.save_dir}")
    print(f"Test type: {args.test_type}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(args.save_dir):
        raise ValueError(f"Dataset directory does not exist: {args.save_dir}")
    
    # Import test modules
    try:
        import sys
        test_script_path = os.path.join(os.path.dirname(__file__), 'test_integration.py')
        if os.path.exists(test_script_path):
            # Run integration tests
            print("Running integration tests...")
            import subprocess
            
            if args.test_type == 'all':
                cmd = [sys.executable, test_script_path, '--test', 'all']
            else:
                cmd = [sys.executable, test_script_path, '--test', args.test_type]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            print("Test Output:")
            print(result.stdout)
            if result.stderr:
                print("Test Errors:")
                print(result.stderr)
            
            if result.returncode == 0:
                print(f"\n{'='*60}")
                print(f"ALL TESTS PASSED!")
                print(f"{'='*60}\n")
            else:
                print(f"\n{'='*60}")
                print(f"TESTS FAILED!")
                print(f"Return code: {result.returncode}")
                print(f"{'='*60}\n")
                raise RuntimeError("Tests failed")
        else:
            print("Integration test script not found. Performing basic validation...")
            # Basic validation
            validate_kitti_structure(args.save_dir)
            
    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        raise


def validate_kitti_structure(dataset_dir):
    """Perform basic validation of KITTI dataset structure."""
    required_dirs = ['training/image_2', 'training/velodyne', 'training/label_2', 'training/calib']
    
    print("Validating KITTI dataset structure...")
    for req_dir in required_dirs:
        full_path = os.path.join(dataset_dir, req_dir)
        if os.path.exists(full_path):
            file_count = len([f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))])
            print(f"  ✓ {req_dir}: {file_count} files")
        else:
            print(f"  ✗ {req_dir}: Missing")
            raise ValueError(f"Required directory missing: {req_dir}")
    
    print("Basic validation completed successfully!")


if __name__ == '__main__':
    main()
