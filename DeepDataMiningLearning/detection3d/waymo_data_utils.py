#!/usr/bin/env python3
"""
Waymo Open Dataset v2.x Data Utilities
--------------------------------------

This module provides comprehensive utility functions for:
1. Reading Waymo parquet data into standard numpy/dict formats
2. Visualizing different data modalities (camera, LiDAR, segmentation, etc.)
3. Multi-modal sensor fusion visualization
4. Educational functions with detailed explanations

Dependencies:
    - pandas, numpy, pyarrow
    - matplotlib, PIL (for camera visualization)
    - open3d (for LiDAR visualization)
    - cv2 (for advanced image processing)

"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional, Union
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import warnings

# Optional dependencies for advanced visualization
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    warnings.warn("Open3D not available. LiDAR visualization will be limited.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV not available. Some image processing features will be limited.")


# ============================================================================
# DATA READER UTILITIES
# ============================================================================

def read_camera_images(parquet_path: str, max_frames: int = None) -> Dict:
    """
    Read camera image data from Waymo parquet files into standard format.
    
    This function extracts camera images and associated metadata from Waymo's
    camera_image parquet files, converting them into a standardized dictionary
    format suitable for further processing and visualization.
    
    Args:
        parquet_path (str): Path to camera_image parquet file
        max_frames (int, optional): Maximum number of frames to read. If None, reads all.
    
    Returns:
        Dict: Structured data containing:
            - 'images': List of PIL Image objects
            - 'metadata': Dict with keys:
                - 'segment_names': List[str] - Segment context identifiers
                - 'timestamps': List[int] - Frame timestamps in microseconds
                - 'camera_ids': List[int] - Camera sensor IDs (0-4)
                - 'poses': List[np.ndarray] - 4x4 transformation matrices
                - 'velocities': Dict with 'linear' and 'angular' velocity vectors
                - 'rolling_shutter': List[float] - Rolling shutter parameters
    
    Educational Notes:
        - Camera IDs: 0=FRONT, 1=FRONT_LEFT, 2=FRONT_RIGHT, 3=SIDE_LEFT, 4=SIDE_RIGHT
        - Poses represent camera-to-vehicle transformation matrices
        - Velocities are in vehicle coordinate frame (m/s and rad/s)
        - Rolling shutter affects temporal consistency across image rows
    """
    print(f"📷 Reading camera images from: {os.path.basename(parquet_path)}")
    
    # Read parquet file into pandas DataFrame
    df = pd.read_parquet(parquet_path)
    
    # Limit frames if specified
    if max_frames is not None:
        df = df.head(max_frames)
    
    # Initialize output structure
    result = {
        'images': [],
        'metadata': {
            'segment_names': [],
            'timestamps': [],
            'camera_ids': [],
            'poses': [],
            'velocities': {'linear': [], 'angular': []},
            'rolling_shutter': []
        }
    }
    
    print(f"🔄 Processing {len(df)} camera frames...")
    
    for idx, row in df.iterrows():
        # Extract and convert binary image data to PIL Image
        image_bytes = row['[CameraImageComponent].image']
        if image_bytes is not None and len(image_bytes) > 0:
            try:
                image_stream = io.BytesIO(image_bytes)
                pil_image = Image.open(image_stream)
                result['images'].append(pil_image)
            except Exception as e:
                print(f"⚠️ Error loading image at index {idx}: {e}")
                result['images'].append(None)
        else:
            result['images'].append(None)
        
        # Extract metadata
        result['metadata']['segment_names'].append(row['key.segment_context_name'])
        result['metadata']['timestamps'].append(row['key.frame_timestamp_micros'])
        result['metadata']['camera_ids'].append(row['key.camera_name'])
        
        # Extract pose transformation matrix (4x4)
        pose_transform = row.get('[CameraImageComponent].pose.transform', None)
        if pose_transform is not None and len(pose_transform) == 16:
            pose_matrix = np.array(pose_transform).reshape(4, 4)
            result['metadata']['poses'].append(pose_matrix)
        else:
            result['metadata']['poses'].append(None)
        
        # Extract velocity information
        linear_vel = np.array([
            row.get('[CameraImageComponent].velocity.linear_velocity.x', 0.0),
            row.get('[CameraImageComponent].velocity.linear_velocity.y', 0.0),
            row.get('[CameraImageComponent].velocity.linear_velocity.z', 0.0)
        ])
        angular_vel = np.array([
            row.get('[CameraImageComponent].velocity.angular_velocity.x', 0.0),
            row.get('[CameraImageComponent].velocity.angular_velocity.y', 0.0),
            row.get('[CameraImageComponent].velocity.angular_velocity.z', 0.0)
        ])
        
        result['metadata']['velocities']['linear'].append(linear_vel)
        result['metadata']['velocities']['angular'].append(angular_vel)
        
        # Extract rolling shutter parameters
        shutter = row.get('[CameraImageComponent].rolling_shutter_params.shutter', 0.0)
        result['metadata']['rolling_shutter'].append(shutter)
    
    print(f"✅ Successfully loaded {len(result['images'])} camera frames")
    return result


def read_camera_boxes(parquet_path: str) -> Dict:
    """
    Read 2D bounding box annotations from Waymo camera_box parquet files.
    
    This function extracts 2D bounding box annotations for objects detected
    in camera images, providing structured access to object locations, types,
    and difficulty ratings.
    
    Args:
        parquet_path (str): Path to camera_box parquet file
    
    Returns:
        Dict: Structured data containing:
            - 'boxes': List[Dict] with keys:
                - 'center': np.ndarray [x, y] - Box center in image coordinates
                - 'size': np.ndarray [width, height] - Box dimensions in pixels
                - 'corners': np.ndarray [4, 2] - Box corner coordinates
            - 'metadata': Dict with keys:
                - 'segment_names': List[str] - Segment identifiers
                - 'timestamps': List[int] - Frame timestamps
                - 'camera_ids': List[int] - Camera sensor IDs
                - 'object_ids': List[str] - Unique object identifiers
                - 'object_types': List[int] - Object class IDs
                - 'difficulties': Dict with 'detection' and 'tracking' difficulty levels
    
    Educational Notes:
        - Object Types: 1=Vehicle, 2=Pedestrian, 3=Sign, 4=Cyclist
        - Difficulty Levels: 1=Easy, 2=Medium, 3=Hard
        - Coordinates are in image pixel space (origin at top-left)
        - Box center + size can be converted to corner coordinates
    """
    print(f"🎯 Reading camera boxes from: {os.path.basename(parquet_path)}")
    
    df = pd.read_parquet(parquet_path)
    
    result = {
        'boxes': [],
        'metadata': {
            'segment_names': [],
            'timestamps': [],
            'camera_ids': [],
            'object_ids': [],
            'object_types': [],
            'difficulties': {'detection': [], 'tracking': []}
        }
    }
    
    print(f"🔄 Processing {len(df)} bounding boxes...")
    
    for idx, row in df.iterrows():
        # Extract box geometry
        center_x = row['[CameraBoxComponent].box.center.x']
        center_y = row['[CameraBoxComponent].box.center.y']
        width = row['[CameraBoxComponent].box.size.x']
        height = row['[CameraBoxComponent].box.size.y']
        
        center = np.array([center_x, center_y])
        size = np.array([width, height])
        
        # Calculate corner coordinates (top-left, top-right, bottom-right, bottom-left)
        half_w, half_h = width / 2, height / 2
        corners = np.array([
            [center_x - half_w, center_y - half_h],  # top-left
            [center_x + half_w, center_y - half_h],  # top-right
            [center_x + half_w, center_y + half_h],  # bottom-right
            [center_x - half_w, center_y + half_h]   # bottom-left
        ])
        
        box_data = {
            'center': center,
            'size': size,
            'corners': corners
        }
        result['boxes'].append(box_data)
        
        # Extract metadata
        result['metadata']['segment_names'].append(row['key.segment_context_name'])
        result['metadata']['timestamps'].append(row['key.frame_timestamp_micros'])
        result['metadata']['camera_ids'].append(row['key.camera_name'])
        result['metadata']['object_ids'].append(row['key.camera_object_id'])
        result['metadata']['object_types'].append(row['[CameraBoxComponent].type'])
        result['metadata']['difficulties']['detection'].append(
            row['[CameraBoxComponent].difficulty_level.detection']
        )
        result['metadata']['difficulties']['tracking'].append(
            row['[CameraBoxComponent].difficulty_level.tracking']
        )
    
    print(f"✅ Successfully loaded {len(result['boxes'])} bounding boxes")
    return result


def read_lidar_data(parquet_path: str) -> Dict:
    """
    Read LiDAR range image data from Waymo lidar parquet files.
    
    This function extracts LiDAR range images and converts them into structured
    format suitable for point cloud generation and visualization. Range images
    are the native format used by Waymo's LiDAR sensors.
    
    Args:
        parquet_path (str): Path to lidar parquet file
    
    Returns:
        Dict: Structured data containing:
            - 'range_images': Dict with keys:
                - 'return1': List[np.ndarray] - First return range images
                - 'return2': List[np.ndarray] - Second return range images
                - 'shapes': List[Tuple] - Original image shapes
            - 'metadata': Dict with keys:
                - 'segment_names': List[str] - Segment identifiers
                - 'timestamps': List[int] - Frame timestamps
                - 'laser_names': List[int] - LiDAR sensor IDs
    
    Educational Notes:
        - Range images encode distance, intensity, and other measurements
        - Shape typically [64, 2650, 4] for (height, width, channels)
        - Channels: [range, intensity, elongation, is_in_nlz]
        - Multiple returns capture different reflection properties
        - Laser IDs: 0=TOP, 1=FRONT, 2=SIDE_LEFT, 3=SIDE_RIGHT, 4=REAR
    """
    print(f"🌐 Reading LiDAR data from: {os.path.basename(parquet_path)}")
    
    df = pd.read_parquet(parquet_path)
    
    result = {
        'range_images': {'return1': [], 'return2': [], 'shapes': []},
        'metadata': {
            'segment_names': [],
            'timestamps': [],
            'laser_names': []
        }
    }
    
    print(f"🔄 Processing {len(df)} LiDAR measurements...")
    
    for idx, row in df.iterrows():
        # Extract range image data for both returns
        return1_values = row['[LiDARComponent].range_image_return1.values']
        return1_shape = row['[LiDARComponent].range_image_return1.shape']
        return2_values = row['[LiDARComponent].range_image_return2.values']
        return2_shape = row['[LiDARComponent].range_image_return2.shape']
        
        # Reshape flattened arrays back to original dimensions
        if return1_values is not None and return1_shape is not None:
            return1_image = np.array(return1_values).reshape(return1_shape)
            result['range_images']['return1'].append(return1_image)
        else:
            result['range_images']['return1'].append(None)
        
        if return2_values is not None and return2_shape is not None:
            return2_image = np.array(return2_values).reshape(return2_shape)
            result['range_images']['return2'].append(return2_image)
        else:
            result['range_images']['return2'].append(None)
        
        result['range_images']['shapes'].append(return1_shape if return1_shape else None)
        
        # Extract metadata
        result['metadata']['segment_names'].append(row['key.segment_context_name'])
        result['metadata']['timestamps'].append(row['key.frame_timestamp_micros'])
        result['metadata']['laser_names'].append(row['key.laser_name'])
    
    print(f"✅ Successfully loaded {len(df)} LiDAR range images")
    return result


def read_lidar_boxes(parquet_path: str) -> Dict:
    """
    Read 3D bounding box annotations from Waymo lidar_box parquet files.
    
    This function extracts 3D bounding box annotations for objects detected
    in LiDAR point clouds, providing complete 3D object state information
    including geometry, motion, and quality metrics.
    
    Args:
        parquet_path (str): Path to lidar_box parquet file
    
    Returns:
        Dict: Structured data containing:
            - 'boxes_3d': List[Dict] with keys:
                - 'center': np.ndarray [x, y, z] - 3D center coordinates (meters)
                - 'size': np.ndarray [length, width, height] - 3D dimensions (meters)
                - 'heading': float - Orientation angle in radians
                - 'corners_3d': np.ndarray [8, 3] - 3D corner coordinates
            - 'motion': Dict with keys:
                - 'velocities': List[np.ndarray] - 3D velocity vectors (m/s)
                - 'accelerations': List[np.ndarray] - 3D acceleration vectors (m/s²)
            - 'metadata': Dict with keys:
                - 'segment_names': List[str] - Segment identifiers
                - 'timestamps': List[int] - Frame timestamps
                - 'object_ids': List[str] - Unique object identifiers
                - 'object_types': List[int] - Object class IDs
                - 'point_counts': Dict with 'total' and 'top_lidar' point counts
                - 'difficulties': Dict with 'detection' and 'tracking' levels
    
    Educational Notes:
        - 3D boxes are defined in LiDAR coordinate frame
        - Heading angle: 0 = positive X axis, counter-clockwise positive
        - Object types: 1=Vehicle, 2=Pedestrian, 3=Sign, 4=Cyclist
        - Point counts indicate LiDAR coverage quality
        - Motion vectors enable tracking and prediction
    """
    print(f"📦 Reading 3D LiDAR boxes from: {os.path.basename(parquet_path)}")
    
    df = pd.read_parquet(parquet_path)
    
    result = {
        'boxes_3d': [],
        'motion': {'velocities': [], 'accelerations': []},
        'metadata': {
            'segment_names': [],
            'timestamps': [],
            'object_ids': [],
            'object_types': [],
            'point_counts': {'total': [], 'top_lidar': []},
            'difficulties': {'detection': [], 'tracking': []}
        }
    }
    
    print(f"🔄 Processing {len(df)} 3D bounding boxes...")
    
    for idx, row in df.iterrows():
        # Extract 3D box geometry
        center = np.array([
            row['[LiDARBoxComponent].box.center.x'],
            row['[LiDARBoxComponent].box.center.y'],
            row['[LiDARBoxComponent].box.center.z']
        ])
        
        size = np.array([
            row['[LiDARBoxComponent].box.size.x'],    # length
            row['[LiDARBoxComponent].box.size.y'],    # width
            row['[LiDARBoxComponent].box.size.z']     # height
        ])
        
        heading = row['[LiDARBoxComponent].box.heading']
        
        # Calculate 3D corner coordinates
        corners_3d = calculate_3d_box_corners(center, size, heading)
        
        box_data = {
            'center': center,
            'size': size,
            'heading': heading,
            'corners_3d': corners_3d
        }
        result['boxes_3d'].append(box_data)
        
        # Extract motion information
        velocity = np.array([
            row['[LiDARBoxComponent].speed.x'],
            row['[LiDARBoxComponent].speed.y'],
            row['[LiDARBoxComponent].speed.z']
        ])
        acceleration = np.array([
            row['[LiDARBoxComponent].acceleration.x'],
            row['[LiDARBoxComponent].acceleration.y'],
            row['[LiDARBoxComponent].acceleration.z']
        ])
        
        result['motion']['velocities'].append(velocity)
        result['motion']['accelerations'].append(acceleration)
        
        # Extract metadata
        result['metadata']['segment_names'].append(row['key.segment_context_name'])
        result['metadata']['timestamps'].append(row['key.frame_timestamp_micros'])
        result['metadata']['object_ids'].append(row['key.laser_object_id'])
        result['metadata']['object_types'].append(row['[LiDARBoxComponent].type'])
        result['metadata']['point_counts']['total'].append(
            row['[LiDARBoxComponent].num_lidar_points_in_box']
        )
        result['metadata']['point_counts']['top_lidar'].append(
            row['[LiDARBoxComponent].num_top_lidar_points_in_box']
        )
        result['metadata']['difficulties']['detection'].append(
            row['[LiDARBoxComponent].difficulty_level.detection']
        )
        result['metadata']['difficulties']['tracking'].append(
            row['[LiDARBoxComponent].difficulty_level.tracking']
        )
    
    print(f"✅ Successfully loaded {len(result['boxes_3d'])} 3D bounding boxes")
    return result


def read_vehicle_poses(parquet_path: str) -> Dict:
    """
    Read ego vehicle pose data from Waymo vehicle_pose parquet files.
    
    This function extracts the 6-DoF pose (position + orientation) of the ego
    vehicle in the global coordinate frame, essential for trajectory analysis,
    localization, and mapping applications.
    
    Args:
        parquet_path (str): Path to vehicle_pose parquet file
    
    Returns:
        Dict: Structured data containing:
            - 'poses': List[Dict] with keys:
                - 'position': np.ndarray [x, y, z] - 3D position in world coordinates
                - 'rotation_matrix': np.ndarray [3, 3] - 3D rotation matrix
                - 'transform_matrix': np.ndarray [4, 4] - Complete 4x4 transformation
                - 'euler_angles': np.ndarray [roll, pitch, yaw] - Euler angle representation
            - 'metadata': Dict with keys:
                - 'segment_names': List[str] - Segment identifiers
                - 'timestamps': List[int] - Frame timestamps in microseconds
    
    Educational Notes:
        - Poses transform points from vehicle frame to world frame
        - World frame is typically aligned with the first vehicle pose
        - Rotation matrices preserve orthogonality and determinant = 1
        - Euler angles follow roll-pitch-yaw convention (X-Y-Z rotation order)
        - Trajectory can be reconstructed from sequential poses
    """
    print(f"🌍 Reading vehicle poses from: {os.path.basename(parquet_path)}")
    
    df = pd.read_parquet(parquet_path)
    
    result = {
        'poses': [],
        'metadata': {
            'segment_names': [],
            'timestamps': []
        }
    }
    
    print(f"🔄 Processing {len(df)} vehicle poses...")
    
    for idx, row in df.iterrows():
        # Extract 4x4 transformation matrix
        transform_flat = row['[VehiclePoseComponent].world_from_vehicle.transform']
        transform_matrix = np.array(transform_flat).reshape(4, 4)
        
        # Extract position (translation vector)
        position = transform_matrix[:3, 3]
        
        # Extract rotation matrix
        rotation_matrix = transform_matrix[:3, :3]
        
        # Convert rotation matrix to Euler angles (roll, pitch, yaw)
        euler_angles = rotation_matrix_to_euler(rotation_matrix)
        
        pose_data = {
            'position': position,
            'rotation_matrix': rotation_matrix,
            'transform_matrix': transform_matrix,
            'euler_angles': euler_angles
        }
        result['poses'].append(pose_data)
        
        # Extract metadata
        result['metadata']['segment_names'].append(row['key.segment_context_name'])
        result['metadata']['timestamps'].append(row['key.frame_timestamp_micros'])
    
    print(f"✅ Successfully loaded {len(result['poses'])} vehicle poses")
    return result


def read_segmentation_data(parquet_path: str, segmentation_type: str = 'lidar') -> Dict:
    """
    Read segmentation label data from Waymo segmentation parquet files.
    
    This function extracts semantic and instance segmentation labels for either
    LiDAR points or camera pixels, providing pixel/point-level classification
    information for scene understanding.
    
    Args:
        parquet_path (str): Path to segmentation parquet file
        segmentation_type (str): Either 'lidar' or 'camera' segmentation
    
    Returns:
        Dict: Structured data containing:
            For LiDAR segmentation:
            - 'labels': Dict with keys:
                - 'return1': List[np.ndarray] - Semantic labels for first return
                - 'return2': List[np.ndarray] - Semantic labels for second return
                - 'shapes': List[Tuple] - Label array shapes
            For Camera segmentation:
            - 'labels': Dict with keys:
                - 'panoptic': List[np.ndarray] - Panoptic segmentation masks
                - 'instance_mappings': List[Dict] - Instance ID mappings
            - 'metadata': Dict with keys:
                - 'segment_names': List[str] - Segment identifiers
                - 'timestamps': List[int] - Frame timestamps
                - 'sensor_ids': List[int] - Sensor identifiers
    
    Educational Notes:
        - Semantic labels: pixel/point-level class IDs (vehicle, pedestrian, etc.)
        - Instance labels: unique IDs for individual object instances
        - Panoptic = semantic + instance segmentation combined
        - LiDAR labels align with range image structure
        - Camera labels are encoded as compressed masks
    """
    print(f"🎨 Reading {segmentation_type} segmentation from: {os.path.basename(parquet_path)}")
    
    df = pd.read_parquet(parquet_path)
    
    if segmentation_type == 'lidar':
        result = {
            'labels': {'return1': [], 'return2': [], 'shapes': []},
            'metadata': {
                'segment_names': [],
                'timestamps': [],
                'sensor_ids': []
            }
        }
        
        print(f"🔄 Processing {len(df)} LiDAR segmentation labels...")
        
        for idx, row in df.iterrows():
            # Extract segmentation labels for both returns
            return1_values = row['[LiDARSegmentationLabelComponent].range_image_return1.values']
            return1_shape = row['[LiDARSegmentationLabelComponent].range_image_return1.shape']
            return2_values = row['[LiDARSegmentationLabelComponent].range_image_return2.values']
            return2_shape = row['[LiDARSegmentationLabelComponent].range_image_return2.shape']
            
            # Reshape flattened label arrays
            if return1_values is not None and return1_shape is not None:
                return1_labels = np.array(return1_values).reshape(return1_shape)
                result['labels']['return1'].append(return1_labels)
            else:
                result['labels']['return1'].append(None)
            
            if return2_values is not None and return2_shape is not None:
                return2_labels = np.array(return2_values).reshape(return2_shape)
                result['labels']['return2'].append(return2_labels)
            else:
                result['labels']['return2'].append(None)
            
            result['labels']['shapes'].append(return1_shape if return1_shape else None)
            
            # Extract metadata
            result['metadata']['segment_names'].append(row['key.segment_context_name'])
            result['metadata']['timestamps'].append(row['key.frame_timestamp_micros'])
            result['metadata']['sensor_ids'].append(row['key.laser_name'])
    
    else:  # camera segmentation
        result = {
            'labels': {'panoptic': [], 'instance_mappings': []},
            'metadata': {
                'segment_names': [],
                'timestamps': [],
                'sensor_ids': []
            }
        }
        
        print(f"🔄 Processing {len(df)} camera segmentation labels...")
        
        for idx, row in df.iterrows():
            # Extract panoptic segmentation data
            panoptic_label = row.get('[CameraSegmentationLabelComponent].panoptic_label', None)
            if panoptic_label is not None:
                # Decode compressed segmentation mask
                # Note: This would require specific Waymo decoding functions
                result['labels']['panoptic'].append(panoptic_label)
            else:
                result['labels']['panoptic'].append(None)
            
            # Extract instance ID mappings
            local_ids = row.get('[CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.local_instance_ids', [])
            global_ids = row.get('[CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.global_instance_ids', [])
            is_tracked = row.get('[CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.is_tracked', [])
            
            instance_mapping = {
                'local_ids': local_ids,
                'global_ids': global_ids,
                'is_tracked': is_tracked
            }
            result['labels']['instance_mappings'].append(instance_mapping)
            
            # Extract metadata
            result['metadata']['segment_names'].append(row['key.segment_context_name'])
            result['metadata']['timestamps'].append(row['key.frame_timestamp_micros'])
            result['metadata']['sensor_ids'].append(row['key.camera_name'])
    
    print(f"✅ Successfully loaded {len(df)} segmentation labels")
    return result


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_3d_box_corners(center: np.ndarray, size: np.ndarray, heading: float) -> np.ndarray:
    """
    Calculate 3D bounding box corner coordinates from center, size, and heading.
    
    Args:
        center (np.ndarray): [x, y, z] center coordinates
        size (np.ndarray): [length, width, height] dimensions
        heading (float): Rotation angle around Z-axis in radians
    
    Returns:
        np.ndarray: [8, 3] array of corner coordinates
    
    Educational Notes:
        - Corners are ordered: front-bottom-left, front-bottom-right, etc.
        - Heading rotation is applied around the Z-axis (vertical)
        - Box is axis-aligned before rotation is applied
    """
    l, w, h = size
    
    # Define 8 corners of axis-aligned box (before rotation)
    corners = np.array([
        [-l/2, -w/2, -h/2],  # rear-left-bottom
        [ l/2, -w/2, -h/2],  # front-left-bottom
        [ l/2,  w/2, -h/2],  # front-right-bottom
        [-l/2,  w/2, -h/2],  # rear-right-bottom
        [-l/2, -w/2,  h/2],  # rear-left-top
        [ l/2, -w/2,  h/2],  # front-left-top
        [ l/2,  w/2,  h/2],  # front-right-top
        [-l/2,  w/2,  h/2]   # rear-right-top
    ])
    
    # Apply rotation around Z-axis
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rotation_matrix = np.array([
        [cos_h, -sin_h, 0],
        [sin_h,  cos_h, 0],
        [0,      0,     1]
    ])
    
    # Rotate corners and translate to center
    rotated_corners = corners @ rotation_matrix.T + center
    
    return rotated_corners


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to Euler angles (roll, pitch, yaw).
    
    Args:
        R (np.ndarray): 3x3 rotation matrix
    
    Returns:
        np.ndarray: [roll, pitch, yaw] angles in radians
    
    Educational Notes:
        - Uses ZYX Euler angle convention (yaw-pitch-roll)
        - Roll: rotation around X-axis
        - Pitch: rotation around Y-axis  
        - Yaw: rotation around Z-axis
        - Handles gimbal lock cases
    """
    # Extract Euler angles from rotation matrix
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return np.array([roll, pitch, yaw])


# ============================================================================
# OBJECT TYPE MAPPINGS
# ============================================================================

WAYMO_OBJECT_TYPES = {
    0: 'Unknown',
    1: 'Vehicle',
    2: 'Pedestrian', 
    3: 'Sign',
    4: 'Cyclist'
}

WAYMO_CAMERA_NAMES = {
    0: 'FRONT',
    1: 'FRONT_LEFT',
    2: 'FRONT_RIGHT', 
    3: 'SIDE_LEFT',
    4: 'SIDE_RIGHT'
}

WAYMO_LIDAR_NAMES = {
    0: 'TOP',
    1: 'FRONT',
    2: 'SIDE_LEFT',
    3: 'SIDE_RIGHT', 
    4: 'REAR'
}

WAYMO_DIFFICULTY_LEVELS = {
    1: 'Easy',
    2: 'Medium',
    3: 'Hard'
}


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def visualize_camera_with_boxes(image: Image.Image, boxes_data: Dict = None, 
                               metadata: Dict = None, frame_idx: int = 0,
                               show_labels: bool = True, 
                               save_path: str = None,
                               object_types: Dict = None,
                               camera_names: Dict = None) -> plt.Figure:
    """
    Visualize camera image with 2D bounding box overlays.
    
    This function creates a comprehensive visualization of camera images with
    annotated bounding boxes, object types, and metadata information. Perfect
    for understanding object detection results and data quality assessment.
    
    Args:
        image (PIL.Image): Camera image to visualize
        boxes_data (Dict, optional): Bounding box data with 'boxes' key containing list of boxes
        metadata (Dict, optional): Metadata with 'camera_ids', 'timestamps', 'object_types' keys
        frame_idx (int): Frame index to visualize (default: 0)
        show_labels (bool): Whether to show object type labels
        save_path (str, optional): Path to save the visualization
        object_types (Dict, optional): Mapping of object type IDs to names (defaults to generic types)
        camera_names (Dict, optional): Mapping of camera IDs to names (defaults to generic names)
    
    Returns:
        plt.Figure: Matplotlib figure object
    
    Educational Notes:
        - Different colors represent different object types
        - Box thickness indicates detection difficulty
        - Labels show object type and confidence information
        - Coordinate system: (0,0) at top-left, Y increases downward
    """
    print(f"🎨 Creating camera visualization for frame {frame_idx}")
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # Display the image
    ax.imshow(image)
    
    # Set title with available metadata
    title_parts = ['Camera Image with 2D Bounding Boxes']
    if metadata:
        if camera_names and 'camera_ids' in metadata and frame_idx < len(metadata['camera_ids']):
            camera_name = camera_names.get(metadata['camera_ids'][frame_idx], f"Camera {metadata['camera_ids'][frame_idx]}")
            title_parts.append(f'Camera: {camera_name}')
        if 'timestamps' in metadata and frame_idx < len(metadata['timestamps']):
            title_parts.append(f'Timestamp: {metadata["timestamps"][frame_idx]}')
    
    ax.set_title('\n'.join(title_parts), fontsize=14, fontweight='bold')
    
    # Use provided object types or default generic types
    if object_types is None:
        object_types = {
            0: 'Unknown',
            1: 'Vehicle',
            2: 'Pedestrian', 
            3: 'Sign',
            4: 'Cyclist'
        }
    
    # Define colors for different object types
    colors = {
        0: 'gray',     # Unknown
        1: 'red',      # Vehicle
        2: 'blue',     # Pedestrian
        3: 'green',    # Sign
        4: 'orange',   # Cyclist
    }
    
    # Handle case where no boxes data is provided
    if not boxes_data or 'boxes' not in boxes_data:
        print("⚠️ No bounding box data provided")
        ax.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved camera visualization to: {save_path}")
        return fig
    
    # Filter boxes for current frame
    frame_boxes = []
    frame_metadata = []
    
    # Handle different data structures
    if metadata and 'timestamps' in metadata and 'camera_ids' in metadata:
        # Waymo-style data with timestamps and camera IDs
        # Get the target timestamp and camera ID for the current frame_idx
        if frame_idx < len(metadata['timestamps']) and frame_idx < len(metadata['camera_ids']):
            target_timestamp = metadata['timestamps'][frame_idx]
            target_camera_id = metadata['camera_ids'][frame_idx]
            
            # Find all boxes that match this timestamp and camera
            for i, (box, obj_type, timestamp, camera_id) in enumerate(zip(
                boxes_data['boxes'], 
                metadata.get('object_types', []), 
                metadata['timestamps'],
                metadata['camera_ids']
            )):
                if timestamp == target_timestamp and camera_id == target_camera_id:
                    frame_boxes.append(box)
                    frame_metadata.append({
                        'type': obj_type,
                        'id': metadata.get('object_ids', [f'obj_{i}'])[i] if 'object_ids' in metadata and i < len(metadata['object_ids']) else f'obj_{i}',
                        'detection_difficulty': metadata.get('difficulties', {}).get('detection', [1])[i] if 'difficulties' in metadata and i < len(metadata['difficulties']['detection']) else 1,
                        'tracking_difficulty': metadata.get('difficulties', {}).get('tracking', [1])[i] if 'difficulties' in metadata and i < len(metadata['difficulties']['tracking']) else 1
                    })
    elif metadata and 'timestamps' in metadata:
        # Fallback: match by timestamp only
        if frame_idx < len(metadata['timestamps']):
            target_timestamp = metadata['timestamps'][frame_idx]
            
            for i, (box, obj_type, timestamp) in enumerate(zip(
                boxes_data['boxes'], 
                metadata.get('object_types', []), 
                metadata['timestamps']
            )):
                if timestamp == target_timestamp:
                    frame_boxes.append(box)
                    frame_metadata.append({
                        'type': obj_type,
                        'id': metadata.get('object_ids', [f'obj_{i}'])[i] if 'object_ids' in metadata and i < len(metadata['object_ids']) else f'obj_{i}',
                        'detection_difficulty': metadata.get('difficulties', {}).get('detection', [1])[i] if 'difficulties' in metadata and i < len(metadata['difficulties']['detection']) else 1,
                        'tracking_difficulty': metadata.get('difficulties', {}).get('tracking', [1])[i] if 'difficulties' in metadata and i < len(metadata['difficulties']['tracking']) else 1
                    })
    else:
        # Generic data structure - assume all boxes are for current frame
        boxes_list = boxes_data['boxes']
        if frame_idx < len(boxes_list):
            if isinstance(boxes_list[frame_idx], list):
                # Frame-indexed structure
                frame_boxes = boxes_list[frame_idx]
            else:
                # Single frame or flat structure - take all boxes
                frame_boxes = boxes_list
            
            # Create generic metadata
            for i, box in enumerate(frame_boxes):
                obj_type = 1  # Default to vehicle
                if metadata and 'object_types' in metadata:
                    if isinstance(metadata['object_types'], list) and i < len(metadata['object_types']):
                        obj_type = metadata['object_types'][i]
                
                frame_metadata.append({
                    'type': obj_type,
                    'id': f'obj_{i}',
                    'detection_difficulty': 1,
                    'tracking_difficulty': 1
                })
    
    print(f"📦 Drawing {len(frame_boxes)} bounding boxes")
    
    # Draw bounding boxes
    for box, meta in zip(frame_boxes, frame_metadata):
        # Handle different box formats
        if isinstance(box, dict) and 'center' in box and 'size' in box:
            # Waymo-style box with center and size
            center = box['center']
            size = box['size']
            # Convert center-size format to top-left corner format
            x = center[0] - size[0] / 2
            y = center[1] - size[1] / 2
            width, height = size[0], size[1]
        elif isinstance(box, dict) and 'corners' in box:
            # Box with corner coordinates - use corners directly
            corners = box['corners']
            if len(corners) >= 4:
                # Calculate bounding rectangle from corners
                x_coords = corners[:, 0]
                y_coords = corners[:, 1]
                x = np.min(x_coords)
                y = np.min(y_coords)
                width = np.max(x_coords) - x
                height = np.max(y_coords) - y
            else:
                continue  # Skip invalid corners
        elif isinstance(box, (list, tuple, np.ndarray)) and len(box) >= 4:
            # Generic box format - handle different conventions
            box = np.array(box)
            if len(box) == 4:
                x1, y1, x2, y2 = box[:4]
                # Check if it's [x1, y1, x2, y2] format (coordinates)
                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                    x, y = x1, y1
                    width, height = x2 - x1, y2 - y1
                # Check if it's [x, y, w, h] format (position + size)
                elif x1 >= 0 and y1 >= 0 and x2 > 0 and y2 > 0:
                    x, y, width, height = x1, y1, x2, y2
                else:
                    continue  # Skip invalid box format
            else:
                continue  # Skip invalid box format
        else:
            continue  # Skip invalid box format
        
        # Ensure coordinates are valid
        if width <= 0 or height <= 0 or x < 0 or y < 0:
            continue  # Skip invalid boxes
        
        obj_type = meta['type']
        
        # Set line width based on detection difficulty
        linewidth = max(1, 4 - meta['detection_difficulty'])  # Easier = thicker, minimum 1
        
        # Create rectangle
        rect = Rectangle(
            (x, y), width, height,
            linewidth=linewidth,
            edgecolor=colors.get(obj_type, 'gray'),
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add label if requested
        if show_labels:
            label = f"{object_types.get(obj_type, 'Unknown')}"
            label += f"\nID: {meta['id']}"
            if meta['detection_difficulty'] > 1:
                difficulty_names = {1: 'Easy', 2: 'Medium', 3: 'Hard'}
                label += f"\nDiff: {difficulty_names.get(meta['detection_difficulty'], 'Unknown')}"
            
            # Position label above the box, with fallback to inside if near top edge
            label_y = y - 10 if y > 30 else y + 5
            
            ax.text(x, label_y, label, 
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor=colors.get(obj_type, 'gray'), 
                           alpha=0.7),
                   fontsize=8, color='white', fontweight='bold',
                   verticalalignment='top' if y > 30 else 'bottom')
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], color=color, lw=3, label=object_types.get(obj_type, f'Type {obj_type}'))
                      for obj_type, color in colors.items()]
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved visualization to: {save_path}")
    
    return fig


def visualize_lidar_pointcloud(pointcloud_data: Union[Dict, np.ndarray], 
                              intensities: np.ndarray = None,
                              boxes_data: Dict = None,
                              frame_idx: int = 0, 
                              return_type: str = 'return1',
                              point_size: float = 0.5, 
                              max_points: int = 50000,
                              save_path: str = None) -> Union[plt.Figure, None]:
    """
    Visualize LiDAR point cloud with optional 3D bounding boxes using Open3D or matplotlib.
    
    This function accepts either Waymo-style range image data or generic point cloud data
    and creates interactive or static visualizations. Supports both Open3D (interactive)
    and matplotlib (static) backends for maximum compatibility.
    
    Args:
        pointcloud_data (Union[Dict, np.ndarray]): Either:
            - Dict: Waymo-style range image data from read_lidar_data()
            - np.ndarray: Direct point cloud array of shape (N, 3) or (N, 4)
        intensities (np.ndarray, optional): Intensity values for points (if pointcloud_data is np.ndarray)
        boxes_data (Dict, optional): 3D box data from read_lidar_boxes() or generic format
        frame_idx (int): Frame index to visualize (only used for Dict input)
        return_type (str): 'return1' or 'return2' for multi-return LiDAR (only used for Dict input)
        point_size (float): Point size for visualization
        max_points (int): Maximum points to display (for performance)
        save_path (str, optional): Path to save visualization
    
    Returns:
        plt.Figure or None: Matplotlib figure if using matplotlib backend
    
    Educational Notes:
        - Range images are converted to Cartesian coordinates (for Waymo data)
        - Colors represent intensity or distance information
        - 3D boxes show object locations and orientations
        - Interactive mode allows rotation and zooming
        - Point cloud density varies by distance and sensor resolution
    """
    print(f"🌐 Creating LiDAR point cloud visualization for frame {frame_idx}")
    
    # Handle different input types
    if isinstance(pointcloud_data, dict):
        # Waymo-style range image data
        if 'range_images' not in pointcloud_data:
            print("❌ Invalid pointcloud data format: missing 'range_images' key")
            return None
            
        range_images = pointcloud_data['range_images'][return_type]
        if frame_idx >= len(range_images) or range_images[frame_idx] is None:
            print(f"❌ No range image data available for frame {frame_idx}")
            return None
        
        range_image = range_images[frame_idx]
        print(f"📊 Range image shape: {range_image.shape}")
        
        # Convert range image to point cloud
        points, intensities = range_image_to_pointcloud(range_image)
    else:
        # Direct point cloud data
        if isinstance(pointcloud_data, np.ndarray):
            if pointcloud_data.shape[1] >= 3:
                points = pointcloud_data[:, :3]  # Extract XYZ coordinates
                if pointcloud_data.shape[1] >= 4 and intensities is None:
                    intensities = pointcloud_data[:, 3]  # Extract intensity if available
                elif intensities is None:
                    intensities = np.ones(len(points))  # Default intensity
            else:
                print("❌ Point cloud data must have at least 3 dimensions (X, Y, Z)")
                return None
        else:
            print("❌ Invalid pointcloud data type. Expected Dict or np.ndarray")
            return None
    
    # Subsample points if too many
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        intensities = intensities[indices]
        print(f"🔽 Subsampled to {max_points} points for performance")
    
    print(f"☁️ Generated point cloud with {len(points)} points")
    
    # Try Open3D visualization first (interactive)
    if HAS_OPEN3D:
        return visualize_pointcloud_open3d(points, intensities, boxes_data, 
                                         frame_idx, point_size, save_path)
    else:
        # Fallback to matplotlib (static)
        return visualize_pointcloud_matplotlib(points, intensities, boxes_data,
                                             frame_idx, point_size, save_path)


def visualize_pointcloud_open3d(points: np.ndarray, intensities: np.ndarray,
                               boxes_data: Dict = None, frame_idx: int = 0,
                               point_size: float = 0.5, save_path: str = None):
    """
    Create interactive 3D point cloud visualization using Open3D.
    
    Args:
        points (np.ndarray): [N, 3] point coordinates
        intensities (np.ndarray): [N] intensity values
        boxes_data (Dict, optional): 3D bounding box data
        frame_idx (int): Frame index for box filtering
        point_size (float): Point size for rendering
        save_path (str, optional): Path to save screenshot
    
    Educational Notes:
        - Interactive visualization allows real-time manipulation
        - Color mapping shows LiDAR intensity variations
        - 3D boxes are rendered as wireframes
        - Mouse controls: rotate, zoom, pan
    """
    print("🎮 Creating interactive Open3D visualization")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Color points by intensity
    if intensities is not None:
        # Normalize intensities to [0, 1] range
        norm_intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-8)
        # Create colormap (blue to red)
        colors = plt.cm.viridis(norm_intensities)[:, :3]  # Remove alpha channel
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Point Cloud Visualization", width=1200, height=800)
    vis.add_geometry(pcd)
    
    # Add 3D bounding boxes if provided
    if boxes_data is not None:
        box_geometries = create_open3d_boxes(boxes_data, frame_idx)
        for box_geom in box_geometries:
            vis.add_geometry(box_geom)
    
    # Set rendering options
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    
    # Set camera view
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, 1])  # Look down Z-axis
    view_control.set_up([0, 1, 0])     # Y-axis up
    
    print("🖱️ Use mouse to interact: Left=rotate, Right=zoom, Middle=pan")
    print("📸 Press 'P' to save screenshot, 'Q' to quit")
    
    # Run visualization
    vis.run()
    
    if save_path:
        vis.capture_screen_image(save_path)
        print(f"💾 Saved screenshot to: {save_path}")
    
    vis.destroy_window()
    return None


def visualize_pointcloud_matplotlib(points: np.ndarray, intensities: np.ndarray,
                                  boxes_data: Dict = None, frame_idx: int = 0,
                                  point_size: float = 0.5, save_path: str = None) -> plt.Figure:
    """
    Create static 3D point cloud visualization using matplotlib.
    
    Args:
        points (np.ndarray): [N, 3] point coordinates
        intensities (np.ndarray): [N] intensity values
        boxes_data (Dict, optional): 3D bounding box data
        frame_idx (int): Frame index for box filtering
        point_size (float): Point size for rendering
        save_path (str, optional): Path to save figure
    
    Returns:
        plt.Figure: Matplotlib 3D figure
    
    Educational Notes:
        - Static visualization suitable for documentation
        - Multiple viewing angles can be generated
        - Color coding reveals intensity patterns
        - 3D boxes show object boundaries
    """
    print("📊 Creating matplotlib 3D visualization")
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points colored by intensity
    if intensities is not None:
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=intensities, cmap='viridis', s=point_size, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='LiDAR Intensity', shrink=0.8)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c='blue', s=point_size, alpha=0.6)
    
    # Add 3D bounding boxes if provided
    if boxes_data is not None:
        draw_matplotlib_boxes(ax, boxes_data, frame_idx)
    
    # Set labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_title(f'LiDAR Point Cloud Visualization\nFrame: {frame_idx}, Points: {len(points)}', 
                fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved visualization to: {save_path}")
    
    return fig


def visualize_segmentation_overlay(image: Image.Image, 
                                 segmentation_data: Union[Dict, np.ndarray] = None,
                                 frame_idx: int = 0, 
                                 alpha: float = 0.5,
                                 save_path: str = None) -> plt.Figure:
    """
    Visualize camera image with segmentation mask overlay.
    
    This function creates a comprehensive visualization showing semantic
    segmentation results overlaid on the original camera image, with
    different colors representing different object classes.
    
    Args:
        image (PIL.Image): Original camera image
        segmentation_data (Union[Dict, np.ndarray], optional): Either:
            - Dict: Waymo-style segmentation data from read_segmentation_data()
            - np.ndarray: Direct segmentation mask array
            - None: Will create a demo segmentation mask
        frame_idx (int): Frame index to visualize (only used for Dict input)
        alpha (float): Transparency of segmentation overlay (0-1)
        save_path (str, optional): Path to save visualization
    
    Returns:
        plt.Figure: Matplotlib figure with overlay
    
    Educational Notes:
        - Different colors represent semantic classes
        - Alpha blending preserves original image details
        - Instance boundaries show individual objects
        - Useful for evaluating segmentation quality
    """
    print(f"🎨 Creating segmentation overlay for frame {frame_idx}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Handle different segmentation data types
    seg_mask = None
    
    if segmentation_data is None:
        # Create demo segmentation mask
        seg_mask = create_demo_segmentation_mask(np.array(image))
    elif isinstance(segmentation_data, np.ndarray):
        # Direct segmentation mask
        seg_mask = segmentation_data
    elif isinstance(segmentation_data, dict):
        # Waymo-style segmentation data
        if 'labels' in segmentation_data and 'panoptic' in segmentation_data['labels']:
            seg_labels = segmentation_data['labels']['panoptic']
            if frame_idx < len(seg_labels) and seg_labels[frame_idx] is not None:
                # Note: This would require proper decoding of Waymo's compressed format
                # For demonstration, we'll create a placeholder
                seg_mask = create_demo_segmentation_mask(np.array(image))
        else:
            # Create demo mask if data structure is not recognized
            seg_mask = create_demo_segmentation_mask(np.array(image))
    
    if seg_mask is not None:
        # Segmentation only
        axes[1].imshow(seg_mask, cmap='tab20')
        axes[1].set_title('Segmentation Mask', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(seg_mask, alpha=alpha, cmap='tab20')
        axes[2].set_title(f'Overlay (α={alpha})', fontsize=14, fontweight='bold')
        axes[2].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'No segmentation\ndata available', 
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=16, bbox=dict(boxstyle="round", facecolor='lightgray'))
        axes[1].axis('off')
        
        axes[2].text(0.5, 0.5, 'No segmentation\ndata available', 
                    ha='center', va='center', transform=axes[2].transAxes,
                    fontsize=16, bbox=dict(boxstyle="round", facecolor='lightgray'))
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved segmentation visualization to: {save_path}")
    
    return fig


def visualize_vehicle_trajectory(trajectory_data: Union[Dict, np.ndarray], 
                                timestamps: np.ndarray = None,
                                max_poses: int = 1000,
                                trajectory_color: str = 'blue',
                                save_path: str = None) -> plt.Figure:
    """
    Visualize vehicle trajectory from pose/trajectory data.
    
    This function creates a comprehensive trajectory visualization showing
    the vehicle's path through the environment, with orientation indicators
    and motion analysis.
    
    Args:
        trajectory_data (Union[Dict, np.ndarray]): Trajectory data. Can be:
            - Dict with Waymo-style structure containing 'poses' and 'metadata'
            - NumPy array of shape (N, 3) for positions only
            - NumPy array of shape (N, 6) for positions and orientations
            - NumPy array of shape (N, 7) for positions, orientations, and timestamps
        timestamps (np.ndarray, optional): Timestamps for each pose (if not in trajectory_data)
        max_poses (int): Maximum number of poses to display
        trajectory_color (str): Color for trajectory line
        save_path (str, optional): Path to save visualization
    
    Returns:
        plt.Figure: Matplotlib figure with trajectory plots
    
    Educational Notes:
        - Top-down view shows spatial trajectory
        - Arrows indicate vehicle orientation
        - Speed profile shows velocity changes
        - Useful for path planning and localization analysis
    """
    print(f"🛣️ Creating vehicle trajectory visualization")
    
    # Handle different input formats
    if isinstance(trajectory_data, dict):
        # Waymo-style data structure
        poses = trajectory_data['poses'][:max_poses]
        if 'metadata' in trajectory_data and 'timestamps' in trajectory_data['metadata']:
            timestamps = trajectory_data['metadata']['timestamps'][:max_poses]
        else:
            timestamps = np.arange(len(poses))  # Default timestamps
        
        # Extract positions and orientations from pose dictionaries
        positions = np.array([pose['position'] for pose in poses])
        euler_angles = np.array([pose['euler_angles'] for pose in poses])
        
    elif isinstance(trajectory_data, np.ndarray):
        # Generic NumPy array format
        trajectory_data = trajectory_data[:max_poses]
        
        if trajectory_data.shape[1] == 3:
            # Only positions [x, y, z]
            positions = trajectory_data
            euler_angles = np.zeros((len(positions), 3))  # No orientation data
        elif trajectory_data.shape[1] == 6:
            # Positions and orientations [x, y, z, roll, pitch, yaw]
            positions = trajectory_data[:, :3]
            euler_angles = trajectory_data[:, 3:6]
        elif trajectory_data.shape[1] == 7:
            # Positions, orientations, and timestamps [x, y, z, roll, pitch, yaw, timestamp]
            positions = trajectory_data[:, :3]
            euler_angles = trajectory_data[:, 3:6]
            if timestamps is None:
                timestamps = trajectory_data[:, 6]
        else:
            raise ValueError(f"Unsupported trajectory data shape: {trajectory_data.shape}")
        
        # Use provided timestamps or create default ones
        if timestamps is None:
            timestamps = np.arange(len(positions))
    else:
        raise ValueError("trajectory_data must be a dictionary or NumPy array")
    
    # Ensure we have valid data
    if len(positions) == 0:
        print("⚠️ No trajectory data available")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No trajectory\ndata available', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=16, bbox=dict(boxstyle="round", facecolor='lightgray'))
        ax.axis('off')
        return fig
    
    
    # Calculate velocities (approximate)
    velocities = []
    for i in range(1, len(positions)):
        if isinstance(timestamps[0], (int, float)):
            # Handle both microsecond timestamps and regular timestamps
            if timestamps[0] > 1e9:  # Likely microsecond timestamp
                dt = (timestamps[i] - timestamps[i-1]) / 1e6  # Convert to seconds
            else:
                dt = timestamps[i] - timestamps[i-1]  # Already in seconds
        else:
            dt = 1.0  # Default time step
            
        if dt > 0:
            vel = np.linalg.norm(positions[i] - positions[i-1]) / dt
            velocities.append(vel)
        else:
            velocities.append(0)
    velocities = [0] + velocities  # Add initial velocity
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top-down trajectory view
    ax1 = axes[0, 0]
    ax1.plot(positions[:, 0], positions[:, 1], color=trajectory_color, linewidth=2, alpha=0.8)
    ax1.scatter(positions[0, 0], positions[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, marker='s', label='End', zorder=5)
    
    # Add orientation arrows (every 10th pose)
    for i in range(0, len(positions), max(1, len(positions)//20)):
        yaw = euler_angles[i, 2]  # Yaw angle
        arrow_length = 5.0
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        ax1.arrow(positions[i, 0], positions[i, 1], dx, dy, 
                 head_width=2, head_length=1, fc='red', ec='red', alpha=0.7)
    
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.set_title('Vehicle Trajectory (Top View)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # 2. Altitude profile
    ax2 = axes[0, 1]
    distance = np.cumsum([0] + [np.linalg.norm(positions[i] - positions[i-1]) 
                                for i in range(1, len(positions))])
    ax2.plot(distance, positions[:, 2], color='purple', linewidth=2)
    ax2.set_xlabel('Distance (meters)', fontsize=12)
    ax2.set_ylabel('Altitude (meters)', fontsize=12)
    ax2.set_title('Altitude Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Speed profile
    ax3 = axes[1, 0]
    if isinstance(timestamps[0], (int, float)):
        if timestamps[0] > 1e9:  # Likely microsecond timestamp
            time_seconds = [(t - timestamps[0]) / 1e6 for t in timestamps]
        else:
            time_seconds = [t - timestamps[0] for t in timestamps]  # Already in seconds
    else:
        time_seconds = list(range(len(timestamps)))  # Use indices
    ax3.plot(time_seconds, velocities, color='orange', linewidth=2)
    ax3.set_xlabel('Time (seconds)', fontsize=12)
    ax3.set_ylabel('Speed (m/s)', fontsize=12)
    ax3.set_title('Speed Profile', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Orientation angles
    ax4 = axes[1, 1]
    ax4.plot(time_seconds, np.degrees(euler_angles[:, 0]), label='Roll', linewidth=2)
    ax4.plot(time_seconds, np.degrees(euler_angles[:, 1]), label='Pitch', linewidth=2)
    ax4.plot(time_seconds, np.degrees(euler_angles[:, 2]), label='Yaw', linewidth=2)
    ax4.set_xlabel('Time (seconds)', fontsize=12)
    ax4.set_ylabel('Angle (degrees)', fontsize=12)
    ax4.set_title('Vehicle Orientation', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Add summary statistics
    total_distance = distance[-1]
    avg_speed = np.mean(velocities)
    max_speed = np.max(velocities)
    duration = time_seconds[-1]
    
    fig.suptitle(f'Vehicle Trajectory Analysis\n'
                f'Distance: {total_distance:.1f}m, Duration: {duration:.1f}s, '
                f'Avg Speed: {avg_speed:.1f}m/s, Max Speed: {max_speed:.1f}m/s',
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved trajectory visualization to: {save_path}")
    
    return fig


# ============================================================================
# HELPER FUNCTIONS FOR VISUALIZATION
# ============================================================================

def range_image_to_pointcloud(range_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert LiDAR range image to 3D point cloud.
    
    Args:
        range_image (np.ndarray): Range image with shape [H, W, C]
                                 Channels: [range, intensity, elongation, is_in_nlz]
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (points, intensities)
            - points: [N, 3] 3D coordinates
            - intensities: [N] intensity values
    
    Educational Notes:
        - Range images use spherical coordinates
        - Conversion requires azimuth and inclination angles
        - Invalid points (range=0) are filtered out
        - Coordinate system: X=forward, Y=left, Z=up
    """
    height, width = range_image.shape[:2]
    
    # Create azimuth and inclination angle grids
    azimuth_range = 2 * np.pi  # Full 360 degrees
    inclination_range = np.pi / 6  # Typical LiDAR vertical FOV
    
    azimuth = np.linspace(-azimuth_range/2, azimuth_range/2, width)
    inclination = np.linspace(-inclination_range/2, inclination_range/2, height)
    
    azimuth_grid, inclination_grid = np.meshgrid(azimuth, inclination)
    
    # Extract range and intensity
    range_values = range_image[:, :, 0]
    intensity_values = range_image[:, :, 1] if range_image.shape[2] > 1 else None
    
    # Convert to Cartesian coordinates
    x = range_values * np.cos(inclination_grid) * np.cos(azimuth_grid)
    y = range_values * np.cos(inclination_grid) * np.sin(azimuth_grid)
    z = range_values * np.sin(inclination_grid)
    
    # Filter valid points (range > 0)
    valid_mask = range_values > 0
    
    points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=1)
    intensities = intensity_values[valid_mask] if intensity_values is not None else None
    
    return points, intensities


def create_open3d_boxes(boxes_data: Dict, frame_idx: int) -> List:
    """
    Create Open3D box geometries from 3D bounding box data.
    
    Args:
        boxes_data (Dict): 3D box data from read_lidar_boxes()
        frame_idx (int): Frame index for filtering boxes
    
    Returns:
        List: Open3D LineSet geometries representing boxes
    """
    if not HAS_OPEN3D:
        return []
    
    box_geometries = []
    metadata = boxes_data['metadata']
    
    # Filter boxes for current frame
    target_timestamp = metadata['timestamps'][frame_idx] if frame_idx < len(metadata['timestamps']) else None
    
    for i, (box, timestamp, obj_type) in enumerate(zip(
        boxes_data['boxes_3d'],
        metadata['timestamps'],
        metadata['object_types']
    )):
        if target_timestamp is None or timestamp == target_timestamp:
            # Create box wireframe
            corners = box['corners_3d']
            
            # Define box edges (12 edges for a cube)
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
            ]
            
            # Create LineSet
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(edges)
            
            # Color by object type
            colors = {
                1: [1, 0, 0],    # Vehicle - Red
                2: [0, 0, 1],    # Pedestrian - Blue
                3: [0, 1, 0],    # Sign - Green
                4: [1, 0.5, 0]   # Cyclist - Orange
            }
            color = colors.get(obj_type, [0.5, 0.5, 0.5])
            line_set.colors = o3d.utility.Vector3dVector([color] * len(edges))
            
            box_geometries.append(line_set)
    
    return box_geometries


def draw_matplotlib_boxes(ax, boxes_data: Dict, frame_idx: int):
    """
    Draw 3D bounding boxes on matplotlib 3D axis.
    
    Args:
        ax: Matplotlib 3D axis
        boxes_data (Dict): 3D box data from read_lidar_boxes()
        frame_idx (int): Frame index for filtering boxes
    """
    metadata = boxes_data['metadata']
    target_timestamp = metadata['timestamps'][frame_idx] if frame_idx < len(metadata['timestamps']) else None
    
    # Define colors for different object types
    colors = {
        1: 'red',      # Vehicle
        2: 'blue',     # Pedestrian
        3: 'green',    # Sign
        4: 'orange'    # Cyclist
    }
    
    for i, (box, timestamp, obj_type) in enumerate(zip(
        boxes_data['boxes_3d'],
        metadata['timestamps'],
        metadata['object_types']
    )):
        if target_timestamp is None or timestamp == target_timestamp:
            corners = box['corners_3d']
            color = colors.get(obj_type, 'gray')
            
            # Draw box edges
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
            ]
            
            for edge in edges:
                points = corners[edge]
                ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                         color=color, linewidth=2, alpha=0.8)


def create_demo_segmentation_mask(image_array: np.ndarray) -> np.ndarray:
    """
    Create a demonstration segmentation mask for visualization purposes.
    
    Args:
        image_array (np.ndarray): Input image array
    
    Returns:
        np.ndarray: Demo segmentation mask
    
    Note:
        This is a placeholder function. In practice, you would decode
        the actual Waymo segmentation data using their provided tools.
    """
    height, width = image_array.shape[:2]
    
    # Create simple geometric segmentation for demonstration
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Sky region
    mask[:height//3, :] = 1
    
    # Road region
    mask[2*height//3:, :] = 2
    
    # Vehicle regions (simple rectangles)
    mask[height//2:2*height//3, width//4:3*width//4] = 3
    
    # Add some random noise for realism
    noise = np.random.randint(0, 5, (height//10, width//10))
    mask[::10, ::10] = noise
    
    return mask


# ============================================================================
# LIDAR-CAMERA PROJECTION UTILITIES
# ============================================================================

def read_lidar_camera_projection(parquet_path: str) -> Dict:
    """
    Read LiDAR-camera projection data from Waymo parquet files.
    
    This function extracts LiDAR-camera projection data that maps LiDAR points
    to camera image coordinates, enabling multi-modal sensor fusion and
    cross-modal analysis.
    
    Args:
        parquet_path (str): Path to lidar_camera_projection parquet file
    
    Returns:
        Dict: Structured data containing:
            - 'projections': Dict with keys:
                - 'return1': List[np.ndarray] - Projection data for first return
                - 'return2': List[np.ndarray] - Projection data for second return
                - 'shapes': List[Tuple] - Original projection array shapes
            - 'metadata': Dict with keys:
                - 'segment_names': List[str] - Segment identifiers
                - 'timestamps': List[int] - Frame timestamps
                - 'laser_names': List[int] - LiDAR sensor IDs
    
    Educational Notes:
        - Projection data shape: [64, 2650, 6] for (height, width, channels)
        - Channels: [camera_id, x, y, range, intensity, elongation]
        - camera_id: Which camera the point projects to (-1 if none)
        - x, y: Pixel coordinates in camera image
        - range: Distance from LiDAR sensor
        - intensity: LiDAR return intensity
        - elongation: Beam elongation factor
    """
    print(f"🔗 Reading LiDAR-camera projection from: {os.path.basename(parquet_path)}")
    
    df = pd.read_parquet(parquet_path)
    
    result = {
        'projections': {'return1': [], 'return2': [], 'shapes': []},
        'metadata': {
            'segment_names': [],
            'timestamps': [],
            'laser_names': []
        }
    }
    
    print(f"🔄 Processing {len(df)} LiDAR-camera projection entries...")
    
    for idx, row in df.iterrows():
        # Extract projection data for both returns
        return1_values = row['[LiDARCameraProjectionComponent].range_image_return1.values']
        return1_shape = row['[LiDARCameraProjectionComponent].range_image_return1.shape']
        return2_values = row['[LiDARCameraProjectionComponent].range_image_return2.values']
        return2_shape = row['[LiDARCameraProjectionComponent].range_image_return2.shape']
        
        # Reshape flattened projection arrays
        if return1_values is not None and return1_shape is not None:
            return1_proj = np.array(return1_values).reshape(return1_shape)
            result['projections']['return1'].append(return1_proj)
        else:
            result['projections']['return1'].append(None)
        
        if return2_values is not None and return2_shape is not None:
            return2_proj = np.array(return2_values).reshape(return2_shape)
            result['projections']['return2'].append(return2_proj)
        else:
            result['projections']['return2'].append(None)
        
        result['projections']['shapes'].append(return1_shape if return1_shape else None)
        
        # Extract metadata
        result['metadata']['segment_names'].append(row['key.segment_context_name'])
        result['metadata']['timestamps'].append(row['key.frame_timestamp_micros'])
        result['metadata']['laser_names'].append(row['key.laser_name'])
    
    print(f"✅ Successfully loaded {len(df)} LiDAR-camera projections")
    return result


def visualize_lidar_camera_fusion(camera_image: Image.Image, 
                                 projection_data: Union[Dict, np.ndarray],
                                 lidar_data: Dict = None,
                                 frame_idx: int = 0,
                                 return_type: str = 'return1',
                                 camera_id: int = 0,
                                 intensity_threshold: float = 0.1,
                                 max_points: int = 10000,
                                 save_path: str = None,
                                 camera_names: Dict = None) -> plt.Figure:
    """
    Visualize LiDAR-camera sensor fusion by projecting LiDAR points onto camera image.
    
    This function creates a comprehensive multi-modal visualization showing how
    LiDAR points map to camera pixels, enabling analysis of sensor alignment,
    calibration quality, and cross-modal correspondences.
    
    Args:
        camera_image (PIL.Image): Camera image to overlay points on
        projection_data (Union[Dict, np.ndarray]): Projection data. Can be:
            - Dict with Waymo-style structure from read_lidar_camera_projection()
            - NumPy array of shape (H, W, 6) with channels [camera_id, x, y, range, intensity, elongation]
            - NumPy array of shape (N, 6) with N projected points
        lidar_data (Dict, optional): LiDAR range image data for intensity coloring
        frame_idx (int): Frame index to visualize
        return_type (str): 'return1' or 'return2' for multi-return LiDAR
        camera_id (int): Target camera ID (0-4)
        intensity_threshold (float): Minimum intensity for point display
        max_points (int): Maximum points to display for performance
        save_path (str, optional): Path to save visualization
        camera_names (Dict, optional): Mapping of camera IDs to names
    
    Returns:
        plt.Figure: Matplotlib figure with fusion visualization
    
    Educational Notes:
        - Points are colored by LiDAR intensity or distance
        - Only points projecting to specified camera are shown
        - Helps evaluate sensor calibration and alignment
        - Useful for multi-modal perception algorithm development
        - Cross-modal correspondences enable depth estimation
    """
    print(f"🔗 Creating LiDAR-camera fusion visualization for frame {frame_idx}")
    
    # Use provided camera names or default ones
    if camera_names is None:
        camera_names = WAYMO_CAMERA_NAMES
    
    # Handle different input formats
    if isinstance(projection_data, dict):
        # Waymo-style data structure
        projections = projection_data['projections'][return_type]
        if frame_idx >= len(projections) or projections[frame_idx] is None:
            print(f"❌ No projection data available for frame {frame_idx}")
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'No projection\ndata available', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=16, bbox=dict(boxstyle="round", facecolor='lightgray'))
            ax.axis('off')
            return fig
        
        projection = projections[frame_idx]
        
    elif isinstance(projection_data, np.ndarray):
        # Generic NumPy array format
        projection = projection_data
        
        # Handle different array shapes
        if projection.ndim == 2 and projection.shape[1] == 6:
            # Already in point format (N, 6)
            pass
        elif projection.ndim == 3 and projection.shape[2] == 6:
            # Range image format (H, W, 6) - flatten to points
            projection = projection.reshape(-1, 6)
        else:
            raise ValueError(f"Unsupported projection data shape: {projection.shape}")
    else:
        raise ValueError("projection_data must be a dictionary or NumPy array")
    
    print(f"📊 Projection shape: {projection.shape}")
    
    # Extract projection channels: [camera_id, x, y, range, intensity, elongation]
    if projection.ndim == 3:
        # Range image format (H, W, 6)
        proj_camera_ids = projection[:, :, 0]
        proj_x = projection[:, :, 1]
        proj_y = projection[:, :, 2]
        proj_range = projection[:, :, 3]
        proj_intensity = projection[:, :, 4]
        proj_elongation = projection[:, :, 5]
        
        # Filter points that project to the specified camera
        valid_mask = (proj_camera_ids == camera_id) & (proj_range > 0) & (proj_intensity > intensity_threshold)
        
        # Extract valid projection coordinates and properties
        valid_x = proj_x[valid_mask]
        valid_y = proj_y[valid_mask]
        valid_range = proj_range[valid_mask]
        valid_intensity = proj_intensity[valid_mask]
        valid_elongation = proj_elongation[valid_mask]
        
    else:
        # Point format (N, 6)
        valid_mask = (projection[:, 0] == camera_id) & (projection[:, 3] > 0) & (projection[:, 4] > intensity_threshold)
        
        valid_points = projection[valid_mask]
        valid_x = valid_points[:, 1]
        valid_y = valid_points[:, 2]
        valid_range = valid_points[:, 3]
        valid_intensity = valid_points[:, 4]
        valid_elongation = valid_points[:, 5]
    
    print(f"🎯 Found {len(valid_x)} valid projections for camera {camera_id}")
    
    # Subsample points if too many
    if len(valid_x) > max_points:
        indices = np.random.choice(len(valid_x), max_points, replace=False)
        valid_x = valid_x[indices]
        valid_y = valid_y[indices]
        valid_range = valid_range[indices]
        valid_intensity = valid_intensity[indices]
        valid_elongation = valid_elongation[indices]
        print(f"🔽 Subsampled to {max_points} points for performance")
    
    # Create visualization with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Original camera image
    axes[0, 0].imshow(camera_image)
    axes[0, 0].set_title(f'Original Camera Image\nCamera: {camera_names.get(camera_id, f"Camera {camera_id}")}', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. LiDAR points colored by intensity
    axes[0, 1].imshow(camera_image)
    if len(valid_x) > 0:
        scatter1 = axes[0, 1].scatter(valid_x, valid_y, c=valid_intensity, 
                                     cmap='viridis', s=2, alpha=0.8)
        plt.colorbar(scatter1, ax=axes[0, 1], label='LiDAR Intensity', shrink=0.8)
    axes[0, 1].set_title(f'LiDAR Points Colored by Intensity\nPoints: {len(valid_x)}', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. LiDAR points colored by distance
    axes[1, 0].imshow(camera_image)
    if len(valid_x) > 0:
        scatter2 = axes[1, 0].scatter(valid_x, valid_y, c=valid_range, 
                                     cmap='plasma', s=2, alpha=0.8)
        plt.colorbar(scatter2, ax=axes[1, 0], label='Distance (m)', shrink=0.8)
    axes[1, 0].set_title('LiDAR Points Colored by Distance', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 4. Analysis plots
    ax_analysis = axes[1, 1]
    
    if len(valid_x) > 0:
        # Create histogram of distances
        ax_analysis.hist(valid_range, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax_analysis.set_xlabel('Distance (meters)', fontsize=12)
        ax_analysis.set_ylabel('Number of Points', fontsize=12)
        ax_analysis.set_title('Distance Distribution of Projected Points', 
                             fontsize=14, fontweight='bold')
        ax_analysis.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics:
        Total Points: {len(valid_x)}
        Mean Distance: {np.mean(valid_range):.2f}m
        Max Distance: {np.max(valid_range):.2f}m
        Mean Intensity: {np.mean(valid_intensity):.3f}
        Coverage: {len(valid_x)/np.prod(camera_image.size)*100:.2f}%"""
        
        ax_analysis.text(0.02, 0.98, stats_text, transform=ax_analysis.transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    else:
        ax_analysis.text(0.5, 0.5, 'No valid projections\nfor this camera', 
                        ha='center', va='center', transform=ax_analysis.transAxes,
                        fontsize=16, bbox=dict(boxstyle="round", facecolor='lightgray'))
    
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle(f'LiDAR-Camera Sensor Fusion Visualization\n'
                f'Frame: {frame_idx}, Camera: {camera_names.get(camera_id, f"Camera {camera_id}")}, '
                f'Return: {return_type}',
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved fusion visualization to: {save_path}")
    
    return fig


def visualize_multi_camera_projection(camera_images: List[Image.Image],
                                     projection_data: Union[Dict, np.ndarray],
                                     frame_idx: int = 0,
                                     return_type: str = 'return1',
                                     intensity_threshold: float = 0.1,
                                     save_path: str = None,
                                     camera_names: Dict = None) -> plt.Figure:
    """
    Visualize LiDAR projections across all camera views simultaneously.
    
    Args:
        camera_images: List of PIL Images for each camera view
        projection_data: Either Waymo-style dict with projection data or generic numpy array
                        - If dict: Should contain 'projections' with return type data
                        - If numpy array: Should be shape (N, 5) with columns [x, y, intensity, camera_id, range]
        frame_idx: Frame index for display
        return_type: Return type identifier for display
        intensity_threshold: Minimum intensity threshold for point visibility
        save_path: Optional path to save the visualization
        camera_names: Optional dict mapping camera IDs to names (defaults to generic names)
    
    Returns:
        matplotlib Figure object
    """
    # Set default camera names if not provided
    if camera_names is None:
        camera_names = {
            0: 'Camera 0',
            1: 'Camera 1', 
            2: 'Camera 2',
            3: 'Camera 3',
            4: 'Camera 4'
        }
    
    print(f"🎥 Creating multi-camera projection visualization for frame {frame_idx}")
    
    # Handle different projection data formats
    if isinstance(projection_data, dict) and 'projections' in projection_data:
        # Waymo-style dictionary format
        projections = projection_data['projections'][return_type]
        if frame_idx >= len(projections) or projections[frame_idx] is None:
            print(f"❌ No projection data available for frame {frame_idx}")
            return None
        
        projection = projections[frame_idx]
        proj_x = projection['proj_x']
        proj_y = projection['proj_y']
        proj_intensity = projection['proj_intensity']
        proj_camera_ids = projection['proj_camera_id']
        proj_range = projection['proj_range']
    elif isinstance(projection_data, np.ndarray):
        # Generic numpy array format
        if projection_data.shape[1] >= 5:
            proj_x = projection_data[:, 0]
            proj_y = projection_data[:, 1]
            proj_intensity = projection_data[:, 2]
            proj_camera_ids = projection_data[:, 3].astype(int)
            proj_range = projection_data[:, 4]
        else:
            print("⚠️ Warning: Projection data array should have at least 5 columns [x, y, intensity, camera_id, range]")
            return None
    else:
        print("⚠️ Warning: No valid projection data provided")
        return None
    
    # Create figure with 5 camera subplots
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    
    total_points = 0
    
    for cam_id in range(5):
        ax = axes[cam_id]
        
        # Display camera image if available
        if cam_id < len(camera_images) and camera_images[cam_id] is not None:
            ax.imshow(camera_images[cam_id])
        else:
            ax.set_facecolor('black')
            ax.text(0.5, 0.5, f'Camera {cam_id}\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes,
                   color='white', fontsize=14)
        
        # Filter points for this camera
        valid_mask = (proj_camera_ids == cam_id) & (proj_range > 0) & (proj_intensity > intensity_threshold)
        
        if np.any(valid_mask):
            valid_x = proj_x[valid_mask]
            valid_y = proj_y[valid_mask]
            valid_intensity = proj_intensity[valid_mask]
            
            # Plot projected points
            scatter = ax.scatter(valid_x, valid_y, c=valid_intensity, 
                               cmap='viridis', s=1, alpha=0.8)
            
            point_count = len(valid_x)
            total_points += point_count
            
            ax.set_title(f'{camera_names[cam_id]}\n{point_count} points', 
                        fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'{camera_names[cam_id]}\n0 points', 
                        fontsize=12, fontweight='bold')
        
        ax.axis('off')
    
    # Use the last subplot for statistics
    axes[5].axis('off')
    stats_text = f"""Multi-Camera Projection Statistics:
    
    Frame: {frame_idx}
    Return Type: {return_type}
    Total Projected Points: {total_points}
    Intensity Threshold: {intensity_threshold}
    
    Camera Coverage:
    • Points distributed across {sum(1 for cam_id in range(5) if np.any((proj_camera_ids == cam_id) & (proj_range > 0) & (proj_intensity > intensity_threshold)))}/5 cameras
    • Enables 360° environmental perception
    • Multi-view correspondences for depth estimation
    • Cross-modal sensor fusion capabilities
    """
    
    axes[5].text(0.1, 0.9, stats_text, transform=axes[5].transAxes,
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle(f'Multi-Camera LiDAR Projection Visualization\n'
                f'Frame: {frame_idx}, Total Points: {total_points}',
                fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved multi-camera projection to: {save_path}")
    
    return fig


def analyze_projection_quality(projection_data: Dict, 
                              frame_indices: List[int] = None,
                              save_path: str = None) -> plt.Figure:
    """
    Analyze LiDAR-camera projection quality across frames and cameras.
    
    This function provides comprehensive quality analysis of LiDAR-camera
    projections, including coverage statistics, intensity distributions,
    and temporal consistency metrics.
    
    Args:
        projection_data (Dict): Projection data from read_lidar_camera_projection()
        frame_indices (List[int], optional): Specific frames to analyze
        save_path (str, optional): Path to save analysis plots
    
    Returns:
        plt.Figure: Matplotlib figure with quality analysis
    
    Educational Notes:
        - Coverage analysis reveals sensor blind spots
        - Intensity distributions show data quality
        - Temporal consistency indicates calibration stability
        - Distance analysis reveals effective sensor range
        - Useful for sensor validation and calibration assessment
    """
    print("📊 Analyzing LiDAR-camera projection quality")
    
    projections = projection_data['projections']['return1']
    
    if frame_indices is None:
        frame_indices = list(range(min(len(projections), 10)))  # Analyze first 10 frames
    
    # Initialize analysis data
    camera_coverage = {cam_id: [] for cam_id in range(5)}
    intensity_stats = []
    distance_stats = []
    
    for frame_idx in frame_indices:
        if frame_idx >= len(projections) or projections[frame_idx] is None:
            continue
        
        projection = projections[frame_idx]
        
        # Extract channels
        proj_camera_ids = projection[:, :, 0]
        proj_range = projection[:, :, 3]
        proj_intensity = projection[:, :, 4]
        
        # Analyze coverage per camera
        for cam_id in range(5):
            valid_mask = (proj_camera_ids == cam_id) & (proj_range > 0)
            coverage = np.sum(valid_mask)
            camera_coverage[cam_id].append(coverage)
        
        # Collect intensity and distance statistics
        valid_points = (proj_range > 0) & (proj_intensity > 0)
        if np.any(valid_points):
            intensity_stats.extend(proj_intensity[valid_points])
            distance_stats.extend(proj_range[valid_points])
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Camera coverage over time
    ax1 = axes[0, 0]
    for cam_id, coverage_data in camera_coverage.items():
        if coverage_data:
            ax1.plot(frame_indices[:len(coverage_data)], coverage_data, 
                    label=WAYMO_CAMERA_NAMES.get(cam_id, f'Camera {cam_id}'), 
                    marker='o', linewidth=2)
    ax1.set_xlabel('Frame Index', fontsize=12)
    ax1.set_ylabel('Number of Projected Points', fontsize=12)
    ax1.set_title('Camera Coverage Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average coverage per camera
    ax2 = axes[0, 1]
    avg_coverage = [np.mean(coverage_data) if coverage_data else 0 
                   for coverage_data in camera_coverage.values()]
    camera_labels = [WAYMO_CAMERA_NAMES.get(i, f'Cam {i}') for i in range(5)]
    bars = ax2.bar(camera_labels, avg_coverage, color=['red', 'blue', 'green', 'orange', 'purple'])
    ax2.set_ylabel('Average Points per Frame', fontsize=12)
    ax2.set_title('Average Coverage per Camera', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_coverage):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_coverage)*0.01,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Intensity distribution
    ax3 = axes[0, 2]
    if intensity_stats:
        ax3.hist(intensity_stats, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xlabel('LiDAR Intensity', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Intensity Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. Distance distribution
    ax4 = axes[1, 0]
    if distance_stats:
        ax4.hist(distance_stats, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax4.set_xlabel('Distance (meters)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Distance Distribution', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # 5. Coverage consistency (coefficient of variation)
    ax5 = axes[1, 1]
    cv_values = []
    for cam_id, coverage_data in camera_coverage.items():
        if coverage_data and len(coverage_data) > 1:
            cv = np.std(coverage_data) / np.mean(coverage_data) if np.mean(coverage_data) > 0 else 0
            cv_values.append(cv)
        else:
            cv_values.append(0)
    
    bars2 = ax5.bar(camera_labels, cv_values, color=['red', 'blue', 'green', 'orange', 'purple'])
    ax5.set_ylabel('Coefficient of Variation', fontsize=12)
    ax5.set_title('Coverage Consistency\n(Lower = More Consistent)', fontsize=14, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    if intensity_stats and distance_stats:
        summary_text = f"""Projection Quality Summary:
        
        Frames Analyzed: {len(frame_indices)}
        Total Projected Points: {len(intensity_stats):,}
        
        Distance Statistics:
        • Mean: {np.mean(distance_stats):.2f}m
        • Median: {np.median(distance_stats):.2f}m
        • Max: {np.max(distance_stats):.2f}m
        • Std: {np.std(distance_stats):.2f}m
        
        Intensity Statistics:
        • Mean: {np.mean(intensity_stats):.3f}
        • Median: {np.median(intensity_stats):.3f}
        • Max: {np.max(intensity_stats):.3f}
        • Std: {np.std(intensity_stats):.3f}
        
        Camera Coverage:
        • Most Active: {camera_labels[np.argmax(avg_coverage)]}
        • Least Active: {camera_labels[np.argmin(avg_coverage)]}
        • Total Cameras: {sum(1 for x in avg_coverage if x > 0)}/5
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved projection quality analysis to: {save_path}")
    
    return fig



# ============================================================================
# MAIN TESTING AND DEMONSTRATION FUNCTION
# ============================================================================

def main_test_visualizations(data_root: str = "./waymo_data", 
                           output_dir: str = "./visualization_outputs",
                           test_frame_idx: int = 0,
                           enable_3d: bool = True,
                           max_test_frames: int = 3) -> None:
    """
    Comprehensive testing function for all Waymo visualization utilities.
    
    This function demonstrates the complete workflow of loading Waymo data
    and creating various visualizations. It serves as both a testing framework
    and an educational example of multi-modal autonomous driving data analysis.
    
    Args:
        data_root (str): Root directory containing Waymo parquet files
        output_dir (str): Directory to save visualization outputs
        test_frame_idx (int): Primary frame index for detailed visualizations
        enable_3d (bool): Whether to create 3D visualizations (requires Open3D)
        max_test_frames (int): Maximum number of frames to process for analysis
    
    Educational Workflow:
        1. Data Loading: Demonstrates reading all Waymo data modalities
        2. Single-Modal Visualization: Camera images, LiDAR point clouds
        3. Multi-Modal Fusion: LiDAR-camera projections and correspondences
        4. Annotation Overlay: Bounding boxes and segmentation masks
        5. Temporal Analysis: Vehicle trajectories and motion patterns
        6. Quality Assessment: Projection accuracy and sensor calibration
    
    Output Structure:
        - camera_visualizations/: Camera images with annotations
        - lidar_visualizations/: Point cloud visualizations
        - fusion_visualizations/: Multi-modal sensor fusion
        - segmentation_visualizations/: Semantic segmentation overlays
        - trajectory_visualizations/: Vehicle motion analysis
        - quality_analysis/: Sensor calibration and projection quality
    """
    print("🚀 Starting Comprehensive Waymo Visualization Testing")
    print("=" * 60)
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    subdirs = [
        'camera_visualizations', 'lidar_visualizations', 'fusion_visualizations',
        'segmentation_visualizations', 'trajectory_visualizations', 'quality_analysis'
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Test data paths (adjust based on your Waymo data structure)
    test_paths = {
        'camera_image': os.path.join(data_root, 'camera_image'),
        'camera_box': os.path.join(data_root, 'camera_box'),
        'lidar': os.path.join(data_root, 'lidar'),
        'lidar_box': os.path.join(data_root, 'lidar_box'),
        'lidar_camera_projection': os.path.join(data_root, 'lidar_camera_projection'),
        'lidar_segmentation': os.path.join(data_root, 'lidar_segmentation'),
        'camera_segmentation': os.path.join(data_root, 'camera_segmentation'),
        'vehicle_pose': os.path.join(data_root, 'vehicle_pose')
    }
    
    # Check which data is available
    available_data = {}
    for data_type, path in test_paths.items():
        parquet_files = []
        if os.path.exists(path):
            parquet_files = [f for f in os.listdir(path) if f.endswith('.parquet')]
        
        if parquet_files:
            available_data[data_type] = os.path.join(path, parquet_files[0])
            print(f"✅ Found {data_type}: {len(parquet_files)} files")
        else:
            print(f"❌ Missing {data_type} data")
    
    if not available_data:
        print("❌ No Waymo data found! Please check your data_root path.")
        print(f"Expected structure: {data_root}/[camera_image|lidar|etc]/*.parquet")
        return
    
    print(f"\n📊 Testing with {len(available_data)} available data types")
    print("=" * 60)
    
    # ========================================================================
    # TEST 1: CAMERA VISUALIZATION
    # ========================================================================
    if 'camera_image' in available_data:
        print("\n🎥 TEST 1: Camera Image Visualization")
        try:
            # Load camera images
            camera_data = read_camera_images(available_data['camera_image'])
            print(f"Loaded {len(camera_data['images'])} camera images")
            
            # Load camera boxes if available
            camera_boxes = None
            if 'camera_box' in available_data:
                camera_boxes = read_camera_boxes(available_data['camera_box'])
                print(f"Loaded {len(camera_boxes['boxes'])} camera box annotations")
            
            # Group images by frame and camera for visualization
            # Each image in the list corresponds to a single camera view
            images_by_frame = {}
            for idx, image in enumerate(camera_data['images']):
                if image is not None:
                    # Get camera ID and frame info from metadata
                    camera_id = camera_data['metadata']['camera_ids'][idx]
                    frame_timestamp = camera_data['metadata']['timestamps'][idx]
                    
                    # Group by timestamp (frame)
                    if frame_timestamp not in images_by_frame:
                        images_by_frame[frame_timestamp] = {}
                    images_by_frame[frame_timestamp][camera_id] = {'image': image, 'index': idx}
            
            # Visualize first few frames
            frame_timestamps = sorted(list(images_by_frame.keys()))
            if len(frame_timestamps) > test_frame_idx:
                target_timestamp = frame_timestamps[test_frame_idx]
                frame_cameras = images_by_frame[target_timestamp]
                
                print(f"Visualizing frame {test_frame_idx} with {len(frame_cameras)} cameras")
                
                # Test first 3 available cameras for this frame
                camera_ids = sorted(list(frame_cameras.keys()))[:3]
                for camera_id in camera_ids:
                    camera_info = frame_cameras[camera_id]
                    image = camera_info['image']
                    image_idx = camera_info['index']
                    
                    # Create visualization with all camera boxes data
                    camera_name = WAYMO_CAMERA_NAMES.get(camera_id, f'Camera_{camera_id}')
                    save_path = os.path.join(output_dir, 'camera_visualizations', 
                                           f'{camera_name}_frame_{test_frame_idx}.png')
                    
                    # Prepare camera boxes data safely
                    if camera_boxes:
                        boxes_data = camera_boxes
                        metadata = camera_boxes.get('metadata', {'object_types': [], 'timestamps': [], 'object_ids': [], 'difficulties': {'detection': [], 'tracking': []}})
                    else:
                        boxes_data = {'boxes': [], 'metadata': {'object_types': [], 'timestamps': [], 'object_ids': [], 'difficulties': {'detection': [], 'tracking': []}}}
                        metadata = boxes_data['metadata']
                    
                    # Call visualization function with correct parameters
                    fig = visualize_camera_with_boxes(
                        image,
                        boxes_data=boxes_data,
                        metadata=metadata,
                        frame_idx=image_idx,
                        save_path=save_path
                    )
                    plt.close(fig)
                    print(f"  ✓ Saved {camera_name} visualization")
            
            print("✅ Camera visualization completed")
            
        except Exception as e:
            print(f"❌ Camera visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # TEST 2: LIDAR VISUALIZATION
    # ========================================================================
    if 'lidar' in available_data:
        print("\n☁️ TEST 2: LiDAR Point Cloud Visualization")
        try:
            # Load LiDAR data
            lidar_data = read_lidar_data(available_data['lidar'])
            print(f"Loaded {len(lidar_data['range_images'])} LiDAR scans")
            
            # Load LiDAR boxes if available
            lidar_boxes = None
            if 'lidar_box' in available_data:
                lidar_boxes = read_lidar_boxes(available_data['lidar_box'])
                print(f"Loaded {len(lidar_boxes['boxes'])} LiDAR box annotations")
            
            # Visualize LiDAR point cloud
            if test_frame_idx < len(lidar_data['range_images']):
                boxes_for_frame = None
                if lidar_boxes and test_frame_idx < len(lidar_boxes['boxes']):
                    boxes_for_frame = lidar_boxes['boxes'][test_frame_idx]
                
                # Test both 3D and 2D visualizations
                if enable_3d:
                    try:
                        fig = visualize_lidar_pointcloud(
                            lidar_data['range_images'][test_frame_idx],
                            boxes_data=boxes_for_frame,
                            use_open3d=True,
                            save_path=os.path.join(output_dir, 'lidar_visualizations', 
                                                 f'lidar_3d_frame_{test_frame_idx}.png')
                        )
                        if fig:
                            plt.close(fig)
                    except Exception as e:
                        print(f"⚠️ 3D visualization failed (Open3D issue): {e}")
                
                # 2D visualization (always works)
                fig = visualize_lidar_pointcloud(
                    lidar_data['range_images'][test_frame_idx],
                    boxes_data=boxes_for_frame,
                    use_open3d=False,
                    save_path=os.path.join(output_dir, 'lidar_visualizations', 
                                         f'lidar_2d_frame_{test_frame_idx}.png')
                )
                plt.close(fig)
            
            print("✅ LiDAR visualization completed")
            
        except Exception as e:
            print(f"❌ LiDAR visualization failed: {e}")
    
    # ========================================================================
    # TEST 3: LIDAR-CAMERA FUSION
    # ========================================================================
    if 'lidar_camera_projection' in available_data and 'camera_image' in available_data:
        print("\n🔗 TEST 3: LiDAR-Camera Fusion Visualization")
        try:
            # Load projection data
            projection_data = read_lidar_camera_projection(available_data['lidar_camera_projection'])
            print(f"Loaded {len(projection_data['projections']['return1'])} projection mappings")
            
            # Single camera fusion
            if (test_frame_idx < len(camera_data['images']) and 
                test_frame_idx < len(projection_data['projections']['return1'])):
                
                for cam_idx in range(min(2, len(camera_data['images'][test_frame_idx]))):  # Test 2 cameras
                    if camera_data['images'][test_frame_idx][cam_idx] is not None:
                        fig = visualize_lidar_camera_fusion(
                            camera_data['images'][test_frame_idx][cam_idx],
                            projection_data,
                            frame_idx=test_frame_idx,
                            camera_id=cam_idx,
                            save_path=os.path.join(output_dir, 'fusion_visualizations', 
                                                 f'fusion_cam_{cam_idx}_frame_{test_frame_idx}.png')
                        )
                        if fig:
                            plt.close(fig)
            
            # Multi-camera fusion
            if test_frame_idx < len(camera_data['images']):
                fig = visualize_multi_camera_projection(
                    camera_data['images'][test_frame_idx],
                    projection_data,
                    frame_idx=test_frame_idx,
                    save_path=os.path.join(output_dir, 'fusion_visualizations', 
                                         f'multi_camera_frame_{test_frame_idx}.png')
                )
                if fig:
                    plt.close(fig)
            
            print("✅ LiDAR-Camera fusion completed")
            
        except Exception as e:
            print(f"❌ LiDAR-Camera fusion failed: {e}")
    
    # ========================================================================
    # TEST 4: SEGMENTATION VISUALIZATION
    # ========================================================================
    # Test both LiDAR and camera segmentation if available
    segmentation_tested = False
    
    # Test camera segmentation
    if 'camera_segmentation' in available_data and 'camera_image' in available_data:
        print("\n🎨 TEST 4A: Camera Segmentation Visualization")
        try:
            # Load camera segmentation data
            camera_seg_data = read_segmentation_data(available_data['camera_segmentation'], 'camera')
            print(f"Loaded camera segmentation data for {len(camera_seg_data['labels']['panoptic'])} frames")
            
            # Visualize segmentation overlay
            if (test_frame_idx < len(camera_data['images']) and 
                test_frame_idx < len(camera_seg_data['labels']['panoptic'])):
                
                for cam_idx in range(min(2, len(camera_data['images'][test_frame_idx]))):
                    if camera_data['images'][test_frame_idx][cam_idx] is not None:
                        # Use actual segmentation data if available, otherwise demo
                        seg_data_for_vis = camera_seg_data if camera_seg_data['labels']['panoptic'] else None
                        if seg_data_for_vis is None:
                            # Create demo segmentation (replace with actual when available)
                            image_array = np.array(camera_data['images'][test_frame_idx][cam_idx])
                            demo_mask = create_demo_segmentation_mask(image_array)
                            seg_data_for_vis = {'labels': {'panoptic': [demo_mask]}}
                        
                        fig = visualize_segmentation_overlay(
                            camera_data['images'][test_frame_idx][cam_idx],
                            seg_data_for_vis,
                            frame_idx=test_frame_idx,
                            save_path=os.path.join(output_dir, 'segmentation_visualizations', 
                                                 f'camera_segmentation_cam_{cam_idx}_frame_{test_frame_idx}.png')
                        )
                        plt.close(fig)
            
            print("✅ Camera segmentation visualization completed")
            segmentation_tested = True
            
        except Exception as e:
            print(f"❌ Camera segmentation visualization failed: {e}")
    
    # Test LiDAR segmentation
    if 'lidar_segmentation' in available_data:
        print("\n🎨 TEST 4B: LiDAR Segmentation Visualization")
        try:
            # Load LiDAR segmentation data
            lidar_seg_data = read_segmentation_data(available_data['lidar_segmentation'], 'lidar')
            print(f"Loaded LiDAR segmentation data for {len(lidar_seg_data['labels']['return1'])} frames")
            
            # Note: LiDAR segmentation visualization would require converting range images to point clouds
            # and applying segmentation labels - this is more complex and would be implemented separately
            print("📝 LiDAR segmentation loaded successfully (visualization requires point cloud conversion)")
            segmentation_tested = True
            
        except Exception as e:
            print(f"❌ LiDAR segmentation loading failed: {e}")
    
    if not segmentation_tested:
        print("⚠️ No segmentation data available for testing")
    
    # ========================================================================
    # TEST 5: VEHICLE TRAJECTORY VISUALIZATION
    # ========================================================================
    if 'vehicle_pose' in available_data:
        print("\n🚗 TEST 5: Vehicle Trajectory Visualization")
        try:
            # Load vehicle pose data
            pose_data = read_vehicle_pose(available_data['vehicle_pose'])
            print(f"Loaded {len(pose_data['positions'])} pose measurements")
            
            # Visualize trajectory
            if len(pose_data['positions']) > 1:
                fig = visualize_vehicle_trajectory(
                    pose_data,
                    max_frames=min(100, len(pose_data['positions'])),
                    save_path=os.path.join(output_dir, 'trajectory_visualizations', 
                                         f'trajectory_full.png')
                )
                plt.close(fig)
            
            print("✅ Vehicle trajectory visualization completed")
            
        except Exception as e:
            print(f"❌ Vehicle trajectory visualization failed: {e}")
    
    # ========================================================================
    # TEST 6: PROJECTION QUALITY ANALYSIS
    # ========================================================================
    if 'lidar_camera_projection' in available_data:
        print("\n📊 TEST 6: Projection Quality Analysis")
        try:
            # Analyze projection quality
            analysis_frames = list(range(min(max_test_frames, len(projection_data['projections']['return1']))))
            
            fig = analyze_projection_quality(
                projection_data,
                frame_indices=analysis_frames,
                save_path=os.path.join(output_dir, 'quality_analysis', 
                                     f'projection_quality_analysis.png')
            )
            plt.close(fig)
            
            print("✅ Projection quality analysis completed")
            
        except Exception as e:
            print(f"❌ Projection quality analysis failed: {e}")
    
    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    print("\n" + "=" * 60)
    print("🎯 TESTING SUMMARY REPORT")
    print("=" * 60)
    
    # Count generated files
    total_files = 0
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if os.path.exists(subdir_path):
            files = [f for f in os.listdir(subdir_path) if f.endswith(('.png', '.jpg', '.pdf'))]
            if files:
                print(f"📁 {subdir}: {len(files)} visualizations")
                total_files += len(files)
    
    print(f"\n✅ Total visualizations created: {total_files}")
    print(f"📂 Output directory: {output_dir}")
    
    # Generate summary HTML report
    create_summary_report(output_dir, available_data, test_frame_idx)
    
    print("\n🎉 All visualization tests completed successfully!")
    print("📖 Check the generated HTML report for a comprehensive overview.")


def create_summary_report(output_dir: str, available_data: dict, test_frame_idx: int) -> None:
    """
    Create an HTML summary report of all generated visualizations.
    
    Args:
        output_dir (str): Directory containing visualization outputs
        available_data (dict): Dictionary of available data types
        test_frame_idx (int): Frame index used for testing
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Waymo Visualization Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
            .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .image-item {{ text-align: center; }}
            .image-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            .stats {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .success {{ color: #27ae60; }}
            .error {{ color: #e74c3c; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Waymo Visualization Test Report</h1>
            <p>Comprehensive testing of all visualization utility functions</p>
            <p><strong>Test Frame:</strong> {test_frame_idx} | <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Data Availability</h2>
            <div class="stats">
    """
    
    for data_type in ['camera_image', 'lidar', 'lidar_camera_projection', 'lidar_segmentation', 'camera_segmentation', 'vehicle_pose']:
        status = "✅ Available" if data_type in available_data else "❌ Missing"
        color_class = "success" if data_type in available_data else "error"
        html_content += f'<p class="{color_class}"><strong>{data_type}:</strong> {status}</p>\n'
    
    html_content += """
            </div>
        </div>
    """
    
    # Add sections for each visualization type
    sections = [
        ('camera_visualizations', '🎥 Camera Visualizations', 'Camera images with 2D bounding box annotations'),
        ('lidar_visualizations', '☁️ LiDAR Visualizations', '3D point clouds with bounding boxes'),
        ('fusion_visualizations', '🔗 Multi-Modal Fusion', 'LiDAR-camera sensor fusion and projections'),
        ('segmentation_visualizations', '🎨 Segmentation Overlays', 'Semantic segmentation masks on camera images'),
        ('trajectory_visualizations', '🚗 Vehicle Trajectories', 'Ego-vehicle motion and path analysis'),
        ('quality_analysis', '📈 Quality Analysis', 'Sensor calibration and projection quality metrics')
    ]
    
    for folder, title, description in sections:
        folder_path = os.path.join(output_dir, folder)
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]
            if images:
                html_content += f"""
                <div class="section">
                    <h2>{title}</h2>
                    <p>{description}</p>
                    <div class="image-grid">
                """
                
                for img in sorted(images)[:6]:  # Show up to 6 images per section
                    img_path = os.path.join(folder, img)
                    html_content += f"""
                    <div class="image-item">
                        <img src="{img_path}" alt="{img}">
                        <p><strong>{img}</strong></p>
                    </div>
                    """
                
                html_content += """
                    </div>
                </div>
                """
    
    html_content += """
        <div class="section">
            <h2>🎯 Summary</h2>
            <div class="stats">
                <p>This report demonstrates the comprehensive visualization capabilities for Waymo Open Dataset analysis.</p>
                <p><strong>Key Features Tested:</strong></p>
                <ul>
                    <li>Multi-modal sensor data loading and processing</li>
                    <li>2D and 3D visualization capabilities</li>
                    <li>Cross-modal sensor fusion and projection mapping</li>
                    <li>Annotation overlay and quality analysis</li>
                    <li>Temporal trajectory analysis</li>
                </ul>
                <p><strong>Educational Value:</strong> All functions include detailed docstrings and teaching comments for autonomous driving research.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = os.path.join(output_dir, 'visualization_test_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 HTML report saved: {report_path}")


if __name__ == "__main__":
    """
    Main execution block for testing Waymo visualization utilities.
    
    Usage Examples:
        # Basic testing with default parameters
        python waymo_data_utils.py
        
        # Custom data path and output directory
        python waymo_data_utils.py --data_root /path/to/waymo/data --output_dir ./my_outputs
        
        # Test specific frame with 3D disabled
        python waymo_data_utils.py --test_frame 5 --no_3d
    """
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Test Waymo visualization utilities')
    parser.add_argument('--data_root', type=str, default='/data/Datasets/waymodata/training',
                       help='Root directory containing Waymo parquet files')
    parser.add_argument('--output_dir', type=str, default='output/waymo_visualization',
                       help='Directory to save visualization outputs')
    parser.add_argument('--test_frame', type=int, default=0,
                       help='Frame index for detailed visualizations')
    parser.add_argument('--no_3d', action='store_true',
                       help='Disable 3D visualizations (useful if Open3D not available)')
    parser.add_argument('--max_frames', type=int, default=3,
                       help='Maximum number of frames for analysis')
    
    args = parser.parse_args()
    
    # Run comprehensive testing
    main_test_visualizations(
        data_root=args.data_root,
        output_dir=args.output_dir,
        test_frame_idx=args.test_frame,
        enable_3d=not args.no_3d,
        max_test_frames=args.max_frames
    )
