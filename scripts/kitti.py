#!/usr/bin/env python3
"""
KITTI Dataset Utilities and Visualization Tool

This module provides comprehensive utilities for working with KITTI format datasets,
including the original KITTI dataset and converted WaymoKITTI format datasets.

Features:
- Dataset download and extraction
- Dataset structure validation
- 2D camera visualization with bounding boxes
- 3D bounding box projection on images
- 3D LiDAR visualization with Mayavi
- Dataset diagnosis and issue detection

Based on nuscenes.py and waymokittiall.py
"""

import tarfile
import glob
import os
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from enum import IntEnum
from PIL import Image
import sys
import urllib.request
import zipfile
import shutil

# Add project root to path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

try:
    import mayavi.mlab as mlab
    MAYAVI_AVAILABLE = True
except ImportError:
    print("Warning: Mayavi not available. 3D LiDAR visualization will be disabled.")
    MAYAVI_AVAILABLE = False

try:
    from DeepDataMiningLearning.detection3d.CalibrationUtils import WaymoCalibration, KittiCalibration, rotx, roty, rotz
    CALIB_UTILS_AVAILABLE = True
except ImportError:
    print("Warning: CalibrationUtils not available. Some calibration functions may not work.")
    CALIB_UTILS_AVAILABLE = False

try:
    from DeepDataMiningLearning.mydetector3d.tools.visual_utils.mayavivisualize_utils import visualize_pts, draw_lidar, draw_gt_boxes3d, draw_scenes
    MAYAVI_UTILS_AVAILABLE = True
except ImportError:
    print("Warning: Mayavi visualization utils not available. Some 3D visualization features may not work.")
    MAYAVI_UTILS_AVAILABLE = False

# KITTI dataset structure definition
KITTI_STRUCTURE = {
    'training': ['image_2', 'image_3', 'velodyne', 'calib', 'label_2'],
    'testing': ['image_2', 'image_3', 'velodyne', 'calib'],
    'waymokitti': ['image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'velodyne', 'calib', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4', 'label_all']
}

REQUIRED_KITTI_DIRS = {
    'standard': ['image_2', 'velodyne', 'calib', 'label_2'],
    'waymokitti': ['image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'velodyne', 'calib', 'label_all']
}

# KITTI download URLs (official dataset)
KITTI_URLS = {
    'left_images': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip',
    'right_images': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_3.zip',
    'velodyne': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip',
    'calib': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip',
    'labels': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip'
}

class BoxVisibility(IntEnum):
    """ Enumerates the various level of box visibility in an image """
    ALL = 0  # Requires all corners are inside the image.
    ANY = 1  # Requires at least one corner visible in the image.
    NONE = 2  # Requires no corners to be inside, i.e. box can be fully outside the image.

class Object3d(object):
    """ 3D object label for KITTI format """

    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[2]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def estimate_difficulty(self):
        """ Function that estimate difficulty to detect the object as defined in kitti website"""
        # height of the bounding box
        bb_height = np.abs(self.ymax - self.ymin)

        if bb_height >= 40 and self.occlusion == 0 and self.truncation <= 0.15:
            return "Easy"
        elif bb_height >= 25 and self.occlusion in [0, 1] and self.truncation <= 0.30:
            return "Moderate"
        elif (
            bb_height >= 25 and self.occlusion in [0, 1, 2] and self.truncation <= 0.50
        ):
            return "Hard"
        else:
            return "Unknown"

    def print_object(self):
        print(
            "Type, truncation, occlusion, alpha: %s, %d, %d, %f"
            % (self.type, self.truncation, self.occlusion, self.alpha)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %f, %f, %f, %f"
            % (self.xmin, self.ymin, self.xmax, self.ymax)
        )
        print("3d bbox h,w,l: %f, %f, %f" % (self.h, self.w, self.l))
        print(
            "3d bbox location, ry: (%f, %f, %f), %f"
            % (self.t[0], self.t[1], self.t[2], self.ry)
        )
        print("Difficulty of estimation: {}".format(self.estimate_difficulty()))

# Color definitions for visualization
INSTANCE_Color = {
    'Car': 'red', 'Pedestrian': 'green', 'Sign': 'yellow', 'Cyclist': 'purple',
    'Van': 'orange', 'Truck': 'brown', 'Person_sitting': 'lightgreen',
    'Tram': 'pink', 'Misc': 'gray'
}

INSTANCE3D_ColorCV2 = {
    'Car': (0, 255, 0), 'Pedestrian': (255, 255, 0), 'Sign': (0, 255, 255), 
    'Cyclist': (127, 127, 64), 'Van': (255, 165, 0), 'Truck': (165, 42, 42),
    'Person_sitting': (144, 238, 144), 'Tram': (255, 192, 203), 'Misc': (128, 128, 128)
}

INSTANCE3D_Color = {
    'Car': (0, 1, 0), 'Pedestrian': (0, 1, 1), 'Sign': (1, 1, 0), 
    'Cyclist': (0.5, 0.5, 0.3), 'Van': (1, 0.65, 0), 'Truck': (0.65, 0.16, 0.16),
    'Person_sitting': (0.56, 0.93, 0.56), 'Tram': (1, 0.75, 0.8), 'Misc': (0.5, 0.5, 0.5)
}

# Camera mappings for different datasets
waymocameraorder = {
    0: 1, 1: 0, 2: 2, 3: 3, 4: 4
}  # Front, front_left, side_left, front_right, side_right

cameraname_map = {
    0: "FRONT", 1: "FRONT_LEFT", 2: "FRONT_RIGHT", 
    3: "SIDE_LEFT", 4: "SIDE_RIGHT"
}

def box_in_image(corners_3d, corners_2d, intrinsic: np.ndarray, imsize: Tuple[int, int], vis_level: int = BoxVisibility.ANY) -> bool:
    """
    Check if a box is visible inside an image without accounting for occlusions.
    Based on KITTI and NuScenes implementation.
    
    Args:
        corners_3d: 3D corners in camera coordinates (3, 8)
        corners_2d: 2D projected corners (2, 8) 
        intrinsic: <float: 3, 3>. Intrinsic camera matrix.
        imsize: (width, height).
        vis_level: One of the enumerations of <BoxVisibility>.
    
    Returns:
        True if visibility condition is satisfied.
    """
    # Check if corners are within image bounds
    visible = np.logical_and(corners_2d[0, :] > 0, corners_2d[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_2d[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_2d[1, :] > 0)
    
    # Check depth - corners should be in front of camera
    in_front = corners_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if vis_level == BoxVisibility.ALL:
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.ANY:
        return any(visible) and all(in_front)
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError("vis_level: {} not valid".format(vis_level))

def download_file(url: str, dest_path: str, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL with progress indication
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        chunk_size: Download chunk size in bytes
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading {os.path.basename(dest_path)} from {url}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Download with progress
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='')
                    else:
                        print(f"\rDownloaded: {downloaded} bytes", end='')
        
        print(f"\n✅ Successfully downloaded {os.path.basename(dest_path)}")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {os.path.basename(dest_path)}: {e}")
        return False

def download_kitti_dataset(download_dir: str, components: List[str] = None) -> bool:
    """
    Download KITTI dataset components
    
    Args:
        download_dir: Directory to download files to
        components: List of components to download (default: all)
        
    Returns:
        True if all downloads successful, False otherwise
    """
    if components is None:
        components = list(KITTI_URLS.keys())
    
    print(f"\n{'='*60}")
    print("KITTI DATASET DOWNLOAD")
    print(f"{'='*60}")
    print(f"Download directory: {download_dir}")
    print(f"Components to download: {', '.join(components)}")
    
    os.makedirs(download_dir, exist_ok=True)
    
    success_count = 0
    for component in components:
        if component not in KITTI_URLS:
            print(f"❌ Unknown component: {component}")
            continue
            
        url = KITTI_URLS[component]
        filename = f"data_object_{component}.zip"
        dest_path = os.path.join(download_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(dest_path):
            print(f"⏭️  {filename} already exists, skipping download")
            success_count += 1
            continue
        
        if download_file(url, dest_path):
            success_count += 1
    
    print(f"\n📊 Download Summary: {success_count}/{len(components)} components downloaded successfully")
    return success_count == len(components)

def extract_kitti_files(download_dir: str, extract_dir: str, components: List[str] = None) -> bool:
    """
    Extract KITTI dataset files from zip archives
    
    Args:
        download_dir: Directory containing downloaded zip files
        extract_dir: Directory to extract files to
        components: List of components to extract (default: all)
        
    Returns:
        True if all extractions successful, False otherwise
    """
    if components is None:
        components = list(KITTI_URLS.keys())
    
    print(f"\n{'='*60}")
    print("KITTI DATASET EXTRACTION")
    print(f"{'='*60}")
    print(f"Source directory: {download_dir}")
    print(f"Extract directory: {extract_dir}")
    
    os.makedirs(extract_dir, exist_ok=True)
    
    success_count = 0
    for component in components:
        filename = f"data_object_{component}.zip"
        zip_path = os.path.join(download_dir, filename)
        
        if not os.path.exists(zip_path):
            print(f"❌ {filename} not found, skipping extraction")
            continue
        
        try:
            print(f"📦 Extracting {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"✅ Successfully extracted {filename}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed to extract {filename}: {e}")
    
    print(f"\n📊 Extraction Summary: {success_count}/{len(components)} components extracted successfully")
    
    # Validate extraction
    if success_count > 0:
        validate_kitti_structure(extract_dir)
    
    return success_count == len(components)

def validate_kitti_structure(kitti_root: str, dataset_type: str = 'auto') -> bool:
    """
    Validate KITTI dataset structure
    
    Args:
        kitti_root: Root directory of the KITTI dataset
        dataset_type: 'standard', 'waymokitti', or 'auto' for auto-detection
        
    Returns:
        True if structure is valid, False otherwise
    """
    print(f"\n{'='*60}")
    print("KITTI DATASET STRUCTURE VALIDATION")
    print(f"{'='*60}")
    
    if not os.path.exists(kitti_root):
        print(f"❌ Dataset root directory does not exist: {kitti_root}")
        return False
    
    print(f"Validating dataset at: {kitti_root}")
    
    # Auto-detect dataset type
    if dataset_type == 'auto':
        if os.path.exists(os.path.join(kitti_root, 'training')):
            # Standard KITTI structure with training/testing folders
            dataset_type = 'standard_split'
            training_dir = os.path.join(kitti_root, 'training')
            testing_dir = os.path.join(kitti_root, 'testing')
        elif os.path.exists(os.path.join(kitti_root, 'image_0')):
            # WaymoKITTI format
            dataset_type = 'waymokitti'
        elif os.path.exists(os.path.join(kitti_root, 'image_2')):
            # Standard KITTI format (flat structure)
            dataset_type = 'standard'
        else:
            print("❌ Cannot auto-detect dataset type")
            return False
    
    print(f"Detected dataset type: {dataset_type}")
    
    structure_valid = True
    
    if dataset_type == 'standard_split':
        # Check training and testing directories
        for split in ['training', 'testing']:
            split_dir = os.path.join(kitti_root, split)
            if os.path.exists(split_dir):
                print(f"✅ {split}/ directory found")
                
                # Check subdirectories
                required_dirs = KITTI_STRUCTURE['training'] if split == 'training' else KITTI_STRUCTURE['testing']
                for subdir in required_dirs:
                    subdir_path = os.path.join(split_dir, subdir)
                    if os.path.exists(subdir_path):
                        file_count = len([f for f in os.listdir(subdir_path) 
                                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bin', '.txt'))])
                        print(f"  ✅ {subdir}/: {file_count} files")
                    else:
                        print(f"  ❌ {subdir}/: missing")
                        structure_valid = False
            else:
                print(f"❌ {split}/ directory missing")
                structure_valid = False
    
    elif dataset_type == 'waymokitti':
        # Check WaymoKITTI structure
        required_dirs = REQUIRED_KITTI_DIRS['waymokitti']
        for dir_name in required_dirs:
            dir_path = os.path.join(kitti_root, dir_name)
            if os.path.exists(dir_path):
                file_count = len([f for f in os.listdir(dir_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bin', '.txt'))])
                print(f"✅ {dir_name}/: {file_count} files")
            else:
                print(f"❌ {dir_name}/: missing")
                structure_valid = False
    
    elif dataset_type == 'standard':
        # Check standard KITTI structure (flat)
        required_dirs = REQUIRED_KITTI_DIRS['standard']
        for dir_name in required_dirs:
            dir_path = os.path.join(kitti_root, dir_name)
            if os.path.exists(dir_path):
                file_count = len([f for f in os.listdir(dir_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bin', '.txt'))])
                print(f"✅ {dir_name}/: {file_count} files")
            else:
                print(f"❌ {dir_name}/: missing")
                structure_valid = False
    
    if structure_valid:
        print("\n✅ Dataset structure validation passed")
    else:
        print("\n❌ Dataset structure validation failed")
    
    return structure_valid

def filter_lidarpoints(pc_velo, point_cloud_range=[0, -15, -5, 90, 15, 4]):
    """Filter LiDAR Points within specified range"""
    mask = (pc_velo[:, 0] >= point_cloud_range[0]) & (pc_velo[:, 0] <= point_cloud_range[3]) \
           & (pc_velo[:, 1] >= point_cloud_range[1]) & (pc_velo[:, 1] <= point_cloud_range[4]) \
           & (pc_velo[:, 2] >= point_cloud_range[2]) & (pc_velo[:, 2] <= point_cloud_range[5]) \
           & (pc_velo[:, 3] <= 1) 
    filteredpoints = pc_velo[mask]
    print(f"Filtered points shape: {filteredpoints.shape}")
    return filteredpoints

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def draw_projected_box3d_matplotlib(ax, qs, color='red', linewidth=2):
    """ Draw 3d bounding box on matplotlib axes with boundary clipping
        qs: (8,2) array of 2D vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    if qs is None or len(qs) < 8:
        return
    
    # Get image bounds from axes
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_min, x_max = min(xlim), max(xlim)
    y_min, y_max = min(ylim), max(ylim)
    
    def clip_line(x1, y1, x2, y2):
        """Clip line to image boundaries using Cohen-Sutherland algorithm"""
        # Define region codes
        INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8
        
        def compute_code(x, y):
            code = INSIDE
            if x < x_min: code |= LEFT
            elif x > x_max: code |= RIGHT
            if y < y_min: code |= BOTTOM
            elif y > y_max: code |= TOP
            return code
        
        code1 = compute_code(x1, y1)
        code2 = compute_code(x2, y2)
        
        while True:
            # Both points inside
            if code1 == 0 and code2 == 0:
                return x1, y1, x2, y2, True
            
            # Both points outside same region
            if code1 & code2:
                return None, None, None, None, False
            
            # At least one point outside
            code_out = code1 if code1 != 0 else code2
            
            if code_out & TOP:
                x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
                y = y_max
            elif code_out & BOTTOM:
                x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
                y = y_min
            elif code_out & RIGHT:
                y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
                x = x_max
            elif code_out & LEFT:
                y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
                x = x_min
            
            if code_out == code1:
                x1, y1 = x, y
                code1 = compute_code(x1, y1)
            else:
                x2, y2 = x, y
                code2 = compute_code(x2, y2)
    
    # Draw the 12 edges of the 3D box with clipping
    # Front face (0,1,2,3)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        x1, y1, x2, y2, visible = clip_line(qs[i, 0], qs[i, 1], qs[j, 0], qs[j, 1])
        if visible:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth)
    
    # Back face (4,5,6,7)
    for k in range(0, 4):
        i, j = k + 4, (k + 1) % 4 + 4
        x1, y1, x2, y2, visible = clip_line(qs[i, 0], qs[i, 1], qs[j, 0], qs[j, 1])
        if visible:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth)
    
    # Connecting edges
    for k in range(0, 4):
        i, j = k, k + 4
        x1, y1, x2, y2, visible = clip_line(qs[i, 0], qs[i, 1], qs[j, 0], qs[j, 1])
        if visible:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth)

def read_label(label_filename):
    """Read KITTI format label file"""
    if os.path.exists(label_filename):
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [Object3d(line) for line in lines]
        return objects
    else:
        return []

def read_multi_label(label_files):
    """Read multiple label files"""
    objectlabels = []
    for label_file in label_files:
        object3dlabel = read_label(label_file)
        objectlabels.append(object3dlabel)
    return objectlabels

def load_image(img_filenames, jpgfile=False):
    """Load multiple images"""
    imgs = []
    for img_filename in img_filenames:
        if jpgfile:
            img_filename = img_filename.replace('.png', '.jpg')
        if os.path.exists(img_filename):
            img = cv2.imread(img_filename)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(rgb)
        else:
            print(f"Warning: Image file not found: {img_filename}")
            imgs.append(None)
    return imgs

def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4, filterpoints=False, point_cloud_range=[0, -15, -5, 90, 15, 4]):
    """Load velodyne point cloud data"""
    if not os.path.exists(velo_filename):
        print(f"Warning: Velodyne file not found: {velo_filename}")
        return None
        
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    
    xpoints = scan[:, 0]
    ypoints = scan[:, 1]
    zpoints = scan[:, 2]
    
    print(f"Point cloud range - X: [{min(xpoints):.2f}, {max(xpoints):.2f}], "
          f"Y: [{min(ypoints):.2f}, {max(ypoints):.2f}], "
          f"Z: [{min(zpoints):.2f}, {max(zpoints):.2f}]")
    
    if filterpoints:
        print(f"Applying point cloud filter: X[{point_cloud_range[0]}, {point_cloud_range[3]}], "
              f"Y[{point_cloud_range[1]}, {point_cloud_range[4]}], "
              f"Z[{point_cloud_range[2]}, {point_cloud_range[5]}]")
        scan = filter_lidarpoints(scan, point_cloud_range)
    
    return scan

def compute_box_3d(obj, dataset='kitti'):
    """ Takes an object3D and returns 3D bounding box corners
        Returns:
            corners_3d: (8,3) array in rect camera coord.
    """
    if not CALIB_UTILS_AVAILABLE:
        print("Warning: CalibrationUtils not available, using basic rotation")
        # Basic rotation matrix around Y axis
        cos_ry = np.cos(obj.ry)
        sin_ry = np.sin(obj.ry)
        R = np.array([[cos_ry, 0, sin_ry],
                      [0, 1, 0],
                      [-sin_ry, 0, cos_ry]])
    else:
        R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l  # length
    w = obj.w  # width
    h = obj.h  # height

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    
    return np.transpose(corners_3d)

def pltshow_image_with_boxes(cameraid, img, objects, layout, cmap=None):
    """Show image with 2D bounding boxes"""
    import matplotlib.patches as patches
    
    ax = plt.subplot(*layout)
    if img is None:
        plt.text(0.5, 0.5, 'Image not available', ha='center', va='center', transform=ax.transAxes)
        return
        
    plt.imshow(img, cmap=cmap)
    plt.title(cameraname_map.get(cameraid, f"Camera {cameraid}"))
    
    if not objects or len(objects) == 0:
        return
        
    for obj in objects:
        if obj.type == "DontCare":
            continue
            
        box = obj.box2d
        objectclass = obj.type
        
        if objectclass in INSTANCE_Color.keys():
            colorlabel = INSTANCE_Color[objectclass]
            [xmin, ymin, xmax, ymax] = box
            width = xmax - xmin
            height = ymax - ymin
            
            if height > 0 and width > 0:
                ax.add_patch(patches.Rectangle(
                    xy=(xmin, ymin),
                    width=width,
                    height=height,
                    linewidth=1,
                    edgecolor=colorlabel,
                    facecolor='none'))
                ax.text(xmin, ymin, objectclass, color=colorlabel, fontsize=8)
        else:
            print(f"Unknown object class: {objectclass}")
    
    plt.grid(False)
    plt.axis('on')

def plt_multiimages(images, objectlabels, datasetname, order=1):
    """Plot multiple images with bounding boxes"""
    plt.figure(order, figsize=(16, 9))
    camera_count = len(images)
    
    for count in range(camera_count):
        if datasetname.lower() == 'waymokitti':
            index = waymocameraorder.get(count, count)
            if index < len(images) and index < len(objectlabels):
                pltshow_image_with_boxes(index, images[index], objectlabels[index], [3, 3, count + 1])
        elif datasetname.lower() == 'kitti':
            index = count
            if index < len(images) and index < len(objectlabels):
                pltshow_image_with_boxes(index, images[index], objectlabels[index], [1, 2, count + 1])

def pltshow_image_with_3Dboxes(cameraid, img, objects, calib, layout, cmap=None):
    """Show image with 3D bounding boxes projected to 2D"""
    ax = plt.subplot(*layout)
    
    if img is None:
        plt.text(0.5, 0.5, 'Image not available', ha='center', va='center', transform=ax.transAxes)
        return
        
    img2 = np.copy(img)
    print(f"Processing camera id: {cameraid}")
    
    z_front_min = 0.1
    
    if not CALIB_UTILS_AVAILABLE:
        print("Warning: CalibrationUtils not available, skipping 3D box projection")
        plt.imshow(img2, cmap=cmap)
        plt.title(cameraname_map.get(cameraid, f"Camera {cameraid}"))
        return
    
    if cameraid == 0:
        for obj in objects:
            if obj.type == "DontCare" or obj is None:
                continue
                
            box3d_pts_3d = compute_box_3d(obj)
            
            if np.any(box3d_pts_3d[:, 2] < z_front_min):
                continue
                
            try:
                box3d_pts_2d, _ = calib.project_cam3d_to_image(box3d_pts_3d, cameraid)
                
                if box3d_pts_2d is not None:
                    if obj.type in INSTANCE3D_ColorCV2.keys():
                        colorlabel = INSTANCE3D_ColorCV2[obj.type]
                        img2 = draw_projected_box3d(img2, box3d_pts_2d, color=colorlabel)
                    else:
                        print(f"Unknown object type for 3D visualization: {obj.type}")
            except Exception as e:
                print(f"Error projecting 3D box: {e}")
    else:
        # Handle other cameras (transform through velodyne coordinate)
        ref_cameraid = 0
        for obj in objects:
            if obj.type == "DontCare" or obj is None:
                continue
                
            try:
                box3d_pts_3d = compute_box_3d(obj)
                box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d, ref_cameraid)
                box3d_pts_3d_cam = calib.project_velo_to_cameraid(box3d_pts_3d_velo, cameraid)
                box3d_pts_2d, _ = calib.project_cam3d_to_image(box3d_pts_3d_cam, cameraid)
                
                if box3d_pts_2d is not None:
                    if obj.type in INSTANCE3D_ColorCV2.keys():
                        colorlabel = INSTANCE3D_ColorCV2[obj.type]
                        img2 = draw_projected_box3d(img2, box3d_pts_2d, color=colorlabel)
            except Exception as e:
                print(f"Error projecting 3D box for camera {cameraid}: {e}")

    plt.imshow(img2, cmap=cmap)
    plt.title(cameraname_map.get(cameraid, f"Camera {cameraid}"))
    plt.grid(False)
    plt.axis('on')

def plt3dbox_images(images, objectlabels, calib, datasetname='kitti'):
    """Plot images with 3D bounding boxes"""
    plt.figure(figsize=(16, 9))
    camera_count = len(images)
    
    for count in range(camera_count):
        if datasetname.lower() == 'waymokitti':
            index = waymocameraorder.get(count, count)
            if index < len(images) and index < len(objectlabels):
                img = images[index]
                object3dlabel = objectlabels[index]
                pltshow_image_with_3Dboxes(index, img, object3dlabel, calib, [3, 3, count + 1])
        elif datasetname.lower() == 'kitti':
            index = count
            if index < len(images) and index < len(objectlabels):
                img = images[index]
                object3dlabel = objectlabels[index]
                pltshow_image_with_3Dboxes(index, img, object3dlabel, calib, [1, 2, count + 1])

def plotlidar_to_image(pts_3d, img, calib, cameraid=0):
    """Project 3D LiDAR points to image plane"""
    if pts_3d is None or img is None:
        print("Warning: Missing point cloud or image data")
        return
        
    if not CALIB_UTILS_AVAILABLE:
        print("Warning: CalibrationUtils not available, skipping LiDAR projection")
        return
        
    fig = plt.figure(figsize=(16, 9))
    plt.imshow(img)

    try:
        pts_2d, pts_depth = calib.project_velo_to_image(pts_3d, cameraid)

        # Remove points outside the image
        inds = pts_2d[:, 0] > 0
        inds = np.logical_and(inds, pts_2d[:, 0] < img.shape[1])
        inds = np.logical_and(inds, pts_2d[:, 1] > 0)
        inds = np.logical_and(inds, pts_2d[:, 1] < img.shape[0])
        inds = np.logical_and(inds, pts_depth > 0)

        plt.scatter(pts_2d[inds, 0], pts_2d[inds, 1], c=-pts_depth[inds], 
                   alpha=0.5, s=1, cmap='viridis')
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error projecting LiDAR to image: {e}")

def pltlidar_with3dbox(pc_velo, object3dlabels, calib, point_cloud_range):
    """Visualize LiDAR point cloud with 3D bounding boxes using Mayavi"""
    if not MAYAVI_AVAILABLE or not MAYAVI_UTILS_AVAILABLE:
        print("Warning: Mayavi or visualization utils not available, skipping 3D LiDAR visualization")
        return
        
    if pc_velo is None:
        print("Warning: No point cloud data available")
        return

    try:
        fig = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
        )
        
        draw_lidar(pc_velo, fig=fig, pts_scale=5, pc_label=False, 
                  color_by_intensity=True, drawregion=True, 
                  point_cloud_range=point_cloud_range)

        # Draw 3D bounding boxes
        ref_cameraid = 0
        for obj in object3dlabels:
            if obj.type == "DontCare":
                continue
                
            try:
                box3d_pts_3d = compute_box_3d(obj)
                box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d, ref_cameraid)
                
                if obj.type in INSTANCE3D_Color.keys():
                    colorlabel = INSTANCE3D_Color[obj.type]
                    draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=colorlabel, label=obj.type)
                else:
                    print(f"Unknown object type for 3D visualization: {obj.type}")
            except Exception as e:
                print(f"Error drawing 3D box for {obj.type}: {e}")

        mlab.show()
        
    except Exception as e:
        print(f"Error in 3D LiDAR visualization: {e}")

def datasetinfo(datasetname):
    """Get dataset-specific information"""
    if datasetname.lower() == 'waymokitti':
        camera_index = 0  # front camera of Waymo is image_0
        max_cameracount = 5
    elif datasetname.lower() == 'kitti':
        camera_index = 2  # front camera of KITTI is image_2
        max_cameracount = 2
    else:
        camera_index = 0
        max_cameracount = 1
    return camera_index, max_cameracount

def getcalibration(datasetname, calibration_file):
    """Get calibration object based on dataset type"""
    if not CALIB_UTILS_AVAILABLE:
        print("Warning: CalibrationUtils not available")
        return None
        
    if not os.path.exists(calibration_file):
        print(f"Warning: Calibration file not found: {calibration_file}")
        return None
        
    try:
        if datasetname.lower() == 'waymokitti':
            calib = WaymoCalibration(calibration_file)
        elif datasetname.lower() == 'kitti':
            calib = KittiCalibration(calibration_file)
        else:
            print(f"Unknown dataset type: {datasetname}")
            return None
        return calib
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None

def count_dataset_files(kitti_root: str, dataset_type: str = 'auto') -> Dict[str, int]:
    """
    Count files in KITTI dataset directories
    
    Args:
        kitti_root: Root directory of the KITTI dataset
        dataset_type: 'standard', 'waymokitti', or 'auto'
        
    Returns:
        Dictionary with file counts by directory
    """
    print(f"\n{'='*60}")
    print("KITTI DATASET FILE COUNT")
    print(f"{'='*60}")
    
    file_counts = {}
    
    # Auto-detect dataset type if needed
    if dataset_type == 'auto':
        if os.path.exists(os.path.join(kitti_root, 'image_0')):
            dataset_type = 'waymokitti'
        elif os.path.exists(os.path.join(kitti_root, 'image_2')):
            dataset_type = 'standard'
        elif os.path.exists(os.path.join(kitti_root, 'training')):
            dataset_type = 'standard_split'
        else:
            print("❌ Cannot determine dataset type")
            return file_counts
    
    print(f"Dataset type: {dataset_type}")
    
    if dataset_type == 'waymokitti':
        dirs_to_check = REQUIRED_KITTI_DIRS['waymokitti']
        for dir_name in dirs_to_check:
            dir_path = os.path.join(kitti_root, dir_name)
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bin', '.txt'))]
                file_counts[dir_name] = len(files)
                print(f"📁 {dir_name}: {len(files)} files")
            else:
                file_counts[dir_name] = 0
                print(f"❌ {dir_name}: directory not found")
    
    elif dataset_type == 'standard':
        dirs_to_check = REQUIRED_KITTI_DIRS['standard']
        for dir_name in dirs_to_check:
            dir_path = os.path.join(kitti_root, dir_name)
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bin', '.txt'))]
                file_counts[dir_name] = len(files)
                print(f"📁 {dir_name}: {len(files)} files")
            else:
                file_counts[dir_name] = 0
                print(f"❌ {dir_name}: directory not found")
    
    elif dataset_type == 'standard_split':
        for split in ['training', 'testing']:
            split_dir = os.path.join(kitti_root, split)
            if os.path.exists(split_dir):
                print(f"\n📂 {split.upper()} split:")
                split_dirs = KITTI_STRUCTURE['training'] if split == 'training' else KITTI_STRUCTURE['testing']
                for dir_name in split_dirs:
                    dir_path = os.path.join(split_dir, dir_name)
                    if os.path.exists(dir_path):
                        files = [f for f in os.listdir(dir_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bin', '.txt'))]
                        file_counts[f"{split}_{dir_name}"] = len(files)
                        print(f"  📁 {dir_name}: {len(files)} files")
                    else:
                        file_counts[f"{split}_{dir_name}"] = 0
                        print(f"  ❌ {dir_name}: directory not found")
    
    return file_counts

def diagnose_kitti_dataset(kitti_root: str) -> Dict[str, Any]:
    """
    Comprehensive KITTI dataset diagnosis
    
    Args:
        kitti_root: Root directory of the KITTI dataset
        
    Returns:
        Dictionary containing diagnosis results
    """
    print(f"\n{'='*60}")
    print("KITTI DATASET COMPREHENSIVE DIAGNOSIS")
    print(f"{'='*60}")
    
    diagnosis = {
        'structure_issues': [],
        'missing_files': [],
        'data_integrity': {},
        'suggestions': [],
        'status': 'unknown',
        'dataset_type': 'unknown'
    }
    
    if not os.path.exists(kitti_root):
        diagnosis['structure_issues'].append(f"Root directory does not exist: {kitti_root}")
        diagnosis['status'] = 'critical'
        return diagnosis
    
    # Detect dataset type
    if os.path.exists(os.path.join(kitti_root, 'training')):
        diagnosis['dataset_type'] = 'standard_split'
    elif os.path.exists(os.path.join(kitti_root, 'image_0')):
        diagnosis['dataset_type'] = 'waymokitti'
    elif os.path.exists(os.path.join(kitti_root, 'image_2')):
        diagnosis['dataset_type'] = 'standard'
    else:
        diagnosis['structure_issues'].append("Cannot determine dataset type")
        diagnosis['status'] = 'critical'
        return diagnosis
    
    print(f"Detected dataset type: {diagnosis['dataset_type']}")
    
    # Check file counts
    file_counts = count_dataset_files(kitti_root, diagnosis['dataset_type'])
    diagnosis['data_integrity'] = file_counts
    
    # Check for consistency issues
    print(f"\n🔍 Checking data consistency...")
    
    if diagnosis['dataset_type'] == 'waymokitti':
        # Check if all camera folders have same number of files
        camera_counts = [file_counts.get(f'image_{i}', 0) for i in range(5)]
        if len(set(camera_counts)) > 1:
            diagnosis['structure_issues'].append(f"Inconsistent camera file counts: {camera_counts}")
        
        # Check if labels match images
        label_counts = [file_counts.get(f'label_{i}', 0) for i in range(5)]
        if camera_counts[0] != label_counts[0]:
            diagnosis['structure_issues'].append(f"Image/label count mismatch: {camera_counts[0]} vs {label_counts[0]}")
    
    elif diagnosis['dataset_type'] == 'standard':
        # Check if image_2 and velodyne have same count
        img_count = file_counts.get('image_2', 0)
        velo_count = file_counts.get('velodyne', 0)
        if img_count != velo_count:
            diagnosis['structure_issues'].append(f"Image/velodyne count mismatch: {img_count} vs {velo_count}")
    
    # Generate suggestions
    if not diagnosis['structure_issues']:
        diagnosis['status'] = 'healthy'
        print("✅ Dataset appears healthy")
    elif len(diagnosis['structure_issues']) <= 2:
        diagnosis['status'] = 'warning'
        print("⚠️  Dataset has minor issues")
    else:
        diagnosis['status'] = 'critical'
        print("❌ Dataset has critical issues")
    
    # Add suggestions based on issues
    if diagnosis['structure_issues']:
        print(f"\n📋 Issues found:")
        for i, issue in enumerate(diagnosis['structure_issues'], 1):
            print(f"  {i}. {issue}")
    
    return diagnosis

def visualize_kitti_sample(kitti_root: str, sample_idx: int, dataset_type: str = 'auto', 
                          camera_count: int = 1, jpgfile: bool = False,
                          point_cloud_range: List[float] = None) -> bool:
    """
    Visualize a KITTI sample with images, 3D boxes, and LiDAR
    
    Args:
        kitti_root: Root directory of KITTI dataset
        sample_idx: Sample index to visualize
        dataset_type: Dataset type ('kitti', 'waymokitti', 'auto')
        camera_count: Number of cameras to visualize
        jpgfile: Whether to use jpg files instead of png
        point_cloud_range: LiDAR point filtering range
        
    Returns:
        True if visualization successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"VISUALIZING KITTI SAMPLE {sample_idx}")
    print(f"{'='*60}")
    
    if point_cloud_range is None:
        if dataset_type == 'waymokitti':
            point_cloud_range = [-100, -60, -8, 100, 60, 8]
        else:
            point_cloud_range = [0, -15, -5, 90, 15, 4]
    
    # Auto-detect dataset type
    if dataset_type == 'auto':
        if os.path.exists(os.path.join(kitti_root, 'image_0')):
            dataset_type = 'waymokitti'
        elif os.path.exists(os.path.join(kitti_root, 'image_2')):
            dataset_type = 'kitti'
        else:
            print("❌ Cannot determine dataset type")
            return False
    
    camera_index, max_cameracount = datasetinfo(dataset_type)
    camera_count = min(camera_count, max_cameracount)
    
    # Construct file paths
    filename = f"{sample_idx:06d}.png"
    
    if dataset_type == 'waymokitti':
        image_files = [os.path.join(kitti_root, f"image_{i}", filename) for i in range(camera_count)]
        labels_files = [os.path.join(kitti_root, f"label_{i}", filename.replace('png', 'txt')) for i in range(camera_count)]
        label_all_file = os.path.join(kitti_root, 'label_all', filename.replace('png', 'txt'))
    else:
        image_files = [os.path.join(kitti_root, f"image_{i+camera_index}", filename) for i in range(camera_count)]
        labels_files = [os.path.join(kitti_root, f"label_{camera_index}", filename.replace('png', 'txt'))]
        label_all_file = labels_files[0]
    
    calibration_file = os.path.join(kitti_root, 'calib', filename.replace('png', 'txt'))
    lidar_filename = os.path.join(kitti_root, 'velodyne', filename.replace('png', 'bin'))
    
    # Check if files exist
    missing_files = []
    for f in image_files + [calibration_file, lidar_filename]:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    try:
        # Load data
        print("📂 Loading data...")
        images = load_image(image_files, jpgfile=jpgfile)
        objectlabels = read_multi_label(labels_files)
        pc_velo = load_velo_scan(lidar_filename, filterpoints=True, point_cloud_range=point_cloud_range)
        calib = getcalibration(dataset_type, calibration_file)
        
        if calib is None:
            print("❌ Failed to load calibration")
            return False
        
        # Visualizations
        print("🖼️  Creating visualizations...")
        
        # 2D bounding boxes
        plt_multiimages(images, objectlabels, dataset_type)
        plt.suptitle(f'Sample {sample_idx} - 2D Bounding Boxes')
        plt.tight_layout()
        plt.show()
        
        # 3D bounding boxes on images
        if CALIB_UTILS_AVAILABLE:
            plt3dbox_images(images, objectlabels, calib, dataset_type)
            plt.suptitle(f'Sample {sample_idx} - 3D Bounding Boxes')
            plt.tight_layout()
            plt.show()
        
        # LiDAR projection on image
        if pc_velo is not None and len(images) > 0 and images[0] is not None:
            plotlidar_to_image(pc_velo, images[0], calib, cameraid=0)
        
        # 3D LiDAR visualization
        if dataset_type == 'waymokitti':
            object3dlabels = read_label(label_all_file)
        else:
            object3dlabels = objectlabels[0] if objectlabels else []
        
        if MAYAVI_AVAILABLE and pc_velo is not None:
            pltlidar_with3dbox(pc_velo, object3dlabels, calib, point_cloud_range)
        
        print("✅ Visualization completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        return False

def detect_dataset_type(root_path):
    """Auto-detect dataset type based on directory structure"""
    if os.path.exists(os.path.join(root_path, 'image_0')):
        return 'waymokitti'
    elif os.path.exists(os.path.join(root_path, 'image_2')):
        return 'kitti'
    elif os.path.exists(os.path.join(root_path, 'training')):
        return 'standard_split'
    else:
        return 'unknown'

def project_to_image(pts_3d, P):
    """Project 3D points to image plane using projection matrix P"""
    try:
        # Convert to homogeneous coordinates
        pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
        
        # Project to image
        pts_2d_homo = np.dot(P, pts_3d_homo.T)
        
        # Convert from homogeneous to 2D coordinates
        pts_2d = pts_2d_homo[:2, :] / pts_2d_homo[2, :]
        
        return pts_2d.T
    except Exception as e:
        print(f"Error in project_to_image: {e}")
        return None

def get_3d_box_corners(obj):
    """Get 3D bounding box corners for an object"""
    try:
        return compute_box_3d(obj)
    except Exception as e:
        print(f"Error getting 3D box corners: {e}")
        return None

def draw_3d_box_mayavi(mlab, corners):
    """Draw 3D bounding box using Mayavi"""
    try:
        if corners is None or len(corners) != 8:
            return
            
        # Define the 12 edges of a 3D box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # top face
            [4, 5], [5, 6], [6, 7], [7, 4],  # bottom face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        
        # Draw each edge
        for edge in edges:
            p1, p2 = corners[edge[0]], corners[edge[1]]
            mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                       color=(1, 0, 0), tube_radius=0.1)
    except Exception as e:
        print(f"Error drawing 3D box with Mayavi: {e}")

def visualize_kitti_sample_2d(root_path, sample_idx, dataset_type='auto', output_dir='output'):
    """Visualize KITTI sample with 2D bounding boxes"""
    try:
        # Detect dataset type if auto
        if dataset_type == 'auto':
            dataset_type = detect_dataset_type(root_path)
        
        # Get file paths with automatic training/testing folder detection
        def find_file_path(root_path, subfolder, filename):
            """Find file in root_path or in training/testing subfolders"""
            # Try direct path first
            direct_path = os.path.join(root_path, subfolder, filename)
            if os.path.exists(direct_path):
                return direct_path
            
            # Try training folder
            training_path = os.path.join(root_path, 'training', subfolder, filename)
            if os.path.exists(training_path):
                return training_path
            
            # Try testing folder
            testing_path = os.path.join(root_path, 'testing', subfolder, filename)
            if os.path.exists(testing_path):
                return testing_path
            
            # Return direct path as fallback (will show proper error later)
            return direct_path
        
        if dataset_type == 'waymokitti':
            image_path = find_file_path(root_path, 'image_0', f'{sample_idx:06d}.png')
            label_path = find_file_path(root_path, 'label_0', f'{sample_idx:06d}.txt')
            calib_path = find_file_path(root_path, 'calib', f'{sample_idx:06d}.txt')
        else:
            image_path = find_file_path(root_path, 'image_2', f'{sample_idx:06d}.png')
            label_path = find_file_path(root_path, 'label_2', f'{sample_idx:06d}.txt')
            calib_path = find_file_path(root_path, 'calib', f'{sample_idx:06d}.txt')
        
        # Check if files exist
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return False
        
        # Load image
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image
        
        img = Image.open(image_path)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        
        # Load and draw bounding boxes if label file exists
        if os.path.exists(label_path):
            objects = read_label(label_path)
            for obj in objects:
                if hasattr(obj, 'box2d') and obj.box2d is not None:
                    x1, y1, x2, y2 = obj.box2d
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Color based on object type
                    color_map = {'Car': 'red', 'Pedestrian': 'blue', 'Cyclist': 'green'}
                    color = color_map.get(obj.type, 'yellow')
                    
                    rect = patches.Rectangle((x1, y1), width, height, 
                                           linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, obj.type, color=color, fontsize=10, weight='bold')
        
        ax.set_title(f'Sample {sample_idx} - 2D Bounding Boxes')
        ax.axis('off')
        
        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'sample_{sample_idx}_2d_bbox.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.show()
        
        print(f"✅ 2D visualization saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error in 2D visualization: {e}")
        return False

def visualize_kitti_sample_3d(root_path, sample_idx, dataset_type='auto', output_dir='output'):
    """Visualize KITTI sample with 3D bounding boxes projected on image"""
    try:
        # Detect dataset type if auto
        if dataset_type == 'auto':
            dataset_type = detect_dataset_type(root_path)
        
        # Get file paths with automatic training/testing folder detection
        def find_file_path(root_path, subfolder, filename):
            """Find file in root_path or in training/testing subfolders"""
            # Try direct path first
            direct_path = os.path.join(root_path, subfolder, filename)
            if os.path.exists(direct_path):
                return direct_path
            
            # Try training folder
            training_path = os.path.join(root_path, 'training', subfolder, filename)
            if os.path.exists(training_path):
                return training_path
            
            # Try testing folder
            testing_path = os.path.join(root_path, 'testing', subfolder, filename)
            if os.path.exists(testing_path):
                return testing_path
            
            # Return direct path as fallback (will show proper error later)
            return direct_path
        
        if dataset_type == 'waymokitti':
            image_path = find_file_path(root_path, 'image_0', f'{sample_idx:06d}.png')
            label_path = find_file_path(root_path, 'label_0', f'{sample_idx:06d}.txt')
            calib_path = find_file_path(root_path, 'calib', f'{sample_idx:06d}.txt')
        else:
            image_path = find_file_path(root_path, 'image_2', f'{sample_idx:06d}.png')
            label_path = find_file_path(root_path, 'label_2', f'{sample_idx:06d}.txt')
            calib_path = find_file_path(root_path, 'calib', f'{sample_idx:06d}.txt')
        
        # Check if files exist
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return False
        if not os.path.exists(calib_path):
            print(f"❌ Calibration file not found: {calib_path}")
            return False
        
        # Load image
        import matplotlib.pyplot as plt
        from PIL import Image
        
        img = Image.open(image_path)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        
        # Load calibration
        if dataset_type == 'waymokitti':
            calib = WaymoCalibration(calib_path)
        else:
            calib = KittiCalibration(calib_path)
        
        # Load and draw 3D bounding boxes if label file exists
        if os.path.exists(label_path):
            objects = read_label(label_path)
            print(f"📊 Found {len(objects)} objects in label file")
            for obj in objects:
                # Check if object has valid 3D information
                if hasattr(obj, 'h') and hasattr(obj, 'w') and hasattr(obj, 'l') and hasattr(obj, 't'):
                    print(f"🎯 Processing {obj.type} object")
                    # Project 3D box to image
                    corners_3d = compute_box_3d(obj, calib.P)
                    if corners_3d is not None:
                        corners_2d = project_to_image(corners_3d, calib.P)
                        if corners_2d is not None:
                            # Get image dimensions for validation
                            img_height, img_width = img.size[1], img.size[0]
                            
                            # Validate if any part of the box is visible in the image
                            visible_corners = 0
                            for corner in corners_2d:
                                if (0 <= corner[0] <= img_width and 0 <= corner[1] <= img_height):
                                    visible_corners += 1
                            
                            # Only draw if at least one corner is visible or if box intersects image bounds
                            box_intersects = (
                                np.any((corners_2d[:, 0] >= 0) & (corners_2d[:, 0] <= img_width)) and
                                np.any((corners_2d[:, 1] >= 0) & (corners_2d[:, 1] <= img_height))
                            )
                            
                            if visible_corners > 0 or box_intersects:
                                # Color based on object type
                                color_map = {'Car': 'red', 'Pedestrian': 'blue', 'Cyclist': 'green'}
                                color = color_map.get(obj.type, 'yellow')
                                
                                # Draw 3D box
                                draw_projected_box3d_matplotlib(ax, corners_2d, color=color)
                                
                                # Add label with smart positioning
                                if len(corners_2d) > 0:
                                    # Find the topmost visible corner for label placement
                                    valid_corners = corners_2d[corners_2d[:, 0] >= 0]  # x >= 0
                                    valid_corners = valid_corners[valid_corners[:, 0] <= img_width]  # x <= width
                                    valid_corners = valid_corners[valid_corners[:, 1] >= 0]  # y >= 0
                                    valid_corners = valid_corners[valid_corners[:, 1] <= img_height]  # y <= height
                                    
                                    if len(valid_corners) > 0:
                                        # Use the corner with minimum y (topmost) for label
                                        top_corner = valid_corners[np.argmin(valid_corners[:, 1])]
                                        label_x, label_y = top_corner[0], top_corner[1]
                                        
                                        # Ensure label stays within image bounds
                                        label_x = max(10, min(label_x, img_width - 50))  # Keep some margin
                                        label_y = max(15, min(label_y - 5, img_height - 5))  # Above the box
                                        
                                        ax.text(label_x, label_y, obj.type, 
                                               color=color, fontsize=10, weight='bold',
                                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
                                    else:
                                        # If no corners are visible, place label at image center
                                        ax.text(img_width//2, 30, f"{obj.type} (off-screen)", 
                                               color=color, fontsize=10, weight='bold',
                                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7),
                                               ha='center')
                                print(f"✅ Drew 3D box for {obj.type} ({visible_corners} corners visible)")
                            else:
                                print(f"⚠️ Skipped {obj.type} - completely outside image bounds")
                        else:
                            print(f"⚠️ Failed to project 3D corners to 2D for {obj.type}")
                    else:
                        print(f"⚠️ Failed to compute 3D corners for {obj.type}")
                else:
                    print(f"⚠️ Object {obj.type} missing 3D information")
        else:
            print(f"⚠️ Label file not found: {label_path}")
        
        ax.set_title(f'Sample {sample_idx} - 3D Bounding Boxes')
        ax.axis('off')
        
        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'sample_{sample_idx}_3d_bbox.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.show()
        
        print(f"✅ 3D visualization saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error in 3D visualization: {e}")
        return False

def visualize_kitti_sample_lidar(root_path, sample_idx, dataset_type='auto', output_dir='output'):
    """Visualize KITTI sample LiDAR point cloud"""
    try:
        # Detect dataset type if auto
        if dataset_type == 'auto':
            dataset_type = detect_dataset_type(root_path)
        
        # Get file paths with automatic training/testing folder detection
        def find_file_path(root_path, subfolder, filename):
            """Find file in root_path or in training/testing subfolders"""
            # Try direct path first
            direct_path = os.path.join(root_path, subfolder, filename)
            if os.path.exists(direct_path):
                return direct_path
            
            # Try training folder
            training_path = os.path.join(root_path, 'training', subfolder, filename)
            if os.path.exists(training_path):
                return training_path
            
            # Try testing folder
            testing_path = os.path.join(root_path, 'testing', subfolder, filename)
            if os.path.exists(testing_path):
                return testing_path
            
            # Return direct path as fallback (will show proper error later)
            return direct_path
        
        velodyne_path = find_file_path(root_path, 'velodyne', f'{sample_idx:06d}.bin')
        if dataset_type == 'waymokitti':
            label_path = find_file_path(root_path, 'label_0', f'{sample_idx:06d}.txt')
        else:
            label_path = find_file_path(root_path, 'label_2', f'{sample_idx:06d}.txt')
        
        # Check if files exist
        if not os.path.exists(velodyne_path):
            print(f"❌ LiDAR file not found: {velodyne_path}")
            return False
        
        # Load point cloud
        points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
        
        # Try to use Mayavi for 3D visualization if available
        try:
            import mayavi.mlab as mlab
            
            # Create 3D plot
            fig = mlab.figure(bgcolor=(0, 0, 0), size=(800, 600))
            
            # Plot points colored by height
            pts = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], 
                               points[:, 2], mode='point', colormap='spectral', 
                               scale_factor=0.1)
            
            # Load and visualize 3D bounding boxes if available
            if os.path.exists(label_path):
                objects = read_label(label_path)
                for obj in objects:
                    if hasattr(obj, 'box3d') and obj.box3d is not None:
                        # Draw 3D bounding box in point cloud
                        corners = get_3d_box_corners(obj)
                        if corners is not None:
                            # Draw box edges
                            draw_3d_box_mayavi(mlab, corners)
            
            mlab.title(f'Sample {sample_idx} - LiDAR Point Cloud')
            
            # Save visualization
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'sample_{sample_idx}_lidar.png')
            mlab.savefig(output_path)
            mlab.show()
            
            print(f"✅ LiDAR visualization saved to: {output_path}")
            return True
            
        except ImportError:
            # Fallback to matplotlib 2D visualization
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Top view (X-Y plane)
            ax1.scatter(points[:, 0], points[:, 1], c=points[:, 2], 
                       cmap='viridis', s=0.1, alpha=0.6)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('Top View')
            ax1.axis('equal')
            
            # Side view (X-Z plane)
            ax2.scatter(points[:, 0], points[:, 2], c=points[:, 1], 
                       cmap='viridis', s=0.1, alpha=0.6)
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Z (m)')
            ax2.set_title('Side View')
            
            plt.suptitle(f'Sample {sample_idx} - LiDAR Point Cloud (2D Views)')
            
            # Save visualization
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'sample_{sample_idx}_lidar_2d.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.show()
            
            print(f"✅ LiDAR 2D visualization saved to: {output_path}")
            print("💡 Install Mayavi for 3D LiDAR visualization: pip install mayavi")
            return True
        
    except Exception as e:
        print(f"❌ Error in LiDAR visualization: {e}")
        return False

def draw_bev_box_2d(ax, box_center, box_size, rotation_y, category, alpha=0.3):
    """
    Draw a 2D bounding box in BEV (Bird's Eye View)
    
    Args:
        ax: matplotlib axis
        box_center: [x, y] center of the box in BEV coordinates
        box_size: [length, width] size of the box
        rotation_y: rotation angle around Y axis
        category: object category for color coding
        alpha: transparency for box fill
    """
    # Get color for category
    color = INSTANCE_Color.get(category, 'gray')
    
    # Create box corners (length x width)
    l, w = box_size[0], box_size[1]
    corners = np.array([
        [-l/2, -w/2],  # rear left
        [l/2, -w/2],   # front left  
        [l/2, w/2],    # front right
        [-l/2, w/2]    # rear right
    ])
    
    # Rotate corners
    cos_r, sin_r = np.cos(rotation_y), np.sin(rotation_y)
    rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    rotated_corners = corners @ rotation_matrix.T
    
    # Translate to box center
    rotated_corners += box_center
    
    # Close the polygon
    rotated_corners = np.vstack([rotated_corners, rotated_corners[0]])
    
    # Draw box outline
    ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], 
            color=color, linewidth=2, alpha=0.8)
    
    # Fill box with semi-transparent color
    ax.fill(rotated_corners[:, 0], rotated_corners[:, 1], 
            color=color, alpha=alpha)
    
    # Add direction indicator (arrow pointing forward)
    front_center = box_center + np.array([l/2 * cos_r, l/2 * sin_r])
    ax.arrow(box_center[0], box_center[1], 
             front_center[0] - box_center[0], front_center[1] - box_center[1],
             head_width=0.5, head_length=0.3, fc=color, ec=color, alpha=0.8)

def visualize_kitti_sample_bev(root_path, sample_idx, dataset_type='auto', output_dir='output'):
    """
    Visualize KITTI sample in Bird's Eye View (BEV) with LiDAR points and 3D bounding boxes
    
    Args:
        root_path: Path to KITTI dataset root
        sample_idx: Sample index to visualize
        dataset_type: Type of dataset ('kitti', 'waymokitti', or 'auto')
        output_dir: Directory to save visualization
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Detect dataset type if auto
        if dataset_type == 'auto':
            dataset_type = 'waymokitti' if 'waymo' in root_path.lower() else 'kitti'
        
        # Set up paths based on dataset type
        if dataset_type == 'waymokitti':
            velodyne_dir = os.path.join(root_path, 'velodyne')
            label_dir = os.path.join(root_path, 'label_all')
            calib_dir = os.path.join(root_path, 'calib')
        else:
            velodyne_dir = os.path.join(root_path, 'velodyne')
            label_dir = os.path.join(root_path, 'label_2')
            calib_dir = os.path.join(root_path, 'calib')
        
        # Load LiDAR points
        velodyne_file = os.path.join(velodyne_dir, f'{sample_idx:06d}.bin')
        if not os.path.exists(velodyne_file):
            print(f"❌ Velodyne file not found: {velodyne_file}")
            return False
        
        points = load_velo_scan(velodyne_file)
        print(f"📊 Loaded {len(points)} LiDAR points")
        
        # Load labels
        label_file = os.path.join(label_dir, f'{sample_idx:06d}.txt')
        objects = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    obj = Object3d(line.strip())
                    objects.append(obj)
            print(f"📊 Loaded {len(objects)} objects")
        else:
            print(f"⚠️ Label file not found: {label_file}")
        
        # Load calibration
        calib_file = os.path.join(calib_dir, f'{sample_idx:06d}.txt')
        if not os.path.exists(calib_file):
            print(f"❌ Calibration file not found: {calib_file}")
            return False
        
        if CALIB_UTILS_AVAILABLE:
            calib = KittiCalibration(calib_file)
        else:
            print("⚠️ CalibrationUtils not available, using identity transformation")
            calib = None
        
        # Create BEV visualization
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Filter points within reasonable range (50m x 50m)
        bev_range = 50
        mask = (np.abs(points[:, 0]) < bev_range) & (np.abs(points[:, 1]) < bev_range)
        filtered_points = points[mask]
        
        # Plot LiDAR points colored by height
        if len(filtered_points) > 0:
            scatter = ax.scatter(filtered_points[:, 0], filtered_points[:, 1], 
                               c=filtered_points[:, 2], cmap='viridis', 
                               s=0.5, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Height (m)')
        
        # Draw 3D bounding boxes
        categories_present = set()
        for obj in objects:
            # Skip objects that are too far or have invalid dimensions
            if (abs(obj.t[0]) > bev_range or abs(obj.t[1]) > bev_range or 
                obj.l <= 0 or obj.w <= 0 or obj.h <= 0):
                continue
            
            # Convert 3D box to BEV coordinates
            # KITTI uses camera coordinate system, need to convert to LiDAR coordinates
            if calib is not None:
                # Transform from camera to velodyne coordinates
                box_center_cam = np.array([obj.t[0], obj.t[1], obj.t[2], 1.0])
                box_center_velo = calib.project_rect_to_velo(box_center_cam[:3].reshape(1, -1))[0]
                box_center_bev = [box_center_velo[0], box_center_velo[1]]
            else:
                # Fallback: assume camera and velodyne are aligned
                box_center_bev = [obj.t[2], -obj.t[0]]  # Simple coordinate transformation
            
            # Box dimensions (length, width)
            box_size = [obj.l, obj.w]
            
            # Rotation angle (convert from camera to velodyne frame)
            rotation_y = obj.ry
            
            # Draw the box
            draw_bev_box_2d(ax, box_center_bev, box_size, rotation_y, obj.type)
            categories_present.add(obj.type)
        
        # Add ego vehicle marker (at origin)
        ego_size = [4.5, 2.0]  # Typical car dimensions
        draw_bev_box_2d(ax, [0, 0], ego_size, 0, 'Ego', alpha=0.5)
        
        # Add range circles
        for radius in [10, 20, 30, 40]:
            circle = plt.Circle((0, 0), radius, fill=False, color='white', 
                              linestyle='--', alpha=0.3, linewidth=1)
            ax.add_patch(circle)
            ax.text(radius * 0.707, radius * 0.707, f'{radius}m', 
                   color='white', fontsize=8, alpha=0.7)
        
        # Create legend for object categories
        if categories_present:
            legend_elements = []
            for category in sorted(categories_present):
                color = INSTANCE_Color.get(category, 'gray')
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                                   facecolor=color, alpha=0.7, 
                                                   label=category))
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1.15, 1))
        
        # Set equal aspect ratio and labels
        ax.set_aspect('equal')
        ax.set_xlabel('X (m) - Forward')
        ax.set_ylabel('Y (m) - Left')
        ax.set_title(f'Sample {sample_idx} - Bird\'s Eye View\n'
                    f'LiDAR Points: {len(filtered_points)}, Objects: {len(objects)}')
        
        # Set axis limits
        ax.set_xlim(-bev_range, bev_range)
        ax.set_ylim(-bev_range, bev_range)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('black')
        
        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'sample_{sample_idx}_bev.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='black')
        plt.show()
        
        print(f"✅ BEV visualization saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error in BEV visualization: {e}")
        return False

def main():
    """
    Main function with enhanced KITTI dataset management
    """
    parser = argparse.ArgumentParser(description="KITTI Dataset Utilities and Visualization Tool")
    parser.add_argument("--root_path", default='/DATA10T/Datasets/Kitti/',
                       help="Root path of KITTI dataset")
    parser.add_argument("--download_dir", default='/DATA10T/Datasets/Kitti/',
                       help="Directory to download files to")
    parser.add_argument("--extract_dir", default='/DATA10T/Datasets/Kitti/training/',
                       help="Directory to extract files to")
    parser.add_argument("--output_dir", default="output",
                       help="Directory to save visualizations")
    
    # Support for command-line mode (backward compatibility)
    parser.add_argument("--mode", 
                       choices=['download', 'extract', 'validate', 'diagnose', 'visualize', 'count'],
                       help="Operation mode (for non-interactive use)")
    parser.add_argument("--index", type=int, default=0,
                       help="Sample index for visualization")
    parser.add_argument("--dataset", choices=['kitti', 'waymokitti', 'auto'], default="auto",
                       help="Dataset type")
    parser.add_argument("--camera_count", type=int, default=1,
                       help="Number of cameras to visualize")
    parser.add_argument("--jpgfile", action='store_true',
                       help="Use JPG files instead of PNG")
    parser.add_argument("--vis_type", choices=['2d', '3d', 'lidar', 'bev'], default='2d',
                       help="Visualization type: 2d (bounding boxes), 3d (3D boxes), lidar (point cloud), bev (bird's eye view)")
    parser.add_argument("--components", nargs='+', choices=list(KITTI_URLS.keys()),
                       help="Dataset components to download/extract")
    parser.add_argument("--point_cloud_range", nargs=6, type=float,
                       help="Point cloud filtering range: xmin ymin zmin xmax ymax zmax")
    
    args = parser.parse_args()
    
    # If mode is specified, run in command-line mode (backward compatibility)
    if args.mode:
        return run_command_line_mode(args)
    
    # Interactive mode
    print("="*60)
    print("KITTI DATASET MANAGEMENT TOOL")
    print("="*60)
    print(f"Dataset root: {args.root_path}")
    print(f"Download directory: {args.download_dir}")
    
    while True:
        print("\nSelect an option:")
        print("1. Download KITTI dataset")
        print("2. Extract downloaded files")
        print("3. Validate dataset structure")
        print("4. Run comprehensive dataset diagnosis")
        print("5. Count files in dataset")
        print("6. Visualize sample with 2D bounding boxes")
        print("7. Visualize sample with 3D bounding boxes")
        print("8. Visualize LiDAR point cloud")
        print("9. Visualize Bird's Eye View (BEV)")
        print("10. Complete setup (download + extract + validate)")
        print("11. Get dataset fix suggestions")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-11): ").strip()
        
        if choice == '1':
            components = input("Enter components to download (press Enter for all): ").strip()
            components_list = components.split() if components else None
            success = download_kitti_dataset(args.download_dir, components_list)
            if success:
                print("✅ Download completed successfully")
            else:
                print("❌ Download failed")
                
        elif choice == '2':
            components = input("Enter components to extract (press Enter for all): ").strip()
            components_list = components.split() if components else None
            success = extract_kitti_files(args.download_dir, args.extract_dir, components_list)
            if success:
                print("✅ Extraction completed successfully")
            else:
                print("❌ Extraction failed")
                
        elif choice == '3':
            dataset_type = input("Enter dataset type (kitti/waymokitti/auto, default: auto): ").strip() or "auto"
            success = validate_kitti_structure(args.root_path, dataset_type)
            if success:
                print("✅ Dataset validation passed")
            else:
                print("❌ Dataset validation failed")
                
        elif choice == '4':
            diagnosis = diagnose_kitti_dataset(args.root_path)
            print(f"\n📊 Diagnosis Status: {diagnosis['status'].upper()}")
            if diagnosis['suggestions']:
                print("💡 Suggestions:")
                for suggestion in diagnosis['suggestions']:
                    print(f"  • {suggestion}")
                    
        elif choice == '5':
            dataset_type = input("Enter dataset type (kitti/waymokitti/auto, default: auto): ").strip() or "auto"
            file_counts = count_dataset_files(args.root_path, dataset_type)
            total_files = sum(file_counts.values())
            print(f"\n📊 Total files: {total_files}")
            
        elif choice == '6':
            try:
                sample_idx = int(input("Enter sample index to visualize (default 0): ") or "0")
                dataset_type = input("Enter dataset type (kitti/waymokitti/auto, default: auto): ").strip() or "auto"
                success = visualize_kitti_sample_2d(args.root_path, sample_idx, dataset_type, args.output_dir)
                if not success:
                    print("❌ 2D visualization failed")
            except ValueError:
                print("❌ Invalid sample index. Please enter a valid integer.")
                
        elif choice == '7':
            try:
                sample_idx = int(input("Enter sample index to visualize (default 0): ") or "0")
                dataset_type = input("Enter dataset type (kitti/waymokitti/auto, default: auto): ").strip() or "auto"
                success = visualize_kitti_sample_3d(args.root_path, sample_idx, dataset_type, args.output_dir)
                if not success:
                    print("❌ 3D visualization failed")
            except ValueError:
                print("❌ Invalid sample index. Please enter a valid integer.")
                
        elif choice == '8':
            try:
                sample_idx = int(input("Enter sample index to visualize (default 0): ") or "0")
                dataset_type = input("Enter dataset type (kitti/waymokitti/auto, default: auto): ").strip() or "auto"
                success = visualize_kitti_lidar(args.root_path, sample_idx, dataset_type, args.output_dir)
                if not success:
                    print("❌ LiDAR visualization failed")
            except ValueError:
                print("❌ Invalid sample index. Please enter a valid integer.")
                
        elif choice == '9':
            try:
                sample_idx = int(input("Enter sample index to visualize (default 0): ") or "0")
                dataset_type = input("Enter dataset type (kitti/waymokitti/auto, default: auto): ").strip() or "auto"
                success = visualize_kitti_sample_bev(args.root_path, sample_idx, dataset_type, args.output_dir)
                if not success:
                    print("❌ BEV visualization failed")
            except ValueError:
                print("❌ Invalid sample index. Please enter a valid integer.")
                
        elif choice == '10':
            print("\n🚀 Running complete setup and validation...")
            # Download
            success = download_kitti_dataset(args.download_dir, None)
            if success:
                print("✅ Download completed")
                # Extract
                success = extract_kitti_files(args.download_dir, args.extract_dir, None)
                if success:
                    print("✅ Extraction completed")
                    # Validate
                    success = validate_kitti_structure(args.root_path, "auto")
                    if success:
                        print("✅ Validation completed")
                        # Diagnose
                        diagnosis = diagnose_kitti_dataset(args.root_path)
                        print(f"📊 Dataset Status: {diagnosis['status'].upper()}")
                    else:
                        print("❌ Validation failed")
                else:
                    print("❌ Extraction failed")
            else:
                print("❌ Download failed")
                
        elif choice == '11':
            diagnosis = diagnose_kitti_dataset(args.root_path)
            if diagnosis['suggestions']:
                print("💡 Dataset Fix Suggestions:")
                for i, suggestion in enumerate(diagnosis['suggestions'], 1):
                    print(f"  {i}. {suggestion}")
            else:
                print("✅ No issues found. Dataset appears healthy.")
                
        elif choice == '0':
            print("Exiting program.")
            break
            
        else:
            print("Invalid choice. Please enter a number between 0-11.")

def run_command_line_mode(args):
    """Run in command-line mode for backward compatibility"""
    print(f"🚗 KITTI Dataset Utilities")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'download':
        success = download_kitti_dataset(args.download_dir, args.components)
        if success:
            print("✅ Download completed successfully")
        else:
            print("❌ Download failed")
            return 1
    
    elif args.mode == 'extract':
        success = extract_kitti_files(args.download_dir, args.extract_dir, args.components)
        if success:
            print("✅ Extraction completed successfully")
        else:
            print("❌ Extraction failed")
            return 1
    
    elif args.mode == 'validate':
        success = validate_kitti_structure(args.root_path, args.dataset)
        if success:
            print("✅ Dataset validation passed")
        else:
            print("❌ Dataset validation failed")
            return 1
    
    elif args.mode == 'diagnose':
        diagnosis = diagnose_kitti_dataset(args.root_path)
        print(f"\n📊 Diagnosis Status: {diagnosis['status'].upper()}")
        if diagnosis['suggestions']:
            print("💡 Suggestions:")
            for suggestion in diagnosis['suggestions']:
                print(f"  • {suggestion}")
    
    elif args.mode == 'count':
        file_counts = count_dataset_files(args.root_path, args.dataset)
        total_files = sum(file_counts.values())
        print(f"\n📊 Total files: {total_files}")
    
    elif args.mode == 'visualize':
        if args.vis_type == '2d':
            success = visualize_kitti_sample_2d(args.root_path, args.index, args.dataset, args.output_dir)
        elif args.vis_type == '3d':
            success = visualize_kitti_sample_3d(args.root_path, args.index, args.dataset, args.output_dir)
        elif args.vis_type == 'lidar':
            success = visualize_kitti_sample_lidar(args.root_path, args.index, args.dataset, args.output_dir)
        elif args.vis_type == 'bev':
            success = visualize_kitti_sample_bev(args.root_path, args.index, args.dataset, args.output_dir)
        else:
            # Default to original function
            success = visualize_kitti_sample(
                args.root_path, 
                args.index, 
                args.dataset,
                args.camera_count,
                args.jpgfile,
                args.point_cloud_range
            )
        if not success:
            print("❌ Visualization failed")
            return 1
    
    print("\n🎉 Operation completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())