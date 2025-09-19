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

# NuScenes dataset structure definition
NUSCENES_STRUCTURE = {
    'samples': ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP', 'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'],
    'sweeps': ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP', 'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'],
    'maps': [],
    'v1.0-trainval': []
}

REQUIRED_ANNOTATION_FILES = [
    'attribute.json',
    'calibrated_sensor.json', 
    'category.json',
    'ego_pose.json',
    'instance.json',
    'log.json',
    'map.json',
    'sample.json',
    'sample_annotation.json',
    'sample_data.json',
    'scene.json',
    'sensor.json',
    'visibility.json'
]

# Default paths
DEFAULT_DATA_ROOT = "/mnt/e/Shared/Dataset/"
DEFAULT_NUSCENES_DIR = os.path.join(DEFAULT_DATA_ROOT, "NuScenes", "v1.0-trainval")
DEFAULT_ZIP_DIR = "/mnt/e/Shared/Dataset/NuScenes/"

class BoxVisibility(IntEnum):
    """ Enumerates the various level of box visibility in an image """
    ALL = 0  # Requires all corners are inside the image.
    ANY = 1  # Requires at least one corner visible in the image.
    NONE = 2  # Requires no corners to be inside, i.e. box can be fully outside the image.

def box_in_image(corners_3d, corners_2d, intrinsic: np.ndarray, imsize: Tuple[int, int], vis_level: int = BoxVisibility.ANY) -> bool:
    """
    Check if a box is visible inside an image without accounting for occlusions.
    Based on NuScenes official devkit implementation.
    
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

def validate_dataset_structure(nuscenes_root: str) -> bool:
    """
    Validate the basic structure of the NuScenes dataset
    
    Args:
        nuscenes_root: Path to the NuScenes dataset root directory
        
    Returns:
        bool: True if structure is valid, False otherwise
    """
    print(f"\nValidating dataset structure at: {nuscenes_root}")
    
    if not os.path.exists(nuscenes_root):
        print(f"❌ Dataset root directory does not exist: {nuscenes_root}")
        return False
    
    # Check for required annotation files
    missing_files = []
    for file_name in REQUIRED_ANNOTATION_FILES:
        file_path = os.path.join(nuscenes_root, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing annotation files: {missing_files}")
        return False
    
    # Check for samples directory
    samples_dir = os.path.join(nuscenes_root, 'samples')
    if not os.path.exists(samples_dir):
        print(f"❌ Samples directory does not exist: {samples_dir}")
        return False
    
    print("✅ Dataset structure validation passed")
    return True

def extract_files(zip_dir: str, extract_dir: str) -> None:
    """
    Extract NuScenes dataset files from tgz archives.
    
    Args:
        zip_dir (str): Path to the directory containing the tgz files
        extract_dir (str): Path to the destination directory for extraction
    """
    # Make sure the destination directory exists
    os.makedirs(extract_dir, exist_ok=True)

    # Find all blob tgz files
    zip_files = glob.glob(os.path.join(zip_dir, "v1.0-trainval*_blobs.tgz"))
    
    # Also look for metadata files
    meta_files = glob.glob(os.path.join(zip_dir, "v1.0-*meta*.tgz"))

    print(f"Found {len(zip_files)} blob tgz files and {len(meta_files)} metadata files.")

    # Extract each file
    all_files = zip_files + meta_files
    for zip_path in all_files:
        print(f"Extracting {os.path.basename(zip_path)}...")
        with tarfile.open(zip_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)

    print("Done extracting all files.")
    print(f"Dataset extracted to: {extract_dir}")
    
    # Validate extraction
    validate_dataset_structure(str(extract_dir))

def check_extracted_folders(extract_dir: str) -> List[str]:
    """
    Check the structure of extracted folders
    
    Args:
        extract_dir: Directory where files were extracted
        
    Returns:
        List of subfolder names
    """
    print("\nChecking extracted folder structure:")
    if os.path.exists(extract_dir):
        print(f"Extraction directory {extract_dir} exists")
        subfolders = [f for f in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, f))]
        print(f"Contains {len(subfolders)} subfolders: {', '.join(subfolders)}")
        return subfolders
    else:
        print(f"Extraction directory {extract_dir} does not exist!")
        return []

def count_files_and_check_annotations(nuscenes_root: str) -> Tuple[Dict[str, int], Dict]:
    """
    Count image files and check annotation files following standard NuScenes structure
    
    Args:
        nuscenes_root: Root directory of the NuScenes dataset
        
    Returns:
        Tuple containing image count by camera type and annotations data
    """
    print("\nCounting image files and checking annotations:")
    
    # Standard NuScenes annotation directory
    annotation_dir = os.path.join(nuscenes_root, "v1.0-trainval")
    
    # Find image folder
    samples_dir = os.path.join(nuscenes_root, "samples")
    
    # Count images by sensor type
    image_count = defaultdict(int)
    total_images = 0
    
    if os.path.exists(samples_dir):
        for sensor_type in NUSCENES_STRUCTURE['samples']:
            sensor_dir = os.path.join(samples_dir, sensor_type)
            if os.path.exists(sensor_dir):
                files = [f for f in os.listdir(sensor_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pcd', '.pcd.bin'))]
                image_count[sensor_type] = len(files)
                total_images += len(files)
        
        print(f"Found samples directory: {samples_dir}")
        print(f"Total file count: {total_images}")
        for sensor_type, count in image_count.items():
            if count > 0:
                print(f"  - {sensor_type}: {count} files")
    else:
        print(f"Samples directory does not exist: {samples_dir}")
    
    # Check annotation files
    annotations = {}
    missing_annotations = []
    
    if os.path.exists(annotation_dir):
        print(f"\nFound annotation directory: {annotation_dir}")
        
        for file_name in REQUIRED_ANNOTATION_FILES:
            file_path = os.path.join(annotation_dir, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        annotations[file_name] = len(data)
                        print(f"  - {file_name}: {len(data)} records")
                    elif isinstance(data, dict):
                        annotations[file_name] = len(data)
                        print(f"  - {file_name}: {len(data)} entries")
                except json.JSONDecodeError:
                    print(f"  - {file_name}: JSON format error")
                    missing_annotations.append(file_name)
            else:
                missing_annotations.append(file_name)
        
        if missing_annotations:
            print(f"\nMissing annotation files: {missing_annotations}")
    else:
        print(f"Annotation directory does not exist: {annotation_dir}")
        missing_annotations = REQUIRED_ANNOTATION_FILES
    
    return image_count, annotations

def diagnose_dataset_issues(nuscenes_root: str) -> Dict[str, Any]:
    """
    Comprehensive dataset validation and issue diagnosis
    
    Args:
        nuscenes_root: Root directory of the NuScenes dataset
        
    Returns:
        Dictionary containing diagnosis results and suggestions
    """
    print("\n" + "="*60)
    print("NUSCENES DATASET DIAGNOSIS")
    print("="*60)
    
    diagnosis = {
        'structure_issues': [],
        'missing_files': [],
        'data_integrity': {},
        'suggestions': [],
        'status': 'unknown'
    }
    
    # Check basic directory structure
    print("\n1. Checking directory structure...")
    
    required_dirs = ['samples', 'sweeps', 'v1.0-trainval']
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = os.path.join(nuscenes_root, dir_name)
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_name}/ found")
        else:
            print(f"  ✗ {dir_name}/ missing")
            missing_dirs.append(dir_name)
            diagnosis['structure_issues'].append(f"Missing directory: {dir_name}")
    
    # Check samples subdirectories
    samples_dir = os.path.join(nuscenes_root, 'samples')
    if os.path.exists(samples_dir):
        print("\n2. Checking samples subdirectories...")
        for sensor_type in NUSCENES_STRUCTURE['samples']:
            sensor_dir = os.path.join(samples_dir, sensor_type)
            if os.path.exists(sensor_dir):
                files = [f for f in os.listdir(sensor_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pcd', '.pcd.bin'))]
                file_count = len(files)
                print(f"  ✓ {sensor_type}: {file_count} files")
                diagnosis['data_integrity'][sensor_type] = file_count
            else:
                print(f"  ✗ {sensor_type}: missing")
                diagnosis['structure_issues'].append(f"Missing sensor directory: {sensor_type}")
    
    # Check sweeps subdirectories
    sweeps_dir = os.path.join(nuscenes_root, 'sweeps')
    if os.path.exists(sweeps_dir):
        print("\n3. Checking sweeps subdirectories...")
        for sensor_type in NUSCENES_STRUCTURE['sweeps']:
            sensor_dir = os.path.join(sweeps_dir, sensor_type)
            if os.path.exists(sensor_dir):
                files = [f for f in os.listdir(sensor_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pcd', '.pcd.bin'))]
                file_count = len(files)
                print(f"  ✓ {sensor_type}: {file_count} files")
                diagnosis['data_integrity'][f"sweeps_{sensor_type}"] = file_count
            else:
                print(f"  ✗ {sensor_type}: missing")
                diagnosis['structure_issues'].append(f"Missing sweeps directory: {sensor_type}")
    
    # Check annotation files
    print("\n4. Checking annotation files...")
    annotation_dir = os.path.join(nuscenes_root, 'v1.0-trainval')
    
    if os.path.exists(annotation_dir):
        for file_name in REQUIRED_ANNOTATION_FILES:
            file_path = os.path.join(annotation_dir, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    record_count = len(data) if isinstance(data, (list, dict)) else 0
                    print(f"  ✓ {file_name}: {record_count} records")
                    diagnosis['data_integrity'][file_name] = record_count
                except json.JSONDecodeError:
                    print(f"  ✗ {file_name}: JSON format error")
                    diagnosis['missing_files'].append(f"{file_name} (corrupted)")
                except Exception as e:
                    print(f"  ✗ {file_name}: Error - {e}")
                    diagnosis['missing_files'].append(f"{file_name} (error)")
            else:
                print(f"  ✗ {file_name}: missing")
                diagnosis['missing_files'].append(file_name)
    else:
        print(f"  ✗ Annotation directory missing: {annotation_dir}")
        diagnosis['structure_issues'].append("Missing v1.0-trainval directory")
        diagnosis['missing_files'].extend(REQUIRED_ANNOTATION_FILES)
    
    # Check data consistency
    print("\n5. Checking data consistency...")
    
    # Check if we have both images and annotations
    has_images = any(count > 0 for key, count in diagnosis['data_integrity'].items() 
                    if 'CAM_' in key)
    
    # Check for LiDAR data in both samples and sweeps
    has_lidar = (diagnosis['data_integrity'].get('LIDAR_TOP', 0) > 0 or 
                diagnosis['data_integrity'].get('sweeps_LIDAR_TOP', 0) > 0)
    
    has_annotations = any(file_name in diagnosis['data_integrity'] 
                         for file_name in REQUIRED_ANNOTATION_FILES)
    
    if has_images:
        print("  ✓ Camera images found")
    else:
        print("  ✗ No camera images found")
        diagnosis['structure_issues'].append("No camera images")
    
    if has_lidar:
        lidar_samples = diagnosis['data_integrity'].get('LIDAR_TOP', 0)
        lidar_sweeps = diagnosis['data_integrity'].get('sweeps_LIDAR_TOP', 0)
        print(f"  ✓ LiDAR data found (samples: {lidar_samples}, sweeps: {lidar_sweeps})")
    else:
        print("  ✗ No LiDAR data found")
        diagnosis['structure_issues'].append("No LiDAR data")
    
    if has_annotations:
        print("  ✓ Annotation files found")
    else:
        print("  ✗ No annotation files found")
        diagnosis['structure_issues'].append("No annotation files")
    
    # Generate suggestions
    print("\n6. Generating suggestions...")
    
    if missing_dirs:
        if 'samples' in missing_dirs:
            diagnosis['suggestions'].append(
                "Missing 'samples' directory - ensure you've extracted the main dataset files"
            )
        if 'sweeps' in missing_dirs:
            diagnosis['suggestions'].append(
                "Missing 'sweeps' directory - you may need to download the sweep data separately"
            )
        if 'v1.0-trainval' in missing_dirs:
            diagnosis['suggestions'].append(
                "Missing 'v1.0-trainval' directory - ensure you've downloaded the metadata/annotations"
            )
    
    if diagnosis['missing_files']:
        diagnosis['suggestions'].append(
            f"Missing annotation files: {', '.join(diagnosis['missing_files'])} - "
            "download the v1.0-trainval metadata package"
        )
    
    if not has_images and not has_lidar:
        diagnosis['suggestions'].append(
            "No sensor data found - check if the dataset was properly extracted to the correct location"
        )
    
    # Determine overall status
    if not diagnosis['structure_issues'] and not diagnosis['missing_files']:
        diagnosis['status'] = 'healthy'
        print("\n  ✓ Dataset appears to be complete and properly structured")
    elif diagnosis['structure_issues'] or len(diagnosis['missing_files']) > 3:
        diagnosis['status'] = 'critical'
        print("\n  ✗ Dataset has critical issues that need to be resolved")
    else:
        diagnosis['status'] = 'warning'
        print("\n  ⚠ Dataset has some issues but may be partially usable")
    
    # Print summary
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    print(f"Status: {diagnosis['status'].upper()}")
    
    if diagnosis['suggestions']:
        print("\nSuggestions:")
        for i, suggestion in enumerate(diagnosis['suggestions'], 1):
            print(f"{i}. {suggestion}")
    
    return diagnosis

def check_nuscenes_data_structure(nuscenes_root: str) -> bool:
    """
    Verify NuScenes dataset structure and data integrity
    
    Args:
        nuscenes_root: Root directory of the NuScenes dataset
        
    Returns:
        True if structure is valid, False otherwise
    """
    print("\n" + "="*60)
    print("NUSCENES DATA STRUCTURE VERIFICATION")
    print("="*60)
    
    # Check if root directory exists
    if not os.path.exists(nuscenes_root):
        print(f"✗ Root directory does not exist: {nuscenes_root}")
        return False
    
    print(f"Checking NuScenes dataset at: {nuscenes_root}")
    
    # Verify directory structure
    structure_valid = True
    
    # Check main directories
    main_dirs = ['samples', 'sweeps', 'v1.0-trainval']
    for dir_name in main_dirs:
        dir_path = os.path.join(nuscenes_root, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ {dir_name}/ directory found")
        else:
            print(f"✗ {dir_name}/ directory missing")
            structure_valid = False
    
    # Check samples structure
    samples_dir = os.path.join(nuscenes_root, 'samples')
    if os.path.exists(samples_dir):
        print("\nChecking samples structure:")
        for sensor_type in NUSCENES_STRUCTURE['samples']:
            sensor_dir = os.path.join(samples_dir, sensor_type)
            if os.path.exists(sensor_dir):
                file_count = len([f for f in os.listdir(sensor_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pcd', '.pcd.bin'))])
                print(f"  ✓ {sensor_type}: {file_count} files")
            else:
                print(f"  ✗ {sensor_type}: missing")
                structure_valid = False
    
    # Check sweeps structure
    sweeps_dir = os.path.join(nuscenes_root, 'sweeps')
    if os.path.exists(sweeps_dir):
        print("\nChecking sweeps structure:")
        for sensor_type in NUSCENES_STRUCTURE['sweeps']:
            sensor_dir = os.path.join(sweeps_dir, sensor_type)
            if os.path.exists(sensor_dir):
                file_count = len([f for f in os.listdir(sensor_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pcd', '.pcd.bin'))])
                print(f"  ✓ {sensor_type}: {file_count} files")
            else:
                print(f"  ✗ {sensor_type}: missing")
                # Sweeps are optional for some use cases
                print(f"    (Note: sweeps are optional for basic usage)")
    
    # Check annotation files
    annotation_dir = os.path.join(nuscenes_root, 'v1.0-trainval')
    if os.path.exists(annotation_dir):
        print("\nChecking annotation files:")
        for file_name in REQUIRED_ANNOTATION_FILES:
            file_path = os.path.join(annotation_dir, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    record_count = len(data) if isinstance(data, (list, dict)) else 0
                    print(f"  ✓ {file_name}: {record_count} records")
                except json.JSONDecodeError:
                    print(f"  ✗ {file_name}: JSON format error")
                    structure_valid = False
                except Exception as e:
                    print(f"  ✗ {file_name}: Error reading file - {e}")
                    structure_valid = False
            else:
                print(f"  ✗ {file_name}: missing")
                structure_valid = False
    
    # Verify data consistency
    print("\nVerifying data consistency:")
    
    # Check if annotation files reference existing sensor data
    if structure_valid:
        try:
            # Load sample and sample_data files
            sample_file = os.path.join(annotation_dir, 'sample.json')
            sample_data_file = os.path.join(annotation_dir, 'sample_data.json')
            
            with open(sample_file, 'r') as f:
                samples = json.load(f)
            with open(sample_data_file, 'r') as f:
                sample_data = json.load(f)
            
            # Check a few samples to verify file references
            missing_files = 0
            checked_files = 0
            
            # Check first 10 sample_data entries directly
            for sd in sample_data[:min(10, len(sample_data))]:
                file_path = os.path.join(nuscenes_root, sd['filename'])
                checked_files += 1
                if not os.path.exists(file_path):
                    missing_files += 1
            
            if missing_files == 0:
                print(f"  ✓ All checked sensor files exist ({checked_files} files)")
            else:
                print(f"  ⚠ {missing_files}/{checked_files} sensor files missing")
                if missing_files > checked_files * 0.1:  # More than 10% missing
                    structure_valid = False
        
        except Exception as e:
            print(f"  ✗ Error verifying data consistency: {e}")
            structure_valid = False
    
    # Final result
    print("\n" + "="*60)
    if structure_valid:
        print("✓ DATASET STRUCTURE IS VALID")
        print("The NuScenes dataset appears to be properly structured and complete.")
    else:
        print("✗ DATASET STRUCTURE HAS ISSUES")
        print("The dataset structure is incomplete or has errors.")
    return structure_valid


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper function that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis and the third axis is the depth.
    
    Based on NuScenes official implementation.
    
    Args:
        points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
        view: <np.float32: n, n> Defines an arbitrary projection (n <= 4).
        normalize: Whether to normalize the remaining coordinate (along the depth axis).
    
    Returns:
        <np.float32: n, n> Mapped points. If normalize=False, the points are not normalized.
    """
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation = None,
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    Based on NuScenes official implementation.
    
    Args:
        translation: <np.float32: 3>. Translation in x, y, z.
        rotation: Rotation in quaternions (w ri rj rk) or Quaternion object.
        inverse: Whether to compute inverse transform matrix.
    
    Returns:
        <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if rotation is None:
        rotation_matrix = np.eye(3)
    else:
        if hasattr(rotation, 'rotation_matrix'):
            # Quaternion object
            rotation_matrix = rotation.rotation_matrix
        else:
            # Convert quaternion array to rotation matrix
            rotation_matrix = quaternion_to_rotation_matrix(rotation)

    if inverse:
        rot_inv = rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: quaternion [w, x, y, z]
    
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def get_3d_box_corners(center, size, rotation):
    """
    Get 8 corners of a 3D bounding box following NuScenes conventions.
    
    NuScenes coordinate system:
    - x points forward (length direction)
    - y points to the left (width direction) 
    - z points up (height direction)
    
    Args:
        center: [x, y, z] center of the box
        size: [width, length, height] dimensions of the box (NuScenes format)
        rotation: quaternion [w, x, y, z] rotation of the box
    
    Returns:
        8x3 array of corner coordinates in NuScenes standard order
    """
    w, l, h = size
    
    # Define box corners in local coordinate system (centered at origin)
    # Following NuScenes convention: x=forward(length), y=left(width), z=up(height)
    # Reorder to match NuScenes standard: [length, width, height] -> [x, y, z]
    corners = np.array([
        # Bottom face (z = -h/2)
        [-l/2, -w/2, -h/2],  # 0: back-right
        [-l/2, +w/2, -h/2],  # 1: back-left  
        [+l/2, +w/2, -h/2],  # 2: front-left
        [+l/2, -w/2, -h/2],  # 3: front-right
        # Top face (z = +h/2)
        [-l/2, -w/2, +h/2],  # 4: back-right-top
        [-l/2, +w/2, +h/2],  # 5: back-left-top
        [+l/2, +w/2, +h/2],  # 6: front-left-top
        [+l/2, -w/2, +h/2]   # 7: front-right-top
    ])
    
    # Apply rotation using quaternion
    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    corners_rotated = np.dot(corners, rotation_matrix.T)
    
    # Translate to center position
    corners_world = corners_rotated + center
    
    return corners_world


def project_3d_box_to_2d(center_3d, size_3d, rotation_3d, cam_translation, cam_rotation, camera_intrinsic, ego_translation, ego_rotation, debug=False):
    """
    Project 3D bounding box to 2D image coordinates using NuScenes coordinate system conventions.
    Based on official NuScenes devkit implementation.
    
    Args:
        center_3d: [x, y, z] center of 3D box in global coordinates
        size_3d: [width, length, height] dimensions of 3D box
        rotation_3d: quaternion [w, x, y, z] rotation of 3D box
        cam_translation: [x, y, z] camera translation relative to ego vehicle
        cam_rotation: quaternion [w, x, y, z] camera rotation relative to ego vehicle
        camera_intrinsic: 3x3 camera intrinsic matrix
        ego_translation: [x, y, z] ego vehicle translation in global coordinates
        ego_rotation: quaternion [w, x, y, z] ego vehicle rotation in global coordinates
        debug: whether to print debug information
    
    Returns:
        8x2 array of 2D corner coordinates, or None if projection fails
    """
    try:
        if debug:
            print(f"DEBUG: Box center_3d: {center_3d}")
            print(f"DEBUG: Box size_3d: {size_3d}")
            print(f"DEBUG: Box rotation_3d: {rotation_3d}")
            print(f"DEBUG: Ego translation: {ego_translation}")
            print(f"DEBUG: Ego rotation: {ego_rotation}")
            print(f"DEBUG: Cam translation: {cam_translation}")
            print(f"DEBUG: Cam rotation: {cam_rotation}")
        
        # Get 3D box corners in global coordinates
        corners_3d_global = get_3d_box_corners(center_3d, size_3d, rotation_3d)
        
        if debug:
            print(f"DEBUG: Corners in global coordinates:")
            for i, corner in enumerate(corners_3d_global):
                print(f"  Corner {i}: {corner}")
        
        # Step 1: Transform from global to ego vehicle coordinate system
        # Use NuScenes transform_matrix function
        global_to_ego = transform_matrix(ego_translation, ego_rotation, inverse=True)
        corners_3d_global_homogeneous = np.ones((corners_3d_global.shape[0], 4))
        corners_3d_global_homogeneous[:, :3] = corners_3d_global
        corners_ego_homogeneous = np.dot(global_to_ego, corners_3d_global_homogeneous.T)
        corners_ego = corners_ego_homogeneous[:3, :].T
        
        if debug:
            print(f"DEBUG: Global to ego transform matrix:")
            print(global_to_ego)
            print(f"DEBUG: Corners in ego coordinates:")
            for i, corner in enumerate(corners_ego):
                print(f"  Corner {i}: {corner}")
        
        # Step 2: Transform from ego to camera coordinate system
        # Use NuScenes transform_matrix function
        ego_to_cam = transform_matrix(cam_translation, cam_rotation, inverse=True)
        corners_ego_homogeneous = np.ones((corners_ego.shape[0], 4))
        corners_ego_homogeneous[:, :3] = corners_ego
        corners_cam_homogeneous = np.dot(ego_to_cam, corners_ego_homogeneous.T)
        corners_cam = corners_cam_homogeneous[:3, :].T
        
        if debug:
            print(f"DEBUG: Ego to cam transform matrix:")
            print(ego_to_cam)
            print(f"DEBUG: Corners in camera coordinates:")
            for i, corner in enumerate(corners_cam):
                print(f"  Corner {i}: {corner}")
        
        # Step 3: Project to image coordinates using NuScenes view_points function
        # Check if any points are behind the camera (z <= 0)
        depths = corners_cam[:, 2]
        if np.any(depths <= 0):
            if debug:
                print(f"DEBUG: Some points behind camera, depths: {depths}")
                behind_camera = depths <= 0
                print(f"DEBUG: Points behind camera: {np.where(behind_camera)[0]}")
            # Still continue with projection but mark as potentially invalid
        
        # Use NuScenes view_points function for projection
        # Convert corners to 3xN format (required by view_points)
        corners_cam_3xN = corners_cam.T  # Shape: (3, N)
        
        # Project using view_points function
        corners_2d_3xN = view_points(corners_cam_3xN, camera_intrinsic, normalize=True)
        
        # Convert back to Nx2 format
        corners_2d = corners_2d_3xN[:2, :].T  # Shape: (N, 2)
        
        if debug:
            print(f"DEBUG: Final 2D corners:")
            for i, corner in enumerate(corners_2d):
                print(f"  Corner {i}: {corner}")
        
        # Return both 2D corners and 3D corners in camera coordinates for visibility check
        return corners_2d, corners_cam_3xN  # Return as Nx2 array and 3xN array
        
    except Exception as e:
        print(f"Projection error: {e}")
        return None, None


def get_2d_bbox_from_3d_projection(corners_2d):
    """
    Get 2D bounding box from 3D projection corners.
    
    Args:
        corners_2d: 8x2 array of 2D corner coordinates from 3D projection
        
    Returns:
        [x_min, y_min, x_max, y_max] or None if invalid
    """
    if corners_2d is None or len(corners_2d) != 8:
        return None
    
    # Filter out invalid coordinates (negative depth projections)
    valid_corners = []
    for corner in corners_2d:
        if not (np.isnan(corner[0]) or np.isnan(corner[1]) or 
                np.isinf(corner[0]) or np.isinf(corner[1])):
            valid_corners.append(corner)
    
    if len(valid_corners) < 4:  # Need at least 4 valid corners
        return None
    
    valid_corners = np.array(valid_corners)
    x_min = np.min(valid_corners[:, 0])
    y_min = np.min(valid_corners[:, 1])
    x_max = np.max(valid_corners[:, 0])
    y_max = np.max(valid_corners[:, 1])
    
    # Check if bounding box is reasonable
    if x_max <= x_min or y_max <= y_min:
        return None
    
    return [x_min, y_min, x_max, y_max]
    
    x_coords = corners_2d[:, 0]
    y_coords = corners_2d[:, 1]
    
    # Filter out invalid coordinates (e.g., behind camera)
    valid_mask = np.isfinite(x_coords) & np.isfinite(y_coords)
    if not np.any(valid_mask):
        return None
    
    x_min = np.min(x_coords[valid_mask])
    x_max = np.max(x_coords[valid_mask])
    y_min = np.min(y_coords[valid_mask])
    y_max = np.max(y_coords[valid_mask])
    
    return [x_min, y_min, x_max, y_max]


def draw_2d_bbox(ax, bbox_2d, category_name="", color='green', linewidth=2, img_width=1600, img_height=900):
    """
    Draw 2D bounding box on image with enhanced visualization and clipping.
    
    Args:
        ax: matplotlib axis object
        bbox_2d: [x_min, y_min, x_max, y_max] bounding box coordinates
        category_name: object category name for label
        color: box color
        linewidth: line width for the box
        img_width: image width for clipping
        img_height: image height for clipping
    """
    if bbox_2d is None:
        return False
    
    x_min, y_min, x_max, y_max = bbox_2d
    
    # Clip bounding box to image boundaries
    x_min = max(0, min(x_min, img_width))
    y_min = max(0, min(y_min, img_height))
    x_max = max(0, min(x_max, img_width))
    y_max = max(0, min(y_max, img_height))
    
    # Check if clipped box is still valid
    if x_max <= x_min or y_max <= y_min:
        return False
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Draw rectangle with enhanced style
    import matplotlib.patches as patches
    rect = patches.Rectangle((x_min, y_min), width, height, 
                           linewidth=linewidth, edgecolor=color, facecolor='none',
                           linestyle='-', alpha=0.8)
    ax.add_patch(rect)
    
    # Add corner markers for better visibility
    corner_size = 8
    corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    for corner_x, corner_y in corners:
        ax.plot(corner_x, corner_y, 's', color=color, markersize=corner_size//2, alpha=0.9)
    
    # Add label if provided with improved styling
    if category_name:
        # Position label above the box, or inside if there's no space above
        label_y = y_min - 8 if y_min > 25 else y_min + 15
        label_x = x_min
        
        # Ensure label is within image bounds
        if label_y < 0:
            label_y = y_min + 15
        if label_x < 0:
            label_x = 5
        
        # Create text with background
        text = ax.text(label_x, label_y, category_name, 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8, edgecolor='white'),
                      fontsize=10, color='white', fontweight='bold',
                      verticalalignment='bottom')
        
        # Add confidence score placeholder (can be extended later)
        # conf_text = f"{category_name} (1.0)"
        # text.set_text(conf_text)
    
    return True


def draw_2d_bbox_from_3d(ax, corners_2d, category_name="", color='blue', linewidth=2, img_width=1600, img_height=900):
    """
    Draw 2D bounding box derived from 3D projection with distinct styling.
    
    Args:
        ax: matplotlib axis object
        corners_2d: 8x2 array of 2D corner coordinates from 3D projection
        category_name: object category name for label
        color: box color
        linewidth: line width for the box
        img_width: image width for clipping
        img_height: image height for clipping
    """
    if corners_2d is None or len(corners_2d) != 8:
        return False
    
    # Get 2D bounding box from 3D projection
    bbox_2d = get_2d_bbox_from_3d_projection(corners_2d)
    if bbox_2d is None:
        return False
    
    x_min, y_min, x_max, y_max = bbox_2d
    
    # Clip to image boundaries
    x_min = max(0, min(x_min, img_width))
    y_min = max(0, min(y_min, img_height))
    x_max = max(0, min(x_max, img_width))
    y_max = max(0, min(y_max, img_height))
    
    # Check if clipped box is still valid
    if x_max <= x_min or y_max <= y_min:
        return False
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Draw dashed rectangle to distinguish from regular 2D bbox
    import matplotlib.patches as patches
    rect = patches.Rectangle((x_min, y_min), width, height, 
                           linewidth=linewidth, edgecolor=color, facecolor='none',
                           linestyle='--', alpha=0.7)
    ax.add_patch(rect)
    
    # Add label with different styling
    if category_name:
        label_y = y_max + 5 if y_max < img_height - 25 else y_max - 15
        label_x = x_min
        
        # Ensure label is within image bounds
        if label_y > img_height:
            label_y = y_max - 15
        if label_x < 0:
            label_x = 5
        
        ax.text(label_x, label_y, f"{category_name} (2D)", 
               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.6, edgecolor='white'),
               fontsize=8, color='white', fontweight='normal',
               verticalalignment='top')
    
    return True


def clip_line_to_image(start_point, end_point, img_width, img_height):
    """
    Clip a line segment to image boundaries using Cohen-Sutherland algorithm.
    
    Args:
        start_point: [x, y] start coordinates
        end_point: [x, y] end coordinates  
        img_width: image width
        img_height: image height
        
    Returns:
        Tuple of (clipped_start, clipped_end) or None if line is completely outside
    """
    # Define region codes
    INSIDE = 0  # 0000
    LEFT = 1    # 0001
    RIGHT = 2   # 0010
    BOTTOM = 4  # 0100
    TOP = 8     # 1000
    
    def compute_code(x, y):
        code = INSIDE
        if x < 0:
            code |= LEFT
        elif x > img_width:
            code |= RIGHT
        if y < 0:
            code |= BOTTOM
        elif y > img_height:
            code |= TOP
        return code
    
    x1, y1 = start_point
    x2, y2 = end_point
    
    code1 = compute_code(x1, y1)
    code2 = compute_code(x2, y2)
    
    while True:
        # Both endpoints inside
        if code1 == 0 and code2 == 0:
            return (x1, y1), (x2, y2)
        
        # Both endpoints outside same region
        if code1 & code2:
            return None
        
        # At least one endpoint outside
        code_out = code1 if code1 != 0 else code2
        
        # Find intersection point
        if code_out & TOP:
            x = x1 + (x2 - x1) * (img_height - y1) / (y2 - y1)
            y = img_height
        elif code_out & BOTTOM:
            x = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
            y = 0
        elif code_out & RIGHT:
            y = y1 + (y2 - y1) * (img_width - x1) / (x2 - x1)
            x = img_width
        elif code_out & LEFT:
            y = y1 + (y2 - y1) * (0 - x1) / (x2 - x1)
            x = 0
        
        # Update point and code
        if code_out == code1:
            x1, y1 = x, y
            code1 = compute_code(x1, y1)
        else:
            x2, y2 = x, y
            code2 = compute_code(x2, y2)


def draw_3d_box_2d(ax, corners_2d, category_name="", color='red', linewidth=2, img_width=1600, img_height=900):
    """
    Draw 3D bounding box on 2D image using projected corners with clipping to image boundaries.
    
    Args:
        ax: matplotlib axis object
        corners_2d: 8x2 array of 2D corner coordinates
        category_name: category name to display above the box
        color: line color for the 3D box
        linewidth: line width
        img_width: image width for clipping (default 1600)
        img_height: image height for clipping (default 900)
    """
    if corners_2d is None or len(corners_2d) != 8:
        return
    
    # Define the 12 edges of a 3D box (connecting corner indices)
    # Bottom face (z=0): corners 0,1,2,3
    # Top face (z=1): corners 4,5,6,7
    edges = [
        # Bottom face edges
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face edges  
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges connecting bottom to top
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    # Count how many edges are drawn
    edges_drawn = 0
    
    # Draw each edge with clipping
    for edge in edges:
        start_idx, end_idx = edge
        start_point = corners_2d[start_idx]
        end_point = corners_2d[end_idx]
        
        # Check if both points are valid (not NaN or Inf)
        if (np.isfinite(start_point).all() and np.isfinite(end_point).all()):
            # Clip line to image boundaries
            clipped_line = clip_line_to_image(start_point, end_point, img_width, img_height)
            
            if clipped_line is not None:
                clipped_start, clipped_end = clipped_line
                ax.plot([clipped_start[0], clipped_end[0]], 
                       [clipped_start[1], clipped_end[1]], 
                       color=color, linewidth=linewidth, alpha=0.8)
                edges_drawn += 1
    
    # Draw center point for reference if it's within image bounds
    center_2d = np.mean(corners_2d, axis=0)
    if (np.isfinite(center_2d).all() and 
        0 <= center_2d[0] <= img_width and 
        0 <= center_2d[1] <= img_height):
        ax.plot(center_2d[0], center_2d[1], 'o', color=color, markersize=4, alpha=0.8)
    
    # Add category label above the 3D box if provided
    if category_name and edges_drawn > 0:
        # Find the topmost point of the 3D box for label placement
        valid_corners = corners_2d[np.isfinite(corners_2d).all(axis=1)]
        if len(valid_corners) > 0:
            # Get the highest point (smallest y coordinate in image space)
            top_y = np.min(valid_corners[:, 1])
            center_x = np.mean(valid_corners[:, 0])
            
            # Place label above the box with some padding
            label_y = max(10, top_y - 15)  # Ensure label is not too close to top edge
            label_x = np.clip(center_x, 50, img_width - 50)  # Keep label within image bounds
            
            # Add text with background for better visibility
            ax.text(label_x, label_y, category_name, 
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                   ha='center', va='bottom')
    
    return edges_drawn


def load_lidar_points(lidar_path: str) -> np.ndarray:
    """
    Load LiDAR point cloud from .pcd.bin file (NuScenes format).
    
    Args:
        lidar_path: Path to the .pcd.bin file
        
    Returns:
        np.ndarray: Point cloud data with shape (N, 4) where columns are [x, y, z, intensity]
    """
    if not os.path.exists(lidar_path):
        print(f"Warning: LiDAR file not found: {lidar_path}")
        return np.array([]).reshape(0, 4)
    
    try:
        # NuScenes .pcd.bin files are binary files with float32 values
        # Each point has 5 values: x, y, z, intensity, ring_index
        # We'll use the first 4 values: x, y, z, intensity
        points = np.fromfile(lidar_path, dtype=np.float32)
        
        # Reshape to (N, 5) - each point has 5 values
        if len(points) % 5 != 0:
            print(f"Warning: LiDAR file {lidar_path} has unexpected format")
            return np.array([]).reshape(0, 4)
            
        points = points.reshape(-1, 5)
        
        # Return only x, y, z, intensity (first 4 columns)
        return points[:, :4]
        
    except Exception as e:
        print(f"Error loading LiDAR file {lidar_path}: {e}")
        return np.array([]).reshape(0, 4)


def visualize_lidar_3d_open3d(points: np.ndarray, annotations: List[Dict], categories: List[Dict], 
                             instances: List[Dict], output_path: str = None, ego_translation: np.ndarray = None, 
                             ego_rotation: np.ndarray = None) -> None:
    """
    Create interactive 3D visualization of LiDAR point cloud with 3D bounding boxes using Open3D.
    
    Args:
        points: LiDAR point cloud data (N, 4) [x, y, z, intensity]
        annotations: List of sample annotations
        categories: List of category definitions
        instances: List of instance definitions
        output_path: Path to save the visualization (optional)
        ego_translation: Ego vehicle translation for coordinate transformation
        ego_rotation: Ego vehicle rotation for coordinate transformation
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Error: Open3D not installed. Please install with: pip install open3d")
        return
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    
    if len(points) > 0:
        # Use only x, y, z coordinates
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Color points by intensity (normalize to 0-1 range)
        if points.shape[1] > 3:
            intensities = points[:, 3]
            # Normalize intensities to 0-1 range
            intensities_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-8)
            # Create colormap (blue to red based on intensity)
            colors = np.zeros((len(points), 3))
            colors[:, 0] = intensities_norm  # Red channel
            colors[:, 2] = 1 - intensities_norm  # Blue channel (inverse)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # Default gray color if no intensity
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    # Create list to store all geometries
    geometries = [pcd]
    
    # Define category colors
    category_colors = {
        'car': [1.0, 0.0, 0.0],           # Red
        'truck': [0.0, 1.0, 0.0],         # Green  
        'bus': [0.0, 0.0, 1.0],           # Blue
        'trailer': [1.0, 1.0, 0.0],       # Yellow
        'construction_vehicle': [1.0, 0.5, 0.0],  # Orange
        'pedestrian': [1.0, 0.0, 1.0],    # Magenta
        'motorcycle': [0.5, 1.0, 0.0],    # Light Green
        'bicycle': [0.0, 1.0, 1.0],       # Cyan
        'traffic_cone': [1.0, 1.0, 0.5],  # Light Yellow
        'barrier': [0.5, 0.0, 1.0]        # Purple
    }
    
    # Draw 3D bounding boxes
    boxes_drawn = 0
    for ann in annotations:
        # Get category name and color
        category_name = "Unknown"
        color = [0.7, 0.7, 0.7]  # Default gray
        
        if ann.get('instance_token'):
            instance = next((inst for inst in instances if inst['token'] == ann['instance_token']), None)
            if instance and instance.get('category_token'):
                category = next((cat for cat in categories if cat['token'] == instance['category_token']), None)
                if category:
                    category_name = category['name']
                    color = category_colors.get(category_name.lower(), [0.7, 0.7, 0.7])
        
        # Get 3D box parameters
        center = np.array(ann['translation'])
        size = np.array(ann['size'])
        rotation = np.array(ann['rotation'])  # quaternion [w, x, y, z]
        
        # Apply coordinate transformation if ego pose is provided
        if ego_translation is not None and ego_rotation is not None:
            # Create transformation matrix from global to ego coordinates
            ego_rot_matrix = quaternion_to_rotation_matrix(ego_rotation)
            
            # Transform center from global to ego coordinates
            center_ego = ego_rot_matrix.T @ (center - ego_translation)
            center = center_ego
            
            # Transform rotation (this is simplified - for full accuracy, quaternion composition should be used)
            # For now, we'll keep the original rotation as it's relative to the object
        
        # Create Open3D oriented bounding box
        # Convert quaternion to rotation matrix
        rot_matrix = quaternion_to_rotation_matrix(rotation)
        
        # Create oriented bounding box
        obb = o3d.geometry.OrientedBoundingBox()
        obb.center = center
        obb.extent = size
        obb.R = rot_matrix
        obb.color = color
        
        # Create wireframe lines for the bounding box
        bbox_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        bbox_lines.paint_uniform_color(color)
        
        geometries.append(bbox_lines)
        boxes_drawn += 1
    
    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    print(f"Created Open3D visualization with {len(points)} points and {boxes_drawn} bounding boxes")
    
    # Try interactive visualization first, fallback to headless mode
    try:
        # Check if we can create a window (not in headless environment)
        test_vis = o3d.visualization.Visualizer()
        can_create_window = test_vis.create_window(window_name="Test", width=100, height=100, visible=False)
        test_vis.destroy_window()
        
        if can_create_window:
            # Interactive visualization
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"LiDAR Point Cloud with 3D Bounding Boxes ({boxes_drawn} boxes)",
                width=1200,
                height=800,
                left=50,
                top=50
            )
        else:
            print("  Interactive visualization not available (headless environment)")
    except Exception as e:
        print(f"  Interactive visualization not available: {e}")
    
    # Save screenshot if output path is provided
    if output_path:
        try:
            # Create a visualizer for saving
            vis = o3d.visualization.Visualizer()
            window_created = vis.create_window(window_name="LiDAR 3D Visualization", width=1200, height=800, visible=False)
            
            if window_created:
                for geom in geometries:
                    vis.add_geometry(geom)
                
                # Set view parameters for better visualization
                ctr = vis.get_view_control()
                if ctr is not None:  # Check if view control is available
                    ctr.set_front([0.0, 0.0, -1.0])
                    ctr.set_lookat([0.0, 0.0, 0.0])
                    ctr.set_up([0.0, -1.0, 0.0])
                    ctr.set_zoom(0.3)
                
                vis.capture_screen_image(output_path)
                vis.destroy_window()
                print(f"  Open3D 3D visualization saved to: {output_path}")
            else:
                print(f"  Warning: Cannot create window for screenshot in headless environment")
                # Alternative: save point cloud data as PLY file
                ply_path = output_path.replace('.png', '.ply').replace('.jpg', '.ply')
                if len(geometries) > 0 and hasattr(geometries[0], 'points'):
                    o3d.io.write_point_cloud(ply_path, geometries[0])
                    print(f"  Point cloud saved as PLY file: {ply_path}")
        except Exception as e:
            print(f"  Warning: Failed to save Open3D screenshot: {e}")
            print(f"  This is normal in headless environments.")


def visualize_lidar_3d(points: np.ndarray, annotations: List[Dict], categories: List[Dict], 
                      instances: List[Dict], output_path: str = None) -> None:
    """
    Create 3D visualization of LiDAR point cloud with 3D bounding boxes.
    
    Args:
        points: LiDAR point cloud data (N, 4) [x, y, z, intensity]
        annotations: List of sample annotations
        categories: List of category definitions
        instances: List of instance definitions
        output_path: Path to save the visualization
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Warning: 3D plotting not available. Install matplotlib with 3D support.")
        return
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(points) > 0:
        # Subsample points for better performance (keep every 10th point)
        step = max(1, len(points) // 10000)
        sampled_points = points[::step]
        
        # Color points by intensity
        intensities = sampled_points[:, 3]
        scatter = ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], 
                           c=intensities, cmap='viridis', s=0.5, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Intensity', shrink=0.8)
    
    # Draw 3D bounding boxes
    boxes_drawn = 0
    for ann in annotations:
        # Get category name
        category_name = "Unknown"
        if ann.get('instance_token'):
            instance = next((inst for inst in instances if inst['token'] == ann['instance_token']), None)
            if instance and instance.get('category_token'):
                category = next((cat for cat in categories if cat['token'] == instance['category_token']), None)
                if category:
                    category_name = category['name']
        
        # Get 3D box parameters
        center = np.array(ann['translation'])
        size = np.array(ann['size'])
        rotation = np.array(ann['rotation'])  # quaternion [w, x, y, z]
        
        # Get 3D box corners
        corners_3d = get_3d_box_corners(center, size, rotation)
        
        # Draw 3D wireframe box
        draw_3d_wireframe(ax, corners_3d, category_name, color='red')
        boxes_drawn += 1
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'LiDAR Point Cloud with 3D Bounding Boxes ({boxes_drawn} boxes)')
    
    # Set equal aspect ratio
    max_range = 50  # meters
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-5, 10])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  3D LiDAR visualization saved to: {output_path}")
    
    plt.close()


def draw_3d_wireframe(ax, corners_3d: np.ndarray, category_name: str = "", color: str = 'red'):
    """
    Draw 3D wireframe bounding box in 3D plot.
    
    Args:
        ax: 3D matplotlib axis
        corners_3d: 3D box corners (8, 3)
        category_name: Object category name
        color: Line color
    """
    # Define the 12 edges of a 3D box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    # Draw edges
    for edge in edges:
        start, end = edge
        ax.plot3D([corners_3d[start, 0], corners_3d[end, 0]],
                  [corners_3d[start, 1], corners_3d[end, 1]],
                  [corners_3d[start, 2], corners_3d[end, 2]], 
                  color=color, linewidth=2)
    
    # Add category label at box center
    if category_name:
        center = np.mean(corners_3d, axis=0)
        ax.text(center[0], center[1], center[2] + 1, category_name, 
                fontsize=8, color=color, weight='bold')


def project_lidar_to_camera(points: np.ndarray, cam_translation: np.ndarray, 
                           cam_rotation: np.ndarray, camera_intrinsic: np.ndarray,
                           ego_translation: np.ndarray, ego_rotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project LiDAR points to camera image coordinates.
    
    Args:
        points: LiDAR points (N, 4) [x, y, z, intensity]
        cam_translation: Camera translation in ego vehicle frame
        cam_rotation: Camera rotation quaternion [w, x, y, z]
        camera_intrinsic: Camera intrinsic matrix (3, 3)
        ego_translation: Ego vehicle translation in global frame
        ego_rotation: Ego vehicle rotation quaternion [w, x, y, z]
        
    Returns:
        Tuple of (projected_points_2d, depths) where:
        - projected_points_2d: (N, 2) image coordinates
        - depths: (N,) depth values for coloring
    """
    if len(points) == 0:
        return np.array([]).reshape(0, 2), np.array([])
    
    # Transform points from LiDAR to ego vehicle frame (LiDAR is already in ego frame for NuScenes)
    points_ego = points[:, :3]  # Use only x, y, z
    
    # Transform from ego vehicle frame to global frame
    ego_rot_matrix = quaternion_to_rotation_matrix(ego_rotation)
    ego_transform = transform_matrix(ego_translation, ego_rot_matrix)
    
    # Add homogeneous coordinate
    points_ego_homo = np.hstack([points_ego, np.ones((len(points_ego), 1))])
    points_global = (ego_transform @ points_ego_homo.T).T[:, :3]
    
    # Transform from global frame to camera frame
    cam_rot_matrix = quaternion_to_rotation_matrix(cam_rotation)
    cam_transform = transform_matrix(cam_translation, cam_rot_matrix, inverse=True)
    
    # Add homogeneous coordinate
    points_global_homo = np.hstack([points_global, np.ones((len(points_global), 1))])
    points_cam = (cam_transform @ points_global_homo.T).T[:, :3]
    
    # Filter points behind camera
    valid_mask = points_cam[:, 2] > 0.1  # At least 10cm in front
    points_cam_valid = points_cam[valid_mask]
    
    if len(points_cam_valid) == 0:
        return np.array([]).reshape(0, 2), np.array([])
    
    # Project to image coordinates
    points_2d = view_points(points_cam_valid.T, camera_intrinsic, normalize=True)
    
    return points_2d[:2].T, points_cam_valid[:, 2]  # Return (x, y) and depths


def visualize_lidar_projection(image_path: str, points: np.ndarray, cam_translation: np.ndarray,
                              cam_rotation: np.ndarray, camera_intrinsic: np.ndarray,
                              ego_translation: np.ndarray, ego_rotation: np.ndarray,
                              output_path: str = None) -> None:
    """
    Visualize LiDAR points projected onto camera image with distance-based coloring.
    
    Args:
        image_path: Path to camera image
        points: LiDAR points (N, 4) [x, y, z, intensity]
        cam_translation: Camera translation
        cam_rotation: Camera rotation quaternion
        camera_intrinsic: Camera intrinsic matrix
        ego_translation: Ego vehicle translation
        ego_rotation: Ego vehicle rotation quaternion
        output_path: Path to save visualization
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found: {image_path}")
        return
    
    # Load image
    from PIL import Image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Project LiDAR points to image
    points_2d, depths = project_lidar_to_camera(points, cam_translation, cam_rotation, 
                                               camera_intrinsic, ego_translation, ego_rotation)
    
    if len(points_2d) == 0:
        print("  No LiDAR points visible in this camera view")
        return
    
    # Filter points within image bounds
    valid_mask = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_width) &
                  (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_height))
    
    points_2d_valid = points_2d[valid_mask]
    depths_valid = depths[valid_mask]
    
    if len(points_2d_valid) == 0:
        print("  No LiDAR points within image bounds")
        return
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.imshow(img)
    
    # Plot LiDAR points colored by distance
    scatter = ax.scatter(points_2d_valid[:, 0], points_2d_valid[:, 1], 
                        c=depths_valid, cmap='jet', s=1, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Distance (m)', rotation=270, labelpad=20)
    
    ax.set_title(f'LiDAR Points Projected to Camera ({len(points_2d_valid)} points)')
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  LiDAR projection saved to: {output_path}")
    
    plt.close()


def visualize_bev_with_boxes(points: np.ndarray, annotations: List[Dict], categories: List[Dict], 
                            instances: List[Dict], output_path: str = None, map_data: Dict = None,
                            ego_translation: np.ndarray = None, ego_rotation: np.ndarray = None) -> None:
    """
    Create Bird's Eye View (BEV) visualization with 2D bounding boxes and optional map overlay.
    
    Args:
        points: LiDAR point cloud data (N, 4) [x, y, z, intensity]
        annotations: List of sample annotations
        categories: List of category definitions
        instances: List of instance definitions
        output_path: Path to save the visualization
        map_data: Optional map data for overlay
        ego_translation: Ego vehicle translation for coordinate transformation
        ego_rotation: Ego vehicle rotation for coordinate transformation
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Set BEV range (in meters)
    bev_range = 50
    ax.set_xlim([-bev_range, bev_range])
    ax.set_ylim([-bev_range, bev_range])
    ax.set_aspect('equal')
    
    # Draw map if available
    if map_data:
        draw_map_overlay(ax, map_data, bev_range)
    
    # Plot LiDAR points in BEV
    if len(points) > 0:
        # Filter points within BEV range
        valid_mask = ((np.abs(points[:, 0]) <= bev_range) & 
                     (np.abs(points[:, 1]) <= bev_range))
        points_bev = points[valid_mask]
        
        if len(points_bev) > 0:
            # Subsample for performance
            step = max(1, len(points_bev) // 20000)
            sampled_points = points_bev[::step]
            
            # Color by height (z-coordinate)
            heights = sampled_points[:, 2]
            scatter = ax.scatter(sampled_points[:, 0], sampled_points[:, 1], 
                               c=heights, cmap='terrain', s=0.5, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Height (m)', shrink=0.8)
    
    # Draw 2D bounding boxes in BEV
    boxes_drawn = 0
    category_counts = {}
    
    for ann in annotations:
        # Get category name
        category_name = "Unknown"
        if ann.get('instance_token'):
            instance = next((inst for inst in instances if inst['token'] == ann['instance_token']), None)
            if instance and instance.get('category_token'):
                category = next((cat for cat in categories if cat['token'] == instance['category_token']), None)
                if category:
                    category_name = category['name']
        
        # Count categories for statistics
        category_counts[category_name] = category_counts.get(category_name, 0) + 1
        
        # Get 3D box parameters
        center = np.array(ann['translation'])
        size = np.array(ann['size'])
        rotation = np.array(ann['rotation'])  # quaternion [w, x, y, z]
        
        # Transform from global coordinates to ego-relative coordinates
        if ego_translation is not None and ego_rotation is not None:
            # Create transformation matrix from global to ego coordinates
            ego_transform = transform_matrix(ego_translation, ego_rotation, inverse=True)
            
            # Transform box center to ego coordinates
            center_homogeneous = np.append(center, 1)  # Convert to homogeneous coordinates
            center_ego = ego_transform @ center_homogeneous
            center = center_ego[:3]  # Extract x, y, z
            
            # Transform rotation (quaternion) to ego frame
            # For simplicity, we'll use the relative rotation
            ego_rot_matrix = quaternion_to_rotation_matrix(ego_rotation)
            box_rot_matrix = quaternion_to_rotation_matrix(rotation)
            relative_rot_matrix = ego_rot_matrix.T @ box_rot_matrix
            
            # Convert back to quaternion (simplified approach)
            # For BEV, we mainly care about yaw rotation
            rotation = rotation  # Keep original for now, can be improved later
        
        # Check if box is within BEV range
        if abs(center[0]) <= bev_range and abs(center[1]) <= bev_range:
            # Draw enhanced 2D box in BEV (top-down view)
            draw_bev_box_2d(ax, center, size, rotation, category_name)
            boxes_drawn += 1
    
    # Add ego vehicle marker with enhanced styling
    ax.plot(0, 0, 'ro', markersize=10, label='Ego Vehicle', markeredgecolor='white', markeredgewidth=2)
    ax.arrow(0, 0, 0, 4, head_width=1.5, head_length=1.5, fc='red', ec='white', linewidth=2)
    
    # Add range circles for reference
    for radius in [10, 20, 30, 40]:
        if radius <= bev_range:
            circle = plt.Circle((0, 0), radius, fill=False, color='gray', alpha=0.3, linestyle='--', linewidth=1)
            ax.add_patch(circle)
    
    # Create color legend for object categories
    from matplotlib.patches import Patch
    
    # Get category colors from draw_bev_box_2d function
    category_colors = {
        'car': '#FF4444',                    # Bright Red
        'truck': '#00CED1',                  # Dark Turquoise  
        'bus': '#1E90FF',                    # Dodger Blue
        'trailer': '#32CD32',                # Lime Green
        'construction_vehicle': '#FFD700',   # Gold
        'pedestrian': '#FF69B4',             # Hot Pink
        'motorcycle': '#FF8C00',             # Dark Orange
        'bicycle': '#00FA9A',                # Medium Spring Green
        'traffic_cone': '#FFFF00',           # Yellow
        'barrier': '#9370DB',                # Medium Purple
        'movable_object': '#20B2AA',         # Light Sea Green
        'animal': '#8B4513',                 # Saddle Brown
        'vehicle': '#DC143C'                 # Crimson (fallback for vehicle types)
    }
    
    # Define emoji icons for each category
    category_icons = {
        'car': '🚗',
        'truck': '🚛',
        'bus': '🚌',
        'trailer': '🚚',
        'construction_vehicle': '🚧',
        'pedestrian': '🚶',
        'motorcycle': '🏍️',
        'bicycle': '🚲',
        'traffic_cone': '🚥',
        'barrier': '🚧',
        'movable_object': '📦',
        'animal': '🐾',
        'vehicle': '🚗'
    }
    
    # Create legend patches for categories that appear in the scene
    legend_elements = []
    for category_name, count in sorted(category_counts.items()):
        if count > 0:  # Only show categories that are present
            color = category_colors.get(category_name.lower(), '#6C5CE7')  # Default purple
            icon = category_icons.get(category_name.lower(), '📦')  # Default box icon
            legend_elements.append(Patch(facecolor=color, label=f'{icon} {category_name} ({count})'))
    
    # Add ego vehicle to legend
    legend_elements.append(Patch(facecolor='red', label='🚗 Ego Vehicle'))
    
    # Create a separate legend for the color coding - positioned outside the plot area
    if legend_elements:
        # Position legend outside the plot area to avoid overlap
        legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1.0), 
                          fontsize=9, title='Object Categories', title_fontsize=11,
                          frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        legend.get_title().set_fontweight('bold')
        # Ensure legend background is white for better readability
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(1)
    
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    
    # Create detailed title with total count
    title = f'Bird\'s Eye View - {boxes_drawn} Objects'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  BEV visualization saved to: {output_path}")
    
    plt.close()


def draw_bev_box_2d(ax, center: np.ndarray, size: np.ndarray, rotation: np.ndarray, 
                   category_name: str = "", color: str = None):
    """
    Draw enhanced 2D bounding box in Bird's Eye View with category-specific colors and direction indicators.
    
    Args:
        ax: Matplotlib axis
        center: Box center [x, y, z]
        size: Box size [width, length, height]
        rotation: Quaternion rotation [w, x, y, z]
        category_name: Object category name
        color: Box color (if None, will use category-specific color)
    """
    # Enhanced category-specific colors with better visibility
    category_colors = {
        'car': '#FF4444',                    # Bright Red
        'truck': '#00CED1',                  # Dark Turquoise  
        'bus': '#1E90FF',                    # Dodger Blue
        'trailer': '#32CD32',                # Lime Green
        'construction_vehicle': '#FFD700',   # Gold
        'pedestrian': '#FF69B4',             # Hot Pink
        'motorcycle': '#FF8C00',             # Dark Orange
        'bicycle': '#00FA9A',                # Medium Spring Green
        'traffic_cone': '#FFFF00',           # Yellow
        'barrier': '#9370DB',                # Medium Purple
        'movable_object': '#20B2AA',         # Light Sea Green
        'animal': '#8B4513',                 # Saddle Brown
        'vehicle': '#DC143C'                 # Crimson (fallback for vehicle types)
    }
    
    # Use category-specific color or default
    if color is None:
        color = category_colors.get(category_name.lower(), '#6C5CE7')  # Default purple
    
    # Get rotation matrix
    rot_matrix = quaternion_to_rotation_matrix(rotation)
    
    # Define box corners in local coordinates (top-down view, only x-y)
    w, l = size[0], size[1]  # width, length
    corners_local = np.array([
        [-w/2, -l/2],  # rear left
        [w/2, -l/2],   # rear right
        [w/2, l/2],    # front right
        [-w/2, l/2]    # front left
    ])
    
    # Rotate corners
    corners_rotated = (rot_matrix[:2, :2] @ corners_local.T).T
    
    # Translate to global position
    corners_global = corners_rotated + center[:2]
    
    # Close the box by adding the first point at the end
    corners_closed = np.vstack([corners_global, corners_global[0]])
    
    # Draw box with enhanced styling
    ax.plot(corners_closed[:, 0], corners_closed[:, 1], color=color, linewidth=2.5, alpha=0.8)
    
    # Fill the box with semi-transparent color
    ax.fill(corners_closed[:, 0], corners_closed[:, 1], color=color, alpha=0.15)
    
    # Add direction indicator (arrow pointing to front of vehicle)
    front_center = (corners_global[2] + corners_global[3]) / 2  # front edge center
    rear_center = (corners_global[0] + corners_global[1]) / 2   # rear edge center
    
    # Draw direction arrow
    direction_vec = front_center - rear_center
    if np.linalg.norm(direction_vec) > 0:
        direction_vec = direction_vec / np.linalg.norm(direction_vec) * min(l * 0.3, 2.0)
        ax.arrow(center[0], center[1], direction_vec[0], direction_vec[1], 
                head_width=min(w * 0.2, 1.0), head_length=min(l * 0.15, 0.8), 
                fc=color, ec=color, alpha=0.7, linewidth=1.5)
    
    # Category labels removed as requested - using color coding instead
    
    # Add corner markers for better visibility
    corner_size = min(w, l) * 0.1
    for corner in corners_global:
        ax.plot(corner[0], corner[1], 'o', color=color, markersize=3, alpha=0.8)


def draw_map_overlay(ax, map_data: Dict, bev_range: float):
    """
    Draw map overlay on BEV visualization with clear distinction between map and no-map cases.
    
    Args:
        ax: Matplotlib axis
        map_data: Map data dictionary (None if no map available)
        bev_range: BEV visualization range in meters
    """
    if map_data is not None:
        # When map data is available, draw simplified map features
        print("  Map overlay: Drawing simplified map features")
        
        # Draw main coordinate axes with enhanced styling
        ax.axhline(y=0, color='darkblue', linestyle='-', alpha=0.8, linewidth=2.5, label='Main Road')
        ax.axvline(x=0, color='darkblue', linestyle='-', alpha=0.8, linewidth=2.5)
        
        # Add simplified secondary grid (every 25m instead of 30m)
        for i in range(-bev_range, bev_range + 1, 25):
            if i != 0:
                ax.axhline(y=i, color='blue', linestyle='-', alpha=0.4, linewidth=1.0)
                ax.axvline(x=i, color='blue', linestyle='-', alpha=0.4, linewidth=1.0)
        
        # Add subtle lane markings (every 20m instead of 10m)
        for i in range(-bev_range, bev_range + 1, 20):
            if i != 0 and i % 25 != 0:  # Skip lines that overlap with secondary grid
                ax.axhline(y=i, color='lightblue', linestyle=':', alpha=0.3, linewidth=0.5)
                ax.axvline(x=i, color='lightblue', linestyle=':', alpha=0.3, linewidth=0.5)
            
        # Add map background with slight tint
        ax.set_facecolor('#f8fbff')  # Very light blue background for map areas
        
    else:
        # When no map data is available, draw simple grid
        print("  Map overlay: No map data - drawing simple coordinate grid")
        
        # Draw a basic coordinate grid (every 25m)
        for i in range(-bev_range, bev_range + 1, 25):
            ax.axhline(y=i, color='gray', linestyle='-', alpha=0.3, linewidth=1.0)
            ax.axvline(x=i, color='gray', linestyle='-', alpha=0.3, linewidth=1.0)
        
        # Minor grid lines every 10m (lighter)
        for i in range(-bev_range, bev_range + 1, 10):
            if i % 25 != 0:  # Skip lines that overlap with major grid
                ax.axhline(y=i, color='lightgray', linestyle=':', alpha=0.2, linewidth=0.3)
                ax.axvline(x=i, color='lightgray', linestyle=':', alpha=0.2, linewidth=0.3)
            
        # Set plain background for no-map case
        ax.set_facecolor('#ffffff')  # White background for no-map areas
    
    # Common elements for both cases
    # Add coordinate labels
    for i in range(-bev_range, bev_range + 1, 20):
        if i != 0:
            ax.text(i, bev_range - 5, f'{i}m', ha='center', va='center', 
                   fontsize=8, color='gray', alpha=0.7)
            ax.text(bev_range - 5, i, f'{i}m', ha='center', va='center', 
                   fontsize=8, color='gray', alpha=0.7, rotation=90)
    
    # Mark the origin (ego vehicle position)
    ax.plot(0, 0, 'ro', markersize=8, label='Ego Vehicle')
    ax.arrow(0, 0, 0, 5, head_width=2, head_length=2, fc='red', ec='red', alpha=0.7)


def visualize_sample_with_boxes(nuscenes_root, sample_idx):
    """
    Visualize a sample with bounding boxes for all cameras and save to local files.
    
    Args:
        nuscenes_root (str): Path to the NuScenes dataset root directory
        sample_idx (int): Index of the sample to visualize
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image
        import numpy as np
        
        # Initialize variables to avoid UnboundLocalError in exception handling
        camera_data = []
        output_dir = ""
        
        # Load annotation files
        annotation_dir = os.path.join(nuscenes_root, 'v1.0-trainval')
        print(f"\nVisualizing sample {sample_idx} with bounding boxes:")
        print(f"Successfully loaded annotation files from: {annotation_dir}")
        
        with open(os.path.join(annotation_dir, 'sample.json'), 'r') as f:
            samples = json.load(f)
        
        with open(os.path.join(annotation_dir, 'sample_data.json'), 'r') as f:
            sample_data = json.load(f)
        
        with open(os.path.join(annotation_dir, 'sample_annotation.json'), 'r') as f:
            sample_annotations = json.load(f)
        
        with open(os.path.join(annotation_dir, 'category.json'), 'r') as f:
            categories = json.load(f)
        
        with open(os.path.join(annotation_dir, 'sensor.json'), 'r') as f:
            sensors = json.load(f)
        
        with open(os.path.join(annotation_dir, 'calibrated_sensor.json'), 'r') as f:
            calibrated_sensors = json.load(f)
        
        with open(os.path.join(annotation_dir, 'ego_pose.json'), 'r') as f:
            ego_poses = json.load(f)
        
        with open(os.path.join(annotation_dir, 'instance.json'), 'r') as f:
            instances = json.load(f)
        
        # Get the sample
        if sample_idx >= len(samples):
            print(f"Error: Sample index {sample_idx} is out of range. Total samples: {len(samples)}")
            return
        
        sample = samples[sample_idx]
        print(f"Sample token: {sample['token']}")
        print(f"Scene token: {sample['scene_token']}")
        print(f"Timestamp: {sample['timestamp']}")
        
        # Find all sample_data entries for this sample
        sample_data_entries = [sd for sd in sample_data if sd['sample_token'] == sample['token']]
        
        # Filter for camera data only (main samples, not sweeps)
        camera_data = [sd for sd in sample_data_entries 
                      if 'CAM' in sd.get('filename', '') and 'samples/' in sd.get('filename', '')]
        
        if not camera_data:
            print("No camera data found for this sample")
            return
        
        # Create mapping from calibrated_sensor_token to channel
        calibrated_to_sensor = {cs['token']: cs['sensor_token'] for cs in calibrated_sensors}
        sensor_to_channel = {s['token']: s['channel'] for s in sensors}
        
        # Get sample annotations for this sample
        sample_anns = [ann for ann in sample_annotations if ann['sample_token'] == sample['token']]
        
        # Create category lookup
        category_lookup = {cat['token']: cat['name'] for cat in categories}
        
        # Create instance to category mapping
        instance_to_category = {inst['token']: inst['category_token'] for inst in instances}
        
        # Create output directory for visualizations
        output_dir = os.path.join(nuscenes_root, 'visualizations', f'sample_{sample_idx}')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Found {len(camera_data)} camera views and {len(sample_anns)} annotations")
        
        # Process each camera view
        saved_files = []
        for cam_data in camera_data:
            try:
                # Get channel name
                calibrated_sensor_token = cam_data['calibrated_sensor_token']
                sensor_token = calibrated_to_sensor.get(calibrated_sensor_token)
                channel = sensor_to_channel.get(sensor_token, 'UNKNOWN')
                
                print(f"Processing {channel}...")
                
                # Load image
                image_path = os.path.join(nuscenes_root, cam_data['filename'])
                if not os.path.exists(image_path):
                    print(f"  Warning: Image file not found: {image_path}")
                    continue
                
                # Load and display image
                img = Image.open(image_path)
                
                # Create side-by-side subplots: left for 3D boxes, right for 2D boxes
                fig, (ax_3d, ax_2d) = plt.subplots(1, 2, figsize=(24, 8))
                
                # Display image on both subplots
                ax_3d.imshow(img)
                ax_3d.set_title(f'{channel} - Sample {sample_idx} (3D Bounding Boxes)')
                
                ax_2d.imshow(img)
                ax_2d.set_title(f'{channel} - Sample {sample_idx} (2D Bounding Boxes)')
                
                # Get camera calibration data
                calibrated_sensor = next((cs for cs in calibrated_sensors 
                                        if cs['token'] == calibrated_sensor_token), None)
                
                if calibrated_sensor and calibrated_sensor.get('camera_intrinsic'):
                    # Project and draw 3D bounding boxes
                    camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
                    cam_translation = np.array(calibrated_sensor['translation'])
                    cam_rotation = np.array(calibrated_sensor['rotation'])  # quaternion [w, x, y, z]
                    
                    # Get ego pose for this sample
                    ego_pose = next((ep for ep in ego_poses if ep['token'] == cam_data['ego_pose_token']), None)
                    if not ego_pose:
                        print(f"  Warning: No ego pose found for {channel}")
                        continue
                    
                    ego_translation = np.array(ego_pose['translation'])
                    ego_rotation = np.array(ego_pose['rotation'])
                    
                    boxes_drawn_3d = 0
                    boxes_drawn_2d = 0
                    for i, ann in enumerate(sample_anns):
                        try:
                            # Get 3D bounding box parameters
                            center_3d = np.array(ann['translation'])  # [x, y, z]
                            size_3d = np.array(ann['size'])  # [width, length, height]
                            rotation_3d = np.array(ann['rotation'])  # quaternion [w, x, y, z]
                            
                            # Project 3D bounding box to 2D
                            projection_result = project_3d_box_to_2d(
                                center_3d, size_3d, rotation_3d,
                                cam_translation, cam_rotation, camera_intrinsic,
                                ego_translation, ego_rotation, debug=(i == 0 and channel == 'CAM_FRONT')
                            )
                            
                            if projection_result[0] is not None:
                                corners_2d, corners_3d_cam = projection_result
                                
                                # Get image dimensions for visibility check
                                img_width, img_height = img.size
                                
                                # Check if box is visible using NuScenes official visibility check
                                is_visible = box_in_image(
                                    corners_3d_cam, corners_2d.T, camera_intrinsic, 
                                    (img_width, img_height), BoxVisibility.ANY
                                )
                                
                                if not is_visible:
                                    if i == 0 and channel == 'CAM_FRONT':  # Debug for first annotation
                                        print(f"    Box {i} not visible, skipping...")
                                    continue
                                
                                # Get category name for labeling
                                category_name = ""
                                if ann.get('instance_token'):
                                    # Get category from instance table
                                    instance = next((inst for inst in instances if inst['token'] == ann['instance_token']), None)
                                    if instance and instance.get('category_token'):
                                        category = next((cat for cat in categories if cat['token'] == instance['category_token']), None)
                                        if category:
                                            category_name = category['name']
                                
                                # Draw 3D bounding box (wireframe) on left subplot
                                draw_3d_box_2d(ax_3d, corners_2d, category_name=category_name, color='red', linewidth=2, 
                                             img_width=img_width, img_height=img_height)
                                boxes_drawn_3d += 1
                                
                                # Calculate and draw 2D bounding box from 3D projection on right subplot
                                bbox_2d_from_3d = get_2d_bbox_from_3d_projection(corners_2d)
                                if bbox_2d_from_3d is not None:
                                    # Draw the projected 2D bbox (blue dashed)
                                    success = draw_2d_bbox_from_3d(ax_2d, corners_2d, f"{category_name} (3D→2D)", 
                                                                 color='blue', linewidth=2,
                                                                 img_width=img_width, img_height=img_height)
                                    if success:
                                        boxes_drawn_2d += 1
                                
                                    # For demonstration, also draw a regular 2D detection box (solid green)
                                    # This would typically come from a 2D detector, but we'll simulate it
                                    x_min, y_min, x_max, y_max = bbox_2d_from_3d
                                    # Create a simulated 2D detection box (slightly smaller)
                                    margin = min(20, (x_max - x_min) * 0.1, (y_max - y_min) * 0.1)
                                    simulated_2d_bbox = [x_min + margin, y_min + margin, 
                                                       x_max - margin, y_max - margin]
                                    
                                    success = draw_2d_bbox(ax_2d, simulated_2d_bbox, f"{category_name} (2D Det)", 
                                                         color='green', linewidth=2,
                                                         img_width=img_width, img_height=img_height)
                                    if success:
                                        boxes_drawn_2d += 1
                                
                        except Exception as e:
                            print(f"    Warning: Failed to project annotation {i}: {e}")
                    
                    # Add annotation info for 3D subplot
                    info_text_3d = f'3D Annotations: {len(sample_anns)} (Drawn: {boxes_drawn_3d})'
                    ax_3d.text(10, 30, info_text_3d, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                           fontsize=12, fontweight='bold')
                    
                    # Add legend for 3D subplot
                    ax_3d.text(10, 70, "3D Wireframe", fontsize=10, 
                              bbox=dict(boxstyle="round,pad=0.2", facecolor="red", alpha=0.8))
                    ax_3d.plot([10, 40], [85, 85], color='red', linestyle='-', linewidth=2)
                    
                    # Add annotation info for 2D subplot
                    info_text_2d = f'2D Annotations: {len(sample_anns)} (Drawn: {boxes_drawn_2d})'
                    ax_2d.text(10, 30, info_text_2d, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                           fontsize=12, fontweight='bold')
                    
                    # Add legend for 2D subplot
                    legend_y_start = 70
                    legend_items_2d = [
                        ("3D→2D Bbox", "blue", "--"),
                        ("2D Detection", "green", "-")
                    ]
                    
                    for i, (label, color, linestyle) in enumerate(legend_items_2d):
                        y_pos = legend_y_start + i * 25
                        # Draw sample line
                        ax_2d.plot([10, 40], [y_pos, y_pos], color=color, linestyle=linestyle, linewidth=2)
                        # Add text
                        ax_2d.text(45, y_pos, label, fontsize=10, verticalalignment='center',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                else:
                    # Fallback: just show annotation count
                    ax_3d.text(10, 30, f'Annotations: {len(sample_anns)} (No camera intrinsics)', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                           fontsize=12, fontweight='bold')
                    ax_2d.text(10, 30, f'Annotations: {len(sample_anns)} (No camera intrinsics)', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                           fontsize=12, fontweight='bold')
                
                ax_3d.axis('off')
                ax_2d.axis('off')
                
                # Save the visualization
                output_path = os.path.join(output_dir, f'{channel}_sample_{sample_idx}.png')
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                saved_files.append(output_path)
                print(f"  Saved visualization to: {output_path}")
                
            except Exception as e:
                print(f"  Error processing camera {channel}: {str(e)}")
        
        # Create a combined visualization
        if saved_files:
            try:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle(f'NuScenes Sample {sample_idx} - All Camera Views', fontsize=16)
                
                camera_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                               'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                
                for i, channel in enumerate(camera_order):
                    row, col = i // 3, i % 3
                    
                    # Find corresponding saved file
                    matching_file = None
                    for saved_file in saved_files:
                        if channel in saved_file:
                            matching_file = saved_file
                            break
                    
                    if matching_file and os.path.exists(matching_file):
                        img = Image.open(matching_file)
                        axes[row, col].imshow(img)
                        axes[row, col].set_title(channel)
                    else:
                        axes[row, col].text(0.5, 0.5, f'{channel}\nNot Available', 
                                          ha='center', va='center', transform=axes[row, col].transAxes)
                    
                    axes[row, col].axis('off')
                
                combined_path = os.path.join(output_dir, f'combined_sample_{sample_idx}.png')
                plt.savefig(combined_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                print(f"Combined visualization saved to: {combined_path}")
                
            except Exception as e:
                print(f"Error creating combined visualization: {e}")
        
        # Save annotation summary
        summary_path = os.path.join(output_dir, 'annotations.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Sample {sample_idx} Annotations\n")
            f.write(f"Sample token: {sample['token']}\n")
            f.write(f"Timestamp: {sample['timestamp']}\n")
            f.write(f"Camera views: {len(camera_data)}\n")
            f.write(f"Total annotations: {len(sample_anns)}\n\n")
            
            for i, ann in enumerate(sample_anns):
                category_token = instance_to_category.get(ann['instance_token'])
                category_name = category_lookup.get(category_token, 'Unknown')
                f.write(f"Annotation {i+1}:\n")
                f.write(f"  Category: {category_name}\n")
                f.write(f"  Instance token: {ann['instance_token']}\n")
                f.write(f"  Visibility: {ann['visibility_token']}\n")
                f.write(f"  Translation: {ann['translation']}\n")
                f.write(f"  Size: {ann['size']}\n")
                f.write(f"  Rotation: {ann['rotation']}\n\n")
        
        # Add new visualizations
        print(f"\n🚀 Creating additional visualizations...")
        
        # 1. LiDAR 3D visualization with 3D boxes
        lidar_data_entries = [sd for sd in sample_data_entries if 'LIDAR_TOP' in sd.get('filename', '')]
        if lidar_data_entries:
            lidar_data = lidar_data_entries[0]  # Get the first LiDAR entry
            lidar_path = os.path.join(nuscenes_root, lidar_data['filename'])
            if os.path.exists(lidar_path):
                print(f"  📡 Processing LiDAR data...")
                lidar_points = load_lidar_points(lidar_path)
                lidar_3d_output = os.path.join(output_dir, f"lidar_3d_sample_{sample_idx}.png")
                visualize_lidar_3d(lidar_points, sample_anns, categories, instances, lidar_3d_output)
                saved_files.append(lidar_3d_output)
                
                # 1.5. Interactive LiDAR 3D visualization with Open3D
                print(f"  🎮 Creating interactive Open3D visualization...")
                lidar_3d_open3d_output = os.path.join(output_dir, f"lidar_3d_open3d_sample_{sample_idx}.png")
                
                # Get ego pose from LiDAR data for coordinate transformation
                lidar_ego_pose_token = lidar_data['ego_pose_token']
                ego_pose = next((ep for ep in ego_poses if ep['token'] == lidar_ego_pose_token), None)
                if ego_pose:
                    ego_translation = np.array(ego_pose['translation'])
                    ego_rotation = np.array(ego_pose['rotation'])
                    visualize_lidar_3d_open3d(lidar_points, sample_anns, categories, instances, 
                                            lidar_3d_open3d_output, ego_translation, ego_rotation)
                else:
                    visualize_lidar_3d_open3d(lidar_points, sample_anns, categories, instances, 
                                            lidar_3d_open3d_output)
                saved_files.append(lidar_3d_open3d_output)
                
                # 2. LiDAR projection to camera images
                for cam_data in camera_data[:2]:  # Process first 2 cameras for performance
                    try:
                        calibrated_sensor_token = cam_data['calibrated_sensor_token']
                        sensor_token = calibrated_to_sensor.get(calibrated_sensor_token)
                        channel = sensor_to_channel.get(sensor_token, 'UNKNOWN')
                        
                        # Get camera calibration
                        calibrated_sensor = next((cs for cs in calibrated_sensors 
                                                if cs['token'] == calibrated_sensor_token), None)
                        if calibrated_sensor:
                            cam_translation = np.array(calibrated_sensor['translation'])
                            cam_rotation = np.array(calibrated_sensor['rotation'])
                            camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
                            
                            # Get ego pose from LiDAR data (not camera data)
                            lidar_ego_pose_token = lidar_data['ego_pose_token']
                            ego_pose = next((ep for ep in ego_poses if ep['token'] == lidar_ego_pose_token), None)
                            if ego_pose:
                                ego_translation = np.array(ego_pose['translation'])
                                ego_rotation = np.array(ego_pose['rotation'])
                                
                                image_path = os.path.join(nuscenes_root, cam_data['filename'])
                                lidar_proj_output = os.path.join(output_dir, f"lidar_projection_{channel}_sample_{sample_idx}.png")
                                
                                print(f"  🎯 Creating LiDAR projection for {channel}...")
                                visualize_lidar_projection(image_path, lidar_points, cam_translation, 
                                                         cam_rotation, camera_intrinsic, ego_translation, 
                                                         ego_rotation, lidar_proj_output)
                                saved_files.append(lidar_proj_output)
                    except Exception as e:
                        print(f"    Warning: Failed to create LiDAR projection for {channel}: {e}")
                
                # 3. BEV visualization with 2D boxes
                print(f"  🗺️  Creating Bird's Eye View...")
                bev_output = os.path.join(output_dir, f"bev_sample_{sample_idx}.png")
                
                # Get ego pose from LiDAR data for coordinate transformation
                lidar_ego_pose_token = lidar_data['ego_pose_token']
                ego_pose = next((ep for ep in ego_poses if ep['token'] == lidar_ego_pose_token), None)
                if ego_pose:
                    ego_translation = np.array(ego_pose['translation'])
                    ego_rotation = np.array(ego_pose['rotation'])
                    visualize_bev_with_boxes(lidar_points, sample_anns, categories, instances, bev_output,
                                           ego_translation=ego_translation, ego_rotation=ego_rotation)
                else:
                    visualize_bev_with_boxes(lidar_points, sample_anns, categories, instances, bev_output)
                saved_files.append(bev_output)
                
                # 4. BEV with map overlay (placeholder implementation)
                print(f"  🗺️  Creating BEV with map overlay...")
                bev_map_output = os.path.join(output_dir, f"bev_with_map_sample_{sample_idx}.png")
                # For now, we'll use a placeholder map_data
                map_data = {"placeholder": True}  # In real implementation, load actual map data
                if ego_pose:
                    visualize_bev_with_boxes(lidar_points, sample_anns, categories, instances, bev_map_output, map_data,
                                           ego_translation=ego_translation, ego_rotation=ego_rotation)
                else:
                    visualize_bev_with_boxes(lidar_points, sample_anns, categories, instances, bev_map_output, map_data)
                saved_files.append(bev_map_output)
            else:
                print(f"  ⚠️  LiDAR file not found: {lidar_path}")
        else:
            print(f"  ⚠️  No LiDAR data found for this sample")

        print(f"\nVisualization completed successfully!")
        print(f"Files saved to: {output_dir}")
        print(f"- Individual camera views: {len([f for f in saved_files if 'CAM_' in f])} files")
        print(f"- LiDAR visualizations: {len([f for f in saved_files if 'lidar' in f])} files")
        print(f"- BEV visualizations: {len([f for f in saved_files if 'bev' in f])} files")
        print(f"- Combined view: combined_sample_{sample_idx}.png")
        print(f"- Annotation summary: annotations.txt")
        
    except KeyError as e:
        print(f"Error: Missing key in annotation data - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file - {e}")
    except Exception as e:
        print(f"Unexpected error during visualization: {e}")
        import traceback
        traceback.print_exc()
        
        # Create a combined visualization
        if camera_data:
            print(f"\n🎨 Creating combined visualization...")
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'NuScenes Sample {sample_idx} - All Camera Views', fontsize=16)
            
            camera_positions = [
                (0, 1, 'CAM_FRONT'),
                (0, 0, 'CAM_FRONT_LEFT'), 
                (0, 2, 'CAM_FRONT_RIGHT'),
                (1, 1, 'CAM_BACK'),
                (1, 0, 'CAM_BACK_LEFT'),
                (1, 2, 'CAM_BACK_RIGHT')
            ]
            
            for row, col, channel in camera_positions:
                ax = axes[row, col]
                
                # Find corresponding camera data
                cam_data = None
                for cd in camera_data:
                    # Get channel name for this camera data
                    calibrated_sensor_token = cd['calibrated_sensor_token']
                    sensor_token = calibrated_to_sensor.get(calibrated_sensor_token)
                    cd_channel = sensor_to_channel.get(sensor_token, 'UNKNOWN')
                    if cd_channel == channel:
                        cam_data = cd
                        break
                
                if cam_data:
                    image_path = os.path.join(nuscenes_root, cam_data['filename'])
                    if os.path.exists(image_path):
                        try:
                            image = cv2.imread(image_path)
                            if image is not None:
                                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                ax.imshow(image_rgb)
                                ax.set_title(f'{channel}', fontsize=12)
                                ax.axis('off')
                            else:
                                ax.text(0.5, 0.5, f'{channel}\nImage not found', 
                                       ha='center', va='center', transform=ax.transAxes)
                                ax.axis('off')
                        except:
                            ax.text(0.5, 0.5, f'{channel}\nError loading', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.axis('off')
                else:
                    ax.text(0.5, 0.5, f'{channel}\nNo data', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            
            plt.tight_layout()
            combined_output = os.path.join(output_dir, f"sample_{sample_idx}_combined.png")
            plt.savefig(combined_output, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  💾 Saved combined visualization: {combined_output}")
        
        print(f"\n✅ Visualization complete for sample {sample_idx}")
        print(f"📁 All visualizations saved to: {output_dir}")
        
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file - {e}")
    except KeyError as e:
        print(f"Error: Missing key in annotation data - {e}")
    except Exception as e:
        print(f"Error: Unexpected error during visualization - {e}")
        import traceback
        traceback.print_exc()


def suggest_dataset_fixes(diagnosis: Dict[str, Any]) -> None:
    """
    Provide detailed suggestions for fixing common NuScenes dataset issues
    
    Args:
        diagnosis: Dictionary containing diagnosis results from diagnose_dataset_issues
    """
    print("\n" + "="*60)
    print("DATASET FIX SUGGESTIONS")
    print("="*60)
    
    if diagnosis['status'] == 'healthy':
        print("✓ Your dataset appears to be healthy! No fixes needed.")
        return
    
    print("Based on the diagnosis, here are specific steps to fix your dataset:\n")
    
    # Structure issues
    if diagnosis['structure_issues']:
        print("🔧 STRUCTURE ISSUES:")
        for issue in diagnosis['structure_issues']:
            if "Missing directory: samples" in issue:
                print("  1. Missing 'samples' directory:")
                print("     - Download the main NuScenes dataset files")
                print("     - Extract: v1.0-trainval_01.tgz through v1.0-trainval_10.tgz")
                print("     - Ensure extraction creates a 'samples' folder")
                
            elif "Missing directory: sweeps" in issue:
                print("  2. Missing 'sweeps' directory:")
                print("     - Download the sweep files (optional for basic usage)")
                print("     - Extract: v1.0-trainval_blobs.tgz")
                print("     - Note: Sweeps provide temporal context but aren't always required")
                
            elif "Missing directory: v1.0-trainval" in issue:
                print("  3. Missing 'v1.0-trainval' directory:")
                print("     - Download the metadata package: v1.0-trainval_meta.tgz")
                print("     - Extract it to create the annotation files")
                
            elif "Missing sensor directory" in issue:
                sensor = issue.split(": ")[-1]
                print(f"  4. Missing sensor directory '{sensor}':")
                print(f"     - Check if the corresponding data files were extracted properly")
                print(f"     - Verify the sensor data is included in your download")
        print()
    
    # Missing files
    if diagnosis['missing_files']:
        print("📁 MISSING FILES:")
        annotation_files = [f for f in diagnosis['missing_files'] if f.endswith('.json')]
        if annotation_files:
            print("  Missing annotation files:")
            for file_name in annotation_files:
                print(f"    - {file_name}")
            print("  Fix: Download and extract v1.0-trainval_meta.tgz")
        print()
    
    # Data integrity issues
    if diagnosis['data_integrity']:
        print("🔍 DATA INTEGRITY:")
        
        # Check for empty directories
        empty_sensors = [k for k, v in diagnosis['data_integrity'].items() 
                        if v == 0 and 'CAM_' in k]
        if empty_sensors:
            print("  Empty camera directories found:")
            for sensor in empty_sensors:
                print(f"    - {sensor}: No image files")
            print("  Fix: Re-extract the main dataset files")
        
        # Check for missing LiDAR
        if diagnosis['data_integrity'].get('LIDAR_TOP', 0) == 0:
            print("  No LiDAR data found:")
            print("    - LiDAR files should be in samples/LIDAR_TOP/")
            print("    - Check if point cloud files (.pcd) were extracted")
        print()
    
    # Download instructions
    print("📥 DOWNLOAD CHECKLIST:")
    print("  Required files for a complete NuScenes v1.0-trainval dataset:")
    print("  ✓ v1.0-trainval_meta.tgz (metadata/annotations)")
    print("  ✓ v1.0-trainval_01.tgz through v1.0-trainval_10.tgz (sensor data)")
    print("  ○ v1.0-trainval_blobs.tgz (sweeps - optional)")
    print()
    
    # Extraction instructions
    print("📦 EXTRACTION INSTRUCTIONS:")
    print("  1. Create a directory for your dataset (e.g., /data/nuscenes/)")
    print("  2. Extract all .tgz files to the same directory:")
    print("     tar -xzf v1.0-trainval_meta.tgz -C /data/nuscenes/")
    print("     tar -xzf v1.0-trainval_01.tgz -C /data/nuscenes/")
    print("     ... (repeat for all files)")
    print("  3. Verify the final structure:")
    print("     /data/nuscenes/")
    print("     ├── samples/")
    print("     ├── sweeps/")
    print("     └── v1.0-trainval/")
    print()
    
    # Common issues
    print("⚠️  COMMON ISSUES:")
    print("  - Partial downloads: Re-download any corrupted .tgz files")
    print("  - Wrong extraction path: Ensure all files extract to the same root directory")
    print("  - Permission issues: Check read/write permissions on the dataset directory")
    print("  - Disk space: NuScenes requires ~350GB for the full trainval set")
    print()
    
    print("💡 TIP: Run this script again after applying fixes to verify the dataset.")
    print("="*60)

def extract_nuscenes_subset(nuscenes_root: str, num_samples: int, output_dir: str = None) -> str:
    """
    Extract a subset of NuScenes samples with their annotations and create a test folder.
    
    Args:
        nuscenes_root: Path to the full NuScenes dataset
        num_samples: Number of samples to extract
        output_dir: Output directory for the subset (default: nuscenes_root + '_subset_N')
    
    Returns:
        Path to the created zip file
    """
    import shutil
    import zipfile
    from datetime import datetime
    
    print(f"🔄 Extracting {num_samples} samples from NuScenes dataset...")
    
    # Set default output directory
    if output_dir is None:
        output_dir = f"{nuscenes_root}_subset_{num_samples}"
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "sweeps"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "v1.0-trainval"), exist_ok=True)
    
    # Create sensor subdirectories
    for sensor in NUSCENES_STRUCTURE['samples']:
        os.makedirs(os.path.join(output_dir, "samples", sensor), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "sweeps", sensor), exist_ok=True)
    
    # Load annotation files
    annotations_dir = os.path.join(nuscenes_root, "v1.0-trainval")
    
    try:
        with open(os.path.join(annotations_dir, "sample.json"), 'r') as f:
            samples = json.load(f)
        
        with open(os.path.join(annotations_dir, "sample_data.json"), 'r') as f:
            sample_data = json.load(f)
        
        with open(os.path.join(annotations_dir, "sample_annotation.json"), 'r') as f:
            sample_annotations = json.load(f)
            
    except FileNotFoundError as e:
        print(f"❌ Error: Required annotation file not found: {e}")
        return None
    
    # Limit samples to available count
    num_samples = min(num_samples, len(samples))
    selected_samples = samples[:num_samples]
    
    print(f"📊 Processing {num_samples} samples...")
    
    # Collect tokens for selected samples
    selected_sample_tokens = {sample['token'] for sample in selected_samples}
    
    # Find related sample_data entries
    related_sample_data = []
    copied_files = set()
    
    for data_entry in sample_data:
        if data_entry['sample_token'] in selected_sample_tokens:
            related_sample_data.append(data_entry)
            
            # Copy the actual data file
            src_file = os.path.join(nuscenes_root, data_entry['filename'])
            dst_file = os.path.join(output_dir, data_entry['filename'])
            
            if os.path.exists(src_file) and data_entry['filename'] not in copied_files:
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)
                copied_files.add(data_entry['filename'])
    
    # Find related annotations
    related_annotations = []
    for annotation in sample_annotations:
        if annotation['sample_token'] in selected_sample_tokens:
            related_annotations.append(annotation)
    
    print(f"📁 Copied {len(copied_files)} data files")
    print(f"📝 Found {len(related_annotations)} annotations")
    
    # Create filtered annotation files
    filtered_annotations = {
        'sample.json': selected_samples,
        'sample_data.json': related_sample_data,
        'sample_annotation.json': related_annotations
    }
    
    # Copy and filter other annotation files
    for ann_file in REQUIRED_ANNOTATION_FILES:
        src_path = os.path.join(annotations_dir, ann_file)
        dst_path = os.path.join(output_dir, "v1.0-trainval", ann_file)
        
        if ann_file in filtered_annotations:
            # Write filtered data
            with open(dst_path, 'w') as f:
                json.dump(filtered_annotations[ann_file], f, indent=2)
        elif os.path.exists(src_path):
            # Copy full file for other annotations (they're usually small)
            shutil.copy2(src_path, dst_path)
    
    print(f"✅ Subset created in: {output_dir}")
    
    # Create zip file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"nuscenes_subset_{num_samples}samples_{timestamp}.zip"
    zip_path = os.path.join(os.path.dirname(output_dir), zip_filename)
    
    print(f"🗜️  Creating zip archive: {zip_filename}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arc_name)
    
    # Get zip file size
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    
    print(f"✅ Zip archive created: {zip_path}")
    print(f"📦 Archive size: {zip_size_mb:.1f} MB")
    print(f"📊 Contains: {num_samples} samples, {len(related_sample_data)} data files, {len(related_annotations)} annotations")
    
    return zip_path

def main():
    """
    Main function with enhanced NuScenes dataset management
    """
    parser = argparse.ArgumentParser(description="NuScenes Dataset Extraction and Validation Tool")
    parser.add_argument("--zip_dir", default=DEFAULT_ZIP_DIR, 
                       help="Directory containing NuScenes zip files")
    parser.add_argument("--extract_dir", default=DEFAULT_NUSCENES_DIR, 
                       help="Directory to extract files to (will become NuScenes root)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("NUSCENES DATASET MANAGEMENT TOOL")
    print("="*60)
    print(f"Source directory: {args.zip_dir}")
    print(f"Target directory: {args.extract_dir}")
    
    while True:
        print("\nSelect an option:")
        print("1. Extract dataset files")
        print("2. Check extracted folder structure")
        print("3. Count files and check annotations")
        print("4. Run comprehensive dataset diagnosis")
        print("5. Verify NuScenes data structure")
        print("6. Visualize sample with annotations")
        print("7. Get dataset fix suggestions")
        print("8. Complete setup (extract + validate)")
        print("9. Extract subset for testing")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-9): ").strip()
        
        if choice == '1':
            extract_files(args.zip_dir, args.extract_dir)
            
        elif choice == '2':
            check_extracted_folders(args.extract_dir)
            
        elif choice == '3':
            count_files_and_check_annotations(args.extract_dir)
            
        elif choice == '4':
            diagnosis = diagnose_dataset_issues(args.extract_dir)
            
        elif choice == '5':
            check_nuscenes_data_structure(args.extract_dir)
            
        elif choice == '6':
            sample_idx = int(input("Enter sample index to visualize (default 0): ") or "0")
            visualize_sample_with_boxes(args.extract_dir, sample_idx)
            
        elif choice == '7':
            diagnosis = diagnose_dataset_issues(args.extract_dir)
            suggest_dataset_fixes(diagnosis)
            
        elif choice == '8':
            print("\n🚀 Running complete setup and validation...")
            extract_files(args.zip_dir, args.extract_dir)
            check_extracted_folders(args.extract_dir)
            count_files_and_check_annotations(args.extract_dir)
            diagnosis = diagnose_dataset_issues(args.extract_dir)
            check_nuscenes_data_structure(args.extract_dir)
            if diagnosis['status'] != 'healthy':
                suggest_dataset_fixes(diagnosis)
        
        elif choice == '9':
            try:
                num_samples = int(input("Enter number of samples to extract (default 10): ") or "10")
                output_dir = input("Enter output directory (press Enter for default): ").strip() or None
                
                zip_path = extract_nuscenes_subset(args.extract_dir, num_samples, output_dir)
                if zip_path:
                    print(f"\n✅ Subset extraction completed successfully!")
                    print(f"📦 Zip file: {zip_path}")
                else:
                    print("\n❌ Subset extraction failed!")
            except ValueError:
                print("❌ Invalid number of samples. Please enter a valid integer.")
            except Exception as e:
                print(f"❌ Error during subset extraction: {e}")
        
        elif choice == '0':
            print("Exiting program.")
            break
            
        else:
            print("Invalid choice. Please enter a number between 0-9.")

if __name__ == "__main__":
    main()
