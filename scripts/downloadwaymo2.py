#!/usr/bin/env python3
"""
Waymo Open Dataset v2 Utilities and Visualization Tool

This module provides comprehensive utilities for working with Waymo Open Dataset v2,
including dataset download, validation, visualization, and diagnostic features.

Features:
- Dataset download from Google Cloud Storage
- Dataset structure validation
- 2D camera visualization with bounding boxes
- 3D bounding box projection on images
- 3D LiDAR visualization with Mayavi
- Dataset diagnosis and issue detection
- Interactive command-line interface

Based on kitti.py and waymo dataset specifications
"""

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
import tarfile
import glob
from google.cloud import storage  # pip install --upgrade google-cloud-storage

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

# Check for optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Waymo Open Dataset v2 structure definition
WAYMO_STRUCTURE = {
    'training': ['camera_images', 'lidar', 'labels'],
    'validation': ['camera_images', 'lidar', 'labels'],
    'testing': ['camera_images', 'lidar'],
    'domain_adaptation_training': ['camera_images', 'lidar', 'labels'],
    'domain_adaptation_validation': ['camera_images', 'lidar', 'labels']
}

WAYMO_BUCKET_FOLDERS = [
    'training',
    'validation', 
    'testing',
    'domain_adaptation_training',
    'domain_adaptation_validation'
]

# Camera names for Waymo dataset
WAYMO_CAMERA_NAMES = {
    0: "FRONT",
    1: "FRONT_LEFT", 
    2: "FRONT_RIGHT",
    3: "SIDE_LEFT",
    4: "SIDE_RIGHT"
}

class BoxVisibility(IntEnum):
    """ Enumerates the various level of box visibility in an image """
    ALL = 0  # Requires all corners are inside the image.
    ANY = 1  # Requires at least one corner visible in the image.
    NONE = 2  # Requires no corners to be inside, i.e. box can be fully outside the image.

# Color definitions for visualization
INSTANCE_Color = {
    'TYPE_VEHICLE': 'red', 'TYPE_PEDESTRIAN': 'green', 'TYPE_SIGN': 'yellow', 
    'TYPE_CYCLIST': 'purple', 'TYPE_UNKNOWN': 'gray'
}

INSTANCE3D_ColorCV2 = {
    'TYPE_VEHICLE': (0, 255, 0), 'TYPE_PEDESTRIAN': (255, 255, 0), 
    'TYPE_SIGN': (0, 255, 255), 'TYPE_CYCLIST': (127, 127, 64), 
    'TYPE_UNKNOWN': (128, 128, 128)
}

INSTANCE3D_Color = {
    'TYPE_VEHICLE': (0, 1, 0), 'TYPE_PEDESTRIAN': (0, 1, 1), 
    'TYPE_SIGN': (1, 1, 0), 'TYPE_CYCLIST': (0.5, 0.5, 0.3), 
    'TYPE_UNKNOWN': (0.5, 0.5, 0.5)
}

class WaymoDatasetDownloader:
    def __init__(self, bucket_name='waymo_open_dataset_v_2_0_1', 
                 destination_directory='waymo_dataset'):
        """
        Initialize the Waymo Dataset Downloader.
        
        Args:
            bucket_name (str): Name of the Google Cloud Storage bucket
            destination_directory (str): Local directory to save downloaded files
        """
        self.bucket_name = bucket_name
        self.destination_directory = os.path.abspath(destination_directory)
        self.progress_file = os.path.join(self.destination_directory, 'download_progress.json')
        
        # Ensure the destination directory exists
        os.makedirs(self.destination_directory, exist_ok=True)
        
        # Initialize Google Cloud Storage client
        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(bucket_name)
            print(f"✅ Connected to Google Cloud Storage bucket: {bucket_name}")
        except Exception as e:
            print(f"❌ Failed to connect to Google Cloud Storage: {e}")
            print("💡 Make sure GOOGLE_APPLICATION_CREDENTIALS is set correctly")
            raise
        
        # Load or initialize download progress
        self.download_progress = self._load_progress()

    def _load_progress(self):
        """
        Load download progress from a JSON file or create a new progress tracker.
        
        Returns:
            dict: A dictionary tracking downloaded files
        """
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_progress(self):
        """
        Save download progress to a JSON file.
        """
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.download_progress, f, indent=2)
        except IOError as e:
            print(f"Error saving progress: {e}")

    def download_folders(self, folders=None, force_redownload=False):
        """
        Download specific folders from the Waymo dataset.
        
        Args:
            folders (list): List of folder names to download. 
                            If None, lists all available folders.
            force_redownload (bool): If True, redownload all files even if already downloaded
        """
        print(f"\n{'='*60}")
        print("WAYMO OPEN DATASET V2 DOWNLOAD")
        print(f"{'='*60}")
        
        # List all blobs in the bucket
        print("📋 Listing bucket contents...")
        try:
            blobs = list(self.bucket.list_blobs())
            print(f"✅ Found {len(blobs)} files in bucket")
        except Exception as e:
            print(f"❌ Failed to list bucket contents: {e}")
            return False
        
        # If no folders specified, list available folders
        if folders is None:
            available_folders = set(blob.name.split('/')[0] for blob in blobs if '/' in blob.name)
            print("\n📁 Available folders:")
            for folder in sorted(available_folders):
                folder_blobs = [b for b in blobs if b.name.startswith(f"{folder}/")]
                total_size = sum(b.size or 0 for b in folder_blobs)
                print(f"  📂 {folder}: {len(folder_blobs)} files ({self._format_size(total_size)})")
            return True
        
        # Download specified folders
        total_downloaded = 0
        total_skipped = 0
        total_failed = 0
        
        for folder in folders:
            print(f"\n📂 Processing folder: {folder}")
            folder_blobs = [blob for blob in blobs if blob.name.startswith(f"{folder}/")]
            
            if not folder_blobs:
                print(f"❌ No files found in folder: {folder}")
                continue
            
            print(f"📊 Found {len(folder_blobs)} files to process")
            
            for i, blob in enumerate(folder_blobs, 1):
                print(f"\n[{i}/{len(folder_blobs)}] Processing: {blob.name}")
                result = self._download_blob(blob, force_redownload)
                
                if result == 'downloaded':
                    total_downloaded += 1
                elif result == 'skipped':
                    total_skipped += 1
                else:
                    total_failed += 1
                
                # Save progress periodically
                if i % 10 == 0:
                    self._save_progress()
        
        # Save final progress
        self._save_progress()
        
        print(f"\n{'='*60}")
        print("📊 DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Downloaded: {total_downloaded}")
        print(f"⏭️  Skipped: {total_skipped}")
        print(f"❌ Failed: {total_failed}")
        print(f"📁 Total processed: {total_downloaded + total_skipped + total_failed}")
        
        return total_failed == 0

    def _download_blob(self, blob, force_redownload=False):
        """
        Download a specific blob (file) with resume and progress tracking.
        
        Args:
            blob (Blob): Google Cloud Storage blob to download
            force_redownload (bool): If True, redownload the file
            
        Returns:
            str: 'downloaded', 'skipped', or 'failed'
        """
        # Create local file path
        local_file_path = os.path.join(self.destination_directory, blob.name)
        
        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Check if file should be downloaded
        if not force_redownload:
            if self.download_progress.get(blob.name) == blob.size:
                print(f"⏭️  Already downloaded ({self._format_size(blob.size or 0)})")
                return 'skipped'
            
            # Check if file exists and has correct size
            if os.path.exists(local_file_path):
                local_size = os.path.getsize(local_file_path)
                if local_size == blob.size:
                    print(f"⏭️  File exists with correct size ({self._format_size(blob.size or 0)})")
                    self.download_progress[blob.name] = blob.size
                    return 'skipped'
        
        # Download the blob
        try:
            print(f"⬇️  Downloading ({self._format_size(blob.size or 0)})...")
            blob.download_to_filename(local_file_path)
            
            # Verify download
            if os.path.exists(local_file_path):
                local_size = os.path.getsize(local_file_path)
                if local_size == blob.size:
                    # Update progress
                    self.download_progress[blob.name] = blob.size
                    print(f"✅ Successfully downloaded")
                    return 'downloaded'
                else:
                    print(f"❌ Size mismatch: expected {blob.size}, got {local_size}")
                    return 'failed'
            else:
                print(f"❌ File not found after download")
                return 'failed'
                
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return 'failed'
    
    def _format_size(self, size_bytes):
        """
        Format file size in human readable format
        
        Args:
            size_bytes (int): Size in bytes
            
        Returns:
            str: Formatted size string
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"

def validate_waymo_structure(waymo_root: str) -> bool:
    """
    Validate Waymo dataset structure
    
    Args:
        waymo_root: Root directory of the Waymo dataset
        
    Returns:
        True if structure is valid, False otherwise
    """
    print(f"\n{'='*60}")
    print("WAYMO DATASET STRUCTURE VALIDATION")
    print(f"{'='*60}")
    
    if not os.path.exists(waymo_root):
        print(f"❌ Dataset root directory does not exist: {waymo_root}")
        return False
    
    print(f"Validating dataset at: {waymo_root}")
    
    structure_valid = True
    total_files = 0
    
    # Check for expected folders
    for split in WAYMO_BUCKET_FOLDERS:
        split_dir = os.path.join(waymo_root, split)
        if os.path.exists(split_dir):
            print(f"✅ {split}/ directory found")
            
            # Count files in this split
            split_files = 0
            for root, dirs, files in os.walk(split_dir):
                split_files += len([f for f in files if f.lower().endswith(('.tfrecord', '.png', '.jpg', '.jpeg'))])
            
            total_files += split_files
            print(f"  📊 {split_files} files")
        else:
            print(f"⚠️  {split}/ directory missing (optional)")
    
    if total_files == 0:
        print("❌ No valid dataset files found")
        structure_valid = False
    else:
        print(f"\n📊 Total dataset files: {total_files}")
    
    if structure_valid:
        print("\n✅ Dataset structure validation passed")
    else:
        print("\n❌ Dataset structure validation failed")
    
    return structure_valid

def count_waymo_files(waymo_root: str) -> Dict[str, int]:
    """
    Count files in Waymo dataset by type and split
    
    Args:
        waymo_root: Root directory of the Waymo dataset
        
    Returns:
        Dictionary with file counts
    """
    file_counts = defaultdict(int)
    
    if not os.path.exists(waymo_root):
        return dict(file_counts)
    
    for split in WAYMO_BUCKET_FOLDERS:
        split_dir = os.path.join(waymo_root, split)
        if os.path.exists(split_dir):
            for root, dirs, files in os.walk(split_dir):
                for file in files:
                    if file.lower().endswith('.tfrecord'):
                        file_counts[f'{split}_tfrecord'] += 1
                    elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_counts[f'{split}_images'] += 1
    
    return dict(file_counts)

def diagnose_waymo_dataset(waymo_root: str) -> Dict[str, Any]:
    """
    Diagnose Waymo dataset for common issues
    
    Args:
        waymo_root: Root directory of the Waymo dataset
        
    Returns:
        Dictionary with diagnosis results
    """
    print(f"\n{'='*60}")
    print("WAYMO DATASET DIAGNOSIS")
    print(f"{'='*60}")
    
    diagnosis = {
        'status': 'unknown',
        'issues': [],
        'suggestions': [],
        'file_counts': {},
        'total_size': 0
    }
    
    if not os.path.exists(waymo_root):
        diagnosis['status'] = 'missing'
        diagnosis['issues'].append('Dataset root directory does not exist')
        diagnosis['suggestions'].append(f'Create directory: {waymo_root}')
        return diagnosis
    
    # Count files and calculate size
    file_counts = count_waymo_files(waymo_root)
    diagnosis['file_counts'] = file_counts
    
    total_files = sum(file_counts.values())
    total_size = 0
    
    for root, dirs, files in os.walk(waymo_root):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                pass
    
    diagnosis['total_size'] = total_size
    
    print(f"📊 Total files: {total_files}")
    print(f"💾 Total size: {WaymoDatasetDownloader(destination_directory='.')._format_size(total_size)}")
    
    # Check for issues
    if total_files == 0:
        diagnosis['status'] = 'empty'
        diagnosis['issues'].append('No dataset files found')
        diagnosis['suggestions'].append('Download dataset files using the download function')
    elif total_files < 100:  # Arbitrary threshold
        diagnosis['status'] = 'incomplete'
        diagnosis['issues'].append('Very few files found - dataset may be incomplete')
        diagnosis['suggestions'].append('Check if download completed successfully')
    else:
        diagnosis['status'] = 'healthy'
    
    # Print diagnosis
    if diagnosis['issues']:
        print("\n⚠️  Issues found:")
        for issue in diagnosis['issues']:
            print(f"  • {issue}")
    
    if diagnosis['suggestions']:
        print("\n💡 Suggestions:")
        for suggestion in diagnosis['suggestions']:
            print(f"  • {suggestion}")
    
    print(f"\n📋 Status: {diagnosis['status'].upper()}")
    
    return diagnosis

def visualize_waymo_sample(waymo_root: str, sample_idx: int = 0, camera: str = 'front'):
    """
    Visualize a Waymo dataset sample
    
    Args:
        waymo_root: Root directory of the Waymo dataset
        sample_idx: Index of the sample to visualize
        camera: Camera view to display ('front', 'front_left', 'front_right', 'side_left', 'side_right')
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available. Install with: pip install matplotlib")
        return
    
    print(f"\n{'='*60}")
    print("WAYMO SAMPLE VISUALIZATION")
    print(f"{'='*60}")
    
    # Look for image files in the dataset
    image_files = []
    for split in WAYMO_BUCKET_FOLDERS:
        split_dir = os.path.join(waymo_root, split)
        if os.path.exists(split_dir):
            for root, dirs, files in os.walk(split_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(root, file))
    
    if not image_files:
        print("❌ No image files found in the dataset")
        return
    
    if sample_idx >= len(image_files):
        print(f"❌ Sample index {sample_idx} out of range. Available: 0-{len(image_files)-1}")
        return
    
    image_path = image_files[sample_idx]
    print(f"📸 Displaying sample {sample_idx}: {os.path.basename(image_path)}")
    
    try:
        if CV2_AVAILABLE:
            # Use OpenCV for image loading
            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                print(f"❌ Failed to load image: {image_path}")
                return
        else:
            # Fallback to matplotlib
            image_rgb = plt.imread(image_path)
        
        # Display the image
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.title(f'Waymo Sample {sample_idx} - {camera.title()} Camera')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Visualization complete")
        
    except Exception as e:
        print(f"❌ Error visualizing sample: {str(e)}")

def show_waymo_statistics(waymo_root: str):
    """
    Show comprehensive statistics about the Waymo dataset
    
    Args:
        waymo_root: Root directory of the Waymo dataset
    """
    print(f"\n{'='*60}")
    print("WAYMO DATASET STATISTICS")
    print(f"{'='*60}")
    
    if not os.path.exists(waymo_root):
        print(f"❌ Dataset directory does not exist: {waymo_root}")
        return
    
    file_counts = count_waymo_files(waymo_root)
    
    # Calculate total size
    total_size = 0
    total_files = 0
    
    for root, dirs, files in os.walk(waymo_root):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
                total_files += 1
            except OSError:
                pass
    
    downloader = WaymoDatasetDownloader(destination_directory='.')
    
    print(f"📊 Dataset Overview:")
    print(f"  • Total files: {total_files:,}")
    print(f"  • Total size: {downloader._format_size(total_size)}")
    print(f"  • Location: {waymo_root}")
    
    if file_counts:
        print(f"\n📁 Files by Split and Type:")
        for key, count in sorted(file_counts.items()):
            split, file_type = key.rsplit('_', 1)
            print(f"  • {split.title()} {file_type}: {count:,} files")
    
    # Show directory structure
    print(f"\n🗂️  Directory Structure:")
    for split in WAYMO_BUCKET_FOLDERS:
        split_dir = os.path.join(waymo_root, split)
        if os.path.exists(split_dir):
            split_size = 0
            split_files = 0
            for root, dirs, files in os.walk(split_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        split_size += os.path.getsize(file_path)
                        split_files += 1
                    except OSError:
                        pass
            
            print(f"  ✅ {split}/ ({split_files:,} files, {downloader._format_size(split_size)})")
        else:
            print(f"  ❌ {split}/ (missing)")

def run_interactive_menu():
    """
    Run an interactive menu for Waymo dataset operations
    """
    print(f"\n{'='*60}")
    print("WAYMO DATASET UTILITY")
    print(f"{'='*60}")
    
    while True:
        print("\n🔧 Available Operations:")
        print("  1. 📥 Download dataset")
        print("  2. ✅ Validate dataset structure")
        print("  3. 🔍 Diagnose dataset")
        print("  4. 📊 Show dataset statistics")
        print("  5. 📸 Visualize sample")
        print("  6. 🚪 Exit")
        
        try:
            choice = input("\nSelect operation (1-6): ").strip()
            
            if choice == '1':
                # Download dataset
                bucket_name = input("Enter GCS bucket name (default: waymo_open_dataset_v_2_0_0): ").strip()
                if not bucket_name:
                    bucket_name = "waymo_open_dataset_v_2_0_0"
                
                dest_dir = input("Enter destination directory (default: ./waymo_data): ").strip()
                if not dest_dir:
                    dest_dir = "./waymo_data"
                
                folders = input("Enter folders to download (comma-separated, default: training): ").strip()
                if not folders:
                    folders = "training"
                folder_list = [f.strip() for f in folders.split(',')]
                
                downloader = WaymoDatasetDownloader(dest_dir)
                downloader.download_folders(bucket_name, folder_list)
                
            elif choice == '2':
                # Validate structure
                waymo_root = input("Enter Waymo dataset root directory (default: ./waymo_data): ").strip()
                if not waymo_root:
                    waymo_root = "./waymo_data"
                validate_waymo_structure(waymo_root)
                
            elif choice == '3':
                # Diagnose dataset
                waymo_root = input("Enter Waymo dataset root directory (default: ./waymo_data): ").strip()
                if not waymo_root:
                    waymo_root = "./waymo_data"
                diagnose_waymo_dataset(waymo_root)
                
            elif choice == '4':
                # Show statistics
                waymo_root = input("Enter Waymo dataset root directory (default: ./waymo_data): ").strip()
                if not waymo_root:
                    waymo_root = "./waymo_data"
                show_waymo_statistics(waymo_root)
                
            elif choice == '5':
                # Visualize sample
                waymo_root = input("Enter Waymo dataset root directory (default: ./waymo_data): ").strip()
                if not waymo_root:
                    waymo_root = "./waymo_data"
                
                sample_idx = input("Enter sample index (default: 0): ").strip()
                try:
                    sample_idx = int(sample_idx) if sample_idx else 0
                except ValueError:
                    sample_idx = 0
                
                camera = input("Enter camera view (front/front_left/front_right/side_left/side_right, default: front): ").strip()
                if not camera:
                    camera = "front"
                
                visualize_waymo_sample(waymo_root, sample_idx, camera)
                
            elif choice == '6':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def run_command_line_mode(args):
    """
    Run command line mode for backward compatibility
    
    Args:
        args: Parsed command line arguments
    """
    if args.mode == 'download':
        downloader = WaymoDatasetDownloader(args.destination or './waymo_data')
        folders = args.folders.split(',') if args.folders else ['training']
        downloader.download_folders(args.bucket or 'waymo-open-dataset-v-2-0-0', folders)
        
    elif args.mode == 'validate':
        validate_waymo_structure(args.dataset_root or './waymo_data')
        
    elif args.mode == 'diagnose':
        diagnose_waymo_dataset(args.dataset_root or './waymo_data')
        
    elif args.mode == 'count':
        show_waymo_statistics(args.dataset_root or './waymo_data')
        
    elif args.mode == 'visualize':
        visualize_waymo_sample(
            args.dataset_root or './waymo_data',
            args.sample_idx or 0,
            args.camera or 'front'
        )
    else:
        print(f"❌ Unknown mode: {args.mode}")

def main():
    """
    Main function with command-line interface and interactive menu
    """
    parser = argparse.ArgumentParser(
        description='Waymo Open Dataset v2.0 Utility - Download, validate, and visualize Waymo dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive menu (recommended)
  python downloadwaymo2.py
  
  # Command line usage
  python downloadwaymo2.py --mode download --bucket waymo_open_dataset_v_2_0_0 --folders training,validation
  python downloadwaymo2.py --mode validate --dataset-root ./waymo_data
  python downloadwaymo2.py --mode diagnose --dataset-root ./waymo_data
  python downloadwaymo2.py --mode count --dataset-root ./waymo_data
  python downloadwaymo2.py --mode visualize --dataset-root ./waymo_data --sample-idx 0
"""
    )
    
    # Optional arguments for command line mode
    parser.add_argument('--mode', default='download',
                       choices=['download', 'validate', 'diagnose', 'count', 'visualize'],
                       help='Operation mode (if not specified, interactive menu will be shown)')
    parser.add_argument('--bucket', 
                       default='waymo_open_dataset_v_2_0_0',
                       help='GCS bucket name (default: waymo_open_dataset_v_2_0_0)')
    parser.add_argument('--folders', 
                       default='training',
                       help='Comma-separated list of folders to download (default: training)')
    parser.add_argument('--destination', 
                       default='/DATA10T/Datasets/waymo_data',
                       help='Destination directory for downloads (default: ./waymo_data)')
    parser.add_argument('--dataset-root', 
                       default='/DATA10T/Datasets/waymo_data',
                       help='Root directory of the Waymo dataset (default: ./waymo_data)')
    parser.add_argument('--sample-idx', 
                       type=int, default=0,
                       help='Sample index for visualization (default: 0)')
    parser.add_argument('--camera', 
                       default='front',
                       choices=['front', 'front_left', 'front_right', 'side_left', 'side_right'],
                       help='Camera view for visualization (default: front)')
    
    args = parser.parse_args()
    
    # Print header
    print(f"\n{'='*60}")
    print("WAYMO OPEN DATASET v2.0 UTILITY")
    print(f"{'='*60}")
    print("Enhanced dataset downloader with validation and visualization")
    
    # Check dependencies
    missing_deps = []
    if not CV2_AVAILABLE:
        missing_deps.append("opencv-python")
    if not MATPLOTLIB_AVAILABLE:
        missing_deps.append("matplotlib")
    if not NUMPY_AVAILABLE:
        missing_deps.append("numpy")
    
    if missing_deps:
        print(f"\n⚠️  Optional dependencies missing: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
    
    try:
        if args.mode:
            # Command line mode
            run_command_line_mode(args)
        else:
            # Interactive menu mode
            run_interactive_menu()
            
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

# Prerequisites:
# 1. Install Google Cloud Storage library:
# pip install google-cloud-storage
#
# 2. Set up Google Cloud Authentication:
# - Create a service account in Google Cloud Console
# - Download the JSON key file
# - Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
#   export GOOGLE_APPLICATION_CREDENTIALS="/mnt/d/sjsu-rf-ohana-f11181b36a10.json"

#mkdir waymodata && gsutil -m cp -r gs://waymo_open_dataset_v_2_0_1/validation ./waymodata