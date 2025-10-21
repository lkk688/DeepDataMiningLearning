#!/usr/bin/env python3
"""
WaymoAllDataset - Comprehensive Waymo Dataset for Multi-Modal Learning
=====================================================================

This dataset extends the Waymo2DDataset to support multiple data modalities:
- '2d' mode: Returns camera images and 2D bounding boxes
- '3d' mode: Returns LiDAR point clouds and 3D bounding boxes  
- 'all' mode: Returns all available Waymo data (images, LiDAR, boxes, etc.)

The dataset utilizes utility functions from waymo_data_utils.py for data loading
and processing, ensuring consistency and leveraging optimized implementations.

"""

import os
import sys
import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings

# Add the detection3d directory to path for waymo_data_utils import
sys.path.append(os.path.dirname(__file__))

try:
    from waymo_data_utils import (
        read_camera_images, read_camera_boxes, read_lidar_data, read_lidar_boxes,
        read_lidar_camera_projection, WAYMO_OBJECT_TYPES, WAYMO_CAMERA_NAMES,
        WAYMO_LIDAR_NAMES
    )
except ImportError as e:
    warnings.warn(f"Could not import waymo_data_utils: {e}")
    # Fallback imports or definitions can be added here

# Add the detection directory to path for dataset imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'detection'))

# Import helper functions from the existing dataset
from dataset_waymov3_1 import (
    _decode_image_rgb, _guess_image_column_name_from_schema, 
    _collect_time_order_index, KEY_SEG, KEY_TS, KEY_CAM,
    BOX_CX, BOX_CY, BOX_W, BOX_H, BOX_TYPE
)


class WaymoAllDataset(Dataset):
    """
    Comprehensive Waymo Dataset supporting multiple data modalities.
    
    This dataset can operate in three modes:
    1. '2d': Returns camera images and 2D bounding boxes (similar to Waymo2DDataset)
    2. '3d': Returns LiDAR point clouds and 3D bounding boxes
    3. 'all': Returns all available Waymo data for multi-modal learning
    
    The dataset builds a unified frame index that aligns data across all modalities
    based on segment name and timestamp, ensuring temporal consistency.
    
    Args:
        root_dir (str): Root directory containing Waymo dataset splits
        split (str): Dataset split ('training', 'validation', 'testing')
        mode (str): Data mode ('2d', '3d', 'all')
        max_frames (int, optional): Maximum number of frames to load
        transform (callable, optional): Transform to apply to images
        lidar_transform (callable, optional): Transform to apply to LiDAR data
        
    Returns:
        Depending on mode:
        - '2d': (image_tensor, target_dict) where target contains 2D boxes
        - '3d': (lidar_data, target_dict) where target contains 3D boxes  
        - 'all': (data_dict, target_dict) where data_dict contains all modalities
    """
    
    def __init__(self, 
                 root_dir: str, 
                 split: str = "training",
                 mode: str = "2d",
                 max_frames: int = None,
                 transform=None,
                 lidar_transform=None):
        
        self.root_dir = root_dir
        self.split = split
        self.mode = mode.lower()
        self.transform = transform
        self.lidar_transform = lidar_transform
        
        # Validate mode
        if self.mode not in ['2d', '3d', 'all']:
            raise ValueError(f"Mode must be '2d', '3d', or 'all', got '{mode}'")
        
        # Define data directories
        self.data_dirs = {
            'camera_image': os.path.join(root_dir, split, "camera_image"),
            'camera_box': os.path.join(root_dir, split, "camera_box"),
            'lidar': os.path.join(root_dir, split, "lidar"),
            'lidar_box': os.path.join(root_dir, split, "lidar_box"),
            'lidar_camera_projection': os.path.join(root_dir, split, "lidar_camera_projection"),
            'camera_calibration': os.path.join(root_dir, split, "camera_calibration"),
            'lidar_calibration': os.path.join(root_dir, split, "lidar_calibration"),
        }
        
        # Check required directories based on mode
        self._validate_directories()
        
        # Build unified frame index
        self.frame_index = []
        self.image_col_by_file = {}
        self._build_frame_index(max_frames)
        
        print(f"✅ WaymoAllDataset initialized in '{self.mode}' mode with {len(self.frame_index)} frames")
    
    def _validate_directories(self):
        """Validate that required directories exist based on the selected mode."""
        required_dirs = []
        
        if self.mode in ['2d', 'all']:
            required_dirs.extend(['camera_image', 'camera_box'])
        
        if self.mode in ['3d', 'all']:
            required_dirs.extend(['lidar', 'lidar_box'])
        
        for dir_name in required_dirs:
            dir_path = self.data_dirs[dir_name]
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
    
    def _build_frame_index(self, max_frames: int = None):
        """
        Build a unified frame index across all data modalities.
        
        This creates a time-ordered list of frames that exist across the required
        modalities, ensuring data alignment for multi-modal learning.
        """
        print(f"🔍 Building frame index for mode '{self.mode}'...")
        
        # Start with camera data as the base (most frames available)
        if self.mode in ['2d', 'all']:
            self._build_camera_frame_index(max_frames)
        elif self.mode == '3d':
            self._build_lidar_frame_index(max_frames)
    
    def _build_camera_frame_index(self, max_frames: int = None):
        """Build frame index based on camera data (similar to Waymo2DDataset)."""
        img_dir = self.data_dirs['camera_image']
        box_dir = self.data_dirs['camera_box']
        
        # Get matching shard files
        all_img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".parquet")])
        valid_files = [f for f in all_img_files if os.path.exists(os.path.join(box_dir, f))]
        
        if not valid_files:
            raise RuntimeError("No matching shards found between camera_image/ and camera_box/.")
        
        total = 0
        for fname in valid_files:
            img_path = os.path.join(img_dir, fname)
            pf = pq.ParquetFile(img_path)
            
            # Find image column name
            img_col = _guess_image_column_name_from_schema(pf)
            self.image_col_by_file[fname] = img_col
            
            # Build time-ordered index
            time_index = _collect_time_order_index(pf)
            for (rg, rg_row, ts, seg, cam) in time_index:
                self.frame_index.append({
                    'file': fname,
                    'row_group': rg,
                    'row_index': rg_row,
                    'timestamp': ts,
                    'segment': seg,
                    'camera_id': cam,
                    'type': 'camera'
                })
                total += 1
                if max_frames is not None and total >= max_frames:
                    break
            if max_frames is not None and total >= max_frames:
                break
    
    def _build_lidar_frame_index(self, max_frames: int = None):
        """Build frame index based on LiDAR data."""
        lidar_dir = self.data_dirs['lidar']
        lidar_box_dir = self.data_dirs['lidar_box']
        
        # Get matching shard files
        all_lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".parquet")])
        valid_files = [f for f in all_lidar_files if os.path.exists(os.path.join(lidar_box_dir, f))]
        
        if not valid_files:
            raise RuntimeError("No matching shards found between lidar/ and lidar_box/.")
        
        total = 0
        for fname in valid_files:
            lidar_path = os.path.join(lidar_dir, fname)
            pf = pq.ParquetFile(lidar_path)
            
            # Read metadata to build index
            for rg_idx in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg_idx, columns=[KEY_SEG, KEY_TS, 'key.laser_name'])
                df = tbl.to_pandas()
                
                for row_idx, row in df.iterrows():
                    self.frame_index.append({
                        'file': fname,
                        'row_group': rg_idx,
                        'row_index': row_idx,
                        'timestamp': int(row[KEY_TS]),
                        'segment': row[KEY_SEG],
                        'laser_id': int(row['key.laser_name']),
                        'type': 'lidar'
                    })
                    total += 1
                    if max_frames is not None and total >= max_frames:
                        break
                if max_frames is not None and total >= max_frames:
                    break
            if max_frames is not None and total >= max_frames:
                break
        
        # Sort by timestamp for temporal consistency
        self.frame_index.sort(key=lambda x: x['timestamp'])
    
    def __len__(self) -> int:
        return len(self.frame_index)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, Dict], Tuple[Dict, Dict]]:
        """
        Get data sample based on the selected mode.
        
        Returns:
            - '2d' mode: (image_tensor, target_dict)
            - '3d' mode: (lidar_data, target_dict)  
            - 'all' mode: (data_dict, target_dict)
        """
        if self.mode == '2d':
            return self._get_2d_sample(idx)
        elif self.mode == '3d':
            return self._get_3d_sample(idx)
        elif self.mode == 'all':
            return self._get_all_sample(idx)
    
    def _get_2d_sample(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get 2D sample (image + 2D boxes) - similar to Waymo2DDataset."""
        frame_info = self.frame_index[idx]
        fname = frame_info['file']
        rg = frame_info['row_group']
        rg_row = frame_info['row_index']
        ts = frame_info['timestamp']
        seg = frame_info['segment']
        cam = frame_info['camera_id']
        
        # Load image
        img_path = os.path.join(self.data_dirs['camera_image'], fname)
        img_pf = pq.ParquetFile(img_path)
        img_col = self.image_col_by_file[fname]
        
        tbl = img_pf.read_row_group(rg, columns=[img_col, KEY_SEG, KEY_TS, KEY_CAM])
        df = tbl.to_pandas()
        row = df.iloc[rg_row]
        
        img_bytes = row[img_col]
        img_rgb = _decode_image_rgb(img_bytes)
        h, w, _ = img_rgb.shape
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        
        # Load 2D boxes
        box_path = os.path.join(self.data_dirs['camera_box'], fname)
        box_pf = pq.ParquetFile(box_path)
        box_tbl = box_pf.read_row_group(0, columns=[KEY_SEG, KEY_TS, KEY_CAM, BOX_CX, BOX_CY, BOX_W, BOX_H, BOX_TYPE])
        box_df = box_tbl.to_pandas()
        
        # Filter boxes for this frame
        frame_boxes = box_df[
            (box_df[KEY_SEG] == seg) &
            (box_df[KEY_TS] == ts) &
            (box_df[KEY_CAM] == cam)
        ]
        
        boxes = []
        labels = []
        
        for _, b in frame_boxes.iterrows():
            cx, cy = float(b[BOX_CX]), float(b[BOX_CY])
            bw, bh = float(b[BOX_W]), float(b[BOX_H])
            
            xmin = max(0.0, cx - bw / 2.0)
            ymin = max(0.0, cy - bh / 2.0)
            xmax = min(float(w), cx + bw / 2.0)
            ymax = min(float(h), cy + bh / 2.0)
            
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(b[BOX_TYPE]))
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "segment": seg,
            "timestamp": ts,
            "camera_id": cam,
        }
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, target
    
    def _get_3d_sample(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get 3D sample (LiDAR point cloud + 3D boxes)."""
        frame_info = self.frame_index[idx]
        fname = frame_info['file']
        rg = frame_info['row_group']
        rg_row = frame_info['row_index']
        ts = frame_info['timestamp']
        seg = frame_info['segment']
        laser_id = frame_info['laser_id']
        
        # Load LiDAR data
        lidar_path = os.path.join(self.data_dirs['lidar'], fname)
        lidar_data = self._load_lidar_frame(lidar_path, rg, rg_row)
        
        # Load 3D boxes
        lidar_box_path = os.path.join(self.data_dirs['lidar_box'], fname)
        boxes_3d = self._load_lidar_boxes_frame(lidar_box_path, seg, ts)
        
        target = {
            "boxes_3d": boxes_3d['boxes'] if boxes_3d else [],
            "labels_3d": boxes_3d['labels'] if boxes_3d else [],
            "frame_id": torch.tensor([idx], dtype=torch.int64),
            "segment": seg,
            "timestamp": ts,
            "laser_id": laser_id,
        }
        
        if self.lidar_transform:
            lidar_data = self.lidar_transform(lidar_data)
        
        return lidar_data, target
    
    def _get_all_sample(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get all available data for multi-modal learning."""
        frame_info = self.frame_index[idx]
        
        # Start with 2D data as base
        img_tensor, target_2d = self._get_2d_sample(idx)
        
        # Try to load corresponding 3D data
        lidar_data = None
        target_3d = None
        
        try:
            # Find corresponding LiDAR frame by timestamp and segment
            seg = frame_info['segment']
            ts = frame_info['timestamp']
            
            lidar_data, target_3d = self._find_and_load_lidar_data(seg, ts)
        except Exception as e:
            print(f"Warning: Could not load LiDAR data for frame {idx}: {e}")
        
        # Combine all data
        data_dict = {
            'image': img_tensor,
            'lidar': lidar_data,
            'has_lidar': lidar_data is not None,
        }
        
        # Combine targets
        target_dict = {
            **target_2d,
            'boxes_3d': target_3d['boxes_3d'] if target_3d else [],
            'labels_3d': target_3d['labels_3d'] if target_3d else [],
            'has_3d_boxes': target_3d is not None,
        }
        
        return data_dict, target_dict
    
    def _load_lidar_frame(self, lidar_path: str, rg: int, rg_row: int) -> Dict[str, Any]:
        """Load LiDAR range image data for a specific frame."""
        pf = pq.ParquetFile(lidar_path)
        
        # Read the specific row
        tbl = pf.read_row_group(rg, columns=[
            '[LiDARComponent].range_image_return1.values',
            '[LiDARComponent].range_image_return1.shape',
            '[LiDARComponent].range_image_return2.values', 
            '[LiDARComponent].range_image_return2.shape'
        ])
        df = tbl.to_pandas()
        row = df.iloc[rg_row]
        
        # Extract range images
        return1_values = row['[LiDARComponent].range_image_return1.values']
        return1_shape = row['[LiDARComponent].range_image_return1.shape']
        return2_values = row['[LiDARComponent].range_image_return2.values']
        return2_shape = row['[LiDARComponent].range_image_return2.shape']
        
        lidar_data = {
            'return1': np.array(return1_values).reshape(return1_shape) if return1_values is not None else None,
            'return2': np.array(return2_values).reshape(return2_shape) if return2_values is not None else None,
            'shapes': [return1_shape, return2_shape]
        }
        
        return lidar_data
    
    def _load_lidar_boxes_frame(self, box_path: str, segment: str, timestamp: int) -> Dict[str, Any]:
        """Load 3D boxes for a specific frame."""
        try:
            pf = pq.ParquetFile(box_path)
            
            # Read all boxes and filter by segment/timestamp
            tbl = pf.read_row_group(0, columns=[
                KEY_SEG, KEY_TS,
                '[LiDARBoxComponent].box.center.x',
                '[LiDARBoxComponent].box.center.y', 
                '[LiDARBoxComponent].box.center.z',
                '[LiDARBoxComponent].box.size.x',
                '[LiDARBoxComponent].box.size.y',
                '[LiDARBoxComponent].box.size.z',
                '[LiDARBoxComponent].box.heading',
                '[LiDARBoxComponent].type'
            ])
            df = tbl.to_pandas()
            
            # Filter for this frame
            frame_boxes = df[
                (df[KEY_SEG] == segment) &
                (df[KEY_TS] == timestamp)
            ]
            
            boxes = []
            labels = []
            
            for _, box in frame_boxes.iterrows():
                center = [
                    float(box['[LiDARBoxComponent].box.center.x']),
                    float(box['[LiDARBoxComponent].box.center.y']),
                    float(box['[LiDARBoxComponent].box.center.z'])
                ]
                size = [
                    float(box['[LiDARBoxComponent].box.size.x']),
                    float(box['[LiDARBoxComponent].box.size.y']),
                    float(box['[LiDARBoxComponent].box.size.z'])
                ]
                heading = float(box['[LiDARBoxComponent].box.heading'])
                
                # Store as [x, y, z, l, w, h, heading] format
                box_3d = center + size + [heading]
                boxes.append(box_3d)
                labels.append(int(box['[LiDARBoxComponent].type']))
            
            return {
                'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 7), dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
            }
            
        except Exception as e:
            print(f"Warning: Could not load 3D boxes: {e}")
            return None
    
    def _find_and_load_lidar_data(self, segment: str, timestamp: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Find and load LiDAR data matching the given segment and timestamp."""
        # This is a simplified implementation - in practice, you might want to
        # build a more efficient lookup table during initialization
        lidar_dir = self.data_dirs['lidar']
        lidar_box_dir = self.data_dirs['lidar_box']
        
        # Search through LiDAR files for matching frame
        for fname in os.listdir(lidar_dir):
            if not fname.endswith('.parquet'):
                continue
                
            lidar_path = os.path.join(lidar_dir, fname)
            pf = pq.ParquetFile(lidar_path)
            
            # Check if this file contains our target frame
            for rg_idx in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg_idx, columns=[KEY_SEG, KEY_TS])
                df = tbl.to_pandas()
                
                matching_rows = df[
                    (df[KEY_SEG] == segment) &
                    (df[KEY_TS] == timestamp)
                ]
                
                if len(matching_rows) > 0:
                    # Found matching frame
                    row_idx = matching_rows.index[0]
                    lidar_data = self._load_lidar_frame(lidar_path, rg_idx, row_idx)
                    
                    # Load corresponding 3D boxes
                    lidar_box_path = os.path.join(lidar_box_dir, fname)
                    boxes_3d = self._load_lidar_boxes_frame(lidar_box_path, segment, timestamp)
                    
                    target_3d = {
                        'boxes_3d': boxes_3d['boxes'] if boxes_3d else [],
                        'labels_3d': boxes_3d['labels'] if boxes_3d else [],
                    }
                    
                    return lidar_data, target_3d
        
        raise ValueError(f"No LiDAR data found for segment {segment}, timestamp {timestamp}")
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata information for a sample."""
        frame_info = self.frame_index[idx]
        return {
            'index': idx,
            'mode': self.mode,
            'segment': frame_info['segment'],
            'timestamp': frame_info['timestamp'],
            'file': frame_info['file'],
            **frame_info
        }


def test_waymo_all_dataset():
    """Test function to verify the dataset works correctly in all modes."""
    # This would be called from a separate test script
    root_dir = "/data/Datasets/waymodata/"  # Update with actual path
    
    print("Testing WaymoAllDataset in different modes...")
    
    # Test 2D mode
    try:
        dataset_2d = WaymoAllDataset(root_dir, split="training", mode="2d", max_frames=10)
        img, target = dataset_2d[0]
        print(f"✅ 2D mode: Image shape {img.shape}, {len(target['boxes'])} boxes")
    except Exception as e:
        print(f"❌ 2D mode failed: {e}")
    
    # Test 3D mode  
    try:
        dataset_3d = WaymoAllDataset(root_dir, split="training", mode="3d", max_frames=10)
        lidar, target = dataset_3d[0]
        print(f"✅ 3D mode: LiDAR data loaded, {len(target['boxes_3d'])} 3D boxes")
    except Exception as e:
        print(f"❌ 3D mode failed: {e}")
    
    # Test all mode
    try:
        dataset_all = WaymoAllDataset(root_dir, split="training", mode="all", max_frames=10)
        data, target = dataset_all[0]
        print(f"✅ All mode: Image {data['image'].shape}, LiDAR available: {data['has_lidar']}")
    except Exception as e:
        print(f"❌ All mode failed: {e}")


if __name__ == "__main__":
    test_waymo_all_dataset()