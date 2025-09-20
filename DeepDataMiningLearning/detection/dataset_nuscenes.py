#!/usr/bin/env python3
"""
Simplified NuScenes Dataset for PyTorch Object Detection

This module provides a streamlined NuScenes dataset implementation that leverages
utility functions from the main nuscenes.py script to reduce code duplication
and complexity.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from enum import IntEnum
import argparse
import sys

# Import utility functions from nuscenes.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
try:
    from nuscenes import (
        # 3D to 2D projection utilities
        project_3d_box_to_2d,
        get_2d_bbox_from_3d_projection,
        quaternion_to_rotation_matrix,
        draw_2d_bbox,
        draw_2d_bbox_from_3d,
        
        # Data loading and validation utilities
        diagnose_dataset_issues,
        check_nuscenes_data_structure,
        load_nuscenes_data,
        box_in_image,
        
        # Constants and structures
        NUSCENES_STRUCTURE,
        REQUIRED_ANNOTATION_FILES,
        BoxVisibility
    )
    print("✅ Successfully imported nuscenes utilities")
except ImportError as e:
    print(f"Warning: Could not import nuscenes utilities: {e}")
    # Fallback definitions if import fails
    NUSCENES_STRUCTURE = {
        'samples': ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP'],
        'sweeps': ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP']
    }
    REQUIRED_ANNOTATION_FILES = ['sample.json', 'sample_data.json', 'sample_annotation.json', 'category.json']
    
    class BoxVisibility(IntEnum):
        ALL = 0
        ANY = 1
        NONE = 2

# NuScenes category mapping (simplified)
NUSCENES_CATEGORIES = {
    'vehicle.car': 0,
    'vehicle.truck': 1,
    'vehicle.bus.bendy': 2,
    'vehicle.bus.rigid': 2,
    'vehicle.trailer': 3,
    'vehicle.construction': 4,
    'human.pedestrian.adult': 5,
    'human.pedestrian.child': 5,
    'human.pedestrian.wheelchair': 5,
    'human.pedestrian.stroller': 5,
    'human.pedestrian.personal_mobility': 5,
    'human.pedestrian.police_officer': 5,
    'human.pedestrian.construction_worker': 5,
    'vehicle.motorcycle': 6,
    'vehicle.bicycle': 7,
    'movable_object.trafficcone': 8,
    'movable_object.barrier': 9
}

CATEGORY_NAMES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
]

DEFAULT_NUSCENES_ROOT = "/DATA10T/Datasets/nuScenes/v1.0-trainval"


class NuScenesDataset(Dataset):
    """
    Simplified NuScenes Dataset for PyTorch object detection training.
    
    This dataset leverages utility functions from nuscenes.py to reduce
    code duplication and complexity while maintaining full functionality.
    """
    
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 camera_types: List[str] = None,
                 transform: Optional[transforms.Compose] = None,
                 target_transform: Optional[transforms.Compose] = None,
                 load_lidar: bool = False,
                 max_samples: Optional[int] = None,
                 validate_on_init: bool = True):
        """
        Initialize simplified NuScenes dataset.
        
        Args:
            root_dir: Root directory of NuScenes dataset
            split: Dataset split ('train', 'val', 'test')
            camera_types: List of camera types to use (default: ['CAM_FRONT'])
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            load_lidar: Whether to load LiDAR data
            max_samples: Maximum number of samples to load (for debugging)
            validate_on_init: Whether to validate dataset structure on initialization
        """
        self.root_dir = root_dir
        self.split = split
        self.camera_types = camera_types or ['CAM_FRONT']
        self.transform = transform
        self.target_transform = target_transform
        self.load_lidar = load_lidar
        self.max_samples = max_samples
        
        # Dataset paths
        self.samples_dir = os.path.join(root_dir, 'samples')
        self.annotation_dir = os.path.join(root_dir, 'v1.0-trainval')
        
        # Validate dataset structure using utility function
        if validate_on_init:
            if not check_nuscenes_data_structure(root_dir):
                raise ValueError(f"Invalid NuScenes dataset structure at {root_dir}")
        
        # Load annotation data using utility function
        print("Loading NuScenes annotation data...")
        try:
            self.nuscenes_data = load_nuscenes_data(root_dir)
            print(f"✅ Loaded {len(self.nuscenes_data['samples'])} samples")
        except Exception as e:
            print(f"❌ Failed to load annotation data: {e}")
            raise
        
        # Create lookup dictionaries for efficient access
        self._create_lookup_dicts()
        
        # Filter and prepare samples for the specified split
        self._prepare_samples()
        
        print(f"✅ Dataset initialized with {len(self.samples)} samples for {split} split")
    
    def _create_lookup_dicts(self):
        """Create lookup dictionaries for efficient data access."""
        # Create token-based lookups
        self.sample_data_lookup = {sd['token']: sd for sd in self.nuscenes_data['sample_data']}
        self.calibrated_sensor_lookup = {cs['token']: cs for cs in self.nuscenes_data['calibrated_sensors']}
        self.ego_pose_lookup = {ep['token']: ep for ep in self.nuscenes_data['ego_poses']}
        self.sensor_lookup = {s['token']: s for s in self.nuscenes_data['sensors']}
        self.category_lookup = {c['token']: c['name'] for c in self.nuscenes_data['categories']}
        self.instance_lookup = {i['token']: i for i in self.nuscenes_data['instances']}
        
        print(f"✅ Created lookup dictionaries for efficient data access")
    
    def _prepare_samples(self):
        """Prepare and filter samples for the dataset."""
        self.samples = []
        
        # Simple split logic - use first 80% for train, rest for val
        all_samples = self.nuscenes_data['samples']
        split_idx = int(0.8 * len(all_samples))
        
        if self.split == 'train':
            selected_samples = all_samples[:split_idx]
        elif self.split == 'val':
            selected_samples = all_samples[split_idx:]
        else:  # test or other
            selected_samples = all_samples
        
        # Filter samples that have the required camera data
        for sample in selected_samples:
            # Check if sample has required camera data
            # Handle both test dataset structure (with 'data' key) and real NuScenes structure
            if 'data' in sample:
                # Test dataset structure
                sample_data_tokens = [sample['data'][cam] for cam in self.camera_types 
                                    if cam in sample['data']]
                
                if len(sample_data_tokens) == len(self.camera_types):
                    self.samples.append(sample)
            else:
                # Real NuScenes structure - check if sample has camera data
                sample_data_list = [sd for sd in self.nuscenes_data['sample_data'] 
                                  if sd['sample_token'] == sample['token']]
                
                # Check if we have data for all required cameras
                available_cameras = set()
                for sd in sample_data_list:
                    try:
                        sensor = self.sensor_lookup[self.calibrated_sensor_lookup[sd['calibrated_sensor_token']]['sensor_token']]
                        if sensor['channel'] in self.camera_types:
                            available_cameras.add(sensor['channel'])
                    except KeyError:
                        continue
                
                if len(available_cameras) >= len(self.camera_types):
                    self.samples.append(sample)
                
            # Apply max_samples limit if specified
            if self.max_samples and len(self.samples) >= self.max_samples:
                break
        
        print(f"✅ Prepared {len(self.samples)} samples for {self.split} split")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, target_dict)
        """
        sample = self.samples[idx]
        
        # Handle different NuScenes dataset structures
        primary_camera = self.camera_types[0]
        
        if 'data' in sample:
            # Test dataset structure with direct 'data' key
            sample_data_token = sample['data'][primary_camera]
        else:
            # Real NuScenes structure - find sample_data by sample_token and camera
            sample_data_list = [sd for sd in self.nuscenes_data['sample_data'] 
                              if sd['sample_token'] == sample['token']]
            
            # Find the camera data for the primary camera
            camera_sample_data = None
            for sd in sample_data_list:
                sensor = self.sensor_lookup[self.calibrated_sensor_lookup[sd['calibrated_sensor_token']]['sensor_token']]
                if sensor['channel'] == primary_camera:
                    camera_sample_data = sd
                    break
            
            if camera_sample_data is None:
                raise ValueError(f"No {primary_camera} data found for sample {sample['token']}")
            
            sample_data = camera_sample_data
            sample_data_token = sample_data['token']
        
        if 'data' in sample:
            sample_data = self.sample_data_lookup[sample_data_token]
        
        # Load image
        image_path = os.path.join(self.root_dir, sample_data['filename'])
        image = self.get_image(image_path)
        
        # Get target annotations
        target = self.get_target(sample['token'], sample_data)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target
    
    def get_image(self, image_path: str) -> Image.Image:
        """Load and return an image."""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (1600, 900), color='black')
    
    def get_target(self, sample_token: str, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get target annotations for a sample.
        
        Args:
            sample_token: Token of the sample
            sample_data: Sample data dictionary
            
        Returns:
            Dictionary containing target annotations
        """
        # Get all annotations for this sample
        sample_annotations = [ann for ann in self.nuscenes_data['sample_annotations'] 
                            if ann['sample_token'] == sample_token]
        
        # Get calibration and pose information
        calibrated_sensor = self.calibrated_sensor_lookup[sample_data['calibrated_sensor_token']]
        ego_pose = self.ego_pose_lookup[sample_data['ego_pose_token']]
        
        # Process annotations
        boxes = []
        labels = []
        
        for ann in sample_annotations:
            # Get category
            instance = self.instance_lookup[ann['instance_token']]
            category_name = self.category_lookup[instance['category_token']]
            
            # Map to simplified category
            if category_name in NUSCENES_CATEGORIES:
                label = NUSCENES_CATEGORIES[category_name]
                
                # Project 3D box to 2D using proper 3D to 2D projection
                bbox_2d = self._get_2d_bbox_from_3d(ann, calibrated_sensor, ego_pose)
                
                if bbox_2d is not None:
                    boxes.append(bbox_2d)
                    labels.append(label)
        
        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([hash(sample_token) % 1000000]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        return target
    
    def _get_2d_bbox_from_3d(self, annotation: Dict[str, Any], 
                            calibrated_sensor: Dict[str, Any],
                            ego_pose: Dict[str, Any]) -> Optional[List[float]]:
        """
        Project 3D bounding box to 2D using proper coordinate transformation.
        
        This function uses the utility functions from nuscenes.py for accurate
        3D to 2D projection following NuScenes coordinate system conventions.
        
        Args:
            annotation: 3D annotation with translation, size, and rotation
            calibrated_sensor: Camera calibration and pose information
            ego_pose: Ego vehicle pose information
            
        Returns:
            [x_min, y_min, x_max, y_max] bounding box or None if projection fails
        """
        try:
            # Extract 3D box parameters from annotation
            center_3d = np.array(annotation['translation'])  # [x, y, z] in global coordinates
            size_3d = np.array(annotation['size'])          # [width, length, height] in meters
            rotation_3d = np.array(annotation['rotation'])   # [w, x, y, z] quaternion
            
            # Extract camera parameters
            cam_translation = np.array(calibrated_sensor['translation'])
            cam_rotation = np.array(calibrated_sensor['rotation'])
            camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
            
            # Extract ego pose parameters
            ego_translation = np.array(ego_pose['translation'])
            ego_rotation = np.array(ego_pose['rotation'])
            
            # Project 3D bounding box to 2D using utility function
            projection_result = project_3d_box_to_2d(
                center_3d, size_3d, rotation_3d,
                cam_translation, cam_rotation, camera_intrinsic,
                ego_translation, ego_rotation,
                debug=False
            )
            
            # Check if projection was successful
            if projection_result[0] is None:
                return None
                
            corners_2d, corners_3d_cam = projection_result
            
            # Check visibility using box_in_image function
            if 'box_in_image' in globals():
                # Standard image size for NuScenes
                img_size = (1600, 900)  # (width, height)
                if not box_in_image(corners_3d_cam, corners_2d.T, camera_intrinsic, img_size, BoxVisibility.ANY):
                    return None
            
            # Get 2D bounding box from projected corners
            bbox_2d = get_2d_bbox_from_3d_projection(corners_2d)
            
            if bbox_2d is None:
                return None
            
            # Validate bounding box
            x_min, y_min, x_max, y_max = bbox_2d
            
            # Image dimensions (standard NuScenes camera resolution)
            img_width, img_height = 1600, 900
            
            # Filter out boxes that are completely outside image bounds
            if x_max < 0 or y_max < 0 or x_min > img_width or y_min > img_height:
                return None
            
            # Clip bounding box to image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)
            
            # Check if clipped box is still valid
            box_width = x_max - x_min
            box_height = y_max - y_min
            
            # Filter out boxes that are too small after clipping
            if box_width < 10 or box_height < 10:
                return None
            
            # Filter out boxes that are unreasonably large (>90% of image)
            if box_width > img_width * 0.9 or box_height > img_height * 0.9:
                return None
            
            return [x_min, y_min, x_max, y_max]
            
        except Exception as e:
            print(f"Error projecting 3D box to 2D: {e}")
            return None
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a sample."""
        sample = self.samples[idx]
        
        # Get comprehensive sample data using utility functions
        try:
            # Get basic sample information
            sample_info = {
                'sample_token': sample['token'],
                'timestamp': sample['timestamp'],
                'scene_token': sample['scene_token'],
                'sample_idx': idx
            }
            
            # Get sample data for each camera
            camera_data = {}
            
            if 'data' in sample:
                # Test dataset structure
                for camera_type in self.camera_types:
                    # Find sample_data for this camera
                    sample_data_token = None
                    for key, token in sample['data'].items():
                        if key == camera_type:
                            sample_data_token = token
                            break
                    
                    if sample_data_token:
                        # Get sample_data record
                        sample_data = self.sample_data_lookup.get(sample_data_token)
                        if sample_data:
                            # Get calibrated sensor and ego pose
                            calibrated_sensor = self.calibrated_sensor_lookup.get(sample_data['calibrated_sensor_token'])
                            ego_pose = self.ego_pose_lookup.get(sample_data['ego_pose_token'])
                            
                            camera_data[camera_type] = {
                                'filename': sample_data['filename'],
                                'timestamp': sample_data['timestamp'],
                                'calibrated_sensor': calibrated_sensor,
                                'ego_pose': ego_pose
                            }
            else:
                # Real NuScenes structure
                sample_data_list = [sd for sd in self.nuscenes_data['sample_data'] 
                                  if sd['sample_token'] == sample['token']]
                
                for sd in sample_data_list:
                    try:
                        sensor = self.sensor_lookup[self.calibrated_sensor_lookup[sd['calibrated_sensor_token']]['sensor_token']]
                        camera_type = sensor['channel']
                        
                        if camera_type in self.camera_types:
                            # Get calibrated sensor and ego pose
                            calibrated_sensor = self.calibrated_sensor_lookup.get(sd['calibrated_sensor_token'])
                            ego_pose = self.ego_pose_lookup.get(sd['ego_pose_token'])
                            
                            camera_data[camera_type] = {
                                'filename': sd['filename'],
                                'timestamp': sd['timestamp'],
                                'calibrated_sensor': calibrated_sensor,
                                'ego_pose': ego_pose
                            }
                    except KeyError:
                        continue
            
            sample_info['camera_data'] = camera_data
            
            # Get annotations for this sample
            sample_annotations = []
            for ann_token in sample.get('anns', []):
                annotation = self.sample_annotation_lookup.get(ann_token)
                if annotation:
                    sample_annotations.append(annotation)
            
            sample_info['annotations'] = sample_annotations
            sample_info['num_annotations'] = len(sample_annotations)
            
            return sample_info
            
        except Exception as e:
            print(f"Warning: Error getting sample info for idx {idx}: {e}")
            # Fallback to basic info
            return {
                'sample_token': sample['token'],
                'timestamp': sample['timestamp'],
                'scene_token': sample['scene_token'],
                'sample_idx': idx,
                'error': str(e)
            }


def create_nuscenes_transforms(train: bool = True) -> transforms.Compose:
    """Create standard transforms for NuScenes dataset."""
    if train:
        return transforms.Compose([
            transforms.Resize((900, 1600)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((900, 1600)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def collate_fn(batch):
    """Custom collate function for NuScenes dataset."""
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)


def main():
    """Test the simplified NuScenes dataset."""
    parser = argparse.ArgumentParser(description='Test NuScenes Dataset')
    parser.add_argument('--root_dir', type=str, default=DEFAULT_NUSCENES_ROOT,
                       help='Root directory of NuScenes dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='Maximum number of samples to test')
    parser.add_argument('--camera_types', nargs='+', default=['CAM_FRONT'],
                       help='Camera types to use')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TESTING SIMPLIFIED NUSCENES DATASET")
    print("="*60)
    
    try:
        # Create dataset with validation disabled for testing
        dataset = NuScenesDataset(
            root_dir=args.root_dir,
            split=args.split,
            camera_types=args.camera_types,
            max_samples=args.max_samples,
            transform=create_nuscenes_transforms(train=False),
            validate_on_init=False  # Disable validation for testing
        )
        
        print(f"\n✅ Dataset created successfully with {len(dataset)} samples")
        
        # Test loading a few samples with visualization
        print("\nTesting sample loading with 2D bounding box visualization...")
        for i in range(min(20, len(dataset))):
            try:
                image, target = dataset[i] #image:[3, 900, 1600]
                print(f"  Sample {i}: Image shape: {image.shape}, "
                      f"Boxes: {target['boxes'].shape[0]}, "
                      f"Labels: {target['labels'].shape[0]}")
                
                # Visualize 2D bounding boxes for data correctness verification
                if target['boxes'].shape[0] > 0:
                    # Convert tensor image back to PIL for visualization
                    if isinstance(image, torch.Tensor):
                        # Denormalize if normalized
                        img_array = image.permute(1, 2, 0).numpy()
                        if img_array.min() >= 0 and img_array.max() <= 1:
                            img_array = (img_array * 255).astype(np.uint8)
                        elif img_array.min() < 0:  # Likely normalized with mean/std
                            # Simple denormalization assuming ImageNet stats
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            img_array = img_array * std + mean
                            img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
                        
                        img_pil = Image.fromarray(img_array)
                    else:
                        img_pil = image
                    
                    # Create matplotlib figure for visualization
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                    ax.imshow(img_pil)
                    ax.set_title(f'Sample {i} - 2D Bounding Boxes ({target["boxes"].shape[0]} boxes)')
                    
                    # Draw 2D bounding boxes using utility functions
                    boxes_drawn = 0
                    for j, (bbox, label) in enumerate(zip(target['boxes'], target['labels'])):
                        bbox_list = bbox.tolist()  # Convert tensor to list
                        category_name = CATEGORY_NAMES[label.item()] if label.item() < len(CATEGORY_NAMES) else f"class_{label.item()}"
                        
                        # Use the visualization function from nuscenes.py
                        try:
                            success = draw_2d_bbox(ax, bbox_list, category_name, 
                                                 color='red', linewidth=2,
                                                 img_width=img_pil.width, img_height=img_pil.height)
                            if success:
                                boxes_drawn += 1
                        except Exception as viz_e:
                            print(f"    Warning: Could not draw bbox {j}: {viz_e}")
                    
                    ax.axis('off')
                    
                    # Save visualization
                    viz_path = f'output/sample_{i}_bbox_visualization.png'
                    plt.tight_layout()
                    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"    ✅ Saved visualization: {viz_path} ({boxes_drawn}/{target['boxes'].shape[0]} boxes drawn)")
                else:
                    print(f"    No bounding boxes found for sample {i}")
                    
            except Exception as e:
                print(f"  ❌ Error loading sample {i}: {e}")
        
        # Test sample info
        if len(dataset) > 0:
            sample_info = dataset.get_sample_info(0)
            print(f"\nSample 0 info keys: {list(sample_info.keys())}")
        
        print("\n✅ Dataset testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Dataset testing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())