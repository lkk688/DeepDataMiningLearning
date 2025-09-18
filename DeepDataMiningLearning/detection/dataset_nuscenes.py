"""
NuScenes Dataset for PyTorch Object Detection Training

This module provides a PyTorch Dataset class for loading and processing NuScenes data
for object detection tasks. It follows the same interface as the KITTI dataset but
handles NuScenes-specific data structures and coordinate transformations.

Author: Generated based on KITTI dataset structure
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# NuScenes dataset structure constants
NUSCENES_STRUCTURE = {
    'samples': [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT', 
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
        'LIDAR_TOP',
        'RADAR_FRONT',
        'RADAR_FRONT_LEFT',
        'RADAR_FRONT_RIGHT',
        'RADAR_BACK_LEFT',
        'RADAR_BACK_RIGHT'
    ],
    'sweeps': [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT', 
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
        'LIDAR_TOP',
        'RADAR_FRONT',
        'RADAR_FRONT_LEFT',
        'RADAR_FRONT_RIGHT',
        'RADAR_BACK_LEFT',
        'RADAR_BACK_RIGHT'
    ]
}

# Required annotation files for NuScenes
REQUIRED_ANNOTATION_FILES = [
    'sample.json',
    'sample_data.json',
    'sample_annotation.json',
    'instance.json',
    'category.json',
    'attribute.json',
    'visibility.json',
    'sensor.json',
    'calibrated_sensor.json',
    'ego_pose.json',
    'log.json',
    'scene.json',
    'map.json'
]

# NuScenes category mapping to detection classes
NUSCENES_CATEGORIES = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'trailer': 3,
    'construction_vehicle': 4,
    'pedestrian': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'traffic_cone': 8,
    'barrier': 9
}

# Category names for visualization and evaluation
CATEGORY_NAMES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
]

# Reverse mapping from class index to category name
CLASS_TO_CATEGORY = {v: k for k, v in NUSCENES_CATEGORIES.items()}


class NuScenesDataset(Dataset):
    """
    NuScenes Dataset for PyTorch object detection training.
    
    This dataset loads NuScenes data and provides it in a format suitable for
    object detection training, similar to the KITTI dataset interface.
    """
    
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 camera_types: List[str] = None,
                 transform: Optional[transforms.Compose] = None,
                 target_transform: Optional[transforms.Compose] = None,
                 load_lidar: bool = False,
                 max_samples: Optional[int] = None):
        """
        Initialize NuScenes dataset.
        
        Args:
            root_dir: Root directory of NuScenes dataset
            split: Dataset split ('train', 'val', 'test')
            camera_types: List of camera types to use (default: ['CAM_FRONT'])
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            load_lidar: Whether to load LiDAR data
            max_samples: Maximum number of samples to load (for debugging)
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
        self.sweeps_dir = os.path.join(root_dir, 'sweeps')
        self.annotation_dir = os.path.join(root_dir, 'v1.0-trainval')
        
        # Validate dataset structure
        self._validate_dataset()
        
        # Load annotations
        self._load_annotations()
        
        # Filter samples based on split and camera types
        self._filter_samples()
        
        print(f"Loaded NuScenes dataset: {len(self.samples)} samples for {split} split")
    
    def _validate_dataset(self):
        """Validate that the dataset has the required structure."""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")
        
        if not os.path.exists(self.samples_dir):
            raise FileNotFoundError(f"Samples directory not found: {self.samples_dir}")
        
        if not os.path.exists(self.annotation_dir):
            raise FileNotFoundError(f"Annotation directory not found: {self.annotation_dir}")
        
        # Check for required annotation files
        missing_files = []
        for file_name in REQUIRED_ANNOTATION_FILES:
            file_path = os.path.join(self.annotation_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            print(f"Warning: Missing annotation files: {missing_files}")
    
    def _load_annotations(self):
        """Load all annotation files."""
        self.annotations = {}
        
        for file_name in REQUIRED_ANNOTATION_FILES:
            file_path = os.path.join(self.annotation_dir, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    self.annotations[file_name.replace('.json', '')] = data
                except json.JSONDecodeError as e:
                    print(f"Error loading {file_name}: {e}")
                    self.annotations[file_name.replace('.json', '')] = []
        
        # Create lookup dictionaries for faster access
        self._create_lookup_dicts()
    
    def _create_lookup_dicts(self):
        """Create lookup dictionaries for faster data access."""
        # Sample data lookup by token
        self.sample_data_by_token = {}
        if 'sample_data' in self.annotations:
            for sd in self.annotations['sample_data']:
                self.sample_data_by_token[sd['token']] = sd
        
        # Sample annotation lookup by sample token
        self.annotations_by_sample = defaultdict(list)
        if 'sample_annotation' in self.annotations:
            for ann in self.annotations['sample_annotation']:
                self.annotations_by_sample[ann['sample_token']].append(ann)
        
        # Category lookup by token
        self.category_by_token = {}
        if 'category' in self.annotations:
            for cat in self.annotations['category']:
                self.category_by_token[cat['token']] = cat
        
        # Instance lookup by token
        self.instance_by_token = {}
        if 'instance' in self.annotations:
            for inst in self.annotations['instance']:
                self.instance_by_token[inst['token']] = inst
        
        # Sensor and calibration lookups
        self.sensor_by_token = {}
        if 'sensor' in self.annotations:
            for sensor in self.annotations['sensor']:
                self.sensor_by_token[sensor['token']] = sensor
        
        self.calibrated_sensor_by_token = {}
        if 'calibrated_sensor' in self.annotations:
            for cs in self.annotations['calibrated_sensor']:
                self.calibrated_sensor_by_token[cs['token']] = cs
        
        # Ego pose lookup
        self.ego_pose_by_token = {}
        if 'ego_pose' in self.annotations:
            for ego in self.annotations['ego_pose']:
                self.ego_pose_by_token[ego['token']] = ego
    
    def _filter_samples(self):
        """Filter samples based on split and camera types."""
        self.samples = []
        
        if 'sample' not in self.annotations:
            print("Warning: No sample annotations found")
            return
        
        # For now, we'll use all samples since NuScenes doesn't have explicit train/val splits
        # In practice, you would implement proper train/val/test splitting
        all_samples = self.annotations['sample']
        
        # Filter samples that have the required camera data
        for sample in all_samples:
            sample_token = sample['token']
            
            # Check if sample has data for at least one of the requested camera types
            has_camera_data = False
            for camera_type in self.camera_types:
                if camera_type in sample['data']:
                    sample_data_token = sample['data'][camera_type]
                    if sample_data_token in self.sample_data_by_token:
                        has_camera_data = True
                        break
            
            if has_camera_data:
                self.samples.append(sample)
                
                # Apply max_samples limit if specified
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, target) where target is a dictionary containing:
            - boxes: Tensor of shape (N, 4) with bounding boxes in (x1, y1, x2, y2) format
            - labels: Tensor of shape (N,) with class labels
            - image_id: Image identifier
            - area: Tensor of shape (N,) with box areas
            - iscrowd: Tensor of shape (N,) indicating crowd annotations
        """
        sample = self.samples[idx]
        sample_token = sample['token']
        
        # Get image data for the first available camera type
        image = None
        image_info = None
        
        for camera_type in self.camera_types:
            if camera_type in sample['data']:
                sample_data_token = sample['data'][camera_type]
                if sample_data_token in self.sample_data_by_token:
                    image_info = self.sample_data_by_token[sample_data_token]
                    image = self.get_image(image_info['filename'])
                    break
        
        if image is None:
            raise ValueError(f"No image data found for sample {idx}")
        
        # Get annotations for this sample
        target = self.get_target(sample_token, image_info)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target
    
    def get_image(self, filename: str) -> Image.Image:
        """
        Load an image from the dataset.
        
        Args:
            filename: Relative path to the image file
            
        Returns:
            PIL Image
        """
        image_path = os.path.join(self.root_dir, filename)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
    
    def get_target(self, sample_token: str, image_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get target annotations for a sample.
        
        Args:
            sample_token: Sample token
            image_info: Image information dictionary
            
        Returns:
            Target dictionary with boxes, labels, etc.
        """
        # Get annotations for this sample
        sample_annotations = self.annotations_by_sample.get(sample_token, [])
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        # Get camera calibration info for 3D to 2D projection
        calibrated_sensor_token = image_info['calibrated_sensor_token']
        ego_pose_token = image_info['ego_pose_token']
        
        calibrated_sensor = self.calibrated_sensor_by_token.get(calibrated_sensor_token, {})
        ego_pose = self.ego_pose_by_token.get(ego_pose_token, {})
        
        for ann in sample_annotations:
            # Get category information
            category_token = ann['category_token']
            instance_token = ann['instance_token']
            
            if category_token in self.category_by_token:
                category = self.category_by_token[category_token]
                category_name = category['name']
                
                # Map category to class index
                if category_name in NUSCENES_CATEGORIES:
                    class_idx = NUSCENES_CATEGORIES[category_name]
                    
                    # Project 3D box to 2D
                    bbox_2d = self._project_3d_to_2d(
                        ann, calibrated_sensor, ego_pose, image_info
                    )
                    
                    if bbox_2d is not None:
                        x1, y1, x2, y2 = bbox_2d
                        
                        # Ensure valid bounding box
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_idx)
                            areas.append((x2 - x1) * (y2 - y1))
                            iscrowd.append(0)  # NuScenes doesn't have crowd annotations
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # Empty target
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([hash(sample_token) % (2**31)]),  # Convert to int32 range
            'area': areas,
            'iscrowd': iscrowd
        }
        
        return target
    
    def _project_3d_to_2d(self, 
                         annotation: Dict[str, Any],
                         calibrated_sensor: Dict[str, Any],
                         ego_pose: Dict[str, Any],
                         image_info: Dict[str, Any]) -> Optional[List[float]]:
        """
        Project 3D bounding box to 2D image coordinates.
        
        Args:
            annotation: 3D annotation dictionary
            calibrated_sensor: Camera calibration information
            ego_pose: Ego vehicle pose information
            image_info: Image information
            
        Returns:
            2D bounding box as [x1, y1, x2, y2] or None if projection fails
        """
        try:
            # Get 3D box parameters
            center_3d = annotation['translation']  # [x, y, z]
            size_3d = annotation['size']  # [width, length, height]
            rotation_3d = annotation['rotation']  # quaternion [w, x, y, z]
            
            # Get camera parameters
            if 'camera_intrinsic' not in calibrated_sensor:
                return None
            
            camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
            cam_translation = np.array(calibrated_sensor['translation'])
            cam_rotation = calibrated_sensor['rotation']  # quaternion
            
            # Get ego pose
            ego_translation = np.array(ego_pose['translation'])
            ego_rotation = ego_pose['rotation']  # quaternion
            
            # Project using the same method as in unzipnuscenes.py
            bbox_2d = self._project_3d_box_to_2d(
                center_3d, size_3d, rotation_3d,
                cam_translation, cam_rotation, camera_intrinsic,
                ego_translation, ego_rotation
            )
            
            return bbox_2d
            
        except Exception as e:
            print(f"Error projecting 3D box to 2D: {e}")
            return None
    
    def _project_3d_box_to_2d(self, center_3d, size_3d, rotation_3d, 
                             cam_translation, cam_rotation, camera_intrinsic,
                             ego_translation, ego_rotation):
        """
        Project 3D bounding box to 2D using NuScenes coordinate transformations.
        
        This is a simplified version of the projection from unzipnuscenes.py
        """
        try:
            # Get 3D box corners
            corners_3d = self._get_3d_box_corners(center_3d, size_3d, rotation_3d)
            
            # Transform from global to ego coordinate system
            corners_3d = corners_3d - ego_translation
            ego_rot_matrix = self._quaternion_to_rotation_matrix(ego_rotation)
            corners_3d = corners_3d @ ego_rot_matrix.T
            
            # Transform from ego to camera coordinate system
            corners_3d = corners_3d - cam_translation
            cam_rot_matrix = self._quaternion_to_rotation_matrix(cam_rotation)
            corners_3d = corners_3d @ cam_rot_matrix.T
            
            # Project to 2D
            corners_2d = []
            for corner in corners_3d:
                if corner[2] > 0:  # Only project points in front of camera
                    point_2d = camera_intrinsic @ corner
                    point_2d = point_2d[:2] / point_2d[2]
                    corners_2d.append(point_2d)
            
            if len(corners_2d) < 4:  # Need at least 4 corners for a valid box
                return None
            
            # Get 2D bounding box from projected corners
            corners_2d = np.array(corners_2d)
            x_min, y_min = corners_2d.min(axis=0)
            x_max, y_max = corners_2d.max(axis=0)
            
            return [float(x_min), float(y_min), float(x_max), float(y_max)]
            
        except Exception as e:
            return None
    
    def _get_3d_box_corners(self, center, size, rotation):
        """Get 8 corners of a 3D bounding box."""
        w, l, h = size
        
        # Define box corners in local coordinate system
        corners = np.array([
            [-w/2, -l/2, -h/2], [w/2, -l/2, -h/2], [w/2, l/2, -h/2], [-w/2, l/2, -h/2],
            [-w/2, -l/2, h/2], [w/2, -l/2, h/2], [w/2, l/2, h/2], [-w/2, l/2, h/2]
        ])
        
        # Apply rotation
        rotation_matrix = self._quaternion_to_rotation_matrix(rotation)
        corners = corners @ rotation_matrix.T
        
        # Translate to center
        corners += center
        
        return corners
    
    def _quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])


def create_nuscenes_transforms(train: bool = True) -> transforms.Compose:
    """
    Create standard transforms for NuScenes dataset.
    
    Args:
        train: Whether to create transforms for training (includes augmentation)
        
    Returns:
        Composed transforms
    """
    if train:
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def collate_fn(batch):
    """
    Custom collate function for NuScenes dataset.
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        Batched images and targets
    """
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)


# Test and demo functions
if __name__ == "__main__":
    # Example usage and testing
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    def test_dataset(root_dir: str, max_samples: int = 5):
        """Test the NuScenes dataset loading."""
        print("Testing NuScenes dataset...")
        
        # Create dataset
        transform = create_nuscenes_transforms(train=False)
        dataset = NuScenesDataset(
            root_dir=root_dir,
            split='train',
            camera_types=['CAM_FRONT'],
            transform=transform,
            max_samples=max_samples
        )
        
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Test loading a few samples
        for i in range(min(3, len(dataset))):
            try:
                image, target = dataset[i]
                print(f"Sample {i}:")
                print(f"  Image shape: {image.shape}")
                print(f"  Number of boxes: {len(target['boxes'])}")
                print(f"  Labels: {target['labels'].tolist()}")
                
                # Visualize first sample
                if i == 0:
                    visualize_sample(image, target, f"Sample {i}")
                    
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
    
    def visualize_sample(image: torch.Tensor, target: Dict[str, Any], title: str = "Sample"):
        """Visualize a sample with bounding boxes."""
        # Convert tensor to PIL image
        if isinstance(image, torch.Tensor):
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            
            # Convert to numpy
            image = image.permute(1, 2, 0).numpy()
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(title)
        
        # Draw bounding boxes
        boxes = target['boxes']
        labels = target['labels']
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan']
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            width = x2 - x1
            height = y2 - y1
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=colors[label % len(colors)], facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            category_name = CLASS_TO_CATEGORY.get(label.item(), f"class_{label}")
            ax.text(x1, y1 - 5, category_name, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[label % len(colors)], alpha=0.7),
                   fontsize=10, color='white')
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Example usage
    if len(os.sys.argv) > 1:
        root_dir = os.sys.argv[1]
        test_dataset(root_dir)
    else:
        print("Usage: python dataset_nuscenes.py <nuscenes_root_dir>")
        print("Example: python dataset_nuscenes.py /path/to/nuscenes/v1.0-trainval")