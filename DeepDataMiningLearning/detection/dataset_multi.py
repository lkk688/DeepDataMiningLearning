import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional
from pathlib import Path
import torch

class MultiDatasetLoader:
    """
    Unified loader for COCO, KITTI, Waymo, NuScenes, Cityscapes, Argoverse.
    
    Args:
        root_dir: Dataset root directory
        dataset_type: One of ['coco', 'kitti', 'waymo', 'nuscenes', 'cityscapes', 'argo']
        split: 'train' or 'val'
        transform: Optional torchvision transforms
    """
    def __init__(self, root_dir: str, dataset_type: str, split: str = 'train', transform=None):
        self.root_dir = Path(root_dir)
        self.dataset_type = dataset_type.lower()
        self.split = split
        self.transform = transform
        
        # Dataset-specific initialization
        self.classes = self._get_dataset_classes()
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} {dataset_type} {split} samples")
        print(f"Available classes: {self.classes}")

    def _get_dataset_classes(self) -> List[str]:
        """Returns class list for current dataset."""
        class_maps = {
            'coco': ['person', 'bicycle', 'car', ...],  # 80 COCO classes
            'kitti': ['car', 'pedestrian', 'cyclist', 'van', 'truck', 'tram'],
            'waymo': ['vehicle', 'pedestrian', 'cyclist', 'sign'],
            'nuscenes': ['car', 'truck', 'pedestrian', 'motorcycle', ...],  # 23 classes
            'cityscapes': ['person', 'rider', 'car', 'truck', 'bus', 'train'],
            'argo': ['vehicle', 'pedestrian', 'cyclist', 'traffic_light']
        }
        return class_maps[self.dataset_type]

    def _load_samples(self) -> List[Dict]:
        """Load dataset samples with unified format."""
        if self.dataset_type == 'coco':
            return self._load_coco()
        elif self.dataset_type == 'kitti':
            return self._load_kitti()
        elif self.dataset_type == 'waymo':
            return self._load_waymo()
        elif self.dataset_type == 'nuscenes':
            return self._load_nuscenes()
        elif self.dataset_type == 'cityscapes':
            return self._load_cityscapes()
        elif self.dataset_type == 'argo':
            return self._load_argoverse()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_type}")

    # ------------ Dataset Specific Loaders ------------
    def _load_coco(self) -> List[Dict]:
        """Load COCO format dataset."""
        ann_file = self.root_dir / 'annotations' / f'instances_{self.split}2017.json'
        with open(ann_file) as f:
            coco = json.load(f)
        
        # Create image_id to annotations mapping
        anns = defaultdict(list)
        for ann in coco['annotations']:
            anns[ann['image_id']].append(ann)
        
        samples = []
        for img in coco['images']:
            samples.append({
                'image_path': self.root_dir / self.split / img['file_name'],
                'boxes': [ann['bbox'] for ann in anns[img['id']]],  # [x,y,w,h]
                'class_ids': [ann['category_id'] for ann in anns[img['id']]]
            })
        return samples

    def _load_kitti(self) -> List[Dict]:
        """Load KITTI format dataset."""
        image_dir = self.root_dir / 'training' / 'image_2'
        label_dir = self.root_dir / 'training' / 'label_2'
        
        samples = []
        for img_path in image_dir.glob('*.png'):
            label_path = label_dir / f"{img_path.stem}.txt"
            
            boxes, class_ids = [], []
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    cls = parts[0].lower()
                    if cls in ['car', 'pedestrian', 'cyclist']:  # Filter relevant classes
                        boxes.append([float(x) for x in parts[4:8]])  # x1,y1,x2,y2
                        class_ids.append(self.classes.index(cls))
            
            samples.append({
                'image_path': img_path,
                'boxes': boxes,
                'class_ids': class_ids
            })
        return samples

    def _load_waymo_v2(self) -> List[Dict]:
        """Load Waymo Open Dataset v2 (direct protobuf format).
        https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_v2.ipynb
        Expected directory structure:
        waymo_v2/
        ├── training/
        │   ├── segment-xxxx.tfrecord -> .bin in v2
        │   └── annotations/ (optional)
        ├── validation/
        └── testing/

        Requires waymo-open-dataset package:
        pip install waymo-open-dataset-tf-2-11-0
        pip install gcsfs waymo-open-dataset-tf-2-12-0==1.6.4
        """
        from waymo_open_dataset import dataset_pb2
        from waymo_open_dataset import label_pb2
        from waymo_open_dataset.protos import metrics_pb2
        import tensorflow as tf  # Required for protobuf deserialization

        samples = []
        split_dir = self.root_dir / self.split

        # Waymo v2 class mapping
        waymo_classes_v2 = {
            label_pb2.Label.TYPE_VEHICLE: 'vehicle',
            label_pb2.Label.TYPE_PEDESTRIAN: 'pedestrian',
            label_pb2.Label.TYPE_SIGN: 'sign',
            label_pb2.Label.TYPE_CYCLIST: 'cyclist'
        }

        # Process each .bin file
        for bin_file in split_dir.glob('*.bin'):
            try:
                # Read frame protobuf
                with open(bin_file, 'rb') as f:
                    frame = dataset_pb2.Frame()
                    frame.ParseFromString(f.read())

                # Process each camera image
                for camera_image in frame.images:
                    img_name = f"{frame.context.name}_{camera_image.name}_{frame.timestamp_micros}.jpg"
                    img_path = split_dir / img_name
                    
                    # Save image if not exists
                    if not img_path.exists():
                        with open(img_path, 'wb') as f:
                            f.write(camera_image.image)

                    # Get camera calibration
                    calib = next(c for c in frame.context.camera_calibrations 
                                if c.name == camera_image.name)

                    boxes = []
                    class_ids = []

                    # Process 3D labels
                    for label in frame.projected_lidar_labels:
                        if label.name != camera_image.name:
                            continue

                        # Project 3D labels to 2D
                        for proto_label in label.labels:
                            # Convert to unified classes
                            if proto_label.type not in waymo_classes_v2:
                                continue

                            # Get 2D box (already projected)
                            box = proto_label.box
                            x_min = box.center_x - box.length / 2
                            y_min = box.center_y - box.width / 2
                            x_max = box.center_x + box.length / 2
                            y_max = box.center_y + box.width / 2

                            # Filter invalid boxes
                            if (x_max <= x_min or y_max <= y_min or
                                x_min < 0 or y_min < 0 or
                                x_max > camera_image.width or y_max > camera_image.height):
                                continue

                            boxes.append([x_min, y_min, x_max, y_max])
                            class_ids.append(
                                self.classes.index(waymo_classes_v2[proto_label.type]))
                    
                    samples.append({
                        'image_path': img_path,
                        'boxes': boxes,
                        'class_ids': class_ids,
                        'original_size': (camera_image.width, camera_image.height)
                    })

            except Exception as e:
                print(f"Error processing {bin_file}: {str(e)}")
                continue

        return samples

    # ------------ Common Interface ------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        orig_size = image.size  # (width, height)
        
        # Convert boxes to tensor
        boxes = torch.as_tensor(sample['boxes'], dtype=torch.float32)
        class_ids = torch.as_tensor(sample['class_ids'], dtype=torch.int64)
        
        # Apply transforms
        if self.transform:
            image, boxes = self.transform(image, boxes)
        
        return {
            'image': image,
            'boxes': boxes,
            'class_ids': class_ids,
            'original_size': orig_size,
            'image_path': str(sample['image_path'])
        }

# --------------------------
# Data Preparation Guide
# --------------------------

"""
Dataset Preparation Instructions:

1. COCO:
   - Download from: https://cocodataset.org/
   - Expected structure:
     coco/
     ├── annotations/
     │   ├── instances_train2017.json
     │   └── instances_val2017.json
     ├── train2017/
     └── val2017/

2. KITTI:
   - Download from: http://www.cvlibs.net/datasets/kitti/
   - Expected structure:
     kitti/
     ├── training/
     │   ├── image_2/  # Left color images
     │   └── label_2/  # Annotation files
     └── testing/

3. Waymo:
   - Download from: https://waymo.com/open/
   - Convert to COCO format using official tools

4. NuScenes:
   - Download from: https://www.nuscenes.org/
   - Use nuscenes-devkit to export annotations

5. Cityscapes:
   - Download from: https://www.cityscapes-dataset.com/
   - Use gtFine annotations for instance detection

6. Argoverse:
   - Download from: https://www.argoverse.org/
   - Use Argoverse API to process annotations
"""

# --------------------------
# Data Visualization
# --------------------------

def visualize_sample(dataset: MultiDatasetLoader, idx: int):
    """Visualize a sample with bounding boxes."""
    sample = dataset[idx]
    image = T.ToPILImage()(sample['image'])
    boxes = sample['boxes'].numpy()
    class_ids = sample['class_ids'].numpy()
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    for box, cls_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        label = f"{dataset.classes[cls_id]}"
        ax.text(x1, y1 - 5, label, color='red', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(f"Dataset: {dataset.dataset_type} | Image: {Path(sample['image_path']).name}")
    plt.axis('off')
    plt.show()

# --------------------------
# Usage Example
# --------------------------

if __name__ == "__main__":
    # Initialize dataset
    dataset = MultiDatasetLoader(
        root_dir="data/coco",
        dataset_type="coco",
        split="train"
    )
    
    # Visualize random samples
    for _ in range(3):
        idx = np.random.randint(len(dataset))
        visualize_sample(dataset, idx)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: x  # Custom collate may be needed
    )
    
    # Check batch
    batch = next(iter(dataloader))
    print(f"Batch contains {len(batch)} samples")
    print(f"Image shape: {batch[0]['image'].shape}")
    print(f"Boxes shape: {batch[0]['boxes'].shape}")