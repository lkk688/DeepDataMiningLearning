import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
import os
import sys
from typing import Any, Callable, List, Optional, Tuple, Dict, Union
from PIL import Image
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add the project root and scripts directory to the path
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
# SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
# sys.path.insert(0, ROOT_DIR)
# sys.path.insert(0, SCRIPTS_DIR)

# Import local modules after path setup
try:
    from DeepDataMiningLearning.detection import trainutils
except ImportError:
    # Fallback for direct execution
    import trainutils

# Import KITTI utilities
try:
    from DeepDataMiningLearning.Utils.kitti import (
        Object3d, 
        visualize_kitti_sample_2d, 
        visualize_kitti_sample_3d,
        detect_dataset_type,
        validate_kitti_structure,  # Fixed function name
        INSTANCE_Color,
        read_label,
        getcalibration
    )
    KITTI_UTILS_AVAILABLE = True
    print("✅ KITTI utilities loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: KITTI utilities not available: {e}")
    KITTI_UTILS_AVAILABLE = False

WrapNewDict = False

def multi_format_visualization_2D(sample_data, data_format: str, output_dir: str = "output", show_plot: bool = True, class_labels: Optional[List[str]] = None):
    """
    Independent visualization utility function that can visualize 2D images with bounding boxes
    based on input sample in three different formats (torch, coco, and yolo).
    
    Args:
        sample_data: Sample data in torch, coco, or yolo format
        data_format: Explicit format specification ("torch", "coco", or "yolo")
        output_dir: Directory to save visualization
        show_plot: Whether to display the plot
        class_labels: Optional list of class names. If provided, will show class names instead of class IDs
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image
        import numpy as np
        import torch
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process data based on specified format
        if data_format.lower() == "torch":
            if not (isinstance(sample_data, tuple) and len(sample_data) == 2):
                print("❌ Torch format expects (image, target) tuple")
                return False
                
            # Torch format: (image, target_dict)
            img_data, target = sample_data
            
            # Handle PIL Image or tensor
            if isinstance(img_data, torch.Tensor):
                if img_data.dim() == 3 and img_data.shape[0] == 3:  # CHW format
                    img_array = img_data.permute(1, 2, 0).numpy() #[3, 375, 1242]=>(375, 1242, 3)
                else:
                    img_array = img_data.numpy()
                
                # Normalize if needed
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                
                image = Image.fromarray(img_array)
            elif isinstance(img_data, Image.Image):
                # Already a PIL Image
                image = img_data
            else:
                # Try to convert to PIL Image
                image = Image.fromarray(np.array(img_data))
            
            # Extract bounding boxes
            if 'boxes' in target:
                boxes = target['boxes']
                labels = target.get('labels', [0] * len(boxes))
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.numpy()
                if isinstance(labels, torch.Tensor):
                    labels = labels.numpy()
            else:
                boxes = []
                labels = []
                
        elif data_format.lower() == "coco":
            if not isinstance(sample_data, dict):
                print("❌ COCO format expects dictionary")
                return False
                
            if 'img' in sample_data and 'annotations' in sample_data:
                # COCO format - use 'img' key for actual image data
                img_data = sample_data['img']
                
                # Convert to PIL Image
                if isinstance(img_data, Image.Image):
                    image = img_data
                elif isinstance(img_data, np.ndarray):
                    image = Image.fromarray(img_data)
                elif isinstance(img_data, torch.Tensor):
                    if img_data.dim() == 3 and img_data.shape[0] == 3:  # CHW format
                        img_array = img_data.permute(1, 2, 0).numpy()
                    else:
                        img_array = img_data.numpy()
                    
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    
                    image = Image.fromarray(img_array)
                else:
                    print("❌ No valid image data found in COCO format")
                    return False
                
                # Extract bounding boxes
                boxes = []
                labels = []
                for ann in sample_data['annotations']:
                    bbox = ann['bbox']  # [x, y, width, height]
                    # Convert to [x1, y1, x2, y2]
                    x1, y1, w, h = bbox
                    boxes.append([x1, y1, x1 + w, y1 + h])
                    labels.append(ann.get('category_id', 0))
            else:
                print("❌ COCO format requires 'img' and 'annotations' keys")
                return False
                
        elif data_format.lower() == "yolo":
            if not isinstance(sample_data, dict):
                print("❌ YOLO format expects dictionary")
                return False
                
            if 'img' in sample_data and 'bboxes' in sample_data:
                # YOLO format
                img_tensor = sample_data['img']
                
                # Convert tensor to PIL Image
                if isinstance(img_tensor, torch.Tensor):
                    if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:  # CHW format
                        img_array = img_tensor.permute(1, 2, 0).numpy()
                    else:
                        img_array = img_tensor.numpy()
                    
                    # Normalize if needed
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    
                    image = Image.fromarray(img_array)
                else:
                    image = img_tensor
                
                # Extract bounding boxes (YOLO format: normalized xywh)
                boxes = sample_data['bboxes']
                labels = sample_data.get('cls', sample_data.get('labels', [0] * len(boxes)))
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.numpy()
                if isinstance(labels, torch.Tensor):
                    labels = labels.numpy()
                
                # Convert from normalized xywh to xyxy
                img_w, img_h = image.size
                converted_boxes = []
                for box in boxes:
                    if len(box) >= 4:
                        x_center, y_center, width, height = box[:4]
                        x1 = (x_center - width/2) * img_w
                        y1 = (y_center - height/2) * img_h
                        x2 = (x_center + width/2) * img_w
                        y2 = (y_center + height/2) * img_h
                        converted_boxes.append([x1, y1, x2, y2])
                boxes = np.array(converted_boxes)
            else:
                print("❌ YOLO format requires 'img' and 'bboxes' keys")
                return False
        else:
            print(f"❌ Unsupported format: {data_format}. Supported formats: torch, coco, yolo")
            return False
        
        # Create visualization (shared visualization code)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f'Multi-Format Visualization ({data_format.upper()} format)')
        
        # Define colors for different classes
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Draw bounding boxes
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                if len(box) >= 4:
                    x1, y1, x2, y2 = box[:4]
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Choose color based on label
                    color = colors[int(labels[i]) % len(colors)] if i < len(labels) else 'red'
                    
                    # Create rectangle patch
                    rect = patches.Rectangle((x1, y1), width, height, 
                                           linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    
                    # Add label text with class name if available
                    if class_labels and i < len(labels) and int(labels[i]) < len(class_labels):
                        label_text = f'{class_labels[int(labels[i])]}'
                    else:
                        label_text = f'Class_{int(labels[i])}' if i < len(labels) else 'Object'
                    
                    ax.text(x1, y1-5, label_text, color=color, fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        
        ax.axis('off')
        
        # Save visualization
        output_path = os.path.join(output_dir, f'multi_format_visualization_{data_format}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        print(f"✅ Visualization saved to: {output_path}")
        print(f"📊 Format: {data_format.upper()}, Boxes: {len(boxes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root: str,
                 train: bool = True,
                 split: str = 'train', #'val' 'test'
                 transform: Optional[Callable] = None,
                 image_dir: str = "image_2", 
                 labels_dir: str = "label_2",
                 output_format: str = "torch",  # "torch", "coco", "yolo"
                 validate_dataset: bool = False):
        """
        Enhanced KITTI Dataset with integrated utilities and format options
        
        Args:
            root: Path to KITTI dataset root
            train: Whether to use training set
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transforms to apply
            image_dir: Image directory name (default: "image_2")
            labels_dir: Labels directory name (default: "label_2")
            output_format: Output format ("torch", "coco", "yolo")
            validate_dataset: Whether to validate dataset structure on init
        """
        self.images = []
        self.targets = []
        self.root = root
        self.train = train
        self.transform = transform
        self.output_format = output_format.lower()
        self._location = "training" if self.train else "testing"
        self.image_dir_name = image_dir
        self.labels_dir_name = labels_dir
        
        # Validate dataset structure if requested
        if validate_dataset and KITTI_UTILS_AVAILABLE:
            print("🔍 Validating KITTI dataset structure...")
            is_valid = validate_kitti_structure(self.root)
            if not is_valid:
                print("⚠️ Dataset validation failed. Some features may not work correctly.")
        
        # Detect dataset type automatically
        if KITTI_UTILS_AVAILABLE:
            self.dataset_type = detect_dataset_type(self.root)
            print(f"📊 Detected dataset type: {self.dataset_type}") #standard_split
        else:
            self.dataset_type = 'kitti'
        
        # load all image files, sorting them to
        # ensure that they are aligned
        self.split = split
        split_dir = Path(self.root) / 'ImageSets' / (self.split + '.txt') #select kitti/ImageSets/val.txt
        
        # Check if ImageSets directory exists, if not create splits
        if not split_dir.exists():
            imagesets_dir = Path(self.root) / 'ImageSets'
            if not imagesets_dir.exists():
                print(f"📁 ImageSets directory not found. Creating train/val splits...")
                success = self._create_imagesets_splits()
                if not success:
                    print("⚠️ Failed to create ImageSets splits. Dataset may not work correctly.")
                    self.sample_id_list = None
                else:
                    # Reload the split file after creation
                    if split_dir.exists():
                        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()]
                    else:
                        print(f"⚠️ Split file {split_dir} still not found after creation.")
                        self.sample_id_list = None
            else:
                print(f"⚠️ Split file {split_dir} not found in existing ImageSets directory.")
                self.sample_id_list = None
        else:
            #sample_id_list: str list
            self.sample_id_list = [x.strip() for x in open(split_dir).readlines()]
            
        if self.sample_id_list:
            print(f"📋 Loaded {len(self.sample_id_list)} samples for {self.split} split") #5984
        else:
            print(f"⚠️ No samples loaded for {self.split} split")
            
        self.root_split_path = os.path.join(self.root, self._location)
        
        # Enhanced class mappings using kitti.py utilities
        # Use standard KITTI class mappings
        self.INSTANCE2id = {'__background__':0,'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian':4, 'Person_sitting':5, 'Cyclist':6, 'Tram':7, 'Misc':8, 'DontCare':9}
        
        self.INSTANCE_CATEGORY_NAMES = ['__background__'] + [k for k in self.INSTANCE2id.keys() if k != '__background__']
        self.id2INSTANCE = {v: k for k, v in self.INSTANCE2id.items()}
        self.numclass = len([k for k in self.INSTANCE2id.keys() if k not in ['__background__', 'DontCare']])
        
        print(f"📋 Loaded {len(self.INSTANCE_CATEGORY_NAMES)} object categories")
        print(f"🎯 Active classes for training: {self.numclass}")

    def _create_imagesets_splits(self, train_ratio=0.8, random_seed=42):
        """
        Create ImageSets splits when they don't exist.
        
        Args:
            train_ratio: Ratio of training samples (default: 0.8)
            random_seed: Random seed for reproducible splits (default: 42)
        """
        import random
        
        imagesets_dir = Path(self.root) / 'ImageSets'
        imagesets_dir.mkdir(parents=True, exist_ok=True)
        
        # Find image directory based on dataset structure
        possible_image_dirs = [
            Path(self.root) / self._location / self.image_dir_name,  # Standard structure
            Path(self.root) / "raw" / self._location / self.image_dir_name,  # Raw structure
            Path(self.root) / self.image_dir_name,  # Direct structure
        ]
        
        image_dir = None
        for img_dir in possible_image_dirs:
            if img_dir.exists():
                image_dir = img_dir
                break
        
        if not image_dir:
            print(f"⚠️ No image directory found. Checked: {[str(d) for d in possible_image_dirs]}")
            return False
        
        # Get all image files (support both .png and .jpg)
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend([f.stem for f in image_dir.glob(ext)])
        
        if not image_files:
            print(f"⚠️ No image files found in {image_dir}")
            return False
        
        # Sort for consistency
        image_files = sorted(image_files)
        
        # Shuffle with fixed seed for reproducibility
        random.seed(random_seed)
        random.shuffle(image_files)
        
        # Split into train and val
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Write split files
        splits = {'train': train_files, 'val': val_files}
        for split_name, files in splits.items():
            split_file = imagesets_dir / f'{split_name}.txt'
            with open(split_file, 'w') as f:
                f.write('\n'.join(files))
            print(f"📝 Created {split_name} split with {len(files)} samples: {split_file}")
        
        # Also create a test split file (empty by default, can be populated later)
        test_file = imagesets_dir / 'test.txt'
        if not test_file.exists():
            with open(test_file, 'w') as f:
                f.write('')  # Empty test split
            print(f"📝 Created empty test split: {test_file}")
        
        return True

    def get_image(self, idx):
        img_file = Path(self.root_split_path) / self.image_dir_name / ('%s.png' % idx)
        assert img_file.exists()
        image = Image.open(img_file)
        return image
    
    def get_label(self, idx):
        """Enhanced label loading with Object3d integration"""
        label_file = Path(self.root_split_path) / self.labels_dir_name / ('%s.txt' % idx)
        assert label_file.exists()
        
        if KITTI_UTILS_AVAILABLE:
            # Use enhanced Object3d parsing from kitti.py
            try:
                objects = read_label(str(label_file))
                target = []
                for obj in objects:
                    target.append({
                        "image_id": idx,
                        "type": obj.type,
                        "truncated": obj.truncation,
                        "occluded": obj.occlusion,
                        "alpha": obj.alpha,
                        "bbox": obj.box2d,  # [x1, y1, x2, y2]
                        "dimensions": [obj.h, obj.w, obj.l],  # [height, width, length]
                        "location": obj.t,  # [x, y, z] in camera coordinates
                        "rotation_y": obj.ry,
                        "score": getattr(obj, 'score', 1.0)  # confidence score if available
                    })
                return target
            except Exception as e:
                print(f"⚠️ Failed to parse with Object3d, falling back to basic parsing: {e}")
        
        # Fallback to original parsing
        target = []
        with open(label_file) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "image_id": idx, #new added to ref the filename
                        "type": line[0], #one of the following: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', or 'DontCare'. 'DontCare' is used for objects that are present but ignored for evaluation.
                        "truncated": float(line[1]), #A value of 0.0 means the object is fully visible, and 1.0 means the object is completely outside the image frame.
                        "occluded": int(line[2]), #integer value indicating the degree of occlusion, where 0 means fully visible, and higher values indicate increasing levels of occlusion.
                        "alpha": float(line[3]), #The observation angle of the object in radians, relative to the camera. It is the angle between the object's heading direction and the positive x-axis of the camera.
                        "bbox": [float(x) for x in line[4:8]], #represent the pixel locations of the top-left and bottom-right corners of the bounding box
                        "dimensions": [float(x) for x in line[8:11]], #3D dimensions of the object (height, width, and length) in meters
                        "location": [float(x) for x in line[11:14]], #3D location of the object's centroid in the camera coordinate system (in meters)
                        "rotation_y": float(line[14]), #The rotation of the object around the y-axis in camera coordinates, in radians.
                    }
                )
        return target #dict list

    def convert_target(self, image_id, target):
        num_objs = len(target)
        boxes = []
        labels = []
        for i in range(num_objs):
            bbox = target[i]['bbox'] ##represent the pixel locations of the top-left and bottom-right corners of the bounding box
            xmin = bbox[0]
            xmax = bbox[2]
            ymin = bbox[1]
            ymax = bbox[3]
            objecttype=target[i]['type']
            if objecttype != 'DontCare' and xmax-xmin>0 and ymax-ymin>0:
                labelid = self.INSTANCE2id[objecttype]
                labels.append(labelid)
                boxes.append([xmin, ymin, xmax, ymax]) #required for Torch [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
        num_objs = len(labels) #update num_objs
        newtarget = {}
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = int(image_id)
        #image_id = torch.tensor([image_id])
        #Important!!! do not make image_id a tensor, otherwise the coco evaluation will send error.
        #image_id = torch.tensor(image_id)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        if num_objs >0:
            newtarget["boxes"] = boxes
            newtarget["labels"] = labels
            #newtarget["masks"] = masks
            newtarget["image_id"] = image_id
            newtarget["area"] = area
            newtarget["iscrowd"] = iscrowd
        else:
            #negative example, ref: https://github.com/pytorch/vision/issues/2144
            newtarget['boxes'] = torch.zeros((0, 4), dtype=torch.float32)#not empty
            target['labels'] = labels #torch.as_tensor(np.array(labels), dtype=torch.int64)#empty
            target['image_id'] =image_id
            target["area"] = area #torch.as_tensor(np.array(target_areas), dtype=torch.float32)#empty
            target["iscrowd"] = iscrowd #torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#empty
        return newtarget

    def __getitem__(self, index: int) -> Union[dict, Tuple[Any, Any]]:
        """Get item at a given index with format selection.

        Args:
            index (int): Index
        Returns:
            Union[dict, Tuple]: Format depends on output_format:
                - "torch": dict with torch tensors
                - "coco": COCO format dictionary  
                - "yolo": YOLO format dictionary

        """
        
        if index >= len(self.sample_id_list):
            print("Index out-of-range")
            return None
            
        imageidx = self.sample_id_list[index]
        image = self.get_image(imageidx)
        
        if self.train:
            target = self.get_label(imageidx) #list of dicts
            target = self.convert_target(imageidx, target)
        else:
            target = {}

        if WrapNewDict:
            target = dict(image_id=imageidx, annotations=target) #new changes, not used now
            
        if self.transform:
            image, target = self.transform(image, target)
        
        # Return format based on output_format
        if self.output_format == "yolo":
            return self._convert_to_yolo_format(image, target, index)
        elif self.output_format == "coco":
            return self._convert_to_coco_format(image, target, index)
        else:  # torch format (default)
            return image, target

    def _convert_to_coco_format(self, img, target, index):
        """
        Convert KITTI format to COCO format
        
        Args:
            img: PIL Image or tensor
            target: dict with 'boxes' and 'labels'
            index: sample index
            
        Returns:
            dict: COCO format dictionary
        """
        import torch
        import numpy as np
        from PIL import Image
        
        # Convert image to array if it's PIL Image
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            img_h, img_w = img_array.shape[:2]
        else:
            if len(img.shape) == 3:
                _, img_h, img_w = img.shape
            else:
                img_h, img_w = img.shape
                
        # Get image info
        image_info = {
            'id': index,
            'width': img_w,
            'height': img_h,
            'file_name': f"{self.sample_id_list[index]}.png"
        }
        
        # Convert annotations
        annotations = []
        boxes = target.get('boxes', [])
        labels = target.get('labels', [])
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.long)
            
            for i, (box, label) in enumerate(zip(boxes, labels)):
                x1, y1, x2, y2 = box.tolist()
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                annotation = {
                    'id': i,
                    'image_id': index,
                    'category_id': int(label),
                    'bbox': [x1, y1, width, height],  # COCO format: [x, y, width, height]
                    'area': area,
                    'iscrowd': 0
                }
                annotations.append(annotation)
        
        # Create categories info
        categories = []
        for cat_id, cat_name in self.id2INSTANCE.items():
            if cat_name != '__background__':
                categories.append({
                    'id': cat_id,
                    'name': cat_name,
                    'supercategory': 'object'
                })
        
        coco_dict = {
            'image': image_info,
            'annotations': annotations,
            'categories': categories,
            'img': img
        }
        
        return coco_dict

    def _convert_to_yolo_format(self, img, target, index):
        """
        Convert KITTI format to YOLO format
        
        Args:
            img: PIL Image or tensor
            target: dict with 'boxes' and 'labels'
            index: sample index
            
        Returns:
            dict: YOLO format dictionary
        """
        import torch
        import numpy as np
        from PIL import Image
        
        # Convert image to tensor if it's PIL Image
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            else:
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
        else:
            img_tensor = img
            
        # Get image dimensions
        if len(img_tensor.shape) == 3:
            _, img_h, img_w = img_tensor.shape
        else:
            img_h, img_w = img_tensor.shape
            
        # Convert bounding boxes to YOLO format (normalized xywh)
        boxes = target.get('boxes', [])
        labels = target.get('labels', [])
        
        if len(boxes) == 0:
            # No objects in image
            yolo_bboxes = torch.zeros((0, 4))
            yolo_labels = torch.zeros((0,), dtype=torch.long)
        else:
            # Convert boxes from [x1, y1, x2, y2] to [x_center, y_center, width, height] normalized
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.long)
            
            # Convert to center format and normalize
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x_center = (x1 + x2) / 2.0 / img_w
            y_center = (y1 + y2) / 2.0 / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            
            yolo_bboxes = torch.stack([x_center, y_center, width, height], dim=1)
            yolo_labels = labels
            
        # Create YOLO format dictionary
        yolo_dict = {
            'img': img_tensor,
            'bboxes': yolo_bboxes,
            'cls': yolo_labels,
            'batch_idx': torch.tensor([index], dtype=torch.long),
            'ori_shape': torch.tensor([img_h, img_w], dtype=torch.long),
            'img_id': torch.tensor([index], dtype=torch.long)
        }
        
        return yolo_dict

    def _parse_target(self, index: int) -> List:
        target = []
        labelfile = self.targets[index]
        full_name = os.path.basename(labelfile)
        file_name = os.path.splitext(full_name)
        imageidx=int(file_name[0]) #filename index 000001
        #img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        #assert img_file.exists()

        with open(labelfile) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "image_id": imageidx, #new added to ref the filename
                        "type": line[0], #one of the following: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', or 'DontCare'. 'DontCare' is used for objects that are present but ignored for evaluation.
                        "truncated": float(line[1]), #A value of 0.0 means the object is fully visible, and 1.0 means the object is completely outside the image frame.
                        "occluded": int(line[2]), #integer value indicating the degree of occlusion, where 0 means fully visible, and higher values indicate increasing levels of occlusion.
                        "alpha": float(line[3]), #The observation angle of the object in radians, relative to the camera. It is the angle between the object's heading direction and the positive x-axis of the camera.
                        "bbox": [float(x) for x in line[4:8]], #represent the pixel locations of the top-left and bottom-right corners of the bounding box
                        "dimensions": [float(x) for x in line[8:11]], #3D dimensions of the object (height, width, and length) in meters
                        "location": [float(x) for x in line[11:14]], #3D location of the object's centroid in the camera coordinate system (in meters)
                        "rotation_y": float(line[14]), #The rotation of the object around the y-axis in camera coordinates, in radians.
                    }
                )
            #Convert to the required format by Torch
            num_objs = len(target)
            boxes = []
            labels = []
            for i in range(num_objs):
                bbox = target[i]['bbox']
                xmin = bbox[0]
                xmax = bbox[2]
                ymin = bbox[1]
                ymax = bbox[3]
                objecttype=target[i]['type']
                #if objecttype != 'DontCare' and xmax-xmin>0 and ymax-ymin>0:
                labelid = self.INSTANCE2id[objecttype]
                labels.append(labelid)
                boxes.append([xmin, ymin, xmax, ymax]) #required for Torch [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
            num_objs = len(labels) #update num_objs
            newtarget = {}
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([index])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            if num_objs >0:
                newtarget["boxes"] = boxes
                newtarget["labels"] = labels
                #newtarget["masks"] = masks
                newtarget["image_id"] = image_id
                newtarget["area"] = area
                newtarget["iscrowd"] = iscrowd
            else:
                #negative example, ref: https://github.com/pytorch/vision/issues/2144
                newtarget['boxes'] = torch.zeros((0, 4), dtype=torch.float32)#not empty
                target['labels'] = labels #torch.as_tensor(np.array(labels), dtype=torch.int64)#empty
                target['image_id'] =image_id
                target["area"] = area #torch.as_tensor(np.array(target_areas), dtype=torch.float32)#empty
                target["iscrowd"] = iscrowd #torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#empty
        return newtarget, imageidx

    def visualize_sample(self, index: int, vis_type: str = "2d", output_dir: str = "output", show_plot: bool = True):
        """
        Visualize a sample using integrated kitti.py utilities
        
        Args:
            index: Sample index to visualize
            vis_type: Visualization type ("2d", "3d", "lidar", "bev")
            output_dir: Directory to save visualizations
            show_plot: Whether to display the plot
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not KITTI_UTILS_AVAILABLE:
            print("⚠️ KITTI utilities not available. Using basic visualization.")
            return self._basic_visualize_sample(index, show_plot)
        
        if index >= len(self.sample_id_list):
            print(f"❌ Index {index} out of range (max: {len(self.sample_id_list)-1})")
            return False
            
        sample_idx = self.sample_id_list[index]
        print(f"🎨 Visualizing sample {sample_idx} (index {index}) with {vis_type} visualization...")
        
        try:
            # Convert sample_idx to integer for visualization functions
            sample_idx_int = int(sample_idx) if isinstance(sample_idx, str) else sample_idx
            
            if vis_type == "2d":
                success = visualize_kitti_sample_2d(self.root_split_path, sample_idx_int, self.dataset_type, output_dir)
            elif vis_type == "3d":
                success = visualize_kitti_sample_3d(self.root_split_path, sample_idx_int, self.dataset_type, output_dir)
            else:
                print(f"⚠️ Visualization type '{vis_type}' not supported with basic integration")
                success = self._basic_visualize_sample(index, show_plot)
                
            if success:
                print(f"✅ Visualization completed successfully!")
            else:
                print(f"❌ Visualization failed")
                
            return success
            
        except Exception as e:
            print(f"❌ Error during visualization: {e}")
            return self._basic_visualize_sample(index, show_plot)
    
    def _basic_visualize_sample(self, index: int, show_plot: bool = True):
        """
        Basic visualization using matplotlib when kitti.py utilities are not available
        
        Args:
            index: Sample index to visualize
            show_plot: Whether to display the plot
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if index >= len(self.sample_id_list):
                print(f"❌ Index {index} out of range")
                return False
                
            # Get image and target
            imageidx = self.sample_id_list[index]
            image = self.get_image(imageidx)
            
            if self.train:
                target = self.get_label(imageidx)
                target = self.convert_target(imageidx, target)
            else:
                target = {'boxes': [], 'labels': []}
            
            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            ax.set_title(f'Sample {imageidx} - 2D Bounding Boxes')
            
            # Draw bounding boxes
            boxes = target.get('boxes', [])
            labels = target.get('labels', [])
            
            if len(boxes) > 0:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.long)
                
                # Color mapping for different classes
                colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray']
                
                for box, label in zip(boxes, labels):
                    x1, y1, x2, y2 = box.tolist()
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Get class name and color
                    class_name = self.id2INSTANCE.get(int(label), f'Class_{int(label)}')
                    color = colors[int(label) % len(colors)]
                    
                    # Skip DontCare objects
                    if class_name == 'DontCare':
                        continue
                    
                    # Draw rectangle
                    rect = patches.Rectangle((x1, y1), width, height, 
                                           linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    
                    # Add label
                    ax.text(x1, y1-5, class_name, color=color, fontsize=10, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
            
            ax.axis('off')
            plt.tight_layout()
            
            if show_plot:
                plt.show()
            
            print(f"✅ Basic visualization completed for sample {imageidx}")
            return True
            
        except Exception as e:
            print(f"❌ Error in basic visualization: {e}")
            return False
    
    def validate_data_accuracy(self, num_samples: int = 5, vis_type: str = "2d"):
        """
        Validate data accuracy by visualizing multiple samples
        
        Args:
            num_samples: Number of samples to validate
            vis_type: Visualization type ("2d", "3d")
            
        Returns:
            dict: Validation results
        """
        print(f"🔍 Validating data accuracy with {num_samples} samples...")
        
        results = {
            'total_samples': num_samples,
            'successful_visualizations': 0,
            'failed_visualizations': 0,
            'samples_with_objects': 0,
            'samples_without_objects': 0,
            'class_distribution': {}
        }
        
        for i in range(min(num_samples, len(self.sample_id_list))):
            try:
                # Get sample data
                imageidx = self.sample_id_list[i]
                target = self.get_label(imageidx) if self.train else []
                
                # Count objects
                if len(target) > 0:
                    results['samples_with_objects'] += 1
                    
                    # Count class distribution
                    for obj in target:
                        obj_type = obj['type']
                        if obj_type not in results['class_distribution']:
                            results['class_distribution'][obj_type] = 0
                        results['class_distribution'][obj_type] += 1
                else:
                    results['samples_without_objects'] += 1
                
                # Try visualization
                success = self.visualize_sample(i, vis_type, show_plot=False)
                if success:
                    results['successful_visualizations'] += 1
                else:
                    results['failed_visualizations'] += 1
                    
            except Exception as e:
                print(f"⚠️ Error validating sample {i}: {e}")
                results['failed_visualizations'] += 1
        
        # Print results
        print("\n📊 Data Accuracy Validation Results:")
        print(f"✅ Successful visualizations: {results['successful_visualizations']}/{num_samples}")
        print(f"❌ Failed visualizations: {results['failed_visualizations']}/{num_samples}")
        print(f"📦 Samples with objects: {results['samples_with_objects']}")
        print(f"📭 Samples without objects: {results['samples_without_objects']}")
        print(f"🏷️ Class distribution: {results['class_distribution']}")
        
        return results

    def visualize_sample_by_format(self, sample_data, output_dir: str = "output", show_plot: bool = True):
        """
        Independent visualization utility function that can visualize 2D images with bounding boxes
        based on input sample in three different formats (torch, coco, and yolo).
        
        Args:
            sample_data: Sample data in torch, coco, or yolo format
            output_dir: Directory to save visualization
            show_plot: Whether to display the plot
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from PIL import Image
            import numpy as np
            import torch
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Detect format and extract image and bounding boxes
            if isinstance(sample_data, tuple) and len(sample_data) == 2:
                 # Torch format: (image, target_dict)
                 img_data, target = sample_data
                 format_type = "torch"
                 
                 # Handle PIL Image or tensor
                 if isinstance(img_data, torch.Tensor):
                     if img_data.dim() == 3 and img_data.shape[0] == 3:  # CHW format
                         img_array = img_data.permute(1, 2, 0).numpy()
                     else:
                         img_array = img_data.numpy()
                     
                     # Normalize if needed
                     if img_array.max() <= 1.0:
                         img_array = (img_array * 255).astype(np.uint8)
                     
                     image = Image.fromarray(img_array)
                 elif isinstance(img_data, Image.Image):
                     # Already a PIL Image
                     image = img_data
                 else:
                     # Try to convert to PIL Image
                     image = Image.fromarray(np.array(img_data))
                 
                 # Extract bounding boxes
                 if 'boxes' in target:
                     boxes = target['boxes']
                     labels = target.get('labels', [0] * len(boxes))
                     if isinstance(boxes, torch.Tensor):
                         boxes = boxes.numpy()
                     if isinstance(labels, torch.Tensor):
                         labels = labels.numpy()
                 else:
                     boxes = []
                     labels = []
                    
            elif isinstance(sample_data, dict):
                if 'image' in sample_data and 'annotations' in sample_data:
                    # COCO format
                    format_type = "coco"
                    img_data = sample_data['image']
                    
                    # Convert to PIL Image
                    if isinstance(img_data, np.ndarray):
                        image = Image.fromarray(img_data)
                    elif isinstance(img_data, torch.Tensor):
                        if img_data.dim() == 3 and img_data.shape[0] == 3:  # CHW format
                            img_array = img_data.permute(1, 2, 0).numpy()
                        else:
                            img_array = img_data.numpy()
                        
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        
                        image = Image.fromarray(img_array)
                    else:
                        print("❌ No image data found in COCO format")
                        return False
                    
                    # Extract bounding boxes
                    boxes = []
                    labels = []
                    for ann in sample_data['annotations']:
                        bbox = ann['bbox']  # [x, y, width, height]
                        # Convert to [x1, y1, x2, y2]
                        x1, y1, w, h = bbox
                        boxes.append([x1, y1, x1 + w, y1 + h])
                        labels.append(ann.get('category_id', 0))
                    
                elif 'img' in sample_data and 'bboxes' in sample_data:
                    # YOLO format
                    format_type = "yolo"
                    img_tensor = sample_data['img']
                    
                    # Convert tensor to PIL Image
                    if isinstance(img_tensor, torch.Tensor):
                        if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:  # CHW format
                            img_array = img_tensor.permute(1, 2, 0).numpy()
                        else:
                            img_array = img_tensor.numpy()
                        
                        # Normalize if needed
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        
                        image = Image.fromarray(img_array)
                    else:
                        image = img_tensor
                    
                    # Extract bounding boxes (YOLO format: normalized xywh)
                    boxes = sample_data['bboxes']
                    labels = sample_data.get('cls', [0] * len(boxes))
                    if isinstance(boxes, torch.Tensor):
                        boxes = boxes.numpy()
                    if isinstance(labels, torch.Tensor):
                        labels = labels.numpy()
                    
                    # Convert from normalized xywh to xyxy
                    img_w, img_h = image.size
                    converted_boxes = []
                    for box in boxes:
                        if len(box) >= 4:
                            x_center, y_center, width, height = box[:4]
                            x1 = (x_center - width/2) * img_w
                            y1 = (y_center - height/2) * img_h
                            x2 = (x_center + width/2) * img_w
                            y2 = (y_center + height/2) * img_h
                            converted_boxes.append([x1, y1, x2, y2])
                    boxes = np.array(converted_boxes)
                    
                elif 'image' in sample_data and 'annotations' in sample_data:
                    # COCO format
                    format_type = "coco"
                    
                    # Load image from file path or use provided image
                    if 'img' in sample_data:
                        img_tensor = sample_data['img']
                        if isinstance(img_tensor, torch.Tensor):
                            if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:  # CHW format
                                img_array = img_tensor.permute(1, 2, 0).numpy()
                            else:
                                img_array = img_tensor.numpy()
                            
                            # Normalize if needed
                            if img_array.max() <= 1.0:
                                img_array = (img_array * 255).astype(np.uint8)
                            
                            image = Image.fromarray(img_array)
                        else:
                            image = img_tensor
                    else:
                        # Try to load from file_name
                        image_info = sample_data['image']
                        if 'file_name' in image_info:
                            img_path = os.path.join(self.root_split_path, self.image_dir, image_info['file_name'])
                            image = Image.open(img_path)
                        else:
                            print("❌ No image data found in COCO format")
                            return False
                    
                    # Extract bounding boxes from annotations
                    annotations = sample_data['annotations']
                    boxes = []
                    labels = []
                    for ann in annotations:
                        if 'bbox' in ann:
                            # COCO bbox format: [x, y, width, height]
                            x, y, w, h = ann['bbox']
                            boxes.append([x, y, x + w, y + h])  # Convert to xyxy
                            labels.append(ann.get('category_id', 0))
                    
                    boxes = np.array(boxes) if boxes else []
                    labels = np.array(labels) if labels else []
                else:
                    print("❌ Unknown dictionary format")
                    return False
            else:
                print("❌ Unsupported sample format")
                return False
            
            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            ax.set_title(f'KITTI Sample Visualization ({format_type.upper()} format)')
            
            # Define colors for different classes
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            # Draw bounding boxes
            if len(boxes) > 0:
                for i, box in enumerate(boxes):
                    if len(box) >= 4:
                        x1, y1, x2, y2 = box[:4]
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Choose color based on label
                        color = colors[int(labels[i]) % len(colors)] if i < len(labels) else 'red'
                        
                        # Create rectangle patch
                        rect = patches.Rectangle((x1, y1), width, height, 
                                               linewidth=2, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)
                        
                        # Add label text
                        label_text = f'Class_{int(labels[i])}' if i < len(labels) else 'Object'
                        ax.text(x1, y1-5, label_text, color=color, fontsize=10, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
            
            ax.axis('off')
            
            # Save visualization
            output_path = os.path.join(output_dir, f'sample_visualization_{format_type}.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            print(f"✅ Visualization saved to: {output_path}")
            print(f"📊 Format: {format_type.upper()}, Boxes: {len(boxes)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in visualization: {e}")
            import traceback
            traceback.print_exc()
            return False

    def __len__(self) -> int:
        if self.sample_id_list is None:
            return 0
        return len(self.sample_id_list)

class MyKittiDetection(torch.utils.data.Dataset):
    def __init__(self, 
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 image_dir: str = "image_2", 
                 labels_dir: str = "label_2"):
        self.images = []
        self.targets = []
        self.root = root
        self.train = train
        self.transform = transform
        self._location = "training" if self.train else "testing"
        self.image_dir_name = image_dir
        self.labels_dir_name = labels_dir
        # load all image files, sorting them to
        # ensure that they are aligned
        image_dir = os.path.join(self.root, self._location, self.image_dir_name)
        if self.train:
            labels_dir = os.path.join(self.root, self._location, self.labels_dir_name)
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            if self.train:
                self.targets.append(os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt"))
        #self.imgs = list(sorted(os.listdir(os.path.join(self.root, "PNGImages"))))
        self.INSTANCE_CATEGORY_NAMES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
        self.INSTANCE2id = {'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian':4, 'Person_sitting':5, 'Cyclist':6, 'Tram':7, 'Misc':8, 'DontCare':9} #background is 0
        self.id2INSTANCE = {v: k for k, v in self.INSTANCE2id.items()}
        self.numclass = 9 #including background, excluding the 'DontCare'

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a list of dictionaries with the following keys:

            - type: str
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float

        """
        #img, target = super().__getitem__(idx)
        if index>len(self.images):
            print("Index out-of-range")
            image = None
        else:
            image = Image.open(self.images[index])
            target, image_id = self._parse_target(index) if self.train else None

        if WrapNewDict:
            target = dict(image_id=image_id, annotations=target) #new changes
        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    def _parse_target(self, index: int) -> List:
        target = []
        labelfile = self.targets[index]
        full_name = os.path.basename(labelfile)
        file_name = os.path.splitext(full_name)
        imageidx=int(file_name[0]) #filename index 000001
        #img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        #assert img_file.exists()

        with open(labelfile) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "image_id": imageidx, #new added to ref the filename
                        "type": line[0], #one of the following: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', or 'DontCare'. 'DontCare' is used for objects that are present but ignored for evaluation.
                        "truncated": float(line[1]), #A value of 0.0 means the object is fully visible, and 1.0 means the object is completely outside the image frame.
                        "occluded": int(line[2]), #integer value indicating the degree of occlusion, where 0 means fully visible, and higher values indicate increasing levels of occlusion.
                        "alpha": float(line[3]), #The observation angle of the object in radians, relative to the camera. It is the angle between the object's heading direction and the positive x-axis of the camera.
                        "bbox": [float(x) for x in line[4:8]], #represent the pixel locations of the top-left and bottom-right corners of the bounding box
                        "dimensions": [float(x) for x in line[8:11]], #3D dimensions of the object (height, width, and length) in meters
                        "location": [float(x) for x in line[11:14]], #3D location of the object's centroid in the camera coordinate system (in meters)
                        "rotation_y": float(line[14]), #The rotation of the object around the y-axis in camera coordinates, in radians.
                    }
                )
            #Convert to the required format by Torch
            num_objs = len(target)
            boxes = []
            labels = []
            for i in range(num_objs):
                bbox = target[i]['bbox']
                xmin = bbox[0]
                xmax = bbox[2]
                ymin = bbox[1]
                ymax = bbox[3]
                objecttype=target[i]['type']
                #if objecttype != 'DontCare' and xmax-xmin>0 and ymax-ymin>0:
                labelid = self.INSTANCE2id[objecttype]
                labels.append(labelid)
                boxes.append([xmin, ymin, xmax, ymax]) #required for Torch [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
            num_objs = len(labels) #update num_objs
            newtarget = {}
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([index])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            if num_objs >0:
                newtarget["boxes"] = boxes
                newtarget["labels"] = labels
                #newtarget["masks"] = masks
                newtarget["image_id"] = image_id
                newtarget["area"] = area
                newtarget["iscrowd"] = iscrowd
            else:
                #negative example, ref: https://github.com/pytorch/vision/issues/2144
                newtarget['boxes'] = torch.zeros((0, 4), dtype=torch.float32)#not empty
                target['labels'] = labels #torch.as_tensor(np.array(labels), dtype=torch.int64)#empty
                target['image_id'] =image_id
                target["area"] = area #torch.as_tensor(np.array(target_areas), dtype=torch.float32)#empty
                target["iscrowd"] = iscrowd #torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#empty
        return newtarget, imageidx

    def __len__(self) -> int:
        return len(self.images)
 

def countobjects(alltypes):
    counter = {}
    for type in alltypes:
        if type not in counter:
            counter[type] = 0
        counter[type] += 1
    return counter

if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    import DeepDataMiningLearning.detection.transforms as T
    def get_transformsimple(train):
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ToDtype(torch.float, scale=True))
        # if train:
        #     transforms.append(RandomHorizontalFlip(0.5))
        return T.Compose(transforms)
    
    class args:
        data_augmentation = 'hflip'
        backend = 'PIL'
        use_v2 = False
        weights = ''
        test_only = False
    
    # Example usage with enhanced KittiDataset
    print("🚀 Enhanced KITTI Dataset Demo")
    print("="*50)
    
    rootPath = '/data/Datasets/kitti/' #'/data/cmpe249-fa23/torchvisiondata/Kitti'
    is_train = True
    
    # Check if the dataset path exists, if not use a test path
    if not os.path.exists(rootPath):
        print(f"⚠️ Dataset path {rootPath} not found. Using current directory for testing.")
        rootPath = os.getcwd()  # Use current directory for testing
    
    # Test different output formats
    formats_to_test = ['torch', 'coco', 'yolo']
    
    for output_format in formats_to_test:
        print(f"\n📋 Testing {output_format.upper()} format:")
        print("-" * 30)
        
        try:
            # Create dataset with enhanced features
            kittidataset = KittiDataset(
                rootPath, 
                train=True, 
                transform=get_transformsimple(is_train),
                output_format=output_format,
                validate_dataset=True  # Enable dataset validation
            )
            
            print(f"📊 Dataset categories: {kittidataset.INSTANCE_CATEGORY_NAMES}")
            print(f"📏 Dataset length: {len(kittidataset)}")
            print(f"🎯 Number of classes: {kittidataset.numclass}")
            
            if len(kittidataset) > 0:
                # Get first sample
                sample = kittidataset[0]
                
                if output_format == 'torch':
                    #Torch Format ✅: (image_tensor, target_dict)
                    img, target = sample
                    print(f"🖼️ Image shape: {img.shape}") #torch.Size([3, 375, 1242])
                    print(f"🏷️ Target keys: {target.keys()}") #dict_keys(['boxes', 'labels', 'image_id', 'area', 'iscrowd'])
                    if 'boxes' in target:
                        print(f"📦 Boxes shape: {target['boxes'].shape}") #torch.Size([5, 4])
                elif output_format == 'coco':
                    #COCO Format ✅: Dictionary with 'img' (image tensor) and 'annotations' (list of bbox annotations)
                    print(f"🏷️ COCO keys: {sample.keys()}") #dict_keys(['image', 'annotations', 'categories', 'img'])
                    print(f"📊 Image info: {sample['image']}") #{'id': 0, 'width': 1242, 'height': 375, 'file_name': '001736.png'}
                    print(f"📦 Number of annotations: {len(sample['annotations'])}") #5
                elif output_format == 'yolo':
                    #YOLO Format ✅: Dictionary with 'img' (image tensor) and 'bboxes' (normalized xywh format)
                    print(f"🏷️ YOLO keys: {sample.keys()}") #dict_keys(['img', 'bboxes', 'cls', 'batch_idx', 'ori_shape', 'img_id'])
                    print(f"🖼️ Image shape: {sample['img'].shape}") #torch.Size([3, 375, 1242])
                    print(f"📦 Bboxes shape: {sample['bboxes'].shape}") #torch.Size([5, 4])
                
                # Test visualization
                print(f"\n🎨 Testing visualization for {output_format} format...")
                success = kittidataset.visualize_sample(0, vis_type="2d", show_plot=False)
                if success:
                    print("✅ Visualization test passed")
                else:
                    print("⚠️ Visualization test failed")
                
                # Test new independent multi_format_visualization_2D function
                print(f"\n🎨 Testing independent multi_format_visualization_2D for {output_format} format...")
                format_success = multi_format_visualization_2D(sample, output_format, output_dir="output", show_plot=False, class_labels=kittidataset.INSTANCE_CATEGORY_NAMES)
                if format_success:
                    print("✅ Independent multi_format_visualization_2D test passed")
                else:
                    print("⚠️ Independent multi_format_visualization_2D test failed")
                    
        except Exception as e:
            print(f"❌ Error testing {output_format} format: {e}")
    
    # Test data accuracy validation
    print(f"\n🔍 Testing data accuracy validation...")
    try:
        kittidataset = KittiDataset(rootPath, train=True, output_format='torch')
        validation_results = kittidataset.validate_data_accuracy(num_samples=3, vis_type="2d")
        print("✅ Data accuracy validation completed")
    except Exception as e:
        print(f"❌ Error in data accuracy validation: {e}")
    
    print(f"\n🎉 Enhanced KITTI Dataset demo completed!")
    
    # Legacy compatibility test
    print(f"\n🔄 Testing legacy MyKittiDetection compatibility...")
    try:
        legacy_dataset = MyKittiDetection(rootPath, train=True, transform=get_transformsimple(is_train))
        print(f"📊 Legacy dataset categories: {legacy_dataset.INSTANCE_CATEGORY_NAMES}")
        print(f"📏 Legacy dataset length: {len(legacy_dataset)}")
        
        if len(legacy_dataset) > 0:
            img, target = legacy_dataset[0]
            print(f"🖼️ Legacy image shape: {img.shape}") #torch.Size([3, 374, 1238])
            print(f"🏷️ Legacy target keys: {target.keys()}") #dict_keys(['boxes', 'labels', 'image_id', 'area', 'iscrowd'])
            print("✅ Legacy compatibility maintained")
    except Exception as e:
        print(f"❌ Error in legacy compatibility test: {e}")
