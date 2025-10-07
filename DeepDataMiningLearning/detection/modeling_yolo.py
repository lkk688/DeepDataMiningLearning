"""
Simple YOLO Detection Model with TorchvisionYoloModel

This module provides a clean implementation focused on TorchvisionYoloModel
for standard torchvision-compatible inference workflows.

Key Features:
- TorchvisionYoloModel wrapper for YOLO models
- Standard torchvision preprocessing pipeline
- Bounding box visualization utilities
- Simple inference interface

"""

# Standard library imports
import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Computer vision and visualization imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Torchvision imports
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, to_tensor

# YOLO-specific imports
from DeepDataMiningLearning.detection.modules.lossv8 import myv8DetectionLoss
from DeepDataMiningLearning.detection.modules.lossv7 import myv7DetectionLoss

# COCO class names for visualization
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]



class TorchvisionYoloModel(nn.Module):
    """
    Torchvision-compatible wrapper for YoloModel.
    
    This wrapper makes YoloModel compatible with torchvision training and evaluation scripts
    by providing the same input/output interface as torchvision detection models like FasterRCNN.
    
    Training mode:
        - Input: images (list of tensors), targets (list of dicts with 'boxes' and 'labels')
        - Output: dict with loss keys {'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'}
    
    Evaluation mode:
        - Input: images (list of tensors)
        - Output: list of dicts with keys {'boxes', 'labels', 'scores'}
    
    Example:
        >>> # Create torchvision-compatible YOLO model
        >>> model = TorchvisionYoloModel('yolov8', scale='n', num_classes=91)
        >>> 
        >>> # Training mode
        >>> model.train()
        >>> images = [torch.rand(3, 640, 640)]
        >>> targets = [{'boxes': torch.tensor([[100, 100, 200, 200]]), 'labels': torch.tensor([1])}]
        >>> losses = model(images, targets)
        >>> 
        >>> # Evaluation mode
        >>> model.eval()
        >>> with torch.no_grad():
        >>>     predictions = model(images)
    """
    
    def __init__(self, model_name='yolov8', scale='n', num_classes=80, state_dict_path=None, map_to_torchvision_classes=True, **kwargs):
        """
        Initialize TorchvisionYoloModel.
        
        Args:
            model_name (str): YOLO model name ('yolov8', 'yolov11', etc.)
            scale (str): Model scale ('n', 's', 'm', 'l', 'x')
            num_classes (int): Number of classes (default: 80 for COCO)
            state_dict_path (str, optional): Path to state_dict file to load weights
            map_to_torchvision_classes (bool): If True, map YOLO's 80 COCO classes to torchvision's 91 classes
            **kwargs: Additional arguments passed to YoloDetectionModel
        """
        super().__init__()
        
        # Store class mapping preference
        self.map_to_torchvision_classes = map_to_torchvision_classes
        
        # Import required modules
        from DeepDataMiningLearning.detection.modules.yolomodels import YoloDetectionModel, load_defaultcfgs, load_checkpoint
        from DeepDataMiningLearning.detection.modules.yolotransform import YoloTransform
        
        # Use the appropriate YAML configuration file based on model_name
        if model_name.startswith('yolov8'):
            cfg = '/home/lkk/Developer/DeepDataMiningLearning/DeepDataMiningLearning/detection/modules/yolov8.yaml'
        elif model_name.startswith('yolov11'):
            cfg = '/home/lkk/Developer/DeepDataMiningLearning/DeepDataMiningLearning/detection/modules/yolov11.yaml'
        elif model_name.startswith('yolov12'):
            cfg = '/home/lkk/Developer/DeepDataMiningLearning/DeepDataMiningLearning/detection/modules/yolov12.yaml'
        else:
            # Default to yolov8 if model name is not recognized
            cfg = '/home/lkk/Developer/DeepDataMiningLearning/DeepDataMiningLearning/detection/modules/yolov8.yaml'
        
        # Create the underlying YOLO model using correct parameters (cfg, scale, ch=3)
        self.yolo_model = YoloDetectionModel(cfg=cfg, scale=scale, ch=3, **kwargs)
        self.num_classes = num_classes
        
        # Load weights if state_dict_path is provided
        if state_dict_path and os.path.exists(state_dict_path):
            print(f"Loading state_dict from: {state_dict_path}")
            self.yolo_model = load_checkpoint(self.yolo_model, state_dict_path)
            print("✓ State dict loaded successfully")
        elif state_dict_path is None:
            # Try to ensure custom checkpoint exists, downloading and converting if necessary
            try:
                from DeepDataMiningLearning.detection.ultralytics_converter import ensure_custom_yolo_checkpoint
                
                # Determine device for checkpoint conversion
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                
                ckpt_file = ensure_custom_yolo_checkpoint(
                    model_name=f"{model_name}{scale}",
                    ckpt_path=None,  # Let the function determine the path
                    weights_dir="./weights",
                    ckpt_dir="./checkpoints", 
                    num_classes=num_classes,
                    device=device
                )
                
                if ckpt_file and os.path.exists(ckpt_file):
                    self.yolo_model = load_checkpoint(self.yolo_model, ckpt_file)
                    print(f"✓ Converted checkpoint loaded from: {ckpt_file}")
                else:
                    print("⚠ No checkpoint loaded - using random weights")
            except Exception as e:
                print(f"⚠ Could not load/convert checkpoint: {e}")
                print("⚠ Using random weights")
        else:
            print(f"⚠ State dict file not found: {state_dict_path}")
            print("⚠ Using random weights")
        
        # Load default configurations for preprocessing
        cfgPath = '/home/lkk/Developer/DeepDataMiningLearning/DeepDataMiningLearning/detection/modules/default.yaml'
        try:
            self.DEFAULT_CFG_DICT = load_defaultcfgs(cfgPath)
        except:
            print(f"⚠ Could not load config from {cfgPath}, using empty config")
            self.DEFAULT_CFG_DICT = {}
        
        # Initialize loss function for training
        if hasattr(self.yolo_model, 'version') and self.yolo_model.version == 'v7':
            self.loss_fn = myv7DetectionLoss(self.yolo_model)
        else:
            self.loss_fn = myv8DetectionLoss(self.yolo_model)
        
        # Transform for input preprocessing using YoloTransform
        self.transform = self._create_transform()
    
    def _create_transform(self):
        """Create YoloTransform for input preprocessing."""
        # Import YoloTransform here to avoid import issues
        from DeepDataMiningLearning.detection.modules.yolotransform import YoloTransform
        # Use YoloTransform similar to the correct implementation
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
        return YoloTransform(min_size=640, max_size=640, device=device, fp16=False, cfgs=self.DEFAULT_CFG_DICT)
    
    def forward(self, images, targets=None):
        """
        Forward pass with torchvision-compatible interface.
        
        Args:
            images (list[Tensor]): List of images, each of shape [C, H, W]
            targets (list[Dict[str, Tensor]], optional): Ground truth targets for training
                Each dict should contain:
                - 'boxes': FloatTensor[N, 4] in format [x1, y1, x2, y2]
                - 'labels': Int64Tensor[N] with class labels
        
        Returns:
            Training mode: Dict[str, Tensor] with loss values
            Evaluation mode: List[Dict[str, Tensor]] with predictions
        """
        # Ensure all images are on the same device as the model
        device = next(self.parameters()).device
        if isinstance(images, list):
            images = [img.to(device) for img in images]
        else:
            images = images.to(device)
            
        if self.training and targets is not None:
            return self._forward_training(images, targets)
        else:
            return self._forward_inference(images)
    
    def _forward_training(self, images, targets):
        """
        Forward pass for training mode.
        
        Returns:
            Dict[str, Tensor]: Loss dictionary compatible with torchvision
        """
        # Convert list of images to batch tensor
        batch_images = self._images_to_batch(images)
        
        # Convert targets to YOLO format
        yolo_targets = self._convert_targets_to_yolo(targets, batch_images.shape)
        
        # Forward pass through YOLO model
        predictions = self.yolo_model(batch_images)
        
        # Compute losses
        losses = self.loss_fn(predictions, yolo_targets)
        
        # Return losses directly - torchvision accepts any loss dict format
        return losses
    
    def _forward_inference(self, images):
        """
        Forward pass for inference mode.
        
        Returns:
            List[Dict[str, Tensor]]: Predictions in torchvision format
        """
        device = next(self.parameters()).device
        
        # Preprocess images using YoloTransform (similar to correct implementation)
        # Convert PIL/numpy images to OpenCV format if needed
        processed_images = []
        original_shapes = []
        
        for img in images:
            if isinstance(img, torch.Tensor):
                # Convert tensor to numpy for YoloTransform
                if img.dim() == 3:  # [C, H, W]
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype('uint8')
                else:
                    img_np = img.cpu().numpy()
                processed_images.append(img_np)
                original_shapes.append(img_np.shape)
            else:
                processed_images.append(img)
                original_shapes.append(img.shape)
        
        # Use YoloTransform for preprocessing
        batch_tensor = self.transform(processed_images)
        
        # Ensure batch tensor is on the correct device
        if isinstance(batch_tensor, torch.Tensor):
            batch_tensor = batch_tensor.to(device)
        
        # Forward pass through YOLO model
        with torch.no_grad():
            predictions = self.yolo_model(batch_tensor)
        
        # Post-process predictions using the model's postprocess method
        detections = self.yolo_model.postprocess(predictions, original_shapes, input_shape=batch_tensor.shape[2:])
        
        # Convert to torchvision format
        results = self._convert_detections_to_torchvision(detections, original_shapes, self.map_to_torchvision_classes)
        
        return results
    
    def _images_to_batch(self, images):
        """Convert list of images to batch tensor."""
        # Find maximum dimensions
        max_h = max(img.shape[-2] for img in images)
        max_w = max(img.shape[-1] for img in images)
        
        # Pad images to same size
        batch_images = []
        for img in images:
            c, h, w = img.shape
            padded_img = torch.zeros(c, max_h, max_w, dtype=img.dtype, device=img.device)
            padded_img[:, :h, :w] = img
            batch_images.append(padded_img)
        
        return torch.stack(batch_images)
    
    def _convert_targets_to_yolo(self, targets, image_shape):
        """Convert torchvision targets to YOLO format."""
        batch_size, _, img_h, img_w = image_shape
        
        # Create batch dictionary format expected by YOLO loss
        batch_targets = {
            'batch_idx': [],
            'cls': [],
            'bboxes': []
        }
        
        for i, target in enumerate(targets):
            boxes = target['boxes']  # [N, 4] in [x1, y1, x2, y2] format
            labels = target['labels']  # [N]
            
            if len(boxes) == 0:
                continue
                
            # Convert to YOLO format: [x_center, y_center, width, height]
            # Normalize coordinates to [0, 1]
            x1, y1, x2, y2 = boxes.unbind(1)
            x_center = (x1 + x2) / 2 / img_w
            y_center = (y1 + y2) / 2 / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            
            # Stack normalized coordinates
            normalized_boxes = torch.stack([x_center, y_center, width, height], dim=1)
            
            # Add to batch
            batch_idx = torch.full((len(boxes),), i, dtype=torch.float32)
            batch_targets['batch_idx'].append(batch_idx)
            batch_targets['cls'].append(labels.float())
            batch_targets['bboxes'].append(normalized_boxes)
        
        # Concatenate all targets
        if batch_targets['batch_idx']:
            batch_targets['batch_idx'] = torch.cat(batch_targets['batch_idx'])
            batch_targets['cls'] = torch.cat(batch_targets['cls'])
            batch_targets['bboxes'] = torch.cat(batch_targets['bboxes'])
        else:
            # Empty targets
            batch_targets['batch_idx'] = torch.empty(0)
            batch_targets['cls'] = torch.empty(0)
            batch_targets['bboxes'] = torch.empty(0, 4)
        
        return batch_targets
    
    def _convert_detections_to_torchvision(self, detections, original_sizes=None, map_to_torchvision_classes=False):
        """Convert YoloDetectionModel postprocess output to torchvision format.
        
        Args:
            detections: YOLO model output (coordinates are already in original image scale)
            original_sizes: Not used anymore since coordinates are already scaled
            map_to_torchvision_classes: If True, map YOLO's 80 COCO classes to torchvision's 91 classes
        """
        # YOLO 80 classes to torchvision 91 classes mapping
        # YOLO uses 0-79, torchvision uses 1-90 with some gaps
        yolo_to_torchvision_mapping = {
            0: 1,   # person
            1: 2,   # bicycle
            2: 3,   # car
            3: 4,   # motorcycle
            4: 5,   # airplane
            5: 6,   # bus
            6: 7,   # train
            7: 8,   # truck
            8: 9,   # boat
            9: 10,  # traffic light
            10: 11, # fire hydrant
            11: 13, # stop sign
            12: 14, # parking meter
            13: 15, # bench
            14: 16, # bird
            15: 17, # cat
            16: 18, # dog
            17: 19, # horse
            18: 20, # sheep
            19: 21, # cow
            20: 22, # elephant
            21: 23, # bear
            22: 24, # zebra
            23: 25, # giraffe
            24: 27, # backpack
            25: 28, # umbrella
            26: 31, # handbag
            27: 32, # tie
            28: 33, # suitcase
            29: 34, # frisbee
            30: 35, # skis
            31: 36, # snowboard
            32: 37, # sports ball
            33: 38, # kite
            34: 39, # baseball bat
            35: 40, # baseball glove
            36: 41, # skateboard
            37: 42, # surfboard
            38: 43, # tennis racket
            39: 44, # bottle
            40: 46, # wine glass
            41: 47, # cup
            42: 48, # fork
            43: 49, # knife
            44: 50, # spoon
            45: 51, # bowl
            46: 52, # banana
            47: 53, # apple
            48: 54, # sandwich
            49: 55, # orange
            50: 56, # broccoli
            51: 57, # carrot
            52: 58, # hot dog
            53: 59, # pizza
            54: 60, # donut
            55: 61, # cake
            56: 62, # chair
            57: 63, # couch
            58: 64, # potted plant
            59: 65, # bed
            60: 67, # dining table
            61: 70, # toilet
            62: 72, # tv
            63: 73, # laptop
            64: 74, # mouse
            65: 75, # remote
            66: 76, # keyboard
            67: 77, # cell phone
            68: 78, # microwave
            69: 79, # oven
            70: 80, # toaster
            71: 81, # sink
            72: 82, # refrigerator
            73: 84, # book
            74: 85, # clock
            75: 86, # vase
            76: 87, # scissors
            77: 88, # teddy bear
            78: 89, # hair drier
            79: 90  # toothbrush
        }
        results = []
        
        for i, detection in enumerate(detections):
            result = {
                'boxes': torch.empty(0, 4),
                'labels': torch.empty(0, dtype=torch.int64),
                'scores': torch.empty(0)
            }
            
            # Handle UltralyticsResult object (as shown in correct implementation)
            if hasattr(detection, 'boxes') and detection.boxes is not None:
                boxes_obj = detection.boxes
                if hasattr(boxes_obj, 'data') and boxes_obj.data is not None:
                    # Extract raw tensor data
                    detection_data = boxes_obj.data  # Should be [N, 6] tensor
                    
                    if len(detection_data) > 0:
                        boxes = detection_data[:, :4]  # Should be x1, y1, x2, y2
                        scores = detection_data[:, 4]
                        labels = detection_data[:, 5].long()
                        
                        # Map YOLO classes to torchvision classes if requested
                        if map_to_torchvision_classes:
                            mapped_labels = torch.zeros_like(labels)
                            for j, label in enumerate(labels):
                                mapped_labels[j] = yolo_to_torchvision_mapping.get(label.item(), label.item())
                            labels = mapped_labels
                        
                        # Check if boxes are in correct format and convert if needed
                        # YOLO boxes might be in different formats, ensure they're in (xmin, ymin, xmax, ymax)
                        if boxes.numel() > 0:
                            # Ensure boxes are in the correct format
                            # If boxes have negative values or seem to be in center format, convert them
                            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                            
                            # Ensure x1 < x2 and y1 < y2 (proper xyxy format)
                            xmin = torch.min(x1, x2)
                            ymin = torch.min(y1, y2)
                            xmax = torch.max(x1, x2)
                            ymax = torch.max(y1, y2)
                            
                            boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)
                            
                            # No coordinate scaling needed since images are not resized
                        
                        result = {
                            'boxes': boxes,
                            'labels': labels,
                            'scores': scores
                        }
            elif isinstance(detection, torch.Tensor):
                # Handle raw tensor format
                if len(detection) > 0:
                    boxes = detection[:, :4]  # Should be x1, y1, x2, y2
                    scores = detection[:, 4]
                    labels = detection[:, 5].long()
                    
                    # Map YOLO classes to torchvision classes if requested
                    if map_to_torchvision_classes:
                        mapped_labels = torch.zeros_like(labels)
                        for j, label in enumerate(labels):
                            mapped_labels[j] = yolo_to_torchvision_mapping.get(label.item(), label.item())
                        labels = mapped_labels
                    
                    # Ensure boxes are in correct format
                    if boxes.numel() > 0:
                        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                        
                        # Ensure x1 < x2 and y1 < y2 (proper xyxy format)
                        xmin = torch.min(x1, x2)
                        ymin = torch.min(y1, y2)
                        xmax = torch.max(x1, x2)
                        ymax = torch.max(y1, y2)
                        
                        boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)
                        
                        # No coordinate scaling needed since images are not resized
                    
                    result = {
                        'boxes': boxes,
                        'labels': labels,
                        'scores': scores
                    }
            
            results.append(result)
        
        return results
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict, handling both wrapped and unwrapped formats."""
        # Try to load directly first
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except:
            # If that fails, try to load into the underlying YOLO model
            yolo_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('yolo_model.'):
                    yolo_state_dict[key[11:]] = value  # Remove 'yolo_model.' prefix
                else:
                    yolo_state_dict[key] = value
            
            return self.yolo_model.load_state_dict(yolo_state_dict, strict=strict)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Get state dict in wrapped format."""
        return super().state_dict(destination, prefix, keep_vars)
    
    @property
    def device(self):
        """Get device of the model."""
        return next(self.parameters()).device
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        return self



def visualize_predictions(image, predictions, conf_threshold=0.5, save_path="inference_result.png"):
    """
    Visualize predictions with bounding boxes using torchvision utilities.
    
    Args:
        image (PIL.Image or torch.Tensor): Input image
        predictions (dict): Predictions with 'boxes', 'labels', 'scores'
        conf_threshold (float): Confidence threshold for filtering detections
        save_path (str): Path to save the visualization
    """
    # Convert PIL image to tensor if needed
    if isinstance(image, Image.Image):
        image_tensor = to_tensor(image)
    else:
        image_tensor = image
    
    # Ensure image is in [0, 255] range for visualization
    if image_tensor.max() <= 1.0:
        image_tensor = (image_tensor * 255).to(torch.uint8)
    else:
        image_tensor = image_tensor.to(torch.uint8)
    
    # Filter predictions by confidence threshold
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    
    # Filter by confidence
    keep = scores >= conf_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    if len(boxes) == 0:
        print(f"No detections above confidence threshold {conf_threshold}")
        # Save original image
        plt.figure(figsize=(12, 8))
        plt.imshow(image_tensor.permute(1, 2, 0))
        plt.axis('off')
        plt.title("No Detections Found")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Create labels with class names and confidence scores
    class_labels = []
    for label, score in zip(labels, scores):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        class_labels.append(f"{class_name}: {score:.2f}")
    
    # Draw bounding boxes using torchvision
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "cyan", "magenta"] * 50  # Ensure enough colors
    result_image = draw_bounding_boxes(
        image_tensor,
        boxes,
        labels=class_labels,
        colors=colors[:len(boxes)],  # Use only as many colors as needed
        width=3,
        font_size=20
    )
    
    # Save visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(result_image.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f"YOLO Detection Results ({len(boxes)} detections)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to {save_path}")
    print(f"📊 Found {len(boxes)} detections above confidence {conf_threshold}")
    
    # Print detection summary
    for i, (label, score, box) in enumerate(zip(labels, scores, boxes)):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        x1, y1, x2, y2 = box.tolist()
        print(f"  {i+1}. {class_name}: {score:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")


def main():
    """
    Simple inference example using TorchvisionYoloModel with standard torchvision pipeline.
    """
    print("🚀 Starting TorchvisionYoloModel Inference Example")
    print("=" * 60)
    
    # Configuration
    image_path = "sampledata/bus.jpg"
    model_name = "yolov8"
    scale = "n"
    conf_threshold = 0.5
    save_path = "output/torchvision_yolo_inference.png"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Please ensure the image exists or update the image_path variable.")
        return
    
    try:
        # 1. Load and preprocess image using standard torchvision pipeline
        print(f"📸 Loading image: {image_path}")
        
        # Standard torchvision preprocessing pipeline
        preprocess = transforms.Compose([
            #transforms.Resize((640, 640)),  # Resize to model input size
            transforms.ToTensor(),          # Convert to tensor [0, 1]
        ])
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        print(f"   Original size: {original_size}")
        
        # Apply preprocessing
        image_tensor = preprocess(image)
        print(f"   Preprocessed size: {image_tensor.shape}") #[3, 640, 640]
        
        # 2. Initialize TorchvisionYoloModel
        print(f"🤖 Initializing TorchvisionYoloModel ({model_name}-{scale})")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        model = TorchvisionYoloModel(model_name=model_name, scale=scale, num_classes=80)
        model.eval()
        model = model.to(device)
        
        # Move image tensor to the same device as model
        image_tensor = image_tensor.to(device)
        
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Image tensor device: {image_tensor.device}")
        print(f"   Model device: {next(model.parameters()).device}")
        
        # 3. Run inference
        print("🔍 Running inference...")
        with torch.no_grad():
            # TorchvisionYoloModel expects list of images
            predictions = model([image_tensor])
        
        # Get predictions for the first (and only) image
        pred = predictions[0]
        
        print(f"   Raw detections: {len(pred['boxes'])}")
        print(f"   Score range: [{pred['scores'].min():.3f}, {pred['scores'].max():.3f}]")
        
        # 4. Visualize results
        print(f"🎨 Creating visualization (conf_threshold={conf_threshold})")
        
        # Use original image for visualization (better quality)
        visualize_predictions(image, pred, conf_threshold=conf_threshold, save_path=save_path)
        
        print("=" * 60)
        print("✅ Inference completed successfully!")
        print(f"📁 Results saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    