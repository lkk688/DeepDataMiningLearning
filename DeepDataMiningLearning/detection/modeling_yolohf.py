import os
import contextlib
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
import re
from typing import Dict, List, Optional, Tuple, Union
import time
import torchvision
import os
import json
import glob
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class YoloTransform:
    """
    Handles preprocessing and postprocessing for YOLO models.
    """
    def __init__(self, min_size=640, max_size=640, device='cuda', fp16=False):
        self.min_size = min_size
        self.max_size = max_size
        self.device = device
        self.fp16 = fp16
        
    def __call__(self, images):
        """
        Preprocess images for YOLO model inference.
        
        Args:
            images (list): List of images (numpy arrays in BGR format)
            
        Returns:
            torch.Tensor: Batch of preprocessed images
        """
        if not isinstance(images, list):
            images = [images]
            
        batch_tensor = []
        for img in images:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img_resized = cv2.resize(img_rgb, (self.min_size, self.max_size))
            
            # Convert to CHW format
            img_chw = img_resized.transpose(2, 0, 1)
            img_tensor = torch.from_numpy(np.ascontiguousarray(img_chw)).float()
            img_tensor /= 255.0  # Normalize to [0, 1]
            
            # Convert to FP16 if using half precision
            if self.fp16 and torch.cuda.is_available():
                img_tensor = img_tensor.half()
                
            batch_tensor.append(img_tensor)
            
        # Stack tensors and move to device
        batch = torch.stack(batch_tensor).to(self.device)
        return batch
    
    def postprocess(self, preds, img_size, orig_img_shapes, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300):
        """
        Postprocess predictions to get final detection results.
        
        Args:
            preds (torch.Tensor): Raw predictions from model
            img_size (tuple): Size of preprocessed image (h, w)
            orig_img_shapes (list): Original image shapes before preprocessing
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
            classes (list): Filter by class
            agnostic (bool): Class-agnostic NMS
            max_det (int): Maximum number of detections per image
            
        Returns:
            list: List of processed predictions
        """
        from DeepDataMiningLearning.detection.modules.utils import yolov8_non_max_suppression, scale_boxes
        
        # Apply NMS
        processed_preds = yolov8_non_max_suppression(
            preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            agnostic=agnostic,
            max_det=max_det
        )
        
        # Scale boxes back to original image size
        detections = []
        for i, pred in enumerate(processed_preds):
            if i < len(orig_img_shapes):  # Ensure we have the original shape
                orig_shape = orig_img_shapes[i]
                # Scale boxes to original image size
                pred[:, :4] = scale_boxes(img_size, pred[:, :4], orig_shape)
            
            # Format detections as expected by the trainer
            detection = {
                "boxes": pred[:, :4].detach().cpu(),
                "scores": pred[:, 4].detach().cpu(),
                "labels": pred[:, 5].detach().cpu()
            }
            detections.append(detection)
        
        return detections


class YoloDetectionModel(nn.Module):
    """YOLOv8 detection model with HuggingFace-compatible interface."""
    def __init__(self, cfg='yolov8n.yaml', scale='n', ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(cfg)
        self.yaml['scale'] = scale
        self.modelname = extract_filename(cfg)
        self.scale = scale
        self.config = {
            "model_type": "yolov8",
            "scale": scale,
            "num_classes": nc or self.yaml.get('nc', 80),
            "image_size": 640,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "max_detections": 300
        }

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        
        # Parse model and get component indices based on scale
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.inplace = self.yaml.get('inplace', True)

        # Get component indices based on scale
        self.backbone_end, self.neck_end = self._get_component_indices(scale)
        
        # Build strides
        m = self.model[-1]
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            m.bias_init()
        elif isinstance(m, IDetect):
            s = 256
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        # Init weights, biases
        initialize_weights(self)
        
        # Create transform for preprocessing and postprocessing
        self.transform = YoloTransform(min_size=640, max_size=640)
    
    def _get_component_indices(self, scale):
        """Get indices to split model into backbone, neck, and head based on scale."""
        # Default indices for different scales
        # Format: (backbone_end, neck_end)
        scale_indices = {
            'n': (9, 15),   # nano
            's': (9, 15),   # small
            'm': (9, 15),   # medium
            'l': (9, 15),   # large
            'x': (9, 15)    # xlarge
        }
        
        # Return indices for the specified scale or default to nano
        return scale_indices.get(scale, scale_indices['n'])
    
    @property
    def backbone(self):
        """Return the backbone part of the model."""
        return self.model[:self.backbone_end]
    
    @property
    def neck(self):
        """Return the neck (FPN) part of the model."""
        return self.model[self.backbone_end:self.neck_end]
    
    @property
    def heads(self):
        """Return the detection heads as a ModuleList."""
        head_modules = list(self.model[self.neck_end:])
        return nn.ModuleList(head_modules)
    
    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model with HuggingFace-compatible interface.
        
        Args:
            x: Input tensor or list of images
            
        Returns:
            dict or list: Detection results
        """
        # Handle different input types
        if isinstance(x, list) and all(isinstance(img, np.ndarray) for img in x):
            # List of numpy arrays (images)
            orig_shapes = [img.shape for img in x]
            x = self.transform(x)
            return self._forward_with_postprocessing(x, orig_shapes, **kwargs)
        
        elif isinstance(x, np.ndarray) and len(x.shape) == 3:
            # Single numpy array (image)
            orig_shape = x.shape
            x = self.transform([x])
            results = self._forward_with_postprocessing(x, [orig_shape], **kwargs)
            return results[0] if results else {}
            
        elif isinstance(x, dict):
            # Training mode with dict input
            return self.loss(x, *args, **kwargs)
            
        else:
            # Regular tensor input
            if self.training:
                return self._predict_once(x)
            else:
                # Get original image shapes if provided
                orig_img_shapes = kwargs.get('orig_img_shapes', None)
                
                if orig_img_shapes and kwargs.get('postprocess', True):
                    # Apply postprocessing
                    return self._forward_with_postprocessing(x, orig_img_shapes, **kwargs)
                else:
                    # Just return raw predictions
                    return self._predict_once(x)[0]
    
    def _forward_with_postprocessing(self, x, orig_shapes, conf_thres=0.25, iou_thres=0.45, **kwargs):
        """
        Forward pass with postprocessing for inference.
        
        Args:
            x (torch.Tensor): Input tensor
            orig_shapes (list): Original image shapes
            conf_thres (float): Confidence threshold
            iou_thres (float): IoU threshold
            
        Returns:
            list: Processed detections
        """
        # Run model
        with torch.no_grad():
            preds = self._predict_once(x)
            
        # Get image size
        img_size = x.shape[2:]  # HW
        
        # Apply postprocessing
        return self.transform.postprocess(
            preds[0] if isinstance(preds, tuple) else preds,
            img_size,
            orig_shapes,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=kwargs.get('max_det', 300)
        )
    
    def loss(self, batch, preds=None):
        """
        Compute loss

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
            
        Returns:
            tuple: (total_loss, loss_items) where loss_items is a dictionary of individual loss components
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        preds = self.forward(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)  # return losssum, lossitems
    
    def init_criterion(self):
        """
        Initialize the loss criterion based on model type.
        
        Returns:
            nn.Module: Loss criterion
        """
        from DeepDataMiningLearning.detection.modules.lossv8 import myv8DetectionLoss
        
        # Get the last layer of the model to determine model type
        m = self.model[-1]
        
        if isinstance(m, Detect):
            # YOLOv8 detection loss
            return myv8DetectionLoss(self.model[-1])
        elif isinstance(m, IDetect):
            # YOLOv7 detection loss
            from DeepDataMiningLearning.detection.modules.lossv7 import myv7DetectionLoss
            return myv7DetectionLoss(self.model[-1])
        else:
            raise NotImplementedError(f"Loss not implemented for model with final layer: {type(m)}")
    
    def _predict_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        
        # In inference mode, return both the predictions and intermediate tensors
        if not self.training:
            return x, y
        # In training mode, just return the predictions
        return x
    
    def forward_backbone(self, x):
        """Forward pass through just the backbone."""
        y = []
        for m in self.backbone:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x
    
    def forward_neck(self, x):
        """Forward pass through just the neck (FPN)."""
        y = []
        for m in self.neck:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x
    
    def forward_heads(self, x):
        """Forward pass through just the detection heads."""
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        return outputs

    def inference_test(self, image_path, conf_thres=0.25, iou_thres=0.45, max_det=300, visualize=True):
        """
        Run inference on a single image and optionally visualize the results.
        
        Args:
            image_path (str): Path to the input image
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
            max_det (int): Maximum number of detections
            visualize (bool): Whether to visualize and save the results
            
        Returns:
            dict: Dictionary containing the processed detections
        """
        # Load image
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Run inference using the new interface
        detections = self.forward(
            img_orig,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det
        )
        
        # Visualize results if requested
        if visualize and len(detections) > 0:
            # Draw boxes on the image
            boxes = detections["boxes"].numpy()
            scores = detections["scores"].numpy()
            labels = detections["labels"].numpy()
            
            # Draw boxes on the image
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                score = scores[i]
                label = int(labels[i])
                
                # Draw bounding box
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(img_orig, (x1, y1), (x2, y2), color, 2)
                
                # Add class name and confidence
                class_name = self.names.get(int(label))
                if class_name is None or class_name == f"{int(label)}":
                    class_name = coco_names.get(int(label), f"class_{label}")
                label_text = f"{class_name}: {score:.2f}"
                cv2.putText(img_orig, label_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save the output image
            output_path = image_path.replace('.', '_detected.')
            cv2.imwrite(output_path, img_orig)
            print(f"Visualization saved to {output_path}")
        
        return detections
    
    def upload_to_huggingface(self, repo_id, token=None, commit_message="Upload YOLO model"):
        """
        Upload the model to HuggingFace Hub.
        
        Args:
            repo_id (str): HuggingFace repository ID (e.g., 'username/model-name')
            token (str, optional): HuggingFace token. If None, will use the token from the environment.
            commit_message (str): Commit message for the upload
            
        Returns:
            str: URL of the uploaded model
        """
        try:
            from huggingface_hub import HfApi, create_repo
            from huggingface_hub.utils import validate_repo_id
        except ImportError:
            raise ImportError("huggingface_hub package is required to upload models. Install it with 'pip install huggingface_hub'.")
        
        # Validate repository ID
        validate_repo_id(repo_id)
        
        # Create repository if it doesn't exist
        api = HfApi()
        try:
            create_repo(repo_id, token=token, exist_ok=True)
        except Exception as e:
            print(f"Repository creation warning (can be ignored if repo exists): {e}")
        
        # Save model state dict
        model_path = f"{self.modelname}_{self.scale}.pt"
        torch.save(self.state_dict(), model_path)
        
        # Save model configuration
        config_path = f"{self.modelname}_{self.scale}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Create model card using the separate function
        from DeepDataMiningLearning.detection.utils.model_card import create_yolo_model_card
        model_card_path = "README.md"
        create_yolo_model_card(
            model_name=f"{self.modelname}_{self.scale}",
            scale=self.scale,
            num_classes=self.yaml['nc'],
            repo_id=repo_id,
            output_path=model_card_path
        )
        
        # Upload files to HuggingFace
        print(f"Uploading model to HuggingFace Hub: {repo_id}")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=model_path,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message
        )
        
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo=config_path,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message
        )
        
        api.upload_file(
            path_or_fileobj=model_card_path,
            path_in_repo=model_card_path,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message
        )
        
        # Clean up temporary files
        os.remove(model_path)
        os.remove(config_path)
        os.remove(model_card_path)
        
        print(f"Model uploaded successfully to: https://huggingface.co/{repo_id}")
        return f"https://huggingface.co/{repo_id}"
    
import os

def create_yolo_model_card(model_name, scale, num_classes, repo_id, output_path="README.md"):
    """
    Create a model card for a YOLO model to be uploaded to HuggingFace.
    
    Args:
        model_name (str): Name of the model
        scale (str): Scale of the model (n, s, m, l, x)
        num_classes (int): Number of classes the model can detect
        repo_id (str): HuggingFace repository ID
        output_path (str): Path to save the model card
        
    Returns:
        str: Path to the created model card
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Create the model card content
    model_card = f"""---
tags:
- object-detection
- yolov8
- computer-vision
---

# {model_name} - YOLOv8 Object Detection Model

This model is a YOLOv8 object detection model with scale '{scale}'. It can detect {num_classes} different classes.

## Model Details

- Model Type: YOLOv8
- Scale: {scale}
- Number of Classes: {num_classes}
- Input Size: 640x640

## Usage

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
from PIL import Image
import requests

# Load model and processor
model = AutoModelForObjectDetection.from_pretrained("{repo_id}")
processor = AutoImageProcessor.from_pretrained("{repo_id}")

# Load image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Process image and perform inference
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Convert outputs to COCO API
results = processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=[(image.height, image.width)]
)[0]

# Print results
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"Detected {{model.config.id2label[label.item()]}} with confidence "
          f"{{round(score.item(), 3)}} at location {{box.tolist()}}")
"""

def testviaHF():
    import cv2
    import torch
    import numpy as np
    from PIL import Image

    # Load the model
    model = AutoModelForObjectDetection.from_pretrained("{repo_id}")

    # Load and preprocess an image
    image_path = "path/to/your/image.jpg"
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Process image and perform inference
    processor = AutoImageProcessor.from_pretrained("{repo_id}")
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Process the outputs
    results = processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=[(image.shape[0], image.shape[1])]
    )[0]

    # Draw bounding boxes on the image
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box.tolist()]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(image, 
                    f"{{model.config.id2label[label.item()]}}: {{round(score.item(), 2)}}", 
                    (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save or display the result
    cv2.imwrite("result.jpg", image)
    cv2.imshow("Result", image)
    cv2.waitKey(0)