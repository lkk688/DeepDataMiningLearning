
"""
YOLOv8 Detection Model Implementation

This module provides a comprehensive implementation of YOLO detection models with enhanced
multi-version support, component access methods, and testing capabilities.

Key Features:
- Multi-version YOLO support (v7, v8, v11, v12)
- Component-wise access to backbone, neck, and detection head
- Comprehensive testing and benchmarking utilities
- Enhanced weight initialization and checkpoint loading
- Detailed tensor shape documentation and validation

Author: Enhanced YOLO Implementation
"""

import contextlib  # Context management utilities
from copy import deepcopy  # Deep copying for model configuration
from pathlib import Path  # Path manipulation utilities
import numpy as np  # Numerical operations
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
import yaml  # YAML configuration file parsing
import re  # Regular expression operations
from typing import Dict, List, Optional, Tuple, Union  # Type hints for better code documentation

# Import YOLO building blocks and components
from DeepDataMiningLearning.detection.modules.block import (
    AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
    Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d,
    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, RepC3, RepConv, MP, SPPCSPC,
    C3K2, C2PSA, PSABlock, MultiHeadAttention, RELAN, A2 
)  # Various building blocks for YOLO architecture

# Import utility functions
from DeepDataMiningLearning.detection.modules.utils import (
    extract_filename, LOGGER, make_divisible, non_max_suppression, scale_boxes
)  # Utility functions for model operations

# Import detection heads
from DeepDataMiningLearning.detection.modules.head import (
    Detect, IDetect, Classify, Pose, RTDETRDecoder, Segment
)  # Detection heads for different tasks

# Import loss functions
from DeepDataMiningLearning.detection.modules.lossv8 import myv8DetectionLoss  # YOLOv8 loss implementation
from DeepDataMiningLearning.detection.modules.lossv7 import myv7DetectionLoss  # YOLOv7 loss implementation
from DeepDataMiningLearning.detection.ultralytics_loss import v8DetectionLoss  # Official Ultralytics v8DetectionLoss
from DeepDataMiningLearning.detection.modules.anchor import check_anchor_order  # Anchor validation utilities
from DeepDataMiningLearning.detection.modules.hyperparameters import get_hyperparameters  # Hyperparameters configuration


def yaml_load(file='data.yaml', append_filename=True):
    """
    Load YAML data from a file or string, with optional filename appending.

    Args:
        file (str, optional): File path or string to load YAML data from. Defaults to 'data.yaml'.
        append_filename (bool, optional): Add the YAML filename under the 'yaml_file' key. Defaults to True.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {'.yaml', '.yml'}, f'Attempting to load non-YAML file {file} with yaml_load()'
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()

    # Add YAML filename to dict and return
    data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
    if append_filename:
        data['yaml_file'] = str(file)

    return data


class YoloDetectionModel(nn.Module):
    """
    Enhanced YOLO Detection Model with Multi-Version Support and Component Access.
    
    This class provides a comprehensive YOLO detection model implementation with:
    - Multi-version support (v7, v8, v11, v12)
    - Component-wise access to backbone, neck, and detection head
    - Enhanced testing and benchmarking capabilities
    - Detailed tensor shape documentation
    - Improved weight initialization and checkpoint loading
    
    Attributes:
        version (str): YOLO version ('v7', 'v8', 'v11', 'v12')
        yaml (dict): Model configuration dictionary
        model (nn.ModuleList): Sequential model layers
        save (list): Indices of layers to save for skip connections
        names (dict): Class names dictionary
        stride (torch.Tensor): Model stride values [8, 16, 32]
        nc (int): Number of classes
        reg_max (int): Regression maximum value
        backbone_indices (list): Indices of backbone layers
        neck_indices (list): Indices of neck layers  
        head_index (int): Index of detection head layer
    
    Example:
        >>> # Create YOLOv8n model
        >>> model = YoloDetectionModel('yolov8n.yaml', scale='n')
        >>> 
        >>> # Access components
        >>> backbone = model.get_backbone()
        >>> neck = model.get_neck()
        >>> head = model.get_head()
        >>> 
        >>> # Test components
        >>> results = model.test_components()
        >>> 
        >>> # Benchmark inference
        >>> benchmark = model.benchmark_inference()
    """
    
    def __init__(self, cfg='yolov8n.yaml', scale='n', ch=3, nc=None, verbose=True, version='v8'):
        """
        Initialize YoloDetectionModel with enhanced multi-version support.
        
        Args:
            cfg (str | dict): Model configuration file path or dictionary
            scale (str): Model scale ('n', 's', 'm', 'l', 'x')
            ch (int): Input channels (default: 3)
            nc (int, optional): Number of classes (overrides config)
            verbose (bool): Enable verbose logging
            version (str): YOLO version ('v7', 'v8', 'v11', 'v12')
            
        Tensor Shapes:
            Input: [B, ch, H, W] where B=batch_size, H=height, W=width
            Output: [B, num_predictions, 4+nc] where nc=num_classes
        """
        super().__init__()
        self.version = version
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(cfg)
        self.yaml['scale'] = scale
        self.yaml['version'] = version
        self.modelname = cfg.get('model_name', f'yolo{version}') if isinstance(cfg, dict) else extract_filename(cfg)

        # Define model configuration
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc

        # Version-specific model parsing
        if verbose:
            LOGGER.info(f"Building YOLO{version.upper()} model with scale '{scale}'")
            
        # Ensure ch is a list for parse_model function
        ch_list = [ch] if isinstance(ch, int) else ch
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch_list, verbose=verbose)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.inplace = self.yaml.get('inplace', True)

        # Initialize layer attributes
        for i, m in enumerate(self.model):
            m.i = i  # Layer index
            # Don't override f attribute - it's already set correctly by parse_model
            if not hasattr(m, 'f'):
                m.f = -1  # From previous layer by default only if not already set

        # Initialize component indices
        self._initialize_component_indices()

        # Build strides with version-aware detection head handling
        m = self.model[-1]  # Detection head
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            
            # Debug: Use profiling during stride calculation
            print(f"Computing stride for {type(m).__name__} head...")
            try:
                stride_outputs = forward(torch.zeros(1, ch, s, s))
                m.stride = torch.tensor([s / x.shape[-2] for x in stride_outputs])
                self.stride = m.stride
                m.bias_init()
                print(f"Stride computation successful: {self.stride}")
            except Exception as e:
                print(f"Error during stride computation: {e}")
                # Try with profiling to see what's happening
                print("Retrying with profiling enabled...")
                try:
                    stride_outputs = self._predict_once(torch.zeros(1, ch, s, s), profile=True)
                    if not isinstance(stride_outputs, (list, tuple)):
                        stride_outputs = [stride_outputs]
                    m.stride = torch.tensor([s / x.shape[-2] for x in stride_outputs])
                    self.stride = m.stride
                    m.bias_init()
                    print(f"Stride computation successful with profiling: {self.stride}")
                except Exception as e2:
                    print(f"Failed even with profiling: {e2}")
                    raise e2
            
            # Store detection head attributes for loss function compatibility
            self.nc = m.nc
            self.reg_max = getattr(m, 'reg_max', 16)
            
        elif isinstance(m, IDetect):  # YOLOv7's IDetect
            s = 256
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            
            self.nc = m.nc
            self.reg_max = getattr(m, 'reg_max', 16)
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        # Initialize weights with enhanced initialization
        initialize_weights(self)
        
        # Add hyperparameters for loss function compatibility
        self.args = get_hyperparameters()
        
        if verbose:
            self._print_model_info()
            
        # Initialize preprocessing and postprocessing components (same as Ultralytics)
        self._init_preprocessing()
        self._init_postprocessing()
    
    def _init_preprocessing(self):
        """Initialize preprocessing components (same as Ultralytics)."""
        from DeepDataMiningLearning.detection.modules.yolotransform import LetterBox
        
        # Preprocessing parameters (same as Ultralytics)
        self.letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32)
        self.fp16 = False  # Use FP16 precision
        
    def _init_postprocessing(self):
        """Initialize postprocessing components (same as Ultralytics)."""
        # Postprocessing parameters (same as Ultralytics)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45   # IoU threshold for NMS
        self.classes = None     # filter by class
        self.agnostic = False   # class-agnostic NMS
        self.multi_label = False # multiple labels per box
        self.max_det = 300      # maximum detections per image
        
    def preprocess(self, images):
        """
        Preprocess images for inference (same as Ultralytics).
        
        Args:
            images: List of images or single image (numpy arrays or PIL Images)
            
        Returns:
            tuple: (preprocessed_tensor, original_shapes)
        """
        import cv2
        import numpy as np
        from PIL import Image
        
        # Handle different input types
        if not isinstance(images, list):
            images = [images]
            
        processed_images = []
        original_shapes = []
        
        for img in images:
            # Convert PIL to numpy if needed
            if isinstance(img, Image.Image):
                img = np.array(img)
                
            # Store original shape
            original_shapes.append(img.shape[:2])  # (H, W)
            
            # Apply letterbox transformation
            img_letterboxed = self.letterbox(image=img)
            
            processed_images.append(img_letterboxed)
        
        # Stack images and convert to tensor
        if processed_images:
            # Convert list of images to batch
            im = np.stack(processed_images)  # (B, H, W, C)
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)
            
            # Convert to float and normalize
            im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            return im, original_shapes
        else:
            return torch.empty(0), []
    
    def postprocess(self, predictions, original_shapes, input_shape=(640, 640)):
        """
        Postprocess predictions (same as Ultralytics).
        
        Args:
            predictions: Raw model predictions
            original_shapes: List of original image shapes [(H, W), ...]
            input_shape: Input image shape (H, W)
            
        Returns:
            List of Results objects
        """
        from DeepDataMiningLearning.detection.modules.utils import yolov8_non_max_suppression, scale_boxes
        
        # Apply NMS
        predictions = yolov8_non_max_suppression(
            predictions,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=self.classes,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det
        )
        
        # Create Results objects
        results = []
        for i, pred in enumerate(predictions):
            if len(original_shapes) > i:
                orig_shape = original_shapes[i]
                
                # Scale boxes back to original image size
                if len(pred):
                    pred[:, :4] = scale_boxes(input_shape, pred[:, :4], orig_shape)
                
                # Create result object
                result = UltralyticsResult(
                    boxes=pred[:, :4] if len(pred) else torch.zeros(0, 4),
                    scores=pred[:, 4] if len(pred) else torch.zeros(0),
                    labels=pred[:, 5] if len(pred) else torch.zeros(0),
                    orig_shape=orig_shape,
                    input_shape=input_shape
                )
                results.append(result)
            else:
                # Empty result
                result = UltralyticsResult(
                    boxes=torch.zeros(0, 4),
                    scores=torch.zeros(0),
                    labels=torch.zeros(0),
                    orig_shape=(640, 640),
                    input_shape=input_shape
                )
                results.append(result)
        
        return results
    
    def predict_with_official_processing(self, images, conf=0.25, iou=0.45, classes=None, verbose=False):
        """
        Perform inference with official preprocessing and postprocessing (same as Ultralytics).
        
        Args:
            images: Input images (list or single image)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            classes: Filter by class indices
            verbose: Print verbose output
            
        Returns:
            List of Results objects
        """
        # Update thresholds
        orig_conf = self.conf_thres
        orig_iou = self.iou_thres
        orig_classes = self.classes
        
        self.conf_thres = conf
        self.iou_thres = iou
        self.classes = classes
        
        # Preprocess
        processed_images, original_shapes = self.preprocess(images)
        
        if processed_images.numel() == 0:
            return []
        
        # Move to device if needed
        device = next(self.parameters()).device
        processed_images = processed_images.to(device)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.forward(processed_images)
        
        # Postprocess
        results = self.postprocess(predictions, original_shapes, processed_images.shape[2:])
        
        if verbose:
            for i, result in enumerate(results):
                print(f"Image {i+1}: {len(result.boxes)} detections")
        
        # Restore original parameters
        self.conf_thres = orig_conf
        self.iou_thres = orig_iou
        self.classes = orig_classes
        
        return results

    def _initialize_component_indices(self):
        """Initialize indices for backbone, neck, and head components."""
        self.backbone_indices = []
        self.neck_indices = []
        self.head_index = len(self.model) - 1  # Last layer is always head
        
        # Identify component boundaries based on layer types and connections
        for i, layer in enumerate(self.model[:-1]):  # Exclude head
            layer_type = type(layer).__name__
            
            # Backbone layers: typically Conv, C2f, SPPF at the beginning
            if i < len(self.model) // 2 and layer_type in ['Conv', 'C2f', 'SPPF', 'C3', 'Bottleneck']:
                self.backbone_indices.append(i)
            # Neck layers: typically Upsample, Concat, C2f in the middle/end
            elif layer_type in ['Upsample', 'Concat', 'C2f', 'C3'] and i not in self.backbone_indices:
                self.neck_indices.append(i)

    def _print_model_info(self):
        """Print comprehensive model information."""
        LOGGER.info(f"YOLO{self.version.upper()} model initialized successfully: {self.modelname}")
        LOGGER.info(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")
        LOGGER.info(f"Model stride: {self.stride.tolist()}")
        LOGGER.info(f"Number of classes: {self.nc}")
        LOGGER.info(f"Backbone layers: {len(self.backbone_indices)} (indices: {self.backbone_indices})")
        LOGGER.info(f"Neck layers: {len(self.neck_indices)} (indices: {self.neck_indices})")
        LOGGER.info(f"Head layer: 1 (index: {self.head_index})")

    # Component Access Methods
    def get_backbone(self) -> nn.Sequential:
        """
        Get the backbone component of the model.
        
        Returns:
            nn.Sequential: Backbone layers for feature extraction
            
        Example:
            >>> model = YoloDetectionModel('yolov8n.yaml')
            >>> backbone = model.get_backbone()
            >>> features = backbone(input_tensor)  # Extract features
        """
        if not self.backbone_indices:
            return nn.Sequential()
        backbone_layers = [self.model[i] for i in self.backbone_indices]
        return nn.Sequential(*backbone_layers)

    def get_neck(self) -> nn.Sequential:
        """
        Get the neck component of the model.
        
        Returns:
            nn.Sequential: Neck layers for feature fusion
            
        Example:
            >>> model = YoloDetectionModel('yolov8n.yaml')
            >>> neck = model.get_neck()
            >>> fused_features = neck(backbone_features)
        """
        if not self.neck_indices:
            return nn.Sequential()
        neck_layers = [self.model[i] for i in self.neck_indices]
        return nn.Sequential(*neck_layers)

    def get_head(self) -> nn.Module:
        """
        Get the detection head of the model.
        
        Returns:
            nn.Module: Detection head for final predictions
            
        Example:
            >>> model = YoloDetectionModel('yolov8n.yaml')
            >>> head = model.get_head()
            >>> predictions = head(neck_features)
        """
        return self.model[self.head_index]

    def get_component_info(self) -> Dict[str, Dict]:
        """
        Get detailed information about model components.
        
        Returns:
            dict: Component information including layer counts and parameter counts
            
        Example:
            >>> model = YoloDetectionModel('yolov8n.yaml')
            >>> info = model.get_component_info()
            >>> print(f"Backbone params: {info['backbone']['params']:,}")
        """
        backbone = self.get_backbone()
        neck = self.get_neck()
        head = self.get_head()
        
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        return {
            'backbone': {
                'layers': len(self.backbone_indices),
                'indices': self.backbone_indices,
                'params': count_params(backbone)
            },
            'neck': {
                'layers': len(self.neck_indices),
                'indices': self.neck_indices,
                'params': count_params(neck)
            },
            'head': {
                'layers': 1,
                'index': self.head_index,
                'params': count_params(head)
            },
            'total': {
                'layers': len(self.model),
                'params': sum(p.numel() for p in self.parameters())
            }
        }

    def forward_backbone(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through backbone only.
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            
        Returns:
            List[torch.Tensor]: Backbone feature maps at different scales
            
        Example:
            >>> model = YoloDetectionModel('yolov8n.yaml')
            >>> x = torch.randn(1, 3, 640, 640)
            >>> backbone_features = model.forward_backbone(x)
            >>> print([f.shape for f in backbone_features])  # Feature shapes
        """
        features = []
        y = []
        
        for i, m in enumerate(self.model):
            if i > max(self.backbone_indices) if self.backbone_indices else -1:
                break
                
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            x = m(x)
            y.append(x if m.i in self.save else None)
            
            if i in self.backbone_indices:
                features.append(x)
                
        return features

    def forward_neck(self, backbone_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through neck only.
        
        Args:
            backbone_features (List[torch.Tensor]): Features from backbone
            
        Returns:
            List[torch.Tensor]: Neck output features
            
        Example:
            >>> backbone_features = model.forward_backbone(x)
            >>> neck_features = model.forward_neck(backbone_features)
        """
        if not self.neck_indices:
            return backbone_features
            
        # For neck processing, we need to use the full model forward logic
        # because neck layers have complex skip connections to backbone layers
        # We'll run a partial forward pass through the entire model
        
        # Start with the last backbone feature
        x = backbone_features[-1] if backbone_features else None
        y = []  # Store all layer outputs (including backbone)
        
        # First, populate y with backbone outputs (layers 0-9)
        for i, bf in enumerate(backbone_features):
            y.append(bf)
        
        # Now process neck layers (layers 10 onwards)
        for i in range(len(backbone_features), len(self.model)):
            m = self.model[i]
            
            # Handle skip connections
            if hasattr(m, 'f') and m.f != -1:
                if isinstance(m.f, int):
                    if m.f >= 0:
                        x = y[m.f]
                    # else: x remains current input (f == -1)
                else:
                    # Multiple inputs for layers like Concat
                    x_list = []
                    for j in m.f:
                        if j == -1:
                            x_list.append(x)
                        else:
                            x_list.append(y[j])
                    
                    # Check if this is a Concat layer
                    if type(m).__name__ == 'Concat':
                        # Check tensor compatibility before concatenating
                        if len(x_list) > 1:
                            # Check spatial dimensions match
                            base_shape = x_list[0].shape[2:]  # H, W
                            for idx, tensor in enumerate(x_list[1:], 1):
                                if tensor.shape[2:] != base_shape:
                                    # Try to resize to match
                                    import torch.nn.functional as F
                                    x_list[idx] = F.interpolate(tensor, size=base_shape, mode='nearest')
                                    print(f"Resized tensor {idx} from {tensor.shape[2:]} to {base_shape}")
                        
                        x = x_list  # Pass list to Concat layer
                    else:
                        # For non-Concat layers, concatenate first
                        # Check spatial dimensions match
                        if len(x_list) > 1:
                            base_shape = x_list[0].shape[2:]  # H, W
                            for idx, tensor in enumerate(x_list[1:], 1):
                                if tensor.shape[2:] != base_shape:
                                    import torch.nn.functional as F
                                    x_list[idx] = F.interpolate(tensor, size=base_shape, mode='nearest')
                        
                        x = torch.cat(x_list, 1)
            # If no skip connection (f == -1), use previous layer output
            # x remains as the output from previous layer
            
            # Forward pass through current layer
            if x is not None:
                x = m(x)
                y.append(x)
        
        # Return the outputs from the neck layers that are used by the head
        # Based on YOLOv8 config: layers 15, 18, 21 are the final outputs
        neck_outputs = []
        if len(y) > 15:
            neck_outputs.append(y[15])  # P3/8-small
        if len(y) > 18:
            neck_outputs.append(y[18])  # P4/16-medium  
        if len(y) > 21:
            neck_outputs.append(y[21])  # P5/32-large
            
        return neck_outputs if neck_outputs else backbone_features

    def forward_head(self, neck_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through detection head only.
        
        Args:
            neck_features (List[torch.Tensor]): Features from neck
            
        Returns:
            torch.Tensor: Detection predictions
            
        Example:
            >>> neck_features = model.forward_neck(backbone_features)
            >>> predictions = model.forward_head(neck_features)
        """
        head = self.get_head()
        return head(neck_features)

    # Main Forward Methods
    def forward(self, x, *args, **kwargs):
        """
        Enhanced forward pass with multi-version support.
        
        Args:
            x (torch.Tensor | dict): Input image tensor or dict with image and labels
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            torch.Tensor | dict: Model predictions or loss dict in training mode
            
        Tensor Shapes:
            Input: [B, 3, H, W] where B=batch_size, H=height, W=width
            Output (inference): [B, num_predictions, 4+nc] where nc=num_classes
            Output (training): dict with loss components
        """
        if isinstance(x, dict):
            # Training mode with batch dict containing image and labels
            img_tensor = x['img']  # Shape: [B, 3, H, W]
            preds = self._predict_once(img_tensor, **kwargs)
            
            # Compute loss if in training mode
            if self.training:
                if not hasattr(self, 'criterion'):
                    self.criterion = self.init_criterion()
                return self.criterion(preds, x)
            else:
                return preds
                
        elif self.training:
            # Training mode with tensor input
            preds = self._predict_once(x, **kwargs)
            return preds
        else:
            # Inference mode
            return self._predict_once(x, **kwargs)

    def _predict_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            profile (bool): Print computation time of each layer
            visualize (bool): Save feature maps

        Returns:
            torch.Tensor: Model output
            
        Tensor Flow:
            Input: [B, 3, H, W] -> Backbone -> Neck -> Head -> Output
        """
        import torch.nn.functional as F
        from DeepDataMiningLearning.detection.modules.block import Concat
        y = []  # Store ALL intermediate outputs (not just save indices)
        
        for i, m in enumerate(self.model):
            # Handle skip connections based on 'from' attribute
            if hasattr(m, 'f') and m.f != -1:
                if isinstance(m.f, int):
                    # Fix: Properly handle negative indices for single from parameter
                    if m.f >= 0:
                        x = y[m.f]  # From specific earlier layer
                    else:
                        # Convert negative index to positive index relative to current position
                        resolved_idx = i + m.f
                        x = y[resolved_idx]  # From earlier layer using resolved index
                else:
                    # From multiple layers - collect inputs for concatenation
                    x_list = []
                    for j in m.f:
                        if j == -1:
                            x_list.append(x)  # Current input
                        else:
                            x_list.append(y[j])  # From earlier layer
                    
                    # Debug shape information
                    if profile:
                        LOGGER.info(f"Layer {i} ({type(m).__name__}) has f={m.f}, collecting inputs:")
                        for idx, tensor in enumerate(x_list):
                            LOGGER.info(f"  Input {idx} from f[{m.f[idx]}]: {tensor.shape}")
                    
                    # Check if this is a Concat layer that expects a list
                    is_concat_layer = (type(m).__name__ == 'Concat')
                    is_detect_layer = (type(m).__name__ in ['Detect', 'IDetect'])
                    
                    if profile:
                        LOGGER.info(f"  Is Concat layer: {is_concat_layer}")
                        LOGGER.info(f"  Is Detect layer: {is_detect_layer}")
                    
                    if is_concat_layer:
                        # Check tensor compatibility before concatenating
                        if len(x_list) > 1:
                            # Check spatial dimensions match
                            base_shape = x_list[0].shape[2:]  # H, W
                            for idx, tensor in enumerate(x_list[1:], 1):
                                if tensor.shape[2:] != base_shape:
                                    if profile:
                                        LOGGER.warning(f"Shape mismatch at layer {i}: tensor 0 has shape {x_list[0].shape}, tensor {idx} has shape {tensor.shape}")
                                    # Try to resize to match
                                    x_list[idx] = F.interpolate(tensor, size=base_shape, mode='nearest')
                                    if profile:
                                        LOGGER.info(f"Resized tensor {idx} to {x_list[idx].shape}")
                        
                        x = x_list  # Pass the list to Concat layer
                        if profile:
                            LOGGER.info(f"  Passing list of {len(x_list)} tensors to Concat layer")
                    elif is_detect_layer:
                        # Detect layer expects a list of tensors from different scales
                        x = x_list  # Pass the list directly to Detect layer
                        if profile:
                            LOGGER.info(f"  Passing list of {len(x_list)} tensors to Detect layer")
                            for idx, tensor in enumerate(x_list):
                                LOGGER.info(f"    Scale {idx}: {tensor.shape}")
                    else:
                        # For non-Concat layers, concatenate first then pass single tensor
                        if len(x_list) > 1:
                            # Check spatial dimensions match
                            base_shape = x_list[0].shape[2:]  # H, W
                            for idx, tensor in enumerate(x_list[1:], 1):
                                if tensor.shape[2:] != base_shape:
                                    if profile:
                                        LOGGER.warning(f"Shape mismatch at layer {i}: tensor 0 has shape {x_list[0].shape}, tensor {idx} has shape {tensor.shape}")
                                    # Try to resize to match
                                    x_list[idx] = F.interpolate(tensor, size=base_shape, mode='nearest')
                                    if profile:
                                        LOGGER.info(f"Resized tensor {idx} to {x_list[idx].shape}")
                            
                            x = torch.cat(x_list, 1)  # Concatenate for non-Concat layers
                            if profile:
                                LOGGER.info(f"  Concatenated to single tensor: {x.shape}")
                        else:
                            x = x_list[0]  # Single tensor
                            if profile:
                                LOGGER.info(f"  Using single tensor: {x.shape}")
            
            # Forward pass through current layer
            try:
                if profile and type(m).__name__ == 'Concat':
                    LOGGER.info(f"About to pass to Concat layer {i}: input type = {type(x)}")
                    if isinstance(x, list):
                        LOGGER.info(f"  List with {len(x)} tensors: {[t.shape for t in x]}")
                    else:
                        LOGGER.info(f"  Single tensor: {x.shape}")
                
                x = m(x)  # Forward through layer
            except (RuntimeError, TypeError) as e:
                if profile:
                    LOGGER.error(f"Error at layer {i} ({type(m).__name__}): {e}")
                    LOGGER.error(f"Input type: {type(x)}")
                    if isinstance(x, torch.Tensor):
                        LOGGER.error(f"Input shape: {x.shape}")
                    elif isinstance(x, list):
                        LOGGER.error(f"Input list shapes: {[t.shape for t in x]}")
                    if hasattr(m, 'f'):
                        LOGGER.error(f"Layer f attribute: {m.f}")
                    if hasattr(m, 'conv') and hasattr(m.conv, 'weight'):
                        LOGGER.error(f"Conv weight shape: {m.conv.weight.shape}")
                raise e
            
            # Always save outputs (needed for skip connections)
            y.append(x)
            
            if profile:
                if isinstance(x, torch.Tensor):
                    shape_info = x.shape
                elif isinstance(x, (list, tuple)):
                    shape_info = [t.shape if isinstance(t, torch.Tensor) else str(type(t)) for t in x]
                else:
                    shape_info = str(type(x))
                LOGGER.info(f"Layer {i}: {type(m).__name__} -> {shape_info}")
                
        return x

    def predict(self, source, save=False, imgsz=640, conf=0.25, iou=0.45, **kwargs):
        """
        Perform inference on images.
        
        Args:
            source: Input source (image path, tensor, etc.)
            save (bool): Save results
            imgsz (int): Input image size
            conf (float): Confidence threshold
            iou (float): IoU threshold for NMS
            **kwargs: Additional arguments
            
        Returns:
            Detection results
        """
        # This is a placeholder - full implementation would handle
        # image preprocessing, inference, and postprocessing
        if isinstance(source, torch.Tensor):
            return self.forward(source)
        else:
            raise NotImplementedError("Image path inference not implemented in this version")

    def train_step(self, batch, optimizer=None):
        """
        Perform a single training step.
        
        Args:
            batch (dict): Training batch with 'img' and 'cls', 'bboxes'
            optimizer: Optimizer for parameter updates
            
        Returns:
            dict: Loss components and metrics
            
        Example:
            >>> batch = {'img': images, 'cls': classes, 'bboxes': boxes}
            >>> loss_dict = model.train_step(batch, optimizer)
            >>> print(f"Total loss: {loss_dict['loss']:.4f}")
        """
        self.train()
        
        # Forward pass
        loss_dict = self.forward(batch)
        
        # Backward pass if optimizer provided
        if optimizer is not None:
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()
            
        return loss_dict

    def init_criterion(self):
        """Initialize loss criterion based on model version."""
        if "v8" in self.modelname:
            return v8DetectionLoss(self)
        elif "v7" in self.modelname:
            return myv7DetectionLoss(self)
        else:
            return v8DetectionLoss(self)  # Default to v8

    def loss(self, batch, preds=None):
        """
        Compute loss for training.

        Args:
            batch (dict): Batch containing images and targets
            preds (torch.Tensor, optional): Pre-computed predictions

        Returns:
            Loss value and components
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        preds = self.forward(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)

    def test_components(self, input_shape=(1, 3, 640, 640), device='cpu', verbose=True):
        """
        Test all model components with detailed tensor shape verification.
        
        Args:
            input_shape (tuple): Input tensor shape (B, C, H, W)
            device (str): Device to run tests on
            verbose (bool): Print detailed information
            
        Returns:
            dict: Test results with tensor shapes and component info
            
        Example:
            >>> model = YoloDetectionModel('yolov8n.yaml')
            >>> results = model.test_components(input_shape=(2, 3, 640, 640))
            >>> print(f"Backbone output shapes: {results['backbone']['output_shapes']}")
        """
        self.eval()
        self.to(device)
        
        if verbose:
            print("=" * 80)
            print("YOLO MODEL COMPONENT TESTING")
            print("=" * 80)
            print(f"Model: {self.modelname} (version: {self.version})")
            print(f"Input shape: {input_shape}")
            print(f"Device: {device}")
            print(f"Number of classes: {self.nc}")
            print(f"Model stride: {self.stride.tolist()}")
            print("-" * 80)
        
        # Create test input
        x = torch.randn(*input_shape).to(device)
        
        results = {
            'input_shape': input_shape,
            'device': device,
            'model_info': self.get_component_info()
        }
        
        with torch.no_grad():
            # Test full forward pass
            if verbose:
                print("Testing full forward pass...")
            try:
                full_output = self.forward(x)
                results['full_forward'] = {
                    'success': True,
                    'output_shape': list(full_output.shape),
                    'output_dtype': str(full_output.dtype)
                }
                if verbose:
                    print(f"✓ Full forward pass successful")
                    print(f"  Output shape: {full_output.shape}")
                    print(f"  Output dtype: {full_output.dtype}")
            except Exception as e:
                results['full_forward'] = {
                    'success': False,
                    'error': str(e)
                }
                if verbose:
                    print(f"✗ Full forward pass failed: {e}")
            
            # Test backbone
            if verbose:
                print("\nTesting backbone component...")
            try:
                backbone_features = self.forward_backbone(x)
                results['backbone'] = {
                    'success': True,
                    'num_features': len(backbone_features),
                    'output_shapes': [list(f.shape) for f in backbone_features],
                    'output_dtypes': [str(f.dtype) for f in backbone_features]
                }
                if verbose:
                    print(f"✓ Backbone test successful")
                    print(f"  Number of feature maps: {len(backbone_features)}")
                    for i, f in enumerate(backbone_features):
                        print(f"  Feature {i}: {f.shape} ({f.dtype})")
            except Exception as e:
                results['backbone'] = {
                    'success': False,
                    'error': str(e)
                }
                if verbose:
                    print(f"✗ Backbone test failed: {e}")
            
            # Test neck
            if verbose:
                print("\nTesting neck component...")
            try:
                if 'backbone' in results and results['backbone']['success']:
                    backbone_features = self.forward_backbone(x)
                    neck_features = self.forward_neck(backbone_features)
                    results['neck'] = {
                        'success': True,
                        'num_features': len(neck_features),
                        'output_shapes': [list(f.shape) for f in neck_features],
                        'output_dtypes': [str(f.dtype) for f in neck_features]
                    }
                    if verbose:
                        print(f"✓ Neck test successful")
                        print(f"  Number of feature maps: {len(neck_features)}")
                        for i, f in enumerate(neck_features):
                            print(f"  Feature {i}: {f.shape} ({f.dtype})")
                else:
                    results['neck'] = {
                        'success': False,
                        'error': 'Backbone test failed, cannot test neck'
                    }
                    if verbose:
                        print("✗ Neck test skipped (backbone failed)")
            except Exception as e:
                results['neck'] = {
                    'success': False,
                    'error': str(e)
                }
                if verbose:
                    print(f"✗ Neck test failed: {e}")
            
            # Test head
            if verbose:
                print("\nTesting detection head...")
            try:
                head = self.get_head()
                # For head testing, we need the actual neck output format
                if 'neck' in results and results['neck']['success']:
                    backbone_features = self.forward_backbone(x)
                    neck_features = self.forward_neck(backbone_features)
                    head_output = self.forward_head(neck_features)
                    results['head'] = {
                        'success': True,
                        'output_shape': list(head_output.shape),
                        'output_dtype': str(head_output.dtype)
                    }
                    if verbose:
                        print(f"✓ Head test successful")
                        print(f"  Output shape: {head_output.shape}")
                        print(f"  Output dtype: {head_output.dtype}")
                else:
                    results['head'] = {
                        'success': False,
                        'error': 'Neck test failed, cannot test head'
                    }
                    if verbose:
                        print("✗ Head test skipped (neck failed)")
            except Exception as e:
                results['head'] = {
                    'success': False,
                    'error': str(e)
                }
                if verbose:
                    print(f"✗ Head test failed: {e}")
        
        if verbose:
            print("-" * 80)
            print("COMPONENT SUMMARY:")
            for component in ['full_forward', 'backbone', 'neck', 'head']:
                if component in results:
                    status = "✓ PASS" if results[component]['success'] else "✗ FAIL"
                    print(f"{component.upper():<15} {status}")
            print("=" * 80)
        
        return results

    def benchmark_inference(self, input_shape=(1, 3, 640, 640), device='cpu', num_runs=100, warmup=10):
        """
        Benchmark inference performance of the model and its components.
        
        Args:
            input_shape (tuple): Input tensor shape
            device (str): Device for benchmarking
            num_runs (int): Number of benchmark runs
            warmup (int): Number of warmup runs
            
        Returns:
            dict: Benchmark results with timing information
            
        Example:
            >>> model = YoloDetectionModel('yolov8n.yaml')
            >>> benchmark = model.benchmark_inference(num_runs=50)
            >>> print(f"Full model FPS: {benchmark['full_model']['fps']:.1f}")
        """
        import time
        
        self.eval()
        self.to(device)
        
        print("=" * 60)
        print("YOLO MODEL INFERENCE BENCHMARK")
        print("=" * 60)
        print(f"Model: {self.modelname}")
        print(f"Input shape: {input_shape}")
        print(f"Device: {device}")
        print(f"Warmup runs: {warmup}")
        print(f"Benchmark runs: {num_runs}")
        print("-" * 60)
        
        x = torch.randn(*input_shape).to(device)
        
        def benchmark_component(func, name):
            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    _ = func(x)
            
            # Benchmark
            torch.cuda.synchronize() if device.startswith('cuda') else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = func(x)
            
            torch.cuda.synchronize() if device.startswith('cuda') else None
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / num_runs
            fps = 1.0 / avg_time
            
            return {
                'total_time': total_time,
                'avg_time': avg_time,
                'fps': fps
            }
        
        results = {}
        
        # Benchmark full model
        results['full_model'] = benchmark_component(self.forward, 'Full Model')
        
        # Benchmark components if possible
        try:
            results['backbone'] = benchmark_component(self.forward_backbone, 'Backbone')
        except:
            results['backbone'] = {'error': 'Benchmark failed'}
        
        # Print results
        print(f"{'Component':<15} {'Time (ms)':<15} {'FPS':<10}")
        print("-" * 40)
        
        for component, result in results.items():
            if 'error' not in result and 'avg_time' in result:
                avg_time_ms = result['avg_time'] * 1000
                fps = result['fps']
                print(f"{component:<15} {avg_time_ms:<15.2f} {fps:<10.1f}")
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"{component:<15} {'ERROR':<15} {error_msg}")
        
        
        print("=" * 60)
        return results


def initialize_weights(model):
    """
    Initialize model weights with improved initialization strategy.
    
    This function applies proper weight initialization to prevent gradient
    vanishing/exploding and improve training stability.
    
    Args:
        model (nn.Module): Model to initialize
        
    Initialization Strategy:
        - Conv2d: Kaiming Normal initialization for better gradient flow
        - BatchNorm2d: Weight=1, Bias=0, proper eps and momentum
        - Linear: Kaiming Normal for weights, zero bias
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            # Enable Kaiming Normal initialization for Conv2d layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif t is nn.BatchNorm2d:
            # Proper BatchNorm initialization
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            m.eps = 1e-3
            m.momentum = 0.03
        elif t is nn.Linear:
            # Kaiming initialization for Linear layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# Define blocks that take two arguments for model parsing
twoargs_blocks = [
    nn.Conv2d, Classify, Conv, ConvTranspose, GhostConv, RepConv, Bottleneck, 
    GhostBottleneck, SPP, SPPF, SPPCSPC, DWConv, Focus, BottleneckCSP, C1, C2, 
    C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3, 
    C3K2, C2PSA, PSABlock, MultiHeadAttention, RELAN, A2
]


def parse_model(d, ch, verbose=True):
    """
    Parse model configuration and build model layers.
    
    Args:
        d (dict): Model configuration dictionary
        ch (list): Input channels
        verbose (bool): Print model information
        
    Returns:
        tuple: (model, save_list) where model is nn.Sequential and save_list contains indices to save
    """
    import ast
    
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    
    nc, act, scales = d['nc'], d.get('act', 'SiLU'), d.get('scales')
    depth, width, max_channels = scales[d['scale']]
    
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    
    # Parse backbone
    for i, (f, n, m, args) in enumerate(d['backbone']):
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    # Special handling for 'anchors' parameter
                    if a == 'anchors':
                        args[j] = d.get('anchors', [])
                    else:
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {Conv, C2f, SPPF, Bottleneck}:
            # Fix: Properly handle negative indices for 'from' parameter
            if f < 0:
                # Negative index means relative to current layer position
                resolved_f = i + f
            else:
                resolved_f = f
            c1, c2 = ch[resolved_f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in {C2f}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is RepConv:
            # RepConv needs special handling - use input channels from previous layer
            if f < 0:
                resolved_f = i + f
            else:
                resolved_f = f
            c1, c2 = ch[resolved_f], args[0]
            # Apply width scaling to RepConv output channels
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is nn.BatchNorm2d:
            if f < 0:
                resolved_f = i + f
            else:
                resolved_f = f
            args = [ch[resolved_f]]
        elif m in {SPPCSPC}:
            # SPPCSPC in YOLOv7 config: [512] means only output channels specified
            # Use actual input channels from previous layer
            if f < 0:
                resolved_f = i + f
            else:
                resolved_f = f
            c1 = ch[resolved_f]  # actual input channels from previous layer
            c2 = args[0]  # output channels from config (first parameter)
            # Apply width scaling to output channels only
            if c2 != nc:  # if c2 not equal to number of classes
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            # SPPCSPC constructor: (c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13))
            # Ensure we have both c1 and c2 parameters
            args = [c1, c2] + (args[1:] if len(args) > 1 else [])
        elif m is Concat:
            if isinstance(f, list):
                resolved_indices = []
                for x in f:
                    if x < 0:
                        resolved_idx = i + x
                    else:
                        resolved_idx = x
                    resolved_indices.append(resolved_idx)
                c2 = sum(ch[idx] for idx in resolved_indices)
            else:
                if f < 0:
                    resolved_f = i + f
                else:
                    resolved_f = f
                c2 = ch[resolved_f]
        elif m is Detect:
            args.append([ch[x] for x in f])
        else:
            if f < 0:
                resolved_f = i + f
            else:
                resolved_f = f
            c2 = ch[resolved_f]
        
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            try:
                params = m_.numel() if hasattr(m_, 'numel') else sum(p.numel() for p in m_.parameters())
                LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{params:10.0f}  {t:<45}{str(args):<30}')  # print
            except:
                LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{"N/A":>10}  {t:<45}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    
    # Parse head
    for i, (f, n, m, args) in enumerate(d['head'], len(d['backbone'])):
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    # Special handling for 'anchors' parameter
                    if a == 'anchors':
                        args[j] = d.get('anchors', [])
                    else:
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {Conv, C2f, SPPF, Bottleneck}:
            # Fix: Properly handle negative indices for 'from' parameter
            if f < 0:
                resolved_f = i + f
            else:
                resolved_f = f
            c1, c2 = ch[resolved_f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in {C2f}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is RepConv:
            # RepConv needs special handling - use input channels from previous layer
            if f < 0:
                resolved_f = i + f
            else:
                resolved_f = f
            c1, c2 = ch[resolved_f], args[0]
            # Apply width scaling to RepConv output channels
            if c2 != nc:  # if c2 not equal to number of classes
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is nn.BatchNorm2d:
            if f < 0:
                resolved_f = i + f
            else:
                resolved_f = f
            args = [ch[resolved_f]]
        elif m is Concat:
            # Fix: Properly handle negative indices and channel summation
            if isinstance(f, list):
                # Handle negative indices properly
                resolved_indices = []
                for x in f:
                    if x < 0:
                        # Negative index: count from current position
                        resolved_idx = i + x
                    else:
                        resolved_idx = x
                    resolved_indices.append(resolved_idx)
                    
                c2 = sum(ch[idx] for idx in resolved_indices)
            else:
                if f < 0:
                    resolved_f = i + f
                else:
                    resolved_f = f
                c2 = ch[resolved_f]
        elif m is Detect:
            args.append([ch[x] for x in f])
        elif m in {SPPCSPC}:
            # SPPCSPC in YOLOv7 config: [512] means only output channels specified
            # Use actual input channels from previous layer
            if f < 0:
                resolved_f = i + f
            else:
                resolved_f = f
            c1 = ch[resolved_f]  # actual input channels from previous layer
            c2 = args[0]  # output channels from config (first parameter)
            # Apply width scaling to output channels only
            if c2 != nc:  # if c2 not equal to number of classes
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            # SPPCSPC constructor: (c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13))
            # Ensure we have both c1 and c2 parameters
            args = [c1, c2] + (args[1:] if len(args) > 1 else [])
        elif m is IDetect:
            # IDetect needs nc, anchors, and ch parameters
            # args already contains [nc, anchors] from the yaml config
            # Use actual channel values from ch array (which includes width scaling)
            actual_channels = [ch[x] for x in f]
            args.append(actual_channels)  # append channel list as the third parameter
            # Set c2 to 0 since IDetect doesn't output channels in the traditional sense
            c2 = 0
        elif m is nn.Upsample:
            if f < 0:
                resolved_f = i + f
            else:
                resolved_f = f
            c2 = ch[resolved_f]
        else:
            # Handle negative indices for the general case
            if isinstance(f, int):
                if f < 0:
                    resolved_f = i + f
                else:
                    resolved_f = f
                c2 = ch[resolved_f]
            elif isinstance(f, list) and len(f) > 0:
                # For list of indices, resolve each negative index
                resolved_indices = []
                for x in f:
                    if x < 0:
                        resolved_indices.append(i + x)
                    else:
                        resolved_indices.append(x)
                c2 = ch[resolved_indices[0]]
            else:
                c2 = ch[-1]
        
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            try:
                params = m_.numel() if hasattr(m_, 'numel') else sum(p.numel() for p in m_.parameters())
                LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{params:10.0f}  {t:<45}{str(args):<30}')  # print
            except:
                LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{"N/A":>10}  {t:<45}{str(args):<30}')  # print
        save.extend(x % (i + len(d['backbone'])) for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    
    return nn.ModuleList(layers), sorted(save)


def intersect_dicts(da, db, exclude=()):
    """Return a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def filter_checkpoint_for_different_classes(ckpt, model_state_dict, exclude_head_layers=True):
    """
    Filter checkpoint to handle different number of classes.
    
    Args:
        ckpt (dict): Checkpoint state dict
        model_state_dict (dict): Model state dict
        exclude_head_layers (bool): Whether to exclude detection head layers
        
    Returns:
        dict: Filtered checkpoint
    """
    if exclude_head_layers:
        # More intelligent head layer detection
        filtered_ckpt = {}
        excluded_count = 0
        
        for k, v in ckpt.items():
            should_exclude = False
            
            # Check if this is a detection head layer by examining shape mismatches
            if k in model_state_dict:
                model_param = model_state_dict[k]
                if v.shape != model_param.shape:
                    # Check if this looks like a detection head layer
                    # Detection heads typically have shapes related to num_classes
                    if ('cv2' in k or 'cv3' in k or 'detect' in k or 'head' in k or 
                        k.endswith('.weight') or k.endswith('.bias')):
                        # Additional check: if the mismatch is in the last dimension and
                        # the difference could be related to class count
                        if len(v.shape) >= 2 and len(model_param.shape) >= 2:
                            if (v.shape[:-1] == model_param.shape[:-1] or  # Last dim different
                                v.shape[0] != model_param.shape[0]):       # First dim different
                                should_exclude = True
                                excluded_count += 1
                                LOGGER.info(f"Excluding head layer {k}: checkpoint{v.shape} vs model{model_param.shape}")
            
            if not should_exclude and k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_ckpt[k] = v
                    
        LOGGER.info(f"Filtered checkpoint: {len(filtered_ckpt)}/{len(ckpt)} layers loaded, {excluded_count} head layers excluded")
        return filtered_ckpt
    else:
        return intersect_dicts(ckpt, model_state_dict)


def load_defaultcfgs(cfgPath):
    """Load default configurations from YAML file."""
    try:
        return yaml_load(cfgPath)
    except Exception as e:
        LOGGER.warning(f"Failed to load config from {cfgPath}: {e}")
        return {}


def load_checkpoint(model, ckpt_file, fp16=False, exclude_head_on_mismatch=True):
    """
    Load checkpoint with enhanced compatibility handling.
    
    Args:
        model: Model to load checkpoint into
        ckpt_file (str): Checkpoint file path
        fp16 (bool): Use half precision
        exclude_head_on_mismatch (bool): Exclude head layers on shape mismatch
        
    Returns:
        Model with loaded weights
    """
    try:
        # Load checkpoint with proper compatibility handling
        ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model' in ckpt:
            state_dict = ckpt['model']
            # If model is an object (like Ultralytics model), get its state_dict
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
            
        # Convert to float32 if needed
        if isinstance(state_dict, dict):
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    state_dict[k] = v.float()
            
        # Filter checkpoint for compatibility
        if exclude_head_on_mismatch:
            state_dict = filter_checkpoint_for_different_classes(
                state_dict, model.state_dict(), exclude_head_layers=True
            )
        else:
            state_dict = intersect_dicts(state_dict, model.state_dict())
            
        # Load filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            LOGGER.info(f"Missing keys in checkpoint: {len(missing_keys)} keys")
        if unexpected_keys:
            LOGGER.info(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            
        LOGGER.info(f"Checkpoint loaded successfully from {ckpt_file}")
        
        if fp16:
            model = model.half()
            
        return model
        
    except Exception as e:
        LOGGER.error(f"Failed to load checkpoint from {ckpt_file}: {e}")
        return model


# Import additional modules for image processing and transforms
import DeepDataMiningLearning.detection.transforms as T
from DeepDataMiningLearning.detection.modules.yolotransform import LetterBox
from PIL import Image


def get_transformsimple(train):
    """Get simple transforms for training/validation."""
    if train:
        return T.Compose([T.ToTensor()])
    else:
        return T.Compose([T.ToTensor()])


def preprocess_img(imagepath, opencvread=True, fp16=False):
    """
    Preprocess image for YOLO inference.
    
    Args:
        imagepath (str): Path to image
        opencvread (bool): Use OpenCV for reading
        fp16 (bool): Use half precision
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    import cv2
    
    if opencvread:
        img = cv2.imread(imagepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = Image.open(imagepath).convert('RGB')
        img = np.array(img)
    
    # Simple preprocessing - resize and normalize
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    if fp16:
        img = img.half()
        
    return img


# Additional imports for model creation and testing
import os
from collections import OrderedDict
import cv2
from DeepDataMiningLearning.detection.modules.yolotransform import YoloTransform
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TABLEAU_COLORS
import random

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


def visualize_detections(image, detections, title="Detections", conf_threshold=0.25, save_path=None):
    """
    Visualize bounding box detections on an image.
    
    Args:
        image: Input image (numpy array in BGR format or PIL Image)
        detections: Detection results in various formats:
            - torch.Tensor: [N, 6] format (x1, y1, x2, y2, conf, class)
            - UltralyticsResult: Ultralytics result object
            - List of boxes: List format
        title: Title for the visualization
        conf_threshold: Confidence threshold for filtering detections
        save_path: Path to save the visualization (optional)
    
    Returns:
        matplotlib figure object
    """
    # Convert image to RGB if it's BGR (OpenCV format)
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR format from OpenCV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
    else:
        image_rgb = np.array(image)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Use the unified parse_detections function for consistent parsing
    boxes, scores, classes = parse_detections(detections, conf_threshold)
    
    # Draw bounding boxes
    if len(boxes) > 0:
        # Get unique classes for color assignment
        unique_classes = np.unique(classes)
        colors = list(TABLEAU_COLORS.values())
        class_colors = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}
        
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Get color for this class
            color = class_colors[cls]
            
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add label with confidence
            class_name = COCO_CLASSES[int(cls)] if int(cls) < len(COCO_CLASSES) else f'class_{int(cls)}'
            label = f'{class_name}: {score:.2f}'
            ax.text(x1, y1 - 5, label, fontsize=10, color=color, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add detection count to title
        ax.set_title(f"{title} ({len(boxes)} detections)", fontsize=14, fontweight='bold')
    else:
        # No detections above threshold
        ax.text(0.5, 0.5, f'No detections above {conf_threshold:.2f} confidence', 
                transform=ax.transAxes, ha='center', va='center', fontsize=16, color='orange',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    return fig


def create_comparison_visualization(image, custom_detections, official_detections, 
                                  conf_threshold=0.25, save_path=None):
    """
    Create side-by-side comparison of custom and official model detections.
    
    Args:
        image: Input image
        custom_detections: Detections from custom model
        official_detections: Detections from official model
        conf_threshold: Confidence threshold for filtering
        save_path: Path to save the comparison
    
    Returns:
        matplotlib figure object
    """
    # Convert image to RGB if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
    else:
        image_rgb = np.array(image)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Custom model visualization
    ax1.imshow(image_rgb)
    ax1.set_title("Custom Model Detections", fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Official model visualization
    ax2.imshow(image_rgb)
    ax2.set_title("Official Model Detections", fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # Process custom detections
    custom_boxes, custom_scores, custom_classes = parse_detections(custom_detections, conf_threshold)
    draw_boxes_on_axis(ax1, custom_boxes, custom_scores, custom_classes, "Custom")
    
    # Process official detections
    official_boxes, official_scores, official_classes = parse_detections(official_detections, conf_threshold)
    draw_boxes_on_axis(ax2, official_boxes, official_scores, official_classes, "Official")
    
    # Add summary statistics
    fig.suptitle(f"Model Comparison (conf >= {conf_threshold:.2f})\n"
                f"Custom: {len(custom_boxes)} detections | Official: {len(official_boxes)} detections", 
                fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison visualization saved to: {save_path}")
    
    return fig


def parse_detections(detections, conf_threshold=0.25):
    """Parse detections from various formats into consistent arrays."""
    
    boxes = []
    scores = []
    classes = []
    
    if detections is None or (isinstance(detections, (list, tuple)) and len(detections) == 0):
        return np.array([]).reshape(0, 4), np.array([]), np.array([])
    
    # Handle list of results (from postprocess method)
    if isinstance(detections, list):
        if len(detections) > 0:
            # Take the first result (assuming single image)
            detection = detections[0]
            if hasattr(detection, 'boxes') and detection.boxes is not None:
                boxes_obj = detection.boxes
                if hasattr(boxes_obj, 'data') and len(boxes_obj.data) > 0:
                    detection_data = boxes_obj.data
                    mask = detection_data[:, 4] >= conf_threshold
                    filtered_data = detection_data[mask]
                    if len(filtered_data) > 0:
                        boxes = filtered_data[:, :4].cpu().numpy()
                        scores = filtered_data[:, 4].cpu().numpy()
                        classes = filtered_data[:, 5].cpu().numpy().astype(int)
                elif hasattr(boxes_obj, 'xyxy') and len(boxes_obj.xyxy) > 0:
                    conf_mask = boxes_obj.conf >= conf_threshold
                    if conf_mask.any():
                        boxes = boxes_obj.xyxy[conf_mask].cpu().numpy()
                        scores = boxes_obj.conf[conf_mask].cpu().numpy()
                        classes = boxes_obj.cls[conf_mask].cpu().numpy().astype(int)
                else:
                    # Handle UltralyticsResult with separate boxes, scores, labels
                    if hasattr(detection, 'boxes') and hasattr(detection, 'scores') and hasattr(detection, 'labels'):
                        if len(detection.boxes) > 0:
                            # Filter by confidence threshold
                            mask = detection.scores >= conf_threshold
                            if mask.any():
                                boxes = detection.boxes[mask].cpu().numpy()
                                scores = detection.scores[mask].cpu().numpy()
                                classes = detection.labels[mask].cpu().numpy().astype(int)
        return (np.array(boxes) if len(boxes) > 0 else np.array([]).reshape(0, 4),
                np.array(scores) if len(scores) > 0 else np.array([]),
                np.array(classes) if len(classes) > 0 else np.array([]))
    
    if isinstance(detections, torch.Tensor):
        print(f"DEBUG: Processing torch.Tensor with shape: {detections.shape}")
        if len(detections) > 0:
            print(f"DEBUG: Confidence scores range: {detections[:, 4].min():.3f} - {detections[:, 4].max():.3f}")
            mask = detections[:, 4] >= conf_threshold
            print(f"DEBUG: Objects above threshold: {mask.sum()}")
            filtered_detections = detections[mask]
            if len(filtered_detections) > 0:
                boxes = filtered_detections[:, :4].cpu().numpy()
                scores = filtered_detections[:, 4].cpu().numpy()
                classes = filtered_detections[:, 5].cpu().numpy().astype(int)
                print(f"DEBUG: Extracted {len(boxes)} boxes, {len(scores)} scores, {len(classes)} classes")
    elif hasattr(detections, 'boxes') and detections.boxes is not None:
        print("DEBUG: Processing UltralyticsResult-like object")
        boxes_obj = detections.boxes
        if hasattr(boxes_obj, 'data') and len(boxes_obj.data) > 0:
            print(f"DEBUG: Using boxes.data with shape: {boxes_obj.data.shape}")
            detection_data = boxes_obj.data
            print(f"DEBUG: Confidence scores range: {detection_data[:, 4].min():.3f} - {detection_data[:, 4].max():.3f}")
            mask = detection_data[:, 4] >= conf_threshold
            print(f"DEBUG: Objects above threshold: {mask.sum()}")
            filtered_data = detection_data[mask]
            if len(filtered_data) > 0:
                boxes = filtered_data[:, :4].cpu().numpy()
                scores = filtered_data[:, 4].cpu().numpy()
                classes = filtered_data[:, 5].cpu().numpy().astype(int)
                print(f"DEBUG: Extracted {len(boxes)} boxes, {len(scores)} scores, {len(classes)} classes")
        elif hasattr(boxes_obj, 'xyxy') and len(boxes_obj.xyxy) > 0:
            print(f"DEBUG: Using boxes.xyxy with shape: {boxes_obj.xyxy.shape}")
            print(f"DEBUG: Confidence scores range: {boxes_obj.conf.min():.3f} - {boxes_obj.conf.max():.3f}")
            conf_mask = boxes_obj.conf >= conf_threshold
            print(f"DEBUG: Objects above threshold: {conf_mask.sum()}")
            if conf_mask.any():
                boxes = boxes_obj.xyxy[conf_mask].cpu().numpy()
                scores = boxes_obj.conf[conf_mask].cpu().numpy()
                classes = boxes_obj.cls[conf_mask].cpu().numpy().astype(int)
                print(f"DEBUG: Extracted {len(boxes)} boxes, {len(scores)} scores, {len(classes)} classes")
    else:
        print(f"DEBUG: Unknown detection format: {type(detections)}")
        if hasattr(detections, '__dict__'):
            print(f"DEBUG: Detection attributes: {list(detections.__dict__.keys())}")
    
    result_boxes = np.array(boxes) if len(boxes) > 0 else np.array([]).reshape(0, 4)
    result_scores = np.array(scores) if len(scores) > 0 else np.array([])
    result_classes = np.array(classes) if len(classes) > 0 else np.array([])
    
    print(f"DEBUG: Returning {len(result_boxes)} boxes, {len(result_scores)} scores, {len(result_classes)} classes")
    return result_boxes, result_scores, result_classes


def draw_boxes_on_axis(ax, boxes, scores, classes, model_name):
    """Draw bounding boxes on a matplotlib axis."""
    if len(boxes) == 0:
        ax.text(0.5, 0.5, f'No {model_name.lower()} detections', 
                transform=ax.transAxes, ha='center', va='center', 
                fontsize=16, color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        return
    
    # Get colors for classes
    unique_classes = np.unique(classes)
    colors = list(TABLEAU_COLORS.values())
    class_colors = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}
    
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        color = class_colors[cls]
        
        # Draw bounding box
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        class_name = COCO_CLASSES[int(cls)] if int(cls) < len(COCO_CLASSES) else f'class_{int(cls)}'
        label = f'{class_name}: {score:.2f}'
        ax.text(x1, y1 - 5, label, fontsize=10, color=color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


class UltralyticsResult:
    """
    Result class compatible with Ultralytics format.
    """
    
    def __init__(self, boxes, scores, labels, orig_shape, input_shape):
        """Initialize result object."""
        self.boxes = UltralyticsBoxes(boxes, scores, labels)
        self.orig_shape = orig_shape  # (H, W)
        self.shape = input_shape      # (H, W)
        
    def __len__(self):
        """Return number of detections."""
        return len(self.boxes)


class UltralyticsBoxes:
    """
    Boxes class compatible with Ultralytics format.
    """
    
    def __init__(self, boxes, scores, labels):
        """Initialize boxes object."""
        self.data = torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1)], dim=1) if len(boxes) else torch.zeros(0, 6)
        self.conf = scores
        self.cls = labels
        self.xyxy = boxes
        
    def __len__(self):
        """Return number of boxes."""
        return len(self.data)
    
    def cpu(self):
        """Move boxes to CPU."""
        self.data = self.data.cpu()
        self.conf = self.conf.cpu()
        self.cls = self.cls.cpu()
        self.xyxy = self.xyxy.cpu()
        return self
    
    def numpy(self):
        """Convert boxes to numpy."""
        return self.data.cpu().numpy()


def create_yolomodel(modelname, num_classes=None, ckpt_file=None, fp16=False, device='cuda:0', scale='n', version=None):
    """
    Create YOLO model with specified configuration.
    
    Args:
        modelname (str): Model configuration name
        num_classes (int, optional): Number of classes
        ckpt_file (str, optional): Checkpoint file path
        fp16 (bool): Use half precision
        device (str): Device to load model on
        scale (str): Model scale
        version (str, optional): YOLO version
        
    Returns:
        tuple: (model, preprocess, classes) - YoloDetectionModel, preprocessing function, class names
    """
    # Detect version if not provided
    if version is None:
        version = detect_yolo_version(modelname)
    
    # Create model
    model = YoloDetectionModel(cfg=modelname, scale=scale, nc=num_classes, version=version)
    
    # Load checkpoint if provided
    if ckpt_file and os.path.exists(ckpt_file):
        model = load_checkpoint(model, ckpt_file, fp16=fp16)
    
    # Move to device and set precision
    model = model.to(device)
    if fp16:
        model = model.half()
    
    # Create preprocessing function (using model's built-in preprocessing)
    preprocess = model.preprocess
    
    # Get class names - use COCO classes as default
    classes = COCO_CLASSES if hasattr(model, 'names') and model.names is None else getattr(model, 'names', COCO_CLASSES)
    
    return model, preprocess, classes


def detect_yolo_version(model_path_or_name):
    """
    Detect YOLO version from model path or name.
    
    Args:
        model_path_or_name (str): Model path or configuration name
        
    Returns:
        str: Detected YOLO version
    """
    name_lower = model_path_or_name.lower()
    
    if 'v7' in name_lower or 'yolov7' in name_lower:
        return 'v7'
    elif 'v11' in name_lower or 'yolov11' in name_lower:
        return 'v11'
    elif 'v12' in name_lower or 'yolov12' in name_lower:
        return 'v12'
    else:
        return 'v8'  # Default to v8


def freeze_yolomodel(model, freeze=[]):
    """
    Freeze specified layers in YOLO model.
    
    Args:
        model: YOLO model
        freeze (list): List of layer indices or names to freeze
        
    Returns:
        Model with frozen layers
    """
    if len(freeze) == 0:
        # No layers to freeze, return model as is
        return model
    
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            LOGGER.info(f'Freezing layer {k}')
            v.requires_grad = False
    return model


def test_model_inference_comparison():
    """
    Comprehensive test function that compares custom YOLO model with official Ultralytics model.
    Tests inference on sample images and analyzes detection results.
    """
    print("="*80)
    print("COMPREHENSIVE YOLO MODEL INFERENCE COMPARISON TEST")
    print("="*80)
    
    # Configuration - use safe device selection and local paths
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device = f'cuda:0' if device_count > 0 else 'cpu'
        print(f"CUDA available with {device_count} devices, using {device}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    fp16 = False
    imagepath = 'sampledata/bus.jpg'
    modelcfg_file = 'DeepDataMiningLearning/detection/modules/yolov8.yaml'  # Use local file
    
    # Import the converter function
    from DeepDataMiningLearning.detection.ultralytics_converter import ensure_custom_yolo_checkpoint
    
    # Try to ensure custom checkpoint exists, downloading and converting if necessary
    ckpt_file = ensure_custom_yolo_checkpoint(
        model_name="yolov8n",
        ckpt_path=None,  # Let the function determine the path
        weights_dir="./weights",
        ckpt_dir="./checkpoints", 
        num_classes=80,  # COCO classes
        device=device
    )
    
    # Load default configurations - use local file
    cfgPath = 'DeepDataMiningLearning/detection/modules/default.yaml' #'default.yaml'
    try:
        DEFAULT_CFG_DICT = load_defaultcfgs(cfgPath)
    except:
        print(f"⚠ Could not load config from {cfgPath}, using empty config")
        DEFAULT_CFG_DICT = {}
    
    print(f"Device: {device}")
    print(f"Image path: {imagepath}")
    print(f"Model config: {modelcfg_file}")
    print(f"Checkpoint: {ckpt_file if ckpt_file else 'None found'}")
    print("-" * 80)
    
    # Test 1: Custom Model Creation and Loading
    print("\n1. TESTING CUSTOM MODEL CREATION")
    print("-" * 40)
    
    try:
        # Create custom YOLO model
        custom_model = YoloDetectionModel(cfg=modelcfg_file, scale='n', ch=3)
        print(f"✓ Custom model created: {custom_model.modelname}")
        
        # Load checkpoint if available
        if ckpt_file and os.path.exists(ckpt_file):
            custom_model = load_checkpoint(custom_model, ckpt_file)
            print(f"✓ Checkpoint loaded successfully")
        else:
            print(f"⚠ No checkpoint loaded - using random weights")
        
        # Configure model
        custom_model = custom_model.to(device).eval()
        custom_model = custom_model.half() if fp16 else custom_model.float()
        
        # Get model info
        component_info = custom_model.get_component_info()
        print(f"✓ Model parameters: {component_info['total']['params']:,}")
        
    except Exception as e:
        print(f"✗ Custom model creation failed: {e}")
        return False
    
    # Test 2: Official Model Loading
    print("\n2. TESTING OFFICIAL MODEL LOADING")
    print("-" * 40)
    
    try:
        # Try to import and load official ultralytics model
        try:
            from ultralytics import YOLO
            official_model = YOLO('yolov8n.pt')
            print(f"✓ Official Ultralytics model loaded")
            official_available = True
        except ImportError:
            print("⚠ Ultralytics not available, skipping official model comparison")
            official_available = False
        except Exception as e:
            print(f"⚠ Official model loading failed: {e}")
            official_available = False
            
    except Exception as e:
        print(f"✗ Official model setup failed: {e}")
        official_available = False
    
    # Test 3: Image Loading and Preprocessing
    print("\n3. TESTING IMAGE PREPROCESSING")
    print("-" * 40)
    
    if not os.path.exists(imagepath):
        print(f"✗ Image not found: {imagepath}")
        return False
    
    try:
        # Load image
        im0 = cv2.imread(imagepath)
        print(f"✓ Image loaded: {im0.shape} (H, W, C)")
        
        # Prepare for custom model
        imgs = [im0]
        origimageshapes = [img.shape for img in imgs]
        
        # Custom preprocessing
        yoyotrans = YoloTransform(min_size=640, max_size=640, device=device, fp16=fp16, cfgs=DEFAULT_CFG_DICT)
        imgtensors = yoyotrans(imgs)
        print(f"✓ Custom preprocessing: {imgtensors.shape}")
        
    except Exception as e:
        print(f"✗ Image preprocessing failed: {e}")
        return False
    
    # Test 4: Custom Model Inference
    print("\n4. TESTING CUSTOM MODEL INFERENCE")
    print("-" * 40)
    
    try:
        # Run inference with custom model
        with torch.no_grad():
            preds = custom_model(imgtensors)
            
            # Handle different prediction formats
            if isinstance(preds, tuple):
                preds = preds[0]  # Take first element if tuple
            
            print(f"✓ Custom model prediction shape: {preds.shape}")
            
            # Post-process predictions using the model's postprocess method
            detections = custom_model.postprocess(preds, origimageshapes, input_shape=imgtensors.shape[2:])
            #list of UltralyticsResult object
            if len(detections) > 0:
                custom_detection = detections[0]
                
                # Handle UltralyticsResult object
                if hasattr(custom_detection, 'boxes') and custom_detection.boxes is not None:
                    boxes_obj = custom_detection.boxes
                    if hasattr(boxes_obj, 'data'):
                        # Extract raw tensor data
                        detection_data = boxes_obj.data  # Should be [N, 6] tensor
                        print(f"✓ Custom detections: {len(detection_data)} objects")
                        
                        if len(detection_data) > 0:
                            boxes = detection_data[:, :4]  # x1, y1, x2, y2
                            scores = detection_data[:, 4]
                            classes = detection_data[:, 5].int()
                            
                            print(f"  - Confidence range: {scores.min():.3f} - {scores.max():.3f}")
                            print(f"  - Unique classes: {torch.unique(classes).tolist()}")
                            print(f"  - Box coordinates range: [{boxes.min():.1f}, {boxes.max():.1f}]")
                    else:
                        print("✓ Custom model: No detection data available")
                        detection_data = torch.empty(0, 6)
                elif isinstance(custom_detection, torch.Tensor):
                    # Handle raw tensor format
                    detection_data = custom_detection
                    print(f"✓ Custom detections: {len(detection_data)} objects")
                    
                    if len(detection_data) > 0:
                        boxes = detection_data[:, :4]  # x1, y1, x2, y2
                        scores = detection_data[:, 4]
                        classes = detection_data[:, 5].int()
                        
                        print(f"  - Confidence range: {scores.min():.3f} - {scores.max():.3f}")
                        print(f"  - Unique classes: {torch.unique(classes).tolist()}")
                        print(f"  - Box coordinates range: [{boxes.min():.1f}, {boxes.max():.1f}]")
                else:
                    print("✓ Custom model: No detections")
                    detection_data = torch.empty(0, 6)
            else:
                print("✓ Custom model: No detections")
                detection_data = torch.empty(0, 6)

            # Custom model visualization
            custom_fig = visualize_detections(
                image=im0,
                detections=detections,
                title="Custom Model Detections",
                conf_threshold=0.25,
                save_path="custom_model_detections.png"
            )

    except Exception as e:
        print(f"✗ Custom model inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Official Model Inference (if available)
    print("\n5. TESTING OFFICIAL MODEL INFERENCE")
    print("-" * 40)
    
    official_results = None
    if official_available:
        try:
            # Run inference with official model
            official_results = official_model(imagepath, verbose=False)
            
            if len(official_results) > 0 and official_results[0].boxes is not None:
                boxes = official_results[0].boxes
                print(f"✓ Official detections: {len(boxes)} objects")
                
                if len(boxes) > 0:
                    conf_scores = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)
                    coords = boxes.xyxy.cpu().numpy()
                    
                    print(f"  - Confidence range: {conf_scores.min():.3f} - {conf_scores.max():.3f}")
                    print(f"  - Unique classes: {np.unique(classes).tolist()}")
                    print(f"  - Box coordinates range: [{coords.min():.1f}, {coords.max():.1f}]")
            else:
                print("✓ Official model: No detections")
            
            # Official model visualization
            official_fig = visualize_detections(
                image=im0,
                detections=official_results,
                title="Official Model Detections", 
                conf_threshold=0.25,
                save_path="official_model_detections.png"
            )

        except Exception as e:
            print(f"✗ Official model inference failed: {e}")
            official_available = False
    


# def test_yolov7weights():
#     """Test YOLOv7 weights loading."""
#     print("Testing YOLOv7 weights loading...")
    
#     try:
#         # Check if yolov7.yaml exists
#         if not os.path.exists('yolov7.yaml'):
#             print("✗ yolov7.yaml not found")
#             return False
            
#         # Create YOLOv7 model
#         model = YoloDetectionModel('yolov7.yaml', scale='n', version='v7')
#         print(f"✓ YOLOv7 model created successfully")
        
#         # Test components with error handling
#         try:
#             results = model.test_components(verbose=False)
#             if results:
#                 print(f"✓ Component tests completed")
#             else:
#                 print(f"⚠ Component tests returned None")
#         except Exception as e:
#             print(f"⚠ Component test failed: {e}")
        
#         return True
        
#     except Exception as e:
#         print(f"✗ YOLOv7 test failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return False


# Main execution and testing
if __name__ == "__main__":
    # Run comprehensive inference comparison test
    print("Starting comprehensive model testing...")
    test_success = test_model_inference_comparison()
    
    if test_success:
        print("\n✓ All tests completed successfully!")
    else:
        print("\n✗ Some tests failed. Check the output above for details.")
    
    # Optional: Run YOLOv7 test as well
    # print("\n" + "="*60)
    # print("ADDITIONAL YOLOV7 TESTING")
    # print("="*60)
    #test_yolov7weights()



