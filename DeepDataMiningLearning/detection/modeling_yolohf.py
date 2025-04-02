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

# Import necessary modules from your project
from DeepDataMiningLearning.detection.modules.block import (
    AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
    Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d,
    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, RepC3, RepConv, MP, SPPCSPC
)
from DeepDataMiningLearning.detection.modules.head import Detect, IDetect, Classify, Pose, RTDETRDecoder, Segment
#from DeepDataMiningLearning.detection.modules.utils import LOGGER
from DeepDataMiningLearning.detection.modules.anchor import check_anchor_order

# Define blocks that take two arguments
twoargs_blocks = [
    nn.Conv2d, Classify, Conv, ConvTranspose, GhostConv, RepConv, Bottleneck, GhostBottleneck, 
    SPP, SPPF, SPPCSPC, DWConv, Focus, BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, 
    nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3
]

coco_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

"""
YOLOv8 model configuration as a Python dictionary.
This module replaces the need to load the YAML file at runtime.
"""

# YOLOv8 configuration dictionary
YOLOV8_CONFIG = {
    # Parameters
    "nc": 80,  # number of classes
    "scales": {
        # [depth, width, max_channels]
        "n": [0.33, 0.25, 1024],  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
        "s": [0.33, 0.50, 1024],  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
        "m": [0.67, 0.75, 768],   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
        "l": [1.00, 1.00, 512],   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
        "x": [1.00, 1.25, 512],   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
    },

    # YOLOv8.0n backbone
    "backbone": [
        # [from, repeats, module, args]
        [-1, 1, "Conv", [64, 3, 2]],  # 0-P1/2
        [-1, 1, "Conv", [128, 3, 2]],  # 1-P2/4
        [-1, 3, "C2f", [128, True]],
        [-1, 1, "Conv", [256, 3, 2]],  # 3-P3/8
        [-1, 6, "C2f", [256, True]],
        [-1, 1, "Conv", [512, 3, 2]],  # 5-P4/16
        [-1, 6, "C2f", [512, True]],
        [-1, 1, "Conv", [1024, 3, 2]],  # 7-P5/32
        [-1, 3, "C2f", [1024, True]],
        [-1, 1, "SPPF", [1024, 5]],  # 9
    ],

    # YOLOv8.0n head
    "head": [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],  # cat backbone P4
        [-1, 3, "C2f", [512]],  # 12

        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],  # cat backbone P3
        [-1, 3, "C2f", [256]],  # 15 (P3/8-small)

        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 12], 1, "Concat", [1]],  # cat head P4
        [-1, 3, "C2f", [512]],  # 18 (P4/16-medium)

        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 9], 1, "Concat", [1]],  # cat head P5
        [-1, 3, "C2f", [1024]],  # 21 (P5/32-large)

        [[15, 18, 21], 1, "Detect", ["nc"]],  # Detect(P3, P4, P5)
    ],
    
    # Additional parameters
    "inplace": True,
    "ch": 3
}

def get_yolo_config(scale='s', nc=80, ch=3):
    """
    Get a copy of the YOLO configuration with the specified scale, number of classes, and channels.
    
    Args:
        scale (str): Model scale - 'n', 's', 'm', 'l', or 'x'
        nc (int): Number of classes
        ch (int): Number of input channels
        
    Returns:
        dict: YOLO configuration dictionary
    """
    # Create a deep copy to avoid modifying the original
    import copy
    config = copy.deepcopy(YOLOV8_CONFIG)
    
    # Update parameters
    config['scale'] = scale
    config['nc'] = nc
    config['ch'] = ch
    
    return config

class YoloTransform:
    """
    Handles preprocessing and postprocessing for YOLO models.
    """
    def __init__(self, min_size=640, max_size=640, device='cuda', fp16=False, use_letterbox=True):
        self.min_size = min_size
        self.max_size = max_size
        self.device = device
        self.fp16 = fp16
        self.use_letterbox = use_letterbox
        if use_letterbox:
            # Initialize letterbox for better aspect ratio handling
            from DeepDataMiningLearning.detection.modules.yolotransform import LetterBox
            self.letterbox = LetterBox((min_size, max_size), auto=True, stride=32)
        
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
            if self.use_letterbox:
                # Use letterbox instead of simple resize to maintain aspect ratio
                img_resized = self.letterbox(image=img_rgb)
            else:
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
                if self.use_letterbox:
                    # Use the existing scale_boxes function which already handles letterbox
                    pred[:, :4] = scale_boxes(img_size, pred[:, :4], orig_shape)
                    #padding default is True , the function assumes the boxes are based on an image that was processed with letterbox padding
                else:
                    # Standard scaling for non-letterbox case (though scale_boxes would work here too)
                    pred[:, :4] = scale_boxes(img_size, pred[:, :4], orig_shape, padding=False)
            
            # Format detections as expected by the trainer
            detection = {
                "boxes": pred[:, :4].detach().cpu(),
                "scores": pred[:, 4].detach().cpu(),
                "labels": pred[:, 5].detach().cpu()
            }
            detections.append(detection)
        
        return detections

# Utility functions
def yaml_load(file='data.yaml', append_filename=True):
    """Load YAML data from a file."""
    assert Path(file).suffix in ('.yaml', '.yml'), f'Attempting to load non-YAML file {file} with yaml_load()'
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        data = yaml.safe_load(s) or {}
        if append_filename:
            data['yaml_file'] = str(file)
        return data

def extract_filename(path):
    """Extract filename from path without extension."""
    return Path(path).stem

def make_divisible(x, divisor):
    """Return nearest x divisible by divisor."""
    return int(np.ceil(x / divisor) * divisor)

def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

# def intersect_dicts(da, db, exclude=()):
#     """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
#     return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            print(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]
    elif all(key in d for key in ('depth_multiple', 'width_multiple')):
        depth = d['depth_multiple']
        width = d['width_multiple']
    
    if "anchors" in d.keys():
        anchors = d['anchors']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
        no = na * (nc + 5)
    else:
        no = nc
   
    if act:
        Conv.default_act = eval(act)
        # if verbose:
        #     LOGGER.info(f"activation: {act}")

    # if verbose:
    #     LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n
        if m in twoargs_blocks:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                args.insert(2, n)
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (IDetect, Detect, Segment, Pose):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        m.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        # if verbose:
        #     LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

from transformers import PretrainedConfig
class YoloConfig(PretrainedConfig):
    """Configuration class for YOLOv8 models."""
    model_type = "yolov8"
    
    def __init__(
        self,
        scale="s",
        nc=80,
        ch=3,
        min_size=640,
        max_size=640,
        use_fp16=False,
        id2label=None,
        label2id=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scale = scale
        self.nc = nc  # number of classes
        self.ch = ch  # number of channels
        self.min_size = min_size
        self.max_size = max_size
        self.use_fp16 = use_fp16
        
        # Set up id2label and label2id mappings
        if id2label is None:
            id2label = {str(i): f"class_{i}" for i in range(nc)}
        self.id2label = id2label
        
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}
        self.label2id = label2id
        
        # Add model architecture info
        self.architectures = ["YoloDetectionModel"]

def register_yolo_architecture():
    """
    Register the YOLOv8 model architecture with the Hugging Face transformers library
    for full integration with the transformers ecosystem.
    """
    from transformers import AutoConfig, AutoModel
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_FOR_OBJECT_DETECTION_MAPPING
    
    # Register the config
    CONFIG_MAPPING.register("yolov8", YoloConfig)
    
    # Register the model architecture
    MODEL_MAPPING.register(YoloConfig, YoloDetectionModel)
    MODEL_FOR_OBJECT_DETECTION_MAPPING.register(YoloConfig, YoloDetectionModel)
    
    print("YOLOv8 architecture registered successfully with Hugging Face transformers")

def register_yolo_architecture2():
    """
    Register the YOLOv8 model architecture with the Hugging Face transformers library
    for full integration with the transformers ecosystem.
    """
    from transformers import AutoConfig, AutoModel, AutoModelForObjectDetection
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_FOR_OBJECT_DETECTION_MAPPING
    from transformers.models.auto.processing_auto import PROCESSOR_MAPPING, AutoProcessor
    
    # Import the image processor
    #from DeepDataMiningLearning.detection.image_processor import YoloImageProcessor
    # Import DETR's image processor
    from transformers import DetrImageProcessor
    
    # Register the config
    CONFIG_MAPPING.register("yolov8", YoloConfig)
    
    # Register the model architecture
    MODEL_MAPPING.register(YoloConfig, YoloDetectionModel)
    MODEL_FOR_OBJECT_DETECTION_MAPPING.register(YoloConfig, YoloDetectionModel)
    
    # Register the image processor
    PROCESSOR_MAPPING.register(YoloConfig, YoloImageProcessor)
    
    print("YOLOv8 architecture registered successfully with Hugging Face transformers")
    
from transformers import ImageProcessingMixin
import numpy as np
from PIL import Image
import torch

class YoloImageProcessor(ImageProcessingMixin):
    """
    Image processor for YOLO models.
    
    This processor handles image resizing, normalization, and formatting for YOLO models.
    """
    
    model_input_names = ["pixel_values"]
    
    def __init__(
        self,
        do_resize=True,
        size=640,
        resample="bilinear",
        do_normalize=True,
        do_rescale=True,
        rescale_factor=1/255.0,
        do_pad=True,
        pad_size_divisor=32,
        pad_value=114,
        do_convert_rgb=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size if isinstance(size, dict) else {"height": size, "width": size}
        self.resample = resample
        self.do_normalize = do_normalize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.do_convert_rgb = do_convert_rgb
        
    def resize(self, image, size, resample="bilinear"):
        """
        Resize an image to the given size.
        """
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image)
            
        if isinstance(size, dict):
            size = (size["height"], size["width"])
        elif isinstance(size, int):
            size = (size, size)
            
        resample_map = {
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST,
            "lanczos": Image.LANCZOS,
        }
        resample = resample_map.get(resample, Image.BILINEAR)
        
        return image.resize(size, resample)
    
    def pad(self, image, pad_size_divisor=32, pad_value=114):
        """
        Pad an image to make its dimensions divisible by pad_size_divisor.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        height, width = image.shape[:2]
        new_height = int(np.ceil(height / pad_size_divisor) * pad_size_divisor)
        new_width = int(np.ceil(width / pad_size_divisor) * pad_size_divisor)
        
        # Create padded image
        padded_image = np.full((new_height, new_width, 3), pad_value, dtype=np.uint8)
        padded_image[:height, :width] = image
        
        return padded_image
    
    def preprocess(
        self,
        images,
        do_resize=None,
        size=None,
        resample=None,
        do_normalize=None,
        do_rescale=None,
        rescale_factor=None,
        do_pad=None,
        pad_size_divisor=None,
        pad_value=None,
        do_convert_rgb=None,
        return_tensors=None,
        **kwargs
    ):
        """
        Preprocess an image or batch of images for YOLO models.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_pad = do_pad if do_pad is not None else self.do_pad
        pad_size_divisor = pad_size_divisor if pad_size_divisor is not None else self.pad_size_divisor
        pad_value = pad_value if pad_value is not None else self.pad_value
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        
        # Handle single image
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]
            
        # Process each image
        processed_images = []
        for image in images:
            # Convert to RGB if needed
            if do_convert_rgb and isinstance(image, Image.Image) and image.mode != "RGB":
                image = image.convert("RGB")
                
            # Resize if needed
            if do_resize:
                image = self.resize(image, size, resample)
                
            # Convert to numpy array if it's a PIL Image
            if isinstance(image, Image.Image):
                image = np.array(image)
                
            # Pad if needed
            if do_pad:
                image = self.pad(image, pad_size_divisor, pad_value)
                
            # Rescale if needed
            if do_rescale:
                image = image * rescale_factor
                
            # Normalize if needed
            if do_normalize:
                # YOLO models typically don't need normalization beyond rescaling
                pass
                
            processed_images.append(image)
            
        # Convert to tensors if requested
        if return_tensors == "pt":
            processed_images = [torch.tensor(img).permute(2, 0, 1).float() for img in processed_images]
            processed_images = torch.stack(processed_images)
            
        return {"pixel_values": processed_images}
    
    def post_process_object_detection(
        self,
        outputs,
        threshold=0.5,
        target_sizes=None,
    ):
        """
        Post-process the raw outputs of the model for object detection.
        """
        # Get predictions
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("pred_logits"))
            boxes = outputs.get("pred_boxes")
        else:
            logits, boxes = outputs
            
        # Convert to list of batches
        results = []
        
        batch_size = logits.shape[0]
        for i in range(batch_size):
            # Get scores and labels
            scores = torch.sigmoid(logits[i])
            labels = torch.arange(scores.shape[1]).unsqueeze(0).expand_as(scores)
            
            # Apply threshold
            mask = scores > threshold
            scores = scores[mask]
            labels = labels[mask]
            boxes_i = boxes[i][mask.any(dim=1)]
            
            # Rescale boxes if target sizes provided
            if target_sizes is not None:
                orig_h, orig_w = target_sizes[i]
                scale_x = orig_w / self.size["width"]
                scale_y = orig_h / self.size["height"]
                
                boxes_i[:, [0, 2]] *= scale_x
                boxes_i[:, [1, 3]] *= scale_y
                
                # Ensure boxes are within image boundaries
                boxes_i[:, 0].clamp_(min=0, max=orig_w)
                boxes_i[:, 1].clamp_(min=0, max=orig_h)
                boxes_i[:, 2].clamp_(min=0, max=orig_w)
                boxes_i[:, 3].clamp_(min=0, max=orig_h)
            
            results.append({
                "scores": scores,
                "labels": labels,
                "boxes": boxes_i
            })
            
        return results
    
class YoloDetectionModel(nn.Module):
    """YOLOv8 detection model with HuggingFace-compatible interface."""
    def __init__(self, cfg=None, scale='n', ch=3, nc=None, device='cuda', use_fp16=False, min_size=640, max_size=640):
        super().__init__()
        # If a config object is provided, use its parameters
        if isinstance(cfg, YoloConfig):
            scale = cfg.scale
            nc = cfg.nc
            ch = cfg.ch
            use_fp16 = cfg.use_fp16
            min_size = cfg.min_size
            max_size = cfg.max_size
        #self.yaml = cfg if isinstance(cfg, dict) else yaml_load(cfg)
        #yaml_path = "DeepDataMiningLearning/detection/modules/yolov8.yaml"
        #self.yaml = yaml_load(yaml_path)
        #Using Python configuration instead of YAML
        self.yaml = get_yolo_config(scale, nc, ch)
        self.yaml['scale'] = scale
        #self.modelname = extract_filename(yaml_path)
        self.modelname = "yolov8"
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
            print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        
        # Parse model and get component indices based on scale
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=False)
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
        self.transform = YoloTransform(min_size=min_size, max_size=max_size, device=device, fp16=use_fp16, use_letterbox=True)
    
        # Add this class method for loading from pretrained
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a YOLOv8 model from a pretrained model directory or Hugging Face Hub.
        
        Args:
            pretrained_model_name_or_path (str): Path to a local directory or HF Hub model ID
            *model_args: Additional positional arguments passed to the model
            **kwargs: Additional keyword arguments passed to the model
            
        Returns:
            YoloDetectionModel: Loaded model instance
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        import os
        import json
        
        # Register architecture to ensure it's recognized
        register_yolo_architecture()
        
        # Load config
        config = None
        try:
            # Try to load the config file
            config_file = cached_file(
                pretrained_model_name_or_path,
                CONFIG_NAME,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            
            if config_file:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                
                # Create config object
                config = YoloConfig(**config_dict)
            else:
                print("Config file not found, using default config")
                config = YoloConfig()
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default config")
            config = YoloConfig()
        
        # Create model instance with config
        model = cls(cfg=config)
        
        # Load weights
        try:
            weights_file = cached_file(
                pretrained_model_name_or_path,
                WEIGHTS_NAME,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            
            if weights_file:
                # Load state dict
                state_dict = torch.load(weights_file, map_location="cpu")
                model.load_state_dict(state_dict)
                print(f"Loaded weights from {weights_file}")
            else:
                print("Weights file not found")
        except Exception as e:
            print(f"Error loading weights: {e}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Set config attribute
        model.config = config
        
        return model
    
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
    
    def forward(self, x=None, pixel_values=None, images=None, **kwargs):
        """
        Forward pass of the model with HuggingFace-compatible interface.
        
        Args:
            x: Original input tensor parameter
            pixel_values: Input tensor (HuggingFace standard name)
            images: Alternative input tensor name
            **kwargs: Additional keyword arguments
            
        Returns:
            dict or list: Detection results
        """
        # Handle different input parameter names for compatibility
        if x is None:
            if pixel_values is not None:
                x = pixel_values
            elif images is not None:
                x = images
            elif 'inputs' in kwargs:
                x = kwargs['inputs']
            else:
                # Try to find any tensor in kwargs
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor) and len(v.shape) == 4:
                        x = v
                        break
                else:
                    raise ValueError("No valid input tensor found in arguments. Expected 'x', 'pixel_values', or 'images'")
        
        # Continue with the existing forward logic
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

    def inference_test(self, image_path, conf_thres=0.25, iou_thres=0.45, max_det=300, visualize=True, output_dir=None):
        """
        Run inference on a single image or all images in a folder and optionally visualize the results.
        
        Args:
            image_path (str): Path to the input image or directory containing images
            conf_thres (float): Confidence threshold for detections (0.0 to 1.0)
            iou_thres (float): IoU threshold for NMS (0.0 to 1.0)
            max_det (int): Maximum number of detections per image
            visualize (bool): Whether to visualize and save the results with bounding boxes
            output_dir (str, optional): Directory to save visualization results. If None, 
                                        will save in the same directory as input images
            
        Returns:
            dict or list: 
                - For a single image: Dictionary containing the processed detections with keys:
                  * "boxes": tensor of shape (num_detections, 4) with coordinates in (x1, y1, x2, y2) format
                  * "scores": tensor of shape (num_detections,) with confidence scores
                  * "labels": tensor of shape (num_detections,) with class indices
                - For a directory: List of dictionaries, one for each image
        """
        # Check if image_path is a directory or a file
        if os.path.isdir(image_path):
            # Process all images in the directory
            image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            
            # Create output directory if specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            # Process each image
            results = []
            for img_file in tqdm(image_files, desc="Processing images"):
                try:
                    # Process single image and append results
                    detection = self._process_single_image(
                        img_file, 
                        conf_thres, 
                        iou_thres, 
                        max_det, 
                        visualize, 
                        output_dir
                    )
                    results.append({
                        'file': img_file,
                        'detections': detection
                    })
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
            
            return results
        else:
            # Process a single image
            return self._process_single_image(
                image_path, 
                conf_thres, 
                iou_thres, 
                max_det, 
                visualize, 
                output_dir
            )
    
    def _process_single_image(self, image_path, conf_thres=0.25, iou_thres=0.45, max_det=300, visualize=True, output_dir=None):
        """
        Helper method to process a single image.
        
        Args:
            image_path (str): Path to the input image
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
            max_det (int): Maximum number of detections
            visualize (bool): Whether to visualize and save the results
            output_dir (str, optional): Directory to save visualization results
            
        Returns:
            dict: Dictionary containing the processed detections
        """
        # Load image
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Run inference using the model's forward method
        detections = self.forward(
            img_orig,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det
        )
        
        # Visualize results if requested
        if visualize and len(detections) > 0:
            # Create a copy of the image for visualization
            img_vis = img_orig.copy()
            
            # Extract detection components
            boxes = detections["boxes"].numpy()
            scores = detections["scores"].numpy()
            labels = detections["labels"].numpy()
            
            # Draw boxes on the image
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                score = scores[i]
                label = int(labels[i])
                
                # Generate a color based on the class label for better visualization
                # This creates a unique color for each class
                color_factor = (label * 50) % 255
                color = (color_factor, 255 - color_factor, 128)
                
                # Draw bounding box
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                
                # Add class name and confidence
                class_name = self.names.get(int(label))
                if class_name is None or class_name == f"{int(label)}":
                    # Try to get class name from COCO names if available
                    try:
                        #from DeepDataMiningLearning.detection.data.coco_names import COCO_NAMES as coco_names
                        class_name = coco_names.get(int(label), f"class_{label}")
                    except ImportError:
                        class_name = f"class_{label}"
                
                label_text = f"{class_name}: {score:.2f}"
                
                # Add a filled rectangle behind text for better visibility
                text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img_vis, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                
                # Add text with white color for better contrast
                cv2.putText(img_vis, label_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Determine output path
            if output_dir:
                # Use the specified output directory
                base_filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"{os.path.splitext(base_filename)[0]}_detected{os.path.splitext(base_filename)[1]}")
            else:
                # Save in the same directory as the input image
                output_path = image_path.replace('.', '_detected.')
            
            # Save the visualization
            cv2.imwrite(output_path, img_vis)
            print(f"Visualization saved to {output_path}")
            
            # Add visualization path to detections
            detections["visualization_path"] = output_path
        
        return detections
    
def upload_to_huggingface(model, repo_id, token=None, commit_message="Upload YOLO model", private=False, 
                          create_model_card=True, example_images=None, model_description=None):
    """
    Upload a YOLO model to HuggingFace Hub.
    
    Args:
        model (YoloDetectionModel): The YOLO model to upload
        repo_id (str): HuggingFace repository ID (e.g., 'username/model-name')
        token (str, optional): HuggingFace token. If None, will use the token from the environment.
        commit_message (str): Commit message for the upload
        private (bool): Whether to create a private repository
        create_model_card (bool): Whether to create a model card
        example_images (list, optional): List of paths to example images to include in the model card
        model_description (str, optional): Custom description for the model card
        
    Returns:
        str: URL of the uploaded model
    """
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
        from huggingface_hub.utils import validate_repo_id
    except ImportError:
        raise ImportError("huggingface_hub package is required to upload models. Install it with 'pip install huggingface_hub'.")
    
    # Create a temporary directory for files to upload
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Create config object if not already part of the model
        if not hasattr(model, 'config') or model.config is None:
            # Create a proper YoloConfig object, not just a dictionary
            config = YoloConfig(
                scale=model.scale,
                nc=model.yaml.get('nc', 80),
                ch=model.yaml.get('ch', 3),
                min_size=model.transform.min_size,
                max_size=model.transform.max_size,
                use_fp16=getattr(model.transform, 'fp16', False)
            )
            # Save config
            config.save_pretrained(temp_dir)
        elif isinstance(model.config, dict):
            # If model.config is a dictionary, convert it to a YoloConfig object
            config = YoloConfig(**model.config)
            config.save_pretrained(temp_dir)
        else:
            # If model already has a config object, save it
            model.config.save_pretrained(temp_dir)
            
        # Save model state dict
        torch.save(model.state_dict(), os.path.join(temp_dir, "pytorch_model.bin"))
        
        # Validate repository ID
        validate_repo_id(repo_id)
        
        # Create repository if it doesn't exist
        api = HfApi()
        try:
            create_repo(repo_id, token=token, exist_ok=True, private=private)
            print(f"Repository created/verified: {repo_id}")
        except Exception as e:
            print(f"Repository creation warning (can be ignored if repo exists): {e}")
        
        # Save model state dict
        model_name = model.modelname
        scale = model.scale
        model_path = os.path.join(temp_dir, f"{model_name}_{scale}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model state dict saved to: {model_path}")
        
        # Save model configuration
        config_path = os.path.join(temp_dir, f"config.json")
        
        # Create a more detailed config
        detailed_config = {
            "model_type": "yolov8",
            "scale": scale,
            "num_classes": model.yaml.get('nc', 80),
            "image_size": [model.transform.min_size, model.transform.max_size],
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "max_detections": 300,
            "backbone_type": model.yaml.get('backbone_type', 'default'),
            "id2label": {str(i): model.names.get(i, f"class_{i}") for i in range(model.yaml.get('nc', 80))},
            "label2id": {model.names.get(i, f"class_{i}"): str(i) for i in range(model.yaml.get('nc', 80))},
            "architectures": ["YoloDetectionModel"],
        }
        
        with open(config_path, 'w') as f:
            json.dump(detailed_config, f, indent=2)
        print(f"Model configuration saved to: {config_path}")
        
        # Create preprocessor config
        preprocessor_config_path = os.path.join(temp_dir, "preprocessor_config.json")
        preprocessor_config = {
            "do_normalize": True,
            "do_resize": True,
            "do_rescale": True,
            "image_mean": [0.0, 0.0, 0.0],  # YOLO uses 0-1 normalization
            "image_std": [1.0, 1.0, 1.0],
            "rescale_factor": 1/255.0,
            "size": {
                "height": model.transform.min_size,
                "width": model.transform.max_size
            },
            "use_letterbox": model.transform.use_letterbox
        }
        
        with open(preprocessor_config_path, 'w') as f:
            json.dump(preprocessor_config, f, indent=2)
        print(f"Preprocessor configuration saved to: {preprocessor_config_path}")
        
        # Create model card if requested
        if create_model_card:
            model_card_path = os.path.join(temp_dir, "README.md")
            create_yolo_model_card(
                model_name=f"{model_name}_{scale}",
                scale=scale,
                num_classes=model.yaml.get('nc', 80),
                repo_id=repo_id,
                output_path=model_card_path,
                example_images=example_images,
                description=model_description
            )
            print(f"Model card created at: {model_card_path}")
        
        # Create a simple example script
        example_script_path = os.path.join(temp_dir, "example.py")
        with open(example_script_path, 'w') as f:
            f.write(f"""
# Example script for using the {model_name}_{scale} model from Hugging Face
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# Load model and processor
model = AutoModelForObjectDetection.from_pretrained("{repo_id}")
processor = AutoImageProcessor.from_pretrained("{repo_id}")

# Function to run inference on an image
def detect_objects(image_path, confidence_threshold=0.25):
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process outputs
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, 
        threshold=confidence_threshold,
        target_sizes=target_sizes
    )[0]
    
    # Print results
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {{model.config.id2label[label.item()]}} with confidence "
            f"{{round(score.item(), 3)}} at location {{box}}"
        )
    
    return results

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "path/to/your/image.jpg"
    detect_objects(image_path)
""")
            print(f"Example script created at: {example_script_path}")
        
        # Upload all files to HuggingFace
        print(f"Uploading files to HuggingFace Hub: {repo_id}")
        upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message
        )
        
        print(f"Model uploaded successfully to: https://huggingface.co/{repo_id}")
        return f"https://huggingface.co/{repo_id}"
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


def upload_to_huggingface2(model, repo_id, token=None, commit_message="Upload model", 
                         private=False, create_model_card=False, example_images=None,
                         model_description=None):
    """
    Upload a model to the Hugging Face Hub.
    
    Args:
        model: The model to upload
        repo_id: The repository ID on Hugging Face Hub
        token: Hugging Face API token
        commit_message: Commit message for the upload
        private: Whether the repository should be private
        create_model_card: Whether to create a model card
        example_images: List of example image paths to include in the model card
        model_description: Custom description for the model card
    """
    import tempfile
    import os
    import shutil
    import json
    from transformers import PretrainedConfig
    
    # Create a temporary directory to save the model files
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Save the model state dict
        model_path = os.path.join(temp_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model state dict to {model_path}")
        
        # Create and save the config
        if hasattr(model, 'config') and isinstance(model.config, PretrainedConfig):
            config = model.config
        else:
            # Create a config object if the model doesn't have one
            from transformers import PretrainedConfig
            config_dict = {
                "model_type": "detr",  # Use DETR model type instead of custom yolov8
                "architectures": ["YoloDetectionModel"],
                "scale": model.scale,
                "num_classes": model.yaml.get('nc', 80),
                "image_size": 640,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "max_detections": 300
            }
            config = PretrainedConfig.from_dict(config_dict)
        
        # Save the config
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, 'w') as f:
            f.write(config.to_json_string())
        print(f"Saved config to {config_path}")
        
        # Add image processor config
        processor_config = {
            "image_processor_type": "DetrImageProcessor",  # Use DETR's image processor "YoloImageProcessor",
            "do_normalize": True,
            "do_resize": True,
            "do_rescale": True,
            "do_pad": True,
            "size": {
                "height": 640,
                "width": 640
            },
            "resample": "bilinear",
            "rescale_factor": 0.00392156862745098,  # 1/255
            "do_convert_rgb": True,
            "pad_size_divisor": 32,
            "pad_value": 114
        }
        
        # Save the image processor config
        processor_config_path = os.path.join(temp_dir, "preprocessor_config.json")
        with open(processor_config_path, 'w') as f:
            json.dump(processor_config, f, indent=2)
        print(f"Saved image processor config to {processor_config_path}")
        
        # Create a model card if requested
        if create_model_card:
            model_name = f"YOLOv8 {model.scale.upper()}"
            readme_path = os.path.join(temp_dir, "README.md")
            create_yolo_model_card(
                model_name=model_name,
                scale=model.scale,
                num_classes=model.yaml.get('nc', 80),
                repo_id=repo_id,
                output_path=readme_path,
                example_images=example_images,
                description=model_description
            )
            print(f"Created model card at {readme_path}")
        
        # Upload to Hugging Face Hub
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Create the repository if it doesn't exist
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        print(f"Created/verified repository: {repo_id}")
        
        # Upload the files
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            commit_message=commit_message,
            token=token
        )
        print(f"Uploaded model to {repo_id}")
        
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
        
import os

def create_yolo_model_card(model_name, scale, num_classes, repo_id, output_path="README.md", 
                          example_images=None, description=None):
    """
    Create a detailed model card for a YOLO model to be uploaded to HuggingFace.
    
    Args:
        model_name (str): Name of the model
        scale (str): Scale of the model (n, s, m, l, x)
        num_classes (int): Number of classes the model can detect
        repo_id (str): HuggingFace repository ID
        output_path (str): Path to save the model card
        example_images (list, optional): List of paths to example images to include in the model card
        description (str, optional): Custom description for the model card
        
    Returns:
        str: Path to the created model card
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Scale descriptions
    scale_descriptions = {
        'n': "nano (smallest, fastest)",
        's': "small (good balance of speed and accuracy)",
        'm': "medium (higher accuracy, moderate speed)",
        'l': "large (high accuracy, slower inference)",
        'x': "xlarge (highest accuracy, slowest inference)"
    }
    
    scale_desc = scale_descriptions.get(scale, f"custom scale '{scale}'")
    
    # Default description if none provided
    if description is None:
        description = f"""
This model is a YOLOv8 object detection model with scale '{scale}' ({scale_desc}). 
It can detect {num_classes} different classes and is optimized for real-time object detection.

YOLOv8 is the latest version in the YOLO (You Only Look Once) family of models and 
offers improved accuracy and speed compared to previous versions.
"""
    
    # Create the model card content
    model_card = f"""---
language: en
license: mit
tags:
- object-detection
- yolov8
- computer-vision
- pytorch
- transformers
datasets:
- coco
---

# {model_name} - YOLOv8 Object Detection Model

{description}

## Model Details

- **Model Type:** YOLOv8
- **Scale:** {scale} ({scale_desc})
- **Number of Classes:** {num_classes}
- **Input Size:** 640x640
- **Framework:** PyTorch + Transformers

## Usage

### With Transformers Pipeline

```python
from transformers import pipeline

detector = pipeline("object-detection", model="{repo_id}")
result = detector("path/to/image.jpg")
print(result)
"""

def test_localmodel(use_fp16=True):
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_path = "DeepDataMiningLearning/detection/modules/yolov8.yaml"
    model = YoloDetectionModel(cfg=yaml_path, scale='s', nc=80, ch=3, \
        device=device, use_fp16=use_fp16, min_size=640, max_size=640)
    model.load_state_dict(torch.load("../modelzoo/yolov8s_statedicts.pt"))
    model = model.to(device)
    model.eval()
    
    # Enable FP16 precision if requested
    if use_fp16 and device.type == 'cuda':
        model = model.half()
        print("Using FP16 precision for faster inference")
    
    model.inference_test(
        image_path="/DATA5T2/Dataset/Kitti/testing/image_2",
        visualize=True,
        output_dir="output/"
    )

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

def test_upload_model():
    """
    Test function to upload a YOLO model to HuggingFace Hub.
    """
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_path = "DeepDataMiningLearning/detection/modules/yolov8.yaml"
    model = YoloDetectionModel(cfg=yaml_path, scale='s', nc=80, ch=3,
    device=device, use_fp16=True, min_size=640, max_size=640)
    
    # Load weights
    model.load_state_dict(torch.load("../modelzoo/yolov8s_statedicts.pt"))
    model = model.to(device)
    model.eval()
    
    # Upload to HuggingFace
    # Replace with your HuggingFace username and desired repository name
    repo_id = "lkk688/yolov8s-model"

    # Example images for the model card (optional)
    example_images = [
        "sampledata/bus.jpg",
        "sampledata/sjsupeople.jpg"
    ]
    
    # Custom description for the model card (optional)
    custom_description = """
    This is a custom modified YOLOv8s model trained on the COCO dataset for object detection.
    It can detect 80 different object classes with good accuracy and speed.
    The model has been optimized for real-time inference on both GPU and CPU.
    """
    
    # Upload the model with model card creation
    upload_to_huggingface(
        model=model,
        repo_id=repo_id,
        token=None,#use system token, login in termal: huggingface-cli login
        commit_message="Upload YOLOv8s model",
        private=False,  # Set to True if you want a private repository
        create_model_card=True,  # This triggers the model card creation
        example_images=example_images,  # Optional: include example images
        model_description=custom_description  # Optional: custom description
    )
    #The model card is automatically created when you call upload_to_huggingface with create_model_card=True .

def upload_onetype_model(scale='s'):
    """
    Test function to upload a YOLO model to HuggingFace Hub.
    
    Args:
        scale (str): Model scale - 'n' (nano), 's' (small), 'm' (medium), 
                    'l' (large), or 'x' (xlarge)
    """
    # Register the YOLO architecture with Transformers
    register_yolo_architecture2()
    
    # Initialize model with the specified scale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a proper config object first
    config = YoloConfig(
        scale=scale,
        nc=80,
        ch=3,
        min_size=640,
        max_size=640,
        use_fp16=True
    )
    # Initialize model with config
    model = YoloDetectionModel(
        cfg=config,
        device=device
    )
    # yaml_path = "DeepDataMiningLearning/detection/modules/yolov8.yaml"
    # model = YoloDetectionModel(cfg=yaml_path, scale=scale, nc=80, ch=3,
    #                           device=device, use_fp16=True, min_size=640, max_size=640)
    
    # Load weights for the specified scale
    weights_path = f"../modelzoo/yolov8{scale}_statedicts.pt"
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()
    
    # Upload to HuggingFace with scale in the repo name
    repo_id = f"lkk688/yolov8{scale}-model"

    # Example images for the model card (optional)
    example_images = [
        "sampledata/bus.jpg",
        "sampledata/sjsupeople.jpg"
    ]
    
    # Custom description based on scale
    scale_descriptions = {
        'n': "nano (smallest and fastest)",
        's': "small (good balance of speed and accuracy)",
        'm': "medium (higher accuracy, moderate speed)",
        'l': "large (high accuracy, slower inference)",
        'x': "xlarge (highest accuracy, slowest inference)"
    }
    
    scale_desc = scale_descriptions.get(scale, f"custom scale '{scale}'")
    
    custom_description = f"""
    This is a custom YOLOv8{scale} model ({scale_desc}) trained on the COCO dataset for object detection.
    It can detect 80 different object classes with good accuracy and speed.
    The model has been optimized for real-time inference on both GPU and CPU.
    """
    
    # Upload the model with model card creation
    try:
        upload_to_huggingface2(
            model=model,
            repo_id=repo_id,
            token=None,  # use system token, login in terminal
            commit_message=f"Upload YOLOv8{scale} model",
            private=False,  # Set to True if you want a private repository
            create_model_card=True,  # This triggers the model card creation
            example_images=example_images,  # Optional: include example images
            model_description=custom_description  # Optional: custom description
        )
        print(f"Successfully uploaded YOLOv8{scale} model to {repo_id}")
    except Exception as e:
        print(f"Error uploading YOLOv8{scale} model: {e}")
        raise

from transformers import AutoModelForObjectDetection, AutoConfig
def test_model_loading(repo_id):
    """Test loading a YOLOv8 model from Hugging Face Hub."""
    print(f"Testing model loading from {repo_id}...")
    
    # Register the model first
    register_yolo_architecture2()
    
    # Try to load the model
    try:
        model = AutoModelForObjectDetection.from_pretrained(repo_id)
        print(f"Successfully loaded model from {repo_id}")
        print(f"Model type: {type(model).__name__}")
        print(f"Model scale: {model.scale}")
        print(f"Number of classes: {model.config.num_classes}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
if __name__ == "__main__":
    #test_localmodel()
    #test_upload_model()
    # Or upload all scales in sequence
    for scale in ['n', 's', 'm', 'l', 'x']:
        try:
            print(f"\n=== Uploading YOLOv8{scale} model ===\n")
            upload_onetype_model(scale)
        except Exception as e:
            print(f"Error uploading YOLOv8{scale}: {e}")
    repo_id = "lkk688/yolov8s-model"
    test_model_loading(repo_id)
    