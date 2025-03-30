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

# KITTI to COCO class mapping
kitti_to_coco = {
    'Car': 2,           # car in COCO
    'Van': 2,           # also car in COCO
    'Truck': 7,         # truck in COCO
    'Pedestrian': 0,    # person in COCO
    'Person_sitting': 0,# also person in COCO
    'Cyclist': 1,       # bicycle in COCO
    'Tram': 6,          # train in COCO
    'Misc': 0,          # default to person
    'DontCare': -1      # ignore
}

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    
    Args:
        prediction (torch.Tensor or list): Predictions tensor with shape [batch_size, num_boxes, num_classes + 5]
                                          or list of tensors
        conf_thres (float): Confidence threshold
        iou_thres (float): IoU threshold
        classes (List[int], optional): Filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        agnostic (bool): Class-agnostic NMS
        multi_label (bool): Multiple labels per box
        max_det (int): Maximum number of detections per image
        
    Returns:
        List[torch.Tensor]: List of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Handle case where prediction is a list
    if isinstance(prediction, list):
        return [non_max_suppression(p, conf_thres, iou_thres, classes, agnostic, multi_label, max_det)[0] 
                if p is not None and len(p) > 0 else torch.zeros((0, 6), device=p.device if p is not None else 'cpu') 
                for p in prediction]
    
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    
    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    # Rest of the function remains the same...
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence
        
        # If none remain process next image
        if not x.shape[0]:
            continue
            
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            
        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]
            
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
            
    return output

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) to (x1, y1, x2, y2) format
    
    Args:
        x (torch.Tensor): Bounding box coordinates (x, y, width, height)
        
    Returns:
        torch.Tensor: Bounding box coordinates (x1, y1, x2, y2)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

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

def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

# Import necessary modules from your project
from DeepDataMiningLearning.detection.modules.block import (
    AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
    Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d,
    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, RepC3, RepConv, MP, SPPCSPC
)
from DeepDataMiningLearning.detection.modules.head import Detect, IDetect, Classify, Pose, RTDETRDecoder, Segment
from DeepDataMiningLearning.detection.modules.utils import LOGGER
from DeepDataMiningLearning.detection.modules.anchor import check_anchor_order

# Define blocks that take two arguments
twoargs_blocks = [
    nn.Conv2d, Classify, Conv, ConvTranspose, GhostConv, RepConv, Bottleneck, GhostBottleneck, 
    SPP, SPPF, SPPCSPC, DWConv, Focus, BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, 
    nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3
]

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
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
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
        if verbose:
            LOGGER.info(f"activation: {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
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
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

class YoloDetectionModel(nn.Module):
    """YOLOv8 detection model with scale-based component extraction."""
    def __init__(self, cfg='yolov8n.yaml', scale='n', ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(cfg)
        self.yaml['scale'] = scale
        self.modelname = extract_filename(cfg)
        self.scale = scale

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
        Forward pass of the model on a single scale.
        Wrapper for `_predict_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        # model.train() self.training=True
        # model.eval() self.training=False
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        elif self.training:
            preds = self._predict_once(x)  # tensor input
            return preds  # training mode, direct output x (three items)
        else:  # inference mode
            preds = self._predict_once(x)  # tensor input
            # In inference mode, _predict_once returns a tuple (preds, y)
            if isinstance(preds, tuple):
                # Apply postprocessing if needed
                if kwargs.get('postprocess', False):
                    # Get original image shapes if provided
                    orig_img_shapes = kwargs.get('orig_img_shapes', None)
                    # Get new image size after preprocessing
                    new_img_size = kwargs.get('new_img_size', (640, 640))
                    
                    # Apply postprocessing similar to YoloTransform
                    processed_preds = self.postprocess(preds[0], new_img_size, orig_img_shapes)
                    return processed_preds
                return preds[0]  # Return just the predictions
            return preds
    
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
        
    def postprocess(self, preds, new_img_size=(640, 640), orig_img_shapes=None, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300):
        """
        Post-process predictions to get final detection results.
        
        Args:
            preds (torch.Tensor): Raw predictions from model
            new_img_size (tuple): Size of preprocessed image (h, w)
            orig_img_shapes (list): Original image shapes before preprocessing
            conf_thres (float): Confidence threshold for filtering detections
            iou_thres (float): IoU threshold for NMS
            classes (list): Filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
            agnostic (bool): Class-agnostic NMS
            max_det (int): Maximum number of detections per image
            
        Returns:
            list: List of processed predictions in the format expected by the trainer
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
        
        # Scale boxes back to original image size if original shapes are provided
        if orig_img_shapes is not None:
            detections = []
            for i, pred in enumerate(processed_preds):
                if i < len(orig_img_shapes):  # Ensure we have the original shape
                    orig_shape = orig_img_shapes[i]
                    # Scale boxes to original image size
                    pred[:, :4] = scale_boxes(new_img_size, pred[:, :4], orig_shape)
                
                # Format detections as expected by the trainer
                detection = {
                    "boxes": pred[:, :4].detach().cpu(),
                    "scores": pred[:, 4].detach().cpu(),
                    "labels": pred[:, 5].detach().cpu()
                }
                detections.append(detection)
            return detections
        
        # If no original shapes provided, just return the processed predictions
        return processed_preds
    
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
        import cv2
        import numpy as np
        from DeepDataMiningLearning.detection.modules.utils import scale_boxes
        
        # Load image
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        # Store original image shape
        orig_shape = img_orig.shape #(1080, 810, 3)
        
        # Preprocess image
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        # Add batch dimension
        if len(img.shape) == 3:
            img = img.unsqueeze(0) #[1, 3, 640, 640]
            
        # Move to device
        device = next(self.parameters()).device
        img = img.to(device)
        
        # Set model to evaluation mode
        self.eval()
        
        # Run inference with postprocessing
        with torch.no_grad():
            detections = self.forward(
                img, 
                postprocess=True,
                orig_img_shapes=[orig_shape],
                new_img_size=(640, 640),
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det
            )
        
        # Visualize results if requested
        if visualize and len(detections) > 0:
            # Get the first image's detections
            det = detections[0]
            boxes = det["boxes"].cpu().numpy()
            scores = det["scores"].cpu().numpy()
            labels = det["labels"].cpu().numpy()
            
            # Draw boxes on the image
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                score = scores[i]
                label = int(labels[i])
                
                # Draw bounding box
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(img_orig, (x1, y1), (x2, y2), color, 2)
                
                # Add class name and confidence instead of label ID
                # Try to get class name from model's names dictionary
                class_name = self.names.get(int(label))
                # If class name is not available, use COCO class names
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

# Create KITTI dataset class
class KITTIDataset(Dataset):
    def __init__(self, root_dir, img_size=640, split='training'):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        
        # Define paths
        self.image_dir = os.path.join(root_dir, split, 'image_2')
        self.label_dir = os.path.join(root_dir, split, 'label_2')
        
        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        self.label_files = [os.path.join(self.label_dir, os.path.basename(f).replace('.png', '.txt')) 
                           for f in self.image_files]
    
    def __len__(self):
        return len(self.image_files)
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        orig_shape = img.shape
        
        # Preprocess image with letterboxing
        img_processed, ratio, pad = self.letterbox(img, new_shape=(self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        
        # Load labels
        label_path = self.label_files[idx]
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_name = parts[0]
                    if cls_name in kitti_to_coco and kitti_to_coco[cls_name] != -1:
                        # KITTI format: [type, truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry]
                        x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                        
                        # Convert to COCO format (x, y, width, height)
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Apply letterbox transformations
                        # Scale coordinates by the resize ratio
                        x1 = x1 * ratio[0]
                        y1 = y1 * ratio[1]
                        width = width * ratio[0]
                        height = height * ratio[1]
                        
                        # Add padding offset
                        x1 = x1 + pad[0]
                        y1 = y1 + pad[1]
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, self.img_size - 1))
                        y1 = max(0, min(y1, self.img_size - 1))
                        width = max(1, min(width, self.img_size - x1))
                        height = max(1, min(height, self.img_size - y1))
                        
                        # Convert to COCO format
                        coco_cls = kitti_to_coco[cls_name]
                        annotations.append({
                            'bbox': [x1, y1, width, height],  # COCO uses [x, y, width, height]
                            'category_id': coco_cls
                        })
        
        return {
            'img': img_tensor,
            'img_path': img_path,
            'orig_shape': orig_shape,
            'annotations': annotations
        }
            
# # Validation code for COCO evaluation and mAP calculation on KITTI dataset
# def validate_kitti(model, data_path, batch_size=8, img_size=640, conf_thres=0.25, iou_thres=0.45):
#     """
#     Validate model on KITTI dataset using COCO metrics
    
#     Args:
#         model: YoloDetectionModel to evaluate
#         data_path: Path to KITTI dataset
#         batch_size: Batch size for validation
#         img_size: Image size for validation
#         conf_thres: Confidence threshold for detections
#         iou_thres: IoU threshold for NMS
        
#     Returns:
#         dict: Dictionary containing mAP metrics
#     """
    
    
    
#     # Create dataset and dataloader
#     dataset = KITTIDataset(data_path, img_size)
    
#     # Use a custom collate function to handle variable-sized annotations
#     def collate_fn(batch):
#         imgs = torch.stack([item['img'] for item in batch])
#         img_paths = [item['img_path'] for item in batch]
#         orig_shapes = [item['orig_shape'] for item in batch]
#         annotations = [item['annotations'] for item in batch]
#         return {
#             'img': imgs,
#             'img_path': img_paths,
#             'orig_shape': orig_shapes,
#             'annotations': annotations
#         }
#     dataloader = DataLoader(dataset, batch_size=batch_size, \
#         shuffle=False, num_workers=4, collate_fn=collate_fn)
    
#     # Set model to evaluation mode
#     model.eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
    
#     # Initialize COCO format results
#     coco_results = []
#     coco_gt = {
#         "images": [],
#         "annotations": [],
#         "categories": [
#             {"id": 0, "name": "person"},
#             {"id": 1, "name": "bicycle"},
#             {"id": 2, "name": "car"},
#             {"id": 3, "name": "motorcycle"},
#             {"id": 4, "name": "airplane"},
#             {"id": 5, "name": "bus"},
#             {"id": 6, "name": "train"},
#             {"id": 7, "name": "truck"}
#         ]
#     }
    
#     # Create a directory to save visualization images
#     vis_dir = os.path.join("output", "validation_vis")
#     os.makedirs(vis_dir, exist_ok=True)
#     # Save a few images with both ground truth and predictions for debugging
#     print(f"Saving visualization images to {vis_dir}")
#     vis_tofolder = True
#     # Process each batch
#     ann_id = 0
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
#             if batch_idx >= 20:  # Only visualize first 10 batches
#                 vis_tofolder = False
#             # Prepare input
#             imgs = batch['img'].to(device) #[8, 3, 256, 640]
#             img_paths = batch['img_path']
#             orig_shapes = batch['orig_shape']
            
#             # Run inference
#             detections = model(
#                 imgs, 
#                 postprocess=True,
#                 orig_img_shapes=orig_shapes,
#                 new_img_size=(img_size, img_size),
#                 conf_thres=conf_thres,
#                 iou_thres=iou_thres
#             )#list of dicts
            
#             # Process detections and ground truth
#             for i, (dets, img_path) in enumerate(zip(detections, img_paths)):
#                 if vis_tofolder:
#                     # Get original image
#                     img_orig = cv2.imread(img_path)
#                     # Create a copy for ground truth visualization
#                     img_gt = img_orig.copy()
            
#                 # Add image to ground truth
#                 img_id = batch_idx * batch_size + i
#                 coco_gt["images"].append({
#                     "id": img_id,
#                     "file_name": os.path.basename(img_path)
#                 })
                
#                 # Add ground truth annotations
#                 for ann in batch['annotations'][i]:
#                     ann_id += 1
#                     x, y, w, h = ann['bbox']
#                     # Skip invalid boxes
#                     if w <= 0 or h <= 0:
#                         continue
#                     coco_gt["annotations"].append({
#                         "id": ann_id,
#                         "image_id": img_id,
#                         "category_id": ann['category_id'],
#                         "bbox": [x, y, w, h],
#                         "area": w * h,
#                         "iscrowd": 0
#                     })
                    
#                     if vis_tofolder: #draw ground truth boxes (green)
#                         # Convert to integer coordinates
#                         x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
#                         category_id = ann['category_id']
#                         # Draw bounding box
#                         cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         # Add class name
#                         class_name = next((cat["name"] for cat in coco_gt["categories"] if cat["id"] == category_id), f"class_{category_id}")
#                         cv2.putText(img_gt, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                
#                 # Add detections to results
#                 boxes = dets["boxes"].cpu().numpy()
#                 scores = dets["scores"].cpu().numpy()
#                 labels = dets["labels"].cpu().numpy()
                
#                 for box, score, label in zip(boxes, scores, labels):
#                     x1, y1, x2, y2 = box
#                     w, h = x2 - x1, y2 - y1
#                     if vis_tofolder: #draw predicted boxes (blue)
#                         # Ensure coordinates are integers for cv2.rectangle
#                         x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
#                         cv2.rectangle(img_orig, (x1_int, y1_int), (x2_int, y2_int), (255, 0, 0), 2)
#                         # Add class name and confidence
#                         class_name = next((cat["name"] for cat in coco_gt["categories"] if cat["id"] == int(label)), f"class_{int(label)}")
#                         label_text = f"{class_name}: {score:.2f}"
#                         cv2.putText(img_orig, label_text, (x1_int, y1_int - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
#                     coco_results.append({
#                         "image_id": img_id,
#                         "category_id": int(label),
#                         "bbox": [float(x1), float(y1), float(w), float(h)],
#                         "score": float(score)
#                     })
#                 if vis_tofolder:
#                     # Save images
#                     base_name = os.path.basename(img_path)
#                     cv2.imwrite(os.path.join(vis_dir, f"{img_id}_gt_{base_name}"), img_gt)
#                     cv2.imwrite(os.path.join(vis_dir, f"{img_id}_pred_{base_name}"), img_orig)
    
#     # Save results and ground truth
#     results_file = os.path.join(data_path, "coco_results.json")
#     gt_file = os.path.join(data_path, "coco_gt.json")
    
#     with open(results_file, 'w') as f:
#         json.dump(coco_results, f)
    
#     with open(gt_file, 'w') as f:
#         json.dump(coco_gt, f)
    
#     # Print statistics for debugging
#     print(f"Number of ground truth annotations: {len(coco_gt['annotations'])}")
#     print(f"Number of detection results: {len(coco_results)}")
#     print(f"Number of images: {len(coco_gt['images'])}")
    
#     # Check if there are any annotations or results
#     if len(coco_gt['annotations']) == 0:
#         print("WARNING: No ground truth annotations found!")
#         return {'mAP': 0, 'mAP_50': 0, 'mAP_75': 0, 'mAP_small': 0, 'mAP_medium': 0, 'mAP_large': 0}
    
#     if len(coco_results) == 0:
#         print("WARNING: No detection results found!")
#         return {'mAP': 0, 'mAP_50': 0, 'mAP_75': 0, 'mAP_small': 0, 'mAP_medium': 0, 'mAP_large': 0}
    
#     # Print sample annotations and detections for debugging
#     print("\nSample ground truth annotation:")
#     print(json.dumps(coco_gt['annotations'][0] if coco_gt['annotations'] else "No annotations", indent=2))
    
#     print("\nSample detection result:")
#     print(json.dumps(coco_results[0] if coco_results else "No detections", indent=2))
    
#     # Evaluate using COCO API
#     try:
#         coco_gt_obj = COCO(gt_file)
#         coco_dt_obj = coco_gt_obj.loadRes(results_file)
#         coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'bbox')
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()
        
#         # Extract metrics
#         metrics = {
#             'mAP': coco_eval.stats[0],  # AP@IoU=0.50:0.95
#             'mAP_50': coco_eval.stats[1],  # AP@IoU=0.50
#             'mAP_75': coco_eval.stats[2],  # AP@IoU=0.75
#             'mAP_small': coco_eval.stats[3],  # AP for small objects
#             'mAP_medium': coco_eval.stats[4],  # AP for medium objects
#             'mAP_large': coco_eval.stats[5],  # AP for large objects
#         }
#     except Exception as e:
#         print(f"Error during COCO evaluation: {e}")
#         metrics = {'mAP': 0, 'mAP_50': 0, 'mAP_75': 0, 'mAP_small': 0, 'mAP_medium': 0, 'mAP_large': 0}
    
#     print(f"Validation Results:")
#     print(f"mAP@0.5:0.95: {metrics['mAP']:.4f}")
#     print(f"mAP@0.5: {metrics['mAP_50']:.4f}")
#     print(f"mAP@0.75: {metrics['mAP_75']:.4f}")
    
#     return metrics

def validate_kitti(model, data_path, img_size=640, batch_size=4, conf_thres=0.25, iou_thres=0.45, vis_tofolder=True):
    """Validate model on KITTI dataset using COCO evaluation metrics."""
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    import cv2
    import os
    import json
    from DeepDataMiningLearning.detection.myevaluator import CocoEvaluator
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = KITTIDataset(data_path, img_size=img_size, split='training')
    
    # Define a custom collate function to handle variable-sized annotations
    def collate_fn(batch):
        imgs = torch.stack([item['img'] for item in batch])
        img_paths = [item['img_path'] for item in batch]
        orig_shapes = [item['orig_shape'] for item in batch]
        annotations = [item['annotations'] for item in batch]
        
        # Create batch with image_id for COCO evaluation
        image_ids = [i for i in range(len(batch))]
        
        return {
            'img': imgs,
            'img_path': img_paths,
            'orig_shape': orig_shapes,
            'annotations': annotations,
            'image_id': image_ids
        }
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # Initialize COCO format for ground truth
    coco_ds = COCO()
    ann_id = 1
    dataset_dict = {"images": [], "categories": [], "annotations": []}
    
    # Add COCO categories
    for coco_id, name in coco_names.items():
        dataset_dict["categories"].append({"id": coco_id, "name": name})
    
    # Create a directory to save visualization images
    if vis_tofolder:
        vis_dir = os.path.join('output', "validation_vis")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Saving visualization images to {vis_dir}")
    
    # Process ground truth annotations
    print("Processing ground truth annotations...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Building COCO GT")):
        for i, (img_path, annotations) in enumerate(zip(batch['img_path'], batch['annotations'])):
            # Create image entry
            image_id = batch_idx * batch_size + i
            img_dict = {
                "id": image_id,
                "file_name": os.path.basename(img_path),
                "height": batch['img'].shape[2],
                "width": batch['img'].shape[3]
            }
            dataset_dict["images"].append(img_dict)
            
            # Process annotations
            for ann in annotations:
                x, y, w, h = ann['bbox']
                # Skip invalid boxes
                if w <= 0 or h <= 0:
                    continue
                
                category_id = ann['category_id']
                
                # Add annotation to dataset
                ann_dict = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0
                }
                dataset_dict["annotations"].append(ann_dict)
                ann_id += 1
    
    # Create COCO ground truth object
    gt_file = os.path.join(data_path, "coco_gt.json")
    with open(gt_file, 'w') as f:
        json.dump(dataset_dict, f)
    
    coco_gt = COCO(gt_file)
    
    # Initialize COCO evaluator
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco_gt, iou_types)
    
    # Run inference and collect predictions
    print("Running model inference...")
    predictions = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            imgs = batch['img'].to(device)
            img_paths = batch['img_path']
            orig_shapes = batch['orig_shape']
            
            # Run inference
            outputs = model(imgs)
            
            # Check if outputs is a tuple (common in YOLOv8 models)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Post-process detections
            processed_outputs = non_max_suppression(
                outputs, 
                conf_thres=conf_thres, 
                iou_thres=iou_thres
            )
            
            # Process each image in the batch
            for i, (det, img_path) in enumerate(zip(processed_outputs, img_paths)):
                image_id = batch_idx * batch_size + i
                
                # Visualize if requested
                if vis_tofolder:
                    # Get original image
                    img_orig = cv2.imread(img_path)
                    img_orig = cv2.resize(img_orig, (img_size, img_size))
                    img_vis = img_orig.copy()
                
                # Create prediction entry for this image
                if det is not None and len(det) > 0:
                    boxes = det[:, :4].cpu()
                    scores = det[:, 4].cpu()
                    labels = det[:, 5].cpu().int()
                    
                    # Format predictions in the way CocoEvaluator expects
                    # CocoEvaluator expects a list of dictionaries with 'boxes', 'scores', 'labels' keys
                    predictions[image_id] = {
                        'boxes': boxes,
                        'scores': scores,
                        'labels': labels
                    }
                    
                    # Visualize detection
                    if vis_tofolder:
                        for box_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                            x1, y1, x2, y2 = box.tolist()
                            
                            # Draw bounding box
                            cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            
                            # Add class name and confidence
                            class_name = coco_names.get(int(label), f"class_{label}")
                            label_text = f"{class_name}: {score:.2f}"
                            cv2.putText(img_vis, label_text, (int(x1), int(y1) - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    # Empty predictions for this image
                    predictions[image_id] = {
                        'boxes': torch.zeros((0, 4)),
                        'scores': torch.zeros(0),
                        'labels': torch.zeros(0, dtype=torch.int64)
                    }
                
                # Save visualization
                if vis_tofolder:
                    base_name = os.path.basename(img_path)
                    cv2.imwrite(os.path.join(vis_dir, f"{image_id}_pred_{base_name}"), img_vis)
                    
    # Save predictions to file (for reference, not needed for CocoEvaluator)
    results_file = os.path.join(data_path, "coco_results.json")
    all_preds = []
    for img_id, pred in predictions.items():
        boxes = pred['boxes']
        scores = pred['scores']
        labels = pred['labels']
        
        for box_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1
            
            all_preds.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score)
            })
    
    with open(results_file, 'w') as f:
        json.dump(all_preds, f)
    
    # Update evaluator with predictions
    coco_evaluator.update(predictions)
    
    # Calculate metrics
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    # Extract metrics
    coco_eval = coco_evaluator.coco_eval["bbox"]
    metrics = {
        'mAP': coco_eval.stats[0],      # AP@IoU=0.50:0.95
        'mAP_50': coco_eval.stats[1],   # AP@IoU=0.50
        'mAP_75': coco_eval.stats[2],   # AP@IoU=0.75
        'mAP_small': coco_eval.stats[3],# AP for small objects
        'mAP_medium': coco_eval.stats[4],# AP for medium objects
        'mAP_large': coco_eval.stats[5] # AP for large objects
    }
    
    print(f"Validation Results:")
    print(f"mAP@0.5:0.95: {metrics['mAP']:.4f}")
    print(f"mAP@0.5: {metrics['mAP_50']:.4f}")
    print(f"mAP@0.75: {metrics['mAP_75']:.4f}")
    
    # Return to training mode
    model.train()
    
    return metrics

def validate_kitti_old(model, data_path, img_size=640, batch_size=4, conf_thres=0.25, iou_thres=0.45, vis_tofolder=True):
    """Validate model on KITTI dataset."""
    device = next(model.parameters()).device
    
    # Create dataset and dataloader
    dataset = KITTIDataset(data_path, img_size=img_size, split='training')
    # Define a custom collate function to handle variable-sized annotations
    def collate_fn(batch):
        imgs = torch.stack([item['img'] for item in batch])
        img_paths = [item['img_path'] for item in batch]
        orig_shapes = [item['orig_shape'] for item in batch]
        annotations = [item['annotations'] for item in batch]
        return {
            'img': imgs,
            'img_path': img_paths,
            'orig_shape': orig_shapes,
            'annotations': annotations
        }
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # Initialize COCO format
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": coco_id, "name": name} for coco_id, name in coco_names.items()
        ]
    }
    coco_results = []
    
    # Create a directory to save visualization images
    if vis_tofolder:
        vis_dir = os.path.join("output", "validation_vis")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Saving visualization images to {vis_dir}")
    
    # Process each batch
    ann_id = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            # Prepare input
            imgs = batch['img'].to(device)
            img_paths = batch['img_path']
            orig_shapes = batch['orig_shape']
            
            # Run inference with lower confidence threshold for visualization
            outputs = model(imgs)
            
            # Check if outputs is a tuple (common in YOLOv8 models)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get the first element which is usually the predictions
                #[4, 84, 2940]
            # Post-process detections
            processed_outputs = non_max_suppression(
                outputs, 
                conf_thres=conf_thres, 
                iou_thres=iou_thres
            )
            
            # Process detections and ground truth
            for i, (det, img_path) in enumerate(zip(processed_outputs, img_paths)):
                if vis_tofolder:
                    # Get original image
                    img_orig = cv2.imread(img_path)
                    # Resize to match model input size for visualization
                    img_orig = cv2.resize(img_orig, (img_size, img_size))
                    # Create a copy for ground truth visualization
                    img_gt = img_orig.copy()
                    img_combined = np.hstack((img_gt, img_orig))
            
                # Add image to ground truth
                img_id = batch_idx * batch_size + i
                coco_gt["images"].append({
                    "id": img_id,
                    "file_name": os.path.basename(img_path)
                })
                
                # Add ground truth annotations
                for ann in batch['annotations'][i]:
                    ann_id += 1
                    x, y, w, h = ann['bbox']
                    # Skip invalid boxes
                    if w <= 0 or h <= 0:
                        continue
                    category_id = ann['category_id']
                    coco_gt["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": category_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    
                    if vis_tofolder: #draw ground truth boxes (green)
                        # Convert to integer coordinates
                        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                        
                        # Draw bounding box
                        cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Add class name
                        class_name = next((cat["name"] for cat in coco_gt["categories"] if cat["id"] == category_id), f"class_{category_id}")
                        cv2.putText(img_gt, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add detections to results
                if det is not None and len(det) > 0:
                    boxes = det[:, :4].cpu().numpy()
                    scores = det[:, 4].cpu().numpy()
                    labels = det[:, 5].cpu().numpy().astype(int)
                    
                    print(f"Image {img_id}: Found {len(boxes)} detections")
                    
                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box
                        w, h = x2 - x1, y2 - y1
                        if vis_tofolder: #draw predicted boxes (blue)
                            # Ensure coordinates are integers for cv2.rectangle
                            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
                            cv2.rectangle(img_orig, (x1_int, y1_int), (x2_int, y2_int), (255, 0, 0), 2)
                            # Add class name and confidence
                            class_name = next((cat["name"] for cat in coco_gt["categories"] if cat["id"] == label), f"class_{label}")
                            label_text = f"{class_name}: {score:.2f}"
                            cv2.putText(img_orig, label_text, (x1_int, y1_int - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        coco_results.append({
                            "image_id": img_id,
                            "category_id": int(label),
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "score": float(score)
                        })
                else:
                    print(f"Image {img_id}: No detections")
                
                if vis_tofolder:
                    # Save images
                    base_name = os.path.basename(img_path)
                    cv2.imwrite(os.path.join(vis_dir, f"{img_id}_combined_{base_name}"), img_combined)
    
    # Save results and ground truth
    results_file = os.path.join(data_path, "coco_results.json")
    gt_file = os.path.join(data_path, "coco_gt.json")
    
    with open(results_file, 'w') as f:
        json.dump(coco_results, f)
    
    with open(gt_file, 'w') as f:
        json.dump(coco_gt, f)
    
    # Print statistics for debugging
    print(f"Number of ground truth annotations: {len(coco_gt['annotations'])}")
    print(f"Number of detection results: {len(coco_results)}")
    print(f"Number of images: {len(coco_gt['images'])}")
    
    # Check if there are any annotations or results
    if len(coco_gt['annotations']) == 0:
        print("WARNING: No ground truth annotations found!")
        return {'mAP': 0, 'mAP_50': 0, 'mAP_75': 0, 'mAP_small': 0, 'mAP_medium': 0, 'mAP_large': 0}
    
    if len(coco_results) == 0:
        print("WARNING: No detection results found!")
        return {'mAP': 0, 'mAP_50': 0, 'mAP_75': 0, 'mAP_small': 0, 'mAP_medium': 0, 'mAP_large': 0}
    
    # Evaluate using COCO API
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        
        coco_gt_obj = COCO(gt_file)
        coco_dt_obj = coco_gt_obj.loadRes(results_file)
        coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {
            'mAP': coco_eval.stats[0],  # AP@IoU=0.50:0.95
            'mAP_50': coco_eval.stats[1],  # AP@IoU=0.50
            'mAP_75': coco_eval.stats[2],  # AP@IoU=0.75
            'mAP_small': coco_eval.stats[3],  # AP for small objects
            'mAP_medium': coco_eval.stats[4],  # AP for medium objects
            'mAP_large': coco_eval.stats[5],  # AP for large objects
        }
    except Exception as e:
        print(f"Error during COCO evaluation: {e}")
        metrics = {'mAP': 0, 'mAP_50': 0, 'mAP_75': 0, 'mAP_small': 0, 'mAP_medium': 0, 'mAP_large': 0}
    
    print(f"Validation Results:")
    print(f"mAP@0.5:0.95: {metrics['mAP']:.4f}")
    print(f"mAP@0.5: {metrics['mAP_50']:.4f}")
    print(f"mAP@0.75: {metrics['mAP_75']:.4f}")
    
    return metrics

import os
def map_boxes_to_original_size(boxes, orig_shape, new_size):
    """
    Map bounding boxes from the model's input size back to the original image size.
    
    Args:
        boxes (numpy.ndarray or torch.Tensor): Bounding boxes in format [x1, y1, x2, y2]
        orig_shape (tuple): Original image shape (height, width, channels)
        new_size (tuple): Size used for model input (height, width)
        
    Returns:
        numpy.ndarray or torch.Tensor: Boxes mapped to original image coordinates
    """
    # Get original dimensions
    orig_h, orig_w = orig_shape[:2]
    new_h, new_w = new_size
    
    # Check if input is torch tensor
    is_tensor = isinstance(boxes, torch.Tensor)
    
    # Convert to numpy if tensor
    if is_tensor:
        device = boxes.device
        boxes_np = boxes.cpu().numpy()
    else:
        boxes_np = boxes.copy()
    
    # Calculate scaling factors
    scale_w = orig_w / new_w
    scale_h = orig_h / new_h
    
    # Apply scaling to coordinates
    boxes_np[:, 0] = boxes_np[:, 0] * scale_w  # x1
    boxes_np[:, 1] = boxes_np[:, 1] * scale_h  # y1
    boxes_np[:, 2] = boxes_np[:, 2] * scale_w  # x2
    boxes_np[:, 3] = boxes_np[:, 3] * scale_h  # y2
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        return torch.from_numpy(boxes_np).to(device)
    else:
        return boxes_np

def visualize_results(img, boxes, labels, scores, class_names, output_path):
    """
    Draw bounding boxes and labels on an image and save it.
    
    Args:
        img: Image to draw on (should be in BGR format for OpenCV)
        boxes: Bounding boxes in [x1, y1, x2, y2] format
        labels: Class labels for each box
        scores: Confidence scores for each box
        class_names: Dictionary mapping class IDs to names
        output_path: Path to save the output image
    """
    # Make a copy to avoid modifying the original
    img_vis = img.copy()
    
    for k, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw bounding box
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        
        # Add class name and confidence
        class_name = class_names.get(int(label), f"class_{label}")
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(img_vis, label_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save the output image
    cv2.imwrite(output_path, img_vis)
    return img_vis
    
def batch_inference(model_path=None, input_folder=None, output_folder=None, conf_thres=0.25, iou_thres=0.45, img_size=640, batch_size=8, use_fp16=True, drawto_originalsize=True):
    """
    Run inference on all images in a folder and save the detected images with bounding boxes to an output folder.
    
    Args:
        model_path (str): Path to the model weights file. If None, uses the default YOLOv8s model.
        input_folder (str): Path to the folder containing images. If None, prompts user to select.
        output_folder (str): Path to save the output images. If None, creates a subfolder in the input folder.
        conf_thres (float): Confidence threshold for detections.
        iou_thres (float): IoU threshold for NMS.
        img_size (int): Size to resize images to before inference.
        batch_size (int): Number of images to process at once.
        use_fp16 (bool): Whether to use FP16 precision for faster inference.
        drawto_originalsize (bool): Whether to draw bounding boxes on original size image.
        
    Returns:
        dict: Statistics about the processed images.
    """
    import os
    import glob
    import cv2
    import torch
    import numpy as np
    from tqdm import tqdm
    from pathlib import Path
    import time
    
    # Handle input folder selection if not provided
    if input_folder is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        input_folder = filedialog.askdirectory(title="Select folder with images")
        if not input_folder:
            print("No folder selected. Exiting.")
            return
    
    # Create output folder if not provided
    if output_folder is None:
        output_folder = os.path.join(input_folder, "detections")
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        # Use default YOLOv8s model
        yaml_path = "DeepDataMiningLearning/detection/modules/yolov8.yaml"
        model = YoloDetectionModel(cfg=yaml_path, scale='s', nc=80)
        model.load_state_dict(torch.load("../modelzoo/yolov8s_statedicts.pt"))
    else:
        # Load model from provided path
        if model_path.endswith('.yaml'):
            # Load from YAML config
            model = YoloDetectionModel(cfg=model_path, scale='s', nc=80)
            # Assume weights are in the same directory with .pt extension
            weights_path = model_path.replace('.yaml', '.pt')
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path))
        else:
            # Load directly from weights file
            yaml_path = "DeepDataMiningLearning/detection/modules/yolov8.yaml"
            model = YoloDetectionModel(cfg=yaml_path, scale='s', nc=80)
            model.load_state_dict(torch.load(model_path))
    
    model = model.to(device)
    model.eval()
    
    # Enable FP16 precision if requested
    if use_fp16 and device.type == 'cuda':
        model = model.half()
        print("Using FP16 precision for faster inference")
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images. Processing...")
    
    # Process images in batches
    stats = {
        'total_images': len(image_files),
        'total_objects': 0,
        'processing_time': 0
    }
    
    # Process in batches
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i:i+batch_size]
        batch_imgs = []
        batch_orig_imgs = []
        batch_shapes = []
        
        # Prepare batch
        for img_path in batch_files:
            # Load and preprocess image
            img_orig = cv2.imread(img_path)
            if img_orig is None:
                print(f"Warning: Could not read {img_path}, skipping")
                continue
                
            # Store original image shape
            orig_shape = img_orig.shape
            batch_shapes.append(orig_shape)
            batch_orig_imgs.append(img_orig)
            
            # Preprocess image - use the same preprocessing as in inference_test
            img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            # Convert to FP16 if using half precision
            if use_fp16 and device.type == 'cuda':
                img = img.half()
                
            batch_imgs.append(img)
        
        if not batch_imgs:
            continue
            
        # Stack images into a batch tensor
        batch_tensor = torch.stack(batch_imgs).to(device)
        
        # Run inference with timing
        start_time = time.time()
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=use_fp16):
                detections = model.forward(
                    batch_tensor, 
                    postprocess=True,
                    orig_img_shapes=batch_shapes,
                    new_img_size=(img_size, img_size),
                    conf_thres=conf_thres,
                    iou_thres=iou_thres
                )
        inference_time = time.time() - start_time
        stats['processing_time'] += inference_time
        
        # Process detections and save images
        for j, (img_path, det) in enumerate(zip(batch_files, detections)):
            if j >= len(batch_orig_imgs):
                continue
                
            img_orig = batch_orig_imgs[j].copy()
            
            # Get filename without extension
            filename = Path(img_path).stem
            output_path = os.path.join(output_folder, f"{filename}_detected.jpg")
            
            # Get detections
            boxes = det["boxes"].cpu().numpy()
            scores = det["scores"].cpu().numpy()
            labels = det["labels"].cpu().numpy()
            
            stats['total_objects'] += len(boxes)
            
            # Draw boxes on the image
            for k, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(img_orig, (x1, y1), (x2, y2), color, 2)
                
                # Add class name and confidence
                class_name = model.names.get(int(label))
                if class_name is None or class_name == f"{int(label)}":
                    class_name = coco_names.get(int(label), f"class_{label}")
                label_text = f"{class_name}: {score:.2f}"
                cv2.putText(img_orig, label_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save the output image
            cv2.imwrite(output_path, img_orig)
    
    # Calculate and print statistics
    avg_time_per_image = stats['processing_time'] / stats['total_images'] if stats['total_images'] > 0 else 0
    avg_objects_per_image = stats['total_objects'] / stats['total_images'] if stats['total_images'] > 0 else 0
    
    print(f"\nProcessing complete!")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Total objects detected: {stats['total_objects']}")
    print(f"Average objects per image: {avg_objects_per_image:.2f}")
    print(f"Total processing time: {stats['processing_time']:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image*1000:.2f} ms")
    print(f"Output saved to: {output_folder}")
    
    return stats

def single_image_inference(model_path=None, image_path=None, output_folder="output/single_debug", conf_thres=0.25, iou_thres=0.45, img_size=640, use_fp16=True):
    """
    Run inference on a single image and save detailed debug information.
    
    Args:
        model_path (str): Path to the model weights file. If None, uses the default YOLOv8s model.
        image_path (str): Path to the input image. If None, prompts user to select.
        output_folder (str): Path to save the output images and debug info.
        conf_thres (float): Confidence threshold for detections.
        iou_thres (float): IoU threshold for NMS.
        img_size (int): Size to resize images to before inference.
        use_fp16 (bool): Whether to use FP16 precision for faster inference.
        
    Returns:
        dict: Detection results and debug information
    """
    import os
    import cv2
    import torch
    import numpy as np
    from pathlib import Path
    import time
    import json
    
    # Handle image selection if not provided
    if image_path is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(title="Select image file", 
                                               filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not image_path:
            print("No image selected. Exiting.")
            return
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        # Use default YOLOv8s model
        yaml_path = "DeepDataMiningLearning/detection/modules/yolov8.yaml"
        model = YoloDetectionModel(cfg=yaml_path, scale='s', nc=80)
        model.load_state_dict(torch.load("../modelzoo/yolov8s_statedicts.pt"))
    else:
        # Load model from provided path
        if model_path.endswith('.yaml'):
            # Load from YAML config
            model = YoloDetectionModel(cfg=model_path, scale='s', nc=80)
            # Assume weights are in the same directory with .pt extension
            weights_path = model_path.replace('.yaml', '.pt')
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path))
        else:
            # Load directly from weights file
            yaml_path = "DeepDataMiningLearning/detection/modules/yolov8.yaml"
            model = YoloDetectionModel(cfg=yaml_path, scale='s', nc=80)
            model.load_state_dict(torch.load(model_path))
    
    model = model.to(device)
    model.eval()
    
    # Enable FP16 precision if requested
    if use_fp16 and device.type == 'cuda':
        model = model.half()
        print("Using FP16 precision for inference")
    
    # Load and preprocess image
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Save original image for reference
    cv2.imwrite(os.path.join(output_folder, "original.jpg"), img_orig)
    
    # Store original image shape
    orig_shape = img_orig.shape
    print(f"Original image shape: {orig_shape}")
    
    # Preprocess image
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    # Save the RGB converted image
    cv2.imwrite(os.path.join(output_folder, "rgb_converted.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # Resize image
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # Save the resized image
    cv2.imwrite(os.path.join(output_folder, "resized.jpg"), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
    
    # Convert to CHW format
    img_chw = img_resized.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_chw)).float()
    img_tensor /= 255.0  # Normalize to [0, 1]
    
    # Convert to FP16 if using half precision
    if use_fp16 and device.type == 'cuda':
        img_tensor = img_tensor.half()
    
    # Add batch dimension and move to device
    img_batch = img_tensor.unsqueeze(0).to(device)
    
    # Run inference with timing
    start_time = time.time()
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=use_fp16):
            # First, get the raw predictions without postprocessing
            raw_preds = model(img_batch)
            # Handle case where raw_preds is a tuple (common in YOLOv8 models)
            if isinstance(raw_preds, tuple):
                # Use the first element which typically contains the detection predictions
                raw_preds = raw_preds[0]
            
            # Then manually handle the postprocessing for debugging
            detections = non_max_suppression(
                raw_preds, 
                conf_thres=conf_thres, 
                iou_thres=iou_thres
            )
    inference_time = time.time() - start_time
    
    # Get detections for the single image
    det = detections[0]  # First image in batch
    
    # Create a copy of original image for visualization
    img_vis = img_orig.copy()
    
    # Calculate scaling factors
    scale_h, scale_w = orig_shape[0] / img_size, orig_shape[1] / img_size
    
    # Process and draw each detection
    if len(det):
        # Convert from xywh to xyxy format if needed (depends on your model output)
        # det[:, :4] = xywh2xyxy(det[:, :4])
        
        # Scale coordinates to original image size
        det[:, 0] *= scale_w  # x1
        det[:, 1] *= scale_h  # y1
        det[:, 2] *= scale_w  # x2
        det[:, 3] *= scale_h  # y2
        
        # Save detection results as JSON for debugging
        detection_results = {
            "boxes": det[:, :4].cpu().numpy().tolist(),
            "scores": det[:, 4].cpu().numpy().tolist(),
            "labels": det[:, 5].cpu().numpy().tolist(),
            "inference_time_ms": inference_time * 1000,
            "original_shape": orig_shape,
            "input_size": img_size,
            "scale_factors": [scale_h, scale_w]
        }
        
        with open(os.path.join(output_folder, "detection_results.json"), 'w') as f:
            json.dump(detection_results, f, indent=2)
        
        # Draw bounding boxes
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            
            # Add class name and confidence
            class_name = model.names.get(int(cls))
            if class_name is None or class_name == f"{int(cls)}":
                class_name = coco_names.get(int(cls), f"class_{cls}")
            label_text = f"{class_name}: {conf:.2f}"
            cv2.putText(img_vis, label_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save visualization
    cv2.imwrite(os.path.join(output_folder, "detection_result.jpg"), img_vis)
    
    print(f"\nProcessing complete!")
    print(f"Number of objects detected: {len(det) if len(det) else 0}")
    print(f"Inference time: {inference_time*1000:.2f} ms")
    print(f"Output saved to: {output_folder}")
    
    return detection_results

def one_test():
    yaml_path = "DeepDataMiningLearning/detection/modules/yolov8.yaml"
    model = YoloDetectionModel(cfg=yaml_path, scale='s', nc=80)
    model.load_state_dict(torch.load("../modelzoo/yolov8s_statedicts.pt"))
    detections = model.inference_test('sampledata/bus.jpg')
    
    # Run validation if KITTI dataset path is provided
    kitti_path = "/DATA5T2/Dataset/Kitti"
    if os.path.exists(kitti_path):
        print(f"Running validation on KITTI dataset at {kitti_path}")
        metrics = validate_kitti(model, kitti_path)
        print(f"Validation complete. mAP@0.5: {metrics['mAP_50']:.4f}")
    else:
        print(f"KITTI dataset not found at {kitti_path}. Skipping validation.")
    
if __name__ == "__main__":
# Example usage:
    #one_test()
    single_image_inference(image_path="/DATA5T2/Dataset/Kitti/testing/image_2/000001.png", 
                          output_folder="output/debug_single")
    batch_inference(input_folder="/DATA5T2/Dataset/Kitti/testing/image_2/", output_folder="output/kitti_testing/")

# Example usage:
# model = YoloDetectionModel(cfg='path/to/yolov8n.yaml', scale='n', nc=80)
# backbone_output = model.forward_backbone(x)
# neck_output = model.forward_neck(backbone_output)
# head_outputs = model.forward_heads(neck_output)