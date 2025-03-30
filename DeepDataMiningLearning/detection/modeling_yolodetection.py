import contextlib
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
import re
from typing import Dict, List, Optional, Tuple, Union

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
                
                # Add label and confidence
                label_text = f"{label}: {score:.2f}"
                cv2.putText(img_orig, label_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save the output image
            output_path = image_path.replace('.', '_detected.')
            cv2.imwrite(output_path, img_orig)
            print(f"Visualization saved to {output_path}")
        
        return detections

if __name__ == "__main__":
# Example usage:
    yaml_path = "DeepDataMiningLearning/detection/modules/yolov8.yaml"
    model = YoloDetectionModel(cfg=yaml_path, scale='s', nc=80)
    model.load_state_dict(torch.load("../modelzoo/yolov8s_statedicts.pt"))
    detections = model.inference_test('sampledata/bus.jpg')

# Example usage:
# model = YoloDetectionModel(cfg='path/to/yolov8n.yaml', scale='n', nc=80)
# backbone_output = model.forward_backbone(x)
# neck_output = model.forward_neck(backbone_output)
# head_outputs = model.forward_heads(neck_output)