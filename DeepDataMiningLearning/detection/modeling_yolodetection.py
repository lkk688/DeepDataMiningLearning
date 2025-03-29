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
                return preds[0]  # Return just the predictions
            return preds
    
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

# Example usage:
# model = YoloDetectionModel(cfg='path/to/yolov8n.yaml', scale='n', nc=80)
# backbone_output = model.forward_backbone(x)
# neck_output = model.forward_neck(backbone_output)
# head_outputs = model.forward_heads(neck_output)