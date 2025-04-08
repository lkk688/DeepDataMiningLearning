import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms
from typing import List, Tuple, Dict, Optional
import math
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision.ops import nms

class Conv(nn.Module):
    """
    Basic Conv block with optional batch norm and activation.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size
        stride: Stride
        padding: Padding
        groups: Groups for depthwise conv
        act: Activation function (True=SiLU, False=Linear, nn.Module=custom)
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, 
                 groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=kernel_size // 2 if padding is None else padding,
            groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """
    Standard bottleneck block with residual connection.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        shortcut: Whether to use shortcut connection
        groups: Groups for depthwise conv
        expansion: Expansion ratio for hidden channels
    """
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions and multiple Bottleneck blocks.
    YOLOv8's improved CSP Bottleneck design.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        n: Number of Bottleneck blocks
        shortcut: Whether to use shortcut connections
        groups: Groups for depthwise conv
        expansion: Expansion ratio for hidden channels
    """
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, groups=1, expansion=0.5):
        super().__init__()
        self.c = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, out_channels, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, groups, expansion=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer.
    Equivalent to SPP but faster by concatenating after maxpool operations.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        k: Kernel size for maxpool
    """
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Detect(nn.Module):
    """
    YOLOv8 Detect head for detection tasks.
    
    Args:
        in_channels: List of input channels from different feature levels
        num_classes: Number of detection classes
        strides: Strides for each feature level
    """
    def __init__(self, in_channels, num_classes, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.nl = len(in_channels)  # Number of detection layers
        self.reg_max = 16  # DFL channels (reg_max + 1) * 4
        self.in_channels = in_channels  # Store input channels for reference

        # Build detection head
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, x, 3), 
                Conv(x, x, 3), 
                nn.Conv2d(x, self.reg_max * 4 + num_classes, 1)
            ) 
            for x in in_channels
        )
        
        # Initialize weights
        self.init_weights()

    # def init_weights(self):
    #     """Initialize weights for detection head."""
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    def init_weights(self):
        """Initialize weights for detection head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
                # Initialize the final classification layer with negative bias
                # This will make the initial confidence scores low (around 0.1)
                if m.out_channels == self.reg_max * 4 + self.num_classes:
                    # Only apply to the classification part of the output
                    # The last self.num_classes filters are for classification
                    cls_bias = torch.ones(m.out_channels) * 0.01
                    # Set a negative bias for classification outputs
                    cls_bias[self.reg_max * 4:] = -2.0  # This gives sigmoid(x) ≈ 0.12
                    if m.bias is not None:
                        m.bias.data = cls_bias
    
    def forward(self, x):
        """Forward pass through detection head."""
        outputs = []
        for i in range(self.nl):
            # Add a check to ensure input channels match expected channels
            if x[i].shape[1] != self.in_channels[i]:
                raise ValueError(f"Input feature map {i} has {x[i].shape[1]} channels, "
                                f"but expected {self.in_channels[i]} channels.")
            # Pass through detection head
            output = self.cv2[i](x[i])
            
            # Debug output shape
            #print(f"Detection head {i} output shape: {output.shape}")
            
            outputs.append(output)
        return outputs

class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) module.
    Proposed in Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes
    for Dense Object Detection.
    """
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        self.weight = nn.Parameter(torch.arange(c1, dtype=torch.float).reshape(1, c1, 1, 1))
        self.conv.weight = self.weight  # Fixed weights for distribution

    def forward(self, x):
        """Applies softmax to channel dimension and computes expected value."""
        # Input shape: [bs, anchors, 4, reg_max]
        # Need to process each box coordinate separately
        
        # Store original shape for reshaping later
        bs, anchors, coords, reg_max = x.shape
        
        # Reshape to [bs*anchors*coords, reg_max, 1, 1] for conv operation
        x = x.reshape(-1, reg_max, 1, 1)
        
        # Apply softmax along reg_max dimension
        x = x.softmax(1)
        
        # Apply convolution (which is effectively a weighted sum with the weights being 0,1,2,...,reg_max-1)
        x = self.conv(x)  # Shape: [bs*anchors*coords, 1, 1, 1]
        
        # Reshape back to original format but without the reg_max dimension
        x = x.reshape(bs, anchors, coords)
        
        return x
        
class YOLOv8(nn.Module):
    """
    YOLOv8 object detection model with proper feature pyramid network.
    
    Args:
        num_classes: Number of detection classes
        in_channels: Number of input channels (default=3 for RGB)
        depth_multiple: Model depth multiplier
        width_multiple: Model width multiplier
    """
    def __init__(self, num_classes=80, in_channels=3, depth_multiple=1.0, width_multiple=1.0):
        super().__init__()
        self.num_classes = num_classes
        
        # Calculate channel dimensions with width multiplier
        c1 = int(64 * width_multiple)    # stem
        c2 = int(128 * width_multiple)   # stage 1
        c3 = int(256 * width_multiple)   # stage 2
        c4 = int(512 * width_multiple)   # stage 3
        c5 = int(512 * width_multiple * width_multiple)  # stage 4
        
        # Backbone
        self.backbone = nn.Sequential(
            # stem
            Conv(in_channels, c1), 
            
            # stage 1 (1/2)
            Conv(c1, c2, 3, 2),
            C2f(c2, c2, n=round(3 * depth_multiple)),
            
            # stage 2 (1/4)
            Conv(c2, c3, 3, 2),
            C2f(c3, c3, n=round(6 * depth_multiple)),
            
            # stage 3 (1/8)
            Conv(c3, c4, 3, 2),
            C2f(c4, c4, n=round(6 * depth_multiple)),
            
            # stage 4 (1/16)
            Conv(c4, c5, 3, 2),
            C2f(c5, c5, n=round(3 * depth_multiple)),
        )
        
        # Neck (Feature Pyramid Network)
        # SPPF on the backbone output
        self.sppf = SPPF(c5, c5)
        
        # Top-down path (upsampling)
        self.lateral_conv1 = Conv(c5, c3, 1, 1)  # Reduce channels for P5->P4
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.C3_p4 = C2f(c3 + c4, c3, n=round(3 * depth_multiple), shortcut=False)
        
        self.lateral_conv2 = Conv(c3, c3, 1, 1)  # Maintain channels for P4->P3
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.C3_p3 = C2f(c3 + c3, c3, n=round(3 * depth_multiple), shortcut=False)
        
        # Bottom-up path (downsampling)
        self.down_conv1 = Conv(c3, c3, 3, 2)  # P3->P4
        self.C3_n3 = C2f(c3 + c3, c3, n=round(3 * depth_multiple), shortcut=False)
        
        self.down_conv2 = Conv(c3, c3, 3, 2)  # P4->P5
        self.C3_n4 = C2f(c3 + c3, c3, n=round(3 * depth_multiple), shortcut=False)
        
        # Detection head
        self.detect = Detect(
            in_channels=[c3, c3, c3],  # P3, P4, P5 all have same channel count
            num_classes=num_classes,
            strides=[8, 16, 32]  # Strides for P3, P4, P5
        )
        
        # DFL for box regression
        self.dfl = DFL(self.detect.reg_max)
        
        # Initialize weights
        self.init_weights()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate model configuration to catch errors early."""
        # Check that detection head input channels match neck output channels
        if not all(ch == self.detect.in_channels[0] for ch in self.detect.in_channels):
            raise ValueError("All detection head input channels should be equal in this implementation")
        
        # Check that DFL reg_max matches detection head reg_max
        if self.dfl.conv.weight.shape[1] != self.detect.reg_max:
            raise ValueError(f"DFL reg_max ({self.dfl.conv.weight.shape[1]}) doesn't match "
                            f"detection head reg_max ({self.detect.reg_max})")

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through YOLOv8 model."""
        # Backbone
        x1 = self.backbone[:4](x)      # P2 (1/4 scale)
        x2 = self.backbone[4:7](x1)    # P3 (1/8 scale)
        x3 = self.backbone[7:](x2)     # P4 (1/16 scale)
        
        # Neck
        # SPPF
        sppf = self.sppf(x3)  # Enhanced P5 features
        
        # Top-down path (upsampling)
        p5 = self.lateral_conv1(sppf)
        p5_up = self.upsample1(p5)
        p4 = torch.cat([p5_up, x2], 1)  # Concatenate with backbone P3
        p4 = self.C3_p4(p4)
        
        p4_up = self.upsample2(self.lateral_conv2(p4))
        p3 = torch.cat([p4_up, x1], 1)  # Concatenate with backbone P2
        p3 = self.C3_p3(p3)  # P3 (1/8 scale, 80x80)
        
        # Bottom-up path (downsampling)
        p3_down = self.down_conv1(p3)
        p4 = torch.cat([p3_down, p4], 1)
        p4 = self.C3_n3(p4)  # P4 (1/16 scale, 40x40)
        
        p4_down = self.down_conv2(p4)
        p5 = torch.cat([p4_down, p5], 1)
        p5 = self.C3_n4(p5)  # P5 (1/32 scale, 20x20)
        
        # Head - pass features from all scales to detection head
        # p3: 1/8 scale (80x80) for large objects
        # p4: 1/16 scale (40x40) for medium objects
        # p5: 1/32 scale (20x20) for small objects
        outputs = self.detect([p3, p4, p5])
        
        if self.training:
            return outputs
        else:
            # Post-process for inference
            return self.post_process(outputs)

    def post_process(self, outputs, conf_thres=0.25, iou_thres=0.45):
        """
        Post-process model outputs for inference.
        
        Args:
            outputs: Raw model outputs
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            
        Returns:
            List of detections (xyxy, conf, cls) for each image
        """
        detections = []
        for output in outputs:
            # Reshape output: (batch, channels, height, width) -> (batch, anchors, classes+box)
            bs, ch, ny, nx = output.shape
            # Calculate the correct number of anchors based on the output shape
            output = output.view(bs, self.detect.reg_max * 4 + self.num_classes, ny * nx).permute(0, 2, 1)
            
            # Decode boxes (DFL)
            box, cls = output.split((self.detect.reg_max * 4, self.num_classes), 2)
            box = self.decode_box(box)
            
            # Apply confidence threshold
            conf, j = cls.sigmoid().max(2, keepdim=True)
            mask = (conf.squeeze(2) > conf_thres)
            box, conf, j = box[mask], conf[mask], j[mask]
            
            # Batched NMS
            if box.shape[0] > 0:
                keep = batched_nms(box, conf, j, iou_thres)
                box, conf, j = box[keep], conf[keep], j[keep]
            
            detections.append(torch.cat([box, conf, j.float()], 1))
        return detections

    def decode_box(self, box):
        """
        Decode predicted boxes from DFL distribution.
        
        Args:
            box: Predicted box distribution (bs, anchors, reg_max * 4)
            
        Returns:
            Decoded boxes in xyxy format
        """
        bs, anchors = box.shape[:2]
        box = box.view(bs, anchors, 4, self.detect.reg_max)
        box = self.dfl(box).view(bs, anchors, 4)
        
        # Convert from xywh to xyxy
        box = torch.cat([
            box[..., :2] - box[..., 2:] / 2,  # x1y1
            box[..., :2] + box[..., 2:] / 2   # x2y2
        ], -1)
        return box

class YOLOv8Loss(nn.Module):
    """
    YOLOv8 loss function combining:
    - Classification loss (BCE): Binary cross-entropy for multi-label classification
    - Box regression loss (DFL + CIoU): Distribution Focal Loss for box coordinate regression
                                       and Complete IoU for better bounding box accuracy
    
    The loss function handles predictions from multiple detection layers (different scales).
    """
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        # BCE loss with no reduction to allow for weighting
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        # Distribution Focal Loss for box coordinate regression
        self.dfl = DFL()
        
    def forward(self, preds, targets):
        """
        Compute YOLOv8 loss.
        
        Args:
            preds: Model predictions (list of outputs from Detect head)
                  Each element has shape [batch_size, (4+num_classes)*reg_max, height, width]
            targets: Ground truth targets dictionary containing:
                    - 'boxes': Target boxes in xyxy format [batch_size, num_targets, 4]
                    - 'class': Target class indices [batch_size, num_targets]
            
        Returns:
            Dictionary of loss components:
            - 'loss': Total loss (box_loss + cls_loss)
            - 'box_loss': Localization loss component
            - 'cls_loss': Classification loss component
        """
        device = preds[0].device
        lcls = torch.zeros(1, device=device)  # Class loss initialization
        lbox = torch.zeros(1, device=device)  # Box loss initialization
        
        # Process each prediction level (P3, P4, P5)
        for i, pred in enumerate(preds):
            # Get dimensions
            bs, ch, ny, nx = pred.shape  # [batch_size, (4+num_classes)*reg_max, height, width]
            
            # Reshape predictions to [batch_size, height*width, 4+num_classes]
            # This transforms grid cell predictions into a sequence of predictions
            pred = pred.view(bs, self.dfl.conv.weight.shape[1] * 4 + self.num_classes, -1).permute(0, 2, 1)
            # pred shape after reshape: [batch_size, height*width, 4*reg_max+num_classes]
            
            # Split predictions into box and class components
            # box_pred shape: [batch_size, height*width, 4*reg_max]
            # cls_pred shape: [batch_size, height*width, num_classes]
            box_pred, cls_pred = pred.split((self.dfl.conv.weight.shape[1] * 4, self.num_classes), 2)
            
            # Reshape box predictions for DFL processing
            # [batch_size, height*width, 4*reg_max] -> [batch_size, height*width, 4, reg_max]
            box_pred = box_pred.view(bs, -1, 4, self.dfl.conv.weight.shape[1])
            
            # Apply DFL to get final box coordinates
            # Output shape: [batch_size, height*width, 4]
            box_pred = self.dfl(box_pred).view(bs, -1, 4)
            
            # In a real implementation, we would match predictions to targets here
            # This would involve:
            # 1. Converting grid cell predictions to actual coordinates
            # 2. Assigning targets to predictions based on IoU and other criteria
            # 3. Creating masks for positive (matched) and negative (unmatched) samples
            
            # For this simplified example, we assume targets are already matched
            # and have the same structure as predictions
            
            # Classification loss calculation
            # Create target tensor with same shape as cls_pred
            # tcls shape: [batch_size, height*width, num_classes]
            tcls = torch.zeros_like(cls_pred)
            
            # In practice, we would use the matching results to assign target classes
            # Here we're using a simplified approach where targets['class'] directly indexes the classes
            # This assumes targets['class'] contains indices of positive classes for each prediction
            if isinstance(targets['class'], int):
                # Handle single class case
                tcls[..., targets['class']] = 1.0
            else:
                # Handle multi-class case
                # This is simplified - in practice, we'd have a more complex assignment
                for b in range(bs):
                    if b < len(targets['class']):
                        tcls[b, :, targets['class'][b]] = 1.0
            
            # Calculate BCE loss for classification
            # Mean over all dimensions to get scalar loss
            cls_loss = self.bce(cls_pred, tcls).mean()
            lcls += cls_loss
            
            # Box regression loss calculation
            # Convert predicted boxes from xywh to xyxy format for IoU calculation
            # pbox shape: [batch_size, height*width, 4]
            pbox = self.xywh2xyxy(box_pred)
            
            # Get target boxes (already in xyxy format)
            # tbox shape should match pbox: [batch_size, height*width, 4]
            # In practice, we would use the matching results to assign target boxes
            tbox = targets['boxes']
            
            # Calculate 1 - IoU as the box loss (higher IoU = lower loss)
            # In a real implementation, we would use CIoU or DIoU for better convergence
            box_loss = (1.0 - self.box_iou(pbox, tbox)).mean()
            lbox += box_loss
        
        # Return combined loss and individual components
        return {
            'loss': lbox + lcls,  # Total loss
            'box_loss': lbox,     # Box regression loss component
            'cls_loss': lcls      # Classification loss component
        }
    
    @staticmethod
    def xywh2xyxy(x):
        """
        Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2] format.
        
        Args:
            x: Tensor of shape [..., 4] containing boxes in xywh format
            
        Returns:
            Tensor of same shape with boxes in xyxy format
        """
        y = x.clone()
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1 = x_center - width/2
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1 = y_center - height/2
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2 = x_center + width/2
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2 = y_center + height/2
        return y
    
    @staticmethod
    def box_iou(box1, box2):
        """
        Compute intersection over union (IoU) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        
        Args:
            box1: First set of boxes, tensor of shape [..., 4]
            box2: Second set of boxes, tensor of shape [..., 4]
            
        Returns:
            IoU values, tensor of shape [...]
        """
        # Get coordinates of intersection rectangles
        # For each coordinate, we take the maximum of the top-left corners
        # and the minimum of the bottom-right corners
        inter_left = torch.max(box1[..., 0], box2[..., 0])    # max of x1 values
        inter_top = torch.max(box1[..., 1], box2[..., 1])     # max of y1 values
        inter_right = torch.min(box1[..., 2], box2[..., 2])   # min of x2 values
        inter_bottom = torch.min(box1[..., 3], box2[..., 3])  # min of y2 values
        
        # Calculate intersection area, ensuring non-negative dimensions
        # If boxes don't overlap, this will be zero
        inter_width = torch.clamp(inter_right - inter_left, min=0)
        inter_height = torch.clamp(inter_bottom - inter_top, min=0)
        inter_area = inter_width * inter_height
        
        # Calculate areas of both boxes
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])  # width1 * height1
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])  # width2 * height2
        
        # Union = sum of areas - intersection
        union_area = box1_area + box2_area - inter_area
        
        # IoU = intersection / union (add small epsilon to avoid division by zero)
        iou = inter_area / (union_area + 1e-7)
        
        return iou

def load_and_preprocess_image(image_path, img_size=640):
    """
    Load and preprocess an image for YOLOv8.
    
    Args:
        image_path: Path to image file
        img_size: Target size for the longer edge
        
    Returns:
        Preprocessed image tensor [1, 3, H, W], original image dimensions, scale factor
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_width, original_height = img.size
    
    # Calculate scaling factor
    scale = min(img_size / original_width, img_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image
    img = img.resize((new_width, new_height), Image.BILINEAR)
    
    # Create a canvas with padding if needed
    padded_img = Image.new('RGB', (img_size, img_size), (114, 114, 114))
    padded_img.paste(img, (0, 0))
    
    # Convert to tensor and normalize
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(padded_img).unsqueeze(0)  # Add batch dimension
    
    return img_tensor, (original_height, original_width), scale

def visualize_detections(image_path, detections, class_names, scale, conf_thresh=0.25):
    """
    Visualize detections on the original image.
    
    Args:
        image_path: Path to original image
        detections: Tensor of detections [N, 6] (x1, y1, x2, y2, conf, cls)
        class_names: List of class names
        scale: Scale factor used during preprocessing
        conf_thresh: Confidence threshold for display
    """
    # Load original image
    img = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # Rescale boxes to original image size
    detections[:, :4] /= scale
    
    # Filter by confidence
    keep = detections[:, 4] > conf_thresh
    detections = detections[keep]
    
    # Draw each detection
    for *xyxy, conf, cls in detections:
        # Create rectangle patch
        x1, y1, x2, y2 = xyxy
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label = f"{class_names[int(cls)]} {conf:.2f}"
        ax.text(x1, y1 - 5, label, color='red', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def test_yolov8_on_image(image_path, model, class_names):
    """
    Test YOLOv8 on a single image.
    
    Args:
        image_path: Path to input image
        model: YOLOv8 model
        class_names: List of COCO class names
    """
    # Load and preprocess image
    img_tensor, original_dims, scale = load_and_preprocess_image(image_path)
    print(f"Input image tensor shape: {img_tensor.shape}")  # [1, 3, 640, 640]
    
    # Run model
    model.eval()
    with torch.no_grad():
        # Forward pass through backbone
        x = img_tensor
        print("\nBackbone stages:")
        x1 = model.backbone[:4](x)
        print(f"Stage 1-2 output: {x1.shape}")  # [1, 256, 160, 160]
        x2 = model.backbone[4:7](x1)
        print(f"Stage 3 output: {x2.shape}")    # [1, 512, 80, 80]
        x3 = model.backbone[7:](x2)
        print(f"Stage 4 output: {x3.shape}")    # [1, 512, 40, 40]
        
        # Neck (FPN)
        print("\nNeck stages:")
        p3 = model.neck[0](x3)  # SPPF
        print(f"SPPF output: {p3.shape}")       # [1, 512, 40, 40]
        p3 = model.neck[1](p3)   # Conv
        print(f"Conv output: {p3.shape}")       # [1, 256, 40, 40]
        p3 = model.neck[2](p3)   # Upsample
        print(f"Upsample output: {p3.shape}")   # [1, 256, 80, 80]
        
        p2 = torch.cat([p3, x2], 1)
        print(f"Concat output: {p2.shape}")     # [1, 768, 80, 80]
        p2 = model.neck[3][0](p2)  # C2f
        print(f"C2f output: {p2.shape}")        # [1, 256, 80, 80]
        p2_conv = model.neck[3][1](p2)  # Conv
        print(f"Conv output: {p2_conv.shape}")  # [1, 256, 80, 80]
        p2_up = model.neck[3][2](p2_conv)  # Upsample
        print(f"Upsample output: {p2_up.shape}")# [1, 256, 160, 160]
        
        p1 = torch.cat([p2_up, x1], 1)
        print(f"Concat output: {p1.shape}")     # [1, 512, 160, 160]
        p1 = model.neck[3][3](p1)  # C2f
        print(f"Final neck output: {p1.shape}") # [1, 256, 160, 160]
        
        # Head - Use the correct feature maps that match the expected channel dimensions
        print("\nHead outputs:")
        # Use p1, p2 (not p2_conv or p2_up), and p3 as inputs to the detection head
        outputs = model.detect([p1, p2, p3])
        for i, o in enumerate(outputs):
            print(f"Detection head {i} output: {o.shape}")
        
        # Post-process
        detections = model.post_process(outputs)[0]
        print(f"\nPost-processed detections: {detections.shape}")  # [N, 6] (x1,y1,x2,y2,conf,cls)
    
    # Visualize
    visualize_detections(image_path, detections, class_names, scale)

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
    'toothbrush'
]

class AdaptivePreprocessing(nn.Module):
    """
    Adaptive preprocessing layer to improve generalization across different image conditions.
    Handles variations in brightness, contrast, and environmental conditions.
    
    Args:
        in_channels (int): Number of input channels (default=3 for RGB)
        out_channels (int): Number of output channels (default=3)
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Global context branch - change output channels to match out_channels
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(in_channels, out_channels, 1),  # Changed from 32 to out_channels
            nn.Sigmoid()
        )
        
        # Local enhancement branch with depth-wise separable convolutions
        self.local_branch = nn.Sequential(
            # Depth-wise separable convolution for efficiency
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )
        
        # Channel attention for adaptive feature weighting
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Gamma and beta parameters for adaptive instance normalization
        self.gamma = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
        # Instance normalization for style adaptation
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=False)

    def forward(self, x):
        # Global context features
        global_features = self.global_context(x)
        
        # Local enhancement
        local_features = self.local_branch(x)
        
        # Apply instance normalization with learnable affine parameters
        normalized = self.instance_norm(local_features)
        adaptive_norm = normalized * self.gamma + self.beta
        
        # Apply channel attention
        channel_weights = self.channel_attention(adaptive_norm)
        
        # Combine global and local features with channel attention
        enhanced = adaptive_norm * channel_weights * global_features
        
        # Residual connection
        return enhanced + x

class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal positional embeddings for DETR-style transformer.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, x):
        """
        Create sinusoidal positional embeddings.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            pos: Positional embeddings [batch_size, embed_dim, height, width]
        """
        bs, _, h, w = x.shape
        
        # Create normalized coordinates
        y_embed = torch.arange(h, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(w, dtype=torch.float32, device=x.device)
        
        if self.normalize:
            y_embed = y_embed / (h - 1) * 2 - 1
            x_embed = x_embed / (w - 1) * 2 - 1
        
        # Create grid of coordinates
        y_embed = y_embed.view(h, 1).repeat(1, w)
        x_embed = x_embed.view(1, w).repeat(h, 1)
        
        # Create positional embeddings
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        pos = pos.unsqueeze(0).repeat(bs, 1, 1, 1)
        
        return pos

class DETRHead(nn.Module):
    """
    DETR-like transformer-based detection head.
    
    Args:
        in_channels (int): Number of input channels
        hidden_dim (int): Hidden dimension of transformer
        num_classes (int): Number of detection classes
        num_queries (int): Number of object queries
        nheads (int): Number of attention heads
        num_encoder_layers (int): Number of transformer encoder layers
        num_decoder_layers (int): Number of transformer decoder layers
        dim_feedforward (int): Dimension of feedforward network
    """
    def __init__(
        self, 
        in_channels=256, 
        hidden_dim=256, 
        num_classes=80,
        num_queries=100,
        nheads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024
    ):
        super().__init__()
        
        # Projection from backbone features to transformer dimension
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding2D(hidden_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Object queries (learnable)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Output heads for class and box prediction
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 3-layer MLP for box regression
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes

    def forward(self, x):
        """
        Args:
            x: Features from the backbone [batch_size, channels, height, width]
            
        Returns:
            pred_logits: Class predictions [batch_size, num_queries, num_classes+1]
            pred_boxes: Box predictions [batch_size, num_queries, 4] (normalized xywh)
        """
        bs, c, h, w = x.shape
        
        # Project features to transformer dimension
        src = self.input_proj(x)  # [batch_size, hidden_dim, height, width]
        
        # Generate positional encodings
        pos = self.pos_encoder(src)  # [batch_size, hidden_dim, height, width]
        
        # Flatten spatial dimensions for transformer
        src = src.flatten(2).permute(0, 2, 1)  # [batch_size, height*width, hidden_dim]
        pos = pos.flatten(2).permute(0, 2, 1)  # [batch_size, height*width, hidden_dim]
        
        # Transformer encoder
        memory = self.transformer_encoder(src + pos)  # [batch_size, height*width, hidden_dim]
        
        # Prepare object queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # [batch_size, num_queries, hidden_dim]
        
        # Transformer decoder
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer_decoder(tgt, memory)  # [batch_size, num_queries, hidden_dim]
        
        # Predict classes and boxes
        pred_logits = self.class_embed(hs)  # [batch_size, num_queries, num_classes+1]
        pred_boxes = self.bbox_embed(hs).sigmoid()  # [batch_size, num_queries, 4] (normalized xywh)
        
        return pred_logits, pred_boxes

class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for transformer-based detection.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding weights
        pe = torch.zeros(d_model, 128, 128)
        d_model_half = d_model // 2
        
        # Create position indices
        y_indices = torch.arange(0, 128).unsqueeze(1).expand(-1, 128).float()
        x_indices = torch.arange(0, 128).unsqueeze(0).expand(128, -1).float()
        
        # Apply sine/cosine positional encoding
        div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * (-math.log(10000.0) / d_model_half))
        
        # Apply encoding to even indices
        pe[0:d_model_half:2, :, :] = torch.sin(x_indices.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[1:d_model_half:2, :, :] = torch.cos(x_indices.unsqueeze(0) * div_term.view(-1, 1, 1))
        
        # Apply encoding to odd indices
        pe[d_model_half::2, :, :] = torch.sin(y_indices.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[d_model_half+1::2, :, :] = torch.cos(y_indices.unsqueeze(0) * div_term.view(-1, 1, 1))
        
        # Register as buffer (not a parameter but should be saved and loaded)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Positional encoding added to input [batch_size, channels, height, width]
        """
        batch_size, _, h, w = x.shape
        
        # Get positional encoding of appropriate size
        pe = F.interpolate(self.pe.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
        pe = pe.repeat(batch_size, 1, 1, 1)
        
        return pe

class MLP(nn.Module):
    """
    Simple multi-layer perceptron with ReLU activations.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class YOLODETRLoss(nn.Module):
    """
    Combined loss function for YOLO-DETR hybrid model.
    Combines YOLOv8 loss with DETR-style loss for transformer-based detection.
    
    Args:
        num_classes (int): Number of detection classes
        detr_weight (float): Weight for DETR loss component
    """
    def __init__(self, num_classes=80, detr_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.detr_weight = detr_weight
        
        # YOLO loss components
        self.yolo_loss = YOLOv8Loss(num_classes=num_classes)
        
        # DETR loss components
        self.matcher = HungarianMatcher()
        self.weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        self.empty_weight = torch.ones(num_classes + 1)
        self.empty_weight[-1] = 0.1  # Lower weight for no-object class
        
    def forward(self, outputs, targets):
        """
        Compute combined YOLO-DETR loss.
        
        Args:
            outputs: Dictionary containing model outputs:
                - 'yolo': List of YOLO detection outputs
                - 'detr': Tuple of (pred_logits, pred_boxes) from DETR head
            targets: Ground truth targets dictionary containing:
                - 'boxes': Target boxes in xyxy format [batch_size, num_targets, 4]
                - 'class': Target class indices [batch_size, num_targets]
            
        Returns:
            Dictionary of loss components
        """
        device = outputs['yolo'][0].device
        
        # Calculate YOLO loss
        yolo_loss_dict = self.yolo_loss(outputs['yolo'], targets)
        
        # Calculate DETR loss
        pred_logits, pred_boxes = outputs['detr']
        
        # Convert targets to DETR format
        detr_targets = []
        for i in range(len(targets['boxes'])):
            detr_targets.append({
                'labels': targets['class'][i],
                'boxes': self.xyxy2xywh(targets['boxes'][i])  # Convert to xywh format
            })
        
        # Match predictions to targets
        indices = self.matcher(pred_logits, pred_boxes, detr_targets)
        
        # Compute DETR classification loss
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(detr_targets, indices)])
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, 
                                   dtype=torch.int64, device=pred_logits.device)
        
        for i, (_, J) in enumerate(indices):
            target_classes[i, J] = target_classes_o
        
        loss_ce = F.cross_entropy(pred_logits.flatten(0, 1), target_classes.flatten(0, 1), 
                                 weight=self.empty_weight.to(pred_logits.device))
        
        # Compute DETR box loss
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(detr_targets, indices)], dim=0)
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum(-1).mean()
        
        # GIoU loss
        loss_giou = 1 - torch.diag(self.generalized_box_iou(
            self.xywh2xyxy(src_boxes),
            self.xywh2xyxy(target_boxes)
        )).mean()
        
        # Combine DETR losses
        detr_loss = loss_ce * self.weight_dict['loss_ce'] + \
                   loss_bbox * self.weight_dict['loss_bbox'] + \
                   loss_giou * self.weight_dict['loss_giou']
        
        # Combine YOLO and DETR losses
        total_loss = yolo_loss_dict['loss'] * (1 - self.detr_weight) + detr_loss * self.detr_weight
        
        return {
            'loss': total_loss,
            'box_loss': yolo_loss_dict['box_loss'],
            'cls_loss': yolo_loss_dict['cls_loss'],
            'detr_loss': detr_loss,
            'detr_ce_loss': loss_ce,
            'detr_bbox_loss': loss_bbox,
            'detr_giou_loss': loss_giou
        }
    
    def _get_src_permutation_idx(self, indices):
        """Helper method for DETR loss calculation."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    @staticmethod
    def xywh2xyxy(x):
        """Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2] format."""
        y = x.clone()
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1 = x_center - width/2
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1 = y_center - height/2
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2 = x_center + width/2
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2 = y_center + height/2
        return y
    
    @staticmethod
    def xyxy2xywh(x):
        """Convert boxes from [x1, y1, x2, y2] to [x_center, y_center, width, height] format."""
        y = x.clone()
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x_center = (x1 + x2) / 2
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y_center = (y1 + y2) / 2
        y[..., 2] = x[..., 2] - x[..., 0]        # width = x2 - x1
        y[..., 3] = x[..., 3] - x[..., 1]        # height = y2 - y1
        return y
    
    @staticmethod
    def generalized_box_iou(boxes1, boxes2):
        """
        Compute generalized IoU between boxes.
        """
        # Standard IoU
        iou = YOLOv8Loss.box_iou(boxes1, boxes2)
        
        # Find the enclosing box
        enc_left = torch.min(boxes1[..., 0], boxes2[..., 0])
        enc_top = torch.min(boxes1[..., 1], boxes2[..., 1])
        enc_right = torch.max(boxes1[..., 2], boxes2[..., 2])
        enc_bottom = torch.max(boxes1[..., 3], boxes2[..., 3])
        
        # Calculate area of enclosing box
        enc_width = enc_right - enc_left
        enc_height = enc_bottom - enc_top
        enc_area = enc_width * enc_height
        
        # Calculate areas of both boxes
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # Union area
        union = boxes1_area + boxes2_area - iou * (boxes1_area + boxes2_area - boxes1_area * boxes2_area)
        
        # GIoU = IoU - (area of enclosing box - union) / area of enclosing box
        giou = iou - (enc_area - union) / (enc_area + 1e-7)
        
        return giou

class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for DETR-style assignment between predictions and targets.
    Computes optimal assignment between predictions and ground truth using Hungarian algorithm.
    """
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Compute assignment between predictions and targets using Hungarian algorithm.
        
        Args:
            outputs: Dict with 'pred_logits' and 'pred_boxes'
                - pred_logits: Class predictions [batch_size, num_queries, num_classes+1]
                - pred_boxes: Box predictions [batch_size, num_queries, 4] (normalized xywh)
            targets: List of dicts with 'labels' and 'boxes'
                - labels: Target class indices [num_targets]
                - boxes: Target boxes [num_targets, 4] (normalized xywh)
                
        Returns:
            List of tuples (pred_idx, target_idx) for each batch item
        """
        bs, num_queries = outputs[0].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs[0].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes+1]
        out_bbox = outputs[1].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # List to store indices for each batch item
        indices = []
        
        # Process each batch item separately
        for b in range(bs):
            # Skip if no targets for this batch item
            if len(targets[b]['labels']) == 0:
                indices.append(([], []))
                continue
            
            # Get target boxes and labels for this batch item
            tgt_bbox = targets[b]['boxes']
            tgt_ids = targets[b]['labels']
            
            # Classification cost: -log(p_c) where p_c is the probability of the correct class
            cost_class = -out_prob[b * num_queries:(b + 1) * num_queries, tgt_ids]
            
            # L1 box distance
            cost_bbox = torch.cdist(out_bbox[b * num_queries:(b + 1) * num_queries], tgt_bbox, p=1)
            
            # GIoU cost
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b * num_queries:(b + 1) * num_queries]),
                box_cxcywh_to_xyxy(tgt_bbox)
            )
            
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            
            # Hungarian algorithm to find optimal assignment
            indices_b = linear_sum_assignment(C.cpu().numpy())
            indices.append((torch.as_tensor(indices_b[0], dtype=torch.int64),
                           torch.as_tensor(indices_b[1], dtype=torch.int64)))
        
        return indices


def box_cxcywh_to_xyxy(x):
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    """
    Compute generalized IoU between boxes.
    
    Args:
        boxes1: First set of boxes in xyxy format
        boxes2: Second set of boxes in xyxy format
        
    Returns:
        Generalized IoU
    """
    # Calculate IoU
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # Compute union
    union = area1[:, None] + area2 - inter
    
    # Compute IoU
    iou = inter / union
    
    # Compute enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    enclosing_area = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # Compute GIoU
    giou = iou - (enclosing_area - union) / enclosing_area
    
    return giou


class YOLOMulti(nn.Module):
    """
    Unified YOLO architecture that can be configured as YOLOv8 or YOLO12 with different scales.
    
    This class provides a flexible implementation that shares common code between
    YOLOv8 and YOLO12 architectures while allowing for specific customizations.
    
    Args:
        version (str): YOLO version to use ('v8' or 'v12')
        scale (str): Model scale ('n', 's', 'm', 'l', 'x')
        num_classes (int): Number of detection classes
        in_channels (int): Number of input channels (default=3 for RGB)
        attention_type (str, optional): Type of attention for YOLO12 ('se', 'cbam', 'transformer', None)
        pretrained (bool): Whether to load pretrained weights
        pretrained_path (str, optional): Path to pretrained weights
    """
    def __init__(
        self, 
        version='v8',
        scale='s',
        num_classes=80, 
        in_channels=3,
        attention_type=None,
        pretrained=False,
        pretrained_path=None
    ):
        super().__init__()
        self.version = version.lower()
        self.scale = scale.lower()
        self.num_classes = num_classes
        self.attention_type = attention_type
        
        # Validate configuration
        self._validate_config()
        
        # Get depth and width multipliers based on scale
        depth_multiple, width_multiple = self._get_scale_multipliers()
        
        # Calculate channel dimensions with width multiplier
        c1 = int(64 * width_multiple)    # stem
        c2 = int(128 * width_multiple)   # stage 1
        c3 = int(256 * width_multiple)   # stage 2
        c4 = int(512 * width_multiple)   # stage 3
        
        # For YOLO12, we use a wider final stage
        if self.version == 'v12':
            c5 = int(512 * width_multiple * 1.5)  # stage 4 (wider than YOLOv8)
        else:
            c5 = int(512 * width_multiple)  # stage 4 (standard for YOLOv8)
        
        # Store channel dimensions for reference
        self.channels = [c1, c2, c3, c4, c5]
        
        # Add adaptive preprocessing layer for custom YOLO-DETR version
        if self.version == 'yolo-detr':
            self.adaptive_layer = AdaptivePreprocessing(in_channels=in_channels, out_channels=in_channels)
        else:
            self.adaptive_layer = nn.Identity()

        # Build backbone
        self.backbone = self._build_backbone()
        
        # Build neck (Feature Pyramid Network)
        self.neck = self._build_neck()
        
        # Detection head
        if self.version == 'yolo-detr':
            # For YOLO-DETR, we use both YOLO detection head and DETR head
            self.detect = Detect(
                in_channels=[c3, c3, c3],  # P3, P4, P5 all have same channel count
                num_classes=num_classes,
                strides=[8, 16, 32]  # Strides for P3, P4, P5
            )
            
            # DETR head for medium scale (P4, 40x40)
            # self.detr_head = DETRHead(
            #     in_channels=c3,  # Same as P4 channels
            #     hidden_dim=256,
            #     num_classes=num_classes,
            #     num_queries=100,  # Number of object queries
            #     nheads=8,
            #     num_encoder_layers=3,
            #     num_decoder_layers=3
            # )
            self.detr_head = DETRHead(
                in_channels=c3,  # Same as P4 channels
                hidden_dim=128,
                num_classes=num_classes,
                num_queries=20,  # Number of object queries
                nheads=4,
                num_encoder_layers=1,
                num_decoder_layers=1
            )
        else:
            # Standard YOLO detection head for other versions
            self.detect = Detect(
                in_channels=[c3, c3, c3],  # P3, P4, P5 all have same channel count
                num_classes=num_classes,
                strides=[8, 16, 32]  # Strides for P3, P4, P5
            )
        
        # DFL for box regression
        self.dfl = DFL(self.detect.reg_max)
        
        # Initialize weights
        self.init_weights()
        
        # Load pretrained weights if specified
        if pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)
    
    def _validate_config(self):
        """Validate model configuration to catch errors early."""
        # Validate version
        #valid_versions = ['v8', 'v12']
        valid_versions = ['v8', 'v12', 'yolo-detr']
        if self.version not in valid_versions:
            raise ValueError(f"Invalid version '{self.version}'. Must be one of {valid_versions}")
        
        # Validate scale
        valid_scales = ['n', 's', 'm', 'l', 'x']
        if self.scale not in valid_scales:
            raise ValueError(f"Invalid scale '{self.scale}'. Must be one of {valid_scales}")
        
        # Validate attention type for YOLO12
        if self.version == 'v12':
            valid_attention_types = ['se', 'cbam', 'transformer', None]
            if self.attention_type not in valid_attention_types:
                raise ValueError(f"Invalid attention_type '{self.attention_type}' for YOLO12. "
                                f"Must be one of {valid_attention_types}")
    
    def _get_scale_multipliers(self):
        """Get depth and width multipliers based on model scale."""
        # Scale to depth/width multipliers mapping
        scale_params = {
            # scale: (depth_multiple, width_multiple)
            'n': (0.33, 0.25),  # nano
            's': (0.33, 0.50),  # small
            'm': (0.67, 0.75),  # medium
            'l': (1.00, 1.00),  # large
            'x': (1.33, 1.25),  # extra large
        }
        
        # Get multipliers for the selected scale
        depth_multiple, width_multiple = scale_params[self.scale]
        
        # For YOLO-DETR, we use slightly larger models for better feature extraction
        # if self.version == 'yolo-detr':
        #     # Increase depth for better feature extraction
        #     depth_multiple *= 1.2
        #     # Increase width for richer features
        #     width_multiple *= 1.1
        # For YOLO-DETR, we use the same multipliers as YOLOv8 to maintain compatibility

        return depth_multiple, width_multiple
    
    
    def _build_backbone(self):
        """Build backbone network based on version and scale."""
        # Get depth and width multipliers
        depth_multiple, width_multiple = self._get_scale_multipliers()
        
        # Calculate channel dimensions with width multiplier
        c1 = int(64 * width_multiple)    # stem
        c2 = int(128 * width_multiple)   # stage 1
        c3 = int(256 * width_multiple)   # stage 2
        c4 = int(512 * width_multiple)   # stage 3
        
        # # For YOLO12 and YOLO-DETR, we use a wider final stage
        # if self.version in ['v12', 'yolo-detr']:
        #     c5 = int(512 * width_multiple * 1.5)  # stage 4 (wider)
        # else:
        #     c5 = int(512 * width_multiple)  # stage 4 (standard)
        # For YOLO12, we use a wider final stage
        # But for YOLO-DETR, use the same structure as YOLOv8
        if self.version == 'v12':
            c5 = int(512 * width_multiple * 1.5)  # stage 4 (wider)
        else:
            c5 = int(512 * width_multiple)  # stage 4 (standard for YOLOv8 and YOLO-DETR)

        # Build backbone
        backbone = nn.Sequential(
            # stem
            Conv(3, c1), 
            
            # stage 1 (1/2)
            Conv(c1, c2, 3, 2),
            C2f(c2, c2, n=round(3 * depth_multiple)),
            
            # stage 2 (1/4)
            Conv(c2, c3, 3, 2),
            C2f(c3, c3, n=round(6 * depth_multiple)),
            
            # stage 3 (1/8)
            Conv(c3, c4, 3, 2),
            C2f(c4, c4, n=round(6 * depth_multiple)),
            
            # stage 4 (1/16)
            Conv(c4, c5, 3, 2),
            C2f(c5, c5, n=round(3 * depth_multiple)),
        )
        
        return backbone
    
    def _build_neck(self):
        """Build neck (Feature Pyramid Network) based on version and scale."""
        # Get channel dimensions
        c3 = self.channels[2]  # P3 channels
        c4 = self.channels[3]  # P4 channels
        c5 = self.channels[4]  # P5 channels
        
        # Build neck components
        # SPPF on the backbone output
        sppf = SPPF(c5, c5)
        
        # Top-down path (upsampling)
        lateral_conv1 = Conv(c5, c3, 1, 1)  # Reduce channels for P5->P4
        upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        C3_p4 = C2f(c3 + c4, c3, n=3, shortcut=False)
        
        lateral_conv2 = Conv(c3, c3, 1, 1)  # Maintain channels for P4->P3
        upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        C3_p3 = C2f(c3 + c3, c3, n=3, shortcut=False)
        
        # Bottom-up path (downsampling)
        down_conv1 = Conv(c3, c3, 3, 2)  # P3->P4
        C3_n3 = C2f(c3 + c3, c3, n=3, shortcut=False)
        
        down_conv2 = Conv(c3, c3, 3, 2)  # P4->P5
        C3_n4 = C2f(c3 + c3, c3, n=3, shortcut=False)
        
        # For YOLO-DETR, add extra attention in the neck
        if self.version == 'yolo-detr':
            # Add transformer-based cross-attention for feature enhancement
            # This helps the model better understand global context
            neck = nn.ModuleList([
                sppf,
                lateral_conv1,
                upsample1,
                nn.Sequential(
                    C3_p4,
                    lateral_conv2,
                    upsample2,
                    C3_p3,
                    down_conv1,
                    C3_n3,
                    down_conv2,
                    C3_n4
                )
            ])
        else:
            # Standard neck for other YOLO versions
            neck = nn.ModuleList([
                sppf,
                lateral_conv1,
                upsample1,
                nn.Sequential(
                    C3_p4,
                    lateral_conv2,
                    upsample2,
                    C3_p3,
                    down_conv1,
                    C3_n3,
                    down_conv2,
                    C3_n4
                )
            ])
        
        return neck
    
    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for detection head to prevent false positives initially
        if hasattr(self, 'detect'):
            # Find the final classification layer in the detection head
            for name, module in self.detect.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Check if this is the final layer (outputs classes + box coords)
                    if module.out_channels == self.detect.reg_max * 4 + self.num_classes:
                        # Initialize bias for classification part with strong negative values
                        # This will make sigmoid(output) close to zero initially
                        if module.bias is not None:
                            # Create a bias tensor
                            bias = torch.zeros_like(module.bias.data)
                            
                            # Set strong negative bias for classification outputs
                            # The last self.num_classes elements are for classification
                            bias[self.detect.reg_max * 4:] = -10.0  # This gives sigmoid(x) ≈ 4.5e-5
                            
                            # Apply the bias
                            module.bias.data = bias
                            print(f"Initialized detection head with conservative bias")
    
    def _load_pretrained(self, pretrained_path):
        """Load pretrained weights from file."""
        if not os.path.exists(pretrained_path):
            print(f"Warning: Pretrained weights file {pretrained_path} not found.")
            return
        
        # Load checkpoint
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Extract model weights
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load weights with shape check
        model_state_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_state_dict and v.shape == model_state_dict[k].shape}
        
        # Report loading status
        missing_keys = set(model_state_dict.keys()) - set(pretrained_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
        
        if missing_keys:
            print(f"Missing keys in pretrained weights: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys in pretrained weights: {len(unexpected_keys)}")
        
        # Load the weights
        model_state_dict.update(pretrained_dict)
        self.load_state_dict(model_state_dict)
        print(f"Loaded pretrained weights from {pretrained_path}")

    def forward(self, x, do_debug=False):
        """
        Forward pass through YOLOMulti model.
        
        Args:
            x: Input tensor [batch_size, 3, height, width]
            
        Returns:
            For training: Raw outputs from detection head(s)
            For inference: Post-processed detections
        """
        # Apply adaptive preprocessing for YOLO-DETR
        x = self.adaptive_layer(x) #[1, 3, 640, 640]=>[1, 3, 640, 640]
        
        # Backbone
        x1 = self.backbone[:4](x)      # P2 (1/4 scale) [1, 192, 160, 160]
        x2 = self.backbone[4:7](x1)    # P3 (1/8 scale) [1, 384, 80, 80]
        x3 = self.backbone[7:](x2)     # P4 (1/16 scale) [1, 384, 40, 40]
        
        # Debug information
        if do_debug == True:
            print(f"Backbone output shapes: x1={x1.shape}, x2={x2.shape}, x3={x3.shape}")
            print(f"SPPF input channels: {self.neck[0].cv1.conv.in_channels}")

        # Neck
        # SPPF
        sppf = self.neck[0](x3)  # Enhanced P5 features [1, 384, 40, 40]
        
        # Top-down path (upsampling)
        p5 = self.neck[1](sppf) #[1, 192, 40, 40]
        p5_up = self.neck[2](p5) #[1, 192, 80, 80]
        p4 = torch.cat([p5_up, x2], 1)  # Concatenate with backbone P3 [1, 576, 80, 80]
        p4 = self.neck[3][0](p4) #[1, 192, 80, 80]
        
        p4_up = self.neck[3][2](self.neck[3][1](p4)) #[1, 192, 160, 160]
        p3 = torch.cat([p4_up, x1], 1)  # Concatenate with backbone P2 [1, 384, 160, 160]
        p3 = self.neck[3][3](p3)  # P3 (1/8 scale, 80x80) [1, 192, 160, 160]
        
        # Bottom-up path (downsampling)
        p3_down = self.neck[3][4](p3) #[1, 192, 80, 80]
        p4 = torch.cat([p3_down, p4], 1) #[1, 384, 80, 80]
        p4 = self.neck[3][5](p4)  # P4 (1/16 scale, 40x40) [1, 192, 80, 80]
        
        p4_down = self.neck[3][6](p4) #[1, 192, 40, 40]
        p5 = torch.cat([p4_down, p5], 1) #[1, 384, 40, 40]
        p5 = self.neck[3][7](p5)  # P5 (1/32 scale, 20x20) [1, 192, 40, 40]
        
        # Detection head(s)
        if self.version == 'yolo-detr':
            # YOLO detection head for all scales
            yolo_outputs = self.detect([p3, p4, p5]) #[1, 144, 160, 160], [1, 144, 80, 80], [1, 144, 40, 40]
            
            # DETR head for medium scale (P4, 40x40)
            detr_logits, detr_boxes = self.detr_head(p4) #[1, 192, 80, 80]=>[1, 100, 81], [1, 100, 4]
            
            # Combine outputs for training
            if self.training:
                return {
                    'yolo': yolo_outputs,
                    'detr': (detr_logits, detr_boxes)
                }
            else:
                # Post-process for inference
                return self.post_process_hybrid(yolo_outputs, detr_logits, detr_boxes)
        else:
            # Standard YOLO detection for other versions
            outputs = self.detect([p3, p4, p5])
            
            if self.training:
                return outputs
            else:
                # Post-process for inference
                return self.post_process(outputs)

    def post_process(self, outputs, conf_thres=0.25, iou_thres=0.45):
        """
        Post-process standard YOLO outputs for inference.
        
        Args:
            outputs: Raw model outputs
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            
        Returns:
            List of detections (xyxy, conf, cls) for each image
        """
        detections = []
        for output in outputs:
            # Reshape output: (batch, channels, height, width) -> (batch, anchors, classes+box)
            bs, ch, ny, nx = output.shape #[1, 144, 160, 160]
            
            # Calculate the expected size and verify it matches
            expected_size = bs * ch * ny * nx
            actual_size = output.numel()
            
            if expected_size != actual_size:
                print(f"Warning: Expected tensor size {expected_size}, got {actual_size}")
                print(f"Shape: {output.shape}, attempting to reshape safely")
            
            # Calculate the correct number of channels for reshape
            num_channels = self.detect.reg_max * 4 + self.num_classes
            
            # Safely reshape by flattening first, then reshaping
            output = output.reshape(bs, -1, ny * nx)  # Flatten channels [1, 144, 25600]
            
            # If channel dimension doesn't match expected, adjust it
            if output.shape[1] != num_channels:
                print(f"Channel mismatch: got {output.shape[1]}, expected {num_channels}")
                # Try to infer the correct reshape based on actual dimensions
                output = output.reshape(bs, num_channels, -1)
            
            # Permute to get the right format
            output = output.permute(0, 2, 1) #[1, 25600, 144]
            
            # Decode boxes (DFL)
            box, cls = output.split((self.detect.reg_max * 4, self.num_classes), 2) #[1, 25600, 64], [1, 25600, 80]
            box = self.decode_box(box) #[1, 25600, 4]
            
            # Apply confidence threshold
            conf, j = cls.sigmoid().max(2, keepdim=True) #[1, 25600, 80]=>[1, 25600, 1], 
            mask = (conf.squeeze(2) > conf_thres) #[1, 25600]
            
            # Process each batch item
            for b in range(bs):
                b_mask = mask[b]
                if not b_mask.any():
                    # No detections for this batch item
                    detections.append(torch.zeros((0, 6), device=output.device))
                    continue
                
                # Get boxes, confidence scores, and class indices for this batch item
                b_box = box[b, b_mask]
                b_conf = conf[b, b_mask]
                b_j = j[b, b_mask]
                
                # Perform NMS
                keep = batched_nms(b_box, b_conf.squeeze(1), b_j.squeeze(1), iou_thres)
                
                # Combine detections: [x1, y1, x2, y2, conf, class]
                det = torch.cat([b_box[keep], b_conf[keep], b_j[keep].float()], 1)
                detections.append(det)
        
        return detections
    
    def post_process_hybrid(self, yolo_outputs, detr_logits, detr_boxes, conf_thres=0.25, iou_thres=0.45):
        """
        Post-process hybrid YOLO-DETR outputs for inference.
        
        Args:
            yolo_outputs: Raw YOLO detection outputs
            detr_logits: DETR class logits [batch_size, num_queries, num_classes+1]
            detr_boxes: DETR box predictions [batch_size, num_queries, 4] (normalized xywh)
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            
        Returns:
            List of fused detections (xyxy, conf, cls) for each image
        """
        bs = yolo_outputs[0].shape[0]
        device = yolo_outputs[0].device
        all_detections = []
        
        # Process YOLO detections
        yolo_detections = self.post_process(yolo_outputs, conf_thres, iou_thres)
        #[[78, 6], [71, 6], [59, 6]]
        # Process DETR detections
        detr_detections = []
        for b in range(bs):
            # Get class probabilities
            prob = F.softmax(detr_logits[b], dim=-1) #[100, 81]
            scores, labels = prob[..., :-1].max(-1)  # Exclude no-object class
            #[100], [100]
            # Filter by confidence
            keep = scores > conf_thres
            scores = scores[keep]
            labels = labels[keep]
            boxes = detr_boxes[b, keep]
            
            # Convert boxes from xywh to xyxy format
            boxes_xyxy = torch.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = x - w/2
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = y - h/2
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = x + w/2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = y + h/2
            
            # Perform NMS
            if len(scores) > 0:
                keep = batched_nms(boxes_xyxy, scores, labels, iou_thres)
                boxes_xyxy = boxes_xyxy[keep]
                scores = scores[keep].unsqueeze(1)
                labels = labels[keep].unsqueeze(1).float()
                
                # Combine: [x1, y1, x2, y2, conf, class]
                det = torch.cat([boxes_xyxy, scores, labels], 1)
            else:
                det = torch.zeros((0, 6), device=device)
            
            detr_detections.append(det)
        
        # Fuse YOLO and DETR detections
        for b in range(bs):
            yolo_det = yolo_detections[b]
            detr_det = detr_detections[b]
            
            # Combine detections
            combined_det = torch.cat([yolo_det, detr_det], 0)
            
            # Perform NMS on combined detections
            if len(combined_det) > 0:
                keep = batched_nms(
                    combined_det[:, :4],
                    combined_det[:, 4],
                    combined_det[:, 5],
                    iou_thres
                )
                combined_det = combined_det[keep]
            
            all_detections.append(combined_det) #[78, 6]
        
        return all_detections

    def decode_box(self, box):
        """
        Decode predicted boxes from DFL distribution.
        
        Args:
            box: Predicted box distribution (bs, anchors, reg_max * 4)
            
        Returns:
            Decoded boxes in xyxy format
        """
        bs, anchors = box.shape[:2]  # [1, 25600, 64]
        
        # Reshape box to separate the reg_max dimension for each coordinate
        # From [bs, anchors, reg_max*4] to [bs, anchors, 4, reg_max]
        box = box.reshape(bs, anchors, 4, self.detect.reg_max) #[1, 25600, 4, 16]
        
        # Apply DFL to each coordinate separately
        # We need to process each batch and anchor separately
        decoded_boxes = []
        for b in range(bs):
            batch_boxes = []
            for a in range(anchors):
                # Get the 4 coordinates with their distributions
                coord_dists = box[b, a]  # Shape: [4, reg_max] [4, 16]
                
                # Apply softmax to each distribution
                coord_dists = F.softmax(coord_dists, dim=1)
                
                # Calculate expected value for each coordinate
                coords = (coord_dists * torch.arange(self.detect.reg_max, device=box.device)).sum(dim=1)
                
                batch_boxes.append(coords)
            
            # Stack all anchors for this batch
            decoded_boxes.append(torch.stack(batch_boxes))
        
        # Stack all batches
        decoded_box = torch.stack(decoded_boxes)  # Shape: [bs, anchors, 4]
        
        # Convert from xywh to xyxy
        decoded_box_xyxy = torch.cat([
            decoded_box[..., :2] - decoded_box[..., 2:] / 2,  # x1y1
            decoded_box[..., :2] + decoded_box[..., 2:] / 2   # x2y2
        ], -1)
        
        return decoded_box_xyxy
    
    def get_model_info(self):
        """Return model information as a dictionary."""
        return {
            'version': self.version,
            'scale': self.scale,
            'num_classes': self.num_classes,
            'attention_type': self.attention_type,
            'parameters': sum(p.numel() for p in self.parameters()),
            'channels': self.channels,
        }

# Helper function to create YOLOMulti models with specific configurations
def create_yolo(
    version='v8', 
    scale='s', 
    num_classes=80, 
    attention_type=None, 
    pretrained=False, 
    pretrained_path=None
):
    """
    Create a YOLO model with the specified configuration.
    
    Args:
        version (str): YOLO version ('v8' or 'v12')
        scale (str): Model scale ('n', 's', 'm', 'l', 'x')
        num_classes (int): Number of detection classes
        attention_type (str, optional): Type of attention for YOLO12 ('se', 'cbam', 'transformer', None)
        pretrained (bool): Whether to load pretrained weights
        pretrained_path (str, optional): Path to pretrained weights
        
    Returns:
        YOLOMulti: Configured YOLO model
    """
    model = YOLOMulti(
        version=version,
        scale=scale,
        num_classes=num_classes,
        attention_type=attention_type,
        pretrained=pretrained,
        pretrained_path=pretrained_path
    )
    
    print(f"Created {version.upper()} model ({scale}) with {model.get_model_info()['parameters']:,} parameters")
    if version == 'v12' and attention_type:
        print(f"Using {attention_type.upper()} attention mechanism")
    
    return model

def train_yolo_detr(
    model,
    data_path,
    dataset_type='coco',  # 'coco' or 'kitti'
    epochs=100,
    batch_size=16,
    img_size=640,
    lr=0.01,
    weight_decay=0.0005,
    momentum=0.937,
    warmup_epochs=3,
    save_dir='output',
    resume=None
):
    """
    Train YOLO-DETR hybrid model on COCO or KITTI dataset.
    
    Args:
        model: YOLO-DETR model
        data_path: Path to dataset
        dataset_type: Type of dataset ('coco' or 'kitti')
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        lr: Initial learning rate
        weight_decay: Weight decay for optimizer
        momentum: Momentum for optimizer
        warmup_epochs: Number of warmup epochs
        save_dir: Directory to save model weights
        resume: Path to checkpoint to resume training from
    """
    import os
    import time
    from datetime import datetime
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Dataset
    from torchvision.datasets import CocoDetection
    from torch.optim import SGD
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = model.to(device)
    
    # Define loss function
    criterion = YOLODETRLoss(num_classes=model.num_classes)
    
    # Define optimizer
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    # Define learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    
    # Custom COCO dataset with transformations
    class COCOYOLODataset(CocoDetection):
        def __init__(self, root, annFile, transform=None):
            super().__init__(root, annFile, transform)
            # Map COCO category IDs to continuous indices
            self.cat_ids = sorted(self.coco.getCatIds())
            self.cat_id_to_continuous_id = {
                coco_id: i for i, coco_id in enumerate(self.cat_ids)
            }
            
            # Get all image IDs that have annotations
            self.ids = []
            for img_id in self.coco.getImgIds():
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                if len(ann_ids) > 0:
                    self.ids.append(img_id)
        
        def __getitem__(self, index):
            img_id = self.ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco.loadAnns(ann_ids)
            
            # Load image
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.root, img_info['file_name'])
            img = Image.open(img_path).convert('RGB')
            
            # Original dimensions
            orig_width, orig_height = img.size
            
            # Calculate scaling factor
            scale = min(img_size / orig_width, img_size / orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.BILINEAR)
            
            # Create a canvas with padding
            padded_img = Image.new('RGB', (img_size, img_size), (114, 114, 114))
            padded_img.paste(img, (0, 0))
            
            # Convert to tensor and normalize
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(padded_img)
            
            # Process annotations
            boxes = []
            classes = []
            
            for ann in annotations:
                # Skip annotations with no area or iscrowd=1
                if ann['area'] <= 0 or ann['iscrowd']:
                    continue
                
                # Get box coordinates (COCO format: [x, y, width, height])
                x, y, w, h = ann['bbox']
                
                # Convert to normalized coordinates and scale to new size
                x1 = x * scale / img_size
                y1 = y * scale / img_size
                x2 = (x + w) * scale / img_size
                y2 = (y + h) * scale / img_size
                
                # Clip to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(1, x2), min(1, y2)
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Add box [x1, y1, x2, y2]
                boxes.append([x1, y1, x2, y2])
                
                # Map COCO category ID to continuous index
                cat_id = ann['category_id']
                class_idx = self.cat_id_to_continuous_id[cat_id]
                classes.append(class_idx)
            
            # Convert to tensors
            if len(boxes) > 0:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                classes = torch.tensor(classes, dtype=torch.int64)
            else:
                # No valid annotations
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                classes = torch.zeros((0), dtype=torch.int64)
            
            return img_tensor, {'boxes': boxes, 'class': classes}
    
    # Custom KITTI dataset
    class KITTIYOLODataset(Dataset):
        def __init__(self, root, split='training'):
            self.root = root
            self.split = split
            self.img_dir = os.path.join(root, split, 'image_2')
            self.label_dir = os.path.join(root, split, 'label_2')
            
            # Get all image files
            self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
            
            # KITTI class mapping (simplified to common classes)
            self.class_map = {
                'Car': 0,
                'Van': 0,  # Map to Car
                'Truck': 1,
                'Pedestrian': 2,
                'Person_sitting': 2,  # Map to Pedestrian
                'Cyclist': 3,
                'Tram': 4,
                'Misc': 5,
                'DontCare': -1  # Ignore
            }
            
            # Count number of classes
            self.num_classes = len(set(v for v in self.class_map.values() if v >= 0))
        
        def __len__(self):
            return len(self.img_files)
        
        def __getitem__(self, index):
            # Load image
            img_file = self.img_files[index]
            img_path = os.path.join(self.img_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            
            # Original dimensions
            orig_width, orig_height = img.size
            
            # Calculate scaling factor
            scale = min(img_size / orig_width, img_size / orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.BILINEAR)
            
            # Create a canvas with padding
            padded_img = Image.new('RGB', (img_size, img_size), (114, 114, 114))
            padded_img.paste(img, (0, 0))
            
            # Convert to tensor and normalize
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(padded_img)
            
            # Load labels
            label_file = os.path.join(self.label_dir, img_file.replace('.png', '.txt'))
            boxes = []
            classes = []
            
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        obj_class = parts[0]
                        
                        # Skip DontCare or unknown classes
                        if obj_class not in self.class_map or self.class_map[obj_class] < 0:
                            continue
                        
                        # KITTI format: [left, top, right, bottom] in pixel coordinates
                        x1 = float(parts[4])
                        y1 = float(parts[5])
                        x2 = float(parts[6])
                        y2 = float(parts[7])
                        
                        # Convert to normalized coordinates and scale to new size
                        x1 = x1 * scale / img_size
                        y1 = y1 * scale / img_size
                        x2 = x2 * scale / img_size
                        y2 = y2 * scale / img_size
                        
                        # Clip to image boundaries
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(1, x2), min(1, y2)
                        
                        # Skip invalid boxes
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Add box [x1, y1, x2, y2]
                        boxes.append([x1, y1, x2, y2])
                        
                        # Get class index
                        class_idx = self.class_map[obj_class]
                        classes.append(class_idx)
            
            # Convert to tensors
            if len(boxes) > 0:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                classes = torch.tensor(classes, dtype=torch.int64)
            else:
                # No valid annotations
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                classes = torch.zeros((0), dtype=torch.int64)
            
            return img_tensor, {'boxes': boxes, 'class': classes}
    
    # Create datasets based on dataset type
    if dataset_type.lower() == 'coco':
        print("Loading COCO dataset...")
        train_dataset = COCOYOLODataset(
            root=os.path.join(data_path, 'train2017'),
            annFile=os.path.join(data_path, 'annotations', 'instances_train2017.json')
        )
        
        val_dataset = COCOYOLODataset(
            root=os.path.join(data_path, 'val2017'),
            annFile=os.path.join(data_path, 'annotations', 'instances_val2017.json')
        )
    elif dataset_type.lower() == 'kitti':
        print("Loading KITTI dataset...")
        # For KITTI, we'll split the training data into train/val
        kitti_dataset = KITTIYOLODataset(root=data_path, split='training')
        
        # Split dataset: 80% train, 20% val
        dataset_size = len(kitti_dataset)
        train_size = int(dataset_size * 0.8)
        val_size = dataset_size - train_size
        
        # Use random_split to create train/val datasets
        from torch.utils.data import random_split
        train_dataset, val_dataset = random_split(
            kitti_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Update model's num_classes if needed
        if model.num_classes != kitti_dataset.num_classes:
            print(f"Warning: Model has {model.num_classes} classes but KITTI dataset has {kitti_dataset.num_classes} classes")
            print("You may need to adjust the model's output layers accordingly")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Choose 'coco' or 'kitti'.")
    
    # Collate function for batching
    def collate_fn(batch):
        return (
            torch.stack([item[0] for item in batch]),
            {
                'boxes': [item[1]['boxes'] for item in batch],
                'class': [item[1]['class'] for item in batch]
            }
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_map = 0
    if resume:
        if os.path.isfile(resume):
            print(f"Loading checkpoint '{resume}'")
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_map = checkpoint.get('best_map', 0)
            print(f"Loaded checkpoint '{resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{resume}'")
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs):
        # Set model to training mode
        model.train()
        
        # Initialize metrics
        epoch_loss = 0
        epoch_box_loss = 0
        epoch_cls_loss = 0
        epoch_detr_loss = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        # Adjust learning rate for warmup
        if epoch < warmup_epochs:
            # Linear warmup
            lr_scale = min(1., (epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = lr * lr_scale
        
        # Training step
        for i, (imgs, targets) in enumerate(pbar):
            # Move data to device
            imgs = imgs.to(device) #[8, 3, 640, 640]
            batch_targets = {
                'boxes': [box.to(device) for box in targets['boxes']],
                'class': [cls.to(device) for cls in targets['class']]
            }
            
            # Forward pass
            outputs = model(imgs)
            
            # Calculate loss
            loss_dict = criterion(outputs, batch_targets)
            loss = loss_dict['loss']
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_box_loss += loss_dict['box_loss'].item()
            epoch_cls_loss += loss_dict['cls_loss'].item()
            epoch_detr_loss += loss_dict['detr_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': epoch_loss / (i + 1),
                'box_loss': epoch_box_loss / (i + 1),
                'cls_loss': epoch_cls_loss / (i + 1),
                'detr_loss': epoch_detr_loss / (i + 1),
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Update learning rate scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Calculate average losses
        avg_loss = epoch_loss / len(train_loader)
        avg_box_loss = epoch_box_loss / len(train_loader)
        avg_cls_loss = epoch_cls_loss / len(train_loader)
        avg_detr_loss = epoch_detr_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Box Loss: {avg_box_loss:.4f}, "
              f"Cls Loss: {avg_cls_loss:.4f}, DETR Loss: {avg_detr_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_detr_loss = 0
        
        print("Running validation...")
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader):
                # Move data to device
                imgs = imgs.to(device)
                batch_targets = {
                    'boxes': [box.to(device) for box in targets['boxes']],
                    'class': [cls.to(device) for cls in targets['class']]
                }
                
                # Forward pass
                outputs = model(imgs)
                
                # Calculate loss
                loss_dict = criterion(outputs, batch_targets)
                val_loss += loss_dict['loss'].item()
                val_detr_loss += loss_dict['detr_loss'].item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        avg_val_detr_loss = val_detr_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}, DETR Loss: {avg_val_detr_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': avg_loss,
            'val_loss': avg_val_loss,
            'best_map': best_map,
            'dataset_type': dataset_type
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(save_dir, f'last_{dataset_type}.pt'))
        
        # Save best model (using validation loss as metric for simplicity)
        # In a real implementation, you would calculate mAP here
        if avg_val_loss < best_map or best_map == 0:
            best_map = avg_val_loss
            torch.save(checkpoint, os.path.join(save_dir, f'best_{dataset_type}.pt'))
            print(f"Saved best model with validation loss: {avg_val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(save_dir, f'{dataset_type}_epoch_{epoch+1}.pt'))
    
    print(f"Training complete on {dataset_type} dataset!")
    return model

class YOLODETRHead(nn.Module):
    """
    DETR-style transformer head for YOLO-DETR hybrid model.
    
    Args:
        in_channels (int): Number of input channels
        hidden_dim (int): Hidden dimension of transformer
        num_classes (int): Number of detection classes
        num_queries (int): Number of object queries
        nheads (int): Number of attention heads
        num_encoder_layers (int): Number of encoder layers
        num_decoder_layers (int): Number of decoder layers
        dim_feedforward (int): Dimension of feedforward network
        dropout (float): Dropout rate
    """
    def __init__(
        self,
        in_channels=256,
        hidden_dim=256,
        num_classes=80,
        num_queries=100,
        nheads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Input projection from backbone features to transformer dimension
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # Positional encoding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            return_intermediate_dec=True
        )
        
        # Object queries (learnable)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 3-layer MLP for box regression
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize weights for transformer and output heads."""
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize bbox_embed with smaller weights
        for layer in self.bbox_embed.layers:
            nn.init.xavier_uniform_(layer.weight, gain=0.01)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through DETR head.
        
        Args:
            x: Input features from backbone [batch_size, channels, height, width]
            
        Returns:
            pred_logits: Class predictions [batch_size, num_queries, num_classes+1]
            pred_boxes: Box predictions [batch_size, num_queries, 4] (normalized xywh)
        """
        bs, c, h, w = x.shape
        
        # Project input features to transformer dimension
        src = self.input_proj(x)  # [batch_size, hidden_dim, height, width]
        
        # Flatten spatial dimensions and transpose to sequence format
        src = src.flatten(2).permute(2, 0, 1)  # [height*width, batch_size, hidden_dim]
        
        # Generate positional encodings
        pos = self.position_embedding(x).flatten(2).permute(2, 0, 1)  # [height*width, batch_size, hidden_dim]
        
        # Object queries with positional encodings
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # [num_queries, batch_size, hidden_dim]
        
        # Create target for decoder (zeros)
        tgt = torch.zeros_like(query_embed)  # [num_queries, batch_size, hidden_dim]
        
        # Transformer forward pass
        # For simplicity, we're using the standard PyTorch Transformer
        # In a real implementation, you might want to use a custom transformer with memory-efficient attention
        memory = self.transformer.encoder(src, src_key_padding_mask=None, pos=pos)
        hs = self.transformer.decoder(
            tgt, memory, 
            memory_key_padding_mask=None,
            pos=pos, query_pos=query_embed
        )
        
        # Get last decoder layer output
        hs = hs[-1]  # [num_queries, batch_size, hidden_dim]
        
        # Transpose to batch-first format
        hs = hs.transpose(0, 1)  # [batch_size, num_queries, hidden_dim]
        
        # Predict classes and boxes
        pred_logits = self.class_embed(hs)  # [batch_size, num_queries, num_classes+1]
        pred_boxes = self.bbox_embed(hs).sigmoid()  # [batch_size, num_queries, 4] (normalized xywh)
        
        return pred_logits, pred_boxes

def test_yolo_inference(image_path, model, class_names, conf_thresh=0.25, iou_thresh=0.45, img_size=640, device=None):
    """
    Test YOLO model on a single image with detailed visualization and debugging information.
    
    Args:
        image_path (str): Path to input image file
        model (nn.Module): YOLO model instance
        class_names (list): List of class names for visualization
        conf_thresh (float): Confidence threshold for filtering detections (default: 0.25)
        iou_thresh (float): IoU threshold for NMS (default: 0.45)
        img_size (int): Input image size for model (default: 640)
        device (torch.device): Device to run inference on (default: auto-detect)
        
    Returns:
        tuple: (processed_image, detections, visualization)
            - processed_image: Preprocessed image tensor
            - detections: Raw detection results
            - visualization: Matplotlib figure with visualized detections
    """
    import time
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms as T
    import matplotlib.patches as patches
    
    # Set device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Running inference on device: {device}")
    
    # Load and preprocess image
    start_time = time.time()
    img_tensor, original_dims, scale = load_and_preprocess_image(image_path, img_size) #[1, 3, 640, 640]
    preprocess_time = time.time() - start_time
    print(f"Preprocessing time: {preprocess_time:.3f}s")
    print(f"Input image tensor shape: {img_tensor.shape}") #[1, 3, 640, 640]
    
    # Move model and input to device
    model = model.to(device)
    img_tensor = img_tensor.to(device)
    
    # Run model in evaluation mode
    model.eval()
    with torch.no_grad():
        # Measure inference time
        start_time = time.time()
        
        # Forward pass
        detections = model(img_tensor)
        #[[74, 6]]
        # Post-process
        if isinstance(detections, dict):
            # For hybrid model, get the combined detections
            detections = detections[0]  # First batch item
        else:
            # For standard YOLO model, get the first batch item
            detections = detections[0]

        inference_time = time.time() - start_time
    
    # Print performance metrics
    print(f"Inference time: {inference_time:.3f}s ({1/inference_time:.1f} FPS)")
    print(f"Total detections: {len(detections)}")
    
    # Visualize
    visualize_detections(image_path, detections, class_names, scale, conf_thresh)
    
    return detections

def test_yolomulti():
    import os
    import urllib.request
    
    # Download example image if not exists
    image_url = "https://images.unsplash.com/photo-1551269901-5c5e14c25df7"
    image_path = "example_image.jpg"
    
    if not os.path.exists(image_path):
        urllib.request.urlretrieve(image_url, image_path)
        print(f"\nDownloaded example image to {image_path}")
    
    # Create different YOLO models
    # Initialize YOLO-DETR hybrid model
    model = YOLOMulti(
        num_classes=80,
        version='yolo-detr',
        scale='m',
        in_channels=3
    )

    print("Creating YOLOv8-s model...")
    yolov8s = create_yolo(version='v8', scale='s')
    
    print("\nCreating YOLOv8-l model...")
    yolov8l = create_yolo(version='v8', scale='l')
    
    print("\nCreating YOLO12-s model with CBAM attention...")
    yolo12s_cbam = create_yolo(version='v12', scale='s', attention_type='cbam')
    
    print("\nCreating YOLO12-m model with Transformer attention...")
    yolo12m_transformer = create_yolo(version='v12', scale='m', attention_type='transformer')
    
    test_yolo_inference(image_path, model, COCO_CLASSES)
    test_yolo_inference(image_path, yolov8s, COCO_CLASSES)
    test_yolo_inference(image_path, yolov8l, COCO_CLASSES)
    test_yolo_inference(image_path, yolo12s_cbam, COCO_CLASSES)
    test_yolo_inference(image_path, yolo12m_transformer, COCO_CLASSES)

def start_training():
    # Initialize YOLO-DETR hybrid model
    model = YOLOMulti(
        num_classes=6, #80,
        version='yolo-detr',
        scale='m',
        in_channels=3
    )
    # Start training
    train_yolo_detr(model=model, data_path="/mnt/f/Dataset/Kitti", dataset_type="kitti", epochs=20, batch_size=4, \
        img_size=640, save_dir = 'output/yolomulti')

if __name__ == "__main__":
    #test_yolomulti()
    start_training()