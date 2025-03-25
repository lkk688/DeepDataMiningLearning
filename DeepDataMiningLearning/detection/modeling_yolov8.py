import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms
from typing import List, Tuple, Dict, Optional

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

        # Build detection head
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, x, 3), 
                Conv(x, x, 3), 
                nn.Conv2d(x, (4 + num_classes) * self.reg_max, 1)) 
            for x in in_channels)
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights for detection head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through detection head."""
        outputs = []
        for i in range(self.nl):
            outputs.append(self.cv2[i](x[i]))
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
        b, c, h, w = x.shape
        x = x.softmax(1)
        return self.conv(x)

class YOLOv8(nn.Module):
    """
    YOLOv8 object detection model.
    
    Args:
        num_classes: Number of detection classes
        in_channels: Number of input channels (default=3 for RGB)
        depth_multiple: Model depth multiplier
        width_multiple: Model width multiplier
    """
    def __init__(self, num_classes=80, in_channels=3, depth_multiple=1.0, width_multiple=1.0):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = nn.Sequential(
            # stem
            Conv(in_channels, int(64 * width_multiple)), 
            
            # stage 1
            Conv(int(64 * width_multiple), int(128 * width_multiple), 3, 2),
            C2f(int(128 * width_multiple), int(128 * width_multiple), 
                n=round(3 * depth_multiple)),
            
            # stage 2
            Conv(int(128 * width_multiple), int(256 * width_multiple), 3, 2),
            C2f(int(256 * width_multiple), int(256 * width_multiple), 
                n=round(6 * depth_multiple)),
            
            # stage 3
            Conv(int(256 * width_multiple), int(512 * width_multiple), 3, 2),
            C2f(int(512 * width_multiple), int(512 * width_multiple), 
                n=round(6 * depth_multiple)),
            
            # stage 4
            Conv(int(512 * width_multiple), int(512 * width_multiple * width_multiple), 3, 2),
            C2f(int(512 * width_multiple * width_multiple), 
                int(512 * width_multiple * width_multiple), 
                n=round(3 * depth_multiple)),
        )
        
        # Neck (Feature Pyramid Network)
        self.neck = nn.Sequential(
            SPPF(int(512 * width_multiple * width_multiple), 
                 int(512 * width_multiple * width_multiple)),
            
            # Upsample path
            Conv(int(512 * width_multiple * width_multiple), 
                 int(256 * width_multiple), 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Top-down fusion
            nn.Sequential(
                C2f(int(768 * width_multiple), int(256 * width_multiple), 
                    n=round(3 * depth_multiple), shortcut=False),
                Conv(int(256 * width_multiple), int(128 * width_multiple), 1, 1),
                nn.Upsample(scale_factor=2, mode='nearest'),
                
                # Middle fusion
                C2f(int(384 * width_multiple), int(128 * width_multiple), 
                    n=round(3 * depth_multiple), shortcut=False),
            ),
        )
        
        # Head
        self.detect = Detect(
            in_channels=[int(128 * width_multiple), 
                        int(256 * width_multiple), 
                        int(512 * width_multiple * width_multiple)],
            num_classes=num_classes,
            strides=[8, 16, 32])
        
        # DFL for box regression
        self.dfl = DFL(self.detect.reg_max)
        
        # Initialize weights
        self.init_weights()

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
        x1 = self.backbone[:4](x)  # stage 1-2
        x2 = self.backbone[4:7](x1)  # stage 3
        x3 = self.backbone[7:](x2)  # stage 4
        
        # Neck
        p3 = self.neck[0](x3)  # SPPF
        p3 = self.neck[1](p3)  # Conv
        p3 = self.neck[2](p3)  # Upsample
        
        p2 = torch.cat([p3, x2], 1)
        p2 = self.neck[3][0](p2)  # C2f
        p2 = self.neck[3][1](p2)  # Conv
        p2 = self.neck[3][2](p2)  # Upsample
        
        p1 = torch.cat([p2, x1], 1)
        p1 = self.neck[3][3](p1)  # C2f
        
        # Head
        outputs = self.detect([p1, p2, p3])
        
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
            # Reshape output: (batch, anchors, (x,y,w,h,conf,cls...))
            bs, _, ny, nx = output.shape
            output = output.view(bs, self.detect.reg_max * 4 + self.num_classes, -1).permute(0, 2, 1)
            
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
    - Classification loss (BCE)
    - Box regression loss (DFL + CIoU)
    """
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dfl = DFL()
        
    def forward(self, preds, targets):
        """
        Compute YOLOv8 loss.
        
        Args:
            preds: Model predictions (list of outputs from Detect head)
            targets: Ground truth targets
            
        Returns:
            Dictionary of loss components
        """
        device = preds[0].device
        lcls = torch.zeros(1, device=device)  # Class loss
        lbox = torch.zeros(1, device=device)  # Box loss
        
        # Process each prediction level
        for i, pred in enumerate(preds):
            # Reshape predictions
            bs, _, ny, nx = pred.shape
            pred = pred.view(bs, 4 + self.num_classes, -1).permute(0, 2, 1)
            
            # Decode boxes
            box_pred, cls_pred = pred.split((4, self.num_classes), 2)
            box_pred = self.dfl(box_pred)
            
            # Match predictions to targets
            # (This is simplified - actual implementation needs anchor matching)
            # For demonstration, we'll assume targets are already matched
            
            # Classification loss
            tcls = torch.zeros_like(cls_pred)
            # Assign target classes (simplified)
            # In practice, this would come from target matching
            tcls[..., targets['class']] = 1.0
            lcls += self.bce(cls_pred, tcls).mean()
            
            # Box regression loss (CIoU)
            # Again simplified - actual implementation needs proper matching
            pbox = self.xywh2xyxy(box_pred)
            tbox = targets['boxes']
            lbox += (1.0 - self.box_iou(pbox, tbox)).mean()
        
        return {
            'loss': lbox + lcls,
            'box_loss': lbox,
            'cls_loss': lcls
        }
    
    @staticmethod
    def xywh2xyxy(x):
        """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
        y = x.clone()
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
        return y
    
    @staticmethod
    def box_iou(box1, box2):
        """
        Compute intersection over union (IoU) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        """
        # Get coordinates of intersection rectangles
        inter_left = torch.max(box1[..., 0], box2[..., 0])
        inter_top = torch.max(box1[..., 1], box2[..., 1])
        inter_right = torch.min(box1[..., 2], box2[..., 2])
        inter_bottom = torch.min(box1[..., 3], box2[..., 3])
        
        # Intersection area
        inter_area = torch.clamp(inter_right - inter_left, min=0) * \
                     torch.clamp(inter_bottom - inter_top, min=0)
        
        # Union area
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union_area = box1_area + box2_area - inter_area
        
        # IoU
        return inter_area / (union_area + 1e-7)

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
        p2 = model.neck[3][1](p2)  # Conv
        print(f"Conv output: {p2.shape}")       # [1, 128, 80, 80]
        p2 = model.neck[3][2](p2)  # Upsample
        print(f"Upsample output: {p2.shape}")   # [1, 128, 160, 160]
        
        p1 = torch.cat([p2, x1], 1)
        print(f"Concat output: {p1.shape}")     # [1, 384, 160, 160]
        p1 = model.neck[3][3](p1)  # C2f
        print(f"Final neck output: {p1.shape}") # [1, 128, 160, 160]
        
        # Head
        print("\nHead outputs:")
        outputs = model.detect([p1, p2, p3])
        for i, o in enumerate(outputs):
            print(f"Detection head {i} output: {o.shape}")  # [1, 68, 80, 80], [1, 68, 40, 40], [1, 68, 20, 20]
        
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

# Example usage
if __name__ == "__main__":
    import os
    import urllib.request
    
    # Initialize model
    model = YOLOv8(num_classes=80)
    print("Model initialized with", sum(p.numel() for p in model.parameters()), "parameters")
    
    # Download example image if not exists
    image_url = "https://images.unsplash.com/photo-1551269901-5c5e14c25df7"
    image_path = "example_image.jpg"
    
    if not os.path.exists(image_path):
        urllib.request.urlretrieve(image_url, image_path)
        print(f"Downloaded example image to {image_path}")
    
    # Run the test
    test_yolov8_on_image(image_path, model, COCO_CLASSES)