import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from typing import List, Tuple, Dict, Optional
import torchvision
from torchvision.transforms import functional as transF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FCOS(nn.Module):
    """
    FCOS: Fully Convolutional One-Stage Object Detection
    
    Key features:
    - Anchor-free design (no need for anchor boxes)
    - Per-pixel prediction on feature maps
    - Multi-level prediction with FPN
    - Center-ness branch for suppressing low-quality predictions
    
    Paper: https://arxiv.org/abs/1904.01355
    """
    
    def __init__(self, backbone: nn.Module, num_classes: int, strides: List[int] = [8, 16, 32, 64, 128]):
        """
        Initialize FCOS model.
        
        Args:
            backbone: Backbone network (typically ResNet) for feature extraction
            num_classes: Number of object classes (including background)
            strides: Strides of feature maps at different FPN levels
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.strides = strides

        self.fpn = FPN(backbone.out_channels)  # FPN processes multi-level features
        
        # Heads now accept lists of channels
        self.cls_head = FCOSHead(self.fpn.out_channels, num_classes)  # [256, 256, 256] for P3-P5
        self.bbox_head = FCOSHead(self.fpn.out_channels, 4)  # 4 for (l, t, r, b)
        self.centerness_head = FCOSHead(self.fpn.out_channels, 1)

        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """
        Initialize model weights following FCOS paper specifications.
        """
        # Initialize classification head with bias -log((1-π)/π), π=0.01
        # This helps stabilize training in the early stages
        nn.init.constant_(self.cls_head.conv_cls.bias, -torch.log(torch.tensor((1 - 0.01) / 0.01)).item())
        
        # Initialize regression head with bias 0.1 (helps with small objects)
        for layer in [self.bbox_head.conv_cls, self.centerness_head.conv_cls]:
            nn.init.constant_(layer.bias, 0.1)
        
        # Initialize all conv layers with normal distribution
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Forward pass of FCOS.
        
        Args:
            images: Input images of shape (batch_size, 3, height, width)
            
        Returns:
            Tuple containing:
            - Dictionary of raw predictions for loss computation
            - List of dictionaries containing post-processed predictions for each image
        """
        # Get features from backbone
        features = self.backbone(images)
        
        # Pass through FPN to get multi-scale features
        fpn_features = self.fpn(features)
        
        # Initialize output containers
        cls_logits = []
        bbox_preds = []
        centerness_preds = []
        
        # Apply heads to each FPN level
        for feature in fpn_features:
            cls_logits.append(self.cls_head(feature))
            bbox_preds.append(self.bbox_head(feature))
            centerness_preds.append(self.centerness_head(feature))
        
        # Prepare raw outputs for loss computation
        raw_outputs = {
            "cls_logits": cls_logits,
            "bbox_preds": bbox_preds,
            "centerness_preds": centerness_preds,
        }
        
        # Post-process predictions (during inference)
        if not self.training:
            return raw_outputs, self.post_process(raw_outputs)
        
        return raw_outputs, []
    
    def post_process(self, raw_outputs: Dict[str, List[torch.Tensor]], 
                    score_thresh: float = 0.05, 
                    nms_thresh: float = 0.5, 
                    topk_candidates: int = 1000) -> List[Dict[str, torch.Tensor]]:
        """
        Post-process raw network outputs to get final detections.
        
        Args:
            raw_outputs: Raw network outputs from forward pass
            score_thresh: Minimum score threshold for keeping detections
            nms_thresh: IoU threshold for non-maximum suppression
            topk_candidates: Keep only topk candidates before NMS
            
        Returns:
            List of dictionaries (one per image) containing:
            - "boxes": Detected bounding boxes [N, 4] (x1, y1, x2, y2 format)
            - "labels": Class labels [N]
            - "scores": Confidence scores [N]
        """
        cls_logits = raw_outputs["cls_logits"]
        bbox_preds = raw_outputs["bbox_preds"]
        centerness_preds = raw_outputs["centerness_preds"]
        
        batch_size = cls_logits[0].shape[0]
        device = cls_logits[0].device
        
        # Initialize results container
        results = [{"boxes": torch.empty(0, 4, device=device), 
                    "labels": torch.empty(0, dtype=torch.int64, device=device), 
                    "scores": torch.empty(0, device=device)} 
                   for _ in range(batch_size)]
        
        # Process each feature level
        for level, (cls_logit, bbox_pred, centerness_pred, stride) in enumerate(zip(
            cls_logits, bbox_preds, centerness_preds, self.strides)):
            
            # Convert predictions to probabilities
            cls_prob = torch.sigmoid(cls_logit)
            centerness = torch.sigmoid(centerness_pred)
            
            # Combine classification score with center-ness
            scores = torch.sqrt(cls_prob * centerness)
            
            # Get height and width of current feature map
            height, width = cls_logit.shape[-2:]
            
            # Generate grid points (locations) for this feature map
            shifts_x = torch.arange(0, width * stride, step=stride, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, height * stride, step=stride, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2  # center of each cell
            
            # Process each image in batch
            for img_idx in range(batch_size):
                # Get predictions for this image
                img_bbox_pred = bbox_pred[img_idx].view(4, -1).permute(1, 0)  # [H*W, 4]
                img_scores = scores[img_idx].view(self.num_classes, -1).permute(1, 0)  # [H*W, C]
                
                # Decode bbox predictions (l, t, r, b) to (x1, y1, x2, y2)
                img_boxes = self.decode_boxes(locations, img_bbox_pred)
                
                # Select top-k candidates
                if topk_candidates < img_scores.shape[0]:
                    topk_scores, topk_idxs = img_scores.flatten().topk(topk_candidates)
                    topk_classes = topk_idxs % self.num_classes
                    topk_box_idxs = topk_idxs // self.num_classes
                    topk_boxes = img_boxes[topk_box_idxs]
                else:
                    topk_scores = img_scores.flatten()
                    topk_classes = torch.arange(self.num_classes, device=device).repeat_interleave(
                        img_scores.shape[0])
                    topk_boxes = img_boxes.repeat(self.num_classes, 1)
                
                # Apply score threshold
                keep_idxs = topk_scores > score_thresh
                topk_scores = topk_scores[keep_idxs]
                topk_classes = topk_classes[keep_idxs]
                topk_boxes = topk_boxes[keep_idxs]
                
                # Skip if no detections
                if topk_scores.numel() == 0:
                    continue
                
                # Apply class-wise NMS
                keep = nms(topk_boxes, topk_scores, nms_thresh)
                topk_scores = topk_scores[keep]
                topk_classes = topk_classes[keep]
                topk_boxes = topk_boxes[keep]
                
                # Merge with previous results
                results[img_idx]["boxes"] = torch.cat([results[img_idx]["boxes"], topk_boxes])
                results[img_idx]["labels"] = torch.cat([results[img_idx]["labels"], topk_classes])
                results[img_idx]["scores"] = torch.cat([results[img_idx]["scores"], topk_scores])
        
        return results
    
    def decode_boxes(self, locations: torch.Tensor, pred_boxes: torch.Tensor) -> torch.Tensor:
        """
        Decode predicted box offsets (l, t, r, b) to (x1, y1, x2, y2) format.
        
        Args:
            locations: Center locations of each prediction [N, 2]
            pred_boxes: Predicted box offsets [N, 4] as (left, top, right, bottom)
            
        Returns:
            Decoded boxes in (x1, y1, x2, y2) format [N, 4]
        """
        x1 = locations[:, 0] - pred_boxes[:, 0]  # x - l
        y1 = locations[:, 1] - pred_boxes[:, 1]  # y - t
        x2 = locations[:, 0] + pred_boxes[:, 2]  # x + r
        y2 = locations[:, 1] + pred_boxes[:, 3]  # y + b
        return torch.stack([x1, y1, x2, y2], dim=1)


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) implementation for FCOS.
    Converts backbone features with different channels into multi-scale features with uniform channels.
    
    Args:
        in_channels_list (list[int]): Number of channels for each backbone output level
        out_channels (int): Number of channels in output feature maps (default=256)
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        
        # Lateral connections (1x1 convs to match channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Output convolutions (3x3 convs to smooth features)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
        # Extra layers for P6 and P7 (used in FCOS for larger objects)
        self.extra_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_list[-1], out_channels, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ),
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
        ])
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for FPN layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Forward pass of FPN.
        
        Args:
            features (list[Tensor]): Feature maps from backbone at different scales
            
        Returns:
            list[Tensor]: FPN feature maps [P3, P4, P5, P6, P7] with uniform channels
        """
        # Process backbone features through lateral connections
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path (starting from highest level)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')
        
        # Process through output convolutions
        features = [
            output_conv(laterals[i])
            for i, output_conv in enumerate(self.output_convs)
        ]
        
        # Add extra pyramid levels (P6, P7)
        for extra_conv in self.extra_convs:
            features.append(extra_conv(features[-1]))
        
        return features


class FCOSHead(nn.Module):
    """
    FCOS prediction head shared between classification, box regression and center-ness.
    
    Each head consists of 4 convolutional layers followed by a final prediction layer.
    """
    
    """Modified to handle lists of input channels."""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.heads = nn.ModuleList([
            self._build_head(in_channels, out_channels)
            for in_channels in in_channels_list
        ])
        # Initialize weights
        self.init_weights()

    def _build_head(self, in_channels, out_channels):
        """Build a head for one FPN level."""
        return nn.Sequential(
            Conv(in_channels, in_channels, 3),
            Conv(in_channels, in_channels, 3),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
    
    def init_weights(self):
        """
        Initialize weights for the head.
        """
        for m in self.convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.normal_(self.conv_cls.weight, std=0.01)
        if self.conv_cls.bias is not None:
            nn.init.constant_(self.conv_cls.bias, 0)
    
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Forward pass of FCOS head.
        
    #     Args:
    #         x: Input feature map from FPN
            
    #     Returns:
    #         Output predictions (classification, bbox, or center-ness)
    #     """
        # x = self.convs(x)
        # return self.conv_cls(x)
    def forward(self, x):
        """x is a list of features from different FPN levels."""
        return [head(f) for head, f in zip(self.heads, x)]


class FCOSLoss(nn.Module):
    """
    FCOS loss function combining:
    - Classification loss (Focal Loss)
    - Regression loss (GIoU Loss)
    - Center-ness loss (Binary Cross-Entropy)
    """
    
    def __init__(self, num_classes: int, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize FCOS loss.
        
        Args:
            num_classes: Number of object classes
            alpha: Alpha parameter for Focal Loss
            gamma: Gamma parameter for Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        
        # Initialize loss functions
        self.cls_loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        self.bbox_loss_fn = nn.SmoothL1Loss(reduction="none")
        self.centerness_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    
    def forward(self, preds: Dict[str, List[torch.Tensor]], 
               targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Compute FCOS losses.
        
        Args:
            preds: Dictionary containing model predictions with keys:
                - "cls_logits": List of classification logits per FPN level [N, C, H, W]
                - "bbox_preds": List of box predictions per FPN level [N, 4, H, W]
                - "centerness_preds": List of center-ness predictions per FPN level [N, 1, H, W]
            targets: List of target dictionaries (one per image) containing:
                - "labels": Tensor of class labels [M]
                - "boxes": Tensor of target boxes [M, 4]
                
        Returns:
            Dictionary containing loss values:
                - "loss_cls": Classification loss
                - "loss_reg": Regression loss
                - "loss_centerness": Center-ness loss
                - "total_loss": Combined total loss
        """
        cls_logits = preds["cls_logits"]
        bbox_preds = preds["bbox_preds"]
        centerness_preds = preds["centerness_preds"]
        
        # Initialize loss accumulators
        loss_cls = 0
        loss_reg = 0
        loss_centerness = 0
        num_pos = 0
        
        # Process each FPN level
        for level in range(len(cls_logits)):
            # Get predictions for this level
            cls_logit = cls_logits[level]
            bbox_pred = bbox_preds[level]
            centerness_pred = centerness_preds[level]
            
            # Get target tensors for this level
            target_labels, target_boxes, target_centerness = self.prepare_targets(
                cls_logit, targets, level)
            
            # Compute classification loss (Focal Loss)
            pos_mask = target_labels >= 0  # -1 indicates background/ignore
            valid_mask = target_labels != -1  # includes both positive and negative samples
            
            # Flatten predictions and targets
            cls_logit = cls_logit.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            centerness_pred = centerness_pred.permute(0, 2, 3, 1).reshape(-1)
            
            target_labels = target_labels.reshape(-1)
            target_boxes = target_boxes.reshape(-1, 4)
            target_centerness = target_centerness.reshape(-1)
            
            # Classification loss (only on valid samples)
            loss_cls += self.cls_loss_fn(
                cls_logit[valid_mask], 
                target_labels[valid_mask].long())
            
            # Only compute regression and center-ness loss for positive samples
            if pos_mask.sum() > 0:
                # Regression loss (GIoU or Smooth L1)
                bbox_loss = self.bbox_loss_fn(
                    bbox_pred[pos_mask], 
                    target_boxes[pos_mask])
                
                # Weight regression loss by center-ness target
                bbox_loss = bbox_loss.sum(dim=1) * target_centerness[pos_mask]
                loss_reg += bbox_loss.sum()
                
                # Center-ness loss (Binary Cross-Entropy)
                loss_centerness += self.centerness_loss_fn(
                    centerness_pred[pos_mask], 
                    target_centerness[pos_mask]).sum()
                
                num_pos += pos_mask.sum()
        
        # Normalize losses by number of positive samples
        if num_pos > 0:
            loss_reg = loss_reg / num_pos
            loss_centerness = loss_centerness / num_pos
        
        # Classification loss is normalized by number of FPN levels
        loss_cls = loss_cls / len(cls_logits)
        
        return {
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "loss_centerness": loss_centerness,
            "total_loss": loss_cls + loss_reg + loss_centerness,
        }
    
    def prepare_targets(self, cls_logit: torch.Tensor, 
                       targets: List[Dict[str, torch.Tensor]], 
                       level: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare target tensors for a specific FPN level.
        
        Args:
            cls_logit: Classification logits for this level [N, C, H, W]
            targets: List of target dictionaries (one per image)
            level: Current FPN level (0-based index)
            
        Returns:
            Tuple containing:
            - target_labels: Tensor of class labels [N, H, W]
            - target_boxes: Tensor of box targets [N, H, W, 4]
            - target_centerness: Tensor of center-ness targets [N, H, W]
        """
        batch_size = cls_logit.shape[0]
        height, width = cls_logit.shape[-2:]
        device = cls_logit.device
        
        # Initialize target tensors
        target_labels = torch.full((batch_size, height, width), -1, dtype=torch.long, device=device)
        target_boxes = torch.zeros((batch_size, height, width, 4), dtype=torch.float32, device=device)
        target_centerness = torch.zeros((batch_size, height, width), dtype=torch.float32, device=device)
        
        # Get stride for this level
        stride = 2 ** (level + 3)  # assuming P3 is level 0 with stride 8
        
        # Generate grid points (locations) for this feature map
        shifts_x = torch.arange(0, width * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, height * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2  # center of each cell
        
        # Process each image in batch
        for img_idx, target in enumerate(targets):
            if len(target["boxes"]) == 0:
                continue  # no objects in this image
            
            boxes = target["boxes"]
            labels = target["labels"]
            
            # Compute distances from each location to each box boundary
            # Shape: [num_locations, num_boxes, 4] (l, t, r, b)
            l = locations[:, 0, None] - boxes[None, :, 0]  # x - x1
            t = locations[:, 1, None] - boxes[None, :, 1]  # y - y1
            r = boxes[None, :, 2] - locations[:, 0, None]  # x2 - x
            b = boxes[None, :, 3] - locations[:, 1, None]  # y2 - y
            reg_targets = torch.stack([l, t, r, b], dim=2)  # [N, M, 4]
            
            # Find boxes that contain each location (all l,t,r,b > 0)
            is_in_boxes = reg_targets.min(dim=2)[0] > 0  # [N, M]
            
            # Limit regression range for each FPN level
            max_regress = reg_targets.max(dim=2)[0]  # [N, M]
            is_cared_in_level = (max_regress >= 4.0) & (max_regress <= 128.0)  # FCOS paper ranges
            
            # Combine conditions
            locations_to_boxes = is_in_boxes & is_cared_in_level
            
            # For each location, assign to box with minimal area (if any)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # [M]
            locations_to_areas = areas[None, :].expand(len(locations), len(areas))  # [N, M]
            locations_to_areas[~locations_to_boxes] = float("inf")
            
            # Find best box for each location
            min_areas, min_area_inds = locations_to_areas.min(dim=1)  # [N]
            
            # Create targets for positive locations (those assigned to a box)
            pos_locations = min_areas != float("inf")
            if pos_locations.sum() == 0:
                continue  # no positive locations for this image at this level
            
            # Get the assigned boxes and labels
            assigned_boxes = boxes[min_area_inds[pos_locations]]  # [K, 4]
            assigned_labels = labels[min_area_inds[pos_locations]]  # [K]
            assigned_reg_targets = reg_targets[pos_locations, min_area_inds[pos_locations]]  # [K, 4]
            
            # Compute center-ness targets (sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b)))
            lr = assigned_reg_targets[:, [0, 2]]
            tb = assigned_reg_targets[:, [1, 3]]
            centerness = (lr.min(dim=1)[0] / lr.max(dim=1)[0]) * (tb.min(dim=1)[0] / tb.max(dim=1)[0])
            centerness = torch.sqrt(centerness)
            
            # Reshape locations to match feature map dimensions
            pos_locations = pos_locations.view(height, width)
            assigned_labels = assigned_labels.view(height, width)
            assigned_reg_targets = assigned_reg_targets.view(height, width, 4)
            centerness = centerness.view(height, width)
            
            # Update target tensors
            target_labels[img_idx, pos_locations] = assigned_labels[pos_locations]
            target_boxes[img_idx, pos_locations] = assigned_reg_targets[pos_locations]
            target_centerness[img_idx, pos_locations] = centerness[pos_locations]
        
        return target_labels, target_boxes, target_centerness


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.
    
    This is a modified version of the original Focal Loss implementation from:
    https://arxiv.org/abs/1708.02002
    
    It addresses class imbalance by down-weighting well-classified examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "sum"):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare classes (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ("sum", "mean", or "none")
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            inputs: Raw classification logits [N, C]
            targets: Ground truth class labels [N] (0-based)
            
        Returns:
            Computed Focal Loss
        """
        # Convert inputs to probabilities
        p = torch.sigmoid(inputs)
        
        # Create one-hot encoded targets
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        
        # Compute binary cross-entropy
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction="none")
        
        # Compute p_t (probability of true class)
        p_t = p * targets_one_hot + (1 - p) * (1 - targets_one_hot)
        
        # Compute modulating factor
        alpha_factor = targets_one_hot * self.alpha + (1 - targets_one_hot) * (1 - self.alpha)
        modulating_factor = (1 - p_t) ** self.gamma
        
        # Combine all factors
        loss = alpha_factor * modulating_factor * ce_loss
        
        # Apply reduction
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:  # "none"
            return loss

class ResNet50Backbone(nn.Module):
    """
    Modified ResNet50 backbone for FCOS.
    Returns feature maps from different stages for FPN.
    """
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        
        # Extract the needed layers
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.layer1 = resnet.layer1  # Output stride 4
        self.layer2 = resnet.layer2  # Output stride 8
        self.layer3 = resnet.layer3  # Output stride 16
        self.layer4 = resnet.layer4  # Output stride 32
        
        # Output channels for each level
        self.out_channels = [256, 512, 1024, 2048]
    
    def forward(self, x):
        """
        Forward pass returning feature maps at different scales.
        
        Args:
            x: Input tensor of shape [N, 3, H, W]
            
        Returns:
            List of feature maps at different scales:
            - layer1: [N, 256, H/4, W/4]
            - layer2: [N, 512, H/8, W/8]
            - layer3: [N, 1024, H/16, W/16]
            - layer4: [N, 2048, H/32, W/32]
        """
        x = self.stem(x)       # [N, 64, H/2, W/2]
        c2 = self.layer1(x)    # [N, 256, H/4, W/4]
        c3 = self.layer2(c2)   # [N, 512, H/8, W/8]
        c4 = self.layer3(c3)   # [N, 1024, H/16, W/16]
        c5 = self.layer4(c4)   # [N, 2048, H/32, W/32]
        
        return [c3, c4, c5]  # Return only the layers we'll use in FPN (stride 8, 16, 32)

def load_and_preprocess_image(image_path, target_size=800):
    """
    Load and preprocess an image for FCOS.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the longer edge
        
    Returns:
        Preprocessed image tensor [1, 3, H, W] and original image dimensions
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    original_width, original_height = img.size
    
    # Resize
    scale = target_size / max(original_width, original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    img = img.resize((new_width, new_height), Image.BILINEAR)
    
    # Convert to tensor and normalize
    img_tensor = transF.to_tensor(img)  # [3, H, W], range [0, 1]
    img_tensor = transF.normalize(img_tensor, 
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
    
    return img_tensor, (original_height, original_width)

def visualize_detections(image_path, detections, score_threshold=0.5):
    """
    Visualize detections on the original image.
    
    Args:
        image_path: Path to the original image
        detections: Dictionary containing:
            - 'boxes': [N, 4] in (x1, y1, x2, y2) format
            - 'labels': [N] class indices
            - 'scores': [N] confidence scores
        score_threshold: Minimum score to display
    """
    # Load original image
    img = Image.open(image_path).convert("RGB")
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # Filter detections by score
    keep = detections['scores'] > score_threshold
    boxes = detections['boxes'][keep]
    scores = detections['scores'][keep]
    labels = detections['labels'][keep]
    
    # COCO class names (for visualization)
    coco_classes = [
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
    
    # Draw each detection
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.cpu().numpy()
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add label text
        label_text = f"{coco_classes[label]}: {score:.2f}"
        ax.text(x1, y1 - 5, label_text, color='red', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def test_fcos_on_image(image_path):
    """
    Test FCOS on a single image.
    
    Args:
        image_path: Path to the input image
    """
    # Initialize model
    backbone = ResNet50Backbone()
    model = FCOS(backbone, num_classes=80)  # 80 COCO classes
    model.eval()
    
    # Load and preprocess image
    img_tensor, original_dims = load_and_preprocess_image(image_path)
    print(f"Input image tensor shape: {img_tensor.shape}")  # [1, 3, H, W]
    # Original image loaded (e.g., 1200x800 RGB)
    # Resized to 800x533 (longer edge=800)
    # Normalized with ImageNet stats
    # Final input tensor: [1, 3, 533, 800] (batch, channels, height, width)
    
    # Forward pass through backbone
    with torch.no_grad():
        features = backbone(img_tensor)
        print("\nBackbone feature maps:")
        for i, feat in enumerate(features, 2):
            print(f"Level C{i}: {feat.shape}")  # C3: [1,512,H/8,W/8], C4: [1,1024,H/16,W/16], C5: [1,2048,H/32,W/32]
        # Backbone (ResNet50) Features
        #     C3 (from layer2): [1, 512, 67, 100] (H/8, W/8)
        #     C4 (from layer3): [1, 1024, 34, 50] (H/16, W/16)
        #     C5 (from layer4): [1, 2048, 17, 25] (H/32, W/32)

        # Forward pass through FCOS
        raw_outputs, detections = model(img_tensor)
        
        print("\nFPN feature maps:")
        for i, feat in enumerate(model.fpn(features), 3):
            print(f"Level P{i}: {feat.shape}")  # P3-P7 with channels 256
        # Feature Pyramid Network (FPN)
        #     P3: [1, 256, 67, 100] (from C3)
        #     P4: [1, 256, 34, 50] (from C4)
        #     P5: [1, 256, 17, 25] (from C5)
        #     P6: [1, 256, 9, 13] (from P5 with stride 2)
        #     P7: [1, 256, 5, 7] (from P6 with stride 2)

        print("\nRaw outputs:")
        for level, (cls_logit, bbox_pred, centerness_pred) in enumerate(zip(
            raw_outputs["cls_logits"], raw_outputs["bbox_preds"], raw_outputs["centerness_preds"])):
            print(f"Level {level}:")
            print(f"  Class logits: {cls_logit.shape}")    # [1,80,H,W]
            print(f"  Bbox preds: {bbox_pred.shape}")       # [1,4,H,W]
            print(f"  Centerness: {centerness_pred.shape}") # [1,1,H,W]
        
        # For each P3-P7 level:
        #     Classification head:
        #     Input: [1, 256, H, W]
        #     Output: [1, 80, H, W] (80 COCO classes)
        #     Bounding box head:
        #     Input: [1, 256, H, W]
        #     Output: [1, 4, H, W] (l, t, r, b offsets)
        #     Center-ness head:
        #     Input: [1, 256, H, W]
        #     Output: [1, 1, H, W] (center-ness score)
        
        # Process detections
        detections = detections[0]  # Get first (and only) image's detections
        print("\nPost-processed detections:")
        print(f"Boxes: {detections['boxes'].shape}")    # [N,4]
        print(f"Labels: {detections['labels'].shape}")  # [N]
        print(f"Scores: {detections['scores'].shape}")  # [N]
        
        # All predictions are decoded and combined across levels
        #     Final detections:
        #     Boxes: [N, 4] (x1, y1, x2, y2 in original image coordinates)
        #     Labels: [N] (class indices)
        #     Scores: [N] (confidence scores)
        
        # Scale boxes back to original image size
        scale = max(original_dims) / 800  # We resized to 800 on longer edge
        detections['boxes'] *= scale
        
        # Visualize
        visualize_detections(image_path, detections)
        
# Example usage:
if __name__ == "__main__":
    import os
    
    # Download an example image if not exists
    image_url = "https://images.unsplash.com/photo-1551269901-5c5e14c25df7"
    image_path = "example_image.jpg"
    
    if not os.path.exists(image_path):
        import urllib.request
        urllib.request.urlretrieve(image_url, image_path)
        print(f"Downloaded example image to {image_path}")
    
    # Run the test
    test_fcos_on_image(image_path)