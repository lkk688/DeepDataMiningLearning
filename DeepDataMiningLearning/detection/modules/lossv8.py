# Import necessary PyTorch modules for neural network operations
import torch  # Core PyTorch library for tensor operations
import torch.nn as nn  # Neural network modules and layers
import torch.nn.functional as F  # Functional interface for neural network operations
from typing import Tuple  # Type hints for function return types

# Import custom modules for YOLO detection tasks
from DeepDataMiningLearning.detection.modules.tal import TaskAlignedAssigner, dist2bbox, make_anchors, bbox2dist
from DeepDataMiningLearning.detection.modules.metrics import bbox_iou  # IoU calculation utilities
from DeepDataMiningLearning.detection.modules.utils import xywh2xyxy  # Bounding box format conversion

# Reference implementation from Ultralytics YOLOv8
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance in object detection.
    
    Wraps focal loss around existing loss function to reduce the relative loss for 
    well-classified examples and focus training on hard negatives.
    
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, ):
        """Initialize FocalLoss module."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """
        Calculate focal loss for binary classification.
        
        Args:
            pred (torch.Tensor): Predicted logits, shape [N, C] where N is batch size, C is classes
            label (torch.Tensor): Ground truth labels, shape [N, C], binary (0 or 1)
            gamma (float): Focusing parameter, higher gamma reduces loss for well-classified examples
            alpha (float): Weighting factor for rare class (typically minority class)
            
        Returns:
            torch.Tensor: Focal loss value, scalar tensor
        """
        # Calculate binary cross entropy loss without reduction
        # Shape: [N, C] -> [N, C]
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        
        # Alternative implementation (commented out):
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TensorFlow-style implementation for better numerical stability
        # Reference: https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        
        # Convert logits to probabilities using sigmoid activation
        # Shape: [N, C] -> [N, C]
        pred_prob = pred.sigmoid()  # prob from logits
        
        # Calculate p_t: probability of true class
        # For positive samples (label=1): p_t = pred_prob
        # For negative samples (label=0): p_t = 1 - pred_prob
        # Shape: [N, C] -> [N, C]
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        
        # Apply modulating factor (1 - p_t)^gamma to focus on hard examples
        # Shape: [N, C] -> [N, C]
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        
        # Apply alpha weighting if specified
        if alpha > 0:
            # Calculate alpha factor for class balancing
            # Shape: [N, C] -> [N, C]
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
            
        # Return mean loss across classes, then sum across batch
        # Shape: [N, C] -> [N] -> scalar
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """
    Distribution Focal Loss (DFL) for regression tasks in object detection.
    
    DFL treats the continuous regression target as a discrete probability distribution
    over a set of potential values, enabling more flexible and accurate regression.
    
    Reference: Generalized Focal Loss - https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, reg_max=16) -> None:
        """
        Initialize the DFL module.
        
        Args:
            reg_max (int): Maximum regression range, defines the discrete distribution size
        """
        super().__init__()
        self.reg_max = reg_max  # Maximum regression value (typically 16 for YOLOv8)

    def __call__(self, pred_dist, target):
        """
        Calculate Distribution Focal Loss for regression targets.
        
        Args:
            pred_dist (torch.Tensor): Predicted distribution logits
                                     Shape: [batch_size * num_anchors, reg_max + 1]
                                     Each row represents a probability distribution over reg_max+1 values
            target (torch.Tensor): Continuous regression targets
                                  Shape: [batch_size * num_anchors] or [batch_size, num_anchors, 4]
                                  Values should be in range [0, reg_max-1]
        
        Returns:
            torch.Tensor: DFL loss values
                         Shape: [batch_size * num_anchors, 1] or matches target shape
        
        Note:
            Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
            https://ieeexplore.ieee.org/document/9792391
        """
        # Clamp target values to valid range [0, reg_max-1-0.01] to avoid NaN
        # The small epsilon (0.01) prevents exact integer values that could cause issues
        # Shape: target -> target (same shape)
        target = target.clamp_(0, self.reg_max - 1 - 0.01)  # avoid NaN
        
        # Get left boundary (floor of target) - integer part
        # Shape: target -> target (same shape, but long dtype)
        tl = target.long()  # target left
        
        # Get right boundary (ceiling of target) - next integer
        # Shape: target -> target (same shape, long dtype)
        tr = tl + 1  # target right
        
        # Calculate weight for left boundary (1 - fractional part)
        # Shape: target -> target (same shape, float dtype)
        wl = tr - target  # weight left
        
        # Calculate weight for right boundary (fractional part)
        # Shape: target -> target (same shape, float dtype)
        wr = 1 - wl  # weight right
        
        # Calculate cross entropy loss for left boundary
        # pred_dist: [N, reg_max+1], tl.view(-1): [N] -> loss_left: [N]
        # Then reshape back to original target shape
        loss_left = F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
        
        # Calculate cross entropy loss for right boundary  
        # pred_dist: [N, reg_max+1], tr.view(-1): [N] -> loss_right: [N]
        # Then reshape back to original target shape
        loss_right = F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        
        # Combine weighted losses and return mean along last dimension
        # Shape: target -> [..., 1] (keepdim=True preserves dimensions)
        return (loss_left + loss_right).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """
    Bounding Box Loss module for object detection training.
    
    Combines IoU-based regression loss with optional Distribution Focal Loss (DFL)
    for more accurate bounding box prediction.
    """

    def __init__(self, reg_max, use_dfl=False):
        """
        Initialize the BboxLoss module with regularization maximum and DFL settings.
        
        Args:
            reg_max (int): Maximum regression range for DFL (typically 16)
            use_dfl (bool): Whether to use Distribution Focal Loss for regression
        """
        super().__init__()
        self.reg_max = reg_max  # Maximum regression value for DFL
        self.use_dfl = use_dfl  # Flag to enable/disable DFL
        # Initialize DFL module if enabled, otherwise None
        self.dfl = DFLoss(self.reg_max) if use_dfl else None

    def forward(self, pred_dist: torch.Tensor, pred_bboxes: torch.Tensor, anchor_points: torch.Tensor, 
                target_bboxes: torch.Tensor, target_scores: torch.Tensor, target_scores_sum: torch.Tensor, 
                fg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate bounding box regression loss (IoU + optional DFL).
        
        Args:
            pred_dist (torch.Tensor): Predicted distribution for bbox regression
                                     Shape: [batch_size, num_anchors, reg_max * 4]
                                     Contains distribution logits for l,t,r,b distances
            pred_bboxes (torch.Tensor): Predicted bounding boxes in xyxy format
                                       Shape: [batch_size, num_anchors, 4]
            anchor_points (torch.Tensor): Anchor center points
                                         Shape: [num_anchors, 2] (x, y coordinates)
            target_bboxes (torch.Tensor): Ground truth bounding boxes in xyxy format
                                         Shape: [batch_size, num_anchors, 4]
            target_scores (torch.Tensor): Target confidence scores for each anchor
                                         Shape: [batch_size, num_anchors, num_classes]
            target_scores_sum (torch.Tensor): Sum of all target scores for normalization
                                             Shape: scalar tensor
            fg_mask (torch.Tensor): Foreground mask indicating positive samples
                                   Shape: [batch_size, num_anchors] (boolean)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - loss_iou: IoU regression loss, scalar tensor
                - loss_dfl: DFL loss (if enabled) or zero tensor, scalar tensor
        """
        # Calculate weights for loss normalization using target scores
        # Only consider foreground (positive) samples for loss calculation
        # Shape: [batch_size, num_anchors, num_classes] -> [num_positive_samples, 1]
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # Calculate IoU between predicted and target bounding boxes
        # Only compute IoU for foreground samples to save computation
        # Shape: [num_positive_samples, 4] vs [num_positive_samples, 4] -> [num_positive_samples]
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        
        # Calculate IoU loss: (1 - IoU) weighted by target scores
        # Higher IoU means lower loss, weight by confidence scores
        # Shape: [num_positive_samples] * [num_positive_samples, 1] -> scalar
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # Calculate Distribution Focal Loss (DFL) if enabled
        if self.use_dfl:
            # Convert target bboxes to distance format (l,t,r,b) for DFL
            # Use reg_max - 1 to match official implementation
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            
            # Apply DFL to predicted distributions and target distances
            # Only compute for foreground samples, weight by target scores
            loss_dfl = self.dfl(pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]) * weight
            
            # Normalize DFL loss by sum of target scores
            # Shape: [num_positive_samples, 1] -> scalar
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            # If DFL is disabled, return zero loss on the same device
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class v8DetectionLoss:
    """
    YOLOv8 Detection Loss class for computing training losses.
    
    This class implements the complete loss computation pipeline for YOLOv8 object detection,
    including classification loss, bounding box regression loss, and distribution focal loss.
    
    The loss computation involves:
    1. Task-aligned assignment of targets to anchors
    2. Classification loss using BCE
    3. Bounding box regression loss using IoU + optional DFL
    """

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """
        Initialize v8DetectionLoss with the model, defining model-related properties and hyperparameters.
        
        Args:
            model: YOLOv8 detection model (must be de-paralleled, not wrapped in DataParallel)
            tal_topk (int): Top-k parameter for Task Aligned Assigner (default: 10)
        """
        # Get model device from model parameters
        device = next(model.parameters()).device  # get model device
        
        # Extract hyperparameters from model arguments
        h = model.args  # hyperparameters object containing loss weights

        # Get the detection head (last layer of the model)
        m = model.model[-1]  # Detect() module
        
        # Initialize binary cross entropy loss for classification
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
        # Store hyperparameters and model properties
        self.hyp = h  # Hyperparameters object
        self.stride = m.stride  # Model strides for different scales [8, 16, 32]
        self.nc = m.nc  # Number of classes (e.g., 80 for COCO)
        self.no = m.nc + m.reg_max * 4  # Number of outputs per anchor (classes + 4*reg_max for bbox)
        self.reg_max = m.reg_max  # Maximum regression range (typically 16)
        self.device = device  # Device for tensor operations

        # Determine if Distribution Focal Loss should be used
        self.use_dfl = m.reg_max > 1  # Use DFL if reg_max > 1

        # Initialize Task Aligned Assigner for target assignment
        # Assigns ground truth targets to appropriate anchor points
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        
        # Initialize bounding box loss module
        self.bbox_loss = BboxLoss(m.reg_max, use_dfl=self.use_dfl).to(device)
        
        # Create projection tensor for DFL (0, 1, 2, ..., reg_max-1)
        # Used to convert distribution to continuous values
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        Preprocess ground truth targets for loss computation.
        
        This method prepares the ground truth data by:
        1. Scaling bounding boxes to match feature map coordinates
        2. Organizing targets by image index
        3. Preparing data structures for the assigner
        
        Args:
            targets (Tensor): Ground truth targets with shape [N, 6] where N is total number of objects
                             Each row: [batch_idx, class_id, x_center, y_center, width, height]
            batch_size (int): Number of images in the batch
            scale_tensor (Tensor): Scaling factors for different feature levels, shape [3, 4]
                                  Used to scale bbox coordinates to feature map coordinates
        
        Returns:
            tuple: (targets_out, targets_out_scaled)
                - targets_out (list): List of tensors, one per image in batch
                                     Each tensor has shape [num_objects_in_image, 5] 
                                     Format: [class_id, x_center, y_center, width, height]
                - targets_out_scaled (list): Same as targets_out but with coordinates scaled
                                            by scale_tensor for multi-scale processing
        """
        # Handle empty targets case
        if targets.shape[0] == 0:  # No ground truth objects in batch
            # Return empty lists for each image in batch
            out = torch.zeros(batch_size, 0, 5, dtype=targets.dtype, device=targets.device)
            return [out[i] for i in range(batch_size)], [out[i] for i in range(batch_size)]

        # Extract image indices (which image each target belongs to)
        i = targets[:, 0]  # Shape: [N], batch indices for each target
        
        # Remove batch index and keep [class, x, y, w, h] format
        targets_out = targets[:, 1:6]  # Shape: [N, 5], remove batch index column
        
        # Group targets by image index
        # Create list of tensors, one for each image in the batch
        targets_list = []
        for img_idx in range(batch_size):
            # Find all targets belonging to current image
            mask = (i == img_idx)  # Boolean mask for current image
            img_targets = targets_out[mask]  # Extract targets for this image
            targets_list.append(img_targets)  # Shape: [num_objects_in_image, 5]
        
        # Create scaled version of targets for multi-scale processing
        targets_scaled_list = []
        for img_targets in targets_list:
            if img_targets.shape[0] > 0:  # If image has targets
                # Clone targets to avoid modifying original
                scaled_targets = img_targets.clone()  # Shape: [num_objects, 5]
                
                # Scale bounding box coordinates (x, y, w, h) using scale_tensor
                # scale_tensor typically contains scaling factors for different feature levels
                scaled_targets[:, 1:5] *= scale_tensor[0]  # Scale bbox coordinates
                targets_scaled_list.append(scaled_targets)
            else:
                # Empty tensor for images with no targets
                targets_scaled_list.append(img_targets)
        
        return targets_list, targets_scaled_list

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """
        Decode predicted distributions to bounding box coordinates.
        
        This method converts the predicted distribution over discrete bins to continuous
        bounding box coordinates using Distribution Focal Loss (DFL) approach.
        
        Args:
            anchor_points (Tensor): Anchor point coordinates with shape [num_anchors, 2]
                                   Each row contains [x_center, y_center] of anchor point
            pred_dist (Tensor): Predicted distributions with shape [batch_size, num_anchors, 4*reg_max]
                               Contains distribution over reg_max bins for each of 4 bbox coordinates
        
        Returns:
            Tensor: Decoded bounding boxes with shape [batch_size, num_anchors, 4]
                   Format: [x1, y1, x2, y2] in absolute coordinates
        """
        # Check if Distribution Focal Loss is enabled
        if self.use_dfl:
            # Reshape predictions to separate the 4 bbox coordinates and reg_max bins
            # From [batch_size, num_anchors, 4*reg_max] to [batch_size, num_anchors, 4, reg_max]
            b, a, c = pred_dist.shape  # batch_size, num_anchors, 4*reg_max
            pred_dist = pred_dist.view(b, a, 4, c // 4)  # Shape: [batch_size, num_anchors, 4, reg_max]
            
            # Apply softmax to convert logits to probability distributions
            pred_dist = pred_dist.softmax(3)  # Softmax over reg_max dimension
            
            # Convert distributions to continuous values using projection
            # self.proj contains [0, 1, 2, ..., reg_max-1]
            pred_dist = pred_dist.matmul(self.proj.type(pred_dist.dtype))  # Shape: [batch_size, num_anchors, 4]
            
            # Convert from distance predictions to absolute coordinates
            # pred_dist contains [left, top, right, bottom] distances from anchor points
            return dist2bbox(pred_dist, anchor_points, xywh=False)  # Returns [x1, y1, x2, y2]
        else:
            # If DFL is not used, directly convert distance predictions to bbox coordinates
            return dist2bbox(pred_dist, anchor_points, xywh=False)  # Shape: [batch_size, num_anchors, 4]

    def __call__(self, preds, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the total detection loss for YOLOv8.
        
        This method orchestrates the complete loss computation pipeline:
        1. Preprocess ground truth targets
        2. Assign targets to anchors using Task Aligned Assignment
        3. Compute classification loss (BCE)
        4. Compute bounding box regression loss (IoU + optional DFL)
        5. Combine losses with appropriate weights
        
        Args:
            preds (Tensor): Model predictions with shape [batch_size, num_anchors, num_outputs]
                           where num_outputs = num_classes + 4*reg_max
                           Contains both classification logits and bbox regression predictions
            batch (dict): Batch data containing:
                         - 'cls': Ground truth class labels
                         - 'bboxes': Ground truth bounding boxes
                         - 'batch_idx': Batch indices for each target
        
        Returns:
            Tuple[Tensor, Tensor]: (total_loss, loss_items)
                - total_loss: Scalar tensor with weighted sum of all losses
                - loss_items: Tensor with individual loss components [bbox_loss, cls_loss, dfl_loss]
        """
        # Initialize loss accumulator
        loss = torch.zeros(3, device=self.device)  # [bbox_loss, cls_loss, dfl_loss]
        
        # Extract feature maps and predictions from model output
        feats = preds[1] if isinstance(preds, tuple) else preds  # Feature maps from detection head
        
        # Get prediction tensor shape information
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)  # Split into bbox regression and classification
        
        # Reshape predictions for processing
        # pred_distri: [batch_size, 4*reg_max, num_anchors] -> [batch_size, num_anchors, 4*reg_max]
        # pred_scores: [batch_size, num_classes, num_anchors] -> [batch_size, num_anchors, num_classes]
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # Shape: [batch_size, num_anchors, num_classes]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # Shape: [batch_size, num_anchors, 4*reg_max]
        
        # Get tensor dimensions
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]  # Number of images in batch
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # Image size
        
        # Generate anchor points for all feature levels
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  # Anchor points and strides
        
        # Preprocess ground truth targets
        # Convert from [batch_idx, class, x, y, w, h] format to per-image lists
        # The targets variable is a tensor that contains the target labels and bounding boxes for the current batch of images.
        # In YOLO format: batch_idx, cls, and bboxes should all have the same number of objects
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1) #[N_objects, 6] [imageid, cls, boxes]
        targets_list, _ = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        
        # Convert list format back to tensor format for compatibility with assigner
        if len(targets_list) > 0 and any(len(t) > 0 for t in targets_list):
            max_targets = max(len(t) for t in targets_list)
            targets_tensor = torch.zeros(batch_size, max_targets, 5, device=self.device, dtype=dtype)
            for i, img_targets in enumerate(targets_list):
                if len(img_targets) > 0:
                    targets_tensor[i, :len(img_targets)] = img_targets
            gt_labels, gt_bboxes = targets_tensor.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        else:
            # No targets in batch
            gt_labels = torch.zeros(batch_size, 0, 1, device=self.device, dtype=dtype)
            gt_bboxes = torch.zeros(batch_size, 0, 4, device=self.device, dtype=dtype)
            mask_gt = torch.zeros(batch_size, 0, 1, device=self.device, dtype=torch.bool)
        
        # Decode predicted bounding boxes from distributions
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # Shape: [batch_size, num_anchors, 4]
        
        # Perform target assignment using Task Aligned Assigner
        # This assigns ground truth targets to the most appropriate anchor points
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),  # Detached classification predictions
            (pred_bboxes.detach() * stride_tensor.unsqueeze(0).unsqueeze(-1)).type(gt_bboxes.dtype),  # Scaled bbox predictions
            anchor_points * stride_tensor.unsqueeze(-1),  # Scaled anchor points
            gt_labels,  # Ground truth class labels
            gt_bboxes,  # Ground truth bounding boxes
            mask_gt  # Mask indicating which images have targets
        )
        
        # Compute classification loss using Binary Cross Entropy
        target_scores_sum = max(target_scores.sum(), 1)  # Normalize factor (avoid division by zero)
        
        # BCE loss for classification
        # target_scores contains soft assignment scores from the assigner
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        
        # Compute bounding box regression loss only for positive samples
        if fg_mask.sum():  # If there are positive anchor assignments
            # Normalize target boxes by stride
            target_bboxes /= stride_tensor
            
            # Compute bbox loss (IoU loss + optional DFL loss)
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
        
        # Apply loss weights from hyperparameters
        # Typically: box=7.5, cls=0.5, dfl=1.5 for YOLOv8
        loss[0] *= self.hyp.box  # Box loss weight
        loss[1] *= self.hyp.cls  # Classification loss weight
        loss[2] *= self.hyp.dfl  # DFL loss weight
        
        # Return loss components as a dictionary for easier interpretation
        return {
            'box_loss': loss[0],
            'cls_loss': loss[1], 
            'dfl_loss': loss[2],
            'total_loss': loss.sum() * batch_size
        }

class myv8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initialize myv8DetectionLoss with the model, defining model-related properties and hyperparameters."""
        device = next(model.parameters()).device  # get model device

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hypbox = 7.5  # (float) box loss gain
        self.hypcls = 0.5  # (float) cls loss gain (scale with pixels)
        self.hypdfl = 1.5  # (float) dfl loss gain
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4  # number of outputs per anchor
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        Preprocess ground truth targets for loss computation.
        
        This method prepares the ground truth data by:
        1. Scaling bounding boxes to match feature map coordinates
        2. Organizing targets by image index
        3. Preparing data structures for the assigner
        
        Args:
            targets (Tensor): Ground truth targets with shape [N, 6] where N is total number of objects
                             Each row: [batch_idx, class_id, x_center, y_center, width, height]
            batch_size (int): Number of images in the batch
            scale_tensor (Tensor): Scaling factors for different feature levels, shape [3, 4]
                                  Used to scale bbox coordinates to feature map coordinates
        
        Returns:
            tuple: (targets_out, targets_out_scaled)
                - targets_out (list): List of tensors, one per image in batch
                                     Each tensor has shape [num_objects_in_image, 5] 
                                     Format: [class_id, x_center, y_center, width, height]
                - targets_out_scaled (list): Same as targets_out but with coordinates scaled
                                            by scale_tensor for multi-scale processing
        """
        # Handle empty targets case
        if targets.shape[0] == 0:  # No ground truth objects in batch
            # Return empty lists for each image in batch
            out = torch.zeros(batch_size, 0, 5, dtype=targets.dtype, device=targets.device)
            return [out[i] for i in range(batch_size)], [out[i] for i in range(batch_size)]

        # Extract image indices (which image each target belongs to)
        i = targets[:, 0]  # Shape: [N], batch indices for each target
        
        # Remove batch index and keep [class, x, y, w, h] format
        targets_out = targets[:, 1:6]  # Shape: [N, 5], remove batch index column
        
        # Group targets by image index
        # Create list of tensors, one for each image in the batch
        targets_list = []
        for img_idx in range(batch_size):
            # Find all targets belonging to current image
            mask = (i == img_idx)  # Boolean mask for current image
            img_targets = targets_out[mask]  # Extract targets for this image
            targets_list.append(img_targets)  # Shape: [num_objects_in_image, 5]
        
        # Create scaled version of targets for multi-scale processing
        targets_scaled_list = []
        for img_targets in targets_list:
            if img_targets.shape[0] > 0:  # If image has targets
                # Clone targets to avoid modifying original
                scaled_targets = img_targets.clone()  # Shape: [num_objects, 5]
                
                # Scale bounding box coordinates (x, y, w, h) using scale_tensor
                # scale_tensor typically contains scaling factors for different feature levels
                scaled_targets[:, 1:5] *= scale_tensor[0]  # Scale bbox coordinates
                targets_scaled_list.append(scaled_targets)
            else:
                # Empty tensor for images with no targets
                targets_scaled_list.append(img_targets)
        
        return targets_list, targets_scaled_list

    def bbox_decode(self, anchor_points, pred_dist): #anchor_points:[8400, 2] pred_dist:[16, 8400, 64]
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        # print(f"bbox_decode - anchor_points shape: {anchor_points.shape}")
        # print(f"bbox_decode - pred_dist shape: {pred_dist.shape}")
        
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch 16, anchors 8400, channels 64
            # print(f"bbox_decode - b: {b}, a: {a}, c: {c}, c//4: {c//4}")
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
            
        return dist2bbox(pred_dist, anchor_points, xywh=False)#(ltrb) to box(xywh

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        
        # Debug: Check feature shapes
        # print(f"Feature shapes: {[f.shape for f in feats]}")
        # print(f"self.no: {self.no}, reg_max: {self.reg_max}, nc: {self.nc}")
        # print(f"Expected split: reg_max*4={self.reg_max * 4}, nc={self.nc}, total={self.reg_max * 4 + self.nc}")
        
        # Concatenate features and split into distribution and scores
        # Each feature map has shape [batch, no, height, width] where no = nc + reg_max * 4
        pred_concat = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2)
        # print(f"pred_concat shape: {pred_concat.shape}")
        pred_distri, pred_scores = pred_concat.split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        # print(f"targets shape: {targets.shape}")
        # print(f"targets content: {targets}")
        targets_list, _ = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        
        # Convert list format back to tensor format for compatibility with assigner
        if len(targets_list) > 0 and any(len(t) > 0 for t in targets_list):
            max_targets = max(len(t) for t in targets_list)
            targets_tensor = torch.zeros(batch_size, max_targets, 5, device=self.device, dtype=dtype)
            for i, img_targets in enumerate(targets_list):
                if len(img_targets) > 0:
                    targets_tensor[i, :len(img_targets)] = img_targets
            gt_labels, gt_bboxes = targets_tensor.split((1, 4), 2)  # cls, xyxy
            
            # Ensure gt_labels are within valid range [0, num_classes-1]
            # Clamp labels to valid range to prevent CUDA index errors
            gt_labels = torch.clamp(gt_labels, 0, self.nc - 1)
            
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        else:
            # No targets in batch
            gt_labels = torch.zeros(batch_size, 0, 1, device=self.device, dtype=dtype)
            gt_bboxes = torch.zeros(batch_size, 0, 4, device=self.device, dtype=dtype)
            mask_gt = torch.zeros(batch_size, 0, 1, device=self.device, dtype=torch.bool)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # Scale anchor points to pixel coordinates to match GT boxes coordinate system
        # anchor_points are in feature map coordinates (e.g., 0.5-79.5), need to convert to pixel coordinates
        anchor_points_scaled = anchor_points * stride_tensor.unsqueeze(-1)
        
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor.unsqueeze(0).unsqueeze(-1)).type(gt_bboxes.dtype),
            anchor_points_scaled, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hypbox  # box gain
        loss[1] *= self.hypcls  # cls gain
        loss[2] *= self.hypdfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    

#https://github.com/lkk688/yolov5/blob/master/utils/loss.py
