import torch
import torch.nn as nn
import torch.nn.functional as F
from DeepDataMiningLearning.detection.modules.tal import TaskAlignedAssigner, dist2bbox, make_anchors, bbox2dist
from DeepDataMiningLearning.detection.modules.metrics import bbox_iou
from DeepDataMiningLearning.detection.modules.utils import xywh2xyxy

#ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py

class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max #15
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        #The weight variable is used to weight the loss for each bounding box.
        #computed as the sum of the target scores for all of the foreground bounding boxes
        #The unsqueeze(-1) operation is used to add a new dimension to the weight variable. This is necessary because the weight variable is a 1-dimensional tensor, 
        #but the loss function expects a 2-dimensional tensor.
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1) #[16, 8400, 80]->[689, 1], K position from foreground
        #weight variable is a 2-dimensional tensor with the same shape as the target_scores variable
        #The weight variable is then used to weight the loss for each bounding box. 
        #This ensures that the loss is higher for bounding boxes that contain objects and lower for bounding boxes that do not contain objects.

        #computes the IoU between each predicted bounding box and each target bounding box.
        #calculate pred_bboxes: box1(1, 4) target_bboxes: to box2(n, 4)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True) #[689, 1]
        #The output of the iou variable is a tensor of shape [num_foreground_bounding_boxes, 1]. This tensor is used to compute the bounding box loss.

        #The weight variable is used to weight the loss for each bounding box. 
        #The target_scores_sum variable is the sum of the target scores for all of the foreground bounding boxes
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum #0.1373

        # DFL loss
        if self.use_dfl:
            #The target_ltrb variable is the target distribution of the bounding boxes.
            #converts the target bounding boxes from the xyxy format to the ltrb format. 
            #The ltrb format is a representation of the bounding box in terms of the left, top, right, and bottom coordinates.
            #The target_ltrb variable is used to compute the distribution focal loss (DFL) loss.
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max) #[16, 8400, 4]
            #The target_ltrb variable is a 3-dimensional tensor with the shape [batch_size, num_anchors, 4]. 
            #the third dimension corresponds to the four coordinates of the bounding box (left, top, right, bottom).

            #The DFL loss is a variant of the focal loss that is designed to address the problem of class imbalance in object detection. 
            #The DFL loss assigns a higher weight to the loss for bounding boxes that are far from the anchor points. 
            #This helps to ensure that the model learns to predict accurate bounding boxes for all objects, regardless of their size or location.
            #The weight variable is used to weight the loss for each bounding box.
            #The view(-1, self.reg_max + 1) operation reshapes the pred_dist variable to a 2-dimensional tensor with self.reg_max + 1 columns.
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight #[689, 1]
            #output 1-dimensional tensor with the DFL loss for all of the foreground bounding boxes.

            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target): #pred_dist:[2756(4K), 16] target:[689(K), 4] 689*4=2756
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # round the low Int (10.8->10) target left [689, 4] center to top left distance
        tr = tl + 1  # target right
        wl = tr - target  # weight left, displacement as the weight l (0.2)
        wr = 1 - wl  # weight right [689, 4], displacement as the weight r 0.8
        
        #computes the DFL loss for a single bounding box. 
        #The tl variable is the target left coordinate of the bounding box, and the tr variable is the target right coordinate of the bounding box. 
        #The wl variable is the weight for the left coordinate, and the wr variable is the weight for the right coordinate.
        #The F.cross_entropy function computes the cross-entropy loss between the predicted distribution and the target distribution. 
        #The view(-1) operation reshapes the predicted distribution and the target distribution to be one-dimensional tensors.
        #reduction='none' return a tensor of the same size as the input tensor
        #The view(tl.shape) operation reshapes the loss tensor to be the same size as the tl and tr tensors. 
        #The * wl and * wr operations multiply the loss tensor by the weight tensors.
        #The mean(-1, keepdim=True) operation computes the mean of the loss tensor along the last dimension, and keeps the dimension. 
        #This gives the average loss for the bounding box.
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

class myv8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        #h = model.args  # hyperparameters

        m = model.model[-1] #model.model[-1]  # Detect() module

        #measures the binary cross entropy between the target and the predicted logits
        #reduction='none' means that the loss is not reduced, and the output is a tensor of the same size as the input.
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        #self.hyp = hyp #h
        self.hypbox = 7.5  # (float) box loss gain
        self.hypcls = 0.5  # (float) cls loss gain (scale with pixels)
        self.hypdfl = 1.5  # (float) dfl loss gain
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        #The reg_max parameter specifies the maximum number of regression targets. 
        #This parameter is used to determine the size of the predicted bounding box distribution.
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        #assign ground truth boxes to anchor boxes. The topk parameter specifies the number of top scoring anchor boxes to assign to each ground truth box
        #The alpha and beta parameters are used to control the balance between the classification loss and the localization loss.
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        
        #custom loss function for bounding box regression. It takes as input the predicted bounding box distribution, the predicted bounding boxes, the anchor points, the target bounding boxes, the target scores, the target scores sum, and the foreground mask. 
        #It returns the bounding box loss and the distribution focal loss (DFL) loss.
        #The DFL loss is a variant of the focal loss that is designed to address the problem of class imbalance in object detection.
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)

        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device) #16

    def preprocess(self, targets, batch_size, scale_tensor): #targets is [batch_idx, cls, bbox]
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index, i.e, batch_idx
            _, counts = i.unique(return_counts=True) #total count of images
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device) #out(b, object counts, 5) [16, 17, 5]
            for j in range(batch_size):
                matches = i == j
                n = matches.sum() #n number of objects in j-th image
                if n:
                    out[j, :n] = targets[matches, 1:] #copy the targets (cls, bbox) to out[j-th image, 0:n objects, 5(cls+box)]
            #normalized xcenter, ycenter, w, h to unnormalized (x1, y1, x2, y2)
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor)) #box*scale to xyxy, scale_tensor=tensor([640., 640., 640., 640.]
        return out #[16, 17, 5]

    def bbox_decode(self, anchor_points, pred_dist): #anchor_points:[8400, 2] pred_dist:[16, 8400, 64]
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch 16, anchors 8400, channels 64
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        #The preds tuple is a collection of feature maps that are output by the model. 
        #The second element [1] of the tuple is typically the feature map that is used for object detection.
        feats = preds[1] if isinstance(preds, tuple) else preds #preds=feats=[[16, 144, 80, 80], [16, 144, 40, 40], [16, 144, 20, 20]]
        
        #takes as input a list of feature maps feats and splits them into two tensors: pred_distri and pred_scores.
        #The view function is used to reshape the feature maps so that they have the same shape. 
        #The torch.cat function is used to concatenate the feature maps along the second dimension.
        #The split function is used to split the concatenated feature maps into two tensors.
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1) 
        #each x[i] become [b, no(144), 80*80] [b, no(144), 40*40] [b, no(144), 20*20], then cat to [b, no(144), 8400]
        #The pred_distri tensor [16, 64(reg_max=16*4), 8400] contains the predicted distribution of the bounding boxes for each anchor box. 
        #The pred_scores tensor [16, 80(nc), 8400] contains the predicted scores for each anchor box.

        pred_scores = pred_scores.permute(0, 2, 1).contiguous() #->[16, 8400, 80]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous() #->[16, 8400, 64]

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0] #16
        #The imgsz variable contains the image size in pixels: computed by multiplying the shape of the first feature map in the feats list by the stride of the model.
        #(80,80)* stride[0]=8 => (640, 640)
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w) stride=[8., 16., 32] [640, 640]
        
        #generate anchor boxes for a given set of feature maps
        #The feats parameter is a list of feature maps that are output by the model. 
        #The stride parameter is a list of strides that are used to downsample the feature maps. 
        #The offset parameter is a value that is added to the anchor boxes to offset them from the center of the grid cells.
        #in tal.py file: first creates a grid of points that are evenly spaced across the feature maps. 
        #The grid points are then used to generate the anchor boxes. The anchor boxes are generated by scaling and translating the grid points by a set of predefined scales and aspect ratios.
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5) #[8400, 2], [8400, 1]
        #The anchor_points variable is a tensor that contains the anchor points. 
        #The stride_tensor variable is a tensor that contains the strides that are used to downsample the feature maps.

        # The targets variable is a tensor that contains the target labels and bounding boxes for the current batch of images.
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1) #[87, 6] [imageid, cls, boxes]
        #The function takes as input the target labels and bounding boxes, the batch size, and the scale tensor. The scale tensor is used to normalize the target bounding boxes to the same size as the input images.
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        #The function then reshapes the tensor so that it has the shape [batch_size, max_num_objects, 5], where max_num_objects is the maximum number of objects in any image in the batch. The function then fills in the missing values in the tensor with zeros.
        #returns the preprocessed target labels and bounding boxes: targets[j-th image, 0:n objects, 5(cls+box)]

        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # [16, 17, 1] cls, xyxy [16, 17, 4]
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0) #[16, 17, 1], among 17 data, mask=1 means contain object, otherwise mask=0

        # pboxes
        #decode the predicted bounding box distribution into bounding boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4) ->[16, 8400, 4]
        #The output of the bbox_decode function is a tensor of predicted bounding boxes. The shape of the tensor is [batch_size, num_anchors, 4]

        #assign ground truth bounding boxes to predicted bounding boxes. The goal of this assignment is to match each predicted bounding box to the ground truth bounding box that it is most similar to. 
        #This information is then used to calculate the loss function.
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt) #target_bboxes[16, 8400, 4], target_scores[16, 8400, 80], fg_mask[16, 8400]

        target_scores_sum = max(target_scores.sum(), 1) #419

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE [16, 8400, 80]

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor #[16, 8400, 4]/stride_tensor[8400, 1] (8,8, .. 32)
            #pred_distri: predicted distribution of bounding boxes
            #the predicted bounding boxes (pred_bboxes), the anchor points (anchor_points), the target bounding boxes (target_bboxes), 
            #the target scores (target_scores), the target scores sum (target_scores_sum), and the foreground mask (fg_mask).
            #BboxLoss: Bbox loss (DFL Loss + CIOU Loss)
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hypbox #self.hyp.box  # box gain
        loss[1] *= self.hypcls #self.hyp.cls  # cls gain
        loss[2] *= self.hypdfl #self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    

#https://github.com/lkk688/yolov5/blob/master/utils/loss.py
