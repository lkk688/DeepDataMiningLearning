import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from torchvision.transforms import functional as F
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import box_iou

import os
import json
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

#pip install pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from modeling_fcos import FCOS, ResNet50Backbone, FCOSLoss

class CocoDetection(torch.utils.data.Dataset):
    """
    COCO-formatted dataset loader.
    
    Args:
        root: Path to the root directory of images
        ann_file: Path to the annotation file
        transforms: Optional transform function
    """
    def __init__(self, root, ann_file, transforms=None):
        self.root = root
        self.transforms = transforms
        
        # Load annotations
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        
        # Create image ID to annotations mapping
        self.img_ids = [img['id'] for img in self.coco['images']]
        self.img_info = {img['id']: img for img in self.coco['images']}
        
        # Create annotation index
        self.anns = defaultdict(list)
        for ann in self.coco['annotations']:
            self.anns[ann['image_id']].append(ann)
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.img_info[img_id]
        
        # Load image
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        anns = self.anns[img_id]
        
        # Extract boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            # Skip small objects (area < 1)
            if ann['area'] <= 1:
                continue
            
            # COCO boxes are [x,y,width,height]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img_id])
        
        # Apply transforms if any
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

class FCOSTransform:
    """
    Transforms for FCOS training:
    - Resize image (shorter side to 800, longer side <= 1333)
    - Normalize with ImageNet stats
    - Convert boxes to FCOS format
    """
    def __init__(self, min_size=800, max_size=1333):
        self.min_size = min_size
        self.max_size = max_size
        
        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def __call__(self, image, target):
        # Original dimensions
        orig_width, orig_height = image.size
        
        # Resize image
        scale = self.min_size / min(orig_width, orig_height)
        if max(orig_width, orig_height) * scale > self.max_size:
            scale = self.max_size / max(orig_width, orig_height)
        
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        image = F.resize(image, (new_height, new_width))
        
        # Resize boxes
        if 'boxes' in target:
            boxes = target['boxes']
            boxes = boxes * scale
            target['boxes'] = boxes
        
        # Convert to tensor and normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        
        return image, target

def collate_fn(batch):
    """
    Collate function for DataLoader to handle variable numbers of objects.
    """
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Train the model for one epoch.
    """
    model.train()
    loss_fn = FCOSLoss(num_classes=81)  # COCO has 80 classes + background
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # Move images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        optimizer.zero_grad()
        raw_outputs, _ = model(images)
        
        # Compute loss
        losses = loss_fn(raw_outputs, targets)
        total_loss = losses['total_loss']
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Logging
        metric_logger.update(loss=total_loss.item(), **losses)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    return metric_logger

@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    loss_fn = FCOSLoss(num_classes=81)
    
    val_metrics = defaultdict(float)
    num_batches = len(data_loader)
    
    for images, targets in tqdm(data_loader, desc="Validation"):
        # Move images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        raw_outputs, detections = model(images)
        
        # Compute loss
        losses = loss_fn(raw_outputs, targets)
        
        # Accumulate metrics
        for k, v in losses.items():
            val_metrics[k] += v.item() / num_batches
        
        # Compute mAP (simplified version)
        val_metrics['mAP'] += compute_map(detections, targets) / num_batches
    
    return val_metrics

def compute_map(detections, targets, iou_threshold=0.5):
    """
    Simplified mAP computation for COCO.
    """
    aps = []
    
    for dets, gts in zip(detections, targets):
        if len(dets['boxes']) == 0:
            aps.append(0.0)
            continue
        
        # Sort detections by score
        scores, idxs = dets['scores'].sort(descending=True)
        boxes = dets['boxes'][idxs]
        labels = dets['labels'][idxs]
        
        # Get ground truth boxes and labels
        gt_boxes = gts['boxes']
        gt_labels = gts['labels']
        
        # Compute IoU
        ious = box_iou(boxes, gt_boxes)
        
        # Match detections to ground truth
        matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
        tp = torch.zeros(len(boxes))
        fp = torch.zeros(len(boxes))
        
        for i in range(len(boxes)):
            # Find best matching ground truth box
            iou = ious[i]
            best_iou, best_j = iou.max(dim=0)
            
            if best_iou >= iou_threshold and not matched[best_j] and labels[i] == gt_labels[best_j]:
                matched[best_j] = True
                tp[i] = 1
            else:
                fp[i] = 1
        
        # Compute precision-recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        
        recalls = tp_cumsum / len(gt_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Compute AP (area under precision-recall curve)
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
    
    return torch.mean(torch.tensor(aps)).item()

def compute_ap(recalls, precisions):
    """
    Compute the average precision given precision and recall.
    """
    # Append sentinel values
    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([1.0]), precisions, torch.tensor([0.0])])
    
    # Compute the precision envelope
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = torch.max(precisions[i - 1], precisions[i])
    
    # Find indices where recall changes
    change_points = torch.where(recalls[1:] != recalls[:-1])[0]
    
    # Sum areas
    ap = torch.sum((recalls[change_points + 1] - recalls[change_points]) * 
                   precisions[change_points + 1])
    
    return ap

class MetricLogger:
    """Utility class for logging metrics during training."""
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)
    
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            
            if i % print_freq == 0 or i == len(iterable) - 1:
                print(header + 
                      f'[{i:{space_fmt}}/{len(iterable):{space_fmt}}]' +
                      f'   eta: {iter_time.avg * (len(iterable) - i):.0f}s' +
                      f'   time: {iter_time.avg:.4f}' +
                      f'   data: {data_time.avg:.4f}' +
                      f'   {str(self)}')
            i += 1
            end = time.time()
        
        total_time = time.time() - start_time
        print(f'{header} Total time: {total_time:.0f}s')

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a window."""
    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or '{median:.4f}'
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.total += value * n
        self.count += n
    
    @property
    def median(self):
        return np.median(self.deque)
    
    @property
    def avg(self):
        return self.total / self.count
    
    @property
    def global_avg(self):
        return self.avg
    
    @property
    def max(self):
        return max(self.deque)
    
    @property
    def value(self):
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

def evaluate_coco(model, data_loader, device, ann_file):
    """
    Official COCO evaluation using COCO API.
    
    Args:
        model: FCOS model
        data_loader: DataLoader for validation set
        device: Device to run evaluation on
        ann_file: Path to COCO annotation file
        
    Returns:
        Dictionary containing COCO evaluation metrics
    """
    model.eval()
    results = []
    ids = []
    
    # Initialize COCO ground truth API
    coco_gt = COCO(ann_file)
    
    for images, targets in tqdm(data_loader, desc="COCO Evaluation"):
        # Move images to device
        images = list(image.to(device) for image in images)
        
        # Forward pass
        with torch.no_grad():
            _, detections = model(images)
        
        # Process detections
        for det, target in zip(detections, targets):
            image_id = target["image_id"].item()
            ids.append(image_id)
            
            # Convert boxes to COCO format [x,y,width,height]
            boxes = det["boxes"].cpu().numpy()
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
            
            # Convert to list of dicts in COCO format
            for box, score, label in zip(boxes, 
                                       det["scores"].cpu().numpy(), 
                                       det["labels"].cpu().numpy()):
                results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x) for x in box],
                    "score": float(score)
                })
    
    # Only evaluate on images that had detections
    if len(results) == 0:
        print("No detections to evaluate!")
        return None
    
    # Load results into COCO API
    coco_dt = coco_gt.loadRes(results)
    
    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Filter images to only those that had detections
    coco_eval.params.imgIds = ids
    
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Collect metrics
    metrics = {
        'AP': coco_eval.stats[0],  # AP @ [0.5:0.95]
        'AP50': coco_eval.stats[1],  # AP @ 0.5
        'AP75': coco_eval.stats[2],  # AP @ 0.75
        'AP_small': coco_eval.stats[3],  # AP for small objects
        'AP_medium': coco_eval.stats[4],  # AP for medium objects
        'AP_large': coco_eval.stats[5],  # AP for large objects
        'AR1': coco_eval.stats[6],  # AR @ max_dets=1
        'AR10': coco_eval.stats[7],  # AR @ max_dets=10
        'AR100': coco_eval.stats[8],  # AR @ max_dets=100
        'AR_small': coco_eval.stats[9],  # AR for small objects
        'AR_medium': coco_eval.stats[10],  # AR for medium objects
        'AR_large': coco_eval.stats[11],  # AR for large objects
    }
    
    return metrics

def main():
    # Training parameters
    data_dir = 'data/COCOoriginal'  # Directory with 'train2017', 'val2017', 'annotations'
    train_ann_file = os.path.join(data_dir, 'annotations/instances_train2017.json')
    val_ann_file = os.path.join(data_dir, 'annotations/instances_val2017.json')
    train_img_dir = os.path.join(data_dir, 'train2017')
    val_img_dir = os.path.join(data_dir, 'val2017')
    
    batch_size = 4
    num_workers = 4
    num_epochs = 12
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    lr_step = [8, 11]  # Epochs to decrease learning rate
    
    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Create datasets
    train_transform = FCOSTransform()
    val_transform = FCOSTransform()
    
    train_dataset = CocoDetection(train_img_dir, train_ann_file, train_transform)
    val_dataset = CocoDetection(val_img_dir, val_ann_file, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn)
    
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn)
    
    # Initialize model
    backbone = ResNet50Backbone()
    model = FCOS(backbone, num_classes=81)  # COCO has 80 classes + background
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_step, gamma=0.1)
    
    # Training loop
    for epoch in range(num_epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device)
        
        # COCO evaluation (run less frequently to save time)
        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            coco_metrics = evaluate_coco(model, val_loader, device, val_ann_file)
        else:
            coco_metrics = None
        
        # Print metrics
        print(f"Epoch {epoch} - Train Loss: {train_metrics.loss.global_avg:.4f}")
        print(f"Epoch {epoch} - Val Loss: {val_metrics['total_loss']:.4f} mAP: {val_metrics['mAP']:.4f}")
        
        if coco_metrics:
            print("COCO Evaluation Metrics:")
            print(f"AP @ [0.5:0.95]: {coco_metrics['AP']:.4f}")
            print(f"AP50: {coco_metrics['AP50']:.4f}")
            print(f"AP75: {coco_metrics['AP75']:.4f}")
            print(f"AP small: {coco_metrics['AP_small']:.4f}")
            print(f"AP medium: {coco_metrics['AP_medium']:.4f}")
            print(f"AP large: {coco_metrics['AP_large']:.4f}")
            
        # Save checkpoint
        if (epoch + 1) % 4 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics.loss.global_avg,
                'val_loss': val_metrics['total_loss'],
                'val_map': val_metrics['mAP'],
                'coco_metrics': coco_metrics if coco_metrics else None
            }
            torch.save(checkpoint, f'fcos_checkpoint_epoch{epoch}.pth')

if __name__ == '__main__':
    from collections import deque
    main()