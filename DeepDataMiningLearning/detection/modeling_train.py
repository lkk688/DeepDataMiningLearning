import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import torchvision.transforms as T
from torchvision.ops import batched_nms
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Suppress warnings (optional)
import warnings
warnings.filterwarnings("ignore")

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CocoDetection(torch.utils.data.Dataset):
    """Loads COCO-format dataset with annotations."""
    def __init__(self, root, ann_file, transform=None):
        self.root = root
        self.transform = transform
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        self.img_ids = [img['id'] for img in self.coco['images']]
        self.img_info = {img['id']: img for img in self.coco['images']}
        self.anns = defaultdict(list)
        for ann in self.coco['annotations']:
            self.anns[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.img_info[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        # Extract boxes and labels
        boxes, labels = [], []
        for ann in self.anns[img_id]:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([img_id])}
        
        if self.transform:
            img, target = self.transform(img, target)
        return img, target

def collate_fn(batch):
    """Custom collate function for variable-sized boxes."""
    return tuple(zip(*batch))

def get_model(model_name, num_classes=80):
    """Returns the selected model."""
    if model_name == "yolov8":
        from modeling_yolov8 import YOLOv8
        return YOLOv8(num_classes=num_classes).to(device)
    # elif model_name == "yolov12":
    #     from yolov12_model import YOLOv12
    #     return YOLOv12(num_classes=num_classes).to(device)
    elif model_name == "fcos":
        from modeling_fcos import FCOS, ResNet50Backbone
        backbone = ResNet50Backbone().to(device)
        return FCOS(backbone, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, optimizer, data_loader, loss_fn, epoch, print_freq=10):
    """Trains the model for one epoch."""
    model.train()
    metric_logger = MetricLogger()
    header = f'Epoch [{epoch}]'
    
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        outputs = model(images)
        losses = loss_fn(outputs, targets)
        losses['total_loss'].backward()
        optimizer.step()
        
        metric_logger.update(**losses)
    
    return metric_logger

def evaluate(model, data_loader, loss_fn):
    """Evaluates model on validation set."""
    model.eval()
    val_metrics = defaultdict(float)
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validation"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            losses = loss_fn(outputs, targets)
            
            for k, v in losses.items():
                val_metrics[k] += v.item() / num_batches
    
    return val_metrics

def get_loss_fn(model_name):
    """Returns the appropriate loss function."""
    if model_name in ["yolov8", "yolov12"]:
        from yolov8_model import YOLOv8Loss
        return YOLOv8Loss(num_classes=80)
    elif model_name == "fcos":
        from fcos_model import FCOSLoss
        return FCOSLoss(num_classes=80)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8/YOLOv12/FCOS")
    parser.add_argument("--model", type=str, default="yolov8", choices=["yolov8", "yolov12", "fcos"])
    parser.add_argument("--data_dir", type=str, default='data/COCOoriginal')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    # Load datasets
    train_dataset = CocoDetection(
        root=os.path.join(args.data_dir, "train2017"),
        ann_file=os.path.join(args.data_dir, "annotations/instances_train2017.json"),
        transform=T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    )
    val_dataset = CocoDetection(
        root=os.path.join(args.data_dir, "val2017"),
        ann_file=os.path.join(args.data_dir, "annotations/instances_val2017.json"),
        transform=T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Initialize model & loss
    model = get_model(args.model)
    loss_fn = get_loss_fn(args.model)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, optimizer, train_loader, loss_fn, epoch)
        val_metrics = evaluate(model, val_loader, loss_fn)
        scheduler.step()

        print(f"Epoch {epoch} | Train Loss: {train_metrics.loss.global_avg:.4f} | Val Loss: {val_metrics['total_loss']:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics.loss.global_avg,
                'val_loss': val_metrics['total_loss'],
            }, f"{args.model}_checkpoint_epoch{epoch}.pth")

if __name__ == "__main__":
    main()