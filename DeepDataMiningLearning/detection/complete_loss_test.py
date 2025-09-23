#!/usr/bin/env python3
"""
Complete loss calculation test for YOLOv8 implementation
"""
import torch
import torch.nn as nn
import numpy as np
from modules.tal import TaskAlignedAssigner
from modules.lossv8 import v8DetectionLoss, DFLoss

class MockArgs:
    def __init__(self):
        self.box = 7.5
        self.cls = 0.5
        self.dfl = 1.5

class MockDetectModule(nn.Module):
    def __init__(self, nc=80, reg_max=16):
        super().__init__()
        self.nc = nc  # number of classes
        self.reg_max = reg_max  # regression max
        self.stride = torch.tensor([8., 16., 32.])  # strides for different scales

class MockModel(nn.Module):
    """Mock model for testing loss calculation"""
    def __init__(self, num_classes=80, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.args = MockArgs()
        # Create dummy parameters to satisfy device detection
        self.dummy_param = nn.Parameter(torch.zeros(1))
        # Create the model structure that v8DetectionLoss expects
        self.model = nn.ModuleList([
            nn.Conv2d(3, 64, 3),
            nn.Conv2d(64, 128, 3),
            MockDetectModule(nc=num_classes, reg_max=reg_max)  # Detection head
        ])

def create_test_data(batch_size=2, num_classes=80, device='cpu'):
    """Create realistic test data for loss calculation"""
    
    # Create multi-scale feature maps (P3, P4, P5)
    strides = [8, 16, 32]
    img_size = 640
    
    # Calculate grid sizes for each scale
    grid_sizes = [img_size // stride for stride in strides]
    total_anchors = sum(gs * gs for gs in grid_sizes)
    
    print(f"Grid sizes: {grid_sizes}")
    print(f"Total anchors: {total_anchors}")
    
    # Create predictions
    pred_scores = torch.randn(batch_size, total_anchors, num_classes, device=device)
    pred_bboxes = torch.randn(batch_size, total_anchors, 4, device=device)
    pred_dists = torch.randn(batch_size, total_anchors, 16 * 4, device=device)  # reg_max * 4
    
    # Create anchor points
    anchor_points = []
    stride_tensor = []
    
    for i, (gs, stride) in enumerate(zip(grid_sizes, strides)):
        # Create grid
        yv, xv = torch.meshgrid(torch.arange(gs), torch.arange(gs), indexing='ij')
        grid = torch.stack([xv, yv], dim=-1).float()
        
        # Convert to pixel coordinates
        grid = (grid + 0.5) * stride
        anchor_points.append(grid.reshape(-1, 2))
        stride_tensor.extend([stride] * (gs * gs))
    
    anchor_points = torch.cat(anchor_points, dim=0).to(device)
    stride_tensor = torch.tensor(stride_tensor, device=device).float()
    
    print(f"Anchor points shape: {anchor_points.shape}")
    print(f"Stride tensor shape: {stride_tensor.shape}")
    
    # Create ground truth data
    max_objects = 3
    gt_labels = torch.zeros(batch_size, max_objects, 1, device=device)
    gt_bboxes = torch.zeros(batch_size, max_objects, 4, device=device)
    mask_gt = torch.zeros(batch_size, max_objects, 1, device=device)
    
    # Add some realistic objects
    for b in range(batch_size):
        num_objects = np.random.randint(1, max_objects + 1)
        
        for obj_idx in range(num_objects):
            # Random class
            gt_labels[b, obj_idx, 0] = np.random.randint(0, num_classes)
            
            # Random bbox in xyxy format
            x1 = np.random.uniform(50, img_size - 150)
            y1 = np.random.uniform(50, img_size - 150)
            x2 = x1 + np.random.uniform(50, 100)
            y2 = y1 + np.random.uniform(50, 100)
            
            gt_bboxes[b, obj_idx] = torch.tensor([x1, y1, x2, y2])
            mask_gt[b, obj_idx, 0] = 1.0
    
    print(f"GT labels shape: {gt_labels.shape}")
    print(f"GT bboxes shape: {gt_bboxes.shape}")
    print(f"Valid objects per batch: {mask_gt.sum(dim=1).squeeze()}")
    
    return {
        'pred_scores': pred_scores,
        'pred_bboxes': pred_bboxes,
        'pred_dists': pred_dists,
        'anchor_points': anchor_points,
        'stride_tensor': stride_tensor,
        'gt_labels': gt_labels,
        'gt_bboxes': gt_bboxes,
        'mask_gt': mask_gt
    }

def test_tal_assignment(data):
    """Test TAL assignment with realistic data"""
    print("\n" + "="*60)
    print("Testing TAL Assignment")
    print("="*60)
    
    tal_assigner = TaskAlignedAssigner(topk=10, num_classes=80, alpha=1.0, beta=6.0)
    
    try:
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = tal_assigner(
            data['pred_scores'].sigmoid(),
            data['pred_bboxes'],
            data['anchor_points'],
            data['gt_labels'],
            data['gt_bboxes'],
            data['mask_gt']
        )
        
        print(f"✅ TAL assignment successful!")
        print(f"Target labels shape: {target_labels.shape}")
        print(f"Target bboxes shape: {target_bboxes.shape}")
        print(f"Target scores shape: {target_scores.shape}")
        print(f"Foreground mask shape: {fg_mask.shape}")
        print(f"Positive samples per batch: {fg_mask.sum(dim=1)}")
        print(f"Total positive samples: {fg_mask.sum().item()}")
        
        return {
            'target_labels': target_labels,
            'target_bboxes': target_bboxes,
            'target_scores': target_scores,
            'fg_mask': fg_mask,
            'target_gt_idx': target_gt_idx
        }
        
    except Exception as e:
        print(f"❌ TAL assignment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_loss_calculation():
    """Test the complete loss calculation pipeline"""
    print("\n" + "="*60)
    print("Testing Loss Calculation")
    print("="*60)
    
    try:
        # Create test data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create mock model and loss function
        mock_model = MockModel(num_classes=80, reg_max=16)
        mock_model.to(device)
        loss_fn = v8DetectionLoss(mock_model, tal_topk=10)
        
        # Prepare targets in the format expected by v8DetectionLoss
        # The loss function expects a dictionary with 'batch_idx', 'cls', and 'bboxes' keys
        batch_indices = []
        classes = []
        bboxes = []
        
        # First batch has 3 objects
        batch_indices.extend([0, 0, 0])  # All belong to batch 0
        classes.extend([0, 1, 2])  # Different classes
        bboxes.extend([
            [0.3, 0.3, 0.4, 0.4],  # x1, y1, x2, y2 (normalized)
            [0.1, 0.1, 0.3, 0.3],
            [0.6, 0.6, 0.9, 0.9]
        ])
        
        # Second batch has 1 object
        batch_indices.extend([1])  # Belongs to batch 1
        classes.extend([1])
        bboxes.extend([[0.2, 0.2, 0.5, 0.5]])
        
        # Create the batch dictionary
        batch = {
            'batch_idx': torch.tensor(batch_indices, device=device, dtype=torch.float32),
            'cls': torch.tensor(classes, device=device, dtype=torch.float32),
            'bboxes': torch.tensor(bboxes, device=device, dtype=torch.float32)
        }
        
        print(f"Prepared batch dictionary with {len(batch_indices)} total objects")
        print(f"  Batch indices: {batch['batch_idx']}")
        print(f"  Classes: {batch['cls']}")
        print(f"  Bboxes shape: {batch['bboxes'].shape}")
        
        # Create predictions in the format expected by v8DetectionLoss
        # The loss function expects feature maps from different scales
        # For simplicity, we'll create 3 feature maps for 3 different scales
        batch_size = 2  # Define batch_size
        num_classes = 80
        reg_max = 16
        
        # Feature map sizes for different scales (typical YOLO setup)
        feat_sizes = [(80, 80), (40, 40), (20, 20)]  # H, W for each scale
        
        feats = []
        for h, w in feat_sizes:
            # Each feature map has (num_classes + 4*reg_max) channels
            num_outputs = num_classes + 4 * reg_max  # 80 + 64 = 144
            feat = torch.randn(batch_size, num_outputs, h, w, device=device)
            feats.append(feat)
        
        print(f"Feature maps shapes: {[f.shape for f in feats]}")
        
        # Calculate losses
        loss_dict = loss_fn(feats, batch)
        
        print(f"✅ Loss calculation successful!")
        print(f"Loss components:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f}")
            else:
                print(f"  {key}: {value}")
        
        total_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
        print(f"Total loss: {total_loss.item():.6f}")
        
        return loss_dict
        
    except Exception as e:
        print(f"❌ Loss calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_bbox2dist_function():
    """Test the bbox2dist function specifically"""
    print("\n" + "="*60)
    print("Testing bbox2dist Function")
    print("="*60)
    
    from modules.tal import bbox2dist
    
    # Create test data
    anchor_points = torch.tensor([[100.0, 100.0], [200.0, 200.0]])
    bboxes = torch.tensor([[80.0, 80.0, 120.0, 120.0], [180.0, 180.0, 220.0, 220.0]])  # xyxy format
    reg_max = 16
    
    print(f"Anchor points: {anchor_points}")
    print(f"Bboxes (xyxy): {bboxes}")
    print(f"Reg max: {reg_max}")
    
    try:
        distances = bbox2dist(anchor_points, bboxes, reg_max)
        print(f"✅ bbox2dist calculation successful!")
        print(f"Distances shape: {distances.shape}")
        print(f"Distances: {distances}")
        
        # Verify distances are within expected range
        max_dist = distances.max().item()
        min_dist = distances.min().item()
        print(f"Distance range: [{min_dist:.3f}, {max_dist:.3f}]")
        
        if max_dist <= reg_max - 0.01:
            print(f"✅ Distances are properly clamped to reg_max-0.01")
        else:
            print(f"❌ Distances exceed reg_max-0.01: {max_dist} > {reg_max-0.01}")
        
        return True
        
    except Exception as e:
        print(f"❌ bbox2dist calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("="*80)
    print("Complete YOLOv8 Loss Calculation Test")
    print("="*80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test bbox2dist function first
    bbox2dist_success = test_bbox2dist_function()
    
    # Create test data
    print(f"\nCreating test data...")
    data = create_test_data(batch_size=2, device=device)
    
    # Test TAL assignment
    tal_results = test_tal_assignment(data)
    
    # Test loss calculation
    loss_results = test_loss_calculation()
    loss_success = loss_results is not None
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"bbox2dist function: {'✅ PASSED' if bbox2dist_success else '❌ FAILED'}")
    print(f"TAL assignment: {'✅ PASSED' if tal_results is not None else '❌ FAILED'}")
    print(f"Loss calculation: {'✅ PASSED' if loss_success else '❌ FAILED'}")
    
    overall_success = bbox2dist_success and tal_results is not None and loss_success
    print(f"\nOverall: {'🎉 ALL TESTS PASSED!' if overall_success else '💥 SOME TESTS FAILED!'}")
    print("="*80)
    
    return overall_success

if __name__ == "__main__":
    main()