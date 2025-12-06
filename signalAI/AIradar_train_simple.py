import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import maximum_filter

# Import your corrected dataset class
from AIradar_datasetv8 import (
    AIRadarDataset,
    evaluate_dataset_metrics,
    _plot_2d_rdm,
    _plot_3d_rdm
)

# ======================================================================
# 1. Model Definition: Simple U-Net for 2D Heatmaps
# ======================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SimpleUNet(nn.Module):
    """
    Standard U-Net for 2D Segmentation.
    Input: [B, 1, H, W] (Normalized Range-Doppler Map)
    Output: [B, 1, H, W] (Logits for detection probability)
    """
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)
        
        # Initialize output bias to favor "background" (0) initially
        # sigmoid(-2.0) ~= 0.12, preventing high initial loss
        with torch.no_grad():
            self.outc.bias.data.fill_(-2.0)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder
        x = self.up1(x4)
        # Handle padding if dimensions don't match exactly after upsampling
        if x.shape != x3.shape:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        if x.shape != x2.shape:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        if x.shape != x1.shape:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        
        logits = self.outc(x)
        return logits

# ======================================================================
# 2. Combined Loss Function (Critical for Imbalanced Data)
# ======================================================================

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, pred, target, smooth=1.):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))
    
    def focal_loss(self, inputs, targets, alpha=0.8, gamma=2):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1-pt)**gamma * bce_loss
        return focal_loss.mean()

    def forward(self, inputs, targets):
        # Weighted sum: 1.0 Focal + 1.0 Dice + 0.5 BCE
        return self.focal_loss(inputs, targets) + self.dice_loss(inputs, targets) + 0.5 * self.bce(inputs, targets)

# ======================================================================
# 3. Data Loading & Preprocessing
# ======================================================================

def normalize_rdm_instance(rdm):
    """
    Normalize RDM to [0, 1] range based on its own min/max (Instance Norm).
    Input: tensor [B, H, W]
    Output: tensor [B, 1, H, W]
    """
    B, H, W = rdm.shape
    rdm = rdm.view(B, -1)
    min_val = rdm.min(dim=1, keepdim=True)[0]
    max_val = rdm.max(dim=1, keepdim=True)[0]
    
    # Avoid division by zero
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    
    rdm_norm = (rdm - min_val) / range_val
    return rdm_norm.view(B, 1, H, W)

def custom_collate(batch):
    """Handles dictionary batching"""
    elem = batch[0]
    collated = {}
    for key in elem:
        if isinstance(elem[key], torch.Tensor):
            collated[key] = torch.stack([d[key] for d in batch])
        elif isinstance(elem[key], (list, dict, str, np.ndarray)):
            collated[key] = [d[key] for d in batch]
        else:
             collated[key] = [d[key] for d in batch]
    return collated

# ======================================================================
# 4. Visualization & Post-Processing
# ======================================================================

def logits_to_detections(prob_map, range_axis, velocity_axis, threshold=0.5):
    """Convert probability map to detection list using NMS."""
    # 1. Threshold
    mask = prob_map > threshold
    
    # 2. Non-Maximum Suppression (3x3 kernel)
    local_max = maximum_filter(prob_map, size=3)
    mask = mask & (prob_map == local_max)
    
    # 3. Extract coordinates
    doppler_idxs, range_idxs = np.where(mask)
    detections = []
    
    for d_idx, r_idx in zip(doppler_idxs, range_idxs):
        detections.append({
            "range_idx": int(r_idx),
            "doppler_idx": int(d_idx),
            "range_m": float(range_axis[r_idx]),
            "velocity_mps": float(velocity_axis[d_idx]),
            "magnitude": float(prob_map[d_idx, r_idx]),
            "angle_deg": 0.0 # Placeholder
        })
    return detections

def plot_training_curves(losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DL Model Training Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# ======================================================================
# 5. Main Training Loop
# ======================================================================

def train_and_evaluate():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    out_dir = "data/dl_simplified_v1"
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Configurations to Train On ---
    train_configs = ['config1', 'config2', 'config_cn0566']
    BATCH_SIZE = 8
    EPOCHS = 15
    SAMPLES_PER_EPOCH = 200 # New random samples generated every epoch per config

    # Initialize Model
    model = SimpleUNet(n_channels=1, n_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CombinedLoss()
    
    loss_history = []

    print("\n--- Starting Training ---")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        total_batches = 0
        
        # Iterate through different radar configs to generalize
        for cfg_name in train_configs:
            # Generate FRESH data on the fly (infinite dataset approach)
            ds = AIRadarDataset(
                num_samples=SAMPLES_PER_EPOCH,
                config_name=cfg_name,
                save_path=out_dir, 
                drawfig=False,
                apply_realistic_effects=True
            )
            
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
            
            for batch in loader:
                # 1. Get RDM [B, H, W]
                rdm = batch['range_doppler_map'].to(device)
                
                # 2. Get Target Mask [B, H, W, 1] -> [B, 1, H, W]
                masks = batch['target_mask'].to(device)
                if masks.shape[-1] == 1:
                    masks = masks.permute(0, 3, 1, 2)
                
                # 3. Normalize Input (Instance Norm)
                inputs = normalize_rdm_instance(rdm)
                
                # 4. Forward
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # 5. Loss
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                total_batches += 1
        
        avg_loss = epoch_loss / total_batches
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch}/{EPOCHS}] Loss: {avg_loss:.4f}")

    # Save Model & Curve
    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pth"))
    plot_training_curves(loss_history, os.path.join(out_dir, "training_curve.png"))
    print("Training Complete.")

    # ==================================================================
    # 6. Evaluation on Test Data
    # ==================================================================
    print("\n--- Starting Evaluation ---")
    model.eval()
    
    # We will evaluate on 'config_cn0566' (Hardware faithful model)
    test_ds = AIRadarDataset(
        num_samples=50,
        config_name='config_cn0566',
        save_path=os.path.join(out_dir, "test_data"),
        drawfig=False
    )
    
    all_tp = 0
    all_fp = 0
    all_fn = 0
    
    vis_count = 0
    max_vis = 5
    
    with torch.no_grad():
        for i in range(len(test_ds)):
            sample = test_ds[i]
            
            # Prepare Input
            rdm = sample['range_doppler_map'].unsqueeze(0).to(device) # [1, H, W]
            inputs = normalize_rdm_instance(rdm) # [1, 1, H, W]
            
            # Inference
            logits = model(inputs)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy() # [H, W]
            
            # Post-Processing
            detections = logits_to_detections(
                prob_map, 
                test_ds.range_axis, 
                test_ds.velocity_axis, 
                threshold=0.5
            )
            
            # Evaluate against GT
            gt_targets = sample['target_info']['targets']
            metrics, matched, unmatched_t, unmatched_d = test_ds._evaluate_metrics(gt_targets, detections)
            
            all_tp += metrics['tp']
            all_fp += metrics['fp']
            all_fn += metrics['fn']
            
            # Visualize a few samples
            if vis_count < max_vis:
                vis_count += 1
                
                # Plot Raw Probability Map + Detections
                save_path_2d = os.path.join(out_dir, f"eval_sample_{i}_2d.png")
                save_path_3d = os.path.join(out_dir, f"eval_sample_{i}_3d.png")
                
                # We use the probability map as the "RDM" for visualization to see what the network thinks
                # But we normalize it to look like dB for the plotting function 
                # (Standard RDM plotting expects dB-like values, but we pass 0-1 prob map here for clarity)
                
                # To make the visualizer happy (it expects dB), we just scale prob for display
                # Or we can plot the original input RDM and overlay DL detections
                
                rdm_display = rdm.squeeze().cpu().numpy()
                rdm_display = rdm_display - rdm_display.max() # Normalize dB for display
                
                _plot_2d_rdm(test_ds, rdm_display, i, metrics, matched, unmatched_t, unmatched_d, save_path_2d)
                _plot_3d_rdm(test_ds, rdm_display, i, gt_targets, detections, save_path_3d)

    # Final Metrics
    precision = all_tp / (all_tp + all_fp + 1e-6)
    recall = all_tp / (all_tp + all_fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    print("\n=== Deep Learning Evaluation Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Visualizations saved to: {out_dir}")

if __name__ == "__main__":
    train_and_evaluate()