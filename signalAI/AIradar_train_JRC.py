import numpy as np
import torch
from torch.utils.data import Dataset
import random
from scipy.constants import c

class JRC_OFDM_Dataset(Dataset):
    def __init__(self, 
                 num_samples=1000, 
                 n_subcarriers=64, 
                 n_symbols=32, 
                 mod_order=16, # 16-QAM for robust training
                 snr_range=(15, 35),
                 fixed_snr=None):
        
        self.num_samples = num_samples
        self.K = n_subcarriers
        self.L = n_symbols
        self.mod_order = mod_order
        self.snr_range = snr_range
        self.fixed_snr = fixed_snr
        
        # Physics / 5G Numerology
        self.subcarrier_spacing = 30e3 
        self.fc = 28e9 
        self.bandwidth = self.K * self.subcarrier_spacing
        
        # Resolutions
        self.range_res = c / (2 * self.bandwidth)
        self.vel_res = c / (2 * self.fc * (self.L/self.subcarrier_spacing))
        
        self.range_axis = np.arange(self.K) * self.range_res
        self.velocity_axis = (np.arange(self.L) - self.L//2) * self.vel_res
        
        # Generate QAM Constellation
        self.constellation = self._generate_qam_constellation(mod_order)

    def _generate_qam_constellation(self, M):
        # Generate rectangular QAM
        k = int(np.log2(M))
        n = np.arange(M)
        a = np.array([2*(i//int(np.sqrt(M))) + 1 - int(np.sqrt(M)) + 1j*(2*(i%int(np.sqrt(M))) + 1 - int(np.sqrt(M))) for i in n])
        # Normalize energy
        return a / np.sqrt(np.mean(np.abs(a)**2))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Transmitter (Communication Data)
        tx_indices = np.random.randint(0, self.mod_order, (self.K, self.L))
        tx_grid = self.constellation[tx_indices] 
        
        # 2. Channel & Targets
        # Start with Direct Path (LoS) = 1.0
        rx_grid = tx_grid.copy()
        
        radar_mask = np.zeros((self.K, self.L), dtype=np.float32)
        target_list = []
        
        num_targets = random.randint(1, 3)
        for _ in range(num_targets):
            # Discrete Grid Locations (Guard bands at edges)
            r_idx = random.randint(4, self.K-4)
            d_idx = random.randint(4, self.L-4)
            
            # Target Physics
            # Strong Targets (0.5 to 1.0 relative to LoS)
            target_amp = random.uniform(0.5, 1.0) 
            target_phase = np.exp(1j * np.random.uniform(0, 2*np.pi))
            
            # Apply to Tx signal (Reflection = Shift in Freq Domain)
            rx_reflection = np.roll(tx_grid, r_idx, axis=0) 
            rx_reflection = np.roll(rx_reflection, d_idx, axis=1)
            
            rx_grid += target_amp * rx_reflection * target_phase
            
            # Ground Truth Mask
            radar_mask[r_idx, d_idx] = 1.0
            
            target_list.append({
                'range': self.range_axis[r_idx],
                'velocity': self.velocity_axis[d_idx],
                'range_idx': r_idx,
                'doppler_idx': d_idx
            })

        # 3. Noise
        if self.fixed_snr is not None:
            snr_db = self.fixed_snr
        else:
            snr_db = random.uniform(*self.snr_range)
            
        sig_pwr = np.mean(np.abs(rx_grid)**2)
        noise_pwr = sig_pwr / (10**(snr_db/10))
        noise = (np.random.randn(*rx_grid.shape) + 1j*np.random.randn(*rx_grid.shape)) * np.sqrt(noise_pwr/2)
        rx_grid += noise
        
        # 4. Prepare Tensors
        # [Real, Imag, Magnitude, Phase]
        rx_real = torch.from_numpy(rx_grid.real).float()
        rx_imag = torch.from_numpy(rx_grid.imag).float()
        rx_mag  = torch.sqrt(rx_real**2 + rx_imag**2)
        rx_phase = torch.atan2(rx_imag, rx_real)
        
        input_tensor = torch.stack([rx_real, rx_imag, rx_mag, rx_phase], dim=0)
        
        return {
            'input': input_tensor,
            'radar_label': torch.from_numpy(radar_mask).unsqueeze(0).float(),
            'comm_label': torch.from_numpy(tx_indices).long(),
            'tx_grid': tx_grid,
            'rx_grid': rx_grid,
            'targets': target_list,
            'snr': snr_db
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from tqdm import tqdm
from scipy.ndimage import maximum_filter

#from AIradar_JRC_Dataset import JRC_OFDM_Dataset

# ======================================================================
# 0. CUSTOM COLLATE (Fixes List Error)
# ======================================================================
def jrc_collate_fn(batch):
    batch_dict = {}
    for key in batch[0]:
        if key == 'targets':
            batch_dict[key] = [sample[key] for sample in batch]
        elif key in ['tx_grid', 'rx_grid']:
            batch_dict[key] = np.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], torch.Tensor):
            batch_dict[key] = torch.stack([sample[key] for sample in batch])
        else:
            batch_dict[key] = torch.tensor([sample[key] for sample in batch])
    return batch_dict

# ======================================================================
# 1. ATTENTION MODEL (Shared Backbone)
# ======================================================================

class CBAMBlock(nn.Module):
    """Channel & Spatial Attention"""
    def __init__(self, in_planes, ratio=16):
        super(CBAMBlock, self).__init__()
        # Channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # Spatial
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attn
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        x = x * self.sigmoid(avg_out + max_out)
        # Spatial Attn
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))
        return x * scale

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class JRC_AttentionNet(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        # Instance Norm centered on signal
        self.norm = nn.InstanceNorm2d(4, affine=True)
        
        # Backbone
        self.entry = nn.Sequential(nn.Conv2d(4, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer1 = ResBlock(64, 64)
        self.att1 = CBAMBlock(64) 
        self.layer2 = ResBlock(64, 128)
        self.att2 = CBAMBlock(128)
        self.layer3 = ResBlock(128, 256)
        
        # Radar Head (Segmentation)
        self.rad_up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
        self.rad_up[-1].bias.data.fill_(-4.0) # Bias for sparsity
        
        # Comm Head (Classification)
        self.comm_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, num_classes, 1) 
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.entry(x)
        x = self.att1(self.layer1(x))
        x = self.att2(self.layer2(x))
        feat = self.layer3(x)
        
        return self.rad_up(feat), self.comm_conv(feat)

# ======================================================================
# 2. METRICS
# ======================================================================

def calculate_metrics(r_logits, r_true, c_logits, c_true):
    # Comm
    c_pred = torch.argmax(c_logits, dim=1)
    acc = (c_pred == c_true).float().mean()
    ber = 1.0 - acc.item()
    
    # Radar F1
    r_prob = torch.sigmoid(r_logits)
    pred = (r_prob > 0.5).float()
    tp = (pred * r_true).sum()
    fp = (pred * (1-r_true)).sum()
    fn = ((1-pred) * r_true).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return f1.item(), ber

# ======================================================================
# 3. VISUALIZATION (2D, 3D, Constellation, BER)
# ======================================================================

def plot_dashboard(model, dataset, device, epoch, save_dir):
    model.eval()
    
    # High SNR sample for visuals
    ds_vis = JRC_OFDM_Dataset(num_samples=1, fixed_snr=35, mod_order=16)
    loader = DataLoader(ds_vis, batch_size=1, collate_fn=jrc_collate_fn)
    batch = next(iter(loader))
    
    with torch.no_grad():
        r_out, c_out = model(batch['input'].to(device))
    
    r_prob = torch.sigmoid(r_out[0,0]).cpu().numpy()
    gt_map = batch['radar_label'][0,0].numpy()
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Radar 2D Map
    ax1 = fig.add_subplot(2, 2, 1)
    im = ax1.imshow(r_prob, aspect='auto', origin='lower', cmap='jet', extent=[0, 100, -50, 50])
    # Overlay GT
    gy, gx = np.where(gt_map > 0.5)
    for x, y in zip(gx, gy):
        # Scale indices to plot extent
        px = x * (100/64); py = (y-16) * (100/32)
        ax1.plot(px, py, 'go', ms=20, mfc='none', mew=3, label='GT')
    ax1.set_title("Radar 2D Detection")
    ax1.set_xlabel("Range (m)"); ax1.set_ylabel("Velocity (m/s)")
    plt.colorbar(im, ax=ax1)
    
    # 2. Radar 3D Surface
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    X, Y = np.meshgrid(np.arange(r_prob.shape[1]), np.arange(r_prob.shape[0]))
    ax2.plot_surface(X, Y, r_prob, cmap='viridis', edgecolor='none')
    ax2.set_title("Radar 3D Confidence Surface")
    ax2.set_zlim(0, 1)
    
    # 3. Constellation Diagram
    ax3 = fig.add_subplot(2, 2, 3)
    rx = batch['rx_grid'][0].flatten()
    centroids = dataset.constellation
    ax3.scatter(rx.real, rx.imag, c='blue', alpha=0.3, s=15, label='Rx Symbols')
    ax3.scatter(centroids.real, centroids.imag, c='red', marker='x', s=50, label='Tx Centroids')
    ax3.set_title("Comm Constellation (16-QAM)")
    ax3.grid(True)
    ax3.legend()
    
    # 4. Training Stats Text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    txt = f"Epoch: {epoch}\nModulation: 16-QAM\nTargets: Strong (0.5-1.0)\nLoS: 1.0"
    ax4.text(0.1, 0.5, txt, fontsize=15, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/dashboard_epoch_{epoch}.png")
    plt.close()

def plot_ber_curve(model, device, save_dir):
    snrs = [0, 5, 10, 15, 20, 25, 30]
    bers = []
    model.eval()
    
    print("\nGenerating BER vs SNR Curve...")
    for s in snrs:
        ds = JRC_OFDM_Dataset(num_samples=100, fixed_snr=s, mod_order=16)
        dl = DataLoader(ds, batch_size=32, collate_fn=jrc_collate_fn)
        errs, total = 0, 0
        with torch.no_grad():
            for b in dl:
                _, c_out = model(b['input'].to(device))
                pred = torch.argmax(c_out, 1)
                errs += (pred != b['comm_label'].to(device)).sum().item()
                total += pred.numel()
        bers.append(errs/total)
    
    plt.figure()
    plt.semilogy(snrs, bers, 'b-o')
    plt.grid(True, which='both')
    plt.title("BER vs SNR (16-QAM JRC)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Symbol Error Rate")
    plt.savefig(f"{save_dir}/ber_curve.png")
    plt.close()

# ======================================================================
# 4. MAIN
# ======================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = "data/JRC_Attention_Final"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Dataset (16-QAM, Strong Targets)
    ds = JRC_OFDM_Dataset(num_samples=1000, mod_order=16)
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=jrc_collate_fn)
    
    # 2. Model
    model = JRC_AttentionNet(num_classes=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Losses
    # Weight Radar 20x because targets are sparse
    rad_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(device))
    comm_crit = nn.CrossEntropyLoss()
    
    loss_hist = []
    
    print("--- Starting JRC Attention Training ---")
    
    for epoch in range(15):
        model.train()
        r_acc_loss, c_acc_loss = 0, 0
        
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            inp = batch['input'].to(device)
            r_true = batch['radar_label'].to(device)
            c_true = batch['comm_label'].to(device)
            
            optimizer.zero_grad()
            r_pred, c_pred = model(inp)
            
            l_r = rad_crit(r_pred, r_true)
            l_c = comm_crit(c_pred, c_true)
            
            # Weighted Sum
            loss = 10.0 * l_r + l_c
            loss.backward()
            optimizer.step()
            
            r_acc_loss += l_r.item()
            c_acc_loss += l_c.item()
            loop.set_postfix(R_Loss=l_r.item(), C_Loss=l_c.item())
        
        loss_hist.append(loss.item())
        
        # Eval
        f1, ber = calculate_metrics(r_pred, r_true, c_pred, c_true)
        print(f"Epoch {epoch+1} -> Radar F1: {f1:.4f} | Comm BER: {ber:.4f}")
        
        # Plot
        plot_dashboard(model, ds, device, epoch+1, save_dir)
        
    # Final BER Curve
    plot_ber_curve(model, device, save_dir)
    
    # Loss Curve
    plt.figure()
    plt.plot(loss_hist)
    plt.title("Total Loss")
    plt.savefig(f"{save_dir}/loss_curve.png")
    
    print("Training Complete.")

if __name__ == "__main__":
    main()