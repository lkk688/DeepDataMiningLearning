"""
AIradar_comm_model_g1e.py

Joint Radar+Comm Deep Model (no attention), generalized across configs.

New in this version:
- Shared communication backbone + per-config heads:
    * One CNN encoder for comm.
    * A separate small conv head per config (indexed by config_id), with
      output channels = that config's modulation order.
- Per-config metrics during training & validation:
    * For each epoch, we print overall metrics AND per-config loss/SER.
- All previous visualization & evaluation logic is preserved:
    * Radar 2D/3D RDM comparison (CFAR vs DL).
    * Comm constellation + eye diagram.
    * SNR sweep plots & radar detection curves.
    * Full evaluation per config with classical vs deep metrics.
"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------
# Import your previous generator code (simulation + base dataset)
# ----------------------------------------------------------------------
# IMPORTANT: change this import if your module has a different name.
from AIradar_comm_dataset_g1 import RADAR_COMM_CONFIGS, AIRadar_Comm_Dataset

# Optional 3D visualization from your existing library
# try:
#     from AIRadarLib.visualization import plot_3d_range_doppler_map_with_ground_truth
#     VISUALIZATION_AVAILABLE = True
# except ImportError:
#     VISUALIZATION_AVAILABLE = False

VISUALIZATION_AVAILABLE = True
def plot_3d_range_doppler_map_with_ground_truth(
    rd_map,
    targets,
    range_resolution,
    velocity_resolution,
    num_range_bins,
    num_doppler_bins,
    title_prefix='',
    save_path=None,
    apply_doppler_centering=True,
    detections=None,
    view_range_limits=None,
    view_velocity_limits=None,
    is_db=False,
    stride=4
):
    """
    Plot a 3D Range-Doppler map with ground truth target annotations and CFAR detections.
    Uses physical units (Meters and m/s) for axes.

    Args:
        rd_map: np.ndarray, shape (2, num_doppler_bins, num_range_bins) if is_db=False, 
                or (num_doppler_bins, num_range_bins) if is_db=True.
        targets: list of dicts, each with keys 'distance', 'velocity', 'rcs'
        range_resolution: float, range resolution in meters
        velocity_resolution: float, velocity resolution in m/s
        num_range_bins: int, number of range bins
        num_doppler_bins: int, number of Doppler bins
        title_prefix: str, prefix for plot title/labeling/saving
        save_path: str, directory to save the figure
        apply_doppler_centering: bool, whether Doppler FFT is centered (affects target position calculation)
        detections: list of dicts, optional CFAR detection results
        view_range_limits: tuple, (min, max) range in meters to display
        view_velocity_limits: tuple, (min, max) velocity in m/s to display
        is_db: bool, if True, rd_map is treated as already in dB.
        stride: int, stride for surface plot downsampling (higher is faster)
    """
    if is_db:
        rd_db = rd_map
    else:
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        rd_db = 20 * np.log10(rd_magnitude + 1e-10)
    
    # Create physical axes
    range_axis = np.arange(num_range_bins) * range_resolution
    if apply_doppler_centering:
        velocity_axis = (np.arange(num_doppler_bins) - num_doppler_bins // 2) * velocity_resolution
    else:
        velocity_axis = np.arange(num_doppler_bins) * velocity_resolution
        
    X, Y = np.meshgrid(range_axis, velocity_axis)
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, rd_db, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8, rstride=stride, cstride=stride)
    
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_zlabel('Magnitude (dB)')
    ax.set_title(f'3D Range-Doppler Map with Ground Truth & Detections - {title_prefix}')
    
    # Set view limits if provided
    if view_range_limits:
        ax.set_xlim(view_range_limits)
    if view_velocity_limits:
        ax.set_ylim(view_velocity_limits)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Magnitude (dB)')

    # Plot Ground Truth Targets
    for i, target in enumerate(targets):
        r_val = target['distance']
        v_val = target['velocity']
        
        # Find indices for z-value lookup
        if apply_doppler_centering:
            r_idx = int(r_val / range_resolution)
            v_idx = int(num_doppler_bins // 2 + v_val / velocity_resolution)
        else:
            r_idx = int(r_val / range_resolution)
            v_idx = int(v_val / velocity_resolution) % num_doppler_bins
            
        if (0 <= r_idx < num_range_bins and 0 <= v_idx < num_doppler_bins):
            z_val = rd_db[v_idx, r_idx]
            # Lift the marker slightly above the surface
            ax.scatter([r_val], [v_val], [z_val + 2], color='red', s=100, marker='o', 
                      label='Ground Truth' if i == 0 else "")
            
            # Add label
            ax.text(r_val, v_val, z_val + 10, 
                   f"T{i+1}", color='black', fontsize=10, fontweight='bold')

    # Plot Detections (TP/FP)
    if detections:
        tp_count = 0
        fp_count = 0
        
        # Helper to check if a detection matches any target
        def is_tp(det_r_idx, det_d_idx, targets_list, r_tol=2, d_tol=2):
            for t in targets_list:
                if apply_doppler_centering:
                    t_r_idx = int(t['distance'] / range_resolution) # Assuming range starts at 0
                    t_d_idx = int(num_doppler_bins // 2 + t['velocity'] / velocity_resolution)
                else:
                    t_r_idx = int(t['distance'] / range_resolution)
                    t_d_idx = int(t['velocity'] / velocity_resolution) % num_doppler_bins
                
                if abs(det_r_idx - t_r_idx) <= r_tol and abs(det_d_idx - t_d_idx) <= d_tol:
                    return True
            return False

        for i, det in enumerate(detections):
            r_idx = det['range_idx']
            d_idx = det['doppler_idx']
            
            if (0 <= r_idx < num_range_bins and 0 <= d_idx < num_doppler_bins):
                z_val = rd_db[d_idx, r_idx]
                
                # Convert indices to physical units for plotting
                r_phys = r_idx * range_resolution
                if apply_doppler_centering:
                    v_phys = (d_idx - num_doppler_bins // 2) * velocity_resolution
                else:
                    v_phys = d_idx * velocity_resolution
                
                if is_tp(r_idx, d_idx, targets):
                    # True Positive
                    ax.scatter([r_phys], [v_phys], [z_val + 2], color='lime', s=80, marker='x', 
                              linewidth=2, label='Correct Detection' if tp_count == 0 else "")
                    tp_count += 1
                else:
                    # False Positive
                    ax.scatter([r_phys], [v_phys], [z_val + 2], color='yellow', s=80, marker='x', 
                              linewidth=2, label='False Positive' if fp_count == 0 else "")
                    fp_count += 1
    
    ax.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
C = 3e8  # speed of light

# ----------------------------------------------------------------------
# Config sets for many-to-many training / eval
# ----------------------------------------------------------------------
# TRAIN_CONFIGS = [
#     "CN0566_TRADITIONAL",
#     "Automotive_77GHz_LongRange",
#     "XBand_10GHz_MediumRange",
# ]
TRAIN_CONFIGS = [
    "CN0566_TRADITIONAL",
    "Automotive_77GHz_LongRange",
    "XBand_10GHz_MediumRange",
    "AUTOMOTIVE_TRADITIONAL",  # make sure this is included
]

VAL_CONFIGS = TRAIN_CONFIGS.copy()

# Optional held-out configs for cross-config evaluation
TEST_CONFIGS = [
    "AUTOMOTIVE_TRADITIONAL",
]

# Mapping from config_name -> integer ID for embedding / heads
CONFIG_ID_MAP = {name: i for i, name in enumerate(RADAR_COMM_CONFIGS.keys())}
NUM_CONFIGS = len(CONFIG_ID_MAP)

# Global max modulation order over all configs (for reference)
MAX_MOD_ORDER = max(cfg.get("mod_order", 4) for cfg in RADAR_COMM_CONFIGS.values())


# ----------------------------------------------------------------------
# Dataset loader that reuses joint_dump.npy if it exists
# ----------------------------------------------------------------------
class RadarCommDumpDataset(Dataset):
    """
    Lightweight dataset that loads from joint_dump.npy written by AIRadar_Comm_Dataset.

    joint_dump.npy entries look like:
        {
            'range_doppler_map': np.array [D,R],
            'cfar_detections': list of detections,
            'target_info': dict (targets + snr_db),
            'ofdm_map': np.array or None,
            'comm_info': dict with Tx/Rx symbols, ints, BER, etc.
        }

    We reconstruct:
        - range_axis / velocity_axis for TRADITIONAL (FMCW) configs
        - a dict sample compatible with RadarCommDeepDataset.
    """

    def __init__(self, config_name: str, num_samples: int, save_path: str):
        super().__init__()
        self.config_name = config_name
        self.config = RADAR_COMM_CONFIGS[config_name]
        self.mode = self.config["mode"]
        assert self.mode == "TRADITIONAL", "RadarCommDumpDataset currently supports TRADITIONAL only."

        self.c = C
        self.fc = self.config["fc"]
        self.lambda_c = self.c / self.fc

        dump_path = os.path.join(save_path, "joint_dump.npy")
        if not os.path.exists(dump_path):
            raise FileNotFoundError(f"joint_dump.npy not found at {dump_path}")

        arr = np.load(dump_path, allow_pickle=True)
        arr = list(arr)
        if len(arr) < num_samples:
            raise ValueError(
                f"Dump for {config_name} has only {len(arr)} samples, "
                f"requested {num_samples}."
            )
        self.records = arr[:num_samples]

        # Deduce RDM shape
        rdm0 = np.array(self.records[0]["range_doppler_map"])
        D, R = rdm0.shape

        # Rebuild axes for FMCW (TRADITIONAL mode)
        Nc = 64  # fixed in original generator
        fs = self.config["radar_fs"]
        B = self.config["radar_B"]
        T = self.config["radar_T"]
        slope = B / T
        Ns = int(fs * T)

        r_res = (self.c * fs) / (2 * slope * Ns)
        v_res = self.lambda_c / (2 * Nc * T)

        self.range_axis = np.arange(R) * r_res
        self.velocity_axis = np.arange(-Nc // 2, Nc // 2) * v_res

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        rdm_np = np.array(rec["range_doppler_map"], dtype=np.float32, copy=True)

        sample = {
            "mode": self.mode,
            "mod_order": self.config.get("mod_order", 4),
            "channel_model": self.config.get("channel_model", "multipath"),
            "range_doppler_map": torch.tensor(rdm_np, dtype=torch.float32),
            "range_axis": self.range_axis,
            "velocity_axis": self.velocity_axis,
            "target_info": rec["target_info"],
            "cfar_detections": rec["cfar_detections"],
            "comm_info": rec["comm_info"],
            "ofdm_map": rec.get("ofdm_map", None),
        }
        return sample

    # copy of AIRadar_Comm_Dataset._evaluate_metrics
    def _evaluate_metrics(self, targets, detections, match_dist_thresh=3.0):
        tp = 0
        range_errors = []
        velocity_errors = []
        unmatched_targets = targets.copy()
        unmatched_detections = detections.copy()
        matched_pairs = []
        for target in targets:
            best_dist = float("inf")
            best_det_idx = -1
            for i, det in enumerate(unmatched_detections):
                d_r = target["range"] - det["range_m"]
                d_v = target["velocity"] - det["velocity_mps"]
                dist = math.sqrt(d_r ** 2 + d_v ** 2)
                if dist < match_dist_thresh and dist < best_dist:
                    best_dist = dist
                    best_det_idx = i
            if best_det_idx != -1:
                tp += 1
                det = unmatched_detections[best_det_idx]
                range_errors.append(abs(target["range"] - det["range_m"]))
                velocity_errors.append(abs(target["velocity"] - det["velocity_mps"]))
                matched_pairs.append((target, det))
                unmatched_detections.pop(best_det_idx)
                unmatched_targets.remove(target)
        fp = len(unmatched_detections)
        fn = len(targets) - tp
        metrics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_range_error": float(np.mean(range_errors)) if range_errors else 0.0,
            "mean_velocity_error": float(np.mean(velocity_errors)) if velocity_errors else 0.0,
            "total_targets": len(targets),
        }
        return metrics, matched_pairs, unmatched_targets, unmatched_detections


def load_or_generate_base_dataset(
    config_name: str,
    split: str,
    num_samples: int,
    data_root: str,
    drawfig: bool = False,
):
    """
    Helper that either:
      - loads an existing joint_dump.npy and wraps it in RadarCommDumpDataset, or
      - calls AIRadar_Comm_Dataset to generate new samples, then wraps in RadarCommDumpDataset.

    This avoids re-running the expensive simulator when data already exists.
    """
    save_path = os.path.join(data_root, split, config_name)
    os.makedirs(save_path, exist_ok=True)
    dump_path = os.path.join(save_path, "joint_dump.npy")

    if os.path.exists(dump_path):
        arr = np.load(dump_path, allow_pickle=True)
        existing = len(arr)
        if existing >= num_samples:
            print(
                f"[Dataset] Reusing existing dump for {config_name} "
                f"(split={split}): {existing} samples >= requested {num_samples}."
            )
            return RadarCommDumpDataset(config_name, num_samples, save_path)
        else:
            more = num_samples - existing
            print(
                f"[Dataset] Existing dump for {config_name} split={split} has {existing} < {num_samples} samples, "
                f"generating {more} more..."
            )
            # Generate extra samples; AIRadar_Comm_Dataset will append to joint_dump.npy
            AIRadar_Comm_Dataset(
                config_name=config_name,
                num_samples=more,
                save_path=save_path,
                drawfig=drawfig,
            )
            return RadarCommDumpDataset(config_name, num_samples, save_path)
    else:
        print(
            f"[Dataset] No dump found for {config_name} split={split}, "
            f"generating {num_samples} samples..."
        )
        AIRadar_Comm_Dataset(
            config_name=config_name,
            num_samples=num_samples,
            save_path=save_path,
            drawfig=drawfig,
        )
        return RadarCommDumpDataset(config_name, num_samples, save_path)


# ----------------------------------------------------------------------
# Deep dataset wrapper: builds radar/comm tensors + labels
# ----------------------------------------------------------------------
def summarize_snr_for_split(base_ds, name: str, max_samples: int = 200):
    snrs = []
    for i in range(min(len(base_ds), max_samples)):
        info = base_ds[i]["target_info"]
        snr_db = float(info.get("snr_db", 0.0))
        snrs.append(snr_db)
    if not snrs:
        print(f"[Diag] No SNR data for {name}")
        return
    snrs = np.array(snrs, dtype=np.float32)
    print(
        f"[Diag] {name}: N={len(snrs)}, "
        f"SNR mean={snrs.mean():.2f} dB, "
        f"min={snrs.min():.2f}, "
        f"max={snrs.max():.2f}"
    )

class RadarCommDeepDataset(Dataset):
    """
    Wraps a base dataset (AIRadar_Comm_Dataset or RadarCommDumpDataset) and
    builds deep-learning-ready tensors.

    Returns:
      radar_input:  [1, D, R] (dB RDM)
      radar_target: [1, D, R] soft heatmap
      comm_input:   [2, N_syms, N_fft] (Rx I/Q)
      comm_target:  [N_syms, N_fft] symbol indices
      meta:         dict with config_id, config_name, mod_order, snr_db
    """

    def __init__(self, base_ds, config_name: str, radar_sigma_cells: float = 1.5):
        super().__init__()
        self.base_ds = base_ds
        self.config_name = config_name
        self.cfg = RADAR_COMM_CONFIGS[config_name]
        assert self.cfg["mode"] == "TRADITIONAL", "Deep dataset supports TRADITIONAL configs only."
        self.radar_sigma_cells = radar_sigma_cells
        self.config_id = CONFIG_ID_MAP[config_name]

    def __len__(self):
        return len(self.base_ds)

    @staticmethod
    def _build_radar_label(rdm, r_axis, v_axis, targets, sigma_cells: float = 1.5, radius: int = 3):
        D_r, R_r = rdm.shape
        label = np.zeros_like(rdm, dtype=np.float32)
        sigma2 = sigma_cells ** 2

        for t in targets:
            r_m = t["range"]
            v_m = t["velocity"]
            r_idx = int(np.argmin(np.abs(r_axis - r_m)))
            v_idx = int(np.argmin(np.abs(v_axis - v_m)))
            if not (0 <= r_idx < R_r and 0 <= v_idx < D_r):
                continue
            for dv in range(-radius, radius + 1):
                for dr in range(-radius, radius + 1):
                    rr = r_idx + dr
                    dd = v_idx + dv
                    if 0 <= rr < R_r and 0 <= dd < D_r:
                        dist2 = dr * dr + dv * dv
                        val = math.exp(-dist2 / (2 * sigma2))
                        label[dd, rr] = max(label[dd, rr], val)
        return label

    def __getitem__(self, idx):
        s = self.base_ds[idx]

        # ---------- Radar inputs ----------
        rdm = np.array(s["range_doppler_map"].numpy(), dtype=np.float32, copy=True)
        r_axis = np.asarray(s["range_axis"])
        v_axis = np.asarray(s["velocity_axis"])
        targets = s["target_info"]["targets"]
        snr_db = float(s["target_info"].get("snr_db", 0.0))

        radar_label = self._build_radar_label(
            rdm, r_axis, v_axis, targets, sigma_cells=self.radar_sigma_cells
        )

        radar_input = torch.tensor(rdm, dtype=torch.float32).unsqueeze(0).contiguous()
        radar_target = torch.tensor(radar_label, dtype=torch.float32).unsqueeze(0).contiguous()

        # ---------- Comm inputs ----------
        comm_info = s["comm_info"]
        if comm_info is None:
            raise RuntimeError("comm_info is missing; ensure TRADITIONAL mode dataset.")

        num_syms = comm_info["num_data_syms"]
        fft_size = comm_info["fft_size"]
        mod_order = comm_info["mod_order"]

        tx_ints = np.array(comm_info["tx_ints"], dtype=np.int64)
        assert tx_ints.size == num_syms * fft_size, "Mismatch between tx_ints and OFDM grid shape."
        comm_label = tx_ints.reshape(num_syms, fft_size)

        rx_syms = np.array(comm_info["rx_symbols"], dtype=np.complex64)
        assert rx_syms.size == num_syms * fft_size, "rx_symbols size mismatch."
        rx_grid = rx_syms.reshape(num_syms, fft_size)

        real = rx_grid.real
        imag = rx_grid.imag
        comm_input_np = np.stack([real, imag], axis=0).astype(np.float32)

        comm_input = torch.tensor(comm_input_np, dtype=torch.float32).contiguous()
        comm_target = torch.tensor(comm_label, dtype=torch.long).contiguous()

        meta = {
            "config_id": self.config_id,
            "config_name": self.config_name,
            "mod_order": mod_order,
            "snr_db": snr_db,
        }

        return radar_input, radar_target, comm_input, comm_target, meta


# ----------------------------------------------------------------------
# Model definition: radar branch + shared comm backbone + per-config heads
# ----------------------------------------------------------------------
import math
from typing import Optional, Dict, Tuple

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + x)
        return out


class ConvStem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RadarBranch(nn.Module):
    def __init__(self, in_ch=1, base_ch=48, num_blocks=4):
        super().__init__()
        self.stem = ConvStem(in_ch, base_ch)
        blocks = []
        dilations = [1, 2, 4, 1][:num_blocks]
        for d in dilations:
            blocks.append(ResidualBlock(base_ch, dilation=d))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

class CommSymbolMLP(nn.Module):
    """
    Symbol-wise demapper:

    For each Rx symbol, we build a feature:
        [I, Q, cond]  (cond = config+SNR embedding)

    Then pass through an MLP, and finally a per-config Linear head:
        hidden -> mod_order (4, 16, 64, ...)

    Input:
      comm_input: [B, 2, H, W]             (I/Q grid)
      cond:       [B, cond_dim]           (config+SNR embedding)
      config_ids: [B] long

    Output:
      comm_logits: [B, mod_order, H, W]
    """

    def __init__(self,
                 cond_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 max_mod_order: int = 64):
        super().__init__()
        self.cond_dim = cond_dim
        in_dim = 2 + cond_dim  # I, Q, cond

        layers = []
        d = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            d = hidden_dim
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

        # Per-config heads: hidden_dim -> mod_order
        heads = [None] * NUM_CONFIGS
        for cfg_name, cfg in RADAR_COMM_CONFIGS.items():
            cfg_id = CONFIG_ID_MAP[cfg_name]
            mod_order = cfg.get("mod_order", max_mod_order)
            heads[cfg_id] = nn.Linear(hidden_dim, mod_order)
        self.heads = nn.ModuleList(heads)

    def forward(self, comm_input, cond, config_ids):
        """
        comm_input: [B, 2, H, W]
        cond:       [B, cond_dim]
        config_ids: [B]

        returns:
          comm_logits: [B, mod_order, H, W]
        """
        B, C_in, H, W = comm_input.shape
        assert C_in == 2, f"Expected 2 channels (I/Q), got {C_in}"

        # Tile cond over all symbols
        cond_tiled = cond.view(B, self.cond_dim, 1, 1).expand(B, self.cond_dim, H, W)
        # [B, 2 + cond_dim, H, W]
        x = torch.cat([comm_input, cond_tiled], dim=1)

        # Flatten symbols: [B, C, H, W] -> [B*H*W, C]
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, 2 + self.cond_dim)

        # Shared MLP
        feats = self.mlp(x)  # [B*H*W, hidden_dim]

        # Single config per batch (your loaders are per-config)
        unique_cfg_ids = torch.unique(config_ids)
        assert unique_cfg_ids.numel() == 1, "Mixed configs in one batch not supported."
        cfg_id_scalar = int(unique_cfg_ids.item())
        head = self.heads[cfg_id_scalar]

        logits_flat = head(feats)  # [B*H*W, mod_order]
        # Reshape back: [B, mod_order, H, W]
        # First [B, H, W, mod_order], then permute
        mod_order = logits_flat.shape[-1]
        logits = logits_flat.view(B, H, W, mod_order).permute(0, 3, 1, 2).contiguous()

        return logits
    
class CommBackbone(nn.Module):
    """
    Stronger communication backbone:

    - CNN stem + residual blocks extract local I/Q features.
    - Transformer encoder along the *subcarrier* axis for each OFDM symbol
      to capture long-range dependencies and structure across frequency.

    Input:  x [B, C_in, H, W]
            (C_in = 2 + cond_dim, H = num_syms, W = fft_size)

    Output: features [B, base_ch, H, W]
    """

    def __init__(self,
                 in_ch: int = 2,
                 base_ch: int = 64,
                 num_blocks: int = 3,
                 num_attn_layers: int = 2,
                 num_attn_heads: int = 4,
                 attn_dropout: float = 0.1):
        super().__init__()

        self.base_ch = base_ch

        # --- CNN stem + residual conv blocks ---
        self.stem = ConvStem(in_ch, base_ch)

        blocks = []
        # mild dilation to grow receptive field
        dilations = [1, 2, 4][:num_blocks]
        for d in dilations:
            blocks.append(ResidualBlock(base_ch, dilation=d))
        self.blocks = nn.Sequential(*blocks)

        # --- Transformer encoder over frequency (subcarriers) ---
        # We'll treat each OFDM symbol row independently:
        # for each (B, symbol_idx) we have a sequence over W subcarriers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=base_ch,
            nhead=num_attn_heads,
            dim_feedforward=base_ch * 4,
            dropout=attn_dropout,
            batch_first=True,   # [batch, seq, dim]
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attn_layers,
        )

    def forward(self, x):
        """
        x: [B, C_in, H, W]
        returns: [B, base_ch, H, W]
        """
        B, C_in, H, W = x.shape

        # --- CNN part ---
        x = self.stem(x)      # [B, base_ch, H, W]
        x = self.blocks(x)    # [B, base_ch, H, W]

        # --- Transformer over frequency ---
        # reshape: treat each symbol row as a sequence over subcarriers
        # current layout: [B, C, H, W]
        # target for transformer: [batch', seq_len, d_model]
        #   batch' = B * H  (each row is an independent sequence)
        #   seq_len = W
        x_perm = x.permute(0, 2, 3, 1)          # [B, H, W, C]
        x_seq = x_perm.reshape(B * H, W, self.base_ch)  # [B*H, W, C]

        # Transformer encoder
        x_seq = self.transformer(x_seq)         # [B*H, W, C]

        # reshape back to [B, C, H, W]
        x_seq = x_seq.reshape(B, H, W, self.base_ch)    # [B, H, W, C]
        x_out = x_seq.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        return x_out


class SharedRawEncoder(nn.Module):
    """
    Small residual CNN that keeps the same (C,H,W) shape.
    Can be applied separately to radar and comm raw inputs.
    """
    def __init__(self, channels: int, hidden_channels: int = None, kernel_size: int = 3):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return None
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.act(out)
        return out

class JointRadarCommNet(nn.Module):
    """
    Joint model with:
      - RadarBranch (unchanged)
      - Shared CommBackbone + per-config heads
      - Config+SNR conditioning for both branches
    """

    def __init__(self, num_configs: int, max_mod_order=64, base_ch=48,
                 num_blocks=4, cond_dim=16):
        super().__init__()
        self.cond_dim = cond_dim

        self.config_embed = nn.Embedding(num_configs, cond_dim)
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

        # Radar: 1 + cond_dim channels
        self.radar_branch = RadarBranch(in_ch=1 + cond_dim, base_ch=base_ch, num_blocks=num_blocks)

        # Comm backbone: 2 + cond_dim channels
        #self.comm_backbone = CommBackbone(in_ch=2 + cond_dim, base_ch=base_ch, num_blocks=num_blocks)
        # NEW: symbol-wise comm demapper
        self.comm_symbol_mlp = CommSymbolMLP(
            cond_dim=cond_dim,
            hidden_dim=128,
            num_layers=3,
            max_mod_order=max_mod_order,
        )

        # Per-config comm heads: one 1x1 conv per config, out_channels = mod_order
        heads = [None] * num_configs
        for cfg_name, cfg in RADAR_COMM_CONFIGS.items():
            cfg_id = CONFIG_ID_MAP[cfg_name]
            mod_order = cfg.get("mod_order", max_mod_order)
            heads[cfg_id] = nn.Conv2d(base_ch, mod_order, kernel_size=1)
        self.comm_heads = nn.ModuleList(heads)

    def forward(self, radar_input, comm_input, config_ids: torch.Tensor, snr_db: torch.Tensor):
        """
        radar_input: [B,1,D,R]
        comm_input:  [B,2,H,W]
        config_ids:  [B] long
        snr_db:      [B] float (in dB)
        """
        B, _, D, R = radar_input.shape
        _, _, H, W = comm_input.shape

        snr_norm = (snr_db / 40.0).clamp(0.0, 2.0).unsqueeze(1)  # [B,1]
        cfg_emb = self.config_embed(config_ids)                  # [B,cond_dim]
        snr_emb = self.snr_mlp(snr_norm)                         # [B,cond_dim]
        cond = torch.tanh(cfg_emb + snr_emb)                     # [B,cond_dim]

        cond_radar = cond.view(B, self.cond_dim, 1, 1).expand(B, self.cond_dim, D, R)
        cond_comm = cond.view(B, self.cond_dim, 1, 1).expand(B, self.cond_dim, H, W)

        radar_in = torch.cat([radar_input, cond_radar], dim=1)
        comm_in = torch.cat([comm_input, cond_comm], dim=1)

        radar_logits = self.radar_branch(radar_in)

        # Comm backbone + per-config head
        #features = self.comm_backbone(comm_in)  # [B, base_ch, H, W]
        # 3) Comm branch: symbol-wise MLP
        #    Comm input remains only [B,2,H,W] (raw I/Q);
        #    cond is injected inside the MLP.
        comm_logits = self.comm_symbol_mlp(comm_input, cond, config_ids)
        
        # We assume a single config per batch (true for our per-config loaders)
        # unique_cfg_ids = torch.unique(config_ids)
        # assert unique_cfg_ids.numel() == 1, "Mixed configs in one batch not supported for per-config heads."
        # cfg_id_scalar = int(unique_cfg_ids.item())
        # head = self.comm_heads[cfg_id_scalar]
        # comm_logits = head(features)           # [B, mod_order, H, W]

        return radar_logits, comm_logits

class FusionTransformer(nn.Module):
    """
    Lightweight Transformer over a small set of tokens:
      - radar token (optional)
      - comm token (optional)
      - cond token (config+SNR embedding, optional)

    It outputs FiLM (gamma, beta) for radar / comm logits.
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_radar: bool = True,
        use_comm: bool = True,
        use_cond: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_radar = use_radar
        self.use_comm = use_comm
        self.use_cond = use_cond

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # [B, N, D]
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # FiLM heads (gamma, beta) for each branch.
        if use_radar:
            self.radar_film = nn.Linear(d_model, 2)
            nn.init.zeros_(self.radar_film.weight)
            nn.init.zeros_(self.radar_film.bias)
        else:
            self.radar_film = None

        if use_comm:
            self.comm_film = nn.Linear(d_model, 2)
            nn.init.zeros_(self.comm_film.weight)
            nn.init.zeros_(self.comm_film.bias)
        else:
            self.comm_film = None

    def forward(
        self,
        radar_token: torch.Tensor | None,
        comm_token: torch.Tensor | None,
        cond_token: torch.Tensor | None = None,
    ):
        """
        radar_token: (B, D) or None
        comm_token:  (B, D) or None
        cond_token:  (B, D) or None
        """
        tokens = []
        kinds = []

        if radar_token is not None and self.use_radar:
            tokens.append(radar_token)
            kinds.append("radar")

        if comm_token is not None and self.use_comm:
            tokens.append(comm_token)
            kinds.append("comm")

        if cond_token is not None and self.use_cond:
            tokens.append(cond_token)
            kinds.append("cond")

        if not tokens:
            return None, None

        # (B, N, D)
        x = torch.stack(tokens, dim=1)
        x = self.encoder(x)  # (B, N, D)

        radar_gamma_beta = None
        comm_gamma_beta = None

        idx = 0
        for kind in kinds:
            if kind == "radar" and self.radar_film is not None:
                radar_gamma_beta = self.radar_film(x[:, idx, :])  # (B, 2)
            elif kind == "comm" and self.comm_film is not None:
                comm_gamma_beta = self.comm_film(x[:, idx, :])    # (B, 2)
            idx += 1

        return radar_gamma_beta, comm_gamma_beta

class JointRadarCommNet_v2(nn.Module):
    """
    Joint model v2:

      - Reuses RadarBranch + CommSymbolMLP (your previous working model).
      - Optional SharedRawEncoder on radar/comm raw inputs (use_shared_raw).
      - Optional FusionTransformer for cross-modal FiLM on logits (use_fusion).

    Interface (same as old JointRadarCommNet):

        forward(radar_input, comm_input, config_ids, snr_db)

      radar_input: [B,1,D,R]
      comm_input:  [B,2,H,W]
      config_ids:  [B] long
      snr_db:      [B] float (dB)
    """

    def __init__(
        self,
        num_configs: int,
        max_mod_order: int = 64,
        base_ch: int = 48,
        num_blocks: int = 4,
        cond_dim: int = 16,
        # sharing options
        use_shared_raw: bool = False,
        shared_raw_hidden: int | None = None,
        use_fusion: bool = False,
        fusion_d_model: int = 128,
        fusion_nhead: int = 4,
        fusion_num_layers: int = 1,
        fusion_dim_feedforward: int = 256,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_configs = num_configs
        self.max_mod_order = max_mod_order

        self.use_shared_raw = use_shared_raw
        self.use_fusion = use_fusion

        # ----- Config + SNR embedding (same as old) -----
        self.config_embed = nn.Embedding(num_configs, cond_dim)
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

        # ----- Radar branch (unchanged) -----
        # Radar input will be [B, 1 + cond_dim, D, R]
        self.radar_branch = RadarBranch(
            in_ch=1 + cond_dim,
            base_ch=base_ch,
            num_blocks=num_blocks,
        )

        # ----- Comm demapper (unchanged) -----
        self.comm_symbol_mlp = CommSymbolMLP(
            cond_dim=cond_dim,
            hidden_dim=128,
            num_layers=3,
            max_mod_order=max_mod_order,
        )

        # ----- Optional raw encoders -----
        if use_shared_raw:
            # raw radar: [B,1,D,R]
            self.radar_raw_encoder = SharedRawEncoder(
                channels=1,
                hidden_channels=shared_raw_hidden,
            )
            # raw comm: [B,2,H,W]
            self.comm_raw_encoder = SharedRawEncoder(
                channels=2,
                hidden_channels=shared_raw_hidden,
            )
        else:
            self.radar_raw_encoder = None
            self.comm_raw_encoder = None

        # ----- Optional fusion transformer -----
        if use_fusion:
            d_model = fusion_d_model

            # Token projections
            # radar: global mean over (D,R), channel=1 -> [B,1] -> proj -> [B,D]
            self.radar_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.radar_proj = nn.Linear(1, d_model)

            # comm: global mean over (C,H,W) -> scalar -> proj -> [B,D]
            self.comm_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.comm_proj = nn.Linear(1, d_model)

            # cond (config+SNR) -> token
            self.cond_proj = nn.Linear(cond_dim, d_model)

            self.fusion = FusionTransformer(
                d_model=d_model,
                nhead=fusion_nhead,
                num_layers=fusion_num_layers,
                dim_feedforward=fusion_dim_feedforward,
                dropout=fusion_dropout,
                use_radar=True,
                use_comm=True,
                use_cond=True,
            )
        else:
            self.fusion = None

    def forward(
        self,
        radar_input: torch.Tensor,
        comm_input: torch.Tensor,
        config_ids: torch.Tensor,
        snr_db: torch.Tensor,
    ):
        """
        radar_input: [B,1,D,R]
        comm_input:  [B,2,H,W]
        config_ids:  [B] long
        snr_db:      [B] float (dB)
        """
        B, _, D, R = radar_input.shape
        _, _, H, W = comm_input.shape

        # ----- Optional raw encoders -----
        if self.use_shared_raw and self.radar_raw_encoder is not None:
            radar_raw = self.radar_raw_encoder(radar_input)
        else:
            radar_raw = radar_input

        if self.use_shared_raw and self.comm_raw_encoder is not None:
            comm_raw = self.comm_raw_encoder(comm_input)
        else:
            comm_raw = comm_input

        # ----- Config + SNR embedding (same as old) -----
        snr_norm = (snr_db / 40.0).clamp(0.0, 2.0).unsqueeze(1)  # [B,1]
        cfg_emb = self.config_embed(config_ids)                  # [B,cond_dim]
        snr_emb = self.snr_mlp(snr_norm)                         # [B,cond_dim]
        cond = torch.tanh(cfg_emb + snr_emb)                     # [B,cond_dim]

        # Radar conditioning: tile cond spatially
        cond_radar = cond.view(B, self.cond_dim, 1, 1).expand(B, self.cond_dim, D, R)
        # (You don't actually need cond_comm for the symbol MLP)
        # cond_comm = cond.view(B, self.cond_dim, 1, 1).expand(B, self.cond_dim, H, W)

        # ----- Radar branch -----
        radar_in = torch.cat([radar_raw, cond_radar], dim=1)     # [B,1+cond_dim,D,R]
        radar_logits = self.radar_branch(radar_in)               # [B,1,D,R]

        # ----- Comm branch (symbol-wise MLP, unchanged) -----
        # CommSymbolMLP takes raw [B,2,H,W] + cond; we feed the (optionally) encoded comm_raw.
        comm_logits = self.comm_symbol_mlp(comm_raw, cond, config_ids)  # [B,mod_order,H,W]

        # ----- Optional fusion (FiLM on logits) -----
        if self.use_fusion and self.fusion is not None:
            # Radar token
            radar_token = None
            if radar_logits is not None:
                # [B,1,D,R] -> [B,1,1,1] -> [B,1]
                pooled_r = self.radar_pool(radar_logits).view(B, 1)
                radar_token = self.radar_proj(pooled_r)  # [B,D]

            # Comm token
            comm_token = None
            if comm_logits is not None:
                # [B,C,H,W] -> [B,C,1,1] -> mean over C -> [B,1]
                pooled_c_spatial = self.comm_pool(comm_logits)   # [B,C,1,1]
                scalar_c = pooled_c_spatial.mean(dim=1).view(B, 1)
                comm_token = self.comm_proj(scalar_c)            # [B,D]

            # Cond token
            cond_token = self.cond_proj(cond)  # [B,D]

            radar_gb, comm_gb = self.fusion(radar_token, comm_token, cond_token)

            # Apply FiLM: logits = logits * (1+gamma) + beta
            if radar_logits is not None and radar_gb is not None:
                gamma_r = radar_gb[:, 0].view(B, 1, 1, 1)
                beta_r = radar_gb[:, 1].view(B, 1, 1, 1)
                radar_logits = radar_logits * (1.0 + gamma_r) + beta_r

            if comm_logits is not None and comm_gb is not None:
                gamma_c = comm_gb[:, 0].view(B, 1, 1, 1)
                beta_c = comm_gb[:, 1].view(B, 1, 1, 1)
                comm_logits = comm_logits * (1.0 + gamma_c) + beta_c

        return radar_logits, comm_logits
# =========================
#   Optional: Focal losses
# =========================

class BinaryFocalLossWithLogits(nn.Module):
    """
    Focal loss for radar detection (binary, on logits).

    Use in place of or combined with BCE:
        L = lambda_bce * BCE + lambda_focal * Focal
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets: same shape (B,1,H,W)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # p_t = p if y=1 else 1-p = exp(-bce)
        p_t = torch.exp(-bce)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class FocalCrossEntropyLoss(nn.Module):
    """
    Focal cross-entropy for communication logits.

    logits: (B, M, T, F)
    targets: (B, T, F)  # symbol indices in [0, mod_order-1], ignore_index optional.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0,
                 ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, M, T, F_ = logits.shape
        # (B,M,T,F) -> (B,T,F,M)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        logits = logits.view(B * T * F_, M)
        targets = targets.view(B * T * F_)

        ce = F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # ignore masked (ignore_index) positions
        valid = (targets != self.ignore_index).float()
        p_t = torch.exp(-ce)
        focal = self.alpha * (1 - p_t) ** self.gamma * ce * valid

        if self.reduction == "mean":
            denom = valid.sum().clamp_min(1.0)
            return focal.sum() / denom
        elif self.reduction == "sum":
            return focal.sum()
        return focal

# ----------------------------------------------------------------------
# Losses & meta extraction helper
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_losses(
    radar_logits,
    radar_target,
    comm_logits,
    comm_target,
    radar_pos_weight: float = 10.0, #2.0,
    lambda_comm: float = 1.0,
    use_focal: bool = True,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    dice_weight: float = 0.0,
):
    """
    Compute joint radar + comm loss.

    - Radar: BCE-with-logits (with pos_weight) + optional focal + optional Dice.
    - Comm:  Cross-entropy over modulation order per (subcarrier, symbol).
    - Shapes:
        radar_logits: (B,1,H_r,W_r)
        radar_target: (B,1,H_t,W_t)
        comm_logits:  (B,M,Hc,Wc)
        comm_target:  (B,Hc,Wc) integer labels in [0, M-1]

    The function automatically interpolates radar_logits to match radar_target
    spatial size if they differ (e.g. 64x1000 → 64x2048).
    """

    # -------------------------------
    # Radar loss (BCE + focal + Dice)
    # -------------------------------
    if radar_logits is not None and radar_target is not None:
        # 1) Make sure spatial sizes match
        if radar_logits.shape != radar_target.shape:
            # assume (B,1,H,W) and only H/W differ
            _, _, H_t, W_t = radar_target.shape
            radar_logits = F.interpolate(
                radar_logits,
                size=(H_t, W_t),
                mode="bilinear",
                align_corners=False,
            )

        # BCE with pos_weight
        pos_weight = torch.tensor(
            [radar_pos_weight],
            device=radar_logits.device,
            dtype=radar_logits.dtype,
        )
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        radar_bce = bce(radar_logits, radar_target)

        radar_loss = radar_bce

        # For focal & dice, we need probabilities
        p = torch.sigmoid(radar_logits)

        # Focal term (optional)
        if use_focal:
            # pt = p if y=1, (1-p) if y=0
            pt = torch.where(radar_target == 1, p, 1.0 - p)
            # alpha_t = alpha for positives, (1-alpha) for negatives
            alpha_pos = torch.full_like(p, focal_alpha)
            alpha_neg = torch.full_like(p, 1.0 - focal_alpha)
            alpha_t = torch.where(radar_target == 1, alpha_pos, alpha_neg)

            focal_map = -alpha_t * (1 - pt).pow(focal_gamma) * torch.log(
                pt.clamp(min=1e-6)
            )
            radar_focal = focal_map.mean()
            radar_loss = radar_loss + radar_focal
        else:
            radar_focal = torch.tensor(0.0, device=radar_logits.device)

        # Dice term (optional)
        if dice_weight > 0.0:
            smooth = 1.0
            p_flat = p.view(p.size(0), -1)
            t_flat = radar_target.view(radar_target.size(0), -1)
            intersection = (p_flat * t_flat).sum(dim=1)
            denom = p_flat.sum(dim=1) + t_flat.sum(dim=1)
            dice = 1.0 - (2.0 * intersection + smooth) / (denom + smooth)
            radar_dice = dice.mean()
            radar_loss = radar_loss + dice_weight * radar_dice
        else:
            radar_dice = torch.tensor(0.0, device=radar_logits.device)

    else:
        # radar-only / comm-only modes
        radar_loss = torch.tensor(
            0.0,
            device=comm_logits.device if comm_logits is not None else "cpu",
        )

    # -------------------------------
    # Communication loss (CE)
    # -------------------------------
    if comm_logits is not None and comm_target is not None:
        B, M, Hc, Wc = comm_logits.shape
        ce = nn.CrossEntropyLoss()

        # (B, M, Hc, Wc) -> (B*Hc*Wc, M)
        logits_flat = comm_logits.permute(0, 2, 3, 1).reshape(-1, M)
        labels_flat = comm_target.reshape(-1).long()

        comm_loss = ce(logits_flat, labels_flat)
    else:
        comm_loss = torch.tensor(
            0.0,
            device=radar_logits.device if radar_logits is not None else "cpu",
        )

    # -------------------------------
    # Total loss
    # -------------------------------
    total_loss = radar_loss + lambda_comm * comm_loss
    return total_loss, radar_loss, comm_loss

def extract_meta_tensors(meta, device):
    """
    DataLoader default_collate turns list-of-dicts into a dict-of-tensors/lists:
      meta = {
          'config_id': LongTensor[B],
          'config_name': list[str],
          'mod_order': LongTensor[B],
          'snr_db': Tensor[B],
      }

    This helper converts that into cfg_ids, snr_db tensors on the target device.
    """
    if isinstance(meta, dict):
        cfg_ids = meta["config_id"]
        if not torch.is_tensor(cfg_ids):
            cfg_ids = torch.tensor(cfg_ids, dtype=torch.long, device=device)
        else:
            cfg_ids = cfg_ids.to(device=device, dtype=torch.long)

        snr_db = meta["snr_db"]
        if not torch.is_tensor(snr_db):
            snr_db = torch.tensor(snr_db, dtype=torch.float32, device=device)
        else:
            snr_db = snr_db.to(device=device, dtype=torch.float32)

        return cfg_ids, snr_db

    # Fallback: older behavior (list of dicts)
    cfg_ids = torch.tensor([m["config_id"] for m in meta], device=device, dtype=torch.long)
    snr_db = torch.tensor([m["snr_db"] for m in meta], device=device, dtype=torch.float32)
    return cfg_ids, snr_db


# ----------------------------------------------------------------------
# Multi-config train/eval loops with per-config metrics
# ----------------------------------------------------------------------
def train_one_epoch_multi(model, train_loaders, optimizer, device,
                          lambda_comm=1.0, grad_clip=1.0):
    model.train()
    total_loss = total_radar = total_comm = 0.0
    total_ser = 0.0
    n_samples = 0

    per_cfg_stats = {
        cfg_name: {"loss": 0.0, "radar": 0.0, "comm": 0.0, "ser": 0.0, "n": 0}
        for cfg_name in train_loaders.keys()
    }

    for cfg_name, loader in train_loaders.items():
        for radar_in, radar_tgt, comm_in, comm_tgt, meta in loader:
            cfg_ids, snr_db = extract_meta_tensors(meta, device)

            radar_in = radar_in.to(device, non_blocking=True)
            radar_tgt = radar_tgt.to(device, non_blocking=True)
            comm_in = comm_in.to(device, non_blocking=True)
            comm_tgt = comm_tgt.to(device, non_blocking=True)

            # NEW: get modulation order from meta
            mod_order = meta["mod_order"].to(device).long()   # (B,)

            bsz = radar_in.size(0)
            optimizer.zero_grad()

            # radar_logits, comm_logits = model(
            #     radar_in,
            #     comm_in,
            #     config_ids=cfg_ids,
            #     snr_db=snr_db,
            #     mod_order=mod_order,
            #     mode="joint",   # or "radar_only"/"comm_only" if you ever want to switch
            # )
            radar_logits, comm_logits = model(
                radar_in, comm_in, config_ids=cfg_ids, snr_db=snr_db
            )
            loss, l_radar, l_comm = compute_losses(
                radar_logits, radar_tgt, comm_logits, comm_tgt,
                lambda_comm=lambda_comm,
            )
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()


            pred = comm_logits.argmax(dim=1)
            ser_batch = (pred != comm_tgt).float().mean().item()

            total_loss += loss.item() * bsz
            total_radar += l_radar.item() * bsz
            total_comm += l_comm.item() * bsz
            total_ser += ser_batch * bsz
            n_samples += bsz

            s = per_cfg_stats[cfg_name]
            s["loss"] += loss.item() * bsz
            s["radar"] += l_radar.item() * bsz
            s["comm"] += l_comm.item() * bsz
            s["ser"] += ser_batch * bsz
            s["n"] += bsz

    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, {}

    overall = (
        total_loss / n_samples,
        total_radar / n_samples,
        total_comm / n_samples,
        total_ser / n_samples,
    )

    per_cfg_avg = {}
    for cfg_name, s in per_cfg_stats.items():
        if s["n"] > 0:
            per_cfg_avg[cfg_name] = {
                "loss": s["loss"] / s["n"],
                "radar": s["radar"] / s["n"],
                "comm": s["comm"] / s["n"],
                "ser": s["ser"] / s["n"],
            }
        else:
            per_cfg_avg[cfg_name] = {
                "loss": 0.0,
                "radar": 0.0,
                "comm": 0.0,
                "ser": 0.0,
            }

    return *overall, per_cfg_avg


@torch.no_grad()
def evaluate_epoch_multi(model, val_loaders, device, lambda_comm=1.0):
    model.eval()
    total_loss = total_radar = total_comm = 0.0
    total_ser = 0.0
    n_samples = 0

    per_cfg_stats = {
        cfg_name: {"loss": 0.0, "radar": 0.0, "comm": 0.0, "ser": 0.0, "n": 0}
        for cfg_name in val_loaders.keys()
    }

    for cfg_name, loader in val_loaders.items():
        for radar_in, radar_tgt, comm_in, comm_tgt, meta in loader:
            cfg_ids, snr_db = extract_meta_tensors(meta, device)

            radar_in = radar_in.to(device, non_blocking=True)
            radar_tgt = radar_tgt.to(device, non_blocking=True)
            comm_in = comm_in.to(device, non_blocking=True)
            comm_tgt = comm_tgt.to(device, non_blocking=True)

            bsz = radar_in.size(0)

            radar_logits, comm_logits = model(
                radar_in, comm_in, config_ids=cfg_ids, snr_db=snr_db
            )
            # mod_order = meta["mod_order"].to(device).long()

            # with torch.no_grad():
            #     radar_logits, comm_logits = model(
            #         radar_in,
            #         comm_in,
            #         config_ids=cfg_ids,
            #         snr_db=snr_db,
            #         mod_order=mod_order,
            #         mode="joint",
            #     )
            loss, l_radar, l_comm = compute_losses(
                radar_logits, radar_tgt, comm_logits, comm_tgt,
                lambda_comm=lambda_comm,
            )

            pred = comm_logits.argmax(dim=1)
            ser_batch = (pred != comm_tgt).float().mean().item()

            total_loss += loss.item() * bsz
            total_radar += l_radar.item() * bsz
            total_comm += l_comm.item() * bsz
            total_ser += ser_batch * bsz
            n_samples += bsz

            s = per_cfg_stats[cfg_name]
            s["loss"] += loss.item() * bsz
            s["radar"] += l_radar.item() * bsz
            s["comm"] += l_comm.item() * bsz
            s["ser"] += ser_batch * bsz
            s["n"] += bsz

    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, {}

    overall = (
        total_loss / n_samples,
        total_radar / n_samples,
        total_comm / n_samples,
        total_ser / n_samples,
    )

    per_cfg_avg = {}
    for cfg_name, s in per_cfg_stats.items():
        if s["n"] > 0:
            per_cfg_avg[cfg_name] = {
                "loss": s["loss"] / s["n"],
                "radar": s["radar"] / s["n"],
                "comm": s["comm"] / s["n"],
                "ser": s["ser"] / s["n"],
            }
        else:
            per_cfg_avg[cfg_name] = {
                "loss": 0.0,
                "radar": 0.0,
                "comm": 0.0,
                "ser": 0.0,
            }

    return *overall, per_cfg_avg


# ----------------------------------------------------------------------
# Radar post-processing (DL heatmap -> detections)
# ----------------------------------------------------------------------
from scipy.ndimage import maximum_filter


def postprocess_radar_heatmap(probs, r_axis, v_axis, cfg,
                              prob_thresh=0.7):
    params = cfg.get("cfar_params", {})
    nms_kernel = params.get("nms_kernel_size", 5)
    min_r = params.get("min_range_m", 0.0)
    min_v = params.get("min_speed_mps", 0.0)
    notch_k = params.get("notch_doppler_bins", 0)
    max_peaks = params.get("max_peaks", None)

    local_max = maximum_filter(probs, size=nms_kernel)
    detections_mask = (probs >= prob_thresh) & (probs == local_max)

    idxs = np.argwhere(detections_mask)
    center = len(v_axis) // 2
    candidates = []

    for d_idx, r_idx in idxs:
        if d_idx >= len(v_axis) or r_idx >= len(r_axis):
            continue
        range_m = r_axis[r_idx]
        vel_mps = v_axis[d_idx]
        if range_m < min_r or abs(vel_mps) < min_v:
            continue
        if notch_k > 0 and abs(d_idx - center) <= notch_k:
            continue
        candidates.append({
            "range_m": float(range_m),
            "velocity_mps": float(vel_mps),
            "range_idx": int(r_idx),
            "doppler_idx": int(d_idx),
            "score": float(probs[d_idx, r_idx]),
        })

    if max_peaks is not None:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:max_peaks]

    pruned = []
    taken = set()
    neigh = params.get("nms_kernel_size", 5)
    for det in candidates:
        key = (det["doppler_idx"] // neigh, det["range_idx"] // neigh)
        if key in taken:
            continue
        taken.add(key)
        pruned.append(det)

    return pruned


def radar_metrics_from_dataset(base_ds, targets, detections):
    metrics, matched_pairs, unmatched_targets, unmatched_detections = base_ds._evaluate_metrics(
        targets, detections
    )
    return metrics, matched_pairs, unmatched_targets, unmatched_detections


# ----------------------------------------------------------------------
# Communication visualization
# ----------------------------------------------------------------------
def generate_qam_constellation(mod_order):
    if mod_order == 4:
        pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    elif mod_order == 16:
        x = np.arange(-3, 4, 2)
        X, Y = np.meshgrid(x, x)
        pts = (X + 1j * Y).flatten() / np.sqrt(10)
    elif mod_order == 64:
        x = np.arange(-7, 8, 2)
        X, Y = np.meshgrid(x, x)
        pts = (X + 1j * Y).flatten() / np.sqrt(42)
    else:
        raise ValueError(f"Unsupported mod_order {mod_order}")
    return pts


def plot_eye_diagram(symbols, sps=2, save_path=None):
    x = np.real(symbols)
    seq = np.repeat(x, sps)
    seg_len = 4 * sps
    n_seg = len(seq) // seg_len

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(n_seg):
        seg = seq[i * seg_len:(i + 1) * seg_len]
        t = np.arange(len(seg)) / sps
        ax.plot(t, seg, alpha=0.2)

    ax.set_xlabel("Symbol time")
    ax.set_ylabel("Amplitude (I)")
    ax.set_title("Eye Diagram (approx)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        return fig, ax


def plot_comm_results(comm_info, pred_ints, mod_order, dl_ser, save_prefix):
    tx_syms = np.array(comm_info["tx_symbols"])
    rx_syms = np.array(comm_info["rx_symbols"])
    const_pts = generate_qam_constellation(mod_order)
    tx_ints = np.array(comm_info["tx_ints"], dtype=int)

    tx_pts = const_pts[tx_ints]
    pred_ints_clipped = np.clip(pred_ints.astype(int), 0, mod_order - 1)
    pred_pts = const_pts[pred_ints_clipped]

    baseline_ber = comm_info.get("ber", 0.0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax0 = ax[0]
    idx = np.arange(len(tx_syms))
    if len(idx) > 2000:
        sel = np.random.choice(idx, 2000, replace=False)
    else:
        sel = idx

    ax0.scatter(np.real(rx_syms[sel]), np.imag(rx_syms[sel]),
                s=8, alpha=0.4, label="Rx (ZF+LS)")
    ax0.scatter(np.real(tx_syms[sel]), np.imag(tx_syms[sel]),
                s=12, alpha=0.6, marker="x", label="Tx")
    ax0.scatter(np.real(pred_pts[sel]), np.imag(pred_pts[sel]),
                s=10, alpha=0.5, marker="+", label="DL Demap")

    ax0.set_title(f"{mod_order}-QAM Constellation")
    ax0.set_xlabel("I")
    ax0.set_ylabel("Q")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8)
    ax0.set_aspect("equal")

    eye_path = save_prefix + "_eye.png"
    plot_eye_diagram(rx_syms, sps=2, save_path=eye_path)
    img = plt.imread(eye_path)
    ax1 = ax[1]
    ax1.imshow(img)
    ax1.axis("off")
    text = (
        f"Baseline BER: {baseline_ber:.3e}\n"
        f"DL SER≈BER: {dl_ser:.3e}\n"
        f"#Symbols: {len(tx_ints)}"
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax1.text(0.02, 0.98, text, transform=ax1.transAxes,
             fontsize=9, verticalalignment="top",
             bbox=props, family="monospace")

    plt.tight_layout()
    const_path = save_prefix + "_constellation_eye.png"
    plt.savefig(const_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Radar visualization (2D & 3D)
# ----------------------------------------------------------------------
def plot_radar_2d_comparison(rdm_db, r_axis, v_axis,
                             targets, cfar_dets, dl_dets,
                             metrics_cfar, metrics_dl,
                             save_path):
    fig, ax = plt.subplots(figsize=(12, 8))

    if len(r_axis) > 1:
        dr = r_axis[1] - r_axis[0]
    else:
        dr = 1.0
    if len(v_axis) > 1:
        dv = v_axis[1] - v_axis[0]
    else:
        dv = 1.0

    extent = [r_axis[0] - dr / 2, r_axis[-1] + dr / 2,
              v_axis[0] - dv / 2, v_axis[-1] + dv / 2]

    im = ax.imshow(rdm_db, extent=extent, origin="lower",
                   cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")

    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Range-Doppler Map: CFAR vs Deep Model")

    for t in targets:
        ax.scatter(t["range"], t["velocity"],
                   facecolors="none", edgecolors="lime",
                   s=150, linewidth=2, label="GT")

    for d in cfar_dets:
        ax.scatter(d["range_m"], d["velocity_mps"], marker="x", color="cyan",
                   s=80, linewidth=2, label="CFAR")

    for d in dl_dets:
        ax.scatter(d["range_m"], d["velocity_mps"], marker="+", color="red",
                   s=80, linewidth=2, label="DL")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    text = (
        "CFAR Metrics:\n"
        f"  TP={metrics_cfar['tp']} FP={metrics_cfar['fp']} FN={metrics_cfar['fn']}\n"
        f"  dR={metrics_cfar['mean_range_error']:.2f} m, dV={metrics_cfar['mean_velocity_error']:.2f} m/s\n"
        "\nDL Metrics:\n"
        f"  TP={metrics_dl['tp']} FP={metrics_dl['fp']} FN={metrics_dl['fn']}\n"
        f"  dR={metrics_dl['mean_range_error']:.2f} m, dV={metrics_dl['mean_velocity_error']:.2f} m/s\n"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.6)
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", bbox=props,
            family="monospace")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_radar_3d(rdm_db, r_axis, v_axis, targets, detections, save_path):
    if VISUALIZATION_AVAILABLE:
        if len(r_axis) > 1:
            range_res = r_axis[1] - r_axis[0]
        else:
            range_res = 1.0
        if len(v_axis) > 1:
            vel_res = v_axis[1] - v_axis[0]
        else:
            vel_res = 1.0

        converted_targets = []
        for t in targets:
            ct = t.copy()
            ct["distance"] = t["range"]
            converted_targets.append(ct)

        cleaned_dets = []
        for d in detections:
            d2 = d.copy()
            d2["range_idx"] = int(d2.get("range_idx", 0))
            d2["doppler_idx"] = int(d2.get("doppler_idx", 0))
            cleaned_dets.append(d2)

        plot_3d_range_doppler_map_with_ground_truth(
            rd_map=rdm_db,
            targets=converted_targets,
            range_resolution=range_res,
            velocity_resolution=vel_res,
            num_range_bins=rdm_db.shape[1],
            num_doppler_bins=rdm_db.shape[0],
            save_path=save_path,
            apply_doppler_centering=True,
            detections=cleaned_dets,
            view_range_limits=(r_axis[0], r_axis[-1]),
            view_velocity_limits=(v_axis[0], v_axis[-1]),
            is_db=True,
            stride=4,
        )
    else:
        R_grid, D_grid = np.meshgrid(r_axis, v_axis)
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(R_grid, D_grid, rdm_db, cmap="viridis")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_zlabel("Mag (dB)")
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


# ----------------------------------------------------------------------
# SNR sweep & radar threshold curves
# ----------------------------------------------------------------------
def make_snr_sweep_plots(per_sample_stats, out_dir):
    snrs = np.array([s["snr_db"] for s in per_sample_stats], dtype=np.float32)

    def f1_from_m(metrics):
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    cfar_prec_all, cfar_rec_all, cfar_f1_all = [], [], []
    dl_prec_all, dl_rec_all, dl_f1_all = [], [], []
    baseline_ber_all, dl_ser_all = [], []

    for s in per_sample_stats:
        m_c = s["metrics_cfar"]
        m_d = s["metrics_dl"]
        p_c, r_c, f_c = f1_from_m(m_c)
        p_d, r_d, f_d = f1_from_m(m_d)
        cfar_prec_all.append(p_c)
        cfar_rec_all.append(r_c)
        cfar_f1_all.append(f_c)
        dl_prec_all.append(p_d)
        dl_rec_all.append(r_d)
        dl_f1_all.append(f_d)
        baseline_ber_all.append(s["baseline_ber"])
        dl_ser_all.append(s["dl_ser"])

    cfar_prec_all = np.array(cfar_prec_all)
    cfar_rec_all = np.array(cfar_rec_all)
    cfar_f1_all = np.array(cfar_f1_all)
    dl_prec_all = np.array(dl_prec_all)
    dl_rec_all = np.array(dl_rec_all)
    dl_f1_all = np.array(dl_f1_all)
    baseline_ber_all = np.array(baseline_ber_all)
    dl_ser_all = np.array(dl_ser_all)

    if len(snrs) == 0:
        return

    snr_min = math.floor(snrs.min())
    snr_max = math.ceil(snrs.max())
    bin_edges = np.arange(snr_min, snr_max + 1, 1.0)
    if len(bin_edges) < 2:
        bin_edges = np.array([snr_min - 0.5, snr_max + 0.5])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def bin_avg(values):
        out = []
        for i in range(len(bin_edges) - 1):
            mask = (snrs >= bin_edges[i]) & (snrs < bin_edges[i + 1])
            if np.any(mask):
                out.append(values[mask].mean())
            else:
                out.append(np.nan)
        return np.array(out)

    cfar_f1_b = bin_avg(cfar_f1_all)
    dl_f1_b = bin_avg(dl_f1_all)
    dl_prec_b = bin_avg(dl_prec_all)
    dl_rec_b = bin_avg(dl_rec_all)
    base_ber_b = bin_avg(baseline_ber_all)
    dl_ser_b = bin_avg(dl_ser_all)

    # 1) Radar F1 vs SNR
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, cfar_f1_b, "o-", label="CFAR F1")
    plt.plot(bin_centers, dl_f1_b, "o-", label="Deep Radar F1")
    plt.xlabel("SNR (dB)")
    plt.ylabel("F1 Score")
    plt.title("Radar F1 vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snr_sweep_radar_f1.png"), dpi=150)
    plt.close()

    # 2) Deep Radar Precision/Recall vs SNR
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, dl_prec_b, "o-", label="Precision")
    plt.plot(bin_centers, dl_rec_b, "o-", label="Recall")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Metric")
    plt.title("Deep Radar Precision/Recall vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snr_sweep_radar_prec_rec.png"), dpi=150)
    plt.close()

    # 3) Comm BER vs SNR (log scale)
    plt.figure(figsize=(8, 5))
    plt.semilogy(bin_centers, base_ber_b, "o-", label="Baseline BER")
    plt.semilogy(bin_centers, dl_ser_b, "o-", label="Deep SER≈BER")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER / SER")
    plt.title("Communication BER vs SNR")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snr_sweep_comm_ber.png"), dpi=150)
    plt.close()


def make_radar_threshold_curves(radar_probs_all, r_axes_all, v_axes_all,
                                targets_all, base_ds, cfg, out_dir):
    thresholds = np.linspace(0.1, 0.9, 17)

    recalls = []
    precisions = []
    avg_fp_per_frame = []

    for T in thresholds:
        total_tp = total_fp = total_fn = total_targets = 0
        for probs, r_axis, v_axis, targets in zip(
            radar_probs_all, r_axes_all, v_axes_all, targets_all
        ):
            dl_dets = postprocess_radar_heatmap(
                probs, r_axis, v_axis, cfg, prob_thresh=float(T)
            )
            metrics, _, _, _ = base_ds._evaluate_metrics(targets, dl_dets)
            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]
            total_targets += metrics["total_targets"]

        recall = total_tp / total_targets if total_targets > 0 else 0.0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recalls.append(recall)
        precisions.append(precision)
        avg_fp_per_frame.append(total_fp / len(radar_probs_all))

    recalls = np.array(recalls)
    precisions = np.array(precisions)
    avg_fp_per_frame = np.array(avg_fp_per_frame)

    # 1) ROC-like: Recall vs Average FP/frame
    plt.figure(figsize=(8, 5))
    plt.plot(avg_fp_per_frame, recalls, "o-")
    for T, x, y in zip(thresholds, avg_fp_per_frame, recalls):
        plt.annotate(f"{T:.2f}", (x, y),
                     textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.xlabel("Average FP per frame")
    plt.ylabel("Recall (TPR)")
    plt.title("Deep Radar Detection Curve (Recall vs FP/frame)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "radar_detection_curve_recall_vs_fp.png"), dpi=150)
    plt.close()

    # 2) Threshold vs Precision/Recall
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, "o-", label="Precision")
    plt.plot(thresholds, recalls, "o-", label="Recall")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Metric")
    plt.title("Deep Radar Precision/Recall vs Threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "radar_prec_rec_vs_threshold.png"), dpi=150)
    plt.close()

    with open(os.path.join(out_dir, "radar_threshold_curve.txt"), "w") as f:
        f.write("# T  precision  recall  avg_fp_per_frame\n")
        for T, p, r, fp in zip(thresholds, precisions, recalls, avg_fp_per_frame):
            f.write(f"{T:.3f} {p:.6f} {r:.6f} {fp:.6f}\n")


# ----------------------------------------------------------------------
# Full evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def run_full_evaluation(model, deep_ds, base_ds, cfg, device, out_dir,
                        prob_thresh=0.7):
    os.makedirs(out_dir, exist_ok=True)

    total_cfar = {"tp": 0, "fp": 0, "fn": 0, "range_err": [], "vel_err": [], "targets": 0}
    total_dl = {"tp": 0, "fp": 0, "fn": 0, "range_err": [], "vel_err": [], "targets": 0}
    baseline_bers = []
    dl_sers = []

    per_sample_stats = []
    radar_probs_all = []
    r_axes_all = []
    v_axes_all = []
    targets_all = []

    for idx in range(len(deep_ds)):
        s = base_ds[idx]
        rdm = s["range_doppler_map"].numpy().astype(np.float32)
        r_axis = np.asarray(s["range_axis"])
        v_axis = np.asarray(s["velocity_axis"])
        targets = s["target_info"]["targets"]
        cfar_dets = s["cfar_detections"]
        comm_info = s["comm_info"]
        snr_db = float(s["target_info"].get("snr_db", 0.0))
        mod_order = comm_info["mod_order"]

        metrics_cfar, _, _, _ = radar_metrics_from_dataset(base_ds, targets, cfar_dets)
        total_cfar["tp"] += metrics_cfar["tp"]
        total_cfar["fp"] += metrics_cfar["fp"]
        total_cfar["fn"] += metrics_cfar["fn"]
        total_cfar["targets"] += metrics_cfar["total_targets"]
        total_cfar["range_err"].append(metrics_cfar["mean_range_error"])
        total_cfar["vel_err"].append(metrics_cfar["mean_velocity_error"])

        radar_in, _, comm_in, comm_tgt, meta = deep_ds[idx]
        cfg_id_tensor = torch.tensor([meta["config_id"]], device=device, dtype=torch.long)
        snr_tensor = torch.tensor([meta["snr_db"]], device=device, dtype=torch.float32)

        radar_in_b = radar_in.unsqueeze(0).to(device)
        comm_in_b = comm_in.unsqueeze(0).to(device)
        comm_tgt_b = comm_tgt.unsqueeze(0).to(device)

        model.eval()
        radar_logits, comm_logits = model(
            radar_in_b, comm_in_b, config_ids=cfg_id_tensor, snr_db=snr_tensor
        )
        radar_probs = torch.sigmoid(radar_logits)[0, 0].cpu().numpy()
        dl_dets = postprocess_radar_heatmap(radar_probs, r_axis, v_axis, cfg,
                                            prob_thresh=prob_thresh)

        metrics_dl, _, _, _ = radar_metrics_from_dataset(base_ds, targets, dl_dets)
        total_dl["tp"] += metrics_dl["tp"]
        total_dl["fp"] += metrics_dl["fp"]
        total_dl["fn"] += metrics_dl["fn"]
        total_dl["targets"] += metrics_dl["total_targets"]
        total_dl["range_err"].append(metrics_dl["mean_range_error"])
        total_dl["vel_err"].append(metrics_dl["mean_velocity_error"])

        baseline_ber = float(comm_info.get("ber", 0.0))
        baseline_bers.append(baseline_ber)

        pred_ints = comm_logits.argmax(dim=1)[0].cpu().numpy().reshape(-1)
        gt_ints = comm_tgt_b.cpu().numpy().reshape(-1)
        ser = float((pred_ints != gt_ints).mean())
        dl_sers.append(ser)

        per_sample_stats.append({
            "snr_db": snr_db,
            "metrics_cfar": metrics_cfar,
            "metrics_dl": metrics_dl,
            "baseline_ber": baseline_ber,
            "dl_ser": ser,
        })
        radar_probs_all.append(radar_probs)
        r_axes_all.append(r_axis)
        v_axes_all.append(v_axis)
        targets_all.append(targets)

        if idx < 10:
            prefix = os.path.join(out_dir, f"sample_{idx:03d}")
            rdm_norm = rdm - np.max(rdm)

            plot_radar_2d_comparison(
                rdm_norm, r_axis, v_axis,
                targets, cfar_dets, dl_dets,
                metrics_cfar, metrics_dl,
                save_path=prefix + "_radar_2d.png",
            )

            plot_radar_3d(
                rdm_norm, r_axis, v_axis,
                targets, dl_dets,
                save_path=prefix + "_radar_3d_dl.png",
            )

            plot_comm_results(comm_info, pred_ints, mod_order, ser,
                              save_prefix=prefix + "_comm")

    def agg(total):
        tp = total["tp"]
        fp = total["fp"]
        fn = total["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_range = float(np.mean(total["range_err"])) if total["range_err"] else 0.0
        mean_vel = float(np.mean(total["vel_err"])) if total["vel_err"] else 0.0
        return precision, recall, f1, mean_range, mean_vel

    p_cfar, r_cfar, f1_cfar, mr_cfar, mv_cfar = agg(total_cfar)
    p_dl, r_dl, f1_dl, mr_dl, mv_dl = agg(total_dl)
    mean_baseline_ber = float(np.mean(baseline_bers)) if baseline_bers else 0.0
    mean_dl_ser = float(np.mean(dl_sers)) if dl_sers else 0.0

    summary = []
    summary.append("=== Radar Metrics (Classical CFAR) ===")
    summary.append(f"Targets: {total_cfar['targets']}")
    summary.append(f"TP={total_cfar['tp']} FP={total_cfar['fp']} FN={total_cfar['fn']}")
    summary.append(f"Precision={p_cfar:.4f} Recall={r_cfar:.4f} F1={f1_cfar:.4f}")
    summary.append(f"Mean Range Error={mr_cfar:.3f} m")
    summary.append(f"Mean Velocity Error={mv_cfar:.3f} m/s")
    summary.append("")
    summary.append("=== Radar Metrics (Deep Model) ===")
    summary.append(f"Targets: {total_dl['targets']}")
    summary.append(f"TP={total_dl['tp']} FP={total_dl['fp']} FN={total_dl['fn']}")
    summary.append(f"Precision={p_dl:.4f} Recall={r_dl:.4f} F1={f1_dl:.4f}")
    summary.append(f"Mean Range Error={mr_dl:.3f} m")
    summary.append(f"Mean Velocity Error={mv_dl:.3f} m/s")
    summary.append("")
    summary.append("=== Communication Metrics ===")
    summary.append(f"Baseline Mean BER={mean_baseline_ber:.5e}")
    summary.append(f"Deep Model SER≈BER={mean_dl_ser:.5e}")

    txt = "\n".join(summary)
    print(txt)

    with open(os.path.join(out_dir, "evaluation_summary.txt"), "w") as f:
        f.write(txt)

    make_snr_sweep_plots(per_sample_stats, out_dir)
    make_radar_threshold_curves(
        radar_probs_all, r_axes_all, v_axes_all,
        targets_all, base_ds, cfg, out_dir
    )
# ----------------------------------------------------------------------
# Utilities & main
# ----------------------------------------------------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def quick_sanity_test(args):
    """
    Minimal end-to-end sanity check:

    1) Load or generate a very small dataset for one config.
    2) Wrap it with RadarCommDeepDataset.
    3) Build JointRadarCommNet.
    4) Run a single forward pass and compute loss + SER.
    """

    print("[Test] Running quick sanity check...")
    device = torch.device(args.device)

    # Use the first training config for the test
    cfg_name = TRAIN_CONFIGS[0]
    print(f"[Test] Using config: {cfg_name}")

    # Small dataset: just a few samples so it's fast
    base_ds = load_or_generate_base_dataset(
        cfg_name,
        split="train",
        num_samples=4,
        data_root=args.data_root,
        drawfig=False,
    )
    deep_ds = RadarCommDeepDataset(base_ds, cfg_name)

    loader = DataLoader(
        deep_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Build model (same hyperparams as in main train path)
    model = JointRadarCommNet_v2(
        num_configs=len(RADAR_COMM_CONFIGS),
        max_mod_order=MAX_MOD_ORDER,
        base_ch=48,
        num_blocks=4,
        cond_dim=16,
        use_shared_raw=True,   # start OFF
        use_fusion=False,       # start OFF
    ).to(device)
    model.eval()

    # Grab a single batch
    radar_in, radar_tgt, comm_in, comm_tgt, meta = next(iter(loader))

    print("[Test] Batch shapes:")
    print(f"  radar_in:  {tuple(radar_in.shape)}")
    print(f"  radar_tgt: {tuple(radar_tgt.shape)}")
    print(f"  comm_in:   {tuple(comm_in.shape)}")
    print(f"  comm_tgt:  {tuple(comm_tgt.shape)}")
    print(f"  meta (example): {meta}")

    # Move to device
    radar_in = radar_in.to(device, non_blocking=True)
    radar_tgt = radar_tgt.to(device, non_blocking=True)
    comm_in = comm_in.to(device, non_blocking=True)
    comm_tgt = comm_tgt.to(device, non_blocking=True)

    # Extract config IDs and SNR from meta
    cfg_ids, snr_db = extract_meta_tensors(meta, device)

    # Forward
    with torch.no_grad():
        radar_logits, comm_logits = model(
            radar_in,
            comm_in,
            config_ids=cfg_ids,
            snr_db=snr_db,
        )
        print("[Test] Output shapes:")
        print(f"  radar_logits: {tuple(radar_logits.shape)}")
        print(f"  comm_logits:  {tuple(comm_logits.shape)}")

        loss, l_radar, l_comm = compute_losses(
            radar_logits,
            radar_tgt,
            comm_logits,
            comm_tgt,
            lambda_comm=args.lambda_comm,
        )

        pred = comm_logits.argmax(dim=1)
        ser = (pred != comm_tgt).float().mean().item()

    print("[Test] Loss summary:")
    print(f"  total_loss = {loss.item():.4e}")
    print(f"  radar_loss = {l_radar.item():.4e}")
    print(f"  comm_loss  = {l_comm.item():.4e}")
    print(f"  SER≈BER    = {ser:.4e}")
    print("[Test] Quick sanity check finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "inference", "test"],
                        default="train")
    parser.add_argument("--train_samples_per_config", type=int, default=400)
    parser.add_argument("--val_samples_per_config", type=int, default=100)
    parser.add_argument("--test_samples_per_config", type=int, default=100)
    parser.add_argument("--data_root", type=str,
                        default="data/AIradar_comm_model_g2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_comm", type=float, default=2.0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="data/AIradar_comm_model_g3")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--prob_thresh", type=float, default=0.7)
    parser.add_argument("--draw_fig_gen", default=True,
                        help="If set, simulator will also draw its own figures (slower).")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    
    # ------------------------ TEST MODE (quick sanity check) ------------------------
    if args.mode == "test":
        quick_sanity_test(args)
        return

    # ------------------- Build train & val datasets/loaders (per config) -------------------
    train_loaders = {}
    base_train_datasets = {}

    for cfg_name in TRAIN_CONFIGS:
        cfg = RADAR_COMM_CONFIGS[cfg_name]
        assert cfg["mode"] == "TRADITIONAL", f"Config {cfg_name} must be TRADITIONAL."
        base_ds_train = load_or_generate_base_dataset(
            cfg_name, "train", args.train_samples_per_config,
            data_root=args.data_root, drawfig=args.draw_fig_gen,
        )
        base_train_datasets[cfg_name] = base_ds_train
        deep_train_ds = RadarCommDeepDataset(base_ds_train, cfg_name)
        train_loaders[cfg_name] = DataLoader(
            deep_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    val_loaders = {}
    base_val_datasets = {}

    for cfg_name in VAL_CONFIGS:
        cfg = RADAR_COMM_CONFIGS[cfg_name]
        assert cfg["mode"] == "TRADITIONAL", f"Config {cfg_name} must be TRADITIONAL."
        base_ds_val = load_or_generate_base_dataset(
            cfg_name, "val", args.val_samples_per_config,
            data_root=args.data_root, drawfig=False,
        )
        base_val_datasets[cfg_name] = base_ds_val
        deep_val_ds = RadarCommDeepDataset(base_ds_val, cfg_name)
        val_loaders[cfg_name] = DataLoader(
            deep_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
    
    print("\n=== SNR diagnostics per config/split ===")
    for cfg_name in TRAIN_CONFIGS:
        summarize_snr_for_split(
            base_train_datasets[cfg_name],
            f"{cfg_name} train",
            max_samples=200,
        )
        summarize_snr_for_split(
            base_val_datasets[cfg_name],
            f"{cfg_name} val",
            max_samples=200,
        )
    print("========================================\n")

    # ------------------------ Build model & optimizer ------------------------
    model = JointRadarCommNet_v2(
        num_configs=len(RADAR_COMM_CONFIGS),
        max_mod_order=MAX_MOD_ORDER,
        base_ch=48,
        num_blocks=4,
        cond_dim=16,
        use_shared_raw=True,   # start OFF
        use_fusion=False,       # start OFF
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # -------------------------- TRAIN MODE --------------------------
    if args.mode == "train":
        os.makedirs(args.out_dir, exist_ok=True)
        best_val_loss = float("inf")
        best_path = None

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_radar, tr_comm, tr_ser, tr_per_cfg = train_one_epoch_multi(
                model, train_loaders, optimizer, device,
                lambda_comm=args.lambda_comm,
            )
            val_loss, val_radar, val_comm, val_ser, val_per_cfg = evaluate_epoch_multi(
                model, val_loaders, device,
                lambda_comm=args.lambda_comm,
            )

            print(
                f"[Epoch {epoch:02d}] "
                f"TrainLoss={tr_loss:.4f} (Radar={tr_radar:.4f}, Comm={tr_comm:.4f}), "
                f"Train SER≈BER={tr_ser:.4e} | "
                f"ValLoss={val_loss:.4f} (Radar={val_radar:.4f}, Comm={val_comm:.4f}), "
                f"Val SER≈BER={val_ser:.4e}"
            )

            # Per-config logs
            for cfg_name in TRAIN_CONFIGS:
                if cfg_name in tr_per_cfg:
                    s_tr = tr_per_cfg[cfg_name]
                    s_val = val_per_cfg.get(cfg_name, {"loss": 0, "radar": 0, "comm": 0, "ser": 0})
                    print(
                        f"  [Epoch {epoch:02d}] {cfg_name}: "
                        f"Train(L={s_tr['loss']:.4f}, R={s_tr['radar']:.4f}, C={s_tr['comm']:.4f}, SER={s_tr['ser']:.4e}) | "
                        f"Val(L={s_val['loss']:.4f}, R={s_val['radar']:.4f}, C={s_val['comm']:.4f}, SER={s_val['ser']:.4e})"
                    )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.out_dir, "joint_net_generalized_best.pt")
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "train_configs": TRAIN_CONFIGS,
                        "val_configs": VAL_CONFIGS,
                        "test_configs": TEST_CONFIGS,
                    },
                    best_path,
                )
                print(f"  -> New best model saved to {best_path}")

        eval_configs = sorted(set(TRAIN_CONFIGS + TEST_CONFIGS))
        print("\n=== Running full evaluation on configs:", eval_configs, "===\n")
        if best_path is not None:
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])

        for cfg_name in eval_configs:
            cfg = RADAR_COMM_CONFIGS[cfg_name]
            if cfg["mode"] != "TRADITIONAL":
                print(f"[Eval] Skipping {cfg_name} (non-TRADITIONAL).")
                continue

            base_test_ds = load_or_generate_base_dataset(
                cfg_name, "test", args.test_samples_per_config,
                data_root=args.data_root, drawfig=False,
            )
            deep_test_ds = RadarCommDeepDataset(base_test_ds, cfg_name)
            eval_dir = os.path.join(args.out_dir, f"eval_{cfg_name}")
            run_full_evaluation(
                model, deep_test_ds, base_test_ds, cfg,
                device, eval_dir, prob_thresh=args.prob_thresh,
            )

    # ------------------------ EVALUATE MODE -------------------------
    elif args.mode == "evaluate":
        assert args.ckpt is not None, "Provide --ckpt for evaluate mode."
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        eval_configs = sorted(set(TRAIN_CONFIGS + TEST_CONFIGS))
        print("\n=== Running full evaluation on configs:", eval_configs, "===\n")

        for cfg_name in eval_configs:
            cfg = RADAR_COMM_CONFIGS[cfg_name]
            if cfg["mode"] != "TRADITIONAL":
                print(f"[Eval] Skipping {cfg_name} (non-TRADITIONAL).")
                continue

            base_test_ds = load_or_generate_base_dataset(
                cfg_name, "test", args.test_samples_per_config,
                data_root=args.data_root, drawfig=False,
            )
            deep_test_ds = RadarCommDeepDataset(base_test_ds, cfg_name)
            eval_dir = os.path.join(args.out_dir, f"eval_{cfg_name}")
            run_full_evaluation(
                model, deep_test_ds, base_test_ds, cfg,
                device, eval_dir, prob_thresh=args.prob_thresh,
            )

    # ------------------------ INFERENCE MODE ------------------------
    elif args.mode == "inference":
        assert args.ckpt is not None, "Provide --ckpt for inference mode."
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()

        cfg_name = TRAIN_CONFIGS[0]
        cfg = RADAR_COMM_CONFIGS[cfg_name]

        base_inf_ds = load_or_generate_base_dataset(
            cfg_name, "inference", args.test_samples_per_config,
            data_root=args.data_root, drawfig=False,
        )
        deep_inf_ds = RadarCommDeepDataset(base_inf_ds, cfg_name)

        idx = args.sample_idx
        if idx < 0 or idx >= len(deep_inf_ds):
            raise IndexError(f"sample_idx {idx} out of range [0, {len(deep_inf_ds) - 1}]")

        radar_in, radar_tgt, comm_in, comm_tgt, meta = deep_inf_ds[idx]
        s = base_inf_ds[idx]
        rdm = s["range_doppler_map"].numpy().astype(np.float32)
        r_axis = np.asarray(s["range_axis"])
        v_axis = np.asarray(s["velocity_axis"])
        targets = s["target_info"]["targets"]
        cfar_dets = s["cfar_detections"]
        comm_info = s["comm_info"]
        mod_order = comm_info["mod_order"]

        out_dir = os.path.join(args.out_dir, f"inference_{cfg_name}_sample_{idx:03d}")
        os.makedirs(out_dir, exist_ok=True)

        radar_in_b = radar_in.unsqueeze(0).to(device)
        comm_in_b = comm_in.unsqueeze(0).to(device)
        comm_tgt_b = comm_tgt.unsqueeze(0).to(device)

        cfg_id_tensor = torch.tensor([meta["config_id"]], device=device, dtype=torch.long)
        snr_tensor = torch.tensor([meta["snr_db"]], device=device, dtype=torch.float32)

        radar_logits, comm_logits = model(
            radar_in_b, comm_in_b, config_ids=cfg_id_tensor, snr_db=snr_tensor
        )
        radar_probs = torch.sigmoid(radar_logits)[0, 0].cpu().numpy()
        dl_dets = postprocess_radar_heatmap(radar_probs, r_axis, v_axis, cfg,
                                            prob_thresh=args.prob_thresh)

        metrics_cfar, _, _, _ = radar_metrics_from_dataset(base_inf_ds, targets, cfar_dets)
        metrics_dl, _, _, _ = radar_metrics_from_dataset(base_inf_ds, targets, dl_dets)

        rdm_norm = rdm - np.max(rdm)
        prefix = os.path.join(out_dir, f"sample_{idx:03d}")

        plot_radar_2d_comparison(
            rdm_norm, r_axis, v_axis,
            targets, cfar_dets, dl_dets,
            metrics_cfar, metrics_dl,
            save_path=prefix + "_radar_2d.png",
        )

        plot_radar_3d(
            rdm_norm, r_axis, v_axis,
            targets, dl_dets,
            save_path=prefix + "_radar_3d_dl.png",
        )

        pred_ints = comm_logits.argmax(dim=1)[0].cpu().numpy().reshape(-1)
        gt_ints = comm_tgt_b.cpu().numpy().reshape(-1)
        ser = float((pred_ints != gt_ints).mean())

        plot_comm_results(comm_info, pred_ints, mod_order, ser,
                          save_prefix=prefix + "_comm")

        with open(os.path.join(out_dir, "inference_summary.txt"), "w") as f:
            f.write("Inference summary\n")
            f.write("CFAR metrics:\n")
            f.write(str(metrics_cfar) + "\n")
            f.write("Deep model metrics:\n")
            f.write(str(metrics_dl) + "\n")
            f.write(f"CFAR baseline BER={comm_info.get('ber', 0.0):.5e}\n")
            f.write(f"Deep model SER≈BER={ser:.5e}\n")


if __name__ == "__main__":
    main()