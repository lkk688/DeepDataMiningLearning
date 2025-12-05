import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from scipy.constants import c
import torch
from torch.utils.data import Dataset
import h5py
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # kept for external imports
from matplotlib import cm  # noqa: F401

# Import visualization functions
try:
    from AIradar_visualization import (
        plot_signal_time_and_spectrum,
        plot_instantaneous_frequency,
        plot_range_doppler_map_with_ground_truth,
        plot_3d_range_doppler_map_with_ground_truth
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: AIradar_visualization not available. Some visualizations will be skipped.")
    VISUALIZATION_AVAILABLE = False

# Import CFAR detection function
try:
    #from AIRadarLib.radar_det import cfar_2d_numpy
    from AIradar_det import cfar_2d_numpy
    CFAR_AVAILABLE = True
except ImportError:
    print("Warning: AIradar_det not available. CFAR detection will be skipped.")
    CFAR_AVAILABLE = False

# --- Constants ---
VIEW_RANGE_LIMITS = (0, 100)
VIEW_VELOCITY_LIMITS = (-48, 48)

# --- Radar Configurations ---
RADAR_CONFIGS = {
    'config1': {
        'name': 'Automotive_77GHz_LongRange',
        'signal_type': 'FMCW',  # Default signal type
        'fc': 77e9,             # 77 GHz
        'B': 1.5e9,             # 1.5 GHz Bandwidth
        'T_chirp': 40e-6,       # 40 μs
        'fs': 51.2e6,           # 51.2 MHz Sampling Rate
        'N_chirps': 128,        # Number of chirps
        'R_max': 100.0,         # Max range 100m
        'description': 'Standard automotive long-range radar configuration',
        'hardware_model': 'generic',
        'num_rx': 1,
        'use_array_factor': False,
        'cfar_params': {        # Tuned for high resolution
            'num_train': 10,
            'num_guard': 4,
            'threshold_offset': 15,
            'nms_kernel_size': 5
        }
    },
    'config2': {
        'name': 'XBand_10GHz_MediumRange',
        'signal_type': 'FMCW',
        'fc': 10e9,             # 10 GHz (X-band)
        'B': 1.0e9,             # 1 GHz Bandwidth
        'T_chirp': 160e-6,      # 160 μs (Increased to reduce v_max)
        'fs': 40e6,             # 40 MHz Sampling Rate (Lower ADC requirement)
        'N_chirps': 128,        # Number of chirps
        'R_max': 100.0,         # Max range 100m
        'description': 'X-band radar for medium range surveillance or robotics',
        'hardware_model': 'generic',
        'num_rx': 1,
        'use_array_factor': False,
        'cfar_params': {        # Tuned for lower resolution/SNR
            'num_train': 24,      # Increased training cells for more stable noise estimate
            'num_guard': 8,       # Increased guard cells to avoid target self-masking
            'threshold_offset': 18, # Adjusted to balance Precision/Recall
            'nms_kernel_size': 7  # Increased NMS kernel to merge clustered detections
        }
    },
    'config_otfs': {
        'name': 'OTFS_Automotive_77GHz',
        'signal_type': 'OTFS',
        'fc': 77e9,             # 77 GHz
        'B': 1.536e9,           # ~1.5 GHz (aligned with OTFS/OFDM-style BW)
        'T_chirp': 40e-6,       # Symbol duration
        'fs': 51.2e6,           # Sampling rate
        'N_chirps': 128,        # Number of OTFS symbols (Doppler bins)
        'N_samples': 512,       # Number of subcarriers (Delay bins)
        'R_max': 100.0,
        'description': 'OTFS Radar configuration (generic automotive)',
        'hardware_model': 'generic',
        'num_rx': 1,
        'use_array_factor': False,
        'cfar_params': {
            'num_train': 16,        # Increased from 10 to 16 for better noise estimation
            'num_guard': 8,         # Increased from 4 to 8 to avoid self-masking
            'threshold_offset': 25, # Increased from 15 to 25 to reduce false positives
            'nms_kernel_size': 9    # Increased from 5 to 9 to merge nearby detections
        }
    },
    'config_phaser': {
        'name': 'Phaser_10GHz_DevKit',
        'signal_type': 'FMCW',
        'fc': 10e9,             # 10 GHz (X-band)
        'B': 500e6,             # 500 MHz Bandwidth (Reference: default_chirp_bw)
        'T_chirp': 500e-6,      # 500 μs (Reference: ramp_time)
        'fs': 2.0e6,            # 2 MHz Sampling Rate (Reference: sample_rate)
        'N_chirps': 64,         # Number of chirps
        'R_max': 100.0,         # Max range ~100m
        'description': 'Phaser Dev Kit configuration (simple model)',
        'hardware_model': 'generic',
        'num_rx': 1,
        'use_array_factor': False,
        'cfar_params': {
            'num_train': 10,
            'num_guard': 4,
            'threshold_offset': 15,
            'nms_kernel_size': 5
        }
    },
    # New, hardware-faithful CN0566 configuration
    'config_cn0566': {
        'name': 'CN0566_Phaser_DevKit',
        'signal_type': 'FMCW',
        'fc': 10.25e9,          # Centered in the 10–10.5 GHz band
        'B': 500e6,             # 500 MHz FMCW bandwidth
        'T_chirp': 500e-6,      # 500 µs ramp time
        'fs': 2.0e6,            # Baseband complex sample rate (Pluto SDR)
        'N_chirps': 64,         # Short CPI
        'R_max': 150.0,         # Practical max range for lab/field demos
        'description': 'Analog Devices CN0566 Phaser radar dev kit (Pluto IF chain model)',
        'hardware_model': 'CN0566',
        'num_rx': 2,            # Rx1, Rx2 (two ADAR1000 beamformers)
        # Array / beam parameters
        'use_array_factor': True,
        'array_N': 8,           # 8-element ULA
        'steering_angles': [0.0, 0.0],  # Two beams pointing boresight by default
        # Hardware impairments (enabled by default for CN0566)
        'model_dc_offset': True,
        'model_iq_imbalance': True,
        'model_phase_noise': True,
        'quantize_adc': True,
        'adc_bits': 12,
        'dc_scale': 0.05,       # DC offset ~5% of RMS
        'iq_gain_std': 0.02,    # 2% gain mismatch
        'iq_phase_std_deg': 3.0,# 3° phase mismatch
        'phase_noise_std': 0.02,# rad per chirp (slow-phase random walk)
        'cfo_std_hz': 200.0,    # small CFO in Hz
        # Clutter/lopag velocity jitters
        'static_clutter_velocity_std': 0.05,  # m/s jitter around 0
        'ground_clutter_velocity_std': 0.3,   # m/s jitter
        'coupling_rcs_db': 0.0,  # Stronger TX leakage
        'cfar_params': {
            'num_train': 16,
            'num_guard': 6,
            'threshold_offset': 18,
            'nms_kernel_size': 7
        }
    }
}

# ======================================================================
# Helper plotting & evaluation functions (for reuse in other modules)
# ======================================================================

def _plot_2d_rdm(dataset_instance, rdm, sample_idx, metrics,
                 matched_pairs, unmatched_targets, unmatched_detections, save_path):
    """
    Plot 2D Range-Doppler Map with annotations.
    """
    fig, ax = plt.subplots(figsize=(12, 8))  # Reduced size for faster plotting

    # Plot range-doppler map
    dr = dataset_instance.range_axis[1] - dataset_instance.range_axis[0]
    dv = dataset_instance.velocity_axis[1] - dataset_instance.velocity_axis[0]
    extent = [dataset_instance.range_axis[0] - dr/2, dataset_instance.range_axis[-1] + dr/2,
              dataset_instance.velocity_axis[0] - dv/2, dataset_instance.velocity_axis[-1] + dv/2]

    im = ax.imshow(rdm, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"Range-Doppler Map with CFAR Detection - Sample {sample_idx}")

    # Enforce view limits
    ax.set_xlim(VIEW_RANGE_LIMITS)
    ax.set_ylim(VIEW_VELOCITY_LIMITS)

    legend_elements = []

    # 1. Plot Ground Truth (Matched vs Missed)
    for target in matched_pairs:
        t = target[0]
        ax.scatter(t['range'], t['velocity'], facecolors='none', edgecolors='lime',
                   s=150, linewidth=2, label='Matched GT')
        # Draw line to detection
        d = target[1]
        ax.plot([t['range'], d['range_m']], [t['velocity'], d['velocity_mps']], 'w--', alpha=0.5)

    for target in unmatched_targets:
        ax.scatter(target['range'], target['velocity'], facecolors='none', edgecolors='red',
                   s=150, linewidth=2, label='Missed GT (FN)')

    # 2. Plot Detections (TP vs FP)
    for pair in matched_pairs:
        d = pair[1]
        ax.scatter(d['range_m'], d['velocity_mps'], marker='x', color='cyan',
                   s=100, linewidth=2, label='True Positive (TP)')

    for det in unmatched_detections:
        ax.scatter(det['range_m'], det['velocity_mps'], marker='x', color='orange',
                   s=100, linewidth=2, label='False Alarm (FP)')

    # Deduplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_elements.extend(by_label.values())

    # Create metrics summary text
    metrics_text = (
        f"Evaluation Metrics:\n"
        f"-------------------\n"
        f"Targets: {metrics['num_targets']}\n"
        f"Detections: {metrics['num_detections']}\n"
        f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}\n"
        f"Range Error (MAE): {metrics['mean_range_error']:.2f} m\n"
        f"Vel Error (MAE): {metrics['mean_velocity_error']:.2f} m/s"
    )

    # Add text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Place legend
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_3d_rdm(dataset_instance, rdm, sample_idx, targets, detections, save_path):
    """
    Plot 3D Range-Doppler Map using the external visualization library.
    """
    if VISUALIZATION_AVAILABLE:
        # rdm is already in dB, so we pass it directly with is_db=True

        # Convert targets format if needed (add 'distance' key)
        converted_targets = []
        for t in targets:
            ct = t.copy()
            ct['distance'] = t['range']
            converted_targets.append(ct)

        # Calculate actual resolutions from axes to ensure alignment with the data
        range_res = dataset_instance.range_axis[1] - dataset_instance.range_axis[0]
        vel_res = dataset_instance.velocity_axis[1] - dataset_instance.velocity_axis[0]

        # Ensure detections have integer indices
        cleaned_detections = []
        if detections:
            for det in detections:
                d_copy = det.copy()
                # Ensure indices are integers
                if 'range_idx' in d_copy:
                    d_copy['range_idx'] = int(d_copy['range_idx'])
                if 'doppler_idx' in d_copy:
                    d_copy['doppler_idx'] = int(d_copy['doppler_idx'])
                cleaned_detections.append(d_copy)

        plot_3d_range_doppler_map_with_ground_truth(
            rd_map=rdm,
            targets=converted_targets,
            range_resolution=range_res,
            velocity_resolution=vel_res,
            num_range_bins=rdm.shape[1],
            num_doppler_bins=rdm.shape[0],
            save_path=save_path,
            apply_doppler_centering=True,
            detections=cleaned_detections,
            view_range_limits=VIEW_RANGE_LIMITS,
            view_velocity_limits=VIEW_VELOCITY_LIMITS,
            is_db=True,
            stride=8  # Downsample for speed
        )


def evaluate_dataset_metrics(dataset, name):
    """
    Evaluate CFAR detection metrics across all samples in a dataset.

    Args:
        dataset: Iterable dataset where each item is a sample dict with keys:
            - 'range_doppler_map': 2D tensor/array of shape [doppler_bins, range_bins]
                                   representing the magnitude (typically dB) of the RD map.
            - 'target_info': dict containing ground-truth target metadata with:
                - 'targets': list[dict] where each dict includes fields like
                             'range' (meters), 'velocity' (m/s), and optionally
                             'range_idx'/'doppler_idx' (int indices).
            - 'cfar_detections': list[dict] CFAR detection outputs with fields such as
                                 'range' (meters), 'velocity' (m/s), 'range_idx', 'doppler_idx'.
        name: Descriptive dataset name used for printing summary.

    Prints:
        - Aggregate precision, recall, F1-score across the dataset
        - Mean absolute errors for range and velocity (meters, m/s)
    """
    print(f"\nEvaluating CFAR Metrics for {name}...")
    all_tp, all_fp, all_fn = 0, 0, 0
    all_range_errors = []
    all_vel_errors = []

    for i in range(len(dataset)):
        sample = dataset[i]
        targets = sample['target_info']['targets']
        detections = sample['cfar_detections']

        metrics, _, _, _ = dataset._evaluate_metrics(targets, detections)

        all_tp += metrics['tp']
        all_fp += metrics['fp']
        all_fn += metrics['fn']
        if metrics['mean_range_error'] > 0:
            all_range_errors.append(metrics['mean_range_error'])
        if metrics['mean_velocity_error'] > 0:
            all_vel_errors.append(metrics['mean_velocity_error'])

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_range_err = np.mean(all_range_errors) if all_range_errors else 0.0
    mean_vel_err = np.mean(all_vel_errors) if all_vel_errors else 0.0

    print(f"--- {name} CFAR Results ---")
    print(f"Samples: {len(dataset)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")
    print(f"Mean Range Error: {mean_range_err:.4f} m")
    print(f"Mean Vel Error:   {mean_vel_err:.4f} m/s")
    print("-" * 30)


# ======================================================================
# Dataset / Simulator
# ======================================================================

class AIRadarDataset(Dataset):
    def __init__(self,
                 num_samples=100,
                 radar_config=None,          # New argument for config selection
                 config_name='config1',      # Default config name
                 fc=None,                    # Overrides config if provided
                 B=None,
                 T_chirp=None,
                 fs=None,                    # Sampling Rate Override
                 N_samples=None,             # Deprecated: Calculated from fs * T_chirp
                 N_chirps=None,
                 R_max=None,
                 SNR_dB_min=20,              # Minimum SNR
                 SNR_dB_max=40,              # Maximum SNR
                 zero_pad_factor=2,          # Zero padding factor
                 max_targets=3,              # Maximum targets per sample
                 save_path='data/radar_corrected_test5',
                 precision='float32',
                 drawfig=False,
                 datapath=None,
                 cfar_params=None,
                 apply_realistic_effects=True,
                 clutter_intensity=1.0):
        """
        Initialize corrected radar dataset with proper FMCW/OTFS parameters and
        optional hardware-specific modeling (CN0566, etc.).
        """

        # Load Base Configuration
        if radar_config is None:
            if config_name in RADAR_CONFIGS:
                cfg = RADAR_CONFIGS[config_name]
                print(f"Loading Radar Configuration: {config_name} ({cfg['name']})")
            else:
                print(f"Warning: Config '{config_name}' not found. Using defaults (config1).")
                cfg = RADAR_CONFIGS['config1']
        else:
            cfg = radar_config

        # Store parameters, allowing overrides
        self.config = cfg  # Store full config
        self.signal_type = cfg.get('signal_type', 'FMCW')
        self.num_samples = num_samples
        self.fc = fc if fc is not None else cfg.get('fc', 77e9)
        self.B = B if B is not None else cfg.get('B', 1.5e9)
        self.T = T_chirp if T_chirp is not None else cfg.get('T_chirp', 40e-6)
        self.Nc = N_chirps if N_chirps is not None else cfg.get('N_chirps', 128)
        self.R_max = R_max if R_max is not None else cfg.get('R_max', 100)
        self.hardware_model = cfg.get('hardware_model', 'generic')

        # Number of RX channels (for CN0566 etc.)
        self.num_rx = int(cfg.get('num_rx', 1))

        # Handle Sampling Rate (fs) and Samples per Chirp (Ns)
        # Priority: 1. Explicit fs arg, 2. Config fs, 3. Explicit N_samples arg, 4. Config N_samples (Legacy)
        self.fs = fs if fs is not None else cfg.get('fs', None)

        if self.fs is not None:
            # Calculate N_samples from fs
            self.Ns = int(self.fs * self.T)
            if N_samples is not None:
                print(f"Info: N_samples argument ({N_samples}) ignored because fs ({self.fs/1e6} MHz) is provided.")
        else:
            # Fallback to N_samples
            self.Ns = N_samples if N_samples is not None else cfg.get('N_samples', 2048)
            self.fs = self.Ns / self.T
            print(f"Info: fs not provided. Calculated fs = {self.fs/1e6:.2f} MHz from N_samples ({self.Ns})")

        self.SNR_dB_min = SNR_dB_min
        self.SNR_dB_max = SNR_dB_max
        self.max_targets = max_targets
        self.save_path = save_path
        self.drawfig = drawfig
        self.precision = precision
        self.apply_realistic_effects = apply_realistic_effects
        self.clutter_intensity = clutter_intensity

        # CFAR Parameters
        default_cfar = {'num_train': 10, 'num_guard': 4, 'threshold_offset': 15, 'nms_kernel_size': 5}
        config_cfar = cfg.get('cfar_params', default_cfar)
        self.cfar_params = cfar_params if cfar_params is not None else config_cfar

        # Derived parameters
        self.lambda_c = c / self.fc
        self.slope = self.B / self.T
        self.zero_pad = zero_pad_factor * self.Ns

        # Hardware impairment flags & parameters (mainly used for config_cn0566)
        self.use_array_factor = bool(cfg.get('use_array_factor', False))
        self.array_N = int(cfg.get('array_N', 8))
        steering_angles = cfg.get('steering_angles', [0.0] * self.num_rx)
        if len(steering_angles) < self.num_rx:
            steering_angles = list(steering_angles) + [steering_angles[-1]] * (self.num_rx - len(steering_angles))
        self.steering_angles = np.array(steering_angles, dtype=float)

        self.model_dc_offset = bool(cfg.get('model_dc_offset', False))
        self.model_iq_imbalance = bool(cfg.get('model_iq_imbalance', False))
        self.model_phase_noise = bool(cfg.get('model_phase_noise', False))
        self.quantize_adc = bool(cfg.get('quantize_adc', False))

        self.dc_scale = float(cfg.get('dc_scale', 0.02))
        self.iq_gain_std = float(cfg.get('iq_gain_std', 0.02))
        self.iq_phase_std_deg = float(cfg.get('iq_phase_std_deg', 2.0))
        self.phase_noise_std = float(cfg.get('phase_noise_std', 0.01))
        self.cfo_std_hz = float(cfg.get('cfo_std_hz', 200.0))
        self.adc_bits = int(cfg.get('adc_bits', 12))

        # Clutter velocity jitters & coupling strength
        self.static_clutter_velocity_std = float(cfg.get('static_clutter_velocity_std', 0.0))
        self.ground_clutter_velocity_std = float(cfg.get('ground_clutter_velocity_std', 0.3))
        self.coupling_rcs_db = float(cfg.get('coupling_rcs_db', -10.0))

        # Time vectors
        self.t_fast = np.arange(self.Ns) / self.fs
        self.t_slow = np.arange(self.Nc) * self.T

        # Nyquist-limited maximum range for FMCW (ensure simulation is physically consistent)
        if self.signal_type == 'FMCW':
            f_nyquist = self.fs / 2.0
            R_max_nyquist = f_nyquist * c / (2 * self.slope)
            if self.R_max > R_max_nyquist:
                print(f"WARNING: Requested R_max={self.R_max:.1f}m exceeds Nyquist-limited "
                      f"R_max={R_max_nyquist:.1f}m. Clipping to physical limit.")
                self.R_max = R_max_nyquist
        self.R_max_physical = self.R_max

        # Axes / resolutions
        if self.signal_type == 'OTFS':
            # OTFS: approximate range axis using delay resolution ~ 1/fs
            range_res = c / (2 * self.fs)
            self.range_resolution = range_res
            self.num_range_bins = int(self.R_max / range_res)
            self.range_axis = np.arange(self.num_range_bins) * range_res
            self.num_doppler_bins = self.Nc
        else:
            # FMCW Range Axis using beat-frequency to range conversion
            range_res_fft = (c * self.fs) / (2 * self.slope * self.zero_pad)
            self.range_axis = np.arange(self.zero_pad // 2) * range_res_fft

            max_bin_idx = int(self.R_max / range_res_fft)
            self.num_range_bins = min(self.zero_pad // 2, max_bin_idx)
            self.range_axis = self.range_axis[:self.num_range_bins]

            self.num_doppler_bins = self.Nc

        # Velocity axis & resolutions
        self.velocity_axis = np.fft.fftshift(np.fft.fftfreq(self.Nc, d=self.T)) * self.lambda_c / 2
        self.range_resolution = c / (2 * self.B)
        self.velocity_resolution = self.lambda_c / (2 * self.Nc * self.T)
        self.max_unambiguous_velocity = self.lambda_c / (4 * self.T)
        self.v_max = self.max_unambiguous_velocity

        # Initialize data containers
        self.time_domain_data = None
        self.range_doppler_maps = None
        self.target_masks = None
        self.target_info = None
        self.cfar_detections = None

        print("\n=== Corrected Radar System Parameters ===")
        print(f"✅ Signal Type           : {self.signal_type}")
        print(f"✅ Hardware Model        : {self.hardware_model}")
        print(f"✅ Center Frequency      : {self.fc/1e9:.2f} GHz")
        print(f"✅ Bandwidth             : {self.B/1e6:.1f} MHz")
        print(f"✅ Chirp/Symbol Duration : {self.T*1e6:.1f} μs")
        print(f"✅ Sample Rate           : {self.fs/1e6:.2f} MHz")
        print(f"✅ Maximum Range (used)  : {self.R_max:.1f} m")
        print(f"✅ Range Resolution      : {self.range_resolution:.2f} m")
        print(f"✅ Maximum Velocity      : {self.v_max:.1f} m/s")
        print(f"✅ Velocity Resolution   : {self.velocity_resolution:.2f} m/s")
        print(f"✅ Samples per Chirp/Sym : {self.Ns}")
        print(f"✅ Number of Chirps/Sym  : {self.Nc}")
        print(f"✅ Num RX Channels       : {self.num_rx}")
        print(f"✅ Range Bins            : {self.num_range_bins}")
        print(f"✅ Doppler Bins          : {self.num_doppler_bins}")
        print(f"✅ CFAR Parameters       : {self.cfar_params}")
        print("========================================\n")

        # Initialize storage for generated data
        # time_domain_data: [num_samples, num_rx, num_chirps, samples_per_chirp, 2]
        self.time_domain_data = None
        self.range_doppler_maps = None
        self.target_masks = None
        self.target_info = None
        self.cfar_detections = None

        if datapath is not None:
            print(f"Loading radar data from {datapath}")
            self._load_data(datapath)
        else:
            print("Generating new radar data")
            self.generate_dataset()

    # ------------------------------------------------------------------
    # Target & channel modeling
    # ------------------------------------------------------------------
    def generate_targets(self, num_targets=None):
        """
        Generate random targets within valid range and velocity limits.

        Returns:
            List of target dictionaries with range, velocity, and RCS.
        """
        if num_targets is None:
            num_targets = random.randint(1, self.max_targets)

        targets = []
        for _ in range(num_targets):
            target_range = np.random.uniform(10, self.R_max - 10)
            target_velocity = np.random.uniform(-self.v_max + 1, self.v_max - 1)
            target_rcs = np.random.uniform(5.0, 30.0)  # dBsm

            targets.append({
                'range': target_range,
                'velocity': target_velocity,
                'rcs': target_rcs,
                'azimuth': np.random.uniform(-30, 30),  # degrees
                'elevation': np.random.uniform(-10, 10)  # degrees
            })

        return targets

    def _generate_clutter_targets(self):
        """Generate static environmental and ground clutter targets."""
        clutter_targets = []

        # Intensity scaling (dB)
        intensity_db = 10 * np.log10(max(self.clutter_intensity, 1e-6))

        # 1. Environmental clutter (strong-ish static)
        num_static = random.randint(5, 15)
        for _ in range(num_static):
            vel = np.random.normal(0.0, self.static_clutter_velocity_std)
            clutter_targets.append({
                'range': random.uniform(5, self.R_max),
                'velocity': vel,
                'rcs': random.uniform(-40, -20) + intensity_db,
                'azimuth': random.uniform(-30, 30),
                'elevation': 0
            })

        # 2. Ground clutter (distributed weaker reflections at near/medium range)
        num_ground = random.randint(20, 50)
        for _ in range(num_ground):
            dist = random.uniform(1.0, self.R_max * 0.5)
            vel = np.random.normal(0.0, self.ground_clutter_velocity_std)
            clutter_targets.append({
                'range': dist,
                'velocity': vel,
                'rcs': random.uniform(-50, -30) + intensity_db,
                'azimuth': random.uniform(-60, 60),
                'elevation': random.uniform(-10, 0)
            })

        return clutter_targets

    def _generate_coupling_target(self):
        """Generate direct coupling / TX leakage target."""
        intensity_db = 10 * np.log10(max(self.clutter_intensity, 1e-6))
        return {
            'range': 0.001,  # effectively 0 m
            'velocity': 0.0,
            'rcs': self.coupling_rcs_db + intensity_db,
            'azimuth': 0.0,
            'elevation': 0.0
        }

    # ------------------------------------------------------------------
    # Array factor / hardware impairments
    # ------------------------------------------------------------------
    def _array_gain(self, azimuth_deg, rx_idx=0):
        """Compute normalized array-factor magnitude for a given azimuth and RX."""
        if not self.use_array_factor or self.array_N <= 1:
            return 1.0

        d = self.lambda_c / 2.0
        N = self.array_N
        theta = np.deg2rad(azimuth_deg)
        theta_steer = np.deg2rad(self.steering_angles[rx_idx])
        k = 2 * np.pi / self.lambda_c
        psi = k * d * (np.sin(theta) - np.sin(theta_steer))

        # Handle psi ~ 0 (limit -> 1)
        denom = N * np.sin(psi / 2.0) + 1e-12
        num = np.sin(N * psi / 2.0)
        AF = num / denom
        return float(np.abs(AF))

    def _apply_hardware_impairments(self, beat):
        """
        Apply CN0566-like hardware impairments to the multi-RX beat signal.

        beat: complex ndarray, shape [num_rx, Nc, Ns]
        """
        # DC offset
        if self.model_dc_offset:
            power = np.mean(np.abs(beat)**2) + 1e-12
            sigma = np.sqrt(power) * self.dc_scale
            dc = (np.random.randn(self.num_rx, 1, 1) + 1j * np.random.randn(self.num_rx, 1, 1)) * sigma
            beat = beat + dc

        # IQ imbalance (per RX)
        if self.model_iq_imbalance:
            for r in range(self.num_rx):
                I = beat[r].real
                Q = beat[r].imag
                gain_q = 1.0 + np.random.normal(0.0, self.iq_gain_std)
                phase_err = np.deg2rad(np.random.normal(0.0, self.iq_phase_std_deg))
                Q_imb = gain_q * (Q * np.cos(phase_err) + I * np.sin(phase_err))
                beat[r] = I + 1j * Q_imb

        # Phase noise + CFO
        if self.model_phase_noise:
            # Slow-time random walk (per chirp)
            pn = np.cumsum(np.random.normal(0.0, self.phase_noise_std, size=self.Nc))
            factor_slow = np.exp(1j * pn)[None, :, None]  # [1, Nc, 1]
            # CFO along fast-time
            f_cfo = np.random.normal(0.0, self.cfo_std_hz)
            factor_fast = np.exp(1j * 2 * np.pi * f_cfo * self.t_fast)[None, None, :]  # [1, 1, Ns]
            beat = beat * factor_slow * factor_fast

        # ADC quantization
        if self.quantize_adc:
            max_abs = np.max(np.abs(beat)) + 1e-12
            norm = beat / max_abs
            q_levels = 2 ** (self.adc_bits - 1) - 1
            Iq = np.round(norm.real * q_levels) / q_levels
            Qq = np.round(norm.imag * q_levels) / q_levels
            beat = (Iq + 1j * Qq) * max_abs

        return beat

    # ------------------------------------------------------------------
    # Signal simulation (FMCW & OTFS)
    # ------------------------------------------------------------------
    def simulate_fmcw_signal(self, targets, snr_db=20):
        """
        Simulate multi-RX FMCW radar beat signals for multiple targets.

        Returns:
            beat_rx: complex ndarray [num_rx, Nc, Ns]
            rdm:     float ndarray [Nc, num_range_bins] in dB (combined across RX)
        """
        # Multi-RX beat
        beat_rx = np.zeros((self.num_rx, self.Nc, self.Ns), dtype=np.complex128)

        if targets:
            # Vectorized target parameter extraction
            ranges = np.array([t['range'] for t in targets])
            velocities = np.array([t['velocity'] for t in targets])
            rcs = np.array([t['rcs'] for t in targets])
            azimuths = np.array([t['azimuth'] for t in targets])

            K = len(ranges)

            # Convert RCS (dBsm) to linear power -> amplitude
            rcs_linear = 10 ** (rcs / 10)
            amplitudes = np.sqrt(rcs_linear)  # [K]

            # Frequencies
            fb = 2 * ranges * self.slope / c        # beat frequency / range
            fd = 2 * velocities / self.lambda_c     # Doppler

            fb_grid = fb[:, None, None]             # [K,1,1]
            fd_grid = fd[:, None, None]             # [K,1,1]
            t_fast_grid = self.t_fast[None, None, :]  # [1,1,Ns]
            t_slow_grid = self.t_slow[None, :, None]  # [1,Nc,1]

            phase = 2 * np.pi * (fb_grid * t_fast_grid + fd_grid * t_slow_grid)
            # Ideal per-target signal before array gains: [K, Nc, Ns]
            signal_k = amplitudes[:, None, None] * np.exp(1j * phase)

            # Per-target, per-RX array gain
            gains = np.ones((K, self.num_rx), dtype=float)
            if self.use_array_factor:
                for k in range(K):
                    for r in range(self.num_rx):
                        gains[k, r] = self._array_gain(azimuths[k], rx_idx=r)

            # Combine targets -> RX channels with einsum:
            # beat_rx[r, i, j] = sum_k gains[k,r] * signal_k[k,i,j]
            beat_rx = np.einsum('kr,kij->rij', gains, signal_k)

        # Apply Hann windows
        window_range = np.hanning(self.Ns)[None, None, :]   # [1,1,Ns]
        window_doppler = np.hanning(self.Nc)[None, :, None] # [1,Nc,1]
        beat_rx *= window_range
        beat_rx *= window_doppler

        # Add AWGN (same SNR per-RX)
        signal_power = np.mean(np.abs(beat_rx) ** 2)
        if signal_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise_scale = np.sqrt(noise_power / 2)
            noise = (np.random.randn(*beat_rx.shape) + 1j * np.random.randn(*beat_rx.shape)) * noise_scale
            beat_rx += noise

        # Apply hardware impairments (CN0566 etc.)
        if self.hardware_model == 'CN0566':
            beat_rx = self._apply_hardware_impairments(beat_rx)

        # Combine RX channels coherently for RDM
        beat_sum = np.sum(beat_rx, axis=0)  # [Nc, Ns]

        # Range FFT (fast-time)
        range_fft = np.fft.fft(beat_sum, n=self.zero_pad, axis=1)
        range_fft = range_fft[:, :self.zero_pad // 2]

        # Doppler FFT (slow-time)
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)

        rdm = 20 * np.log10(np.abs(doppler_fft) + 1e-6)

        # Crop to valid range bins
        if rdm.shape[1] > self.num_range_bins:
            rdm = rdm[:, :self.num_range_bins]

        return beat_rx, rdm

    # OTFS mod/demod helpers (single-RX)
    def _otfs_modulate(self, dd_grid):
        """
        Modulates a Delay-Doppler (M x N) grid to a time-domain signal.
        Tx Chain: ISFFT (DD->TF) -> Heisenberg (TF->Time)
        """
        # dd_grid: [Ns, Nc] = (Delay, Doppler)
        tf_grid = np.fft.fft(dd_grid, axis=0)
        tf_grid = np.fft.ifft(tf_grid, axis=1)
        time_domain_grid = np.fft.ifft(tf_grid, axis=0)
        tx_signal = time_domain_grid.flatten(order='F')
        return tx_signal

    def _otfs_demodulate(self, rx_signal):
        """
        Demodulates a time-domain signal back to a Delay-Doppler grid.
        Rx Chain: Wigner (Time->TF) -> SFFT (TF->DD)
        """
        time_domain_grid = rx_signal.reshape((self.Ns, self.Nc), order='F')
        tf_grid = np.fft.fft(time_domain_grid, axis=0)
        dd_grid = np.fft.fft(tf_grid, axis=1)
        dd_grid = np.fft.ifft(dd_grid, axis=0)
        return dd_grid

    def simulate_otfs_signal(self, targets, snr_db=20):
        """
        Simulate OTFS radar signal (single-RX).
        Returns:
            beat_rx: [1, Nc, Ns] complex
            ddm_db: [Nc, num_range_bins] dB delay-Doppler map
        """
        # 1. QPSK symbols
        num_symbols = self.Ns * self.Nc
        bits = np.random.randint(0, 4, num_symbols)
        mod_map = {
            0: (1 + 1j) / np.sqrt(2),
            1: (1 - 1j) / np.sqrt(2),
            2: (-1 + 1j) / np.sqrt(2),
            3: (-1 - 1j) / np.sqrt(2)
        }
        symbols = np.array([mod_map[b] for b in bits])
        tx_dd_grid = symbols.reshape((self.Ns, self.Nc))

        # 2. Modulate to time
        tx_signal = self._otfs_modulate(tx_dd_grid)

        # 3. Apply channel
        n_samples = tx_signal.size
        rx_signal = np.zeros(n_samples, dtype=complex)
        time_vector = np.arange(n_samples) / self.fs

        for target in targets:
            range_m = target['range']
            velocity_mps = target['velocity']
            rcs = target['rcs']

            amplitude = np.sqrt(10 ** (rcs / 10))

            delay_sec = 2 * range_m / c
            delay_samples = int(round(delay_sec * self.fs))

            if delay_samples < n_samples:
                delayed_signal = np.roll(tx_signal, delay_samples)
                delayed_signal[:delay_samples] = 0

                doppler_hz = 2 * velocity_mps * self.fc / c
                doppler_shift = np.exp(1j * 2 * np.pi * doppler_hz * time_vector)

                rx_signal += amplitude * delayed_signal * doppler_shift

        # 4. AWGN
        signal_power = np.mean(np.abs(rx_signal) ** 2)
        if signal_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * np.sqrt(noise_power / 2)
            rx_signal += noise

        # 5. Demodulate
        rx_dd_grid = self._otfs_demodulate(rx_signal)

        # 6. Simple DD-domain channel estimation / map
        rx_dd_fft = np.fft.fft2(rx_dd_grid)
        tx_dd_fft = np.fft.fft2(tx_dd_grid)
        epsilon = 1e-6
        ddm_fft = rx_dd_fft / (tx_dd_fft + epsilon)
        ddm_complex = np.fft.ifft2(ddm_fft)

        # 7. Format
        ddm_transposed = ddm_complex.T  # [Nc, Ns]
        ddm_shifted = np.fft.fftshift(ddm_transposed, axes=0)
        ddm_mag = np.abs(ddm_shifted)
        ddm_db = 20 * np.log10(ddm_mag + 1e-6)

        # Crop to valid range bins
        if ddm_db.shape[1] > self.num_range_bins:
            ddm_db = ddm_db[:, :self.num_range_bins]

        # Time-domain reshape for storage: [Nc, Ns] -> [1, Nc, Ns]
        rx_time_reshaped = rx_signal.reshape((self.Ns, self.Nc), order='F').T  # [Nc, Ns]
        beat_rx = rx_time_reshaped[None, :, :]  # [1, Nc, Ns]

        return beat_rx, ddm_db

    # ------------------------------------------------------------------
    # Ground truth masks & CFAR
    # ------------------------------------------------------------------
    def create_target_mask(self, targets):
        """
        Create binary mask for target locations in range-doppler map.

        Returns:
            mask: [doppler_bins, range_bins] (2D)
        """
        mask = np.zeros((self.num_doppler_bins, self.num_range_bins))

        for target in targets:
            range_idx = np.argmin(np.abs(self.range_axis - target['range']))
            velocity_idx = np.argmin(np.abs(self.velocity_axis - target['velocity']))

            for di in range(-1, 2):
                for ri in range(-1, 2):
                    v_idx = velocity_idx + di
                    r_idx = range_idx + ri
                    if 0 <= v_idx < self.num_doppler_bins and 0 <= r_idx < self.num_range_bins:
                        mask[v_idx, r_idx] = 1.0

        return mask

    def _cfar_2d_custom(self, rd_map_db, num_train=8, num_guard=4,
                        range_res=0.5, doppler_res=0.25,
                        max_range=100, max_speed=50,
                        threshold_offset=4, nms_kernel_size=3,
                        mtd=False):
        """
        Custom 2D CFAR (GO-CFAR) over dB-domain RD maps.
        """
        from scipy.signal import convolve2d
        from scipy.ndimage import maximum_filter

        rows, cols = rd_map_db.shape

        k = num_guard + num_train
        window_size = 2 * k + 1
        full_kernel = np.ones((window_size, window_size), dtype=np.float32)
        guard_area = np.zeros_like(full_kernel)
        guard_area[num_train:num_train + 2 * num_guard + 1,
                   num_train:num_train + 2 * num_guard + 1] = 1
        train_kernel = full_kernel - guard_area

        horiz_kernel = train_kernel.copy()
        horiz_kernel[num_train:num_train + 2 * num_guard + 1, :] = 0
        vert_kernel = train_kernel.copy()
        vert_kernel[:, num_train:num_train + 2 * num_guard + 1] = 0

        noise_h = convolve2d(rd_map_db, horiz_kernel / np.sum(horiz_kernel),
                             mode='same', boundary='symm')
        noise_v = convolve2d(rd_map_db, vert_kernel / np.sum(vert_kernel),
                             mode='same', boundary='symm')
        noise_est = np.maximum(noise_h, noise_v)

        threshold = noise_est + threshold_offset
        detections = rd_map_db > threshold

        if nms_kernel_size > 1:
            local_max = maximum_filter(rd_map_db, size=nms_kernel_size)
            detections &= (rd_map_db == local_max)

        doppler_idxs, range_idxs = np.where(detections)
        results = []

        num_doppler = rows

        for d_idx, r_idx in zip(doppler_idxs, range_idxs):
            range_m = r_idx * range_res
            velocity_mps = (d_idx - num_doppler // 2) * doppler_res

            if not (0.5 < range_m < max_range and abs(velocity_mps) < max_speed):
                continue

            if mtd and abs(velocity_mps) < 1.0:
                continue

            results.append({
                "range_idx": r_idx,
                "doppler_idx": d_idx,
                "range_m": range_m,
                "velocity_mps": velocity_mps,
                "angle_deg": None
            })

        return results

    def cfar_detection(self, rd_map):
        """
        Perform CFAR detection on a Range-Doppler map (in dB).

        Returns:
            List of detection dictionaries.
        """
        range_res = self.range_axis[1] - self.range_axis[0]
        velocity_res = self.velocity_axis[1] - self.velocity_axis[0]

        mtd_enabled = self.apply_realistic_effects

        cfar_results = self._cfar_2d_custom(
            rd_map,
            num_train=self.cfar_params.get('num_train', 10),
            num_guard=self.cfar_params.get('num_guard', 4),
            range_res=range_res,
            doppler_res=velocity_res,
            max_range=self.R_max,
            max_speed=50,
            threshold_offset=self.cfar_params.get('threshold_offset', 15),
            nms_kernel_size=self.cfar_params.get('nms_kernel_size', 5),
            mtd=mtd_enabled
        )

        for detection in cfar_results:
            d_idx = detection['doppler_idx']
            r_idx = detection['range_idx']
            if 0 <= d_idx < rd_map.shape[0] and 0 <= r_idx < rd_map.shape[1]:
                detection['magnitude'] = rd_map[d_idx, r_idx]

        return cfar_results

    # ------------------------------------------------------------------
    # Dataset generation / IO
    # ------------------------------------------------------------------
    def generate_dataset(self):
        """
        Generate the radar dataset including:
        - Target generation
        - Signal simulation (FMCW / OTFS)
        - Range-Doppler / delay-Doppler maps
        - CFAR detections and GT masks
        """
        print(f"Generating {self.num_samples} samples...")

        # Allocate arrays
        # time_domain_data: [N, num_rx, Nc, Ns, 2]
        self.time_domain_data = np.zeros(
            (self.num_samples, self.num_rx, self.Nc, self.Ns, 2),
            dtype=self.precision
        )
        self.range_doppler_maps = np.zeros(
            (self.num_samples, self.num_doppler_bins, self.num_range_bins),
            dtype=self.precision
        )
        self.target_masks = np.zeros(
            (self.num_samples, self.num_doppler_bins, self.num_range_bins, 1),
            dtype=self.precision
        )

        self.target_info = []
        self.cfar_detections = []

        vis_path = os.path.join(self.save_path, 'visualizations')
        if self.drawfig and not os.path.exists(vis_path):
            os.makedirs(vis_path)

        for i in tqdm(range(self.num_samples)):
            # Ground-truth targets
            targets = self.generate_targets()

            # SNR for this frame
            snr_db = random.uniform(self.SNR_dB_min, self.SNR_dB_max)

            # (Optional) TX chirp visualization (FMCW only, first sample)
            if self.drawfig and i == 0 and VISUALIZATION_AVAILABLE and self.signal_type == 'FMCW':
                t = np.linspace(0, self.T, int(self.fs * self.T))
                tx_signal = np.cos(2 * np.pi * (self.fc * t + 0.5 * self.slope * t**2))

                plot_signal_time_and_spectrum(
                    signal=tx_signal,
                    sample_rate=self.fs,
                    total_duration=self.T,
                    title_prefix="TX Chirp",
                    textstr=f"fc={self.fc/1e9:.1f}GHz, B={self.B/1e6:.1f}MHz",
                    normalize=False,
                    save_path=os.path.join(vis_path, f"tx_chirp_{i}.png"),
                    draw_window=False
                )

                plot_instantaneous_frequency(
                    signal=tx_signal,
                    sample_rate=self.fs,
                    total_duration=self.T,
                    slope=self.slope,
                    bandwidth=self.B,
                    center_freq=self.fc,
                    title_prefix="TX Chirp",
                    textstr=f"Bandwidth: {self.B/1e6:.1f} MHz\nSlope: {self.slope/1e12:.2f} THz/s",
                    save_path=os.path.join(vis_path, f"tx_chirp_freq_{i}.png")
                )

            # Build full channel target list
            sim_targets = list(targets)
            if self.apply_realistic_effects:
                sim_targets.extend(self._generate_clutter_targets())
                sim_targets.append(self._generate_coupling_target())

            # Simulate based on signal type
            if self.signal_type == 'OTFS':
                beat_signal, rdm = self.simulate_otfs_signal(sim_targets, snr_db)
            else:
                beat_signal, rdm = self.simulate_fmcw_signal(sim_targets, snr_db)

            # Optional beat-signal visualization (first 3 frames)
            if self.drawfig and i < 3 and VISUALIZATION_AVAILABLE:
                # First RX, first chirp
                beat_chirp = beat_signal[0, 0, :]
                plot_signal_time_and_spectrum(
                    signal=beat_chirp,
                    sample_rate=self.fs,
                    total_duration=self.T,
                    title_prefix="Beat Signal",
                    textstr=None,
                    normalize=False,
                    save_path=os.path.join(vis_path, f"beat_signal_{i}.png"),
                    draw_window=False
                )

            # Store time-domain data (real/imag)
            self.time_domain_data[i, :, :, :, 0] = beat_signal.real.astype(self.precision)
            self.time_domain_data[i, :, :, :, 1] = beat_signal.imag.astype(self.precision)

            # Store RD/DD map
            self.range_doppler_maps[i] = rdm.astype(self.precision)

            # Target masks (based only on primary moving targets)
            mask = self.create_target_mask(targets)
            self.target_masks[i, :, :, 0] = mask.astype(self.precision)

            # CFAR detections
            cfar_results = self.cfar_detection(rdm)
            self.cfar_detections.append(cfar_results)

            self.target_info.append({
                'targets': targets,
                'snr_db': snr_db,
                'sample_idx': i,
                'cfar_detections': cfar_results
            })

            # Visualize first few samples
            if self.drawfig and i < 5:
                self.plot_sample(i, targets, rdm, vis_path, detections=cfar_results)

        print(f"Dataset generation complete. Saving to {self.save_path}")
        self.save_dataset()

    def _evaluate_metrics(self, targets, detections, match_dist_thresh=3.0):
        """
        Evaluate detection performance metrics (TP, FP, FN, errors).
        """
        tp = 0
        range_errors = []
        velocity_errors = []

        unmatched_targets = targets.copy()
        unmatched_detections = detections.copy()
        matched_pairs = []

        for target in targets:
            best_det = None
            best_dist = float('inf')
            best_det_idx = -1

            for i, det in enumerate(unmatched_detections):
                d_r = target['range'] - det['range_m']
                d_v = target['velocity'] - det['velocity_mps']
                dist = np.sqrt(d_r ** 2 + d_v ** 2)

                if dist < match_dist_thresh and dist < best_dist:
                    best_dist = dist
                    best_det = det
                    best_det_idx = i

            if best_det:
                tp += 1
                range_errors.append(abs(target['range'] - best_det['range_m']))
                velocity_errors.append(abs(target['velocity'] - best_det['velocity_mps']))
                matched_pairs.append((target, best_det))

                unmatched_targets.remove(target)
                unmatched_detections.pop(best_det_idx)

        fp = len(unmatched_detections)
        fn = len(unmatched_targets)

        mean_range_error = np.mean(range_errors) if range_errors else 0.0
        mean_velocity_error = np.mean(velocity_errors) if velocity_errors else 0.0

        metrics = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'mean_range_error': mean_range_error,
            'mean_velocity_error': mean_velocity_error,
            'num_targets': len(targets),
            'num_detections': len(detections)
        }

        return metrics, matched_pairs, unmatched_targets, unmatched_detections

    def plot_sample(self, sample_idx, targets, rdm, save_dir, detections=None):
        """
        Plot RD map (2D+3D) with GT & CFAR detections for a sample.
        """
        rdm_norm = rdm - np.max(rdm)

        detection_results = detections if detections is not None else self.cfar_detection(rdm)
        metrics, matched_pairs, unmatched_targets, unmatched_detections = self._evaluate_metrics(
            targets, detection_results
        )

        save_path_2d = os.path.join(save_dir, f"rdm_sample_{sample_idx}.png")
        _plot_2d_rdm(self, rdm_norm, sample_idx, metrics,
                     matched_pairs, unmatched_targets, unmatched_detections, save_path_2d)

        save_path_3d = os.path.join(save_dir, f"rdm_3d_sample_{sample_idx}.png")
        _plot_3d_rdm(self, rdm_norm, sample_idx, targets, detection_results, save_path_3d)

        print(f"Saved 2D visualization: {save_path_2d}")
        print(f"Saved 3D visualization: {save_path_3d}")
        print(f"CFAR detected {metrics['num_detections']} targets vs {metrics['num_targets']} ground truth targets")

    def save_dataset(self):
        """
        Save the generated dataset to HDF5 format.
        """
        os.makedirs(self.save_path, exist_ok=True)
        save_file = os.path.join(self.save_path, "radar_dataset.h5")

        with h5py.File(save_file, 'w') as f:
            f.create_dataset('time_domain_data', data=self.time_domain_data, compression='gzip')
            f.create_dataset('range_doppler_maps', data=self.range_doppler_maps, compression='gzip')
            f.create_dataset('target_masks', data=self.target_masks, compression='gzip')

            f.create_dataset('range_axis', data=self.range_axis)
            f.create_dataset('velocity_axis', data=self.velocity_axis)

            f.attrs['fc'] = self.fc
            f.attrs['B'] = self.B
            f.attrs['T_chirp'] = self.T
            f.attrs['N_samples'] = self.Ns
            f.attrs['N_chirps'] = self.Nc
            f.attrs['R_max'] = self.R_max
            f.attrs['fs'] = self.fs
            f.attrs['range_resolution'] = self.range_resolution
            f.attrs['velocity_resolution'] = self.velocity_resolution
            f.attrs['num_rx'] = self.num_rx

            import json
            target_info_str = [json.dumps(info, default=str) for info in self.target_info]
            f.create_dataset('target_info', data=target_info_str, dtype=h5py.string_dtype())

        print(f"Dataset saved to: {save_file}")

    def _load_data(self, datapath):
        """
        Load existing dataset from HDF5 file.
        """
        with h5py.File(datapath, 'r') as f:
            self.time_domain_data = f['time_domain_data'][:]
            self.range_doppler_maps = f['range_doppler_maps'][:]
            self.target_masks = f['target_masks'][:]

            import json
            target_info_str = f['target_info'][:]
            self.target_info = [json.loads(info) for info in target_info_str]

            self.cfar_detections = []
            for info in self.target_info:
                if 'cfar_detections' in info:
                    self.cfar_detections.append(info['cfar_detections'])
                else:
                    self.cfar_detections.append([])

        print(f"Loaded dataset with {len(self.target_info)} samples")
        self.num_samples = len(self.target_info)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return self.num_samples if self.range_doppler_maps is not None else 0

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns dict:
            - time_domain: [num_rx, Nc, Ns, 2] (torch.float32)
            - range_doppler_map: [Nd, Nr]
            - target_mask: [Nd, Nr, 1]
            - target_info: metadata dict
            - cfar_detections: list[dict]
            - range_axis, velocity_axis: numpy arrays
        """
        if self.range_doppler_maps is None:
            raise ValueError("Dataset not generated or loaded")

        raw_np = self.time_domain_data[idx]

        if np.iscomplexobj(raw_np):
            raw_tensor = torch.from_numpy(raw_np)
            time_domain_formatted = torch.stack([raw_tensor.real, raw_tensor.imag], dim=-1).float()
        elif getattr(raw_np, 'dtype', None) is not None and raw_np.dtype.names is not None:
            if 'r' in raw_np.dtype.names:
                real = torch.from_numpy(raw_np['r'])
                imag = torch.from_numpy(raw_np['i'])
            elif 'real' in raw_np.dtype.names:
                real = torch.from_numpy(raw_np['real'])
                imag = torch.from_numpy(raw_np['imag'])
            else:
                real = torch.from_numpy(raw_np[raw_np.dtype.names[0]])
                imag = torch.from_numpy(raw_np[raw_np.dtype.names[1]])
            time_domain_formatted = torch.stack([real, imag], dim=-1).float()
        else:
            t = torch.from_numpy(raw_np).float()
            if t.dim() == 4 and t.shape[-1] == 2:
                time_domain_formatted = t
            elif t.dim() == 3:
                if t.shape[-1] == 2:
                    time_domain_formatted = t
                else:
                    time_domain_formatted = torch.stack([t, torch.zeros_like(t)], dim=-1)
            else:
                time_domain_formatted = torch.stack([t, torch.zeros_like(t)], dim=-1)

        return {
            'time_domain': time_domain_formatted,
            'range_doppler_map': torch.from_numpy(self.range_doppler_maps[idx]).float(),
            'target_mask': torch.from_numpy(self.target_masks[idx]).float(),
            'target_info': self.target_info[idx],
            'cfar_detections': self.cfar_detections[idx] if hasattr(self, 'cfar_detections')
                               and idx < len(self.cfar_detections) else [],
            'range_axis': self.range_axis,
            'velocity_axis': self.velocity_axis
        }


# ======================================================================
# Standalone test / demo
# ======================================================================
if __name__ == "__main__":
    num_samples = 100
    outpath = 'output/radar_datasetv8'

    # --- 1. Config 1: 77 GHz automotive FMCW ---
    print("\n" + "="*50)
    print("Generating Data for Config 1 (77 GHz Automotive FMCW)")
    print("="*50)

    dataset_c1 = AIRadarDataset(
        num_samples=num_samples,
        config_name='config1',
        drawfig=True,
        save_path=os.path.join(outpath, 'config1'),
        apply_realistic_effects=True,
        clutter_intensity=0.1,
    )

    print(f"\nConfig 1 Dataset created with {len(dataset_c1)} samples")
    evaluate_dataset_metrics(dataset_c1, "Config 1 (77GHz FMCW)")

    print(f"Generating visualizations for Config 1 (first 3 samples)...")
    for i in range(min(3, len(dataset_c1))):
        sample = dataset_c1[i]
        rdm = sample['range_doppler_map'].numpy()
        rdm_norm = rdm - np.max(rdm)
        targets = sample['target_info']['targets']
        detections = sample['cfar_detections']
        metrics, matched_pairs, unmatched_targets, unmatched_detections = dataset_c1._evaluate_metrics(
            targets, detections
        )

        save_dir = os.path.join(outpath, 'config1')
        _plot_2d_rdm(dataset_c1, rdm_norm, i, metrics,
                     matched_pairs, unmatched_targets, unmatched_detections,
                     os.path.join(save_dir, f"rdm_sample_{i}.png"))
        _plot_3d_rdm(dataset_c1, rdm_norm, i, targets, detections,
                     os.path.join(save_dir, f"rdm_3d_sample_{i}.png"))

    # --- 2. Config 2: 10 GHz X-band FMCW (generic robotics radar) ---
    print("\n" + "="*50)
    print("Generating Data for Config 2 (10 GHz X-Band FMCW)")
    print("="*50)

    dataset_c2 = AIRadarDataset(
        num_samples=num_samples,
        config_name='config2',
        drawfig=True,
        save_path=os.path.join(outpath, 'config2'),
        apply_realistic_effects=True,
        clutter_intensity=0.1,
    )

    print(f"\nConfig 2 Dataset created with {len(dataset_c2)} samples")
    evaluate_dataset_metrics(dataset_c2, "Config 2 (10GHz FMCW)")

    print(f"Generating visualizations for Config 2 (first 3 samples)...")
    for i in range(min(3, len(dataset_c2))):
        sample = dataset_c2[i]
        rdm = sample['range_doppler_map'].numpy()
        rdm_norm = rdm - np.max(rdm)
        targets = sample['target_info']['targets']
        detections = sample['cfar_detections']
        metrics, matched_pairs, unmatched_targets, unmatched_detections = dataset_c2._evaluate_metrics(
            targets, detections
        )

        save_dir = os.path.join(outpath, 'config2')
        _plot_2d_rdm(dataset_c2, rdm_norm, i, metrics,
                     matched_pairs, unmatched_targets, unmatched_detections,
                     os.path.join(save_dir, f"rdm_sample_{i}.png"))
        _plot_3d_rdm(dataset_c2, rdm_norm, i, targets, detections,
                     os.path.join(save_dir, f"rdm_3d_sample_{i}.png"))

    # --- 3. Config OTFS: 77 GHz OTFS radar (generic) ---
    print("\n" + "="*50)
    print("Generating Data for Config OTFS (77 GHz OTFS)")
    print("="*50)

    dataset_otfs = AIRadarDataset(
        num_samples=num_samples,
        config_name='config_otfs',
        drawfig=True,
        save_path=os.path.join(outpath, 'config_otfs'),
        apply_realistic_effects=True,
        clutter_intensity=0.1,
    )

    print(f"\nConfig OTFS Dataset created with {len(dataset_otfs)} samples")
    evaluate_dataset_metrics(dataset_otfs, "Config OTFS")

    print(f"Generating visualizations for Config OTFS (first 3 samples)...")
    for i in range(min(3, len(dataset_otfs))):
        sample = dataset_otfs[i]
        rdm = sample['range_doppler_map'].numpy()
        rdm_norm = rdm - np.max(rdm)
        targets = sample['target_info']['targets']
        detections = sample['cfar_detections']
        metrics, matched_pairs, unmatched_targets, unmatched_detections = dataset_otfs._evaluate_metrics(
            targets, detections
        )

        save_dir = os.path.join(outpath, 'config_otfs')
        _plot_2d_rdm(dataset_otfs, rdm_norm, i, metrics,
                     matched_pairs, unmatched_targets, unmatched_detections,
                     os.path.join(save_dir, f"rdm_sample_{i}.png"))
        _plot_3d_rdm(dataset_otfs, rdm_norm, i, targets, detections,
                     os.path.join(save_dir, f"rdm_3d_sample_{i}.png"))

    # --- 4. Config CN0566: hardware-faithful Phaser dev kit model ---
    print("\n" + "="*50)
    print("Generating Data for Config CN0566 (Phaser Dev Kit)")
    print("="*50)

    dataset_cn = AIRadarDataset(
        num_samples=num_samples,
        config_name='config_cn0566',
        drawfig=True,
        save_path=os.path.join(outpath, 'config_cn0566'),
        apply_realistic_effects=True,
        clutter_intensity=0.3,  # a bit stronger clutter for CN0566
    )

    print(f"\nConfig CN0566 Dataset created with {len(dataset_cn)} samples")
    evaluate_dataset_metrics(dataset_cn, "Config CN0566")

    print(f"Generating visualizations for Config CN0566 (first 3 samples)...")
    for i in range(min(3, len(dataset_cn))):
        sample = dataset_cn[i]
        rdm = sample['range_doppler_map'].numpy()
        rdm_norm = rdm - np.max(rdm)
        targets = sample['target_info']['targets']
        detections = sample['cfar_detections']
        metrics, matched_pairs, unmatched_targets, unmatched_detections = dataset_cn._evaluate_metrics(
            targets, detections
        )

        save_dir = os.path.join(outpath, 'config_cn0566')
        _plot_2d_rdm(dataset_cn, rdm_norm, i, metrics,
                     matched_pairs, unmatched_targets, unmatched_detections,
                     os.path.join(save_dir, f"rdm_sample_{i}.png"))
        _plot_3d_rdm(dataset_cn, rdm_norm, i, targets, detections,
                     os.path.join(save_dir, f"rdm_3d_sample_{i}.png"))

    print("\n" + "="*50)
    print("Generation and Evaluation Complete.")
    print(f"Visualizations (if enabled) saved to {outpath}")
    print("==================================================")