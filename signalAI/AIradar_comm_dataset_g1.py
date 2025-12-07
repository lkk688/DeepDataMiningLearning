import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from scipy.constants import c
import torch
from torch.utils.data import Dataset
import h5py
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter

# Check for AIRadarLib (Optional integration)
try:
    from AIRadarLib.visualization import plot_3d_range_doppler_map_with_ground_truth
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# ======================================================================
# Configurations: Hardware-Aligned (CN0566 / AD9361)
# ======================================================================

RADAR_COMM_CONFIGS = {
    # Mode A: Traditional Separation (CN0566 Hardware Limit)
    'CN0566_TRADITIONAL': {
        'mode': 'TRADITIONAL',
        'fc': 10.25e9,
        'mod_order': 16,
        
        # Radar Params (FMCW)
        'radar_B': 500e6,
        'radar_T': 500e-6,
        'radar_fs': 2e6,
        
        # Comm Params (OFDM)
        'comm_B': 40e6,
        'comm_fs': 61.44e6,
        'comm_fft_size': 64,
        'comm_cp_len': 16,
        'channel_model': 'multipath',
        
        'R_max': 150.0,
        'num_rx': 1,
        'cfar_params': {'num_train': 12, 'num_guard': 4, 'threshold_offset': 15, 'nms_kernel_size': 5}
    },

    # Mode B: Integrated Sensing and Comm (CN0566 Hardware Limit)
    'CN0566_OTFS_ISAC': {
        'mode': 'OTFS',
        'fc': 10.25e9,
        'mod_order': 4,          # QPSK for robustness in OTFS
        
        # Joint Params
        'B': 40e6,
        'fs': 61.44e6,
        
        'N_doppler': 64,         # Doppler Bins
        'N_delay': 64,           # Delay Bins
        'T_symbol': 16e-6,
        'channel_model': 'multipath',
        
        'R_max': 150.0,
        'num_rx': 1,
        'cfar_params': {
            'num_train': 28,
            'num_guard': 8,
            'threshold_offset': 0.15,
            'std_scale': 2.0,
            'use_sigma': True,
            'nms_kernel_size': 7,
            'global_percentile': 99.0,
            'max_peaks': 60,
            'min_range_m': 2.0,
            'min_speed_mps': 0.8,
            'notch_doppler_bins': 3
        }
    },
    
    # Config from AIradar_datasetv8.py: Automotive 77GHz Long Range (High Performance)
    'Automotive_77GHz_LongRange': {
        'mode': 'TRADITIONAL',
        'fc': 77e9,             # 77 GHz
        'mod_order': 4,         # QPSK for better BER
        
        # Radar Params (FMCW)
        'radar_B': 1.5e9,       # 1.5 GHz Bandwidth
        'radar_T': 40e-6,       # 40 μs
        'radar_fs': 51.2e6,     # 51.2 MHz Sampling Rate
        
        # Comm Params (OFDM)
        'comm_B': 400e6,
        'comm_fs': 512e6,
        'comm_fft_size': 1024,
        'comm_cp_len': 72,
        'channel_model': 'multipath',
        
        'R_max': 100.0,
        'num_rx': 1,
        'cfar_params': {'num_train': 10, 'num_guard': 4, 'threshold_offset': 15, 'nms_kernel_size': 5}
    },

    # Config from AIradar_datasetv8.py: X-Band Medium Range
    'XBand_10GHz_MediumRange': {
        'mode': 'TRADITIONAL',
        'fc': 10e9,             # 10 GHz (X-band)
        'mod_order': 16,        # 16-QAM
        
        # Radar Params
        'radar_B': 1.0e9,       # 1 GHz Bandwidth
        'radar_T': 160e-6,      # 160 μs
        'radar_fs': 40e6,       # 40 MHz
        
        # Comm Params
        'comm_B': 40e6,
        'comm_fs': 40e6,
        'comm_fft_size': 64,
        'comm_cp_len': 16,
        'channel_model': 'multipath',
        
        'R_max': 100.0,
        'num_rx': 1,
        'cfar_params': {'num_train': 24, 'num_guard': 8, 'threshold_offset': 18, 'nms_kernel_size': 7}
    },

    # Mode C: Realistic Automotive (Traditional Separated)
    'AUTOMOTIVE_TRADITIONAL': {
        'mode': 'TRADITIONAL',
        'fc': 77e9,
        'mod_order': 64,
        
        'radar_B': 1.5e9,
        'radar_T': 60e-6,
        'radar_fs': 50e6,
        
        'comm_B': 400e6,
        'comm_fs': 512e6,
        'comm_fft_size': 1024,
        'comm_cp_len': 72,
        'channel_model': 'multipath',
        
        'R_max': 250.0,
        'num_rx': 4,
        'cfar_params': {'num_train': 16, 'num_guard': 4, 'threshold_offset': 15, 'nms_kernel_size': 7}
    },

    # Mode D: Realistic Automotive (ISAC OTFS)
    'AUTOMOTIVE_OTFS_ISAC': {
        'mode': 'OTFS',
        'fc': 77e9,
        'mod_order': 4,
        
        'B': 1.536e9,            # 1.536 GHz (Matched to v8)
        'fs': 51.2e6,            # 51.2 MHz
        
        'N_doppler': 128,        # Matched to N_chirps in v8
        'N_delay': 512,          # Matched to N_samples in v8
        'T_symbol': 40e-6,       # Matched to T_chirp in v8
        'channel_model': 'multipath',
        
        'R_max': 100.0,
        'num_rx': 1,
        'cfar_params': {
            'num_train': 32,
            'num_guard': 8,
            'threshold_offset': 0.12,
            'std_scale': 2.0,
            'use_sigma': True,
            'nms_kernel_size': 7,
            'global_percentile': 99.0,
            'max_peaks': 60,
            'min_range_m': 2.0,
            'min_speed_mps': 1.0,
            'notch_doppler_bins': 3
        }
    }
}

# ======================================================================
# Helper: Visualization
# ======================================================================

def plot_combined_sample(sample_data, save_path):
    """
    Plots a dashboard of Radar (RDM) and Comm (Constellation) results.
    """
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 4)

    # 1. Range-Doppler Map (Radar)
    ax_rdm = fig.add_subplot(gs[:, :2])
    rdm = sample_data['range_doppler_map'].numpy()
    rdm_db = rdm - np.max(rdm) # Normalize to 0 dB peak
    
    r_axis = sample_data['range_axis']
    v_axis = sample_data['velocity_axis']
    
    # Check if RDM needs transpose for plotting
    # Typically imshow expects [Rows, Cols] -> [Y-axis, X-axis]
    # We want X-axis = Range, Y-axis = Velocity (Doppler)
    # If rdm is [Doppler, Range], then:
    # Rows = Doppler (Y), Cols = Range (X).
    # This matches.
    
    # extent = [left, right, bottom, top]
    # left/right = Range min/max
    # bottom/top = Velocity min/max
    extent = [r_axis[0], r_axis[-1], v_axis[0], v_axis[-1]]
    
    im = ax_rdm.imshow(rdm_db, aspect='auto', origin='lower', cmap='viridis', 
                       extent=extent, vmin=-60, vmax=0)
    plt.colorbar(im, ax=ax_rdm, label='Power (dB)')
    
    targets = sample_data['target_info']['targets']
    for t in targets:
        ax_rdm.scatter(t['range'], t['velocity'], s=150, edgecolor='lime', facecolor='none', lw=2, label='GT')
    
    dets = sample_data['cfar_detections']
    for d in dets:
        ax_rdm.scatter(d['range_m'], d['velocity_mps'], marker='x', s=100, color='red', label='CFAR')
        
    ax_rdm.set_title(f"Radar: Range-Doppler (Mode: {sample_data['mode']})\nTargets: {len(targets)} | Dets: {len(dets)}")
    ax_rdm.set_xlabel("Range (m)")
    ax_rdm.set_ylabel("Velocity (m/s)")
    ax_rdm.legend(loc='upper right')

    # 2. Comm Constellation (Tx vs Rx)
    ax_const = fig.add_subplot(gs[0, 2])
    
    tx_syms = sample_data['comm_info']['tx_symbols']
    rx_syms = sample_data['comm_info']['rx_symbols']
    
    if len(tx_syms) > 1000:
        idx = np.random.choice(len(tx_syms), 1000, replace=False)
        tx_syms = tx_syms[idx]
        rx_syms = rx_syms[idx]

    ax_const.scatter(np.real(rx_syms), np.imag(rx_syms), alpha=0.5, s=10, c='blue', label='Rx (Eq)')
    ax_const.scatter(np.real(tx_syms), np.imag(tx_syms), alpha=0.6, s=10, c='red', marker='x', label='Tx')
    
    mod_order = sample_data.get('mod_order', 4)
    ax_const.set_title(f"Comm: {mod_order}-QAM\nBER: {sample_data['comm_info']['ber']:.2e} | SNR: {sample_data['target_info']['snr_db']:.1f} dB")
    ax_const.set_xlabel("I")
    ax_const.set_ylabel("Q")
    ax_const.grid(True, alpha=0.3)
    ax_const.legend()
    ax_const.set_aspect('equal')
    
    # 3. Text Stats
    ax_text = fig.add_subplot(gs[1, 2:])
    ax_text.axis('off')
    
    mode_str = sample_data['mode']
    info_str = f"CONFIGURATION: {mode_str}\n"
    info_str += "-"*30 + "\n"
    if mode_str == 'TRADITIONAL':
        info_str += "RADAR: FMCW\nCOMM: OFDM w/ LS Channel Est.\n"
    else:
        info_str += "ISAC: OTFS w/ Perfect CSI Equalization\n"
        
    info_str += f"Channel: {sample_data.get('channel_model', 'AWGN')}\n"
    info_str += "\nPERFORMANCE:\n"
    info_str += f"BER: {sample_data['comm_info']['ber']:.5f}\n"
    info_str += f"Radar Range Error: {sample_data.get('metrics', {}).get('mean_range_error', 0):.2f} m\n"
    
    ax_text.text(0.1, 0.9, info_str, fontfamily='monospace', fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def _plot_2d_rdm(dataset_instance, rdm, sample_idx, metrics,
                 matched_pairs, unmatched_targets, unmatched_detections, save_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    dr = dataset_instance.range_axis[1] - dataset_instance.range_axis[0]
    dv = dataset_instance.velocity_axis[1] - dataset_instance.velocity_axis[0]
    extent = [dataset_instance.range_axis[0] - dr/2, dataset_instance.range_axis[-1] + dr/2,
              dataset_instance.velocity_axis[0] - dv/2, dataset_instance.velocity_axis[-1] + dv/2]
    im = ax.imshow(rdm, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"Range-Doppler Map with CFAR Detection - Sample {sample_idx}")
    ax.set_xlim((dataset_instance.range_axis[0], dataset_instance.range_axis[-1]))
    ax.set_ylim((dataset_instance.velocity_axis[0], dataset_instance.velocity_axis[-1]))
    legend_elements = []
    for t, d in matched_pairs:
        ax.scatter(t['range'], t['velocity'], facecolors='none', edgecolors='lime',
                   s=150, linewidth=2)
        ax.plot([t['range'], d['range_m']], [t['velocity'], d['velocity_mps']], 'w--', alpha=0.5)
    for t in unmatched_targets:
        ax.scatter(t['range'], t['velocity'], facecolors='none', edgecolors='red',
                   s=150, linewidth=2)
    for d in [p[1] for p in matched_pairs]:
        ax.scatter(d['range_m'], d['velocity_mps'], marker='x', color='cyan',
                   s=100, linewidth=2)
    for det in unmatched_detections:
        ax.scatter(det['range_m'], det['velocity_mps'], marker='x', color='orange',
                   s=100, linewidth=2)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_elements.extend(by_label.values())
    metrics_text = (
        f"Evaluation Metrics:\n"
        f"-------------------\n"
        f"Targets: {metrics['total_targets']}\n"
        f"Detections: {metrics['tp'] + metrics['fp']}\n"
        f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}\n"
        f"Range Error (MAE): {metrics['mean_range_error']:.2f} m\n"
        f"Vel Error (MAE): {metrics['mean_velocity_error']:.2f} m/s"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def _plot_3d_rdm(dataset_instance, rdm, sample_idx, targets, detections, save_path):
    if VISUALIZATION_AVAILABLE:
        converted_targets = []
        for t in targets:
            ct = t.copy()
            ct['distance'] = t['range']
            converted_targets.append(ct)
        range_res = dataset_instance.range_axis[1] - dataset_instance.range_axis[0]
        vel_res = dataset_instance.velocity_axis[1] - dataset_instance.velocity_axis[0]
        cleaned_detections = []
        if detections:
            for det in detections:
                d_copy = det.copy()
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
            view_range_limits=(dataset_instance.range_axis[0], dataset_instance.range_axis[-1]),
            view_velocity_limits=(dataset_instance.velocity_axis[0], dataset_instance.velocity_axis[-1]),
            is_db=True,
            stride=8
        )

# ======================================================================
# Main Dataset Class
# ======================================================================

class AIRadar_Comm_Dataset(Dataset):
    def __init__(self,
                 config_name='CN0566_TRADITIONAL',
                 num_samples=100,
                 save_path='data/radar_comm_dataset',
                 drawfig=False,
                 clutter_intensity=0.1):
        
        self.config = RADAR_COMM_CONFIGS[config_name]
        self.mode = self.config['mode']
        self.num_samples = num_samples
        self.save_path = save_path
        self.drawfig = drawfig
        self.clutter_intensity = clutter_intensity
        
        # Load params based on mode
        self.fc = self.config['fc']
        self.cfar_params = self.config['cfar_params']
        self.mod_order = self.config.get('mod_order', 4)
        self.channel_model_type = self.config.get('channel_model', 'awgn')
        
        if self.mode == 'TRADITIONAL':
            self.radar_B = self.config['radar_B']
            self.radar_T = self.config['radar_T']
            self.radar_fs = self.config['radar_fs']
            self.radar_slope = self.radar_B / self.radar_T
            self.radar_Ns = int(self.radar_fs * self.radar_T)
            self.radar_Nc = 64
            
            self.comm_B = self.config['comm_B']
            self.comm_fs = self.config['comm_fs']
            self.comm_fft = self.config['comm_fft_size']
            self.comm_cp = self.config['comm_cp_len']
            
        elif self.mode == 'OTFS':
            self.B = self.config['B']
            self.fs = self.config['fs']
            self.Nd = self.config['N_doppler']
            self.Nt = self.config['N_delay']
            
        self.c = 3e8
        self.lambda_c = self.c / self.fc
        
        self.data_samples = []
        
        os.makedirs(self.save_path, exist_ok=True)
        if self.drawfig:
            os.makedirs(os.path.join(self.save_path, 'vis'), exist_ok=True)
            
        self.generate_dataset()

    # ------------------------------------------------------------------
    # Communication Helpers: Modulation & Channel
    # ------------------------------------------------------------------
    def _generate_qam_symbols(self, num_symbols, mod_order=4):
        """Generate random M-QAM symbols"""
        if mod_order == 4:
            # QPSK
            pts = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        elif mod_order == 16:
            # 16-QAM
            x = np.arange(-3, 4, 2)
            y = np.arange(-3, 4, 2)
            X, Y = np.meshgrid(x, y)
            pts = (X + 1j*Y).flatten() / np.sqrt(10)
        elif mod_order == 64:
            # 64-QAM
            x = np.arange(-7, 8, 2)
            y = np.arange(-7, 8, 2)
            X, Y = np.meshgrid(x, y)
            pts = (X + 1j*Y).flatten() / np.sqrt(42)
        else:
            raise ValueError(f"Modulation order {mod_order} not supported yet.")
            
        ints = np.random.randint(0, mod_order, num_symbols)
        symbols = pts[ints]
        return symbols, ints, pts

    def _demodulate_qam(self, rx_symbols, mod_order=4, const_pts=None):
        """Minimum Distance Demodulation"""
        if const_pts is None:
            _, _, const_pts = self._generate_qam_symbols(0, mod_order)
            
        # Broadcast subtract: [N_rx, 1] - [1, M] = [N_rx, M]
        dists = np.abs(rx_symbols[:, None] - const_pts[None, :])
        demod_ints = np.argmin(dists, axis=1)
        return demod_ints

    def _apply_fading_channel(self, signal, fs, snr_db):
        """
        Apply Multipath Fading + AWGN.
        Model: Tapped Delay Line with Exponential Power Decay.
        """
        if self.channel_model_type != 'multipath':
            # Pure AWGN
            sig_pow = np.mean(np.abs(signal)**2)
            noise_pow = sig_pow / (10**(snr_db/10))
            noise = (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))) * np.sqrt(noise_pow/2)
            return signal + noise, np.array([1.0])

        # Realistic Tapped Delay Line (TDL)
        # Random number of taps (2 to 5)
        num_taps = np.random.randint(2, 6)
        
        # Max delay spread
        # Reduced to 200ns (shorter than CP/Symbol duration) to avoid ISI
        max_delay = 200e-9 
        delays_sec = np.sort(np.random.uniform(0, max_delay, num_taps))
        delays_sec[0] = 0 # First tap at 0
        delays_samp = np.round(delays_sec * fs).astype(int)
        
        # Exponential power decay
        powers_db = -np.random.uniform(2, 5, num_taps) * np.arange(num_taps) # e.g. 0, -3, -6...
        powers_lin = 10**(powers_db/10)
        powers_lin /= np.sum(powers_lin) # Normalize energy
        
        # Rayleigh Fading Coefficients
        # h = sqrt(P) * (randn + j*randn)
        taps = np.sqrt(powers_lin) * (np.random.randn(num_taps) + 1j*np.random.randn(num_taps)) / np.sqrt(2)
        
        # Construct Impulse Response (Sparse)
        max_samp = delays_samp[-1]
        h_imp = np.zeros(max_samp + 1, dtype=np.complex128)
        h_imp[delays_samp] = taps
        
        # Apply Channel
        rx_signal_clean = np.convolve(signal, h_imp, mode='full') # Use full to capture all delays
        
        # Add AWGN
        sig_pow = np.mean(np.abs(rx_signal_clean)**2)
        noise_pow = sig_pow / (10**(snr_db/10))
        noise = (np.random.randn(len(rx_signal_clean)) + 1j*np.random.randn(len(rx_signal_clean))) * np.sqrt(noise_pow/2)
        
        return rx_signal_clean + noise, h_imp

    # ------------------------------------------------------------------
    # Simulation: Traditional (OFDM w/ LS Estimation)
    # ------------------------------------------------------------------
    def _simulate_traditional(self, targets, snr_db):
        
        # --- 1. OFDM Communication Simulation ---
        Nfft = self.comm_fft
        Ncp = self.comm_cp
        num_data_syms = 14
        
        # Generate Preamble (Pilot Symbol) - Known at Rx
        _, _, const_pts = self._generate_qam_symbols(0, self.mod_order)
        # Use simple QPSK for Pilot for robustness
        pilot_syms, _, _ = self._generate_qam_symbols(Nfft, mod_order=4) 
        
        # Generate Data
        total_data_qam = num_data_syms * Nfft
        data_syms, data_ints, _ = self._generate_qam_symbols(total_data_qam, self.mod_order)
        data_grid = data_syms.reshape(num_data_syms, Nfft)
        
        # Construct Frame: [Pilot, Data, Data...]
        full_grid = np.vstack([pilot_syms[None, :], data_grid])
        
        # IFFT -> Time
        ifft_out = np.fft.ifft(full_grid, axis=1)
        
        # Add CP
        cp = ifft_out[:, -Ncp:]
        ofdm_time = np.hstack([cp, ifft_out]).flatten()
        
        # Apply Fading Channel
        rx_time_full, h_true = self._apply_fading_channel(ofdm_time, self.comm_fs, snr_db)
        
        # Oracle Synchronization: Align using the first tap delay
        # In practice, this is done via Preamble Correlation
        # Find first significant tap
        first_tap_idx = np.argmax(np.abs(h_true) > 0)
        
        rx_time = rx_time_full[first_tap_idx : first_tap_idx + len(ofdm_time)]
        
        # Rx Processing: Remove CP and FFT
        rx_reshaped = rx_time.reshape(num_data_syms + 1, Nfft + Ncp) # +1 for pilot
        rx_no_cp = rx_reshaped[:, Ncp:]
        rx_grid = np.fft.fft(rx_no_cp, axis=1)
        
        # Channel Estimation (LS)
        Y_pilot = rx_grid[0, :]
        X_pilot = pilot_syms
        H_est = Y_pilot / (X_pilot + 1e-10) # LS Estimate
        
        # Equalization (Zero Forcing)
        Y_data = rx_grid[1:, :]
        # Broadcast H_est
        X_hat_grid = Y_data / (H_est[None, :] + 1e-10)
        rx_const = X_hat_grid.flatten()
        
        # Demodulate
        demod_ints = self._demodulate_qam(rx_const, self.mod_order, const_pts)
        errors = np.sum(data_ints != demod_ints)
        ber = errors / len(data_ints)
        
        # --- 2. FMCW Radar Simulation ---
        # Fixed: Added Hanning Window to reduce sidelobes and False Positives
        Nc = self.radar_Nc
        Ns = self.radar_Ns
        fs = self.radar_fs
        slope = self.radar_slope
        
        t_fast = np.arange(Ns) / fs
        t_slow = np.arange(Nc) * self.radar_T
        
        beat_signal = np.zeros((Nc, Ns), dtype=np.complex64)
        
        for t in targets:
            fb = slope * 2 * t['range'] / self.c
            fd = 2 * t['velocity'] / self.lambda_c
            phase = 2 * np.pi * (fb * t_fast[None, :] + fd * t_slow[:, None])
            amp = np.sqrt(10**(t['rcs']/10))
            beat_signal += amp * np.exp(1j * phase)
            
        sig_pow_rad = np.mean(np.abs(beat_signal)**2)
        if sig_pow_rad > 0:
            noise_pow_rad = sig_pow_rad / (10**(snr_db/10))
            noise_rad = (np.random.randn(Nc, Ns) + 1j*np.random.randn(Nc, Ns)) * np.sqrt(noise_pow_rad/2)
            beat_signal += noise_rad
            
        # Apply Windowing to suppress sidelobes (Fix for High FP)
        win_range = np.hanning(Ns)[None, :]
        win_doppler = np.hanning(Nc)[:, None]
        beat_signal_win = beat_signal * win_range * win_doppler

        r_fft = np.fft.fft(beat_signal_win, axis=1)
        rd_map = np.fft.fftshift(np.fft.fft(r_fft, axis=0), axes=0)
        rd_map_db = 20*np.log10(np.abs(rd_map) + 1e-9)
        
        r_res = (self.c * fs) / (2 * slope * Ns)
        v_res = self.lambda_c / (2 * Nc * self.radar_T)
        r_axis = np.arange(Ns) * r_res
        v_axis = np.arange(-Nc//2, Nc//2) * v_res
        
        return {
            'rd_map': rd_map_db,
            'r_axis': r_axis,
            'v_axis': v_axis,
            'comm_info': {
                'ber': ber,
                'tx_symbols': data_grid.flatten(),
                'rx_symbols': rx_const,
                'num_data_syms': num_data_syms,
                'fft_size': Nfft,
                'tx_ints': data_ints,
                'mod_order': self.mod_order
            },
            'ofdm_map': 20*np.log10(np.abs(X_hat_grid) + 1e-9)
        }

    def _otfs_modulate(self, dd_grid, Nd, Nt):
        """
        Modulates a Delay-Doppler (Nd x Nt) grid to a time-domain signal.
        Tx Chain: ISFFT (DD->TF) -> Heisenberg (TF->Time)
        Aligned with AIradar_datasetv8.py logic.
        """
        # dd_grid: [Nd, Nt] (Doppler, Delay)
        # v8 uses 'F' flattening and axis swaps.
        # Let's replicate v8 exactly.
        # In v8: Ns=Delay, Nc=Doppler. dd_grid input is [Ns, Nc].
        # In g1: Nt=Delay, Nd=Doppler. dd_grid input is [Nd, Nt].
        
        # Transpose to match v8 input [Ns, Nc] -> [Nt, Nd]
        dd_grid_v8 = dd_grid.T 
        
        # v8 Logic:
        # tf_grid = fft(dd_grid, axis=0)
        # tf_grid = ifft(tf_grid, axis=1)
        # time_domain_grid = ifft(tf_grid, axis=0)
        # tx_signal = flatten(order='F')
        
        tf_grid = np.fft.fft(dd_grid_v8, axis=0)
        tf_grid = np.fft.ifft(tf_grid, axis=1)
        time_domain_grid = np.fft.ifft(tf_grid, axis=0)
        tx_signal = time_domain_grid.flatten(order='F')
        
        return tx_signal

    def _otfs_demodulate(self, rx_signal, Nd, Nt):
        """
        Demodulates a time-domain signal back to a Delay-Doppler grid.
        Rx Chain: Wigner (Time->TF) -> SFFT (TF->DD)
        Aligned with AIradar_datasetv8.py logic.
        """
        # v8 Logic:
        # time_domain_grid = rx_signal.reshape((Ns, Nc), order='F')
        # tf_grid = fft(time_domain_grid, axis=0)
        # dd_grid = fft(tf_grid, axis=1)
        # dd_grid = ifft(dd_grid, axis=0)
        
        # Use Nt (Delay/Ns) and Nd (Doppler/Nc)
        time_domain_grid = rx_signal.reshape((Nt, Nd), order='F')
        tf_grid = np.fft.fft(time_domain_grid, axis=0)
        dd_grid = np.fft.fft(tf_grid, axis=1)
        dd_grid = np.fft.ifft(dd_grid, axis=0)
        
        # Transpose back to [Nd, Nt] to match g1 convention
        return dd_grid.T

    # ------------------------------------------------------------------
    # Simulation: OTFS (ISAC w/ Fading & Equalization)
    # ------------------------------------------------------------------
    def _simulate_otfs(self, targets, snr_db):
        """
        Simulates Integrated Sensing and Communication (ISAC) using OTFS.
        Using logic ported from AIradar_datasetv8.py for RDM correctness.
        """
        Nd = self.Nd # Doppler
        Nt = self.Nt # Delay
        
        tx_syms, tx_ints, const_pts = self._generate_qam_symbols(Nd * Nt, self.mod_order)
        dd_comm = tx_syms.reshape(Nd, Nt)
        # Radar pilot grid (single strong delta) to stabilize sensing correlation
        pilot_gain = 6.0
        dd_pilot = np.zeros((Nd, Nt), dtype=np.complex128)
        p_d, p_t = Nd//2, Nt//2
        dd_pilot[p_d, p_t] = pilot_gain
        dd_grid = dd_comm + dd_pilot
        
        # Modulate to time domain (using v8 logic)
        tx_signal = self._otfs_modulate(dd_grid, Nd, Nt)
        
        # 2. RADAR CHANNEL (Monostatic Reflection) - v8 Logic
        n_samples = tx_signal.size
        rx_radar = np.zeros(n_samples, dtype=complex)
        time_vector = np.arange(n_samples) / self.fs
        
        for t in targets:
            range_m = t['range']
            velocity_mps = t['velocity']
            rcs = t['rcs']
            
            amplitude = np.sqrt(10 ** (rcs / 10))
            
            delay_sec = 2 * range_m / self.c
            delay_samples = int(round(delay_sec * self.fs))
            
            if delay_samples < n_samples:
                delayed_signal = np.roll(tx_signal, delay_samples)
                delayed_signal[:delay_samples] = 0
                
                doppler_hz = 2 * velocity_mps * self.fc / self.c
                doppler_shift = np.exp(1j * 2 * np.pi * doppler_hz * time_vector)
                
                rx_radar += amplitude * delayed_signal * doppler_shift
                
        # Radar Noise (AWGN)
        sig_pow = np.mean(np.abs(rx_radar)**2)
        if sig_pow > 0:
            snr_linear = 10**(snr_db/10)
            noise_pow = sig_pow / snr_linear
            noise = (np.random.randn(n_samples) + 1j*np.random.randn(n_samples)) * np.sqrt(noise_pow/2)
            rx_radar += noise
            
        # 3. COMM CHANNEL (One-way Fading) - Keep g1 logic for comms
        # Apply Multipath Fading
        rx_comm_full, h_true = self._apply_fading_channel(tx_signal, self.fs, snr_db)
        
        # Oracle Synchronization
        first_tap_idx = np.argmax(np.abs(h_true) > 0)
        if first_tap_idx + len(tx_signal) <= len(rx_comm_full):
             rx_comm = rx_comm_full[first_tap_idx : first_tap_idx + len(tx_signal)]
        else:
             # Padding if needed (edge case)
             rx_comm = rx_comm_full[first_tap_idx:]
             rx_comm = np.pad(rx_comm, (0, len(tx_signal) - len(rx_comm)))

        # 4. Processing
        
        # --- Comm Receiver ---
        # Equalization in TF domain similar to v8 structure
        rx_time_grid_comm = rx_comm.reshape((Nt, Nd), order='F')
        rx_tf_grid_comm = np.fft.fft(rx_time_grid_comm, axis=0)
        
        # Channel H(f)
        H_freq_1d = np.fft.fft(h_true, n=Nt)
        H_tf = np.tile(H_freq_1d[:, None], (1, Nd)) # Tile across Doppler
        
        rx_tf_eq = rx_tf_grid_comm / (H_tf + 1e-10)
        
        # Continue v8 demodulation steps from TF
        # dd_grid = np.fft.fft(tf_grid, axis=1)
        # dd_grid = np.fft.ifft(dd_grid, axis=0)
        comm_dd_grid_T = np.fft.fft(rx_tf_eq, axis=1)
        comm_dd_grid_T = np.fft.ifft(comm_dd_grid_T, axis=0)
        
        comm_dd_grid = comm_dd_grid_T.T # Back to [Nd, Nt]
        
        rx_const = comm_dd_grid.flatten()
        # Exclude pilot cell from BER calculation to avoid bias
        pilot_flat_idx = p_d * Nt + p_t
        mask = np.ones(rx_const.shape[0], dtype=bool)
        mask[pilot_flat_idx] = False
        demod_ints = self._demodulate_qam(rx_const[mask], self.mod_order, const_pts)
        errors = np.sum(tx_ints[mask] != demod_ints)
        ber = errors / np.sum(mask)
        
        # --- Radar Processor (v8 Logic) ---
        # 1. Demodulate Rx radar signal to DD domain
        rx_dd_radar = self._otfs_demodulate(rx_radar, Nd, Nt)
        
        # DD-domain deconvolution aligned to v8 (with mild windowing)
        w_dopp = np.hanning(Nd)
        w_delay = np.hanning(Nt)
        win2d = np.outer(w_dopp, w_delay)
        rx_win = rx_dd_radar * win2d
        tx_win = dd_grid * win2d

        rx_dd_fft = np.fft.fft2(rx_win)
        tx_dd_fft = np.fft.fft2(tx_win)
        epsilon = 1e-6
        ddm_fft = rx_dd_fft / (tx_dd_fft + epsilon)
        ddm_complex = np.fft.ifft2(ddm_fft)
        
        # 3. Format for RDM
        # ddm_complex is [Nd, Nt] (Doppler, Delay)
        # v8's ddm_db output expects [Doppler, Range]
        # v8's ddm_complex is [Ns, Nc] (Delay, Doppler) -> Transpose -> [Nc, Ns] (Doppler, Delay)
        # g1's ddm_complex is already [Nd, Nt] = [Doppler, Delay]
        
        # However, we must ensure that the output RDM orientation matches what CFAR expects.
        # CFAR typically expects [Doppler, Range].
        
        # In v8: ddm_transposed = ddm_complex.T  # [Nc, Ns]
        # In g1: ddm_complex is already [Nd, Nt] which corresponds to [Doppler, Delay]
        # So we might not need transpose if we just want [Doppler, Delay] for RDM.
        # However, v8 does: ddm_shifted = np.fft.fftshift(ddm_transposed, axes=0)
        # If v8's ddm_transposed is [Doppler, Delay], and it shifts axis 0 (Doppler).
        
        # Let's assume g1's ddm_complex is [Nd, Nt] = [Doppler, Delay].
        # So we just shift axis 0.
        
        ddm_shifted = np.fft.fftshift(ddm_complex, axes=0)
        ddm_mag = np.abs(ddm_shifted)
        rd_map_db_full = 20*np.log10(ddm_mag + 1e-6)
        
        r_res = self.c / (2 * self.fs)
        num_range_bins = int(self.config.get('R_max', 100.0) / r_res)
        num_range_bins = max(1, min(num_range_bins, rd_map_db_full.shape[1]))
        rd_map_db = rd_map_db_full[:, :num_range_bins]
        # Keep absolute dB for CFAR; normalization is applied only in visualization
        r_axis = np.arange(num_range_bins) * r_res
        
        T_sym = self.config.get('T_symbol', 40e-6)
        v_axis = np.fft.fftshift(np.fft.fftfreq(Nd, d=T_sym)) * self.lambda_c / 2
        
        return {
            'rd_map': rd_map_db,
            'r_axis': r_axis,
            'v_axis': v_axis,
            'channel_model': self.channel_model_type,
            'mod_order': self.mod_order,
            'comm_info': {
                'ber': ber,
                'tx_symbols': tx_syms,
                'rx_symbols': rx_const,
                'tx_ints': tx_ints,
                'mod_order': self.mod_order
            }
        }

    # ------------------------------------------------------------------
    # CFAR (Unchanged)
    # ------------------------------------------------------------------
    def _run_cfar(self, rdm_db, r_axis, v_axis):
        """
        Constant False Alarm Rate (CFAR) Detector (CA-CFAR).
        Detects targets in the Range-Doppler Map.
        """
        params = self.cfar_params
        nt = params['num_train']
        ng = params['num_guard']
        thresh = params['threshold_offset']
        
        # Mode-specific preprocessing: use linear magnitude for OTFS CFAR
        if self.mode == 'OTFS':
            lin = np.power(10.0, rdm_db / 20.0)
            gp = self.cfar_params.get('global_percentile', None)
            if gp is not None:
                pval = np.percentile(lin, gp)
                lin = np.minimum(lin, pval)
            norm_rdm = lin
        else:
            norm_rdm = rdm_db.copy()
            gp = self.cfar_params.get('global_percentile', None)
            if gp is not None:
                pval = np.percentile(norm_rdm, gp)
                norm_rdm = np.minimum(norm_rdm, pval)
        
        kernel_size = 1 + 2*(nt + ng)
        kernel = np.ones((kernel_size, kernel_size))
        guard_region = 1 + 2*ng
        start_g = nt
        end_g = nt + guard_region
        kernel[start_g:end_g, start_g:end_g] = 0
        kernel /= np.sum(kernel)
        noise_mu = convolve2d(norm_rdm, kernel, mode='same', boundary='symm')
        noise_sq = convolve2d(norm_rdm**2, kernel, mode='same', boundary='symm')
        noise_sigma = np.sqrt(np.maximum(noise_sq - noise_mu**2, 1e-6))
        if self.cfar_params.get('use_sigma', False):
            detections = norm_rdm > (noise_mu + self.cfar_params.get('std_scale', 3.0) * noise_sigma + thresh)
        else:
            detections = norm_rdm > (noise_mu + thresh)
        
        # Non-Maximum Suppression (NMS)
        if params['nms_kernel_size'] > 1:
            local_max = maximum_filter(norm_rdm, size=params['nms_kernel_size'])
            detections = detections & (norm_rdm == local_max)
            
        idxs = np.argwhere(detections)
        results = []
        min_r = params.get('min_range_m', 0.0)
        min_v = params.get('min_speed_mps', 0.0)
        notch_k = params.get('notch_doppler_bins', 0)
        center = len(v_axis) // 2
        candidates = []
        for idx in idxs:
            d_idx, r_idx = idx
            if d_idx >= len(v_axis) or r_idx >= len(r_axis): continue
            range_m = r_axis[r_idx]
            vel_mps = v_axis[d_idx]
            # Filter artifacts and near-zero clutter
            if range_m < min_r or abs(vel_mps) < min_v: continue
            if notch_k > 0 and abs(d_idx - center) <= notch_k: continue
            candidates.append({
                'range_m': range_m,
                'velocity_mps': vel_mps,
                'range_idx': r_idx,
                'doppler_idx': d_idx,
                'power': norm_rdm[d_idx, r_idx]
            })
        # Limit number of peaks by score (power)
        max_peaks = params.get('max_peaks', None)
        if max_peaks is not None:
            candidates.sort(key=lambda x: x['power'], reverse=True)
            candidates = candidates[:max_peaks]

        # Connected-component pruning: retain local maxima within neighborhoods
        pruned = []
        taken = set()
        neigh = params.get('nms_kernel_size', 5)
        for det in candidates:
            key = (det['doppler_idx']//neigh, det['range_idx']//neigh)
            if key in taken:
                continue
            pruned.append(det)
            taken.add(key)
        return pruned
    
    # ------------------------------------------------------------------
    # Metrics Calculation
    # ------------------------------------------------------------------
    def _evaluate_metrics(self, targets, detections, match_dist_thresh=3.0):
        tp = 0
        range_errors = []
        velocity_errors = []
        unmatched_targets = targets.copy()
        unmatched_detections = detections.copy()
        matched_pairs = []
        for target in targets:
            best_dist = float('inf')
            best_det_idx = -1
            for i, det in enumerate(unmatched_detections):
                d_r = target['range'] - det['range_m']
                d_v = target['velocity'] - det['velocity_mps']
                dist = np.sqrt(d_r**2 + d_v**2)
                if dist < match_dist_thresh and dist < best_dist:
                    best_dist = dist
                    best_det_idx = i
            if best_det_idx != -1:
                tp += 1
                det = unmatched_detections[best_det_idx]
                range_errors.append(abs(target['range'] - det['range_m']))
                velocity_errors.append(abs(target['velocity'] - det['velocity_mps']))
                matched_pairs.append((target, det))
                unmatched_detections.pop(best_det_idx)
                unmatched_targets.remove(target)
        fp = len(unmatched_detections)
        fn = len(targets) - tp
        metrics = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'mean_range_error': np.mean(range_errors) if range_errors else 0.0,
            'mean_velocity_error': np.mean(velocity_errors) if velocity_errors else 0.0,
            'total_targets': len(targets)
        }
        return metrics, matched_pairs, unmatched_targets, unmatched_detections

    # ------------------------------------------------------------------
    # Data Generation Loop
    # ------------------------------------------------------------------
    def generate_dataset(self):
        print(f"Generating {self.num_samples} samples in {self.mode} mode...")
        print(f"Config: {self.mod_order}-QAM | Channel: {self.channel_model_type}")
        
        for i in tqdm(range(self.num_samples)):
            num_t = np.random.randint(1, 4)
            targets = []
            for _ in range(num_t):
                targets.append({
                    'range': np.random.uniform(5, self.config['R_max'] * 0.8),
                    'velocity': np.random.uniform(-15, 15),
                    'rcs': np.random.uniform(10, 30)
                })
                
            snr = np.random.uniform(25, 40) # Increased SNR for better BER
            
            if self.mode == 'TRADITIONAL':
                out = self._simulate_traditional(targets, snr)
            else:
                out = self._simulate_otfs(targets, snr)
                
            dets = self._run_cfar(out['rd_map'], out['r_axis'], out['v_axis'])
            
            self.range_axis = out['r_axis']
            self.velocity_axis = out['v_axis']
            sample = {
                'mode': self.mode,
                'mod_order': self.mod_order,
                'channel_model': self.channel_model_type,
                'range_doppler_map': torch.tensor(out['rd_map'], dtype=torch.float32),
                'range_axis': out['r_axis'],
                'velocity_axis': out['v_axis'],
                'target_info': {'targets': targets, 'snr_db': snr},
                'comm_info': out['comm_info'],
                'cfar_detections': dets,
                'ofdm_map': out.get('ofdm_map', None)
            }
            
            # Simple per-sample error metric for dataset attribute
            errs = []
            for t in targets:
                dists = [abs(t['range'] - d['range_m']) for d in dets]
                if dists: errs.append(min(dists))
            mean_err = np.mean(errs) if errs else 0.0
            sample['metrics'] = {'mean_range_error': mean_err}
            
            self.data_samples.append(sample)
            
            if self.drawfig:
                plot_combined_sample(sample, os.path.join(self.save_path, f'vis/sample_{i}_{self.mode}.png'))
                rdm = sample['range_doppler_map'].numpy()
                rdm_norm = rdm - np.max(rdm)
                metrics, matched_pairs, unmatched_targets, unmatched_detections = self._evaluate_metrics(targets, dets)
                _plot_2d_rdm(self, rdm_norm, i, metrics,
                             matched_pairs, unmatched_targets, unmatched_detections,
                             os.path.join(self.save_path, f'vis/rdm_sample_{i}.png'))
                _plot_3d_rdm(self, rdm_norm, i, targets, dets,
                             os.path.join(self.save_path, f'vis/rdm_3d_sample_{i}.png'))

            # Dump minimal tensors for DL script
            dump_item = {
                'range_doppler_map': sample['range_doppler_map'].numpy(),
                'cfar_detections': sample['cfar_detections'],
                'target_info': sample['target_info'],
                'ofdm_map': sample.get('ofdm_map', None),
                'comm_info': sample.get('comm_info', None)
            }
            dump_path = os.path.join(self.save_path, 'joint_dump.npy')
            existing = []
            if os.path.exists(dump_path):
                try:
                    existing = list(np.load(dump_path, allow_pickle=True))
                except Exception:
                    existing = []
            existing.append(dump_item)
            np.save(dump_path, np.array(existing, dtype=object))

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return self.data_samples[idx]

def evaluate_dataset_metrics(dataset, name):
    """Aggregate metrics across the entire dataset"""
    total_tp, total_fp, total_fn = 0, 0, 0
    total_targets = 0
    all_range_errors = []
    all_vel_errors = []
    
    print(f"\n--- Evaluating Metrics for {name} ---")
    
    for i in range(len(dataset)):
        sample = dataset[i]
        targets = sample['target_info']['targets']
        detections = sample['cfar_detections']
        
        metrics, _, _, _ = dataset._evaluate_metrics(targets, detections)
        
        total_tp += metrics['tp']
        total_fp += metrics['fp']
        total_fn += metrics['fn']
        total_targets += metrics['total_targets']
        
        if metrics['tp'] > 0: # Approximation for aggregation
            # Re-calculating to append to list, or just use what we have if we returned lists
            # For simplicity, relying on tp/fp counts mostly. 
            pass 
            
    # Calculate Precision/Recall/F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate average errors across valid samples
    avg_range_error = np.mean([s['metrics']['mean_range_error'] for s in dataset])
    
    # Add BER Stats
    all_ber = [d['comm_info']['ber'] for d in dataset]
    avg_ber = np.mean(all_ber) if all_ber else 0.0
    
    print(f"  > Total Targets: {total_targets}")
    print(f"  > True Positives (TP): {total_tp}")
    print(f"  > False Positives (FP): {total_fp}")
    print(f"  > False Negatives (FN): {total_fn}")
    print(f"  > Precision: {precision:.4f}")
    print(f"  > Recall: {recall:.4f}")
    print(f"  > F1-Score: {f1:.4f}")
    print(f"  > Mean Range Error: {avg_range_error:.4f} m")
    print(f"  > Mean BER: {avg_ber:.5f}")
    print("-" * 40)

# ======================================================================
# Run Demonstration
# ======================================================================
if __name__ == "__main__":
    output_base_dir = "data/AIradar_comm_dataset_g1b"
    
    print(f"\n{'='*60}")
    print(f"Starting Comprehensive Demonstration")
    print(f"Output Directory: {output_base_dir}")
    print(f"{'='*60}\n")

    for config_name in RADAR_COMM_CONFIGS.keys():
        if 'OTFS' in config_name:
            print(f"Skipping {config_name} (OTFS not stable yet)")
            continue
        print(f"\n--- Testing Configuration: {config_name} ---")
        save_path = os.path.join(output_base_dir, config_name)
        
        # Instantiate Dataset (generates data & visualizations)
        ds = AIRadar_Comm_Dataset(
            config_name=config_name, 
            num_samples=5, 
            save_path=save_path, 
            drawfig=True
        )
        
        # Run aggregated evaluation
        evaluate_dataset_metrics(ds, config_name)
        
        print(f"Generating detailed visualizations for {config_name}...")
        print(f"Visualizations saved to {os.path.join(save_path, 'vis')}")

    print(f"\n{'='*60}")
    print("Demonstration Complete.")
    print(f"{'='*60}")

#all FMCW and traditional approaches are work well, OTFS solutions have problems
