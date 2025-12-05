import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from scipy import signal
import sys

#from AIRadarLib.datautil import normalize_spectrum, find_peak_frequency, apply_window, calculate_spectrum

def apply_window(signal, window_type="blackman"):
    """Apply a window function to the signal."""
    if window_type == "blackman":
        window = np.blackman(len(signal))
    elif window_type == "hamming":
        window = np.hamming(len(signal))
    elif window_type == "hann":
        window = np.hanning(len(signal))
    elif window_type == "rect":
        window = np.ones(len(signal))
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    return signal * window, window

def calculate_spectrum(signal, N_fft):
    """Compute the FFT and return the magnitude spectrum in dB."""
    fft_data = np.fft.fftshift(np.fft.fft(signal, n=N_fft))
    spectrum = 20 * np.log10(np.abs(fft_data) + 1e-10)
    return spectrum

def normalize_spectrum(spectrum, reference=None):
    """Normalize the spectrum for better comparison."""
    if reference is None:
        reference = np.max(spectrum)
    return spectrum - reference

def find_peak_frequency(spectrum, freq_axis):
    """Find the peak frequency and its value in the spectrum."""
    peak_idx = np.argmax(spectrum)
    peak_freq = freq_axis[peak_idx]
    peak_val = spectrum[peak_idx]
    return peak_freq, peak_val


def plot_detection_results(
    rd_map,
    target_mask,
    targets,
    detection_results,
    range_resolution,
    velocity_resolution,
    num_doppler_bins,
    num_range_bins,
    save_path=None,
    title="Radar Detection Results",
    show_plot=True,
    figsize=(12, 10),
    dpi=100,
    apply_doppler_centering=True
):
    """
    Plot radar detection results, target mask, and ground truth target locations in a single figure.
    
    Args:
        rd_map: Range-Doppler map with shape [num_rx, 2, num_doppler_bins, num_range_bins]
        target_mask: Target mask with shape [num_doppler_bins, num_range_bins, 1]
        targets: List of target dictionaries with 'distance' and 'velocity' keys
        detection_results: List of detection dictionaries with 'range_idx', 'doppler_idx', etc.
        range_resolution: Range resolution in meters
        velocity_resolution: Velocity resolution in m/s
        num_doppler_bins: Number of Doppler bins
        num_range_bins: Number of range bins
        save_path: Path to save the figure (if None, figure is not saved)
        title: Title of the figure
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height) in inches
        dpi: DPI for the figure
        apply_doppler_centering: bool, whether Doppler and Range FFT are centered (affects target position calculation)
        
    Returns:
        Figure and axes objects
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Extract magnitude of range-Doppler map (use first RX antenna)
    rd_magnitude = np.sqrt(rd_map[0, 0, :, :]**2 + rd_map[0, 1, :, :]**2)
    
    # Normalize RD map for better visualization
    rd_magnitude_norm = 20 * np.log10(rd_magnitude / np.max(rd_magnitude) + 1e-10)
    rd_magnitude_norm = np.clip(rd_magnitude_norm, -40, 0)  # Clip to dynamic range
    
    # Create range and Doppler axes
    range_axis = np.arange(num_range_bins) * range_resolution
    doppler_axis = (np.arange(num_doppler_bins) - num_doppler_bins // 2) * velocity_resolution
    
    # Plot range-Doppler map
    im = ax.imshow(
        rd_magnitude_norm,
        aspect='auto',
        cmap='jet',
        origin='lower',
        extent=[0, range_axis[-1], doppler_axis[0], doppler_axis[-1]],
        interpolation='none',
        vmin=-40,
        vmax=0
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Magnitude (dB)')
    
    # Create a custom colormap for the target mask (transparent to red)
    colors = [(1, 0, 0, 0), (1, 0, 0, 0.7)]  # Red with varying alpha
    target_cmap = LinearSegmentedColormap.from_list('target_mask', colors)
    
    # Plot target mask as overlay with transparency
    if target_mask is not None:
        # Reshape if needed and transpose for correct orientation
        #Binary mask with shape [num_doppler_bins, num_range_bins, 1]
        mask_plot = target_mask.reshape(num_doppler_bins, num_range_bins)
        ax.imshow(
            mask_plot,
            aspect='auto',
            cmap=target_cmap,
            origin='lower',
            extent=[0, range_axis[-1], doppler_axis[0], doppler_axis[-1]],
            interpolation='none'
        )
    
    # Plot ground truth target locations
    if targets:
        # Use the actual distance and velocity values directly
        # No need to adjust for centering as we're plotting in physical units (meters and m/s)
        # not in bin indices
        target_ranges = [target['distance'] for target in targets]
        target_velocities = [target['velocity'] for target in targets]
        ax.scatter(
            target_ranges,
            target_velocities,
            c='lime',
            marker='o',
            s=100,
            edgecolors='black',
            linewidths=1.5,
            label='Ground Truth'
        )
    
    # Plot CFAR detection results
    if detection_results:
        detection_ranges = []
        detection_velocities = []
        
        for detection in detection_results:
            # Check if detection has range_idx and doppler_idx or range and velocity
            if 'range_idx' in detection and 'doppler_idx' in detection:
                range_val = detection['range_idx'] * range_resolution
                doppler_val = (detection['doppler_idx'] - num_doppler_bins // 2) * velocity_resolution
            elif 'range' in detection and 'velocity' in detection:
                range_val = detection['range']
                doppler_val = detection['velocity']
            else:
                continue
                
            detection_ranges.append(range_val)
            detection_velocities.append(doppler_val)
        
        if detection_ranges:
            ax.scatter(
                detection_ranges,
                detection_velocities,
                c='white',
                marker='x',
                s=80,
                linewidths=2,
                label='CFAR Detections'
            )
    
    # Set labels and title
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    # Set axis limits
    ax.set_xlim(0, range_axis[-1])
    ax.set_ylim(doppler_axis[0], doppler_axis[-1])
    
    # Add text with detection statistics
    if targets and detection_results:
        num_targets = len(targets)
        num_detections = len(detection_results)
        
        stats_text = f"Targets: {num_targets}\nDetections: {num_detections}"
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax

def plot_signal_comparison(original_signal, processed_signal, fs, title, time_domain_ylim=None):
    """
    Plot time and frequency domain comparison of original and processed signals
    
    Args:
        original_signal: The original input signal
        processed_signal: The signal after processing
        fs: Sampling frequency in Hz
        title: Plot title
        time_domain_ylim: Y-axis limits for time domain plot (optional)
    """
    # Create figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Time vector for plotting (only show a portion for clarity)
    plot_samples = min(1000, len(original_signal))
    t = np.arange(plot_samples) / fs
    
    # Time domain plots
    axs[0, 0].plot(t, np.real(original_signal[:plot_samples]), 'b-', label='Real')
    axs[0, 0].plot(t, np.imag(original_signal[:plot_samples]), 'r-', label='Imag')
    axs[0, 0].set_title('Original Signal (Time Domain)')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].legend()
    if time_domain_ylim:
        axs[0, 0].set_ylim(time_domain_ylim)
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(t, np.real(processed_signal[:plot_samples]), 'b-', label='Real')
    axs[0, 1].plot(t, np.imag(processed_signal[:plot_samples]), 'r-', label='Imag')
    axs[0, 1].set_title('Processed Signal (Time Domain)')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].legend()
    if time_domain_ylim:
        axs[0, 1].set_ylim(time_domain_ylim)
    axs[0, 1].grid(True)
    
    # Frequency domain plots
    f_orig, Pxx_orig = signal.welch(original_signal, fs, nperseg=1024, return_onesided=False)
    f_orig = np.fft.fftshift(f_orig)
    Pxx_orig = np.fft.fftshift(Pxx_orig)
    
    f_proc, Pxx_proc = signal.welch(processed_signal, fs, nperseg=1024, return_onesided=False)
    f_proc = np.fft.fftshift(f_proc)
    Pxx_proc = np.fft.fftshift(Pxx_proc)
    
    # Convert to dB
    Pxx_orig_db = 10 * np.log10(Pxx_orig + 1e-10)
    Pxx_proc_db = 10 * np.log10(Pxx_proc + 1e-10)
    
    axs[1, 0].plot(f_orig, Pxx_orig_db)
    axs[1, 0].set_title('Original Signal (Frequency Domain)')
    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].set_ylabel('Power Spectral Density (dB/Hz)')
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(f_proc, Pxx_proc_db)
    axs[1, 1].set_title('Processed Signal (Frequency Domain)')
    axs[1, 1].set_xlabel('Frequency (Hz)')
    axs[1, 1].set_ylabel('Power Spectral Density (dB/Hz)')
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_range_doppler_map_with_ground_truth(
    rd_map,
    targets,
    range_resolution,
    velocity_resolution,
    num_range_bins,
    num_doppler_bins,
    title_prefix='',
    save_path=None,
    apply_doppler_centering=True
):
    """
    Plot a 2D Range-Doppler map with ground truth target annotations.

    Args:
        rd_map: np.ndarray, shape (2, num_doppler_bins, num_range_bins), real and imaginary parts of RD map
        targets: list of dicts, each with keys 'distance', 'velocity', 'rcs'
        range_resolution: float, range resolution in meters
        velocity_resolution: float, velocity resolution in m/s
        num_range_bins: int, number of range bins
        num_doppler_bins: int, number of Doppler bins
        title_prefix: str, prefix for plot title/labeling/saving)
        save_path: str, directory to save the figure
        apply_doppler_centering: bool, whether Doppler FFT is centered (affects target position calculation)
    """
    rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2) #(1024, 1024)
    rd_db = 20 * np.log10(rd_magnitude + 1e-10)
    vmin = np.max(rd_db) - 40  # Dynamic range of 40 dB

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Range-Doppler Map with Ground Truth - {title_prefix}", fontsize=16)
    plt.imshow(rd_db, aspect='auto', cmap='jet', vmin=vmin)
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    plt.title('Range-Doppler Map')

    for i, target in enumerate(targets):
        # Calculate range and Doppler bins for each target
        if apply_doppler_centering:
            # When centered, zero range is at num_range_bins/2
            range_bin = int(num_range_bins // 2 + target['distance'] / range_resolution)
            # When centered, zero velocity is at num_doppler_bins/2
            doppler_bin = int(num_doppler_bins // 2 + target['velocity'] / velocity_resolution)
        else:
            # When not centered, zero range is at bin 0
            range_bin = int(target['distance'] / range_resolution)
            # When not centered, zero velocity is at bin 0
            doppler_bin = int(target['velocity'] / velocity_resolution) % num_doppler_bins
            
        if (0 <= range_bin < num_range_bins and 0 <= doppler_bin < num_doppler_bins):
            plt.plot(range_bin, doppler_bin, 'ro', markersize=10, markeredgecolor='white')
            plt.text(
                range_bin + 2, doppler_bin,
                f"Target {i+1}\nR: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s\nRCS: {target['rcs']:.1f}m²",
                color='white', fontsize=9, backgroundcolor='black',
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='white', boxstyle='round')
            )
    # if len(targets) == 0:
    #     plt.text(
    #         num_range_bins//2, num_doppler_bins//2,
    #         "No targets in this scene",
    #         color='white', fontsize=12, backgroundcolor='red',
    #         ha='center', va='center'
    #     )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    # os.makedirs(save_path, exist_ok=True)
    # plt.savefig(os.path.join(save_path, f'sample_{sample_idx}_rd_map_2d.png'))
    # plt.close()

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

def plot_cfar_vs_ground_truth(
    targets,
    detection_results,
    sample_idx,
    range_resolution,
    velocity_resolution,
    num_range_bins,
    num_doppler_bins,
    save_path,
    create_target_mask_func,
    apply_doppler_centering=True
):
    """
    Plot CFAR detection results versus ground truth target mask.

    Args:
        targets: list of dicts, each with keys 'distance', 'velocity', 'rcs'
        detection_results: list of dicts, each with keys 'range_bin', 'doppler_bin', 'distance', 'velocity', 'snr'
        sample_idx: int, index of the current sample (for labeling/saving)
        range_resolution: float, range resolution in meters
        velocity_resolution: float, velocity resolution in m/s
        num_range_bins: int, number of range bins
        num_doppler_bins: int, number of Doppler bins
        save_path: str, directory to save the figure
        create_target_mask_func: function, generates a target mask from targets
        apply_doppler_centering: bool, whether Doppler and Range FFT are centered (affects target position calculation)
    """
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"CFAR Detection vs Ground Truth - Sample {sample_idx}", fontsize=16)
    # Ground truth target mask
    plt.subplot(2, 1, 1)
    target_mask = create_target_mask_func(targets)
    plt.imshow(target_mask[:, :, 0], aspect='auto', cmap='gray')
    plt.colorbar(label='Target Presence')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    plt.title('Ground Truth Target Mask')
    for target in targets:
        if apply_doppler_centering:
            # When centered, zero range is at num_range_bins/2
            range_bin = int(num_range_bins // 2 + target['distance'] / range_resolution)
            # When centered, zero velocity is at num_doppler_bins/2
            doppler_bin = int(num_doppler_bins // 2 + target['velocity'] / velocity_resolution)
        else:
            # When not centered, zero range is at bin 0
            range_bin = int(target['distance'] / range_resolution)
            # When not centered, zero velocity is at bin 0
            doppler_bin = int(target['velocity'] / velocity_resolution) % num_doppler_bins
        if (0 <= range_bin < num_range_bins and 0 <= doppler_bin < num_doppler_bins):
            plt.plot(range_bin, doppler_bin, 'ro', markersize=8)
            plt.text(
                range_bin + 1, doppler_bin + 1,
                f"R: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s",
                color='white', fontsize=8, backgroundcolor='black'
            )
    # CFAR detection results
    plt.subplot(2, 1, 2)
    cfar_map = np.zeros((num_doppler_bins, num_range_bins))
    for target in detection_results:
        cfar_map[target['doppler_bin'], target['range_bin']] = 1
    plt.imshow(cfar_map, aspect='auto', cmap='gray')
    plt.colorbar(label='Detection')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    plt.title('CFAR Detection Results')
    for target in detection_results:
        plt.plot(target['range_bin'], target['doppler_bin'], 'bo', markersize=8)
        plt.text(
            target['range_bin'] + 1, target['doppler_bin'] + 1,
            f"R: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s\nSNR: {target['snr']:.1f}dB",
            color='white', fontsize=8, backgroundcolor='blue'
        )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'sample_{sample_idx}_detection.png'))
    plt.close()

def plot_signal_time_and_spectrum(
    signal,
    sample_rate,
    total_duration,
    title_prefix="Signal",
    window_type="blackman",
    N_fft=8192,
    bandwidth=None,
    center_freq=None,
    zoom_margin=0,
    textstr=None,
    highlight_peak=True,
    normalize=True,
    save_path=None,
    draw_window=True
):
    """
    Plot the time-domain and spectrum of a complex signal with advanced options and optimized clarity.

    Args:
        signal: np.ndarray, 1D complex array (the signal to plot)
        sample_rate: float, sample rate in Hz
        total_duration: float, total duration of the signal in seconds
        title_prefix: str, prefix for plot titles
        window_type: str, type of window to use for spectrum ('blackman', 'hamming', 'hann', 'rect')
        N_fft: int, FFT size
        bandwidth: float or None, bandwidth in Hz to highlight and zoom
        center_freq: float or None, center frequency in Hz to highlight region (used with bandwidth)
        zoom_margin: float, fraction of bandwidth for zoom margin
        textstr: str or None, text to display on spectrum plot
        highlight_peak: bool, whether to highlight the peak frequency
        normalize: bool, whether to normalize spectra for comparison
        save_path: str or None, if provided, save the figure to this path
        draw_window: bool, whether to draw the window in the time-domain plot
    """
    # Enhanced time axis with microsecond units for better readability
    t = np.linspace(0, total_duration, len(signal))
    t_us = t * 1e6  # Convert to microseconds for better readability
    
    signal_windowed, window = apply_window(signal, window_type=window_type)
    
    # Enhanced frequency axis handling
    if center_freq is not None:
        freq_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1 / sample_rate)) + center_freq
    else:
        freq_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1 / sample_rate))
    freq_axis_mhz = freq_axis / 1e6  # MHz

    # Compute spectra with enhanced dynamic range
    spectrum_orig = calculate_spectrum(signal, N_fft)
    spectrum_win = calculate_spectrum(signal_windowed, N_fft)

    # Enhanced normalization with better dynamic range control
    if normalize:
        ref = max(np.max(spectrum_orig), np.max(spectrum_win))
        spectrum_orig = normalize_spectrum(spectrum_orig, reference=ref)
        spectrum_win = normalize_spectrum(spectrum_win, reference=ref)
        # Limit dynamic range to improve visibility
        spectrum_orig = np.maximum(spectrum_orig, -80)  # 80 dB dynamic range
        spectrum_win = np.maximum(spectrum_win, -80)

    # Find peak frequency
    peak_freq, peak_val = find_peak_frequency(spectrum_win, freq_axis_mhz) if highlight_peak else (None, None)

    # Create figure with optimized size and layout
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{title_prefix} - Time and Spectrum", fontsize=16, fontweight='bold')

    # Enhanced time domain plot
    axs[0].plot(t_us, np.real(signal), 'b-', label='Real', linewidth=1.5, alpha=0.8)
    axs[0].plot(t_us, np.imag(signal), 'r--', label='Imaginary', linewidth=1.5, alpha=0.8)
    if draw_window and len(window) == len(signal):
        # Scale window to signal amplitude for better visualization
        window_scaled = window * np.max(np.abs(signal)) * 0.8
        axs[0].plot(t_us, window_scaled, 'g-', label=f'{window_type.capitalize()} Window', 
                   linewidth=2, alpha=0.6)
    
    axs[0].set_title(f"{title_prefix} (Time Domain)", fontsize=14, fontweight='bold')
    axs[0].set_xlabel("Time (μs)", fontsize=12)
    axs[0].set_ylabel("Amplitude", fontsize=12)
    axs[0].legend(loc='upper right', framealpha=0.9)
    axs[0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    axs[0].set_facecolor('#f8f9fa')
    
    # Add minor ticks for better precision
    axs[0].minorticks_on()
    axs[0].grid(True, which='minor', alpha=0.1, linestyle=':')

    # Enhanced spectrum plot with better colors and styling
    if draw_window:
        axs[1].plot(freq_axis_mhz, spectrum_orig, 'b-', label='Original', 
                   linewidth=1.2, alpha=0.6)
        axs[1].plot(freq_axis_mhz, spectrum_win, 'r-', 
                   label=f'{window_type.capitalize()} Windowed', 
                   linewidth=2, alpha=0.9)
    else:
        axs[1].plot(freq_axis_mhz, spectrum_orig, 'b-', label='Spectrum', 
                   linewidth=2, alpha=0.9)
    
    axs[1].set_title(f"{title_prefix} Spectrum", fontsize=14, fontweight='bold')
    axs[1].set_xlabel("Frequency (MHz)", fontsize=12)
    axs[1].set_ylabel("Magnitude (dB)", fontsize=12)
    axs[1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    axs[1].set_facecolor('#f8f9fa')
    
    # Add minor ticks for better precision
    axs[1].minorticks_on()
    axs[1].grid(True, which='minor', alpha=0.1, linestyle=':')

    # Enhanced bandwidth highlighting with better visual cues
    if bandwidth is not None and center_freq is not None:
        bandwidth_mhz = bandwidth / 1e6
        center_freq_mhz = center_freq / 1e6
        f_start = center_freq_mhz - bandwidth_mhz / 2
        f_end = center_freq_mhz + bandwidth_mhz / 2
        
        # Enhanced bandwidth visualization
        axs[1].axvspan(f_start, f_end, alpha=0.15, color='orange', 
                      label=f'Bandwidth: {bandwidth_mhz:.1f} MHz')
        axs[1].axvline(f_start, color='red', linestyle='--', linewidth=2, 
                      label=f'Start: {f_start:.1f} MHz')
        axs[1].axvline(f_end, color='green', linestyle='--', linewidth=2, 
                      label=f'End: {f_end:.1f} MHz')
        axs[1].axvline(center_freq_mhz, color='purple', linestyle=':', linewidth=2, 
                      label=f'Center: {center_freq_mhz:.1f} MHz')
        
        # Smart zoom with margin
        if zoom_margin > 0:
            margin = bandwidth_mhz * zoom_margin
            axs[1].set_xlim([f_start - margin, f_end + margin])
    elif bandwidth is not None and zoom_margin > 0:
        bandwidth_mhz = bandwidth / 1e6
        f_start = -bandwidth_mhz / 2
        f_end = bandwidth_mhz / 2
        
        axs[1].axvspan(f_start, f_end, alpha=0.15, color='orange', 
                      label=f'Bandwidth: {bandwidth_mhz:.1f} MHz')
        axs[1].axvline(f_start, color='red', linestyle='--', linewidth=2)
        axs[1].axvline(f_end, color='green', linestyle='--', linewidth=2)
        
        margin = bandwidth_mhz * zoom_margin
        axs[1].set_xlim([f_start - margin, f_end + margin])

    # Enhanced peak highlighting
    if highlight_peak and peak_freq is not None:
        axs[1].axvline(peak_freq, color='magenta', linestyle='-', linewidth=3, 
                      label=f'Peak: {peak_freq:.2f} MHz')
        # Better annotation positioning
        y_range = axs[1].get_ylim()
        annotation_y = peak_val + (y_range[1] - y_range[0]) * 0.1
        axs[1].annotate(f'Peak\n{peak_freq:.2f} MHz\n{peak_val:.1f} dB', 
                       xy=(peak_freq, peak_val), 
                       xytext=(peak_freq, annotation_y),
                       arrowprops=dict(arrowstyle='->', color='magenta', lw=2),
                       fontsize=10, color='magenta', fontweight='bold',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='magenta', alpha=0.8))

    # Enhanced legend positioning
    axs[1].legend(loc='upper right', framealpha=0.9, fontsize=10)

    # Enhanced text display
    if textstr is not None:
        props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                    edgecolor='navy', alpha=0.8)
        axs[1].text(0.02, 0.98, textstr, transform=axs[1].transAxes, 
                   fontsize=10, verticalalignment='top', bbox=props,
                   fontweight='bold')

    # Enhanced layout and styling
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Improve overall appearance
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_instantaneous_frequency(signal, sample_rate, total_duration, slope, bandwidth=None, center_freq=None, 
                                title_prefix="Chirp", textstr=None, save_path=None, draw_window=False):
    """
    Plot the instantaneous frequency of a chirp signal to visualize the frequency sweeping band.
    
    Args:
        signal: Complex chirp signal
        sample_rate: Sample rate in Hz
        total_duration: Total duration of the signal in seconds
        slope: FMCW slope in Hz/s
        bandwidth: Signal bandwidth in Hz (optional)
        center_freq: Center frequency in Hz (optional)
        title_prefix: Prefix for the plot title
        textstr: Additional text to display on the plot
        save_path: Path to save the figure
        draw_window: Whether to draw window function
    """

    
    # Create time axis
    num_samples = len(signal)
    t = np.linspace(0, total_duration, num_samples)
    
    # Calculate instantaneous frequency
    # For FMCW chirp: f(t) = f0 + slope * t
    # where f0 is the starting frequency
    if bandwidth is not None:
        f0 = center_freq - bandwidth/2 if center_freq is not None else 0
        inst_freq = f0 + slope * t
    else:
        # If bandwidth not provided, just show relative frequency change
        inst_freq = slope * t
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot instantaneous frequency
    plt.plot(t * 1e6, inst_freq / 1e6, 'b-', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Time (μs)')
    plt.ylabel('Frequency (MHz)')
    plt.title(f'{title_prefix} Instantaneous Frequency')
    plt.grid(True)
    
    # Add bandwidth markers if provided
    if bandwidth is not None:
        if center_freq is not None:
            plt.axhline(y=(center_freq - bandwidth/2)/1e6, color='r', linestyle='--', label='Start Frequency')
            plt.axhline(y=(center_freq + bandwidth/2)/1e6, color='g', linestyle='--', label='End Frequency')
        else:
            plt.axhline(y=0, color='r', linestyle='--', label='Start Frequency')
            plt.axhline(y=bandwidth/1e6, color='g', linestyle='--', label='End Frequency')
        plt.legend()
    
    # Add text information if provided
    if textstr is not None:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()