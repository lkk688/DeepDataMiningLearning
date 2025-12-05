import numpy as np
from scipy.signal import convolve2d
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
from scipy.ndimage import label
import torch
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
#pip install numpy matplotlib scipy
import matplotlib.pyplot as plt
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch.nn.functional as F

def cfar_2d_detect(
    rd_map,             # shape: [num_rx, 2, num_doppler, num_range]
    num_train=8,        # Number of training cells in each dimension
    num_guard=4,        # Number of guard cells in each dimension
    rate_fa=1e-5,       # False alarm rate
    range_res=0.5,      # meters per range bin
    doppler_res=0.25,   # m/s per doppler bin
    max_range=100,      # meters
    max_speed=50        # m/s
):
    num_rx, _, num_doppler, num_range = rd_map.shape

    # Step 1: Combine IQ channels to get magnitude
    real = rd_map[:, 0]
    imag = rd_map[:, 1]
    mag = np.sqrt(real**2 + imag**2)  # [num_rx, doppler, range]

    # Step 2: Average over receivers
    mag = np.mean(mag, axis=0)  # [doppler, range]

    # Step 3: Convert to log-scale
    mag_db = 20 * np.log10(mag + 1e-12)

    # Step 4: Define CFAR kernel
    window_size = 2 * (num_guard + num_train) + 1
    kernel = np.ones((window_size, window_size))
    kernel[num_train:num_train + 2 * num_guard + 1, num_train:num_train + 2 * num_guard + 1] = 0
    num_cells = np.sum(kernel)

    # Step 5: Compute noise level using convolution
    noise_level = convolve2d(mag_db, kernel / num_cells, mode='same', boundary='symm')
    
    # Step 6: Threshold (can be improved with adaptive thresholding)
    threshold = noise_level + 12.0  # dB offset based on false alarm rate

    # Step 7: Detect peaks above threshold
    detections = (mag_db > threshold)

    # Step 8: Apply motion constraints
    range_idxs, doppler_idxs = np.where(detections)

    valid_detections = []
    for r_idx, d_idx in zip(range_idxs, doppler_idxs):
        range_m = r_idx * range_res
        speed_mps = (d_idx - num_doppler // 2) * doppler_res  # Center Doppler

        if 1 < range_m < max_range and abs(speed_mps) < max_speed:
            valid_detections.append((r_idx, d_idx, range_m, speed_mps))

    return valid_detections
    #Return detections as list of (range_idx, doppler_idx, range_m, velocity_mps).

def cfar_2d_numpy(
    rd_map, num_train=8, num_guard=4, range_res=0.5, doppler_res=0.25,
    max_range=100, max_speed=50, method='GO', nms_kernel_size=3,
    estimate_aoa=True, carrier_wavelength=0.0375, rx_spacing=0.05
):
    """
    CFAR (Constant False Alarm Rate) 2D Detector - NumPy Version

    Parameters:
        rd_map : ndarray
            Input range-Doppler map of shape [num_rx, 2, num_doppler, num_range]
            - num_rx: number of receive antennas
            - 2: [real, imag] channels
            - num_doppler: Doppler bins (velocity axis)
            - num_range: Range bins (distance axis)

        num_train : int
            Number of training cells per side (excluding guard and CUT).

        num_guard : int
            Number of guard cells around the Cell Under Test (CUT).

        range_res : float
            Range resolution in meters per bin.

        doppler_res : float
            Doppler resolution in m/s per bin.

        max_range : float
            Maximum detection range in meters (used for filtering).

        max_speed : float
            Maximum absolute speed in m/s (used for filtering).

        method : str
            CFAR type: 'CA' = Cell Averaging, 'GO' = Greatest-Of, 'SO' = Smallest-Of.

        nms_kernel_size : int
            Size of kernel for non-maximum suppression (odd number recommended).

        estimate_aoa : bool
            Whether to estimate AoA (requires at least 2 Rx antennas).

        carrier_wavelength : float
            Radar wavelength in meters (e.g., 0.0375 m for 77 GHz radar).

        rx_spacing : float
            Distance between receive antennas in meters.

    Returns:
        List[dict]: Each detection contains:
            - range_idx, doppler_idx: bin coordinates
            - range_m, velocity_mps: physical units
            - angle_deg: estimated AoA (if available)
    """
    num_rx, _, num_doppler, num_range = rd_map.shape

    # Convert real + imag → complex
    real = rd_map[:, 0]
    imag = rd_map[:, 1]
    complex_map = real + 1j * imag

    # Compute magnitude and convert to dB
    mag = np.abs(complex_map).mean(axis=0)  # Shape: [num_doppler, num_range]
    mag_db = 20 * np.log10(mag + 1e-12)

    # CFAR window kernel setup
    k = num_guard + num_train
    window_size = 2 * k + 1
    full_kernel = np.ones((window_size, window_size), dtype=np.float32)
    guard_area = np.zeros_like(full_kernel)
    guard_area[num_train:num_train + 2*num_guard + 1,
               num_train:num_train + 2*num_guard + 1] = 1
    train_kernel = full_kernel - guard_area  # Mask for training cells

    def compute_thresholds(mag_db):
        if method == 'CA':
            # Cell Averaging CFAR:
            # \[
            # T(x,y) = \frac{1}{N} \sum_{(i,j) \in \text{train}} P(i,j)
            # \]
            noise_est = convolve2d(mag_db, train_kernel / np.sum(train_kernel),
                                   mode='same', boundary='symm')
        elif method in ['GO', 'SO']:
            # Split into horizontal and vertical training regions
            horiz_kernel = train_kernel.copy()
            horiz_kernel[num_train:num_train + 2*num_guard + 1, :] = 0
            vert_kernel = train_kernel.copy()
            vert_kernel[:, num_train:num_train + 2*num_guard + 1] = 0

            noise_h = convolve2d(mag_db, horiz_kernel / np.sum(horiz_kernel),
                                 mode='same', boundary='symm')
            noise_v = convolve2d(mag_db, vert_kernel / np.sum(vert_kernel),
                                 mode='same', boundary='symm')

            noise_est = np.maximum(noise_h, noise_v) if method == 'GO' else np.minimum(noise_h, noise_v)
        else:
            raise ValueError("Invalid CFAR method")

        # Add constant offset (empirical, or derived from desired P_fa)
        return noise_est + 12  # dB

    # Calculate threshold and detect
    threshold = compute_thresholds(mag_db)
    detections = mag_db > threshold

    # Non-maximum suppression
    if nms_kernel_size > 1:
        local_max = maximum_filter(mag_db, size=nms_kernel_size)
        detections &= (mag_db == local_max)

    doppler_idxs, range_idxs = np.where(detections)
    results = []

    for d_idx, r_idx in zip(doppler_idxs, range_idxs):
        range_m = r_idx * range_res
        velocity_mps = (d_idx - num_doppler // 2) * doppler_res

        if not (1 < range_m < max_range and abs(velocity_mps) < max_speed):
            continue

        # Estimate AoA using:
        # \[
        # \theta = \arcsin\left( \frac{\Delta \phi \cdot \lambda}{2\pi d} \right)
        # \]
        angle_deg = None
        if estimate_aoa and num_rx >= 2:
            phase_diff = np.angle(complex_map[1, d_idx, r_idx]) - np.angle(complex_map[0, d_idx, r_idx])
            sin_theta = np.clip(phase_diff * carrier_wavelength / (2 * np.pi * rx_spacing), -1, 1)
            angle_deg = np.degrees(np.arcsin(sin_theta))

        results.append({
            "range_idx": r_idx,
            "doppler_idx": d_idx,
            "range_m": range_m,
            "velocity_mps": velocity_mps,
            "angle_deg": angle_deg
        })

    return results


def cfar_2d_advanced(
    rd_map, num_train=8, num_guard=4, range_res=0.5, doppler_res=0.25,
    max_range=100, max_speed=50, method='GO', pfa=1e-5, nms_kernel_size=3,
    estimate_aoa=False, carrier_wavelength=0.0375, rx_spacing=0.05,
    suppress_zero_doppler_width=0, min_snr_db=6.0
):
    """
    Advanced CFAR:
    - Operates on linear power domain (not dB)
    - Threshold derived from desired false alarm rate (Pfa)
    - GO/SO/CA supported (GO default)
    - Non-maximum suppression and connected-component pruning to one peak per blob
    - Optional suppression of near-zero Doppler band

    Returns List[dict] with keys: range_idx, doppler_idx, range_m, velocity_mps, angle_deg
    """
    num_rx, _, num_doppler, num_range = rd_map.shape

    real = rd_map[:, 0]
    imag = rd_map[:, 1]
    complex_map = real + 1j * imag
    mag = np.abs(complex_map).mean(axis=0)  # [doppler, range]
    power = mag ** 2                         # linear power

    # CFAR window
    k = num_guard + num_train
    window_size = 2 * k + 1
    full_kernel = np.ones((window_size, window_size), dtype=np.float32)
    guard_area = np.zeros_like(full_kernel)
    guard_area[num_train:num_train + 2*num_guard + 1,
               num_train:num_train + 2*num_guard + 1] = 1
    train_kernel = full_kernel - guard_area

    # Helper to compute alpha from Pfa for CA on N training cells
    def alpha_from_pfa(N):
        N = max(1, int(N))
        return N * (pfa ** (-1.0 / N) - 1.0)

    if method == 'CA':
        N_tot = np.sum(train_kernel)
        noise_est = convolve2d(power, train_kernel / N_tot, mode='same', boundary='symm')
        alpha = alpha_from_pfa(N_tot)
        threshold = noise_est * alpha
    elif method in ['GO', 'SO']:
        horiz_kernel = train_kernel.copy()
        horiz_kernel[num_train:num_train + 2*num_guard + 1, :] = 0
        vert_kernel = train_kernel.copy()
        vert_kernel[:, num_train:num_train + 2*num_guard + 1] = 0

        N_h = np.sum(horiz_kernel)
        N_v = np.sum(vert_kernel)
        noise_h = convolve2d(power, horiz_kernel / N_h, mode='same', boundary='symm')
        noise_v = convolve2d(power, vert_kernel / N_v, mode='same', boundary='symm')
        alpha_h = alpha_from_pfa(N_h)
        alpha_v = alpha_from_pfa(N_v)
        thr_h = noise_h * alpha_h
        thr_v = noise_v * alpha_v
        threshold = np.maximum(thr_h, thr_v) if method == 'GO' else np.minimum(thr_h, thr_v)
    else:
        raise ValueError("Invalid CFAR method")

    detections = power > threshold

    # Suppress near-zero Doppler band if requested
    if suppress_zero_doppler_width and suppress_zero_doppler_width > 0:
        center = num_doppler // 2
        bw = int(suppress_zero_doppler_width)
        detections[max(0, center - bw):min(num_doppler, center + bw + 1), :] = False

    # NMS based on dB magnitude
    if nms_kernel_size > 1:
        mag_db = 20 * np.log10(mag + 1e-12)
        local_max = maximum_filter(mag_db, size=nms_kernel_size)
        detections &= (mag_db == local_max)

    # Connected-component labeling: keep single peak per blob
    labeled, num_blobs = label(detections)
    results = []

    if num_blobs > 0:
        for blob_id in range(1, num_blobs + 1):
            ys, xs = np.where(labeled == blob_id)
            if ys.size == 0:
                continue
            blob_powers = power[ys, xs]
            best = np.argmax(blob_powers)
            d_idx = ys[best]
            r_idx = xs[best]

            range_m = r_idx * range_res
            velocity_mps = (d_idx - num_doppler // 2) * doppler_res

            if not (1 < range_m < max_range and abs(velocity_mps) < max_speed):
                continue

            # SNR check against local threshold
            snr_db = 10.0 * np.log10((power[d_idx, r_idx] + 1e-12) / (threshold[d_idx, r_idx] + 1e-12))
            if snr_db < (min_snr_db or 0.0):
                continue

            angle_deg = None
            if estimate_aoa and num_rx >= 2:
                phase_diff = np.angle(complex_map[1, d_idx, r_idx]) - np.angle(complex_map[0, d_idx, r_idx])
                sin_theta = np.clip(phase_diff * carrier_wavelength / (2 * np.pi * rx_spacing), -1, 1)
                angle_deg = np.degrees(np.arcsin(sin_theta))

            results.append({
                "range_idx": r_idx,
                "doppler_idx": d_idx,
                "range_m": range_m,
                "velocity_mps": velocity_mps,
                "angle_deg": angle_deg
            })

    return results


def cfar_2d_pytorch(
    rd_map, num_train=8, num_guard=4, range_res=0.5, doppler_res=0.25,
    max_range=100, max_speed=50, method='GO', nms_kernel_size=3,
    estimate_aoa=True, carrier_wavelength=0.0375, rx_spacing=0.05, device='cuda'
):
    '''
    CFAR (Constant False Alarm Rate) 2D Detector - PyTorch Version

    Parameters:
        rd_map : ndarray
            Input range-Doppler map of shape [num_rx, 2, num_doppler, num_range]
            - num_rx: number of receive antennas
            - 2: real and imaginary channels
            - num_doppler: Doppler bins (velocity axis)
            - num_range: range bins (distance axis)

        num_train : int
            Number of training cells in each dimension (excluding guard and CUT)

        num_guard : int
            Number of guard cells in each dimension around the Cell Under Test (CUT)

        range_res : float
            Range resolution in meters/bin

        doppler_res : float
            Doppler resolution in m/s/bin

        max_range : float
            Maximum detection range in meters

        max_speed : float
            Maximum absolute speed in m/s

        method : str
            CFAR type: 'CA' = Cell Averaging, 'GO' = Greatest-Of, 'SO' = Smallest-Of

        nms_kernel_size : int
            Size of kernel for non-maximum suppression (odd integer)

        estimate_aoa : bool
            Whether to estimate Angle of Arrival (requires at least 2 Rx)

        carrier_wavelength : float
            Wavelength of the radar carrier in meters

        rx_spacing : float
            Distance between Rx antennas in meters

        device : str
            PyTorch device ('cuda' or 'cpu')

    Returns:
        List[dict] with keys:
            - range_idx, doppler_idx: bin indices
            - range_m, velocity_mps: physical values
            - angle_deg: AoA estimate (if available)
    '''
    # Convert to torch tensor and move to device
    rd_map = torch.tensor(rd_map, dtype=torch.float32, device=device)
    num_rx, _, num_doppler, num_range = rd_map.shape

    # Extract real and imaginary parts and convert to complex
    real = rd_map[:, 0]
    imag = rd_map[:, 1]
    complex_map = torch.complex(real, imag)  # shape: [num_rx, num_doppler, num_range]

    # Compute magnitude and convert to dB
    mag = torch.abs(complex_map).mean(dim=0)  # [num_doppler, num_range]
    mag_db = 20 * torch.log10(mag + 1e-12)

    # Create CFAR window masks
    k = num_guard + num_train
    window_size = 2 * k + 1
    full_kernel = torch.ones((window_size, window_size), device=device)
    guard_area = torch.zeros((window_size, window_size), device=device)
    guard_area[num_train:num_train + 2*num_guard + 1, num_train:num_train + 2*num_guard + 1] = 1
    train_kernel = full_kernel - guard_area

    def conv2d_torch(data, kernel):
        # Applies 2D convolution with same padding
        data = data.unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W]
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape [1, 1, kH, kW]
        # Use explicit integer padding for functional conv2d. Some
        # environments do not support padding='same' for F.conv2d.
        pad_h = int(kernel.shape[-2] // 2)
        pad_w = int(kernel.shape[-1] // 2)
        kernel = kernel.to(dtype=data.dtype)
        return torch.nn.functional.conv2d(
            data,
            kernel,
            padding=(pad_h, pad_w)
        )[0, 0]


    def compute_thresholds(mag_db):
        """
        Compute CFAR noise threshold using different methods.

        CA-CFAR:
            \[
            T = \frac{1}{N} \sum_{\text{training cells}} P_{ij}
            \]

        GO-CFAR:
            \[
            T = \max(\text{horizontal}, \text{vertical}) + \Delta
            \]

        SO-CFAR:
            \[
            T = \min(\text{horizontal}, \text{vertical}) + \Delta
            \]
        """
        if method == 'CA':
            noise_est = conv2d_torch(mag_db, train_kernel / train_kernel.sum())
        elif method in ['GO', 'SO']:
            horiz_kernel = train_kernel.clone()
            horiz_kernel[num_train:num_train + 2*num_guard + 1, :] = 0
            vert_kernel = train_kernel.clone()
            vert_kernel[:, num_train:num_train + 2*num_guard + 1] = 0
            noise_h = conv2d_torch(mag_db, horiz_kernel / horiz_kernel.sum())
            noise_v = conv2d_torch(mag_db, vert_kernel / vert_kernel.sum())
            noise_est = torch.max(noise_h, noise_v) if method == 'GO' else torch.min(noise_h, noise_v)
        else:
            raise ValueError("Unknown method")
        return noise_est + 12  # dB threshold offset

    threshold = compute_thresholds(mag_db)
    detections = (mag_db > threshold)

    # Non-maximum suppression
    if nms_kernel_size > 1:
        local_max = torch.nn.functional.max_pool2d(
            mag_db[None, None], kernel_size=nms_kernel_size, stride=1, padding=nms_kernel_size // 2
        )[0, 0]
        detections = detections & (mag_db == local_max)

    doppler_idxs, range_idxs = torch.nonzero(detections, as_tuple=True)
    results = []
    for d_idx, r_idx in zip(doppler_idxs.tolist(), range_idxs.tolist()):
        rng = r_idx * range_res
        vel = (d_idx - num_doppler // 2) * doppler_res
        if rng < 1 or rng > max_range or abs(vel) > max_speed:
            continue
        angle_deg = None
        if estimate_aoa and num_rx >= 2:
            phase_diff = torch.angle(complex_map[1, d_idx, r_idx]) - torch.angle(complex_map[0, d_idx, r_idx])
            # AoA using phase difference:
            # \[
            # \theta = \arcsin\left( \frac{\Delta\phi \cdot \lambda}{2\pi d} \right)
            # \]
            sin_theta = torch.clamp(phase_diff * carrier_wavelength / (2 * torch.pi * rx_spacing), -1, 1)
            angle_deg = torch.arcsin(sin_theta).item() * 180 / np.pi
        results.append({
            "range_idx": r_idx,
            "doppler_idx": d_idx,
            "range_m": rng,
            "velocity_mps": vel,
            "angle_deg": angle_deg
        })
    return results

def generate_synthetic_rd_map(num_rx=2, num_doppler=128, num_range=256, targets=5):
    """
    Generate synthetic range-Doppler map with random injected targets

    Returns:
        rd_map : ndarray of shape [num_rx, 2, num_doppler, num_range]
    """
    rd_map = np.random.normal(0, 1, (num_rx, 2, num_doppler, num_range)) * 0.2  # background noise

    for _ in range(targets):
        d = np.random.randint(num_doppler // 4, 3 * num_doppler // 4)
        r = np.random.randint(num_range // 4, 3 * num_range // 4)
        amplitude = np.random.uniform(5, 15)
        phase = np.random.uniform(0, 2 * np.pi, num_rx)

        for rx in range(num_rx):
            rd_map[rx, 0, d, r] += amplitude * np.cos(phase[rx])  # real
            rd_map[rx, 1, d, r] += amplitude * np.sin(phase[rx])  # imag

    return rd_map

import numpy as np

def generate_clear_synthetic_rd_map(num_rx=2, num_doppler=128, num_range=256, targets=5, noise_power=0.05):
    """
    Generate a clean synthetic range-Doppler map with visible targets and low noise.

    Parameters:
        num_rx : int
            Number of receive antennas.
        num_doppler : int
            Number of Doppler bins (velocity dimension).
        num_range : int
            Number of range bins.
        targets : int
            Number of synthetic targets to inject.
        noise_power : float
            Standard deviation of background noise.

    Returns:
        rd_map : ndarray of shape [num_rx, 2, num_doppler, num_range]
            Complex IQ data with clear targets.
    """
    rd_map = np.random.normal(0, noise_power, (num_rx, 2, num_doppler, num_range))  # low noise

    for _ in range(targets):
        d = np.random.randint(num_doppler // 4, 3 * num_doppler // 4)
        r = np.random.randint(num_range // 4, 3 * num_range // 4)
        amplitude = np.random.uniform(10, 20)  # Stronger target
        width = np.random.randint(3, 7)        # Spread over small area
        phase = np.random.uniform(0, 2 * np.pi, num_rx)

        for rx in range(num_rx):
            for i in range(-width, width + 1):
                for j in range(-width, width + 1):
                    dd = np.clip(d + i, 0, num_doppler - 1)
                    rr = np.clip(r + j, 0, num_range - 1)
                    rd_map[rx, 0, dd, rr] += amplitude * np.cos(phase[rx]) * np.exp(-0.1 * (i**2 + j**2))
                    rd_map[rx, 1, dd, rr] += amplitude * np.sin(phase[rx]) * np.exp(-0.1 * (i**2 + j**2))

    return rd_map

def main():
    # Generate synthetic radar range-Doppler map
    rd_map = generate_clear_synthetic_rd_map(num_rx=2, num_doppler=128, num_range=256, targets=5)
    #(2, 2, 128, 256)
    # Run CFAR detection, cfar_2d_pytorch or cfar_2d_numpy
    detections = cfar_2d_pytorch(
        rd_map,
        num_train=8,
        num_guard=4,
        range_res=0.5,          # meters per range bin
        doppler_res=0.25,       # m/s per Doppler bin
        max_range=100,
        max_speed=50,
        method='GO',
        estimate_aoa=True,
        device='mps'
    )#list of dicts

    # Prepare magnitude for visualization
    complex_map = rd_map[:, 0] + 1j * rd_map[:, 1]
    mag = np.abs(complex_map).mean(axis=0)
    mag_db = 20 * np.log10(mag + 1e-12)

    # Plot range-Doppler map
    plt.figure(figsize=(12, 6))
    plt.imshow(mag_db, aspect='auto', origin='lower', cmap='hot',
               extent=[0, mag_db.shape[1], -mag_db.shape[0]//2, mag_db.shape[0]//2])
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin (centered)')
    plt.title('Synthetic Range-Doppler Map with CFAR Detections')

    # Overlay detections
    for det in detections:
        plt.plot(det["range_idx"], det["doppler_idx"] - mag_db.shape[0] // 2, 'bo')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()