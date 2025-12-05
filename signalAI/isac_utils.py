"""
isac_utils.py

Reusable utilities for ISAC (Integrated Sensing and Communication) experiments.

This file is intended to be relatively stable and "library-like":
- Low-level physics & signal processing utilities
- Dataset simulation and simple visualizations
- Traditional FMCW / OTFS radar and OFDM / OTFS communications baselines
- Deep-learning model definitions and generic training loops

Experiment-specific wiring, hyperparameters, and what to run live in isac_main.py.
"""

import os
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------
# OPTIONAL SCIPY
# ---------------------------------------------------------------------
try:
    import scipy.ndimage as ndi
    SCIPY = True
except ImportError:
    ndi = None
    SCIPY = False
    print("Warning: SciPy not installed. Falling back to NumPy-only ops where needed.")

from isac_vis_utils import (
    visualize_scene_3d_matplotlib,
    export_scene_to_ply_open3d,
    viz_fmcw_extras,
    viz_otfs_extras,
    viz_channel_scatterers,
)

# ---------------------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Device: {DEVICE} ---")

C0 = 299_792_458.0  # speed of light


# ---------------------------------------------------------------------
# SYSTEM PARAMS
# ---------------------------------------------------------------------
@dataclass
class SystemParams:
    """
    Global radar / comm system parameters.

    Attributes
    ----------
    fc : float
        Carrier frequency (Hz).
    B : float
        FMCW modulation bandwidth (Hz).
    fs : float
        ADC sampling rate (Hz), should be >= B.
    M : int
        Number of chirps (slow-time / Doppler dimension).
    N : int
        Samples per chirp (fast-time / range FFT size).
    H : float
        Radar sensor height above ground (m).
    az_fov : float
        Horizontal FOV in degrees.
    el_fov : float
        Vertical FOV in degrees.
    bev_r_max : float
        BEV clamp range for downstream visualization (m).
    """
    fc: float = 77e9
    B: float = 150e6
    fs: float = 150e6
    M: int = 512
    N: int = 512
    H: float = 1.8
    az_fov: float = 60.0
    el_fov: float = 20.0
    bev_r_max: float = 50.0

    @property
    def lambda_m(self):
        return C0 / self.fc

    @property
    def T_chirp(self):
        return self.N / self.fs

    @property
    def slope(self):
        return self.B / self.T_chirp  # FMCW slope S = B/T

    # ---------------- FMCW / OTFS axes ----------------
    def fmcw_axes(self):
        """
        Range–Doppler axes for FMCW.

        Returns
        -------
        ra : np.ndarray
            One-sided range axis (0 .. N/2-1).
        va : np.ndarray
            Doppler velocity axis (fftshifted slow-time).
        """
        ra = (C0 / (2.0 * self.B)) * np.arange(self.N // 2)
        f_d = np.fft.fftshift(np.fft.fftfreq(self.M, d=self.T_chirp))
        va = (self.lambda_m / 2.0) * f_d
        return ra, va
    
    def otfs_axes(self):
        """
        Delay–Doppler axes for the OTFS / pulsed radar.

        - Range resolution: c / (2 fs)
        - Doppler resolution: same PRF as FMCW (1 / T_chirp)
        """
        # full N range bins (unlike FMCW which keeps only 0..N/2-1)
        ra = (C0 / (2.0 * self.fs)) * np.arange(self.N)
        f_d = np.fft.fftshift(np.fft.fftfreq(self.M, d=self.T_chirp))
        va = (self.lambda_m / 2.0) * f_d
        return ra, va
    
    # def otfs_axes(self):
    #     """
    #     Delay–Doppler axes consistent with otfs_torch():
    #       - range bin k:  r_k = (c / (2 fs)) * k
    #       - Doppler bin l: v_l = l_res * (l - M/2)
    #     """
    #     k_res = C0 / (2.0 * self.fs)
    #     l_res = (self.lambda_m / 2.0) * (self.fs / (self.M * self.N))

    #     r = k_res * np.arange(self.N)
    #     v = l_res * (np.arange(self.M) - self.M // 2)
    #     return r, v

    # def otfs_axes(self):
    #     """
    #     Approximate delay–Doppler axes for the OTFS grid.
    #     """
    #     r = np.linspace(0, (C0 / (2 * self.fs)) * self.N, self.N)
    #     v = np.linspace(
    #         -(self.lambda_m / 2) * (self.fs / (self.N * self.M)) * (self.M / 2),
    #         +(self.lambda_m / 2) * (self.fs / (self.N * self.M)) * (self.M / 2),
    #         self.M,
    #     )
    #     return r, v


# ---------------------------------------------------------------------
# BASIC UTILS / METRICS / SCALES
# ---------------------------------------------------------------------
def to_torch(x):
    return torch.tensor(x, device=DEVICE, dtype=torch.float32)


def _safe_f1(tp, fp, fn):
    if tp == 0 and (fp + fn) > 0:
        return 0.0
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r + 1e-9)


def _precision_recall(tp, fp, fn):
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return precision, recall


def _db_scale(rd):
    """
    Convert linear magnitude map to dB for visualization.
    If the map already looks like dB (big dynamic range), return as-is.
    """
    rd = np.asarray(rd)
    if np.nanmax(rd) > 100:
        return rd
    return 20 * np.log10(np.abs(rd) + 1e-9)


# ---------------------------------------------------------------------
# SUBPIXEL PEAK REFINEMENT (for DL & CFAR peaks)
# ---------------------------------------------------------------------
def _subpixel_quadratic(heat: np.ndarray, y: int, x: int):
    """Return (dy, dx) in [-1,1] via 1D quadratic peak fit around (y,x)."""
    H, W = heat.shape
    y0, y1, y2 = max(0, y - 1), y, min(H - 1, y + 1)
    x0, x1, x2 = max(0, x - 1), x, min(W - 1, x + 1)

    def quad_peak(a, b, c):
        denom = (a - 2 * b + c)
        if abs(denom) < 1e-9:
            return 0.0
        t = 0.5 * float(a - c) / float(denom)
        return float(np.clip(t, -1.0, 1.0))

    dy = quad_peak(heat[y0, x1], heat[y1, x1], heat[y2, x1])
    dx = quad_peak(heat[y1, x0], heat[y1, x1], heat[y1, x2])
    return dy, dx


def _rv_from_idx_with_subpix(y, x, dy, dx, ra, va):
    """
    Convert peak index + subpixel offset to (range, vel) via linear interpolation.

    va is vertical axis (rows), ra is horizontal axis (cols).
    """
    y_f = np.clip(y + dy, 0, len(va) - 1)
    x_f = np.clip(x + dx, 0, len(ra) - 1)
    y0, y1 = int(np.floor(y_f)), min(int(np.floor(y_f)) + 1, len(va) - 1)
    x0, x1 = int(np.floor(x_f)), min(int(np.floor(x_f)) + 1, len(ra) - 1)
    wy = y_f - y0
    wx = x_f - x0
    v = (1 - wy) * va[y0] + wy * va[y1]
    r = (1 - wx) * ra[x0] + wx * ra[x1]
    return r, v


# ---------------------------------------------------------------------
# SCENE RAYCASTING (BOXES + GROUND PLANE)
# ---------------------------------------------------------------------
def raycast_torch(
    sp: SystemParams,
    gts,
    num_az: int = 1024,
    num_el: int = 128,
    intensity_cube: float = 255.0,
    intensity_ground: float = 100.0,
    use_ground_reflection: bool = True,
    return_labels: bool = False,
    # LiDAR-like intensity options (all optional; default = OFF for backward compat)
    lidar_like_intensity: bool = False,
    rho_ground: float = 0.3,     # approximate ground reflectivity
    rho_cube: float = 0.7,       # approximate object reflectivity
    lidar_scale: float = 300.0,  # global intensity scale factor
    angle_exp: float = 1.5,      # exponent on cos(theta) falloff
    noise_std: float = 0.02,     # relative additive noise on intensity
):
    """
    Ray-based scene rendering with simple box / ground intersections.

    Scene model
    -----------
    - Sensor (radar/LiDAR) is at position (0, 0, sp.H).
    - Emits a dense set of rays over azimuth/elevation FOV.
    - Intersects:
        * ground plane z = 0   (optional)
        * axis-aligned boxes (cubes) from `gts`.

    Ground-truth objects `gts`:
        Each element is a dict:
          {
            "c": [x, y, z],      # center [m]
            "s": [sx, sy, sz],   # size  [m]
            "v": [vx, vy, vz],   # velocity [m/s]
          }

    Parameters
    ----------
    sp : SystemParams
        Global system configuration (FOV, sensor height, etc.)

    gts : list[dict]
        List of ground-truth boxes (see above format).

    num_az : int, optional
        Number of azimuth samples across sp.az_fov.
        Default = 1024 (backward-compatible with older code).

    num_el : int, optional
        Number of elevation samples across sp.el_fov.
        Default = 128.

    intensity_cube : float, optional
        Legacy per-hit intensity for cube hits when
        `lidar_like_intensity == False`.
        Ignored in LiDAR mode.

    intensity_ground : float, optional
        Legacy per-hit intensity for ground hits when
        `lidar_like_intensity == False`.
        Ignored in LiDAR mode.

    use_ground_reflection : bool, optional
        If False, skip ground-plane intersection, so only cubes
        produce hits. Default True.

    return_labels : bool, optional
        If False (default), returns exactly the original 3-tuple:
            (pts, its, vels)
        If True, returns:
            (pts, its, vels, labels)
        where labels encodes object type:
            0      → ground
            1..G   → index of gt object in `gts`
        (only for rays that actually hit something).

    lidar_like_intensity : bool, optional
        If False (default), use simple fixed intensities:
            ground → intensity_ground
            cubes  → intensity_cube
        If True, use a simple LiDAR-like model:
            I ∝ rho * cos(theta)^angle_exp / R^2 + noise,
        where:
            - rho_ground / rho_cube set relative reflectivities
            - R is range (m)
            - theta is incidence angle between ray and surface normal.

    rho_ground, rho_cube : float, optional
        Approximate reflectivities (0–1) for ground / cubes in
        LiDAR mode.

    lidar_scale : float, optional
        Global intensity scale used in LiDAR mode before clipping to
        [0, 255].

    angle_exp : float, optional
        Exponent applied to cos(theta) in LiDAR mode; larger values
        make grazing angles dimmer.

    noise_std : float, optional
        Relative Gaussian noise in LiDAR mode. Intensity is perturbed
        by ~noise_std * (typical intensity).

    Returns
    -------
    pts : (N, 3) torch.float32
        Hit positions in world coordinates [m].

    its : (N,) torch.float32
        Per-hit intensity values. Either:
          - fixed cube/ground constants (legacy mode), or
          - LiDAR-like scaled values (LiDAR mode),
        all clipped to [0, 255].

    vels : (N, 3) torch.float32
        Velocity vector [m/s] at each hit point.
        - Ground hits → [0,0,0]
        - Cube hits   → gt["v"] of the box that was hit.

    labels : (N,) torch.int64   (only if return_labels=True)
        Integer label per hit:
          - 0     → ground
          - 1..G  → index of gt box in `gts` (1-based)
    """
    # ------------------------------------------------------------------
    # 1. Build the ray directions over azimuth / elevation
    # ------------------------------------------------------------------
    # az: (num_az,) azimuth angles [rad] over [-FOV/2, +FOV/2]
    az = torch.linspace(
        np.deg2rad(-sp.az_fov / 2.0),
        np.deg2rad(sp.az_fov / 2.0),
        num_az,
        device=DEVICE,
    )
    # el: (num_el,) elevation angles [rad] over [-FOV/2, +FOV/2]
    el = torch.linspace(
        np.deg2rad(-sp.el_fov / 2.0),
        np.deg2rad(sp.el_fov / 2.0),
        num_el,
        device=DEVICE,
    )

    # EL, AZ: (num_el, num_az) grids of elevation & azimuth angles
    EL, AZ = torch.meshgrid(el, az, indexing="ij")

    # rays: (num_el * num_az, 3)
    # Each row is a unit direction vector in world coordinates.
    #   x = cos(el)*cos(az)
    #   y = cos(el)*sin(az)
    #   z = sin(el)
    rays = torch.stack(
        [
            torch.cos(EL) * torch.cos(AZ),
            torch.cos(EL) * torch.sin(AZ),
            torch.sin(EL),
        ],
        dim=-1,
    ).reshape(-1, 3)

    # Sensor origin: (3,) [m]
    pos = torch.tensor([0.0, 0.0, sp.H], device=DEVICE)

    # Number of rays
    R = rays.shape[0]

    # ------------------------------------------------------------------
    # 2. Allocate per-ray intersection accumulators
    # ------------------------------------------------------------------
    # t_min: (R,)  distance along ray from sensor to nearest hit [m]
    # Initialize with a large sentinel value (no hit yet).
    t_min = torch.full((R,), 100.0, device=DEVICE)

    # hits_int: (R,)  raw intensity value for each ray (0 = no hit)
    hits_int = torch.zeros((R,), device=DEVICE)

    # hits_vel: (R,3)  velocity at each ray's nearest hit.
    # Initialize to zero (used for ground or no hit).
    hits_vel = torch.zeros((R, 3), device=DEVICE)

    # hits_label: (R,)  label per ray:
    #   -1 = no hit, 0 = ground, 1..G = object index in gts
    hits_label = torch.full((R,), -1, dtype=torch.long, device=DEVICE)

    # ------------------------------------------------------------------
    # 3. Ground-plane intersection (z = 0), optional
    # ------------------------------------------------------------------
    if use_ground_reflection:
        # mask_g: (R,)  which rays are pointing downwards (z < 0)
        mask_g = rays[:, 2] < -2e-2  # avoid near-horizontal rays

        # t_g: (R,)  distance where ray hits z=0 plane:
        #   pos.z + t * rays.z = 0 → t = -pos.z / rays.z
        t_g = -pos[2] / rays[:, 2]

        # mask_valid_g: (R,)  rays that actually hit ground in front
        # of the sensor and closer than any previous hit.
        mask_valid_g = mask_g & (t_g > 0.0) & (t_g < t_min)

        # Update nearest hit distance
        t_min[mask_valid_g] = t_g[mask_valid_g]

        # Legacy intensity for ground hits (if not using LiDAR mode)
        hits_int[mask_valid_g] = float(intensity_ground)

        # Ground has zero velocity
        hits_vel[mask_valid_g] = 0.0

        # Label ground as 0
        hits_label[mask_valid_g] = 0

    # ------------------------------------------------------------------
    # 4. Box (cube) intersections
    # ------------------------------------------------------------------
    if gts:
        # Cs: (G,3) centers; Ss: (G,3) sizes; Vs: (G,3) velocities
        Cs = torch.stack([to_torch(gt["c"]) for gt in gts])  # centers
        Ss = torch.stack([to_torch(gt["s"]) for gt in gts])  # sizes
        Vs = torch.stack([to_torch(gt["v"]) for gt in gts])  # velocities
        G = Cs.shape[0]

        # ro: (1,1,3) ray origins (sensor position, broadcasted)
        ro = pos.view(1, 1, 3)

        # rd: (R,1,3) ray directions, add tiny epsilon to avoid divide-by-zero
        rd = rays.view(R, 1, 3) + 1e-9

        # t1, t2: (R,G,3) param distances where ray hits min/max faces
        # for each axis of the box.
        t1 = (Cs - Ss / 2.0 - ro) / rd
        t2 = (Cs + Ss / 2.0 - ro) / rd

        # tn: (R,G) nearest entering distance among the 3 axes
        tn = torch.max(torch.min(t1, t2), dim=-1)[0]

        # tf: (R,G) farthest exiting distance among the 3 axes
        tf = torch.min(torch.max(t1, t2), dim=-1)[0]

        # mask_hit: (R,G)  True if the ray intersects box g
        #  (tn < tf) ensures there is an interval of overlap
        #  (tn > 0) ensures box is in front of sensor
        mask_hit = (tn < tf) & (tn > 0.0)

        # Invalidate non-hits by setting tn → +inf
        tn[~mask_hit] = np.inf

        # min_t: (R,)  nearest box intersection distance
        # min_idx: (R,) index of box that is closest
        min_t, min_idx = torch.min(tn, dim=1)

        # mask_t: (R,)  rays where a box is closer than current best (ground/other)
        mask_t = min_t < t_min

        # Update nearest distance
        t_min[mask_t] = min_t[mask_t]

        # Legacy: set hit intensity to fixed cube value
        hits_int[mask_t] = float(intensity_cube)

        # Set cube velocity from the chosen box
        hits_vel[mask_t] = Vs[min_idx[mask_t]]

        # Labels: cube index+1 (so 1..G are boxes, 0 remains ground)
        hits_label[mask_t] = (min_idx[mask_t] + 1).long()

    # ------------------------------------------------------------------
    # 5. Filter out rays with no hit at all
    # ------------------------------------------------------------------
    # mask_hit_any: (R,)  rays that hit either ground or a cube
    mask_hit_any = hits_label >= 0

    if not mask_hit_any.any():
        # No hits at all: return empty tensors, matching old types.
        pts_empty = torch.empty((0, 3), device=DEVICE)
        its_empty = torch.empty((0,), device=DEVICE)
        vels_empty = torch.empty((0, 3), device=DEVICE)
        if return_labels:
            labels_empty = torch.empty((0,), dtype=torch.long, device=DEVICE)
            return pts_empty, its_empty, vels_empty, labels_empty
        else:
            return pts_empty, its_empty, vels_empty

    # Effective number of hits:
    #   N = number of rays that actually intersected something
    N = int(mask_hit_any.sum().item())

    # ------------------------------------------------------------------
    # 6. Compute hit positions (for rays that truly hit something)
    # ------------------------------------------------------------------
    # t_hit: (N,) distances to hit for valid rays
    t_hit = t_min[mask_hit_any]

    # rays_hit: (N,3) subset of ray directions that hit
    rays_hit = rays[mask_hit_any]

    # pts_hit: (N,3) world coordinates of hit points:
    #   pos + t * dir
    pts_hit = pos + t_hit.unsqueeze(1) * rays_hit

    # vels_hit: (N,3) velocities at hit points
    vels_hit = hits_vel[mask_hit_any]

    # labels_hit: (N,) labels of hit points
    labels_hit = hits_label[mask_hit_any]

    # intensities (legacy initial values)
    its_hit = hits_int[mask_hit_any]

    # ------------------------------------------------------------------
    # 7. Optional LiDAR-like intensity model
    # ------------------------------------------------------------------
    if lidar_like_intensity:
        # Range R_all: (N,)  physical distance to each hit [m]
        R_all = t_hit

        # Per-hit reflectivity ρ based on label:
        #   label 0 → ground
        #   label 1..G → cubes
        rho = torch.where(
            labels_hit == 0,
            torch.tensor(rho_ground, device=DEVICE),
            torch.tensor(rho_cube, device=DEVICE),
        )

        # normals: (N,3) surface normal at each hit
        normals = torch.zeros_like(pts_hit)

        # Ground hits: normal = +z
        ground_mask = labels_hit == 0
        normals[ground_mask, 2] = 1.0

        # Cube hits: approximate normal as vector from hit point to box center.
        if gts:
            # Rebuild Cs so we can index by box id
            Cs = torch.stack([to_torch(gt["c"]) for gt in gts])  # (G,3)
            cube_mask = labels_hit > 0
            if cube_mask.any():
                # obj_idx: (Nc,) in [0, G-1]
                obj_idx = labels_hit[cube_mask] - 1
                # centers for those hits: (Nc,3)
                c = Cs[obj_idx]
                # vector from hit point to center
                v = c - pts_hit[cube_mask]
                normals[cube_mask] = v / (torch.norm(v, dim=1, keepdim=True) + 1e-6)

        # d_in: (N,3) incoming direction at target (from target to sensor)
        d_in = -rays_hit

        # cos(theta): (N,) angle between incoming ray and normal,
        # clamped to [0,1] so back-facing surfaces give 0.
        cos_theta = torch.clamp(
            torch.sum(d_in * normals, dim=-1),
            min=0.0,
            max=1.0,
        )

        # Simple LiDAR intensity model:
        #   I ∝ lidar_scale * ρ * cos(theta)^angle_exp / R^2
        EPS_R = 1e-3
        I = (
            lidar_scale
            * rho
            * (cos_theta ** angle_exp)
            / (R_all ** 2 + EPS_R)
        )

        # Add relative Gaussian noise
        if noise_std > 0.0:
            I = I + noise_std * I * torch.randn_like(I)

        # Clip to 0..255 (8-bit-ish)
        I = torch.clamp(I, 0.0, 255.0)

        its_hit = I

    # ------------------------------------------------------------------
    # 8. Return, keeping backward compatibility
    # ------------------------------------------------------------------
    if return_labels:
        return pts_hit, its_hit, vels_hit, labels_hit
    else:
        return pts_hit, its_hit, vels_hit

# ---------------------------------------------------------------------
# FMCW / OTFS RADAR SIMULATION
# ---------------------------------------------------------------------
import numpy as np
import torch

# ---------------------------------------------------------------------
# Common helper: convert raycast hits → point scatterers
# ---------------------------------------------------------------------
def _prepare_scatters_from_raycast(
    pts,
    its,
    vels,
    sp,
    min_range: float = 0.1,
    base_amp: float = 1e3,
):
    """
    Convert raycast_torch outputs into radial ranges, radial velocities,
    and complex amplitudes for point scatterers.

    Parameters
    ----------
    pts : (P, 3) torch.Tensor or array
        Hit positions in world coordinates [m].

    its : (P,) torch.Tensor or array
        Intensity / reflectivity per hit. Typically:
          - large for strong targets (e.g., cars),
          - smaller for ground or weak clutter.
        We do NOT assume any particular scaling, only that higher means
        stronger reflection.

    vels : (P, 3) torch.Tensor or array
        Velocity vector [m/s] at each hit.

    sp : SystemParams
        Only sp.H and sp.lambda_m are used here.

    min_range : float, default 0.1
        Minimum allowed range [m] to avoid self-intersection / numerical
        instabilities very close to the radar phase center.

    base_amp : float, default 1e3
        Global scale factor for amplitudes. This lets you tune the overall
        SNR without changing your CFAR thresholds everywhere.

    Returns
    -------
    scat : dict or None
        If valid scatterers exist:
          {
            "P_rel": (P', 3) torch.Tensor   # positions minus radar center
            "R"    : (P',) torch.Tensor    # ranges [m]
            "vr"   : (P',) torch.Tensor    # LOS radial velocities [m/s]
            "amp"  : (P',) torch.Tensor    # complex amplitude magnitudes
          }
        If no valid scatterer remains after masking, returns None.
    """
    # Ensure tensors on correct device
    if not isinstance(pts, torch.Tensor):
        pts = torch.tensor(pts, dtype=torch.float32, device=DEVICE)
    if not isinstance(its, torch.Tensor):
        its = torch.tensor(its, dtype=torch.float32, device=DEVICE)
    if not isinstance(vels, torch.Tensor):
        vels = torch.tensor(vels, dtype=torch.float32, device=DEVICE)

    # Radar phase center (0, 0, H)
    radar_center = torch.tensor(
        [0.0, 0.0, sp.H], dtype=torch.float32, device=DEVICE
    )

    # Positions relative to radar phase center (line-of-sight vector)
    P_rel = pts - radar_center            # (P, 3)

    # Euclidean range [m]
    R = torch.norm(P_rel, dim=1)          # (P,)

    # Mask out extremely close points
    mask = R > min_range
    P_rel = P_rel[mask]
    R = R[mask]
    vels = vels[mask]
    its = its[mask]

    if P_rel.numel() == 0:
        return None

    # Line-of-sight unit vector u_hat
    u_hat = P_rel / R.unsqueeze(1)        # (P', 3)

    # Radial velocity v_r = v · u_hat  [m/s]
    vr = torch.sum(vels * u_hat, dim=1)   # (P',)

    # Amplitude model:
    #   amp ≈ base_amp * (normalized intensity) / R^2
    # intensity normalization: map to [0, 1] using a soft scaling
    # instead of hard 255 assumption to keep it general.
    its_norm = its / (its.max() + 1e-6)
    amp = base_amp * its_norm / (R ** 2 + 1e-6)  # (P',)

    return dict(P_rel=P_rel, R=R, vr=vr, amp=amp)


# ---------------------------------------------------------------------
# Common helper: optional moving target indication (MTI)
# ---------------------------------------------------------------------
def _apply_mti_highpass(iq: torch.Tensor, order: int = 1):
    """
    Apply a simple slow-time high-pass (MTI) filter along the pulse dimension.

    Parameters
    ----------
    iq : (M, N) complex64 torch.Tensor
        Time-domain IQ cube, with M pulses × N fast-time samples.

    order : int, default 1
        - 0 : no MTI (return iq unchanged)
        - 1 : 1st-order difference: y[m] = x[m] - x[m-1]
        - 2 : 2nd-order: y[m] = x[m] - 2x[m-1] + x[m-2]
        >2 : clamped to 2 (for simplicity).

    Returns
    -------
    iq_out : (M, N) complex64 torch.Tensor
        High-pass filtered IQ.
    """
    if order <= 0:
        return iq

    order = int(order)
    order = min(order, 2)  # clamp

    M = iq.shape[0]
    iq_out = iq.clone()

    if order == 1:
        # y[m] = x[m] - x[m-1]
        iq_out[1:] = iq[1:] - iq[:-1]
        iq_out[0] = 0.0
    elif order == 2:
        # y[m] = x[m] - 2x[m-1] + x[m-2]
        iq_out[2:] = iq[2:] - 2.0 * iq[1:-1] + iq[:-2]
        iq_out[:2] = 0.0

    return iq_out


# ---------------------------------------------------------------------
# Common helper: apply 2-D Hann window
# ---------------------------------------------------------------------
def _apply_hann_2d(iq: torch.Tensor, sp: SystemParams):
    """
    Apply separable Hann windows in range and Doppler to reduce sidelobes.

    Parameters
    ----------
    iq : (M, N) complex64 torch.Tensor
        Time-domain IQ cube before FFT.

    sp : SystemParams
        Uses sp.M and sp.N.

    Returns
    -------
    iq_win : (M, N) complex64 torch.Tensor
        Windowed IQ.
    """
    M, N = sp.M, sp.N
    w_r = torch.hann_window(N, device=iq.device)  # range / fast-time
    w_d = torch.hann_window(M, device=iq.device)  # Doppler / slow-time
    return iq * (w_d[:, None] * w_r[None, :])


# ---------------------------------------------------------------------
# Common helper: build FMCW RD ground-truth grid from scatterers
# ---------------------------------------------------------------------
def _make_rd_groundtruth(R, vr, amp, sp: SystemParams):
    """
    Build a sparse complex RD grid representing "ideal" point responses.

    Parameters
    ----------
    R : (P,) torch.Tensor
        Target ranges [m].

    vr : (P,) torch.Tensor
        Target radial velocities [m/s].

    amp : (P,) torch.Tensor
        Target complex amplitude magnitudes (we keep them real & positive).

    sp : SystemParams
        Uses fmcw_axes() to define the RD grid.

    Returns
    -------
    H_gt : (M, N//2) np.ndarray, complex64
        Sparse RD grid with impulses at the nearest (range, velocity) bins.
    """
    ra, va = sp.fmcw_axes()  # range axis (N//2,), velocity axis (M,)

    H_gt = np.zeros((sp.M, sp.N // 2), dtype=np.complex64)

    R_np = R.detach().cpu().numpy()
    vr_np = vr.detach().cpu().numpy()
    amp_np = amp.detach().cpu().numpy()

    for rp, vp, ap in zip(R_np, vr_np, amp_np):
        # nearest range bin
        ix = np.searchsorted(ra, rp)
        ix = np.clip(ix, 0, ra.size - 1)
        # nearest Doppler bin
        iy = np.searchsorted(va, vp)
        iy = np.clip(iy, 0, va.size - 1)
        H_gt[iy, ix] += ap + 0j

    return H_gt


# ---------------------------------------------------------------------
# Common helper: build OTFS DD ground-truth grid from scatterers
# ---------------------------------------------------------------------
def _make_dd_groundtruth(R, vr, amp, sp: SystemParams):
    """
    Build a sparse complex Delay–Doppler ground-truth grid.

    Parameters
    ----------
    R : (P,) torch.Tensor
        Target ranges [m].

    vr : (P,) torch.Tensor
        Target radial velocities [m/s].

    amp : (P,) torch.Tensor
        Target complex amplitude magnitudes.

    sp : SystemParams
        Uses fs and T_chirp to define delay and Doppler grids.

    Returns
    -------
    H_gt : (M, N) np.ndarray, complex64
        Sparse DD grid with impulses at nearest (delay, Doppler) bins.
    delays : (N,) np.ndarray
        Delay axis in seconds.
    f_dop : (M,) np.ndarray
        Doppler axis in Hz (fftshifted).
    """
    M, N = sp.M, sp.N
    # Delay axis (seconds) per fast-time sample
    delays = np.arange(N, dtype=np.float32) / sp.fs
    # Doppler axis (Hz), centered by fftshift
    f_dop = np.fft.fftshift(np.fft.fftfreq(M, d=sp.T_chirp))

    H_gt = np.zeros((M, N), dtype=np.complex64)

    R_np = R.detach().cpu().numpy()
    vr_np = vr.detach().cpu().numpy()
    amp_np = amp.detach().cpu().numpy()

    for rp, vp, ap in zip(R_np, vr_np, amp_np):
        # ideal delay
        tau_p = 2.0 * rp / C0               # seconds
        ix = int(np.round(tau_p * sp.fs))   # nearest delay bin
        ix = np.clip(ix, 0, N - 1)

        # Doppler frequency
        fd_p = 2.0 * vp / sp.lambda_m
        iy = int(np.argmin(np.abs(f_dop - fd_p)))  # nearest Doppler bin

        H_gt[iy, ix] += ap + 0j

    return H_gt, delays, f_dop

# ---------------------------------------------------------------------
# FMCW: transmitter + channel + receiver → baseband IQ
# ---------------------------------------------------------------------
def _fmcw_baseband_iq(
    scat,
    sp: SystemParams,
    noise_std: float = 1e-3,
    mti_order: int = 1,
):
    """
    FMCW baseband (beat signal) IQ generation.

    Conceptually split into:
      - "Transmitter": implicit chirp structure via beat frequencies.
      - "Channel"   : superposition of scatterer returns.
      - "Receiver"  : de-chirped baseband IQ (beat signal).

    Parameters
    ----------
    scat : dict
        Output of _prepare_scatters_from_raycast, with keys:
          - "R"   : (P,) ranges [m]
          - "vr"  : (P,) radial velocities [m/s]
          - "amp" : (P,) amplitudes (real, positive)
        We only need these here.

    sp : SystemParams
        Uses M, N, fs, T_chirp, slope, lambda_m.

    noise_std : float, default 1e-3
        Standard deviation of complex AWGN added to IQ.

    mti_order : int, default 1
        Moving target indication order passed to _apply_mti_highpass.

    Returns
    -------
    iq : (M, N) complex64 torch.Tensor
        FMCW baseband IQ cube (after channel + AWGN + MTI).
    """
    M, N = sp.M, sp.N #512, 512

    R = scat["R"]      # (P,) [3647]
    vr = scat["vr"]    # (P,) [3647]
    amp = scat["amp"]  # (P,) [3647] (most are 0)

    # Fast-time and slow-time sampling
    t_fast = torch.arange(N, device=DEVICE, dtype=torch.float32) / sp.fs
    t_slow = torch.arange(M, device=DEVICE, dtype=torch.float32) * sp.T_chirp

    # Beat range frequency f_r ≈ 2*S*R/c
    k_r = 2.0 * sp.slope / C0
    f_r = k_r * R              # (P,)

    # Doppler frequency f_d ≈ 2*v_r/λ
    f_d = 2.0 * vr / sp.lambda_m   # (P,)

    # Allocate IQ cube (M pulses × N samples)
    iq = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)

    # Build phase for each scatterer in a batched fashion
    P_count = R.shape[0]
    BATCH = 4096
    for i0 in range(0, P_count, BATCH):
        i1 = min(i0 + BATCH, P_count)
        fb_r = f_r[i0:i1]  # (B,)
        fb_d = f_d[i0:i1]  # (B,)
        ab = amp[i0:i1]    # (B,)

        # phase_p(m, n) = 2π ( f_r,p t_fast[n] + f_d,p t_slow[m] )
        phase = 2j * np.pi * (
            fb_r[:, None, None] * t_fast[None, None, :] +
            fb_d[:, None, None] * t_slow[None, :, None]
        )  # (B, M, N)

        iq += torch.sum(ab[:, None, None] * torch.exp(phase), dim=0)

    # Add complex AWGN
    iq = iq + (torch.randn(M, N, device=DEVICE) +
               1j * torch.randn(M, N, device=DEVICE)) * noise_std

    # Optional moving target indication
    iq = _apply_mti_highpass(iq, order=mti_order)

    return iq #[512, 512]


# ---------------------------------------------------------------------
# FMCW: signal processing → Range–Doppler map
# ---------------------------------------------------------------------
def _fmcw_rd_from_iq(iq: torch.Tensor, sp: SystemParams):
    """
    Convert FMCW baseband IQ cube into a Range–Doppler map via 2-D FFT.

    Steps
    -----
    1) Apply 2-D Hann window.
    2) FFT along fast-time (range) and keep positive ranges (N/2 bins).
    3) FFT along slow-time (Doppler) with fftshift to center zero Doppler.

    Parameters
    ----------
    iq : (M, N) complex64 torch.Tensor
        Time-domain baseband IQ.

    sp : SystemParams
        Uses M, N and fmcw_axes().

    Returns
    -------
    rd_db : (M, N//2) np.ndarray, float32
        RD magnitude in dB.

    rd_complex : (M, N//2) np.ndarray, complex64
        Complex RD map (useful for advanced processing).
    """
    M, N = sp.M, sp.N

    # Window
    iq_win = _apply_hann_2d(iq, sp)

    # Range FFT
    RFFT = torch.fft.fft(iq_win, dim=1)         # (M, N)
    RFFT = RFFT[:, : N // 2]                    # positive ranges

    # Doppler FFT + shift
    RD = torch.fft.fftshift(torch.fft.fft(RFFT, dim=0), dim=0)  # (M, N//2)

    RD_mag = torch.abs(RD).clamp_min(1e-12)
    rd_db = (20.0 * torch.log10(RD_mag)).cpu().numpy().astype(np.float32)

    rd_complex = RD.detach().cpu().numpy().astype(np.complex64)
    return rd_db, rd_complex #(512, 256)


# ---------------------------------------------------------------------
# Public wrapper: FMCW radar simulation
# ---------------------------------------------------------------------
def fmcw_torch(
    pts,
    its,
    vels,
    sp: SystemParams,
    labels=None,
    gts=None,
    viz_dir=None,
    return_extra: bool = False,
    mti_order: int = 1,
    noise_std: float = 1e-3,
):
    """
    High-level FMCW Range–Doppler simulation from raycast hits.

    Internally uses the modular components:

      1) _prepare_scatters_from_raycast  (geometry + amplitudes)
      2) _fmcw_baseband_iq               (TX/Channel/RX baseband)
      3) _fmcw_rd_from_iq                (2D FFT → RD map)
      4) _make_rd_groundtruth            (sparse RD ground-truth grid)

    Parameters
    ----------
    pts, its, vels
        Outputs from raycast_torch (see its documentation).

    sp : SystemParams
        Global radar configuration.

    labels : optional
        Object labels from raycast_torch (0 ground, 1..N objects).
        Currently not used inside this function, but kept for future
        per-class amplitude modeling.

    gts : list[dict], optional
        Ground truth boxes (center, size, velocity). Not used directly
        here, but can be stored in the extra dict for later plotting.

    viz_dir : str or Path, optional
        If provided, you can hook in quick visualizations here (RD maps,
        time-domain plots, etc.). For brevity we omit the full plotting
        code in this snippet, but the hook is kept.

    return_extra : bool, default False
        If True, returns (rd_db, extra) where extra includes:
          - "iq"      : time-domain IQ cube (M, N) complex64 numpy
          - "RD"      : complex RD map (M, N//2) complex64 numpy
          - "H_gt"    : ground-truth RD grid (M, N//2) complex64 numpy
          - "ranges"  : range axis (N//2,)
          - "vels"    : Doppler axis (M,)
          - "R", "vr", "amp"  : torch Tensors for each target
          - "gts"     : the original GT list (if provided)

    mti_order : int, default 1
        Moving target indication order passed to _apply_mti_highpass.

    noise_std : float, default 1e-3
        Complex AWGN standard deviation used in _fmcw_baseband_iq.

    Returns
    -------
    out : tuple
        Backward-compatible behavior:
          - If return_extra=False: (rd_db,) where rd_db has shape (M, N//2).
          - If return_extra=True : rd_db, extra
    """
    # --- Handle empty scene -------------------------------------------
    if pts is None or len(pts) == 0:
        rd_empty = np.zeros((sp.M, sp.N // 2), dtype=np.float32)
        if return_extra:
            ra, va = sp.fmcw_axes()
            extra = dict(
                iq=np.zeros((sp.M, sp.N), dtype=np.complex64),
                RD=np.zeros((sp.M, sp.N // 2), dtype=np.complex64),
                H_gt=np.zeros((sp.M, sp.N // 2), dtype=np.complex64),
                ranges=ra,
                vels=va,
                R=torch.empty(0, device=DEVICE),
                vr=torch.empty(0, device=DEVICE),
                amp=torch.empty(0, device=DEVICE),
                gts=gts,
            )
            return rd_empty, extra
        return (rd_empty,)

    # 1) Prepare scatterers from raycast hits
    scat = _prepare_scatters_from_raycast(pts, its, vels, sp)
    if scat is None:
        rd_empty = np.zeros((sp.M, sp.N // 2), dtype=np.float32)
        if return_extra:
            ra, va = sp.fmcw_axes()
            extra = dict(
                iq=np.zeros((sp.M, sp.N), dtype=np.complex64),
                RD=np.zeros((sp.M, sp.N // 2), dtype=np.complex64),
                H_gt=np.zeros((sp.M, sp.N // 2), dtype=np.complex64),
                ranges=ra,
                vels=va,
                R=torch.empty(0, device=DEVICE),
                vr=torch.empty(0, device=DEVICE),
                amp=torch.empty(0, device=DEVICE),
                gts=gts,
            )
            return rd_empty, extra
        return (rd_empty,)

    # 2) Generate baseband IQ (TX+channel+RX)
    iq = _fmcw_baseband_iq(scat, sp, noise_std=noise_std, mti_order=mti_order) #[512, 512]

    # 3) Signal processing: RD map
    rd_db, RD_complex = _fmcw_rd_from_iq(iq, sp) #(512, 256)

    # 4) RD ground-truth grid
    H_gt = _make_rd_groundtruth(scat["R"], scat["vr"], scat["amp"], sp) #(512, 256)
    ra, va = sp.fmcw_axes() #(256,) (512,)

    if not return_extra:
        return (rd_db,)

    extra = dict(
        iq=iq.detach().cpu().numpy().astype(np.complex64), #baseband IQ [512, 512]
        RD=RD_complex, #(512, 256)
        H_gt=H_gt, #(512, 256)
        ranges=ra, #(256,)
        vels=va, #(512,)
        R=scat["R"], #[3647]
        vr=scat["vr"],
        amp=scat["amp"],
        gts=gts,
    )
    return rd_db, extra

# ---------------------------------------------------------------------
# OTFS-like / pulsed radar: baseband IQ generation
# ---------------------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F

def _otfs_params_from_sp(sp):
    """
    Extract OTFS-related parameters from SystemParams.

    We interpret:
      - M_otfs := sp.M         (number of OFDM symbols in a frame)
      - N_sc   := sp.N         (number of subcarriers per OFDM symbol)
      - fs     := sp.fs        (baseband sampling rate, Hz)

    We also choose a subcarrier spacing Δf and CP length. In many OTFS
    implementations:
        T_sym = 1 / Δf
    must be an integer multiple of 1/fs so that each OFDM symbol has
    N_time = T_sym * fs samples.

    For simplicity we set:
        N_time = N_sc
        T_sym  = N_time / fs
        Δf     = 1 / T_sym = fs / N_sc

    So each OFDM symbol uses N_sc time samples and N_sc subcarriers,
    which keeps the numerology square and easy to debug.

    If your physical system uses a different numerology, you can change
    this function only, and the rest of the OTFS code will still work.
    """
    M_otfs = int(sp.M)
    N_sc   = int(sp.N)
    fs     = float(sp.fs)
    Ts     = 1.0 / fs

    N_time = N_sc                     # time samples per OFDM symbol
    T_sym  = N_time * Ts              # OFDM symbol duration (no CP)
    delta_f = 1.0 / T_sym             # subcarrier spacing (Hz)

    # CP length: a few percent of symbol length (configurable)
    cp_len = getattr(sp, "cp_len", max(16, N_time // 16))

    params = dict(
        M=M_otfs,
        N=N_sc,
        fs=fs,
        Ts=Ts,
        N_time=N_time,
        T_sym=T_sym,
        delta_f=delta_f,
        cp_len=cp_len,
    )
    return params

C0 = 299_792_458.0  # speed of light
def _build_otfs_channel_from_gts(gts, sp: SystemParams, otfs_par):
    """
    Build a sparse OTFS delay–Doppler channel from ground-truth boxes.

    This is used by `otfs_torch_full` to create the physical
    delay–Doppler response H(τ, ν) before OTFS modulation.

    Parameters
    ----------
    gts : list of dict
        Ground-truth objects. Each dict has keys:
            'c' : [x, y, z]  center [m]
            's' : [sx, sy, sz] size  [m]  (not used here)
            'v' : [vx, vy, vz] velocity [m/s]
    sp : SystemParams
        Global radar / carrier config (c, λ, etc.).
    otfs_par : dict
        OTFS parameters used by `otfs_torch_full`, must contain:
            'M'      : number of Doppler bins
            'N'      : number of delay bins
            'T_sym'  : OTFS symbol period [s]
            'Ts'     : base sampling period [s] for discrete delay
                       (or any delay resolution you want).

    Returns
    -------
    taps : list of dict
        Each tap is a dict with at least:

          {
            "n_delay": int      # discrete delay bin index in [0, N-1]
            "m_dopp" : int      # discrete Doppler bin index (centered around 0)
            "alpha"  : torch.complex64  # complex channel gain
            "tau"    : float    # continuous delay [s]
            "nu"     : float    # continuous Doppler shift [Hz]
            "f_d"    : float    # alias for Doppler [Hz] (for legacy code)
          }

        This format matches what `_apply_delay_doppler_channel_time`
        expects, so lines like `tap["alpha"].to(device)` and
        `f_d = tap["f_d"]` work.

    Notes
    -----
    * All delay / Doppler index math is done in NumPy / Python floats
      to avoid calling `round()` on torch.Tensors.
    * Range R and radial velocity v_r are computed from geometry:
         R   = || c - radar_pos ||
         v_r = ((c - radar_pos)/R) · v
    * Delay / Doppler grid mapping:
         tau_res      = Ts
         doppler_res  = 1 / (M * T_sym)
         n_delay      = round( tau / tau_res )
         m_dopp       = round( nu / doppler_res )   # centered around 0
    """
    import numpy as np
    import torch

    M = int(otfs_par["M"])
    N = int(otfs_par["N"])
    T_sym = float(otfs_par["T_sym"])
    Ts = float(otfs_par["Ts"])

    # Doppler grid resolution (Hz per bin)
    doppler_res = 1.0 / (M * T_sym)

    # Radar phase center (same convention as raycast_torch)
    radar_pos = np.array([0.0, 0.0, float(sp.H)], dtype=np.float64)

    taps = []

    for gt in gts:
        # GT center and velocity as float64 arrays
        c = np.asarray(gt["c"], dtype=np.float64)
        v = np.asarray(gt["v"], dtype=np.float64)

        # Line-of-sight geometry
        d = c - radar_pos                      # vector from radar to target
        R = np.linalg.norm(d)                  # range [m]
        if R < 0.1:
            # ignore degenerate very-close points
            continue

        u = d / R                              # unit LOS vector
        vr = float(np.dot(u, v))               # radial velocity [m/s]

        # Continuous delay & Doppler
        tau = float(2.0 * R / C0)              # two-way delay [s]
        nu  = float(2.0 * vr / sp.lambda_m)    # Doppler [Hz]

        # --------- Discrete delay index (0 .. N-1) ----------
        tau_over_Ts = tau / Ts                 # pure Python float
        n_delay = int(np.round(tau_over_Ts))
        n_delay = int(np.clip(n_delay, 0, N - 1))

        # --------- Discrete Doppler index (centered) --------
        # m_dopp ~ nu / doppler_res, centered around 0.
        m_dopp = int(np.round(nu / doppler_res))

        # --------- Complex gain model -----------------------
        # Simple 1/R^2 amplitude model; random phase could be added.
        amp = 1.0 / (R ** 2 + 1e-6)
        gain_complex = amp * np.exp(1j * 0.0)

        # Make alpha a torch.complex64 tensor so .to(device) works later.
        alpha = torch.tensor(gain_complex, dtype=torch.complex64)

        tap = {
            "n_delay": n_delay,   # integer delay bin index
            "m_dopp": m_dopp,     # integer Doppler bin index (centered)
            "alpha": alpha,       # complex gain (torch tensor)
            "tau": tau,           # continuous delay [s] (for debugging)
            "nu": nu,             # continuous Doppler [Hz]
            "f_d": nu,            # alias for Doppler [Hz] for legacy code
        }
        taps.append(tap)

    return taps

def build_otfs_H_gt_from_taps(taps, M, N, amp_min=0.0):
    """
    Build an analytic OTFS delay–Doppler ground-truth grid H_gt
    from a list of channel taps.

    Parameters
    ----------
    taps : list[dict]
        Each tap dict should contain:
            'n_delay' : int   # delay index  (0..N-1)
            'm_dopp'  : int   # Doppler index (0..M-1)
            'alpha'   : scalar or 0-dim torch.Tensor (complex gain)
        Additional keys (tau, nu, f_d, ...) are ignored here.

    M, N : int
        Size of the DD grid (M Doppler bins, N delay bins).

    amp_min : float
        Optional minimum |alpha| to keep a tap. Set very small
        (e.g. 1e-6) to keep almost all taps.

    Returns
    -------
    H_gt : np.ndarray, shape (M, N), complex64
        Sparse ground-truth DD grid with impulses at tap locations.

    Physical meaning
    ----------------
    In an ideal single-pilot OTFS radar frame, a tap with
    (n_delay = ℓ, m_dopp = k, alpha = h_{k,ℓ}) produces an impulse
    in H(k, ℓ). This function paints those impulses directly onto
    the DD grid, without sidelobes or noise.
    """
    import numpy as np
    import torch

    H_gt = np.zeros((M, N), dtype=np.complex64)

    if taps is None:
        return H_gt

    for tap in taps:
        if tap is None:
            continue

        n_delay = int(tap["n_delay"])
        m_dopp  = int(tap["m_dopp"])
        alpha   = tap["alpha"]

        # Convert alpha → numpy complex scalar
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().cpu().numpy()
            # If it's still an array, take scalar
            alpha = np.complex64(alpha.astype(np.complex64))
        else:
            alpha = np.complex64(alpha)

        if np.abs(alpha) < amp_min:
            continue

        if 0 <= m_dopp < M and 0 <= n_delay < N:
            H_gt[m_dopp, n_delay] += alpha  # (row = Doppler, col = delay)

    return H_gt

def _otfs_isfft_dd_to_tf(X_dd: torch.Tensor) -> torch.Tensor:
    """
    Discrete inverse symplectic finite Fourier transform (ISFFT):

    DD domain  X_dd[l, k]  (delay l, Doppler k)
      -> TF domain X_tf[n, m] (time index n, subcarrier index m)

    We implement one common convention:

        X_tf = F_M^H  X_dd  F_N

    where F_M, F_N are unitary DFT matrices (1/sqrt(M)) * FFT.
    """
    M, N = X_dd.shape
    device = X_dd.device

    # Unitary DFT matrices (you can cache these for speed)
    F_M = torch.fft.fft(torch.eye(M, device=device)) / np.sqrt(M)
    F_N = torch.fft.fft(torch.eye(N, device=device)) / np.sqrt(N)

    # ISFFT: X_tf = F_M^H X_dd F_N
    X_tf = F_M.conj().T @ X_dd @ F_N
    return X_tf


def _otfs_sfft_tf_to_dd(Y_tf: torch.Tensor) -> torch.Tensor:
    """
    Discrete symplectic finite Fourier transform (SFFT):

    TF domain Y_tf[n, m] -> DD domain Y_dd[l, k].

    Inverse of _otfs_isfft_dd_to_tf, with the same convention:

        Y_dd = F_M  Y_tf  F_N^H
    """
    M, N = Y_tf.shape
    device = Y_tf.device

    F_M = torch.fft.fft(torch.eye(M, device=device)) / np.sqrt(M)
    F_N = torch.fft.fft(torch.eye(N, device=device)) / np.sqrt(N)

    Y_dd = F_M @ Y_tf @ F_N.conj().T
    return Y_dd


def _otfs_tx_waveform_from_dd(X_dd: torch.Tensor, otfs_par: dict):
    """
    Complete OTFS transmitter:

      DD grid X_dd[l,k] --ISFFT--> TF grid X_tf[n,m]
                        --OFDM--> time-domain s[n]

    Steps
    -----
    1) ISFFT: X_tf = ISFFT{X_dd}
    2) For each OFDM symbol n (0..M-1):
         - Take X_tf[n,:] as subcarrier symbols.
         - IFFT across subcarriers to get time samples.
         - Add CP of length cp_len.
    3) Concatenate all OFDM symbols into one 1-D waveform s[n].

    Parameters
    ----------
    X_dd : (M, N) complex64 tensor
        OTFS DD QAM grid.
    otfs_par : dict
        OTFS parameters from _otfs_params_from_sp.

    Returns
    -------
    s : (L,) complex64 tensor
        Time-domain transmit waveform.
    X_tf : (M, N) complex64 tensor
        TF grid after ISFFT (for debugging).
    """
    M = otfs_par["M"]
    N_sc = otfs_par["N"]
    cp_len = otfs_par["cp_len"]

    assert X_dd.shape == (M, N_sc)

    device = X_dd.device

    # 1) DD -> TF via ISFFT
    X_tf = _otfs_isfft_dd_to_tf(X_dd)  # (M, N_sc)

    # 2) OFDM modulation per time index (row)
    # IFFT across subcarriers (dim=1)
    # Note: torch.fft.ifft is unnormalized, so we divide by sqrt(N_sc)
    ofdm_time = torch.fft.ifft(X_tf, dim=1) * np.sqrt(N_sc)  # (M, N_sc)

    # 3) CP + concatenation
    L_sym = N_sc + cp_len
    L_tot = M * L_sym
    s = torch.zeros(L_tot, dtype=torch.complex64, device=device)

    for n in range(M):
        start = n * L_sym
        sym = ofdm_time[n]                     # (N_sc,)
        cp  = sym[-cp_len:]                    # (cp_len,)
        s[start : start + cp_len] = cp
        s[start + cp_len : start + L_sym] = sym

    return s, X_tf


def _apply_delay_doppler_channel_time(
    s_tx: torch.Tensor,
    taps,
    otfs_par: dict,
    device: torch.device | None = None,
    noise_std: float = 0.0,
    mti_order: int = 0,
):
    """
    Apply a discrete delay–Doppler channel to a *time-domain* OTFS/OFDM signal.

    Parameters
    ----------
    s_tx : torch.Tensor, shape (T,)
        Complex baseband transmit signal in time domain. This is typically
        the result of OTFS modulation followed by OFDM + CP insertion:
            T = M * (Nfft + cp_len).

    taps : list[dict]
        Each tap describes a single scatterer / path. Expected keys:
          - "alpha"   : complex scalar amplitude (path gain).
          - "n_delay" : integer sample delay (in units of Ts).
          - "f_d"     : Doppler frequency in Hz for this path.

        Example:
            {
                "alpha": 0.5 + 0.2j,
                "n_delay": 37,
                "f_d": 45.0,   # Hz
            }

    otfs_par : dict
        OTFS / OFDM structural parameters, e.g.:
            {
                "M": M,           # Doppler bins (OFDM symbols)
                "N": N,           # delay bins  (subcarriers)
                "T_sym": T_sym,   # OFDM / OTFS symbol duration [s]
                "Ts": Ts,         # base sampling interval [s]
                "Nfft": Nfft,     # FFT length used in OFDM
                "cp_len": cp_len  # CP length in samples
            }

    device : torch.device or None, default None
        Device to run on. If None, we infer from s_tx.device.

    noise_std : float, default 0.0
        Standard deviation of complex AWGN to add at the end of the channel.

    mti_order : int, default 0
        Optional MTI (moving–target indication) order along slow-time.
        Implemented after reshaping the 1-D signal into (M, samples_per_sym),
        then flattened back to 1-D before returning.

    Returns
    -------
    r_rx_time : torch.Tensor, shape (T,)
        1-D complex baseband received signal in time domain. This matches the
        original s_tx layout, so functions like `_otfs_demod_frame_time`
        (which expect a 1-D vector) can be used directly.
    """
    # ---------------- basic shapes / axes ----------------
    if device is None:
        device = s_tx.device

    s = s_tx.to(device)                         # (T,)
    Ts = float(otfs_par["Ts"])
    M = int(otfs_par["M"])
    Nfft = int(otfs_par.get("Nfft", otfs_par["N"]))
    cp_len = int(otfs_par.get("cp_len", 0))

    samples_per_sym = Nfft + cp_len
    T_total = M * samples_per_sym
    assert s.numel() == T_total, \
        f"s_tx length {s.numel()} != M*(Nfft+cp_len)={T_total}"

    # Continuous-time axis for Doppler phase
    t_all = torch.arange(T_total, device=device, dtype=torch.float32) * Ts  # (T,)

    # Initialize received time-domain signal
    r = torch.zeros_like(s)

    # ---------------- sum of delayed & Doppler-shifted copies ----------
    for tap in taps:
        # Required keys in each tap dict
        alpha = tap["alpha"].to(device)                  # complex scalar
        n_delay = int(tap["n_delay"])                    # integer samples
        f_d = float(tap["f_d"])                          # Hz

        # Create delayed version:
        #   s_delayed[n] = s[n - n_delay]   for n >= n_delay, else 0
        s_delayed = torch.zeros_like(s)
        if n_delay < T_total:
            s_delayed[n_delay:] = s[: T_total - n_delay]

        # Complex Doppler modulation exp(j 2π f_d t)
        phase = torch.exp(2j * np.pi * f_d * t_all)      # (T,)
        r += alpha * s_delayed * phase

    # ---------------- AWGN ---------------------------------------------
    if noise_std > 0.0:
        noise = (torch.randn_like(r) + 1j * torch.randn_like(r)) * noise_std
        r = r + noise

    # ---------------- optional MTI along slow-time --------------------
    if mti_order > 0:
        # Reshape to [M, samples_per_sym] so that "slow-time" is the first dim
        r_mat = r.view(M, samples_per_sym)
        for _ in range(mti_order):
            # simple first-order difference along slow-time:
            # r'[0,:] = r[0,:];  r'[m,:] = r[m,:] - r[m-1,:]
            r_mat = torch.cat(
                [r_mat[0:1, :], r_mat[1:, :] - r_mat[:-1, :]],
                dim=0,
            )
        # Flatten back to 1-D so downstream code still sees a vector
        r = r_mat.reshape(-1)

    # Final 1-D received waveform
    return r

def _otfs_rx_dd_from_waveform(r: torch.Tensor, otfs_par: dict):
    """
    OTFS receiver: time-domain waveform -> DD grid.

    Steps (inverse of _otfs_tx_waveform_from_dd):

      1) Segment r into M OFDM symbols, remove CP.
      2) FFT across subcarriers to obtain TF grid Y_tf[n, m].
      3) SFFT: Y_dd = SFFT{Y_tf} to obtain delay–Doppler domain.

    Parameters
    ----------
    r : (L,) complex64 tensor
        Received baseband waveform.
    otfs_par : dict
        OTFS parameters (M, N, cp_len, etc.)

    Returns
    -------
    Y_dd : (M, N) complex64 tensor
        Estimated DD domain response.
    Y_tf : (M, N) complex64 tensor
        TF grid at the receiver (for debugging).
    """
    M     = otfs_par["M"]
    N_sc  = otfs_par["N"]
    cp_len= otfs_par["cp_len"]

    device = r.device
    L_sym = N_sc + cp_len
    L_tot = L_sym * M
    assert r.shape[0] >= L_tot, "Received waveform shorter than OTFS frame"

    # 1) Remove CP and reshape per symbol
    y_time = torch.zeros((M, N_sc), dtype=torch.complex64, device=device)
    for n in range(M):
        start = n * L_sym
        seg   = r[start + cp_len : start + L_sym]  # (N_sc,)
        y_time[n] = seg

    # 2) FFT across subcarriers to get TF grid
    # torch.fft.fft is unnormalized, so divide by sqrt(N_sc) to match TX
    Y_tf = torch.fft.fft(y_time, dim=1) / np.sqrt(N_sc)  # (M, N_sc)

    # 3) TF -> DD via SFFT
    Y_dd = _otfs_sfft_tf_to_dd(Y_tf)
    return Y_dd, Y_tf

def _make_otfs_dd_groundtruth(taps, sp, otfs_par):
    """
    Build a sparse ground-truth delay–Doppler grid from channel taps.

    We discretize:
        - delay using the same sample grid as OTFS (N_time samples per symbol)
        - Doppler using FFT bins over slow-time (M bins).

    Parameters
    ----------
    taps : list of dict
        Delay–Doppler taps from _build_otfs_channel_from_gts.
    sp : SystemParams
    otfs_par : dict

    Returns
    -------
    H_gt : (M, N) complex64 ndarray
        DD channel grid.
    delay_axis_m : (N,) float32 ndarray
        Range/delay axis in meters (two-way).
    doppler_axis_hz : (M,) float32 ndarray
        Doppler axis in Hz (fftshift'ed).
    """
    M    = otfs_par["M"]
    N_sc = otfs_par["N"]
    fs   = otfs_par["fs"]
    T_sym= otfs_par["T_sym"]

    device = DEVICE
    H = torch.zeros((M, N_sc), dtype=torch.complex64, device=device)

    if taps:
        # Doppler bin spacing and axis (unshifted then shifted)
        f_res = 1.0 / (M * T_sym)  # Doppler resolution (Hz)

        for tap in taps:
            n_delay = int(tap["n_delay"])
            f_d     = float(tap["f_d"])
            alpha   = tap["alpha"].to(device)

            # Map sample delay to delay bin index in [0, N_sc-1]
            # Here we assume symbol length N_sc, so we just clip
            l_idx = int(np.clip(n_delay, 0, N_sc - 1))

            # Map Doppler frequency to FFT bin (0..M-1), then fftshift later
            k_unshift = int(round(f_d / f_res))  # may be negative or >M
            k_idx = (k_unshift + M) % M          # wrap into 0..M-1

            H[k_idx, l_idx] += alpha

    # Axes
    delay_axis_m = (np.arange(N_sc, dtype=np.float32) / fs * C0 / 2.0)
    doppler_axis_hz = np.fft.fftshift(np.fft.fftfreq(M, d=T_sym))

    # fftshift H along Doppler (rows) to align with doppler_axis_hz
    H = torch.fft.fftshift(H, dim=0)
    return H.cpu().numpy(), delay_axis_m, doppler_axis_hz

import torch
import numpy as np

def _make_dd_groundtruth_from_taps(
    taps,
    otfs_par,
    device=DEVICE,
):
    """
    Build a 'ground truth' delay–Doppler grid H_gt(M, N) directly
    from the discrete tap list.

    Parameters
    ----------
    taps : list[dict]
        Each tap is a dict with at least:
          - 'n_delay' : int   (delay index, 0..N-1)
          - 'm_dopp'  : int   (Doppler index, 0..M-1, *unshifted*)
          - 'alpha'   : complex-valued torch scalar
          - 'tau'     : float (continuous delay in seconds, optional)
          - 'nu'      : float (Doppler in Hz, optional)
          - 'f_d'     : float (same as nu, optional)

        Only 'n_delay', 'm_dopp', 'alpha' are strictly needed to
        build H_gt, the rest are for reference / debugging.

    otfs_par : dict
        OTFS parameter dict, expected keys:
          - "M"    : Doppler bins
          - "N"    : delay bins
          - "T_sym": OTFS symbol duration [s]
          - "Ts"   : base sampling interval [s]

    device : torch.device
        Device for intermediate tensors. Final H_gt and axes are
        returned as numpy arrays for easy plotting.

    Returns
    -------
    H_gt_np : (M, N) np.complex64
        Sparse delay–Doppler ground truth grid with taps placed at
        (m_dopp, n_delay).

    delays : (N,) np.float32
        Delay axis in seconds (unshifted).

    f_dop : (M,) np.float32
        Doppler axis in Hz, *fftshifted* to match typical DD display
        (DC in the center), consistent with 'dopplers' you already
        store in extras.
    """
    M = int(otfs_par["M"])
    N = int(otfs_par["N"])
    T_sym = float(otfs_par["T_sym"])
    Ts = float(otfs_par["Ts"])

    # Allocate complex64 DD grid on device for accumulation
    H_gt = torch.zeros((M, N), dtype=torch.complex64, device=device)

    if taps is None or len(taps) == 0:
        # Empty → just return zeros + axes
        delays = (torch.arange(N, device=device) * Ts).float()
        f_dop = torch.fft.fftshift(torch.fft.fftfreq(M, d=T_sym)).float()
        return (
            H_gt.cpu().numpy(),
            delays.cpu().numpy(),
            f_dop.cpu().numpy(),
        )

    # Rasterize each tap into (m_dopp, n_delay)
    for tap in taps:
        # Indices come from channel construction (assumed integer)
        m = int(tap["m_dopp"])
        n = int(tap["n_delay"])

        # Safety clamp to grid bounds
        if not (0 <= m < M and 0 <= n < N):
            continue

        # Complex gain for this tap
        alpha = tap["alpha"]
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.to(device=device, dtype=torch.complex64)
        else:
            alpha = torch.as_tensor(alpha, dtype=torch.complex64, device=device)

        # Accumulate (allow multiple taps to land on same bin)
        H_gt[m, n] += alpha

    # Build axes: delay (seconds) and Doppler (Hz, fftshifted)
    delays = (torch.arange(N, device=device) * Ts).float()
    f_dop = torch.fft.fftshift(torch.fft.fftfreq(M, d=T_sym)).float()

    return (
        H_gt.cpu().numpy(),        # (M, N) complex64
        delays.cpu().numpy(),      # (N,)
        f_dop.cpu().numpy(),       # (M,)
    )
    
# ------------------------------------------------------------
# OTFS modulation: DD grid -> time-domain baseband waveform
# ------------------------------------------------------------
def _otfs_modulate_frame_dd(X_dd: torch.Tensor, otfs_par: dict) -> torch.Tensor:
    """
    OTFS modulation (discrete-time, baseband).

    Input
    -----
    X_dd : (M, N) torch.complex64
        Delay–Doppler grid X[k, l] in DD domain. For radar we
        usually put a single strong pilot at one (k,l) location.

    otfs_par : dict
        OTFS configuration, must contain at least:
          - "M"    : number of Doppler bins (slow-time symbols)
          - "N"    : number of delay bins  (subcarriers)
          - "Nfft" : OFDM FFT size (usually == N)
          - "cp_len" : cyclic prefix length (samples)
        Any extra keys are ignored here.

    Output
    ------
    s_tx : (M * (Nfft + cp_len),) torch.complex64
        Time-domain baseband waveform ready for the channel.

    Pipeline
    --------
    1) ISFFT (very simple version): DD -> TF
          X_tf = ISFFT{X_dd} implemented as a 2-D IFFT.
       This gives X_tf[m, n] (time-frequency grid).

    2) Per-OFDM-symbol IFFT: for each Doppler index m,
          OFDM symbol x_ofdm[m, :] = IFFT over N subcarriers.

    3) CP insertion: prepend cp_len samples per symbol.

    4) Serialisation: stack symbols row-wise into a 1D stream.
    """
    assert X_dd.ndim == 2, "X_dd must be 2-D (M, N)"
    M, N = X_dd.shape
    M_cfg = int(otfs_par["M"])
    N_cfg = int(otfs_par["N"])
    assert M == M_cfg and N == N_cfg, \
        f"X_dd shape {(M,N)} != otfs_par (M={M_cfg}, N={N_cfg})"

    device = X_dd.device
    Nfft = int(otfs_par.get("Nfft", N))
    cp_len = int(otfs_par.get("cp_len", 0))
    assert Nfft == N, "For this simple impl, Nfft must equal N."

    # 1) ISFFT (DD -> TF).  We use a simple 2D IFFT pair.
    #    Scale by sqrt(MN) so that FFT/ IFFT are approximately unitary.
    X_tf = torch.fft.ifft2(X_dd) * np.sqrt(M * N)      # (M, N) complex

    # 2) OFDM modulation over frequency (axis 1).
    #    Each row m is one OFDM symbol in time domain.
    x_ofdm = torch.fft.ifft(X_tf, n=Nfft, dim=1)       # (M, Nfft)

    # 3) CP insertion.
    if cp_len > 0:
        cp = x_ofdm[:, -cp_len:]                       # (M, cp_len)
        x_cp = torch.cat([cp, x_ofdm], dim=1)          # (M, cp_len+Nfft)
    else:
        x_cp = x_ofdm                                  # (M, Nfft)

    # 4) Serialise OFDM symbols into a 1D stream (row-major).
    s_tx = x_cp.reshape(-1).to(torch.complex64)        # (M*(Nfft+cp_len),)
    return s_tx

# ------------------------------------------------------------
# OTFS demodulation: time-domain waveform -> DD grid
# ------------------------------------------------------------
def _otfs_demod_frame_time(r_rx: torch.Tensor, otfs_par: dict) -> torch.Tensor:
    """
    OTFS demodulation (discrete-time, baseband).

    Input
    -----
    r_rx : (L,) torch.complex64
        Received time-domain baseband waveform after channel.

    otfs_par : dict
        OTFS configuration with the same keys as used in
        `_otfs_modulate_frame_dd`:
          - "M", "N", "Nfft", "cp_len"

    Output
    ------
    Y_dd : (M, N) torch.complex64
        Demodulated Delay–Doppler grid, approximately equal to
        H[k,l] ⊙ X_dd when the channel is diagonal in DD domain.
        With a single DD pilot at (0,0), Y_dd ≈ H[k,l].

    Pipeline
    --------
    1) Reshape r_rx into M OFDM symbols + CP.
    2) Remove CP.
    3) FFT over frequency (subcarriers) to get TF grid R_tf.
    4) SFFT (2D FFT) to go TF -> DD:
           Y_dd = SFFT{R_tf}.
    """
    assert r_rx.ndim == 1, "r_rx must be a 1-D time-domain vector"
    M = int(otfs_par["M"])
    N = int(otfs_par["N"])
    Nfft = int(otfs_par.get("Nfft", N))
    cp_len = int(otfs_par.get("cp_len", 0))
    assert Nfft == N, "For this simple impl, Nfft must equal N."

    device = r_rx.device

    sym_len = Nfft + cp_len          # samples per OFDM symbol (with CP)
    total_needed = M * sym_len
    if r_rx.numel() < total_needed:
        raise ValueError(
            f"r_rx length {r_rx.numel()} < required {total_needed} "
            f"for M={M}, Nfft={Nfft}, cp_len={cp_len}"
        )

    # 1) Reshape into (M, sym_len)
    r_mat = r_rx[:total_needed].reshape(M, sym_len)  # (M, cp+Nfft)

    # 2) CP removal
    if cp_len > 0:
        r_no_cp = r_mat[:, cp_len:]                  # (M, Nfft)
    else:
        r_no_cp = r_mat                              # (M, Nfft)

    # 3) FFT over frequency dimension to return to TF grid
    R_tf = torch.fft.fft(r_no_cp, n=Nfft, dim=1)     # (M, Nfft) complex

    # 4) SFFT (TF -> DD).  Use 2D FFT with 1/sqrt(MN) scaling.
    Y_dd = torch.fft.fft2(R_tf) / np.sqrt(M * N)     # (M, N) complex
    return Y_dd.to(torch.complex64)

def _apply_mti_highpass_dd(H_dd: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Simple MTI-style high-pass filtering in the *delay–Doppler* domain.

    This is meant for OTFS radar channel estimates H_dd(m, n):

        m = Doppler index (slow-time / velocity bin, size M)
        n = delay index   (range / delay bin, size N)

    Physical intuition
    ------------------
    • Static / very-slow clutter tends to concentrate near f_d ≈ 0 for all
      delays n. In the DD grid this appears as a strong low-frequency
      component along the Doppler (m) dimension.

    • MTI (Moving Target Indication) suppresses this by high-pass filtering
      along the Doppler dimension: we remove the per-delay DC component
      and optionally apply finite differences along m.

    Parameters
    ----------
    H_dd : torch.Tensor, complex, shape (M, N)
        Complex delay–Doppler channel (e.g., H_est from OTFS demod).

    order : int, default 1
        0 : no MTI, returns H_dd unchanged.
        1 : subtract per-delay DC term (mean over m) – basic clutter removal.
        2+ : after DC removal, apply (order-1) stages of first-difference
             along Doppler (m) to further emphasize moving targets.

    Returns
    -------
    H_hp : torch.Tensor, complex, shape (M, N)
        High-pass filtered DD channel, same dtype/device as input.

    Notes
    -----
    • This operation is *purely along Doppler* (dim=0); delay structure is
      left unchanged.
    • For most radar OTFS experiments, order=1 is usually enough.
      Higher orders will increasingly amplify noise and fast movers.
    """
    if order <= 0:
        return H_dd

    # Ensure we preserve dtype/device
    H_hp = H_dd

    # --------------------------------------------------------------
    # 1) Remove Doppler-DC per delay bin: H_hp(m,n) ← H_dd(m,n) - mean_m H_dd(m,n)
    #    This is the simplest MTI: static clutter at all delays is suppressed.
    # --------------------------------------------------------------
    dc = H_hp.mean(dim=0, keepdim=True)   # shape (1, N), complex
    H_hp = H_hp - dc

    # --------------------------------------------------------------
    # 2) Optional higher-order MTI: finite difference along Doppler
    #    For each additional order, apply a first-difference:
    #
    #       H(m,n) ← H(m,n) - H(m-1,n)
    #
    #    This is the DD-domain analogue of a 1-pole high-pass in slow time.
    # --------------------------------------------------------------
    for _ in range(max(0, order - 1)):
        H_shift = torch.roll(H_hp, shifts=1, dims=0)  # shift in Doppler index
        H_hp = H_hp - H_shift

    return H_hp

def _build_otfs_H_gt_from_taps(
    taps,
    otfs_par,
    use_complex_amp: bool = True,
):
    """
    Build a sparse delay–Doppler *ground-truth* grid H_gt(M,N) from
    the discrete OTFS channel taps.

    Parameters
    ----------
    taps : list of dict
        Each tap is expected to have:
            'n_delay' : int   (delay index, 0..N-1)
            'm_dopp'  : int   (Doppler index, 0..M-1)
            'alpha'   : torch.Tensor or complex (tap gain)
            optionally: 'tau', 'nu', 'f_d' (continuous parameters)
    otfs_par : dict
        Should contain:
            'M' : number of Doppler bins
            'N' : number of delay bins
    use_complex_amp : bool
        If True we use the complex tap gain. If False, we use |alpha|.

    Returns
    -------
    H_gt : np.ndarray, shape (M, N), complex64
        Sparse DD grid with one (or more) bright bins at tap indices.
    """
    M = int(otfs_par["M"])
    N = int(otfs_par["N"])
    H_gt = np.zeros((M, N), dtype=np.complex64)

    if not taps:
        return H_gt

    for tap in taps:
        # Discrete indices in DD grid
        m = int(tap.get("m_dopp", 0)) % M
        n = int(tap.get("n_delay", 0)) % N

        alpha = tap.get("alpha", 0.0)
        # Convert alpha to Python complex
        if isinstance(alpha, torch.Tensor):
            alpha_val = alpha.detach().cpu().numpy().astype(np.complex64)
        else:
            alpha_val = np.complex64(alpha)

        if not use_complex_amp:
            alpha_val = np.abs(alpha_val).astype(np.float32)

        H_gt[m, n] += alpha_val

    return H_gt

#Transmits a single strong pilot in the DD grid
#visualize is the channel H[k,l] (delay–Doppler impulse response), not the random data symbols.
def otfs_torch_full_radar(
    gts,
    sp: SystemParams,
    otfs_par: dict,
    noise_std: float = 1e-3,
    pilot_amplitude: float = 1.0,
    pilot_pos=(0, 0),
    return_extra: bool = False,
    mti_order: int = 0,
):
    """
    Full OTFS radar pipeline driven by geometric GT (gts), not raycast hits.

    Pipeline:
      1) Build a small number of *physical* channel taps from gts:
            tau_l  (delay), nu_l (Doppler), alpha_l (complex gain),
            plus discrete indices n_delay, m_dopp.
      2) Place a pilot in DD grid X_dd and modulate to time-domain s_tx.
      3) Apply delay–Doppler channel in time domain → r_rx.
      4) Demodulate OTFS (time → DD) to obtain H_est(m,n).
      5) Build a sparse ground-truth H_gt(m,n) from taps.
      6) (Optional) MTI-like high-pass in Doppler on both H_est, H_gt.
      7) Convert H_est to dB for visualization / training.

    This is a *radar-style* OTFS frame: the channel is the “target scene”.
    """

    device = DEVICE
    M = int(otfs_par["M"])
    N = int(otfs_par["N"])

    # 1) Geometry → OTFS channel taps in DD
    taps = _build_otfs_channel_from_gts(gts, sp, otfs_par)
    # taps: [{'n_delay', 'm_dopp', 'alpha', 'tau', 'nu', 'f_d'}, ...]

    # 2) Build a DD grid with a *single pilot* at pilot_pos
    X_dd = torch.zeros((M, N), dtype=torch.complex64, device=device)
    m_p, n_p = pilot_pos
    X_dd[m_p % M, n_p % N] = pilot_amplitude + 0j

    # 3) OTFS modulation: DD → time-domain
    #    s_tx is a 1-D complex time series with length M * (N + cp_len)
    s_tx = _otfs_modulate_frame_dd(X_dd, otfs_par)        # (T,) complex torch

    # 4) Pass through delay–Doppler channel in time domain
    r_rx = _apply_delay_doppler_channel_time(
        s_tx,
        taps,
        otfs_par,
        noise_std=noise_std,
    )  # (T,) complex torch

    # 5) OTFS demodulation: time-domain → DD
    H_est = _otfs_demod_frame_time(r_rx, otfs_par)        # (M, N) complex torch

    # 6) Build *analytical* DD ground-truth from taps
    H_gt_np = _build_otfs_H_gt_from_taps(taps, otfs_par)  # (M, N) complex64

    # 7) Optional MTI in Doppler (high-pass along m index)
    if mti_order > 0:
        H_est = _apply_mti_highpass_dd(H_est, order=mti_order)

    # 8) Axes: continuous delay [s] and Doppler [Hz]
    #    (You already print these in extra_otfs.)
    delays = otfs_par["Ts"] * np.arange(N, dtype=np.float64)  # sample delays
    dopplers = np.fft.fftshift(np.fft.fftfreq(M, d=otfs_par["T_sym"]))

    # 9) Convert H_est to numpy dB for visualization / training
    H_est_np = H_est.detach().cpu().numpy().astype(np.complex64)
    dd_db = 20.0 * np.log10(np.abs(H_est_np) + 1e-12)

    if not return_extra:
        return dd_db,  # keep tuple compatibility

    extra = {
        # Measured DD channel estimate
        "H_est": H_est_np,          # complex (M,N)
        "dd_db": dd_db,             # float (M,N)

        # Analytical ground truth from taps
        "H_gt": H_gt_np,            # complex (M,N)

        # Axes
        "delays": delays,           # (N,) seconds
        "dopplers": dopplers,       # (M,) Hz

        # Channel model details
        "taps": taps,               # list of dicts
        "s_tx": s_tx.detach().cpu().numpy().astype(np.complex64),
        "r_rx": r_rx.detach().cpu().numpy().astype(np.complex64),

        # Original GT for overlay
        "gts": gts,
    }

    return dd_db, extra

#communication-style OTFS frame (lots of QAM symbols)
#With random QAM all over the DD grid, the demodulated symbol power is almost flat across all bins, so the plot looks like noise even though the channel taps exist.
def otfs_torch_full(
    gts,
    sp: SystemParams,
    noise_std: float = 1e-3,
    return_extra: bool = False,
):
    """
    Full OTFS radar simulation (single frame) using GT-based channel.

    This implements the *complete* OTFS chain:

      DD domain:
        1) Choose DD QAM grid X_dd (radar training frame).
        2) Build DD channel taps from GT targets.

      TX:
        3) ISFFT: X_dd -> X_tf (time–frequency grid).
        4) Heisenberg (OFDM) modulation: X_tf -> s[n] (time waveform).

      Channel:
        5) Apply time-domain delay–Doppler channel taps + AWGN.

      RX:
        6) OFDM demodulation: r[n] -> Y_tf.
        7) SFFT: Y_tf -> Y_dd (estimated DD response).

      Output:
        8) DD magnitude map in dB: 20 log10 |Y_dd|.

    Parameters
    ----------
    gts : list of dict
        Ground-truth objects, each with keys 'c', 'v', 's'.
    sp : SystemParams
        Radar/system parameters (fs, M, N, lambda_m, H, ...).
    noise_std : float
        AWGN std-dev in the time domain.
    return_extra : bool, default False
        If True, also return an `extra` dict with many intermediate
        quantities for visualization and debugging.

    Returns
    -------
    dd_db : (M, N) float32 ndarray
        OTFS delay–Doppler magnitude in dB.
    extra : dict (only if return_extra=True)
        Contains:
          - 'X_dd'  : TX DD grid (M,N) complex64
          - 'X_tf'  : TX TF grid (M,N) complex64
          - 's_tx'  : TX waveform (L,) complex64
          - 'r_rx'  : RX waveform (L,) complex64
          - 'Y_tf'  : RX TF grid (M,N) complex64
          - 'Y_dd'  : RX DD grid (M,N) complex64
          - 'H_gt'  : GT DD grid (M,N) complex64
          - 'delays_m'   : (N,) range/delay axis [m]
          - 'doppler_hz' : (M,) Doppler axis [Hz]
          - 'taps'       : list of channel taps (alpha, n_delay, f_d)
    """
    device = DEVICE
    otfs_par = _otfs_params_from_sp(sp)
    M, N_sc = otfs_par["M"], otfs_par["N"]

    if not gts:
        # no targets -> just noise
        dd_empty = np.zeros((M, N_sc), dtype=np.float32)
        if return_extra:
            delays_m = (np.arange(N_sc, dtype=np.float32) /
                        otfs_par["fs"] * C0 / 2.0)
            doppler_hz = np.fft.fftshift(np.fft.fftfreq(M, d=otfs_par["T_sym"]))
            extra = dict(
                X_dd=np.zeros((M, N_sc), np.complex64),
                X_tf=np.zeros((M, N_sc), np.complex64),
                s_tx=np.zeros(M * (N_sc + otfs_par["cp_len"]), np.complex64),
                r_rx=np.zeros(M * (N_sc + otfs_par["cp_len"]), np.complex64),
                Y_tf=np.zeros((M, N_sc), np.complex64),
                Y_dd=np.zeros((M, N_sc), np.complex64),
                H_gt=np.zeros((M, N_sc), np.complex64),
                delays_m=delays_m,
                doppler_hz=doppler_hz,
                taps=[],
            )
            return dd_empty, extra
        return (dd_empty,)

    # 1) Build DD channel from GTs
    taps = _build_otfs_channel_from_gts(gts, sp, otfs_par)

    # 2) Choose DD QAM grid (for radar, often all-ones pilot)
    X_dd = torch.ones((M, N_sc), dtype=torch.complex64, device=device)

    # 3–4) OTFS TX: DD -> TF -> time-domain
    s_tx, X_tf = _otfs_tx_waveform_from_dd(X_dd, otfs_par)

    # 5) Channel: delay–Doppler taps in time domain
    r_rx = _apply_delay_doppler_channel_time(s_tx, taps, otfs_par,
                                             noise_std=noise_std)

    # 6–7) OTFS RX: time-domain -> TF -> DD
    Y_dd, Y_tf = _otfs_rx_dd_from_waveform(r_rx, otfs_par)

    # 8) Magnitude in dB (this is your OTFS DD "radar image")
    mag = torch.abs(Y_dd).clamp_min(1e-12)
    dd_db = (20.0 * torch.log10(mag)).cpu().numpy().astype(np.float32)

    if not return_extra:
        return (dd_db,)

    # Ground-truth DD grid for visualization
    H_gt, delays_m, doppler_hz = _make_otfs_dd_groundtruth(taps, sp, otfs_par)

    extra = dict(
        X_dd=X_dd.detach().cpu().numpy().astype(np.complex64),
        X_tf=X_tf.detach().cpu().numpy().astype(np.complex64),
        s_tx=s_tx.detach().cpu().numpy().astype(np.complex64),
        r_rx=r_rx.detach().cpu().numpy().astype(np.complex64),
        Y_tf=Y_tf.detach().cpu().numpy().astype(np.complex64),
        Y_dd=Y_dd.detach().cpu().numpy().astype(np.complex64),
        H_gt=H_gt.astype(np.complex64),
        delays_m=delays_m.astype(np.float32),
        doppler_hz=doppler_hz.astype(np.float32),
        taps=taps,
    )
    return dd_db, extra

###simplified otfs_torch
def _pulsed_baseband_iq(
    scat,
    sp: SystemParams,
    noise_std: float = 1e-3,
    mti_order: int = 0,
):
    """
    OTFS-like pulsed radar baseband IQ generation (delay–Doppler).

    Conceptual steps:
      - "Transmitter": short pulses at each PRI (implicitly modeled
        by placing energy at discrete delays).
      - "Channel"   : delays set by ranges, Doppler phase across pulses
        set by radial velocities.
      - "Receiver"  : sampled baseband IQ (after matched filtering,
        simplified here as impulses at delay taps).

    Parameters
    ----------
    scat : dict
        Output of _prepare_scatters_from_raycast with keys "R", "vr", "amp".

    sp : SystemParams
        Uses M, N, fs, T_chirp (as PRI), lambda_m.

    noise_std : float, default 1e-3
        Complex AWGN standard deviation.

    mti_order : int, default 0
        Optional MTI order along slow-time (same method as FMCW). For pure
        delay–Doppler, you may want mti_order=0; but you can turn it on
        for stronger clutter rejection.

    Returns
    -------
    iq : (M, N) complex64 torch.Tensor
        Pulsed radar IQ cube.
    """
    M, N = sp.M, sp.N

    R = scat["R"]      # (P,)
    vr = scat["vr"]    # (P,)
    amp = scat["amp"]  # (P,)

    # Two-way delay τ_p = 2R_p / c, discrete index n0 = round(τ_p * fs)
    delay_s = 2.0 * R / C0
    n0 = (delay_s * sp.fs).round().long()
    n0 = torch.clamp(n0, 0, N - 1)

    # Doppler frequency f_d,p = 2 v_r / λ
    f_d = 2.0 * vr / sp.lambda_m  # (P,)

    # Pulse indices
    m_idx = torch.arange(M, device=DEVICE, dtype=torch.float32)  # (M,)

    # IQ cube
    iq = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)

    # For each scatterer, place impulse at delay n0[p] with Doppler phase
    for p in range(R.shape[0]):
        phase_m = 2j * np.pi * (f_d[p] * m_idx * sp.T_chirp)   # (M,)
        iq[:, n0[p]] += amp[p] * torch.exp(phase_m)

    # Add AWGN
    iq = iq + (torch.randn(M, N, device=DEVICE) +
               1j * torch.randn(M, N, device=DEVICE)) * noise_std

    # Optional MTI
    iq = _apply_mti_highpass(iq, order=mti_order)

    return iq


# ---------------------------------------------------------------------
# OTFS-like: signal processing → Delay–Doppler map
# ---------------------------------------------------------------------
def _otfs_dd_from_iq(iq: torch.Tensor, sp: SystemParams):
    """
    Convert OTFS-like pulsed IQ cube into a Delay–Doppler map via 2-D FFT.

    Steps
    -----
    1) Apply 2-D Hann window.
    2) FFT along fast-time (delay dimension).
    3) FFT along slow-time (Doppler) with fftshift.

    Parameters
    ----------
    iq : (M, N) complex64 torch.Tensor
        Pulsed baseband IQ.

    sp : SystemParams
        Uses M, N, fs, T_chirp.

    Returns
    -------
    dd_db : (M, N) np.ndarray, float32
        Delay–Doppler magnitude in dB.

    dd_complex : (M, N) np.ndarray, complex64
        Complex DD map.
    """
    M, N = sp.M, sp.N

    iq_win = _apply_hann_2d(iq, sp)

    # FFT along delay (fast-time)
    F_delay = torch.fft.fft(iq_win, dim=1)  # (M, N)

    # FFT along slow-time (Doppler) + shift
    DD = torch.fft.fftshift(torch.fft.fft(F_delay, dim=0), dim=0)  # (M, N)

    DD_mag = torch.abs(DD).clamp_min(1e-12)
    dd_db = (20.0 * torch.log10(DD_mag)).cpu().numpy().astype(np.float32)

    dd_complex = DD.detach().cpu().numpy().astype(np.complex64)
    return dd_db, dd_complex


# ---------------------------------------------------------------------
# Public wrapper: OTFS-like radar simulation
# ---------------------------------------------------------------------
def otfs_torch(
    pts,
    its,
    vels,
    sp: SystemParams,
    labels=None,
    gts=None,
    viz_dir=None,
    return_extra: bool = False,
    mti_order: int = 0,
    noise_std: float = 1e-3,
):
    """
    High-level OTFS-like pulsed radar simulation from raycast hits.

    Reuses the same modular components as FMCW:

      1) _prepare_scatters_from_raycast  (geometry + amplitude model)
      2) _pulsed_baseband_iq             (TX/Channel/RX baseband)
      3) _otfs_dd_from_iq                (2D FFT → Delay–Doppler map)
      4) _make_dd_groundtruth            (sparse DD ground-truth grid)

    Parameters
    ----------
    pts, its, vels
        Outputs from raycast_torch.

    sp : SystemParams
        Global radar configuration.

    labels, gts, viz_dir
        Same semantics as fmcw_torch; kept for symmetry and further use.

    return_extra : bool, default False
        If True, returns (dd_db, extra) where extra includes:
          - "iq"       : time-domain IQ cube (M, N) complex64 numpy
          - "DD"       : complex DD map (M, N) complex64 numpy
          - "H_gt"     : ground-truth DD grid (M, N) complex64 numpy
          - "delays"   : delay axis (seconds, N,)
          - "dopplers" : Doppler axis (Hz, M,)
          - "R", "vr", "amp" : torch Tensors for scatterers
          - "gts"      : original GT list

    mti_order : int, default 0
        Optional MTI order along slow-time. For OTFS-like DD maps you may
        choose 0 (no MTI) or 1 for simple clutter rejection.

    noise_std : float, default 1e-3
        Complex AWGN standard deviation.

    Returns
    -------
    out : tuple
        Backward-compatible behavior:
          - If return_extra=False: (dd_db,)  with dd_db shape (M, N).
          - If return_extra=True : dd_db, extra
    """
    # Empty-scene handling
    if pts is None or len(pts) == 0:
        dd_empty = np.zeros((sp.M, sp.N), dtype=np.float32)
        if return_extra:
            H_gt, delays, f_dop = (
                np.zeros_like(dd_empty, dtype=np.complex64),
                np.arange(sp.N) / sp.fs,
                np.fft.fftshift(np.fft.fftfreq(sp.M, d=sp.T_chirp)),
            )
            extra = dict(
                iq=np.zeros((sp.M, sp.N), dtype=np.complex64),
                DD=np.zeros((sp.M, sp.N), dtype=np.complex64),
                H_gt=H_gt,
                delays=delays,
                dopplers=f_dop,
                R=torch.empty(0, device=DEVICE),
                vr=torch.empty(0, device=DEVICE),
                amp=torch.empty(0, device=DEVICE),
                gts=gts,
            )
            return dd_empty, extra
        return (dd_empty,)

    # 1) Prepare scatterers
    scat = _prepare_scatters_from_raycast(pts, its, vels, sp)
    if scat is None:
        dd_empty = np.zeros((sp.M, sp.N), dtype=np.float32)
        if return_extra:
            H_gt, delays, f_dop = (
                np.zeros_like(dd_empty, dtype=np.complex64),
                np.arange(sp.N) / sp.fs,
                np.fft.fftshift(np.fft.fftfreq(sp.M, d=sp.T_chirp)),
            )
            extra = dict(
                iq=np.zeros((sp.M, sp.N), dtype=np.complex64),
                DD=np.zeros((sp.M, sp.N), dtype=np.complex64),
                H_gt=H_gt,
                delays=delays,
                dopplers=f_dop,
                R=torch.empty(0, device=DEVICE),
                vr=torch.empty(0, device=DEVICE),
                amp=torch.empty(0, device=DEVICE),
                gts=gts,
            )
            return dd_empty, extra
        return (dd_empty,)

    # 2) Baseband IQ
    iq = _pulsed_baseband_iq(scat, sp, noise_std=noise_std, mti_order=mti_order)

    # 3) Signal processing → DD map
    dd_db, DD_complex = _otfs_dd_from_iq(iq, sp)

    # 4) Ground-truth DD grid
    H_gt, delays, f_dop = _make_dd_groundtruth(
        scat["R"], scat["vr"], scat["amp"], sp
    )

    if not return_extra:
        return (dd_db,)

    extra = dict(
        iq=iq.detach().cpu().numpy().astype(np.complex64),
        DD=DD_complex,
        H_gt=H_gt,
        delays=delays,
        dopplers=f_dop,
        R=scat["R"],
        vr=scat["vr"],
        amp=scat["amp"],
        gts=gts,
    )
    return dd_db, extra
# def fmcw_torch(pts, its, vels, sp: SystemParams):
#     """
#     Synthesizes a FMCW Range–Doppler map in dB.

#     Returns
#     -------
#     rd_db : np.ndarray
#         RD magnitude in dB, shape (M, N/2).
#     """
#     M, N = sp.M, sp.N
#     iq = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)
#     if len(pts) == 0:
#         return (np.zeros((M, N // 2), np.float32),)

#     P = pts - to_torch([0, 0, sp.H])
#     R = torch.norm(P, dim=1)
#     mask = R > 0.1
#     P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
#     amp = torch.where(its == 255, 1e6, 1e-1) / (R ** 2 + 1e-6)
#     vr = torch.sum(P / R.unsqueeze(1) * vels, dim=1)

#     t_f = torch.arange(N, device=DEVICE) / sp.fs
#     t_s = torch.arange(M, device=DEVICE) * sp.T_chirp
#     k_r = 2 * sp.slope / C0
#     k_v = 2 / sp.lambda_m

#     BATCH = 4096
#     for i in range(0, len(R), BATCH):
#         rb = R[i : i + BATCH]
#         vrb = vr[i : i + BATCH]
#         ab = amp[i : i + BATCH]
#         phase = 2j * np.pi * (
#             (k_r * rb[:, None, None]) * t_f[None, None, :]
#             + (k_v * vrb[:, None, None]) * t_s[None, :, None]
#         )
#         iq += torch.sum(ab[:, None, None] * torch.exp(phase), dim=0)

#     # noise + 1st-order MTI
#     iq = iq + (torch.randn(M, N, device=DEVICE) +
#                1j * torch.randn(M, N, device=DEVICE)) * 1e-4
#     iq[1:] -= iq[:-1].clone()
#     iq[0] = 0

#     w_r = torch.hann_window(N, device=DEVICE)
#     w_d = torch.hann_window(M, device=DEVICE)
#     iq = iq * (w_d[:, None] * w_r[None, :])

#     RFFT = torch.fft.fft(iq, dim=1)
#     RFFT = RFFT[:, : N // 2]
#     RD = torch.fft.fftshift(torch.fft.fft(RFFT, dim=0), dim=0)

#     RD_mag = torch.abs(RD).clamp_min(1e-12)
#     rd_db = 20 * torch.log10(RD_mag).cpu().numpy()
#     return (rd_db,)


# def otfs_torch(pts, its, vels, sp: SystemParams):
#     """
#     OTFS / pulsed radar time-domain simulation → Delay–Doppler map.

#     We build a slow-time / fast-time IQ cube similar to FMCW, but instead of
#     chirping within each pulse we place scatterers as *delayed impulses* in
#     fast-time with a Doppler-induced phase progression across pulses.

#     Steps
#     -----
#     1) For each target:
#          - Range R  → two-way delay τ = 2R/c → discrete index n0.
#          - Radial velocity v_r → Doppler f_d = 2 v_r / λ.
#     2) For each pulse m:
#          iq[m, n0] += amp * exp(j 2π f_d m T_pri)
#     3) Add complex AWGN.
#     4) Apply Hann windows in range & Doppler.
#     5) 2-D FFT (range then Doppler + fftshift) → delay–Doppler map H_dd.
#     6) Return 20 log10 |H_dd| as a float32 numpy array of shape (M, N).
#     """
#     M, N = sp.M, sp.N

#     # time-domain IQ cube (M pulses × N fast-time samples)
#     iq = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)

#     if len(pts) == 0:
#         return (np.zeros((M, N), dtype=np.float32),)

#     # --- geometry & kinematics -----------------------------------------
#     P = pts - to_torch([0.0, 0.0, sp.H])   # relative to radar phase center
#     R = torch.norm(P, dim=1)              # ranges
#     mask = R > 0.1                        # ignore self / extremely close
#     P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
#     if P.numel() == 0:
#         return (np.zeros((M, N), dtype=np.float32),)

#     # amplitude with simple 1/R^2 decay, bright "car" vs dim clutter
#     amp = torch.where(its == 255, 1e6, 1e-1) / (R ** 2 + 1e-6)

#     # radial velocity for each target (project onto line-of-sight)
#     vr = torch.sum(P / R.unsqueeze(1) * vels, dim=1)   # (P,)

#     # --- convert to delay & Doppler ------------------------------------
#     # two-way delay τ = 2R/c, discrete delay index n0 = τ * fs
#     delay_s = 2.0 * R / C0                    # seconds
#     n0 = (delay_s * sp.fs).round().long()
#     n0 = torch.clamp(n0, 0, N - 1)            # stay inside fast-time grid

#     # Doppler frequency for each target
#     f_d = 2.0 * vr / sp.lambda_m             # Hz
#     m_idx = torch.arange(M, device=DEVICE)   # pulse indices 0..M-1
#     # phase_m has shape (P, M): phase across pulses for each scatterer
#     phase_m = 2j * np.pi * (f_d[:, None] * m_idx[None, :] * sp.T_chirp)

#     # --- accumulate echoes into the IQ cube ----------------------------
#     for p in range(R.shape[0]):
#         iq[:, n0[p]] += amp[p] * torch.exp(phase_m[p])

#     # --- add complex AWGN ----------------------------------------------
#     noise_std = 1e-4
#     iq = iq + (torch.randn(M, N, device=DEVICE) +
#                1j * torch.randn(M, N, device=DEVICE)) * noise_std

#     # --- windowing in both dimensions to reduce sidelobes --------------
#     w_r = torch.hann_window(N, device=DEVICE)
#     w_d = torch.hann_window(M, device=DEVICE)
#     iq = iq * (w_d[:, None] * w_r[None, :])

#     # --- 2D FFT → Delay–Doppler map -----------------------------------
#     # range FFT (full N), then Doppler FFT with fftshift
#     RFFT = torch.fft.fft(iq, dim=1)                    # (M, N)
#     DD   = torch.fft.fftshift(torch.fft.fft(RFFT, dim=0), dim=0)

#     DD_mag = torch.abs(DD).clamp_min(1e-12)
#     rd_db = (20.0 * torch.log10(DD_mag)).cpu().numpy().astype(np.float32)
#     return (rd_db,)


# ---------------------------------------------------------------------
# 2D MOVING SUM & CFAR DETECTOR
# ---------------------------------------------------------------------
def _moving_sum_2d(a, r, c):
    """Fast sliding-window sum with integral image (no SciPy required)."""
    if r == 0 and c == 0:
        return a.copy()
    ap = np.pad(a, ((r, r), (c, c)), mode="edge")
    S = ap.cumsum(axis=0).cumsum(axis=1)
    H, W = a.shape
    s22 = S[2 * r : 2 * r + H, 2 * c : 2 * c + W]
    s02 = S[0:H, 2 * c : 2 * c + W]
    s20 = S[2 * r : 2 * r + H, 0:W]
    s00 = S[0:H, 0:W]
    return s22 - s02 - s20 + s00


def nms2d(arr, kernel=3):
    """Plain python NMS for 2D array (no SciPy)."""
    k = max(3, int(kernel) | 1)
    pad = k // 2
    ap = np.pad(arr, ((pad, pad), (pad, pad)), mode="edge")
    max_nb = np.full_like(arr, -np.inf)
    for di in range(-pad, pad + 1):
        for dj in range(-pad, pad + 1):
            if di == 0 and dj == 0:
                continue
            view = ap[pad + di : pad + di + arr.shape[0],
                      pad + dj : pad + dj + arr.shape[1]]
            max_nb = np.maximum(max_nb, view)
    return arr > max_nb


def cfar2d_ca(
    rd_db,
    train=(10, 8),
    guard=(2, 2),
    pfa=1e-4,
    min_snr_db=8.0,
    notch_doppler_bins=2,
    apply_nms=True,
    max_peaks=60,
    return_stats=False,
):
    """
    2D CA-CFAR on RD map in dB.

    Parameters
    ----------
    rd_db : np.ndarray
        RD map in dB.
    train, guard : (int, int)
        Training and guard window sizes (rows, cols).
    pfa : float
        Target probability of false alarm.
    min_snr_db : float
        Minimum SNR (in dB) to keep a detection.
    notch_doppler_bins : int
        Notch around zero-Doppler to suppress ground clutter.
    apply_nms : bool
        Whether to run 2D NMS over detections.
    max_peaks : int or None
        Maximum number of peaks to keep globally.

    Returns
    -------
    det : np.ndarray bool
        Binary detection mask.
    """
    rd_lin = 10.0 ** (rd_db / 10.0)
    H, W = rd_lin.shape
    mid = H // 2

    # simple ground notch
    if notch_doppler_bins > 0:
        k = int(notch_doppler_bins)
        rd_lin[mid - k : mid + k + 1, :] = np.minimum(
            rd_lin[mid - k : mid + k + 1, :],
            np.percentile(rd_lin, 10),
        )

    Tr, Tc = train
    Gr, Gc = guard
    tot = _moving_sum_2d(rd_lin, Tr + Gr, Tc + Gc)
    gpl = _moving_sum_2d(rd_lin, Gr, Gc)
    train_sum = tot - gpl

    n_train = (2 * (Tr + Gr) + 1) * (2 * (Tc + Gc) + 1) - (2 * Gr + 1) * (2 * Gc + 1)
    noise = np.maximum(train_sum / max(n_train, 1), 1e-12)
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    thresh = alpha * noise

    det = rd_lin > thresh
    snr_db = 10.0 * np.log10(np.maximum(rd_lin / noise, 1e-12))
    if min_snr_db and min_snr_db > 0:
        det &= snr_db >= min_snr_db

    if apply_nms:
        det &= nms2d(rd_lin, kernel=3)

    if max_peaks is not None and np.any(det):
        yy, xx = np.where(det)
        vals = rd_lin[yy, xx]
        if len(vals) > max_peaks:
            idx = np.argpartition(-vals, max_peaks - 1)[:max_peaks]
            keep = np.zeros_like(det, dtype=bool)
            keep[yy[idx], xx[idx]] = True
            det = keep

    if return_stats:
        return det, noise, snr_db
    return det


# ------------------------------------------------------------
# Unified SNR + Local Max + Clustering Peak Detector
# ------------------------------------------------------------
def detect_peaks_snr_localmax(
    z_complex,
    snr_thr_db=12.0,
    local_max_neigh=3,
    cluster=True,
    cluster_min_size=1,
):
    """
    Generic detector for FMCW RD or OTFS Delay–Doppler maps.

    Returns
    -------
    dets : list of dict
        Each dict has keys {'i','j','val_db','snr_db'}.
    z_db : np.ndarray
        dB magnitude map.
    noise_floor : float
        Estimated noise floor (20th percentile of z_db).
    """
    z_abs = np.abs(z_complex)
    z_db = 20 * np.log10(z_abs + 1e-12)

    # 1) Estimate noise floor
    noise_floor = np.percentile(z_db, 20)

    # 2) Threshold by SNR
    cand = z_db > (noise_floor + snr_thr_db)

    # 3) Local maxima (SciPy if available, else simple equality vs max_filter)
    if SCIPY and ndi is not None:
        max_filt = ndi.maximum_filter(z_db, size=local_max_neigh, mode="nearest")
        local_max = (z_db == max_filt)
    else:
        # fallback: "local max" == strictly larger than all neighbors handled
        max_filt = z_db.copy()
        local_max = np.ones_like(z_db, dtype=bool)  # keep all above SNR
    det_mask = cand & local_max

    # No clustering requested
    if not cluster:
        ys, xs = np.where(det_mask)
        dets = []
        for y, x in zip(ys, xs):
            val = z_db[y, x]
            snr = val - noise_floor
            dets.append(
                {"i": int(y), "j": int(x), "val_db": float(val), "snr_db": float(snr)}
            )
        return dets, z_db, noise_floor

    # 4) Clustering: keep strongest per blob
    if SCIPY and ndi is not None:
        labels, num = ndi.label(det_mask)
    else:
        # trivial fallback: each detection is its own cluster
        labels = np.zeros_like(det_mask, dtype=int)
        ys, xs = np.where(det_mask)
        for idx, (y, x) in enumerate(zip(ys, xs), start=1):
            labels[y, x] = idx
        num = len(ys)

    dets = []
    for lab in range(1, num + 1):
        mask = labels == lab
        if mask.sum() < cluster_min_size:
            continue
        vals = z_db[mask]
        idx = np.argmax(vals)
        y, x = np.argwhere(mask)[idx]
        val = z_db[y, x]
        snr = val - noise_floor
        dets.append(
            {"i": int(y), "j": int(x), "val_db": float(val), "snr_db": float(snr)}
        )

    return dets, z_db, noise_floor


# ---------------------------------------------------------------------
# VISUALIZATION (2D / 3D RD)
# ---------------------------------------------------------------------
def plot_rd(ax, rd_db, ra, va, title,
            dynamic_db=35, percentile_clip=99.2, cmap='magma'):
    top = np.percentile(rd_db, percentile_clip)
    vmin = top - dynamic_db
    im = ax.imshow(
        rd_db,
        extent=[ra[0], ra[-1], va[0], va[-1]],
        origin='lower', aspect='auto',
        cmap=cmap, vmin=vmin, vmax=top
    )
    ax.set_title(title)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    return im


def viz_rd_2d_compare(path, rd_f_db, rd_o_db, gts, sp: SystemParams, cfar_cfg=None):
    """
    2×1 panel: FMCW RD + OTFS DD.
    Returns:
        det_f_mask, ra_f, va_f, noise_f, snr_f  for FMCW.
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    pos = np.array([0, 0, sp.H])
    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    if cfar_cfg is None:
        cfar_cfg = dict(
            train=(10, 8), guard=(2, 2),
            pfa=1e-4, min_snr_db=8.0,
            notch_doppler_bins=2,
            apply_nms=True, max_peaks=60
        )

    # FMCW panel + CFAR
    im = plot_rd(
        ax[0], rd_f_db, ra_f, va_f,
        "FMCW Range–Doppler",
        dynamic_db=35, percentile_clip=99.2, cmap='magma'
    )
    plt.colorbar(im, ax=ax[0], label='dB')

    det_f_mask, noise_f, snr_f = cfar2d_ca(
        rd_f_db, **cfar_cfg, return_stats=True
    )
    fy, fx = np.where(det_f_mask)
    if fy.size:
        ax[0].scatter(
            ra_f[fx], va_f[fy],
            s=60, facecolors='none', edgecolors='cyan',
            linewidths=1.8, label='CFAR'
        )

    # OTFS panel (no CFAR here by default)
    im2 = plot_rd(
        ax[1], rd_o_db, ra_o, va_o,
        "OTFS Delay–Doppler",
        dynamic_db=35, percentile_clip=99.2, cmap='magma'
    )
    plt.colorbar(im2, ax=ax[1], label='dB')

    # overlay GTs on both panels
    for i, (ra, va, rd) in enumerate(
        [(ra_f, va_f, rd_f_db), (ra_o, va_o, rd_o_db)]
    ):
        for gt in gts:
            P = np.array(gt['c']) - pos
            r = np.linalg.norm(P)
            v = np.dot(P / (np.linalg.norm(P) + 1e-9), gt['v'])
            if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
                ax[i].plot(
                    r, v, 'wx', ms=10, mew=2,
                    label='GT' if i == 0 else ""
                )
                ax[i].text(
                    r + 1, v + 0.3,
                    f"{r:.0f} m, {v:.1f} m/s",
                    color='white', fontsize=9, weight='bold'
                )

    for i in range(2):
        ax[i].grid(alpha=0.25, linestyle=':')
        ax[i].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()

    return det_f_mask, ra_f, va_f, noise_f, snr_f


def viz_rd_3d_compare(path, rd_f_db, rd_o_db, gts, sp: SystemParams):
    """
    3D surface comparison for FMCW vs OTFS (both already in dB).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(18, 8))
    pos = np.array([0, 0, sp.H])
    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    for i, (rd, ra, va, name) in enumerate(
        [(rd_f_db, ra_f, va_f, "FMCW"),
         (rd_o_db, ra_o, va_o, "OTFS")]
    ):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        R, V = np.meshgrid(ra, va)

        floor = np.percentile(rd, 99.5) - 40.0
        surf = np.maximum(rd, floor)

        ax.plot_surface(
            R, V, surf,
            cmap='viridis',
            rstride=2, cstride=2,
            linewidth=0, antialiased=True, alpha=0.85
        )

        # GT markers
        for gt in gts:
            P = np.array(gt['c']) - pos
            r = np.linalg.norm(P)
            v = np.dot(P / (np.linalg.norm(P) + 1e-9), gt['v'])
            if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
                ax.scatter(
                    [r], [v], [np.max(rd) + 5],
                    c='r', marker='x', s=120, linewidths=2, zorder=10
                )

        ax.set_title(f"{name} 3D (dB)")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_xlim(0, ra[-1])
        ax.set_ylim(va[0], va[-1])
        ax.view_init(45, -110)

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()


def viz_rd_3d_with_dets(
    path, rd_raw, ra, va, det_mask, gts, sp, title="RD with Detections & GT"
):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    rd = _db_scale(rd_raw)
    lo, hi = np.percentile(rd, [5, 99.7])
    R, V = np.meshgrid(ra, va)
    surf = np.clip(rd, lo, hi)

    ax.plot_surface(
        R,
        V,
        surf,
        cmap="viridis",
        rstride=2,
        cstride=2,
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )

    if det_mask is not None and det_mask.any():
        ys, xs = np.where(det_mask)
        ax.scatter(
            ra[xs],
            va[ys],
            surf[ys, xs],
            s=18,
            c="cyan",
            depthshade=False,
            label="Detections",
        )

    pos = np.array([0, 0, sp.H])
    for gt in gts:
        P = np.array(gt["c"]) - pos
        r = np.linalg.norm(P)
        v = np.dot(P / r, gt["v"])
        if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
            ax.scatter([r], [v], [hi], c="r", marker="x", s=100, label="GT")

    ax.set_title(title + " (dB)")
    ax.set_xlabel("Range/Delay (m)")
    ax.set_ylabel("Doppler (m/s)")
    ax.set_xlim(0, ra[-1])
    ax.set_ylim(va[0], va[-1])
    ax.set_zlim(lo, hi)
    ax.view_init(35, -120)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()




# ---------------------------------------------------------------------
# DETECTIONS / GT METRICS
# ---------------------------------------------------------------------
def extract_detections(rd_db, det_mask, ra, va, noise_db=None, snr_db=None):
    yy, xx = np.where(det_mask)
    dets = []
    for y, x in zip(yy, xx):
        det = {
            "r": float(ra[x]),
            "v": float(va[y]),
            "mag_db": float(rd_db[y, x]),
        }
        if snr_db is not None:
            det["snr_db"] = float(snr_db[y, x])
        if noise_db is not None:
            det["noise_db"] = float(noise_db[y, x])
        dets.append(det)
    return dets


def _gt_rv_az(gts, sp: SystemParams):
    """
    Map GT cubes to (range, radial velocity, azimuth, size).
    """
    pos = np.array([0.0, 0.0, sp.H])
    out = []
    for gt in gts:
        c = np.array(gt["c"])
        v = np.array(gt["v"])
        s = np.array(gt["s"])
        d = c - pos
        r = np.linalg.norm(d)
        if r < 1e-6:
            continue
        u = d / r
        vr = float(np.dot(u, v))
        az = float(np.arctan2(c[1], c[0]))  # ground-plane azimuth
        out.append({"c": c, "r": float(r), "v": vr, "az": az, "s": s})
    return out


def _match_dets_to_gts(dets, gtinfo, w_r=1.0, w_v=0.5):
    """
    Simple nearest-neighbor matching in (r,v) with one-to-one constraint.

    Returns
    -------
    pairs : list of (det, gt_index, cost)
    unpaired : list of det dicts
    """
    used = set()
    pairs = []
    unpaired = []

    for d in dets:
        best_i = None
        best_cost = 1e12
        for gi, g in enumerate(gtinfo):
            cost = w_r * abs(d["r"] - g["r"]) + w_v * abs(d["v"] - g["v"])
            if cost < best_cost:
                best_cost = cost
                best_i = gi

        if best_i is None:
            unpaired.append(d)
            continue

        if best_i not in used:
            used.add(best_i)
            pairs.append((d, best_i, best_cost))
        else:
            unpaired.append(d)

    return pairs, unpaired


def _compute_metrics_from_pairs(pairs, gtinfo, sp):
    """
    For matched pairs only, collect |Δr| and |Δv|.
    """
    TP = len(pairs)
    FP = 0
    er_r = []
    er_v = []

    for det, gi, _ in pairs:
        g = gtinfo[gi]
        er_r.append(abs(det["r"] - g["r"]))
        er_v.append(abs(det["v"] - g["v"]))

    metrics = dict(TP=TP, FP=FP, er_r=er_r, er_v=er_v)
    return metrics, er_r, er_v

# ---------------------------------------------------------------------
# SCENE / BEV VISUALIZATION + RADAR METRICS HELPERS
# ---------------------------------------------------------------------
def _draw_box_xy(ax, c, s, **kwargs):
    """
    Draw a top-down rectangle for a 3D box (center c, size s) on the XY-plane.
    """
    cx, cy, cz = c
    sx, sy, sz = s
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    xs = [x0, x1, x1, x0, x0]
    ys = [y0, y0, y1, y1, y0]
    ax.plot(xs, ys, **kwargs)


def viz_bev_scene(prefix, pts, gts, sp: SystemParams):
    """
    Visualize a single scene in BEV:
      - Top-down XY scatter of raycast points (clamped by sp.bev_r_max)
      - GT boxes overlayed.

    Parameters
    ----------
    prefix : str or Path
        Output file prefix; will write prefix + "_bev.png".
    pts : torch.Tensor or np.ndarray, shape (N,3)
        Raycast hit points.
    gts : list of dict
        Each GT has keys {'c','s','v'} as in simulation.
    sp : SystemParams
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    pts_np = pts.detach().cpu().numpy() if isinstance(pts, torch.Tensor) else np.asarray(pts)
    if pts_np.size == 0:
        print("[BEV] No points to visualize.")
        return

    xy = pts_np[:, :2]
    r = np.linalg.norm(xy, axis=1)
    mask = r <= sp.bev_r_max
    xy = xy[mask]

    fig, ax = plt.subplots(figsize=(6, 6))
    if len(xy) > 0:
        ax.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.3, label="Raycast hits")

    for gt in gts:
        _draw_box_xy(ax, gt["c"], gt["s"], color="r", linewidth=2)
        ax.scatter(gt["c"][0], gt["c"][1], c="r", marker="x", s=50)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-sp.bev_r_max, sp.bev_r_max)
    ax.set_ylim(0, sp.bev_r_max)  # forward half-plane; adjust if you want full 360°
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Scene BEV (top-down)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")

    out_path = prefix.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"[BEV] Scene BEV saved to {out_path}")

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import Rectangle

# ---- BEV helper structures ----

def bev_gt_info(gts, sp: SystemParams):
    """GT info for BEV metrics: adds range r, azimuth, and size."""
    pos = np.array([0.0, 0.0, sp.H])
    out = []
    for gt in gts:
        c = np.array(gt['c'])
        v = np.array(gt['v'])
        s = np.array(gt['s'])
        d = c - pos
        r = np.linalg.norm(d)
        if r < 1e-6:
            continue
        u = d / r
        vr = float(np.dot(u, v))
        az = float(np.arctan2(c[1], c[0]))
        out.append({'c': c, 'r': float(r), 'v': vr, 'az': az, 's': s})
    return out


def bev_match_dets_to_gts(dets, gtinfo, w_r=1.0, w_v=0.5):
    """
    Simple nearest-neighbor matching in (r,v); returns pairs (det, gi, cost).
    (One detection per GT; extra detections will be unmatched.)
    """
    used = set()
    pairs = []
    for d in dets:
        best_g = None
        best_cost = 1e9
        for gi, g in enumerate(gtinfo):
            if gi in used:
                continue
            cost = w_r * abs(d['r'] - g['r']) + w_v * abs(d['v'] - g['v'])
            if cost < best_cost:
                best_cost = cost
                best_g = gi
        if best_g is not None:
            used.add(best_g)
            pairs.append((d, best_g, best_cost))
    return pairs


def _bev_inside_cube_xy(x, y, g):
    cx, cy = g['c'][0], g['c'][1]
    sx, sy = g['s'][0], g['s'][1]
    return (cx - sx/2.0 <= x <= cx + sx/2.0) and (cy - sy/2.0 <= y <= cy + sy/2.0)


def _bev_project_det_xy(det, g):
    az = g['az']
    x = det['r'] * np.cos(az)
    y = det['r'] * np.sin(az)
    return x, y


def bev_metrics_from_pairs(pairs, gtinfo, sp: SystemParams):
    """
    BEV metrics:
      - TP: projected detections that land inside the GT XY footprint
      - FP: projected detections that miss (or out of BEV range)
      - Precision / Recall / F1
    """
    TP, FP = 0, 0
    per_gt_tp = {i: 0 for i in range(len(gtinfo))}
    tp_pts, fp_pts = [], []

    for det, gi, _ in pairs:
        g = gtinfo[gi]
        x, y = _bev_project_det_xy(det, g)

        if not (0 <= x <= sp.bev_r_max and
                -sp.bev_r_max/2 <= y <= sp.bev_r_max/2):
            FP += 1
            fp_pts.append((x, y, gi))
            continue

        if _bev_inside_cube_xy(x, y, g):
            TP += 1
            per_gt_tp[gi] += 1
            tp_pts.append((x, y, gi))
        else:
            FP += 1
            fp_pts.append((x, y, gi))

    detected_gts = sum(1 for _, v in per_gt_tp.items() if v > 0)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = detected_gts / max(1, len(gtinfo))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = dict(
        TP=TP, FP=FP,
        detected_gts=detected_gts,
        total_gts=len(gtinfo),
        precision=precision,
        recall=recall,
        f1=f1,
    )
    return metrics, tp_pts, fp_pts


def viz_bev_scene(path_prefix, pts, gts, sp: SystemParams):
    """
    3D BEV-style scene: radar, raycast points, GT boxes, clamped to 0–bev_r_max.
    Saves `${path_prefix}_bev_scene.png`.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    radar = np.array([0, 0, sp.H])
    ax.plot([radar[0]], [radar[1]], [radar[2]], 'ko', ms=8, label='Radar')

    if len(pts) > 0:
        p = pts.detach().cpu().numpy()[::10]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                   s=0.5, c=p[:, 2], alpha=0.3, cmap='viridis')

    for gt in gts:
        c = np.array(gt['c'])
        s = np.array(gt['s'])
        dx, dy, dz = s / 2
        corners = np.array([
            [c[0]+i*dx, c[1]+j*dy, c[2]+k*dz]
            for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]
        ])
        edges = [
            [corners[0], corners[1]], [corners[0], corners[2]],
            [corners[0], corners[4]], [corners[7], corners[6]],
            [corners[7], corners[5]], [corners[7], corners[3]],
            [corners[2], corners[6]], [corners[2], corners[3]],
            [corners[1], corners[5]], [corners[1], corners[3]],
            [corners[4], corners[5]], [corners[4], corners[6]],
        ]
        ax.add_collection3d(Line3DCollection(edges, colors='r', lw=2))

    ax.set_xlim(0, sp.bev_r_max)
    ax.set_ylim(-sp.bev_r_max/2, sp.bev_r_max/2)
    ax.set_zlim(0, 15)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title("3D Scene (Raycast, 0–50 m)")
    ax.view_init(30, -110)

    plt.tight_layout()
    plt.savefig(f"{path_prefix}_bev_scene.png", dpi=180, bbox_inches='tight')
    plt.close()


def _draw_bev_panel(ax, dets, gts, sp: SystemParams,
                    title="BEV", ring_step=10):
    gtinfo = bev_gt_info(gts, sp)
    pairs = bev_match_dets_to_gts(dets, gtinfo, w_r=1.0, w_v=0.5)
    metrics, tp_pts, fp_pts = bev_metrics_from_pairs(pairs, gtinfo, sp)

    ax.scatter([0], [0], marker='*', s=140, c='black', label='Radar')

    for rr in np.arange(ring_step, sp.bev_r_max + 1e-6, ring_step):
        circ = plt.Circle((0, 0), rr, color='gray',
                          fill=False, alpha=0.22, lw=0.8)
        ax.add_artist(circ)

    # GT rectangles
    for g in gtinfo:
        cx, cy = g['c'][0], g['c'][1]
        sx, sy = g['s'][0], g['s'][1]
        rect = Rectangle(
            (cx - sx/2, cy - sy/2),
            sx, sy,
            linewidth=1.8,
            edgecolor='r', facecolor='none', alpha=0.9,
            label='GT' if 'GT' not in ax.get_legend_handles_labels()[1] else None
        )
        ax.add_patch(rect)
        ax.plot([cx], [cy], 'rx', ms=6)

    if tp_pts:
        ax.scatter(
            [p[0] for p in tp_pts], [p[1] for p in tp_pts],
            s=64, facecolors='none', edgecolors='lime',
            linewidths=2.0, label='TP (in cube)'
        )
    if fp_pts:
        ax.scatter(
            [p[0] for p in fp_pts], [p[1] for p in fp_pts],
            s=64, facecolors='none', edgecolors='orange',
            linewidths=2.0, label='FP (out of cube)'
        )

    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, sp.bev_r_max)
    ax.set_ylim(-sp.bev_r_max/2, sp.bev_r_max/2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(loc='upper left')

    txt = (
        f"TP: {metrics['TP']}   FP: {metrics['FP']}\n"
        f"Precision: {metrics['precision']:.2f}\n"
        f"Recall: {metrics['recall']:.2f} "
        f"({metrics['detected_gts']}/{metrics['total_gts']})\n"
        f"F1: {metrics['f1']:.2f}"
    )
    ax.text(
        0.98, 0.02, txt,
        transform=ax.transAxes, va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  alpha=0.85, linewidth=0.8)
    )

    return metrics

import numpy as np
import matplotlib.pyplot as plt

def _gt_range_list(gts, sp):
    """Return list of ground-truth ranges (meters) from radar origin."""
    pos = np.array([0.0, 0.0, sp.H])
    r_list = []
    for gt in gts:
        c = np.array(gt["c"])
        d = c - pos
        r_list.append(float(np.linalg.norm(d)))
    return np.array(r_list, dtype=float)


def _range_only_match_metrics(dets, gts, sp, thr_r=3.0):
    """
    Very simple 1D (range-only) matching used for the old BEV figure:

      - Each detection has only a range r.
      - Match to the nearest GT in range within thr_r (meters); one-to-one.
      - Everything else is FP; any GT not matched is FN.
    """
    if len(gts) == 0:
        return dict(TP=0, FP=len(dets), FN=0,
                    precision=0.0, recall=0.0, f1=0.0), [], list(range(len(dets)))

    r_gt = _gt_range_list(gts, sp)    # (G,)
    used = np.zeros(len(gts), dtype=bool)

    tp_idx = []
    fp_idx = []

    for i, d in enumerate(dets):
        r_det = float(d["r"])
        j = int(np.argmin(np.abs(r_gt - r_det)))
        if (abs(r_gt[j] - r_det) <= thr_r) and (not used[j]):
            tp_idx.append(i)
            used[j] = True
        else:
            fp_idx.append(i)

    TP = len(tp_idx)
    FP = len(fp_idx)
    FN = len(gts) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    metrics = dict(TP=TP, FP=FP, FN=FN,
                   precision=precision, recall=recall, f1=f1)
    return metrics, tp_idx, fp_idx

def viz_scene_bev_compare(path, dets_fmcw, dets_otfs, gts, sp,
                          range_max=None, match_thr_m=3.0):
    """
    Draw scene_bev_compare in the *old* style:

      - Two panels: FMCW detections (left), OTFS detections (right)
      - X axis: range (m), Y axis: a small strip around 0 (essentially 1D)
      - Black 'x' = GT in range, y=0
      - Green circles = TP detections (in range window of some GT)
      - Orange circles = FP detections
      - Titles show P/R/F1:  e.g. "FMCW detections  P=0.03, R=1.00, F1=0.06"

    Returns:
        metrics_f, metrics_o  (dict) for FMCW / OTFS respectively.
    """
    if range_max is None:
        range_max = getattr(sp, "bev_r_max", 50.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    panels = [("FMCW", dets_fmcw, axes[0]),
              ("OTFS", dets_otfs, axes[1])]

    metrics_out = []

    # Pre-compute GT ranges for plotting
    r_gt = _gt_range_list(gts, sp)
    y_gt = np.zeros_like(r_gt)

    for name, dets, ax in panels:
        # 1D matching metrics (range only, like old code)
        metrics, tp_idx, fp_idx = _range_only_match_metrics(
            dets, gts, sp, thr_r=match_thr_m
        )
        metrics_out.append(metrics)

        # --- Plot GT as black 'x' -------------------------------------
        if len(r_gt) > 0:
            ax.scatter(r_gt, y_gt, marker="x", c="k", label="GT")

        # --- Plot detections: TP vs FP --------------------------------
        if tp_idx:
            r_tp = [dets[i]["r"] for i in tp_idx]
            ax.scatter(r_tp,
                       np.zeros(len(r_tp)),
                       s=40,
                       facecolors="none",
                       edgecolors="green",
                       label="TP")
        if fp_idx:
            r_fp = [dets[i]["r"] for i in fp_idx]
            ax.scatter(r_fp,
                       np.zeros(len(r_fp)),
                       s=40,
                       facecolors="none",
                       edgecolors="orange",
                       label="FP")

        # --- Axes & title (metrics in title) --------------------------
        ax.set_xlim(0.0, range_max)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.25, linestyle=":")

        P = metrics["precision"]
        R = metrics["recall"]
        F1 = metrics["f1"]
        ax.set_title(
            f"{name} detections\n"
            f"P={P:.2f}, R={R:.2f}, F1={F1:.2f}"
        )
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

    metrics_f, metrics_o = metrics_out
    return metrics_f, metrics_o

# def viz_scene_bev_compare(path, dets_fmcw, dets_otfs, gts, sp: SystemParams):
#     """
#     2×1 BEV: FMCW vs OTFS detections vs GT, with BEV metrics per panel.
#     """
#     fig, axes = plt.subplots(1, 2, figsize=(16, 7))

#     _draw_bev_panel(axes[0], dets_fmcw, gts, sp, title="BEV (FMCW)")
#     _draw_bev_panel(axes[1], dets_otfs, gts, sp, title="BEV (OTFS)")

#     plt.tight_layout()
#     plt.savefig(path, dpi=180, bbox_inches='tight')
#     plt.close()

def _metrics_from_matches(pairs, unpaired_dets, gtinfo, sp=None):
    """
    Given matches (pairs) and unmatched detections, compute radar metrics:
      - TP, FP, FN, precision, recall, F1
      - range / velocity errors (er_r, er_v from _compute_metrics_from_pairs)
    """
    metrics_base, er_r, er_v = _compute_metrics_from_pairs(pairs, gtinfo, sp)
    TP = metrics_base["TP"]
    FP = metrics_base["FP"] + len(unpaired_dets)
    FN = max(len(gtinfo) - TP, 0)

    prec, rec = _precision_recall(TP, FP, FN)
    f1 = _safe_f1(TP, FP, FN)

    metrics_base.update(
        dict(
            FP=FP,
            FN=FN,
            precision=prec,
            recall=rec,
            f1=f1,
            er_r=er_r,
            er_v=er_v,
        )
    )
    return metrics_base


def compute_radar_metrics(dets, gts, sp: SystemParams, w_r=1.0, w_v=0.5):
    """
    Convenience wrapper to compute radar detection metrics for one detector.

    Parameters
    ----------
    dets : list of dict
        Output from extract_detections(...) or rd_dl_infer_to_points(...).
    gts : list of dict
        GT cube definitions.
    sp : SystemParams
    w_r, w_v : float
        Weights for nearest-neighbor matching in (range, velocity).

    Returns
    -------
    metrics : dict
        Contains TP, FP, FN, precision, recall, f1 and error lists er_r, er_v.
    """
    gtinfo = _gt_rv_az(gts, sp)
    pairs, unpaired = _match_dets_to_gts(dets, gtinfo, w_r=w_r, w_v=w_v)
    metrics = _metrics_from_matches(pairs, unpaired, gtinfo, sp)
    return metrics


def viz_scene_bev_compare(path, dets_f, dets_o, gts, sp: SystemParams,
                          w_r=1.0, w_v=0.5):
    """
    Compare FMCW vs OTFS detections in a BEV-style view and show metrics.

    We:
      - Match detections to GT in (range, velocity).
      - Approximate BEV position of each detection using the matched GT azimuth.
      - Plot:
          left panel  : FMCW (CFAR or DL) detections vs GT
          right panel : OTFS detections vs GT
      - Title bar of each panel includes precision / recall / F1.

    Parameters
    ----------
    path : str or Path
        Output path for the figure (.pdf/.png).
    dets_f, dets_o : list of dict
        Detections for FMCW / OTFS (with 'r' & 'v').
    gts : list of dict
    sp : SystemParams
    """
    import matplotlib.pyplot as plt

    path = Path(path)
    gtinfo = _gt_rv_az(gts, sp)

    # Match & metrics for FMCW
    pairs_f, unp_f = _match_dets_to_gts(dets_f, gtinfo, w_r=w_r, w_v=w_v)
    metrics_f = _metrics_from_matches(pairs_f, unp_f, gtinfo, sp)

    # Match & metrics for OTFS
    pairs_o, unp_o = _match_dets_to_gts(dets_o, gtinfo, w_r=w_r, w_v=w_v)
    metrics_o = _metrics_from_matches(pairs_o, unp_o, gtinfo, sp)

    def dets_to_xy(pairs, unpaired, gtinfo):
        xs_tp, ys_tp = [], []
        xs_fp, ys_fp = [], []

        # True positives: use GT azimuth for BEV projection
        for det, gi, _ in pairs:
            g = gtinfo[gi]
            az = g["az"]
            r = det["r"]
            x = r * np.cos(az)
            y = r * np.sin(az)
            xs_tp.append(x)
            ys_tp.append(y)

        # False positives: put them on x-axis (azimuth=0) for visualization only
        for det in unpaired:
            r = det["r"]
            x = r
            y = 0.0
            xs_fp.append(x)
            ys_fp.append(y)

        return np.array(xs_tp), np.array(ys_tp), np.array(xs_fp), np.array(ys_fp)

    xs_tp_f, ys_tp_f, xs_fp_f, ys_fp_f = dets_to_xy(pairs_f, unp_f, gtinfo)
    xs_tp_o, ys_tp_o, xs_fp_o, ys_fp_o = dets_to_xy(pairs_o, unp_o, gtinfo)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax in axs:
        for gt in gts:
            _draw_box_xy(ax, gt["c"], gt["s"], color="k", linewidth=1.5)
            ax.scatter(gt["c"][0], gt["c"][1], c="k", marker="x", s=40, label="GT")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-sp.bev_r_max, sp.bev_r_max)
        ax.set_ylim(0, sp.bev_r_max)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linestyle=":", alpha=0.4)

    # FMCW panel
    ax = axs[0]
    if xs_tp_f.size > 0:
        ax.scatter(xs_tp_f, ys_tp_f, c="g", s=30, label="TP")
    if xs_fp_f.size > 0:
        ax.scatter(xs_fp_f, ys_fp_f, c="r", marker="x", s=40, label="FP")
    ax.set_title(
        f"FMCW detections\nP={metrics_f['precision']:.2f}, "
        f"R={metrics_f['recall']:.2f}, F1={metrics_f['f1']:.2f}"
    )

    # OTFS panel
    ax = axs[1]
    if xs_tp_o.size > 0:
        ax.scatter(xs_tp_o, ys_tp_o, c="g", s=30, label="TP")
    if xs_fp_o.size > 0:
        ax.scatter(xs_fp_o, ys_fp_o, c="r", marker="x", s=40, label="FP")
    ax.set_title(
        f"OTFS detections\nP={metrics_o['precision']:.2f}, "
        f"R={metrics_o['recall']:.2f}, F1={metrics_o['f1']:.2f}"
    )

    # Keep legend compact: only use legend from first panel
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"[BEV] Scene BEV comparison saved to {path}")

    return metrics_f, metrics_o

# ---------------------------------------------------------------------
# RADAR DL BASICS (heatmaps, UNet)
# ---------------------------------------------------------------------
def _rd_normalize(rd_db, top_p=99.5, dyn_db=40.0):
    top = np.percentile(rd_db, top_p)
    rd = np.clip(rd_db, top - dyn_db, top)
    rd = (rd - (top - dyn_db)) / dyn_db
    return rd.astype(np.float32)


def _heatmap_from_gts(shape, ra, va, gts, sp, sigma_pix=(2.0, 2.0)):
    H, W = shape
    pos = np.array([0, 0, sp.H])
    yy, xx = np.mgrid[0:H, 0:W]
    hm = np.zeros((H, W), np.float32)

    for gt in gts:
        P = np.array(gt["c"]) - pos
        r = np.linalg.norm(P)
        v = np.dot(P / (np.linalg.norm(P) + 1e-9), gt["v"])
        if not (0 <= r <= ra[-1] and va[0] <= v <= va[-1]):
            continue
        ix = np.searchsorted(ra, r)
        ix = np.clip(ix, 0, W - 1)
        iy = np.searchsorted(va, v)
        iy = np.clip(iy, 0, H - 1)
        sx, sy = sigma_pix[1], sigma_pix[0]
        g = np.exp(-((xx - ix) ** 2 / (2 * sx ** 2) +
                     (yy - iy) ** 2 / (2 * sy ** 2)))
        hm = np.maximum(hm, g)
    return hm


class UNetLite(nn.Module):
    """
    Lightweight UNet-style network for RD → heatmap.
    """

    def __init__(self, in_ch=1, ch=32):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
        )
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(
            nn.Conv2d(ch, 2 * ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * ch, 2 * ch, 3, padding=1),
            nn.ReLU(),
        )
        self.p2 = nn.MaxPool2d(2)
        self.b = nn.Sequential(
            nn.Conv2d(2 * ch, 4 * ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4 * ch, 4 * ch, 3, padding=1),
            nn.ReLU(),
        )
        self.u2 = nn.ConvTranspose2d(4 * ch, 2 * ch, 2, stride=2)
        self.d2 = nn.Sequential(
            nn.Conv2d(4 * ch, 2 * ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * ch, 2 * ch, 3, padding=1),
            nn.ReLU(),
        )
        self.u1 = nn.ConvTranspose2d(2 * ch, ch, 2, stride=2)
        self.d1 = nn.Sequential(
            nn.Conv2d(2 * ch, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
        )
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        b = self.b(self.p2(e2))
        d2 = self.d2(torch.cat([self.u2(b), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


def focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal BCE loss for dense heatmap regression (logits vs [0,1] targets).
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    loss = (w * (1 - pt).pow(gamma) * bce).mean()
    return loss


# ---------------------------------------------------------------------
# RADAR DL INFERENCE → PEAKS
# ---------------------------------------------------------------------
@torch.no_grad()
def rd_dl_infer_to_points(logits, ra, va, thr=0.35, max_peaks=64):
    """
    Convert UNet output logits to a list of range/velocity detections.

    Returns
    -------
    dets : list of dict
        Each dict has keys {'r','v','score'}.
    """
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask = prob > thr
    if not mask.any():
        return []
    if SCIPY and ndi is not None:
        mxf = ndi.maximum_filter(prob, size=3)
        peaks = (prob == mxf) & mask
    else:
        mxf = prob.copy()
        peaks = mask
    yy, xx = np.where(peaks)
    if len(yy) > max_peaks:
        vals = prob[yy, xx]
        idx = np.argpartition(-vals, max_peaks - 1)[:max_peaks]
        yy, xx = yy[idx], xx[idx]
    dets = []
    for y, x in zip(yy, xx):
        dy, dx = _subpixel_quadratic(prob, int(y), int(x))
        r, v = _rv_from_idx_with_subpix(int(y), int(x), dy, dx, ra, va)
        dets.append({"r": float(r), "v": float(v), "score": float(prob[y, x])})
    return dets


# ---------------------------------------------------------------------
# COMMUNICATIONS (QPSK / OFDM / OTFS + BER)
# ---------------------------------------------------------------------
def _rand_bits(n, rng):
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def _qpsk_gray_mod(bits):
    """
    Gray-mapped QPSK modulator.

    bits : (..., 2) uint8
    """
    b0 = bits[..., 0]
    b1 = bits[..., 1]
    I = np.where(
        (b0 == 0) & (b1 == 0),
        1.0,
        np.where(
            (b0 == 0) & (b1 == 1),
            -1.0,
            np.where((b0 == 1) & (b1 == 1), -1.0, 1.0),
        ),
    )
    Q = np.where(
        (b0 == 0) & (b1 == 0),
        1.0,
        np.where(
            (b0 == 0) & (b1 == 1),
            1.0,
            np.where((b0 == 1) & (b1 == 1), -1.0, -1.0),
        ),
    )
    s = (I + 1j * Q) / np.sqrt(2.0)
    return s


def _qpsk_gray_demod(symbols):
    """
    Hard-decision Gray-mapped QPSK demodulator.
    """
    I = np.real(symbols)
    Q = np.imag(symbols)
    b0 = (Q < 0).astype(np.uint8)
    b1 = (I < 0).astype(np.uint8)
    return np.stack([b0, b1], axis=-1)


def _awgn(x, ebn0_db, bits_per_sym, cp_ratio=0.0, rng=None):
    """
    AWGN channel, parameterized by Eb/N0.
    """
    if rng is None:
        rng = np.random.default_rng()
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    r_eff = bits_per_sym * (1.0 / (1.0 + cp_ratio))
    Es = 1.0
    Eb = Es / r_eff
    N0 = Eb / ebn0
    sigma2 = N0
    n = (
        rng.normal(scale=np.sqrt(sigma2 / 2), size=x.shape)
        + 1j * rng.normal(scale=np.sqrt(sigma2 / 2), size=x.shape)
    )
    return x + n


# ---------- OFDM ----------
def ofdm_mod(bits, Nfft=256, cp_len=32, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bits = bits.reshape(-1, 2)
    nsym = bits.shape[0] // Nfft
    bits = bits[: nsym * Nfft].reshape(nsym, Nfft, 2)
    syms = _qpsk_gray_mod(bits)
    x = np.fft.ifft(syms, n=Nfft, axis=1, norm="ortho")
    if cp_len > 0:
        cp = x[:, -cp_len:]
        x_cp = np.concatenate([cp, x], axis=1)
    else:
        x_cp = x
    return x_cp, syms


def ofdm_demod(rx, Nfft=256, cp_len=32):
    if cp_len > 0:
        rx = rx[:, cp_len : cp_len + Nfft]
    Sy = np.fft.fft(rx, n=Nfft, axis=1, norm="ortho")
    return Sy


def ofdm_tx_rx_ber(ebn0_db, Nfft=256, cp_len=32, n_ofdm_sym=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bits_per_sym = 2
    nbits = n_ofdm_sym * Nfft * bits_per_sym
    bits = _rand_bits(nbits, rng)
    tx, _ = ofdm_mod(bits, Nfft=Nfft, cp_len=cp_len, rng=rng)
    cp_ratio = cp_len / Nfft
    rx = _awgn(tx, ebn0_db, bits_per_sym=bits_per_sym, cp_ratio=cp_ratio, rng=rng)
    Sy = ofdm_demod(rx, Nfft=Nfft, cp_len=cp_len)
    hard_bits = _qpsk_gray_demod(Sy.reshape(-1))
    hard_bits = hard_bits.reshape(-1, 2)
    bits_hat = hard_bits[: len(bits)].reshape(-1)
    ber = np.mean(bits != bits_hat)
    return ber


# ---------- OTFS (simple) ----------
def otfs_mod(bits, M=64, N=256, cp_len=32, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bits = bits.reshape(M * N, 2)[: M * N].reshape(M, N, 2)
    X_dd = _qpsk_gray_mod(bits)

    X_tf = np.fft.ifft(
        np.fft.fft(X_dd, n=N, axis=1, norm="ortho"), n=M, axis=0, norm="ortho"
    )
    tx = np.fft.ifft(X_tf, n=N, axis=1, norm="ortho")
    if cp_len > 0:
        cp = tx[:, -cp_len:]
        tx = np.concatenate([cp, tx], axis=1)
    return tx, X_dd


def otfs_demod(rx, M=64, N=256, cp_len=32):
    if cp_len > 0:
        rx = rx[:, cp_len : cp_len + N]
    Y_tf = np.fft.fft(rx, n=N, axis=1, norm="ortho")
    Y_dd = np.fft.ifft(
        np.fft.fft(Y_tf, n=M, axis=0, norm="ortho"), n=N, axis=1, norm="ortho"
    )
    return Y_dd


def otfs_tx_rx_ber(ebn0_db, M=64, N=256, cp_len=32, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bits_per_sym = 2
    nbits = M * N * bits_per_sym
    bits = _rand_bits(nbits, rng)
    tx, _ = otfs_mod(bits, M=M, N=N, cp_len=cp_len, rng=rng)
    cp_ratio = cp_len / N
    rx = _awgn(tx, ebn0_db, bits_per_sym=bits_per_sym, cp_ratio=cp_ratio, rng=rng)
    Ydd = otfs_demod(rx, M=M, N=N, cp_len=cp_len)
    hard_bits = _qpsk_gray_demod(Ydd.reshape(-1))
    hard_bits = hard_bits.reshape(-1, 2)
    bits_hat = hard_bits[: len(bits)].reshape(-1)
    ber = np.mean(bits != bits_hat)
    return ber


# ---------- BER sweep ----------
def run_ber_sweep_and_plot(
    path_png,
    ebn0_db_list=np.arange(0, 21, 2),
    ofdm_cfg=dict(Nfft=256, cp_len=32, n_ofdm_sym=400),
    otfs_cfg=dict(M=64, N=256, cp_len=32),
    rng_seed=1234,
):
    """
    Generate BER vs Eb/N0 curves for OFDM & OTFS baselines and plot them.

    Returns
    -------
    ebn0_db_list, ber_ofdm, ber_otfs, ber_theory_qpsk
    """
    rng = np.random.default_rng(rng_seed)
    ber_ofdm, ber_otfs = [], []

    for eb in ebn0_db_list:
        ber_ofdm.append(ofdm_tx_rx_ber(eb, **ofdm_cfg, rng=rng))
        ber_otfs.append(otfs_tx_rx_ber(eb, **otfs_cfg, rng=rng))

    ber_ofdm = np.array(ber_ofdm)
    ber_otfs = np.array(ber_otfs)

    ebn0_lin = 10.0 ** (np.array(ebn0_db_list) / 10.0)
    ber_theory_qpsk = np.array(
        [0.5 * math.erfc(math.sqrt(x)) for x in ebn0_lin], dtype=float
    )

    plt.figure(figsize=(7.8, 5.6))
    plt.semilogy(
        ebn0_db_list,
        ber_ofdm + 1e-12,
        marker="o",
        label="FMCW-Comm (OFDM, QPSK)",
    )
    plt.semilogy(
        ebn0_db_list,
        ber_otfs + 1e-12,
        marker="s",
        label="OTFS-Comm (QPSK)",
    )
    plt.semilogy(
        ebn0_db_list,
        ber_theory_qpsk + 1e-12,
        linestyle="--",
        label="Theory QPSK (AWGN)",
    )
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.title("BER vs Eb/N0: OFDM vs OTFS vs Theory")

    for i in range(0, len(ebn0_db_list), 2):
        x = ebn0_db_list[i]
        yo = ber_ofdm[i]
        yt = ber_otfs[i]
        plt.text(
            x,
            yo * 1.15 + 1e-14,
            f"{yo:.2e}",
            fontsize=8,
            ha="center",
            va="bottom",
        )
        plt.text(
            x,
            yt * 0.85 + 1e-14,
            f"{yt:.2e}",
            fontsize=8,
            ha="center",
            va="top",
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=170, bbox_inches="tight")
    plt.close()

    return ebn0_db_list, ber_ofdm, ber_otfs, ber_theory_qpsk


# ---------------------------------------------------------------------
# COMM METRICS + BER COMPARISON VISUALIZATION
# ---------------------------------------------------------------------
def _snr_at_target_ber(eb_axis, ber, target_ber=1e-3):
    """
    Return smallest Eb/N0 (dB) where BER <= target_ber. None if not reached.
    """
    for e, b in zip(eb_axis, ber):
        if b <= target_ber:
            return float(e)
    return None


def compute_comm_metrics(
    eb_axis,
    ber_base,
    ber_dl=None,
    target_ber=1e-3,
):
    """
    Compute simple communication metrics for one scheme (OFDM or OTFS).

    Parameters
    ----------
    eb_axis : array-like
        Eb/N0 values (dB).
    ber_base : array-like
        Baseline BER (e.g., hard QPSK).
    ber_dl : array-like or None
        DL demapper BER; if None we only report baseline metrics.
    target_ber : float
        Target BER threshold for "operating point".

    Returns
    -------
    metrics : dict
        Contains:
          - 'eb_axis'
          - 'ber_base'
          - 'ber_dl' (if provided)
          - 'snr_base_at_target'
          - 'snr_dl_at_target'
          - 'snr_gain_db'  (base - dl) if both exist.
    """
    eb_axis = np.asarray(eb_axis)
    ber_base = np.asarray(ber_base)

    snr_base = _snr_at_target_ber(eb_axis, ber_base, target_ber)
    snr_dl = None
    if ber_dl is not None:
        ber_dl = np.asarray(ber_dl)
        snr_dl = _snr_at_target_ber(eb_axis, ber_dl, target_ber)
    gain = None
    if snr_base is not None and snr_dl is not None:
        gain = snr_base - snr_dl

    return dict(
        eb_axis=eb_axis.tolist(),
        ber_base=ber_base.tolist(),
        ber_dl=None if ber_dl is None else ber_dl.tolist(),
        target_ber=float(target_ber),
        snr_base_at_target=snr_base,
        snr_dl_at_target=snr_dl,
        snr_gain_db=gain,
    )


def viz_ber_compare_with_dl(
    path,
    eb_axis,
    ber_ofdm_base,
    ber_otfs_base,
    ber_theory,
    ber_ofdm_dl=None,
    ber_otfs_dl=None,
    title="Comm BER: Baseline vs DL",
):
    """
    Visualization helper for BER curves: OFDM/OTFS baseline + optional DL curves.

    Parameters
    ----------
    path : str or Path
        Output path (.pdf/.png).
    eb_axis : array-like
    ber_ofdm_base, ber_otfs_base, ber_theory : array-like
    ber_ofdm_dl, ber_otfs_dl : array-like or None
        DL demapper BER curves (optional).
    """
    import matplotlib.pyplot as plt

    path = Path(path)
    eb_axis = np.asarray(eb_axis)
    ber_ofdm_base = np.asarray(ber_ofdm_base)
    ber_otfs_base = np.asarray(ber_otfs_base)
    ber_theory = np.asarray(ber_theory)

    plt.figure(figsize=(8, 6))
    plt.semilogy(
        eb_axis,
        ber_ofdm_base + 1e-12,
        "o-",
        label="OFDM baseline (hard QPSK)",
    )
    plt.semilogy(
        eb_axis,
        ber_otfs_base + 1e-12,
        "s-",
        label="OTFS baseline (hard QPSK)",
    )
    if ber_ofdm_dl is not None:
        ber_ofdm_dl = np.asarray(ber_ofdm_dl)
        plt.semilogy(
            eb_axis,
            ber_ofdm_dl + 1e-12,
            "o--",
            label="OFDM DL demapper",
        )
    if ber_otfs_dl is not None:
        ber_otfs_dl = np.asarray(ber_otfs_dl)
        plt.semilogy(
            eb_axis,
            ber_otfs_dl + 1e-12,
            "s--",
            label="OTFS DL demapper",
        )

    plt.semilogy(
        eb_axis,
        ber_theory + 1e-12,
        "k:",
        label="Theory QPSK (AWGN)",
    )
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"[COMM] BER comparison saved to {path}")

# ---------------------------------------------------------------------
# COMM DL: BATCH GEN + DEMAPPER MODEL + BER
# ---------------------------------------------------------------------
def _bits_to_qpsk_grid(bits, H, W):
    bits = bits.reshape(H * W, 2)
    syms = _qpsk_gray_mod(bits)
    return syms.reshape(H, W)


def _grid_feats(S, use_mag_phase=False):
    if use_mag_phase:
        mag = np.abs(S)
        ang = np.angle(S)
        x = np.stack([S.real, S.imag, mag, ang], axis=0).astype(np.float32)
    else:
        x = np.stack([S.real, S.imag], axis=0).astype(np.float32)
    return x


def comm_dl_gen_batch_OFDM(ebn0_db, batch=8, Nfft=256, cp_len=32, n_sym=8, rng=None):
    """
    Generate a batch of OFDM noisy grids for DL demapper training.
    """
    if rng is None:
        rng = np.random.default_rng()
    bits_per_sym = 2
    H, W = n_sym, Nfft
    x_list, y_list = [], []
    for _ in range(batch):
        bits = _rand_bits(H * W * bits_per_sym, rng)
        Xf = _bits_to_qpsk_grid(bits, H, W)
        tx = np.fft.ifft(Xf, n=W, axis=1, norm="ortho")
        if cp_len > 0:
            tx = np.concatenate([tx[:, -cp_len:], tx], axis=1)
        cp_ratio = cp_len / W
        rx = _awgn(tx, ebn0_db, bits_per_sym=2, cp_ratio=cp_ratio, rng=rng)
        if cp_len > 0:
            rx = rx[:, cp_len : cp_len + W]
        Yf = np.fft.fft(rx, n=W, axis=1, norm="ortho")
        x = _grid_feats(Yf, use_mag_phase=False)
        y = bits.reshape(H, W, 2).transpose(2, 0, 1)
        x_list.append(x)
        y_list.append(y.astype(np.float32))
    X = torch.from_numpy(np.stack(x_list))
    Y = torch.from_numpy(np.stack(y_list))
    return X, Y


def comm_dl_gen_batch_OTFS(ebn0_db, batch=8, M=64, N=256, cp_len=32, rng=None):
    """
    Generate a batch of OTFS delay–Doppler grids for DL demapper training.
    """
    if rng is None:
        rng = np.random.default_rng()
    bits_per_sym = 2
    x_list, y_list = [], []
    for _ in range(batch):
        bits = _rand_bits(M * N * bits_per_sym, rng)
        tx, _ = otfs_mod(bits, M=M, N=N, cp_len=cp_len, rng=rng)
        cp_ratio = cp_len / N
        rx = _awgn(tx, ebn0_db, bits_per_sym=2, cp_ratio=cp_ratio, rng=rng)
        Ydd = otfs_demod(rx, M=M, N=N, cp_len=cp_len)
        x = _grid_feats(Ydd, use_mag_phase=False)
        y = bits.reshape(M, N, 2).transpose(2, 0, 1)
        x_list.append(x)
        y_list.append(y.astype(np.float32))
    X = torch.from_numpy(np.stack(x_list))
    Y = torch.from_numpy(np.stack(y_list))
    return X, Y


class CommDemapperCNN(nn.Module):
    """
    Simple convolutional demapper:
    input  : (B, 2, H, W) (real & imag)
    output : (B, 2, H, W) (bit logits)
    """

    def __init__(self, in_ch=2, width=32, depth=3, out_ch=2):
        super().__init__()
        layers = [nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, out_ch, 1)

    def forward(self, x):
        return self.head(self.backbone(x))


def build_dd_channel_from_rays(
    pts: torch.Tensor,
    its: torch.Tensor,
    vels: torch.Tensor,
    sp: SystemParams,
    M: int,
    N: int,
):
    """
    Build a discrete delay–Doppler channel H_dd(l, k) from raycast returns.

    This is similar to the "toy OTFS radar" mapping, but interpreted as a
    *channel*:

        - Each ray / scatterer becomes one DD tap.
        - Delay index from range R.
        - Doppler index from radial velocity v_r.
        - Amplitude based on "intensity" and 1/R^2 spreading.

    Parameters
    ----------
    pts : (P, 3) torch.float32
        Hit positions from raycast_torch (world coordinates).

    its : (P,) torch.float32
        Intensities (e.g. lidar-like power). We assume higher = stronger tap.

    vels : (P, 3) torch.float32
        Velocities at the hit points.

    sp : SystemParams
        System parameters (fc, fs, lambda, M, N, etc).

    M, N : int
        Desired DD grid size (M Doppler bins × N delay bins) for the channel.

    Returns
    -------
    H_dd : (M, N) torch.complex64
        Complex delay–Doppler channel coefficients.
    """
    H = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)

    if pts.numel() == 0:
        return H

    # Shift to radar phase center at (0,0,H)
    P = pts - to_torch([0.0, 0.0, sp.H])   # (P,3)
    R = torch.norm(P, dim=1)               # (P,)

    # Ignore extremely near points (avoid self / numerical issues)
    mask = R > 0.1
    P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
    if P.numel() == 0:
        return H

    # 1 / R^2 spreading + intensity scaling:
    # treat "strong reflectors" (e.g. cube faces) more powerful than ground.
    # Here we assume 'its' already encodes this; we just normalize by R^2.
    amp = its / (R ** 2 + 1e-6)           # (P,) real amplitudes

    # Radial velocity v_r: projection of velocity onto line-of-sight
    vr = torch.sum(P / R.unsqueeze(1) * vels, dim=1)  # (P,)

    # Map physical range & radial velocity into DD indices.
    # These "resolutions" are the same as your earlier OTFS toy model.
    #
    #   k_res: nominal range step per delay bin (meters).
    #   l_res: nominal velocity step per Doppler bin (m/s).
    #
    k_res = C0 / (2.0 * sp.fs)                       # ~ range resolution (m)
    l_res = (sp.lambda_m / 2.0) * (sp.fs / (M * N))  # ~ velocity resolution (m/s)

    # Delay bin index ("k" in many OTFS papers) from range R
    k = torch.clamp((R / k_res).long(), 0, N - 1)  # (P,) in [0, N-1]

    # Doppler bin index ("l") from radial velocity vr
    l = torch.clamp((vr / l_res).long() + M // 2, 0, M - 1)  # (P,) in [0, M-1]

    # Accumulate complex taps into the H_dd grid
    # (here amplitude is real; you could add random phase if desired)
    H.view(-1).scatter_add_(0, (l * N + k).view(-1), amp.to(torch.complex64))

    return H

def comm_dl_gen_batch_OFDM_geom(
    ebn0_db: float,
    sp: SystemParams,
    batch: int = 8,
    Nfft: int = 256,
    cp_len: int = 32,
    n_sym: int = 8,
    rng=None,
):
    """
    Generate a mini-batch of OFDM frames with a *geometry-based* TF channel.

    For each sample in the batch:
      1) Sample a random scene via GT boxes.
      2) Raycast to get (pts, its, vels).
      3) Convert rays into per-subcarrier channel H_tf[m, k] using:
           - delay index (from range) → frequency-dependent phase slope
           - Doppler (from radial velocity) → per-symbol phase rotation
      4) Insert pilot OFDM symbol (m=0) using known QPSK = 1+0j.
      5) Pass QPSK grid S_tf through the channel:
             Y_tf[m,k] = H_tf[m,k] * S_tf[m,k] + AWGN.
      6) LS channel estimate from pilots:
             H_est[k] = Y_tf[0,k] / S_tf[0,k]
             H_est[m,k] = H_est[k]  (assume slow time-invariance).
      7) Equalization:
             Y_eq[m,k] = Y_tf[m,k] / (H_est[m,k] + eps)
      8) Features:
             X = stack([Re(Y_eq), Im(Y_eq)], axis=0) → (2, n_sym, Nfft)
      9) Labels:
             bits grid shaped as (2, n_sym, Nfft) (2 bits/QPSK).

    Returned tensors:
      X_batch : (B, 2, n_sym, Nfft) float32
      Y_batch : (B, 2, n_sym, Nfft) float32 (bit labels 0/1)
    """
    if rng is None:
        rng = np.random.default_rng()

    B = batch
    bits_per_sym = 2
    Ts = Nfft / sp.fs                 # OFDM symbol duration (approx)
    eps = 1e-6

    X_list = []
    Y_list = []

    for _ in range(B):
        # ----- 1) Random scene + raycast -----
        # Simple scene generator (reuse RadarSimDataset-style)
        def _rand_gts(sp_local, rng_local):
            k = rng_local.integers(1, 3)
            gts_local = []
            for _ in range(k):
                r = rng_local.uniform(8.0, 48.0)
                az = rng_local.uniform(
                    -np.deg2rad(sp_local.az_fov / 2.0),
                    np.deg2rad(sp_local.az_fov / 2.0),
                )
                x = r * np.cos(az)
                y = r * np.sin(az)
                vx = rng_local.uniform(-20.0, 20.0)
                vy = rng_local.uniform(-6.0, 6.0)
                gts_local.append(
                    {"c": [x, y, 1.0], "s": [4, 2, 2], "v": [vx, vy, 0.0]}
                )
            return gts_local

        gts = _rand_gts(sp, rng)
        pts, its, vels, _labels = raycast_torch(
            sp,
            gts,
            lidar_like_intensity=True,
            return_labels=True,
            use_ground_reflection=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if pts.numel() == 0:
            # Degenerate scene: fall back to flat AWGN channel
            H_tf = np.ones((n_sym, Nfft), dtype=np.complex64)
        else:
            # ----- 2) Build TF channel from rays -----
            # geometry
            P = pts - to_torch([0.0, 0.0, sp.H])   # (P,3)
            R = torch.norm(P, dim=1)               # (P,)

            mask = R > 0.1
            P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
            if P.numel() == 0:
                H_tf = np.ones((n_sym, Nfft), dtype=np.complex64)
            else:
                amp = its / (R ** 2 + 1e-6)        # (P,) real gains
                vr  = torch.sum(P / R.unsqueeze(1) * vels, dim=1)  # (P,)

                # Convert geometry to delay & Doppler
                delay_s = 2.0 * R / C0            # (P,) seconds
                # approximate sample delay in baseband domain
                n_delay = (delay_s * sp.fs).cpu().numpy()  # (P,) float
                # Doppler frequency (Hz)
                f_d = (2.0 * vr / sp.lambda_m).cpu().numpy()  # (P,)

                # Build H_tf[m,k] directly:
                #   H[m,k] = sum_p a_p exp(-j2π k n_delay_p / Nfft) exp(j2π f_d_p m Ts)
                m_idx = np.arange(n_sym, dtype=np.float32)[None, :]      # (1, M)
                k_idx = np.arange(Nfft, dtype=np.float32)[None, None, :] # (1,1,N)

                # reshape for broadcasting: (P,1,1), (P,1,1), etc.
                a_p = amp.cpu().numpy().reshape(-1, 1, 1)   # (P,1,1)
                n_p = n_delay.reshape(-1, 1, 1)            # (P,1,1)
                fd_p = f_d.reshape(-1, 1, 1)               # (P,1,1)

                # phase due to delay (per subcarrier k)
                phase_k = -2j * np.pi * k_idx * n_p / float(Nfft)    # (P,1,N)
                # phase due to Doppler (per OFDM symbol m)
                phase_m = 2j * np.pi * fd_p * m_idx * Ts             # (P,M,1)

                H_tf = np.sum(
                    a_p * np.exp(phase_k) * np.exp(phase_m),
                    axis=0,
                )    # (M, Nfft) complex64-ish

        # ----- 3) QPSK data + pilots -----
        # bits: (n_sym * Nfft * 2,)
        bits = _rand_bits(n_sym * Nfft * bits_per_sym, rng)  # uint8
        bits = bits.reshape(n_sym, Nfft, 2)                  # (M,N,2)

        # Pilot pattern: first OFDM symbol m=0 all set to bit "00" => symbol 1+1j/√2
        bits[0, :, 0] = 0
        bits[0, :, 1] = 0

        # map to QPSK symbols S_tf (M,N)
        S_tf = _qpsk_gray_mod(bits)                          # complex

        # ----- 4) Pass through channel with AWGN -----
        # Y_tf = H_tf * S_tf + noise
        Y_tf = H_tf * S_tf

        # add AWGN in TF domain with proper Eb/N0
        Y_tf = _awgn(
            Y_tf,
            ebn0_db=ebn0_db,
            bits_per_sym=bits_per_sym,
            cp_ratio=cp_len / Nfft,
            rng=rng,
        )

        # ----- 5) Simple LS channel estimation from pilots (m=0) -----
        H_est = Y_tf[0, :] / (S_tf[0, :] + 1e-8)   # (Nfft,)
        H_est = np.tile(H_est[None, :], (n_sym, 1))  # (M,Nfft)

        # ----- 6) Equalization -----
        Y_eq = Y_tf / (H_est + 1e-6)   # (M,Nfft)

        # ----- 7) Build feature & label grids -----
        # X_feats: (2, M, Nfft) with real/imag channels
        X_feats = np.stack([Y_eq.real, Y_eq.imag], axis=0).astype(np.float32)

        # Label bits as (2, M, Nfft)
        Y_bits = bits.transpose(2, 0, 1).astype(np.float32)  # (2,M,N)

        X_list.append(X_feats)
        Y_list.append(Y_bits)

    X_batch = torch.from_numpy(np.stack(X_list, axis=0))  # (B,2,M,Nfft)
    Y_batch = torch.from_numpy(np.stack(Y_list, axis=0))  # (B,2,M,Nfft)

    return X_batch, Y_batch

def comm_dl_gen_batch_OTFS_geom(
    ebn0_db: float,
    sp: SystemParams,
    batch: int = 6,
    M: int = 64,
    N: int = 256,
    rng=None,
):
    """
    Generate a mini-batch of OTFS grids with a geometry-based DD channel.

    For each sample in the batch:
      1) Random scene + raycast → (pts, its, vels).
      2) Build DD channel H_dd(l,k) via `build_dd_channel_from_rays`.
      3) Generate QPSK DD data X_dd(l,k), with one "pilot row" l=0:
           - l=0 row uses known bits "00" → symbol 1+1j/√2.
           - l>0 rows random bits.
      4) Pass through channel in DD domain (simplified OTFS model):
           Y_dd = H_dd * X_dd + noise_dd
         (This corresponds to a diagonalized DD channel, ignoring DD
          coupling — fine for training a demapper that expects DD grids.)
      5) LS channel estimate from pilot row:
           H_est[k] = Y_dd[0,k] / X_dd[0,k]
           H_est[l,k] = H_est[k]  (assume invariant over Doppler index).
      6) Equalization:
           Y_eq = Y_dd / (H_est + eps)
      7) Features:
           X = [Re(Y_eq), Im(Y_eq)] → (2,M,N)
      8) Labels:
           bit grid → (2,M,N).

    This is not a *full* OTFS time-domain simulation, but ties the channel
    statistics to the raycast geometry and uses true pilot-based estimation.
    """
    if rng is None:
        rng = np.random.default_rng()

    B = batch
    bits_per_sym = 2
    eps = 1e-6

    X_list = []
    Y_list = []

    for _ in range(B):
        # ----- 1) Random scene + raycast -----
        def _rand_gts(sp_local, rng_local):
            k = rng_local.integers(1, 3)
            gts_local = []
            for _ in range(k):
                r = rng_local.uniform(8.0, 48.0)
                az = rng_local.uniform(
                    -np.deg2rad(sp_local.az_fov / 2.0),
                    np.deg2rad(sp_local.az_fov / 2.0),
                )
                x = r * np.cos(az)
                y = r * np.sin(az)
                vx = rng_local.uniform(-20.0, 20.0)
                vy = rng_local.uniform(-6.0, 6.0)
                gts_local.append(
                    {"c": [x, y, 1.0], "s": [4, 2, 2], "v": [vx, vy, 0.0]}
                )
            return gts_local

        gts = _rand_gts(sp, rng)
        pts, its, vels, _labels = raycast_torch(
            sp,
            gts,
            lidar_like_intensity=True,
            return_labels=True,
            use_ground_reflection=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # ----- 2) Build DD channel from rays -----
        H_dd_torch = build_dd_channel_from_rays(pts, its, vels, sp, M=M, N=N)
        H_dd = H_dd_torch.cpu().numpy()    # (M,N) complex64

        # If no hits, fallback to flat channel
        if np.allclose(H_dd, 0.0):
            H_dd = np.ones((M, N), dtype=np.complex64)

        # ----- 3) QPSK DD data with pilot row l=0 -----
        bits = _rand_bits(M * N * bits_per_sym, rng).reshape(M, N, 2)
        # row l=0 is pilot with "00" bits
        bits[0, :, 0] = 0
        bits[0, :, 1] = 0

        X_dd = _qpsk_gray_mod(bits)   # (M,N) complex QPSK

        # ----- 4) Pass through DD channel with AWGN -----
        # Simplified diagonal DD channel:
        Y_dd = H_dd * X_dd

        # add noise — treat each DD bin like a subcarrier
        Y_dd = _awgn(
            Y_dd,
            ebn0_db=ebn0_db,
            bits_per_sym=bits_per_sym,
            cp_ratio=0.0,    # no CP notion here
            rng=rng,
        )

        # ----- 5) LS channel estimate from pilot row l=0 -----
        H_est_row = Y_dd[0, :] / (X_dd[0, :] + 1e-8)   # (N,)
        H_est = np.tile(H_est_row[None, :], (M, 1))    # (M,N)

        # ----- 6) Equalization -----
        Y_eq = Y_dd / (H_est + eps)    # (M,N)

        # ----- 7) Features & labels -----
        X_feats = np.stack([Y_eq.real, Y_eq.imag], axis=0).astype(np.float32)
        Y_bits = bits.transpose(2, 0, 1).astype(np.float32)  # (2,M,N)

        X_list.append(X_feats)
        Y_list.append(Y_bits)

    X_batch = torch.from_numpy(np.stack(X_list, axis=0))  # (B,2,M,N)
    Y_batch = torch.from_numpy(np.stack(Y_list, axis=0))  # (B,2,M,N)

    return X_batch, Y_batch

def train_comm_demap(
    model,
    gen_batch_fn,
    cfg,
    snr_min=0,
    snr_max=18,
    epochs=5,
    steps_per_epoch=200,
    lr=3e-4,
    device=None,
    tag="OFDM",
):
    """
    Generic training loop for communication DL demapper.

    The key idea: `gen_batch_fn` encapsulates the *entire* PHY pipeline for
    each mini-batch:

      bits → QPSK → OFDM/OTFS framing → (GEOMETRIC) CHANNEL → pilots →
      channel estimation → equalization → feature grid X, label bits Y.

    Typical usage
    -------------
    - For OFDM with AWGN only (toy):
        gen_batch_fn = comm_dl_gen_batch_OFDM

    - For OFDM with raycast-based geometric channel + pilots:
        gen_batch_fn = comm_dl_gen_batch_OFDM_geom

    - For OTFS with raycast-based geometric DD channel + pilots:
        gen_batch_fn = comm_dl_gen_batch_OTFS_geom

    The training loop itself is agnostic: it just repeatedly calls
    `X, Y = gen_batch_fn(ebn0_db, **cfg)` and trains a CNN mapping
    feature grids X → bit logits Y.

    Parameters
    ----------
    model : nn.Module
        The demapper CNN (e.g., CommDemapperCNN).

    gen_batch_fn : callable
        Signature must be:
          X, Y = gen_batch_fn(ebn0_db, **cfg)
        where:
          - X: (B, C, H, W) float32 features (e.g. [Re, Im] of equalized symbols).
          - Y: (B, 2, H, W) uint8 or float32 bit labels (2 bits per QPSK symbol).

    cfg : dict
        Extra configuration passed into gen_batch_fn (Nfft, M, N, cp_len, batch, sp, etc).

    snr_min, snr_max : float
        Eb/N0 range for random sampling during training (uniform in [snr_min, snr_max]).

    epochs, steps_per_epoch : int
        Training schedule.

    lr : float
        Learning rate.

    device : torch.device or None
        Training device.

    tag : str
        Short name printed in logs ("OFDM", "OTFS", "OFDM-GEO", ...).
    """
    device = device or DEVICE
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        loss_ep = 0.0
        for _ in range(steps_per_epoch):
            eb = np.random.uniform(snr_min, snr_max)
            X, Y = gen_batch_fn(eb, **cfg)  # X: (B,C,H,W), Y: (B,2,H,W)
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = F.binary_cross_entropy_with_logits(logits, Y)
            loss.backward()
            opt.step()
            loss_ep += loss.item()
        print(f"[{tag} Demap DL] epoch {ep}/{epochs} loss {loss_ep/steps_per_epoch:.4f}")
    return model

@torch.no_grad()
def comm_demap_ber_curve(model, gen_batch_fn, cfg, ebn0_db_list, device=None):
    """
    Computes BER across Eb/N0 values using a trained demapper model.

    model(X) -> logits (B,2,H,W)
    gen_batch_fn(eb, **cfg) -> (X,Y) with Y (B,2,H,W)
    """
    device = device or DEVICE
    model = model.to(device).eval()
    cfg = dict(cfg)

    bers = []
    for eb in ebn0_db_list:
        X, Y = gen_batch_fn(eb, **cfg)
        X = X.to(device)
        logits = model(X)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        bits_hat = (probs > 0.5).astype(np.uint8)
        bits_gt = Y.numpy().astype(np.uint8)
        ber = np.mean(bits_hat != bits_gt)
        bers.append(ber)

    return np.array(bers)


# ---------------------------------------------------------------------
# RADAR DATASET (DISK-BASED) + SIMULATION
# ---------------------------------------------------------------------
class RadarDiskDataset(torch.utils.data.Dataset):
    """
    Disk-based radar dataset for RD→heatmap training.

    Expects .npz files with 'rd_f_db' and 'heatmap' arrays.
    """

    def __init__(self, folder, normalize=True):
        self.files = sorted(Path(folder).glob("*.npz"))
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        z = np.load(self.files[idx], allow_pickle=True)
        rd = z["rd_f_db"].astype(np.float32)
        hm = z["heatmap"].astype(np.float32)
        if self.normalize:
            rd = _rd_normalize(rd)
        x = torch.from_numpy(rd)[None, ...]
        y = torch.from_numpy(hm)[None, ...]
        return x, y


def make_radar_loaders(root, batch=6, workers=0):
    """
    Build torch DataLoaders for disk-based radar dataset under root/radar/{train,val}.
    """
    root = Path(root)
    tr_dir = root / "radar" / "train"
    va_dir = root / "radar" / "val"
    tr = RadarDiskDataset(tr_dir)
    va = RadarDiskDataset(va_dir)

    if len(tr) == 0:
        raise FileNotFoundError(
            f"No .npz files in {tr_dir}. Did you simulate the dataset?"
        )
    if len(va) == 0:
        raise FileNotFoundError(
            f"No .npz files in {va_dir}. Did validation simulation finish?"
        )

    dl_tr = torch.utils.data.DataLoader(
        tr, batch_size=batch, shuffle=True, num_workers=workers
    )
    dl_va = torch.utils.data.DataLoader(
        va, batch_size=batch, shuffle=False, num_workers=workers
    )
    return dl_tr, dl_va

#merged to RadarSimDataset.simulate_to_disk
def simulate_dataset(
    out_dir,
    sp: SystemParams,
    n_train: int = 1500,
    n_val: int = 300,
    seed: int = 2025,
    snr_list=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20),
    num_vis_samples: int = 5,
):
    """
    Simulate a disk-based ISAC dataset (radar + comm) and optionally generate
    a small number of visualizations for debugging.

    This function is used by Step 1 in isac_main.py.

    ---------------------------
    Radar (per sample, saved to .npz)
    ---------------------------
    Each file under:
        out_dir/radar/{train,val}/XXXXXX.npz
    contains:

      - rd_f_db       : (M, N//2)  float32
          FMCW Range–Doppler map in dB.

      - rd_o_db       : (M, N)     float32
          OTFS-like Delay–Doppler map in dB.

      - heatmap       : (M, N//2)  float32
          GT heatmap on the FMCW RD grid (kept for backward compatibility).

      - heatmap_fmcw  : (M, N//2)  float32
          Same as 'heatmap' (explicit FMCW name).

      - heatmap_otfs  : (M, N)     float32
          GT heatmap on the OTFS DD grid.

      - gts           : str (JSON)
          Serialized list of GT objects. Each GT has:
              'c': [x,y,z]  center in meters
              's': [sx,sy,sz] size extents
              'v': [vx,vy,vz] velocity in m/s

      - gt_c          : (K, 3) float32
          Target centers [x,y,z].

      - gt_v          : (K, 3) float32
          Target velocities [vx,vy,vz].

      - gt_s          : (K, 3) float32
          Target box sizes [sx,sy,sz].

      - gt_r          : (K,)   float32
          Range from radar to each target center.

      - gt_vr         : (K,)   float32
          Radial velocity of each target (projection onto LOS).

      - gt_az         : (K,)   float32
          Azimuth angle of each target center in the XY plane (rad).

    ---------------------------
    Communications (spec only)
    ---------------------------
    Under:
        out_dir/comm/train_spec.json
        out_dir/comm/val_spec.json

    we store JSON lists of dicts:
        { "seed": int, "ebn0_db": float }

    These can be used later to generate OFDM/OTFS comm frames with a
    controlled Eb/N0 distribution.

    ---------------------------
    Visualization (optional)
    ---------------------------
    For each split ('train' and 'val'), only the first `num_vis_samples`
    samples are visualized. For those, we generate:

      - 3D scene scatter + GT boxes (matplotlib).
      - PLY export of the point cloud (Open3D).
      - FMCW signal pipeline plots (via viz_fmcw_extras).
      - OTFS signal pipeline plots (via viz_otfs_extras).
      - Scatter of channel scatterers (range vs v_r).
      - Side-by-side plots:
          * FMCW RD vs FMCW GT heatmap.
          * OTFS DD vs OTFS GT heatmap.
    """
    out = Path(out_dir)
    radar_train_dir = out / "radar" / "train"
    radar_val_dir   = out / "radar" / "val"
    comm_dir        = out / "comm"
    vis_dir         = out / "radar" / "vis"

    radar_train_dir.mkdir(parents=True, exist_ok=True)
    radar_val_dir.mkdir(parents=True, exist_ok=True)
    comm_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1) Random scene generator (targets in front of radar)
    # ------------------------------------------------------------------
    def _rand_gts_two(sp_local, rng_local):
        """
        Generate a small random scene with 1–2 cuboid targets.

        Targets:
          - Range r in [8, 48] m.
          - Azimuth within the radar FOV.
          - Random lateral velocities vx, vy.
          - Fixed height z=1.0 m and box size [4,2,2].

        Returns
        -------
        gts_local : list of dict
            Each dict has keys 'c', 's', 'v'.
        """
        k = rng_local.integers(1, 3)  # number of targets: 1 or 2
        gts_local = []
        for _ in range(k):
            # Sample range and azimuth
            r = rng_local.uniform(8.0, 48.0)
            az = rng_local.uniform(
                -np.deg2rad(sp_local.az_fov / 2.0),
                np.deg2rad(sp_local.az_fov / 2.0),
            )

            # Convert to XY position
            x = r * np.cos(az)
            y = r * np.sin(az)
            
            # --- more realistic "car" style box ---
            veh_height = 1.6               # [m] total height
            veh_z_center = veh_height / 2  # so bottom sits at z≈0
            size_xyz = [4.0, 2.0, veh_height]

            # Random lateral velocities
            vx = rng_local.uniform(-20.0, 20.0)
            vy = rng_local.uniform(-6.0, 6.0)

            gts_local.append(
                {
                    "c": [float(x), float(y), float(veh_z_center)],
                    "s": [float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])],
                    "v": [float(vx), float(vy), 0.0],
                }
            )
        return gts_local

    # ------------------------------------------------------------------
    # 2) Per-sample visualization helper
    # ------------------------------------------------------------------
    def _visualize_sample(
        split,
        idx,
        pts,
        its,
        labels,
        gts,
        rd_f_db,
        dd_o_db,
        hm_f,
        hm_o,
        ra_f,
        va_f,
        ra_o,
        va_o,
        extra_f,
        extra_o,
    ):
        """
        Generate all debugging visualizations for a single sample.

        This is only called for idx < num_vis_samples for each split.
        """
        base_name = f"{split}_{idx:06d}"
        # ---- 3D scene viz & PLY export (Matplotlib + Open3D) ----------
        scene_png = vis_dir / f"scene3d_{base_name}.png"
        scene_ply = vis_dir / f"scene_{base_name}.ply"

        # visualize_scene_3d_matplotlib(
        #     sp,
        #     pts,
        #     labels=labels,
        #     gts=gts,
        #     save_path=str(scene_png),
        #     subsample=10,
        # )
        visualize_scene_3d_matplotlib(
            pts.cpu().numpy(),
            labels=labels.cpu().numpy() if labels is not None else None,
            gts=gts,
            sensor_pos=(0.0, 0.0, sp.H),
            save_path=str(scene_png),
            subsample_hits=10,
        )
        export_scene_to_ply_open3d(
            pts,
            labels=labels,
            its=its,
            save_path=str(scene_ply),
        )

        # ---- FMCW signal pipeline (tx, IQ cubes, RD map, etc.) -------
        viz_fmcw_extras(
            extra_fmcw=extra_f,
            out_prefix=vis_dir / f"sample_{base_name}_fmcw",
            fs=sp.fs,
            ra_axis=ra_f,
            va_axis=va_f,
            radar_pos=(0.0, 0.0, sp.H),
            gts=gts,
        )

        # ---- OTFS signal pipeline (IQ, DD map, etc.) ------------------
        viz_otfs_extras(
            extra_otfs=extra_o,
            out_prefix=vis_dir / f"sample_{base_name}_otfs",
            delay_axis=ra_o,
            doppler_axis=va_o,
            radar_pos=(0.0, 0.0, sp.H),
            gts=gts,
        )

        # ---- Channel scatterers (range vs v_r) ------------------------
        viz_channel_scatterers(
            extra_chan=extra_f,  # or extra_o; both contain R/vr
            out_path=vis_dir / f"sample_{base_name}_channel_fmcw.png",
        )

        # ---- Heatmap comparison: FMCW RD vs FMCW GT heatmap ----------
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # Left: RD map
        im0 = axs[0].imshow(
            rd_f_db,
            extent=[ra_f[0], ra_f[-1], va_f[0], va_f[-1]],
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        axs[0].set_title("FMCW RD (measured)")
        axs[0].set_xlabel("Range [m]")
        axs[0].set_ylabel("Radial velocity [m/s]")
        axs[0].grid(alpha=0.25, linestyle=":")
        cbar0 = plt.colorbar(im0, ax=axs[0])
        cbar0.set_label("Power [dB]")

        # Right: GT heatmap
        im1 = axs[1].imshow(
            hm_f,
            extent=[ra_f[0], ra_f[-1], va_f[0], va_f[-1]],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        axs[1].set_title("FMCW GT heatmap")
        axs[1].set_xlabel("Range [m]")
        axs[1].grid(alpha=0.25, linestyle=":")
        cbar1 = plt.colorbar(im1, ax=axs[1])
        cbar1.set_label("Heatmap value")

        plt.tight_layout()
        plt.savefig(vis_dir / f"sample_{base_name}_fmcw_rd_vs_heatmap.png", dpi=170)
        plt.close(fig)

        # ---- Heatmap comparison: OTFS DD vs OTFS GT heatmap ----------
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        im0 = axs[0].imshow(
            dd_o_db,
            extent=[ra_o[0], ra_o[-1], va_o[0], va_o[-1]],
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        axs[0].set_title("OTFS-like DD (measured)")
        axs[0].set_xlabel("Range / Delay [m]")
        axs[0].set_ylabel("Velocity / Doppler [m/s]")
        axs[0].grid(alpha=0.25, linestyle=":")
        cbar0 = plt.colorbar(im0, ax=axs[0])
        cbar0.set_label("Power [dB]")

        im1 = axs[1].imshow(
            hm_o,
            extent=[ra_o[0], ra_o[-1], va_o[0], va_o[-1]],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        axs[1].set_title("OTFS GT heatmap")
        axs[1].set_xlabel("Range / Delay [m]")
        axs[1].grid(alpha=0.25, linestyle=":")
        cbar1 = plt.colorbar(im1, ax=axs[1])
        cbar1.set_label("Heatmap value")

        plt.tight_layout()
        plt.savefig(vis_dir / f"sample_{base_name}_otfs_dd_vs_heatmap.png", dpi=170)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 3) Single-sample simulation (core logic)
    # ------------------------------------------------------------------
    def _simulate_one_sample(idx: int, split: str):
        """
        Simulate one radar sample:
          1) Random GT scene.
          2) Raycast → point cloud + intensities + velocities.
          3) FMCW and OTFS simulations → RD/DD maps (+ extras).
          4) Heatmaps from GT.
          5) Save NPZ.
          6) (Optional) visualizations for first `num_vis_samples` samples.
        """
        # 1) Random scene
        gts = _rand_gts_two(sp, rng)  # list of GT dicts

        # 2) Raycast: generate point-level returns
        pts, its, vels, labels = raycast_torch(
            sp,
            gts,
            lidar_like_intensity=True,  # intensity shaped like LiDAR
            return_labels=True,         # label each hit (object index / ground)
            use_ground_reflection=False, # include weaker ground returns
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        # 3) FMCW & OTFS simulations (with extra debug objects)
        rd_f_db, extra_f = fmcw_torch(
            pts,
            its,
            vels,
            sp,
            mti_order=1,       # FMCW radar: MTI to suppress static clutter
            noise_std=5e-4,
            return_extra=True,
        )

        dd_o_db, extra_o = otfs_torch(
            pts,
            its,
            vels,
            sp,
            mti_order=1,       # OTFS DD is already a sparse representation
            noise_std=5e-4,
            return_extra=True,
        )

        # 4) Axes + GT heatmaps
        ra_f, va_f = sp.fmcw_axes()  # one-sided range axis / velocity axis
        ra_o, va_o = sp.otfs_axes()  # delay/range axis / Doppler axis

        hm_f = _heatmap_from_gts(rd_f_db.shape, ra_f, va_f, gts, sp)
        hm_o = _heatmap_from_gts(dd_o_db.shape, ra_o, va_o, gts, sp)

        # 5) Structured GT arrays (location & velocity, plus derived quantities)
        gt_c = np.array([gt["c"] for gt in gts], dtype=np.float32)  # (K,3)
        gt_v = np.array([gt["v"] for gt in gts], dtype=np.float32)  # (K,3)
        gt_s = np.array([gt["s"] for gt in gts], dtype=np.float32)  # (K,3)

        radar_pos = np.array([0.0, 0.0, sp.H], dtype=np.float32)
        d = gt_c - radar_pos[None, :]                # (K,3) vector from radar
        gt_r = np.linalg.norm(d, axis=1)            # range
        u = d / np.maximum(gt_r[:, None], 1e-6)     # unit LOS
        gt_vr = np.sum(u * gt_v, axis=1)            # radial velocity
        gt_az = np.arctan2(gt_c[:, 1], gt_c[:, 0])  # azimuth angle in XY

        # 6) Save radar sample to NPZ
        save_dir = radar_train_dir if split == "train" else radar_val_dir
        save_path = save_dir / f"{idx:06d}.npz"

        np.savez_compressed(
            save_path,
            rd_f_db=rd_f_db.astype(np.float32),
            rd_o_db=dd_o_db.astype(np.float32),
            heatmap=hm_f.astype(np.float32),       # legacy key (FMCW)
            heatmap_fmcw=hm_f.astype(np.float32),
            heatmap_otfs=hm_o.astype(np.float32),
            gts=json.dumps(gts),                   # full JSON (for debugging)
            gt_c=gt_c,
            gt_v=gt_v,
            gt_s=gt_s,
            gt_r=gt_r.astype(np.float32),
            gt_vr=gt_vr.astype(np.float32),
            gt_az=gt_az.astype(np.float32),
        )

        # 7) Optional visualizations for the first num_vis_samples
        if idx < num_vis_samples:
            _visualize_sample(
                split,
                idx,
                pts,
                its,
                labels,
                gts,
                rd_f_db,
                dd_o_db,
                hm_f,
                hm_o,
                ra_f,
                va_f,
                ra_o,
                va_o,
                extra_f,
                extra_o,
            )

    # ------------------------------------------------------------------
    # 4) Generate radar splits (train + val)
    # ------------------------------------------------------------------
    print(f"[DATA] Simulating radar {n_train} train + {n_val} val samples → {out_dir}")
    for i in tqdm(range(n_train), desc="radar-train"):
        _simulate_one_sample(i, "train")
    for i in tqdm(range(n_val), desc="radar-val"):
        _simulate_one_sample(i, "val")

    # ------------------------------------------------------------------
    # 5) Communication spec generation (for later OFDM/OTFS sims)
    # ------------------------------------------------------------------
    def _make_comm_spec(n_items: int):
        """
        Create a list of comm simulation specs, each with a random seed
        and a target Eb/N0 from snr_list (cycled).
        """
        specs = []
        for i in range(n_items):
            specs.append(
                {
                    "seed": int(rng.integers(0, 2**31 - 1)),
                    "ebn0_db": float(snr_list[i % len(snr_list)]),
                }
            )
        return specs

    comm_train_spec = _make_comm_spec(max(n_train // 4, 400))
    comm_val_spec   = _make_comm_spec(max(n_val // 4, 100))

    with open(comm_dir / "train_spec.json", "w") as f:
        json.dump(comm_train_spec, f, indent=2)
    with open(comm_dir / "val_spec.json", "w") as f:
        json.dump(comm_val_spec, f, indent=2)

    print("[DATA] Done.")


def dataset_exists(root: str) -> bool:
    root = Path(root)
    have_radar = (
        (root / "radar" / "train").exists()
        and any((root / "radar" / "train").glob("*.npz"))
        and (root / "radar" / "val").exists()
        and any((root / "radar" / "val").glob("*.npz"))
    )
    have_comm = (
        (root / "comm" / "train_spec.json").exists()
        and (root / "comm" / "val_spec.json").exists()
    )
    return have_radar and have_comm


def simulate_if_missing(out_dir, sp, **kwargs):
    """
    Convenience wrapper: simulate dataset only if it doesn't already exist.
    """
    if dataset_exists(out_dir):
        print(f"[DATA] Found existing dataset under {out_dir} — skip simulation.")
        return
    print(f"[DATA] Simulating dataset → {out_dir}")
    #simulate_dataset(out_dir=out_dir, sp=sp, **kwargs)
    RadarSimDataset.simulate_to_disk(
        out_dir=out_dir,
        sp=sp,
        n_train=1500,
        n_val=300,
        seed=2025,
        num_vis_samples=5,
    )


import numpy as np
import matplotlib.pyplot as plt

def _gt_rv_from_boxes(gts, sp):
    """Return lists of (r, v) for each GT box using radar position & velocity."""
    pos = np.array([0.0, 0.0, sp.H])
    rs, vs = [], []
    for gt in gts:
        c = np.array(gt["c"], dtype=float)
        v = np.array(gt["v"], dtype=float)
        d = c - pos
        r = float(np.linalg.norm(d))
        if r < 1e-6:
            continue
        u = d / r
        vr = float(np.dot(u, v))
        rs.append(r); vs.append(vr)
    return rs, vs


def _plot_map_and_heatmap_single(
    path,
    rd_db,
    hm,
    ra,
    va,
    gts,
    sp,
    title_prefix="FMCW",
    cmap_map="magma",
    cmap_hm="viridis",
):
    """
    Generic helper: left = RD/DD map, right = GT heatmap.

    Parameters
    ----------
    path : str or Path
        Output filename.
    rd_db : (H,W) ndarray
        Range–Doppler or Delay–Doppler power in dB.
    hm : (H,W) ndarray
        GT heatmap on the same grid.
    ra, va : 1D ndarray
        Range and velocity axes for the grid.
    gts : list of GT boxes.
    sp : SystemParams
    title_prefix : str
        "FMCW" or "OTFS" (used for titles).
    """
    H, W = rd_db.shape
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    rs, vs = _gt_rv_from_boxes(gts, sp)

    # --------------- left: RD/DD map ----------------
    top = np.percentile(rd_db, 99.2)
    vmin = top - 35.0
    im0 = ax[0].imshow(
        rd_db,
        extent=[ra[0], ra[-1], va[0], va[-1]],
        origin="lower",
        aspect="auto",
        cmap=cmap_map,
        vmin=vmin,
        vmax=top,
    )
    ax[0].set_title(f"{title_prefix} map (dB)")
    ax[0].set_xlabel("Range (m)")
    ax[0].set_ylabel("Velocity (m/s)")
    ax[0].grid(alpha=0.25, linestyle=":")

    # mark GT on map
    for r, v in zip(rs, vs):
        if ra[0] <= r <= ra[-1] and va[0] <= v <= va[-1]:
            ax[0].plot(r, v, "wx", ms=9, mew=2)

    cb0 = fig.colorbar(im0, ax=ax[0])
    cb0.set_label("Power (dB)")

    # --------------- right: GT heatmap --------------
    im1 = ax[1].imshow(
        hm,
        extent=[ra[0], ra[-1], va[0], va[-1]],
        origin="lower",
        aspect="auto",
        cmap=cmap_hm,
    )
    ax[1].set_title(f"{title_prefix} GT heatmap")
    ax[1].set_xlabel("Range (m)")
    ax[1].set_ylabel("Velocity (m/s)")
    ax[1].grid(alpha=0.25, linestyle=":")

    for r, v in zip(rs, vs):
        if ra[0] <= r <= ra[-1] and va[0] <= v <= va[-1]:
            ax[1].plot(r, v, "wx", ms=9, mew=2)

    cb1 = fig.colorbar(im1, ax=ax[1])
    cb1.set_label("Heatmap (a.u.)")

    fig.suptitle(f"{title_prefix} RD / DD and GT heatmap", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
def visualize_radar_dataset_examples(root, sp, n_examples=4, split="train"):
    """
    For a few samples, draw 3 views:
      - FMCW: map + GT heatmap
      - OTFS: DD map + GT heatmap
      - 3D scene BEV (optional, if pts are available in NPZ)
    """
    import numpy as np
    root = Path(root)
    radar_dir = root / "radar" / split
    vis_dir = root / "radar" / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    files = sorted(radar_dir.glob("*.npz"))[:n_examples]
    for i, fpath in enumerate(files):
        data = np.load(fpath, allow_pickle=True)
        rd_f_db = data["rd_f_db"]
        rd_o_db = data["rd_o_db"]
        hm_f = data["hm_f"]
        hm_o = data["hm_o"]
        gts = json.loads(str(data["gts"]))

        # FMCW figure
        _plot_map_and_heatmap_single(
            vis_dir / f"vis_{split}_{i:03d}_fmcw.png",
            rd_f_db,
            hm_f,
            ra_f,
            va_f,
            gts,
            sp,
            title_prefix="FMCW RD",
        )

        # OTFS figure
        _plot_map_and_heatmap_single(
            vis_dir / f"vis_{split}_{i:03d}_otfs.png",
            rd_o_db,
            hm_o,
            ra_o,
            va_o,
            gts,
            sp,
            title_prefix="OTFS DD",
        )

        # (optional) BEV scene if you've stored raycast points per sample
        # viz_bev_scene(vis_dir / f"vis_{split}_{i:03d}_scene", pts, gts, sp)
        
# ---------------------------------------------------------------------
# RADAR TRAINING (UNET-LITE)
# ---------------------------------------------------------------------
class RadarNPZDataset(torch.utils.data.Dataset):
    """
    Radar dataset that loads precomputed radar samples from disk (.npz).

    Files are expected to be created by `simulate_dataset(...)` and may contain:
      - FMCW:
          rd_f_db        : (M, N_f)   [dB]  (N_f = N//2)
          heatmap_fmcw   : (M, N_f)
        legacy alias:
          heatmap        : (M, N_f)   (treated as FMCW heatmap)

      - OTFS:
          rd_o_db        : (M, N)     [dB]
          heatmap_otfs   : (M, N)

    This class can expose either FMCW or OTFS samples, depending on `radar_mode`.

    Parameters
    ----------
    root : str or Path
        Root directory of the dataset, e.g. "./output/isac_main2".
        Radar samples are searched under root / "radar" / split.

    split : {"train", "val"}
        Which subset to load.

    radar_mode : {"fmcw", "otfs"}
        Which radar representation to expose:
          - "fmcw" → (1, M, N_f) RD maps
          - "otfs" → (1, M, N)   DD maps

    normalize : bool, default True
        Whether to run `_rd_normalize` on the dB maps before returning.
        (For OTFS we reuse the same dynamic-range normalization.)
    """

    def __init__(self, root, split="train", radar_mode="fmcw", normalize=True):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.radar_mode = radar_mode.lower()
        assert self.radar_mode in ("fmcw", "otfs"), \
            f"Unsupported radar_mode={radar_mode}; use 'fmcw' or 'otfs'."
        self.normalize = normalize

        radar_dir = self.root / "radar" / split
        if not radar_dir.exists():
            raise FileNotFoundError(
                f"[RadarNPZDataset] Directory not found: {radar_dir}. "
                "Run simulate_dataset(...) first."
            )

        self.files = sorted(radar_dir.glob("*.npz"))
        if not self.files:
            raise RuntimeError(
                f"[RadarNPZDataset] No .npz files found in {radar_dir}. "
                "Run simulate_dataset(...) first."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns
        -------
        x : (1, M, W) torch.float32
            Normalized radar map (FMCW RD or OTFS DD).

        y : (1, M, W) torch.float32
            Ground-truth heatmap aligned with x.
        """
        path = self.files[idx]
        data = np.load(path)

        if self.radar_mode == "fmcw":
            # ----- FMCW RD map -----
            rd_db = data["rd_f_db"]              # (M, N_f)
            # Prefer explicit FMCW key; fall back to legacy 'heatmap'
            if "heatmap_fmcw" in data:
                hm = data["heatmap_fmcw"]
            else:
                hm = data["heatmap"]
        else:
            # ----- OTFS DD map -----
            rd_db = data["rd_o_db"]              # (M, N)
            hm = data["heatmap_otfs"]

        if self.normalize:
            rd = _rd_normalize(rd_db)            # → (M, W) in [0,1]
        else:
            rd = rd_db.astype(np.float32)

        x = torch.from_numpy(rd[None, ...])      # (1, M, W)
        y = torch.from_numpy(hm[None, ...])      # (1, M, W)
        return x, y
    
class RadarSimDataset(torch.utils.data.Dataset):
    """
    Radar simulation dataset used in two ways:

    1) **Online / on-the-fly mode (this Dataset class itself)**

       Each __getitem__ does:
         - Sample a random 3D scene with moving cuboids (cars).
         - Call `raycast_torch` → point-level hits (pts, its, vels).
         - For the chosen `radar_mode` ("fmcw" or "otfs"):
             * FMCW  → `fmcw_torch`  → Range–Doppler (RD) map in dB.
             * OTFS   → `otfs_torch_full` → Delay–Doppler (DD) map in dB.
         - Convert the map to a normalized tensor x.
         - Build a GT heatmap aligned with that map using `_heatmap_from_gts`
           → tensor y.

       This mode is handy when you want infinite synthetic data without
       touching disk.

    2) **Offline / disk mode via the classmethod `simulate_to_disk(...)`**

       This method:
         - Uses the same random scene generator as online mode.
         - Simulates both FMCW and OTFS radar per sample.
         - Writes .npz files under:
               out_dir/radar/{train,val}/XXXXXX.npz
           plus communication specs under out_dir/comm.
         - Optionally generates a small number of visualization PNGs and
           PLY point clouds for debugging.

       The on-disk format is documented in `simulate_to_disk(...)`, and is
       what `isac_main.py` uses in Part 1.
    """

    # ------------------------------------------------------------------
    # ctor: online / on-the-fly dataset
    # ------------------------------------------------------------------
    def __init__(
        self,
        sp: SystemParams,
        n_items: int = 2000,
        rng_seed: int = 123,
        min_targets: int = 1,
        max_targets: int = 3,
        radar_mode: str = "fmcw",
    ):
        super().__init__()
        self.sp = sp
        self.n = int(n_items)
        self.rng = np.random.default_rng(rng_seed)
        self.min_t = int(min_targets)
        self.max_t = int(max_targets)
        self.radar_mode = radar_mode.lower()
        assert self.radar_mode in ("fmcw", "otfs"), \
            f"Unsupported radar_mode={radar_mode}; use 'fmcw' or 'otfs'."

        # Precompute the axes used when building heatmaps in __getitem__.
        if self.radar_mode == "fmcw":
            # FMCW RD axes:  range [m], radial velocity [m/s]
            self.ra, self.va = sp.fmcw_axes()
        else:
            # For online OTFS, we approximate axes from sp; in the *disk*
            # dataset we use exact axes from otfs_torch_full extras.
            self.ra, self.va = sp.otfs_axes()

    # ------------------------------------------------------------------
    # shared GT scene sampler (used by both online dataset and disk sim)
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_random_gts(
        sp: SystemParams,
        rng: np.random.Generator,
        min_targets: int,
        max_targets: int,
    ):
        """
        Sample a small random scene with `k` moving cuboids ("cars").

        Geometric model:
          - Radar is at (0, 0, sp.H) looking down the +X axis.
          - Each target:
              range r    ~ U[8, 48] m
              azimuth az ~ U[-az_fov/2, +az_fov/2]  (radians)
              x = r cos(az), y = r sin(az)
              z center   = vehicle_height/2 so that the bottom sits near z=0.
              size (L,W,H) ~ fixed [4, 2, vehicle_height]
              lateral velocities vx ~ U[-20,20], vy ~ U[-6,6], vz = 0

        Returns
        -------
        gts : list of dict
            Each dict has keys:
               'c' : [x, y, z] center position in meters.
               's' : [sx, sy, sz] box sizes in meters.
               'v' : [vx, vy, vz] velocity in m/s.
        """
        k = rng.integers(min_targets, max_targets + 1)
        gts = []

        vehicle_height = 1.6  # m, approximate car height
        z_center = vehicle_height / 2.0

        for _ in range(k):
            # Range / azimuth sampling within FOV
            r = rng.uniform(8.0, 48.0)
            az = rng.uniform(
                -np.deg2rad(sp.az_fov / 2.0),
                np.deg2rad(sp.az_fov / 2.0),
            )

            x = r * np.cos(az)
            y = r * np.sin(az)

            vx = rng.uniform(-20.0, 20.0)
            vy = rng.uniform(-6.0, 6.0)

            gts.append(
                {
                    "c": [float(x), float(y), float(z_center)],
                    "s": [4.0, 2.0, float(vehicle_height)],
                    "v": [float(vx), float(vy), 0.0],
                }
            )
        return gts

    # ------------------------------------------------------------------
    # online / on-the-fly access
    # ------------------------------------------------------------------
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        """
        Returns a single training pair (x, y) for online simulation.

        x : (1, M, W) torch.float32
            Normalized radar map (FMCW RD or OTFS DD).
            W = N//2 for FMCW (one-sided spectrum), W = N for OTFS.

        y : (1, M, W) torch.float32
            GT heatmap on the same grid, generated from GT boxes in the
            (range, radial-velocity) domain.
        """
        sp = self.sp

        # 1) Random scene
        gts = self._sample_random_gts(
            sp=sp,
            rng=self.rng,
            min_targets=self.min_t,
            max_targets=self.max_t,
        )

        # 2) Raycast → point-level hits (used by FMCW; also useful for viz)
        pts, its, vels, _labels = raycast_torch(
            sp,
            gts,
            num_az = 512,
            num_el= 64,
            lidar_like_intensity=True,
            return_labels=True,
            use_ground_reflection=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 3) Radar map from selected mode
        if self.radar_mode == "fmcw":
            rd_db, _extra = fmcw_torch(
                pts,
                its,
                vels,
                sp,
                mti_order=1,        # simple MTI to suppress static clutter
                noise_std=5e-4,
                return_extra=True,
            )
            ra = self.ra
            va = self.va

        else:  # "otfs"
            # Full OTFS radar: uses GT-based time-domain channel + OTFS
            dd_db, extra_o = otfs_torch_full(
                gts,
                sp,
                noise_std=5e-4,
                return_extra=True,
            )
            rd_db = dd_db

            # Axes from OTFS extras: delays in meters, Doppler in Hz.
            # Convert Doppler (Hz) to radial velocity [m/s]:
            ra = extra_o["delays_m"]                      # (N,)
            va = extra_o["doppler_hz"] * sp.lambda_m / 2  # (M,)

        # 4) Normalize map and build heatmap on the same grid
        rd = _rd_normalize(rd_db)                 # (M, W)
        hm = _heatmap_from_gts(rd.shape, ra, va, gts, sp)

        x = torch.from_numpy(rd[None, ...])       # (1, M, W)
        y = torch.from_numpy(hm[None, ...])       # (1, M, W)
        return x, y

    # ------------------------------------------------------------------
    #  offline: simulate full disk dataset (old simulate_dataset)
    # ------------------------------------------------------------------
    @classmethod
    def simulate_to_disk(
        cls,
        out_dir,
        sp: SystemParams,
        n_train: int = 1500,
        n_val: int = 300,
        seed: int = 2025,
        snr_list=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20),
        num_vis_samples: int = 5,
        min_targets: int = 1,
        max_targets: int = 2,
    ):
        """
        Simulate a disk-based ISAC dataset (radar + comm) and optionally
        generate a small number of visualizations for debugging.

        This is a **class method** so that it can share logic with the
        online dataset (`RadarSimDataset._sample_random_gts`).

        ---------------------------
        Radar (per sample, saved to .npz)
        ---------------------------
        Each file under:
            out_dir/radar/{train,val}/XXXXXX.npz
        contains:

          - rd_f_db       : (M, N//2)  float32
              FMCW Range–Doppler map in dB.

          - rd_o_db       : (M, N)     float32
              OTFS Delay–Doppler map in dB (from full OTFS chain).

          - heatmap       : (M, N//2)  float32
              GT heatmap on the FMCW RD grid (kept for backward compat).

          - heatmap_fmcw  : (M, N//2)  float32
          - heatmap_otfs  : (M, N)     float32

          - gts           : JSON string
              Serialized list of GT objects with keys 'c', 's', 'v'.

          - gt_c, gt_v, gt_s : (K,3) float32
          - gt_r, gt_vr, gt_az : (K,) float32
              Range, radial velocity, and azimuth of each target center.

        ---------------------------
        Communications (spec only)
        ---------------------------
        Under:
            out_dir/comm/train_spec.json
            out_dir/comm/val_spec.json

        we store JSON lists of dicts:
            { "seed": int, "ebn0_db": float }

        These are later consumed by OFDM/OTFS comm simulators.

        ---------------------------
        Visualization (optional)
        ---------------------------
        For each split ('train' and 'val'), only the first `num_vis_samples`
        samples are visualized. We generate:

          - 3D scene scatter + GT boxes (matplotlib).
          - PLY export of the point cloud (Open3D).
          - FMCW signal pipeline plots (via viz_fmcw_extras).
          - OTFS signal pipeline plots (via viz_otfs_extras).
          - Scatter of channel scatterers (range vs v_r).
          - Side-by-side plots:
              * FMCW RD vs FMCW GT heatmap.
              * OTFS DD vs OTFS GT heatmap.
        """
        out = Path(out_dir)
        radar_train_dir = out / "radar" / "train"
        radar_val_dir   = out / "radar" / "val"
        comm_dir        = out / "comm"
        vis_dir         = out / "radar" / "vis"

        radar_train_dir.mkdir(parents=True, exist_ok=True)
        radar_val_dir.mkdir(parents=True, exist_ok=True)
        comm_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed)

        # --------------------------------------------------------------
        # local helper: visualizations for the first num_vis_samples
        # --------------------------------------------------------------
        def _visualize_sample(
            split: str,
            idx: int,
            pts,
            its,
            labels,
            gts,
            rd_f_db,
            dd_o_db,
            hm_f,
            hm_o,
            ra_f,
            va_f,
            ra_o,
            va_o,
            extra_f,
            extra_o,
        ):
            """
            Generate debugging plots for a single sample.

            All inputs are numpy or torch arrays already aligned to
            the radar grids.
            """
            base_name = f"{split}_{idx:06d}"

            # 3D scene + PLY export
            scene_png = vis_dir / f"scene3d_{base_name}.png"
            scene_ply = vis_dir / f"scene_{base_name}.ply"

            visualize_scene_3d_matplotlib(
                pts.cpu().numpy(),
                labels=labels.cpu().numpy() if labels is not None else None,
                gts=gts,
                sensor_pos=(0.0, 0.0, sp.H),
                save_path=str(scene_png),
                subsample_hits=10,
            )
            export_scene_to_ply_open3d(
                pts,
                labels=labels,
                its=its,
                save_path=str(scene_ply),
            )

            # FMCW pipeline (IQ, RD, ground truth, etc.)
            viz_fmcw_extras(
                extra_fmcw=extra_f,
                out_prefix=vis_dir / f"sample_{base_name}_fmcw",
                fs=sp.fs,
                ra_axis=ra_f,
                va_axis=va_f,
                radar_pos=(0.0, 0.0, sp.H),
                gts=gts,
            )

            # OTFS pipeline (time-domain, DD map, GT grid, etc.)
            # Note: we pass range axis in meters and velocity axis in m/s.
            viz_otfs_extras(
                extra_otfs=extra_o,
                out_prefix=vis_dir / f"sample_{base_name}_otfs",
                delay_axis=ra_o,
                doppler_axis=va_o,
                radar_pos=(0.0, 0.0, sp.H),
                gts=gts,
            )

            # Scatter of channel scatterers (range vs v_r)
            viz_channel_scatterers(
                extra_chan=extra_f,  # FMCW extras carry R / vr / amp
                out_path=vis_dir / f"sample_{base_name}_channel_fmcw.png",
            )

            # FMCW RD vs heatmap
            fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            im0 = axs[0].imshow(
                rd_f_db,
                extent=[ra_f[0], ra_f[-1], va_f[0], va_f[-1]],
                origin="lower",
                aspect="auto",
                cmap="magma",
            )
            axs[0].set_title("FMCW RD (measured)")
            axs[0].set_xlabel("Range [m]")
            axs[0].set_ylabel("Radial velocity [m/s]")
            axs[0].grid(alpha=0.25, linestyle=":")
            cbar0 = plt.colorbar(im0, ax=axs[0])
            cbar0.set_label("Power [dB]")

            im1 = axs[1].imshow(
                hm_f,
                extent=[ra_f[0], ra_f[-1], va_f[0], va_f[-1]],
                origin="lower",
                aspect="auto",
                cmap="viridis",
            )
            axs[1].set_title("FMCW GT heatmap")
            axs[1].set_xlabel("Range [m]")
            axs[1].grid(alpha=0.25, linestyle=":")
            cbar1 = plt.colorbar(im1, ax=axs[1])
            cbar1.set_label("Heatmap value")

            plt.tight_layout()
            plt.savefig(vis_dir / f"sample_{base_name}_fmcw_rd_vs_heatmap.png", dpi=170)
            plt.close(fig)

            # OTFS DD vs heatmap
            fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            im0 = axs[0].imshow(
                dd_o_db,
                extent=[ra_o[0], ra_o[-1], va_o[0], va_o[-1]],
                origin="lower",
                aspect="auto",
                cmap="magma",
            )
            axs[0].set_title("OTFS DD (measured)")
            axs[0].set_xlabel("Range / Delay [m]")
            axs[0].set_ylabel("Velocity / Doppler [m/s]")
            axs[0].grid(alpha=0.25, linestyle=":")
            cbar0 = plt.colorbar(im0, ax=axs[0])
            cbar0.set_label("Power [dB]")

            im1 = axs[1].imshow(
                hm_o,
                extent=[ra_o[0], ra_o[-1], va_o[0], va_o[-1]],
                origin="lower",
                aspect="auto",
                cmap="viridis",
            )
            axs[1].set_title("OTFS GT heatmap")
            axs[1].set_xlabel("Range / Delay [m]")
            axs[1].grid(alpha=0.25, linestyle=":")
            cbar1 = plt.colorbar(im1, ax=axs[1])
            cbar1.set_label("Heatmap value")

            plt.tight_layout()
            plt.savefig(vis_dir / f"sample_{base_name}_otfs_dd_vs_heatmap.png", dpi=170)
            plt.close(fig)

        # --------------------------------------------------------------
        # core per-sample simulation used for both train and val
        # --------------------------------------------------------------
        def _simulate_one_sample(idx: int, split: str):
            """
            Simulate one radar sample and write its .npz file.

            Steps:
              1) Random GT scene.
              2) Raycast → pts, its, vels, labels.
              3) FMCW RD via fmcw_torch (full IQ pipeline).
              4) OTFS DD via otfs_torch_full (full OTFS chain).
              5) Build axes and GT heatmaps.
              6) Save all arrays to disk.
              7) If idx < num_vis_samples, generate visualizations.
            """
            # 1) Random scene
            gts = cls._sample_random_gts(
                sp=sp,
                rng=rng,
                min_targets=min_targets,
                max_targets=max_targets,
            )

            # 2) Raycast for FMCW + 3D viz
            pts, its, vels, labels = raycast_torch(
                sp,
                gts,
                num_az = 512,
                num_el = 64,
                lidar_like_intensity=True,
                return_labels=True,
                use_ground_reflection=False,
            )#[3647, 3]=>[899, 3]
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            # 3) FMCW simulation
            rd_f_db, extra_f = fmcw_torch(
                pts,
                its,
                vels,
                sp,
                mti_order=1,
                noise_std=5e-4,
                return_extra=True,
            )#(512, 256)
            ra_f, va_f = sp.fmcw_axes()  # range [m], velocity [m/s]

            # ------------------------------------------------------------------
            # OTFS parameter dictionary (CP length is optional)
            # ------------------------------------------------------------------

            # If SystemParams has cp_len, use it; otherwise fall back to a
            # reasonable default (e.g. 1/8 of N) or even 0 if you prefer.
            cp_len_default = sp.N // 8        # 64 typical OFDM choice (12.5% CP)
            cp_len = int(getattr(sp, "cp_len", cp_len_default))

            otfs_par = dict(
                M=sp.M,                # Doppler bins (slow-time symbols)
                N=sp.N,                # delay bins (subcarriers)
                T_sym=sp.T_chirp,      # OTFS symbol / OFDM symbol duration [s]
                Ts=1.0 / sp.fs,        # base sampling interval [s]
                Nfft=sp.N,             # OFDM FFT size (we keep Nfft == N)
                cp_len=cp_len,         # CP length in *samples*
            )
            # dd_o_db, extra_o = otfs_torch_full(
            #     gts,
            #     sp,
            #     noise_std=5e-4,
            #     return_extra=True,
            # )
            dd_o_db, extra_o = otfs_torch_full_radar(
                gts,            # use geometry for OTFS radar
                sp,
                otfs_par,
                noise_std=5e-4,
                pilot_amplitude=1.0,
                pilot_pos=(0, 0),
                return_extra=True,
                mti_order=1
            )
            # extra_o contains:
            #   - "H_est"   : estimated DD response (M, N) complex This is the estimated delay–Doppler channel
            #   - "H_gt"    : DD ground-truth (M, N) complex (currently mostly zeros)
            #   - "delays"  : delay axis in seconds, shape (N,)
            #   - "dopplers": Doppler axis in Hz, shape (M,)
            #   - "taps"    : list of dicts with (n_delay, m_dopp, alpha, tau, nu, f_d)
            #   - "s_tx", "r_rx": transmit and received time-domain signals
            #   - "gts"     : original GT list
            
            # Convert delay [s] → equivalent range [m]:
            #   τ = 2R / c  ⇒  R = c τ / 2
            delays_s = np.asarray(extra_o["delays"], dtype=np.float64)   # (512,) (N,)
            ra_o = delays_s * C0 / 2.0    #(512,) (N,) range [m]

            # Convert Doppler [Hz] → radial velocity [m/s]:
            #   f_d = 2 v_r / λ  ⇒  v_r = f_d λ / 2
            dopplers_hz = np.asarray(extra_o["dopplers"], dtype=np.float64)  # (M,)
            va_o = dopplers_hz * (sp.lambda_m / 2.0)                        # (M,) m/s
            
            
            # # Axes from OTFS extras: delays in meters, Doppler (Hz) → v_r (m/s)
            # ra_o = extra_o["delays_m"]                      # (N,)
            # va_o = extra_o["doppler_hz"] * sp.lambda_m / 2  # (M,)

            # ------------------------------------------------------------------
            # 4) GT heatmaps aligned with FMCW RD and OTFS DD grids
            # ------------------------------------------------------------------
            hm_f = _heatmap_from_gts(rd_f_db.shape, ra_f, va_f, gts, sp)   # (512, 256) FMCW grid
            hm_o = _heatmap_from_gts(dd_o_db.shape, ra_o, va_o, gts, sp)   # (512, 512) OTFS grid

            # ------------------------------------------------------------------
            # 5) Structured GT arrays (centers, velocities, sizes, derived metrics)
            # ------------------------------------------------------------------
            gt_c = np.array([gt["c"] for gt in gts], dtype=np.float32)  # (K, 3) (1, 3)
            gt_v = np.array([gt["v"] for gt in gts], dtype=np.float32)  # (K, 3)
            gt_s = np.array([gt["s"] for gt in gts], dtype=np.float32)  # (K, 3)

            radar_pos = np.array([0.0, 0.0, sp.H], dtype=np.float32)
            d = gt_c - radar_pos[None, :]                 # (K, 3) vector from radar
            gt_r = np.linalg.norm(d, axis=1)             # range [m]
            u = d / np.maximum(gt_r[:, None], 1e-6)      # unit LOS
            gt_vr = np.sum(u * gt_v, axis=1)             # radial velocity [m/s]
            gt_az = np.arctan2(gt_c[:, 1], gt_c[:, 0])   # azimuth [rad]

            # ------------------------------------------------------------------
            # 6) Save radar sample to NPZ (FMCW + OTFS + GT)
            # ------------------------------------------------------------------
            save_dir = radar_train_dir if split == "train" else radar_val_dir
            save_path = save_dir / f"{idx:06d}.npz"
            np.savez_compressed(
                save_path,
                rd_f_db=rd_f_db.astype(np.float32),
                rd_o_db=dd_o_db.astype(np.float32),
                heatmap=hm_f.astype(np.float32),       # legacy key
                heatmap_fmcw=hm_f.astype(np.float32),
                heatmap_otfs=hm_o.astype(np.float32),
                gts=json.dumps(gts),
                gt_c=gt_c,
                gt_v=gt_v,
                gt_s=gt_s,
                gt_r=gt_r.astype(np.float32),
                gt_vr=gt_vr.astype(np.float32),
                gt_az=gt_az.astype(np.float32),
            )

            # 7) Optional viz
            if idx < num_vis_samples:
                _visualize_sample(
                    split,
                    idx,
                    pts,
                    its,
                    labels,
                    gts,
                    rd_f_db,
                    dd_o_db,
                    hm_f,
                    hm_o,
                    ra_f,
                    va_f,
                    ra_o,
                    va_o,
                    extra_f,
                    extra_o,
                )

        # --------------------------------------------------------------
        # loop over splits
        # --------------------------------------------------------------
        print(f"[DATA] Simulating radar {n_train} train + {n_val} val samples → {out_dir}")
        for i in tqdm(range(n_train), desc="radar-train"):
            _simulate_one_sample(i, "train")
        for i in tqdm(range(n_val), desc="radar-val"):
            _simulate_one_sample(i, "val")

        # --------------------------------------------------------------
        # comm spec generation (for later OFDM/OTFS comm sims)
        # --------------------------------------------------------------
        def _make_comm_spec(n_items: int):
            specs = []
            for i in range(n_items):
                specs.append(
                    {
                        "seed": int(rng.integers(0, 2**31 - 1)),
                        "ebn0_db": float(snr_list[i % len(snr_list)]),
                    }
                )
            return specs

        comm_train_spec = _make_comm_spec(max(n_train // 4, 400))
        comm_val_spec   = _make_comm_spec(max(n_val // 4, 100))

        with open(comm_dir / "train_spec.json", "w") as f:
            json.dump(comm_train_spec, f, indent=2)
        with open(comm_dir / "val_spec.json", "w") as f:
            json.dump(comm_val_spec, f, indent=2)

        print("[DATA] Done.")


def train_radar_model(
    sp: SystemParams,
    data_root: str | Path | None = None,
    radar_mode: str = "fmcw",
    epochs: int = 5,
    batch: int = 6,
    lr: float = 1e-3,
    n_train: int = 800,
    n_val: int = 200,
    device=None,
    regenerate_if_missing: bool = True,
):
    """
    Train UNetLite radar detector on either FMCW or OTFS maps.

    Data source options
    -------------------
    1) Disk-backed dataset (preferred):
         - Set `data_root` to a directory where `simulate_dataset(...)`
           has written .npz files.
         - Set `radar_mode` to "fmcw" or "otfs".
         - If the radar .npz files are missing and `regenerate_if_missing=True`,
           this will call `simulate_dataset(data_root, sp, ...)` for you.

    2) On-the-fly simulated dataset:
         - Set `data_root=None`.
         - Data is generated by `RadarSimDataset` at every access.

    Parameters
    ----------
    sp : SystemParams
        Radar configuration.

    data_root : str | Path | None
        Root directory of dataset (contains "radar/train" and "radar/val").
        If None, uses on-the-fly simulation only.

    radar_mode : {"fmcw", "otfs"}
        Which radar representation to learn:
          - "fmcw" : RD maps of shape (M, N//2)
          - "otfs" : DD maps of shape (M, N)

    epochs, batch, lr, n_train, n_val, device, regenerate_if_missing :
        Usual training hyperparameters and behaviour flags.

    Returns
    -------
    net : UNetLite
        Trained UNetLite model (best-val checkpoint).
    """
    radar_mode = radar_mode.lower()
    assert radar_mode in ("fmcw", "otfs"), \
        f"Unsupported radar_mode={radar_mode}; use 'fmcw' or 'otfs'."

    device = device or DEVICE

    # ----------------------------------------------------------
    # 1) Decide: disk dataset vs on-the-fly simulation
    # ----------------------------------------------------------
    use_disk = data_root is not None
    data_root = Path(data_root) if data_root is not None else None

    if use_disk:
        radar_train_dir = data_root / "radar" / "train"
        radar_val_dir   = data_root / "radar" / "val"

        train_exists = radar_train_dir.exists() and any(radar_train_dir.glob("*.npz"))
        val_exists   = radar_val_dir.exists()   and any(radar_val_dir.glob("*.npz"))

        if not (train_exists and val_exists):
            if regenerate_if_missing:
                print(
                    f"[train_radar_model] No radar .npz in {data_root}; "
                    "calling simulate_dataset(...) to generate them."
                )
                # simulate_dataset(
                #     out_dir=data_root,
                #     sp=sp,
                #     n_train=n_train,
                #     n_val=n_val,
                #     seed=2025,
                # )
                RadarSimDataset.simulate_to_disk(
                    out_dir=data_root,
                    sp=sp,
                    n_train=n_train,
                    n_val=n_val,
                    seed=2025,
                    num_vis_samples=5,
                )
            else:
                raise FileNotFoundError(
                    f"[train_radar_model] Radar .npz not found under {data_root} "
                    "and regenerate_if_missing=False."
                )

        ds_tr = RadarNPZDataset(
            root=data_root,
            split="train",
            radar_mode=radar_mode,
            normalize=True,
        )
        ds_va = RadarNPZDataset(
            root=data_root,
            split="val",
            radar_mode=radar_mode,
            normalize=True,
        )
        print(
            f"[train_radar_model] Using disk dataset ({radar_mode}) from {data_root} "
            f"(train={len(ds_tr)}, val={len(ds_va)})"
        )
    else:
        # Purely simulated in-memory dataset
        ds_tr = RadarSimDataset(
            sp,
            n_items=n_train,
            rng_seed=2025,
            radar_mode=radar_mode,
        )
        ds_va = RadarSimDataset(
            sp,
            n_items=n_val,
            rng_seed=2026,
            radar_mode=radar_mode,
        )
        print(
            f"[train_radar_model] Using on-the-fly simulated dataset ({radar_mode}) "
            f"(train={n_train}, val={n_val})"
        )

    dl_tr = torch.utils.data.DataLoader(
        ds_tr, batch_size=batch, shuffle=True,  num_workers=0
    )
    dl_va = torch.utils.data.DataLoader(
        ds_va, batch_size=batch, shuffle=False, num_workers=0
    )

    # ----------------------------------------------------------
    # 2) Model, optimizer, training loop
    # ----------------------------------------------------------
    net = UNetLite(in_ch=1, ch=32).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)

    best = {"loss": float("inf"), "state": None}

    for ep in range(1, epochs + 1):
        # ----- train -----
        net.train()
        loss_tr = 0.0
        for x, y in dl_tr:
            # x, y : (B, 1, M, W)
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = net(x)
            loss = focal_bce_with_logits(logits, y, alpha=0.25, gamma=2.0)
            loss.backward()
            opt.step()
            loss_tr += loss.item() * x.size(0)
        loss_tr /= len(ds_tr)

        # ----- val -----
        net.eval()
        loss_va = 0.0
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(device)
                y = y.to(device)
                logits = net(x)
                loss = focal_bce_with_logits(logits, y)
                loss_va += loss.item() * x.size(0)
        loss_va /= len(ds_va)

        print(
            f"[RadarDL-{radar_mode}] epoch {ep}/{epochs}  "
            f"train {loss_tr:.4f}  val {loss_va:.4f}"
        )

        if loss_va < best["loss"]:
            best["loss"] = loss_va
            best["state"] = {k: v.cpu() for k, v in net.state_dict().items()}

    if best["state"] is not None:
        net.load_state_dict(best["state"])
    else:
        print("[RadarDL] Warning: no best state recorded; returning last epoch weights.")

    return net


def load_radar_from_ckpt(root, device=None, prefer_best=True):
    """
    Utility to load radar UNetLite from checkpoints under root/checkpoints.
    """
    device = device or DEVICE
    ckpt_dir = Path(root) / "checkpoints"
    best_path = ckpt_dir / "radar_unet_best_only.pt"
    full_path = ckpt_dir / "radar_unet.pt"

    net = UNetLite().to(device)
    if prefer_best and best_path.exists():
        state = torch.load(best_path, map_location=device)
        net.load_state_dict(state, strict=True)
        print(f"[LOAD] Loaded radar model (best weights): {best_path}")
    elif full_path.exists():
        state = torch.load(full_path, map_location=device)
        net.load_state_dict(state["model"], strict=True)
        print(f"[LOAD] Loaded radar model (last checkpoint): {full_path}")
    else:
        raise FileNotFoundError(f"No radar checkpoint found in {ckpt_dir}.")
    net.eval()
    return net