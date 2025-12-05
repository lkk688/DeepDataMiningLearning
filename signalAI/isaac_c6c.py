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

try:
    import scipy.ndimage as ndi
    SCIPY = True
except ImportError:
    SCIPY = False
    print("Warning: Scipy not installed. Using NumPy-only ops.")

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
    fc: float = 77e9
    B: float = 150e6      # FMCW bandwidth
    fs: float = 150e6     # ADC sampling rate (>= B)
    M: int = 512          # chirps (Doppler bins)
    N: int = 512          # samples per chirp (Range FFT size)
    H: float = 1.8        # sensor height
    az_fov: float = 60.0  # azimuth FOV in degrees
    el_fov: float = 20.0  # elevation FOV in degrees
    bev_r_max: float = 50.0  # BEV clamp range (m)

    @property
    def lambda_m(self): 
        return C0 / self.fc

    @property
    def T_chirp(self):  
        return self.N / self.fs

    @property
    def slope(self):    
        return self.B / self.T_chirp  # S = B/T

    def fmcw_axes(self):
        """
        FMCW RD axes:
          - range: bins 0..N/2-1 (one-sided)
          - Doppler: fftshifted slow-time axis
        """
        ra = (C0 / (2.0 * self.B)) * np.arange(self.N // 2)
        f_d = np.fft.fftshift(np.fft.fftfreq(self.M, d=self.T_chirp))
        va = (self.lambda_m / 2.0) * f_d
        return ra, va

    def otfs_axes(self):
        """
        Keep your previous OTFS extents for consistency.
        """
        r = np.linspace(0, (C0 / (2 * self.fs)) * self.N, self.N)
        v = np.linspace(-(self.lambda_m/2)*(self.fs/(self.N*self.M))*(self.M/2),
                         (self.lambda_m/2)*(self.fs/(self.N*self.M))*(self.M/2), self.M)
        return r, v


# ---------------------------------------------------------------------
# UTILS
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
    Ensure data is in dB for visualization.
    If it already looks like dB (large dynamic range), keep as-is.
    """
    rd = np.asarray(rd)
    if np.nanmax(rd) > 100:
        return rd
    return 20 * np.log10(np.abs(rd) + 1e-9)


# ---------------------------------------------------------------------
# SUBPIXEL PEAK REFINEMENT
# ---------------------------------------------------------------------
def _subpixel_quadratic(heat: np.ndarray, y: int, x: int):
    """Return (dy, dx) in [-1,1] via 1D quadratic peak fit around (y,x)."""
    H, W = heat.shape
    y0, y1, y2 = max(0, y-1), y, min(H-1, y+1)
    x0, x1, x2 = max(0, x-1), x, min(W-1, x+1)

    def quad_peak(a, b, c):
        denom = (a - 2*b + c)
        if abs(denom) < 1e-9:
            return 0.0
        t = 0.5 * float(a - c) / float(denom)
        return float(np.clip(t, -1.0, 1.0))

    dy = quad_peak(heat[y0, x1], heat[y1, x1], heat[y2, x1])
    dx = quad_peak(heat[y1, x0], heat[y1, x1], heat[y1, x2])
    return dy, dx


def _rv_from_idx_with_subpix(y, x, dy, dx, ra, va):
    """
    Convert peak index + subpixel offest to (range, vel) using linear interpolation.
    va is vertical axis (rows), ra is horizontal axis (cols).
    """
    y_f = np.clip(y + dy, 0, len(va) - 1)
    x_f = np.clip(x + dx, 0, len(ra) - 1)
    y0, y1 = int(np.floor(y_f)), min(int(np.floor(y_f)) + 1, len(va)-1)
    x0, x1 = int(np.floor(x_f)), min(int(np.floor(x_f)) + 1, len(ra)-1)
    wy = y_f - y0
    wx = x_f - x0
    v = (1-wy)*va[y0] + wy*va[y1]
    r = (1-wx)*ra[x0] + wx*ra[x1]
    return r, v


# ---------------------------------------------------------------------
# BOX RAYCAST SCENE
# ---------------------------------------------------------------------
def raycast_torch(sp: SystemParams, gts):
    """
    Simple box-ray intersection:
      - ground plane
      - axis-aligned cubes (GTs)
    Returns:
      pts:  (N,3) hit positions
      its:  (N,) intensities
      vels: (N,3) velocities at hit
    """
    # angles
    az = torch.linspace(np.deg2rad(-sp.az_fov/2), np.deg2rad(sp.az_fov/2), 1024, device=DEVICE)
    el = torch.linspace(np.deg2rad(-sp.el_fov/2), np.deg2rad(sp.el_fov/2), 128, device=DEVICE)
    EL, AZ = torch.meshgrid(el, az, indexing='ij')
    rays = torch.stack(
        [torch.cos(EL)*torch.cos(AZ),
         torch.cos(EL)*torch.sin(AZ),
         torch.sin(EL)], dim=-1
    ).reshape(-1, 3)
    pos = torch.tensor([0., 0., sp.H], device=DEVICE)

    t_min = torch.full((rays.shape[0],), 100.0, device=DEVICE)
    hits_int = torch.zeros((rays.shape[0],), device=DEVICE)
    hits_vel = torch.zeros((rays.shape[0], 3), device=DEVICE)

    # ground plane z=0
    mask_g = (rays[:, 2] < -2e-2)
    t_g = -pos[2] / rays[:, 2]
    mask_valid_g = mask_g & (t_g > 0) & (t_g < t_min)
    t_min[mask_valid_g] = t_g[mask_valid_g]
    hits_int[mask_valid_g] = 100.0

    # cubes
    if gts:
        Cs = torch.stack([to_torch(gt['c']) for gt in gts])
        Ss = torch.stack([to_torch(gt['s']) for gt in gts])
        Vs = torch.stack([to_torch(gt['v']) for gt in gts])

        ro = pos.view(1, 1, 3)
        rd = rays.view(-1, 1, 3) + 1e-9
        t1 = (Cs - Ss/2 - ro) / rd
        t2 = (Cs + Ss/2 - ro) / rd
        tn = torch.max(torch.min(t1, t2), dim=-1)[0]
        tf = torch.min(torch.max(t1, t2), dim=-1)[0]
        mask_hit = (tn < tf) & (tn > 0)
        tn[~mask_hit] = np.inf

        min_t, min_idx = torch.min(tn, dim=1)
        mask_t = min_t < t_min
        t_min[mask_t] = min_t[mask_t]
        hits_int[mask_t] = 255.0
        hits_vel[mask_t] = Vs[min_idx[mask_t]]

    mask = hits_int > 0
    return pos + t_min[mask].unsqueeze(1) * rays[mask], hits_int[mask], hits_vel[mask]


# ---------------------------------------------------------------------
# FMCW SIM
# ---------------------------------------------------------------------
def fmcw_torch(pts, its, vels, sp: SystemParams):
    """
    Synthesizes RD map in dB.
    Returns (rd_db,)
    """
    M, N = sp.M, sp.N
    iq = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)
    if len(pts) == 0:
        return (np.zeros((M, N//2), np.float32),)

    P = pts - to_torch([0, 0, sp.H])
    R = torch.norm(P, dim=1)
    mask = R > 0.1
    P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
    amp = torch.where(its == 255, 1e6, 1e-1) / (R**2 + 1e-6)
    vr = torch.sum(P / R.unsqueeze(1) * vels, dim=1)

    t_f = torch.arange(N, device=DEVICE) / sp.fs
    t_s = torch.arange(M, device=DEVICE) * sp.T_chirp
    k_r = 2 * sp.slope / C0
    k_v = 2 / sp.lambda_m

    BATCH = 4096
    for i in range(0, len(R), BATCH):
        rb = R[i:i+BATCH]
        vrb = vr[i:i+BATCH]
        ab = amp[i:i+BATCH]
        phase = 2j * np.pi * (
            (k_r * rb[:, None, None]) * t_f[None, None, :]
            + (k_v * vrb[:, None, None]) * t_s[None, :, None]
        )
        iq += torch.sum(ab[:, None, None] * torch.exp(phase), dim=0)

    # noise + 1st-order MTI
    iq = iq + (torch.randn(M, N, device=DEVICE)
               + 1j*torch.randn(M, N, device=DEVICE)) * 1e-4
    iq[1:] -= iq[:-1].clone()
    iq[0] = 0

    w_r = torch.hann_window(N, device=DEVICE)
    w_d = torch.hann_window(M, device=DEVICE)
    iq = iq * (w_d[:, None] * w_r[None, :])

    RFFT = torch.fft.fft(iq, dim=1)
    RFFT = RFFT[:, :N//2]
    RD = torch.fft.fftshift(torch.fft.fft(RFFT, dim=0), dim=0)

    RD_mag = torch.abs(RD).clamp_min(1e-12)
    rd_db = 20 * torch.log10(RD_mag).cpu().numpy()
    return (rd_db,)


# ---------------------------------------------------------------------
# OTFS RADAR (toy)
# ---------------------------------------------------------------------
def otfs_torch(pts, its, vels, sp: SystemParams):
    M, N = sp.M, sp.N
    H = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)
    if len(pts) == 0:
        return (np.zeros((M, N), np.float32),)

    P = pts - to_torch([0, 0, sp.H])
    R = torch.norm(P, dim=1)
    mask = R > 0.1
    P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
    amp = torch.where(its == 255, 1e6, 1e-1) / (R**2 + 1e-6)
    vr = torch.sum(P / R.unsqueeze(1) * vels, dim=1)

    k_res = C0 / (2 * sp.fs)
    l_res = (sp.lambda_m / 2) * (sp.fs / (sp.M * sp.N))

    k = torch.clamp((R / k_res).long(), 0, N-1)
    l = torch.clamp((vr / l_res).long() + M//2, 0, M-1)
    H.view(-1).scatter_add_(0, (l*N + k).view(-1), amp.to(torch.complex64))
    H += (torch.randn(M, N, device=DEVICE)
          + 1j*torch.randn(M, N, device=DEVICE))*1e-4

    rd_db = 20 * torch.log10(torch.abs(H).clamp_min(1e-12)).cpu().numpy()
    return (rd_db,)


# ---------------------------------------------------------------------
# 2D MOVING SUM & CFAR
# ---------------------------------------------------------------------
def _moving_sum_2d(a, r, c):
    """Fast sliding-window sum with integral image (no SciPy)."""
    if r == 0 and c == 0:
        return a.copy()
    ap = np.pad(a, ((r, r), (c, c)), mode='edge')
    S = ap.cumsum(axis=0).cumsum(axis=1)
    H, W = a.shape
    s22 = S[2*r:2*r+H, 2*c:2*c+W]
    s02 = S[0:H,          2*c:2*c+W]
    s20 = S[2*r:2*r+H,    0:W]
    s00 = S[0:H,          0:W]
    return s22 - s02 - s20 + s00


def nms2d(arr, kernel=3):
    """Plain python NMS for 2D array."""
    k = max(3, int(kernel) | 1)
    pad = k // 2
    ap = np.pad(arr, ((pad, pad), (pad, pad)), mode='edge')
    max_nb = np.full_like(arr, -np.inf)
    for di in range(-pad, pad+1):
        for dj in range(-pad, pad+1):
            if di == 0 and dj == 0:
                continue
            view = ap[pad+di:pad+di+arr.shape[0], pad+dj:pad+dj+arr.shape[1]]
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
    return_stats=False
):
    """
    2D CA-CFAR on RD map in dB.
    """
    rd_lin = 10.0 ** (rd_db / 10.0)
    H, W = rd_lin.shape
    mid = H // 2

    # simple ground notch
    if notch_doppler_bins > 0:
        k = int(notch_doppler_bins)
        rd_lin[mid - k: mid + k + 1, :] = np.minimum(
            rd_lin[mid - k: mid + k + 1, :],
            np.percentile(rd_lin, 10)
        )

    Tr, Tc = train
    Gr, Gc = guard
    tot = _moving_sum_2d(rd_lin, Tr + Gr, Tc + Gc)
    gpl = _moving_sum_2d(rd_lin, Gr, Gc)
    train_sum = tot - gpl

    n_train = (2*(Tr+Gr)+1)*(2*(Tc+Gc)+1) - (2*Gr+1)*(2*Gc+1)
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
            idx = np.argpartition(-vals, max_peaks-1)[:max_peaks]
            keep = np.zeros_like(det, dtype=bool)
            keep[yy[idx], xx[idx]] = True
            det = keep

    if return_stats:
        return det, noise, snr_db
    return det

# ------------------------------------------------------------
# Unified SNR + Local Max + Clustering Peak Detector (NEW)
# ------------------------------------------------------------
from scipy.ndimage import maximum_filter, label

def detect_peaks_snr_localmax(
    z_complex,
    snr_thr_db=12.0,
    local_max_neigh=3,
    cluster=True,
    cluster_min_size=1,
):
    """
    Generic detector for FMCW RD or OTFS Delay–Doppler maps.
    Returns:
       dets: list of dict {'i','j','val_db','snr_db'}
       z_db: full dB map
       noise_floor: estimated noise level
    """
    z_abs = np.abs(z_complex)
    z_db  = 20*np.log10(z_abs + 1e-12)

    # 1) Estimate noise floor
    noise_floor = np.percentile(z_db, 20)

    # 2) Threshold by SNR
    cand = z_db > (noise_floor + snr_thr_db)

    # 3) Local maxima
    max_filt = maximum_filter(z_db, size=local_max_neigh, mode="nearest")
    local_max = (z_db == max_filt)
    det_mask  = cand & local_max

    # If no clustering wanted → simple outputs
    if not cluster:
        ys, xs = np.where(det_mask)
        dets = []
        for y,x in zip(ys,xs):
            val = z_db[y,x]
            snr = val - noise_floor
            dets.append({"i":int(y),"j":int(x),"val_db":float(val),"snr_db":float(snr)})
        return dets, z_db, noise_floor

    # 4) Clustering: keep strongest per blob
    labels, num = label(det_mask)
    dets = []
    for lab in range(1, num+1):
        mask = labels == lab
        if mask.sum() < cluster_min_size:
            continue
        vals = z_db[mask]
        idx  = np.argmax(vals)
        y,x = np.argwhere(mask)[idx]
        val = z_db[y,x]
        snr = val - noise_floor
        dets.append({"i":int(y),"j":int(x),"val_db":float(val),"snr_db":float(snr)})

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


@torch.no_grad()
def viz_rd_2d_compare(path, rd_f_db, rd_o_db, gts, sp: SystemParams):
    """
    Side-by-side 2D maps for FMCW & OTFS.
    Returns:
        det_f_mask, ra_f, va_f, noise_f, snr_f
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    pos = np.array([0, 0, sp.H])

    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    rd_f = np.asarray(rd_f_db).astype(np.float32)
    rd_o = np.asarray(rd_o_db).astype(np.float32)

    noise_level = np.median(rd_f)
    noise_f = np.full_like(rd_f, noise_level, dtype=np.float32)
    snr_f = rd_f - noise_f

    det_f_mask = cfar2d_ca(rd_f, pfa=1e-4)

    for i, (rd_raw, ra, va, name) in enumerate([
        (rd_f, ra_f, va_f, "FMCW"),
        (rd_o, ra_o, va_o, "OTFS")
    ]):
        rd = _db_scale(rd_raw)
        lo, hi = np.percentile(rd, [5, 99.7])
        im = ax[i].pcolormesh(
            ra, va, rd, cmap='turbo', vmin=lo, vmax=hi, shading='auto'
        )
        cbar = plt.colorbar(im, ax=ax[i])
        cbar.set_label("Amplitude (dB)")

        for gt in gts:
            P = np.array(gt['c']) - pos
            r = np.linalg.norm(P)
            v = np.dot(P / (np.linalg.norm(P) + 1e-9), gt['v'])
            if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
                ax[i].plot(r, v, 'wx', ms=10, mew=2)

        ax[i].set_title(f"{name} Range–Doppler")
        ax[i].set_xlabel("Range (m)")
        ax[i].set_xlim(0, ra[-1])
        ax[i].set_ylim(va[0], va[-1])
        ax[i].set_ylabel("Velocity (m/s)")

        if name == "FMCW":
            ys, xs = np.where(det_f_mask)
            if ys.size > 0:
                ax[i].scatter(
                    ra[xs], va[ys],
                    s=40, facecolors='none', edgecolors='cyan',
                    linewidths=1.0, label="CFAR"
                )
                ax[i].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return det_f_mask, ra_f, va_f, noise_f, snr_f


def viz_rd_3d_compare(path, rd_f, rd_o, gts, sp: SystemParams):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(18, 8))
    pos = np.array([0, 0, sp.H])
    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    for i, (rd_raw, ra, va, name) in enumerate(
        [(rd_f, ra_f, va_f, "FMCW"), (rd_o, ra_o, va_o, "OTFS")]
    ):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        rd = _db_scale(rd_raw)
        lo, hi = np.percentile(rd, [5, 99.7])
        R, V = np.meshgrid(ra, va)
        surf = np.clip(rd, lo, hi)
        ax.plot_surface(
            R, V, surf, cmap='viridis',
            rstride=2, cstride=2, linewidth=0, antialiased=True
        )
        for gt in gts:
            P = np.array(gt['c']) - pos
            r = np.linalg.norm(P)
            v = np.dot(P/r, gt['v'])
            if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
                ax.scatter([r], [v], [hi], c='r', marker='x', s=80)
        ax.set_title(f"{name} 3D (dB)")
        ax.set_xlabel("Range(m)")
        ax.set_ylabel("Vel(m/s)")
        ax.set_xlim(0, ra[-1])
        ax.set_ylim(va[0], va[-1])
        ax.set_zlim(lo, hi)
        ax.view_init(35, -120)

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def viz_rd_3d_with_dets(
    path, rd_raw, ra, va, det_mask, gts, sp, title="RD with Detections & GT"
):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    rd = _db_scale(rd_raw)
    lo, hi = np.percentile(rd, [5, 99.7])
    R, V = np.meshgrid(ra, va)
    surf = np.clip(rd, lo, hi)

    ax.plot_surface(
        R, V, surf, cmap='viridis',
        rstride=2, cstride=2,
        linewidth=0, antialiased=True, alpha=0.95
    )

    if det_mask is not None and det_mask.any():
        ys, xs = np.where(det_mask)
        ax.scatter(
            ra[xs], va[ys], surf[ys, xs],
            s=18, c='cyan', depthshade=False, label='Detections'
        )

    pos = np.array([0, 0, sp.H])
    for gt in gts:
        P = np.array(gt['c']) - pos
        r = np.linalg.norm(P)
        v = np.dot(P/r, gt['v'])
        if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
            ax.scatter([r], [v], [hi], c='r', marker='x', s=100, label='GT')

    ax.set_title(title + " (dB)")
    ax.set_xlabel("Range/Delay (m)")
    ax.set_ylabel("Doppler (m/s)")
    ax.set_xlim(0, ra[-1])
    ax.set_ylim(va[0], va[-1])
    ax.set_zlim(lo, hi)
    ax.view_init(35, -120)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------
# DETECTIONS / BEV / METRICS
# ---------------------------------------------------------------------
def extract_detections(rd_db, det_mask, ra, va, noise_db=None, snr_db=None):
    yy, xx = np.where(det_mask)
    dets = []
    for y, x in zip(yy, xx):
        det = {
            'r': float(ra[x]),
            'v': float(va[y]),
            'mag_db': float(rd_db[y, x])
        }
        if snr_db is not None:
            det['snr_db'] = float(snr_db[y, x])
        if noise_db is not None:
            det['noise_db'] = float(noise_db[y, x])
        dets.append(det)
    return dets


def _gt_rv_az(gts, sp: SystemParams):
    """
    Map GT cubes to (range, radial velocity, azimuth, size).
    """
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
        az = float(np.arctan2(c[1], c[0]))  # ground-plane azimuth
        out.append({'c': c, 'r': float(r), 'v': vr, 'az': az, 's': s})
    return out


def _match_dets_to_gts(dets, gtinfo, w_r=1.0, w_v=0.5):
    """
    Simple nearest-neighbor matching in (r,v) with one-to-one constraint.
    """
    used = set()
    pairs = []
    unpaired = []

    for d in dets:
        best_i = None
        best_cost = 1e12
        for gi, g in enumerate(gtinfo):
            cost = w_r * abs(d['r'] - g['r']) + w_v * abs(d['v'] - g['v'])
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
    FP should be handled externally if needed.
    """
    TP = len(pairs)
    FP = 0
    er_r = []
    er_v = []

    for det, gi, _ in pairs:
        g = gtinfo[gi]
        er_r.append(abs(det['r'] - g['r']))
        er_v.append(abs(det['v'] - g['v']))

    metrics = dict(
        TP=TP, FP=FP,
        er_r=er_r, er_v=er_v
    )
    return metrics, er_r, er_v


# ---------------------------------------------------------------------
# HEATMAP GT, RADAR DL BASICS
# ---------------------------------------------------------------------
def _rd_normalize(rd_db, top_p=99.5, dyn_db=40.0):
    top = np.percentile(rd_db, top_p)
    rd = np.clip(rd_db, top-dyn_db, top)
    rd = (rd - (top-dyn_db)) / dyn_db
    return rd.astype(np.float32)


def _heatmap_from_gts(shape, ra, va, gts, sp, sigma_pix=(2.0, 2.0)):
    H, W = shape
    pos = np.array([0, 0, sp.H])
    yy, xx = np.mgrid[0:H, 0:W]
    hm = np.zeros((H, W), np.float32)

    for gt in gts:
        P = np.array(gt['c']) - pos
        r = np.linalg.norm(P)
        v = np.dot(P / (np.linalg.norm(P) + 1e-9), gt['v'])
        if not (0 <= r <= ra[-1] and va[0] <= v <= va[-1]):
            continue
        ix = np.searchsorted(ra, r)
        ix = np.clip(ix, 0, W-1)
        iy = np.searchsorted(va, v)
        iy = np.clip(iy, 0, H-1)
        sx, sy = sigma_pix[1], sigma_pix[0]
        g = np.exp(-((xx-ix)**2/(2*sx**2) + (yy-iy)**2/(2*sy**2)))
        hm = np.maximum(hm, g)
    return hm


class UNetLite(nn.Module):
    def __init__(self, in_ch=1, ch=32):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU()
        )
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(
            nn.Conv2d(ch, 2*ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(2*ch, 2*ch, 3, padding=1), nn.ReLU()
        )
        self.p2 = nn.MaxPool2d(2)
        self.b = nn.Sequential(
            nn.Conv2d(2*ch, 4*ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(4*ch, 4*ch, 3, padding=1), nn.ReLU()
        )
        self.u2 = nn.ConvTranspose2d(4*ch, 2*ch, 2, stride=2)
        self.d2 = nn.Sequential(
            nn.Conv2d(4*ch, 2*ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(2*ch, 2*ch, 3, padding=1), nn.ReLU()
        )
        self.u1 = nn.ConvTranspose2d(2*ch, ch, 2, stride=2)
        self.d1 = nn.Sequential(
            nn.Conv2d(2*ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU()
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
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = p*targets + (1-p)*(1-targets)
    w = alpha*targets + (1-alpha)*(1-targets)
    loss = (w * (1-pt).pow(gamma) * bce).mean()
    return loss


# ---------------------------------------------------------------------
# RADAR DL INFERENCE → PEAKS
# ---------------------------------------------------------------------
@torch.no_grad()
def rd_dl_infer_to_points(logits, ra, va, thr=0.35, max_peaks=64):
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask = prob > thr
    if not mask.any():
        return []
    if SCIPY:
        mxf = ndi.maximum_filter(prob, size=3)
        peaks = (prob == mxf) & mask
    else:
        mxf = prob.copy()
        peaks = mask
    yy, xx = np.where(peaks)
    if len(yy) > max_peaks:
        vals = prob[yy, xx]
        idx = np.argpartition(-vals, max_peaks-1)[:max_peaks]
        yy, xx = yy[idx], xx[idx]
    dets = []
    for y, x in zip(yy, xx):
        dy, dx = _subpixel_quadratic(prob, int(y), int(x))
        r, v = _rv_from_idx_with_subpix(int(y), int(x), dy, dx, ra, va)
        dets.append({'r': float(r), 'v': float(v), 'score': float(prob[y, x])})
    return dets


# ---------------------------------------------------------------------
# COMMUNICATIONS (QPSK / OFDM / OTFS + BER)
# ---------------------------------------------------------------------
def _rand_bits(n, rng):
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def _qpsk_gray_mod(bits):
    b0 = bits[..., 0]
    b1 = bits[..., 1]
    I = np.where((b0==0) & (b1==0),  1.0,
        np.where((b0==0) & (b1==1), -1.0,
        np.where((b0==1) & (b1==1), -1.0,  1.0)))
    Q = np.where((b0==0) & (b1==0),  1.0,
        np.where((b0==0) & (b1==1),  1.0,
        np.where((b0==1) & (b1==1), -1.0, -1.0)))
    s = (I + 1j*Q) / np.sqrt(2.0)
    return s


def _qpsk_gray_demod(symbols):
    I = np.real(symbols)
    Q = np.imag(symbols)
    b0 = (Q < 0).astype(np.uint8)
    b1 = (I < 0).astype(np.uint8)
    return np.stack([b0, b1], axis=-1)


def _awgn(x, ebn0_db, bits_per_sym, cp_ratio=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    r_eff = bits_per_sym * (1.0 / (1.0 + cp_ratio))
    Es = 1.0
    Eb = Es / r_eff
    N0 = Eb / ebn0
    sigma2 = N0
    n = (rng.normal(scale=np.sqrt(sigma2/2), size=x.shape)
         + 1j*rng.normal(scale=np.sqrt(sigma2/2), size=x.shape))
    return x + n


# ---------- OFDM ----------
def ofdm_mod(bits, Nfft=256, cp_len=32, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bits = bits.reshape(-1, 2)
    nsym = bits.shape[0] // Nfft
    bits = bits[:nsym*Nfft].reshape(nsym, Nfft, 2)
    syms = _qpsk_gray_mod(bits)
    x = np.fft.ifft(syms, n=Nfft, axis=1, norm='ortho')
    if cp_len > 0:
        cp = x[:, -cp_len:]
        x_cp = np.concatenate([cp, x], axis=1)
    else:
        x_cp = x
    return x_cp, syms


def ofdm_demod(rx, Nfft=256, cp_len=32):
    if cp_len > 0:
        rx = rx[:, cp_len:cp_len+Nfft]
    Sy = np.fft.fft(rx, n=Nfft, axis=1, norm='ortho')
    return Sy


def ofdm_tx_rx_ber(ebn0_db, Nfft=256, cp_len=32, n_ofdm_sym=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bits_per_sym = 2
    nbits = n_ofdm_sym * Nfft * bits_per_sym
    bits = _rand_bits(nbits, rng)
    tx, _ = ofdm_mod(bits, Nfft=Nfft, cp_len=cp_len, rng=rng)
    cp_ratio = cp_len / Nfft
    rx = _awgn(tx, ebn0_db, bits_per_sym=bits_per_sym,
               cp_ratio=cp_ratio, rng=rng)
    Sy = ofdm_demod(rx, Nfft=Nfft, cp_len=cp_len)
    hard_bits = _qpsk_gray_demod(Sy.reshape(-1))
    hard_bits = hard_bits.reshape(-1, 2)
    bits_hat = hard_bits[:len(bits)].reshape(-1)
    ber = np.mean(bits != bits_hat)
    return ber


# ---------- OTFS (simple) ----------
def otfs_mod(bits, M=64, N=256, cp_len=32, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bits = bits.reshape(M*N, 2)[:M*N].reshape(M, N, 2)
    X_dd = _qpsk_gray_mod(bits)

    X_tf = np.fft.ifft(
        np.fft.fft(X_dd, n=N, axis=1, norm='ortho'),
        n=M, axis=0, norm='ortho'
    )
    tx = np.fft.ifft(X_tf, n=N, axis=1, norm='ortho')
    if cp_len > 0:
        cp = tx[:, -cp_len:]
        tx = np.concatenate([cp, tx], axis=1)
    return tx, X_dd


def otfs_demod(rx, M=64, N=256, cp_len=32):
    if cp_len > 0:
        rx = rx[:, cp_len:cp_len+N]
    Y_tf = np.fft.fft(rx, n=N, axis=1, norm='ortho')
    Y_dd = np.fft.ifft(
        np.fft.fft(Y_tf, n=M, axis=0, norm='ortho'),
        n=N, axis=1, norm='ortho'
    )
    return Y_dd


def otfs_tx_rx_ber(ebn0_db, M=64, N=256, cp_len=32, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bits_per_sym = 2
    nbits = M * N * bits_per_sym
    bits = _rand_bits(nbits, rng)
    tx, Xdd = otfs_mod(bits, M=M, N=N, cp_len=cp_len, rng=rng)
    cp_ratio = cp_len / N
    rx = _awgn(tx, ebn0_db, bits_per_sym=bits_per_sym,
               cp_ratio=cp_ratio, rng=rng)
    Ydd = otfs_demod(rx, M=M, N=N, cp_len=cp_len)
    hard_bits = _qpsk_gray_demod(Ydd.reshape(-1))
    hard_bits = hard_bits.reshape(-1, 2)
    bits_hat = hard_bits[:len(bits)].reshape(-1)
    ber = np.mean(bits != bits_hat)
    return ber


# ---------- BER sweep ----------
def run_ber_sweep_and_plot(
    path_png,
    ebn0_db_list=np.arange(0, 21, 2),
    ofdm_cfg=dict(Nfft=256, cp_len=32, n_ofdm_sym=400),
    otfs_cfg=dict(M=64, N=256, cp_len=32),
    rng_seed=1234
):
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
    plt.semilogy(ebn0_db_list, ber_ofdm + 1e-12, marker='o',
                 label='FMCW-Comm (OFDM, QPSK)')
    plt.semilogy(ebn0_db_list, ber_otfs + 1e-12, marker='s',
                 label='OTFS-Comm (QPSK)')
    plt.semilogy(ebn0_db_list, ber_theory_qpsk + 1e-12, linestyle='--',
                 label='Theory QPSK (AWGN)')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('BER vs Eb/N0: OFDM vs OTFS vs Theory')

    for i in range(0, len(ebn0_db_list), 2):
        x = ebn0_db_list[i]
        yo = ber_ofdm[i]
        yt = ber_otfs[i]
        plt.text(x, yo*1.15 + 1e-14, f"{yo:.2e}", fontsize=8,
                 ha='center', va='bottom')
        plt.text(x, yt*0.85 + 1e-14, f"{yt:.2e}", fontsize=8,
                 ha='center', va='top')

    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=170, bbox_inches='tight')
    plt.close()

    return ebn0_db_list, ber_ofdm, ber_otfs, ber_theory_qpsk


# ---------------------------------------------------------------------
# COMM DL: BATCH GEN + DEMAPPER MODEL + BER
# ---------------------------------------------------------------------
def _bits_to_qpsk_grid(bits, H, W):
    bits = bits.reshape(H*W, 2)
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
    if rng is None:
        rng = np.random.default_rng()
    bits_per_sym = 2
    H, W = n_sym, Nfft
    x_list, y_list = [], []
    for _ in range(batch):
        bits = _rand_bits(H*W*bits_per_sym, rng)
        Xf = _bits_to_qpsk_grid(bits, H, W)
        tx = np.fft.ifft(Xf, n=W, axis=1, norm='ortho')
        if cp_len > 0:
            tx = np.concatenate([tx[:, -cp_len:], tx], axis=1)
        cp_ratio = cp_len / W
        rx = _awgn(tx, ebn0_db, bits_per_sym=2,
                   cp_ratio=cp_ratio, rng=rng)
        if cp_len > 0:
            rx = rx[:, cp_len:cp_len+W]
        Yf = np.fft.fft(rx, n=W, axis=1, norm='ortho')
        x = _grid_feats(Yf, use_mag_phase=False)
        y = bits.reshape(H, W, 2).transpose(2, 0, 1)
        x_list.append(x)
        y_list.append(y.astype(np.float32))
    X = torch.from_numpy(np.stack(x_list))
    Y = torch.from_numpy(np.stack(y_list))
    return X, Y


def comm_dl_gen_batch_OTFS(ebn0_db, batch=8, M=64, N=256, cp_len=32, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bits_per_sym = 2
    x_list, y_list = [], []
    for _ in range(batch):
        bits = _rand_bits(M*N*bits_per_sym, rng)
        tx, _ = otfs_mod(bits, M=M, N=N, cp_len=cp_len, rng=rng)
        cp_ratio = cp_len / N
        rx = _awgn(tx, ebn0_db, bits_per_sym=2,
                   cp_ratio=cp_ratio, rng=rng)
        Ydd = otfs_demod(rx, M=M, N=N, cp_len=cp_len)
        x = _grid_feats(Ydd, use_mag_phase=False)
        y = bits.reshape(M, N, 2).transpose(2, 0, 1)
        x_list.append(x)
        y_list.append(y.astype(np.float32))
    X = torch.from_numpy(np.stack(x_list))
    Y = torch.from_numpy(np.stack(y_list))
    return X, Y


class CommDemapperCNN(nn.Module):
    def __init__(self, in_ch=2, width=32, depth=3, out_ch=2):
        super().__init__()
        layers = [nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU()]
        for _ in range(depth-1):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, out_ch, 1)

    def forward(self, x):
        return self.head(self.backbone(x))


def train_comm_demap(
    model, gen_batch_fn, cfg, snr_min=0, snr_max=18,
    epochs=5, steps_per_epoch=200, lr=3e-4, device=None, tag="OFDM"
):
    device = device or DEVICE
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        model.train()
        loss_ep = 0
        for _ in range(steps_per_epoch):
            eb = np.random.uniform(snr_min, snr_max)
            X, Y = gen_batch_fn(eb, **cfg)
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
    Computes BER across Eb/N0 values.
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
    root = Path(root)
    tr_dir = root/"radar"/"train"
    va_dir = root/"radar"/"val"
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


def simulate_dataset(
    out_dir,
    sp: SystemParams,
    n_train=1500,
    n_val=300,
    seed=2025,
    snr_list=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20),
):
    """
    Simple dataset:
      radar/train, radar/val: rd_f_db, heatmap, gts
      comm: train_spec.json, val_spec.json
    """
    out = Path(out_dir)
    (out/"radar"/"train").mkdir(parents=True, exist_ok=True)
    (out/"radar"/"val").mkdir(parents=True, exist_ok=True)
    (out/"comm").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    def _rand_gts_two(sp, rng):
        k = rng.integers(1, 3)
        gts = []
        for _ in range(k):
            r = rng.uniform(8, 48)
            az = rng.uniform(
                -np.deg2rad(sp.az_fov/2), np.deg2rad(sp.az_fov/2)
            )
            x = r*np.cos(az)
            y = r*np.sin(az)
            vx = rng.uniform(-20, 20)
            vy = rng.uniform(-6, 6)
            gts.append({
                'c': [float(x), float(y), 1.0],
                's': [4.0, 2.0, 2.0],
                'v': [float(vx), float(vy), 0.0]
            })
        return gts

    def _one_sample(idx, split):
        gts = _rand_gts_two(sp, rng)
        pts, its, vels = raycast_torch(sp, gts)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        (rd_f_db,) = fmcw_torch(pts, its, vels, sp)
        ra_f, va_f = sp.fmcw_axes()
        hm = _heatmap_from_gts(rd_f_db.shape, ra_f, va_f, gts, sp)
        save_path = out/"radar"/split/f"{idx:06d}.npz"
        np.savez_compressed(
            save_path,
            rd_f_db=rd_f_db.astype(np.float32),
            heatmap=hm.astype(np.float32),
            gts=json.dumps(gts)
        )

    print(f"[DATA] Simulating radar {n_train} train + {n_val} val samples → {out_dir}")
    for i in tqdm(range(n_train), desc="radar-train"):
        _one_sample(i, "train")
    for i in tqdm(range(n_val), desc="radar-val"):
        _one_sample(i, "val")

    def _make_comm_spec(n_items):
        specs = []
        for i in range(n_items):
            specs.append({
                "seed": int(rng.integers(0, 2**31-1)),
                "ebn0_db": float(snr_list[i % len(snr_list)])
            })
        return specs

    comm_train_spec = _make_comm_spec(max(n_train//4, 400))
    comm_val_spec = _make_comm_spec(max(n_val//4, 100))
    with open(out/"comm"/"train_spec.json", "w") as f:
        json.dump(comm_train_spec, f, indent=2)
    with open(out/"comm"/"val_spec.json", "w") as f:
        json.dump(comm_val_spec, f, indent=2)
    print("[DATA] Done.")


def dataset_exists(root: str) -> bool:
    root = Path(root)
    have_radar = (
        (root/"radar"/"train").exists()
        and any((root/"radar"/"train").glob("*.npz"))
        and (root/"radar"/"val").exists()
        and any((root/"radar"/"val").glob("*.npz"))
    )
    have_comm = (
        (root/"comm"/"train_spec.json").exists()
        and (root/"comm"/"val_spec.json").exists()
    )
    return have_radar and have_comm


def simulate_if_missing(out_dir, sp, **kwargs):
    if dataset_exists(out_dir):
        print(f"[DATA] Found existing dataset under {out_dir} — skip simulation.")
        return
    print(f"[DATA] Simulating dataset → {out_dir}")
    simulate_dataset(out_dir=out_dir, sp=sp, **kwargs)


# ---------------------------------------------------------------------
# RADAR TRAINING (UNET-LITE)
# ---------------------------------------------------------------------
def train_radar_model(
    sp, epochs=5, batch=6, lr=1e-3, n_train=800, n_val=200, device=None
):
    device = device or DEVICE
    net = UNetLite().to(device)
    ds_tr = RadarSimDataset(sp, n_train, rng_seed=2025)
    ds_va = RadarSimDataset(sp, n_val, rng_seed=2026)
    dl_tr = torch.utils.data.DataLoader(
        ds_tr, batch_size=batch, shuffle=True, num_workers=0
    )
    dl_va = torch.utils.data.DataLoader(
        ds_va, batch_size=batch, shuffle=False, num_workers=0
    )
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    best = {'loss': 1e9, 'state': None}
    for ep in range(1, epochs+1):
        net.train()
        loss_tr = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = net(x)
            loss = focal_bce_with_logits(logits, y, alpha=0.25, gamma=2.0)
            loss.backward()
            opt.step()
            loss_tr += loss.item()*x.size(0)
        loss_tr /= len(ds_tr)
        net.eval()
        loss_va = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                logits = net(x)
                loss = focal_bce_with_logits(logits, y)
                loss_va += loss.item()*x.size(0)
        loss_va /= len(ds_va)
        print(f"[RadarDL] epoch {ep}/{epochs}  train {loss_tr:.4f}  val {loss_va:.4f}")
        if loss_va < best['loss']:
            best = {'loss': loss_va,
                    'state': {k: v.cpu() for k, v in net.state_dict().items()}}
    net.load_state_dict(best['state'])
    return net


class RadarSimDataset(torch.utils.data.Dataset):
    def __init__(
        self, sp, n_items=2000, rng_seed=123, min_targets=1, max_targets=3
    ):
        self.sp = sp
        self.n = n_items
        self.rng = np.random.default_rng(rng_seed)
        self.min_t = min_targets
        self.max_t = max_targets
        self.ra, self.va = sp.fmcw_axes()

    def _rand_gts(self):
        k = self.rng.integers(self.min_t, self.max_t+1)
        gts = []
        for _ in range(k):
            r = self.rng.uniform(8, 45)
            az = self.rng.uniform(
                -np.deg2rad(self.sp.az_fov/2),
                np.deg2rad(self.sp.az_fov/2)
            )
            x = r*np.cos(az)
            y = r*np.sin(az)
            vx = self.rng.uniform(-20, 20)
            vy = self.rng.uniform(-5, 5)
            gts.append({
                'c': [x, y, 1.0],
                's': [4, 2, 2],
                'v': [vx, vy, 0]
            })
        return gts

    def __getitem__(self, idx):
        gts = self._rand_gts()
        pts, its, vels = raycast_torch(self.sp, gts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        (rd_db,) = fmcw_torch(pts, its, vels, self.sp)
        rd = _rd_normalize(rd_db)
        hm = _heatmap_from_gts(rd.shape, self.ra, self.va, gts, self.sp)
        x = torch.from_numpy(rd[None, ...])
        y = torch.from_numpy(hm[None, ...])
        return x, y

    def __len__(self):
        return self.n


# ---------------------------------------------------------------------
# EVALUATION + VISUALIZATION (FMCW / OTFS, CFAR vs DL, COMM DL vs BASE)
# ---------------------------------------------------------------------
def evaluate_and_visualize(
    out_dir,
    sp: SystemParams,
    radar_net, ofdm_model, otfs_model,
    gts_eval=None
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if gts_eval is None:
        gts_eval = [
            {'c': [20,  0, 1], 's': [4, 2, 2], 'v': [12,  0, 0]},
            {'c': [50, -5, 1], 's': [5, 3, 3], 'v': [-18,  5, 0]}
        ]

    print("[EVAL] Simulating eval scene...")
    pts, its, vels = raycast_torch(sp, gts_eval)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    (rd_f_db,) = fmcw_torch(pts, its, vels, sp)
    (rd_o_db,) = otfs_torch(pts, its, vels, sp)

    det_f_mask, ra_f, va_f, noise_f, snr_f = viz_rd_2d_compare(
        out/"compare_2d.pdf", rd_f_db, rd_o_db, gts_eval, sp
    )
    viz_rd_3d_compare(out/"compare_3d.pdf", rd_f_db, rd_o_db, gts_eval, sp)
    viz_rd_3d_with_dets(
        out/"fmcw_3d_with_dets.pdf",
        rd_f_db, ra_f, va_f, det_f_mask, gts_eval, sp,
        title="FMCW RD with Detections & GT"
    )

    cfar_otfs_cfg = dict(
        train=(10, 8), guard=(2, 2), pfa=1e-4,
        min_snr_db=6.0, notch_doppler_bins=0,
        apply_nms=True, max_peaks=80
    )
    det_o_mask = cfar2d_ca(rd_o_db, **cfar_otfs_cfg)
    ra_o, va_o = sp.otfs_axes()
    viz_rd_3d_with_dets(
        out/"otfs_3d_with_dets.pdf",
        rd_o_db, ra_o, va_o, det_o_mask, gts_eval, sp,
        title="OTFS Delay–Doppler with Detections & GT"
    )

    dets_f = extract_detections(rd_f_db, det_f_mask, ra_f, va_f, snr_db=snr_f)
    dets_o = extract_detections(rd_o_db, det_o_mask, ra_o, va_o)
    # BEV etc. can be added here as needed (you already have those helpers).

    # Radar DL vs CFAR (simple metrics on this scene)
    gtinfo = _gt_rv_az(gts_eval, sp)
    pairs_cfar, unp_cfar = _match_dets_to_gts(dets_f, gtinfo, w_r=1.0, w_v=0.5)
    metrics_cfar, _, _ = _compute_metrics_from_pairs(pairs_cfar, gtinfo, sp)

    rd_in = torch.from_numpy(_rd_normalize(rd_f_db))[None, None].to(DEVICE)
    radar_net.eval()
    with torch.no_grad():
        logits = radar_net(rd_in)
    dets_dl = rd_dl_infer_to_points(
        logits, ra_f, va_f, thr=0.40, max_peaks=32
    )
    pairs_dl, unp_dl = _match_dets_to_gts(dets_dl, gtinfo, w_r=1.0, w_v=0.5)
    metrics_dl, _, _ = _compute_metrics_from_pairs(pairs_dl, gtinfo, sp)
    print("[Radar] CFAR metrics:", metrics_cfar)
    print("[Radar] DL   metrics:", metrics_dl)

    # BER sweeps
    print("[EVAL] BER sweeps...")
    eb_axis, ber_ofdm, ber_otfs, ber_theory = run_ber_sweep_and_plot(
        out/"ber_compare.pdf",
        ebn0_db_list=np.arange(0, 21, 2),
        ofdm_cfg=dict(Nfft=256, cp_len=32, n_ofdm_sym=800),
        otfs_cfg=dict(M=64, N=256, cp_len=32),
        rng_seed=2025
    )

    ofdm_cfg = dict(Nfft=256, cp_len=32, n_sym=8, batch=8)
    otfs_cfg = dict(M=64, N=256, cp_len=32, batch=6)

    ber_ofdm_dl = comm_demap_ber_curve(
        ofdm_model, comm_dl_gen_batch_OFDM, ofdm_cfg, eb_axis
    )
    ber_otfs_dl = comm_demap_ber_curve(
        otfs_model, comm_dl_gen_batch_OTFS, otfs_cfg, eb_axis
    )

    plt.figure(figsize=(8, 6))
    plt.semilogy(eb_axis, ber_ofdm+1e-12, 'o-', label='OFDM baseline (hard QPSK)')
    plt.semilogy(eb_axis, ber_otfs+1e-12, 's-', label='OTFS baseline (hard QPSK)')
    plt.semilogy(eb_axis, ber_ofdm_dl+1e-12, 'o--', label='OFDM DL demapper')
    plt.semilogy(eb_axis, ber_otfs_dl+1e-12, 's--', label='OTFS DL demapper')
    plt.semilogy(eb_axis, ber_theory+1e-12, 'k:', label='Theory QPSK (AWGN)')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('Comm BER: Baseline vs DL')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out/"ber_compare_with_dl.pdf", dpi=170)
    plt.close()

    with open(out/"radar_metrics.json", "w") as f:
        json.dump({"cfar": metrics_cfar, "dl": metrics_dl}, f, indent=2)

    print("[EVAL] Done.")


# ---------------------------------------------------------------------
# HIGH-LEVEL TRAINING / ENTRYPOINTS
# ---------------------------------------------------------------------
def new_training():
    root = "./output/isac_c4"
    Path(root).mkdir(parents=True, exist_ok=True)
    ckpts = Path(root) / "checkpoints"
    ckpts.mkdir(exist_ok=True)
    sp = SystemParams()

    simulate_if_missing(
        root, sp,
        n_train=1500, n_val=300, seed=2025,
        snr_list=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    )

    radar_net = train_radar_model(
        sp, epochs=6, batch=6, lr=1e-3, n_train=1200, n_val=300
    )

    # save radar ckpt
    torch.save(
        {
            "epoch": 0,
            "model": radar_net.state_dict(),
            "optim": None,
            "val_loss": 0.0,
        },
        ckpts/"radar_unet.pt"
    )
    torch.save(radar_net.state_dict(), ckpts/"radar_unet_best_only.pt")

    # Comm demappers
    print("Training OFDM demapper DL...")
    ofdm_cfg = dict(Nfft=256, cp_len=32, n_sym=8, batch=8)
    ofdm_model = CommDemapperCNN(in_ch=2)
    ofdm_model = train_comm_demap(
        ofdm_model, comm_dl_gen_batch_OFDM,
        ofdm_cfg, snr_min=0, snr_max=18,
        epochs=5, steps_per_epoch=200, tag="OFDM"
    )

    print("Training OTFS demapper DL...")
    otfs_cfg = dict(M=64, N=256, cp_len=32, batch=6)
    otfs_model = CommDemapperCNN(in_ch=2)
    otfs_model = train_comm_demap(
        otfs_model, comm_dl_gen_batch_OTFS,
        otfs_cfg, snr_min=0, snr_max=18,
        epochs=5, steps_per_epoch=200, tag="OTFS"
    )

    torch.save({"epoch": 5, "model": ofdm_model.state_dict()},
               ckpts/"comm_ofdm.pt")
    torch.save({"epoch": 5, "model": otfs_model.state_dict()},
               ckpts/"comm_otfs.pt")

    evaluate_and_visualize(
        out_dir=root,
        sp=sp,
        radar_net=radar_net,
        ofdm_model=ofdm_model,
        otfs_model=otfs_model,
        gts_eval=[
            {'c':[20,  0, 1], 's':[4,2,2], 'v':[ 12,  0, 0]},
            {'c':[50, -5, 1], 's':[5,3,3], 'v':[-18,  5, 0]}
        ]
    )


def load_radar_from_ckpt(root, device=None, prefer_best=True):
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
        raise FileNotFoundError(
            f"No radar checkpoint found in {ckpt_dir}."
        )
    net.eval()
    return net


def run_validation_from_root(
    root,
    dl_thr_sweep=(0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    cfar_pfa_sweep=(1e-2, 1e-3, 1e-4, 1e-5),
    do_otfs=True,
    max_samples=None
):
    """
    Simple wrapper: loads radar model + runs eval_radar_on_val_set (you can
    plug your richer val pipeline in).
    """
    root = Path(root)
    sp = SystemParams()
    radar_net = load_radar_from_ckpt(root, device=DEVICE, prefer_best=True)
    out_dir = root/"val_eval"
    out_dir.mkdir(exist_ok=True)
    print("[VAL] (Placeholder) – hook your full eval pipeline here.")
    # you can integrate eval_radar_on_val_set / eval_radar_on_val_set_dual here


def launch_mdmt_training():
    """
    Entry point for multi-domain / multi-task training. This calls the
    integrated RadarCommNet training you had; here we just reuse new_training.
    """
    new_training()


if __name__ == "__main__":
    # Example: run the high-level training pipeline
    new_training()