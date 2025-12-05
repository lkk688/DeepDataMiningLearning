import os, numpy as np, matplotlib.pyplot as plt
import torch
from dataclasses import dataclass

try:
    import scipy.ndimage as ndi
    SCIPY = True
except ImportError:
    SCIPY = False
    print("Warning: Scipy not installed. Using NumPy-only ops.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Device: {DEVICE} ---")
C0 = 299_792_458.0

# ================= PARAMS =================
@dataclass
class SystemParams:
    fc: float = 77e9
    B:  float = 150e6     # FMCW bandwidth (sets range resolution)
    fs: float = 150e6     # ADC sample-rate (>= B)
    M:  int   = 512       # chirps (Doppler bins)
    N:  int   = 512       # samples per chirp (Range FFT size)
    H:  float = 1.8
    az_fov: float = 60.0
    el_fov: float = 20.0
    bev_r_max: float = 50.0   # meters (BEV range clamp)

    @property
    def lambda_m(self): return C0 / self.fc
    @property
    def T_chirp(self):  return self.N / self.fs
    @property
    def slope(self):    return self.B / self.T_chirp  # S = B/T

    # Axes for FMCW processing used below (one-sided range, centered Doppler)
    def fmcw_axes(self):
        # Range: bins 0..N/2-1 -> R_k = c * (k*fs/N) / (2*S) == c*k/(2*B)
        ra = (C0 / (2.0 * self.B)) * np.arange(self.N // 2)
        # Doppler: f_d bins via slow-time PRF=1/T, then v = (λ/2) f_d
        f_d = np.fft.fftshift(np.fft.fftfreq(self.M, d=self.T_chirp))
        va = (self.lambda_m / 2.0) * f_d
        return ra, va

    # Keep previous OTFS extents for consistency with your sim
    def otfs_axes(self):
        r = np.linspace(0, (C0 / (2 * self.fs)) * self.N, self.N)
        v = np.linspace(-(self.lambda_m/2)*(self.fs/(self.N*self.M))*(self.M/2),
                         (self.lambda_m/2)*(self.fs/(self.N*self.M))*(self.M/2), self.M)
        return r, v

import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------- Sub-pixel refinement (reduces |Δr|, |Δv|) ----------
def _subpixel_quadratic(heat: np.ndarray, y: int, x: int):
    """Return (dy, dx) in [-1,1] via 1D quadratic peak fit around (y,x)."""
    H, W = heat.shape
    y0, y1, y2 = max(0, y-1), y, min(H-1, y+1)
    x0, x1, x2 = max(0, x-1), x, min(W-1, x+1)

    def quad_peak(a, b, c):
        # fits (-1,a), (0,b), (1,c) -> peak at t=(a-c)/(2*(a-2b+c))
        denom = (a - 2*b + c)
        if abs(denom) < 1e-9:
            return 0.0
        t = 0.5 * float(a - c) / float(denom)
        return float(np.clip(t, -1.0, 1.0))

    dy = quad_peak(heat[y0, x1], heat[y1, x1], heat[y2, x1])
    dx = quad_peak(heat[y1, x0], heat[y1, x1], heat[y1, x2])
    return dy, dx

# ---------- Safe F1 & PR helpers ----------
def _safe_f1(tp, fp, fn):
    if tp == 0 and (fp + fn) > 0:
        return 0.0
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)

def _precision_recall(tp, fp, fn):
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    return precision, recall

# ---------- Convert peak indices (+subpixel) to (range, vel) ----------
def _rv_from_idx_with_subpix(y, x, dy, dx, ra, va):
    # va is vertical axis (rows), ra is horizontal axis (cols)
    # Convert fractional index to physical units by linear interpolation
    y_f = np.clip(y + dy, 0, len(va) - 1)
    x_f = np.clip(x + dx, 0, len(ra) - 1)
    # Local interpolation (nearest bins):
    y0, y1 = int(np.floor(y_f)), min(int(np.floor(y_f)) + 1, len(va)-1)
    x0, x1 = int(np.floor(x_f)), min(int(np.floor(x_f)) + 1, len(ra)-1)
    wy = y_f - y0
    wx = x_f - x0
    v = (1-wy)*va[y0] + wy*va[y1]
    r = (1-wx)*ra[x0] + wx*ra[x1]
    return r, v

# ================= Utils =================
# ---- config normalizers (put near your comm helpers) ----
def cfg_for_ofdm_gen(cfg: dict):
    """
    Map a general OFDM cfg into what comm_dl_gen_batch_OFDM expects.
    Accepts keys: Nfft, cp_len, n_sym, n_ofdm_sym, batch
    """
    c = dict(cfg)  # shallow copy
    # generator typically wants 'n_sym'
    if "n_sym" not in c and "n_ofdm_sym" in c:
        c["n_sym"] = c.pop("n_ofdm_sym")
    # avoid duplicate 'batch' if caller passes batch explicitly
    c.pop("batch", None)
    return c

def cfg_for_ofdm_ber(cfg: dict):
    """
    Map a general OFDM cfg into what ofdm_tx_rx_ber expects.
    Accepts keys: Nfft, cp_len, n_sym, n_ofdm_sym
    """
    c = dict(cfg)
    # BER sim expects 'n_ofdm_sym'
    if "n_ofdm_sym" not in c and "n_sym" in c:
        c["n_ofdm_sym"] = c.pop("n_sym")
    # drop keys it doesn't know
    keep = ("Nfft", "cp_len", "n_ofdm_sym")
    return {k: c[k] for k in keep if k in c}

def cfg_for_otfs_gen(cfg: dict):
    c = dict(cfg)
    c.pop("batch", None)
    return c

def cfg_for_otfs_ber(cfg: dict):
    keep = ("M", "N", "cp_len")
    return {k: cfg[k] for k in keep if k in cfg}

def to_torch(x): return torch.tensor(x, device=DEVICE, dtype=torch.float32)

def _moving_sum_2d(a, r, c):
    if r == 0 and c == 0: return a.copy()
    ap = np.pad(a, ((r, r), (c, c)), mode='edge')
    S = ap.cumsum(axis=0).cumsum(axis=1)
    H, W = a.shape
    s22 = S[2*r:2*r+H, 2*c:2*c+W]
    s02 = S[0:H,       2*c:2*c+W]
    s20 = S[2*r:2*r+H, 0:W]
    s00 = S[0:H,       0:W]
    return s22 - s02 - s20 + s00

def nms2d(arr, kernel=3):
    k = max(3, int(kernel) | 1)
    pad = k // 2
    ap = np.pad(arr, ((pad, pad), (pad, pad)), mode='edge')
    max_nb = np.full_like(arr, -np.inf)
    for di in range(-pad, pad + 1):
        for dj in range(-pad, pad + 1):
            if di == 0 and dj == 0: continue
            view = ap[pad+di:pad+di+arr.shape[0], pad+dj:pad+dj+arr.shape[1]]
            max_nb = np.maximum(max_nb, view)
    return arr > max_nb

def cfar2d_ca(rd_db,
              train=(10, 8), guard=(2, 2),
              pfa=1e-4, min_snr_db=8.0,
              notch_doppler_bins=2,
              apply_nms=True, max_peaks=60,
              return_stats=False):
    rd_lin = 10.0 ** (rd_db / 10.0)
    H, W = rd_lin.shape
    mid = H // 2
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
            idx = np.argpartition(-vals, max_peaks - 1)[:max_peaks]
            keep = np.zeros_like(det, dtype=bool)
            keep[yy[idx], xx[idx]] = True
            det = keep
    if return_stats:
        return det, noise, snr_db
    return det

def plot_rd(ax, rd_db, ra, va, title, dynamic_db=35, percentile_clip=99.2, cmap='magma'):
    top = np.percentile(rd_db, percentile_clip)
    vmin = top - dynamic_db
    im = ax.imshow(rd_db, extent=[ra[0], ra[-1], va[0], va[-1]],
                   origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=top)
    ax.set_title(title)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    return im

# ================= Raycast & Sims =================
def raycast_torch(sp: SystemParams, gts):
    az = torch.linspace(np.deg2rad(-sp.az_fov/2), np.deg2rad(sp.az_fov/2), 1024, device=DEVICE)
    el = torch.linspace(np.deg2rad(-sp.el_fov/2), np.deg2rad(sp.el_fov/2), 128, device=DEVICE)
    EL, AZ = torch.meshgrid(el, az, indexing='ij')
    rays = torch.stack([torch.cos(EL)*torch.cos(AZ), torch.cos(EL)*torch.sin(AZ), torch.sin(EL)], dim=-1).reshape(-1, 3)
    pos = torch.tensor([0.,0.,sp.H], device=DEVICE)

    t_min = torch.full((rays.shape[0],), 100.0, device=DEVICE)
    hits_int = torch.zeros((rays.shape[0],), device=DEVICE)
    hits_vel = torch.zeros((rays.shape[0], 3), device=DEVICE)

    # ground plane
    mask_g = (rays[:, 2] < -2e-2)
    t_g = -pos[2] / rays[:, 2]
    mask_valid_g = mask_g & (t_g > 0) & (t_g < t_min)
    t_min[mask_valid_g] = t_g[mask_valid_g]
    hits_int[mask_valid_g] = 100.0

    if gts:
        Cs = torch.stack([to_torch(gt['c']) for gt in gts])
        Ss = torch.stack([to_torch(gt['s']) for gt in gts])
        Vs = torch.stack([to_torch(gt['v']) for gt in gts])
        ro = pos.view(1,1,3); rd = rays.view(-1,1,3)+1e-9
        t1 = (Cs-Ss/2-ro)/rd; t2 = (Cs+Ss/2-ro)/rd
        tn = torch.max(torch.min(t1,t2), dim=-1)[0]
        tf = torch.min(torch.max(t1,t2), dim=-1)[0]
        mask_hit = (tn < tf) & (tn > 0)
        tn[~mask_hit] = np.inf
        min_t, min_idx = torch.min(tn, dim=1)
        mask_t = min_t < t_min
        t_min[mask_t] = min_t[mask_t]
        hits_int[mask_t] = 255.0
        hits_vel[mask_t] = Vs[min_idx[mask_t]]

    mask = hits_int > 0
    return pos + t_min[mask].unsqueeze(1)*rays[mask], hits_int[mask], hits_vel[mask]

# --------- FMCW with correct RD formation & axes ----------
def fmcw_torch(pts, its, vels, sp: SystemParams):
    M, N = sp.M, sp.N
    iq = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)
    if len(pts)==0:
        return (np.zeros((M, N//2)),)

    P = pts - to_torch([0,0,sp.H]); R = torch.norm(P, dim=1)
    mask = R > 0.1; P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
    amp = torch.where(its==255, 1e6, 1e-1) / (R**2 + 1e-6)
    vr  = torch.sum(P/R.unsqueeze(1)*vels, dim=1)

    t_f = torch.arange(N, device=DEVICE)/sp.fs             # fast-time samples within chirp
    t_s = torch.arange(M, device=DEVICE)*sp.T_chirp        # slow-time (chirp index spacing)
    k_r = 2*sp.slope/C0                                    # beat freq = k_r * R
    k_v = 2/sp.lambda_m                                    # Doppler freq = k_v * v

    BATCH = 4096
    for i in range(0, len(R), BATCH):
        rb, vrb, ab = R[i:i+BATCH], vr[i:i+BATCH], amp[i:i+BATCH]
        phase = 2j*np.pi*( (k_r*rb[:,None,None])*t_f[None,None,:] + (k_v*vrb[:,None,None])*t_s[None,:,None] )
        iq += torch.sum(ab[:,None,None]*torch.exp(phase), dim=0)

    # noise + 1st-order MTI
    iq = iq + (torch.randn(M,N,device=DEVICE)+1j*torch.randn(M,N,device=DEVICE)) * 1e-4
    iq[1:] -= iq[:-1].clone(); iq[0]=0

    # window
    w_r = torch.hann_window(N,device=DEVICE)
    w_d = torch.hann_window(M,device=DEVICE)
    iq = iq * (w_d[:,None] * w_r[None,:])

    # Range FFT (one-sided 0..N/2-1), then Doppler FFT with fftshift
    RFFT = torch.fft.fft(iq, dim=1)
    RFFT = RFFT[:, :N//2]
    RD   = torch.fft.fftshift(torch.fft.fft(RFFT, dim=0), dim=0)

    RD_mag = torch.abs(RD).clamp_min(1e-12)
    rd_db = 20*torch.log10(RD_mag).cpu().numpy()
    return (rd_db,)

# --------- OTFS stays as before (toy mapping) ----------
def otfs_torch(pts, its, vels, sp: SystemParams):
    M, N = sp.M, sp.N
    H = torch.zeros((M,N), dtype=torch.complex64, device=DEVICE)
    if len(pts)==0: return (np.zeros((M,N)),)

    P = pts - to_torch([0,0,sp.H]); R = torch.norm(P, dim=1)
    mask = R > 0.1; P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
    amp = torch.where(its==255, 1e6, 1e-1) / (R**2 + 1e-6)
    vr  = torch.sum(P/R.unsqueeze(1)*vels, dim=1)

    k_res = C0 / (2 * sp.fs)
    l_res = (sp.lambda_m / 2) * (sp.fs / (sp.M * sp.N))

    k = torch.clamp((R / k_res).long(), 0, N-1)
    l = torch.clamp((vr / l_res).long() + M//2, 0, M-1)
    H.view(-1).scatter_add_(0, (l*N + k).view(-1), amp.to(torch.complex64))
    H += (torch.randn(M,N,device=DEVICE)+1j*torch.randn(M,N,device=DEVICE))*1e-4

    rd_db = (20*torch.log10(torch.abs(H).clamp_min(1e-12))).cpu().numpy()
    return (rd_db,)

# ================= Visualization (2D/3D RD) =================
# def _db_scale(rd):
#     # if already in dB it’s ok; else convert |X| -> dB
#     if np.nanmax(rd) < 100:  # heuristic; raw |X| typically << 100
#         rd = 20*np.log10(np.abs(rd)+1e-9)
#     return rd

def _db_scale(rd):
    """Ensure data is in dB for visualization."""
    rd = np.asarray(rd)
    # Heuristic: if already looks like dB, keep as is
    if np.nanmax(rd) > 100:  # probably already dB (or large dynamic values)
        return rd
    # Otherwise treat as magnitude and convert to dB
    return 20 * np.log10(np.abs(rd) + 1e-9)

@torch.no_grad()
def viz_rd_2d_compare(path, rd_f_db, rd_o_db, gts, sp: SystemParams):
    """
    Draw side-by-side 2D maps for FMCW & OTFS and
    RETURN:
        det_f_mask : FMCW CFAR detection mask (H,W) bool
        ra_f, va_f : FMCW range & velocity axes
        noise_f    : simple noise estimate (H,W)
        snr_f      : "SNR" map (rd_f_db - noise_f)
    This keeps backward compatibility with:
        det_f_mask, ra_f, va_f, noise_f, snr_f = viz_rd_2d_compare(...)
    """
    fig, ax = plt.subplots(1, 2, figsize=(16,6))
    pos = np.array([0,0,sp.H])

    # Axes
    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    # We'll also prepare a CFAR mask + simple noise/SNR for FMCW
    # If you have a more advanced noise estimator, you can replace this.
    rd_f = np.asarray(rd_f_db).astype(np.float32)
    rd_o = np.asarray(rd_o_db).astype(np.float32)

    # --- Simple noise estimate for FMCW: median over map ---
    noise_level = np.median(rd_f)
    noise_f = np.full_like(rd_f, noise_level, dtype=np.float32)
    snr_f = rd_f - noise_f

    # --- FMCW CFAR for mask (use your global cfar2d_ca defaults) ---
    try:
        det_f_mask = cfar2d_ca(rd_f, pfa=1e-4)
    except TypeError:
        # If your cfar2d_ca has a different signature, adjust here
        det_f_mask = cfar2d_ca(rd_f)

    for i, (rd_raw, ra, va, name) in enumerate([
        (rd_f, ra_f, va_f, "FMCW"),
        (rd_o, ra_o, va_o, "OTFS")
    ]):
        rd = _db_scale(rd_raw)

        # Dynamic range: percentile-based for a "full" surface
        lo, hi = np.percentile(rd, [5, 99.7])
        im = ax[i].pcolormesh(ra, va, rd, cmap='turbo',
                              vmin=lo, vmax=hi, shading='auto')
        cbar = plt.colorbar(im, ax=ax[i])
        cbar.set_label("Amplitude (dB)")

        # GT markers
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

        # Optional: for the FMCW panel, overlay CFAR dets
        if name == "FMCW":
            ys, xs = np.where(det_f_mask)
            if ys.size > 0:
                ax[i].scatter(ra[xs], va[ys],
                              s=40, facecolors='none', edgecolors='cyan',
                              linewidths=1.0, label="CFAR")
                ax[i].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Return values for downstream code expecting the old API
    return det_f_mask, ra_f, va_f, noise_f, snr_f

def viz_rd_3d_compare(path, rd_f, rd_o, gts, sp):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(18,8))
    pos = np.array([0,0,sp.H])
    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    for i, (rd_raw, ra, va, name) in enumerate([(rd_f, ra_f, va_f, "FMCW"), (rd_o, ra_o, va_o, "OTFS")]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        rd = _db_scale(rd_raw)
        lo, hi = np.percentile(rd, [5, 99.7])
        R, V = np.meshgrid(ra, va)
        surf = np.clip(rd, lo, hi)
        ax.plot_surface(R, V, surf, cmap='viridis', rstride=2, cstride=2, linewidth=0, antialiased=True)
        for gt in gts:
            P = np.array(gt['c']) - pos; r = np.linalg.norm(P); v = np.dot(P/r, gt['v'])
            if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
                ax.scatter([r],[v],[hi], c='r', marker='x', s=80)
        ax.set_title(f"{name} 3D (dB)"); ax.set_xlabel("Range(m)"); ax.set_ylabel("Vel(m/s)")
        ax.set_xlim(0, ra[-1]); ax.set_ylim(va[0], va[-1]); ax.set_zlim(lo, hi)
        ax.view_init(35, -120)

    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()
    
# ================= NEW: 3D RD with Detections vs GT =================
def extract_detections(rd_db, det_mask, ra, va, noise_db=None, snr_db=None):
    yy, xx = np.where(det_mask)
    dets = []
    for y, x in zip(yy, xx):
        det = {'r': float(ra[x]), 'v': float(va[y]), 'mag_db': float(rd_db[y, x])}
        if snr_db is not None: det['snr_db'] = float(snr_db[y, x])
        if noise_db is not None: det['noise_db'] = float(noise_db[y, x])
        dets.append(det)
    return dets

def viz_rd_3d_with_dets(path, rd_raw, ra, va, det_mask, gts, sp, title="RD with Detections & GT"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    rd = _db_scale(rd_raw)
    lo, hi = np.percentile(rd, [5, 99.7])
    R, V = np.meshgrid(ra, va)
    surf = np.clip(rd, lo, hi)

    ax.plot_surface(R, V, surf, cmap='viridis', rstride=2, cstride=2, linewidth=0, antialiased=True, alpha=0.95)

    # detections
    if det_mask is not None and det_mask.any():
        ys, xs = np.where(det_mask)
        ax.scatter(ra[xs], va[ys], surf[ys, xs], s=18, c='cyan', depthshade=False, label='Detections')

    # GT
    pos = np.array([0,0,sp.H])
    for gt in gts:
        P = np.array(gt['c']) - pos; r = np.linalg.norm(P); v = np.dot(P/r, gt['v'])
        if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
            ax.scatter([r],[v],[hi], c='r', marker='x', s=100, label='GT')

    ax.set_title(title + " (dB)")
    ax.set_xlabel("Range/Delay (m)"); ax.set_ylabel("Doppler (m/s)")
    ax.set_xlim(0, ra[-1]); ax.set_ylim(va[0], va[-1]); ax.set_zlim(lo, hi)
    ax.view_init(35, -120)
    # avoid "no handles" warning
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right')

    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()
    
# ================= NEW: BEV scene + BEV overlay of detections =================
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def _gt_rv_az(gts, sp: SystemParams):
    pos = np.array([0.0, 0.0, sp.H])
    out = []
    for gt in gts:
        c = np.array(gt['c']); v = np.array(gt['v'])
        d = c - pos; r = np.linalg.norm(d)
        if r < 1e-6: continue
        u = d / r
        vr = float(np.dot(u, v))
        az = float(np.arctan2(c[1], c[0]))  # ground-plane azimuth
        out.append({'c': c, 'r': float(r), 'v': vr, 'az': az})
    return out

def _match_dets_to_gts(dets, gtinfo, w_r=1.0, w_v=0.5):
    """
    Always return:
        pairs = [(det, gi, cost)]
        unpaired = [det,...]
    """
    used = set()
    pairs = []
    unpaired = []

    for d in dets:
        best_i = None
        best_cost = 1e12
        for gi, g in enumerate(gtinfo):
            # cost in RD/DD domain
            cost = w_r * abs(d['r'] - g['r']) + w_v * abs(d['v'] - g['v'])
            if cost < best_cost:
                best_cost = cost
                best_i = gi

        if best_i is None:
            unpaired.append(d)
            continue

        # one-to-one match
        if best_i not in used:
            used.add(best_i)
            pairs.append((d, best_i, best_cost))
        else:
            unpaired.append(d)

    return pairs, unpaired

def viz_bev_scene(path_prefix, pts, gts, sp: SystemParams):
    """3D BEV-like scene: radar, raycast points, GT boxes, clamped to 0..50 m."""
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    radar = np.array([0,0,sp.H])
    ax.plot([radar[0]],[radar[1]],[radar[2]], 'ko', ms=8, label='Radar')

    if len(pts)>0:
        p = pts.detach().cpu().numpy()[::10]
        ax.scatter(p[:,0], p[:,1], p[:,2], s=0.5, c=p[:,2], alpha=0.3, cmap='viridis')
    for gt in gts:
        c, s = np.array(gt['c']), np.array(gt['s']); dx,dy,dz = s/2
        corn = np.array([[c[0]+i*dx, c[1]+j*dy, c[2]+k*dz] for i in [-1,1] for j in [-1,1] for k in [-1,1]])
        edges = [[corn[0], corn[1]], [corn[0], corn[2]], [corn[0], corn[4]], [corn[7], corn[6]],
                 [corn[7], corn[5]], [corn[7], corn[3]], [corn[2], corn[6]], [corn[2], corn[3]],
                 [corn[1], corn[5]], [corn[1], corn[3]], [corn[4], corn[5]], [corn[4], corn[6]]]
        ax.add_collection3d(Line3DCollection(edges, colors='r', lw=2))

    # Clamp to 0..50 m in X; +/- 25 m in Y
    ax.set_xlim(0, sp.bev_r_max); 
    ax.set_ylim(-sp.bev_r_max/2, sp.bev_r_max/2); 
    ax.set_zlim(0, 15)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title("3D Scene (Raycast, 0–50 m)")
    ax.view_init(30, -110)
    plt.tight_layout(); plt.savefig(f"{path_prefix}_bev_scene.pdf"); plt.close()

def viz_bev_dets_vs_gt(path_prefix, dets, gts, sp: SystemParams):
    """BEV comparison (detections placed at matched GT azimuth for visualization)."""
    gtinfo = _gt_rv_az(gts, sp)
    pairs, unpaired = _match_dets_to_gts(dets, gtinfo, w_r=1.0, w_v=0.5)

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.scatter([0],[0], marker='*', s=120, c='k', label='Radar (XY)')

    # Range rings up to 50 m
    for rr in np.arange(10, sp.bev_r_max+1e-6, 10):
        circ = plt.Circle((0,0), rr, color='gray', fill=False, alpha=0.25, lw=0.8)
        ax.add_artist(circ)
        ax.text(rr, 0, f"{rr:.0f}m", color='gray', fontsize=8)

    # GT points
    for g in gtinfo:
        xg = g['r']*np.cos(g['az']); yg = g['r']*np.sin(g['az'])
        if 0 <= xg <= sp.bev_r_max and -sp.bev_r_max/2 <= yg <= sp.bev_r_max/2:
            ax.scatter([xg],[yg], c='r', s=60, marker='x', label='GT' if 'GT' not in ax.get_legend_handles_labels()[1] else None)

    # Matched detections at GT azimuth; line shows range error
    for det, g, cost in pairs:
        xd = det['r']*np.cos(g['az']); yd = det['r']*np.sin(g['az'])
        if 0 <= xd <= sp.bev_r_max and -sp.bev_r_max/2 <= yd <= sp.bev_r_max/2:
            ax.scatter([xd],[yd], facecolors='none', edgecolors='c', s=80, label='Det (CFAR)' if 'Det (CFAR)' not in ax.get_legend_handles_labels()[1] else None)
            xg = g['r']*np.cos(g['az']); yg = g['r']*np.sin(g['az'])
            ax.plot([xg, xd], [yg, yd], 'c--', linewidth=1)

    # Unpaired detections (project along +X)
    for d in unpaired:
        if 0 <= d['r'] <= sp.bev_r_max:
            ax.scatter([d['r']], [0], facecolors='none', edgecolors='orange', s=60,
                       label='Det (unmatched)' if 'Det (unmatched)' not in ax.get_legend_handles_labels()[1] else None)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, sp.bev_r_max); 
    ax.set_ylim(-sp.bev_r_max/2, sp.bev_r_max/2)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title("BEV: CFAR Detections vs Ground Truth (0–50 m)")
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(f"{path_prefix}_bev_compare.pdf", dpi=180); plt.close()

from matplotlib.patches import Rectangle

def _gt_rv_az(gts, sp):
    pos = np.array([0.0, 0.0, sp.H])
    out = []
    for gt in gts:
        c = np.array(gt['c']); v = np.array(gt['v'])
        d = c - pos; r = np.linalg.norm(d)
        if r < 1e-6: continue
        u = d / r
        vr = float(np.dot(u, v))
        az = float(np.arctan2(c[1], c[0]))  # ground-plane azimuth
        out.append({'c': c, 'r': float(r), 'v': vr, 'az': az, 's': np.array(gt['s'])})
    return out

# def _match_dets_to_gts(dets, gt_rv, w_r=1.0, w_v=0.5):
#     used = []  # 允许多个检测匹配到同一 GT，用列表而不是 set（如果你想“一对一”，把它改回 set 即可）
#     pairs = []
#     for d in dets:
#         best_g = None; best_cost = 1e9
#         for gi, g in enumerate(gt_rv):
#             cost = w_r*abs(d['r'] - g['r']) + w_v*abs(d['v'] - g['v'])
#             if cost < best_cost:
#                 best_cost = cost; best_g = gi
#         if best_g is not None:
#             pairs.append((d, best_g, best_cost))
#             used.append(best_g)
#     return pairs

def _inside_cube_xy(x, y, g):
    cx, cy = g['c'][0], g['c'][1]
    sx, sy = g['s'][0], g['s'][1]
    return (cx - sx/2.0 <= x <= cx + sx/2.0) and (cy - sy/2.0 <= y <= cy + sy/2.0)

def _project_det_xy_using_gt_az(det, g):
    # 用匹配 GT 的方位角把 (range) 投影到 BEV
    az = g['az']
    x = det['r'] * np.cos(az)
    y = det['r'] * np.sin(az)
    return x, y

# def _compute_metrics_from_pairs(pairs, gtinfo, sp):
#     """
#     pairs: list of (det, gi, cost), gi 为 gtinfo 的索引
#     规则：检测点投影到匹配 GT 的 az；若 (x,y) 落在该 GT cube 的 XY 范围内 => TP，否则 FP。
#     Recall：被至少一个 TP 覆盖的 GT / GT 总数。
#     """
#     TP, FP = 0, 0
#     per_gt_tp = {i:0 for i in range(len(gtinfo))}
#     tp_pts, fp_pts = [], []

#     for det, gi, _ in pairs:
#         g = gtinfo[gi]
#         x, y = _project_det_xy_using_gt_az(det, g)
#         # 限定 BEV 范围（0..sp.bev_r_max, -sp.bev_r_max/2..+sp.bev_r_max/2）
#         if not (0 <= x <= sp.bev_r_max and -sp.bev_r_max/2 <= y <= sp.bev_r_max/2):
#             # 画面之外也计为 FP（可按需改成忽略）
#             FP += 1
#             fp_pts.append((x,y,gi))
#             continue
#         if _inside_cube_xy(x, y, g):
#             TP += 1
#             per_gt_tp[gi] += 1
#             tp_pts.append((x,y,gi))
#         else:
#             FP += 1
#             fp_pts.append((x,y,gi))

#     detected_gts = sum(1 for k,v in per_gt_tp.items() if v > 0)
#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
#     recall    = detected_gts / max(1, len(gtinfo))
#     f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

#     metrics = dict(TP=TP, FP=FP, detected_gts=detected_gts,
#                    total_gts=len(gtinfo), precision=precision, recall=recall, f1=f1)
#     return metrics, tp_pts, fp_pts

def _compute_metrics_from_pairs(pairs, gtinfo, sp):
    TP = len(pairs)
    FP = 0  # caller passes FP separately

    er_r = []
    er_v = []

    for det, gi, cost in pairs:
        g = gtinfo[gi]
        er_r.append(abs(det['r'] - g['r']))
        er_v.append(abs(det['v'] - g['v']))

    metrics = dict(
        TP=TP,
        FP=FP,   # FP should be added outside
        er_r=er_r,
        er_v=er_v
    )
    return metrics, er_r, er_v

def _draw_bev_panel(ax, dets, gts, sp, title="BEV", ring_step=10):
    gtinfo = _gt_rv_az(gts, sp)
    pairs  = _match_dets_to_gts(dets, gtinfo, w_r=1.0, w_v=0.5)
    metrics, tp_pts, fp_pts = _compute_metrics_from_pairs(pairs, gtinfo, sp)

    # 雷达与量程环
    ax.scatter([0],[0], marker='*', s=140, c='k', label='Radar')
    for rr in np.arange(ring_step, sp.bev_r_max+1e-6, ring_step):
        circ = plt.Circle((0,0), rr, color='gray', fill=False, alpha=0.22, lw=0.8)
        ax.add_artist(circ)

    # 画 GT 的 XY footprint（矩形）
    for g in gtinfo:
        cx, cy = g['c'][0], g['c'][1]
        sx, sy = g['s'][0], g['s'][1]
        rect = Rectangle((cx - sx/2, cy - sy/2), sx, sy, linewidth=1.8,
                         edgecolor='r', facecolor='none', alpha=0.9, label='GT' if 'GT' not in ax.get_legend_handles_labels()[1] else None)
        ax.add_patch(rect)
        ax.plot([cx],[cy],'rx',ms=6)

    # 画 TP / FP
    if tp_pts:
        ax.scatter([p[0] for p in tp_pts], [p[1] for p in tp_pts],
                   s=64, facecolors='none', edgecolors='lime', linewidths=2.0,
                   label='TP (in cube)')
    if fp_pts:
        ax.scatter([p[0] for p in fp_pts], [p[1] for p in fp_pts],
                   s=64, facecolors='none', edgecolors='orange', linewidths=2.0,
                   label='FP (out of cube)')

    # 轴与布局
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, sp.bev_r_max)
    ax.set_ylim(-sp.bev_r_max/2, sp.bev_r_max/2)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(loc='upper left')

    # 指标文本框
    txt = (f"TP: {metrics['TP']}   FP: {metrics['FP']}\n"
           f"Precision: {metrics['precision']:.2f}\n"
           f"Recall: {metrics['recall']:.2f} ({metrics['detected_gts']}/{metrics['total_gts']})\n"
           f"F1: {metrics['f1']:.2f}")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85, linewidth=0.8))

    return metrics

def viz_scene_bev_compare(path, dets_fmcw, dets_otfs, gts, sp):
    """输出 scene_bev_compare.png：左 FMCW，右 OTFS，并在每个面板上写指标"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    _draw_bev_panel(axes[0], dets_fmcw, gts, sp, title='BEV (FMCW)')
    _draw_bev_panel(axes[1], dets_otfs, gts, sp, title='BEV (OTFS)')

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()

# ================= COMMUNICATIONS (OFDM / OTFS) =================
def _rand_bits(n, rng):
    return rng.integers(0, 2, size=n, dtype=np.uint8)

def _qpsk_gray_mod(bits):
    """bits shape (..., 2), Gray mapping: 00->(1+1j)/√2, 01->(-1+1j)/√2, 11->(-1-1j)/√2, 10->(1-1j)/√2"""
    b0 = bits[..., 0]
    b1 = bits[..., 1]
    # Gray -> symbols: I = 1-2*b0 XOR b1 ? 采用常见映射如下：
    # 定义表：00->(1+1j), 01->(-1+1j), 11->(-1-1j), 10->(1-1j)
    I = np.where((b0==0) & (b1==0),  1.0,
        np.where((b0==0) & (b1==1), -1.0,
        np.where((b0==1) & (b1==1), -1.0,  1.0)))
    Q = np.where((b0==0) & (b1==0),  1.0,
        np.where((b0==0) & (b1==1),  1.0,
        np.where((b0==1) & (b1==1), -1.0, -1.0)))
    s = (I + 1j*Q) / np.sqrt(2.0)  # Es = 1
    return s

def _qpsk_gray_demod(symbols):
    """
    Inverse of mapping:
      00 -> (+1,+1), 01 -> (-1,+1), 11 -> (-1,-1), 10 -> (+1,-1)  (then /√2)
    Correct hard decision:
      b0 = 1 if Q < 0 else 0
      b1 = 1 if I < 0 else 0
    """
    I = np.real(symbols)
    Q = np.imag(symbols)
    b0 = (Q < 0).astype(np.uint8)
    b1 = (I < 0).astype(np.uint8)
    return np.stack([b0, b1], axis=-1)

def _awgn(x, ebn0_db, bits_per_sym, cp_ratio=0.0, rng=None):
    """
    x: complex baseband samples, Es normalized to 1 on average before CP.
    ebn0_db: desired Eb/N0 in dB
    bits_per_sym: e.g., 2 for QPSK
    cp_ratio: Ncp / Nfft  (effective rate = bits_per_sym * 1/(1+cp_ratio))
    """
    if rng is None:
        rng = np.random.default_rng()
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    r_eff = bits_per_sym * (1.0 / (1.0 + cp_ratio))
    Es = 1.0   # by construction
    Eb = Es / r_eff
    N0 = Eb / ebn0
    sigma2_complex = N0  # E[|n|^2] = N0; each of I,Q has var N0/2
    n = (rng.normal(scale=np.sqrt(sigma2_complex/2), size=x.shape)
         + 1j*rng.normal(scale=np.sqrt(sigma2_complex/2), size=x.shape))
    return x + n

# ---------- OFDM (用于 FMCW 的通信分支) ----------
def ofdm_mod(bits, Nfft=256, cp_len=32, rng=None):
    """
    QPSK + OFDM; unitary IFFT/FFT so Es=1. Map all subcarriers (可按需空洞 DC 或保护带)。
    bits: length should be multiple of 2*Nfft
    """
    if rng is None:
        rng = np.random.default_rng()
    bits = bits.reshape(-1, 2)  # pairs
    nsym = bits.shape[0] // Nfft
    bits = bits[:nsym*Nfft].reshape(nsym, Nfft, 2)
    syms = _qpsk_gray_mod(bits)           # (nsym, Nfft)
    # IFFT with unitary scale
    x = np.fft.ifft(syms, n=Nfft, axis=1, norm='ortho')  # (nsym, Nfft)
    # add CP
    if cp_len > 0:
        cp = x[:, -cp_len:]
        x_cp = np.concatenate([cp, x], axis=1)           # (nsym, Nfft+cp_len)
    else:
        x_cp = x
    return x_cp, syms

def ofdm_demod(rx, Nfft=256, cp_len=32):
    if cp_len > 0:
        rx = rx[:, cp_len:cp_len+Nfft]
    # FFT with unitary scale
    Sy = np.fft.fft(rx, n=Nfft, axis=1, norm='ortho')  # (nsym, Nfft)
    return Sy

def ofdm_tx_rx_ber(ebn0_db, Nfft=256, cp_len=32, n_ofdm_sym=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bits_per_sym = 2
    nbits = n_ofdm_sym * Nfft * bits_per_sym
    bits = _rand_bits(nbits, rng)
    tx, ref_syms = ofdm_mod(bits, Nfft=Nfft, cp_len=cp_len, rng=rng)
    cp_ratio = cp_len / Nfft
    rx = _awgn(tx, ebn0_db, bits_per_sym=bits_per_sym, cp_ratio=cp_ratio, rng=rng)
    Sy = ofdm_demod(rx, Nfft=Nfft, cp_len=cp_len)
    # 相当于扁平 AWGN 信道，直接判决
    hard_bits = _qpsk_gray_demod(Sy.reshape(-1))
    hard_bits = hard_bits.reshape(-1,2)
    bits_hat = hard_bits[:len(bits)].reshape(-1)
    ber = np.mean(bits != bits_hat)
    return ber

# ---------- OTFS (通信) ----------
def otfs_mod(bits, M=64, N=256, cp_len=32, rng=None):
    # bits -> X_dd (QPSK)
    bits = bits.reshape(M*N, 2)[:M*N].reshape(M, N, 2)
    X_dd = _qpsk_gray_mod(bits)                       # (M,N)

    # ISFFT (DD -> TF): along N (delay) use FFT, along M (doppler/time slots) use IFFT
    X_tf = np.fft.ifft(np.fft.fft(X_dd, n=N, axis=1, norm='ortho'), n=M, axis=0, norm='ortho')  # (M,N)

    # Heisenberg (TF -> time) with rectangular pulses == OFDM per time slot
    tx = np.fft.ifft(X_tf, n=N, axis=1, norm='ortho')  # (M,N) time samples per slot
    if cp_len > 0:
        cp = tx[:, -cp_len:]
        tx = np.concatenate([cp, tx], axis=1)          # (M,N+cp)
    return tx, X_dd

def otfs_demod(rx, M=64, N=256, cp_len=32):
    if cp_len > 0:
        rx = rx[:, cp_len:cp_len+N]
    Y_tf = np.fft.fft(rx, n=N, axis=1, norm='ortho')           # (M,N)
    # SFFT inverse of above: along M use FFT, along N use IFFT to go back to DD
    Y_dd = np.fft.ifft(np.fft.fft(Y_tf, n=M, axis=0, norm='ortho'), n=N, axis=1, norm='ortho')
    return Y_dd  # (M,N)

def otfs_mod_old(bits, M=64, N=256, cp_len=32, rng=None):
    """
    最小实现：DD 域放 QPSK，先对 N 做 DFT (频域)，再对 M 做 IDFT (时域)，
    得到 M 个 OFDM 符号（每个 N 子载波），每个 OFDM 加 CP。
    全部 FFT 用 norm='ortho' 保能量单位化。
    """
    if rng is None:
        rng = np.random.default_rng()
    bits = bits.reshape(-1, 2)
    nsym_needed = M*N
    bits = bits[:nsym_needed].reshape(M, N, 2)
    X_dd = _qpsk_gray_mod(bits)           # (M, N)

    # OTFS 变换：X_dd --DFT_N--> X_fd --IDFT_M--> s(t) 的 M 个 OFDM 符号
    X_fd = np.fft.fft(X_dd, n=N, axis=1, norm='ortho')   # along delay-> frequency
    x_td = np.fft.ifft(X_fd, n=M, axis=0, norm='ortho')  # along Doppler-> time slots

    # 序列化为 M 个 OFDM 符号，每个做 IFFT(N) + CP
    # 注意：x_td 的形状 (M,N)，恰好每一行是一个 OFDM 的频域符号（已 unitary）
    # 这里再次 IFFT 会重复？——采用“OFDM 等效实现”：直接把 X_fd 作为每个时隙的频域符号做 IFFT
    # 为了贴近常见实现，更稳妥是：每个时隙的频域符号就是 X_fd 的对应行：
    ofdm_freq = X_fd  # (M, N)
    tx_time = np.fft.ifft(ofdm_freq, n=N, axis=1, norm='ortho')  # (M,N)
    if cp_len > 0:
        cp = tx_time[:, -cp_len:]
        tx_time = np.concatenate([cp, tx_time], axis=1)           # (M, N+cp)
    return tx_time, X_dd

def otfs_demod_old(rx, M=64, N=256, cp_len=32):
    """反过程：去 CP -> FFT(N) -> 得到每个时隙的频域符号 -> 2D 逆变换回 DD 域判决。"""
    if cp_len > 0:
        rx = rx[:, cp_len:cp_len+N]
    Yf = np.fft.fft(rx, n=N, axis=1, norm='ortho')       # (M,N)
    # 2D 逆变换：先 DFT^-1_M (即 FFT_M)，再 DFT^-1_N (即 IFFT_N)
    Ytd = np.fft.fft(Yf, n=M, axis=0, norm='ortho')      # along time slots -> Doppler
    Ydd = np.fft.ifft(Ytd, n=N, axis=1, norm='ortho')    # along frequency -> delay
    return Ydd  # (M,N)

def otfs_tx_rx_ber(ebn0_db, M=64, N=256, cp_len=32, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bits_per_sym = 2
    nbits = M * N * bits_per_sym
    bits = _rand_bits(nbits, rng)
    tx, Xdd = otfs_mod(bits, M=M, N=N, cp_len=cp_len, rng=rng)
    cp_ratio = cp_len / N
    rx = _awgn(tx, ebn0_db, bits_per_sym=bits_per_sym, cp_ratio=cp_ratio, rng=rng)
    Ydd = otfs_demod(rx, M=M, N=N, cp_len=cp_len)
    hard_bits = _qpsk_gray_demod(Ydd.reshape(-1))
    hard_bits = hard_bits.reshape(-1,2)
    bits_hat = hard_bits[:len(bits)].reshape(-1)
    ber = np.mean(bits != bits_hat)
    return ber

# ---------- BER sweep + 绘图 ----------
import math

def run_ber_sweep_and_plot(path_png,
                           ebn0_db_list = np.arange(0, 21, 2),
                           ofdm_cfg     = dict(Nfft=256, cp_len=32, n_ofdm_sym=400),
                           otfs_cfg     = dict(M=64, N=256, cp_len=32),
                           rng_seed=1234):
    rng = np.random.default_rng(rng_seed)

    # 1) empirical BERs
    ber_ofdm = []
    ber_otfs = []
    for eb in ebn0_db_list:
        #ber_ofdm.append(ofdm_tx_rx_ber(eb, **ofdm_cfg, rng=rng))
        # ofdm_tx_rx_ber
        # ber_ofdm.append(
        #     ofdm_tx_rx_ber(eb, **cfg_for_ofdm_ber(ofdm_cfg), rng=rng)
        # )
        ber_ofdm.append(ofdm_tx_rx_ber(eb, **ofdm_cfg_ber, rng=rng))
        #ber_otfs.append(otfs_tx_rx_ber(eb, **otfs_cfg,  rng=rng))
        # otfs_tx_rx_ber
        # ber_otfs.append(
        #     otfs_tx_rx_ber(eb, **cfg_for_otfs_ber(otfs_cfg), rng=rng)
        # )
        ber_otfs.append(otfs_tx_rx_ber(eb, **otfs_cfg_ber, rng=rng))
    ber_ofdm = np.array(ber_ofdm)
    ber_otfs = np.array(ber_otfs)

    # 2) theory: QPSK on AWGN => BER = 0.5 * erfc( sqrt(Eb/N0) )
    ebn0_lin = 10.0 ** (np.array(ebn0_db_list) / 10.0)
    ber_theory_qpsk = np.array([0.5 * math.erfc(math.sqrt(x)) for x in ebn0_lin], dtype=float)

    # 3) plot
    plt.figure(figsize=(7.8,5.6))
    plt.semilogy(ebn0_db_list, ber_ofdm + 1e-12, marker='o', label='FMCW-Comm (OFDM, QPSK)')
    plt.semilogy(ebn0_db_list, ber_otfs + 1e-12, marker='s', label='OTFS-Comm (QPSK)')
    plt.semilogy(ebn0_db_list, ber_theory_qpsk + 1e-12, linestyle='--', label='Theory QPSK (AWGN)')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.xlabel('Eb/N0 (dB)'); plt.ylabel('BER'); plt.title('BER vs Eb/N0: OFDM vs OTFS vs Theory')

    # annotate a few points (optional)
    for i in range(0, len(ebn0_db_list), 2):
        x = ebn0_db_list[i]
        yo = ber_ofdm[i]; yt = ber_otfs[i]
        plt.text(x, yo*1.15 + 1e-14, f"{yo:.2e}", fontsize=8, ha='center', va='bottom')
        plt.text(x, yt*0.85 + 1e-14, f"{yt:.2e}", fontsize=8, ha='center', va='top')

    plt.legend(); plt.tight_layout(); plt.savefig(path_png, dpi=170, bbox_inches='tight'); plt.close()

    return ebn0_db_list, ber_ofdm, ber_otfs, ber_theory_qpsk

def run_ber_sweep_and_plot_old(path_png,
                           ebn0_db_list = np.arange(0, 21, 2),
                           ofdm_cfg     = dict(Nfft=256, cp_len=32, n_ofdm_sym=400),
                           otfs_cfg     = dict(M=64, N=256, cp_len=32),
                           rng_seed=1234):
    rng = np.random.default_rng(rng_seed)
    ber_ofdm = []
    ber_otfs = []
    for eb in ebn0_db_list:
        ber_o = ofdm_tx_rx_ber(eb, **ofdm_cfg, rng=rng)
        ber_t = otfs_tx_rx_ber(eb, **otfs_cfg,  rng=rng)
        ber_ofdm.append(ber_o)
        ber_otfs.append(ber_t)

    ber_ofdm = np.array(ber_ofdm)
    ber_otfs = np.array(ber_otfs)

    plt.figure(figsize=(7.5,5.5))
    plt.semilogy(ebn0_db_list, ber_ofdm + 1e-12, marker='o', label='FMCW-Comm (OFDM, QPSK)')
    plt.semilogy(ebn0_db_list, ber_otfs + 1e-12, marker='s', label='OTFS-Comm (QPSK)')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('BER vs Eb/N0: OFDM (FMCW-Comm) vs OTFS')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=170, bbox_inches='tight')
    plt.close()

    return ebn0_db_list, ber_ofdm, ber_otfs

# ================= RADAR DL BACKEND =================
import torch.nn as nn
import torch.nn.functional as F

def _rd_normalize(rd_db, top_p=99.5, dyn_db=40.0):
    top = np.percentile(rd_db, top_p)
    rd = np.clip(rd_db, top-dyn_db, top)
    rd = (rd - (top-dyn_db)) / dyn_db  # -> [0,1]
    return rd.astype(np.float32)

def _heatmap_from_gts(shape, ra, va, gts, sp, sigma_pix=(2.0,2.0)):
    """Return HxW heatmap with Gaussians at GT positions."""
    H, W = shape
    pos = np.array([0,0,sp.H])
    yy, xx = np.mgrid[0:H, 0:W]
    Y = va[yy]  # (H,W)
    X = ra[xx]
    hm = np.zeros((H,W), np.float32)
    for gt in gts:
        P = np.array(gt['c']) - pos
        r = np.linalg.norm(P)
        v = np.dot(P/r, gt['v'])
        # skip if out of RD bounds
        if not (0 <= r <= ra[-1] and va[0] <= v <= va[-1]): 
            continue
        # nearest pixel index
        ix = np.searchsorted(ra, r); ix = np.clip(ix, 0, W-1)
        iy = np.searchsorted(va, v); iy = np.clip(iy, 0, H-1)
        # Gaussian
        sx, sy = sigma_pix[1], sigma_pix[0]
        g = np.exp(-((xx-ix)**2/(2*sx**2) + (yy-iy)**2/(2*sy**2)))
        hm = np.maximum(hm, g)
    return hm

class UNetLite(nn.Module):
    def __init__(self, in_ch=1, ch=32):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch,ch,3,padding=1), nn.ReLU(), nn.Conv2d(ch,ch,3,padding=1), nn.ReLU())
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(nn.Conv2d(ch,2*ch,3,padding=1), nn.ReLU(), nn.Conv2d(2*ch,2*ch,3,padding=1), nn.ReLU())
        self.p2 = nn.MaxPool2d(2)
        self.b  = nn.Sequential(nn.Conv2d(2*ch,4*ch,3,padding=1), nn.ReLU(), nn.Conv2d(4*ch,4*ch,3,padding=1), nn.ReLU())
        self.u2 = nn.ConvTranspose2d(4*ch,2*ch,2,stride=2)
        self.d2 = nn.Sequential(nn.Conv2d(4*ch,2*ch,3,padding=1), nn.ReLU(), nn.Conv2d(2*ch,2*ch,3,padding=1), nn.ReLU())
        self.u1 = nn.ConvTranspose2d(2*ch,ch,2,stride=2)
        self.d1 = nn.Sequential(nn.Conv2d(2*ch,ch,3,padding=1), nn.ReLU(), nn.Conv2d(ch,ch,3,padding=1), nn.ReLU())
        self.out = nn.Conv2d(ch,1,1)
    def forward(self,x):
        e1 = self.e1(x)            # HxW
        e2 = self.e2(self.p1(e1))  # H/2 x W/2
        b  = self.b(self.p2(e2))   # H/4 x W/4
        d2 = self.d2(torch.cat([self.u2(b), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)        # logits

def focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    # logits, targets: (B,1,H,W)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = p*targets + (1-p)*(1-targets)
    w = alpha*targets + (1-alpha)*(1-targets)
    loss = (w * (1-pt).pow(gamma) * bce).mean()
    return loss

class RadarSimDataset(torch.utils.data.Dataset):
    def __init__(self, sp, n_items=2000, rng_seed=123, min_targets=1, max_targets=3):
        self.sp = sp; self.n = n_items; self.rng = np.random.default_rng(rng_seed)
        self.min_t = min_targets; self.max_t = max_targets
        self.ra, self.va = sp.fmcw_axes()
    def _rand_gts(self):
        k = self.rng.integers(self.min_t, self.max_t+1)
        gts=[]
        for _ in range(k):
            r = self.rng.uniform(8, 45)
            az= self.rng.uniform(-np.deg2rad(self.sp.az_fov/2), np.deg2rad(self.sp.az_fov/2))
            x = r*np.cos(az); y=r*np.sin(az)
            vx = self.rng.uniform(-20,20); vy=self.rng.uniform(-5,5)
            gts.append({'c':[x,y,1.0], 's':[4,2,2], 'v':[vx,vy,0]})
        return gts
    def __getitem__(self, idx):
        gts = self._rand_gts()
        pts, its, vels = raycast_torch(self.sp, gts)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        (rd_db,) = fmcw_torch(pts, its, vels, self.sp)  # (M, N/2)
        rd = _rd_normalize(rd_db)
        hm = _heatmap_from_gts(rd.shape, self.ra, self.va, gts, self.sp)
        x = torch.from_numpy(rd[None,...])     # (1,H,W)
        y = torch.from_numpy(hm[None,...])     # (1,H,W)
        return x, y
    def __len__(self): return self.n

@torch.no_grad()
def rd_dl_infer_to_points(logits, ra, va, thr=0.35, max_peaks=64):
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask = prob > thr
    if not mask.any():
        return []
    # NMS: keep local maxima
    from scipy.ndimage import maximum_filter
    mxf = maximum_filter(prob, size=3)
    peaks = (prob==mxf) & mask
    yy, xx = np.where(peaks)
    if len(yy)>max_peaks:
        vals = prob[yy,xx]; idx = np.argpartition(-vals, max_peaks-1)[:max_peaks]
        yy,xx = yy[idx],xx[idx]
    dets=[{'r':float(ra[x]), 'v':float(va[y]), 'score':float(prob[y,x])} for y,x in zip(yy,xx)]
    return dets

def train_radar_model(sp, epochs=5, batch=6, lr=1e-3, n_train=800, n_val=200, device=None):
    device = device or DEVICE
    net = UNetLite().to(device)
    ds_tr = RadarSimDataset(sp, n_train, rng_seed=2025)
    ds_va = RadarSimDataset(sp, n_val, rng_seed=2026)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=0)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    best = {'loss':1e9, 'state':None}
    for ep in range(1, epochs+1):
        net.train(); loss_tr=0
        for x,y in dl_tr:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = net(x)
            loss = focal_bce_with_logits(logits, y, alpha=0.25, gamma=2.0)
            loss.backward(); opt.step()
            loss_tr += loss.item()*x.size(0)
        loss_tr /= len(ds_tr)
        # val
        net.eval(); loss_va=0
        with torch.no_grad():
            for x,y in dl_va:
                x,y = x.to(device), y.to(device)
                logits = net(x)
                loss = focal_bce_with_logits(logits, y)
                loss_va += loss.item()*x.size(0)
        loss_va /= len(ds_va)
        print(f"[RadarDL] epoch {ep}/{epochs}  train {loss_tr:.4f}  val {loss_va:.4f}")
        if loss_va < best['loss']:
            best = {'loss':loss_va, 'state':{k:v.cpu() for k,v in net.state_dict().items()}}
    net.load_state_dict(best['state'])
    return net

# ================= COMM DL BACKEND =================
class CommDemapperCNN(nn.Module):
    def __init__(self, in_ch=2, width=32, depth=3, out_ch=2):
        super().__init__()
        layers = [nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU()]
        for _ in range(depth-1):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, out_ch, 1)  # 2 bits/logits per grid element
    def forward(self, x):  # x: (B,C,H,W)
        h = self.backbone(x)
        y = self.head(h)   # (B,2,H,W)
        return y

def _bits_to_qpsk_grid(bits, H, W):
    bits = bits.reshape(H*W, 2)
    syms = _qpsk_gray_mod(bits)  # (H*W,)
    return syms.reshape(H, W)

def _grid_feats(S, use_mag_phase=False):
    # S: complex grid (H,W)
    if use_mag_phase:
        mag = np.abs(S); ang = np.angle(S)
        x = np.stack([S.real, S.imag, mag, ang], axis=0).astype(np.float32)
    else:
        x = np.stack([S.real, S.imag], axis=0).astype(np.float32)
    return x

def comm_dl_gen_batch_OFDM(ebn0_db, batch=8, Nfft=256, cp_len=32, n_sym=8, rng=None):
    # Generate a mini-batch of OFDM frames and labels
    if rng is None: rng = np.random.default_rng()
    bits_per_sym = 2
    H, W = n_sym, Nfft   # treat (time, freq) as (H,W)
    x_list=[]; y_list=[]
    for _ in range(batch):
        bits = _rand_bits(H*W*bits_per_sym, rng)
        Xf = _bits_to_qpsk_grid(bits, H, W)   # (H,W) QPSK in freq per OFDM symbol/time
        # TX: IFFT per row + CP
        tx = np.fft.ifft(Xf, n=W, axis=1, norm='ortho')
        if cp_len>0:
            tx = np.concatenate([tx[:, -cp_len:], tx], axis=1)
        cp_ratio = cp_len / W
        rx = _awgn(tx, ebn0_db, bits_per_sym=2, cp_ratio=cp_ratio, rng=rng)
        # RX: remove CP + FFT
        if cp_len>0:
            rx = rx[:, cp_len:cp_len+W]
        Yf = np.fft.fft(rx, n=W, axis=1, norm='ortho')  # (H,W)
        x = _grid_feats(Yf, use_mag_phase=False)         # (C,H,W)
        y = bits.reshape(H, W, 2).transpose(2,0,1)       # (2,H,W)
        x_list.append(x); y_list.append(y.astype(np.float32))
    X = torch.from_numpy(np.stack(x_list))  # (B,C,H,W)
    Y = torch.from_numpy(np.stack(y_list))  # (B,2,H,W)
    return X, Y

def comm_dl_gen_batch_OTFS(ebn0_db, batch=8, M=64, N=256, cp_len=32, rng=None):
    # Use the OTFS functions already provided to generate one frame then slice into batch
    if rng is None: rng = np.random.default_rng()
    bits_per_sym=2
    x_list=[]; y_list=[]
    for _ in range(batch):
        bits = _rand_bits(M*N*bits_per_sym, rng)
        tx, Xdd = otfs_mod(bits, M=M, N=N, cp_len=cp_len, rng=rng)  # (M,N+cp)
        cp_ratio = cp_len / N
        rx = _awgn(tx, ebn0_db, bits_per_sym=2, cp_ratio=cp_ratio, rng=rng)
        Ydd = otfs_demod(rx, M=M, N=N, cp_len=cp_len)              # (M,N)
        x = _grid_feats(Ydd, use_mag_phase=False)                  # (2,M,N)
        y = bits.reshape(M, N, 2).transpose(2,0,1)                 # (2,M,N)
        x_list.append(x); y_list.append(y.astype(np.float32))
    X = torch.from_numpy(np.stack(x_list))
    Y = torch.from_numpy(np.stack(y_list))
    return X, Y

def train_comm_demap(model, gen_batch_fn, cfg, snr_min=0, snr_max=18, epochs=5, steps_per_epoch=200, lr=3e-4, device=None, tag="OFDM"):
    device = device or DEVICE
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        model.train(); loss_ep=0
        for step in range(steps_per_epoch):
            eb = np.random.uniform(snr_min, snr_max)
            X,Y = gen_batch_fn(eb, **cfg)
            X,Y = X.to(device), Y.to(device)
            opt.zero_grad()
            logits = model(X)               # (B,2,H,W)
            loss = F.binary_cross_entropy_with_logits(logits, Y)
            loss.backward(); opt.step()
            loss_ep += loss.item()
        print(f"[{tag} Demap DL] epoch {ep}/{epochs} loss {loss_ep/steps_per_epoch:.4f}")
    return model

# @torch.no_grad()
# def comm_demap_ber_curve(model, gen_batch_fn, cfg, ebn0_db_list, device=None):
#     device = device or DEVICE
#     model = model.to(device)
#     bers=[]
#     for eb in ebn0_db_list:
#         X,Y = gen_batch_fn(eb, **cfg)
#         X = X.to(device)
#         logits = model(X)
#         bits_hat = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(np.uint8) # (B,2,H,W)
#         bits_gt  = Y.numpy().astype(np.uint8)
#         ber = np.mean(bits_hat != bits_gt)
#         bers.append(ber)
#     return np.array(bers)
@torch.no_grad()
def comm_demap_ber_curve(model, gen_batch_fn, cfg, ebn0_db_list, device=None):
    """
    Computes BER across Eb/N0 values.
    - model(X) -> logits with shape (B,2) or (B,2,H,W)
    - gen_batch_fn(eb, **cfg) -> (X, Y) with Y as (B,2) or (B,2,H,W)
    """
    device = device or DEVICE
    model = model.to(device).eval()
    # avoid double-passing "batch" if your gen already uses it positionally
    #cfg = {k: v for k, v in cfg.items() if k != "batch"}
    cfg = dict(cfg)
    # sanitize 'batch' here; you pass batch via cfg or not? choose one:
    pass_batch = "batch" in cfg
    if not pass_batch:
        cfg.pop("batch", None)

    bers = []
    for eb in ebn0_db_list:
        X, Y = gen_batch_fn(eb, **cfg)   # or include batch=batch_size inside cfg if needed
        X = X.to(device)
        logits = model(X)                # (B,2) or (B,2,H,W)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        bits_hat = (probs > 0.5).astype(np.uint8)

        bits_gt = Y.numpy().astype(np.uint8)
        # Align shapes
        if bits_hat.ndim == 2 and bits_gt.ndim == 4:
            bits_gt = (bits_gt.mean(axis=(-1, -2)) > 0.5).astype(np.uint8)
        elif bits_hat.ndim == 4 and bits_gt.ndim == 2:
            bits_gt = np.broadcast_to(bits_gt[:, :, None, None], bits_hat.shape)

        ber = np.mean(bits_hat != bits_gt)
        bers.append(ber)

    return np.array(bers)

# ===================== NEW: Modular pipeline =====================
from pathlib import Path
from tqdm import tqdm
import json
import math
import pickle

# ---------- 1) DATASET SIMULATION & SAVE ----------
def simulate_dataset(
    out_dir,
    sp: SystemParams,
    n_train=1500,
    n_val=300,
    seed=2025,
    snr_list=(0,2,4,6,8,10,12,14,16,18,20),  # for comm reproducibility
):
    """
    Saves:
      radar/
        train/  *.npz  -> rd_f_db (MxN/2), heatmap (MxN/2), gts(list-serialized)
        val/    *.npz
      comm/
        train_spec.json  -> list of dict {seed, ebn0_db}
        val_spec.json
    Notes:
      - Radar RD maps are precomputed for speed.
      - Comm is generated on-the-fly during training but spec (seeds & SNRs) saved to be reproducible.
    """
    out = Path(out_dir)
    (out / "radar/train").mkdir(parents=True, exist_ok=True)
    (out / "radar/val").mkdir(parents=True, exist_ok=True)
    (out / "comm").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    def _rand_gts_two(sp, rng):
        k = rng.integers(1, 3)  # 1 or 2 targets
        gts=[]
        for _ in range(k):
            r  = rng.uniform(8, 48)
            az = rng.uniform(-np.deg2rad(sp.az_fov/2), np.deg2rad(sp.az_fov/2))
            x  = r*np.cos(az); y=r*np.sin(az)
            vx = rng.uniform(-20,20); vy=rng.uniform(-6,6)
            gts.append({'c':[float(x),float(y),1.0],
                        's':[4.0,2.0,2.0],
                        'v':[float(vx),float(vy),0.0]})
        return gts

    def _one_sample(idx, split):
        gts = _rand_gts_two(sp, rng)
        pts, its, vels = raycast_torch(sp, gts)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        (rd_f_db,) = fmcw_torch(pts, its, vels, sp)
        # build heatmap label
        ra_f, va_f = sp.fmcw_axes()
        hm = _heatmap_from_gts(rd_f_db.shape, ra_f, va_f, gts, sp)

        save_path = out / "radar" / split / f"{idx:06d}.npz"
        np.savez_compressed(save_path, rd_f_db=rd_f_db.astype(np.float32), heatmap=hm.astype(np.float32), gts=json.dumps(gts))

    print(f"[DATA] Simulating radar {n_train} train + {n_val} val samples → {out_dir}")
    for i in tqdm(range(n_train), desc="radar-train"):
        _one_sample(i, "train")
    for i in tqdm(range(n_val), desc="radar-val"):
        _one_sample(i, "val")

    # Communication spec (we’ll generate on-the-fly but keep deterministic seeds + SNRs)
    def _make_comm_spec(n_items):
        specs=[]
        for i in range(n_items):
            specs.append({
                "seed": int(rng.integers(0, 2**31-1)),
                "ebn0_db": float(snr_list[i % len(snr_list)])
            })
        return specs

    comm_train_spec = _make_comm_spec(max(n_train//4, 400))  # you can scale these
    comm_val_spec   = _make_comm_spec(max(n_val//4, 100))
    with open(out/"comm/train_spec.json", "w") as f: json.dump(comm_train_spec, f, indent=2)
    with open(out/"comm/val_spec.json", "w") as f:   json.dump(comm_val_spec, f, indent=2)
    print("[DATA] Done.")

# ---------- 2) DATALOADERS ----------
class RadarDiskDataset(torch.utils.data.Dataset):
    def __init__(self, folder, normalize=True):
        self.files = sorted(Path(folder).glob("*.npz"))
        self.normalize = normalize
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        z = np.load(self.files[idx], allow_pickle=True)
        rd = z["rd_f_db"].astype(np.float32)
        hm = z["heatmap"].astype(np.float32)
        if self.normalize:
            rd = _rd_normalize(rd)
        x = torch.from_numpy(rd)[None,...]  # (1,H,W)
        y = torch.from_numpy(hm)[None,...]
        return x, y

# def make_radar_loaders(root, batch=6, workers=0):
#     tr = RadarDiskDataset(Path(root)/"radar/train")
#     va = RadarDiskDataset(Path(root)/"radar/val")
#     dl_tr = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, num_workers=workers)
#     dl_va = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False, num_workers=workers)
#     return dl_tr, dl_va

def make_radar_loaders(root, batch=6, workers=0):
    root = Path(root)
    tr_dir = root/"radar"/"train"
    va_dir = root/"radar"/"val"
    tr = RadarDiskDataset(tr_dir)
    va = RadarDiskDataset(va_dir)

    if len(tr) == 0:
        raise FileNotFoundError(f"No .npz files found in {tr_dir}. "
                                f"Did you pass the correct root? Expected {root}, not {root/'radar'}.")

    if len(va) == 0:
        raise FileNotFoundError(f"No .npz files found in {va_dir}. "
                                f"Did dataset simulation finish?")

    dl_tr = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True,  num_workers=workers)
    dl_va = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False, num_workers=workers)
    return dl_tr, dl_va

# For communication, we’ll fetch batches using the earlier generators,
# but we’ll honor the saved (seed, SNR) specs to be reproducible.
def iter_comm_batches(spec_file, kind="OFDM", batch=8, ofdm_cfg=None, otfs_cfg=None):
    with open(spec_file, "r") as f: 
        specs = json.load(f)
    
    ofdm_cfg  = {k:v for k,v in (ofdm_cfg or {}).items()  if k != "batch"}
    otfs_cfg  = {k:v for k,v in (otfs_cfg or {}).items()  if k != "batch"}

    for item in specs:
        eb = item["ebn0_db"]; seed = int(item["seed"])
        rng = np.random.default_rng(seed)
        if kind == "OFDM":
            X,Y = comm_dl_gen_batch_OFDM(eb, batch=batch, rng=rng, **(ofdm_cfg or {}))
        else:
            X,Y = comm_dl_gen_batch_OTFS(eb, batch=batch, rng=rng, **(otfs_cfg or {}))
        yield X, Y, eb

# ---------- 3) TRAINING (radar + comm), CHECKPOINT/RESUME ----------
def train_both_models(
    data_root,
    ckpt_dir,
    sp: SystemParams,
    radar_epochs=6,
    radar_batch=6,
    radar_lr=1e-3,
    comm_epochs=5,
    comm_steps_per_epoch=200,
    ofdm_cfg=dict(Nfft=256, cp_len=32, n_sym=8, batch=8),
    otfs_cfg=dict(M=64, N=256, cp_len=32, batch=6),
    resume=True,
):
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Radar
    print("[TRAIN] Radar DL (U-Net Lite)")
    net = UNetLite().to(DEVICE)
    optim = torch.optim.AdamW(net.parameters(), lr=radar_lr)
    start_ep = 1

    radar_ckpt = ckpt_dir/"radar_unet.pt"
    if resume and radar_ckpt.exists():
        s = torch.load(radar_ckpt, map_location=DEVICE)
        net.load_state_dict(s["model"])
        optim.load_state_dict(s["optim"])
        start_ep = s["epoch"]+1
        print(f"[TRAIN] Resumed radar from epoch {s['epoch']} (val_loss {s['val_loss']:.4f})")

    #dl_tr, dl_va = make_radar_loaders(Path(data_root)/"radar", batch=radar_batch)
    dl_tr, dl_va = make_radar_loaders(Path(data_root), batch=radar_batch)

    best_val = math.inf
    for ep in range(start_ep, radar_epochs+1):
        net.train(); loss_tr=0.0
        pbar = tqdm(dl_tr, desc=f"Radar train ep{ep}/{radar_epochs}")
        for x,y in pbar:
            x,y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            logits = net(x)
            loss = focal_bce_with_logits(logits, y, alpha=0.25, gamma=2.0)
            loss.backward(); optim.step()
            loss_tr += loss.item()*x.size(0)
            pbar.set_postfix(loss=loss.item())
        loss_tr /= len(dl_tr.dataset)

        net.eval(); loss_va=0.0
        with torch.no_grad():
            for x,y in dl_va:
                x,y = x.to(DEVICE), y.to(DEVICE)
                logits = net(x)
                loss = focal_bce_with_logits(logits, y)
                loss_va += loss.item()*x.size(0)
        loss_va /= len(dl_va.dataset)
        print(f"[Radar] ep {ep}: train {loss_tr:.4f}  val {loss_va:.4f}")

        # save checkpoint (always) + best
        torch.save({"epoch":ep, "model":net.state_dict(), "optim":optim.state_dict(), "val_loss":loss_va}, radar_ckpt)
        if loss_va < best_val:
            best_val = loss_va
            torch.save(net.state_dict(), ckpt_dir/"radar_unet_best_only.pt")

    # Communication: OFDM + OTFS demappers
    print("[TRAIN] Comm DL demappers")
    ofdm_model = CommDemapperCNN(in_ch=2).to(DEVICE)
    otfs_model = CommDemapperCNN(in_ch=2).to(DEVICE)
    ofdm_opt = torch.optim.AdamW(ofdm_model.parameters(), lr=3e-4)
    otfs_opt = torch.optim.AdamW(otfs_model.parameters(), lr=3e-4)

    ofdm_ckpt = ckpt_dir/"comm_ofdm.pt"
    otfs_ckpt = ckpt_dir/"comm_otfs.pt"

    if resume and ofdm_ckpt.exists():
        s=torch.load(ofdm_ckpt, map_location=DEVICE)
        ofdm_model.load_state_dict(s["model"]); ofdm_opt.load_state_dict(s["optim"])
        print(f"[TRAIN] Resumed OFDM demapper (epoch {s['epoch']})")
    if resume and otfs_ckpt.exists():
        s=torch.load(otfs_ckpt, map_location=DEVICE)
        otfs_model.load_state_dict(s["model"]); otfs_opt.load_state_dict(s["optim"])
        print(f"[TRAIN] Resumed OTFS demapper (epoch {s['epoch']})")

    # epochs with generator based on saved specs
    train_spec_ofdm = Path(data_root)/"comm/train_spec.json"
    train_spec_otfs = Path(data_root)/"comm/train_spec.json"  # same spec list (SNR/seed), separate nets

    for ep in range(1, comm_epochs+1):
        ofdm_model.train(); otfs_model.train()
        # loop steps, each takes one spec-batch
        it_ofdm = iter_comm_batches(train_spec_ofdm, kind="OFDM", batch=ofdm_cfg["batch"], ofdm_cfg=ofdm_cfg)
        it_otfs = iter_comm_batches(train_spec_otfs, kind="OTFS", batch=otfs_cfg["batch"], otfs_cfg=otfs_cfg)
        pbar = tqdm(range(comm_steps_per_epoch), desc=f"Comm train ep{ep}/{comm_epochs}")
        for _ in pbar:
            # OFDM step
            Xo,Yo,eb = next(it_ofdm, None)
            if Xo is None: break
            Xo,Yo = Xo.to(DEVICE), Yo.to(DEVICE)
            ofdm_opt.zero_grad()
            lo = F.binary_cross_entropy_with_logits(ofdm_model(Xo), Yo)
            lo.backward(); ofdm_opt.step()

            # OTFS step
            Xt,Yt,eb2 = next(it_otfs, None)
            if Xt is None: break
            Xt,Yt = Xt.to(DEVICE), Yt.to(DEVICE)
            otfs_opt.zero_grad()
            lt = F.binary_cross_entropy_with_logits(otfs_model(Xt), Yt)
            lt.backward(); otfs_opt.step()

            pbar.set_postfix(ofdm=lo.item(), otfs=lt.item())

        # save checkpoints each epoch
        torch.save({"epoch":ep, "model":ofdm_model.state_dict(), "optim":ofdm_opt.state_dict()}, ofdm_ckpt)
        torch.save({"epoch":ep, "model":otfs_model.state_dict(), "optim":otfs_opt.state_dict()}, otfs_ckpt)

    return net, ofdm_model, otfs_model

# ---------- 4) EVALUATION + VISUALIZATION ----------
def evaluate_and_visualize(
    out_dir,
    sp: SystemParams,
    radar_net, ofdm_model, otfs_model,
    gts_eval=None
):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # (a) Make one evaluation scene (or use given)
    if gts_eval is None:
        gts_eval = [
            {'c':[20,  0, 1], 's':[4,2,2], 'v':[ 12,  0, 0]},
            {'c':[50, -5, 1], 's':[5,3,3], 'v':[-18,  5, 0]}
        ]

    print("[EVAL] Simulating eval scene...")
    pts, its, vels = raycast_torch(sp, gts_eval)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    (rd_f_db,) = fmcw_torch(pts, its, vels, sp)
    (rd_o_db,) = otfs_torch(pts, its, vels, sp)

    # 2D/3D comparisons + detections
    det_f_mask, ra_f, va_f, noise_f, snr_f = viz_rd_2d_compare(out/"compare_2d.pdf", rd_f_db, rd_o_db, gts_eval, sp)
    viz_rd_3d_compare(out/"compare_3d.pdf", rd_f_db, rd_o_db, gts_eval, sp)
    viz_rd_3d_with_dets(out/"fmcw_3d_with_dets.pdf", rd_f_db, ra_f, va_f, det_f_mask, gts_eval, sp, title="FMCW RD with Detections & GT")

    cfar_otfs_cfg = dict(train=(10,8), guard=(2,2), pfa=1e-4, min_snr_db=6.0, notch_doppler_bins=0, apply_nms=True, max_peaks=80)
    det_o_mask = cfar2d_ca(rd_o_db, **cfar_otfs_cfg)
    ra_o, va_o = sp.otfs_axes()
    viz_rd_3d_with_dets(out/"otfs_3d_with_dets.pdf", rd_o_db, ra_o, va_o, det_o_mask, gts_eval, sp, title="OTFS Delay–Doppler with Detections & GT")

    # BEV compare (CFAR)
    viz_bev_scene(out/"scene", pts, gts_eval, sp)
    dets_f = extract_detections(rd_f_db, det_f_mask, ra_f, va_f, snr_db=snr_f)
    dets_o = extract_detections(rd_o_db, det_o_mask, ra_o, va_o)
    viz_scene_bev_compare(out/"scene_bev_compare.pdf", dets_f, dets_o, gts_eval, sp)

    # Radar DL vs CFAR on this scene
    gtinfo = _gt_rv_az(gts_eval, sp)
    pairs_cfar = _match_dets_to_gts(dets_f, gtinfo, w_r=1.0, w_v=0.5)
    metrics_cfar, _, _ = _compute_metrics_from_pairs(pairs_cfar, gtinfo, sp)

    rd_in = torch.from_numpy(_rd_normalize(rd_f_db))[None,None].to(DEVICE)
    radar_net.eval()
    with torch.no_grad():
        logits = radar_net(rd_in)
    dets_dl = rd_dl_infer_to_points(logits, ra_f, va_f, thr=0.40, max_peaks=32)
    pairs_dl   = _match_dets_to_gts(dets_dl, gtinfo, w_r=1.0, w_v=0.5)
    metrics_dl, _, _ = _compute_metrics_from_pairs(pairs_dl, gtinfo, sp)
    print("[Radar] CFAR metrics:", metrics_cfar)
    print("[Radar] DL   metrics:", metrics_dl)

    # BER sweeps: baseline & theory
    print("[EVAL] BER sweeps...")
    eb_axis, ber_ofdm, ber_otfs, ber_theory = run_ber_sweep_and_plot(out/"ber_compare.pdf",
        ebn0_db_list=np.arange(0,21,2),
        ofdm_cfg=dict(Nfft=256, cp_len=32, n_ofdm_sym=800),
        otfs_cfg=dict(M=64, N=256, cp_len=32),
        rng_seed=2025)

    # DL demapper curves (reuse batch generators at those SNRs)
    ofdm_cfg = dict(Nfft=256, cp_len=32, n_sym=8, batch=8)
    otfs_cfg = dict(M=64, N=256, cp_len=32, batch=6)
    #ber_ofdm_dl = comm_demap_ber_curve(ofdm_model, lambda eb, **kw: comm_dl_gen_batch_OFDM(eb, **ofdm_cfg), eb_axis)
    #ber_otfs_dl = comm_demap_ber_curve(otfs_model, lambda eb, **kw: comm_dl_gen_batch_OTFS(eb, **otfs_cfg), eb_axis)

    ber_ofdm_dl = comm_demap_ber_curve(ofdm_model, comm_dl_gen_batch_OFDM, ofdm_cfg, eb_axis)
    ber_otfs_dl = comm_demap_ber_curve(otfs_model, comm_dl_gen_batch_OTFS, otfs_cfg, eb_axis)

    # Combined plot
    plt.figure(figsize=(8,6))
    plt.semilogy(eb_axis, ber_ofdm+1e-12, 'o-', label='OFDM baseline (hard QPSK)')
    plt.semilogy(eb_axis, ber_otfs+1e-12, 's-', label='OTFS baseline (hard QPSK)')
    plt.semilogy(eb_axis, ber_ofdm_dl+1e-12,  'o--', label='OFDM DL demapper')
    plt.semilogy(eb_axis, ber_otfs_dl+1e-12,  's--', label='OTFS DL demapper')
    plt.semilogy(eb_axis, ber_theory+1e-12,   'k:',  label='Theory QPSK (AWGN)')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.xlabel('Eb/N0 (dB)'); plt.ylabel('BER'); plt.title('Comm BER: Baseline vs DL')
    plt.legend(); plt.tight_layout(); plt.savefig(out/"ber_compare_with_dl.pdf", dpi=170); plt.close()

    # Persist metrics
    with open(out/"radar_metrics.json","w") as f:
        json.dump({"cfar":metrics_cfar, "dl":metrics_dl}, f, indent=2)

    print("[EVAL] Done.")

from pathlib import Path

def show_example_results(out_dir, sp: SystemParams, n_samples=3, seed=2026):
    """
    Make n_samples random scenes (<=2 cubes), and for each save:
      - scene_bev_scene.png
      - scene_bev_compare.png
      - fmcw_3d_with_dets.png
      - otfs_3d_with_dets.png
      - compare_2d.png
      - compare_3d.png
    """
    out_dir = Path(out_dir) / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def _rand_gts_two(sp, rng):
        k = int(rng.integers(1, 3))  # 1 or 2 targets
        gts=[]
        for _ in range(k):
            r  = float(rng.uniform(8, 48))
            az = float(rng.uniform(-np.deg2rad(sp.az_fov/2), np.deg2rad(sp.az_fov/2)))
            x, y = r*np.cos(az), r*np.sin(az)
            vx   = float(rng.uniform(-20, 20))
            vy   = float(rng.uniform(-6, 6))
            gts.append({'c':[x, y, 1.0], 's':[4.0,2.0,2.0], 'v':[vx,vy,0.0]})
        return gts

    cfar_otfs_cfg = dict(train=(10,8), guard=(2,2), pfa=1e-4,
                         min_snr_db=6.0, notch_doppler_bins=0,
                         apply_nms=True, max_peaks=80)

    for i in range(n_samples):
        sample_dir = out_dir / f"sample_{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # ---- generate one random scene ----
        gts = _rand_gts_two(sp, rng)
        print(f"[EXAMPLE] sample {i}: {len(gts)} target(s)")

        pts, its, vels = raycast_torch(sp, gts)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        (rd_f_db,) = fmcw_torch(pts, its, vels, sp)   # (M, N//2)
        (rd_o_db,) = otfs_torch(pts, its, vels, sp)   # (M, N)

        # ---- 2D/3D comparison figures ----
        det_f_mask, ra_f, va_f, noise_f, snr_f = viz_rd_2d_compare(
            sample_dir/"compare_2d.pdf", rd_f_db, rd_o_db, gts, sp
        )
        viz_rd_3d_compare(sample_dir/"compare_3d.pdf", rd_f_db, rd_o_db, gts, sp)

        # ---- FMCW 3D with CFAR dets ----
        viz_rd_3d_with_dets(
            sample_dir/"fmcw_3d_with_dets.pdf",
            rd_f_db, ra_f, va_f, det_f_mask, gts, sp,
            title="FMCW RD with Detections & GT"
        )

        # ---- OTFS 3D with CFAR dets ----
        det_o_mask = cfar2d_ca(rd_o_db, **cfar_otfs_cfg)
        ra_o, va_o = sp.otfs_axes()
        viz_rd_3d_with_dets(
            sample_dir/"otfs_3d_with_dets.pdf",
            rd_o_db, ra_o, va_o, det_o_mask, gts, sp,
            title="OTFS Delay–Doppler with Detections & GT"
        )

        # ---- BEV scene + BEV compare (CFAR) ----
        viz_bev_scene(sample_dir/"scene", pts, gts, sp)  # saves scene_bev_scene.png
        dets_f = extract_detections(rd_f_db, det_f_mask, ra_f, va_f, snr_db=snr_f)
        dets_o = extract_detections(rd_o_db, det_o_mask, ra_o, va_o)
        viz_scene_bev_compare(sample_dir/"scene_bev_compare.pdf", dets_f, dets_o, gts, sp)

    print(f"[EXAMPLE] Saved {n_samples} sample result sets to: {out_dir}")

def dataset_exists(root: str) -> bool:
    root = Path(root)
    have_radar = (root/"radar"/"train").exists() and any((root/"radar"/"train").glob("*.npz")) \
              and (root/"radar"/"val").exists()   and any((root/"radar"/"val").glob("*.npz"))
    have_comm  = (root/"comm"/"train_spec.json").exists() and (root/"comm"/"val_spec.json").exists()
    return have_radar and have_comm

def simulate_if_missing(out_dir, sp, **kwargs):
    if dataset_exists(out_dir):
        print(f"[DATA] Found existing dataset under {out_dir} — skip simulation.")
        return
    print(f"[DATA] Simulating dataset → {out_dir}")
    simulate_dataset(out_dir=out_dir, sp=sp, **kwargs)  # uses your existing simulator

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

def _metrics_add(acc, m):
    for k in ["TP","FP","detected_gts","total_gts"]:
        acc[k] = acc.get(k,0) + int(m.get(k,0))
    return acc

def _metrics_finalize(m):
    tp, fp = m.get("TP",0), m.get("FP",0)
    tot = m.get("total_gts",1)
    det = m.get("detected_gts",0)
    precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
    recall    = det / tot if tot>0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    m.update(dict(precision=precision, recall=recall, f1=f1))
    return m

def _tp_errors(pairs, gtinfo):
    """Return arrays of |Δr|, |Δv| for TPs (after cube-in-bounds check inside _compute_metrics_from_pairs)."""
    # We re-run the inside-cube check to classify TP/FP same way _compute_metrics_from_pairs does.
    dr, dv = [], []
    # NOTE: pairs: list of (det, gi, cost)
    for det, gi, _ in pairs:
        g = gtinfo[gi]
        # projected (x,y) and inside check are enforced in _compute_metrics_from_pairs; to approximate here, 
        # collect errors for all pairs (good proxy)—or if you want exact TP-only, pass back TP indices from _compute_metrics_from_pairs.
        r_err = abs(det['r'] - g['r'])
        v_err = abs(det['v'] - g['v'])
        dr.append(r_err); dv.append(v_err)
    return np.array(dr, float), np.array(dv, float)

@torch.no_grad()
def _detect_dl_on_fmcw(rd_f_db, ra_f, va_f, net, thr=0.40, max_peaks=64):
    x = torch.from_numpy(_rd_normalize(rd_f_db))[None,None].to(DEVICE)
    net.eval()
    logits = net(x)
    return rd_dl_infer_to_points(logits, ra_f, va_f, thr=thr, max_peaks=max_peaks)

def _detect_cfar_fmcw(rd_f_db, ra_f, va_f, snr_db=None, cfar_cfg=None):
    mask = cfar2d_ca(rd_f_db, **(cfar_cfg or dict(train=(10,8), guard=(2,2), pfa=1e-4, min_snr_db=6.0)))
    return extract_detections(rd_f_db, mask, ra_f, va_f, snr_db)

def _detect_cfar_otfs_from_gts(gts, sp, cfar_cfg=None):
    # Re-generate OTFS RD for the same scene, then CFAR
    pts, its, vels = raycast_torch(sp, gts)
    if DEVICE.type=="cuda": torch.cuda.synchronize()
    (rd_o_db,) = otfs_torch(pts, its, vels, sp)
    ra_o, va_o = sp.otfs_axes()
    mask = cfar2d_ca(rd_o_db, **(cfar_cfg or dict(train=(10,8), guard=(2,2), pfa=1e-4, min_snr_db=6.0)))
    dets = extract_detections(rd_o_db, mask, ra_o, va_o, snr_db=None)
    return dets, rd_o_db, ra_o, va_o

def eval_radar_on_val_set(
    data_root,
    sp: SystemParams,
    radar_net,
    out_dir,
    dl_thr_sweep=(0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    cfar_pfa_sweep=(1e-2, 1e-3, 1e-4, 1e-5),
    do_otfs=True,
    max_samples=None
):
    data_root = Path(data_root)
    out_dir   = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    val_files = sorted((data_root/"radar"/"val").glob("*.npz"))
    if max_samples: val_files = val_files[:max_samples]

    # Accumulators
    acc_dl_fixed = dict(); acc_cfar_fixed = dict()
    dl_pr = []      # (thr, precision, recall, f1)
    cfar_pr = []    # (pfa, precision, recall, f1)
    dr_dl, dv_dl = [], []
    dr_cf, dv_cf = [], []

    # Fixed settings (for headline bars)
    dl_thr_fixed = 0.40
    cfar_fixed   = dict(train=(10,8), guard=(2,2), pfa=1e-4, min_snr_db=6.0, apply_nms=True, max_peaks=80)

    ra_f, va_f = sp.fmcw_axes()

    # Sweep DL thresholds
    for thr in dl_thr_sweep:
        acc = dict()
        for f in val_files:
            z = np.load(f, allow_pickle=True)
            rd_f_db = z["rd_f_db"].astype(np.float32)
            gts = json.loads(str(z["gts"]))
            # DL dets
            dets_dl = _detect_dl_on_fmcw(rd_f_db, ra_f, va_f, radar_net, thr=thr)
            gtinfo = _gt_rv_az(gts, sp)
            pairs  = _match_dets_to_gts(dets_dl, gtinfo, w_r=1.0, w_v=0.5)
            metrics, _, _ = _compute_metrics_from_pairs(pairs, gtinfo, sp)
            _metrics_add(acc, metrics)
        _metrics_finalize(acc)
        dl_pr.append((thr, acc["precision"], acc["recall"], acc["f1"]))
        if abs(thr - dl_thr_fixed) < 1e-9:
            acc_dl_fixed = acc.copy()

    # Sweep CFAR Pfas (FMCW only; uses saved RD maps)
    for pfa in cfar_pfa_sweep:
        acc = dict()
        cfg = dict(train=(10,8), guard=(2,2), pfa=pfa, min_snr_db=6.0, apply_nms=True, max_peaks=80)
        for f in val_files:
            z = np.load(f, allow_pickle=True)
            rd_f_db = z["rd_f_db"].astype(np.float32)
            gts = json.loads(str(z["gts"]))
            dets_cf = _detect_cfar_fmcw(rd_f_db, ra_f, va_f, snr_db=None, cfar_cfg=cfg)
            gtinfo = _gt_rv_az(gts, sp)
            pairs  = _match_dets_to_gts(dets_cf, gtinfo, w_r=1.0, w_v=0.5)
            metrics, _, _ = _compute_metrics_from_pairs(pairs, gtinfo, sp)
            _metrics_add(acc, metrics)
        _metrics_finalize(acc)
        cfar_pr.append((pfa, acc["precision"], acc["recall"], acc["f1"]))
        if abs(pfa - cfar_fixed["pfa"]) < 1e-16:
            acc_cfar_fixed = acc.copy()

    # Error CDFs at the fixed settings (collect TP-pair errors approximately)
    for f in val_files:
        z = np.load(f, allow_pickle=True)
        rd_f_db = z["rd_f_db"].astype(np.float32)
        gts = json.loads(str(z["gts"]))
        gtinfo = _gt_rv_az(gts, sp)

        # DL fixed
        dets_dl = _detect_dl_on_fmcw(rd_f_db, ra_f, va_f, radar_net, thr=dl_thr_fixed)
        pairs   = _match_dets_to_gts(dets_dl, gtinfo, w_r=1.0, w_v=0.5)
        d1, d2  = _tp_errors(pairs, gtinfo)
        if d1.size: dr_dl.append(d1); dv_dl.append(d2)

        # CFAR fixed
        dets_cf = _detect_cfar_fmcw(rd_f_db, ra_f, va_f, snr_db=None, cfar_cfg=cfar_fixed)
        pairs   = _match_dets_to_gts(dets_cf, gtinfo, w_r=1.0, w_v=0.5)
        c1, c2  = _tp_errors(pairs, gtinfo)
        if c1.size: dr_cf.append(c1); dv_cf.append(c2)

    dr_dl = np.concatenate(dr_dl) if dr_dl else np.array([])
    dv_dl = np.concatenate(dv_dl) if dv_dl else np.array([])
    dr_cf = np.concatenate(dr_cf) if dr_cf else np.array([])
    dv_cf = np.concatenate(dv_cf) if dv_cf else np.array([])

    # Optional: OTFS CFAR across val set
    acc_otfs = dict()
    if do_otfs:
        cfg_otfs = dict(train=(10,8), guard=(2,2), pfa=1e-4, min_snr_db=6.0, apply_nms=True, max_peaks=80)
        for f in val_files:
            z = np.load(f, allow_pickle=True)
            gts = json.loads(str(z["gts"]))
            dets_o, _, _, _ = _detect_cfar_otfs_from_gts(gts, sp, cfar_cfg=cfg_otfs)
            gtinfo = _gt_rv_az(gts, sp)
            pairs  = _match_dets_to_gts(dets_o, gtinfo, w_r=1.0, w_v=0.5)
            metrics, _, _ = _compute_metrics_from_pairs(pairs, gtinfo, sp)
            _metrics_add(acc_otfs, metrics)
        _metrics_finalize(acc_otfs)

    # Persist raw results
    with open(out_dir/"val_radar_summary.json","w") as f:
        json.dump({
            "dl_fixed": acc_dl_fixed,
            "cfar_fixed": acc_cfar_fixed,
            "dl_pr": [{"thr":t,"P":p,"R":r,"F1":f1} for (t,p,r,f1) in dl_pr],
            "cfar_pr": [{"pfa":pfa,"P":p,"R":r,"F1":f1} for (pfa,p,r,f1) in cfar_pr],
            "otfs_cfar": acc_otfs if do_otfs else None
        }, f, indent=2)

    # ====== FIGURES ======
    # (A) Bars for headline numbers
    plt.figure(figsize=(6,4))
    names, f1s = ["DL (FMCW)", "CFAR (FMCW)"], [acc_dl_fixed["f1"], acc_cfar_fixed["f1"]]
    plt.bar(names, f1s)
    plt.ylabel("F1"); plt.title("Validation F1 (Fixed Settings)")
    plt.tight_layout(); plt.savefig(out_dir/"val_f1_bars.pdf", dpi=170); plt.close()

    # (B) PR-style sweeps
    plt.figure(figsize=(6.5,4.5))
    plt.plot([x[1] for x in dl_pr],  [x[2] for x in dl_pr], 'o-', label="DL thr sweep")
    plt.plot([x[1] for x in cfar_pr],[x[2] for x in cfar_pr],'s--', label="CFAR pfa sweep")
    plt.xlabel("Precision"); plt.ylabel("Recall"); plt.title("Precision–Recall (Validation)")
    plt.grid(True, alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"val_precision_recall.pdf", dpi=170); plt.close()

    # (C) CDFs of localization errors
    def _cdf(ax, arr, label):
        if arr.size==0:
            ax.plot([], [], label=label); return
        a = np.sort(arr); y = np.linspace(0,1,len(a),endpoint=False)
        ax.plot(a, y, label=label)

    plt.figure(figsize=(12,4))
    ax1 = plt.subplot(1,2,1)
    _cdf(ax1, dr_dl, "DL FMCW |Δr|"); _cdf(ax1, dr_cf, "CFAR FMCW |Δr|")
    ax1.set_xlabel("|Δrange| (m)"); ax1.set_ylabel("CDF"); ax1.grid(True, alpha=0.4); ax1.set_title("Range Error CDF"); ax1.legend()

    ax2 = plt.subplot(1,2,2)
    _cdf(ax2, dv_dl, "DL FMCW |Δv|"); _cdf(ax2, dv_cf, "CFAR FMCW |Δv|")
    ax2.set_xlabel("|Δvelocity| (m/s)"); ax2.set_ylabel("CDF"); ax2.grid(True, alpha=0.4); ax2.set_title("Velocity Error CDF"); ax2.legend()

    plt.tight_layout(); plt.savefig(out_dir/"val_error_cdfs.pdf", dpi=170); plt.close()

    # (D) If OTFS requested, add a small summary bar
    if do_otfs:
        plt.figure(figsize=(4.5,3.5))
        plt.bar(["CFAR (OTFS)"], [acc_otfs["f1"]])
        plt.ylabel("F1"); plt.title("Validation F1 (OTFS CFAR)")
        plt.tight_layout(); plt.savefig(out_dir/"val_otfs_f1.pdf", dpi=170); plt.close()

    print(f"[VAL] Saved figures to {out_dir}")

def val_montage_examples(data_root, sp, radar_net, out_dir, k=4, seed=42):
    """Dump k random validation samples with: compare_2d/3d, *_3d_with_dets, scene_bev_compare."""
    rng = np.random.default_rng(seed)
    data_root = Path(data_root)
    val_files = sorted((data_root/"radar"/"val").glob("*.npz"))
    if not val_files: 
        print("[VAL] No validation files found."); return
    picks = list(rng.choice(val_files, size=min(k,len(val_files)), replace=False))
    out_dir = Path(out_dir)/"val_examples"; out_dir.mkdir(parents=True, exist_ok=True)

    cfar_otfs_cfg = dict(train=(10,8), guard=(2,2), pfa=1e-4, min_snr_db=6.0, apply_nms=True, max_peaks=80)
    ra_f, va_f = sp.fmcw_axes()

    for i, f in enumerate(picks):
        z = np.load(f, allow_pickle=True)
        rd_f_db = z["rd_f_db"].astype(np.float32)
        gts = json.loads(str(z["gts"]))
        # Rebuild OTFS for the same gts
        pts, its, vels = raycast_torch(sp, gts); 
        if DEVICE.type=="cuda": torch.cuda.synchronize()
        (rd_o_db,) = otfs_torch(pts, its, vels, sp)

        sample_dir = out_dir/f"sample_{i:02d}"; sample_dir.mkdir(exist_ok=True)
        # 2D/3D compares
        det_f_mask, ra_f, va_f, noise_f, snr_f = viz_rd_2d_compare(sample_dir/"compare_2d.pdf", rd_f_db, rd_o_db, gts, sp)
        viz_rd_3d_compare(sample_dir/"compare_3d.pdf", rd_f_db, rd_o_db, gts, sp)

        # FMCW 3D with dets (CFAR)
        viz_rd_3d_with_dets(sample_dir/"fmcw_3d_with_dets.pdf", rd_f_db, ra_f, va_f, det_f_mask, gts, sp, title="FMCW RD with Detections & GT")

        # OTFS 3D with dets (CFAR)
        det_o_mask = cfar2d_ca(rd_o_db, **cfar_otfs_cfg)
        ra_o, va_o = sp.otfs_axes()
        viz_rd_3d_with_dets(sample_dir/"otfs_3d_with_dets.pdf", rd_o_db, ra_o, va_o, det_o_mask, gts, sp, title="OTFS Delay–Doppler with Detections & GT")

        # BEV compare (CFAR)
        dets_f = extract_detections(rd_f_db, det_f_mask, ra_f, va_f, snr_db=snr_f)
        dets_o = extract_detections(rd_o_db, det_o_mask, ra_o, va_o)
        viz_scene_bev_compare(sample_dir/"scene_bev_compare.pdf", dets_f, dets_o, gts, sp)

def new_training():
    # ============== paths & params ==============
    root = "./output/isac_c4"
    Path(root).mkdir(parents=True, exist_ok=True)
    ckpts = Path(root)/"checkpoints"
    sp = SystemParams()

    # ============== (1) DATASET ==============
    # simulate_dataset(
    #     out_dir=root,
    #     sp=sp,
    #     n_train=1500,
    #     n_val=300,
    #     seed=2025,
    #     snr_list=(0,2,4,6,8,10,12,14,16,18,20)
    # )
    simulate_if_missing(root, sp, n_train=1500, n_val=300, seed=2025,
                    snr_list=(0,2,4,6,8,10,12,14,16,18,20))

    # ============== (2) TRAIN (radar + comm) ==============
    radar_net, ofdm_model, otfs_model = train_both_models(
        data_root=root,
        ckpt_dir=ckpts,
        sp=sp,
        radar_epochs=6,
        radar_batch=6,
        radar_lr=1e-3,
        comm_epochs=5,
        comm_steps_per_epoch=200,
        ofdm_cfg=dict(Nfft=256, cp_len=32, n_sym=8, batch=8),
        otfs_cfg=dict(M=64, N=256, cp_len=32, batch=6),
        resume=True,   # set False to start fresh
    )

    # ============== (3) EVALUATION + (4) VIS ==============
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

    show_example_results(root, sp, n_samples=10, seed=2027)


def test_training():
        # ============== paths & params ==============
    root = "./output/isac_c3"
    os.makedirs(root, exist_ok=True)
    sp = SystemParams()

    # 场景目标（保持与你之前一致；注意修正引号）
    gts = [
        {'c':[20,  0, 1], 's':[4,2,2], 'v':[ 12,  0, 0]},
        {'c':[50, -5, 1], 's':[5,3,3], 'v':[-18,  5, 0]}
    ]

    # ============== pipeline ==============
    print("Simulating raycast...")
    pts, its, vels = raycast_torch(sp, gts)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    print(f"Raycast hits: {len(pts)}")

    print("Synthesizing FMCW & OTFS RD maps...")
    (rd_f_db,) = fmcw_torch(pts, its, vels, sp)   # shape (M, N//2)
    (rd_o_db,) = otfs_torch(pts, its, vels, sp)   # shape (M, N)

    print("Saving 2D/3D RD comparisons...")
    # 2D 对比（会返回 FMCW 的 CFAR mask 与坐标轴）
    det_f_mask, ra_f, va_f, noise_f, snr_f = viz_rd_2d_compare(
        f"{root}/compare_2d.pdf", rd_f_db, rd_o_db, gts, sp
    )
    # 3D 曲面对比（FMCW/OTFS）
    viz_rd_3d_compare(f"{root}/compare_3d.pdf", rd_f_db, rd_o_db, gts, sp)

    # ============== FMCW: 3D with dets ==============
    viz_rd_3d_with_dets(
        f"{root}/fmcw_3d_with_dets.pdf",
        rd_f_db, ra_f, va_f, det_f_mask, gts, sp,
        title="FMCW RD with Detections & GT"
    )

    # ============== OTFS: CFAR + 3D with dets ==============
    cfar_otfs_cfg = dict(
        train=(10, 8), guard=(2, 2),
        pfa=1e-4, min_snr_db=6.0,
        notch_doppler_bins=0,   # DD 域无直流杂波
        apply_nms=True, max_peaks=80
    )
    det_o_mask = cfar2d_ca(rd_o_db, **cfar_otfs_cfg)
    ra_o, va_o = sp.otfs_axes()

    viz_rd_3d_with_dets(
        f"{root}/otfs_3d_with_dets.pdf",
        rd_o_db, ra_o, va_o, det_o_mask, gts, sp,
        title="OTFS Delay–Doppler with Detections & GT"
    )

    # ============== BEV: 场景 & 对比评估 ==============
    print("Saving BEV figures...")
    # 场景 3D 视图（0–50m）
    viz_bev_scene(f"{root}/scene", pts, gts, sp)

    # 提取 FMCW/OTFS 的 (r,v) 检测，生成双面板 BEV（含 TP/FP/Precision/Recall/F1）
    dets_f = extract_detections(rd_f_db, det_f_mask, ra_f, va_f, snr_db=snr_f)
    dets_o = extract_detections(rd_o_db, det_o_mask, ra_o, va_o)

    viz_scene_bev_compare(
        f"{root}/scene_bev_compare.pdf",
        dets_f, dets_o, gts, sp
    )

        # ============== COMM: BER curves (OFDM for FMCW comm, OTFS for OTFS comm) ==============
    print("Running BER sweeps (communication)...")
    eb_axis, ber_ofdm, ber_otfs, ber_theory = run_ber_sweep_and_plot(
        f"{root}/ber_compare.pdf",
        ebn0_db_list=np.arange(0, 21, 2),
        ofdm_cfg=dict(Nfft=256, cp_len=32, n_ofdm_sym=600),
        otfs_cfg=dict(M=64, N=256, cp_len=32),
        rng_seed=2025
    )
    print("BER figure saved to ber_compare.pdf")

    print("Done.")

        # ===================== Radar DL vs CFAR =====================
    print("Training Radar DL detector...")
    radar_net = train_radar_model(sp, epochs=6, batch=6, lr=1e-3, n_train=1200, n_val=300)

    # Evaluate on a fresh scene (your current gts)
    ra_f, va_f = sp.fmcw_axes()
    det_f_mask, ra_f, va_f, noise_f, snr_f = viz_rd_2d_compare(f"{root}/compare_2d.pdf", rd_f_db, rd_o_db, gts, sp)
    # CFAR detections:
    dets_cfar = extract_detections(rd_f_db, det_f_mask, ra_f, va_f, snr_db=snr_f)
    # DL detections:
    rd_in = torch.from_numpy(_rd_normalize(rd_f_db))[None,None].to(DEVICE)
    radar_net.eval()
    with torch.no_grad():
        logits = radar_net(rd_in)
    dets_dl = rd_dl_infer_to_points(logits, ra_f, va_f, thr=0.40, max_peaks=32)

    # Compare metrics using your existing BEV evaluator (cube-in-bounds)
    # (reuse helper utilities from your earlier BEV section)
    # CFAR
    gtinfo = _gt_rv_az(gts, sp)
    pairs_cfar = _match_dets_to_gts(dets_cfar, gtinfo, w_r=1.0, w_v=0.5)
    metrics_cfar, _, _ = _compute_metrics_from_pairs(pairs_cfar, gtinfo, sp)
    # DL
    pairs_dl   = _match_dets_to_gts(dets_dl, gtinfo, w_r=1.0, w_v=0.5)
    metrics_dl, _, _ = _compute_metrics_from_pairs(pairs_dl, gtinfo, sp)
    print("[Radar] CFAR metrics:", metrics_cfar)
    print("[Radar] DL metrics  :", metrics_dl)

    # ===================== Communication DL vs Baseline =====================
    print("Training OFDM demapper DL...")
    ofdm_cfg = dict(Nfft=256, cp_len=32, n_sym=8, batch=8)
    ofdm_model = CommDemapperCNN(in_ch=2)
    ofdm_model = train_comm_demap(ofdm_model, comm_dl_gen_batch_OFDM, ofdm_cfg, snr_min=0, snr_max=18, epochs=5, steps_per_epoch=200, tag="OFDM")

    print("Training OTFS demapper DL...")
    otfs_cfg = dict(M=64, N=256, cp_len=32, batch=6)
    otfs_model = CommDemapperCNN(in_ch=2)
    otfs_model = train_comm_demap(otfs_model, comm_dl_gen_batch_OTFS, otfs_cfg, snr_min=0, snr_max=18, epochs=5, steps_per_epoch=200, tag="OTFS")

    # BER curves vs classical (hard decision)
    eb_axis = np.arange(0, 21, 2)
    ber_ofdm_dl  = comm_demap_ber_curve(ofdm_model, comm_dl_gen_batch_OFDM, ofdm_cfg, eb_axis)
    #ber_otfs_dl  = comm_demap_ber_curve(otfs_model, comm_dl_gen_batch_OTFS, otfs_cfg, eb_axis)
    ber_otfs_dl = comm_demap_ber_curve(ofdm_model, comm_dl_gen_batch_OTFS, otfs_cfg, eb_axis)  # <-- same model
    
    # Classical baselines already available (we also computed theory earlier):
    _, ber_ofdm_base, ber_otfs_base, ber_theory = run_ber_sweep_and_plot(f"{root}/ber_compare.pdf",
                                                                          ebn0_db_list=eb_axis,
                                                                          ofdm_cfg=dict(Nfft=256, cp_len=32, n_ofdm_sym=800),
                                                                          otfs_cfg=dict(M=64, N=256, cp_len=32),
                                                                          rng_seed=2025)

    # Plot combined comparison
    plt.figure(figsize=(8,6))
    plt.semilogy(eb_axis, ber_ofdm_base+1e-12, 'o-', label='OFDM baseline (hard QPSK)')
    plt.semilogy(eb_axis, ber_otfs_base+1e-12, 's-', label='OTFS baseline (hard QPSK)')
    plt.semilogy(eb_axis, ber_ofdm_dl+1e-12,  'o--', label='OFDM DL demapper')
    plt.semilogy(eb_axis, ber_otfs_dl+1e-12,  's--', label='OTFS DL demapper')
    plt.semilogy(eb_axis, ber_theory+1e-12,   'k:',  label='Theory QPSK (AWGN)')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.xlabel('Eb/N0 (dB)'); plt.ylabel('BER'); plt.title('Comm BER: Baseline vs DL')
    plt.legend(); plt.tight_layout(); plt.savefig(f"{root}/ber_compare_with_dl.pdf", dpi=170); plt.close()

from pathlib import Path
import torch

# ---- loader: radar model from checkpoints ----
def load_radar_from_ckpt(root, device=None, prefer_best=True):
    """
    Load trained radar UNet from <root>/checkpoints.
    If prefer_best=True, try 'radar_unet_best_only.pt' first; else fall back to 'radar_unet.pt'.
    """
    device = device or DEVICE
    ckpt_dir = Path(root) / "checkpoints"
    best_path = ckpt_dir / "radar_unet_best_only.pt"
    full_path = ckpt_dir / "radar_unet.pt"

    net = UNetLite().to(device)
    if prefer_best and best_path.exists():
        state = torch.load(best_path, map_location=device)
        # best_only is weights only
        net.load_state_dict(state, strict=True)
        print(f"[LOAD] Loaded radar model (best weights): {best_path}")
    elif full_path.exists():
        state = torch.load(full_path, map_location=device)
        net.load_state_dict(state["model"], strict=True)
        print(f"[LOAD] Loaded radar model (last checkpoint): {full_path} (epoch {state.get('epoch')})")
    else:
        raise FileNotFoundError(
            f"No radar checkpoint found in {ckpt_dir}.\n"
            f"Expected one of: {best_path.name} or {full_path.name}"
        )
    net.eval()
    return net

# ---- convenience: one-call validation entrypoint ----
def run_validation_from_root(
    root,
    dl_thr_sweep=(0.25,0.30,0.35,0.40,0.45,0.50),
    cfar_pfa_sweep=(1e-2,1e-3,1e-4,1e-5),
    do_otfs=True,
    max_samples=None
):
    """
    Loads radar model + runs validation figures using eval_radar_on_val_set.
    Assumes dataset exists under:
        <root>/radar/val/*.npz
    And checkpoints under:
        <root>/checkpoints/
    Outputs figures to:
        <root>/val_eval/
    """
    root = Path(root)
    # 1) build params
    sp = SystemParams()
    # 2) load model
    radar_net = load_radar_from_ckpt(root, device=DEVICE, prefer_best=True)
    # 3) run eval (this function reads val npz files from disk)
    out_dir = root / "val_eval"
    eval_radar_on_val_set(
        data_root=root,
        sp=sp,
        radar_net=radar_net,
        out_dir=out_dir,
        dl_thr_sweep=dl_thr_sweep,
        cfar_pfa_sweep=cfar_pfa_sweep,
        do_otfs=do_otfs,
        max_samples=max_samples
    )
    print(f"[VAL] Finished. See figures in: {out_dir}")

# def new_evaluation(root = "./output/isac_c4"):
#     # Run quantitative validation and figures
#     eval_radar_on_val_set(
#         data_root=root,
#         sp=sp,
#         radar_net=radar_net,
#         out_dir=Path(root)/"val_eval",
#         dl_thr_sweep=(0.25,0.30,0.35,0.40,0.45,0.50),
#         cfar_pfa_sweep=(1e-2,1e-3,1e-4,1e-5),
#         do_otfs=True,
#         max_samples=None   # or an int to speed up
#     )

#     # (Optional) dump k qualitative samples from val set
#     val_montage_examples(root, sp, radar_net, Path(root)/"val_eval", k=4, seed=2027)

from pathlib import Path
import torch, json, numpy as np
import matplotlib.pyplot as plt

def load_two_radars_from_ckpt(root, device=None, prefer_best=True):
    """
    Looks for:
      - <root>/checkpoints/radar_unet_fmcw_best_only.pt (or radar_unet_best_only.pt)  ← FMCW model
      - <root>/checkpoints/radar_unet_otfs_best_only.pt                               ← OTFS model (optional)
    Falls back to the generic 'radar_unet.pt' for FMCW if best_only missing.
    If OTFS-specific model is missing, we reuse FMCW model for OTFS.
    """
    device = device or DEVICE
    ckpt = Path(root) / "checkpoints"

    # FMCW
    cand_fm = [
        ckpt/"radar_unet_fmcw_best_only.pt",
        ckpt/"radar_unet_best_only.pt",          # legacy name
    ]
    fallback_fm = ckpt/"radar_unet.pt"

    fm = UNetLite().to(device)
    loaded_fm = False
    for p in cand_fm:
        if p.exists():
            fm.load_state_dict(torch.load(p, map_location=device), strict=False)
            print(f"[LOAD] FMCW DL model: {p}")
            loaded_fm = True
            break
    if not loaded_fm:
        if fallback_fm.exists():
            s = torch.load(fallback_fm, map_location=device)
            fm.load_state_dict(s["model"], strict=False)
            print(f"[LOAD] FMCW DL model (last ckpt): {fallback_fm}")
            loaded_fm = True
        else:
            raise FileNotFoundError("No FMCW radar checkpoint found.")

    # OTFS (optional)
    otfs_path = ckpt/"radar_unet_otfs_best_only.pt"
    if otfs_path.exists():
        ot = UNetLite().to(device)
        ot.load_state_dict(torch.load(otfs_path, map_location=device), strict=False)
        print(f"[LOAD] OTFS DL model: {otfs_path}")
    else:
        ot = fm  # reuse FMCW net if no OTFS-specific net
        print("[LOAD] OTFS DL model not found; reusing FMCW model for OTFS.")

    fm.eval(); ot.eval()
    return fm, ot

# @torch.no_grad()
# def _dl_detect(rd_db, ra, va, net, thr=0.40, max_peaks=64):
#     x = torch.from_numpy(_rd_normalize(rd_db))[None,None].to(DEVICE)
#     logits = net.eval()(x)
#     return rd_dl_infer_to_points(logits, ra, va, thr=thr, max_peaks=max_peaks)

# @torch.no_grad()
# def _dl_detect(rd_db, ra, va, net, thr=0.40, max_peaks=64, adapt=False, domain="fmcw"):
#     x = torch.from_numpy(_rd_normalize(rd_db))[None,None].to(DEVICE)
#     net.eval()
#     #logits = net(x)                              # (1,1,H,W)
#     logits = net.forward_radar(x, domain=domain)
#     if adapt:                                    # enable for OTFS
#         m = logits.mean(); 
#         s = logits.std().clamp_min(1e-6)
#         logits = (logits - m) / s * 2.5         # z-score + temperature
#     prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    
#     #
#     #return heatmap_to_points(prob, ra, va, thr=thr, max_peaks=max_peaks)
    
#     #keep everything in PyTorch and your rd_dl_infer_to_points operates directly on GPU tensors
#     return rd_dl_infer_to_points(torch.from_numpy(prob)[None,None].to(DEVICE),
#                              ra, va, thr=thr, max_peaks=max_peaks)

@torch.no_grad()
def _dl_detect(rd_db, ra, va, net, thr=0.40, max_peaks=64, adapt=True, domain="fmcw"):
    """
    rd_db: (H,W) input map in dB (or any scale; we'll normalize)
    net  : RadarCommNet (must use radar head with domain='fmcw'/'otfs')
    Returns: list of dicts with keys {'r','v','score','y','x'} (r=range/delay, v=doppler)
    """
    # normalize and to torch
    x = torch.from_numpy(_rd_normalize(rd_db))[None, None].to(DEVICE)
    net.eval()

    # --- Forward the correct radar head ---
    logits = net.forward_radar(x, domain=domain)  # (1,1,H,W)

    # --- Adaptive calibration (stabilizes thresholds across frames) ---
    if adapt:
        m = logits.mean()
        s = logits.std().clamp_min(1e-6)
        logits = (logits - m) / s * 2.5

    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    # --- Peak picking (simple threshold + local maxima) ---
    # Build a binary mask and then NMS by keeping top 'max_peaks' by score
    mask = prob >= float(thr)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return []

    scores = prob[ys, xs]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    dets = []
    used = np.zeros_like(prob, dtype=bool)
    radius = 2  # local suppression radius

    for y, x, sc in zip(ys, xs, scores):
        if used[max(0,y-radius):y+radius+1, max(0,x-radius):x+radius+1].any():
            continue
        # Sub-pixel refine around this peak
        dy, dx = _subpixel_quadratic(prob, int(y), int(x))
        r, v = _rv_from_idx_with_subpix(int(y), int(x), dy, dx, ra, va)
        dets.append(dict(r=float(r), v=float(v), score=float(sc), y=int(y), x=int(x)))
        used[max(0,y-radius):y+radius+1, max(0,x-radius):x+radius+1] = True
        if len(dets) >= max_peaks:
            break

    return dets

def _cfar_detect(rd_db, ra, va, snr_db=None, cfar_cfg=None):
    mask = cfar2d_ca(rd_db, **(cfar_cfg or dict(train=(10,8), guard=(2,2), pfa=1e-4, min_snr_db=6.0,
                                                apply_nms=True, max_peaks=80)))
    return extract_detections(rd_db, mask, ra, va, snr_db)


@torch.no_grad()
def eval_radar_on_val_set_dual(
    data_root,
    sp,
    net_fmcw,
    net_otfs,
    out_dir,
    dl_thr_sweep_fmcw=(0.25, 0.30, 0.35, 0.40, 0.45),
    dl_thr_sweep_otfs=(0.05, 0.10, 0.15, 0.20, 0.30),
    cfar_pfa_sweep=(1e-3, 1e-4),
    match_fn=None,             # pass your relaxed 5% cube matcher here
    max_samples=200
):
    """
    Evaluate FMCW & OTFS for CFAR vs DL on the validation split.
    Saves: F1 bars, PR curves, error CDFs to out_dir.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --------- Load validation set ----------
    va_dir = Path(data_root) / "radar" / "val"
    files = sorted(va_dir.glob("*.npz"))
    if max_samples is not None:
        files = files[:max_samples]

    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    # Accumulators for PR (counts per threshold) and error CDFs
    # Format: dict[method] -> dict[thr] -> [tp, fp, fn, errors_r[], errors_v[]]
    methods = ["FMCW-CFAR", "FMCW-DL", "OTFS-CFAR", "OTFS-DL"]
    acc = {m: {} for m in methods}

    # Initialize bins
    for pfa in cfar_pfa_sweep:
        acc["FMCW-CFAR"][pfa] = dict(tp=0, fp=0, fn=0, er_r=[], er_v=[])
        acc["OTFS-CFAR"][pfa] = dict(tp=0, fp=0, fn=0, er_r=[], er_v=[])

    for thr in dl_thr_sweep_fmcw:
        acc["FMCW-DL"][thr] = dict(tp=0, fp=0, fn=0, er_r=[], er_v=[])

    for thr in dl_thr_sweep_otfs:
        acc["OTFS-DL"][thr] = dict(tp=0, fp=0, fn=0, er_r=[], er_v=[])

    # --------- Iterate samples ----------
    for f in files:
        z = np.load(f, allow_pickle=True)
        gts = json.loads(str(z["gts"]))

        # FMCW CFAR
        rd_f = z["rd_f_db"].astype(np.float32)
        detmask_f = None
        for pfa in cfar_pfa_sweep:
            detmask_f = cfar2d_ca(rd_f, pfa=float(pfa))
            dets_f_cfar = extract_detections(rd_f, detmask_f, ra_f, va_f)
            gtinfo = _gt_rv_az(gts, sp)
            if match_fn is not None:
                pairs, unpaired = match_fn(dets_f_cfar, gtinfo)
            else:
                pairs, unpaired = _match_dets_to_gts(dets_f_cfar, gtinfo, w_r=1.0, w_v=0.5)

            tp = len(pairs); fp = len(unpaired); fn = len(gtinfo) - tp
            acc["FMCW-CFAR"][pfa]["tp"] += tp
            acc["FMCW-CFAR"][pfa]["fp"] += fp
            acc["FMCW-CFAR"][pfa]["fn"] += fn

            # errors (use matched pairs)
            for d, g in pairs:
                acc["FMCW-CFAR"][pfa]["er_r"].append(abs(d["r"] - g["r"]))
                acc["FMCW-CFAR"][pfa]["er_v"].append(abs(d["v"] - g["v"]))

        # FMCW DL
        for thr in dl_thr_sweep_fmcw:
            dets_f_dl = _dl_detect(rd_f, ra_f, va_f, net_fmcw, thr=float(thr), adapt=True, domain="fmcw")
            gtinfo = _gt_rv_az(gts, sp)
            if match_fn is not None:
                pairs, unpaired = match_fn(dets_f_dl, gtinfo)
            else:
                pairs, unpaired = _match_dets_to_gts(dets_f_dl, gtinfo, w_r=1.0, w_v=0.5)

            tp = len(pairs); fp = len(unpaired); fn = len(gtinfo) - tp
            acc["FMCW-DL"][thr]["tp"] += tp
            acc["FMCW-DL"][thr]["fp"] += fp
            acc["FMCW-DL"][thr]["fn"] += fn
            for d, g in pairs:
                acc["FMCW-DL"][thr]["er_r"].append(abs(d["r"] - g["r"]))
                acc["FMCW-DL"][thr]["er_v"].append(abs(d["v"] - g["v"]))

        # OTFS CFAR
        rd_o = z["rd_o_db"].astype(np.float32)
        for pfa in cfar_pfa_sweep:
            detmask_o = cfar2d_ca(rd_o, pfa=float(pfa))
            dets_o_cfar = extract_detections(rd_o, detmask_o, ra_o, va_o)
            gtinfo = _gt_rv_az(gts, sp)
            if match_fn is not None:
                pairs, unpaired = match_fn(dets_o_cfar, gtinfo)
            else:
                pairs, unpaired = _match_dets_to_gts(dets_o_cfar, gtinfo, w_r=1.0, w_v=0.5)

            tp = len(pairs); fp = len(unpaired); fn = len(gtinfo) - tp
            acc["OTFS-CFAR"][pfa]["tp"] += tp
            acc["OTFS-CFAR"][pfa]["fp"] += fp
            acc["OTFS-CFAR"][pfa]["fn"] += fn
            for d, g in pairs:
                acc["OTFS-CFAR"][pfa]["er_r"].append(abs(d["r"] - g["r"]))
                acc["OTFS-CFAR"][pfa]["er_v"].append(abs(d["v"] - g["v"]))

        # OTFS DL (lower thresholds by design)
        for thr in dl_thr_sweep_otfs:
            dets_o_dl = _dl_detect(rd_o, ra_o, va_o, net_otfs, thr=float(thr), adapt=True, domain="otfs")
            gtinfo = _gt_rv_az(gts, sp)
            if match_fn is not None:
                pairs, unpaired = match_fn(dets_o_dl, gtinfo)
            else:
                pairs, unpaired = _match_dets_to_gts(dets_o_dl, gtinfo, w_r=1.0, w_v=0.5)

            tp = len(pairs); fp = len(unpaired); fn = len(gtinfo) - tp
            acc["OTFS-DL"][thr]["tp"] += tp
            acc["OTFS-DL"][thr]["fp"] += fp
            acc["OTFS-DL"][thr]["fn"] += fn
            for d, g in pairs:
                acc["OTFS-DL"][thr]["er_r"].append(abs(d["r"] - g["r"]))
                acc["OTFS-DL"][thr]["er_v"].append(abs(d["v"] - g["v"]))

    # --------- Choose best thresholds (max F1) for bars & CDF ----------
    def _best_key(d):
        best_k, best_f1 = None, -1
        for k, S in d.items():
            f1 = _safe_f1(S["tp"], S["fp"], S["fn"])
            if f1 > best_f1:
                best_f1, best_k = f1, k
        return best_k, best_f1

    best_fmcw_cfar_k, best_fmcw_cfar_f1 = _best_key(acc["FMCW-CFAR"])
    best_fmcw_dl_k,   best_fmcw_dl_f1   = _best_key(acc["FMCW-DL"])
    best_otfs_cfar_k, best_otfs_cfar_f1 = _best_key(acc["OTFS-CFAR"])
    best_otfs_dl_k,   best_otfs_dl_f1   = _best_key(acc["OTFS-DL"])

    # --------- Figure 1: F1 bars ----------
    labels = ["FMCW+CFAR", "FMCW+DL", "OTFS+CFAR", "OTFS+DL"]
    values = [
        best_fmcw_cfar_f1 if best_fmcw_cfar_f1>=0 else 0.0,
        best_fmcw_dl_f1   if best_fmcw_dl_f1  >=0 else 0.0,
        best_otfs_cfar_f1 if best_otfs_cfar_f1>=0 else 0.0,
        best_otfs_dl_f1   if best_otfs_dl_f1  >=0 else 0.0,
    ]
    plt.figure(figsize=(7,5))
    plt.bar(labels, values)
    plt.ylim(0, 1.0)
    plt.ylabel("F1 (best over sweep)")
    plt.title("Validation F1 — FMCW/OTFS, CFAR vs DL")
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout(); plt.savefig(out_dir/"val_f1_bars_dual.pdf"); plt.close()

    # --------- Figure 2: PR curves ----------
    def _curve_points(d):
        xs, ys = [], []
        for k, S in d.items():
            p, r = _precision_recall(S["tp"], S["fp"], S["fn"])
            xs.append(r); ys.append(p)
        # sort by recall
        order = np.argsort(xs)
        return np.array(xs)[order], np.array(ys)[order]

    plt.figure(figsize=(7,5))
    r,p = _curve_points(acc["FMCW-CFAR"]); plt.plot(r, p, 'o-', label="FMCW CFAR")
    r,p = _curve_points(acc["FMCW-DL"]);   plt.plot(r, p, 'o-', label="FMCW DL")
    r,p = _curve_points(acc["OTFS-CFAR"]); plt.plot(r, p, 's--', label="OTFS CFAR")
    r,p = _curve_points(acc["OTFS-DL"]);   plt.plot(r, p, 's--', label="OTFS DL")
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall (threshold sweeps)")
    plt.grid(True, linestyle=':', alpha=0.6); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir/"val_precision_recall_dual.pdf"); plt.close()

    # --------- Figure 3: Error CDFs (|Δr|, |Δv|) using best operating points ----------
    def _cdf(vals):
        if len(vals)==0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0])
        vals = np.sort(np.array(vals, float))
        y = np.linspace(0, 1, len(vals), endpoint=True)
        return vals, y

    plt.figure(figsize=(12,5))
    # |Δr|
    plt.subplot(1,2,1)
    for name, key in [("FMCW CFAR", best_fmcw_cfar_k), ("FMCW DL", best_fmcw_dl_k),
                      ("OTFS CFAR", best_otfs_cfar_k), ("OTFS DL", best_otfs_dl_k)]:
        if key is None: continue
        if "FMCW" in name and "CFAR" in name: S = acc["FMCW-CFAR"][key]
        elif "FMCW" in name:                 S = acc["FMCW-DL"][key]
        elif "OTFS" in name and "CFAR" in name: S = acc["OTFS-CFAR"][key]
        else:                                   S = acc["OTFS-DL"][key]
        x,y = _cdf(S["er_r"]); plt.plot(x, y, label=name)
    plt.xlabel("|ΔRange| (m)"); plt.ylabel("CDF"); plt.title("Error CDF — Range")
    plt.grid(True, linestyle=':', alpha=0.6); plt.legend()

    # |Δv|
    plt.subplot(1,2,2)
    for name, key in [("FMCW CFAR", best_fmcw_cfar_k), ("FMCW DL", best_fmcw_dl_k),
                      ("OTFS CFAR", best_otfs_cfar_k), ("OTFS DL", best_otfs_dl_k)]:
        if key is None: continue
        if "FMCW" in name and "CFAR" in name: S = acc["FMCW-CFAR"][key]
        elif "FMCW" in name:                 S = acc["FMCW-DL"][key]
        elif "OTFS" in name and "CFAR" in name: S = acc["OTFS-CFAR"][key]
        else:                                   S = acc["OTFS-DL"][key]
        x,y = _cdf(S["er_v"]); plt.plot(x, y, label=name)
    plt.xlabel("|ΔVelocity| (m/s)"); plt.ylabel("CDF"); plt.title("Error CDF — Velocity")
    plt.grid(True, linestyle=':', alpha=0.6); plt.legend()

    plt.tight_layout(); plt.savefig(out_dir/"val_error_cdfs_dual.pdf"); plt.close()


def _load_unet_weights(path, device):
    net = UNetLite().to(device)
    state = torch.load(path, map_location=device)
    # state can be weights-only or {"model": ...}
    if isinstance(state, dict) and "model" in state:
        net.load_state_dict(state["model"], strict=False)
    else:
        net.load_state_dict(state, strict=False)
    net.eval()
    return net

def run_dual_validation_from_root(
    root,
    dl_thr_sweep=(0.25,0.30,0.35,0.40,0.45,0.50),
    cfar_pfa_sweep=(1e-2,1e-3,1e-4,1e-5),
    max_samples=None,
    # NEW: let caller point to exact ckpts (or use defaults)
    fmcw_ckpt="auto",
    otfs_ckpt="auto",
    enforce_otfs=True,     # if True, error out when OTFS ckpt missing
):
    root = Path(root)
    ckpt_dir = root / "checkpoints"
    sp = SystemParams()

    # ---- resolve FMCW ckpt ----
    if fmcw_ckpt == "auto":
        # prefer best-only, fall back to last checkpoint
        cand = [
            ckpt_dir/"radar_unet_fmcw_best_only.pt",
            ckpt_dir/"radar_unet_best_only.pt",     # legacy name
        ]
        fmcw_path = next((p for p in cand if p.exists()), ckpt_dir/"radar_unet.pt")
    else:
        fmcw_path = Path(fmcw_ckpt)

    if not fmcw_path.exists():
        raise FileNotFoundError(f"FMCW checkpoint not found: {fmcw_path}")

    net_fmcw = _load_unet_weights(fmcw_path, DEVICE)
    print(f"[LOAD] FMCW DL model: {fmcw_path}")

    # ---- resolve OTFS ckpt (use the *new* OTFS model) ----
    if otfs_ckpt == "auto":
        otfs_path = ckpt_dir/"radar_unet_otfs_best_only.pt"
    else:
        otfs_path = Path(otfs_ckpt)

    if otfs_path.exists():
        net_otfs = _load_unet_weights(otfs_path, DEVICE)
        print(f"[LOAD] OTFS  DL model: {otfs_path}")
    else:
        msg = f"[LOAD] OTFS DL checkpoint not found: {otfs_path}"
        if enforce_otfs:
            raise FileNotFoundError(msg + " (set enforce_otfs=False to reuse FMCW model)")
        print(msg + " — reusing FMCW model for OTFS.")
        net_otfs = net_fmcw

    # ---- run dual evaluation with the two distinct nets ----
    eval_radar_on_val_set_dual(
        data_root=root,
        sp=sp,
        net_fmcw=net_fmcw,
        net_otfs=net_otfs,
        out_dir=root/"val_eval_dual2",
        dl_thr_sweep=dl_thr_sweep,
        cfar_pfa_sweep=cfar_pfa_sweep,
        max_samples=max_samples,
    )


# ---------- Modality-aware disk dataset for radar ----------
class RadarDiskDatasetModal(torch.utils.data.Dataset):
    """
    modality: "fmcw" or "otfs"
    If the .npz lacks OTFS tensors, set generate_otfs_on_the_fly=True to rebuild OTFS from gts (slower).
    """
    def __init__(self, folder, sp: SystemParams, modality="fmcw",
                 normalize=True, generate_otfs_on_the_fly=False):
        self.files = sorted(Path(folder).glob("*.npz"))
        self.sp = sp
        self.modality = modality
        self.normalize = normalize
        self.gen_otfs = generate_otfs_on_the_fly

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        z = np.load(self.files[idx], allow_pickle=True)
        gts = json.loads(str(z["gts"]))

        if self.modality == "fmcw":
            rd = z["rd_f_db"].astype(np.float32)
            hm = (z["heatmap_f"] if "heatmap_f" in z else z["heatmap"]).astype(np.float32)
        else:
            if "rd_o_db" in z:
                rd = z["rd_o_db"].astype(np.float32)
            else:
                # fallback: rebuild OTFS DD from gts (slower, but works)
                pts, its, vels = raycast_torch(self.sp, gts)
                if DEVICE.type == "cuda": torch.cuda.synchronize()
                (rd,) = otfs_torch(pts, its, vels, self.sp)
                rd = rd.astype(np.float32)
            if "heatmap_o" in z:
                hm = z["heatmap_o"].astype(np.float32)
            else:
                ra_o, va_o = self.sp.otfs_axes()
                hm = _heatmap_from_gts(rd.shape, ra_o, va_o, gts, self.sp).astype(np.float32)

        if self.normalize:
            rd = _rd_normalize(rd)
        x = torch.from_numpy(rd)[None, ...]  # (1,H,W)
        y = torch.from_numpy(hm)[None, ...]
        return x, y

# ---------- Modality-aware loaders ----------
def make_radar_loaders_modal(root, sp: SystemParams, batch=6, workers=0, modality="fmcw",
                             generate_otfs_on_the_fly=False):
    root = Path(root)
    tr_dir = root / "radar" / "train"
    va_dir = root / "radar" / "val"
    tr = RadarDiskDatasetModal(tr_dir, sp, modality=modality,
                               generate_otfs_on_the_fly=generate_otfs_on_the_fly)
    va = RadarDiskDatasetModal(va_dir, sp, modality=modality,
                               generate_otfs_on_the_fly=generate_otfs_on_the_fly)

    if len(tr) == 0:
        raise FileNotFoundError(f"No samples in {tr_dir} for modality={modality}")
    if len(va) == 0:
        raise FileNotFoundError(f"No samples in {va_dir} for modality={modality}")

    dl_tr = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True,  num_workers=workers)
    dl_va = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False, num_workers=workers)
    return dl_tr, dl_va

def finetune_otfs_from_fmcw(root, lr=5e-4, epochs=4, freeze_stem=True):
    root = Path(root)
    # 1) build params
    sp = SystemParams()

    ckpt = Path(root)/"checkpoints"
    # load FMCW
    net = UNetLite().to(DEVICE)
    state = torch.load(ckpt/"radar_unet_best_only.pt", map_location=DEVICE)
    net.load_state_dict(state, strict=False)
    if freeze_stem:
        for n,p in net.named_parameters():
            if n.startswith(("enc1","enc2")):   # adjust to your UNet module names
                p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p:p.requires_grad, net.parameters()), lr=lr)
    # 2) modality-aware loaders (set generate_otfs_on_the_fly=True if your NPZs lack rd_o_db)
    dl_tr, dl_va = make_radar_loaders_modal(
        root, sp, batch=6, modality="otfs", generate_otfs_on_the_fly=True
    )

    best = 1e9
    for ep in range(1, epochs+1):
        net.train(); tr=0
        for x,y in tqdm(dl_tr, desc=f"FT-OTFS ep{ep}/{epochs}"):
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = focal_bce_with_logits(net(x), y, alpha=0.25, gamma=2.0)
            loss.backward(); opt.step()
            tr += loss.item()*x.size(0)
        tr/=len(dl_tr.dataset)

        net.eval(); va=0
        with torch.no_grad():
            for x,y in dl_va:
                x,y = x.to(DEVICE), y.to(DEVICE)
                va += focal_bce_with_logits(net(x), y).item()*x.size(0)
        va/=len(dl_va.dataset)
        print(f"[FT-OTFS] ep{ep}: train {tr:.4f}  val {va:.4f}")

        # save
        torch.save({"epoch":ep,"model":net.state_dict()}, ckpt/"radar_unet_otfs.pt")
        if va < best:
            best = va
            torch.save(net.state_dict(), ckpt/"radar_unet_otfs_best_only.pt")
    return net

import torch, torch.nn as nn, torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c//r, 1)
        self.fc2 = nn.Conv2d(c//r, c, 1)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x,1)
        s = F.relu(self.fc1(s)); s = torch.sigmoid(self.fc2(s))
        return x * s

class ASPP(nn.Module):
    def __init__(self, c, rates=(1,6,12,18)):
        super().__init__()
        self.br = nn.ModuleList([nn.Conv2d(c, c//4, 3, padding=r, dilation=r) for r in rates])
        self.proj = nn.Conv2d(c, c, 1)
    def forward(self, x):
        xs = [F.relu(b(x)) for b in self.br]
        return F.relu(self.proj(torch.cat(xs,1)))

def norm_layer(c, use_group=True):
    return nn.GroupNorm(num_groups=8, num_channels=c) if use_group else nn.BatchNorm2d(c)

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, use_group=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p)
        self.norm = norm_layer(c_out, use_group)
        self.se   = SEBlock(c_out)
    def forward(self, x):
        return self.se(F.relu(self.norm(self.conv(x))))

class Calib(nn.Module):
    def __init__(self): super().__init__(); self.a = nn.Parameter(torch.tensor(1.0)); self.b = nn.Parameter(torch.tensor(0.0))
    def forward(self, x): return self.a*x + self.b

class RadarCommNet(nn.Module):
    def __init__(self, in_ch_radar=1, in_ch_comm=2, base=48, use_group_norm=True):
        super().__init__()
        # Shared encoder
        self.enc1 = ConvBNReLU(in_ch_radar, base, use_group=use_group_norm)
        self.enc2 = ConvBNReLU(base, base*2, s=2, use_group=use_group_norm)
        self.enc3 = ConvBNReLU(base*2, base*4, s=2, use_group=use_group_norm)
        self.aspp = ASPP(base*4)

        # Decoder (shared)
        self.dec3 = ConvBNReLU(base*4, base*2, use_group=use_group_norm)
        self.dec2 = ConvBNReLU(base*2, base,   use_group=use_group_norm)
        self.out_fmcw = nn.Conv2d(base, 1, 1)
        self.out_otfs = nn.Conv2d(base, 1, 1)

        # Per-domain calibrations (before sigmoid)
        self.calib_fmcw = Calib()
        self.calib_otfs = Calib()

        # Communication demappers (small CNNs)
        # old (global pooling -> (B,2))
        # self.dem_ofdm = nn.Sequential(
        #     nn.Conv2d(in_ch_comm, 32, 3, padding=1), nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        #     nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        #     nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 2)
        # )

        # new (per-cell logits -> (B,2,H,W))
        self.dem_ofdm = nn.Sequential(
            nn.Conv2d(in_ch_comm, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 2, 1)  # no pooling, keeps H,W
        )
        self.dem_otfs = nn.Sequential(
            nn.Conv2d(in_ch_comm, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 2, 1)
        )

    def forward_radar(self, x, domain="fmcw"):
        # U-Net like path
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
        z  = self.aspp(e3)
        d3 = F.interpolate(self.dec3(z), size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = F.interpolate(self.dec2(d3), size=e1.shape[-2:], mode='bilinear', align_corners=False)
        logits = self.out_fmcw(d2) if domain=="fmcw" else self.out_otfs(d2)
        logits = (self.calib_fmcw if domain=="fmcw" else self.calib_otfs)(logits)
        return logits  # (B,1,H,W)

    def forward_comm(self, grid, domain="ofdm"):
        return self.dem_ofdm(grid) if domain=="ofdm" else self.dem_otfs(grid)

def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs*targets).sum(dim=(2,3)) + eps
    den = (probs.pow(2)+targets.pow(2)).sum(dim=(2,3)) + eps
    return 1 - (num/den).mean()

def radar_loss(logits, targets, alpha=0.25, gamma=2.0, dice_w=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
    pt  = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
    fl  = (alpha*(1-targets)*(pt**gamma)*(-torch.log(1-pt)) + (1-alpha)*targets*((1-pt)**gamma)*(-torch.log(pt))).mean()
    dl  = dice_loss_with_logits(logits, targets)
    return fl + bce + dice_w*dl

def comm_loss(logits, bits):
    # both (B,2,H,W)
    return F.binary_cross_entropy_with_logits(logits, bits.float())

def calib_reg(model, w=1e-4):
    reg = (model.calib_fmcw.a-1).pow(2)+(model.calib_fmcw.b).pow(2)+(model.calib_otfs.a-1).pow(2)+(model.calib_otfs.b).pow(2)
    return w*reg

from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---- Communication configs: keep GEN and BER separate ----
# For TRAINING BATCH GENERATOR: uses n_sym (number of OFDM symbols per batch)
ofdm_cfg_gen = dict(Nfft=256, cp_len=32, n_sym=8)
otfs_cfg_gen = dict(M=64, N=256, cp_len=32)

# For BER SWEEP SIMULATOR: uses n_ofdm_sym (total symbols per Eb/N0 point)
ofdm_cfg_ber = dict(Nfft=256, cp_len=32, n_ofdm_sym=600)
otfs_cfg_ber = dict(M=64, N=256, cp_len=32)

def train_multidomain_multitask(
    data_root, sp, epochs=12, batch_radar=6, batch_comm_ofdm=8, batch_comm_otfs=6,
    lr=3e-4, out_root="./output/isac_pro", resume=True, rng_seed=2025
):
    root = Path(out_root); (root/"checkpoints").mkdir(parents=True, exist_ok=True)
    # Datasets
    dl_fm_tr, dl_fm_va = make_radar_loaders_modal(data_root, sp, batch=batch_radar, modality="fmcw")
    dl_ot_tr, dl_ot_va = make_radar_loaders_modal(data_root, sp, batch=batch_radar, modality="otfs", generate_otfs_on_the_fly=False)

    # Comm specs (reuse your JSON or on-the-fly)
    #ofdm_cfg = dict(Nfft=256, cp_len=32, n_sym=8, n_ofdm_sym=600, batch=batch_comm_ofdm)
    #otfs_cfg = dict(M=64, N=256, cp_len=32, batch=batch_comm_otfs)
    


    # Model & opt
    net = RadarCommNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    start_ep = 1

    # Resume?
    ckpt = root/"checkpoints"/"mdmt.pt"
    if resume and ckpt.exists():
        s = torch.load(ckpt, map_location=DEVICE)
        net.load_state_dict(s["model"]); opt.load_state_dict(s["optim"]); start_ep = s["epoch"]+1
        print(f"[RESUME] epoch {s['epoch']}")

    # Epochs
    for ep in range(start_ep, epochs+1):
        net.train()
        pbar = tqdm(range(max(len(dl_fm_tr), len(dl_ot_tr), 200)), desc=f"Train ep{ep}/{epochs}")
        it_fm = iter(dl_fm_tr); it_ot = iter(dl_ot_tr)

        for _ in pbar:
            loss = 0.0

            # ---- Radar FMCW ----
            try:
                x_f, y_f = next(it_fm)
                x_f, y_f = x_f.to(DEVICE), y_f.to(DEVICE)
                logits_f = net.forward_radar(x_f, domain="fmcw")
                loss += radar_loss(logits_f, y_f)
            except StopIteration:
                pass

            # ---- Radar OTFS ----
            try:
                x_o, y_o = next(it_ot)
                x_o, y_o = x_o.to(DEVICE), y_o.to(DEVICE)
                logits_o = net.forward_radar(x_o, domain="otfs")
                loss += radar_loss(logits_o, y_o)
            except StopIteration:
                pass

            # ---- Comm OFDM ----
            #Xo, Yo = comm_dl_gen_batch_OFDM(ebn0_db=np.random.uniform(6,16), batch=batch_comm_ofdm, **ofdm_cfg)
            #ofdm_cfg itself already contains "batch": batch_comm_ofdm.
            # Xo, Yo = comm_dl_gen_batch_OFDM(
            #     ebn0_db=np.random.uniform(6,16),
            #     **{k:v for k,v in ofdm_cfg.items() if k != "batch"}
            # )
            # Xo, Yo = comm_dl_gen_batch_OFDM(
            #     ebn0_db=np.random.uniform(6,16),
            #     batch=batch_comm_ofdm,
            #     **cfg_for_ofdm_gen(ofdm_cfg)
            # )
            Xo, Yo = comm_dl_gen_batch_OFDM(
                ebn0_db=np.random.uniform(6,16),
                batch=batch_comm_ofdm,
                **ofdm_cfg_gen
            )
            Xo, Yo = Xo.to(DEVICE), Yo.to(DEVICE)
            logits_comm_ofdm = net.forward_comm(Xo, domain="ofdm")
            loss += 0.5*comm_loss(logits_comm_ofdm, Yo)

            # ---- Comm OTFS ----
            #Xt, Yt = comm_dl_gen_batch_OTFS(ebn0_db=np.random.uniform(6,16), batch=batch_comm_otfs, **otfs_cfg)
            # Xt, Yt = comm_dl_gen_batch_OTFS(
            #     ebn0_db=np.random.uniform(6,16),
            #     batch=batch_comm_otfs,
            #     **cfg_for_otfs_gen(otfs_cfg)
            # )
            Xt, Yt = comm_dl_gen_batch_OTFS(
                ebn0_db=np.random.uniform(6,16),
                batch=batch_comm_otfs,
                **otfs_cfg_gen
            )
            Xt, Yt = Xt.to(DEVICE), Yt.to(DEVICE)
            logits_comm_otfs = net.forward_comm(Xt, domain="otfs")
            loss += 0.5*comm_loss(logits_comm_otfs, Yt)

            loss = loss + calib_reg(net, w=1e-4)

            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss))

        # ---- Save checkpoint each epoch
        torch.save({"epoch":ep, "model":net.state_dict(), "optim":opt.state_dict()}, ckpt)

        # ---- Evaluate & visualize
        # eval_and_visualize_epoch(
        #     net, data_root, sp, root/f"epochs/ep_{ep:02d}",
        #     ofdm_cfg, otfs_cfg, ebn0_list=[0,6,10,14,18]
        # )
        eval_and_visualize_epoch(
            net,
            data_root,
            sp,
            root / f"epochs/ep_{ep:02d}",
            ofdm_cfg_gen,          # ← was ofdm_cfg
            otfs_cfg_gen,          # ← was otfs_cfg
            ebn0_list=[0, 6, 10, 14, 18]
        )

class DemapHead(nn.Module):
    """Wrap RadarCommNet comm head so it looks like a normal model."""
    def __init__(self, net, domain: str):
        super().__init__()
        assert domain in ("ofdm", "otfs")
        self.net = net
        self.domain = domain
    def forward(self, x):
        # x: (B, 2, H, W) real/imag grid
        return self.net.forward_comm(x, domain=self.domain)
    
@torch.no_grad()
def eval_and_visualize_epoch(net, data_root, sp, out_dir, ofdm_cfg, otfs_cfg, ebn0_list):
    out = Path(out_dir); 
    out.mkdir(parents=True, exist_ok=True)
    # ---- Quantitative RADAR on val (DL vs CFAR, FMCW & OTFS)
    # eval_radar_on_val_set_dual(
    #     data_root=data_root, sp=sp,
    #     net_fmcw=net, net_otfs=net,    # same net, different heads
    #     out_dir=out,
    #     dl_thr_sweep=(0.25,0.35,0.45),  # fast sweep
    #     cfar_pfa_sweep=(1e-3,1e-4),
    #     max_samples=150                 # speed
    # )
    # Use your relaxed 5% cube matcher if you implemented it:
# match_fn = lambda dets, gtinfo: _match_dets_to_gts_bev_relaxed(dets, gtinfo, scale=0.05)

    eval_radar_on_val_set_dual(
        data_root="./output/isac_big",
        sp=sp,
        net_fmcw=net,   # same model, FMCW head is used via domain="fmcw"
        net_otfs=net,   # same model, OTFS head is used via domain="otfs"
        out_dir=Path("./output/isac_big/epochs/ep_XX"),  # your epoch folder
        dl_thr_sweep_fmcw=(0.25,0.30,0.35,0.40,0.45),
        dl_thr_sweep_otfs=(0.05,0.10,0.15,0.20,0.30),
        cfar_pfa_sweep=(1e-3,1e-4),
        match_fn=None,       # or your relaxed matcher
        max_samples=200
    )

    # ---- Communication BER curves (quick points)
    eb_axis = np.array(ebn0_list, float)
    
    ofdm_head = DemapHead(net, "ofdm").eval().to(DEVICE)
    otfs_head = DemapHead(net, "otfs").eval().to(DEVICE)

    #ber_ofdm = comm_demap_ber_curve(ofdm_head, comm_dl_gen_batch_OFDM, ofdm_cfg, eb_axis, device=DEVICE)
    #ber_otfs = comm_demap_ber_curve(otfs_head, comm_dl_gen_batch_OTFS, otfs_cfg, eb_axis, device=DEVICE)
    ber_ofdm = comm_demap_ber_curve(ofdm_head, comm_dl_gen_batch_OFDM, ofdm_cfg_gen, eb_axis, device=DEVICE)
    ber_otfs = comm_demap_ber_curve(otfs_head, comm_dl_gen_batch_OTFS, otfs_cfg_gen, eb_axis, device=DEVICE)
    # ber_ofdm = comm_demap_ber_curve(lambda eb: net.forward_comm(*comm_dl_gen_batch_OFDM(eb, **ofdm_cfg)[0:1], domain="ofdm"),
    #                                 comm_dl_gen_batch_OFDM, ofdm_cfg, eb_axis)
    # ber_otfs = comm_demap_ber_curve(lambda eb: net.forward_comm(*comm_dl_gen_batch_OTFS(eb, **otfs_cfg)[0:1], domain="otfs"),
    #                                 comm_dl_gen_batch_OTFS, otfs_cfg, eb_axis)
    
    # Plot (and compare to your baselines/theory)
    eb_axis2, base_ofdm, base_otfs, theory = run_ber_sweep_and_plot(out/"ber_compare.png",
                                                                     ebn0_db_list=eb_axis,
                                                                     ofdm_cfg=dict(**ofdm_cfg, n_ofdm_sym=400),
                                                                     otfs_cfg=dict(M=otfs_cfg["M"], N=otfs_cfg["N"], cp_len=otfs_cfg["cp_len"]),
                                                                     rng_seed=2025)
    plt.figure(figsize=(7,5))
    plt.semilogy(eb_axis, ber_ofdm+1e-12, 'o-', label='OFDM DL demapper')
    plt.semilogy(eb_axis, ber_otfs+1e-12, 's-', label='OTFS DL demapper')
    plt.semilogy(eb_axis2, base_ofdm+1e-12, 'o--', label='OFDM baseline')
    plt.semilogy(eb_axis2, base_otfs+1e-12, 's--', label='OTFS baseline')
    plt.semilogy(eb_axis2, theory+1e-12, 'k:', label='QPSK theory')
    plt.grid(True, which='both', linestyle=':'); plt.xlabel('Eb/N0 (dB)'); plt.ylabel('BER')
    plt.title('Comm BER per epoch'); plt.legend(); plt.tight_layout()
    plt.savefig(out/"ber_compare_with_dl.png", dpi=170); plt.close()

    # ---- Qualitative: overlay a few samples
    val_montage_examples(data_root, sp, net, out, k=3, seed=2027)


def launch_mdmt_training():
    # ---- paths & params ----
    root = "./output/isac_mdmt"   # change if you like
    ckpt_dir = Path(root)/"checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sp = SystemParams()

    # ---- ensure dataset ready (radar + comm specs) ----
    simulate_if_missing(
        out_dir=root,
        sp=sp,
        n_train=1500,
        n_val=300,
        seed=2025,
        # If your simulator supports it, keep OTFS maps too for OTFS DL:
        save_otfs=True,
        snr_list=(0,2,4,6,8,10,12,14,16,18,20)
    )

    # ---- training hyper-params ----
    epochs = 12
    lr = 3e-4
    batch_radar = 6
    batch_comm_ofdm = 8
    batch_comm_otfs = 6

    # ---- kick off training (per-epoch eval & figs saved automatically) ----
    train_multidomain_multitask(
        data_root=root,
        sp=sp,
        epochs=epochs,
        batch_radar=batch_radar,
        batch_comm_ofdm=batch_comm_ofdm,
        batch_comm_otfs=batch_comm_otfs,
        lr=lr,
        out_root=root,
        resume=True
    )

    # (Optional) final dual-waveform validation using the trained model(s)
    # run_dual_validation_from_root(root, max_samples=None)

from pathlib import Path
from dataclasses import dataclass
import json, os, math
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# ----------------------- configurable distributions -----------------------
@dataclass
class SceneDist:
    max_targets: int = 3
    r_min: float = 6.0
    r_max: float = 80.0
    az_deg: float = 60.0
    vx_min: float = -25.0
    vx_max: float =  25.0
    vy_min: float =  -6.0
    vy_max: float =   6.0
    cube_min: tuple = (2.0, 1.5, 1.5)  # (x,y,z) min size
    cube_max: tuple = (5.0, 3.0, 3.0)  # (x,y,z) max size

@dataclass
class ClutterDist:
    ground_return_db: float = -25.0    # higher = stronger ground
    speckle_db: float = -35.0          # white-like floor
    ghost_prob: float = 0.15           # “mirror” ghost across Doppler
    rd_jitter_bins: int = 2            # random +/- bin jitter
    drop_stripe_prob: float = 0.1      # vertical/horizontal stripes

# --------------------------- helper: rand scene ---------------------------
def _rand_scene(sp, rng, sd: SceneDist):
    K = rng.integers(1, sd.max_targets+1)
    gts = []
    for _ in range(K):
        r  = float(rng.uniform(sd.r_min, sd.r_max))
        az = float(rng.uniform(-np.deg2rad(sd.az_deg/2), np.deg2rad(sd.az_deg/2)))
        x, y = r*np.cos(az), r*np.sin(az)
        vx   = float(rng.uniform(sd.vx_min, sd.vx_max))
        vy   = float(rng.uniform(sd.vy_min, sd.vy_max))
        sx = float(rng.uniform(*sd.cube_min[:1]+sd.cube_max[:1])) if isinstance(sd.cube_min, tuple) else 4.0
        sy = float(rng.uniform(*sd.cube_min[1:2]+sd.cube_max[1:2])) if isinstance(sd.cube_min, tuple) else 2.0
        sz = float(rng.uniform(*sd.cube_min[2:3]+sd.cube_max[2:3])) if isinstance(sd.cube_min, tuple) else 2.0
        gts.append({'c':[x, y, 1.0], 's':[sx, sy, sz], 'v':[vx, vy, 0.0]})
    return gts

def _apply_artifacts(rd_db, rng, cd: ClutterDist):
    rd = rd_db.copy()
    H, W = rd.shape
    # Add a floor + speckle
    rd += rng.normal(0, 1.0, rd.shape) * 0.0  # amplitude already in dB
    rd = np.maximum(rd, np.max(rd) + cd.speckle_db)
    # Simple “ghosts” (roll in Doppler)
    if rng.random() < cd.ghost_prob:
        shift = rng.integers(-5, 6)
        rd = np.maximum(rd, np.roll(rd, shift, axis=0) + cd.ground_return_db)
    # Stripe dropout/jitter
    if rng.random() < cd.drop_stripe_prob:
        if rng.random() < 0.5:  # vertical
            col = rng.integers(0, W); rd[:, col] -= 20
        else:
            row = rng.integers(0, H); rd[row, :] -= 20
    return rd

# --------------------------- one sample synth ----------------------------
def _synth_one(idx, split, out_dir, sp, ebn0_db, seed, sd: SceneDist, cd: ClutterDist, save_otfs=True):
    rng = np.random.default_rng(int(seed) + idx)
    gts = _rand_scene(sp, rng, sd)

    pts, its, vels = raycast_torch(sp, gts)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    (rd_f_db,) = fmcw_torch(pts, its, vels, sp)
    ra_f, va_f = sp.fmcw_axes()
    heat_f = _heatmap_from_gts(rd_f_db.shape, ra_f, va_f, gts, sp)

    if save_otfs:
        (rd_o_db,) = otfs_torch(pts, its, vels, sp)
        ra_o, va_o = sp.otfs_axes()
        heat_o = _heatmap_from_gts(rd_o_db.shape, ra_o, va_o, gts, sp)
    else:
        rd_o_db = np.zeros((sp.M, sp.N), np.float32); heat_o = np.zeros_like(rd_o_db)

    # Artifacts for diversity
    rd_f_db = _apply_artifacts(rd_f_db, rng, cd)
    if save_otfs:
        rd_o_db = _apply_artifacts(rd_o_db, rng, cd)

    # Save
    save_path = Path(out_dir)/"radar"/split/f"{idx:07d}.npz"
    np.savez_compressed(
        save_path,
        rd_f_db=rd_f_db.astype(np.float32),
        heatmap_f=heat_f.astype(np.float32),
        rd_o_db=rd_o_db.astype(np.float32),
        heatmap_o=heat_o.astype(np.float32),
        gts=json.dumps(gts),
        ebn0_db=float(ebn0_db)
    )

# --------------------------- comm spec writer ---------------------------
def _write_comm_specs(out_dir, n_train, n_val, eb_grid=(0,2,4,6,8,10,12,14,16,18,20), seed=2025):
    rng = np.random.default_rng(seed)
    def _make(n):
        specs=[]
        for _ in range(n):
            specs.append(dict(
                ebn0_db=float(rng.choice(eb_grid)),
                seed=int(rng.integers(0, 1<<31))
            ))
        return specs
    comm_dir = Path(out_dir)/"comm"
    comm_dir.mkdir(parents=True, exist_ok=True)
    with open(comm_dir/"train_spec.json","w") as f: json.dump(_make(n_train*2), f) # *2 to cover both mods
    with open(comm_dir/"val_spec.json","w") as f:   json.dump(_make(n_val), f)

import multiprocessing as mp
# ----------------------------- SHARDED build ----------------------------
def build_big_dataset(
    out_dir: str,
    sp: SystemParams,
    n_train=10000,
    n_val=2000,
    shards=8,
    seed=2025,
    save_otfs=True,
    scene_dist: SceneDist = SceneDist(max_targets=3, r_min=6, r_max=90),
    clutter_dist: ClutterDist = ClutterDist(),
    snr_buckets=(0,2,4,6,8,10,12,14,16,18,20),
    workers=None,
    overwrite=False
):
    """
    Creates a large dataset:
      <out_dir>/radar/{train,val}/*.npz  (rd_f_db, heatmap_f, rd_o_db, heatmap_o, gts, ebn0_db)
      <out_dir>/comm/train_spec.json, val_spec.json
    """
    out = Path(out_dir)
    (out/"radar"/"train").mkdir(parents=True, exist_ok=True)
    (out/"radar"/"val").mkdir(parents=True, exist_ok=True)

    if not overwrite:
        train_exist = any((out/"radar"/"train").glob("*.npz"))
        val_exist   = any((out/"radar"/"val").glob("*.npz"))
        if train_exist and val_exist:
            print(f"[DATA] Exists at {out}. Use overwrite=True to rebuild.")
            _write_comm_specs(out, n_train, n_val, seed=seed)
            return

        # Decide parallelism early — but don't use mp on CUDA
    use_cuda = (DEVICE.type == "cuda")
    if workers is None:
        workers = 1 if use_cuda else max(1, mp.cpu_count() // 2)


    for split in ("train","val"):
        N = dict(train=n_train, val=n_val)[split]
        ebs = np.array(list(snr_buckets), dtype=float)
        eb_seq = np.resize(ebs, N)
        tasks = [
            (split, i, float(eb_seq[i]), out, sp, seed, scene_dist, clutter_dist, save_otfs)
            for i in range(N)
        ]

        print(f"[DATA] Generating {split}: {N} samples → {out}")

        if use_cuda or workers == 1:
            # Single-process path (safe with GPU)
            for t in tqdm(tasks, desc=f"{split}-radar"):
                _build_dataset_runner(t)
        else:
            # CPU-parallel path only (no CUDA ops inside worker!)
            with mp.Pool(workers) as pool:
                for _ in tqdm(pool.imap_unordered(_build_dataset_runner, tasks, chunksize=32),
                              total=N, desc=f"{split}-radar"):
                    pass

    _write_comm_specs(out, n_train, n_val, seed=seed)
    print("[DATA] Done.")

# TOP-LEVEL worker so it’s picklable
def _build_dataset_runner(args):
    # args: (split, i, eb, out_dir, sp, seed, scene_dist, clutter_dist, save_otfs)
    split, i, eb, out_dir, sp, seed, scene_dist, clutter_dist, save_otfs = args
    _synth_one(i, split, out_dir, sp, eb, seed, scene_dist, clutter_dist, save_otfs)

# ================= Run =================
if __name__ == '__main__':
    # #new_training()
    # root = "./output/isac_c4"
    # #finetune_otfs_from_fmcw(root)
    # run_validation_from_root(
    #     root,
    #     dl_thr_sweep=(0.25,0.30,0.35,0.40,0.45,0.50),
    #     cfar_pfa_sweep=(1e-2,1e-3,1e-4,1e-5),
    #     do_otfs=True,
    #     max_samples=None  # set e.g. 200 for a quicker pass
    # )
    # #run_dual_validation_from_root("./output/isac_c4", max_samples=None)
    # run_dual_validation_from_root(
    #     "./output/isac_c4",
    #     max_samples=None,
    #     enforce_otfs=True  # require the OTFS model; raises if missing
    # )
    sp = SystemParams()

    ROOT = "./output/isac_big"   # keep this consistent

    # 1) Build (or rebuild) a big dataset once
    build_big_dataset(
        out_dir=ROOT,
        sp=sp,
        n_train=20000,
        n_val=4000,
        save_otfs=True,          # keep OTFS maps for OTFS DL
        overwrite=False          # set True only when you really want to regenerate
    )

    # 2) Train the multi-domain / multi-task model on that dataset
    #    (launch_mdmt_training will skip simulation when it detects data)
    def launch_mdmt_training2(root=ROOT):
        ckpt_dir = Path(root)/"checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # OPTIONAL: if you keep the simulate_if_missing call inside, it will no-op since data exists
        # simulate_if_missing(out_dir=root, sp=sp, ...)

        train_multidomain_multitask(
            data_root=root,       # <--- MUST match the dataset root above
            sp=sp,
            epochs=12,
            batch_radar=6,
            batch_comm_ofdm=8,
            batch_comm_otfs=6,
            lr=3e-4,
            out_root=root,        # saves ckpts and epoch visuals under this root
            resume=True
        )

    # kick it off
    launch_mdmt_training2(ROOT)