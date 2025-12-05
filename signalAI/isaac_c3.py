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

# ================= Utils =================
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
def viz_rd_2d_compare(path, rd_f_db, rd_o_db, gts, sp: SystemParams, cfar_cfg=None):
    fig, ax = plt.subplots(1, 2, figsize=(16,6))
    pos = np.array([0,0,sp.H])
    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    if cfar_cfg is None:
        cfar_cfg = dict(train=(10, 8), guard=(2, 2), pfa=1e-4,
                        min_snr_db=8.0, notch_doppler_bins=2,
                        apply_nms=True, max_peaks=60)

    im = plot_rd(ax[0], rd_f_db, ra_f, va_f, "FMCW Range–Doppler", dynamic_db=35, percentile_clip=99.2, cmap='magma')
    plt.colorbar(im, ax=ax[0], label='dB')

    det_f, noise_f, snr_f = cfar2d_ca(rd_f_db, **cfar_cfg, return_stats=True)
    fy, fx = np.where(det_f)
    if fy.size:
        ax[0].scatter(ra_f[fx], va_f[fy], s=60, facecolors='none', edgecolors='cyan', linewidths=1.8, label='CFAR')

    im2 = plot_rd(ax[1], rd_o_db, ra_o, va_o, "OTFS Delay–Doppler", dynamic_db=35, percentile_clip=99.2, cmap='magma')
    plt.colorbar(im2, ax=ax[1], label='dB')

    for i, (ra, va, rd) in enumerate([(ra_f, va_f, rd_f_db), (ra_o, va_o, rd_o_db)]):
        for gt in gts:
            P = np.array(gt['c']) - pos
            r = np.linalg.norm(P)
            v = np.dot(P/r, gt['v'])
            if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
                ax[i].plot(r, v, 'wx', ms=10, mew=2, label='GT' if i==0 else "")
                ax[i].text(r+1, v+0.3, f"{r:.0f} m, {v:.1f} m/s", color='white', fontsize=9, weight='bold')

    for i in range(2):
        ax[i].grid(alpha=0.25, linestyle=':')
        ax[i].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()

    return (det_f, ra_f, va_f, noise_f, snr_f)

def viz_rd_3d_compare(path, rd_f_db, rd_o_db, gts, sp: SystemParams):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(18,8))
    pos = np.array([0,0,sp.H])
    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    for i, (rd, ra, va, name) in enumerate([(rd_f_db, ra_f, va_f, "FMCW"), (rd_o_db, ra_o, va_o, "OTFS")]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        R, V = np.meshgrid(ra, va)
        floor = np.percentile(rd, 99.5) - 40
        surf = np.maximum(rd, floor)
        ax.plot_surface(R, V, surf, cmap='viridis', rstride=2, cstride=2, alpha=0.85)
        for gt in gts:
            P = np.array(gt['c']) - pos; r = np.linalg.norm(P); v = np.dot(P/r, gt['v'])
            if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
                ax.scatter([r],[v],[np.max(rd)+5], c='r', marker='x', s=120, linewidths=2, zorder=10)
        ax.set_title(f"{name} 3D")
        ax.set_xlabel("Range (m)"); ax.set_ylabel("Velocity (m/s)")
        ax.set_xlim(0, ra[-1]); ax.set_ylim(va[0], va[-1]); ax.view_init(45, -110)

    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches='tight'); plt.close()

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

def viz_rd_3d_with_dets(path, rd_db, ra, va, det_mask, gts, sp: SystemParams, title="FMCW RD with Detections & GT"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    pos = np.array([0,0,sp.H])
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    R, V = np.meshgrid(ra, va)
    floor = np.percentile(rd_db, 99.5) - 40
    surf = np.maximum(rd_db, floor)
    ax.plot_surface(R, V, surf, cmap='viridis', rstride=2, cstride=2, alpha=0.85)

    # GT markers
    for gt in gts:
        P = np.array(gt['c']) - pos; r = np.linalg.norm(P); v = np.dot(P/r, gt['v'])
        if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
            ax.scatter([r],[v],[np.max(rd_db)+5], c='r', marker='x', s=140, linewidths=2, label='GT')

    # Detections as cyan spheres
    yx = np.where(det_mask)
    if yx[0].size:
        zvals = rd_db[yx]
        ax.scatter(ra[yx[1]], va[yx[0]], zvals, c='c', s=30, depthshade=True, label='CFAR')

    ax.set_title(title)
    ax.set_xlabel("Range (m)"); ax.set_ylabel("Velocity (m/s)"); ax.set_zlabel("Power (dB)")
    ax.set_xlim(0, ra[-1]); ax.set_ylim(va[0], va[-1]); ax.view_init(35, -115)
    if ax.get_legend_handles_labels()[1]: 
        ax.legend(loc='upper left')
    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches='tight'); plt.close()

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

def _match_dets_to_gts(dets, gt_rv, w_r=1.0, w_v=0.5):
    used = set(); pairs = []; 
    for di, d in enumerate(dets):
        best_cost = 1e9; best_g = None
        for gi, g in enumerate(gt_rv):
            if gi in used: continue
            cost = w_r*abs(d['r'] - g['r']) + w_v*abs(d['v'] - g['v'])
            if cost < best_cost: best_cost = cost; best_g = gi
        if best_g is not None:
            used.add(best_g); pairs.append((d, gt_rv[best_g], best_cost))
    unpaired = [d for d in dets if all(d is not p[0] for p in pairs)]
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
    plt.tight_layout(); plt.savefig(f"{path_prefix}_bev_scene.png"); plt.close()

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
    plt.tight_layout(); plt.savefig(f"{path_prefix}_bev_compare.png", dpi=180); plt.close()

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

def _match_dets_to_gts(dets, gt_rv, w_r=1.0, w_v=0.5):
    used = []  # 允许多个检测匹配到同一 GT，用列表而不是 set（如果你想“一对一”，把它改回 set 即可）
    pairs = []
    for d in dets:
        best_g = None; best_cost = 1e9
        for gi, g in enumerate(gt_rv):
            cost = w_r*abs(d['r'] - g['r']) + w_v*abs(d['v'] - g['v'])
            if cost < best_cost:
                best_cost = cost; best_g = gi
        if best_g is not None:
            pairs.append((d, best_g, best_cost))
            used.append(best_g)
    return pairs

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

def _compute_metrics_from_pairs(pairs, gtinfo, sp):
    """
    pairs: list of (det, gi, cost), gi 为 gtinfo 的索引
    规则：检测点投影到匹配 GT 的 az；若 (x,y) 落在该 GT cube 的 XY 范围内 => TP，否则 FP。
    Recall：被至少一个 TP 覆盖的 GT / GT 总数。
    """
    TP, FP = 0, 0
    per_gt_tp = {i:0 for i in range(len(gtinfo))}
    tp_pts, fp_pts = [], []

    for det, gi, _ in pairs:
        g = gtinfo[gi]
        x, y = _project_det_xy_using_gt_az(det, g)
        # 限定 BEV 范围（0..sp.bev_r_max, -sp.bev_r_max/2..+sp.bev_r_max/2）
        if not (0 <= x <= sp.bev_r_max and -sp.bev_r_max/2 <= y <= sp.bev_r_max/2):
            # 画面之外也计为 FP（可按需改成忽略）
            FP += 1
            fp_pts.append((x,y,gi))
            continue
        if _inside_cube_xy(x, y, g):
            TP += 1
            per_gt_tp[gi] += 1
            tp_pts.append((x,y,gi))
        else:
            FP += 1
            fp_pts.append((x,y,gi))

    detected_gts = sum(1 for k,v in per_gt_tp.items() if v > 0)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = detected_gts / max(1, len(gtinfo))
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

    metrics = dict(TP=TP, FP=FP, detected_gts=detected_gts,
                   total_gts=len(gtinfo), precision=precision, recall=recall, f1=f1)
    return metrics, tp_pts, fp_pts

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
        ber_ofdm.append(ofdm_tx_rx_ber(eb, **ofdm_cfg, rng=rng))
        ber_otfs.append(otfs_tx_rx_ber(eb, **otfs_cfg,  rng=rng))
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

@torch.no_grad()
def comm_demap_ber_curve(model, gen_batch_fn, cfg, ebn0_db_list, device=None):
    device = device or DEVICE
    model = model.to(device)
    bers=[]
    for eb in ebn0_db_list:
        X,Y = gen_batch_fn(eb, **cfg)
        X = X.to(device)
        logits = model(X)
        bits_hat = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(np.uint8) # (B,2,H,W)
        bits_gt  = Y.numpy().astype(np.uint8)
        ber = np.mean(bits_hat != bits_gt)
        bers.append(ber)
    return np.array(bers)

# ================= Run =================
if __name__ == '__main__':
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
        f"{root}/compare_2d.png", rd_f_db, rd_o_db, gts, sp
    )
    # 3D 曲面对比（FMCW/OTFS）
    viz_rd_3d_compare(f"{root}/compare_3d.png", rd_f_db, rd_o_db, gts, sp)

    # ============== FMCW: 3D with dets ==============
    viz_rd_3d_with_dets(
        f"{root}/fmcw_3d_with_dets.png",
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
        f"{root}/otfs_3d_with_dets.png",
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
        f"{root}/scene_bev_compare.png",
        dets_f, dets_o, gts, sp
    )

        # ============== COMM: BER curves (OFDM for FMCW comm, OTFS for OTFS comm) ==============
    print("Running BER sweeps (communication)...")
    eb_axis, ber_ofdm, ber_otfs, ber_theory = run_ber_sweep_and_plot(
        f"{root}/ber_compare.png",
        ebn0_db_list=np.arange(0, 21, 2),
        ofdm_cfg=dict(Nfft=256, cp_len=32, n_ofdm_sym=600),
        otfs_cfg=dict(M=64, N=256, cp_len=32),
        rng_seed=2025
    )
    print("BER figure saved to ber_compare.png")

    print("Done.")

        # ===================== Radar DL vs CFAR =====================
    print("Training Radar DL detector...")
    radar_net = train_radar_model(sp, epochs=6, batch=6, lr=1e-3, n_train=1200, n_val=300)

    # Evaluate on a fresh scene (your current gts)
    ra_f, va_f = sp.fmcw_axes()
    det_f_mask, ra_f, va_f, noise_f, snr_f = viz_rd_2d_compare(f"{root}/compare_2d.png", rd_f_db, rd_o_db, gts, sp)
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
    ber_otfs_dl  = comm_demap_ber_curve(otfs_model, comm_dl_gen_batch_OTFS, otfs_cfg, eb_axis)

    # Classical baselines already available (we also computed theory earlier):
    _, ber_ofdm_base, ber_otfs_base, ber_theory = run_ber_sweep_and_plot(f"{root}/ber_compare.png",
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
    plt.legend(); plt.tight_layout(); plt.savefig(f"{root}/ber_compare_with_dl.png", dpi=170); plt.close()