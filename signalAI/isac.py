#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISAC Simulation (OTFS debug-heavy, alignment-aware, dual-token modes)
--------------------------------------------------------------------

Key improvements for OTFS:
• Two token modes:
  - 'deconv'       : impulse-pilot -> Hhat, 2-D Wiener deconvolution, optional auto-alignment
  - 'raw_plus_H'   : feed [Re/Im X_DD_obs, Re/Im Hhat] to the network (learn equalizer)
• Robust 2-D Wiener: gamma from noise estimate + |H| floor, prints spectral stats
• Auto 2-D circular alignment (optional) between Xeq and Xgt (fixes off-by-(dn,dm) bugs)
• Rich debugging: SNR, EVM/corr pre/post, best shift, |H| stats, saved scatters & heatmaps

OFDM path:
• Comb pilots + LS + 2-D nearest interpolation + pre-EQ (same as before)

Author: AISensing
"""

import os
import math
import argparse
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Config -----------------------------
@dataclass
class ISACConfig:
    seed: int = 42
    device: str = "cpu"

    # Grid
    N_SC: int = 32
    M_SYM: int = 16
    CP_LEN: int = 8

    # Channel constraints (keep learnable/OFDM-friendly)
    MAX_DELAY: int = 4            # <= CP_LEN-1
    MAX_DOPPLER: float = 1e-3     # cycles/sample (small ICI)

    # OFDM pilots
    ofdm_df: int = 8
    ofdm_dt: int = 4
    pilot_value: complex = 1.0 + 0.0j

    # OTFS options
    otfs_token_mode: str = "deconv"     # "deconv" or "raw_plus_H"
    otfs_wiener_c: float = 1e-3         # base gamma
    otfs_gamma_from_noise: bool = True  # estimate gamma from measured SNR
    otfs_H_floor: float = 1e-3          # floor on |Hf| to avoid divide-by-0
    otfs_align_after_deconv: bool = True  # auto 2-D circular align Xeq to Xgt
    otfs_save_raw_and_deconv_scatter: bool = True

    # Train/Eval
    epochs: int = 8
    batch_size: int = 64
    lr: float = 2e-3
    train_samples: int = 6000
    val_samples: int = 800
    test_samples: int = 800

    # SNR
    snr_train_min: float = 0.0
    snr_train_max: float = 20.0
    snr_eval_list: tuple = (0, 5, 10, 15, 20)

    # Mods
    train_mods: tuple = ("BPSK", "QPSK", "QAM16", "QAM64")
    eval_mods: tuple  = ("BPSK", "QPSK", "QAM16", "QAM64", "QAM256")

    # Model
    d_model: int = 128
    nhead: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    # Output & debug
    outdir: str = "./outputs_isac"
    debug_save_every: int = 400
    debug_max_figs: int = 10
    verbose_debug: bool = True

# ----------------------------- Utils -----------------------------
def mmse_equalize(Xh, Hhat, snr_db):
    """Per-RE MMSE equalization assuming Es=1 (constellations normalized)."""
    snr_lin = 10**(snr_db/10.0)
    den = (np.abs(Hhat)**2 + 1.0/snr_lin)
    return np.conj(Hhat) / den * Xh

def set_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed)

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    p = np.mean(np.abs(signal)**2) + 1e-12
    nvar = p / (10**(snr_db/10.0))
    noise = np.sqrt(nvar/2.0) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return (signal + noise.astype(np.complex64)).astype(np.complex64)

def measure_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    n = noisy - clean
    ps = np.mean(np.abs(clean)**2) + 1e-12
    pn = np.mean(np.abs(n)**2) + 1e-12
    return float(10*np.log10(ps/pn))

def evm_rms(x_hat: np.ndarray, x_ref: np.ndarray, mask: np.ndarray=None) -> float:
    if mask is not None:
        x_hat = x_hat.reshape(-1)[mask.reshape(-1)]
        x_ref = x_ref.reshape(-1)[mask.reshape(-1)]
    num = np.mean(np.abs(x_hat - x_ref)**2) + 1e-12
    den = np.mean(np.abs(x_ref)**2) + 1e-12
    return float(np.sqrt(num/den))

def pearson_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray=None) -> float:
    if mask is not None:
        a = a.reshape(-1)[mask.reshape(-1)]; b = b.reshape(-1)[mask.reshape(-1)]
    a = a.reshape(-1); b = b.reshape(-1)
    if a.size < 2: return 0.0
    ar = a.real; br = b.real
    cov = np.cov(ar, br); denom = np.sqrt(cov[0,0]*cov[1,1]) + 1e-12
    return float(cov[0,1]/denom)

def _normalize_tokens(tokens: np.ndarray) -> np.ndarray:
    m = tokens.mean(axis=0, keepdims=True)
    s = tokens.std(axis=0, keepdims=True) + 1e-6
    return (tokens - m)/s

def save_scatter(points: np.ndarray, title: str, out_png: str, s=6):
    plt.figure(); plt.scatter(points.real, points.imag, s=s, alpha=0.5)
    plt.axhline(0,color='k',lw=0.5); plt.axvline(0,color='k',lw=0.5)
    plt.title(title); plt.xlabel("I"); plt.ylabel("Q")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def save_heatmap(mat: np.ndarray, title: str, out_png: str):
    plt.figure(); plt.imshow(np.abs(mat), aspect='auto'); plt.colorbar()
    plt.title(title); plt.tight_layout(); plt.savefig(out_png); plt.close()

# ----------------------------- Constellations -----------------------------
def gray_to_binary(g):
    b = 0
    while g: b ^= g; g >>= 1
    return b

def bits_from_int(x, k):
    return [(x >> i) & 1 for i in range(k-1, -1, -1)]

def qam_square_constellation(M):
    k = int(round(math.log2(M))); m_side = int(round(math.sqrt(M))); assert m_side*m_side == M
    bps = k//2; levels = np.arange(-(m_side-1), (m_side-1)+2, 2)
    pts=[]; bits_list=[]
    for gi in range(m_side):
        i_idx = gray_to_binary(gi)
        for gq in range(m_side):
            q_idx = gray_to_binary(gq)
            I = levels[i_idx]; Q = levels[q_idx]
            pts.append(I + 1j*Q)
            bits_list.append(bits_from_int(gi, bps) + bits_from_int(gq, bps))
    pts = np.array(pts, np.complex64); pts = pts/np.sqrt(np.mean(np.abs(pts)**2))
    bits_arr = np.array(bits_list, np.int64)
    return pts, bits_arr

def bpsk_constellation():
    pts = np.array([1+0j, -1+0j], np.complex64); bits = np.array([[0],[1]], np.int64)
    return pts, bits

def get_constellation(mod: str):
    m = mod.upper()
    if m=="BPSK": return bpsk_constellation()
    if m=="QPSK": return qam_square_constellation(4)
    if m=="QAM16": return qam_square_constellation(16)
    if m=="QAM64": return qam_square_constellation(64)
    if m=="QAM256": return qam_square_constellation(256)
    raise ValueError("Unsupported modulation: "+mod)

def modulate_bits(bits, mod):
    pts, bits_tbl = get_constellation(mod); k = bits_tbl.shape[1]
    bits = np.array(bits, np.int64).reshape(-1, k)
    pow2 = (1 << np.arange(k-1, -1, -1)).astype(np.int64)
    tbl_idx = (bits_tbl*pow2).sum(1); LUT = -np.ones(2**k, np.int64); LUT[tbl_idx] = np.arange(len(tbl_idx))
    idx = (bits*pow2).sum(1); return pts[LUT[idx]]

def nearest_bits(symbols, mod):
    pts, bits_tbl = get_constellation(mod)
    d2 = np.abs(symbols.reshape(-1,1) - pts.reshape(1,-1))**2
    idx = np.argmin(d2, axis=1); return bits_tbl[idx].reshape(-1)

# ----------------------------- Channel & OFDM/OTFS ops -----------------------------
def rand_delay_doppler_paths(n_paths=3, max_delay=4, max_doppler=1e-3):
    delays = np.random.randint(0, max_delay+1, size=n_paths)
    dopplers = np.random.uniform(-max_doppler, max_doppler, size=n_paths)
    gains = (np.random.randn(n_paths) + 1j*np.random.randn(n_paths)).astype(np.complex64) / np.sqrt(2*n_paths)
    return delays, dopplers, gains

def apply_dd_channel_blockwise(signal: np.ndarray, block_len: int, M_blocks: int,
                               delays: np.ndarray, dopplers: np.ndarray, gains: np.ndarray) -> np.ndarray:
    y = np.zeros_like(signal, dtype=np.complex64)
    for m in range(M_blocks):
        a = m*block_len; b = a+block_len
        blk = signal[a:b]
        n_global = a + np.arange(block_len, dtype=np.float32)
        yblk = np.zeros_like(blk, dtype=np.complex64)
        for d, fd, g in zip(delays, dopplers, gains):
            blk_del = np.roll(blk, int(d))
            phase = np.exp(1j * 2*np.pi * fd * n_global).astype(np.complex64)
            yblk += g * blk_del * phase
        y[a:b] = yblk
    return y

# OTFS (simple SFFT surrogate)
def otfs_modulate_dd_grid(X_nm):
    N, M = X_nm.shape
    x_tf = np.fft.ifft2(X_nm)
    tx = [np.fft.ifft(x_tf[:, m]) for m in range(M)]
    return np.concatenate(tx).astype(np.complex64)

def otfs_demodulate_to_dd(rx, N, M):
    x_tf = np.zeros((N, M), dtype=np.complex64)
    for m in range(M):
        blk = rx[m*N:(m+1)*N]
        x_tf[:, m] = np.fft.fft(blk)
    return np.fft.fft2(x_tf)

def otfs_impulse_pilot_grid(N, M, amp=1.0+0j, pos=(0,0)):
    X = np.zeros((N,M), dtype=np.complex64); X[pos[0], pos[1]] = amp; return X

def otfs_2d_wiener_deconv(Xh_data, Hhat, gamma=1e-3, Hfloor=1e-3, dbg_prefix=None, outdir=None):
    Yf = np.fft.fft2(Xh_data); Hf = np.fft.fft2(Hhat)
    mag = np.abs(Hf); mag = np.maximum(mag, Hfloor)
    denom = (mag**2 + gamma)
    Xeq = np.fft.ifft2(Yf * np.conj(Hf) / denom)
    if dbg_prefix and outdir:
        os.makedirs(outdir, exist_ok=True)
        stats = (float(np.min(mag)), float(np.median(mag)), float(np.max(mag)))
        with open(os.path.join(outdir, f"{dbg_prefix}_wiener_stats.txt"), "w") as f:
            f.write(f"|Hf| min/med/max: {stats}\n")
            f.write(f"gamma={gamma} Hfloor={Hfloor}\n")
    return Xeq

# OFDM
def ofdm_modulate_grid(X_fm, N, M, CP_LEN):
    tx=[]
    for m in range(M):
        sym_t = np.fft.ifft(X_fm[:, m])
        with_cp = np.concatenate([sym_t[-CP_LEN:], sym_t])
        tx.append(with_cp)
    return np.concatenate(tx).astype(np.complex64)

def ofdm_demodulate(rx, N, M, CP_LEN):
    X = np.zeros((N, M), dtype=np.complex64)
    idx=0
    for m in range(M):
        blk = rx[idx:idx+N+CP_LEN]; idx += N+CP_LEN
        t = blk[CP_LEN:]; X[:, m] = np.fft.fft(t)
    return X

# OFDM pilots
def build_pilot_mask(N, M, df, dt):
    mask = np.zeros((N, M), dtype=bool)
    for m in range(0, M, max(1, dt)):
        for n in range(0, N, max(1, df)):
            mask[n, m] = True
    return mask

def bilinear_interp_from_pilots(Xh, pilot_mask, pilot_value, df, dt):
    """
    Bilinear upsampling from the coarse pilot grid to full NxM.
    Assumes df|N and dt|M (true for your settings: 32/8 and 16/4).
    """
    N, M = Xh.shape
    n_grid = np.arange(0, N, max(1, df))
    m_grid = np.arange(0, M, max(1, dt))
    # coarse LS on the pilot lattice
    Hc = Xh[np.ix_(n_grid, m_grid)] / (pilot_value + 1e-12)   # [Nc, Mc]

    Nc, Mc = Hc.shape
    Hhat = np.zeros((N, M), dtype=np.complex64)
    for n in range(N):
        i0 = (n // df); i1 = min(i0 + 1, Nc - 1)
        tn = (n - i0*df) / float(df) if i1 > i0 else 0.0
        for m in range(M):
            j0 = (m // dt); j1 = min(j0 + 1, Mc - 1)
            tm = (m - j0*dt) / float(dt) if j1 > j0 else 0.0
            Hhat[n, m] = (
                (1-tn)*(1-tm)*Hc[i0, j0] +
                 tn   *(1-tm)*Hc[i1, j0] +
                (1-tn)* tm   *Hc[i0, j1] +
                 tn   * tm   *Hc[i1, j1]
            )
    return Hhat

def nearest_interp_from_pilots(Hls, pilot_mask, df, dt):
    N, M = pilot_mask.shape
    n_grid = np.arange(0, N, max(1, df)); m_grid = np.arange(0, M, max(1, dt))
    n_idx = np.clip(np.round(np.arange(N)/max(1,df)).astype(int), 0, len(n_grid)-1)
    m_idx = np.clip(np.round(np.arange(M)/max(1,dt)).astype(int), 0, len(m_grid)-1)
    return Hls[n_grid[n_idx][:,None], m_grid[m_idx][None,:]]

def estimate_channel_from_pilots(Xh, pilot_mask, pilot_value, df, dt, mode="bilinear"):
    if mode == "bilinear":
        return bilinear_interp_from_pilots(Xh, pilot_mask, pilot_value, df, dt)
    else:
        # fallback: nearest
        Hls = np.zeros_like(Xh)
        Hls[pilot_mask] = Xh[pilot_mask] / (pilot_value + 1e-12)
        return nearest_interp_from_pilots(Hls, pilot_mask, df, dt)

# ----------------------------- Alignment (critical for OTFS) -----------------------------
def best_circ_shift(A: np.ndarray, B: np.ndarray):
    """
    Find 2-D circular shift (dn,dm) maximizing correlation between complex A and B.
    Returns (dn, dm, peak_corr_val).
    """
    Af = np.fft.fft2(A); Bf = np.fft.fft2(B)
    corr = np.fft.ifft2(Af * np.conj(Bf))
    k = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    dn = int(k[0]); dm = int(k[1])
    peak = float(np.abs(corr[k]))
    return dn, dm, peak

def circ_roll_2d(X, dn, dm):
    return np.roll(np.roll(X, -dn, axis=0), -dm, axis=1)

# ----------------------------- Dataset helpers -----------------------------
def _fill_grid_with_bits(N, M, mask, bits, mod):
    pts, bits_tbl = get_constellation(mod); k = bits_tbl.shape[1]
    assert bits.size == mask.sum()*k
    data_syms = modulate_bits(bits, mod)
    X = np.zeros((N, M), dtype=np.complex64); X[mask] = data_syms.astype(np.complex64)
    return X

# ----------------------------- Datasets -----------------------------
class CommTrainDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: ISACConfig, n_samples, waveform='OTFS', debug_dir=None):
        super().__init__()
        self.cfg = cfg; self.n = n_samples; self.waveform = waveform
        self.debug_dir = debug_dir; self._saved = 0

    def __len__(self): return self.n

    def __getitem__(self, idx):
        c = self.cfg; N, M, CP = c.N_SC, c.M_SYM, c.CP_LEN
        mod = np.random.choice(c.train_mods)
        delays, dopplers, gains = rand_delay_doppler_paths(3, c.MAX_DELAY, c.MAX_DOPPLER)

        if self.waveform == 'OFDM':
            pilot_mask = build_pilot_mask(N, M, c.ofdm_df, c.ofdm_dt); data_mask = ~pilot_mask
            _, bits_tbl = get_constellation(mod); k = bits_tbl.shape[1]
            bits = np.random.randint(0, 2, size=(data_mask.sum()*k,), dtype=np.int64)
            X_data = _fill_grid_with_bits(N, M, data_mask, bits, mod)
            X_tx = X_data.copy(); X_tx[pilot_mask] = c.pilot_value

            tx = ofdm_modulate_grid(X_tx, N, M, CP)
            rx_clean = apply_dd_channel_blockwise(tx, N+CP, M, delays, dopplers, gains)
            snr_db = np.random.uniform(c.snr_train_min, c.snr_train_max)
            rx = add_awgn(rx_clean, snr_db)

            Xh = ofdm_demodulate(rx, N, M, CP)
            Hhat = estimate_channel_from_pilots(Xh, pilot_mask, c.pilot_value, c.ofdm_df, c.ofdm_dt, mode="bilinear")
            # BEFORE:
            # Xeq = Xh / (Hhat + 1e-8)

            # AFTER:
            Xeq = mmse_equalize(Xh, Hhat, snr_db)

            if self.debug_dir and idx % c.debug_save_every == 0 and self._saved < c.debug_max_figs:
                msnr = measure_snr_db(rx_clean, rx)
                evm_pre = evm_rms(Xh, X_tx); evm_post = evm_rms(Xeq, X_tx); corr_post = pearson_corr(Xeq, X_tx)
                print(f"[DBG OFDM] idx={idx} SNR={msnr:.2f}dB EVMpre={evm_pre:.3f} EVMpost={evm_post:.3f} corr_post={corr_post:.3f}")
                os.makedirs(self.debug_dir, exist_ok=True)
                save_heatmap(Hhat, "|Ĥ| OFDM", os.path.join(self.debug_dir, f"ofdm_Hhat_{idx}.png"))
                save_scatter(Xeq.reshape(-1)[data_mask.reshape(-1)], "OFDM equalized data REs",
                             os.path.join(self.debug_dir, f"ofdm_const_{idx}.png"))
                self._saved += 1

            tokens = np.stack([Xeq.real, Xeq.imag], axis=-1).astype(np.float32).reshape(N*M, 2)
            tokens = _normalize_tokens(tokens)
            y_iq = np.stack([X_data.real, X_data.imag], axis=-1).astype(np.float32).reshape(N*M, 2)
            mask_flat = data_mask.reshape(-1)

        else:
            # OTFS: impulse pilot frame for Hhat; data frame for X; two token modes
            _, bits_tbl = get_constellation(mod); k = bits_tbl.shape[1]
            bits = np.random.randint(0, 2, size=(N*M*k,), dtype=np.int64)
            X_data = modulate_bits(bits, mod).reshape(N, M)
            X_pilot = otfs_impulse_pilot_grid(N, M, amp=1.0+0j, pos=(0,0))

            tx_p = otfs_modulate_dd_grid(X_pilot)
            tx_d = otfs_modulate_dd_grid(X_data)

            rxp_clean = apply_dd_channel_blockwise(tx_p, N, M, delays, dopplers, gains)
            rxd_clean = apply_dd_channel_blockwise(tx_d, N, M, delays, dopplers, gains)
            snr_db = np.random.uniform(c.snr_train_min, c.snr_train_max)
            rxp = add_awgn(rxp_clean, snr_db); rxd = add_awgn(rxd_clean, snr_db)

            Xh_p = otfs_demodulate_to_dd(rxp, N, M)   # ≈ H in DD
            Hhat = Xh_p
            Xh_d = otfs_demodulate_to_dd(rxd, N, M)

            debug_now = self.debug_dir and idx % c.debug_save_every == 0 and self._saved < c.debug_max_figs
            if c.otfs_token_mode == "deconv":
                # choose gamma adaptively from SNR (rough)
                gamma = c.otfs_wiener_c
                if c.otfs_gamma_from_noise:
                    msnr = measure_snr_db(rxd_clean, rxd)
                    # Heuristic: higher noise -> larger gamma
                    gamma = c.otfs_wiener_c * (10**(-msnr/10.0) * N * M + 1.0)
                Xeq = otfs_2d_wiener_deconv(Xh_d, Hhat, gamma=gamma, Hfloor=c.otfs_H_floor,
                                            dbg_prefix=f"otfs_{idx}", outdir=self.debug_dir if debug_now else None)

                # Optional 2-D alignment to ground truth
                dn = dm = 0
                if c.otfs_align_after_deconv:
                    dn, dm, peak = best_circ_shift(Xeq, X_data)
                    if debug_now:
                        print(f"[DBG OTFS] idx={idx} best shift dn={dn} dm={dm} peak={peak:.3f}")
                    Xeq = circ_roll_2d(Xeq, dn, dm)

                if debug_now:
                    msnr_p = measure_snr_db(rxp_clean, rxp); msnr_d = measure_snr_db(rxd_clean, rxd)
                    evm_pre = evm_rms(Xh_d, X_data); evm_post = evm_rms(Xeq, X_data); corr_post = pearson_corr(Xeq, X_data)
                    print(f"[DBG OTFS] idx={idx} SNRp/dat={msnr_p:.2f}/{msnr_d:.2f}dB  EVMpre={evm_pre:.3f} EVMpost={evm_post:.3f} corr_post={corr_post:.3f}")
                    os.makedirs(self.debug_dir, exist_ok=True)
                    save_heatmap(Hhat, "|Ĥ| OTFS (DD)", os.path.join(self.debug_dir, f"otfs_Hhat_{idx}.png"))
                    if c.otfs_save_raw_and_deconv_scatter:
                        save_scatter(Xh_d.reshape(-1), "OTFS raw DD obs", os.path.join(self.debug_dir, f"otfs_raw_const_{idx}.png"))
                        save_scatter(Xeq.reshape(-1), "OTFS deconv DD eq", os.path.join(self.debug_dir, f"otfs_deconv_const_{idx}.png"))
                    self._saved += 1

                tokens = np.stack([Xeq.real, Xeq.imag], axis=-1).astype(np.float32).reshape(N*M, 2)
                tokens = _normalize_tokens(tokens)
                y_iq = np.stack([X_data.real, X_data.imag], axis=-1).astype(np.float32).reshape(N*M, 2)
                mask_flat = np.ones(N*M, dtype=bool)

            elif c.otfs_token_mode == "raw_plus_H":
                # feed raw obs + Hhat (4 channels) for the model to learn equalization
                if debug_now:
                    msnr_p = measure_snr_db(rxp_clean, rxp); msnr_d = measure_snr_db(rxd_clean, rxd)
                    evm_raw = evm_rms(Xh_d, X_data); corr_raw = pearson_corr(Xh_d, X_data)
                    print(f"[DBG OTFS] idx={idx} SNRp/dat={msnr_p:.2f}/{msnr_d:.2f}dB  RAW EVM={evm_raw:.3f} raw_corr={corr_raw:.3f} (model will learn EQ)")
                    os.makedirs(self.debug_dir, exist_ok=True)
                    save_heatmap(Hhat, "|Ĥ| OTFS (DD)", os.path.join(self.debug_dir, f"otfs_Hhat_{idx}.png"))
                    save_scatter(Xh_d.reshape(-1), "OTFS raw DD obs", os.path.join(self.debug_dir, f"otfs_raw_const_{idx}.png"))
                    self._saved += 1

                t4 = np.stack([Xh_d.real, Xh_d.imag, Hhat.real, Hhat.imag], axis=-1).astype(np.float32).reshape(N*M, 4)
                t4 = _normalize_tokens(t4)
                tokens = t4
                y_iq = np.stack([X_data.real, X_data.imag], axis=-1).astype(np.float32).reshape(N*M, 2)
                mask_flat = np.ones(N*M, dtype=bool)
            else:
                raise ValueError(f"Unknown otfs_token_mode: {c.otfs_token_mode}")

        return torch.from_numpy(tokens), torch.from_numpy(y_iq), torch.from_numpy(mask_flat)

class CommEvalDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: ISACConfig, n_samples, waveform='OTFS', snr_db=10, modulation="QPSK"):
        super().__init__()
        self.cfg = cfg; self.n = n_samples; self.waveform = waveform
        self.snr_db = snr_db; self.mod = modulation

    def __len__(self): return self.n

    def __getitem__(self, idx):
        c = self.cfg; N, M, CP = c.N_SC, c.M_SYM, c.CP_LEN
        delays, dopplers, gains = rand_delay_doppler_paths(3, c.MAX_DELAY, c.MAX_DOPPLER)

        if self.waveform == 'OFDM':
            pilot_mask = build_pilot_mask(N, M, c.ofdm_df, c.ofdm_dt); data_mask = ~pilot_mask
            _, bits_tbl = get_constellation(self.mod); k = bits_tbl.shape[1]
            bits = np.random.randint(0,2,size=(data_mask.sum()*k,), dtype=np.int64)
            X_data = _fill_grid_with_bits(N, M, data_mask, bits, self.mod)
            X_tx = X_data.copy(); X_tx[pilot_mask] = c.pilot_value
            tx = ofdm_modulate_grid(X_tx, N, M, CP)
            rx_clean = apply_dd_channel_blockwise(tx, N+CP, M, delays, dopplers, gains)
            rx = add_awgn(rx_clean, self.snr_db)
            Xh = ofdm_demodulate(rx, N, M, CP)
            Hhat = estimate_channel_from_pilots(Xh, pilot_mask, c.pilot_value, c.ofdm_df, c.ofdm_dt)
            #Xeq = Xh / (Hhat + 1e-8)
            # Xeq = Xh / (Hhat + 1e-8)
            Xeq = mmse_equalize(Xh, Hhat, self.snr_db)
            tokens = np.stack([Xeq.real, Xeq.imag], axis=-1).astype(np.float32).reshape(N*M, 2)
            tokens = _normalize_tokens(tokens)
            y_iq = np.stack([X_data.real, X_data.imag], axis=-1).astype(np.float32).reshape(N*M, 2)
            return torch.from_numpy(tokens), torch.from_numpy(y_iq), torch.from_numpy(bits.astype(np.int64)), torch.from_numpy(data_mask.reshape(-1))

        else:
            _, bits_tbl = get_constellation(self.mod); k = bits_tbl.shape[1]
            bits = np.random.randint(0,2,size=(N*M*k,), dtype=np.int64)
            X_data = modulate_bits(bits, self.mod).reshape(N, M)
            X_pilot = otfs_impulse_pilot_grid(N, M, amp=1.0+0j, pos=(0,0))
            tx_p = otfs_modulate_dd_grid(X_pilot); tx_d = otfs_modulate_dd_grid(X_data)
            rxp_clean = apply_dd_channel_blockwise(tx_p, N, M, delays, dopplers, gains)
            rxd_clean = apply_dd_channel_blockwise(tx_d, N, M, delays, dopplers, gains)
            rxp = add_awgn(rxp_clean, self.snr_db); rxd = add_awgn(rxd_clean, self.snr_db)
            Xh_p = otfs_demodulate_to_dd(rxp, N, M); Hhat = Xh_p
            Xh_d = otfs_demodulate_to_dd(rxd, N, M)

            if c.otfs_token_mode == "deconv":
                gamma = c.otfs_wiener_c
                if c.otfs_gamma_from_noise:
                    msnr = measure_snr_db(rxd_clean, rxd)
                    gamma = c.otfs_wiener_c * (10**(-msnr/10.0) * N * M + 1.0)
                Xeq = otfs_2d_wiener_deconv(Xh_d, Hhat, gamma=gamma, Hfloor=c.otfs_H_floor)
                if c.otfs_align_after_deconv:
                    dn, dm, _ = best_circ_shift(Xeq, X_data)
                    Xeq = circ_roll_2d(Xeq, dn, dm)
                tokens = np.stack([Xeq.real, Xeq.imag], axis=-1).astype(np.float32).reshape(N*M, 2)
                tokens = _normalize_tokens(tokens)
            else:
                t4 = np.stack([Xh_d.real, Xh_d.imag, Hhat.real, Hhat.imag], axis=-1).astype(np.float32).reshape(N*M, 4)
                tokens = _normalize_tokens(t4)

            y_iq = np.stack([X_data.real, X_data.imag], axis=-1).astype(np.float32).reshape(N*M, 2)
            data_mask = np.ones(N*M, dtype=bool)
            return torch.from_numpy(tokens), torch.from_numpy(y_iq), torch.from_numpy(bits.astype(np.int64)), torch.from_numpy(data_mask)

# ----------------------------- Model -----------------------------
class CommTransformer(nn.Module):
    def __init__(self, seq_len, in_dim=2, d_model=128, nhead=4, n_layers=2, dropout=0.1, device="cpu"):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(in_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=2*d_model, dropout=dropout,
                                         batch_first=True)
        self.backbone = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(d_model, 2)
        self.register_buffer("pos", self._pos(seq_len, d_model, device))

    @staticmethod
    def _pos(L, d, device):
        pos = torch.arange(0, L, dtype=torch.float32, device=device).unsqueeze(1)
        i = torch.arange(0, d, dtype=torch.float32, device=device).unsqueeze(0)
        angle_rates = 1.0 / torch.pow(10000, (2*(i//2))/d)
        angles = pos * angle_rates
        pe = torch.zeros(L, d, device=device); pe[:,0::2] = torch.sin(angles[:,0::2]); pe[:,1::2] = torch.cos(angles[:,1::2])
        return pe

    def forward(self, x):
        h = self.input_proj(x) + self.pos[:x.shape[1], :]
        z = self.backbone(h)
        return self.head(z)

# ----------------------------- Train/Eval -----------------------------
def train_comm_model(cfg: ISACConfig, waveform='OTFS'):
    device = torch.device(cfg.device); set_seed(cfg.seed)
    seq_len = cfg.N_SC * cfg.M_SYM

    in_dim = 2 if (waveform=='OFDM' or cfg.otfs_token_mode=='deconv') else 4
    debug_dir = os.path.join(cfg.outdir, f"debug_{waveform.lower()}")
    model = CommTransformer(seq_len=seq_len, in_dim=in_dim, d_model=cfg.d_model,
                            nhead=cfg.nhead, n_layers=cfg.n_layers,
                            dropout=cfg.dropout, device=cfg.device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    tr_ds = CommTrainDataset(cfg, cfg.train_samples, waveform=waveform, debug_dir=debug_dir)
    va_ds = CommTrainDataset(cfg, cfg.val_samples, waveform=waveform, debug_dir=None)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False)

    best_val = 1e9; best_state=None
    for ep in range(cfg.epochs):
        model.train(); s=0.0; n=0
        for it, (x, y, mask) in enumerate(tr_loader):
            x=x.to(device); y=y.to(device); mask=mask.to(device).bool()
            yhat = model(x)
            loss = F.mse_loss(yhat[mask], y[mask])
            opt.zero_grad(); loss.backward(); opt.step()
            s += loss.item()*x.shape[0]; n += x.shape[0]
            if cfg.verbose_debug and ep==0 and it % 100 == 0:
                print(f"[DBG {waveform}] ep={ep+1} it={it} token_mean={x.mean().item():.3f} token_std={x.std().item():.3f}")
        tr = s/max(1,n)

        model.eval(); s=0.0; n=0
        with torch.no_grad():
            for x, y, mask in va_loader:
                x=x.to(device); y=y.to(device); mask=mask.to(device).bool()
                yhat = model(x)
                loss = F.mse_loss(yhat[mask], y[mask])
                s += loss.item()*x.shape[0]; n += x.shape[0]
        va = s/max(1,n)

        if va < best_val:
            best_val = va; best_state = {k:v.cpu() for k,v in model.state_dict().items()}
        print(f"[COMM {waveform}] Epoch {ep+1}/{cfg.epochs} train={tr:.4f} val={va:.4f}")

    model.load_state_dict(best_state)
    return model

def eval_comm_ber(model: nn.Module, cfg: ISACConfig, waveform='OTFS',
                  snr_db=10, modulation="QPSK", n_samples=512, save_debug=False, tag=""):
    device = torch.device(cfg.device); model.eval()
    ds = CommEvalDataset(cfg, n_samples, waveform=waveform, snr_db=snr_db, modulation=modulation)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    total_bits=0; err_bits=0; dbg_done=False
    with torch.no_grad():
        for x, y, bits_true, data_mask in loader:
            pred = model(x.to(device)).cpu().numpy()
            sym = (pred[...,0] + 1j*pred[...,1]).reshape(-1)
            mask = data_mask.numpy().reshape(-1).astype(bool)
            sym_data = sym[mask]
            bits_hat = nearest_bits(sym_data, modulation)
            bt = bits_true.numpy().reshape(-1)
            L = min(len(bt), len(bits_hat))
            total_bits += L; err_bits += np.sum(bits_hat[:L] != bt[:L])
            if save_debug and not dbg_done:
                out_dir = os.path.join(cfg.outdir, "eval_debug"); os.makedirs(out_dir, exist_ok=True)
                save_scatter(sym_data, f"{waveform} NN equalized ({modulation}, {snr_db} dB)",
                             os.path.join(out_dir, f"{waveform}_nn_const_{modulation}_{snr_db}{tag}.png"))
                dbg_done=True
    return err_bits/max(1,total_bits)

# ----------------------------- Classical baselines -----------------------------
def classical_comm_ber_ofdm(cfg: ISACConfig, snr_db, modulation="QPSK"):
    N,M,CP = cfg.N_SC, cfg.M_SYM, cfg.CP_LEN
    pilot_mask = build_pilot_mask(N,M,cfg.ofdm_df,cfg.ofdm_dt); data_mask = ~pilot_mask
    _, bits_tbl = get_constellation(modulation); k = bits_tbl.shape[1]
    bits = np.random.randint(0,2,size=(data_mask.sum()*k,), dtype=np.int64)
    X_data = _fill_grid_with_bits(N,M,data_mask,bits,modulation); X_tx = X_data.copy(); X_tx[pilot_mask] = cfg.pilot_value
    tx = ofdm_modulate_grid(X_tx, N, M, CP)
    delays, dopplers, gains = rand_delay_doppler_paths(3, cfg.MAX_DELAY, cfg.MAX_DOPPLER)
    rx_clean = apply_dd_channel_blockwise(tx, N+CP, M, delays, dopplers, gains); rx = add_awgn(rx_clean, snr_db)
    Xh = ofdm_demodulate(rx, N, M, CP); Hhat = estimate_channel_from_pilots(Xh, pilot_mask, cfg.pilot_value, cfg.ofdm_df, cfg.ofdm_dt)
    Xeq = Xh / (Hhat + 1e-8); 
    sym = Xeq.reshape(-1)[data_mask.reshape(-1)]
    bits_hat = nearest_bits(sym, modulation); return np.mean(bits_hat != bits[:len(bits_hat)])

def classical_comm_ber_otfs(cfg: ISACConfig, snr_db, modulation="QPSK"):
    N,M = cfg.N_SC, cfg.M_SYM
    _, bits_tbl = get_constellation(modulation); k = bits_tbl.shape[1]
    bits = np.random.randint(0,2,size=(N*M*k,), dtype=np.int64)
    X_data = modulate_bits(bits, modulation).reshape(N, M); X_pilot = otfs_impulse_pilot_grid(N, M, 1.0+0j, (0,0))
    tx_p = otfs_modulate_dd_grid(X_pilot); tx_d = otfs_modulate_dd_grid(X_data)
    delays, dopplers, gains = rand_delay_doppler_paths(3, cfg.MAX_DELAY, cfg.MAX_DOPPLER)
    rxp_clean = apply_dd_channel_blockwise(tx_p, N, M, delays, dopplers, gains)
    rxd_clean = apply_dd_channel_blockwise(tx_d, N, M, delays, dopplers, gains)
    rxp = add_awgn(rxp_clean, snr_db); rxd = add_awgn(rxd_clean, snr_db)
    Xh_p = otfs_demodulate_to_dd(rxp, N, M); Xh_d = otfs_demodulate_to_dd(rxd, N, M)

    # Robust Wiener with gamma from SNR
    gamma = cfg.otfs_wiener_c
    msnr = measure_snr_db(rxd_clean, rxd)
    gamma = cfg.otfs_wiener_c * (10**(-msnr/10.0) * N * M + 1.0)
    Xeq = otfs_2d_wiener_deconv(Xh_d, Xh_p, gamma=gamma, Hfloor=cfg.otfs_H_floor)
    if cfg.otfs_align_after_deconv:
        dn, dm, _ = best_circ_shift(Xeq, X_data); Xeq = circ_roll_2d(Xeq, dn, dm)
    bits_hat = nearest_bits(Xeq.reshape(-1), modulation)
    return np.mean(bits_hat != bits[:len(bits_hat)])

# ----------------------------- Sanity fig -----------------------------
def debug_otfs_single_target_fig(N, M, snr_db=20, delay_bin=3, doppler_bin=5, save_png=None):
    X_nm = np.zeros((N, M), dtype=np.complex64); X_nm[delay_bin, doppler_bin] = 1+0j
    tx = otfs_modulate_dd_grid(X_nm)
    delays = np.array([delay_bin]); dopplers = np.array([0.0]); gains = np.array([1+0j], dtype=np.complex64)
    rx = apply_dd_channel_blockwise(tx, N, M, delays, dopplers, gains); rx = add_awgn(rx, snr_db)
    Xh = otfs_demodulate_to_dd(rx, N, M)
    mag = np.abs(Xh); plt.figure(); plt.imshow(mag, aspect='auto'); plt.colorbar()
    plt.title(f"OTFS single target @ (delay={delay_bin}, doppler={doppler_bin})")
    plt.xlabel("Doppler"); plt.ylabel("Delay"); plt.tight_layout()
    if save_png: plt.savefig(save_png); plt.close()
    else: plt.show()

# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--outdir", type=str, default="./outputs/isac")
    ap.add_argument("--otfs_token_mode", type=str, default=None, help="override: deconv | raw_plus_H")
    args = ap.parse_args()

    cfg = ISACConfig(device=args.device, epochs=args.epochs, outdir=args.outdir)
    if args.otfs_token_mode is not None:
        cfg.otfs_token_mode = args.otfs_token_mode
    os.makedirs(cfg.outdir, exist_ok=True)
    set_seed(cfg.seed)

    print(f"CP_LEN={cfg.CP_LEN}, MAX_DELAY={cfg.MAX_DELAY} (≤ CP_LEN-1); MAX_DOPPLER={cfg.MAX_DOPPLER}")
    print(f"OTFS token mode: {cfg.otfs_token_mode} | align_after_deconv={cfg.otfs_align_after_deconv}")

    print("=== Training neural receivers ===")
    comm_ofdm = train_comm_model(cfg, 'OFDM')
    comm_otfs = train_comm_model(cfg, 'OTFS')

    # BER sweep + saved scatters for a few points
    rows=[]
    for mod in cfg.eval_mods:
        for snr in cfg.snr_eval_list:
            ber_ofdm_class = classical_comm_ber_ofdm(cfg, snr, modulation=mod)
            ber_otfs_class = classical_comm_ber_otfs(cfg, snr, modulation=mod)
            save_dbg = (mod in ("BPSK","QPSK")) and (snr in (10, 20))
            ber_ofdm_nn = eval_comm_ber(comm_ofdm, cfg, 'OFDM', snr, mod, cfg.test_samples, save_dbg, tag="_ofdm")
            ber_otfs_nn = eval_comm_ber(comm_otfs, cfg, 'OTFS', snr, mod, cfg.test_samples, save_dbg, tag="_otfs")
            rows += [
                {"Waveform":"OFDM","Model":"Classical","Mod":mod,"SNR(dB)":snr,"BER":ber_ofdm_class},
                {"Waveform":"OTFS","Model":"Classical","Mod":mod,"SNR(dB)":snr,"BER":ber_otfs_class},
                {"Waveform":"OFDM","Model":"Transformer","Mod":mod,"SNR(dB)":snr,"BER":ber_ofdm_nn},
                {"Waveform":"OTFS","Model":"Transformer","Mod":mod,"SNR(dB)":snr,"BER":ber_otfs_nn},
            ]
            print(f"[SNR={snr:2d} | {mod:6s}] OFDM cls/nn: {ber_ofdm_class:.3e}/{ber_ofdm_nn:.3e} | OTFS cls/nn: {ber_otfs_class:.3e}/{ber_otfs_nn:.3e}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg.outdir, "comm_metrics.csv"); df.to_csv(csv_path, index=False)

    # BER plots
    for mod in cfg.eval_mods:
        sub = df[df.Mod==mod]
        if len(sub)==0: continue
        plt.figure()
        for (wave, model) in [("OFDM","Classical"),("OFDM","Transformer"),("OTFS","Classical"),("OTFS","Transformer")]:
            xs=[]; ys=[]
            for snr in cfg.snr_eval_list:
                q = sub[(sub.Waveform==wave)&(sub.Model==model)&(sub["SNR(dB)"]==snr)]
                if len(q)==0: continue
                xs.append(snr); ys.append(q["BER"].values[0])
            if xs: plt.plot(xs, ys, marker='o', label=f"{wave} {model}")
        plt.yscale('log'); plt.xlabel("SNR (dB)"); plt.ylabel("BER"); plt.title(f"BER vs SNR — {mod}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(cfg.outdir, f"ber_{mod}.png")); plt.close()

    debug_otfs_single_target_fig(cfg.N_SC, cfg.M_SYM, snr_db=15, delay_bin=2, doppler_bin=7,
                                 save_png=os.path.join(cfg.outdir, "debug_otfs_dd.png"))

    print(f"Saved metrics to {csv_path}")
    print(f"Figures in {cfg.outdir}")

if __name__ == "__main__":
    main()