"""
isac_main.py

High-level entry point for ISAC experiments.

This file wires together reusable utilities from isac_utils.py into four
conceptual stages:

  1) Dataset simulation with visualizations
     - radar scenes + GT heatmaps, plus quick-look figures.

  2) Traditional FMCW / OTFS radar and OFDM / OTFS communication baselines
     - FMCW vs OTFS Range–Doppler / Delay–Doppler visualization (from disk).
     - CFAR-based radar detection & BEV metrics aggregated over many samples.
     - BER vs Eb/N0 for OFDM & OTFS with hard QPSK demapping.

  3) Deep learning model training
     - UNetLite for radar RD→heatmap.
     - CNN demappers for OFDM and OTFS.

  4) Deep-learning evaluation & comparison with traditional methods
     - Radar: CFAR vs DL detector on a selected validation sample.
     - Comm : baseline BER vs DL demapper BER.

When you want to CHANGE EXPERIMENTS (hyperparameters, CFAR thresholds,
evaluation scenes, etc.), primarily edit THIS file.
The math-heavy utilities live in isac_utils.py.
"""

from pathlib import Path
import json

import torch

from isac_utils import (
    DEVICE,
    SystemParams,
    # Part 1 / dataset
    simulate_if_missing,
    visualize_radar_dataset_examples,
    # Radar DL
    train_radar_model,
    # Comm DL
    CommDemapperCNN,
    comm_dl_gen_batch_OFDM,
    comm_dl_gen_batch_OTFS,
    train_comm_demap,
    comm_demap_ber_curve,
    # Radar simulators & viz (used only in small places)
    raycast_torch,
    viz_rd_2d_compare,
    viz_scene_bev_compare,
    # Radar traditional detection & metrics
    cfar2d_ca,
    extract_detections,
    compute_radar_metrics,
    _rd_normalize,
    rd_dl_infer_to_points,
    # Radar BEV scene (optional)
    viz_bev_scene,
    # Comm baseline BER utilities
    run_ber_sweep_and_plot,
    viz_ber_compare_with_dl,
    compute_comm_metrics,
)

# ---------------------------------------------------------------------
# PART 2: traditional FMCW / OTFS radar + comm baselines (from dataset)
# ---------------------------------------------------------------------
def run_traditional_baselines_from_dataset(
    root,
    sp: SystemParams,
    split: str = "val",
    n_vis_samples: int = 4,
    max_samples: int | None = None,
):
    """
    Part 2: classical baselines using the DISK dataset (no RD/DD regeneration).

    Radar:
      - Loads saved FMCW RD maps and OTFS DD maps + GT from
          root / "radar" / split / *.npz
        (expects keys: rd_f_db, rd_o_db, gts)
      - Applies tuned CFAR for FMCW and OTFS.
      - For the first `n_vis_samples`:
          * Saves RD/DD comparison plots.
          * Saves BEV comparison plots with TP/FP overlays.
      - Aggregates TP/FP/FN and range/velocity errors across ALL samples
        to produce global metrics for FMCW and OTFS.

    Communications:
      - Runs run_ber_sweep_and_plot once for OFDM & OTFS (hard QPSK).
      - Saves ber_compare_baseline.pdf and comm_baseline_ber.npz.

    Returns
    -------
    metrics_f_global : dict
        Global FMCW CFAR metrics over all processed samples.
    metrics_o_global : dict
        Global OTFS CFAR metrics over all processed samples.
    eb_axis, ber_ofdm, ber_otfs, ber_theory : np.ndarray
        Baseline BER curves for later DL comparison.
    """
    import numpy as np  # local import to keep top-level clean

    root = Path(root)
    radar_dir = root / "radar" / split
    vis_dir = root / "radar_eval" / split
    vis_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(radar_dir.glob("*.npz"))
    if not files:
        print(f"[PART 2] No radar NPZ files found in {radar_dir}")
        return None, None, None, None, None, None

    if max_samples is not None:
        files = files[:max_samples]

    print(f"[PART 2] Evaluating classical radar baselines on {len(files)} samples from {radar_dir}")

    # Physical axes (shared by all samples)
    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    # Tuned CFAR configs (a bit stricter than the raw defaults)
    cfar_fmcw_cfg = dict(
        train=(10, 8),
        guard=(2, 2),
        pfa=3e-5,
        min_snr_db=12.0,
        notch_doppler_bins=2,
        apply_nms=True,
        max_peaks=40,
    )
    cfar_otfs_cfg = dict(
        train=(12, 10),
        guard=(2, 2),
        pfa=1e-5,
        min_snr_db=10.0,
        notch_doppler_bins=0,
        apply_nms=True,
        max_peaks=40,
    )

    # ----------------------- aggregators -----------------------
    agg_f = dict(TP=0, FP=0, FN=0, er_r=[], er_v=[])
    agg_o = dict(TP=0, FP=0, FN=0, er_r=[], er_v=[])

    def _accumulate(m: dict, agg: dict):
        agg["TP"] += int(m.get("TP", 0))
        agg["FP"] += int(m.get("FP", 0))
        agg["FN"] += int(m.get("FN", 0))
        agg["er_r"].extend(list(m.get("er_r", [])))
        agg["er_v"].extend(list(m.get("er_v", [])))

    # ------------------- per-sample loop ----------------------
    for idx, fpath in enumerate(files):
        data = np.load(fpath, allow_pickle=True)
        rd_f_db = data["rd_f_db"]  # (M, N//2)
        rd_o_db = data["rd_o_db"]  # (M, N)
        gts = json.loads(str(data["gts"])) if "gts" in data else []

        # ---- FMCW CFAR detections & metrics ----
        det_f_mask = cfar2d_ca(rd_f_db, **cfar_fmcw_cfg)
        dets_f = extract_detections(rd_f_db, det_f_mask, ra_f, va_f)
        metrics_f = compute_radar_metrics(dets_f, gts, sp, w_r=1.0, w_v=0.5)
        _accumulate(metrics_f, agg_f)

        # ---- OTFS CFAR detections & metrics ----
        det_o_mask = cfar2d_ca(rd_o_db, **cfar_otfs_cfg)
        dets_o = extract_detections(rd_o_db, det_o_mask, ra_o, va_o)
        metrics_o = compute_radar_metrics(dets_o, gts, sp, w_r=1.0, w_v=0.5)
        _accumulate(metrics_o, agg_o)

        # ---- optional visualizations for first n_vis_samples ----
        if idx < n_vis_samples:
            # 2D RD/DD comparison (FMCW CFAR is applied inside viz as well)
            viz_rd_2d_compare(
                vis_dir / f"sample_{idx:03d}_rd2d.pdf",
                rd_f_db,
                rd_o_db,
                gts,
                sp,
                cfar_cfg=cfar_fmcw_cfg,
            )

            # BEV comparison with TP/FP overlays (returns per-sample metrics)
            viz_scene_bev_compare(
                vis_dir / f"sample_{idx:03d}_bev_compare.pdf",
                dets_f,
                dets_o,
                gts,
                sp,
            )

            # Optional: 3D scene with raycast points (small subset only)
            if gts:
                pts, its, vels = raycast_torch(sp, gts)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                viz_bev_scene(
                    vis_dir / f"sample_{idx:03d}_scene",
                    pts,
                    gts,
                    sp,
                )

    # ------------------- global metrics ------------------------
    def _finalize(agg: dict, label: str):
        TP, FP, FN = agg["TP"], agg["FP"], agg["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_er_r = float(np.mean(np.abs(agg["er_r"]))) if agg["er_r"] else None
        mean_er_v = float(np.mean(np.abs(agg["er_v"]))) if agg["er_v"] else None
        metrics = dict(
            TP=TP,
            FP=FP,
            FN=FN,
            precision=precision,
            recall=recall,
            f1=f1,
            mean_abs_er_r=mean_er_r,
            mean_abs_er_v=mean_er_v,
        )
        print(f"[PART 2] Global radar metrics ({label}): {metrics}")
        return metrics

    metrics_f_global = _finalize(agg_f, "FMCW CFAR")
    metrics_o_global = _finalize(agg_o, "OTFS CFAR")

    # Save radar metrics JSON
    with open(root / f"radar_traditional_metrics_{split}.json", "w") as f:
        json.dump(
            {"fmcw": metrics_f_global, "otfs": metrics_o_global},
            f,
            indent=2,
        )

    # ----------------- comm baselines (OFDM / OTFS) ----------
    print("[PART 2] Running communication BER sweeps (OFDM & OTFS)...")
    eb_axis, ber_ofdm, ber_otfs, ber_theory = run_ber_sweep_and_plot(
        root / "ber_compare_baseline.pdf",
        ebn0_db_list=list(range(0, 21, 2)),
        ofdm_cfg=dict(Nfft=256, cp_len=32, n_ofdm_sym=800),
        otfs_cfg=dict(M=64, N=256, cp_len=32),
        rng_seed=2025,
    )
    print("[PART 2] Baseline BER figure written to ber_compare_baseline.pdf")

    # Save BER curves for reuse in Part 4
    import numpy as _np
    _np.savez(
        root / "comm_baseline_ber.npz",
        eb_axis=_np.asarray(eb_axis, dtype=float),
        ber_ofdm=_np.asarray(ber_ofdm, dtype=float),
        ber_otfs=_np.asarray(ber_otfs, dtype=float),
        ber_theory=_np.asarray(ber_theory, dtype=float),
    )

    print("[PART 2] Done.\n")
    return metrics_f_global, metrics_o_global, eb_axis, ber_ofdm, ber_otfs, ber_theory


# ---------------------------------------------------------------------
# PART 4: DL evaluation vs traditional (single sample + BER curves)
# ---------------------------------------------------------------------
def evaluate_and_visualize(
    out_dir,
    sp: SystemParams,
    radar_net,
    ofdm_model,
    otfs_model,
    sample_split: str = "val",
    sample_idx: int = 0,
    eb_axis=None,
    ber_ofdm_base=None,
    ber_otfs_base=None,
    ber_theory=None,
):
    """
    Part 4: High-level evaluation harness for DL vs traditional methods.

    This function assumes that:
      - Part 1 has already created a disk dataset under `out_dir/radar/...`
      - Part 2 has already generated OFDM/OTFS baseline BER curves (optional).

    Radar (single sample):
      1) Load one NPZ sample from disk (FMCW RD, OTFS DD, GT boxes).
      2) Apply *the same* CFAR config as in Part 2 to:
           - FMCW RD map  → classical FMCW detections.
           - OTFS DD map  → classical OTFS detections.
      3) Run the trained UNetLite on the FMCW RD map and convert its
         heatmap output into a detection list.
      4) Compute CFAR vs DL metrics and save a BEV comparison figure.

    Communications (BER curves):
      1) If baseline BER arrays are not provided, load them from
         `comm_baseline_ber.npz` or recompute via `run_ber_sweep_and_plot`.
         These remain *AWGN baselines* (no geometry).
      2) Evaluate trained OFDM/OTFS DL demappers on the same Eb/N0 grid,
         but using the **geometry-based channel generators**:
             - `comm_dl_gen_batch_OFDM_geom`
             - `comm_dl_gen_batch_OTFS_geom`
         which internally:
             - call `raycast_torch` to build a multipath channel;
             - insert pilot symbols;
             - perform LS channel estimation + equalization;
             - output feature grids X and bit labels Y.
      3) Plot baseline vs DL BER and compute simple metrics such as
         “SNR at target BER”.

    Parameters
    ----------
    out_dir : str or Path
        Root output directory for this experiment (same as used in Part 1/2).

    sp : SystemParams
        Global ISAC system configuration (ranges, Doppler, sampling, etc.).

    radar_net : nn.Module
        Trained UNetLite radar detector.

    ofdm_model : nn.Module
        Trained OFDM demapper CNN.

    otfs_model : nn.Module
        Trained OTFS demapper CNN.

    sample_split : {"train","val"}
        Which radar split to pull the evaluation sample from.

    sample_idx : int
        Index into the sorted NPZ list under radar/{split} to evaluate.

    eb_axis, ber_ofdm_base, ber_otfs_base, ber_theory : np.ndarray or None
        Optional pre-computed comm baseline arrays. If any is None, they
        are loaded from `comm_baseline_ber.npz` under `out_dir`, or
        recomputed via `run_ber_sweep_and_plot`.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 1) RADAR: load one NPZ sample (FMCW + OTFS) from disk
    # =========================================================
    radar_dir = out / "radar" / sample_split
    files = sorted(radar_dir.glob("*.npz"))
    if not files:
        print(f"[PART 4] No radar files found under {radar_dir}")
        return

    sample_idx = max(0, min(sample_idx, len(files) - 1))
    fpath = files[sample_idx]
    data = np.load(fpath, allow_pickle=True)

    # Saved by simulate_dataset():
    #   rd_f_db : FMCW RD map  (M, N//2) [dB]
    #   rd_o_db : OTFS DD map  (M, N)    [dB]
    #   gts     : JSON string of GT boxes
    rd_f_db = data["rd_f_db"]
    rd_o_db = data["rd_o_db"]
    gts = json.loads(str(data["gts"])) if "gts" in data else []

    print(f"[PART 4] Evaluating DL vs classical on sample "
          f"{sample_idx} ({fpath.name}) from split='{sample_split}'")

    # FMCW and OTFS axes (range/velocity vs delay/Doppler)
    ra_f, va_f = sp.fmcw_axes()   # FMCW range & velocity axes
    ra_o, va_o = sp.otfs_axes()   # OTFS delay & Doppler axes

    # =========================================================
    # 2) RADAR: CFAR configurations (must match Part 2)
    # =========================================================
    cfar_fmcw_cfg = dict(
        train=(10, 8),
        guard=(2, 2),
        pfa=3e-5,
        min_snr_db=12.0,
        notch_doppler_bins=2,
        apply_nms=True,
        max_peaks=40,
    )

    cfar_otfs_cfg = dict(
        train=(12, 10),
        guard=(2, 2),
        pfa=1e-5,
        min_snr_db=10.0,
        notch_doppler_bins=0,
        apply_nms=True,
        max_peaks=40,
    )

    # =========================================================
    # 3) RADAR: CFAR vs DL on this sample
    # =========================================================
    # --- FMCW CFAR detections ---
    det_f_mask = cfar2d_ca(rd_f_db, **cfar_fmcw_cfg)
    dets_f = extract_detections(rd_f_db, det_f_mask, ra_f, va_f)
    metrics_cfar_f = compute_radar_metrics(dets_f, gts, sp, w_r=1.0, w_v=0.5)

    # --- OTFS CFAR detections ---
    det_o_mask = cfar2d_ca(rd_o_db, **cfar_otfs_cfg)
    dets_o = extract_detections(rd_o_db, det_o_mask, ra_o, va_o)
    metrics_cfar_o = compute_radar_metrics(dets_o, gts, sp, w_r=1.0, w_v=0.5)

    # --- DL radar detector on FMCW RD ---
    rd_norm = _rd_normalize(rd_f_db)                 # (M, N//2), float32
    rd_in = torch.from_numpy(rd_norm)[None, None].to(DEVICE)  # (1,1,M,N//2)

    radar_net.eval()
    with torch.no_grad():
        logits = radar_net(rd_in)                    # (1,1,M,N//2) or (1,C,M,N)
    dets_dl = rd_dl_infer_to_points(
        logits, ra_f, va_f, thr=0.40, max_peaks=32
    )
    metrics_dl_f = compute_radar_metrics(dets_dl, gts, sp, w_r=1.0, w_v=0.5)

    print("[PART 4][Radar] CFAR metrics (FMCW):", metrics_cfar_f)
    print("[PART 4][Radar] CFAR metrics (OTFS):", metrics_cfar_o)
    print("[PART 4][Radar] DL   metrics (FMCW):", metrics_dl_f)

    # --- Optional: BEV comparison figure for this sample ---
    viz_scene_bev_compare(
        out / f"scene_bev_compare_sample{sample_idx:03d}.pdf",
        dets_f,
        dets_o,
        gts,
        sp,
    )

    # =========================================================
    # 4) COMM: baseline BER (AWGN) – load or recompute
    # =========================================================
    if eb_axis is None or ber_ofdm_base is None or ber_otfs_base is None:
        ber_npz = out / "comm_baseline_ber.npz"
        if ber_npz.exists():
            arrs = np.load(ber_npz)
            eb_axis = arrs["eb_axis"]
            ber_ofdm_base = arrs["ber_ofdm"]
            ber_otfs_base = arrs["ber_otfs"]
            ber_theory = arrs["ber_theory"]
            print("[PART 4] Loaded baseline BER arrays from comm_baseline_ber.npz")
        else:
            print("[PART 4] Baseline BER not found – recomputing quickly (AWGN)...")
            eb_axis, ber_ofdm_base, ber_otfs_base, ber_theory = run_ber_sweep_and_plot(
                out / "ber_compare_baseline.pdf",
                ebn0_db_list=list(range(0, 21, 2)),
                ofdm_cfg=dict(Nfft=256, cp_len=32, n_ofdm_sym=400),
                otfs_cfg=dict(M=64, N=256, cp_len=32),
                rng_seed=2025,
            )
            # Save for reuse
            np.savez(
                ber_npz,
                eb_axis=eb_axis,
                ber_ofdm=ber_ofdm_base,
                ber_otfs=ber_otfs_base,
                ber_theory=ber_theory,
            )
            print("[PART 4] Baseline BER arrays saved to comm_baseline_ber.npz")

    # Make sure Eb/N0 axis is a numpy array
    eb_axis = np.asarray(eb_axis, dtype=np.float32)

    # =========================================================
    # 5) COMM: DL demappers vs baselines
    #     (now using geometry-based channels + pilots)
    # =========================================================
    print("[PART 4][Comm] Evaluating DL demappers with geometry-based channels...")

    # OFDM geometry-based configuration:
    #   - sp   : SystemParams for raycast + channel construction
    #   - batch: mini-batch size per Eb/N0 during BER evaluation
    #   - Nfft : OFDM FFT size
    #   - cp_len: CP length
    #   - n_sym: # OFDM symbols per frame (including pilot symbol)
    ofdm_cfg_geom = dict(
        sp=sp,
        batch=8,
        Nfft=256,
        cp_len=32,
        n_sym=8,
    )

    # OTFS geometry-based configuration:
    #   - M, N : Doppler × delay DD grid size
    #   - sp   : SystemParams for raycast + DD channel build
    otfs_cfg_geom = dict(
        sp=sp,
        batch=6,
        M=64,
        N=256,
    )

    # --- OFDM DL demapper BER with geometry-based channel ---
    ber_ofdm_dl = comm_demap_ber_curve(
        ofdm_model,
        comm_dl_gen_batch_OFDM_geom,
        ofdm_cfg_geom,
        eb_axis,
    )

    # --- OTFS DL demapper BER with geometry-based channel ---
    ber_otfs_dl = comm_demap_ber_curve(
        otfs_model,
        comm_dl_gen_batch_OTFS_geom,
        otfs_cfg_geom,
        eb_axis,
    )

    # Combined BER comparison plot:
    #   - Baselines: AWGN-based from run_ber_sweep_and_plot (Part 2).
    #   - DL curves: geometry-based channel evaluation.
    viz_ber_compare_with_dl(
        out / "ber_compare_with_dl.pdf",
        eb_axis,
        ber_ofdm_base,
        ber_otfs_base,
        ber_theory,
        ber_ofdm_dl=ber_ofdm_dl,
        ber_otfs_dl=ber_otfs_dl,
        title="Comm BER: Baseline (AWGN) vs DL (Geom Channel)",
    )
    print("[COMM] BER comparison saved to", out / "ber_compare_with_dl.pdf")

    # =========================================================
    # 6) COMM: simple metrics – "SNR at target BER"
    # =========================================================
    metrics_comm_ofdm = compute_comm_metrics(
        eb_axis, ber_ofdm_base, ber_dl=ber_ofdm_dl, target_ber=1e-3
    )
    metrics_comm_otfs = compute_comm_metrics(
        eb_axis, ber_otfs_base, ber_dl=ber_otfs_dl, target_ber=1e-3
    )
    print("[PART 4][Comm] OFDM metrics:", metrics_comm_ofdm)
    print("[PART 4][Comm] OTFS metrics:", metrics_comm_otfs)

    # =========================================================
    # 7) Save summary JSON (radar + comm metrics)
    # =========================================================
    with open(out / "radar_metrics_dl_vs_cfar.json", "w") as f:
        json.dump(
            {
                "cfar_fmcw": metrics_cfar_f,
                "cfar_otfs": metrics_cfar_o,
                "dl_fmcw": metrics_dl_f,
                "sample_index": int(sample_idx),
                "sample_split": sample_split,
            },
            f,
            indent=2,
        )

    with open(out / "comm_metrics_dl_vs_cfar.json", "w") as f:
        json.dump(
            {
                "ofdm": metrics_comm_ofdm,
                "otfs": metrics_comm_otfs,
                "eb_axis": list(map(float, eb_axis)),
            },
            f,
            indent=2,
        )

    print("[PART 4] Evaluation complete. Metrics saved.\n")
    
# ---------------------------------------------------------------------
# TOP-LEVEL PIPELINE (Parts 1–4)
# ---------------------------------------------------------------------
from isac_utils import comm_dl_gen_batch_OFDM_geom, comm_dl_gen_batch_OTFS_geom
def run_isac_experiment(out_dir="./output/isac_main1"):
    """
    Main pipeline tying together the 4 parts:

      1) Dataset simulation (+ quick visualization)
      2) Traditional baselines (radar & comm) from disk dataset
      3) Deep models (radar UNet, comm demappers)
      4) DL vs traditional evaluation (single sample + BER curves)

    Adjust:
      - root paths / checkpoint layout
      - SystemParams
      - training hyperparameters
      - CFAR thresholds & Eb/N0 grids
    from here.
    """
    import numpy as np

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    ckpts = root / "checkpoints"
    ckpts.mkdir(exist_ok=True)

    # ---------------- Global system config ----------------
    sp = SystemParams()

    # =====================================================
    # Part 1: Dataset simulation + visualization
    # =====================================================
    print("[PART 1] Simulating disk dataset (if missing)...")
    simulate_if_missing(
        root,
        sp,
        n_train=1500,
        n_val=300,
        seed=2025,
        snr_list=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20),
    )

    print("[PART 1] Visualizing example training samples...")
    visualize_radar_dataset_examples(root, sp, n_examples=4, split="train")
    print("[PART 1] Visualizing example validation samples...")
    visualize_radar_dataset_examples(root, sp, n_examples=2, split="val")
    print("[PART 1] Done.\n")

    # =====================================================
    # Part 2: Traditional radar & comm baselines (from disk)
    # =====================================================
    metrics_f_trad, metrics_o_trad, eb_axis, ber_ofdm_base, ber_otfs_base, ber_theory = (
        run_traditional_baselines_from_dataset(
            root,
            sp,
            split="val",
            n_vis_samples=4,
            max_samples=None,
        )
    )

    # =====================================================
    # Part 3: Train deep radar model (UNetLite)
    # =====================================================
    print("[PART 3] Training radar UNetLite detector...")
    radar_net_fmcw = train_radar_model(
        sp,
        data_root=root,
        radar_mode="fmcw",
        epochs=6,
        batch=6,
        lr=1e-3,
    )
    # Save radar checkpoints
    torch.save(
        {
            "epoch": 0,
            "model": radar_net_fmcw.state_dict(),
            "optim": None,
            "val_loss": 0.0,
        },
        ckpts / "radar_fmcw.pt",
    )
    torch.save(radar_net_fmcw.state_dict(), ckpts / "radar_fmcw_best_only.pt")
    
    radar_net_otfs = train_radar_model(
        sp,
        data_root=root,
        radar_mode="otfs",
        epochs=6,
        batch=6,
        lr=1e-3,
    )
    # Save radar checkpoints
    torch.save(
        {
            "epoch": 0,
            "model": radar_net_otfs.state_dict(),
            "optim": None,
            "val_loss": 0.0,
        },
        ckpts / "radar_unet.pt",
    )
    torch.save(radar_net_otfs.state_dict(), ckpts / "radar_otfs_best_only.pt")

    # =====================================================
    # Part 3: Train deep comm demappers (OFDM & OTFS)
    # =====================================================
    print("Training OFDM demapper DL (geometry-based)...")
    ofdm_cfg_geom = dict(
        sp=sp,
        batch=8,
        Nfft=256,
        cp_len=32,
        n_sym=8,
    )
    ofdm_model = CommDemapperCNN(in_ch=2)
    ofdm_model = train_comm_demap(
        ofdm_model,
        gen_batch_fn=comm_dl_gen_batch_OFDM_geom,
        cfg=ofdm_cfg_geom,
        snr_min=0,
        snr_max=18,
        epochs=5,
        steps_per_epoch=200,
        lr=3e-4,
        tag="OFDM-GEO",
    )

    print("Training OTFS demapper DL (geometry-based)...")
    otfs_cfg_geom = dict(
        sp=sp,
        batch=6,
        M=64,
        N=256,
    )
    otfs_model = CommDemapperCNN(in_ch=2)
    otfs_model = train_comm_demap(
        otfs_model,
        gen_batch_fn=comm_dl_gen_batch_OTFS_geom,
        cfg=otfs_cfg_geom,
        snr_min=0,
        snr_max=18,
        epochs=5,
        steps_per_epoch=200,
        lr=3e-4,
        tag="OTFS-GEO",
    )

    torch.save({"epoch": 5, "model": ofdm_model.state_dict()}, ckpts / "comm_ofdm.pt")
    torch.save({"epoch": 5, "model": otfs_model.state_dict()}, ckpts / "comm_otfs.pt")
    print("[PART 3] Done.\n")

    # =====================================================
    # Part 4: DL vs traditional evaluation
    # =====================================================
    evaluate_and_visualize(
        out_dir=root,
        sp=sp,
        radar_net=radar_net,
        ofdm_model=ofdm_model,
        otfs_model=otfs_model,
        sample_split="val",
        sample_idx=0,
        eb_axis=np.asarray(eb_axis) if eb_axis is not None else None,
        ber_ofdm_base=ber_ofdm_base,
        ber_otfs_base=ber_otfs_base,
        ber_theory=ber_theory,
    )


# ---------------------------------------------------------------------
# Optional hooks for future extensions
# ---------------------------------------------------------------------
def run_validation_from_root(
    root,
    dl_thr_sweep=(0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    cfar_pfa_sweep=(1e-2, 1e-3, 1e-4, 1e-5),
    do_otfs=True,
    max_samples=None,
):
    """
    Placeholder hook for richer validation pipelines on disk-based datasets.

    You can plug in:
      - RadarDiskDataset + make_radar_loaders
      - sweeps over DL output thresholds vs CFAR Pfa
      - metrics logging to JSON / CSV
    """
    root = Path(root)
    sp = SystemParams()
    print("[VAL] (Placeholder) – extend run_validation_from_root() as needed.")
    print(f"      Suggested sweeps: DL thr {dl_thr_sweep}, CFAR Pfa {cfar_pfa_sweep}")
    if do_otfs:
        print("      OTFS evaluation is enabled (but not implemented yet here).")


def launch_mdmt_training():
    """
    Entry point reserved for future multi-domain / multi-task training
    experiments.

    For now, it simply calls run_isac_experiment().
    """
    run_isac_experiment()


if __name__ == "__main__":
    # Default entry point: full ISAC pipeline.
    run_isac_experiment()