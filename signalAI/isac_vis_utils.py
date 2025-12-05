from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
import torch

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt
import numpy as np


def visualize_scene_3d_matplotlib(
    pts,
    labels=None,
    gts=None,
    sensor_pos=(0.0, 0.0, 1.8),
    ground_z=0.0,
    subsample_hits=10,
    figsize=(8, 8),
    save_path=None,
):
    """
    3D visualisation of a single raycast scene.

    Parameters
    ----------
    pts : (N, 3) array-like
        Ray intersection points in world coordinates.  This is typically the
        `pts` returned by `raycast_torch` (converted to numpy).
    labels : (N,) array-like of int, optional
        Integer label for each hit point:
          - 0 = ground
          - 1..K = object index  (matches GT list order if you used that scheme)
        If None, all hits are drawn with a single colour.
    gts : list of dict, optional
        Ground-truth boxes. Each element:
          {"c": [x,y,z], "s": [sx,sy,sz], "v": [vx,vy,vz]}.
        Only "c" and "s" are used for drawing wireframe boxes.
    sensor_pos : 3-tuple
        (x, y, z) of the radar / lidar sensor.
    ground_z : float
        Height of the ground plane (default 0.0).
    subsample_hits : int
        Subsample factor for ray segments.  If >1, only every k-th hit
        point is used when drawing beams from sensor to hits.
    figsize : 2-tuple
        Matplotlib figure size.
    save_path : str or Path, optional
        If provided, figure is saved instead of shown.

    Notes
    -----
    - This is purely a visualisation helper; it does NOT modify any data.
    - It is agnostic to radar vs lidar; we just assume "sensor at sensor_pos"
      and hits in `pts`.
    """
    pts = np.asarray(pts)
    assert pts.ndim == 2 and pts.shape[1] == 3, "pts must be (N,3)"

    if labels is not None:
        labels = np.asarray(labels)
        assert labels.shape[0] == pts.shape[0], "labels must match pts length"

    sensor_pos = np.asarray(sensor_pos, dtype=float)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # ------------- draw ground plane ----------------------------------
    # A simple rectangular patch around the sensor and hits.
    if pts.size > 0:
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    else:
        x_min, x_max = -10.0, 10.0
        y_min, y_max = -10.0, 10.0

    Xg, Yg = np.meshgrid(
        np.linspace(x_min, x_max, 10),
        np.linspace(y_min, y_max, 10),
    )
    Zg = np.full_like(Xg, ground_z)
    ax.plot_surface(
        Xg, Yg, Zg,
        alpha=0.15,
        color="lightgray",
        edgecolor="none",
    )

    # ------------- draw sensor position -------------------------------
    ax.scatter(
        [sensor_pos[0]],
        [sensor_pos[1]],
        [sensor_pos[2]],
        marker="*",
        s=120,
        c="k",
        label=f"Sensor (H={sensor_pos[2]:.2f} m)",
    )

    # vertical line to ground to emphasise height
    ax.plot(
        [sensor_pos[0], sensor_pos[0]],
        [sensor_pos[1], sensor_pos[1]],
        [ground_z, sensor_pos[2]],
        "k--",
        linewidth=1.0,
    )

    # ------------- draw GT boxes (wireframes) -------------------------
    if gts:
        for i, gt in enumerate(gts, start=1):
            c = np.asarray(gt["c"], dtype=float)
            s = np.asarray(gt["s"], dtype=float)
            dx, dy, dz = s / 2.0
            # 8 corners of the axis-aligned box
            corners = np.array(
                [
                    [c[0] - dx, c[1] - dy, c[2] - dz],
                    [c[0] + dx, c[1] - dy, c[2] - dz],
                    [c[0] - dx, c[1] + dy, c[2] - dz],
                    [c[0] + dx, c[1] + dy, c[2] - dz],
                    [c[0] - dx, c[1] - dy, c[2] + dz],
                    [c[0] + dx, c[1] - dy, c[2] + dz],
                    [c[0] - dx, c[1] + dy, c[2] + dz],
                    [c[0] + dx, c[1] + dy, c[2] + dz],
                ]
            )

            # edges as pairs of corner indices
            edges_idx = [
                (0, 1), (0, 2), (0, 4),
                (7, 6), (7, 5), (7, 3),
                (2, 3), (2, 6),
                (1, 3), (1, 5),
                (4, 5), (4, 6),
            ]
            edges = [(corners[i0], corners[i1]) for i0, i1 in edges_idx]
            lc = Line3DCollection(
                edges,
                colors="r",
                linewidths=1.5,
                alpha=0.9,
            )
            ax.add_collection3d(lc)
            ax.text(
                c[0],
                c[1],
                c[2] + dz + 0.1,
                f"{i}",
                color="red",
                fontsize=9,
            )

    # ------------- scatter hits by label ------------------------------
    if pts.size > 0:
        if labels is None:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=2,
                c="tab:blue",
                alpha=0.4,
                label="Hits",
            )
        else:
            # ground hits
            mask_g = labels == 0
            if np.any(mask_g):
                ax.scatter(
                    pts[mask_g, 0],
                    pts[mask_g, 1],
                    pts[mask_g, 2],
                    s=2,
                    c="tab:brown",
                    alpha=0.4,
                    label="Ground hits",
                )
            # object hits (each object id > 0)
            obj_ids = np.unique(labels[labels > 0])
            for oid in obj_ids:
                m = labels == oid
                ax.scatter(
                    pts[m, 0],
                    pts[m, 1],
                    pts[m, 2],
                    s=3,
                    alpha=0.6,
                    label=f"Object {oid} hits",
                )

    # ------------- optional ray segments from sensor ------------------
    if pts.size > 0:
        step = max(1, int(subsample_hits))
        P_sub = pts[::step]
        segments = []
        for p in P_sub:
            segments.append([sensor_pos, p])
        if segments:
            segs = Line3DCollection(
                segments,
                colors="tab:gray",
                linewidths=0.5,
                alpha=0.4,
            )
            ax.add_collection3d(segs)

    # ------------- axes & view ----------------------------------------
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Raycast Scene 3D")

    # nice z-limits: include sensor and boxes, small margin
    z_min = min(ground_z - 0.5, pts[:, 2].min() - 0.5 if pts.size else ground_z - 0.5)
    z_max = max(sensor_pos[2] + 0.5, pts[:, 2].max() + 0.5 if pts.size else sensor_pos[2] + 0.5)
    ax.set_zlim(z_min, z_max)

    ax.legend(loc="upper right", fontsize=8)
    ax.view_init(elev=30, azim=-120)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def export_scene_to_ply_open3d(
    pts,
    labels=None,
    its=None,
    save_path="scene.ply",
    ground_color=(160, 110, 60),  # RGB for ground, 0–255
    cube_colors=None,
    default_color=(80, 160, 255),
):
    """
    Export a raycast scene as a PLY point cloud using Open3D.

    This is *optional* and only works if open3d is installed.

    Parameters
    ----------
    pts : (N, 3) torch.Tensor or np.ndarray
        Hit positions in world coordinates [m] from `raycast_torch`.

    labels : (N,) torch.Tensor or np.ndarray, optional
        Object labels (0=ground, 1..G=cubes). If None, all points share
        `default_color`.

    its : (N,) torch.Tensor or np.ndarray, optional
        Intensities (0..255). If provided, can be mapped into brightness.

    save_path : str
        PLY output path, e.g., "scene.ply".

    ground_color : (3,) tuple[int]
        Base RGB color for ground points (0..255).

    cube_colors : dict[int, tuple[int,int,int]], optional
        Mapping from object-id → RGB. If None, a small color palette will
        be generated automatically.

    default_color : (3,) tuple[int]
        Color for points when labels are None.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[WARN] open3d is not installed. `export_scene_to_ply_open3d` skipped.")
        return

    # Convert tensors -> numpy
    if isinstance(pts, torch.Tensor):
        pts_np = pts.detach().cpu().numpy()
    else:
        pts_np = np.asarray(pts)

    N = pts_np.shape[0]
    if N == 0:
        print("[PLY] No points to export.")
        return

    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)
        assert labels_np.shape[0] == N
    else:
        labels_np = None

    if its is not None:
        if isinstance(its, torch.Tensor):
            its_np = its.detach().cpu().numpy()
        else:
            its_np = np.asarray(its)
        assert its_np.shape[0] == N
        # Normalize [0,255] → [0,1]
        its_norm = np.clip(its_np, 0, 255) / 255.0
    else:
        its_norm = None

    # Colors: (N,3) in [0,1]
    colors = np.zeros((N, 3), dtype=np.float32)

    if labels_np is None:
        # All points same color (scaled by intensity if provided)
        base = np.array(default_color, dtype=np.float32) / 255.0
        colors[:] = base
        if its_norm is not None:
            colors *= its_norm[:, None]
    else:
        # Prepare cube color palette if not given
        if cube_colors is None:
            # Simple tab10 palette for up to 10 objects
            cmap = plt.cm.get_cmap("tab10", 10)
            cube_colors = {}
            for i in range(1, 11):
                rgb = np.array(cmap(i - 1)[:3]) * 255.0
                cube_colors[i] = tuple(int(v) for v in rgb)

        ground_rgb = np.array(ground_color, dtype=np.float32) / 255.0

        for i in range(N):
            lbl = labels_np[i]
            if lbl == 0:  # ground
                base = ground_rgb
            elif lbl in cube_colors:
                base = np.array(cube_colors[lbl], dtype=np.float32) / 255.0
            else:
                base = np.array(default_color, dtype=np.float32) / 255.0

            if its_norm is not None:
                colors[i] = base * its_norm[i]
            else:
                colors[i] = base

    # Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(str(save_path), pcd, write_ascii=True)
    print(f"[PLY] Scene exported to {save_path}")


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ================================================================
# Helper: generic RD / DD plotting
# ================================================================
def plot_rd_generic(
    ax,
    rd_db,
    x_axis,
    y_axis,
    title,
    dynamic_db=35,
    percentile_clip=99.2,
    cmap="magma",
    xlabel="Range / Delay",
    ylabel="Velocity / Doppler",
):
    """
    Generic helper to visualize a 2D radar map (Range–Doppler or Delay–Doppler).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the image.
    rd_db : (H, W) array-like
        Magnitude in dB (e.g., FMCW RD or OTFS DD map).
    x_axis : (W,) array-like
        Horizontal axis values (e.g., range in meters).
    y_axis : (H,) array-like
        Vertical axis values (e.g., velocity in m/s).
    title : str
        Plot title.
    dynamic_db : float
        Dynamic range in dB to display (top - dynamic_db).
    percentile_clip : float
        Percentile used to define the "top" dB level for dynamic range.
    cmap : str
        Matplotlib colormap name.
    xlabel : str
        Label for horizontal axis.
    ylabel : str
        Label for vertical axis.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The image handle (you can use it for colorbars, etc.).
    """
    rd_db = np.asarray(rd_db)
    x_axis = np.asarray(x_axis)
    y_axis = np.asarray(y_axis)

    # Use a high percentile as "top" and show only dynamic_db dB below it
    top = np.percentile(rd_db, percentile_clip)
    vmin = top - dynamic_db

    im = ax.imshow(
        rd_db,
        extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=top,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


# ================================================================
# Helper: project GT boxes to (R, v_r) radial coordinates
# ================================================================
def project_gts_to_rv(gts, radar_pos=(0.0, 0.0, 0.0)):
    """
    Convert ground-truth 3D boxes into (range, radial velocity) pairs.

    This is purely geometric and does not depend on any SystemParams.

    Parameters
    ----------
    gts : list of dict
        Each GT has at least:
          - 'c': [x, y, z]  center position in meters
          - 'v': [vx, vy, vz] velocity vector in m/s
        (Other keys such as 's' can exist but are ignored here.)
    radar_pos : tuple of 3 floats
        Radar phase-center position in world coordinates (x, y, z) [m].

    Returns
    -------
    rv_list : list of dict
        Each entry:
          - 'r': float   range [m]
          - 'v': float   radial velocity [m/s]
    """
    radar_pos = np.asarray(radar_pos, dtype=float)
    rv_list = []

    for gt in gts:
        c = np.asarray(gt["c"], dtype=float)
        v = np.asarray(gt["v"], dtype=float)

        # Vector from radar to target center
        d = c - radar_pos
        r = np.linalg.norm(d)

        # Skip degenerate range
        if r < 1e-6:
            continue

        # Unit line-of-sight vector
        u = d / r

        # Radial velocity: projection of target velocity onto LOS
        vr = float(np.dot(u, v))

        rv_list.append({"r": float(r), "v": vr})

    return rv_list


# ================================================================
# 1) FMCW visualization from extra_fmcw dict
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def build_fmcw_H_gt_from_scatters(
    R_tensor,
    vr_tensor,
    amp_tensor,
    ranges_axis,
    vels_axis,
    rd_shape=None,
    amp_threshold=1e-6,
):
    """
    Build an analytic ground-truth FMCW Range–Doppler grid H_gt from
    per-scatterer geometry:

        - R_tensor   : (P,) torch.Tensor of per-scatterer ranges [m].
        - vr_tensor  : (P,) torch.Tensor of per-scatterer radial velocities [m/s].
        - amp_tensor : (P,) torch.Tensor of (real) amplitudes.
        - ranges_axis: (K,) numpy array, RD range axis [m].
        - vels_axis  : (M,) numpy array, RD velocity axis [m/s].
        - rd_shape   : optional (M,K); if None, inferred from axes.

    Strategy:
      * For each scatterer with amp > amp_threshold:
          - Find nearest range bin k and velocity bin m.
          - Accumulate its complex gain into H_gt[m, k].
      * This produces a sparse RD map with impulses at the “true”
        bins of each scatterer.

    This is conceptually similar to what you did for OTFS H_gt using taps.
    """

    import numpy as np
    import torch

    # Convert to numpy
    R = R_tensor.detach().cpu().numpy().ravel()
    vr = vr_tensor.detach().cpu().numpy().ravel()
    amp = amp_tensor.detach().cpu().numpy().ravel()

    ranges_axis = np.asarray(ranges_axis)
    vels_axis   = np.asarray(vels_axis)

    if rd_shape is None:
        M = vels_axis.shape[0]
        K = ranges_axis.shape[0]
    else:
        M, K = rd_shape

    H_gt = np.zeros((M, K), dtype=np.complex64)

    for r_i, vr_i, a_i in zip(R, vr, amp):
        # Skip effectively zero-amplitude scatters
        if abs(a_i) < amp_threshold:
            continue

        # Nearest range/Doppler bins
        k = int(np.argmin(np.abs(ranges_axis - r_i)))
        m = int(np.argmin(np.abs(vels_axis   - vr_i)))

        if 0 <= m < M and 0 <= k < K:
            # Add real amplitude; if you want a phase model, you can
            # make it complex, e.g., a_i * np.exp(1j * phi_i).
            H_gt[m, k] += np.complex64(a_i)

    return H_gt

def viz_fmcw_extras(
    extra_fmcw,
    out_prefix,
    fs=None,
    ra_axis=None,
    va_axis=None,
    radar_pos=(0.0, 0.0, 0.0),
    gts=None,
    show=False,
):
    """
    Visualize intermediate FMCW radar objects stored in `extra_fmcw`.

    This version is aligned with fmcw_torch(..., return_extra=True):

        extra_fmcw = {
            "iq"     : (M,N) complex64   # IQ cube (after MTI/window)
            "RD"     : (M,K) complex64   # complex RD map (K=N//2)
            "H_gt"   : (M,K) complex64   # ground-truth RD grid (optional)
            "ranges" : (K,) float        # range axis [m]
            "vels"   : (M,) float        # velocity axis [m/s]
            "R"      : (P,) torch.Tensor # per-scatterer ranges [m]
            "vr"     : (P,) torch.Tensor # per-scatterer radial vel [m/s]
            "amp"    : (P,) torch.Tensor # per-scatterer amplitude
            "gts"    : list of GT dicts (optional)
            # optional legacy keys: 'tx_chirp', 't_fast', 'iq_raw', 'iq_mti',
            # 'iq_win', 'RD_cplx', 'rd_db'
        }

    Key ideas:
      - Each image is normalized per-map to avoid "flat" plots.
      - If H_gt is present but all zeros, we reconstruct a sparse
        ground-truth RD map from (R, vr, amp, ranges, vels) so you can
        see where the analytic peaks should live.

    Physically:
      - Measured RD map: full 2D FFT of IQ, including windowing, MTI,
        clutter, sidelobes, etc.
      - Ground-truth RD map: very sparse impulses at the nearest
        (range, Doppler) bins for each simulated scatterer.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------
    # 0) Choose GT list if not explicitly given
    # ------------------------------------------------------
    if gts is None and "gts" in extra_fmcw and extra_fmcw["gts"] is not None:
        gts = extra_fmcw["gts"]

    # ------------------------------------------------------
    # Helper: project GT boxes to (range, radial velocity)
    # ------------------------------------------------------
    def _project_gts_to_rv(gts_list, radar_pos_local):
        """
        Map 3D GT boxes to (range, radial velocity) for overlay on RD maps.
        """
        rv_list = []
        if not gts_list:
            return rv_list
        radar_pos_local = np.asarray(radar_pos_local, dtype=float)
        for gt in gts_list:
            c = np.asarray(gt["c"], dtype=float)
            v = np.asarray(gt["v"], dtype=float)
            d = c - radar_pos_local
            r = float(np.linalg.norm(d))
            if r < 1e-6:
                continue
            u = d / r
            vr = float(np.dot(u, v))
            rv_list.append({"r": r, "v": vr})
        return rv_list

    # ------------------------------------------------------
    # Helper: choose range / velocity axes
    # ------------------------------------------------------
    def _get_axes(extra, ra_axis_user, va_axis_user, M, K):
        """
        Decide which range / velocity axes to use for plotting.

        Preference:
          1) User-provided ra_axis / va_axis.
          2) extra["ranges"] / extra["vels"].
          3) Simple indices.
        """
        # X-axis (range)
        if ra_axis_user is not None:
            ra = np.asarray(ra_axis_user)
            xlab = "Range [m]"
        elif "ranges" in extra:
            ra = np.asarray(extra["ranges"])
            xlab = "Range [m]"
        else:
            ra = np.arange(K)
            xlab = "Range bin"

        # Y-axis (velocity)
        if va_axis_user is not None:
            va = np.asarray(va_axis_user)
            ylab = "Radial velocity [m/s]"
        elif "vels" in extra:
            va = np.asarray(extra["vels"])
            ylab = "Radial velocity [m/s]"
        else:
            va = np.arange(M)
            ylab = "Doppler bin"

        return ra, va, xlab, ylab

    # ------------------------------------------------------
    # Helper: generic RD map plotter (normalized dB)
    # ------------------------------------------------------
    def _plot_rd_map(rd_complex_or_db, ra, va, xlab, ylab, title, fname_suffix):
        """
        Plot a Range–Doppler heatmap with per-map normalization:

          - If input is complex, use magnitude, convert to dB.
          - Shift so map max = 0 dB.
          - Clamp color scale to [-60, 0] dB for good contrast.

        Visually, you should see bright peaks at target bins and
        sidelobes / noise elsewhere.
        """
        arr = np.asarray(rd_complex_or_db)
        if np.iscomplexobj(arr):
            mag = np.abs(arr)
            db = 20.0 * np.log10(mag + 1e-12)
        else:
            db = arr.astype(float)

        M, K = db.shape
        max_val = float(np.max(db))
        min_val = float(np.min(db))
        print(f"[viz_fmcw_extras] {fname_suffix} dB range: [{min_val:.2f}, {max_val:.2f}]")

        db_norm = db - max_val  # max → 0 dB
        vmin = -60.0
        vmax = 0.0

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(
            db_norm,
            origin="lower",
            aspect="auto",
            extent=[ra[0], ra[-1], va[0], va[-1]],
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
        )
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Power [dB] (relative)")
        ax.grid(alpha=0.25, linestyle=":")

        plt.tight_layout()
        plt.savefig(out_prefix.with_name(out_prefix.name + fname_suffix), dpi=170)
        plt.close(fig)

    # ------------------------------------------------------
    # 1) Transmit chirp (if available)
    # ------------------------------------------------------
    if "tx_chirp" in extra_fmcw:
        tx = np.asarray(extra_fmcw["tx_chirp"])  # (N,)
        N = tx.shape[0]

        if "t_fast" in extra_fmcw:
            t_fast = np.asarray(extra_fmcw["t_fast"])
        elif fs is not None:
            t_fast = np.arange(N) / float(fs)
        else:
            t_fast = np.arange(N)

        fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axs[0].plot(t_fast, tx.real, label="Re{tx}")
        axs[0].plot(t_fast, tx.imag, label="Im{tx}", alpha=0.7)
        axs[0].set_ylabel("Amplitude")
        axs[0].set_title("FMCW transmit chirp (time domain)")
        axs[0].legend()
        axs[0].grid(alpha=0.3, linestyle=":")

        axs[1].plot(t_fast, np.abs(tx))
        axs[1].set_ylabel("|tx|")
        axs[1].set_xlabel(
            "Fast-time [s]"
            if ("t_fast" in extra_fmcw or fs is not None)
            else "Fast-time sample index"
        )
        axs[1].grid(alpha=0.3, linestyle=":")

        plt.tight_layout()
        plt.savefig(out_prefix.with_name(out_prefix.name + "_fmcw_tx_time.png"), dpi=170)
        plt.close(fig)

        # Spectrum
        TXF = np.fft.fftshift(np.fft.fft(tx))
        if fs is not None:
            freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / fs))
            xlab = "Frequency [Hz]"
        else:
            freqs = np.arange(N)
            xlab = "Frequency bin"

        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        ax.plot(freqs, 20.0 * np.log10(np.abs(TXF) + 1e-12))
        ax.set_xlabel(xlab)
        ax.set_ylabel("Magnitude [dB]")
        ax.set_title("FMCW chirp spectrum")
        ax.grid(alpha=0.3, linestyle=":")
        plt.tight_layout()
        plt.savefig(out_prefix.with_name(out_prefix.name + "_fmcw_tx_spectrum.png"), dpi=170)
        plt.close(fig)

    # ------------------------------------------------------
    # 2) IQ cube(s)
    # ------------------------------------------------------
    def _viz_iq(iq, suffix, title):
        iq = np.asarray(iq)
        M, N = iq.shape

        mag = np.abs(iq)
        mag_min = float(np.min(mag))
        mag_max = float(np.max(mag))
        print(f"[viz_fmcw_extras] {suffix} |IQ| range: [{mag_min:.3e}, {mag_max:.3e}]")

        if mag_max > mag_min:
            mag_norm = (mag - mag_min) / (mag_max - mag_min)
        else:
            mag_norm = np.zeros_like(mag)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(
            mag_norm,
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        ax.set_xlabel("Fast-time n")
        ax.set_ylabel("Slow-time m (chirp index)")
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Normalized |IQ|")
        ax.grid(alpha=0.25, linestyle=":")
        plt.tight_layout()
        plt.savefig(out_prefix.with_name(out_prefix.name + suffix), dpi=170)
        plt.close(fig)

    if "iq_raw" in extra_fmcw:
        _viz_iq(extra_fmcw["iq_raw"], "_fmcw_iq_raw.png", "FMCW IQ cube (raw)")
    if "iq_mti" in extra_fmcw:
        _viz_iq(extra_fmcw["iq_mti"], "_fmcw_iq_mti.png", "FMCW IQ cube (after MTI)")
    if "iq_win" in extra_fmcw:
        _viz_iq(extra_fmcw["iq_win"], "_fmcw_iq_win.png", "FMCW IQ cube (after window)")
    if "iq" in extra_fmcw:
        _viz_iq(extra_fmcw["iq"], "_fmcw_iq.png", "FMCW IQ cube (main)")

    # ------------------------------------------------------
    # 3) Measured RD map
    # ------------------------------------------------------
    rd_complex = None
    rd_db = None

    if "RD" in extra_fmcw:
        rd_complex = np.asarray(extra_fmcw["RD"])
    elif "RD_cplx" in extra_fmcw:
        rd_complex = np.asarray(extra_fmcw["RD_cplx"])
    elif "rd_db" in extra_fmcw:
        rd_db = np.asarray(extra_fmcw["rd_db"])

    if rd_complex is not None:
        M, K = rd_complex.shape
        ra, va, xlab, ylab = _get_axes(extra_fmcw, ra_axis, va_axis, M, K)
        _plot_rd_map(
            rd_complex,
            ra,
            va,
            xlab,
            ylab,
            title="FMCW Range–Doppler (measured, normalized)",
            fname_suffix="_fmcw_rd_measured.png",
        )
    elif rd_db is not None:
        M, K = rd_db.shape
        ra, va, xlab, ylab = _get_axes(extra_fmcw, ra_axis, va_axis, M, K)
        _plot_rd_map(
            rd_db,
            ra,
            va,
            xlab,
            ylab,
            title="FMCW Range–Doppler (measured, from rd_db)",
            fname_suffix="_fmcw_rd_measured.png",
        )

    # ------------------------------------------------------
    # 4) RD ground-truth (H_gt) – with fallback from scatters
    # ------------------------------------------------------
    if "H_gt" in extra_fmcw:
        H = np.asarray(extra_fmcw["H_gt"])
        M2, K2 = H.shape

        # Decide axes for GT map
        ra_gt, va_gt, xlab_gt, ylab_gt = _get_axes(extra_fmcw, ra_axis, va_axis, M2, K2)

        # If H_gt is all zeros but we have scatter info, rebuild H_gt
        if not np.any(H) and all(k in extra_fmcw for k in ("R", "vr", "amp")):
            print("[viz_fmcw_extras] H_gt is all zeros; rebuilding from (R, vr, amp).")
            H = build_fmcw_H_gt_from_scatters(
                extra_fmcw["R"],
                extra_fmcw["vr"],
                extra_fmcw["amp"],
                ranges_axis=ra_gt,
                vels_axis=va_gt,
                rd_shape=(M2, K2),
            )

        # Now plot (even if still zero; you'll know if no scatter matched)
        _plot_rd_map(
            H,
            ra_gt,
            va_gt,
            xlab_gt,
            ylab_gt,
            title="FMCW RD – ground-truth grid (H_gt, normalized)",
            fname_suffix="_fmcw_rd_gt.png",
        )

    if show:
        plt.show()

def viz_otfs_extras(
    extra_otfs,
    out_prefix,
    delay_axis=None,
    doppler_axis=None,
    radar_pos=(0.0, 0.0, 0.0),
    gts=None,
    show=False,
):
    """
    Visualize intermediate OTFS radar objects stored in `extra_otfs`.

    This is aligned with otfs_torch_full_radar(..., return_extra=True):

        extra_otfs = {
            "H_est"   : (M,N) complex64   # DD channel estimate from OTFS
            "dd_db"   : (M,N) float32     # 20*log10|H_est|
            "H_gt"    : (M,N) complex64   # analytical sparse DD ground truth
            "delays"  : (N,) float        # delay axis [s]
            "dopplers": (M,) float        # Doppler axis [Hz]
            "taps"    : list[dict]        # channel taps (n_delay, m_dopp, alpha…)
            "s_tx"    : (T,) complex64    # transmit OTFS time-domain signal
            "r_rx"    : (T,) complex64    # received time-domain signal
            "gts"     : list of GT dicts (optional)
        }

    Plots:
      1) Time-domain magnitude of s_tx and r_rx (to show pulse train & SNR).
      2) Measured DD map |H_est| in dB, normalized (should show blobs at taps).
      3) Ground-truth DD map |H_gt| in dB, normalized (clean sparse blobs).
    """

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # Helper: project GT boxes to (range, radial velocity) in DD
    # (used only for overlay markers, same as FMCW)
    # -----------------------------------------------------------
    def _project_gts_to_rv(gts_list, radar_pos_local):
        rv_list = []
        if not gts_list:
            return rv_list
        radar_pos_local = np.asarray(radar_pos_local, dtype=float)
        for gt in gts_list:
            c = np.asarray(gt["c"], dtype=float)
            v = np.asarray(gt["v"], dtype=float)
            d = c - radar_pos_local
            r = float(np.linalg.norm(d))
            if r < 1e-6:
                continue
            u = d / r
            vr = float(np.dot(u, v))
            rv_list.append({"r": r, "v": vr})
        return rv_list

    # -----------------------------------------------------------
    # 0) Optional time-domain signals s_tx, r_rx
    # -----------------------------------------------------------
    if "s_tx" in extra_otfs and "r_rx" in extra_otfs:
        s_tx = np.asarray(extra_otfs["s_tx"])
        r_rx = np.asarray(extra_otfs["r_rx"])
        T = s_tx.shape[0]
        t_axis = np.arange(T)  # sample index; you can rescale by Ts if desired

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(t_axis, np.abs(s_tx))
        axs[0].set_ylabel("|s_tx|")
        axs[0].set_title("OTFS transmit signal magnitude vs time")
        axs[0].grid(alpha=0.25, linestyle=":")

        axs[1].plot(t_axis, np.abs(r_rx))
        axs[1].set_ylabel("|r_rx|")
        axs[1].set_xlabel("Time sample index")
        axs[1].set_title("OTFS received signal magnitude vs time")
        axs[1].grid(alpha=0.25, linestyle=":")

        plt.tight_layout()
        plt.savefig(
            out_prefix.with_name(out_prefix.name + "_otfs_time_signals.png"), dpi=170
        )
        plt.close(fig)

    # -----------------------------------------------------------
    # 1) Measured DD map (H_est)
    # -----------------------------------------------------------
    if "dd_db" in extra_otfs:
        dd_db = np.asarray(extra_otfs["dd_db"])
    elif "H_est" in extra_otfs:
        H_est = np.asarray(extra_otfs["H_est"])
        dd_db = 20.0 * np.log10(np.abs(H_est) + 1e-12)
    else:
        dd_db = None

    if dd_db is not None:
        M, N = dd_db.shape

        # Axes (delay → meters if you want; here we keep generic)
        if delay_axis is not None:
            da = np.asarray(delay_axis)
            xlab = "Range / delay [m]"
        elif "delays" in extra_otfs:
            da = np.asarray(extra_otfs["delays"])
            xlab = "Delay [s]"
        else:
            da = np.arange(N)
            xlab = "Delay bin"

        if doppler_axis is not None:
            fa = np.asarray(doppler_axis)
            ylab = "Velocity / Doppler [m/s]"
        elif "dopplers" in extra_otfs:
            fa = np.asarray(extra_otfs["dopplers"])
            ylab = "Doppler [Hz]"
        else:
            fa = np.arange(M)
            ylab = "Doppler bin"

        # Normalize per-map to make peaks visible
        max_val = float(np.max(dd_db))
        min_val = float(np.min(dd_db))
        print(f"[viz_otfs_extras] DD measured dB range: [{min_val:.2f}, {max_val:.2f}]")
        dd_norm = dd_db - max_val     # max → 0 dB
        vmin = -60.0
        vmax = 0.0

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(
            dd_norm,
            origin="lower",
            aspect="auto",
            extent=[da[0], da[-1], fa[0], fa[-1]],
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
        )
        ax.set_title("OTFS delay–Doppler (measured H_est, normalized)")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Power [dB] (relative to max in this map)")

        # Optional GT overlay in (range, vr)
        if gts is not None:
            rv_list = _project_gts_to_rv(gts, radar_pos)
            for i, rv in enumerate(rv_list):
                r = rv["r"]
                v = rv["v"]
                if not (da[0] <= r <= da[-1]) or not (fa[0] <= v <= fa[-1]):
                    continue
                ax.plot(r, v, "wx", ms=8, mew=2)
                ax.text(r + 0.7, v + 0.5, f"{i}", color="white", fontsize=8)

        ax.grid(alpha=0.25, linestyle=":")
        plt.tight_layout()
        plt.savefig(
            out_prefix.with_name(out_prefix.name + "_otfs_dd_measured.png"), dpi=170
        )
        plt.close(fig)

    # -----------------------------------------------------------
    # 2) Ground-truth DD map (H_gt)
    # -----------------------------------------------------------
    if "H_gt" in extra_otfs:
        H_gt = np.asarray(extra_otfs["H_gt"])
        if np.iscomplexobj(H_gt):
            H_gt_db = 20.0 * np.log10(np.abs(H_gt) + 1e-12)
        else:
            H_gt_db = H_gt.astype(float)

        M2, N2 = H_gt_db.shape

        if delay_axis is not None:
            da = np.asarray(delay_axis)
            xlab = "Range / delay [m]"
        elif "delays" in extra_otfs:
            da = np.asarray(extra_otfs["delays"])
            xlab = "Delay [s]"
        else:
            da = np.arange(N2)
            xlab = "Delay bin"

        if doppler_axis is not None:
            fa = np.asarray(doppler_axis)
            ylab = "Velocity / Doppler [m/s]"
        elif "dopplers" in extra_otfs:
            fa = np.asarray(extra_otfs["dopplers"])
            ylab = "Doppler [Hz]"
        else:
            fa = np.arange(M2)
            ylab = "Doppler bin"

        if np.any(H_gt_db):
            max_val = float(np.max(H_gt_db))
            min_val = float(np.min(H_gt_db))
            print(f"[viz_otfs_extras] DD GT dB range: [{min_val:.2f}, {max_val:.2f}]")
            H_norm = H_gt_db - max_val
        else:
            print("[viz_otfs_extras] H_gt is all zeros.")
            H_norm = H_gt_db
        vmin = -60.0
        vmax = 0.0

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(
            H_norm,
            origin="lower",
            aspect="auto",
            extent=[da[0], da[-1], fa[0], fa[-1]],
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
        )
        ax.set_title("OTFS DD – ground-truth grid (H_gt, normalized)")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Power [dB] (relative)")
        ax.grid(alpha=0.25, linestyle=":")
        plt.tight_layout()
        plt.savefig(
            out_prefix.with_name(out_prefix.name + "_otfs_dd_gt.png"), dpi=170
        )
        plt.close(fig)

    if show:
        plt.show()

# ================================================================
# 3) Channel scatterer visualization (general)
# ================================================================
def viz_channel_scatterers(
    extra_chan,
    out_path,
    title="Channel scatterers: range vs radial velocity",
):
    """
    Visualize per-scatterer channel parameters in (range, radial velocity) space.

    This function is general-purpose and does not depend on SystemParams.

    Expected keys in extra_chan:
      - 'R_targets'   : (P,) float
          Ranges of scatterers [m].
      - 'vr_targets'  : (P,) float
          Radial velocities of scatterers [m/s].
      - 'amp_targets' : (P,) complex or float (optional)
          Complex or real amplitudes (e.g., channel gains). If provided,
          dots are color-coded by amplitude magnitude in dB.

    Parameters
    ----------
    extra_chan : dict
        Dictionary containing per-scatterer channel info. Often this is
        the same as extra_fmcw or extra_otfs (or a subset).
    out_path : str or Path
        Path to save the figure.
    title : str
        Plot title.
    """
    out_path = Path(out_path)
    R = np.asarray(extra_chan.get("R_targets", []))
    vr = np.asarray(extra_chan.get("vr_targets", []))
    amp = np.asarray(extra_chan.get("amp_targets", []))

    if R.size == 0 or vr.size == 0:
        print("[viz_channel_scatterers] No R_targets / vr_targets in extra_chan; skipping.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    if amp.size == R.size:
        # Color-code by amplitude in dB
        amp_db = 20.0 * np.log10(np.abs(amp) + 1e-12)
        sc = ax.scatter(R, vr, c=amp_db, s=40, cmap="viridis")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Amplitude [dB]")
    else:
        # No amplitude provided; simple scatter
        ax.scatter(R, vr, s=40)

    ax.set_xlabel("Range R [m]")
    ax.set_ylabel("Radial velocity v_r [m/s]")
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)