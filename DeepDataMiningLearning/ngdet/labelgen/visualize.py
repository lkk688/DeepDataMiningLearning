"""Visualization for the label generator: semantic overlay + depth colormap, per-camera tiling,
lossless PNG frames + an MP4. Reuses ngdet's `VideoWriter` (detection/verify_datasets_video.py).

Note: MP4 (H.264) mildly bands the flat mask colours (chroma subsampling); the PNG frames are
lossless — use those for figures, the MP4 for quick scrubbing.
"""
from __future__ import annotations
import numpy as np
import cv2


def sem_overlay(raw_rgb, sem, colors, sky_id, alpha=0.55):
    """raw_rgb [H,W,3] uint8, sem [H,W] ids -> blended overlay (sky untinted)."""
    color = colors[sem]
    m = (sem != sky_id)[..., None]
    return np.where(m, ((1 - alpha) * raw_rgb + alpha * color).astype(np.uint8), raw_rgb)


def _turbo(x01):
    """x in [0,1] (any shape) -> RGB uint8 via TURBO."""
    q = (np.clip(x01, 0, 1) * 255).astype(np.uint8)
    flat = cv2.cvtColor(cv2.applyColorMap(q.reshape(-1, 1), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return flat.reshape(*x01.shape, 3)


def depth_colormap(depth, uvz=None, max_depth=60.0, dim_bg=0.5):
    """depth [H,W] -> TURBO colormap. LiDAR points (uvz=[P,3]=(u,v,z)) are overlaid as 2x2 dots
    COLOURED BY THEIR OWN DEPTH (same TURBO scale) so you read the measured LiDAR depth directly
    and compare against the dense completion; the dense background is dimmed so the points pop.
    A LiDAR dot that matches its surrounding colour = accurate completion; a mismatch is visible."""
    cm = _turbo(depth / max_depth)
    if uvz is not None and len(uvz):
        cm = (cm.astype(np.float32) * dim_bg).astype(np.uint8)       # dim dense bg
        H, W = depth.shape
        u = np.round(uvz[:, 0]).astype(int); v = np.round(uvz[:, 1]).astype(int); z = uvz[:, 2]
        ok = (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
        u, v = u[ok], v[ok]
        pc = _turbo(z[ok] / max_depth)                              # per-point depth colour
        for dv in (0, 1):                                          # 2x2 dot for visibility
            for du in (0, 1):
                cm[v + dv, u + du] = pc
    return cm


def tile(panels, labels, cols=3):
    """List of [H,W,3] panels -> grid with a label per tile (contiguous for cv2.putText)."""
    panels = [np.ascontiguousarray(p, np.uint8) for p in panels]
    for p, lab in zip(panels, labels):
        cv2.putText(p, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    rows = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)]
    w = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == w else np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    return np.vstack(rows)


def write_video(path, frames, fps=2):
    """Write frames [H,W,3 RGB] to an MP4 via ngdet's VideoWriter (H.264)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from detection.verify_datasets_video import VideoWriter
    vw = VideoWriter(path, fps=fps)
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
