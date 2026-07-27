# SPDX-License-Identifier: Apache-2.0
"""Occupancy-conditioned trajectory model for AlpaSim closed-loop.

This is the driver plugin for the closed-loop occupancy->planning study. It implements
the standard :class:`BaseTrajectoryModel` contract (camera images + ego-status + command
-> (T,2) trajectory) so it plugs into AlpaSim exactly like the transfuser/alpamayo drivers.

M1 stage = STUB: a constant-velocity policy derived from the ego speed. This is NOT throwaway
code -- it is the **ego-only / no-occ ablation arm** of the experiment (the CV prior that our
open-loop study showed already matches nuScenes GT). Once it drives in closed loop, we wire the
occupancy backbone (GaussianOcc / DINOv2-LSS) + CV-residual planner as the **+occ arm**, and the
closed-loop score delta Δ(+occ − ego-only) is the perception contribution we want to prove.

Config knobs (driver-config.yaml -> model):
    use_occ: false        # stub CV (ego-only). true -> run occ backbone + residual planner (M1b).
    horizon_s: 4.0        # trajectory horizon
"""
from __future__ import annotations

from typing import Any

import numpy as np

from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)


class OccModel(BaseTrajectoryModel):
    """Occupancy-conditioned planner (stub CV arm; occ wiring in M1b)."""

    def __init__(
        self,
        camera_ids: list[str],
        output_frequency_hz: int,
        horizon_s: float = 4.0,
        use_occ: bool = False,
        device: Any = None,
    ) -> None:
        self._camera_ids = camera_ids
        self._freq = int(output_frequency_hz)
        self._horizon_s = float(horizon_s)
        self._use_occ = bool(use_occ)
        self._device = device
        self._n = max(1, int(round(self._horizon_s * self._freq)))  # #waypoints
        # occ backbone + planner head are loaded lazily in M1b when use_occ=True.
        self._occ = None
        self._planner = None

    # ---- factory -------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        model_cfg: Any,
        device: Any,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "OccModel":
        horizon_s = float(getattr(model_cfg, "horizon_s", 4.0))
        use_occ = bool(getattr(model_cfg, "use_occ", False))
        model = cls(
            camera_ids=camera_ids,
            output_frequency_hz=output_frequency_hz,
            horizon_s=horizon_s,
            use_occ=use_occ,
            device=device,
        )
        if use_occ:
            model._load_occ(model_cfg, device)
        return model

    def _load_occ(self, model_cfg: Any, device: Any) -> None:
        """M1b: load DA3 metric-depth as the (front-cam, label-free) occupancy provider."""
        from depth_anything_3.api import DepthAnything3
        name = getattr(model_cfg, "da3_model", "depth-anything/da3metric-large")
        self._occ = DepthAnything3.from_pretrained(name).to(device).eval()
        self._device = device
        self._safe_margin_m = float(getattr(model_cfg, "safe_margin_m", 6.0))
        self._depth_scale = float(getattr(model_cfg, "depth_scale", 1.0))
        self._occ_mode = str(getattr(model_cfg, "occ_mode", "cap"))     # "cap" (Q1) | "steer" (Q2)
        self._n_bins = int(getattr(model_cfg, "n_bins", 7))             # azimuth bins across the forward FOV
        self._steer_trigger_m = float(getattr(model_cfg, "steer_trigger_m", 20.0))
        self._steer_gain_m = float(getattr(model_cfg, "steer_gain_m", 3.0))

    # ---- required interface -------------------------------------------
    def _encode_command(self, command: DriveCommand) -> Any:
        # canonical one-hot [left, right, straight] (matches our train_planning head)
        return np.array(
            [command == DriveCommand.LEFT, command == DriveCommand.RIGHT,
             command in (DriveCommand.STRAIGHT, DriveCommand.UNKNOWN)],
            dtype=np.float32,
        )

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        dt = 1.0 / self._freq
        v = max(0.0, float(prediction_input.speed))          # m/s, provided by the sim
        # Constant-velocity forward extrapolation in the rig frame (x forward, y left).
        # This is the ego-status prior; the +occ arm will add a learned residual on top.
        xs = np.array([(k + 1) * dt * v for k in range(self._n)], dtype=np.float32)
        ys = np.zeros(self._n, dtype=np.float32)
        traj = np.stack([xs, ys], axis=1)                    # (T,2)
        if self._use_occ:
            prof = self._obstacle_profile(prediction_input)  # per-azimuth nearest depth (bins L->R)
            if prof is not None:
                mid = len(prof) // 2
                d_center = float(np.nanmin(prof[max(0, mid - 1):mid + 2]))  # straight-ahead clearance
                if self._occ_mode == "steer" and np.isfinite(d_center) and d_center < self._steer_trigger_m:
                    # Q2: steer toward the clearest azimuth AND brake to its clearance.
                    prof_c = np.minimum(prof, 60.0)          # cap inf so score is finite
                    idx = np.arange(self._n_bins)
                    score = prof_c - 0.6 * np.abs(idx - mid)  # prefer clear AND near-center (least steering)
                    best = int(np.argmax(score))
                    # image-left column (small index) = scene-left = rig +y(left); so steer +y toward smaller idx
                    lateral = (mid - best) / max(1, mid) * self._steer_gain_m
                    cap = max(0.0, float(prof_c[best]) - self._safe_margin_m)
                    traj[:, 0] = np.minimum(traj[:, 0], cap)
                    steps = np.linspace(0, 1, self._n)       # ramp the lateral shift over the horizon
                    traj[:, 1] = traj[:, 1] + lateral * steps
                elif np.isfinite(d_center):                  # "cap" (Q1): brake only
                    cap = max(0.0, d_center - self._safe_margin_m)
                    traj[:, 0] = np.minimum(traj[:, 0], cap)
        headings = self._compute_headings_from_trajectory(traj)
        return ModelPrediction(trajectory_xy=traj, headings=headings)

    def _obstacle_profile(self, prediction_input: PredictionInput):
        """Per-azimuth nearest-obstacle depths (m) across the forward FOV, from DA3 metric depth.
        Returns an (n_bins,) array (left->right), each = robust (5th-pct) non-sky depth in that
        column band of the lower image. DA3METRIC returns metres directly (is_metric; ~[2.6,128] m
        on real frames), so no focal scaling. None if no camera frame."""
        import torch
        frames = prediction_input.camera_images.get(self._camera_ids[0])
        if not frames:
            return None
        fr = frames[-1]
        img = getattr(fr, "image", None)                     # CameraFrame or (ts, image) tuple
        if img is None:
            img = fr[-1] if isinstance(fr, (tuple, list)) else fr
        img = np.asarray(img)                                # HWC uint8 RGB
        with torch.no_grad():
            pred = self._occ.inference([img])

        def _arr(x):
            return np.asarray(x.detach().cpu()) if hasattr(x, "detach") else np.asarray(x)

        d_m = _arr(pred.depth)[0] * self._depth_scale        # (H,W) metres
        sky = _arr(pred.sky)[0] if getattr(pred, "sky", None) is not None else None
        h, w = d_m.shape
        r0, r1 = int(h * 0.45), int(h * 0.92)                # lower rows = road ahead
        c0, c1 = int(w * 0.30), int(w * 0.70)                # central FOV (drive corridor)
        prof = np.full(self._n_bins, np.inf, dtype=np.float32)
        edges = np.linspace(c0, c1, self._n_bins + 1).astype(int)
        for b in range(self._n_bins):
            band = d_m[r0:r1, edges[b]:edges[b + 1]]
            m = band > 0
            if sky is not None:
                m &= sky[r0:r1, edges[b]:edges[b + 1]] < 0.5
            v = band[m]
            if v.size >= 15:
                prof[b] = float(np.percentile(v, 5))
        return prof

    # ---- properties ----------------------------------------------------
    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return 1

    @property
    def output_frequency_hz(self) -> int:
        return self._freq
