"""
MultiTaskBEVFusion – wraps any BEVFusion-style mmdet3d detector and adds
an occupancy head (and later a radar encoder) without modifying the
original model classes.

Architecture
------------
                     ┌──────────────────────────────┐
  batch_inputs ──────►  inner det model (BEVFusionCA) │
                     │    img_backbone → FPN → VT      │
                     │    SECOND → FireRPF              │
                     │    ConvFuser ──► fused_bev ─────┼──► OccHead → occ_logits
                     │    bbox_head ──► det_losses     │
                     └──────────────────────────────┘

The fused BEV tensor is captured via a PyTorch forward hook registered on
`det_model.fusion_layer`.  This is transparent to the inner model and
requires no changes to BEVFusionCA.

Radar extension (Phase 2)
--------------------------
Set use_radar=True and provide a radar_encoder_cfg.  The radar encoder
produces a radar BEV [B, R_ch, H, W] that is concatenated with the
existing LiDAR BEV *before* ConvFuser, by replacing fusion_layer's
in_channels.  See radar_stub.py for the expected encoder interface.

Registration
------------
@MODELS.register_module() makes this usable as type='MultiTaskBEVFusion'
in an mmdet3d config.  However it is also instantiated directly by
train.py when occ_head / radar are specified outside the model dict.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from mmdet3d.registry import MODELS

from .occ_head import BEVOccHead
from .depth_head import DepthHead, build_depth_gt
from .query_occ_head import QueryOccHead


@MODELS.register_module()
class MultiTaskBEVFusion(nn.Module):
    """
    Args:
        det_model:         config dict for the inner detector (e.g. BEVFusionCA).
                           Alternatively pass an already-built nn.Module.
        occ_head:          BEVOccHead | QueryOccHead instance or config dict.
                           None → no occ supervision.
                             • BEVOccHead: dense voxel-logits head; bigger memory.
                             • QueryOccHead: world-tensor + MLP queries; ~3-14×
                               smaller memory, decouples grid resolution from
                               training cost (P2.11).
        occ_loss_weight:   scalar weight on all occ loss terms.
        depth_head:        DepthHead instance or config dict. None → no depth supervision.
        depth_loss_weight: scalar weight on the depth loss term.
        use_radar:         reserved for Phase 2; currently ignored.
        radar_encoder:     config dict for RadarPillarEncoder (Phase 2).
    """

    def __init__(
        self,
        det_model: Any,                          # dict or nn.Module
        occ_head: Optional[Dict] = None,
        occ_loss_weight: float = 1.0,
        depth_head: Optional[Any] = None,        # DepthHead or dict
        depth_loss_weight: float = 1.0,
        cam_bbox_head: Optional[Any] = None,     # 2nd detection head (config dict) on camera-only BEV
        cam_loss_weight: float = 1.0,
        depth_from_vt: bool = False,             # supervise the view_transform's OWN depthnet
                                                 # (BEVDepth-lift) instead of a separate DepthHead
        use_radar: bool = False,
        radar_encoder: Optional[Dict] = None,
    ):
        super().__init__()
        self.depth_from_vt = bool(depth_from_vt)

        # ---- build / store inner detector ----
        if isinstance(det_model, dict):
            self.det_model: nn.Module = MODELS.build(det_model)
        else:
            self.det_model = det_model

        # ---- dedicated camera-only detection head ----
        # A SECOND TransFusion-style head that runs on the camera-only deep BEV
        # (det_model.extract_feat with BEV_MODALITY=camera). It NEVER sees the
        # LiDAR BEV, so it cannot become LiDAR-anchored the way the shared head is
        # (that anchoring is why modality-dropout and BEV-KD failed to give camera
        # detection). Trained on the same GT boxes as the main head.
        self.cam_bbox_head: Optional[nn.Module] = None
        if cam_bbox_head is not None:
            self.cam_bbox_head = (
                cam_bbox_head if isinstance(cam_bbox_head, nn.Module)
                else MODELS.build(cam_bbox_head)
            )
            # WARM-START from the (already-loaded) main head. A from-scratch
            # TransFusion head blows the heatmap loss to ~3000 and drowns the
            # backbone; copying the trained main-head weights starts the cam head
            # near the main loss (~2-3) so it only has to ADAPT to camera BEV.
            main_head = getattr(self.det_model, "bbox_head", None)
            if main_head is not None:
                missing = self.cam_bbox_head.load_state_dict(
                    main_head.state_dict(), strict=False)
                print(f"[cam_bbox_head] warm-started from main bbox_head "
                      f"(missing={len(missing.missing_keys)}, "
                      f"unexpected={len(missing.unexpected_keys)})")
        self.cam_loss_weight = cam_loss_weight

        # ---- occupancy head ----
        # Accept either the dense `BEVOccHead` or the query-based `QueryOccHead`.
        self.occ_head: Optional[nn.Module] = None
        if occ_head is not None:
            if isinstance(occ_head, (BEVOccHead, QueryOccHead)):
                self.occ_head = occ_head
            else:
                cfg = dict(occ_head)
                head_type = cfg.pop("type", "BEVOccHead")
                if head_type in ("QueryOccHead", "query"):
                    self.occ_head = QueryOccHead(**cfg)
                else:
                    self.occ_head = BEVOccHead(**cfg)

        self.occ_loss_weight = occ_loss_weight

        # ---- depth head (BEVDepth-style auxiliary supervision) ----
        self.depth_head: Optional[DepthHead] = None
        if depth_head is not None:
            if isinstance(depth_head, DepthHead):
                self.depth_head = depth_head
            else:
                cfg_d = dict(depth_head)
                cfg_d.pop("type", None)
                self.depth_head = DepthHead(**cfg_d)
        self.depth_loss_weight = depth_loss_weight

        # ---- radar encoder (Phase 2 stub) ----
        self.use_radar = use_radar
        if use_radar and radar_encoder is not None:
            from .radar_stub import RadarPillarEncoder
            cfg_r = dict(radar_encoder)
            cfg_r.pop("type", None)
            self.radar_encoder: Optional[nn.Module] = RadarPillarEncoder(**cfg_r)
        else:
            self.radar_encoder = None

        # ---- hook to capture fused BEV from ConvFuser ----
        self._fused_bev: Optional[torch.Tensor] = None
        if self.occ_head is not None:
            self._register_bev_hook()

        # ---- hook to capture camera-side P3 features + projection metadata
        #      for depth-head supervision ----
        self._cam_feat_p3: Optional[torch.Tensor] = None
        self._cam_lidar2img: Optional[torch.Tensor] = None
        self._cam_img_aug: Optional[torch.Tensor] = None
        if self.depth_head is not None or self.depth_from_vt:
            self._register_camera_hook()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register_bev_hook(self) -> None:
        """Register a forward hook on fusion_layer to capture fused BEV."""
        fusion = getattr(self.det_model, "fusion_layer", None)
        if fusion is None:
            raise AttributeError(
                "MultiTaskBEVFusion: det_model has no 'fusion_layer' attribute. "
                "Cannot capture fused BEV for occupancy head. "
                "Check that your model has a ConvFuser named 'fusion_layer'."
            )

        def _hook(module, inp, output):
            # output may be a tuple in some fusion variants; take first tensor
            self._fused_bev = output[0] if isinstance(output, (list, tuple)) else output

        fusion.register_forward_hook(_hook)

    def _register_camera_hook(self) -> None:
        """
        Register a forward pre-hook on view_transform to capture (a) the P3
        camera feature tensor, (b) lidar2img, and (c) img_aug_matrix.

        BEVFusionCA.extract_img_feat calls:
            self.view_transform(x_in, points, lidar2image,
                                camera_intrinsics, camera2lidar,
                                img_aug_matrix, lidar_aug_matrix, img_metas)
        so we read positional args by index.
        """
        vt = getattr(self.det_model, "view_transform", None)
        if vt is None:
            raise AttributeError(
                "MultiTaskBEVFusion: det_model has no 'view_transform' attribute. "
                "Cannot capture camera features for depth head."
            )

        def _hook(module, args, kwargs):
            x = args[0] if len(args) >= 1 else kwargs.get("x", None)
            if isinstance(x, (list, tuple)):
                self._cam_feat_p3 = x[0]  # base scale (P3)
            else:
                self._cam_feat_p3 = x
            # lidar2image is positional arg #2 (index 2) in BEVFusionCA call.
            self._cam_lidar2img = args[2] if len(args) >= 3 else kwargs.get("lidar2img", None)
            # img_aug_matrix is positional arg #5 (index 5).
            self._cam_img_aug  = args[5] if len(args) >= 6 else kwargs.get("img_aug_matrix", None)

        vt.register_forward_pre_hook(_hook, with_kwargs=True)

    # ------------------------------------------------------------------
    # mmdet3d / mmengine interface delegation
    # ------------------------------------------------------------------

    @property
    def data_preprocessor(self):
        return self.det_model.data_preprocessor

    @data_preprocessor.setter
    def data_preprocessor(self, v):
        self.det_model.data_preprocessor = v

    def train(self, mode: bool = True):
        super().train(mode)
        self.det_model.train(mode)
        if self.occ_head is not None:
            self.occ_head.train(mode)
        if self.depth_head is not None:
            self.depth_head.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def loss(
        self,
        batch_inputs: Dict[str, Any],
        data_samples: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        1. Run inner detector → detection losses (also fires the BEV + camera hooks).
        2. If occ_head and occ_gt present → add occupancy losses.
        3. If depth_head present → project LiDAR points into camera grids,
           build pseudo-depth GT on the fly, add depth supervision loss.
        """
        losses = self.det_model.loss(batch_inputs, data_samples)

        # ---- Occupancy auxiliary ----
        if self.occ_head is not None and self._fused_bev is not None:
            occ_gt = batch_inputs.get("occ_gt", None)
            if occ_gt is not None:
                occ_gt = occ_gt.to(self._fused_bev.device)
                if isinstance(self.occ_head, QueryOccHead):
                    # Query head does forward+sub-sample inside its own loss(),
                    # so we never materialize dense [B, K, Z, H, W] logits.
                    occ_losses = self.occ_head.loss(self._fused_bev, occ_gt)
                else:
                    # Dense BEVOccHead path: forward then loss(logits, gt).
                    occ_logits = self.occ_head(self._fused_bev)
                    occ_losses = self.occ_head.loss(occ_logits, occ_gt)
                for k, v in occ_losses.items():
                    if k.startswith("loss"):
                        losses[k] = v * self.occ_loss_weight
                    else:
                        # Diagnostics (e.g. occ_query_count) pass through unweighted.
                        losses[k] = v

        # ---- Depth auxiliary ----
        if (
            self.depth_head is not None
            and self._cam_feat_p3 is not None
            and self._cam_lidar2img is not None
        ):
            points = batch_inputs.get("points", None)
            if points is not None:
                # Build per-camera depth labels on-the-fly from raw LiDAR points.
                depth_gt = build_depth_gt(
                    points=points,
                    lidar2img=self._cam_lidar2img,
                    img_aug_matrix=self._cam_img_aug,
                    image_size=self.depth_head.image_size,
                    feature_size=self.depth_head.feature_size,
                    dbound=self.depth_head.dbound,
                    d_bins=self.depth_head.d_bins,
                )
                # Forward depth head under the outer (bf16) autocast for memory efficiency,
                # but cast logits to fp32 before the cross-entropy loss for numerical stability.
                depth_logits = self.depth_head(self._cam_feat_p3)
                depth_losses = self.depth_head.loss(depth_logits.float(), depth_gt)
                for k, v in depth_losses.items():
                    if k.startswith("loss"):
                        losses[k] = v * self.depth_loss_weight
                    else:
                        # diagnostics (e.g. depth_coverage) — pass through unweighted
                        losses[k] = v

        # ---- Depth supervision on the view_transform's OWN depthnet (BEVDepth-lift) ----
        if self.depth_from_vt and self._cam_lidar2img is not None:
            vt = getattr(self.det_model, "view_transform", None)
            logits = getattr(vt, "_last_depth_logits", None) if vt is not None else None
            points = batch_inputs.get("points", None)
            if logits is not None and points is not None:
                depth_gt = build_depth_gt(
                    points=points,
                    lidar2img=self._cam_lidar2img,
                    img_aug_matrix=self._cam_img_aug,
                    image_size=tuple(vt.image_size),
                    feature_size=tuple(vt.feature_size),
                    dbound=tuple(vt.dbound),
                    d_bins=int(vt.D_bins),
                )
                # Focal cross-entropy (BEVDepth/Look-Before-You-Fuse): focus on hard
                # depth pixels instead of plain CE (which plateaued).
                ce = torch.nn.functional.cross_entropy(
                    logits.float(), depth_gt, ignore_index=-1, reduction='none')
                pt = torch.exp(-ce)
                focal = ((1.0 - pt) ** 2.0) * ce
                mask = (depth_gt != -1)
                loss_d = focal[mask].mean() if mask.any() else focal.sum() * 0.0
                losses["loss_depth"] = loss_d * self.depth_loss_weight

        # ---- Camera-only detection head ----
        # Extra forward through the SAME backbone in camera-only mode (LiDAR BEV
        # zeroed), then the dedicated camera head supervised on the GT boxes. The
        # main head's full-LC loss above already preserves L/LC.
        if self.cam_bbox_head is not None:
            import os
            prev_mod = os.environ.get("BEV_MODALITY", "")
            os.environ["BEV_MODALITY"] = "camera"
            try:
                metas = [d.metainfo for d in data_samples]
                cam_feats = self.det_model.extract_feat(batch_inputs, metas)
                cam_losses = self.cam_bbox_head.loss(cam_feats, data_samples)
            finally:
                os.environ["BEV_MODALITY"] = prev_mod
            for k, v in cam_losses.items():
                key = f"cam_{k}"
                if isinstance(v, torch.Tensor) and "loss" in k:
                    losses[key] = v * self.cam_loss_weight
                else:
                    losses[key] = v  # diagnostics (e.g. matched_ious) pass through

        return losses

    # ------------------------------------------------------------------
    # Inference / evaluation delegation
    # ------------------------------------------------------------------

    def predict(self, batch_inputs, data_samples):
        return self.det_model.predict(batch_inputs, data_samples)

    def test_step(self, data):
        return self.det_model.test_step(data)

    def train_step(self, data, optim_wrapper):
        return self.det_model.train_step(data, optim_wrapper)

    def forward(self, inputs=None, data_samples=None, mode: str = "tensor", **kwargs):
        """Delegate to det_model for Runner compatibility."""
        return self.det_model.forward(inputs, data_samples, mode=mode, **kwargs)

    # ------------------------------------------------------------------
    # Named parameters / state dict helpers
    # ------------------------------------------------------------------

    def det_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict of the inner detector only (for eval-only checkpoints)."""
        return self.det_model.state_dict()

    def occ_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.occ_head is None:
            return None
        return {f"occ_head.{k}": v for k, v in self.occ_head.state_dict().items()}
