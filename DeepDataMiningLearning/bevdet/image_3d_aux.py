"""
B11b — FCOS3D-lite per-camera aux head with depth + 3D-center awareness.

Motivation
----------
B11 added a 2D-only aux head (per-class heatmap + 2D wh) supervised by
``cam_instances.bbox``. The diagnostic in EXPERIMENTS.md §5.x showed
that the head trained well as a *2D detector* (101k paired detections at
IoU≥0.3 across 6019 val samples) but **regressed 3D NDS by 0.0044**
because the FPN features it shaped were geometry-blind:

  • hallucinated 39 808 bicycles (narrow-vertical shape prior)
  • under-fired on barriers (low-contrast in image, but LiDAR-easy)
  • class-agreement only 54 % on spatially-paired matches

The 2D objective competed with the CA-LSS lifting objective for FPN
capacity. To fix this, B11b replaces the 2D-only head with an
**FCOS3D-lite** head that supervises FOUR per-cell quantities derived
from cam_instances (all of which we previously discarded):

  • cls heatmap        (focal loss vs Gaussian heatmap, same as B11)
  • 2D (w, h)          (smooth-L1, same as B11)
  • log-depth          (NEW; smooth-L1, supervised by cam_instances.depth)
  • 3D-center offset   (NEW; smooth-L1, supervised by cam_instances.center_2d
                         in augmented-image pixel coords minus cell center)

Why this should reverse the regression
--------------------------------------
The log-depth and 3D-center-offset supervisions force the FPN to encode
*geometric* structure (occlusion, parallax, scale-from-depth) — exactly
the signal CA-LSS needs to lift the image into BEV. A pure 2D shape
detector encodes appearance only; the FPN's shared capacity then drifts
away from depth-relevant features. With direct depth supervision, the
two objectives become complementary instead of competing.

This file mirrors ``image2d_aux.py``'s structure for diffability:

  1. ``LoadCamInstances3DAux`` — pipeline transform; reads
     ``info['cam_instances']`` and exposes (boxes, labels, depth,
     center_2d) per camera in the augmented-image frame.
  2. ``Image3DAuxHead`` — shared 2-conv backbone + 4 prediction heads
     (cls, wh, depth, center3d_offset).
  3. ``Image3DAuxBEVFusion`` — detector subclass of
     FlowGuidedTemporalBEVFusion that hooks FPN P3, runs the head, and
     adds 4 aux losses in ``loss()``.

All head outputs are zero-init (except cls bias = −2.19 focal prior) so
that at iter 0 the head adds zero residual to the FPN gradient. The
losses themselves are non-zero from iter 0 (depth bias is set to
log(20m) so the predicted depth starts at the dataset median).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.transforms import BaseTransform
from mmdet3d.registry import MODELS, TRANSFORMS

from .flow_guided_temporal import FlowGuidedTemporalBEVFusion
from .image2d_aux import _draw_gaussian_2d, _gaussian_radius, gaussian_focal_loss


# ---------------------------------------------------------------------------
# Pipeline transform
# ---------------------------------------------------------------------------

@TRANSFORMS.register_module()
class LoadCamInstances3DAux(BaseTransform):
    """
    Extends ``LoadCamInstances2D`` to also carry per-instance:

      * ``depth``     — scalar depth of the 3D box center (meters)
      * ``center_2d`` — projected 2D center of the 3D center (pixels)

    Output: ``results['cam_inst_3d_aux']`` is a list of length
    ``num_cameras``; each element is::

        {'boxes':     np.float32[N, 4]  (x1 y1 x2 y2 in augmented pixels),
         'labels':    np.int64[N],
         'depth':     np.float32[N]     (meters, of the 3D center),
         'center_2d': np.float32[N, 2]  (cx, cy in augmented pixels)}

    Apply this transform *after* ``ImageAug3D`` so that
    ``results['img_aug_matrix']`` is available and the bboxes /
    center_2d are mapped into the augmented (256×704) image frame.

    Boxes whose 2D footprint becomes smaller than ``clip_min_size`` after
    cropping are dropped (consistent with ``LoadCamInstances2D``).
    """

    def __init__(self,
                 image_size: Tuple[int, int] = (256, 704),
                 clip_min_size: float = 4.0):
        self.image_size = tuple(image_size)
        self.clip_min_size = float(clip_min_size)

    def _apply_img_aug_to_xy(self, xy: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Apply img_aug_matrix 4x4 to an array of [N, 2] pixel coords."""
        R = M[:2, :2]
        t = M[:2, 3]
        return xy @ R.T + t

    def transform(self, results: dict) -> dict:
        cam_inst_dict = results.get('cam_instances', None)
        if cam_inst_dict is None:
            results['cam_inst_3d_aux'] = []
            return results

        cam_keys = list(results['images'].keys()) if 'images' in results \
            else list(cam_inst_dict.keys())
        img_aug = results.get('img_aug_matrix', None)
        H_out, W_out = self.image_size

        out: List[dict] = []
        for c_idx, cam in enumerate(cam_keys):
            inst_list = cam_inst_dict.get(cam, [])
            if not inst_list:
                out.append({
                    'boxes': np.zeros((0, 4), dtype=np.float32),
                    'labels': np.zeros((0,), dtype=np.int64),
                    'depth': np.zeros((0,), dtype=np.float32),
                    'center_2d': np.zeros((0, 2), dtype=np.float32),
                })
                continue

            boxes = np.asarray([b['bbox'] for b in inst_list], dtype=np.float32)
            labels = np.asarray([b['bbox_label'] for b in inst_list], dtype=np.int64)
            depth = np.asarray([b.get('depth', np.nan) for b in inst_list],
                               dtype=np.float32)
            center_2d = np.asarray([b.get('center_2d', [0.0, 0.0])
                                    for b in inst_list], dtype=np.float32)

            if img_aug is not None and c_idx < len(img_aug):
                M = np.asarray(img_aug[c_idx], dtype=np.float32)
                # Project the 4 corners through the 2D affine + recompute xyxy.
                R = M[:2, :2]
                t = M[:2, 3]
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                corners = np.stack([
                    np.stack([x1, y1], axis=-1),
                    np.stack([x2, y1], axis=-1),
                    np.stack([x1, y2], axis=-1),
                    np.stack([x2, y2], axis=-1),
                ], axis=1)                                          # [N, 4, 2]
                corners_t = corners @ R.T + t                       # [N, 4, 2]
                x_aug = corners_t[..., 0]
                y_aug = corners_t[..., 1]
                boxes_aug = np.stack([
                    x_aug.min(-1), y_aug.min(-1),
                    x_aug.max(-1), y_aug.max(-1),
                ], axis=-1)
                # center_2d transforms as a single point.
                center_2d = self._apply_img_aug_to_xy(center_2d, M)
            else:
                boxes_aug = boxes

            # Clip + drop tiny boxes (out-of-frame after crop).
            boxes_aug[:, 0::2] = boxes_aug[:, 0::2].clip(0, W_out - 1)
            boxes_aug[:, 1::2] = boxes_aug[:, 1::2].clip(0, H_out - 1)
            w = boxes_aug[:, 2] - boxes_aug[:, 0]
            h = boxes_aug[:, 3] - boxes_aug[:, 1]
            keep = (w >= self.clip_min_size) & (h >= self.clip_min_size)
            # Depth must be positive and finite.
            keep &= np.isfinite(depth) & (depth > 0.5)

            out.append({
                'boxes': boxes_aug[keep].astype(np.float32),
                'labels': labels[keep].astype(np.int64),
                'depth': depth[keep].astype(np.float32),
                'center_2d': center_2d[keep].astype(np.float32),
            })

        results['cam_inst_3d_aux'] = out
        return results


# ---------------------------------------------------------------------------
# Target builder for the FCOS3D-lite head
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_3d_aux_targets(
    cam_inst_3d_per_sample: List[List[dict]],
    feat_h: int, feat_w: int,
    img_h: int, img_w: int,
    num_classes: int,
    device: torch.device,
):
    """
    Returns:
        cls_hm     : [B, Nc, num_classes, Hf, Wf]   Gaussian heatmap
        wh_t       : [B, Nc, 2,            Hf, Wf]  (w, h) in feature cells
        log_d_t    : [B, Nc, 1,            Hf, Wf]  log-depth (meters)
        c3d_off_t  : [B, Nc, 2,            Hf, Wf]  pixel offset from cell
                                                    center to projected 3D
                                                    center
        pos_mask   : [B, Nc, 1,            Hf, Wf]  binary mask of object
                                                    center cells
    """
    B = len(cam_inst_3d_per_sample)
    if B == 0:
        return None
    Nc = len(cam_inst_3d_per_sample[0])
    stride_x = img_w / feat_w
    stride_y = img_h / feat_h

    cls_hm = torch.zeros((B, Nc, num_classes, feat_h, feat_w), device=device)
    wh_t = torch.zeros((B, Nc, 2, feat_h, feat_w), device=device)
    log_d_t = torch.zeros((B, Nc, 1, feat_h, feat_w), device=device)
    c3d_off_t = torch.zeros((B, Nc, 2, feat_h, feat_w), device=device)
    pos_mask = torch.zeros((B, Nc, 1, feat_h, feat_w), device=device)

    for b in range(B):
        cams = cam_inst_3d_per_sample[b]
        for c in range(Nc):
            if c >= len(cams):
                continue
            d = cams[c]
            boxes = d.get('boxes')
            labels = d.get('labels')
            depth_arr = d.get('depth')
            c2d = d.get('center_2d')
            if boxes is None or len(boxes) == 0:
                continue
            boxes_t = torch.as_tensor(boxes, device=device, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, device=device, dtype=torch.long)
            depth_t = torch.as_tensor(depth_arr, device=device, dtype=torch.float32)
            c2d_t = torch.as_tensor(c2d, device=device, dtype=torch.float32)

            x1, y1, x2, y2 = boxes_t[:, 0], boxes_t[:, 1], boxes_t[:, 2], boxes_t[:, 3]
            cx_px = 0.5 * (x1 + x2)
            cy_px = 0.5 * (y1 + y2)
            w_px = x2 - x1
            h_px = y2 - y1
            cx_feat = cx_px / stride_x
            cy_feat = cy_px / stride_y
            w_feat = w_px / stride_x
            h_feat = h_px / stride_y
            cx_i = cx_feat.long().clamp(0, feat_w - 1)
            cy_i = cy_feat.long().clamp(0, feat_h - 1)
            cell_center_x = (cx_i.float() + 0.5) * stride_x
            cell_center_y = (cy_i.float() + 0.5) * stride_y
            c3d_off_x = c2d_t[:, 0] - cell_center_x       # offset in PIXEL units
            c3d_off_y = c2d_t[:, 1] - cell_center_y
            log_d = torch.log(depth_t.clamp(min=0.5))      # natural log

            for n in range(boxes_t.shape[0]):
                cls = int(labels_t[n].item())
                if cls < 0 or cls >= num_classes:
                    continue
                r = _gaussian_radius(float(h_feat[n].item()),
                                     float(w_feat[n].item()))
                _draw_gaussian_2d(cls_hm[b, c, cls],
                                  (int(cy_i[n].item()), int(cx_i[n].item())), r)
                wh_t[b, c, 0, cy_i[n], cx_i[n]] = float(w_feat[n].item())
                wh_t[b, c, 1, cy_i[n], cx_i[n]] = float(h_feat[n].item())
                log_d_t[b, c, 0, cy_i[n], cx_i[n]] = float(log_d[n].item())
                c3d_off_t[b, c, 0, cy_i[n], cx_i[n]] = float(c3d_off_x[n].item())
                c3d_off_t[b, c, 1, cy_i[n], cx_i[n]] = float(c3d_off_y[n].item())
                pos_mask[b, c, 0, cy_i[n], cx_i[n]] = 1.0

    return cls_hm, wh_t, log_d_t, c3d_off_t, pos_mask


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

class Image3DAuxHead(nn.Module):
    """
    Shared (across cameras) per-pixel FCOS3D-lite head.

    Input  : FPN P3 features [B*Nc, C_in, Hf, Wf]
    Output (dict):
        cls_logit : [B*Nc, num_classes, Hf, Wf]
        wh        : [B*Nc, 2,           Hf, Wf]
        log_depth : [B*Nc, 1,           Hf, Wf]
        c3d_off   : [B*Nc, 2,           Hf, Wf]
    """

    INIT_LOG_DEPTH = 3.0   # log(20m) — nuScenes 3D-center depth median

    def __init__(self, in_channels: int = 256, num_classes: int = 10,
                 hidden_channels: int = 256, prior_logit_bias: float = -2.19):
        super().__init__()
        self.num_classes = int(num_classes)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.cls_head = nn.Conv2d(hidden_channels, num_classes, 1, bias=True)
        self.wh_head = nn.Conv2d(hidden_channels, 2, 1, bias=True)
        self.depth_head = nn.Conv2d(hidden_channels, 1, 1, bias=True)
        self.c3d_head = nn.Conv2d(hidden_channels, 2, 1, bias=True)

        # cls: focal prior so sigmoid(b) ≈ 0.1
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.constant_(self.cls_head.bias, prior_logit_bias)
        # wh: zero-init (only meaningful at positive cells; mask-gated loss)
        nn.init.zeros_(self.wh_head.weight)
        nn.init.zeros_(self.wh_head.bias)
        # depth: zero weight, bias = log(20m) so initial predictions are
        # already in the right order of magnitude.
        nn.init.zeros_(self.depth_head.weight)
        nn.init.constant_(self.depth_head.bias, self.INIT_LOG_DEPTH)
        # 3D-center offset: zero-init (offset starts at 0 = cell center)
        nn.init.zeros_(self.c3d_head.weight)
        nn.init.zeros_(self.c3d_head.bias)

    def forward(self, p3: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.shared(p3)
        return {
            'cls_logit': self.cls_head(x),
            'wh': self.wh_head(x),
            'log_depth': self.depth_head(x),
            'c3d_off': self.c3d_head(x),
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@MODELS.register_module()
class Image3DAuxBEVFusion(FlowGuidedTemporalBEVFusion):
    """
    FlowGuidedTemporalBEVFusion + per-camera FCOS3D-lite aux head.

    Args:
        image3d_aux: dict with keys
            * num_classes (int, default 10)
            * loss_weight_cls   (default 0.3)
            * loss_weight_wh    (default 0.1)
            * loss_weight_depth (default 0.2)        ← NEW
            * loss_weight_c3d   (default 0.1)        ← NEW
            * hidden_channels   (int, default 256)
            * image_size        (tuple, default (256, 704))
            * in_channels       (int, default 256)
        **flow_guided_kwargs: forwarded to FlowGuidedTemporalBEVFusion.
    """

    def __init__(self,
                 image3d_aux: Optional[Dict] = None,
                 **flow_guided_kwargs):
        super().__init__(**flow_guided_kwargs)
        cfg = image3d_aux or {}
        self.i3d_num_classes = int(cfg.get('num_classes', 10))
        self.i3d_w_cls   = float(cfg.get('loss_weight_cls',   0.3))
        self.i3d_w_wh    = float(cfg.get('loss_weight_wh',    0.1))
        self.i3d_w_depth = float(cfg.get('loss_weight_depth', 0.2))
        self.i3d_w_c3d   = float(cfg.get('loss_weight_c3d',   0.1))
        self.i3d_image_size = tuple(cfg.get('image_size', (256, 704)))
        hidden = int(cfg.get('hidden_channels', 256))
        img_ch = int(cfg.get('in_channels', 256))

        self.image3d_head = Image3DAuxHead(
            in_channels=img_ch,
            num_classes=self.i3d_num_classes,
            hidden_channels=hidden,
        )

        # FPN P3 capture (same hook pattern as B11).
        self._cached_image_p3: Optional[torch.Tensor] = None

        def _capture_p3(module, inputs, output):
            self._cached_image_p3 = output[0] if isinstance(output, (list, tuple)) else output

        if self.img_neck is not None:
            self.img_neck.register_forward_hook(_capture_p3)

        print(f'[Image3DAuxBEVFusion] B11b = FG-TCA + FCOS3D-lite head: '
              f'num_classes={self.i3d_num_classes}, '
              f'w_cls={self.i3d_w_cls}, w_wh={self.i3d_w_wh}, '
              f'w_depth={self.i3d_w_depth}, w_c3d={self.i3d_w_c3d}, '
              f'image_size={self.i3d_image_size}, in_channels={img_ch}')

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        self._cached_image_p3 = None
        losses = super().loss(batch_inputs_dict, batch_data_samples, **kwargs)

        p3 = self._cached_image_p3
        self._cached_image_p3 = None
        if p3 is None:
            return losses

        with torch.autocast('cuda', enabled=False):
            preds = self.image3d_head(p3.float())

        # Reshape: [B*Nc, C, Hf, Wf] → [B, Nc, C, Hf, Wf]
        B = len(batch_data_samples)
        BNc = preds['cls_logit'].shape[0]
        if BNc % B != 0:
            return losses
        Nc = BNc // B
        Hf, Wf = preds['cls_logit'].shape[-2], preds['cls_logit'].shape[-1]

        def _b(x):
            return x.view(B, Nc, x.shape[1], Hf, Wf)

        cls_logit = _b(preds['cls_logit'])
        wh = _b(preds['wh'])
        log_depth = _b(preds['log_depth'])
        c3d_off = _b(preds['c3d_off'])

        # Pull per-sample cam_inst_3d_aux from metainfo.
        cam_inst_per_sample: List[List[dict]] = []
        for s in batch_data_samples:
            ci = s.metainfo.get('cam_inst_3d_aux', None)
            cam_inst_per_sample.append(ci if ci is not None else [])

        if not any(len(cis) > 0 for cis in cam_inst_per_sample):
            return losses

        img_h, img_w = self.i3d_image_size
        targets = build_3d_aux_targets(
            cam_inst_per_sample,
            feat_h=Hf, feat_w=Wf,
            img_h=img_h, img_w=img_w,
            num_classes=self.i3d_num_classes,
            device=cls_logit.device,
        )
        cls_hm_t, wh_t, log_d_t, c3d_off_t, pos_mask = targets

        # cls: focal loss on sigmoid heatmap
        pred_hm = cls_logit.sigmoid()
        loss_cls = gaussian_focal_loss(pred_hm, cls_hm_t)

        # Box / depth / c3d only at positive cells (object centers).
        m2 = pos_mask.expand_as(wh)
        m1 = pos_mask                                   # [B, Nc, 1, H, W]
        n_pos = pos_mask.sum().clamp(min=1.0)

        loss_wh = F.smooth_l1_loss(wh * m2, wh_t * m2, reduction='sum') / n_pos
        loss_depth = F.smooth_l1_loss(log_depth * m1, log_d_t * m1,
                                      reduction='sum') / n_pos
        loss_c3d = F.smooth_l1_loss(c3d_off * m2, c3d_off_t * m2,
                                    reduction='sum') / n_pos

        losses['loss_3d_aux_cls']   = self.i3d_w_cls   * loss_cls
        losses['loss_3d_aux_wh']    = self.i3d_w_wh    * loss_wh
        losses['loss_3d_aux_depth'] = self.i3d_w_depth * loss_depth
        losses['loss_3d_aux_c3d']   = self.i3d_w_c3d   * loss_c3d
        return losses
