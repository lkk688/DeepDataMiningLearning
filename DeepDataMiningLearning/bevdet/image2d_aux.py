"""
B11 — Per-camera 2D detection aux head.

Motivation
----------
B7→B10c (every temporal experiment) failed to move NDS more than +0.003.
The diagnostic in section 5.9 of EXPERIMENTS.md points at one structural
gap that has *not* been touched: the image branch is the only modality
trained purely indirectly. It receives gradient ONLY via the 3D-detection
loss, which propagates back through view-transform → fusion → ConvFuser →
pts_neck → pts_backbone → bbox_head. By the time the gradient reaches
Swin's FPN it has been heavily diluted by the LiDAR-dominated fusion.

B11 adds a per-camera 2D detection head consumed directly off the FPN
P3 features. Supervised by nuScenes' annotated 2D boxes
(``info['cam_instances']``, projected to augmented image coords via the
per-camera ``img_aug_matrix``). The gradient signal goes:

    loss_2d_aux → image2d_head → FPN P3 → img_neck → img_backbone (Swin)

Same "direct supervision channel" trick that finally let the temporal
block train in B10c — applied to the image branch this time. The head
itself is ~70 k params; negligible compute on top of the 6-cam pipeline.

This file contains three components:
  1. ``LoadCamInstances2D`` — pipeline transform that reads
     ``results['cam_instances']``, projects each 2D box through
     ``img_aug_matrix`` (per camera) into the augmented-image pixel
     frame, and stashes the result in ``results['cam_inst_2d']`` so
     ``Pack3DDetInputs`` carries it to the model via metainfo.
  2. ``Image2DAuxHead`` — 2-conv shared (across cameras) head producing
     per-class heatmap + (w, h) regression at the FPN P3 grid.
  3. ``Image2DAuxBEVFusion`` — detector wrapping FlowGuidedTemporalBEVFusion;
     captures P3 via a forward hook on ``img_neck`` and adds the
     ``loss_2d_aux_hm`` + ``loss_2d_aux_wh`` terms in ``loss()``.

The 2D head's heatmap output uses a CenterNet-style sigmoid prior bias
(``-2.19``, so sigmoid(b) ≈ 0.1) so initial focal-loss is bounded. The
``wh`` channels are zero-init. Together with a small ``loss_weight=0.5``
this gives the image branch a strong supervision signal without
overwhelming the 3D loss.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.transforms import BaseTransform
from mmdet3d.registry import MODELS, TRANSFORMS

from .flow_guided_temporal import FlowGuidedTemporalBEVFusion


# ---------------------------------------------------------------------------
# Pipeline transform: read cam_instances, apply img_aug_matrix, pack
# ---------------------------------------------------------------------------

@TRANSFORMS.register_module()
class LoadCamInstances2D(BaseTransform):
    """
    For each camera, collect (boxes, labels) from ``results['cam_instances']``
    and apply that camera's 2D portion of ``img_aug_matrix`` to the boxes
    so they live in the post-augmentation pixel frame.

    Output: ``results['cam_inst_2d']`` is a list of length ``num_cameras``,
    each element a dict ``{'boxes': np.ndarray [N, 4] in xyxy pixel coords,
    'labels': np.ndarray [N]}``.

    Camera order must match the order ``img_aug_matrix`` is built in
    (the BEVFusion pipelines iterate ``info['images']`` in dict order,
    which Python preserves: CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT,
    CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT).
    """

    def __init__(self, image_size: Tuple[int, int] = (256, 704),
                 clip_min_size: float = 4.0):
        self.image_size = tuple(image_size)
        self.clip_min_size = float(clip_min_size)

    def transform(self, results: dict) -> dict:
        cam_inst_dict = results.get('cam_instances', None)
        if cam_inst_dict is None:
            results['cam_inst_2d'] = []
            return results

        # Iteration order must match the order images were loaded /
        # img_aug_matrix was built in. NuScenesDataset uses dict insertion
        # order on info['images'].
        cam_keys = list(results['images'].keys()) if 'images' in results \
            else list(cam_inst_dict.keys())

        img_aug = results.get('img_aug_matrix', None)   # list of 4x4 per cam
        H_out, W_out = self.image_size

        out: List[dict] = []
        for c_idx, cam in enumerate(cam_keys):
            inst_list = cam_inst_dict.get(cam, [])
            if not inst_list:
                out.append({
                    'boxes': np.zeros((0, 4), dtype=np.float32),
                    'labels': np.zeros((0,), dtype=np.int64),
                })
                continue

            boxes = np.asarray([b['bbox'] for b in inst_list], dtype=np.float32)
            labels = np.asarray([b['bbox_label'] for b in inst_list], dtype=np.int64)

            if img_aug is not None and c_idx < len(img_aug):
                M = np.asarray(img_aug[c_idx], dtype=np.float32)   # 4x4
                R = M[:2, :2]
                t = M[:2, 3]
                # Project all 4 corners through the 2D affine and recompute xyxy.
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                corners = np.stack([
                    np.stack([x1, y1], axis=-1),
                    np.stack([x2, y1], axis=-1),
                    np.stack([x1, y2], axis=-1),
                    np.stack([x2, y2], axis=-1),
                ], axis=1)                                          # [N, 4, 2]
                corners_t = corners @ R.T + t                       # [N, 4, 2]
                x_aug = corners_t[..., 0]                           # [N, 4]
                y_aug = corners_t[..., 1]                           # [N, 4]
                boxes_aug = np.stack([
                    x_aug.min(-1), y_aug.min(-1),
                    x_aug.max(-1), y_aug.max(-1),
                ], axis=-1)
            else:
                boxes_aug = boxes

            # Clip to image bounds; drop boxes that became too small.
            boxes_aug[:, 0::2] = boxes_aug[:, 0::2].clip(0, W_out - 1)
            boxes_aug[:, 1::2] = boxes_aug[:, 1::2].clip(0, H_out - 1)
            w = boxes_aug[:, 2] - boxes_aug[:, 0]
            h = boxes_aug[:, 3] - boxes_aug[:, 1]
            keep = (w >= self.clip_min_size) & (h >= self.clip_min_size)
            boxes_aug = boxes_aug[keep]
            labels = labels[keep]

            out.append({
                'boxes': boxes_aug.astype(np.float32),
                'labels': labels.astype(np.int64),
            })

        results['cam_inst_2d'] = out
        return results


# ---------------------------------------------------------------------------
# Heatmap target generation (CenterNet style)
# ---------------------------------------------------------------------------

def _gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> int:
    """CornerNet/CenterNet box-shape → Gaussian radius."""
    a1 = 1.0
    b1 = (h + w)
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = max(0.0, b1 * b1 - 4 * a1 * c1) ** 0.5
    r1 = (b1 + sq1) / (2 * a1)
    a2 = 4.0
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = max(0.0, b2 * b2 - 4 * a2 * c2) ** 0.5
    r2 = (b2 + sq2) / (2 * a2)
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = max(0.0, b3 * b3 - 4 * a3 * c3) ** 0.5
    r3 = (b3 + sq3) / (2 * a3)
    return int(max(0, min(r1, r2, r3)))


def _draw_gaussian_2d(hm: torch.Tensor, center: Tuple[int, int],
                      radius: int) -> None:
    """In-place CenterNet Gaussian on hm[H, W]."""
    if radius < 1:
        radius = 1
    diam = 2 * radius + 1
    sigma = diam / 6.0
    H, W = hm.shape[-2], hm.shape[-1]
    y, x = int(center[0]), int(center[1])
    yy = torch.arange(0, diam, device=hm.device).view(-1, 1).float()
    xx = torch.arange(0, diam, device=hm.device).view(1, -1).float()
    g = torch.exp(-((yy - radius) ** 2 + (xx - radius) ** 2) / (2 * sigma ** 2))
    top, bottom = max(0, y - radius), min(H, y + radius + 1)
    left, right = max(0, x - radius), min(W, x + radius + 1)
    g_top = max(0, radius - y)
    g_left = max(0, radius - x)
    g_bottom = g_top + (bottom - top)
    g_right = g_left + (right - left)
    if top < bottom and left < right:
        hm[..., top:bottom, left:right] = torch.maximum(
            hm[..., top:bottom, left:right],
            g[g_top:g_bottom, g_left:g_right],
        )


@torch.no_grad()
def build_2d_targets(
    cam_inst_2d_per_sample: List[List[dict]],   # [B] of [N_cam] of dict
    feat_h: int, feat_w: int,
    img_h: int, img_w: int,
    num_classes: int,
    device: torch.device,
):
    """
    Build (heatmap, wh_target, mask) tensors at the FPN P3 grid.

    Returns:
        hm:       [B, N_cam, num_classes, feat_h, feat_w]   float
        wh:       [B, N_cam, 2,            feat_h, feat_w]  float  (only valid where mask=1)
        wh_mask:  [B, N_cam, 1,            feat_h, feat_w]  float
    """
    B = len(cam_inst_2d_per_sample)
    if B == 0:
        return None
    N_cam = len(cam_inst_2d_per_sample[0])
    stride_x = img_w / feat_w
    stride_y = img_h / feat_h

    hm = torch.zeros((B, N_cam, num_classes, feat_h, feat_w), device=device)
    wh = torch.zeros((B, N_cam, 2, feat_h, feat_w), device=device)
    wh_mask = torch.zeros((B, N_cam, 1, feat_h, feat_w), device=device)

    for b in range(B):
        cams = cam_inst_2d_per_sample[b]
        for c in range(N_cam):
            if c >= len(cams):
                continue
            d = cams[c]
            boxes = d.get('boxes')
            labels = d.get('labels')
            if boxes is None or len(boxes) == 0:
                continue
            # Convert to feature-grid coordinates.
            boxes_t = torch.as_tensor(boxes, device=device, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, device=device, dtype=torch.long)
            x1, y1, x2, y2 = boxes_t[:, 0], boxes_t[:, 1], boxes_t[:, 2], boxes_t[:, 3]
            cx_feat = (0.5 * (x1 + x2)) / stride_x
            cy_feat = (0.5 * (y1 + y2)) / stride_y
            w_feat = (x2 - x1) / stride_x
            h_feat = (y2 - y1) / stride_y
            cx_i = cx_feat.long().clamp(0, feat_w - 1)
            cy_i = cy_feat.long().clamp(0, feat_h - 1)
            for n in range(boxes_t.shape[0]):
                cls = int(labels_t[n].item())
                if cls < 0 or cls >= num_classes:
                    continue
                r = _gaussian_radius(float(h_feat[n].item()),
                                     float(w_feat[n].item()))
                _draw_gaussian_2d(hm[b, c, cls], (int(cy_i[n].item()),
                                                  int(cx_i[n].item())), r)
                # wh target stored at center cell only
                wh[b, c, 0, cy_i[n], cx_i[n]] = float(w_feat[n].item())
                wh[b, c, 1, cy_i[n], cx_i[n]] = float(h_feat[n].item())
                wh_mask[b, c, 0, cy_i[n], cx_i[n]] = 1.0

    return hm, wh, wh_mask


def gaussian_focal_loss(
    pred_sigmoid: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    gamma: float = 4.0,
) -> torch.Tensor:
    """Standard CenterNet focal loss on a sigmoid'd heatmap."""
    eps = 1e-6
    pos_weights = target.eq(1).float()
    neg_weights = (1 - target).pow(gamma)
    pos_loss = -((1 - pred_sigmoid) ** alpha) * \
        torch.log(pred_sigmoid.clamp(min=eps)) * pos_weights
    neg_loss = -(pred_sigmoid ** alpha) * \
        torch.log((1 - pred_sigmoid).clamp(min=eps)) * neg_weights
    num_pos = pos_weights.sum().clamp(min=1.0)
    return (pos_loss + neg_loss).sum() / num_pos


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

class Image2DAuxHead(nn.Module):
    """
    Shared (across cameras) per-pixel 2D detection head.

    Input  : FPN P3 features [B*Nc, C_in, Hf, Wf]
    Output : heatmap (sigmoid) [B*Nc, num_classes, Hf, Wf]
             wh (raw)          [B*Nc, 2,           Hf, Wf]
    """

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

        # Heatmap prior: bias = -2.19  →  sigmoid(b) ≈ 0.1
        nn.init.normal_(self.cls_head.weight, std=0.01)
        with torch.no_grad():
            self.cls_head.bias.fill_(prior_logit_bias)
        # wh: zero-init (irrelevant when wh_mask gates the loss)
        nn.init.zeros_(self.wh_head.weight)
        nn.init.zeros_(self.wh_head.bias)

    def forward(self, p3: torch.Tensor):
        x = self.shared(p3)
        hm_logit = self.cls_head(x)
        wh = self.wh_head(x)
        return hm_logit, wh


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@MODELS.register_module()
class Image2DAuxBEVFusion(FlowGuidedTemporalBEVFusion):
    """
    FlowGuidedTemporalBEVFusion + per-camera 2D detection aux head.

    Args:
        image2d_aux: dict with keys
            * num_classes (int, default 10)
            * loss_weight_hm (float, default 0.5)
            * loss_weight_wh (float, default 0.1)
            * hidden_channels (int, default 256)
            * image_size (tuple, default (256, 704)) — passed to the
              loss target builder; must match the pipeline's image size.
        **flow_guided_kwargs: forwarded to FlowGuidedTemporalBEVFusion.
    """

    def __init__(self,
                 image2d_aux: Optional[Dict] = None,
                 **flow_guided_kwargs):
        super().__init__(**flow_guided_kwargs)
        cfg = image2d_aux or {}
        self.image2d_num_classes = int(cfg.get('num_classes', 10))
        self.image2d_loss_weight_hm = float(cfg.get('loss_weight_hm', 0.5))
        self.image2d_loss_weight_wh = float(cfg.get('loss_weight_wh', 0.1))
        self.image2d_image_size = tuple(cfg.get('image_size', (256, 704)))
        hidden = int(cfg.get('hidden_channels', 256))

        # Infer image-FPN channel count from img_neck output (defaults to 256).
        img_ch = int(cfg.get('in_channels', 256))

        self.image2d_head = Image2DAuxHead(
            in_channels=img_ch,
            num_classes=self.image2d_num_classes,
            hidden_channels=hidden,
        )

        # Capture FPN P3 via a forward hook on img_neck. P3 is the first
        # output level (highest resolution).
        self._cached_image_p3: Optional[torch.Tensor] = None

        def _capture_p3(module, inputs, output):
            self._cached_image_p3 = output[0] if isinstance(output, (list, tuple)) else output

        if self.img_neck is not None:
            self.img_neck.register_forward_hook(_capture_p3)

        print(f'[Image2DAuxBEVFusion] B11 = FG-TCA + 2D aux head: '
              f'num_classes={self.image2d_num_classes}, '
              f'loss_weight_hm={self.image2d_loss_weight_hm}, '
              f'loss_weight_wh={self.image2d_loss_weight_wh}, '
              f'image_size={self.image2d_image_size}, in_channels={img_ch}')

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        self._cached_image_p3 = None
        losses = super().loss(batch_inputs_dict, batch_data_samples, **kwargs)

        p3 = self._cached_image_p3
        self._cached_image_p3 = None
        if p3 is None:
            return losses

        # p3 is [B*Nc, C, Hf, Wf] — run our head.
        with torch.autocast('cuda', enabled=False):
            hm_logit, wh = self.image2d_head(p3.float())

        # Reshape to [B, Nc, ...] using the batch size implied by data_samples.
        B = len(batch_data_samples)
        BNc, _, Hf, Wf = hm_logit.shape
        if BNc % B != 0:
            return losses  # malformed; skip aux loss for this step
        Nc = BNc // B
        hm_logit_b = hm_logit.view(B, Nc, self.image2d_num_classes, Hf, Wf)
        wh_b = wh.view(B, Nc, 2, Hf, Wf)

        # Pull per-sample cam_inst_2d from metainfo.
        cam_inst_per_sample: List[List[dict]] = []
        for s in batch_data_samples:
            ci = s.metainfo.get('cam_inst_2d', None)
            cam_inst_per_sample.append(ci if ci is not None else [])

        # No GT this step? skip.
        if not any(len(cis) > 0 for cis in cam_inst_per_sample):
            return losses

        img_h, img_w = self.image2d_image_size
        hm_t, wh_t, wh_mask = build_2d_targets(
            cam_inst_per_sample,
            feat_h=Hf, feat_w=Wf,
            img_h=img_h, img_w=img_w,
            num_classes=self.image2d_num_classes,
            device=hm_logit.device,
        )

        pred_hm = hm_logit_b.sigmoid()
        loss_hm = gaussian_focal_loss(pred_hm, hm_t)

        m = wh_mask.expand_as(wh_b)
        n_pos = m.sum().clamp(min=1.0)
        wh_diff = (wh_b - wh_t) * m
        loss_wh = F.smooth_l1_loss(wh_b * m, wh_t * m, reduction='sum') / n_pos

        losses['loss_2d_aux_hm'] = self.image2d_loss_weight_hm * loss_hm
        losses['loss_2d_aux_wh'] = self.image2d_loss_weight_wh * loss_wh
        return losses
