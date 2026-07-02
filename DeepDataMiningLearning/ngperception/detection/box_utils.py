"""
ngperception.detection.box_utils
================================

3D box coding + **pure-PyTorch 3D IoU / NMS** — the one piece OpenPCDet (and our source
`2D3DFusion/mydetector3d`) implements as a custom CUDA op (`ops/iou3d_nms`). Replacing it in
pure torch is what lets a PointPillars-style detector run with only torch + numpy, no spconv,
no compiled ops. Boxes are the standard 7-DoF LiDAR format **[x, y, z, dx, dy, dz, heading]**
(centre, size, yaw). Provenance: `ResidualCoder` follows OpenPCDet's box_coder_utils.

Two IoU paths (see PLAN.md M0/M2):
  * `boxes_bev_iou_aligned` — fully vectorised **axis-aligned** BEV IoU (exact for yaw≈0,
    an approximation under rotation). Fast enough for target assignment + NMS at anchor scale.
  * `rotated_iou_bev` — **exact** rotated IoU via corner + Sutherland–Hodgman clipping
    (looped, for eval / unit tests / small N). Vectorising this for the training path is M2.
"""
from __future__ import annotations
import numpy as np
import torch


class ResidualCoder:
    """Encode/decode 7-DoF boxes relative to anchors (OpenPCDet ResidualCoder)."""

    def __init__(self, code_size: int = 7):
        self.code_size = code_size

    def encode(self, boxes, anchors):
        anchors = anchors.clone(); boxes = boxes.clone()
        anchors[:, 3:6] = anchors[:, 3:6].clamp_min(1e-5)
        boxes[:, 3:6] = boxes[:, 3:6].clamp_min(1e-5)
        xa, ya, za, dxa, dya, dza, ra = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg = torch.split(boxes, 1, dim=-1)
        diag = torch.sqrt(dxa ** 2 + dya ** 2)
        return torch.cat([(xg - xa) / diag, (yg - ya) / diag, (zg - za) / dza,
                          torch.log(dxg / dxa), torch.log(dyg / dya), torch.log(dzg / dza),
                          rg - ra], dim=-1)

    def decode(self, enc, anchors):
        xa, ya, za, dxa, dya, dza, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, dxt, dyt, dzt, rt = torch.split(enc, 1, dim=-1)
        diag = torch.sqrt(dxa ** 2 + dya ** 2)
        return torch.cat([xt * diag + xa, yt * diag + ya, zt * dza + za,
                          torch.exp(dxt) * dxa, torch.exp(dyt) * dya, torch.exp(dzt) * dza,
                          rt + ra], dim=-1)


def boxes_to_corners_bev(boxes):
    """(N,7) -> (N,4,2) BEV corners in CCW order (TR,TL,BL,BR), honoring heading."""
    x, y, dx, dy, ang = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]
    cx = torch.stack([dx / 2, -dx / 2, -dx / 2, dx / 2], dim=1)
    cy = torch.stack([dy / 2, dy / 2, -dy / 2, -dy / 2], dim=1)
    cos, sin = torch.cos(ang)[:, None], torch.sin(ang)[:, None]
    return torch.stack([cx * cos - cy * sin + x[:, None],
                        cx * sin + cy * cos + y[:, None]], dim=-1)   # (N,4,2)


def boxes_bev_iou_aligned(a, b):
    """Axis-aligned BEV IoU (N,M): treats dx,dy as axis extents (ignores yaw). Vectorised."""
    ax1, ax2 = a[:, 0] - a[:, 3] / 2, a[:, 0] + a[:, 3] / 2
    ay1, ay2 = a[:, 1] - a[:, 4] / 2, a[:, 1] + a[:, 4] / 2
    bx1, bx2 = b[:, 0] - b[:, 3] / 2, b[:, 0] + b[:, 3] / 2
    by1, by2 = b[:, 1] - b[:, 4] / 2, b[:, 1] + b[:, 4] / 2
    iw = (torch.min(ax2[:, None], bx2[None]) - torch.max(ax1[:, None], bx1[None])).clamp(min=0)
    ih = (torch.min(ay2[:, None], by2[None]) - torch.max(ay1[:, None], by1[None])).clamp(min=0)
    inter = iw * ih
    area_a, area_b = (a[:, 3] * a[:, 4])[:, None], (b[:, 3] * b[:, 4])[None]
    return inter / (area_a + area_b - inter + 1e-6)


def boxes_iou3d_aligned(a, b):
    """3D IoU (N,M) = axis-aligned BEV IoU × height overlap."""
    za1, za2 = a[:, 2] - a[:, 5] / 2, a[:, 2] + a[:, 5] / 2
    zb1, zb2 = b[:, 2] - b[:, 5] / 2, b[:, 2] + b[:, 5] / 2
    hov = (torch.min(za2[:, None], zb2[None]) - torch.max(za1[:, None], zb1[None])).clamp(min=0)
    bev = boxes_bev_iou_aligned(a, b)                                # reuse the 2D overlap ratio
    # rebuild absolute intersection to combine with height cleanly:
    ax1, ax2 = a[:, 0] - a[:, 3] / 2, a[:, 0] + a[:, 3] / 2
    ay1, ay2 = a[:, 1] - a[:, 4] / 2, a[:, 1] + a[:, 4] / 2
    bx1, bx2 = b[:, 0] - b[:, 3] / 2, b[:, 0] + b[:, 3] / 2
    by1, by2 = b[:, 1] - b[:, 4] / 2, b[:, 1] + b[:, 4] / 2
    iw = (torch.min(ax2[:, None], bx2[None]) - torch.max(ax1[:, None], bx1[None])).clamp(min=0)
    ih = (torch.min(ay2[:, None], by2[None]) - torch.max(ay1[:, None], by1[None])).clamp(min=0)
    inter = iw * ih * hov
    vol_a = (a[:, 3] * a[:, 4] * a[:, 5])[:, None]
    vol_b = (b[:, 3] * b[:, 4] * b[:, 5])[None]
    return inter / (vol_a + vol_b - inter + 1e-6)


def nms_aligned(boxes, scores, thresh, pre_max=4096, post_max=500):
    """Greedy NMS on axis-aligned BEV IoU. boxes (N,7), scores (N,) -> kept indices (torch)."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)[:pre_max]
    boxes = boxes[order]
    keep = []
    idx = torch.arange(boxes.shape[0], device=boxes.device)
    while idx.numel() > 0 and len(keep) < post_max:
        i = idx[0]
        keep.append(i.item())
        if idx.numel() == 1:
            break
        iou = boxes_bev_iou_aligned(boxes[i:i + 1], boxes[idx[1:]])[0]
        idx = idx[1:][iou <= thresh]
    return order[torch.tensor(keep, dtype=torch.long, device=boxes.device)]


# --------------------------------------------------------------------------- #
# Exact rotated BEV IoU (numpy, Sutherland–Hodgman) — eval / unit tests (M2 vectorises this)
# --------------------------------------------------------------------------- #
def _poly_clip(subject, clip):
    """Clip convex polygon `subject` by convex polygon `clip` (both CCW lists of (x,y))."""
    out = subject
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = (b[0] - a[0], b[1] - a[1])
        inp, out = out, []
        if not inp:
            break
        for j in range(len(inp)):
            p, q = inp[j - 1], inp[j]
            sp = edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])   # >0 => left/inside
            sq = edge[0] * (q[1] - a[1]) - edge[1] * (q[0] - a[0])
            if sq >= 0:
                if sp < 0:
                    out.append(_line_isect(p, q, a, b))
                out.append(q)
            elif sp >= 0:
                out.append(_line_isect(p, q, a, b))
    return out


def _line_isect(p, q, a, b):
    d1 = (q[0] - p[0], q[1] - p[1]); d2 = (b[0] - a[0], b[1] - a[1])
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return q
    t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / denom
    return (p[0] + t * d1[0], p[1] + t * d1[1])


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def rotated_iou_bev(boxes_a, boxes_b):
    """Exact rotated BEV IoU (Na,Nb) via polygon clipping. numpy/torch in, numpy out."""
    A = boxes_a.detach().cpu().numpy() if torch.is_tensor(boxes_a) else np.asarray(boxes_a)
    B = boxes_b.detach().cpu().numpy() if torch.is_tensor(boxes_b) else np.asarray(boxes_b)
    ca = boxes_to_corners_bev(torch.as_tensor(A, dtype=torch.float32)).numpy()
    cb = boxes_to_corners_bev(torch.as_tensor(B, dtype=torch.float32)).numpy()
    out = np.zeros((len(A), len(B)), np.float32)
    for i in range(len(A)):
        pa = [tuple(p) for p in ca[i]]; area_a = _poly_area(pa)
        for j in range(len(B)):
            pb = [tuple(p) for p in cb[j]]
            inter = _poly_area(_poly_clip(pa, pb))
            out[i, j] = inter / (area_a + _poly_area(pb) - inter + 1e-6)
    return out
