"""
ngperception.lane.lane_iou
==========================

**LineIoU** (CLRNet) and **LaneIoU** (CLRerNet) — pure-torch, no mmdet.

A lane, in the CLRNet representation, is a set of x-coordinates sampled at a fixed
grid of image rows (y positions). Two lanes overlap well when their x's agree *row by
row*. A plain L1 on x's treats every row independently and is blind to how "thick" a
match should be; **LineIoU** instead turns each sampled point into a short horizontal
segment of radius ``r`` pixels and measures interval IoU, summed over rows:

    overlap_i = min(px_i+r, tx_i+r) − max(px_i−r, tx_i−r)      (can be < 0 → a gap)
    union_i   = max(px_i+r, tx_i+r) − min(px_i−r, tx_i−r)
    IoU       = Σ_i overlap_i / Σ_i union_i          (over valid rows only)
    loss      = 1 − IoU

Summing overlap and union *before* dividing (rather than per-row IoU) is the CLRNet
trick — it stays differentiable and well-behaved even where segments miss entirely.

**LaneIoU** (CLRerNet, WACV'24) fixes a bias: a *tilted* lane covers more horizontal
extent per row, so a fixed radius under-counts overlap on slanted lanes and biases
confidence toward near-vertical ones. LaneIoU scales the radius per row by the local
slope ``√(1 + (dx/dy)²)`` (the true horizontal footprint of a segment of constant
*perpendicular* width), giving tilt-invariant overlap — used for both the loss and for
confidence/assignment. Enable with ``tilt=True``.

Everything here works on x's in **pixel** units at shared, known y rows.
"""
from __future__ import annotations
import torch


def _radii(xs: torch.Tensor, sample_ys: torch.Tensor, r: float, tilt: bool) -> torch.Tensor:
    """Per-point horizontal radius. Constant ``r`` for LineIoU; slope-scaled for LaneIoU.

    xs:        (..., N) x at each of N sample rows (pixels)
    sample_ys: (N,)     the row y positions (pixels), strictly monotonic
    returns:   (..., N) radius per point
    """
    if not tilt:
        return xs.new_full(xs.shape, r)
    # local slope dx/dy via central differences on the (target) polyline
    dx = torch.zeros_like(xs)
    dx[..., 1:-1] = xs[..., 2:] - xs[..., :-2]
    dx[..., 0] = xs[..., 1] - xs[..., 0]
    dx[..., -1] = xs[..., -1] - xs[..., -2]
    dy = torch.zeros_like(xs)
    dy[..., 1:-1] = sample_ys[2:] - sample_ys[:-2]
    dy[..., 0] = sample_ys[1] - sample_ys[0]
    dy[..., -1] = sample_ys[-1] - sample_ys[-2]
    slope = dx / (dy + 1e-6)
    return r * torch.sqrt(1.0 + slope * slope)


def line_iou(pred_xs: torch.Tensor, target_xs: torch.Tensor, sample_ys: torch.Tensor,
             r: float = 7.5, valid: torch.Tensor | None = None, tilt: bool = False,
             eps: float = 1e-6) -> torch.Tensor:
    """Row-summed interval IoU between predicted and target lane x's.

    pred_xs, target_xs: (M, N) matched pairs of per-row x (pixels)
    sample_ys:          (N,)   row y positions (pixels)
    valid:              (M, N) bool mask of rows where the target is defined (else all)
    r:                  segment radius in pixels
    tilt:               LaneIoU (slope-scaled radius) vs LineIoU (constant)
    returns:            (M,) IoU per lane in [~-inf, 1]; use ``1 - line_iou`` as loss
    """
    rp = _radii(pred_xs, sample_ys, r, tilt)
    rt = _radii(target_xs, sample_ys, r, tilt)
    px1, px2 = pred_xs - rp, pred_xs + rp
    tx1, tx2 = target_xs - rt, target_xs + rt
    overlap = torch.min(px2, tx2) - torch.max(px1, tx1)
    union = torch.max(px2, tx2) - torch.min(px1, tx1)
    if valid is not None:
        overlap = overlap * valid
        union = union * valid + (~valid) * eps        # avoid inflating union on invalid rows
    return overlap.sum(dim=-1) / (union.sum(dim=-1) + eps)


def line_iou_loss(pred_xs, target_xs, sample_ys, r=7.5, valid=None, tilt=False):
    """1 − LineIoU/LaneIoU, averaged over the M matched lanes (scalar)."""
    if pred_xs.numel() == 0:
        return pred_xs.sum() * 0.0
    return (1.0 - line_iou(pred_xs, target_xs, sample_ys, r=r, valid=valid, tilt=tilt)).mean()


def line_iou_cost(pred_xs: torch.Tensor, target_xs: torch.Tensor, sample_ys: torch.Tensor,
                  r: float = 7.5, tilt: bool = False) -> torch.Tensor:
    """Pairwise (P, G) IoU cost matrix for label assignment — every prior vs every GT.

    A vectorised outer version of :func:`line_iou`: broadcasts P predictions against G
    targets. Returns IoU (higher = better match); the assigner turns it into a cost.
    """
    P, N = pred_xs.shape
    G = target_xs.shape[0]
    rp = _radii(pred_xs, sample_ys, r, tilt).unsqueeze(1)        # (P,1,N)
    rt = _radii(target_xs, sample_ys, r, tilt).unsqueeze(0)      # (1,G,N)
    px1 = (pred_xs.unsqueeze(1) - rp)
    px2 = (pred_xs.unsqueeze(1) + rp)
    tx1 = (target_xs.unsqueeze(0) - rt)
    tx2 = (target_xs.unsqueeze(0) + rt)
    valid = (target_xs >= 0).unsqueeze(0).expand(P, G, N)       # (P,G,N)
    overlap = (torch.min(px2, tx2) - torch.max(px1, tx1)) * valid
    union = (torch.max(px2, tx2) - torch.min(px1, tx1)) * valid + (~valid) * 1e-6
    return overlap.sum(-1) / (union.sum(-1) + 1e-6)             # (P,G)
