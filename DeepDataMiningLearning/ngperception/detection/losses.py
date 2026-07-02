"""
ngperception.detection.losses
==============================

Detection losses harvested from `2D3DFusion/mydetector3d/utils/loss_utils.py` (OpenPCDet
lineage), cleaned to be device-agnostic (no hardcoded `.cuda()`): sigmoid **focal**
classification loss + code-weighted **smooth-L1** box-regression loss.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SigmoidFocalClassificationLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    @staticmethod
    def _bce_with_logits(x, target):
        return torch.clamp(x, min=0) - x * target + torch.log1p(torch.exp(-torch.abs(x)))

    def forward(self, pred, target, weights):
        """pred/target: (B,N,C) logits/one-hot; weights: (B,N)."""
        p = torch.sigmoid(pred)
        alpha_w = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - p) + (1.0 - target) * p
        focal_w = alpha_w * torch.pow(pt, self.gamma)
        loss = focal_w * self._bce_with_logits(pred, target)
        if weights.dim() == 2:
            weights = weights.unsqueeze(-1)
        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta: float = 1.0 / 9.0, code_weights=None):
        super().__init__()
        self.beta = beta
        self.code_weights = None if code_weights is None else torch.tensor(code_weights, dtype=torch.float32)

    def forward(self, pred, target, weights=None):
        """pred/target: (B,N,code); weights: (B,N)."""
        target = torch.where(torch.isnan(target), pred, target)     # ignore nan targets
        diff = pred - target
        if self.code_weights is not None:
            diff = diff * self.code_weights.to(diff.device).view(1, 1, -1)
        n = torch.abs(diff)
        loss = torch.where(n < self.beta, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        return loss
