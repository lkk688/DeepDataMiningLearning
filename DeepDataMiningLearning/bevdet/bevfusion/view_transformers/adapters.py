# projects/bevdet/bevfusion/view_transformers/adapters.py
from __future__ import annotations
from typing import List, Dict, Sequence, Union
import torch

__all__ = ['call_view_transform', 'call_view_transform_lss']

# ---------- helpers ----------
def _infer_num_cams_from_metas(metas: List[Dict]) -> int:
    m0 = metas[0]
    for k in ('cam2img', 'ori_cam2img', 'lidar2img', 'ori_lidar2img'):
        if k in m0:
            return len(m0[k])
    raise RuntimeError('Cannot infer num_cams from metas')

def _ensure_BNCHW(x: torch.Tensor, metas: List[Dict]) -> torch.Tensor:
    """
    Normalize feature tensor to shape (B, N, C, H, W).
    Accepts 4D (B*N, C, H, W) or 5D with N in dim1 or dim2.
    """
    if x.dim() == 5:
        B, d1, d2, H, W = x.shape
        N = _infer_num_cams_from_metas(metas)
        if d1 == N:             # (B, N, C, H, W)
            return x
        if d2 == N:             # (B, C, N, H, W) -> (B, N, C, H, W)
            return x.permute(0, 2, 1, 3, 4).contiguous()
        raise AssertionError(f'Unexpected 5D shape {tuple(x.shape)} with N={N}')
    elif x.dim() == 4:
        BN, C, H, W = x.shape
        N = _infer_num_cams_from_metas(metas)
        assert BN % N == 0, f'Cannot split BN={BN} by N={N}'
        B = BN // N
        return x.view(B, N, C, H, W).contiguous()
    else:
        raise AssertionError(f'Expected 4D/5D, got {x.dim()}D: {tuple(x.shape)}')

def _to_layout(x_bnchw: torch.Tensor, expected_layout: str) -> torch.Tensor:
    """BNCHW -> expected layout used by VT; supports 'BNCHW' or 'BCNHW'."""
    if expected_layout.upper() == 'BCNHW':
        # (B, N, C, H, W) -> (B, C, N, H, W)
        return x_bnchw.permute(0, 2, 1, 3, 4).contiguous()
    return x_bnchw  # BNCHW

def _stack_metas_4x4(metas: List[Dict], key: str, device, dtype=torch.float32) -> torch.Tensor:
    """
    Stack a per-sample list of Nv 4x4 matrices into (B, Nv, 4, 4).
    metas[i][key] is typically a list/ndarray of length Nv.
    """
    return torch.stack(
        [torch.stack([torch.as_tensor(M, device=device, dtype=dtype) for M in m[key]], dim=0)
         for m in metas],
        dim=0
    )

# ---------- public APIs ----------
def call_view_transform(
    vt,
    x: Union[torch.Tensor, Sequence[torch.Tensor]],
    metas: List[Dict]
):
    """
    Generic adapter for view-transform modules that accept:
      - a list of multi-level feature maps, each shaped either (B, N, C, H, W)
        or (B, C, N, H, W) depending on vt.expected_layout,
      - and `metas` for geometry.
    """
    expected = getattr(vt, 'expected_layout', 'BNCHW')

    if isinstance(x, (list, tuple)):
        lvls = []
        for xi in x:
            x_bnchw = _ensure_BNCHW(xi, metas)
            lvls.append(_to_layout(x_bnchw, expected))
        # set vt.num_cams if present
        N = lvls[0].shape[2] if expected.upper() == 'BCNHW' else lvls[0].shape[1]
        try:
            vt.num_cams = N
        except Exception:
            pass
        return vt(lvls, metas)
    else:
        x_bnchw = _ensure_BNCHW(x, metas)
        x_for_vt = _to_layout(x_bnchw, expected)
        N = x_for_vt.shape[2] if expected.upper() == 'BCNHW' else x_for_vt.shape[1]
        try:
            vt.num_cams = N
        except Exception:
            pass
        # wrap as 1-level list for a unified VT call
        return vt([x_for_vt], metas)

def call_view_transform_lss(
    vt,
    x: Union[torch.Tensor, Sequence[torch.Tensor]],
    metas: List[Dict],
    points=None,
    level: int = 0,
):
    """
    Adapter for DepthLSSTransform-style modules whose forward signature is:
        vt(img_feats(B,N,C,H,W), points, lidar2img, cam_intr, cam2lidar, img_aug, lidar_aug, metas)
    It picks one level from the (possibly) multi-level image features.
    """
    # pick a single level tensor
    xi = x[level] if isinstance(x, (list, tuple)) else x
    x_bnchw = _ensure_BNCHW(xi, metas)       # (B, N, C, H, W)
    B, N, C, H, W = x_bnchw.shape
    device = x_bnchw.device

    # stack geometry
    lidar2img = _stack_metas_4x4(metas, 'lidar2img', device)
    cam_intr  = _stack_metas_4x4(metas, 'cam2img', device)
    cam2lidar = _stack_metas_4x4(metas, 'cam2lidar', device)

    # aug matrices (default eye if missing)
    eye = torch.eye(4, device=device)
    img_aug  = torch.stack([torch.as_tensor(m.get('img_aug_matrix',  eye), device=device, dtype=torch.float32) for m in metas], dim=0)
    lidar_aug= torch.stack([torch.as_tensor(m.get('lidar_aug_matrix',eye), device=device, dtype=torch.float32) for m in metas], dim=0)

    # call DepthLSS
    return vt(x_bnchw, points, lidar2img, cam_intr, cam2lidar, img_aug, lidar_aug, metas)