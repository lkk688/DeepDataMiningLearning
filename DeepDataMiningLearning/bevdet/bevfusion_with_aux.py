# projects/bevdet/bevfusion_with_aux.py
#
# Auxiliary-supervised variant of BEVFusion. Adds an image-BEV head that
# predicts a class-agnostic center heatmap and combines it with the main
# detection loss. This file also includes robust helpers for extracting
# multi-view image tensors and stacking calibration/augmentation matrices.
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import inspect
"""This module provides an auxiliary-supervised variant of BEVFusion.

Key additions:
- An image-BEV auxiliary head that predicts a class-agnostic center heatmap.
- Robust utilities to extract multi-view image tensors and stack calibration
  and augmentation matrices from metainfo.
"""

# Import the mmdet3d registry with a safe fallback so editors and static
# analysis can resolve references even outside the mmdet3d runtime.
try:
    from mmdet3d.registry import MODELS
except Exception:
    class _DummyRegistry:
        def register_module(self, *args, **kwargs):
            def decorator(cls):
                return cls
            return decorator

        def build(self, *args, **kwargs):
            return None

    MODELS = _DummyRegistry()

# Import the base BEVFusion implementation. Prefer the projects path used by
# mmdet3d custom imports; fall back to local package import. As a last resort,
# define a minimal stub to keep linters quiet in non-runtime contexts.
try:
    from projects.bevdet.bevfusion.bevfusion import BEVFusion
except Exception:
    try:
        from bevdet.bevfusion.bevfusion import BEVFusion
    except Exception:
        class BEVFusion(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
# from projects.bevdet.painting_context import set_painting_context, clear_painting_context
# (Optional painting-context utilities; kept disabled to avoid side effects.)

def _draw_gaussian(heatmap: torch.Tensor, center: Tuple[int, int], radius: int) -> None:
    """In-place draw a 2D Gaussian on a heatmap of shape (H, W).

    The Gaussian is centered at `center` with a radius controlling its spread.
    Values are blended with `torch.maximum` to avoid overwriting stronger peaks.
    """
    y, x = int(center[0]), int(center[1])
    H, W = heatmap.shape[-2], heatmap.shape[-1]
    diameter = 2 * radius + 1
    # build gaussian kernel
    sigma = diameter / 6.0
    yy = torch.arange(0, diameter, device=heatmap.device).view(-1, 1).float()
    xx = torch.arange(0, diameter, device=heatmap.device).view(1, -1).float()
    g = torch.exp(-((yy - radius) ** 2 + (xx - radius) ** 2) / (2 * sigma ** 2))

    top, bottom = max(0, y - radius), min(H, y + radius + 1)
    left, right = max(0, x - radius), min(W, x + radius + 1)

    g_top = max(0, radius - y)
    g_left = max(0, radius - x)
    g_bottom = g_top + (bottom - top)
    g_right = g_left + (right - left)

    if top < bottom and left < right:
        heatmap[..., top:bottom, left:right] = torch.maximum(
            heatmap[..., top:bottom, left:right],
            g[g_top:g_bottom, g_left:g_right]
        )

def _boxes_center_xy(data_sample) -> Optional[torch.Tensor]:
    """Extract LiDAR-frame centers (x, y) from a Det3DDataSample if present.

    Different mmdet3d versions use slightly different field names. This helper
    handles common variants so the auxiliary target generation remains robust.
    Returns a tensor of shape [N, 2] or None if centers are unavailable.
    """
    # Different mmdet3d versions use slightly different field names; handle
    # multiple variants to stay compatible across releases.
    b = None
    if hasattr(data_sample, 'gt_instances_3d') and data_sample.gt_instances_3d is not None:
        gi = data_sample.gt_instances_3d
        if hasattr(gi, 'bboxes_3d') and gi.bboxes_3d is not None:
            b = gi.bboxes_3d
            if hasattr(b, 'tensor'):
                t = b.tensor  # [N, 7] or similar; first two entries are x, y
            else:
                t = torch.as_tensor(b, device='cpu')  # fallback
            return t[..., :2]
    if hasattr(data_sample, 'gt_bboxes_3d') and data_sample.gt_bboxes_3d is not None:
        b = data_sample.gt_bboxes_3d
        if hasattr(b, 'tensor'):
            t = b.tensor
        else:
            t = torch.as_tensor(b, device='cpu')
        return t[..., :2]
    return None

# ---------- helpers: stack meta matrices safely & fast ----------

def _stack_lidar2image(meta_list, device):
    """Stack per-sample lidar-to-image matrices into a batched tensor.

    Prefer the direct `lidar2img` matrices ([Nc, 4, 4]). If missing, construct
    them via `lidar2cam` and either `cam2img` or `ori_cam2img` by embedding K
    into a 4x4 projection matrix. Returns [B, Nc, 4, 4] float32 on `device`.
    """
    mats_B = []
    for m in meta_list:
        if 'lidar2img' in m:
            t = _as_tensor_mat_list(m['lidar2img'], device, expect_shape_last=(4, 4))
            mats_B.append(t)
            continue

        # Fallback: L2I = P @ L2C, where P is a 4x4 with K in top-left 3x3
        if 'lidar2cam' in m and ('cam2img' in m or 'ori_cam2img' in m):
            L2C = _as_tensor_mat_list(m['lidar2cam'], device, expect_shape_last=(4, 4))  # [Nc,4,4]
            Kkey = 'cam2img' if 'cam2img' in m else 'ori_cam2img'
            K   = _as_tensor_mat_list(m[Kkey], device, expect_shape_last=(3, 3))         # [Nc,3,3]
            Nc  = L2C.shape[0]
            P   = torch.zeros((Nc, 4, 4), device=device, dtype=torch.float32)
            P[..., :3, :3] = K
            P[..., 3, 3]   = 1.0
            L2I = torch.matmul(P, L2C)  # [Nc,4,4]
            mats_B.append(L2I)
            continue

        raise KeyError("Need 'lidar2img' or ('lidar2cam' + 'cam2img/ori_cam2img') in metainfo.")
    return torch.stack(mats_B, dim=0)  # [B, Nc, 4, 4]

def _as_tensor_mat_list(mats, device, expect_shape_last=None):
    """Convert a list/ndarray of Nc matrices to a tensor [Nc, a, b].

    Uses `np.asarray` first to avoid PyTorch's slow path when converting a
    list of ndarrays. Optionally pads/crops between 3x3 and 4x4 to match
    `expect_shape_last`.
    """
    if isinstance(mats, (list, tuple)):
        arr = np.asarray(mats, dtype=np.float32)       # fast path, no PyTorch warning
        t = torch.from_numpy(arr).to(device=device)    # [Nc, a, b], float32
    else:
        t = torch.as_tensor(mats, dtype=torch.float32, device=device)

    if expect_shape_last is not None and t.shape[-2:] != expect_shape_last:
        # simple-safe pad/crop between 3x3 and 4x4
        if t.shape[-2:] == (3, 3) and expect_shape_last == (4, 4):
            pad = torch.zeros((t.shape[0], 4, 4), device=device, dtype=torch.float32)
            pad[..., :3, :3] = t
            pad[..., 3, 3] = 1.0
            t = pad
        elif t.shape[-2:] == (4, 4) and expect_shape_last == (3, 3):
            t = t[..., :3, :3]
        else:
            raise AssertionError(f'bad mat shape {t.shape[-2:]}, expect {expect_shape_last}')
    return t

def _stack_meta_mats(meta_list, keys, device, expect_shape_last=None, default_eye=None):
    """Stack per-sample, per-view matrices into [B, Nc, a, b].

    - `keys`: tuple of acceptable dictionary keys (e.g. ('cam2img','ori_cam2img')).
    - If none of the keys are found and `default_eye` is provided, builds an
      identity matrix per view using the inferred number of cameras.
    """
    mats_B = []
    for m in meta_list:
        found = None
        for k in keys:
            if k in m:
                found = k
                break
        if found is None:
            if default_eye is None:
                raise KeyError(f"None of keys {keys} found in metainfo.")
            # infer Nc from any known per-view entry
            Nc = None
            for cand in ('cam2img', 'ori_cam2img', 'lidar2img', 'lidar2cam', 'cam2lidar'):
                if cand in m:
                    Nc = len(m[cand])
                    break
            assert Nc is not None, "Cannot infer Nc to build default identity matrices."
            eye = torch.eye(default_eye, dtype=torch.float32, device=device).unsqueeze(0).repeat(Nc, 1, 1)
            mats_B.append(eye)
        else:
            mats_B.append(_as_tensor_mat_list(m[found], device, expect_shape_last))
    return torch.stack(mats_B, dim=0)  # [B, Nc, a, b]

def _stack_cam2lidar(meta_list, device):
    """Stack camera-to-lidar matrices, inverting lidar-to-camera when needed.

    Prefers `cam2lidar` if available; otherwise computes its inverse from
    `lidar2cam`. Returns [B, Nc, 4, 4] float32 on `device`.
    """
    mats_B = []
    for m in meta_list:
        if 'cam2lidar' in m:
            mats_B.append(_as_tensor_mat_list(m['cam2lidar'], device, expect_shape_last=(4, 4)))
        elif 'lidar2cam' in m:
            L2C = _as_tensor_mat_list(m['lidar2cam'], device, expect_shape_last=(4, 4))
            # Use torch.inverse for broader compatibility across environments.
            C2L = torch.inverse(L2C)  # [Nc,4,4]
            mats_B.append(C2L)
        else:
            raise KeyError("Need 'cam2lidar' or 'lidar2cam' in metainfo.")
    return torch.stack(mats_B, dim=0)  # [B, Nc, 4, 4]

def _get_imgs_tensor(inputs):
    """Return a 5D image tensor [B, Nc, 3, H, W] from common input layouts.

    Tries keys 'img' and 'imgs' at the top level, then within a nested 'inputs'
    dict. If a list/tuple of tensors is found, stacks along the batch dim.
    Raises informative errors if not found or the rank is incorrect.
    """
    cand = None
    # direct keys
    if isinstance(inputs, dict):
        for k in ('img', 'imgs'):
            if k in inputs and inputs[k] is not None:
                cand = inputs[k]
                break
        # nested under 'inputs'
        if cand is None and 'inputs' in inputs and isinstance(inputs['inputs'], dict):
            inner = inputs['inputs']
            for k in ('img', 'imgs'):
                if k in inner and inner[k] is not None:
                    cand = inner[k]
                    break

    # stack list/tuple to tensor
    if isinstance(cand, (list, tuple)):
        # typical collate gives Tensor already; but be robust
        if len(cand) == 0:
            cand = None
        elif isinstance(cand[0], torch.Tensor):
            # guess batch dimension
            cand = torch.stack(cand, dim=0)
        else:
            # list of per-sample dicts etc. -> cannot handle here
            cand = None

    if cand is None:
        raise RuntimeError(
            "No multi-view image tensor found in `inputs`. "
            "Tried keys: ['img','imgs'] and nested under 'inputs'. "
            f"Available top-level keys: {list(inputs.keys()) if isinstance(inputs, dict) else type(inputs)}. "
            "Please ensure your pipeline includes BEVLoadMultiViewImageFromFiles "
            "and Pack3DDetInputs(keys=['img', ...])."
        )

    if not isinstance(cand, torch.Tensor):
        raise TypeError(f"`img` is not a Tensor: got {type(cand)}")
    if cand.ndim != 5:
        raise ValueError(f"`img` must be 5D [B,Nc,3,H,W], got shape {tuple(cand.shape)}")

    return cand

def _call_extract_img_feat(fn, imgs, points, cam_intr, cam2lid, img_aug, lidar_aug, img_metas):
    """Call `extract_img_feat` robustly across known upstream signatures.

    Tries two common orders:
      A) (img, points, cam_intr, cam2lid, img_aug, lidar_aug, img_metas)
      B) (img, points, cam_intr, cam2lid, img_metas, img_aug, lidar_aug)
    Falls back to (img, points, img_metas) for older minimal signatures.
    """
    # A) aug mats before metas
    try:
        return fn(imgs, deepcopy(points), cam_intr, cam2lid, img_aug, lidar_aug, img_metas)
    except TypeError as eA:
        # B) metas before aug mats
        try:
            return fn(imgs, deepcopy(points), cam_intr, cam2lid, img_metas, img_aug, lidar_aug)
        except TypeError as eB:
            # minimal fallback (rare)
            try:
                return fn(imgs, deepcopy(points), img_metas)
            except TypeError:
                print(f"[extract_img_feat] signature: {inspect.signature(fn)}")
                # Bubble up the first error to aid diagnosis.
                raise eA
            
@MODELS.register_module()
class BEVFusionWithAux(BEVFusion):
    """BEVFusion + an auxiliary supervision on the image BEV branch.

    Aux head predicts a BEV center heatmap (class-agnostic) at the same spatial
    size as the image BEV feature produced by `view_transform`. Targets are
    generated from GT 3D boxes by projecting (x,y) to BEV grid indices.

    Config example (see below):
      model = dict(
          type='BEVFusionWithAux',
          aux_cfg=dict(loss_weight=0.2, radius_cells=2)
      )
    """

    def __init__(self, aux_cfg: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.aux_on = aux_cfg is not None
        if self.aux_on:
            # The auxiliary head operates on the image BEV produced by the
            # view transformer. We keep it lightweight: a small conv tower
            # ending in a single-channel heatmap.
            oc = getattr(self.view_transform, 'out_channels', 64)  # default
            self.img_aux_head = nn.Sequential(
                nn.Conv2d(oc, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1)
            )
            self.aux_weight: float = float(aux_cfg.get('loss_weight', 0.2))
            # Gaussian radius measured in BEV cells. This could be made
            # adaptive to box size; we provide a robust fixed default here.
            self.aux_radius_cells: int = int(aux_cfg.get('radius_cells', 2))
            self.aux_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

            # Cache BEV grid bounds used to map LiDAR (x,y) -> grid (iy, ix)
            self._xbound = tuple(getattr(self.view_transform, 'xbound'))
            self._ybound = tuple(getattr(self.view_transform, 'ybound'))
            self._downsample = int(getattr(self.view_transform, 'downsample', 1))

    @torch.no_grad()
    def _build_aux_target(self, data_samples: List, H: int, W: int, device) -> torch.Tensor:
        """Build a class-agnostic BEV center heatmap target. Shape [B, 1, H, W].

        Converts ground-truth box centers from LiDAR coordinates to BEV grid
        indices using the `xbound`/`ybound` and the effective resolution.
        """
        xmin, xmax, dx = self._xbound
        ymin, ymax, dy = self._ybound
        dx_eff = dx * self._downsample
        dy_eff = dy * self._downsample

        target = torch.zeros((len(data_samples), 1, H, W), device=device, dtype=torch.float32)

        for b, ds in enumerate(data_samples):
            centers_xy = _boxes_center_xy(ds)  # [N,2] in LiDAR frame
            if centers_xy is None:
                continue
            if centers_xy.device != device:
                centers_xy = centers_xy.to(device)

            # Map to BEV grid indices (iy, ix)
            ix = torch.floor((centers_xy[:, 0] - xmin) / dx_eff).long()
            iy = torch.floor((centers_xy[:, 1] - ymin) / dy_eff).long()
            # Filter out-of-bounds indices
            valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            ix = ix[valid].tolist()
            iy = iy[valid].tolist()

            for (yy, xx) in zip(iy, ix):
                _draw_gaussian(target[b, 0], (yy, xx), radius=self.aux_radius_cells)

        return target

    def loss(self, inputs: Dict, data_samples: List, **kwargs) -> Dict:
        """Compute the standard BEVFusion loss plus an auxiliary image-BEV loss.

        A forward hook is registered on `self.view_transform` to capture its
        output (the image BEV feature). This avoids re-calling the upstream
        feature extractors and keeps compatibility across versions.
        """
        # 1) Hook to capture image BEV (output of view_transform)
        captured = {}

        def _vt_hook(module, in_tup, out):
            # out is expected to be Tensor [B, C, Hy, Hx]
            captured['img_bev'] = out

        handle = self.view_transform.register_forward_hook(_vt_hook)

        # 2) Run the original loss (image & lidar branches + fusion + head)
        try:
            losses = super().loss(inputs, data_samples, **kwargs)
        finally:
            handle.remove()

        # 3) Add auxiliary heatmap loss on the captured image BEV
        if self.aux_on:
            img_bev = captured.get('img_bev', None)  # e.g., [B, 64, 180, 180]
            if img_bev is None:
                raise RuntimeError(
                    'AUX: view_transform output was not captured. '
                    'Please ensure the upstream pipeline calls `self.view_transform`.'
                )

            # shape [B, C, H, W]
            B, C, H, W = img_bev.shape

            pred = self.img_aux_head(img_bev)  # [B,1,H,W]
            target = self._build_aux_target(data_samples, H, W, img_bev.device)  # [B,1,H,W]
            aux_loss = self.aux_loss_fn(pred, target)
            losses['loss_aux_img_bev'] = aux_loss * float(self.aux_weight)

        return losses
    
    # ---- Legacy variant kept for reference/testing with explicit calls ----
    def loss_old(self, inputs: Dict, data_samples: List, **kwargs) -> Dict:
        """Legacy variant mirroring BEVFusion.loss with explicit calls.

        Kept for reference and testing; computes the auxiliary heatmap loss
        using explicitly extracted image and point features.
        """
        # 0) Prepare common pieces
        batch_input_metas = [item.metainfo for item in data_samples]
        # Robustly get image tensor [B, Nc, 3, H, W]
        imgs = _get_imgs_tensor(inputs)  # image tensor extracted from inputs
        points = inputs.get('points', None)

        # Device inference: prefer the device of the image tensor; otherwise
        # fall back to points' device or CUDA if available.
        device = imgs.device if isinstance(imgs, torch.Tensor) else (
            points[0].device if isinstance(points, (list, tuple)) and len(points) > 0 and isinstance(points[0], torch.Tensor)
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Stack per-view matrices with shape checks and safe padding/cropping.
        cam_intr   = _stack_meta_mats(batch_input_metas, ('cam2img','ori_cam2img'), device, expect_shape_last=(3, 3))
        cam2lid    = _stack_cam2lidar(batch_input_metas, device)                 # [B,Nc,4,4]
        img_aug    = _stack_meta_mats(batch_input_metas, ('img_aug_matrix',), device, expect_shape_last=(3, 3), default_eye=3)
        lidar_aug  = _stack_meta_mats(batch_input_metas, ('lidar_aug_matrix',), device, expect_shape_last=(4, 4), default_eye=4)
        lidar2imag = _stack_lidar2image(batch_input_metas, device)               # [B,Nc,4,4]

        assert imgs.ndim == 5, f"imgs shape={tuple(imgs.shape)}"
        assert cam_intr.shape[-2:]  == (3,3)
        assert img_aug.shape[-2:]   == (3,3)
        assert cam2lid.shape[-2:]   == (4,4)
        assert lidar_aug.shape[-2:] == (4,4)
        assert lidar2imag.shape[-2:] == (4,4)

        # Pass exactly eight positional arguments to match the printed
        # `extract_img_feat` signature from your environment:
        # (img, points, lidar2image, camera_intrinsics, camera2lidar,
        #  img_aug_matrix, lidar_aug_matrix, img_metas)
        img_bev = self.extract_img_feat(
            imgs,
            deepcopy(points),
            lidar2imag,   # [B,Nc,4,4]
            cam_intr,     # [B,Nc,3,3]
            cam2lid,      # [B,Nc,4,4]
            img_aug,      # [B,Nc,3,3]
            lidar_aug,    # [B,Nc,4,4]
            batch_input_metas   # 8th: image metas per batch
        )  # example BEV feature: [B, 64, Hy, Hx]

        # 3) LiDAR branch BEV (original API may take just points, or points+metas)
        # pts_bev = self.extract_pts_feat(points, batch_input_metas)  # -> [B, C_pts, Hy, Hx]
        try:
            # `points` is a list of length B; each element has shape [Ni, 5]
            # with columns: x, y, z, intensity, time_lag.
            pts_bev = self.extract_pts_feat(points)
        except TypeError:
            pts_bev = self.extract_pts_feat(points, batch_input_metas)
        
        # 4) Fuse and compute main detection loss (unchanged)
        fused = self.fusion_layer([img_bev, pts_bev])
        losses = self.bbox_head.loss(fused, data_samples, **kwargs)

        # 5) Auxiliary supervision on image BEV (optional)
        if self.aux_on:
            B, C, H, W = img_bev.shape
            pred = self.img_aux_head(img_bev)  # [B,1,H,W]
            target = self._build_aux_target(data_samples, H, W, img_bev.device)  # [B,1,H,W]
            aux_loss = self.aux_loss_fn(pred, target)
            losses['loss_aux_img_bev'] = aux_loss * float(self.aux_weight)

        return losses