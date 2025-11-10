# projects/bevdet/bevfusion_with_aux.py
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import inspect
from mmdet3d.registry import MODELS
from projects.bevdet.bevfusion.bevfusion import BEVFusion  # 复用你现有的 BEVFusion
#from projects.bevdet.painting_context import set_painting_context, clear_painting_context

def _draw_gaussian(heatmap: torch.Tensor, center: Tuple[int, int], radius: int) -> None:
    """In-place draw a 2D Gaussian on heatmap (HxW)."""
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
    """Try to extract LiDAR-frame centers (x,y) from a Det3DDataSample."""
    # mmdet3d 不同版本字段命名略有差异，这里做兼容
    b = None
    if hasattr(data_sample, 'gt_instances_3d') and data_sample.gt_instances_3d is not None:
        gi = data_sample.gt_instances_3d
        if hasattr(gi, 'bboxes_3d') and gi.bboxes_3d is not None:
            b = gi.bboxes_3d
            if hasattr(b, 'tensor'):
                t = b.tensor  # [N, 7] 或 [N, …], 前两位 x,y
            else:
                t = torch.as_tensor(b, device='cpu')  # 兜底
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
    """Prefer 'lidar2img' ([Nc,4,4]); if missing, build from 'lidar2cam' + ('cam2img' or 'ori_cam2img').
    Returns: [B, Nc, 4, 4] float32 on device.
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
    """Convert a list/ndarray of Nc matrices to a tensor [Nc, a, b] on device (float32).
    Use np.asarray first to avoid the slow path warning in torch.as_tensor(list-of-ndarray)."""
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
    """
    Stack per-sample, per-view matrices into [B, Nc, a, b].
    `keys` is a tuple of alternatives (e.g., ('cam2img','ori_cam2img')).
    If none present and default_eye is given, build identity per view.
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
    """Prefer 'cam2lidar'; else invert 'lidar2cam'. Return [B, Nc, 4, 4] on device."""
    mats_B = []
    for m in meta_list:
        if 'cam2lidar' in m:
            mats_B.append(_as_tensor_mat_list(m['cam2lidar'], device, expect_shape_last=(4, 4)))
        elif 'lidar2cam' in m:
            L2C = _as_tensor_mat_list(m['lidar2cam'], device, expect_shape_last=(4, 4))
            C2L = torch.linalg.inv(L2C)  # [Nc,4,4]
            mats_B.append(C2L)
        else:
            raise KeyError("Need 'cam2lidar' or 'lidar2cam' in metainfo.")
    return torch.stack(mats_B, dim=0)  # [B, Nc, 4, 4]

def _get_imgs_tensor(inputs):
    """Return a 5D image tensor [B, Nc, 3, H, W] from various possible layouts.
    Tries keys: 'img', 'imgs', and nested 'inputs' dict. Stacks list/tuple if needed.
    Raises a clear error if not found or wrong rank.
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
    """
    Robustly call `extract_img_feat` across different upstream signatures:
    Try two common orders:
      A) (img, points, cam_intr, cam2lid, img_aug, lidar_aug, img_metas)
      B) (img, points, cam_intr, cam2lid, img_metas, img_aug, lidar_aug)
    Fallback: (img, points, img_metas) for older minimal signatures.
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
                # 把最先抛出的错误冒泡，便于定位
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
            oc = getattr(self.view_transform, 'out_channels', 64)  # default
            self.img_aux_head = nn.Sequential(
                nn.Conv2d(oc, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1)
            )
            self.aux_weight: float = float(aux_cfg.get('loss_weight', 0.2))
            # gaussian半径（以 BEV cell 为单位）；也可根据 box 尺寸自适应，这里提供固定半径的稳健默认
            self.aux_radius_cells: int = int(aux_cfg.get('radius_cells', 2))
            self.aux_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

            # 记录 BEV 网格边界用于 (x,y)->(iy,ix) 映射
            self._xbound = tuple(getattr(self.view_transform, 'xbound'))
            self._ybound = tuple(getattr(self.view_transform, 'ybound'))
            self._downsample = int(getattr(self.view_transform, 'downsample', 1))

    @torch.no_grad()
    def _build_aux_target(self, data_samples: List, H: int, W: int, device) -> torch.Tensor:
        """Build a class-agnostic BEV center heatmap target. Shape [B,1,H,W]."""
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

            # 映射到BEV网格索引（iy, ix）
            ix = torch.floor((centers_xy[:, 0] - xmin) / dx_eff).long()
            iy = torch.floor((centers_xy[:, 1] - ymin) / dy_eff).long()
            # 过滤越界
            valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            ix = ix[valid].tolist()
            iy = iy[valid].tolist()

            for (yy, xx) in zip(iy, ix):
                _draw_gaussian(target[b, 0], (yy, xx), radius=self.aux_radius_cells)

        return target

    def loss(self, inputs: Dict, data_samples: List, **kwargs) -> Dict:
        """Reuse upstream BEVFusion.loss and add an auxiliary loss on image BEV.

        We register a forward hook on `self.view_transform` to capture the image
        BEV feature when BEVFusion runs its normal pipeline internally. This
        avoids calling `extract_img_feat` / `extract_pts_feat` ourselves and thus
        side-steps signature mismatches across branches.
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

        # 3) Add AUX loss on the captured image BEV
        if self.aux_on:
            img_bev = captured.get('img_bev', None) #[4, 64, 180, 180]
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
    
    # ---- replace your BEVFusionWithAux.loss(...) with this version ----
    def loss_old(self, inputs: Dict, data_samples: List, **kwargs) -> Dict:
        """Same as BEVFusion.loss, plus an auxiliary image-BEV heatmap loss."""
        # 0) Prepare common pieces
        batch_input_metas = [item.metainfo for item in data_samples]
        #imgs   = inputs.get('img', None)
        # 0) Robustly get image tensor [B, Nc, 3, H, W]
        imgs = _get_imgs_tensor(inputs) #imgs key
        points = inputs.get('points', None)

        # # device for meta tensors
        # device = None
        # if imgs is not None and isinstance(imgs, torch.Tensor):
        #     device = imgs.device
        # elif isinstance(points, (list, tuple)) and len(points) > 0 and isinstance(points[0], torch.Tensor):
        #     device = points[0].device
        # else:
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device 推断：优先用图像张量的 device
        device = imgs.device if isinstance(imgs, torch.Tensor) else (
            points[0].device if isinstance(points, (list, tuple)) and len(points) > 0 and isinstance(points[0], torch.Tensor)
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # device 的推断沿用你已有代码
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

        # ⚠️ 按你打印出来的真实签名严丝合缝地传 8 个“位置参数”：
        # (x, points, lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix, img_metas)
        img_bev = self.extract_img_feat(
            imgs,
            deepcopy(points),
            lidar2imag,   # [B,Nc,4,4]
            cam_intr,     # [B,Nc,3,3]
            cam2lid,      # [B,Nc,4,4]
            img_aug,      # [B,Nc,3,3]
            lidar_aug,    # [B,Nc,4,4]
            batch_input_metas   # ← 第8个：img_metas
        )#bev feature: [4, 64, 180, 180]

        # 3) LiDAR branch BEV (original API usually takes just points + metas)
        #pts_bev = self.extract_pts_feat(points, batch_input_metas)  # -> [B, C_pts, Hy, Hx]
        try:
            pts_bev = self.extract_pts_feat(points) #points is batch len list, each one is [Ni,5]points, 5 means: xyz,intensity,time_lag
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