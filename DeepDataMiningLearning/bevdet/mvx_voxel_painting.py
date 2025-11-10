from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from projects.bevdet.painting_context import get_painting_context

@MODELS.register_module()
class PaintedWrapperVFE(BaseModule):
    """Wrap an existing VFE (e.g., HardSimpleVFE/DynamicVFE) and optionally
    'paint' per-voxel features with multi-view image descriptors sampled from
    the FPN feature (stashed by a small pre-hook on view_transform).

    If the painting context is missing, this module simply returns the base
    VFE outputs unchanged. So enabling/disabling painting is non-intrusive.
    """

    def __init__(self,
                 base_vfe: dict,
                 point_cloud_range: List[float],
                 voxel_size: List[float],
                 image_size: List[int],     # [Himg, Wimg]
                 feature_size: List[int],   # [Hf, Wf]
                 img_feat_out: int = 32,
                 cam_pool: str = 'avg',     # 'avg' or 'max'
                 fuse: str = 'gated',       # 'gated'|'add'|'concat_linear'
                 detach_img: bool = True,
                 align_corners: bool = True,
                 chunk_voxels: int = 200000,
                 init_cfg=None,
                 **kwargs): # ★ absorb extra keys like num_features, norm_cfg...
        super().__init__(init_cfg)
        # --- absorb/remember extras (esp. num_features) ---
        # Some configs pass num_features at the top-level VFE cfg so that
        # downstream sparse encoders can read it at build time.
        declared_num_features = kwargs.pop('num_features', None)
        self._extra_cfg = kwargs  # keep anything else for debugging if needed

        self.base_vfe = MODELS.build(base_vfe)

        self.pc_range = point_cloud_range
        self.voxel_size = voxel_size
        self.Himg, self.Wimg = image_size
        self.Hf, self.Wf = feature_size
        assert cam_pool in ('avg', 'max')
        self.cam_pool, self.img_feat_out = cam_pool, int(img_feat_out)
        assert fuse in ('gated', 'add', 'concat_linear')
        self.fuse = fuse
        self.detach_img = detach_img
        self.align_corners = align_corners
        self.chunk = int(chunk_voxels)
        
        # Infer the output channels (C0) of the base VFE so that:
        # 1) we can expose self.num_features for downstream modules
        # 2) our painting fusion keeps the output dim = C0
        C0 = getattr(self.base_vfe, 'num_features', None)
        if C0 is None:
            C0 = getattr(self.base_vfe, 'out_channels', None)
        if C0 is None and hasattr(self.base_vfe, 'feat_channels'):
            fc = self.base_vfe.feat_channels
            if isinstance(fc, (list, tuple)) and len(fc) > 0:
                C0 = fc[-1]
            elif isinstance(fc, int):
                C0 = fc
        if C0 is None:
            C0 = declared_num_features  # last resort
        if C0 is None:
            raise ValueError('PaintedWrapperVFE: cannot infer base VFE output channels. '
                             'Please set base_vfe.feat_channels or pass num_features in config.')
        self.num_features = int(C0)   # ★ expose attribute for downstream encoders

        # Small heads (lazy-built when we know C0 and Cimg)
        self._proj = None
        self._fuse_add = None
        self._gate = None
        self._fuse_cat = None

    @torch.no_grad()
    def _voxel_centers(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """coords: [M,4] = [b,z,y,x] -> centers: [M,3] xyz in LiDAR"""
        b = coords[:, 0].long()
        zyx = coords[:, 1:].float()
        vx, vy, vz = self.voxel_size
        x_min, y_min, z_min = self.pc_range[:3]
        x = (zyx[:, 2] + 0.5) * vx + x_min
        y = (zyx[:, 1] + 0.5) * vy + y_min
        z = (zyx[:, 0] + 0.5) * vz + z_min
        return torch.stack([x, y, z], dim=-1).to(coords), b

    def _l2i(self, metas: List[dict], device) -> torch.Tensor:
        """Build lidar2image matrices; prefer 'lidar2img' else compose."""
        mats = []
        for m in metas:
            if 'lidar2img' in m:
                Li = torch.as_tensor(m['lidar2img'], dtype=torch.float32, device=device)
            else:
                L2C = torch.as_tensor(m['lidar2cam'], dtype=torch.float32, device=device)
                K = torch.as_tensor(m.get('cam2img', m.get('ori_cam2img')), dtype=torch.float32, device=device)
                Nc = L2C.shape[0]
                P = torch.zeros((Nc, 4, 4), device=device, dtype=torch.float32)
                P[..., :3, :3] = K
                P[..., 3, 3] = 1.0
                Li = P @ L2C
            mats.append(Li)
        return torch.stack(mats, dim=0)  # [B,Nc,4,4]

    def _ensure_heads(self, C0: int, Cimg: int, device):
        if self._proj is None:
            self._proj = nn.Sequential(
                nn.Linear(Cimg, self.img_feat_out, bias=False),
                nn.BatchNorm1d(self.img_feat_out),
                nn.ReLU(inplace=True)
            ).to(device)
        if self.fuse == 'concat_linear':
            if self._fuse_cat is None:
                self._fuse_cat = nn.Linear(C0 + self.img_feat_out, C0, bias=True).to(device)
        elif self.fuse == 'add':
            if self._fuse_add is None:
                self._fuse_add = nn.Linear(self.img_feat_out, C0, bias=False).to(device)
        else:  # gated
            if self._fuse_add is None:
                self._fuse_add = nn.Linear(self.img_feat_out, C0, bias=False).to(device)
            if self._gate is None:
                self._gate = nn.Sequential(
                    nn.Linear(self.img_feat_out, C0, bias=True),
                    nn.Sigmoid()
                ).to(device)

    def _sample_img_desc(self, fpn: torch.Tensor, metas: List[dict],
                         centers: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
        """fpn: [B,Nc,Cimg,Hf,Wf] -> per-voxel image desc [M,Cimg]"""
        device = fpn.device
        B, Nc, Cimg, Hf, Wf = fpn.shape
        L2I = self._l2i(metas, device)
        sx, sy = self.Wf / float(self.Wimg), self.Hf / float(self.Himg)

        desc = torch.zeros((centers.shape[0], Cimg), device=device, dtype=fpn.dtype)
        for st in range(0, centers.shape[0], self.chunk):
            ed = min(st + self.chunk, centers.shape[0])
            xyz = centers[st:ed]                 # [m,3]
            bids = batch_ids[st:ed].long()       # [m]
            ones = torch.ones((xyz.shape[0], 1), device=device, dtype=xyz.dtype)
            xyz1 = torch.cat([xyz, ones], dim=-1)  # [m,4]

            parts = []
            for b in range(B):
                sel = (bids == b)
                if not torch.any(sel):
                    continue
                xyz1_b = xyz1[sel]  # [mb,4]
                mb = xyz1_b.shape[0]
                pts = (L2I[b] @ xyz1_b.t().unsqueeze(0)).transpose(1, 2)  # [Nc,mb,4]
                u, v, w = pts[..., 0], pts[..., 1], pts[..., 2].clamp(min=1e-6)
                u, v = u / w, v / w
                valid = (w > 0) & (u >= 0) & (v >= 0) & (u <= self.Wimg - 1) & (v <= self.Himg - 1)

                xf, yf = u * sx, v * sy
                gx = (xf / (Wf - 1) * 2) - 1.0
                gy = (yf / (Hf - 1) * 2) - 1.0
                grid = torch.stack([gx, gy], dim=-1)  # [Nc,mb,2]

                feat = F.grid_sample(
                    input=fpn[b],                 # [Nc,Cimg,Hf,Wf]
                    grid=grid.unsqueeze(2),       # [Nc,mb,1,2]
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=self.align_corners
                ).squeeze(3).transpose(1, 0)     # -> [mb,Nc,Cimg]
                feat = feat * valid.transpose(0, 1).unsqueeze(-1)
                if self.cam_pool == 'avg':
                    denom = valid.sum(dim=0).clamp(min=1).unsqueeze(-1)
                    feat = feat.sum(dim=1) / denom    # [mb,Cimg]
                else:
                    feat = feat.max(dim=1).values     # [mb,Cimg]
                parts.append((sel, feat))

            if parts:
                buf = torch.zeros((xyz.shape[0], Cimg), device=device, dtype=fpn.dtype)
                for sel, ft in parts:
                    buf[sel] = ft
                desc[st:ed] = buf
        return desc

    def forward(self, *args, **kwargs):
        out = self.base_vfe(*args, **kwargs)
        # unpack common returns
        if isinstance(out, (list, tuple)):
            if len(out) == 3:
                voxel_feats, voxel_coords, voxel_num_points = out
            elif len(out) == 2:
                voxel_feats, voxel_coords = out
                voxel_num_points = None
            else:
                return out
        elif isinstance(out, dict):
            voxel_feats = out['voxel_feats']
            voxel_coords = out['voxel_coords']
            voxel_num_points = out.get('voxel_num_points', None)
        else:
            return out

        # read painting context (if not set, do nothing)
        fpn, metas = get_painting_context()
        if (fpn is None) or (metas is None):
            return out

        if self.detach_img:
            fpn = fpn.detach()

        centers, batch_ids = self._voxel_centers(voxel_coords)  # [M,3], [M]
        B, Nc, Cimg, Hf, Wf = fpn.shape
        img_desc = self._sample_img_desc(fpn, metas, centers, batch_ids)  # [M,Cimg]

        C0 = voxel_feats.shape[1]
        self._ensure_heads(C0, Cimg, voxel_feats.device)

        img_proj = self._proj(img_desc)  # [M,img_feat_out]
        if self.fuse == 'concat_linear':
            fused = torch.cat([voxel_feats, img_proj], dim=1)
            voxel_feats = self._fuse_cat(fused)
        elif self.fuse == 'add':
            voxel_feats = voxel_feats + self._fuse_add(img_proj)
        else:  # gated
            add_on = self._fuse_add(img_proj)         # [M,C0]
            gate = self._gate(img_proj)               # [M,C0] in [0,1]
            voxel_feats = voxel_feats + gate * add_on

        # return in the same structure
        if isinstance(out, (list, tuple)):
            return type(out)((voxel_feats, voxel_coords, voxel_num_points)[:len(out)])
        else:
            out['voxel_feats'] = voxel_feats
            return out