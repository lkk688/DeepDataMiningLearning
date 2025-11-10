import math
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
def _dbg(*args, **kwargs):
    if os.environ.get('DEBUG_VT', '0') == '1':
        print(*args, **kwargs)
from mmdet3d.registry import MODELS

try:
    # mmcv>=2.x
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
except Exception as e:
    raise ImportError(
        "MultiScaleDeformableAttention not found. Please install mmcv>=2.0. "
        f"Original error: {e}"
    )


def _build_bev_grid(xbound, ybound):
    """Create BEV grid centers in LiDAR XY; return (1, H*W, 2), H, W."""
    xs = torch.arange(xbound[0] + xbound[2] / 2, xbound[1], xbound[2])
    ys = torch.arange(ybound[0] + ybound[2] / 2, ybound[1], ybound[2])
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    bev_xy = torch.stack([xx, yy], dim=-1)  # (H,W,2)
    H, W = bev_xy.shape[:2]
    bev_xy = bev_xy.reshape(-1, 2).unsqueeze(0)  # (1,H*W,2)
    return bev_xy, H, W


def _project_lidar_to_img(pts_lidar, lidar2img):
    """
    pts_lidar: (B, Nv, Nk, 3)
    lidar2img: (B, Nv, 4, 4)
    Returns:
      uv: (B, Nv, Nk, 2)  in pixel coordinates
      depth_cam: (B, Nv, Nk)  camera-Z
      valid: (B, Nv, Nk)   positive depth
    """
    B, Nv, Nk, _ = pts_lidar.shape
    ones = torch.ones((B, Nv, Nk, 1), dtype=pts_lidar.dtype, device=pts_lidar.device)
    pts_h = torch.cat([pts_lidar, ones], dim=-1)  # (B,Nv,Nk,4)
    cam = torch.einsum('bvij,bvkj->bvki', lidar2img, pts_h)  # (B,Nv,Nk,4)
    z = cam[..., 2].clamp(min=1e-5)
    uv = cam[..., :2] / z.unsqueeze(-1)
    valid = z > 1e-4
    return uv, z, valid


class _DeformBlock(nn.Module):
    """Tiny block wrapping MSDeformAttn for BEV queries."""
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4, ffn_ratio=2.0, dropout=0.0):
        super().__init__()
        self.attn = MultiScaleDeformableAttention(
            embed_dims, num_levels=num_levels, num_heads=num_heads, num_points=num_points)
        self.ln1 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, int(embed_dims * ffn_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(embed_dims * ffn_ratio), embed_dims),
        )
        self.ln2 = nn.LayerNorm(embed_dims)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, ref_pts, value, spatial_shapes, level_start_index, key_padding_mask=None):
        attn_out = self.attn(
            query=q, reference_points=ref_pts, value=value,
            spatial_shapes=spatial_shapes, level_start_index=level_start_index,
            key_padding_mask=key_padding_mask)
        x = self.ln1(q + self.drop(attn_out))
        y = self.ffn(x)
        y = self.ln2(x + self.drop(y))
        return y


@MODELS.register_module()
class BEVCrossAttnTransform(nn.Module):
    """
    BEVFormer-like view transformer with optional sparse depth distillation.

    Inputs:
      mlvl_feats: list length = num_levels, each (B, Nv, C_in, H, W)
      metas: list of dict, len = B; must contain 'lidar2img' (Nv x 4x4), 'img_shape' (H_img, W_img)
      points (optional): list of length B, each tensor (Ni, 5/4/3...) with XYZ in LiDAR frame.

    Returns:
      If self.return_aux: (bev: (B,C,H_bev,W_bev), aux_losses: dict)
      else: bev only.
    """
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 xbound=(-54.0, 54.0, 0.3),
                 ybound=(-54.0, 54.0, 0.3),
                 zbound=(-10.0, 10.0, 4.0),
                 num_cams=6,
                 num_levels=None, 
                 expected_layout='BCNHW',
                 num_points=4,
                 num_heads=8,
                 num_layers=2,
                 depth_distill_weight=0.1,      # weight for sparse depth L1
                 depth_head_on_level=1,         # which FPN level to predict depth (stride≈8)
                 return_aux=True):
        super().__init__()
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.num_points = num_points
        self.out_channels = out_channels
        self.depth_distill_weight = depth_distill_weight
        self.depth_head_on_level = depth_head_on_level
        self.return_aux = return_aux
        #self.expected_layout = 'BCNHW'   # VT wants (B, C, N, H, W)
        self.num_levels = num_levels         # let it be None (auto)
        self.expected_layout = expected_layout
        

        bev_xy, H, W = _build_bev_grid(xbound, ybound)
        self.register_buffer('bev_xy', bev_xy)  # (1, H*W, 2)
        self.bev_h, self.bev_w = H, W

        z_min, z_max, z_step = zbound
        zs = torch.arange(z_min, z_max, z_step)
        self.register_buffer('z_anchors', zs)  # (K,)

        self.bev_embed = nn.Parameter(torch.zeros(H * W, out_channels))
        nn.init.xavier_uniform_(self.bev_embed)

        # project each level to out_channels for attention value
        self.proj = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1) for _ in range(num_levels)])

        self.blocks = nn.ModuleList([
            _DeformBlock(embed_dims=out_channels, num_heads=num_heads,
                         num_levels=num_levels * num_cams,
                         num_points=self.num_points)
            for _ in range(num_layers)
        ])

        # simple depth head for sparse distillation (per-camera, 1x1 conv on chosen level)
        self.depth_head = nn.Conv2d(in_channels, 1, kernel_size=1)

    # ------------ helpers ------------
    def _canonize_BCNHW(self, x: torch.Tensor) -> torch.Tensor:
        """接受 (B,C,N,H,W) 或 (B,N,C,H,W)，返回 (B,C,N,H,W)。"""
        assert x.dim() == 5, f'Expect 5D, got {tuple(x.shape)}'
        if x.shape[2] == self.num_cams:  # (B,C,N,H,W)
            return x
        if x.shape[1] == self.num_cams:  # (B,N,C,H,W) -> (B,C,N,H,W)
            return x.permute(0, 2, 1, 3, 4).contiguous()
        raise AssertionError(
            f'Cannot infer camera dim from {tuple(x.shape)} with num_cams={self.num_cams}')

    def _build_value_and_shapes(self, mlvl_feats: list):
        """
        mlvl_feats: list of L tensors; each is (B,C,N,H,W) or (B,N,C,H,W).
        Returns:
        value: (B, sum_l (N*H_l*W_l), C_embed)
        spatial_shapes: (L*N, 2)   # 每个 level 的 (H,W) 重复 Nv 次
        level_start_index: (L*N,)
        """
        # 规范到 (B,C,N,H,W)
        def _canon(x):
            assert x.dim() == 5
            if x.shape[2] == self.num_cams:     # (B,C,N,H,W)
                return x
            if x.shape[1] == self.num_cams:     # (B,N,C,H,W) -> (B,C,N,H,W)
                return x.permute(0, 2, 1, 3, 4).contiguous()
            raise AssertionError(f'Bad shape {tuple(x.shape)} for num_cams={self.num_cams}')

        x0 = _canon(mlvl_feats[0])
        B, C0, Nv, _, _ = x0.shape
        assert Nv == self.num_cams

        value_list = []
        shapes_list = []
        for lvl, x in enumerate(mlvl_feats):
            x = _canon(x)                       # (B,C,N,H,W)
            Bx, Cx, Nv_x, H, W = x.shape
            assert Bx == B and Nv_x == Nv

            _dbg(f'[VT][lvl{lvl}] in (B,C,N,H,W)=', (B, Cx, Nv, H, W))
            # ====== 这里是关键：给 Conv2d 的输入要是 (B, C_in, Tokens, 1) ======
            # (B,C,N,H,W) -> (B,C,N*H*W,1)
            x_seq = x.reshape(B, Cx, Nv * H * W, 1).contiguous()
            _dbg(f'[VT][lvl{lvl}] x_seq to Conv2d (B,C,T,1)=', tuple(x_seq.shape))

            # 1x1 卷积投影: (B, C_in, Tokens, 1) -> (B, C_embed, Tokens, 1)
            v2d = self.proj[lvl](x_seq)         # Conv2d 1x1
            _dbg(f'[VT][lvl{lvl}] v2d shape =', tuple(v2d.shape))
            # 收回到 (B, Tokens, C_embed)
            v = v2d.squeeze(-1).transpose(1, 2).contiguous()
            _dbg(f'[VT][lvl{lvl}] v shape   =', tuple(v.shape))
            value_list.append(v)

            for _ in range(Nv):
                shapes_list.append([H, W])

        value = torch.cat(value_list, dim=1)    # (B, sum_l Nv*H*W, C_embed)
        spatial_shapes = torch.as_tensor(shapes_list, dtype=torch.long, device=value.device)
        hw = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        level_start_index = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=value.device),
            torch.cumsum(hw, dim=0)[:-1]], dim=0)

        assert int(hw.sum().item()) == int(value.shape[1]), \
            f"sum(HW)={int(hw.sum())} vs value tokens={int(value.shape[1])}"
        return value, spatial_shapes, level_start_index

    @torch.no_grad()
    def _build_reference_points(self, metas, spatial_shapes, mlvl_feats):
        """
        Build reference points for MSDeformAttn:
        returns (B, Nq, L', K, 2) in [0,1], where L' = num_levels * num_cams
        """
        B = len(metas)
        device = mlvl_feats[0].device
        Nv = self.num_cams
        K = self.z_anchors.numel()
        Nq = self.bev_xy.shape[1]

        bev_xy = self.bev_xy.to(device).expand(B, -1, -1)  # (B,Nq,2)
        z = self.z_anchors.to(device)[None, None, :].expand(B, Nq, K)  # (B,Nq,K)
        xyz = torch.stack([bev_xy[..., 0].unsqueeze(-1).expand(-1, -1, K),
                           bev_xy[..., 1].unsqueeze(-1).expand(-1, -1, K),
                           z], dim=-1)  # (B,Nq,K,3)

        #lidar2img = torch.stack([torch.stack(m['lidar2img']) for m in metas], dim=0).to(device)  # (B,Nv,4,4)
        device = self._get_device_from_feats(mlvl_feats)  # 或者用 value.device

        # 统一转 Tensor，并在需要时补成 4x4
        lidar2img = self._to_tensor_stack(metas, 'lidar2img', device, dtype=torch.float32, pad_to_4x4=True)
        cam2img   = self._to_tensor_stack(metas, 'cam2img',   device, dtype=torch.float32, pad_to_4x4=True) \
                    if 'cam2img' in metas[0] else None
        cam2lidar = self._to_tensor_stack(metas, 'cam2lidar', device, dtype=torch.float32, pad_to_4x4=True) \
                    if 'cam2lidar' in metas[0] else None
        img_aug   = self._to_tensor_stack(metas, 'img_aug_matrix',   device, dtype=torch.float32, pad_to_4x4=True) \
                    if 'img_aug_matrix' in metas[0] else None
        lidar_aug = self._to_tensor_stack(metas, 'lidar_aug_matrix', device, dtype=torch.float32, pad_to_4x4=True) \
                    if 'lidar_aug_matrix' in metas[0] else None
        
        uv, depth, valid = _project_lidar_to_img(
            xyz.reshape(B, 1, Nq * K, 3).expand(-1, Nv, -1, -1), lidar2img
        )  # uv: (B,Nv,NqK,2)

        HWs = [(int(h), int(w)) for (h, w) in spatial_shapes.tolist()]  # length L' = Nv * num_levels
        Lp = len(HWs)
        assert Lp == self.num_levels * Nv, f"spatial_shapes length {Lp} != num_levels*num_cams"

        ref_list = []
        for lvl in range(self.num_levels):
            for cam in range(Nv):
                H, W = HWs[lvl * Nv + cam]
                uv_cam = uv[:, cam]  # (B,NqK,2)
                u = uv_cam[..., 0] / (W - 1.0)
                v = uv_cam[..., 1] / (H - 1.0)
                ref = torch.stack([u, v], dim=-1)  # (B,NqK,2)
                val_cam = valid[:, cam]
                in_bounds = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1) & val_cam
                ref = torch.where(in_bounds.unsqueeze(-1), ref, torch.full_like(ref, -2.0))
                ref = ref.view(B, -1, K, 2)
                ref_list.append(ref)

        ref_pts = torch.stack(ref_list, dim=2)  # (B, Nq, L', K, 2)
        return ref_pts

    def _sparse_depth_loss(self, mlvl_feats: List[torch.Tensor], metas, points, level_idx=1) -> torch.Tensor:
        """Project LiDAR to each image, supervise a 1x1 depth head on mlvl_feats[level_idx].
        We supervise camera-Z (positive), Smooth L1 on sparse pixels.
        """
        if points is None or len(points) == 0 or self.depth_distill_weight <= 0:
            return mlvl_feats[0].new_zeros([])

        feat = mlvl_feats[level_idx]  # (B,Nv,C,H,W)
        B, Nv, C, Hf, Wf = feat.shape
        # predict depth on this level
        pred = self.depth_head(feat.flatten(0,1))  # (B*Nv, 1, Hf, Wf)
        pred = pred.unflatten(0, (B, Nv))  # (B,Nv,1,Hf,Wf)

        # gather metas
        device = feat.device
        lidar2img = torch.stack([torch.stack(m['lidar2img']) for m in metas], dim=0).to(device)  # (B,Nv,4,4)

        total = feat.new_zeros([])
        count = 0
        # 每个样本单独投影，控制采样粒度避免过慢
        for b in range(B):
            if points[b] is None:
                continue
            pts = points[b]
            assert pts.dim() == 2 and pts.size(-1) >= 3, "points must contain XYZ in LiDAR"
            xyz = pts[:, :3].to(device)  # (Nb,3)
            # 采样上限，避免超大帧太慢
            if xyz.size(0) > 60000:
                idx = torch.randperm(xyz.size(0), device=device)[:60000]
                xyz = xyz[idx]

            # tile to cameras -> (Nv, Nk, 3)
            Nk = xyz.size(0)
            xyz_rep = xyz.unsqueeze(0).expand(Nv, Nk, 3)
            uv, z, valid = _project_lidar_to_img(
                xyz_rep.unsqueeze(0), lidar2img[b:b+1])  # uv/z/valid: (1,Nv,Nk,..)
            uv, z, valid = uv[0], z[0], valid[0]      # (Nv,Nk,2), (Nv,Nk)

            for cam in range(Nv):
                # 将像素坐标缩放到 feature 分辨率
                H_img, W_img = metas[b]['img_shape'][:2]
                # 这里假设该 level 的下采样 stride = round(H_img / Hf)
                sf_h = max(1, round(H_img / Hf))
                sf_w = max(1, round(W_img / Wf))
                u = (uv[cam, :, 0] / sf_w).round().long()
                v = (uv[cam, :, 1] / sf_h).round().long()

                m = (valid[cam]
                     & (u >= 0) & (u < Wf)
                     & (v >= 0) & (v < Hf))
                if m.sum() == 0:
                    continue
                z_cam = z[cam, m]     # (Ns,)
                # 只用近处 & 合理深度
                m2 = (z_cam > 1e-3) & (z_cam < 200.0)
                if m2.sum() == 0:
                    continue

                sel = torch.nonzero(m, as_tuple=False).squeeze(-1)[m2]
                uu = u[sel]
                vv = v[sel]
                pred_cam = pred[b, cam, 0, vv, uu]  # (Ns,)
                # Smooth L1 到 camera-Z
                total = total + F.smooth_l1_loss(pred_cam, z_cam[m2], reduction='sum')
                count += int(m2.sum().item())

        if count == 0:
            return feat.new_zeros([])
        return total / count

    # ------------ forward ------------
    def _to_tensor_stack(self, metas, key, device, dtype=torch.float32, pad_to_4x4=False):
        """把 metas[i][key]（list[np.ndarray] 或 np.ndarray）变成 (B, Nv, h, w) 的 Tensor。"""
        batch_list = []
        for m in metas:
            arr = m.get(key, None)
            if arr is None:
                batch_list.append(None)
                continue
            if isinstance(arr, list):
                arr = np.stack(arr, axis=0)     # (Nv, h, w)
            arr = torch.as_tensor(arr, dtype=dtype, device=device)
            if arr.dim() == 2:                  # (h, w) -> (1, h, w)
                arr = arr.unsqueeze(0)
            if pad_to_4x4 and arr.shape[-2:] == (3, 3):
                # 把 3×3 内参补成 4×4 齐次
                Nv = arr.shape[0]
                pad_col = torch.zeros((Nv, 3, 1), dtype=dtype, device=device)
                top = torch.cat([arr, pad_col], dim=2)               # (Nv, 3, 4)
                bottom = torch.tensor([0, 0, 0, 1], dtype=dtype, device=device).view(1, 1, 4)
                bottom = bottom.expand(Nv, -1, -1)                   # (Nv, 1, 4)
                arr = torch.cat([top, bottom], dim=1)                # (Nv, 4, 4)
            batch_list.append(arr)

        # 若有样本缺失该 key，用单位阵补齐
        if any(a is None for a in batch_list):
            # 推断 Nv
            some = next(a for a in batch_list if a is not None)
            Nv = some.shape[0]
            eye = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).expand(Nv, 4, 4)
            batch_list = [a if a is not None else eye for a in batch_list]

        return torch.stack(batch_list, dim=0)    # (B, Nv, h, w)


    def _get_device_from_feats(self, mlvl_feats):
        if isinstance(mlvl_feats, (list, tuple)):
            return mlvl_feats[0].device
        return mlvl_feats.device

    def forward(self, mlvl_feats, metas):
        # 接受 Tensor 或 List[Tensor]
        if isinstance(mlvl_feats, torch.Tensor):
            mlvl_feats = [mlvl_feats]

        # 推断相机数
        if not hasattr(self, 'num_cams') or self.num_cams is None:
            self.num_cams = len(metas[0]['cam2img'])  # nuScenes 通常 6

        # 推断/容忍 level 个数
        if getattr(self, 'num_levels', None) is None:
            self.num_levels = len(mlvl_feats)
        elif len(mlvl_feats) != self.num_levels:
            if self.training:
                print(f'[BEVCrossAttn] overwrite num_levels {self.num_levels} -> {len(mlvl_feats)}')
            self.num_levels = len(mlvl_feats)

        # 规范形状以拿到 B
        x0 = self._canonize_BCNHW(mlvl_feats[0]) if hasattr(self, '_canonize_BCNHW') else mlvl_feats[0]
        B = x0.shape[0]
        _dbg('[VT] num_cams =', self.num_cams, '| num_levels =', len(mlvl_feats))

        # 6) 构造 value / spatial_shapes / level_start_index
        # 注意：_build_value_and_shapes 默认按 BNCHW 取维度（x.shape[1] 是 N，x.shape[2] 是 C）
        value, spatial_shapes, level_start_index = self._build_value_and_shapes(mlvl_feats)

        _dbg('[VT] spatial_shapes[:6]=', spatial_shapes[:6].tolist(),
            '| value tokens=', int(value.shape[1]))
        
        # 7) BEV queries + reference points
        q = self.bev_embed[None, :, :].expand(B, -1, -1).to(value.dtype)  # (B, Nq, C)
        with torch.no_grad():
            ref_pts = self._build_reference_points(metas, spatial_shapes, mlvl_feats)  # (B, Nq, L', K, 2)

        # 8) 逐块 cross-attn
        x = q
        for blk in self.blocks:
            x = blk(x, ref_pts, value, spatial_shapes, level_start_index, key_padding_mask=None)

        # 9) 回到 BEV (B, C, H_bev, W_bev)
        bev = x.transpose(1, 2).contiguous().view(B, self.out_channels, self.bev_h, self.bev_w)

        # 这里不要再引用未定义的 points/aux；如果要蒸馏，放到上层（外面）做
        return bev