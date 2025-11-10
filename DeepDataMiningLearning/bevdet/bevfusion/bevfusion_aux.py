from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from projects.BEVFusion.bevfusion.bevfusion import BEVFusion as OrigBEVFusion
from projects.bevdet.bevfusion.view_transformers.adapters import call_view_transform

# --- Optional: helpers (place near the top of bevfusion_aux.py) ---
def _metas_to_plain_dicts(batch_input_metas: List) -> List[Dict]:
    """mmengine gives Det3DDataSample; convert to plain dict for adapters."""
    if len(batch_input_metas) == 0:
        return batch_input_metas
    if hasattr(batch_input_metas[0], 'metainfo'):  # Det3DDataSample
        return [m.metainfo for m in batch_input_metas]
    return batch_input_metas  # already dicts

def _infer_num_cams(meta0) -> int:
    for k in ('cam2img', 'ori_cam2img', 'lidar2img', 'ori_lidar2img'):
        if k in meta0:
            return len(meta0[k])
    raise RuntimeError('Cannot infer num_cams from metas[0]')


def _has_any(obj, names):
    return any(getattr(obj, n, None) is not None for n in names)

    # 放在 imports 下面、类定义之前
def _reshape_feats_to_BNCHW(img_feats, imgs):
    """Ensure img_feats is (B, N, C, H, W) given imgs of shape (B, N, C, H, W)."""
    if isinstance(img_feats, (list, tuple)):
        img_feats = img_feats[0]            # 有些 neck 会返回多层特征，取一层

    assert img_feats.dim() == 4, f'Expected (B*N,C,H,W), got {img_feats.shape}'
    B, N = imgs.shape[:2]
    BN, C, H, W = img_feats.shape
    assert BN == B * N, f'BN={BN} but B*N={B*N} (B={B}, N={N})'
    img_feats = img_feats.view(B, N, C, H, W).contiguous()
    return img_feats

@MODELS.register_module(name='BEVFusionAuxDepth')
class BEVFusionAuxDepth(OrigBEVFusion):
    """BEVFusion with optional sparse-depth distillation on the camera branch.

    - Keeps base forward contract (extract_feat -> feats).
    - Adds extract_feat_with_aux -> (feats, aux_losses) for loss().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Infer branch availability from actually attached components
        self.has_img_branch = _has_any(
            self, ('img_backbone', 'img_neck', 'view_transform'))
        self.has_pts_branch = _has_any(
            self, ('pts_voxel_encoder', 'pts_middle_encoder',
                   'pts_backbone', 'pts_neck'))

        # Optional aliases if your code referenced these names before
        self.with_image = self.has_img_branch
        self.with_lidar = self.has_pts_branch

    # ---------- helpers ----------

    def _normalize_imgs(self, batch_inputs_dict: Dict) -> Optional[Tensor]:
        """Return imgs as a 5D tensor [B, N, C, H, W] or None.
        Accepts either 'imgs' (already 5D) or 'img' (4D -> expand N=1).
        """
        imgs = batch_inputs_dict.get('imgs', None)
        if imgs is None:
            imgs = batch_inputs_dict.get('img', None)
        if imgs is None:
            return None

        # common cases:
        #  - imgs: [B, N, C, H, W]
        #  - img : [B, C, H, W]  -> expand as N=1
        if imgs.dim() == 4:
            B, C, H, W = imgs.shape
            imgs = imgs.view(B, 1, C, H, W).contiguous()
        elif imgs.dim() != 5:
            raise ValueError(f'Unexpected image tensor shape: {tuple(imgs.shape)}')
        return imgs.contiguous()

    # --- add these helpers near the top of the class ---

    def _normalize_metas(self, batch_input_metas):
        """Accepts List[Det3DDataSample] or List[Dict]; returns List[Dict]."""
        if not batch_input_metas:
            return batch_input_metas
        if isinstance(batch_input_metas[0], dict):
            return batch_input_metas
        # Det3DDataSample or other obj with .metainfo
        return [getattr(m, 'metainfo', m) for m in batch_input_metas]

    def _gather_metas(self, batch_input_metas, imgs):
        """Pack meta matrices into tensors; batch_input_metas is List[Dict]."""
        # ensure dicts
        metas = self._normalize_metas(batch_input_metas)

        lidar2image, cam_intr, cam2lidar = [], [], []
        img_aug, lidar_aug = [], []
        for m in metas:
            lidar2image.append(m['lidar2img'])
            cam_intr.append(m['cam2img'])
            cam2lidar.append(m['cam2lidar'])
            img_aug.append(m.get('img_aug_matrix', np.eye(4)))
            lidar_aug.append(m.get('lidar_aug_matrix', np.eye(4)))
        dev = imgs.device
        return (
            imgs.new_tensor(np.asarray(lidar2image), device=dev),
            imgs.new_tensor(np.asarray(cam_intr),     device=dev),
            imgs.new_tensor(np.asarray(cam2lidar),    device=dev),
            imgs.new_tensor(np.asarray(img_aug),      device=dev),
            imgs.new_tensor(np.asarray(lidar_aug),    device=dev),
        )
    
    # def _gather_metas(self, batch_input_metas: List[Dict], imgs: Tensor):
    #     """Pack meta matrices into tensors on same device as imgs."""
    #     lidar2image, cam_intr, cam2lidar = [], [], []
    #     img_aug, lidar_aug = [], []
    #     for meta in batch_input_metas:
    #         lidar2image.append(meta['lidar2img'])
    #         cam_intr.append(meta['cam2img'])
    #         cam2lidar.append(meta['cam2lidar'])
    #         img_aug.append(meta.get('img_aug_matrix', np.eye(4)))
    #         lidar_aug.append(meta.get('lidar_aug_matrix', np.eye(4)))
    #     dev = imgs.device
    #     return (
    #         imgs.new_tensor(np.asarray(lidar2image), device=dev),
    #         imgs.new_tensor(np.asarray(cam_intr), device=dev),
    #         imgs.new_tensor(np.asarray(cam2lidar), device=dev),
    #         imgs.new_tensor(np.asarray(img_aug), device=dev),
    #         imgs.new_tensor(np.asarray(lidar_aug), device=dev),
    #     )

    # ---------- main flow ----------

    def extract_feat_with_aux(
        self,
        batch_inputs_dict: Dict[str, Optional[Tensor]],
        batch_input_metas: List[Dict],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Return (feats, aux_losses).

        - 相机分支：backbone/neck -> 多层特征 (B,N,C,H,W)
        * 若 view_transform 是 LSS：用 call_view_transform_lss（自动拼几何矩阵）
        * 若是 Cross-Attn：用 call_view_transform（走 (B,C,N,H,W)/List[Tensor]）
        - LiDAR 分支：保持 FP32 体素化与中间编码
        - 融合后过 BEV backbone/neck
        - 可选：稀疏深度蒸馏（smooth L1）
        """
        # ------ 输入取值 ------
        imgs: Optional[Tensor] = self._normalize_imgs(batch_inputs_dict)
        points = batch_inputs_dict.get('points', None)
        feats_list: List[Tensor] = []
        aux_losses: Dict[str, Tensor] = {}

        # ------ 相机分支 ------
        if getattr(self, 'has_img_branch', False) and imgs is not None:
            B, N, C, H, W = imgs.shape
            x = imgs.view(B * N, C, H, W).contiguous()

            # backbone + neck
            x = self.img_backbone(x)
            x = self.img_neck(x)

            # 统一成多层 (B,N,C,H,W) 列表
            if isinstance(x, (list, tuple)):
                mlvl: List[Tensor] = []
                for lvl, xi in enumerate(x):
                    BN, C2, H2, W2 = xi.shape
                    assert BN == B * N, f'neck lvl{lvl}: BN={BN} != B*N={B*N}'
                    mlvl.append(xi.view(B, N, C2, H2, W2).contiguous())
            else:
                BN, C2, H2, W2 = x.shape
                assert BN == B * N, f'neck: BN={BN} != B*N={B*N}'
                mlvl = [x.view(B, N, C2, H2, W2).contiguous()]

            # 选择 VT 路径：LSS or Cross-Attn（自动判断，也可在 config 里设 self.vt_kind）
            vt = self.view_transform
            vt_kind = getattr(self, 'vt_kind', None)
            vt_name = vt.__class__.__name__.lower() if vt is not None else ''

            use_lss = False
            if vt_kind is not None:
                use_lss = (str(vt_kind).lower() == 'lss')
            else:
                # Heuristic：DepthLSSTransform 类名或带 dbound 字段认为是 LSS
                use_lss = ('depthlss' in vt_name) or hasattr(vt, 'dbound')

            # 调用视角变换
            if use_lss:
                # LSS 只用一层（与官方 BEVFusion 一致，默认 level=0）
                level = getattr(self, 'lss_level', 0)
                from .view_transformers.adapters import call_view_transform_lss
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    img_bev = call_view_transform_lss(
                        vt,                # DepthLSSTransform
                        mlvl,              # List[(B,N,C,H,W)] 或单层 Tensor
                        batch_input_metas,
                        points=points,
                        level=level
                    )
            else:
                # Cross-Attn：吃多层，适配器会转成 VT 期望布局
                from .view_transformers.adapters import call_view_transform
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    img_bev = call_view_transform(
                        vt,                # BEVCrossAttnTransform
                        mlvl,              # List[(B,N,C,H,W)]
                        batch_input_metas
                    )

            if getattr(self, 'debug_shapes', False):
                print('[DBG][IMG] levels:',
                    [tuple(t.shape) for t in mlvl],
                    '| bev:', tuple(img_bev.shape))
            feats_list.append(img_bev)

            # 稀疏深度蒸馏（可选）
            if 'sparse_depth' in batch_inputs_dict and hasattr(self, 'depth_head'):
                depth_level = getattr(self, 'depth_head_on_level', 0)
                depth_level = min(depth_level, len(mlvl) - 1)
                xl = mlvl[depth_level]                                # (B,N,C,H,W)
                pred_depth = self.depth_head(xl.view(B * N, xl.size(2), xl.size(3), xl.size(4)))
                target = batch_inputs_dict['sparse_depth']            # 形状需与 head 输出对齐
                aux_losses['loss_depth_distill'] = F.smooth_l1_loss(pred_depth, target)

        # ------ LiDAR 分支 ------
        if getattr(self, 'has_pts_branch', False) and points is not None:
            # 体素化保持 FP32，避免 AMP 数值问题
            with torch.autocast('cuda', enabled=False):
                pts = [p.float() for p in points]
                feats, coords, sizes = self.voxelize(pts)
                batch_size = coords[-1, 0] + 1
            x_pts = self.pts_middle_encoder(feats, coords, batch_size)
            if getattr(self, 'debug_shapes', False):
                print('[DBG][PTS] mid-enc:', tuple(x_pts.shape))
            feats_list.append(x_pts)

        # ------ 融合 & BEV 主干 ------
        if self.fusion_layer is not None:
            x = self.fusion_layer(feats_list)
        else:
            assert len(feats_list) == 1, 'No fusion_layer but two branches present.'
            x = feats_list[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        if getattr(self, 'debug_shapes', False):
            print('[DBG][BEV] out:', tuple(x.shape))

        return x, aux_losses
   
    
    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        metas = self._normalize_metas(batch_input_metas)
        feats, _ = self.extract_feat_with_aux(batch_inputs_dict, metas)
        return feats

    # def loss(
    #     self,
    #     batch_inputs_dict: Dict[str, Optional[Tensor]],
    #     batch_data_samples: List[Det3DDataSample],
    #     **kwargs
    # ):
    #     batch_input_metas = [d.metainfo for d in batch_data_samples]
    #     feats, aux_losses = self.extract_feat_with_aux(batch_inputs_dict, batch_input_metas)

    #     losses = {}
    #     if self.with_bbox_head:
    #         losses.update(self.bbox_head.loss(feats, batch_data_samples))
    #     losses.update(aux_losses)
    #     return losses
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        metas = self._normalize_metas(batch_data_samples)
        feats, aux_losses = self.extract_feat_with_aux(batch_inputs_dict, metas)

        losses = {}
        if self.with_bbox_head:
            losses.update(self.bbox_head.loss(feats, batch_data_samples))
        losses.update(aux_losses)
        return losses

    def loss(self, batch_inputs_dict: Dict[str, Any], batch_data_samples: List[Dict]):
        """Add aux losses to the original losses dict."""
        feats, aux_losses = self.extract_feat(batch_inputs_dict, batch_data_samples)
        pred_dict = self.bbox_head(feats)
        losses = self.bbox_head.loss(pred_dict, batch_data_samples)
        losses.update(aux_losses)  # merge sparse depth loss, if present
        return losses

    def predict(self, batch_inputs_dict: Dict[str, Any], batch_data_samples: List[Dict], **kwargs):
        feats, _ = self.extract_feat(batch_inputs_dict, batch_data_samples)
        pred_dict = self.bbox_head(feats)
        results_list = self.bbox_head.predict(pred_dict, batch_data_samples, **kwargs)
        return results_list