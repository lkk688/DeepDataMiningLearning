# projects/bevdet/bevfusion/bevfusion_ca.py
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from typing import List, Union

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor, nn
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList

from .bevfusion.ops import Voxelization

import threading
_TLS = threading.local()

def set_painting_context(fpn_feats, batch_metas):
    _TLS.fpn_feats = fpn_feats
    _TLS.batch_metas = batch_metas

def get_painting_context():
    fpn = getattr(_TLS, 'fpn_feats', None)
    metas = getattr(_TLS, 'batch_metas', None)
    return fpn, metas

def clear_painting_context():
    for k in ('fpn_feats', 'batch_metas'):
        if hasattr(_TLS, k):
            delattr(_TLS, k)

@MODELS.register_module()
class StackNeck(nn.Module):
    """
    Sequentially applies multiple necks:
      out = neck_k(...(neck_2(neck_1(x))))
    Works with Tensor or list/tuple outputs transparently.
    """
    def __init__(self, necks: List[dict]):
        super().__init__()
        assert isinstance(necks, (list, tuple)) and len(necks) >= 1
        self.necks = nn.ModuleList([MODELS.build(nc) for nc in necks])

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]):
        out = x
        for neck in self.necks:
            out = neck(out)  # each neck decides Tensor vs list I/O
            # normalize tuple→list for downstream consistency
            if isinstance(out, tuple):
                out = list(out)
        return out

# ---------- small utilities ----------
def _boxes_center_xy(data_sample) -> Optional[torch.Tensor]:
    """Extract (x,y) centers from GT boxes in LiDAR frame."""
    if hasattr(data_sample, 'gt_instances_3d') and data_sample.gt_instances_3d is not None:
        gi = data_sample.gt_instances_3d
        if hasattr(gi, 'bboxes_3d') and gi.bboxes_3d is not None:
            b = gi.bboxes_3d
            t = b.tensor if hasattr(b, 'tensor') else torch.as_tensor(b, device='cpu')
            return t[..., :2]
    if hasattr(data_sample, 'gt_bboxes_3d') and data_sample.gt_bboxes_3d is not None:
        b = data_sample.gt_bboxes_3d
        t = b.tensor if hasattr(b, 'tensor') else torch.as_tensor(b, device='cpu')
        return t[..., :2]
    return None

def _draw_gaussian(heatmap: torch.Tensor, center: Tuple[int, int], radius: int) -> None:
    """In-place draw a tiny Gaussian on (H×W) heatmap (class-agnostic target)."""
    y, x = int(center[0]), int(center[1])
    H, W = heatmap.shape[-2], heatmap.shape[-1]
    diam = 2 * radius + 1
    sigma = diam / 6.0
    yy = torch.arange(0, diam, device=heatmap.device).view(-1, 1).float()
    xx = torch.arange(0, diam, device=heatmap.device).view(1, -1).float()
    g = torch.exp(-((yy - radius) ** 2 + (xx - radius) ** 2) / (2 * sigma ** 2))

    top, bottom = max(0, y - radius), min(H, y + radius + 1)
    left, right = max(0, x - radius), min(W, x + radius + 1)
    g_top = max(0, radius - y)
    g_left = max(0, radius - x)
    g_bottom = g_top + (bottom - top)
    g_right = g_left + (right - left)

    if top < bottom and left < right:
        heatmap[..., top:bottom, left:right] = torch.maximum(
            heatmap[..., top:bottom, left:right], g[g_top:g_bottom, g_left:g_right]
        )

@MODELS.register_module()
class BEVFusionCA(Base3DDetector):
    """
    BEVFusion with Cross-Attention view transform (CA-LSS), optional:
      • AUX BEV heatmap supervision (class-agnostic);
      • Voxel painting pre-hook for MVX-style early fusion;
      • Two-scale tokens (e.g., P3 + P4) fed to view_transform.

    This keeps BEVFusion’s public interfaces (extract_feat/predict/loss) so your
    configs and heads remain compatible.
    """

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        # ---- New knobs ----
        aux_cfg: Optional[Dict] = None,       # e.g., dict(loss_weight=0.1, radius_cells=2)
        voxel_painting_on: bool = False,      # stash VT input for VFE wrapper
        use_two_scale_tokens: bool = False,   # feed [P3,P4] into view_transform
        **kwargs,
    ) -> None:
        # voxelization config is passed in the preprocessor (BEVFusion convention)
        voxelize_cfg = {}
        if data_preprocessor is not None and 'voxelize_cfg' in data_preprocessor:
            voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # ---- Voxelization path ----
        self.voxelize_reduce = bool(voxelize_cfg.pop('voxelize_reduce', True))
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        # ---- Build submodules (unchanged API) ----
        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
        self.img_backbone = MODELS.build(img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        self.fusion_layer = MODELS.build(fusion_layer) if fusion_layer is not None else None
        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)
        self.bbox_head = MODELS.build(bbox_head)
        self.seg_head = MODELS.build(seg_head) if seg_head is not None else None

        # ---- Aux BEV head (tiny, train-time only) ----
        self.aux_on = aux_cfg is not None
        if self.aux_on:
            oc = getattr(self.view_transform, 'out_channels', 64)
            self.img_aux_head = nn.Sequential(
                nn.Conv2d(oc, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
            )
            self.aux_weight: float = float(aux_cfg.get('loss_weight', 0.1))
            self.aux_radius_cells: int = int(aux_cfg.get('radius_cells', 2))
            self.aux_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

            # cache BEV grid meta (from VT) to convert (x,y) → (iy,ix)
            self._xbound = tuple(getattr(self.view_transform, 'xbound'))
            self._ybound = tuple(getattr(self.view_transform, 'ybound'))
            self._downsample = int(getattr(self.view_transform, 'downsample', 1))

        # ---- Training knobs ----
        self.voxel_painting_on = bool(voxel_painting_on)
        self.use_two_scale_tokens = bool(use_two_scale_tokens)

        self.init_weights()

    # Boilerplate
    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log_vars = []
        for name, val in losses.items():
            if isinstance(val, torch.Tensor):
                log_vars.append([name, val.mean()])
            elif is_list_of(val, torch.Tensor):
                log_vars.append([name, sum(v.mean() for v in val)])
            else:
                raise TypeError(f'{name} is not a tensor or list of tensors')
        loss = sum(v for k, v in log_vars if 'loss' in k)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore
        if dist.is_available() and dist.is_initialized():
            for k, v in log_vars.items():
                v = v.data.clone()
                dist.all_reduce(v.div_(dist.get_world_size()))
                log_vars[k] = v.item()
        else:
            log_vars = {k: v.item() for k, v in log_vars.items()}
        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self): return hasattr(self, 'bbox_head') and self.bbox_head is not None
    @property
    def with_seg_head(self):  return hasattr(self, 'seg_head') and self.seg_head is not None

    # ---------- Image branch ----------
    def extract_img_feat(
        self,
        x,                      # [B,Nc,3,H,W]
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        """
        Runs backbone+neck → prepares tokens for VT.

        If `use_two_scale_tokens=True`, we pass [P3, P4] to VT (multiscale);
        otherwise just P3. CrossAttnLSSTransform accepts either a tensor
        [B,Nc,C,Hf,Wf] or a list of such tensors (it fuses internally).
        """
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        # Swin returns a list of stages; GeneralizedLSSFPN returns list[P3,P4,P5]
        x = self.img_backbone(x)
        x = self.img_neck(x)
        
        if hasattr(self, 'img_rpf_neck') and self.img_rpf_neck is not None:
            x = self.img_rpf_neck(x)   # keeps P3/P4/P5 shapes

        def _reshape_lvl(t):  # [B*Nc, C, Hf, Wf] → [B, Nc, C, Hf, Wf]
            C_, Hf, Wf = t.shape[1:]
            return t.view(B, N, C_, Hf, Wf)

        # Prepare inputs to VT
        if isinstance(x, (list, tuple)):
            P3 = _reshape_lvl(x[0])
            if self.use_two_scale_tokens and len(x) >= 2:
                P4 = _reshape_lvl(x[1])
                x_in = [P3, P4]           # multiscale input into VT
            else:
                x_in = P3
        else:
            x_in = _reshape_lvl(x)

        # IMPORTANT: This call is where the camera BEV is produced.
        # If you register a forward hook on self.view_transform, you can
        # capture that BEV tensor (used below for AUX loss).
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            bev = self.view_transform(
                x_in,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return bev

    # ---------- LiDAR branch ----------
    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [p.float() for p in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                f, c, n = ret
            else:  # dynamic voxelizer
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0 and self.voxelize_reduce:
            sizes = torch.cat(sizes, dim=0)
            feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
            feats = feats.contiguous()
        return feats, coords, sizes

    # ---------- Top-level feature path ----------
    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []

        if imgs is not None:
            imgs = imgs.contiguous()
            # Build per-batch camera projection / augmentation tensors
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for meta in batch_input_metas:
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.asarray(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))

            img_feature = self.extract_img_feat(
                imgs, deepcopy(points),
                lidar2image, camera_intrinsics, camera2lidar,
                img_aug_matrix, lidar_aug_matrix, batch_input_metas
            )
            features.append(img_feature)

        # LiDAR BEV (SECOND/spconv → BEV)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        # Late fusion in BEV space (BEVFusion-style)
        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)
        res = self.add_pred_to_datasample(batch_data_samples, outputs)
        return res

    # ---------- AUX loss + voxel painting (integrated cleanly) ----------
    @torch.no_grad()
    def _build_aux_target(self, data_samples: List, H: int, W: int, device) -> torch.Tensor:
        """Build class-agnostic center heatmap on the BEV grid."""
        xmin, xmax, dx = self._xbound
        ymin, ymax, dy = self._ybound
        dx_eff = dx * self._downsample
        dy_eff = dy * self._downsample

        target = torch.zeros((len(data_samples), 1, H, W), device=device, dtype=torch.float32)
        for b, ds in enumerate(data_samples):
            centers_xy = _boxes_center_xy(ds)
            if centers_xy is None:
                continue
            if centers_xy.device != device:
                centers_xy = centers_xy.to(device)
            ix = torch.floor((centers_xy[:, 0] - xmin) / dx_eff).long()
            iy = torch.floor((centers_xy[:, 1] - ymin) / dy_eff).long()
            valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            for (yy, xx) in zip(iy[valid].tolist(), ix[valid].tolist()):
                _draw_gaussian(target[b, 0], (yy, xx), radius=self.aux_radius_cells)
        return target

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample], **kwargs) -> Dict[str, torch.Tensor]:
        """
        We keep the standard BEVFusion loss path, but:
          • If AUX is enabled, register a forward hook on view_transform to capture
            the camera BEV (the output of VT). Compute aux loss on that feature.
          • If voxel_painting_on, register a pre-forward hook to stash the VT input
            (P3 or [P3,P4]) + metas into a thread-local store read by your VFE wrapper.
        """
        # Hooks around VT
        vt_handle = None
        vt_pre = None
        captured = {}

        if self.aux_on:
            def _vt_hook(module, in_tup, out):
                captured['img_bev'] = out  # [B, C_bev, Hy, Hx]
            vt_handle = self.view_transform.register_forward_hook(_vt_hook)

        if self.voxel_painting_on:
            metas = [item.metainfo for item in batch_data_samples]
            def _vt_pre_hook(module, in_tup):
                fpn_feats = in_tup[0]
                # If multiscale, pass the base level to painting (or modify your
                # painting code to accept a list).
                if isinstance(fpn_feats, (list, tuple)):
                    fpn_feats = fpn_feats[0]
                set_painting_context(fpn_feats, metas)
            vt_pre = self.view_transform.register_forward_pre_hook(_vt_pre_hook)

        try:
            batch_input_metas = [item.metainfo for item in batch_data_samples]
            feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

            losses = {}
            if self.with_bbox_head:
                losses.update(self.bbox_head.loss(feats, batch_data_samples))

            if self.aux_on:
                img_bev = captured.get('img_bev', None)
                if img_bev is None:
                    raise RuntimeError("AUX: view_transform output not captured (check VT call path).")
                B, C, H, W = img_bev.shape
                pred = self.img_aux_head(img_bev)  # [B,1,H,W]
                target = self._build_aux_target(batch_data_samples, H, W, img_bev.device)
                losses['loss_aux_img_bev'] = self.aux_loss_fn(pred, target) * float(self.aux_weight)

            return losses
        finally:
            if vt_handle is not None:
                vt_handle.remove()
            if vt_pre is not None:
                vt_pre.remove()
                clear_painting_context()