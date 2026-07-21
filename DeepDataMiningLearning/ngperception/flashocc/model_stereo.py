"""
flashocc.model_stereo
=====================
Standalone, modern-PyTorch (torch 2.x, py310, NO mmcv/mmdet/mmdet3d) port of the
**supervised** FlashOcc **BEVStereo4DOCC** model (R50 + 4D temporal + stereo cost
volume), for exact-fidelity checkpoint loading (`strict=True` -> 0 missing / 0
unexpected) and forward inference.

Ported faithfully from Others/FlashOCC/projects/mmdet3d_plugin/:
  - models/detectors/bevstereo4d.py   (BEVStereo4D: extract_stereo_ref_feat,
                                        prepare_bev_feat, extract_img_feat)
  - models/detectors/bevdet4d.py       (BEVDet4D: prepare_inputs, shift_feature,
                                        gen_grid)
  - models/detectors/bevdet.py         (BEVDet: image_encoder, bev_encoder)
  - models/detectors/bevdet_occ.py     (BEVStereo4DOCC forward path)
  - models/necks/view_transformer.py   (LSSViewTransformer / BEVDepth / BEVStereo)
  - models/model_utils/depthnet.py     (DepthNet w/ cost volume + SE/ASPP/mlp)
  - models/backbones/resnet.py         (CustomResNet BEV encoder)  [reused]
  - models/necks/lss_fpn.py            (FPN_LSS)                    [reused]
  - models/necks/fpn.py                (CustomFPN)                  [reused]
  - models/dense_heads/bev_occ_head.py (BEVOCCHead2D)              [reused]

Config baked in from projects/configs/flashocc/flashocc-r50-4d-stereo.py:
  grid_config = x[-40,40,0.4] y[-40,40,0.4] z[-1,5.4,6.4] depth[1.0,45.0,0.5] (D=88)
  input_size (256,704); numC_Trans=80; sid=True; downsample=16
  multi_adj_frame_id_cfg=(1,2,1) -> num_adj=1 -> num_frame=2, +extra_ref=1 -> 3
  img_backbone ResNet50 out_indices (0,2,3) -> [layer1(stereo ref), layer3, layer4]
  img_neck CustomFPN [1024,2048]->256 out_ids[0]
  img_view_transformer LSSViewTransformerBEVStereo in256 out80, DepthNet
      (use_dcn=False, aspp_mid_channels=96, stereo=True, bias=5.)
  pre_process CustomResNet numC_input=80 num_layer[1] num_channels[80] stride[1]
  img_bev_encoder_backbone CustomResNet numC_input=160 (=80*(num_adj+1))
      num_channels[160,320,640]
  img_bev_encoder_neck FPN_LSS in 640+160 out256
  occ_head BEVOCCHead2D in256 out256 Dz16 num_classes18 use_predicter

OUTPUT of forward(img_inputs): occ logits (B, 18, 200, 200, 16)
  == (B, num_classes, Dx, Dy, Dz), ready for cross-entropy vs (B,200,200,16).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the already-verified single-frame building blocks + the bev_pool_v2 op.
from .model import (
    bev_pool_v2,
    conv_module,
    BasicBlock,
    CustomResNet,
    FPN_LSS,
    LSSViewTransformer,
    ResNet50Backbone,  # noqa: F401  (kept for reference / possible external use)
)


# ======================================================================================
# 0. mmcv-ConvModule-compatible helper: names the conv submodule `.conv` (+ optional act)
#    so state_dict keys match `<prefix>.conv.weight` exactly (as in the checkpoint).
# ======================================================================================
class ConvModule(nn.Module):
    """Minimal mmcv.cnn.ConvModule: `.conv` (Conv2d) [+ ReLU act]. norm not needed here."""

    def __init__(self, in_c, out_c, k, stride=1, padding=0, bias=True, act='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, bias=bias)
        self.act = nn.ReLU(inplace=True) if act == 'relu' else None

    def forward(self, x):
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x


# ======================================================================================
# 0b. CustomFPN (image neck) with mmcv ConvModule naming (`.conv`), no norm / no act.
# ======================================================================================
class CustomFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs, start_level=0,
                 out_ids=(0,), upsample_mode='nearest'):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.backbone_end_level = self.num_ins
        self.out_ids = list(out_ids)
        self.upsample_mode = upsample_mode
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            self.lateral_convs.append(
                ConvModule(in_channels[i], out_channels, 1, bias=True, act=None))
            if i in self.out_ids:
                self.fpn_convs.append(
                    ConvModule(out_channels, out_channels, 3, padding=1, bias=True, act=None))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [lat(inputs[i + self.start_level])
                    for i, lat in enumerate(self.lateral_convs)]
        used = len(laterals)
        for i in range(used - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode=self.upsample_mode)
        outs = [self.fpn_convs[i](laterals[i]) for i in self.out_ids]
        return outs


# ======================================================================================
# 0c. BEVOCCHead2D with mmcv ConvModule naming for final_conv (`.conv`, +ReLU).
# ======================================================================================
class BEVOCCHead2D(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, Dz=16, num_classes=18, use_predicter=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        self.num_classes = num_classes
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(in_dim, out_channels, 3, stride=1, padding=1,
                                     bias=True, act='relu')
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

    def forward(self, img_feats):
        # (B, C, Dy, Dx) -> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)
        return occ_pred  # (B, Dx, Dy, Dz, num_classes)


# ======================================================================================
# 1. Image backbone: torchvision ResNet50 returning layer1 (stereo ref), layer3, layer4.
#    Matches mmdet ResNet(depth=50, out_indices=(0,2,3)) -> keys conv1/bn1/layer1..layer4.
# ======================================================================================
class ResNet50BackboneStereo(nn.Module):
    """torchvision resnet50; forward -> [layer1(256,/4), layer3(1024,/16), layer4(2048,/32)].

    Also exposes stem + layer1 for the stereo-reference feature path.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        net = resnet50(weights=weights)
        # Keep the exact submodule names so state_dict keys match img_backbone.*
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def stem(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def forward_layer1(self, x):
        """Stereo reference feat: stem + layer1 -> (B*N, 256, fH/4, fW/4)."""
        x = self.stem(x)
        x = self.layer1(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)    # (B, 256,  H/4,  W/4)   out_index 0  (stereo ref)
        c2 = self.layer2(c1)   # (B, 512,  H/8,  W/8)   out_index 1  (unused)
        c3 = self.layer3(c2)   # (B, 1024, H/16, W/16)  out_index 2
        c4 = self.layer4(c3)   # (B, 2048, H/32, W/32)  out_index 3
        return [c1, c3, c4]


# ======================================================================================
# 2. DepthNet (with stereo cost volume) — faithful port of model_utils/depthnet.py.
#    DCN dropped (use_dcn=False in config). BasicBlock reused from .model.
# ======================================================================================
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation,
                                     bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.atrous_conv(x)))


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256):
        super().__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(inplanes, mid_channels, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[3], dilation=dilations[3])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.dropout(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    """DepthNet with camera-aware context/depth SE, ASPP, and stereo cost volume."""

    def __init__(self, in_channels, mid_channels, context_channels, depth_channels,
                 use_dcn=False, use_aspp=True, with_cp=False, stereo=False, bias=0.0,
                 aspp_mid_channels=-1):
        super().__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1,
                                      stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)

        depth_conv_input_channels = mid_channels
        downsample = None
        if stereo:
            depth_conv_input_channels += depth_channels
            downsample = nn.Conv2d(depth_conv_input_channels, mid_channels, 1, 1, 0)
            cost_volumn_net = []
            for _ in range(2):
                cost_volumn_net.extend([
                    nn.Conv2d(depth_channels, depth_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(depth_channels)])
            self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
            self.bias = bias

        depth_conv_list = [
            BasicBlock(depth_conv_input_channels, mid_channels, downsample=downsample),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            if aspp_mid_channels < 0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        assert not use_dcn, "DCN not supported in standalone port (config uses use_dcn=False)"
        depth_conv_list.append(
            nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.with_cp = with_cp
        self.depth_channels = depth_channels

    def gen_grid(self, metas, B, N, D, H, W, hi, wi):
        frustum = metas['frustum']      # (D, fH_s, fW_s, 3)  3:(u,v,d)
        points = frustum - metas['post_trans'].view(B, N, 1, 1, 1, 3)
        points = torch.inverse(metas['post_rots']).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        rots = metas['k2s_sensor'][:, :, :3, :3].contiguous()
        trans = metas['k2s_sensor'][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas['intrins']))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points += trans.view(B, N, 1, 1, 1, 3, 1)
        neg_mask = points[..., 2, 0] < 1e-3
        points = metas['intrins'].view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points = points[..., :2, :] / points[..., 2:3, :]
        points = metas['post_rots'][..., :2, :2].view(B, N, 1, 1, 1, 2, 2).matmul(
            points).squeeze(-1)
        points += metas['post_trans'][..., :2].view(B, N, 1, 1, 1, 2)
        px = points[..., 0] / (wi - 1.0) * 2.0 - 1.0
        py = points[..., 1] / (hi - 1.0) * 2.0 - 1.0
        px[neg_mask] = -2
        py[neg_mask] = -2
        grid = torch.stack([px, py], dim=-1)
        grid = grid.view(B * N, D * H, W, 2)
        return grid

    def calculate_cost_volumn(self, metas):
        prev, curr = metas['cv_feat_list']
        group_size = 4
        _, c, hf, wf = curr.shape
        hi, wi = hf * 4, wf * 4
        B, N, _ = metas['post_trans'].shape
        D, H, W, _ = metas['frustum'].shape
        grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).to(curr.dtype)

        prev = prev.view(B * N, -1, H, W)
        curr = curr.view(B * N, -1, H, W)
        cost_volumn = 0
        for fid in range(curr.shape[1] // group_size):
            prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...]
            wrap_prev = F.grid_sample(prev_curr, grid, align_corners=True,
                                      padding_mode='zeros')
            curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
            cost_volumn_tmp = curr_tmp.unsqueeze(2) - \
                wrap_prev.view(B * N, -1, D, H, W)
            cost_volumn_tmp = cost_volumn_tmp.abs().sum(dim=1)
            cost_volumn += cost_volumn_tmp
        if not self.bias == 0:
            invalid = wrap_prev[:, 0, ...].view(B * N, D, H, W) == 0
            cost_volumn[invalid] = cost_volumn[invalid] + self.bias
        cost_volumn = - cost_volumn
        cost_volumn = cost_volumn.softmax(dim=1)
        return cost_volumn

    def forward(self, x, mlp_input, stereo_metas=None):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        if stereo_metas is not None:
            if stereo_metas['cv_feat_list'][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas['downsample']) / stereo_metas['cv_downsample']
                cost_volumn = torch.zeros(
                    (BN, self.depth_channels, int(H * scale_factor),
                     int(W * scale_factor))).to(x)
            else:
                with torch.no_grad():
                    cost_volumn = self.calculate_cost_volumn(stereo_metas)
            cost_volumn = self.cost_volumn_net(cost_volumn)
            depth = torch.cat([depth, cost_volumn], dim=1)
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)


# ======================================================================================
# 3. LSSViewTransformerBEVStereo — subclass of the single-frame LSSViewTransformer.
#    Replaces the trivial Conv2d depth_net with DepthNet; adds mlp_input + cv_frustum.
# ======================================================================================
class LSSViewTransformerBEVStereo(LSSViewTransformer):
    def __init__(self, grid_config, input_size, downsample=16, in_channels=256,
                 out_channels=80, sid=True, collapse_z=True, loss_depth_weight=0.05,
                 depthnet_cfg=None):
        super().__init__(grid_config=grid_config, input_size=input_size,
                         downsample=downsample, in_channels=in_channels,
                         out_channels=out_channels, accelerate=False, sid=sid,
                         collapse_z=collapse_z)
        depthnet_cfg = depthnet_cfg or {}
        self.loss_depth_weight = loss_depth_weight
        # Replace the base trivial depth_net (Conv2d) with the full DepthNet.
        self.depth_net = DepthNet(
            in_channels=self.in_channels,
            mid_channels=self.in_channels,
            context_channels=self.out_channels,
            depth_channels=self.D,
            **depthnet_cfg)
        # stereo cost-volume frustum at downsample=4 (not persisted in state_dict).
        cv_frustum = self.create_frustum(grid_config['depth'], input_size, downsample=4)
        self.register_buffer('cv_frustum', cv_frustum, persistent=False)

    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0], intrin[:, :, 1, 1], intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0], post_rot[:, :, 0, 1], post_tran[:, :, 0],
            post_rot[:, :, 1, 0], post_rot[:, :, 1, 1], post_tran[:, :, 1],
            bda[:, :, 0, 0], bda[:, :, 0, 1], bda[:, :, 1, 0], bda[:, :, 1, 1],
            bda[:, :, 2, 2],
        ], dim=-1)
        sensor2ego = sensor2ego[:, :, :3, :].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)  # (B, N, 27)
        return mlp_input

    def forward(self, input, stereo_metas=None):
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input, stereo_metas)   # (B*N, D + C_ctx, fH, fW)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        bev_feat, depth = self.view_transform(input, depth, tran_feat)
        return bev_feat, depth


# ======================================================================================
# 4. Top-level FlashOccBEVStereo4D detector.
# ======================================================================================
GRID_CONFIG = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

_CKPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "Others", "FlashOCC", "ckpts", "flashocc-r50-4d-stereo.pth")


class FlashOccBEVStereo4D(nn.Module):
    """Supervised FlashOcc BEVStereo4DOCC (camera-only, 4D temporal + stereo).

    forward(img_inputs) -> occ_logits (B, 18, 200, 200, 16) == (B, ncls, Dx, Dy, Dz)

    img_inputs is the standard BEVDet4D tuple/list of 7 tensors:
        imgs         : (B, N, 3, 256, 704)  with N = 6 * num_frame = 18
        sensor2egos  : (B, N, 4, 4)   cam->ego (per view)
        ego2globals  : (B, N, 4, 4)   ego->global (per view)
        intrins      : (B, N, 3, 3)   camera K at 256x704
        post_rots    : (B, N, 3, 3)   image-aug rotation
        post_trans   : (B, N, 3)      image-aug translation
        bda          : (B, 3, 3)      BEV data-aug rotation
    View ordering follows the reference prepare_inputs EXACTLY:
        imgs   flat dim-1 laid out as view(B, N_views=6, num_frame=3, ...)  (cam-major)
        calibs flat dim-1 laid out as view(B, num_frame=3, N_views=6, ...)  (frame-major)
    """

    def __init__(self, grid_config=GRID_CONFIG, input_size=(256, 704),
                 numC_Trans=80, num_adj=1, num_classes=18, Dz=16,
                 pretrained_img=False):
        super().__init__()
        # ---- temporal / stereo frame bookkeeping (BEVDet4D + BEVStereo4D __init__) ----
        self.num_adj = num_adj
        self.align_after_view_transfromation = False
        self.with_prev = True
        self.extra_ref_frames = 1
        self.num_frame = num_adj + 1          # 2
        self.temporal_frame = self.num_frame  # 2
        self.num_frame += self.extra_ref_frames  # 3
        self.grid = None
        self.pre_process = True

        # ---- submodules (names MUST match checkpoint prefixes) ----
        self.img_backbone = ResNet50BackboneStereo(pretrained=pretrained_img)
        self.img_neck = CustomFPN(in_channels=[1024, 2048], out_channels=256,
                                  num_outs=1, start_level=0, out_ids=[0])
        self.img_view_transformer = LSSViewTransformerBEVStereo(
            grid_config=grid_config, input_size=input_size, in_channels=256,
            out_channels=numC_Trans, sid=True, downsample=16,
            loss_depth_weight=0.05,
            depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96,
                              stereo=True, bias=5.))
        numC_input = numC_Trans * (num_adj + 1)  # 160
        self.img_bev_encoder_backbone = CustomResNet(
            numC_input=numC_input,
            num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8])
        self.img_bev_encoder_neck = FPN_LSS(
            in_channels=numC_Trans * 8 + numC_Trans * 2, out_channels=256)
        self.pre_process_net = CustomResNet(
            numC_input=numC_Trans, num_layer=[1], num_channels=[numC_Trans],
            stride=[1], backbone_output_ids=[0])
        self.occ_head = BEVOCCHead2D(in_dim=256, out_dim=256, Dz=Dz,
                                     num_classes=num_classes, use_predicter=True)

    # ------------------------------------------------------------------ encoders
    def image_encoder(self, img, stereo=False):
        """img: (B, N, 3, H, W) -> x: (B, N, 256, fH, fW), stereo_feat or None."""
        B, N, C, imH, imW = img.shape
        imgs = img.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)          # [layer1, layer3, layer4]
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]                        # [layer3, layer4]
        x = self.img_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        _, output_dim, oH, oW = x.shape
        x = x.view(B, N, output_dim, oH, oW)
        return x, stereo_feat

    def extract_stereo_ref_feat(self, x):
        """x: (B, N, 3, H, W) -> (B*N, 256, fH/4, fW/4) via stem + layer1."""
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        return self.img_backbone.forward_layer1(x)

    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    # ------------------------------------------------------------------ temporal align
    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None):
        B, C, H, W = input.shape
        if self.grid is None:
            xs = torch.linspace(0, W - 1, W, dtype=input.dtype,
                                device=input.device).view(1, W).expand(H, W)
            ys = torch.linspace(0, H - 1, H, dtype=input.dtype,
                                device=input.device).view(H, 1).expand(H, W)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
            self.grid = grid
        else:
            grid = self.grid
        grid = grid.view(1, H, W, 3).expand(B, H, W, 3).view(B, H, W, 3, 1)
        curr_sensor2keyego = sensor2keyegos[0][:, 0:1, :, :]
        prev_sensor2keyego = sensor2keyegos[1][:, 0:1, :, :]
        bda_ = torch.zeros((B, 1, 4, 4), dtype=grid.dtype).to(grid)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        curr_sensor2keyego = bda_.matmul(curr_sensor2keyego)
        if bda_adj is not None:
            bda_ = torch.zeros((B, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        prev_sensor2keyego = bda_.matmul(prev_sensor2keyego)
        keyego2adjego = curr_sensor2keyego.matmul(torch.inverse(prev_sensor2keyego))
        keyego2adjego = keyego2adjego.unsqueeze(dim=1)
        keyego2adjego = keyego2adjego[..., [True, True, False, True], :][..., [True, True, False, True]]
        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(keyego2adjego).matmul(feat2bev)
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([W - 1.0, H - 1.0], dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        return grid

    def shift_feature(self, input, sensor2keyegos, bda, bda_adj=None):
        grid = self.gen_grid(input, sensor2keyegos, bda, bda_adj=bda_adj)
        return F.grid_sample(input, grid.to(input.dtype), align_corners=True)

    # ------------------------------------------------------------------ inputs
    def prepare_inputs(self, img_inputs, stereo=False):
        B, N, C, H, W = img_inputs[0].shape
        N = N // self.num_frame     # N_views = 6
        imgs = img_inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = img_inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3),
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
            bda, curr2adjsensor

    # ------------------------------------------------------------------ per-frame bev
    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin, post_rot,
                         post_tran, bda, mlp_input, feat_prev_iv, k2s_sensor,
                         extra_ref_frame):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, stereo_feat
        x, stereo_feat = self.image_encoder(img, stereo=True)
        metas = dict(
            k2s_sensor=k2s_sensor,
            intrins=intrin,
            post_rots=post_rot,
            post_trans=post_tran,
            frustum=self.img_view_transformer.cv_frustum.to(x),
            cv_downsample=4,
            downsample=self.img_view_transformer.downsample,
            grid_config=self.img_view_transformer.grid_config,
            cv_feat_list=[feat_prev_iv, stereo_feat],
        )
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, stereo_feat

    # ------------------------------------------------------------------ full img feat
    def extract_img_feat(self, img_inputs):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
            bda, curr2adjsensor = self.prepare_inputs(img_inputs, stereo=True)

        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input, feat_prev_iv,
                               curr2adjsensor[fid], extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv = self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                if not key_frame:
                    feat_prev_iv = feat_curr_iv

        # with_prev=True: skip the zero-fill branch.
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame - 2):
                bev_feat_list[adj_id] = self.shift_feature(
                    bev_feat_list[adj_id],
                    [sensor2keyegos[0], sensor2keyegos[self.num_frame - 2 - adj_id]],
                    bda)

        bev_feat = torch.cat(bev_feat_list, dim=1)   # (B, 160, 200, 200)
        x = self.bev_encoder(bev_feat)
        return [x], depth_key_frame

    # ------------------------------------------------------------------ forward
    def forward(self, img_inputs):
        """img_inputs: list/tuple of 7 tensors (see class docstring).
        Returns occ logits (B, 18, 200, 200, 16)."""
        img_feats, _depth = self.extract_img_feat(img_inputs)
        occ_pred = self.occ_head(img_feats[0])   # (B, Dx, Dy, Dz, ncls)
        occ_pred = occ_pred.permute(0, 4, 1, 2, 3).contiguous()  # (B, ncls, Dx, Dy, Dz)
        return occ_pred

    def bev_feature(self, img_inputs):
        """The (B, 256, Dx, Dy) BEV feature fed to the occ head — the shared representation for
        multi-task heads (detection, world-model) on top of the strong occupancy backbone."""
        img_feats, _ = self.extract_img_feat(img_inputs)
        return img_feats[0]                       # (B, 256, 200, 200)

    # ------------------------------------------------------------------ ckpt loader
    @classmethod
    def from_official_checkpoint(cls, ckpt_path=None, map_location='cpu',
                                 strict=True, **kwargs):
        """Build the model and load the official checkpoint's 'state_dict'."""
        model = cls(pretrained_img=False, **kwargs)
        ckpt_path = ckpt_path or _CKPT_PATH
        sd = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if 'state_dict' in sd:
            sd = sd['state_dict']
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        return model, missing, unexpected
