"""
flashocc.model
==============
Standalone, modern-PyTorch (torch 2.x, NO mmcv/mmdet/mmdet3d) port of the single-frame
FlashOcc **BEVDetOCC** model (ResNet50 + CustomFPN + LSS view transform + CustomResNet BEV
encoder + FPN_LSS + BEVOCCHead2D), for label-free occupancy training.

Ported from Others/FlashOCC/projects/mmdet3d_plugin/:
  - models/necks/view_transformer.py  (LSSViewTransformer base)
  - models/backbones/resnet.py        (CustomResNet BEV encoder)
  - models/necks/lss_fpn.py           (FPN_LSS)
  - models/necks/fpn.py               (CustomFPN)
  - models/dense_heads/bev_occ_head.py(BEVOCCHead2D)
  - models/detectors/bevdet.py + bevdet_occ.py (forward path)

Config baked in from projects/configs/flashocc/flashocc-r50.py:
  grid_config = x[-40,40,0.4] y[-40,40,0.4] z[-1,5.4,6.4] depth[1.0,45.0,0.5]
  input_size (256,704); numC_Trans=64; ResNet50 out layer3(1024)/layer4(2048)
  CustomFPN [1024,2048]->256 num_outs1 out_ids[0]
  LSSViewTransformer in256 out64 sid=False collapse_z=True downsample16 -> D=88, Dz=1
  CustomResNet numC_input64 num_channels[128,256,512]
  FPN_LSS in 512+128 out256
  BEVOCCHead2D in256 out256 Dz16 num_classes18 use_predicter -> (B,200,200,16,18)

OUTPUT LAYOUT of forward(imgs, rots, trans, intrins):
    occ_logits with shape (B, 18, 200, 200, 16) == (B, num_classes, Dx, Dy, Dz).
    This is the (B, C, X, Y, Z) layout that plugs directly into
    gaussian4d.train_student.teacher_loss / F.cross_entropy against a
    (B, 200, 200, 16) integer teacher-semantics target.

The pre-compiled CUDA op `bev_pool_v2` is (re)built/loaded via torch.utils.cpp_extension.load
from the FlashOCC ops sources; the python autograd wrapper (QuickCumsumCuda / bev_pool_v2) is
copied verbatim from ops/bev_pool_v2/bev_pool.py.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# 0. bev_pool_v2 CUDA op: load the already-compiling FlashOCC sources via cpp_extension.
# --------------------------------------------------------------------------------------
_CUDA_HOME = os.environ.setdefault("CUDA_HOME", "/data/rnd-liu/cuda_home2")
_cuda_bin = os.path.join(_CUDA_HOME, "bin")
if _cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _cuda_bin + os.pathsep + os.environ.get("PATH", "")
_cuda_lib = os.path.join(_CUDA_HOME, "lib64")
if _cuda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _cuda_lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

_OPS_SRC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "Others", "FlashOCC", "projects", "mmdet3d_plugin",
        "ops", "bev_pool_v2", "src",
    )
)

_bev_pool_v2_ext = None


def _load_ext():
    """Lazily JIT-load (or reuse the cached build of) the bev_pool_v2 CUDA extension."""
    global _bev_pool_v2_ext
    if _bev_pool_v2_ext is None:
        from torch.utils.cpp_extension import load
        _bev_pool_v2_ext = load(
            name="bev_pool_v2_ext",
            sources=[
                os.path.join(_OPS_SRC, "bev_pool.cpp"),
                os.path.join(_OPS_SRC, "bev_pool_cuda.cu"),
            ],
            verbose=True,
        )
    return _bev_pool_v2_ext


# --------------------------------------------------------------------------------------
# 1. bev_pool_v2 python wrapper — copied verbatim from ops/bev_pool_v2/bev_pool.py
#    (only `from . import bev_pool_v2_ext` -> the JIT-loaded module).
# --------------------------------------------------------------------------------------
class QuickCumsumCuda(torch.autograd.Function):
    r"""BEVPoolv2 implementation for Lift-Splat-Shoot view transformation."""

    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        ext = _load_ext()
        ranks_bev = ranks_bev.int()
        depth = depth.contiguous().float()
        feat = feat.contiguous().float()
        ranks_depth = ranks_depth.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()

        out = feat.new_zeros(bev_feat_shape)  # (B, D_Z, D_Y, D_X, C)

        ext.bev_pool_v2_forward(
            depth, feat, out, ranks_depth, ranks_feat, ranks_bev,
            interval_lengths, interval_starts,
        )

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ext = _load_ext()
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors

        order = ranks_feat.argsort()
        ranks_feat, ranks_depth, ranks_bev = \
            ranks_feat[order], ranks_depth[order], ranks_bev[order]
        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts_bp = torch.where(kept)[0].int()
        interval_lengths_bp = torch.zeros_like(interval_starts_bp)
        interval_lengths_bp[:-1] = interval_starts_bp[1:] - interval_starts_bp[:-1]
        interval_lengths_bp[-1] = ranks_bev.shape[0] - interval_starts_bp[-1]

        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous()
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths_bp = interval_lengths_bp.contiguous()
        interval_starts_bp = interval_starts_bp.contiguous()

        depth_grad = depth.new_zeros(depth.shape)
        feat_grad = feat.new_zeros(feat.shape)
        out_grad = out_grad.contiguous()
        ext.bev_pool_v2_backward(
            out_grad, depth_grad, feat_grad, depth, feat,
            ranks_depth, ranks_feat, ranks_bev,
            interval_lengths_bp, interval_starts_bp,
        )
        return depth_grad, feat_grad, None, None, None, None, None, None, None, None


def bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    """Returns bev feature in shape (B, C, Dz, Dy, Dx)."""
    x = QuickCumsumCuda.apply(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                              bev_feat_shape, interval_starts, interval_lengths)
    x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, Dz, Dy, Dx)
    return x


# --------------------------------------------------------------------------------------
# 2. small helpers replacing mmcv ConvModule / build_norm_layer.
# --------------------------------------------------------------------------------------
def conv_module(in_c, out_c, k, stride=1, padding=0, bias=True, norm=False, act=None,
                dim=2):
    """Conv (+ optional BN) (+ optional act). act in {None, 'relu'}."""
    Conv = nn.Conv2d if dim == 2 else nn.Conv3d
    BN = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
    layers = [Conv(in_c, out_c, k, stride=stride, padding=padding, bias=bias)]
    if norm:
        layers.append(BN(out_c))
    if act == 'relu':
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------------------
# 3. Image backbone: torchvision ResNet50, returning layer3 (1024) + layer4 (2048).
# --------------------------------------------------------------------------------------
class ResNet50Backbone(nn.Module):
    """torchvision resnet50 truncated to feature maps; returns [C3(1024,/16), C4(2048,/32)]."""

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        net = resnet50(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)   # (B, 1024, H/16, W/16)
        c4 = self.layer4(c3)  # (B, 2048, H/32, W/32)
        return [c3, c4]


# --------------------------------------------------------------------------------------
# 4. CustomFPN (image neck). Faithful port; our config: [1024,2048]->256, out_ids=[0],
#    no norm, no act (nearest-upsample top-down).
# --------------------------------------------------------------------------------------
class CustomFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs, start_level=0,
                 out_ids=(0,), upsample_mode='nearest'):
        super().__init__()
        assert isinstance(in_channels, (list, tuple))
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
                conv_module(in_channels[i], out_channels, 1, bias=True, norm=False, act=None))
            if i in self.out_ids:
                self.fpn_convs.append(
                    conv_module(out_channels, out_channels, 3, padding=1, bias=True,
                                norm=False, act=None))

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


# --------------------------------------------------------------------------------------
# 5. LSSViewTransformer (base) — ported. force_fp32/registry/DepthNet subclasses dropped.
# --------------------------------------------------------------------------------------
class LSSViewTransformer(nn.Module):
    def __init__(self, grid_config, input_size, downsample=16, in_channels=512,
                 out_channels=64, accelerate=False, sid=False, collapse_z=True):
        super().__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.sid = sid
        frustum = self.create_frustum(grid_config['depth'], input_size, downsample)
        self.register_buffer('frustum', frustum, persistent=False)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2d(in_channels, self.D + self.out_channels,
                                   kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.initial_flag = True
        self.collapse_z = collapse_z

    def create_grid_infos(self, x, y, z, **kwargs):
        self.register_buffer('grid_lower_bound',
                             torch.Tensor([cfg[0] for cfg in [x, y, z]]), persistent=False)
        self.register_buffer('grid_interval',
                             torch.Tensor([cfg[2] for cfg in [x, y, z]]), persistent=False)
        self.register_buffer('grid_size',
                             torch.Tensor([(cfg[1] - cfg[0]) / cfg[2] for cfg in [x, y, z]]),
                             persistent=False)

    def create_frustum(self, depth_cfg, input_size, downsample):
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float) \
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D - 1) *
                              torch.log((depth_cfg_t[1] - 1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        x = torch.linspace(0, W_in - 1, W_feat, dtype=torch.float) \
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat, dtype=torch.float) \
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)
        return torch.stack((x, y, d), -1)  # (D, fH, fW, 3)

    def get_ego_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda):
        B, N, _, _ = sensor2ego.shape
        # post-transformation
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))
        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2], int(self.grid_size[2]),
                int(self.grid_size[1]), int(self.grid_size[0])]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)  # (B, N, fH, fW, C)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Dz, Dy, Dx, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts, interval_lengths)
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)  # (B, C*Dz, Dy, Dx)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor):
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        ranks_depth = torch.arange(0, num_points, dtype=torch.int, device=coor.device)
        ranks_feat = torch.arange(0, num_points // D, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) / self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.arange(0, B).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)
        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = coor[kept], ranks_depth[kept], ranks_feat[kept]
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]
        kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return (ranks_bev.int().contiguous(), ranks_depth.int().contiguous(),
                ranks_feat.int().contiguous(), interval_starts.int().contiguous(),
                interval_lengths.int().contiguous())

    def view_transform_core(self, input, depth, tran_feat):
        B, N, C, H, W = input[0].shape
        coor = self.get_ego_coor(*input[1:7])
        bev_feat = self.voxel_pooling_v2(
            coor, depth.view(B, N, self.D, H, W),
            tran_feat.view(B, N, self.out_channels, H, W))
        return bev_feat, depth

    def view_transform(self, input, depth, tran_feat):
        return self.view_transform_core(input, depth, tran_feat)

    def forward(self, input):
        """input = [x(B,N,C_in,fH,fW), sensor2ego, ego2global, intrins,
                    post_rots, post_trans, bda]. Returns (bev_feat(B,C*Dz,Dy,Dx), depth)."""
        x = input[0]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        return self.view_transform(input, depth, tran_feat)


# --------------------------------------------------------------------------------------
# 6. CustomResNet BEV encoder (uses a plain ResNet BasicBlock, replacing mmdet's).
# --------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class CustomResNet(nn.Module):
    def __init__(self, numC_input, num_layer=(2, 2, 2), num_channels=None,
                 stride=(2, 2, 2), backbone_output_ids=None):
        super().__init__()
        assert len(num_layer) == len(stride)
        num_channels = [numC_input * 2 ** (i + 1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        curr_numC = numC_input
        for i in range(len(num_layer)):
            layer = [BasicBlock(curr_numC, num_channels[i], stride=stride[i],
                                downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                                     stride[i], 1))]
            curr_numC = num_channels[i]
            layer.extend([BasicBlock(curr_numC, num_channels[i], stride=1, downsample=None)
                          for _ in range(num_layer[i] - 1)])
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


# --------------------------------------------------------------------------------------
# 7. FPN_LSS (BEV neck) — ported.
# --------------------------------------------------------------------------------------
class FPN_LSS(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4,
                 input_feature_index=(0, 2), extra_upsample=2, lateral=None):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)
        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * channels_factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels * channels_factor),
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1, padding=0),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, 1, padding=0, bias=False),
                nn.BatchNorm2d(lateral),
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        if self.extra_upsample:
            x = self.up2(x)
        return x


# --------------------------------------------------------------------------------------
# 8. BEVOCCHead2D — ported (forward only; loss/get_occ machinery dropped).
# --------------------------------------------------------------------------------------
class BEVOCCHead2D(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, Dz=16, num_classes=18, use_predicter=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        self.num_classes = num_classes
        out_channels = out_dim if use_predicter else num_classes * Dz
        # mmcv ConvModule default act_cfg=ReLU, norm_cfg=None -> Conv2d + ReLU.
        self.final_conv = conv_module(in_dim, out_channels, 3, stride=1, padding=1,
                                      bias=True, norm=False, act='relu')
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


# --------------------------------------------------------------------------------------
# 9. Top-level model.
# --------------------------------------------------------------------------------------
GRID_CONFIG = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}


class FlashOccBEVDet(nn.Module):
    """Single-frame FlashOcc BEVDetOCC (camera-only).

    forward(imgs, rots, trans, intrins) -> occ_logits (B, 18, 200, 200, 16)
        imgs    : (B, 6, 3, 256, 704)  normalized
        rots    : (B, 6, 3, 3)  cam->ego rotation
        trans   : (B, 6, 3)     cam->ego translation
        intrins : (B, 6, 3, 3)  camera K at the 256x704 resolution
    post_rots=I, post_trans=0, ego2global=I, bda=I are synthesized internally.
    """

    def __init__(self, grid_config=GRID_CONFIG, input_size=(256, 704), numC_Trans=64,
                 num_classes=18, Dz=16, pretrained_img=True):
        super().__init__()
        self.img_backbone = ResNet50Backbone(pretrained=pretrained_img)
        self.img_neck = CustomFPN(in_channels=[1024, 2048], out_channels=256,
                                  num_outs=1, start_level=0, out_ids=[0])
        self.img_view_transformer = LSSViewTransformer(
            grid_config=grid_config, input_size=input_size, in_channels=256,
            out_channels=numC_Trans, sid=False, collapse_z=True, downsample=16)
        self.img_bev_encoder_backbone = CustomResNet(
            numC_input=numC_Trans,
            num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8])
        self.img_bev_encoder_neck = FPN_LSS(
            in_channels=numC_Trans * 8 + numC_Trans * 2, out_channels=256)
        self.occ_head = BEVOCCHead2D(in_dim=256, out_dim=256, Dz=Dz,
                                     num_classes=num_classes, use_predicter=True)

    def image_encoder(self, img):
        B, N, C, imH, imW = img.shape
        imgs = img.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)          # [C3(1024), C4(2048)]
        x = self.img_neck(x)                 # [ (B*N,256,fH,fW) ]
        if isinstance(x, (list, tuple)):
            x = x[0]
        _, output_dim, oH, oW = x.shape
        x = x.view(B, N, output_dim, oH, oW)
        return x

    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def _build_calib(self, rots, trans, intrins):
        """Build the LSSViewTransformer input calib list from our per-cam tensors."""
        B, N = rots.shape[:2]
        dev, dt = rots.device, rots.dtype
        sensor2ego = torch.zeros(B, N, 4, 4, device=dev, dtype=dt)
        sensor2ego[..., :3, :3] = rots
        sensor2ego[..., :3, 3] = trans
        sensor2ego[..., 3, 3] = 1.0
        ego2global = torch.eye(4, device=dev, dtype=dt).view(1, 1, 4, 4).expand(B, N, 4, 4)
        post_rots = torch.eye(3, device=dev, dtype=dt).view(1, 1, 3, 3).expand(B, N, 3, 3)
        post_trans = torch.zeros(B, N, 3, device=dev, dtype=dt)
        bda = torch.eye(3, device=dev, dtype=dt).view(1, 3, 3).expand(B, 3, 3)
        return sensor2ego, ego2global, intrins, post_rots, post_trans, bda

    def forward(self, imgs, rots, trans, intrins):
        """Returns occ logits (B, num_classes=18, Dx=200, Dy=200, Dz=16)."""
        x = self.image_encoder(imgs)  # (B, N, 256, fH, fW)
        calib = self._build_calib(rots, trans, intrins)
        bev_feat, _depth = self.img_view_transformer([x] + list(calib))  # (B, 64, 200, 200)
        bev_feat = self.bev_encoder(bev_feat)                            # (B, 256, 200, 200)
        occ_pred = self.occ_head(bev_feat)  # (B, Dx, Dy, Dz, num_classes)
        # -> (B, num_classes, Dx, Dy, Dz) for (B,C,X,Y,Z) cross-entropy against (B,X,Y,Z).
        occ_pred = occ_pred.permute(0, 4, 1, 2, 3).contiguous()
        return occ_pred
