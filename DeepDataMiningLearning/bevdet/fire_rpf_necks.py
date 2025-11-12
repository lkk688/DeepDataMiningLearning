"""
fire_rpf_necks.py (CBAM edition)

Single-file implementations of two light, shape-preserving necks with optional
CBAM attention inside each block:
  • FireRPF2DNeck  — image-side post-FPN enhancer (keeps P3/P4/P5 shapes)
  • FireRPFNeck    — LiDAR-side BEV neck that mirrors SECONDFPN behavior

New in this revision
--------------------
- Added **CBAM** (Convolutional Block Attention Module): Channel → Spatial
  attention with sigmoid gates. Enabled by default; can be disabled per neck.
- Block structure now follows the provided diagram:
    FireBlock (Squeeze 1×1 → Expand {1×1|3×3} → Concat → BN) + residual
    → CBAM → (optional) receptive-field mixing → output.

Design goals
------------
- Zero interface surprises: preserve #levels, spatial strides, and channels.
- Small parameter count: Fire (Squeeze/Expand) + light RF mixer + CBAM.
- Safe defaults: residuals on when shape-compatible; BN+ReLU; blocks_per_stage=1.
- H100-friendly: pure Conv2d/BN/ReLU/Pooling; no custom CUDA ops.

I/O conventions used in comments
--------------------------------
- B: batch size, C: channels, H/W: spatial dims (per-camera or BEV), L: #levels
- Image FPN levels: P3 (stride 8), P4 (16), P5 (32) typical
- LiDAR SECOND levels: e.g., [C_low=128 at higher res, C_high=256 at lower res]

Usage (config)
---------------
custom_imports = dict(
    imports=['projects.bevdet.fire_rpf_necks'],
    allow_failed_imports=False,
)

# Image RPF (shape-preserving):
img_rpf_neck=dict(
    type='FireRPF2DNeck', in_channels=[256,256,256], out_channels=256,
    num_outs=3, blocks_per_stage=1, with_residual=True,
    use_cbam=True, ca_reduction=16, sa_kernel=7
)

# LiDAR RPF (SECONDFPN-compatible):
pts_neck=dict(
    type='FireRPFNeck', in_channels=[128,256], out_channels=256,
    num_outs=2, upsample_strides=[1,2], blocks_per_stage=1, with_residual=True,
    use_cbam=True, ca_reduction=16, sa_kernel=7
)

Author: your friendly future self
"""
from __future__ import annotations
from typing import List, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # MMEngine/MMDet3D
    from mmengine.model import BaseModule
    from mmdet3d.registry import MODELS
except Exception as e:  # pragma: no cover
    raise ImportError("This file must be used inside an MMDetection3D environment.")


# ---------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------
class ConvBNAct(BaseModule):
    """Conv2d + BN + ReLU.

    Args:
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int
        padding: Optional[int] (default computes 'same')
        groups: int
        bias: bool
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        bias: bool = False,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=bias)
        # Keep standard BN2d and ReLU unless a different type is requested.
        self.bn = nn.BatchNorm2d(out_channels) if (norm_cfg is None or norm_cfg.get('type','BN2d')=='BN2d') else nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if (act_cfg is None or act_cfg.get('type','ReLU')=='ReLU') else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FireBlock2D(BaseModule):
    """Fire block (SqueezeNet-style) with end BN and optional residual.

    Pattern: 1×1 squeeze → (1×1 expand || 3×3 expand) → concat → BN → (+res)

    Shapes:
      in:  [B, Cin,  H,  W]
      out: [B, Cout, H', W']  (stride=1 keeps H'=H/W'=W)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float = 0.25,
        stride: int = 1,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
        with_residual: bool = True,
    ) -> None:
        super().__init__()
        assert 0 < squeeze_ratio <= 1.0
        s = max(8, int(in_channels * squeeze_ratio))
        e1 = out_channels // 2
        e3 = out_channels - e1

        self.squeeze = ConvBNAct(in_channels, s, kernel_size=1, stride=stride,
                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.expand1x1 = nn.Conv2d(s, e1, kernel_size=1, bias=False)
        self.expand3x3 = nn.Conv2d(s, e3, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        # Residual is applied **after** BN if shapes match and stride==1
        self.use_res = with_residual and (stride == 1) and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.squeeze(x)
        y = torch.cat([self.expand1x1(z), self.expand3x3(z)], dim=1)
        y = self.bn(y)
        if self.use_res:
            y = y + x
        return F.relu(y, inplace=True)


class ChannelAttention(BaseModule):
    """Channel attention in CBAM.

    Two pooled descriptors (avg & max over H×W) → shared MLP (1×1 convs) → sum → sigmoid.
    Scale the input channels with the gate.
    """
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        attn = self.mlp(avg) + self.mlp(mx)
        gate = self.sigmoid(attn)
        return x * gate


class SpatialAttention(BaseModule):
    """Spatial attention in CBAM.

    AvgPool & MaxPool along channels → concat → 7×7 conv → sigmoid → spatial gate.
    """
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in (3, 7)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        gate = self.sigmoid(self.conv(s))
        return x * gate


class CBAM(BaseModule):
    """Convolutional Block Attention Module: Channel → Spatial."""
    def __init__(self, channels: int, reduction: int = 16, sa_kernel: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(sa_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class RFPMix2D(BaseModule):
    """Tiny receptive-field mixer: depthwise 3×3 + dilated 3×3 + PW mix.

    Shapes: in/out [B, C, H, W]
    """
    def __init__(self, channels: int, norm_cfg: Optional[dict] = None, act_cfg: Optional[dict] = None) -> None:
        super().__init__()
        self.dw3 = ConvBNAct(channels, channels, kernel_size=3, groups=channels,
                             norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dil3 = ConvBNAct(channels, channels, kernel_size=3, groups=channels,
                              norm_cfg=norm_cfg, act_cfg=act_cfg)
        # set dilation and padding for shape preservation
        self.dil3.conv.dilation = (2, 2)
        self.dil3.conv.padding = (2, 2)
        self.pw = ConvBNAct(channels, channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw3(x)
        y = self.dil3(y)
        y = self.pw(y)
        return y


class FireRPFBlock2D(BaseModule):
    """FireBlock + optional CBAM + RFPMix + (optional) outer residual.

    Default ordering matches the diagram:
        FireBlock (with inner residual when shapes match) → CBAM → RFPMix

    Args:
        in_channels, out_channels: int
        with_residual: whether to apply an **outer** residual (on top of inner one)
        stride: only 1 is recommended in these necks; residuals disabled otherwise
        use_cbam: enable CBAM; ca_reduction & sa_kernel control its size
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        with_residual: bool = True,
        stride: int = 1,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
        use_cbam: bool = True,
        ca_reduction: int = 16,
        sa_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.fire = FireBlock2D(in_channels, out_channels, stride=stride,
                                norm_cfg=norm_cfg, act_cfg=act_cfg,
                                with_residual=True)  # inner residual when shapes match
        self.cbam = CBAM(out_channels, ca_reduction, sa_kernel) if use_cbam else None
        self.mix = RFPMix2D(out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.use_outer_res = with_residual and (stride == 1) and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fire(x)                 # Fire + inner residual
        if self.cbam is not None:
            y = self.cbam(y)             # CBAM gating
        y = self.mix(y)                  # receptive-field mixing
        if self.use_outer_res:
            y = y + x                    # outer residual for stability
        return y


# ---------------------------------------------------------------------
# 2D Image Neck (post-FPN enhancer): FireRPF2DNeck
# ---------------------------------------------------------------------
@MODELS.register_module()
class FireRPF2DNeck(BaseModule):
    """Shape-preserving neck for image features with CBAM blocks.

    Placement: AFTER your FPN to enhance P3/P4/P5 without changing shapes.

    Args:
        in_channels:  list[int], per-level input channels (e.g., [256,256,256])
        out_channels: int, per-level output channels (same for all levels)
        num_outs:     int, number of outputs to produce (usually == len(in_channels))
        blocks_per_stage: how many FireRPF blocks per level
        with_residual: use outer residual in the per-level blocks
        use_cbam: enable CBAM inside each block (default True)
        ca_reduction: channel-attention reduction ratio (default 16)
        sa_kernel: spatial-attention kernel size (7 or 3; default 7)
        init_cfg:     optional init cfg forwarded to BaseModule

    Forward I/O:
        x: list[Tensor] with shapes   [B, Cin[i], Hi, Wi] for i in 0..L-1
        y: list[Tensor] with shapes   [B, out_channels, Hi, Wi] for same i, length == num_outs
    """
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_outs: int,
        blocks_per_stage: int = 1,
        with_residual: bool = True,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
        use_cbam: bool = True,
        ca_reduction: int = 16,
        sa_kernel: int = 7,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_outs, int) and num_outs > 0
        assert num_outs <= len(in_channels)
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs

        stages = []
        for cin in self.in_channels[:num_outs]:
            layers: List[nn.Module] = []
            # Align channels if needed (1×1 conv, stride 1)
            if cin != out_channels:
                layers.append(ConvBNAct(cin, out_channels, kernel_size=1,
                                        norm_cfg=norm_cfg, act_cfg=act_cfg))
                cin_eff = out_channels
            else:
                cin_eff = cin
            # Stack small FireRPF+CBAM blocks (stride 1)
            for _ in range(blocks_per_stage):
                layers.append(FireRPFBlock2D(cin_eff, out_channels,
                                             with_residual=with_residual,
                                             stride=1, norm_cfg=norm_cfg, act_cfg=act_cfg,
                                             use_cbam=use_cbam,
                                             ca_reduction=ca_reduction,
                                             sa_kernel=sa_kernel))
                cin_eff = out_channels
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        assert isinstance(feats, (list, tuple)) and len(feats) >= self.num_outs, \
            "FireRPF2DNeck expects a list of feature maps."
        outs: List[torch.Tensor] = []
        for i in range(self.num_outs):
            outs.append(self.stages[i](feats[i]))
        return outs


# ---------------------------------------------------------------------
# LiDAR BEV Neck (SECONDFPN-compatible): FireRPFNeck
# ---------------------------------------------------------------------
@MODELS.register_module()
class FireRPFNeck(BaseModule):
    """LiDAR BEV neck mimicking SECONDFPN interface, with CBAM blocks.

    Typical SECOND setup (nuScenes-like): inputs are 2 levels
      x[0]: [B, 128, H,   W  ]   (higher spatial res)
      x[1]: [B, 256, H/2, W/2]

    This neck returns two outputs upsampled/aligned to the highest spatial res:
      y[0]: [B, 256, H,   W  ]   (from x[0], 1×1 + FireRPF+CBAM)
      y[1]: [B, 256, H,   W  ]   (from x[1], upsample stride 2 + FireRPF+CBAM)

    Args:
        in_channels:       list[int], e.g., [128, 256]
        out_channels:      int, unified channel count per output (e.g., 256)
        num_outs:          int, number of outputs (e.g., 2)
        upsample_strides:  list[int], spatial upsample factor per level (e.g., [1, 2])
        upsample_cfg:      dict, optional. Supported keys:
                           - type: 'interp' (default) or 'deconv'
                           - mode: when type=='interp', e.g., 'bilinear'
                           - align_corners: bool for interpolate
                           - bias: for deconv; kernel_size/stride are inferred from stride
        blocks_per_stage:  int, FireRPF+CBAM blocks per level
        with_residual:     bool, outer residual inside block
        use_cbam:          bool, enable CBAM inside each block
        ca_reduction:      int, channel attention reduction ratio
        sa_kernel:         int, spatial attention kernel size (3 or 7)
        init_cfg:          optional init cfg forwarded to BaseModule
    """
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_outs: int,
        upsample_strides: Optional[Sequence[int]] = None,
        upsample_cfg: Optional[dict] = None,
        blocks_per_stage: int = 1,
        with_residual: bool = True,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
        use_cbam: bool = True,
        ca_reduction: int = 16,
        sa_kernel: int = 7,
        init_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_outs, int) and num_outs > 0
        assert num_outs <= len(in_channels)
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs

        # Upsample strategy
        if upsample_strides is None:
            ups = [1] + [2 for _ in range(num_outs - 1)]
        else:
            ups = list(upsample_strides)
            assert len(ups) >= num_outs
        self.upsample_strides = ups[:num_outs]

        upcfg = upsample_cfg or {}
        self.up_type = upcfg.get('type', 'interp')  # 'interp' | 'deconv'
        self.up_mode = upcfg.get('mode', 'bilinear')
        self.up_align = bool(upcfg.get('align_corners', False))
        self.up_bias = bool(upcfg.get('bias', False))

        # Per-level channel align and processing blocks
        self.align_convs = nn.ModuleList()
        self.process_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for i in range(self.num_outs):
            cin = self.in_channels[i]
            self.align_convs.append(ConvBNAct(cin, out_channels, kernel_size=1,
                                              norm_cfg=norm_cfg, act_cfg=act_cfg))
            # Build per-level FireRPF+CBAM stack
            level_blocks: List[nn.Module] = []
            for _ in range(blocks_per_stage):
                level_blocks.append(FireRPFBlock2D(out_channels, out_channels,
                                                   with_residual=with_residual,
                                                   stride=1, norm_cfg=norm_cfg, act_cfg=act_cfg,
                                                   use_cbam=use_cbam,
                                                   ca_reduction=ca_reduction,
                                                   sa_kernel=sa_kernel))
            self.process_blocks.append(nn.Sequential(*level_blocks))

            # Upsampler per level
            s = self.upsample_strides[i]
            if s == 1:
                self.upsamplers.append(nn.Identity())
            else:
                if self.up_type == 'deconv':
                    # Simple stride-s deconvolution (kernel=stride) to scale spatial dims
                    self.upsamplers.append(
                        nn.ConvTranspose2d(out_channels, out_channels,
                                           kernel_size=s, stride=s, padding=0,
                                           bias=self.up_bias)
                    )
                else:  # 'interp'
                    self.upsamplers.append(nn.Identity())  # call interpolate in forward

    def _upsample(self, x: torch.Tensor, i: int, ref_hw: Sequence[int]) -> torch.Tensor:
        s = self.upsample_strides[i]
        if s == 1:
            y = x
        else:
            if self.up_type == 'deconv':
                y = self.upsamplers[i](x)
            else:
                y = F.interpolate(x, scale_factor=s, mode=self.up_mode, align_corners=self.up_align)
        # Safety: force exact match to reference spatial size
        if y.shape[-2:] != tuple(ref_hw):
            y = F.interpolate(y, size=tuple(ref_hw), mode=self.up_mode, align_corners=self.up_align)
        return y

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        assert isinstance(feats, (list, tuple)) and len(feats) >= self.num_outs, \
            "FireRPFNeck expects a list of feature maps from SECOND."
        outs: List[torch.Tensor] = []

        ref_h, ref_w = feats[0].shape[-2:]
        for i in range(self.num_outs):
            x = self.align_convs[i](feats[i])            # [B, C, Hi, Wi]
            x = self._upsample(x, i, (ref_h, ref_w))    # align to level-0 spatial size
            x = self.process_blocks[i](x)               # FireRPF+CBAM
            outs.append(x)
        return outs

@MODELS.register_module()
class SqueezeFPN(BaseModule):
    """A tiny FPN-like neck that gently mixes multi-scale features.

    Inputs:  feats[i] = [B, Cin[i], Hi, Wi] (i=0..L-1), Hi/Wi form a pyramid (P3..)
    Outputs: list of [B, C, Hi, Wi] with C=out_channels, same #levels.

    NOTE: This is a **very** small alternative — useful as a placeholder.
    """
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_outs: int,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        assert num_outs <= len(in_channels)
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs

        # Per-level lateral 1×1 to out_channels
        self.laterals = nn.ModuleList([
            ConvBNAct(cin, out_channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for cin in self.in_channels[:num_outs]
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        outs = [self.laterals[i](feats[i]) for i in range(self.num_outs)]
        # Simple top-down add (without upsample to keep original shapes)
        for i in range(self.num_outs - 2, -1, -1):
            hi, wi = outs[i].shape[-2:]
            up = F.interpolate(outs[i + 1], size=(hi, wi), mode='bilinear', align_corners=False)
            outs[i] = outs[i] + up
        return outs


# ---------------------------------------------------------------------
# Identity necks (optional fallbacks for toggling without code changes)
# ---------------------------------------------------------------------
@MODELS.register_module()
class Identity2DNeck(BaseModule):
    """Pass-through for a list of image feature maps.

    Requires each in_channels[i] == out_channels.
    """
    def __init__(self, in_channels: Sequence[int], out_channels: int, num_outs: int, **kwargs) -> None:
        super().__init__()
        assert all(c == out_channels for c in in_channels[:num_outs]), \
            'Identity2DNeck requires in_channels == out_channels per level'
        self.num_outs = num_outs

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return list(feats[:self.num_outs])


@MODELS.register_module()
class IdentityNeck(BaseModule):
    """Pass-through for a list of BEV feature maps.

    Requires each in_channels[i] == out_channels.
    """
    def __init__(self, in_channels: Sequence[int], out_channels: int, num_outs: int, **kwargs) -> None:
        super().__init__()
        assert all(c == out_channels for c in in_channels[:num_outs]), \
            'IdentityNeck requires in/out channels match per level'
        self.num_outs = num_outs

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return list(feats[:self.num_outs])
