"""
ngperception.detection.pointpillars
====================================

A **pure-PyTorch PointPillars** LiDAR 3D detector — no spconv, no mmcv, no compiled ops.
The algorithm is harvested from `2D3DFusion/mydetector3d` (an OpenPCDet fork): pillar VFE,
BEV scatter, 2-D BEV backbone, anchor head, residual box coder and focal/smooth-L1 losses
are all pure `torch.nn`; the only CUDA piece there (`ops/iou3d_nms`) is replaced by
[`box_utils`](box_utils.py)'s pure-torch IoU/NMS. Pillarization (points→pillars), which
OpenPCDet does with a spconv voxel generator, is a small vectorised numpy routine here.

Pipeline (per sample): points → **pillarize** → **PillarVFE** (per-pillar PointNet) →
**scatter** to a BEV pseudo-image → **BEV backbone** (2-D CNN) → **anchor head**
(cls + box conv) → decode + **NMS**.

Boxes are 7-DoF **[x, y, z, dx, dy, dz, heading]**. Defaults are KITTI-Car. This is the M0
skeleton (see PLAN.md): the fast IoU path is axis-aligned (exact rotated IoU is in box_utils
for eval); M1 wires the real KITTI loader + mAP, M3 fuses it onto the shared occ encoder.

    python -m DeepDataMiningLearning.ngperception.detection.pointpillars   # forward + loss smoke
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_utils import (ResidualCoder, boxes_bev_iou_aligned, nms_aligned,
                        rotated_iou_assign, limit_period)
from .losses import SigmoidFocalClassificationLoss, WeightedSmoothL1Loss


# --------------------------------------------------------------------------- #
# pillarization: points -> (pillars, counts, coords) — vectorised numpy, no spconv
# --------------------------------------------------------------------------- #
def pillarize(points, pc_range, voxel_size, max_points=32, max_pillars=30000):
    """points (N,C>=3) -> voxels (P,max_points,C), num (P,), coords (P,3) int [x,y,z].
    Groups points into pillars (nz=1) with a per-pillar point cap. Pure numpy."""
    points = np.asarray(points, np.float32)
    pcr, vs = np.asarray(pc_range, np.float32), np.asarray(voxel_size, np.float32)
    grid = np.round((pcr[3:] - pc_range[:3]) / vs).astype(np.int64)         # (nx,ny,nz)
    m = np.all((points[:, :3] >= pcr[:3]) & (points[:, :3] < pcr[3:]), axis=1)
    points = points[m]
    if len(points) == 0:                                                    # empty guard
        return (np.zeros((0, max_points, points.shape[1] if points.ndim == 2 else 4), np.float32),
                np.zeros(0, np.int64), np.zeros((0, 3), np.int64))
    idx = np.floor((points[:, :3] - pcr[:3]) / vs).astype(np.int64)
    idx = np.clip(idx, 0, grid - 1)
    keys = idx[:, 0] * (grid[1] * grid[2]) + idx[:, 1] * grid[2] + idx[:, 2]
    uniq, inv = np.unique(keys, return_inverse=True)
    P = len(uniq)
    order = np.argsort(inv, kind="stable")
    inv_s, pts_s = inv[order], points[order]
    counts = np.bincount(inv_s, minlength=P)
    within = np.arange(len(inv_s)) - np.repeat(np.cumsum(counts) - counts, counts)  # pos in pillar
    sel = within < max_points
    voxels = np.zeros((P, max_points, points.shape[1]), np.float32)
    voxels[inv_s[sel], within[sel]] = pts_s[sel]
    num = np.minimum(counts, max_points).astype(np.int64)
    coords = np.stack([uniq // (grid[1] * grid[2]),
                       (uniq // grid[2]) % grid[1], uniq % grid[2]], axis=1)   # (P,3) x,y,z
    if P > max_pillars:                                                     # cap (keep densest)
        keep = np.argsort(-num)[:max_pillars]
        voxels, num, coords = voxels[keep], num[keep], coords[keep]
    return voxels, num, coords


class PillarVFE(nn.Module):
    """Per-pillar PointNet (OpenPCDet PillarVFE): 10-dim point features -> pillar embedding."""

    def __init__(self, num_point_features, voxel_size, pc_range, out_channels=64, use_abs_xyz=True):
        super().__init__()
        self.use_abs_xyz = use_abs_xyz
        in_ch = num_point_features + (6 if use_abs_xyz else 3)
        self.linear = nn.Linear(in_ch, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.vx, self.vy, self.vz = voxel_size
        self.x0 = self.vx / 2 + pc_range[0]
        self.y0 = self.vy / 2 + pc_range[1]
        self.z0 = self.vz / 2 + pc_range[2]
        self.out_channels = out_channels

    def forward(self, voxels, num, coords):
        """voxels (P,T,C), num (P,), coords (P,3)[x,y,z] -> pillar_features (P,out)."""
        pts_mean = voxels[:, :, :3].sum(1, keepdim=True) / num.type_as(voxels).clamp(min=1).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - pts_mean
        f_center = torch.zeros_like(voxels[:, :, :3])
        f_center[:, :, 0] = voxels[:, :, 0] - (coords[:, 0].type_as(voxels).unsqueeze(1) * self.vx + self.x0)
        f_center[:, :, 1] = voxels[:, :, 1] - (coords[:, 1].type_as(voxels).unsqueeze(1) * self.vy + self.y0)
        f_center[:, :, 2] = voxels[:, :, 2] - (coords[:, 2].type_as(voxels).unsqueeze(1) * self.vz + self.z0)
        feats = [voxels, f_cluster, f_center] if self.use_abs_xyz else [voxels[..., 3:], f_cluster, f_center]
        feats = torch.cat(feats, dim=-1)                                    # (P,T,in_ch)
        # mask padded points
        mask = (torch.arange(voxels.shape[1], device=voxels.device)[None, :] < num[:, None]).type_as(voxels)
        feats = feats * mask.unsqueeze(-1)
        x = self.linear(feats)                                              # (P,T,out)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1).relu()
        return x.max(dim=1)[0]                                              # (P,out)


def scatter_bev(pillar_features, coords, nx, ny):
    """Scatter pillar features to a dense BEV image. -> (C, ny, nx)."""
    C = pillar_features.shape[1]
    canvas = pillar_features.new_zeros(C, ny * nx)
    flat = (coords[:, 1] * nx + coords[:, 0]).long()                        # y*nx + x
    canvas[:, flat] = pillar_features.t()
    return canvas.view(C, ny, nx)


class BaseBEVBackbone(nn.Module):
    """2-D BEV backbone (OpenPCDet): multi-scale conv blocks + transpose-conv upsample, concat."""

    def __init__(self, in_channels, layer_nums=(3, 5, 5), layer_strides=(2, 2, 2),
                 num_filters=(64, 128, 256), upsample_strides=(1, 2, 4), num_upsample=(128, 128, 128)):
        super().__init__()
        self.blocks, self.deblocks = nn.ModuleList(), nn.ModuleList()
        c_in_list = [in_channels, *num_filters[:-1]]
        for i in range(len(layer_nums)):
            layers = [nn.ZeroPad2d(1),
                      nn.Conv2d(c_in_list[i], num_filters[i], 3, stride=layer_strides[i], padding=0, bias=False),
                      nn.BatchNorm2d(num_filters[i], eps=1e-3, momentum=0.01), nn.ReLU()]
            for _ in range(layer_nums[i]):
                layers += [nn.Conv2d(num_filters[i], num_filters[i], 3, padding=1, bias=False),
                           nn.BatchNorm2d(num_filters[i], eps=1e-3, momentum=0.01), nn.ReLU()]
            self.blocks.append(nn.Sequential(*layers))
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(num_filters[i], num_upsample[i], upsample_strides[i],
                                   stride=upsample_strides[i], bias=False),
                nn.BatchNorm2d(num_upsample[i], eps=1e-3, momentum=0.01), nn.ReLU()))
        self.num_bev_features = sum(num_upsample)

    def forward(self, x):
        ups = []
        for blk, deb in zip(self.blocks, self.deblocks):
            x = blk(x)
            ups.append(deb(x))
        return torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]            # (B, sum_up, H, W)


class BasicBlock2d(nn.Module):
    """ResNet basic residual block (2-D)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                      nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(x)) + idt)


class ResBEVBackbone(nn.Module):
    """PillarNeXt-style BEV backbone: a ResNet encoder + FPN neck, pure 2-D conv (no spconv).
    Stronger than BaseBEVBackbone (residual blocks + multi-scale fusion); same output stride (/2)
    so the detection heads plug in unchanged."""

    def __init__(self, in_channels, out_channels=384, blocks=(2, 2, 2), channels=(64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, channels[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels[0], eps=1e-3, momentum=0.01), nn.ReLU(inplace=True))
        self.stages = nn.ModuleList()
        cin = channels[0]
        for n, cout in zip(blocks, channels):
            layers = [BasicBlock2d(cin, cout, stride=2)] + [BasicBlock2d(cout, cout) for _ in range(n - 1)]
            self.stages.append(nn.Sequential(*layers)); cin = cout
        fdim = out_channels // len(channels)
        self.lat = nn.ModuleList([nn.Conv2d(c, fdim, 1) for c in channels])   # FPN laterals
        self.num_bev_features = fdim * len(channels)

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for st in self.stages:
            x = st(x); feats.append(x)                              # /2, /4, /8
        target = feats[0].shape[-2:]
        outs = [self.lat[i](f) for i, f in enumerate(feats)]
        outs = [o if o.shape[-2:] == target else
                F.interpolate(o, size=target, mode="bilinear", align_corners=False) for o in outs]
        return torch.cat(outs, dim=1)                               # (B, fdim*3, H/2, W/2)


def make_bev_backbone(kind, in_channels):
    return ResBEVBackbone(in_channels) if kind == "res" else BaseBEVBackbone(in_channels)


class AnchorHead(nn.Module):
    """Single-group anchor head: cls + box conv, anchor generation, target assign, loss, decode."""

    def __init__(self, in_channels, pc_range, num_classes=1,
                 anchor_sizes=((3.9, 1.6, 1.56),), anchor_rotations=(0, np.pi / 2),
                 anchor_bottom=-1.78, pos_thresh=0.6, neg_thresh=0.45,
                 rotated_assign=False, use_dir=False, dir_offset=0.7854, dir_weight=0.2):
        super().__init__()
        self.pcr = list(pc_range)
        self.num_classes = num_classes
        self.sizes, self.rots, self.bottom = list(anchor_sizes), list(anchor_rotations), anchor_bottom
        self.A = len(self.sizes) * len(self.rots)
        self.pos_thresh, self.neg_thresh = pos_thresh, neg_thresh
        self.rotated_assign = rotated_assign            # M2: rotated-IoU target assignment
        self.use_dir = use_dir                          # M2: direction classifier (2 bins)
        self.dir_offset, self.dir_weight = dir_offset, dir_weight
        self.coder = ResidualCoder()
        self.cls_conv = nn.Conv2d(in_channels, self.A * num_classes, 1)
        self.box_conv = nn.Conv2d(in_channels, self.A * 7, 1)
        if use_dir:
            self.dir_conv = nn.Conv2d(in_channels, self.A * 2, 1)
        self.focal = SigmoidFocalClassificationLoss()
        self.smooth = WeightedSmoothL1Loss(code_weights=[1.0] * 7)
        nn.init.constant_(self.cls_conv.bias, -np.log((1 - 0.01) / 0.01))   # focal prior

    def generate_anchors(self, H, W, device):
        xs = torch.linspace(self.pcr[0], self.pcr[3], W, device=device)
        ys = torch.linspace(self.pcr[1], self.pcr[4], H, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")                      # (H,W)
        out = []
        for s in self.sizes:
            for r in self.rots:
                z = torch.full_like(xx, self.bottom + s[2] / 2)
                out.append(torch.stack([xx, yy, z, torch.full_like(xx, s[0]),
                                        torch.full_like(xx, s[1]), torch.full_like(xx, s[2]),
                                        torch.full_like(xx, r)], dim=-1))    # (H,W,7)
        return torch.stack(out, dim=2).reshape(-1, 7)                       # (H*W*A, 7)

    def forward(self, feat):
        B, _, H, W = feat.shape
        cls = self.cls_conv(feat).permute(0, 2, 3, 1).reshape(B, H * W * self.A, self.num_classes)
        box = self.box_conv(feat).permute(0, 2, 3, 1).reshape(B, H * W * self.A, 7)
        out = {"cls": cls, "box": box, "anchors": self.generate_anchors(H, W, feat.device)}
        if self.use_dir:
            out["dir"] = self.dir_conv(feat).permute(0, 2, 3, 1).reshape(B, H * W * self.A, 2)
        return out

    def _dir_target(self, headings):
        """2-bin direction target from GT heading (OpenPCDet convention)."""
        offset_rot = limit_period(headings - self.dir_offset, 0.0, 2 * np.pi)
        return torch.floor(offset_rot / np.pi).long().clamp(0, 1)

    def assign(self, anchors, gt):
        """anchors (Na,7); gt (Ng,8)[box7, label]. -> cls_t, reg_t, cls_w, reg_w, dir_t, pos."""
        Na = anchors.shape[0]
        cls_t = anchors.new_zeros(Na, self.num_classes)
        reg_t = anchors.new_zeros(Na, 7)
        reg_w = anchors.new_zeros(Na)
        dir_t = anchors.new_zeros(Na, dtype=torch.long)
        if gt.shape[0] == 0:
            return cls_t, reg_t, torch.ones(Na, device=anchors.device), reg_w, dir_t
        iou = (rotated_iou_assign(anchors, gt[:, :7]) if self.rotated_assign
               else boxes_bev_iou_aligned(anchors, gt[:, :7]))             # (Na,Ng)
        maxiou, arg = iou.max(dim=1)
        pos, neg = maxiou >= self.pos_thresh, maxiou < self.neg_thresh
        matched = gt[:, :7][arg]
        cls_t[pos, gt[:, 7].long()[arg][pos]] = 1.0
        reg_t[pos] = self.coder.encode(anchors[pos], matched[pos])
        reg_w[pos] = 1.0
        dir_t[pos] = self._dir_target(matched[pos][:, 6])
        return cls_t, reg_t, (pos | neg).float(), reg_w, dir_t

    def get_loss(self, pred, gt_list):
        anchors = pred["anchors"]
        cls_t, reg_t, cls_w, reg_w, dir_t = [], [], [], [], []
        for gt in gt_list:
            ct, rt, cw, rw, dt = self.assign(anchors, gt.to(anchors.device))
            cls_t.append(ct); reg_t.append(rt); cls_w.append(cw); reg_w.append(rw); dir_t.append(dt)
        cls_t, reg_t = torch.stack(cls_t), torch.stack(reg_t)
        cls_w, reg_w, dir_t = torch.stack(cls_w), torch.stack(reg_w), torch.stack(dir_t)
        num_pos = reg_w.sum().clamp(min=1.0)
        cls_loss = self.focal(pred["cls"], cls_t, cls_w).sum() / num_pos
        reg_loss = self.smooth(pred["box"], reg_t, reg_w).sum() / num_pos
        loss = cls_loss + 2.0 * reg_loss
        stats = {"cls": cls_loss.item(), "reg": reg_loss.item(), "num_pos": int(num_pos.item())}
        if self.use_dir:
            dir_loss = (torch.nn.functional.cross_entropy(pred["dir"].flatten(0, 1), dir_t.flatten(),
                                                          reduction="none").view_as(reg_w) * reg_w).sum() / num_pos
            loss = loss + self.dir_weight * dir_loss
            stats["dir"] = dir_loss.item()
        return loss, stats

    @torch.no_grad()
    def predict(self, pred, score_thresh=0.1, nms_thresh=0.01):
        anchors = pred["anchors"]
        outs = []
        for b in range(pred["cls"].shape[0]):
            scores, labels = pred["cls"][b].sigmoid().max(dim=1)
            boxes = self.coder.decode(pred["box"][b], anchors)
            if self.use_dir:                                               # correct 180° flip
                dir_lbl = pred["dir"][b].argmax(dim=1)
                period = np.pi
                r = limit_period(boxes[:, 6] - self.dir_offset, 0.0, period)
                boxes[:, 6] = r + self.dir_offset + period * dir_lbl.to(boxes.dtype)
            keep = scores > score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if boxes.shape[0]:
                sel = nms_aligned(boxes, scores, nms_thresh)
                boxes, scores, labels = boxes[sel], scores[sel], labels[sel]
            outs.append({"boxes": boxes, "scores": scores, "labels": labels})
        return outs


class PointPillars(nn.Module):
    """Pure-PyTorch PointPillars (LiDAR-only). forward takes a list of per-sample point arrays."""

    def __init__(self, num_point_features=4, num_classes=1,
                 pc_range=(0, -39.68, -3, 69.12, 39.68, 1), voxel_size=(0.16, 0.16, 4),
                 max_points=32, max_pillars=30000, vfe_channels=64,
                 rotated_assign=False, use_dir=False, backbone="base"):
        super().__init__()
        self.pc_range, self.voxel_size = list(pc_range), list(voxel_size)
        self.max_points, self.max_pillars = max_points, max_pillars
        self.nx = int(round((pc_range[3] - pc_range[0]) / voxel_size[0]))
        self.ny = int(round((pc_range[4] - pc_range[1]) / voxel_size[1]))
        self.vfe = PillarVFE(num_point_features, voxel_size, pc_range, out_channels=vfe_channels)
        self.backbone = make_bev_backbone(backbone, vfe_channels)
        self.head = AnchorHead(self.backbone.num_bev_features, pc_range, num_classes=num_classes,
                               rotated_assign=rotated_assign, use_dir=use_dir)

    def _bev(self, points):
        dev = next(self.parameters()).device
        voxels, num, coords = pillarize(points, self.pc_range, self.voxel_size,
                                        self.max_points, self.max_pillars)
        voxels = torch.from_numpy(voxels).to(dev)
        num = torch.from_numpy(num).to(dev)
        coords = torch.from_numpy(coords).to(dev)
        if voxels.shape[0] == 0:
            return points_new_zeros(dev, self.vfe.out_channels, self.ny, self.nx)
        pf = self.vfe(voxels, num, coords)
        return scatter_bev(pf, coords, self.nx, self.ny)

    def forward(self, points_list):
        bev = torch.stack([self._bev(p) for p in points_list])              # (B,C,ny,nx)
        return self.head(self.backbone(bev))


def points_new_zeros(dev, C, ny, nx):
    return torch.zeros(C, ny, nx, device=dev)


# =========================================================================== #
# forward + loss smoke test:  python -m ...detection.pointpillars
# =========================================================================== #
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointPillars(num_point_features=4, num_classes=1).to(dev)
    p = sum(x.numel() for x in model.parameters()) / 1e6
    # two fake samples: random points in-range (x∈[0,69], y∈[-39,39], z∈[-3,1], intensity)
    def fake_pts(n=8000):
        xyz = np.random.rand(n, 4).astype(np.float32)
        xyz[:, 0] *= 69.0; xyz[:, 1] = xyz[:, 1] * 78 - 39; xyz[:, 2] = xyz[:, 2] * 4 - 3
        return xyz
    pts = [fake_pts(), fake_pts()]
    pred = model(pts)
    print(f"params={p:.1f}M  cls={tuple(pred['cls'].shape)}  box={tuple(pred['box'].shape)}  "
          f"anchors={tuple(pred['anchors'].shape)}")
    # fake GT boxes per sample: (Ng,8) [x,y,z,dx,dy,dz,heading,label]
    gt = [torch.tensor([[30., 0., -1., 3.9, 1.6, 1.56, 0.3, 0.],
                        [15., -8., -1., 3.9, 1.6, 1.56, 1.2, 0.]]),
          torch.tensor([[40., 5., -1., 3.9, 1.6, 1.56, 0.0, 0.]])]
    loss, stats = model.head.get_loss(pred, gt)
    print(f"loss={loss.item():.3f}  {stats}")
    loss.backward()
    gnorm = sum(x.grad.norm().item() for x in model.parameters() if x.grad is not None)
    dets = model.head.predict(pred, score_thresh=0.0)
    print(f"grad_norm={gnorm:.2f}  det0_boxes={tuple(dets[0]['boxes'].shape)}")
    assert pred["cls"].shape[0] == 2 and pred["box"].shape[-1] == 7
    assert loss.item() > 0 and gnorm > 0
    print("OK: forward + loss + backward + predict")
