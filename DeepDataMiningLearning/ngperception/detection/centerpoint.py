"""
ngperception.detection.centerpoint
===================================

A pure-PyTorch **CenterPoint** — the other classic 3D-detection paradigm: **anchor-free,
center-based**. Instead of matching anchors, it predicts a per-class **BEV centre heatmap**
(objects are Gaussian peaks) plus per-cell regression (sub-voxel offset, height, log-size,
sin/cos yaw). It shares our pillar front-end (`PillarVFE` + scatter + `BaseBEVBackbone` from
[`pointpillars.py`](pointpillars.py)) and only swaps the head.

Why it's a clean pure-torch fit: **decoding is a 3×3 max-pool "NMS"** on the heatmap — no
rotated-IoU NMS, no anchors, no spconv. Targets are rendered with the standard CenterNet
Gaussian; the loss is penalty-reduced focal on the heatmap + L1 on the regression at centres.

    python -m DeepDataMiningLearning.ngperception.detection.centerpoint   # forward+loss+decode smoke
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointpillars import PillarVFE, scatter_bev, make_bev_backbone, pillarize


# --------------------------------------------------------------------------- #
# CenterNet Gaussian target rendering (numpy; few GT per frame)
# --------------------------------------------------------------------------- #
def gaussian_radius(h, w, min_overlap=0.5):
    a1, b1, c1 = 1, (h + w), w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(max(b1 * b1 - 4 * a1 * c1, 0))) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + np.sqrt(max(b2 * b2 - 4 * a2 * c2, 0))) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + np.sqrt(max(b3 * b3 - 4 * a3 * c3, 0))) / 2
    return max(0, int(min(r1, r2, r3)))


def draw_gaussian(hm, cx, cy, radius):
    d = 2 * radius + 1
    sigma = d / 6.0
    m = np.arange(-radius, radius + 1)
    g = np.exp(-(m[:, None] ** 2 + m[None, :] ** 2) / (2 * sigma * sigma + 1e-9))
    H, W = hm.shape
    l, r = min(cx, radius), min(W - cx, radius + 1)
    t, b = min(cy, radius), min(H - cy, radius + 1)
    if r <= -l or b <= -t:
        return
    hm[cy - t:cy + b, cx - l:cx + r] = np.maximum(
        hm[cy - t:cy + b, cx - l:cx + r], g[radius - t:radius + b, radius - l:radius + r])


class CenterHead(nn.Module):
    def __init__(self, in_channels, num_classes, pc_range, voxel_size, nx, ny, hidden=64):
        super().__init__()
        self.num_classes = num_classes
        self.pcr = list(pc_range); self.vs = list(voxel_size); self.nx, self.ny = nx, ny
        def head(out, bias=0.0):
            m = nn.Sequential(nn.Conv2d(in_channels, hidden, 3, padding=1), nn.BatchNorm2d(hidden),
                              nn.ReLU(inplace=True), nn.Conv2d(hidden, out, 1))
            nn.init.constant_(m[-1].bias, bias)
            return m
        self.hm = head(num_classes, bias=-2.19)                 # centre heatmap (focal prior)
        self.reg = head(2)                                      # sub-voxel offset
        self.height = head(1)                                   # z
        self.dim = head(3)                                      # log dx,dy,dz
        self.rot = head(2)                                      # sin,cos yaw

    def forward(self, feat):
        return {"hm": self.hm(feat), "reg": self.reg(feat), "height": self.height(feat),
                "dim": self.dim(feat), "rot": self.rot(feat)}

    def _feat_scale(self, H, W):
        return (self.nx / W) * self.vs[0], (self.ny / H) * self.vs[1]        # metres per feat cell

    def assign(self, gts, H, W):
        """gts (Ng,8) -> dense target maps for one sample."""
        sx, sy = self._feat_scale(H, W)
        hm = np.zeros((self.num_classes, H, W), np.float32)
        reg = np.zeros((2, H, W), np.float32); hei = np.zeros((1, H, W), np.float32)
        dim = np.zeros((3, H, W), np.float32); rot = np.zeros((2, H, W), np.float32)
        mask = np.zeros((1, H, W), np.float32)
        for g in gts.cpu().numpy():
            x, y, z, dx, dy, dz, yaw, cls = g
            fx = (x - self.pcr[0]) / sx; fy = (y - self.pcr[1]) / sy
            cx, cy = int(fx), int(fy)
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            radius = gaussian_radius(dy / sy, dx / sx)
            draw_gaussian(hm[int(cls)], cx, cy, radius)
            reg[:, cy, cx] = [fx - cx, fy - cy]
            hei[0, cy, cx] = z
            dim[:, cy, cx] = np.log(np.clip([dx, dy, dz], 1e-3, None))
            rot[:, cy, cx] = [np.sin(yaw), np.cos(yaw)]
            mask[0, cy, cx] = 1.0
        t = lambda a: torch.from_numpy(a)
        return t(hm), t(reg), t(hei), t(dim), t(rot), t(mask)

    def get_loss(self, pred, gt_list):
        B, _, H, W = pred["hm"].shape
        dev = pred["hm"].device
        tg = [self.assign(g.to(dev), H, W) for g in gt_list]
        hm, reg, hei, dim, rot, mask = [torch.stack([t[i] for t in tg]).to(dev) for i in range(6)]
        p = torch.clamp(pred["hm"].sigmoid(), 1e-4, 1 - 1e-4)              # gaussian focal
        pos = hm.eq(1).float(); neg_w = torch.pow(1 - hm, 4)
        pos_loss = torch.log(p) * torch.pow(1 - p, 2) * pos
        neg_loss = torch.log(1 - p) * torch.pow(p, 2) * neg_w * (1 - pos)
        npos = pos.sum().clamp(min=1)
        hm_loss = -(pos_loss.sum() + neg_loss.sum()) / npos
        reg_pred = torch.cat([pred["reg"], pred["height"], pred["dim"], pred["rot"]], 1)
        reg_tgt = torch.cat([reg, hei, dim, rot], 1)
        l1 = (torch.abs(reg_pred - reg_tgt) * mask).sum() / mask.sum().clamp(min=1)
        loss = hm_loss + 0.25 * l1
        return loss, {"hm": hm_loss.item(), "reg": l1.item(), "npos": int(npos.item())}

    @torch.no_grad()
    def predict(self, pred, score_thresh=0.1, topk=100, nms_thresh=None):
        # nms_thresh is accepted for a common interface with the anchor head but unused:
        # CenterPoint's "NMS" is the 3x3 heatmap max-pool below.
        B, C, H, W = pred["hm"].shape
        sx, sy = self._feat_scale(H, W)
        hm = pred["hm"].sigmoid()
        peak = (F.max_pool2d(hm, 3, stride=1, padding=1) == hm).float() * hm  # 3x3 max-pool NMS
        outs = []
        for b in range(B):
            scores, idx = peak[b].reshape(-1).topk(min(topk, H * W * C))
            cls = idx // (H * W); rem = idx % (H * W); cy = rem // W; cx = rem % W
            keep = scores > score_thresh
            scores, cls, cy, cx = scores[keep], cls[keep], cy[keep], cx[keep]
            reg = pred["reg"][b, :, cy, cx].t(); hei = pred["height"][b, 0, cy, cx]
            dim = pred["dim"][b, :, cy, cx].t().exp(); rot = pred["rot"][b, :, cy, cx].t()
            x = (cx.float() + reg[:, 0]) * sx + self.pcr[0]
            y = (cy.float() + reg[:, 1]) * sy + self.pcr[1]
            yaw = torch.atan2(rot[:, 0], rot[:, 1])
            boxes = torch.stack([x, y, hei, dim[:, 0], dim[:, 1], dim[:, 2], yaw], 1)
            outs.append({"boxes": boxes, "scores": scores, "labels": cls})
        return outs


class CenterPoint(nn.Module):
    def __init__(self, num_point_features=4, num_classes=1,
                 pc_range=(0, -39.68, -3, 69.12, 39.68, 1), voxel_size=(0.16, 0.16, 4),
                 max_points=32, max_pillars=30000, vfe_channels=64, backbone="base"):
        super().__init__()
        self.pc_range, self.voxel_size = list(pc_range), list(voxel_size)
        self.max_points, self.max_pillars = max_points, max_pillars
        self.nx = int(round((pc_range[3] - pc_range[0]) / voxel_size[0]))
        self.ny = int(round((pc_range[4] - pc_range[1]) / voxel_size[1]))
        self.vfe = PillarVFE(num_point_features, voxel_size, pc_range, out_channels=vfe_channels)
        self.backbone = make_bev_backbone(backbone, vfe_channels)
        self.head = CenterHead(self.backbone.num_bev_features, num_classes,
                               pc_range, voxel_size, self.nx, self.ny)

    def _bev(self, points):
        dev = next(self.parameters()).device
        v, n, c = pillarize(points, self.pc_range, self.voxel_size, self.max_points, self.max_pillars)
        v, n, c = torch.from_numpy(v).to(dev), torch.from_numpy(n).to(dev), torch.from_numpy(c).to(dev)
        if v.shape[0] == 0:
            return torch.zeros(self.vfe.out_channels, self.ny, self.nx, device=dev)
        return scatter_bev(self.vfe(v, n, c), c, self.nx, self.ny)

    def forward(self, points_list):
        bev = torch.stack([self._bev(p) for p in points_list])
        return self.head(self.backbone(bev))


# =========================================================================== #
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = CenterPoint(4, 1).to(dev)
    p = sum(x.numel() for x in m.parameters()) / 1e6
    def fp(n=8000):
        x = np.random.rand(n, 4).astype(np.float32); x[:, 0] *= 69; x[:, 1] = x[:, 1] * 78 - 39; x[:, 2] = x[:, 2] * 4 - 3
        return x
    pts = [fp(), fp()]
    pred = m(pts)
    print(f"params={p:.1f}M  hm={tuple(pred['hm'].shape)}  dim={tuple(pred['dim'].shape)}")
    gt = [torch.tensor([[30., 0, -1, 3.9, 1.6, 1.56, 0.3, 0.], [15., -8, -1, 3.9, 1.6, 1.56, 1.2, 0.]]),
          torch.tensor([[40., 5, -1, 3.9, 1.6, 1.56, 0.0, 0.]])]
    loss, st = m.head.get_loss(pred, gt); print(f"loss={loss.item():.3f} {st}")
    loss.backward()
    gn = sum(x.grad.norm().item() for x in m.parameters() if x.grad is not None)
    dets = m.head.predict(pred, score_thresh=0.0)
    print(f"grad_norm={gn:.1f}  det0={tuple(dets[0]['boxes'].shape)}")
    assert loss.item() > 0 and gn > 0
    print("OK: CenterPoint forward + loss + backward + decode")
