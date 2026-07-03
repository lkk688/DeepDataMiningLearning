"""
ngperception.detection.bev_transformer
=======================================

A **BEV transformer detector** — the attention/DETR paradigm, the third model in this module
and the one that needs *no* sparse conv precisely because **attention is pure torch**. It shares
the pillar front-end + (PillarNeXt-style) BEV backbone with the CNN detectors, then replaces the
dense head with a **set-prediction transformer decoder**:

- flatten the BEV feature map to tokens (+ a learned positional embedding);
- a small stack of **object queries** cross-attends to those tokens (`nn.TransformerDecoder`);
- each query emits a class + a 3-D box (normalised centre, log-size, sin/cos yaw);
- training matches queries to GT **1-to-1 with the Hungarian algorithm** (bipartite matching),
  then applies class CE (unmatched → "no-object") + box L1 — no anchors, no NMS.

This is a DETR3D-lite: the cleanest demonstration that modern (transformer) detection fits the
pure-torch constraint. Decoding is just "take the confident queries".

    python -m DeepDataMiningLearning.ngperception.detection.bev_transformer   # fwd+loss+decode smoke
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointpillars import PillarVFE, scatter_bev, make_bev_backbone, pillarize

try:
    from scipy.optimize import linear_sum_assignment
except Exception:                                                  # greedy fallback
    def linear_sum_assignment(cost):
        cost = np.asarray(cost); rows, cols, used = [], [], set()
        for r in np.argsort(cost.min(1)):
            c = int(np.argmin([cost[r, j] if j not in used else 1e9 for j in range(cost.shape[1])]))
            if c not in used:
                rows.append(r); cols.append(c); used.add(c)
        return np.array(rows), np.array(cols)


class MLP(nn.Module):
    def __init__(self, dim, hidden, out, n=3):
        super().__init__()
        layers = []
        d = dim
        for _ in range(n - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]; d = hidden
        layers += [nn.Linear(d, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BEVTransformerHead(nn.Module):
    def __init__(self, in_channels, num_classes, pc_range, num_queries=100, dim=128,
                 nheads=8, layers=3):
        super().__init__()
        self.pcr = list(pc_range); self.num_classes = num_classes
        self.input_proj = nn.Conv2d(in_channels, dim, 1)
        self.pos_mlp = nn.Linear(2, dim)                           # positional from normalised cell (x,y)
        self.query = nn.Embedding(num_queries, dim)
        self.ref = nn.Embedding(num_queries, 2)                    # learned reference point (logit)
        nn.init.uniform_(self.ref.weight, -2.0, 2.0)              # spread refs across the BEV
        layer = nn.TransformerDecoderLayer(dim, nheads, dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.cls_head = nn.Linear(dim, num_classes + 1)            # +1 = no-object
        self.box_head = MLP(dim, dim, 8)                           # dxn,dyn,z,logdx,logdy,logdz,sin,cos

    def _pos(self, H, W, device):
        ys, xs = torch.meshgrid(torch.linspace(0, 1, H, device=device),
                                torch.linspace(0, 1, W, device=device), indexing="ij")
        return self.pos_mlp(torch.stack([xs, ys], -1).reshape(-1, 2))          # (HW, dim)

    def forward(self, feat):
        B, _, H, W = feat.shape
        src = self.input_proj(feat).flatten(2).permute(0, 2, 1)               # (B,HW,dim)
        src = src + self._pos(H, W, feat.device)[None]
        hs = self.decoder(self.query.weight[None].expand(B, -1, -1), src)     # (B,Nq,dim)
        return {"cls": self.cls_head(hs), "box": self.box_head(hs)}

    def decode_boxes(self, box):
        """(...,8) -> (...,7) world. Centre = learned **reference point** + predicted offset
        (DETR reference-point trick: regress a small offset, not an absolute sigmoid over the
        whole scene — the key to fast convergence)."""
        ref = self.ref.weight                                     # (Nq,2) broadcasts over batch
        x = self.pcr[0] + (ref[:, 0] + box[..., 0]).sigmoid() * (self.pcr[3] - self.pcr[0])
        y = self.pcr[1] + (ref[:, 1] + box[..., 1]).sigmoid() * (self.pcr[4] - self.pcr[1])
        dxyz = box[..., 3:6].clamp(-4, 4).exp()
        yaw = torch.atan2(box[..., 6], box[..., 7])
        return torch.stack([x, y, box[..., 2], dxyz[..., 0], dxyz[..., 1], dxyz[..., 2], yaw], -1)

    def get_loss(self, pred, gt_list):
        cls, box = pred["cls"], pred["box"]
        B, Nq, _ = cls.shape
        dec = self.decode_boxes(box)                                          # (B,Nq,7)
        tgt = cls.new_full((B, Nq), self.num_classes, dtype=torch.long)       # default: no-object
        box_loss = cls.new_zeros(()); n_match = 0
        for b in range(B):
            gt = gt_list[b].to(cls.device)
            if len(gt) == 0:
                continue
            prob = cls[b].softmax(-1)                                         # (Nq,C+1)
            cost = -prob[:, gt[:, 7].long()] + 0.1 * torch.cdist(dec[b][:, :2], gt[:, :2], p=1)
            r, c = linear_sum_assignment(cost.detach().cpu().numpy())
            r = torch.as_tensor(r, device=cls.device); c = torch.as_tensor(c, device=cls.device)
            tgt[b, r] = gt[c, 7].long()
            sincos_p = torch.stack([box[b, r, 6], box[b, r, 7]], -1)
            sincos_t = torch.stack([torch.sin(gt[c, 6]), torch.cos(gt[c, 6])], -1)
            box_loss = box_loss + 0.1 * F.l1_loss(dec[b][r][:, :6], gt[c][:, :6]) \
                + F.l1_loss(sincos_p, sincos_t)
            n_match += len(r)
        cls_loss = F.cross_entropy(cls.reshape(-1, self.num_classes + 1), tgt.reshape(-1))
        loss = cls_loss + box_loss / max(B, 1)
        return loss, {"cls": cls_loss.item(), "box": float(box_loss.item() / max(B, 1)), "n_match": n_match}

    @torch.no_grad()
    def predict(self, pred, score_thresh=0.3, **kw):
        cls, dec = pred["cls"], self.decode_boxes(pred["box"])
        outs = []
        for b in range(cls.shape[0]):
            prob = cls[b].softmax(-1)[:, :self.num_classes]                   # drop no-object
            score, label = prob.max(-1)
            keep = score > score_thresh
            outs.append({"boxes": dec[b][keep], "scores": score[keep], "labels": label[keep]})
        return outs


class BEVTransformerDet(nn.Module):
    def __init__(self, num_point_features=4, num_classes=1,
                 pc_range=(0, -39.68, -3, 69.12, 39.68, 1), voxel_size=(0.16, 0.16, 4),
                 max_points=32, max_pillars=30000, vfe_channels=64, backbone="res", num_queries=100):
        super().__init__()
        self.pc_range, self.voxel_size = list(pc_range), list(voxel_size)
        self.max_points, self.max_pillars = max_points, max_pillars
        self.nx = int(round((pc_range[3] - pc_range[0]) / voxel_size[0]))
        self.ny = int(round((pc_range[4] - pc_range[1]) / voxel_size[1]))
        self.vfe = PillarVFE(num_point_features, voxel_size, pc_range, out_channels=vfe_channels)
        self.backbone = make_bev_backbone(backbone, vfe_channels)
        self.head = BEVTransformerHead(self.backbone.num_bev_features, num_classes, pc_range,
                                       num_queries=num_queries)

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
    m = BEVTransformerDet(4, 1).to(dev)
    p = sum(x.numel() for x in m.parameters()) / 1e6
    def fp(n=8000):
        x = np.random.rand(n, 4).astype(np.float32); x[:, 0] *= 69; x[:, 1] = x[:, 1] * 78 - 39; x[:, 2] = x[:, 2] * 4 - 3
        return x
    pts = [fp(), fp()]
    pred = m(pts)
    print(f"params={p:.1f}M  cls={tuple(pred['cls'].shape)}  box={tuple(pred['box'].shape)}")
    gt = [torch.tensor([[30., 0, -1, 3.9, 1.6, 1.56, 0.3, 0.], [15., -8, -1, 3.9, 1.6, 1.56, 1.2, 0.]]),
          torch.tensor([[40., 5, -1, 3.9, 1.6, 1.56, 0.0, 0.]])]
    loss, st = m.head.get_loss(pred, gt); print(f"loss={loss.item():.3f} {st}")
    loss.backward()
    gn = sum(x.grad.norm().item() for x in m.parameters() if x.grad is not None)
    dets = m.head.predict(pred, score_thresh=0.0)
    print(f"grad_norm={gn:.1f}  det0={tuple(dets[0]['boxes'].shape)}")
    assert loss.item() > 0 and gn > 0
    print("OK: BEV transformer forward + Hungarian loss + backward + decode")
