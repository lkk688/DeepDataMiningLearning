"""
ngperception.detection.bev_transformer
=======================================

A **BEV transformer detector** — the attention/DETR paradigm, pure torch (attention needs no
spconv). It shares the pillar front-end + BEV backbone with the CNN detectors and replaces the
head with a **deformable-attention set-prediction decoder**:

- object **queries** each carry a **reference point**; instead of attending to *all* BEV tokens
  (vanilla DETR — slow), each query **samples a few points around its reference** via
  `F.grid_sample` (**deformable attention**, pure torch, no CUDA op);
- **iterative refinement**: each decoder layer updates the reference point from its predicted
  box centre, so later layers attend closer to the object (Deformable-DETR / DINO recipe);
- **auxiliary losses**: every layer is Hungarian-matched to GT and supervised.

This is the fix for the vanilla-DETR slow convergence the plain version hit (§10.1): deformable
attention + reference points + iterative refinement are the shared foundation of every modern
DETR-family detector (Deformable-DETR, DINO, RT-DETR, BEVFormer). Decoding is "take confident
queries" — no anchors, no NMS.

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


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(0, 1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class MLP(nn.Module):
    def __init__(self, dim, hidden, out, n=3):
        super().__init__()
        layers, d = [], dim
        for _ in range(n - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]; d = hidden
        layers += [nn.Linear(d, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeformableAttention(nn.Module):
    """Single-scale deformable attention (pure torch via grid_sample). Each query samples
    `n_points` locations per head around its reference point on the BEV value map."""

    def __init__(self, dim, n_heads=8, n_points=4):
        super().__init__()
        self.n_heads, self.n_points, self.head_dim = n_heads, n_points, dim // n_heads
        self.sampling_offsets = nn.Linear(dim, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(dim, n_heads * n_points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.sampling_offsets.weight); nn.init.zeros_(self.sampling_offsets.bias)
        nn.init.zeros_(self.attention_weights.weight); nn.init.zeros_(self.attention_weights.bias)

    def forward(self, query, ref, value_map):
        """query (B,Nq,dim); ref (B,Nq,2) in [0,1]; value_map (B,dim,H,W)."""
        B, Nq, dim = query.shape
        H, W = value_map.shape[-2:]
        value = self.value_proj(value_map.flatten(2).transpose(1, 2))          # (B,HW,dim)
        value = value.view(B, H * W, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
        value = value.reshape(B * self.n_heads, self.head_dim, H, W)
        off = self.sampling_offsets(query).view(B, Nq, self.n_heads, self.n_points, 2)
        w = self.attention_weights(query).view(B, Nq, self.n_heads, self.n_points).softmax(-1)
        norm = torch.tensor([W, H], device=query.device, dtype=query.dtype)
        loc = ref[:, :, None, None, :] + off / norm                           # (B,Nq,nh,np,2) in [0,1]
        grid = (2 * loc - 1).permute(0, 2, 1, 3, 4).reshape(B * self.n_heads, Nq, self.n_points, 2)
        sampled = F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        w = w.permute(0, 2, 1, 3).reshape(B * self.n_heads, 1, Nq, self.n_points)
        out = (sampled * w).sum(-1).view(B, self.n_heads * self.head_dim, Nq).transpose(1, 2)
        return self.output_proj(out)


class DeformableDecoderLayer(nn.Module):
    def __init__(self, dim, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.cross_attn = DeformableAttention(dim, n_heads, n_points)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(inplace=True), nn.Linear(dim * 4, dim))
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, q, ref, value_map, qpos):
        qk = q + qpos
        q = self.n1(q + self.self_attn(qk, qk, q)[0])
        q = self.n2(q + self.cross_attn(q + qpos, ref, value_map))
        return self.n3(q + self.ffn(q))


class BEVTransformerHead(nn.Module):
    def __init__(self, in_channels, num_classes, pc_range, num_queries=100, dim=128,
                 nheads=8, layers=3, n_points=4):
        super().__init__()
        self.pcr = list(pc_range); self.num_classes = num_classes
        self.value_proj = nn.Conv2d(in_channels, dim, 1)
        self.query = nn.Embedding(num_queries, dim)
        self.query_pos = nn.Embedding(num_queries, dim)
        self.ref_init = nn.Linear(dim, 2)                          # initial reference from query_pos
        self.layers = nn.ModuleList([DeformableDecoderLayer(dim, nheads, n_points) for _ in range(layers)])
        self.cls_head = nn.ModuleList([nn.Linear(dim, num_classes + 1) for _ in range(layers)])
        self.box_head = nn.ModuleList([MLP(dim, dim, 8) for _ in range(layers)])   # dref(2),z,logdxyz(3),sin,cos

    def forward(self, feat):
        value_map = self.value_proj(feat)
        B = feat.shape[0]
        q = self.query.weight[None].expand(B, -1, -1)
        qpos = self.query_pos.weight[None].expand(B, -1, -1)
        ref = self.ref_init(qpos).sigmoid()                        # (B,Nq,2) in [0,1]
        cls_out, box_out, ref_out = [], [], []
        for i, layer in enumerate(self.layers):
            q = layer(q, ref, value_map, qpos)
            box = self.box_head[i](q)
            new_ref = (inverse_sigmoid(ref) + box[..., :2]).sigmoid()          # iterative refinement
            cls_out.append(self.cls_head[i](q)); box_out.append(box); ref_out.append(new_ref)
            ref = new_ref.detach()
        return {"cls": cls_out, "box": box_out, "ref": ref_out}    # lists over decoder layers

    def decode_boxes(self, box, ref):
        """box (...,8), ref (...,2 in [0,1]) -> (...,7) world [x,y,z,dx,dy,dz,heading]."""
        x = self.pcr[0] + ref[..., 0] * (self.pcr[3] - self.pcr[0])
        y = self.pcr[1] + ref[..., 1] * (self.pcr[4] - self.pcr[1])
        dxyz = box[..., 3:6].clamp(-4, 4).exp()
        yaw = torch.atan2(box[..., 6], box[..., 7])
        return torch.stack([x, y, box[..., 2], dxyz[..., 0], dxyz[..., 1], dxyz[..., 2], yaw], -1)

    def _layer_loss(self, cls, box, ref, gt_list):
        B, Nq, _ = cls.shape
        dec = self.decode_boxes(box, ref)
        tgt = cls.new_full((B, Nq), self.num_classes, dtype=torch.long)
        box_loss = cls.new_zeros(())
        for b in range(B):
            gt = gt_list[b].to(cls.device)
            if len(gt) == 0:
                continue
            prob = cls[b].softmax(-1)
            cost = -prob[:, gt[:, 7].long()] + 0.1 * torch.cdist(dec[b][:, :2], gt[:, :2], p=1)
            r, c = linear_sum_assignment(cost.detach().cpu().numpy())
            r = torch.as_tensor(r, device=cls.device); c = torch.as_tensor(c, device=cls.device)
            tgt[b, r] = gt[c, 7].long()
            sincos_p = torch.stack([box[b, r, 6], box[b, r, 7]], -1)
            sincos_t = torch.stack([torch.sin(gt[c, 6]), torch.cos(gt[c, 6])], -1)
            box_loss = box_loss + 0.5 * F.l1_loss(dec[b][r][:, :6], gt[c][:, :6]) + F.l1_loss(sincos_p, sincos_t)
        cls_loss = F.cross_entropy(cls.reshape(-1, self.num_classes + 1), tgt.reshape(-1))
        return cls_loss + box_loss / max(B, 1), cls_loss, box_loss / max(B, 1)

    def get_loss(self, pred, gt_list):
        """Sum the set-prediction loss over ALL decoder layers (auxiliary losses)."""
        total = 0.0; last = (None, None)
        for i in range(len(pred["cls"])):
            l, cl, bl = self._layer_loss(pred["cls"][i], pred["box"][i], pred["ref"][i], gt_list)
            total = total + l; last = (cl, bl)
        n_match = sum(len(g) for g in gt_list)
        return total, {"cls": last[0].item(), "box": float(last[1].item()), "n_match": n_match}

    @torch.no_grad()
    def predict(self, pred, score_thresh=0.3, **kw):
        cls = pred["cls"][-1]                                       # final layer
        dec = self.decode_boxes(pred["box"][-1], pred["ref"][-1])
        outs = []
        for b in range(cls.shape[0]):
            prob = cls[b].softmax(-1)[:, :self.num_classes]
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
    pred = m([fp(), fp()])
    print(f"params={p:.1f}M  layers={len(pred['cls'])}  cls={tuple(pred['cls'][-1].shape)}")
    gt = [torch.tensor([[30., 0, -1, 3.9, 1.6, 1.56, 0.3, 0.], [15., -8, -1, 3.9, 1.6, 1.56, 1.2, 0.]]),
          torch.tensor([[40., 5, -1, 3.9, 1.6, 1.56, 0.0, 0.]])]
    loss, st = m.head.get_loss(pred, gt); print(f"loss={loss.item():.3f} {st}")
    loss.backward()
    gn = sum(x.grad.norm().item() for x in m.parameters() if x.grad is not None)
    dets = m.head.predict(pred, score_thresh=0.0)
    print(f"grad_norm={gn:.1f}  det0={tuple(dets[0]['boxes'].shape)}")
    assert loss.item() > 0 and gn > 0
    print("OK: deformable BEV transformer fwd + aux Hungarian loss + backward + decode")
