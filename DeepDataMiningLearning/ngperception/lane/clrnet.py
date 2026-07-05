"""
ngperception.lane.clrnet
========================

A **pure-PyTorch CLRNet** (Cross-Layer Refinement Network for lane detection, CVPR'22),
with the **CLRerNet** (WACV'24) **LaneIoU** option — no mmdet, no mmcv, no compiled ops.
This is the base detector for the lane research directions in ``DESIGN.md``; getting a
real F1 here validates the line-anchor representation before we extend it with
uncertainty / temporal / 3-D heads.

The idea in one screen
----------------------
A lane is a thin, long curve — bad for a plain segmentation net. CLRNet represents a lane
as a **line anchor** ("prior"): a start point on the image border + an angle, sampled at a
**fixed grid of image rows**. So a lane prediction is just *"for each of N rows, what is the
lane's x"* (plus start-row, length, and a fg/bg score). Detection becomes a set of line
proposals refined against features — like anchor-based object detection, but for lines.

Pipeline (this file):
  1. **Backbone + FPN**  — torchvision ResNet → 3 feature levels (strides 8/16/32),
     each projected to a common ``fdim`` (the FPN laterals).
  2. **Line-anchor priors** — a fixed bank of straight-line anchors (border start × angles);
     each gives a base x at every sample row.
  3. **ROIGather (lite)** — for each prior, sample features *along its line* at the N rows,
     on every FPN level (``grid_sample``, the same pure-torch deformable trick as the BEV
     transformer), pool → one feature vector per prior. (The full CLRNet does 3 cascaded
     refinement stages + line-attention; we do a single stage — documented simplification.)
  4. **Heads** — per prior: cls (fg/bg), start-row, length, and a per-row x refinement.
  5. **Assignment** — SimOTA-style dynamic-k matching (cls cost + LineIoU cost).
  6. **Loss** — focal cls + **LineIoU / LaneIoU** regression + smooth-L1 on start/length.
  7. **Decode** — softmax score → threshold → LineIoU-NMS → lane point sets.

See ``lane_iou.py`` for LineIoU/LaneIoU. Geometry convention: ``sample_ys`` are pixel row
positions, **decreasing** (index 0 = image bottom), matching CLRNet.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .lane_iou import line_iou, line_iou_loss, line_iou_cost


# ----------------------------------------------------------------------------- priors
def generate_priors(img_w, img_h, n_bottom=16, n_side=6, n_angles=9):
    """A fixed bank of straight-line anchors as (start_x, start_y, angle_deg).

    Starts tile the **bottom edge** (most lanes exit the bottom) plus the **left/right
    edges** (lanes entering from the side); each start fans out over ``n_angles`` angles.
    Returns ``(P, 3)`` float tensor.
    """
    angles = torch.linspace(30.0, 150.0, n_angles)          # deg from +x axis; ~vertical mid
    priors = []
    for sx in torch.linspace(0.0, img_w, n_bottom):
        for a in angles:
            priors.append((float(sx), float(img_h), float(a)))
    for sy in torch.linspace(0.0, img_h * 0.6, n_side):     # side entries, upper 60%
        for a in angles:
            priors.append((0.0, float(sy), float(a)))       # left edge
            priors.append((float(img_w), float(sy), float(a)))  # right edge
    return torch.tensor(priors, dtype=torch.float32)


def prior_base_xs(priors, sample_ys):
    """Straight-line x at each sample row for every prior. ``(P,3),(N,) -> (P,N)`` pixels.

    Line through (start_x, start_y) at angle θ: for a row y, x = start_x + (start_y − y)/tanθ.
    θ=90° → vertical (x=start_x). tanθ handles the sign for lanes leaning left/right.
    """
    sx, sy, deg = priors[:, 0:1], priors[:, 1:2], priors[:, 2:3]
    theta = torch.deg2rad(deg)
    tan = torch.tan(theta).clamp(min=-1e4, max=1e4)
    # guard near-horizontal (tan~0) → treat as vertical to avoid blow-up
    inv = torch.where(tan.abs() < 1e-3, torch.zeros_like(tan), 1.0 / tan)
    return sx + (sy - sample_ys[None, :]) * inv            # (P,N)


# ----------------------------------------------------------------------------- backbone
class ResNetFPN(nn.Module):
    """torchvision ResNet trunk → 3 FPN laterals at a common channel width ``fdim``."""

    def __init__(self, backbone="resnet18", fdim=64, pretrained=True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, backbone)(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        chans = {"resnet18": (128, 256, 512), "resnet34": (128, 256, 512),
                 "resnet50": (512, 1024, 2048)}[backbone]
        self.lat = nn.ModuleList([nn.Conv2d(c, fdim, 1) for c in chans])
        self.fdim = fdim

    def forward(self, x):
        x = self.layer1(self.stem(x))
        c3 = self.layer2(x)     # stride 8
        c4 = self.layer3(c3)    # stride 16
        c5 = self.layer4(c4)    # stride 32
        return [l(c) for l, c in zip(self.lat, (c3, c4, c5))]


# ----------------------------------------------------------------------------- model
class CLRNet(nn.Module):
    N_EXTRA = 2  # reg head extras before the N x-offsets: [start_row, length]

    def __init__(self, img_w=800, img_h=320, n_offsets=72, backbone="resnet18",
                 fdim=64, prior_dim=64, pretrained=True,
                 n_bottom=16, n_side=6, n_angles=9,
                 iou_r=7.5, lane_iou_tilt=False):
        super().__init__()
        self.img_w, self.img_h, self.N = img_w, img_h, n_offsets
        self.iou_r, self.tilt = iou_r, lane_iou_tilt
        # sample rows: pixel y decreasing from bottom (index 0) to top
        sample_ys = torch.linspace(img_h, 0.0, n_offsets)
        priors = generate_priors(img_w, img_h, n_bottom, n_side, n_angles)
        self.register_buffer("sample_ys", sample_ys)
        self.register_buffer("priors", priors)                       # (P,3)
        self.register_buffer("prior_xs", prior_base_xs(priors, sample_ys))  # (P,N)
        self.P = priors.shape[0]

        self.backbone = ResNetFPN(backbone, fdim, pretrained)
        self.prior_embed = nn.Parameter(torch.randn(self.P, prior_dim) * 0.02)
        roi_in = fdim * 3 + prior_dim
        self.shared = nn.Sequential(
            nn.Linear(roi_in, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.cls_head = nn.Linear(256, 2)
        self.reg_head = nn.Linear(256, self.N_EXTRA + n_offsets)
        nn.init.constant_(self.cls_head.bias, 0.0)
        nn.init.normal_(self.reg_head.weight, std=1e-3)
        nn.init.constant_(self.reg_head.bias, 0.0)

    # ---- feature sampling along each prior line (ROIGather-lite) ---------------
    def _roi_feat(self, feats, xs):
        """Sample the 3 FPN levels along prior lines. feats:list[(B,C,H,W)], xs:(P,N).

        Builds a grid from (xs, sample_ys) normalised to [-1,1] and pools features over
        the N points per prior → (B, P, 3*C).
        """
        B = feats[0].shape[0]
        gx = xs / self.img_w * 2 - 1                      # (P,N)
        gy = self.sample_ys / self.img_h * 2 - 1          # (N,)
        grid = torch.stack([gx, gy[None, :].expand_as(gx)], dim=-1)   # (P,N,2)
        grid = grid[None].expand(B, -1, -1, -1)           # (B,P,N,2)
        pooled = []
        for f in feats:
            s = F.grid_sample(f, grid, mode="bilinear", padding_mode="zeros",
                              align_corners=False)         # (B,C,P,N)
            pooled.append(s.mean(dim=-1))                  # (B,C,P)
        return torch.cat(pooled, dim=1).permute(0, 2, 1)   # (B,P,3C)

    def forward(self, images):
        """images: (B,3,H,W) → dict with cls_logits (B,P,2), xs (B,P,N) pixels,
        start (B,P) & length (B,P) normalised."""
        feats = self.backbone(images)
        roi = self._roi_feat(feats, self.prior_xs)                       # (B,P,3C)
        emb = self.prior_embed[None].expand(roi.shape[0], -1, -1)        # (B,P,prior_dim)
        h = self.shared(torch.cat([roi, emb], dim=-1))                   # (B,P,256)
        cls_logits = self.cls_head(h)                                    # (B,P,2)
        reg = self.reg_head(h)                                           # (B,P,2+N)
        start = reg[..., 0]
        length = reg[..., 1]
        # x-refinement is regressed in *normalised* units (× img_w) so the head's outputs
        # stay O(1) — otherwise reaching a lane ~200 px from a prior needs ~img_w/lr steps.
        xs = self.prior_xs[None] + reg[..., self.N_EXTRA:] * self.img_w  # refine base line
        return {"cls_logits": cls_logits, "xs": xs, "start": start, "length": length}

    # ---- GT encoding ----------------------------------------------------------
    def encode_lane(self, pts):
        """A GT lane (list/array of (x,y) in resized-image pixels) → per-row target.

        Interpolates the lane's x at each sample row inside its y-span.
        Returns xs (N,) pixels (−1 where undefined), valid (N,) bool, start_norm, length_norm.
        """
        pts = torch.as_tensor(pts, dtype=torch.float32).cpu()
        sample_ys = self.sample_ys.cpu()                 # encode on CPU, get_loss moves to device
        ys, xo = pts[:, 1], pts[:, 0]
        order = torch.argsort(ys)
        ys, xo = ys[order], xo[order]
        xs = torch.full((self.N,), -1.0)
        valid = torch.zeros(self.N, dtype=torch.bool)
        ymin, ymax = ys.min(), ys.max()
        for i, sy in enumerate(sample_ys):
            if sy < ymin - 1 or sy > ymax + 1:
                continue
            # linear interp x at row sy (ys ascending)
            j = torch.searchsorted(ys, sy).clamp(1, len(ys) - 1)
            y0, y1, x0, x1 = ys[j - 1], ys[j], xo[j - 1], xo[j]
            t = (sy - y0) / (y1 - y0 + 1e-6)
            x = x0 + t * (x1 - x0)
            if 0 <= x <= self.img_w:
                xs[i] = x
                valid[i] = True
        if valid.sum() == 0:
            return xs, valid, 0.0, 0.0
        idx = torch.nonzero(valid).squeeze(1)
        start_norm = idx.min().item() / (self.N - 1)
        length_norm = (idx.max() - idx.min() + 1).item() / self.N
        return xs, valid, start_norm, length_norm

    # ---- loss -----------------------------------------------------------------
    def get_loss(self, out, targets, cls_w=1.0, iou_w=2.0, reg_w=0.5):
        """targets: list (len B) of lane lists; each lane = (K,2) (x,y) resized pixels."""
        B = out["cls_logits"].shape[0]
        dev = out["cls_logits"].device
        cls_loss = out["cls_logits"].new_zeros(())
        iou_loss = out["cls_logits"].new_zeros(())
        ext_loss = out["cls_logits"].new_zeros(())
        n_pos_total = 0
        for b in range(B):
            lanes = targets[b]
            cls_target = torch.zeros(self.P, dtype=torch.long, device=dev)
            pred_xs = out["xs"][b]                                  # (P,N)
            if len(lanes) > 0:
                tgt_xs, tgt_valid, tgt_start, tgt_len = [], [], [], []
                for ln in lanes:
                    xs_i, v_i, s_i, l_i = self.encode_lane(ln)
                    tgt_xs.append(xs_i); tgt_valid.append(v_i)
                    tgt_start.append(s_i); tgt_len.append(l_i)
                tgt_xs = torch.stack(tgt_xs).to(dev)                # (G,N)
                tgt_valid = torch.stack(tgt_valid).to(dev)
                tgt_start = torch.tensor(tgt_start, device=dev)
                tgt_len = torch.tensor(tgt_len, device=dev)
                # assignment: dynamic-k SimOTA on (cls + iou) cost
                assign = self._assign(out["cls_logits"][b], pred_xs, tgt_xs)
                pos = assign >= 0
                cls_target[pos] = 1
                n_pos = int(pos.sum())
                n_pos_total += n_pos
                if n_pos > 0:
                    g = assign[pos]                                 # (n_pos,)
                    px = pred_xs[pos]                               # (n_pos,N)
                    tx = tgt_xs[g]                                  # (n_pos,N)
                    tv = tgt_valid[g]
                    iou_loss = iou_loss + line_iou_loss(
                        px, tx, self.sample_ys, r=self.iou_r, valid=tv, tilt=self.tilt)
                    ext_loss = ext_loss + F.smooth_l1_loss(
                        out["start"][b][pos], tgt_start[g]) + F.smooth_l1_loss(
                        out["length"][b][pos], tgt_len[g])
            cls_loss = cls_loss + self._focal(out["cls_logits"][b], cls_target)
        n = max(1, n_pos_total)
        total = cls_w * cls_loss / B + iou_w * iou_loss / max(1, B) + reg_w * ext_loss / max(1, B)
        return total, {"cls": float((cls_loss / B).detach()), "iou": float((iou_loss / max(1, B)).detach()),
                       "reg": float((ext_loss / max(1, B)).detach()), "n_pos": n_pos_total}

    def _focal(self, logits, target, alpha=0.25, gamma=2.0):
        """2-way focal loss over all priors (fg vs bg), summed then normalised by #priors."""
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        pt = p.gather(1, target[:, None]).squeeze(1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        a = torch.where(target == 1, alpha, 1 - alpha)
        loss = -a * (1 - pt).clamp(min=1e-6) ** gamma * logpt
        return loss.sum() / max(1, int((target == 1).sum()))

    @torch.no_grad()
    def _assign(self, cls_logits, pred_xs, tgt_xs, topq=4, cost_iou_w=3.0):
        """SimOTA-lite: per GT pick dynamic-k lowest-cost priors; resolve conflicts by cost.

        Returns assign (P,) long: GT index per prior, or −1 if negative.
        """
        P = pred_xs.shape[0]
        G = tgt_xs.shape[0]
        iou = line_iou_cost(pred_xs, tgt_xs, self.sample_ys, r=self.iou_r, tilt=self.tilt)  # (P,G)
        prob_fg = F.softmax(cls_logits, dim=-1)[:, 1:2]                      # (P,1)
        cls_cost = -torch.log(prob_fg.clamp(min=1e-6)).expand(P, G)
        cost = cls_cost + cost_iou_w * (1 - iou)
        # dynamic k per GT from its top-q IoU mass
        topk_iou = iou.topk(min(topq, P), dim=0).values                      # (q,G)
        dyn_k = topk_iou.clamp(min=0).sum(0).round().long().clamp(min=1, max=topq)
        match = torch.zeros(P, G, dtype=torch.bool, device=pred_xs.device)
        for g in range(G):
            k = int(dyn_k[g])
            sel = cost[:, g].topk(k, largest=False).indices
            match[sel, g] = True
        assign = torch.full((P,), -1, dtype=torch.long, device=pred_xs.device)
        matched = match.any(dim=1)
        if matched.any():
            # each matched prior → its lowest-cost GT
            assign[matched] = cost[matched].argmin(dim=1)
        return assign

    # ---- decode ---------------------------------------------------------------
    @torch.no_grad()
    def decode(self, out, score_thresh=0.4, nms_iou=0.5, nms_r=None):
        """Per image → list of lanes; each lane = (M,2) (x,y) in **resized** pixels.

        softmax fg score → threshold → LineIoU-NMS → point set over valid rows.
        """
        nms_r = nms_r or self.iou_r
        B = out["cls_logits"].shape[0]
        results = []
        for b in range(B):
            score = F.softmax(out["cls_logits"][b], dim=-1)[:, 1]       # (P,)
            keep = score > score_thresh
            if keep.sum() == 0:
                results.append([]); continue
            idx = torch.nonzero(keep).squeeze(1)
            xs = out["xs"][b][idx]                                       # (K,N)
            sc = score[idx]
            start = out["start"][b][idx].clamp(0, 1)
            length = out["length"][b][idx].clamp(0, 1)
            order = sc.argsort(descending=True)
            xs, sc, start, length = xs[order], sc[order], start[order], length[order]
            kept = self._nms(xs, nms_iou, nms_r)
            lanes = []
            for k in kept:
                s0 = int(round(start[k].item() * (self.N - 1)))
                nl = int(round(length[k].item() * self.N))
                rows = range(s0, min(self.N, s0 + max(nl, 2)))
                pts = [(float(xs[k, i]), float(self.sample_ys[i])) for i in rows
                       if 0 <= xs[k, i] <= self.img_w]
                if len(pts) >= 2:
                    lanes.append(torch.tensor(pts))
            results.append(lanes)
        return results

    def _nms(self, xs, nms_iou, r):
        """Greedy LineIoU NMS over already score-sorted lane x-rows. Returns kept indices."""
        K = xs.shape[0]
        suppressed = torch.zeros(K, dtype=torch.bool)
        kept = []
        for i in range(K):
            if suppressed[i]:
                continue
            kept.append(i)
            if i + 1 >= K:
                break
            rest = torch.arange(i + 1, K)
            live = rest[~suppressed[rest]]
            if len(live) == 0:
                continue
            iou = line_iou(xs[i:i + 1].expand(len(live), -1), xs[live],
                           self.sample_ys, r=r, tilt=self.tilt).cpu()
            suppressed[live[iou > nms_iou]] = True
        return kept
