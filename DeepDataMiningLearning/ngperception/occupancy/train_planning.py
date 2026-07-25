"""
occupancy.train_planning
=======================
Direction-4 head: **occupancy -> open-loop ego planning** (nuScenes, ST-P3/UniAD/OccNet protocol).
Freeze the occ backbone (the winner from the backbone benchmark), tap its semantic-occupancy output,
and train a lightweight planning head -> 6 future waypoints (3 s @ 0.5 s). Metrics: L2 @ 1/2/3 s and
collision rate vs the Occ3D-GT occupancy. Single-GPU, finetune-budget (no VLA training).

    python -m DeepDataMiningLearning.ngperception.occupancy.train_planning \
        --nusc <nusc>/v1.0-trainval --gts <nusc>/v1.0-trainval/gts \
        --pretrained output/occ_dinov2_large/lss_occ.pth --backbone dinov2_large \
        --max-samples 4000 --epochs 12 --out-dir output/backbone_bench/plan_dinov2_large
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .models.lss_occ import LSSOccupancy
from .datasets_train import NuScenesOccTrainDataset
from .geom import PC_RANGE, VOXEL_SIZE, GRID_SIZE

T = 6                      # 6 waypoints, 0.5 s apart = 3 s horizon
FREE = 17
# obstacle classes for collision (objects + walls), not drivable/flat
OBST = list(range(0, 11)) + [15, 16]      # others+objects + manmade + vegetation


def ego_future(nusc, token, n=T):
    """Future ego (x,y) in the CURRENT ego frame from the next n keyframes (0.5 s each)."""
    from pyquaternion import Quaternion
    s = nusc.get("sample", token)
    lsd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
    ep0 = nusc.get("ego_pose", lsd["ego_pose_token"])
    R0 = Quaternion(ep0["rotation"]).rotation_matrix; t0 = np.array(ep0["translation"])
    fut = []
    cur = s
    for _ in range(n):
        cur = nusc.get("sample", cur["next"]) if cur["next"] else cur
        lsd_i = nusc.get("sample_data", cur["data"]["LIDAR_TOP"])
        epi = nusc.get("ego_pose", lsd_i["ego_pose_token"])
        p = R0.T @ (np.array(epi["translation"]) - t0)      # global -> current ego
        fut.append(p[:2])
    return np.asarray(fut, np.float32)                      # (T,2)


class PlanDataset(Dataset):
    """Occ-net images/GT + ego-future waypoints + a 3-way command derived from the GT trajectory."""
    def __init__(self, base: NuScenesOccTrainDataset, nusc):
        self.base, self.nusc = base, nusc

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = self.base[i]
        tok = self.base.occ.items[i][1]
        fut = ego_future(self.nusc, tok)                    # (T,2)
        lat = fut[-1, 1]                                    # final lateral -> command
        cmd = np.array([lat > 2.0, lat < -2.0, abs(lat) <= 2.0], np.float32)  # left/right/straight
        d["ego_fut"] = torch.from_numpy(fut)
        d["cmd"] = torch.from_numpy(cmd)
        d["semantics_gt"] = d["semantics"] if "semantics" in d else torch.zeros(1)
        return d


def collate(b):
    return {k: torch.stack([x[k] for x in b]) for k in b[0]}


class PlanHead(nn.Module):
    """Semantic-occupancy BEV -> waypoints. (B,18,X,Y,Z) -> collapse Z -> CNN -> +cmd -> (B,T,2)."""
    def __init__(self, ncls=18, nz=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(ncls * nz, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1))
        self.mlp = nn.Sequential(nn.Linear(128 + 3, 256), nn.ReLU(True), nn.Linear(256, T * 2))

    def forward(self, occ, cmd):
        B, C, X, Y, Z = occ.shape
        bev = occ.permute(0, 1, 4, 2, 3).reshape(B, C * Z, X, Y)
        f = self.enc(bev).flatten(1)
        return self.mlp(torch.cat([f, cmd], 1)).reshape(B, T, 2)


class PlanHeadBC(nn.Module):
    """Patch-Policy-style planning: attend over DENSE spatial BEV tokens (not global-pool) with a
    transformer, plus a command token; a `query` token reads out the trajectory. Block-causal INTERFACE
    (`mask`) for temporal windows — single-frame = full intra-frame attention (its degenerate case).
    Follows LeCun et al. 'Patch Policy': dense frozen-ViT features + lightweight transformer head."""
    def __init__(self, ncls=18, nz=16, dim=192, layers=4, heads=6, grid=25):
        super().__init__()
        self.grid = grid
        self.embed = nn.Conv2d(ncls * nz, dim, kernel_size=200 // grid, stride=200 // grid)  # BEV -> gridxgrid tokens
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        self.cmd = nn.Linear(3, dim)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(dim, T * 2)
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, occ, cmd, mask=None):
        B, C, X, Y, Z = occ.shape
        bev = occ.permute(0, 1, 4, 2, 3).reshape(B, C * Z, X, Y)
        tok = self.embed(bev).flatten(2).transpose(1, 2)                      # (B, grid^2, dim)
        tok = tok + self.pos
        seq = torch.cat([self.query.expand(B, -1, -1), self.cmd(cmd)[:, None], tok], 1)  # query + cmd + patches
        out = self.tr(seq, mask=mask)[:, 0]                                    # read the query token
        return self.out(out).reshape(B, T, 2)


def collision_rate(waypoints, sem):
    """Fraction of predicted waypoints landing in an obstacle voxel of the Occ3D-GT (any z).
    NaN/inf-safe: non-finite predictions are skipped (rate over valid waypoints), so a diverged
    step never crashes eval."""
    lo = np.asarray(PC_RANGE[:2], np.float32); gx, gy = int(GRID_SIZE[0]), int(GRID_SIZE[1])
    obst_col = np.isin(sem, OBST).any(-1)                   # (X,Y) any obstacle along z
    hit = 0; n = 0
    for w in waypoints:
        if not np.isfinite(w).all():                        # skip NaN/inf waypoints
            continue
        n += 1
        ix = int((w[0] - lo[0]) / VOXEL_SIZE); iy = int((w[1] - lo[1]) / VOXEL_SIZE)
        if 0 <= ix < gx and 0 <= iy < gy and obst_col[ix, iy]:
            hit += 1
    return hit / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Occupancy -> open-loop planning (L2 + collision).")
    ap.add_argument("--nusc", required=True); ap.add_argument("--gts", required=True)
    ap.add_argument("--pretrained", required=True); ap.add_argument("--backbone", default="dinov2_large")
    ap.add_argument("--decoder-layers", type=int, default=4); ap.add_argument("--decoder-hidden", type=int, default=96)
    ap.add_argument("--refine-iters", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=4000); ap.add_argument("--val-samples", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--head", choices=["cnn", "transformer"], default="transformer",
                    help="transformer = Patch-Policy-style dense-token attention (default)")
    args = ap.parse_args()
    dev = args.device
    from nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    occ = LSSOccupancy(backbone=args.backbone, decoder_hidden=args.decoder_hidden,
                       decoder_layers=args.decoder_layers, refine_iters=args.refine_iters,
                       lidar_fusion=False).to(dev)
    occ.load_state_dict(torch.load(args.pretrained, map_location=dev), strict=False)
    occ.eval()
    for p in occ.parameters():
        p.requires_grad = False
    head = (PlanHeadBC() if args.head == "transformer" else PlanHead()).to(dev)
    print(f"[plan] head={args.head}", flush=True)
    tr = sorted(create_splits_scenes()["train"]); va = sorted(create_splits_scenes()["val"])
    ihw, dsf = occ.image_hw, occ.downsample
    tds = PlanDataset(NuScenesOccTrainDataset(args.gts, nusc, image_hw=ihw, downsample=dsf, scenes=tr,
                                              max_samples=args.max_samples), nusc)
    vds = PlanDataset(NuScenesOccTrainDataset(args.gts, nusc, image_hw=ihw, downsample=dsf, scenes=va,
                                              max_samples=args.val_samples), nusc)
    tl = DataLoader(tds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate, drop_last=True)
    vl = DataLoader(vds, batch_size=1, num_workers=2, collate_fn=collate)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[plan] {len(tds)} train / {len(vds)} val | frozen {args.backbone} occ -> plan head", flush=True)
    for ep in range(args.epochs):
        head.train()
        for it, b in enumerate(tl):
            with torch.no_grad():
                out = occ(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev), b["intrins"].to(dev))
                o = out[0] if isinstance(out, (tuple, list)) else out       # (B,18,X,Y,Z)
            pred = head(o.float(), b["cmd"].to(dev))
            # ADE with eps INSIDE sqrt: sqrt(sum_sq) has infinite gradient at zero distance
            # (pred==gt), which produces NaN grads that corrupt the head; +1e-6 removes the singularity.
            loss = ((pred - b["ego_fut"].to(dev)) ** 2).sum(-1).add(1e-6).sqrt().mean()
            if not torch.isfinite(loss):                                        # skip diverged step
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)      # stabilize transformer head
            if not torch.isfinite(gnorm):                                       # belt+suspenders: skip NaN/inf grads
                opt.zero_grad(); continue
            opt.step()
            if it % 50 == 0:
                print(f"  ep{ep} it{it}: L2={loss.item():.3f}", flush=True)
        # eval
        head.eval(); l2s = {1: [], 2: [], 3: []}; cols = []
        with torch.no_grad():
            for b in vl:
                out = occ(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev), b["intrins"].to(dev))
                o = out[0] if isinstance(out, (tuple, list)) else out
                pred = head(o.float(), b["cmd"].to(dev))[0].cpu().numpy()      # (T,2)
                gt = b["ego_fut"][0].numpy()
                d = np.linalg.norm(pred - gt, axis=1)                          # (T,)
                for s, k in ((1, 2), (2, 4), (3, 6)):
                    l2s[s].append(d[:k].mean())
                cols.append(collision_rate(pred, b["semantics"][0].numpy()))
        print(f"[plan] epoch {ep}: L2 1s={np.nanmean(l2s[1]):.3f} 2s={np.nanmean(l2s[2]):.3f} "
              f"3s={np.nanmean(l2s[3]):.3f} | collision={np.nanmean(cols):.4f}", flush=True)
        torch.save(head.state_dict(), os.path.join(args.out_dir, "plan_head.pth"))


if __name__ == "__main__":
    main()
