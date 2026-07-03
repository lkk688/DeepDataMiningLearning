"""
ngperception.detection.train_nuscenes
======================================

Train + eval PointPillars on a **small nuScenes** subset — a first "can it train at all on
nuScenes?" check. nuScenes differs from KITTI: **360°** range, coarser voxels, sparser
single-sweep LiDAR, slightly larger cars — so the model gets a nuScenes config
(range/voxel/anchors). One nuScenes devkit is built and shared by the train/val datasets
(`num_workers=0`, since the devkit object isn't fork-safe).

    python -m DeepDataMiningLearning.ngperception.detection.train_nuscenes \
        --root /mnt/e/Shared/Dataset/NuScenes/v1.0-trainval --max-frames 300 --val-frames 100 --epochs 40
"""
from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader

from .pointpillars import PointPillars
from .centerpoint import CenterPoint
from .nuscenes_dataset import NuScenesCarDataset
from .kitti_dataset import collate
from .eval3d import eval_map

# nuScenes-Car config (OpenPCDet-style): 360° range, 0.2 m pillars, larger anchor.
NUSC_PCR = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
NUSC_VOXEL = (0.2, 0.2, 8.0)
NUSC_ANCHOR = ((4.63, 1.97, 1.74),)
NUSC_BOTTOM = -1.8


@torch.no_grad()
def evaluate(model, loader, dev, iou_threshs=(0.5, 0.7)):
    model.eval()
    preds, gts = [], []
    for b in loader:
        pred = model([p.numpy() for p in b["points"]])
        for d in model.head.predict(pred, score_thresh=0.3, nms_thresh=0.1):
            preds.append({k: v.cpu() for k, v in d.items()})
        for g in b["gt"]:
            gts.append({"boxes": g[:, :7], "labels": g[:, 7].long()})
    return {f"mAP@{t}": eval_map(preds, gts, 1, iou_thresh=t)["mAP"] for t in iou_threshs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/e/Shared/Dataset/NuScenes/v1.0-trainval")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--val-frames", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--model", choices=["pointpillars", "centerpoint"], default="pointpillars")
    ap.add_argument("--backbone", choices=["base", "res"], default="res")
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.root, verbose=False)   # built once, shared
    train_ds = NuScenesCarDataset(split="train", pc_range=NUSC_PCR, max_frames=args.max_frames, nusc=nusc)
    val_ds = train_ds if args.overfit else \
        NuScenesCarDataset(split="val", pc_range=NUSC_PCR, max_frames=args.val_frames, nusc=nusc)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=0)

    cfg = dict(num_point_features=4, num_classes=1, pc_range=NUSC_PCR, voxel_size=NUSC_VOXEL, backbone=args.backbone)
    if args.model == "centerpoint":
        model = CenterPoint(**cfg).to(dev)
    else:
        model = PointPillars(anchor_sizes=NUSC_ANCHOR, anchor_bottom=NUSC_BOTTOM, **cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(train_ld)))
    print(f"[nusc-det] {len(train_ds)} train / {len(val_ds)} val | {args.model}/{args.backbone} | "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M | grid {model.nx}x{model.ny}", flush=True)

    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for b in train_ld:
            pred = model([p.numpy() for p in b["points"]])
            loss, stats = model.head.get_loss(pred, b["gt"])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); n += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            m = evaluate(model, val_ld, dev)
            print(f"[nusc-det] epoch {ep}: loss={tot/max(n,1):.3f}  "
                  + "  ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
    print("[nusc-det] done", flush=True)


if __name__ == "__main__":
    main()
