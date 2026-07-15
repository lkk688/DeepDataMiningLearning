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
import os
import torch
from torch.utils.data import DataLoader

from .pointpillars import PointPillars
from .centerpoint import CenterPoint
from .nuscenes_dataset import NuScenesCarDataset, NUSC_10CLASS
from .kitti_dataset import collate
from .eval3d import eval_map, nusc_class_ap

# nuScenes config: 360° range, 0.2 m pillars. Per-class anchor sizes (OpenPCDet nuScenes).
NUSC_PCR = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
NUSC_VOXEL = (0.2, 0.2, 8.0)
NUSC_ANCHOR = ((4.63, 1.97, 1.74),)                     # car-only default
NUSC_BOTTOM = -1.8
# 10-class: car,truck,constr,bus,trailer,barrier,motorcycle,bicycle,pedestrian,traffic_cone
NUSC_10_SIZES = [(4.63, 1.97, 1.74), (6.93, 2.51, 2.84), (6.37, 2.85, 3.19), (10.5, 2.94, 3.47),
                 (12.29, 2.90, 3.87), (0.50, 2.53, 0.98), (2.11, 0.77, 1.47), (1.70, 0.60, 1.28),
                 (0.73, 0.67, 1.77), (0.41, 0.41, 1.07)]
NUSC_10_BOTTOMS = [-0.95, -0.6, -0.225, -0.085, 0.115, -1.33, -1.085, -1.18, -0.91, -1.28]


@torch.no_grad()
def evaluate(model, loader, dev, ncls=1, iou_thresh=0.5):
    model.eval()
    preds, gts = [], []
    for b in loader:
        pred = model([p.numpy() for p in b["points"]])
        for d in model.head.predict(pred, score_thresh=0.1, nms_thresh=0.1):
            preds.append({k: v.cpu() for k, v in d.items()})
        for g in b["gt"]:
            gts.append({"boxes": g[:, :7], "labels": g[:, 7].long()})
    r = eval_map(preds, gts, ncls, iou_thresh=iou_thresh)
    # nuScenes uses center-distance matching (0.5/1/2/4 m), far more lenient than IoU@0.5
    return {"IoU@.5": r["AP"][0], "carAP_cd": nusc_class_ap(preds, gts, 0)}   # class 0 = car


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
    ap.add_argument("--sweeps", type=int, default=1, help="LiDAR sweeps to aggregate (10 = standard)")
    ap.add_argument("--lidar-cache", default=None, help="dir to cache aggregated multi-sweep points")
    ap.add_argument("--multiclass", action="store_true", help="10-class (CBGS-style) vs car-only")
    ap.add_argument("--pos-thresh", type=float, default=0.5)
    ap.add_argument("--neg-thresh", type=float, default=0.35)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--out-dir", default=None, help="dir to save best/final checkpoints")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader workers (dataset is now fork-safe: each rebuilds its own devkit)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ncls = 10 if args.multiclass else 1

    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.root, verbose=False)   # built once, shared
    dkw = dict(pc_range=NUSC_PCR, nusc=nusc, sweeps=args.sweeps, lidar_cache=args.lidar_cache,
               class_map=(NUSC_10CLASS if args.multiclass else None))
    train_ds = NuScenesCarDataset(split="train", max_frames=args.max_frames, **dkw)
    val_ds = train_ds if args.overfit else \
        NuScenesCarDataset(split="val", max_frames=args.val_frames, **dkw)
    nw = args.num_workers
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                          num_workers=nw, persistent_workers=nw > 0)     # persistent: build devkit once/worker
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate,
                        num_workers=min(4, nw))                          # transient: freed after each eval

    cfg = dict(num_point_features=4, num_classes=ncls, pc_range=NUSC_PCR, voxel_size=NUSC_VOXEL,
               backbone=args.backbone, max_pillars=60000)      # 10-sweep = far more points
    if args.model == "centerpoint":
        model = CenterPoint(num_point_features=4, num_classes=ncls, pc_range=NUSC_PCR,
                            voxel_size=NUSC_VOXEL, backbone=args.backbone, max_pillars=60000).to(dev)
    else:
        sizes = NUSC_10_SIZES if args.multiclass else NUSC_ANCHOR
        bottoms = NUSC_10_BOTTOMS if args.multiclass else NUSC_BOTTOM
        model = PointPillars(anchor_sizes=sizes, anchor_bottom=bottoms,
                             pos_thresh=args.pos_thresh, neg_thresh=args.neg_thresh, **cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(train_ld)))
    print(f"[nusc-det] {len(train_ds)} train / {len(val_ds)} val | {args.model}/{args.backbone} | "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M | grid {model.nx}x{model.ny}", flush=True)

    # config needed to rebuild the model at eval time (kept alongside the weights)
    ckpt_cfg = dict(model=args.model, backbone=args.backbone, num_classes=ncls,
                    multiclass=args.multiclass, sweeps=args.sweeps,
                    pc_range=NUSC_PCR, voxel_size=NUSC_VOXEL, max_pillars=60000,
                    anchor_sizes=(NUSC_10_SIZES if args.multiclass else list(NUSC_ANCHOR)),
                    anchor_bottom=(NUSC_10_BOTTOMS if args.multiclass else NUSC_BOTTOM))

    def _save(tag):
        if not args.out_dir:
            return
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, f"nusc_det_{tag}.pth")
        torch.save({"state_dict": model.state_dict(), "cfg": ckpt_cfg}, path)
        print(f"[nusc-det] saved -> {path}", flush=True)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for b in train_ld:
            pred = model([p.numpy() for p in b["points"]])
            loss, stats = model.head.get_loss(pred, b["gt"])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); n += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            m = evaluate(model, val_ld, dev, ncls=ncls)
            print(f"[nusc-det] epoch {ep}: loss={tot/max(n,1):.3f}  "
                  + "  ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
            if m.get("carAP_cd", 0.0) > best:      # save best-by-carAP_cd
                best = m["carAP_cd"]; _save("best")
    _save("final")
    print("[nusc-det] done", flush=True)


if __name__ == "__main__":
    main()
