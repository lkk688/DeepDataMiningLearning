"""
ngperception.detection.train_kitti
===================================

Train + eval the pure-torch PointPillars on **KITTI Car** (single class). This is M1: get the
whole pipeline running end-to-end on real data — data → train → eval mAP — with no
spconv/mmcv. Eval uses the self-contained BEV-AP (`eval3d`); the official `kitti_eval/` is
HPC-only (numba-CUDA). Full-schedule training (80 ep) is an H100 job; locally we verify with
`--overfit` (a few frames should reach high AP, proving the pipeline learns) and short runs.

    # overfit sanity (pipeline learns): 8 frames, 60 ep -> AP should climb high
    python -m DeepDataMiningLearning.ngperception.detection.train_kitti \
        --root /mnt/e/Shared/Dataset/Kitti --overfit --max-frames 8 --epochs 60
    # short real train:
    python -m DeepDataMiningLearning.ngperception.detection.train_kitti \
        --root /mnt/e/Shared/Dataset/Kitti --max-frames 400 --val-frames 100 --epochs 20
"""
from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader

from .pointpillars import PointPillars
from .centerpoint import CenterPoint
from .bev_transformer import BEVTransformerDet
from .kitti_dataset import KittiCarDataset, collate
from .eval3d import eval_map


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
    return {f"mAP@{t}": eval_map(preds, gts, num_classes=1, iou_thresh=t)["mAP"] for t in iou_threshs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/e/Shared/Dataset/Kitti")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--val-frames", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--model", choices=["pointpillars", "centerpoint", "bev_transformer"],
                    default="pointpillars")
    ap.add_argument("--backbone", choices=["base", "res"], default="base",
                    help="BEV backbone: base (SSD-style) or res (PillarNeXt-style ResNet+FPN)")
    ap.add_argument("--overfit", action="store_true", help="train & eval on the SAME frames (sanity)")
    ap.add_argument("--rotated-assign", action="store_true", help="M2: rotated-IoU target assignment")
    ap.add_argument("--use-dir", action="store_true", help="M2: direction-classifier head")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    train_ds = KittiCarDataset(args.root, "train", max_frames=args.max_frames)
    val_ds = train_ds if args.overfit else KittiCarDataset(args.root, "val", max_frames=args.val_frames)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=4)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=4)

    if args.model == "centerpoint":
        model = CenterPoint(num_point_features=4, num_classes=1, backbone=args.backbone).to(dev)
    elif args.model == "bev_transformer":
        model = BEVTransformerDet(num_point_features=4, num_classes=1, backbone=args.backbone).to(dev)
    else:
        model = PointPillars(num_point_features=4, num_classes=1, backbone=args.backbone,
                             rotated_assign=args.rotated_assign, use_dir=args.use_dir).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(train_ld)))
    print(f"[pp-kitti] {len(train_ds)} train / {len(val_ds)} val | "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params | overfit={args.overfit}", flush=True)

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
            print(f"[pp-kitti] epoch {ep}: loss={tot/max(n,1):.3f}  "
                  + "  ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
    print("[pp-kitti] done", flush=True)


if __name__ == "__main__":
    main()
