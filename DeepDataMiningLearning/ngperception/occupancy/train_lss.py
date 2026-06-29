"""
ngperception.occupancy.train_lss
================================

Train the depth-supervised LSS occupancy net. Two losses:
  * **occupancy CE** — per-voxel cross-entropy vs Occ3D GT, on camera-visible voxels.
  * **depth CE** — cross-entropy on the lift's predicted depth distribution vs the
    LiDAR-projected depth bins (the BEVDepth "depth supervision" that makes the lift
    geometrically faithful).  total = occ_ce + λ · depth_ce

Then evaluate with the same Occ3D mIoU we built (`evaluator.py`).

Example
-------
python -m DeepDataMiningLearning.ngperception.occupancy.train_lss \
    --gts /path/to/gts --nusc /mnt/e/Shared/Dataset/NuScenes/v1.0-trainval \
    --max-samples 400 --epochs 6 --batch-size 1
"""

from __future__ import annotations
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models.lss_occ import LSSOccupancy
from .datasets_train import NuScenesOccTrainDataset
from .evaluator import OccupancyEvaluator


def occ_loss(occ, semantics, mask_camera):
    """occ: (B,C,X,Y,Z); semantics: (B,X,Y,Z); mask_camera: (B,X,Y,Z) bool."""
    B, C = occ.shape[:2]
    logit = occ.permute(0, 2, 3, 4, 1).reshape(-1, C)
    tgt = semantics.reshape(-1)
    m = mask_camera.reshape(-1)
    return F.cross_entropy(logit[m], tgt[m])


def depth_loss(depth_pred, depth_gt):
    """depth_pred: (B,N,D,h,w) prob; depth_gt: (B,N,h,w) bin idx (-1 invalid)."""
    B, N, D, h, w = depth_pred.shape
    logp = torch.log(depth_pred.clamp_min(1e-6)).permute(0, 1, 3, 4, 2).reshape(-1, D)
    tgt = depth_gt.reshape(-1)
    m = tgt >= 0
    if m.sum() == 0:
        return depth_pred.new_zeros(())
    return F.nll_loss(logp[m], tgt[m])


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    ev = OccupancyEvaluator()
    with torch.no_grad():
        for i, b in enumerate(loader):
            occ, _ = model(b["imgs"].to(device), b["rots"].to(device),
                           b["trans"].to(device), b["intrins"].to(device))
            pred = occ.argmax(1).cpu().numpy()
            for j in range(pred.shape[0]):
                ev.add(pred[j], b["semantics"][j].numpy(), b["mask_camera"][j].numpy())
            if max_batches and i + 1 >= max_batches:
                break
    return ev.summarize(verbose=False)


def main():
    ap = argparse.ArgumentParser(description="Train depth-supervised LSS occupancy.")
    ap.add_argument("--gts", required=True)
    ap.add_argument("--nusc", default="/mnt/e/Shared/Dataset/NuScenes/v1.0-trainval")
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--val-samples", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--depth-weight", type=float, default=1.0)
    ap.add_argument("--backbone", choices=["resnet18", "dinov2"], default="resnet18")
    ap.add_argument("--amp", action="store_true", help="mixed precision (faster, less memory)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="DeepDataMiningLearning/ngperception/output/lss_occ")
    ap.add_argument("--smoke", action="store_true", help="2 samples, 1 epoch, no save")
    args = ap.parse_args()

    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    n = 2 if args.smoke else args.max_samples
    dev = args.device
    model = LSSOccupancy(backbone=args.backbone).to(dev)
    ihw, ds_factor = model.image_hw, model.downsample
    train_ds = NuScenesOccTrainDataset(args.gts, nusc, image_hw=ihw, downsample=ds_factor,
                                       max_samples=n)
    val_ds = NuScenesOccTrainDataset(args.gts, nusc, image_hw=ihw, downsample=ds_factor,
                                     max_samples=args.val_samples, stride=7)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=1, num_workers=2, collate_fn=collate)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    print(f"[lss_occ] backbone={args.backbone} {len(train_ds)} train / {len(val_ds)} val | "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M trainable")

    epochs = 1 if args.smoke else args.epochs
    for ep in range(epochs):
        model.train()
        for it, b in enumerate(train_ld):
            with torch.cuda.amp.autocast(enabled=args.amp):
                occ, depth = model(b["imgs"].to(dev), b["rots"].to(dev),
                                   b["trans"].to(dev), b["intrins"].to(dev))
                l_occ = occ_loss(occ, b["semantics"].to(dev), b["mask_camera"].to(dev))
                l_dep = depth_loss(depth.float(), b["depth_gt"].to(dev))
                loss = l_occ + args.depth_weight * l_dep
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if it % 20 == 0:
                print(f"  ep{ep} it{it}: loss={loss.item():.3f} "
                      f"(occ={l_occ.item():.3f} depth={l_dep.item():.3f})", flush=True)
            if args.smoke and it >= 1:
                break
        m = evaluate(model, val_ld, dev, max_batches=3 if args.smoke else None)
        print(f"[lss_occ] epoch {ep}: val mIoU={m['mIoU']:.3f} geo_IoU={m['geo_IoU']:.3f}", flush=True)

    if not args.smoke:
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.out_dir, "lss_occ.pth"))
        print(f"[lss_occ] saved -> {args.out_dir}/lss_occ.pth")


if __name__ == "__main__":
    main()
