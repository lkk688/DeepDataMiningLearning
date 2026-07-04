"""
ngperception.occupancy.train_multitask
=======================================

**M3 — full-stack multi-task: occupancy + detection on one shared encoder.** Trains the
`LSSOccupancy` fused camera+LiDAR encoder with **two heads**: the 3-D occupancy decoder (Occ3D
CE + LiDAR depth CE) *and* the `VoxelDetHead` (nuScenes car boxes). Because both hang off the
same fused voxel volume, the **camera / LiDAR / fusion ablation is inherited** (`--lidar-fusion`,
`--lidar-only`) — the same knobs that gave occ 0.298→0.493 now also gate detection, for free.

    python -m DeepDataMiningLearning.ngperception.occupancy.train_multitask \
        --gts <occ3d_gts> --nusc <nuscenes> --lidar-fusion --max-samples 400 --epochs 8
"""
from __future__ import annotations
import argparse
import os
import torch
from torch.utils.data import DataLoader

from .models.lss_occ import LSSOccupancy
from .datasets_train import NuScenesOccTrainDataset
from .evaluator import OccupancyEvaluator
from .train_lss import occ_loss, depth_loss
from ..detection.eval3d import eval_map


def collate_mt(batch):
    out = {}
    for k in batch[0]:
        out[k] = [b[k] for b in batch] if k == "det_gt" else torch.stack([b[k] for b in batch])
    return out


@torch.no_grad()
def evaluate(model, loader, dev, max_batches=None):
    model.eval()
    ev = OccupancyEvaluator()
    preds, gts = [], []
    for i, b in enumerate(loader):
        lv = b["lidar_vox"].to(dev) if "lidar_vox" in b else None
        occ, _, aux = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev),
                            b["intrins"].to(dev), lidar_vox=lv)
        pred = occ.argmax(1).cpu().numpy()
        for j in range(pred.shape[0]):
            ev.add(pred[j], b["semantics"][j].numpy(), b["mask_camera"][j].numpy())
        if "det" in aux:
            for d in model.det_head.predict(aux["det"], score_thresh=0.2, nms_thresh=0.1):
                preds.append({k: v.cpu() for k, v in d.items()})
            for g in b["det_gt"]:
                gts.append({"boxes": g[:, :7], "labels": g[:, 7].long()})
        if max_batches and i + 1 >= max_batches:
            break
    occ_m = ev.summarize(verbose=False)
    car_ap = eval_map(preds, gts, 1, 0.5)["mAP"] if preds else 0.0
    return occ_m["mIoU"], occ_m["geo_IoU"], car_ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gts", required=True); ap.add_argument("--nusc", required=True)
    ap.add_argument("--max-samples", type=int, default=400); ap.add_argument("--val-samples", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--lidar-fusion", action="store_true"); ap.add_argument("--lidar-only", action="store_true")
    ap.add_argument("--det-weight", type=float, default=1.0); ap.add_argument("--depth-weight", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="DeepDataMiningLearning/ngperception/output/lss_multitask")
    args = ap.parse_args()
    dev = args.device
    if args.lidar_only:
        args.lidar_fusion = True

    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    model = LSSOccupancy(backbone=args.backbone, lidar_fusion=args.lidar_fusion,
                         lidar_only=args.lidar_only, det_classes=1,
                         det_anchor_sizes=((4.6, 1.97, 1.74),)).to(dev)
    ihw, ds = model.image_hw, model.downsample
    dkw = dict(image_hw=ihw, downsample=ds, lidar_fusion=True, det_boxes=True)   # fusion vox always needed by det grid
    train_ds = NuScenesOccTrainDataset(args.gts, nusc, max_samples=args.max_samples, **dkw)
    val_ds = NuScenesOccTrainDataset(args.gts, nusc, max_samples=args.val_samples, stride=7, **dkw)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mt,
                          num_workers=args.num_workers, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=1, collate_fn=collate_mt, num_workers=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    print(f"[mt] {len(train_ds)} train / {len(val_ds)} val | fusion={args.lidar_fusion} lidar_only={args.lidar_only} | "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)

    for ep in range(args.epochs):
        model.train()
        for b in train_ld:
            lv = b["lidar_vox"].to(dev)
            occ, depth, aux = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev),
                                    b["intrins"].to(dev), lidar_vox=lv)
            l_occ = occ_loss(occ, b["semantics"].to(dev), b["mask_camera"].to(dev))
            l_dep = depth_loss(depth.float(), b["depth_gt"].to(dev))
            l_det, _ = model.det_head.get_loss(aux["det"], b["det_gt"])
            loss = l_occ + args.depth_weight * l_dep + args.det_weight * l_det
            opt.zero_grad(); loss.backward(); opt.step()
        mIoU, geo, car = evaluate(model, val_ld, dev)
        print(f"[mt] epoch {ep}: occ_mIoU={mIoU:.3f} geo={geo:.3f} | car_AP@0.5={car:.3f}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "lss_multitask.pth"))
    print("[mt] done", flush=True)


if __name__ == "__main__":
    main()
