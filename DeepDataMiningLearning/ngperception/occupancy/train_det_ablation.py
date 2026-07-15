"""
ngperception.occupancy.train_det_ablation
==========================================

Backbone-reuse ablation for 3-D detection: attach a 10-class `VoxelDetHead` onto the
**trained occupancy fused-voxel encoder** (the camera+LiDAR model at occ mIoU ~0.60) and
train detection on nuScenes, reusing the learned scene features.

Arms (all vs the standalone PointPillars-res baseline, `detection/train_nuscenes.py`):
  --pretrained <occ.pth> --freeze-encoder   -> B: frozen reuse (do occ features transfer?)
  --pretrained <occ.pth>                    -> C: finetuned reuse
  (no --pretrained)                         -> from-scratch shared-encoder control
  --lidar-fusion / --lidar-only / neither   -> the free camera/LiDAR/fusion modality ablation

Boxes are in the **ego** frame (the occ grid); eval to official nuScenes metrics via
`eval_det_ablation_official.py` (ego->global). Saves {state_dict, cfg} for that evaluator.
"""
from __future__ import annotations
import argparse
import os
import torch
from torch.utils.data import DataLoader

from .models.lss_occ import LSSOccupancy
from .datasets_train import NuScenesOccTrainDataset
from .train_lss import occ_loss, depth_loss
from .train_multitask import collate_mt, evaluate
from ..detection.nuscenes_dataset import NUSC_10CLASS
from ..detection.train_nuscenes import NUSC_10_SIZES

# ego-frame anchor bottoms: M3 car det used -1.0 in the ego grid; reuse for all 10 classes.
NUSC_10_BOTTOMS_EGO = [-1.0] * 10


def main():
    ap = argparse.ArgumentParser(description="Occupancy-backbone 3D detection ablation.")
    ap.add_argument("--gts", required=True); ap.add_argument("--nusc", required=True)
    ap.add_argument("--pretrained", default=None, help="occ fusion checkpoint to reuse (None = from scratch)")
    ap.add_argument("--freeze-encoder", action="store_true", help="train only the det head (arm B)")
    ap.add_argument("--backbone", default="dinov2_base")
    ap.add_argument("--decoder-layers", type=int, default=4); ap.add_argument("--decoder-hidden", type=int, default=96)
    ap.add_argument("--refine-iters", type=int, default=2)
    ap.add_argument("--lidar-fusion", action="store_true"); ap.add_argument("--lidar-only", action="store_true")
    ap.add_argument("--det-head", choices=["anchor", "center"], default="anchor",
                    help="arm D: center = anchor-free CenterPoint head on the shared backbone")
    ap.add_argument("--occ-weight", type=float, default=0.0, help=">0 keeps an occ regularizer when finetuning")
    ap.add_argument("--max-samples", type=int, default=28130); ap.add_argument("--val-samples", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--cosine", action="store_true"); ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.device
    if args.lidar_only:
        args.lidar_fusion = True

    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    model = LSSOccupancy(backbone=args.backbone, decoder_hidden=args.decoder_hidden,
                         decoder_layers=args.decoder_layers, refine_iters=args.refine_iters,
                         lidar_fusion=args.lidar_fusion, lidar_only=args.lidar_only,
                         det_classes=10, det_anchor_sizes=NUSC_10_SIZES,
                         det_anchor_bottom=NUSC_10_BOTTOMS_EGO,
                         det_head_type=args.det_head).to(dev)

    if args.pretrained:                              # reuse the trained occ encoder (det head stays random)
        sd = torch.load(args.pretrained, map_location=dev)
        miss, unexp = model.load_state_dict(sd, strict=False)
        det_miss = [k for k in miss if k.startswith("det_head")]
        print(f"[det-abl] loaded {args.pretrained}: {len(miss)-len(det_miss)} non-det missing, "
              f"{len(det_miss)} det-head keys random-init, {len(unexp)} unexpected", flush=True)

    if args.freeze_encoder:                          # arm B: only the det head learns
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("det_head")

    from nuscenes.utils.splits import create_splits_scenes
    train_scenes = sorted(create_splits_scenes()["train"])
    val_scenes = sorted(create_splits_scenes()["val"])
    ihw, ds = model.image_hw, model.downsample
    dkw = dict(image_hw=ihw, downsample=ds, lidar_fusion=True, det_boxes=True, det_class_map=NUSC_10CLASS)
    n = 2 if args.smoke else args.max_samples
    train_ds = NuScenesOccTrainDataset(args.gts, nusc, scenes=train_scenes, max_samples=n, **dkw)
    val_ds = NuScenesOccTrainDataset(args.gts, nusc, scenes=val_scenes, max_samples=args.val_samples, **dkw)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mt,
                          num_workers=args.num_workers, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=1, collate_fn=collate_mt, num_workers=2)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    epochs = 1 if args.smoke else args.epochs
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(1, len(train_ld)))
             if args.cosine else None)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    print(f"[det-abl] {len(train_ds)} train / {len(val_ds)} val | fusion={args.lidar_fusion} "
          f"lidar_only={args.lidar_only} frozen={args.freeze_encoder} | "
          f"{sum(p.numel() for p in trainable)/1e6:.1f}M trainable", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_cfg = dict(backbone=args.backbone, decoder_layers=args.decoder_layers,
                    decoder_hidden=args.decoder_hidden, refine_iters=args.refine_iters,
                    lidar_fusion=args.lidar_fusion, lidar_only=args.lidar_only,
                    det_anchor_sizes=NUSC_10_SIZES, det_anchor_bottom=NUSC_10_BOTTOMS_EGO,
                    det_head_type=args.det_head)
    best_car = -1.0
    for ep in range(epochs):
        if args.freeze_encoder:                      # frozen encoder in eval (no BN-stat drift); head trains
            model.eval(); model.det_head.train()
        else:
            model.train()
        for it, b in enumerate(train_ld):
            with torch.cuda.amp.autocast(enabled=args.amp):
                lv = b["lidar_vox"].to(dev)
                occ, depth, aux = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev),
                                        b["intrins"].to(dev), lidar_vox=lv)
                l_det, _ = model.det_head.get_loss(aux["det"], b["det_gt"])
                loss = l_det
                if args.occ_weight > 0 and not args.freeze_encoder:
                    loss = loss + args.occ_weight * occ_loss(occ, b["semantics"].to(dev),
                                                             b["mask_camera"].to(dev))
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(opt); scaler.update()
            if sched is not None:
                sched.step()
            if it % 50 == 0:
                print(f"  ep{ep} it{it}: det_loss={l_det.item():.3f}", flush=True)
            if args.smoke and it >= 1:
                break
        mIoU, geo, car = evaluate(model, val_ld, dev)
        print(f"[det-abl] epoch {ep}: car_AP@0.5={car:.3f}  (occ_mIoU={mIoU:.3f})", flush=True)
        if car > best_car:                           # keep the peak (a late BN/fp16 collapse can't erase it)
            best_car = car
            torch.save({"state_dict": model.state_dict(), "cfg": ckpt_cfg},
                       os.path.join(args.out_dir, "det_abl.pth"))
            print(f"[det-abl]   saved best (car_AP={car:.3f})", flush=True)
    print(f"[det-abl] done (best car_AP={best_car:.3f}) -> {args.out_dir}/det_abl.pth", flush=True)


if __name__ == "__main__":
    main()
