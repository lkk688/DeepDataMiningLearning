"""
ngperception.occupancy.train_modality_robust
=============================================

ONE model, BOTH tasks (occupancy + 10-class detection), robust to ANY modality
(camera-only / LiDAR-only / camera+LiDAR fusion) — every modality a positive gain.

Recipe: a fusion `LSSOccupancy` (center det head) trained with **per-batch modality dropout**
(randomly hide the camera or LiDAR branch, keeping the decoder's channel layout via
`forward(drop_camera=, drop_lidar=)`), so the single trunk learns to work with any subset.
Warm-started from our trained fusion trunk+occ-head + center det-head. occ-weight 1.0 (the
conflict is mild at that weight). Evaluated under all 3 modalities every epoch.

    python -m ...occupancy.train_modality_robust --gts <gts> --nusc <nusc> \
        --occ-ckpt output/lss_occ_full_fusion/lss_occ.pth \
        --det-ckpt output/abl_dcenter_frozen_sub/det_abl.pth --out-dir output/mod_robust
"""
from __future__ import annotations
import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from .models.lss_occ import LSSOccupancy
from .datasets_train import NuScenesOccTrainDataset
from .train_lss import occ_loss
from .train_multitask import collate_mt
from .evaluator import OccupancyEvaluator
from ..detection.nuscenes_dataset import NUSC_10CLASS
from ..detection.train_nuscenes import NUSC_10_SIZES
from ..detection.eval3d import eval_map

NUSC_10_BOTTOMS_EGO = [-1.0] * 10
# (drop_camera, drop_lidar) per modality
MODALITIES = {"fusion": (False, False), "camera": (False, True), "lidar": (True, False)}


@torch.no_grad()
def eval_modality(model, loader, dev, drop_camera, drop_lidar):
    """occ mIoU + det car BEV-AP@0.5 under one modality."""
    model.eval()
    ev = OccupancyEvaluator(); preds, gts = [], []
    for b in loader:
        lv = b["lidar_vox"].to(dev)
        occ, _, aux = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev),
                            b["intrins"].to(dev), lidar_vox=lv,
                            drop_camera=drop_camera, drop_lidar=drop_lidar)
        pred = occ.argmax(1).cpu().numpy()
        for j in range(pred.shape[0]):
            ev.add(pred[j], b["semantics"][j].numpy(), b["mask_camera"][j].numpy())
        for d in model.det_head.predict(aux["det"], score_thresh=0.2, nms_thresh=0.1):
            preds.append({k: v.cpu() for k, v in d.items()})
        for g in b["det_gt"]:
            gts.append({"boxes": g[:, :7], "labels": g[:, 7].long()})
    mIoU = ev.summarize(verbose=False)["mIoU"]
    car = eval_map(preds, gts, 1, 0.5)["mAP"] if preds else 0.0
    return mIoU, car


def main():
    ap = argparse.ArgumentParser(description="Modality-robust multi-task occ+det (camera/lidar/fusion).")
    ap.add_argument("--gts", required=True); ap.add_argument("--nusc", required=True)
    ap.add_argument("--occ-ckpt", required=True); ap.add_argument("--det-ckpt", default=None)
    ap.add_argument("--backbone", default="dinov2_base")
    ap.add_argument("--decoder-layers", type=int, default=4); ap.add_argument("--decoder-hidden", type=int, default=96)
    ap.add_argument("--refine-iters", type=int, default=2)
    ap.add_argument("--occ-weight", type=float, default=1.0); ap.add_argument("--det-weight", type=float, default=1.0)
    ap.add_argument("--mod-probs", default="0.34,0.33,0.33", help="P(fusion),P(camera),P(lidar) per batch")
    ap.add_argument("--max-samples", type=int, default=8000); ap.add_argument("--val-samples", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--cosine", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.device
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    probs = [float(x) for x in args.mod_probs.split(",")]
    mod_names = ["fusion", "camera", "lidar"]

    from nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    model = LSSOccupancy(backbone=args.backbone, decoder_hidden=args.decoder_hidden,
                         decoder_layers=args.decoder_layers, refine_iters=args.refine_iters,
                         lidar_fusion=True, det_classes=10, det_anchor_sizes=NUSC_10_SIZES,
                         det_anchor_bottom=NUSC_10_BOTTOMS_EGO, det_head_type="center").to(dev)
    miss, _ = model.load_state_dict(torch.load(args.occ_ckpt, map_location=dev), strict=False)
    print(f"[mod-robust] occ ckpt: {len(miss)} missing (det head)", flush=True)
    if args.det_ckpt:
        det_sd = torch.load(args.det_ckpt, map_location=dev)["state_dict"]
        model.load_state_dict({k: v for k, v in det_sd.items() if k.startswith("det_head")}, strict=False)
        print("[mod-robust] warm-started det head", flush=True)

    train_scenes = sorted(create_splits_scenes()["train"]); val_scenes = sorted(create_splits_scenes()["val"])
    ihw, dsf = model.image_hw, model.downsample
    dkw = dict(image_hw=ihw, downsample=dsf, lidar_fusion=True, det_boxes=True, det_class_map=NUSC_10CLASS)
    n = 2 if args.smoke else args.max_samples
    train_ds = NuScenesOccTrainDataset(args.gts, nusc, scenes=train_scenes, max_samples=n, **dkw)
    val_ds = NuScenesOccTrainDataset(args.gts, nusc, scenes=val_scenes, max_samples=args.val_samples, **dkw)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mt,
                          num_workers=args.num_workers, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=1, collate_fn=collate_mt, num_workers=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    epochs = 1 if args.smoke else args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if args.cosine else None
    print(f"[mod-robust] {len(train_ds)} train / {len(val_ds)} val | probs f/c/l={probs} | "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = dict(backbone=args.backbone, decoder_layers=args.decoder_layers, decoder_hidden=args.decoder_hidden,
               refine_iters=args.refine_iters, lidar_fusion=True, lidar_only=False,
               det_anchor_sizes=NUSC_10_SIZES, det_anchor_bottom=NUSC_10_BOTTOMS_EGO, det_head_type="center")
    best = -1.0
    for ep in range(epochs):
        model.train()
        for it, b in enumerate(train_ld):
            mod = random.choices(mod_names, weights=probs, k=1)[0]      # per-batch modality dropout
            dc, dl = MODALITIES[mod]
            lv = b["lidar_vox"].to(dev)
            occ, _, aux = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev),
                                b["intrins"].to(dev), lidar_vox=lv, drop_camera=dc, drop_lidar=dl)
            l_occ = occ_loss(occ, b["semantics"].to(dev), b["mask_camera"].to(dev))
            l_det, _ = model.det_head.get_loss(aux["det"], b["det_gt"])
            loss = args.occ_weight * l_occ + args.det_weight * l_det
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if it % 50 == 0:
                print(f"  ep{ep} it{it} [{mod}]: occ={l_occ.item():.3f} det={l_det.item():.3f}", flush=True)
            if args.smoke and it >= 2:
                break
        if sched is not None:
            sched.step()
        # eval all 3 modalities -> verify every one is a positive, usable result
        res = {}
        for name, (dc, dl) in MODALITIES.items():
            res[name] = eval_modality(model, val_ld, dev, dc, dl)
        line = "  ".join(f"{k}:occ={v[0]:.3f}/det={v[1]:.3f}" for k, v in res.items())
        worst = min(v[0] + v[1] for v in res.values())            # robustness = the weakest modality
        total = sum(v[0] + v[1] for v in res.values())
        print(f"[mod-robust] epoch {ep}: {line} | worst={worst:.3f} total={total:.3f}", flush=True)
        if worst > best:                                          # maximize the WORST modality (robustness)
            best = worst
            torch.save({"state_dict": model.state_dict(), "cfg": cfg}, os.path.join(args.out_dir, "mod_robust.pth"))
            print(f"[mod-robust]   saved best (worst modality occ+det={worst:.3f})", flush=True)
    print(f"[mod-robust] done (best worst-modality={best:.3f}) -> {args.out_dir}/mod_robust.pth", flush=True)


if __name__ == "__main__":
    main()
