"""
ngperception.occupancy.train_multitask_joint
=============================================

Joint occupancy + 10-class detection on ONE shared fused-voxel encoder, built to **resolve
the occ-head vs det-head conflict** the ablation quantified (finetuning the shared trunk for
detection trades −0.244 occ mIoU for +0.021 det mAP).

Warm-starts from our trained weights (no from-scratch):
  --occ-ckpt  lss_occ_full_fusion/lss_occ.pth      (shared trunk + occ head)
  --det-ckpt  abl_dcenter_frozen_sub/det_abl.pth   (center det head)

Three recipes, reporting BOTH occ mIoU and det car_AP each epoch:
  --mode frozen  : freeze the trunk+occ-head, train only the det head   (baseline, no conflict)
  --mode naive   : joint finetune, loss = occ + det_weight*det           (shows the conflict)
  --mode pcgrad  : joint finetune with PCGrad gradient surgery on the    (target: beat the trade-off)
                   shared params (project conflicting per-task grads)

Center head + fp32 (fp16 corrupts BatchNorm running stats) + best-saving by (occ+det) score.
"""
from __future__ import annotations
import argparse
import os
import torch
from torch.utils.data import DataLoader

from .models.lss_occ import LSSOccupancy
from .datasets_train import NuScenesOccTrainDataset
from .train_lss import occ_loss
from .train_multitask import collate_mt, evaluate
from ..detection.nuscenes_dataset import NUSC_10CLASS
from ..detection.train_nuscenes import NUSC_10_SIZES

NUSC_10_BOTTOMS_EGO = [-1.0] * 10


def pcgrad_combine(losses, params, opt):
    """PCGrad: per-task grads, project away pairwise conflicts, set combined .grad. fp32 path."""
    grads = []
    for i, loss in enumerate(losses):
        opt.zero_grad(set_to_none=True)
        loss.backward(retain_graph=(i < len(losses) - 1))
        grads.append([(p.grad.detach().clone() if p.grad is not None else None) for p in params])

    def flat(gs):
        return torch.cat([(g.flatten() if g is not None else torch.zeros(p.numel(), device=p.device))
                          for g, p in zip(gs, params)])
    flats = [flat(g) for g in grads]
    proj = [f.clone() for f in flats]
    for i in range(len(proj)):                       # project task i off any task j it conflicts with
        for j in range(len(flats)):
            if i == j:
                continue
            dot = torch.dot(proj[i], flats[j])
            if dot < 0:
                proj[i] = proj[i] - dot / (flats[j].dot(flats[j]) + 1e-12) * flats[j]
    combined = sum(proj)
    opt.zero_grad(set_to_none=True)
    idx = 0
    for p in params:
        n = p.numel()
        p.grad = combined[idx:idx + n].view_as(p).clone()
        idx += n


def main():
    ap = argparse.ArgumentParser(description="Joint occ+det on a shared encoder (conflict study).")
    ap.add_argument("--gts", required=True); ap.add_argument("--nusc", required=True)
    ap.add_argument("--occ-ckpt", required=True, help="fusion occ ckpt: shared trunk + occ head")
    ap.add_argument("--det-ckpt", default=None, help="center det-head ckpt to warm-start the det head")
    ap.add_argument("--mode", choices=["frozen", "naive", "pcgrad"], default="pcgrad")
    ap.add_argument("--backbone", default="dinov2_base")
    ap.add_argument("--decoder-layers", type=int, default=4); ap.add_argument("--decoder-hidden", type=int, default=96)
    ap.add_argument("--refine-iters", type=int, default=2)
    ap.add_argument("--occ-weight", type=float, default=1.0); ap.add_argument("--det-weight", type=float, default=1.0)
    ap.add_argument("--max-samples", type=int, default=8000); ap.add_argument("--val-samples", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.device

    from nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    model = LSSOccupancy(backbone=args.backbone, decoder_hidden=args.decoder_hidden,
                         decoder_layers=args.decoder_layers, refine_iters=args.refine_iters,
                         lidar_fusion=True, det_classes=10, det_anchor_sizes=NUSC_10_SIZES,
                         det_anchor_bottom=NUSC_10_BOTTOMS_EGO, det_head_type="center").to(dev)
    # warm-start: occ ckpt -> trunk + occ head; det ckpt -> det head
    miss, unexp = model.load_state_dict(torch.load(args.occ_ckpt, map_location=dev), strict=False)
    print(f"[mt-joint] occ ckpt: {len(miss)} missing (det head), {len(unexp)} unexpected", flush=True)
    if args.det_ckpt:
        det_sd = torch.load(args.det_ckpt, map_location=dev)["state_dict"]
        model.load_state_dict({k: v for k, v in det_sd.items() if k.startswith("det_head")}, strict=False)
        print("[mt-joint] warm-started det head from", args.det_ckpt, flush=True)

    # "trunk" = shared encoder + LiDAR branch (produces vox); occ decoder & det head are task-specific necks
    is_det = lambda n: n.startswith("det_head")
    is_occ_neck = lambda n: n.startswith("decoder")
    if args.mode == "frozen":                        # only the det head learns; occ stays put
        for n, p in model.named_parameters():
            p.requires_grad = is_det(n)
    trainable = [p for _, p in model.named_parameters() if p.requires_grad]
    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    epochs = 1 if args.smoke else args.epochs
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if args.cosine else None)

    from nuscenes.utils.splits import create_splits_scenes as _splits
    train_scenes = sorted(_splits()["train"]); val_scenes = sorted(_splits()["val"])
    ihw, dsf = model.image_hw, model.downsample
    dkw = dict(image_hw=ihw, downsample=dsf, lidar_fusion=True, det_boxes=True, det_class_map=NUSC_10CLASS)
    n = 2 if args.smoke else args.max_samples
    train_ds = NuScenesOccTrainDataset(args.gts, nusc, scenes=train_scenes, max_samples=n, **dkw)
    val_ds = NuScenesOccTrainDataset(args.gts, nusc, scenes=val_scenes, max_samples=args.val_samples, **dkw)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mt,
                          num_workers=args.num_workers, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=1, collate_fn=collate_mt, num_workers=2)
    print(f"[mt-joint] mode={args.mode} | {len(train_ds)} train / {len(val_ds)} val | "
          f"{sum(p.numel() for p in trainable)/1e6:.1f}M trainable", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = dict(backbone=args.backbone, decoder_layers=args.decoder_layers, decoder_hidden=args.decoder_hidden,
               refine_iters=args.refine_iters, lidar_fusion=True, lidar_only=False,
               det_anchor_sizes=NUSC_10_SIZES, det_anchor_bottom=NUSC_10_BOTTOMS_EGO, det_head_type="center")
    best = -1.0
    for ep in range(epochs):
        if args.mode == "frozen":
            model.eval(); model.det_head.train()     # frozen trunk in eval (no BN drift); det head trains
        else:
            model.train()
        for it, b in enumerate(train_ld):
            lv = b["lidar_vox"].to(dev)
            occ, depth, aux = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev),
                                    b["intrins"].to(dev), lidar_vox=lv)
            l_det, _ = model.det_head.get_loss(aux["det"], b["det_gt"])
            if args.mode == "frozen":                # occ fixed -> only det loss trains the head
                opt.zero_grad(); (args.det_weight * l_det).backward()
            else:
                l_occ = occ_loss(occ, b["semantics"].to(dev), b["mask_camera"].to(dev))
                if args.mode == "pcgrad":
                    pcgrad_combine([args.occ_weight * l_occ, args.det_weight * l_det], trainable, opt)
                else:                                # naive: sum the losses
                    opt.zero_grad(); (args.occ_weight * l_occ + args.det_weight * l_det).backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()
            if it % 50 == 0:
                extra = "" if args.mode == "frozen" else f" occ={l_occ.item():.3f}"
                print(f"  ep{ep} it{it}: det={l_det.item():.3f}{extra}", flush=True)
            if args.smoke and it >= 1:
                break
        if sched is not None:
            sched.step()
        mIoU, geo, car = evaluate(model, val_ld, dev)
        score = mIoU + car                            # multi-task: reward BOTH
        print(f"[mt-joint] epoch {ep}: occ_mIoU={mIoU:.3f} det_car_AP={car:.3f} (score={score:.3f})", flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "cfg": cfg},
                       os.path.join(args.out_dir, "mt_joint.pth"))
            print(f"[mt-joint]   saved best (occ={mIoU:.3f} det={car:.3f})", flush=True)
    print(f"[mt-joint] done (best occ+det score={best:.3f}) -> {args.out_dir}/mt_joint.pth", flush=True)


if __name__ == "__main__":
    main()
