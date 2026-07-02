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


def lovasz_grad(gt_sorted):
    """Gradient of the Lovász extension of the Jaccard loss (Berman et al., CVPR'18)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, ignore=17):
    """Multi-class Lovász-softmax on flat (P,C) probs / (P,) labels — a differentiable
    surrogate for mIoU. `ignore` skips a class (free) so it optimizes the scored classes."""
    C = probas.shape[1]
    losses = []
    for c in range(C):
        if c == ignore:
            continue
        fg = (labels == c).float()
        if fg.sum() == 0:                      # class absent in this batch -> no signal
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def occ_loss(occ, semantics, mask_camera, class_w=None, lovasz_w=0.0):
    """occ: (B,C,X,Y,Z); semantics: (B,X,Y,Z); mask_camera: (B,X,Y,Z) bool.
    CE (optionally class-balanced) + optional Lovász-softmax (direct mIoU surrogate)."""
    B, C = occ.shape[:2]
    logit = occ.permute(0, 2, 3, 4, 1).reshape(-1, C)[mask_camera.reshape(-1)]
    tgt = semantics.reshape(-1)[mask_camera.reshape(-1)]
    loss = F.cross_entropy(logit, tgt, weight=class_w)
    if lovasz_w > 0:
        loss = loss + lovasz_w * lovasz_softmax_flat(logit.float().softmax(1), tgt, ignore=17)
    return loss


def compute_class_weights(occ_ds, n_classes=18, power=0.25, cache=None):
    """Inverse-frequency class weights from the Occ3D GT over camera-visible voxels only
    (what the CE scores). `power` tempers it (0=uniform, 1=full inverse-freq); normalized to
    mean 1 and clamped so the loss scale is unchanged. Cached (only GT npz loads, no images)."""
    import numpy as _np
    if cache and os.path.exists(cache):
        return torch.tensor(_np.load(cache), dtype=torch.float32)
    counts = _np.zeros(n_classes, _np.float64)
    for i in range(len(occ_ds)):
        s = occ_ds[i]
        sem = s.semantics[s.mask_camera].reshape(-1)
        counts += _np.bincount(sem, minlength=n_classes)[:n_classes]
    freq = counts / max(counts.sum(), 1.0)
    w = (freq + 1e-6) ** (-power)
    w = _np.clip(w / w.mean(), 0.2, 8.0).astype(_np.float32)
    if cache:
        _np.save(cache, w)
    return torch.tensor(w, dtype=torch.float32)


def depth_loss(depth_pred, depth_gt):
    """depth_pred: (B,N,D,h,w) prob; depth_gt: (B,N,h,w) bin idx (-1 invalid). Hard CE —
    used for the *precise* LiDAR target."""
    B, N, D, h, w = depth_pred.shape
    logp = torch.log(depth_pred.clamp_min(1e-6)).permute(0, 1, 3, 4, 2).reshape(-1, D)
    tgt = depth_gt.reshape(-1)
    m = tgt >= 0
    if m.sum() == 0:
        return depth_pred.new_zeros(())
    return F.nll_loss(logp[m], tgt[m])


# region weights for the occ (rendered) depth target: cluttered background (terrain,
# manmade, vegetation) is hard/unlearnable -> low weight; road & objects -> full weight.
REGION_W = torch.ones(18); REGION_W[[14, 15, 16]] = 0.3


def depth_loss_flex(depth_pred, bins, region=None, window=0, use_region=False):
    """Depth loss with toggleable design (for the ablation). window=0 -> hard CE; window>0 ->
    tolerant windowed-mass. use_region -> per-cell weight by surface class (REGION_W)."""
    B, N, D, h, w = depth_pred.shape
    pred = depth_pred.permute(0, 1, 3, 4, 2).reshape(-1, D)
    tgt = bins.reshape(-1)
    m = tgt >= 0
    if m.sum() == 0:
        return depth_pred.new_zeros(())
    pred, tgt = pred[m], tgt[m]
    if window > 0:
        csum = torch.cumsum(pred, dim=1)
        hi = (tgt + window).clamp(max=D - 1)
        lo = (tgt - window - 1)
        mass = csum.gather(1, hi[:, None]).squeeze(1)
        mass = mass - torch.where(lo >= 0, csum.gather(1, lo.clamp(min=0)[:, None]).squeeze(1),
                                  torch.zeros_like(mass))
        per = -torch.log(mass.clamp_min(1e-6))
    else:
        per = -torch.log(pred.gather(1, tgt[:, None]).squeeze(1).clamp_min(1e-6))  # hard CE
    if use_region and region is not None:
        wt = REGION_W.to(pred.device)[region.reshape(-1)[m].clamp(min=0)]
        return (per * wt).sum() / wt.sum().clamp_min(1.0)
    return per.mean()


def depth_loss_occ_windowed(depth_pred, occ_bins, occ_region, k=2):
    """Tolerant occ-depth loss: reward total predicted depth-probability **mass within ±k
    bins** of the (voxel-quantized) target — classification-in-a-window, not regression —
    so the 0.4 m quantization can't hard-pull the prediction. Each cell is weighted by its
    surface class (REGION_W)."""
    B, N, D, h, w = depth_pred.shape
    pred = depth_pred.permute(0, 1, 3, 4, 2).reshape(-1, D)        # (M,D) prob
    tgt = occ_bins.reshape(-1)
    reg = occ_region.reshape(-1)
    m = tgt >= 0
    if m.sum() == 0:
        return depth_pred.new_zeros(())
    pred, tgt, reg = pred[m], tgt[m], reg[m]
    csum = torch.cumsum(pred, dim=1)
    hi = (tgt + k).clamp(max=D - 1)
    lo = (tgt - k - 1)
    mass = csum.gather(1, hi[:, None]).squeeze(1)
    has_lo = lo >= 0
    mass = mass - torch.where(has_lo, csum.gather(1, lo.clamp(min=0)[:, None]).squeeze(1),
                              torch.zeros_like(mass))
    wt = REGION_W.to(pred.device)[reg.clamp(min=0)]
    loss = -(torch.log(mass.clamp_min(1e-6)) * wt)
    return loss.sum() / wt.sum().clamp_min(1.0)


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    ev = OccupancyEvaluator()
    with torch.no_grad():
        for i, b in enumerate(loader):
            lv = b["lidar_vox"].to(device) if "lidar_vox" in b else None
            occ = model(b["imgs"].to(device), b["rots"].to(device),
                        b["trans"].to(device), b["intrins"].to(device), lidar_vox=lv)[0]
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
    ap.add_argument("--depth-source", choices=["lidar", "occ", "combined", "lidar_multi"],
                    default="lidar",
                    help="LiDAR | Occ3D-rendered | combined | lidar_multi (aggregated sweeps)")
    ap.add_argument("--lidar-sweeps", type=int, default=1, help="sweeps to aggregate (lidar_multi)")
    ap.add_argument("--lidar-cache", default=None, help="dir to cache aggregated multi-sweep points")
    ap.add_argument("--seed", type=int, default=None, help="random seed (for multi-seed runs)")
    ap.add_argument("--depth-tolerant", type=int, default=0,
                    help="lidar_multi loss: ±bins tolerant window (0=hard CE)")
    ap.add_argument("--depth-region", action="store_true",
                    help="lidar_multi loss: weight by surface region (road/obj vs clutter)")
    ap.add_argument("--occ-weight", type=float, default=0.5, help="weight of the occ term in 'combined'")
    ap.add_argument("--occ-window", type=int, default=2, help="±bins tolerance for the occ term")
    ap.add_argument("--occ-lovasz", type=float, default=0.0,
                    help="weight of the Lovász-softmax term on the occ head (0=off; mIoU surrogate)")
    ap.add_argument("--occ-class-balance", action="store_true",
                    help="inverse-frequency class weights on the occ CE")
    ap.add_argument("--occ-cb-power", type=float, default=0.25,
                    help="tempering exponent for class weights (0=uniform, 1=full inverse-freq)")
    ap.add_argument("--occ-cb-cache", default=None, help="cache file for computed class weights")
    ap.add_argument("--backbone", choices=["resnet18", "dinov2", "dinov2_base"], default="resnet18")
    ap.add_argument("--decoder-layers", type=int, default=2)
    ap.add_argument("--decoder-hidden", type=int, default=64)
    ap.add_argument("--feat-upsample", type=int, default=1,
                    help="upsample backbone features for a finer lift/supervision grid (2 -> 36x100)")
    ap.add_argument("--refine-iters", type=int, default=1,
                    help="iterative render-and-refine lift passes (1=single-shot; 2 = one refine)")
    ap.add_argument("--lidar-fusion", action="store_true",
                    help="fuse voxelized LiDAR as an input (train+inference), not just depth sup.")
    ap.add_argument("--refine-occ-weight", type=float, default=0.5,
                    help="deep-supervision weight on intermediate refine stages' occ loss")
    ap.add_argument("--cosine", action="store_true", help="cosine LR schedule")
    ap.add_argument("--amp", action="store_true", help="mixed precision (faster, less memory)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="DeepDataMiningLearning/ngperception/output/lss_occ")
    ap.add_argument("--smoke", action="store_true", help="2 samples, 1 epoch, no save")
    args = ap.parse_args()

    from nuscenes import NuScenes
    gen = None
    if args.seed is not None:
        import numpy as _np, random as _rnd
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        _np.random.seed(args.seed); _rnd.seed(args.seed)
        gen = torch.Generator(); gen.manual_seed(args.seed)
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    n = 2 if args.smoke else args.max_samples
    dev = args.device
    model = LSSOccupancy(backbone=args.backbone, decoder_hidden=args.decoder_hidden,
                         decoder_layers=args.decoder_layers, feat_upsample=args.feat_upsample,
                         refine_iters=args.refine_iters, lidar_fusion=args.lidar_fusion).to(dev)
    ihw, ds_factor = model.image_hw, model.downsample
    train_ds = NuScenesOccTrainDataset(args.gts, nusc, image_hw=ihw, downsample=ds_factor,
                                       max_samples=n, depth_source=args.depth_source,
                                       lidar_sweeps=args.lidar_sweeps, lidar_cache=args.lidar_cache,
                                       lidar_fusion=args.lidar_fusion)
    val_ds = NuScenesOccTrainDataset(args.gts, nusc, image_hw=ihw, downsample=ds_factor,
                                     max_samples=args.val_samples, stride=7,
                                     depth_source=args.depth_source, lidar_sweeps=args.lidar_sweeps,
                                     lidar_cache=args.lidar_cache, lidar_fusion=args.lidar_fusion)
    class_w = None
    if args.occ_class_balance:
        class_w = compute_class_weights(train_ds.occ, power=args.occ_cb_power,
                                        cache=args.occ_cb_cache).to(dev)
        print(f"[lss_occ] class weights (min/max): {class_w.min():.2f}/{class_w.max():.2f}", flush=True)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate, drop_last=True,
                          generator=gen)
    val_ld = DataLoader(val_ds, batch_size=1, num_workers=2, collate_fn=collate)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    epochs = 1 if args.smoke else args.epochs
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(1, len(train_ld)))
             if args.cosine else None)
    print(f"[lss_occ] backbone={args.backbone} dec={args.decoder_layers}x{args.decoder_hidden} "
          f"{len(train_ds)} train / {len(val_ds)} val | "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M trainable")

    for ep in range(epochs):
        model.train()
        for it, b in enumerate(train_ld):
            with torch.cuda.amp.autocast(enabled=args.amp):
                lv = b["lidar_vox"].to(dev) if "lidar_vox" in b else None
                occ, depth, aux = model(b["imgs"].to(dev), b["rots"].to(dev),
                                        b["trans"].to(dev), b["intrins"].to(dev), lidar_vox=lv)
                sem_d, mask_d = b["semantics"].to(dev), b["mask_camera"].to(dev)
                l_occ = occ_loss(occ, sem_d, mask_d, class_w=class_w, lovasz_w=args.occ_lovasz)
                if len(aux["occ"]) > 1:            # deep supervision on the earlier refine stages
                    for occ_i in aux["occ"][:-1]:
                        l_occ = l_occ + args.refine_occ_weight * occ_loss(
                            occ_i, sem_d, mask_d, class_w=class_w, lovasz_w=args.occ_lovasz)
                depth = depth.float()
                if args.depth_source == "combined":
                    l_dep = depth_loss(depth, b["depth_gt"].to(dev)) + args.occ_weight * \
                        depth_loss_occ_windowed(depth, b["occ_depth"].to(dev),
                                                b["occ_region"].to(dev), k=args.occ_window)
                elif args.depth_source == "lidar_multi":
                    l_dep = depth_loss_flex(depth, b["depth_gt"].to(dev),
                                            b["occ_region"].to(dev),
                                            window=args.depth_tolerant, use_region=args.depth_region)
                else:
                    l_dep = depth_loss(depth, b["depth_gt"].to(dev))
                loss = l_occ + args.depth_weight * l_dep
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if sched is not None:
                sched.step()
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
