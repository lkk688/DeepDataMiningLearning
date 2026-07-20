"""
flashocc.train_flashocc
=======================
Train the ported **FlashOcc/BEVDet-Occ** backbone **label-free** from our voxel-soft teacher (no human
3D labels) — the external-validation backbone for the label-free occupancy result. Reuses the exact
label-free target + loss from the LSS experiments (`gaussian4d/train_student.py`): only the model
changes (LSSOccupancy -> FlashOccBEVDet), so the comparison to backbone set #1 is clean.

Requires the CUDA env for the bev_pool_v2 op (see docs/TUTORIAL_FlashOcc.md §2):
    export CUDA_HOME=/data/rnd-liu/cuda_home2 PATH=$CUDA_HOME/bin:$PATH \
           LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST=9.0

    python -m DeepDataMiningLearning.ngperception.flashocc.train_flashocc \
        --nusc <nusc>/v1.0-trainval --gts <nusc>/v1.0-trainval/gts \
        --teacher-cache <nusc>/teacher_cache_2x2soft/voxel10 \
        --epochs 24 --batch-size 4 --amp --factorized --out-dir output/flashocc_r50_voxelsoft
"""
from __future__ import annotations
import argparse
import os
import torch
from torch.utils.data import DataLoader

# reuse the label-free dataset + losses from the LSS student (single source of truth)
from ..gaussian4d.train_student import (StudentDataset, collate,
                                        teacher_loss, teacher_loss_soft, teacher_loss_factorized)


def main():
    ap = argparse.ArgumentParser(description="Label-free FlashOcc training from the voxel-soft teacher.")
    ap.add_argument("--nusc", required=True); ap.add_argument("--gts", required=True)
    ap.add_argument("--teacher-cache", required=True)
    ap.add_argument("--model", default="bevdet_r50", choices=["bevdet_r50"])
    ap.add_argument("--epochs", type=int, default=24); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lovasz", type=float, default=0.1)
    ap.add_argument("--factorized", action="store_true"); ap.add_argument("--sem-weight", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    dev = args.device

    from ..flashocc.model import FlashOccBEVDet
    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    model = FlashOccBEVDet().to(dev)
    ds = StudentDataset(args.teacher_cache, nusc, args.gts, image_hw=(256, 704), downsample=16)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate, drop_last=True)
    mode = "FACTORIZED geom+sem" if args.factorized else ("SOFT top-K" if ds.soft else "HARD CE")
    print(f"[flashocc] {len(ds)} frames | FlashOcc-{args.model} label-free | teacher="
          f"{os.path.basename(args.teacher_cache.rstrip('/'))} | {mode}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    os.makedirs(args.out_dir, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        for it, b in enumerate(ld):
            with torch.cuda.amp.autocast(enabled=args.amp):
                occ = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev), b["intrins"].to(dev))
                if args.factorized:
                    assert ds.soft, "--factorized needs a soft teacher cache"
                    loss = teacher_loss_factorized(occ, b["tsem"].to(dev), b["tweight"].to(dev),
                                                   b["tsoft_idx"].to(dev), b["tsoft_prob"].to(dev),
                                                   sem_w=args.sem_weight)
                elif ds.soft:
                    loss = teacher_loss_soft(occ, b["tsem"].to(dev), b["tweight"].to(dev),
                                             b["tsoft_idx"].to(dev), b["tsoft_prob"].to(dev), lovasz_w=args.lovasz)
                else:
                    loss = teacher_loss(occ, b["tsem"].to(dev), b["tweight"].to(dev), lovasz_w=args.lovasz)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            if it % 50 == 0:
                print(f"  ep{ep} it{it}: loss={loss.item():.3f}", flush=True)
        torch.save(model.state_dict(), os.path.join(args.out_dir, "flashocc.pth"))
        print(f"[flashocc] epoch {ep} saved -> {args.out_dir}/flashocc.pth", flush=True)


if __name__ == "__main__":
    main()
