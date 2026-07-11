"""
ngperception.occupancy.eval_val
================================

Evaluate a trained LSS-occupancy checkpoint on the **official Occ3D-nuScenes val split**
(the 150 official val scenes, `nuscenes.utils.splits.val`) and report the full **per-class
IoU**, mIoU, and geometric IoU — the real Occ3D metric, over all camera-visible voxels.

Unlike `train_lss.py`'s in-loop `--val-samples` (a small stride-sampled subset drawn from
*all* scenes), this filters strictly to the official val scenes, so it is the correct
held-out measure — *provided the checkpoint was trained on train-scenes only*.

Example
-------
python -m DeepDataMiningLearning.ngperception.occupancy.eval_val \
    --gts <gts> --nusc <nuscenes> --ckpt output/lss_occ_full/lss_occ.pth \
    --backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --refine-iters 2
# add --lidar-fusion for the fusion checkpoint.
"""
from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader

from .models.lss_occ import LSSOccupancy
from .datasets_train import NuScenesOccTrainDataset
from .evaluator import OccupancyEvaluator, OCC3D_CLASSES, NUM_SEMANTIC
from .train_lss import collate


def main():
    ap = argparse.ArgumentParser(description="Official Occ3D-nuScenes val eval (per-class IoU).")
    ap.add_argument("--gts", required=True)
    ap.add_argument("--nusc", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="dinov2_base")
    ap.add_argument("--decoder-layers", type=int, default=4)
    ap.add_argument("--decoder-hidden", type=int, default=96)
    ap.add_argument("--refine-iters", type=int, default=2)
    ap.add_argument("--feat-upsample", type=int, default=1)
    ap.add_argument("--lidar-fusion", action="store_true")
    ap.add_argument("--lidar-cache", default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="cap val frames (debug); default = all")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from nuscenes import NuScenes
    from nuscenes.utils import splits
    dev = args.device

    model = LSSOccupancy(backbone=args.backbone, decoder_hidden=args.decoder_hidden,
                         decoder_layers=args.decoder_layers, feat_upsample=args.feat_upsample,
                         refine_iters=args.refine_iters, lidar_fusion=args.lidar_fusion).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[eval_val] loaded {args.ckpt} | backbone={args.backbone} "
          f"dec={args.decoder_layers}x{args.decoder_hidden} refine={args.refine_iters} "
          f"fusion={args.lidar_fusion}")

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    val_scenes = sorted(splits.val)                       # 150 official val scene names
    ihw, ds_factor = model.image_hw, model.downsample
    ds = NuScenesOccTrainDataset(args.gts, nusc, image_hw=ihw, downsample=ds_factor,
                                 scenes=val_scenes, max_samples=args.max_samples,
                                 lidar_cache=args.lidar_cache, lidar_fusion=args.lidar_fusion)
    print(f"[eval_val] official val: {len(val_scenes)} scenes -> {len(ds)} frames")
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate)

    ev = OccupancyEvaluator()
    with torch.no_grad():
        for i, b in enumerate(ld):
            lv = b["lidar_vox"].to(dev) if "lidar_vox" in b else None
            occ = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev),
                        b["intrins"].to(dev), lidar_vox=lv)[0]
            pred = occ.argmax(1).cpu().numpy()
            for j in range(pred.shape[0]):
                ev.add(pred[j], b["semantics"][j].numpy(), b["mask_camera"][j].numpy())
            if (i + 1) % 20 == 0:
                print(f"  {(i+1)*args.batch_size}/{len(ds)} frames", flush=True)

    out = ev.summarize(verbose=False)
    print("\n=== official Occ3D-nuScenes val ===")
    print(f"frames={out['num_samples']}  mIoU={out['mIoU']:.4f}  geo_IoU={out['geo_IoU']:.4f}")
    print("\n--- per-class IoU (17 semantic classes) ---")
    per = out["per_class"]
    width = max(len(k) for k in per)
    for c in range(NUM_SEMANTIC):
        name = OCC3D_CLASSES[c]
        print(f"  {c:2d}  {name:<{width}}  {per[name]:.4f}")


if __name__ == "__main__":
    main()
