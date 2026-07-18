"""
gaussian4d.eval_student
=======================
Evaluate a camera-only student on the **official Occ3D-nuScenes val split** and report the Phase-1
gate metrics: mIoU, geo-IoU, and **tail-class IoU** (the classes where the Gaussian teacher's
less-quantization/uncertainty advantages should show). Camera-only inference (no LiDAR at test).

    python -m DeepDataMiningLearning.ngperception.gaussian4d.eval_student \
        --nusc <nuscenes> --gts <gts> --ckpt output/student_gaussian/student.pth
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..occupancy.models.lss_occ import LSSOccupancy
from ..occupancy.datasets_train import NuScenesOccTrainDataset
from ..occupancy.evaluator import OccupancyEvaluator, OCC3D_CLASSES
from .teachers.base import TAIL_CLASSES, CLASS_NAMES


def collate(b):
    return {k: torch.stack([x[k] for x in b]) for k in b[0]}


def main():
    ap = argparse.ArgumentParser(description="Camera-only student eval on Occ3D val (mIoU/geo/tail).")
    ap.add_argument("--nusc", required=True); ap.add_argument("--gts", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="dinov2_base")
    ap.add_argument("--decoder-layers", type=int, default=4); ap.add_argument("--decoder-hidden", type=int, default=96)
    ap.add_argument("--max-samples", type=int, default=None); ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    model = LSSOccupancy(backbone=args.backbone, decoder_hidden=args.decoder_hidden,
                         decoder_layers=args.decoder_layers, refine_iters=1, lidar_fusion=False).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)); model.eval()
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    ds = NuScenesOccTrainDataset(args.gts, nusc, image_hw=model.image_hw, downsample=model.downsample,
                                 scenes=sorted(splits.val), max_samples=args.max_samples, stride=1)
    ld = DataLoader(ds, batch_size=1, num_workers=args.num_workers, collate_fn=collate)
    print(f"[eval-student] {len(ds)} val frames | camera-only | {args.ckpt}", flush=True)
    ev = OccupancyEvaluator()
    with torch.no_grad():
        for i, b in enumerate(ld):
            occ = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev), b["intrins"].to(dev))[0]
            pred = occ.argmax(1).cpu().numpy()
            for j in range(pred.shape[0]):
                ev.add(pred[j], b["semantics"][j].numpy(), b["mask_camera"][j].numpy())
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(ds)}", flush=True)
    s = ev.summarize(verbose=False)
    tail = {CLASS_NAMES[c]: s["per_class"][OCC3D_CLASSES[c]] for c in TAIL_CLASSES}
    print(f"\n===== STUDENT (camera-only) on Occ3D val =====")
    print(f"  mIoU = {s['mIoU']:.4f}   geo-IoU = {s['geo_IoU']:.4f}   "
          f"tail-IoU = {np.mean(list(tail.values())):.4f}")
    print("  tail per-class: " + "  ".join(f"{k}={v:.3f}" for k, v in tail.items()))
    print("  top classes: " + "  ".join(f"{k}={v:.2f}" for k, v in
                                          sorted(s["per_class"].items(), key=lambda x: -x[1])[:6]))


if __name__ == "__main__":
    main()
