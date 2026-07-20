"""
flashocc.eval_stereo
===================
Evaluate the ported **BEVStereo4DOCC** with the OFFICIAL supervised checkpoint on Occ3D-nuScenes val —
the in-codebase supervised **ceiling** (published mIoU 37.84). Loads the 562-param checkpoint (strict),
builds the BEVDet4D temporal-stereo inputs from the devkit (flashocc/data_stereo.py), and scores with
the same evaluator as the label-free runs.

    export CUDA_HOME=/data/rnd-liu/cuda_home2 PATH=$CUDA_HOME/bin:$PATH \
           LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST=9.0
    python -m DeepDataMiningLearning.ngperception.flashocc.eval_stereo \
        --nusc <nusc>/v1.0-trainval --gts <nusc>/v1.0-trainval/gts --max-samples 200
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from ..occupancy.datasets import Occ3DNuScenesDataset
from ..occupancy.evaluator import OccupancyEvaluator, OCC3D_CLASSES
from ..gaussian4d.teachers.base import TAIL_CLASSES, CLASS_NAMES


def main():
    ap = argparse.ArgumentParser(description="FlashOcc-4D-stereo supervised-ceiling eval on Occ3D val.")
    ap.add_argument("--nusc", required=True); ap.add_argument("--gts", required=True)
    ap.add_argument("--ckpt", default=None, help="official checkpoint (default: baked-in path)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    from ..flashocc.model_stereo import FlashOccBEVStereo4D
    from ..flashocc.data_stereo import build_img_inputs

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    model, miss, unexp = FlashOccBEVStereo4D.from_official_checkpoint(args.ckpt)
    print(f"[eval-stereo] checkpoint strict-load: {len(miss)} missing / {len(unexp)} unexpected", flush=True)
    model = model.to(dev).eval()
    occ = Occ3DNuScenesDataset(args.gts, scenes=sorted(splits.val))
    items = occ.items[: args.max_samples] if args.max_samples else occ.items
    print(f"[eval-stereo] {len(items)} val frames | official checkpoint (supervised ceiling)", flush=True)
    ev = OccupancyEvaluator()
    with torch.no_grad():
        for i, (sc, tok, lp) in enumerate(items):
            inp = [t.unsqueeze(0).to(dev) for t in build_img_inputs(nusc, tok)]
            out = model(inp)                                   # (1,18,200,200,16)
            pred = out.argmax(1)[0].cpu().numpy()              # (200,200,16)
            g = np.load(lp)
            ev.add(pred, g["semantics"].astype(np.uint8), g["mask_camera"].astype(bool))
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(items)}", flush=True)
    s = ev.summarize(verbose=False)
    tail = {CLASS_NAMES[c]: s["per_class"][OCC3D_CLASSES[c]] for c in TAIL_CLASSES}
    print(f"\n===== FlashOcc-4D-stereo (supervised ceiling) on Occ3D val =====")
    print(f"  mIoU = {s['mIoU']:.4f}   geo-IoU = {s['geo_IoU']:.4f}   "
          f"tail-IoU = {np.mean(list(tail.values())):.4f}   (published mIoU = 0.3784)")
    print("  per-class: " + "  ".join(f"{k}={v:.3f}" for k, v in
                                       sorted(s["per_class"].items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
