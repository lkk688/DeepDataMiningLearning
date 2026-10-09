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
    ap.add_argument("--persample-out", default=None, help="per-sample [tp,fp,fn] counts (.npz) for benchmarks")
    ap.add_argument("--pred-dir", default=None, help="save predicted occ grids (uint8 .npz) for visualization")
    ap.add_argument("--pred-every", type=int, default=10, help="save every k-th prediction to --pred-dir")
    ap.add_argument("--common-counts-out", default=None,
                    help="also write per-sample counts in Qwen-Drive's 10-class taxonomy (pred AND gt mapped), "
                         "so FlashOcc and Qwen-Drive can be compared on one label space")
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
    common = None
    if args.common_counts_out:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bevdet', 'lidar_tokens'))
        from qwen_drive_labels import occ3d_to_qd
        from ..occupancy.common_taxonomy import TaxonomyCounter
        common = TaxonomyCounter()
    with torch.no_grad():
        for i, (sc, tok, lp) in enumerate(items):
            inp = [t.unsqueeze(0).to(dev) for t in build_img_inputs(nusc, tok)]
            out = model(inp)                                   # (1,18,200,200,16)
            pred = out.argmax(1)[0].cpu().numpy()              # (200,200,16)
            g = np.load(lp)
            ev.add(pred, g["semantics"].astype(np.uint8), g["mask_camera"].astype(bool))
            if common is not None:
                common.add(occ3d_to_qd(pred), occ3d_to_qd(g["semantics"].astype(np.uint8)), g["mask_camera"].astype(bool))
            if args.pred_dir and i % args.pred_every == 0:
                import os
                os.makedirs(args.pred_dir, exist_ok=True)
                np.savez_compressed(os.path.join(args.pred_dir, f"{tok}.npz"), pred=pred.astype(np.uint8))
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(items)}", flush=True)
    s = ev.summarize(verbose=False)
    if common is not None:
        common.dump(args.common_counts_out, [tok for (_, tok, _) in items], meta=vars(args))
    if args.persample_out:
        ev.dump(args.persample_out, [tok for (_, tok, _) in items], meta=vars(args))
        print("wrote per-sample counts ->", args.persample_out, flush=True)
    tail = {CLASS_NAMES[c]: s["per_class"][OCC3D_CLASSES[c]] for c in TAIL_CLASSES}
    print(f"\n===== FlashOcc-4D-stereo (supervised ceiling) on Occ3D val =====")
    print(f"  mIoU = {s['mIoU']:.4f}   geo-IoU = {s['geo_IoU']:.4f}   "
          f"tail-IoU = {np.mean(list(tail.values())):.4f}   (published mIoU = 0.3784)")
    print("  per-class: " + "  ".join(f"{k}={v:.3f}" for k, v in
                                       sorted(s["per_class"].items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
