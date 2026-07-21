"""
flashocc.eval_det_stereo
=======================
Official nuScenes mAP/NDS for the detection head trained on the frozen supervised 4D-stereo backbone.
Reuses the submission + DetectionEval machinery from occupancy.eval_det_ablation_official; only the
model + input pipeline differ (FlashOcc temporal-stereo via data_stereo).

    export CUDA_HOME=/data/rnd-liu/cuda_home2 PATH=$CUDA_HOME/bin:$PATH \
           LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST=9.0
    python -m DeepDataMiningLearning.ngperception.flashocc.eval_det_stereo \
        --nusc <nusc>/v1.0-trainval --gts <nusc>/v1.0-trainval/gts \
        --det-head output/det_on_stereo_frozen/det_head.pth --out-dir output/det_on_stereo_eval
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import torch

from ..occupancy.datasets import Occ3DNuScenesDataset
from ..occupancy.eval_det_ablation_official import ego_box_to_global
from ..detection.eval_nuscenes_official import DET_NAMES


def main():
    ap = argparse.ArgumentParser(description="Official nuScenes eval: det head on frozen 4D-stereo backbone.")
    ap.add_argument("--nusc", required=True); ap.add_argument("--gts", required=True)
    ap.add_argument("--det-head", required=True, help="trained det_head.pth")
    ap.add_argument("--out-dir", default="output/det_on_stereo_eval")
    ap.add_argument("--score-thresh", type=float, default=0.05); ap.add_argument("--nms-thresh", type=float, default=0.2)
    ap.add_argument("--max-frames", type=int, default=None); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    from .train_det_stereo import FlashOccDet
    from .data_stereo import build_img_inputs

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    model = FlashOccDet().to(dev)
    model.det_head.load_state_dict(torch.load(args.det_head, map_location=dev)); model.eval()
    occ = Occ3DNuScenesDataset(args.gts, scenes=sorted(splits.val))
    items = occ.items[: args.max_frames] if args.max_frames else occ.items
    print(f"[det-stereo-eval] {len(items)} val frames | frozen 4D-stereo + trained CenterHead", flush=True)

    results = {}
    with torch.no_grad():
        for i, (sc, tok, lp) in enumerate(items):
            inp = [t.unsqueeze(0).to(dev) for t in build_img_inputs(nusc, tok)]
            pred = model(inp)
            det = model.det_head.predict(pred, score_thresh=args.score_thresh, nms_thresh=args.nms_thresh)[0]
            boxes = det["boxes"].cpu().numpy(); scores = det["scores"].cpu().numpy()
            labels = det["labels"].cpu().numpy()
            order = np.argsort(-scores)[:500]
            results[tok] = [ego_box_to_global(nusc, tok, boxes[j], scores[j], DET_NAMES[int(labels[j])])
                            for j in order]
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(items)}", flush=True)

    res_path = os.path.join(args.out_dir, "results_nusc.json")
    with open(res_path, "w") as f:
        json.dump({"meta": {"use_camera": True, "use_lidar": False, "use_radar": False,
                            "use_map": False, "use_external": False}, "results": results}, f)
    print(f"[det-stereo-eval] wrote {res_path} ({sum(len(v) for v in results.values())} boxes)", flush=True)

    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.detection.config import config_factory
    de = DetectionEval(nusc, config=config_factory("detection_cvpr_2019"), result_path=res_path,
                       eval_set="val", output_dir=args.out_dir, verbose=True)
    summary = de.main(render_curves=False)
    print("\n========= OFFICIAL nuScenes metrics: detection head on FROZEN supervised 4D-stereo =========")
    print(f"  NDS = {summary['nd_score']:.4f}   mAP = {summary['mean_ap']:.4f}   (cf. LSS-backbone det NDS 0.206)")
    for name, ap_ in summary["mean_dist_aps"].items():
        print(f"    {name:<22} {ap_:.4f}")


if __name__ == "__main__":
    main()
