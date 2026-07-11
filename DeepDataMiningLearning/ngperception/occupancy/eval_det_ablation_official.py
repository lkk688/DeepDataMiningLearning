"""
ngperception.occupancy.eval_det_ablation_official
==================================================

Official nuScenes `DetectionEval` for the occupancy-backbone detector (arms B/C/D from
`train_det_ablation.py`). The det head emits boxes in the **ego** frame (the occ grid), so
the transform to global is ego->global (ego-pose only) — vs the LiDAR->global used for the
standalone PointPillars evaluator.

    python -m DeepDataMiningLearning.ngperception.occupancy.eval_det_ablation_official \
        --gts <gts> --nusc <nuscenes> --ckpt output/det_abl_frozen_fusion/det_abl.pth \
        --out-dir output/det_abl_eval
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models.lss_occ import LSSOccupancy
from .datasets_train import NuScenesOccTrainDataset
from .train_multitask import collate_mt
from ..detection.nuscenes_dataset import NUSC_10CLASS
from ..detection.eval_nuscenes_official import DET_NAMES


def ego_box_to_global(nusc, token, box7, score, name):
    """box7 = [x,y,z,l,w,h,yaw] in the EGO frame -> a nuScenes results dict in global."""
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    x, y, z, l, w, h, yaw = [float(v) for v in box7]
    box = Box([x, y, z], [w, l, h], Quaternion(axis=[0, 0, 1], radians=yaw))   # size = w,l,h
    sd = nusc.get("sample_data", nusc.get("sample", token)["data"]["LIDAR_TOP"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    box.rotate(Quaternion(pose["rotation"])); box.translate(np.array(pose["translation"]))  # ego->global
    return {"sample_token": token, "translation": box.center.tolist(), "size": box.wlh.tolist(),
            "rotation": box.orientation.elements.tolist(), "velocity": [0.0, 0.0],
            "detection_name": name, "detection_score": float(score), "attribute_name": ""}


def main():
    ap = argparse.ArgumentParser(description="Official nuScenes eval for the occ-backbone detector.")
    ap.add_argument("--gts", required=True); ap.add_argument("--nusc", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="output/det_abl_eval")
    ap.add_argument("--score-thresh", type=float, default=0.05)
    ap.add_argument("--nms-thresh", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=4); ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    from nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from ..detection.train_nuscenes import NUSC_10_SIZES

    ck = torch.load(args.ckpt, map_location=dev); cfg = ck["cfg"]
    model = LSSOccupancy(backbone=cfg["backbone"], decoder_hidden=cfg["decoder_hidden"],
                         decoder_layers=cfg["decoder_layers"], refine_iters=cfg["refine_iters"],
                         lidar_fusion=cfg["lidar_fusion"], lidar_only=cfg["lidar_only"],
                         det_classes=10, det_anchor_sizes=cfg["det_anchor_sizes"],
                         det_anchor_bottom=cfg["det_anchor_bottom"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    print(f"[abl-eval] {args.ckpt} | fusion={cfg['lidar_fusion']} lidar_only={cfg['lidar_only']}")

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    val_scenes = sorted(create_splits_scenes()["val"])
    ihw, dsf = model.image_hw, model.downsample
    ds = NuScenesOccTrainDataset(args.gts, nusc, image_hw=ihw, downsample=dsf, scenes=val_scenes,
                                 max_samples=args.max_frames, lidar_fusion=True, det_boxes=False,
                                 det_class_map=NUSC_10CLASS)
    print(f"[abl-eval] official val: {len(ds)} frames")
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mt,
                    num_workers=args.num_workers)

    results = {}
    idx = 0
    with torch.no_grad():
        for b in ld:
            lv = b["lidar_vox"].to(dev)
            _, _, aux = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev),
                              b["intrins"].to(dev), lidar_vox=lv)
            dets = model.det_head.predict(aux["det"], score_thresh=args.score_thresh,
                                          nms_thresh=args.nms_thresh)
            for det in dets:
                token = ds.occ.items[idx][1]; idx += 1
                boxes = det["boxes"].cpu().numpy(); scores = det["scores"].cpu().numpy()
                labels = det["labels"].cpu().numpy()
                order = np.argsort(-scores)[:500]
                results[token] = [ego_box_to_global(nusc, token, boxes[j], scores[j],
                                                    DET_NAMES[int(labels[j])]) for j in order]
            if idx % 200 < args.batch_size:
                print(f"  {idx}/{len(ds)} frames", flush=True)

    res_path = os.path.join(args.out_dir, "results_nusc.json")
    meta = {"use_camera": True, "use_lidar": cfg["lidar_fusion"], "use_radar": False,
            "use_map": False, "use_external": False}
    with open(res_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f)
    print(f"[abl-eval] wrote {res_path} ({sum(len(v) for v in results.values())} boxes)")

    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.detection.config import config_factory
    de = DetectionEval(nusc, config=config_factory("detection_cvpr_2019"), result_path=res_path,
                       eval_set="val", output_dir=args.out_dir, verbose=True)
    summary = de.main(render_curves=False)
    print("\n================ OFFICIAL nuScenes metrics (occ-backbone detector) ================")
    print(f"  NDS = {summary['nd_score']:.4f}   mAP = {summary['mean_ap']:.4f}")
    for name, ap_ in summary["mean_dist_aps"].items():
        print(f"    {name:<22} {ap_:.4f}")
    for k, v in summary["tp_errors"].items():
        print(f"    {k:<10} {v:.4f}")


if __name__ == "__main__":
    main()
