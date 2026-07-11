"""
ngperception.detection.eval_nuscenes_official
==============================================

Evaluate a trained nuScenes detector with the **official nuScenes devkit metric**
(`nuscenes.eval.detection.DetectionEval`): NDS, mAP (mean over center-distance thresholds
0.5/1/2/4 m and 10 classes), per-class AP, and the TP errors mATE/mASE/mAOE/mAVE/mAAE.

Pipeline: rebuild the model from a checkpoint saved by `train_nuscenes.py` (state_dict + cfg),
run it over the **full official val split**, transform each predicted box LiDAR->global,
write a nuScenes results JSON, then run the devkit evaluator.

Caveats (honest): our PointPillars predicts no velocity and no attribute, so `velocity=[0,0]`
and `attribute_name=""` — mAVE/mAAE are therefore penalized (and drag NDS). mAP and per-class
AP are the clean headline numbers.

    python -m DeepDataMiningLearning.ngperception.detection.eval_nuscenes_official \
        --root <nuscenes> --ckpt output/nusc_det_ckpt/nusc_det_best.pth \
        --lidar-cache output/nusc_det_lidar_cache --out-dir output/nusc_det_eval
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch

# index -> official nuScenes detection_name (order = NUSC_10CLASS values)
DET_NAMES = ["car", "truck", "construction_vehicle", "bus", "trailer",
             "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]


def build_model(cfg, dev):
    from .pointpillars import PointPillars
    from .centerpoint import CenterPoint
    common = dict(num_point_features=4, num_classes=cfg["num_classes"], pc_range=cfg["pc_range"],
                  voxel_size=cfg["voxel_size"], backbone=cfg["backbone"], max_pillars=cfg["max_pillars"])
    if cfg["model"] == "centerpoint":
        model = CenterPoint(**common)
    else:
        model = PointPillars(anchor_sizes=cfg["anchor_sizes"], anchor_bottom=cfg["anchor_bottom"],
                             **common)
    return model.to(dev)


def lidar_box_to_global(nusc, sample, box7, score, name):
    """box7 = [x,y,z,l,w,h,yaw] in LIDAR_TOP frame -> a nuScenes results dict in the global frame."""
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    x, y, z, l, w, h, yaw = [float(v) for v in box7]
    box = Box([x, y, z], [w, l, h], Quaternion(axis=[0, 0, 1], radians=yaw))   # size = w,l,h
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    box.rotate(Quaternion(cs["rotation"])); box.translate(np.array(cs["translation"]))     # sensor->ego
    box.rotate(Quaternion(pose["rotation"])); box.translate(np.array(pose["translation"]))  # ego->global
    return {"sample_token": sample["token"],
            "translation": box.center.tolist(),
            "size": box.wlh.tolist(),
            "rotation": box.orientation.elements.tolist(),
            "velocity": [0.0, 0.0],
            "detection_name": name,
            "detection_score": float(score),
            "attribute_name": ""}


def main():
    ap = argparse.ArgumentParser(description="Official nuScenes DetectionEval on a trained detector.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lidar-cache", default=None)
    ap.add_argument("--out-dir", default="output/nusc_det_eval")
    ap.add_argument("--score-thresh", type=float, default=0.05)
    ap.add_argument("--nms-thresh", type=float, default=0.2)
    ap.add_argument("--max-frames", type=int, default=None, help="cap val frames (debug); default = all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    from nuscenes import NuScenes
    from .nuscenes_dataset import NuScenesCarDataset, NUSC_10CLASS
    from .train_nuscenes import NUSC_PCR

    ck = torch.load(args.ckpt, map_location=dev)
    cfg = ck["cfg"]
    model = build_model(cfg, dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    print(f"[eval-official] {args.ckpt} | {cfg['model']}/{cfg['backbone']} | {cfg['num_classes']}-class")

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.root, verbose=False)
    ds = NuScenesCarDataset(dataroot=args.root, split="val", max_frames=args.max_frames,
                            pc_range=NUSC_PCR, nusc=nusc, sweeps=cfg["sweeps"],
                            lidar_cache=args.lidar_cache,
                            class_map=(NUSC_10CLASS if cfg["multiclass"] else None))
    print(f"[eval-official] official val: {len(ds)} frames")

    results = {}
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            token = s["id"]
            sample = nusc.get("sample", token)
            pred = model([s["points"].numpy()])
            det = model.head.predict(pred, score_thresh=args.score_thresh, nms_thresh=args.nms_thresh)[0]
            boxes, scores, labels = (det["boxes"].cpu().numpy(),
                                     det["scores"].cpu().numpy(), det["labels"].cpu().numpy())
            # keep the top-500 by score (nuScenes caps at 500/sample)
            order = np.argsort(-scores)[:500]
            entries = [lidar_box_to_global(nusc, sample, boxes[j], scores[j], DET_NAMES[int(labels[j])])
                       for j in order]
            results[token] = entries
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(ds)} frames  (last: {len(entries)} dets)", flush=True)

    res_path = os.path.join(args.out_dir, "results_nusc.json")
    meta = {"use_camera": False, "use_lidar": True, "use_radar": False,
            "use_map": False, "use_external": False}
    with open(res_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f)
    print(f"[eval-official] wrote {res_path} ({sum(len(v) for v in results.values())} boxes)")

    # ---- official devkit evaluation ----
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.detection.config import config_factory
    cfg_eval = config_factory("detection_cvpr_2019")
    de = DetectionEval(nusc, config=cfg_eval, result_path=res_path, eval_set="val",
                       output_dir=args.out_dir, verbose=True)
    summary = de.main(render_curves=False)

    print("\n================ OFFICIAL nuScenes metrics ================")
    print(f"  NDS   = {summary['nd_score']:.4f}")
    print(f"  mAP   = {summary['mean_ap']:.4f}")
    print("  --- per-class AP (mean over 0.5/1/2/4 m) ---")
    for name, ap_ in summary["mean_dist_aps"].items():
        print(f"    {name:<22} {ap_:.4f}")
    print("  --- TP errors (lower better) ---")
    for k, v in summary["tp_errors"].items():
        print(f"    {k:<10} {v:.4f}")
    print(f"\n  (velocity/attribute not predicted -> mAVE/mAAE penalized; mAP is the clean headline)")


if __name__ == "__main__":
    main()
