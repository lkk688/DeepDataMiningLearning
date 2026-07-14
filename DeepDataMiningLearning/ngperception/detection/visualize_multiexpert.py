"""
ngperception.detection.visualize_multiexpert
============================================

BEV visualization of the **Path B multi-expert** predictions — the SAME nuScenes scene detected by
each routed expert (PETR camera / BEVFusion-L LiDAR / BEVFusion-LC fusion), boxes colored by class,
GT as black dashes. Reads the experts' nuScenes results JSONs (dump with `tools/test.py
--cfg-options test_evaluator.jsonfile_prefix=... test_evaluator.format_only=True`).

    python -m DeepDataMiningLearning.ngperception.detection.visualize_multiexpert \
        --nusc <nuscenes> --out $OUT/viz/multiexpert.png \
        --preds "PETR (cam 0.383):.../petr_viz/preds/pred_instances_3d/results_nusc.json" \
                "BEVFusion-L (lidar 0.643):.../bevl_viz/.../results_nusc.json" \
                "BEVFusion-LC (fused 0.684):.../bevlc_viz/.../results_nusc.json"
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from ..occupancy.visualize_det import box_corners_bev, DET_NAMES, DET_COLORS

NAME2IDX = {n: i for i, n in enumerate(
    ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
     "motorcycle", "bicycle", "pedestrian", "traffic_cone"])}


def to_ego(nusc, token, entries, score_thresh):
    """nuScenes global-frame result dicts -> (N,7) ego boxes + labels, filtered by score."""
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    sd = nusc.get("sample_data", nusc.get("sample", token)["data"]["LIDAR_TOP"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    eq = Quaternion(pose["rotation"]); et = np.array(pose["translation"])
    boxes, labels = [], []
    for p in entries:
        if p.get("detection_score", 1.0) < score_thresh:
            continue
        cls = NAME2IDX.get(p["detection_name"])
        if cls is None:
            continue
        b = Box(p["translation"], p["size"], Quaternion(p["rotation"]))
        b.translate(-et); b.rotate(eq.inverse)                       # global -> ego
        x, y, z = b.center; w, l, h = b.wlh
        if not (-40 <= x < 40 and -40 <= y < 40):
            continue
        R = b.orientation.rotation_matrix
        boxes.append([x, y, z, l, w, h, float(np.arctan2(R[1, 0], R[0, 0]))]); labels.append(cls)
    return np.array(boxes, np.float32).reshape(-1, 7), np.array(labels, np.int64)


def gt_ego(nusc, token):
    from pyquaternion import Quaternion
    sd = nusc.get("sample_data", nusc.get("sample", token)["data"]["LIDAR_TOP"])
    _, boxes, _ = nusc.get_sample_data(sd["token"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    Rl = Quaternion(cs["rotation"]); tl = np.array(cs["translation"])
    out = []
    for b in boxes:
        if not b.name.startswith(("vehicle", "human", "movable")):
            continue
        b.rotate(Rl); b.translate(tl)                                # lidar -> ego
        x, y, z = b.center; w, l, h = b.wlh
        if -40 <= x < 40 and -40 <= y < 40:
            v = b.orientation.rotation_matrix[:, 0]
            out.append([x, y, z, l, w, h, float(np.arctan2(v[1], v[0]))])
    return np.array(out, np.float32).reshape(-1, 7)


def draw(ax, boxes, labels, gt, title):
    for g in gt:
        c = np.vstack([box_corners_bev(np.r_[g, 0]), box_corners_bev(np.r_[g, 0])[0]])
        ax.plot(c[:, 0], c[:, 1], "k--", lw=1.0, alpha=0.6)
    for b, l in zip(boxes, labels):
        c = np.vstack([box_corners_bev(np.r_[b, 0]), box_corners_bev(np.r_[b, 0])[0]])
        ax.plot(c[:, 0], c[:, 1], "-", color=DET_COLORS[int(l)], lw=1.5)
    ax.plot(0, 0, "k^", ms=8)                                        # ego
    ax.set_xlim(-40, 40); ax.set_ylim(-40, 40); ax.set_aspect("equal")
    ax.set_title(f"{title}  ({len(boxes)} det / {len(gt)} GT)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description="BEV visualization of Path B multi-expert predictions.")
    ap.add_argument("--nusc", required=True)
    ap.add_argument("--preds", nargs="+", required=True, help="name:results_nusc.json entries")
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--token", default=None, help="specific sample_token; else auto-pick a busy one")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    import os
    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)

    entries = [(e.split(":", 1)[0], json.load(open(e.split(":", 1)[1]))["results"]) for e in args.preds]
    common = set.intersection(*[set(r.keys()) for _, r in entries])
    token = args.token or max(common, key=lambda t: len(entries[0][1][t]))   # busiest scene by expert 0
    gt = gt_ego(nusc, token)

    fig, axes = plt.subplots(1, len(entries), figsize=(5 * len(entries), 5))
    axes = np.atleast_1d(axes)
    for ax, (name, res) in zip(axes, entries):
        b, l = to_ego(nusc, token, res.get(token, []), args.score_thresh)
        draw(ax, b, l, gt, name)
    fig.legend([mp.Patch(color=DET_COLORS[c]) for c in range(10)] + [plt.Line2D([], [], ls="--", c="k")],
               DET_NAMES + ["GT"], loc="lower center", ncol=6, fontsize=7, frameon=False)
    fig.suptitle(f"Path B multi-expert detection (BEV, ego) — token {token[:10]}", fontsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight"); print(f"[viz] wrote {args.out}")


if __name__ == "__main__":
    main()
