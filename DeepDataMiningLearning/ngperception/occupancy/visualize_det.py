"""
ngperception.occupancy.visualize_det
=====================================

Inference + **BEV detection visualization** for the occupancy-backbone detector — renders
predicted 3-D boxes (colored by class, top-down) over the LiDAR occupancy, with the GT boxes as
dashed outlines. matplotlib (robust). Supports side-by-side comparison of two checkpoints (e.g.
anchor-vs-center head, or scratch-vs-finetune backbone).

    # single checkpoint (its cfg picks anchor/center head + modality)
    python -m DeepDataMiningLearning.ngperception.occupancy.visualize_det \
        --gts $GTS --nusc $NUSC --sample-idx 0 --out $OUT/viz/det.png \
        --ckpt $OUT/abl_dcenter_fusion/det_abl.pth

    # comparison: anchor head vs center head
    python -m ...occupancy.visualize_det --gts $GTS --nusc $NUSC --sample-idx 0 \
        --compare "anchor:$OUT/abl_finetune_fusion_sub/det_abl.pth" \
                  "center:$OUT/abl_dcenter_fusion/det_abl.pth" --out $OUT/viz/det_compare.png
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

DET_NAMES = ["car", "truck", "constr", "bus", "trailer", "barrier", "motorcycle",
             "bicycle", "pedestrian", "traffic_cone"]
DET_COLORS = np.array([[0, 150, 245], [0, 255, 255], [255, 127, 0], [255, 120, 50],
                       [135, 60, 0], [255, 0, 0], [255, 0, 255], [255, 192, 203],
                       [255, 240, 0], [160, 32, 240]], np.float32) / 255.0


def box_corners_bev(b):
    """[x,y,z,dx,dy,dz,yaw] -> 4 (x,y) corners."""
    x, y, _, dx, dy, _, yaw = b[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    xc = np.array([dx, dx, -dx, -dx]) / 2
    yc = np.array([dy, -dy, -dy, dy]) / 2
    return np.stack([x + xc * c - yc * s, y + xc * s + yc * c], 1)


def build_model(ck, dev):
    from .models.lss_occ import LSSOccupancy
    cfg = ck["cfg"]
    m = LSSOccupancy(backbone=cfg["backbone"], decoder_hidden=cfg["decoder_hidden"],
                     decoder_layers=cfg["decoder_layers"], refine_iters=cfg["refine_iters"],
                     lidar_fusion=cfg["lidar_fusion"], lidar_only=cfg.get("lidar_only", False),
                     det_classes=10, det_anchor_sizes=cfg["det_anchor_sizes"],
                     det_anchor_bottom=cfg["det_anchor_bottom"],
                     det_head_type=cfg.get("det_head_type", "anchor")).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


@torch.no_grad()
def infer_boxes(model, ds, i, dev, score_thresh=0.3, drop_camera=False, drop_lidar=False):
    from torch.utils.data import default_collate
    b = default_collate([ds[i]])
    lv = b["lidar_vox"].to(dev)
    _, _, aux = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev),
                      b["intrins"].to(dev), lidar_vox=lv,
                      drop_camera=drop_camera, drop_lidar=drop_lidar)
    d = model.det_head.predict(aux["det"], score_thresh=score_thresh, nms_thresh=0.2)[0]
    return (d["boxes"].cpu().numpy(), d["scores"].cpu().numpy(), d["labels"].cpu().numpy(),
            b["lidar_vox"][0].numpy())


def draw(ax, boxes, labels, gt, lidar_vox, title):
    import matplotlib.patches as mp
    occ = lidar_vox[0].sum(2) > 0                                # BEV LiDAR occupancy (x,y)
    ax.imshow(np.where(occ, 0.75, 1.0).T[::-1], cmap="gray", vmin=0, vmax=1,
              extent=[-40, 40, -40, 40])
    for g in gt:                                                 # GT: black dashed
        c = np.vstack([box_corners_bev(g), box_corners_bev(g)[0]])
        ax.plot(c[:, 0], c[:, 1], "k--", lw=1.0, alpha=0.7)
    for b, l in zip(boxes, labels):                             # pred: colored solid
        c = np.vstack([box_corners_bev(b), box_corners_bev(b)[0]])
        ax.plot(c[:, 0], c[:, 1], "-", color=DET_COLORS[int(l)], lw=1.4)
    ax.set_xlim(-40, 40); ax.set_ylim(-40, 40); ax.set_aspect("equal")
    ax.set_title(f"{title}  ({len(boxes)} det, {len(gt)} GT)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description="BEV detection visualization (occ-backbone detector).")
    ap.add_argument("--gts", required=True); ap.add_argument("--nusc", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--compare", nargs="+", default=None, help="name:ckpt entries (2 for side-by-side)")
    ap.add_argument("--modality-compare", action="store_true",
                    help="render ONE --ckpt under fusion/camera-only/lidar-only (modality-robust model)")
    ap.add_argument("--sample-idx", type=int, default=0); ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    import os
    from nuscenes import NuScenes
    from .datasets_train import NuScenesOccTrainDataset
    from ..detection.nuscenes_dataset import NUSC_10CLASS

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    entries = ([("prediction", args.ckpt)] if not args.compare else
               [(e.split(":", 1)[0], e.split(":", 1)[1]) for e in args.compare])
    m0 = build_model(torch.load(entries[0][1], map_location=dev), dev)
    ds = NuScenesOccTrainDataset(args.gts, nusc, image_hw=m0.image_hw, downsample=m0.downsample,
                                 max_samples=args.sample_idx + 1, lidar_fusion=True,
                                 det_boxes=True, det_class_map=NUSC_10CLASS)
    gt = ds[args.sample_idx]["det_gt"].numpy()

    if args.modality_compare:                          # ONE model under 3 modalities
        mods = [("fusion", False, False), ("camera-only", False, True), ("lidar-only", True, False)]
        renders = []
        for name, dc, dl in mods:
            boxes, _, labels, lv = infer_boxes(m0, ds, args.sample_idx, dev, args.score_thresh, dc, dl)
            renders.append((name, boxes, labels, lv))
    else:
        renders = []
        for name, ckpt in entries:
            m = m0 if ckpt == entries[0][1] else build_model(torch.load(ckpt, map_location=dev), dev)
            boxes, _, labels, lv = infer_boxes(m, ds, args.sample_idx, dev, args.score_thresh)
            renders.append((name, boxes, labels, lv))

    fig, axes = plt.subplots(1, len(renders), figsize=(5 * len(renders), 5))
    axes = np.atleast_1d(axes)
    for ax, (name, boxes, labels, lv) in zip(axes, renders):
        draw(ax, boxes, labels, gt, lv, name)
    fig.legend([mp.Patch(color=DET_COLORS[c]) for c in range(10)] + [plt.Line2D([], [], ls="--", c="k")],
               DET_NAMES + ["GT"], loc="lower center", ncol=6, fontsize=7, frameon=False)
    fig.suptitle(f"3-D detection (BEV, ego frame) — sample {args.sample_idx}", fontsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight"); print(f"[viz] wrote {args.out}")


if __name__ == "__main__":
    main()
