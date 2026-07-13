"""
ngperception.occupancy.visualize
=================================

Inference + **BEV semantic visualization** of the LSS occupancy model — renders the predicted
200x200x16 voxel grid as a top-down colored semantic map (matplotlib; robust, no open3d/EGL).
Supports side-by-side comparison of several checkpoints/modalities against the Occ3D GT.

    # single checkpoint vs GT
    python -m DeepDataMiningLearning.ngperception.occupancy.visualize \
        --gts $GTS --nusc $NUSC --sample-idx 0 --out $OUT/viz/occ.png \
        --ckpt $OUT/lss_occ_full_fusion/lss_occ.pth --lidar-fusion \
        --backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --refine-iters 2

    # camera-only vs fusion vs GT (comparison figure)
    python -m ...occupancy.visualize --gts $GTS --nusc $NUSC --sample-idx 0 \
        --compare camera:$OUT/lss_occ_full/lss_occ.pth fusion:$OUT/lss_occ_full_fusion/lss_occ.pth \
        --out $OUT/viz/occ_compare.png
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

# Occ3D-nuScenes 17-class BEV palette (RGB 0-1); index 17 = free (white)
OCC3D_PALETTE = np.array([
    [0, 0, 0], [255, 120, 50], [255, 192, 203], [255, 255, 0], [0, 150, 245],
    [0, 255, 255], [255, 127, 0], [255, 0, 0], [255, 240, 150], [135, 60, 0],
    [160, 32, 240], [255, 0, 255], [139, 137, 137], [75, 0, 75], [150, 240, 80],
    [230, 230, 250], [0, 175, 0], [255, 255, 255],
], np.float32) / 255.0
CLASS_NAMES = ["others", "barrier", "bicycle", "bus", "car", "constr_veh", "motorcycle",
               "pedestrian", "traffic_cone", "trailer", "truck", "driveable", "other_flat",
               "sidewalk", "terrain", "manmade", "vegetation", "free"]


def bev_semantic(sem_grid):
    """(200,200,16) class grid -> (200,200) top-down class map: the topmost occupied voxel."""
    occupied = sem_grid != 17                                  # 17 = free
    top = np.where(occupied.any(2), (occupied.shape[2] - 1) -
                   occupied[:, :, ::-1].argmax(2), -1)         # highest occupied z per (x,y)
    xs, ys = np.meshgrid(np.arange(200), np.arange(200), indexing="ij")
    out = np.full((200, 200), 17, np.int64)
    m = top >= 0
    out[m] = sem_grid[xs[m], ys[m], top[m]]
    return out


def render_panel(ax, sem_grid, title):
    bev = bev_semantic(sem_grid)
    ax.imshow(OCC3D_PALETTE[bev].transpose(1, 0, 2)[::-1], origin="upper")
    ax.set_title(title, fontsize=11); ax.axis("off")


def build_model(ckpt, args, lidar_fusion, dev):
    from .models.lss_occ import LSSOccupancy
    m = LSSOccupancy(backbone=args.backbone, decoder_hidden=args.decoder_hidden,
                     decoder_layers=args.decoder_layers, refine_iters=args.refine_iters,
                     lidar_fusion=lidar_fusion).to(dev)
    sd = torch.load(ckpt, map_location=dev)
    m.load_state_dict(sd.get("state_dict", sd), strict=False); m.eval()
    return m


@torch.no_grad()
def infer(model, ds, i, dev, drop_camera=False, drop_lidar=False):
    from torch.utils.data import default_collate
    b = default_collate([ds[i]])
    lv = b["lidar_vox"].to(dev) if "lidar_vox" in b else None
    occ = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev),
                b["intrins"].to(dev), lidar_vox=lv,
                drop_camera=drop_camera, drop_lidar=drop_lidar)[0]
    return occ.argmax(1)[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="BEV semantic visualization of LSS occupancy.")
    ap.add_argument("--gts", required=True); ap.add_argument("--nusc", required=True)
    ap.add_argument("--ckpt", default=None); ap.add_argument("--lidar-fusion", action="store_true")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="name:ckpt entries; fusion ckpts auto-detected by name containing 'fusion'/'lidar'")
    ap.add_argument("--modality-compare", action="store_true",
                    help="render ONE --ckpt (a modality-robust/fusion model) under fusion/camera/lidar")
    ap.add_argument("--backbone", default="dinov2_base")
    ap.add_argument("--decoder-layers", type=int, default=4); ap.add_argument("--decoder-hidden", type=int, default=96)
    ap.add_argument("--refine-iters", type=int, default=2)
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import os
    from nuscenes import NuScenes
    from .datasets_train import NuScenesOccTrainDataset

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    # a fusion-capable dataset (lidar_vox always provided) so any modality checkpoint / drop runs
    if args.modality_compare:                          # ONE fusion model under 3 modalities
        m0 = build_model(args.ckpt, args, True, dev)
        ds = NuScenesOccTrainDataset(args.gts, nusc, image_hw=m0.image_hw, downsample=m0.downsample,
                                     max_samples=args.sample_idx + 1, lidar_fusion=True)
        mods = [("fusion", False, False), ("camera-only", False, True), ("lidar-only", True, False)]
        panels = [(name, infer(m0, ds, args.sample_idx, dev, dc, dl)) for name, dc, dl in mods]
    else:
        entries = ([("prediction", args.ckpt, args.lidar_fusion)] if not args.compare else
                   [(e.split(":", 1)[0], e.split(":", 1)[1],
                     "fusion" in e.lower() or "lidar" in e.lower()) for e in args.compare])
        m0 = build_model(entries[0][1], args, entries[0][2], dev)
        ds = NuScenesOccTrainDataset(args.gts, nusc, image_hw=m0.image_hw, downsample=m0.downsample,
                                     max_samples=args.sample_idx + 1, lidar_fusion=True)
        panels = []
        for name, ckpt, fus in entries:
            m = m0 if ckpt == entries[0][1] else build_model(ckpt, args, fus, dev)
            panels.append((name, infer(m, ds, args.sample_idx, dev)))
    panels.append(("Occ3D GT", ds.occ[args.sample_idx].semantics))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (name, grid) in zip(axes, panels):
        render_panel(ax, grid, name)
    # compact legend of the common classes
    import matplotlib.patches as mp
    keep = [1, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 16]
    fig.legend([mp.Patch(color=OCC3D_PALETTE[c]) for c in keep],
               [CLASS_NAMES[c] for c in keep], loc="lower center", ncol=7, fontsize=7, frameon=False)
    fig.suptitle(f"Occupancy (BEV top-down semantics) — sample {args.sample_idx}", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight"); print(f"[viz] wrote {args.out}")


if __name__ == "__main__":
    main()
