"""
ngperception.occupancy.viz_depth_arms
=====================================
Figures for the depth-FM / point-FM arms (PAPER_DRAFT §4.7).

1. `ldcm_depth_completion.png` — the depth-completion ladder on one nuScenes camera:
   RGB | sparse LiDAR input (8 rings) | MoGe alone | LDCM@8 | LDCM@32.
   This *is* the finding, visually: MoGe alone has plausible structure but no metric scale;
   the metric surface only appears once the Poisson step ingests the sparse returns.
2. `arms_beam_ladder.png` — occ mIoU vs how many LiDAR rings the depth prior ingests,
   against the camera baseline and the explicit-fusion reference.

    python -m DeepDataMiningLearning.ngperception.occupancy.viz_depth_arms \
        --gts <gts> --nusc <nusc> --out <docs_dir>
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np


def _colorize(d, vmin, vmax, cmap):
    v = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgb = (cmap(v)[..., :3] * 255).astype(np.uint8)
    rgb[d <= 0] = 30                                   # missing = dark
    return rgb


def _dilate(mask_img, k=3):
    """Fatten sparse points so they are visible in a figure."""
    out = mask_img.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out = np.maximum(out, np.roll(np.roll(mask_img, dy, 0), dx, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gts", required=True); ap.add_argument("--nusc", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--index", type=int, default=12, help="which occ sample to render")
    ap.add_argument("--cam", default="CAM_FRONT")
    ap.add_argument("--H", type=int, default=252); ap.add_argument("--W", type=int, default=700)
    ap.add_argument("--ldcm-path", default="/home/lkk688/Developer/DeepDataMiningLearning/Others/LDCM")
    args = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from PIL import Image
    import torch
    from nuscenes import NuScenes
    from pyquaternion import Quaternion as Q
    from .datasets import Occ3DNuScenesDataset
    from .cache_ldcm_depth import sparse_lidar_depth

    cmap = colormaps["turbo"]
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    occ = Occ3DNuScenesDataset(args.gts, scenes=None)
    tok = occ[args.index].sample_token
    sample = nusc.get("sample", tok)
    sd = nusc.get("sample_data", sample["data"][args.cam])
    img = np.asarray(Image.open(os.path.join(nusc.dataroot, sd["filename"]))
                     .convert("RGB").resize((args.W, args.H)), np.float32) / 255.0

    pr8 = sparse_lidar_depth(nusc, sample, sd, Q, args.H, args.W, beams=8)
    pr32 = sparse_lidar_depth(nusc, sample, sd, Q, args.H, args.W, beams=0)

    sys.path.insert(0, args.ldcm_path)
    from ldcm import LDCMModel
    model = LDCMModel.from_pretrained("pkqbajng/LDCM",
                                      moge_path="Ruicheng/moge-2-vits-normal").cuda().eval()
    x = torch.from_numpy(img).permute(2, 0, 1)[None].cuda()
    outs = {}
    for name, prior in (("8", pr8), ("32", pr32)):
        p = torch.from_numpy(prior)[None, None].cuda()
        with torch.inference_mode():
            o = model.infer(x, p)
        outs[name] = {k: o[k][0, 0].float().cpu().numpy() if o[k].dim() == 4
                      else o[k][0].float().cpu().numpy() for k in ("depth_pred", "mono_depth")}

    mono = outs["32"]["mono_depth"]
    # MoGe is up-to-scale: median-align purely so the *shape* is visible on the same colourbar.
    mono_show = mono * (np.median(pr32[pr32 > 0]) / max(np.median(mono), 1e-6))
    vmin, vmax = 2.0, 58.0

    panels = [
        ("(a) RGB", (img * 255).astype(np.uint8), None),
        ("(b) sparse LiDAR input — 8 rings", _dilate(_colorize(pr8, vmin, vmax, cmap)), None),
        ("(c) MoGe alone — NO sparse depth (up-to-scale)",
         _colorize(mono_show, vmin, vmax, cmap), None),
        ("(d) LDCM @ 8 rings", _colorize(outs["8"]["depth_pred"], vmin, vmax, cmap), None),
        ("(e) LDCM @ 32 rings", _colorize(outs["32"]["depth_pred"], vmin, vmax, cmap), None),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.5 * len(panels)))
    for ax, (title, im, _) in zip(axes, panels):
        ax.imshow(im); ax.set_title(title, fontsize=10, loc="left", pad=6); ax.axis("off")
    fig.suptitle("LDCM depth completion — the metric surface appears only once the sparse LiDAR is ingested",
                 fontsize=11, y=0.998)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    fig.subplots_adjust(top=0.955, hspace=0.28, left=0.02, right=0.86)
    fig.colorbar(sm, ax=axes.tolist(), shrink=0.42, pad=0.02, label="depth (m)")
    f1 = os.path.join(args.out, "ldcm_depth_completion.png")
    fig.savefig(f1, dpi=130); plt.close(fig)
    print("wrote", f1)

    # ---- beam ladder ----
    rings = [0, 8, 32]
    miou = [0.173, 0.243, 0.253]                       # b3 / b2 / b1  (PAPER_DRAFT §4.7)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(range(len(rings)), miou, "o-", lw=2.2, ms=9, color="#1f77b4",
            label="camera lift + LDCM depth prior")
    for i, (r, m) in enumerate(zip(rings, miou)):
        ax.annotate(f"{m:.3f}", (i, m), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9)
    ax.axhline(0.183, ls="--", c="#888", lw=1.6, label="b0 camera baseline (no prior) 0.183")
    ax.axhline(0.299, ls=":", c="#2ca02c", lw=1.8, label="a0 explicit 32-beam fusion 0.299")
    ax.set_xticks(range(len(rings))); ax.set_xticklabels([f"{r} rings" for r in rings])
    ax.set_xlabel("LiDAR rings the depth prior ingests")
    ax.set_ylabel("occ mIoU (val)")
    ax.set_title("The depth-FM gain tracks the ingested LiDAR — not the model\n"
                 "at 0 rings it reproduces the VGGT null", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    f2 = os.path.join(args.out, "arms_beam_ladder.png")
    fig.savefig(f2, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", f2)


if __name__ == "__main__":
    main()
