"""
ngperception.lane.run_lane
==========================

Run a **SOTA lane-detection model** on road images and render the result. First backend is
**YOLOPv2** (CAIC-AD, a self-contained TorchScript panoptic-driving model) — one forward gives
**lane lines + drivable area + vehicle boxes**, so it's an ideal first "what does SOTA see"
demo, pure PyTorch, no mmcv. Runs on the surround-camera images we already have (nuScenes
CAM_FRONT / KITTI image_2); no lane labels needed for inference.

    python -m DeepDataMiningLearning.ngperception.lane.run_lane \
        --model yolopv2 --dataset nuscenes --n 30 --video

Weights: YOLOPv2 TorchScript (~156 MB) at `--weights` (default the staged copy). Download once:
    curl -L -o yolopv2.pt https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
"""
from __future__ import annotations
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

DATASETS = {
    "nuscenes": "/mnt/e/Shared/Dataset/NuScenes/v1.0-trainval/samples/CAM_FRONT",
    "kitti": "/mnt/e/Shared/Dataset/Kitti/training/image_2",
}
DEFAULT_W = "/home/lkk688/Developer/occ3d_data/lane_models/yolopv2.pt"


def letterbox(img, new=(384, 640), pad=114):
    """Resize (keep aspect) + pad to `new` (H,W). Returns padded array + (scale, padx, pady)."""
    h0, w0 = img.shape[:2]
    r = min(new[0] / h0, new[1] / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    im = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    out = np.full((new[0], new[1], 3), pad, np.uint8)
    py, px = (new[0] - nh) // 2, (new[1] - nw) // 2
    out[py:py + nh, px:px + nw] = im
    return out, r, px, py


def overlay(img0, da_mask, ll_mask, r, px, py):
    """Paint drivable area (green) + lane lines (red) back onto the original image."""
    h0, w0 = img0.shape[:2]
    vis = img0.copy()
    for mask, color in [(da_mask, (0, 200, 0)), (ll_mask, (255, 40, 40))]:
        # mask is at letterbox res; crop the padded region and resize to original
        m = mask[py:py + int(round(h0 * r)), px:px + int(round(w0 * r))]
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w0, h0), Image.NEAREST)) > 127
        vis[m] = (0.5 * vis[m] + 0.5 * np.array(color)).astype(np.uint8)
    return vis


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolopv2", choices=["yolopv2"])
    ap.add_argument("--weights", default=DEFAULT_W)
    ap.add_argument("--dataset", default="nuscenes", choices=list(DATASETS))
    ap.add_argument("--images", default=None, help="glob overriding --dataset")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--hw", type=int, nargs=2, default=[384, 640])
    ap.add_argument("--ll-thresh", type=float, default=0.4, help="lane-line probability threshold")
    ap.add_argument("--video", action="store_true", help="also write an mp4 of the sequence")
    ap.add_argument("--out-dir", default="DeepDataMiningLearning/ngperception/output/lane")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    paths = sorted(glob.glob(args.images)) if args.images else \
        sorted(glob.glob(os.path.join(DATASETS[args.dataset], "*")))
    paths = paths[:args.n]
    os.makedirs(args.out_dir, exist_ok=True)
    model = torch.jit.load(args.weights, map_location=dev).eval()
    print(f"[lane] {args.model} on {len(paths)} {args.dataset} images -> {args.out_dir}", flush=True)

    frames = []
    for i, p in enumerate(paths):
        img0 = np.asarray(Image.open(p).convert("RGB"))
        lb, r, px, py = letterbox(img0, tuple(args.hw))
        inp = torch.from_numpy(lb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        det, da, ll = model(inp)
        da_mask = da.argmax(1)[0].cpu().numpy().astype(bool)          # drivable area (2-ch probs)
        ll_mask = (ll[0, 0] > args.ll_thresh).cpu().numpy()          # lane lines (ll is already a prob)
        vis = overlay(img0, da_mask, ll_mask, r, px, py)
        Image.fromarray(vis).save(os.path.join(args.out_dir, f"lane_{i:04d}.png"))
        if args.video:
            frames.append(vis)
        if i % 10 == 0:
            print(f"  {i+1}/{len(paths)} lane px={int(ll_mask.sum())} drivable px={int(da_mask.sum())}", flush=True)

    if args.video and frames:
        import imageio.v2 as imageio
        h, w = frames[0].shape[:2]
        frames = [f if f.shape[:2] == (h, w) else np.asarray(Image.fromarray(f).resize((w, h))) for f in frames]
        imageio.mimsave(os.path.join(args.out_dir, "lane_demo.mp4"), frames, fps=6,
                        codec="libx264", pixelformat="yuv420p")
        print(f"[lane] wrote {args.out_dir}/lane_demo.mp4", flush=True)
    print("[lane] done", flush=True)


if __name__ == "__main__":
    main()
