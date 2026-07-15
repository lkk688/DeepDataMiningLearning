"""Compare DepthAnything-V2 Small vs Large for the segment-level metric depth target. For one
camera per sample, renders [image | Small metric depth | Large metric depth] side by side (LiDAR
points coloured by their own depth on both) so you can judge whether the Large backbone's better
shape is worth the extra compute for offline label generation. See TUTORIAL.md §24.

  python -m DeepDataMiningLearning.ngdet.labelgen.compare_depth --source nuscenes \
      --dataroot /data/.../nuScenes/v1.0-trainval --num 12 --cam-name FRONT --out ngdet/output/depth_cmp
"""
import argparse
import os
import numpy as np
import cv2
import imageio.v2 as imageio

from .labeler import GroundedLabeler, NUSC_TAXONOMY
from .sources import NuScenesSource, WaymoSource, KittiSource, ImageFolderSource
from .physicalai import PhysicalAISource
from .visualize import depth_colormap


def _build_source(args, hw):
    if args.source == "nuscenes":
        return NuScenesSource(args.dataroot, args.version, hw, start=args.start, num=args.num, stride=args.stride)
    if args.source == "waymo":
        return WaymoSource(args.root, hw, start=args.start, num=args.num, stride=args.stride)
    if args.source == "physicalai":
        return PhysicalAISource(args.root, hw, start=args.start, num=args.num, stride=args.stride)
    if args.source == "kitti":
        return KittiSource(args.root, image_hw=hw, start=args.start, num=args.num, stride=args.stride)
    return ImageFolderSource(args.folder, hw)


def _rel_depth(model, proc, pil, dev):
    import torch
    import torch.nn.functional as F
    W, H = pil.size
    inp = proc(images=pil, return_tensors="pt").to(dev)
    with torch.no_grad():
        d = model(**inp).predicted_depth
    return F.interpolate(d[None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy().astype(np.float32)


def _label(img, txt):
    img = np.array(img, np.uint8)                                # force a writable copy
    cv2.putText(img, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["nuscenes", "waymo", "kitti", "physicalai", "folder"], default="nuscenes")
    ap.add_argument("--dataroot", default="/data/rnd-liu/Datasets/nuScenes/v1.0-trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--root", default=""); ap.add_argument("--folder", default="")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--image-h", type=int, default=384); ap.add_argument("--image-w", type=int, default=640)
    ap.add_argument("--cam-name", default="FRONT", help="substring; picks that camera per sample")
    ap.add_argument("--small", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--large", default="depth-anything/Depth-Anything-V2-Large-hf")
    ap.add_argument("--out", default="ngdet/output/depth_cmp"); ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    hw = (args.image_h, args.image_w)
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    lab = GroundedLabeler(device=args.device, taxonomy=NUSC_TAXONOMY, depth_ckpt=args.small)
    lp = AutoImageProcessor.from_pretrained(args.large)
    lm = AutoModelForDepthEstimation.from_pretrained(args.large).to(args.device).eval()
    src = _build_source(args, hw)
    frames = []
    for i, (key, cams) in enumerate(src):
        pick = next((c for c in cams if args.cam_name.upper() in c[0].upper()), cams[0] if cams else None)
        if pick is None:
            continue
        name, pil, uvz = pick
        raw = np.asarray(pil)
        _, masks = lab.semantic(pil)
        d_small = lab.depth(pil, uvz, masks, drel=lab._depth_anything(pil).astype(np.float32))
        d_large = lab.depth(pil, uvz, masks, drel=_rel_depth(lm, lp, pil, args.device))
        panel = np.hstack([
            _label(raw, f"{key[:8]} {name}"),
            _label(depth_colormap(d_small, uvz, lab.max_depth), "Small"),
            _label(depth_colormap(d_large, uvz, lab.max_depth), "Large")])
        imageio.imwrite(os.path.join(args.out, f"cmp_{i:03d}.png"), panel)
        frames.append(panel)
        print(f"[cmp] {i+1} {key} {name}", flush=True)
    if frames:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from detection.verify_datasets_video import VideoWriter
        vw = VideoWriter(os.path.join(args.out, "depth_compare.mp4"), fps=args.fps)
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"[cmp] wrote depth_compare.mp4 ({len(frames)} frames) to {args.out}")


if __name__ == "__main__":
    main()
