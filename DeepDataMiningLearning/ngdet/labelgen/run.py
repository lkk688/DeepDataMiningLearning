"""CLI: generate dense 2-D labels (open-vocab semantic + segment-level metric depth) and write
human-eval visualizations. See ngdet/TUTORIAL.md §24.

  # nuScenes (6 cams + LiDAR metric depth), 20 keyframes, best-shape depth:
  python -m DeepDataMiningLearning.ngdet.labelgen.run --source nuscenes \
      --dataroot /data/.../nuScenes/v1.0-trainval --start 1000 --num 20 \
      --depth-ckpt depth-anything/Depth-Anything-V2-Large-hf --out ngdet/output/labelgen --video

  # any image folder (semantic + relative depth, no LiDAR):
  python -m DeepDataMiningLearning.ngdet.labelgen.run --source folder --folder path/to/imgs --out ... --video
"""
import argparse
import os
import numpy as np
import imageio.v2 as imageio

from .labeler import GroundedLabeler, NUSC_TAXONOMY
from .sources import NuScenesSource, ImageFolderSource, WaymoSource, KittiSource
from .physicalai import PhysicalAISource
from .av2 import AV2Source
from .visualize import sem_overlay, depth_colormap, tile, write_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["nuscenes", "waymo", "kitti", "physicalai", "av2", "folder"], default="nuscenes")
    ap.add_argument("--dataroot", default="/data/rnd-liu/Datasets/nuScenes/v1.0-trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--root", default="", help="dataset root for waymo (extracted) / kitti")
    ap.add_argument("--folder", default="")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--image-h", type=int, default=256); ap.add_argument("--image-w", type=int, default=704)
    ap.add_argument("--depth-ckpt", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--dino-ckpt", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--box-thresh", type=float, default=0.30); ap.add_argument("--text-thresh", type=float, default=0.25)
    ap.add_argument("--out", default="ngdet/output/labelgen")
    ap.add_argument("--save-npz", action="store_true", help="also cache sem+depth per sample (.npz)")
    ap.add_argument("--video", action="store_true"); ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    hw = (args.image_h, args.image_w)
    lab = GroundedLabeler(device=args.device, taxonomy=NUSC_TAXONOMY, depth_ckpt=args.depth_ckpt,
                          dino_ckpt=args.dino_ckpt, box_thresh=args.box_thresh, text_thresh=args.text_thresh)
    tax = NUSC_TAXONOMY
    if args.source == "nuscenes":
        src = NuScenesSource(args.dataroot, args.version, hw, start=args.start, num=args.num, stride=args.stride)
        cols = 3
    elif args.source == "waymo":
        src = WaymoSource(args.root, hw, start=args.start, num=args.num, stride=args.stride); cols = 3
    elif args.source == "physicalai":
        src = PhysicalAISource(args.root, hw, start=args.start, num=args.num, stride=args.stride); cols = 3
    elif args.source == "av2":
        src = AV2Source(args.root, hw, start=args.start, num=args.num, stride=args.stride); cols = 3
    elif args.source == "kitti":
        src = KittiSource(args.root, image_hw=hw, start=args.start, num=args.num, stride=args.stride); cols = 1
    else:
        src = ImageFolderSource(args.folder, hw); cols = 1
    sem_frames, dep_frames = [], []
    for i, (key, cams) in enumerate(src):
        sem_panels, dep_panels, names, sems, deps = [], [], [], [], []
        for name, pil, uvz in cams:
            raw = np.asarray(pil)
            out = lab.label(pil, uvz)
            sem_panels.append(sem_overlay(raw, out["sem"], tax.colors, tax.sky_id))
            dep_panels.append(depth_colormap(out["depth"], uvz, lab.max_depth))
            names.append(name); sems.append(out["sem"].astype(np.uint8)); deps.append(out["depth"].astype(np.float16))
        if not sem_panels:                                     # e.g. a LiDAR-only log (no cameras)
            print(f"[labelgen] {i+1} {key}: no camera frames, skipped", flush=True); continue
        sem_t = tile(sem_panels, names, cols); dep_t = tile(dep_panels, names, cols)
        imageio.imwrite(os.path.join(args.out, f"sem_{i:03d}.png"), sem_t)
        imageio.imwrite(os.path.join(args.out, f"dep_{i:03d}.png"), dep_t)
        sem_frames.append(sem_t); dep_frames.append(dep_t)
        if args.save_npz:
            np.savez_compressed(os.path.join(args.out, f"{key}.npz"),
                                sem=np.stack(sems), depth=np.stack(deps))
        print(f"[labelgen] {i+1}/{len(src)} {key}", flush=True)
    if args.video and sem_frames:
        write_video(os.path.join(args.out, "semantic.mp4"), sem_frames, args.fps)
        write_video(os.path.join(args.out, "depth.mp4"), dep_frames, args.fps)
        print(f"[labelgen] wrote semantic.mp4 + depth.mp4 to {args.out}")


if __name__ == "__main__":
    main()
