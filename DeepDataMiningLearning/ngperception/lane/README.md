# ngperception/lane — lane detection & prediction

The **road-layout** branch of `ngperception`: lane lines, drivable area, ego-lane, and (roadmap)
3-D lanes + topology. Same philosophy as the rest — **run SOTA first, pure PyTorch, understand
before scaling**. Full walkthrough (paradigms, SOTA landscape, datasets, metrics): **[TUTORIAL.md](TUTORIAL.md)**.

## Status

1. **YOLOPv2** (panoptic driving: lane lines + drivable area + vehicle detection, a self-contained
   TorchScript, no mmcv) running on real road images (nuScenes CAM_FRONT / KITTI image_2) —
   qualitative inference, real-time, pure PyTorch.
2. **Pure-torch CLRNet** (`clrnet.py`) — the line-anchor SOTA representation + **LineIoU/LaneIoU**
   (`lane_iou.py`) reimplemented **without mmdet**: ResNet+FPN → line-anchor priors → ROIGather →
   SimOTA assignment → decode/NMS, with an official-style CULane **F1** (`culane_metric.py`).
   Validated end-to-end on a synthetic sanity set (overfit F1 0.857, iou-loss 0.88→0.02); drop-in
   for real CULane on the H100. See **TUTORIAL §6**.

Roadmap in TUTORIAL §7 (UFLDv2 row-wise lanes, real CULane/TuSimple eval, fine-tuning, 3-D lanes on
OpenLane). Two research directions (temporal+uncertainty, 3-D lane graph) in [DESIGN.md](DESIGN.md).

## Run

```bash
# download YOLOPv2 weights once (~156 MB):
curl -L -o /home/lkk688/Developer/occ3d_data/lane_models/yolopv2.pt \
    https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt

# run on road images -> overlays + an mp4:
python -m DeepDataMiningLearning.ngperception.lane.run_lane --model yolopv2 --dataset nuscenes --n 30 --video
python -m DeepDataMiningLearning.ngperception.lane.run_lane --dataset kitti --n 10
```

Outputs land in `output/lane/` (`lane_*.png` overlays, `lane_demo.mp4`). `--ll-thresh` sets the
lane-line probability threshold (default 0.4), `--images '<glob>'` overrides the dataset.

## Layout

```
lane/
├── run_lane.py         # SOTA lane inference (YOLOPv2 backend) + overlay/video rendering
├── clrnet.py           # pure-torch CLRNet: backbone+FPN, line-anchor priors, ROIGather, heads, decode
├── lane_iou.py         # LineIoU (CLRNet) + LaneIoU (CLRerNet) loss/cost — pure torch
├── culane_dataset.py   # CULane .lines.txt loader + synthetic sanity generator (same interface)
├── culane_metric.py    # official-style IoU-matched CULane F1 (pure numpy)
├── train_clrnet.py     # trainer/eval: overfit-sanity, synthetic, and --dataset culane (H100)
├── DESIGN.md           # two research directions (temporal+uncertainty; 3-D lane graph/topology)
└── TUTORIAL.md         # the teaching doc: task shapes, paradigms, SOTA, datasets, metrics, roadmap
```

## Model notes

- **YOLOPv2** output: `da` (drivable, 2-ch probabilities → argmax), `ll` (lane, **1-ch
  probability** — threshold directly, it is *not* a logit). Detection head (`out[0]`) is available
  but not yet rendered.
- SOTA lane-F1 leaders (**CLRNet / CLRerNet**, CULane F1 ~80–81) are mmdet-based → a separate env,
  like the occupancy SOTA (TUTORIAL §5). The pure-PyTorch strong options are **UFLDv2** (row-wise)
  and **YOLOPv2** (panoptic).
