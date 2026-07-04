# Lane detection & prediction — a tutorial

Lanes are the road's *structure*: where can I drive, which lane am I in, where do the markings
go. This module is the lane branch of `ngperception` — start by **running SOTA models** on road
images, then fine-tune, then push to **3-D lanes**. Same philosophy as the rest of the framework:
pure PyTorch first, understand before scaling.

> Companion modules: [depth](../depth/) (per-pixel geometry), [detection](../detection/) (objects
> as boxes), [occupancy](../occupancy/) (dense what-is-where). Lanes add the *road layout*.

---

## 1. The task — and its many shapes

"Lane detection" is really several tasks:

| task | output | example use |
|---|---|---|
| **lane-line detection** | the painted markings, as instances/curves | lane keeping, LDW |
| **ego-lane / lane assignment** | which lane the car is in | ACC, lane-change |
| **drivable area** | free road surface (mask) | path planning |
| **3-D lanes** | lanes in metric 3-D (not just image pixels) | planning, HD-map-free driving |
| **lane/road topology** | how lanes connect (graph) | routing, intersections |

The first three are **2-D** (image space); the last two are the modern frontier. Most classic
work is 2-D lane-line detection on a front camera.

## 2. Paradigms — how models represent a lane

A lane line is a thin, long, sparse curve — awkward for a plain segmentation net. The field has
five families:

1. **Segmentation + clustering** (SCNN, LaneNet, RESA): per-pixel "lane vs not", then cluster
   pixels into instances. Simple, but post-processing-heavy and slow.
2. **Row-wise classification** (UFLD, **UFLDv2**): for each image row, *classify which column* the
   lane crosses. Turns detection into a cheap classification → **very fast** (300+ FPS).
3. **Keypoint** (FOLOLane, GANet): predict lane keypoints + associations, like pose estimation.
4. **Curve / polynomial** (PolyLaneNet, **LSTR**, BezierLaneNet): regress a parametric curve
   (polynomial / Bézier) per lane. Compact, smooth, no clustering.
5. **Anchor-based line** (LaneATT, **CLRNet**, **CLRerNet**): line "anchors" (like object anchors
   but for lines) refined across feature levels. **Current SOTA** on CULane.

**3-D lanes** (OpenLane): **PersFormer**, **Anchor3DLane**, **LATR** — lift image lanes into a BEV
/ 3-D space (often with a transformer + camera geometry), so lanes have real depth and curvature.

## 3. Datasets

| dataset | scale | what | metric |
|---|---|---|---|
| **TuSimple** | 6.4 k highway clips | lane lines, easy | accuracy (points on GT) |
| **CULane** | 133 k frames, 9 scenarios | lane lines, hard (night, crowd, no-line) | **F1** (IoU-matched, 30 px) |
| **LLAMAS** | 100 k, auto-labeled | highway lanes | F1 |
| **CurveLanes** | 150 k, curvy | hard curves | F1 |
| **OpenLane** | 200 k, on Waymo | **3-D lanes** + topology | 3-D F1, X/Z error |
| **OpenLane-V2** | + topology | lane graph, traffic elements | OLS |

CULane F1 is the standard "how good" number; OpenLane is the 3-D frontier.

## 4. SOTA landscape (CULane test F1, higher = better)

| model | year | paradigm | CULane F1 | notes |
|---|---|---|---|---|
| SCNN | 2018 | seg | 71.6 | the classic |
| UFLD | 2020 | row | 68.4 | very fast |
| LaneATT | 2021 | anchor-line | 77.0 | |
| **UFLDv2** | 2022 | row | 76.0 | fast + strong, pure PyTorch |
| **CLRNet** | 2022 | anchor-line | 80.5 | cross-layer refinement |
| **CLRerNet** | 2023 | anchor-line | **81.4** | current SOTA |
| **YOLOPv2** | 2022 | multi-task seg | (panoptic) | lane **+ drivable + detection**, one net |

The top lane-F1 models (CLRNet/CLRerNet) are **mmdet-based** (heavy env, like the occupancy SOTA
in §2.5). The pure-PyTorch, easy-to-run strong models are **UFLDv2** (row-wise) and **YOLOPv2**
(panoptic multi-task) — the same "runs in the main torch env" trade-off as elsewhere.

## 5. What we ran first — YOLOPv2 (panoptic driving)

We started with **[YOLOPv2](https://github.com/CAIC-AD/YOLOPv2)** because it is a **self-contained
TorchScript** (no mmcv, no framework) that does **three tasks in one forward**: vehicle detection,
**drivable-area** segmentation, and **lane-line** segmentation. Ideal first "what does SOTA see"
demo on the surround images we already have.

Run it (`run_lane.py`) on nuScenes / KITTI front cameras — no lane labels needed for inference:

```bash
python -m DeepDataMiningLearning.ngperception.lane.run_lane \
    --model yolopv2 --dataset nuscenes --n 30 --video      # -> output/lane/{lane_*.png, lane_demo.mp4}
python -m DeepDataMiningLearning.ngperception.lane.run_lane --dataset kitti --n 10
```

The model outputs (verified): `da` (drivable, 2-ch probs → argmax) and `ll` (lane, **1-ch
probability** — threshold `ll > 0.4` directly, *not* a logit; that one detail is the difference
between 217 clean lane pixels and 145 k garbage ones). We overlay drivable area (green) and lane
lines (red) back onto the original image:

![lane demo](../docs/lane_demo_frame.png)

On nuScenes CAM_FRONT the lane mask is a thin ~200–550 px set of markings and the drivable area a
~50–70 k px road region — sensible panoptic output, real-time, pure PyTorch.

## 6. Roadmap

- **L1 — run more paradigms.** Add **UFLDv2** (row-wise, structured lane *instances* + curves, not
  just a mask) and note **CLRNet/CLRerNet** as the mmdet-based F1 leaders (separate env, like the
  occupancy GaussianOcc case).
- **L2 — proper eval.** Stage TuSimple/CULane, report accuracy / F1 against GT (right now we only
  do qualitative inference on unlabeled road images).
- **L3 — fine-tune** a lane model on a target domain (mirrors depth §1.7 / detection fine-tuning).
- **L4 — 3-D lanes** (OpenLane): PersFormer / LATR — lift lanes into BEV, the natural bridge to
  the occupancy & detection BEV features (a fourth head on the shared encoder, extending M3).
- **L5 — prediction/topology**: lane graph + trajectory prediction, toward planning.

See [README.md](README.md) for the module layout.
