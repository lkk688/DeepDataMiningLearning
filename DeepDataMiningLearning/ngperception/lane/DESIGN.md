# Lane detection — research directions (design)

Two directions, deliberately different in scope:

- **Direction 1** improves the **2-D front-camera** lane detector (CLRNet/CLRerNet family) with
  **temporal consistency** and **uncertainty** — a focused, publishable extension of the SOTA head.
- **Direction 2** moves to **3-D BEV lanes + topology** (OpenLane-V2) — which *converges with the
  occupancy/detection BEV encoder* (M3), i.e. a fourth head on the shared front-end.

Both respect the framework's rule: **pure PyTorch**. CLRNet/CLRerNet upstream are mmdet-based, so
Direction 1 begins by reimplementing the CLRNet *head* (line-anchor + LaneIoU) in plain torch — a
bounded, ~few-hundred-line job on top of a torchvision backbone + FPN.

---

## 0. What we're extending — CLRNet / CLRerNet in one screen

- **Representation:** a lane is a **line anchor** — a start point (x,y) + angle θ + a per-row set
  of x-offsets. So a lane = "for each image row, where is the line" (structured, not a mask).
- **Cross-Layer Refinement (CLR):** start coarse anchors on high-level FPN features, **refine
  across levels** to finer features; **ROIGather** samples features *along each lane line* +
  global attention.
- **Head:** per anchor → {cls (lane / not), start/θ/length, per-row Δx}.
- **Loss:** focal cls + **LineIoU** (a differentiable line-overlap IoU, CLRNet's key idea) +
  smooth-L1 offsets + aux segmentation.
- **CLRerNet (WACV'24):** replaces LineIoU with **LaneIoU** (accounts for local lane *tilt*), and
  uses it for both the loss *and* confidence/assignment → better-calibrated confidence, CULane
  F1 **81.4**.

The per-row-offset regression and the confidence are exactly the two places uncertainty plugs in;
the anchors are exactly what propagates across time.

---

## Direction 1 — Temporal + Uncertainty-aware CLRerNet

### 1a. Uncertainty-aware (the cheaper, do-first half)

**Aleatoric (data) uncertainty — per-point variance.** The offset head predicts, per row, a
**mean *and* a log-variance** `(Δx, log σ²)` instead of just `Δx`. Train the offsets with a
**Gaussian NLL** in place of smooth-L1:

```
L_reg = Σ_rows [ (Δx − Δx_gt)² / (2σ²) + ½ log σ² ]
```

This gives a **per-point confidence band** around each lane at ~zero extra cost (one extra output
channel + a loss swap). Wide σ where the lane is faint/occluded/far; tight where it's crisp.

**Existence/confidence uncertainty.** CLRerNet already couples confidence to LaneIoU; extend it to
a **calibrated existence probability** (temperature-scale on a held-out set, or an evidential head
predicting a Beta over "lane exists"). Downstream planning wants "is there a lane, how sure".

**Epistemic (optional):** MC-dropout at inference (K forward passes, dropout on) → variance across
passes. Cheap ensemble; only if the aleatoric head isn't enough.

*Why first:* it's a **head/loss change**, no new data, and it's independently useful — a lane
detector that says "I'm unsure here" is directly safer.

### 1b. Temporal (the higher-value, harder half)

Lanes are near-static in the world; across a video they should move only by **ego-motion**. Three
levels, in increasing power:

1. **Output EMA (baseline):** exponential smoothing of lane params across frames, ego-motion-warped.
   Learned nothing; a strong-ish baseline to beat.
2. **Feature-level temporal fusion:** warp the previous frame's BEV/feature map by ego-motion and
   fuse with the current (BEVFormer-style **temporal self-attention**, or a ConvGRU). Robust to
   momentary occlusion (a car covering a lane for 1 frame).
3. **Lane track-queries (the strong version):** give each lane a **persistent query** carried
   across frames (like MOTR/track-queries in MOT). At frame *t*, the confident lanes of *t−1*,
   warped by ego-motion, become **anchor priors** for *t*'s refinement. This yields temporal
   consistency **and** free **lane IDs / tracking** (which lane is which over time).

**The synergy (why 1a+1b belong together):** uncertainty **gates** the temporal fusion — a lane
carried from the past is trusted in proportion to its (low) variance, and temporal agreement in
turn *reduces* uncertainty. Concretely: fuse `x_t = w·x_{t-1,warp} + (1−w)·x_t^{det}` with
`w ∝ 1/σ²_{t-1}`. That's a clean, defensible contribution: *uncertainty-gated temporal lane
refinement.*

### 1c. Implementation path (pure torch)

1. **Reimplement the CLRNet head** on a torchvision ResNet/DLA + FPN: line-anchor generation,
   ROIGather (sample features along the line via `grid_sample` — we already wrote a
   `grid_sample`-based deformable op for the BEV transformer, §10.1 of detection), the offset/cls
   heads, and **LaneIoU** loss. (~300–500 lines; no mmdet.)
2. Add the **variance head + Gaussian NLL** (1a).
3. Add the **track-query buffer + ego-motion warp + uncertainty-gated fusion** (1b).
4. **Data:** CULane/TuSimple for the base detector (image F1). Temporal needs *video* — **VIL-100**
   (video-instance lanes) or **OpenLane** clips; nuScenes CAM_FRONT sequences give video but no
   lane GT (usable for *unsupervised* temporal-consistency loss or pseudo-labels from the base
   model).

**Effort/risk:** medium. The base head reimpl is the bulk; uncertainty is low-risk; temporal
track-queries are the research novelty (and the risky/valuable bit). All pure-torch.

---

## Direction 2 — OpenLane-V2 / 3-D lane graph / topology

This is a **task shift**, not just a head tweak: from 2-D image lane *lines* to **3-D BEV lane
centerlines + a lane graph + traffic-element assignment**. And it **converges with our BEV
encoder** — the natural M4→M5 extension of the full-stack model.

### 2a. The OpenLane-V2 task

- **3-D lane centerlines** (directed polylines in ego/BEV metres, not image pixels).
- **Traffic elements** (lights/signs) — a 2-D detection on the front image.
- **Topology:** lane-lane connectivity (which centerline follows which) **and** lane-to-traffic-
  element assignment (which light governs which lane).
- **Metric OLS** = combination of lane DET, TE DET, and **TOP** (topology precision on the graphs).
- SOTA lineage: **TopoNet**, **TopoMLP**, **LaneGAP**, **TopoLogic**.

### 2b. Architecture — and how it reuses what we built

This maps almost 1:1 onto pieces we already have:

```
surround cams ─┐
               ├─► shared BEV encoder (M3: LSS lift + LiDAR fusion → BEV) ──► lane-query decoder
LiDAR ─────────┘        (occupancy + detection already hang off this)          (DETR-style, deformable)
                                                                                     │
                            ┌────────────────────────────────────────────────────────┤
                            ▼                    ▼                     ▼               ▼
                    3-D lane centerlines   lane-lane topology    lane-TE topology   traffic elements
                    (polyline per query)   (pairwise MLP over     (pairwise MLP)     (2-D det on image)
                                            query embeddings)
```

- **Lane-query decoder = our BEV deformable-attention transformer** (detection/`bev_transformer.py`).
  We built object queries that cross-attend to BEV features with deformable sampling + iterative
  refinement — a **lane query** is the same thing, but decodes an **ordered set of points
  (a polyline)** instead of a box (LATR / TopoNet are exactly query-based lane decoders).
- **Topology head = a pairwise MLP** over the lane-query embeddings: for queries *i,j*, MLP(concat
  or difference of embeddings) → adjacency logit (does lane *i* connect to *j*). This is TopoNet's
  core; it's ~50 lines on top of the query embeddings we already produce.
- **Traffic elements** = a 2-D detection head on the front image (reuse detection paradigms); the
  **lane-TE assignment** is another pairwise MLP between lane queries and TE queries.
- **The shared encoder is M3's** — so 3-D lanes become the *fourth head* on the same BEV front-end
  as occupancy and detection. That is the full-stack payoff: **one BEV encoder → occupancy (dense)
  + detection (boxes) + 3-D lanes (polylines) + topology (graph)**.

### 2c. Implementation path

1. **3-D lane decoder:** reuse `bev_transformer.py`'s deformable decoder; change the box head to a
   **polyline head** (regress N ordered BEV points + a validity/class), matched to GT lanes with
   Hungarian (we already have the matcher) using a **chamfer/polyline distance** cost.
2. **Topology head:** pairwise MLP over lane-query embeddings → lane-lane adjacency; a second one
   for lane-TE. BCE loss on the adjacency matrices.
3. **Traffic-element head:** a small 2-D detector on the front image (or reuse an off-the-shelf).
4. **Data:** **OpenLane-V2** (built on nuScenes + Argoverse2) — we have nuScenes, so the nuScenes
   subset is stage-able. Metric OLS via the official devkit (numpy, no spconv).

**Effort/risk:** larger than Direction 1, but **high leverage** because it reuses the M3 shared
encoder + the deformable BEV transformer + the Hungarian matcher — so it's mostly *new heads on
existing machinery*, not a new stack. It's also the most novel: it makes the framework a genuine
**unified BEV perception model** (dense + object + lane + graph).

---

## Recommended sequencing

1. **L1 (now-ish):** pure-torch CLRNet head + LaneIoU on CULane → a real F1 number (validates the
   base before extending). Add **UFLDv2** as a second runnable paradigm for contrast.
2. **Direction 1a:** variance head + Gaussian NLL (cheap, independently useful, low risk).
3. **Direction 1b:** temporal track-queries + uncertainty-gated fusion (the 2-D research novelty).
4. **Direction 2:** 3-D lane + topology heads on the **M3 shared BEV encoder** (the full-stack,
   highest-leverage move — reuses the deformable BEV transformer + Hungarian matcher).

The through-line: everything is a **head on shared machinery** (backbone+FPN for 2-D, the M3 BEV
encoder + deformable transformer for 3-D), so each step is bounded and pure-torch, and Direction 2
completes the unified-BEV vision the occupancy/detection work started.
