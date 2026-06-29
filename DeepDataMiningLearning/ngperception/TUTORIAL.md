# ngperception Tutorial — downstream perception, from basics to SOTA

This tutorial extends the ngdet detection tutorial to the *downstream* perception tasks
that consume an image (and often the detector's boxes): **depth/distance**, then
segmentation, tracking, and lane detection. Same recipe each time: understand the task,
compare models **basic → SOTA by inference**, then **fine-tune**.

---

## 1. Monocular depth estimation & per-object distance

### 1.1 The task

Given one RGB image, predict a **depth** for every pixel. Two sub-flavors:

- **Relative depth** — the *ordering* of depths is correct but the global scale (and
  often a shift) is unknown. Models trained on huge mixed data (MiDaS, Depth-Anything)
  are relative: they generalize anywhere but can't tell you metres.
- **Metric depth** — absolute depth in **metres**. Needed for distance ("18 m away"),
  3D lifting, planning. Models are trained/fine-tuned per camera-domain (ZoeDepth-KITTI,
  Depth-Anything-V2-Metric-Outdoor).

Why does scale go missing? A single image is **scale-ambiguous** — a toy car up close and
a real car far away project to the same pixels. Metric models break the tie using learned
priors about object sizes and the specific camera; that is why they are domain-specific.

### 1.2 The model ladder (what we compare)

| era | model | type | idea |
|---|---|---|---|
| 2019 baseline | **MiDaS / DPT** (`Intel/dpt-hybrid-midas`, `Intel/dpt-large`) | relative | ViT/CNN hybrid trained on 10+ mixed datasets; robust relative depth |
| 2024 SOTA (rel) | **Depth-Anything-V2** (`-Small/-Base/-Large-hf`) | relative | DINOv2 encoder + 62M pseudo-labelled images; sharp, fast |
| metric | **ZoeDepth** (`Intel/zoedepth-kitti`) | metric | MiDaS backbone + a metric "bins" head fine-tuned on KITTI |
| metric (SOTA) | **Depth-Anything-V2-Metric** (`...-Metric-Outdoor-Small-hf`) | metric | DA-V2 encoder fine-tuned for metric KITTI |

All load through **one adapter** (`estimators/hf_depth.py`) via
`AutoModelForDepthEstimation` — add any of them with `--models hf_depth:<hub-id>`.

### 1.3 Ground truth from LiDAR (no manual labels)

KITTI ships a Velodyne scan + calibration per frame, so we *make* GT depth by projecting
the point cloud into the camera (`datasets.py`):

```
X_cam = R0_rect · Tr_velo_to_cam · X_velo      # LiDAR point -> rectified camera frame
[u v w]ᵀ = P2 · [X_cam; 1];  u/=w, v/=w        # -> pixel
depth(v,u) = X_cam.z                            # metres; keep nearest on collision
```

This yields a **sparse** map (~4–5 % of pixels — LiDAR is far sparser than the image),
which is exactly what the KITTI depth benchmark scores against.

### 1.4 The metrics — where they come from, with formulas

All metrics compare a predicted depth $\hat d_i$ to GT $d_i$ over the $N$ valid (LiDAR)
pixels $i$. They are the **Eigen et al. (2014)** set — the standard since the first deep
single-image depth paper — split into *error* metrics (lower better) and *threshold
accuracy* (higher better).

**Error metrics.**

$$
\text{AbsRel}=\frac{1}{N}\sum_i \frac{|\hat d_i-d_i|}{d_i}
\qquad
\text{SqRel}=\frac{1}{N}\sum_i \frac{(\hat d_i-d_i)^2}{d_i}
$$

$$
\text{RMSE}=\sqrt{\frac{1}{N}\sum_i (\hat d_i-d_i)^2}
\qquad
\text{RMSE}_{\log}=\sqrt{\frac{1}{N}\sum_i (\log \hat d_i-\log d_i)^2}
$$

Why four? They weight errors differently — picking one hides failure modes:

- **AbsRel** divides by $d_i$, so a 1 m error on a 5 m car and on a 50 m truck count
  *proportionally*. This scale-free property makes it the headline number.
- **SqRel** squares the numerator → punishes a few large blunders much harder than many
  small ones (sensitive to outliers).
- **RMSE** is in **metres** and is *not* normalized by depth, so it is dominated by the
  **far** field (a 10 % error at 60 m hurts 12× more than at 5 m). Good for "how many
  metres off on average", bad at seeing near-field quality.
- **RMSE_log** measures error in **log space**: $\log\hat d-\log d=\log(\hat d/d)$ is a
  *ratio* error, so near and far contribute evenly — the complement to plain RMSE.

**Threshold accuracy** ($\delta_k$). The fraction of pixels whose prediction is within a
factor $1.25^k$ of GT:

$$
\delta_k=\frac{1}{N}\Big|\Big\{\,i:\ \max\!\Big(\tfrac{\hat d_i}{d_i},\ \tfrac{d_i}{\hat d_i}\Big)<1.25^{\,k}\,\Big\}\Big|,\quad k\in\{1,2,3\}
$$

The $\max(\hat d/d,\,d/\hat d)$ is a symmetric ratio (penalizes over- and under-estimates
equally); $1.25^1=1.25$ (within 25 %), $1.25^2\approx1.56$, $1.25^3\approx1.95$.
$\delta_1$ ("percent of pixels essentially correct") is the most cited accuracy number.

**The "Garg crop".** LiDAR returns are unreliable at the image top (sky) and edges, so
the field standardized on **Garg et al. (2016)**'s center crop
($y\in[0.408,0.992]H$, $x\in[0.036,0.964]W$) — we evaluate only inside it for
comparability with published numbers.

### 1.4.1 The alignment subtlety (relative vs metric)

A **relative** model predicts depth only up to a global scale $s$: it outputs
$\hat d_i = d_i/s$ for some unknown $s$. Scoring that directly is meaningless, so we first
estimate $s$ and rescale. We use **median scaling** (the Monodepth2 protocol):

$$
\hat d_i \leftarrow \hat d_i\cdot\frac{\operatorname{median}(d)}{\operatorname{median}(\hat d)}
$$

The median is a *robust* scale estimate (insensitive to a few wild pixels). This is what
"aligned" means in the results table; **metric** models are scored **as-is** (no $s$
given). The honest reading: a relative model is handed a free global scale, a metric model
must be right outright — which is exactly why the two columns can disagree (§1.5).

### 1.5 Results — inference comparison

> Run: `python -m DeepDataMiningLearning.ngperception.depth.run_eval --models … --max-images 60 --stride 40`

![depth bars](docs/depth_bars.png)

60 KITTI frames, LiDAR GT, Garg crop. **Aligned** = median scale-invariant (fair for all);
**Metric** = true metres (no alignment — only meaningful for metric models):

| model | type | aligned AbsRel↓ | aligned δ1↑ | metric AbsRel↓ | metric δ1↑ |
|---|---|---|---|---|---|
| `Intel/dpt-hybrid-midas` (2019) | relative | 0.146 | 0.832 | — | — |
| `Depth-Anything-V2-Small` (2024) | relative | 0.113 | 0.882 | — | — |
| `Intel/zoedepth-kitti` | metric | **0.077** | **0.933** | 0.853 | 0.000 |
| `Depth-Anything-V2-Metric-Outdoor-Small` | metric | 0.121 | 0.853 | **0.122** | **0.833** |

Three lessons fall out of this one table:

1. **Basic → SOTA (relative):** Depth-Anything-V2 (0.113) clearly beats the 2019 DPT
   baseline (0.146) on the same scale-invariant metric.
2. **Domain specialization wins on accuracy:** ZoeDepth, *fine-tuned on KITTI*, has the
   best **aligned** score of all (0.077, δ1 0.933) — a strong argument for the fine-tuning
   we do next.
3. **"Metric" is a property of the checkpoint, not just the architecture.** Look at the
   metric columns: `Depth-Anything-V2-Metric-Outdoor` gives genuine metres (AbsRel 0.122),
   but the HF `zoedepth-kitti` port is **mis-scaled** — it outputs ~0.6–9.5 instead of
   ~4–80 m, so as-metric it collapses (δ1 = 0.000) even though its *shape* is excellent
   (aligned 0.077). **Always validate a metric model's absolute range against GT before
   trusting its metres** — the framework prints both columns precisely so this can't hide.

### 1.6 Composable add-on — per-object distance

Depth becomes a *downstream head of detection*: run an ngdet detector, then read the
depth map inside each box (`fusion.py`, median over the central 60 % to avoid box-edge
background). Use a **correctly-calibrated metric** model so the numbers are real metres:

```bash
python -m DeepDataMiningLearning.ngperception.depth.run_eval \
    --models hf_depth:depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf \
    --max-images 5 --viz --detector hf_detr:facebook/detr-resnet-50
# -> output/depth/viz/fuse_*.png : boxes labelled "vehicle 24m" over a depth colormap
```

![fusion example](docs/fuse_distance_example.png)

Top: DETR boxes annotated with per-object distance; bottom: the metric depth map (bright =
near). The two oncoming cars read ~12 m and ~24 m. `--viz` is restricted to metric models —
running it on a relative model would print meaningless (un-scaled) distances.

Add `--video` to render the same fusion as an **mp4** over consecutive frames
(`output/depth/depth_fusion_<model>.mp4`, 1224×740, distance-annotated RGB over depth):

```bash
python -m DeepDataMiningLearning.ngperception.depth.run_eval \
    --models hf_depth:depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf \
    --max-images 30 --stride 1 --video --video-frames 30 --fps 5 \
    --detector hf_detr:facebook/detr-resnet-50
```

This is the whole point of `ngperception`: detection answers *what & where (2D)*, depth
adds *how far*, and together they are the first step toward 3D perception.

### 1.7 Fine-tuning Depth-Anything, and mixing KITTI + Waymo + nuScenes

**Is Depth-Anything easy to fine-tune?** Yes. It is a standard model — a DINOv2 encoder
+ a DPT regression head — so it trains like any HF `AutoModelForDepthEstimation`: a
supervised loop with a depth loss, masked to the sparse LiDAR-GT pixels. You can LoRA the
DINOv2 encoder (cheap, like ngdet §21–22) or full-tune the head. No bespoke decoder, no
custom kernels — much easier than the detection VLMs.

The standard loss is the **scale-invariant log loss (SiLog)** from Eigen et al., over the
valid pixels with $g_i=\log\hat d_i-\log d_i$:

$$
\mathcal{L}_{\text{SiLog}}=\sqrt{\;\frac{1}{N}\sum_i g_i^{2}\;-\;\frac{\lambda}{N^{2}}\Big(\sum_i g_i\Big)^{2}}\,,\qquad \lambda\in[0,1]\ (\approx0.85)
$$

The first term is the log-error; the **second term subtracts the mean log-error squared**,
which removes a constant per-image scale ($\hat d\!\to\!s\hat d$ shifts every $g_i$ by
$\log s$, and the second term cancels it). So $\lambda\!\to\!1$ trains *relative* depth and
$\lambda\!=\!0$ trains *metric* depth — SiLog is the training-time twin of the median
alignment we use at eval (§1.4.1).

**Can we mix KITTI + Waymo + nuScenes?** Yes — and it is the right move for generalization
— but with one crucial caveat:

- **Scale-invariant (relative) fine-tuning on the mix is clean.** SiLog with $\lambda\!\approx\!1$
  is camera-agnostic, so pooling three domains just makes the model more robust (same idea
  as ngdet's mixed dataset, §17).
- **Metric fine-tuning on a mix is subtle — cameras differ.** Metric depth depends on the
  **intrinsics**: a car at 20 m fills different pixel heights under KITTI's ($f\!\approx\!720$ px),
  Waymo's, and nuScenes' ($f\!\approx\!1260$ px) focal lengths. Train metric depth on pooled
  cameras naively and the model sees contradictory targets. The fixes are what ZoeDepth /
  **Metric3D** do: condition on intrinsics or warp to a **canonical camera** (rescale depth
  by $f_{\text{canon}}/f_{\text{cam}}$). So: mix freely for relative; for metric, either
  fine-tune per-camera-domain or add intrinsic conditioning.

**Dataset availability for depth GT** (LiDAR projected to camera, as in §1.3):

| dataset | LiDAR + calib here? | depth GT status |
|---|---|---|
| KITTI | ✅ `velodyne` + `calib` | **wired** (`KITTIDepthDataset`) |
| nuScenes | ✅ `samples/LIDAR_TOP` + `v1.0-trainval` calib | buildable (project via ego-pose + calibrated_sensor) |
| Waymo | ⚠️ this subset has camera only (no `lidar*` parquets) | needs the LiDAR + `lidar_camera_projection` components downloaded |
| nuImages | ❌ 2D annotations only, no LiDAR | not possible (use full nuScenes) |

So the eval/training **already runs on KITTI**, extends cleanly to **nuScenes** (LiDAR is
present — just needs the projection wired), and needs the **Waymo LiDAR** parquets before
Waymo depth GT can be built. That is the next build.

---

## 2. 3D semantic occupancy — and how far depth alone gets you

Occupancy prediction is the current frontier of camera perception: instead of boxes, you
predict a **dense 3D voxel grid** of semantics around the car (Occ3D-nuScenes:
$200\times200\times16$ voxels at 0.4 m, 17 classes + free). It handles arbitrary shapes and
unknown objects, and is the representation behind Tesla-style occupancy networks.

Depth is the natural bridge to it — so before training a heavy occupancy net, we ask a
concrete question: **how far does a depth foundation model + 2D segmentation get you,
with no occupancy training at all?** This is the ViPOcc "vision priors" question, and it
gives a measured baseline the learned nets must beat.

### 2.1 The depth→occupancy baseline

For each of the **6 surround cameras** (`predictors/depth_lift.py`):

1. metric depth (Depth-Anything-V2-Metric) → per-pixel range;
2. back-project every pixel to a 3D point, $X_{\text{cam}}=d\,K^{-1}[u,v,1]^\top$;
3. lift to the ego frame, $X_{\text{ego}}=R_{c\to e}X_{\text{cam}}+t_{c\to e}$
   (the Occ3D grid is in the **ego frame** — verified by projecting the LiDAR sweep, §2.2);
4. tag each point with a **Cityscapes** segmentation label (SegFormer), mapped to the 18
   Occ3D classes; sky and far/zero depth are dropped;
5. **splat** points into voxels → a predicted semantic grid.

No training — pure geometry + 2D priors. Metric: official Occ3D **mIoU over 17 classes on
camera-visible voxels** (`mask_camera`), plus a class-agnostic **geometric IoU**.

### 2.2 Validating the geometry (and an honest ceiling)

The Occ3D GT is **densified** (aggregated multi-sweep + completion) — ~30–42 k occupied
voxels per frame. A *single* LiDAR sweep fills only ~6 k voxels, so projecting the real
LiDAR scores just **geo-IoU ≈ 0.17** (≈97 % precise, but ~15 % recall — it physically can't
see the densified GT). That number is the **single-shot geometric ceiling**, and it tells
us up front that any single-frame lift (LiDAR or depth) is recall-bound; the headroom above
it is exactly what learned occupancy nets buy by *hallucinating* the dense, occluded grid.

### 2.3 Results — runnable baselines vs the SOTA target

Evaluated on the **official Occ3D val split** (`annotations.json` val scenes), 30 samples,
6 cameras, `SegFormer-b2` (Cityscapes) semantics, camera-mask mIoU:

| method | mIoU↑ | geo-IoU↑ | runnable here? |
|---|---|---|---|
| LiDAR oracle (single sweep, geometry) | — | **0.167** | ✅ |
| depth→occ, DA-V2-Metric-**Small** + seg | **0.014** | 0.093 | ✅ |
| depth→occ, DA-V2-Metric-**Base** + seg | 0.011 | 0.085 | ✅ |
| — *published SOTA (learned, camera-only)* — | | | |
| CTF-Occ (Occ3D paper) | 28.5 | — | ✗ (mmdet3d) |
| FlashOcc | ~32 | — | ✗ |
| **Dr.Occ (depth-guided)** | **43.4** | — | ✗ |
| EFFOcc (18.4 M params) | 50.5 | — | ✗ |

(`run_eval` prints the SOTA reference table after every run; the learned nets need
mmdetection3d, which is incompatible with this env's torch 2.10, so they are cited, not
re-run.) Three things to read off:

1. **A bigger depth model does *not* help** — DA-V2-Metric-Base is slightly *worse* than
   Small (mIoU 0.011 vs 0.014). Depth *quality* is not the bottleneck; the **single-shot
   lift** is. This is the key negative result of the sweep.
2. **Geometry:** depth recovers ~56 % of the single-sweep LiDAR's geo-IoU (0.093 vs 0.167).
   Monocular depth is *dense but imprecise* — a small metric error splats a voxel into the
   wrong cell, so it loses to even a sparse-but-exact LiDAR sweep.
3. **The gap is ~20–35× in mIoU** (≈1.4 → 28–50). Only the large static surfaces survive in
   the baseline (`terrain`, `driveable_surface`, `manmade`, `vegetation` ~0.04–0.07);
   small/dynamic classes are ≈0. That gap is the whole research opportunity — and the
   strongest learned nets that close it (Dr.Occ, 43.4) are **depth-guided**, confirming
   depth is a valuable *signal* even though depth *alone* is far from enough.

### 2.4 What this says about the direction

A depth-foundation baseline is **dense but imprecise** (depth errors splat voxels into the
wrong cell) and **single-shot** (no memory of occluded space), so it sits well below
learned occupancy nets (SOTA Occ3D mIoU is ~0.4+). That gap is the research opportunity,
and it points at concrete, fundable work:

- **Depth as supervision / prior** for a learned BEV→voxel head (BEVDepth / ViPOcc idea):
  start from this baseline's geometry, learn the occluded completion.
- **Temporal fusion**: aggregate several frames' lifts (ego-motion compensated) to close
  the recall gap the single-sweep ceiling exposes.
- **Better metric depth across cameras** (§1.7 canonical-camera): the 6 nuScenes cams have
  very different intrinsics, so metric error directly becomes voxel error.

The point of the baseline is not to win — it is to **quantify the headroom** and make the
case for the learned model concrete, on data you already have.

### 2.5 Case study — running a real (self-supervised) occupancy net: GaussianOcc

The learned SOTA nets in §2.3 need mmdetection3d (incompatible with modern torch). The one
we *did* get running is **GaussianOcc** (ICCV'25) — a **fully self-supervised** 3D-Gaussian-
Splatting method. It's the cleanest to install (no mmcv; the 3DGS rasterizer + simple-knn),
and on Occ3D-nuScenes `val_tiny` (30 samples) it scores **mIoU ≈ 8.4** — ~6× our depth-lift
baseline, but far below the supervised nets (28–50). That gap is the price of using *no 3D
labels*. (Setup is fiddly: ~9 source patches — see the internal notes.)

**How it works — and where the "Gaussians" come from.** The key idea: represent the scene
as a set of **3D Gaussians** (each has a position, scale, rotation, opacity, and a semantic
class), and the **network predicts those Gaussians directly from the 6 surround images**.
There is no pre-built scene — the Gaussians *are* the network's output:

```
6 camera images ──encoder──► image features ──lift/decoder──► a set of 3D Gaussians
                                                              (the predicted "scene")
        ┌──────────────────────────────────────────────┴───────────────────┐
   (a) Gaussian-splat the Gaussians back into each camera                   │
       → rendered DEPTH + rendered 2D SEMANTICS         (training signal)    │
   (b) query the Gaussian field at the 200×200×16 voxel centres             │
       → per-voxel class = the OCCUPANCY output         (what we evaluate)   ▼
```

**Self-supervision (no 3D GT).** Training never sees occupancy labels. It minimizes:
(1) **photometric/depth consistency** — splatting the Gaussians into one frame should
reproduce what the *other* cameras/adjacent timestamps actually saw; (2) **2D semantic
consistency** — the rendered semantics should match 2D labels (from GroundedSAM). So the
only supervision is "your Gaussians, when rendered, must look like the images" — and the 3D
occupancy falls out of the same Gaussians. (This is exactly the *rendering supervision* idea
we flagged as a v2 upgrade for our own LSS head.)

**Why it's noisy (mIoU 8.4).** Rendering only constrains what the cameras *see*. Behind
objects and far away the Gaussians are under-determined, so the model **over-fills along
camera rays** — the characteristic radial "fan" in the 3D render below. Supervised nets,
trained on the densified Occ3D GT, learn to *complete* that occluded space; that completion
is most of the 8 → 40 mIoU gap.

### 2.6 Visualizing occupancy: open3d, not mayavi

GaussianOcc renders **depth** as an image (it's a depth-rendering method), but the
**occupancy** is a raw voxel grid (`probability`, shape `(18,200,200,16)` → `argmax` →
class per voxel). To *see* it you need a 3D renderer. The papers use **mayavi** (VTK + Qt,
painful headless). We use **open3d** instead — newer, `pip`-installable, and its
`OffscreenRenderer` runs **headless via EGL** out of the box (works on this WSL box). Build
a coloured `VoxelGrid` from the occupied voxel centres and render from an oblique BEV angle:

![gaussianocc 3d occupancy](docs/gaussianocc_occ3d_frame.png)

Now the structure is legible: magenta = driveable road, greens = terrain/vegetation, blue =
cars, red/orange = barriers/objects, with the ego gap in the centre. A flat top-down BEV
(collapsing the 16 height levels) reads far worse — the renderer choice matters for
*legibility*, though not for the underlying mIoU.

### 2.7 What we can build on this

- **Borrow the rendering supervision.** Add a differentiable-render loss to our planned
  depth-supervised LSS head (the GaussRender idea): supervise with Occ3D GT where it exists
  *and* with self-supervised rendering everywhere — best of both.
- **Adapt self-supervised occupancy to label-poor domains.** GaussianOcc needs no 3D GT, so
  it can train on KITTI / Waymo (which lack occupancy labels) from images + 2D priors alone.
- **Supervised-vs-self-supervised study.** Compare our (planned) GT-supervised LSS occ vs
  GaussianOcc self-sup on the same Occ3D mIoU — quantify exactly what the 3D labels buy.
- **Reuse the Gaussians downstream.** The predicted Gaussian field also gives novel-view
  synthesis and scene completion — handles beyond occupancy.

Bottom line: GaussianOcc is our **reference point and idea source** for the GS route; the
main build remains the controllable pure-PyTorch depth-supervised LSS occupancy head.

### 2.8 Our depth-supervised LSS occupancy — results

The pure-PyTorch, GT-supervised LSS head from §2.7 (`models/lss_occ.py` + `train_lss.py`),
trained on **only 500 nuScenes samples for 8 epochs (~20 min, 3.5 GB)**:

| method | mIoU↑ | geo-IoU↑ | training | runs in main env? |
|---|---|---|---|---|
| depth→occ lift baseline (§2.3) | 0.014 | 0.09 | none | ✅ |
| LiDAR oracle (single sweep) | — | 0.18 | none | ✅ |
| GaussianOcc (self-supervised) | 0.084 | — | pretrained, no 3D GT | ✗ (separate env, ~9 patches) |
| our LSS occ — **ResNet18** | 0.092 | 0.547 | 500 samples, 8 ep (~20 min) | ✅ pure PyTorch |
| our LSS occ — **DINOv2 (frozen) + 3× data** | **0.152** | **0.626** | 1500 samples, 10 ep, AMP (~40 min) | ✅ pure PyTorch |
| supervised SOTA (CTF/FlashOcc/…) | 28–50 | — | full train, mmdet3d | ✗ |

Three things this shows:

1. **It already edges out GaussianOcc** (9.2 vs 8.4 mIoU) with ~20 min of training on a tiny
   slice — and unlike GaussianOcc it runs in the **main torch-2.10 env**, no mmcv, no patches.
2. **Learned completion is real.** Its **geo-IoU 0.547 is ~3× the single-sweep LiDAR ceiling**
   (0.18) and ~6× the depth-lift baseline (0.09). A single-shot projection can only mark what
   it directly observes; the *learned* model fills the densified, occluded Occ3D GT — exactly
   the headroom §2.2 said the learning buys.
3. **It scales predictably.** Swapping the ResNet-18 for a **frozen DINOv2-small** backbone
   and tripling the data (500 → 1500) lifts mIoU **0.092 → 0.152 (+65 %)** and geo-IoU
   0.547 → 0.626 — with *AMP it actually trains faster* (~6 it/s, 2.7 GB) since the backbone
   is frozen and only the head learns. Tellingly, DINOv2's **epoch-0 mIoU (0.083) already
   matched ResNet's *final***: the foundation-model features are doing the heavy lifting.
   Both runs were still climbing at the last epoch — more data, a longer schedule, a larger
   DINOv2, and a deeper voxel decoder are the obvious next steps toward the supervised SOTA
   band, all on infrastructure we fully control.

The depth supervision is doing its job: the depth CE drops alongside the occupancy CE, so
the lift's geometry sharpens as the occupancy learns (the BEVDepth effect).

```bash
# strongest config: frozen DINOv2 + AMP + more data
python -m DeepDataMiningLearning.ngperception.occupancy.train_lss \
    --gts <occ3d_gts> --nusc <nuscenes_root> \
    --backbone dinov2 --amp --max-samples 1500 --epochs 10 --depth-weight 1.0
```
