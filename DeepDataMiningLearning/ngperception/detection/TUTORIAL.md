# Pure-PyTorch 3D Object Detection — PointPillars from scratch

A teaching walkthrough of `ngperception/detection/`: a LiDAR 3D object detector built with
**only `torch` + `numpy`** — no `spconv`, no `mmcv`, no compiled CUDA ops. It runs in the same
environment as the rest of the framework, and every piece is small enough to read. We build it
by **harvesting the algorithm from OpenPCDet** (via the local `2D3DFusion/mydetector3d` fork)
and replacing its one CUDA dependency (`iou3d_nms`) with pure-torch geometry.

The companion module is [`../occupancy/`](../occupancy/) (3D semantic occupancy). Detection and
occupancy are *the same shape of problem* — encode a scene into BEV/voxel features, then attach
a task head — which is why they will eventually share one encoder (roadmap §10).

> Everything here was developed and verified on a single RTX 3090. Commands use
> `P=/home/lkk688/miniconda/envs/py312/bin/python`; data is extracted KITTI at
> `/mnt/e/Shared/Dataset/Kitti`.

---

## 1. The task

Given a LiDAR point cloud, predict **7-DoF 3-D boxes** `[x, y, z, dx, dy, dz, heading]`
(centre, size, yaw) for each object, in the LiDAR frame. We start with **KITTI Car** (one
class). The metric is **average precision (AP)** at a BEV/3-D IoU threshold (0.5 "easy",
0.7 "strict").

## 2. Why PointPillars — the one classic detector with no sparse conv

LiDAR is sparse: a voxel grid over a scene is >99 % empty, so voxel-based detectors
(SECOND, CenterPoint, PV-RCNN) use **sparse 3-D convolution** (`spconv`, a compiled CUDA
library). PointPillars sidesteps this with the **pillar trick**: collapse the whole Z column
into a single "pillar", encode each pillar into a vector, **scatter** those vectors into a 2-D
BEV pseudo-image, and run an ordinary **2-D CNN**. No 3-D conv, no sparse ops — perfect for a
pure-PyTorch build. The pipeline:

```
points → pillarize → PillarVFE → scatter to BEV → 2-D CNN backbone → anchor head → decode + NMS
        (numpy)     (PointNet)   (dense image)     (Conv2d)          (Conv2d)      (pure-torch IoU)
```

## 3. The architecture, with the code

### 3.1 Pillarize — points → pillars (`pointpillars.py: pillarize`)

OpenPCDet builds pillars with an `spconv` voxel generator. We do it with a **vectorised numpy**
group-by: hash each point to a pillar index, sort, and take up to `max_points` per pillar.

```python
idx  = np.floor((points[:, :3] - pcr[:3]) / vs).astype(np.int64)          # per-point pillar (x,y,z)
keys = idx[:, 0]*(grid[1]*grid[2]) + idx[:, 1]*grid[2] + idx[:, 2]        # flatten to a hash
uniq, inv = np.unique(keys, return_inverse=True)                          # P unique pillars
order = np.argsort(inv, kind="stable"); inv_s = inv[order]                # group points by pillar
within = np.arange(len(inv_s)) - np.repeat(np.cumsum(counts)-counts, counts)   # pos within pillar
voxels[inv_s[within<max_points], within[within<max_points]] = pts_s[within<max_points]
```

Output: `voxels (P, max_points, C)`, `num (P,)`, `coords (P,3)` — no Python loop over points.

### 3.2 PillarVFE — a PointNet per pillar (`PillarVFE`)

Each pillar's points are turned into a **10-D feature**: the raw `[x,y,z,intensity]`, the offset
to the pillar's **point-mean** (`f_cluster`), and the offset to the pillar's **geometric centre**
(`f_center`). A shared `Linear → BatchNorm → ReLU` then a **max over points** gives one vector
per pillar (the classic PointNet symmetric function):

```python
f_cluster = voxels[:,:,:3] - points_mean            # relative to the pillar's point centroid
f_center  = voxels[:,:,:3] - pillar_geometric_centre
feats = cat([voxels, f_cluster, f_center], -1)      # (P, T, 10)
x = relu(bn(linear(feats)))                         # (P, T, 64)
pillar_feature = x.max(dim=1)[0]                     # (P, 64)  <- permutation-invariant
```

### 3.3 Scatter → BEV, then a 2-D backbone

`scatter_bev` drops each pillar vector back at its `(x,y)` cell of an empty `(64, ny, nx)`
canvas (most cells stay zero). Then `BaseBEVBackbone` is a plain multi-scale 2-D CNN (three
stride-2 blocks + transpose-conv upsamples, concatenated) — exactly the SSD-style backbone,
`nn.Conv2d` throughout.

### 3.4 Anchor head — boxes from a conv (`AnchorHead`)

At every BEV cell we place **anchors** (a Car-sized box at 0° and 90°). Two `1×1` convs predict,
per anchor, a **class logit** and a **7-D box residual**. The residual is decoded against the
anchor by the **`ResidualCoder`** (OpenPCDet's parameterisation — centre offset normalised by
the anchor diagonal, log-size, angle delta):

```python
xt = (xg-xa)/diag ; zt = (zg-za)/dza ; dxt = log(dxg/dxa) ; rt = rg-ra    # encode
xg = xt*diag+xa   ; dxg = exp(dxt)*dxa ; rg = rt+ra                       # decode
```

### 3.5 The one thing OpenPCDet does in CUDA — we do in torch (`box_utils.py`)

3-D IoU and NMS are OpenPCDet's compiled `iou3d_nms`. We provide two pure-torch paths:
- **`boxes_bev_iou_aligned`** — vectorised axis-aligned BEV IoU (exact when yaw≈0). Fast; used
  for NMS and the default target assignment.
- **`rotated_iou_bev_paired_torch`** — *exact* rotated IoU (§7), used for rotated assignment.

`nms_aligned` is a plain greedy NMS on the axis-aligned IoU. **No compiled code anywhere.**

### 3.6 Losses (`losses.py`)

Sigmoid **focal loss** for classification (handles the anchor foreground/background imbalance)
+ **smooth-L1** for the 7-D box residual, normalised by the number of positive anchors — both
lifted from OpenPCDet and de-`.cuda()`'d.

## 4. The subtle KITTI bit — camera→LiDAR boxes (`kitti_dataset.py`)

KITTI labels are 3-D boxes in the **rectified camera** frame; the model works in the **LiDAR**
frame. Getting this transform wrong silently corrupts every box, so we harvest it exactly:

```python
xyz_lidar = calib.rect_to_lidar(xyz_camera)         # via R0_rect and Tr_velo_to_cam
xyz_lidar[:, 2] += h/2                               # KITTI z is box-bottom -> box-centre
box_lidar = [x, y, z, l, w, h, -(ry + pi/2)]        # [x,y,z,dx,dy,dz,heading]
```

Verification (`python -m ...detection.kitti_dataset`): a loaded Car comes out at size
**4.15 × 1.73 × 1.57 m** — textbook KITTI Car (~3.9 × 1.6 × 1.5) — at a sensible ground-level
position. That one check confirms calibration, axis order, and the bottom→centre shift.

## 5. Target assignment — the heart of anchor training (`AnchorHead.assign`)

Training an anchor detector is mostly about **which anchors are "positive"**. For each anchor we
take its max IoU with any GT: `IoU ≥ 0.6` → positive (regress that GT), `IoU < 0.45` → negative
(background), in-between → ignored. Positives get an encoded box target + class; the focal loss
covers pos+neg, the box loss only positives. **The IoU used here decides everything** — which is
why §6 is about replacing the axis-aligned IoU with a rotated one.

## 6. M2 — rotated-IoU assignment + a direction classifier (what helped, what didn't)

We first **diagnosed** the accuracy ceiling by printing, per GT box, the best detection's IoU
and its error breakdown after a strong overfit. The finding overturned the obvious guess:

> Heading was **already well learned** (mod-π error 0.02–0.09 rad; and a car box is
> 180°-symmetric so a π-flip doesn't even change IoU). The real limiter was **assignment
> confusion in multi-car scenes** — axis-aligned IoU ignores orientation, so it marks anchors of
> *both* rotations near a car as positive, giving muddy regression targets.

So M2 adds two standard fixes, **both opt-in flags**, and we A/B'd them (overfit, 6 frames /
18 cars, 3 seeds):

| config | mAP@0.5 (3-seed) | reading |
|---|---|---|
| baseline (axis-aligned assign) | 0.225 ± 0.097 | over-assigns: num_pos = 276 |
| **`--rotated-assign`** | **0.323 ± 0.046** | clean assign (num_pos 123): higher mean, **½ variance** |
| `--rotated-assign --use-dir` | 0.141 (1 seed) | direction classifier **hurt** |

**Rotated-IoU assignment** gives a modest mean gain (+0.098, ~1.6σ at n=3) but a robust
**halving of variance** — the cleaner target set stabilises training, exactly as the diagnosis
predicted. The **direction classifier hurt**: because the raw angle regression already resolves
heading, its 2-bin inference correction only perturbs good headings — so it stays **off by
default**. (Lesson: diagnose before you "fix"; the intuitive culprit — heading — was not the
bottleneck, and the textbook fix for it made things worse.)

## 7. M2b — vectorising rotated IoU in pure torch (the fun algorithm)

Rotated assignment first used a numpy Sutherland–Hodgman clip per candidate pair (~2× iter
time). M2b replaces it with a **fully vectorised** convex-polygon intersection over `P` box
pairs at once (`rotated_iou_bev_paired_torch`). For two convex quads the intersection polygon's
vertices are only three kinds of point:

1. **corners of A inside B** (and of B inside A) — a left-of-all-edges test;
2. **edge×edge intersections** — all 4×4 segment pairs, keep the ones inside both segments;

giving ≤24 candidate points per pair. We then need their polygon area without a Python loop.
The trick that makes it branchless:

```python
c   = masked_centroid(pts)                      # centroid of the valid candidates
ang = atan2(pts - c); ang[invalid] = +inf       # invalid points sort to the end
sp  = pts[argsort(ang)]                          # candidates in CCW order, valid ones first
sp[invalid] = sp[:, 0]                            # <-- fold invalids onto the FIRST vertex
area = shoelace(sp)                               # closing edge (v_{k-1}->v_0) now appears; extras are 0
```

Folding every invalid point onto the first valid vertex makes the closing edge of the real
polygon fall out of the shoelace sum, while all the degenerate edges contribute exactly zero —
so one fixed-size shoelace handles a variable number of intersection vertices. It **matches the
numpy version to 1e-4** on random boxes and is **~65× faster per pair** (5100 pairs in 14.7 ms
on GPU). Rotated assignment now costs ~the same as the axis-aligned baseline (46 s vs 44 s per
250-iter overfit), so it is **practical for full-split training** — the prerequisite for turning
the overfit gain into real val-AP on H100.

## 8. Results (M0 → M2b)

| check | result | notes |
|---|---|---|
| box ops (unit tests) | coder round-trip 0; IoU 1.0/0.333/0.707; NMS dedups | `box_utils` / `eval3d` self-tests |
| forward + loss + backward | ✓ (4.8 M params, 107 k anchors) | random-point smoke |
| KITTI loader | Car box 4.15×1.73×1.57 m | cam→LiDAR transform verified |
| **overfit** (6 frames) | **18 dets for 18 GT cars, mAP@0.5 = 0.45** | localisation correct |
| **generalisation** (val 150 held-out) | val **mAP@0.5 0 → 0.324** over 30 ep | learns, not memorises |
| M2 rotated assign | 0.225 → **0.323**, variance halved | cleaner targets |
| M2b torch rotated IoU | numpy-exact, **65× faster** | rotated assign now ~free |

Honest ceilings: **mAP@0.7 ≈ 0.036** (needs tight boxes — regression precision + full training);
these numbers are **6.7 % of the KITTI split, 30 epochs, from scratch**, on one 3090. Competitive
KITTI numbers are a full-schedule **H100** job (§10) — the point here is a *correct, readable,
dependency-free* pipeline you can learn from and extend, not a leaderboard entry.

## 9. Datasets — KITTI, nuScenes, Waymo (one interface)

Every loader returns the **same dict** — `{"points": (N,C), "gt": (M,8)}` with each box
`[x, y, z, dx, dy, dz, heading, label]` in the **LiDAR frame** — so any model trains on any
dataset unchanged. The only per-dataset work is getting boxes into that frame:

| dataset | file | boxes come from | verified |
|---|---|---|---|
| **KITTI** | `kitti_dataset.py` | camera-frame labels → `rect_to_lidar` (R0, Tr_velo_to_cam) | Car 4.15×1.73×1.57 m |
| **nuScenes** | `nuscenes_dataset.py` | devkit `get_sample_data` (already LiDAR-frame) + `wlh`/quaternion | Car 4.25×1.64×1.44 m |
| **Waymo** (KITTI-export) | `waymo_dataset.py` | reuses KITTI transform w/ front-cam `Tr_velo_to_cam_0` + `label_all` | 31 cars in-bounds, 4.86×2.14×1.77 m |

Each loader has a `--root … ` sanity `__main__` that loads a frame and prints the Car size —
the one check that confirms calibration, axis order, and the bottom→centre shift are right.
(Data notes: KITTI + nuScenes are staged in full; only a **1-frame** Waymo sample is local, so
the Waymo loader is verified to *load* but full Waymo training is an H100/data job.)

## 10. A second model — CenterPoint (center-based vs anchor-based)

PointPillars (§3–6) is **anchor-based**: place boxes everywhere, classify/regress each, match by
IoU. **CenterPoint** ([`centerpoint.py`](centerpoint.py)) is the other classic paradigm —
**anchor-free, center-based** — and it drops onto the *same* pillar front-end, only swapping the
head:

- predict a **per-class BEV Gaussian heatmap** (each object is a peak) + per-cell regression
  (sub-voxel offset, height, log-size, sin/cos yaw);
- targets: render a CenterNet Gaussian at each object's BEV centre; loss = penalty-reduced focal
  on the heatmap + L1 on the regression at centres;
- **decode is a 3×3 max-pool on the heatmap** ("keep local maxima") — no anchors, no rotated NMS,
  no spconv. That is why it's such a clean pure-torch fit.

The comparison is itself a lesson. On the same 6-frame KITTI overfit:

| model | paradigm | overfit mAP@0.5 |
|---|---|---|
| PointPillars | anchor-based (axis-aligned assign) | 0.45 |
| **CenterPoint** | center-based | **0.925** |

CenterPoint overfits *far* better — because it **sidesteps the very anchor-assignment ambiguity**
that §6 was fighting: there is no "which of the two anchor rotations near this car is positive",
just one centre per object. (This is not a claim that CenterPoint > PointPillars in general —
full-data numbers depend on tuning — but it cleanly shows *why* the center paradigm became
dominant.) Both share `PillarVFE` + scatter + backbone, so this is a ~150-line head swap.

### 10.1 A third paradigm — the BEV transformer (DETR)

[`bev_transformer.py`](bev_transformer.py) adds the **attention/set-prediction** paradigm — the
one that fits pure-torch *because attention needs no spconv*. Object **queries** emit a class +
box; training matches queries to GT **1-to-1 with the Hungarian algorithm** (no anchors, no NMS).

This section is a two-act teaching story about **why modern DETRs look the way they do**:

**Act 1 — the vanilla version fails.** A plain decoder where every query attends to *all* ~13 k
BEV tokens (`nn.TransformerDecoder`) **trains but never localises**: on the 6-frame overfit it
sits at **mAP@0.5 = 0.000** even after 2 000 iterations (the loss falls, but boxes stay too
coarse for IoU 0.5). This is the famous **vanilla-DETR slow-convergence** problem, reproduced.

**Act 2 — the modern recipe fixes it.** We add the three ingredients shared by every modern
DETR (Deformable-DETR / DINO / RT-DETR / BEVFormer), all **pure torch**:

- **deformable attention** (`DeformableAttention`): each query samples only `n_points=4`
  locations *around its reference point* via `F.grid_sample` — not all 13 k tokens. Sparse,
  fast, no CUDA op.
- **iterative refinement**: each decoder layer updates its reference point from its predicted
  box centre, so deeper layers attend right at the object.
- **auxiliary losses**: every layer is Hungarian-matched and supervised.

The same overfit now **converges to mAP@0.5 = 0.925** (0.00 → 0.09 @500it → 0.43 @750 → 0.88
@1000 → 0.925) — matching CenterPoint and far above the anchor PointPillars overfit (0.45).

| BEV transformer | overfit mAP@0.5 |
|---|---|
| vanilla full-attention | 0.000 (stuck) |
| **+ deformable attn + iterative refinement + aux loss** | **0.925** |

That jump *is* the lesson: **deformable attention (sampling around a reference), not a fancier
backbone or more compute, is what makes DETR-family detectors converge.** It's also the reason
adopting RT-DETR wholesale would be the wrong move for a LiDAR-BEV task — RT-DETR is a *2-D image*
detector; the part worth taking is exactly this decoder, which is what we built. (One caveat kept
honest: training shows a transient dip around it1250 before recovering — the head is a bit
lr-sensitive; a cosine schedule smooths it.)

### 10.2 A stronger backbone — PillarNeXt-style ResBEV (the lever that clearly helped)

Both CNN heads take a `backbone=` switch: `base` (the plain SSD-style `BaseBEVBackbone`) or
`res` — a **ResNet basic-block encoder + FPN neck** ([`ResBEVBackbone`](pointpillars.py)), the
PillarNeXt insight that a *well-structured* pillar backbone rivals voxel/spconv methods. On the
same 6-frame overfit (seed 0):

| model | backbone | overfit mAP@0.5 |
|---|---|---|
| PointPillars | base (SSD) | 0.17 |
| PointPillars | **res (PillarNeXt-style)** | **0.63** |
| CenterPoint | base | 0.93 |
| CenterPoint | **res** | **1.00** (lr 2e-3) |

The residual+FPN backbone is a **large, clean win** (+0.46 for PointPillars) — the biggest lever
among these additions, and it's pure 2-D conv. One training note worth keeping: CenterPoint-res
**diverged at lr 3e-3 but hit 1.00 at lr 2e-3** — the heatmap head + deeper backbone is
lr-sensitive, a reminder to sweep lr when you change the backbone.

> Out of scope (honestly): **SECOND / CenterPoint-voxel / PV-RCNN / VoxelNeXt** all need **sparse
> 3-D convolution** (`spconv`, compiled CUDA). The three paradigms here — anchor (PointPillars),
> center (CenterPoint), attention (BEV-transformer) — are all reachable *without* it, on the
> pillar/BEV/attention route that modern spconv-free work (PillarNeXt, DSVT, FlatFormer) follows.

## 11. Reproduce — exact commands

```bash
P=/home/lkk688/miniconda/envs/py312/bin/python
cd <repo root>

# --- unit / smoke tests (seconds each) ---
$P -m DeepDataMiningLearning.ngperception.detection.pointpillars    # anchor model: fwd+loss+predict
$P -m DeepDataMiningLearning.ngperception.detection.centerpoint     # center model: fwd+loss+decode
$P -m DeepDataMiningLearning.ngperception.detection.eval3d          # BEV-AP evaluator self-test

# --- dataset loader sanity (prints a Car's size in the LiDAR frame) ---
$P -m DeepDataMiningLearning.ngperception.detection.kitti_dataset    --root /mnt/e/Shared/Dataset/Kitti
$P -m DeepDataMiningLearning.ngperception.detection.nuscenes_dataset --root /mnt/e/Shared/Dataset/NuScenes/v1.0-trainval
$P -m DeepDataMiningLearning.ngperception.detection.waymo_dataset    # 1-frame local sample

# --- overfit sanity. --model {pointpillars,centerpoint,bev_transformer}, --backbone {base,res} ---
$P -m DeepDataMiningLearning.ngperception.detection.train_kitti \
    --root /mnt/e/Shared/Dataset/Kitti --overfit --max-frames 12 --epochs 80
$P -m DeepDataMiningLearning.ngperception.detection.train_kitti --backbone res \
    --root /mnt/e/Shared/Dataset/Kitti --overfit --max-frames 12 --epochs 80   # PillarNeXt-style (stronger)
$P -m DeepDataMiningLearning.ngperception.detection.train_kitti --model centerpoint --backbone res \
    --root /mnt/e/Shared/Dataset/Kitti --overfit --max-frames 12 --epochs 120 --lr 2e-3

# --- short real train (train 500 / val 150; ~30 min on a 3090) -> val mAP@0.5 ~0.32 ---
$P -m DeepDataMiningLearning.ngperception.detection.train_kitti \
    --root /mnt/e/Shared/Dataset/Kitti --max-frames 500 --val-frames 150 --epochs 30

# --- M2: rotated-IoU assignment (now GPU-fast via M2b) ---
$P -m DeepDataMiningLearning.ngperception.detection.train_kitti \
    --root /mnt/e/Shared/Dataset/Kitti --max-frames 500 --val-frames 150 --epochs 30 --rotated-assign

# --- full-schedule (H100): full split, long schedule, rotated assignment ---
$P -m DeepDataMiningLearning.ngperception.detection.train_kitti \
    --root /mnt/e/Shared/Dataset/Kitti --max-frames 3712 --val-frames 3769 --epochs 80 --rotated-assign
```

Flags: `--overfit` (train==val, sanity), `--rotated-assign` (M2/M2b rotated IoU assignment),
`--use-dir` (direction classifier — off by default, it hurt), `--batch-size`, `--lr`, `--epochs`.

## 12. Where this goes (roadmap)

- **M3 — shared encoder.** Attach this detection head onto the occupancy module's fused
  camera+LiDAR *voxel* volume (collapse Z → BEV): one front-end, two heads (occ + det).
- **M4 — free modality ablation.** The shared encoder already carries `--lidar-only` /
  `--lidar-fusion` (occ §2.8.3), so camera/LiDAR/fusion ablation for **detection** comes for free.
- **M5 — multitask.** Train occupancy + detection jointly — the first slice of a full-stack AD
  perception model, all pure PyTorch.

See [../PLAN.md](../PLAN.md) for the internal roadmap and [README.md](README.md) for the module
map, and [../TUTORIAL.md](../TUTORIAL.md) for the depth + occupancy tutorials this extends.
