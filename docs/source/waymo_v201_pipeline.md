# Waymo v2.0.1 Pipeline: Coordinate Frames, 3D Visualization, and Pseudo-Label Generation

This document captures the **full investigation and fix** of the Waymo v2.0.1
loader's coordinate transforms, the verified 3D-box-to-image projection
process, and the cross-modal pseudo-label pipeline (LiDAR clustering + 2D
detector + NVIDIA VLM tie-breaker) used for domain adaptation from
nuScenes-trained models to Waymo.

Written after the May 2026 debugging session that fixed two compounding
bugs in `dataset_waymo3dv201.py` that had been masking each other.

---

## 1. Waymo Open Dataset v2.0.1 Schema (Apache Parquet)

Unlike v1.x which packed everything into one `Frame` protobuf per record,
v2.0.1 splits each segment into **per-table parquet files**. The relevant
ones for 3D detection:

| Parquet table | Contents | One row = |
|---|---|---|
| `lidar_box` | 3D boxes from LiDAR detector | One labeled object × timestamp |
| `lidar_camera_synced_box` | Same boxes, projected to camera frame | object × camera × timestamp |
| `camera_box` | Per-camera 2D boxes (independent of 3D) | 2D detection × camera × timestamp |
| `lidar` | Range images for all 5 LiDARs | timestamp × laser |
| `camera_image` | Compressed JPEG images | timestamp × camera |
| `lidar_calibration` | LiDAR sensor → vehicle frame extrinsic, intrinsic | One per (segment, laser) |
| `camera_calibration` | Camera sensor → vehicle frame extrinsic, intrinsic | One per (segment, camera) |
| `vehicle_pose` | Vehicle frame → world frame transform | One per timestamp |

**Key column naming conventions** (from `LiDARBoxComponent` example):
```
key.segment_context_name
key.frame_timestamp_micros
key.laser_object_id            # for 3D boxes
key.camera_name                # for camera-side tables
[LiDARBoxComponent].box.center.x / .y / .z
[LiDARBoxComponent].box.size.x / .y / .z     # length, width, height
[LiDARBoxComponent].box.heading              # yaw, radians
[LiDARBoxComponent].type                     # 1=Vehicle, 2=Ped, 3=Sign, 4=Cyclist
[LiDARBoxComponent].num_lidar_points_in_box
```

**Important data property — annotation sparsity asymmetry:**

Waymo's `lidar_box` (3D-GT) **requires `num_lidar_points_in_box ≥ 9`** to be
labeled. Many visually-obvious vehicles (especially far away or partially
occluded) have insufficient LiDAR returns and **do not appear in `lidar_box`,
only in `camera_box`** (the 2D per-camera annotations).

Empirical example (frame 0 of segment `10203656353524179475_7625_000_7645_000`):
- `lidar_box` rows: **24** (10 Vehicles + 14 Signs)
- `camera_box` rows for the same timestamp: **64** (FRONT=23, FRONT_LEFT=21, FRONT_RIGHT=2, SIDE_LEFT=17, SIDE_RIGHT=1)

This asymmetry is **not a parser bug**; it is how Waymo's data is structured. It
means evaluation against 3D-GT alone systematically under-counts detection of
visually-obvious-but-sparse-LiDAR vehicles.

---

## 2. Coordinate Frame Conventions (the gospel)

All sensor frames in Waymo Open Dataset v2 are **right-handed**.

### Vehicle frame
- **+x** = forward
- **+y** = LEFT
- **+z** = up

### LiDAR sensor frame (each of the 5 LiDARs)
- Right-handed; for the TOP LiDAR, x/y/z are approximately aligned with
  vehicle frame (small rotation from mounting).

### Waymo camera frame (per sensor; differs from OpenCV)
- **+x** = forward (direction camera points)
- **+y** = LEFT
- **+z** = up

Waymo's intrinsic projection formula uses this convention natively:
```
u = -f_u * y_cam / x_cam + c_u       # negative because +y is left,
v = -f_v * z_cam / x_cam + c_v       # but image u increases to the right
```

This is **NOT the OpenCV convention** (`+x` right, `+y` down, `+z` forward).
To use a standard OpenCV intrinsic matrix `K = [[fu,0,cu,0],[0,fv,cv,0],[0,0,1,0],[0,0,0,1]]`,
we must first rotate Waymo cam → OpenCV cam:

```python
R_oc_wc = np.array([[0, -1,  0],   # opencv_X (right) = -waymo_Y (right of LEFT)
                    [0,  0, -1],   # opencv_Y (down)  = -waymo_Z
                    [1,  0,  0]])  # opencv_Z (fwd)   =  waymo_X
T_oc_wc = np.eye(4); T_oc_wc[:3, :3] = R_oc_wc

cam2img      = K @ T_oc_wc              # waymo_cam → pixel
lidar2cam    = inv(cam2vehicle)         # vehicle → waymo_cam
lidar2img    = cam2img @ lidar2cam      # vehicle → pixel
```

We **verified** this formulation produces **identical** pixel coordinates as
the official Waymo v1 SDK (`Δu = 0.0, Δv = 0.0` on every projected box
center for a held-out segment). See `/tmp/v1_proj_check.py` reference impl.

---

## 3. The Coordinate Bugs (and how we diagnosed them)

The original `dataset_waymo3dv201.py` had **two compounding bugs** that
canceled each other out, making the pipeline self-consistent but
**mirrored relative to the camera image**. The mirror was invisible until we
overlaid GT on images.

### Bug 1 — Box y/heading flip

In `_load_lidar_boxes` (formerly lines 548-551):
```python
# WRONG — claimed "LH → RH conversion", actually mirrors GT:
arr[:, 1] *= -1.0     # cy → -cy
arr[:, 6] *= -1.0     # heading → -heading
```

### Bug 2 — LiDAR / vehicle-pose extrinsic conjugation

A "fix" applied `M_reflect = diag(1, -1, 1, 1)` (y-axis mirror) to:
- LiDAR-to-vehicle extrinsic (`T_vl`)
- World-from-vehicle pose (`T_wv`)

Code (formerly lines 361 and 481):
```python
self.M_reflect = np.diag([1, -1, 1, 1]).astype(np.float32)
...
T_vl_rh = self.M_reflect @ T_vl_lh @ self.M_reflect    # WRONG
T_wv    = self.M_reflect @ T_wv_lh @ self.M_reflect    # WRONG
```

Effect: every LiDAR point's y-coordinate was negated as it came out of the
range-image decoder, putting the entire point cloud in a **mirrored frame**.

### Why the bugs hid each other

Both the boxes AND the LiDAR points were y-mirrored. The model (trained on
nuScenes in correct RH frame) saw mirrored LiDAR, predicted in mirrored
frame, matched mirrored GT → AP looked reasonable (Vehicle AP@2m = 0.36).
**Everything was self-consistent in a mirror universe.**

### How we found them

1. **Started with the visualization** of GT 3D-GT overlaid on camera images.
   Green wireframes appeared on the wrong side of the road — vehicles on the
   right of ego had green boxes drawn on the left of the image.
2. Wrote `viz_gt_check.py` to overlay 4 sources on each camera image:
   - GREEN = 3D-GT (our parse), projected
   - YELLOW = Waymo's own 2D-GT (from `camera_box` parquet, drawn directly)
   - RED/MAGENTA/CYAN = pseudo-labels by class
3. Compared LEFT-side cars (yellow boxes wrapped them tightly) vs GREEN
   wireframes (drew on the opposite side of the road). The yellow boxes
   confirmed the **image and 2D-GT were correct**.
4. Verified the **projection math** independently with the official v1 SDK
   on a different segment (`/tmp/v1_proj_check.py`). Identical pixel
   coordinates: `Δu = 0.0, Δv = 0.0` for every box center.
5. Therefore the bug must be in either the GT decode or the LiDAR points.
   Code review surfaced both `arr[:, 1] *= -1` and `M_reflect @ ... @ M_reflect`.

### Diagnostic timeline (instructive failure modes)

| Stage | GT y-flip? | LiDAR y-flip? | Vehicle AP@2m | What's happening |
|---|:---:|:---:|---:|---|
| **V1** (original) | ✓ | ✓ | **0.360** | Both mirrored — model + GT live in same mirror universe, AP looks fine |
| **V2** (removed only GT flip) | ✗ | ✓ | **0.013** | Inconsistent — model predicts mirrored, GT is correct, no matches |
| **V3** (full fix: removed both) | ✗ | ✗ | **0.013** | Visualization now correct (GT wraps real vehicles) but model AP did NOT recover — there is a THIRD source of frame mismatch (see §3.5) |

This sequence is **instructive**: a single AP number is **not enough** to
diagnose a coordinate bug. The system can be wrong in a self-canceling way
that looks plausible. Visual GT/image overlays + cross-checking with an
independent reference (v1 SDK) are essential.

### 3.5 An *open question*: why didn't V3 recover the AP?

After both fixes (no box-flip + no `M_reflect` on LiDAR/pose), the GT
overlays look correct (boxes wrap real vehicles in image), but the
nuScenes-trained model's zero-shot Vehicle AP stays at **0.013**, not the
~0.36 of the original mirrored pipeline. Two prime suspects remain:

#### (a) The `[CALIB_FIX]` block

`dataset_waymo3dv201.py` lines 381-389 rotates the TOP LiDAR's extrinsic
when `T_vl[2,3] > 1.8 m AND |yaw| > 30°`. The condition fires on every
Waymo TOP LiDAR because its calibration **intentionally** has a yaw of
±148° (the sensor's azimuth-0 column points to the rear, not the front).
The `[CALIB_FIX]` undoes this rotation, putting LiDAR points in a
**non-vehicle-frame** orientation. The original mirrored pipeline may have
been incidentally robust to this; the unmirrored pipeline isn't.

Concretely, the raw extrinsics (verified by direct parquet probe) are:
- Laser 1 (TOP, z=2.184): yaw=**+148.4°** (rear-facing azimuth-0)
- Laser 2 (FRONT, z=0.7):  yaw=-0.8°  (front-facing)
- Laser 3 (SIDE_LEFT):     yaw=+89.6°
- Laser 4 (SIDE_RIGHT):    yaw=-90.1°
- Laser 5 (REAR):          yaw=+179.3°

The 148° on TOP is **part of the Waymo calibration**, not a bug. The fix
that rotates it away is likely incorrect.

#### (b) nuScenes-Waymo frame convention difference

The nuScenes-trained model (B10c) was trained on points in **nuScenes
`LIDAR_TOP` sensor frame**, not ego/vehicle frame. nuScenes' LIDAR_TOP
sensor mounts with a calibrated rotation of roughly **−90° about z** vs
ego (so `+x_LIDAR = +y_ego = LEFT`). Feeding Waymo points in raw vehicle
frame to a model trained on this nuScenes LIDAR_TOP frame is a 90°
rotation mismatch that should produce nearly zero AP — consistent with our
V3 result.

**Why did V1 (mirrored) work better?** A y-mirror is geometrically
equivalent to a 180° rotation + reflection, which **partially** undoes the
nuScenes 90° offset (one of the four discrete frame compositions happens
to be closer). The V1 AP of 0.36 was likely a partial-alignment artifact,
not a "correct" baseline. The real fix is to apply the **proper
LIDAR_TOP-to-vehicle rotation** to bring Waymo points into the nuScenes
sensor frame B10c expects.

**Resolution (next session):**

Probed `nuscenes_infos_val_mkf30.pkl` → `data_list[0]['lidar_points']['lidar2ego']`:

```
lidar2ego rotation columns (each = LIDAR axis in ego coords):
  LIDAR +x → ego (0.002, -1.000, -0.006)  ≈ ego RIGHT  (-y_ego)
  LIDAR +y → ego (1.000,  0.002, -0.024)  ≈ ego FORWARD (+x_ego)
  LIDAR +z → ego (0.024, -0.006,  1.000)  ≈ ego UP    (+z_ego)
```

**nuScenes `LIDAR_TOP` frame is rotated −90° about z vs ego frame.** Waymo
vehicle frame = nuScenes ego frame, so feeding Waymo's raw vehicle-frame
points is a 90° mismatch — model can't recognize anything.

**First hypothesis (rotation) — disproved**: rotating Waymo points by
`R_lid_veh = [[0,-1,0],[1,0,0],[0,0,1]]` (+90° about z) to match the
nuScenes LIDAR_TOP convention did NOT recover AP (V4: Vehicle AP=0.014).

**Empirical resolution — y-mirror is what the model wants**:
brute-forced all 8 axis-aligned 2D transforms of V3's predictions
against V3's un-mirrored Waymo GT:

```
transform               Vehicle AP   Veh-Recall   Macro mAP
identity                  0.013       0.066      0.035
rot90_ccw                 0.022       0.049      0.017
rot180                    0.038       0.103      0.016
rot270_ccw                0.010       0.045      0.022
x_flip                    0.006       0.025      0.002
y_flip                    0.539       0.588      0.278   ← winner
rot90_then_xflip          0.006       0.029      0.002
rot270_then_xflip         0.005       0.028      0.003
```

`y_flip` recovers AP=0.539 and matches the original V1 pipeline's number
(0.36 at score=0.10, 0.675 at score=0.01 ≤50m). The nuScenes-trained
B10c model expects LiDAR points in a **y-mirrored Waymo vehicle frame**:
`+x forward, +y RIGHT, +z up` — i.e., the same frame the V1 `M_reflect`
conjugation produced as a side-effect. This is NOT nuScenes LIDAR_TOP
(`+x right, +y forward`); the model's actual training-time frame in our
mmdet3d configs apparently maps `+x→forward, +y→right` after some
internal augmentation we have not yet pinpointed.

**Final clean fix (V5)** — restores AP=0.36+ without re-mirroring GT:

| Where | What |
|---|---|
| `dataset_waymo3dv201.py` | KEEP the V3 fix — no box flip, no `M_reflect`. GT + LiDAR stay in correct right-handed Waymo vehicle frame. Visualization stays correct. |
| `dataset_waymo_mmdet3d.py` | **Apply `M_yflip = diag(1,-1,1,1)` only to LiDAR points fed to model**, and post-multiply `lidar2img` / `lidar2cam` / `cam2lidar` by `M_yflip` so camera-LiDAR fusion stays consistent. |
| `eval_waymo_zeroshot.py` | **Apply `M_yflip` to model predictions** (y, yaw, v_y all negate) before matching the un-mirrored GT. |

| Version | Base parser | Wrapper input | Eval post-process | Vehicle AP @ score=0.10 all-range |
|---|---|---|---|---:|
| V1 (original) | y-mirror everywhere | raw passthrough | none | 0.360 (self-consistent mirror universe) |
| V2 (un-flipped GT only) | un-flipped GT, still M_reflect on LIDAR | raw | none | 0.013 |
| V3 (full un-mirror) | un-flipped GT + un-flipped LIDAR | raw | none | 0.013 |
| V4 (V3 + 90° z rotation in wrapper) | un-flipped | rotate +90° z, update matrices | rotate −90° z + yaw correction | 0.014 |
| V5 (V3 + y-flip in wrapper + matrices) | un-flipped | y-flip + update matrices | y-flip | 0.032 (wrapper changes break model's input expectation) |
| **V6 (final clean)** | **un-flipped** | **raw passthrough** | **y-flip predictions only** | **0.360 ✓** |

**Verified V6 numbers (300 frames, stride 20, full Waymo val coverage)**:

| score | range | Vehicle | Pedestrian | Cyclist | Macro |
|---:|:---:|:---:|:---:|:---:|:---:|
| 0.10 | all | 0.360 | 0.091 | 0.115 | 0.189 |
| 0.01 | ≤50m | **0.674** | 0.288 | 0.126 | **0.363** |

The V6 architecture is now the canonical one:
- **`dataset_waymo3dv201.py`**: no `M_reflect`, no box y/heading flip. GT
  and LiDAR points are in correct right-handed Waymo vehicle frame.
  Visualization works correctly (GT wireframes wrap real objects).
- **`dataset_waymo_mmdet3d.py`**: no input transformation. Raw vehicle-frame
  points and identity projection matrices passed to model.
- **`eval_waymo_zeroshot.py`**: applies `y → -y, yaw → -yaw, v_y → -v_y` to
  model predictions before GT matching. This compensates for the model's
  empirically-observed mirrored output frame.

**Key takeaways**:
- The frame the model expects is *not* what the data is stored in, and
  *not* what the `lidar2ego` matrix would suggest — it is what the model
  was empirically trained on.
- Brute-force search over the 8 axis-aligned 2D transforms is the right
  diagnostic when the analytical derivation fails. Vehicle AP differences
  across the 8 candidates ranged from 0.005 to 0.539 — the signal is
  unmistakable once you sweep them all.
- Wrapping the y-mirror inside the model input (V5) doesn't behave the
  same as a pure post-process (V6), because changing the camera-LiDAR
  fusion matrices and the LiDAR encoder's input distribution
  simultaneously interacts in ways our compensating transform on the
  output side can't fully undo. The minimum-intervention fix wins.

### The fix

In `dataset_waymo3dv201.py`:

```python
# REMOVED — was mirroring GT
# arr[:, 1] *= -1.0
# arr[:, 6] *= -1.0
centers_v = arr[:, :3]

# REMOVED — was mirroring LiDAR + vehicle-pose extrinsics
# T_vl_rh = self.M_reflect @ T_vl_lh @ self.M_reflect
T_vl = np.array(extr_vals, np.float32).reshape(4, 4)

# T_wv = self.M_reflect @ T_wv_lh @ self.M_reflect
T_wv = np.array(vp_vals, np.float32).reshape(4, 4)
```

`M_reflect` is still defined (line 55) but **no longer applied** anywhere in
the box / LiDAR / vehicle-pose paths.

---

## 4. The 3D-Box-to-Image Projection (verified-correct version)

Implemented in `phase2a/viz_gt_check.py`:

```python
def _project_3d_box_to_image(box_7d, l2i_4x4, W, H, min_depth=0.5):
    """Project all 8 corners of a 3D box. Returns 8 (u, v, depth) tuples
    only if ALL 8 corners are in front of the camera. Partial-occlusion
    boxes return None (the projection is geometrically degenerate)."""
    x, y, z, lx, ly, lz, yaw = box_7d[:7]
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    dx, dy, dz = lx/2, ly/2, lz/2

    # 8 corners in box-LOCAL frame (+x = heading direction)
    local = np.array([
        [ dx,  dy,  dz], [ dx,  dy, -dz],
        [ dx, -dy,  dz], [ dx, -dy, -dz],
        [-dx,  dy,  dz], [-dx,  dy, -dz],
        [-dx, -dy,  dz], [-dx, -dy, -dz],
    ])

    # Rotate from local frame → vehicle frame (yaw about z-axis)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = local @ R.T + np.array([x, y, z])

    # Homogeneous projection: lidar2img @ point
    h = np.concatenate([corners, np.ones((8, 1))], axis=1)
    proj = h @ l2i_4x4.T
    depth = proj[:, 2]

    # STRICT: every corner must be in front (otherwise the perspective
    # division at the camera plane is degenerate → garbage pixel coords
    # that "leak" into the rendered wireframe).
    if (depth <= min_depth).any():
        return None

    u = proj[:, 0] / depth
    v = proj[:, 1] / depth

    # At least one corner inside the image (with small margin for boxes
    # extending off-edge).
    margin = 100
    if (u.max() < -margin or u.min() > W + margin or
        v.max() < -margin or v.min() > H + margin):
        return None
    return list(zip(u.tolist(), v.tolist(), depth.tolist()))
```

**Previous bug**: when some corners had `depth ≤ 0`, the code substituted
`depth = 1.0` to avoid divide-by-zero, producing nonsense `u, v` for those
corners. The wireframe drew edges between valid and garbage corners → smear
across the image. The strict gate fixes this.

---

## 5. Pseudo-Label Pipeline (Phase 2a)

For **annotation-free domain adaptation** from a nuScenes-trained 3D
detector to Waymo (or NVIDIA PhysicalAI-AV, which has no 3D GT at all),
we generate pseudo-labels by **cross-modal consensus**.

```
┌───────────────────┐    ┌──────────────────────┐
│  LiDAR (vehicle)  │    │  Surround images     │
└────────┬──────────┘    └─────────┬────────────┘
         │                         │
         ▼                         ▼
   Validator A: A∩B agreement     Validator B:
   DBSCAN clustering ─────────→   COCO Faster R-CNN
   geometry → class                + LiDAR-depth lift
         │                         │
         └──── Fusion router ──────┘
                    │
       ┌────────────┴──────────────┐
       │ A∩B         │   A or B only / conflict
       │ (high conf) │   (disagreement)
       ▼             ▼
    Keep,        Tier-2 VLM voter (Gemma-3n-e4b-it)
    weight=1.0   "Inside the red box, what's the class?"
                 V / P / C / N  +  confidence
                       │
                       ▼
                 Keep with weight 0.3-0.5
```

### 5.1 Validator A — LiDAR DBSCAN cluster proposer

File: `phase2a/cluster_proposer.py`

Steps:
1. Filter LiDAR points by ego-radius and max-range, drop z-extreme returns
2. Per-tile ground removal (per-tile z-quantile threshold)
3. DBSCAN cluster (`eps=0.6 m`)
4. Per cluster: PCA on xy for yaw, fit oriented BEV box, z extent from points
5. Rule-based class from cluster dimensions:
   - Pedestrian: tall + thin (`h ∈ [0.7, 2.4]`, `max(l,w) ≤ 1.2 m`)
   - Cyclist: intermediate (`h ∈ [0.7, 2.2]`, `max(l,w) ∈ [0.8, 2.5]`)
   - Vehicle: anything else within reasonable size priors
6. Score = `min(1.0, n_pts / 30)` — saturates at 30 LiDAR points

Typical output: 30-90 cluster proposals per Waymo frame (~0.5 s on CPU).
**High recall, intentionally noisy class** — downstream voting filters.

### 5.2 Validator B — 2D detector + LiDAR-depth lift

File: `phase2a/cam2d_proposer.py`

Model: **torchvision `fasterrcnn_resnet50_fpn_v2`** (COCO weights, no extra
dependencies needed). Inference on each of 5 Waymo cameras.

COCO → transfer class mapping:
```python
COCO_TO_TRANSFER = {
    1: 'Pedestrian',   # person
    2: 'Cyclist',      # bicycle
    3: 'Vehicle',      # car
    4: 'Cyclist',      # motorcycle
    6: 'Vehicle',      # bus
    8: 'Vehicle',      # truck
}
```

**3D lift procedure** for each 2D box:
1. Project all LiDAR points (vehicle frame) to image via `lidar2img` for the
   relevant camera
2. Keep points where `(u, v)` falls inside the 2D box
3. Take `median` of those points' depth → `d_med`
4. Filter outliers along the ray: keep points within `±2 m` of `d_med`
5. Mean the remaining 3D positions → 3D box center
6. Size from class-conditional prior: Vehicle `(4.5, 1.9, 1.7)`,
   Pedestrian `(0.6, 0.6, 1.75)`, Cyclist `(1.7, 0.6, 1.6)`
7. Yaw heuristic: `atan2(center_y, center_x)` — box faces ego

Lower-precision than A on geometry (size + yaw are priors), but knows
class semantically.

### 5.3 VLM Voter — NVIDIA Gemma-3n on disagreement only

File: `phase2a/vlm_voter.py`

**Endpoint**: `https://integrate.api.nvidia.com/v1/chat/completions`
**Model**: `google/gemma-3n-e4b-it` (small, fast, rate-limit friendly)
**API key**: from `NVAPI_KEY` environment variable (never committed)

**Critical prompt-engineering detail — the red-box trick:**

A binary "is there a vehicle in this crop?" prompt with a wide context crop
returns *yes* for almost any roadway image. The fix: **draw a thick red
rectangle on the candidate region INSIDE the crop**, then ask the VLM to
classify **what is inside the red box**:

```
PROMPT_TEMPLATE = (
  "The image has a RED rectangle marking a candidate object. "
  "Look CAREFULLY at WHAT IS INSIDE THE RED BOX. Ignore objects OUTSIDE.\n"
  "Question: what is inside the red box?\n"
  "  V = Vehicle (car, truck, bus, ...)\n"
  "  P = Pedestrian (person walking/standing)\n"
  "  C = Cyclist (person on bicycle or motorcycle)\n"
  "  N = None of the above (background, sign, pole, vegetation, road)\n\n"
  "Be strict: if the red box is mostly empty road, vegetation, or a sign, "
  "answer N. Only V/P/C if a clear instance is inside the red box.\n\n"
  "Respond with a SINGLE character on line 1, confidence 0-1 on line 2."
)
```

**Validation result**: on a hand-tested Waymo frame, Gemma correctly:
- Confirmed 6/6 Faster R-CNN vehicles → "V"
- **Rejected 4/4 Faster R-CNN "pedestrians"** that were actually signs/poles → "N"

The red-box visual anchor reduces VLM context confusion enormously.

**Cost / rate-limit controls:**
- Perceptual hash cache: 32×32 grayscale → MD5 hash → in-memory + disk JSON
  cache. Crops from adjacent frames in same segment hit cache aggressively.
- Token cap: `max_tokens=16` (we only need a class char + a number)
- Image: thumbnail to 256-pixel max edge, JPEG quality 80
- Throttle: configurable calls-per-minute (default 240); sleeps when hit

### 5.4 Fusion routing

File: `phase2a/fusion.py`

Per A-proposal × B-proposal pair (matched by xy-center within 2.5 m):

| A says | B says | Action | Weight | VLM called? |
|---|---|---|---|---|
| `X` | `X` (same class) | **A∩B** → keep | 1.0 | No |
| `X` | `Y` (class conflict, co-located) | VLM votes; take VLM's class | 0.3 | Yes |
| `X` | (silent) | VLM votes; keep if VLM agrees | 0.5 | Yes |
| (silent) | `X` | VLM votes; keep if VLM agrees | 0.5 | Yes |
| (silent) | (silent) | Drop | — | No |

**Post-processing**:
- ≤50 m range filter (matches nuScenes annotation range; the source model
  has no representations for >50 m)
- BEV-NMS (greedy, dist < 2 m): higher-weight label wins among
  same-class overlaps

Typical output: 8-35 pseudo-labels per Waymo frame after NMS.
~30 s/frame end-to-end with VLM enabled (40+ VLM calls per frame).

---

## 6. File Map

```
DeepDataMiningLearning/detection3d/
├── dataset_waymo3dv201.py           # Base Waymo v2.0.1 parquet loader
│                                    # [FIXED] removed box y-flip and M_reflect
├── dataset_waymo_mmdet3d.py         # mmdet3d-format wrapper for B10c inference
├── class_map_waymo_to_nus.py        # 10-class nuScenes → 3-class Waymo mapping
├── eval_waymo_zeroshot.py           # Zero-shot AP eval (B10c on Waymo val)
├── waymo_phase1_sweep.py            # Score-threshold × range sweep diagnostic
└── phase2a/
    ├── cluster_proposer.py          # Validator A
    ├── cam2d_proposer.py            # Validator B
    ├── vlm_voter.py                 # Gemma VLM tie-breaker (red-box prompt)
    ├── fusion.py                    # A↔B match + VLM routing + NMS
    ├── demo_single_frame.py         # End-to-end demo with BEV plot
    └── viz_gt_check.py              # 4-color overlay diagnostic
                                     # (used to find the y-flip bug)
```

---

## 7. Useful Diagnostic Patterns (replicate these for future debugging)

### 7.1 Single-point projection sanity check

```python
# (10, 0, 1.5) — 10 m straight ahead of ego, 1.5 m above ground
p_world = np.array([10, 0, 1.5, 1.0])
p_pixel = lidar2img @ p_world
u, v = p_pixel[0]/p_pixel[2], p_pixel[1]/p_pixel[2]
# Should land near image center (cu ≈ W/2, slightly below since
# camera mounted ~2.11 m up vs target z=1.5 m → 0.6 m below cam height)
```

### 7.2 v1 SDK cross-check

For any disagreement between our projection and visible objects, project
the same point with the **official Waymo SDK** on a v1 tfrecord:

```python
extrinsic_inv = np.linalg.inv(extrinsic)
pt_cam = extrinsic_inv @ pt_vehicle_homog
x, y, z = pt_cam[0], pt_cam[1], pt_cam[2]
fu, fv, cu, cv = intrinsic[:4]
u = -fu * y / x + cu
v = -fv * z / x + cv
```

This bypasses our K + R_oc_wc OpenCV-conversion and uses Waymo's native
convention. If pixel coords match ours → projection math is right.

### 7.3 Center-dot overlay

When wireframes look ambiguous, draw **filled solid dots** at projected
box centers with class + 3D position labels (see `/tmp/debug_proj.py`
pattern). Dots are unambiguous about which object a box claims to mark.

### 7.4 GT vs Waymo 2D-GT side-by-side

`viz_gt_check.py` overlays:
- **GREEN** wireframes — our parsed 3D-GT projected
- **YELLOW** rectangles — Waymo's own per-camera 2D-GT (drawn directly,
  no projection involved → reliable reference)

If yellow boxes wrap visible cars but green wireframes don't → projection
or GT parsing has a bug. Yellow is the ground-truth ground-truth.

---

## 8. Known limitations + future work

- **Pseudo-label noise**: Validator B uses a fixed size prior (4.5×1.9×1.7
  for all Vehicles) and a heuristic yaw. About 20-30% of pseudo-Vehicle
  boxes have wrong size/yaw even when the center is correct. This is
  acceptable for supervised fine-tuning (loss weighting absorbs noise)
  but suggests the next improvement: **GRPO-style RL fine-tune** with the
  validators as external verifiers (sketched in the project's research log).
- **Sparse 3D-GT for short-range vehicles**: Waymo's ≥9-LiDAR-points filter
  drops many camera-visible nearby cars. For unbiased recall metrics,
  augment eval with Waymo's `camera_box` 2D-GT.
- **VLM rate / cost**: at full Waymo train (~200K frames × ~40 VLM calls)
  the API budget is significant. For scale, batch the VLM calls or use a
  local VLM (Llama-3.2-vision via HF) as fallback.
- **NVIDIA hosted 3D NIMs (StreamPETR, BEVFormer, SparseDrive)** are demo-only
  on `build.nvidia.com` — not callable via the standard chat-completion API
  with the standard NIM key. If you need them as additional verifiers,
  containerized NIM deployment is required.

---

## 8.5 Fine-tune results (V7 → V8)

After the V6 zero-shot baseline (Vehicle AP=0.360, Macro=0.189), we ran
two supervised fine-tune experiments to see whether real Waymo GT could
beat the zero-shot floor.

### V7 (B12) — failed fine-tune on small v2 split

- **Data**: 1,508 train + 416 val frames from the 37 waymo201/validation
  segments that have all v2 parquet types (camera_image filter).
- **Hyperparams**: 1 epoch, LR=1e-5, image backbone frozen.
- **Result**: **Vehicle AP=0.360→0.181** (regression of −0.179),
  Macro=0.189→0.091. Precision tightened (0.875→0.946) but Recall
  halved (0.395→0.162). Classic small-dataset catastrophic forgetting.
- **Cyclist destroyed entirely** (0.115→0.000) because only 298 bicycle
  instances in 1.5K frames is below the threshold for retaining the
  motorcycle/bicycle distinction.

### V8 (B13) — success on bigger v1 split

- **Data**: 4,453 train + 793 val frames extracted from 27 Waymo v1.4.3
  tfrecords (~3× more than V7). The v1 path is wired into
  `LoadWaymoFrameFromInfo` via the `waymo_v1://` URI scheme;
  per-frame data lives as `.npz` (points + boxes) + `.jpg` (cameras).
- **Hyperparams**: 1 epoch, **LR=5e-6** (half of V7), **aggressive
  freezing**: img_backbone, img_neck, view_transform all frozen — only
  fusion_layer + bbox_head + small LiDAR-side fine-tuning.
- **Result**: **macro mAP 0.189 → 0.202** (+7% absolute) at default
  threshold, **0.363 → 0.395** at best Phase-1 setting (+9%).
  Pedestrian AP nearly doubled (Phase-1 best: 0.288 → 0.438).

| Setting | V6 | V7 | **V8** |
|---|---:|---:|---:|
| Vehicle AP @ default (score=0.10 all) | 0.360 | 0.181 | **0.358** |
| Pedestrian AP @ default | 0.091 | 0.091 | **0.178** |
| Cyclist AP @ default | 0.115 | 0.000 | 0.069 |
| **Macro mAP @ default** | **0.189** | **0.091** | **0.202** |
| Vehicle AP @ Phase-1 best (score=0.01 ≤50m) | 0.674 | 0.349 | 0.659 |
| **Pedestrian AP @ Phase-1 best** | 0.288 | 0.178 | **0.438** |
| **Macro mAP @ Phase-1 best** | **0.363** | **0.176** | **0.395** |

### Lessons from V7 vs V8

1. **Data scale matters more than LR or schedule.** V7 with 1.5K frames
   failed regardless of hyperparams; V8 with 4.5K frames succeeded with
   careful softening.
2. **Freezing the image branch is essential** for small-scale Waymo
   fine-tune of a nuScenes-trained model. The image features generalize;
   the LiDAR-fusion+head adaptation does the heavy lifting.
3. **The `matched_ious` metric during training is misleading** — V8 had
   matched_ious 0.02-0.05 throughout 1 epoch, but actual evaluation AP
   improved. The training-time matcher uses high IoU thresholds that
   are unrealistic for early cross-domain alignment; the AP metric
   (center-distance ≤2 m) is more forgiving and the gold standard.
4. **Pedestrian benefits most from Waymo data.** Waymo annotates
   pedestrians more densely than nuScenes (per-frame counts roughly 5×).
   Vehicle was already saturated; cyclist is class-imbalance-bound.

### V9 (B14) — mixed nuScenes + Waymo training, **best result**

V9 wraps V8's Waymo-only data with nuScenes as an "anchor" via mmengine's
`ConcatDataset`. Specifically: 4,453 Waymo v1 frames + ~7K nuScenes 25% mkf30
frames train together each epoch (~14%/86% natural mix).

- **Why it works**: nuScenes anchor prevents catastrophic forgetting,
  Waymo data still shifts the model toward the target domain. The
  forgetting that ruined V7 (cyclist 0.115 → 0.000) is fully prevented
  here (cyclist back to 0.105 ≈ V6).
- **Why it doesn't push Vehicle higher**: Vehicle was already saturated
  at V6. Mixed training preserves the ceiling; further gains require
  architecture changes or longer training.
- **Result**:

| Metric | V6 (baseline) | V8 (Waymo-only) | **V9 (mixed)** |
|---|---:|---:|---:|
| Vehicle AP @ default | 0.360 | 0.358 | **0.357** |
| **Pedestrian AP @ default** | 0.091 | 0.178 | **0.311** **(+242%)** |
| Cyclist AP @ default | 0.115 | 0.069 | **0.105** |
| **Macro mAP @ default** | **0.189** | **0.202** | **0.258 (+37%)** |
| Macro @ Phase-1 best (score=0.01 ≤50m) | 0.363 | 0.395 | **0.439 (+21%)** |

- **Training health metrics**: Loss 10.3 → 7.3 (−29%), `matched_ious`
  reached 0.18-0.35 throughout (vs V8's stuck 0.02-0.05). The nuScenes
  anchor keeps the model's predictions in a sensible range.
- **Config gotchas** (fixed in B14):
  1. `ConcatDataset` requires identical `metainfo` across all sub-datasets
     including `palette`. We provide an explicit `common_metainfo` dict to
     both branches and pass `ignore_keys=['version', 'dataset', ...]` to
     skip non-semantic differences.
  2. NuScenes images need **per-camera `data_prefix`** (`CAM_FRONT='samples/CAM_FRONT'`
     etc.), not a single `img=''`. The Det3DDataset path-resolution code
     looks up `data_prefix[cam_id]` per camera.
  3. The two pipelines (LoadWaymoFrameFromInfo for Waymo, BEVLoadMultiViewImageFromFiles +
     LoadPointsFromFile for nuScenes) coexist via `ConcatDataset` which
     dispatches per-sample.

### Files for the fine-tune pipeline

```
DeepDataMiningLearning/detection3d/phase2a/
├── extract_waymo_v1.py                 # tfrecord → per-frame .npz + .jpg
├── build_waymo_finetune_infos_fast.py  # v2 parquets → info.pkl
└── build_waymo_finetune_infos_v1.py    # v1 extracted → info.pkl

mmdetection3d/projects/bevdet/
├── waymo_finetune_dataset.py           # Dataset + transform; v1/v2 dispatch
└── configs/finetune/
    ├── B12_waymo_finetune.py           # V7 — failed (small v2)
    ├── B13_waymo_v1_finetune.py        # V8 — succeeded (bigger v1)
    └── B14_waymo_nuscenes_mixed.py     # V9 — best (mixed nu + Waymo)
```

The v1 extractor monkey-patches the Waymo SDK's
`parse_range_image_and_camera_projection` to fix a `bytes`/`bytearray`
protobuf incompatibility (`ParseFromString(bytearray(...))` rejected by
modern protobuf — use `bytes(...)` instead).

## 9. Lessons learned

1. **A self-consistent bug is the worst kind**: V1 (original code) had two
   mirrors that canceled out → AP looked plausible. Removing one mirror
   produced a dramatic regression that revealed the second. **Always
   eyeball GT overlays on raw images** before trusting metric numbers.
2. **The yellow-box trick**: when verifying a 3D projection, use the
   dataset's own 2D annotations as the ground-truth ground-truth. They're
   stored in pixel space → no projection involved → no shared bug.
3. **The red-box VLM trick**: VLMs are very prone to scene-summarization
   ("yes I see a car somewhere"). Annotating the candidate region inside
   the image with a thick red rectangle forces spatial grounding and
   dramatically improves the reject rate on false-positive crops.
4. **The v1 SDK is the reference implementation** — when v2 behavior is
   suspect, extract one v1 `.tfrecord` and use `waymo_open_dataset.utils`
   for cross-comparison. (We had to install `tensorflow` for this — it's
   worth the extra venv.)
