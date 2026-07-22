# Plan: cross-dataset labeled-occupancy pretraining pool (PhysicalAI + Waymo/AV2 → nuScenes det)

**Motivation (from the seeded label-efficiency result, 2026-07-21).** LABEL-BASED occ pretraining
(Occ3D-GT) gives **+35% detection mAP at 2k labels** (0.163 vs 0.121 scratch), while the LABEL-FREE
voxel-soft pretext is null. So occ→detection transfer is real but **teacher-quality-bounded** — the
lever is *more/better labeled occ pretraining data*. This plan scales the labeled occ pretraining
pool cross-dataset, then measures the lift on the nuScenes low-label detection curve.

## Target: one unified occ-label format (all datasets convert to this)

Match Occ3D-nuScenes so the existing trainer/eval/teacher code applies unchanged:
- grid **200×200×16**, voxel **0.4 m**, ego-frame, `PC_RANGE = [-40,-40,-1, 40,40,5.4]`.
- per frame: `semantics (200,200,16) uint8` (classes 0–16 + **17=free**) + `mask (200,200,16) bool`
  (supervise where observed — camera- or lidar-visible), saved as `labels.npz` (same as our teacher
  targets / Occ3D GT). Reuses `gaussian4d/teachers/*` (voxelize, raycast free-space) and the occ
  trainer (`occupancy/train_lss.py`) + evaluator with zero changes.

## Per-frame generation recipe (dataset-agnostic core)

1. **Geometry** — LiDAR points → ego frame → voxelize (`teachers.base.voxel_indices`) → occupied.
2. **Foreground semantics** — 3D boxes → point-in-box test → assign the box's class to points/voxels
   inside it (car/truck/bus/…, mapped to the Occ3D/nuScenes-10 foreground ids). This is the
   label-based signal that carries the detection-transfer benefit.
3. **Background semantics** — occupied voxels not in any box:
   - if the dataset has **LiDAR point semseg** → map to Occ3D background (drivable/sidewalk/terrain/
     manmade/vegetation);
   - else → **FM projection** (project points into cameras, sample cached SegFormer+GSAM like our
     `ngdet/labelgen`) OR a single coarse `manmade/other` background class for a v1 pool.
4. **Free space** — ray-cast from the LiDAR sensor origin (`teachers.raycast.free_space_mask`) →
   free voxels (weight = small); unobserved = ignore.
5. **Class taxonomy map** — each dataset's vocab → the 18-class Occ3D space; foreground alignment to
   the **nuScenes-10** det classes is what matters for transfer (car, truck, bus, trailer,
   construction_vehicle, pedestrian, motorcycle, bicycle, barrier, traffic_cone; note AV2/PhysicalAI
   name variants, e.g. cyclist=`rider`).

**Design choice to test:** v1 = **foreground-from-boxes + geometry + free** (no full background
semantics) — cheapest, and detection cares about foreground; if that already lifts the curve, full
background semantics is optional. v2 adds background (semseg or FM).

## Datasets (inventory — to be filled by the running scoping pass)

- **Argoverse 2 (AV2) sensor** — `…/thesis-nurec/data/argoverse2_raw/sensor/{train,val}`. **train 100
  logs / 15,664 sweeps; val 50 / 7,881**. LiDAR `sensors/lidar/<ts_ns>.feather` (x,y,z,intensity,
  **ego frame**, motion-compensated). Boxes `annotations.feather` (ego frame; `timestamp_ns` matches
  lidar filename; center tx/ty/tz, size l/w/h, quat) — **28 categories**. Calib/pose:
  `egovehicle_SE3_sensor.feather`, `intrinsics.feather`, `city_SE3_egovehicle.feather`. 9-cam pinhole.
  Feather = plain `pd.read_feather` (no `av2` devkit needed; devkit NOT installed). **No LiDAR semseg.**
- **PhysicalAI** — two copies: (A) self-contained NCore packed-zarr
  `…/thesis-nurec/data/pai/ncore/clips` (**1,147 clips**, `.zarr.itar` = tar-of-zarr; LiDAR as
  direction×distance ray-bundle in rig frame; obstacle autolabels embedded in cuboids `.zattrs`); (B)
  HF-parquet mirror `/fs/atipa/data/rnd-liu/Datasets/PhysicalAI-AV/` (DracoPy LiDAR, egomotion +
  calib parquet local; `obstacle.offline` boxes fetched from HF on demand). **9 cuboid classes**
  (rider=cyclist). Reusable primitives in `PhysicalAI-Drive/…/pseudolabel_physicalai/`:
  `annotate_clip.py` (load_calib/decode_lidar/ftheta_project), `build_obstacle_dataset.py`
  (`gt_boxes_at`, `yaw_of`), `eval_cyclist.py` (extract_lidar/load_gt). env **py312**
  (pandas/pyarrow/DracoPy). **No LiDAR semseg.**
- **Waymo v2.0.1** — data at `/fs/atipa/data/rnd-liu/Datasets/waymo201/validation` (**201 segments ≈
  40k frames**; parquet: `lidar/ lidar_box/ lidar_calibration/ vehicle_pose/ camera_*`). **Reusable
  loader IN OUR REPO**: `DeepDataMiningLearning/detection3d/dataset_waymo3dv201.py` →
  `Waymo3DDataset` returns per frame: LiDAR `(N,5)` **in vehicle frame** (+X fwd/+Y left/+Z up),
  boxes_3d `(M,7)` [x,y,z,dx,dy,dz,yaw] vehicle frame, `labels` (1=Vehicle 2=Pedestrian 3=Sign
  4=Cyclist), `T_vl` (lidar→vehicle), `world_from_vehicle`, multi-sweep fusion. **This is the readiest
  ingestion source** — vehicle frame = ego, so LiDAR+box → occ grid is direct. Foreground map to
  nuScenes-10: Vehicle→car, Pedestrian→pedestrian, Cyclist→bicycle (Sign→drop/barrier). NOTE Waymo
  vehicle-frame Z origin (ground at rear axle) differs from nuScenes ego — set the grid z-range per
  dataset (or shift) so the 16 z-bins cover the right band.
- **Waymo v1.4.3** — 798 **training** tfrecords at
  `/fs/atipa/data/cmpe249-fa25/waymov143_individuals/training/` (~160k frames, larger pool) but raw
  tfrecords (needs `waymo-open-dataset`+TF to parse; the `kitti_format/` conversion is EMPTY). Use as
  a scale-up after v2-validation proves the pipeline. Also a v2-parquet finetune loader exists at
  `PhysicalAI-Drive/…/bevfusion/waymo/{dataset_waymo3dv201.py, waymo_finetune_dataset.py}`.

⟹ **v1 ingestion order: Waymo v2 (readiest, our loader) → AV2 → PhysicalAI.**

## PIVOTAL INSIGHT (from the inventory): none of Waymo/AV2/PhysicalAI has LiDAR semseg

All three give **LiDAR geometry + 3D boxes** but **no per-point semantics** — so a cheap cross-dataset
occ label is **foreground-from-boxes + geometry + free ONLY** (no road/building/vegetation stuff
classes). But the nuScenes arm that gave +35% used **full Occ3D-GT semantics**. So the whole
cross-dataset bet hinges on one untested question, cheaply answerable on nuScenes FIRST:

**STEP 0 (decisive, cheap, do before any ingestion):** build a nuScenes **foreground-only** occ target
(nuScenes 3D boxes → point-in-box foreground voxels + geometry + free; drop background stuff classes),
pretrain the occ model on it, and run the same low-label det transfer. 
- If foreground-only ≈ full-Occ3D (+35%) → the detection benefit comes from *object* occupancy, which
  cross-dataset boxes CAN provide → green-light Waymo/AV2/PhysicalAI ingestion.
- If foreground-only ≪ full-Occ3D → background stuff-semantics matter, and cross-dataset needs a
  per-dataset **camera-FM projection** stage (expensive) → reconsider scope before ingesting.

This turns a large speculative pipeline into a single cheap nuScenes ablation that de-risks it.

## Experiment (gated on Step 0)

1. **Step 0** above (nuScenes-only, reuses existing infra) — decides the whole direction.
2. If green: ingest **Waymo v2** (readiest, `Waymo3DDataset`) → unified foreground occ `labels.npz`
   (CPU/IO, no GPU), then AV2, then PhysicalAI. z-band per dataset (Waymo/AV2/PAI frame conventions).
3. Pretrain occ on **nuScenes-fg ∪ Waymo ∪ AV2 ∪ PhysicalAI** (mixed pool), camera-only, same arch.
4. Transfer to **nuScenes low-label detection** (`run_le_occ3d.sh` style: 2k/4k/8k, seeds) → does
   cross-dataset labeled-occ pretraining beat the nuScenes-only occ3d arm (0.163@2k)?
5. Figure: label-efficiency curves — scratch / nuScenes-occ3d / nuScenes-fg-only /
   **cross-dataset-occ** / voxel-soft.

## Open questions / gaps
- Background semantics without LiDAR semseg (FM projection cost vs coarse class).
- Cross-dataset domain gap (sensor layout, ego frame conventions, class definitions) — ego-frame
  voxel occ is fairly sensor-agnostic, but foreground class alignment needs care.
- Ego-grid orientation per dataset (AV2/PhysicalAI axis conventions vs nuScenes) — must match.
