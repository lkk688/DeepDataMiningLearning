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

- **Argoverse 2 (AV2) sensor** — `…/thesis-nurec/data/argoverse2_raw/sensor/{train,val}`. LiDAR +
  3D annotations (`annotations.feather`) + `av2` devkit. [formats/counts: pending inventory]
- **PhysicalAI** — `…/thesis-nurec/data/pai/{raw,…}`; loaders in `MyRepo/PhysicalAI-Drive`
  (worldmodel_drive, py312) which already parse LiDAR + `obstacle.offline` 3D autolabels.
  [formats/reusable loaders: pending inventory]
- **Waymo** — not present on this system; defer (needs download) or drop from v1.

## Experiment (the forward-looking scaling result)

1. Ingest AV2 + PhysicalAI → unified `labels.npz` occ pool (CPU/IO, no GPU).
2. Pretrain the occ model (`train_lss.py`) on **nuScenes-Occ3D ∪ AV2 ∪ PhysicalAI** (mixed pool),
   camera-only, same arch as the label-efficiency det backbone.
3. Transfer to **nuScenes low-label detection** (`run_le_occ3d.sh` style: 2k/4k/8k, seeds) →
   does cross-dataset labeled-occ pretraining beat the nuScenes-only occ3d arm (0.163@2k)?
4. Figure: label-efficiency curves — scratch / nuScenes-occ3d / **cross-dataset-occ** / voxel-soft.

## Open questions / gaps
- Background semantics without LiDAR semseg (FM projection cost vs coarse class).
- Cross-dataset domain gap (sensor layout, ego frame conventions, class definitions) — ego-frame
  voxel occ is fairly sensor-agnostic, but foreground class alignment needs care.
- Ego-grid orientation per dataset (AV2/PhysicalAI axis conventions vs nuScenes) — must match.
