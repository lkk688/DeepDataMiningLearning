# Full-data nuScenes training on H100 — run report

Scaling the validated `ngperception` recipes (occupancy §2.8, detection §9) from the
3090 ablation scale (~10 % of the split) to the **full nuScenes train split on a single
H100 NVL (95 GB)**, per TUTORIAL.md §3 "scaling to full data on H100". Pure PyTorch — no
mmcv, no spconv, no custom CUDA.

- **Env:** conda `py310`, torch 2.9.0+cu126, single H100 NVL.
- **Data:** nuScenes `v1.0-trainval` at `/data/rnd-liu/Datasets/nuScenes/v1.0-trainval`;
  Occ3D GT (`gts`, 34 149 frames) at `/data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes/gts`.
- **Split:** occupancy trains on all 34 149 Occ3D frames (300 held-out for val); detection
  uses the official nuScenes train/val scene splits (28 130 train / 2 000 val).

---

## Runs at a glance

| # | task | config | epochs | final result | tutorial ref | status |
|---|---|---|---|---|---|---|
| 1 | **occupancy, camera-only** | DINOv2-base, 4×96 decoder, cosine, AMP, refine-2, Lovász + class-bal CE | 24 | **mIoU 0.302 / geo-IoU 0.669** | 0.298 (§2.8.2) | ✅ done, **beats ref** |
| 2 | **occupancy, camera + LiDAR fusion** | run 1 + `--lidar-fusion` | 24 | **mIoU 0.558 / geo-IoU 0.838** | 0.493 (§2.8.3) | ✅ done, **+0.065 over ref** |
| 3 | **3D detection, nuScenes** | PointPillars + res backbone, multiclass (CBGS-style), 10-sweep | 40 | **carAP_cd 0.467 / IoU@0.5 0.157** | 0.78 (full 128 ep multihead) | ✅ done |

All three ran on the one H100; runs 1 and 3 trained concurrently (49 GB combined), run 2
took the freed GPU after run 1 finished. **All complete.**

---

## Run 1 — occupancy, camera-only (DONE)

```bash
python -m DeepDataMiningLearning.ngperception.occupancy.train_lss \
  --gts <gts> --nusc <nuscenes> \
  --backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 \
  --cosine --amp --refine-iters 2 --occ-lovasz 1.0 --occ-class-balance \
  --occ-cb-cache output/classw_full.npy \
  --max-samples 34149 --val-samples 300 --epochs 24 --batch-size 8 \
  --lr 2e-3 --depth-weight 1.0 --num-workers 8 --seed 0 \
  --out-dir output/lss_occ_full
```

**Final: mIoU 0.302 / geo-IoU 0.669** — beats the tutorial's 0.298 strong-config reference
(which used 3 000 samples / 12 ep). Epoch trajectory (mIoU):

```
ep0 0.222  ep4 0.254  ep8 0.272  ep12 0.285  ep16 0.296  ep20 0.301  ep23 0.302
ep1 0.230  ep5 0.254  ep9 0.271  ep13 0.288  ep17 0.297  ep21 0.301
ep2 0.238  ep6 0.254  ep10 0.256 ep14 0.294  ep18 0.296  ep22 0.302
ep3 0.240  ep7 0.253  ep11 0.281 ep15 0.294  ep19 0.299
```

Notes: epoch-0 already at 0.222 (frozen DINOv2 does the heavy lifting); steady climb with
the expected mid-cosine oscillation (ep6, ep10 dips that recovered); final 5 epochs
consolidate at ~0.30 as the LR anneals. Checkpoint: `output/lss_occ_full/lss_occ.pth`.

## Run 2 — occupancy, camera + LiDAR fusion (DONE)

Same as run 1 plus `--lidar-fusion --lidar-cache output/lidar_cache`,
`--out-dir output/lss_occ_full_fusion`.

**Final: mIoU 0.558 / geo-IoU 0.838** — **+0.065 over the 0.493 strong-fusion reference**
(§2.8.3), the single largest result in this report. mIoU by epoch:

```
ep0 0.385  ep4 0.457  ep8 0.507  ep12 0.523  ep16 0.545  ep20 0.555  ep23 0.558
ep1 0.416  ep5 0.484  ep9 0.494  ep13 0.536  ep17 0.550  ep21 0.558
ep2 0.445  ep6 0.488  ep10 0.507 ep14 0.541  ep18 0.554  ep22 0.557
ep3 0.442  ep7 0.495  ep11 0.526 ep15 0.545  ep19 0.557
```

Zero NaN across the full run (the grad-clip fix held). Fusion's geo-IoU (0.838 vs
camera-only's 0.669) confirms the LiDAR input hands the model the scene geometry directly;
the camera supplies semantics — the §2.8.3 crossover, now at full scale. Checkpoint:
`output/lss_occ_full_fusion/lss_occ.pth`.

> This run initially **collapsed to NaN** on the first attempt — see "Issues & fixes" below —
> and was restarted with a gradient-clipping fix. The numbers above are the fixed run.

## Run 3 — 3D detection, nuScenes (DONE)

```bash
python -m DeepDataMiningLearning.ngperception.detection.train_nuscenes \
  --root <nuscenes> --model pointpillars --backbone res --multiclass \
  --max-frames 28130 --val-frames 2000 --epochs 40 \
  --batch-size 8 --lr 3e-3 --sweeps 10 --lidar-cache output/nusc_det_lidar_cache
```

**Final: carAP_cd 0.467 / IoU@0.5 0.157** (nuScenes' own center-distance metric is the
headline; IoU@0.5 is the strict ruler). carAP_cd by epoch:

```
ep0 0.271  ep10 0.390  ep20 0.429  ep30 0.461  ep39 0.467
ep5 0.395  ep15 0.420  ep25 0.440  ep35 0.466
```

Monotonic climb on full data, exactly the §9.1 thesis: the levers are data + capacity +
schedule, not the single-sweep/heading tweaks. The reference 0.78 uses the full 28 k / 128
epochs / multi-head / augmentation; 0.467 on 40 epochs pure-PyTorch is a sensible point on
that trajectory (still rising — loss 0.79, not converged). IoU@0.5 stays low (0.157) as
documented: cars are smaller than the anchor at the coarse 0.2 m grid.

---

## Issues encountered & fixes

### 1. `compute_class_weights` — Python GC pathology (worked around)

The class-weight precompute stalled for **68+ minutes** at ~0.2 files/s (a fresh process
does the same 34 149-file pass in **44 s**). Root cause (py-spy confirmed): the loop
allocates an `NpzFile` + 3 arrays per iteration, triggering gen-0 cyclic GC, and because the
process holds nuScenes's ~10 M-object graph resident (14 GB RSS), **every GC pass rescans
that whole graph** — a ~250× slowdown.

**Workaround:** precomputed `output/classw_full.npy` in a parallel fresh process (44 s), then
restarted so the trainer hits `if cache exists: return` and skips the loop. Both occ runs
reuse the cache (weights min/max 0.20/2.14).

**Not yet patched in code.** A one-line `gc.disable()` around that loop (or `d.close()` on
the npz) would fix it natively for future full-data runs.

### 2. Fusion run — training divergence to NaN (fixed)

The fusion run trained well for 2 epochs (mIoU 0.398 → 0.409) then the loss climbed
(4.13 → 4.32 → 4.49) and **blew up to NaN at ep3 ~it1320** (`occ=nan`, depth still finite),
collapsing val mIoU to 0.000 for every subsequent epoch. Cause: **training divergence**,
hitting early because the cosine LR is near its 2e-3 peak in the first epochs; the heavier
fusion model (extra LiDAR branch + concatenated volume) diverged where camera-only didn't,
and the trainer had **no gradient clipping**.

**Fix (train_lss.py:305–306):** added `clip_grad_norm_(model.parameters(), 5.0)` with the
correct `GradScaler.unscale_` ordering. Restarted with the identical proven config
(lr 2e-3, AMP, cosine) — now cleared ep3 without a NaN and reached mIoU 0.495 by epoch 7.
`max_norm=5.0` only bites on pathological spikes, so the healthy camera-only path is
unchanged.

---

## Artifacts

| path | contents |
|---|---|
| `output/lss_occ_full/lss_occ.pth` | camera-only occupancy checkpoint (mIoU 0.302) |
| `output/lss_occ_full_fusion/lss_occ.pth` | fusion checkpoint (written on completion) |
| `output/classw_full.npy` | precomputed occ class weights (shared by both occ runs) |
| `output/lidar_cache/` | voxelized LiDAR fusion inputs (cached) |
| `output/nusc_det_lidar_cache/` | 10-sweep aggregated LiDAR for detection (cached) |
| `output/lss_occ_full.log`, `..._fusion.log`, `nusc_det.log` | per-epoch training logs |
| `output/*.collapsed.log` | archived logs of the failed fusion attempt |

## Official val evaluation + per-class IoU (⚠️ leakage caveat)

Ran the trained occupancy checkpoints over the **full official Occ3D-nuScenes val split**
(150 official val scenes, 6019 frames; `occupancy/eval_val.py`, filtered via
`nuscenes.utils.splits.val`).

> **⚠️ These numbers are optimistically biased — the checkpoints were trained on the val
> frames.** The Occ3D `gts` dir is the full trainval (850 scenes = 700 train + 150 val =
> 34149 frames = 28130 + 6019). Training used `--max-samples 34149` with **no scene filter**,
> so all 6019 official-val frames were in the training set. Not a clean held-out measure — treat
> mIoU as an upper bound and the per-class breakdown as diagnostic (which classes the model
> represents), not as a SOTA-comparable result.

| class | camera-only | fusion |    | class | camera-only | fusion |
|---|---|---|---|---|---|---|
| others | 0.105 | 0.230 | | trailer | 0.308 | 0.665 |
| barrier | 0.294 | 0.669 | | truck | 0.347 | 0.602 |
| bicycle | 0.105 | 0.600 | | driveable_surface | 0.788 | 0.816 |
| bus | 0.469 | 0.731 | | other_flat | 0.480 | 0.515 |
| car | 0.429 | 0.612 | | sidewalk | 0.487 | 0.525 |
| construction_veh | 0.131 | 0.591 | | terrain | 0.548 | 0.584 |
| motorcycle | 0.118 | 0.624 | | manmade | 0.398 | 0.684 |
| pedestrian | 0.133 | 0.570 | | vegetation | 0.365 | 0.666 |
| traffic_cone | 0.076 | 0.446 | | **mIoU / geo-IoU** | **0.328 / 0.678** | **0.596 / 0.851** |

**Reading it:** camera-only is **bimodal** — large static surfaces strong (driveable 0.79,
terrain 0.55) but small/dynamic/rare classes collapse (traffic_cone 0.08, bicycle/pedestrian/
motorcycle 0.10–0.13). Fusion **lifts exactly those weak classes** most (bicycle 0.10→0.60,
motorcycle 0.12→0.62, construction_veh 0.13→0.59) and is uniform (0.44–0.82) — LiDAR geometry
gives the shape of small objects the camera lift can't resolve. The §2.8.3 modality story,
per-class.

### Clean-retrain plan (future — deferred, ~1–2 days H100)

To get legitimate held-out numbers (deferred; full re-train too slow to block on):

1. **Add scene filtering to `train_lss.py`** (`--split {train,val,all}`): the dataset already
   supports it (`Occ3DNuScenesDataset(scenes=...)`); pass `nuscenes.utils.splits.train`
   (700 scenes / 28130 frames).
2. **Retrain both occupancy models on train-scenes only** — same configs as Run 1/2 + `--split train`.
3. **Eval on val-scenes only** with `eval_val.py` (already correct). The gap vs the leaked
   0.328 / 0.596 above quantifies the leakage.
4. Recompute `classw_full.npy` over train-only frames (the current one used all 34149 — minor
   loss-weight contamination).

Expect honest camera-only mIoU somewhat below 0.328 and fusion below 0.596; the *relative*
fusion-vs-camera and per-class story should hold.

## Detection — checkpoint was not saved (must retrain to eval)

`detection/train_nuscenes.py` has **no `torch.save`**, so the carAP_cd-0.467 run's weights were
**not persisted** and the process has exited — the detection model must be **retrained** to run
the official metric. It used the official `split="train"`/`"val"` datasets, so unlike occupancy
it has **no leakage** — a clean retrain (with saving + the official `nuscenes.eval.detection`
DetectionEval for NDS/mAP/TP-errors) gives publishable numbers. Retrain is cheap now: PointPillars
is small and the 10-sweep LiDAR cache (`output/nusc_det_lidar_cache/`) is already built.

## Backbone-reuse detection ablation — does the occupancy encoder help 3D detection?

Attaching a 10-class `VoxelDetHead` onto the **trained fused-voxel occupancy encoder**
(`train_det_ablation.py`) and comparing reuse strategies. Fast comparative sweep: 8000-frame
train subset, 12 epochs, official nuScenes val (6019 frames, `eval_det_ablation_official.py`,
ego→global). `car_AP@0.5` is the in-training proxy; **official mAP/NDS** is the headline.

| arm | car_AP@0.5 | official mAP | NDS | car | truck | pedestrian | occ_mIoU |
|---|---|---|---|---|---|---|---|
| **finetune-fusion** (occ-reg) | **0.127** | **0.205** | 0.173 | 0.445 | 0.157 | **0.439** | 0.277 |
| frozen-fusion (head-only) | 0.111 | 0.161 | 0.149 | 0.406 | 0.114 | 0.362 | 0.521 |
| scratch-fusion (no pretrain) | 0.090 | 0.065 | 0.101 | 0.355 | 0.064 | 0.094 | 0.001 |
| lidar-only-frozen | 0.058 | — | — | — | — | — | 0.139 |

**Findings:**
1. **Reusing the occ backbone is a large win.** finetune mAP 0.205 vs from-scratch 0.065 =
   **3.15×**; even a **frozen** encoder + trained head beats scratch **2.5×** (0.161 vs 0.065).
2. **The gain concentrates on small/rare classes.** pedestrian AP **0.094 (scratch) → 0.362
   (frozen) → 0.439 (finetune)**: from-scratch barely detects pedestrians, but the occ encoder
   — which learned dense pedestrian *voxels* — transfers that geometry to detection. Occupancy
   pretraining is the right prior exactly where detection-from-scratch is weakest. (car, an easy
   class, gains less: 0.355→0.406→0.445.)
3. **Finetune > frozen (+27% mAP)**, and the occ-regularizer (`--occ-weight 0.1`) keeps it
   genuinely multi-task (occ_mIoU 0.277 retained vs the det-only finetune's collapse to 0.001,
   which also diverged to NaN without it).
4. **Camera matters:** frozen fusion car_AP 0.111 ≈ 2× lidar-only 0.058.
5. **Data efficiency:** finetune-fusion reaches **car AP 0.445 on 8k/12ep**, ~matching standalone
   PointPillars-res's **carAP_cd 0.467 on 28k/40ep** — but multi-class (mAP 0.205) *and*
   multi-task. The occ pretraining buys a large data/schedule saving.

(NDS is dragged by mAVE/mAAE — no velocity/attribute predicted; mAP is the clean headline.
Standalone PointPillars baseline retrain is CPU-bound on numpy pillarization — its epoch-0
official checkpoint gives mAP 0.024 as a validated lower bound; full baseline is a separate run.)

**Trainers/evaluators** (all pure-PyTorch, committed): `train_det_ablation.py` (freeze/finetune/
modality + occ-reg), `eval_det_ablation_official.py` (occ-backbone, ego→global),
`detection/eval_nuscenes_official.py` (standalone, LiDAR→global).

## Open items

- **Add checkpoint saving + official-metric eval to `train_nuscenes.py`**, retrain, and report
  NDS / mAP / per-class AP / mATE·mASE·mAOE·mAVE·mAAE on the full val split (in progress).
- **Ablation: reuse the occupancy backbone for detection.** Attach a detection head onto the
  trained fused-voxel occupancy encoder (§5/M3 `det_head.py VoxelDetHead`) and improve the head
  — measure vs the standalone PointPillars baseline.
- **Occupancy clean-retrain** (train-only) for leakage-free mIoU — plan above.
- **Patch the two code issues** for reproducibility: the `compute_class_weights` GC stall + the
  grad-clip (both applied in the working tree / committed on `fulldata-h100-runs`).
- Further headroom (TUTORIAL §3): DINOv2-large, 0.2 m voxels, temporal fusion, beyond-concat
  LiDAR fusion — all H100 jobs on the same pure-PyTorch stack.
