# Q-Mix v2 (Dense-Q-Mix) + cross-dataset transfer orchestration

This folder holds the **orchestration code and recovery tooling** for the
Q-Mix experiments (nuScenes → Waymo cross-dataset 3D detection transfer).
It was created after an HPC session loss wiped `/tmp`, taking with it both
the running jobs *and* the pseudo-label (PL) frame data — which had been
extracted to scratch `/tmp` and only **symlinked** into the dataset tree.

> **Robustness rule (learned the hard way): never put data or canonical
> scripts in `/tmp`.** `/tmp` is node-local and ephemeral on this HPC; it is
> wiped when the GPU/session ends. Only `/fs/atipa/data/...` and
> `/data/rnd-liu/...` (the same robust NFS, `172.16.3.13:/zpool/raid001`)
> survive. Logs may live in `/tmp`; data and code must not.

## Robust layout (source of truth)

| Artifact | Canonical location |
|---|---|
| Sampler implementation | `mmdetection3d/projects/bevdet/qmix.py` (`QMixWeightedSampler`) |
| B32 config (Q-Mix v2) | `mmdetection3d/projects/bevdet/configs/finetune/B32_qmix_v2.py` |
| Pareto configs | `.../finetune/B30_mix_nus12.py`, `B31_mix_nus06.py` |
| Orchestration scripts | `this folder/scripts/` |
| Recovery state map | `this folder/state/seg2tar.json`, `pl_segments.txt` |
| **Extracted per-frame npz (GT+PL)** | `/data/rnd-liu/Datasets/waymo_v1_extracted/` |
| info pkls | `DeepDataMiningLearning/data/waymo_finetune/*.pkl` |
| Raw Waymo v1.4.3 training tars | `/fs/atipa/data/rnd-liu/Datasets/waymo143/training/training/` |

## Method (what Q-Mix v2 is)

Dense-learning insight (Feng & Liu, *Nat. Commun.* 2026): overcome the
**Curse of Rarity** by **resampling** training data proportional to its
informativeness (gradient contribution × exposure), **excluding**
non-informative samples, **without biasing** the gradient. Translated to
supervised cross-dataset 3D detection in `QMixWeightedSampler`:

* **Resample frames, never scale the loss.** (v0 scaled the per-instance
  loss by reliability → biased the rare-class gradient → failed.)
* Per-frame weight = `clip(1 + MAX_inst[ rarity(class) × sparsity(lidar_pts)
  × range ], 1, w_max)` — MAX (not sum) so dense scenes don't dominate.
* **Per-source normalization** to mean 1.0 → keeps the Waymo:nuScenes:PL
  mix ratio fixed (anti-seesaw; the B30/B31 Pareto sweep confirmed mix
  ratio drives the nuScenes-NDS vs Waymo-Macro tradeoff).
* PL **reliability is a hard gate** (drop noisy PLs), not a loss weight.
* `w_max` cap prevents the v1b ×5 overshoot that collapsed Pedestrian.

## Scripts

* `scripts/reextract_pl2.sh` — **PL data recovery.** Re-extracts the 37 PL
  Waymo segments from the 3 training tars they live in (per-segment `tar`
  extraction → `extract_waymo_v1.py` → per-frame npz). Use this if the PL
  npz are ever lost again. (`reextract_pl.sh` is the earlier buggy version,
  kept for reference — it used multi-pattern `tar` which exits non-zero on
  these non-standard archives.)
* `scripts/persist_pl_data.sh` — rsyncs the extracted npz to the canonical
  `/data/rnd-liu/Datasets/waymo_v1_extracted/` store and verifies all
  GT+PL frames resolve.
* `scripts/b32_launch.sh` — trains B32 (Q-Mix v2) + 300-frame Waymo eval
  once PL data is verified and the GPU is free.
* `scripts/recovery_chain.sh` — the post-session-loss recovery chain
  (B32 → B31 nuScenes back-eval → occ viz).
* `state/seg2tar.json` — map of each PL segment → (training tar, member),
  built by scanning tar indices. Lets re-extraction pull only the ~35 GB of
  needed tfrecords instead of untarring all 720 GB.

## Recover PL data from scratch (if lost again)

```bash
# 1) (if seg2tar.json is missing) rebuild the segment→tar map by scanning
#    tar indices of /fs/atipa/data/rnd-liu/Datasets/waymo143/training/training
# 2) re-extract the 37 PL segments into the extracted root:
bash scripts/reextract_pl2.sh          # ~1 h: pull tfrecords + npz extract
# 3) persist to the canonical Datasets store:
bash scripts/persist_pl_data.sh
```

## Results context (300-frame Waymo eval, AP@2m)

| Variant | Macro | Cyclist | note |
|---|---|---|---|
| B20 baseline (uniform mix) | 0.183 | 0.136 | strong baseline |
| v0 (loss reweighting) | 0.158 | 0.061 | biased gradient — failed |
| v1/B26 (cyclist ×3 oversample) | 0.182 | 0.140 | ties baseline |
| **B32 v2 (dense sampler)** | *running* | *running* | unbiased, source-preserving |

Pareto (nuScenes mix ratio → Waymo Macro / nuScenes NDS): B14 25% =
0.268/0.4819; B30 12.5% = 0.230/0.4189; B31 6.25% = 0.224/(re-running).
