# Related work (2025–2026) for label-free occupancy pseudo-labeling — findings & what to reuse

Eight recent papers (6 read 2026-07-21; **Sonata + LDCM added 2026-09-03**), mapped to our direction (label-free occ pseudo-labeling → detection
label-efficiency transfer). **The single most important one for us is TT-Occ** — read the positioning note.

## 1. TT-Occ — *Test-Time 3D Occupancy Prediction* (CVPR'26, arXiv 2503.08485, code: Xian-Bei/TT-Occ)
- **What:** label-free occ **without any training** — incrementally builds *time-aware 3D Gaussians*
  from raw LiDAR/camera streams, uses **vision foundation models** for open-vocab semantics, voxelizes
  at any resolution. Beats trained self-supervised occ on **Occ3D-nuScenes** + nuCraft.
- **Why it matters most:** this is essentially a *working label-free occ label generator* (VFM +
  temporal Gaussians, no 3D labels) — the same problem our `DynamicOccTeacher` tackles. Two takeaways:
  - **It validates the direction** (label-free occ from VFMs + temporal aggregation is now SOTA), and
  - **It sharpens our positioning:** label-free occ *prediction* is increasingly solved. **Our
    contribution must be the TRANSFER angle** — *does a label-free occ pretext confer detection
    label-efficiency?* — which none of these papers do. Our story is "label-free occ as a **pretext
    for label-efficient detection**," not "better label-free occ mIoU."
- **Reuse:** (a) treat TT-Occ as a **strong baseline / alternative pseudo-label generator** to compare
  our DynamicOcc against; (b) its **temporal-Gaussian aggregation + VFM open-vocab** is a recipe for
  our background/stuff semantics (our current FM projection is single-view/noisy). TT-Occ is expensive
  *per-scene test-time optimization*; **our offline-cache-then-pretrain framing is the practical
  differentiator** (generate once, distill into a fast camera student, transfer).

## 2. OnlinePG — *Online Open-Vocab Panoptic Mapping w/ 3DGS* (CVPR'26, arXiv 2603.18510)
- **What:** lifts **noisy 2D VLM semantics** (LSeg/EntitySeg) into a consistent 3D panoptic map via a
  **local→global sliding window** + a **3D segment-clustering graph** (geometric overlap + semantic
  similarity + multi-view consensus) + confidence-weighted per-voxel fusion. **Cannot handle dynamic
  objects** (needs depth+pose).
- **Reuse (directly fixes our known gap):** our FM background semantics are single-view and noisy —
  OnlinePG's **multi-view/temporal consensus clustering + per-voxel confidence weighting** is exactly
  how to denoise them into cleaner static-background occ labels. Adopt the confidence-weighted voxel
  fusion for our `assign_semantics` step. (Its dynamic-object blindness is *why* we separate dynamics
  via boxes/tracks — complementary.)

## 3. PanDA — *UDA for Multimodal 3D Panoptic Segmentation* (CVPR'26, arXiv 2604.19379)
- **What:** first unsupervised domain adaptation for multimodal 3D panoptic seg. **Asymmetric
  multimodal drop** (simulate modality degradation → domain-invariant features) + **DualRefine**
  (pseudo-label refinement fusing complementary **2D visual + 3D geometric** priors) for reliable
  target-domain supervision.
- **Reuse (directly relevant to cross-dataset):** our cross-dataset pretraining IS a domain-shift
  problem. **DualRefine's 2D+3D cross-modal pseudo-label refinement** is a recipe to clean our
  cross-dataset occ pseudo-labels (Waymo/AV2/PhysicalAI); **modality-drop** connects to our
  modality-robust backbone idea and improves robustness across sensor rigs.

## 4. ExtrinSplat — *Decoupling Geometry & Semantics for Open-Vocab 3DGS* (arXiv 2509.22225)
- **What:** open-vocab 3D understanding by **clustering Gaussians into object groups → VLM text
  descriptions** (object-level, not per-point embeddings); huge storage/time savings.
- **Reuse:** confirms **object-level semantics beat per-point** — our DynamicOcc already labels
  foreground by *object box class* (not per-point FM), which is the right call. The
  geometry/semantics **decoupling** also mirrors our factorized geom-vs-semantic loss.

## 5. SPAN — *Spatial-Projection Alignment for Monocular 3D Detection* (CVPR'26, arXiv 2511.06702)
- **What:** camera 3D det consistency via **Spatial Point Alignment** (global 3D-box spatial
  constraint) + **3D–2D Projection Alignment** (projected 3D box must sit inside the 2D box) +
  Hierarchical Task Learning (curriculum).
- **Reuse:** the **3D→2D projection-consistency loss** is a cheap **auxiliary self-supervision** for
  our camera occ→det student (project predicted 3D occupancy/boxes into image, enforce agreement with
  2D FM masks) — a label-free geometric signal. Curriculum (HTL) is a stability trick for our multi-task heads.

## 6. IDESplat — *Iterative Depth Probability for Generalizable 3DGS* (CVPR'26, arXiv 2601.03824)
- **What:** feed-forward 3DGS where depth (→ Gaussian centers) is refined by **iterative cascading
  warps + epipolar attention** (Depth Probability Boosting Unit); SOTA novel-view synthesis, strong
  cross-dataset generalization, tiny params.
- **Reuse (indirect):** depth is the crux of the camera lift. **Iterative multi-view depth refinement**
  could sharpen our *camera student's* geometry (we currently lean on LiDAR depth for the teacher);
  useful if we push a camera-only pseudo-label branch. Lowest priority for the current plan.

## 7. Sonata — *Self-Supervised Learning of Reliable Point Representations* (CVPR'25, arXiv 2503.16429, Pointcept × Meta)
- **What:** a **point-cloud foundation model**. PTv3 encoder-only (108M) **self-distilled over 140k point
  clouds** — no labels, **no teacher model**. Names the **"geometric shortcut"**: in 3D, representations
  collapse onto low-level spatial cues (surface normal, height) because point *position* is handed
  straight to the operators; fixed by **obscuring spatial information** (coarser scales, masked jitter,
  progressive mask schedule) and forcing reliance on input features. Outdoor: **nuScenes 81.7 mIoU**
  finetuned (80.4 scratch) / **66.1 linear-probe**, Waymo 72.9, SemanticKITTI 72.6. Data-efficiency:
  1% of ScanNet 25.8 → **45.3**; 20 labelled points/scene 60.1 → **70.5**.
- **Why it matters (this is the important one for the label-free arc):** our whole label-free story is
  *"mine occ pseudo-labels from un-annotated LiDAR, then pretrain"* — and our result is that such
  pretexts are **teacher-quality-bounded** (voxel-soft null; DynamicOcc *negative* despite +60% better
  foreground agreement). Sonata answers the **same question a different way**: skip labels entirely and
  **self-distill the point encoder**. Because it has **no teacher at all**, it is the clean instrument
  to decide whether our ceiling is a **teacher-quality bound** or a **pretext bound**. That is a
  decisive experiment we could not run before.
- **Second use — it names our VGGT negative.** The geometric shortcut is an independently-derived
  mechanism for our Finding A: VGGT's geometry-specialised tokens are the *wrong bias* for a **semantic**
  occupancy head (−33% vs DINOv2). Different modality, same failure mode: geometry capacity is not the
  binding constraint; semantic capacity is.
- **The gap it leaves us (our differentiator):** Sonata reports **only semantic segmentation** outdoors —
  **no 3D detection**. So its strong nuScenes number does **not** establish detection transfer, which is
  exactly what our §4.2 decoupling says must not be assumed. Our harness measures precisely that.
- **Reuse:** (a) frozen PTv3 as the **LiDAR branch** of the fusion column — the point-side counterpart
  our camera-only FM benchmark lacks; (b) its data-efficiency protocol (1% / 20-points-per-scene) is a
  ready template for our label-efficiency curves; (c) "obscure spatial info" is a design rule for any
  geometry-heavy pretext of ours — don't let the pretext be solvable from position alone.

## 8. LDCM — *Large Depth Completion Model from Sparse Observations* (ICLR'26, arXiv 2605.30115, code `aigc3d/LDCM`)
- **What:** a **depth foundation model for camera + sparse LiDAR**. A frozen monocular depth FM
  (DepthAnythingV2-S) gives relative depth; a **Poisson gradient-field alignment** fuses it with the
  sparse depth (minimise gradient-field difference subject to the observed sparse values, after a global
  affine scale/shift) into a coherent **metric** coarse depth; a DINOv2 ViT-B **dual encoder** (RGB +
  coarse depth, prompt-fused) then regresses an **intrinsic-free point map** (per-pixel 3D coords in
  camera space) rather than a depth map. ~2.7M samples / 11 datasets. **Zero-shot across 1–10% random,
  keypoint, and 64/32/16/8-beam LiDAR** sparsity; KITTI rel. **0.026** vs 0.042 prior best.
- **Why it matters:** our lift is **depth-supervised** and depth is its bottleneck — yet we measured a
  geometry FM (VGGT) as a **wash** as a depth prior (ablations #2/#3). The *mechanism* we gave was
  redundancy: a learned depth head **plus LiDAR depth supervision** already carried that signal. LDCM is
  different **in kind** — **metric** and **conditioned on the sparse observations** — so the regimes
  where it should pay are exactly the ones our VGGT null does **not** cover: **sparse / low-beam** LiDAR
  and **cross-rig** transfer. This makes "does a depth FM help the lift?" a *live* question again, with
  a stated, falsifiable condition.
- **Reuse:** (a) **drop-in depth prior** in our existing `vggt_depth` slot (`lss_occ.py`) via the
  `cache_*_depth.py` pattern — a same-slot A/B against VGGT, swept over beam count; (b) the
  **intrinsic-free point map** is a clean answer to the per-rig **intrinsics coupling** that complicates
  the cross-dataset occ pool (Waymo/AV2/PhysicalAI) in `PLAN_CROSSDATASET_OCC_PRETRAIN.md`; (c) the
  **Poisson gradient-field alignment alone** is a cheap standalone trick — it is how to fuse our noisy FM
  depth with sparse LiDAR into a better *teacher* depth, independent of LDCM's weights.

---

## Cross-cutting takeaways → concrete actions

1. **Position against TT-Occ.** Label-free occ *prediction* is now strong (TT-Occ). Our novelty is the
   **transfer**: label-free occ pretext → **detection label-efficiency** (the +35% we're chasing).
   Add TT-Occ as a baseline pseudo-label generator in the Step-0 ablation if time allows.
2. **Denoise FM semantics (OnlinePG + PanDA).** Replace single-view FM projection with **multi-view
   consensus + per-voxel confidence** (OnlinePG) and **2D+3D cross-modal refinement** (PanDA). This
   directly targets the background-semantics gap we flagged, and cross-dataset domain shift.
3. **Keep object-level foreground (ExtrinSplat) + dynamic/static separation (ours).** Our DynamicOcc
   already does this; the literature confirms it's the right structure.
4. **Cheap label-free geometric auxiliary (SPAN).** 3D→2D projection consistency as an extra
   self-supervised loss for the camera student — no labels needed.
5. **Modality-drop robustness (PanDA)** for the cross-dataset / multi-sensor pretraining pool.
6. **Test the teacher-vs-pretext bound (Sonata).** A frozen, teacher-free point FM as the LiDAR branch
   is the decisive arm: if it lifts low-label detection where our pseudo-label pretexts did not, the
   ceiling was **teacher quality**; if it also flattens, the ceiling is the **pretext/transfer** itself.
7. **Re-open the depth-prior question under the right conditions (LDCM).** Same slot as the VGGT
   ablation, but swept over **LiDAR beam count** — our VGGT null was measured only in the dense-sweep,
   depth-supervised regime where the prior is redundant by construction.

**Net:** none of these papers do label-free-occ → **detection-transfer** — *including Sonata, which
reports segmentation only* — so our thesis stays differentiated. They hand us concrete upgrades for the
two weak spots (noisy FM semantics; cross-dataset robustness), a strong baseline (TT-Occ) to benchmark
our pseudo-labels against, and — new with Sonata/LDCM — the **two missing modality axes**: a teacher-free
**point FM** that turns our "label-free is teacher-bounded" claim into a decidable experiment, and a
**metric, sparse-conditioned depth FM** that re-opens the depth-prior question in the low-beam/cross-rig
regime our VGGT null never tested.
