# Related work (2025–2026) for label-free occupancy pseudo-labeling — findings & what to reuse

Six recent papers, read 2026-07-21, mapped to our direction (label-free occ pseudo-labeling → detection
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

**Net:** none of these papers do label-free-occ → **detection-transfer**, so our thesis stays
differentiated. They hand us concrete upgrades for the two weak spots (noisy FM semantics; cross-dataset
robustness) and a strong baseline (TT-Occ) to benchmark our pseudo-labels against.
