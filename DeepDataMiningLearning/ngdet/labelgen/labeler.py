"""Grounded 2-D label generator — open-vocab semantic segmentation + segment-level metric depth.

ngdet's detectors give BOXES; this module gives dense PER-PIXEL labels: a semantic map and a
metric depth map, produced entirely by FROZEN foundation models (no training, no 3-D labels).
It is the "label engine" behind auto-labeling for occupancy / BEV / depth tasks, and a teaching
example of composing open-vocab detection + promptable segmentation + monocular depth + a sparse
sensor.

Recipe (each piece plays to its strength):
  SEMANTIC = SegFormer stuff (road/sidewalk/veg/building/sky — amorphous regions a box can't frame)
             OVERLAID with GROUNDED-SAM things: Grounding-DINO open-vocab boxes -> SAM masks ->
             crisp, in-domain car/pedestrian/truck/… (the classes a Cityscapes segmenter mislabels
             out of domain). Each tool where it is strong: SegFormer for stuff, Grounded-SAM for things.
  DEPTH    = a SEGMENT-LEVEL METRIC prior (not sparse-point L1, which is noisy): DepthAnything gives
             dense relative shape; a GLOBAL affine to projected LiDAR sets metric scale; a PER-SAM-
             SEGMENT shift snaps each object to its own LiDAR median (1 DOF -> cannot sign-flip).
             Dense, object-consistent, metric. LiDAR is optional: without it, depth is relative.

All models load from HF `transformers` (no extra installs). Swap the DepthAnything checkpoint for
`…-Large-hf` when generating OFFLINE labels (best shape) or `…-Small-hf` for speed. See TUTORIAL §24.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# --- taxonomy -------------------------------------------------------------------------------------
# A class is a THING (localizable object -> Grounding-DINO+SAM) or STUFF (amorphous region ->
# SegFormer) or SKY (-> no occupancy). id is the output label; color is for visualization.
@dataclass
class Taxonomy:
    names: list                       # id -> name
    colors: np.ndarray                # [C,3] uint8
    dino_things: list                 # [(prompt_term, id)], matched by substring against DINO phrase
    cs_to_id: np.ndarray              # Cityscapes-19 trainId -> this taxonomy id (-1 = drop/sky)
    sky_id: int
    @property
    def dino_prompt(self):
        return " ".join(f"{t}." for t, _ in self.dino_things)


# nuScenes / Occ3D-style default (17 semantic + sky). Colors match the Occ3D palette.
_OCC = np.array([[0, 0, 0], [255, 120, 50], [255, 192, 203], [255, 255, 0], [0, 150, 245],
                 [0, 255, 255], [200, 180, 0], [255, 0, 0], [255, 240, 150], [135, 60, 0],
                 [160, 32, 240], [255, 0, 255], [175, 0, 75], [75, 0, 75], [150, 240, 80],
                 [230, 230, 250], [0, 175, 0], [255, 255, 255]], np.uint8)
NUSC_TAXONOMY = Taxonomy(
    names=['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
           'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat',
           'sidewalk', 'terrain', 'manmade', 'vegetation', 'free'],
    colors=_OCC,
    # dropped construction_vehicle/trailer from the PROMPT: rare + DINO grounds cars to them.
    dino_things=[("barrier", 1), ("traffic cone", 8), ("bus", 3), ("truck", 10), ("bicycle", 2),
                 ("motorcycle", 6), ("pedestrian", 7), ("person", 7), ("car", 4)],
    #            road drive|sidewalk|bldg/wall/pole/light/sign->manmade|fence->barrier|veg|terrain|
    #            SKY|person/rider->ped|car|truck|bus|train->others|motorcycle|bicycle
    cs_to_id=np.array([11, 13, 15, 15, 1, 15, 15, 15, 16, 14, -1, 7, 7, 4, 10, 3, 0, 6, 2], np.int64),
    sky_id=17,
)


class GroundedLabeler:
    """Frozen open-vocab semantic + segment-level metric-depth labeler. All models lazy-loaded."""

    def __init__(self, device="cuda", taxonomy: Taxonomy = NUSC_TAXONOMY,
                 seg_ckpt="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
                 dino_ckpt="IDEA-Research/grounding-dino-tiny",
                 sam_ckpt="facebook/sam-vit-base",
                 depth_ckpt="depth-anything/Depth-Anything-V2-Small-hf",
                 box_thresh=0.30, text_thresh=0.25, max_depth=60.0):
        import torch
        from transformers import (SegformerForSemanticSegmentation, AutoProcessor,
                                  AutoModelForZeroShotObjectDetection, SamModel, SamProcessor,
                                  AutoImageProcessor, AutoModelForDepthEstimation)
        self.torch, self.dev, self.tax = torch, device, taxonomy
        self.box_thresh, self.text_thresh, self.max_depth = box_thresh, text_thresh, max_depth
        self.seg = SegformerForSemanticSegmentation.from_pretrained(seg_ckpt).to(device).eval()
        self.dino_p = AutoProcessor.from_pretrained(dino_ckpt)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_ckpt).to(device).eval()
        self.sam_p = SamProcessor.from_pretrained(sam_ckpt)
        self.sam = SamModel.from_pretrained(sam_ckpt).to(device).eval()
        self.dep_p = AutoImageProcessor.from_pretrained(depth_ckpt)
        self.dep = AutoModelForDepthEstimation.from_pretrained(depth_ckpt).to(device).eval()

    # ----- semantic -----
    def _segformer_stuff(self, pil):
        import torch, torch.nn.functional as F
        W, H = pil.size
        with torch.no_grad():
            inp = self.seg_processor_norm(pil)
            logits = self.seg(pixel_values=inp.to(self.dev)).logits
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        cs = logits.argmax(1)[0].cpu().numpy()
        occ = self.tax.cs_to_id[cs]
        stuff = np.where(np.isin(cs, [0, 1, 8, 9, 2, 3, 5, 6, 7]), occ, -1)   # road/sw/veg/terrain/bldg…
        return stuff, (cs == 10)                                             # stuff ids, sky mask

    def seg_processor_norm(self, pil):
        """ImageNet-normalized tensor [1,3,H,W] (SegFormer/DepthAnything share this norm)."""
        import torch
        a = np.asarray(pil).astype(np.float32) / 255.0
        a = (a - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.from_numpy(a.transpose(2, 0, 1)[None].astype("float32"))

    def _grounded_sam(self, pil):
        import torch
        W, H = pil.size
        gi = self.dino_p(images=pil, text=self.tax.dino_prompt, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            go = self.dino(**gi)
        res = self.dino_p.post_process_grounded_object_detection(
            go, gi["input_ids"], threshold=self.box_thresh, text_threshold=self.text_thresh,
            target_sizes=[(H, W)])[0]
        boxes = res["boxes"]; phrases = res.get("text_labels", res["labels"]); scores = res["scores"]
        if len(boxes) == 0:
            return []
        si = self.sam_p(pil, input_boxes=[[b.tolist() for b in boxes]], return_tensors="pt").to(self.dev)
        with torch.no_grad():
            so = self.sam(**si)
        masks = self.sam_p.image_processor.post_process_masks(
            so.pred_masks.cpu(), si["original_sizes"].cpu(), si["reshaped_input_sizes"].cpu())[0]
        iou = so.iou_scores[0].cpu().numpy()
        out = []
        for i in range(masks.shape[0]):
            ph = phrases[i] if isinstance(phrases[i], str) else str(phrases[i])
            cls = next((c for term, c in self.tax.dino_things if term in ph.lower()), None)
            if cls is None:
                continue
            out.append((masks[i, int(iou[i].argmax())].numpy().astype(bool), cls, float(scores[i])))
        return out

    def semantic(self, pil):
        """PIL image -> (sem [H,W] int label map, seg_masks [(mask,cls)] object segments)."""
        stuff, sky = self._segformer_stuff(pil)
        sem = np.where(stuff >= 0, stuff, self.tax.sky_id).astype(np.int64)
        sem[sky] = self.tax.sky_id
        seg_masks = []
        for mask, cls, score in sorted(self._grounded_sam(pil), key=lambda t: t[2]):   # weak first
            sem[mask] = cls
            seg_masks.append((mask, cls))
        return sem, seg_masks

    # ----- soft semantic (per-pixel class DISTRIBUTION, for the Gaussian-teacher soft-semantics 2x2) -----
    def _occ_distribution(self, cs_prob):
        """Cityscapes softmax (19,H,W) -> Occ3D class distribution (18,H,W) via the taxonomy map
        (mass of every Cityscapes class is added to its Occ3D target; dropped/sky -> free). Preserves
        confusion structure (e.g. car/truck, bicycle/motorcycle) that a top-1 label throws away."""
        C, H, W = cs_prob.shape
        dist = np.zeros((len(self.tax.names), H, W), np.float32)
        for cs_id in range(C):
            oid = int(self.tax.cs_to_id[cs_id])
            dist[oid if oid >= 0 else self.tax.sky_id] += cs_prob[cs_id]
        return dist

    def semantic_soft(self, pil, topk=3):
        """PIL -> per-pixel top-k Occ3D class distribution: idx (H,W,K) uint8 + prob (H,W,K) f16.
        SegFormer gives the stuff distribution; each Grounded-SAM thing BLENDS its class in with the
        detection score as confidence (thing gets `score` mass, the rest keeps the FM distribution),
        so the soft label carries both detection confidence and the underlying confusion."""
        import torch, torch.nn.functional as F
        W, H = pil.size
        with torch.no_grad():
            logits = self.seg(pixel_values=self.seg_processor_norm(pil).to(self.dev)).logits
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            cs_prob = logits.softmax(1)[0].cpu().numpy()          # (19,H,W)
        dist = self._occ_distribution(cs_prob)                    # (18,H,W)
        for mask, cls, score in sorted(self._grounded_sam(pil), key=lambda t: t[2]):
            d = dist[:, mask] * (1.0 - score)
            d[cls] += score
            dist[:, mask] = d
        dist /= dist.sum(0, keepdims=True).clip(1e-6)
        idx = np.argsort(-dist, axis=0)[:topk]                    # (K,H,W)
        pk = np.take_along_axis(dist, idx, 0)
        return idx.transpose(1, 2, 0).astype(np.uint8), pk.transpose(1, 2, 0).astype(np.float16)

    # ----- depth -----
    def _depth_anything(self, pil):
        import torch, torch.nn.functional as F
        W, H = pil.size
        inp = self.dep_p(images=pil, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            d = self.dep(**inp).predicted_depth
        return F.interpolate(d[None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    def depth(self, pil, lidar_uv_z=None, seg_masks=None, drel=None):
        """DepthAnything relative shape -> metric via LiDAR. lidar_uv_z [P,3]=(u,v,z_metric) of LiDAR
        returns projected into THIS camera (or None -> relative depth returned as-is). seg_masks from
        semantic() -> per-object shift. `drel` lets you pass a precomputed relative-depth map (e.g.
        from a different DepthAnything variant, for a Small-vs-Large comparison). Returns [H,W] float32."""
        W, H = pil.size
        if drel is None:
            drel = self._depth_anything(pil).astype(np.float32)
        if lidar_uv_z is None or len(lidar_uv_z) < 20:
            return drel                                              # relative-only
        u = np.round(lidar_uv_z[:, 0]).astype(int); v = np.round(lidar_uv_z[:, 1]).astype(int)
        z = lidar_uv_z[:, 2].astype(np.float32)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, z = u[ok], v[ok], z[ok]
        A = np.stack([drel[v, u], np.ones_like(z)], 1)               # GLOBAL affine -> metric scale
        (a, b), *_ = np.linalg.lstsq(A, z, rcond=None)
        metric = a * drel + b
        for mask, cls in (seg_masks or []):                         # PER-SEGMENT shift to LiDAR median
            inside = mask[v, u]
            if inside.sum() >= 10:
                shift = np.median(z[inside]) - np.median(metric[mask])
                if abs(shift) < 40:
                    metric[mask] = metric[mask] + shift
        return np.clip(metric, 0.0, self.max_depth)

    def label(self, pil, lidar_uv_z=None):
        """Convenience: -> dict(sem [H,W], depth [H,W], masks [(mask,cls)])."""
        sem, seg_masks = self.semantic(pil)
        depth = self.depth(pil, lidar_uv_z, seg_masks)
        return {"sem": sem, "depth": depth, "masks": seg_masks}
