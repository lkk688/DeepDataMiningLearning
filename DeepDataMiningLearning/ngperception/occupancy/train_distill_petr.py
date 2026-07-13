"""
ngperception.occupancy.train_distill_petr
=========================================

**Path A** — distill a strong camera-only teacher (PETR, mmdet3d, camera mAP 0.383) into our
BEV model's **camera detection path**, as a single model. Architecture mismatch (PETR is
query-based, no BEV features) rules out feature KD, so this is **response/pseudo-label KD**:
PETR's predicted boxes are the "camera-achievable" target the camera student learns to reproduce
(softer + more camera-aligned than raw GT, which includes camera-invisible objects).

Recipe (fusion-anchored, like train_modality_robust): every batch trains **fusion on GT**
(anchor, keeps fusion+occ strong) + the **camera path on PETR pseudo-boxes** (the distillation).
Warm-started from the modality-robust checkpoint. Honest ceiling: capped by the BEV lift-splat
camera architecture (expect ~0.15–0.25 camera mAP, not PETR's 0.383).

    # 1) generate PETR preds on train frames (mmdet3d, py310):
    #    python tools/test.py <petr cfg> <ckpt> --cfg-options \
    #       test_dataloader.dataset.ann_file=nuscenes_infos_train.pkl \
    #       test_evaluator.jsonfile_prefix=work_dirs/petr_train/petr_preds test_evaluator.format_only=True
    # 2) distill:
    python -m DeepDataMiningLearning.ngperception.occupancy.train_distill_petr \
        --gts $GTS --nusc $NUSC --occ-ckpt $OUT/mod_robust/mod_robust.pth \
        --petr-preds .../petr_preds/results_nusc.json --out-dir $OUT/distill_petr
"""
from __future__ import annotations
import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from .models.lss_occ import LSSOccupancy
from .datasets_train import NuScenesOccTrainDataset
from .train_lss import occ_loss
from .train_multitask import collate_mt
from .train_modality_robust import eval_modality, MODALITIES
from ..detection.nuscenes_dataset import NUSC_10CLASS
from ..detection.train_nuscenes import NUSC_10_SIZES

NUSC_10_BOTTOMS_EGO = [-1.0] * 10
DET_NAMES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
             "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
NAME2IDX = {n: i for i, n in enumerate(DET_NAMES)}


def petr_boxes_ego(nusc, token, results, score_thresh):
    """PETR global-frame predictions for `token` -> (M,8) [x,y,z,l,w,h,yaw,cls] in the EGO grid frame."""
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    sd = nusc.get("sample_data", nusc.get("sample", token)["data"]["LIDAR_TOP"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    ego_q = Quaternion(pose["rotation"]); ego_t = np.array(pose["translation"])
    out = []
    for p in results.get(token, []):
        if p["detection_score"] < score_thresh:
            continue
        cls = NAME2IDX.get(p["detection_name"])
        if cls is None:
            continue
        box = Box(p["translation"], p["size"], Quaternion(p["rotation"]))     # global
        box.translate(-ego_t); box.rotate(ego_q.inverse)                      # global -> ego
        x, y, z = box.center; w, l, h = box.wlh
        R = box.orientation.rotation_matrix
        if not (-40 <= x < 40 and -40 <= y < 40):
            continue
        out.append([x, y, z, l, w, h, float(np.arctan2(R[1, 0], R[0, 0])), float(cls)])
    return torch.from_numpy(np.array(out, np.float32).reshape(-1, 8))


def main():
    ap = argparse.ArgumentParser(description="Distill PETR (camera teacher) into our camera path.")
    ap.add_argument("--gts", required=True); ap.add_argument("--nusc", required=True)
    ap.add_argument("--occ-ckpt", required=True, help="modality-robust / fusion warm-start")
    ap.add_argument("--petr-preds", required=True, help="PETR results_nusc.json (global-frame preds)")
    ap.add_argument("--petr-score", type=float, default=0.3, help="min PETR confidence to distill")
    ap.add_argument("--backbone", default="dinov2_base")
    ap.add_argument("--decoder-layers", type=int, default=4); ap.add_argument("--decoder-hidden", type=int, default=96)
    ap.add_argument("--refine-iters", type=int, default=2)
    ap.add_argument("--occ-weight", type=float, default=1.0); ap.add_argument("--det-weight", type=float, default=1.0)
    ap.add_argument("--max-samples", type=int, default=8000); ap.add_argument("--val-samples", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--cosine", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.device
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    from nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    petr = json.load(open(args.petr_preds))["results"]
    print(f"[distill] PETR preds for {len(petr)} tokens (score>={args.petr_score})", flush=True)

    model = LSSOccupancy(backbone=args.backbone, decoder_hidden=args.decoder_hidden,
                         decoder_layers=args.decoder_layers, refine_iters=args.refine_iters,
                         lidar_fusion=True, det_classes=10, det_anchor_sizes=NUSC_10_SIZES,
                         det_anchor_bottom=NUSC_10_BOTTOMS_EGO, det_head_type="center").to(dev)
    ck = torch.load(args.occ_ckpt, map_location=dev)
    model.load_state_dict(ck.get("state_dict", ck), strict=False)
    print(f"[distill] warm-started from {args.occ_ckpt}", flush=True)

    train_scenes = sorted(create_splits_scenes()["train"]); val_scenes = sorted(create_splits_scenes()["val"])
    ihw, dsf = model.image_hw, model.downsample
    dkw = dict(image_hw=ihw, downsample=dsf, lidar_fusion=True, det_boxes=True, det_class_map=NUSC_10CLASS)
    n = 2 if args.smoke else args.max_samples
    train_ds = NuScenesOccTrainDataset(args.gts, nusc, scenes=train_scenes, max_samples=n, **dkw)
    val_ds = NuScenesOccTrainDataset(args.gts, nusc, scenes=val_scenes, max_samples=args.val_samples, **dkw)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mt,
                          num_workers=args.num_workers, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=1, collate_fn=collate_mt, num_workers=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    epochs = 1 if args.smoke else args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if args.cosine else None
    print(f"[distill] {len(train_ds)} train / {len(val_ds)} val | {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = dict(backbone=args.backbone, decoder_layers=args.decoder_layers, decoder_hidden=args.decoder_hidden,
               refine_iters=args.refine_iters, lidar_fusion=True, lidar_only=False,
               det_anchor_sizes=NUSC_10_SIZES, det_anchor_bottom=NUSC_10_BOTTOMS_EGO, det_head_type="center")
    best = -1.0
    for ep in range(epochs):
        model.train()
        for it, b in enumerate(train_ld):
            imgs, rots = b["imgs"].to(dev), b["rots"].to(dev)
            trans, intr, lv = b["trans"].to(dev), b["intrins"].to(dev), b["lidar_vox"].to(dev)
            sem, mc = b["semantics"].to(dev), b["mask_camera"].to(dev)
            # PETR pseudo-boxes (ego) per sample, looked up by token
            petr_gt = [petr_boxes_ego(nusc, train_ds.occ.items[idx.item()][1], petr, args.petr_score)
                       for idx in b["sample_idx"]]

            occ_f, _, aux_f = model(imgs, rots, trans, intr, lidar_vox=lv)          # fusion anchor (GT)
            l_occ = occ_loss(occ_f, sem, mc)
            l_det_f, _ = model.det_head.get_loss(aux_f["det"], b["det_gt"])
            occ_c, _, aux_c = model(imgs, rots, trans, intr, lidar_vox=lv, drop_lidar=True)  # camera path
            l_occ_c = occ_loss(occ_c, sem, mc)
            l_det_c, _ = model.det_head.get_loss(aux_c["det"], petr_gt)             # <-- distill from PETR
            loss = args.occ_weight * (l_occ + l_occ_c) + args.det_weight * (l_det_f + l_det_c)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            if it % 50 == 0:
                npetr = sum(len(g) for g in petr_gt)
                print(f"  ep{ep} it{it}: det_f={l_det_f.item():.3f} det_cam(petr)={l_det_c.item():.3f} "
                      f"(petr boxes {npetr})", flush=True)
            if args.smoke and it >= 2:
                break
        if sched is not None:
            sched.step()
        res = {name: eval_modality(model, val_ld, dev, dc, dl) for name, (dc, dl) in MODALITIES.items()}
        line = "  ".join(f"{k}:occ={v[0]:.3f}/det={v[1]:.3f}" for k, v in res.items())
        cam_det = res["camera"][1]
        print(f"[distill] epoch {ep}: {line} | camera_det={cam_det:.3f}", flush=True)
        if cam_det > best:                                                        # maximize camera-only det
            best = cam_det
            torch.save({"state_dict": model.state_dict(), "cfg": cfg}, os.path.join(args.out_dir, "distill_petr.pth"))
            print(f"[distill]   saved best (camera_det={cam_det:.3f})", flush=True)
    print(f"[distill] done (best camera_det={best:.3f}) -> {args.out_dir}/distill_petr.pth", flush=True)


if __name__ == "__main__":
    main()
