"""
ngperception.occupancy.train_gsocc_det
======================================
Train a detection head on the **frozen GaussianOcc label-free backbone** (external-validation data
point for the cheap-label claim). The GaussianOcc voxel features are precomputed to BEV
(`cache_gsocc_bev.py` -> (64,200,200) per token, ego frame X-fwd/Y-left [-40,40]); here we attach
our `VoxelDetHead` on that BEV and train ONLY the head (backbone frozen = features cached), then
evaluate with the official nuScenes DetectionEval. Answers: does a *published* self-supervised
occupancy backbone transfer to detection?

    # train:
    python -m DeepDataMiningLearning.ngperception.occupancy.train_gsocc_det \
        --nusc <nuscenes> --bev-cache <gsocc_bev_cache> --train-tokens /tmp/gsocc_tokens/train4k.txt \
        --epochs 24 --out-dir output/gsocc_det
    # eval (official mAP):
    python -m ...train_gsocc_det --eval --nusc <nuscenes> --bev-cache <cache> \
        --val-tokens /tmp/gsocc_tokens/val2k.txt --ckpt output/gsocc_det/gsocc_det.pth
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .det_head import VoxelDetHead
from .datasets_train import NuScenesOccTrainDataset
from ..detection.nuscenes_dataset import NUSC_10CLASS
from ..detection.train_nuscenes import NUSC_10_SIZES

PCR = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
BOTTOMS = [-1.0] * 10


class BevDetDataset(Dataset):
    """Cached GaussianOcc BEV (64,200,200) + our ego-frame det GT, keyed by token."""

    def __init__(self, bev_cache, token_file, nusc, gts):
        self.cache = bev_cache
        toks = [t.strip() for t in open(token_file) if t.strip()]
        self.tokens = [t for t in toks if os.path.isfile(os.path.join(bev_cache, t + ".npy"))]
        self._gt = NuScenesOccTrainDataset(gts, nusc, lidar_fusion=False, det_boxes=True,
                                           det_class_map=NUSC_10CLASS, max_samples=1)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        tok = self.tokens[i]
        bev = np.load(os.path.join(self.cache, tok + ".npy")).astype(np.float32)   # (64,200,200)=(C,X,Y)
        return {"bev": torch.from_numpy(bev), "det_gt": self._gt._det_boxes(tok), "token": tok}


def collate(b):
    return {"bev": torch.stack([x["bev"] for x in b]),
            "det_gt": [x["det_gt"] for x in b], "token": [x["token"] for x in b]}


def build_head(dev):
    return VoxelDetHead(in_channels=64, nz=1, pc_range=PCR, num_classes=10,
                        anchor_sizes=NUSC_10_SIZES, anchor_bottom=BOTTOMS).to(dev)


def run_eval(args, dev):
    import json
    from nuscenes import NuScenes
    from .eval_det_ablation_official import ego_box_to_global
    from ..detection.nuscenes_dataset import NUSC_10CLASS as _CM
    names = list(_CM.keys())
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    head = build_head(dev)
    head.load_state_dict(torch.load(args.ckpt, map_location=dev)); head.eval()
    ds = BevDetDataset(args.bev_cache, args.val_tokens, nusc, args.gts)
    ld = DataLoader(ds, batch_size=4, num_workers=6, collate_fn=collate)
    results = {}
    with torch.no_grad():
        for bi, b in enumerate(ld):
            vox = b["bev"].to(dev).unsqueeze(-1)                     # (B,64,200,200,1)
            out = head.predict(head(vox), score_thresh=args.score_thresh, nms_thresh=0.2)
            for j, tok in enumerate(b["token"]):
                boxes, scores, labels = out[j]
                results[tok] = [ego_box_to_global(nusc, tok, boxes[k], scores[k], names[int(labels[k])])
                                for k in range(len(boxes))]
            if (bi + 1) % 50 == 0:
                print(f"  eval {bi+1}/{len(ld)}", flush=True)
    res_path = os.path.join(args.out_dir, "results_nusc.json")
    with open(res_path, "w") as f:
        json.dump({"meta": {"use_camera": True, "use_lidar": False, "use_radar": False,
                            "use_map": False, "use_external": False}, "results": results}, f)
    print(f"[gsocc-det] wrote {res_path} ({sum(len(v) for v in results.values())} boxes)", flush=True)
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.detection.config import config_factory
    de = DetectionEval(nusc, config=config_factory("detection_cvpr_2019"), result_path=res_path,
                       eval_set="val", output_dir=args.out_dir, verbose=True)
    s = de.main(render_curves=False)
    print(f"\n===== GaussianOcc-backbone detector: NDS = {s['nd_score']:.4f}  mAP = {s['mean_ap']:.4f} =====")
    for k, v in s["mean_dist_aps"].items():
        print(f"    {k:<22} {v:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusc", required=True); ap.add_argument("--gts",
                    default="/data/rnd-liu/Datasets/nuScenes/v1.0-trainval/gts")
    ap.add_argument("--bev-cache", required=True)
    ap.add_argument("--train-tokens", default=None); ap.add_argument("--val-tokens", default=None)
    ap.add_argument("--epochs", type=int, default=24); ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval", action="store_true"); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--score-thresh", type=float, default=0.05)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    if args.eval:
        run_eval(args, dev); return

    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    head = build_head(dev)
    ds = BevDetDataset(args.bev_cache, args.train_tokens, nusc, args.gts)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate, drop_last=True)
    print(f"[gsocc-det] {len(ds)} frames | head-only on frozen GaussianOcc BEV | {sum(p.numel() for p in head.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)
    for ep in range(args.epochs):
        head.train()
        for it, b in enumerate(ld):
            vox = b["bev"].to(dev).unsqueeze(-1)                     # (B,64,200,200,1)
            pred = head(vox)
            loss, _ = head.get_loss(pred, [g.to(dev) for g in b["det_gt"]])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0); opt.step()
            if it % 50 == 0:
                print(f"  ep{ep} it{it}: det_loss={loss.item():.3f}", flush=True)
        torch.save(head.state_dict(), os.path.join(args.out_dir, "gsocc_det.pth"))
        print(f"[gsocc-det] epoch {ep} saved", flush=True)


if __name__ == "__main__":
    main()
