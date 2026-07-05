"""
ngperception.lane.train_clrnet
=============================

Train / evaluate the pure-torch **CLRNet** (``clrnet.py``) on **CULane** or on the
**synthetic** sanity set. Same workflow as the detection module: de-risk locally
(overfit a few samples, watch loss fall and F1 rise), then run the full training on the
H100 with the same command + ``--dataset culane``.

Local de-risk (no download — proves assignment / LaneIoU / decode / F1 are correct):

    python -m DeepDataMiningLearning.ngperception.lane.train_clrnet \
        --dataset synthetic --overfit 8 --epochs 200 --lane-iou

Full CULane run (H100):

    python -m DeepDataMiningLearning.ngperception.lane.train_clrnet \
        --dataset culane --root /data/CULane \
        --train-list list/train_gt.txt --val-list list/test.txt \
        --backbone resnet34 --epochs 15 --bs 24 --lr 6e-4 --lane-iou
"""
from __future__ import annotations
import argparse
import os
import time
import torch
from torch.utils.data import DataLoader, Subset

from .clrnet import CLRNet
from .culane_dataset import CULaneDataset, SyntheticLanes, collate_lanes
from .culane_metric import CULaneF1


def build_data(args):
    if args.dataset == "synthetic":
        train = SyntheticLanes(args.max_train or 512, args.img_h, args.img_w, seed=0)
        val = SyntheticLanes(args.max_val or 64, args.img_h, args.img_w, seed=777)
    else:
        train = CULaneDataset(args.root, args.train_list, args.img_h, args.img_w,
                              args.max_train, augment=args.augment)
        val = CULaneDataset(args.root, args.val_list, args.img_h, args.img_w, args.max_val)
    if args.overfit:
        train = Subset(train, list(range(args.overfit)))
        val = train                                     # overfit: eval on the same few
    return train, val


@torch.no_grad()
def evaluate(model, loader, dev, score_thresh, nms_iou):
    model.eval()
    metric = CULaneF1(model.img_h, model.img_w)
    for imgs, lanes, _ in loader:
        out = model(imgs.to(dev))
        preds = model.decode(out, score_thresh=score_thresh, nms_iou=nms_iou)
        preds = [[p.cpu().numpy() for p in per] for per in preds]
        gts = [[l.numpy() for l in per] for per in lanes]
        metric.update(preds, gts)
    return metric.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "culane"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--train-list", default="list/train_gt.txt")
    ap.add_argument("--val-list", default="list/test.txt")
    ap.add_argument("--img-h", type=int, default=320)
    ap.add_argument("--img-w", type=int, default=800)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--augment", action="store_true", help="train-time flip+photometric aug (generalisation)")
    ap.add_argument("--lane-iou", action="store_true", help="LaneIoU (tilt) instead of LineIoU")
    ap.add_argument("--iou-r", type=float, default=7.5)
    ap.add_argument("--overfit", type=int, default=0, help="overfit first N samples (sanity)")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--score-thresh", type=float, default=0.4)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--resume", default=None, help="load a checkpoint before train/eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="DeepDataMiningLearning/ngperception/output/lane/clrnet.pt")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    train_ds, val_ds = build_data(args)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_lanes, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_lanes)

    model = CLRNet(args.img_w, args.img_h, backbone=args.backbone,
                   pretrained=not args.no_pretrained, iou_r=args.iou_r,
                   lane_iou_tilt=args.lane_iou).to(dev)
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[clrnet] resumed {args.resume}", flush=True)
    print(f"[clrnet] {args.backbone} | {model.P} priors | {model.N} rows "
          f"| {'LaneIoU' if args.lane_iou else 'LineIoU'} | dataset={args.dataset} "
          f"| train={len(train_ds)} val={len(val_ds)}", flush=True)

    if args.epochs == 0:                                 # eval-only
        m = evaluate(model, val_loader, dev, args.score_thresh, args.nms_iou)
        print(f"[clrnet] EVAL F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"tp/fp/fn={m['tp']}/{m['fp']}/{m['fn']}", flush=True)
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        agg = {"cls": 0, "iou": 0, "reg": 0, "n_pos": 0, "nb": 0}
        for imgs, lanes, _ in train_loader:
            out = model(imgs.to(dev))
            loss, parts = model.get_loss(out, lanes)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            opt.step()
            for k in ("cls", "iou", "reg", "n_pos"):
                agg[k] += parts[k]
            agg["nb"] += 1
        sched.step()
        nb = max(1, agg["nb"])
        msg = (f"[ep {ep:03d}] loss cls={agg['cls']/nb:.3f} iou={agg['iou']/nb:.3f} "
               f"reg={agg['reg']/nb:.3f} pos/b={agg['n_pos']/nb:.1f} "
               f"lr={sched.get_last_lr()[0]:.1e} ({time.time()-t0:.1f}s)")
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m = evaluate(model, val_loader, dev, args.score_thresh, args.nms_iou)
            msg += (f"  | F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
                    f"tp/fp/fn={m['tp']}/{m['fp']}/{m['fn']}")
        print(msg, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
    print(f"[clrnet] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
