"""
ngperception.detection.eval3d
=============================

A **self-contained, pure-PyTorch 3D detection AP** (no numba, no CUDA) — BEV average
precision with rotated-IoU matching, KITTI-style R40 interpolation. This is the locally
runnable evaluator for the M0 skeleton; the harvested `kitti_eval/` gives official-KITTI
parity but needs numba+CUDA (it segfaults under this WSL2 setup — use it on HPC).

Usage: `eval_map(preds, gts, iou_thresh=0.5)` where `preds`/`gts` are per-frame lists of
dicts (preds: {boxes(N,7), scores(N,), labels(N,)}; gts: {boxes(M,7), labels(M,)}).
"""
from __future__ import annotations
import numpy as np
import torch

from .box_utils import rotated_iou_bev


def average_precision_r40(rec, prec, n_points=40):
    """KITTI R40: mean of max-precision at 40 evenly spaced recall levels."""
    rec, prec = np.asarray(rec), np.asarray(prec)
    ap = 0.0
    for i in range(1, n_points + 1):
        r = i / n_points
        p = prec[rec >= r].max() if np.any(rec >= r) else 0.0
        ap += p
    return ap / n_points


def eval_class_bev_ap(preds, gts, cls, iou_thresh=0.5):
    """BEV AP for one class across frames. preds/gts: per-frame dicts. Returns (ap, precision-recall)."""
    entries = []                                            # (score, frame, pred_idx)
    n_gt = 0
    gt_boxes_per_frame, gt_used = [], []
    for f, (pd, gt) in enumerate(zip(preds, gts)):
        gm = gt["labels"] == cls
        gb = gt["boxes"][gm]
        gt_boxes_per_frame.append(gb)
        gt_used.append(np.zeros(len(gb), bool))
        n_gt += len(gb)
        pm = pd["labels"] == cls
        for i in np.where(pm.numpy() if torch.is_tensor(pm) else pm)[0]:
            entries.append((float(pd["scores"][i]), f, int(i)))
    entries.sort(key=lambda e: -e[0])
    tp, fp = np.zeros(len(entries)), np.zeros(len(entries))
    for k, (sc, f, i) in enumerate(entries):
        gb = gt_boxes_per_frame[f]
        if len(gb) == 0:
            fp[k] = 1; continue
        pbox = preds[f]["boxes"][i:i + 1]
        ious = rotated_iou_bev(pbox, gb)[0]                 # (M,)
        j = int(ious.argmax())
        if ious[j] >= iou_thresh and not gt_used[f][j]:
            tp[k] = 1; gt_used[f][j] = True
        else:
            fp[k] = 1
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rec = tp_c / max(n_gt, 1)
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return average_precision_r40(rec, prec), (prec, rec)


def eval_map(preds, gts, num_classes=1, iou_thresh=0.5):
    """Mean BEV AP over classes. Returns {'mAP':..., 'AP':[per-class]}."""
    aps = [eval_class_bev_ap(preds, gts, c, iou_thresh)[0] for c in range(num_classes)]
    return {"mAP": float(np.mean(aps)), "AP": [float(a) for a in aps]}


def center_distance_ap(preds, gts, cls, dist_thresh=2.0):
    """AP with nuScenes-style **center-distance** matching (BEV centre within `dist_thresh` m),
    not IoU. This is the actual nuScenes detection ruler — far more lenient than IoU@0.5."""
    entries, n_gt, gt_boxes_per_frame, gt_used = [], 0, [], []
    for f, (pd, gt) in enumerate(zip(preds, gts)):
        gm = (gt["labels"] == cls)
        gb = gt["boxes"][gm]
        gt_boxes_per_frame.append(gb); gt_used.append(np.zeros(len(gb), bool)); n_gt += len(gb)
        pm = (pd["labels"] == cls)
        for i in np.where(pm.numpy() if torch.is_tensor(pm) else pm)[0]:
            entries.append((float(pd["scores"][i]), f, int(i)))
    entries.sort(key=lambda e: -e[0])
    tp, fp = np.zeros(len(entries)), np.zeros(len(entries))
    for k, (sc, f, i) in enumerate(entries):
        gb = gt_boxes_per_frame[f]
        if len(gb) == 0:
            fp[k] = 1; continue
        d = torch.norm(gb[:, :2] - preds[f]["boxes"][i, :2], dim=1)     # BEV centre distance
        j = int(d.argmin())
        if float(d[j]) < dist_thresh and not gt_used[f][j]:
            tp[k] = 1; gt_used[f][j] = True
        else:
            fp[k] = 1
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rec = tp_c / max(n_gt, 1)
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return average_precision_r40(rec, prec)


def nusc_class_ap(preds, gts, cls, dists=(0.5, 1.0, 2.0, 4.0)):
    """nuScenes per-class AP = mean of center-distance AP over the 4 distance thresholds."""
    return float(np.mean([center_distance_ap(preds, gts, cls, d) for d in dists]))


# =========================================================================== #
# self-test:  python -m ...detection.eval3d
# =========================================================================== #
if __name__ == "__main__":
    torch.manual_seed(0)
    # frame 0: 2 GT cars; frame 1: 1 GT car
    gts = [{"boxes": torch.tensor([[10., 0., -1., 3.9, 1.6, 1.5, 0.1],
                                   [20., 5., -1., 3.9, 1.6, 1.5, 1.0]]), "labels": torch.tensor([0, 0])},
           {"boxes": torch.tensor([[30., -3., -1., 3.9, 1.6, 1.5, 0.0]]), "labels": torch.tensor([0])}]
    # perfect predictions (same boxes, high score) -> AP should be ~1.0
    perfect = [{"boxes": g["boxes"].clone(), "scores": torch.ones(len(g["boxes"])),
                "labels": g["labels"].clone()} for g in gts]
    print("perfect preds  -> mAP", round(eval_map(perfect, gts)["mAP"], 3), "(expect 1.0)")
    # add a false positive far away with a HIGHER score than the TPs -> precision drops -> AP drops
    noisy = [dict(p) for p in perfect]
    noisy[0] = {"boxes": torch.cat([perfect[0]["boxes"], torch.tensor([[50., 20., -1., 3.9, 1.6, 1.5, 0.]])]),
                "scores": torch.tensor([1.0, 1.0, 1.5]), "labels": torch.tensor([0, 0, 0])}
    print("+1 high-score FP-> mAP", round(eval_map(noisy, gts)["mAP"], 3), "(should be < 1.0)")
    # miss one GT (drop a pred) -> recall drops
    miss = [dict(p) for p in perfect]
    miss[0] = {"boxes": perfect[0]["boxes"][:1], "scores": torch.ones(1), "labels": torch.tensor([0])}
    print("miss one GT    -> mAP", round(eval_map(miss, gts)["mAP"], 3), "(should be < 1.0)")
    print("OK: eval3d")
