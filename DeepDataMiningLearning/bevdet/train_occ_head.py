"""Train a dedicated BEV occupancy head on a FROZEN detector.

The original occupancy head was a low-weight auxiliary regulariser trained
for 3 epochs and never converged into a usable predictor (its output is
~0.5 everywhere with no height structure). Here we freeze the trained
BEVFusionCA detector, capture its fused BEV feature, and train a fresh
BEVOccHead to convergence on binary LiDAR pseudo-occupancy targets
(occupied = voxel contains a LiDAR return). This yields a real 3D
occupancy field for Tesla-style visualization.

  conda run -n py310 python -m DeepDataMiningLearning.bevdet.train_occ_head
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

MMDET3D = '/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d'
DDML = '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning'
sys.path.insert(0, MMDET3D); sys.path.insert(0, DDML)
os.chdir(MMDET3D)

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS
import projects.bevdet.cross_attn_lss2  # noqa
import projects.bevdet.bevfusion_ca     # noqa
from DeepDataMiningLearning.bevdet.train.occ_head import BEVOccHead, build_pseudo_occ_gt

CFG = f'{MMDET3D}/projects/bevdet/configs/ablation/B10c_flow_guided_warmstart_fixed.py'
DET_CKPT = f'{MMDET3D}/work_dirs/ablation_B10c/epoch_3.pth'
OUT_CKPT = f'{MMDET3D}/work_dirs/ablation_B10c/occ_head_trained.pth'
PC_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
BEV_HW, NUM_Z = 180, 16
DEVICE = 'cuda:0'
EPOCHS, MAX_ITERS = 4, 4000


def build_bev_capture(model):
    feat = {}
    model.fusion_layer.register_forward_hook(
        lambda m, i, o: feat.__setitem__('bev', o[0] if isinstance(o, (tuple, list)) else o))
    return feat


def main():
    model = init_model(CFG, DET_CKPT, device=DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    feat = build_bev_capture(model)

    # occupied class is rare -> weight it up; Lovasz handles the rest.
    cw = torch.tensor([0.5, 4.0], device=DEVICE)
    occ = BEVOccHead(in_channels=256, num_classes=2, num_z=NUM_Z,
                     z_range=[-5.0, 3.0], hidden_channels=128,
                     class_weights=cw).to(DEVICE).train()

    # train data: nuScenes 25% train, with the no-augmentation val pipeline
    cfg = Config.fromfile(CFG)
    val_dl = cfg.get('val_dataloader')
    train_ann = 'nuscenes_infos_train_25pct_mkf30.pkl'
    ds_cfg = dict(val_dl['dataset'])
    ds_cfg['ann_file'] = train_ann
    ds_cfg['test_mode'] = True
    ds = DATASETS.build(ds_cfg)
    n = len(ds)
    print(f'[occ-train] {n} frames; {EPOCHS} epochs, cap {MAX_ITERS} iters',
          flush=True)

    opt = torch.optim.AdamW(occ.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.RandomState(0)
    it = 0; t0 = time.time()
    for ep in range(EPOCHS):
        order = rng.permutation(n)
        for si in order:
            if it >= MAX_ITERS:
                break
            try:
                sample = ds[int(si)]
            except Exception:
                continue
            pts = sample['inputs'].get('points')
            if pts is None:
                continue
            data = pseudo_collate([sample])
            feat.clear()
            with torch.no_grad():
                model.test_step(data)
            bev = feat.get('bev')
            if bev is None:
                continue
            bev = bev.detach()
            gt = build_pseudo_occ_gt(pts.to(DEVICE), PC_RANGE,
                                     BEV_HW, BEV_HW, NUM_Z, DEVICE)
            gt = gt.unsqueeze(0)                     # [1,Z,H,W]
            logits = occ(bev)                         # [1,2,Z,H,W]
            losses = occ.loss(logits, gt)
            loss = sum(losses.values())
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 50 == 0:
                with torch.no_grad():
                    p1 = logits.softmax(1)[:, 1]
                    occ_frac = float((gt == 1).float().mean())
                    pred_frac = float((p1 > 0.5).float().mean())
                print(f'[occ-train] ep{ep} it{it} loss={loss.item():.3f} '
                      f'ce={losses["loss_occ_ce"].item():.3f} '
                      f'gt_occ={occ_frac:.3f} pred_occ={pred_frac:.3f} '
                      f'({(time.time()-t0)/max(it,1):.2f}s/it)', flush=True)
            if it > 0 and it % 1000 == 0:   # periodic save so progress persists
                torch.save({'state_dict': {f'occ_head.{k}': v
                            for k, v in occ.state_dict().items()}}, OUT_CKPT)
                print(f'[occ-train] checkpoint saved at it{it}', flush=True)
            it += 1
        if it >= MAX_ITERS:
            break

    torch.save({'state_dict': {f'occ_head.{k}': v
                               for k, v in occ.state_dict().items()}}, OUT_CKPT)
    print(f'[occ-train] saved {OUT_CKPT} after {it} iters')


if __name__ == '__main__':
    main()
