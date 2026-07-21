"""
flashocc.train_det_stereo
========================
Add a **detection head on top of the (supervised) 4D-stereo occupancy backbone** — the first probe of
the "strong occ backbone as a multi-task / world-model foundation" direction. The FlashOcc BEVStereo4D
backbone is FROZEN (its strong occ features are the shared representation); a CenterPoint head
(`VoxelCenterHead`) is trained on its BEV feature for nuScenes 10-class 3D detection. Answers: does a
strong occupancy representation carry detection?

Reuses: `data_stereo` (FlashOcc temporal-stereo inputs), `VoxelCenterHead` + nuScenes det GT from
`NuScenesOccTrainDataset(det_boxes=True)`. fp32 (amp corrupts BN running stats).

    export CUDA_HOME=/data/rnd-liu/cuda_home2 PATH=$CUDA_HOME/bin:$PATH \
           LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST=9.0
    python -m DeepDataMiningLearning.ngperception.flashocc.train_det_stereo \
        --nusc <nusc>/v1.0-trainval --gts <nusc>/v1.0-trainval/gts \
        --epochs 12 --batch-size 4 --out-dir output/det_on_stereo_frozen
"""
from __future__ import annotations
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..occupancy.geom import PC_RANGE, GRID_SIZE
from ..occupancy.det_head import VoxelCenterHead
from ..occupancy.datasets_train import NuScenesOccTrainDataset
from ..detection.nuscenes_dataset import NUSC_10CLASS
from .data_stereo import build_img_inputs


class StereoDetDataset(Dataset):
    """Per keyframe: FlashOcc temporal-stereo img_inputs + nuScenes 10-class ego-frame det GT."""
    def __init__(self, nusc, gts, scenes, max_samples=None):
        self.nusc = nusc
        self._base = NuScenesOccTrainDataset(gts, nusc, image_hw=(256, 704), downsample=16,
                                             scenes=scenes, det_boxes=True, det_class_map=NUSC_10CLASS,
                                             lidar_fusion=False, max_samples=max_samples, stride=1)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, i):
        tok = self._base.occ[i].sample_token
        inp = build_img_inputs(self.nusc, tok)                 # 7 tensors (unbatched)
        det_gt = self._base._det_boxes(tok)                    # (M,8) ego-frame boxes
        return {"img_inputs": inp, "det_gt": det_gt}


def collate(batch):
    n = len(batch[0]["img_inputs"])
    img_inputs = [torch.stack([b["img_inputs"][k] for b in batch]) for k in range(n)]
    return {"img_inputs": img_inputs, "det_gt": [b["det_gt"] for b in batch]}


class FlashOccDet(nn.Module):
    """Frozen BEVStereo4D backbone -> BEV feat (B,256,200,200) -> VoxelCenterHead (10-class).
    random_backbone=True keeps the backbone RANDOM-init (no occ pretraining) = the linear-probe
    control: how much of the detection signal is the OCC representation vs a random BEV projection."""
    def __init__(self, random_backbone=False):
        super().__init__()
        from .model_stereo import FlashOccBEVStereo4D
        if random_backbone:
            self.backbone = FlashOccBEVStereo4D(pretrained_img=False)   # random (control)
        else:
            self.backbone, miss, unexp = FlashOccBEVStereo4D.from_official_checkpoint()
            assert len(miss) == 0 and len(unexp) == 0, (len(miss), len(unexp))
        for p in self.backbone.parameters():
            p.requires_grad = False
        gx, gy, gz = [int(v) for v in GRID_SIZE]
        self.det_head = VoxelCenterHead(256, 1, list(PC_RANGE), num_classes=10,
                                        voxel_size=(0.4, 0.4), nx=gx, ny=gy)

    def forward(self, img_inputs):
        self.backbone.eval()
        with torch.no_grad():
            bev = self.backbone.bev_feature(img_inputs)        # (B,256,200,200)
        return self.det_head(bev.unsqueeze(-1))                # vox (B,256,200,200,1)


def main():
    ap = argparse.ArgumentParser(description="Detection head on the frozen supervised 4D-stereo backbone.")
    ap.add_argument("--nusc", required=True); ap.add_argument("--gts", required=True)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=None); ap.add_argument("--device", default="cuda")
    ap.add_argument("--random-backbone", action="store_true", help="control: frozen RANDOM backbone")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    dev = args.device
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    ds = StereoDetDataset(nusc, args.gts, scenes=sorted(splits.train), max_samples=args.max_samples)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate, drop_last=True)
    model = FlashOccDet(random_backbone=args.random_backbone).to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    bb = "RANDOM (control)" if args.random_backbone else "supervised 4D-stereo (occ-pretrained)"
    print(f"[det-on-stereo] {len(ds)} train frames | frozen {bb} backbone + CenterHead | "
          f"{sum(p.numel() for p in trainable)/1e6:.1f}M trainable", flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    os.makedirs(args.out_dir, exist_ok=True)
    for ep in range(args.epochs):
        model.det_head.train()
        for it, b in enumerate(ld):
            inp = [t.to(dev) for t in b["img_inputs"]]
            gt = [g.to(dev) for g in b["det_gt"]]
            pred = model(inp)
            l_det, _ = model.det_head.get_loss(pred, gt)
            opt.zero_grad(); l_det.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0); opt.step()
            if it % 50 == 0:
                print(f"  ep{ep} it{it}: det_loss={l_det.item():.3f}", flush=True)
        torch.save(model.det_head.state_dict(), os.path.join(args.out_dir, "det_head.pth"))
        print(f"[det-on-stereo] epoch {ep} saved -> {args.out_dir}/det_head.pth", flush=True)


if __name__ == "__main__":
    main()
