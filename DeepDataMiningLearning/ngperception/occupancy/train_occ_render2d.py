"""
ngperception.occupancy.train_occ_render2d
=========================================
**Label-FREE occupancy pretraining** via 2D pseudo-label rendering — the cheap-label pretext.

Instead of Occ3D 3D occupancy GT (expensive — built on 3D boxes + LiDAR seg), we supervise the
predicted voxel field only by *rendering it back to 2D* and matching cheap, frozen-FM pseudo-labels:
  - **semantic**  <- `ngdet/labelgen` (SegFormer stuff + Grounded-SAM things, in the Occ3D class space)
  - **depth**     <- LiDAR-projected metric depth (free sensor signal)
  - **geometry**  <- LiDAR-voxel occupancy (free)
No 3D occupancy labels, no detection labels, no human annotation. This is the GaussianOcc idea
(2D-rendered self-supervision) but with our high-quality pseudo-labels instead of a photometric loss.
The resulting backbone is then transferred to detection (`train_det_ablation.py --pretrained ...`) to
test whether the label-efficiency benefit survives a *label-free* pretext.

    python -m DeepDataMiningLearning.ngperception.occupancy.train_occ_render2d \
        --gts <gts> --nusc <nuscenes> --pseudo-cache <labelgen_cache> \
        --epochs 24 --batch-size 4 --amp --out-dir output/occ_render2d
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .models.lss_occ import LSSOccupancy
from .geom import CAMS
from .datasets_train import NuScenesOccTrainDataset

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
# labelgen NuScenesSource order -> our geom.CAMS order
LABELGEN_CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                 "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
REORDER = [LABELGEN_CAMS.index(c) for c in CAMS]    # -> [1,2,0,4,3,5]
FREE = 17


class RenderPretrainDataset(Dataset):
    """Frames that have a cached labelgen pseudo-label npz. Yields images + calib + LiDAR voxel/depth
    + the (camera-reordered, feature-res) pseudo semantic/depth targets. No Occ3D GT loaded."""

    def __init__(self, pseudo_cache, nusc, gts, image_hw=(252, 700), downsample=14):
        self.cache = pseudo_cache
        self.nusc = nusc
        self.H, self.W = image_hw
        self.fH, self.fW = self.H // downsample, self.W // downsample
        self.tokens = [f[:-4] for f in sorted(os.listdir(pseudo_cache)) if f.endswith(".npz")]
        # a helper dataset instance purely for its LiDAR voxel/depth methods (token-keyed)
        self._lh = NuScenesOccTrainDataset(gts, nusc, image_hw=image_hw, downsample=downsample,
                                           lidar_fusion=True, max_samples=1)
        from pyquaternion import Quaternion
        self.Q = Quaternion

    def __len__(self):
        return len(self.tokens)

    def _block(self, a, mode):
        """(H,W) -> (fH,fW). mode 'sem' = center subsample; 'depth' = min positive per block."""
        a = a.reshape(self.fH, self.H // self.fH, self.fW, self.W // self.fW)
        if mode == "sem":
            return a[:, a.shape[1] // 2, :, a.shape[3] // 2]
        d = a.transpose(0, 2, 1, 3).reshape(self.fH, self.fW, -1).astype(np.float32)
        d[d <= 0] = 1e9
        m = d.min(-1)
        m[m > 1e8] = 0.0
        return m

    def __getitem__(self, i):
        from PIL import Image
        tok = self.tokens[i]
        sample = self.nusc.get("sample", tok)
        imgs, Ks, Rs, ts, ldeps = [], [], [], [], []
        for cam in CAMS:
            sd = self.nusc.get("sample_data", sample["data"][cam])
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            img = Image.open(os.path.join(self.nusc.dataroot, sd["filename"])).convert("RGB")
            ow, oh = img.size
            sx, sy = self.W / ow, self.H / oh
            arr = (np.asarray(img.resize((self.W, self.H)), np.float32) / 255.0 - _MEAN) / _STD
            imgs.append(torch.from_numpy(arr.transpose(2, 0, 1)))
            K = np.array(cs["camera_intrinsic"], np.float32)
            Ks.append(torch.from_numpy(np.diag([sx, sy, 1]).astype(np.float32) @ K))
            R = self.Q(cs["rotation"]).rotation_matrix.astype(np.float32)
            t = np.array(cs["translation"], np.float32)
            Rs.append(torch.from_numpy(R)); ts.append(torch.from_numpy(t))
            ldeps.append(torch.from_numpy(self._lh._lidar_depth(tok, sd["token"], K, R, t, ow, oh, sx, sy)))
        pk = np.load(os.path.join(self.cache, tok + ".npz"))
        sem = pk["sem"][REORDER].astype(np.int64)                    # (6,H,W) occ order
        dep = pk["depth"][REORDER].astype(np.float32)
        sem_f = np.stack([self._block(sem[c], "sem") for c in range(6)])      # (6,fH,fW)
        dep_f = np.stack([self._block(dep[c], "depth") for c in range(6)])    # (6,fH,fW)
        return {
            "imgs": torch.stack(imgs), "intrins": torch.stack(Ks),
            "rots": torch.stack(Rs), "trans": torch.stack(ts),
            "lidar_vox": self._lh._lidar_voxel(tok),
            "lidar_depth": torch.stack(ldeps),                       # (6,fH,fW) bin idx (unused here)
            "pseudo_sem": torch.from_numpy(sem_f),                   # (6,fH,fW) 0..17
            "pseudo_depth": torch.from_numpy(dep_f),                 # (6,fH,fW) metric (0=invalid)
        }


def collate(b):
    return {k: torch.stack([x[k] for x in b]) for k in b[0]}


def main():
    ap = argparse.ArgumentParser(description="Label-free occ pretraining via 2D pseudo-label rendering.")
    ap.add_argument("--gts", required=True); ap.add_argument("--nusc", required=True)
    ap.add_argument("--pseudo-cache", required=True)
    ap.add_argument("--backbone", default="dinov2_base")
    ap.add_argument("--decoder-layers", type=int, default=4); ap.add_argument("--decoder-hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=24); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--sem-weight", type=float, default=1.0); ap.add_argument("--depth-weight", type=float, default=0.5)
    ap.add_argument("--occ-weight", type=float, default=0.5, help="LiDAR-voxel geometric occ BCE")
    ap.add_argument("--amp", action="store_true"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.device
    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)

    model = LSSOccupancy(backbone=args.backbone, decoder_hidden=args.decoder_hidden,
                         decoder_layers=args.decoder_layers, refine_iters=1,
                         lidar_fusion=True).to(dev)
    ds = RenderPretrainDataset(args.pseudo_cache, nusc, args.gts,
                               image_hw=model.image_hw, downsample=model.downsample)
    if args.smoke:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(2, len(ds)))))
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate, drop_last=True)
    print(f"[render2d] {len(ds)} frames | backbone={args.backbone} | pretext=sem+depth+lidar-occ (NO 3D GT)",
          flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    nz = model.nz

    for ep in range(1 if args.smoke else args.epochs):
        model.train()
        for it, b in enumerate(ld):
            with torch.cuda.amp.autocast(enabled=args.amp):
                occ, _, _ = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev),
                                  b["intrins"].to(dev), lidar_vox=b["lidar_vox"].to(dev))
                geom = model.get_geometry(b["rots"].to(dev), b["trans"].to(dev), b["intrins"].to(dev))
                rsem, rdep = model.render_2d(occ, geom)              # (B,N,C,h,w),(B,N,h,w)
                B, N, C, h, w = rsem.shape
                psem = b["pseudo_sem"].to(dev).view(B * N, h, w)
                pdep = b["pseudo_depth"].to(dev).view(B * N, h, w)
                # 2D semantic CE (ignore sky=17 mapped to free; valid where pseudo != sky)
                logs = torch.log(rsem.clamp_min(1e-6)).view(B * N, C, h, w)
                valid = (psem != FREE)
                l_sem = F.nll_loss(logs, psem.clamp(0, C - 1), reduction="none")
                l_sem = (l_sem * valid).sum() / valid.sum().clamp_min(1)
                # 2D depth L1 where pseudo depth valid
                rd = rdep.view(B * N, h, w)
                dv = pdep > 0.1
                l_dep = (F.l1_loss(rd, pdep, reduction="none") * dv).sum() / dv.sum().clamp_min(1)
                # LiDAR-voxel geometric occ BCE
                lv = b["lidar_vox"].to(dev)                          # (B,3,nx,ny,nz)
                locc = (lv.abs().sum(1) > 0).float()                 # any lidar point -> occupied
                pocc = 1.0 - occ.softmax(1)[:, FREE]                 # predicted occupied
                l_occ = F.binary_cross_entropy(pocc.clamp(1e-4, 1 - 1e-4), locc)
                loss = args.sem_weight * l_sem + args.depth_weight * l_dep + args.occ_weight * l_occ
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            if it % 50 == 0:
                print(f"  ep{ep} it{it}: loss={loss.item():.3f} (sem={l_sem.item():.3f} "
                      f"dep={l_dep.item():.3f} occ={l_occ.item():.3f})", flush=True)
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.out_dir, "occ_render2d.pth"))
        print(f"[render2d] epoch {ep} saved -> {args.out_dir}/occ_render2d.pth", flush=True)


if __name__ == "__main__":
    main()
