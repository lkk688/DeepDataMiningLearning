"""
gaussian4d.train_student
========================
Train a **camera-only student** occupancy net against a cached label-free **teacher target**
(`build_teacher.py` output). The only thing that changes between Phase-1 arms is `--teacher-cache`
(voxel1 / voxel10 / gaussian / gaussian10 / render2d / occ3d-gt), so the comparison is clean.

Supervision (no human 3D labels): weighted CE to the teacher `semantics`, per-voxel-weighted by the
teacher `weight` (uncertainty — the Gaussian teacher's down-weighting has a real effect here) + a
small free-space term; LiDAR depth on the lift (free, part of the offline teacher). Camera-only
model (`lidar_fusion=False`) → inference uses cameras only.

    python -m DeepDataMiningLearning.ngperception.gaussian4d.train_student \
        --nusc <nuscenes> --gts <gts> --teacher-cache <teacher_cache/gaussian> \
        --epochs 24 --batch-size 4 --amp --out-dir output/student_gaussian
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..occupancy.models.lss_occ import LSSOccupancy
from ..occupancy.geom import CAMS
from ..occupancy.datasets_train import NuScenesOccTrainDataset
from ..occupancy.train_lss import lovasz_softmax_flat, depth_loss

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
FREE = 17


class StudentDataset(Dataset):
    """Camera images + calib + LiDAR depth (free lift supervision) + the cached teacher target."""

    def __init__(self, teacher_cache, nusc, gts, image_hw=(252, 700), downsample=14):
        from pyquaternion import Quaternion
        self.tc, self.nusc, self.Q = teacher_cache, nusc, Quaternion
        self.H, self.W = image_hw
        self.fH, self.fW = self.H // downsample, self.W // downsample
        self.tokens = [f[:-4] for f in sorted(os.listdir(teacher_cache)) if f.endswith(".npz")]
        self.soft = "soft_idx" in np.load(os.path.join(teacher_cache, self.tokens[0] + ".npz"))
        self._lh = NuScenesOccTrainDataset(gts, nusc, image_hw=image_hw, downsample=downsample,
                                           lidar_fusion=False, max_samples=1)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        from PIL import Image
        tok = self.tokens[i]
        sample = self.nusc.get("sample", tok)
        imgs, Ks, Rs, ts, deps = [], [], [], [], []
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
            deps.append(torch.from_numpy(self._lh._lidar_depth(tok, sd["token"], K, R, t, ow, oh, sx, sy)))
        d = np.load(os.path.join(self.tc, tok + ".npz"))
        out = {"imgs": torch.stack(imgs), "intrins": torch.stack(Ks),
               "rots": torch.stack(Rs), "trans": torch.stack(ts),
               "depth_gt": torch.stack(deps),
               "tsem": torch.from_numpy(d["semantics"].astype(np.int64)),      # (200,200,16)
               "tweight": torch.from_numpy(d["weight"].astype(np.float32))}
        if self.soft:                                                          # (200,200,16,K) top-K dist
            out["tsoft_idx"] = torch.from_numpy(d["soft_idx"].astype(np.int64))
            out["tsoft_prob"] = torch.from_numpy(d["soft_prob"].astype(np.float32))
        return out


def collate(b):
    return {k: torch.stack([x[k] for x in b]) for k in b[0]}


def teacher_loss(occ, tsem, tweight, lovasz_w=0.1):
    """Weighted CE to teacher semantics, per-voxel-weighted by the teacher `weight` directly:
    occupied = teacher confidence, ray-verified free = small constant (set in the teacher), unknown
    = 0 (NOT supervised — the ray-aware advantage). + Lovász on the occupied region. occ (B,18,X,Y,Z)."""
    C = occ.shape[1]
    ce = F.cross_entropy(occ, tsem, reduction="none")               # (B,X,Y,Z)
    l = (ce * tweight).sum() / tweight.sum().clamp_min(1.0)
    if lovasz_w > 0:
        probs = occ.softmax(1).permute(0, 2, 3, 4, 1).reshape(-1, C)
        l = l + lovasz_w * lovasz_softmax_flat(probs, tsem.reshape(-1), ignore=FREE)
    return l


def teacher_loss_soft(occ, tsem, tweight, tsoft_idx, tsoft_prob, lovasz_w=0.1, ent_floor=0.3):
    """SOFT distillation: on occupied voxels, soft cross-entropy to the teacher's top-K FM class
    DISTRIBUTION (not a single argmax) so car/truck & bicycle/motorcycle confusion is preserved;
    per-voxel weight is modulated by teacher confidence = (1 - normalised entropy of the top-K dist),
    clamped to [ent_floor,1] — ambiguous voxels supervise softly, not not-at-all. Free/unknown voxels
    keep the hard-CE path (toward FREE, weight = teacher free_w / 0). occ (B,18,X,Y,Z)."""
    C = occ.shape[1]; K = tsoft_idx.shape[-1]
    logp = F.log_softmax(occ, 1).permute(0, 2, 3, 4, 1)              # (B,X,Y,Z,C)
    occupied = tsem != FREE                                          # (B,X,Y,Z)
    idx = tsoft_idx.clamp(0, C - 1)                                  # free voxels -> idx 0, masked below
    lp_k = torch.gather(logp, -1, idx)                              # (B,X,Y,Z,K)
    soft_ce = -(tsoft_prob * lp_k).sum(-1)                          # (B,X,Y,Z) cross-entropy to dist
    ent = -(tsoft_prob.clamp_min(1e-6).log() * tsoft_prob).sum(-1)  # teacher entropy
    conf = (1.0 - ent / np.log(K)).clamp(ent_floor, 1.0)            # confidence modulation
    ce_free = F.cross_entropy(occ, tsem, reduction="none")          # hard CE (used off-occupied)
    per = torch.where(occupied, soft_ce, ce_free)
    w = torch.where(occupied, tweight * conf, tweight)
    l = (per * w).sum() / w.sum().clamp_min(1.0)
    if lovasz_w > 0:
        probs = occ.softmax(1).permute(0, 2, 3, 4, 1).reshape(-1, C)
        l = l + lovasz_w * lovasz_softmax_flat(probs, tsem.reshape(-1), ignore=FREE)
    return l


def teacher_loss_factorized(occ, tsem, tweight, tsoft_idx, tsoft_prob, sem_w=1.0, ent_floor=0.3):
    """FACTORIZED distillation (Gaussian-rescue): geometry and semantics are TRULY separated so
    semantic uncertainty can NOT leak into the occupancy signal.

        L = L_geom(1 - p_free, occ/free)              # binary, full 18-way, weight = teacher geom conf
          + sem_w * L_sem(p(c | occupied), q_c)       # 17-way conditional-on-occupied soft-CE,
                                                      #   modulated ONLY by teacher semantic confidence

    L_geom: binary occupied-vs-free via log p_free = log_softmax[FREE], log p_occ = logsumexp(non-free
      log-probs); target 1 on occupied (teacher class != free), 0 on ray-free; unknown (weight 0)
      ignored; weighted by the teacher's geometry `weight` — IDENTICAL mechanism for voxel & gaussian.
    L_sem: on occupied voxels only, renormalise the student to the 17 non-free classes and soft-CE to
      the teacher top-K distribution; per-voxel weight = confidence (1 - normalised entropy), clamped
      to [ent_floor,1]. Uncertainty touches semantics ONLY — the occupied mask is untouched.
    NO Lovász (it mixes geometry+semantics); both arms use this exact loss for a fair rescue."""
    lsm = F.log_softmax(occ, 1)                                     # (B,18,X,Y,Z)
    log_pfree = lsm[:, FREE]                                        # (B,X,Y,Z)
    log_pocc = torch.logsumexp(lsm[:, :FREE], dim=1)               # log P(not free)
    occupied = tsem != FREE                                         # geom occupied (teacher)
    geom_mask = tweight > 0                                         # occupied or ray-free (ignore unknown)
    occ_t = occupied.float()
    l_geom_vox = -(occ_t * log_pocc + (1 - occ_t) * log_pfree)     # BCE, numerically clean
    l_geom = (l_geom_vox * tweight * geom_mask).sum() / (tweight * geom_mask).sum().clamp_min(1.0)
    # --- semantics, conditioned on occupied, 17-way ---
    K = tsoft_idx.shape[-1]
    lp17 = F.log_softmax(occ[:, :FREE], 1).permute(0, 2, 3, 4, 1)   # (B,X,Y,Z,17) renorm to non-free
    idx = tsoft_idx.clamp(0, FREE - 1)
    lp_k = torch.gather(lp17, -1, idx)                             # (B,X,Y,Z,K)
    soft_ce = -(tsoft_prob * lp_k).sum(-1)
    ent = -(tsoft_prob.clamp_min(1e-6).log() * tsoft_prob).sum(-1)
    conf = (1.0 - ent / np.log(K)).clamp(ent_floor, 1.0)
    w_sem = conf * occupied.float()
    l_sem = (soft_ce * w_sem).sum() / w_sem.sum().clamp_min(1.0)
    return l_geom + sem_w * l_sem


def main():
    ap = argparse.ArgumentParser(description="Train camera-only student on a label-free teacher target.")
    ap.add_argument("--nusc", required=True); ap.add_argument("--gts", required=True)
    ap.add_argument("--teacher-cache", required=True)
    ap.add_argument("--backbone", default="dinov2_base")
    ap.add_argument("--decoder-layers", type=int, default=4); ap.add_argument("--decoder-hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=24); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--depth-weight", type=float, default=1.0); ap.add_argument("--lovasz", type=float, default=0.1)
    ap.add_argument("--amp", action="store_true"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--factorized", action="store_true",
                    help="Gaussian rescue: factorized geom+semantic loss (uncertainty on semantics only)")
    ap.add_argument("--sem-weight", type=float, default=1.0, help="lambda on the semantic term (factorized)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    dev = args.device
    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    model = LSSOccupancy(backbone=args.backbone, decoder_hidden=args.decoder_hidden,
                         decoder_layers=args.decoder_layers, refine_iters=1,
                         lidar_fusion=False).to(dev)                 # camera-only student
    ds = StudentDataset(args.teacher_cache, nusc, args.gts, image_hw=model.image_hw,
                        downsample=model.downsample)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate, drop_last=True)
    print(f"[student] {len(ds)} frames | camera-only | teacher={os.path.basename(args.teacher_cache.rstrip('/'))}"
          f" | {'FACTORIZED geom+sem (rescue)' if args.factorized else ('SOFT top-K distillation' if ds.soft else 'HARD argmax CE')}",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    for ep in range(args.epochs):
        model.train()
        for it, b in enumerate(ld):
            with torch.cuda.amp.autocast(enabled=args.amp):
                occ, depth, _ = model(b["imgs"].to(dev), b["rots"].to(dev), b["trans"].to(dev),
                                      b["intrins"].to(dev))
                if args.factorized:
                    assert ds.soft, "--factorized needs a soft teacher cache"
                    l_occ = teacher_loss_factorized(occ, b["tsem"].to(dev), b["tweight"].to(dev),
                                                    b["tsoft_idx"].to(dev), b["tsoft_prob"].to(dev),
                                                    sem_w=args.sem_weight)
                elif ds.soft:
                    l_occ = teacher_loss_soft(occ, b["tsem"].to(dev), b["tweight"].to(dev),
                                              b["tsoft_idx"].to(dev), b["tsoft_prob"].to(dev), lovasz_w=args.lovasz)
                else:
                    l_occ = teacher_loss(occ, b["tsem"].to(dev), b["tweight"].to(dev), lovasz_w=args.lovasz)
                l_dep = depth_loss(depth.float(), b["depth_gt"].to(dev))
                loss = l_occ + args.depth_weight * l_dep
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            if it % 50 == 0:
                print(f"  ep{ep} it{it}: loss={loss.item():.3f} (occ={l_occ.item():.3f} dep={l_dep.item():.3f})",
                      flush=True)
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.out_dir, "student.pth"))
        print(f"[student] epoch {ep} saved -> {args.out_dir}/student.pth", flush=True)


if __name__ == "__main__":
    main()
