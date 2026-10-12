"""Fine-tune the occ DECODER to use a robust foundation-depth (DA-V2-Metric) lift instead of the
fragile learned depth head. Backbone + depth head frozen; only the 3D decoder is trained, so it learns
to map DA-V2-lifted voxels -> occupancy. Loss = av2ft_prec recipe (foreground box loss + free/precision).

Run (alpasim venv, from repo root with PYTHONPATH=repo):
  python -m DeepDataMiningLearning.ngperception.occupancy.xdomain.finetune_decoder_dav2 \
    --train_scenes /tmp/shield/xdomain_clean9 --dav2_dir /tmp/shield/dav2_train \
    --pretrained .../lss_occ_av2ft_prec.pth --out .../lss_dav2_decoder/lss_dav2.pth --epochs 4
"""
import sys, os, glob, argparse, numpy as np, torch, torch.nn.functional as F
from types import SimpleNamespace
from DeepDataMiningLearning.ngperception.occupancy.visualize import build_model
from DeepDataMiningLearning.ngperception.occupancy.xdomain.train_lss_av2ft import lss_inputs

AV2_6 = ["ring_front_center", "ring_front_left", "ring_front_right",
         "ring_side_left", "ring_side_right", "ring_rear_left"]
OBJ = [2, 3, 4, 5, 6, 7, 9, 10]
DEFAULT_PRE = "/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/ngperception/output/lss_occ_av2ft_prec/lss_occ_av2ft_prec.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_scenes", required=True)
    ap.add_argument("--dav2_dir", required=True)
    ap.add_argument("--pretrained", default=DEFAULT_PRE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--fg_weight", type=float, default=1.0)
    ap.add_argument("--free_weight", type=float, default=1.0)
    ap.add_argument("--use_lidar_free", action="store_true",
                    help="precision loss at ray-traced LiDAR-free voxels (artifact-aware) instead of box-complement")
    ap.add_argument("--unfreeze_encoder", action="store_true",
                    help="also train the DINOv2 encoder/ctx (semantic fg-bg discrimination = the precision lever)")
    ap.add_argument("--encoder_lr", type=float, default=1e-5)
    ap.add_argument("--sigma", type=float, default=1.5)
    a = ap.parse_args()
    dev = "cuda"

    m = build_model(a.pretrained, SimpleNamespace(backbone="dinov2_base", decoder_hidden=96,
                    decoder_layers=4, refine_iters=2), False, dev)
    if a.unfreeze_encoder:
        for p in m.encoder.parameters():
            p.requires_grad = True        # train the SEMANTIC ctx too (precision lever)
        m.encoder.train()
        opt = torch.optim.Adam([{"params": m.decoder.parameters(), "lr": a.lr},
                                {"params": m.encoder.parameters(), "lr": a.encoder_lr}])
        print(f"[dav2-ft] ENCODER UNFROZEN (encoder_lr={a.encoder_lr})", flush=True)
    else:
        for p in m.encoder.parameters():
            p.requires_grad = False       # freeze backbone + depth head (depth unused)
        m.encoder.eval()
        opt = torch.optim.Adam([p for p in m.decoder.parameters() if p.requires_grad], lr=a.lr)
    dlo, dhi, dstep, D = m.dlo, m.dhi, m.dstep, m.D
    centers = (torch.arange(D, device=dev) * dstep + dlo).view(D, 1, 1)
    obj_idx = torch.tensor(OBJ, device=dev)

    def dav2_dist(scene, base, h, w):
        dists = []
        for cam in AV2_6:
            p = f"{a.dav2_dir}/{scene}/{base}_{cam}.npy"
            if not os.path.exists(p):
                return None
            da = torch.tensor(np.load(p), device=dev)
            da = F.interpolate(da[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            da = da.clamp(dlo, dhi - 1e-3)
            pri = torch.exp(-((centers - da.view(1, h, w)) / a.sigma) ** 2)
            dists.append(pri / pri.sum(0, keepdim=True).clamp_min(1e-6))
        return torch.stack(dists, 0)       # (6,D,h,w)

    def forward_swap(imgs, rots, trans, Ks, scene, base):
        B, N = imgs.shape[:2]
        if a.unfreeze_encoder:
            ctx, _ = m.encoder(imgs.flatten(0, 1))          # (6,C,h,w) TRAINABLE
        else:
            with torch.no_grad():
                ctx, _ = m.encoder(imgs.flatten(0, 1))      # (6,C,h,w) frozen
        C, h, w = ctx.shape[1], ctx.shape[2], ctx.shape[3]
        depth = dav2_dist(scene, base, h, w)
        if depth is None:
            return None
        depth = depth.view(B, N, D, h, w)
        ctx = ctx.view(B, N, C, h, w)
        geom = m.get_geometry(rots, trans, Ks)
        lifted = depth.unsqueeze(2) * ctx.unsqueeze(3)      # (B,N,C,D,h,w)
        vox = m.voxel_pool(geom, lifted)
        return m.decoder(vox)                                # (B,ncls,nx,ny,nz) -- trains

    npzs = sorted(glob.glob(os.path.join(a.train_scenes, "*/f*.npz")))
    print(f"[dav2-ft] {len(npzs)} frames | epochs {a.epochs} | lr {a.lr} | fg {a.fg_weight} free {a.free_weight}", flush=True)
    for ep in range(a.epochs):
        order = np.random.permutation(len(npzs)); tot = fgl = frl = 0.0; nseen = 0
        for it, idx in enumerate(order):
            npz = npzs[idx]; scene = npz.split("/")[-2]; base = os.path.basename(npz)[:-4]
            sdir = os.path.dirname(os.path.realpath(npz))
            (imgs, rots, trans, Ks), fg, free = lss_inputs(npz, m.image_hw, sdir, dev)
            occ = forward_swap(imgs, rots, trans, Ks, scene, base)
            if occ is None:
                continue
            sp = occ.softmax(1)
            fgprob = sp.index_select(1, obj_idx).sum(1)[0]  # (nx,ny,nz) P(foreground)
            fgloss = -(fgprob[fg].clamp_min(1e-6).log()).mean() if fg.any() else torch.zeros((), device=dev)
            if a.use_lidar_free and free is not None:
                free_mask = free.bool() & (~fg)        # LiDAR-confirmed free (artifact-aware), excl. boxes
            else:
                free_mask = ~fg                         # box-complement fallback
            freeloss = (-((1.0 - fgprob[free_mask]).clamp_min(1e-6).log()).mean()
                        if free_mask.any() else torch.zeros((), device=dev))
            loss = a.fg_weight * fgloss + a.free_weight * freeloss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); fgl += float(fgloss); frl += float(freeloss); nseen += 1
            if (it + 1) % 50 == 0:
                print(f"  ep{ep} {it+1}/{len(order)} loss {tot/nseen:.4f} fg {fgl/nseen:.4f} free {frl/nseen:.4f}", flush=True)
        print(f"[ep{ep}] loss {tot/max(nseen,1):.4f} fg {fgl/max(nseen,1):.4f} free {frl/max(nseen,1):.4f} (n={nseen})", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(m.state_dict(), a.out)
    print(f"[dav2-ft] saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
