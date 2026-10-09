"""Phase 4: domain-adapt the DINOv2 LSS to AV2 renders WITHOUT dense semantic GT.

AV2 renders give class-agnostic object boxes (GT locations, no class) but no dense occ GT. Naive
box-only supervision has no free-space signal -> over-prediction (FlashOcc's failure mode). Instead:
  * FOREGROUND loss at object-box voxels: maximize P(foreground object class) = -log(sum softmax over
    OBJ_CLASSES). Adapts the encoder to render-domain object appearance; directly targets FG_REC.
  * FROZEN-TEACHER KD at all other voxels: KL(student || pretrained-teacher) preserves the pretrained
    road/free/background behavior (anti-forgetting) so the model does not drift into over-prediction.
Held-out eval (the 4 eval scenes) measures FG_REC before/after. Run in py310 (dinov2).

  python -m ...xdomain.train_lss_av2ft --train_scenes /tmp/shield/xdomain_train --epochs 4 \
      --out .../lss_occ_av2ft.pth"""
import sys, os, glob, argparse, math
import numpy as np, torch
import torch.nn.functional as F
from types import SimpleNamespace
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import av2_common as C, metrics as MET
IMAGENET_MEAN=np.array([0.485,0.456,0.406]); IMAGENET_STD=np.array([0.229,0.224,0.225])
OBJ=MET.OBJ_CLASSES
PRETRAINED="/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/ngperception/output/lss_occ_full/lss_occ.pth"

def lss_inputs(npz, image_hw, frames_dir, dev):
    d=np.load(npz,allow_pickle=True); cams=list(d["cams"]); idx6=C.select6(cams); H,W=image_hw; base=os.path.basename(npz)[:-4]
    imgs=[]; Ks=[]; rots=[]; trans=[]
    for ci in idx6:
        im=Image.open(os.path.join(frames_dir,f"{base}_t0_{cams[ci]}.jpg")).convert("RGB"); W0,H0=im.size
        im=im.resize((W,H),Image.BILINEAR); arr=np.asarray(im).astype(np.float32)/255.
        arr=(arr-IMAGENET_MEAN)/IMAGENET_STD; imgs.append(arr.transpose(2,0,1))
        K=d["K"][0,ci].copy(); K[0]*=W/W0; K[1]*=H/H0; Ks.append(K)
        rots.append(d["s2e"][0,ci,:3,:3]); trans.append(d["s2e"][0,ci,:3,3])
    t=lambda a: torch.tensor(np.array(a),dtype=torch.float32,device=dev)[None]
    fg=C.boxes_to_mask(np.concatenate([d["dyn"],d["sta"]]) if len(d["sta"]) else d["dyn"])
    free=None
    if "lidar_free" in d.files:                              # LiDAR ray-traversed free space (precision GT)
        free=np.unpackbits(d["lidar_free"])[:200*200*16].reshape(200,200,16).astype(bool)
        free=torch.tensor(free,device=dev)
    return (t(imgs),t(rots),t(trans),t(Ks)), torch.tensor(fg,device=dev), free

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_scenes",required=True); ap.add_argument("--pretrained",default=PRETRAINED)
    ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=4)
    ap.add_argument("--lr",type=float,default=1e-5); ap.add_argument("--kd_weight",type=float,default=1.0)
    ap.add_argument("--fg_weight",type=float,default=1.0); ap.add_argument("--free_weight",type=float,default=1.0)
    ap.add_argument("--freeze_backbone",action="store_true")
    ap.add_argument("--seed",type=int,default=0); a=ap.parse_args(); dev="cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    from DeepDataMiningLearning.ngperception.occupancy.visualize import build_model
    margs=SimpleNamespace(backbone="dinov2_base",decoder_hidden=96,decoder_layers=4,refine_iters=2)
    student=build_model(a.pretrained,margs,False,dev); student.train()
    teacher=build_model(a.pretrained,margs,False,dev); teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    if a.freeze_backbone:
        for n,p in student.named_parameters():
            if "backbone" in n: p.requires_grad_(False)
    opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    obj_idx=torch.tensor(OBJ,device=dev)

    frames=sorted(glob.glob(os.path.join(a.train_scenes,"*/f*.npz")))
    print(f"[av2ft] {len(frames)} train frames | epochs {a.epochs} | lr {a.lr} | kd {a.kd_weight} fg {a.fg_weight}",flush=True)
    for ep in range(a.epochs):
        order=np.random.permutation(len(frames)); tot=fgl=kdl=frl=0.0; nfg=nfree=0
        for it,fi in enumerate(order):
            npz=frames[fi]; sdir=os.path.dirname(npz)
            (imgs,rots,trans,intr),fg,free=lss_inputs(npz,student.image_hw,sdir,dev)
            slog=student(imgs,rots,trans,intr)[0]                 # (1,18,200,200,16)
            with torch.no_grad(): tlog=teacher(imgs,rots,trans,intr)[0]
            sp=F.softmax(slog,1)
            fgprob=sp.index_select(1,obj_idx).sum(1)[0]           # (200,200,16) P(foreground)
            fgloss=-(fgprob[fg].clamp_min(1e-6).log()).mean() if fg.any() else torch.zeros((),device=dev)
            # PRECISION loss: the sim's object GT is COMPLETE, so any foreground OUTSIDE a box is a true
            # false positive -> penalize P(foreground) at non-box voxels: -log(1-P_fg). Self-balancing
            # (near-zero where the model is already correct, large only on the false positives).
            nb=~fg
            freeloss=-((1.0-fgprob[nb]).clamp_min(1e-6).log()).mean(); nfree+=1
            # KD at non-box voxels: preserve the pretrained multi-class bg (road/free/building) distribution
            s_ls=F.log_softmax(slog,1)[0].permute(1,2,3,0)[nb]
            t_p =F.softmax(tlog,1)[0].permute(1,2,3,0)[nb]
            kd=F.kl_div(s_ls, t_p, reduction="batchmean")
            loss=a.fg_weight*fgloss + a.free_weight*freeloss + a.kd_weight*kd
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=float(loss); fgl+=float(fgloss); kdl+=float(kd); frl+=float(freeloss); nfg+=int(fg.any())
            if (it+1)%50==0: print(f"  ep{ep} {it+1}/{len(order)} loss {tot/(it+1):.4f} fg {fgl/(it+1):.4f} free {frl/(it+1):.4f} kd {kdl/(it+1):.4f}",flush=True)
        print(f"[ep{ep}] loss {tot/len(order):.4f} fg {fgl/len(order):.4f} free {frl/len(order):.4f} kd {kdl/len(order):.4f} (boxes {nfg}, lidar {nfree})",flush=True)
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    torch.save(student.state_dict(), a.out); print(f"[av2ft] saved -> {a.out}")

if __name__=="__main__": main()
