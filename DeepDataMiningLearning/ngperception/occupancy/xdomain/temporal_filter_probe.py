"""Stage-0 temporal probe (go/no-go for recurrent/SSM occupancy): a HAND-CODED ego-warp + EMA filter over
consecutive LSS occupancy predictions. No training. For each dense scene clip, run the LSS per frame,
warp the running belief into the current ego frame by the known ego motion, blend (EMA in prob space),
and compare the accumulated occupancy to the single-frame prediction on:
  FG_REC (does occlusion-fill lift obstacle recall?), obj_IoU/occ_frac, and FLICKER (frame-to-frame
  foreground IoU after ego-warp — higher = more temporally stable).
This is the baseline any learned recurrent (Mamba/SSM) state must beat. Run in py310.
  python -m ...xdomain.temporal_filter_probe --scenes /tmp/shield/xdomain_eval_dense --ckpt <pth> --alpha 0.5"""
import sys, os, glob, argparse
import numpy as np, torch
import torch.nn.functional as F
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(__file__))
import av2_common as C
from train_lss_av2ft import lss_inputs
VOX=0.4; ORIG=np.array([-40.,-40.,-1.]); GS=200

def warp_prob(vol, R, t, dev):
    """vol (Cc,200,200,16) prob in ego_src -> ego_dst. Planar (xy) rigid warp via 2D grid_sample.
    src_xy = R2 @ dst_xy + t2 (R2,t2 = xy blocks of the ego_src<-ego_dst transform)."""
    Cc=vol.shape[0]
    ax=(np.arange(GS)+0.5)*VOX+ORIG[0]; ay=(np.arange(GS)+0.5)*VOX+ORIG[1]
    Xd,Yd=torch.meshgrid(torch.tensor(ax,device=dev),torch.tensor(ay,device=dev),indexing="ij")
    R2=torch.tensor(R[:2,:2],dtype=torch.float32,device=dev); t2=torch.tensor(t[:2],dtype=torch.float32,device=dev)
    Xs=R2[0,0]*Xd+R2[0,1]*Yd+t2[0]; Ys=R2[1,0]*Xd+R2[1,1]*Yd+t2[1]
    xn=2*(Xs-ORIG[0])/(GS*VOX)-1; yn=2*(Ys-ORIG[1])/(GS*VOX)-1
    grid=torch.stack([yn,xn],-1)[None].float()                    # (1,200,200,2): last dim (W=Y,H=X)
    inp=vol.reshape(Cc*16,GS,GS)[None]                            # (1,Cc*16,X,Y)
    out=F.grid_sample(inp,grid,mode="bilinear",padding_mode="zeros",align_corners=False)
    return out[0].reshape(Cc,GS,GS,16)

def fg_iou(a,b):
    u=(a|b).sum(); return float((a&b).sum()/u) if u else 1.0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--scenes",required=True); ap.add_argument("--ckpt",required=True)
    ap.add_argument("--alpha",type=float,default=0.5,help="EMA: acc=alpha*cur+(1-alpha)*warp(acc)")
    ap.add_argument("--warmup",type=int,default=6)
    ap.add_argument("--classaware",action="store_true",
                    help="ego-warp+EMA the STATIC classes only; take DYNAMIC (foreground) classes from the current frame")
    a=ap.parse_args(); dev="cuda"
    DYN=torch.tensor([2,3,4,5,6,7,9,10],device=dev)                 # foreground/dynamic object classes
    STATIC=torch.tensor([c for c in range(18) if c not in [2,3,4,5,6,7,9,10]],device=dev)
    from DeepDataMiningLearning.ngperception.occupancy.visualize import build_model
    m=build_model(a.ckpt,SimpleNamespace(backbone="dinov2_base",decoder_hidden=96,decoder_layers=4,refine_iters=2),False,dev); m.eval()
    S={"single":[], "accum":[]}; flick={"single":[], "accum":[]}
    for sdir in sorted(glob.glob(os.path.join(a.scenes,"*/"))):
        npzs=C.scene_frames(sdir)
        print(f"[probe] scene {os.path.basename(sdir.rstrip('/'))}: {len(npzs)} frames",flush=True)
        acc=None; prevsingle=None; prevacc=None; prev_e2g=None
        for fi,npz in enumerate(npzs):
            if fi%30==0: print(f"    frame {fi}",flush=True)
            d=np.load(npz,allow_pickle=True)
            (imgs,rots,trans,intr),_,_=lss_inputs(npz,m.image_hw,sdir,dev)
            with torch.no_grad(): prob=F.softmax(m(imgs,rots,trans,intr)[0],1)[0]   # (18,200,200,16)
            e2g=d["e2g"][0,0]                                         # ego->world this frame
            if acc is None:
                acc=prob.clone()
            else:
                Reg_s=prev_e2g[:3,:3]; teg_s=prev_e2g[:3,3]; Reg_d=e2g[:3,:3]; teg_d=e2g[:3,3]
                R=Reg_s.T@Reg_d; t=Reg_s.T@(teg_d-teg_s)            # ego_dst -> ego_src (to sample prev)
                acc_w=warp_prob(acc,R,t,dev)
                if a.classaware:
                    # static classes accumulate (rigid ego-warp is valid); dynamic come from current frame
                    acc=acc_w.clone()
                    acc[STATIC]=a.alpha*prob[STATIC]+(1-a.alpha)*acc_w[STATIC]
                    acc[DYN]=prob[DYN]
                else:
                    acc=a.alpha*prob + (1-a.alpha)*acc_w
            prev_e2g=e2g
            dyn,allg=C.gt_masks(npz)
            ps=prob.argmax(0).cpu().numpy(); pa=acc.argmax(0).cpu().numpy()
            if fi>=a.warmup:
                S["single"].append(C.eval_pred(ps,dyn,allg)); S["accum"].append(C.eval_pred(pa,dyn,allg))
                # flicker vs previous (warped) foreground argmax
                fgs=np.isin(ps,[2,3,4,5,6,7,9,10]); fga=np.isin(pa,[2,3,4,5,6,7,9,10])
                if prevsingle is not None:
                    flick["single"].append(fg_iou(fgs,prevsingle)); flick["accum"].append(fg_iou(fga,prevacc))
                prevsingle=fgs; prevacc=fga
    print(f"\n=== TEMPORAL-FILTER PROBE (ego-warp + EMA alpha={a.alpha}) vs single-frame ===")
    print(f"ckpt: {os.path.basename(a.ckpt)}   frames scored: {len(S['single'])}")
    for k in ["single","accum"]:
        agg={kk:np.mean([r[kk] for r in S[k]]) for kk in S[k][0]}
        fl=np.mean(flick[k]) if flick[k] else float('nan')
        print(f"  {k:7s}: FG_REC {agg['fg_recall']:.3f}  occ_frac {agg['occ_frac']:.3f}  obj_IoU {agg['obj_iou']:.3f}  fg_frac {agg['fg_frac']:.3f}  flicker(fg-IoU) {fl:.3f}")

if __name__=="__main__": main()
