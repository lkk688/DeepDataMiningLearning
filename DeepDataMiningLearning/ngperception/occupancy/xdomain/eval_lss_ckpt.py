"""Evaluate one LSS checkpoint on the held-out AV2 eval scenes -> FG_REC/occ_frac/obj_IoU (av2_common
metrics), for the Phase-4 before/after comparison. Run in py310.
  python -m ...xdomain.eval_lss_ckpt --scenes /tmp/shield/xdomain_scenes --ckpt <pth> --tag av2ft"""
import sys, os, glob, argparse
import numpy as np, torch
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(__file__))
import av2_common as C
from train_lss_av2ft import lss_inputs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--scenes",required=True); ap.add_argument("--ckpt",required=True)
    ap.add_argument("--tag",default="lss"); ap.add_argument("--out",default="/tmp/shield/xdomain_results")
    a=ap.parse_args(); dev="cuda"; os.makedirs(a.out,exist_ok=True)
    from DeepDataMiningLearning.ngperception.occupancy.visualize import build_model
    m=build_model(a.ckpt,SimpleNamespace(backbone="dinov2_base",decoder_hidden=96,decoder_layers=4,refine_iters=2),False,dev)
    m.eval()
    allpf=[]
    for sdir in sorted(glob.glob(os.path.join(a.scenes,"*/"))):
        scene=os.path.basename(sdir.rstrip("/")); pf=[]
        for npz in C.scene_frames(sdir):
            (imgs,rots,trans,intr),_,_=lss_inputs(npz,m.image_hw,sdir,dev)
            with torch.no_grad(): pred=m(imgs,rots,trans,intr)[0].argmax(1)[0].cpu().numpy()
            dyn,allg=C.gt_masks(npz); pf.append(C.eval_pred(pred,dyn,allg)); allpf.append(pf[-1])
        mm={k:np.mean([r[k] for r in pf]) for k in pf[0]}
        print(f"  {scene}: FG_REC {mm['fg_recall']:.3f}  occ_frac {mm['occ_frac']:.3f}  obj_IoU {mm['obj_iou']:.3f}")
    m2=C.save_results(os.path.join(a.out,f"{a.tag}_ALL.json"),f"LSS {a.tag}",allpf)
    print(f"\n{a.tag} on held-out AV2 ({len(allpf)} frames): FG_REC {m2['fg_recall']:.3f}  occ_frac {m2['occ_frac']:.3f}  obj_IoU {m2['obj_iou']:.3f}")

if __name__=="__main__": main()
