"""Run the fine-tuned LSS on a dense scene clip -> save per-frame occupancy (200,200,16) + the front
camera image, for the temporal occupancy video. Run in py310.
  python -m ...xdomain.predict_occ_sequence --scene /tmp/shield/xdomain_eval_dense/0d37aee4 --ckpt <pth> --out <dir>"""
import sys, os, glob, argparse, shutil
import numpy as np, torch
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(__file__))
import av2_common as C
from train_lss_av2ft import lss_inputs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--scene",required=True); ap.add_argument("--ckpt",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--stride",type=int,default=1); a=ap.parse_args(); dev="cuda"
    os.makedirs(a.out,exist_ok=True)
    from DeepDataMiningLearning.ngperception.occupancy.visualize import build_model
    m=build_model(a.ckpt,SimpleNamespace(backbone="dinov2_base",decoder_hidden=96,decoder_layers=4,refine_iters=2),False,dev); m.eval()
    npzs=C.scene_frames(a.scene)[::a.stride]
    for i,npz in enumerate(npzs):
        (imgs,rots,trans,intr),_,_=lss_inputs(npz,m.image_hw,a.scene,dev)
        with torch.no_grad(): pred=m(imgs,rots,trans,intr)[0].argmax(1)[0].cpu().numpy().astype(np.uint8)
        np.save(os.path.join(a.out,f"occ_{i:04d}.npy"),pred)
        base=os.path.basename(npz)[:-4]
        src=os.path.join(a.scene,f"{base}_t0_ring_front_center.jpg")
        if os.path.exists(src): shutil.copy(src,os.path.join(a.out,f"cam_{i:04d}.jpg"))
        # also save the GT boxes (dyn) for optional overlay
        d=np.load(npz,allow_pickle=True); np.save(os.path.join(a.out,f"boxes_{i:04d}.npy"),
                                                  np.concatenate([d["dyn"],d["sta"]]) if len(d["sta"]) else d["dyn"])
        if i%20==0: print(f"  {i}/{len(npzs)}",flush=True)
    print(f"saved {len(npzs)} frames -> {a.out}")

if __name__=="__main__": main()
