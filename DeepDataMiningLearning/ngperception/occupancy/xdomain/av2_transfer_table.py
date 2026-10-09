"""Cross-domain occupancy TRANSFER on AV2 surround renders — LSS + FlashOcc rows (both coexist in one
py310 process). Standardized on the SAME 6 AV2 cameras (av2_common.AV2_6) as OPUS/FusionOcc/GaussianOcc,
across all scene dirs. Writes per-model JSON (aggregate_table.py assembles the final table).
Headline metric = fg_recall (foreground-object-class recall on actor boxes); see av2_common.eval_pred.
Run in py310 with the bev_pool_v2 CUDA env (see av2_run_flashocc.py header)."""
import sys, os, glob, argparse
import numpy as np, torch
from types import SimpleNamespace
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import av2_common as C
import av2_run_flashocc as FO

IMAGENET_MEAN=np.array([0.485,0.456,0.406]); IMAGENET_STD=np.array([0.229,0.224,0.225])

def lss_inputs(npz, image_hw, frames_dir, dev):
    """our-LSS input (single current frame t0), AV2_6 cameras."""
    d=np.load(npz,allow_pickle=True); cams=list(d["cams"]); idx6=C.select6(cams); H,W=image_hw; base=os.path.basename(npz)[:-4]
    imgs=[]; Ks=[]; rots=[]; trans=[]
    for ci in idx6:
        im=Image.open(os.path.join(frames_dir,f"{base}_t0_{cams[ci]}.jpg")).convert("RGB"); W0,H0=im.size
        im=im.resize((W,H),Image.BILINEAR); arr=np.asarray(im).astype(np.float32)/255.
        arr=(arr-IMAGENET_MEAN)/IMAGENET_STD; imgs.append(arr.transpose(2,0,1))
        K=d["K"][0,ci].copy(); K[0]*=W/W0; K[1]*=H/H0; Ks.append(K)
        rots.append(d["s2e"][0,ci,:3,:3]); trans.append(d["s2e"][0,ci,:3,3])
    t=lambda a: torch.tensor(np.array(a),dtype=torch.float32,device=dev)[None]
    return (t(imgs),t(rots),t(trans),t(Ks))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--scenes",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--lss_ckpt",default="/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/ngperception/output/lss_occ_full/lss_occ.pth")
    a=ap.parse_args(); dev="cuda"; os.makedirs(a.out,exist_ok=True)
    from DeepDataMiningLearning.ngperception.flashocc.model_stereo import FlashOccBEVStereo4D, _CKPT_PATH
    fo=FlashOccBEVStereo4D(pretrained_img=False)
    sd=torch.load(_CKPT_PATH,map_location="cpu",weights_only=False); fo.load_state_dict(sd.get("state_dict",sd),strict=True); fo=fo.to(dev).eval()
    from DeepDataMiningLearning.ngperception.occupancy.visualize import build_model
    lss=build_model(a.lss_ckpt,SimpleNamespace(backbone="dinov2_base",decoder_hidden=96,decoder_layers=4,refine_iters=2),False,dev)

    scene_dirs=sorted(glob.glob(os.path.join(a.scenes,"*/")))
    agg={"FlashOcc":[], "LSS":[]}
    for sdir in scene_dirs:
        scene=os.path.basename(sdir.rstrip("/")); pf={"FlashOcc":[], "LSS":[]}
        for npz in C.scene_frames(sdir):
            dyn,allg=C.gt_masks(npz)
            inp,_=FO.build_inputs(npz,sdir,dev,num_frame=3)
            with torch.no_grad(): pfo=fo(inp).argmax(1)[0].cpu().numpy()
            pf["FlashOcc"].append(C.eval_pred(pfo,dyn,allg)); agg["FlashOcc"].append(pf["FlashOcc"][-1])
            imgs,rt,tr,intr=lss_inputs(npz,lss.image_hw,sdir,dev)
            with torch.no_grad(): pl=lss(imgs,rt,tr,intr)[0].argmax(1)[0].cpu().numpy()
            pf["LSS"].append(C.eval_pred(pl,dyn,allg)); agg["LSS"].append(pf["LSS"][-1])
        for k in pf:
            m=C.save_results(os.path.join(a.out,f"{'flashocc' if k=='FlashOcc' else 'lss'}_{scene}.json"),k,pf[k])
        print(f"{scene}: FlashOcc FG_REC {np.mean([r['fg_recall'] for r in pf['FlashOcc']]):.3f} | LSS FG_REC {np.mean([r['fg_recall'] for r in pf['LSS']]):.3f}")
    for k,tag in (("FlashOcc","flashocc"),("LSS","lss")):
        m=C.save_results(os.path.join(a.out,f"{tag}_ALL.json"),k,agg[k])
        print(f"{k} ALL ({len(agg[k])} frames): FG_REC {m['fg_recall']:.3f}  occ_frac {m['occ_frac']:.3f}  obj_IoU {m['obj_iou']:.3f}")

if __name__=="__main__": main()
