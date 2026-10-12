"""Cross-domain eval: run OPUSv2-L (nuScenes-trained, sparse-query occ, 8 key frames, camera-only) on
AV2 surround renders. Builds OPUS's exact input: 6 cams (AV2_6) x 8 frames, BGR uint8 256x704,
frame-major current-first, geometry as ego2img = ida @ K_pad @ ego2cam (ego-motion absorbed into the
projection, per compose_ego2img). Writes per-scene JSON results. Run in py310 (mmdet3d scope).

  PYTHONPATH=<repo>:<mmdet3d> python -m DeepDataMiningLearning.ngperception.occupancy.xdomain.av2_run_opus \
      --scenes /tmp/shield/xdomain_scenes --out /tmp/shield/xdomain_results"""
import sys, os, glob, argparse
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import av2_common as C

INPUT_HW=(256,704)  # (fH,fW)

def ida_matrix(sx, sy):
    """4x4 encoding the anisotropic full-resize (no crop). ego2img is left-multiplied by this."""
    m=np.eye(4,dtype=np.float32); m[0,0]=sx; m[1,1]=sy; return m

def load_bgr(path):
    """Anisotropic full-resize to 256x704 (no crop; matches FlashOcc/LSS policy); BGR uint8 CHW + (sx,sy)."""
    im=Image.open(path).convert("RGB"); W,H=im.size; fH,fW=INPUT_HW
    sx=fW/W; sy=fH/H
    im=im.resize((fW,fH))
    bgr=np.asarray(im,np.uint8)[:,:,::-1]                       # RGB->BGR (model flips to RGB)
    return np.ascontiguousarray(bgr.transpose(2,0,1)), (sx,sy)

def ego2cam4(Rsg,tsg,Reg_cur,teg_cur):
    """4x4 mapping CURRENT-ego points -> camera-f frame (column-vector convention)."""
    Rec=Rsg.T@Reg_cur; tec=Rsg.T@(teg_cur-tsg)
    M=np.eye(4,dtype=np.float32); M[:3,:3]=Rec; M[:3,3]=tec; return M

def build_opus_input(npz, frames_dir, dev):
    d=np.load(npz,allow_pickle=True); base=os.path.basename(npz)[:-4]
    idx6=C.select6(d["cams"]); F=d["s2e"].shape[0]
    Reg_cur=d["e2g"][0,idx6[0],:3,:3]; teg_cur=d["e2g"][0,idx6[0],:3,3]     # current ego->world
    imgs=[]; ego2img=[]; fnames=[]
    for f in range(F):
        for ci in idx6:
            c=list(d["cams"])[ci]
            chw,(sx,sy)=load_bgr(os.path.join(frames_dir,f"{base}_t{f}_{c}.jpg"))
            imgs.append(chw)
            Reg=d["e2g"][f,ci,:3,:3]; teg=d["e2g"][f,ci,:3,3]
            Rce=d["s2e"][f,ci,:3,:3]; tce=d["s2e"][f,ci,:3,3]
            Rsg=Reg@Rce; tsg=Reg@tce+teg                                   # cam->world
            Kpad=np.eye(4,dtype=np.float32); Kpad[:3,:3]=d["K"][f,ci]
            e2i=ida_matrix(sx,sy)@Kpad@ego2cam4(Rsg,tsg,Reg_cur,teg_cur)
            ego2img.append(e2i.astype(np.float32)); fnames.append(f"{base}_t{f}_{c}")
    imgs=torch.from_numpy(np.stack(imgs))[None].to(dev)                     # (1,6*F,3,256,704) uint8
    meta=dict(filename=fnames, ego2img=ego2img, ego2occ=np.eye(4,dtype=np.float32))
    return imgs, meta

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--scenes",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); dev="cuda"; os.makedirs(a.out,exist_ok=True)
    from mmengine.registry import init_default_scope; init_default_scope("mmdet3d")
    from projects.OPUS.opus import OPUSInference, sparse2dense
    CKPT="/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/opusv2-l_r50_704x256_8f_nusc-occ3d_100e.pth"
    model=OPUSInference(variant="l"); model.load_checkpoint(CKPT); model=model.to(dev).eval()
    scene_dirs=sorted(glob.glob(os.path.join(a.scenes,"*/")))
    all_pf=[]
    for sd in scene_dirs:
        scene=os.path.basename(sd.rstrip("/")); pf=[]
        for npz in C.scene_frames(sd):
            model.reset()
            imgs,meta=build_opus_input(npz,sd,dev)
            out=model.infer(imgs,[dict(meta)],image_loader=None)[0]
            pred,_=sparse2dense(out["occ_loc"],out["sem_pred"],dense_shape=(200,200,16),empty_value=17)
            pred=np.asarray(pred)
            dyn,allg=C.gt_masks(npz); pf.append(C.eval_pred(pred,dyn,allg)); all_pf.append(pf[-1])
        m=C.save_results(os.path.join(a.out,f"opus_{scene}.json"),"OPUSv2-L (cam,8f)",pf)
        print(f"  {scene}: FG_REC {m['fg_recall']:.3f}  occ_frac {m['occ_frac']:.3f}  obj_IoU {m['obj_iou']:.3f}  (n={len(pf)})")
    mm=C.save_results(os.path.join(a.out,"opus_ALL.json"),"OPUSv2-L (cam,8f)",all_pf)
    print(f"\nOPUSv2-L on AV2 (ALL {len(all_pf)} frames): FG_REC {mm['fg_recall']:.3f}  occ_frac {mm['occ_frac']:.3f}  obj_IoU {mm['obj_iou']:.3f}")

if __name__=="__main__": main()
