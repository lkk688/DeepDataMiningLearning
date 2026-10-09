"""Cross-domain eval: run the SUPERVISED FlashOcc BEVStereo4DOCC teacher (nuScenes-trained, verified
0.381 mIoU in-domain) on extracted AV2 surround TEMPORAL windows -> object recall/precision/IoU +
occupied-fraction vs actor-box GT. Camera-only. Run in py310 with the bev_pool_v2 CUDA env:

  PYTHONPATH=<repo> CUDA_HOME=/data/rnd-liu/cuda_home2 \
  PATH=<py310bin>:/data/rnd-liu/cuda_home2/bin:$PATH \
  LD_LIBRARY_PATH=/data/rnd-liu/cuda_home2/lib64:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST=9.0 \
  python -m DeepDataMiningLearning.ngperception.occupancy.xdomain.av2_run_flashocc --frames <dir>

Feeds AV2 images/calib in FlashOcc's exact input layout (imgs CAM-major cam*F+f; calib FRAME-major
f*C+c; K unscaled, resize/crop encoded in post_rot/post_tran; RGB->BGR + ImageNet norm)."""
import sys, os, glob, math, argparse
import numpy as np, torch
from PIL import Image
sys.path.insert(0,os.path.dirname(__file__))
import metrics as MET
from DeepDataMiningLearning.ngperception.flashocc.model_stereo import FlashOccBEVStereo4D, _CKPT_PATH

VOX=0.4; ORIG=np.array([-40.,-40.,-1.]); GS=np.array([200,200,16])
INPUT_HW=(256,704); _MEAN=np.array([123.675,116.28,103.53],np.float32); _STD=np.array([58.395,57.12,57.375],np.float32)

def boxes_to_mask(boxes):
    m=np.zeros(tuple(GS),bool)
    xs=ORIG[0]+(np.arange(200)+.5)*VOX; ys=ORIG[1]+(np.arange(200)+.5)*VOX; zs=ORIG[2]+(np.arange(16)+.5)*VOX
    X,Y,Z=np.meshgrid(xs,ys,zs,indexing="ij")
    for cx,cy,cz,yaw,sx,sy,sz in boxes:
        c,s=math.cos(-yaw),math.sin(-yaw)
        lx=(X-cx)*c-(Y-cy)*s; ly=(X-cx)*s+(Y-cy)*c; lz=Z-cz
        m|=(np.abs(lx)<=sx/2)&(np.abs(ly)<=sy/2)&(np.abs(lz)<=sz/2)
    return m

def img_transform(pil):
    """Anisotropic full-resize to (fH,fW), NO crop — preserves all content across AV2's portrait
    front_center + landscape ring cams (nuScenes-style width-resize+bottom-crop would discard most of a
    portrait image). Geometry stays exact: post_rot=diag(sx,sy) encodes the per-axis scale, K unscaled.
    Same policy for every camera model in the table (LSS/FlashOcc/OPUS) -> airtight, lossless."""
    W,H=pil.size; fH,fW=INPUT_HW
    sx=float(fW)/float(W); sy=float(fH)/float(H)
    img=pil.resize((fW,fH))
    a=np.asarray(img,np.float32)[...,::-1]                  # RGB->BGR (matches to_rgb reversal)
    a=(a-_MEAN)/_STD
    chw=torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1)))
    R=torch.eye(3); T=torch.zeros(3)
    R[0,0]=sx; R[1,1]=sy
    return chw,R,T

def build_inputs(npz, frames_dir, dev, num_frame=3):
    """AV2_6 cameras, first `num_frame` temporal frames (FlashOcc uses 3)."""
    import av2_common as AC
    d=np.load(npz,allow_pickle=True); cams=list(d["cams"]); idx6=AC.select6(cams)
    F=min(num_frame,d["s2e"].shape[0]); base=os.path.basename(npz)[:-4]
    imgs_cm=[]                                              # CAM-major: c*F+f
    for ci in idx6:
        c=cams[ci]
        for f in range(F):
            p=os.path.join(frames_dir,f"{base}_t{f}_{c}.jpg")
            chw,_,_=img_transform(Image.open(p).convert("RGB")); imgs_cm.append(chw)
    s2e_fm=[]; e2g_fm=[]; K_fm=[]; pr_fm=[]; pt_fm=[]       # FRAME-major: f*6+c
    for f in range(F):
        for ci in idx6:
            c=cams[ci]
            p=os.path.join(frames_dir,f"{base}_t{f}_{c}.jpg")
            _,R,T=img_transform(Image.open(p).convert("RGB"))
            s2e_fm.append(torch.tensor(d["s2e"][f,ci],dtype=torch.float32))
            e2g_fm.append(torch.tensor(d["e2g"][f,ci],dtype=torch.float32))
            K_fm.append(torch.tensor(d["K"][f,ci],dtype=torch.float32))
            pr_fm.append(R); pt_fm.append(T)
    B=lambda L: torch.stack(L)[None].to(dev)
    inp=[B(imgs_cm),B(s2e_fm),B(e2g_fm),B(K_fm),B(pr_fm),B(pt_fm),torch.eye(3)[None].to(dev)]
    return inp,d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--frames",required=True); a=ap.parse_args()
    dev="cuda"
    model=FlashOccBEVStereo4D(pretrained_img=False)
    sd=torch.load(_CKPT_PATH,map_location="cpu",weights_only=False); sd=sd.get("state_dict",sd)
    model.load_state_dict(sd,strict=True); model=model.to(dev).eval()
    fs=sorted(glob.glob(os.path.join(a.frames,"f*.npz")))
    oi=[]; of=[]; gi=[]; rec=[]; prc=[]
    for f in fs:
        inp,d=build_inputs(f,a.frames,dev)
        with torch.no_grad(): out=model(inp)               # (1,18,200,200,16)
        pred=out.argmax(1)[0].cpu().numpy()                # (200,200,16), 17=free
        dyn_gt=boxes_to_mask(d["dyn"])
        all_gt=boxes_to_mask(np.concatenate([d["dyn"],d["sta"]])) if len(d["sta"]) else dyn_gt
        oi.append(MET.object_occ_iou(pred,dyn_gt)); of.append(float((pred!=17).mean()))
        r,pr=MET.object_recall_precision(pred,dyn_gt); rec.append(r); prc.append(pr)
        gi.append(MET.geo_iou(pred,np.where(all_gt,4,17)))
    print(f"\n=== FlashOcc BEVStereo4D (nuScenes-trained, cam-only) on AV2 SURROUND renders, {len(fs)} frames ===")
    print(f"  object RECALL (occ where actor boxes are):          {np.mean(rec):.3f}")
    print(f"  object precision (of occ near boxes):               {np.mean(prc):.3f}")
    print(f"  object-occ IoU (pred occ vs actor boxes):           {np.mean(oi):.3f}")
    print(f"  geo-IoU vs all-box (coarse):                        {np.mean(gi):.3f}")
    print(f"  predicted occupied fraction:                        {np.mean(of):.3f}")

if __name__=="__main__": main()
