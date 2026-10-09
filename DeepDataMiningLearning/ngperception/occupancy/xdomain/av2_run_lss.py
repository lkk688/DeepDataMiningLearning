"""Cross-domain eval: run our LSS occ model on extracted AV2 surround frames -> object-occ IoU vs
actor-box GT (class-agnostic) + occupied-fraction. Run with py310."""
import sys, os, glob, math, argparse, numpy as np, torch
from types import SimpleNamespace
from PIL import Image
sys.path.insert(0,"/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/ngperception/occupancy/xdomain")
import metrics as MET
VOX=0.4; ORIG=np.array([-40.,-40.,-1.]); GS=np.array([200,200,16])

def boxes_to_mask(boxes):
    """(N,7)[cx,cy,cz,yaw,sx,sy,sz] ego-frame -> (200,200,16) bool occupancy."""
    m=np.zeros(tuple(GS),bool)
    xs=ORIG[0]+(np.arange(200)+.5)*VOX; ys=ORIG[1]+(np.arange(200)+.5)*VOX; zs=ORIG[2]+(np.arange(16)+.5)*VOX
    X,Y,Z=np.meshgrid(xs,ys,zs,indexing="ij")
    for cx,cy,cz,yaw,sx,sy,sz in boxes:
        c,s=math.cos(-yaw),math.sin(-yaw)
        lx=(X-cx)*c-(Y-cy)*s; ly=(X-cx)*s+(Y-cy)*c; lz=Z-cz
        m|=(np.abs(lx)<=sx/2)&(np.abs(ly)<=sy/2)&(np.abs(lz)<=sz/2)
    return m

def build_inputs(frame, image_hw, dev):
    d=np.load(frame,allow_pickle=True); cams=list(d["cams"]); H,W=image_hw
    imgs=[]; Ks=[]
    for i,c in enumerate(cams):
        im=Image.open(frame.replace(".npz",f"_{c}.jpg")).convert("RGB"); W0,H0=im.size
        im=im.resize((W,H),Image.BILINEAR)
        arr=np.asarray(im).astype(np.float32)/255.
        arr=(arr-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
        imgs.append(arr.transpose(2,0,1))
        K=d["K"][i].copy(); K[0]*=W/W0; K[1]*=H/H0; Ks.append(K)
    t=lambda a: torch.tensor(np.array(a),dtype=torch.float32,device=dev)[None]
    return (t(imgs), t(d["rots"]), t(d["trans"]), t(Ks)), d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--frames",required=True)
    ap.add_argument("--ckpt",default="/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/ngperception/output/lss_occ_full/lss_occ.pth")
    ap.add_argument("--backbone",default="dinov2_base"); a=ap.parse_args()
    from DeepDataMiningLearning.ngperception.occupancy.visualize import build_model
    dev="cuda"; args=SimpleNamespace(backbone=a.backbone,decoder_hidden=96,decoder_layers=4,refine_iters=2)
    m=build_model(a.ckpt,args,False,dev)
    print("model image_hw",m.image_hw)
    fs=sorted(glob.glob(os.path.join(a.frames,"f*.npz")))
    oi=[]; of=[]; gi=[]; rec=[]; prc=[]
    for f in fs:
        (imgs,rots,trans,intr),d=build_inputs(f,m.image_hw,dev)
        with torch.no_grad():
            occ=m(imgs,rots,trans,intr)[0]
        pred=occ.argmax(1)[0].cpu().numpy()
        dyn_gt=boxes_to_mask(d["dyn"]); all_gt=boxes_to_mask(np.concatenate([d["dyn"],d["sta"]])) if len(d["sta"]) else dyn_gt
        oi.append(MET.object_occ_iou(pred,dyn_gt)); of.append(float((pred!=17).mean()))
        r,pr=MET.object_recall_precision(pred,dyn_gt); rec.append(r); prc.append(pr)
        gi.append(MET.geo_iou(pred,np.where(all_gt,4,17)))  # all-box as coarse geo GT
    print(f"\n=== our LSS (nuScenes-trained camera-only) on AV2 SURROUND renders, {len(fs)} frames ===")
    print(f"  object RECALL (occ where actor boxes are):          {np.mean(rec):.3f}")
    print(f"  object precision (of occ near boxes):               {np.mean(prc):.3f}")
    print(f"  object-occ IoU (pred dynamic-class vs actor boxes): {np.mean(oi):.3f}")
    print(f"  geo-IoU vs all-box (coarse):                        {np.mean(gi):.3f}")
    print(f"  predicted occupied fraction:                        {np.mean(of):.3f}")
    print(f"  (nuScenes ref: object classes contribute to mIoU 0.302; occ-frac ~0.05 on GT)")
if __name__=="__main__": main()
