"""In-domain reference for OPUSv2-L: run it via its NATIVE OPUSNuScenesDataset (correct ego2img) on
nuScenes Occ3D val, computing the same FG_REC/obj_IoU metrics used on AV2 (object GT = Occ3D semantics
in OBJ_CLASSES, camera-mask). Proves the model+checkpoint work and gives the retention denominator, so
the AV2 collapse (occ_frac 0.025) can be attributed to domain gap rather than the adapter. Run in py310."""
import sys, os, glob, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import av2_common as C, metrics as MET
NUSC_ROOT="/data/rnd-liu/Datasets/nuScenes/v1.0-trainval"; OCC_ROOT=NUSC_ROOT+"/gts"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=12); ap.add_argument("--out",required=True)
    a=ap.parse_args(); dev="cuda"; os.makedirs(a.out,exist_ok=True)
    from mmengine.registry import init_default_scope; init_default_scope("mmdet3d")
    from projects.OPUS.opus import OPUSInference, OPUSNuScenesDataset, sparse2dense
    CKPT="/fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/opusv2-l_r50_704x256_8f_nusc-occ3d_100e.pth"
    model=OPUSInference(variant="l"); model.load_checkpoint(CKPT); model=model.to(dev).eval()
    ds=OPUSNuScenesDataset(num_frames=8, run_offline=True, limit=a.n)
    pf=[]
    for i in range(min(a.n,len(ds))):
        s=ds[i]; tok=s.get("token"); scene=s.get("scene_name")
        gtp=os.path.join(OCC_ROOT,scene,tok,"labels.npz")
        if not os.path.exists(gtp): continue
        g=np.load(gtp); sem=g["semantics"]; mcam=g["mask_camera"].astype(bool)
        model.reset()
        out=model.infer(s["imgs"].to(dev),[dict(s["meta"])],image_loader=s.get("load_frame"))[0]
        pred,_=sparse2dense(out["occ_loc"],out["sem_pred"],dense_shape=(200,200,16),empty_value=17)
        pred=np.asarray(pred); obj_gt=np.isin(sem,MET.OBJ_CLASSES)
        r_occ,_=MET.object_recall_precision(pred,obj_gt,mcam); fg_rec,fg_frac=MET.object_class_recall(pred,obj_gt,mcam)
        pf.append(dict(occ_frac=float(((pred!=17)&mcam).sum()/mcam.sum()),fg_frac=fg_frac,occ_recall=r_occ,
                       fg_recall=fg_rec,obj_iou=MET.object_occ_iou(pred,obj_gt,mcam),geo_iou=0.0))
    m=C.save_results(os.path.join(a.out,"opus_INDOMAIN.json"),"OPUSv2-L (cam,8f)",pf)
    print(f"OPUSv2-L IN-DOMAIN (nuScenes, {len(pf)} frames): FG_REC {m['fg_recall']:.3f}  occ_frac {m['occ_frac']:.3f}  obj_IoU {m['obj_iou']:.3f}")

if __name__=="__main__": main()
