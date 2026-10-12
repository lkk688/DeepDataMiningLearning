"""In-domain reference + validation for FusionOcc (cam+LiDAR fusion, Swin-B). Runs it via its NATIVE
FusionOccDataset on nuScenes val, computing the same FG_REC/obj_IoU used elsewhere (object GT = Occ3D
semantics in OBJ_CLASSES, camera-mask). Confirms the model+checkpoint work in this env and gives the
retention denominator BEFORE investing in an AV2 LiDAR render. Run in py310 (mmdet3d scope)."""
import sys, os, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import av2_common as C, metrics as MET
INFOS="/data/rnd-liu/Datasets/nuScenes/v1.0-trainval/fusionocc-nuscenes_infos_val.pkl"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=12); ap.add_argument("--out",required=True)
    a=ap.parse_args(); dev="cuda"; os.makedirs(a.out,exist_ok=True)
    from mmengine.registry import init_default_scope; init_default_scope("mmdet3d")
    from projects.FusionOcc.fusionocc import build_fusionocc
    from projects.FusionOcc.fusionocc.data import FusionOccDataset
    model=build_fusionocc(device=dev); ds=FusionOccDataset(INFOS)
    pf=[]
    for i in range(min(a.n,len(ds))):
        s=ds[i]; occ_path=s.get("occ_path")
        if occ_path and os.path.isdir(occ_path): occ_path=os.path.join(occ_path,"labels.npz")
        if not occ_path or not os.path.exists(occ_path): continue
        g=np.load(occ_path); sem=g["semantics"]; mcam=g["mask_camera"].astype(bool)
        with torch.no_grad():
            logits=model.forward_occ_logits(points=[s["points"][0].to(dev)],
                img_inputs=[t.to(dev) for t in s["img_inputs"]], sparse_depth=[s["sparse_depth"].to(dev)])
            pred=logits.softmax(-1).max(-1)[1][0].to(torch.uint8).cpu().numpy()   # (200,200,16)
        obj_gt=np.isin(sem,MET.OBJ_CLASSES)
        r_occ,_=MET.object_recall_precision(pred,obj_gt,mcam); fg_rec,fg_frac=MET.object_class_recall(pred,obj_gt,mcam)
        pf.append(dict(occ_frac=float(((pred!=17)&mcam).sum()/mcam.sum()),fg_frac=fg_frac,occ_recall=r_occ,
                       fg_recall=fg_rec,obj_iou=MET.object_occ_iou(pred,obj_gt,mcam),geo_iou=0.0))
    m=C.save_results(os.path.join(a.out,"fusionocc_INDOMAIN.json"),"FusionOcc (cam+LiDAR)",pf)
    print(f"FusionOcc IN-DOMAIN (nuScenes, {len(pf)} frames): FG_REC {m['fg_recall']:.3f}  occ_frac {m['occ_frac']:.3f}  obj_IoU {m['obj_iou']:.3f}")

if __name__=="__main__": main()
