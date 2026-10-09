"""In-domain reference for the cross-domain table: run the SAME stereo FlashOcc (flashocc-r50-4d-stereo)
on nuScenes Occ3D val via build_img_inputs, computing the SAME metrics used on AV2, where the 'object
GT' = voxels whose Occ3D GT semantics are foreground OBJECT classes (OBJ_CLASSES). Answers whether the
AV2 numbers (occ_frac 0.83 / FG_REC 0.04) are a DOMAIN GAP or the model's intrinsic behavior.
Run in py310 with the bev_pool_v2 CUDA env."""
import sys, os, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import metrics as MET
NUSC="/data/rnd-liu/Datasets/nuScenes/v1.0-trainval"; GTS=NUSC+"/gts"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=12); ap.add_argument("--out",default=None); a=ap.parse_args(); dev="cuda"
    from nuscenes import NuScenes; from nuscenes.utils import splits
    from DeepDataMiningLearning.ngperception.occupancy.datasets_train import NuScenesOccTrainDataset
    from DeepDataMiningLearning.ngperception.flashocc.model_stereo import FlashOccBEVStereo4D, _CKPT_PATH
    from DeepDataMiningLearning.ngperception.flashocc.data_stereo import build_img_inputs
    nusc=NuScenes(version="v1.0-trainval", dataroot=NUSC, verbose=False)
    ds=NuScenesOccTrainDataset(GTS, nusc, image_hw=(256,704), downsample=16, scenes=sorted(splits.val),
                               max_samples=a.n, stride=1)
    m=FlashOccBEVStereo4D(pretrained_img=False)
    sd=torch.load(_CKPT_PATH,map_location="cpu",weights_only=False); m.load_state_dict(sd.get("state_dict",sd),strict=True)
    m=m.to(dev).eval()
    from types import SimpleNamespace
    from DeepDataMiningLearning.ngperception.occupancy.visualize import build_model
    lss=build_model("/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/ngperception/output/lss_occ_full/lss_occ.pth",
                    SimpleNamespace(backbone="dinov2_base",decoder_hidden=96,decoder_layers=4,refine_iters=2),False,dev)
    def score(pred,obj_gt,mcam):
        r_occ,_=MET.object_recall_precision(pred,obj_gt,mcam)
        fg_rec,fg_frac=MET.object_class_recall(pred,obj_gt,mcam)
        return (float(((pred!=17)&mcam).sum()/mcam.sum()),fg_frac,r_occ,fg_rec,MET.object_occ_iou(pred,obj_gt,mcam))
    accs={"FlashOcc":[], "LSS":[]}
    for i in range(len(ds)):
        tok=ds.occ[i].sample_token
        b=ds[i]; sem=b["semantics"].numpy() if torch.is_tensor(b["semantics"]) else b["semantics"]
        mcam=b["mask_camera"].numpy().astype(bool) if torch.is_tensor(b["mask_camera"]) else b["mask_camera"].astype(bool)
        obj_gt=np.isin(sem,MET.OBJ_CLASSES)                     # real object voxels (dense GT)
        inp=[t[None].to(dev) for t in build_img_inputs(nusc,tok)]
        with torch.no_grad(): pred=m(inp).argmax(1)[0].cpu().numpy()
        accs["FlashOcc"].append(score(pred,obj_gt,mcam))
        with torch.no_grad():
            pl=lss(b["imgs"][None].to(dev),b["rots"][None].to(dev),b["trans"][None].to(dev),b["intrins"][None].to(dev))
        accs["LSS"].append(score(pl[0].argmax(1)[0].cpu().numpy(),obj_gt,mcam))
    print(f"\n=== IN-DOMAIN reference (nuScenes Occ3D val, {len(ds)} frames, camera-mask) ===")
    print(f"  {'model':10s} {'occ_frac':>8s} {'fg_frac':>8s} {'occ_rec':>8s} {'FG_REC':>8s} {'obj_IoU':>8s}")
    import json, os as _os
    tag={"FlashOcc":"flashocc","LSS":"lss"}
    for k,v in accs.items():
        A=np.array(v).mean(0); print(f"  {k:10s} {A[0]:8.3f} {A[1]:8.3f} {A[2]:8.3f} {A[3]:8.3f} {A[4]:8.3f}")
        if getattr(a,"out",None):
            _os.makedirs(a.out,exist_ok=True)
            mean=dict(occ_frac=A[0],fg_frac=A[1],occ_recall=A[2],fg_recall=A[3],obj_iou=A[4])
            json.dump(dict(model=k,n=len(v),mean=mean),open(_os.path.join(a.out,f"{tag[k]}_INDOMAIN.json"),"w"),indent=2)

if __name__=="__main__": main()
