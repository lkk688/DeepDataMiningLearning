"""Run the trained LSS occupancy model on a nuScenes scene's cameras -> predicted (200,200,16)
semantics per frame, saved as <token>.npy. Camera-only by default (the real camera->occ estimate)."""
import argparse, os, numpy as np, torch
from types import SimpleNamespace
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scene",required=True); ap.add_argument("--outdir",required=True)
    ap.add_argument("--ckpt",default="output/lss_occ_full/lss_occ.pth")
    ap.add_argument("--gts",default="/data/rnd-liu/Datasets/nuScenes/v1.0-trainval/gts")
    ap.add_argument("--nusc",default="/data/rnd-liu/Datasets/nuScenes/v1.0-trainval")
    ap.add_argument("--backbone",default="dinov2_base"); ap.add_argument("--decoder-hidden",type=int,default=96)
    ap.add_argument("--decoder-layers",type=int,default=4); ap.add_argument("--refine-iters",type=int,default=2)
    ap.add_argument("--fusion",action="store_true")
    a=ap.parse_args()
    from nuscenes import NuScenes
    from DeepDataMiningLearning.ngperception.occupancy.datasets_train import NuScenesOccTrainDataset
    from DeepDataMiningLearning.ngperception.occupancy.visualize import build_model, infer
    dev="cuda"
    args=SimpleNamespace(backbone=a.backbone,decoder_hidden=a.decoder_hidden,
                         decoder_layers=a.decoder_layers,refine_iters=a.refine_iters)
    m=build_model(a.ckpt,args,a.fusion,dev)
    print("model loaded; image_hw",m.image_hw,"downsample",m.downsample)
    nusc=NuScenes(version="v1.0-trainval",dataroot=a.nusc,verbose=False)
    ds=NuScenesOccTrainDataset(a.gts,nusc,image_hw=m.image_hw,downsample=m.downsample,
                               scenes=[a.scene],lidar_fusion=a.fusion)
    os.makedirs(a.outdir,exist_ok=True); n=len(ds)
    for i in range(n):
        tok=ds.occ.items[i][1]
        pred=infer(m,ds,i,dev,drop_lidar=not a.fusion)
        np.save(os.path.join(a.outdir,f"{tok}.npy"),pred.astype(np.uint8))
        print(f"  {i+1}/{n} {tok[:10]} occ_voxels={int((pred!=17).sum())}",end="\r")
    print(f"\ndone: {n} predictions -> {a.outdir}")
if __name__=="__main__": main()
