"""Shared config for the cross-domain occ transfer table. All nuScenes-trained teachers were trained on
6 surround cameras; several (OPUS/FusionOcc/GaussianOcc) hardcode `//6`. AV2 has 7 ring cameras, so we
standardize the WHOLE table on the same 6 AV2 cameras (identical input for every model = airtight), fed
with each camera's own calibration. Because BEV pooling is calibration-driven, the camera *order* /
nuScenes-slot identity is irrelevant to correctness — only the count (6) and per-camera calib matter.

Choice: keep front_center (forward, most decision-relevant) + the two front-diagonals + both sides +
one rear; drop ring_rear_right to reach 6 (7->6 is necessarily asymmetric). This subset is used by every
model, so the comparison is fair; the only cost is objects at the rear-right may be unobserved by all."""
import os, glob, math, json
import numpy as np
import metrics as MET

AV2_6 = ["ring_front_center","ring_front_left","ring_front_right","ring_side_left","ring_side_right","ring_rear_left"]
VOX=0.4; ORIG=np.array([-40.,-40.,-1.]); GS=(200,200,16)

def select6(npz_cams):
    """Return the indices into a saved npz `cams` array that pick AV2_6, in AV2_6 order. Raises if any
    of the 6 is missing (keeps the table honest: a scene lacking a required camera is skipped upstream)."""
    cams=list(npz_cams)
    return [cams.index(c) for c in AV2_6]

def boxes_to_mask(boxes):
    """(N,7)[cx,cy,cz,yaw,sx,sy,sz] ego-frame -> (200,200,16) bool occupancy."""
    m=np.zeros(GS,bool)
    xs=ORIG[0]+(np.arange(200)+.5)*VOX; ys=ORIG[1]+(np.arange(200)+.5)*VOX; zs=ORIG[2]+(np.arange(16)+.5)*VOX
    X,Y,Z=np.meshgrid(xs,ys,zs,indexing="ij")
    for cx,cy,cz,yaw,sx,sy,sz in boxes:
        c,s=math.cos(-yaw),math.sin(-yaw)
        lx=(X-cx)*c-(Y-cy)*s; ly=(X-cx)*s+(Y-cy)*c; lz=Z-cz
        m|=(np.abs(lx)<=sx/2)&(np.abs(ly)<=sy/2)&(np.abs(lz)<=sz/2)
    return m

def eval_pred(pred, dyn_gt, all_gt):
    """The row metrics (headline = fg_recall). pred (200,200,16) argmax classes, 17=free."""
    r_occ,_=MET.object_recall_precision(pred,dyn_gt)
    fg_rec,fg_frac=MET.object_class_recall(pred,dyn_gt)
    return dict(occ_frac=float((pred!=17).mean()), fg_frac=fg_frac, occ_recall=r_occ,
               fg_recall=fg_rec, obj_iou=MET.object_occ_iou(pred,dyn_gt),
               geo_iou=MET.geo_iou(pred,np.where(all_gt,4,17)))

def scene_frames(scene_dir):
    return sorted(glob.glob(os.path.join(scene_dir,"f*.npz")))

def gt_masks(npz):
    d=np.load(npz,allow_pickle=True)
    dyn=boxes_to_mask(d["dyn"]); allg=boxes_to_mask(np.concatenate([d["dyn"],d["sta"]])) if len(d["sta"]) else dyn
    return dyn,allg

def save_results(path, model, per_frame):
    """per_frame: list of metric dicts. Writes {model, n, mean:{...}} JSON."""
    keys=per_frame[0].keys() if per_frame else []
    mean={k:float(np.mean([r[k] for r in per_frame])) for k in keys}
    json.dump(dict(model=model, n=len(per_frame), mean=mean, per_frame=per_frame), open(path,"w"), indent=2)
    return mean
