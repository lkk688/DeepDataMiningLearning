"""Occupancy metrics for cross-domain eval. Works with (200,200,16) Occ3D-nuScenes grids (18-class,
17=free). Semantic mIoU needs dense semantic GT (nuScenes only); geometric IoU + object-occ IoU need
only geometry / object masks (available on the AV2 renders)."""
import numpy as np
N_CLS=18; FREE=17
# dynamic object classes (available on AV2 via actor boxes): car,truck,bus,trailer,constr,ped,motorcycle,bicycle
OBJ_CLASSES=[2,3,4,5,6,7,9,10]

def geo_iou(pred, gt, mask=None):
    """class-agnostic occupied/free IoU."""
    p=(pred!=FREE); g=(gt!=FREE)
    if mask is not None: p=p&mask; g=g&mask
    inter=(p&g).sum(); union=(p|g).sum()
    return float(inter/union) if union else 0.0

def semantic_miou(pred, gt, mask=None):
    """per-class IoU over the 17 semantic classes; returns (miou, per_class dict)."""
    ious={}
    for c in range(FREE):
        p=(pred==c); g=(gt==c)
        if mask is not None: p=p&mask; g=g&mask
        u=(p|g).sum()
        if g.sum()==0 and p.sum()==0: continue          # class absent -> skip (Occ3D convention)
        ious[c]=float((p&g).sum()/u) if u else 0.0
    miou=float(np.mean(list(ious.values()))) if ious else 0.0
    return miou, ious

def object_occ_iou(pred, obj_gt_mask, mask=None):
    """IoU of predicted dynamic-object voxels vs an object-occupancy GT mask (bool grid).
    obj_gt_mask: (200,200,16) bool = voxels inside any actor box."""
    p=np.isin(pred,OBJ_CLASSES)
    if mask is not None: p=p&mask; obj_gt_mask=obj_gt_mask&mask
    u=(p|obj_gt_mask).sum()
    return float((p&obj_gt_mask).sum()/u) if u else 0.0

def summarize(pred, gt=None, obj_gt=None, cam_mask=None):
    out={}
    if gt is not None:
        out["geo_iou"]=geo_iou(pred,gt,cam_mask)
        m,_=semantic_miou(pred,gt,cam_mask); out["miou"]=m
    if obj_gt is not None:
        out["obj_iou"]=object_occ_iou(pred,obj_gt,cam_mask)
    out["occ_frac"]=float((pred!=FREE).mean())
    return out

def object_recall_precision(pred, obj_gt_mask, mask=None):
    """recall = fraction of object voxels the model predicts OCCUPIED (any class); precision = of the
    model's occupied voxels inside the object region, fraction that are objects. Recall answers
    'does it detect the object', robust to global over-prediction."""
    occ=(pred!=FREE)
    if mask is not None: occ=occ&mask; obj_gt_mask=obj_gt_mask&mask
    tp=(occ&obj_gt_mask).sum(); g=obj_gt_mask.sum(); p=occ.sum()
    return float(tp/g) if g else 0.0, float(tp/p) if p else 0.0

def object_class_recall(pred, obj_gt_mask, mask=None):
    """DISCRIMINATIVE cross-domain metric (robust to dense over-prediction of ground/buildings):
    fraction of actor-box voxels predicted as a FOREGROUND OBJECT class (OBJ_CLASSES), i.e. the model
    recognizes the obstacle *as an object*, not merely 'occupied' by drivable/terrain/manmade. Also
    returns the global foreground-object-class fraction as the chance-level reference."""
    fg=np.isin(pred,OBJ_CLASSES)
    if mask is not None: fg=fg&mask; obj_gt_mask=obj_gt_mask&mask
    g=obj_gt_mask.sum()
    return (float((fg&obj_gt_mask).sum()/g) if g else 0.0, float(fg.mean()))
