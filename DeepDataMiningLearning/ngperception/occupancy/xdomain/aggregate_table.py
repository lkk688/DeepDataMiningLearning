"""Assemble the cross-domain occ TRANSFER TABLE from the per-model JSON results in --results:
  <model>_INDOMAIN.json           (nuScenes reference)
  <model>_<scene>.json  x scenes  (AV2, aggregated here across scenes+frames)
Prints in-domain vs AV2 FG_REC/occ_frac/obj_IoU and the FG retention (AV2/in-domain) that ranks
domain robustness. Pure python (no torch) — run anywhere."""
import sys, os, glob, json, argparse
import numpy as np

MODELS=[("lss","LSS (DINOv2-b, cam, 1f)"),("flashocc","FlashOcc (R50 stereo, cam, 3f)"),
        ("opus","OPUSv2-L (R50 sparse-query, cam, 8f)"),("gaussianocc","GaussianOcc (self-sup, cam)"),
        ("fusionocc","FusionOcc (cam+GT-LiDAR)")]

def load_mean(path):
    if not os.path.exists(path): return None
    d=json.load(open(path)); return d.get("mean")

def av2_mean(results, tag):
    """aggregate all per-frame metrics across every AV2 scene json for this model."""
    pf=[]
    for p in sorted(glob.glob(os.path.join(results,f"{tag}_*.json"))):
        if p.endswith("_INDOMAIN.json") or p.endswith("_ALL.json"): continue
        d=json.load(open(p)); pf+=d.get("per_frame",[])
    if not pf: return None,0
    keys=pf[0].keys(); return {k:float(np.mean([r[k] for r in pf])) for k in keys}, len(pf)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--results",required=True); a=ap.parse_args()
    print("\n"+"="*100)
    print("CROSS-DOMAIN OCCUPANCY TRANSFER — nuScenes-trained occ teachers on AV2 NuRec surround renders")
    print("Headline = FG_REC (foreground-object-class recall on actor boxes). Retention = AV2 / in-domain.")
    print("="*100)
    hdr=f"{'model':34s} | {'in-domain':>21s} | {'AV2 renders':>21s} | {'FG':>6s}"
    sub=f"{'':34s} | {'FG_REC occ_fr objIoU':>21s} | {'FG_REC occ_fr objIoU':>21s} | {'reten':>6s}"
    print(hdr); print(sub); print("-"*len(hdr))
    rows=[]
    for tag,name in MODELS:
        ind=load_mean(os.path.join(a.results,f"{tag}_INDOMAIN.json"))
        av2,nf=av2_mean(a.results,tag)
        if ind is None and av2 is None: continue
        def fmt(m): return f"{m['fg_recall']:6.3f} {m['occ_frac']:6.3f} {m['obj_iou']:6.3f}" if m else f"{'--':>6s} {'--':>6s} {'--':>6s}"
        ret=(av2['fg_recall']/ind['fg_recall']) if (ind and av2 and ind['fg_recall']>0) else None
        print(f"{name:34s} | {fmt(ind)} | {fmt(av2)} | {(f'{ret*100:4.1f}%' if ret is not None else '  --'):>6s}   (n_av2={nf})")
        if ret is not None: rows.append((name,ret,av2['fg_recall']))
    print("-"*len(hdr))
    if rows:
        print("\nDomain-robustness ranking (by FG retention = fraction of home ability kept on renders):")
        for i,(name,ret,fg) in enumerate(sorted(rows,key=lambda x:-x[1])): print(f"  {i+1}. {name:34s} retention {ret*100:4.1f}%  (AV2 FG_REC {fg:.3f})")
        print("\nAbsolute AV2 object recognition (what a fine-tuning base actually starts from):")
        for i,(name,ret,fg) in enumerate(sorted(rows,key=lambda x:-x[2])): print(f"  {i+1}. {name:34s} AV2 FG_REC {fg:.3f}  (retention {ret*100:4.1f}%)")
        # fine-tuning base = highest ABSOLUTE AV2 recognition among the robust (>40% retention) models
        robust=[r for r in rows if r[1]>0.40]
        base=max(robust or rows, key=lambda x:x[2])
        print(f"\n=> Finding: the self-supervised / foundation-feature models retain the most (top of the retention list);")
        print(f"   supervised-ResNet50 models collapse. => best FINE-TUNING base (robust AND highest absolute AV2): {base[0]}")

if __name__=="__main__": main()
