#!/usr/bin/env python3
"""3D voxel rendering of DENSE SEMANTIC occupancy (Occ3D-nuScenes), the classic colored-voxel style.

This is the *perception* dense-semantic occupancy (200x200x16, 18 classes) from the Occ3D GT (or an
LSS/BEV model prediction of the same shape) — NOT the closed-loop reflex's actor-box occupancy. Range
[-40,40]x[-40,40]x[-1,5.4] m at 0.4 m voxels.

Usage:
  visualize_voxel3d.py <labels.npz | pred.npy> <out.png> [--camera-mask] [--crop M] [--elev E --azim A]
"""
import sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Occ3D-nuScenes 18-class palette (0..16 occupied, 17 = free). Standard SurroundOcc/Occ3D colors.
NAMES = ["others","barrier","bicycle","bus","car","constr.veh","motorcycle","pedestrian",
         "traffic_cone","trailer","truck","driveable","other_flat","sidewalk","terrain",
         "manmade","vegetation","free"]
CMAP = np.array([
    [ 50, 50, 50],[255,120, 50],[255,192,203],[255,255,  0],[  0,150,245],[  0,255,255],
    [200,180,  0],[255,  0,  0],[255,240,150],[135, 60,  0],[160, 32,240],[255,  0,255],
    [175,  0, 75],[ 75,  0, 75],[150,240, 80],[230,230,250],[  0,175,  0],[255,255,255],
], dtype=float)/255.0
VOX=0.4; X0=Y0=-40.0; Z0=-1.0

def load(path):
    if path.endswith(".npz"):
        d=np.load(path); return d["semantics"], (d["mask_camera"] if "mask_camera" in d.files else None)
    return np.load(path), None

def downsample(sem, f):
    """majority-class downsample by factor f (ignoring free=17 so objects survive)."""
    X,Y,Z=sem.shape; X,Y,Z=X//f*f,Y//f*f,Z//f*f
    b=sem[:X,:Y,:Z].reshape(X//f,f,Y//f,f,Z//f,f).transpose(0,2,4,1,3,5).reshape(X//f,Y//f,Z//f,-1)
    counts=np.stack([(b==c).sum(-1) for c in range(17)],-1)
    out=np.full(b.shape[:3],17,np.uint8); occ=counts.sum(-1)>0
    out[occ]=counts.argmax(-1)[occ].astype(np.uint8); return out

def render_voxels(sem, out, crop, elev, azim, title, ds):
    """solid shaded cubes via matplotlib ax.voxels (no external GL)."""
    if ds>1: sem=downsample(sem,ds)
    vox=VOX*ds
    filled=(sem!=17)&(sem!=0)
    nx,ny,nz=sem.shape
    # crop indices to |x|,|y|<crop
    if crop>0:
        xi=np.abs(X0+(np.arange(nx)+.5)*vox)<crop; yi=np.abs(Y0+(np.arange(ny)+.5)*vox)<crop
        keepx=np.where(xi)[0]; keepy=np.where(yi)[0]
        sem=sem[keepx[0]:keepx[-1]+1, keepy[0]:keepy[-1]+1]; filled=filled[keepx[0]:keepx[-1]+1, keepy[0]:keepy[-1]+1]
        nx,ny,nz=sem.shape
    fc=np.zeros(sem.shape+(4,)); fc[...,:3]=CMAP[np.clip(sem,0,16)]; fc[...,3]=np.where(filled,1.0,0.0)
    fig=plt.figure(figsize=(12,8)); ax=fig.add_subplot(111,projection="3d")
    ax.voxels(filled, facecolors=fc, edgecolor=(0,0,0,0.08), linewidth=0.15, shade=True)
    ax.set_box_aspect((nx,ny,nz*3.0)); ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_axis_off()
    present=sorted(set(sem[filled].tolist()))
    ax.legend(handles=[Patch(fc=CMAP[c], label=NAMES[c]) for c in present],
              loc="upper left", fontsize=9, framealpha=.9)
    ax.set_title(title or f"Dense semantic occupancy (Occ3D GT) — solid voxels, {vox:.1f} m", fontsize=12)
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"[voxels] {int(filled.sum())} cubes ({vox:.1f} m) classes {present} -> {out}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--camera-mask", action="store_true", help="show only camera-visible voxels (eval region)")
    ap.add_argument("--crop", type=float, default=0.0, help="keep |x|,|y| < crop metres (0 = full 80 m)")
    ap.add_argument("--elev", type=float, default=28); ap.add_argument("--azim", type=float, default=-60)
    ap.add_argument("--title", default=None)
    ap.add_argument("--mode", choices=["scatter","voxels"], default="scatter")
    ap.add_argument("--downsample", type=int, default=2, help="voxels mode: majority-downsample factor")
    a=ap.parse_args()

    sem, mcam = load(a.src)
    if a.mode=="voxels":
        render_voxels(sem, a.out, a.crop, a.elev, a.azim, a.title, a.downsample); return
    occ = sem != 17
    if a.camera_mask and mcam is not None: occ &= mcam.astype(bool)
    ii,jj,kk = np.nonzero(occ)
    cls = sem[ii,jj,kk]
    keep = cls != 0                                   # drop 'others/noise'
    ii,jj,kk,cls = ii[keep],jj[keep],kk[keep],cls[keep]
    x = X0+(ii+.5)*VOX; y = Y0+(jj+.5)*VOX; z = Z0+(kk+.5)*VOX
    if a.crop>0:
        m=(np.abs(x)<a.crop)&(np.abs(y)<a.crop); x,y,z,cls=x[m],y[m],z[m],cls[m]
    col = CMAP[cls]

    fig=plt.figure(figsize=(11,7)); ax=fig.add_subplot(111,projection="3d")
    # draw ground/flat classes first (drivable/other_flat/sidewalk/terrain) then structures on top
    order = np.argsort(np.isin(cls,[11,12,13,14]).astype(int))[::-1]
    msize = 22 if a.crop and a.crop<=32 else 9      # bigger markers when cropped -> solid voxel look
    ax.scatter(x[order],y[order],z[order], c=col[order], marker="s", s=msize,
               depthshade=True, edgecolors="none")
    ax.scatter([0],[0],[Z0+0.8],c="k",marker="^",s=140)  # ego at origin
    R = a.crop if a.crop>0 else 40
    ax.set_xlim(-R,R); ax.set_ylim(-R,R); ax.set_zlim(Z0, Z0+16*VOX)
    ax.set_box_aspect((2*R, 2*R, 16*VOX*3.0))          # exaggerate z so structures read
    ax.view_init(elev=a.elev, azim=a.azim)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zticks([])
    present=sorted(set(cls.tolist()))
    ax.legend(handles=[Patch(fc=CMAP[c], label=NAMES[c]) for c in present],
              loc="upper left", bbox_to_anchor=(0.0,0.92), fontsize=8, ncol=1, framealpha=.9)
    ax.set_title(a.title or "Dense semantic occupancy (Occ3D-nuScenes GT), 3D voxels — 200x200x16, 0.4 m",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[voxel3d] {len(x)} occupied voxels, classes {present} -> {a.out}")

if __name__=="__main__":
    main()
