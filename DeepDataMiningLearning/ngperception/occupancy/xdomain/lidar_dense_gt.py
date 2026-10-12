"""Build dense GEOMETRIC occupancy GT from the sim's GT-geometry LiDAR (Phase 4 / §4.5). Points are in
the ego/rig frame (x fwd, y left, z up) — same frame as the Occ3D grid. Returns per-voxel labels:
  occupied: voxel contains >=1 LiDAR return
  free    : voxel is traversed by a ray from the sensor origin to a return (before the endpoint)
  unknown : neither (occluded / beyond range) -> not supervised
Used to add a FREE-SPACE (precision) term to the LSS fine-tune and, later, FusionOcc's LiDAR input."""
import numpy as np
VOX=0.4; ORIG=np.array([-40.,-40.,-1.]); GS=np.array([200,200,16])

def _vox_idx(pts):
    ijk=np.floor((pts-ORIG)/VOX).astype(np.int64)
    ok=np.all((ijk>=0)&(ijk<GS),axis=1)
    return ijk[ok], ok

def lidar_occ_free(points, step=0.4, max_range=45.0):
    """points (N,3) ego frame -> (occupied (200,200,16) bool, free bool). Ray-march from origin."""
    occ=np.zeros(tuple(GS),bool); free=np.zeros(tuple(GS),bool)
    p=points[np.isfinite(points).all(1)]
    d=np.linalg.norm(p,axis=1)
    p=p[(d>1e-3)&(d<max_range)]; d=np.linalg.norm(p,axis=1)
    # occupied = endpoint voxels
    ijk,ok=_vox_idx(p); occ[ijk[:,0],ijk[:,1],ijk[:,2]]=True
    # free = sampled voxels along each ray from origin to (d-step)
    dirs=p/d[:,None]
    maxn=int(np.ceil(d.max()/step))
    for s in range(maxn):
        t=(s+0.5)*step
        m=t<(d-step*0.5)                      # stop one step short of the return
        if not m.any(): continue
        samp=dirs[m]*t
        ii,okk=_vox_idx(samp)
        free[ii[:,0],ii[:,1],ii[:,2]]=True
    free&=~occ                                # occupied overrides free
    return occ, free

if __name__=="__main__":
    import sys, glob, os
    dd=sys.argv[1]  # a lidar_dump dir
    f=sorted(glob.glob(dd+"/*.npy"))[0]; pts=np.load(f)
    occ,free=lidar_occ_free(pts)
    print(f"lidar dense GT from {os.path.basename(f)} ({len(pts)} pts):")
    print(f"  occupied voxels: {occ.sum()} ({occ.mean():.3f})   free voxels: {free.sum()} ({free.mean():.3f})")
    print(f"  unknown (unsupervised): {1-occ.mean()-free.mean():.3f}")
    # z-profile: ground (low z) should be mostly occupied
    for z in [0,4,8,12]:
        print(f"  z-slice {z} (h={ORIG[2]+z*VOX:.1f}m): occ {occ[:,:,z].mean():.3f} free {free[:,:,z].mean():.3f}")
