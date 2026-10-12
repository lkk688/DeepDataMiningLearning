"""Parse an AV2 NuRec recon's rig_trajectories.usda -> per-camera OpenCV intrinsics + camera->rig
extrinsics (7 ring cameras), and the sensor-rig pose (ego->world) per timestamp. USD matrices are
row-major (row-vector convention): p_world = p_local @ M, so translation = M[3,:3], R = M[:3,:3]."""
import zipfile, re, numpy as np
RING=["ring_front_center","ring_front_left","ring_front_right","ring_side_left","ring_side_right",
      "ring_rear_left","ring_rear_right"]

def _mat(s):  # "( (..),(..),(..),(..) )" -> 4x4
    nums=[float(x) for x in re.findall(r'-?\d+\.?\d*(?:e-?\d+)?', s)]
    return np.array(nums[:16],dtype=np.float64).reshape(4,4)

def load_recon_calib(usdz_path):
    d=zipfile.ZipFile(usdz_path).read("rig_trajectories.usda").decode("utf-8","ignore")
    cams={}
    for name in RING:
        i=d.find(f'def Camera "{name}"')
        if i<0: continue
        blk=d[i:i+1600]
        g=lambda k: float(re.search(rf'{k}\s*=\s*(-?\d+\.?\d*)',blk).group(1))
        fx,fy,cx,cy=g("openCVFx"),g("openCVFy"),g("fthetaCx"),g("fthetaCy")
        W,H=g("fthetaWidth"),g("fthetaHeight")
        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        tm=re.search(r'matrix4d xformOp:transform\s*=\s*(\([^;]*?\)\s*\))',blk)
        cam2rig=_mat(tm.group(1))                          # row-major (p_rig = p_cam @ cam2rig)
        R=cam2rig[:3,:3].T                                 # -> column: p_rig = R @ p_cam + t
        t=cam2rig[3,:3]
        cams[name]=dict(K=K,R=R,t=t,W=int(W),H=int(H))
    # ego (sensor_rig) pose timeSamples
    k=d.find("sensor_rig_0"); s=d[k:k+400000]
    ts=re.search(r'xformOp:transform\.timeSamples\s*=\s*\{(.*?)\n\s*\}',s,re.S).group(1)
    ego={}
    for m in re.finditer(r'([\d.]+)\s*:\s*(\(\s*\([^}]*?\)\s*\))\s*,?',ts):
        try: ego[float(m.group(1))]=_mat(m.group(2))       # ego->world (row-major)
        except Exception: pass
    return cams, ego

if __name__=="__main__":
    import sys
    cams,ego=load_recon_calib(sys.argv[1])
    print(f"{len(cams)} cameras, {len(ego)} ego-pose keyframes")
    for n,c in cams.items():
        print(f"  {n}: {c['W']}x{c['H']} fx={c['K'][0,0]:.0f} cx={c['K'][0,2]:.0f} t={np.round(c['t'],2)}")
