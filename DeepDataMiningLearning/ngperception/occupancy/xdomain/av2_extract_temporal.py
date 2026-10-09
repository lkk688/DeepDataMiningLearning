"""Extract AV2 surround-render TEMPORAL windows for BEVDet4D-Stereo teachers (FlashOcc/OPUS).
For each of N sampled keyframes, saves a `num_frame`-frame history (current + prev at `dt_us` spacing):
per (frame,cam) image + intrinsics K + sensor2ego(4x4) + ego2global(4x4), plus current-frame actor
boxes (dyn/sta) in the current ego frame. Run with the alpasim venv (protobuf reader).
Saves <out>/f{n:03d}_t{f}_{cam}.jpg + <out>/f{n:03d}.npz.

npz keys: cams(C,), K(F,C,3,3), s2e(F,C,4,4) cam->ego, e2g(F,C,4,4) ego->world (frame 0 = current),
          dyn(Nd,7), sta(Ns,7) current-frame ego boxes [cx,cy,cz,yaw,sx,sy,sz]."""
import sys, os, io, asyncio, math, base64
import numpy as np
from PIL import Image
from alpasim_utils.logs import async_read_pb_log
from google.protobuf.json_format import MessageToDict
RING=["ring_front_center","ring_front_left","ring_front_right","ring_side_left","ring_side_right","ring_rear_left","ring_rear_right"]

def qmat(q):
    w,x,y,z=q.get("w",1),q.get("x",0),q.get("y",0),q.get("z",0)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def pose_RT(p):
    v=p["vec"]; return qmat(p["quat"]), np.array([v.get("x",0.),v.get("y",0.),v.get("z",0.)])
def T44(R,t):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=t; return M

async def run(asl, out, nframes=12, num_frame=3, dt_us=500000, lidar_dump=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out, exist_ok=True)
    renders={}; images={}; egoposes={}; sizes={}; static={}; actorposes={}
    async for e in async_read_pb_log(asl):
        w=e.WhichOneof(e.DESCRIPTOR.oneofs[0].name); d=MessageToDict(e)
        if w=="render_request":
            r=d["renderRequest"]; ci=r["cameraIntrinsics"]; op=ci["opencvPinholeParam"]; cam=ci["logicalId"]; t=int(r["frameStartUs"])
            K=np.array([[op["focalLengthX"],0,op["principalPointX"]],[0,op["focalLengthY"],op["principalPointY"]],[0,0,1]])
            Rcw,tcw=pose_RT(r["sensorPose"]["startPose"])
            renders.setdefault(t,{})[cam]=(K,Rcw,tcw)
        elif w=="driver_camera_image":
            c=d["driverCameraImage"]["cameraImage"]; t=int(c["frameStartUs"]); images.setdefault(t,{})[c["logicalId"]]=c["imageBytes"]
        elif w=="actor_poses":
            ap=d["actorPoses"]; t=int(ap["timestampUs"]); m={}
            for a in ap["actorPoses"]:
                R,tt=pose_RT(a["actorPose"]); m[a["actorId"]]=(R,tt)
                if a["actorId"]=="EGO": egoposes[t]=(R,tt)
            actorposes[t]=m
        elif w=="traffic_session_request":
            for o in d["trafficSessionRequest"].get("loggedObjectTrajectories",[]):
                a=o["aabb"]; sizes[o["objectId"]]=np.array([a["sizeX"],a["sizeY"],a["sizeZ"]]); static[o["objectId"]]=bool(o.get("isStatic",False))
    rts=[t for t in sorted(renders) if len([c for c in RING if c in renders[t] and t in images and c in images[t]])>=6]
    def nearest(ts,t): return min(ts,key=lambda x:abs(x-t))
    # optional LiDAR dense GT (Phase 4 §4.5): index lidar_dump/lidar_<t>_<cam>.npy by timestamp
    lidar_ts=None
    if lidar_dump and os.path.isdir(lidar_dump):
        import re, glob as _g
        m={}
        for p in _g.glob(os.path.join(lidar_dump,"lidar_*_*.npy")):
            mt=re.match(r"lidar_(\d+)_",os.path.basename(p))
            if mt: m.setdefault(int(mt.group(1)),p)
        lidar_ts=m
        print(f"[lidar] indexed {len(m)} lidar timestamps")
    # keyframes need num_frame history at dt spacing -> only pick t with enough past render coverage
    lo=rts[0]+dt_us*(num_frame-1)
    cand=[t for t in rts if t>=lo]
    idxs=np.linspace(0,len(cand)-1,min(nframes,len(cand))).astype(int)
    n=0
    for gi in idxs:
        t0=cand[gi]
        frame_ts=[nearest(rts,t0-dt_us*f) for f in range(num_frame)]   # [curr, prev1, prev2]
        cams=[c for c in RING if c in renders[t0] and c in images[t0]]
        K=np.zeros((num_frame,len(cams),3,3)); s2e=np.zeros((num_frame,len(cams),4,4)); e2g=np.zeros((num_frame,len(cams),4,4))
        ok=True
        for fi,ft in enumerate(frame_ts):
            et=nearest(egoposes,ft); Reg,teg=egoposes[et]
            for ci,c in enumerate(cams):
                if c not in renders[ft] or ft not in images or c not in images[ft]: ok=False; break
                Kc,Rcw,tcw=renders[ft][c]
                Rce=Reg.T@Rcw; tce=Reg.T@(tcw-teg)               # cam->ego (this frame's ego)
                K[fi,ci]=Kc; s2e[fi,ci]=T44(Rce,tce); e2g[fi,ci]=T44(Reg,teg)
                raw=images[ft][c]; b=base64.b64decode(raw) if isinstance(raw,str) else raw
                Image.open(io.BytesIO(bytes(b))).convert("RGB").save(f"{out}/f{n:03d}_t{fi}_{c}.jpg")
            if not ok: break
        if not ok: continue
        # current-frame boxes in current ego
        et=nearest(egoposes,t0); Reg,teg=egoposes[et]; at=nearest(actorposes,t0); apos=actorposes[at]
        dyn=[]; sta=[]
        for oid,(R,tt) in apos.items():
            if oid=="EGO" or oid not in sizes: continue
            ce=Reg.T@(tt-teg); yaw=math.atan2((Reg.T@R)[1,0],(Reg.T@R)[0,0])
            rec=(ce[0],ce[1],ce[2],yaw,*sizes[oid])
            (sta if static.get(oid) else dyn).append(rec)
        extra={}
        if lidar_ts:
            lt=nearest(list(lidar_ts.keys()),t0)
            if abs(lt-t0)<=60000:                            # within 60ms
                from lidar_dense_gt import lidar_occ_free
                pts=np.load(lidar_ts[lt]).astype(np.float32)
                occ,free=lidar_occ_free(pts)
                extra["lidar_occ"]=np.packbits(occ); extra["lidar_free"]=np.packbits(free)
                extra["lidar_pts"]=pts                            # raw N×3 ego/rig points (for FM point-painting)
        np.savez(f"{out}/f{n:03d}.npz", cams=cams, K=K, s2e=s2e, e2g=e2g,
                 dyn=np.array(dyn,dtype=np.float32) if dyn else np.zeros((0,7),np.float32),
                 sta=np.array(sta,dtype=np.float32) if sta else np.zeros((0,7),np.float32), **extra)
        n+=1
    print(f"extracted {n} temporal keyframes (num_frame={num_frame}, dt={dt_us/1e6}s, {len(cams)} cams) -> {out}")

if __name__=="__main__":
    asl,out=sys.argv[1],sys.argv[2]
    nf=int(sys.argv[3]) if len(sys.argv)>3 else 12
    numf=int(sys.argv[4]) if len(sys.argv)>4 else 8    # temporal window (OPUS needs 8 key frames)
    ldump=sys.argv[5] if len(sys.argv)>5 else None     # <logdir>/lidar_dump for dense geometric GT
    asyncio.run(run(asl,out,nf,num_frame=numf,lidar_dump=ldump))
