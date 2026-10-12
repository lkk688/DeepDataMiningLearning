"""Extract per-frame AV2 surround-render data from a rollout.asl -> images + camera calib (K, cam->ego)
+ ego pose + dynamic/static actor boxes. Run with the alpasim venv (protobuf reader). Samples ~N frames.
Saves <out>/f{idx}_{cam}.jpg + <out>/f{idx}.npz."""
import sys, os, io, asyncio, math
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
    v=p["vec"]; return qmat(p["quat"]), np.array([v.get("x",0),v.get("y",0),v.get("z",0)])

async def run(asl, out, nframes):
    os.makedirs(out, exist_ok=True)
    renders={}   # frameStartUs -> {cam: (K, Rcw, tcw)}
    images={}    # frameStartUs -> {cam: jpgbytes}
    egoposes={}  # t -> (R,t)  from actor_poses EGO
    sizes={}; static={}  # objectId -> size, isStatic
    actorposes={}# t -> {id: (R,t)}
    async for e in async_read_pb_log(asl):
        w=e.WhichOneof(e.DESCRIPTOR.oneofs[0].name); d=MessageToDict(e)
        if w=="render_request":
            r=d["renderRequest"]; ci=r["cameraIntrinsics"]; op=ci["opencvPinholeParam"]; cam=ci["logicalId"]; t=int(r["frameStartUs"])
            K=np.array([[op["focalLengthX"],0,op["principalPointX"]],[0,op["focalLengthY"],op["principalPointY"]],[0,0,1]])
            Rcw,tcw=pose_RT(r["sensorPose"]["startPose"])
            renders.setdefault(t,{})[cam]=(K,Rcw,tcw,int(ci["resolutionW"]),int(ci["resolutionH"]))
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
    # frames that have all 7 renders + images
    good=[t for t in sorted(renders) if len([c for c in RING if c in renders[t] and t in images and c in images[t]])>=6]
    idxs=np.linspace(0,len(good)-1,min(nframes,len(good))).astype(int)
    n=0
    for gi in idxs:
        t=good[gi]
        # nearest ego/actor pose timestamp
        et=min(egoposes,key=lambda x:abs(x-t)); Reg,teg=egoposes[et]
        at=min(actorposes,key=lambda x:abs(x-t)); apos=actorposes[at]
        cams=[c for c in RING if c in renders[t] and c in images[t]]
        Ks=[]; rots=[]; trans=[]
        for c in cams:
            K,Rcw,tcw,W,H=renders[t][c]
            Image.open(io.BytesIO(bytes(__import__("base64").b64decode(images[t][c]) if isinstance(images[t][c],str) else images[t][c]))).convert("RGB").save(f"{out}/f{n:03d}_{c}.jpg")
            # cam->ego = world->ego @ cam->world ; ego->world = (Reg,teg)
            Rce=Reg.T@Rcw; tce=Reg.T@(tcw-teg)
            Ks.append(K); rots.append(Rce); trans.append(tce)
        # boxes in ego frame: dynamic (non-static, non-EGO) + static
        dyn=[]; sta=[]
        for oid,(R,tt) in apos.items():
            if oid=="EGO" or oid not in sizes: continue
            # object center in ego
            ce=Reg.T@(tt-teg); yaw=math.atan2((Reg.T@R)[1,0],(Reg.T@R)[0,0])
            rec=(ce[0],ce[1],ce[2],yaw,*sizes[oid])
            (sta if static.get(oid) else dyn).append(rec)
        np.savez(f"{out}/f{n:03d}.npz", cams=cams, K=np.array(Ks), rots=np.array(rots), trans=np.array(trans),
                 dyn=np.array(dyn,dtype=np.float32) if dyn else np.zeros((0,7),np.float32),
                 sta=np.array(sta,dtype=np.float32) if sta else np.zeros((0,7),np.float32))
        n+=1
    print(f"extracted {n} frames ({len(cams)} cams each) -> {out}")

if __name__=="__main__":
    asyncio.run(run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 16))
