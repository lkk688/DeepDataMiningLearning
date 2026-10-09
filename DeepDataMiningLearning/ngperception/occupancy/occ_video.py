"""Temporal occupancy video: front camera + predicted dense 3D occupancy (fine-tuned DINOv2 LSS on AV2
renders), side by side, frame by frame. Reads occ_XXXX.npy / cam_XXXX.jpg (from predict_occ_sequence.py).
Run with the headless vizenv (pyvista + vtk-osmesa). ALWAYS `unset DISPLAY` first.
  vizenv/bin/python occ_video.py --frames /tmp/shield/occ_video/0d37aee4 --out /tmp/shield/occ_video/occ_0d37aee4.mp4"""
import os, sys, glob, argparse
os.environ["PYVISTA_OFF_SCREEN"]="true"; os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW","vtkOSOpenGLRenderWindow")
import numpy as np, pyvista as pv, imageio.v2 as imageio
from PIL import Image, ImageDraw
pv.OFF_SCREEN=True

NAMES=["others","barrier","bicycle","bus","car","constr.veh","motorcycle","pedestrian","traffic_cone",
       "trailer","truck","driveable","other_flat","sidewalk","terrain","manmade","vegetation","free"]
CMAP=np.array([[50,50,50],[255,120,50],[255,192,203],[255,255,0],[0,150,245],[0,255,255],[200,180,0],
    [255,0,0],[255,240,150],[135,60,0],[160,32,240],[255,0,255],[175,0,75],[75,0,75],[150,240,80],
    [230,230,250],[0,175,0],[255,255,255]],dtype=np.uint8)
VOX=0.4; X0=Y0=-40.0; Z0=-1.0; CROP=24.0

def render_occ(sem):
    """Camera-aligned chase view (ego at bottom-center, +x forward going into the scene) so it lines up
    with the front camera. Ego = black box + yellow heading cone; small region around the ego cleared so
    the marker is visible; no edge lines; tight forward crop."""
    plotter=pv.Plotter(off_screen=True, window_size=(900,760), lighting="light kit")  # fresh per frame (OSMesa)
    nx,ny,nz=sem.shape
    s=sem.copy()
    ix=slice(int((-2.6-X0)/VOX),int((2.6-X0)/VOX)); iy=slice(int((-1.5-Y0)/VOX),int((1.5-Y0)/VOX))
    s[ix,iy,:]=17                                               # clear voxels around ego so marker shows
    grid=pv.ImageData(dimensions=(nx+1,ny+1,nz+1), spacing=(VOX,VOX,VOX), origin=(X0,Y0,Z0))
    grid.cell_data["sem"]=s.flatten(order="F")
    occ=grid.threshold([0.5,16.5], scalars="sem")               # drop free(17) & others(0)
    if occ.n_cells:
        occ=occ.clip_box([-CROP,CROP,-CROP,CROP,Z0,Z0+16*VOX], invert=False)
    if occ.n_cells:
        lab=np.clip(occ["sem"].astype(int),0,16); occ["rgb"]=CMAP[lab]
        plotter.add_mesh(occ, scalars="rgb", rgb=True, ambient=0.45, diffuse=0.72, specular=0.1,
                         smooth_shading=False)
    plotter.add_mesh(pv.Cube(center=(0,0,Z0+0.7), x_length=4.5,y_length=2.0,z_length=1.6), color=(0.05,0.05,0.05))
    plotter.add_mesh(pv.Cone(center=(3.4,0,Z0+0.8), direction=(1,0,0), height=1.8, radius=0.8), color=(1,0.9,0))
    plotter.set_background("white")
    plotter.camera_position=[(-16,0,11),(16,0,1),(0,0,1)]        # behind+above ego, looking forward (+x up)
    img=np.asarray(plotter.screenshot(return_img=True))
    plotter.close(); pv.close_all()
    import gc; gc.collect()
    return img

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--frames",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--fps",type=int,default=10); a=ap.parse_args()
    occs=sorted(glob.glob(os.path.join(a.frames,"occ_*.npy")))
    W=1680; H=760; CAMW=760
    writer=imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8)
    for i,op in enumerate(occs):
        sem=np.load(op); occ_img=render_occ(sem)
        cam_p=os.path.join(os.path.dirname(op),os.path.basename(op).replace("occ_","cam_").replace(".npy",".jpg"))
        canvas=Image.new("RGB",(W,H),(20,20,20))
        if os.path.exists(cam_p):
            cam=Image.open(cam_p).convert("RGB")                 # portrait 1550x2048
            cam=cam.resize((int(cam.width*H/cam.height),H))      # fit height 760 (keeps aspect)
            canvas.paste(cam,(max(0,(CAMW-cam.width)//2),0))     # center in the 760-wide left panel
        canvas.paste(Image.fromarray(occ_img),(CAMW+20,0))
        d=ImageDraw.Draw(canvas)
        d.text((30,10),"ring_front_center (AV2 NuRec render)",fill=(255,255,0))
        d.text((CAMW+30,10),f"predicted occupancy (fine-tuned DINOv2 LSS) - black=ego, yellow=heading   frame {i:03d}",fill=(0,0,0))
        writer.append_data(np.asarray(canvas))
        if i%20==0: print(f"  {i}/{len(occs)}",flush=True)
    writer.close(); print(f"[occ_video] {len(occs)} frames -> {a.out}")

if __name__=="__main__": main()
