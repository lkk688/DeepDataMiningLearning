"""NVIDIA PhysicalAI-AV source for the label generator.

PhysicalAI is the hard case: **f-theta polynomial fisheye** intrinsics (not a pinhole K),
**Draco-compressed** LiDAR, and camera frames inside **mp4** videos that must be time-synced to the
LiDAR spins. The calibration + decode + f-theta projection here are copied (and generalized to
multi-camera) from our validated autolabel pipeline
(`PhysicalAI-Drive/physicalai_autolabel/scripts/annotate_clip.py`), so this Source is self-contained
in ngdet. Requires `DracoPy` (LiDAR) and OpenCV (`cv2.VideoCapture` decodes the mp4 — no PyAV needed).
"""
from __future__ import annotations
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image

# display_name -> PhysicalAI sensor name. Front trio are the 120°-FOV cameras (validated);
# rear pair are 70°-FOV. Each camera has its own f-theta `fw_poly`.
PAV_CAMS = [("FRONT_LEFT", "camera_cross_left_120fov"),
            ("FRONT", "camera_front_wide_120fov"),
            ("FRONT_RIGHT", "camera_cross_right_120fov"),
            ("BACK_LEFT", "camera_rear_left_70fov"),
            ("BACK_RIGHT", "camera_rear_right_70fov")]
LIDAR = "lidar_top_360fov"


def quat_to_R(qx, qy, qz, qw):
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def load_calib(root, clip, cam_names):
    """-> (intr_by_cam {name: {cx,cy,W,H,fw}}, ext {sensor: (R,t)}) for one clip."""
    ci = pd.concat([pd.read_parquet(p) for p in
                    glob.glob(f"{root}/calibration/camera_intrinsics/*.parquet")]).reset_index()
    se = pd.concat([pd.read_parquet(p) for p in
                    glob.glob(f"{root}/calibration/sensor_extrinsics/*.parquet")]).reset_index()
    intr = {}
    for cam in cam_names:
        rows = ci[(ci.clip_id == clip) & (ci.camera_name == cam)]
        if len(rows) == 0:
            continue
        r = rows.iloc[0]
        intr[cam] = dict(cx=r.cx, cy=r.cy, W=int(r.width), H=int(r.height),
                         fw=[r[f"fw_poly_{i}"] for i in range(5)])
    ext = {r.sensor_name: (quat_to_R(r.qx, r.qy, r.qz, r.qw), np.array([r.x, r.y, r.z]))
           for _, r in se[se.clip_id == clip].iterrows()}
    return intr, ext


def ftheta_project(pts_cam, intr):
    """pts_cam Nx3 in camera frame (z forward) -> (px Nx2, valid). f-theta: angle from the optical
    axis -> pixel radius via the forward polynomial `fw`."""
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    r3 = np.linalg.norm(pts_cam, axis=1) + 1e-9
    theta = np.arccos(np.clip(z / r3, -1, 1))
    fw = intr["fw"]
    rpix = fw[0] + fw[1] * theta + fw[2] * theta ** 2 + fw[3] * theta ** 3 + fw[4] * theta ** 4
    rho = np.sqrt(x * x + y * y) + 1e-9
    u = intr["cx"] + rpix * x / rho
    v = intr["cy"] + rpix * y / rho
    valid = (z > 0.3) & (theta < np.radians(75)) & (u >= 0) & (u < intr["W"]) & (v >= 0) & (v < intr["H"])
    return np.stack([u, v], 1), valid


def decode_lidar(root, clip):
    """-> [(reference_timestamp, pts Nx3), ...] per LiDAR spin (Draco-decoded)."""
    import DracoPy
    lp = glob.glob(f"{root}/lidar/{LIDAR}/*/{clip}.{LIDAR}.parquet")
    df = pd.read_parquet(lp[0])
    out = []
    for _, row in df.iterrows():
        pc = DracoPy.decode(bytes(row["draco_encoded_pointcloud"]))
        out.append((int(row["reference_timestamp"]),
                    np.asarray(pc.points, dtype=np.float32).reshape(-1, 3)))
    return out


def lidar_to_cam(pts, ext, cam_name):
    """LiDAR points -> camera frame via the per-sensor SE3 extrinsics (sensor->ego)."""
    Rl, tl = ext[LIDAR]; Rc, tc = ext[cam_name]
    p_ego = (Rl @ pts.T).T + tl
    return (Rc.T @ (p_ego - tc).T).T


class PhysicalAISource:
    """PhysicalAI-AV clips -> multi-camera frames + LiDAR f-theta-projected depth."""

    def __init__(self, root, image_hw=(384, 640), cams=PAV_CAMS, start=0, num=20, stride=5,
                 ref_cam="camera_front_wide_120fov", clip=None):
        import cv2
        self.cv2, self.root, self.cams, self.image_hw = cv2, root, cams, image_hw
        self.ref_cam, self.stride, self.start, self.num = ref_cam, stride, start, num
        mp4s = sorted(glob.glob(f"{root}/camera/{ref_cam}/{ref_cam}.chunk_*/*.mp4"))
        clips = [os.path.basename(m).split(".")[0] for m in mp4s]
        self.clips = [clip] if clip else clips[:1]                    # default: first clip

    def _mp4(self, clip, cam):
        hits = glob.glob(f"{self.root}/camera/{cam}/{cam}.chunk_*/{clip}*.mp4")
        return hits[0] if hits else None

    def _ts(self, mp4, clip):
        p = sorted(glob.glob(f"{os.path.dirname(mp4)}/{clip}*timestamps.parquet"))
        return pd.read_parquet(p[0])["timestamp"].values

    def __len__(self):
        return self.num

    def __iter__(self):
        cv2 = self.cv2
        for clip in self.clips:
            intr, ext = load_calib(self.root, clip, [c for _, c in self.cams])
            lid = decode_lidar(self.root, clip)
            lid_ts = np.array([t for t, _ in lid])
            caps = {cam: cv2.VideoCapture(self._mp4(clip, cam)) for _, cam in self.cams
                    if self._mp4(clip, cam)}
            cam_ts = {cam: self._ts(self._mp4(clip, cam), clip) for cam in caps}
            ref = cam_ts[self.ref_cam]
            frames = list(range(self.start, len(ref), self.stride))[:self.num]
            for fi in frames:
                t0 = ref[fi]
                pts = lid[int(np.argmin(np.abs(lid_ts - t0)))][1]     # nearest LiDAR spin
                out = []
                for disp, cam in self.cams:
                    if cam not in caps or cam not in intr:
                        continue
                    fj = int(np.argmin(np.abs(cam_ts[cam] - t0)))     # nearest frame in this cam
                    caps[cam].set(cv2.CAP_PROP_POS_FRAMES, fj)
                    ok, bgr = caps[cam].read()
                    if not ok:
                        continue
                    pc = lidar_to_cam(pts, ext, cam)
                    px, valid = ftheta_project(pc, intr[cam])
                    z = pc[valid][:, 2]
                    W0, H0 = intr[cam]["W"], intr[cam]["H"]
                    sx, sy = self.image_hw[1] / W0, self.image_hw[0] / H0
                    uvz = np.stack([px[valid][:, 0] * sx, px[valid][:, 1] * sy, z], 1)
                    im = Image.fromarray(bgr[:, :, ::-1]).resize((self.image_hw[1], self.image_hw[0]))
                    out.append((disp, im, uvz))
                yield f"{clip}_{fi:04d}", out
