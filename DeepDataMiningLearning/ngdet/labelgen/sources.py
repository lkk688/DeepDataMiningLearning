"""Data sources for the label generator.

`ImageFolderSource` — any folder of images (semantic + RELATIVE depth; no LiDAR).
`NuScenesSource`   — nuScenes keyframes: 6 surround cameras + LiDAR projected into each (metric depth).
                     Uses nuscenes-devkit directly (self-contained); implements the LiDAR->camera
                     projection ngdet's box-only NuScenesDataset does not.

Each source yields, per sample, a list of (cam_name, PIL.Image, lidar_uv_z or None) — exactly what
`GroundedLabeler.label(pil, lidar_uv_z)` consumes.
"""
from __future__ import annotations
import glob
import os
import numpy as np
from PIL import Image

NUSC_CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
             "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]        # 2x3 display order


def project_pinhole(pts_cam, K, orig_wh, out_hw):
    """pts_cam [P,3] in a Z-FORWARD pinhole camera frame, K [3,3] -> [M,3]=(u,v,z) at out_hw.
    The one shared primitive: every dataset's Source just has to put its LiDAR into a z-forward
    camera frame and hand it here. orig_wh = (W,H) the K is calibrated for."""
    z = pts_cam[:, 2]
    ok = z > 0.5
    p = pts_cam[ok]
    u = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]
    v = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    sx, sy = out_hw[1] / orig_wh[0], out_hw[0] / orig_wh[1]        # scale K's image -> resized
    return np.stack([u * sx, v * sy, z[ok]], 1)


def _homog(p):
    return np.concatenate([p, np.ones((len(p), 1), p.dtype)], 1)


def project_ftheta(pts_cam, cx, cy, fw_poly, orig_wh, out_hw):
    """f-theta / polynomial fisheye projection (NVIDIA PhysicalAI cameras). pts_cam [P,3] in a
    z-FORWARD camera frame; `fw_poly` = [c0..c4] mapping the incidence angle theta (rad) to a pixel
    radius. -> [M,3]=(u,v,z). Use this instead of project_pinhole for fisheye/f-theta intrinsics."""
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    ok = z > 0.5
    x, y, z = x[ok], y[ok], z[ok]
    theta = np.arctan2(np.sqrt(x * x + y * y), z)                 # angle from optical axis
    r = sum(fw_poly[i] * theta ** i for i in range(len(fw_poly)))  # angle -> pixel radius
    phi = np.arctan2(y, x)
    u = cx + r * np.cos(phi); v = cy + r * np.sin(phi)
    sx, sy = out_hw[1] / orig_wh[0], out_hw[0] / orig_wh[1]
    return np.stack([u * sx, v * sy, z], 1)


class ImageFolderSource:
    def __init__(self, folder, image_hw=None, exts=("jpg", "png", "jpeg")):
        self.files = sorted(f for e in exts for f in glob.glob(os.path.join(folder, f"*.{e}")))
        self.image_hw = image_hw

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for f in self.files:
            im = Image.open(f).convert("RGB")
            if self.image_hw:
                im = im.resize((self.image_hw[1], self.image_hw[0]))
            yield os.path.basename(f), [(os.path.basename(f), im, None)]


class NuScenesSource:
    """nuScenes keyframes -> 6 cameras + LiDAR-projected depth per camera."""

    def __init__(self, dataroot, version="v1.0-trainval", image_hw=(256, 704),
                 cams=NUSC_CAMS, start=0, num=None, stride=1):
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.cams, self.image_hw = cams, image_hw
        samples = [s for s in self.nusc.sample]
        samples = samples[start::stride]
        self.samples = samples[:num] if num else samples

    def __len__(self):
        return len(self.samples)

    def _project_lidar(self, sample, cam):
        """LiDAR sweep -> (u,v,z) in the camera image (resized). Standard nuScenes transform chain:
        lidar -> ego(t_lidar) -> global -> ego(t_cam) -> cam -> image."""
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.geometry_utils import view_points
        from pyquaternion import Quaternion
        lsd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lsd["filename"]))
        cs_l = self.nusc.get("calibrated_sensor", lsd["calibrated_sensor_token"])
        ep_l = self.nusc.get("ego_pose", lsd["ego_pose_token"])
        pc.rotate(Quaternion(cs_l["rotation"]).rotation_matrix); pc.translate(np.array(cs_l["translation"]))
        pc.rotate(Quaternion(ep_l["rotation"]).rotation_matrix); pc.translate(np.array(ep_l["translation"]))
        csd = self.nusc.get("sample_data", sample["data"][cam])
        ep_c = self.nusc.get("ego_pose", csd["ego_pose_token"])
        cs_c = self.nusc.get("calibrated_sensor", csd["calibrated_sensor_token"])
        pc.translate(-np.array(ep_c["translation"])); pc.rotate(Quaternion(ep_c["rotation"]).rotation_matrix.T)
        pc.translate(-np.array(cs_c["translation"])); pc.rotate(Quaternion(cs_c["rotation"]).rotation_matrix.T)
        z = pc.points[2]
        K = np.array(cs_c["camera_intrinsic"])
        uv = view_points(pc.points[:3], K, normalize=True)[:2]        # at ORIGINAL 1600x900
        H0, W0 = csd["height"], csd["width"]
        sy, sx = self.image_hw[0] / H0, self.image_hw[1] / W0         # scale to resized image
        m = z > 0.5
        return np.stack([uv[0][m] * sx, uv[1][m] * sy, z[m]], 1)

    def __iter__(self):
        for s in self.samples:
            token = s["token"]
            out = []
            for cam in self.cams:
                csd = self.nusc.get("sample_data", s["data"][cam])
                im = Image.open(os.path.join(self.nusc.dataroot, csd["filename"])).convert("RGB")
                im = im.resize((self.image_hw[1], self.image_hw[0]))
                out.append((cam, im, self._project_lidar(s, cam)))
            yield token, out


class WaymoSource:
    """Waymo (pre-extracted npz at .../waymo_v1_extracted): 5 cameras + vehicle-frame LiDAR.
    Per frame f_XXXX.npz has `lidar (N,5)` in VEHICLE frame; f_XXXX_cam_C_calib.npz has pinhole
    `intrinsic` + `cam2vehicle`. Waymo's camera frame is x-FORWARD/z-up, so we swap to a z-forward
    pinhole (x,y,z)->(-y,-z,x) before project_pinhole."""
    WAYMO_CAMS = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}

    def __init__(self, root, image_hw=(256, 704), start=0, num=None, stride=1, segments=None):
        segs = sorted(glob.glob(os.path.join(root, "*/")))
        if segments:
            segs = [s for s in segs if os.path.basename(s.rstrip("/")) in segments]
        self.samples = []
        for s in segs:
            frames = sorted(glob.glob(os.path.join(s, "f_[0-9]*.npz")))
            frames = [f for f in frames if "_cam_" not in os.path.basename(f)]
            for f in frames:
                self.samples.append((s, os.path.basename(f)[2:6]))
        self.samples = self.samples[start::stride]
        self.samples = self.samples[:num] if num else self.samples
        self.image_hw = image_hw

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        for seg, fid in self.samples:
            fr = np.load(os.path.join(seg, f"f_{fid}.npz"))
            lidar = fr["lidar"][:, :3].astype(np.float32)                 # vehicle frame
            out = []
            for cid in [2, 1, 3, 4, 5]:                                  # FL,F,FR,SL,SR display order
                cp = os.path.join(seg, f"f_{fid}_cam_{cid}_calib.npz")
                ip = os.path.join(seg, f"f_{fid}_cam_{cid}.jpg")
                if not (os.path.exists(cp) and os.path.exists(ip)):
                    continue
                c = np.load(cp); K = c["intrinsic"][:3, :3]
                veh2cam = np.linalg.inv(c["cam2vehicle"])
                pc = (veh2cam @ _homog(lidar).T).T[:, :3]                # Waymo cam frame (x-fwd)
                pin = np.stack([-pc[:, 1], -pc[:, 2], pc[:, 0]], 1)      # -> z-forward pinhole
                uvz = project_pinhole(pin, K, (int(c["width"]), int(c["height"])), self.image_hw)
                im = Image.open(ip).convert("RGB").resize((self.image_hw[1], self.image_hw[0]))
                out.append((self.WAYMO_CAMS[cid], im, uvz))
            yield f"{os.path.basename(seg.rstrip('/'))}_{fid}", out


class KittiSource:
    """KITTI (root/{training}/{image_2,velodyne,calib}): single front camera + Velodyne.
    calib: lidar->rect = R0 @ Tr_velo_to_cam; rect->image = P2. TEMPLATE — no local data here to
    validate against, but the calib path is canonical."""
    def __init__(self, root, split="training", image_hw=(375, 1242), start=0, num=None, stride=1):
        self.root, self.split, self.image_hw = root, split, image_hw
        ids = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(root, split, "image_2", "*.png")))
        ids = ids[start::stride]
        self.ids = ids[:num] if num else ids

    def _calib(self, fid):
        rows = {}
        for line in open(os.path.join(self.root, self.split, "calib", fid + ".txt")):
            if ":" in line:
                k, v = line.split(":", 1); rows[k.strip()] = np.array(v.split(), np.float32)
        return rows["P2"].reshape(3, 4), rows["R0_rect"].reshape(3, 3), rows["Tr_velo_to_cam"].reshape(3, 4)

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for fid in self.ids:
            P2, R0, V2C = self._calib(fid)
            pts = np.fromfile(os.path.join(self.root, self.split, "velodyne", fid + ".bin"),
                              np.float32).reshape(-1, 4)[:, :3]
            R0h = np.eye(4, dtype=np.float32); R0h[:3, :3] = R0
            V2Ch = np.eye(4, dtype=np.float32); V2Ch[:3, :] = V2C
            rect = (R0h @ V2Ch @ _homog(pts).T).T[:, :3]                 # rect-cam frame (z-fwd)
            im = Image.open(os.path.join(self.root, self.split, "image_2", fid + ".png")).convert("RGB")
            W0, H0 = im.size
            # rect->image uses P2 (fx,fy,cx,cy in P2[:, :3]); reuse project_pinhole with K=P2[:, :3]
            uvz = project_pinhole(rect, P2[:, :3], (W0, H0), self.image_hw)
            out = [("FRONT", im.resize((self.image_hw[1], self.image_hw[0])), uvz)]
            yield fid, out
