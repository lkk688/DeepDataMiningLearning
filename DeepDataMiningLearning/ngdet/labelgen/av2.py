"""Argoverse2 (sensor) source for the label generator.

Uses the official `av2` devkit (Argoverse2 API) for the LiDAR→camera projection — importantly the
**motion-compensated** projection (AV2's LiDAR sweep and each camera frame have different timestamps;
the ego moves between them, so points must be warped through the city frame). The devkit isn't
pip-installed here, so we add its source tree to sys.path (override `av2_src` / `AV2_API_SRC` env).
7 ring cameras + the merged up/down LiDAR. Data is read-only (e.g. the student thesis AV2 tree).
"""
from __future__ import annotations
import os
import sys
import glob
import numpy as np
from PIL import Image

AV2_API_SRC = os.environ.get("AV2_API_SRC", "/data/rnd-liu/Develop/av2-api/src")
# display_name -> AV2 ring camera
AV2_RING = [("FRONT_LEFT", "ring_front_left"), ("FRONT", "ring_front_center"),
            ("FRONT_RIGHT", "ring_front_right"), ("SIDE_LEFT", "ring_side_left"),
            ("REAR_LEFT", "ring_rear_left"), ("REAR_RIGHT", "ring_rear_right")]


class AV2Source:
    def __init__(self, data_dir, image_hw=(384, 640), cams=AV2_RING, start=0, num=20, stride=1,
                 log_id=None, av2_src=AV2_API_SRC):
        if av2_src and av2_src not in sys.path:
            sys.path.insert(0, av2_src)
        from pathlib import Path
        from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
        self.dl = AV2SensorDataLoader(data_dir=Path(data_dir), labels_dir=Path(data_dir))
        self.cams, self.image_hw = cams, image_hw
        logs = self.dl.get_log_ids()
        self.log = log_id or logs[0]
        ts = self.dl.get_ordered_log_lidar_timestamps(self.log)
        self.ts = ts[start::stride][:num]

    def __len__(self):
        return len(self.ts)

    def __iter__(self):
        from av2.utils.io import read_lidar_sweep
        for lts in self.ts:
            pts = read_lidar_sweep(self.dl.get_lidar_fpath(self.log, lts), attrib_spec="xyz")  # ego frame
            out = []
            for disp, cam in self.cams:
                fpath = self.dl.get_closest_img_fpath(self.log, cam, lts)
                if fpath is None:
                    continue
                cts = int(fpath.stem)                                    # camera timestamp (ns)
                uv, pcam, valid = self.dl.project_ego_to_img_motion_compensated(
                    pts, cam, cts, lts, self.log)
                im = Image.open(fpath).convert("RGB")
                W0, H0 = im.size
                sx, sy = self.image_hw[1] / W0, self.image_hw[0] / H0
                uvz = np.stack([uv[valid, 0] * sx, uv[valid, 1] * sy, pcam[valid, 2]], 1)
                out.append((disp, im.resize((self.image_hw[1], self.image_hw[0])), uvz))
            yield f"{self.log[:8]}_{lts}", out
