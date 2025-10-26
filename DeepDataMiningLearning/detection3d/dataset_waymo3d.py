import os
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

KEY_SEG = "key.segment_context_name"
KEY_TS  = "key.frame_timestamp_micros"
LASER_NAME = "key.laser_name"

RI_VALS1 = "[LiDARComponent].range_image_return1.values"
RI_SHAPE1 = "[LiDARComponent].range_image_return1.shape"

BOX_X = "[LiDARBoxComponent].box.center.x"
BOX_Y = "[LiDARBoxComponent].box.center.y"
BOX_Z = "[LiDARBoxComponent].box.center.z"
BOX_L = "[LiDARBoxComponent].box.size.x"
BOX_W = "[LiDARBoxComponent].box.size.y"
BOX_H = "[LiDARBoxComponent].box.size.z"
BOX_HEADING = "[LiDARBoxComponent].box.heading"
BOX_TYPE = "[LiDARBoxComponent].type"


def _read_list_column(col):
    """Safely convert PyArrow or numpy list to numpy array."""
    if hasattr(col, "values"):
        return np.array([x.as_py() for x in col.values])
    if isinstance(col, (list, np.ndarray)):
        out = []
        for x in col:
            if hasattr(x, "as_py"):
                out.append(x.as_py())
            else:
                out.append(x)
        return np.array(out)
    if hasattr(col, "as_py"):
        return np.array(col.as_py())
    raise TypeError(f"Unsupported column type: {type(col)}")


def _decode_range_image(row):
    vals = _read_list_column(row[RI_VALS1])
    shape = _read_list_column(row[RI_SHAPE1])
    if len(shape) < 3:
        raise ValueError(f"Invalid range image shape: {shape}")
    H, W, C = map(int, shape[:3])
    arr = np.array(vals, dtype=np.float32).reshape(H, W, C)
    return arr


def _spherical_to_cartesian(range_img, inclinations, extrinsic):
    """
    Convert Waymo LiDAR range image to Cartesian points (vehicle frame).

    Fixes:
      • Flip vertical sign (Waymo beam inclination is downward positive).
      • Clamp invalid ranges.
      • Uses T_vehicle←lidar extrinsic (4×4).
    """
    H, W = range_img.shape
    # Uniform azimuth sweep
    azimuth = np.linspace(-np.pi, np.pi, W, endpoint=False)

    # Sanitize ranges
    r = np.nan_to_num(range_img, nan=0.0, posinf=0.0, neginf=0.0)
    r = np.clip(r, 0.0, 300.0)

    # Broadcast angles
    incl = inclinations.reshape(H, 1)

    # --- Spherical to Cartesian (LiDAR sensor frame) ---
    # Waymo inclinations: downward beams are NEGATIVE, so use -sin to make ground z≈0.
    x = r * np.cos(incl) * np.cos(azimuth)   # forward
    y = r * np.cos(incl) * np.sin(azimuth)   # left
    z = -r * np.sin(incl)                    # up  ← flipped sign

    pts = np.stack([x, y, z, np.ones_like(z)], axis=-1).reshape(-1, 4)

    # --- Transform to vehicle frame ---
    pts_vehicle = pts @ extrinsic.T
    xyz = pts_vehicle[:, :3]

    # Remove NaN/Inf
    xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
    return xyz

def spherical_to_cartesian_general(
    range_img: np.ndarray,
    inclinations: np.ndarray,
    extrinsic: np.ndarray,
    flip_rows: bool = False,
    flip_cols: bool = False,
    azimuth_offset: float = 0.0,
) -> np.ndarray:
    """
    Convert range image to 3D points with configurable flips and azimuth offset.
    """
    H, W = range_img.shape

    # sanitize ranges
    r = np.nan_to_num(range_img, nan=0.0, posinf=0.0, neginf=0.0)
    r = np.clip(r, 0.0, 300.0)

    inc = inclinations.astype(np.float32)
    if flip_rows:
        inc = inc[::-1]
    inc = inc.reshape(H, 1)

    # az = np.linspace(-np.pi, np.pi, W, endpoint=False).astype(np.float32)
    # if flip_cols:
    #     az = az[::-1]

    # az = az + np.pi / 2.0 #new add
    # az = az + float(azimuth_offset)
    # ------------------------------------------------------------
    # Build azimuth grid (horizontal scanning angles)
    # ------------------------------------------------------------
    W = range_img.shape[1]

    # Column index from 0 .. W-1
    col_idx = np.arange(W, dtype=np.float32)

    # Convert to radians: [-π, π)
    az = (2 * np.pi * col_idx / W) - np.pi

    # 测试：完全移除方位角调整，仅依赖外参矩阵
    # az = az - np.pi / 2.0  # 暂时注释掉方位角调整

    # Optional: reverse column order if scan direction is mirrored
    if flip_cols:
        az = az[::-1]

    # Waymo downward beams have negative inclinations → use -sin to make +Z up
    x = r * np.cos(inc) * np.cos(az)   # X: forward
    y = r * np.cos(inc) * np.sin(az)   # Y: left  
    z = -r * np.sin(inc)               # Z: up

    pts = np.stack([x, y, z, np.ones_like(z)], axis=-1).reshape(-1, 4)
    xyz = (pts @ extrinsic.T)[:, :3]
    return np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)

def _get_calibration_fields(pf: pq.ParquetFile):
    """
    Detect calibration field names in lidar_calibration parquet.
    Works with both old '[LiDARCalibrationComponent].extrinsic.transform'
    and the newer shorthand 'item' column.
    """
    names = set(pf.schema.names)

    # --- find extrinsic ---
    extr_field = None
    for cand in names:
        if "extrinsic" in cand and "transform" in cand:
            extr_field = cand
            break
    # Waymo v2.1 sometimes flattens it to just 'item'
    if extr_field is None and "item" in names:
        extr_field = "item"

    # --- find beam inclinations ---
    beam_field = None
    beam_min, beam_max = None, None
    for n in names:
        if "beam_inclination.values" in n:
            beam_field = n
        elif "beam_inclination.min" in n:
            beam_min = n
        elif "beam_inclination.max" in n:
            beam_max = n

    if not extr_field:
        raise KeyError(f"Cannot find extrinsic field in {list(names)}")

    out = {
        "extr_mode": "matrix",
        "extr_field": extr_field,
        "beam_mode": "values" if beam_field else "minmax",
        "beam_field": beam_field,
        "beam_min": beam_min,
        "beam_max": beam_max,
    }
    return out

def _points_in_obb_xy_count(
    xyz: np.ndarray,
    boxes: np.ndarray,
    yaw_sign: int = -1,
    k_sample: int = 150_000
) -> int:
    """
    Fast approximate metric: how many points fall inside any box footprint in XY plane.

    We ignore full 3D IoU and just test 2D overlap for speed.
    """
    import numpy as np

    if xyz.shape[0] > k_sample:
        idx = np.random.choice(xyz.shape[0], k_sample, replace=False)
        P = xyz[idx]
    else:
        P = xyz

    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    count = 0

    for b in boxes:
        cx, cy, cz, dx, dy, dz, yaw = b.astype(np.float32)
        yaw = yaw_sign * yaw  # allow ± convention
        c, s = np.cos(yaw), np.sin(yaw)

        # translate to box frame
        x = X - cx
        y = Y - cy

        # rotate to align with box axes
        xr =  c * x + s * y
        yr = -s * x + c * y

        inside_xy = (np.abs(xr) <= dx * 0.5) & (np.abs(yr) <= dy * 0.5)
        inside_z  = (Z >= cz - dz * 0.6) & (Z <= cz + dz * 0.6)
        count += int(np.count_nonzero(inside_xy & inside_z))
    return count

def auto_calibrate_range_to_vehicle(
    range_img: np.ndarray,
    inclinations: np.ndarray,
    extrinsic: np.ndarray,
    boxes3d: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Automatically determine the correct row/column flip and azimuth offset
    to align LiDAR point clouds with Waymo 3D boxes.

    Waymo’s LiDAR range images vary slightly by sensor type (TOP, SIDE, etc.),
    so this helper tries multiple combinations of:
        • flip_rows   : reverse vertical beam order
        • flip_cols   : reverse horizontal azimuth sweep
        • azimuth_offset : shift where column 0 starts (in radians)
        • yaw_sign    : +1 or -1 (for box orientation convention)

    The best configuration is chosen by maximizing
    the number of LiDAR points that fall inside any 3D box footprint.

    Args:
        range_img   : (H,W) array of range distances in meters
        inclinations: (H,) vertical beam angles in radians
        extrinsic   : (4,4) LiDAR→Vehicle homogeneous matrix
        boxes3d     : (M,7) 3D boxes in vehicle coordinates

    Returns:
        xyz_best : (N,3) point cloud (vehicle frame) using best configuration
        cfg_best : dict with keys {flip_rows, flip_cols, azimuth_offset, yaw_sign}
    """
    import numpy as np

    # ------------------------------------------------------------
    # 1️⃣ Parameter search grid
    # ------------------------------------------------------------
    flip_options = [False, True]
    # include ±90° (π/2) offsets — Waymo often needs +π/2
    offset_options = [0.0, np.pi/2, -np.pi/2, np.pi, 3*np.pi/2]
    yaw_sign_options = [+1, -1]

    best_score = -1
    best_xyz = None
    best_cfg = None

    # ------------------------------------------------------------
    # 2️⃣ Iterate through all combinations
    # ------------------------------------------------------------
    for fr in flip_options:
        for fc in flip_options:
            for off in offset_options:
                # Generate candidate cloud
                xyz = spherical_to_cartesian_general(
                    range_img, inclinations, extrinsic,
                    flip_rows=fr, flip_cols=fc, azimuth_offset=off
                )
                # Score this cloud
                for ys in yaw_sign_options:
                    score = _points_in_obb_xy_count(
                        xyz, boxes3d, yaw_sign=ys, k_sample=150_000
                    )
                    if score > best_score:
                        best_score = score
                        best_xyz = xyz
                        best_cfg = {
                            "flip_rows": fr,
                            "flip_cols": fc,
                            "azimuth_offset": float(off),
                            "yaw_sign": ys,
                            "score": int(score),
                        }

    # ------------------------------------------------------------
    # 3️⃣ Return the best result
    # ------------------------------------------------------------
    print(f"[AUTO] Best alignment: score={best_cfg['score']}  "
          f"flip_rows={best_cfg['flip_rows']}  "
          f"flip_cols={best_cfg['flip_cols']}  "
          f"offset={best_cfg['azimuth_offset']:.2f} rad  "
          f"yaw_sign={best_cfg['yaw_sign']}")
    return best_xyz, best_cfg

def rotate_xyz_90(xyz: np.ndarray, direction: str) -> np.ndarray:
    """
    Apply an exact 90° rotation about +Z to xyz points.
    direction: "ccw" (counter-clockwise) or "cw" (clockwise)
    """
    if direction == "ccw":
        # Rz(+90°): x' = -y, y' = x
        R = np.array([[0.0, -1.0, 0.0],
                      [1.0,  0.0, 0.0],
                      [0.0,  0.0, 1.0]], dtype=np.float32)
    elif direction == "cw":
        # Rz(-90°): x' = y, y' = -x
        R = np.array([[0.0,  1.0, 0.0],
                      [-1.0, 0.0, 0.0],
                      [0.0,  0.0, 1.0]], dtype=np.float32)
    else:
        raise ValueError("direction must be 'ccw' or 'cw'")
    return xyz @ R.T

def _read_list_scalar(cell):
    """
    Read a SINGLE parquet cell that stores a pyarrow List<double> or similar.

    Waymo calibration 'item' or '...extrinsic.transform' columns contain
    a list of 16 doubles representing the 4×4 extrinsic matrix.

    This helper returns a flat Python list[float] or None.
    Works for:
        - pyarrow.lib.ListScalar / ListValue
        - Python list / numpy.ndarray
        - None / NaN
    """
    import numpy as np

    try:
        # Case 1: Arrow ListScalar/ListValue
        if hasattr(cell, "as_py"):
            val = cell.as_py()
            if val is None:
                return None
            return list(val)
        # Case 2: already Python list or ndarray
        if isinstance(cell, (list, np.ndarray)):
            return list(cell)
    except Exception as e:
        print(f"[WARN] _read_list_scalar failed for type {type(cell)}: {e}")
    return None

def rotate_z(xyz: np.ndarray, deg: float):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return xyz @ R.T

# 选择若干近处的大车，估 XY 主方向 vs box heading
def _median_yaw_error(xyz, boxes3d):
    import numpy as np
    if boxes3d.shape[0] == 0: return None
    arr = boxes3d.numpy() if hasattr(boxes3d, "numpy") else boxes3d
    L, W = arr[:,3], arr[:,4]
    mask = (L > 3.0) & (W > 1.4)
    arr = arr[mask] if mask.any() else arr
    if arr.shape[0] == 0: return None
    # 按距离排序取前几辆
    d = np.linalg.norm(arr[:, :2], axis=1); take = np.argsort(d)[:8]
    arr = arr[take]

    ds = []
    for b in arr:
        cx,cy,cz,dx,dy,dz,yaw = b
        sel = (np.abs(xyz[:,0]-cx) < dx*0.6) & (np.abs(xyz[:,1]-cy) < dy*0.6)
        P = xyz[sel]
        if P.shape[0] < 200: continue
        # PCA 主方向
        Q = P[:,:2] - b[:2]
        C = np.cov(Q.T); evals, evecs = np.linalg.eigh(C)
        v = evecs[:, np.argmax(evals)]
        ang = np.arctan2(v[1], v[0])
        d = ang - yaw      # 注意：绘制时才用 -yaw，这里与标注同号比较
        d = (d + np.pi) % (2*np.pi) - np.pi
        if d >  np.pi/2: d -= np.pi
        if d < -np.pi/2: d += np.pi
        ds.append(d)
    return None if len(ds)==0 else float(np.median(ds))


import os
import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset


class Waymo3DDataset(Dataset):
    """
    Waymo v2.x → 3D LiDAR dataset (safe baseline).
    Step 1: Reconstruct point cloud in VEHICLE frame (only LiDAR→Vehicle extrinsic).
    Step 2 (optional): If return_world=True, also transform to WORLD using vehicle_pose.
    """

    def __init__(self, root_dir, split="training", max_frames=None, return_world=False):
        self.root = root_dir
        self.split = split
        self.return_world = return_world

        self.lidar_dir  = os.path.join(root_dir, split, "lidar")
        self.calib_dir  = os.path.join(root_dir, split, "lidar_calibration")
        self.box_dir    = os.path.join(root_dir, split, "lidar_box")
        self.vpose_dir  = os.path.join(root_dir, split, "vehicle_pose")  # 仅当 return_world=True 时使用

        for p in [self.lidar_dir, self.calib_dir, self.box_dir]:
            if not os.path.isdir(p):
                raise FileNotFoundError(p)
        if self.return_world and not os.path.isdir(self.vpose_dir):
            raise FileNotFoundError(self.vpose_dir)

        files = [f for f in os.listdir(self.lidar_dir) if f.endswith(".parquet")]
        valid = [f for f in files if os.path.exists(os.path.join(self.box_dir, f))]
        if not valid:
            raise RuntimeError("No matching lidar/lidar_box shards found.")

        self.frame_index = []
        total = 0
        for fname in valid:
            pf = pq.ParquetFile(os.path.join(self.lidar_dir, fname))
            df = pf.read_row_group(0, columns=["key.segment_context_name","key.frame_timestamp_micros"]).to_pandas()
            for seg, ts in zip(df["key.segment_context_name"], df["key.frame_timestamp_micros"]):
                self.frame_index.append((fname, int(ts), seg))
                total += 1
                if max_frames and total >= max_frames:
                    break
            if max_frames and total >= max_frames:
                break

        print(f"✅ Waymo3DDataset frames: {len(self.frame_index)} | return_world={self.return_world}")


    @staticmethod
    def _decode_pixel_pose(row):
        kv = "[LiDARPoseComponent].range_image_return1.values"
        ks = "[LiDARPoseComponent].range_image_return1.shape"
        vals = row[kv].as_py() if hasattr(row[kv], "as_py") else row[kv]
        shp  = row[ks].as_py() if hasattr(row[ks], "as_py") else row[ks]
        arr  = np.array(vals, np.float32).reshape(shp, order="C")  # [H,W,6]
        return arr

    @staticmethod
    def _decode_range_image(row, return_id=1):
        kv = f"[LiDARComponent].range_image_return{return_id}.values"
        ks = f"[LiDARComponent].range_image_return{return_id}.shape"
        vals = row[kv].as_py() if hasattr(row[kv], "as_py") else row[kv]
        shp  = row[ks].as_py() if hasattr(row[ks], "as_py") else row[ks]
        arr  = np.array(vals, dtype=np.float32).reshape(shp, order="C")
        return arr  # [H,W,C], C>=2: range,intensity,...

    def __getitem__(self, idx):
        fname, ts, seg = self.frame_index[idx]

        # ---------- 1) read one LiDAR return (per-sensor row) ----------
        pf = pq.ParquetFile(os.path.join(self.lidar_dir, fname))
        df = pf.read_row_group(0).to_pandas()
        row = df[(df["key.segment_context_name"] == seg) &
                 (df["key.frame_timestamp_micros"] == ts)].iloc[0]
        laser_id = int(row["key.laser_name"])

        ri = self._decode_range_image(row, return_id=1)
        rng   = np.nan_to_num(ri[..., 0], nan=0.0); rng[rng < 0] = 0.0
        inten = ri[..., 1] if ri.shape[-1] > 1 else np.zeros_like(rng)
        H, W  = rng.shape

        # ---------- 2) calibration (beam inclinations + LiDAR→Vehicle extrinsic) ----------
        pf_cal = pq.ParquetFile(os.path.join(self.calib_dir, fname))
        df_cal = pf_cal.read_row_group(0).to_pandas()
        crow = df_cal[(df_cal["key.segment_context_name"] == seg) &
                      (df_cal["key.laser_name"] == laser_id)].iloc[0]

        inc_min = float(crow["[LiDARCalibrationComponent].beam_inclination.min"])
        inc_max = float(crow["[LiDARCalibrationComponent].beam_inclination.max"])
        inclinations = np.linspace(inc_min, inc_max, H, dtype=np.float32)
        if np.max(np.abs(inclinations)) > np.pi:
            inclinations = np.deg2rad(inclinations)

        # extrinsic 4x4 (row-major)
        extr_col = max([c for c in crow.index if ("extrinsic" in c or str(c).endswith("item"))], key=len)
        extr_vals = crow[extr_col]
        extr_vals = extr_vals.as_py() if hasattr(extr_vals, "as_py") else extr_vals
        T_vl = np.array(extr_vals, dtype=np.float32).reshape(4, 4, order="C")  # LiDAR->Vehicle

        # ---------- 3) spherical → LiDAR Cartesian (pay attention to signs/order) ----------
        incl = inclinations[::-1].reshape(H, 1)                      # vertical flip (bottom→top)
        az   = np.linspace(np.pi, -np.pi, W, endpoint=False, dtype=np.float32)  # azimuth decreases right→left

        cos_i, sin_i = np.cos(incl), np.sin(incl)
        cos_a, sin_a = np.cos(az),   np.sin(az)

        Xl = rng * cos_i * cos_a
        Yl = -rng * cos_i * sin_a   # ← Waymo 的 Y 方向需要负号
        Zl = rng * sin_i

        pts_l = np.stack([Xl, Yl, Zl, np.ones_like(Zl)], axis=-1).reshape(-1, 4)

        # ---------- 4) LiDAR→Vehicle (一次、且仅一次) ----------
        pts_v = (pts_l @ T_vl.T)[:, :3]   # Vehicle frame point cloud (baseline对齐坐标)
        xyz_vehicle = np.nan_to_num(pts_v, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- 5) Vehicle→World (可选；若需要就对点和框一起变换) ----------
        if self.return_world:
            pf_vp = pq.ParquetFile(os.path.join(self.vpose_dir, fname))
            df_vp = pf_vp.read_row_group(0).to_pandas()
            vrow = df_vp[(df_vp["key.segment_context_name"] == seg) &
                         (df_vp["key.frame_timestamp_micros"] == ts)].iloc[0]
            vp_vals = vrow["[VehiclePoseComponent].world_from_vehicle.transform"]
            vp_vals = vp_vals.as_py() if hasattr(vp_vals, "as_py") else vp_vals
            T_wv = np.array(vp_vals, dtype=np.float32).reshape(4, 4, order="C")  # row-major

            pts_vh = np.concatenate([xyz_vehicle, np.ones((xyz_vehicle.shape[0],1), np.float32)], axis=1)
            xyz_world = (pts_vh @ T_wv.T)[:, :3]
            XYZ = xyz_world
        else:
            T_wv = None
            XYZ = xyz_vehicle

        # ---------- 6) intensity normalize ----------
        inten = inten.reshape(-1, 1).astype(np.float32)
        inten = inten / (inten.max() + 1e-6)
        lidar = torch.tensor(np.concatenate([XYZ, inten], axis=1), dtype=torch.float32)

        # ---------- 7) 3D boxes (Vehicle frame; do NOT transform here) ----------
        pf_box = pq.ParquetFile(os.path.join(self.box_dir, fname))
        df_box = pf_box.read_row_group(0).to_pandas()
        rows = df_box[(df_box["key.segment_context_name"] == seg) &
                      (df_box["key.frame_timestamp_micros"] == ts)]
        if len(rows) == 0:
            boxes_v = torch.zeros((0, 7), dtype=torch.float32)
            labels  = torch.zeros((0,), dtype=torch.int64)
        else:
            box_fields = [
                "[LiDARBoxComponent].box.center.x",
                "[LiDARBoxComponent].box.center.y",
                "[LiDARBoxComponent].box.center.z",
                "[LiDARBoxComponent].box.size.x",
                "[LiDARBoxComponent].box.size.y",
                "[LiDARBoxComponent].box.size.z",
                "[LiDARBoxComponent].box.heading",
            ]
            label_field = "[LiDARBoxComponent].type"
            boxes_v = torch.tensor(rows[box_fields].to_numpy(), dtype=torch.float32)
            labels  = torch.tensor(rows[label_field].to_numpy(), dtype=torch.int64)

        # ---------- 8) assemble ----------
        target = {
            "boxes_3d": boxes_v,             # Vehicle frame
            "labels": labels,
            "segment": seg,
            "timestamp": ts,
            "laser_id": laser_id,
        }
        if self.return_world:
            target["world_from_vehicle"] = torch.tensor(T_wv, dtype=torch.float32)

        # ---------- 9) quick debug (first frame) ----------
        if idx == 0:
            rng_stats = (XYZ[:,0].min(), XYZ[:,0].max(), XYZ[:,1].min(), XYZ[:,1].max(), XYZ[:,2].min(), XYZ[:,2].max())
            print(f"[DEBUG] points={lidar.shape[0]}  range XYZ: X[{rng_stats[0]:.1f},{rng_stats[1]:.1f}] "
                  f"Y[{rng_stats[2]:.1f},{rng_stats[3]:.1f}] Z[{rng_stats[4]:.1f},{rng_stats[5]:.1f}]")
            if self.return_world:
                print("[DEBUG] T_wv trans:", T_wv[:3,3])

        return lidar, target

def test_parquet():
    calib_path = "/mnt/e/Shared/Dataset/waymodata/training/lidar_calibration/9758342966297863572_875_230_895_230.parquet"
    pf = pq.ParquetFile(calib_path)
    print(pf.schema)

import open3d as o3d
import numpy as np
def visualize_open3d(
    lidar: "torch.Tensor|np.ndarray",
    boxes3d: "torch.Tensor|np.ndarray|None" = None,
    labels: "torch.Tensor|np.ndarray|None" = None,
    point_size: float = 1.0,
    color_by_intensity: bool = True,
    invert_yaw_for_open3d: bool = True,
    axis_size: float = 5.0,
    save_ply_path: str | None = None,
):
    """
    Interactive Open3D visualization for LiDAR point clouds + 3D bounding boxes.

    Coordinate Convention (Waymo Vehicle Frame):
        +X : forward
        +Y : left
        +Z : up

    Parameters
    ----------
    lidar : torch.Tensor | np.ndarray
        LiDAR points, shape [N,4] or [N,3], (x, y, z, intensity)
    boxes3d : torch.Tensor | np.ndarray | None
        3D boxes, shape [M,7] (x, y, z, dx, dy, dz, yaw)
    labels : torch.Tensor | np.ndarray | None
        Class labels for boxes (optional)
    point_size : float
        Size of rendered points in Open3D viewer
    color_by_intensity : bool
        If True, map grayscale colors by intensity value
    invert_yaw_for_open3d : bool
        Some datasets have opposite yaw convention.
        Set False if boxes appear globally rotated.
    axis_size : float
        Length of coordinate axes drawn at origin
    save_ply_path : str | None
        If provided, saves the point cloud and boxes as PLY files to this path.
        Example: "frame_0000.ply"
    """

    import numpy as np
    import open3d as o3d
    import os

    # --- Convert to NumPy arrays ---
    if hasattr(lidar, "detach"):
        lidar = lidar.detach().cpu().numpy()
    if boxes3d is not None and hasattr(boxes3d, "detach"):
        boxes3d = boxes3d.detach().cpu().numpy()
    if labels is not None and hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    # --- Create Open3D point cloud ---
    pts = lidar[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    if color_by_intensity and lidar.shape[1] >= 4:
        inten = lidar[:, 3].astype(np.float32)

        # Use np.ptp for NumPy 2.0 compatibility (old .ptp() removed)
        i_min = float(np.min(inten))
        i_max = float(np.max(inten))
        i_ptp = np.ptp(inten) if hasattr(np, "ptp") else (i_max - i_min)

        # Normalize intensity to [0, 1]
        inten = (inten - i_min) / (i_ptp + 1e-12)
        colors = np.stack([inten, inten, inten], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    geoms = [pcd]

    # --- Draw 3D bounding boxes ---
    if boxes3d is not None and len(boxes3d) > 0:
        def color_for(k):
            """Color palette for distinct class labels."""
            tab10 = [
                (0.121, 0.466, 0.705), (1.000, 0.498, 0.054), (0.172, 0.627, 0.172),
                (0.839, 0.153, 0.157), (0.580, 0.404, 0.741), (0.549, 0.337, 0.294),
                (0.890, 0.467, 0.761), (0.498, 0.498, 0.498), (0.737, 0.741, 0.133),
                (0.090, 0.745, 0.811),
            ]
            return tab10[int(k) % len(tab10)]

        for i, b in enumerate(boxes3d):
            x, y, z, dx, dy, dz, yaw = map(float, b[:7])
            yaw_draw = -yaw if invert_yaw_for_open3d else yaw

            # Build rotation matrix for yaw
            R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, yaw_draw))
            obb = o3d.geometry.OrientedBoundingBox(center=[x, y, z], R=R, extent=[dx, dy, dz])

            # Assign color
            if labels is not None and i < len(labels):
                obb.color = color_for(labels[i])
            else:
                obb.color = (1.0, 0.3, 0.0)
            geoms.append(obb)

    # --- Add coordinate axes (X=red, Y=green, Z=blue) ---
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
    geoms.append(axis)

    # --- Save to .PLY file (optional) ---
    if save_ply_path is not None:
        base, _ = os.path.splitext(save_ply_path)
        ply_pcd_path = base + "_points.ply"
        o3d.io.write_point_cloud(ply_pcd_path, pcd)
        print(f"[SAVE] Point cloud saved to: {ply_pcd_path}")

        if boxes3d is not None and len(boxes3d) > 0:
            # Collect all bounding box edges as a single LineSet
            all_points, all_lines, all_colors = [], [], []
            point_offset = 0
            for obb in geoms:
                if isinstance(obb, o3d.geometry.OrientedBoundingBox):
                    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
                    pts = np.asarray(ls.points)
                    lines = np.asarray(ls.lines) + point_offset
                    cols = np.asarray(ls.colors)
                    all_points.append(pts)
                    all_lines.append(lines)
                    all_colors.append(cols)
                    point_offset += len(pts)

            if all_points:
                box_lineset = o3d.geometry.LineSet()
                box_lineset.points = o3d.utility.Vector3dVector(np.vstack(all_points))
                box_lineset.lines  = o3d.utility.Vector2iVector(np.vstack(all_lines))
                box_lineset.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

                box_ply_path = base + "_boxes.ply"
                o3d.io.write_line_set(box_ply_path, box_lineset)
                print(f"[SAVE] Bounding boxes saved to: {box_ply_path}")

    # --- Create interactive viewer ---
    # --- Create interactive viewer (robust for headless systems) ---
    vis = o3d.visualization.Visualizer()
    success = vis.create_window("Waymo LiDAR + Boxes", width=1440, height=810, visible=True)

    if not success:
        print("[WARN] Open3D failed to create a window (likely headless mode). "
            "Skipping interactive visualization. PLY files have been saved if requested.")
        return

    # Add geometries to viewer
    for g in geoms:
        vis.add_geometry(g)

    # Render options (safe)
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = float(point_size)
        opt.background_color = np.asarray([0.93, 0.93, 0.93])
        opt.show_coordinate_frame = False
    else:
        print("[WARN] Render options unavailable (headless Open3D build).")

    # View control (safe)
    ctr = vis.get_view_control()
    if ctr is not None:
        ctr.set_up([0.0, 0.0, 1.0])       # Z up
        ctr.set_front([0.0, -1.0, 0.0])   # look toward -Y
        ctr.set_lookat([0.0, 0.0, 0.0])   # focus origin
        ctr.set_zoom(0.5)
    else:
        print("[WARN] View control unavailable (headless Open3D build).")

    vis.run()
    vis.destroy_window()


def download_waymo_folder(LOCAL_DIR = "/data/Datasets/waymodata/", SPLIT="training"):
    import os
    import subprocess

    # Configuration
    BUCKET_PREFIX = "gs://waymo_open_dataset_v_2_0_1/training/"
    
    # Ensure the local directory exists
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # List all subdirectories in the training bucket
    # Using `gsutil ls` with a trailing '/' lists only folders.
    command = f"gsutil ls {BUCKET_PREFIX}"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        folders_to_download = result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Error listing bucket contents: {e.stderr}")
        exit(1)

    print(f"Found {len(folders_to_download)} folders in the bucket.")

    for remote_folder in folders_to_download:
        folder_name = os.path.basename(remote_folder.strip('/'))
        local_folder_path = os.path.join(LOCAL_DIR, SPLIT, folder_name)

        # Check if the local folder already exists
        if os.path.exists(local_folder_path) or folder_name==SPLIT:
            print(f"Skipping existing folder: {local_folder_path}")
        else:
            print(f"Downloading new folder: {folder_name}...")
            os.makedirs(local_folder_path, exist_ok=True)
            # Fix: Copy contents of remote folder to local folder, not the folder itself
            # Add trailing /* to copy all files inside the remote folder
            remote_folder_contents = remote_folder.rstrip('/') + '/*'
            download_command = f"gsutil -m cp -r {remote_folder_contents} {local_folder_path}/"
            try:
                subprocess.run(download_command, shell=True, check=True)
                print(f"Successfully downloaded {folder_name}.")
            except subprocess.CalledProcessError as e:
                print(f"Error downloading {folder_name}: {e.stderr}")

    print("Download script finished.")



import numpy as np
import torch
import matplotlib.pyplot as plt

def transform_boxes_to_world(boxes_3d, world_from_vehicle):
    """Vehicle → World for 3D boxes."""
    if boxes_3d is None or boxes_3d.numel() == 0:
        return boxes_3d.clone() if torch.is_tensor(boxes_3d) else boxes_3d
    centers = torch.cat(
        [boxes_3d[:, :3], torch.ones((boxes_3d.size(0), 1), dtype=boxes_3d.dtype)],
        dim=1
    )
    centers_w = centers @ world_from_vehicle.T
    centers_w = centers_w[:, :3]
    R = world_from_vehicle[:3, :3]
    yaw_off = torch.atan2(R[1, 0], R[0, 0])
    yaw_w = boxes_3d[:, 6] + yaw_off
    sizes = boxes_3d[:, 3:6]
    return torch.cat([centers_w, sizes, yaw_w.unsqueeze(1)], dim=1)

def _median_signed_angle_diff(a, b):
    """median of wrapped (a-b) in [-pi,pi]."""
    d = a - b
    d = (d + np.pi) % (2*np.pi) - np.pi
    return np.median(d), np.median(np.abs(d))

def _pca_yaw(points_xy):
    """coarse yaw estimate via PCA (principal axis) in [-pi,pi]."""
    if points_xy.shape[0] < 20:
        return None
    P = points_xy - points_xy.mean(0, keepdims=True)
    C = np.cov(P.T)
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, np.argmax(vals)]  # principal axis
    yaw = np.arctan2(v[1], v[0])  # [-pi,pi]
    return yaw

def _diagnose_yaw_config_vehicle(points_xyz, boxes_vehicle, radius=4.0):
    """
    在 Vehicle 模式下，用 PCA 估出每个 box 周围点的主方向，与 box.yaw 做对比。
    输出两套常见投影约定(A/B)的中位角误差，帮助你判断应使用哪一套。
    """
    if boxes_vehicle is None or boxes_vehicle.numel() == 0:
        print("[DIAG] no boxes to diagnose yaw config.")
        return

    pts = points_xyz
    yaw_gt = []
    yaw_est = []

    for b in boxes_vehicle.numpy():
        x, y, z, dx, dy, dz, yaw = b
        mask = (np.abs(pts[:, 0] - x) < max(dx, radius)) & \
               (np.abs(pts[:, 1] - y) < max(dy, radius)) & \
               (np.abs(pts[:, 2] - z) < max(dz, 2.0))
        local = pts[mask][:, :2]
        est = _pca_yaw(local)
        if est is not None:
            yaw_gt.append(yaw)
            yaw_est.append(est)

    if len(yaw_gt) < 3:
        print("[DIAG] not enough points around boxes for PCA yaw diagnosis.")
        return

    yaw_gt = np.array(yaw_gt)
    yaw_est = np.array(yaw_est)

    # 配置 A: 你当前常用的一套（例如：Yl = - r*cos(i)*sin(az)，az: π→-π）
    # 配置 B: 另一套等价约定（Yl = + r*cos(i)*sin(az)，az: -π→π）
    # NOTE: 这里我们只能从角度差来“相对”判断，哪一套更接近 0 就选哪一套。
    med_signed_A, med_abs_A = _median_signed_angle_diff(yaw_gt, yaw_est)
    # B 相当于把 yaw 的符号翻转（或 +π），用 -yaw_gt 对比看是否更小
    med_signed_B, med_abs_B = _median_signed_angle_diff(-yaw_gt, yaw_est)

    print(f"[DIAG] yaw median diff (A) signed={np.degrees(med_signed_A):.1f}° | abs={np.degrees(med_abs_A):.1f}°")
    print(f"[DIAG] yaw median diff (B) signed={np.degrees(med_signed_B):.1f}° | abs={np.degrees(med_abs_B):.1f}°")

    if med_abs_B + 1e-3 < med_abs_A:
        print("[DIAG] → 建议切换到方案 B（把 box yaw 取反，或改用 Yl=+r*cos(i)*sin(az) 与 az: -π→π 的组合）。")
    else:
        print("[DIAG] → 继续使用方案 A（你当前的组合更接近真实）。")


def main():

    # 先在 Vehicle 下检查：return_world=False
    ds = Waymo3DDataset("/data/Datasets/waymodata/", split="training", max_frames=3, return_world=False)
    lidar, target = ds[0]

    print("\n========== BASIC INFO ==========")
    print("points:", lidar.shape)
    print("boxes :", target["boxes_3d"].shape)
    if len(target["labels"]):
        try:
            print("labels:", target["labels"].unique())
        except Exception:
            print("labels: ok")

    inten = lidar[:, 3].numpy()
    print(f"Intensity range: {inten.min():.3f} → {inten.max():.3f}")

    # Vehicle 模式下做角度诊断
    _diagnose_yaw_config_vehicle(lidar[:, :3].numpy(), target["boxes_3d"], radius=4.0)

    # ===== Vehicle frame BEV 可视化 =====
    print("\n========== MATPLOTLIB (VEHICLE FRAME) ==========")
    pts = lidar[:, :3].numpy()
    plt.figure(figsize=(7, 7))
    plt.scatter(pts[:, 0], pts[:, 1], s=0.1, c='k', alpha=0.3)
    for b in target["boxes_3d"].numpy():
        x, y, z, dx, dy, dz, yaw = b
        # 注意：Vehicle 下先不要取反 yaw，若诊断显示应当取反，再改。
        cs, sn = np.cos(yaw), np.sin(yaw)
        poly = np.array([
            [x + dx/2*cs - dy/2*sn, y + dx/2*sn + dy/2*cs],
            [x - dx/2*cs - dy/2*sn, y - dx/2*sn + dy/2*cs],
            [x - dx/2*cs + dy/2*sn, y - dx/2*sn - dy/2*cs],
            [x + dx/2*cs + dy/2*sn, y + dx/2*sn - dy/2*cs],
        ])
        plt.plot(*np.vstack([poly, poly[0]]).T, 'r-', lw=1)
    plt.axis("equal"); plt.title("BEV (Vehicle)"); plt.xlabel("X"); plt.ylabel("Y"); plt.grid(True)
    plt.show()

    print("\n========== OPEN3D (VEHICLE FRAME) ==========")
    visualize_open3d(
        lidar,                      # Vehicle 下的点
        target["boxes_3d"],         # Vehicle 下的框
        labels=target["labels"],
        invert_yaw_for_open3d=False, # 先保持 False；若诊断建议 B，再切 True/或在绘制时用 -yaw
        save_ply_path="output/frame_0000.ply"
    )

    # ===== World frame 测试（不再对点云做二次变换！）=====
    print("\n========== WORLD FRAME TEST ==========")
    ds_w = Waymo3DDataset("/data/Datasets/waymodata/", split="training", max_frames=1, return_world=True)
    lidar_w, target_w = ds_w[0]

    print("[WORLD] points:", lidar_w.shape, " | boxes:", target_w["boxes_3d"].shape)
    # 关键：此时点云已经在 World；**只**把 boxes 变到 World：
    boxes_world = transform_boxes_to_world(target_w["boxes_3d"], target_w["world_from_vehicle"])

    # 简单检查：点云坐标可能离原点很远（正常），但 boxes_world 应与点云同处一团
    pts_w = lidar_w[:, :3].numpy()
    print(f"[WORLD] points XYZ range: X[{pts_w[:,0].min():.1f},{pts_w[:,0].max():.1f}] "
          f"Y[{pts_w[:,1].min():.1f},{pts_w[:,1].max():.1f}] Z[{pts_w[:,2].min():.1f},{pts_w[:,2].max():.1f}]")
    print(f"[WORLD] T_wv translation: {target_w['world_from_vehicle'][:3,3].numpy()}")

    # BEV（World）
    plt.figure(figsize=(7, 7))
    plt.scatter(pts_w[:, 0], pts_w[:, 1], s=0.1, c='k', alpha=0.3)
    for b in boxes_world.numpy():
        x, y, z, dx, dy, dz, yaw = b
        cs, sn = np.cos(yaw), np.sin(yaw)
        poly = np.array([
            [x + dx/2*cs - dy/2*sn, y + dx/2*sn + dy/2*cs],
            [x - dx/2*cs - dy/2*sn, y - dx/2*sn + dy/2*cs],
            [x - dx/2*cs + dy/2*sn, y - dx/2*sn - dy/2*cs],
            [x + dx/2*cs + dy/2*sn, y + dx/2*sn - dy/2*cs],
        ])
        plt.plot(*np.vstack([poly, poly[0]]).T, 'r-', lw=1)
    plt.axis("equal"); plt.title("BEV (World)"); plt.xlabel("X"); plt.ylabel("Y"); plt.grid(True)
    plt.show()

    print("\n========== OPEN3D (WORLD FRAME) ==========")
    visualize_open3d(
        lidar_w,            # 已在 World 的点
        boxes_world,        # 变到 World 的框
        labels=target_w["labels"],
        invert_yaw_for_open3d=False,
        save_ply_path="output/frame_0000_2.ply"
    )

if __name__ == "__main__":
    main()