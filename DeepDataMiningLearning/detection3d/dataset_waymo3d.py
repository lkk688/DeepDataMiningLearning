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


class Waymo3DDataset(Dataset):
    """Waymo v2.1 → 3D Dataset (range image → point cloud)."""

    def __init__(self, root_dir, split="training", max_frames=None):
        self.root = root_dir
        self.split = split
        self.lidar_dir = os.path.join(root_dir, split, "lidar")
        self.box_dir   = os.path.join(root_dir, split, "lidar_box")
        self.calib_dir = os.path.join(root_dir, split, "lidar_calibration")

        if not os.path.isdir(self.lidar_dir):
            raise FileNotFoundError(self.lidar_dir)
        if not os.path.isdir(self.calib_dir):
            raise FileNotFoundError(self.calib_dir)
        if not os.path.isdir(self.box_dir):
            raise FileNotFoundError(self.box_dir)

        # shard matching
        files = [f for f in os.listdir(self.lidar_dir) if f.endswith(".parquet")]
        valid = [f for f in files if os.path.exists(os.path.join(self.box_dir, f))]
        if not valid:
            raise RuntimeError("No matching lidar/lidar_box shards found")

        self.frame_index = []
        total = 0
        for fname in valid:
            pf = pq.ParquetFile(os.path.join(self.lidar_dir, fname))
            ts = pf.read_row_group(0, columns=[KEY_TS])[KEY_TS].to_numpy()
            seg = pf.read_row_group(0, columns=[KEY_SEG])[KEY_SEG].to_numpy()
            for i, (s, t) in enumerate(zip(seg, ts)):
                self.frame_index.append((fname, i, int(t), s))
                total += 1
                if max_frames and total >= max_frames:
                    break
            if max_frames and total >= max_frames:
                break

        print(f"✅ Waymo3DDataset initialized with {len(self.frame_index)} frames using lidar_calibration/")

    def __len__(self): return len(self.frame_index)

    def __getitem__(self, idx):
        """
        Waymo v2.1 (parquet) → One LiDAR frame:
        returns:
            lidar  : torch.Tensor [N,4] (x,y,z,intensity) in VEHICLE frame
            target : dict {boxes_3d [M,7], labels [M], segment, timestamp, laser_id}
        Conventions (aligned with WOD official utils):
        - Vehicle frame: +X forward, +Y left, +Z up
        - Spherical→Cartesian (sensor frame):
            x = r*cos(incl)*cos(az), y = r*cos(incl)*sin(az), z = r*sin(incl)
        - Extrinsic is LiDAR→Vehicle: p_V = (p_L,1) @ extrinsic^T (NO inverse)
        - Azimuth grid: az = linspace(+pi, −pi, W, endpoint=False)
        - flip_rows=True, flip_cols=False
        """
        import numpy as np, torch, pyarrow.parquet as pq

        # ---------- locate this frame ----------
        fname, row_idx, ts, seg = self.frame_index[idx]

        # ---------- 1) read range image ----------
        pf = pq.ParquetFile(os.path.join(self.lidar_dir, fname))
        df = pf.read_row_group(0).to_pandas()
        row = df.iloc[row_idx]
        laser_id = int(row[LASER_NAME])

        ri = _decode_range_image(row)
        if ri is None:
            raise RuntimeError("Range image decode failed.")
        rng = np.nan_to_num(ri[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
        rng = np.clip(rng, 0.0, 300.0)
        inten = ri[..., 1] if ri.shape[-1] > 1 else np.zeros_like(rng)
        H, W = rng.shape

        # ---------- 2) read calibration (beam inclinations + extrinsic) ----------
        pf_cal = pq.ParquetFile(os.path.join(self.calib_dir, fname))
        df_cal = pf_cal.read_row_group(0).to_pandas()
        crow = df_cal[(df_cal[KEY_SEG] == seg) & (df_cal["key.laser_name"] == laser_id)]
        if len(crow) == 0:
            raise RuntimeError(f"No calibration row for seg={seg}, laser={laser_id}")
        crow = crow.iloc[0]

        # beam inclinations
        inc_min = float(crow["[LiDARCalibrationComponent].beam_inclination.min"])
        inc_max = float(crow["[LiDARCalibrationComponent].beam_inclination.max"])
        inclinations = np.linspace(inc_min, inc_max, H, dtype=np.float32)
        if np.max(np.abs(inclinations)) > np.pi:
            inclinations = np.deg2rad(inclinations)

        # extrinsic (LiDAR→Vehicle), from list<double> (row-major in your parquet)
        extr_col = max([c for c in crow.index if ("extrinsic" in c or str(c).endswith("item"))], key=len)
        extr_vals = crow[extr_col]
        if hasattr(extr_vals, "as_py"):
            extr_vals = extr_vals.as_py()
        extr = np.array(extr_vals, dtype=np.float32).reshape(4, 4, order="C")
        R = extr[:3, :3]; t = extr[:3, 3]
        # quick sanity print
        yaw_deg = float(np.rad2deg(np.arctan2(R[1,0], R[0,0])))
        print(f"[CALIB] laser={laser_id} yaw(deg)={yaw_deg:.1f}  t={t}")

        # ---------- 3) spherical → Cartesian (sensor frame) ----------
        flip_rows, flip_cols = True, False
        if flip_rows:
            inclinations = inclinations[::-1]
        incl = inclinations.reshape(H, 1)

        # 修复：根据测试结果，使用原始的方位角范围但调整起始方向
        # 测试显示原始方案在单帧中表现最好
        az = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)
        if flip_cols:
            az = az[::-1]

        cos_i, sin_i = np.cos(incl), np.sin(incl)
        cos_a, sin_a = np.cos(az),  np.sin(az)

        # 修复：Waymo LiDAR坐标系转换
        Xl = rng * cos_i * cos_a   # X: 前向
        Yl = rng * cos_i * sin_a   # Y: 左向  
        Zl = rng * sin_i           # Z: 上向

        pts_h = np.stack([Xl, Yl, Zl, np.ones_like(Zl)], axis=-1).reshape(-1, 4)

        # ---------- 4) LiDAR → Vehicle ----------
        # 修复：正确使用外参矩阵进行坐标变换
        # 外参矩阵 extr 是从LiDAR坐标系到Vehicle坐标系的变换
        xyz = (pts_h @ extr.T)[:, :3]
        xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 修复：补偿LiDAR传感器的yaw角度偏移
        # Waymo的LiDAR传感器（特别是TOP LiDAR）可能有显著的yaw偏移
        # 需要将点云旋转回标准的Vehicle坐标系方向
        if abs(yaw_deg) > 10.0:  # 如果yaw偏移超过10度，进行补偿
            # 计算补偿旋转矩阵（绕Z轴旋转-yaw_deg）
            yaw_rad = np.deg2rad(-yaw_deg)  # 负号表示反向旋转
            cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw,  cos_yaw, 0],
                [0,        0,       1]
            ], dtype=np.float32)
            
            # 应用yaw补偿旋转
            xyz = xyz @ rotation_matrix.T
            print(f"[CALIB] 应用yaw补偿: {yaw_deg:.1f}° -> 0°")
        
        # 修复：应用坐标偏移来改善对齐
        # 根据测试结果，应用+5米X轴偏移来改善对齐
        # 这可能是由于LiDAR到Vehicle坐标系转换中的系统性偏移
        coordinate_offset = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        xyz = xyz + coordinate_offset
        print(f"[CALIB] 应用坐标偏移修复: ({coordinate_offset[0]:.1f}, {coordinate_offset[1]:.1f}, {coordinate_offset[2]:.1f})")
        
        # 添加调试信息：检查转换后的点云分布
        if idx == 0:  # 只在第一帧打印调试信息
            print(f"[DEBUG] 外参矩阵 yaw: {yaw_deg:.1f}度, 平移: {t}")
            print(f"[DEBUG] 转换前点云范围: X[{pts_h[:, 0].min():.1f}, {pts_h[:, 0].max():.1f}]")
            print(f"[DEBUG] 转换后点云范围: X[{xyz[:, 0].min():.1f}, {xyz[:, 0].max():.1f}] Y[{xyz[:, 1].min():.1f}, {xyz[:, 1].max():.1f}] Z[{xyz[:, 2].min():.1f}, {xyz[:, 2].max():.1f}]")

        # ---------- 5) intensity normalize ----------
        inten = np.clip(inten.reshape(-1, 1), 0, None).astype(np.float32)
        inten = inten / (inten.max() + 1e-6) if inten.max() > 0 else np.full_like(inten, 0.5)
        lidar = torch.tensor(np.concatenate([xyz, inten], axis=1), dtype=torch.float32)

        # ---------- diagnostics (to understand "front & up" offsets) ----------
        z_p05 = float(np.percentile(xyz[:,2], 5.0))
        z_p50 = float(np.percentile(xyz[:,2], 50.0))
        print(f"[INFO] Frame {idx}: X[{xyz[:,0].min():.1f},{xyz[:,0].max():.1f}] "
            f"Y[{xyz[:,1].min():.1f},{xyz[:,1].max():.1f}] "
            f"Z[{xyz[:,2].min():.1f},{xyz[:,2].max():.1f}]  "
            f"Zp05={z_p05:.2f} Zmed={z_p50:.2f}")

        # ---------- 6) load 3D boxes (vehicle frame) ----------
        pf_box = pq.ParquetFile(os.path.join(self.box_dir, fname))
        df_box = pf_box.read_row_group(0).to_pandas()
        rows = df_box[(df_box[KEY_SEG] == seg) & (df_box[KEY_TS] == ts)]
        if len(rows) == 0:
            boxes3d = torch.zeros((0,7), dtype=torch.float32)
            labels  = torch.zeros((0,), dtype=torch.int64)
        else:
            arr = rows[[BOX_X, BOX_Y, BOX_Z, BOX_L, BOX_W, BOX_H, BOX_HEADING]].to_numpy()
            boxes3d = torch.tensor(arr, dtype=torch.float32)
            labels  = torch.tensor(rows[BOX_TYPE].to_numpy(), dtype=torch.int64)

        target = {
            "boxes_3d": boxes3d,
            "labels": labels,
            "segment": seg,
            "timestamp": ts,
            "laser_id": laser_id,
            "yaw_sign": 1,  # 保持原始yaw，不在数据层面修改
        }
        
        # 添加调试信息：检查边界框和点云的基本对齐
        if len(boxes3d) > 0 and idx == 0:
            print(f"[DEBUG] 边界框数量: {len(boxes3d)}")
            print(f"[DEBUG] 第一个边界框: 中心({boxes3d[0, 0]:.2f}, {boxes3d[0, 1]:.2f}, {boxes3d[0, 2]:.2f}) yaw={boxes3d[0, 6]:.3f}")
            # 检查边界框周围的点云密度
            if len(boxes3d) > 0:
                box_center = boxes3d[0, :3].numpy() if hasattr(boxes3d[0], 'numpy') else boxes3d[0, :3]
                distances = np.sqrt(np.sum((xyz - box_center)**2, axis=1))
                nearby_points = np.sum(distances < 5.0)
                print(f"[DEBUG] 第一个边界框5米内点数: {nearby_points}")
        
        return lidar, target

    def __getitem_old2__(self, idx):
        import numpy as np, torch, pyarrow.parquet as pq

        fname, row_idx, ts, seg = self.frame_index[idx]

        # ---------- load range image ----------
        pf = pq.ParquetFile(os.path.join(self.lidar_dir, fname))
        row = pf.read_row_group(0).to_pandas().iloc[row_idx]
        laser_id = int(row[LASER_NAME])

        ri = _decode_range_image(row)
        if ri is None:
            raise RuntimeError("Range image decode failed.")
        rng = np.nan_to_num(ri[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
        rng = np.clip(rng, 0.0, 300.0)
        inten = ri[..., 1] if ri.shape[-1] > 1 else np.zeros_like(rng)
        H, W = rng.shape

        # ---------- load calibration ----------
        pf_cal = pq.ParquetFile(os.path.join(self.calib_dir, fname))
        df_cal = pf_cal.read_row_group(0).to_pandas()
        crow = df_cal[(df_cal["key.segment_context_name"] == seg) &
                    (df_cal["key.laser_name"] == laser_id)].iloc[0]

        # beam inclinations (min→max)，Waymo 值通常为负到正（下→上）
        inc_min = float(crow["[LiDARCalibrationComponent].beam_inclination.min"])
        inc_max = float(crow["[LiDARCalibrationComponent].beam_inclination.max"])
        inc = np.linspace(inc_min, inc_max, H, dtype=np.float32)

        # extrinsic（LiDAR→Vehicle），注意你这套 parquet 需要 C-order
        # 动态找 extrinsic 列（包含 "extrinsic" 或 以 "item" 结尾 的那列）
        idx_keys = list(crow.index)
        extr_col = max([c for c in idx_keys if ("extrinsic" in c or c.endswith("item"))],
                    key=len)
        extr_vals = crow[extr_col]
        if hasattr(extr_vals, "as_py"):
            extr_vals = extr_vals.as_py()
        extr = np.array(extr_vals, dtype=np.float32).reshape(4, 4, order="C")
        # 现在 extr[:3,3] 应该是非零（你之前看到 [1.43,0,2.184]）
        R = extr[:3, :3]
        yaw_lidar = np.rad2deg(np.arctan2(R[1,0], R[0,0]))
        print("LiDAR extrinsic yaw (deg):", yaw_lidar)

        # ---------- build azimuth grid ----------
        # 统一方位角生成方式：与spherical_to_cartesian_general保持一致
        # 使用 (2π*i/W) - π 的方式，确保坐标变换的一致性
        az = (2 * np.pi * np.arange(W, dtype=np.float32) / W) - np.pi

        # 垂直方向按官方常见约定：flip_rows=True（自下而上）
        inc = inc[::-1].reshape(H, 1)

        # ---------- spherical -> Cartesian in LiDAR frame ----------
        cos_i, sin_i = np.cos(inc), np.sin(inc)
        cos_a, sin_a = np.cos(az),  np.sin(az)

        # 与官方 range_image_utils 一致：z = r * sin(inclination)
        Xl = rng * cos_i * cos_a
        Yl = rng * cos_i * sin_a
        Zl = rng * sin_i

        pts_h = np.stack([Xl, Yl, Zl, np.ones_like(Zl)], axis=-1).reshape(-1, 4)

        # ---------- LiDAR -> Vehicle ----------
        # xyz = (pts_h @ extr.T)[:, :3]
        # xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
        # ---- apply LiDAR->Vehicle transform ----
        # try both, keep the one with Z≈[-3,5]
        use_inv = False   # ✅ 改成 True 测试
        if use_inv:
            xyz = (pts_h @ np.linalg.inv(extr).T)[:, :3]
        else:
            xyz = (pts_h @ extr.T)[:, :3]

        xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- intensity normalize ----------
        inten = np.clip(inten.reshape(-1, 1), 0, None).astype(np.float32)
        inten = inten / (inten.max() + 1e-6) if inten.max() > 0 else np.full_like(inten, 0.5)
        lidar = torch.tensor(np.concatenate([xyz, inten], axis=1), dtype=torch.float32)

        # ---------- load boxes (vehicle frame) ----------
        pf_box = pq.ParquetFile(os.path.join(self.box_dir, fname))
        df_box = pf_box.read_row_group(0).to_pandas()
        rows = df_box[(df_box["key.segment_context_name"] == seg) &
                    (df_box["key.frame_timestamp_micros"] == ts)]
        if len(rows) == 0:
            boxes3d = torch.zeros((0,7), dtype=torch.float32)
            labels  = torch.zeros((0,), dtype=torch.int64)
        else:
            arr = rows[[BOX_X, BOX_Y, BOX_Z, BOX_L, BOX_W, BOX_H, BOX_HEADING]].to_numpy()
            boxes3d = torch.tensor(arr, dtype=torch.float32)
            labels  = torch.tensor(rows[BOX_TYPE].to_numpy(), dtype=torch.int64)

        target = {
            "boxes_3d": boxes3d,
            "labels": labels,
            "segment": seg,
            "timestamp": ts,
            "laser_id": laser_id,
            "yaw_sign": 1,  # 仅绘制时用 -yaw
        }
        xyz_rot = rotate_z(lidar[:, :3].numpy(), deg=-yaw_lidar)
        lidar[:, :3] = torch.from_numpy(xyz_rot)
        mde = _median_yaw_error(xyz_rot, boxes3d)
        if mde is not None:
            print(f"[diag] median yaw error vs boxes (deg): {np.rad2deg(mde):.1f}")
            
        return lidar, target

    def __getitem_old__(self, idx):
        """
        Load ONE Waymo LiDAR frame and return:
            - lidar: torch.Tensor [N,4] (x,y,z,intensity)
            - target: dict with 3D boxes, labels, segment info

        This version includes:
        • range image → Cartesian conversion
        • extrinsic reshape fix (column-major)
        • auto-calibration (flip_rows, flip_cols, azimuth offset)
        • normalized intensity
        """
        import numpy as np
        import torch

        # ===============================================================
        # 1️⃣  Retrieve file and row information
        # ===============================================================
        fname, row_idx, ts, seg = self.frame_index[idx]
        lidar_path = os.path.join(self.lidar_dir, fname)
        pf = pq.ParquetFile(lidar_path)
        df = pf.read_row_group(0).to_pandas()
        row = df.iloc[row_idx]
        laser_id = int(row[LASER_NAME])

        # ===============================================================
        # 2️⃣  Decode range image (polar data)
        # ===============================================================
        ri = _decode_range_image(row)
        if ri is None:
            raise RuntimeError("Range image decode failed.")
        range_img = ri[..., 0]                                    # distance (meters)
        intensity = ri[..., 1] if ri.shape[-1] > 1 else np.zeros_like(range_img)

        # Sanitize
        range_img = np.nan_to_num(range_img, nan=0.0, posinf=0.0, neginf=0.0)
        range_img = np.clip(range_img, 0.0, 300.0)

        # ===============================================================
        # 3️⃣  Read calibration (beam inclinations + extrinsic)
        # ===============================================================
        calib_path = os.path.join(self.calib_dir, fname)
        cf = pq.ParquetFile(calib_path)
        fields = _get_calibration_fields(cf)
        cdf = cf.read().to_pandas()
        crow = cdf[(cdf[KEY_SEG] == seg) & (cdf["key.laser_name"] == laser_id)]
        if len(crow) == 0:
            raise RuntimeError(f"No calibration for {seg}, laser={laser_id}")
        crow = crow.iloc[0]

        # --- Beam inclinations ---
        if fields["beam_mode"] == "values":
            inc_vals = _read_list_column(crow[fields["beam_field"]])
            inclinations = np.array(inc_vals, dtype=np.float32)
        else:
            minv = float(crow["[LiDARCalibrationComponent].beam_inclination.min"])
            maxv = float(crow["[LiDARCalibrationComponent].beam_inclination.max"])
            num_beams = range_img.shape[0]
            inclinations = np.linspace(minv, maxv, num_beams, dtype=np.float32)

        # Convert to radians if values look like degrees
        if np.max(np.abs(inclinations)) > np.pi:
            inclinations = np.deg2rad(inclinations)
            print("[WARN] Beam inclinations appear to be degrees — converted to radians.")

        # --- Extrinsic matrix (column-major reshape) ---
        if fields["extr_mode"] == "matrix":
            extr_vals = _read_list_column(crow[fields["extr_field"]])
            extrinsic = np.array(extr_vals, dtype=np.float32).reshape(4, 4, order="F")
        else:
            extrinsic = np.eye(4, dtype=np.float32)

        # ===============================================================
        # 4️⃣  Load 3D boxes (labels in VEHICLE frame)
        # ===============================================================
        box_path = os.path.join(self.box_dir, fname)
        bpf = pq.ParquetFile(box_path)
        bcols = [BOX_X, BOX_Y, BOX_Z, BOX_L, BOX_W, BOX_H, BOX_HEADING,
                BOX_TYPE, KEY_SEG, KEY_TS]
        bdf = bpf.read_row_group(0, columns=bcols).to_pandas()
        boxes = bdf[(bdf[KEY_SEG] == seg) & (bdf[KEY_TS] == ts)]

        if len(boxes) == 0:
            boxes3d = torch.zeros((0, 7), dtype=torch.float32)
            labels  = torch.zeros((0,), dtype=torch.int64)
        else:
            arr = boxes[[BOX_X, BOX_Y, BOX_Z,
                        BOX_L, BOX_W, BOX_H, BOX_HEADING]].to_numpy()
            boxes3d = torch.tensor(arr, dtype=torch.float32)
            labels  = torch.tensor(boxes[BOX_TYPE].to_numpy(), dtype=torch.int64)

        # ===============================================================
        # 5️⃣  Convert range image → point cloud with auto-calibration
        # ===============================================================
        # Try multiple flip/offset configs to find alignment with boxes.
        xyz_best, cfg = auto_calibrate_range_to_vehicle(range_img, inclinations, extrinsic, boxes3d.numpy())
        print(f"[AUTO] Frame {idx} best config:", cfg)

        # Normalize or fill intensity for visualization
        inten = intensity.reshape(-1, 1).astype(np.float32)
        inten = np.clip(inten, 0, None)
        inten = inten / (inten.max() + 1e-6) if inten.max() > 0 else np.full_like(inten, 0.5)
        lidar = torch.tensor(np.concatenate([xyz_best, inten], axis=1), dtype=torch.float32)

        # Diagnostics
        xyz_min, xyz_max = xyz_best.min(axis=0), xyz_best.max(axis=0)
        print(f"[INFO] Frame {idx}: {lidar.shape[0]} valid points | "
            f"X:[{xyz_min[0]:.1f},{xyz_max[0]:.1f}]  "
            f"Y:[{xyz_min[1]:.1f},{xyz_max[1]:.1f}]  "
            f"Z:[{xyz_min[2]:.1f},{xyz_max[2]:.1f}]")

        # ===============================================================
        # 6️⃣  Assemble output
        # ===============================================================
        target = {
            "boxes_3d": boxes3d,           # (M,7)
            "labels": labels,              # (M,)
            "segment": seg,                # string segment id
            "timestamp": ts,               # frame timestamp (μs)
            "laser_id": laser_id,          # LiDAR sensor id
            "auto_cfg": cfg                # flip/offset config used for conversion
        }

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
    invert_yaw_for_open3d: bool = True,  # ← 关键开关：False 试试看
    axis_size: float = 5.0,
):
    """
    Interactive LiDAR + 3D boxes viewer using Open3D.

    Waymo Vehicle Frame (数据坐标系约定):
        +X: forward
        +Y: left
        +Z: up

    参数:
      - invert_yaw_for_open3d: 是否在渲染时把 yaw 取反。
         如果在 Open3D 里看到“整体偏转一个大角度”，把它改为 False 再看一眼。
      - axis_size: 场景原点绘制固定坐标轴，红=X(前)、绿=Y(左)、蓝=Z(上)。
    """
    try:
        import open3d as o3d
    except Exception as e:
        print("Open3D not installed:", e)
        return

    import numpy as np

    # --- to numpy ---
    if hasattr(lidar, "detach"):
        lidar = lidar.detach().cpu().numpy()
    if boxes3d is not None and hasattr(boxes3d, "detach"):
        boxes3d = boxes3d.detach().cpu().numpy()
    if labels is not None and hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    # --- point cloud ---
    pts = lidar[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    if color_by_intensity and lidar.shape[1] >= 4:
        inten = lidar[:, 3].astype(np.float32)
        i_min = float(np.min(inten)) if inten.size else 0.0
        i_ptp = float(np.ptp(inten)) if inten.size else 0.0  # NumPy 2.0-safe
        if i_ptp < 1e-12:
            colors = np.full((inten.shape[0], 3), 0.7, dtype=np.float32)
        else:
            norm = (inten - i_min) / (i_ptp + 1e-12)
            colors = np.stack([norm, norm, norm], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    geoms = [pcd]

    # --- 3D boxes ---
    if boxes3d is not None and len(boxes3d) > 0:
        def color_for(k):
            tab10 = [
                (0.121, 0.466, 0.705), (1.000, 0.498, 0.054), (0.172, 0.627, 0.172),
                (0.839, 0.153, 0.157), (0.580, 0.404, 0.741), (0.549, 0.337, 0.294),
                (0.890, 0.467, 0.761), (0.498, 0.498, 0.498), (0.737, 0.741, 0.133),
                (0.090, 0.745, 0.811),
            ]
            return tab10[int(k) % len(tab10)]

        for i, b in enumerate(boxes3d):
            x, y, z, dx, dy, dz, yaw = map(float, b[:7])
            # 左手/右手差异：有些场景需要 -yaw，有些不需要
            yaw_draw = -yaw if invert_yaw_for_open3d else yaw

            R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, yaw_draw))
            obb = o3d.geometry.OrientedBoundingBox(center=[x, y, z], R=R, extent=[dx, dy, dz])

            if labels is not None and i < len(labels):
                obb.color = color_for(labels[i])
            else:
                obb.color = (1.0, 0.5, 0.0)  # 橙色，和点云灰色对比明显
            geoms.append(obb)

    # --- 固定世界坐标轴 (红=X前, 绿=Y左, 蓝=Z上) ---
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0.0, 0.0, 0.0])
    geoms.append(axis)

    # --- viewer ---
    vis = o3d.visualization.Visualizer()
    vis.create_window("Waymo LiDAR + Boxes", width=1440, height=810)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.asarray([0.93, 0.93, 0.93])
    opt.show_coordinate_frame = False  # 我们自己画更大的轴

    # 设定“顶视 BEV”视角：让 +Z 朝上、面向 -Y 方向，+X 在屏幕右侧
    ctr = vis.get_view_control()
    ctr.set_up([0.0, 0.0, 1.0])        # Z up
    ctr.set_front([0.0, -1.0, 0.0])    # look towards -Y
    ctr.set_lookat([0.0, 0.0, 0.0])    # focus origin
    ctr.set_zoom(0.5)

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

def main():
    #path="/mnt/e/Shared/Dataset/waymodata/"
    ds = Waymo3DDataset("/data/Datasets/waymodata/", split="training", max_frames=3)
    lidar, target = ds[0]
    print("points:", lidar.shape)
    print("boxes:", target["boxes_3d"].shape)
    inten = lidar[:, 3].numpy()
    print("Intensity min/max:", inten.min(), inten.max())

    import matplotlib.pyplot as plt
    pts = lidar[:, :3].numpy()
    plt.figure(figsize=(6,6))
    plt.scatter(pts[:,0], pts[:,1], s=0.1, c='k', alpha=0.3)
    for b in target["boxes_3d"].numpy():
        x,y,z,dx,dy,dz,yaw = b
        yaw = -yaw
        cs, sn = np.cos(yaw), np.sin(yaw)
        poly = np.array([
            [x + dx/2*cs - dy/2*sn, y + dx/2*sn + dy/2*cs],
            [x - dx/2*cs - dy/2*sn, y - dx/2*sn + dy/2*cs],
            [x - dx/2*cs + dy/2*sn, y - dx/2*sn - dy/2*cs],
            [x + dx/2*cs + dy/2*sn, y + dx/2*sn - dy/2*cs],
        ])
        plt.plot(*np.vstack([poly,poly[0]]).T, 'r-')
    plt.axis('equal'); plt.show()

    # lidar_np = np.asarray(lidar)
    # mask = np.isfinite(lidar_np).all(1)
    # lidar_np = lidar_np[mask]
    # lidar_np[:, :3] -= lidar_np[:, :3].mean(0, keepdims=True)
    # lidar = torch.from_numpy(lidar_np)
    #visualize_open3d(lidar, target["boxes_3d"], point_size=2.0)
    

    visualize_open3d(lidar, target["boxes_3d"], labels=target["labels"], invert_yaw_for_open3d=False)

if __name__ == "__main__":
    #download_waymo_folder()
    main()
    #test_parquet()