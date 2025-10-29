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


import os
import numpy as np
import torch
import pyarrow.parquet as pq
from torch.utils.data import Dataset


class Waymo3DDataset(Dataset):
    """
    Waymo Open Dataset v2.x → Unified 3D LiDAR + Box Dataset.

    This class loads:
        - LiDAR range images (Parquet format)
        - LiDAR calibration (beam inclinations + extrinsic)
        - Vehicle poses (for world frame)
        - 3D bounding boxes (LiDARBoxComponent)
    and converts them into consistent Vehicle or World coordinates.

    ✅ Coordinate conventions:
        Vehicle frame: +X forward, +Y left, +Z up  (Right-handed)
        LiDAR extrinsic:  T_vehicle←lidar  (row-major, no inverse)
        Box heading:      clockwise in dataset → negated to CCW
    """

    def __init__(self, root_dir, split="training", max_frames=None, return_world=False):
        """
        Args:
            root_dir:    Path to Waymo dataset root.
            split:       "training" | "validation" | "testing"
            max_frames:  Limit number of frames (for debugging)
            return_world: If True, transforms points and boxes into World frame.
        """
        self.root = root_dir
        self.split = split
        self.return_world = return_world

        # --- Dataset folder paths ---
        self.lidar_dir = os.path.join(root_dir, split, "lidar")
        self.calib_dir = os.path.join(root_dir, split, "lidar_calibration")
        self.box_dir = os.path.join(root_dir, split, "lidar_box")
        self.vpose_dir = os.path.join(root_dir, split, "vehicle_pose")

        for p in [self.lidar_dir, self.calib_dir, self.box_dir, self.vpose_dir]:
            if not os.path.isdir(p):
                raise FileNotFoundError(p)

        # --- Index all lidar frames ---
        files = [f for f in os.listdir(self.lidar_dir) if f.endswith(".parquet")]
        valid = [f for f in files if os.path.exists(os.path.join(self.box_dir, f))]
        self.frame_index = []
        total = 0
        for fname in valid:
            pf = pq.ParquetFile(os.path.join(self.lidar_dir, fname))
            df = pf.read_row_group(0, columns=["key.segment_context_name", "key.frame_timestamp_micros"]).to_pandas()
            for seg, ts in zip(df["key.segment_context_name"], df["key.frame_timestamp_micros"]):
                self.frame_index.append((fname, int(ts), seg))
                total += 1
                if max_frames and total >= max_frames:
                    break
            if max_frames and total >= max_frames:
                break

        print(f"✅ Waymo3DDataset initialized with {len(self.frame_index)} frames | return_world={self.return_world}")

    # ---------------------------------------------------------------------
    @staticmethod
    def _decode_range_image(row, return_id=1):
        """
        Decode one range image (Parquet: flattened float list).
        Returns a numpy array [H,W,C] (float32).
        """
        kv = f"[LiDARComponent].range_image_return{return_id}.values"
        ks = f"[LiDARComponent].range_image_return{return_id}.shape"
        vals = row[kv].as_py() if hasattr(row[kv], "as_py") else row[kv]
        shp = row[ks].as_py() if hasattr(row[ks], "as_py") else row[ks]
        arr = np.array(vals, np.float32).reshape(shp, order="C")  # Waymo = row-major
        return arr

    # ---------------------------------------------------------------------
    def __getitem__(self, idx):
        fname, ts, seg = self.frame_index[idx]

        # ---------- 1️⃣ Read all LiDAR rows for this frame ----------
        pf = pq.ParquetFile(os.path.join(self.lidar_dir, fname))
        df = pf.read_row_group(0).to_pandas()
        rows_t = df[(df["key.segment_context_name"] == seg) &
                    (df["key.frame_timestamp_micros"] == ts)]
        if len(rows_t) == 0:
            raise RuntimeError("No LiDAR rows for this frame.")
            

        # ---------- 2️⃣ Identify the TOP LiDAR (highest mount) ----------
        pf_cal = pq.ParquetFile(os.path.join(self.calib_dir, fname))
        df_cal = pf_cal.read_row_group(0).to_pandas()

        if idx == 0:
            print("Available calibration entries in this file:")
            print(df_cal[["key.segment_context_name", "key.laser_name",
                        "[LiDARCalibrationComponent].beam_inclination.min",
                        "[LiDARCalibrationComponent].beam_inclination.max"]].head(10))

        def get_T_vl(lid):
            """Robustly fetch LiDAR→Vehicle extrinsic, ensuring correct ID match."""
            # Force both sides to int32 for comparison
            lid_int = int(lid)
            key_lids = df_cal["key.laser_name"].astype(np.int32)
            seg_mask = df_cal["key.segment_context_name"] == seg
            lid_mask = key_lids == lid_int
            sel = df_cal[seg_mask & lid_mask]
            if len(sel) == 0:
                raise RuntimeError(f"No calibration row found for seg={seg}, laser_name={lid_int}.")
            crow = sel.iloc[0]

            # Extract extrinsic (row-major)
            extr_cols = [c for c in crow.index if "extrinsic" in str(c)]
            extr_col = min(extr_cols, key=len)
            extr_vals = crow[extr_col]
            extr_vals = extr_vals.as_py() if hasattr(extr_vals, "as_py") else extr_vals
            extr = np.array(extr_vals, np.float32).reshape(4, 4, order="C")

            # Quick sanity check: TOP LiDAR translation ~[1.4, 0, 2.0], yaw≈0°
            t = extr[:3, 3]
            yaw_deg = np.degrees(np.arctan2(extr[1, 0], extr[0, 0]))
            print(f"[CALIB_CHECK] lid={lid_int}  trans={t.round(3)}  yaw={yaw_deg:.1f}°")
            return extr

        if idx == 0:
            for lid in sorted(rows_t["key.laser_name"].unique()):
                _ = get_T_vl(lid)

        # Pick LiDAR with largest Z translation (the TOP sensor)
        cand_ids = sorted(rows_t["key.laser_name"].unique().tolist())
        heights = {}
        for lid in cand_ids:
            T_vl = get_T_vl(lid)
            heights[lid] = float(T_vl[2, 3])
        laser_id = max(heights, key=heights.get)
        row = rows_t[rows_t["key.laser_name"] == laser_id].iloc[0]

        # ---------- Detect RAW vs VIRTUAL range image ----------
        pose_key = "[LiDARComponent].range_image_pose.values"
        pose_shape_key = "[LiDARComponent].range_image_pose.shape"

        if pose_key in row and pose_shape_key in row:
            vals = row[pose_key]
            # Waymo stores as list<float> for [H,W,6]; empty if virtual
            if hasattr(vals, "as_py"):
                vals = vals.as_py()
            is_raw = len(vals) > 0
        else:
            # Newer parquet files might use 'pose_compressed' instead
            compressed_key = "[LiDARComponent].range_image_pose_compressed"
            if compressed_key in row:
                data = row[compressed_key]
                if hasattr(data, "as_py"):
                    data = data.as_py()
                is_raw = data is not None and len(data) > 0
            else:
                is_raw = False

        mode = "RAW (needs extrinsic)" if is_raw else "VIRTUAL (already in vehicle frame)"
        print(f"[INFO] Range image type detected: {mode}")


        # ---------- Build LiDAR→Vehicle extrinsic (row-major) ----------
        T_vl = get_T_vl(laser_id)

        # Compute yaw from its rotation block
        R = T_vl[:3, :3]
        yaw_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

        # --- Fix mixed calibration: enforce TOP yaw≈0 for z>2 m sensors ---
        # if T_vl[2, 3] > 1.8 and abs(yaw_deg) > 30.0:
        #     print(f"[CALIB_FIX] Detected bad yaw ({yaw_deg:.1f}°) for TOP LiDAR; resetting rotation to identity.")
        #     T_vl[:3, :3] = np.eye(3, dtype=np.float32)
        #     yaw_deg = 0.0
        # Keep translation t, preserve pitch/roll, remove only the bad yaw
        t = T_vl[:3, 3].copy()

        # Extract the (wrong) yaw from R (already computed as yaw_deg)
        yaw_rad = np.deg2rad(yaw_deg)

        # Build Z-rotation that cancels that yaw
        cy, sy = np.cos(-yaw_rad), np.sin(-yaw_rad)
        Rz_fix = np.array([[cy, -sy, 0.0],
                        [sy,  cy, 0.0],
                        [0.0, 0.0, 1.0]], dtype=np.float32)

        # Apply on the RIGHT to remove yaw while preserving the original tilt:
        # R_corrected = R * Rz(-yaw)
        R_corr = R @ Rz_fix

        T_vl[:3, :3] = R_corr
        T_vl[:3,  3] = t
        yaw_deg = 0.0  # yaw corrected to ~0 for logging


        print(f"[INFO] Using LiDAR id={laser_id} "
            f"(height={T_vl[2,3]:.2f} m, yaw≈{yaw_deg:.1f}°)")

        # ---------- 3️⃣ Decode the range image ----------
        ri = self._decode_range_image(row, return_id=1)
        rng = np.clip(np.nan_to_num(ri[..., 0], nan=0.0), 0.0, 300.0)
        inten = ri[..., 1] if ri.shape[-1] > 1 else np.zeros_like(rng)
        H, W = rng.shape

        # ---------- 4️⃣ Beam inclinations ----------
        # crow = df_cal[(df_cal["key.segment_context_name"] == seg) &
        #               (df_cal["key.laser_name"] == laser_id)].iloc[0]
        # inc_min = float(crow["[LiDARCalibrationComponent].beam_inclination.min"])
        # inc_max = float(crow["[LiDARCalibrationComponent].beam_inclination.max"])
        # inclinations = np.linspace(inc_min, inc_max, H, dtype=np.float32)
        # if np.max(np.abs(inclinations)) > np.pi:
        #     inclinations = np.deg2rad(inclinations)
        # ---------- 4️⃣ Beam inclinations (prefer full vector) ----------
        crow = df_cal[(df_cal["key.segment_context_name"] == seg) &
                    (df_cal["key.laser_name"] == laser_id)].iloc[0]

        # Try to read the full non-uniform vector first (name may vary depending on your export)
        cand_cols = [
            "[LiDARCalibrationComponent].beam_inclinations.values",
            "[LiDARCalibrationComponent].beam_inclination.values",
            "beam_inclinations",  # some exporters shorten names
        ]
        inc_vals = None
        for c in cand_cols:
            if c in crow and crow[c] is not None:
                v = crow[c]
                if hasattr(v, "as_py"): v = v.as_py()
                if v is not None and len(v) > 0:
                    inc_vals = np.array(v, dtype=np.float32)
                    break

        if inc_vals is not None:
            inclinations = inc_vals
        else:
            # fallback: uniform spacing if vector not provided
            inc_min = float(crow["[LiDARCalibrationComponent].beam_inclination.min"])
            inc_max = float(crow["[LiDARCalibrationComponent].beam_inclination.max"])
            inclinations = np.linspace(inc_min, inc_max, H, dtype=np.float32)

        # Units: some parquet dumps keep degrees; convert if needed
        if np.max(np.abs(inclinations)) > np.pi:
            inclinations = np.deg2rad(inclinations)

        # Sanity
        if len(inclinations) != H:
            # Rare exporter mismatch: resample to H to avoid misalignment
            inclinations = np.interp(
                np.linspace(0, len(inclinations)-1, H),
                np.arange(len(inclinations)),
                inclinations
            ).astype(np.float32)

        # ---------- 5️⃣ Convert Spherical → LiDAR Cartesian ----------
        # Waymo’s convention: +X forward, +Y left, +Z up
        incl = inclinations[::-1].reshape(H, 1)                    # flip vertically
        az = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)
        cos_i, sin_i = np.cos(incl), np.sin(incl)
        cos_a, sin_a = np.cos(az), np.sin(az)

        Xl = rng * cos_i * cos_a
        Yl = rng * cos_i * sin_a
        Zl = rng * sin_i
        pts_l = np.stack([Xl, Yl, Zl, np.ones_like(Zl)], axis=-1).reshape(-1, 4)

        # ---------- 6️⃣ LiDAR → Vehicle (apply extrinsic) ----------
        #pts_v = (pts_l @ T_vl.T)[:, :3]
        #Waymo’s vehicle convention (+X forward, +Y left, +Z up)
        #(X-red, Y-green, Z-blue)
        is_raw = True
        if is_raw:    # range_image_pose_compressed exists
            pts_v = (pts_l @ T_vl.T)[:, :3]
        else:          # virtual range image
            ## For virtual range images, the points are already in vehicle frame
            pts_v = pts_l[:, :3]
        
        xyz_vehicle = np.nan_to_num(pts_v, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- 7️⃣ Optional Vehicle → World ----------
        if self.return_world:
            pf_vp = pq.ParquetFile(os.path.join(self.vpose_dir, fname))
            df_vp = pf_vp.read_row_group(0).to_pandas()
            vrow = df_vp[(df_vp["key.segment_context_name"] == seg) &
                         (df_vp["key.frame_timestamp_micros"] == ts)].iloc[0]
            vp_vals = vrow["[VehiclePoseComponent].world_from_vehicle.transform"]
            vp_vals = vp_vals.as_py() if hasattr(vp_vals, "as_py") else vp_vals
            T_wv = np.array(vp_vals, np.float32).reshape(4, 4, order="C")
            pts_vh = np.concatenate([xyz_vehicle, np.ones((xyz_vehicle.shape[0], 1), np.float32)], axis=1)
            xyz_world = (pts_vh @ T_wv.T)[:, :3]
            XYZ = xyz_world
        else:
            T_wv = None
            XYZ = xyz_vehicle

        # ---------- 8️⃣ Normalize intensity ----------
        inten = inten.reshape(-1, 1).astype(np.float32)
        inten = inten / (inten.max() + 1e-6)
        lidar = torch.tensor(np.concatenate([XYZ, inten], axis=1), dtype=torch.float32)

        # ---------- 9️⃣ Load and Transform 3D Boxes ----------
        pf_box = pq.ParquetFile(os.path.join(self.box_dir, fname))
        df_box = pf_box.read_row_group(0).to_pandas()
        rows_b = df_box[(df_box["key.segment_context_name"] == seg) &
                        (df_box["key.frame_timestamp_micros"] == ts) ]
        if len(rows_b) == 0:
            boxes_any = torch.zeros((0, 7), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            f = [
                "[LiDARBoxComponent].box.center.x",
                "[LiDARBoxComponent].box.center.y",
                "[LiDARBoxComponent].box.center.z",
                "[LiDARBoxComponent].box.size.x",
                "[LiDARBoxComponent].box.size.y",
                "[LiDARBoxComponent].box.size.z",
                "[LiDARBoxComponent].box.heading",
            ]
            arr = rows_b[f].to_numpy().astype(np.float32)
            # 6] = -arr[:, 6]  # clockwise → CCW (right-handed yaw)
            # --- Boxes are in VEHICLE frame, but appear mirrored about +X (left/right flipped).
            # Fix handedness by reflecting across X: y -> -y and yaw -> -yaw.
            arr[:, 1] *= -1.0        # y
            arr[:, 6] *= -1.0        # yaw (CCW)

            # # --- LiDAR → Vehicle centers ---
            # centers_l = arr[:, :3]
            # centers_l_h = np.concatenate([centers_l, np.ones((centers_l.shape[0], 1), np.float32)], axis=1)
            # centers_v = (centers_l_h @ T_vl.T)[:, :3]

            # # --- Adjust headings by LiDAR yaw ---
            # lidar_yaw_rad = np.deg2rad(yaw_deg)
            # headings_v = arr[:, 6] + lidar_yaw_rad
            # headings_v = (headings_v + np.pi) % (2 * np.pi) - np.pi
            # sizes = arr[:, 3:6]

            # # Combine back to VEHICLE frame boxes
            # boxes_vehicle = np.concatenate([centers_v, sizes, headings_v[:, None]], axis=1)
            # --- Boxes are already in VEHICLE frame ---
            centers_v  = arr[:, :3]        # use directly
            sizes      = arr[:, 3:6]
            headings_v = arr[:, 6]         # already vehicle-frame yaw (after CW→CCW flip)

            # boxes_vehicle = np.concatenate(
            #     [centers_v, sizes, headings_v[:, None]], axis=1
            # )
            boxes_vehicle = np.concatenate([centers_v, sizes, headings_v[:, None]], axis=1).astype(np.float32)

            # --- Optionally: Vehicle → World for boxes (if return_world=True) ---
            if self.return_world:
                centers_v_h = np.concatenate([centers_v, np.ones((centers_v.shape[0], 1), np.float32)], axis=1)
                centers_w = (centers_v_h @ T_wv.T)[:, :3]
                yaw_wv = float(np.arctan2(T_wv[1, 0], T_wv[0, 0]))  # vehicle yaw in world
                headings_w = headings_v + yaw_wv
                headings_w = (headings_w + np.pi) % (2 * np.pi) - np.pi
                boxes_any = torch.tensor(np.concatenate([centers_w, sizes, headings_w[:, None]], axis=1),
                                         dtype=torch.float32)
            else:
                boxes_any = torch.tensor(boxes_vehicle, dtype=torch.float32)

            labels = torch.tensor(rows_b["[LiDARBoxComponent].type"].to_numpy(), dtype=torch.int64)


        z_pts = np.percentile(xyz_vehicle[:,2], [1, 50, 99])
        z_box_bottom = centers_v[:,2] - sizes[:,2]*0.5
        print(f"[CHECK] points Z p01/50/99 = {z_pts}")
        print(f"[CHECK] box bottom z: mean={float(z_box_bottom.mean()):.2f}  min={float(z_box_bottom.min()):.2f}")


        # ---------- 🔟 Assemble final output ----------
        target = {
            "boxes_3d": boxes_any,        # same frame as lidar (Vehicle or World)
            "labels": labels,
            "segment": seg,
            "timestamp": ts,
            "laser_id": int(laser_id),
        }
        if self.return_world:
            target["world_from_vehicle"] = torch.tensor(T_wv, dtype=torch.float32)

        # ---------- 🧭 Debug summary ----------
        if idx == 0:
            print(f"[DEBUG] Frame {idx}: {len(lidar)} pts | "
                  f"X[{XYZ[:,0].min():.1f},{XYZ[:,0].max():.1f}] "
                  f"Y[{XYZ[:,1].min():.1f},{XYZ[:,1].max():.1f}] "
                  f"Z[{XYZ[:,2].min():.1f},{XYZ[:,2].max():.1f}] "
                  f"| boxes={len(boxes_any)}")

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

# ---------- LiDAR → Image projection function (for your CameraCalibration schema) ----------
def project_lidar_to_image_v2_old(points_vehicle: np.ndarray, camera_row):
    """
    Project right-handed VEHICLE-frame LiDAR points onto a Waymo camera image.

    Args:
        points_vehicle : (N,3) LiDAR points in VEHICLE frame (+X fwd, +Y left, +Z up)
        camera_row     : one row from camera_calibration parquet

    Returns:
        uv      : (M,2) pixel coordinates (u,v)
        depth   : (M,)  depth values (Z in camera frame)
        mask    : (N,)  boolean mask of projected points
        width   : image width
        height  : image height
    """
    # ---- 1️⃣ Intrinsics ----
    fx = float(camera_row["[CameraCalibrationComponent].intrinsic.f_u"])
    fy = float(camera_row["[CameraCalibrationComponent].intrinsic.f_v"])
    cx = float(camera_row["[CameraCalibrationComponent].intrinsic.c_u"])
    cy = float(camera_row["[CameraCalibrationComponent].intrinsic.c_v"])
    width  = int(camera_row["[CameraCalibrationComponent].width"])
    height = int(camera_row["[CameraCalibrationComponent].height"])

    # ---- 2️⃣ Extrinsic: Vehicle → Camera (row-major) ----
    extr_vals = camera_row["[CameraCalibrationComponent].extrinsic.transform"]
    extr_vals = extr_vals.as_py() if hasattr(extr_vals, "as_py") else extr_vals
    T_cv = np.array(extr_vals, dtype=np.float32).reshape(4, 4, order="C")

    # ---- 3️⃣ Transform points to camera coordinates ----
    pts_vh = np.c_[points_vehicle, np.ones((points_vehicle.shape[0], 1), np.float32)]
    pts_c  = (pts_vh @ T_cv.T)[:, :3]       # Vehicle → Camera
    #Xc, Yc, Zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

    # Convert from Waymo camera coordinate frame to image plane convention:
    #   Waymo camera:  +X right, +Y down, +Z forward
    #   OpenCV project: +X right, +Y down, +Z forward
    # BUT our LiDAR→Camera transform currently has +Y up, +Z forward.
    # So swap Y/Z or flip signs so optical axis matches.

    Xc, Yc, Zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

    # ---- fix orientation ----
    # If u,v are huge (hundreds of thousands) while Z>0, try these one at a time:
    # 1.  negate both X and Y  (180° roll)
    # 2.  or swap Y/Z depending on exporter.
    # For Waymo v2.0–v2.01 parquet (FRONT camera), the correct fix is:
    Yc = -Yc         # flip vertical axis
    Zc = -Zc         # flip optical axis

    # Now Zc points forward in the pinhole sense.


    print("[DEBUG] Camera extrinsic (Vehicle→Camera):\n", T_cv)
    print("[DEBUG] Example LiDAR→Vehicle point:", points_vehicle[0])
    pts_vh = np.c_[points_vehicle, np.ones((points_vehicle.shape[0], 1), np.float32)]
    pts_c_test = (pts_vh @ T_cv.T)[:, :3]
    print("[DEBUG] Camera-space Z range:", float(pts_c_test[:,2].min()), "→", float(pts_c_test[:,2].max()))
    print("[DEBUG] Positive-Z ratio:", np.mean(pts_c_test[:,2] > 0))

    # ---- 4️⃣ Keep points in front of camera ----
    mask_front = Zc > 1e-3
    Xc, Yc, Zc = Xc[mask_front], Yc[mask_front], Zc[mask_front]

    # Flip camera Y axis to match Waymo's image convention (+Y down)
    Yc = -Yc

    # ---- 5️⃣ Perspective projection ----
    u = fx * (Xc / Zc) + cx
    v = fy * (Yc / Zc) + cy
    uv = np.stack([u, v], axis=-1)

    print("[DEBUG] u range:", u.min(), "→", u.max())
    print("[DEBUG] v range:", v.min(), "→", v.max())
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    print("[DEBUG] points inside image bounds:", np.count_nonzero(valid), "/", len(u))

    # ---- 6️⃣ Keep only pixels inside image bounds ----
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    uv_valid   = uv[valid]
    depth      = Zc[valid]
    final_mask = np.zeros(points_vehicle.shape[0], dtype=bool)
    valid_idx  = np.where(mask_front)[0][valid]
    final_mask[valid_idx] = True

    return uv_valid, depth, final_mask, width, height

def project_lidar_to_image_v2_old2(points_vehicle: np.ndarray, camera_row):
    fx = float(camera_row["[CameraCalibrationComponent].intrinsic.f_u"])
    fy = float(camera_row["[CameraCalibrationComponent].intrinsic.f_v"])
    cx = float(camera_row["[CameraCalibrationComponent].intrinsic.c_u"])
    cy = float(camera_row["[CameraCalibrationComponent].intrinsic.c_v"])
    width  = int(camera_row["[CameraCalibrationComponent].width"])
    height = int(camera_row["[CameraCalibrationComponent].height"])

    # Parquet stores Camera→Vehicle, invert it
    extr_vals = camera_row["[CameraCalibrationComponent].extrinsic.transform"]
    extr_vals = extr_vals.as_py() if hasattr(extr_vals, "as_py") else extr_vals
    T_vc = np.array(extr_vals, np.float32).reshape(4, 4, order="C")
    T_cv = np.linalg.inv(T_vc)

    #New test
    extr_C = np.array(extr_vals, np.float32).reshape(4, 4, order="C")
    extr_F = np.array(extr_vals, np.float32).reshape(4, 4, order="F")
    T_vc_C = extr_C
    T_vc_F = extr_F
    for tag, T_vc in [("C",T_vc_C),("F",T_vc_F)]:
        T_cv = np.linalg.inv(T_vc)
        pts_vh = np.c_[points_vehicle, np.ones((points_vehicle.shape[0],1),np.float32)]
        pts_c  = (pts_vh @ T_cv.T)[:,:3]
        Zc = pts_c[:,2]
        print(f"[{tag}] Z range {Zc.min():.2f}→{Zc.max():.2f}, median {np.median(Zc):.2f}")

    T_vc = T_vc_F
    T_cv = np.linalg.inv(T_vc)
    print("R part:\n", T_vc[:3,:3])
    print("T_vc (Camera→Vehicle):\n", T_vc)
    print("T_cv (Vehicle→Camera):\n", T_cv)

    pts_vh = np.c_[points_vehicle, np.ones((points_vehicle.shape[0], 1), np.float32)]
    # Vehicle → Camera
    pts_c  = (pts_vh @ T_cv.T)[:, :3]


    # ---- Fix camera handedness (Waymo camera axes) ----
    # Rotate 180° about X:  Y→−Y, Z→−Z
    pts_c[:, 1] *= -1.0
    pts_c[:, 2] *= -1.0

    Xc, Yc, Zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

    print("Zc stats:", np.min(Zc), np.median(Zc), np.max(Zc))

    mask_front = Zc > 0
    Xc, Yc, Zc = Xc[mask_front], Yc[mask_front], Zc[mask_front]
    u = fx * (Xc / Zc) + cx
    v = fy * (Yc / Zc) + cy
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    
    print("[DEBUG] u range:", u.min(), "→", u.max())
    print("[DEBUG] v range:", v.min(), "→", v.max())
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    print("[DEBUG] points inside image bounds:", np.count_nonzero(valid), "/", len(u))


    uv = np.stack([u[valid], v[valid]], axis=-1)
    depth = Zc[valid]
    mask = np.zeros(points_vehicle.shape[0], dtype=bool)
    valid_idx = np.where(mask_front)[0][valid]
    mask[valid_idx] = True
    return uv, depth, mask, width, height

import numpy as np

def _read_intrinsics(camera_row):
    fx = float(camera_row["[CameraCalibrationComponent].intrinsic.f_u"])
    fy = float(camera_row["[CameraCalibrationComponent].intrinsic.f_v"])
    cx = float(camera_row["[CameraCalibrationComponent].intrinsic.c_u"])
    cy = float(camera_row["[CameraCalibrationComponent].intrinsic.c_v"])
    width  = int(camera_row["[CameraCalibrationComponent].width"])
    height = int(camera_row["[CameraCalibrationComponent].height"])
    return fx, fy, cx, cy, width, height

def _to_numpy_4x4(vals):
    if hasattr(vals, "as_py"):
        vals = vals.as_py()
    return np.array(vals, np.float32)

def _fix_row_translation(T):
    """
    If translation is in the last ROW (tx,ty,tz,1),
    move it to the last COLUMN (as homogeneous expects).
    """
    T = T.copy()
    # Heuristic: if last column is ~[0,0,0,1] but last row has non-zero xyz
    if np.allclose(T[:3, 3], 0, atol=1e-7) and (np.linalg.norm(T[3, :3]) > 1e-6) and abs(T[3, 3] - 1.0) < 1e-6:
        T[:3, 3] = T[3, :3]
        T[3, :3] = 0.0
    return T

def _count_in_image(points_vehicle, T_cv, fx, fy, cx, cy, W, H, sample=20000):
    """Project a sample with Vehicle→Camera matrix; return (#inside, stats)."""
    N = len(points_vehicle)
    if N == 0:
        return 0, (0, 0, 0, 0, 0)
    idx = np.linspace(0, N-1, min(sample, N)).astype(int)
    pts_v = points_vehicle[idx]

    pts_vh = np.c_[pts_v, np.ones((pts_v.shape[0], 1), np.float32)]
    pts_c  = (pts_vh @ T_cv.T)[:, :3]

    Xc, Yc, Zc = pts_c[:,0], pts_c[:,1], pts_c[:,2]

    # Keep only points in front of the camera (Z forward)
    front = Zc > 1e-6
    if not np.any(front):
        return 0, (Zc.min(), np.median(Zc), Zc.max(), 0, 0)

    Xc, Yc, Zc = Xc[front], Yc[front], Zc[front]
    u = fx * (Xc / Zc) + cx
    v = fy * (Yc / Zc) + cy

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    n_in = int(np.count_nonzero(inside))
    stats = (float(np.min(Zc)), float(np.median(Zc)), float(np.max(Zc)),
             float(np.min(u)), float(np.max(u)))
    return n_in, stats

import numpy as np

def project_lidar_to_image_v2_old3(points_vehicle: np.ndarray, camera_row):
    """
    Project right-handed VEHICLE-frame LiDAR points onto the camera image plane.
    Handles Waymo Parquet exports where the 4×4 extrinsic may have translation
    stored in the last *row* instead of the last column, and may be C- or F-order.

    Args:
        points_vehicle : (N,3) LiDAR points in VEHICLE frame (+X fwd, +Y left, +Z up)
        camera_row     : one row from camera_calibration parquet

    Returns:
        uv      : (M,2) pixel coordinates (u,v)
        depth   : (M,)  depth values (Z in camera frame)
        mask    : (N,)  boolean mask of projected points
        width   : image width
        height  : image height
    """

    # ---------- 1️⃣  Intrinsics ----------
    fx = float(camera_row["[CameraCalibrationComponent].intrinsic.f_u"])
    fy = float(camera_row["[CameraCalibrationComponent].intrinsic.f_v"])
    cx = float(camera_row["[CameraCalibrationComponent].intrinsic.c_u"])
    cy = float(camera_row["[CameraCalibrationComponent].intrinsic.c_v"])
    width  = int(camera_row["[CameraCalibrationComponent].width"])
    height = int(camera_row["[CameraCalibrationComponent].height"])

    # ---------- 2️⃣  Read & repair extrinsic ----------
    extr_vals = camera_row["[CameraCalibrationComponent].extrinsic.transform"]
    if hasattr(extr_vals, "as_py"):
        extr_vals = extr_vals.as_py()
    vec = np.array(extr_vals, np.float32)

    # Two possible memory layouts
    T_vc_C = vec.reshape(4, 4, order="C")
    T_vc_F = vec.reshape(4, 4, order="F")

    def fix_translation(T):
        """If translation lives in last row, move to last column."""
        T = T.copy()
        if np.allclose(T[:3, 3], 0, atol=1e-7) and np.linalg.norm(T[3, :3]) > 1e-6:
            t = T[3, :3].copy()
            T[3, :3] = 0.0
            T[:3, 3] = t
            print(f"[FIX] moved translation row→column: {t}")
        return T

    T_vc_C = fix_translation(T_vc_C)
    T_vc_F = fix_translation(T_vc_F)

    # ---------- 3️⃣  Pick the layout that yields sane Z (0–100 m) ----------
    def z_stats(T_vc):
        T_cv = np.linalg.inv(T_vc)
        pts_vh = np.c_[points_vehicle, np.ones((points_vehicle.shape[0], 1), np.float32)]
        Z = (pts_vh @ T_cv.T)[:, 2]
        return np.median(Z), np.sum(Z > 0)

    medC, posC = z_stats(T_vc_C)
    medF, posF = z_stats(T_vc_F)
    if 0 < medF < 200 and posF > posC:
        T_vc = T_vc_F
        tag = "F"
    else:
        T_vc = T_vc_C
        tag = "C"

    # The 3x3 rotation block might be transposed
    R = T_vc[:3, :3]
    print("det(R) =", np.linalg.det(R))
    print("orthogonality error =", np.linalg.norm(R.T @ R - np.eye(3)))
    if np.linalg.norm(np.cross(R[:, 0], R[:, 1]) - R[:, 2]) > 0.01:
        # quick sanity: column vectors not orthogonal → transpose block
        print("[FIX] transposing rotation block")
        T_vc[:3, :3] = R.T

    T_cv = np.linalg.inv(T_vc)
    print(f"[CHOICE] using order={tag}, median Z={medF if tag=='F' else medC:.2f}")

    # ---------- 4️⃣  Transform points & project ----------
    pts_vh = np.c_[points_vehicle, np.ones((points_vehicle.shape[0], 1), np.float32)]
    pts_c  = (pts_vh @ T_cv.T)[:, :3]

    # ---- fix optical axis orientation ----
    pts_c[:, 1] *= -1.0     # Waymo cameras: +Y down in image
    pts_c[:, 2] *= -1.0     # flip Z so +Z points forward

    Xc, Yc, Zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

    # Keep only points in front of camera
    front = Zc > 1e-6
    if not np.any(front):
        print("[WARN] No LiDAR points with positive Z in camera frame.")
        return (np.zeros((0, 2), np.float32),
                np.zeros((0,), np.float32),
                np.zeros(points_vehicle.shape[0], dtype=bool),
                width, height)

    Xc, Yc, Zc = Xc[front], Yc[front], Zc[front]

    # Pinhole projection (Waymo camera convention already +Z forward)
    u = fx * (Xc / Zc) + cx
    v = fy * (Yc / Zc) + cy

    print("Zc stats:", Zc.min(), np.median(Zc), Zc.max())
    print("u range:", u.min(), "→", u.max())
    print("v range:", v.min(), "→", v.max())

    # ---------- 5️⃣  Clip to image bounds ----------
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    uv = np.stack([u[inside], v[inside]], axis=-1).astype(np.float32)
    depth = Zc[inside].astype(np.float32)

    mask = np.zeros(points_vehicle.shape[0], dtype=bool)
    idx_front = np.where(front)[0]
    mask[idx_front[inside]] = True

    print(f"[INFO] Projected {len(uv)} / {len(points_vehicle)} LiDAR points into image.")
    return uv, depth, mask, width, height

import numpy as np

def project_lidar_to_image_v2_old4(points_vehicle: np.ndarray, camera_row):
    """
    Project VEHICLE-frame LiDAR points (+X forward, +Y left, +Z up) to camera image.
    Robust against Parquet extrinsic quirks:
      • translation stored in last *row* → moved to last column
      • memory order ambiguity (C vs F)
    We pick the candidate that yields the most in-image points, scoring AFTER
    camera-frame fix (+X right, +Y down, +Z forward).
    Returns: uv [M,2], depth [M], mask [N], width, height
    """

    # ---- intrinsics
    fx = float(camera_row["[CameraCalibrationComponent].intrinsic.f_u"])
    fy = float(camera_row["[CameraCalibrationComponent].intrinsic.f_v"])
    cx = float(camera_row["[CameraCalibrationComponent].intrinsic.c_u"])
    cy = float(camera_row["[CameraCalibrationComponent].intrinsic.c_v"])
    width  = int(camera_row["[CameraCalibrationComponent].width"])
    height = int(camera_row["[CameraCalibrationComponent].height"])

    # ---- read extrinsic list
    extr_vals = camera_row["[CameraCalibrationComponent].extrinsic.transform"]
    if hasattr(extr_vals, "as_py"):
        extr_vals = extr_vals.as_py()
    vec16 = np.array(extr_vals, np.float32)

    # ---- row→col translation repair
    def fix_translation(T):
        T = T.copy()
        # If last column is ~[0,0,0,1] and last row has [tx,ty,tz,1], move it
        if np.allclose(T[:3, 3], 0, atol=1e-7) and abs(T[3, 3] - 1.0) < 1e-6 and np.linalg.norm(T[3, :3]) > 1e-9:
            t = T[3, :3].copy()
            T[3, :3] = 0.0
            T[:3, 3] = t
            print(f"[FIX] moved translation row→column: {t}")
        return T

    T_vc_C = fix_translation(vec16.reshape(4, 4, order="C"))
    T_vc_F = fix_translation(vec16.reshape(4, 4, order="F"))

    # ---- sanity: ensure rotation is orthonormal; if not, transpose its 3x3 block
    def make_rotation_ok(T):
        T = T.copy()
        R = T[:3, :3]
        ortho_err = np.linalg.norm(R.T @ R - np.eye(3))
        if ortho_err > 1e-3:
            # try transposing the 3x3 block
            R = R.T
            if np.linalg.norm(R.T @ R - np.eye(3)) < ortho_err:
                T[:3, :3] = R
                print("[FIX] transposed rotation block")
        return T

    T_vc_C = make_rotation_ok(T_vc_C)
    T_vc_F = make_rotation_ok(T_vc_F)

    # ---- scoring: project a sample AFTER applying camera-frame fix
    def score_candidate(T_vc, sample=20000, zmin=0.3):
        # invert: Camera→Vehicle -> Vehicle→Camera
        #T_cv = np.linalg.inv(T_vc)
        # wrong – you already have Vehicle→Camera
        # T_cv = np.linalg.inv(T_vc)

        # correct – use it directly
        T_cv = T_vc

        R = T_cv[:3,:3]
        print("det(R) =", np.linalg.det(R))
        print("orthogonality error =", np.linalg.norm(R.T @ R - np.eye(3)))

        N = len(points_vehicle)
        if N == 0:
            return (0, T_cv, (0,0,0,0,0))
        idx = np.linspace(0, N - 1, min(sample, N)).astype(int)
        pts_v = points_vehicle[idx]
        pts_vh = np.c_[pts_v, np.ones((pts_v.shape[0], 1), np.float32)]
        pts_c = (pts_vh @ T_cv.T)[:, :3]

        # Camera-frame fix: +X right, +Y down, +Z forward
        pts_c[:, 1] *= -1.0
        pts_c[:, 2] *= -1.0

        Xc, Yc, Zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
        front = Zc > zmin  # ignore near-zero depths (numerical blow-up)
        if not np.any(front):
            return (0, T_cv, (float(Zc.min()), float(np.median(Zc)), float(Zc.max()), 0.0, 0.0))

        Xc, Yc, Zc = Xc[front], Yc[front], Zc[front]
        u = fx * (Xc / Zc) + cx
        v = fy * (Yc / Zc) + cy
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        nin = int(np.count_nonzero(inside))

        stats = (float(Zc.min()), float(np.median(Zc)), float(Zc.max()),
                 float(np.min(u)), float(np.max(u)))
        return (nin, T_cv, stats)

    nin_C, T_cv_C, stats_C = score_candidate(T_vc_C)
    nin_F, T_cv_F, stats_F = score_candidate(T_vc_F)

    print("R_C:\n", T_vc_C[:3,:3])
    print("R_F:\n", T_vc_F[:3,:3])

    R = T_vc_F[:3,:3]  # or C
    yaw_deg = np.degrees(np.arctan2(R[1,0], R[0,0]))
    print("yaw F =", yaw_deg)
    R = T_vc_C[:3,:3]  # or C
    yaw_deg = np.degrees(np.arctan2(R[1,0], R[0,0]))
    print("yaw C =", yaw_deg)

    # choose best by in-image count, tie-breaker on median Z near ~15m
    def score_tuple(nin, stats):
        zmed = stats[1]
        return (nin, -abs(zmed - 15.0))

    choice = ("C", T_vc_C, T_cv_C, stats_C, nin_C)
    if score_tuple(nin_F, stats_F) > score_tuple(nin_C, stats_C):
        choice = ("F", T_vc_F, T_cv_F, stats_F, nin_F)

    tag, T_vc, T_cv, stats, nin = choice
    print(f"[CHOICE] using order={tag}  in-image(sample)={nin}  "
          f"Zc[min/med/max]={stats[0]:.2f}/{stats[1]:.2f}/{stats[2]:.2f}  "
          f"u[min/max]={stats[3]:.1f}/{stats[4]:.1f}")

    # ---- final projection (all points) with the chosen T_cv
    pts_vh = np.c_[points_vehicle, np.ones((points_vehicle.shape[0], 1), np.float32)]
    pts_c = (pts_vh @ T_cv.T)[:, :3]
    # camera-frame fix once
    # pts_c[:, 1] *= -1.0
    # pts_c[:, 2] *= -1.0

    pts_c[:, 0] *= -1.0   # X
    pts_c[:, 2] *= -1.0   # Z

    Xc, Yc, Zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
    front = Zc > 0.3
    if not np.any(front):
        return (np.zeros((0, 2), np.float32),
                np.zeros((0,), np.float32),
                np.zeros(points_vehicle.shape[0], dtype=bool),
                width, height)

    Xc, Yc, Zc = Xc[front], Yc[front], Zc[front]
    u = fx * (Xc / Zc) + cx
    v = fy * (Yc / Zc) + cy
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    uv    = np.stack([u[inside], v[inside]], axis=-1).astype(np.float32)
    depth = Zc[inside].astype(np.float32)

    mask = np.zeros(points_vehicle.shape[0], dtype=bool)
    idx_front = np.where(front)[0]
    mask[idx_front[inside]] = True

    print(f"[INFO] Projected {len(uv)} / {len(points_vehicle)} LiDAR points into image.")
    return uv, depth, mask, width, height

import numpy as np

def project_lidar_to_image_v2_old5(points_vehicle: np.ndarray, camera_row):
    """
    Project VEHICLE-frame LiDAR points (+X fwd, +Y left, +Z up)
    onto a Waymo camera image (right-handed pinhole convention).

    Handles the common Parquet quirks:
      • column-major flattening of the 4×4 extrinsic
      • translation stored in the last *row*
      • matrix already Vehicle→Camera  (no inversion needed)
      • camera optical axis requires 180° rotation about Y (X→-X, Z→-Z)

    Returns:
        uv      : (M, 2) pixel coordinates
        depth   : (M,)  depth (Z in camera frame)
        mask    : (N,)  bool mask of LiDAR points that project into the image
        width   : image width
        height  : image height
    """

    # ---------- 1️⃣  Camera intrinsics ----------
    fx = float(camera_row["[CameraCalibrationComponent].intrinsic.f_u"])
    fy = float(camera_row["[CameraCalibrationComponent].intrinsic.f_v"])
    cx = float(camera_row["[CameraCalibrationComponent].intrinsic.c_u"])
    cy = float(camera_row["[CameraCalibrationComponent].intrinsic.c_v"])
    width  = int(camera_row["[CameraCalibrationComponent].width"])
    height = int(camera_row["[CameraCalibrationComponent].height"])

    # ---------- 2️⃣  Read and repair extrinsic ----------
    # extr_vals = camera_row["[CameraCalibrationComponent].extrinsic.transform"]
    # if hasattr(extr_vals, "as_py"):
    #     extr_vals = extr_vals.as_py()
    # vec = np.array(extr_vals, np.float32)

    # # Column-major reshape (Waymo Parquet)
    # T_vc = vec.reshape(4, 4, order="F")
    # print("Translation column (m):", T_vc[:3,3])

    extr_vals = camera_row["[CameraCalibrationComponent].extrinsic.transform"]
    if hasattr(extr_vals, "as_py"):
        extr_vals = extr_vals.as_py()
    vals = np.array(extr_vals, np.float32)

    print("extrinsic list:", extr_vals)

    # Manually construct 4×4 (column-major, translation in last row)
    T_vc = np.eye(4, dtype=np.float32)
    T_vc[:3, :3] = vals[:9].reshape(3, 3, order="F")
    T_vc[:3, 3]  = vals[12:15]          # the last row’s first 3 entries
    print("Translation (m):", T_vc[:3,3])

    T_cv = T_vc
    pts_vh = np.c_[points_vehicle, np.ones((points_vehicle.shape[0],1),np.float32)]
    pts_c = (pts_vh @ T_cv.T)[:,:3]
    pts_c[:,[0,2]] *= -1

    print("R =\n", T_vc[:3,:3])
    print("t =", T_vc[:3,3])

    # Move translation from last row → last column if needed
    if np.allclose(T_vc[:3, 3], 0, atol=1e-7) and np.linalg.norm(T_vc[3, :3]) > 1e-6:
        t = T_vc[3, :3].copy()
        T_vc[3, :3] = 0
        T_vc[:3, 3] = t
        print(f"[FIX] moved translation row→column: {t}")

    # Sanity: rotation orthogonality
    R = T_vc[:3, :3]
    detR = np.linalg.det(R)
    ortho_err = np.linalg.norm(R.T @ R - np.eye(3))
    print(f"[DEBUG] det(R)={detR:.3f}, ortho_err={ortho_err:.2e}")

    # Already Vehicle→Camera
    T_cv = T_vc

    # ---------- 3️⃣  Transform points ----------
    pts_vh = np.c_[points_vehicle, np.ones((points_vehicle.shape[0], 1), np.float32)]
    pts_c  = (pts_vh @ T_cv.T)[:, :3]

    # Camera-frame fix: 180° about Y  (X→-X, Z→-Z)
    pts_c[:, 0] *= -1.0
    pts_c[:, 2] *= -1.0

    Xc, Yc, Zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

    # ---------- 4️⃣  Filter & project ----------
    mask_front = Zc > 0.3  # discard near-zero or behind-camera points
    if not np.any(mask_front):
        print("[WARN] No points with positive depth.")
        return (np.zeros((0, 2), np.float32),
                np.zeros((0,), np.float32),
                np.zeros(points_vehicle.shape[0], dtype=bool),
                width, height)

    Xc, Yc, Zc = Xc[mask_front], Yc[mask_front], Zc[mask_front]
    u = fx * (Xc / Zc) + cx
    v = fy * (Yc / Zc) + cy

    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    uv     = np.stack([u[inside], v[inside]], axis=-1).astype(np.float32)
    depth  = Zc[inside].astype(np.float32)

    mask = np.zeros(points_vehicle.shape[0], dtype=bool)
    idx_front = np.where(mask_front)[0]
    mask[idx_front[inside]] = True

    # ---------- 5️⃣  Diagnostics ----------
    print(f"Zc stats: {Zc.min():.2f}  {np.median(Zc):.2f}  {Zc.max():.2f}")
    print(f"u range: {u.min():.1f} → {u.max():.1f}")
    print(f"v range: {v.min():.1f} → {v.max():.1f}")
    print(f"[INFO] Projected {len(uv)} / {len(points_vehicle)} LiDAR points into image.")

    return uv, depth, mask, width, height


import numpy as np

def project_lidar_to_image_v2(points_vehicle: np.ndarray, camera_row, debug=False, depth_threshold=0.3):
    """
    Project VEHICLE-frame LiDAR points (+X fwd, +Y left, +Z up)
    onto a Waymo camera image with optimized performance and accuracy.

    Args:
        points_vehicle: (N, 3) or (N, 4) array of LiDAR points in vehicle frame
        camera_row: Waymo camera calibration row from parquet file
        debug: Whether to print debug information (default: False for performance)
        depth_threshold: Minimum depth to consider valid points
        
    Returns:
        uv: (M, 2) projected pixel coordinates [u, v]
        depth: (M,) depth values for projected points
        mask: (N,) boolean mask indicating which input points were projected
        width: Image width
        height: Image height

    Optimized for real-world LiDAR data projection with improved:
    - Coordinate system handling
    - Performance (reduced redundant calculations)
    - Robustness for edge cases
    - Memory efficiency
    """
    # -------- Extract intrinsics --------
    fx = float(camera_row["[CameraCalibrationComponent].intrinsic.f_u"])
    fy = float(camera_row["[CameraCalibrationComponent].intrinsic.f_v"])
    cx = float(camera_row["[CameraCalibrationComponent].intrinsic.c_u"])
    cy = float(camera_row["[CameraCalibrationComponent].intrinsic.c_v"])
    width  = int(camera_row["[CameraCalibrationComponent].width"])
    height = int(camera_row["[CameraCalibrationComponent].height"])
    
    # Validate and normalize intrinsics
    if fx > 10000 or fy > 10000:
        fx *= 0.001
        fy *= 0.001
        if debug:
            print(f"[FIX] Scaled large focal lengths: fx={fx:.1f}, fy={fy:.1f}")
    elif fx < 100 or fy < 100:
        fx *= 1000.0
        fy *= 1000.0
        if debug:
            print(f"[FIX] Scaled small focal lengths: fx={fx:.1f}, fy={fy:.1f}")
    
    if debug:
        print(f"[DEBUG] Camera: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}, {width}x{height}")

    # -------- Extract and process extrinsics --------
    extr_vals = camera_row["[CameraCalibrationComponent].extrinsic.transform"]
    extr_vals = extr_vals.as_py() if hasattr(extr_vals, "as_py") else extr_vals
    vals = np.array(extr_vals, dtype=np.float32)
    assert vals.size == 16, f"Expected 16 extrinsic values, got {vals.size}"

    # Use optimized matrix interpretation based on determinant and orthogonality
    R_row = np.array([[vals[0], vals[1], vals[2]],
                     [vals[4], vals[5], vals[6]],
                     [vals[8], vals[9], vals[10]]], dtype=np.float32)
    t_row = np.array([vals[3], vals[7], vals[11]], dtype=np.float32)
    
    R_col = np.array([[vals[0], vals[4], vals[8]],
                     [vals[1], vals[5], vals[9]],
                     [vals[2], vals[6], vals[10]]], dtype=np.float32)
    t_col = np.array([vals[12], vals[13], vals[14]], dtype=np.float32)
    
    # Quick selection based on rotation matrix quality
    det_row = np.linalg.det(R_row)
    det_col = np.linalg.det(R_col)
    ortho_err_row = np.linalg.norm(R_row.T @ R_row - np.eye(3))
    ortho_err_col = np.linalg.norm(R_col.T @ R_col - np.eye(3))
    
    if (abs(det_col - 1.0) < abs(det_row - 1.0) and ortho_err_col < ortho_err_row):
        R, t = R_col, t_col
        matrix_type = "column-major"
    else:
        R, t = R_row, t_row
        matrix_type = "row-major"
    
    # Intelligent translation unit correction
    t_magnitude = np.max(np.abs(t))
    if t_magnitude > 100:  # Millimeters
        t /= 1000.0
        if debug: print(f"[FIX] Scaled translation from mm to m: {t_magnitude:.1f} -> {np.max(np.abs(t)):.3f}")
    elif t_magnitude > 10:  # Centimeters
        t /= 100.0
        if debug: print(f"[FIX] Scaled translation from cm to m: {t_magnitude:.1f} -> {np.max(np.abs(t)):.3f}")
    elif t_magnitude < 0.1:  # Kilometers or incorrect
        t *= 1000.0
        if debug: print(f"[FIX] Scaled translation from km to m: {t_magnitude:.3f} -> {np.max(np.abs(t)):.1f}")
    
    if debug:
        print(f"[INFO] Using {matrix_type} matrix, det(R)={np.linalg.det(R):.3f}, |t|_max={np.max(np.abs(t)):.3f}")

    # Build transformation matrix
    T_cv = np.eye(4, dtype=np.float32)
    T_cv[:3, :3] = R
    T_cv[:3, 3] = t

    # -------- Process LiDAR points --------
    points_xyz = points_vehicle[:, :3] if points_vehicle.shape[1] == 4 else points_vehicle
    N = points_xyz.shape[0]
    
    if debug:
        print(f"[DEBUG] Processing {N} LiDAR points")
        print(f"  Vehicle frame ranges: X[{points_xyz[:, 0].min():.1f}, {points_xyz[:, 0].max():.1f}], "
              f"Y[{points_xyz[:, 1].min():.1f}, {points_xyz[:, 1].max():.1f}], "
              f"Z[{points_xyz[:, 2].min():.1f}, {points_xyz[:, 2].max():.1f}]")

    # -------- Optimized coordinate transformation --------
    # Test coordinate system interpretations efficiently
    pts_homogeneous = np.c_[points_xyz.astype(np.float32), np.ones((N, 1), dtype=np.float32)]
    
    # Direct transformation
    pts_c_direct = (pts_homogeneous @ T_cv.T)[:, :3]
    
    # Waymo-to-standard coordinate conversion
    points_converted = points_xyz.copy()
    points_converted[:, 0] = -points_xyz[:, 1]  # -Y -> X (left to right)
    points_converted[:, 1] = -points_xyz[:, 2]  # -Z -> Y (up to down)
    points_converted[:, 2] = points_xyz[:, 0]   # X -> Z (forward)
    
    pts_homogeneous_conv = np.c_[points_converted.astype(np.float32), np.ones((N, 1), dtype=np.float32)]
    pts_c_converted = (pts_homogeneous_conv @ T_cv.T)[:, :3]
    
    # Select best coordinate system based on positive depth ratio
    depth_direct = pts_c_direct[:, 2]
    depth_converted = pts_c_converted[:, 2]
    
    positive_ratio_direct = np.sum(depth_direct > 0) / len(depth_direct)
    positive_ratio_converted = np.sum(depth_converted > 0) / len(depth_converted)
    
    if positive_ratio_converted > positive_ratio_direct:
        pts_c = pts_c_converted
        coord_method = "converted"
        if debug:
            print(f"[INFO] Selected converted coordinate system (positive depth: {positive_ratio_converted:.2f} vs {positive_ratio_direct:.2f})")
    else:
        pts_c = pts_c_direct
        coord_method = "direct"
        if debug:
            print(f"[INFO] Selected direct coordinate system (positive depth: {positive_ratio_direct:.2f} vs {positive_ratio_converted:.2f})")

    # -------- Optimized pinhole projection --------
    # Handle depth coordinate system
    Zc = pts_c[:, 2]
    positive_ratio = np.sum(Zc > 0) / len(Zc)
    
    if positive_ratio < 0.5:  # Most depths are negative
        Zc = -Zc  # Flip Z-axis
        if debug: print(f"[FIX] Flipped Z-axis (positive ratio: {positive_ratio:.2f} -> {np.sum(Zc > 0) / len(Zc):.2f})")
    
    # Handle X-axis orientation
    Xc = pts_c[:, 0]
    Yc = pts_c[:, 1]
    
    # Test X-axis flip by checking projection bounds
    valid_depth_mask = Zc > 0.1
    if np.any(valid_depth_mask):
        u_test = fx * (Xc[valid_depth_mask] / Zc[valid_depth_mask]) + cx
        u_flipped_test = fx * (-Xc[valid_depth_mask] / Zc[valid_depth_mask]) + cx
        
        in_bounds_original = np.sum((u_test >= 0) & (u_test < width))
        in_bounds_flipped = np.sum((u_flipped_test >= 0) & (u_flipped_test < width))
        
        if in_bounds_flipped > in_bounds_original:
            Xc = -Xc
            if debug: print(f"[FIX] Flipped X-axis (in-bounds: {in_bounds_original} -> {in_bounds_flipped})")

    # -------- filter & project --------
    # Apply minimum detection range filter (LiDAR blind zone)
    min_detection_range = 1.0  # Most automotive LiDARs have ~1m minimum range
    
    # Filter points too close to the sensor (in vehicle frame)
    vehicle_distance = np.sqrt(points_xyz[:, 0]**2 + points_xyz[:, 1]**2 + points_xyz[:, 2]**2)
    range_valid = vehicle_distance >= min_detection_range
    
    if debug:
        print(f"[INFO] Range filter: {np.sum(range_valid)}/{N} points pass minimum range check")
    
    # Apply range filter to camera coordinates
    Xc_filtered = Xc[range_valid]
    Yc_filtered = Yc[range_valid]
    Zc_filtered = Zc[range_valid]
    
    # Use only positive depth points for projection
    front = (Zc_filtered > depth_threshold)
    
    if not np.any(front):
        if debug: print(f"[WARNING] No valid points for projection after filtering")
        return (np.zeros((0, 2), np.float32), np.zeros((0,), np.float32), 
                np.zeros(N, dtype=bool), width, height)

    # Get valid points for projection
    Xc_front, Yc_front, Zc_front = Xc_filtered[front], Yc_filtered[front], Zc_filtered[front]
    
    # Project to image coordinates with improved precision
    u = fx * (Xc_front / Zc_front) + cx
    v = fy * (Yc_front / Zc_front) + cy
    
    # Test Y-axis orientation for better results
    v_alt = fy * (-Yc_front / Zc_front) + cy
    
    # Choose better Y projection based on in-bounds ratio
    v_std_in_bounds = np.sum((v >= 0) & (v < height))
    v_alt_in_bounds = np.sum((v_alt >= 0) & (v_alt < height))
    
    if v_alt_in_bounds > v_std_in_bounds:
        v = v_alt
        if debug: print(f"[FIX] Using Y-flipped projection (in-bounds: {v_std_in_bounds} -> {v_alt_in_bounds})")
    
    # Filter points within image boundaries with margin for robustness
    margin = 1.0  # pixel margin to avoid edge effects
    inside = ((u >= margin) & (u < width - margin) & 
              (v >= margin) & (v < height - margin))

    uv = np.stack([u[inside], v[inside]], axis=-1).astype(np.float32)
    depth = Zc_front[inside].astype(np.float32)

    # Create mask for original point indices
    mask = np.zeros(N, dtype=bool)
    if len(uv) > 0:
        # Map back to original indices considering both filters
        valid_indices = np.where(range_valid)[0]
        front_indices = valid_indices[front]
        final_valid_indices = front_indices[inside]
        mask[final_valid_indices] = True

    if debug:
        print(f"[INFO] Final projection: {len(uv)}/{N} points ({100*len(uv)/N:.1f}%)")
        if len(uv) > 0:
            print(f"  Depth range: [{depth.min():.1f}, {depth.max():.1f}]m")
            print(f"  UV range: u[{uv[:, 0].min():.1f}, {uv[:, 0].max():.1f}], "
                  f"v[{uv[:, 1].min():.1f}, {uv[:, 1].max():.1f}]")

    return uv, depth, mask, width, height

import cv2
def main():

    # 先在 Vehicle 下检查：return_world=False
    ds = Waymo3DDataset("/data/Datasets/waymodata/", split="training", max_frames=40, return_world=False)
    lidar, target = ds[30]

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
    #Waymo’s vehicle convention (+X forward, +Y left, +Z up)
    #(X-red, Y-green, Z-blue)
    visualize_open3d(
        lidar,                      # Vehicle 下的点
        target["boxes_3d"],         # Vehicle 下的框
        labels=target["labels"],
        invert_yaw_for_open3d=False, # 先保持 False；若诊断建议 B，再切 True/或在绘制时用 -yaw
        save_ply_path="output/frame_0000.ply"
    )

    # ===== LiDAR → Image Projection =====
    # ===== LiDAR → Image Projection =====
    print("\n========== LIDAR → CAMERA PROJECTION ==========")

    # LiDAR points already in VEHICLE (right-handed) frame
    points_vehicle = lidar[:, :3].numpy()

    # Load camera calibration for the same segment
    fname = ds.frame_index[30][0]
    pf_cam = pq.ParquetFile(os.path.join(ds.root, ds.split, "camera_calibration", fname))
    df_cam = pf_cam.read_row_group(0).to_pandas()

    # Select FRONT camera (key.camera_name = 1)
    cam_row = df_cam[df_cam["key.camera_name"] == 1].iloc[0]

    # ---- Project LiDAR points to camera image ----
    uv, depth, mask, width, height = project_lidar_to_image_v2(points_vehicle, cam_row)
    print(f"[INFO] Projected {len(uv)} LiDAR points into FRONT camera image.")

    # ---- Load camera image ----
    pf_img = pq.ParquetFile(os.path.join(ds.root, ds.split, "camera_image", fname))
    df_img = pf_img.read_row_group(0).to_pandas()
    img_row = df_img[df_img["key.camera_name"] == 1].iloc[0]

    fx = float(cam_row["[CameraCalibrationComponent].intrinsic.f_u"])
    fy = float(cam_row["[CameraCalibrationComponent].intrinsic.f_v"])
    cx = float(cam_row["[CameraCalibrationComponent].intrinsic.c_u"])
    cy = float(cam_row["[CameraCalibrationComponent].intrinsic.c_v"])
    width  = int(cam_row["[CameraCalibrationComponent].width"])
    height = int(cam_row["[CameraCalibrationComponent].height"])
    print(f"[DEBUG] Intrinsics fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}, size=({width},{height})")

    # img_bytes = img_row["[CameraImageComponent].image"]
    # img = cv2.imdecode(np.frombuffer(img_bytes.as_py(), np.uint8), cv2.IMREAD_COLOR)
    img_data = img_row["[CameraImageComponent].image"]
    # Some parquet readers return a pyarrow.Scalar, some return plain bytes
    if hasattr(img_data, "as_py"):
        img_data = img_data.as_py()
    # Now img_data is guaranteed to be bytes
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    # ---- Draw projected LiDAR points ----
    for (u, v) in uv.astype(int):
        cv2.circle(img, (u, v), 1, (0, 255, 0), -1)

    #cv2.imshow("LiDAR → Camera (Front)", img)
    cv2.namedWindow("LiDAR → Camera (Front)", cv2.WINDOW_NORMAL)
    cv2.imshow("LiDAR → Camera (Front)", img)
    cv2.resizeWindow("LiDAR → Camera (Front)", 1280, 720)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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