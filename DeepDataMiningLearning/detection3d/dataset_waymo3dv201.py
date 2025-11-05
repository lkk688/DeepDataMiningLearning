import os
import io
import numpy as np
import torch
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Open3D Imports (for 3D visualization) ---
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


class Waymo3DDataset(Dataset):
    """
    Waymo Open Dataset v2.x → Unified 3D Multi-Frame LiDAR + Box + Image Dataset.

    Loads and fuses multiple LiDAR sweeps and provides all associated ground truth
    (3D boxes, 2D boxes, images) for the *current* frame.

    ✅ Coordinate conventions:
        Vehicle frame: +X forward, +Y left, +Z up  (Right-handed)
        World frame:   East-North-Up (ENU)
        Box heading:   CCW (right-handed yaw)

    ✅ LiDAR Output (N, 5):
        [X, Y, Z, Intensity, TimeDelta]
        - (X, Y, Z) are in the requested frame (Vehicle or World).
        - Intensity is normalized [0, 1].
        - TimeDelta is in seconds (0.0 for current, < 0.0 for past).
    """

    def __init__(self, root_dir, split="training", max_frames=None, 
                 return_world=False, num_sweeps=1):
        """
        Args:
            root_dir (str): Path to Waymo dataset root.
            split (str): "training" | "validation" | "testing"
            max_frames (int, optional): Limit number of frames (for debugging).
            return_world (bool): If True, transforms points and boxes into World frame.
                                 If False, transforms to current Vehicle frame.
            num_sweeps (int): Number of LiDAR frames to fuse. 1 = single frame.
        """
        self.root = root_dir
        self.split = split
        self.return_world = return_world
        self.num_sweeps = num_sweeps

        # --- NEW: Reflection matrix for LH -> RH conversion ---
        # This flips the Y-axis (and Y-axis-related rotations)
        # M = diag(1, -1, 1, 1)
        self.M_reflect = np.diag([1, -1, 1, 1]).astype(np.float32)

        if num_sweeps < 1:
            raise ValueError("num_sweeps must be 1 or greater.")

        # --- Dataset folder paths ---
        self.lidar_dir = os.path.join(root_dir, split, "lidar")
        self.calib_dir = os.path.join(root_dir, split, "lidar_calibration")
        self.box_dir = os.path.join(root_dir, split, "lidar_box")
        self.vpose_dir = os.path.join(root_dir, split, "vehicle_pose")
        self.image_dir = os.path.join(root_dir, split, "camera_image")
        self.box_2d_dir = os.path.join(root_dir, split, "camera_box")
        # --- NEW: Add this path ---
        self.cam_calib_dir = os.path.join(root_dir, split, "camera_calibration")

        # --- NEW: Add self.cam_calib_dir to the check ---
        check_dirs = [self.lidar_dir, self.calib_dir, self.box_dir, 
                      self.vpose_dir, self.image_dir, self.cam_calib_dir]

        if split != "testing":
             check_dirs.append(self.box_2d_dir) # 2D boxes may not exist for test
        
        for p in check_dirs:
            if not os.path.isdir(p):
                raise FileNotFoundError(f"Required directory not found: {p}")
        if not os.path.isdir(self.box_2d_dir) and split != "testing":
             print(f"[WARN] 2D box directory not found: {self.box_2d_dir}")
             self.box_2d_dir = None


        # --- Index all lidar frames ---
        files = [f for f in os.listdir(self.lidar_dir) if f.endswith(".parquet")]
        # Ensure corresponding image and 3d box files exist
        valid = [f for f in files if 
                 os.path.exists(os.path.join(self.box_dir, f)) and 
                 os.path.exists(os.path.join(self.image_dir, f)) and
                 os.path.exists(os.path.join(self.vpose_dir, f)) and
                 os.path.exists(os.path.join(self.calib_dir, f))
                 ]
        
        self.frame_index = []
        total = 0
        for fname in sorted(valid): # Sort files to ensure segments are contiguous
            pf = pq.ParquetFile(os.path.join(self.lidar_dir, fname))
            df = pf.read_row_group(0, columns=["key.segment_context_name", "key.frame_timestamp_micros"]).to_pandas()
            # df (256,2) Sort by timestamp to ensure chronological order within segment
            df = df.sort_values(by="key.frame_timestamp_micros")
            for seg, ts in zip(df["key.segment_context_name"], df["key.frame_timestamp_micros"]):
                self.frame_index.append((fname, int(ts), seg)) # saves (fname, ts, seg)
                total += 1
                if max_frames and total >= max_frames:
                    break
            if max_frames and total >= max_frames:
                break

        # --- Class and Camera Maps ---
        # 3D Labels: 0: UNDEFINED, 1: VEHICLE, 2: PEDESTRIAN, 3: SIGN, 4: CYCLIST
        self.label_map_3d = { 1: "Vehicle", 2: "Pedestrian", 3: "Sign", 4: "Cyclist" }
        # 2D Labels: 1: TYPE_VEHICLE, 2: TYPE_PEDESTRIAN, 3: TYPE_CYCLIST, 4: TYPE_OTHER
        self.label_map_2d = { 1: "Vehicle", 2: "Pedestrian", 3: "Cyclist", 4: "Other" }
        # Camera Names: 1: FRONT, 2: FRONT_LEFT, 3: FRONT_RIGHT, 4: SIDE_LEFT, 5: SIDE_RIGHT
        self.camera_map = { 1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT" }

        print(f"✅ Waymo3DDataset initialized with {len(self.frame_index)} frames | "
              f"return_world={self.return_world} | num_sweeps={self.num_sweeps}")

    # ---------------------------------------------------------------------
    @staticmethod
    def _decode_range_image(row, return_id=1):
        """
        Decode one range image (Parquet: flattened float list). Row is one lidar (7,)
        Returns a numpy array [H,W,C] (float32).
        """
        kv = f"[LiDARComponent].range_image_return{return_id}.values"
        ks = f"[LiDARComponent].range_image_return{return_id}.shape"
        vals = row[kv].as_py() if hasattr(row[kv], "as_py") else row[kv] # (678400,)
        shp = row[ks].as_py() if hasattr(row[ks], "as_py") else row[ks] #(3,) [  64, 2650,    4]
        arr = np.array(vals, np.float32).reshape(shp, order="C")  # Waymo = row-major
        return arr #(64, 2650, 4)

    # ---------------------------------------------------------------------
    # ---------- NEW: PUBLIC __getitem__ (SWEEP MANAGER) ----------
    # ---------------------------------------------------------------------

    def __getitem__(self, idx):
        """
        Loads the current frame and (if num_sweeps > 1) fuses past frames.
        
        Returns:
            lidar (torch.Tensor): (N, 5) [X, Y, Z, Intensity, TimeDelta]
            target (dict): Ground truth and metadata for the *current* frame.
                'boxes_3d' (torch.Tensor): (M, 7)
                'labels' (torch.Tensor): (M,)
                'surround_views' (list): List of dicts for 5 cameras.
                'T_vl_sweeps' (list): List of (num_sweeps) T_vl matrices.
                'T_wv_sweeps' (list): List of (num_sweeps) T_wv matrices.
                'timestamps_sweeps' (list): List of (num_sweeps) timestamps.
                ... (other metadata)
        """
        
        # --- 1. Find all valid frame indices for this sweep ---
        # `indices_to_load` will be in chronological order [oldest, ..., current]
        indices_to_load = []
        current_fname, current_ts, current_seg = self.frame_index[idx]
        
        for k in range(self.num_sweeps):
            sweep_idx = idx - k
            # Check for start of dataset or segment boundary
            if sweep_idx < 0 or self.frame_index[sweep_idx][2] != current_seg:
                # Reached boundary. Pad by repeating the earliest valid frame.
                pad_idx = indices_to_load[0] if indices_to_load else idx
                indices_to_load = [pad_idx] * (self.num_sweeps - len(indices_to_load)) + indices_to_load
                break
            
            indices_to_load.insert(0, sweep_idx) # Insert at front
        
        # Handle case where idx=0 and num_sweeps > 1
        if not indices_to_load:
             indices_to_load = [idx] * self.num_sweeps

        # --- 2. Load data for all frames in the sweep ---
        """
        # Data format for each frame:
        # lidar: torch.Tensor of shape (N, 4) containing [x, y, z, intensity] in vehicle frame
        # target: dict containing the following keys:
        #   - "boxes_3d": torch.Tensor of shape (M, 7) [x, y, z, dx, dy, dz, yaw] in vehicle/world frame
        #   - "labels": torch.Tensor of shape (M,) containing object class IDs (1=Vehicle, 2=Pedestrian, 3=Sign, 4=Cyclist)
        #   - "segment": str, segment context name from Waymo dataset
        #   - "timestamp": int, frame timestamp in microseconds
        #   - "laser_id": int, LiDAR sensor ID (1=TOP, 2=FRONT, 3=SIDE_LEFT, 4=SIDE_RIGHT, 5=REAR)
        #   - "T_vl_current": torch.Tensor of shape (4, 4) transformation matrix from LiDAR to vehicle frame
        #   - "world_from_vehicle": torch.Tensor of shape (4, 4) transformation matrix from vehicle to world frame
        #   - "surround_views": list of dicts, each containing camera data (see format below)
        
        # Surround view data format for each camera:
            # Each dict in surround_views list contains:
            #   - "image": torch.Tensor of shape (H, W, 3) containing RGB image data (uint8)
            #   - "boxes_2d": torch.Tensor of shape (K, 4) containing 2D bounding boxes [x_min, y_min, x_max, y_max]
            #   - "labels_2d": torch.Tensor of shape (K,) containing 2D object class IDs matching 3D labels
            #   - "camera_name": int, camera sensor ID (1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT)
        """
        all_lidar_data = []
        all_target_data = []
        for i in indices_to_load:
            lidar, target = self._get_frame(i)
            all_lidar_data.append(lidar)
            all_target_data.append(target)
            
        # --- 3. Get current frame data (the anchor) ---
        # The last item in the list is the current frame (idx)
        current_lidar = all_lidar_data[-1]
        current_target = all_target_data[-1]
        
        # Get current pose (world_from_vehicle)
        # T_wv_current is guaranteed to exist by _get_frame
        T_wv_current = current_target["world_from_vehicle"]
        T_current_from_world = torch.inverse(T_wv_current)

        # --- 4. Fuse point clouds ---
        # Add time delta 0.0 to current frame
        time_col_current = torch.zeros_like(current_lidar[..., :1])
        fused_points = [torch.cat([current_lidar, time_col_current], dim=-1)]
        
        # Initialize the "additional info" lists
        T_vl_sweeps = [current_target.pop("T_vl_current")]
        T_wv_sweeps = [current_target["world_from_vehicle"]]
        timestamps_sweeps = [current_target["timestamp"]]

        # Loop through past frames (all *except* the last one)
        for i in range(self.num_sweeps - 1):
            sweep_lidar = all_lidar_data[i]
            sweep_target = all_target_data[i]
            T_wv_sweep = sweep_target["world_from_vehicle"]
            
            # Points from _get_frame are either in Vehicle or World frame
            if self.return_world:
                # Lidar is already in WORLD frame. Transform to CURRENT VEHICLE frame.
                pts_h = torch.cat([sweep_lidar[..., :3], torch.ones_like(sweep_lidar[..., :1])], dim=-1)
                pts_current_v = (pts_h.float() @ T_current_from_world.T)[..., :3]
            else:
                # Lidar is in SWEEP VEHICLE frame. Transform to WORLD, then to CURRENT VEHICLE.
                T_current_from_sweep_vehicle = T_current_from_world @ T_wv_sweep.float()
                pts_h = torch.cat([sweep_lidar[..., :3], torch.ones_like(sweep_lidar[..., :1])], dim=-1)
                pts_current_v = (pts_h.float() @ T_current_from_sweep_vehicle.T)[..., :3]

            # Add time delta
            time_delta_sec = (current_target['timestamp'] - sweep_target['timestamp']) / 1e6
            time_col = torch.full_like(sweep_lidar[..., :1], -time_delta_sec) # Past is negative

            # Concat [X, Y, Z, Intensity, TimeDelta]
            fused_points.append(torch.cat([pts_current_v, sweep_lidar[..., 3:4], time_col], dim=-1))

            # Store additional info
            T_vl_sweeps.append(sweep_target.pop("T_vl_current"))
            T_wv_sweeps.append(sweep_target["world_from_vehicle"])
            timestamps_sweeps.append(sweep_target["timestamp"])

        final_lidar = torch.cat(fused_points, dim=0)

        # --- 5. Add additional info to final target ---
        current_target["T_vl_sweeps"] = T_vl_sweeps
        current_target["T_wv_sweeps"] = T_wv_sweeps
        current_target["timestamps_sweeps"] = timestamps_sweeps

        # Pop transient key, no longer needed
        if "world_from_vehicle" in current_target and not self.return_world:
             # We return it if return_world=True, but pop it otherwise
             # (It's already saved in T_wv_sweeps)
             current_target.pop("world_from_vehicle")


        # --- 6. Debug summary for the ANCHOR frame ---
        if idx == 0 and self.num_sweeps > 1:
            print(f"[DEBUG] Multi-sweep frame {idx}: {len(final_lidar)} pts fused from {self.num_sweeps} sweeps | "
                  f"Time [{final_lidar[:,4].min():.2f}s, {final_lidar[:,4].max():.2f}s] | "
                  f"boxes={len(current_target['boxes_3d'])}")

        # --- 7. FINAL OUTPUT FORMAT ---
        """
        RETURN VALUES:
        
        final_lidar: torch.Tensor of shape (N, 5)
            - Multi-sweep fused point cloud in current vehicle frame
            - Columns: [X, Y, Z, Intensity, TimeDelta]
            - X, Y, Z: 3D coordinates in meters (current vehicle frame)
            - Intensity: LiDAR reflection intensity (0.0-1.0)
            - TimeDelta: Time offset in seconds (0.0 for current frame, negative for past frames)
            
        current_target: dict containing ground truth and metadata for the CURRENT frame
            Core 3D Detection Data:
            - "boxes_3d": torch.Tensor (M, 7) [x, y, z, dx, dy, dz, yaw] in vehicle/world frame
            - "labels": torch.Tensor (M,) object class IDs (1=Vehicle, 2=Pedestrian, 3=Sign, 4=Cyclist)
            
            Frame Metadata:
            - "segment": str, Waymo segment context name
            - "timestamp": int, frame timestamp in microseconds
            - "laser_id": int, LiDAR sensor ID (1=TOP, 2=FRONT, 3=SIDE_LEFT, 4=SIDE_RIGHT, 5=REAR)
            
            Multi-Sweep Fusion Data:
            - "T_vl_sweeps": list of (num_sweeps) torch.Tensor (4, 4) LiDAR-to-vehicle transforms
            - "T_wv_sweeps": list of (num_sweeps) torch.Tensor (4, 4) vehicle-to-world transforms  
            - "timestamps_sweeps": list of (num_sweeps) int timestamps in microseconds
            
            Camera Data:
            - "surround_views": list of 5 camera dicts, each containing:
                * "image": torch.Tensor (H, W, 3) RGB image data (uint8)
                * "boxes_2d": torch.Tensor (K, 4) 2D bounding boxes [x_min, y_min, x_max, y_max]
                * "labels_2d": torch.Tensor (K,) 2D object class IDs matching 3D labels
                * "camera_name": int, camera sensor ID (1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT)
            
            Optional (if return_world=True):
            - "world_from_vehicle": torch.Tensor (4, 4) vehicle-to-world transformation matrix
        """
        return final_lidar, current_target


    # ---------------------------------------------------------------------
    # ---------- NEW: PRIVATE _get_frame (OLD __getitem__) ----------
    # ---------------------------------------------------------------------
    
    def _get_frame(self, idx):
        """
        Loads all data for a SINGLE frame.
        Convert the range image into a 3D point cloud.
        This was the original __getitem__ method.
        Spherical (Range Image): The native format from the LiDAR sensor (range, azimuth, inclination).
        LiDAR Frame ($l$): A 3D Cartesian system (X, Y, Z) relative to the specific LiDAR sensor.
        
        Returns:
            lidar (torch.Tensor): (N, 4) [X, Y, Z, Intensity] in Vehicle or World
            target (dict): All metadata, boxes, and images for this frame.
        """
        fname, ts, seg = self.frame_index[idx]

        # ---------- 1️⃣ Read all LiDAR rows for this frame ----------
        pf = pq.ParquetFile(os.path.join(self.lidar_dir, fname))
        df = pf.read_row_group(0).to_pandas() #(256, 7)
        rows_t = df[(df["key.segment_context_name"] == seg) &
                    (df["key.frame_timestamp_micros"] == ts)] #(5, 7)
        if len(rows_t) == 0:
            raise RuntimeError(f"No LiDAR rows for this frame: {seg}/{ts}")

        # ---------- 2️⃣ Identify the TOP LiDAR (highest mount) ----------
        pf_cal = pq.ParquetFile(os.path.join(self.calib_dir, fname))
        df_cal = pf_cal.read_row_group(0).to_pandas() #(5, 6)

        def get_T_vl(lid):
            """Robustly fetch LiDAR→Vehicle extrinsic, ensuring correct ID match."""
            lid_int = int(lid)
            key_lids = df_cal["key.laser_name"].astype(np.int32)
            sel = df_cal[(df_cal["key.segment_context_name"] == seg) & (key_lids == lid_int)]
            if len(sel) == 0:
                raise RuntimeError(f"No calibration row found for seg={seg}, laser_name={lid_int}.")
            crow = sel.iloc[0]

            extr_cols = [c for c in crow.index if "extrinsic" in str(c)]
            # --- FIX: Use min(key=len) to find the *shortest* extrinsic name ---
            # This avoids nested/alternative component names
            extr_col = min(extr_cols, key=len) 
            extr_vals = crow[extr_col]
            extr_vals = extr_vals.as_py() if hasattr(extr_vals, "as_py") else extr_vals
            #return np.array(extr_vals, np.float32).reshape(4, 4, order="C")

            T_vl_lh = np.array(extr_vals, np.float32).reshape(4, 4, order="C")
            
            # --- FIX: Convert loaded LH matrix to RH ---
            T_vl_rh = self.M_reflect @ T_vl_lh @ self.M_reflect
            return T_vl_rh

        # Pick LiDAR with largest Z translation (the TOP sensor)
        cand_ids = sorted(rows_t["key.laser_name"].unique().tolist()) #5 lidars: [1, 2, 3, 4, 5]
        heights = {}
        for lid in cand_ids:
            heights[lid] = float(get_T_vl(lid)[2, 3])
        laser_id = max(heights, key=heights.get) #1 is highest
        row = rows_t[rows_t["key.laser_name"] == laser_id].iloc[0] #get TOP lidar (7,)
        T_vl = get_T_vl(laser_id) # (4, 4) LiDAR -> Vehicle
        
        # --- FIX: Add calibration yaw correction (from "working" code) ---
        R_vl = T_vl[:3, :3].copy()
        t_vl = T_vl[:3, 3].copy()
        yaw_deg = float(np.degrees(np.arctan2(R_vl[1, 0], R_vl[0, 0])))

        # Check for and correct bad yaw values
        if T_vl[2, 3] > 1.8 and abs(yaw_deg) > 30.0:
            print(f"[CALIB_FIX] Detected bad yaw ({yaw_deg:.1f}°) for TOP LiDAR; correcting.")
            yaw_rad = np.deg2rad(yaw_deg)
            cy, sy = np.cos(-yaw_rad), np.sin(-yaw_rad)
            Rz_fix = np.array([[cy, -sy, 0.0], [sy,  cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            # Apply on the RIGHT: R_corrected = R_original * Rz(-yaw)
            R_vl = R_vl @ Rz_fix
            T_vl[:3, :3] = R_vl
            # T_vl[:3, 3] = t_vl # Translation is unchanged
        
        # ---------- 3️⃣ Decode the range image ----------
        ri = self._decode_range_image(row, return_id=1) #(64, 2650, 4)
        rng = np.clip(np.nan_to_num(ri[..., 0], nan=0.0), 0.0, 300.0) #(64, 2650)
        inten = ri[..., 1] if ri.shape[-1] > 1 else np.zeros_like(rng)
        H, W = rng.shape #64, 2650

        # ---------- 4️⃣ Beam inclinations ----------
        crow = df_cal[(df_cal["key.segment_context_name"] == seg) &
                      (df_cal["key.laser_name"] == laser_id)].iloc[0]
        """
        The Waymo LiDAR sensor has non-uniform beam spacing. Using linspace creates a warped or "curved" point cloud, making alignment with the straight-edged boxes impossible.
        The new code (in Step 4) fixes this by loading the actual inclination vector from the calibration file, which provides the precise angle for each beam:
        """
        # inc_min = float(crow["[LiDARCalibrationComponent].beam_inclination.min"])
        # inc_max = float(crow["[LiDARCalibrationComponent].beam_inclination.max"])
        # inclinations = np.linspace(inc_min, inc_max, H, dtype=np.float32) #(64,)
        # if np.max(np.abs(inclinations)) > np.pi:
        #     inclinations = np.deg2rad(inclinations)
        # --- FIX: Load the full non-uniform inclination vector ---
        cand_cols = [
            "[LiDARCalibrationComponent].beam_inclinations.values",
            "[LiDARCalibrationComponent].beam_inclination.values",
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
            # Fallback: uniform spacing
            inc_min = float(crow["[LiDARCalibrationComponent].beam_inclination.min"])
            inc_max = float(crow["[LiDARCalibrationComponent].beam_inclination.max"])
            inclinations = np.linspace(inc_min, inc_max, H, dtype=np.float32)

        if np.max(np.abs(inclinations)) > np.pi:
            inclinations = np.deg2rad(inclinations)
            
        if len(inclinations) != H:
            # Resample if H doesn't match inclinations vector length
            inclinations = np.interp(
                np.linspace(0, len(inclinations)-1, H),
                np.arange(len(inclinations)),
                inclinations
            ).astype(np.float32)

        # ---------- 5️⃣ Convert Spherical → LiDAR Cartesian ----------
        incl = inclinations[::-1].reshape(H, 1)
        az = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)
        cos_i, sin_i = np.cos(incl), np.sin(incl)
        cos_a, sin_a = np.cos(az), np.sin(az)

        Xl = rng * cos_i * cos_a
        Yl = rng * cos_i * sin_a  # <-- This is +Y, creating a standard RH system
        Zl = rng * sin_i
        
        # pts_l is now RIGHT-HANDED
        pts_l = np.stack([Xl, Yl, Zl, np.ones_like(Zl)], axis=-1).reshape(-1, 4)

        self.visualize_range_image_raw(rng, inclinations)

        # ---------- 6️⃣ LiDAR → Vehicle (apply extrinsic) ----------
        r"""
        applies the sensor's extrinsic calibration
        Let $T_{v \leftarrow l}$ be the 4x4 extrinsic matrix T_vl: l to v (LiDAR Frame $\to$ Vehicle Frame (Points))
        $$ \\ P\_v = T\_{v \leftarrow l} \cdot P\_l$$
        The code uses (pts @ T.T), which is the row-vector equivalent of 
        the standard column-vector math $P_v = T_{v \leftarrow l} \cdot P_l$.
        The range image in Waymo can be RAW or virtual, here we use RAW
        """
        pts_v = (pts_l @ T_vl.T)[:, :3] #(169600, 3)
        xyz_vehicle = np.nan_to_num(pts_v, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- 7️⃣ Load Vehicle Pose (ALWAYS) & Optionally Transform ----------
        # We MUST load T_wv for multi-frame fusion, even if return_world=False
        try:
            pf_vp = pq.ParquetFile(os.path.join(self.vpose_dir, fname))
            df_vp = pf_vp.read_row_group(0).to_pandas()
            vrow = df_vp[(df_vp["key.segment_context_name"] == seg) &
                         (df_vp["key.frame_timestamp_micros"] == ts)].iloc[0] #(3,)
            vp_vals = vrow["[VehiclePoseComponent].world_from_vehicle.transform"]
            vp_vals = vp_vals.as_py() if hasattr(vp_vals, "as_py") else vp_vals #(16,)
            T_wv = np.array(vp_vals, np.float32).reshape(4, 4, order="C") #(4, 4)

            T_wv_lh = np.array(vp_vals, np.float32).reshape(4, 4, order="C")
            
            # --- FIX: Convert loaded LH matrix to RH ---
            T_wv = self.M_reflect @ T_wv_lh @ self.M_reflect
        except Exception as e:
            print(f"[WARN] Failed to load vehicle pose for {seg}/{ts}: {e}")
            T_wv = np.eye(4, dtype=np.float32) # Identity fallback

        r"""
        This (optionally) applies the vehicle's global pose
        T_wv is the (4, 4) Vehicle->World matrix
        Let $T_{w \leftarrow v}$ be the 4x4 vehicle pose matrix T_wv.
        $$ \\ P\_w = T\_{w \leftarrow v} \cdot P\_v$$
        """
        if self.return_world:
            pts_vh = np.concatenate([xyz_vehicle, np.ones((xyz_vehicle.shape[0], 1), np.float32)], axis=1)
            xyz_world = (pts_vh @ T_wv.T)[:, :3]
            XYZ = xyz_world # Points are in World Frame
        else:
            XYZ = xyz_vehicle # Points are in Vehicle Frame
        #XYZ: (169600, 3)
        # ---------- 8️⃣ Normalize intensity ----------
        inten = inten.reshape(-1, 1).astype(np.float32) #(169600, 1)
        inten = inten / (inten.max() + 1e-6)
        lidar = torch.tensor(np.concatenate([XYZ, inten], axis=1), dtype=torch.float32) #[169600, 4]

        # ---------- 9️⃣ Load and Transform 3D Boxes ----------
        # pf_box = pq.ParquetFile(os.path.join(self.box_dir, fname))
        # df_box = pf_box.read_row_group(0).to_pandas() #(7013, 21)
        # rows_b = df_box[(df_box["key.segment_context_name"] == seg) &
        #                 (df_box["key.frame_timestamp_micros"] == ts) &
        #                 (df_box["key.laser_name"] == laser_id)]
        
        #The lidar_box component only provides annotations for the TOP LiDAR sensor
        #the key.laser_name column is considered redundant by Waymo and is omitted from the parquet file.
        
        # ---------- 9️⃣ Load and Transform 3D Boxes ----------
        """
        3D Box Heading Transformation (LiDAR $\to$ Vehicle)
        The heading $\theta$ (yaw) must be correctly rotated by the 3D extrinsic matrix, 
        accounting for the sensor's roll and pitch. 
        This is not a simple addition of yaw angles.
        The code correctly calculates the new yaw by finding the yaw component of the combined 3x3 rotation matrix.
        """
        pf_box = pq.ParquetFile(os.path.join(self.box_dir, fname))
        df_box = pf_box.read_row_group(0).to_pandas()
        
        # [DEBUG] You can add this line to see the 21 columns you have:
        #print(df_box.columns.to_list()) 
        #['key.segment_context_name', 'key.frame_timestamp_micros', 'key.laser_object_id', '[LiDARBoxComponent].box.center.x', '[LiDARBoxComponent].box.center.y', '[LiDARBoxComponent].box.center.z', '[LiDARBoxComponent].box.size.x', '[LiDARBoxComponent].box.size.y', '[LiDARBoxComponent].box.size.z', '[LiDARBoxComponent].box.heading', '[LiDARBoxComponent].type', '[LiDARBoxComponent].num_lidar_points_in_box', '[LiDARBoxComponent].num_top_lidar_points_in_box', '[LiDARBoxComponent].speed.x', '[LiDARBoxComponent].speed.y', '[LiDARBoxComponent].speed.z', '[LiDARBoxComponent].acceleration.x', '[LiDARBoxComponent].acceleration.y', '[LiDARBoxComponent].acceleration.z', '[LiDARBoxComponent].difficulty_level.detection', '[LiDARBoxComponent].difficulty_level.tracking']
        
        # --- FIX: Remove the 'key.laser_name' filter ---
        rows_b = df_box[(df_box["key.segment_context_name"] == seg) &
                        (df_box["key.frame_timestamp_micros"] == ts)]

        if len(rows_b) == 0:
            boxes_any = torch.zeros((0, 7), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            f = ["[LiDARBoxComponent].box.center.x", "[LiDARBoxComponent].box.center.y",
                 "[LiDARBoxComponent].box.center.z", "[LiDARBoxComponent].box.size.x",
                 "[LiDARBoxComponent].box.size.y", "[LiDARBoxComponent].box.size.z",
                 "[LiDARBoxComponent].box.heading"]
            arr = rows_b[f].to_numpy().astype(np.float32)
            # --- FIX: Boxes are in VEHICLE frame, but flipped handedness ---
            #arr[:, 1] *= -1.0        # y -> -y
            #arr[:, 6] *= -1.0        # heading -> -heading

            # --- FIX: Convert LH box params to RH box params ---
            # 1. Flip the Y-center coordinate
            arr[:, 1] *= -1.0        # cy -> -cy
            # 2. Flip the heading
            # (LH clockwise -> LH ccw) & (LH ccw -> RH ccw)
            arr[:, 6] *= -1.0
            
            centers_v = arr[:, :3]
            sizes = arr[:, 3:6]
            headings_v = arr[:, 6]

            boxes_vehicle = np.concatenate([centers_v, sizes, headings_v[:, None]], axis=1)

            # --- 3. Optionally: Vehicle -> World ---
            if self.return_world:
                R_wv = T_wv[:3, :3]
                
                # Transform centers
                centers_v_h = np.concatenate([centers_v, np.ones((centers_v.shape[0], 1), np.float32)], axis=1)
                centers_w = (centers_v_h @ T_wv.T)[:, :3]

                # Transform headings (using the correct 3D method)
                cos_h_v, sin_h_v = np.cos(headings_v), np.sin(headings_v)
                R_wb_00 = R_wv[0, 0] * cos_h_v + R_wv[0, 1] * sin_h_v
                R_wb_10 = R_wv[1, 0] * cos_h_v + R_wv[1, 1] * sin_h_v
                headings_w = np.arctan2(R_wb_10, R_wb_00)
                
                boxes_any = torch.tensor(np.concatenate([centers_w, sizes, headings_w[:, None]], axis=1),
                                         dtype=torch.float32)
            else:
                boxes_any = torch.tensor(boxes_vehicle, dtype=torch.float32)
                
            # --- 4. Output corners and labels ---
            #boxes_any = torch.tensor(corners_any, dtype=torch.float32) 
            labels = torch.tensor(rows_b["[LiDARBoxComponent].type"].to_numpy(), dtype=torch.int64)

        # ---------- 🔟 Assemble final output dict for this frame ----------
        target = {
            "boxes_3d": boxes_any,        #(N, 7) In Vehicle or World frame
            "labels": labels,
            "segment": seg,
            "timestamp": ts,
            "laser_id": int(laser_id),
            # --- Add poses for fusion ---
            "T_vl_current": torch.tensor(T_vl, dtype=torch.float32),
            "world_from_vehicle": torch.tensor(T_wv, dtype=torch.float32),
        }

        # ---------- 1️⃣1️⃣ Load Surround Images & 2D Boxes (and Camera Calibration) ----------
        pf_img = pq.ParquetFile(os.path.join(self.image_dir, fname))
        df_img = pf_img.read_row_group(0).to_pandas()
        rows_img = df_img[(df_img["key.segment_context_name"] == seg) &
                          (df_img["key.frame_timestamp_micros"] == ts)]
        
        # --- FIX: Load from self.cam_calib_dir, NOT self.calib_dir ---
        pf_cal_cam = pq.ParquetFile(os.path.join(self.cam_calib_dir, fname)) 
        df_cal_cam = pf_cal_cam.read_row_group(0).to_pandas()
        rows_cal_cam = df_cal_cam[(df_cal_cam["key.segment_context_name"] == seg)]

        df_b2d = None
        if self.box_2d_dir:
            pf_b2d = pq.ParquetFile(os.path.join(self.box_2d_dir, fname))
            df_b2d = pf_b2d.read_row_group(0).to_pandas()
            rows_b2d = df_b2d[(df_b2d["key.segment_context_name"] == seg) &
                              (df_b2d["key.frame_timestamp_micros"] == ts)]

        surround_views = []
        for cam_id, cam_name in self.camera_map.items():
            img_row = rows_img[rows_img["key.camera_name"] == cam_id]
            if len(img_row) == 0: continue
            
            img_row = img_row.iloc[0]
            img_bytes = img_row["[CameraImageComponent].image"]
            img_pil = Image.open(io.BytesIO(img_bytes))
            img_np = np.array(img_pil, dtype=np.uint8)

            # --- NEW: Get Camera Calibration for THIS camera ---
            cam_cal_row = rows_cal_cam[rows_cal_cam["key.camera_name"] == cam_id]
            if len(cam_cal_row) == 0:
                print(f"[WARN] No calibration found for camera {cam_id} - skipping.")
                continue
            cam_cal_row = cam_cal_row.iloc[0]

            # --- FIX: Use the robust helper function ---
            try:
                K, T_cv, img_dims = self._get_camera_calibration_robust(
                    cam_cal_row, self.M_reflect, debug=(idx==0) # Only debug first frame
                )
                img_H, img_W = img_dims # Unpack image dims
            except Exception as e:
                print(f"[ERROR] Failed to get robust calibration for cam {cam_id}: {e}")
                continue

            # --- NEW: Get Image Dimensions (from image itself or calibration) ---
            # It's safest to get from the image itself, but calibration usually has it too.
            img_H, img_W = img_np.shape[0], img_np.shape[1]
            img_H_from_calib = int(cam_cal_row["[CameraCalibrationComponent].height"])
            img_W_from_calib = int(cam_cal_row["[CameraCalibrationComponent].width"])
            


            boxes_xyxy, labels_2d = np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
            if df_b2d is not None:
                box_rows = rows_b2d[rows_b2d["key.camera_name"] == cam_id]
                if len(box_rows) > 0:
                    f_2d = ["[CameraBoxComponent].box.center.x", "[CameraBoxComponent].box.center.y",
                            "[CameraBoxComponent].box.size.x", "[CameraBoxComponent].box.size.y"]
                    boxes_cwh = box_rows[f_2d].to_numpy().astype(np.float32)
                    boxes_xyxy = np.zeros_like(boxes_cwh)
                    boxes_xyxy[:, 0] = boxes_cwh[:, 0] - boxes_cwh[:, 2] / 2.0 # x_min
                    boxes_xyxy[:, 1] = boxes_cwh[:, 1] - boxes_cwh[:, 3] / 2.0 # y_min
                    boxes_xyxy[:, 2] = boxes_cwh[:, 0] + boxes_cwh[:, 2] / 2.0 # x_max
                    boxes_xyxy[:, 3] = boxes_cwh[:, 1] + boxes_cwh[:, 3] / 2.0 # y_max
                    labels_2d = box_rows["[CameraBoxComponent].type"].to_numpy().astype(np.int64)
            
            surround_views.append({
                "image": torch.tensor(img_np), 
                "boxes_2d": torch.tensor(boxes_xyxy),
                "labels_2d": torch.tensor(labels_2d), 
                "camera_name": cam_name,
                "camera_id": cam_id,
                "K": torch.tensor(K, dtype=torch.float32),          # NEW
                "T_cv": torch.tensor(T_cv, dtype=torch.float32),    # NEW
                "image_dims": (img_H, img_W)                        # NEW
            })
            
            # Surround view data format for each camera:
            # Each dict in surround_views list contains:
            #   - "image": torch.Tensor of shape (H, W, 3) containing RGB image data (uint8)
            #   - "boxes_2d": torch.Tensor of shape (K, 4) containing 2D bounding boxes [x_min, y_min, x_max, y_max]
            #   - "labels_2d": torch.Tensor of shape (K,) containing 2D object class IDs matching 3D labels
            #   - "camera_name": int, camera sensor ID (1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT)
            
        target["surround_views"] = surround_views
        
        # ---------- 🧭 Debug summary (only for first call) ----------
        if self.num_sweeps == 1:
            print(f"[DEBUG] Single-frame {idx}: {len(lidar)} pts | "
                  f"boxes={len(boxes_any)}")
            
            # --- (Your other visualization calls can be commented out for now) ---
            
            # --- RUN TEST 1: Check LiDAR Coordinate System ---
            print("\n" + "="*30 + " DEBUG STEP 1 " + "="*30)
            print("Checking LiDAR Range Image coordinate system...")
            # We must pass the UNFLIPPED inclinations (max to min)
            # Your code flips it here: incl = inclinations[::-1]
            # So we pass the original 'inclinations'
            self.visualize_range_image_raw(rng, inclinations)
            
            
            # --- RUN TEST 2: Check Camera Projection ---
            print("\n" + "="*30 + " DEBUG STEP 2 " + "="*30)
            print("Checking Camera Projection Handedness...")
            
            # Find the FRONT camera data
            front_cam_data = None
            for view in target["surround_views"]:
                if view["camera_id"] == 1: # 1 is FRONT
                    front_cam_data = view
                    break
            
            if front_cam_data:
                self.visualize_camera_projection_test(
                    front_cam_data,
                    self._project_lidar_to_camera_image # Pass the function itself
                )
            else:
                print("[ERROR] Could not find FRONT camera to run test.")

        return lidar, target

    @staticmethod
    def _get_camera_calibration_robust(cam_cal_row, M_reflect, debug=True):
        """
        Loads camera intrinsics and robustly determines/validates/converts
        the extrinsic matrix (T_cv) from the Parquet data.

        Applies RH conversion T_cv_rh = T_cv_lh @ M.

        Returns: K (3x3), T_cv (4x4, RH Vehicle -> RH Std Camera), img_dims (H, W)
        """
        # --- 1. Intrinsics (K) ---
        # (Load K and img_dims - unchanged)
        fx = float(cam_cal_row["[CameraCalibrationComponent].intrinsic.f_u"])
        fy = float(cam_cal_row["[CameraCalibrationComponent].intrinsic.f_v"])
        cx = float(cam_cal_row["[CameraCalibrationComponent].intrinsic.c_u"])
        cy = float(cam_cal_row["[CameraCalibrationComponent].intrinsic.c_v"])
        W = int(cam_cal_row["[CameraCalibrationComponent].width"])
        H = int(cam_cal_row["[CameraCalibrationComponent].height"])
        img_dims = (H, W)
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        # (Debug prints - unchanged)

        # --- 2. Extrinsics (T_cv_lh) - Load and Validate ---
        # (Load vals, test row/col, choose R_lh, t_lh, apply scaling - unchanged)
        extr_vals = cam_cal_row["[CameraCalibrationComponent].extrinsic.transform"]
        extr_vals = extr_vals.as_py() if hasattr(extr_vals, "as_py") else extr_vals
        vals = np.array(extr_vals, dtype=np.float32)
        assert vals.size == 16, f"Unexpected extrinsic size: {vals.size}"
        R_row = np.array([[vals[0], vals[1], vals[2]], [vals[4], vals[5], vals[6]], [vals[8], vals[9], vals[10]]])
        t_row = np.array([vals[3], vals[7], vals[11]])
        R_col = np.array([[vals[0], vals[4], vals[8]], [vals[1], vals[5], vals[9]], [vals[2], vals[6], vals[10]]])
        t_col = np.array([vals[12], vals[13], vals[14]])
        det_row, ortho_err_row = np.linalg.det(R_row), np.linalg.norm(R_row.T @ R_row - np.eye(3))
        det_col, ortho_err_col = np.linalg.det(R_col), np.linalg.norm(R_col.T @ R_col - np.eye(3))
        # (Choose R_lh, t_lh based on validation - unchanged)
        if abs(det_col - 1.0) < abs(det_row - 1.0) * 0.9 and ortho_err_col < ortho_err_row * 0.9:
             R_lh = R_col; t_lh = t_col
             if debug: print("[INFO] Using column-major extrinsic interpretation.")
        else:
             R_lh = R_row; t_lh = t_row
             if debug: print("[INFO] Using row-major extrinsic interpretation.")
        # (Apply translation scaling - unchanged)
        t_mag = np.linalg.norm(t_lh)
        if t_mag > 50: t_lh /= 1000.0; # ... print ...
        elif t_mag < 0.01 and t_mag > 1e-6: t_lh *= 1000.0; # ... print ...

        # Construct the validated LH matrix
        T_cv_lh = np.eye(4, dtype=np.float32)
        T_cv_lh[:3, :3] = R_lh
        T_cv_lh[:3, 3]  = t_lh

        # --- 3. Convert LH extrinsic to RH extrinsic: T_cv_rh = T_cv_lh @ M ---
        T_cv = T_cv_lh @ M_reflect # Note: M_reflect is diag(1,-1,1,1)

        if debug:
             R_final = T_cv[:3,:3]
             t_final = T_cv[:3,3]
             print(f"[DEBUG] Final T_cv (RH Vehicle -> RH Std Camera) after T_lh @ M:")
             print(f"  det(R)={np.linalg.det(R_final):.3f}, ortho_err={np.linalg.norm(R_final.T @ R_final - np.eye(3)):.2e}")
             print(f"  t={t_final.round(3)}")

        return K, T_cv, img_dims
    # ---------------------------------------------------------------------
    # ---------- VISUALIZATION & HELPER METHODS ----------
    # ---------------------------------------------------------------------

    @staticmethod
    def _get_box_corners(boxes_7d):
        """
        Converts (N, 7) 3D box format [cx, cy, cz, dx, dy, dz, heading]
        to (N, 8, 3) corners.
        
        NOTE: Assumes 'cz' is the GEOMETRIC center (z = -h/2 to h/2).
        """
        if isinstance(boxes_7d, torch.Tensor):
            boxes_7d = boxes_7d.cpu().numpy()
        if boxes_7d.ndim == 1:
            boxes_7d = boxes_7d[np.newaxis, :]
            
        N = boxes_7d.shape[0]
        if N == 0:
            return np.zeros((0, 8, 3), dtype=np.float32)

        centers = boxes_7d[:, :3] # (N, 3) [cx, cy, cz_geom]
        dims = boxes_7d[:, 3:6]    # (N, 3) [l, w, h]
        headings = boxes_7d[:, 6]  # (N,)

        # --- 1. Get 8 corners in local box frame (axis-aligned) ---
        lwh = dims / 2.0 # (N, 3)

        # X-axis offsets [-l/2, l/2]
        l_corners = np.concatenate([-lwh[:, 0:1],  lwh[:, 0:1],  lwh[:, 0:1], -lwh[:, 0:1],
                                    -lwh[:, 0:1],  lwh[:, 0:1],  lwh[:, 0:1], -lwh[:, 0:1]], axis=1) # (N, 8)
        # Y-axis offsets [-w/2, w/2] (Waymo is +Y left)
        w_corners = np.concatenate([ lwh[:, 1:2],  lwh[:, 1:2], -lwh[:, 1:2], -lwh[:, 1:2],
                                     lwh[:, 1:2],  lwh[:, 1:2], -lwh[:, 1:2], -lwh[:, 1:2]], axis=1) # (N, 8)
        # Z-axis offsets [-h/2, h/2]
        z_corners = np.concatenate([-lwh[:, 2:], -lwh[:, 2:], -lwh[:, 2:], -lwh[:, 2:],
                                     lwh[:, 2:],  lwh[:, 2:],  lwh[:, 2:],  lwh[:, 2:]], axis=1) # (N, 8)

        corners_local = np.stack([l_corners, w_corners, z_corners], axis=2) # (N, 8, 3)

        # --- 2. Rotate corners by heading ---
        cos_h, sin_h = np.cos(headings), np.sin(headings)
        
        R = np.zeros((N, 3, 3), dtype=np.float32)
        R[:, 0, 0] = cos_h
        R[:, 0, 1] = -sin_h
        R[:, 1, 0] = sin_h
        R[:, 1, 1] = cos_h
        R[:, 2, 2] = 1.0

        corners_rotated = (R @ corners_local.transpose(0, 2, 1)).transpose(0, 2, 1)

        # --- 3. Translate corners ---
        corners_global = corners_rotated + centers[:, np.newaxis, :]
        return corners_global  # (N, 8, 3)

    @staticmethod
    def _project_points_to_image(points_lidar, inc_min, inc_max, H, W):
        """ Projects 3D points in LiDAR frame to (u, v) image coordinates. """
        X, Y, Z = points_lidar[:, 0], points_lidar[:, 1], points_lidar[:, 2]
        range = np.sqrt(X**2 + Y**2 + Z**2)
        azimuth = np.arctan2(Y, X) 
        inclination = np.arcsin(Z / (range + 1e-8))

        u = (azimuth + np.pi) / (2 * np.pi) * W
        u = np.clip(np.floor(u), 0, W - 1).astype(np.int32)
        v = (inclination - inc_max) / (inc_min - inc_max) * (H - 1)
        v = np.clip(np.floor(v), 0, H - 1).astype(np.int32)
        return np.stack([u, v], axis=-1)

    @staticmethod
    def visualize_camera_projection_test(camera_data_item, 
                                         project_function, 
                                         ax=None):
        """
        Validates camera intrinsics and extrinsics by projecting
        the VEHICLE's coordinate axes onto the image.
        
        Args:
            camera_data_item (dict): A single item from 'surround_views'.
            project_function (callable): Your _project_lidar_to_camera_image method.
            ax (matplotlib.axis, optional): Axis to plot on.
        """
        
        # --- 1. Extract Camera Data ---
        image = camera_data_item["image"].cpu().numpy()
        K = camera_data_item["K"].cpu().numpy()
        T_cv = camera_data_item["T_cv"].cpu().numpy()
        image_dims = camera_data_item["image_dims"]
        H, W = image_dims

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"Camera Projection Test ({camera_data_item['camera_name']})")
        ax.axis('off')
        
        # --- 2. Validate Intrinsic Matrix 'K' ---
        f_u, f_v = K[0, 0], K[1, 1]
        c_u, c_v = K[0, 2], K[1, 2]
        
        text = (f"Intrinsic Matrix (K) Validation:\n"
                f"  Image Dims (W, H): ({W}, {H})\n"
                f"  Principal Pt (c_u, c_v): ({c_u:.1f}, {c_v:.1f})\n"
                f"  Focal Len (f_u, f_v): ({f_u:.1f}, {f_v:.1f})\n\n"
                f"  Check: c_u ≈ W/2?  ({np.abs(c_u - W/2) < W*0.1})\n"
                f"  Check: c_v ≈ H/2?  ({np.abs(c_v - H/2) < H*0.1})")
        
        print(text)
        if not (np.abs(c_u - W/2) < W*0.1):
             print("[WARN] c_u is not near the image center. Intrinsic matrix may be wrong.")

        # --- 3. Create Test Points in VEHICLE Frame (RH) ---
        # +X Forward, +Y Left, +Z Up
        test_points_vehicle = np.array([
            [0, 0, 0],     # Origin (Ego)
            [5, 0, 0],     # 5m Forward (+X)
            [0, 5, 0],     # 5m Left (+Y)
            [0, -5, 0],    # 5m Right (-Y)
            [0, 0, 5]      # 5m Up (+Z)
        ], dtype=np.float32)
        
        labels = ["Origin", "+X (Fwd)", "+Y (Left)", "-Y (Right)", "+Z (Up)"]
        colors = ["magenta", "red", "green", "blue", "cyan"]

        # --- 4. Project points using your function ---
        projected_uv, _, visible_mask = project_function(
            test_points_vehicle, K, T_cv, image_dims
        )
        
        # --- 5. Draw Results ---
        # Re-index projected points based on the visibility mask
        visible_labels = np.array(labels)[visible_mask]
        visible_colors = np.array(colors)[visible_mask]
        
        for (u, v), label, color in zip(projected_uv, visible_labels, visible_colors):
            ax.scatter(u, v, s=200, c=color, marker='x', linewidth=3)
            ax.text(u + 10, v, f"{label} @ ({u}, {v})", 
                    color='white', fontsize=10,
                    bbox=dict(facecolor=color, alpha=0.7, pad=0.1))
                    
        if 'fig' in locals():
            plt.show()

    @staticmethod
    def _project_lidar_to_camera_image(points_vehicle, K, T_cv, image_dims):
        """
        Projects 3D LiDAR points (RH Vehicle frame) onto a 2D camera image
        using the combined Vehicle-to-Image projection method.
        Assumes T_cv maps RH Vehicle -> RH Standard Camera Frame (+Z Fwd).
        """
        # --- 1. Slice to get XYZ ---
        points_xyz = points_vehicle[:, :3] # Input is RH Vehicle Frame

        if points_xyz.shape[0] == 0: # Check if there are any points to project
            return np.zeros((0, 2)), np.zeros((0,)), np.zeros(0, dtype=bool)

        H, W = image_dims # Get image height and width

        # --- 2. Construct the Combined Vehicle-to-Image Projection Matrix ---
        # Assumes T_cv maps Vehicle(RH) -> Standard Camera(RH: +X Right, +Y Down, +Z Fwd)
        # K is the 3x3 intrinsic matrix.
        # P = K @ T_cv[:3, :] results in a 3x4 matrix.
        P_vehicle_to_image = K @ T_cv[:3, :] # (3x3) @ (3x4) -> (3x4)

        # --- 3. Transform Vehicle points to Homogeneous Image Coords ---
        # Convert points to homogeneous coordinates (N, 4)
        points_h_vehicle = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float32)], axis=1)

        # Project points using the combined matrix: (N, 4) @ (4, 3) -> (N, 3)  [u*w, v*w, w]
        uvw_homogeneous = points_h_vehicle @ P_vehicle_to_image.T

        # --- 4. Extract Depth (w) and Filter Points Behind Camera ---
        # 'w' (depth) is the 3rd component, corresponding to the Z-axis in the Standard Camera Frame (Forward)
        depths = uvw_homogeneous[:, 2]
        # Filter out points behind or too close to the camera plane
        in_front_mask = depths > 0.01

        # --- DEBUG ---
        print(f"\n--- Projecting {points_xyz.shape[0]} points (M@T Method) ---")
        print(f"Standard Camera Coords (from P matrix) sample:\n{uvw_homogeneous[:5].round(1)}") # Note: These aren't true 3D coords, but projected homogeneous image coords
        print(f"Depths (w = Zs) sample: {depths[:5].round(2)}")
        print(f"Points in front: {np.sum(in_front_mask)} / {points_xyz.shape[0]}")
        # --- END DEBUG ---

        # If no points are in front, return empty arrays
        if not np.any(in_front_mask):
             print("[DEBUG] No points passed the 'in_front' filter.")
             return np.zeros((0, 2)), np.zeros((0,)), np.zeros(points_vehicle.shape[0], dtype=bool)

        # Select only the points (and their depths) that are in front
        uvw_in_front = uvw_homogeneous[in_front_mask]
        depths_in_front = depths[in_front_mask]

        # --- 5. Normalize to get (u, v) Pixel Coordinates ---
        # Divide [u*w, v*w] by w to get [u, v]
        # Use np.maximum to prevent division by zero or very small numbers
        safe_depths = np.maximum(depths_in_front[:, None], 1e-6)
        projected_coords = uvw_in_front[:, :2] / safe_depths # (N_front, 2) [u, v]

        # --- DEBUG ---
        print(f"Final Pixel Coords [u, v] sample:\n{projected_coords[:5].round(1)}")
        # --- END DEBUG ---

        # --- 6. Filter points outside image bounds ---
        u_coords = projected_coords[:, 0]
        v_coords = projected_coords[:, 1]
        # Check if u and v are within the valid pixel range [0, W-1] and [0, H-1]
        in_image_mask_step6 = (u_coords >= 0) & (u_coords < W) & \
                               (v_coords >= 0) & (v_coords < H)

        # --- DEBUG ---
        print(f"Points within image bounds: {np.sum(in_image_mask_step6)} / {np.sum(in_front_mask)}")
        u_vis = u_coords[in_image_mask_step6]
        v_vis = v_coords[in_image_mask_step6]
        if len(u_vis) > 0: # Check if there are any visible points before calculating min/max
            print(f"Visible Pixel Range: u=[{u_vis.min():.0f}, {u_vis.max():.0f}], v=[{v_vis.min():.0f}, {v_vis.max():.0f}] (W={W}, H={H})")
        # --- END DEBUG ---

        # Apply the final filter to get coordinates and depths of visible points
        final_projected_coords = projected_coords[in_image_mask_step6]
        final_depths = depths_in_front[in_image_mask_step6] # Return Zs (depth)

        # --- 7. Reconstruct the full boolean mask for the original input points ---
        visible_mask = np.zeros(points_vehicle.shape[0], dtype=bool) # Initialize mask for all input points
        temp_mask_indices = np.where(in_front_mask)[0] # Indices of points that passed the 'in_front' filter
        if temp_mask_indices.shape[0] > 0: # Check if any points passed the first filter
            final_indices_relative = np.where(in_image_mask_step6)[0] # Indices (relative to 'in_front') that also passed the image bounds filter
            if final_indices_relative.shape[0] > 0: # Check if any points passed the second filter
                original_indices_passed = temp_mask_indices[final_indices_relative] # Map back to original indices
                visible_mask[original_indices_passed] = True # Set mask to True for visible points

        # Return the final projected coordinates, depths, and the visibility mask
        return final_projected_coords.astype(np.int32), final_depths, visible_mask

    @staticmethod
    def _project_lidar_to_camera_image_old(points_vehicle, K, T_cv, image_dims):
        """
        Projects 3D LiDAR points (in Vehicle frame) onto a 2D camera image.
        """
        # --- 1. Slice to get XYZ ---
        # Input `points_vehicle` is (N, 5) in a RIGHT-HANDED system (+Y Left)
        points_xyz = points_vehicle[:, :3].copy()
        
        if points_xyz.shape[0] == 0:
            return np.zeros((0, 2)), np.zeros((0,)), np.zeros(0, dtype=bool)

        H, W = image_dims

        # --- 2. FIX: Convert to LEFT-HANDED system for camera extrinsics ---
        # The T_cv matrix expects a vehicle frame with +Y Right.
        # We must mirror our points by flipping the Y coordinate.
        #points_xyz[:, 1] *= -1.0  # Y -> -Y

        # --- 3. Transform points from Vehicle Frame to Camera Frame ---
        # Make points homogeneous (N, 4)
        points_h_vehicle = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float32)], axis=1)
        
        # Apply T_cv (Camera from Vehicle) extrinsic transform
        # This now works because T_cv is receiving the (Left-Handed) points it expects.
        points_h_camera = points_h_vehicle @ T_cv.T 
        
        # Extract 3D camera coordinates (N, 3)
        points_camera = points_h_camera[:, :3]

        # --- 4. Filter points: only keep those in front of the camera ---
        # In the Waymo camera frame, +Z is forward.
        depths = points_camera[:, 2] 
        in_front_mask = depths > 0.01 
        
        # --- 5. Apply Intrinsic Projection (Camera Frame to Image Plane) ---
        # (This part of your code was already correct)
        points_normalized_camera = points_camera[in_front_mask, :2] / depths[in_front_mask, None]
        points_normalized_h = np.concatenate([points_normalized_camera, np.ones((points_normalized_camera.shape[0], 1))], axis=1)
        uv_homogeneous = points_normalized_h @ K.T
        projected_coords = uv_homogeneous[:, :2] / uv_homogeneous[:, 2:]
        
        # --- 6. Filter points: only keep those within image bounds ---
        u_coords = projected_coords[:, 0]
        v_coords = projected_coords[:, 1]
        
        in_image_mask = (u_coords >= 0) & (u_coords < W) & \
                        (v_coords >= 0) & (v_coords < H)
        
        # (Rest of the function to build the final mask is unchanged)
        final_projected_coords = projected_coords[in_image_mask]
        final_depths = depths[in_front_mask][in_image_mask]

        visible_mask = np.zeros(points_vehicle.shape[0], dtype=bool)
        temp_mask_indices = np.where(in_front_mask)[0]
        if temp_mask_indices.shape[0] > 0:
             visible_mask[temp_mask_indices[in_image_mask]] = True
        
        return final_projected_coords.astype(np.int32), final_depths, visible_mask
        
    @staticmethod
    def _project_lidar_to_camera_image_old(points_vehicle, K, T_cv, image_dims):
        """
        Projects 3D LiDAR points (in Vehicle frame) onto a 2D camera image.
        
        Args:
            points_vehicle (np.ndarray): (N, 3) XYZ points in the Vehicle frame.
            K (np.ndarray): (3, 3) Camera intrinsic matrix.
            T_cv (np.ndarray): (4, 4) Extrinsic matrix (Camera from Vehicle).
            image_dims (tuple): (H, W) of the camera image.
            
        Returns:
            projected_coords (np.ndarray): (N_visible, 2) (u, v) pixel coordinates.
            visible_points_depth (np.ndarray): (N_visible,) depth values for visible points.
            visible_mask (np.ndarray): (N,) boolean mask indicating which input points are visible.
        """
        # --- FIX: Ensure points are (N, 3) by slicing ---
        # The input might be (N, 4) or (N, 5) with intensity/time.
        points_vehicle = points_vehicle[:, :3]

        if points_vehicle.shape[0] == 0:
            return np.zeros((0, 2)), np.zeros((0,)), np.zeros(0, dtype=bool)

        H, W = image_dims

        # --- 1. Transform points from Vehicle Frame to Camera Frame ---
        # Make points homogeneous (N, 4)
        points_h_vehicle = np.concatenate([points_vehicle, np.ones((points_vehicle.shape[0], 1), dtype=np.float32)], axis=1)
        
        # Apply T_cv (Camera from Vehicle) extrinsic transform
        # The result 'points_h_camera' is (N, 4) in the camera's homogeneous coordinate system
        points_h_camera = points_h_vehicle @ T_cv.T 
        
        # Extract 3D camera coordinates (N, 3)
        points_camera = points_h_camera[:, :3]

        # --- 2. Filter points: only keep those in front of the camera ---
        # In camera coordinates, +Z is typically forward.
        # Points behind the camera (Z < 0) are not visible.
        depths = points_camera[:, 2] # Z-coordinate in camera frame is depth
        
        # Filter 1: Z > 0 (in front of camera)
        # Add a small epsilon to avoid issues at Z=0.
        in_front_mask = depths > 0.01 
        
        # --- 3. Apply Intrinsic Projection (Camera Frame to Image Plane) ---
        # (u, v) = K @ (X_c / Z_c, Y_c / Z_c)
        # Divide X_c, Y_c by Z_c to get normalized homogeneous coordinates
        points_normalized_camera = points_camera[in_front_mask, :2] / depths[in_front_mask, None] # (N_front, 2)
        
        # Pad with 1s for matrix multiplication with K
        points_normalized_h = np.concatenate([points_normalized_camera, np.ones((points_normalized_camera.shape[0], 1))], axis=1) # (N_front, 3)
        
        # Apply intrinsic matrix K to get pixel coordinates (u, v, w)
        uv_homogeneous = points_normalized_h @ K.T # (N_front, 3)
        
        # Normalize by the third component (w) to get (u, v)
        projected_coords = uv_homogeneous[:, :2] / uv_homogeneous[:, 2:] # (N_front, 2)
        
        # --- 4. Filter points: only keep those within image bounds ---
        u_coords = projected_coords[:, 0]
        v_coords = projected_coords[:, 1]
        
        # Filter 2: within image width [0, W-1] and height [0, H-1]
        in_image_mask = (u_coords >= 0) & (u_coords < W) & \
                        (v_coords >= 0) & (v_coords < H)
        
        # Apply final filter
        final_projected_coords = projected_coords[in_image_mask]
        final_depths = depths[in_front_mask][in_image_mask]

        # Reconstruct the full boolean mask for the original input points
        visible_mask = np.zeros(points_vehicle.shape[0], dtype=bool)
        temp_mask_indices = np.where(in_front_mask)[0]
        visible_mask[temp_mask_indices[in_image_mask]] = True
        
        return final_projected_coords.astype(np.int32), final_depths, visible_mask
    # ---------------------------------------------------------------------
    # ---------- START: 2D VISUALIZATION METHOD ----------
    # ---------------------------------------------------------------------

    @staticmethod
    def visualize_range_image_raw(rng_image, inclinations):
        """
        Visualizes the raw 2D LiDAR range image and labels its axes
        to understand the native sensor coordinate system.
        
        Args:
            rng_image (np.ndarray): (H, W) raw range data.
            inclinations (np.ndarray): (H,) vector of beam inclinations (radians),
                                       in its original, unflipped order (max to min).
        """
        H, W = rng_image.shape
        
        # Use log scale for better visibility
        vis_range = np.log1p(np.clip(rng_image, 0, 80.0))
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        im = ax.imshow(vis_range, cmap='jet', aspect='auto')
        plt.colorbar(im, ax=ax, label="Log(1 + Range (m))")
        
        ax.set_title("LiDAR Range Image Coordinate System Check")

        # --- Label Y-Axis (Inclination) ---
        # The image H (rows) correspond to the H beams
        ax.set_ylabel("Inclination (Beam Index)")
        # Show ticks at the top, middle, and bottom
        ax.set_yticks([0, H // 2, H - 1])
        ax.set_yticklabels([
            f"Row 0 (Max Incl: {np.degrees(inclinations[0]):.1f}°)",
            f"Row {H // 2}",
            f"Row {H - 1} (Min Incl: {np.degrees(inclinations[-1]):.1f}°)"
        ])

        # --- Label X-Axis (Azimuth) ---
        # Your _get_frame Step 5 uses: az = np.linspace(-np.pi, np.pi, W)
        ax.set_xlabel("Azimuth (Horizontal Angle)")
        # Show ticks at left, center, and right
        ax.set_xticks([0, W // 2, W - 1])
        ax.set_xticklabels([
            "Col 0 (Azimuth = -π / -180°)",
            "Col 1325 (Azimuth = 0°)\n[SENSOR FORWARD, +X]",
            "Col 2649 (Azimuth = +π / +180°)"
        ])
        
        # --- Add Coordinate System Analysis Text ---
        # Based on your _get_frame Step 5:
        # Xl = rng * cos_i * cos_a
        # Yl = rng * cos_i * sin_a
        #
        # Test 1: Point to the LEFT
        # Azimuth = +pi/2 (Col ~1987) -> cos_a=0, sin_a=1 -> Xl=0, Yl > 0
        # Test 2: Point to the RIGHT
        # Azimuth = -pi/2 (Col ~662) -> cos_a=0, sin_a=-1 -> Xl=0, Yl < 0
        #
        # CONCLUSION: +Y is LEFT. This is a RIGHT-HANDED system.
        
        text = ("Coordinate System Check (based on _get_frame Step 5):\n"
                "Azimuth 0° (+X) is Sensor Forward.\n"
                "Azimuth +90° (+Y) is Sensor Left.\n"
                "Azimuth -90° (-Y) is Sensor Right.\n"
                "Conclusion: This code generates a RIGHT-HANDED Point Cloud (+Y Left)")
        ax.text(0.5, -0.25, text, ha='center', va='top', 
                transform=ax.transAxes, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
        
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        plt.show()

    @staticmethod
    def visualize_camera_with_lidar(camera_data_item, lidar_points_vehicle, 
                                    boxes_7d_vehicle=None, boxes_labels_3d=None, 
                                    title="", ax=None):
        """
        Draws 3D LiDAR points (colored by depth) and optionally 3D boxes 
        onto a 2D camera image.
        
        Args:
            camera_data_item (dict): A single dictionary item from 'surround_views' 
                                     containing 'image', 'K', 'T_cv', 'image_dims'.
            lidar_points_vehicle (torch.Tensor or np.ndarray): (N, 3) XYZ points 
                                                                in the Vehicle frame.
            boxes_7d_vehicle (torch.Tensor or np.ndarray, optional): (M, 7) boxes 
                                                                     in Vehicle frame.
            boxes_labels_3d (torch.Tensor or np.ndarray, optional): (M,) int labels for 3D boxes.
            title (str): Title for the plot.
            ax (matplotlib.axis, optional): Axis to plot on.
        """
        
        # --- 1. Extract Camera Data ---
        image = camera_data_item["image"].cpu().numpy()
        K = camera_data_item["K"].cpu().numpy()
        T_cv = camera_data_item["T_cv"].cpu().numpy()
        image_dims = camera_data_item["image_dims"]
        camera_name = camera_data_item["camera_name"]

        # --- 2. Prepare LiDAR Points ---
        if isinstance(lidar_points_vehicle, torch.Tensor):
            lidar_points_vehicle = lidar_points_vehicle.cpu().numpy()
        
        if lidar_points_vehicle.shape[0] == 0:
            print("[WARN] No LiDAR points to project.")
            return

        # --- 3. Project LiDAR Points onto the Camera Image ---
        # This returns (u, v) pixel coordinates, their depth, and a mask
        projected_uv, projected_depths, visible_mask = \
            Waymo3DDataset._project_lidar_to_camera_image(lidar_points_vehicle, K, T_cv, image_dims)

        # --- 4. Setup Plot ---
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
        ax.imshow(image)
        ax.set_title(f"{title} - {camera_name} (LiDAR Projected)")
        ax.axis('off')

        # --- 5. Plot Projected LiDAR Points ---
        if projected_uv.shape[0] > 0:
            # Color points by inverse depth (closer points are brighter/warmer)
            # Clip depth to a reasonable range (e.g., 0-80m) for better visualization
            display_depths = np.clip(projected_depths, 0, 80) 
            
            # Normalize depths for colormap
            if display_depths.max() > display_depths.min():
                norm_depths = (display_depths - display_depths.min()) / (display_depths.max() - display_depths.min())
            else:
                norm_depths = np.zeros_like(display_depths)

            # Use a colormap (e.g., 'viridis' or 'hot')
            cmap = plt.cm.viridis
            colors = cmap(1.0 - norm_depths) # 1.0 - norm_depths to make closer points warmer

            # Scatter plot points
            ax.scatter(projected_uv[:, 0], projected_uv[:, 1], 
                       s=5, c=colors, marker='.', alpha=0.7) # Adjust size (s) and alpha as needed
        else:
            print("[INFO] No LiDAR points visible in this camera view.")

        # --- 6. Optionally: Project and Plot 3D Bounding Boxes ---
        if boxes_7d_vehicle is not None and boxes_7d_vehicle.shape[0] > 0:
            if isinstance(boxes_7d_vehicle, torch.Tensor):
                boxes_7d_vehicle = boxes_7d_vehicle.cpu().numpy()
            if isinstance(boxes_labels_3d, torch.Tensor):
                boxes_labels_3d = boxes_labels_3d.cpu().numpy()

            # --- Get 8 corners for each box in Vehicle Frame ---
            corners_vehicle = Waymo3DDataset._get_box_corners(boxes_7d_vehicle) # (M, 8, 3)
            
            # --- Define box lines for drawing (same as visualize_3d) ---
            box_lines = [
                [0, 1], [1, 2], [2, 3], [3, 0], # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4], # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical pillars
            ]
            
            label_colors_map = {
                1: 'green',  # Vehicle
                2: 'red',    # Pedestrian
                4: 'yellow', # Cyclist
            }

            for i in range(boxes_7d_vehicle.shape[0]):
                single_box_corners_vehicle = corners_vehicle[i] # (8, 3)
                
                # Project all 8 corners of this box
                box_projected_uv, box_projected_depths, box_visible_mask = \
                    Waymo3DDataset._project_lidar_to_camera_image(
                        single_box_corners_vehicle, K, T_cv, image_dims
                    )
                
                # Check if at least some corners are visible
                if box_projected_uv.shape[0] > 0:
                    # Get the color for the box
                    label_id = int(boxes_labels_3d[i]) if boxes_labels_3d is not None and i < len(boxes_labels_3d) else 0
                    box_color = label_colors_map.get(label_id, 'orange') # Default orange for unknown

                    # Draw the 12 lines of the box
                    for line in box_lines:
                        p1_idx, p2_idx = line[0], line[1]
                        
                        # Only draw line if both points are visible
                        if box_visible_mask[p1_idx] and box_visible_mask[p2_idx]:
                            p1_uv = box_projected_uv[np.where(np.where(box_visible_mask)[0] == p1_idx)[0]][0]
                            p2_uv = box_projected_uv[np.where(np.where(box_visible_mask)[0] == p2_idx)[0]][0]
                            
                            ax.plot([p1_uv[0], p2_uv[0]], [p1_uv[1], p2_uv[1]],
                                    color=box_color, linewidth=1.5, alpha=0.8)
                        # Optional: Draw partially visible lines if one point is visible
                        # This would be more complex, involving clipping lines at image boundaries.
                        # For simplicity, we only draw if both endpoints are visible.

        if 'fig' in locals():
            plt.show()

    @staticmethod
    def visualize_surround_view(surround_views, label_map=None):
        """
        Visualizes the 5 surround-view camera images with 2D boxes.
        
        Args:
            surround_views (list): A list of dictionaries, one for each camera.
                                   From target['surround_views'].
            label_map (dict): Maps int labels to string names.
        """
        if not surround_views:
            print("[WARN] No surround views to visualize.")
            return

        # --- Create a 2x3 grid for the 5 cameras ---
        # (Front-Left, Front, Front-Right)
        # (Side-Left, Empty, Side-Right)
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        fig.suptitle("Surround-View Cameras", fontsize=16)
        
        # Map camera names to subplot positions
        # (row, col)
        cam_positions = {
            "FRONT_LEFT": (0, 0),
            "FRONT": (0, 1),
            "FRONT_RIGHT": (0, 2),
            "SIDE_LEFT": (1, 0),
            "SIDE_RIGHT": (1, 2),
        }
        
        # Colors for 2D box types
        label_colors = {
            1: 'g', # Vehicle
            2: 'r', # Pedestrian
            3: 'y', # Cyclist
            4: 'b', # Other
        }
        default_color = 'gray'

        # Turn off the empty subplot (1, 1)
        axes[1, 1].axis('off')

        for view in surround_views:
            cam_name = view["camera_name"]
            if cam_name not in cam_positions:
                continue
                
            r, c = cam_positions[cam_name]
            ax = axes[r, c]
            
            image = view["image"]
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            ax.imshow(image)
            ax.set_title(f"{cam_name} ({view['camera_id']})")
            ax.axis('off')
            
            # --- Draw 2D Boxes ---
            boxes_2d = view["boxes_2d"]
            labels_2d = view["labels_2d"]
            if isinstance(boxes_2d, torch.Tensor):
                boxes_2d = boxes_2d.cpu().numpy()
            if isinstance(labels_2d, torch.Tensor):
                labels_2d = labels_2d.cpu().numpy()

            for i in range(len(boxes_2d)):
                box = boxes_2d[i] # [x_min, y_min, x_max, y_max]
                label_id = int(labels_2d[i])
                
                x_min, y_min, x_max, y_max = box
                w = x_max - x_min
                h = y_max - y_min
                
                color = label_colors.get(label_id, default_color)
                
                # Create a rectangle patch
                rect = patches.Rectangle(
                    (x_min, y_min), w, h,
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label text
                if label_map:
                    label_text = label_map.get(label_id, "UNK")
                    ax.text(x_min, y_min - 5, label_text, 
                            color='white',
                            fontsize=8,
                            bbox=dict(facecolor=color, alpha=0.5, pad=0.1))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    @staticmethod
    def visualize_3d(lidar_points, boxes_7d=None, labels=None, label_map=None,
                     headless=False, save_path="scene_vis.ply"):
        """
        Visualizes LiDAR points and 3D boxes using Open3D.
        This function expects boxes in (N, 7) format and calls
        _get_box_corners() internally.

        - Points are colored by height (Z-axis).
        - A coordinate frame is drawn at the origin with XYZ labels.
        - 3D labels are drawn close to the top-center of boxes.
        
        Args:
            lidar_points (torch.Tensor or np.ndarray): (N, 3+) XYZ...
            boxes_7d (torch.Tensor or np.ndarray): (M, 7) [cx,cy,cz,dx,dy,dz,h]
            labels (torch.Tensor or np.ndarray): (M,) int labels
            label_map (dict): Maps int labels to string names.
            headless (bool): If True, save to file instead of opening window.
            save_path (str): Base path for saving .ply files.
        """
        
        # --- 0. Auto-detect headless mode (if not forced) ---
        if not headless and os.environ.get('DISPLAY') is None:
            print("[INFO] No display detected. Forcing headless mode.")
            headless = True

        # --- 1. Prepare Data ---
        if isinstance(lidar_points, torch.Tensor):
            lidar_points = lidar_points.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # --- 2. Create Point Cloud Geometry ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
        
        z_values = lidar_points[:, 2]
        if z_values.max() > z_values.min():
            z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min())
        else:
            z_norm = np.zeros_like(z_values)
        colors = plt.get_cmap("jet")(z_norm)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # --- 3. Create Coordinate Frame Geometry ---
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.5, origin=[0, 0, 0]
        )
        
        # --- 4. Create Box & Label Geometries ---
        box_geometries = []
        box_labels_3d = [] 
        
        if boxes_7d is not None and len(boxes_7d) > 0:
            
            # --- CRITICAL: Call _get_box_corners ---
            try:
                # Assuming Waymo3DDataset class is available or _get_box_corners is a static method
                corners_3d = Waymo3DDataset._get_box_corners(boxes_7d) # (N, 8, 3)
            except Exception as e:
                print(f"[ERROR] Failed to call _get_box_corners: {e}")
                return

            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0], # Bottom face (corners 0,1,2,3 from _get_box_corners)
                [4, 5], [5, 6], [6, 7], [7, 4], # Top face (corners 4,5,6,7)
                [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical pillars
            ]
            
            label_colors = {
                1: [0, 1, 0], # Vehicle (Green)
                2: [1, 0, 0], # Pedestrian (Red)
                4: [1, 1, 0], # Cyclist (Yellow)
            }
            default_color = [0.5, 0.5, 0.5] # Other (Gray)

            for i in range(corners_3d.shape[0]):
                box_corners = corners_3d[i]
                label_id = int(labels[i]) if labels is not None and i < len(labels) else 0
                
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(box_corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                color = label_colors.get(label_id, default_color)
                line_set.paint_uniform_color(color)
                box_geometries.append(line_set)
                
                # --- Prepare 3D Label (Close to Box) ---
                label_text = str(label_id)
                if label_map and label_id in label_map:
                    label_text = label_map[label_id]
                
                # Position label at the center of the TOP face of the box
                # Corners 4,5,6,7 form the top face. Averaging them gives the top center.
                # Adjusted to be only slightly above the box (e.g., +0.1m)
                label_pos = box_corners[4:].mean(axis=0) + [0, 0, 0.1] 
                box_labels_3d.append((label_pos, label_text))

        # --- 5. Combine Geometries ---
        geometries = [pcd, mesh_frame] + box_geometries

        # --- 6. Execute Visualization or Save ---
        if headless:
            print(f"[INFO] Headless mode: Saving geometries to {save_path}...")
            # Derive output paths for points and boxes
            base, ext = os.path.splitext(save_path)
            if ext == "":
                ext = ".ply"
            points_path = f"{base}_points{ext}"
            boxes_path = f"{base}_boxes{ext}"

            # Save point cloud
            if len(pcd.points) == 0:
                print("[WARN] Point cloud has no points; skipping save.")
            else:
                o3d.io.write_point_cloud(points_path, pcd)
                print(f"[INFO] Point cloud saved to {points_path}")

            # Merge all box LineSets into a single LineSet and save
            if len(box_geometries) > 0:
                merged = o3d.geometry.LineSet()
                all_points = []
                all_lines = []
                all_colors = []
                offset = 0
                for ls in box_geometries:
                    pts = np.asarray(ls.points)
                    lines = np.asarray(ls.lines)
                    # If colors are not set, default to gray per line
                    colors = np.asarray(ls.colors) if ls.colors is not None else np.tile(np.array([0.5, 0.5, 0.5]), (lines.shape[0], 1))
                    all_points.append(pts)
                    all_lines.append(lines + offset)
                    all_colors.append(colors)
                    offset += pts.shape[0]
                if len(all_points) > 0:
                    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
                    merged.lines = o3d.utility.Vector2iVector(np.vstack(all_lines))
                    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
                    o3d.io.write_line_set(boxes_path, merged)
                    print(f"[INFO] 3D bounding boxes saved to {boxes_path}")
            else:
                print("[INFO] No 3D boxes to save.")
        else:
            # --- Open Interactive Window ---
            print("[INFO] Opening 3D visualizer... (Press 'Q' to close)")
            try:
                app = gui.Application.instance
                app.initialize()

                vis = o3d.visualization.O3DVisualizer("Waymo 3D Visualization", 1280, 720)
                
                vis.add_geometry("points", pcd)
                vis.add_geometry("frame", mesh_frame)
                
                # --- Add XYZ labels to the coordinate frame ---
                # Create a material for the labels
                text_material = rendering.MaterialRecord()
                text_material.base_color = [0.8, 0.8, 0.8, 1.0] # Light gray
                text_material.shader = "defaultLit"
                text_material.line_width = 2
                
                vis.add_3d_label([1.6, 0, 0], "X") # Slightly beyond the red axis
                vis.add_3d_label([0, 1.6, 0], "Y") # Slightly beyond the green axis
                vis.add_3d_label([0, 0, 1.6], "Z") # Slightly beyond the blue axis

                for i, box_geom in enumerate(box_geometries):
                    vis.add_geometry(f"box_{i}", box_geom)
                
                # Add 3D labels for boxes
                for i, (pos, text) in enumerate(box_labels_3d):
                    vis.add_3d_label(pos, text)
                    
                # Set camera to a reasonable default and reset for best view
                vis.setup_camera(60.0, pcd.get_center(), [0, 0, 30], [0, 0, 1])
                vis.reset_camera_to_default() # This is often crucial after setup_camera
                
                app.run()
                
            except Exception as e:
                # Fallback to simple visualizer if O3DVisualizer fails
                print(f"[ERROR] Failed to start O3DVisualizer (likely due to missing GUI or version mismatch): {e}")
                print("       Falling back to simple visualizer (no labels, limited camera control).")
                o3d.visualization.draw_geometries(geometries)
    
    @staticmethod
    def visualize_bev(lidar_points, boxes_7d=None, ax=None, 
                      point_size=0.1, range_m=80.0, title="BEV Visualization"):
        """
        Visualizes LiDAR points and 3D boxes in Bird's-Eye View (BEV).
        This function expects boxes in (N, 7) format and calls
        _get_box_corners() internally.
        
        Assumes points and boxes are in the *same* coordinate frame.
        
        Args:
            lidar_points (torch.Tensor or np.ndarray): (N, 3+) XYZ...
            boxes_7d (torch.Tensor or np.ndarray): (M, 7) [cx,cy,cz,dx,dy,dz,h]
            ax (matplotlib.axis, optional): Axis to plot on.
            point_size (float): Size of plotted points.
            range_m (float or tuple): Plot range. Can be float (e.g., 80.0 for +/-80m) 
                                     or tuple ((-x,x), (-y,y)).
            title (str): Plot title.
        """
        
        # --- 1. Handle Inputs ---
        if isinstance(lidar_points, torch.Tensor):
            lidar_points = lidar_points.cpu().numpy()
        
        # --- 2. Setup Plot ---
        if ax is None:
            # Create new plot if one isn't provided
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.set_aspect('equal') # Critical for BEV
        
        # Configure plot range
        if isinstance(range_m, (int, float)):
            x_range = y_range = (-range_m, range_m)
        else:
            x_range, y_range = range_m
            
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel("X (m) [Vehicle Forward]")
        ax.set_ylabel("Y (m) [Vehicle Left]")
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.5)

        # --- 3. Plot Ego Vehicle ---
        # Draw a marker at (0,0) to show the vehicle's origin
        ax.plot(0, 0, 'rx', markersize=10, mew=2, label="Ego Origin (0,0)")
        
        # --- 4. Plot LiDAR Points ---
        # Filter points to be within the plot range for performance
        mask = (lidar_points[:, 0] >= x_range[0]) & (lidar_points[:, 0] <= x_range[1]) & \
               (lidar_points[:, 1] >= y_range[0]) & (lidar_points[:, 1] <= y_range[1])
        points_filtered = lidar_points[mask]
        
        # Color points by intensity (column 3) if available, else by height (column 2)
        if points_filtered.shape[1] > 3:
            colors = points_filtered[:, 3] # Intensity
            cmap = 'gray'
        else:
            colors = points_filtered[:, 2] # Z-height
            cmap = 'jet'
            
        ax.scatter(points_filtered[:, 0], points_filtered[:, 1], 
                   s=point_size, c=colors, cmap=cmap, alpha=0.5)

        # --- 5. Plot Boxes ---
        if boxes_7d is not None and len(boxes_7d) > 0:
            
            # --- CRITICAL: Call _get_box_corners ---
            # Convert (N, 7) [c, d, h] format to (N, 8, 3) corners
            # This requires the Waymo3DDataset class to be available
            try:
                corners_3d = Waymo3DDataset._get_box_corners(boxes_7d) # (N, 8, 3)
            except Exception as e:
                print(f"[ERROR] Failed to call _get_box_corners: {e}")
                return

            # --- Get the 4 bottom corners for the 2D polygon ---
            # The _get_box_corners (geometric center) function produces this order:
            # 0: [-l/2,  w/2, -h/2] (Back-Left-Bottom)
            # 1: [ l/2,  w/2, -h/2] (Front-Left-Bottom)
            # 2: [ l/2, -w/2, -h/2] (Front-Right-Bottom)
            # 3: [-l/2, -w/2, -h/2] (Back-Right-Bottom)
            bev_corners = corners_3d[:, [0, 1, 2, 3], :2] # (N, 4, 2)
            
            # Draw each box
            for i in range(bev_corners.shape[0]):
                # Draw the 2D polygon
                ax.add_patch(patches.Polygon(bev_corners[i],
                                             closed=True,
                                             color='b',
                                             fill=False,
                                             linewidth=1.5))
                
                # --- Draw heading indicator line ---
                # Get the center of the bottom face
                center = bev_corners[i].mean(axis=0)
                # Get the midpoint of the front-bottom edge (corners 1 and 2)
                front_mid = bev_corners[i, [1, 2], :].mean(axis=0)
                
                # Draw line from center to front
                ax.plot([center[0], front_mid[0]],
                        [center[1], front_mid[1]],
                        'b-', linewidth=1.5, alpha=0.8)

        ax.legend(loc='upper right')
        
        # Show plot if it was created by this function
        if 'fig' in locals():
            plt.show()

def main():

    DATA_ROOT = "/data/Datasets/waymodata/" # ⚠️ Update this path
    
    # ==================================================================
    print("\n\n========== TEST 1: Vehicle Frame (Single Sweep) ==========")
    # ==================================================================
    return_world = False
    ds_vehicle = Waymo3DDataset(
        DATA_ROOT, 
        split="training", 
        max_frames=80, 
        return_world=return_world,
        num_sweeps=1
    )
    lidar, target = ds_vehicle[70] # Get the first frame

    print("\n========== BASIC INFO (Vehicle Frame) ==========")
    print("points:", lidar.shape) # (N, 5) -> [x, y, z, i, t]
    print("boxes :", target["boxes_3d"].shape)
    if len(target["labels"]):
        print("labels:", target["labels"].unique())

    inten = lidar[:, 3].numpy()
    time = lidar[:, 4].numpy()
    print(f"Intensity range: {inten.min():.3f} → {inten.max():.3f}")
    print(f"Time delta range: {time.min():.3f} → {time.max():.3f}")

    print("\n--- VISUALIZING (Vehicle Frame) ---")

    # --- Test 2D Surround View ---
    Waymo3DDataset.visualize_surround_view(
        target["surround_views"], 
        ds_vehicle.label_map_2d
    )

    # Lidar to Image
    # if target["surround_views"]:
    #     print("[INFO] Calling Camera with LiDAR Visualizer...")
    #     for i, camera_data in enumerate(target["surround_views"]):
    #         # Create a separate subplot for each camera
    #         fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #         Waymo3DDataset.visualize_camera_with_lidar(
    #             camera_data_item=camera_data,
    #             lidar_points_vehicle=lidar, # Pass the vehicle-frame LiDAR points
    #             boxes_7d_vehicle=target["boxes_3d"] if not return_world else None, # Only pass vehicle-frame boxes
    #             boxes_labels_3d=target["labels"],
    #             title=f"Lidar on Camera {camera_data['camera_name']}",
    #             ax=ax
    #         )


    # --- Test BEV ---
    Waymo3DDataset.visualize_bev(
        lidar, 
        target["boxes_3d"], 
        title="TEST 1: BEV (Vehicle Frame)"
    )


    # --- Test 3D ---
    Waymo3DDataset.visualize_3d(
        lidar, 
        target["boxes_3d"], 
        target["labels"], 
        ds_vehicle.label_map_3d,
        headless=True # Set True if on a server
    )


if __name__ == "__main__":
    main()