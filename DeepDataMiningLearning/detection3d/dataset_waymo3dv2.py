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

        if num_sweeps < 1:
            raise ValueError("num_sweeps must be 1 or greater.")

        # --- Dataset folder paths ---
        self.lidar_dir = os.path.join(root_dir, split, "lidar")
        self.calib_dir = os.path.join(root_dir, split, "lidar_calibration")
        self.box_dir = os.path.join(root_dir, split, "lidar_box")
        self.vpose_dir = os.path.join(root_dir, split, "vehicle_pose")
        self.image_dir = os.path.join(root_dir, split, "camera_image")
        self.box_2d_dir = os.path.join(root_dir, split, "camera_box")

        # Check that all required directories exist
        check_dirs = [self.lidar_dir, self.calib_dir, self.box_dir, 
                      self.vpose_dir, self.image_dir]
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
            """Get LiDAR→Vehicle extrinsic (4×4, row-major)"""
            crow = df_cal[(df_cal["key.segment_context_name"] == seg) &
                          (df_cal["key.laser_name"] == lid)].iloc[0]
            extr_col = max([c for c in crow.index if "extrinsic" in str(c)], key=len)
            extr_vals = crow[extr_col]
            extr_vals = extr_vals.as_py() if hasattr(extr_vals, "as_py") else extr_vals
            return np.array(extr_vals, np.float32).reshape(4, 4, order="C")

        cand_ids = sorted(rows_t["key.laser_name"].unique().tolist()) #5 lidars: [1, 2, 3, 4, 5]
        heights = {}
        for lid in cand_ids:
            heights[lid] = float(get_T_vl(lid)[2, 3])
        laser_id = max(heights, key=heights.get) #1 is highest
        row = rows_t[rows_t["key.laser_name"] == laser_id].iloc[0] #get TOP lidar (7,)
        T_vl = get_T_vl(laser_id) # (4, 4) LiDAR -> Vehicle
        R_vl = T_vl[:3, :3]
        
        # ---------- 3️⃣ Decode the range image ----------
        ri = self._decode_range_image(row, return_id=1) #(64, 2650, 4)
        rng = np.clip(np.nan_to_num(ri[..., 0], nan=0.0), 0.0, 300.0) #(64, 2650)
        inten = ri[..., 1] if ri.shape[-1] > 1 else np.zeros_like(rng)
        H, W = rng.shape #64, 2650

        # ---------- 4️⃣ Beam inclinations ----------
        crow = df_cal[(df_cal["key.segment_context_name"] == seg) &
                      (df_cal["key.laser_name"] == laser_id)].iloc[0]
        inc_min = float(crow["[LiDARCalibrationComponent].beam_inclination.min"])
        inc_max = float(crow["[LiDARCalibrationComponent].beam_inclination.max"])
        inclinations = np.linspace(inc_min, inc_max, H, dtype=np.float32) #(64,)
        if np.max(np.abs(inclinations)) > np.pi:
            inclinations = np.deg2rad(inclinations)

        # ---------- 5️⃣ Convert Spherical → LiDAR Cartesian ----------
        incl = inclinations[::-1].reshape(H, 1) # flip vertically (64, 1)
        az = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32) #(2650,)
        cos_i, sin_i = np.cos(incl), np.sin(incl) #(64, 1)
        cos_a, sin_a = np.cos(az), np.sin(az) #(2650,)

        r"""
        Turn the range image into a 3D point cloud
        Given range $\rho$ (rng), azimuth $\alpha$ (az), and inclination $\iota$ (incl):
        $$ \\ P\_l = \begin{pmatrix} X\_l \\ Y\_l \\ Z\_l \end{pmatrix} =
        \begin{pmatrix}
        \rho \cdot \cos(\iota) \cdot \cos(\alpha) \\
        \rho \cdot \cos(\iota) \cdot \sin(\alpha) \\
        \rho \cdot \sin(\iota)
        \end{pmatrix}$$
        """
        Xl = rng * cos_i * cos_a #(64, 2650)
        Yl = rng * cos_i * sin_a #(64, 2650)
        Zl = rng * sin_i #(64, 2650)
        pts_l = np.stack([Xl, Yl, Zl, np.ones_like(Zl)], axis=-1).reshape(-1, 4) #(169600, 4)

        # ---------- 6️⃣ LiDAR → Vehicle (apply extrinsic) ----------
        r"""
        applies the sensor's extrinsic calibration
        Let $T_{v \leftarrow l}$ be the 4x4 extrinsic matrix T_vl: l to v (LiDAR Frame $\to$ Vehicle Frame (Points))
        $$ \\ P\_v = T\_{v \leftarrow l} \cdot P\_l$$
        The code uses (pts @ T.T), which is the row-vector equivalent of 
        the standard column-vector math $P_v = T_{v \leftarrow l} \cdot P_l$.
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
        print(df_box.columns.to_list()) 
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
            
            centers_l = arr[:, :3]
            sizes = arr[:, 3:6]
            headings_l = -arr[:, 6]  # clockwise → CCW

            # --- 1. Reconstruct 8 corners in LiDAR frame (where they are "flat") ---
            # Create (N, 7) box parameters in LiDAR frame
            boxes_l_7d = np.concatenate([centers_l, sizes, headings_l[:, None]], axis=1)
            
            # Use _get_box_corners to get (N, 8, 3) corners in LiDAR frame
            # This MUST be the "bottom-face" version of the function
            corners_l = self._get_box_corners(boxes_l_7d) 

            # --- 2. Transform 8 corners from LiDAR -> Vehicle ---
            # Make corners homogeneous (N*8, 4)
            corners_l_h = np.concatenate(
                [corners_l.reshape(-1, 3), np.ones((corners_l.shape[0] * 8, 1), np.float32)], 
                axis=1
            )
            # Apply T_vl transform (LiDAR -> Vehicle)
            corners_v_h = corners_l_h @ T_vl.T
            
            # Reshape back to (N, 8, 3)
            corners_v = corners_v_h[:, :3].reshape(-1, 8, 3)

            # --- 3. Optionally: Vehicle -> World ---
            if self.return_world:
                # Make vehicle corners homogeneous
                corners_v_h = np.concatenate(
                    [corners_v.reshape(-1, 3), np.ones((corners_v.shape[0] * 8, 1), np.float32)],
                    axis=1
                )
                # Apply T_wv (Vehicle -> World) transform
                corners_w_h = corners_v_h @ T_wv.T
                corners_any = corners_w_h[:, :3].reshape(-1, 8, 3)
            else:
                corners_any = corners_v
                
            # --- 4. Output corners and labels ---
            boxes_any = torch.tensor(corners_any, dtype=torch.float32) 
            labels = torch.tensor(rows_b["[LiDARBoxComponent].type"].to_numpy(), dtype=torch.int64)

        # ---------- 🔟 Assemble final output dict for this frame ----------
        target = {
            "boxes_3d": boxes_any,        # In Vehicle or World frame
            "labels": labels,
            "segment": seg,
            "timestamp": ts,
            "laser_id": int(laser_id),
            # --- Add poses for fusion ---
            "T_vl_current": torch.tensor(T_vl, dtype=torch.float32),
            "world_from_vehicle": torch.tensor(T_wv, dtype=torch.float32),
        }

        # ---------- 1️⃣1️⃣ Load Surround Images & 2D Boxes ----------
        pf_img = pq.ParquetFile(os.path.join(self.image_dir, fname))
        df_img = pf_img.read_row_group(0).to_pandas()
        rows_img = df_img[(df_img["key.segment_context_name"] == seg) &
                          (df_img["key.frame_timestamp_micros"] == ts)]
        
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
            
            # Surround view data format for each camera:
            # Each dict in surround_views list contains:
            #   - "image": torch.Tensor of shape (H, W, 3) containing RGB image data (uint8)
            #   - "boxes_2d": torch.Tensor of shape (K, 4) containing 2D bounding boxes [x_min, y_min, x_max, y_max]
            #   - "labels_2d": torch.Tensor of shape (K,) containing 2D object class IDs matching 3D labels
            #   - "camera_name": int, camera sensor ID (1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT)
            surround_views.append({
                "image": torch.tensor(img_np), "boxes_2d": torch.tensor(boxes_xyxy),
                "labels_2d": torch.tensor(labels_2d), "camera_name": cam_name,
                "camera_id": cam_id  # <-- New
            })
            
        target["surround_views"] = surround_views
        
        # ---------- 🧭 Debug summary (only for first call) ----------
        if idx == 0 and self.num_sweeps == 1:
            print(f"[DEBUG] Single-frame {idx}: {len(lidar)} pts | "
                  f"X[{XYZ[:,0].min():.1f},{XYZ[:,0].max():.1f}] "
                  f"Y[{XYZ[:,1].min():.1f},{XYZ[:,1].max():.1f}] "
                  f"Z[{XYZ[:,2].min():.1f},{XYZ[:,2].max():.1f}] "
                  f"| boxes={len(boxes_any)}")

        return lidar, target


    # ---------------------------------------------------------------------
    # ---------- VISUALIZATION & HELPER METHODS ----------
    # ---------------------------------------------------------------------

    @staticmethod
    def _get_box_corners(boxes_7d):
        """
        Converts (N, 7) 3D box format [cx, cy, cz, dx, dy, dz, heading]
        to (N, 8, 3) corners.
        
        NOTE: Assumes 'cz' is the center of the BOTTOM face.
        """
        if isinstance(boxes_7d, torch.Tensor):
            boxes_7d = boxes_7d.cpu().numpy()
        if boxes_7d.ndim == 1:
            boxes_7d = boxes_7d[np.newaxis, :]
            
        N = boxes_7d.shape[0]
        if N == 0:
            return np.zeros((0, 8, 3), dtype=np.float32)

        centers = boxes_7d[:, :3] # (N, 3) [cx, cy, cz_bottom]
        dims = boxes_7d[:, 3:6]    # (N, 3) [l, w, h]
        headings = boxes_7d[:, 6]  # (N,)

        # --- 1. Get 8 corners in local box frame (axis-aligned) ---
        l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        
        # Z-axis offsets [0, h]
        z_corners = np.concatenate([np.zeros_like(h), np.zeros_like(h), np.zeros_like(h), np.zeros_like(h), h, h, h, h], axis=1) # (N, 8)
        # X-axis offsets [-l/2, l/2]
        l_corners = np.concatenate([-l/2,  l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2], axis=1) # (N, 8)
        # Y-axis offsets [-w/2, w/2] (Waymo is +Y left)
        w_corners = np.concatenate([ w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2], axis=1) # (N, 8)

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
        # The 'centers' are [cx, cy, cz_bottom].
        # The rotated local corners have z=[0, h].
        # Adding them together places the box correctly from cz_bottom to cz_bottom + h.
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

    # ( ... Other visualization methods: visualize_range_image, visualize_bev, 
    #   visualize_3d, visualize_surround_view ... )
    # (These are unchanged from your previous versions)
    # ---------------------------------------------------------------------
    # ---------- START: 2D VISUALIZATION METHOD ----------
    # ---------------------------------------------------------------------

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
    def visualize_3d(lidar_points, boxes_8_corners=None, labels=None, label_map=None,
                     headless=False, save_path="scene_vis.ply"):
        """
        Visualizes LiDAR points and 3D boxes using Open3D.

        - Points are colored by height (Z-axis).
        - A coordinate frame is drawn at the origin.
        - 3D labels are drawn on top of boxes.
        - In headless mode, saves geometry to .ply files (labels are lost).
        
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
        if isinstance(boxes_8_corners, torch.Tensor):
            boxes_8_corners = boxes_8_corners.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # --- 2. Create Point Cloud Geometry ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
        
        # Color by Z-height
        z_values = lidar_points[:, 2]
        if z_values.max() > z_values.min():
            z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min())
        else:
            z_norm = np.zeros_like(z_values)
        colors = plt.get_cmap("jet")(z_norm)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # --- 3. Create Coordinate Frame Geometry ---
        # X=Red, Y=Green, Z=Blue
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.5, origin=[0, 0, 0]
        )

        # --- 4. Create Box & Label Geometries ---
        box_geometries = []
        box_labels_3d = [] # For O3DVisualizer
        
        if boxes_8_corners is not None and len(boxes_8_corners) > 0:
            # --- FIX: Input is now corners, not 7D boxes ---
            if isinstance(boxes_8_corners, torch.Tensor):
                corners_3d = boxes_8_corners.cpu().numpy() # (M, 8, 3)
            else:
                corners_3d = boxes_8_corners
            
            # --- REMOVED THE CALL TO _get_box_corners ---

        
            # Standard 12 lines for a box
            # This order assumes the corner_offsets from your _get_box_corners:
            # [(-1,1,-1), (1,1,-1), (1,-1,-1), (-1,-1,-1),
            #  (-1,1,1), (1,1,1), (1,-1,1), (-1,-1,1)]
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0], # Bottom 4
                [4, 5], [5, 6], [6, 7], [7, 4], # Top 4
                [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical
            ]
            
            # Use different colors for different classes
            label_colors = {
                1: [0, 1, 0], # Vehicle (Green)
                2: [1, 0, 0], # Pedestrian (Red)
                4: [1, 1, 0], # Cyclist (Yellow)
            }
            default_color = [0.5, 0.5, 0.5] # Other (Gray)

            for i in range(corners_3d.shape[0]):
                box_corners = corners_3d[i]
                label_id = int(labels[i]) if labels is not None else 0
                
                # --- Create LineSet for the box ---
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(box_corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                color = label_colors.get(label_id, default_color)
                line_set.paint_uniform_color(color)
                box_geometries.append(line_set)
                
                # --- Prepare 3D Label ---
                label_text = str(label_id)
                if label_map and label_id in label_map:
                    label_text = label_map[label_id]
                
                # Position label at the top-center of the box
                label_pos = box_corners[4:].mean(axis=0) + [0, 0, 0.2] # 20cm above top
                box_labels_3d.append((label_pos, label_text))

        # --- 5. Combine Geometries ---
        geometries = [pcd, mesh_frame] + box_geometries

        # --- 6. Execute Visualization or Save ---
        if headless:
            print(f"[INFO] Headless mode: Saving {len(geometries)} geometries to {save_path} prefix.")
            print("       NOTE: 3D text labels cannot be saved to .ply files.")
            
            # --- Save to File ---
            # We must save different geometry types to different files.
            # .ply supports PointCloud and TriangleMesh (but not LineSet well)
            points_path = save_path.replace(".ply", "_points.ply")
            o3d.io.write_point_cloud(points_path, pcd)
            print(f"       ... Saved points to {points_path}")

            # For boxes, we must convert LineSet to a TriangleMesh (thin cylinders)
            # Combine frame and all box meshes into one
            scene_mesh = mesh_frame
            for i, line_set in enumerate(box_geometries):
                # create_from_line_set is only in o3d 0.17+
                # A more compatible way is to just save the line sets
                box_path = save_path.replace(".ply", f"_box_{i:03d}.ply")
                o3d.io.write_line_set(box_path, line_set)
            
            frame_path = save_path.replace(".ply", "_frame.ply")
            o3d.io.write_triangle_mesh(frame_path, mesh_frame)
            print(f"       ... Saved boxes and frame to multiple files.")

        else:
            # --- Open Interactive Window ---
            print("[INFO] Opening 3D visualizer... (Press 'Q' to close)")
            try:
                # Use the new (v0.15+) O3DVisualizer to support 3D labels
                app = gui.Application.instance
                app.initialize()

                vis = o3d.visualization.O3DVisualizer("Waymo 3D Visualization", 1280, 720)
                
                # Add geometries with unique names
                vis.add_geometry("points", pcd)
                vis.add_geometry("frame", mesh_frame)
                for i, box_geom in enumerate(box_geometries):
                    vis.add_geometry(f"box_{i}", box_geom)
                
                # Add 3D labels
                for i, (pos, text) in enumerate(box_labels_3d):
                    vis.add_3d_label(pos, text)
                    
                # Set camera view (optional, but nice)
                # (Point cloud center, look-at, up-vector)
                vis.setup_camera(60.0, pcd.get_center(), [0, 0, 30], [0, 0, 1])
                vis.reset_camera_to_default()

                app.run()
                
            except Exception as e:
                print(f"[ERROR] Failed to start O3DVisualizer: {e}")
                print("       Falling back to simple visualizer (no labels).")
                o3d.visualization.draw_geometries(geometries)
    
    @staticmethod
    def visualize_bev(lidar_points, boxes_8_corners=None, ax=None, 
                      point_size=0.1, range_m=80.0, title="BEV Visualization"):
        """
        Visualizes LiDAR points and boxes in Bird's-Eye View (BEV).
        Assumes points and boxes are in the *same* coordinate frame.
        
        Args:
            lidar_points: (N, 3+) torch.Tensor or np.ndarray (XYZ...)
            boxes_7d: (M, 7) torch.Tensor or np.ndarray [cx,cy,cz,dx,dy,dz,h]
            ax: Optional matplotlib axis.
            point_size: Size of plotted points.
            range_m: Plot range. Can be float (e.g., 80.0 for +/-80m) 
                     or tuple ((-x,x), (-y,y)).
            title: Plot title.
        """
        
        # 1. --- Handle Inputs ---
        if isinstance(lidar_points, torch.Tensor):
            lidar_points = lidar_points.cpu().numpy()
        if isinstance(boxes_8_corners, torch.Tensor):
            boxes_8_corners = boxes_8_corners.cpu().numpy()

        # 2. --- Setup Plot ---
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.set_aspect('equal')
        
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

        # 3. --- Plot Ego Vehicle ---
        # (Assumes 0,0 is the origin, which is true for Vehicle Frame)
        ax.plot(0, 0, 'rx', markersize=10, mew=2, label="Ego Vehicle")
        
        # 4. --- Plot LiDAR Points ---
        # Filter points outside the range to speed up plotting
        mask = (lidar_points[:, 0] >= x_range[0]) & (lidar_points[:, 0] <= x_range[1]) & \
               (lidar_points[:, 1] >= y_range[0]) & (lidar_points[:, 1] <= y_range[1])
        points_filtered = lidar_points[mask]
        
        # Use intensity for color if available (assuming 4th col)
        if points_filtered.shape[1] > 3:
            colors = points_filtered[:, 3] # Intensity
            cmap = 'gray'
        else:
            colors = 'k'
            cmap = None
            
        ax.scatter(points_filtered[:, 0], points_filtered[:, 1], 
                   s=point_size, c=colors, cmap=cmap, alpha=0.5)

        # 5. --- Plot Boxes ---
        if boxes_8_corners is not None and len(boxes_8_corners) > 0:
            if isinstance(boxes_8_corners, torch.Tensor):
                corners_3d = boxes_8_corners.cpu().numpy() # (M, 8, 3)
            else:
                corners_3d = boxes_8_corners
            
            # --- REMOVED THE CALL TO _get_box_corners ---

            # Get the bottom 4 corners (XY plane)
            # This assumes the corner ordering from your _get_box_corners:
            # 0: back-left-bottom
            # 1: front-left-bottom
            # 2: front-right-bottom
            # 3: back-right-bottom
            bev_corners = corners_3d[:, [0, 1, 2, 3], :2] # (M, 4, 2)
            
            for i in range(bev_corners.shape[0]):
                # Draw the polygon
                ax.add_patch(patches.Polygon(bev_corners[i],
                                             closed=True,
                                             color='b',
                                             fill=False,
                                             linewidth=1.5))
                
                # Draw heading indicator (line from center to front-mid)
                front_mid = bev_corners[i, [1, 2], :].mean(axis=0)
                center = bev_corners[i].mean(axis=0) # Get center of bottom face
                ax.plot([center[0], front_mid[0]],
                        [center[1], front_mid[1]],
                        'b-', linewidth=1.5, alpha=0.8)

        ax.legend(loc='upper right')
        
        if 'fig' in locals():
            plt.show()

def main():

    DATA_ROOT = "/data/Datasets/waymodata/" # ⚠️ Update this path
    
    # ==================================================================
    print("\n\n========== TEST 1: Vehicle Frame (Single Sweep) ==========")
    # ==================================================================
    ds_vehicle = Waymo3DDataset(
        DATA_ROOT, 
        split="training", 
        max_frames=80, 
        return_world=False,
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
        headless=False # Set True if on a server
    )

    # --- Test Trajectory (needs manual transform to world) ---
    T_wv_current = target["T_wv_sweeps"][-1]
    pts_v_h = torch.cat([lidar[:,:3], torch.ones_like(lidar[:,:1])], dim=-1)
    pts_w = (pts_v_h.float() @ T_wv_current.T)[..., :3]
    Waymo3DDataset.visualize_trajectory_bev(
        pts_w, 
        target["T_wv_sweeps"],
        title="TEST 1: Trajectory (from Vehicle Frame)"
    )

    # =================================================================
    print("\n\n========== TEST 2: World Frame (Single Sweep) ==========")
    # =================================================================
    ds_world = Waymo3DDataset(
        DATA_ROOT, 
        split="training", 
        max_frames=3, 
        return_world=True, # <-- CHANGED
        num_sweeps=1
    )
    lidar_w, target_w = ds_world[0]

    print("\n--- VISUALIZING (World Frame) ---")
    
    # --- Test BEV (World) ---
    # Note: BEV will look offset because points are centered at (0,0) in Vehicle,
    # but are now at their true global (X,Y) coordinates.
    Waymo3DDataset.visualize_bev(
        lidar_w, 
        target_w["boxes_3d"], 
        title="TEST 2: BEV (World Frame)",
        # Use trajectory function for better auto-ranging in world coords
    )

    # --- Test 3D (World) ---
    Waymo3DDataset.visualize_3d(
        lidar_w, 
        target_w["boxes_3d"], 
        target_w["labels"], 
        ds_world.label_map_3d,
        headless=False
    )

    # --- Test Trajectory (World) ---
    # This is the most direct test for world coordinates
    Waymo3DDataset.visualize_trajectory_bev(
        lidar_w, # Already in world frame
        target_w["T_wv_sweeps"],
        title="TEST 2: Trajectory (World Frame)"
    )


    # =================================================================
    print("\n\n========== TEST 3: Vehicle Frame (Multi-Sweep) ==========")
    # =================================================================
    ds_multi = Waymo3DDataset(
        DATA_ROOT, 
        split="training", 
        max_frames=10, # Need a few frames to sweep
        return_world=False,
        num_sweeps=5     # <-- CHANGED
    )
    # Get a frame index > num_sweeps to ensure a full sweep buffer
    lidar_m, target_m = ds_multi[5] 

    print("\n========== BASIC INFO (Multi-Sweep) ==========")
    print("points:", lidar_m.shape) # Should have many more points
    print("boxes :", target_m["boxes_3d"].shape)
    
    inten_m = lidar_m[:, 3].numpy()
    time_m = lidar_m[:, 4].numpy()
    print(f"Intensity range: {inten_m.min():.3f} → {inten_m.max():.3f}")
    # Time should now be negative for past sweeps
    print(f"Time delta range: {time_m.min():.3f}s → {time_m.max():.3f}s")
    
    print("\n--- VISUALIZING (Multi-Sweep) ---")

    # --- Test BEV (Multi-Sweep) ---
    Waymo3DDataset.visualize_bev(
        lidar_m, 
        target_m["boxes_3d"], 
        title="TEST 3: BEV (Multi-Sweep, Vehicle Frame)"
    )

    # --- Test 3D (Multi-Sweep) ---
    Waymo3DDataset.visualize_3d(
        lidar_m, 
        target_m["boxes_3d"], 
        target_m["labels"], 
        ds_multi.label_map_3d,
        headless=False
    )
    
    # --- Test Trajectory (Multi-Sweep) ---
    T_wv_current_m = target_m["T_wv_sweeps"][-1]
    pts_v_h_m = torch.cat([lidar_m[:,:3], torch.ones_like(lidar_m[:,:1])], dim=-1)
    pts_w_m = (pts_v_h_m.float() @ T_wv_current_m.T)[..., :3]
    Waymo3DDataset.visualize_trajectory_bev(
        pts_w_m, 
        target_m["T_wv_sweeps"],
        title="TEST 3: Trajectory (Multi-Sweep)"
    )


if __name__ == "__main__":
    main()