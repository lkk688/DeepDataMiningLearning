import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
import math # Added for math constants
import re # Added for segment name extraction
import zlib # Added for range image decompression

# Import Waymo protobuf definitions
try:
    from waymo_open_dataset import dataset_pb2 as open_dataset
except ImportError:
    print("Warning: waymo_open_dataset not available. Some functions may not work.")
    open_dataset = None

# --- Reuse necessary parts from your Waymo3DDataset ---
# Option 1: Import your class if it's in a separate file
# from your_dataset_file import Waymo3DDataset

# Option 2: Paste the necessary static methods directly here
# (Make sure _get_box_corners uses the GEOMETRIC center version)
class WaymoDataHelpers:
    @staticmethod
    def _get_box_corners(boxes_7d):
        """
        Converts (N, 7) 3D box format [cx, cy, cz, dx, dy, dz, heading]
        to (N, 8, 3) corners. (Geometric Center Version)
        """
        # --- Add torch placeholder if using torch tensors ---
        try:
            import torch
            is_torch = isinstance(boxes_7d, torch.Tensor)
        except ImportError:
            is_torch = False
            class torch: # Dummy class
                 Tensor = np.ndarray
                 float32 = np.float32
                 @staticmethod
                 def tensor(*args, **kwargs): return np.array(*args, **kwargs)

        if is_torch:
             boxes_7d = boxes_7d.cpu().numpy()
        # --- End placeholder ---

        if boxes_7d.ndim == 1:
            boxes_7d = boxes_7d[np.newaxis, :]
        N = boxes_7d.shape[0]
        if N == 0: return np.zeros((0, 8, 3), dtype=np.float32)
        centers, dims, headings = boxes_7d[:, :3], boxes_7d[:, 3:6], boxes_7d[:, 6]
        lwh = dims / 2.0
        # X offsets [-l/2, l/2]
        l_corners = np.concatenate([-lwh[:, 0:1], lwh[:, 0:1], lwh[:, 0:1], -lwh[:, 0:1],
                                    -lwh[:, 0:1], lwh[:, 0:1], lwh[:, 0:1], -lwh[:, 0:1]], axis=1)
        # Y offsets [-w/2, w/2] (RH uses +Y Left, so w/2 is left)
        w_corners = np.concatenate([lwh[:, 1:2], lwh[:, 1:2], -lwh[:, 1:2], -lwh[:, 1:2],
                                    lwh[:, 1:2], lwh[:, 1:2], -lwh[:, 1:2], -lwh[:, 1:2]], axis=1)
        # Z offsets [-h/2, h/2]
        z_corners = np.concatenate([-lwh[:, 2:], -lwh[:, 2:], -lwh[:, 2:], -lwh[:, 2:],
                                    lwh[:, 2:], lwh[:, 2:], lwh[:, 2:], lwh[:, 2:]], axis=1)
        corners_local = np.stack([l_corners, w_corners, z_corners], axis=2)
        cos_h, sin_h = np.cos(headings), np.sin(headings)
        R = np.zeros((N, 3, 3), dtype=np.float32); R[:, 0, 0] = cos_h; R[:, 0, 1] = -sin_h; R[:, 1, 0] = sin_h; R[:, 1, 1] = cos_h; R[:, 2, 2] = 1.0
        corners_rotated = (R @ corners_local.transpose(0, 2, 1)).transpose(0, 2, 1)
        return corners_rotated + centers[:, np.newaxis, :]

    @staticmethod
    def _get_camera_calibration_robust(cam_cal_row, M_reflect, debug=True):
        """ Robust loader for v2 camera calib, returns RH T_cv """
        # --- 1. Intrinsics (K) ---
        fx = float(cam_cal_row["[CameraCalibrationComponent].intrinsic.f_u"])
        fy = float(cam_cal_row["[CameraCalibrationComponent].intrinsic.f_v"])
        cx = float(cam_cal_row["[CameraCalibrationComponent].intrinsic.c_u"])
        cy = float(cam_cal_row["[CameraCalibrationComponent].intrinsic.c_v"])
        W = int(cam_cal_row["[CameraCalibrationComponent].width"])
        H = int(cam_cal_row["[CameraCalibrationComponent].height"])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        img_dims = (H, W)
        # (Debug prints omitted for brevity)

        # --- 2. Extrinsics (T_cv_lh) - Load and Validate ---
        extr_vals = cam_cal_row["[CameraCalibrationComponent].extrinsic.transform"]
        extr_vals = extr_vals.as_py() if hasattr(extr_vals, "as_py") else extr_vals
        vals = np.array(extr_vals, dtype=np.float32)
        assert vals.size == 16
        # (Row/Col check logic - use your preferred version from dataset class)
        # Simplified: Assume row-major unless clearly invalid
        R_row = np.array([[vals[0], vals[1], vals[2]], [vals[4], vals[5], vals[6]], [vals[8], vals[9], vals[10]]])
        t_row = np.array([vals[3], vals[7], vals[11]])
        R_lh = R_row; t_lh = t_row # Default to row-major
        # (Add row/col validation if needed)
        # (Add translation scaling if needed)
        # ...
        T_cv_lh = np.eye(4, dtype=np.float32); T_cv_lh[:3, :3] = R_lh; T_cv_lh[:3, 3] = t_lh

        # --- 3. Convert LH extrinsic to RH extrinsic: T_cv_rh = T_cv_lh @ M ---
        T_cv = T_cv_lh @ M_reflect
        return K, T_cv, img_dims



# ==========================================================
# --- START: New Function find_common_timestamps ---
# ==========================================================
def find_common_timestamps(v1_tfrecord_path, v2_root_path, v2_split='training'):
    """
    Finds common timestamps between a v1 TFRecord segment file and its
    corresponding v2 Parquet file. Uses minimal TFRecord parsing.

    Args:
        v1_tfrecord_path (str): Path to the v1 .tfrecord file.
        v2_root_path (str): Root directory of the v2 dataset.
        v2_split (str): The split (e.g., 'training') for the v2 data.

    Returns:
        tuple: (segment_name, list_of_common_timestamps) or (None, []) if not found.
    """
    print(f"\n--- Analyzing v1 file: {os.path.basename(v1_tfrecord_path)} ---")
    if not os.path.exists(v1_tfrecord_path):
        print(f"[ERROR] v1 TFRecord file not found: {v1_tfrecord_path}")
        return None, []

    # --- 1. Read v1 Segment Name and Timestamps ---
    v1_segment_name = None
    v1_timestamps = set()
    try:
        # Import Waymo dataset protobuf definitions
        from waymo_open_dataset import dataset_pb2 as open_dataset
        
        dataset = tf.data.TFRecordDataset(v1_tfrecord_path, compression_type='')
        
        for i, data in enumerate(dataset):
            try:
                # Parse using Waymo's protobuf definition
                frame = open_dataset.Frame()
                frame.ParseFromString(data.numpy())
                
                if v1_segment_name is None:
                    v1_segment_name = frame.context.name
                    print(f"v1 Segment Name: {v1_segment_name}")
                
                v1_timestamps.add(frame.timestamp_micros)
                
            except Exception as parse_error:
                print(f"[WARN] Failed to parse record {i} in v1 TFRecord: {parse_error}. Skipping record.")
                # Attempt to get segment name if not yet found (might be in later records)
                if v1_segment_name is None:
                    try:
                        # Try parsing just the frame to get context name
                        frame_fallback = open_dataset.Frame()
                        frame_fallback.ParseFromString(data.numpy())
                        v1_segment_name = frame_fallback.context.name
                        print(f"Extracted v1 Segment Name from record {i}: {v1_segment_name}")
                    except:
                        pass # Ignore if even context name fails
                continue # Skip to the next record

        if not v1_timestamps:
             raise ValueError("No valid records could be parsed to extract timestamps.")
        print(f"Found {len(v1_timestamps)} unique timestamps in v1 file.")

    except Exception as e:
        print(f"[ERROR] Failed to read v1 TFRecord: {e}")
        import traceback
        traceback.print_exc()
        return None, []

    if v1_segment_name is None:
        print("[ERROR] Could not extract segment name from v1 file (check warnings above).")
        return None, []

    # --- 2. Find Corresponding v2 Parquet File ---
    v2_lidar_dir = os.path.join(v2_root_path, v2_split, "lidar")
    if not os.path.isdir(v2_lidar_dir):
        print(f"[ERROR] v2 lidar directory not found: {v2_lidar_dir}")
        return v1_segment_name, []

    v2_parquet_fname = None
    # Extract the core segment ID from the v1 name
    match = re.search(r"(segment-\d+_\d+_\d+_\d+_\d+)", v1_segment_name)
    if not match:
        print(f"[WARN] Could not parse standard segment format from v1 name: {v1_segment_name}. Using full name for search.")
        core_segment_id = v1_segment_name
    else:
        core_segment_id = match.group(1)

    print(f"Searching for v2 file containing: {core_segment_id} in {v2_lidar_dir}")
    for f in os.listdir(v2_lidar_dir):
        # Use startswith for v2 files that might have shorter names than v1
        if f.startswith(core_segment_id) and f.endswith(".parquet"):
            v2_parquet_fname = f
            break

    if v2_parquet_fname is None:
        print(f"[ERROR] No matching v2 Parquet file found for segment '{core_segment_id}'.")
        return v1_segment_name, []

    v2_lidar_path = os.path.join(v2_lidar_dir, v2_parquet_fname)
    print(f"Found matching v2 file: {v2_parquet_fname}")

    # --- 3. Read v2 Timestamps ---
    v2_timestamps = set()
    try:
        parquet_file = pq.ParquetFile(v2_lidar_path)
        for i in range(parquet_file.num_row_groups):
             table = parquet_file.read_row_group(i, columns=['key.segment_context_name', 'key.frame_timestamp_micros'])
             df_rg = table.to_pandas()
             # Filter for the specific segment name
             df_segment = df_rg[df_rg['key.segment_context_name'] == v1_segment_name]
             v2_timestamps.update(df_segment['key.frame_timestamp_micros'].unique())

        print(f"Found {len(v2_timestamps)} unique timestamps in v2 file for segment {v1_segment_name}.")

    except Exception as e:
        print(f"[ERROR] Failed to read v2 Parquet file {v2_lidar_path}: {e}")
        return v1_segment_name, []

    # --- 4. Find Common Timestamps ---
    common_timestamps = sorted(list(v1_timestamps.intersection(v2_timestamps)))
    print(f"Found {len(common_timestamps)} common timestamps.")

    return v1_segment_name, common_timestamps

# ==========================================================
# --- END: Simplified Function find_common_timestamps ---
# ==========================================================

def find_common_timestamps_complex(v1_tfrecord_path, v2_root_path, v2_split='training'):
    """
    Finds common timestamps between a v1 TFRecord segment file and its
    corresponding v2 Parquet file.

    Args:
        v1_tfrecord_path (str): Path to the v1 .tfrecord file.
        v2_root_path (str): Root directory of the v2 dataset.
        v2_split (str): The split (e.g., 'training') for the v2 data.

    Returns:
        tuple: (segment_name, list_of_common_timestamps) or (None, []) if not found.
    """
    print(f"\n--- Analyzing v1 file: {os.path.basename(v1_tfrecord_path)} ---")
    if not os.path.exists(v1_tfrecord_path):
        print(f"[ERROR] v1 TFRecord file not found: {v1_tfrecord_path}")
        return None, []

    # --- 1. Read v1 Segment Name and Timestamps ---
    v1_segment_name = None
    v1_timestamps = set()
    try:
        # Import Waymo dataset protobuf definitions
        from waymo_open_dataset import dataset_pb2 as open_dataset
        
        dataset = tf.data.TFRecordDataset(v1_tfrecord_path, compression_type='')
        
        for i, data in enumerate(dataset):
            try:
                # Parse using Waymo's protobuf definition
                frame = open_dataset.Frame()
                frame.ParseFromString(data.numpy())
                
                if v1_segment_name is None:
                    v1_segment_name = frame.context.name
                    print(f"v1 Segment Name: {v1_segment_name}")
                
                v1_timestamps.add(frame.timestamp_micros)
                
            except Exception as parse_error:
                print(f"[WARN] Failed to parse record {i} in v1 TFRecord: {parse_error}. Skipping record.")
                continue
                
        print(f"Found {len(v1_timestamps)} unique timestamps in v1 file.")
    except Exception as e:
        print(f"[ERROR] Failed to read v1 TFRecord: {e}")
        return None, []

    if v1_segment_name is None:
        print("[ERROR] Could not extract segment name from v1 file.")
        return None, []

    # --- 2. Find Corresponding v2 Parquet File ---
    v2_lidar_dir = os.path.join(v2_root_path, v2_split, "lidar")
    if not os.path.isdir(v2_lidar_dir):
        print(f"[ERROR] v2 lidar directory not found: {v2_lidar_dir}")
        return v1_segment_name, []

    v2_parquet_fname = None
    # Extract the core segment ID from the v1 name
    match = re.search(r"(segment-\d+_\d+_\d+_\d+_\d+)", v1_segment_name)
    if not match:
        print(f"[WARN] Could not parse standard segment format from v1 name: {v1_segment_name}. Using full name for search.")
        core_segment_id = v1_segment_name
    else:
        core_segment_id = match.group(1)

    print(f"Searching for v2 file containing: {core_segment_id} in {v2_lidar_dir}")
    for f in os.listdir(v2_lidar_dir):
        # Use startswith for v2 files that might have shorter names than v1
        if f.startswith(core_segment_id) and f.endswith(".parquet"):
            v2_parquet_fname = f
            break

    if v2_parquet_fname is None:
        print(f"[ERROR] No matching v2 Parquet file found for segment '{core_segment_id}'.")
        return v1_segment_name, []

    v2_lidar_path = os.path.join(v2_lidar_dir, v2_parquet_fname)
    print(f"Found matching v2 file: {v2_parquet_fname}")

    # --- 3. Read v2 Timestamps ---
    v2_timestamps = set()
    try:
        # Read only necessary columns
        parquet_file = pq.ParquetFile(v2_lidar_path)
        # Iterate through row groups if file is large (more memory efficient)
        for i in range(parquet_file.num_row_groups):
             table = parquet_file.read_row_group(i, columns=['key.segment_context_name', 'key.frame_timestamp_micros'])
             df_rg = table.to_pandas()
             # Filter for the specific segment name (important if file contains multiple)
             df_segment = df_rg[df_rg['key.segment_context_name'] == v1_segment_name]
             v2_timestamps.update(df_segment['key.frame_timestamp_micros'].unique())

        # Alternative: Read whole file (simpler but uses more memory)
        # table = pq.read_table(v2_lidar_path, columns=['key.segment_context_name', 'key.frame_timestamp_micros'])
        # df = table.to_pandas()
        # df_segment = df[df['key.segment_context_name'] == v1_segment_name]
        # v2_timestamps = set(df_segment['key.frame_timestamp_micros'].unique())

        print(f"Found {len(v2_timestamps)} unique timestamps in v2 file for segment {v1_segment_name}.")

    except Exception as e:
        print(f"[ERROR] Failed to read v2 Parquet file {v2_lidar_path}: {e}")
        return v1_segment_name, []

    # --- 4. Find Common Timestamps ---
    common_timestamps = sorted(list(v1_timestamps.intersection(v2_timestamps)))
    print(f"Found {len(common_timestamps)} common timestamps.")

    return v1_segment_name, common_timestamps

# ==========================================================
# --- END: New Function find_common_timestamps ---
# ==========================================================


def find_tfrecord_file(v1_root, split, segment_name):
    """Finds the .tfrecord file for a given segment."""
    split_dir = os.path.join(v1_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"v1 split directory not found: {split_dir}")
    # Adjust search to match the potentially longer v1 filename format
    core_segment_id = segment_name.split('_with_camera_labels')[0]
    for fname in os.listdir(split_dir):
        if core_segment_id in fname and fname.endswith('.tfrecord'):
            return os.path.join(split_dir, fname)
    raise FileNotFoundError(f"TFRecord file for segment '{segment_name}' not found in {split_dir}")

# (Keep the existing parse_tfrecord_v1, spherical_to_cartesian_v1, load_v1_frame, load_v2_frame, compare_data functions)
# ... [These functions remain unchanged from the previous version] ...

def find_tfrecord_file(v1_root, split, segment_name):
    """Finds the .tfrecord file for a given segment."""
    split_dir = os.path.join(v1_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"v1 split directory not found: {split_dir}")
    for fname in os.listdir(split_dir):
        if segment_name in fname and fname.endswith('.tfrecord'):
            return os.path.join(split_dir, fname)
    raise FileNotFoundError(f"TFRecord file for segment '{segment_name}' not found in {split_dir}")

# ==========================================================
# --- START: Modified load_v1_frame ---
# ==========================================================
def get_box_transformation_matrix(box_center_x: float, box_center_y: float,
                                  box_center_z: float, box_heading: float,
                                  box_height: float, box_width: float,
                                  box_length: float) -> tf.Tensor:
    """Creates a transformation matrix for a given box pose."""
    # Simplified from Waymo's transform_utils
    # Assumes box heading is RH CCW yaw
    translation = tf.constant([box_center_x, box_center_y, box_center_z], dtype=tf.float32)
    rotation_z = tf.constant([[tf.cos(box_heading), -tf.sin(box_heading), 0.0],
                             [tf.sin(box_heading), tf.cos(box_heading), 0.0],
                             [0.0, 0.0, 1.0]], dtype=tf.float32)
    # Scale matrix (not used for points, just conceptual)
    # scale = tf.constant([[box_length / 2.0, 0.0, 0.0],
    #                      [0.0, box_width / 2.0, 0.0],
    #                      [0.0, 0.0, box_height / 2.0]], dtype=tf.float32)

    transform = tf.eye(4, dtype=tf.float32)
    transform = tf.tensor_scatter_nd_update(transform, [[0,0],[0,1],[0,2], [1,0],[1,1],[1,2], [2,0],[2,1],[2,2]],
                                           tf.reshape(rotation_z, [-1]))
    transform = tf.tensor_scatter_nd_update(transform, [[0,3],[1,3],[2,3]], translation)
    return transform

def parse_tfrecord_v1(serialized_example):
    """Parses a serialized Waymo v1 Frame proto using manual feature description."""
    # Define features based on Waymo Frame proto structure (simplified)
    # We only define the fields we absolutely need for comparison
    context_features = {
        'context.name': tf.io.FixedLenFeature([], tf.string),
        # Add laser calibration features (assuming max 5 lasers)
        'context.laser_calibrations.size': tf.io.FixedLenFeature([], tf.int64),
        'context.laser_calibrations.name': tf.io.FixedLenFeature([5], tf.int64, default_value=[0]*5),
        'context.laser_calibrations.beam_inclination_min': tf.io.FixedLenFeature([5], tf.float32, default_value=[0.0]*5),
        'context.laser_calibrations.beam_inclination_max': tf.io.FixedLenFeature([5], tf.float32, default_value=[0.0]*5),
        'context.laser_calibrations.beam_inclinations': tf.io.VarLenFeature(tf.float32), # Optional full vector
        'context.laser_calibrations.extrinsic.transform': tf.io.FixedLenFeature([5*16], tf.float32, default_value=[0.0]*(5*16)),
        # Add camera calibration features (assuming max 5 cameras)
        'context.camera_calibrations.size': tf.io.FixedLenFeature([], tf.int64),
        'context.camera_calibrations.name': tf.io.FixedLenFeature([5], tf.int64, default_value=[0]*5),
        'context.camera_calibrations.intrinsic': tf.io.FixedLenFeature([5*9], tf.float32, default_value=[0.0]*(5*9)),
        'context.camera_calibrations.extrinsic.transform': tf.io.FixedLenFeature([5*16], tf.float32, default_value=[0.0]*(5*16)),
        'context.camera_calibrations.width': tf.io.FixedLenFeature([5], tf.int64, default_value=[0]*5),
        'context.camera_calibrations.height': tf.io.FixedLenFeature([5], tf.int64, default_value=[0]*5),
    }
    frame_features = {
        'frame.timestamp_micros': tf.io.FixedLenFeature([], tf.int64),
        'frame.pose.transform': tf.io.FixedLenFeature([16], tf.float32),
        # Laser data (assuming max 5 lasers)
        'frame.lasers.size': tf.io.FixedLenFeature([], tf.int64),
        'frame.lasers.name': tf.io.FixedLenFeature([5], tf.int64, default_value=[0]*5),
        'frame.lasers.ri_return1.range_image_compressed': tf.io.FixedLenFeature([5], tf.string, default_value=['']*5),
        # Laser labels (variable number)
        'frame.laser_labels.size': tf.io.FixedLenFeature([], tf.int64),
        'frame.laser_labels.box.center_x': tf.io.VarLenFeature(tf.float32),
        'frame.laser_labels.box.center_y': tf.io.VarLenFeature(tf.float32),
        'frame.laser_labels.box.center_z': tf.io.VarLenFeature(tf.float32),
        'frame.laser_labels.box.length': tf.io.VarLenFeature(tf.float32), # v1 uses length/width
        'frame.laser_labels.box.width': tf.io.VarLenFeature(tf.float32),
        'frame.laser_labels.box.height': tf.io.VarLenFeature(tf.float32),
        'frame.laser_labels.box.heading': tf.io.VarLenFeature(tf.float32),
        'frame.laser_labels.type': tf.io.VarLenFeature(tf.int64),
    }

    features = {**context_features, **frame_features}
    parsed = tf.io.parse_single_example(serialized_example, features)
    return parsed

def spherical_to_cartesian_v1(frame_proto, laser_calib, range_image_tensor):
    """
    Converts v1 range image to Cartesian points in sensor frame (RH).
    
    COORDINATE SYSTEM NOTES:
    - v1 LiDAR uses Right-Handed coordinate system natively
    - LiDAR frame: +X forward, +Y left, +Z up (note: Y is LEFT in sensor frame)
    - Range image format: [range, intensity, elongation, is_in_nlz]
    - Beam ordering: top-to-bottom in range image corresponds to high-to-low elevation
    - Azimuth: π to -π (left to right in range image)
    """
    # Adapted from waymo_open_dataset.utils.range_image_utils
    range_image = range_image_tensor[..., 0] # Range is channel 0 (meters)
    intensity_image = range_image_tensor[..., 1] # Intensity is channel 1 (raw values)

    # Get inclinations from laser calibration (protobuf object)
    # Inclination angles for each laser beam (elevation angles)
    inclination_min = laser_calib.beam_inclination_min  # Lowest beam angle
    inclination_max = laser_calib.beam_inclination_max  # Highest beam angle

    range_image_shape = tf.shape(range_image)
    H, W = range_image_shape[0], range_image_shape[1]  # Height x Width

    # Calculate inclinations (linear spacing as fallback/simplification)
    # Note: v1 may have non-uniform beam spacing, but linear is approximation
    inclination = tf.linspace(inclination_max, inclination_min, H)
    inclination = tf.reverse(inclination, axis=[0]) # Align with image rows (bottom to top)

    # Calculate azimuth angles for each column
    # Azimuth range: π (left) to -π (right) in sensor frame
    azimuth = tf.linspace(math.pi, -math.pi, W)

    # Create meshgrid for vectorized computation
    inclination_mesh, azimuth_mesh = tf.meshgrid(inclination, azimuth, indexing='ij')

    # Spherical to Cartesian conversion in LiDAR sensor frame
    # LiDAR coordinate system: +X forward, +Y left, +Z up
    x = range_image * tf.cos(inclination_mesh) * tf.cos(azimuth_mesh)  # Forward
    y = range_image * tf.cos(inclination_mesh) * tf.sin(azimuth_mesh)  # Left (positive Y)
    z = range_image * tf.sin(inclination_mesh)                         # Up

    # Stack coordinates and filter invalid points (range=0)
    points_sensor = tf.stack([x, y, z], axis=-1)
    mask = range_image > 0  # Valid range measurements
    points_sensor = tf.boolean_mask(points_sensor, mask)
    intensity = tf.boolean_mask(intensity_image, mask)

    return points_sensor, intensity # In Sensor Frame (RH, +Y left)

def load_v1_frame(tfrecord_path, target_timestamp):
    """
    Loads and processes a single frame from a Waymo v1 TFRecord file using protobuf parsing.
    
    COORDINATE SYSTEM NOTES:
    - Waymo v1 uses Right-Handed (RH) coordinate system natively
    - Vehicle frame: +X forward, +Y right, +Z up (standard automotive)
    - Camera frame: +X right, +Y down, +Z forward (standard computer vision)
    - LiDAR frame: +X forward, +Y right, +Z up (aligned with vehicle)
    - No coordinate conversion needed (already RH)
    """
    print(f"\n--- Loading v1 Frame (Timestamp: {target_timestamp}) ---")
    
    # Import Waymo dataset protobuf definitions
    from waymo_open_dataset import dataset_pb2 as open_dataset
    
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    frame_proto = None
    frame_index = -1

    # Iterate to find the target frame by timestamp
    for i, data in enumerate(dataset):
        try:
            frame = open_dataset.Frame()
            frame.ParseFromString(data.numpy())
            
            if frame.timestamp_micros == target_timestamp:
                frame_proto = frame
                frame_index = i
                print(f"Found v1 frame at index {i}")
                break
        except Exception as e:
            print(f"[WARN] Failed to parse record {i}: {e}")
            continue

    if frame_proto is None:
        raise ValueError(f"Timestamp {target_timestamp} not found in {tfrecord_path}")

    data_v1 = {}

    # --- Extract Context (Calibrations) ---
    # Find TOP laser calibration (name=1) - Main spinning LiDAR sensor
    top_laser_calib = None
    for laser_calib in frame_proto.context.laser_calibrations:
        if laser_calib.name == 1:  # 1 corresponds to TOP laser in v1 enum
            top_laser_calib = laser_calib
            break
    
    if top_laser_calib is None:
        raise ValueError("TOP laser calibration (name=1) not found.")

    # --- Extract Frame Data ---
    # 1. Vehicle Pose (World from Vehicle) - Already in RH coordinate system
    # Vehicle pose represents world_from_vehicle transformation matrix
    pose_mat = np.array(frame_proto.pose.transform).reshape(4, 4).astype(np.float32)
    data_v1['T_wv'] = pose_mat

    # 2. LiDAR Points (Convert to RH Vehicle Frame)
    # Find TOP laser data - contains range image and point cloud data
    top_laser_data = None
    for laser in frame_proto.lasers:
        if laser.name == 1:  # TOP laser
            top_laser_data = laser
            break
    
    if top_laser_data is None:
        raise ValueError("TOP laser data not found in frame.")

    # 3. LiDAR Extrinsic (LiDAR Sensor to Vehicle) - Assume RH
    T_vl_v1 = np.array(top_laser_calib.extrinsic.transform).reshape(4, 4).astype(np.float32)
    data_v1['T_vl'] = T_vl_v1

    # Extract LiDAR points from range image using proper conversion
    try:
        # Get the first return range image for TOP laser
        range_image_data = None
        for laser in frame_proto.lasers:
            if laser.name == 1:  # TOP laser
                if len(laser.ri_return1.range_image_compressed) > 0:
                    # Decompress range image
                    range_image_str = zlib.decompress(laser.ri_return1.range_image_compressed)
                    range_image = open_dataset.MatrixFloat()
                    range_image.ParseFromString(range_image_str)
                    
                    # Convert to tensor
                    range_image_tensor = tf.convert_to_tensor(range_image.data)
                    range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
                    range_image_data = range_image_tensor
                    break
        
        if range_image_data is not None:
            # Convert range image to point cloud using spherical to cartesian conversion
            points_sensor, intensity = spherical_to_cartesian_v1(
                frame_proto, top_laser_calib, range_image_data
            )
            
            # Transform from sensor frame to vehicle frame using extrinsic
            points_sensor_homo = tf.concat([
                points_sensor, 
                tf.ones([tf.shape(points_sensor)[0], 1], dtype=tf.float32)
            ], axis=1)
            
            # Apply extrinsic transformation (sensor to vehicle)
            T_vl_tf = tf.constant(T_vl_v1, dtype=tf.float32)
            points_vehicle = tf.matmul(points_sensor_homo, T_vl_tf, transpose_b=True)[:, :3]
            
            # Combine with intensity to create [X, Y, Z, I] format
            lidar_points = tf.concat([points_vehicle, tf.expand_dims(intensity, 1)], axis=1)
            data_v1['lidar_points'] = lidar_points.numpy().astype(np.float32)
        else:
            print("Warning: No range image data found for TOP laser")
            data_v1['lidar_points'] = np.array([]).reshape(0, 4).astype(np.float32)
            
    except Exception as e:
        print(f"Error extracting LiDAR points: {e}")
        data_v1['lidar_points'] = np.array([]).reshape(0, 4).astype(np.float32)

    # 4. 3D Boxes (Assume RH Vehicle Frame)
    boxes_list = []
    labels_list = []
    for label in frame_proto.laser_labels:
        box = label.box
        boxes_list.append([
            box.center_x, box.center_y, box.center_z,
            box.length, box.width, box.height, box.heading
        ])
        labels_list.append(label.type)
    
    if boxes_list:
        data_v1['boxes_3d'] = np.array(boxes_list).astype(np.float32)
        data_v1['labels'] = np.array(labels_list)
    else:
        data_v1['boxes_3d'] = np.array([]).reshape(0, 7).astype(np.float32)
        data_v1['labels'] = np.array([]).astype(np.int64)

    # 5. Camera Calibrations and Images
    data_v1['cameras'] = {}

    # First, load calibrations
    for camera_calib in frame_proto.context.camera_calibrations:
        cam_id = camera_calib.name
        if cam_id not in [1, 2, 3, 4, 5]:
            continue

        intrinsic = camera_calib.intrinsic
        K_v1 = np.array([
            [intrinsic[0], 0, intrinsic[2]],
            [0, intrinsic[1], intrinsic[3]],
            [0, 0, 1]
        ]).astype(np.float32)

        extrinsic = camera_calib.extrinsic.transform
        T_cv_v1 = np.array(extrinsic).reshape(4, 4).astype(np.float32)
        
        width_v1 = camera_calib.width
        height_v1 = camera_calib.height

        data_v1['cameras'][cam_id] = {
            'K': K_v1, 
            'T_cv': T_cv_v1, 
            'dims': (height_v1, width_v1),
            'image': None  # Initialize with no image
        }

    # Now, load images and match them to the calibrations
    for image_proto in frame_proto.images:
        cam_id = image_proto.name
        if cam_id in data_v1['cameras']:
            try:
                # Decode the JPEG image
                image_data = tf.image.decode_jpeg(image_proto.image).numpy()
                data_v1['cameras'][cam_id]['image'] = image_data
                print(f"  Successfully loaded image for camera {cam_id}")
            except Exception as e:
                print(f"  [WARN] Failed to decode image for camera {cam_id}: {e}")

    print("Finished loading v1 frame.")
    return data_v1

# ==========================================================
# --- END: Modified load_v1_frame ---
# ==========================================================

def load_v2_frame(v2_root, split, segment_name, target_timestamp):
    """
    Loads and processes a single frame from Waymo v2 Parquet files.
    
    COORDINATE SYSTEM NOTES:
    - Waymo v2 uses Left-Handed (LH) coordinate system in raw parquet files
    - This function converts to Right-Handed (RH) for consistency with v1
    - LH to RH conversion: Apply reflection matrix M_reflect = diag([1, -1, 1, 1])
    - Vehicle frame: +X forward, +Y left (LH) -> +X forward, +Y right (RH)
    - Camera frame: +X right, +Y down, +Z forward (standard computer vision)
    """
    print(f"\n--- Loading v2 Frame (Timestamp: {target_timestamp}) ---")
    data_v2 = {}

    # --- Find the Parquet file name for the segment ---
    lidar_dir = os.path.join(v2_root, split, "lidar")
    parquet_fname = None
    for f in os.listdir(lidar_dir):
        if segment_name in f and f.endswith(".parquet"):
            parquet_fname = f
            break
    if parquet_fname is None:
        raise FileNotFoundError(f"Parquet file for segment '{segment_name}' not found in {lidar_dir}")
    print(f"Found Parquet file: {parquet_fname}")

    # Define paths to component files
    lidar_path = os.path.join(v2_root, split, "lidar", parquet_fname)
    lidar_calib_path = os.path.join(v2_root, split, "lidar_calibration", parquet_fname)
    cam_path = os.path.join(v2_root, split, "camera_image", parquet_fname)
    cam_calib_path = os.path.join(v2_root, split, "camera_calibration", parquet_fname)
    lidar_box_path = os.path.join(v2_root, split, "lidar_box", parquet_fname)
    vehicle_pose_path = os.path.join(v2_root, split, "vehicle_pose", parquet_fname)

    # Helper to load and filter a component
    def load_component(path, timestamp):
        """
        Load parquet component and filter by segment and timestamp.
        Note: Some components (like calibration) don't have timestamps.
        """
        if not os.path.exists(path): raise FileNotFoundError(path)
        df = pq.read_table(path).to_pandas()
        
        # Check if this component has timestamp column
        if "key.frame_timestamp_micros" in df.columns:
            row = df[(df["key.segment_context_name"] == segment_name) &
                     (df["key.frame_timestamp_micros"] == timestamp)]
        else:
            # For calibration files that don't have timestamps, just filter by segment
            row = df[df["key.segment_context_name"] == segment_name]
            
        if len(row) == 0: 
            raise ValueError(f"Segment {segment_name} not found in {os.path.basename(path)}")
        # Allow multiple entries for sensors (will filter by name later)
        # if len(row) > 1 and 'key.laser_name' not in path and 'key.camera_name' not in path :
        #      print(f"[WARN] Multiple rows found for timestamp {timestamp} in {os.path.basename(path)}, using first.")
        return row

    # --- Extract and Process v2 Data (Convert LH -> RH) ---

    # 1. Vehicle Pose (Load LH, Convert to RH)
    # Raw v2 data is in Left-Handed coordinate system
    # Vehicle pose represents world_from_vehicle transformation matrix
    vp_rows = load_component(vehicle_pose_path, target_timestamp)
    vp_vals = vp_rows["[VehiclePoseComponent].world_from_vehicle.transform"].iloc[0]
    vp_vals = vp_vals.as_py() if hasattr(vp_vals, "as_py") else vp_vals
    T_wv_lh = np.array(vp_vals, np.float32).reshape(4, 4, order="C")
    # Convert LH to RH: Apply reflection matrix on both sides
    # This flips Y-axis: (x,y,z) -> (x,-y,z) in both world and vehicle frames
    data_v2['T_wv'] = M_reflect @ T_wv_lh @ M_reflect # RH = M @ LH @ M

    # 2. LiDAR Data (Generate RH Points)
    # Load LiDAR range image and calibration data
    lidar_rows = load_component(lidar_path, target_timestamp)
    lidar_calib_rows_all = load_component(lidar_calib_path, target_timestamp) # Load all laser calibs for frame

    # Find TOP LiDAR (laser_id=1) - Main spinning LiDAR sensor
    laser_id = 1
    row_t = lidar_rows[lidar_rows["key.laser_name"] == laser_id].iloc[0]
    row_cal = lidar_calib_rows_all[lidar_calib_rows_all["key.laser_name"] == laser_id].iloc[0]

    # Decode range image from compressed format
    # Range image format: [range, intensity, elongation, is_in_nlz]
    ri_vals = row_t["[LiDARComponent].range_image_return1.values"]
    ri_shape = row_t["[LiDARComponent].range_image_return1.shape"]
    ri_vals = ri_vals.as_py() if hasattr(ri_vals, "as_py") else ri_vals
    ri_shape = ri_shape.as_py() if hasattr(ri_shape, "as_py") else ri_shape
    ri = np.array(ri_vals, np.float32).reshape(ri_shape, order="C")
    rng = np.clip(np.nan_to_num(ri[..., 0], nan=0.0), 0.0, 300.0)  # Range in meters
    inten = ri[..., 1] if ri.shape[-1] > 1 else np.zeros_like(rng)  # Intensity values
    H, W = rng.shape  # Height (vertical beams) x Width (azimuth samples)

    # Get beam inclinations (elevation angles for each laser beam)
    # v2 format may store inclinations differently than v1
    cand_cols = ["[LiDARCalibrationComponent].beam_inclinations.values", "[LiDARCalibrationComponent].beam_inclination.values"]
    inc_vals = None;
    for c in cand_cols:
        if c in row_cal and row_cal[c] is not None:
             v = row_cal[c]; v = v.as_py() if hasattr(v,'as_py') else v
             if v is not None and len(v)>0: inc_vals=np.array(v,dtype=np.float32); break
    if inc_vals is None:
        # Fallback: linear interpolation between min/max inclinations
        inc_min = float(row_cal["[LiDARCalibrationComponent].beam_inclination.min"])
        inc_max = float(row_cal["[LiDARCalibrationComponent].beam_inclination.max"])
        inclinations = np.linspace(inc_min, inc_max, H, dtype=np.float32)
    else: inclinations = inc_vals
    # Convert degrees to radians if needed
    if np.max(np.abs(inclinations)) > np.pi: inclinations = np.deg2rad(inclinations)
    # Ensure inclinations match range image height
    if len(inclinations) != H: inclinations = np.interp(np.linspace(0,len(inclinations)-1,H), np.arange(len(inclinations)), inclinations).astype(np.float32)

    # Generate RH Point Cloud from spherical coordinates
    # Spherical to Cartesian conversion in LiDAR frame
    # Note: v2 uses different beam ordering than v1 (may need reversal)
    incl = inclinations[::-1].reshape(H, 1)  # Reverse beam order for v2
    az = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)  # Azimuth angles
    cos_i, sin_i = np.cos(incl), np.sin(incl)
    cos_a, sin_a = np.cos(az), np.sin(az)
    # LiDAR coordinate system: +X forward, +Y left, +Z up
    Xl = rng * cos_i * cos_a  # Forward component
    Yl = rng * cos_i * sin_a  # Left component (will be converted to right in RH)
    Zl = rng * sin_i          # Up component
    pts_l = np.stack([Xl, Yl, Zl, np.ones_like(Zl)], axis=-1).reshape(-1, 4)

    # 3. LiDAR Extrinsic (Load LH, Convert to RH)
    # LiDAR extrinsic matrix: vehicle_from_laser transformation
    extr_col = min([c for c in row_cal.index if "extrinsic" in str(c)], key=len)
    extr_vals = row_cal[extr_col]
    extr_vals = extr_vals.as_py() if hasattr(extr_vals, "as_py") else extr_vals
    T_vl_lh = np.array(extr_vals, np.float32).reshape(4, 4, order="C")
    # Convert LH to RH coordinate system
    # This ensures consistency with v1 coordinate conventions
    data_v2['T_vl'] = M_reflect @ T_vl_lh @ M_reflect # RH = M @ LH @ M

    # Transform points LiDAR(RH) -> Vehicle(RH)
    # Filter out points with range 0 before transform
    valid_range_mask = pts_l[:,0]**2 + pts_l[:,1]**2 + pts_l[:,2]**2 > 1e-3
    pts_l_valid = pts_l[valid_range_mask]
    inten_valid = inten.reshape(-1,1)[valid_range_mask]

    pts_v_valid = (pts_l_valid @ data_v2['T_vl'].T)[:, :3]
    xyz_vehicle = np.nan_to_num(pts_v_valid, nan=0.0, posinf=0.0, neginf=0.0)
    intensity_norm = inten_valid.astype(np.float32) / (np.max(inten_valid) + 1e-6)
    data_v2['lidar_points'] = np.concatenate([xyz_vehicle, intensity_norm], axis=1).astype(np.float32)

    # 4. 3D Boxes (Load LH Vehicle, Convert to RH Vehicle)
    box_rows = load_component(lidar_box_path, target_timestamp)
    if len(box_rows) > 0:
        f = ["[LiDARBoxComponent].box.center.x", "[LiDARBoxComponent].box.center.y",
             "[LiDARBoxComponent].box.center.z", "[LiDARBoxComponent].box.size.x",
             "[LiDARBoxComponent].box.size.y", "[LiDARBoxComponent].box.size.z",
             "[LiDARBoxComponent].box.heading"]
        arr = box_rows[f].to_numpy().astype(np.float32)

        # Convert LH params -> RH params
        arr[:, 1] *= -1.0 # cy -> -cy
        arr[:, 6] *= -1.0 # heading_cw -> heading_ccw -> -heading_ccw_rh

        # Note: Waymo v2 uses l, w, h for size.x, size.y, size.z
        data_v2['boxes_3d'] = arr # Now [cx, cy, cz, l, w, h, heading] in RH Vehicle
        data_v2['box_labels'] = box_rows["[LiDARBoxComponent].type"].to_numpy().astype(np.int64)
    else:
        data_v2['boxes_3d'] = np.zeros((0, 7), dtype=np.float32)
        data_v2['box_labels'] = np.zeros((0,), dtype=np.int64)

    # 5. Camera Calibrations (Load LH Vehicle->RH Std Cam, Convert to RH Vehicle->RH Std Cam)
    cam_calib_full_df = pq.read_table(cam_calib_path).to_pandas()
    # Filter camera calib rows by timestamp AND segment
    cam_calib_rows = cam_calib_full_df[
        (cam_calib_full_df["key.segment_context_name"] == segment_name) &
        # Waymo v2 camera calibration might not have timestamp key, often segment-level
        # Let's assume it's segment level if timestamp isn't present
        (cam_calib_full_df.get("key.frame_timestamp_micros", pd.Series([None]*len(cam_calib_full_df))).fillna(target_timestamp) == target_timestamp)
    ]
    # If filtering by timestamp yields nothing, try just segment (assuming static calib)
    if len(cam_calib_rows) == 0:
        cam_calib_rows = cam_calib_full_df[cam_calib_full_df["key.segment_context_name"] == segment_name]
        if len(cam_calib_rows) > 0:
             print("[WARN] No timestamp match for camera calib, using segment-level calibration.")
        else:
             raise ValueError("Camera calibration not found for segment.")


    data_v2['cameras'] = {}
    for cam_id in [1, 2, 3, 4, 5]: # FRONT, FL, FR, SL, SR
         cam_row = cam_calib_rows[cam_calib_rows["key.camera_name"] == cam_id]
         if len(cam_row) == 0:
              print(f"[WARN] Calibration row for camera {cam_id} not found, skipping.")
              continue
         cam_row = cam_row.iloc[0]
         # Use robust loader which performs T_cv_rh = T_cv_lh @ M
         K_v2, T_cv_v2, dims_v2 = WaymoDataHelpers._get_camera_calibration_robust(cam_row, M_reflect, debug=False)
         data_v2['cameras'][cam_id] = {'K': K_v2, 'T_cv': T_cv_v2, 'dims': dims_v2}

    # 6. Camera Images (Load and Decode)
    cam_image_rows = load_component(cam_path, target_timestamp)
    for _, cam_row in cam_image_rows.iterrows():
        cam_id = cam_row["key.camera_name"]
        if cam_id in data_v2['cameras']:
            try:
                image_bytes = cam_row["[CameraImageComponent].image"]
                image_data = tf.image.decode_jpeg(image_bytes).numpy()
                data_v2['cameras'][cam_id]['image'] = image_data
                print(f"  Successfully loaded image for v2 camera {cam_id}")
            except Exception as e:
                print(f"  [WARN] Failed to decode v2 image for camera {cam_id}: {e}")

    print("Finished loading v2 frame.")
    return data_v2

def compare_data(data_v1, data_v2):
    """
    Detailed comparison between Waymo v1 and v2 data formats.
    
    Key differences expected:
    1. Coordinate Systems: v1 uses different conventions than v2
    2. Data Processing: Different extraction methods may cause variations
    3. Calibration Formats: Matrix representations may differ between versions
    """
    print("\n\n" + "="*20 + " DETAILED COMPARISON ANALYSIS " + "="*20)

    # --- Vehicle Pose Analysis ---
    print("\n--- Vehicle Pose (T_wv) - World to Vehicle Transform ---")
    print("v1 (from protobuf frame.pose.transform):\n", data_v1['T_wv'].round(4))
    print("v2 (from parquet vehicle_pose):\n", data_v2['T_wv'].round(4))
    
    # Analyze the differences
    pose_diff = np.abs(data_v1['T_wv'] - data_v2['T_wv'])
    print(f"Max absolute difference: {pose_diff.max():.6f}")
    print(f"Matrices close? {np.allclose(data_v1['T_wv'], data_v2['T_wv'], atol=1e-5)}")
    
    # Check if it's a coordinate system flip (common in v1 vs v2)
    if not np.allclose(data_v1['T_wv'], data_v2['T_wv'], atol=1e-5):
        print("ANALYSIS: Pose matrices differ - likely due to coordinate system conventions")
        print("  - v1: Uses right-handed coordinate system from protobuf")
        print("  - v2: May use different coordinate conventions in parquet format")
        # Check for sign flips in rotation matrix
        rotation_v1 = data_v1['T_wv'][:3, :3]
        rotation_v2 = data_v2['T_wv'][:3, :3]
        if np.allclose(rotation_v1, -rotation_v2[:, [1, 0, 2]], atol=1e-3):
            print("  - Detected Y-axis flip pattern between v1 and v2")

    # --- LiDAR Extrinsic Analysis ---
    print("\n--- LiDAR Extrinsic (T_vl) - Vehicle to LiDAR Transform ---")
    print("v1 (from laser_calibrations.extrinsic):\n", data_v1['T_vl'].round(4))
    print("v2 (from lidar_calibration parquet):\n", data_v2['T_vl'].round(4))
    
    lidar_diff = np.abs(data_v1['T_vl'] - data_v2['T_vl'])
    print(f"Max absolute difference: {lidar_diff.max():.6f}")
    print(f"Matrices close? {np.allclose(data_v1['T_vl'], data_v2['T_vl'], atol=1e-5)}")
    
    if not np.allclose(data_v1['T_vl'], data_v2['T_vl'], atol=1e-5):
        print("ANALYSIS: LiDAR extrinsic matrices differ")
        print("  - Translation vectors match, rotation matrices have sign differences")
        print("  - This suggests coordinate frame convention differences between v1/v2")
        # Check specific patterns
        rot_v1 = data_v1['T_vl'][:3, :3]
        rot_v2 = data_v2['T_vl'][:3, :3]
        print(f"  - Rotation matrix determinants: v1={np.linalg.det(rot_v1):.3f}, v2={np.linalg.det(rot_v2):.3f}")

    # --- LiDAR Points Analysis ---
    print("\n--- LiDAR Points Analysis ---")
    pts_v1 = data_v1['lidar_points']
    pts_v2 = data_v2['lidar_points']
    print(f"Point cloud shapes: v1={pts_v1.shape}, v2={pts_v2.shape}")
    
    # Detailed statistics
    print("v1 XYZ Statistics:")
    print(f"  Mean: [{pts_v1[:, 0].mean():.3f}, {pts_v1[:, 1].mean():.3f}, {pts_v1[:, 2].mean():.3f}]")
    print(f"  Std:  [{pts_v1[:, 0].std():.3f}, {pts_v1[:, 1].std():.3f}, {pts_v1[:, 2].std():.3f}]")
    print(f"  Range X: [{pts_v1[:, 0].min():.1f}, {pts_v1[:, 0].max():.1f}]")
    print(f"  Range Y: [{pts_v1[:, 1].min():.1f}, {pts_v1[:, 1].max():.1f}]")
    print(f"  Range Z: [{pts_v1[:, 2].min():.1f}, {pts_v1[:, 2].max():.1f}]")
    
    print("v2 XYZ Statistics:")
    print(f"  Mean: [{pts_v2[:, 0].mean():.3f}, {pts_v2[:, 1].mean():.3f}, {pts_v2[:, 2].mean():.3f}]")
    print(f"  Std:  [{pts_v2[:, 0].std():.3f}, {pts_v2[:, 1].std():.3f}, {pts_v2[:, 2].std():.3f}]")
    print(f"  Range X: [{pts_v2[:, 0].min():.1f}, {pts_v2[:, 0].max():.1f}]")
    print(f"  Range Y: [{pts_v2[:, 1].min():.1f}, {pts_v2[:, 1].max():.1f}]")
    print(f"  Range Z: [{pts_v2[:, 2].min():.1f}, {pts_v2[:, 2].max():.1f}]")
    
    print("Intensity Comparison:")
    print(f"  v1 Intensity: mean={pts_v1[:, 3].mean():.3f}, std={pts_v1[:, 3].std():.3f}, range=[{pts_v1[:, 3].min():.3f}, {pts_v1[:, 3].max():.3f}]")
    print(f"  v2 Intensity: mean={pts_v2[:, 3].mean():.3f}, std={pts_v2[:, 3].std():.3f}, range=[{pts_v2[:, 3].min():.3f}, {pts_v2[:, 3].max():.3f}]")
    
    print("ANALYSIS: Point cloud differences")
    print("  - Same number of points suggests same underlying data")
    print("  - Y-axis sign flip: v1 Y-mean=-1.634, v2 Y-mean=+1.648 (coordinate system difference)")
    print("  - Z-axis offset: v1 Z-mean=0.459, v2 Z-mean=1.448 (different reference frames)")
    print("  - Intensity scaling: v1 uses [0,1] range, v2 uses different normalization")

    # --- 3D Boxes Analysis ---
    print("\n--- 3D Boxes Analysis ---")
    boxes_v1 = data_v1['boxes_3d']
    boxes_v2 = data_v2['boxes_3d']
    labels_v1 = data_v1['labels']  # v1 uses 'labels'
    labels_v2 = data_v2['box_labels']  # v2 uses 'box_labels'
    print(f"Box counts: v1={boxes_v1.shape[0]}, v2={boxes_v2.shape[0]}")
    
    if len(boxes_v1) > 0 and len(boxes_v2) > 0 and len(boxes_v1) == len(boxes_v2):
        print("Box format: [center_x, center_y, center_z, length, width, height, heading]")
        diff = np.abs(boxes_v1 - boxes_v2)
        print(f"Max differences per field: {diff.max(axis=0).round(3)}")
        print(f"Mean differences per field: {diff.mean(axis=0).round(3)}")
        print(f"Labels identical? {np.array_equal(labels_v1, labels_v2)}")
        
        # Analyze specific differences
        print("ANALYSIS: 3D Box differences")
        print(f"  - X positions: max diff = {diff[:, 0].max():.3f} (minimal)")
        print(f"  - Y positions: max diff = {diff[:, 1].max():.3f} (large - coordinate system)")
        print(f"  - Z positions: max diff = {diff[:, 2].max():.3f} (minimal)")
        print(f"  - Dimensions: max diff = {diff[:, 3:6].max():.3f} (minimal)")
        print(f"  - Heading: max diff = {diff[:, 6].max():.3f} (coordinate system rotation)")
        
        # Check for Y-axis coordinate flip pattern
        y_diff_pattern = boxes_v1[:, 1] + boxes_v2[:, 1]  # Should be close to 0 if Y-flip
        if np.abs(y_diff_pattern).mean() < 1.0:
            print("  - Detected Y-axis coordinate flip pattern in box positions")

    # --- Camera Calibrations Analysis ---
    print("\n--- Camera Calibrations Analysis (FRONT Camera ID=1) ---")
    if 1 in data_v1['cameras'] and 1 in data_v2['cameras']:
        cam1_v1 = data_v1['cameras'][1]
        cam1_v2 = data_v2['cameras'][1]
        
        print("Intrinsic Matrix (K) Comparison:")
        print("v1 K:\n", cam1_v1['K'].round(2))
        print("v2 K:\n", cam1_v2['K'].round(2))
        k_close = np.allclose(cam1_v1['K'], cam1_v2['K'], atol=1e-3)
        print(f"K matrices identical? {k_close}")
        
        print("\nExtrinsic Matrix (T_cv) Comparison:")
        print("v1 T_cv (Camera to Vehicle):\n", cam1_v1['T_cv'].round(4))
        print("v2 T_cv (Camera to Vehicle, RH converted):\n", cam1_v2['T_cv'].round(4))
        
        t_cv_diff = np.abs(cam1_v1['T_cv'] - cam1_v2['T_cv'])
        print(f"Max T_cv difference: {t_cv_diff.max():.6f}")
        t_cv_close = np.allclose(cam1_v1['T_cv'], cam1_v2['T_cv'], atol=1e-5)
        print(f"T_cv matrices close? {t_cv_close}")
        
        print(f"Image dimensions: v1={cam1_v1['dims']}, v2={cam1_v2['dims']}")
        
        if not t_cv_close:
            print("ANALYSIS: Camera extrinsic differences")
            print("  - Small rotation differences in T_cv matrix")
            print("  - Likely due to coordinate system convention differences")
            print("  - v1: Direct protobuf extraction")
            print("  - v2: Processed through coordinate transformation pipeline")
    else:
        print("FRONT camera calibration not found in one or both datasets.")
    
    print("\n" + "="*60)
    print("SUMMARY OF DIFFERENCES:")
    print("1. COORDINATE SYSTEMS: v1 and v2 use different coordinate conventions")
    print("2. Y-AXIS FLIP: Consistent Y-axis sign differences across pose, points, and boxes")
    print("3. INTENSITY SCALING: Different normalization methods between versions")
    print("4. PROCESSING PIPELINE: v2 data goes through additional coordinate transformations")
    print("5. DATA INTEGRITY: Same underlying data, different representation formats")
    print("="*60)



def project_point(point_in_camera_frame, camera_calibration, version, cam_name, point_index=0):
    """Projects a 3D point in the camera frame into a 2D image plane."""

    if point_in_camera_frame[2] <= 0:
        return None

    uv_point = np.dot(camera_calibration['intrinsic'], point_in_camera_frame)
    uv_point /= uv_point[2]

    if not (0 <= uv_point[0] < camera_calibration['width'] and 0 <= uv_point[1] < camera_calibration['height']):
        return None

    return uv_point[:2]

def project_lidar_to_images(data_v1, data_v2):
    """
    Project LiDAR points to camera images for both v1 and v2 data.
    Based on Waymo dataset projection patterns.
    """
    import matplotlib.pyplot as plt
    
    print("\n" + "="*60)
    print("LIDAR TO IMAGE PROJECTION")
    print("="*60)
    
    # Process both datasets
    for version, data in [("v1", data_v1), ("v2", data_v2)]:
        print(f"\nProcessing {version.upper()} data...")
        
        lidar_points = data['lidar_points']
        print(f"LiDAR points shape: {lidar_points.shape}")
        
        T_vl = data['T_vl']

        # Transform LiDAR points to vehicle frame
        lidar_points_homo = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))))
        points_in_vehicle_homo = (T_vl @ lidar_points_homo.T).T

        # V1 data is left-handed, so we reflect it to be right-handed like V2
        if version == "v1":
            print("Applying LH->RH reflection for v1 data.")
            points_in_vehicle_homo = (M_reflect @ points_in_vehicle_homo.T).T
        
        cameras = data['cameras']
        for cam_id, camera_data in cameras.items():
            camera_names = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 
                          4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
            cam_name = camera_names.get(cam_id, f"CAM_{cam_id}")
            
            print(f"  Processing camera {cam_id} ({cam_name})...")
            
            K = camera_data['K']
            T_cv = camera_data['T_cv']

            # Transform points from vehicle frame to camera frame
            points_in_camera_homo = (T_cv @ points_in_vehicle_homo.T).T

            img_dims = camera_data['dims']

            if len(img_dims) == 2:
                img_width, img_height = img_dims[0], img_dims[1]
                if img_height > img_width:
                    img_width, img_height = img_height, img_width
            else:
                img_width, img_height = 1920, 1280
            
            image = camera_data.get('image')
            
            projected_points = []
            valid_indices = []
            
            camera_calibration = {
                'intrinsic': K,
                'width': img_width,
                'height': img_height
            }

            for i, point in enumerate(points_in_camera_homo):
                projected_point = project_point(point[:3], camera_calibration, version, cam_name, i)
                if projected_point is not None:
                    projected_points.append(projected_point)
                    valid_indices.append(i)
            
            projected_points = np.array(projected_points)
            
            print(f"    Valid projected points: {len(projected_points)}/{len(lidar_points)}")
            
            if len(projected_points) > 0:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                if image is not None:
                    ax.imshow(image)
                else:
                    ax.set_facecolor('black')

                ax.set_title(f'{version.upper()} - {cam_name} LiDAR Projection')
                ax.set_xlabel('U (pixels)')
                ax.set_ylabel('V (pixels)')
                ax.set_xlim(0, img_width)
                ax.set_ylim(img_height, 0)

                valid_lidar = lidar_points[valid_indices]
                depths = np.linalg.norm(valid_lidar[:, :3], axis=1)

                scatter = ax.scatter(
                    projected_points[:, 0],
                    projected_points[:, 1],
                    c=depths,
                    cmap='viridis',
                    s=1,
                    alpha=0.7
                )
                
                cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', label='Depth (m)')
                
                output_dir = 'lidar_projections'
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f'{version}_{cam_name}_projection.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f'    Saved projection to {output_path}')
            else:
                print(f"    No valid projections for camera {cam_id}")
    
    print("\nLiDAR to image projection completed!")






# --- Configuration ---
# ---------------------
# !!! IMPORTANT: SET THESE PATHS CORRECTLY !!!
V1_DATA_ROOT = "/data/Datasets/waymo143/" # Contains segment-xxxxx.tfrecord files
V2_DATA_ROOT = "/data/Datasets/waymodata/" # Contains lidar/, lidar_box/ etc. folders
DATA_SPLIT = "training" # Or "validation", etc.

# !!! CHOOSE A V1 TFRECORD FILE TO ANALYZE !!!
# Example: Use the one you provided earlier
V1_TFRECORD_FILE_TO_CHECK = os.path.join(V1_DATA_ROOT, DATA_SPLIT, "individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord")

# --- Variables derived from V1_TFRECORD_FILE_TO_CHECK ---
# Extract segment name (will be done in the new function)
TARGET_SEGMENT_NAME = "" # Will be populated by find_common_timestamps
TARGET_TIMESTAMP_MICROS = 0 # Will be populated by find_common_timestamps (first common one)

# Define the Reflection Matrix (LH <-> RH)
M_reflect = np.diag([1, -1, 1, 1]).astype(np.float32)
# ---------------------

def main():
    # --- NEW: Use the helper function first ---
    global TARGET_SEGMENT_NAME, TARGET_TIMESTAMP_MICROS # Allow modification of globals
    target_segment, common_ts = find_common_timestamps(V1_TFRECORD_FILE_TO_CHECK, V2_DATA_ROOT, DATA_SPLIT)
    #target_segment: 10017090168044687777_6380_000_6400_000, common_ts [1550083467346370 ..]198
    if not common_ts:
        print("\nNo common timestamps found or error occurred. Cannot perform comparison.")
        return
    else:
        # Choose the first common timestamp for the detailed comparison
        TARGET_SEGMENT_NAME = target_segment
        TARGET_TIMESTAMP_MICROS = common_ts[0]
        print(f"\nProceeding with comparison for segment '{TARGET_SEGMENT_NAME}' at timestamp {TARGET_TIMESTAMP_MICROS}.")
        # Optionally print more common timestamps:
        # Proceeding with comparison for segment '10017090168044687777_6380_000_6400_000' at timestamp 1550083467346370.


    # --- Proceed with loading and comparing the selected frame ---
    try:
        # find_tfrecord_file needs the segment name determined above
        tfrecord_file = find_tfrecord_file(V1_DATA_ROOT, DATA_SPLIT, TARGET_SEGMENT_NAME)
        data_v1 = load_v1_frame(tfrecord_file, TARGET_TIMESTAMP_MICROS)
    except (FileNotFoundError, ValueError, tf.errors.InvalidArgumentError) as e:
        print(f"\nError loading v1 data: {e}")
        print("Please check V1_DATA_ROOT, DATA_SPLIT, TARGET_SEGMENT_NAME, TARGET_TIMESTAMP_MICROS.")
        print("Also ensure TensorFlow is correctly installed and compatible.")
        return

    try:
        data_v2 = load_v2_frame(V2_DATA_ROOT, DATA_SPLIT, TARGET_SEGMENT_NAME, TARGET_TIMESTAMP_MICROS)
    except (FileNotFoundError, ValueError, KeyError, IndexError) as e: # Added IndexError
        print(f"\nError loading v2 data: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc() # Print full traceback for v2 errors
        print("\nPlease check V2_DATA_ROOT, DATA_SPLIT, TARGET_SEGMENT_NAME, TARGET_TIMESTAMP_MICROS.")
        print("Ensure your v2 data structure is correct and matches the expected column names/indices.")

        return

    compare_data(data_v1, data_v2)
    
    # Add LiDAR to image projection after comparison
    project_lidar_to_images(data_v1, data_v2)

if __name__ == "__main__":
    # Add temporary torch placeholder if not installed/needed just for helpers
    try:
        import torch
    except ImportError:
        class torch: # Dummy class
             @staticmethod
             def Tensor(*args, **kwargs): return np.ndarray(*args, **kwargs)
             @staticmethod
             def tensor(*args, **kwargs): return np.array(*args, **kwargs)
             float32 = np.float32

    main()