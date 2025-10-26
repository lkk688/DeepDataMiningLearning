#!/usr/bin/env python3
"""
Waymo Open Dataset Parquet Inspector

This script provides comprehensive inspection and analysis tools for Waymo Open Dataset
stored in Parquet format. It includes utilities for examining different data modalities
including camera images, LiDAR point clouds, 3D bounding boxes, segmentation masks,
and sensor calibration data.

The script works with the following Waymo data structure:
- camera_image/: RGB camera images from 5 cameras (FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT)
- camera_box/: 2D bounding box annotations for camera images
- lidar/: LiDAR range images and point cloud data from TOP and 4 SIDE sensors
- lidar_box/: 3D bounding box annotations for LiDAR data
- lidar_camera_projection/: Projection mappings between LiDAR points and camera pixels
- lidar_segmentation/: Point-wise semantic segmentation labels for LiDAR data
- camera_segmentation/: Pixel-wise segmentation masks for camera images
- vehicle_pose/: Vehicle ego-motion and trajectory information
- lidar_hkp/: Human keypoint annotations in LiDAR coordinate system

For comprehensive data reading and visualization utilities, see waymo_data_utils.py

"""

import os, argparse, pyarrow.parquet as pq, pandas as pd
import io
from PIL import Image
import numpy as np

def read_and_preview(folder_path, nrows=3):
    """Read one parquet file from the folder and preview."""
    files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        print(f"⚠️ No parquet files found in {folder_path}")
        return
    fpath = os.path.join(folder_path, files[0])
    print(f"\n=== Reading {fpath} ===")
    pf = pq.ParquetFile(fpath)
    print("Schema:\n", pf.schema, "\n")
    tbl = pf.read_row_group(0)
    df = tbl.to_pandas()
    print(df.head(nrows))
    print(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns\n")
    return fpath, df  # Return file path and dataframe for further processing


def extract_camera_images(folder_path, max_images=3):
    """
    Extract and process camera image objects from the chosen parquet file.
    
    This function demonstrates how to:
    1. Load binary image data from the parquet file
    2. Convert binary JPEG/PNG bytes to PIL Image objects
    3. Extract metadata associated with each camera frame
    4. Display image properties and associated pose/velocity data
    
    Args:
        folder_path (str): Path to the camera_image folder containing parquet files
        max_images (int): Maximum number of images to process for demonstration
    """
    print("\n🔍 EXTRACTING CAMERA IMAGES --------------------")
    
    # Get list of parquet files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        print("⚠️ No parquet files found for camera image extraction")
        return
    
    # Load the first parquet file
    fpath = os.path.join(folder_path, files[0])
    print(f"📂 Processing: {os.path.basename(fpath)}")
    
    try:
        # Read the parquet file into a pandas DataFrame
        df = pd.read_parquet(fpath)
        print(f"📊 Loaded {len(df)} camera frames from parquet file")
        
        # Process a subset of images for demonstration
        num_to_process = min(max_images, len(df))
        print(f"🖼️ Processing first {num_to_process} camera frames...\n")
        
        for idx in range(num_to_process):
            row = df.iloc[idx]
            
            # Extract metadata for this camera frame
            segment_name = row['key.segment_context_name']
            timestamp = row['key.frame_timestamp_micros']
            camera_id = row['key.camera_name']
            
            print(f"--- Camera Frame {idx + 1} ---")
            print(f"  📍 Segment: {segment_name}")
            print(f"  ⏰ Timestamp: {timestamp} μs")
            print(f"  📷 Camera ID: {camera_id}")
            
            # Extract binary image data
            image_bytes = row['[CameraImageComponent].image']
            
            if image_bytes is not None and len(image_bytes) > 0:
                try:
                    # Convert binary data to PIL Image
                    image_stream = io.BytesIO(image_bytes)
                    pil_image = Image.open(image_stream)
                    
                    # Display image properties
                    print(f"  🖼️ Image size: {pil_image.size} (W×H)")
                    print(f"  🎨 Image mode: {pil_image.mode}")
                    print(f"  📏 Binary size: {len(image_bytes):,} bytes")
                    
                    # Convert to numpy array for further processing if needed
                    img_array = np.array(pil_image)
                    print(f"  🔢 Array shape: {img_array.shape}")
                    print(f"  📈 Pixel value range: [{img_array.min()}, {img_array.max()}]")
                    
                except Exception as e:
                    print(f"  ❌ Error processing image: {e}")
            else:
                print(f"  ⚠️ No image data found for this frame")
            
            # Extract pose and velocity information if available
            try:
                # Pose transformation matrix (4x4 matrix flattened to list)
                if '[CameraImageComponent].pose.transform' in row and row['[CameraImageComponent].pose.transform'] is not None:
                    pose_transform = row['[CameraImageComponent].pose.transform']
                    if len(pose_transform) == 16:  # 4x4 matrix flattened
                        pose_matrix = np.array(pose_transform).reshape(4, 4)
                        print(f"  🧭 Pose matrix available: {pose_matrix.shape}")
                        print(f"      Translation: [{pose_matrix[0,3]:.3f}, {pose_matrix[1,3]:.3f}, {pose_matrix[2,3]:.3f}]")
                
                # Linear velocity
                vel_x = row.get('[CameraImageComponent].velocity.linear_velocity.x', 0)
                vel_y = row.get('[CameraImageComponent].velocity.linear_velocity.y', 0) 
                vel_z = row.get('[CameraImageComponent].velocity.linear_velocity.z', 0)
                if any([vel_x, vel_y, vel_z]):
                    print(f"  🏃 Linear velocity: [{vel_x:.3f}, {vel_y:.3f}, {vel_z:.3f}] m/s")
                
                # Angular velocity
                ang_x = row.get('[CameraImageComponent].velocity.angular_velocity.x', 0)
                ang_y = row.get('[CameraImageComponent].velocity.angular_velocity.y', 0)
                ang_z = row.get('[CameraImageComponent].velocity.angular_velocity.z', 0)
                if any([ang_x, ang_y, ang_z]):
                    print(f"  🔄 Angular velocity: [{ang_x:.3f}, {ang_y:.3f}, {ang_z:.3f}] rad/s")
                
                # Rolling shutter parameters
                shutter = row.get('[CameraImageComponent].rolling_shutter_params.shutter', None)
                if shutter is not None:
                    print(f"  📸 Rolling shutter: {shutter:.6f}")
                    
            except Exception as e:
                print(f"  ⚠️ Error extracting pose/velocity data: {e}")
            
            print()  # Empty line for readability
            
    except Exception as e:
        print(f"❌ Error reading parquet file: {e}")
    
    print("✅ Camera image extraction completed\n")


def inspect_camera_image(root, split):
    """
    Folder: camera_image/
    Contains compressed RGB images (JPEG/PNG bytes) and per-frame keys.
    Based on actual schema analysis, each parquet file contains 15 columns with rich metadata.
    """
    folder = os.path.join(root, split, "camera_image")
    print("\n📷 CAMERA IMAGE --------------------------------")
    print("• Stores one row per camera frame (for all cameras).")
    print("• Schema contains 15 columns with the following key data:")
    print("  - index: unique row identifier")
    print("  - key.segment_context_name: segment / sequence ID (string)")
    print("  - key.frame_timestamp_micros: microsecond timestamp (int64)")
    print("  - key.camera_name: integer camera ID 0-5 (int8)")
    print("  - [CameraImageComponent].image: binary JPEG/PNG image bytes")
    print("  - [CameraImageComponent].pose.transform: 4x4 transformation matrix (list of doubles)")
    print("  - [CameraImageComponent].velocity: linear (x,y,z) and angular velocity (x,y,z)")
    print("  - [CameraImageComponent].pose_timestamp: pose timestamp (double)")
    print("  - [CameraImageComponent].rolling_shutter_params: camera timing parameters\n")
    read_and_preview(folder)
    
    # Extract and process camera image data from the first parquet file
    extract_camera_images(folder)


def inspect_camera_box(root, split):
    """
    Folder: camera_box/
    Contains 2D bounding boxes (in image pixel coordinates) for all cameras.
    Based on actual schema analysis, each parquet file contains 12 columns with detailed annotation data.
    """
    folder = os.path.join(root, split, "camera_box")
    print("\n🎯 CAMERA BOX ----------------------------------")
    print("• Each row = one labeled object in one camera frame.")
    print("• Schema contains 12 columns with comprehensive annotation data:")
    print("  - index: unique row identifier")
    print("  - key.segment_context_name: segment / sequence ID (string)")
    print("  - key.frame_timestamp_micros: microsecond timestamp (int64)")
    print("  - key.camera_name: integer camera ID 0-5 (int8)")
    print("  - key.camera_object_id: unique object identifier per camera (string)")
    print("  - [CameraBoxComponent].box.center.x: bounding box center X coordinate (double)")
    print("  - [CameraBoxComponent].box.center.y: bounding box center Y coordinate (double)")
    print("  - [CameraBoxComponent].box.size.x: bounding box width in pixels (double)")
    print("  - [CameraBoxComponent].box.size.y: bounding box height in pixels (double)")
    print("  - [CameraBoxComponent].type: object class type ID (int8)")
    print("  - [CameraBoxComponent].difficulty_level.detection: detection difficulty (int8)")
    print("  - [CameraBoxComponent].difficulty_level.tracking: tracking difficulty (int8)")
    print("• Use together with camera_image/ for 2D detection tasks.\n")
    read_and_preview(folder)


def inspect_camera_calibration(root, split):
    """
    Folder: camera_calibration/
    Provides intrinsic/extrinsic calibration for every camera in each segment.
    """
    folder = os.path.join(root, split, "camera_calibration")
    print("\n📐 CAMERA CALIBRATION ---------------------------")
    print("• Stores per-camera calibration matrices and parameters.")
    print("• Contains 15 columns with detailed camera calibration data:")
    print("  - key.segment_context_name (String): Segment identifier")
    print("  - key.camera_name (Int8): Camera identifier (0-4 for different cameras)")
    print("  - [CameraCalibrationComponent].intrinsic.f_u (Double): Focal length in u direction")
    print("  - [CameraCalibrationComponent].intrinsic.f_v (Double): Focal length in v direction")
    print("  - [CameraCalibrationComponent].intrinsic.c_u (Double): Principal point u coordinate")
    print("  - [CameraCalibrationComponent].intrinsic.c_v (Double): Principal point v coordinate")
    print("  - [CameraCalibrationComponent].intrinsic.k1 (Double): Radial distortion coefficient 1")
    print("  - [CameraCalibrationComponent].intrinsic.k2 (Double): Radial distortion coefficient 2")
    print("  - [CameraCalibrationComponent].intrinsic.p1 (Double): Tangential distortion coefficient 1")
    print("  - [CameraCalibrationComponent].intrinsic.p2 (Double): Tangential distortion coefficient 2")
    print("  - [CameraCalibrationComponent].intrinsic.k3 (Double): Radial distortion coefficient 3")
    print("  - [CameraCalibrationComponent].extrinsic.transform (List): 4×4 transformation matrix")
    print("  - [CameraCalibrationComponent].width (Int32): Image width in pixels")
    print("  - [CameraCalibrationComponent].height (Int32): Image height in pixels")
    print("  - [CameraCalibrationComponent].rolling_shutter_direction (Int8): Rolling shutter direction\n")
    read_and_preview(folder)
    


def inspect_lidar(root, split):
    """
    Folder: lidar/
    Contains LiDAR range images (return1/2) for all LiDARs.
    Each row = one LiDAR sensor at one timestamp.
    """
    folder = os.path.join(root, split, "lidar")
    print("\n🌐 LIDAR ---------------------------------------")
    print("• Core sensor data: range images (not point clouds).")
    print("• Contains 7 columns with LiDAR range image data:")
    print("  - index (String): Unique identifier for each LiDAR measurement")
    print("  - key.segment_context_name (String): Segment identifier")
    print("  - key.frame_timestamp_micros (Int64): Frame timestamp in microseconds")
    print("  - key.laser_name (Int8): LiDAR sensor ID (TOP, FRONT, SIDE_LEFT, SIDE_RIGHT, REAR)")
    print("  - [LiDARComponent].range_image_return1.values (List[Float]): First return range image data")
    print("  - [LiDARComponent].range_image_return1.shape (List[Int32]): Shape of first return image")
    print("  - [LiDARComponent].range_image_return2.values (List[Float]): Second return range image data")
    print("  - [LiDARComponent].range_image_return2.shape (List[Int32]): Shape of second return image")
    print("• Range images are encoded as flattened arrays with shape information for reconstruction.\n")
    read_and_preview(folder)


def inspect_lidar_box(root, split):
    """
    Folder: lidar_box/
    3D bounding boxes labeled in vehicle coordinate frame.
    """
    folder = os.path.join(root, split, "lidar_box")
    print("\n📦 LIDAR BOX -----------------------------------")
    print("• 3D bounding boxes labeled in Vehicle coordinate frame.")
    print("• Each row represents one 3D object detection in a LiDAR frame.")
    print("• Contains 21 columns with comprehensive 3D object annotation data:")
    print("  - index (String): Unique identifier for each 3D bounding box")
    print("  - key.segment_context_name (String): Segment identifier")
    print("  - key.frame_timestamp_micros (Int64): Frame timestamp in microseconds")
    print("  - key.laser_object_id (String): Unique object ID for tracking across frames")
    print("  - [LiDARBoxComponent].box.center.x/y/z (Double): 3D box center coordinates in meters")
    print("  - [LiDARBoxComponent].box.size.x/y/z (Double): 3D box dimensions (length/width/height) in meters")
    print("  - [LiDARBoxComponent].box.heading (Double): Object orientation angle in radians")
    print("  - [LiDARBoxComponent].type (Int8): Object category ID (vehicle, pedestrian, cyclist, etc.)")
    print("  - [LiDARBoxComponent].num_lidar_points_in_box (Int64): Total LiDAR points inside the box")
    print("  - [LiDARBoxComponent].num_top_lidar_points_in_box (Int64): Top LiDAR points inside the box")
    print("  - [LiDARBoxComponent].speed.x/y/z (Double): Object velocity in m/s along each axis")
    print("  - [LiDARBoxComponent].acceleration.x/y/z (Double): Object acceleration in m/s² along each axis")
    print("  - [LiDARBoxComponent].difficulty_level.detection (Int8): Detection difficulty rating")
    print("  - [LiDARBoxComponent].difficulty_level.tracking (Int8): Tracking difficulty rating")
    print("• Provides complete 3D object state including geometry, motion, and annotation quality metrics.\n")
    fpath, df = read_and_preview(folder)
    print(f"Reading {fpath}")
    pf = pq.ParquetFile(fpath)
    print("Schema:\n", pf.schema)

    # Read small sample
    df = pf.read_row_group(0).to_pandas()
    print(df.head(3))

    # ---- Sanity checks ----
    # 1️⃣ Check box sizes and positions
    size_cols = ["[LiDARBoxComponent].box.size.x",
                 "[LiDARBoxComponent].box.size.y",
                 "[LiDARBoxComponent].box.size.z"]
    center_cols = ["[LiDARBoxComponent].box.center.x",
                   "[LiDARBoxComponent].box.center.y",
                   "[LiDARBoxComponent].box.center.z"]
    head_col = "[LiDARBoxComponent].box.heading"

    if all(c in df for c in size_cols + center_cols + [head_col]):
        sizes = df[size_cols].to_numpy()
        centers = df[center_cols].to_numpy()
        headings = df[head_col].to_numpy()

        # Check for NaN or negative sizes
        if np.any(~np.isfinite(sizes)):
            print("⚠️ Warning: NaN or Inf found in box sizes.")
        if np.any(sizes <= 0):
            print("⚠️ Warning: Non-positive box dimension detected.")

        # Print statistical summary
        print(f"Box dimension stats (L,W,H): mean={sizes.mean(0)}, min={sizes.min(0)}, max={sizes.max(0)}")
        print(f"Box center range X/Y/Z: min={centers.min(0)}, max={centers.max(0)}")

        # Check heading in radians or degrees
        max_head = np.abs(headings).max()
        if max_head > np.pi * 2:
            print(f"⚠️ Heading values exceed 2π (max={max_head:.2f}); likely in degrees.")
        else:
            print(f"✅ Heading values appear in radians (max={max_head:.2f}).")
    else:
        print("⚠️ Missing expected box columns; cannot validate numeric ranges.")

# ----------------------------------------------------------------------
# Helper to read Waymo Parquet "List" columns safely.
# Waymo often stores arrays (e.g., extrinsic.transform, beam_inclination.values)
# as Arrow List columns. This function converts them to a numpy array of scalars.
# ----------------------------------------------------------------------
def _read_list_column(col):
    """
    Convert a PyArrow or Pandas column of List type into a numpy array.

    Handles cases where each cell is:
      - pyarrow.lib.ListValue
      - Python list / numpy array
      - pyarrow.Scalar (single value)
    """
    import numpy as np
    # Case 1: PyArrow ListValue
    if hasattr(col, "values"):
        return np.array([x.as_py() for x in col.values])
    # Case 2: Python list or numpy array
    if isinstance(col, (list, np.ndarray)):
        out = []
        for x in col:
            if hasattr(x, "as_py"):
                out.append(x.as_py())
            else:
                out.append(x)
        return np.array(out)
    # Case 3: PyArrow Scalar
    if hasattr(col, "as_py"):
        return np.array(col.as_py())
    raise TypeError(f"Unsupported column type: {type(col)}")

def inspect_lidar_calibration(root, split):
    """
    Folder: lidar_calibration/
    Provides beam inclination angles and extrinsic transforms for LiDARs.
    """
    folder = os.path.join(root, split, "lidar_calibration")
    print("\n🧭 LIDAR CALIBRATION ---------------------------")
    print("• Stores per-LiDAR sensor calibration parameters and geometric transforms.")
    print("• Contains 6 columns with detailed LiDAR calibration data:")
    print("  - key.segment_context_name (String): Segment identifier")
    print("  - key.laser_name (Int8): LiDAR sensor identifier (TOP, FRONT, SIDE_LEFT, SIDE_RIGHT, REAR)")
    print("  - [LiDARCalibrationComponent].extrinsic.transform (List[Double]): 4×4 transformation matrix")
    print("    └─ Transforms from LiDAR coordinate frame to vehicle coordinate frame")
    print("  - [LiDARCalibrationComponent].beam_inclination.min (Double): Minimum beam inclination angle")
    print("  - [LiDARCalibrationComponent].beam_inclination.max (Double): Maximum beam inclination angle")
    print("  - [LiDARCalibrationComponent].beam_inclination.values (List[Double]): Individual beam angles")
    print("    └─ Array of inclination angles for each laser beam in the sensor")
    print("• Essential for converting LiDAR range images to 3D point clouds and coordinate transformations.\n")
    fpath, df = read_and_preview(folder)
    print(f"Reading {fpath}")
    pf = pq.ParquetFile(fpath)
    print("Schema:\n", pf.schema)

    df = pf.read_row_group(0).to_pandas()
    print(df.head(3))

    # ---- Beam inclination check ----
    if "[LiDARCalibrationComponent].beam_inclination.min" in df and "[LiDARCalibrationComponent].beam_inclination.max" in df:
        minv = df["[LiDARCalibrationComponent].beam_inclination.min"].astype(float)
        maxv = df["[LiDARCalibrationComponent].beam_inclination.max"].astype(float)
        print(f"Beam inclinations (min,max): {minv.iloc[0]:.4f}, {maxv.iloc[0]:.4f}")

        # Check if likely degrees or radians
        if np.abs(maxv.iloc[0]) > np.pi:
            print("⚠️ Detected degree values — convert to radians before computing XYZ.")
        else:
            print("✅ Beam inclinations appear in radians.")
    elif "[LiDARCalibrationComponent].beam_inclination.values" in df:
        vals = _read_list_column(df.iloc[0]["[LiDARCalibrationComponent].beam_inclination.values"])
        print(f"Beam_inclination.values length={len(vals)}, range=({vals.min():.4f},{vals.max():.4f})")
        if np.abs(vals).max() > np.pi:
            print("⚠️ Detected degree values — convert to radians.")
    else:
        print("⚠️ Beam inclination fields missing.")

    # ---- Extrinsic transform check ----
    extr_field = "[LiDARCalibrationComponent].extrinsic.transform"
    if extr_field in df.columns:
        extr = np.array(_read_list_column(df.iloc[0][extr_field]), dtype=np.float32)
        if extr.size != 16:
            print(f"⚠️ Extrinsic transform size {extr.size}, expected 16.")
        else:
            M = extr.reshape(4, 4)
            print("Extrinsic (first entry):\n", M)
            # Sanity: bottom row should be [0,0,0,1]
            if not np.allclose(M[3], [0, 0, 0, 1], atol=1e-3):
                print("⚠️ Extrinsic bottom row not [0,0,0,1]; check reshape order (C/F).")
            # Check translation magnitude
            t = M[:3, 3]
            if np.linalg.norm(t) > 10:
                print(f"⚠️ Large translation {t}, probably wrong reshape order.")
            else:
                print(f"✅ Reasonable LiDAR translation vector {t}.")
    else:
        print("⚠️ No extrinsic.transform column found; verify schema.")


def inspect_lidar_pose(root, split):
    """
    Folder: lidar_pose/
    Contains LiDAR poses (transform from LiDAR to vehicle/world).
    
    Parquet Schema (5 columns):
    - index (string): Unique identifier for each pose record
    - key.segment_context_name (string): Segment context identifier
    - key.frame_timestamp_micros (int64): Frame timestamp in microseconds
    - key.laser_name (int32): LiDAR sensor identifier (0=TOP, 1=FRONT, 2=SIDE_LEFT, 3=SIDE_RIGHT, 4=REAR)
    - [LiDARPoseComponent].range_image_return1.values (list<float>): Flattened pose transformation matrix values
    - [LiDARPoseComponent].range_image_return1.shape (list<int32>): Shape dimensions [64, 2650, 6]
    
    The pose data is encoded as a flattened array with shape [64, 2650, 6], where:
    - 64: Height dimension of the range image
    - 2650: Width dimension of the range image  
    - 6: Pose parameters (3 for translation xyz, 3 for rotation)
    
    This provides the 6-DoF transformation from LiDAR coordinate frame to vehicle coordinate frame,
    essential for multi-sensor fusion, temporal alignment, and world coordinate transformations.
    """
    folder = os.path.join(root, split, "lidar_pose")
    print("\n🚗 LIDAR POSE ----------------------------------")
    print("• Provides LiDAR sensor pose (orientation + translation).")
    print("• Useful for multi-frame fusion or world alignment.\n")
    read_and_preview(folder)


def inspect_vehicle_pose(root, split):
    """
    Folder: vehicle_pose/
    Gives the 6-DoF pose of the ego vehicle in the global coordinate frame.
    
    Parquet Schema (4 columns):
    - index (string): Unique identifier for each pose record
    - key.segment_context_name (string): Segment context identifier
    - key.frame_timestamp_micros (int64): Frame timestamp in microseconds
    - [VehiclePoseComponent].world_from_vehicle.transform (list<double>): 4x4 transformation matrix (flattened)
    
    The transformation matrix is stored as a flattened list of 16 double values representing
    a 4x4 homogeneous transformation matrix that transforms points from vehicle coordinate
    frame to world coordinate frame. The matrix format is:
    
    [R11, R12, R13, Tx,
     R21, R22, R23, Ty, 
     R31, R32, R33, Tz,
     0,   0,   0,   1 ]
    
    Where:
    - R11-R33: 3x3 rotation matrix elements
    - Tx, Ty, Tz: Translation vector (vehicle position in world coordinates)
    - Bottom row: [0, 0, 0, 1] for homogeneous coordinates
    
    This provides the complete 6-DoF pose (position + orientation) of the ego vehicle
    at each timestamp, essential for trajectory analysis, localization, and mapping.
    """
    folder = os.path.join(root, split, "vehicle_pose")
    print("\n🌍 VEHICLE POSE --------------------------------")
    print("• Vehicle pose (position + orientation) at each timestamp.")
    print("• Fields: position.x/y/z, orientation.x/y/z/w (quaternion)\n")
    read_and_preview(folder)


def inspect_lidar_camera_projection(root, split):
    """
    Folder: lidar_camera_projection/
    Pre-computed mapping from each LiDAR point to all camera pixels.
    
    Parquet Schema (7 columns):
    - index (string): Unique identifier for each projection record
    - key.segment_context_name (string): Segment context identifier
    - key.frame_timestamp_micros (int64): Frame timestamp in microseconds
    - key.laser_name (int32): LiDAR sensor identifier (0=TOP, 1=FRONT, 2=SIDE_LEFT, 3=SIDE_RIGHT, 4=REAR)
    - [LiDARCameraProjectionComponent].range_image_return1.values (list<float>): Flattened projection data for first return
    - [LiDARCameraProjectionComponent].range_image_return1.shape (list<int32>): Shape dimensions [64, 2650, 6]
    - [LiDARCameraProjectionComponent].range_image_return2.values (list<float>): Flattened projection data for second return
    - [LiDARCameraProjectionComponent].range_image_return2.shape (list<int32>): Shape dimensions [64, 2650, 6]
    
    The projection data is encoded as flattened arrays with shape [64, 2650, 6], where:
    - 64: Height dimension of the range image
    - 2650: Width dimension of the range image
    - 6: Projection parameters per pixel (camera_id, x, y, range, intensity, elongation)
    
    This data provides pre-computed mappings from LiDAR range image pixels to camera image coordinates,
    enabling efficient LiDAR-camera sensor fusion, 3D-2D correspondence verification, and multi-modal
    perception algorithms. The large file size (≈50GB) reflects the dense projection mappings across
    all LiDAR sensors and camera views.
    """
    folder = os.path.join(root, split, "lidar_camera_projection")
    print("\n🔗 LIDAR-CAMERA PROJECTION ---------------------")
    print("• Very large (≈50GB).")
    print("• Each row contains flattened arrays:")
    print("  - [LiDARCameraProjectionComponent].camera_projection.(x,y)")
    print("  - camera_id, range, etc.")
    print("• Used for LiDAR→Camera fusion verification.\n")
    read_and_preview(folder, nrows=1)


def inspect_lidar_hkp(root, split):
    """
    Folder: lidar_hkp/
    Health-Keypoint-Pose diagnostics for LiDAR sensors.
    
    Parquet Schema (8 columns):
    - index (string): Unique identifier for each HKP record
    - key.segment_context_name (string): Segment context identifier
    - key.frame_timestamp_micros (int64): Frame timestamp in microseconds
    - key.laser_object_id (string): LiDAR object identifier for human keypoint tracking
    - [LiDARHumanKeypointsComponent].lidar_keypoints[*].type (list<int32>): Keypoint type identifiers
    - [LiDARHumanKeypointsComponent].lidar_keypoints[*].keypoint_3d.location_m.x (list<double>): X coordinates in meters
    - [LiDARHumanKeypointsComponent].lidar_keypoints[*].keypoint_3d.location_m.y (list<double>): Y coordinates in meters
    - [LiDARHumanKeypointsComponent].lidar_keypoints[*].keypoint_3d.location_m.z (list<double>): Z coordinates in meters
    - [LiDARHumanKeypointsComponent].lidar_keypoints[*].keypoint_3d.visibility.is_occluded (list<boolean>): Occlusion flags
    
    This dataset contains human keypoint annotations detected in LiDAR point clouds, providing:
    - 3D human pose estimation data with precise spatial coordinates
    - Keypoint type classification (head, shoulders, joints, etc.)
    - Visibility and occlusion information for each keypoint
    - Object tracking IDs for temporal consistency across frames
    
    Note: This file may be empty (0 rows) in many segments as human keypoint annotations
    are sparse and only available when humans are present and clearly visible in the LiDAR data.
    Used primarily for human pose estimation research and pedestrian behavior analysis,
    not typically required for standard autonomous driving perception tasks.
    """
    folder = os.path.join(root, split, "lidar_hkp")
    print("\n🩺 LIDAR HKP -----------------------------------")
    print("• Records LiDAR health and sync info:")
    print("  - temperature, voltage, time_offset")
    print("  - keypoints (3D calibration reference points)")
    print("• Used for internal QA; not needed for perception training.\n")
    read_and_preview(folder)


def inspect_segmentation(root, split):
    """
    Folder: lidar_segmentation/ or camera_segmentation/
    Contains per-point or per-pixel semantic labels.
    
    LiDAR Segmentation Parquet Schema (7 columns):
    - index: String - Unique identifier for each record
    - key.segment_context_name: String - Segment context identifier
    - key.frame_timestamp_micros: int64 - Frame timestamp in microseconds
    - key.laser_name: int32 - LiDAR sensor identifier (0-4 for TOP, FRONT, SIDE_LEFT, SIDE_RIGHT, REAR)
    - [LiDARSegmentationLabelComponent].range_image_return1.values: List[int32] - Flattened semantic labels for first return
    - [LiDARSegmentationLabelComponent].range_image_return1.shape: List[int32] - Shape [64, 2650, 2] for range image dimensions
    - [LiDARSegmentationLabelComponent].range_image_return2.values: List[int32] - Flattened semantic labels for second return
    - [LiDARSegmentationLabelComponent].range_image_return2.shape: List[int32] - Shape [64, 2650, 2] for range image dimensions
    
    Range Image Shape [64, 2650, 2]:
    - 64: Vertical resolution (laser beams)
    - 2650: Horizontal resolution (azimuth samples)
    - 2: Two channels per pixel (semantic class ID and instance ID)
    
    Camera Segmentation Parquet Schema (10 columns):
    - index: String - Unique identifier for each record
    - key.segment_context_name: String - Segment context identifier
    - key.frame_timestamp_micros: int64 - Frame timestamp in microseconds
    - key.camera_name: int32 - Camera sensor identifier (0-4 for FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT)
    - [CameraSegmentationLabelComponent].panoptic_label_divisor: int32 - Divisor for panoptic segmentation encoding
    - [CameraSegmentationLabelComponent].panoptic_label: binary - Encoded panoptic segmentation mask
    - [CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.local_instance_ids: List[int32] - Local instance IDs
    - [CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.global_instance_ids: List[int32] - Global instance IDs
    - [CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.is_tracked: List[boolean] - Tracking status flags
    - [CameraSegmentationLabelComponent].sequence_id: String - Sequence identifier for tracking
    - [CameraSegmentationLabelComponent].num_cameras_covered: binary - Number of cameras covering each instance
    
    Applications:
    - Semantic segmentation: Pixel/point-level classification (vehicle, pedestrian, cyclist, etc.)
    - Instance segmentation: Individual object instance identification
    - Panoptic segmentation: Combined semantic and instance segmentation
    - Multi-sensor fusion: Consistent labeling across LiDAR and camera modalities
    - Tracking: Temporal consistency of object instances across frames
    """
    for sub in ["lidar_segmentation", "camera_segmentation"]:
        folder = os.path.join(root, split, sub)
        if os.path.isdir(folder):
            print(f"\n🎨 {sub.upper()} ------------------------------")
            print("• Semantic class labels for each LiDAR point or camera pixel.")
            print("• Typically has integer class IDs per sample.\n")
            read_and_preview(folder)

import os, numpy as np, pyarrow.parquet as pq

def print_waymo_debug(root, split="training", frame_idx=0, max_boxes=5):
    """
    Deep debug Waymo v2.1 parquet consistency:
      - lidar/   : range image
      - lidar_calibration/ : extrinsics & beam inclinations
      - lidar_box/ : 3D boxes
    """
    # 1️⃣  选择这一帧的基本信息
    lidar_dir = os.path.join(root, split, "lidar")
    box_dir   = os.path.join(root, split, "lidar_box")
    calib_dir = os.path.join(root, split, "lidar_calibration")
    pose_dir  = os.path.join(root, split, "lidar_pose")

    files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".parquet")])
    if len(files) == 0:
        raise FileNotFoundError(lidar_dir)
    fname = files[0]
    print(f"✅ Using lidar shard: {fname}")

    pf = pq.ParquetFile(os.path.join(lidar_dir, fname))
    df = pf.read_row_group(0).to_pandas()
    print(f"[lidar] rows={len(df)} cols={list(df.columns)}")
    row = df.iloc[frame_idx]
    seg, ts, laser_id = row["key.segment_context_name"], int(row["key.frame_timestamp_micros"]), int(row["key.laser_name"])
    print(f"segment={seg}\ntimestamp={ts}\nlaser_id={laser_id}")

    # 2️⃣  验证 box 文件是否有对应帧
    pf_box = pq.ParquetFile(os.path.join(box_dir, fname))
    df_box = pf_box.read_row_group(0).to_pandas()
    same_seg = df_box[df_box["key.segment_context_name"] == seg]
    same_ts  = same_seg[same_seg["key.frame_timestamp_micros"] == ts]
    print(f"[lidar_box] same segment rows={len(same_seg)}, same timestamp rows={len(same_ts)}")
    if len(same_ts):
        b = same_ts.iloc[0]
        print("First box sample:")
        print(f" center=({b['[LiDARBoxComponent].box.center.x']:.2f},"
              f"{b['[LiDARBoxComponent].box.center.y']:.2f},"
              f"{b['[LiDARBoxComponent].box.center.z']:.2f}), "
              f"size=({b['[LiDARBoxComponent].box.size.x']:.2f},"
              f"{b['[LiDARBoxComponent].box.size.y']:.2f},"
              f"{b['[LiDARBoxComponent].box.size.z']:.2f}), "
              f"heading={np.rad2deg(b['[LiDARBoxComponent].box.heading']):.1f}°")
    else:
        print("⚠️  No box rows with same timestamp (可能是时间戳对不上).")

    # 3️⃣  读取 calibration
    pf_cal = pq.ParquetFile(os.path.join(calib_dir, fname))
    df_cal = pf_cal.read_row_group(0).to_pandas()
    match = df_cal[(df_cal["key.segment_context_name"] == seg) &
                   (df_cal["key.laser_name"] == laser_id)]
    print(f"[lidar_calibration] same seg+laser rows={len(match)}")
    if len(match):
        rowc = match.iloc[0]
        inc_min = float(rowc.get("[LiDARCalibrationComponent].beam_inclination.min", np.nan))
        inc_max = float(rowc.get("[LiDARCalibrationComponent].beam_inclination.max", np.nan))
        extr_col = max([c for c in rowc.index if ("extrinsic" in c or str(c).endswith("item"))], key=len)
        extr_vals = rowc[extr_col]
        if hasattr(extr_vals, "as_py"):
            extr_vals = extr_vals.as_py()
        extr = np.array(extr_vals, dtype=np.float32).reshape(4,4,order="C")
        R = extr[:3,:3]; t = extr[:3,3]
        yaw = np.rad2deg(np.arctan2(R[1,0], R[0,0]))
        print(f" beam_inclination range=({inc_min:.4f}, {inc_max:.4f})")
        print(f" extrinsic translation={t}")
        print(f" extrinsic yaw(deg)={yaw:.2f}")
        print(" extrinsic matrix:\n", np.array2string(extr, formatter={'float_kind':lambda x:f'{x:8.4f}'}))
    else:
        print("⚠️  No calibration row for this segment+laser.")

    # 4️⃣  验证 pose 文件（如有）
    pose_path = os.path.join(pose_dir, fname)
    if os.path.exists(pose_path):
        pf_pose = pq.ParquetFile(pose_path)
        df_pose = pf_pose.read_row_group(0).to_pandas()
        same_seg_pose = df_pose[df_pose["key.segment_context_name"] == seg]
        same_ts_pose  = same_seg_pose[same_seg_pose["key.frame_timestamp_micros"] == ts]
        print(f"[lidar_pose] same segment rows={len(same_seg_pose)}, same timestamp rows={len(same_ts_pose)}")
    else:
        print("⚠️  No lidar_pose folder found.")

    # 5️⃣  打印前 N 个 box 的中心和 heading，方便比对
    if len(same_ts):
        print("\nTop few box centers/headings:")
        for i, (_, r) in enumerate(same_ts.head(max_boxes).iterrows()):
            print(f"  #{i}: center=({r['[LiDARBoxComponent].box.center.x']:.2f}, "
                  f"{r['[LiDARBoxComponent].box.center.y']:.2f}, "
                  f"{r['[LiDARBoxComponent].box.center.z']:.2f}), "
                  f"yaw={np.rad2deg(r['[LiDARBoxComponent].box.heading']):.1f}°")

    print("\n✅ Debug print finished.")

import os, numpy as np, pyarrow.parquet as pq

def debug_waymo_full(root, split="training", fname=None, frame_idx=0, max_boxes=5):
    """
    Extended Waymo parquet debugger:
      - list all LiDAR laser_names in this shard
      - print calibration yaw / translation for each
      - print range image shape per laser
      - print sample 3D boxes info
      - print LiDAR pose rows count
    """
    lidar_dir = os.path.join(root, split, "lidar")
    box_dir   = os.path.join(root, split, "lidar_box")
    calib_dir = os.path.join(root, split, "lidar_calibration")
    pose_dir  = os.path.join(root, split, "lidar_pose")

    files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".parquet")])
    if fname is None:
        fname = files[0]
    if not os.path.exists(os.path.join(lidar_dir, fname)):
        raise FileNotFoundError(fname)

    print(f"✅ Using lidar shard: {fname}\n")

    # ---------- lidar ----------
    pf = pq.ParquetFile(os.path.join(lidar_dir, fname))
    df = pf.read_row_group(0).to_pandas()
    print(f"[lidar] rows={len(df)}")
    lasers = sorted(df["key.laser_name"].unique())
    print("  laser_names present:", lasers)
    seg = df.iloc[0]["key.segment_context_name"]
    ts  = int(df.iloc[frame_idx]["key.frame_timestamp_micros"])
    print(f"  segment={seg}")
    print(f"  timestamp(example row {frame_idx})={ts}\n")

    # 统计每个 laser_id 的帧数与 range image shape
    for lid in lasers:
        rows = df[df["key.laser_name"] == lid]
        if len(rows)==0: continue
        shape_col = "[LiDARComponent].range_image_return1.shape"
        shape = rows.iloc[0][shape_col]
        if hasattr(shape,"as_py"): shape = shape.as_py()
        print(f"  laser_id={lid}  rows={len(rows)}  shape={shape}")

    # ---------- lidar_calibration ----------
    pf_cal = pq.ParquetFile(os.path.join(calib_dir, fname))
    df_cal = pf_cal.read_row_group(0).to_pandas()
    print(f"\n[lidar_calibration] rows={len(df_cal)}")
    for lid in sorted(df_cal["key.laser_name"].unique()):
        crow = df_cal[df_cal["key.laser_name"]==lid].iloc[0]
        beam_min = float(crow.get("[LiDARCalibrationComponent].beam_inclination.min", np.nan))
        beam_max = float(crow.get("[LiDARCalibrationComponent].beam_inclination.max", np.nan))
        extr_col = max([c for c in crow.index if ("extrinsic" in c or str(c).endswith("item"))], key=len)
        extr_vals = crow[extr_col]
        if hasattr(extr_vals,"as_py"): extr_vals = extr_vals.as_py()
        extr = np.array(extr_vals, dtype=np.float32).reshape(4,4,order="C")
        R,t = extr[:3,:3], extr[:3,3]
        yaw = np.rad2deg(np.arctan2(R[1,0],R[0,0]))
        print(f"  laser_id={lid} yaw(deg)={yaw:6.1f}  trans={t}  beam_range=({beam_min:.3f},{beam_max:.3f})")

    # ---------- lidar_box ----------
    pf_box = pq.ParquetFile(os.path.join(box_dir, fname))
    df_box = pf_box.read_row_group(0).to_pandas()
    same_seg = df_box[df_box["key.segment_context_name"]==seg]
    same_ts  = same_seg[same_seg["key.frame_timestamp_micros"]==ts]
    print(f"\n[lidar_box] same segment rows={len(same_seg)}, same timestamp rows={len(same_ts)}")
    if len(same_ts)==0:
        # 列出最邻近时间戳差异
        diff = np.sort(np.unique(np.abs(df_box["key.frame_timestamp_micros"]-ts)))
        print("  ⚠️ no exact timestamp match; nearest delta(us):", diff[:3])
    else:
        for i,(_,b) in enumerate(same_ts.head(max_boxes).iterrows()):
            heading_deg = np.rad2deg(b["[LiDARBoxComponent].box.heading"])
            center = (b["[LiDARBoxComponent].box.center.x"],
                      b["[LiDARBoxComponent].box.center.y"],
                      b["[LiDARBoxComponent].box.center.z"])
            size   = (b["[LiDARBoxComponent].box.size.x"],
                      b["[LiDARBoxComponent].box.size.y"],
                      b["[LiDARBoxComponent].box.size.z"])
            print(f"  box#{i}: center={np.round(center,2)}, size={np.round(size,2)}, heading={heading_deg:6.1f}°")

    # ---------- lidar_pose ----------
    pose_path = os.path.join(pose_dir, fname)
    if os.path.exists(pose_path):
        pf_pose = pq.ParquetFile(pose_path)
        df_pose = pf_pose.read_row_group(0).to_pandas()
        seg_poses = df_pose[df_pose["key.segment_context_name"]==seg]
        ts_match  = seg_poses[seg_poses["key.frame_timestamp_micros"]==ts]
        print(f"\n[lidar_pose] same seg={len(seg_poses)} rows, timestamp match={len(ts_match)}")
        print("  laser_names in pose file:", np.unique(seg_poses["key.laser_name"]))
    else:
        print("\n⚠️ No lidar_pose folder.")

    print("\n✅ Debug summary finished.")

import os, numpy as np, pyarrow.parquet as pq
import matplotlib.pyplot as plt
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

RI_VALS1 = "[LiDARComponent].range_image_return1.values"
RI_SHAPE1 = "[LiDARComponent].range_image_return1.shape"
def _decode_range_image(row):
    vals = _read_list_column(row[RI_VALS1])
    shape = _read_list_column(row[RI_SHAPE1])
    if len(shape) < 3:
        raise ValueError(f"Invalid range image shape: {shape}")
    H, W, C = map(int, shape[:3])
    arr = np.array(vals, dtype=np.float32).reshape(H, W, C)
    return arr
def merge_and_debug_lidars(root, split="training", fname=None, frame_idx=0, save_ply=False):
    """
    Merge all LiDARs (1~5) into one vehicle-frame point cloud.
    Print yaw/translation, point counts, and merged stats.
    """

    lidar_dir = os.path.join(root, split, "lidar")
    calib_dir = os.path.join(root, split, "lidar_calibration")

    # ----------- choose file -----------
    files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".parquet")])
    if fname is None:
        fname = files[0]
    print(f"✅ Using lidar shard: {fname}")

    # ----------- read lidar frames -----------
    pf = pq.ParquetFile(os.path.join(lidar_dir, fname))
    df = pf.read_row_group(0).to_pandas()
    seg = df.iloc[0]["key.segment_context_name"]
    ts = int(df.iloc[frame_idx]["key.frame_timestamp_micros"])
    print(f"segment={seg}, timestamp={ts}")

    # ----------- read calibration -----------
    pf_cal = pq.ParquetFile(os.path.join(calib_dir, fname))
    df_cal = pf_cal.read_row_group(0).to_pandas()
    calibs = {}
    for lid in sorted(df_cal["key.laser_name"].unique()):
        crow = df_cal[df_cal["key.laser_name"]==lid].iloc[0]
        extr_col = max([c for c in crow.index if ("extrinsic" in c or str(c).endswith("item"))], key=len)
        extr_vals = crow[extr_col]
        if hasattr(extr_vals, "as_py"): extr_vals = extr_vals.as_py()
        extr = np.array(extr_vals, dtype=np.float32).reshape(4,4,order="C")
        R,t = extr[:3,:3], extr[:3,3]
        yaw = np.rad2deg(np.arctan2(R[1,0], R[0,0]))
        calibs[lid] = {"extr":extr, "yaw":yaw, "t":t}
        print(f"  laser_id={lid} yaw(deg)={yaw:6.1f}  trans={np.round(t,3)}")

    # ----------- decode one frame per LiDAR -----------
    all_points = []
    stats = []
    for lid, group in df.groupby("key.laser_name"):
        row = group[group["key.frame_timestamp_micros"]==ts]
        if len(row)==0:
            print(f"⚠️  laser {lid}: no frame at timestamp {ts}")
            continue
        row = row.iloc[0]
        ri = _decode_range_image(row)
        if ri is None:
            print(f"⚠️  laser {lid}: decode failed.")
            continue
        rng = np.nan_to_num(ri[...,0], nan=0.0, posinf=0.0, neginf=0.0)
        rng = np.clip(rng, 0.0, 300.0)
        inten = ri[...,1] if ri.shape[-1]>1 else np.zeros_like(rng)
        H,W = rng.shape
        inc_min, inc_max = float(crow["[LiDARCalibrationComponent].beam_inclination.min"]), \
                           float(crow["[LiDARCalibrationComponent].beam_inclination.max"])
        inc = np.linspace(inc_min, inc_max, H, dtype=np.float32)[::-1].reshape(H,1)
        az  = np.linspace(np.pi, -np.pi, W, endpoint=False, dtype=np.float32)
        cos_i, sin_i = np.cos(inc), np.sin(inc)
        cos_a, sin_a = np.cos(az),  np.sin(az)
        x = rng * cos_i * cos_a
        y = rng * cos_i * sin_a
        z = rng * sin_i
        pts_h = np.stack([x,y,z,np.ones_like(z)],axis=-1).reshape(-1,4)
        extr = calibs[lid]["extr"]
        xyz = (pts_h @ extr.T)[:,:3]
        xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
        inten = inten.reshape(-1,1).astype(np.float32)
        inten = inten / (inten.max()+1e-6) if inten.max()>0 else np.full_like(inten,0.5)
        pts = np.concatenate([xyz,inten],axis=1)
        all_points.append(pts)
        stats.append((lid, pts.shape[0], xyz.min(0), xyz.max(0)))
        print(f"  laser {lid}: points={pts.shape[0]:6d}  X[{xyz[:,0].min():.1f},{xyz[:,0].max():.1f}]  "
              f"Y[{xyz[:,1].min():.1f},{xyz[:,1].max():.1f}]  Z[{xyz[:,2].min():.1f},{xyz[:,2].max():.1f}]")

    if not all_points:
        print("⚠️  No LiDAR data decoded."); return

    merged = np.concatenate(all_points, axis=0)
    print(f"\n✅ merged points={merged.shape[0]:,}")
    print(f"  range X[{merged[:,0].min():.1f},{merged[:,0].max():.1f}] "
          f"Y[{merged[:,1].min():.1f},{merged[:,1].max():.1f}] "
          f"Z[{merged[:,2].min():.1f},{merged[:,2].max():.1f}]")

    # ----------- optional: save ply for Open3D ----------
    if save_ply:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged[:,:3]))
        colors = np.stack([merged[:,3]]*3, axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        ply_path = os.path.join(root, f"{seg}_merged.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"💾 saved merged point cloud to {ply_path}")

    # ----------- simple 2D schematic of sensor positions ----------
    plt.figure(figsize=(6,6))
    for lid, v in calibs.items():
        x,y,z = v["t"]
        yaw = np.deg2rad(v["yaw"])
        dx,dy = np.cos(yaw), np.sin(yaw)
        plt.scatter(x, y, marker='o', label=f"L{lid}")
        plt.arrow(x, y, dx*0.8, dy*0.8, head_width=0.2, color='k')
        plt.text(x+0.3, y+0.3, f"{lid}\n{v['yaw']:.0f}°", fontsize=9)
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.axis('equal')
    plt.title(f"LiDAR positions (segment {seg[:6]}...)")
    plt.xlabel("+X forward (m)")
    plt.ylabel("+Y left (m)")
    plt.legend()
    plt.show()

    return merged, calibs

import open3d as o3d
import numpy as np

# ============================================================================
# Waymo Visualization Functions (Based on Official Tutorial)
# ============================================================================

def plot_range_image_helper(data, name, layout, vmin=0, vmax=1, cmap='gray'):
    """
    Plots range image based on official Waymo tutorial.
    
    Args:
        data: range image data
        name: the image title
        layout: plt layout
        vmin: minimum value of the passed data
        vmax: maximum value of the passed data
        cmap: color map
    """
    import matplotlib.pyplot as plt
    plt.subplot(*layout)
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(name)
    plt.grid(False)
    plt.axis('off')

def visualize_range_images(root, split="training", fname=None, frame_idx=0, save_path=None):
    """
    Visualize range images for all LiDAR sensors based on official tutorial.
    
    Args:
        root: Waymo dataset root path
        split: dataset split (training/validation/testing)
        fname: specific parquet file name
        frame_idx: frame index to visualize
        save_path: optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    lidar_dir = os.path.join(root, split, "lidar")
    calib_dir = os.path.join(root, split, "lidar_calibration")
    
    files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".parquet")])
    if not files:
        raise FileNotFoundError(f"No lidar parquet found in {lidar_dir}")
    if fname is None:
        fname = files[0]
    
    print(f"✅ Visualizing range images from: {fname}")
    
    # Load lidar data
    pf = pq.ParquetFile(os.path.join(lidar_dir, fname))
    df = pf.read_row_group(0).to_pandas()
    
    seg = df.iloc[0]["key.segment_context_name"]
    ts = int(df.iloc[frame_idx]["key.frame_timestamp_micros"])
    lasers = sorted(df["key.laser_name"].unique())
    
    print(f"Segment: {seg}, Timestamp: {ts}")
    print(f"Available LiDAR sensors: {lasers}")
    
    # Create figure for all range images
    plt.figure(figsize=(20, 12))
    
    layout_idx = 1
    for laser_id in lasers:
        # Get data for this laser
        laser_rows = df[df["key.laser_name"] == laser_id]
        if len(laser_rows) == 0:
            continue
            
        row = laser_rows.iloc[frame_idx % len(laser_rows)]
        
        # Decode range image
        ri = _decode_range_image(row)
        if ri is None:
            print(f"⚠️ Failed to decode range image for laser {laser_id}")
            continue
            
        # Extract channels
        range_data = np.nan_to_num(ri[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
        intensity_data = ri[..., 1] if ri.shape[-1] > 1 else np.zeros_like(range_data)
        elongation_data = ri[..., 2] if ri.shape[-1] > 2 else np.zeros_like(range_data)
        
        # Create mask for valid points
        valid_mask = range_data > 0
        range_data = np.where(valid_mask, range_data, np.ones_like(range_data) * 1e10)
        
        # Plot range, intensity, and elongation
        plot_range_image_helper(
            range_data, f'Laser {laser_id} - Range', 
            [len(lasers), 3, layout_idx], vmax=75, cmap='gray'
        )
        plot_range_image_helper(
            intensity_data, f'Laser {laser_id} - Intensity', 
            [len(lasers), 3, layout_idx + 1], vmax=1.5, cmap='gray'
        )
        plot_range_image_helper(
            elongation_data, f'Laser {laser_id} - Elongation', 
            [len(lasers), 3, layout_idx + 2], vmax=1.5, cmap='gray'
        )
        
        layout_idx += 3
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Range images saved to: {save_path}")
    plt.show()

def convert_range_image_to_point_cloud(root, split="training", fname=None, frame_idx=0, 
                                     return_index=0, include_intensity=True):
    """
    Convert range images to point cloud based on official tutorial methodology.
    
    Args:
        root: Waymo dataset root path
        split: dataset split
        fname: parquet file name
        frame_idx: frame index
        return_index: 0 for first return, 1 for second return
        include_intensity: whether to include intensity information
        
    Returns:
        points: (N, 3) or (N, 4) array of points [x, y, z] or [x, y, z, intensity]
        laser_labels: (N,) array indicating which laser each point came from
    """
    lidar_dir = os.path.join(root, split, "lidar")
    calib_dir = os.path.join(root, split, "lidar_calibration")
    
    files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".parquet")])
    if fname is None:
        fname = files[0]
    
    # Load data
    pf = pq.ParquetFile(os.path.join(lidar_dir, fname))
    df = pf.read_row_group(0).to_pandas()
    
    pf_cal = pq.ParquetFile(os.path.join(calib_dir, fname))
    df_cal = pf_cal.read_row_group(0).to_pandas()
    
    ts = int(df.iloc[frame_idx]["key.frame_timestamp_micros"])
    
    all_points = []
    all_laser_labels = []
    
    for laser_id in sorted(df["key.laser_name"].unique()):
        # Get calibration for this laser
        calib_row = df_cal[df_cal["key.laser_name"] == laser_id].iloc[0]
        extr_col = max([c for c in calib_row.index if ("extrinsic" in c or str(c).endswith("item"))], key=len)
        extr_vals = calib_row[extr_col]
        if hasattr(extr_vals, "as_py"):
            extr_vals = extr_vals.as_py()
        
        extr = np.array(extr_vals, dtype=np.float32).reshape(4, 4, order="C")
        
        # Get beam inclination range
        inc_min = float(calib_row["[LiDARCalibrationComponent].beam_inclination.min"])
        inc_max = float(calib_row["[LiDARCalibrationComponent].beam_inclination.max"])
        
        # Get range image data
        laser_rows = df[df["key.laser_name"] == laser_id]
        row = laser_rows[laser_rows["key.frame_timestamp_micros"] == ts]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        
        # Decode range image (handle both returns)
        if return_index == 0:
            vals_key = "[LiDARComponent].range_image_return1.values"
            shape_key = "[LiDARComponent].range_image_return1.shape"
        else:
            vals_key = "[LiDARComponent].range_image_return2.values"
            shape_key = "[LiDARComponent].range_image_return2.shape"
            
        if vals_key not in row or shape_key not in row:
            continue
            
        vals = row[vals_key]
        shp = row[shape_key]
        if hasattr(vals, "as_py"):
            vals = vals.as_py()
        if hasattr(shp, "as_py"):
            shp = shp.as_py()
            
        if not vals or not shp:
            continue
            
        H, W, C = shp
        ri = np.array(vals, dtype=np.float32).reshape(H, W, C)
        
        # Extract range and intensity
        range_data = np.nan_to_num(ri[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
        intensity_data = ri[..., 1] if ri.shape[-1] > 1 else np.zeros_like(range_data)
        
        # Create spherical coordinates
        inclination = np.linspace(inc_min, inc_max, H, dtype=np.float32)[::-1].reshape(H, 1)
        azimuth = np.linspace(np.pi, -np.pi, W, endpoint=False, dtype=np.float32)
        
        # Convert to Cartesian coordinates (LiDAR frame)
        cos_incl = np.cos(inclination)
        sin_incl = np.sin(inclination)
        cos_az = np.cos(azimuth)
        sin_az = np.sin(azimuth)
        
        x = range_data * cos_incl * cos_az
        y = range_data * cos_incl * sin_az
        z = range_data * sin_incl
        
        # Transform to vehicle frame
        points_lidar = np.stack([x, y, z, np.ones_like(z)], axis=-1).reshape(-1, 4)
        points_vehicle = (points_lidar @ extr.T)[:, :3]
        
        # Filter valid points
        valid_mask = range_data.reshape(-1) > 0
        points_vehicle = points_vehicle[valid_mask]
        
        if include_intensity:
            intensity_flat = intensity_data.reshape(-1)[valid_mask]
            points_vehicle = np.column_stack([points_vehicle, intensity_flat])
        
        all_points.append(points_vehicle)
        all_laser_labels.extend([laser_id] * len(points_vehicle))
    
    if not all_points:
        return np.empty((0, 4 if include_intensity else 3)), np.array([])
    
    points = np.vstack(all_points)
    laser_labels = np.array(all_laser_labels)
    
    print(f"✅ Converted {len(points):,} points from {len(set(all_laser_labels))} LiDAR sensors")
    return points, laser_labels

def rgba_from_range(r):
    """
    Generates a color based on range (from official tutorial).
    
    Args:
        r: the range value of a given point
    Returns:
        The color for a given range
    """
    import matplotlib.pyplot as plt
    c = plt.get_cmap('jet')((r % 20.0) / 20.0)
    c = list(c)
    c[-1] = 0.5  # alpha
    return c

def plot_camera_image(camera_image):
    """Plot camera image (from official tutorial)."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 12))
    plt.imshow(camera_image)
    plt.grid(False)
    plt.axis('off')

def visualize_camera_projection(root, split="training", fname=None, frame_idx=0, 
                              camera_name=1, save_path=None):
    """
    Visualize LiDAR points projected onto camera image based on official tutorial.
    
    Args:
        root: Waymo dataset root path
        split: dataset split
        fname: parquet file name
        frame_idx: frame index
        camera_name: camera ID (1=FRONT, 2=FRONT_LEFT, etc.)
        save_path: optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Load camera image
    camera_dir = os.path.join(root, split, "camera_image")
    files = sorted([f for f in os.listdir(camera_dir) if f.endswith(".parquet")])
    if fname is None:
        fname = files[0]
    
    pf_cam = pq.ParquetFile(os.path.join(camera_dir, fname))
    df_cam = pf_cam.read_row_group(0).to_pandas()
    
    # Get camera image
    cam_rows = df_cam[df_cam["key.camera_name"] == camera_name]
    if len(cam_rows) == 0:
        print(f"No camera data found for camera {camera_name}")
        return
    
    cam_row = cam_rows.iloc[frame_idx % len(cam_rows)]
    img_data = cam_row["[CameraImageComponent].image"]
    if hasattr(img_data, "as_py"):
        img_data = img_data.as_py()
    
    # Decode JPEG image
    from PIL import Image
    import io
    camera_image = np.array(Image.open(io.BytesIO(img_data)))
    
    # Get point cloud
    points, _ = convert_range_image_to_point_cloud(root, split, fname, frame_idx, include_intensity=True)
    
    # Load camera calibration
    calib_dir = os.path.join(root, split, "camera_calibration")
    pf_cal = pq.ParquetFile(os.path.join(calib_dir, fname))
    df_cal = pf_cal.read_row_group(0).to_pandas()
    
    cam_calib = df_cal[df_cal["key.camera_name"] == camera_name].iloc[0]
    
    # Get intrinsic matrix
    intrinsic_vals = cam_calib["[CameraCalibrationComponent].intrinsic"]
    if hasattr(intrinsic_vals, "as_py"):
        intrinsic_vals = intrinsic_vals.as_py()
    intrinsic = np.array(intrinsic_vals, dtype=np.float32).reshape(3, 3)
    
    # Get extrinsic matrix
    extr_col = max([c for c in cam_calib.index if ("extrinsic" in c or str(c).endswith("item"))], key=len)
    extr_vals = cam_calib[extr_col]
    if hasattr(extr_vals, "as_py"):
        extr_vals = extr_vals.as_py()
    extrinsic = np.array(extr_vals, dtype=np.float32).reshape(4, 4)
    
    # Project points to camera
    points_homo = np.column_stack([points[:, :3], np.ones(len(points))])
    points_cam = (points_homo @ extrinsic.T)[:, :3]
    
    # Filter points in front of camera
    front_mask = points_cam[:, 2] > 0
    points_cam = points_cam[front_mask]
    ranges = np.linalg.norm(points[:, :3][front_mask], axis=1)
    
    # Project to image plane
    points_2d = (points_cam @ intrinsic.T)
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    
    # Filter points within image bounds
    h, w = camera_image.shape[:2]
    valid_mask = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & 
                  (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h))
    
    points_2d = points_2d[valid_mask]
    ranges = ranges[valid_mask]
    
    # Visualize
    plt.figure(figsize=(20, 12))
    plt.imshow(camera_image)
    
    # Plot projected points with range-based colors
    colors = [rgba_from_range(r) for r in ranges]
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c=colors, s=1, alpha=0.8)
    
    plt.title(f'Camera {camera_name} with LiDAR Projection (Frame {frame_idx})')
    plt.grid(False)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Camera projection saved to: {save_path}")
    plt.show()
    
    print(f"✅ Projected {len(points_2d):,} LiDAR points onto camera {camera_name}")

def visualize_merged_open3d(merged_points, calibs=None, boxes3d=None, axis_size=5.0):
    """
    Visualize merged multi-LiDAR point cloud + sensor positions + boxes in Open3D.
    Enhanced version based on official tutorial patterns.

    Args:
        merged_points: (N,4) array [x,y,z,intensity]  (vehicle frame)
        calibs: dict from merge_and_debug_lidars() {laser_id: {"t":(3,), "yaw":deg}}
        boxes3d: optional (M,7) [x,y,z,dx,dy,dz,yaw] in vehicle frame
        axis_size: world coordinate axis size
    """
    import open3d as o3d

    geoms = []

    # --- Point cloud with intensity-based coloring ---
    xyz = merged_points[:, :3]
    inten = merged_points[:, 3] if merged_points.shape[1] > 3 else np.ones(len(merged_points), np.float32)
    inten = np.asarray(inten, np.float32)

    # Robust normalization for NumPy 2.x compatibility
    i_min = float(np.min(inten)) if inten.size else 0.0
    i_max = float(np.max(inten)) if inten.size else 1.0
    i_range = i_max - i_min
    if i_range < 1e-12:
        inten_norm = np.full_like(inten, 0.7)
    else:
        inten_norm = (inten - i_min) / (i_range + 1e-6)

    # Create point cloud with jet colormap (similar to tutorial)
    import matplotlib.pyplot as plt
    colormap = plt.get_cmap('jet')
    colors = colormap(inten_norm)[:, :3]  # RGB only
    
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geoms.append(pcd)

    # --- 3D bounding boxes (Waymo Vehicle Frame) ---
    if boxes3d is not None and len(boxes3d) > 0:
        if hasattr(boxes3d, "detach"):
            boxes3d = boxes3d.detach().cpu().numpy()
        for i, b in enumerate(boxes3d):
            x, y, z, dx, dy, dz, yaw = map(float, b[:7])
            # Waymo uses right-hand coordinate system
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[ c,-s,0],[s,c,0],[0,0,1]], np.float32)
            obb = o3d.geometry.OrientedBoundingBox(center=[x,y,z], R=R, extent=[dx,dy,dz])
            # Color boxes by index for better visibility
            box_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 
                         (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]
            obb.color = box_colors[i % len(box_colors)]
            geoms.append(obb)

    # --- LiDAR sensor positions and orientations ---
    if calibs is not None:
        for lid, v in calibs.items():
            t = np.asarray(v["t"], np.float32)
            yaw = np.deg2rad(v["yaw"])
            
            # Position marker (sphere)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            sphere.translate(t)
            sphere.paint_uniform_color([0.0, 0.8, 0.0])
            geoms.append(sphere)
            
            # Orientation arrow
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cone_height=0.2, cone_radius=0.05,
                cylinder_height=0.8, cylinder_radius=0.03)
            Rz = o3d.geometry.get_rotation_matrix_from_xyz((0,0,yaw))
            arrow.rotate(Rz)
            arrow.translate(t)
            arrow.paint_uniform_color([0.1, 0.1, 0.9])
            geoms.append(arrow)

    # --- Coordinate frame ---
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0,0,0])
    geoms.append(axis)

    # --- Visualization setup ---
    try:
        vis = o3d.visualization.Visualizer()
        window_created = vis.create_window("Waymo LiDAR Visualization (Tutorial Style)", width=1440, height=810)
        
        if not window_created:
            print("⚠️ Cannot create Open3D window (headless environment). Saving point cloud instead...")
            # Save point cloud as PLY file for offline viewing
            if len(geoms) > 0 and hasattr(geoms[0], 'points'):
                o3d.io.write_point_cloud("/tmp/waymo_pointcloud.ply", geoms[0])
                print("✅ Point cloud saved to /tmp/waymo_pointcloud.ply")
            return
        
        for g in geoms:
            vis.add_geometry(g)

        # Render options
        opt = vis.get_render_option()
        if opt is not None:
            opt.point_size = 2.0  # Slightly larger points for better visibility
            opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background like tutorial
            opt.show_coordinate_frame = True

        # Camera setup for bird's eye view (common in autonomous driving)
        ctr = vis.get_view_control()
        if ctr is not None:
            ctr.set_up([0, 0, 1])      # Z-up
            ctr.set_front([0, -1, 0])  # Look from positive Y towards negative Y
            ctr.set_lookat([0, 0, 0])  # Look at origin
            ctr.set_zoom(0.3)

        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(f"⚠️ Open3D visualization failed: {e}")
        print("This is normal in headless environments. Saving point cloud data instead...")
        
        # Fallback: save point cloud data
        if len(geoms) > 0:
            for i, geom in enumerate(geoms):
                if hasattr(geom, 'points') and len(geom.points) > 0:
                    filename = f"/tmp/waymo_geometry_{i}.ply"
                    o3d.io.write_point_cloud(filename, geom)
                    print(f"✅ Geometry {i} saved to {filename}")
        
        # Print summary statistics
        total_points = sum(len(g.points) if hasattr(g, 'points') else 0 for g in geoms)
        print(f"📊 Visualization Summary:")
        print(f"   • Total geometries: {len(geoms)}")
        print(f"   • Total points: {total_points:,}")
        if boxes3d is not None:
            print(f"   • Bounding boxes: {len(boxes3d)}")
        if calibs is not None:
            print(f"   • LiDAR sensors: {len(calibs)}")

import os
import numpy as np
import pyarrow.parquet as pq

def print_extrinsic_info(root, split="training", fname=None, frame_idx=0):
    """
    Full diagnostic for Waymo LiDAR parquet extrinsics and point ranges.
    Prints:
      - segment, timestamp
      - each LiDAR's yaw/translation/beam range
      - compare reshape order C vs F
      - point cloud range using both extr.T and inv(extr).T
    """

    lidar_dir = os.path.join(root, split, "lidar")
    calib_dir = os.path.join(root, split, "lidar_calibration")
    pose_dir  = os.path.join(root, split, "lidar_pose")
    box_dir   = os.path.join(root, split, "lidar_box")

    files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".parquet")])
    if not files:
        raise FileNotFoundError(f"No lidar parquet found in {lidar_dir}")
    if fname is None:
        fname = files[0]

    print(f"✅ Using lidar shard: {fname}")

    # --- lidar ---
    pf = pq.ParquetFile(os.path.join(lidar_dir, fname))
    df = pf.read_row_group(0).to_pandas()
    seg = df.iloc[0]["key.segment_context_name"]
    ts  = int(df.iloc[frame_idx]["key.frame_timestamp_micros"])
    lasers = sorted(df["key.laser_name"].unique())
    print(f"[lidar] segment={seg}, timestamp={ts}, lasers={lasers}")

    # --- calibration ---
    pf_cal = pq.ParquetFile(os.path.join(calib_dir, fname))
    df_cal = pf_cal.read_row_group(0).to_pandas()
    print(f"\n[lidar_calibration] rows={len(df_cal)}")

    # store extrinsics for later
    extr_dict = {}

    for lid in sorted(df_cal["key.laser_name"].unique()):
        crow = df_cal[df_cal["key.laser_name"] == lid].iloc[0]
        extr_col = max([c for c in crow.index if ("extrinsic" in c or str(c).endswith("item"))], key=len)
        extr_vals = crow[extr_col]
        if hasattr(extr_vals, "as_py"):
            extr_vals = extr_vals.as_py()

        # try both reshape orders
        extr_C = np.array(extr_vals, dtype=np.float32).reshape(4,4,order="C")
        extr_F = np.array(extr_vals, dtype=np.float32).reshape(4,4,order="F")

        # translation sanity
        tC, tF = extr_C[:3,3], extr_F[:3,3]
        yawC = np.rad2deg(np.arctan2(extr_C[1,0], extr_C[0,0]))
        yawF = np.rad2deg(np.arctan2(extr_F[1,0], extr_F[0,0]))

        print(f"\nlaser_id={lid}")
        print(f"  extr_C trans={np.round(tC,3)} yaw(deg)={yawC:6.1f}")
        print(f"  extr_F trans={np.round(tF,3)} yaw(deg)={yawF:6.1f}")

        # keep both versions for testing
        extr_dict[lid] = {"C": extr_C, "F": extr_F}

    # --- test one frame per LiDAR ---
    print("\n[range-image test per LiDAR] --------------------")
    for lid in lasers:
        rows = df[df["key.laser_name"] == lid]
        if len(rows) == 0:
            continue
        row = rows.iloc[frame_idx % len(rows)]
        ri = _decode_range_image(row)
        if ri is None:
            print(f"⚠️ laser {lid}: decode failed"); continue
        rng = np.nan_to_num(ri[...,0], nan=0.0, posinf=0.0, neginf=0.0)
        H,W = rng.shape
        beam_min = float(df_cal[df_cal["key.laser_name"]==lid].iloc[0]
                         ["[LiDARCalibrationComponent].beam_inclination.min"])
        beam_max = float(df_cal[df_cal["key.laser_name"]==lid].iloc[0]
                         ["[LiDARCalibrationComponent].beam_inclination.max"])
        inc = np.linspace(beam_min, beam_max, H, dtype=np.float32)[::-1].reshape(H,1)
        az  = np.linspace(np.pi, -np.pi, W, endpoint=False, dtype=np.float32)
        cos_i, sin_i = np.cos(inc), np.sin(inc)
        cos_a, sin_a = np.cos(az),  np.sin(az)
        x = rng * cos_i * cos_a
        y = rng * cos_i * sin_a
        z = rng * sin_i
        pts_h = np.stack([x,y,z,np.ones_like(z)], axis=-1).reshape(-1,4)

        for order in ["C","F"]:
            extr = extr_dict[lid][order]
            xyz1 = (pts_h @ extr.T)[:,:3]
            xyz2 = (pts_h @ np.linalg.inv(extr).T)[:,:3]
            print(f"\nlaser {lid} using order={order}")
            print(f"  -> extr.T  X[{xyz1[:,0].min():.1f},{xyz1[:,0].max():.1f}] "
                  f"Y[{xyz1[:,1].min():.1f},{xyz1[:,1].max():.1f}] "
                  f"Z[{xyz1[:,2].min():.1f},{xyz1[:,2].max():.1f}]")
            print(f"  -> inv(extr).T  X[{xyz2[:,0].min():.1f},{xyz2[:,0].max():.1f}] "
                  f"Y[{xyz2[:,1].min():.1f},{xyz2[:,1].max():.1f}] "
                  f"Z[{xyz2[:,2].min():.1f},{xyz2[:,2].max():.1f}]")

    # --- optional: pose / box consistency ---
    pose_path = os.path.join(pose_dir, fname)
    if os.path.exists(pose_path):
        pf_pose = pq.ParquetFile(pose_path)
        df_pose = pf_pose.read_row_group(0).to_pandas()
        same_seg = df_pose[df_pose["key.segment_context_name"]==seg]
        ts_match = same_seg[same_seg["key.frame_timestamp_micros"]==ts]
        print(f"\n[lidar_pose] rows in seg={len(same_seg)}, timestamp match={len(ts_match)}")

    pf_box = pq.ParquetFile(os.path.join(box_dir, fname))
    df_box = pf_box.read_row_group(0).to_pandas()
    same_seg = df_box[df_box["key.segment_context_name"]==seg]
    same_ts  = same_seg[same_seg["key.frame_timestamp_micros"]==ts]
    print(f"\n[lidar_box] same seg rows={len(same_seg)}, timestamp match={len(same_ts)}")
    if len(same_ts)>0:
        cz = same_ts["[LiDARBoxComponent].box.center.z"].mean()
        print(f"  mean box center z={cz:.2f}")

    print("\n✅ Finished extrinsic / range debug.")

import os, numpy as np, pyarrow.parquet as pq

def _decode_range_image(row):
    # 你已有的函数；这里占位
    vals1 = row["[LiDARComponent].range_image_return1.values"]
    shp1  = row["[LiDARComponent].range_image_return1.shape"]
    if hasattr(vals1, "as_py"): vals1 = vals1.as_py()
    if hasattr(shp1,  "as_py"): shp1  = shp1.as_py()
    H, W, C = shp1
    arr = np.array(vals1, dtype=np.float32).reshape(H, W, C)
    return arr  # [..., 0]=range, [..., 1]=intensity (如有)

def merge_and_debug_lidars_FIXED(root, split="training", fname=None, frame_idx=0):
    lidar_dir = os.path.join(root, split, "lidar")
    calib_dir = os.path.join(root, split, "lidar_calibration")

    files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".parquet")])
    if fname is None:
        fname = files[0]
    print(f"✅ Using lidar shard: {fname}")

    pf = pq.ParquetFile(os.path.join(lidar_dir, fname))
    df = pf.read_row_group(0).to_pandas()

    seg = df.iloc[0]["key.segment_context_name"]
    ts  = int(df.iloc[frame_idx]["key.frame_timestamp_micros"])
    print(f"segment={seg}, timestamp={ts}")

    # --- load calibration once, build dict per lidar_id ---
    pf_cal = pq.ParquetFile(os.path.join(calib_dir, fname))
    df_cal = pf_cal.read_row_group(0).to_pandas()

    calibs = {}
    for lid in sorted(df_cal["key.laser_name"].unique()):
        c_row = df_cal[df_cal["key.laser_name"] == lid].iloc[0]
        # extrinsic
        extr_col = max([c for c in c_row.index if ("extrinsic" in c or str(c).endswith("item"))], key=len)
        extr_vals = c_row[extr_col]
        if hasattr(extr_vals, "as_py"): extr_vals = extr_vals.as_py()
        extr = np.array(extr_vals, dtype=np.float32).reshape(4,4, order="C")  # ✅ order="C"
        R, t = extr[:3,:3], extr[:3,3]
        yaw_deg = float(np.rad2deg(np.arctan2(R[1,0], R[0,0])))
        # beam inclinations
        inc_min = float(c_row["[LiDARCalibrationComponent].beam_inclination.min"])
        inc_max = float(c_row["[LiDARCalibrationComponent].beam_inclination.max"])
        calibs[lid] = dict(extr=extr, t=t, yaw=yaw_deg, inc_min=inc_min, inc_max=inc_max)
        print(f"  lidar {lid}: yaw={yaw_deg:.1f}°, t={np.round(t,3)}, beam=({inc_min:.3f},{inc_max:.3f})")

    # --- decode each lidar at the same timestamp, convert to vehicle frame ---
    all_pts = []
    for lid, g in df.groupby("key.laser_name"):
        row = g[g["key.frame_timestamp_micros"] == ts]
        if len(row) == 0:
            print(f"⚠️ lidar {lid}: no row at ts={ts}")
            continue
        row = row.iloc[0]
        ri = _decode_range_image(row)
        if ri is None:
            print(f"⚠️ lidar {lid}: decode failed")
            continue
        rng = np.nan_to_num(ri[...,0], nan=0.0, posinf=0.0, neginf=0.0)
        inten = ri[...,1] if ri.shape[-1] > 1 else np.zeros_like(rng)
        H, W = rng.shape

        inc_min = calibs[lid]["inc_min"]  # ✅ per-lidar
        inc_max = calibs[lid]["inc_max"]
        incl = np.linspace(inc_min, inc_max, H, dtype=np.float32)[::-1].reshape(H,1)  # flip_rows=True
        az   = np.linspace(np.pi, -np.pi, W, endpoint=False, dtype=np.float32)       # flip_cols=False

        cos_i, sin_i = np.cos(incl), np.sin(incl)
        cos_a, sin_a = np.cos(az),   np.sin(az)

        x = rng * cos_i * cos_a
        y = rng * cos_i * sin_a
        z = rng * sin_i

        pts_h = np.stack([x,y,z,np.ones_like(z)], axis=-1).reshape(-1,4)
        extr  = calibs[lid]["extr"]
        xyz   = (pts_h @ extr.T)[:, :3]  # ✅ LiDAR→Vehicle (不要取逆)
        xyz   = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)

        inten = inten.reshape(-1,1).astype(np.float32)
        # NumPy 2.0 safe normalization
        i_min = float(np.min(inten)) if inten.size else 0.0
        i_max = float(np.max(inten)) if inten.size else 1.0
        inten = (inten - i_min) / ( (i_max - i_min) + 1e-6 )

        pts = np.concatenate([xyz, inten], axis=1)
        all_pts.append(pts)

        print(f"    lidar {lid}: N={pts.shape[0]:6d}  "
              f"X[{xyz[:,0].min():.1f},{xyz[:,0].max():.1f}] "
              f"Y[{xyz[:,1].min():.1f},{xyz[:,1].max():.1f}] "
              f"Z[{xyz[:,2].min():.1f},{xyz[:,2].max():.1f}]")

    if not all_pts:
        raise RuntimeError("No lidar decoded.")

    merged = np.concatenate(all_pts, axis=0)
    print(f"\n✅ merged: N={merged.shape[0]:,}  "
          f"X[{merged[:,0].min():.1f},{merged[:,0].max():.1f}] "
          f"Y[{merged[:,1].min():.1f},{merged[:,1].max():.1f}] "
          f"Z[{merged[:,2].min():.1f},{merged[:,2].max():.1f}]")
    return merged, calibs

import open3d as o3d

def visualize_merged_open3d_SAFE(merged_points, calibs=None, boxes3d=None, axis_size=5.0):
    geoms = []
    xyz = merged_points[:, :3]
    inten = merged_points[:, 3] if merged_points.shape[1] > 3 else np.ones(len(merged_points), np.float32)
    inten = np.asarray(inten, np.float32)
    i_min = float(np.min(inten)) if inten.size else 0.0
    i_max = float(np.max(inten)) if inten.size else 1.0
    inten_norm = (inten - i_min) / ( (i_max - i_min) + 1e-6 )

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(np.stack([inten_norm]*3, axis=1))
    geoms.append(pcd)

    # boxes: 按 Waymo Vehicle frame 绘制，不做额外 yaw 取反
    if boxes3d is not None and len(boxes3d) > 0:
        if hasattr(boxes3d, "detach"): boxes3d = boxes3d.detach().cpu().numpy()
        for b in boxes3d:
            x, y, z, dx, dy, dz, yaw = map(float, b[:7])
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[ c,-s,0],[s,c,0],[0,0,1]], np.float32)
            obb = o3d.geometry.OrientedBoundingBox(center=[x,y,z], R=R, extent=[dx,dy,dz])
            obb.color = (1.0, 0.0, 0.0)
            geoms.append(obb)

    if calibs is not None:
        for lid, v in calibs.items():
            t = np.asarray(v["t"], np.float32)
            yaw = np.deg2rad(v["yaw"])
            # 位置小球
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
            s.paint_uniform_color([0.1,0.8,0.1])
            s.translate(t); geoms.append(s)
            # 朝向箭头
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cone_height=0.2, cone_radius=0.05,
                cylinder_height=0.8, cylinder_radius=0.03)
            Rz = o3d.geometry.get_rotation_matrix_from_xyz((0,0,yaw))
            arrow.rotate(Rz); arrow.translate(t)
            arrow.paint_uniform_color([0.1,0.1,0.9])
            geoms.append(arrow)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0,0,0])
    geoms.append(axis)

    vis = o3d.visualization.Visualizer()
    vis.create_window("Waymo merged LiDAR + boxes", width=1440, height=810)
    for g in geoms: vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0.93,0.93,0.93])

    ctr = vis.get_view_control()
    ctr.set_up([0,0,1]); ctr.set_front([0,-1,0]); ctr.set_lookat([0,0,0]); ctr.set_zoom(0.5)

    vis.run(); vis.destroy_window()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/Datasets/waymodata/", help="Root of Waymo dataset")
    ap.add_argument("--split", default="training", help="Split folder (training/validation/testing)")
    args = ap.parse_args()

    print(f"\n🧩 Inspecting Waymo Dataset: {args.root}/{args.split}\n")
    inspect_camera_image(args.root, args.split)
    inspect_camera_box(args.root, args.split)
    inspect_camera_calibration(args.root, args.split)
    inspect_lidar(args.root, args.split)
    inspect_lidar_box(args.root, args.split)
    inspect_lidar_calibration(args.root, args.split)
    inspect_lidar_pose(args.root, args.split)
    inspect_vehicle_pose(args.root, args.split)
    inspect_lidar_camera_projection(args.root, args.split)
    inspect_lidar_hkp(args.root, args.split)
    inspect_segmentation(args.root, args.split)

if __name__ == "__main__":
    #print_waymo_debug("/data/Datasets/waymodata", split="training", frame_idx=0)
    # debug_waymo_full("/data/Datasets/waymodata", split="training",
    #              fname="10017090168044687777_6380_000_6400_000.parquet",
    #              frame_idx=0)
    # merged_pts, calibs = merge_and_debug_lidars(
    #     root="/data/Datasets/waymodata",
    #     split="training",
    #     fname="10017090168044687777_6380_000_6400_000.parquet",
    #     frame_idx=0,
    #     save_ply=True
    # )
    #visualize_merged_open3d(merged_pts, calibs)
    # print_extrinsic_info(
    #     root="/data/Datasets/waymodata",
    #     split="training",
    #     fname="10017090168044687777_6380_000_6400_000.parquet",
    #     frame_idx=0
    # )
    merged_pts, calibs = merge_and_debug_lidars_FIXED(
        root="/data/Datasets/waymodata",
        split="training",
        fname="10017090168044687777_6380_000_6400_000.parquet",
        frame_idx=0
    )

    # 若已有 boxes3d（vehicle frame）
    # visualize_merged_open3d_SAFE(merged_pts, calibs, boxes3d=target["boxes_3d"])
    visualize_merged_open3d_SAFE(merged_pts, calibs)

    #main()