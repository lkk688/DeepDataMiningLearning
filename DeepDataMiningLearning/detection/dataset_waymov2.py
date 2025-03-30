import os
import io
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
import pyarrow.parquet as pq
import torch
import torch.utils.data
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import cv2

# Configure plotting
plt.rcParams["figure.figsize"] = [32, 18]


def parse_range_image(range_image, camera_projection, range_image_pose=None):
    """
    Parse range image data from parquet files

    Args:
        range_image: Range image data
        camera_projection: Camera projection data
        range_image_pose: Range image pose data

    Returns:
        Parsed range image data
    """
    range_image_tensor = range_image
    range_image_mask = range_image_tensor[..., 0] > 0

    # Extract lidar point data
    point_data = {
        "range": range_image_tensor[..., 0],
        "intensity": range_image_tensor[..., 1],
        "elongation": range_image_tensor[..., 2],
        "x": range_image_tensor[..., 3],
        "y": range_image_tensor[..., 4],
        "z": range_image_tensor[..., 5],
    }

    return point_data, range_image_mask


class WaymoDatasetV2(torch.utils.data.Dataset):
    """PyTorch Dataset for Waymo Open Dataset v2 with parquet files"""

    def __init__(self, data_path, max_frames=None):
        """
        Initialize the dataset

        Args:
            data_path (str): Path to parquet files
            max_frames (int, optional): Maximum number of frames to load
        """
        self.data_path = data_path
        self.parquet_files = []
        self.segment_names = []

        # Find all parquet files in the directory
        if os.path.isdir(data_path):
            # Look for segment directories
            for item in os.listdir(data_path):
                segment_dir = os.path.join(data_path, item)
                if os.path.isdir(segment_dir):
                    # Check if this directory contains camera_image and other required files
                    camera_image_path = os.path.join(segment_dir, "camera_image")
                    if os.path.exists(camera_image_path):
                        self.segment_names.append(item)
            
            # If no segment directories found, look for parquet files directly
            if not self.segment_names:
                for root, _, files in os.walk(data_path):
                    for file in files:
                        if file.endswith(".parquet"):
                            self.parquet_files.append(os.path.join(root, file))
        elif os.path.isfile(data_path) and data_path.endswith(".parquet"):
            self.parquet_files = [data_path]

        # Sort for consistent loading
        self.segment_names.sort()
        self.parquet_files.sort()

        # Load metadata
        self.metadata = []
        self.frame_indices = []
        self._load_metadata(max_frames)
        
        # Store calibration information for each segment
        self.calibration_cache = {}
        
        # Import dask for parquet reading
        try:
            import dask.dataframe as dd
            self.dd = dd
        except ImportError:
            print("Warning: dask not installed. Please install with: pip install 'dask[complete]'")
            self.dd = None

    def _load_metadata(self, max_frames):
        """Load metadata from parquet files"""
        total_frames = 0

        # If we have segment names, use those
        if self.segment_names:
            for segment_idx, segment_name in enumerate(self.segment_names):
                # Add frame indices for this segment
                self.frame_indices.append((segment_idx, segment_name))
                total_frames += 1

                if max_frames is not None and total_frames >= max_frames:
                    return
        else:
            # Otherwise use parquet files directly
            for file_idx, path in enumerate(self.parquet_files):
                # Load parquet file
                try:
                    parquet_file = pq.ParquetFile(path)
                    num_rows = parquet_file.metadata.num_rows

                    # Add frame indices
                    for row_idx in range(num_rows):
                        self.frame_indices.append((file_idx, row_idx))
                        total_frames += 1

                        if max_frames is not None and total_frames >= max_frames:
                            return
                except Exception as e:
                    print(f"Error loading metadata from {path}: {e}")

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        """
        Get a frame from the dataset

        Args:
            idx (int): Index of the frame

        Returns:
            dict: Frame data
        """
        if self.dd is None:
            print("Error: dask not available. Cannot load data.")
            return self._create_empty_frame()
            
        # Get the frame index
        frame_info = self.frame_indices[idx]
        
        # Check if we're using segment names or file indices
        if len(frame_info) == 2 and isinstance(frame_info[1], str):
            # We're using segment names
            segment_idx, segment_name = frame_info
            return self._load_from_segment(segment_name)
        else:
            # We're using file indices
            file_idx, row_idx = frame_info
            parquet_path = self.parquet_files[file_idx]
            return self._load_from_parquet(parquet_path, row_idx)
    
    def _load_from_parquet(self, parquet_path, row_idx):
        """
        Load data from a parquet file at a specific row index
        
        Args:
            parquet_path: Path to the parquet file
            row_idx: Row index to load
            
        Returns:
            dict: Frame data
        """
        try:
            # Extract file_idx from the parquet_path
            file_idx = 0
            for idx, path in enumerate(self.parquet_files):
                if path == parquet_path:
                    file_idx = idx
                    break
                    
            # Use dask to read the parquet file
            df = self.dd.read_parquet(parquet_path)
            
            # Convert to pandas and get the specific row
            df_pd = df.compute()
            
            if row_idx >= len(df_pd):
                print(f"Error: Row index {row_idx} out of bounds for file with {len(df_pd)} rows")
                return self._create_empty_frame()
                
            row = df_pd.iloc[row_idx]
            
            # Extract frame data
            frame_data = {}
            for column in df_pd.columns:
                frame_data[column] = [row[column]]
            
            # Process camera images
            images = []
            image_names = []
            
            # Look for camera image data in the frame
            image_columns = [col for col in frame_data if isinstance(col, str) and 
                            'image' in col.lower() and 'name' not in col.lower()]
            
            for key in image_columns:
                if len(frame_data[key]) > 0:
                    camera_image = frame_data[key][0]
                    
                    # Extract camera name/index
                    camera_name_key = key.replace("image", "name")
                    if camera_name_key in frame_data and len(frame_data[camera_name_key]) > 0:
                        camera_name = frame_data[camera_name_key][0]
                    else:
                        # Extract camera index from key if name not available
                        camera_parts = key.split("_")
                        camera_name = camera_parts[1] if len(camera_parts) > 1 else "unknown"
                    
                    # Decode image
                    if isinstance(camera_image, bytes):
                        img = cv2.imdecode(
                            np.frombuffer(camera_image, np.uint8), cv2.IMREAD_COLOR
                        )
                        if img is None:
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                        
                        # Convert to PyTorch tensor and normalize
                        img_tensor = torch.from_numpy(img).float() / 255.0
                        # HWC to CHW format
                        img_tensor = img_tensor.permute(2, 0, 1)
                        
                        images.append(img_tensor)
                        image_names.append(camera_name)
            
            # Extract labels (bounding boxes)
            labels = []
            
            # Look for box data in the frame
            box_columns = {}
            for col in frame_data:
                if isinstance(col, str):
                    if 'center' in col.lower() and ('.x' in col.lower() or '_x' in col.lower()):
                        box_columns['center_x'] = col
                    elif 'center' in col.lower() and ('.y' in col.lower() or '_y' in col.lower()):
                        box_columns['center_y'] = col
                    elif 'center' in col.lower() and ('.z' in col.lower() or '_z' in col.lower()):
                        box_columns['center_z'] = col
                    elif 'length' in col.lower():
                        box_columns['length'] = col
                    elif 'width' in col.lower():
                        box_columns['width'] = col
                    elif 'height' in col.lower():
                        box_columns['height'] = col
                    elif 'heading' in col.lower():
                        box_columns['heading'] = col
                    elif 'type' in col.lower() and 'name' not in col.lower():
                        box_columns['type'] = col
            
            # If we have the necessary box columns, extract boxes
            if 'center_x' in box_columns and 'center_y' in box_columns and 'center_z' in box_columns:
                # This is a simplified approach - in a real implementation, you'd need to handle
                # multiple boxes per frame properly
                center_x = frame_data[box_columns['center_x']][0]
                center_y = frame_data[box_columns['center_y']][0]
                center_z = frame_data[box_columns['center_z']][0]
                
                length = frame_data[box_columns['length']][0] if 'length' in box_columns else 1.0
                width = frame_data[box_columns['width']][0] if 'width' in box_columns else 1.0
                height = frame_data[box_columns['height']][0] if 'height' in box_columns else 1.0
                
                heading = frame_data[box_columns['heading']][0] if 'heading' in box_columns else 0.0
                
                label_type = frame_data[box_columns['type']][0] if 'type' in box_columns else 0
                
                labels.append({
                    "center": [center_x, center_y, center_z],
                    "size": [length, width, height],
                    "heading": heading,
                    "type": label_type,
                    "name": self._get_label_name(label_type),
                })
            
            # Extract point cloud data
            point_cloud = self._extract_point_cloud_from_frame(frame_data)
            
            # Extract calibration information
            calibration = self._extract_calibration_from_frame(frame_data)
            
            # Get segment context name for metadata
            segment_context_name = None
            for col in frame_data:
                if isinstance(col, str) and 'segment' in col.lower() and 'name' in col.lower():
                    segment_context_name = frame_data[col][0]
                    break
            
            return {
                "images": images,
                "image_names": image_names,
                "point_cloud": point_cloud,
                "labels": labels,
                "calibration": calibration,
                "metadata": {key: frame_data[key][0] for key in frame_data},
                "frame_index": (file_idx, row_idx),
                "segment_context_name": segment_context_name
            }
            
        except Exception as e:
            print(f"Error loading from parquet file {parquet_path}: {e}")
            return self._create_empty_frame()
    
    def _extract_point_cloud_from_frame(self, frame_data):
        """Extract point cloud data from frame data"""
        # Look for point cloud data in the frame
        point_columns = {}
        for col in frame_data:
            if isinstance(col, str):
                if 'point' in col.lower() and '.x' in col.lower():
                    point_columns['x'] = col
                elif 'point' in col.lower() and '.y' in col.lower():
                    point_columns['y'] = col
                elif 'point' in col.lower() and '.z' in col.lower():
                    point_columns['z'] = col
                elif 'intensity' in col.lower():
                    point_columns['intensity'] = col
        
        # If we have the necessary point columns, extract points
        if 'x' in point_columns and 'y' in point_columns and 'z' in point_columns:
            # Get point coordinates
            points_x = np.array(frame_data[point_columns['x']][0])
            points_y = np.array(frame_data[point_columns['y']][0])
            points_z = np.array(frame_data[point_columns['z']][0])
            
            # Get intensity if available
            if 'intensity' in point_columns:
                intensity = np.array(frame_data[point_columns['intensity']][0])
            else:
                intensity = np.zeros_like(points_x)
            
            # Create point cloud tensor
            points = np.column_stack((points_x, points_y, points_z, intensity, 
                                      np.zeros_like(points_x), np.zeros_like(points_x)))
            return torch.from_numpy(points.astype(np.float32))
        else:
            return torch.zeros((0, 6))
    
    def _extract_calibration_from_frame(self, frame_data):
        """Extract calibration information from frame data"""
        calibration = {}
        
        # Look for calibration data in the frame
        for col in frame_data:
            if isinstance(col, str) and 'intrinsic' in col.lower():
                # Extract camera index
                camera_parts = col.split('_')
                camera_idx = int(camera_parts[1]) if len(camera_parts) > 1 and camera_parts[1].isdigit() else 0
                
                # Initialize camera calibration if not exists
                if camera_idx not in calibration:
                    calibration[camera_idx] = {}
                
                # Get intrinsic matrix
                intrinsic_data = frame_data[col][0]
                if isinstance(intrinsic_data, (list, np.ndarray)) and len(intrinsic_data) >= 9:
                    intrinsic = np.array(intrinsic_data[:9]).reshape(3, 3)
                else:
                    # Default intrinsic
                    intrinsic = np.array([
                        [1000, 0, 512],
                        [0, 1000, 512],
                        [0, 0, 1]
                    ])
                
                calibration[camera_idx]["intrinsic"] = intrinsic
            
            elif isinstance(col, str) and 'extrinsic' in col.lower():
                # Extract camera index
                camera_parts = col.split('_')
                camera_idx = int(camera_parts[1]) if len(camera_parts) > 1 and camera_parts[1].isdigit() else 0
                
                # Initialize camera calibration if not exists
                if camera_idx not in calibration:
                    calibration[camera_idx] = {}
                
                # Get extrinsic matrix
                extrinsic_data = frame_data[col][0]
                if isinstance(extrinsic_data, (list, np.ndarray)) and len(extrinsic_data) >= 16:
                    extrinsic = np.array(extrinsic_data[:16]).reshape(4, 4)
                else:
                    # Default extrinsic
                    extrinsic = np.array([
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]
                    ])
                
                calibration[camera_idx]["extrinsic"] = extrinsic
        
        return calibration
    
    def _get_label_name(self, label_type):
        """Map label type to name"""
        label_map = {
            0: "Unknown",
            1: "Vehicle",
            2: "Pedestrian",
            3: "Cyclist",
            4: "Sign",
        }
        return label_map.get(label_type, f"Type {label_type}")
    
    def _load_from_segment(self, segment_name):
        """
        Load data from a segment directory using the approach from test()
        
        Args:
            segment_name: Name of the segment
            
        Returns:
            dict: Frame data
        """
        # Create a function to read parquet files for this segment
        def read(tag):
            paths = f"{self.data_path}/{tag}/{segment_name}.parquet"
            try:
                return self.dd.read_parquet(paths)
            except Exception as e:
                print(f"Error reading {tag} for segment {segment_name}: {e}")
                return None
        
        # Read camera images
        cam_image_df = read('camera_image')
        if cam_image_df is None:
            return self._create_empty_frame()
        
        # Find the image column
        image_column = next((col for col in cam_image_df.columns 
                            if 'image' in col.lower() and 'name' not in col.lower()), None)
        if not image_column:
            print(f"Could not find image column for segment {segment_name}. Available columns: {list(cam_image_df.columns)}")
            return self._create_empty_frame()
        
        # Read box labels if available
        try:
            cam_box_df = read('camera_box')
            has_boxes = cam_box_df is not None
        except:
            has_boxes = False
            cam_box_df = None
        
        # Process camera images
        images = []
        image_names = []
        labels = []
        
        try:
            # Convert to pandas DataFrame for processing
            df_pd = cam_image_df.compute()
            
            # Find timestamp and camera columns
            timestamp_col = next((col for col in df_pd.columns if 'timestamp' in col.lower()), None)
            camera_col = next((col for col in df_pd.columns if 'camera' in col.lower() and 'name' in col.lower()), None)
            
            # Process each camera image
            for idx, row in df_pd.iterrows():
                try:
                    # Extract image data
                    image_bytes = row[image_column]
                    
                    if isinstance(image_bytes, bytes):
                        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                        
                        # Get camera name/id
                        if camera_col and camera_col in row:
                            camera_name = row[camera_col]
                        else:
                            camera_name = f"camera_{idx}"
                        
                        # Convert to PyTorch tensor and normalize
                        img_tensor = torch.from_numpy(img).float() / 255.0
                        # HWC to CHW format
                        img_tensor = img_tensor.permute(2, 0, 1)
                        
                        images.append(img_tensor)
                        image_names.append(camera_name)
                except Exception as e:
                    print(f"Error processing image at index {idx}: {e}")
            
            # Process box labels if available
            if has_boxes and cam_box_df is not None:
                try:
                    box_df = cam_box_df.compute()
                    
                    # Find box coordinate columns
                    center_x_col = next((col for col in box_df.columns 
                                        if 'center' in col.lower() and ('.x' in col.lower() or '_x' in col.lower())), None)
                    center_y_col = next((col for col in box_df.columns 
                                        if 'center' in col.lower() and ('.y' in col.lower() or '_y' in col.lower())), None)
                    center_z_col = next((col for col in box_df.columns 
                                        if 'center' in col.lower() and ('.z' in col.lower() or '_z' in col.lower())), None)
                    
                    # Find size columns
                    length_col = next((col for col in box_df.columns if 'length' in col.lower()), None)
                    width_col = next((col for col in box_df.columns if 'width' in col.lower()), None)
                    height_col = next((col for col in box_df.columns if 'height' in col.lower()), None)
                    
                    # Find heading column
                    heading_col = next((col for col in box_df.columns if 'heading' in col.lower()), None)
                    
                    # Find type column
                    type_col = next((col for col in box_df.columns if 'type' in col.lower() and 'name' not in col.lower()), None)
                    
                    if center_x_col and center_y_col and center_z_col:
                        for _, box_row in box_df.iterrows():
                            try:
                                center_x = float(box_row[center_x_col])
                                center_y = float(box_row[center_y_col])
                                center_z = float(box_row[center_z_col])
                                
                                # Get dimensions
                                length = float(box_row[length_col]) if length_col else 1.0
                                width = float(box_row[width_col]) if width_col else 1.0
                                height = float(box_row[height_col]) if height_col else 1.0
                                
                                # Get heading
                                heading = float(box_row[heading_col]) if heading_col else 0.0
                                
                                # Get type
                                label_type = int(box_row[type_col]) if type_col else 0
                                
                                labels.append({
                                    "center": [center_x, center_y, center_z],
                                    "size": [length, width, height],
                                    "heading": heading,
                                    "type": label_type,
                                    "name": self._get_label_name(label_type),
                                })
                            except Exception as e:
                                print(f"Error processing box: {e}")
                except Exception as e:
                    print(f"Error processing box labels: {e}")
        except Exception as e:
            print(f"Error processing camera images: {e}")
        
        # Extract LiDAR points if available
        point_cloud = self._extract_lidar_points_from_segment(segment_name)
        
        # Extract calibration information
        calibration = self._extract_calibration_from_segment(segment_name)
        
        return {
            "images": images,
            "image_names": image_names,
            "point_cloud": point_cloud,
            "labels": labels,
            "calibration": calibration,
            "metadata": {
                "segment_name": segment_name
            },
            "frame_index": (0, segment_name),
            "segment_context_name": segment_name
        }
    
    def _extract_lidar_points_from_segment(self, segment_name):
        """Extract LiDAR points from a segment"""
        try:
            # Try to read lidar data
            def read(tag):
                paths = f"{self.data_path}/{tag}/{segment_name}.parquet"
                try:
                    return self.dd.read_parquet(paths)
                except Exception as e:
                    print(f"Error reading {tag} for segment {segment_name}: {e}")
                    return None
            
            # Try different possible lidar file names
            lidar_tags = ['lidar', 'laser', 'point_cloud', 'lidar_points']
            lidar_df = None
            
            for tag in lidar_tags:
                lidar_df = read(tag)
                if lidar_df is not None:
                    break
            
            if lidar_df is None:
                return torch.zeros((0, 6))
            
            # Find point coordinate columns
            x_col = next((col for col in lidar_df.columns if '.x' in col.lower() or '_x' in col.lower()), None)
            y_col = next((col for col in lidar_df.columns if '.y' in col.lower() or '_y' in col.lower()), None)
            z_col = next((col for col in lidar_df.columns if '.z' in col.lower() or '_z' in col.lower()), None)
            
            # Find intensity column
            intensity_col = next((col for col in lidar_df.columns if 'intensity' in col.lower()), None)
            
            if x_col and y_col and z_col:
                df_pd = lidar_df.compute()
                
                # Extract point coordinates
                points_x = df_pd[x_col].values
                points_y = df_pd[y_col].values
                points_z = df_pd[z_col].values
                
                # Extract intensity if available
                if intensity_col:
                    intensity = df_pd[intensity_col].values
                else:
                    intensity = np.zeros_like(points_x)
                
                # Create point cloud tensor
                points = np.column_stack((points_x, points_y, points_z, intensity, np.zeros_like(points_x), np.zeros_like(points_x)))
                return torch.from_numpy(points.astype(np.float32))
            else:
                return torch.zeros((0, 6))
        except Exception as e:
            print(f"Error extracting LiDAR points: {e}")
            return torch.zeros((0, 6))
    
    def _extract_calibration_from_segment(self, segment_name):
        """Extract calibration information from a segment"""
        # Check if we have cached calibration for this segment
        if segment_name in self.calibration_cache:
            return self.calibration_cache[segment_name]
        
        calibration = {}
        
        try:
            # Try to read calibration data
            def read(tag):
                paths = f"{self.data_path}/{tag}/{segment_name}.parquet"
                try:
                    return self.dd.read_parquet(paths)
                except Exception as e:
                    return None
            
            # Try different possible calibration file names
            calib_tags = ['calibration', 'calib', 'camera_calibration']
            calib_df = None
            
            for tag in calib_tags:
                calib_df = read(tag)
                if calib_df is not None:
                    break
            
            if calib_df is not None:
                df_pd = calib_df.compute()
                
                # Find intrinsic and extrinsic columns
                for col in df_pd.columns:
                    if 'intrinsic' in col.lower():
                        # Extract camera index
                        camera_parts = col.split('_')
                        camera_idx = int(camera_parts[1]) if len(camera_parts) > 1 and camera_parts[1].isdigit() else 0
                        
                        # Initialize camera calibration if not exists
                        if camera_idx not in calibration:
                            calibration[camera_idx] = {}
                        
                        # Get intrinsic matrix
                        intrinsic_data = df_pd[col].iloc[0]
                        if isinstance(intrinsic_data, (list, np.ndarray)) and len(intrinsic_data) >= 9:
                            intrinsic = np.array(intrinsic_data[:9]).reshape(3, 3)
                        else:
                            # Default intrinsic
                            intrinsic = np.array([
                                [1000, 0, 512],
                                [0, 1000, 512],
                                [0, 0, 1]
                            ])
                        
                        calibration[camera_idx]["intrinsic"] = intrinsic
                    
                    elif 'extrinsic' in col.lower():
                        # Extract camera index
                        camera_parts = col.split('_')
                        camera_idx = int(camera_parts[1]) if len(camera_parts) > 1 and camera_parts[1].isdigit() else 0
                        
                        # Initialize camera calibration if not exists
                        if camera_idx not in calibration:
                            calibration[camera_idx] = {}
                        
                        # Get extrinsic matrix
                        extrinsic_data = df_pd[col].iloc[0]
                        if isinstance(extrinsic_data, (list, np.ndarray)) and len(extrinsic_data) >= 16:
                            extrinsic = np.array(extrinsic_data[:16]).reshape(4, 4)
                        else:
                            # Default extrinsic
                            extrinsic = np.array([
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1]
                            ])
                        
                        calibration[camera_idx]["extrinsic"] = extrinsic
        except Exception as e:
            print(f"Error extracting calibration: {e}")
        
        # Cache the calibration for future use
        self.calibration_cache[segment_name] = calibration
        
        return calibration
    
    def _create_empty_frame(self):
        """Create an empty frame data structure"""
        return {
            "images": [],
            "image_names": [],
            "point_cloud": torch.zeros((0, 6)),
            "labels": [],
            "calibration": {},
            "metadata": {},
            "frame_index": (0, 0),
            "segment_context_name": ""
        }


def visualize_camera_image(frame_data, camera_index=0):
    """
    Visualize a camera image with 3D bounding box projections

    Args:
        frame_data: Dictionary containing frame data
        camera_index: Camera image index
    """
    if len(frame_data["images"]) <= camera_index:
        print(f"No camera image at index {camera_index}")
        return

    # Get image tensor and convert to numpy
    img_tensor = frame_data["images"][camera_index]
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    plt.figure(figsize=(16, 12))
    plt.imshow(img)
    camera_name = (
        frame_data["image_names"][camera_index]
        if "image_names" in frame_data
        else f"Camera {camera_index}"
    )
    plt.title(f"Camera Image: {camera_name}")

    # Draw 2D bounding boxes if available
    # In v2, we need to check if we have 2D boxes for this specific camera
    camera_labels = []
    for label in frame_data["labels"]:
        # Check if this label has 2D box information for this camera
        if "boxes_2d" in label and camera_index in label["boxes_2d"]:
            camera_labels.append(label)

    for label in camera_labels:
        box_2d = label["boxes_2d"][camera_index]
        # Extract box parameters
        x, y, width, height = box_2d

        # Create and draw rectangle
        rect = patches.Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        plt.gca().add_patch(rect)

        # Add label text
        plt.text(
            x - width / 2, y - height / 2 - 10, label["name"], color="red", fontsize=10
        )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_lidar_points(frame_data, ax=None):
    """
    Visualize LiDAR points in 3D space

    Args:
        frame_data: Dictionary containing frame data
        ax: Matplotlib 3D axis (optional)
    """
    # Get point cloud
    points_tensor = frame_data["point_cloud"]
    points_all = points_tensor.numpy()

    # Create figure if not provided
    if ax is None:
        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection="3d")

    # Plot LiDAR points (downsample for visualization)
    point_step = 100  # Adjust for more or fewer points
    ax.scatter(
        points_all[::point_step, 0],  # x
        points_all[::point_step, 1],  # y
        points_all[::point_step, 2],  # z
        s=0.5,
        c=(
            points_all[::point_step, 3] if points_all.shape[1] > 3 else "gray"
        ),  # color by intensity if available
        cmap="viridis",
        alpha=0.5,
    )

    # Plot 3D bounding boxes
    for label in frame_data["labels"]:
        center = label["center"]
        dimensions = label["size"]
        heading = label["heading"]

        # Get the 8 corners of the box
        corners = get_3d_box_corners(center, dimensions, heading)

        # Draw box edges
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Connecting edges
        ]

        for i, j in edges:
            ax.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                "r-",
            )

    # Set axis labels and properties
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    ax.set_title("LiDAR Point Cloud with 3D Bounding Boxes")

    # Set axis limits for better visualization
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-5, 5])

    return ax


def get_3d_box_corners(center, dimensions, heading):
    """
    Get the 8 corners of a 3D bounding box

    Args:
        center: Box center [x, y, z]
        dimensions: Box dimensions [length, width, height]
        heading: Box heading (rotation around Z axis)

    Returns:
        corners: 8x3 array of corner coordinates
    """
    # Box dimensions
    length, width, height = dimensions

    # Create box corners
    x_corners = [
        length / 2,
        length / 2,
        -length / 2,
        -length / 2,
        length / 2,
        length / 2,
        -length / 2,
        -length / 2,
    ]
    y_corners = [
        width / 2,
        -width / 2,
        -width / 2,
        width / 2,
        width / 2,
        -width / 2,
        -width / 2,
        width / 2,
    ]
    z_corners = [0, 0, 0, 0, height, height, height, height]

    # Combine corners
    corners = np.vstack([x_corners, y_corners, z_corners]).T

    # Apply rotation
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)
    rotation_matrix = np.array(
        [[cos_heading, -sin_heading, 0], [sin_heading, cos_heading, 0], [0, 0, 1]]
    )

    corners = np.dot(corners, rotation_matrix.T)

    # Apply translation
    corners += np.array(center)

    return corners


def visualize_bird_eye_view(frame_data):
    """
    Visualize bird's eye view of LiDAR points and bounding boxes

    Args:
        frame_data: Dictionary containing frame data
    """
    # Get point cloud
    points_tensor = frame_data["point_cloud"]
    points_all = points_tensor.numpy()

    # Create figure
    plt.figure(figsize=(16, 16))

    # Plot LiDAR points (bird's eye view)
    point_step = 10  # Adjust for more or fewer points
    plt.scatter(
        points_all[::point_step, 0],  # x (forward)
        points_all[::point_step, 1],  # y (left)
        s=0.1,
        c=(
            points_all[::point_step, 3] if points_all.shape[1] > 3 else "gray"
        ),  # color by intensity if available
        cmap="viridis",
        alpha=0.5,
    )

    # Plot 2D bounding boxes (top-down view)
    for label in frame_data["labels"]:
        center_x, center_y, _ = label["center"]
        length, width, _ = label["size"]
        heading = label["heading"]

        # Calculate corner coordinates
        cos_heading = math.cos(heading)
        sin_heading = math.sin(heading)

        # Create rectangle
        corners = np.array(
            [
                [-length / 2, -width / 2],
                [length / 2, -width / 2],
                [length / 2, width / 2],
                [-length / 2, width / 2],
                [-length / 2, -width / 2],  # Close the rectangle
            ]
        )

        # Rotate and translate corners
        for i in range(len(corners)):
            x, y = corners[i]
            corners[i, 0] = x * cos_heading - y * sin_heading + center_x
            corners[i, 1] = x * sin_heading + y * cos_heading + center_y

        # Plot box
        plt.plot(corners[:, 0], corners[:, 1], "r-", linewidth=2)

        # Add label text
        plt.text(center_x, center_y, label["name"], color="blue", fontsize=8)

    # Set axis properties
    plt.axis("equal")
    plt.xlim([-20, 50])
    plt.ylim([-20, 20])
    plt.xlabel("X (meters) - Forward")
    plt.ylabel("Y (meters) - Left")
    plt.title("Bird's Eye View - LiDAR and Bounding Boxes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_combined_visualization(frame_data, camera_index=0):
    """
    Create a combined visualization with camera image, LiDAR, and bird's eye view

    Args:
        frame_data: Dictionary containing frame data
        camera_index: Camera image index
    """
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(24, 8))

    # 1. Camera image with 2D boxes
    if len(frame_data["images"]) > camera_index:
        img_tensor = frame_data["images"][camera_index]
        img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        ax1 = fig.add_subplot(131)
        ax1.imshow(img)
        camera_name = (
            frame_data["image_names"][camera_index]
            if "image_names" in frame_data
            else f"Camera {camera_index}"
        )
        ax1.set_title(f"Camera: {camera_name}")

        # Draw bounding boxes
        camera_labels = []
        for label in frame_data["labels"]:
            if "boxes_2d" in label and camera_index in label["boxes_2d"]:
                camera_labels.append(label)

        for label in camera_labels:
            box_2d = label["boxes_2d"][camera_index]
            x, y, width, height = box_2d

            rect = patches.Rectangle(
                (x - width / 2, y - height / 2),
                width,
                height,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax1.add_patch(rect)

        ax1.axis("off")
    else:
        ax1 = fig.add_subplot(131)
        ax1.text(0.5, 0.5, "No camera image available", ha="center")
        ax1.axis("off")

    # 2. 3D LiDAR visualization
    ax2 = fig.add_subplot(132, projection="3d")
    visualize_lidar_points(frame_data, ax=ax2)

    # 3. Bird's eye view
    ax3 = fig.add_subplot(133)

    # Get point cloud
    points_tensor = frame_data["point_cloud"]
    points_all = points_tensor.numpy()

    # Plot LiDAR points (bird's eye view)
    point_step = 10
    ax3.scatter(
        points_all[::point_step, 0],
        points_all[::point_step, 1],
        s=0.1,
        c=points_all[::point_step, 3] if points_all.shape[1] > 3 else "gray",
        cmap="viridis",
        alpha=0.5,
    )

    # Plot 2D bounding boxes
    for label in frame_data["labels"]:
        center_x, center_y, _ = label["center"]
        length, width, _ = label["size"]
        heading = label["heading"]

        cos_heading = math.cos(heading)
        sin_heading = math.sin(heading)

        corners = np.array(
            [
                [-length / 2, -width / 2],
                [length / 2, -width / 2],
                [length / 2, width / 2],
                [-length / 2, width / 2],
                [-length / 2, -width / 2],
            ]
        )

        for i in range(len(corners)):
            x, y = corners[i]
            corners[i, 0] = x * cos_heading - y * sin_heading + center_x
            corners[i, 1] = x * sin_heading + y * cos_heading + center_y

        ax3.plot(corners[:, 0], corners[:, 1], "r-", linewidth=2)

    ax3.set_aspect("equal")
    ax3.set_xlim([-20, 50])
    ax3.set_ylim([-20, 20])
    ax3.set_xlabel("X (meters)")
    ax3.set_ylabel("Y (meters)")
    ax3.set_title("Bird's Eye View")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


# Example usage showing how to convert to PyTorch tensors for training
def prepare_batch_for_model(batch_data):
    """
    Prepare a batch of data for model training

    Args:
        batch_data: Batch of data from DataLoader

    Returns:
        inputs: Model inputs
        targets: Model targets
    """
    # Example for object detection task
    inputs = {
        "images": [sample["images"] for sample in batch_data],
        "point_clouds": [sample["point_cloud"] for sample in batch_data],
    }

    # Example targets (bounding boxes)
    targets = []
    for sample in batch_data:
        sample_targets = []
        for label in sample["labels"]:
            sample_targets.append(
                {
                    "boxes": torch.tensor(
                        label["center"] + label["size"]
                    ),  # [x, y, z, l, w, h]
                    "labels": torch.tensor(label["type"]),
                    "heading": torch.tensor(label["heading"]),
                }
            )
        targets.append(sample_targets)

    return inputs, targets

def test():
    import numpy as np
    import warnings
    import dask.dataframe as dd #python -m pip install "dask[complete]"    # Install everything
    import matplotlib.pyplot as plt
    from matplotlib import patches
    import cv2
    import os
    from PIL import Image
    import io
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Example dataset directory and context name
    dataset_dir = "/mnt/e/Dataset/waymodata/training"
    context_name = "11076364019363412893_1711_000_1731_000"
    output_dir = "/mnt/e/Dataset/waymodata/visualization"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the parquet files
    def read(tag: str) -> np.ndarray:
        """
        Creates a NumPy array for the component specified by its tag.
        Reads the data from files and converts it into a NumPy array.
        """
        paths = f"{dataset_dir}/{tag}/{context_name}.parquet"
        return dd.read_parquet(paths)

    # Lazily read camera images and boxes 
    cam_image_df = read('camera_image')
    
    # Print column names to debug
    print("Camera image columns:", list(cam_image_df.columns))
    
    # Find the image column name (it might be 'image', 'camera_image', etc.)
    image_column = next((col for col in cam_image_df.columns if 'image' in col.lower() and 'name' not in col.lower()), None)
    if not image_column:
        print("Could not find image column. Available columns:", list(cam_image_df.columns))
        return
    
    print(f"Using image column: {image_column}")
    
    # Get all camera images
    camera_image_df = cam_image_df
    
    # Read box labels
    cam_box_df = read('camera_box')
    print("Camera box columns:", list(cam_box_df.columns))
    
    # Find common join keys
    common_keys = []
    for col in cam_image_df.columns:
        if col in cam_box_df.columns and ('key' in col or 'id' in col or 'timestamp' in col):
            common_keys.append(col)
    
    print("Common join keys:", common_keys)
    
    # If no common keys found, use default keys
    if not common_keys:
        common_keys = [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.camera_name',
        ]
    
    # Inner join the camera_image table with the camera_box table
    try:
        df = camera_image_df.merge(
            cam_box_df,
            on=common_keys,
            how='inner',
        )
        
        # Convert to pandas DataFrame for easier processing
        df_pd = df.compute()
        print(f"Joined dataframe has {len(df_pd)} rows")
        
        # If the dataframe is empty, try a different approach
        if len(df_pd) == 0:
            print("Joined dataframe is empty. Trying to process images without boxes.")
            df_pd = camera_image_df.compute()
    
    except Exception as e:
        print(f"Error joining dataframes: {e}")
        print("Processing only camera images without boxes")
        df_pd = camera_image_df.compute()
    
    # Group by appropriate columns for processing
    # First, find timestamp and camera columns
    timestamp_col = next((col for col in df_pd.columns if 'timestamp' in col.lower()), None)
    camera_col = next((col for col in df_pd.columns if 'camera' in col.lower() and 'name' in col.lower()), None)
    
    if timestamp_col and camera_col:
        print(f"Grouping by {timestamp_col} and {camera_col}")
        grouped = df_pd.groupby([timestamp_col, camera_col])
    else:
        # If we can't find appropriate columns, just process each row
        print("Could not find timestamp or camera columns for grouping")
        # Create a dummy groupby object
        df_pd['_dummy_group'] = 1
        grouped = [(('unknown', 'unknown'), df_pd)]
    
    # Process each group (each image with its boxes)
    for group_key, group in grouped:
        # Get the first row which contains the image data
        row = group.iloc[0]
        
        # Extract and decode the image
        try:
            # Try to get image data using the identified column
            image_bytes = row[image_column]
            
            if isinstance(image_bytes, bytes):
                img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            else:
                print(f"Image data not available or not in bytes format: {type(image_bytes)}")
                continue
                
        except Exception as e:
            print(f"Error decoding image: {e}")
            continue
        
        # Create figure for visualization
        plt.figure(figsize=(16, 12))
        plt.imshow(img)
        
        # Get timestamp and camera info for title
        timestamp, camera_id = group_key
        
        # Get camera name for title
        camera_name_map = {
            1: "FRONT",
            2: "FRONT_LEFT",
            3: "FRONT_RIGHT",
            4: "SIDE_LEFT",
            5: "SIDE_RIGHT"
        }
        camera_name = camera_name_map.get(camera_id, f"Camera {camera_id}")
        plt.title(f"Camera: {camera_name}, Timestamp: {timestamp}")
        
        # Draw bounding boxes for this image if available
        box_count = 0
        for _, box_row in group.iterrows():
            # Skip the first row (which is the same as 'row')
            if box_row.equals(row) and box_count > 0:
                continue
                
            box_count += 1
            
            # Try to find box coordinates
            try:
                # Get all column names
                all_cols = list(box_row.index)
                
                # Find center coordinates
                center_x_col = next((col for col in all_cols if 'center' in col.lower() and ('.x' in col.lower() or '_x' in col.lower())), None)
                center_y_col = next((col for col in all_cols if 'center' in col.lower() and ('.y' in col.lower() or '_y' in col.lower())), None)
                
                # Find size/dimensions
                width_col = next((col for col in all_cols if ('width' in col.lower() or 'size' in col.lower() and '.x' in col.lower())), None)
                height_col = next((col for col in all_cols if ('height' in col.lower() or 'length' in col.lower() or 'size' in col.lower() and '.y' in col.lower())), None)
                
                # Find label type
                type_col = next((col for col in all_cols if 'type' in col.lower() and 'name' not in col.lower()), None)
                
                if center_x_col and center_y_col:
                    center_x = box_row[center_x_col]
                    center_y = box_row[center_y_col]
                    
                    # Get width and height
                    width = box_row[width_col] if width_col else 50  # Default if not found
                    height = box_row[height_col] if height_col else 50  # Default if not found
                    
                    # Get label type
                    label_type = box_row[type_col] if type_col else 0
                    
                    # Map label type to name
                    label_map = {
                        0: "Unknown",
                        1: "Vehicle",
                        2: "Pedestrian",
                        3: "Cyclist",
                        4: "Sign",
                    }
                    label_name = label_map.get(label_type, f"Type {label_type}")
                    
                    # Create and draw rectangle
                    rect = patches.Rectangle(
                        (center_x - width/2, center_y - height/2),
                        width, height,
                        linewidth=2,
                        edgecolor='r',
                        facecolor='none'
                    )
                    plt.gca().add_patch(rect)
                    
                    # Add label text
                    plt.text(
                        center_x - width/2, center_y - height/2 - 10,
                        label_name, color='red', fontsize=10
                    )
            except Exception as e:
                print(f"Error processing box: {e}")
                continue
        
        # Save the visualization
        timestamp_str = str(timestamp).replace(":", "-").replace(" ", "_")
        camera_str = str(camera_id).replace(":", "-").replace(" ", "_")
        output_filename = f"{output_dir}/camera_{camera_str}_timestamp_{timestamp_str}.png"
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        
        print(f"Saved visualization to {output_filename}")
    
    print("Visualization complete!")

def visualize_lidar_on_image(frame_data, camera_index=0):
    """
    Visualize LiDAR points projected onto a camera image
    
    Args:
        frame_data: Dictionary containing frame data
        camera_index: Camera image index
    """
    if len(frame_data["images"]) <= camera_index:
        print(f"No camera image at index {camera_index}")
        return
    
    # Get image tensor and convert to numpy
    img_tensor = frame_data["images"][camera_index]
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Get point cloud
    points_tensor = frame_data["point_cloud"]
    points_all = points_tensor.numpy()
    
    # Get camera calibration information
    # In a real implementation, this would come from the dataset
    # For Waymo dataset, we need extrinsic and intrinsic matrices
    if "calibration" in frame_data and camera_index in frame_data["calibration"]:
        calibration = frame_data["calibration"][camera_index]
    else:
        # If calibration is not available, we'll use placeholder values
        # This is just for demonstration - in practice, you need actual calibration data
        print("Warning: Using placeholder calibration values")
        # Placeholder intrinsic matrix (focal length, principal point)
        intrinsic = np.array([
            [1000, 0, img.shape[1]/2],
            [0, 1000, img.shape[0]/2],
            [0, 0, 1]
        ])
        # Placeholder extrinsic matrix (rotation, translation from LiDAR to camera)
        # This assumes LiDAR is at the origin and camera is looking forward
        extrinsic = np.array([
            [0, -1, 0, 0],  # x_cam = -y_lidar
            [0, 0, -1, 0],  # y_cam = -z_lidar
            [1, 0, 0, 0],   # z_cam = x_lidar
            [0, 0, 0, 1]
        ])
        calibration = {"intrinsic": intrinsic, "extrinsic": extrinsic}
    
    # Project 3D points to 2D image plane
    projected_points = []
    projected_depths = []
    
    # Get only points in front of the camera (positive z in camera frame)
    for i in range(0, len(points_all), 10):  # Downsample for efficiency
        point = points_all[i]
        x, y, z = point[:3]
        
        # Convert from LiDAR to camera coordinate system
        # This is a simplified version - actual transformation depends on calibration
        point_lidar = np.array([x, y, z, 1.0])
        point_camera = np.dot(calibration["extrinsic"], point_lidar)
        
        # Skip points behind the camera
        if point_camera[2] <= 0:
            continue
        
        # Project to image plane
        point_normalized = point_camera[:3] / point_camera[2]  # Normalize by depth
        point_image = np.dot(calibration["intrinsic"], point_normalized)
        
        # Check if point is within image bounds
        if (0 <= point_image[0] < img.shape[1] and 
            0 <= point_image[1] < img.shape[0]):
            projected_points.append((int(point_image[0]), int(point_image[1])))
            projected_depths.append(point_camera[2])  # Store depth for coloring
    
    # Create figure
    plt.figure(figsize=(16, 12))
    plt.imshow(img)
    
    # Normalize depths for coloring
    if projected_depths:
        min_depth = min(projected_depths)
        max_depth = max(projected_depths)
        normalized_depths = [(d - min_depth) / (max_depth - min_depth) if max_depth > min_depth else 0.5 for d in projected_depths]
    
        # Plot projected points with depth-based coloring
        for i, (px, py) in enumerate(projected_points):
            plt.scatter(px, py, c=[normalized_depths[i]], cmap='jet', s=1, alpha=0.5)
    
    camera_name = (
        frame_data["image_names"][camera_index]
        if "image_names" in frame_data
        else f"Camera {camera_index}"
    )
    plt.title(f"LiDAR Points Projected on Camera: {camera_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.colorbar(label="Depth (normalized)")
    plt.show()

def visualize_lidar_on_image_with_boxes(frame_data, camera_index=0):
    """
    Visualize LiDAR points and 3D bounding boxes projected onto a camera image
    
    Args:
        frame_data: Dictionary containing frame data
        camera_index: Camera image index
    """
    if len(frame_data["images"]) <= camera_index:
        print(f"No camera image at index {camera_index}")
        return
    
    # Get image tensor and convert to numpy
    img_tensor = frame_data["images"][camera_index]
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Create figure
    plt.figure(figsize=(16, 12))
    plt.imshow(img)
    
    # Get camera calibration information
    if "calibration" in frame_data and camera_index in frame_data["calibration"]:
        calibration = frame_data["calibration"][camera_index]
    else:
        # Placeholder calibration
        intrinsic = np.array([
            [1000, 0, img.shape[1]/2],
            [0, 1000, img.shape[0]/2],
            [0, 0, 1]
        ])
        extrinsic = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        calibration = {"intrinsic": intrinsic, "extrinsic": extrinsic}
    
    # Project LiDAR points
    points_tensor = frame_data["point_cloud"]
    points_all = points_tensor.numpy()
    
    # Downsample and project points
    point_step = 20  # Adjust for more or fewer points
    projected_points = []
    projected_depths = []
    
    for i in range(0, len(points_all), point_step):
        point = points_all[i]
        x, y, z = point[:3]
        
        # Convert to camera coordinates
        point_lidar = np.array([x, y, z, 1.0])
        point_camera = np.dot(calibration["extrinsic"], point_lidar)
        
        if point_camera[2] <= 0:
            continue
        
        # Project to image
        point_normalized = point_camera[:3] / point_camera[2]
        point_image = np.dot(calibration["intrinsic"], point_normalized)
        
        if (0 <= point_image[0] < img.shape[1] and 
            0 <= point_image[1] < img.shape[0]):
            projected_points.append((int(point_image[0]), int(point_image[1])))
            projected_depths.append(point_camera[2])
    
    # Normalize depths for coloring
    if projected_depths:
        min_depth = min(projected_depths)
        max_depth = max(projected_depths)
        normalized_depths = [(d - min_depth) / (max_depth - min_depth) if max_depth > min_depth else 0.5 for d in projected_depths]
    
        # Plot projected points
        for i, (px, py) in enumerate(projected_points):
            plt.scatter(px, py, c=[normalized_depths[i]], cmap='jet', s=1, alpha=0.5)
    
    # Project 3D bounding boxes to image
    for label in frame_data["labels"]:
        center = label["center"]
        dimensions = label["size"]
        heading = label["heading"]
        
        # Get 3D box corners
        corners_3d = get_3d_box_corners(center, dimensions, heading)
        
        # Project corners to image
        corners_2d = []
        for corner in corners_3d:
            # Convert to homogeneous coordinates
            point_lidar = np.append(corner, 1.0)
            
            # Transform to camera coordinates
            point_camera = np.dot(calibration["extrinsic"], point_lidar)
            
            # Skip if behind camera
            if point_camera[2] <= 0:
                corners_2d = []  # Skip this box if any corner is behind camera
                break
            
            # Project to image
            point_normalized = point_camera[:3] / point_camera[2]
            point_image = np.dot(calibration["intrinsic"], point_normalized)
            
            corners_2d.append((int(point_image[0]), int(point_image[1])))
        
        # Draw 3D box if all corners are visible
        if corners_2d:
            # Define edges for visualization
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                (0, 4), (1, 5), (2, 6), (3, 7),  # Connecting edges
            ]
            
            # Draw edges
            for i, j in edges:
                plt.plot(
                    [corners_2d[i][0], corners_2d[j][0]],
                    [corners_2d[i][1], corners_2d[j][1]],
                    'g-', linewidth=1
                )
            
            # Add label text at the center of the box
            center_x = sum(c[0] for c in corners_2d) / len(corners_2d)
            center_y = sum(c[1] for c in corners_2d) / len(corners_2d)
            plt.text(
                center_x, center_y, label["name"],
                color='green', fontsize=10, 
                bbox=dict(facecolor='black', alpha=0.5)
            )
    
    camera_name = (
        frame_data["image_names"][camera_index]
        if "image_names" in frame_data
        else f"Camera {camera_index}"
    )
    plt.title(f"LiDAR and 3D Boxes Projected on Camera: {camera_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.colorbar(label="Depth (normalized)")
    plt.show()

def extract_calibration_from_waymo(frame_data):
    """
    Extract calibration information from Waymo dataset frame
    
    Args:
        frame_data: Dictionary containing frame data
        
    Returns:
        Dictionary of calibration information for each camera
    """
    calibration = {}
    
    # Check if calibration data is available in the frame
    if "metadata" in frame_data:
        metadata = frame_data["metadata"]
        
        # Look for camera calibration data
        for key in metadata:
            if "camera" in key.lower() and "calibration" in key.lower():
                camera_id = int(key.split("_")[1]) if "_" in key else 0
                
                # Extract intrinsic parameters
                intrinsic_key = f"camera_{camera_id}_intrinsic"
                if intrinsic_key in metadata:
                    intrinsic = np.array(metadata[intrinsic_key]).reshape(3, 3)
                else:
                    # Default intrinsic if not found
                    intrinsic = np.array([
                        [1000, 0, 512],
                        [0, 1000, 512],
                        [0, 0, 1]
                    ])
                
                # Extract extrinsic parameters (lidar to camera transform)
                extrinsic_key = f"camera_{camera_id}_extrinsic"
                if extrinsic_key in metadata:
                    extrinsic = np.array(metadata[extrinsic_key]).reshape(4, 4)
                else:
                    # Default extrinsic if not found
                    extrinsic = np.array([
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]
                    ])
                
                calibration[camera_id] = {
                    "intrinsic": intrinsic,
                    "extrinsic": extrinsic
                }
    
    return calibration

# Main execution
def main():
    """
    Main function to demonstrate loading and visualizing Waymo Open Dataset v2
    """
    # Path to your downloaded Waymo Open Dataset v2 parquet files
    data_path = "/mnt/e/Dataset/waymodata/training"

    # Create dataset (limit to 5 frames for demonstration)
    dataset = WaymoDatasetV2(data_path, max_frames=5)
    print(f"Loaded {len(dataset)} frames")

    # Try multiple frames until we find one with camera images
    frame_with_images = None
    for i in range(min(len(dataset), 10)):  # Try up to 10 frames
        frame_data = dataset[i]
        if len(frame_data["images"]) > 0:
            frame_with_images = frame_data
            print(f"Found frame with camera images at index {i}")
            break
    
    # If we couldn't find any frames with images, use the first frame anyway
    if frame_with_images is None:
        print("Warning: Could not find any frames with camera images in the first 10 frames")
        if len(dataset) > 0:
            frame_with_images = dataset[0]
        else:
            print("Error: Dataset is empty")
            return
    
    # 1. Visualize Camera Image
    print("Visualizing camera image...")
    if len(frame_with_images["images"]) > 0:
        visualize_camera_image(frame_with_images, camera_index=0)
    else:
        print("No camera images available in this frame")

    # 2. Visualize LiDAR Points
    print("Visualizing LiDAR points...")
    if len(frame_with_images["point_cloud"]) > 0:
        plt.figure(figsize=(16, 16))
        ax = plt.subplot(111, projection="3d")
        visualize_lidar_points(frame_with_images, ax)
        plt.show()
    else:
        print("No LiDAR points available in this frame")

    # 3. Visualize Bird's Eye View
    print("Visualizing bird's eye view...")
    if len(frame_with_images["point_cloud"]) > 0:
        visualize_bird_eye_view(frame_with_images)

    # 4. Visualize LiDAR points projected on camera image
    print("Visualizing LiDAR points on camera image...")
    if len(frame_with_images["point_cloud"]) > 0 and len(frame_with_images["images"]) > 0:
        visualize_lidar_on_image(frame_with_images, camera_index=0)
    
    # 5. Visualize LiDAR points and 3D boxes projected on camera image
    print("Visualizing LiDAR points and 3D boxes on camera image...")
    if len(frame_with_images["point_cloud"]) > 0 and len(frame_with_images["images"]) > 0:
        visualize_lidar_on_image_with_boxes(frame_with_images, camera_index=0)

    # 6. Combined Visualization
    print("Creating combined visualization...")
    create_combined_visualization(frame_with_images)

    print("Dataset loaded successfully and ready for PyTorch training!")

    # Create PyTorch DataLoader for batch processing
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x,  # Use custom collate function to handle variable-sized data
    )

    # Example of processing a batch
    for batch in dataloader:
        inputs, targets = prepare_batch_for_model(batch)
        print("Batch processed for model training")
        break  # Just process one batch for demonstration

    print("Dataset loaded successfully and ready for PyTorch training!")


if __name__ == "__main__":
    main()
