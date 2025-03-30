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

        # Find all parquet files in the directory
        if os.path.isdir(data_path):
            for root, _, files in os.walk(data_path):
                for file in files:
                    if file.endswith(".parquet"):
                        self.parquet_files.append(os.path.join(root, file))
        elif os.path.isfile(data_path) and data_path.endswith(".parquet"):
            self.parquet_files = [data_path]

        self.parquet_files.sort()  # Sort files for consistent loading

        # Load metadata
        self.metadata = []
        self.frame_indices = []
        self._load_metadata(max_frames)

    def _load_metadata(self, max_frames):
        """Load metadata from parquet files"""
        total_frames = 0

        for file_idx, path in enumerate(self.parquet_files):
            # Load parquet file
            parquet_file = pq.ParquetFile(path)
            num_rows = parquet_file.metadata.num_rows

            # Add frame indices
            for row_idx in range(num_rows):
                self.frame_indices.append((file_idx, row_idx))
                total_frames += 1

                if max_frames is not None and total_frames >= max_frames:
                    return

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        """
        Get a frame by index

        Returns:
            Dict containing images, point cloud, labels, and metadata
        """
        file_idx, row_idx = self.frame_indices[idx]
        parquet_path = self.parquet_files[file_idx]

        # Read specific row from parquet file
        table = pq.read_table(parquet_path, filters=[("frame_idx", "=", row_idx)])
        frame_data = table.to_pydict()

        # Process camera images
        images = []
        image_names = []

        # Extract camera images from the parquet data
        # In v2, these might have names like 'camera_<NAME>_image' or similar
        for key in frame_data:
            if "camera" in key and "image" in key and "name" not in key:
                camera_image = frame_data[key][0]  # Get the image data
                camera_name_key = key.replace("image", "name")
                camera_name = (
                    frame_data[camera_name_key][0]
                    if camera_name_key in frame_data
                    else key
                )

                # Decode JPEG image
                if isinstance(camera_image, bytes):
                    img = cv2.imdecode(
                        np.frombuffer(camera_image, np.uint8), cv2.IMREAD_COLOR
                    )
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                else:
                    # Handle other formats or missing images
                    img = np.zeros((100, 100, 3), dtype=np.uint8)  # Default placeholder

                # Convert to PyTorch tensor and normalize
                img_tensor = torch.from_numpy(img).float() / 255.0
                # HWC to CHW format
                img_tensor = img_tensor.permute(2, 0, 1)

                images.append(img_tensor)
                image_names.append(camera_name)

        # Process LiDAR point cloud
        point_cloud = self._extract_lidar_points(frame_data)

        # Process labels (bounding boxes)
        labels = self._extract_labels(frame_data)

        return {
            "images": images,
            "image_names": image_names,
            "point_cloud": point_cloud,
            "labels": labels,
            "metadata": {
                key: frame_data[key][0]
                for key in frame_data
                if len(frame_data[key]) > 0
            },
            "frame_index": (file_idx, row_idx),
        }

    def _extract_lidar_points(self, frame_data):
        """Extract LiDAR points from frame data"""
        # Initialize an empty array for points
        points_all = np.zeros((0, 6))  # x, y, z, intensity, elongation, range

        # Look for point cloud data in the frame
        # In v2 format, this might be 'lidar_points' or similar
        lidar_keys = [
            key
            for key in frame_data
            if "lidar" in key and ("points" in key or "xyz" in key)
        ]

        for key in lidar_keys:
            if len(frame_data[key]) > 0:
                points = np.array(frame_data[key][0])

                # Check if points is not empty
                if points.size > 0:
                    # Add additional information if available
                    intensity_key = key.replace("points", "intensity").replace(
                        "xyz", "intensity"
                    )
                    elongation_key = key.replace("points", "elongation").replace(
                        "xyz", "elongation"
                    )
                    range_key = key.replace("points", "range").replace("xyz", "range")

                    # Create feature array
                    features = np.zeros((points.shape[0], 6))
                    features[:, :3] = points[:, :3]  # x, y, z

                    # Add additional features if available
                    if (
                        intensity_key in frame_data
                        and len(frame_data[intensity_key]) > 0
                    ):
                        features[:, 3] = np.array(frame_data[intensity_key][0])
                    if (
                        elongation_key in frame_data
                        and len(frame_data[elongation_key]) > 0
                    ):
                        features[:, 4] = np.array(frame_data[elongation_key][0])
                    if range_key in frame_data and len(frame_data[range_key]) > 0:
                        features[:, 5] = np.array(frame_data[range_key][0])

                    points_all = np.vstack((points_all, features))

        # Convert to PyTorch tensor
        points_tensor = torch.from_numpy(points_all).float()

        return points_tensor

    def _extract_labels(self, frame_data):
        """Extract bounding box labels from frame data"""
        labels = []

        # Look for label data in the frame
        # In v2 format, this might be under keys like 'labels_type', 'labels_id', etc.
        type_key = next(
            (key for key in frame_data if "label" in key and "type" in key), None
        )
        box_key_prefix = next(
            (
                key.split("_")[0]
                for key in frame_data
                if "label" in key and "box" in key
            ),
            "labels",
        )

        if type_key and len(frame_data[type_key]) > 0:
            types = np.array(frame_data[type_key][0])

            # Get the corresponding box data
            centers_x = np.array(
                frame_data.get(
                    f"{box_key_prefix}_box_center_x", [np.zeros_like(types)]
                )[0]
            )
            centers_y = np.array(
                frame_data.get(
                    f"{box_key_prefix}_box_center_y", [np.zeros_like(types)]
                )[0]
            )
            centers_z = np.array(
                frame_data.get(
                    f"{box_key_prefix}_box_center_z", [np.zeros_like(types)]
                )[0]
            )
            lengths = np.array(
                frame_data.get(f"{box_key_prefix}_box_length", [np.zeros_like(types)])[
                    0
                ]
            )
            widths = np.array(
                frame_data.get(f"{box_key_prefix}_box_width", [np.zeros_like(types)])[0]
            )
            heights = np.array(
                frame_data.get(f"{box_key_prefix}_box_height", [np.zeros_like(types)])[
                    0
                ]
            )
            headings = np.array(
                frame_data.get(f"{box_key_prefix}_box_heading", [np.zeros_like(types)])[
                    0
                ]
            )

            # Create label dictionaries
            for i in range(len(types)):
                labels.append(
                    {
                        "center": [centers_x[i], centers_y[i], centers_z[i]],
                        "size": [lengths[i], widths[i], heights[i]],
                        "heading": headings[i],
                        "type": types[i],
                        "name": self._get_label_name(types[i]),
                    }
                )

        return labels

    def _get_label_name(self, type_id):
        """Convert label type ID to name"""
        # This mapping should match the Waymo dataset label types
        label_map = {
            0: "Unknown",
            1: "Vehicle",
            2: "Pedestrian",
            3: "Cyclist",
            4: "Sign",
        }
        return label_map.get(type_id, "Unknown")


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



import numpy as np
import warnings
import dask.dataframe as dd #python -m pip install "dask[complete]"    # Install everything
warnings.simplefilter(action="ignore", category=FutureWarning)

# Example dataset directory and context name
dataset_dir = "/mnt/e/Dataset/waymodata/training"
context_name = "11076364019363412893_1711_000_1731_000"

#https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_v2.ipynb
#You can use any existing library which supports the Apache Parquet files to read the dataset, for example PyArrow, Pandas, Dask or any other. We recommend to use Dask to access entire dataset, because it supports larger-then-memory tables and distributed processing.
def read(tag: str) -> np.ndarray:
    """
    Creates a NumPy array for the component specified by its tag.
    Reads the data from files and converts it into a NumPy array.
    """
    paths = f"{dataset_dir}/{tag}/{context_name}.parquet"
    # Load parquet file
    #parquet_file = pq.ParquetFile(paths)
    #num_rows = parquet_file.metadata.num_rows
    return dd.read_parquet(paths)


# Lazily read camera images and boxes 
cam_image_df = read('camera_image')
cam_image_df.head()
# Filter the images from camera=1
# NOTE: We could also use push down filters while reading the parquet files as well
# Details https://docs.dask.org/en/stable/generated/dask.dataframe.read_parquet.html#dask.dataframe.read_parquet
camera_image_df = cam_image_df[cam_image_df['key.camera_name'] == 1]

cam_box_df = read('camera_box')
cam_box_df.head()
# Inner join the camera_image table with the camera_box table.
df = camera_image_df.merge(
    cam_box_df,
    on=[
        'key.segment_context_name',
        'key.frame_timestamp_micros',
        'key.camera_name',
    ],
    how='inner',
)
df.head()
# Create corresponding components from the raw
_, row = next(iter(df.iterrows()))

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

    # Show sample visualizations for the first frame
    if len(dataset) > 0:
        frame_data = dataset[0]

        # 1. Visualize Camera Image
        print("Visualizing camera image...")
        if len(frame_data["images"]) > 0:
            visualize_camera_image(frame_data, camera_index=0)
        else:
            print("No camera images available in this frame")

        # 2. Visualize LiDAR Points
        print("Visualizing LiDAR points...")
        if len(frame_data["point_cloud"]) > 0:
            plt.figure(figsize=(16, 16))
            ax = plt.subplot(111, projection="3d")
            visualize_lidar_points(frame_data, ax)
            plt.show()
        else:
            print("No LiDAR points available in this frame")

        # 3. Visualize Bird's Eye View
        print("Visualizing bird's eye view...")
        if len(frame_data["point_cloud"]) > 0:
            visualize_bird_eye_view(frame_data)

        # 4. Combined Visualization
        print("Creating combined visualization...")
        create_combined_visualization(frame_data)

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
