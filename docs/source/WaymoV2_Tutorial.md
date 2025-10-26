# 🧭 Comprehensive Tutorial: Understanding and Visualizing the **Waymo Open Dataset v2**

This tutorial provides an **in-depth, practical, and mathematical** explanation of how to interpret, transform, and visualize **Waymo v2.1 LiDAR and 3D box data**.  
It's written for researchers and developers who want to deeply understand how Waymo structures its data, what coordinate frames are used, and how to correctly align LiDAR, camera, and annotations.

> **Code Reference**: This tutorial is complemented by the comprehensive data inspection utilities in [`waymo_parquet_inspector.py`](../../DeepDataMiningLearning/detection3d/waymo_parquet_inspector.py), which provides detailed field analysis and validation tools for all Waymo data components.

---

## Table of Contents
1. [Dataset Overview](#1-dataset-overview)
2. [File Structure & Contents](#2-file-structure--contents)
3. [Coordinate Frames](#3-coordinate-frames)
4. [Camera Data Components](#4-camera-data-components)
5. [LiDAR Data Components](#5-lidar-data-components)
6. [Calibration Data](#6-calibration-data)
7. [LiDAR-to-Vehicle Transform Mathematics](#7-lidar-to-vehicle-transform-mathematics)
8. [3D Box Definitions](#8-3d-box-definitions)
9. [Coordinate Alignment: LiDAR ↔ Box](#9-coordinate-alignment-lidar--box)
10. [2D and 3D Box Visualization](#10-2d-and-3d-box-visualization)
11. [Common Pitfalls & Debug Tips](#11-common-pitfalls--debug-tips)
12. [References](#12-references)

---

## 1️⃣ Dataset Overview

### 📊 Dataset Scale & Format
Waymo Open Dataset (WOD) provides synchronized **LiDAR** and **camera** data with ground-truth **3D bounding boxes**.

- **2,030 segments** of 20s each, collected at 10Hz (**390,000 frames**)
- Diverse geographies and conditions
- **Perception object assets** data in a modular format (v2.0.0)
- Extracted perception objects from multi-sensor data: all 5 cameras and the top lidar

### 🔧 Sensor Configuration
**LiDAR Sensors:**
- 1 mid-range lidar
- 4 short-range lidars

**Camera Sensors:**
- 5 cameras (front and sides)

**Data Synchronization:**
- Synchronized lidar and camera data
- Lidar to camera projections
- Sensor calibrations and vehicle poses

### 🏷️ Annotation Categories

#### **3D Bounding Box Labels**
- **4 object classes**: Vehicles, Pedestrians, Cyclists, Signs
- **High-quality labels for lidar data**: 1,200 segments
- **12.6M 3D bounding box labels** with tracking IDs on lidar data

#### **2D Bounding Box Labels**
- **High-quality labels for camera data**: 1,000 segments  
- **11.8M 2D bounding box labels** with tracking IDs on camera data

#### **2D Video Panoptic Segmentation**
- **Subset**: 100k camera images
- **28 classes** including:
  - **Vehicles**: Car, Bus, Truck, Other Large Vehicle, Trailer, Ego Vehicle, Motorcycle, Bicycle
  - **People**: Pedestrian, Cyclist, Motorcyclist
  - **Animals**: Ground Animal, Bird
  - **Infrastructure**: Pole, Sign, Traffic Light, Construction Cone, Pedestrian Object, Building
  - **Road Elements**: Road, Sidewalk, Road Marker, Lane Marker
  - **Environment**: Vegetation, Sky, Ground
  - **Motion States**: Static, Dynamic
- **Instance segmentation** labels for Vehicle, Pedestrian and Cyclist classes
- Consistent both across cameras and over time

#### **Key Point Labels**
- **2 object classes**: Pedestrians and Cyclists
- **14 key points** from nose to ankle
- **200k object frames** with 2D key point labels
- **10k object frames** with 3D key point labels

#### **3D Semantic Segmentation**
- **Segmentation labels**: 1,150 segments
- **23 classes** including:
  - **Vehicles**: Car, Truck, Bus, Other Vehicle
  - **People**: Motorcyclist, Bicyclist, Pedestrian
  - **Objects**: Bicycle, Motorcycle, Sign, Traffic Light, Pole, Construction Cone
  - **Environment**: Building, Vegetation, Tree Trunk, Curb
  - **Road Elements**: Road, Lane Marker, Walkable, Sidewalk, Other Ground
  - **Undefined**: Undefined

### 🗺️ Map Data
- **3D road graph data** for each segment
- Includes: lane centers, lane boundaries, road boundaries, crosswalks, speed bumps, stop signs, and entrances to driveways

### 🔗 Cross-Modal Associations
- **Association of 2D and 3D bounding boxes**
- Corresponding object IDs provided for **2 object classes**: Pedestrians and Cyclists

### 🎯 Challenge Data
- **3D Camera-Only Detection Challenge**: 80 segments of 20s camera imagery

### 🚀 Advanced Features
LiDAR features include:
- **3D point cloud sequences** that support 3D object shape reconstruction

Camera features include:
- Sequences of camera patches from the most_visible_camera
- Projected lidar returns on the corresponding camera
- Per-pixel camera rays information
- Auto-labeled 2D panoptic segmentation that supports **object NeRF reconstruction**

The **v2.0.1 Parquet version** is designed for efficient columnar access and includes:

| Component | Folder | Description | Schema Columns |
|------------|---------|-------------|----------------|
| Camera Images | `camera_image/` | RGB frames with pose/velocity metadata | 15 columns with binary JPEG/PNG data |
| Camera Boxes | `camera_box/` | 2D bounding boxes in pixel coordinates | 12 columns with detection annotations |
| Camera Calibration | `camera_calibration/` | Intrinsics + extrinsics for all cameras | 15 columns with calibration matrices |
| LiDAR Range Images | `lidar/` | Raw per-LiDAR sensor range images | 7 columns with flattened range data |
| 3D Boxes | `lidar_box/` | Ground-truth boxes in **Vehicle Frame** | 21 columns with 3D object state |
| LiDAR Calibration | `lidar_calibration/` | Beam inclinations and extrinsic transforms | 6 columns with sensor parameters |
| LiDAR Poses | `lidar_pose/` | Per-pixel transforms LiDAR → Vehicle → World | Pose transformation matrices |
| Segmentation | `lidar_segmentation/` | Point-wise semantic labels | Semantic segmentation masks |
| Projections | `lidar_camera_projection/` | LiDAR-to-camera pixel mappings | Cross-modal alignment data |

---

## 2️⃣ File Structure & Contents

Each **segment** is approximately 20 seconds long, split into multiple Parquet files with standardized naming:

```
segment_id_start_end.parquet
Example: 10017090168044687777_6380_000_6400_000.parquet
```

Inside each data folder (e.g., `training/lidar/`), files contain rows corresponding to **sensor measurements** at specific **timestamps**.

### Parquet Schema Structure

All Waymo v2.01 data follows a consistent schema pattern:

```
Key Fields (Common across all data types):
├── index: Unique row identifier (String)
├── key.segment_context_name: Segment ID (String)  
├── key.frame_timestamp_micros: Timestamp in microseconds (Int64)
└── key.[sensor]_name: Sensor identifier (Int8)

Component Fields:
└── [ComponentType].[field_hierarchy]: Actual data values
    ├── Scalar values: Direct numeric/string data
    ├── List values: Arrays (e.g., transformation matrices)
    └── Nested structures: Complex hierarchical data
```

> **Note**: The `[ComponentType]` follows the pattern `[SensorType]Component` (e.g., `CameraImageComponent`, `LiDARBoxComponent`). See the [inspector code](../../DeepDataMiningLearning/detection3d/waymo_parquet_inspector.py) for detailed field analysis of each component type.

**LiDAR IDs (v2.1 five-sensor setup):**

| ID | Location | Yaw (deg) | Position (m) |
|----|-----------|-----------|--------------|
| 1 | Roof edge / back-right | +148° | [1.43, 0.0, 2.18] |
| 2 | Front bumper | 0° | [4.07, 0.0, 0.69] |
| 3 | Left side | +90° | [3.25, +1.02, 0.98] |
| 4 | Right side | −90° | [3.25, −1.02, 0.98] |
| 5 | Rear | 180° | [−1.15, 0.0, 0.46] |

> ⚠️ v2.1 no longer includes the 360° **Top LiDAR** used in early WOD versions.

---

## 3️⃣ Coordinate Frames

Understanding coordinate systems is the foundation for correct visualization and data alignment.

### Vehicle Frame (Primary Reference Frame)
- **Origin**: Vehicle center (geometric center of the ego vehicle)
- **Axes**: 
  - **+X**: Forward direction (vehicle's front)
  - **+Y**: Left direction (driver's left)
  - **+Z**: Upward direction (towards sky)
- **Usage**: All 3D bounding boxes and calibration extrinsics are defined in this frame
- **Mathematical Representation**: Right-handed coordinate system

$$\mathbf{p}_{\text{vehicle}} = \begin{bmatrix} x \\ y \\ z \end{bmatrix} \text{ where } \begin{cases} x > 0 & \text{forward} \\ y > 0 & \text{left} \\ z > 0 & \text{up} \end{cases}$$

### LiDAR Frame (Sensor-Specific)
- **Origin**: Individual LiDAR sensor center
- **Axes**: Same orientation as vehicle frame but translated/rotated
  - **+X**: Sensor's forward direction
  - **+Y**: Sensor's left direction  
  - **+Z**: Sensor's upward direction
- **Usage**: Raw range image data is initially in this frame
- **Transform**: Each LiDAR has a unique extrinsic transformation to vehicle frame

### Camera Frame (OpenCV Convention)
- **Origin**: Camera optical center
- **Axes**:
  - **+X**: Right direction (image columns)
  - **+Y**: Down direction (image rows)
  - **+Z**: Forward direction (optical axis, into the scene)
- **Usage**: Camera images and 2D bounding boxes
- **Projection**: 3D points project to 2D image coordinates via intrinsic matrix

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} X_c/Z_c \\ Y_c/Z_c \\ 1 \end{bmatrix}$$

where $\mathbf{K}$ is the camera intrinsic matrix and $(X_c, Y_c, Z_c)$ are coordinates in camera frame.

---

## 4️⃣ Camera Data Components

### Camera Images (`camera_image/`)

**Schema**: 15 columns containing RGB image data and comprehensive metadata

| Field | Type | Description |
|-------|------|-------------|
| `index` | String | Unique row identifier |
| `key.segment_context_name` | String | Segment/sequence identifier |
| `key.frame_timestamp_micros` | Int64 | Frame timestamp in microseconds |
| `key.camera_name` | Int8 | Camera ID (0-4 for FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT) |
| `[CameraImageComponent].image` | Binary | JPEG/PNG compressed image bytes |
| `[CameraImageComponent].pose.transform` | List[Double] | 4×4 transformation matrix (16 elements) |
| `[CameraImageComponent].velocity.linear_velocity.{x,y,z}` | Double | Linear velocity components (m/s) |
| `[CameraImageComponent].velocity.angular_velocity.{x,y,z}` | Double | Angular velocity components (rad/s) |
| `[CameraImageComponent].pose_timestamp` | Double | Pose measurement timestamp |
| `[CameraImageComponent].rolling_shutter_params.shutter` | Double | Rolling shutter timing parameter |

**Usage Example**:
```python
# Extract image from binary data
image_bytes = row['[CameraImageComponent].image']
pil_image = Image.open(io.BytesIO(image_bytes))

# Extract pose matrix (4x4 transformation)
pose_flat = row['[CameraImageComponent].pose.transform']  # 16 elements
pose_matrix = np.array(pose_flat).reshape(4, 4)
```

### Camera Boxes (`camera_box/`)

**Schema**: 12 columns containing 2D bounding box annotations

| Field | Type | Description |
|-------|------|-------------|
| `key.camera_object_id` | String | Unique object identifier per camera |
| `[CameraBoxComponent].box.center.{x,y}` | Double | Bounding box center coordinates (pixels) |
| `[CameraBoxComponent].box.size.{x,y}` | Double | Bounding box dimensions (width, height in pixels) |
| `[CameraBoxComponent].type` | Int8 | Object class type ID |
| `[CameraBoxComponent].difficulty_level.detection` | Int8 | Detection difficulty rating (1-5) |
| `[CameraBoxComponent].difficulty_level.tracking` | Int8 | Tracking difficulty rating (1-5) |

### Camera Calibration (`camera_calibration/`)

**Schema**: 15 columns containing intrinsic and extrinsic calibration parameters

| Field | Type | Description |
|-------|------|-------------|
| `[CameraCalibrationComponent].intrinsic.f_u` | Double | Focal length in u direction (pixels) |
| `[CameraCalibrationComponent].intrinsic.f_v` | Double | Focal length in v direction (pixels) |
| `[CameraCalibrationComponent].intrinsic.c_u` | Double | Principal point u coordinate (pixels) |
| `[CameraCalibrationComponent].intrinsic.c_v` | Double | Principal point v coordinate (pixels) |
| `[CameraCalibrationComponent].intrinsic.k1,k2,k3` | Double | Radial distortion coefficients |
| `[CameraCalibrationComponent].intrinsic.p1,p2` | Double | Tangential distortion coefficients |
| `[CameraCalibrationComponent].extrinsic.transform` | List[Double] | 4×4 camera-to-vehicle transformation |
| `[CameraCalibrationComponent].width` | Int32 | Image width in pixels |
| `[CameraCalibrationComponent].height` | Int32 | Image height in pixels |

**Intrinsic Matrix Construction**:
$$\mathbf{K} = \begin{bmatrix} f_u & 0 & c_u \\ 0 & f_v & c_v \\ 0 & 0 & 1 \end{bmatrix}$$

---

## 5️⃣ LiDAR Data Components

### LiDAR Range Images (`lidar/`)

**Schema**: 11 columns containing range image data and sensor metadata

Each LiDAR captures a **range image** instead of a raw point cloud. This is a 2D representation where each pixel encodes distance, intensity, and other measurements.

| Field | Type | Description |
|-------|------|-------------|
| `index` | String | Unique row identifier |
| `key.segment_context_name` | String | Segment/sequence identifier |
| `key.frame_timestamp_micros` | Int64 | Frame timestamp in microseconds |
| `key.laser_name` | Int8 | LiDAR sensor ID (0-4 for TOP, FRONT, SIDE_LEFT, SIDE_RIGHT, REAR) |
| `[LiDARComponent].range_image_return1.range` | Binary | First return range data (compressed) |
| `[LiDARComponent].range_image_return1.intensity` | Binary | First return intensity data (compressed) |
| `[LiDARComponent].range_image_return1.elongation` | Binary | First return elongation data (compressed) |
| `[LiDARComponent].range_image_return2.range` | Binary | Second return range data (compressed) |
| `[LiDARComponent].range_image_return2.intensity` | Binary | Second return intensity data (compressed) |
| `[LiDARComponent].range_image_return2.elongation` | Binary | Second return elongation data (compressed) |
| `[LiDARComponent].camera_projection_exclusion_mask` | Binary | Exclusion mask for camera projections |

**Range Image Structure**:
- **Dimensions**: Typically H×W where H varies by sensor (64-200 rows), W is azimuth resolution
- **Encoding**: Each pixel encodes distance measurement in meters
- **Returns**: Two returns per laser beam (first and second reflection)
- **Compression**: Data is compressed using Waymo's proprietary format

---

## 6️⃣ LiDAR-to-Vehicle Transform Mathematics

Converting LiDAR range images to 3D point clouds in the vehicle coordinate frame requires several mathematical transformations.

### Step 1: Range Image to Spherical Coordinates

Each pixel $(u, v)$ in the range image corresponds to spherical coordinates:

$$\begin{align}
\text{azimuth} &= \frac{2\pi \cdot u}{W} - \pi \\
\text{inclination} &= \text{beam\_inclinations}[v] \\
\text{range} &= \text{range\_image}[v, u]
\end{align}$$

where $W$ is the azimuth resolution (typically 2650 for Waymo LiDAR).

### Step 2: Spherical to Cartesian Conversion (LiDAR Frame)

Convert spherical coordinates to 3D Cartesian coordinates in the LiDAR sensor frame:

$$\begin{align}
x_{\text{lidar}} &= \text{range} \cdot \cos(\text{inclination}) \cdot \cos(\text{azimuth}) \\
y_{\text{lidar}} &= \text{range} \cdot \cos(\text{inclination}) \cdot \sin(\text{azimuth}) \\
z_{\text{lidar}} &= \text{range} \cdot \sin(\text{inclination})
\end{align}$$

### Step 3: LiDAR-to-Vehicle Transformation

Apply the extrinsic calibration matrix to transform from LiDAR frame to vehicle frame:

$$\begin{bmatrix} x_{\text{vehicle}} \\ y_{\text{vehicle}} \\ z_{\text{vehicle}} \\ 1 \end{bmatrix} = \mathbf{T}_{\text{lidar→vehicle}} \begin{bmatrix} x_{\text{lidar}} \\ y_{\text{lidar}} \\ z_{\text{lidar}} \\ 1 \end{bmatrix}$$

where $\mathbf{T}_{\text{lidar→vehicle}}$ is the 4×4 transformation matrix from the calibration data:

$$\mathbf{T}_{\text{lidar→vehicle}} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

with $\mathbf{R} \in \mathbb{R}^{3 \times 3}$ being the rotation matrix and $\mathbf{t} \in \mathbb{R}^{3}$ the translation vector.

### Complete Transformation Pipeline

The complete transformation from range image pixel to vehicle coordinates:

$$\mathbf{p}_{\text{vehicle}} = \mathbf{T}_{\text{lidar→vehicle}} \cdot \mathbf{f}_{\text{spherical→cartesian}}(\text{range}[v,u], \text{azimuth}(u), \text{inclination}(v))$$

**Implementation Note**: The waymo_parquet_inspector.py script provides detailed field analysis for understanding the exact data formats and transformations.

### LiDAR Boxes (`lidar_box/`)

**Schema**: 18 columns containing 3D bounding box annotations

| Field | Type | Description |
|-------|------|-------------|
| `key.laser_object_id` | String | Unique object identifier per LiDAR |
| `[LiDARBoxComponent].box.center.{x,y,z}` | Double | 3D bounding box center in vehicle frame (meters) |
| `[LiDARBoxComponent].box.size.{x,y,z}` | Double | 3D bounding box dimensions (length, width, height in meters) |
| `[LiDARBoxComponent].box.heading` | Double | Object orientation angle (radians) |
| `[LiDARBoxComponent].type` | Int8 | Object class type ID |
| `[LiDARBoxComponent].id` | String | Persistent object tracking ID |
| `[LiDARBoxComponent].detection_difficulty_level` | Int8 | Detection difficulty rating (1-5) |
| `[LiDARBoxComponent].tracking_difficulty_level` | Int8 | Tracking difficulty rating (1-5) |
| `[LiDARBoxComponent].num_lidar_points_in_box` | Int32 | Number of LiDAR points inside the box |

**3D Box Representation**:
$$\mathbf{Box} = \{\mathbf{c}, \mathbf{s}, \theta\} \text{ where } \begin{cases} \mathbf{c} = [c_x, c_y, c_z]^T & \text{center position} \\ \mathbf{s} = [s_x, s_y, s_z]^T & \text{size (L×W×H)} \\ \theta & \text{heading angle} \end{cases}$$

### LiDAR Calibration (`lidar_calibration/`)

**Schema**: 8 columns containing sensor calibration parameters

| Field | Type | Description |
|-------|------|-------------|
| `[LiDARCalibrationComponent].extrinsic.transform` | List[Double] | 4×4 LiDAR-to-vehicle transformation matrix |
| `[LiDARCalibrationComponent].beam_inclinations` | List[Double] | Vertical beam angle inclinations (radians) |
| `[LiDARCalibrationComponent].beam_inclination_min` | Double | Minimum beam inclination angle |
| `[LiDARCalibrationComponent].beam_inclination_max` | Double | Maximum beam inclination angle |

---

## 7️⃣ Additional Data Components

### Projected LiDAR (`projected_lidar_labels/`)

**Schema**: 14 columns containing LiDAR points projected onto camera images

| Field | Type | Description |
|-------|------|-------------|
| `[ProjectedLiDARLabelsComponent].box.center.{x,y}` | Double | Projected 2D box center (pixels) |
| `[ProjectedLiDARLabelsComponent].box.size.{x,y}` | Double | Projected 2D box size (pixels) |
| `[ProjectedLiDARLabelsComponent].type` | Int8 | Object class type ID |
| `[ProjectedLiDARLabelsComponent].id` | String | Object tracking ID |
| `[ProjectedLiDARLabelsComponent].detection_difficulty_level` | Int8 | Detection difficulty (1-5) |

### Segmentation Labels (`lidar_segmentation/`)

**Schema**: 7 columns containing point-wise semantic segmentation

| Field | Type | Description |
|-------|------|-------------|
| `[LiDARSegmentationLabelComponent].pointcloud_to_image_projection` | Binary | Point-to-pixel mapping data |
| `[LiDARSegmentationLabelComponent].segmentation_label` | Binary | Per-point semantic labels |
| `[LiDARSegmentationLabelComponent].instance_id_to_global_id_mapping` | Binary | Instance ID mappings |

### Statistics (`stats/`)

**Schema**: 9 columns containing frame-level statistics and metadata

| Field | Type | Description |
|-------|------|-------------|
| `[StatsComponent].location` | String | Geographic location identifier |
| `[StatsComponent].time_of_day` | String | Time period (Dawn, Day, Dusk, Night) |
| `[StatsComponent].weather` | String | Weather conditions |
| `[StatsComponent].camera_object_counts` | List[Int32] | Object counts per camera |
| `[StatsComponent].lidar_object_counts` | List[Int32] | Object counts per LiDAR |

**Usage for Data Analysis**:
```python
# Filter by weather conditions
sunny_frames = df[df['[StatsComponent].weather'] == 'sunny']

# Analyze object distribution
total_objects = df['[StatsComponent].lidar_object_counts'].apply(sum)
---

## 8️⃣ Range Image Processing

Each pixel encodes `(range, intensity, elongation, ...)` data in compressed binary format.

**Typical Range Image Dimensions**:

| Sensor | Shape (H, W, C) | Field of View |
|---------|-----------------|---------------|
| TOP LiDAR (#0) | 64 × 2650 × 4 | 360° horizontal |
| FRONT LiDAR (#1) | 200 × 600 × 4 | ~100° horizontal |
| SIDE_LEFT (#2) | 200 × 600 × 4 | ~100° horizontal |
| SIDE_RIGHT (#3) | 200 × 600 × 4 | ~100° horizontal |
| REAR (#4) | 200 × 600 × 4 | ~100° horizontal |

### Range Image to Point Cloud Conversion

**Step 1**: Decode the compressed range/intensity data from binary format
**Step 2**: Apply coordinate transformations to get 3D points in vehicle frame
**Step 3**: Filter invalid points (range = 0)

---

## 9️⃣ LiDAR Calibration Mathematics

### Extrinsic Transformation Matrix

The extrinsic matrix $\mathbf{T}_{V \leftarrow L}$ transforms points from LiDAR frame to vehicle frame:

$$\mathbf{T}_{V \leftarrow L} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

**Storage Format**: Row-major order (`order="C"`) in Parquet, stored as 16-element list

**Example Transformation Matrix**:
```
[[-0.8478, -0.5304, -0.0025,  1.43 ],
 [ 0.5304, -0.8478,  0.0002,  0.00 ],
 [-0.0022, -0.0012,  1.0000,  2.184],
 [ 0.0000,  0.0000,  0.0000,  1.0000]]
```
→ Rotation yaw ≈ 148°, translation ≈ (1.43, 0.0, 2.18) meters

### Beam Inclination Angles

Vertical angles for each row in the range image, typically distributed linearly between minimum and maximum inclination values.

$$\theta_v = \text{beam\_inclinations}[v] \text{ for row } v \in [0, H-1]$$

---

## 🔟 Complete LiDAR Processing Pipeline

### Mathematical Transformation Steps

For each LiDAR pixel at position $(u, v)$ with range $r$:

**Step 1: Spherical Coordinates**
$$\begin{align}
\phi &= \frac{2\pi \cdot u}{W} - \pi \quad \text{(azimuth angle)} \\
\theta &= \text{beam\_inclinations}[v] \quad \text{(inclination angle)} \\
r &= \text{range\_image}[v, u] \quad \text{(distance in meters)}
\end{align}$$

**Step 2: LiDAR Frame Cartesian Coordinates**
$$\begin{align}
x_L &= r \cos(\theta) \cos(\phi) \\
y_L &= r \cos(\theta) \sin(\phi) \\
z_L &= r \sin(\theta)
\end{align}$$

**Step 3: Homogeneous Coordinates**
$$\mathbf{p}_L = \begin{bmatrix} x_L \\ y_L \\ z_L \\ 1 \end{bmatrix}$$

**Step 4: Transform to Vehicle Frame**
$$\mathbf{p}_V = \mathbf{T}_{V \leftarrow L} \cdot \mathbf{p}_L$$

### Implementation in NumPy

```python
# Create homogeneous coordinate matrix
pts_h = np.stack([x_L, y_L, z_L, np.ones_like(z_L)], axis=-1).reshape(-1, 4)

# Transform to vehicle frame (do NOT invert the matrix)
xyz_vehicle = (pts_h @ extrinsic_matrix.T)[:, :3]
```

**Important**: The dataset stores LiDAR→Vehicle transforms directly. Do not invert the matrix.

---

## 1️⃣1️⃣ 3D Bounding Box Specifications

### Box Parameters in Vehicle Frame

Each 3D bounding box in `lidar_box/` is defined by:

| Parameter | Field | Description |
|-----------|-------|-------------|
| **Center** | `[LiDARBoxComponent].box.center.{x,y,z}` | Box center position (meters) |
| **Size** | `[LiDARBoxComponent].box.size.{x,y,z}` | Length (X), Width (Y), Height (Z) |
| **Heading** | `[LiDARBoxComponent].box.heading` | Yaw angle (radians, CCW from +X axis) |
| **Type** | `[LiDARBoxComponent].type` | Object class (vehicle, pedestrian, cyclist) |

**Important Note**: Box center Z-coordinate represents the object's geometric center, not the bottom.

### 3D Box Mathematical Representation

$$\mathbf{Box} = \{\mathbf{c}, \mathbf{s}, \psi\}$$

where:
- $\mathbf{c} = [c_x, c_y, c_z]^T$ is the center position
- $\mathbf{s} = [s_x, s_y, s_z]^T$ is the size vector (length × width × height)  
- $\psi$ is the heading angle (yaw rotation about Z-axis)

---

## 1️⃣2️⃣ Multi-Sensor Data Fusion

### Coordinate Alignment Process

**Step 1**: Decode range images from all 5 LiDAR sensors
**Step 2**: Transform each sensor's points to vehicle frame using respective extrinsics
**Step 3**: Merge all point clouds into unified coordinate system

```python
# Process each LiDAR sensor
all_points = []
for sensor_id in range(5):  # 0=TOP, 1=FRONT, 2=SIDE_LEFT, 3=SIDE_RIGHT, 4=REAR
    # Extract sensor-specific data
    range_data = decode_range_image(sensor_data[sensor_id])
    extrinsic = get_extrinsic_matrix(sensor_id)
    
    # Transform to vehicle frame
    points_vehicle = transform_to_vehicle_frame(range_data, extrinsic)
    all_points.append(points_vehicle)

# Merge all sensors
merged_pointcloud = np.concatenate(all_points, axis=0)
```

**Result**: Both point cloud and 3D boxes are now in the same vehicle coordinate frame and align perfectly.

---

## 1️⃣3️⃣ Visualization and Projection

### 3D Visualization with Open3D

```python
import open3d as o3d
import numpy as np

# Create point cloud visualization
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_vehicle))
pcd.paint_uniform_color([0.6, 0.6, 0.6])
geometries = [pcd]

# Add 3D bounding boxes
for box_data in boxes_3d:
    x, y, z = box_data['center']
    dx, dy, dz = box_data['size'] 
    yaw = box_data['heading']
    
    # Create rotation matrix for yaw
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    
    # Create oriented bounding box
    obb = o3d.geometry.OrientedBoundingBox(
        center=[x, y, z], 
        R=R, 
        extent=[dx, dy, dz]
    )
    obb.color = (1, 0, 0)  # Red color
    geometries.append(obb)

# Add coordinate frame
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
geometries.append(axis)

# Visualize
o3d.visualization.draw_geometries(geometries)
```

### 2D Projection onto Camera Images

**Mathematical Projection Pipeline**:

1. **Transform 3D points to camera frame**:
   $$\mathbf{p}_C = \mathbf{T}_{C \leftarrow V} \cdot \mathbf{p}_V$$

2. **Project to image plane**:
   $$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} X_C/Z_C \\ Y_C/Z_C \\ 1 \end{bmatrix}$$

3. **Apply distortion correction** (if needed):
   $$\begin{align}
   r^2 &= u_n^2 + v_n^2 \\
   u_d &= u_n(1 + k_1r^2 + k_2r^4 + k_3r^6) + 2p_1u_nv_n + p_2(r^2 + 2u_n^2) \\
   v_d &= v_n(1 + k_1r^2 + k_2r^4 + k_3r^6) + p_1(r^2 + 2v_n^2) + 2p_2u_nv_n
   \end{align}$$

```python
def project_3d_to_2d(points_3d, camera_intrinsic, camera_extrinsic):
    """Project 3D points to camera image coordinates"""
    # Transform to camera frame
    vehicle_to_camera = np.linalg.inv(camera_extrinsic)
    points_homogeneous = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    points_camera = (vehicle_to_camera @ points_homogeneous.T).T[:, :3]
    
    # Project to image plane
    points_2d_homogeneous = (camera_intrinsic @ points_camera.T).T
    image_points = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]
    depths = points_camera[:, 2]
    
    return image_points, depths
```

---

## 1️⃣4️⃣ Common Issues and Solutions

### Troubleshooting Guide

| **Problem** | **Cause** | **Solution** |
|-------------|-----------|--------------|
| Box appears "floating" above ground | LiDAR mounted ~2m high, box Z is object center | This is normal behavior |
| Box appears "in front of" points | Using single LiDAR sensor only | Merge all 5 LiDAR sensors |
| Point cloud mirrored/flipped | Used `np.linalg.inv(extrinsic)` | Use `extrinsic.T` for matrix multiplication |
| Translation values all zeros | Used `order='F'` for reshape | Use `order='C'` (row-major) |
| Beam angles incorrect | Reused wrong beam inclinations | Read sensor-specific beam ranges |
| Point cloud appears "warped" | Mixed sensors with wrong extrinsics | Verify yaw angle per LiDAR sensor |

### Best Practices

✅ **Recommended Settings**:
- Use `order="C"` for array reshaping
- Apply `extrinsic.T` for transformations (do not invert)
- Set `flip_rows=True, flip_cols=False` for range image processing
- Use `azimuth = np.linspace(np.pi, -np.pi, W)` for azimuth calculation

✅ **Validation Checks**:
- Verify point cloud and boxes align in 3D visualization
- Check that merged multi-LiDAR coverage is 360°
- Ensure camera projections fall within image boundaries
- Validate coordinate frame orientations match expected directions

---

## 1️⃣5️⃣ References and Resources

### Official Documentation
- **Waymo Open Dataset Repository**: [https://github.com/waymo-research/waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset)
- **Range Image Utilities**: `range_image_utils.py` in official repo
- **Coordinate Conventions**: Waymo Open Dataset Paper, CVPR 2020

### Community Tools and Converters
- **OpenCOOD**: Multi-modal 3D detection framework with Waymo support
- **OpenMMLab**: MMDetection3D parser examples
- **Waymo2KITTI**: Format conversion utilities (GitHub community)

### Data Analysis Tools
- **Field Inspector**: <mcfile name="waymo_parquet_inspector.py" path="/home/lkk/Developer/DeepDataMiningLearning/DeepDataMiningLearning/detection3d/waymo_parquet_inspector.py"></mcfile> - Comprehensive schema analysis
- **Visualization Scripts**: Open3D and Matplotlib integration examples

---

## ✨ Summary

### Complete Processing Pipeline

| **Step** | **Action** | **Coordinate Frame** |
|----------|------------|---------------------|
| 1 | Decode range image | LiDAR frame |
| 2 | Apply extrinsic transform | Vehicle frame |
| 3 | Merge all sensors | Vehicle frame |
| 4 | Visualize with boxes | Vehicle frame |
| 5 | Project to cameras | Camera/Image frame |

### Key Mathematical Transformations

$$\text{Range Image} \xrightarrow{\text{spherical→cartesian}} \text{LiDAR Frame} \xrightarrow{\mathbf{T}_{V \leftarrow L}} \text{Vehicle Frame} \xrightarrow{\mathbf{T}_{C \leftarrow V}} \text{Camera Frame}$$

When implemented correctly, the merged multi-LiDAR point cloud aligns perfectly with Waymo's 3D bounding boxes and camera images, enabling robust multi-modal perception and analysis.

