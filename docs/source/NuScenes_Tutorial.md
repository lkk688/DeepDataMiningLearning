# NuScenes Dataset Tutorial: Coordinate Transformations and Bounding Box Processing

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Structure](#dataset-structure)
3. [Coordinate Systems](#coordinate-systems)
4. [Coordinate Transformations](#coordinate-transformations)
5. [3D Bounding Box Processing](#3d-bounding-box-processing)
6. [2D Projection Pipeline](#2d-projection-pipeline)
7. [Code Implementation](#code-implementation)
8. [Common Issues and Solutions](#common-issues-and-solutions)
9. [Best Practices](#best-practices)

## Introduction

The NuScenes dataset is a large-scale autonomous driving dataset that provides multimodal sensor data including cameras, LiDAR, and radar. One of the most challenging aspects of working with NuScenes is understanding and correctly implementing the coordinate system transformations required to project 3D annotations onto 2D camera images.

This tutorial focuses on the critical processes of:
- **Coordinate System Transformations**: Converting between global, ego vehicle, and camera coordinate systems
- **3D Bounding Box Processing**: Generating and manipulating 3D bounding boxes
- **2D Projection**: Projecting 3D bounding boxes onto camera images
- **Ensuring Correctness**: Validation techniques and common pitfalls

## Dataset Structure

NuScenes organizes data hierarchically:

```
nuscenes/
├── samples/           # Keyframes (2Hz) with all sensor data
├── sweeps/           # Intermediate frames (20Hz) for LiDAR/radar
├── maps/             # HD maps for each location
└── v1.0-trainval/    # Annotation files (JSON)
    ├── sample.json
    ├── sample_data.json
    ├── sample_annotation.json
    ├── ego_pose.json
    ├── calibrated_sensor.json
    └── ...
```

## Dataset Annotation Format

NuScenes provides comprehensive annotations in JSON format, with all spatial annotations defined in the **global coordinate system**.

### Annotation Types and Coordinate Systems

#### 1. 3D Bounding Box Annotations
**File**: `sample_annotation.json`  
**Coordinate System**: Global coordinates  
**Format**: Center + Size + Orientation

```json
{
    "token": "unique_annotation_id",
    "sample_token": "sample_id",
    "instance_token": "object_instance_id",
    "category_name": "car",
    "translation": [x, y, z],           # 3D center in global coordinates (meters)
    "size": [width, length, height],    # Bounding box dimensions (meters)
    "rotation": [w, x, y, z],          # Quaternion orientation in global frame
    "visibility": 2,                    # Visibility level (1-4)
    "attribute_tokens": ["moving"],     # Object attributes
    "num_lidar_pts": 150,              # Number of LiDAR points inside box
    "num_radar_pts": 5                 # Number of radar points inside box
}
```

**Key Points**:
- **Translation**: 3D center position `[x, y, z]` in global coordinates
- **Size**: Box dimensions `[width, length, height]` in meters
- **Rotation**: Quaternion `[w, x, y, z]` representing orientation in global frame
- **Coordinate Convention**: Right-handed system (X=East, Y=North, Z=Up)

#### 2. Ego Vehicle Pose
**File**: `ego_pose.json`  
**Coordinate System**: Global coordinates  
**Purpose**: Vehicle position and orientation at each timestamp

```json
{
    "token": "ego_pose_token",
    "timestamp": 1532402927647951,
    "translation": [x, y, z],          # Ego position in global coordinates
    "rotation": [w, x, y, z]           # Ego orientation quaternion in global frame
}
```

#### 3. Sensor Calibration
**File**: `calibrated_sensor.json`  
**Coordinate System**: Relative to ego vehicle  
**Purpose**: Sensor position and orientation relative to ego vehicle

```json
{
    "token": "sensor_calibration_token",
    "sensor_token": "sensor_id",
    "translation": [x, y, z],          # Sensor position relative to ego vehicle
    "rotation": [w, x, y, z],          # Sensor orientation relative to ego vehicle
    "camera_intrinsic": [[fx, 0, cx],  # Camera intrinsic matrix (cameras only)
                         [0, fy, cy],
                         [0, 0, 1]]
}
```

#### 4. Sample Data
**File**: `sample_data.json`  
**Purpose**: Links sensor data files to timestamps and calibrations

```json
{
    "token": "sample_data_token",
    "sample_token": "sample_id",
    "ego_pose_token": "ego_pose_id",
    "calibrated_sensor_token": "sensor_calibration_id",
    "filename": "samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg",
    "fileformat": "jpg",
    "timestamp": 1532402927612460,
    "is_key_frame": true
}
```

### Coordinate Transformation Process Using Sample Data

The sample data structure enables the complete coordinate transformation pipeline. Here's how to use the linked data for transformations:

#### Step-by-Step Transformation Workflow

**1. Load Required Data Using Sample Data Tokens**

```python
# Using the sample_data.json information
sample_data = {
    "token": "sample_data_token",
    "sample_token": "sample_id", 
    "ego_pose_token": "ego_pose_id",
    "calibrated_sensor_token": "sensor_calibration_id",
    "filename": "samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg",
    "timestamp": 1532402927612460
}

# Load corresponding data using tokens
ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
sensor_calibration = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
sample_annotations = nusc.get('sample', sample_data['sample_token'])['anns']
```

**2. Extract Transformation Matrices**

```python
# From ego_pose.json (Global coordinates)
ego_pose_data = {
    "translation": [463.12, 1080.45, 1.84],    # Ego position in global coordinates
    "rotation": [0.9659, 0.0, 0.0, 0.2588]     # Ego orientation quaternion
}

# From calibrated_sensor.json (Relative to ego vehicle)  
sensor_calibration_data = {
    "translation": [1.70, 0.0, 1.54],          # Camera position relative to ego
    "rotation": [0.7071, 0.0, 0.0, 0.7071],    # Camera orientation relative to ego
    "camera_intrinsic": [[1266.4, 0.0, 816.3], # Camera intrinsic matrix
                         [0.0, 1266.4, 491.5],
                         [0.0, 0.0, 1.0]]
}
```

**3. Complete Transformation Pipeline Example**

```python
import numpy as np
from pyquaternion import Quaternion

# Example: Transform 3D bounding box from global to image coordinates

# Step 1: Get 3D bounding box in global coordinates
bbox_annotation = {
    "translation": [465.2, 1085.1, 1.2],       # Box center in global coordinates
    "size": [4.5, 1.8, 1.5],                   # [length, width, height]
    "rotation": [0.9848, 0.0, 0.0, 0.1736]     # Box orientation quaternion
}

# Generate 8 corners of 3D bounding box in global coordinates
box_corners_global = get_3d_box_corners(
    bbox_annotation['translation'],
    bbox_annotation['size'], 
    bbox_annotation['rotation']
)

# Step 2: Global → Ego Vehicle transformation
ego_translation = np.array(ego_pose_data['translation'])
ego_rotation = Quaternion(ego_pose_data['rotation'])
global_to_ego = transform_matrix(ego_translation, ego_rotation, inverse=True)

box_corners_ego = global_to_ego @ np.vstack([box_corners_global, np.ones((1, 8))])

# Step 3: Ego Vehicle → Camera transformation  
cam_translation = np.array(sensor_calibration_data['translation'])
cam_rotation = Quaternion(sensor_calibration_data['rotation'])
ego_to_cam = transform_matrix(cam_translation, cam_rotation, inverse=True)

box_corners_camera = ego_to_cam @ box_corners_ego

# Step 4: Camera → Image projection
camera_intrinsic = np.array(sensor_calibration_data['camera_intrinsic'])
image_points = view_points(box_corners_camera[:3, :], camera_intrinsic, normalize=True)

# Step 5: Extract 2D bounding box
x_coords = image_points[0, :]
y_coords = image_points[1, :]
bbox_2d = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

print(f"2D Bounding Box: {bbox_2d}")
# Output: [245.3, 180.7, 580.1, 420.9] (pixel coordinates)
```

**4. Data Flow Visualization**

```
Sample Data Token Flow:
┌─────────────────┐    ego_pose_token    ┌─────────────────┐
│   sample_data   │ ──────────────────→  │    ego_pose     │
│                 │                      │ (Global coords) │
└─────────────────┘                      └─────────────────┘
         │                                        │
         │ calibrated_sensor_token                │ translation, rotation
         ▼                                        ▼
┌─────────────────┐                      ┌─────────────────┐
│ sensor_calib    │                      │ Transform Matrix│
│ (Ego relative)  │ ──────────────────→  │ Global → Ego    │
└─────────────────┘                      └─────────────────┘
         │                                        │
         │ camera_intrinsic                       │
         ▼                                        ▼
┌─────────────────┐                      ┌─────────────────┐
│ Camera Matrix   │                      │ 3D → 2D Project │
│ (Pixel coords)  │ ──────────────────→  │ Final Result    │
└─────────────────┘                      └─────────────────┘
```

#### Key Implementation Points

1. **Token-Based Data Linking**: Use `ego_pose_token` and `calibrated_sensor_token` to fetch transformation data
2. **Timestamp Synchronization**: The `timestamp` field ensures temporal alignment between sensors
3. **Coordinate System Chain**: Global → Ego → Camera → Image coordinates
4. **Matrix Composition**: Combine transformation matrices for efficient batch processing

```python
# Efficient batch transformation
combined_transform = camera_intrinsic @ ego_to_cam @ global_to_ego
final_2d_points = combined_transform @ global_3d_points_homogeneous
```

#### Practical Usage Example

```python
def process_sample_data(nusc, sample_data_token):
    """Complete pipeline from sample data token to 2D projections"""
    
    # 1. Load all required data using tokens
    sample_data = nusc.get('sample_data', sample_data_token)
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    sensor_calib = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    
    # 2. Get all annotations for this sample
    sample = nusc.get('sample', sample_data['sample_token'])
    
    # 3. Process each annotation
    results = []
    for ann_token in sample['anns']:
        annotation = nusc.get('sample_annotation', ann_token)
        
        # Transform from global to image coordinates
        bbox_2d = project_3d_box_to_2d(
            annotation, ego_pose, sensor_calib
        )
        
        results.append({
            'category': annotation['category_name'],
            'bbox_2d': bbox_2d,
            'visibility': annotation['visibility']
        })
    
    return results
```

This workflow demonstrates how the sample data structure enables seamless coordinate transformations by linking all necessary calibration and pose information through tokens.

### Annotation Coordinate System Summary

| Annotation Type | Coordinate System | Reference Frame | Units |
|----------------|------------------|-----------------|-------|
| **3D Bounding Boxes** | Global | World coordinates | Meters |
| **Ego Poses** | Global | World coordinates | Meters |
| **Sensor Calibration** | Ego Vehicle | Relative to ego | Meters |
| **Camera Intrinsics** | Camera | Pixel coordinates | Pixels |

### Object Categories

NuScenes includes 23 object categories across different domains:

**Vehicles**: `car`, `truck`, `bus`, `trailer`, `construction_vehicle`, `emergency_vehicle`, `motorcycle`, `bicycle`

**Humans**: `adult`, `child`, `police_officer`, `construction_worker`

**Objects**: `traffic_cone`, `barrier`, `debris`, `pushable_pullable`, `movable_object`

**Static**: `animal` (rare cases)

### Visibility Levels

Annotations include visibility information for occlusion handling:

- **Level 1**: 0-40% visible
- **Level 2**: 40-60% visible  
- **Level 3**: 60-80% visible
- **Level 4**: 80-100% visible

### Implementation Notes

When working with NuScenes annotations:

1. **All 3D annotations are in global coordinates** - transform to desired coordinate system
2. **Quaternions use [w, x, y, z] format** - be careful with different libraries' conventions
3. **Box dimensions follow [width, length, height]** - width=left-right, length=front-back
4. **Use `num_lidar_pts` and `num_radar_pts`** for filtering low-quality annotations
5. **Check visibility levels** for occlusion-aware training

## Coordinate Systems

Understanding NuScenes coordinate systems is crucial for correct transformations:

### 1. Global Coordinate System
- **Origin**: Arbitrary reference point in the world
- **Axes**: Right-handed coordinate system
  - X: East direction
  - Y: North direction  
  - Z: Up direction (gravity opposite)
- **Units**: Meters
- **Usage**: All ego poses and annotations are defined in global coordinates

### 2. Ego Vehicle Coordinate System
- **Origin**: Center of the ego vehicle's rear axle
- **Axes**: Right-handed coordinate system relative to vehicle
  - X: Forward direction (vehicle's front)
  - Y: Left direction (vehicle's left side)
  - Z: Up direction (vehicle's roof)
- **Units**: Meters
- **Usage**: Sensor calibrations are defined relative to ego vehicle

### 3. Camera Coordinate System
- **Origin**: Camera's optical center
- **Axes**: Standard computer vision convention
  - X: Right direction (image width)
  - Y: Down direction (image height)
  - Z: Forward direction (into the scene)
- **Units**: Meters
- **Usage**: 3D points in camera space before projection

### 4. Image Coordinate System
- **Origin**: Top-left corner of the image
- **Axes**: 2D pixel coordinates
  - u: Horizontal axis (0 to image_width-1)
  - v: Vertical axis (0 to image_height-1)
- **Units**: Pixels
- **Usage**: Final 2D bounding box coordinates

## Coordinate Transformations

The transformation pipeline varies depending on the visualization type. Here's a comprehensive overview:

### Transformation Pipelines by Visualization Type

#### 1. 2D Image Bounding Box
```
Global → Ego Vehicle → Camera → Image (2D Projection)
```
**Purpose**: Project 3D objects onto 2D camera images for object detection
**Output**: 2D rectangular bounding boxes in pixel coordinates

#### 2. 3D Image Bounding Box  
```
Global → Ego Vehicle → Camera → Image (3D Wireframe)
```
**Purpose**: Visualize 3D object structure overlaid on camera images
**Output**: 3D wireframe boxes projected onto 2D image plane

#### 3. BEV (Bird's Eye View)
```
Global → Ego Vehicle → BEV Projection
```
**Purpose**: Top-down view for spatial understanding and path planning
**Output**: 2D boxes in ego vehicle coordinate system (X-Y plane)

#### 4. 3D LiDAR Bounding Box
```
Global → Ego Vehicle → LiDAR Sensor
```
**Purpose**: 3D object detection and tracking in point cloud data
**Output**: 3D oriented bounding boxes in LiDAR coordinate system

### Mathematical Foundation

Each transformation uses homogeneous coordinates and 4x4 transformation matrices:

```
T = [R  t]
    [0  1]
```

Where:
- `R`: 3x3 rotation matrix
- `t`: 3x1 translation vector

## Detailed Transformation Processes by Visualization Type

### 1. 2D Image Bounding Box Transformation

**Complete Pipeline**: Global → Ego Vehicle → Camera → Image (2D Projection)

#### Step 1: Global to Ego Vehicle
```python
# Get ego pose (position and orientation in global coordinates)
ego_translation = [x, y, z]  # ego position in global coordinates
ego_rotation = [w, x, y, z]  # ego orientation quaternion

# Create inverse transformation matrix (global → ego)
global_to_ego = transform_matrix(ego_translation, ego_rotation, inverse=True)

# Apply transformation to 3D box corners
box_corners_ego = global_to_ego @ box_corners_global_homogeneous
```

#### Step 2: Ego Vehicle to Camera
```python
# Get camera calibration (position and orientation relative to ego)
cam_translation = [x, y, z]  # camera position relative to ego
cam_rotation = [w, x, y, z]  # camera orientation quaternion

# Create inverse transformation matrix (ego → camera)
ego_to_cam = transform_matrix(cam_translation, cam_rotation, inverse=True)

# Apply transformation
box_corners_camera = ego_to_cam @ box_corners_ego
```

#### Step 3: Camera to Image (2D Projection)
```python
# Project 3D camera coordinates to 2D image pixels
# Using camera intrinsic matrix
image_points = view_points(box_corners_camera, camera_intrinsic, normalize=True)

# Extract 2D bounding box from projected corners
x_coords = image_points[0, :]  # u coordinates
y_coords = image_points[1, :]  # v coordinates

# Create 2D bounding box
bbox_2d = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
```

**Key Characteristics**:
- **Output**: 2D rectangular box `[x_min, y_min, x_max, y_max]` in pixel coordinates
- **Use Case**: Object detection, tracking in camera images
- **Coordinate System**: Image pixels (u, v)
- **Implementation**: See [`project_3d_box_to_2d()`](scripts/nuscenes.py#L810-L910) function

### 2. 3D Image Bounding Box Transformation

**Complete Pipeline**: Global → Ego Vehicle → Camera → Image (3D Wireframe)

#### Steps 1-2: Same as 2D (Global → Ego → Camera)
The first two steps are identical to 2D bounding box transformation.

#### Step 3: 3D Wireframe Projection
```python
# Project all 8 corners of 3D box to image
corners_3d_camera = ego_to_cam @ (global_to_ego @ box_corners_global)
image_corners = view_points(corners_3d_camera, camera_intrinsic, normalize=True)

# Define 3D box edges (12 edges connecting 8 corners)
edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],  # top face
    [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
]

# Draw wireframe on image
for edge in edges:
    start_point = image_corners[:, edge[0]]
    end_point = image_corners[:, edge[1]]
    # Draw line from start_point to end_point
```

**Key Characteristics**:
- **Output**: 8 projected corner points + 12 connecting edges
- **Use Case**: 3D object visualization on camera images
- **Coordinate System**: Image pixels (u, v) with depth information preserved
- **Visualization**: Wireframe overlay showing 3D structure

### 3. BEV (Bird's Eye View) Transformation

**Simplified Pipeline**: Global → Ego Vehicle → BEV Projection

#### Step 1: Global to Ego Vehicle (Same as above)
```python
global_to_ego = transform_matrix(ego_translation, ego_rotation, inverse=True)
box_corners_ego = global_to_ego @ box_corners_global_homogeneous
```

#### Step 2: BEV Projection (Top-Down View)
```python
# Extract X-Y coordinates (ignore Z for top-down view)
bev_points = box_corners_ego[:2, :]  # Take only X, Y coordinates

# Convert to BEV image coordinates
# Typically: X-forward, Y-left in ego coordinates
# BEV image: X-right, Y-up in image coordinates
bev_x = bev_points[1, :] * scale + offset_x  # Y_ego → X_bev
bev_y = -bev_points[0, :] * scale + offset_y  # -X_ego → Y_bev

# Create 2D polygon from projected corners
bev_polygon = list(zip(bev_x, bev_y))
```

**Key Characteristics**:
- **Output**: 2D polygon in BEV coordinate system
- **Use Case**: Path planning, spatial reasoning, multi-object tracking
- **Coordinate System**: Top-down view (X-Y plane of ego vehicle)
- **Advantages**: No occlusion, consistent scale, easy distance measurement

### 4. 3D LiDAR Bounding Box Transformation

**Simplified Pipeline**: Global → Ego Vehicle → LiDAR Sensor

#### Step 1: Global to Ego Vehicle (Same as above)
```python
global_to_ego = transform_matrix(ego_translation, ego_rotation, inverse=True)
box_corners_ego = global_to_ego @ box_corners_global_homogeneous
```

#### Step 2: Ego to LiDAR Sensor (if needed)
```python
# Most LiDAR sensors are mounted close to ego vehicle center
# Often LiDAR coordinate system ≈ ego coordinate system
# If transformation needed:
lidar_translation = [x, y, z]  # LiDAR position relative to ego
lidar_rotation = [w, x, y, z]  # LiDAR orientation quaternion

ego_to_lidar = transform_matrix(lidar_translation, lidar_rotation, inverse=True)
box_corners_lidar = ego_to_lidar @ box_corners_ego
```

#### Step 3: 3D Box Representation
```python
# 3D bounding box in LiDAR coordinates
# Typically represented as:
# - Center: [x, y, z]
# - Dimensions: [length, width, height]  
# - Orientation: yaw angle or quaternion

box_3d = {
    'center': np.mean(box_corners_lidar[:3, :], axis=1),
    'size': [length, width, height],
    'orientation': yaw_angle
}
```

**Key Characteristics**:
- **Output**: 3D oriented bounding box with center, size, and orientation
- **Use Case**: 3D object detection, autonomous driving, robotics
- **Coordinate System**: 3D LiDAR sensor coordinates
- **Advantages**: Direct 3D measurements, no projection distortion

## Transformation Summary

| Visualization Type | Pipeline | Output Format | Primary Use Case |
|-------------------|----------|---------------|------------------|
| **2D Image BBox** | Global→Ego→Camera→Image | `[x_min, y_min, x_max, y_max]` | Object detection |
| **3D Image BBox** | Global→Ego→Camera→Image | 8 corners + 12 edges | 3D visualization |
| **BEV** | Global→Ego→BEV | 2D polygon | Path planning |
| **3D LiDAR BBox** | Global→Ego→LiDAR | Center + Size + Orientation | 3D detection |

### Original Transformation Steps (for reference)

### 1. Global to Ego Vehicle Transformation

**Purpose**: Transform from world coordinates to ego vehicle coordinates

**Implementation**: See [`transform_matrix()`](scripts/nuscenes.py#L647-L685) function

```python
# Get ego pose (position and orientation in global coordinates)
ego_translation = [x, y, z]  # ego position in global coordinates
ego_rotation = [w, x, y, z]  # ego orientation quaternion

# Create inverse transformation matrix (global → ego)
global_to_ego = transform_matrix(ego_translation, ego_rotation, inverse=True)

# Apply transformation
points_ego = global_to_ego @ points_global_homogeneous
```

**Key Points**:
- Use `inverse=True` to get global→ego transformation
- Ego pose represents ego vehicle's position/orientation in global coordinates
- Inverse transformation moves from global to ego coordinate system

### 2. Ego Vehicle to Camera Transformation

**Purpose**: Transform from ego vehicle coordinates to camera coordinates

**Implementation**: See [`transform_matrix()`](scripts/nuscenes.py#L647-L685) function

```python
# Get camera calibration (position and orientation relative to ego)
cam_translation = [x, y, z]  # camera position relative to ego
cam_rotation = [w, x, y, z]  # camera orientation quaternion

# Create inverse transformation matrix (ego → camera)
ego_to_cam = transform_matrix(cam_translation, cam_rotation, inverse=True)

# Apply transformation
points_camera = ego_to_cam @ points_ego_homogeneous
```

**Key Points**:
- Camera calibration is relative to ego vehicle
- Use `inverse=True` to get ego→camera transformation
- Results in 3D points in camera coordinate system

### 3. Camera to Image Transformation

**Purpose**: Project 3D camera coordinates to 2D image pixels

**Implementation**: See [`view_points()`](scripts/nuscenes.py#L567-L646) function

```python
# Camera intrinsic matrix
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]

# Project 3D points to 2D
points_2d = view_points(points_3d_camera, K, normalize=True)
```

**Key Points**:
- Uses perspective projection: `u = fx * X/Z + cx`, `v = fy * Y/Z + cy`
- Points with Z ≤ 0 are behind the camera and invalid
- `normalize=True` performs the division by Z coordinate

## 3D Bounding Box Processing

### Bounding Box Representation

NuScenes represents 3D bounding boxes with:
- **Center**: `[x, y, z]` in global coordinates
- **Size**: `[width, length, height]` in meters
- **Rotation**: Quaternion `[w, x, y, z]` in global coordinates

### Corner Point Generation

**Implementation**: See [`get_3d_box_corners()`](scripts/nuscenes.py#L704-L767) function

The function generates 8 corner points of a 3D bounding box:

```python
def get_3d_box_corners(center, size, rotation):
    """
    Generate 8 corner points of 3D bounding box.
    
    Corner ordering (NuScenes convention):
    Bottom face (z = -height/2):
      1 ---- 0
      |      |
      |      |  
      2 ---- 3
    
    Top face (z = +height/2):
      5 ---- 4
      |      |
      |      |
      6 ---- 7
    """
```

**Process**:
1. Define unit box corners centered at origin
2. Scale by box dimensions
3. Apply rotation using quaternion
4. Translate to final position

**Key Points**:
- Corner ordering follows NuScenes convention
- Vehicle front direction aligns with +Y axis in box coordinates
- Rotation is applied before translation

## 2D Projection Pipeline

### Complete Pipeline Implementation

**Implementation**: See [`project_3d_box_to_2d()`](scripts/nuscenes.py#L810-L946) function

The complete pipeline transforms 3D bounding boxes to 2D image coordinates:

```python
def project_3d_box_to_2d(center_3d, size_3d, rotation_3d, 
                        cam_translation, cam_rotation, camera_intrinsic,
                        ego_translation, ego_rotation, debug=False):
    """
    Complete transformation pipeline:
    Global → Ego Vehicle → Camera → Image
    """
```

### Step-by-Step Process

#### Step 1: Generate 3D Box Corners
```python
# Generate 8 corner points in global coordinates
corners_3d_global = get_3d_box_corners(center_3d, size_3d, rotation_3d)
```

#### Step 2: Global → Ego Transformation
```python
# Transform from global to ego vehicle coordinates
global_to_ego = transform_matrix(ego_translation, ego_rotation, inverse=True)
corners_ego = apply_transformation(global_to_ego, corners_3d_global)
```

#### Step 3: Ego → Camera Transformation
```python
# Transform from ego to camera coordinates
ego_to_cam = transform_matrix(cam_translation, cam_rotation, inverse=True)
corners_camera = apply_transformation(ego_to_cam, corners_ego)
```

#### Step 4: Camera → Image Projection
```python
# Project 3D camera coordinates to 2D image pixels
corners_2d = view_points(corners_camera.T, camera_intrinsic, normalize=True)
```

### Visibility and Validation

**Implementation**: See [`box_in_image()`](scripts/nuscenes.py#L60-L91) function

```python
def box_in_image(corners_3d, corners_2d, intrinsic, imsize, vis_level):
    """
    Check if bounding box is visible in image.
    
    vis_level options:
    - BoxVisibility.ALL: All corners must be inside image
    - BoxVisibility.ANY: At least one corner must be visible  
    - BoxVisibility.NONE: No visibility requirement
    """
```

## Code Implementation

### Key Functions and Their Roles

1. **[`view_points()`](scripts/nuscenes.py#L567-L646)**
   - Handles perspective projection from 3D to 2D
   - Applies camera intrinsic matrix
   - Normalizes homogeneous coordinates

2. **[`transform_matrix()`](scripts/nuscenes.py#L647-L685)**
   - Creates 4x4 transformation matrices
   - Handles translation and rotation
   - Supports inverse transformations

3. **[`quaternion_to_rotation_matrix()`](scripts/nuscenes.py#L686-L703)**
   - Converts quaternions to rotation matrices
   - Handles both scipy and manual implementations
   - Ensures numerical stability

4. **[`get_3d_box_corners()`](scripts/nuscenes.py#L704-L767)**
   - Generates 8 corner points of 3D bounding box
   - Follows NuScenes corner ordering convention
   - Applies scaling, rotation, and translation

5. **[`project_3d_box_to_2d()`](scripts/nuscenes.py#L810-L946)**
   - Complete transformation pipeline
   - Handles all coordinate system conversions
   - Provides debug information and error handling

### Data Loading and Validation Functions

6. **[`load_nuscenes_data()`](scripts/nuscenes.py#L2323-L2381)**
   - Loads all NuScenes JSON annotation files
   - Returns structured dictionary with scenes, samples, annotations
   - Handles file validation and error reporting

7. **[`prepare_sample_data()`](scripts/nuscenes.py#L2382-L2471)**
   - Extracts data for specific sample index
   - Organizes annotations, camera data, and sensor info
   - Provides ready-to-use data structure for visualization

8. **[`validate_dataset_structure()`](scripts/nuscenes.py#L92-L127)**
   - Validates NuScenes directory structure
   - Checks for required folders and annotation files
   - Returns boolean validation result

9. **[`diagnose_dataset_issues()`](scripts/nuscenes.py#L251-L433)**
   - Comprehensive dataset health check
   - Identifies missing files, corrupted data, and structure issues
   - Provides detailed diagnostic report

### Visualization and Rendering Functions

10. **[`visualize_lidar_3d_open3d()`](scripts/nuscenes.py#L1398-L1769)**
    - Interactive 3D LiDAR point cloud visualization
    - Uses Open3D for high-quality rendering
    - Supports 3D bounding box overlays

11. **[`visualize_bev_with_boxes()`](scripts/nuscenes.py#L1998-L2177)**
    - Bird's Eye View (BEV) visualization
    - Projects LiDAR points to 2D top-down view
    - Renders 3D boxes as 2D rectangles in BEV space

12. **[`visualize_lidar_projection()`](scripts/nuscenes.py#L1930-L1997)**
    - Projects LiDAR points onto camera images
    - Color-codes points by distance or intensity
    - Handles camera-LiDAR calibration

13. **[`create_combined_visualization()`](scripts/nuscenes.py#L2758-L2807)**
    - Creates multi-panel visualization layouts
    - Combines camera, LiDAR, and BEV views
    - Generates publication-ready figures

### Drawing and Rendering Utilities

14. **[`draw_3d_box_2d()`](scripts/nuscenes.py#L1211-L1288)**
    - Draws 3D bounding box wireframes on 2D images
    - Handles perspective projection and clipping
    - Supports custom colors and line styles

15. **[`draw_2d_bbox()`](scripts/nuscenes.py#L1003-L1075)**
    - Draws 2D bounding boxes on images
    - Includes category labels and confidence scores
    - Handles image boundary clipping

16. **[`draw_bev_box_2d()`](scripts/nuscenes.py#L2178-L2258)**
    - Renders 2D boxes in Bird's Eye View
    - Handles rotation and scaling in BEV space
    - Supports transparency and color coding

### LiDAR Processing Functions

17. **[`load_lidar_points()`](scripts/nuscenes.py#L1363-L1397)**
    - Loads LiDAR point cloud from .pcd files
    - Handles different point cloud formats
    - Returns numpy array with x, y, z, intensity

18. **[`transform_lidar_to_ego()`](scripts/nuscenes.py#L1325-L1362)**
    - Transforms LiDAR points to ego vehicle coordinate system
    - Applies sensor calibration parameters
    - Handles translation and rotation transformations

19. **[`project_lidar_to_camera()`](scripts/nuscenes.py#L1876-L1929)**
    - Projects 3D LiDAR points to camera image plane
    - Applies full transformation pipeline
    - Filters points behind camera or outside image

### Coordinate System Utilities

20. **[`get_lidar_calibration_info()`](scripts/nuscenes.py#L1289-L1324)**
    - Extracts LiDAR sensor calibration parameters
    - Retrieves translation and rotation from ego vehicle
    - Returns calibration dictionary for transformations

21. **[`process_camera_data()`](scripts/nuscenes.py#L2472-L2570)**
    - Processes camera sensor data and calibration
    - Extracts intrinsic and extrinsic parameters
    - Prepares camera info for coordinate transformations

22. **[`process_3d_annotations_for_camera()`](scripts/nuscenes.py#L2571-L2677)**
    - Filters and processes 3D annotations for specific camera
    - Applies visibility checks and coordinate transformations
    - Returns camera-specific annotation data

### Utility and Helper Functions

23. **[`get_2d_bbox_from_3d_projection()`](scripts/nuscenes.py#L952-L1002)**
    - Computes 2D bounding box from projected 3D corners
    - Finds min/max coordinates of projected points
    - Handles edge cases and invalid projections

24. **[`clip_line_to_image()`](scripts/nuscenes.py#L1138-L1210)**
    - Clips 3D box edges to image boundaries
    - Implements line-rectangle intersection algorithm
    - Prevents drawing outside image bounds

25. **[`extract_nuscenes_subset()`](scripts/nuscenes.py#L3179-L3318)**
    - Creates smaller dataset subset for testing
    - Copies relevant samples and annotations
    - Maintains data structure integrity

### Usage Example

```python
# Load NuScenes data
nuscenes_data = load_nuscenes_data(nuscenes_root)
sample_data = prepare_sample_data(nuscenes_data, sample_idx=0)

# Validate dataset first
if not validate_dataset_structure(nuscenes_root):
    diagnosis = diagnose_dataset_issues(nuscenes_root)
    print("Dataset issues found:", diagnosis)

# Get annotation and camera info
annotation = sample_data['annotations'][0]  # First annotation
camera_info = sample_data['cameras']['CAM_FRONT']

# Extract 3D bounding box parameters
center_3d = annotation['translation']
size_3d = annotation['size'] 
rotation_3d = annotation['rotation']

# Extract camera and ego pose information
cam_translation = camera_info['translation']
cam_rotation = camera_info['rotation']
camera_intrinsic = camera_info['camera_intrinsic']
ego_translation = camera_info['ego_translation']
ego_rotation = camera_info['ego_rotation']

# Project 3D box to 2D
corners_2d, corners_3d = project_3d_box_to_2d(
    center_3d, size_3d, rotation_3d,
    cam_translation, cam_rotation, camera_intrinsic,
    ego_translation, ego_rotation, debug=True
)

# Check visibility
is_visible = box_in_image(
    corners_3d, corners_2d, camera_intrinsic, 
    (1600, 900), BoxVisibility.ANY
)
```

## Common Issues and Solutions

### 1. Incorrect Coordinate System Assumptions

**Problem**: Mixing up coordinate system conventions
**Solution**: Always verify axis directions and origins
```python
# Global: X=East, Y=North, Z=Up
# Ego: X=Forward, Y=Left, Z=Up  
# Camera: X=Right, Y=Down, Z=Forward
```

### 2. Transformation Matrix Order

**Problem**: Applying transformations in wrong order
**Solution**: Follow the pipeline: Global → Ego → Camera → Image
```python
# Correct order
global_to_ego = transform_matrix(ego_translation, ego_rotation, inverse=True)
ego_to_cam = transform_matrix(cam_translation, cam_rotation, inverse=True)
```

### 3. Quaternion Normalization

**Problem**: Unnormalized quaternions causing incorrect rotations
**Solution**: Always normalize quaternions before use
```python
q = np.array(quaternion)
q = q / np.linalg.norm(q)  # Normalize
```

### 4. Behind-Camera Points

**Problem**: Points with negative Z coordinates in camera space
**Solution**: Check depth values and handle appropriately
```python
depths = corners_camera[:, 2]
if np.any(depths <= 0):
    print("Warning: Some points behind camera")
```

### 5. Homogeneous Coordinates

**Problem**: Forgetting to use homogeneous coordinates for transformations
**Solution**: Always add 1 as 4th dimension
```python
points_homogeneous = np.ones((points.shape[0], 4))
points_homogeneous[:, :3] = points
```

## Best Practices

### 1. Validation and Testing

- Always validate transformations with known test cases
- Check intermediate results at each transformation step
- Use debug mode to trace coordinate transformations
- Verify corner ordering matches NuScenes convention

### 2. Error Handling

- Check for points behind camera (Z ≤ 0)
- Validate input parameters (non-zero quaternions, valid matrices)
- Handle edge cases (very small/large bounding boxes)
- Provide meaningful error messages

### 3. Performance Optimization

- Batch process multiple boxes when possible
- Cache transformation matrices when processing multiple annotations
- Use vectorized operations instead of loops
- Pre-compute frequently used values

### 4. Code Organization

- Separate coordinate transformation logic from visualization
- Use consistent parameter naming across functions
- Document coordinate system conventions clearly
- Provide usage examples and test cases

### 5. Debugging Techniques

```python
# Enable debug mode for detailed transformation info
corners_2d, corners_3d = project_3d_box_to_2d(..., debug=True)

# Visualize intermediate results
plt.figure(figsize=(15, 5))
plt.subplot(131); plot_global_coordinates(corners_global)
plt.subplot(132); plot_ego_coordinates(corners_ego)  
plt.subplot(133); plot_camera_coordinates(corners_camera)
```

### 6. Coordinate System Verification

```python
def verify_coordinate_systems():
    """Verify coordinate system transformations with known points."""
    # Test point at ego vehicle center
    ego_center = np.array([0, 0, 0, 1])  # Origin in ego coordinates
    
    # Should transform to ego_translation in global coordinates
    global_point = ego_to_global @ ego_center
    assert np.allclose(global_point[:3], ego_translation)
    
    print("Coordinate system verification passed!")
```

This tutorial provides a comprehensive guide to understanding and implementing coordinate transformations in the NuScenes dataset. The key to success is understanding the coordinate system conventions, following the transformation pipeline correctly, and validating results at each step.