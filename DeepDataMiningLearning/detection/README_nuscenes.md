# NuScenes Dataset for PyTorch Object Detection

This module provides a PyTorch Dataset class for loading and processing NuScenes data for object detection tasks. It follows the same interface as the KITTI dataset but handles NuScenes-specific data structures and coordinate transformations.

## Files

- `dataset_nuscenes.py` - Main dataset implementation
- `test_nuscenes_dataset.py` - Test script for validation
- `README_nuscenes.md` - This documentation file

## Features

- **PyTorch Dataset Interface**: Compatible with PyTorch DataLoader and training pipelines
- **3D to 2D Projection**: Converts 3D bounding boxes to 2D image coordinates
- **Multiple Camera Support**: Supports all NuScenes camera types (CAM_FRONT, CAM_FRONT_LEFT, etc.)
- **Category Mapping**: Maps NuScenes categories to detection class indices
- **Data Validation**: Validates dataset structure and required files
- **Transforms Support**: Compatible with torchvision transforms
- **Batch Processing**: Custom collate function for efficient batching

## Supported Categories

The dataset supports 10 object categories from NuScenes:

1. `car` (class 0)
2. `truck` (class 1)
3. `bus` (class 2)
4. `trailer` (class 3)
5. `construction_vehicle` (class 4)
6. `pedestrian` (class 5)
7. `motorcycle` (class 6)
8. `bicycle` (class 7)
9. `traffic_cone` (class 8)
10. `barrier` (class 9)

## Usage

### Basic Usage

```python
from dataset_nuscenes import NuScenesDataset, create_nuscenes_transforms

# Create dataset
transform = create_nuscenes_transforms(train=True)
dataset = NuScenesDataset(
    root_dir='/path/to/nuscenes/v1.0-trainval',
    split='train',
    camera_types=['CAM_FRONT'],
    transform=transform
)

# Get a sample
image, target = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Number of boxes: {len(target['boxes'])}")
print(f"Labels: {target['labels']}")
```

### With DataLoader

```python
from torch.utils.data import DataLoader
from dataset_nuscenes import collate_fn

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4
)

for images, targets in dataloader:
    # Training loop
    pass
```

### Multiple Cameras

```python
# Use multiple camera views
dataset = NuScenesDataset(
    root_dir='/path/to/nuscenes/v1.0-trainval',
    camera_types=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'],
    transform=transform
)
```

## Dataset Structure

The NuScenes dataset should be organized as follows:

```
nuscenes_root/
├── samples/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── LIDAR_TOP/
│   └── RADAR_*/
├── sweeps/
│   └── (similar structure)
└── v1.0-trainval/
    ├── sample.json
    ├── sample_data.json
    ├── sample_annotation.json
    ├── instance.json
    ├── category.json
    ├── attribute.json
    ├── visibility.json
    ├── sensor.json
    ├── calibrated_sensor.json
    ├── ego_pose.json
    ├── log.json
    ├── scene.json
    └── map.json
```

## Target Format

The dataset returns targets in the standard PyTorch object detection format:

```python
target = {
    'boxes': torch.Tensor,      # Shape: (N, 4) - [x1, y1, x2, y2]
    'labels': torch.Tensor,     # Shape: (N,) - class indices
    'image_id': torch.Tensor,   # Shape: (1,) - unique image identifier
    'area': torch.Tensor,       # Shape: (N,) - box areas
    'iscrowd': torch.Tensor     # Shape: (N,) - crowd annotations (always 0)
}
```

## Testing

Run the test script to validate your dataset:

```bash
python test_nuscenes_dataset.py /path/to/nuscenes/v1.0-trainval
```

The test script will:
- Validate dataset structure
- Test basic loading functionality
- Test with transforms
- Test DataLoader integration
- Test multiple camera support

## Parameters

### NuScenesDataset Parameters

- `root_dir` (str): Root directory of NuScenes dataset
- `split` (str): Dataset split ('train', 'val', 'test') - currently uses all samples
- `camera_types` (List[str]): Camera types to use (default: ['CAM_FRONT'])
- `transform` (Optional): Transform to apply to images
- `target_transform` (Optional): Transform to apply to targets
- `load_lidar` (bool): Whether to load LiDAR data (not implemented)
- `max_samples` (Optional[int]): Maximum number of samples (for debugging)

### Available Camera Types

- `CAM_FRONT`
- `CAM_FRONT_RIGHT`
- `CAM_FRONT_LEFT`
- `CAM_BACK`
- `CAM_BACK_LEFT`
- `CAM_BACK_RIGHT`

## Implementation Details

### 3D to 2D Projection

The dataset implements 3D to 2D bounding box projection using:

1. **Coordinate Transformations**: Global → Ego → Camera coordinates
2. **Quaternion Rotations**: Proper handling of 3D rotations
3. **Camera Intrinsics**: Projection using camera calibration parameters
4. **Visibility Filtering**: Only projects boxes visible to the camera

### Data Loading

- **Lazy Loading**: Images and annotations are loaded on-demand
- **Error Handling**: Graceful handling of missing or corrupted data
- **Caching**: Annotation lookup dictionaries for fast access
- **Validation**: Checks for required files and valid data structure

## Limitations

1. **Train/Val Split**: Currently uses all samples (no explicit split implementation)
2. **LiDAR Support**: LiDAR data loading is not implemented
3. **Radar Support**: Radar data is not used
4. **Temporal Information**: Does not use temporal sequences
5. **3D Detection**: Focused on 2D detection only

## Troubleshooting

### Common Issues

1. **Missing Files**: Ensure all required JSON files are present in `v1.0-trainval/`
2. **Image Not Found**: Check that image paths in JSON files match actual file locations
3. **Empty Dataset**: Verify camera types match available data
4. **Projection Errors**: Check camera calibration data completeness

### Debug Mode

Use `max_samples` parameter to limit dataset size for debugging:

```python
dataset = NuScenesDataset(
    root_dir=root_dir,
    max_samples=10  # Only load 10 samples
)
```

## Integration with Training

This dataset is compatible with popular object detection frameworks:

- **Detectron2**: Use with custom data registration
- **MMDetection**: Compatible with COCO-style annotations
- **YOLOv5/YOLOv8**: Convert targets to YOLO format
- **Faster R-CNN**: Direct compatibility with PyTorch models

## Performance Tips

1. **Use Multiple Workers**: Set `num_workers > 0` in DataLoader
2. **Precompute Projections**: Cache 2D boxes if memory allows
3. **Filter Empty Images**: Skip images with no annotations
4. **Optimize Transforms**: Use efficient image preprocessing
5. **Batch Size**: Adjust based on GPU memory and image resolution

## Contributing

To extend this dataset:

1. Add support for additional camera types
2. Implement train/val/test splits
3. Add LiDAR data loading
4. Implement temporal sequence support
5. Add data augmentation specific to autonomous driving

## License

This implementation is based on the NuScenes dataset structure and follows the same licensing terms as the original NuScenes dataset.