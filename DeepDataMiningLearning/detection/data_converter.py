"""
Data conversion utilities for exporting KITTI dataset to Ultralytics YOLO format
"""
import os
import shutil
import yaml
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

def export_kitti_to_ultralytics_format(dataset, output_dir, split_name="train"):
    """
    Export KITTI dataset to Ultralytics YOLO file system format
    
    Args:
        dataset: KittiDataset instance with output_format="yolo"
        output_dir: Output directory for exported data
        split_name: Split name (train/val/test)
    
    Returns:
        dict: Dataset configuration for Ultralytics
    """
    print(f"📦 Exporting KITTI dataset to Ultralytics format...")
    print(f"   Split: {split_name}")
    print(f"   Output directory: {output_dir}")
    print(f"   Dataset size: {len(dataset)} samples")
    
    # Create directory structure
    output_path = Path(output_dir)
    images_dir = output_path / "images" / split_name
    labels_dir = output_path / "labels" / split_name
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get class names from dataset
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
    elif hasattr(dataset, 'class_names'):
        class_names = dataset.class_names
    elif hasattr(dataset, 'INSTANCE_CATEGORY_NAMES'):
        # Use the actual class names from KITTI dataset, excluding background
        class_names = [name for name in dataset.INSTANCE_CATEGORY_NAMES if name not in ['__background__', 'DontCare']]
    else:
        # Default KITTI classes (including Misc which is class 8 in KITTI)
        class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    
    num_classes = len(class_names)
    
    # Export each sample
    exported_count = 0
    for idx in tqdm(range(len(dataset)), desc=f"Exporting {split_name} data"):
        try:
            # Get sample from dataset (YOLO format)
            sample = dataset[idx]
            
            # Extract data from YOLO format
            if isinstance(sample, dict):
                image_tensor = sample.get('image', sample.get('img'))
                bboxes = sample.get('bboxes', sample.get('boxes'))
                labels = sample.get('labels', sample.get('cls'))
                
                # Handle different tensor formats
                if isinstance(image_tensor, torch.Tensor):
                    # Convert tensor to PIL Image
                    if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
                        # CHW format
                        image_tensor = image_tensor.permute(1, 2, 0)
                    
                    # Denormalize if needed (assuming ImageNet normalization)
                    if image_tensor.max() <= 1.0:
                        image_tensor = image_tensor * 255.0
                    
                    image_array = image_tensor.cpu().numpy().astype('uint8')
                    image = Image.fromarray(image_array)
                else:
                    image = image_tensor
            else:
                # Handle tuple format (image, target)
                image, target = sample
                bboxes = target.get('boxes')
                labels = target.get('labels')
            
            # Generate unique filename
            image_filename = f"{split_name}_{idx:06d}.jpg"
            label_filename = f"{split_name}_{idx:06d}.txt"
            
            # Save image
            image_path = images_dir / image_filename
            if isinstance(image, Image.Image):
                image.save(image_path, 'JPEG', quality=95)
            else:
                # Handle other image formats
                Image.fromarray(image).save(image_path, 'JPEG', quality=95)
            
            # Save labels in YOLO format
            label_path = labels_dir / label_filename
            with open(label_path, 'w') as f:
                if bboxes is not None and len(bboxes) > 0:
                    # Convert tensors to numpy if needed
                    if isinstance(bboxes, torch.Tensor):
                        bboxes = bboxes.cpu().numpy()
                    if isinstance(labels, torch.Tensor):
                        labels = labels.cpu().numpy()
                    
                    for bbox, label in zip(bboxes, labels):
                        # YOLO format: class_id center_x center_y width height (normalized)
                        # bbox should already be in normalized xywh format from KittiDataset
                        if len(bbox) == 4:  # [x_center, y_center, width, height]
                            class_id = int(label)
                            
                            # Map KITTI class IDs to Ultralytics format (0-based indexing)
                            # KITTI uses: 1=Car, 2=Van, 3=Truck, 4=Pedestrian, 5=Person_sitting, 6=Cyclist, 7=Tram, 8=Misc
                            # Ultralytics expects: 0-based indexing
                            if class_id > 0:  # Skip background (class 0)
                                ultralytics_class_id = class_id - 1  # Convert to 0-based
                                if ultralytics_class_id < num_classes:  # Ensure within valid range
                                    x_center, y_center, width, height = bbox
                                    f.write(f"{ultralytics_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            exported_count += 1
            
        except Exception as e:
            print(f"⚠️  Error exporting sample {idx}: {e}")
            continue
    
    print(f"✅ Successfully exported {exported_count}/{len(dataset)} samples")
    
    # Create dataset configuration
    dataset_config = {
        'path': str(output_path.absolute()),
        'train': f"images/{split_name}" if split_name == "train" else None,
        'val': f"images/{split_name}" if split_name == "val" else None,
        'test': f"images/{split_name}" if split_name == "test" else None,
        'nc': num_classes,
        'names': class_names
    }
    
    # Remove None values
    dataset_config = {k: v for k, v in dataset_config.items() if v is not None}
    
    return dataset_config

def create_ultralytics_yaml_config(train_config, val_config=None, test_config=None, output_path=None):
    """
    Create Ultralytics YAML configuration file
    
    Args:
        train_config: Training dataset configuration
        val_config: Validation dataset configuration (optional)
        test_config: Test dataset configuration (optional)
        output_path: Path to save YAML file
    
    Returns:
        str: Path to created YAML file
    """
    # Merge configurations
    config = train_config.copy()
    
    if val_config:
        config['val'] = val_config.get('val', val_config.get('train'))
    if test_config:
        config['test'] = test_config.get('test', test_config.get('train'))
    
    # Default output path
    if output_path is None:
        output_path = Path(config['path']) / "dataset.yaml"
    else:
        output_path = Path(output_path)
    
    # Save YAML configuration
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"📄 Created dataset configuration: {output_path}")
    print(f"   Classes: {config['nc']}")
    print(f"   Names: {config['names']}")
    
    return str(output_path)

def export_kitti_for_ultralytics(args, output_base_dir=None):
    """
    Complete export pipeline for KITTI dataset to Ultralytics format
    
    Args:
        args: Training arguments
        output_base_dir: Base directory for exported data
    
    Returns:
        str: Path to dataset YAML configuration file
    """
    from DeepDataMiningLearning.detection.dataset import get_dataset
    
    if output_base_dir is None:
        if args.output_dir:
            output_base_dir = os.path.join(args.output_dir, "ultralytics_data")
        else:
            output_base_dir = "./ultralytics_data"
    
    print(f"🔄 Starting KITTI to Ultralytics export pipeline...")
    print(f"   Base output directory: {output_base_dir}")
    
    # Create base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    configs = {}
    
    # Export training data
    print("\n📊 Exporting training data...")
    train_dataset, _ = get_dataset(args.dataset, is_train=True, is_val=False, args=args, output_format="yolo")
    train_config = export_kitti_to_ultralytics_format(train_dataset, output_base_dir, "train")
    configs['train'] = train_config
    
    # Export validation data
    print("\n📊 Exporting validation data...")
    val_dataset, _ = get_dataset(args.dataset, is_train=False, is_val=True, args=args, output_format="yolo")
    val_config = export_kitti_to_ultralytics_format(val_dataset, output_base_dir, "val")
    configs['val'] = val_config
    
    # Create combined YAML configuration
    yaml_path = create_ultralytics_yaml_config(
        train_config, 
        val_config, 
        output_path=os.path.join(output_base_dir, "dataset.yaml")
    )
    
    print(f"\n🎉 Export completed successfully!")
    print(f"   Dataset YAML: {yaml_path}")
    print(f"   Ready for Ultralytics training!")
    
    return yaml_path