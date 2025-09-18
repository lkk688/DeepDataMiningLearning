#!/usr/bin/env python3
"""
Test script for NuScenes Dataset

This script tests the NuScenesDataset class to ensure it works correctly
with the NuScenes data structure.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dataset_nuscenes import NuScenesDataset, create_nuscenes_transforms, collate_fn
except ImportError as e:
    print(f"Error importing dataset_nuscenes: {e}")
    print("Make sure dataset_nuscenes.py is in the same directory as this test script.")
    sys.exit(1)


def test_dataset_basic(root_dir: str):
    """Test basic dataset functionality."""
    print("=" * 50)
    print("Testing NuScenes Dataset - Basic Functionality")
    print("=" * 50)
    
    try:
        # Create dataset with minimal configuration
        dataset = NuScenesDataset(
            root_dir=root_dir,
            split='train',
            camera_types=['CAM_FRONT'],
            transform=None,  # No transforms for basic test
            max_samples=5    # Limit samples for testing
        )
        
        print(f"✓ Dataset created successfully")
        print(f"✓ Number of samples: {len(dataset)}")
        
        if len(dataset) == 0:
            print("⚠ Warning: No samples found in dataset")
            return False
        
        # Test loading first sample
        image, target = dataset[0]
        print(f"✓ First sample loaded successfully")
        print(f"  - Image type: {type(image)}")
        print(f"  - Image size: {image.size if hasattr(image, 'size') else 'N/A'}")
        print(f"  - Target keys: {list(target.keys())}")
        print(f"  - Number of boxes: {len(target['boxes'])}")
        print(f"  - Labels: {target['labels'].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in basic test: {e}")
        return False


def test_dataset_with_transforms(root_dir: str):
    """Test dataset with transforms."""
    print("\n" + "=" * 50)
    print("Testing NuScenes Dataset - With Transforms")
    print("=" * 50)
    
    try:
        # Create transforms
        transform = create_nuscenes_transforms(train=True)
        
        # Create dataset with transforms
        dataset = NuScenesDataset(
            root_dir=root_dir,
            split='train',
            camera_types=['CAM_FRONT'],
            transform=transform,
            max_samples=3
        )
        
        print(f"✓ Dataset with transforms created successfully")
        
        if len(dataset) == 0:
            print("⚠ Warning: No samples found in dataset")
            return False
        
        # Test loading sample with transforms
        image, target = dataset[0]
        print(f"✓ Sample with transforms loaded successfully")
        print(f"  - Image type: {type(image)}")
        print(f"  - Image shape: {image.shape if hasattr(image, 'shape') else 'N/A'}")
        print(f"  - Image dtype: {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in transform test: {e}")
        return False


def test_dataloader(root_dir: str):
    """Test dataset with PyTorch DataLoader."""
    print("\n" + "=" * 50)
    print("Testing NuScenes Dataset - DataLoader")
    print("=" * 50)
    
    try:
        # Create dataset
        transform = create_nuscenes_transforms(train=False)
        dataset = NuScenesDataset(
            root_dir=root_dir,
            split='train',
            camera_types=['CAM_FRONT'],
            transform=transform,
            max_samples=4
        )
        
        if len(dataset) == 0:
            print("⚠ Warning: No samples found in dataset")
            return False
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
        
        print(f"✓ DataLoader created successfully")
        
        # Test loading a batch
        for batch_idx, (images, targets) in enumerate(dataloader):
            print(f"✓ Batch {batch_idx} loaded successfully")
            print(f"  - Batch size: {len(images)}")
            print(f"  - Images shape: {images.shape}")
            print(f"  - Number of targets: {len(targets)}")
            
            for i, target in enumerate(targets):
                print(f"  - Target {i}: {len(target['boxes'])} boxes, labels: {target['labels'].tolist()}")
            
            # Only test first batch
            break
        
        return True
        
    except Exception as e:
        print(f"✗ Error in DataLoader test: {e}")
        return False


def test_multiple_cameras(root_dir: str):
    """Test dataset with multiple camera types."""
    print("\n" + "=" * 50)
    print("Testing NuScenes Dataset - Multiple Cameras")
    print("=" * 50)
    
    try:
        # Test with multiple camera types
        camera_types = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        
        dataset = NuScenesDataset(
            root_dir=root_dir,
            split='train',
            camera_types=camera_types,
            transform=None,
            max_samples=3
        )
        
        print(f"✓ Dataset with multiple cameras created successfully")
        print(f"✓ Camera types: {camera_types}")
        print(f"✓ Number of samples: {len(dataset)}")
        
        if len(dataset) > 0:
            image, target = dataset[0]
            print(f"✓ Sample loaded successfully with multiple camera support")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in multiple camera test: {e}")
        return False


def print_dataset_info(root_dir: str):
    """Print information about the dataset structure."""
    print("\n" + "=" * 50)
    print("Dataset Structure Information")
    print("=" * 50)
    
    # Check directory structure
    samples_dir = os.path.join(root_dir, 'samples')
    annotation_dir = os.path.join(root_dir, 'v1.0-trainval')
    
    print(f"Root directory: {root_dir}")
    print(f"Samples directory exists: {os.path.exists(samples_dir)}")
    print(f"Annotation directory exists: {os.path.exists(annotation_dir)}")
    
    if os.path.exists(samples_dir):
        camera_dirs = [d for d in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, d))]
        print(f"Available camera types: {camera_dirs}")
    
    if os.path.exists(annotation_dir):
        annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
        print(f"Available annotation files: {annotation_files}")


def main():
    """Main test function."""
    if len(sys.argv) != 2:
        print("Usage: python test_nuscenes_dataset.py <nuscenes_root_dir>")
        print("Example: python test_nuscenes_dataset.py /path/to/nuscenes/v1.0-trainval")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    
    if not os.path.exists(root_dir):
        print(f"Error: Root directory does not exist: {root_dir}")
        sys.exit(1)
    
    print("NuScenes Dataset Test Suite")
    print(f"Testing with root directory: {root_dir}")
    
    # Print dataset information
    print_dataset_info(root_dir)
    
    # Run tests
    tests = [
        ("Basic Functionality", test_dataset_basic),
        ("With Transforms", test_dataset_with_transforms),
        ("DataLoader", test_dataloader),
        ("Multiple Cameras", test_multiple_cameras),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func(root_dir)
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The NuScenes dataset is working correctly.")
    else:
        print("⚠ Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main()