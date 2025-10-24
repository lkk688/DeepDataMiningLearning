#!/usr/bin/env python3
"""
Test script for Open3D 3D LiDAR visualization functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    # Check OpenCV
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__} is available")
    except ImportError:
        missing_deps.append("opencv-python")
        print("❌ OpenCV (cv2) is not installed")
    
    # Check PIL/Pillow
    try:
        from PIL import Image
        print(f"✅ Pillow is available")
    except ImportError:
        missing_deps.append("Pillow")
        print("❌ Pillow (PIL) is not installed")
    
    # Check Open3D
    try:
        import open3d as o3d
        print(f"✅ Open3D {o3d.__version__} is available")
    except ImportError:
        missing_deps.append("open3d")
        print("❌ Open3D is not installed")
    
    # Check matplotlib
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__} is available")
    except ImportError:
        missing_deps.append("matplotlib")
        print("❌ Matplotlib is not installed")
    
    # Check numpy
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} is available")
    except ImportError:
        missing_deps.append("numpy")
        print("❌ NumPy is not installed")
    
    return missing_deps

def test_open3d_visualization():
    """Test the Open3D 3D visualization functionality."""
    
    print("🧪 Testing Open3D 3D LiDAR visualization...")
    print("\n📋 Checking dependencies...")
    
    # Check dependencies first
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    print("\n✅ All dependencies are available!")
    
    # Dataset configuration
    nuscenes_root = "/mnt/e/Shared/Dataset/NuScenes/v1.0-trainval"
    sample_idx = 0
    
    print(f"\n📁 Dataset root: {nuscenes_root}")
    print(f"📊 Sample index: {sample_idx}")
    
    # Check if dataset path exists
    if not os.path.exists(nuscenes_root):
        print(f"❌ Dataset path does not exist: {nuscenes_root}")
        return False
    
    # Create output directory for visualizations
    output_dir = "/tmp/nuscenes_visualization_test"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Import the visualization function only after dependency check
        from nuscenes import visualize_sample_with_boxes
        
        # Test the visualization
        print("\n🚀 Starting visualization...")
        visualize_sample_with_boxes(nuscenes_root, sample_idx, output_dir)
        print("\n✅ Open3D visualization test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        if "cv2" in str(e):
            print("💡 Hint: Install OpenCV with: pip install opencv-python")
        elif "PIL" in str(e) or "Pillow" in str(e):
            print("💡 Hint: Install Pillow with: pip install Pillow")
        elif "open3d" in str(e).lower():
            print("💡 Hint: Install Open3D with: pip install open3d")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_open3d_visualization()