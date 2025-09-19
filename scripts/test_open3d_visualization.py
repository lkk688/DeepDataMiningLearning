#!/usr/bin/env python3
"""
Test script for Open3D 3D LiDAR visualization functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unzipnuscenes import visualize_sample_with_boxes

def test_open3d_visualization():
    """Test the Open3D 3D visualization functionality."""
    
    # Dataset configuration
    nuscenes_root = "/mnt/e/Shared/Dataset/NuScenes/v1.0-trainval"
    sample_idx = 0
    
    print("🧪 Testing Open3D 3D LiDAR visualization...")
    print(f"Dataset root: {nuscenes_root}")
    print(f"Sample index: {sample_idx}")
    
    try:
        # Test the visualization
        visualize_sample_with_boxes(nuscenes_root, sample_idx)
        print("\n✅ Open3D visualization test completed successfully!")
        
    except ImportError as e:
        if "open3d" in str(e).lower():
            print(f"❌ Open3D import error: {e}")
            print("Please install Open3D with: pip install open3d")
        else:
            print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_open3d_visualization()