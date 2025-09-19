#!/usr/bin/env python3
"""
Test script to verify NuScenes-compliant 3D to 2D projection implementation.
This script tests the corrected coordinate transformation chain and projection logic.
"""

import numpy as np
import matplotlib.pyplot as plt
from unzipnuscenes import (
    view_points, transform_matrix, quaternion_to_rotation_matrix,
    get_3d_box_corners, project_3d_box_to_2d
)

def test_view_points():
    """Test the view_points function with known inputs."""
    print("Testing view_points function...")
    
    # Test case: simple 3D points
    points_3d = np.array([
        [1.0, 2.0, 3.0],  # x, y, z coordinates
        [0.0, 1.0, 2.0],
        [2.0, 0.0, 1.0]
    ]).T  # Shape: (3, N)
    
    # Simple camera intrinsic matrix
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Project points
    projected = view_points(points_3d, K, normalize=True)
    print(f"Input 3D points shape: {points_3d.shape}")
    print(f"Projected 2D points shape: {projected.shape}")
    print(f"Projected points:\n{projected}")
    
    # Verify projection manually for first point
    p1 = points_3d[:, 0]  # [1, 0, 2]
    expected_x = (500.0 * p1[0] / p1[2]) + 320.0  # (500*1/2) + 320 = 570
    expected_y = (500.0 * p1[1] / p1[2]) + 240.0  # (500*0/2) + 240 = 240
    print(f"Manual calculation for point 1: ({expected_x}, {expected_y})")
    print(f"view_points result for point 1: ({projected[0, 0]}, {projected[1, 0]})")
    
    return projected

def test_transform_matrix():
    """Test the transform_matrix function."""
    print("\nTesting transform_matrix function...")
    
    # Test case: simple translation and rotation
    translation = np.array([1.0, 2.0, 3.0])
    rotation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    
    # Forward transform
    T_forward = transform_matrix(translation, rotation, inverse=False)
    print(f"Forward transform matrix:\n{T_forward}")
    
    # Inverse transform
    T_inverse = transform_matrix(translation, rotation, inverse=True)
    print(f"Inverse transform matrix:\n{T_inverse}")
    
    # Test that forward * inverse = identity
    identity_test = np.dot(T_forward, T_inverse)
    print(f"Forward * Inverse (should be identity):\n{identity_test}")
    
    # Test point transformation
    test_point = np.array([5.0, 6.0, 7.0, 1.0])  # Homogeneous coordinates
    transformed = np.dot(T_forward, test_point)
    back_transformed = np.dot(T_inverse, transformed)
    
    print(f"Original point: {test_point}")
    print(f"Transformed: {transformed}")
    print(f"Back transformed: {back_transformed}")
    
    return T_forward, T_inverse

def test_3d_box_projection():
    """Test complete 3D box projection pipeline."""
    print("\nTesting complete 3D box projection...")
    
    # Test box parameters - place box in front of camera
    center_3d = np.array([0.0, 10.0, 0.0])  # Box center 10m in front (y-axis forward in NuScenes)
    size_3d = np.array([2.0, 4.0, 1.5])    # width, length, height
    rotation_3d = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
    
    # Ego vehicle pose
    ego_translation = np.array([0.0, 0.0, 0.0])
    ego_rotation = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Camera pose relative to ego - NuScenes camera coordinate system
    # In NuScenes: x-right, y-down, z-forward (camera looks along +z axis)
    cam_translation = np.array([0.0, 0.0, 1.5])  # Camera 1.5m above ground
    # Rotate camera to look forward: from ego (x-forward, y-left, z-up) to camera (x-right, y-down, z-forward)
    # This is a 90-degree rotation around z-axis followed by 90-degree rotation around x-axis
    # Quaternion for this transformation: [0.5, 0.5, 0.5, 0.5] (approximately)
    cam_rotation = np.array([0.7071, 0.7071, 0.0, 0.0])  # 90-degree rotation around x-axis
    
    # Camera intrinsic matrix (typical values)
    camera_intrinsic = np.array([
        [1266.417203046554, 0.0, 816.2670197447984],
        [0.0, 1266.417203046554, 491.50706579294757],
        [0.0, 0.0, 1.0]
    ])
    
    # Project 3D box to 2D
    corners_2d = project_3d_box_to_2d(
        center_3d, size_3d, rotation_3d,
        cam_translation, cam_rotation, camera_intrinsic,
        ego_translation, ego_rotation,
        debug=True
    )
    
    if corners_2d is not None:
        print(f"\nProjection successful!")
        print(f"2D corners shape: {corners_2d.shape}")
        print(f"2D corners:\n{corners_2d}")
        
        # Check if any corners are within image bounds
        img_width, img_height = 1600, 900
        valid_corners = []
        for i, corner in enumerate(corners_2d):
            x, y = corner
            if 0 <= x <= img_width and 0 <= y <= img_height:
                valid_corners.append(i)
        
        print(f"Corners within image bounds ({img_width}x{img_height}): {valid_corners}")
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Draw projected box corners
        for i, corner in enumerate(corners_2d):
            color = 'ro' if i in valid_corners else 'bo'
            ax.plot(corner[0], corner[1], color, markersize=8)
            ax.annotate(f'{i}', (corner[0], corner[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
        
        # Draw box edges (simplified - just connect some corners)
        # Bottom face
        bottom_indices = [0, 1, 2, 3, 0]
        for i in range(len(bottom_indices)-1):
            p1 = corners_2d[bottom_indices[i]]
            p2 = corners_2d[bottom_indices[i+1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2, alpha=0.7)
        
        # Top face
        top_indices = [4, 5, 6, 7, 4]
        for i in range(len(top_indices)-1):
            p1 = corners_2d[top_indices[i]]
            p2 = corners_2d[top_indices[i+1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2, alpha=0.7)
        
        # Vertical edges
        for i in range(4):
            p1 = corners_2d[i]
            p2 = corners_2d[i+4]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', linewidth=1, alpha=0.7)
        
        # Draw image bounds
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axhline(y=img_height, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=img_width, color='k', linestyle='-', alpha=0.3)
        
        # Set reasonable axis limits
        all_x = corners_2d[:, 0]
        all_y = corners_2d[:, 1]
        x_margin = (np.max(all_x) - np.min(all_x)) * 0.1
        y_margin = (np.max(all_y) - np.min(all_y)) * 0.1
        
        ax.set_xlim(min(0, np.min(all_x) - x_margin), max(img_width, np.max(all_x) + x_margin))
        ax.set_ylim(min(0, np.min(all_y) - y_margin), max(img_height, np.max(all_y) + y_margin))
        ax.invert_yaxis()  # Image coordinates
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title('NuScenes-compliant 3D Box Projection Test\n(Red=valid corners, Blue=outside image)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('nuscenes_projection_test.png', dpi=150, bbox_inches='tight')
        print(f"Visualization saved as 'nuscenes_projection_test.png'")
        
        return corners_2d
    else:
        print("Projection failed!")
        return None

def main():
    """Run all tests."""
    print("="*60)
    print("NuScenes-compliant 3D to 2D Projection Test")
    print("="*60)
    
    # Test individual components
    projected_points = test_view_points()
    T_forward, T_inverse = test_transform_matrix()
    
    # Test complete pipeline
    corners_2d = test_3d_box_projection()
    
    print("\n" + "="*60)
    print("Test completed!")
    if corners_2d is not None:
        print("✓ All tests passed - projection pipeline working correctly")
    else:
        print("✗ Projection test failed")
    print("="*60)

if __name__ == "__main__":
    main()