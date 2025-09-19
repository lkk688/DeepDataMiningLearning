#!/usr/bin/env python3
"""
Test script for separated 2D and 3D bounding box visualization.
This script tests the improved visualization functions with side-by-side display.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Add the current directory to Python path to import the functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from unzipnuscenes.py
from unzipnuscenes import (
    quaternion_to_rotation_matrix,
    get_3d_box_corners,
    project_3d_box_to_2d,
    draw_3d_box_2d,
    draw_2d_bbox,
    draw_2d_bbox_from_3d,
    get_2d_bbox_from_3d_projection
)

def test_separated_visualization():
    """Test the separated 2D and 3D visualization functionality."""
    print("Testing separated 2D and 3D visualization...")
    
    # Create a test image (simulated camera view)
    img_width, img_height = 1600, 900
    test_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 128  # Gray background
    
    # Add some visual elements to make it look like a street scene
    # Road
    test_image[600:900, :, :] = [80, 80, 80]  # Dark gray road
    # Lane markings
    test_image[750:770, 200:1400:100, :] = [255, 255, 255]  # White lane markings
    
    # Convert to PIL Image
    img = Image.fromarray(test_image)
    
    # Create side-by-side subplots
    fig, (ax_3d, ax_2d) = plt.subplots(1, 2, figsize=(24, 8))
    
    # Display image on both subplots
    ax_3d.imshow(img)
    ax_3d.set_title('Test Scene - 3D Bounding Boxes')
    
    ax_2d.imshow(img)
    ax_2d.set_title('Test Scene - 2D Bounding Boxes')
    
    # Test parameters (simulated NuScenes data)
    # Camera intrinsic matrix (typical values)
    camera_intrinsic = np.array([
        [1266.417203046554, 0.0, 816.2670197447984],
        [0.0, 1266.417203046554, 491.50706579294757],
        [0.0, 0.0, 1.0]
    ])
    
    # Test 3D bounding boxes (in global coordinates)
    test_boxes = [
        {
            'center': np.array([10.0, 5.0, 1.0]),  # Car in front
            'size': np.array([1.8, 4.5, 1.5]),    # width, length, height
            'rotation': np.array([1.0, 0.0, 0.0, 0.1]),  # slight rotation
            'category': 'car'
        },
        {
            'center': np.array([15.0, -2.0, 1.2]),  # Car to the right
            'size': np.array([1.9, 4.8, 1.6]),
            'rotation': np.array([0.9, 0.0, 0.0, 0.3]),  # more rotation
            'category': 'car'
        },
        {
            'center': np.array([8.0, 8.0, 2.5]),   # Truck further away
            'size': np.array([2.5, 8.0, 3.0]),
            'rotation': np.array([1.0, 0.0, 0.0, 0.0]),  # no rotation
            'category': 'truck'
        }
    ]
    
    # Simulated ego pose and camera pose
    ego_translation = np.array([0.0, 0.0, 0.0])
    ego_rotation = np.array([1.0, 0.0, 0.0, 0.0])  # no rotation
    
    # Camera mounted on front of vehicle
    cam_translation = np.array([1.5, 0.0, 1.8])  # forward, center, up
    cam_rotation = np.array([1.0, 0.0, 0.0, 0.0])  # no rotation
    
    boxes_drawn_3d = 0
    boxes_drawn_2d = 0
    
    # Process each test box
    for i, box in enumerate(test_boxes):
        try:
            # Project 3D bounding box to 2D
            corners_2d = project_3d_box_to_2d(
                box['center'], box['size'], box['rotation'],
                cam_translation, cam_rotation, camera_intrinsic,
                ego_translation, ego_rotation, debug=(i == 0)
            )
            
            if corners_2d is not None:
                category_name = box['category']
                
                # Draw 3D bounding box (wireframe) on left subplot
                draw_3d_box_2d(ax_3d, corners_2d, color='red', linewidth=2, 
                             img_width=img_width, img_height=img_height)
                boxes_drawn_3d += 1
                
                # Calculate and draw 2D bounding box from 3D projection on right subplot
                bbox_2d_from_3d = get_2d_bbox_from_3d_projection(corners_2d)
                if bbox_2d_from_3d is not None:
                    # Draw the projected 2D bbox (blue dashed)
                    success = draw_2d_bbox_from_3d(ax_2d, corners_2d, f"{category_name} (3D→2D)", 
                                                 color='blue', linewidth=2,
                                                 img_width=img_width, img_height=img_height)
                    if success:
                        boxes_drawn_2d += 1
                
                    # Draw a simulated 2D detection box (solid green)
                    x_min, y_min, x_max, y_max = bbox_2d_from_3d
                    # Create a simulated 2D detection box (slightly smaller)
                    margin = min(20, (x_max - x_min) * 0.1, (y_max - y_min) * 0.1)
                    simulated_2d_bbox = [x_min + margin, y_min + margin, 
                                       x_max - margin, y_max - margin]
                    
                    success = draw_2d_bbox(ax_2d, simulated_2d_bbox, f"{category_name} (2D Det)", 
                                         color='green', linewidth=2,
                                         img_width=img_width, img_height=img_height)
                    if success:
                        boxes_drawn_2d += 1
                
        except Exception as e:
            print(f"Warning: Failed to project test box {i}: {e}")
    
    # Add annotation info for 3D subplot
    info_text_3d = f'3D Test Boxes: {len(test_boxes)} (Drawn: {boxes_drawn_3d})'
    ax_3d.text(10, 30, info_text_3d, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
           fontsize=12, fontweight='bold')
    
    # Add legend for 3D subplot
    ax_3d.text(10, 70, "3D Wireframe", fontsize=10, 
              bbox=dict(boxstyle="round,pad=0.2", facecolor="red", alpha=0.8))
    ax_3d.plot([10, 40], [85, 85], color='red', linestyle='-', linewidth=2)
    
    # Add annotation info for 2D subplot
    info_text_2d = f'2D Test Boxes: {len(test_boxes)} (Drawn: {boxes_drawn_2d})'
    ax_2d.text(10, 30, info_text_2d, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
           fontsize=12, fontweight='bold')
    
    # Add legend for 2D subplot
    legend_y_start = 70
    legend_items_2d = [
        ("3D→2D Bbox", "blue", "--"),
        ("2D Detection", "green", "-")
    ]
    
    for i, (label, color, linestyle) in enumerate(legend_items_2d):
        y_pos = legend_y_start + i * 25
        # Draw sample line
        ax_2d.plot([10, 40], [y_pos, y_pos], color=color, linestyle=linestyle, linewidth=2)
        # Add text
        ax_2d.text(45, y_pos, label, fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    ax_3d.axis('off')
    ax_2d.axis('off')
    
    # Save the test visualization
    output_path = 'test_separated_visualization_result.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"✓ Separated visualization test completed successfully!")
    print(f"✓ 3D boxes drawn: {boxes_drawn_3d}")
    print(f"✓ 2D boxes drawn: {boxes_drawn_2d}")
    print(f"✓ Test result saved to: {output_path}")
    
    return True

def test_3d_box_corners():
    """Test the improved 3D box corner calculation."""
    print("\nTesting 3D box corner calculation...")
    
    # Test case 1: Simple box with no rotation
    center = np.array([0.0, 0.0, 0.0])
    size = np.array([2.0, 4.0, 1.5])  # width, length, height
    rotation = np.array([1.0, 0.0, 0.0, 0.0])  # no rotation
    
    corners = get_3d_box_corners(center, size, rotation)
    
    print(f"Test box: center={center}, size={size}")
    print("Expected corners (no rotation):")
    expected = [
        [-1.0, -2.0, -0.75],  # bottom-back-left
        [+1.0, -2.0, -0.75],  # bottom-back-right
        [+1.0, +2.0, -0.75],  # bottom-front-right
        [-1.0, +2.0, -0.75],  # bottom-front-left
        [-1.0, -2.0, +0.75],  # top-back-left
        [+1.0, -2.0, +0.75],  # top-back-right
        [+1.0, +2.0, +0.75],  # top-front-right
        [-1.0, +2.0, +0.75]   # top-front-left
    ]
    
    print("Actual corners:")
    for i, corner in enumerate(corners):
        print(f"  Corner {i}: {corner}")
        expected_corner = np.array(expected[i])
        if np.allclose(corner, expected_corner, atol=1e-10):
            print(f"    ✓ Matches expected: {expected_corner}")
        else:
            print(f"    ✗ Expected: {expected_corner}, got: {corner}")
    
    print("✓ 3D box corner calculation test completed!")
    return True

def test_quaternion_rotation():
    """Test quaternion to rotation matrix conversion."""
    print("\nTesting quaternion rotation...")
    
    # Test case: 90-degree rotation around Z-axis
    # Quaternion for 90-degree rotation around Z: [cos(45°), 0, 0, sin(45°)]
    angle = np.pi / 2  # 90 degrees
    quat = np.array([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])
    
    rotation_matrix = quaternion_to_rotation_matrix(quat)
    
    print(f"90-degree Z rotation quaternion: {quat}")
    print("Rotation matrix:")
    print(rotation_matrix)
    
    # Expected rotation matrix for 90-degree Z rotation
    expected = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])
    
    if np.allclose(rotation_matrix, expected, atol=1e-10):
        print("✓ Rotation matrix matches expected result!")
    else:
        print("✗ Rotation matrix does not match expected result")
        print("Expected:")
        print(expected)
    
    return True

if __name__ == "__main__":
    print("Running separated visualization tests...")
    
    # Run all tests
    test_quaternion_rotation()
    test_3d_box_corners()
    test_separated_visualization()
    
    print("\n" + "="*50)
    print("All separated visualization tests completed!")
    print("="*50)