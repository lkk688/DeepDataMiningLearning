#!/usr/bin/env python3
"""
Test script for enhanced BEV visualization with improved bounding boxes
"""

import numpy as np
import matplotlib.pyplot as plt
from unzipnuscenes import draw_bev_box_2d, quaternion_to_rotation_matrix

def test_enhanced_bev_boxes():
    """Test the enhanced BEV bounding box visualization"""
    print("Testing enhanced BEV bounding box visualization...")
    
    # Create test figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Set BEV range
    bev_range = 50
    ax.set_xlim([-bev_range, bev_range])
    ax.set_ylim([-bev_range, bev_range])
    ax.set_aspect('equal')
    
    # Test data - simulate different vehicle types
    test_boxes = [
        {
            'center': np.array([10.0, 5.0, 1.5]),
            'size': np.array([1.8, 4.5, 1.6]),  # car dimensions
            'rotation': np.array([1.0, 0.0, 0.0, 0.0]),  # no rotation
            'category': 'car'
        },
        {
            'center': np.array([-15.0, -8.0, 2.0]),
            'size': np.array([2.5, 8.0, 3.0]),  # truck dimensions
            'rotation': np.array([0.9, 0.0, 0.0, 0.3]),  # some rotation
            'category': 'truck'
        },
        {
            'center': np.array([20.0, -15.0, 1.8]),
            'size': np.array([2.2, 12.0, 3.5]),  # bus dimensions
            'rotation': np.array([0.8, 0.0, 0.0, 0.6]),  # more rotation
            'category': 'bus'
        },
        {
            'center': np.array([-5.0, 25.0, 0.5]),
            'size': np.array([0.6, 1.8, 1.8]),  # pedestrian dimensions
            'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
            'category': 'pedestrian'
        },
        {
            'center': np.array([8.0, -20.0, 1.0]),
            'size': np.array([0.8, 2.2, 1.4]),  # motorcycle dimensions
            'rotation': np.array([0.95, 0.0, 0.0, 0.31]),
            'category': 'motorcycle'
        }
    ]
    
    # Draw test boxes
    boxes_drawn = 0
    category_counts = {}
    
    for box in test_boxes:
        center = box['center']
        size = box['size']
        rotation = box['rotation']
        category = box['category']
        
        # Count categories
        category_counts[category] = category_counts.get(category, 0) + 1
        
        # Check if box is within BEV range
        if abs(center[0]) <= bev_range and abs(center[1]) <= bev_range:
            # Draw enhanced BEV box
            draw_bev_box_2d(ax, center, size, rotation, category)
            boxes_drawn += 1
    
    # Add ego vehicle marker
    ax.plot(0, 0, 'ro', markersize=10, label='Ego Vehicle', markeredgecolor='white', markeredgewidth=2)
    ax.arrow(0, 0, 0, 4, head_width=1.5, head_length=1.5, fc='red', ec='white', linewidth=2)
    
    # Add range circles for reference
    for radius in [10, 20, 30, 40]:
        if radius <= bev_range:
            circle = plt.Circle((0, 0), radius, fill=False, color='gray', alpha=0.3, linestyle='--', linewidth=1)
            ax.add_patch(circle)
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    
    # Create detailed title with category statistics
    category_stats = ', '.join([f"{cat}: {count}" for cat, count in sorted(category_counts.items())])
    title = f'Enhanced BEV Visualization Test - {boxes_drawn} Objects'
    if category_stats:
        title += f'\n({category_stats})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # Save test result
    output_path = '/home/lkk/Developer/DeepDataMiningLearning/scripts/bev_enhanced_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced BEV test visualization saved to: {output_path}")
    print(f"Successfully drew {boxes_drawn} enhanced bounding boxes")
    print(f"Categories tested: {list(category_counts.keys())}")
    
    return output_path

if __name__ == "__main__":
    test_enhanced_bev_boxes()
    print("Enhanced BEV visualization test completed!")