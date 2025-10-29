#!/usr/bin/env python3
"""
Improved LiDAR to Camera Projection Implementation
Based on the reference from: https://github.com/lkk688/WaymoObjectDetection

Key improvements:
1. Proper projection pipeline following reference implementation
2. Image boundary filtering (only points within image width/height)
3. Correct depth extraction from camera coordinates
4. No arbitrary scaling - uses proper calibration matrices
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
import os

def project_velo_to_cameraid(pts_3d_velo: np.ndarray, T_vl: np.ndarray, T_cv: np.ndarray) -> np.ndarray:
    """
    Project velodyne points to camera coordinate system.
    
    Args:
        pts_3d_velo: LiDAR points already in vehicle coordinates (N, 3)
        T_vl: Vehicle to LiDAR transformation matrix (4, 4) - not used since points are already in vehicle coords
        T_cv: Camera to vehicle transformation matrix (4, 4)
    
    Returns:
        pts_3d_cam: Points in camera coordinate system (N, 3)
    """
    # Convert to homogeneous coordinates
    pts_3d_velo_hom = np.hstack([pts_3d_velo, np.ones((pts_3d_velo.shape[0], 1))])
    
    # Transform from vehicle to camera coordinates
    # Since points are already in vehicle coordinates, we only need T_vc = inv(T_cv)
    T_vc = np.linalg.inv(T_cv)
    
    # Apply transformation: vehicle -> camera
    pts_3d_cam_hom = (T_vc @ pts_3d_velo_hom.T).T
    return pts_3d_cam_hom[:, :3]

def project_cam3d_to_image(pts_3d_cam: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D camera coordinates to 2D image coordinates.
    
    Args:
        pts_3d_cam: Points in camera coordinate system (N, 3)
        K: Camera intrinsic matrix (3, 3)
    
    Returns:
        pts_2d: 2D image coordinates (N, 2)
        depths: Depth values (N,)
    """
    # Convert to homogeneous coordinates
    pts_3d_cam_hom = np.hstack([pts_3d_cam, np.ones((pts_3d_cam.shape[0], 1))])
    
    # Project to image coordinates
    pts_2d_hom = (K @ pts_3d_cam[:, :3].T).T  # (N, 3)
    
    # Normalize by depth (Z coordinate)
    depths = pts_2d_hom[:, 2]
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]
    
    return pts_2d, depths

def project_velo_to_image(pts_3d_velo: np.ndarray, 
                         T_vl: np.ndarray, 
                         T_cv: np.ndarray, 
                         K: np.ndarray,
                         image_shape: Tuple[int, int],
                         coordinate_system: str = 'v2') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete pipeline to project LiDAR points to image coordinates with proper filtering.
    
    Args:
        pts_3d_velo: LiDAR points in vehicle coordinates (N, 3 or more)
        T_vl: Vehicle to LiDAR transformation matrix (4, 4)
        T_cv: Camera to vehicle transformation matrix (4, 4)
        K: Camera intrinsic matrix (3, 3)
        image_shape: Image dimensions (height, width)
        coordinate_system: 'v1' or 'v2' for coordinate system handling
    
    Returns:
        pts_2d_valid: Valid 2D image coordinates (M, 2)
        depths_valid: Corresponding depth values (M,)
        valid_indices: Original indices of valid points (M,)
    """
    # Handle different input formats - take only X, Y, Z coordinates
    if pts_3d_velo.shape[1] >= 3:
        points_3d = pts_3d_velo[:, :3]
    else:
        raise ValueError(f"LiDAR points must have at least 3 dimensions, got {pts_3d_velo.shape[1]}")
    
    # Apply coordinate system specific transformations
    if coordinate_system == 'v1':
        # For v1, apply Z-axis flip to handle coordinate system difference
        points_3d = points_3d.copy()
        points_3d[:, 2] = -points_3d[:, 2]
    
    # Step 1: Project velodyne to camera coordinate system
    pts_3d_cam = project_velo_to_cameraid(points_3d, T_vl, T_cv)
    
    # Step 2: Filter points in front of camera (positive Z in camera coordinates)
    valid_z_mask = pts_3d_cam[:, 2] > 0.1  # Small threshold to avoid division by zero
    
    if not np.any(valid_z_mask):
        return np.array([]), np.array([]), np.array([])
    
    pts_3d_cam_valid = pts_3d_cam[valid_z_mask]
    valid_z_indices = np.where(valid_z_mask)[0]
    
    # Step 3: Project 3D camera coordinates to 2D image coordinates
    pts_2d, depths = project_cam3d_to_image(pts_3d_cam_valid, K)
    
    # Step 4: Filter points within image boundaries (key step from reference)
    height, width = image_shape
    valid_bounds_mask = ((pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < width) &
                        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < height))
    
    # Final valid points
    pts_2d_valid = pts_2d[valid_bounds_mask]
    depths_valid = depths[valid_bounds_mask]  # This is imgfov_pc_rect[i, 2] from reference
    valid_indices = valid_z_indices[valid_bounds_mask]
    
    return pts_2d_valid, depths_valid, valid_indices

def visualize_improved_projection(lidar_points: np.ndarray,
                                camera_image: np.ndarray,
                                T_vl: np.ndarray,
                                T_cv: np.ndarray,
                                K: np.ndarray,
                                coordinate_system: str = 'v2',
                                max_distance: float = 80.0,
                                save_path: Optional[str] = None,
                                title: str = "Improved LiDAR Projection") -> plt.Figure:
    """
    Visualize the improved LiDAR projection on camera image.
    
    Args:
        lidar_points: LiDAR points (N, 3+)
        camera_image: Camera image array
        T_vl: Vehicle to LiDAR transformation matrix
        T_cv: Camera to vehicle transformation matrix  
        K: Camera intrinsic matrix
        coordinate_system: 'v1' or 'v2'
        max_distance: Maximum distance for visualization
        save_path: Path to save the visualization
        title: Title for the plot
    
    Returns:
        matplotlib Figure object
    """
    # Filter points by distance
    distances = np.linalg.norm(lidar_points[:, :3], axis=1)
    distance_mask = distances <= max_distance
    filtered_points = lidar_points[distance_mask]
    filtered_distances = distances[distance_mask]
    
    # Project to image
    image_shape = camera_image.shape[:2]  # (height, width)
    pts_2d, depths, valid_indices = project_velo_to_image(
        filtered_points, T_vl, T_cv, K, image_shape, coordinate_system
    )
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Display camera image
    ax.imshow(camera_image)
    
    # Overlay projected LiDAR points
    if len(pts_2d) > 0:
        # Use depth for coloring (this is the imgfov_pc_rect[i, 2] from reference)
        scatter = ax.scatter(pts_2d[:, 0], pts_2d[:, 1], 
                           c=depths, s=2.0, alpha=0.8, 
                           cmap='viridis', vmin=0, vmax=max_distance,
                           edgecolors='none')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Depth (m)', rotation=270, labelpad=15)
        
        print(f"Projection Results:")
        print(f"  Total LiDAR points: {len(lidar_points)}")
        print(f"  Points within {max_distance}m: {len(filtered_points)}")
        print(f"  Points projected to image: {len(pts_2d)}")
        print(f"  Image bounds: {image_shape[1]}x{image_shape[0]} (WxH)")
        print(f"  Projected X range: [{pts_2d[:, 0].min():.1f}, {pts_2d[:, 0].max():.1f}]")
        print(f"  Projected Y range: [{pts_2d[:, 1].min():.1f}, {pts_2d[:, 1].max():.1f}]")
        print(f"  Depth range: [{depths.min():.2f}, {depths.max():.2f}]m")
    else:
        print(f"No LiDAR points successfully projected to image bounds")
    
    # Set title and labels
    ax.set_title(f"{title} - {coordinate_system.upper()} Data", fontsize=14, fontweight='bold')
    ax.set_xlabel('Image X (pixels)')
    ax.set_ylabel('Image Y (pixels)')
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    return fig

def demo_improved_projection():
    """
    Demo function to test the improved projection implementation.
    """
    print("Improved LiDAR Projection Demo")
    print("=" * 50)
    
    # This is a placeholder demo - in practice, you would load actual data
    print("This implementation provides:")
    print("1. Proper projection pipeline following WaymoObjectDetection reference")
    print("2. Image boundary filtering (only points within image width/height)")
    print("3. Correct depth extraction from camera coordinates")
    print("4. No arbitrary scaling - uses proper calibration matrices")
    print("\nTo use with actual data:")
    print("  pts_2d, depths, indices = project_velo_to_image(lidar_points, T_vl, T_cv, K, image_shape)")
    print("  fig = visualize_improved_projection(lidar_points, camera_image, T_vl, T_cv, K)")

if __name__ == "__main__":
    demo_improved_projection()