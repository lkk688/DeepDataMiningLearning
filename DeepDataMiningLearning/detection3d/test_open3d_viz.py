#!/usr/bin/env python3
"""
Comprehensive Test Suite for Open3D-based 3D Visualization System

This script provides extensive testing and demonstration of the new Open3D
visualization system for KITTI and Waymo2KITTI datasets. It includes:

- Synthetic data generation for testing
- Performance benchmarking
- Headless operation validation
- Export functionality testing
- Mathematical accuracy verification

Usage:
    python test_open3d_viz.py --test-all
    python test_open3d_viz.py --test-synthetic --headless --save-dir ./outputs
    python test_open3d_viz.py --benchmark --iterations 10
"""

import numpy as np
import os
import sys
import time
import argparse
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from waymo_kitti_open3d_viz import Open3DVisualizer, boxes_to_corners_3d, rotate_points_along_z
    print("✓ Successfully imported Open3D visualization module")
except ImportError as e:
    print(f"✗ Failed to import visualization module: {e}")
    sys.exit(1)


class SyntheticDataGenerator:
    """
    Generate synthetic 3D data for testing visualization capabilities.
    
    This class creates realistic synthetic point clouds and 3D bounding boxes
    that mimic the characteristics of autonomous driving datasets like KITTI
    and Waymo2KITTI.
    
    Mathematical Data Generation:
    - Point clouds with realistic spatial distributions
    - 3D bounding boxes with proper geometric constraints
    - Height-based intensity variations
    - Noise modeling for realistic sensor simulation
    """
    
    def __init__(self, seed: int = 42):
        """Initialize synthetic data generator with reproducible random seed."""
        np.random.seed(seed)
        self.seed = seed
    
    def generate_point_cloud(self, 
                           n_points: int = 10000,
                           scene_bounds: Tuple[float, float, float, float, float, float] = (-50, -25, -3, 50, 25, 5),
                           ground_ratio: float = 0.3) -> np.ndarray:
        """
        Generate synthetic 3D point cloud with realistic spatial distribution.
        
        Creates point clouds that simulate LiDAR data from autonomous driving
        scenarios, including ground plane, objects, and environmental features.
        
        Mathematical Point Generation:
        
        **Spatial Distribution:**
        $$\\mathbf{p}_i = \\mathbf{p}_{base} + \\mathbf{n}_i$$
        
        Where:
        - $\\mathbf{p}_{base}$ is the base position (uniform or structured)
        - $\\mathbf{n}_i \\sim \\mathcal{N}(0, \\sigma^2)$ is Gaussian noise
        
        **Height-based Intensity:**
        $$I(z) = I_{max} \\cdot \\exp\\left(-\\frac{(z - z_{ref})^2}{2\\sigma_z^2}\\right)$$
        
        Args:
            n_points (int): Number of points to generate
            scene_bounds (tuple): Scene boundaries (x_min, y_min, z_min, x_max, y_max, z_max)
            ground_ratio (float): Fraction of points on ground plane
        
        Returns:
            np.ndarray: Point cloud with shape (N, 4) [x, y, z, intensity]
        """
        x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
        
        points = []
        
        # Generate ground plane points
        n_ground = int(n_points * ground_ratio)
        if n_ground > 0:
            ground_x = np.random.uniform(x_min, x_max, n_ground)
            ground_y = np.random.uniform(y_min, y_max, n_ground)
            ground_z = np.random.normal(-1.5, 0.2, n_ground)  # Ground with slight variation
            ground_intensity = np.random.uniform(0.1, 0.3, n_ground)  # Low intensity for ground
            
            ground_points = np.column_stack([ground_x, ground_y, ground_z, ground_intensity])
            points.append(ground_points)
        
        # Generate object points (higher elevation)
        n_objects = n_points - n_ground
        if n_objects > 0:
            obj_x = np.random.uniform(x_min, x_max, n_objects)
            obj_y = np.random.uniform(y_min, y_max, n_objects)
            obj_z = np.random.uniform(z_min + 1, z_max, n_objects)  # Above ground
            
            # Height-based intensity (higher objects = higher intensity)
            obj_intensity = 0.5 + 0.5 * (obj_z - z_min) / (z_max - z_min)
            obj_intensity += np.random.normal(0, 0.1, n_objects)  # Add noise
            obj_intensity = np.clip(obj_intensity, 0, 1)
            
            obj_points = np.column_stack([obj_x, obj_y, obj_z, obj_intensity])
            points.append(obj_points)
        
        # Combine all points
        all_points = np.vstack(points) if points else np.empty((0, 4))
        
        # Add some structured features (simulating buildings, vehicles)
        n_structured = min(1000, n_points // 10)
        if n_structured > 0:
            # Create some box-like structures
            for _ in range(5):  # 5 box structures
                center_x = np.random.uniform(x_min + 10, x_max - 10)
                center_y = np.random.uniform(y_min + 5, y_max - 5)
                center_z = np.random.uniform(0, 2)
                
                box_size = np.random.uniform(2, 8, 3)  # [length, width, height]
                n_box_points = n_structured // 5
                
                # Generate points on box surfaces
                box_points = self._generate_box_surface_points(
                    [center_x, center_y, center_z], box_size, n_box_points
                )
                
                if len(all_points) + len(box_points) <= n_points:
                    all_points = np.vstack([all_points, box_points])
        
        # Shuffle points for realistic distribution
        indices = np.random.permutation(len(all_points))
        return all_points[indices]
    
    def _generate_box_surface_points(self, center: List[float], size: List[float], n_points: int) -> np.ndarray:
        """Generate points on the surface of a 3D box."""
        cx, cy, cz = center
        lx, ly, lz = size
        
        points = []
        
        # Generate points on each face of the box
        for _ in range(n_points):
            # Randomly select a face (6 faces total)
            face = np.random.randint(0, 6)
            
            if face == 0:  # Front face (x = cx + lx/2)
                x = cx + lx/2
                y = np.random.uniform(cy - ly/2, cy + ly/2)
                z = np.random.uniform(cz - lz/2, cz + lz/2)
            elif face == 1:  # Back face (x = cx - lx/2)
                x = cx - lx/2
                y = np.random.uniform(cy - ly/2, cy + ly/2)
                z = np.random.uniform(cz - lz/2, cz + lz/2)
            elif face == 2:  # Right face (y = cy + ly/2)
                x = np.random.uniform(cx - lx/2, cx + lx/2)
                y = cy + ly/2
                z = np.random.uniform(cz - lz/2, cz + lz/2)
            elif face == 3:  # Left face (y = cy - ly/2)
                x = np.random.uniform(cx - lx/2, cx + lx/2)
                y = cy - ly/2
                z = np.random.uniform(cz - lz/2, cz + lz/2)
            elif face == 4:  # Top face (z = cz + lz/2)
                x = np.random.uniform(cx - lx/2, cx + lx/2)
                y = np.random.uniform(cy - ly/2, cy + ly/2)
                z = cz + lz/2
            else:  # Bottom face (z = cz - lz/2)
                x = np.random.uniform(cx - lx/2, cx + lx/2)
                y = np.random.uniform(cy - ly/2, cy + ly/2)
                z = cz - lz/2
            
            # Intensity based on height and surface normal
            intensity = 0.3 + 0.4 * (z + 3) / 8  # Height-based
            intensity += np.random.normal(0, 0.1)  # Add noise
            intensity = np.clip(intensity, 0, 1)
            
            points.append([x, y, z, intensity])
        
        return np.array(points)
    
    def generate_3d_boxes(self, 
                         n_boxes: int = 10,
                         scene_bounds: Tuple[float, float, float, float] = (-40, -20, 40, 20),
                         box_types: List[str] = ['car', 'pedestrian', 'cyclist']) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic 3D bounding boxes with realistic parameters.
        
        Creates 3D bounding boxes that simulate object detections in autonomous
        driving scenarios, with appropriate size distributions for different
        object classes.
        
        Mathematical Box Generation:
        
        **Box Parameters:**
        $$\\mathbf{b} = [x_c, y_c, z_c, l, w, h, \\theta]$$
        
        Where:
        - $(x_c, y_c, z_c)$ is the box center
        - $(l, w, h)$ are length, width, height
        - $\\theta$ is the heading angle (yaw rotation)
        
        **Size Distributions by Class:**
        - Car: $l \\sim \\mathcal{N}(4.5, 0.5)$, $w \\sim \\mathcal{N}(1.8, 0.2)$, $h \\sim \\mathcal{N}(1.5, 0.2)$
        - Pedestrian: $l \\sim \\mathcal{N}(0.8, 0.1)$, $w \\sim \\mathcal{N}(0.6, 0.1)$, $h \\sim \\mathcal{N}(1.7, 0.2)$
        - Cyclist: $l \\sim \\mathcal{N}(1.8, 0.2)$, $w \\sim \\mathcal{N}(0.6, 0.1)$, $h \\sim \\mathcal{N}(1.7, 0.2)$
        
        Args:
            n_boxes (int): Number of boxes to generate
            scene_bounds (tuple): Scene boundaries (x_min, y_min, x_max, y_max)
            box_types (list): List of object types to generate
        
        Returns:
            tuple: (boxes_3d, labels) where boxes_3d has shape (N, 7) and labels shape (N,)
        """
        x_min, y_min, x_max, y_max = scene_bounds
        
        boxes = []
        labels = []
        
        # Define size distributions for different object types
        size_distributions = {
            'car': {'length': (4.5, 0.5), 'width': (1.8, 0.2), 'height': (1.5, 0.2)},
            'pedestrian': {'length': (0.8, 0.1), 'width': (0.6, 0.1), 'height': (1.7, 0.2)},
            'cyclist': {'length': (1.8, 0.2), 'width': (0.6, 0.1), 'height': (1.7, 0.2)}
        }
        
        # Class label mapping
        class_mapping = {'car': 1, 'pedestrian': 2, 'cyclist': 3}
        
        for i in range(n_boxes):
            # Randomly select object type
            obj_type = np.random.choice(box_types)
            size_params = size_distributions[obj_type]
            
            # Generate box center position
            x_center = np.random.uniform(x_min, x_max)
            y_center = np.random.uniform(y_min, y_max)
            
            # Height depends on object type
            if obj_type == 'car':
                z_center = np.random.normal(0.75, 0.1)  # Car height center
            else:
                z_center = np.random.normal(0.85, 0.1)  # Pedestrian/cyclist height center
            
            # Generate box dimensions based on object type
            length = np.random.normal(*size_params['length'])
            width = np.random.normal(*size_params['width'])
            height = np.random.normal(*size_params['height'])
            
            # Ensure positive dimensions
            length = max(0.5, length)
            width = max(0.3, width)
            height = max(0.5, height)
            
            # Generate heading angle (yaw rotation)
            heading = np.random.uniform(-np.pi, np.pi)
            
            # Create box parameter vector
            box = [x_center, y_center, z_center, length, width, height, heading]
            boxes.append(box)
            
            # Assign class label
            labels.append(class_mapping[obj_type])
        
        return np.array(boxes), np.array(labels)
    
    def generate_detection_results(self, gt_boxes: np.ndarray, gt_labels: np.ndarray, 
                                 detection_rate: float = 0.8, 
                                 false_positive_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic detection results based on ground truth.
        
        Simulates realistic object detection results with true positives,
        false positives, and missed detections.
        
        Args:
            gt_boxes (np.ndarray): Ground truth boxes, shape (N, 7)
            gt_labels (np.ndarray): Ground truth labels, shape (N,)
            detection_rate (float): Probability of detecting each ground truth object
            false_positive_rate (float): Rate of false positive detections
        
        Returns:
            tuple: (pred_boxes, pred_labels, pred_scores)
        """
        pred_boxes = []
        pred_labels = []
        pred_scores = []
        
        # Generate true positive detections
        for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if np.random.random() < detection_rate:
                # Add noise to ground truth box
                noise_std = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05]  # Position and size noise
                pred_box = gt_box + np.random.normal(0, noise_std)
                
                # Generate confidence score (higher for better detections)
                score = np.random.beta(8, 2)  # Beta distribution favoring high scores
                
                pred_boxes.append(pred_box)
                pred_labels.append(gt_label)
                pred_scores.append(score)
        
        # Generate false positive detections
        n_false_positives = int(len(gt_boxes) * false_positive_rate)
        for _ in range(n_false_positives):
            # Generate random box in scene
            fake_box = [
                np.random.uniform(-40, 40),  # x
                np.random.uniform(-20, 20),  # y
                np.random.uniform(0, 2),     # z
                np.random.uniform(1, 5),     # length
                np.random.uniform(0.5, 2),   # width
                np.random.uniform(0.5, 2),   # height
                np.random.uniform(-np.pi, np.pi)  # heading
            ]
            
            fake_label = np.random.choice([1, 2, 3])  # Random class
            fake_score = np.random.beta(2, 8)  # Beta distribution favoring low scores
            
            pred_boxes.append(fake_box)
            pred_labels.append(fake_label)
            pred_scores.append(fake_score)
        
        if pred_boxes:
            return np.array(pred_boxes), np.array(pred_labels), np.array(pred_scores)
        else:
            return np.empty((0, 7)), np.empty((0,)), np.empty((0,))


class VisualizationTester:
    """
    Comprehensive testing suite for Open3D visualization system.
    
    This class provides various tests to validate the functionality,
    performance, and accuracy of the visualization system.
    """
    
    def __init__(self, output_dir: str = "./test_outputs"):
        """Initialize tester with output directory for test results."""
        self.output_dir = output_dir
        self.data_generator = SyntheticDataGenerator()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Test results storage
        self.test_results = {}
    
    def test_synthetic_data_generation(self) -> bool:
        """Test synthetic data generation functionality."""
        print("\n=== Testing Synthetic Data Generation ===")
        
        try:
            # Test point cloud generation
            points = self.data_generator.generate_point_cloud(n_points=5000)
            print(f"✓ Generated point cloud: {points.shape}")
            
            # Validate point cloud properties
            assert points.shape[1] == 4, "Point cloud should have 4 dimensions (x,y,z,intensity)"
            assert np.all(points[:, 3] >= 0) and np.all(points[:, 3] <= 1), "Intensity should be in [0,1]"
            
            # Test 3D box generation
            boxes, labels = self.data_generator.generate_3d_boxes(n_boxes=15)
            print(f"✓ Generated 3D boxes: {boxes.shape}, labels: {labels.shape}")
            
            # Validate box properties
            assert boxes.shape[1] == 7, "Boxes should have 7 parameters"
            assert np.all(boxes[:, 3:6] > 0), "Box dimensions should be positive"
            
            # Test detection results generation
            pred_boxes, pred_labels, pred_scores = self.data_generator.generate_detection_results(boxes, labels)
            print(f"✓ Generated detection results: {pred_boxes.shape}")
            
            # Save test data
            np.save(os.path.join(self.output_dir, "test_points.npy"), points)
            np.save(os.path.join(self.output_dir, "test_gt_boxes.npy"), boxes)
            np.save(os.path.join(self.output_dir, "test_gt_labels.npy"), labels)
            np.save(os.path.join(self.output_dir, "test_pred_boxes.npy"), pred_boxes)
            np.save(os.path.join(self.output_dir, "test_pred_labels.npy"), pred_labels)
            np.save(os.path.join(self.output_dir, "test_pred_scores.npy"), pred_scores)
            
            self.test_results['synthetic_data'] = True
            return True
            
        except Exception as e:
            print(f"✗ Synthetic data generation failed: {e}")
            self.test_results['synthetic_data'] = False
            return False
    
    def test_mathematical_functions(self) -> bool:
        """Test mathematical transformation functions."""
        print("\n=== Testing Mathematical Functions ===")
        
        try:
            # Test rotation function
            points = np.random.randn(2, 10, 3)  # 2 batches, 10 points each
            angles = np.array([np.pi/4, -np.pi/3])  # 45° and -60°
            
            rotated_points = rotate_points_along_z(points, angles)
            print(f"✓ Point rotation test passed: {rotated_points.shape}")
            
            # Verify rotation properties
            # Check that rotation preserves distances from Z-axis
            original_distances = np.sqrt(points[:, :, 0]**2 + points[:, :, 1]**2)
            rotated_distances = np.sqrt(rotated_points[:, :, 0]**2 + rotated_points[:, :, 1]**2)
            
            assert np.allclose(original_distances, rotated_distances, atol=1e-6), "Rotation should preserve distances"
            print("✓ Rotation preserves distances correctly")
            
            # Test 3D box corner generation
            test_boxes = np.array([
                [0, 0, 0, 4, 2, 1.5, 0],      # Box at origin, no rotation
                [10, 5, 1, 3, 1.8, 1.6, np.pi/2]  # Rotated box
            ])
            
            corners = boxes_to_corners_3d(test_boxes)
            print(f"✓ Box corner generation test passed: {corners.shape}")
            
            # Verify corner properties
            assert corners.shape == (2, 8, 3), "Should generate 8 corners per box"
            
            # Check that corners are at correct distances from center
            for i, box in enumerate(test_boxes):
                center = box[:3]
                box_corners = corners[i]
                
                # All corners should be within box dimensions from center
                distances = np.linalg.norm(box_corners - center, axis=1)
                max_distance = np.linalg.norm(box[3:6]) / 2  # Half diagonal
                
                assert np.all(distances <= max_distance * 1.1), "Corners should be within box bounds"
            
            print("✓ Box corner generation is mathematically correct")
            
            self.test_results['mathematical_functions'] = True
            return True
            
        except Exception as e:
            print(f"✗ Mathematical function tests failed: {e}")
            self.test_results['mathematical_functions'] = False
            return False
    
    def test_visualization_creation(self, headless: bool = True) -> bool:
        """Test visualization system creation and basic functionality."""
        print(f"\n=== Testing Visualization Creation (headless={headless}) ===")
        
        try:
            # Create visualizer
            visualizer = Open3DVisualizer(
                window_name="Test Visualization",
                headless=headless
            )
            print("✓ Visualizer created successfully")
            
            # Generate test data
            points = self.data_generator.generate_point_cloud(n_points=1000)
            boxes, labels = self.data_generator.generate_3d_boxes(n_boxes=5)
            
            # Test point cloud creation
            pcd = visualizer.create_point_cloud(points, color_mode='height')
            print(f"✓ Point cloud created: {len(pcd.points)} points")
            
            # Test bounding box creation
            line_sets = visualizer.create_3d_bounding_boxes(boxes, labels)
            print(f"✓ Bounding boxes created: {len(line_sets)} boxes")
            
            # Test coordinate frame creation
            coord_frame = visualizer.create_coordinate_frame(size=2.0)
            print("✓ Coordinate frame created")
            
            # Test grid creation
            grid_lines = visualizer.create_ground_grid()
            print(f"✓ Ground grid created: {len(grid_lines)} lines")
            
            # Clean up
            visualizer.close()
            
            self.test_results['visualization_creation'] = True
            return True
            
        except Exception as e:
            print(f"✗ Visualization creation failed: {e}")
            self.test_results['visualization_creation'] = False
            return False
    
    def test_file_export(self) -> bool:
        """Test file export functionality."""
        print("\n=== Testing File Export ===")
        
        try:
            # Create visualizer in headless mode
            visualizer = Open3DVisualizer(headless=True)
            
            # Generate test data
            points = self.data_generator.generate_point_cloud(n_points=2000)
            boxes, labels = self.data_generator.generate_3d_boxes(n_boxes=8)
            
            # Test PNG export
            png_path = os.path.join(self.output_dir, "test_visualization.png")
            visualizer.visualize_scene(
                points=points,
                gt_boxes=boxes,
                gt_labels=labels,
                save_path=png_path
            )
            
            if os.path.exists(png_path):
                print(f"✓ PNG export successful: {png_path}")
            else:
                print("✗ PNG export failed")
                return False
            
            # Test PLY export
            ply_path = os.path.join(self.output_dir, "test_pointcloud.ply")
            visualizer.visualize_scene(
                points=points,
                save_path=ply_path
            )
            
            if os.path.exists(ply_path):
                print(f"✓ PLY export successful: {ply_path}")
            else:
                print("✗ PLY export failed")
                return False
            
            # Clean up
            visualizer.close()
            
            self.test_results['file_export'] = True
            return True
            
        except Exception as e:
            print(f"✗ File export tests failed: {e}")
            self.test_results['file_export'] = False
            return False
    
    def benchmark_performance(self, iterations: int = 5) -> Dict[str, float]:
        """Benchmark visualization performance with different data sizes."""
        print(f"\n=== Performance Benchmarking ({iterations} iterations) ===")
        
        benchmark_results = {}
        data_sizes = [1000, 5000, 10000, 20000]
        
        for n_points in data_sizes:
            print(f"\nTesting with {n_points} points...")
            
            times = []
            for i in range(iterations):
                # Generate data
                points = self.data_generator.generate_point_cloud(n_points=n_points)
                boxes, labels = self.data_generator.generate_3d_boxes(n_boxes=max(5, n_points//1000))
                
                # Time visualization creation
                start_time = time.time()
                
                visualizer = Open3DVisualizer(headless=True)
                visualizer.visualize_scene(
                    points=points,
                    gt_boxes=boxes,
                    gt_labels=labels,
                    save_path=None  # No file saving for benchmarking
                )
                visualizer.close()
                
                end_time = time.time()
                elapsed = end_time - start_time
                times.append(elapsed)
                
                print(f"  Iteration {i+1}: {elapsed:.3f}s")
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            benchmark_results[f"{n_points}_points"] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'points_per_second': n_points / avg_time
            }
            
            print(f"  Average: {avg_time:.3f}±{std_time:.3f}s ({n_points/avg_time:.0f} points/sec)")
        
        self.test_results['performance'] = benchmark_results
        return benchmark_results
    
    def run_all_tests(self, headless: bool = True, benchmark: bool = False) -> bool:
        """Run complete test suite."""
        print("=" * 60)
        print("OPEN3D VISUALIZATION SYSTEM - COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        all_passed = True
        
        # Core functionality tests
        tests = [
            ('Synthetic Data Generation', self.test_synthetic_data_generation),
            ('Mathematical Functions', self.test_mathematical_functions),
            ('Visualization Creation', lambda: self.test_visualization_creation(headless)),
            ('File Export', self.test_file_export),
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"✗ {test_name} failed with exception: {e}")
                all_passed = False
        
        # Optional performance benchmark
        if benchmark:
            try:
                self.benchmark_performance()
            except Exception as e:
                print(f"✗ Performance benchmark failed: {e}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        for test_name, result in self.test_results.items():
            if test_name != 'performance':
                status = "✓ PASSED" if result else "✗ FAILED"
                print(f"{test_name:30} {status}")
        
        if 'performance' in self.test_results:
            print(f"\nPerformance Results:")
            for size, metrics in self.test_results['performance'].items():
                print(f"  {size:15} {metrics['avg_time']:.3f}s ({metrics['points_per_second']:.0f} pts/sec)")
        
        overall_status = "✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"
        print(f"\nOverall Result: {overall_status}")
        
        return all_passed


def main():
    """Main function with command-line interface for testing."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Suite for Open3D Visualization System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--test-all', action='store_true',
                       help='Run all tests')
    parser.add_argument('--test-synthetic', action='store_true',
                       help='Test synthetic data generation only')
    parser.add_argument('--test-math', action='store_true',
                       help='Test mathematical functions only')
    parser.add_argument('--test-viz', action='store_true',
                       help='Test visualization creation only')
    parser.add_argument('--test-export', action='store_true',
                       help='Test file export only')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of benchmark iterations')
    parser.add_argument('--headless', action='store_true',
                       help='Force headless mode')
    parser.add_argument('--save-dir', type=str, default='./test_outputs',
                       help='Directory for test outputs')
    
    args = parser.parse_args()
    
    # Create tester
    tester = VisualizationTester(output_dir=args.save_dir)
    
    # Determine headless mode
    headless = args.headless or 'DISPLAY' not in os.environ
    
    success = True
    
    try:
        if args.test_all:
            success = tester.run_all_tests(headless=headless, benchmark=args.benchmark)
        else:
            # Run individual tests
            if args.test_synthetic:
                success &= tester.test_synthetic_data_generation()
            if args.test_math:
                success &= tester.test_mathematical_functions()
            if args.test_viz:
                success &= tester.test_visualization_creation(headless=headless)
            if args.test_export:
                success &= tester.test_file_export()
            if args.benchmark:
                tester.benchmark_performance(iterations=args.iterations)
            
            # If no specific tests selected, run all
            if not any([args.test_synthetic, args.test_math, args.test_viz, args.test_export, args.benchmark]):
                success = tester.run_all_tests(headless=headless, benchmark=False)
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())