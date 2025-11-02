#!/usr/bin/env python3
"""
Integration test script for waymokittiall_open3d.py

This script tests the Open3D-based visualization system with synthetic data
that mimics the KITTI and Waymo2KITTI dataset formats.

Tests include:
1. Point cloud loading and visualization
2. 3D bounding box rendering
3. Headless mode with PLY export
4. Compatibility with original data structures
5. Performance benchmarking

Author: AI Assistant
Date: 2024
"""

import numpy as np
import os
import sys
import tempfile
import time
from typing import List, Tuple
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Global variables for imports
IMPORT_SUCCESS = False
OPEN3D_AVAILABLE = False
waymokittiall_open3d = None

try:
    import waymokittiall_open3d
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORT_SUCCESS = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


class SyntheticDataGenerator:
    """Generate synthetic KITTI-format data for testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        np.random.seed(seed)
    
    def generate_point_cloud(self, n_points: int = 10000, 
                           point_range: List[float] = [-50, -25, -3, 50, 25, 2]) -> np.ndarray:
        """
        Generate synthetic LiDAR point cloud.
        
        Args:
            n_points: Number of points to generate
            point_range: [xmin, ymin, zmin, xmax, ymax, zmax]
            
        Returns:
            Point cloud array (N, 4) with XYZ and intensity
        """
        # Generate random points within range
        x = np.random.uniform(point_range[0], point_range[3], n_points)
        y = np.random.uniform(point_range[1], point_range[4], n_points)
        z = np.random.uniform(point_range[2], point_range[5], n_points)
        
        # Generate intensity values (0-1)
        intensity = np.random.uniform(0, 1, n_points)
        
        # Add some structure (ground plane, objects)
        # Ground points
        n_ground = n_points // 4
        ground_indices = np.random.choice(n_points, n_ground, replace=False)
        z[ground_indices] = np.random.uniform(-0.2, 0.2, n_ground)  # Near ground level
        
        # Object points (higher intensity)
        n_objects = n_points // 8
        object_indices = np.random.choice(n_points, n_objects, replace=False)
        intensity[object_indices] = np.random.uniform(0.7, 1.0, n_objects)
        
        return np.column_stack([x, y, z, intensity])
    
    def generate_objects(self, n_objects: int = 5) -> List:
        """
        Generate synthetic Object3d instances.
        
        Args:
            n_objects: Number of objects to generate
            
        Returns:
            List of Object3d instances
        """
        if not IMPORT_SUCCESS:
            return []
            
        objects = []
        object_types = ['Car', 'Pedestrian', 'Cyclist']
        
        for i in range(n_objects):
            # Generate KITTI format label line
            obj_type = np.random.choice(object_types)
            truncated = np.random.uniform(0, 0.5)
            occluded = np.random.randint(0, 3)
            alpha = np.random.uniform(-np.pi, np.pi)
            
            # 2D bbox (dummy values)
            xmin = np.random.uniform(100, 500)
            ymin = np.random.uniform(100, 300)
            xmax = xmin + np.random.uniform(50, 200)
            ymax = ymin + np.random.uniform(30, 100)
            
            # 3D dimensions
            if obj_type == 'Car':
                h, w, l = 1.5, 1.8, 4.2
            elif obj_type == 'Pedestrian':
                h, w, l = 1.7, 0.6, 0.6
            else:  # Cyclist
                h, w, l = 1.6, 0.6, 1.8
            
            # 3D location
            x = np.random.uniform(-20, 20)
            y = np.random.uniform(-10, 10)
            z = np.random.uniform(-1, 1)
            
            # Rotation
            ry = np.random.uniform(-np.pi, np.pi)
            
            # Create label line
            label_line = f"{obj_type} {truncated:.2f} {occluded} {alpha:.2f} " \
                        f"{xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} " \
                        f"{h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f}"
            
            objects.append(waymokittiall_open3d.Object3d(label_line))
        
        return objects
    
    def create_test_files(self, output_dir: str) -> Tuple[str, str]:
        """
        Create temporary test files in KITTI format.
        
        Args:
            output_dir: Directory to create files
            
        Returns:
            Tuple of (velodyne_file, label_file)
        """
        # Generate data
        points = self.generate_point_cloud()
        objects = self.generate_objects()
        
        # Create velodyne file
        velodyne_file = os.path.join(output_dir, "000001.bin")
        points.astype(np.float32).tofile(velodyne_file)
        
        # Create label file
        label_file = os.path.join(output_dir, "000001.txt")
        with open(label_file, 'w') as f:
            for obj in objects:
                label_line = f"{obj.type} {obj.truncation:.2f} {obj.occlusion} {obj.alpha:.2f} " \
                            f"{obj.xmin:.2f} {obj.ymin:.2f} {obj.xmax:.2f} {obj.ymax:.2f} " \
                            f"{obj.h:.2f} {obj.w:.2f} {obj.l:.2f} " \
                            f"{obj.t[0]:.2f} {obj.t[1]:.2f} {obj.t[2]:.2f} {obj.ry:.2f}"
                f.write(label_line + "\n")
        
        return velodyne_file, label_file


class IntegrationTester:
    """Comprehensive integration testing suite."""
    
    def __init__(self):
        """Initialize tester."""
        self.data_gen = SyntheticDataGenerator()
        self.test_results = {}
    
    def test_imports(self) -> bool:
        """Test if all required modules can be imported."""
        print("Testing imports...")
        
        if not IMPORT_SUCCESS:
            print("❌ Failed to import waymokittiall_open3d module")
            return False
        
        if not OPEN3D_AVAILABLE:
            print("❌ Open3D not available")
            return False
        
        print("✅ All imports successful")
        return True
    
    def test_point_cloud_creation(self) -> bool:
        """Test point cloud creation and coloring."""
        print("Testing point cloud creation...")
        
        if not IMPORT_SUCCESS:
            print("❌ Cannot test - import failed")
            return False
        
        try:
            # Generate test data
            points = self.data_gen.generate_point_cloud(1000)
            
            # Create visualizer
            viz = waymokittiall_open3d.Open3DVisualizer(headless=True)
            
            # Create point cloud
            pcd = viz.create_point_cloud(points, color_by_intensity=True)
            
            # Verify point cloud properties
            assert len(pcd.points) == 1000, f"Expected 1000 points, got {len(pcd.points)}"
            assert len(pcd.colors) == 1000, f"Expected 1000 colors, got {len(pcd.colors)}"
            
            # Test without intensity coloring
            pcd_no_intensity = viz.create_point_cloud(points[:, :3], color_by_intensity=False)
            assert len(pcd_no_intensity.colors) == 1000, "Color assignment failed"
            
            print("✅ Point cloud creation successful")
            return True
            
        except Exception as e:
            print(f"❌ Point cloud creation failed: {e}")
            return False
    
    def test_bounding_box_creation(self) -> bool:
        """Test 3D bounding box creation."""
        print("Testing bounding box creation...")
        
        if not IMPORT_SUCCESS:
            print("❌ Cannot test - import failed")
            return False
        
        try:
            # Generate test objects
            objects = self.data_gen.generate_objects(3)
            
            # Create visualizer
            viz = waymokittiall_open3d.Open3DVisualizer(headless=True)
            
            for obj in objects:
                # Test corner generation
                corners = viz._object3d_to_corners(obj)
                assert corners.shape == (8, 3), f"Expected (8,3) corners, got {corners.shape}"
                
                # Test bounding box creation
                bbox = viz.create_bounding_box(corners, color=(1, 0, 0))
                assert len(bbox.lines) == 12, f"Expected 12 lines, got {len(bbox.lines)}"
                assert len(bbox.points) == 8, f"Expected 8 points, got {len(bbox.points)}"
            
            print("✅ Bounding box creation successful")
            return True
            
        except Exception as e:
            print(f"❌ Bounding box creation failed: {e}")
            return False
    
    def test_coordinate_frame_and_grid(self) -> bool:
        """Test coordinate frame and grid creation."""
        print("Testing coordinate frame and grid...")
        
        if not IMPORT_SUCCESS:
            print("❌ Cannot test - import failed")
            return False
        
        try:
            viz = waymokittiall_open3d.Open3DVisualizer(headless=True)
            
            # Test coordinate frame
            coord_frame = viz.create_coordinate_frame(size=5.0)
            assert len(coord_frame.vertices) > 0, "Coordinate frame has no vertices"
            
            # Test ground grid
            grid = viz.create_ground_grid(size=50.0, step=5.0)
            assert len(grid.lines) > 0, "Grid has no lines"
            assert len(grid.points) > 0, "Grid has no points"
            
            print("✅ Coordinate frame and grid creation successful")
            return True
            
        except Exception as e:
            print(f"❌ Coordinate frame/grid creation failed: {e}")
            return False
    
    def test_mathematical_functions(self) -> bool:
        """Test mathematical utility functions."""
        print("Testing mathematical functions...")
        
        if not IMPORT_SUCCESS:
            print("❌ Cannot test - import failed")
            return False
        
        try:
            # Test numpy/torch conversion
            np_array = np.random.rand(10, 3)
            torch_tensor, was_numpy = waymokittiall_open3d.check_numpy_to_torch(np_array)
            assert was_numpy, "Should detect numpy array"
            
            # Test point rotation
            points = np.random.rand(2, 5, 3)
            angles = np.array([0.5, -0.3])
            rotated = waymokittiall_open3d.rotate_points_along_z(points, angles)
            assert rotated.shape == points.shape, "Rotation changed shape"
            
            # Test box corner generation
            boxes = np.array([[0, 0, 0, 2, 2, 2, 0.5], [5, 5, 1, 3, 3, 3, -0.2]])
            corners = waymokittiall_open3d.boxes_to_corners_3d(boxes)
            assert corners.shape == (2, 8, 3), f"Expected (2,8,3), got {corners.shape}"
            
            print("✅ Mathematical functions working correctly")
            return True
            
        except Exception as e:
            print(f"❌ Mathematical functions failed: {e}")
            return False
    
    def test_data_loading(self) -> bool:
        """Test data loading functions."""
        print("Testing data loading...")
        
        if not IMPORT_SUCCESS:
            print("❌ Cannot test - import failed")
            return False
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create test files
                velodyne_file, label_file = self.data_gen.create_test_files(temp_dir)
                
                # Test velodyne loading
                points = waymokittiall_open3d.load_velo_scan(velodyne_file, filterpoints=True)
                assert points.shape[1] == 4, f"Expected 4 columns, got {points.shape[1]}"
                assert len(points) > 0, "No points loaded"
                
                # Test label loading
                objects = waymokittiall_open3d.read_label(label_file)
                assert len(objects) > 0, "No objects loaded"
                assert all(hasattr(obj, 'type') for obj in objects), "Objects missing type attribute"
                
                # Test point filtering
                filtered = waymokittiall_open3d.filter_lidarpoints(points, [-25, -25, -2, 25, 25, 2])
                assert len(filtered) <= len(points), "Filtering increased point count"
                
                print("✅ Data loading successful")
                return True
                
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
    
    def test_headless_visualization(self) -> bool:
        """Test headless visualization with PLY export."""
        print("Testing headless visualization...")
        
        if not IMPORT_SUCCESS:
            print("❌ Cannot test - import failed")
            return False
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate test data
                points = self.data_gen.generate_point_cloud(500)
                objects = self.data_gen.generate_objects(2)
                
                # Test headless visualization
                viz = waymokittiall_open3d.Open3DVisualizer(headless=True)
                
                # Create output path
                output_path = os.path.join(temp_dir, "test_scene.ply")
                
                # Visualize scene
                viz.visualize_scene(
                    points=points,
                    boxes=objects,
                    point_cloud_range=[-30, -30, -3, 30, 30, 3],
                    save_path=output_path,
                    show_coordinate_frame=True,
                    show_ground_grid=True
                )
                
                # Verify output file exists
                assert os.path.exists(output_path), "PLY file not created"
                assert os.path.getsize(output_path) > 0, "PLY file is empty"
                
                print("✅ Headless visualization successful")
                return True
                
        except Exception as e:
            print(f"❌ Headless visualization failed: {e}")
            return False
    
    def test_integration_function(self) -> bool:
        """Test the main integration function."""
        print("Testing integration function...")
        
        if not IMPORT_SUCCESS:
            print("❌ Cannot test - import failed")
            return False
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create test files
                velodyne_file, label_file = self.data_gen.create_test_files(temp_dir)
                
                # Load data
                points = waymokittiall_open3d.load_velo_scan(velodyne_file)
                objects = waymokittiall_open3d.read_label(label_file)
                
                # Create dummy calibration
                class DummyCalib:
                    pass
                calib = DummyCalib()
                
                # Test visualization function
                output_path = os.path.join(temp_dir, "integration_test.ply")
                
                waymokittiall_open3d.visualize_lidar_with_boxes_open3d(
                    pc_velo=points,
                    object3dlabels=objects,
                    calib=calib,
                    point_cloud_range=[-50, -25, -3, 50, 25, 2],
                    save_path=output_path,
                    headless=True
                )
                
                # Verify output
                assert os.path.exists(output_path), "Integration output not created"
                
                print("✅ Integration function successful")
                return True
                
        except Exception as e:
            print(f"❌ Integration function failed: {e}")
            return False
    
    def benchmark_performance(self) -> dict:
        """Benchmark visualization performance."""
        print("Benchmarking performance...")
        
        if not IMPORT_SUCCESS:
            print("❌ Cannot benchmark - import failed")
            return {}
        
        results = {}
        
        try:
            # Test different point cloud sizes
            sizes = [1000, 5000, 10000, 20000]
            
            for size in sizes:
                print(f"  Testing {size} points...")
                
                # Generate data
                points = self.data_gen.generate_point_cloud(size)
                objects = self.data_gen.generate_objects(5)
                
                # Time visualization
                start_time = time.time()
                
                viz = waymokittiall_open3d.Open3DVisualizer(headless=True)
                viz.visualize_scene(
                    points=points,
                    boxes=objects,
                    point_cloud_range=[-50, -25, -3, 50, 25, 2],
                    save_path=None,
                    show_coordinate_frame=True,
                    show_ground_grid=True
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                results[size] = {
                    'time': processing_time,
                    'points_per_second': size / processing_time
                }
                
                print(f"    {size} points: {processing_time:.3f}s ({size/processing_time:.0f} pts/s)")
            
            print("✅ Performance benchmarking complete")
            return results
            
        except Exception as e:
            print(f"❌ Performance benchmarking failed: {e}")
            return {}
    
    def run_all_tests(self) -> dict:
        """Run all integration tests."""
        print("=" * 60)
        print("Running Open3D Integration Tests")
        print("=" * 60)
        
        tests = [
            ('imports', self.test_imports),
            ('point_cloud_creation', self.test_point_cloud_creation),
            ('bounding_box_creation', self.test_bounding_box_creation),
            ('coordinate_frame_grid', self.test_coordinate_frame_and_grid),
            ('mathematical_functions', self.test_mathematical_functions),
            ('data_loading', self.test_data_loading),
            ('headless_visualization', self.test_headless_visualization),
            ('integration_function', self.test_integration_function),
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n[{passed+1}/{total}] {test_name.replace('_', ' ').title()}")
            print("-" * 40)
            
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
            except Exception as e:
                print(f"❌ Test {test_name} crashed: {e}")
                results[test_name] = False
        
        # Run performance benchmark
        print(f"\n[{total+1}/{total+1}] Performance Benchmark")
        print("-" * 40)
        benchmark_results = self.benchmark_performance()
        results['benchmark'] = benchmark_results
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Passed: {passed}/{total} tests")
        print(f"Success Rate: {passed/total*100:.1f}%")
        
        if passed == total:
            print("🎉 All tests passed! Open3D integration is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return results


def main():
    """Main function for running integration tests."""
    parser = argparse.ArgumentParser(description="Integration tests for Open3D visualization")
    parser.add_argument("--test", choices=['all', 'imports', 'basic', 'performance'], 
                       default='all', help="Which tests to run")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    tester = IntegrationTester()
    
    if args.test == 'all':
        results = tester.run_all_tests()
    elif args.test == 'imports':
        results = {'imports': tester.test_imports()}
    elif args.test == 'basic':
        results = {
            'imports': tester.test_imports(),
            'point_cloud': tester.test_point_cloud_creation(),
            'bounding_box': tester.test_bounding_box_creation(),
        }
    elif args.test == 'performance':
        results = {'benchmark': tester.benchmark_performance()}
    
    if args.verbose:
        print("\nDetailed Results:")
        for test_name, result in results.items():
            print(f"  {test_name}: {result}")
    
    return results


if __name__ == "__main__":
    main()