import tarfile
import glob
import os

# Path to where your zip files are stored
zip_dir = "/mnt/e/Shared/Dataset/NuScenes/"
# Destination folder
extract_dir = os.path.join(zip_dir, "v1.0-trainval")

# Make sure the destination directory exists
os.makedirs(extract_dir, exist_ok=True)

    # Find all blob tgz files
    zip_files = glob.glob(os.path.join(zip_dir, "v1.0-trainval*_blobs.tgz"))
    
    # Also look for metadata files
    meta_files = glob.glob(os.path.join(zip_dir, "v1.0-*meta*.tgz"))

    print(f"Found {len(zip_files)} blob tgz files and {len(meta_files)} metadata files.")

    # Extract each file
    all_files = zip_files + meta_files
    for zip_path in all_files:
        print(f"Extracting {os.path.basename(zip_path)}...")
        with tarfile.open(zip_path, 'r:gz') as tar_ref:
            tar_ref.extractall(nuscenes_root)

    print("Done extracting all files.")
    print(f"Dataset extracted to: {nuscenes_root}")
    
    # Validate extraction
    validate_dataset_structure(str(nuscenes_root))

def check_extracted_folders(extract_dir: str) -> List[str]:
    """
    Check the structure of extracted folders
    
    Args:
        extract_dir: Directory where files were extracted
        
    Returns:
        List of subfolder names
    """
    print("\nChecking extracted folder structure:")
    if os.path.exists(extract_dir):
        print(f"Extraction directory {extract_dir} exists")
        subfolders = [f for f in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, f))]
        print(f"Contains {len(subfolders)} subfolders: {', '.join(subfolders)}")
        return subfolders
    else:
        print(f"Extraction directory {extract_dir} does not exist!")
        return []

def count_files_and_check_annotations(nuscenes_root: str) -> Tuple[Dict[str, int], Dict]:
    """
    Count image files and check annotation files following standard NuScenes structure
    
    Args:
        nuscenes_root: Root directory of the NuScenes dataset
        
    Returns:
        Tuple containing image count by camera type and annotations data
    """
    print("\nCounting image files and checking annotations:")
    
    # Standard NuScenes annotation directory
    annotation_dir = os.path.join(nuscenes_root, "v1.0-trainval")
    
    # Find image folder
    samples_dir = os.path.join(nuscenes_root, "samples")
    
    # Count images by sensor type
    image_count = defaultdict(int)
    total_images = 0
    
    if os.path.exists(samples_dir):
        for sensor_type in NUSCENES_STRUCTURE['samples']:
            sensor_dir = os.path.join(samples_dir, sensor_type)
            if os.path.exists(sensor_dir):
                files = [f for f in os.listdir(sensor_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pcd', '.pcd.bin'))]
                image_count[sensor_type] = len(files)
                total_images += len(files)
        
        print(f"Found samples directory: {samples_dir}")
        print(f"Total file count: {total_images}")
        for sensor_type, count in image_count.items():
            if count > 0:
                print(f"  - {sensor_type}: {count} files")
    else:
        print(f"Samples directory does not exist: {samples_dir}")
    
    # Check annotation files
    annotations = {}
    missing_annotations = []
    
    if os.path.exists(annotation_dir):
        print(f"\nFound annotation directory: {annotation_dir}")
        
        for file_name in REQUIRED_ANNOTATION_FILES:
            file_path = os.path.join(annotation_dir, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        annotations[file_name] = len(data)
                        print(f"  - {file_name}: {len(data)} records")
                    elif isinstance(data, dict):
                        annotations[file_name] = len(data)
                        print(f"  - {file_name}: {len(data)} entries")
                except json.JSONDecodeError:
                    print(f"  - {file_name}: JSON format error")
                    missing_annotations.append(file_name)
            else:
                missing_annotations.append(file_name)
        
        if missing_annotations:
            print(f"\nMissing annotation files: {missing_annotations}")
    else:
        print(f"Annotation directory does not exist: {annotation_dir}")
        missing_annotations = REQUIRED_ANNOTATION_FILES
    
    return image_count, annotations

def diagnose_dataset_issues(nuscenes_root: str) -> Dict[str, Any]:
    """
    Comprehensive dataset validation and issue diagnosis
    
    Args:
        nuscenes_root: Root directory of the NuScenes dataset
        
    Returns:
        Dictionary containing diagnosis results and suggestions
    """
    print("\n" + "="*60)
    print("NUSCENES DATASET DIAGNOSIS")
    print("="*60)
    
    diagnosis = {
        'structure_issues': [],
        'missing_files': [],
        'data_integrity': {},
        'suggestions': [],
        'status': 'unknown'
    }
    
    # Check basic directory structure
    print("\n1. Checking directory structure...")
    
    required_dirs = ['samples', 'sweeps', 'v1.0-trainval']
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = os.path.join(nuscenes_root, dir_name)
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_name}/ found")
        else:
            print(f"  ✗ {dir_name}/ missing")
            missing_dirs.append(dir_name)
            diagnosis['structure_issues'].append(f"Missing directory: {dir_name}")
    
    # Check samples subdirectories
    samples_dir = os.path.join(nuscenes_root, 'samples')
    if os.path.exists(samples_dir):
        print("\n2. Checking samples subdirectories...")
        for sensor_type in NUSCENES_STRUCTURE['samples']:
            sensor_dir = os.path.join(samples_dir, sensor_type)
            if os.path.exists(sensor_dir):
                files = [f for f in os.listdir(sensor_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pcd', '.pcd.bin'))]
                file_count = len(files)
                print(f"  ✓ {sensor_type}: {file_count} files")
                diagnosis['data_integrity'][sensor_type] = file_count
            else:
                print(f"  ✗ {sensor_type}: missing")
                diagnosis['structure_issues'].append(f"Missing sensor directory: {sensor_type}")
    
    # Check sweeps subdirectories
    sweeps_dir = os.path.join(nuscenes_root, 'sweeps')
    if os.path.exists(sweeps_dir):
        print("\n3. Checking sweeps subdirectories...")
        for sensor_type in NUSCENES_STRUCTURE['sweeps']:
            sensor_dir = os.path.join(sweeps_dir, sensor_type)
            if os.path.exists(sensor_dir):
                files = [f for f in os.listdir(sensor_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pcd', '.pcd.bin'))]
                file_count = len(files)
                print(f"  ✓ {sensor_type}: {file_count} files")
                diagnosis['data_integrity'][f"sweeps_{sensor_type}"] = file_count
            else:
                print(f"  ✗ {sensor_type}: missing")
                diagnosis['structure_issues'].append(f"Missing sweeps directory: {sensor_type}")
    
    # Check annotation files
    print("\n4. Checking annotation files...")
    annotation_dir = os.path.join(nuscenes_root, 'v1.0-trainval')
    
    if os.path.exists(annotation_dir):
        for file_name in REQUIRED_ANNOTATION_FILES:
            file_path = os.path.join(annotation_dir, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    record_count = len(data) if isinstance(data, (list, dict)) else 0
                    print(f"  ✓ {file_name}: {record_count} records")
                    diagnosis['data_integrity'][file_name] = record_count
                except json.JSONDecodeError:
                    print(f"  ✗ {file_name}: JSON format error")
                    diagnosis['missing_files'].append(f"{file_name} (corrupted)")
                except Exception as e:
                    print(f"  ✗ {file_name}: Error - {e}")
                    diagnosis['missing_files'].append(f"{file_name} (error)")
            else:
                print(f"  ✗ {file_name}: missing")
                diagnosis['missing_files'].append(file_name)
    else:
        print(f"  ✗ Annotation directory missing: {annotation_dir}")
        diagnosis['structure_issues'].append("Missing v1.0-trainval directory")
        diagnosis['missing_files'].extend(REQUIRED_ANNOTATION_FILES)
    
    # Check data consistency
    print("\n5. Checking data consistency...")
    
    # Check if we have both images and annotations
    has_images = any(count > 0 for key, count in diagnosis['data_integrity'].items() 
                    if 'CAM_' in key)
    
    # Check for LiDAR data in both samples and sweeps
    has_lidar = (diagnosis['data_integrity'].get('LIDAR_TOP', 0) > 0 or 
                diagnosis['data_integrity'].get('sweeps_LIDAR_TOP', 0) > 0)
    
    has_annotations = any(file_name in diagnosis['data_integrity'] 
                         for file_name in REQUIRED_ANNOTATION_FILES)
    
    if has_images:
        print("  ✓ Camera images found")
    else:
        print("  ✗ No camera images found")
        diagnosis['structure_issues'].append("No camera images")
    
    if has_lidar:
        lidar_samples = diagnosis['data_integrity'].get('LIDAR_TOP', 0)
        lidar_sweeps = diagnosis['data_integrity'].get('sweeps_LIDAR_TOP', 0)
        print(f"  ✓ LiDAR data found (samples: {lidar_samples}, sweeps: {lidar_sweeps})")
    else:
        print("  ✗ No LiDAR data found")
        diagnosis['structure_issues'].append("No LiDAR data")
    
    if has_annotations:
        print("  ✓ Annotation files found")
    else:
        print("  ✗ No annotation files found")
        diagnosis['structure_issues'].append("No annotation files")
    
    # Generate suggestions
    print("\n6. Generating suggestions...")
    
    if missing_dirs:
        if 'samples' in missing_dirs:
            diagnosis['suggestions'].append(
                "Missing 'samples' directory - ensure you've extracted the main dataset files"
            )
        if 'sweeps' in missing_dirs:
            diagnosis['suggestions'].append(
                "Missing 'sweeps' directory - you may need to download the sweep data separately"
            )
        if 'v1.0-trainval' in missing_dirs:
            diagnosis['suggestions'].append(
                "Missing 'v1.0-trainval' directory - ensure you've downloaded the metadata/annotations"
            )
    
    if diagnosis['missing_files']:
        diagnosis['suggestions'].append(
            f"Missing annotation files: {', '.join(diagnosis['missing_files'])} - "
            "download the v1.0-trainval metadata package"
        )
    
    if not has_images and not has_lidar:
        diagnosis['suggestions'].append(
            "No sensor data found - check if the dataset was properly extracted to the correct location"
        )
    
    # Determine overall status
    if not diagnosis['structure_issues'] and not diagnosis['missing_files']:
        diagnosis['status'] = 'healthy'
        print("\n  ✓ Dataset appears to be complete and properly structured")
    elif diagnosis['structure_issues'] or len(diagnosis['missing_files']) > 3:
        diagnosis['status'] = 'critical'
        print("\n  ✗ Dataset has critical issues that need to be resolved")
    else:
        diagnosis['status'] = 'warning'
        print("\n  ⚠ Dataset has some issues but may be partially usable")
    
    # Print summary
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    print(f"Status: {diagnosis['status'].upper()}")
    
    if diagnosis['suggestions']:
        print("\nSuggestions:")
        for i, suggestion in enumerate(diagnosis['suggestions'], 1):
            print(f"{i}. {suggestion}")
    
    return diagnosis

def check_nuscenes_data_structure(nuscenes_root: str) -> bool:
    """
    Verify NuScenes dataset structure and data integrity
    
    Args:
        nuscenes_root: Root directory of the NuScenes dataset
        
    Returns:
        True if structure is valid, False otherwise
    """
    print("\n" + "="*60)
    print("NUSCENES DATA STRUCTURE VERIFICATION")
    print("="*60)
    
    # Check if root directory exists
    if not os.path.exists(nuscenes_root):
        print(f"✗ Root directory does not exist: {nuscenes_root}")
        return False
    
    print(f"Checking NuScenes dataset at: {nuscenes_root}")
    
    # Verify directory structure
    structure_valid = True
    
    # Check main directories
    main_dirs = ['samples', 'sweeps', 'v1.0-trainval']
    for dir_name in main_dirs:
        dir_path = os.path.join(nuscenes_root, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ {dir_name}/ directory found")
        else:
            print(f"✗ {dir_name}/ directory missing")
            structure_valid = False
    
    # Check samples structure
    samples_dir = os.path.join(nuscenes_root, 'samples')
    if os.path.exists(samples_dir):
        print("\nChecking samples structure:")
        for sensor_type in NUSCENES_STRUCTURE['samples']:
            sensor_dir = os.path.join(samples_dir, sensor_type)
            if os.path.exists(sensor_dir):
                file_count = len([f for f in os.listdir(sensor_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pcd', '.pcd.bin'))])
                print(f"  ✓ {sensor_type}: {file_count} files")
            else:
                print(f"  ✗ {sensor_type}: missing")
                structure_valid = False
    
    # Check sweeps structure
    sweeps_dir = os.path.join(nuscenes_root, 'sweeps')
    if os.path.exists(sweeps_dir):
        print("\nChecking sweeps structure:")
        for sensor_type in NUSCENES_STRUCTURE['sweeps']:
            sensor_dir = os.path.join(sweeps_dir, sensor_type)
            if os.path.exists(sensor_dir):
                file_count = len([f for f in os.listdir(sensor_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pcd', '.pcd.bin'))])
                print(f"  ✓ {sensor_type}: {file_count} files")
            else:
                print(f"  ✗ {sensor_type}: missing")
                # Sweeps are optional for some use cases
                print(f"    (Note: sweeps are optional for basic usage)")
    
    # Check annotation files
    annotation_dir = os.path.join(nuscenes_root, 'v1.0-trainval')
    if os.path.exists(annotation_dir):
        print("\nChecking annotation files:")
        for file_name in REQUIRED_ANNOTATION_FILES:
            file_path = os.path.join(annotation_dir, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    record_count = len(data) if isinstance(data, (list, dict)) else 0
                    print(f"  ✓ {file_name}: {record_count} records")
                except json.JSONDecodeError:
                    print(f"  ✗ {file_name}: JSON format error")
                    structure_valid = False
                except Exception as e:
                    print(f"  ✗ {file_name}: Error reading file - {e}")
                    structure_valid = False
            else:
                print(f"  ✗ {file_name}: missing")
                structure_valid = False
    
    # Verify data consistency
    print("\nVerifying data consistency:")
    
    # Check if annotation files reference existing sensor data
    if structure_valid:
        try:
            # Load sample and sample_data files
            sample_file = os.path.join(annotation_dir, 'sample.json')
            sample_data_file = os.path.join(annotation_dir, 'sample_data.json')
            
            with open(sample_file, 'r') as f:
                samples = json.load(f)
            with open(sample_data_file, 'r') as f:
                sample_data = json.load(f)
            
            # Check a few samples to verify file references
            missing_files = 0
            checked_files = 0
            
            # Check first 10 sample_data entries directly
            for sd in sample_data[:min(10, len(sample_data))]:
                file_path = os.path.join(nuscenes_root, sd['filename'])
                checked_files += 1
                if not os.path.exists(file_path):
                    missing_files += 1
            
            if missing_files == 0:
                print(f"  ✓ All checked sensor files exist ({checked_files} files)")
            else:
                print(f"  ⚠ {missing_files}/{checked_files} sensor files missing")
                if missing_files > checked_files * 0.1:  # More than 10% missing
                    structure_valid = False
        
        except Exception as e:
            print(f"  ✗ Error verifying data consistency: {e}")
            structure_valid = False
    
    # Final result
    print("\n" + "="*60)
    if structure_valid:
        print("✓ DATASET STRUCTURE IS VALID")
        print("The NuScenes dataset appears to be properly structured and complete.")
    else:
        print("✗ DATASET STRUCTURE HAS ISSUES")
        print("The dataset structure is incomplete or has errors.")
    return structure_valid


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: quaternion [w, x, y, z]
    
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def get_3d_box_corners(center, size, rotation):
    """
    Get 8 corners of a 3D bounding box.
    
    Args:
        center: [x, y, z] center of the box
        size: [width, length, height] dimensions of the box
        rotation: quaternion [w, x, y, z] rotation of the box
    
    Returns:
        8x3 array of corner coordinates
    """
    w, l, h = size
    
    # Define box corners in local coordinate system (centered at origin)
    corners = np.array([
        [-w/2, -l/2, -h/2],  # bottom-back-left
        [w/2, -l/2, -h/2],   # bottom-back-right
        [w/2, l/2, -h/2],    # bottom-front-right
        [-w/2, l/2, -h/2],   # bottom-front-left
        [-w/2, -l/2, h/2],   # top-back-left
        [w/2, -l/2, h/2],    # top-back-right
        [w/2, l/2, h/2],     # top-front-right
        [-w/2, l/2, h/2]     # top-front-left
    ])
    
    # Apply rotation
    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    corners = corners @ rotation_matrix.T
    
    # Translate to center
    corners += center
    
    return corners


def project_3d_box_to_2d(center_3d, size_3d, rotation_3d, cam_translation, cam_rotation, camera_intrinsic, ego_translation, ego_rotation, debug=False):
    """
    Project 3D bounding box to 2D image coordinates using NuScenes coordinate system conventions.
    
    Args:
        center_3d: [x, y, z] center of 3D box in global coordinates
        size_3d: [width, length, height] dimensions of 3D box
        rotation_3d: quaternion [w, x, y, z] rotation of 3D box
        cam_translation: [x, y, z] camera translation relative to ego vehicle
        cam_rotation: quaternion [w, x, y, z] camera rotation relative to ego vehicle
        camera_intrinsic: 3x3 camera intrinsic matrix
        ego_translation: [x, y, z] ego vehicle translation in global coordinates
        ego_rotation: quaternion [w, x, y, z] ego vehicle rotation in global coordinates
        debug: whether to print debug information
    
    Returns:
        8x2 array of 2D corner coordinates, or None if projection fails
    """
    try:
        if debug:
            print(f"DEBUG: Box center_3d: {center_3d}")
            print(f"DEBUG: Box size_3d: {size_3d}")
            print(f"DEBUG: Box rotation_3d: {rotation_3d}")
            print(f"DEBUG: Ego translation: {ego_translation}")
            print(f"DEBUG: Ego rotation: {ego_rotation}")
            print(f"DEBUG: Cam translation: {cam_translation}")
            print(f"DEBUG: Cam rotation: {cam_rotation}")
        
        # Get 3D box corners in global coordinates
        corners_3d_global = get_3d_box_corners(center_3d, size_3d, rotation_3d)
        
        if debug:
            print(f"DEBUG: Corners in global coordinates:")
            for i, corner in enumerate(corners_3d_global):
                print(f"  Corner {i}: {corner}")
        
        # Step 1: Transform from global to ego vehicle coordinate system
        # Following NuScenes convention: inverse transform = R^T * (points - t)
        ego_rotation_matrix = quaternion_to_rotation_matrix(ego_rotation)
        corners_ego = np.dot(ego_rotation_matrix.T, (corners_3d_global - ego_translation).T).T
        
        if debug:
            print(f"DEBUG: Ego rotation matrix:")
            print(ego_rotation_matrix)
            print(f"DEBUG: Corners in ego coordinates:")
            for i, corner in enumerate(corners_ego):
                print(f"  Corner {i}: {corner}")
        
        # Step 2: Transform from ego to camera coordinate system
        # Following NuScenes convention: inverse transform = R^T * (points - t)
        cam_rotation_matrix = quaternion_to_rotation_matrix(cam_rotation)
        corners_cam = np.dot(cam_rotation_matrix.T, (corners_ego - cam_translation).T).T
        
        if debug:
            print(f"DEBUG: Cam rotation matrix:")
            print(cam_rotation_matrix)
            print(f"DEBUG: Corners in camera coordinates:")
            for i, corner in enumerate(corners_cam):
                print(f"  Corner {i}: {corner}")
        
        # Step 3: Use view_points-like projection (following NuScenes devkit approach)
        # Convert to homogeneous coordinates for projection
        corners_cam_homogeneous = np.ones((4, len(corners_cam)))
        corners_cam_homogeneous[:3, :] = corners_cam.T
        
        # Apply camera intrinsic matrix
        viewpad = np.eye(4)
        viewpad[:3, :3] = camera_intrinsic
        
        # Project points
        projected_points = np.dot(viewpad, corners_cam_homogeneous)[:3, :]
        
        if debug:
            print(f"DEBUG: Projected points before normalization:")
            for i, point in enumerate(projected_points.T):
                print(f"  Point {i}: {point}")
        
        # Check if any points are behind the camera (z <= 0)
        depths = projected_points[2, :]
        if np.any(depths <= 0):
            if debug:
                print(f"DEBUG: Some points behind camera, depths: {depths}")
                behind_camera = depths <= 0
                print(f"DEBUG: Points behind camera: {np.where(behind_camera)[0]}")
            # Don't return None immediately, let's see if we can still get a valid projection
        
        # Normalize by depth (perspective division)
        corners_2d = projected_points[:2, :] / projected_points[2:3, :]
        
        if debug:
            print(f"DEBUG: Final 2D corners:")
            for i, corner in enumerate(corners_2d.T):
                print(f"  Corner {i}: {corner}")
        
        return corners_2d.T  # Return as Nx2 array
        
    except Exception as e:
        print(f"Projection error: {e}")
        return None


def get_2d_bbox_from_3d_projection(corners_2d):
    """
    Calculate 2D bounding box from projected 3D corners.
    
    Args:
        corners_2d: 8x2 array of projected 3D box corners
    
    Returns:
        [x_min, y_min, x_max, y_max] or None if invalid
    """
    if corners_2d is None or len(corners_2d) != 8:
        return None
    
    x_coords = corners_2d[:, 0]
    y_coords = corners_2d[:, 1]
    
    # Filter out invalid coordinates (e.g., behind camera)
    valid_mask = np.isfinite(x_coords) & np.isfinite(y_coords)
    if not np.any(valid_mask):
        return None
    
    x_min = np.min(x_coords[valid_mask])
    x_max = np.max(x_coords[valid_mask])
    y_min = np.min(y_coords[valid_mask])
    y_max = np.max(y_coords[valid_mask])
    
    return [x_min, y_min, x_max, y_max]


def draw_2d_bbox(ax, bbox_2d, category_name="", color='green', linewidth=2):
    """
    Draw 2D bounding box on image.
    
    Args:
        ax: matplotlib axis object
        bbox_2d: [x_min, y_min, x_max, y_max] bounding box coordinates
        category_name: object category name for label
        color: box color
        linewidth: line width for the box
    """
    if bbox_2d is None:
        return
    
    x_min, y_min, x_max, y_max = bbox_2d
    width = x_max - x_min
    height = y_max - y_min
    
    # Draw rectangle
    import matplotlib.patches as patches
    rect = patches.Rectangle((x_min, y_min), width, height, 
                           linewidth=linewidth, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    
    # Add label if provided
    if category_name:
        ax.text(x_min, y_min - 5, category_name, 
               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
               fontsize=8, color='white', fontweight='bold')


def draw_3d_box_2d(ax, corners_2d, color='red', linewidth=2):
    """
    Draw 3D bounding box on 2D image using projected corners with improved visualization.
    
    Args:
        ax: matplotlib axis object
        corners_2d: 8x2 array of 2D corner coordinates
        color: line color for the 3D box
        linewidth: line width
    """
    if corners_2d is None or len(corners_2d) != 8:
        return
    
    # Define the 12 edges of a 3D box (connecting corner indices)
    # Bottom face (z=0): corners 0,1,2,3
    # Top face (z=1): corners 4,5,6,7
    edges = [
        # Bottom face edges
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face edges  
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges connecting bottom to top
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    # Draw each edge
    for edge in edges:
        start_idx, end_idx = edge
        start_point = corners_2d[start_idx]
        end_point = corners_2d[end_idx]
        
        # Check if both points are valid (not NaN or Inf)
        if (np.isfinite(start_point).all() and np.isfinite(end_point).all()):
            ax.plot([start_point[0], end_point[0]], 
                   [start_point[1], end_point[1]], 
                   color=color, linewidth=linewidth, alpha=0.8)
    
    # Draw center point for reference
    center_2d = np.mean(corners_2d, axis=0)
    if np.isfinite(center_2d).all():
        ax.plot(center_2d[0], center_2d[1], 'o', color=color, markersize=4, alpha=0.8)
    edges = [
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face  
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    # Draw edges
    for edge in edges:
        start_idx, end_idx = edge
        start_point = corners_2d[start_idx]
        end_point = corners_2d[end_idx]
        
        # Check if points are within reasonable image bounds
        if (0 <= start_point[0] <= 2000 and 0 <= start_point[1] <= 2000 and
            0 <= end_point[0] <= 2000 and 0 <= end_point[1] <= 2000):
            
            ax.plot([start_point[0], end_point[0]], 
                   [start_point[1], end_point[1]], 
                   'r-', linewidth=2, alpha=0.8)


def visualize_sample_with_boxes(nuscenes_root, sample_idx):
    """
    Visualize a sample with bounding boxes for all cameras and save to local files.
    
    Args:
        nuscenes_root (str): Path to the NuScenes dataset root directory
        sample_idx (int): Index of the sample to visualize
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image
        import numpy as np
        
        # Load annotation files
        annotation_dir = os.path.join(nuscenes_root, 'v1.0-trainval')
        print(f"\nVisualizing sample {sample_idx} with bounding boxes:")
        print(f"Successfully loaded annotation files from: {annotation_dir}")
        
        with open(os.path.join(annotation_dir, 'sample.json'), 'r') as f:
            samples = json.load(f)
        
        with open(os.path.join(annotation_dir, 'sample_data.json'), 'r') as f:
            sample_data = json.load(f)
        
        with open(os.path.join(annotation_dir, 'sample_annotation.json'), 'r') as f:
            sample_annotations = json.load(f)
        
        with open(os.path.join(annotation_dir, 'category.json'), 'r') as f:
            categories = json.load(f)
        
        with open(os.path.join(annotation_dir, 'sensor.json'), 'r') as f:
            sensors = json.load(f)
        
        with open(os.path.join(annotation_dir, 'calibrated_sensor.json'), 'r') as f:
            calibrated_sensors = json.load(f)
        
        with open(os.path.join(annotation_dir, 'ego_pose.json'), 'r') as f:
            ego_poses = json.load(f)
        
        with open(os.path.join(annotation_dir, 'instance.json'), 'r') as f:
            instances = json.load(f)
        
        # Get the sample
        if sample_idx >= len(samples):
            print(f"Error: Sample index {sample_idx} is out of range. Total samples: {len(samples)}")
            return
        
        sample = samples[sample_idx]
        print(f"Sample token: {sample['token']}")
        print(f"Scene token: {sample['scene_token']}")
        print(f"Timestamp: {sample['timestamp']}")
        
        # Find all sample_data entries for this sample
        sample_data_entries = [sd for sd in sample_data if sd['sample_token'] == sample['token']]
        
        # Filter for camera data only (main samples, not sweeps)
        camera_data = [sd for sd in sample_data_entries 
                      if 'CAM' in sd.get('filename', '') and 'samples/' in sd.get('filename', '')]
        
        if not camera_data:
            print("No camera data found for this sample")
            return
        
        # Create mapping from calibrated_sensor_token to channel
        calibrated_to_sensor = {cs['token']: cs['sensor_token'] for cs in calibrated_sensors}
        sensor_to_channel = {s['token']: s['channel'] for s in sensors}
        
        # Get sample annotations for this sample
        sample_anns = [ann for ann in sample_annotations if ann['sample_token'] == sample['token']]
        
        # Create category lookup
        category_lookup = {cat['token']: cat['name'] for cat in categories}
        
        # Create instance to category mapping
        instance_to_category = {inst['token']: inst['category_token'] for inst in instances}
        
        # Create output directory for visualizations
        output_dir = os.path.join(nuscenes_root, 'visualizations', f'sample_{sample_idx}')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Found {len(camera_data)} camera views and {len(sample_anns)} annotations")
        
        # Process each camera view
        saved_files = []
        for cam_data in camera_data:
            try:
                # Get channel name
                calibrated_sensor_token = cam_data['calibrated_sensor_token']
                sensor_token = calibrated_to_sensor.get(calibrated_sensor_token)
                channel = sensor_to_channel.get(sensor_token, 'UNKNOWN')
                
                print(f"Processing {channel}...")
                
                # Load image
                image_path = os.path.join(nuscenes_root, cam_data['filename'])
                if not os.path.exists(image_path):
                    print(f"  Warning: Image file not found: {image_path}")
                    continue
                
                # Load and display image
                img = Image.open(image_path)
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax.imshow(img)
                ax.set_title(f'{channel} - Sample {sample_idx}')
                
                # Get camera calibration data
                calibrated_sensor = next((cs for cs in calibrated_sensors 
                                        if cs['token'] == calibrated_sensor_token), None)
                
                if calibrated_sensor and calibrated_sensor.get('camera_intrinsic'):
                    # Project and draw 3D bounding boxes
                    camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
                    cam_translation = np.array(calibrated_sensor['translation'])
                    cam_rotation = np.array(calibrated_sensor['rotation'])  # quaternion [w, x, y, z]
                    
                    # Get ego pose for this sample
                    ego_pose = next((ep for ep in ego_poses if ep['token'] == cam_data['ego_pose_token']), None)
                    if not ego_pose:
                        print(f"  Warning: No ego pose found for {channel}")
                        continue
                    
                    ego_translation = np.array(ego_pose['translation'])
                    ego_rotation = np.array(ego_pose['rotation'])
                    
                    boxes_drawn = 0
                    for i, ann in enumerate(sample_anns):
                        try:
                            # Get 3D bounding box parameters
                            center_3d = np.array(ann['translation'])  # [x, y, z]
                            size_3d = np.array(ann['size'])  # [width, length, height]
                            rotation_3d = np.array(ann['rotation'])  # quaternion [w, x, y, z]
                            
                            # Project 3D bounding box to 2D
                            corners_2d = project_3d_box_to_2d(
                                center_3d, size_3d, rotation_3d,
                                cam_translation, cam_rotation, camera_intrinsic,
                                ego_translation, ego_rotation, debug=(i == 0 and channel == 'CAM_FRONT')
                            )
                            
                            if corners_2d is not None:
                                # Get category name for labeling
                                category_name = ""
                                if ann.get('category_token'):
                                    category = next((cat for cat in categories if cat['token'] == ann['category_token']), None)
                                    if category:
                                        category_name = category['name']
                                
                                # Draw 3D bounding box
                                draw_3d_box_2d(ax, corners_2d, color='red', linewidth=2)
                                
                                # Calculate and draw 2D bounding box
                                bbox_2d = get_2d_bbox_from_3d_projection(corners_2d)
                                if bbox_2d is not None:
                                    draw_2d_bbox(ax, bbox_2d, category_name, color='green', linewidth=2)
                                
                                boxes_drawn += 1
                                
                        except Exception as e:
                            print(f"    Warning: Failed to project annotation {i}: {e}")
                    
                    # Add annotation info as text
                    ax.text(10, 30, f'Annotations: {len(sample_anns)} (Drawn: {boxes_drawn})', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                           fontsize=12, fontweight='bold')
                else:
                    # Fallback: just show annotation count
                    ax.text(10, 30, f'Annotations: {len(sample_anns)} (No camera intrinsics)', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                           fontsize=12, fontweight='bold')
                
                ax.axis('off')
                
                # Save the visualization
                output_path = os.path.join(output_dir, f'{channel}_sample_{sample_idx}.png')
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                saved_files.append(output_path)
                print(f"  Saved visualization to: {output_path}")
                
            except Exception as e:
                print(f"  Error processing camera {channel}: {str(e)}")
        
        # Create a combined visualization
        if saved_files:
            try:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle(f'NuScenes Sample {sample_idx} - All Camera Views', fontsize=16)
                
                camera_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                               'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                
                for i, channel in enumerate(camera_order):
                    row, col = i // 3, i % 3
                    
                    # Find corresponding saved file
                    matching_file = None
                    for saved_file in saved_files:
                        if channel in saved_file:
                            matching_file = saved_file
                            break
                    
                    if matching_file and os.path.exists(matching_file):
                        img = Image.open(matching_file)
                        axes[row, col].imshow(img)
                        axes[row, col].set_title(channel)
                    else:
                        axes[row, col].text(0.5, 0.5, f'{channel}\nNot Available', 
                                          ha='center', va='center', transform=axes[row, col].transAxes)
                    
                    axes[row, col].axis('off')
                
                combined_path = os.path.join(output_dir, f'combined_sample_{sample_idx}.png')
                plt.savefig(combined_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                print(f"Combined visualization saved to: {combined_path}")
                
            except Exception as e:
                print(f"Error creating combined visualization: {e}")
        
        # Save annotation summary
        summary_path = os.path.join(output_dir, 'annotations.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Sample {sample_idx} Annotations\n")
            f.write(f"Sample token: {sample['token']}\n")
            f.write(f"Timestamp: {sample['timestamp']}\n")
            f.write(f"Camera views: {len(camera_data)}\n")
            f.write(f"Total annotations: {len(sample_anns)}\n\n")
            
            for i, ann in enumerate(sample_anns):
                category_token = instance_to_category.get(ann['instance_token'])
                category_name = category_lookup.get(category_token, 'Unknown')
                f.write(f"Annotation {i+1}:\n")
                f.write(f"  Category: {category_name}\n")
                f.write(f"  Instance token: {ann['instance_token']}\n")
                f.write(f"  Visibility: {ann['visibility_token']}\n")
                f.write(f"  Translation: {ann['translation']}\n")
                f.write(f"  Size: {ann['size']}\n")
                f.write(f"  Rotation: {ann['rotation']}\n\n")
        
        print(f"\nVisualization completed successfully!")
        print(f"Files saved to: {output_dir}")
        print(f"- Individual camera views: {len(saved_files)} files")
        print(f"- Combined view: combined_sample_{sample_idx}.png")
        print(f"- Annotation summary: annotations.txt")
        
    except KeyError as e:
        print(f"Error: Missing key in annotation data - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file - {e}")
    except Exception as e:
        print(f"Unexpected error during visualization: {e}")
        import traceback
        traceback.print_exc()
        
        # Create a combined visualization
        if camera_data:
            print(f"\n🎨 Creating combined visualization...")
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'NuScenes Sample {sample_idx} - All Camera Views', fontsize=16)
            
            camera_positions = [
                (0, 1, 'CAM_FRONT'),
                (0, 0, 'CAM_FRONT_LEFT'), 
                (0, 2, 'CAM_FRONT_RIGHT'),
                (1, 1, 'CAM_BACK'),
                (1, 0, 'CAM_BACK_LEFT'),
                (1, 2, 'CAM_BACK_RIGHT')
            ]
            
            for row, col, channel in camera_positions:
                ax = axes[row, col]
                
                # Find corresponding camera data
                cam_data = None
                for cd in camera_data:
                    if cd['channel'] == channel:
                        cam_data = cd
                        break
                
                if cam_data:
                    image_path = os.path.join(nuscenes_root, cam_data['filename'])
                    if os.path.exists(image_path):
                        try:
                            image = cv2.imread(image_path)
                            if image is not None:
                                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                ax.imshow(image_rgb)
                                ax.set_title(f'{channel}', fontsize=12)
                                ax.axis('off')
                            else:
                                ax.text(0.5, 0.5, f'{channel}\nImage not found', 
                                       ha='center', va='center', transform=ax.transAxes)
                                ax.axis('off')
                        except:
                            ax.text(0.5, 0.5, f'{channel}\nError loading', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.axis('off')
                else:
                    ax.text(0.5, 0.5, f'{channel}\nNo data', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            
            plt.tight_layout()
            combined_output = os.path.join(output_dir, f"sample_{sample_idx}_combined.png")
            plt.savefig(combined_output, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  💾 Saved combined visualization: {combined_output}")
        
        print(f"\n✅ Visualization complete for sample {sample_idx}")
        print(f"📁 All visualizations saved to: {output_dir}")
        
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file - {e}")
    except KeyError as e:
        print(f"Error: Missing key in annotation data - {e}")
    except Exception as e:
        print(f"Error: Unexpected error during visualization - {e}")
        import traceback
        traceback.print_exc()


def suggest_dataset_fixes(diagnosis: Dict[str, Any]) -> None:
    """
    Provide detailed suggestions for fixing common NuScenes dataset issues
    
    Args:
        diagnosis: Dictionary containing diagnosis results from diagnose_dataset_issues
    """
    print("\n" + "="*60)
    print("DATASET FIX SUGGESTIONS")
    print("="*60)
    
    if diagnosis['status'] == 'healthy':
        print("✓ Your dataset appears to be healthy! No fixes needed.")
        return
    
    print("Based on the diagnosis, here are specific steps to fix your dataset:\n")
    
    # Structure issues
    if diagnosis['structure_issues']:
        print("🔧 STRUCTURE ISSUES:")
        for issue in diagnosis['structure_issues']:
            if "Missing directory: samples" in issue:
                print("  1. Missing 'samples' directory:")
                print("     - Download the main NuScenes dataset files")
                print("     - Extract: v1.0-trainval_01.tgz through v1.0-trainval_10.tgz")
                print("     - Ensure extraction creates a 'samples' folder")
                
            elif "Missing directory: sweeps" in issue:
                print("  2. Missing 'sweeps' directory:")
                print("     - Download the sweep files (optional for basic usage)")
                print("     - Extract: v1.0-trainval_blobs.tgz")
                print("     - Note: Sweeps provide temporal context but aren't always required")
                
            elif "Missing directory: v1.0-trainval" in issue:
                print("  3. Missing 'v1.0-trainval' directory:")
                print("     - Download the metadata package: v1.0-trainval_meta.tgz")
                print("     - Extract it to create the annotation files")
                
            elif "Missing sensor directory" in issue:
                sensor = issue.split(": ")[-1]
                print(f"  4. Missing sensor directory '{sensor}':")
                print(f"     - Check if the corresponding data files were extracted properly")
                print(f"     - Verify the sensor data is included in your download")
        print()
    
    # Missing files
    if diagnosis['missing_files']:
        print("📁 MISSING FILES:")
        annotation_files = [f for f in diagnosis['missing_files'] if f.endswith('.json')]
        if annotation_files:
            print("  Missing annotation files:")
            for file_name in annotation_files:
                print(f"    - {file_name}")
            print("  Fix: Download and extract v1.0-trainval_meta.tgz")
        print()
    
    # Data integrity issues
    if diagnosis['data_integrity']:
        print("🔍 DATA INTEGRITY:")
        
        # Check for empty directories
        empty_sensors = [k for k, v in diagnosis['data_integrity'].items() 
                        if v == 0 and 'CAM_' in k]
        if empty_sensors:
            print("  Empty camera directories found:")
            for sensor in empty_sensors:
                print(f"    - {sensor}: No image files")
            print("  Fix: Re-extract the main dataset files")
        
        # Check for missing LiDAR
        if diagnosis['data_integrity'].get('LIDAR_TOP', 0) == 0:
            print("  No LiDAR data found:")
            print("    - LiDAR files should be in samples/LIDAR_TOP/")
            print("    - Check if point cloud files (.pcd) were extracted")
        print()
    
    # Download instructions
    print("📥 DOWNLOAD CHECKLIST:")
    print("  Required files for a complete NuScenes v1.0-trainval dataset:")
    print("  ✓ v1.0-trainval_meta.tgz (metadata/annotations)")
    print("  ✓ v1.0-trainval_01.tgz through v1.0-trainval_10.tgz (sensor data)")
    print("  ○ v1.0-trainval_blobs.tgz (sweeps - optional)")
    print()
    
    # Extraction instructions
    print("📦 EXTRACTION INSTRUCTIONS:")
    print("  1. Create a directory for your dataset (e.g., /data/nuscenes/)")
    print("  2. Extract all .tgz files to the same directory:")
    print("     tar -xzf v1.0-trainval_meta.tgz -C /data/nuscenes/")
    print("     tar -xzf v1.0-trainval_01.tgz -C /data/nuscenes/")
    print("     ... (repeat for all files)")
    print("  3. Verify the final structure:")
    print("     /data/nuscenes/")
    print("     ├── samples/")
    print("     ├── sweeps/")
    print("     └── v1.0-trainval/")
    print()
    
    # Common issues
    print("⚠️  COMMON ISSUES:")
    print("  - Partial downloads: Re-download any corrupted .tgz files")
    print("  - Wrong extraction path: Ensure all files extract to the same root directory")
    print("  - Permission issues: Check read/write permissions on the dataset directory")
    print("  - Disk space: NuScenes requires ~350GB for the full trainval set")
    print()
    
    print("💡 TIP: Run this script again after applying fixes to verify the dataset.")
    print("="*60)

def extract_nuscenes_subset(nuscenes_root: str, num_samples: int, output_dir: str = None) -> str:
    """
    Extract a subset of NuScenes samples with their annotations and create a test folder.
    
    Args:
        nuscenes_root: Path to the full NuScenes dataset
        num_samples: Number of samples to extract
        output_dir: Output directory for the subset (default: nuscenes_root + '_subset_N')
    
    Returns:
        Path to the created zip file
    """
    import shutil
    import zipfile
    from datetime import datetime
    
    print(f"🔄 Extracting {num_samples} samples from NuScenes dataset...")
    
    # Set default output directory
    if output_dir is None:
        output_dir = f"{nuscenes_root}_subset_{num_samples}"
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "sweeps"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "v1.0-trainval"), exist_ok=True)
    
    # Create sensor subdirectories
    for sensor in NUSCENES_STRUCTURE['samples']:
        os.makedirs(os.path.join(output_dir, "samples", sensor), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "sweeps", sensor), exist_ok=True)
    
    # Load annotation files
    annotations_dir = os.path.join(nuscenes_root, "v1.0-trainval")
    
    try:
        with open(os.path.join(annotations_dir, "sample.json"), 'r') as f:
            samples = json.load(f)
        
        with open(os.path.join(annotations_dir, "sample_data.json"), 'r') as f:
            sample_data = json.load(f)
        
        with open(os.path.join(annotations_dir, "sample_annotation.json"), 'r') as f:
            sample_annotations = json.load(f)
            
    except FileNotFoundError as e:
        print(f"❌ Error: Required annotation file not found: {e}")
        return None
    
    # Limit samples to available count
    num_samples = min(num_samples, len(samples))
    selected_samples = samples[:num_samples]
    
    print(f"📊 Processing {num_samples} samples...")
    
    # Collect tokens for selected samples
    selected_sample_tokens = {sample['token'] for sample in selected_samples}
    
    # Find related sample_data entries
    related_sample_data = []
    copied_files = set()
    
    for data_entry in sample_data:
        if data_entry['sample_token'] in selected_sample_tokens:
            related_sample_data.append(data_entry)
            
            # Copy the actual data file
            src_file = os.path.join(nuscenes_root, data_entry['filename'])
            dst_file = os.path.join(output_dir, data_entry['filename'])
            
            if os.path.exists(src_file) and data_entry['filename'] not in copied_files:
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)
                copied_files.add(data_entry['filename'])
    
    # Find related annotations
    related_annotations = []
    for annotation in sample_annotations:
        if annotation['sample_token'] in selected_sample_tokens:
            related_annotations.append(annotation)
    
    print(f"📁 Copied {len(copied_files)} data files")
    print(f"📝 Found {len(related_annotations)} annotations")
    
    # Create filtered annotation files
    filtered_annotations = {
        'sample.json': selected_samples,
        'sample_data.json': related_sample_data,
        'sample_annotation.json': related_annotations
    }
    
    # Copy and filter other annotation files
    for ann_file in REQUIRED_ANNOTATION_FILES:
        src_path = os.path.join(annotations_dir, ann_file)
        dst_path = os.path.join(output_dir, "v1.0-trainval", ann_file)
        
        if ann_file in filtered_annotations:
            # Write filtered data
            with open(dst_path, 'w') as f:
                json.dump(filtered_annotations[ann_file], f, indent=2)
        elif os.path.exists(src_path):
            # Copy full file for other annotations (they're usually small)
            shutil.copy2(src_path, dst_path)
    
    print(f"✅ Subset created in: {output_dir}")
    
    # Create zip file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"nuscenes_subset_{num_samples}samples_{timestamp}.zip"
    zip_path = os.path.join(os.path.dirname(output_dir), zip_filename)
    
    print(f"🗜️  Creating zip archive: {zip_filename}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arc_name)
    
    # Get zip file size
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    
    print(f"✅ Zip archive created: {zip_path}")
    print(f"📦 Archive size: {zip_size_mb:.1f} MB")
    print(f"📊 Contains: {num_samples} samples, {len(related_sample_data)} data files, {len(related_annotations)} annotations")
    
    return zip_path

def main():
    """
    Main function with enhanced NuScenes dataset management
    """
    parser = argparse.ArgumentParser(description="NuScenes Dataset Extraction and Validation Tool")
    parser.add_argument("--zip_dir", default=DEFAULT_ZIP_DIR, 
                       help="Directory containing NuScenes zip files")
    parser.add_argument("--extract_dir", default=DEFAULT_NUSCENES_DIR, 
                       help="Directory to extract files to (will become NuScenes root)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("NUSCENES DATASET MANAGEMENT TOOL")
    print("="*60)
    print(f"Source directory: {args.zip_dir}")
    print(f"Target directory: {args.extract_dir}")
    
    while True:
        print("\nSelect an option:")
        print("1. Extract dataset files")
        print("2. Check extracted folder structure")
        print("3. Count files and check annotations")
        print("4. Run comprehensive dataset diagnosis")
        print("5. Verify NuScenes data structure")
        print("6. Visualize sample with annotations")
        print("7. Get dataset fix suggestions")
        print("8. Complete setup (extract + validate)")
        print("9. Extract subset for testing")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-9): ").strip()
        
        if choice == '1':
            extract_files(args.zip_dir, args.extract_dir)
            
        elif choice == '2':
            check_extracted_folders(args.extract_dir)
            
        elif choice == '3':
            count_files_and_check_annotations(args.extract_dir)
            
        elif choice == '4':
            diagnosis = diagnose_dataset_issues(args.extract_dir)
            
        elif choice == '5':
            check_nuscenes_data_structure(args.extract_dir)
            
        elif choice == '6':
            sample_idx = int(input("Enter sample index to visualize (default 0): ") or "0")
            visualize_sample_with_boxes(args.extract_dir, sample_idx)
            
        elif choice == '7':
            diagnosis = diagnose_dataset_issues(args.extract_dir)
            suggest_dataset_fixes(diagnosis)
            
        elif choice == '8':
            print("\n🚀 Running complete setup and validation...")
            extract_files(args.zip_dir, args.extract_dir)
            check_extracted_folders(args.extract_dir)
            count_files_and_check_annotations(args.extract_dir)
            diagnosis = diagnose_dataset_issues(args.extract_dir)
            check_nuscenes_data_structure(args.extract_dir)
            if diagnosis['status'] != 'healthy':
                suggest_dataset_fixes(diagnosis)
        
        elif choice == '9':
            try:
                num_samples = int(input("Enter number of samples to extract (default 10): ") or "10")
                output_dir = input("Enter output directory (press Enter for default): ").strip() or None
                
                zip_path = extract_nuscenes_subset(args.extract_dir, num_samples, output_dir)
                if zip_path:
                    print(f"\n✅ Subset extraction completed successfully!")
                    print(f"📦 Zip file: {zip_path}")
                else:
                    print("\n❌ Subset extraction failed!")
            except ValueError:
                print("❌ Invalid number of samples. Please enter a valid integer.")
            except Exception as e:
                print(f"❌ Error during subset extraction: {e}")
        
        elif choice == '0':
            print("Exiting program.")
            break
            
        else:
            print("Invalid choice. Please enter a number between 0-9.")

if __name__ == "__main__":
    main()
