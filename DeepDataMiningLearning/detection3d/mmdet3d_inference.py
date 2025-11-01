import os
import argparse
from pathlib import Path
import numpy as np

try:
    # Use mmdet3d's high-level inferencer
    from mmdet3d.apis import Det3DInferencer
except ImportError:
    print("Error: This script requires 'mmdetection3d' and its dependencies.")
    print("Please follow the mmdet3d installation guide:")
    print("https://mmdetection3d.readthedocs.io/en/latest/get_started.html")
    exit()

try:
    import open3d as o3d
except ImportError:
    print("Error: This script requires 'open3d' for visualization.")
    print("Please install it: pip install open3d")
    exit()


def load_lidar_file(file_path):
    """
    Loads a LiDAR file (.bin, .ply, .pcd) and returns (N, 3) points.
    """
    ext = os.path.splitext(file_path)[-1]
    
    if ext == '.bin':
        # Assuming KITTI-style .bin (x, y, z, intensity)
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return points[:, :3]
    elif ext in ['.ply', '.pcd']:
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def create_open3d_bbox(bbox_tensor):
    """
    Converts a 7D mmdet3d bbox tensor (x, y, z, l, w, h, yaw)
    into an open3d.geometry.LineSet for visualization.
    """
    center = bbox_tensor[:3]
    extent = bbox_tensor[3:6] # l, w, h
    yaw = bbox_tensor[6]
    
    # Open3D's rotation matrix is from z-axis
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
    
    # Create an OrientedBoundingBox
    # Note: mmdet3d extent is (l, w, h), o3d extent is (w, l, h) or (l, w, h)
    # Let's assume (l, w, h) matches o3d's (x, y, z) extent
    o3d_bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
    
    # Create a LineSet from the bounding box
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_bbox)
    return line_set

def visualize_with_open3d(lidar_file, predictions, out_dir, basename, headless=False):
    """
    Visualizes the point cloud and predicted boxes using Open3D.
    Saves to .ply in headless mode, otherwise shows an interactive window.
    """
    # Load the point cloud
    points = load_lidar_file(lidar_file)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Grey for points
    
    # Get predictions
    bboxes_tensor = predictions.bboxes_3d.tensor.cpu().numpy()
    
    # Create geometries list
    geometries = [pcd]
    
    # Create one combined LineSet for all boxes
    all_box_lines = o3d.geometry.LineSet()
    for bbox in bboxes_tensor:
        o3d_box_lines = create_open3d_bbox(bbox)
        all_box_lines += o3d_box_lines # Combine LineSets
        
    all_box_lines.paint_uniform_color([1.0, 0.0, 0.0]) # Red for boxes
    geometries.append(all_box_lines)
    
    if headless:
        print(f"  > Headless mode. Saving to .ply files in {out_dir}")
        pcd_file = Path(out_dir) / f"{basename}_points.ply"
        bbox_file = Path(out_dir) / f"{basename}_bboxes.ply"
        
        o3d.io.write_point_cloud(str(pcd_file), pcd)
        o3d.io.write_line_set(str(bbox_file), all_box_lines)
        print(f"  > Saved points: {pcd_file}")
        print(f"  > Saved bboxes: {bbox_file}")
    else:
        print(f"  > Displaying Open3D visualization for {basename}...")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Inference: {basename}"
        )

def find_matching_file(basename, directory, extensions):
    """
    Finds a file in a directory with the same basename and one of the extensions.
    """
    if not directory or not os.path.isdir(directory):
        return None
    
    for ext in extensions:
        file_path = Path(directory) / f"{basename}{ext}"
        if file_path.exists():
            return str(file_path)
    return None

def build_input_dict(lidar_file, img_dir, calib_dir):
    """
    Builds the input dictionary for the inferencer based on provided files.
    """
    basename = Path(lidar_file).stem
    
    # Start with the required LiDAR file
    input_dict = {'pts': str(lidar_file)}
    
    # --- 1. Find matching image file ---
    # Models often expect '.png', but let's check for '.jpg' too
    img_exts = ['.png', '.jpg', '.jpeg']
    img_file = find_matching_file(basename, img_dir, img_exts)
    if img_dir and img_file:
        input_dict['img'] = img_file
    elif img_dir:
        print(f"Warning: --img-dir provided, but no matching image for {basename} found.")

    # --- 2. Find matching calibration file ---
    # KITTI uses .txt, nuScenes/Waymo use .json or .pkl (but those are complex)
    # We'll assume KITTI-style .txt for this general script
    calib_exts = ['.txt']
    calib_file = find_matching_file(basename, calib_dir, calib_exts)
    if calib_dir and calib_file:
        # The key 'calib' is what mmdet3d's KITTI pipeline expects
        input_dict['calib'] = calib_file
    elif calib_dir:
        print(f"Warning: --calib-dir provided, but no matching calib file for {basename} found.")

    return input_dict

def main(args):
    # --- 1. Initialize Model ---
    print("Initializing Det3DInferencer...")
    
    # Check if 'model' is a file path
    if os.path.isfile(args.model):
        if not args.checkpoint:
            print(f"Error: --checkpoint is required when 'model' ({args.model}) is a config file.")
            exit()
        print(f"Loading local model from config: {args.model}")
        inferencer = Det3DInferencer(
            args.model,
            args.checkpoint,
            device=args.device
        )
    else:
        # Assume 'model' is a model name
        if args.checkpoint:
            print(f"Loading model '{args.model}' with local checkpoint: {args.checkpoint}")
            inferencer = Det3DInferencer(
                args.model,
                args.checkpoint,
                device=args.device
            )
        else:
            print(f"Loading model '{args.model}' with auto-downloaded checkpoint.")
            # Pass None to weights to trigger auto-download
            inferencer = Det3DInferencer(
                args.model,
                None, 
                device=args.device
            )

    # Create the output directory if it doesn't exist
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for headless mode
    is_headless = args.headless or not os.environ.get('DISPLAY')
    if is_headless:
        print("Running in headless mode. Visualizations will be saved to files.")
    
    # --- 2. Gather all inputs ---
    inputs_list = []
    
    if os.path.isfile(args.input_path):
        # Single file inference
        if args.input_path.endswith(('.bin', '.ply', '.pcd')):
            inputs_list.append(
                build_input_dict(args.input_path, args.img_dir, args.calib_dir)
            )
        else:
            print(f"Error: Input file {args.input_path} is not a supported LiDAR format.")
            return
            
    elif os.path.isdir(args.input_path):
        # Folder inference
        print(f"Scanning folder: {args.input_path}")
        for fname in sorted(os.listdir(args.input_path)):
            if fname.endswith(('.bin', '.ply', '.pcd')):
                lidar_file = os.path.join(args.input_path, fname)
                inputs_list.append(
                    build_input_dict(lidar_file, args.img_dir, args.calib_dir)
                )
    else:
        print(f"Error: Input path {args.input_path} is not a valid file or directory.")
        return

    if not inputs_list:
        print("Error: No valid input LiDAR files found.")
        return
        
    print(f"Found {len(inputs_list)} samples to infer.")

    # --- 3. Run Inference & Visualize ---
    for single_input in inputs_list:
        lidar_file = single_input['pts']
        basename = Path(lidar_file).stem
        print(f"\nRunning inference on: {lidar_file}")
        
        # Run inference, get prediction dictionary back
        # We set show=False and out_dir=None to stop the inferencer
        # from doing its own visualization. We will do it manually.
        results_dict = inferencer(
            single_input,
            show=False,
            out_dir=None, # We are handling visualization
            pred_score_thr=args.score_thr
        )
        
        # Save the raw predictions (JSON)
        pred_path = Path(args.out_dir) / f"{basename}_predictions.json"
        print(f"  > Saving raw predictions to {pred_path}")
        # The inferencer returns results in 'predictions', let's save them
        # This part is complex as Det3DDataSample is not directly JSON-serializable
        # For simplicity, we'll just save the boxes/scores/labels
        preds = results_dict['predictions'][0].pred_instances_3d
        pred_data = {
            'bboxes_3d': preds.bboxes_3d.tensor.cpu().numpy().tolist(),
            'scores_3d': preds.scores_3d.cpu().numpy().tolist(),
            'labels_3d': preds.labels_3d.cpu().numpy().tolist(),
        }
        # A simple way to save (though not the official format)
        try:
            import json
            with open(pred_path, 'w') as f:
                json.dump(pred_data, f, indent=2)
        except Exception as e:
            print(f"  > Warning: Could not save prediction JSON. {e}")

        # Manually visualize using Open3D
        visualize_with_open3d(
            lidar_file,
            results_dict['predictions'][0].pred_instances_3d,
            args.out_dir,
            basename,
            headless=is_headless
        )
        
    print(f"\nInference complete. Results saved in {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMDetection3D Inference Script")
    parser.add_argument('model', help="Model name (e.g., 'pointpillars_kitti') or path to config file.")
    parser.add_argument('input_path', help="Path to a single LiDAR file (.bin, .ply) or a directory of LiDAR files.")
    parser.add_argument('out_dir', help="Directory to save prediction results and visualizations.")
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="(Optional) Path to checkpoint. Required if 'model' is a config file. "
                             "If 'model' is a name, will auto-download if not provided.")
    parser.add_argument('--img-dir', type=str, default=None,
                        help="(Optional) Directory of camera images for multi-modal models. "
                             "File names must match LiDAR files (e.g., 00001.bin and 00001.png).")
    parser.add_argument('--calib-dir', type=str, default=None,
                        help="(Optional) Directory of calibration files (e.g., KITTI-style .txt). "
                             "File names must match LiDAR files (e.g., 00001.bin and 00001.txt).")
    
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help="Score threshold for filtering predictions.")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="Device to use for inference (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument('--headless', action='store_true',
                        help="Run in headless mode. Will save visualizations to .ply files "
                             "instead of opening an interactive window.")
    
    args = parser.parse_args()
    main(args)