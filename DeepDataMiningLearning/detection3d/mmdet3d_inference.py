import os
import argparse
from pathlib import Path
import numpy as np

try:
    # Use mmdet3d's high-level inferencers
    from mmdet3d.apis import (
        LidarDet3DInferencer,
        MonoDet3DInferencer,
        MultiModalityDet3DInferencer
    )
except ImportError:
    print("Error: This script requires 'mmdetection3d' and its dependencies.")
    print("Could not import LidarDet3DInferencer, MonoDet3DInferencer, or MultiModalityDet3DInferencer.")
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
    Only used for lidar/multi-modal visualization.
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
    o3d_bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
    
    # Create a LineSet from the bounding box
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_bbox)
    return line_set

def visualize_with_open3d(lidar_file, predictions_dict, out_dir, basename, headless=False):
    """
    Visualizes the point cloud and predicted boxes using Open3D.
    Saves to .ply in headless mode, otherwise shows an interactive window.
    
    Args:
        lidar_file (str): Path to the LiDAR point cloud file.
        predictions_dict (dict): The dictionary containing prediction results,
                                 e.g., from results_dict['predictions'][0]['pred_instances_3d']
    """
    # Load the point cloud
    points = load_lidar_file(lidar_file)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Grey for points
    
    # Get predictions
    # FIX: 'predictions' is now a dict. Get 'bboxes_3d' from it.
    # The data is already a list or numpy array.
    bboxes_list = predictions_dict['bboxes_3d']
    bboxes_tensor = np.array(bboxes_list)
    
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

def build_input_dict_lidar(lidar_file, img_dir, calib_dir):
    """
    Builds the input dictionary for lidar or multi-modal models.
    """
    basename = Path(lidar_file).stem
    
    # Start with the required LiDAR file
    # FIX: The key must be 'points', not 'pts'
    input_dict = {'points': str(lidar_file)}
    
    # --- 1. Find matching image file ---
    img_exts = ['.png', '.jpg', '.jpeg']
    img_file = find_matching_file(basename, img_dir, img_exts)
    if img_dir and img_file:
        input_dict['img'] = img_file
    elif img_dir:
        print(f"Warning: --img-dir provided, but no matching image for {basename} found.")

    # --- 2. Find matching calibration file ---
    calib_exts = ['.txt']
    calib_file = find_matching_file(basename, calib_dir, calib_exts)
    if calib_dir and calib_file:
        input_dict['calib'] = calib_file
    elif calib_dir:
        print(f"Warning: --calib-dir provided, but no matching calib file for {basename} found.")

    return input_dict

def build_input_dict_mono(img_file, calib_dir):
    """
    Builds the input dictionary for monocular models.
    """
    basename = Path(img_file).stem
    input_dict = {'img': str(img_file)}
    
    # Find matching calibration file
    calib_exts = ['.txt'] # KITTI-style
    calib_file = find_matching_file(basename, calib_dir, calib_exts)
    if calib_dir and calib_file:
        input_dict['calib'] = calib_file
    elif calib_dir:
        print(f"Warning: --calib-dir provided, but no matching calib file for {basename} found.")
    
    return input_dict

# --- Defaults ---
DEFAULT_MODEL = '/home/lkk688/Developer/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
#DEFAULT_CHECKPOINT = 'https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_ktti-3d-car_20220331_134606-d42d15ed.pth'
DEFAULT_CHECKPOINT = '/home/lkk688/Developer/mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
DEFAULT_INPUT = '/home/lkk688/Developer/mmdetection3d/demo/data/kitti/000008.bin'
# --- End Defaults ---

def main(args):
    # --- 1. Initialize Model ---
    print(f"Initializing {args.modality} inferencer...")
    
    model_path = args.model
    checkpoint_path = args.checkpoint
    
    # --- New Logic for default checkpoint ---
    if args.model == DEFAULT_MODEL and args.checkpoint is None:
        print(f"Using default model, setting default checkpoint: {DEFAULT_CHECKPOINT}")
        checkpoint_path = DEFAULT_CHECKPOINT
    # --- End New Logic ---
    
    # Handle auto-download for model names
    if not os.path.isfile(args.model):
        # It's a model name, set weights to None for auto-download
        # if checkpoint is not provided
        if not checkpoint_path: # Changed from args.checkpoint
            checkpoint_path = None
        print(f"Loading model '{args.model}' with checkpoint: {checkpoint_path or 'auto-download'}")
    else:
        # It's a config file, checkpoint is required
        if not checkpoint_path: # Changed from args.checkpoint
            print(f"Error: --checkpoint is required when 'model' ({args.model}) is a config file and not the default.")
            exit()
        print(f"Loading local model from config: {args.model}")

    # Select the correct inferencer class
    if args.modality == 'lidar':
        InferencerClass = LidarDet3DInferencer
    elif args.modality == 'mono':
        InferencerClass = MonoDet3DInferencer
    elif args.modality == 'multi-modal':
        InferencerClass = MultiModalityDet3DInferencer
    else:
        # This should be caught by argparse 'choices', but as a safeguard:
        print(f"Error: Unknown modality '{args.modality}'")
        exit()
        
    inferencer = InferencerClass(
        model_path,
        checkpoint_path,
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
    
    if args.modality == 'mono':
        # --- Monocular Inference Path (Images) ---
        if os.path.isfile(args.input_path):
            if args.input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                inputs_list.append(
                    build_input_dict_mono(args.input_path, args.calib_dir)
                )
            else:
                print(f"Error: Monocular mode selected, but input {args.input_path} is not an image.")
                return
        elif os.path.isdir(args.input_path):
            print(f"Scanning image folder: {args.input_path}")
            for fname in sorted(os.listdir(args.input_path)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_file = os.path.join(args.input_path, fname)
                    inputs_list.append(
                        build_input_dict_mono(img_file, args.calib_dir)
                    )
        else:
             print(f"Error: Input path {args.input_path} is not a valid file or directory.")
             return
             
    else:
        # --- LiDAR or Multi-Modal Inference Path (Points) ---
        if os.path.isfile(args.input_path):
            if args.input_path.endswith(('.bin', '.ply', '.pcd')):
                inputs_list.append(
                    build_input_dict_lidar(args.input_path, args.img_dir, args.calib_dir)
                )
            else:
                print(f"Error: {args.modality} mode selected, but input {args.input_path} is not a LiDAR file.")
                return
        elif os.path.isdir(args.input_path):
            print(f"Scanning LiDAR folder: {args.input_path}")
            for fname in sorted(os.listdir(args.input_path)):
                if fname.endswith(('.bin', '.ply', '.pcd')):
                    lidar_file = os.path.join(args.input_path, fname)
                    inputs_list.append(
                        build_input_dict_lidar(lidar_file, args.img_dir, args.calib_dir)
                    )
        else:
             print(f"Error: Input path {args.input_path} is not a valid file or directory.")
             return

    if not inputs_list:
        print("Error: No valid input files found.")
        return
        
        print(f"Found {len(inputs_list)} samples to infer.")

    # --- 3. Run Inference & Visualize ---
    for single_input in inputs_list:
        
        # Get basename from the primary input
        # FIX: The key must be 'points', not 'pts'
        primary_input_key = 'img' if args.modality == 'mono' else 'points'
        basename = Path(single_input[primary_input_key]).stem
        print(f"\nRunning inference on input: {basename}")

        if args.modality == 'mono':
            # --- Monocular Visualization Path ---
            # Use the inferencer's built-in visualizer
            # It will save files to out_dir
            print(f"  > Using built-in visualizer. Saving to {args.out_dir}")
            inferencer(
                single_input,
                show=False,
                out_dir=args.out_dir, # Tell the inferencer where to save
                pred_score_thr=args.score_thr
            )
            
        else:
            # --- LiDAR / Multi-Modal Visualization Path (Custom Open3D) ---
            # FIX: The key must be 'points', not 'pts'
            lidar_file = single_input['points']
            
            results_dict = inferencer(
                single_input,
                show=False,
                out_dir=args.out_dir, # FIX: Pass the real out_dir to avoid NoneType error
                pred_score_thr=args.score_thr
            )
            
            # FIX: The inferencer returns a dict. Access predictions via keys.
            # Get the dictionary for the first (and only) prediction
            # This dict *directly* contains 'bboxes_3d', 'scores_3d', 'labels_3d'
            pred_dict = results_dict['predictions'][0]

            # Save the raw predictions (JSON)
            pred_path = Path(args.out_dir) / f"{basename}_predictions.json"
            print(f"  > Saving raw predictions to {pred_path}")
            
            # The data is already in a serializable format
            pred_data = pred_dict 
            
            try:
                import json
                # Create a serializable copy to handle numpy arrays
                serializable_pred_data = {}
                for k, v in pred_data.items():
                    if isinstance(v, np.ndarray):
                        serializable_pred_data[k] = v.tolist()
                    else:
                        serializable_pred_data[k] = v
                
                with open(pred_path, 'w') as f:
                    json.dump(serializable_pred_data, f, indent=2)
            except Exception as e:
                print(f"  > Warning: Could not save prediction JSON. {e}")

            # Manually visualize using Open3D
            # FIX: Pass the 'pred_dict' to the visualizer
            visualize_with_open3d(
                lidar_file,
                pred_dict,
                args.out_dir,
                basename,
                headless=is_headless
            )
            
    print(f"\nInference complete. Results saved in {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMDetection3D Inference Script")
    
    # Changed from positional to optional with defaults
    parser.add_argument('--model', type=str, 
                        default=DEFAULT_MODEL,
                        help="Model name (e.g., 'pointpillars_kitti') or path to config file.")
    parser.add_argument('--input-path', type=str, 
                        default=DEFAULT_INPUT,
                        help="Path to input. For 'lidar'/'multi-modal', a LiDAR file or folder. For 'mono', an image file or folder.")
    parser.add_argument('--out-dir', type=str, 
                        default='./inference_results',
                        help="Directory to save prediction results and visualizations.")
    
    # Changed required=True to default='lidar'
    parser.add_argument('--modality', type=str, default='lidar',
                        choices=['lidar', 'mono', 'multi-modal'],
                        help="Modality of the model (e.g., 'lidar', 'mono', 'multi-modal').")
    
    # Set default checkpoint to None, logic in main() will handle it
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="(Optional) Path or URL to checkpoint. If 'model' is a name, will auto-download if not provided."
                             f" Defaults to {DEFAULT_CHECKPOINT} if default model is used.")
    
    parser.add_argument('--img-dir', type=str, default=None,
                        help="(Optional) For 'multi-modal', directory of camera images. For 'mono', this is ignored.")
    parser.add_argument('--calib-dir', type=str, default=None,
                        help="(Optional) Directory of calibration files (e.g., KITTI-style .txt).")
    
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help="Score threshold for filtering predictions.")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="Device to use for inference (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument('--headless', default=True,
                        help="Run in headless mode. Will save visualizations to .ply files "
                             "instead of opening an interactive window.")
    
    args = parser.parse_args()
    main(args)