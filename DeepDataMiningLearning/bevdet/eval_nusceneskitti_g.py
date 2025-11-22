import mmcv
import torch
import os
import time
from pathlib import Path
# --- UPDATED IMPORTS ---
from mmengine.config import Config
# This is a guess based on your environment's errors.
# We are trying all common locations.
# REMOVED the MMDataParallel try-except block, as single_gpu_test handles it.

from mmengine.runner import load_checkpoint
# --- END UPDATED IMPORTS ---

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Try to import Open3D, but don't make it a hard requirement
# if the user only wants to run evaluation.
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: 'open3d' package not found. --o3d-vis-dir will not be available.")

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + Open3D Visualization Helpers
# + (Integrated from user request)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def load_lidar_file(lidar_file):
    """Loads a LiDAR point cloud from a .bin file (KITTI/nuScenes format)."""
    if str(lidar_file).endswith('.bin'):
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5) # nuScenes
        if points.shape[1] == 5:
            # nuScenes: x, y, z, intensity, ring_index
            return points
        elif points.shape[1] == 4:
            # KITTI: x, y, z, intensity
            return points
    
    # Fallback for other formats, assuming simple .bin
    try:
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        return points
    except Exception as e:
        print(f"Error loading {lidar_file}: {e}. Trying 5 columns.")
        try:
            points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)
            return points
        except Exception as e2:
            print(f"Error loading {lidar_file} as 4 or 5 columns: {e2}")
            return np.empty((0, 3))


def color_points_by_height(points):
    """Colors points by their Z-coordinate using a turbo colormap."""
    z_values = points[:, 2]
    if len(z_values) == 0:
        return np.empty((0, 3))
        
    z_min, z_max = np.min(z_values), np.max(z_values)
    norm = plt.Normalize(z_min, z_max)
    cmap = get_cmap('turbo')
    colors = cmap(norm(z_values))[:, :3]  # Get RGB, discard alpha
    return colors


def create_open3d_bbox(bbox, color):
    """
    Creates an Open3D LineSet for a 3D bounding box.
    Assumes bbox format: [x, y, z, l, w, h, yaw] (center format)
    """
    x, y, z, l, w, h, yaw = bbox
    center = [x, y, z]
    
    # Create rotation matrix
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([[cos_yaw, -sin_yaw, 0],
                  [sin_yaw,  cos_yaw, 0],
                  [0, 0, 1]])
    
    # 8 corners of the bounding box
    corners_local = np.array([
        [ l/2,  w/2,  h/2],
        [ l/2,  w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [ l/2, -w/2,  h/2],
        [-l/2,  w/2,  h/2],
        [-l/2,  w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2, -w/2,  h/2]
    ])
    
    corners_global = corners_local @ R.T + center
    
    # Define lines
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_global)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set


def get_bbox_center(bbox):
    """Assumes [x, y, z, ...] format."""
    return bbox[:3]


def get_bbox_top_center(bbox):
    """Assumes [x, y, z, l, w, h, ...] format."""
    return [bbox[0], bbox[1], bbox[2] + bbox[5] / 2]


def create_text_label_3d(text, pos, color, size):
    """Stub function for creating a simple dot as a placeholder."""
    # O3D's text rendering is complex; use a simple sphere as a marker
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=size)
    marker.paint_uniform_color(color)
    marker.translate(pos)
    return marker


def create_text_stroke_label(text, pos, color, scale):
    """Stub function for 3D text. Returns an empty geometry."""
    print(f"Info: 3D text label '{text}' visualization is not implemented.")
    return o3d.geometry.LineSet()

def combine_line_sets(line_sets, color=None):
    """Combines a list of O3D LineSets into one."""
    combined_points = []
    combined_lines = []
    combined_colors = []
    point_offset = 0

    for ls in line_sets:
        points = np.asarray(ls.points)
        lines = np.asarray(ls.lines)
        colors = np.asarray(ls.colors)
        
        combined_points.append(points)
        combined_lines.append(lines + point_offset)
        
        if color is not None:
            combined_colors.extend([color for _ in range(len(lines))])
        else:
            combined_colors.extend(colors)
        
        point_offset += len(points)

    if not combined_points:
        return o3d.geometry.LineSet()

    combined_ls = o3d.geometry.LineSet()
    combined_ls.points = o3d.utility.Vector3dVector(np.vstack(combined_points))
    combined_ls.lines = o3d.utility.Vector2iVector(np.vstack(combined_lines))
    combined_ls.colors = o3d.utility.Vector3dVector(combined_colors)
    
    return combined_ls


def draw_projected_boxes_on_image(*args, **kwargs):
    """Stub function for 2D projection."""
    print("Info: 2D projection visualization is not implemented.")
    pass


def visualize_with_open3d(lidar_file, predictions_dict, gt_bboxes, out_dir, basename,
                          headless=False, img_file=None, calib_file=None):
    """
    Visualizes the point cloud and predicted/gt boxes using Open3D with enhanced features.
    Saves to .ply in headless mode, otherwise shows an interactive window.
    
    Args:
        lidar_file: Path to LiDAR point cloud file
        predictions_dict: Dictionary containing prediction results
        gt_bboxes: List of ground truth 3D bounding boxes
        out_dir: Output directory for saving files
        basename: Base name for output files
        headless: Whether to run in headless mode
        img_file: Optional path to corresponding image file
        calib_file: Optional path to calibration file
    """
    # Load the point cloud (N, 4)
    points = load_lidar_file(lidar_file)
    if points.shape[0] == 0:
        print(f"Warning: Could not load or found 0 points in {lidar_file}. Skipping viz.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Color points by height with high contrast colors (blue to red)
    pcd_colors = color_points_by_height(points)
    if pcd_colors.shape[0] == pcd.points.size:
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    
    # Get predicted boxes and labels
    pred_bboxes_list = predictions_dict['bboxes_3d'].tensor.cpu().numpy()
    pred_bboxes_tensor = np.array(pred_bboxes_list)
    
    # Get predicted labels if available
    pred_labels = predictions_dict.get('labels_3d', {}).tensor.cpu().numpy()
    pred_scores = predictions_dict.get('scores_3d', {}).tensor.cpu().numpy()
    
    # Create geometries list starting with point cloud
    geometries = [pcd]
    
    # Add compact coordinate frame at origin (smaller to avoid overflow)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geometries.append(coordinate_frame)
    
    # Create geometries for predicted boxes (Green)
    pred_line_sets = []
    pred_text_line_sets = []
    # Resolve class names if provided in metainfo; fallback to KITTI classes
    metainfo = predictions_dict.get('metainfo', {})
    class_names = metainfo.get('classes', None)
    if class_names is None:
        class_names = ['Car', 'Pedestrian', 'Cyclist'] # Fallback

    for i, bbox in enumerate(pred_bboxes_tensor):
        bbox_lines = create_open3d_bbox(bbox, color=[0.0, 1.0, 0.0])  # Green
        pred_line_sets.append(bbox_lines)
        geometries.append(bbox_lines)
        # Center marker: single green dot for predictions
        center_pos = get_bbox_center(bbox)
        pred_center = create_text_label_3d('', center_pos, color=[0.0, 1.0, 0.0], size=0.14)
        geometries.append(pred_center)

        # Add class label text at top center of box
        cls_id = None
        if isinstance(pred_labels, (list, np.ndarray)) and i < len(pred_labels):
            try:
                cls_id = int(pred_labels[i])
            except Exception:
                cls_id = None
        cls_name = class_names[cls_id] if (cls_id is not None and 0 <= cls_id < len(class_names)) else 'OBJ'
        top_pos = get_bbox_top_center(bbox)
        text_ls = create_text_stroke_label(cls_name, top_pos, color=[1.0, 1.0, 1.0], scale=0.6)
        geometries.append(text_ls)
        pred_text_line_sets.append(text_ls)
    
    # Create geometries for ground truth boxes (Red)
    gt_line_sets = []
    for i, bbox in enumerate(gt_bboxes):
        # GT boxes might be in a different format (e.g., LiDARBox3D)
        # Convert to numpy [x, y, z, l, w, h, yaw]
        if hasattr(bbox, 'tensor'):
             bbox_np = bbox.tensor.cpu().numpy().flatten()
        else:
             bbox_np = np.array(bbox)

        if bbox_np.shape[0] == 7: # Make sure it's a 7-dim box
            bbox_lines = create_open3d_bbox(bbox_np, color=[1.0, 0.0, 0.0])  # Red
            gt_line_sets.append(bbox_lines)
            geometries.append(bbox_lines)
            # Center marker: single red dot for GT
            gt_center = get_bbox_center(bbox_np)
            gt_center_marker = create_text_label_3d('', gt_center, color=[1.0, 0.0, 0.0], size=0.12)
            geometries.append(gt_center_marker)
        else:
            print(f"Warning: Skipping GT box with unexpected shape: {bbox_np.shape}")


    # Generate 2D visualization if image and calibration data are provided
    if img_file and calib_file:
        try:
            img_2d_vis_path = Path(out_dir) / f"{basename}_2d_vis.png"
            draw_projected_boxes_on_image(img_file, calib_file, pred_bboxes_tensor, gt_bboxes, str(img_2d_vis_path), pred_labels=pred_labels, class_names=class_names)
        except Exception as e:
            print(f"  > Warning: Could not generate 2D visualization. {e}")
    
    if headless:
        print(f"  > Headless mode. Saving to .ply files in {out_dir}")
        pcd_file = Path(out_dir) / f"{basename}_points.ply"
        axes_file = Path(out_dir) / f"{basename}_axes.ply"
        pred_bbox_file = Path(out_dir) / f"{basename}_pred_bboxes.ply"
        pred_label_file = Path(out_dir) / f"{basename}_pred_labels.ply"
        gt_bbox_file = Path(out_dir) / f"{basename}_gt_bboxes.ply"
        
        o3d.io.write_point_cloud(str(pcd_file), pcd)
        
        # Save coordinate frame mesh
        o3d.io.write_triangle_mesh(str(axes_file.with_suffix('.ply')), coordinate_frame)
        
        # Save bounding boxes (combine into single LineSet for each group)
        if len(pred_line_sets) > 0:
            combined_pred = combine_line_sets(pred_line_sets, color=[0.0, 1.0, 0.0])
            o3d.io.write_line_set(str(pred_bbox_file), combined_pred)
        if len(gt_line_sets) > 0:
            combined_gt = combine_line_sets(gt_line_sets, color=[1.0, 0.0, 0.0])
            o3d.io.write_line_set(str(gt_bbox_file), combined_gt)
            
        print(f"  > Saved points: {pcd_file}")
        print(f"  > Saved coordinate axes: {axes_file}")
        if len(pred_bboxes_tensor) > 0:
            print(f"  > Saved pred bboxes: {pred_bbox_file}")
        if len(gt_bboxes) > 0:
            print(f"  > Saved gt bboxes: {gt_bbox_file}")
        # Save predicted top text labels in headless mode
        if len(pred_text_line_sets) > 0:
            combined_text = combine_line_sets(pred_text_line_sets, color=[1.0, 1.0, 1.0])
            o3d.io.write_line_set(str(pred_label_file), combined_text)
    else:
        print(f"  > Displaying Open3D visualization for {basename}...")
        print(f"  > Point cloud colored with turbo colormap (rainbow-like, high contrast)")
        print(f"  > Coordinate axes with arrows: X=Red, Y=Green, Z=Blue")
        print(f"  > Predicted boxes: Green, Ground truth boxes: Red")
        print(f"  > Markers: Green pred center, Red GT center, White top text")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Enhanced 3D Visualization: {basename}",
            width=1400,
            height=900
        )

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + Main Script Logic
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def parse_dict_str(arg_str):
    """Helper function to parse key=value strings from command line."""
    arg_dict = {}
    if arg_str:
        for arg in arg_str:
            key, value = arg.split('=', 1)
            # Try to infer type
            if value.lower() == 'true':
                arg_dict[key] = True
            elif value.lower() == 'false':
                arg_dict[key] = False
            elif '.' in value or 'e-' in value:
                try:
                    arg_dict[key] = float(value)
                except ValueError:
                    arg_dict[key] = value
            else:
                try:
                    arg_dict[key] = int(value)
                except ValueError:
                    arg_dict[key] = value
    return arg_dict


def plot_results(metric, dataset_type, save_path):
    """
    Generates and saves a publication-ready bar chart of evaluation results.
    """
    print(f"Attempting to generate results plot... saving to {save_path}")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    class_names = []
    class_aps = []

    if dataset_type == 'nuscenes':
        # nuScenes keys are like 'mAP_car', 'mAP_truck', etc.
        for key, value in metric.items():
            if key.startswith('mAP_') and key != 'mAP':
                class_name = key.replace('mAP_', '').replace('_', ' ').title()
                class_names.append(class_name)
                class_aps.append(value)
        
        if not class_aps:
            print("Could not find nuScenes mAP_... keys in metric dict. Skipping plot.")
            return

        title = 'nuScenes: Mean Average Precision (mAP) by Class'
        ylabel = 'mAP'
        color = 'skyblue'
        rotation = 45

    elif dataset_type == 'kitti':
        # KITTI keys are like 'Car_3d_moderate_mAP', 'Pedestrian_3d_moderate_mAP'
        classes = ['Car', 'Pedestrian', 'Cyclist']
        difficulty = 'moderate'
        
        for class_name in classes:
            key = f'{class_name}_3d_{difficulty}_mAP'
            if key in metric:
                class_names.append(class_name)
                class_aps.append(metric[key])
        
        if not class_aps:
            # Try parsing legacy/other formats
            for class_name in classes:
                 key = f'{class_name}_3d_{difficulty}'
                 if key in metric and isinstance(metric[key], (list, np.ndarray)) and len(metric[key]) > 0:
                    try:
                        class_names.append(class_name)
                        class_aps.append(metric[key][0]) # Assuming mAP is the first value
                    except (TypeError, IndexError):
                        pass # Skip if format is not as expected
            
            if not class_aps:
                print("Could not find KITTI *_3d_moderate_mAP keys in metric dict. Skipping plot.")
                return
            
        title = 'KITTI: mAP (3D, Moderate) by Class'
        ylabel = 'mAP (Moderate)'
        color = 'coral'
        rotation = 0
        
    else:
        print(f"Unknown dataset_type for plotting: {dataset_type}. Skipping plot.")
        return

    try:
        # --- Plotting Logic ---
        plt.figure(figsize=(max(12, len(class_names) * 1.5), 8))
        bars = plt.bar(class_names, class_aps, color=color)
        plt.xlabel('Class', fontsize=14, labelpad=10)
        plt.ylabel(ylabel, fontsize=14, labelpad=10)
        plt.title(title, fontsize=16, pad=20)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval,
                     f'{yval:.3f}', va='bottom', ha='center', fontsize=10) 

        plt.xticks(rotation=rotation, ha='right' if rotation > 0 else 'center', fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, max(class_aps) * 1.2 if class_aps else 1.0)
        plt.tight_layout()
        
        plt.savefig(save_path)
        print(f"Successfully saved results plot to {save_path}")
        plt.close()

    except Exception as e:
        print(f"Error while generating plot: {e}")

import argparse
def main():
    parser = argparse.ArgumentParser(description='MMDet3D Unified Evaluation Script')
    
    # --- Core Arguments ---
    parser.add_argument("--config", type=str, default="work_dirs/mybevfusion7_new/mybevfusion7_crossattnaux_painting.py", help="Path to MMDet3D config .py")
    parser.add_argument("--checkpoint", type=str, default="work_dirs/mybevfusion7_new/epoch_4.pth", help="Path to model checkpoint .pth")
    parser.add_argument(
        '--dataset-type', 
        type=str, 
        required=True, 
        choices=['nuscenes', 'kitti'],
        help='The type of dataset to evaluate.')
    
    # --- Standard Eval Arguments ---
    parser.add_argument('--out', help='output result file in .pkl format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format a submission file without running evaluation')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        help='Key-value pairs (e.g., "jsonfile_prefix=./results/prefix") '
             'to be passed to the dataset.evaluate() method.')

    # --- 2D/BEV Visualization (MMDet3D internal) ---
    parser.add_argument(
        '--show', 
        action='store_true', 
        help='Show 2D/BEV visualization results (requires GUI)')
    parser.add_argument(
        '--show-dir', 
        help='Directory where 2D/BEV visualization results will be saved')
    
    # --- Plotting ---
    parser.add_argument(
        '--plot-dir', 
        help='Directory where result plots (e.g., mAP charts) will be saved')

    # --- Open3D Visualization (New) ---
    parser.add_argument(
        '--o3d-vis-dir',
        help='Directory to save advanced Open3D .ply visualizations. '
             'This triggers a separate visualization loop.')
    parser.add_argument(
        '--o3d-vis-limit',
        type=int,
        default=10,
        help='Limit the number of samples for Open3D visualization.')
    parser.add_argument(
        '--o3d-headless',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Run Open3D visualization in headless mode (saves .ply files). '
             'Use --no-o3d-headless to show an interactive window.')

    args = parser.parse_args()

    # --- 1. Load Config ---
    cfg = Config.fromfile(args.config)
    
    # Ensure test-time settings
    cfg.model.pretrained = None  # We are loading a checkpoint
    
    # --- 2. Model & Dataset Configuration Check ---
    print("\n--- Model & Dataset Configuration Check ---")
    try:
        if 'metainfo' in cfg.data.test and 'classes' in cfg.data.test.metainfo:
            class_names = cfg.data.test.metainfo.classes
            print(f"Config-defined classes: {class_names}")
        elif 'class_names' in cfg.data.test:
             class_names = cfg.data.test.class_names
             print(f"Config-defined classes: {class_names}")
        else:
            print("Warning: Could not find class names in config.")
            class_names = []

        is_nuscenes_config = any('car' in c.lower() for c in class_names) and len(class_names) > 3
        is_kitti_config = any('car' in c.lower() for c in class_names) and len(class_names) <= 3

        if args.dataset_type == 'nuscenes':
            if is_nuscenes_config:
                print("-> Config classes seem to match 'nuscenes' evaluation.")
            elif is_kitti_config:
                print("-> WARNING: Config classes look like KITTI, but you selected 'nuscenes' evaluation.")
            else:
                print("-> Config classes are ambiguous; proceeding with 'nuscenes' evaluation.")
        
        elif args.dataset_type == 'kitti':
            if is_kitti_config:
                 print("-> Config classes seem to match 'kitti' evaluation.")
            elif is_nuscenes_config:
                print("-> WARNING: Config classes look like nuScenes, but you selected 'kitti' evaluation.")
            else:
                print("-> Config classes are ambiguous; proceeding with 'kitti' evaluation.")

    except Exception as e:
        print(f"Warning: Could not perform model check. Error: {e}")
    print("------------------------------------------\n")

    
    # --- 3. Build Dataset and Dataloader ---
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,  # Single-GPU test
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # --- 4. Build Model & Load Checkpoint ---
    print("Building model...")
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # --- 5. Run single_gpu_test ---
    model.cuda()
    model.eval()

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    # REMOVED: model = MMDataParallel(model, device_ids=[0])
    # single_gpu_test will wrap the model internally.

    print("Running evaluation (this may take a while)...")
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)

    total_time = time.time() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    num_samples = len(data_loader.dataset)
    avg_latency_ms = (total_time / num_samples) * 1000

    # --- 6. Process Results ---
    eval_kwargs = parse_dict_str(args.eval_options)
    
    if args.format_only:
        print(f"Formatting results for submission...")
        dataset.format_results(outputs, **eval_kwargs)
        if args.dataset_type == 'nuscenes':
            prefix = eval_kwargs.get('jsonfile_prefix', 'results')
            print(f"Submission JSONs saved with prefix: {prefix}")
        elif args.dataset_type == 'kitti':
            prefix = eval_kwargs.get('submission_prefix', 'results')
            print(f"Submission files saved with prefix: {prefix}")

    if args.out:
        print(f"Saving results to {args.out}...")
        mmcv.dump(outputs, args.out)

    print("Calculating metrics...")
    metric = dataset.evaluate(outputs, **eval_kwargs)
    print("--- Evaluation Results ---")
    print(metric)

    print("\n--- Performance Metrics ---")
    print(f"Peak GPU Memory: {peak_memory_mb:.2f} MB")
    print(f"Total Eval Time: {total_time:.2f} s")
    print(f"Num Samples: {num_samples}")
    print(f"Average Latency: {avg_latency_ms:.2f} ms/sample\n")

    # --- 7. Plotting ---
    if args.plot_dir:
        plot_name = f'{args.dataset_type}_map_by_class.png'
        save_path = os.path.join(args.plot_dir, plot_name)
        plot_results(metric, args.dataset_type, save_path)

    # --- 8. Open3D Visualization Loop ---
    if args.o3d_vis_dir:
        if not OPEN3D_AVAILABLE:
            print("Error: --o3d-vis-dir was specified, but 'open3d' is not installed.")
            print("Please install it ('pip install open3d') and try again.")
        else:
            print(f"\n--- Starting Open3D Visualization ---")
            print(f"Saving results to: {args.o3d_vis_dir}")
            print(f"Visualizing first {args.o3d_vis_limit} samples.")
            os.makedirs(args.o3d_vis_dir, exist_ok=True)
            
            count = 0
            for output, data in zip(outputs, data_loader):
                if count >= args.o3d_vis_limit:
                    break
                
                try:
                    # Re-format 'data' to be device-agnostic (it's on GPU from dataloader)
                    # This is a simplification; a robust way would be to get paths from dataset
                    img_metas = data['img_metas'][0].data[0]
                    
                    lidar_file = img_metas['pts_filename']
                    basename = Path(lidar_file).stem
                    print(f"Visualizing sample {count+1}/{args.o3d_vis_limit}: {basename}")

                    # Get GT Boxes
                    gt_bboxes = data['gt_bboxes_3d'][0].data
                    
                    # Get Image/Calib (if available, for 2D projection stub)
                    img_file = img_metas.get('filename', None)
                    calib_data = img_metas.get('calib', None)

                    # 'output' is the predictions_dict
                    # Add metainfo to output dict for class name resolution
                    output['metainfo'] = {'classes': dataset.CLASSES}

                    visualize_with_open3d(
                        lidar_file,
                        output,
                        gt_bboxes,
                        args.o3d_vis_dir,
                        basename,
                        headless=args.o3d_headless,
                        img_file=img_file,
                        calib_file=calib_data # Pass calib data (even if stubbed)
                    )
                
                except Exception as e:
                    print(f"Error during O3D visualization for sample {count}: {e}")
                    import traceback
                    traceback.print_exc()

                count += 1
            print("--- Open3D Visualization Complete ---")


if __name__ == '__main__':
    main()