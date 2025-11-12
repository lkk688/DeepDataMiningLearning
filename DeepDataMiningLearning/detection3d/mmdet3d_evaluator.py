import argparse
import os
import sys
import warnings
from pathlib import Path

# --- Dependencies ---
# You must have a full mmdetection3d environment
try:
    import torch
    from torch.utils.data import DataLoader

    import mmengine
    from mmengine.config import Config
    from mmengine.registry import MODELS, DATASETS, EVALUATOR
    from mmengine.runner import Runner
    from mmengine.dataset import DefaultSampler
    
    # We must import the modules to register them in mmengine
    import mmdet3d.datasets
    import mmdet3d.models

    from mmdet3d.structures import LiDARInstance3DBoxes
    from mmdet3d.evaluation import KittiMetric, NuScenesMetric

    # We need to import the visualization tools from run_inference.py
    # This assumes it's in the same directory or accessible in PYTHONPATH
    from mmdet3d_inference2 import (
        load_lidar_file, 
        visualize_with_open3d,
        load_kitti_gt_labels
    )

except ImportError:
    print("Error: This script requires a full 'mmdetection3d' environment.")
    print("Please follow the mmdet3d installation guide.")
    print("https://mmdetection3d.readthedocs.io/en/latest/get_started.html")
    print("It also assumes 'run_inference.py' is in the same directory.")
    exit()

try:
    import open3d as o3d
    import numpy as np
except ImportError:
    print("Error: This script requires 'open3d' and 'numpy'.")
    exit()
    
# Suppress warnings
warnings.filterwarnings("ignore")

def fix_config_paths(cfg, mmdet3d_root):
    """
    Recursively finds and fixes 'data_root', 'info_path', etc.
    This version is more robust and fixes top-level vars first.
    """
    if mmdet3d_root is None:
        print("Warning: mmdet3d_root is None. Cannot fix data paths.")
        return cfg

    mmdet3d_root = Path(mmdet3d_root)
    print(f"Fixing data paths in config relative to: {mmdet3d_root}")

    # --- 1. Fix Top-Level Path Variables ---
    # Configs often define 'data_root = "data/kitti/"' at the top.
    # We must fix these first, so nested paths built from them
    # are correct.
    for key in cfg:
        if isinstance(cfg[key], str):
            # Unconditional fix if it's a relative data path
            if cfg[key].startswith('data/'):
                new_path = mmdet3d_root / cfg[key]
                print(f"  Fixing top-level var: {key} -> {new_path}")
                cfg[key] = str(new_path)

    # --- 2. Fix Nested Paths ---
    # Paths that are relative to mmdet3d_root
    # (e.g., 'data/kitti/' or 'data/kitti/kitti_dbinfos_train.pkl')
    # FIX: Added 'ann_file' to the list of keys
    path_keys_to_fix = ['data_root', 'info_path', 'ann_file']

    def _fix_recursive(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in path_keys_to_fix and isinstance(value, str):
                    # Unconditional fix if it's a relative data path
                    if value.startswith('data/'):
                        new_path = mmdet3d_root / value
                        print(f"  Fixing nested path: {key} -> {new_path}")
                        obj[key] = str(new_path)
                
                # Special check for 'data_prefix' (velodyne vs velodyne_reduced)
                if key == 'data_prefix' and 'pts' in value and mmdet3d_root:
                    # This logic is complex, as it needs the (already fixed) data_root
                    # We'll rely on the user having the correct velodyne folder.
                    # A simple robustness check:
                    if 'velodyne_reduced' in value['pts']:
                        alt_path_str = value['pts'].replace('velodyne_reduced', 'velodyne')
                        # We can't easily check os.path.exists here, as data_root is
                        # not passed down.
                        # print(f"  Note: 'data_prefix' robustness for velodyne/velodyne_reduced is best handled in dataset init.")
                        pass

                _fix_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                _fix_recursive(item)

    _fix_recursive(cfg)
    return cfg


def main(args):
    # --- 1. Load Config ---
    cfg = Config.fromfile(args.config)
    
    # --- 2. Fix Data Paths ---
    mmdet3d_root_path = args.mmdet3d_root
    # If not provided via arg, try env var
    if mmdet3d_root_path is None:
        mmdet3d_root_path = os.environ.get("MMDET3D_ROOT")
    
    mmdet3d_root = None
    if mmdet3d_root_path:
        mmdet3d_root = Path(mmdet3d_root_path)
        if not (mmdet3d_root / 'tools' / 'test.py').exists():
            print(f"Warning: --mmdet3d-root provided '{mmdet3d_root}', but 'tools/test.py' not found inside.")
            print("Data paths may not be fixed correctly.")
    
    if mmdet3d_root:
        print(f"Using mmdetection3d root at: {mmdet3d_root}")
        cfg = fix_config_paths(cfg, mmdet3d_root)
    else:
        print("No mmdetection3d root provided. Assuming data paths in config are correct.")

    # --- 3. Set up Output Directory ---
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # Default to eval_results/<config_name_without_py>
        config_name = Path(args.config).stem
        out_dir = Path(f"eval_results/{config_name}")
        
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluation results will be saved to: {out_dir}")
    if args.vis_samples > 0:
        print(f"Visualizations will be saved to: {vis_dir}")

    # --- 4. Initialize Dataloader ---
    # We use the 'test_dataloader' config, as it's for evaluation
    # (It's often identical to 'val_dataloader')
    if 'test_dataloader' not in cfg:
        print("Error: 'test_dataloader' not found in config. Using 'val_dataloader'.")
        dataloader_cfg = cfg.val_dataloader
    else:
        dataloader_cfg = cfg.test_dataloader

    # Override batch size and workers for this script
    dataloader_cfg.batch_size = args.batch_size
    dataloader_cfg.num_workers = args.num_workers
    dataloader_cfg.sampler = DefaultSampler(dataloader_cfg.dataset, shuffle=False)
    
    print("Building test dataset...")
    # This is where the FileNotFoundError occurs if paths are wrong
    test_dataset = DATASETS.build(dataloader_cfg.dataset)
    
    # Manually set collate_fn from the dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=DefaultSampler(test_dataset, shuffle=False),
        collate_fn=test_dataset.collate_fn
    )
    
    # --- 5. Initialize Model ---
    print("Building model...")
    cfg.model.train_cfg = None # We are in test mode
    
    # Register all mmdet3d models
    try:
        Runner.get_model_from_cfg(cfg) 
    except Exception:
        pass # May already be registered
        
    model = MODELS.build(cfg.model)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # Check if checkpoint is nested (e.g., 'state_dict')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set up device
    device = 'cuda:0' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded and set to 'eval' mode on {device}.")

    # --- 6. Initialize Metric ---
    print("Building evaluator...")
    if 'test_evaluator' not in cfg:
        print("Error: 'test_evaluator' not found in config. Using 'val_evaluator'.")
        evaluator_cfg = cfg.val_evaluator
    else:
        evaluator_cfg = cfg.test_evaluator
        
    evaluator = EVALUATOR.build(evaluator_cfg)
    
    # Give the evaluator the dataset's metainfo
    evaluator.dataset_meta = test_dataset.metainfo

    # --- 7. Run Custom Evaluation Loop ---
    print("Starting evaluation loop...")
    
    # 'data_samples' is the format the evaluator expects
    all_predictions = [] 
    
    num_samples = 0
    max_samples = args.max_samples if args.max_samples else float('inf')
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if num_samples >= max_samples:
                print(f"Reached max_samples ({args.max_samples}). Stopping evaluation.")
                break

            # --- a. Move data to device ---
            # 'inputs' is a dict, 'data_samples' is a list
            inputs = {}
            if 'inputs' in batch:
                 inputs = {k: v.to(device) for k, v in batch['inputs'].items() if torch.is_tensor(v)}
            
            data_samples = [sample.to(device) for sample in batch['data_samples']]
            
            # --- b. Forward pass (predict mode) ---
            # Returns a list of Det3DDataSample objects
            pred_list = model(inputs, data_samples, mode='predict')
            
            # --- c. Store predictions for metric ---
            # We must detach and move to CPU for the evaluator
            for pred in pred_list:
                pred.cpu()
            all_predictions.extend(pred_list)

            # --- d. Save Visualizations ---
            for j in range(len(pred_list)):
                if num_samples < args.vis_samples:
                    print(f"  Saving visualization for sample {num_samples}...")
                    try:
                        # Get the prediction and the corresponding ground truth
                        pred_sample = pred_list[j]
                        gt_sample = data_samples[j].cpu() # Move GT to CPU for file access
                        
                        # Get basename from the input file path
                        # 'pts_path' is a common key, 'img_path' for mono
                        if 'pts_path' in gt_sample:
                            basename = Path(gt_sample.pts_path).stem
                        elif 'img_path' in gt_sample:
                            basename = Path(gt_sample.img_path).stem
                        else:
                            basename = f"sample_{num_samples:06d}"

                        # Get predicted boxes (from tensor)
                        pred_bboxes_3d = pred_sample.pred_instances_3d.bboxes_3d.tensor.numpy()
                        pred_scores = pred_sample.pred_instances_3d.scores_3d.numpy()
                        
                        # Apply score threshold for visualization
                        keep_mask = pred_scores >= args.vis_score_thr
                        pred_bboxes_3d = pred_bboxes_3d[keep_mask]
                        
                        # Get ground truth boxes
                        gt_bboxes_3d = gt_sample.gt_instances_3d.bboxes_3d.tensor.numpy()
                        
                        # --- Re-use visualization logic from run_inference.py ---
                        
                        # 1. Get LiDAR file path
                        lidar_file = gt_sample.pts_path
                        
                        # 2. Create a dictionary of prediction results
                        # This mimics the format our visualizer expects
                        vis_pred_dict = {
                            'bboxes_3d': pred_bboxes_3d,
                            'scores_3d': pred_scores[keep_mask],
                            'labels_3d': pred_sample.pred_instances_3d.labels_3d.numpy()[keep_mask]
                        }
                        
                        # 3. Call the visualizer in headless mode
                        visualize_with_open3d(
                            lidar_file,
                            vis_pred_dict,
                            gt_bboxes_3d, # Pass GT boxes
                            vis_dir,
                            basename,
                            headless=True # Always save to file
                        )
                        
                        # Note: 2D visualization is complex as it requires
                        # finding matching image/calib files, which are not
                        # always loaded by default in the test-time pipeline.
                        
                    except Exception as e:
                        print(f"  > Warning: Failed to save visualization for sample {num_samples}. {e}")
                
                num_samples += 1
                if num_samples >= max_samples:
                    break
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {num_samples} samples...")


    print(f"Evaluation finished. Processed {num_samples} samples.")

    # --- 8. Compute Metrics ---
    print("Computing metrics...")
    
    # The evaluator takes the list of predictions and ground truths
    # We pass 'all_predictions' as it contains the GT in `data_samples`
    metrics = evaluator.evaluate(all_predictions)
    
    print("\n--- Evaluation Results ---")
    print(metrics)
    
    # Save results to a file
    results_file = out_dir / "evaluation_metrics.json"
    try:
        # Convert numpy types to native Python types for JSON
        import json
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                serializable_metrics[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                serializable_metrics[k] = int(v)
            else:
                serializable_metrics[k] = v
                
        with open(results_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        print(f"\nMetrics saved to: {results_file}")
    except Exception as e:
        print(f"\nError saving metrics to JSON: {e}")
        print("Raw metrics dict:")
        print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMDetection3D Custom Evaluation Script")
    
    parser.add_argument('--config', type=str, default="/data/rnd-liu/MyRepo/mmdetection3d/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py", help="Path to the model config file")
    parser.add_argument('--checkpoint', type=str, default="/data/rnd-liu/MyRepo/mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth", help="Path to the checkpoint .pth file")
    
    parser.add_argument('--out-dir', type=str, default="outputs/mm3d_eval",
                        help="Directory to save metrics and visualizations. "
                             "(Default: eval_results/<config_name>)")
    
    parser.add_argument('--max-samples', type=int, default=None,
                        help="Maximum number of samples to evaluate (Default: all)")
    
    parser.add_argument('--vis-samples', type=int, default=10,
                        help="Number of samples to save 3D visualization for (Default: 10)")
    parser.add_argument('--vis-score-thr', type=float, default=0.3,
                        help="Score threshold for visualizing predicted boxes (Default: 0.3)")
                        
    parser.add_argument('--batch-size', type=int, default=1,
                        help="Batch size for data loading (Default: 1)")
    parser.add_argument('--num-workers', type=int, default=2,
                        help="Number of data loader workers (Default: 2)")
    
    parser.add_argument('--cpu-only', action='store_true',
                        help="Run evaluation on CPU only")
    
    parser.add_argument('--mmdet3d-root', type=str, default="/data/rnd-liu/MyRepo/mmdetection3d/",
                        help="Path to your cloned mmdetection3d repository. "
                             "If not set, will check MMDET3D_ROOT env var. "
                             "Used to fix relative data paths.")
                        
    args = parser.parse_args()
    main(args)
    
