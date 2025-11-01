import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse

# --- Dependencies ---
# You must have a full mmdetection3d environment
try:
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet3d.models import build_detector
    from mmdet3d.registry import MODELS
    from mmdet3d.utils import setup_logger
except ImportError:
    print("Error: This script requires a full 'mmdetection3d' environment.")
    print("Please follow the mmdet3d installation guide.")
    print("https://mmdetection3d.readthedocs.io/en/latest/get_started.html")
    exit()

# --- Import Real Data Loaders ---
from data_utils.kitti_loader import get_kitti_dataloader
from data_utils.nuscenes_loader import get_nuscenes_dataloader
from data_utils.waymo_loader import get_waymo_dataloader

# --- Configs ---
MMDET3D_ROOT = os.environ.get("MMDET3D_ROOT", None)
if MMDET3D_ROOT is None:
    print("Warning: MMDET3D_ROOT environment variable not set.")
    print("Assuming config is in './mmdetection3d/configs/...'")
    MMDET3D_ROOT = "./mmdetection3d" # Adjust this path

DATA_ROOT = os.environ.get("DATA_ROOT", None)
if DATA_ROOT is None:
    print("Warning: DATA_ROOT environment variable not set.")
    print("Assuming data is in './data/...'")
    DATA_ROOT = "./data" # Adjust this path

# --- Model Configs ---
MODEL_CONFIGS = {
    'kitti': 'pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py',
    'nuscenes': 'pointpillars/pointpillars_hv_fpn_sbn-all_8xb2-2x_nus-3d.py',
    'waymo': 'pointpillars/pointpillars_hv_secfpn_8xb4-12e_waymoD5-3d-3class.py'
}

def main(args):
    # --- 1. Configuration ---
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = args.epochs
    
    dataset_name = args.dataset
    if dataset_name not in MODEL_CONFIGS:
        print(f"Error: Dataset '{dataset_name}' not supported.")
        print("Choose from: kitti, nuscenes, waymo")
        return

    # --- 2. Create Dataset and DataLoader ---
    print(f"Initializing data loader for: {dataset_name}")
    data_loader_func = {
        'kitti': get_kitti_dataloader,
        'nuscenes': get_nuscenes_dataloader,
        'waymo': get_waymo_dataloader,
    }[dataset_name]
    
    train_loader = data_loader_func(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    # Get metainfo from the dataset object
    metainfo = train_loader.dataset.metainfo

    # --- 3. Initialize Model, Optimizer ---
    config_path = os.path.join(MMDET3D_ROOT, 'configs', MODEL_CONFIGS[dataset_name])
    print(f"Loading config from: {config_path}")
    cfg = Config.fromfile(config_path)

    # --- Modify Config for our Data ---
    cfg.model.metainfo = metainfo
    # We are not using the mmdet3d data preprocessor,
    # as our pipeline's "Pack3DDetInputs" does the work.
    # We must replace it with a simpler one.
    cfg.model.data_preprocessor = {
        'type': 'Det3DDataPreprocessor',
        'voxel': False, # Our pipeline already voxelizes
        'mean_dims': [1.6, 3.9, 1.56], # Example, adjust as needed
        'std_dims': [0.8, 0.8, 0.8],   # Example, adjust as needed
    }
    # For PointPillars, the voxelization is part of the model
    if 'PointPillars' in cfg.model.type:
         cfg.model.data_preprocessor.voxel = True
         cfg.model.data_preprocessor.voxel_layer = cfg.model.voxel_encoder.voxel_layer

    # Build the model
    print("Building model...")
    Runner.get_model_from_cfg(cfg) # Registers mmdet3d models
    model = MODELS.build(cfg.model)

    # Use the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    # --- 4. The Custom Training Loop ---
    print("--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for i, batch in enumerate(train_loader):
            # --- a. Get data and move to device ---
            inputs = {k: v.to(device) for k, v in batch['inputs'].items() if torch.is_tensor(v)}
            data_samples = [sample.to(device) for sample in batch['data_samples']]
            
            # --- b. Forward pass (and loss calculation) ---
            loss_dict = model(inputs, data_samples, mode='loss')
            
            # --- c. Compute total loss ---
            loss = sum(loss_dict.values())
            
            # --- d. Backward pass ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                 print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"*** Epoch {epoch+1} Finished, Average Loss: {avg_loss:.4f} ***\n")

    print("--- Training Finished ---")
    model_save_path = f"{dataset_name}_model_final.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom MMDetection3D Training Loop")
    parser.add_argument('--dataset', type=str, required=True, choices=['kitti', 'nuscenes', 'waymo'],
                        help="Dataset to train on.")
    parser.add_storage_action(
        '--epochs', type=int, default=5, help="Number of epochs to train.")
    parser.add_argument(
        '--batch_size', type=int, default=2, help="Batch size per GPU.")
    parser.add_argument(
        '--num_workers', type=int, default=2, help="Number of data loader workers.")
    
    args = parser.parse_args()
    main(args)