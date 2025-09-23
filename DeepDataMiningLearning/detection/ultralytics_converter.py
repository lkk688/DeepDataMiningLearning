"""
Ultralytics YOLO to Custom YOLO Converter

This module provides functionality to convert Ultralytics YOLO models to the custom YOLO
checkpoint format expected by the DeepDataMiningLearning detection framework.
"""

import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_ultralytics_to_custom_yolo(
    ultralytics_model_path: str,
    output_ckpt_path: str,
    model_name: str = "yolov8n",
    num_classes: Optional[int] = None,
    device: str = "cpu"
) -> bool:
    """
    Convert Ultralytics YOLO model to custom YOLO checkpoint format.
    
    Args:
        ultralytics_model_path (str): Path to the Ultralytics YOLO model (.pt file)
        output_ckpt_path (str): Path where the converted checkpoint will be saved
        model_name (str): Name of the YOLO model (e.g., 'yolov8n', 'yolov8s')
        num_classes (int, optional): Number of classes in the model
        device (str): Device to load the model on
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Check if Ultralytics model exists
        if not os.path.exists(ultralytics_model_path):
            logger.error(f"Ultralytics model not found at: {ultralytics_model_path}")
            return False
            
        # Try to import ultralytics
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("Ultralytics package not found. Please install it with: pip install ultralytics")
            return False
            
        logger.info(f"Loading Ultralytics model from: {ultralytics_model_path}")
        
        # Load the Ultralytics model
        ultralytics_model = YOLO(ultralytics_model_path)
        
        # Get the PyTorch model
        pytorch_model = ultralytics_model.model
        
        # Extract state dict
        state_dict = pytorch_model.state_dict()
        
        # Convert state dict keys to match custom YOLO format
        # The custom YOLO expects keys to start with 'model.'
        converted_state_dict = {}
        
        for key, value in state_dict.items():
            # If key doesn't start with 'model.', add the prefix
            if not key.startswith('model.'):
                new_key = f'model.{key}'
            else:
                new_key = key #in here
            converted_state_dict[new_key] = value
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_ckpt_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Save the converted checkpoint
        logger.info(f"Saving converted checkpoint to: {output_ckpt_path}")
        torch.save(converted_state_dict, output_ckpt_path)
        
        # Verify the saved checkpoint
        if os.path.exists(output_ckpt_path):
            # Load and verify the checkpoint
            loaded_ckpt = torch.load(output_ckpt_path, map_location='cpu')
            logger.info(f"Conversion successful! Checkpoint contains {len(loaded_ckpt)} parameters")
            logger.info(f"Sample keys: {list(loaded_ckpt.keys())[:5]}")
            return True
        else:
            logger.error("Failed to save converted checkpoint")
            return False
            
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        return False

def get_ultralytics_model_path(model_name: str, weights_dir: str = "./weights") -> str:
    """
    Get the expected path for an Ultralytics model.
    
    Args:
        model_name (str): Name of the YOLO model (e.g., 'yolov8n')
        weights_dir (str): Directory where weights are stored
        
    Returns:
        str: Path to the Ultralytics model file
    """
    return os.path.join(weights_dir, f"{model_name}.pt")

def get_custom_ckpt_path(model_name: str, ckpt_dir: str = "./checkpoints") -> str:
    """
    Get the expected path for a custom YOLO checkpoint.
    
    Args:
        model_name (str): Name of the YOLO model (e.g., 'yolov8n')
        ckpt_dir (str): Directory where checkpoints are stored
        
    Returns:
        str: Path to the custom checkpoint file
    """
    return os.path.join(ckpt_dir, f"{model_name}_custom.pt")

def ensure_custom_yolo_checkpoint(
    model_name: str,
    ckpt_path: Optional[str] = None,
    weights_dir: str = "./weights",
    ckpt_dir: str = "./checkpoints",
    num_classes: Optional[int] = None,
    device: str = "cpu"
) -> Optional[str]:
    """
    Ensure a custom YOLO checkpoint exists, converting from Ultralytics if necessary.
    
    Args:
        model_name (str): Name of the YOLO model (e.g., 'yolov8n')
        ckpt_path (str, optional): Specific checkpoint path to check/create
        weights_dir (str): Directory where Ultralytics weights are stored
        ckpt_dir (str): Directory where custom checkpoints are stored
        num_classes (int, optional): Number of classes in the model
        device (str): Device to load the model on
        
    Returns:
        str: Path to the custom checkpoint if successful, None otherwise
    """
    # Determine checkpoint path
    if ckpt_path is None:
        ckpt_path = get_custom_ckpt_path(model_name, ckpt_dir)
    
    # Check if custom checkpoint already exists
    if os.path.exists(ckpt_path):
        logger.info(f"Custom checkpoint already exists: {ckpt_path}")
        return ckpt_path
    
    # Look for Ultralytics model to convert
    ultralytics_path = get_ultralytics_model_path(model_name, weights_dir)
    
    if not os.path.exists(ultralytics_path):
        # Try downloading the model using ultralytics
        try:
            from ultralytics import YOLO
            logger.info(f"Downloading {model_name} model...")
            model = YOLO(f"{model_name}.pt")  # This will download if not exists
            # Move to weights directory
            os.makedirs(weights_dir, exist_ok=True)
            if os.path.exists(f"{model_name}.pt"):
                import shutil
                shutil.move(f"{model_name}.pt", ultralytics_path)
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            return None
    
    # Convert Ultralytics model to custom format
    if os.path.exists(ultralytics_path):
        logger.info(f"Converting Ultralytics model to custom format...")
        success = convert_ultralytics_to_custom_yolo(
            ultralytics_path, 
            ckpt_path, 
            model_name, 
            num_classes, 
            device
        )
        
        if success:
            return ckpt_path
        else:
            logger.error("Conversion failed")
            return None
    else:
        logger.error(f"Ultralytics model not found: {ultralytics_path}")
        return None

if __name__ == "__main__":
    # Test the conversion
    model_name = "yolov8n"
    weights_dir = "./weights"
    ckpt_dir = "./checkpoints"
    
    # Ensure directories exist
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Test conversion
    ckpt_path = ensure_custom_yolo_checkpoint(
        model_name=model_name,
        weights_dir=weights_dir,
        ckpt_dir=ckpt_dir,
        num_classes=80,  # COCO classes
        device="cpu"
    )
    
    if ckpt_path:
        print(f"Success! Custom checkpoint available at: {ckpt_path}")
    else:
        print("Conversion failed!")