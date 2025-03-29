import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, WeightedRandomSampler
#from torch.cuda.amp import GradScaler, autocast
from torch.amp import autocast
from transformers import ViTModel  # For transformer experts
from collections import defaultdict
import os
import numpy as np
import argparse
import cv2
import random 
from tqdm import tqdm
import json
from pathlib import Path

# Add CUDA memory management configuration
# This needs to be set before any CUDA operations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Check if CUDA is available and initialize properly
if torch.cuda.is_available():
    try:
        # Try initializing CUDA with default settings
        torch.cuda.init()
    except RuntimeError as e:
        print(f"CUDA initialization warning: {e}")
        print("Attempting to continue with CPU...")
        
# --------------------------
# 1. Model Architecture
# --------------------------

class CSPDarknet(nn.Module):
    """CSPDarknet backbone for YOLOv8."""
    def __init__(self, base_channels=64, depth_multiple=1.0):
        super().__init__()
        
        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.SiLU()
        )
        
        # CSP stages with increasing channels and downsampling
        channels = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]
        self.stages = nn.ModuleList()
        
        for i in range(4):
            # Downsample
            downsample = nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.SiLU()
            )
            
            # CSP block
            n_bottlenecks = max(round(3 * depth_multiple), 1) if i > 0 else 1
            stage = nn.Sequential(
                downsample,
                self._make_csp_block(channels[i+1], channels[i+1], n_bottlenecks)
            )
            
            self.stages.append(stage)
    
    def _make_csp_block(self, in_channels, out_channels, n_bottlenecks):
        """Create a CSP (Cross Stage Partial) block with bottlenecks."""
        # Create a custom CSP block with proper forward method
        class CSPBlock(nn.Module):
            def __init__(self, in_channels, out_channels, n_bottlenecks, parent):
                super().__init__()
                self.c_ = out_channels // 2
                
                # Main branch with bottlenecks
                main_layers = []
                main_layers.append(nn.Conv2d(in_channels, self.c_, 1, 1, 0))
                main_layers.append(nn.BatchNorm2d(self.c_))
                main_layers.append(nn.SiLU())
                
                # Add bottleneck blocks
                for _ in range(n_bottlenecks):
                    main_layers.append(parent._make_bottleneck(self.c_))
                
                self.main = nn.Sequential(*main_layers)
                
                # Shortcut branch
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, self.c_, 1, 1, 0),
                    nn.BatchNorm2d(self.c_),
                    nn.SiLU()
                )
                
                # Final fusion conv
                self.fusion = nn.Sequential(
                    nn.Conv2d(2 * self.c_, out_channels, 1, 1, 0),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU()
                )
            
            def forward(self, x):
                # Process main and shortcut branches
                main_out = self.main(x)
                shortcut_out = self.shortcut(x)
                
                # Concatenate along channel dimension
                x = torch.cat([main_out, shortcut_out], dim=1)
                
                # Apply fusion conv
                return self.fusion(x)
        
        # Return an instance of the CSP block
        return CSPBlock(in_channels, out_channels, n_bottlenecks, self)
    
    def _make_bottleneck(self, channels):
        """Create a bottleneck block."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        x = self.stem(x)
        features = [x]
        
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        # Return P3, P4, P5 features
        return features[2:]  # [256, 512, 1024] channels typically

class FPN(nn.Module):
    """Feature Pyramid Network for YOLOv8."""
    def __init__(self):
        super().__init__()
        
        # Top-down pathway
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(1024, 512, 1),  # P5 -> P5'
            nn.Conv2d(512, 256, 1),   # P4 -> P4'
            nn.Conv2d(256, 256, 1),   # P3 -> P3' (added)
        ])
        
        # Additional projection for upsampled features to match channel dimensions
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(512, 256, 1),   # Project P5' to match P4'
            nn.Conv2d(256, 256, 1),   # Project P4' to match P3'
        ])
        
        # Smooth layers
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(512, 512, 3, padding=1),  # P5'' 
            nn.Conv2d(256, 256, 3, padding=1),  # P4''
            nn.Conv2d(256, 256, 3, padding=1),  # P3''
        ])
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, features):
        # features: [P3, P4, P5] with channels [256, 512, 1024]
        p3, p4, p5 = features
        
        # Top-down pathway
        p5_lat = self.lateral_convs[0](p5)  # 1024 -> 512
        p4_lat = self.lateral_convs[1](p4)  # 512 -> 256
        p3_lat = self.lateral_convs[2](p3)  # 256 -> 256
        
        # Project upsampled features to match channel dimensions
        p5_proj = self.proj_convs[0](p5_lat)  # 512 -> 256
        
        # Fusion (with matching dimensions)
        p4_td = p4_lat + self.upsample(p5_proj)  # Both are 256 channels now
        p3_td = p3_lat + self.upsample(p4_td)    # Both are 256 channels
        
        # Smoothing
        p5_out = p5_lat  # No smoothing for P5 to maintain original size
        p4_out = self.smooth_convs[1](p4_td)  # Using index 1 for P4
        p3_out = self.smooth_convs[2](p3_td)  # Using index 2 for P3
        
        return [p3_out, p4_out, p5_out]  # [256, 256, 512] channels
    
class CNNExpert(nn.Module):
    """CNN-based expert with dynamic depth."""
    def __init__(self, in_channels, num_classes, depth=3):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU()
            ]
        layers.append(nn.Conv2d(in_channels, 4 + num_classes, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class TransformerExpert(nn.Module):
    """Vision Transformer expert with custom implementation."""
    def __init__(self, in_channels, num_classes, patch_size=16):
        super().__init__()
        # Project input features to embedding dimension
        self.proj = nn.Conv2d(in_channels, 768, kernel_size=patch_size, stride=patch_size)
        
        # Create a simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, 
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        
        # Position embeddings
        self.register_buffer("position_ids", torch.arange(197).expand((1, -1)))
        self.position_embeddings = nn.Embedding(197, 768)
        
        # Output head
        self.head = nn.Linear(768, 4 + num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize cls token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize position embeddings
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
        
    def forward(self, x):
        # Project features
        x = self.proj(x)  # [B, 768, H/P, W/P]
        
        # Reshape: [B, 768, H/P, W/P] -> [B, H/P*W/P, 768]
        batch_size = x.shape[0]
        x = x.flatten(2).transpose(1, 2)
        
        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        seq_length = x.shape[1]
        position_ids = self.position_ids[:, :seq_length]
        x = x + self.position_embeddings(position_ids)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Use CLS token for classification
        x = x[:, 0]
        
        # Apply classification head
        return self.head(x).unsqueeze(-1).unsqueeze(-1)  # Reshape to [B, C, 1, 1]
    
class ViTExpert(nn.Module):
    """Vision Transformer expert."""
    def __init__(self, in_channels, num_classes, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 768, kernel_size=patch_size, stride=patch_size)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.head = nn.Linear(768, 4 + num_classes)
        
    def forward(self, x):
        # Project features to ViT embedding dimension
        x = self.proj(x)  # [B, 768, H/P, W/P]
        
        # Reshape to match ViT input format
        # First, permute to [B, H/P, W/P, 768]
        x = x.permute(0, 2, 3, 1)
        
        # Create a dummy tensor for pixel_values with the right shape
        # ViT expects [B, 3, H, W] for pixel_values
        # We'll create a dummy tensor and use the ViT's internal processing
        batch_size = x.shape[0]
        dummy_pixels = torch.zeros(batch_size, 3, 224, 224, device=x.device)
        
        # Use the ViT model with the dummy pixels, but extract and use our features
        with torch.no_grad():
            # Get the ViT's embedding layer to process our features
            embeddings = self.vit.embeddings(dummy_pixels)
            
            # Replace the embeddings with our projected features
            # Reshape our features to match the embedding shape
            seq_length = embeddings.shape[1]
            x_flat = x.reshape(batch_size, -1, 768)
            
            # If our sequence length is different, we need to adjust
            if x_flat.shape[1] != seq_length:
                # Pad or truncate to match the expected sequence length
                if x_flat.shape[1] < seq_length:
                    padding = torch.zeros(batch_size, seq_length - x_flat.shape[1], 768, device=x.device)
                    x_flat = torch.cat([x_flat, padding], dim=1)
                else:
                    x_flat = x_flat[:, :seq_length, :]
            
            # Process through the transformer encoder
            encoder_outputs = self.vit.encoder(x_flat)
            x = encoder_outputs.last_hidden_state[:, 0]  # Use CLS token
        
        # Apply classification head
        return self.head(x).unsqueeze(-1).unsqueeze(-1)  # Reshape to [B, C, 1, 1]
       

class MoEHead(nn.Module):
    """Mixed CNN/Transformer expert head."""
    def __init__(self, in_channels, num_classes, num_experts=4, transformer_experts=1):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels  # Store in_channels as an instance variable
        self.experts = nn.ModuleList([
            CNNExpert(in_channels, num_classes, depth=2 + i) 
            for i in range(num_experts - transformer_experts)
        ] + [
            TransformerExpert(in_channels, num_classes)
            for _ in range(transformer_experts)
        ])
        
        # Dynamic gating
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_experts),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Check input dimensions for debugging
        _, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Input channel dimension {C} doesn't match expected in_channels {self.in_channels}")
            
        gate_weights = self.gate(x)
        
        # Get spatial dimensions of input for resizing expert outputs
        _, _, H, W = x.shape
        
        if not self.training:  # Expert choice during inference
            expert_idx = torch.argmax(gate_weights, dim=1)
            output = self.experts[expert_idx](x)
            # Ensure output has correct spatial dimensions
            if output.shape[-2:] != (H, W):
                output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
            return output
        
        # Process each expert and ensure consistent output size
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)
            # Resize if needed to match input spatial dimensions
            if output.shape[-2:] != (H, W):
                output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
            expert_outputs.append(output)
        
        # Now all outputs have the same spatial dimensions
        outputs = torch.stack(expert_outputs)
        return torch.einsum('e...,be->b...', outputs, gate_weights)

# --------------------------
# 2. Traditional YOLOv8 Model
# --------------------------

class YOLOv8Head(nn.Module):
    """Traditional YOLOv8 detection head."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 4 + num_classes, 1)  # Box coords (4) + class probs
        )
    
    def forward(self, x):
        return self.conv(x)

class TraditionalYOLOv8(nn.Module):
    """Traditional YOLOv8 model without MoE."""
    def __init__(self, datasets_config):
        super().__init__()
        # Backbone and FPN (shared)
        self.backbone = CSPDarknet()
        self.neck = FPN()
        
        # Unified class space
        self.all_classes = self._get_unified_classes(datasets_config)
        
        # Traditional YOLOv8 heads
        self.heads = nn.ModuleList([
            YOLOv8Head(256, len(self.all_classes)),  # For p3_out (256 channels)
            YOLOv8Head(256, len(self.all_classes)),  # For p4_out (256 channels)
            YOLOv8Head(512, len(self.all_classes))   # For p5_out (512 channels)
        ])
        
        # Dataset-specific class masks
        self.class_masks = {}
        for name, dataset_info in datasets_config.items():
            if "unified_class_mapping" in dataset_info:
                # Get the unified classes that this dataset can detect
                unified_classes_in_dataset = set(dataset_info["unified_class_mapping"].values())
                # Create mask based on unified classes
                self.class_masks[name] = torch.tensor([cls in unified_classes_in_dataset 
                                                     for cls in self.all_classes])
            else:
                # Fallback for datasets without mapping
                self.class_masks[name] = torch.tensor([cls in dataset_info.get("classes", [])] 
                                                     for cls in self.all_classes)
    
    def _get_unified_classes(self, datasets_config):
        """Create a unified class space from all datasets, combining similar classes."""
        # Define class synonyms/mappings to combine similar classes
        class_synonyms = {
            # Vehicle-related classes
            "vehicle": "vehicle",
            "car": "vehicle",
            "truck": "truck",
            "bus": "bus",
            "motorcycle": "motorcycle",
            "bicycle": "bicycle",
            "trailer": "trailer",
            "construction_vehicle": "construction_vehicle",
            "van": "vehicle",
            "tram": "tram",
            
            # Person-related classes
            "person": "person",
            "pedestrian": "person",
            "person_sitting": "person",
            "cyclist": "cyclist",
            
            # Traffic-related classes
            "traffic_sign": "traffic_sign",
            "traffic_light": "traffic_light",
            "stop_sign": "traffic_sign",
            "sign": "traffic_sign",
            "fire_hydrant": "fire_hydrant",
            "parking_meter": "parking_meter",
            "barrier": "barrier",
            "traffic_cone": "traffic_cone",
            
            # Animals
            "bird": "bird",
            "cat": "cat",
            "dog": "dog",
            "horse": "horse",
            "sheep": "sheep",
            "cow": "cow",
            "elephant": "elephant",
            "bear": "bear",
            "zebra": "zebra",
            "giraffe": "giraffe",
            
            # Other objects
            "boat": "boat",
            "airplane": "airplane",
            "bench": "bench",
            "misc": "misc",
        }
        
        # Collect all classes from all datasets
        all_classes_raw = set()
        for dataset_name, dataset_info in datasets_config.items():
            if "classes" in dataset_info:
                all_classes_raw.update(dataset_info["classes"])
        
        # Map to unified classes using synonyms
        unified_classes = set()
        for cls in all_classes_raw:
            # If class has a defined mapping, use it; otherwise keep as is
            mapped_class = class_synonyms.get(cls, cls)
            unified_classes.add(mapped_class)
        
        # Update class masks in datasets_config to reflect the new unified classes
        for dataset_name, dataset_info in datasets_config.items():
            if "classes" in dataset_info:
                # Create a mapping from original classes to unified classes
                class_mapping = {cls: class_synonyms.get(cls, cls) for cls in dataset_info["classes"]}
                # Store this mapping for later use in creating class masks
                dataset_info["unified_class_mapping"] = class_mapping
        
        return sorted(list(unified_classes))  # Sort for deterministic ordering
    
    def forward(self, x, dataset_name=None):
        features = self.backbone(x)
        fpn_features = self.neck(features)
        
        outputs = [head(feat) for head, feat in zip(self.heads, fpn_features)]
        
        if dataset_name:
            # Apply class mask for dataset-specific training
            mask = self.class_masks[dataset_name].to(x.device)
            
            # Make sure mask has the right shape for broadcasting
            for i, output in enumerate(outputs):
                # Check dimensions to avoid errors
                num_classes = output.shape[-1] - 4  # Subtract 4 for the box coordinates
                
                if num_classes != len(mask):
                    # Fix the mask to match the output
                    if len(mask) < num_classes:
                        # Pad mask with zeros
                        padded_mask = torch.zeros(num_classes, device=mask.device)
                        padded_mask[:len(mask)] = mask
                        mask = padded_mask
                    else:
                        # Truncate mask
                        mask = mask[:num_classes]
                
                # Apply mask to class predictions only (indices 4 and beyond)
                output[..., 4:] *= mask
        
        return outputs

# --------------------------
# 2. Enhanced YOLOv8 Model
# --------------------------

class GeneralizedYOLOv8(nn.Module):
    def __init__(self, datasets_config):
        super().__init__()
        # Backbone and FPN (shared)
        self.backbone = CSPDarknet()
        self.neck = FPN()
        
        # Unified class space
        self.all_classes = self._get_unified_classes(datasets_config) #39
        
        # MoE heads - Fix the channel dimensions to match FPN output
        # The FPN returns [p3_out, p4_out, p5_out] with channels [256, 256, 512]
        self.heads = nn.ModuleList([
            MoEHead(256, len(self.all_classes), num_experts=4, transformer_experts=1),  # For p3_out (256 channels)
            MoEHead(256, len(self.all_classes), num_experts=4, transformer_experts=1),  # For p4_out (256 channels)
            MoEHead(512, len(self.all_classes), num_experts=4, transformer_experts=1)   # For p5_out (512 channels)
        ])
        
        # Dataset-specific class masks
        # self.class_masks = {
        #     name: torch.tensor([cls in datasets_config[name]["classes"] 
        #                       for cls in self.all_classes])
        #     for name in datasets_config
        # }
        # Dataset-specific class masks
        self.class_masks = {}
        for name, dataset_info in datasets_config.items():
            if "unified_class_mapping" in dataset_info:
                # Get the unified classes that this dataset can detect
                unified_classes_in_dataset = set(dataset_info["unified_class_mapping"].values())
                # Create mask based on unified classes
                self.class_masks[name] = torch.tensor([cls in unified_classes_in_dataset 
                                                     for cls in self.all_classes])
            else:
                # Fallback for datasets without mapping
                self.class_masks[name] = torch.tensor([cls in dataset_info.get("classes", [])] )

    # def _get_unified_classes(self, datasets_config):
    #     """Create a unified class space from all datasets, regardless of enabled status."""
    #     all_classes = set()
    #     for dataset_name, dataset_info in datasets_config.items():
    #         # Include classes from all datasets, even if not enabled for training
    #         if "classes" in dataset_info:
    #             all_classes.update(dataset_info["classes"])
    #     return sorted(list(all_classes))  # Sort for deterministic ordering
    
    def _get_unified_classes(self, datasets_config):
        """Create a unified class space from all datasets, combining similar classes."""
        # Define class synonyms/mappings to combine similar classes
        class_synonyms = {
            # Vehicle-related classes
            "vehicle": "vehicle",
            "car": "vehicle",
            "truck": "truck",
            "bus": "bus",
            "motorcycle": "motorcycle",
            "bicycle": "bicycle",
            "trailer": "trailer",
            "construction_vehicle": "construction_vehicle",
            "van": "vehicle",
            "tram": "tram",
            
            # Person-related classes
            "person": "person",
            "pedestrian": "person",
            "person_sitting": "person",
            "cyclist": "cyclist",
            
            # Traffic-related classes
            "traffic_sign": "traffic_sign",
            "traffic_light": "traffic_light",
            "stop_sign": "traffic_sign",
            "sign": "traffic_sign",
            "fire_hydrant": "fire_hydrant",
            "parking_meter": "parking_meter",
            "barrier": "barrier",
            "traffic_cone": "traffic_cone",
            
            # Animals
            "bird": "bird",
            "cat": "cat",
            "dog": "dog",
            "horse": "horse",
            "sheep": "sheep",
            "cow": "cow",
            "elephant": "elephant",
            "bear": "bear",
            "zebra": "zebra",
            "giraffe": "giraffe",
            
            # Other objects
            "boat": "boat",
            "airplane": "airplane",
            "bench": "bench",
            "misc": "misc",
        }
        
        # Collect all classes from all datasets
        all_classes_raw = set()
        for dataset_name, dataset_info in datasets_config.items():
            if "classes" in dataset_info:
                all_classes_raw.update(dataset_info["classes"])
        
        # Map to unified classes using synonyms
        unified_classes = set()
        for cls in all_classes_raw:
            # If class has a defined mapping, use it; otherwise keep as is
            mapped_class = class_synonyms.get(cls, cls)
            unified_classes.add(mapped_class)
        
        # Update class masks in datasets_config to reflect the new unified classes
        for dataset_name, dataset_info in datasets_config.items():
            if "classes" in dataset_info:
                # Create a mapping from original classes to unified classes
                class_mapping = {cls: class_synonyms.get(cls, cls) for cls in dataset_info["classes"]}
                # Store this mapping for later use in creating class masks
                dataset_info["unified_class_mapping"] = class_mapping
        
        return sorted(list(unified_classes))  # Sort for deterministic ordering
    
    def forward(self, x, dataset_name=None):
        features = self.backbone(x)
        fpn_features = self.neck(features)
        
        outputs = [head(feat) for head, feat in zip(self.heads, fpn_features)]
        
        if dataset_name:
            # Apply class mask for dataset-specific training
            mask = self.class_masks[dataset_name].to(x.device)
            
            # Make sure mask has the right shape for broadcasting
            for i, output in enumerate(outputs):
                # Check dimensions to avoid errors
                num_classes = output.shape[-1] - 4  # Subtract 4 for the box coordinates
                
                if num_classes != len(mask):
                    # Instead of warning, let's fix the mask to match the output
                    if len(mask) < num_classes:
                        # Pad mask with zeros
                        padded_mask = torch.zeros(num_classes, device=mask.device)
                        padded_mask[:len(mask)] = mask
                        mask = padded_mask
                    else:
                        # Truncate mask
                        mask = mask[:num_classes]
                
                # Apply mask to class predictions only (indices 4 and beyond)
                output[..., 4:] *= mask
        
        return outputs

# --------------------------
# 3. Complete Training Script
# --------------------------
from torch.utils.data import Dataset
class KITTIDetectionDataset(Dataset):
    """Dataset for KITTI object detection."""
    
    def __init__(self, root_dir, transforms=None, class_map=None):
        """
        Args:
            root_dir (string): Directory with KITTI data (images and labels)
            transforms (callable, optional): Optional transforms to be applied on a sample
            class_map (dict, optional): Mapping from KITTI class names to model class indices
        """
        self.root_dir = root_dir
        self.transforms = transforms if transforms is not None else []
        self.class_map = class_map if class_map is not None else {}
        
        # KITTI directory structure
        self.image_dir = os.path.join(root_dir, "image_2")
        self.label_dir = os.path.join(root_dir, "label_2")
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                  if f.endswith('.png') or f.endswith('.jpg')])
        
        # Cache class counts for balanced sampling
        self._cache_class_counts()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image file path
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get corresponding label file
        label_file = os.path.join(self.label_dir, img_name.replace('.png', '.txt').replace('.jpg', '.txt'))
        
        # Parse labels
        boxes = []
        labels = []
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:  # KITTI format has at least 9 parts
                        continue
                    
                    obj_type = parts[0]
                    # Skip objects that are not in our class map
                    if obj_type not in self.class_map:
                        continue
                    
                    # KITTI format: [left, top, right, bottom] in pixel coordinates
                    bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                    
                    # Add to our lists
                    boxes.append(bbox)
                    labels.append(self.class_map[obj_type])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'orig_size': torch.as_tensor([image.shape[0], image.shape[1]]),
        }
        
        # Apply transforms
        for transform in self.transforms:
            image, target = transform(image, target)
        
        return image, target
    
    def _cache_class_counts(self):
        """Cache the count of each class for balanced sampling."""
        self.class_counts = defaultdict(int)
        self.sample_to_class = {}
        
        for idx, img_name in enumerate(self.image_files):
            label_file = os.path.join(self.label_dir, img_name.replace('.png', '.txt').replace('.jpg', '.txt'))
            
            # Default class for empty samples
            primary_class = -1  # -1 represents "no objects"
            
            if os.path.exists(label_file):
                class_counts = defaultdict(int)
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 9:
                            continue
                        
                        obj_type = parts[0]
                        if obj_type in self.class_map:
                            class_idx = self.class_map[obj_type]
                            class_counts[class_idx] += 1
                
                # Determine primary class (most frequent)
                if class_counts:
                    primary_class = max(class_counts.items(), key=lambda x: x[1])[0]
            
            # Store primary class for this sample
            self.sample_to_class[idx] = primary_class
            self.class_counts[primary_class] += 1
    
    def get_class_counts(self):
        """Get counts of each class for balanced sampling."""
        return [self.class_counts[i] for i in range(max(self.class_counts.keys()) + 1)]
    
    def get_sample_class(self, idx):
        """Get the primary class for a sample."""
        return self.sample_to_class[idx]
    
class MultiDatasetLoss(nn.Module):
    """Handles different class sets across datasets."""
    def __init__(self, all_classes, dataset_metadata):
        super().__init__()
        self.all_classes = all_classes
        self.dataset_metadata = dataset_metadata
        
    def forward(self, preds, targets, dataset_name):
        # Get class mapping for this dataset
        class_map = {
            cls: i for i, cls in enumerate(self.dataset_metadata[dataset_name]["classes"])
        }
        
        # Compute loss only for available classes
        total_loss = 0
        batch_size = len(targets)
        
        for pred in preds:
            batch_loss = 0
            
            # Process each image in the batch separately
            for i in range(batch_size):
                target = targets[i]  # Get target for this image
                
                # Skip if no boxes
                if len(target["boxes"]) == 0:
                    continue
                    
                # Get predictions for this image
                # Assuming pred has shape [batch_size, H, W, 4+num_classes]
                # Extract the predictions for this image
                img_pred = pred[i]
                
                # Bounding box loss (always present)
                # Reshape predictions to match target boxes
                # This assumes img_pred contains predictions for each grid cell
                # We need to extract predictions for the cells that contain objects
                
                # For simplicity, let's use the first prediction for each target box
                # In a real implementation, you'd match predictions to ground truth boxes
                # The shape mismatch is happening here - we need to ensure pred_boxes and target boxes have compatible shapes
                
                # Get the shape of the prediction tensor
                pred_shape = img_pred.shape
                
                # Extract box predictions and reshape them to match target boxes
                # Assuming img_pred has shape [H, W, 4+num_classes] or similar
                # We need to reshape it to match the target boxes which are [N, 4]
                # where N is the number of ground truth boxes
                
                # First, reshape the predictions to a 2D tensor where each row represents a prediction
                flattened_pred = img_pred.reshape(-1, pred_shape[-1])
                
                # Extract the box coordinates (first 4 elements of each prediction)
                all_pred_boxes = flattened_pred[:, :4]
                
                # For now, just use the first N predictions for N ground truth boxes
                # In a real implementation, you'd use IoU matching or similar
                num_gt_boxes = len(target["boxes"])
                if len(all_pred_boxes) >= num_gt_boxes:
                    pred_boxes = all_pred_boxes[:num_gt_boxes]
                    box_loss = F.smooth_l1_loss(pred_boxes, target["boxes"])
                else:
                    # If we have fewer predictions than ground truth boxes, pad the predictions
                    padding = torch.zeros(num_gt_boxes - len(all_pred_boxes), 4, device=all_pred_boxes.device)
                    pred_boxes = torch.cat([all_pred_boxes, padding], dim=0)
                    box_loss = F.smooth_l1_loss(pred_boxes, target["boxes"])
                
                # Classification loss (masked)
                # Similarly, reshape the class predictions
                all_cls_pred = flattened_pred[:, 4:]
                
                # Use the same indices as for the boxes
                if len(all_cls_pred) >= num_gt_boxes:
                    cls_pred = all_cls_pred[:num_gt_boxes]
                else:
                    # Pad if needed
                    cls_padding = torch.zeros(num_gt_boxes - len(all_cls_pred), all_cls_pred.shape[1], 
                                             device=all_cls_pred.device)
                    cls_pred = torch.cat([all_cls_pred, cls_padding], dim=0)
                
                # Initialize cls_target with zeros
                cls_target = torch.zeros_like(cls_pred)
                
                for j, cls_idx in enumerate(target["labels"]):
                    if cls_idx.item() in class_map:
                        cls_target[j, class_map[cls_idx.item()]] = 1.0
                
                cls_loss = F.binary_cross_entropy_with_logits(cls_pred, cls_target)
                batch_loss += box_loss + cls_loss
            
            # Average loss over batch
            if batch_size > 0:
                batch_loss /= batch_size
            
            total_loss += batch_loss
        
        return {"total_loss": total_loss}
    
class YOLOv8Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])

        # Model selection 
        if config["model_type"]=="moe":
            self.model = GeneralizedYOLOv8(config["datasets"]).to(self.device)
        else:
            self.model = TraditionalYOLOv8(config["datasets"]).to(self.device)
        # Setup training components (optimizer, scheduler, etc.)
        self._setup_training()
        
        # Data loaders
        self.train_loaders = self._create_dataloaders("train")
        self.val_loaders = self._create_dataloaders("val")
        
        # Initialize training state
        self.start_epoch = 0
        self.best_val_map = 0.0
        
        # Resume from checkpoint if specified
        if config.get("resume_from", None):
            self._load_checkpoint(config["resume_from"])

    def _setup_training(self):
        """Set up training components."""
        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Set up learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Set up gradient scaler for mixed precision training
        # Update GradScaler initialization to use the new format
        if hasattr(torch.amp, 'GradScaler'):  # Check if the new API is available
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.config["amp"])
        else:
            # Fallback to old API for backward compatibility
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.config["amp"])
        
        # Set up loss function
        #self.loss_fn = YOLOLoss(self.model.all_classes)
        # Set up loss function
        self.loss_fn = MultiDatasetLoss(self.model.all_classes, self.config["datasets"])
    
    def _create_optimizer(self):
        """Create optimizer based on config."""
        if self.config["optimizer"] == "adamw":
            return AdamW(self.model.parameters(), lr=self.config["lr"], 
                         weight_decay=self.config["weight_decay"])
        else:
            return SGD(self.model.parameters(), lr=self.config["lr"], 
                       momentum=0.9, weight_decay=self.config["weight_decay"])
    
    def _create_scheduler(self):
        """Create learning rate scheduler based on config."""
        if self.config["scheduler"] == "cosine":
            return CosineAnnealingLR(
                self.optimizer, T_max=self.config["epochs"], eta_min=1e-6)
        else:
            # Calculate total steps based on all train loaders
            total_steps = sum(len(loader) for loader in self.train_loaders.values()) * self.config["epochs"]
            return OneCycleLR(
                self.optimizer, max_lr=self.config["lr"], total_steps=total_steps)
    
    def _save_checkpoint(self, epoch, val_metrics=None, is_best=False):
        """Save training checkpoint.
        
        Args:
            epoch: Current epoch number
            val_metrics: Validation metrics dictionary
            is_best: Whether this is the best model so far
        """
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'best_val_map': self.best_val_map,
        }
        
        if val_metrics:
            checkpoint['val_metrics'] = val_metrics
        
        # Save regular checkpoint
        model_name = self.config.get("model_name", "yolomoe")
        checkpoint_path = checkpoint_dir / f"{model_name}_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save latest checkpoint (for resuming)
        latest_path = checkpoint_dir / f"{model_name}_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_path = checkpoint_dir / f"{model_name}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved to {best_path}")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Load checkpoint on CPU to avoid GPU memory issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Move optimizer state to correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Set starting epoch and best validation score
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_map = checkpoint.get('best_val_map', 0.0)
        
        print(f"Resumed training from epoch {self.start_epoch}")
        print(f"Previous best mAP: {self.best_val_map:.4f}")
         
    def _create_dataloaders(self, split):
        """Create dataloaders for each dataset in the configuration.
        
        Args:
            split: Either 'train' or 'val'
            
        Returns:
            Dictionary mapping dataset names to their respective DataLoaders
        """
        loaders = {}
        
        for dataset_name, dataset_config in self.config["datasets"].items():
            # Skip datasets that are not enabled
            if not dataset_config.get("enabled", True):
                continue
                
            # Get dataset path
            dataset_path = dataset_config["path"]
            
            # Create dataset-specific transforms
            if split == "train":
                transforms = self._get_train_transforms(dataset_name)
            else:
                transforms = self._get_val_transforms(dataset_name)
            
            # Create dataset based on type
            if dataset_name == "coco":
                dataset = self._create_coco_dataset(dataset_path, split, transforms)
            elif dataset_name == "kitti":
                dataset = self._create_kitti_dataset(dataset_path, split, transforms)
            elif dataset_name == "waymo":
                dataset = self._create_waymo_dataset(dataset_path, split, transforms)
            elif dataset_name == "argo":
                dataset = self._create_argo_dataset(dataset_path, split, transforms)
            elif dataset_name == "nuscenes":
                dataset = self._create_nuscenes_dataset(dataset_path, split, transforms)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
            
            # Create sampler for balanced training
            if split == "train" and self.config.get("balanced_sampling", False):
                sampler = self._create_balanced_sampler(dataset, dataset_name)
            else:
                sampler = None
            
            # Create dataloader
            shuffle = (split == "train" and sampler is None)
            loaders[dataset_name] = DataLoader(
                dataset,
                batch_size=self.config["batch_size"],
                shuffle=shuffle,
                sampler=sampler,
                num_workers=self.config.get("num_workers", 4),
                pin_memory=True,
                collate_fn=self._collate_fn
            )
        
        return loaders
    
    def _get_train_transforms(self, dataset_name):
        """Get dataset-specific training transforms."""
        # Base transforms for all datasets
        transforms = [
            # Apply augmentations first
            lambda img, targets: self._random_horizontal_flip(img, targets, p=0.5),
            lambda img, targets: self._random_scale(img, targets, scale_range=(0.8, 1.2)),
            #lambda img, targets: (self._color_jitter(img), targets),
            
            # Dataset-specific augmentations
            *([lambda img, targets: self._random_crop(img, targets, p=0.3)] if dataset_name == "coco" else []),
            *([lambda img, targets: self._random_weather_sim(img, targets, p=0.3)] 
              if dataset_name in ["kitti", "waymo", "argo", "nuscenes"] else []),
            
            # Always apply resize as the final transform to ensure consistent size
            lambda img, targets: self._resize_sample(img, targets, self.config.get("input_size", 640)),
        ]
        
        return transforms
    
    def _get_val_transforms(self, dataset_name):
        """Get dataset-specific validation transforms."""
        # Only resize for validation
        return [
            lambda img, targets: self._resize_sample(img, targets, self.config.get("input_size", 640))
        ]
    
    def _create_coco_dataset(self, path, split, transforms):
        """Create COCO dataset."""
        from pycocotools.coco import COCO
        
        # Map COCO classes to unified class space
        class_map = {
            coco_id: self.model.all_classes.index(cls_name)
            for coco_id, cls_name in self.config["datasets"]["coco"]["class_map"].items()
            if cls_name in self.model.all_classes
        }
        
        # Create custom COCO dataset with transforms
        anno_file = f"{path}/annotations/instances_{split}2017.json"
        img_dir = f"{path}/{split}2017"
        
        # Return dataset with transforms and class mapping
        return COCODetectionDataset(
            img_dir, anno_file, transforms=transforms, class_map=class_map
        )
    
    def _create_kitti_dataset(self, path, split, transforms):
        """Create KITTI dataset."""
        # Get the unified class mapping
        unified_mapping = self.config["datasets"]["kitti"].get("unified_class_mapping", {})
        
        # Map KITTI classes to unified class space
        class_map = {}
        for kitti_cls, cls_name in self.config["datasets"]["kitti"]["class_map"].items():
            # Map to unified class if available
            unified_cls = unified_mapping.get(cls_name, cls_name)
            if unified_cls in self.model.all_classes:
                class_map[kitti_cls] = self.model.all_classes.index(unified_cls)
        
        # Return dataset with transforms and class mapping
        if split == "train":
            return KITTIDetectionDataset(
                f"{path}/training", transforms=transforms, class_map=class_map
            )
        else:
            return KITTIDetectionDataset(
                f"{path}/testing", transforms=transforms, class_map=class_map
            )
    
    def _create_waymo_dataset(self, path, split, transforms):
        """Create Waymo dataset."""
        # Map Waymo classes to unified class space
        class_map = {
            waymo_cls: self.model.all_classes.index(cls_name)
            for waymo_cls, cls_name in self.config["datasets"]["waymo"]["class_map"].items()
            if cls_name in self.model.all_classes
        }
        
        # Return dataset with transforms and class mapping
        return WaymoDetectionDataset(
            f"{path}/{split}", transforms=transforms, class_map=class_map
        )
    
    def _create_argo_dataset(self, path, split, transforms):
        """Create Argoverse dataset."""
        # Map Argoverse classes to unified class space
        class_map = {
            argo_cls: self.model.all_classes.index(cls_name)
            for argo_cls, cls_name in self.config["datasets"]["argo"]["class_map"].items()
            if cls_name in self.model.all_classes
        }
        
        # Return dataset with transforms and class mapping
        return ArgoDetectionDataset(
            f"{path}/{split}", transforms=transforms, class_map=class_map
        )
    
    def _create_nuscenes_dataset(self, path, split, transforms):
        """Create nuScenes dataset."""
        # Map nuScenes classes to unified class space
        class_map = {
            nuscenes_cls: self.model.all_classes.index(cls_name)
            for nuscenes_cls, cls_name in self.config["datasets"]["nuscenes"]["class_map"].items()
            if cls_name in self.model.all_classes
        }
        
        # Return dataset with transforms and class mapping
        return NuScenesDetectionDataset(
            f"{path}/{split}", transforms=transforms, class_map=class_map
        )
    
    def _create_balanced_sampler(self, dataset, dataset_name):
        """Create a weighted sampler to balance classes."""
        # Get class weights based on dataset statistics
        class_counts = dataset.get_class_counts()
        
        # Calculate weights (inverse frequency)
        weights = [1.0 / max(count, 1) for count in class_counts]
        
        # Assign weights to samples
        sample_weights = [weights[dataset.get_sample_class(i)] for i in range(len(dataset))]
        
        # Create weighted sampler
        return WeightedRandomSampler(
            sample_weights, 
            num_samples=len(dataset),
            replacement=True
        )
    
    # def _collate_fn(self, batch):
    #     """Custom collate function to handle variable sized objects."""
    #     # Images should already be tensors of the same size from the transforms
    #     images = torch.stack([item[0] for item in batch])
    #     targets = [item[1] for item in batch]
    #     return images, targets
    # def _collate_fn(self, batch):
    #     """Custom collate function to handle variable sized objects."""
    #     # Convert NumPy arrays to tensors if needed
    #     images = [torch.from_numpy(item[0]).permute(2, 0, 1).float() if isinstance(item[0], np.ndarray) 
    #              else item[0] for item in batch]
    #     images = torch.stack(images)
    #     targets = [item[1] for item in batch]
    #     return images, targets
    def _collate_fn(self, batch):
        """Custom collate function to handle variable sized objects."""
        # Ensure all images have the same size
        input_size = self.config.get("input_size", 640)
        
        # Resize any images that might have different dimensions
        processed_batch = []
        for img, target in batch:
            if isinstance(img, torch.Tensor):
                # Check if resize is needed
                if img.shape[-2:] != (input_size, input_size):
                    img, target = self._resize_sample(img, target, input_size)
            processed_batch.append((img, target))
        
        # Now stack the images
        images = torch.stack([item[0] for item in processed_batch])
        targets = [item[1] for item in processed_batch]
        
        return images, targets
    
    # Image transformation utility methods
    def _resize_sample(self, image, targets, size):
        """Resize image and adjust bounding box coordinates."""
        # Get original dimensions
        if isinstance(image, torch.Tensor):
            c, h, w = image.shape
        else:
            h, w = image.shape[:2]
        
        # Resize image
        if isinstance(image, torch.Tensor):
            # Use interpolate for tensor images
            image = F.interpolate(
                image.unsqueeze(0),  # Add batch dimension
                size=(size, size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension
        else:
            # Use cv2 for numpy arrays
            image = cv2.resize(image, (size, size))
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Adjust bounding box coordinates if there are any
        if 'boxes' in targets and len(targets['boxes']) > 0:
            # Scale factors
            scale_x, scale_y = size / w, size / h
            
            # Scale boxes
            boxes = targets['boxes'].clone()
            boxes[:, 0] *= scale_x  # x1
            boxes[:, 1] *= scale_y  # y1
            boxes[:, 2] *= scale_x  # x2
            boxes[:, 3] *= scale_y  # y2
            
            targets['boxes'] = boxes
        
        # Update original size in targets
        if 'orig_size' in targets:
            targets['orig_size'] = torch.tensor([h, w])
        
        return image, targets
    
    def _random_horizontal_flip(self, image, targets, p=0.5):
        """Randomly flip image horizontally and adjust bounding boxes."""
        if random.random() < p:
            # Flip image horizontally
            if isinstance(image, torch.Tensor):
                image = image.flip(-1)  # Flip the last dimension (width)
            else:
                image = cv2.flip(image, 1)  # 1 for horizontal flip
            
            # Flip bounding boxes if they exist
            if 'boxes' in targets and len(targets['boxes']) > 0:
                h, w = image.shape[-2:] if isinstance(image, torch.Tensor) else image.shape[:2]
                boxes = targets['boxes'].clone()
                # Flip x-coordinates: new_x = width - old_x
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]  # Swap and invert x1, x2
                targets['boxes'] = boxes
        
        return image, targets
    
    def _random_scale(self, image, targets, scale_range=(0.8, 1.2)):
        """Randomly scale image and adjust bounding boxes."""
        # Choose random scale factor
        scale = random.uniform(scale_range[0], scale_range[1])
        
        if isinstance(image, torch.Tensor):
            # Get original dimensions
            c, h, w = image.shape
            
            # Calculate new dimensions
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize image using interpolate
            image = F.interpolate(
                image.unsqueeze(0),  # Add batch dimension
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension
        else:
            # Get original dimensions
            h, w = image.shape[:2]
            
            # Calculate new dimensions
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize image
            image = cv2.resize(image, (new_w, new_h))
        
        # Scale bounding boxes if they exist
        if 'boxes' in targets and len(targets['boxes']) > 0:
            boxes = targets['boxes'].clone()
            boxes *= scale  # Scale all coordinates
            targets['boxes'] = boxes
        
        # Update original size in targets
        if 'orig_size' in targets:
            targets['orig_size'] = torch.tensor([new_h, new_w])
        
        return image, targets
    
    def _color_jitter(self, image):
        """Apply color jittering to image."""
        # Define jitter parameters
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        hue = 0.1  # Use a positive value for the range [-hue, hue]
        
        if isinstance(image, torch.Tensor):
            # Convert to PIL for torchvision transforms
            from torchvision import transforms
            from PIL import Image
            import numpy as np
            
            # Convert tensor to PIL
            if image.dim() == 3:  # CHW format
                image_np = image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            else:
                image_np = image.cpu().numpy()
            
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            
            # Apply color jitter
            color_jitter = transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue  # This will use the range [-hue, hue]
            )
            image_pil = color_jitter(image_pil)
            
            # Convert back to tensor
            image = transforms.ToTensor()(image_pil)
        else:
            # Apply color jitter to numpy array
            # Convert to HSV for easier manipulation
            image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Apply jitter
            # For numpy implementation, we can still use the random range directly
            hue_shift = random.uniform(-0.1, 0.1) * 180  # Scale to degrees for HSV
            image_hsv[:, :, 0] = np.clip(image_hsv[:, :, 0] + hue_shift, 0, 180)  # Hue
            image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] * saturation, 0, 1)  # Saturation
            image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2] * brightness, 0, 1)  # Value/Brightness
            
            # Convert back to RGB
            image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
            
            # Apply contrast
            image = np.clip((image - 0.5) * contrast + 0.5, 0, 1)
            image = (image * 255).astype(np.uint8)
        
        return image
    
    def _random_crop(self, image, targets, p=0.3):
        """Randomly crop image and adjust bounding boxes."""
        if random.random() >= p:
            return image, targets
        
        if isinstance(image, torch.Tensor):
            c, h, w = image.shape
        else:
            h, w = image.shape[:2]
        
        # Random crop dimensions (50-100% of original size)
        new_h = int(h * random.uniform(0.5, 1.0))
        new_w = int(w * random.uniform(0.5, 1.0))
        
        # Random crop position
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        # Crop image
        if isinstance(image, torch.Tensor):
            image = image[:, top:top+new_h, left:left+new_w]
        else:
            image = image[top:top+new_h, left:left+new_w]
        
        # Adjust bounding boxes if they exist
        if 'boxes' in targets and len(targets['boxes']) > 0:
            boxes = targets['boxes'].clone()
            
            # Adjust coordinates
            boxes[:, 0] = torch.clamp(boxes[:, 0] - left, min=0, max=new_w)  # x1
            boxes[:, 1] = torch.clamp(boxes[:, 1] - top, min=0, max=new_h)   # y1
            boxes[:, 2] = torch.clamp(boxes[:, 2] - left, min=0, max=new_w)  # x2
            boxes[:, 3] = torch.clamp(boxes[:, 3] - top, min=0, max=new_h)   # y2
            
            # Filter out boxes that are too small after cropping
            valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            
            # Update targets with valid boxes only
            if valid_boxes.sum() > 0:
                for k in targets:
                    if k == 'boxes':
                        targets[k] = boxes[valid_boxes]
                    elif k == 'labels' and len(targets[k]) == len(valid_boxes):
                        targets[k] = targets[k][valid_boxes]
        
        # Update original size in targets
        if 'orig_size' in targets:
            targets['orig_size'] = torch.tensor([new_h, new_w])
        
        return image, targets
    
    def _random_weather_sim(self, image, targets, p=0.3):
        """Simulate different weather conditions."""
        if random.random() >= p:
            return image, targets
        
        # Choose a random weather effect
        weather_type = random.choice(['rain', 'snow', 'fog', 'night'])
        
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy for processing
            if image.dim() == 3:  # CHW format
                image_np = image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            else:
                image_np = image.cpu().numpy()
        else:
            image_np = image.copy()
        
        # Ensure image is in float format for processing
        image_np = image_np.astype(np.float32) / 255.0 if image_np.dtype == np.uint8 else image_np
        
        # Apply selected weather effect
        if weather_type == 'rain':
            # Add rain streaks
            rain_drops = np.random.random(image_np.shape[:2]) * 0.3
            rain_mask = rain_drops > 0.97
            
            # Create rain streaks
            for i in range(20):  # Number of iterations to create streaks
                rain_mask = np.roll(rain_mask, shift=1, axis=0)
                image_np[rain_mask] = np.clip(image_np[rain_mask] * 0.95 + 0.05, 0, 1)  # Brighten
        
        elif weather_type == 'snow':
            # Add snow flakes
            snow_drops = np.random.random(image_np.shape[:2]) * 0.3
            snow_mask = snow_drops > 0.97
            
            # Create snow effect
            image_np[snow_mask] = np.clip(image_np[snow_mask] * 0.8 + 0.2, 0, 1)  # Whiten
            
            # Add slight blur for snow effect
            image_np = cv2.GaussianBlur(image_np, (3, 3), 0.5)
        
        elif weather_type == 'fog':
            # Create fog effect
            fog_intensity = random.uniform(0.3, 0.5)
            fog = np.ones_like(image_np) * fog_intensity
            image_np = cv2.addWeighted(image_np, 1 - fog_intensity, fog, fog_intensity, 0)
            
            # Add slight blur for fog effect
            image_np = cv2.GaussianBlur(image_np, (5, 5), 1.0)
        
        elif weather_type == 'night':
            # Simulate night time
            brightness_factor = random.uniform(0.4, 0.7)
            image_np = image_np * brightness_factor
            
            # Add slight blue tint for night effect
            blue_tint = np.zeros_like(image_np)
            blue_tint[:, :, 2] = 0.1  # Add blue channel
            image_np = np.clip(image_np + blue_tint, 0, 1)
        
        # Convert back to original format
        if isinstance(image, torch.Tensor):
            # Convert back to tensor
            image_np = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)
            image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0  # HWC -> CHW
        else:
            # Convert back to uint8 if needed
            image = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)
        
        return image, targets
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        # Track total batches for progress bar
        total_batches = sum(len(loader) for loader in self.train_loaders.values())
        progress_bar = tqdm(total=total_batches, desc=f"Epoch {epoch+1} Training")
        
        for dataset_name, loader in self.train_loaders.items():
            for batch_idx, (images, targets) in enumerate(loader):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Mixed precision forward - update autocast to include device_type
                with autocast(device_type='cuda', enabled=self.config["amp"]):
                    outputs = self.model(images, dataset_name)
                    loss_dict = self.loss_fn(outputs, targets, dataset_name)
                    loss = loss_dict["total_loss"] / self.config["gradient_accumulation"]
                    
                # Backward
                self.scaler.scale(loss).backward()
                
                # Free up memory
                del outputs, loss_dict
                torch.cuda.empty_cache()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config["gradient_accumulation"] == 0:
                    # Gradient clipping
                    if self.config["clip_grad"]:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config["max_grad_norm"])
                    
                    # First optimizer step, then scheduler step (correct order)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    # Move scheduler.step() after optimizer.step()
                    self.scheduler.step()
                
                total_loss += loss.item()
                
                # Update progress bar with current loss
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                progress_bar.update(1)
                
                # Expert sparsity regularization - only apply for MoE model
                if self.config["expert_sparsity"] > 0 and self.config.get("model_type", "moe") == "moe":
                    # We need to generate some dummy input for the gate
                    # since we don't have access to the intermediate features here
                    gate_loss = 0
                    for head in self.model.heads:
                        try:
                            # Skip if this head doesn't have a gate (traditional YOLOv8 heads)
                            if not hasattr(head, 'gate'):
                                continue
                                
                            # Instead of trying to guess the input dimension,
                            # create a dummy feature map with the same spatial dimensions as the input
                            # but with the correct channel dimension for each head
                            if hasattr(head, 'in_channels'):
                                gate_in_features = head.in_channels
                            else:
                                # Default channel dimensions from FPN outputs
                                gate_in_features = 256 if head != self.model.heads[-1] else 512
                                
                            # Create a dummy feature map with correct dimensions
                            # The gate expects a 4D tensor [B, C, H, W], not a 2D tensor
                            dummy_input = torch.randn(
                                images.size(0),  # batch size
                                gate_in_features,  # channels
                                8,  # arbitrary height
                                8,  # arbitrary width
                                device=self.device
                            )
                            
                            # Apply the gate to get routing probabilities
                            gate_probs = head.gate(dummy_input)
                            
                            # Calculate sparsity loss (encourage using fewer experts)
                            if gate_probs.dim() >= 2:
                                gate_loss += torch.mean(torch.sum(gate_probs, dim=1))
                            else:
                                gate_loss += torch.mean(gate_probs)
                                
                        except Exception as e:
                            # Skip this head if there's an error
                            print(f"Warning: Error in expert sparsity calculation: {e}")
                            continue
                    
                    # Only normalize if we have valid heads with gates
                    valid_heads = [head for head in self.model.heads if hasattr(head, 'gate')]
                    if len(valid_heads) > 0:
                        gate_loss /= len(valid_heads)
                        # Check if gate_loss is a tensor before calling item()
                        if isinstance(gate_loss, torch.Tensor):
                            total_loss += self.config["expert_sparsity"] * gate_loss.item()
                        else:
                            total_loss += self.config["expert_sparsity"] * gate_loss
                            
        progress_bar.close()
        return total_loss / len(self.train_loaders)

    def train(self):
        """Train the model for the specified number of epochs."""
        # Training loop
        for epoch in range(self.start_epoch, self.config["epochs"]):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Print metrics
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f}")
            
            # Print validation metrics for each dataset
            for dataset_name, metrics in val_metrics.items():
                print(f"Validation metrics for {dataset_name}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
            
            # Calculate average mAP across all datasets
            avg_map = np.mean([metrics.get("mAP", 0) for metrics in val_metrics.values()])
            print(f"Average mAP: {avg_map:.4f}")
            
            # Check if this is the best model so far
            is_best = avg_map > self.best_val_map
            if is_best:
                self.best_val_map = avg_map
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.config.get("early_stopping", False) and epoch > 10:
                # Check if validation mAP has improved in the last N epochs
                if avg_map < self.best_val_map * 0.95:  # 5% tolerance
                    print(f"Early stopping triggered. No improvement for several epochs.")
                    break
        
        print(f"Training completed. Best mAP: {self.best_val_map:.4f}")
        
    def validate(self):
        self.model.eval()
        val_metrics = {}
        
        # For mAP calculation
        all_predictions = defaultdict(list)
        all_targets = defaultdict(list)
        
        with torch.no_grad():
            for dataset_name, loader in self.val_loaders.items():
                dataset_metrics = defaultdict(float)
                
                # Create progress bar for validation
                progress_bar = tqdm(loader, desc=f"Validating {dataset_name}")
                
                for images, targets in progress_bar:
                    images = images.to(self.device)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    # Forward pass - update autocast to include device_type if used
                    with autocast(device_type='cuda', enabled=self.config["amp"]):
                        outputs = self.model(images, dataset_name)
                        
                    # Calculate loss
                    loss_dict = self.loss_fn(outputs, targets, dataset_name)
                    
                    # Store loss metrics
                    for k, v in loss_dict.items():
                        # Check if v is a tensor or a float
                        if isinstance(v, torch.Tensor):
                            dataset_metrics[k] += v.item() / len(loader)
                        else:
                            # If v is already a float, just add it
                            dataset_metrics[k] += v / len(loader)
                    
                    # Post-process predictions for mAP calculation
                    batch_predictions = self._post_process_predictions(outputs, dataset_name)
                    
                    # Store predictions and targets for mAP calculation
                    for i, (preds, target) in enumerate(zip(batch_predictions, targets)):
                        img_id = f"{dataset_name}_{len(all_predictions[dataset_name])}"
                        
                        # Store predictions with image id
                        for pred in preds:
                            # Format: [image_id, x1, y1, x2, y2, confidence, class_id]
                            all_predictions[dataset_name].append([
                                img_id, 
                                pred[0], pred[1], pred[2], pred[3],  # bbox
                                pred[4],  # confidence
                                pred[5]   # class_id
                            ])
                        
                        # Store targets with image id
                        if 'boxes' in target and len(target['boxes']) > 0:
                            for box_idx, box in enumerate(target['boxes']):
                                # Format: [image_id, x1, y1, x2, y2, class_id]
                                all_targets[dataset_name].append([
                                    img_id,
                                    box[0].item(), box[1].item(), box[2].item(), box[3].item(),
                                    target['labels'][box_idx].item()
                                ])
                    
                    # Update progress bar with current loss
                    if "total_loss" in loss_dict:
                        current_loss = loss_dict["total_loss"]
                        if isinstance(current_loss, torch.Tensor):
                            current_loss = current_loss.item()
                        progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
                
                # Calculate mAP for this dataset
                if len(all_predictions[dataset_name]) > 0:
                    mAP_metrics = self._calculate_mAP(
                        all_predictions[dataset_name], 
                        all_targets[dataset_name],
                        dataset_name
                    )
                    
                    # Add mAP metrics to dataset metrics
                    for k, v in mAP_metrics.items():
                        dataset_metrics[k] = v
                
                # Store all metrics for this dataset
                val_metrics[dataset_name] = dict(dataset_metrics)
                
                # Close progress bar
                progress_bar.close()
        
        return val_metrics
    
    def _post_process_predictions(self, outputs, dataset_name):
        """Post-process model outputs to get bounding boxes, confidence scores, and class IDs.
        
        Args:
            outputs: Model outputs
            dataset_name: Name of the dataset
            
        Returns:
            List of predictions for each image in the batch
            Each prediction is [x1, y1, x2, y2, confidence, class_id]
        """
        batch_predictions = []
        
        # Get confidence threshold (can be configured per dataset)
        conf_threshold = 0.25  # Default confidence threshold
        nms_threshold = 0.45   # Default NMS threshold
        
        # Process each image in the batch
        for img_idx, img_pred in enumerate(outputs):
            # Apply confidence threshold
            if isinstance(img_pred, torch.Tensor):
                # Reshape predictions to [num_preds, 4+num_classes]
                pred_shape = img_pred.shape
                flattened_pred = img_pred.reshape(-1, pred_shape[-1])
                
                # Extract box coordinates and class scores
                boxes = flattened_pred[:, :4]
                scores = flattened_pred[:, 4:]
                
                # Get confidence and class ID for each prediction
                max_scores, class_ids = torch.max(scores, dim=1)
                
                # Filter by confidence threshold
                mask = max_scores > conf_threshold
                filtered_boxes = boxes[mask]
                filtered_scores = max_scores[mask]
                filtered_class_ids = class_ids[mask]
                
                # Apply NMS (Non-Maximum Suppression)
                keep_indices = self._non_max_suppression(
                    filtered_boxes, filtered_scores, nms_threshold
                )
                
                # Create final predictions
                image_predictions = []
                for i in keep_indices:
                    box = filtered_boxes[i].cpu().numpy()
                    score = filtered_scores[i].item()
                    class_id = filtered_class_ids[i].item()
                    
                    image_predictions.append([
                        box[0], box[1], box[2], box[3],  # bbox
                        score,                           # confidence
                        class_id                         # class_id
                    ])
                
                batch_predictions.append(image_predictions)
            else:
                # Handle case where output is not a tensor
                batch_predictions.append([])
        
        return batch_predictions
    
    def _non_max_suppression(self, boxes, scores, threshold):
        """Apply Non-Maximum Suppression to remove overlapping boxes.
        
        Args:
            boxes: Tensor of shape [N, 4] containing bounding boxes
            scores: Tensor of shape [N] containing confidence scores
            threshold: IoU threshold for suppression
            
        Returns:
            List of indices to keep
        """
        # Convert to numpy for easier processing
        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
            scores_np = scores.cpu().numpy()
        else:
            boxes_np = np.array(boxes)
            scores_np = np.array(scores)
        
        # Get coordinates
        x1 = boxes_np[:, 0]
        y1 = boxes_np[:, 1]
        x2 = boxes_np[:, 2]
        y2 = boxes_np[:, 3]
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by confidence score
        order = scores_np.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with rest of the boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def _calculate_mAP(self, predictions, targets, dataset_name):
        """Calculate mean Average Precision using COCO-style evaluation.
        
        Args:
            predictions: List of predictions [image_id, x1, y1, x2, y2, confidence, class_id]
            targets: List of targets [image_id, x1, y1, x2, y2, class_id]
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary of mAP metrics
        """
        # Try to use pycocotools if available for COCO-style evaluation
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            import json
            import tempfile
            
            # Group by image_id
            pred_by_image = defaultdict(list)
            for pred in predictions:
                img_id = pred[0]
                pred_by_image[img_id].append(pred[1:])  # Remove image_id
            
            target_by_image = defaultdict(list)
            for target in targets:
                img_id = target[0]
                target_by_image[img_id].append(target[1:])  # Remove image_id
            
            # Create COCO format annotations
            coco_gt = {"images": [], "annotations": [], "categories": []}
            coco_dt = []
            
            # Add categories
            for i, cls_name in enumerate(self.config["datasets"][dataset_name]["classes"]):
                coco_gt["categories"].append({
                    "id": i,
                    "name": cls_name,
                    "supercategory": "none"
                })
            
            # Add images and annotations
            ann_id = 0
            for img_id, img_targets in target_by_image.items():
                # Add image
                coco_gt["images"].append({
                    "id": img_id,
                    "width": self.config["input_size"],
                    "height": self.config["input_size"],
                    "file_name": f"{img_id}.jpg"
                })
                
                # Add annotations
                for target in img_targets:
                    x1, y1, x2, y2, cls_id = target
                    width = x2 - x1
                    height = y2 - y1
                    
                    coco_gt["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls_id),
                        "bbox": [x1, y1, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    ann_id += 1
                
                # Add predictions
                if img_id in pred_by_image:
                    for pred in pred_by_image[img_id]:
                        x1, y1, x2, y2, score, cls_id = pred
                        width = x2 - x1
                        height = y2 - y1
                        
                        coco_dt.append({
                            "image_id": img_id,
                            "category_id": int(cls_id),
                            "bbox": [x1, y1, width, height],
                            "score": float(score)
                        })
            
            # Save to temporary files
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f_gt:
                json.dump(coco_gt, f_gt)
                gt_path = f_gt.name
            
            # Check if we have any predictions
            if len(coco_dt) == 0:
                print(f"Warning: No predictions for {dataset_name}, skipping COCO evaluation")
                # Clean up temporary file
                os.remove(gt_path)
                return {
                    "mAP": 0.0,
                    "mAP_50": 0.0,
                    "mAP_75": 0.0,
                    "mAP_small": 0.0,
                    "mAP_medium": 0.0,
                    "mAP_large": 0.0,
                }
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f_dt:
                json.dump(coco_dt, f_dt)
                dt_path = f_dt.name
            
            # Load COCO API
            coco_gt = COCO(gt_path)
            coco_dt = coco_gt.loadRes(dt_path)
            
            # Run evaluation
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Clean up temporary files
            os.remove(gt_path)
            os.remove(dt_path)
            
            # Extract metrics
            metrics = {
                "mAP": coco_eval.stats[0],  # AP@IoU=0.5:0.95
                "mAP_50": coco_eval.stats[1],  # AP@IoU=0.5
                "mAP_75": coco_eval.stats[2],  # AP@IoU=0.75
                "mAP_small": coco_eval.stats[3],  # AP for small objects
                "mAP_medium": coco_eval.stats[4],  # AP for medium objects
                "mAP_large": coco_eval.stats[5],  # AP for large objects
            }
            
            return metrics
            
        except (ImportError, ModuleNotFoundError):
            # Fallback to a simpler mAP calculation if pycocotools is not available
            print(f"Warning: pycocotools not available, using simplified mAP calculation for {dataset_name}")
            return self._calculate_simple_mAP(predictions, targets, dataset_name)
        
    def _calculate_simple_mAP(self, predictions, targets, dataset_name):
        """Calculate a simplified version of mAP when pycocotools is not available.
        
        Args:
            predictions: List of predictions [image_id, x1, y1, x2, y2, confidence, class_id]
            targets: List of targets [image_id, x1, y1, x2, y2, class_id]
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with mAP metrics
        """
        # Group predictions by class
        preds_by_class = defaultdict(list)
        for pred in predictions:
            img_id, x1, y1, x2, y2, conf, cls_id = pred
            preds_by_class[int(cls_id)].append([img_id, x1, y1, x2, y2, conf])
        
        # Group targets by class
        targets_by_class = defaultdict(list)
        for target in targets:
            img_id, x1, y1, x2, y2, cls_id = target
            targets_by_class[int(cls_id)].append([img_id, x1, y1, x2, y2])
        
        # Calculate AP for each class
        aps = []
        for cls_id in targets_by_class.keys():
            if cls_id in preds_by_class:
                ap = self._calculate_ap_for_class(
                    preds_by_class[cls_id], 
                    targets_by_class[cls_id],
                    iou_threshold=0.5
                )
                aps.append(ap)
        
        # Calculate mAP
        mAP = sum(aps) / len(aps) if aps else 0
        
        return {
            "mAP": mAP,
            "mAP_50": mAP,  # Same as mAP since we only calculated for IoU=0.5
        }
    
    def _calculate_ap_for_class(self, predictions, targets, iou_threshold=0.5):
        """Calculate Average Precision for a single class.
        
        Args:
            predictions: List of predictions [image_id, x1, y1, x2, y2, confidence]
            targets: List of targets [image_id, x1, y1, x2, y2]
            iou_threshold: IoU threshold for considering a prediction as correct
            
        Returns:
            Average Precision for this class
        """
        # Sort predictions by confidence (descending)
        predictions = sorted(predictions, key=lambda x: x[5], reverse=True)
        
        # Group targets by image_id for faster lookup
        targets_by_image = defaultdict(list)
        for target in targets:
            img_id = target[0]
            targets_by_image[img_id].append(target[1:])  # Remove image_id
        
        # Initialize precision and recall points
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        
        # Number of targets (ground truth objects)
        n_targets = len(targets)
        
        # For each prediction, check if it's a true positive or false positive
        for i, pred in enumerate(predictions):
            img_id = pred[0]
            pred_box = pred[1:5]  # x1, y1, x2, y2
            
            # If this image has targets
            if img_id in targets_by_image:
                img_targets = targets_by_image[img_id]
                
                # Find the target with highest IoU
                max_iou = -1
                max_idx = -1
                
                for j, target_box in enumerate(img_targets):
                    iou = self._calculate_iou(pred_box, target_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = j
                
                # If IoU exceeds threshold, it's a true positive
                if max_iou >= iou_threshold:
                    tp[i] = 1
                    # Remove the matched target to prevent multiple matches
                    img_targets.pop(max_idx)
                    targets_by_image[img_id] = img_targets
                else:
                    fp[i] = 1
            else:
                # No targets for this image, so it's a false positive
                fp[i] = 1
        
        # Compute cumulative precision and recall
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        recall = cumsum_tp / n_targets if n_targets > 0 else np.zeros_like(cumsum_tp)
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)
        
        # Compute average precision using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        
        return ap
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU between the two boxes
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou

# --------------------------
# 4. Configuration
# --------------------------

config = {
    # Model
    # Model
    "model_type": "traditional",  # "moe" or "traditional"
    "datasets": {
        "coco": {
            "enabled": True,
            "classes": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
                       "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
                       "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
            "path": "data/coco",
            "class_map": {
                1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
                6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
                # ... other COCO classes
            }
        },
        "kitti": {
            "enabled": True,
            "classes": ["car", "pedestrian", "cyclist", "truck", "van", "tram", "person_sitting", "misc"],
            "path": "data/kitti",
            "class_map": {
                "Car": "car", "Pedestrian": "pedestrian", "Cyclist": "cyclist", 
                "Truck": "truck", "Van": "van", "Tram": "tram",
                "Person_sitting": "person_sitting", "Misc": "misc"
            }
        },
        "waymo": {
            "enabled": True,
            "classes": ["vehicle", "pedestrian", "cyclist", "sign"],
            "path": "data/waymo",
            "class_map": {
                1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "sign"
            }
        },
        "argo": {
            "enabled": True,
            "classes": ["vehicle", "pedestrian", "bicycle", "motorcycle", "bus", "truck", 
                       "traffic_sign", "traffic_light", "stop_sign"],
            "path": "data/argo",
            "class_map": {
                "VEHICLE": "vehicle", "PEDESTRIAN": "pedestrian", "BICYCLE": "bicycle",
                "MOTORCYCLE": "motorcycle", "BUS": "bus", "TRUCK": "truck",
                "TRAFFIC_SIGN": "traffic_sign", "TRAFFIC_LIGHT": "traffic_light",
                "STOP_SIGN": "stop_sign"
            }
        },
        "nuscenes": {
            "enabled": True,
            "classes": ["car", "truck", "bus", "trailer", "construction_vehicle", 
                       "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier"],
            "path": "data/nuscenes",
            "class_map": {
                "car": "car", "truck": "truck", "bus": "bus", "trailer": "trailer",
                "construction_vehicle": "construction_vehicle", "pedestrian": "pedestrian",
                "motorcycle": "motorcycle", "bicycle": "bicycle", 
                "traffic_cone": "traffic_cone", "barrier": "barrier"
            }
        }
    },
    
    # Training
    "epochs": 100,
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "optimizer": "adamw",  # "adamw" or "sgd"
    "scheduler": "cosine",  # "cosine" or "onecycle"
    "input_size": 640,     # Input image size
    "num_workers": 4,      # Number of workers for data loading
    "balanced_sampling": True,  # Use balanced sampling for training
    
    # Performance optimizations
    "amp": True,  # Automatic Mixed Precision
    "gradient_accumulation": 4,
    "clip_grad": True,
    "max_grad_norm": 1.0,
    "expert_sparsity": 0.01,  # L1 regularization on gate weights
    
    # Expert configuration
    "num_experts": 4,
    "transformer_experts": 1,
    
    # Hardware
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --------------------------
# 5. Main Execution
# --------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 MoE on KITTI dataset")
    parser.add_argument("--data_path", type=str, default="/mnt/e/Dataset/Kitti/", 
                        help="Path to KITTI dataset, /mnt/e/Dataset/Kitti/, /DATA5T2/Dataset/Kitti")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--input_size", type=int, default=640, 
                        help="Input image size")
    parser.add_argument("--output_dir", type=str, default="outputs/kitti", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--model_type", type=str, default="moe", choices=["moe", "traditional"], 
                        help="Model type: moe (with experts) or traditional")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start with the full configuration
    training_config = config.copy()
    
    training_config["model_type"] = args.model_type
    
    # Override resume path if specified in command line
    if args.resume:
        training_config["resume_from"] = args.resume
    
    # Update only the KITTI path and enable status
    training_config["datasets"]["kitti"]["path"] = args.data_path
    training_config["datasets"]["kitti"]["enabled"] = True
    
    # Disable other datasets but keep their class information
    for dataset_name in ["coco", "waymo", "argo", "nuscenes"]:
        if dataset_name in training_config["datasets"]:
            training_config["datasets"][dataset_name]["enabled"] = False
    
    # Update training parameters
    training_config["epochs"] = args.epochs
    training_config["batch_size"] = args.batch_size
    training_config["lr"] = args.lr
    training_config["input_size"] = args.input_size
    training_config["gradient_accumulation"] = 4  # Smaller value for single dataset
    
    # Set PyTorch memory allocation configuration
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Initialize trainer
    trainer = YOLOv8Trainer(training_config)
    
    # Train model
    trainer.train()
    
if __name__ == "__main__":
    main()