"""Unified Dataset Class for Image Classification

Supports torchvision, timm, and huggingface datasets with a consistent interface.
Provides data loading, preprocessing, augmentation, and visualization utilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
try:
    import timm
    from timm.data import create_transform, resolve_data_config
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[INFO] timm not available. Install with: pip install timm")
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path

# Optional HuggingFace imports
try:
    from datasets import load_dataset
    from transformers import AutoImageProcessor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[INFO] HuggingFace datasets/transformers not available. Install with: pip install datasets transformers")


class UnifiedImageDataset:
    """Unified dataset class supporting multiple data sources and preprocessing pipelines."""
    
    def __init__(
        self,
        data_source: str = "torchvision",
        dataset_name: str = "CIFAR10",
        data_path: Optional[str] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        **kwargs
    ):
        """
        Initialize unified dataset.
        
        Args:
            data_source: Source type ('torchvision', 'timm', 'huggingface', 'folder')
            dataset_name: Name of the dataset
            data_path: Path to data (for folder/custom datasets)
            split: Dataset split ('train', 'val', 'test')
            transform: Transform pipeline
            target_transform: Target transform pipeline
            download: Whether to download dataset if not available
            **kwargs: Additional arguments for specific dataset loaders
        """
        self.data_source = data_source.lower()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.kwargs = kwargs
        
        # Initialize dataset
        self.dataset = None
        self.classes = None
        self.num_classes = None
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset based on data source."""
        if self.data_source == "torchvision":
            self._load_torchvision_dataset()
        elif self.data_source == "timm":
            self._load_timm_dataset()
        elif self.data_source == "huggingface":
            self._load_huggingface_dataset()
        elif self.data_source == "folder":
            self._load_folder_dataset()
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")
    
    def _load_torchvision_dataset(self):
        """Load torchvision dataset."""
        dataset_class = getattr(datasets, self.dataset_name, None)
        if dataset_class is None:
            raise ValueError(f"Dataset {self.dataset_name} not found in torchvision.datasets")
        
        train_split = self.split in ['train', 'training']
        
        self.dataset = dataset_class(
            root=self.data_path or './data',
            train=train_split,
            download=self.download,
            transform=self.transform,
            target_transform=self.target_transform,
            **self.kwargs
        )
        
        # Get class information
        if hasattr(self.dataset, 'classes'):
            self.classes = self.dataset.classes
            self.num_classes = len(self.classes)
        else:
            # Infer from dataset
            self._infer_classes()
    
    def _load_timm_dataset(self):
        """Load dataset using timm's data utilities."""
        if not TIMM_AVAILABLE:
            raise ImportError("timm not available. Install with: pip install timm")
            
        if self.data_path is None:
            raise ValueError("data_path is required for timm datasets")
        
        # Use ImageFolder for timm datasets
        if self.split == 'train':
            data_dir = os.path.join(self.data_path, 'train')
        elif self.split == 'val':
            data_dir = os.path.join(self.data_path, 'val')
        else:
            data_dir = os.path.join(self.data_path, self.split)
        
        self.dataset = datasets.ImageFolder(
            root=data_dir,
            transform=self.transform,
            target_transform=self.target_transform
        )
        
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
    
    def _load_huggingface_dataset(self):
        """Load HuggingFace dataset."""
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace datasets not available. Install with: pip install datasets transformers")
        
        # Load dataset
        hf_dataset = load_dataset(self.dataset_name, split=self.split, **self.kwargs)
        
        # Wrap in PyTorch dataset
        self.dataset = HuggingFaceDatasetWrapper(
            hf_dataset,
            transform=self.transform,
            target_transform=self.target_transform
        )
        
        # Get class information
        if hasattr(hf_dataset.features, 'label'):
            self.classes = hf_dataset.features['label'].names
            self.num_classes = len(self.classes)
        else:
            self._infer_classes()
    
    def _load_folder_dataset(self):
        """Load dataset from folder structure."""
        if self.data_path is None:
            raise ValueError("data_path is required for folder datasets")
        
        self.dataset = datasets.ImageFolder(
            root=self.data_path,
            transform=self.transform,
            target_transform=self.target_transform
        )
        
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
    
    def _infer_classes(self):
        """Infer class information from dataset."""
        if hasattr(self.dataset, '__len__') and len(self.dataset) > 0:
            # Sample a few items to infer classes
            targets = []
            for i in range(min(100, len(self.dataset))):
                _, target = self.dataset[i]
                targets.append(target)
            
            unique_targets = sorted(list(set(targets)))
            self.num_classes = len(unique_targets)
            self.classes = [f"class_{i}" for i in unique_targets]
        else:
            self.num_classes = 1000  # Default
            self.classes = [f"class_{i}" for i in range(self.num_classes)]
    
    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ) -> DataLoader:
        """Create DataLoader for the dataset."""
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )
    
    def get_transforms(self, model_name: str = None, model_type: str = "torchvision") -> transforms.Compose:
        """Get appropriate transforms for the model type."""
        if model_type == "torchvision":
            return self._get_torchvision_transforms()
        elif model_type == "timm":
            return self._get_timm_transforms(model_name)
        elif model_type == "huggingface":
            return self._get_huggingface_transforms(model_name)
        else:
            return self._get_default_transforms()
    
    def _get_torchvision_transforms(self) -> transforms.Compose:
        """Get torchvision transforms."""
        if self.split in ['train', 'training']:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _get_timm_transforms(self, model_name: str) -> transforms.Compose:
        """Get timm transforms based on model configuration."""
        if not TIMM_AVAILABLE:
            print("Warning: timm not available, using default transforms")
            return self._get_default_transforms()
            
        if model_name:
            try:
                # Create a dummy model to get data config
                model = timm.create_model(model_name, pretrained=False)
                data_config = resolve_data_config(model.pretrained_cfg)
                return create_transform(**data_config, is_training=(self.split == 'train'))
            except:
                pass
        
        # Fallback to default transforms
        return self._get_default_transforms()
    
    def _get_huggingface_transforms(self, model_name: str) -> transforms.Compose:
        """Get HuggingFace transforms."""
        if HF_AVAILABLE and model_name:
            try:
                processor = AutoImageProcessor.from_pretrained(model_name)
                # Convert HF processor to torchvision transforms
                return transforms.Compose([
                    transforms.Resize((processor.size['height'], processor.size['width'])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
                ])
            except:
                pass
        
        return self._get_default_transforms()
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def visualize_samples(self, num_samples: int = 8, figsize: Tuple[int, int] = (12, 8)):
        """Visualize sample images from the dataset."""
        if len(self.dataset) == 0:
            print("Dataset is empty")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()
        
        # Sample random indices
        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
            
            image, label = self.dataset[idx]
            
            # Convert tensor to PIL Image if needed
            if torch.is_tensor(image):
                # Denormalize if normalized
                if image.min() < 0:  # Likely normalized
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image = image * std + mean
                    image = torch.clamp(image, 0, 1)
                
                image = transforms.ToPILImage()(image)
            
            axes[i].imshow(image)
            axes[i].set_title(f"Class: {self.classes[label] if self.classes else label}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution in the dataset."""
        if len(self.dataset) == 0:
            return {}
        
        class_counts = {}
        for _, label in self.dataset:
            class_name = self.classes[label] if self.classes else str(label)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return class_counts
    
    def __len__(self) -> int:
        return len(self.dataset) if self.dataset else 0
    
    def __getitem__(self, idx):
        return self.dataset[idx] if self.dataset else None


class HuggingFaceDatasetWrapper(Dataset):
    """Wrapper for HuggingFace datasets to work with PyTorch DataLoader."""
    
    def __init__(self, hf_dataset, transform=None, target_transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # Extract image and label
        image = item.get('image', item.get('img', None))
        label = item.get('label', item.get('labels', 0))
        
        if image is None:
            raise ValueError("No image found in dataset item")
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


def test_unified_dataset():
    """Test function for UnifiedImageDataset."""
    print("Testing UnifiedImageDataset...")
    
    # Test torchvision dataset
    print("\n1. Testing torchvision CIFAR10...")
    try:
        dataset = UnifiedImageDataset(
            data_source="torchvision",
            dataset_name="CIFAR10",
            split="train",
            download=True
        )
        print(f"Dataset loaded: {len(dataset)} samples, {dataset.num_classes} classes")
        print(f"Classes: {dataset.classes[:5]}...")  # Show first 5 classes
        
        # Test dataloader
        dataloader = dataset.get_dataloader(batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        print(f"Batch shape: {batch[0].shape}, {batch[1].shape}")
        
        # Test visualization
        dataset.visualize_samples(num_samples=4)
        
    except Exception as e:
        print(f"Error testing torchvision dataset: {e}")
    
    # Test folder dataset (if exists)
    print("\n2. Testing folder dataset...")
    try:
        # This would work if you have a folder structure
        # dataset = UnifiedImageDataset(
        #     data_source="folder",
        #     data_path="path/to/your/image/folder",
        #     split="train"
        # )
        print("Folder dataset test skipped (no path provided)")
    except Exception as e:
        print(f"Error testing folder dataset: {e}")
    
    print("\nUnifiedImageDataset testing completed!")


if __name__ == "__main__":
    test_unified_dataset()