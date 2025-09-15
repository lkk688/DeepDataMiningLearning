"""Unified Model Class for Image Classification

Supports torchvision, timm, and huggingface models with a consistent interface.
Provides model loading, inference, and training utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
from torchvision.models import get_model, get_model_weights, list_models
try:
    import timm
    import timm.optim
    import timm.scheduler
    from timm.utils import ModelEmaV2, AverageMeter
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    # Fallback implementations
    class AverageMeter:
        def __init__(self):
            self.reset()
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    
    class ModelEmaV2:
        def __init__(self, model, decay=0.9999):
            self.model = model
            self.decay = decay
        def update(self, model):
            pass
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import os

# Optional HuggingFace imports
try:
    from transformers import (
        AutoImageProcessor, 
        AutoModelForImageClassification,
        TrainingArguments,
        Trainer
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[INFO] HuggingFace transformers not available. Install with: pip install transformers")

# Try to get torchinfo for model summary
try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    print("[INFO] torchinfo not available. Install with: pip install torchinfo")


class UnifiedImageClassifier:
    """Unified model class supporting multiple model sources and training methods."""
    
    def __init__(
        self,
        model_source: str = "torchvision",
        model_name: str = "resnet50",
        num_classes: Optional[int] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.2,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize unified image classifier.
        
        Args:
            model_source: Source type ('torchvision', 'timm', 'huggingface')
            model_name: Name of the model
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
            dropout_rate: Dropout rate for classifier head
            device: Device to use ('cuda', 'cpu', 'mps')
            **kwargs: Additional arguments for specific model loaders
        """
        self.model_source = model_source.lower()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate
        self.kwargs = kwargs
        
        # Set device
        if device is None:
            self.device = self._get_best_device()
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = None
        self.processor = None
        self.class_names = None
        self._load_model()
        
        # Move model to device
        if self.model:
            self.model.to(self.device)
    
    def _get_best_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _load_model(self):
        """Load model based on model source."""
        if self.model_source == "torchvision":
            self._load_torchvision_model()
        elif self.model_source == "timm":
            self._load_timm_model()
        elif self.model_source == "huggingface":
            self._load_huggingface_model()
        else:
            raise ValueError(f"Unsupported model source: {self.model_source}")
    
    def _load_torchvision_model(self):
        """Load torchvision model."""
        available_models = list_models(module=torchvision.models)
        if self.model_name not in available_models:
            raise ValueError(f"Model {self.model_name} not found in torchvision models")
        
        # Get model with weights
        if self.pretrained:
            weights_enum = get_model_weights(self.model_name)
            weights = weights_enum.DEFAULT
            self.model = get_model(self.model_name, weights=weights)
            
            # Get class names from weights
            if hasattr(weights, 'meta') and 'categories' in weights.meta:
                self.class_names = weights.meta['categories']
        else:
            self.model = get_model(self.model_name, weights=None)
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            self._freeze_backbone()
        
        # Modify classifier if num_classes is specified
        if self.num_classes is not None:
            self._modify_classifier()
        else:
            self.num_classes = self._get_num_classes()
    
    def _load_timm_model(self):
        """Load timm model."""
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for timm models. Install with: pip install timm")
        try:
            self.model = timm.create_model(
                self.model_name,
                pretrained=self.pretrained,
                num_classes=self.num_classes or 1000,
                drop_rate=self.dropout_rate,
                **self.kwargs
            )
            
            if self.num_classes is None:
                self.num_classes = getattr(self.model, 'num_classes', 1000)
            
            # Freeze backbone if requested
            if self.freeze_backbone:
                self._freeze_backbone()
                
        except Exception as e:
            raise ValueError(f"Failed to load timm model {self.model_name}: {e}")
    
    def _load_huggingface_model(self):
        """Load HuggingFace model."""
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers not available")
        
        try:
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            if self.num_classes is not None:
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_name,
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                    **self.kwargs
                )
            else:
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_name,
                    **self.kwargs
                )
                self.num_classes = self.model.config.num_labels
            
            # Get class names if available
            if hasattr(self.model.config, 'id2label'):
                self.class_names = list(self.model.config.id2label.values())
            
            # Freeze backbone if requested
            if self.freeze_backbone:
                self._freeze_backbone()
                
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace model {self.model_name}: {e}")
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        if self.model_source == "huggingface":
            # Freeze all parameters except classifier
            for name, param in self.model.named_parameters():
                if 'classifier' not in name and 'head' not in name:
                    param.requires_grad = False
        else:
            # For torchvision and timm models
            # Freeze all parameters except the last layer
            for name, param in self.model.named_parameters():
                if any(layer_name in name for layer_name in ['classifier', 'head', 'fc']):
                    continue
                param.requires_grad = False
        
        print(f"Backbone frozen. Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def _modify_classifier(self):
        """Modify classifier head for custom number of classes."""
        if self.model_source == "huggingface":
            # HuggingFace models handle this in initialization
            return
        
        # Find the classifier layer
        classifier_layer = None
        classifier_name = None
        
        for name, module in self.model.named_modules():
            if name in ['classifier', 'head', 'fc'] and isinstance(module, nn.Linear):
                classifier_layer = module
                classifier_name = name
                break
        
        if classifier_layer is None:
            # Try to find the last linear layer
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Linear):
                    classifier_layer = module
                    classifier_name = name
                    break
        
        if classifier_layer is not None:
            # Create new classifier
            new_classifier = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(classifier_layer.in_features, self.num_classes)
            )
            
            # Replace the classifier
            if classifier_name == 'classifier':
                self.model.classifier = new_classifier
            elif classifier_name == 'head':
                self.model.head = new_classifier
            elif classifier_name == 'fc':
                self.model.fc = new_classifier
            else:
                print(f"Warning: Could not replace classifier layer {classifier_name}")
        else:
            print("Warning: Could not find classifier layer to modify")
    
    def _get_num_classes(self) -> int:
        """Get number of classes from model."""
        if hasattr(self.model, 'num_classes'):
            return self.model.num_classes
        elif hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'out_features'):
            return self.model.classifier.out_features
        elif hasattr(self.model, 'head') and hasattr(self.model.head, 'out_features'):
            return self.model.head.out_features
        elif hasattr(self.model, 'fc') and hasattr(self.model.fc, 'out_features'):
            return self.model.fc.out_features
        else:
            return 1000  # Default
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        self.model.eval()
        with torch.no_grad():
            if self.model_source == "huggingface":
                outputs = self.model(x)
                return outputs.logits
            else:
                return self.model(x)
    
    def predict(
        self, 
        x: torch.Tensor, 
        return_probs: bool = True, 
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions on input tensor."""
        logits = self.forward(x)
        
        if return_probs:
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, top_k, dim=1)
            return top_probs, top_indices
        else:
            _, top_indices = torch.topk(logits, top_k, dim=1)
            return logits, top_indices
    
    def predict_single(
        self, 
        image: torch.Tensor, 
        return_probs: bool = True, 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Make prediction on a single image."""
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        image = image.to(self.device)
        probs_or_logits, indices = self.predict(image, return_probs, top_k)
        
        # Convert to CPU and numpy
        probs_or_logits = probs_or_logits.cpu().numpy()[0]
        indices = indices.cpu().numpy()[0]
        
        # Create result dictionary
        result = {
            'indices': indices,
            'scores': probs_or_logits,
        }
        
        # Add class names if available
        if self.class_names:
            result['classes'] = [self.class_names[idx] for idx in indices]
        
        return result
    
    def get_optimizer(
        self, 
        optimizer_name: str = "adamw", 
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        **kwargs
    ) -> torch.optim.Optimizer:
        """Get optimizer for training."""
        if self.model_source == "timm" and TIMM_AVAILABLE:
            # Use timm's optimizer factory
            return timm.optim.create_optimizer_v2(
                self.model,
                opt=optimizer_name,
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        else:
            # Use standard PyTorch optimizers
            optimizer_class = getattr(torch.optim, optimizer_name.title(), torch.optim.AdamW)
            return optimizer_class(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
    
    def get_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_name: str = "cosine",
        num_epochs: int = 100,
        **kwargs
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Get learning rate scheduler."""
        if self.model_source == "timm" and TIMM_AVAILABLE and scheduler_name == "cosine":
            # Use timm's cosine scheduler
            return timm.scheduler.CosineLRScheduler(
                optimizer,
                t_initial=num_epochs,
                lr_min=1e-6,
                warmup_t=5,
                warmup_lr_init=1e-5,
                **kwargs
            )
        else:
            # Use standard PyTorch schedulers
            if scheduler_name == "cosine":
                return lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, **kwargs)
            elif scheduler_name == "step":
                return lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, **kwargs)
            elif scheduler_name == "plateau":
                return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, **kwargs)
            else:
                raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    def train_epoch(
        self,
        dataloader,
        optimizer,
        criterion,
        scheduler=None,
        ema_model=None,
        epoch=0
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        num_steps_per_epoch = len(dataloader)
        num_updates = epoch * num_steps_per_epoch
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if self.model_source == "huggingface":
                outputs = self.model(data, labels=target)
                loss = outputs.loss
                logits = outputs.logits
            else:
                logits = self.model(data)
                loss = criterion(logits, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update EMA model
            if ema_model is not None:
                ema_model.update(self.model)
            
            # Update scheduler (for timm schedulers)
            if scheduler is not None and hasattr(scheduler, 'step_update'):
                num_updates += 1
                scheduler.step_update(num_updates=num_updates)
            
            # Calculate accuracy
            pred = logits.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += target.size(0)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def evaluate(
        self,
        dataloader,
        criterion=None
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                loss = None
                if self.model_source == "huggingface":
                    outputs = self.model(data, labels=target if criterion else None)
                    logits = outputs.logits
                    if criterion and hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    elif criterion:
                        loss = criterion(logits, target)
                else:
                    logits = self.model(data)
                    if criterion:
                        loss = criterion(logits, target)
                
                if criterion and loss is not None:
                    total_loss += loss.item()
                
                # Calculate accuracy
                pred = logits.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += target.size(0)
                
                # Store predictions and targets
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader) if criterion else 0.0
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy, np.array(all_predictions), np.array(all_targets)
    
    def save_model(self, path: str):
        """Save model state dict."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_source': self.model_source,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model state dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Update metadata
        self.model_source = checkpoint.get('model_source', self.model_source)
        self.model_name = checkpoint.get('model_name', self.model_name)
        self.num_classes = checkpoint.get('num_classes', self.num_classes)
        self.class_names = checkpoint.get('class_names', self.class_names)
        
        print(f"Model loaded from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'model_source': self.model_source,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'class_names': self.class_names[:10] if self.class_names else None  # Show first 10
        }
        
        return info
    
    def print_model_summary(self, input_size: Tuple[int, ...] = (1, 3, 224, 224)):
        """Print model summary."""
        if TORCHINFO_AVAILABLE:
            summary(
                self.model,
                input_size=input_size,
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
            )
        else:
            info = self.get_model_info()
            print("\nModel Information:")
            for key, value in info.items():
                print(f"{key}: {value}")


def test_unified_model():
    """Test function for UnifiedImageClassifier."""
    print("Testing UnifiedImageClassifier...")
    
    # Test torchvision model
    print("\n1. Testing torchvision ResNet50...")
    try:
        model = UnifiedImageClassifier(
            model_source="torchvision",
            model_name="resnet50",
            num_classes=10,
            pretrained=True,
            freeze_backbone=True
        )
        
        print(f"Model loaded: {model.model_name}")
        print(f"Device: {model.device}")
        print(f"Number of classes: {model.num_classes}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(model.device)
        output = model.forward(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # Test prediction
        result = model.predict_single(dummy_input[0])
        print(f"Prediction result keys: {result.keys()}")
        
        # Print model info
        info = model.get_model_info()
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Trainable parameters: {info['trainable_parameters']:,}")
        
    except Exception as e:
        print(f"Error testing torchvision model: {e}")
    
    # Test timm model
    print("\n2. Testing timm EfficientNet...")
    try:
        model = UnifiedImageClassifier(
            model_source="timm",
            model_name="efficientnet_b0",
            num_classes=10,
            pretrained=True
        )
        
        print(f"Model loaded: {model.model_name}")
        print(f"Number of classes: {model.num_classes}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(model.device)
        output = model.forward(dummy_input)
        print(f"Output shape: {output.shape}")
        
    except Exception as e:
        print(f"Error testing timm model: {e}")
    
    # Test HuggingFace model (if available)
    if HF_AVAILABLE:
        print("\n3. Testing HuggingFace ViT...")
        try:
            model = UnifiedImageClassifier(
                model_source="huggingface",
                model_name="google/vit-base-patch16-224",
                num_classes=10,
                pretrained=True
            )
            
            print(f"Model loaded: {model.model_name}")
            print(f"Number of classes: {model.num_classes}")
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224).to(model.device)
            output = model.forward(dummy_input)
            print(f"Output shape: {output.shape}")
            
        except Exception as e:
            print(f"Error testing HuggingFace model: {e}")
    else:
        print("\n3. HuggingFace models not available")
    
    print("\nUnifiedImageClassifier testing completed!")


if __name__ == "__main__":
    test_unified_model()