"""Utility Functions for Unified Image Classification

Provides helper functions for visualization, model analysis, data processing,
and other common tasks in image classification workflows.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
import json
import time
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFont
import random

# Optional imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torchvision.transforms as transforms
    from torchvision.utils import make_grid
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'size_mb': size_mb,
        'param_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024
    }


def analyze_model(model: nn.Module) -> Dict[str, Any]:
    """Comprehensive model analysis."""
    analysis = {
        'model_type': type(model).__name__,
        'parameters': count_parameters(model),
        'size': get_model_size(model)
    }
    
    # Count layer types
    layer_counts = defaultdict(int)
    for name, module in model.named_modules():
        layer_counts[type(module).__name__] += 1
    
    analysis['layer_counts'] = dict(layer_counts)
    
    return analysis


def visualize_dataset_samples(
    dataset,
    class_names: Optional[List[str]] = None,
    num_samples: int = 16,
    samples_per_class: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """Visualize random samples from dataset."""
    if samples_per_class is not None and class_names is not None:
        # Sample specific number per class
        samples_to_show = []
        labels_to_show = []
        
        # Group samples by class
        class_samples = defaultdict(list)
        for i, (_, label) in enumerate(dataset):
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_samples[label].append(i)
        
        # Sample from each class
        for class_id in range(len(class_names)):
            if class_id in class_samples:
                available_samples = class_samples[class_id]
                selected = random.sample(
                    available_samples, 
                    min(samples_per_class, len(available_samples))
                )
                for idx in selected:
                    image, label = dataset[idx]
                    samples_to_show.append(image)
                    labels_to_show.append(label)
    else:
        # Random sampling
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        samples_to_show = []
        labels_to_show = []
        
        for idx in indices:
            image, label = dataset[idx]
            samples_to_show.append(image)
            labels_to_show.append(label)
    
    # Create visualization
    num_samples = len(samples_to_show)
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        row, col = i // cols, i % cols
        
        # Convert tensor to numpy for visualization
        image = samples_to_show[i]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
            if image.shape[0] == 3:  # CHW to HWC
                image = np.transpose(image, (1, 2, 0))
        
        # Normalize for display
        if image.max() <= 1.0:
            image = np.clip(image, 0, 1)
        else:
            image = np.clip(image / 255.0, 0, 1)
        
        axes[row, col].imshow(image)
        axes[row, col].axis('off')
        
        # Add label
        label = labels_to_show[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        
        title = f"Class: {label}"
        if class_names and label < len(class_names):
            title = f"Class: {class_names[label]}"
        
        axes[row, col].set_title(title, fontsize=10)
    
    # Hide unused subplots
    for i in range(num_samples, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dataset visualization saved: {save_path}")
    
    plt.show()


def plot_class_distribution(
    dataset,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """Plot class distribution in dataset."""
    # Count classes
    class_counts = Counter()
    
    for _, label in dataset:
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_counts[label] += 1
    
    # Prepare data for plotting
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    # Create labels
    if class_names:
        labels = [class_names[c] if c < len(class_names) else f"Class {c}" for c in classes]
    else:
        labels = [f"Class {c}" for c in classes]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    bars = ax1.bar(range(len(classes)), counts)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Class Distribution')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Distribution (Percentage)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved: {save_path}")
    
    plt.show()
    
    # Print statistics
    total_samples = sum(counts)
    print(f"\nDataset Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {len(classes)}")
    print(f"Average samples per class: {total_samples / len(classes):.1f}")
    print(f"Min samples per class: {min(counts)}")
    print(f"Max samples per class: {max(counts)}")


def create_data_augmentation_preview(
    dataset,
    transform,
    num_samples: int = 4,
    num_augmentations: int = 4,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """Preview data augmentation effects."""
    # Get random samples
    indices = random.sample(range(len(dataset)), num_samples)
    
    fig, axes = plt.subplots(num_samples, num_augmentations + 1, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Get original image
        original_image, label = dataset[idx]
        
        # Show original
        if isinstance(original_image, torch.Tensor):
            img_np = original_image.numpy()
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
        else:
            img_np = np.array(original_image)
        
        # Normalize for display
        if img_np.max() <= 1.0:
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = np.clip(img_np / 255.0, 0, 1)
        
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Show augmentations
        for j in range(num_augmentations):
            # Apply transform
            if isinstance(original_image, torch.Tensor):
                # Convert to PIL for transforms
                pil_image = transforms.ToPILImage()(original_image)
                augmented = transform(pil_image)
            else:
                augmented = transform(original_image)
            
            # Convert back for display
            if isinstance(augmented, torch.Tensor):
                aug_np = augmented.numpy()
                if aug_np.shape[0] == 3:
                    aug_np = np.transpose(aug_np, (1, 2, 0))
            else:
                aug_np = np.array(augmented)
            
            # Normalize for display
            if aug_np.max() <= 1.0:
                aug_np = np.clip(aug_np, 0, 1)
            else:
                aug_np = np.clip(aug_np / 255.0, 0, 1)
            
            axes[i, j + 1].imshow(aug_np)
            axes[i, j + 1].set_title(f'Aug {j + 1}')
            axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Augmentation preview saved: {save_path}")
    
    plt.show()


def plot_training_curves(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """Plot comprehensive training curves."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Loss curves
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Accuracy curves
    if 'train_accuracy' in history and 'val_accuracy' in history:
        axes[0, 1].plot(history['train_accuracy'], label='Train Acc', color='blue')
        axes[0, 1].plot(history['val_accuracy'], label='Val Acc', color='red')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Learning rate
    if 'train_learning_rate' in history:
        axes[0, 2].plot(history['train_learning_rate'], color='green')
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].grid(True)
    
    # Top-5 accuracy (if available)
    if 'val_top5_accuracy' in history:
        axes[1, 0].plot(history['val_top5_accuracy'], color='purple')
        axes[1, 0].set_title('Top-5 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-5 Accuracy')
        axes[1, 0].grid(True)
    
    # Loss difference (overfitting indicator)
    if 'train_loss' in history and 'val_loss' in history:
        loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
        axes[1, 1].plot(loss_diff, color='orange')
        axes[1, 1].set_title('Overfitting Indicator (Val - Train Loss)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].grid(True)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Accuracy difference
    if 'train_accuracy' in history and 'val_accuracy' in history:
        acc_diff = np.array(history['train_accuracy']) - np.array(history['val_accuracy'])
        axes[1, 2].plot(acc_diff, color='brown')
        axes[1, 2].set_title('Generalization Gap (Train - Val Acc)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy Difference')
        axes[1, 2].grid(True)
        axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved: {save_path}")
    
    plt.show()


def calculate_dataset_statistics(
    dataloader,
    channels: int = 3
) -> Dict[str, np.ndarray]:
    """Calculate dataset mean and std for normalization."""
    print("Calculating dataset statistics...")
    
    # Initialize accumulators
    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    total_samples = 0
    
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, channels, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    stats = {
        'mean': mean.numpy(),
        'std': std.numpy(),
        'total_samples': total_samples
    }
    
    print(f"Dataset statistics calculated:")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std: {stats['std']}")
    print(f"  Total samples: {stats['total_samples']}")
    
    return stats


def create_model_comparison_table(
    models_info: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> str:
    """Create a comparison table for different models."""
    if not models_info:
        return "No models to compare"
    
    # Create table header
    headers = ['Model', 'Parameters', 'Size (MB)', 'Accuracy', 'Inference Time (ms)']
    
    # Prepare data
    table_data = []
    for info in models_info:
        row = [
            info.get('name', 'Unknown'),
            f"{info.get('parameters', 0):,}",
            f"{info.get('size_mb', 0):.2f}",
            f"{info.get('accuracy', 0):.4f}",
            f"{info.get('inference_time_ms', 0):.2f}"
        ]
        table_data.append(row)
    
    # Create table string
    col_widths = [max(len(str(row[i])) for row in [headers] + table_data) for i in range(len(headers))]
    
    def format_row(row):
        return ' | '.join(str(item).ljust(width) for item, width in zip(row, col_widths))
    
    table_str = format_row(headers) + '\n'
    table_str += '-' * len(table_str) + '\n'
    
    for row in table_data:
        table_str += format_row(row) + '\n'
    
    print("Model Comparison:")
    print(table_str)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"Model comparison saved: {save_path}")
    
    return table_str


def save_experiment_config(
    config: Dict[str, Any],
    save_path: str
):
    """Save experiment configuration to JSON."""
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Experiment config saved: {save_path}")


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from JSON."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Experiment config loaded: {config_path}")
    return config


def benchmark_inference_speed(
    model,
    input_shape: Tuple[int, ...],
    device: torch.device,
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """Benchmark model inference speed."""
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Warmup
    print(f"Warming up for {warmup_runs} runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Benchmark
    print(f"Benchmarking for {num_runs} runs...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    times = np.array(times)
    stats = {
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'median_time_ms': float(np.median(times)),
        'fps': float(1000 / np.mean(times))
    }
    
    print(f"Inference speed benchmark results:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    return stats


def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Test model analysis
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 10)
    )
    
    analysis = analyze_model(model)
    print(f"Model analysis: {analysis}")
    
    # Test dataset creation and visualization
    dummy_data = torch.randn(50, 3, 32, 32)
    dummy_targets = torch.randint(0, 5, (50,))
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
    
    class_names = [f"class_{i}" for i in range(5)]
    
    # Test class distribution
    plot_class_distribution(dummy_dataset, class_names)
    
    # Test dataset samples visualization
    visualize_dataset_samples(dummy_dataset, class_names, num_samples=8)
    
    # Test benchmark
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    benchmark_stats = benchmark_inference_speed(
        model, (3, 32, 32), device, num_runs=10, warmup_runs=2
    )
    
    print("Utility functions testing completed!")


if __name__ == "__main__":
    test_utils()