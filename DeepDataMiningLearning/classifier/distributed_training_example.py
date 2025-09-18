#!/usr/bin/env python3
"""
Distributed Training Example for TrainingEngine

This script demonstrates how to use the TrainingEngine with multi-GPU distributed training.
It shows both single-GPU and multi-GPU training configurations.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path

# Import the training engine
from training_engine import TrainingEngine, launch_distributed_training, distributed_train_worker


class ResNetModel(nn.Module):
    """Simple ResNet-like model for demonstration."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def get_cifar10_datasets(data_dir='./data'):
    """Get CIFAR-10 train and validation datasets."""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )
    
    return train_dataset, val_dataset


def single_gpu_training(args):
    """Run single GPU training."""
    print("Starting single GPU training...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get datasets
    train_dataset, val_dataset = get_cifar10_datasets(args.data_dir)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = ResNetModel(num_classes=10).to(device)
    
    # Setup training components
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    criterion = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2
    )
    
    # Create training engine
    engine = TrainingEngine(
        model=model,
        device=device,
        use_amp=args.use_amp,
        use_ema=args.use_ema,
        experiment_name=args.experiment_name,
        save_dir=args.save_dir
    )
    
    # Train
    history = engine.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=args.epochs,
        save_best=True,
        save_last=True,
        early_stopping_patience=args.patience
    )
    
    print("Single GPU training completed!")
    print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    
    # Plot training history
    engine.plot_training_history(os.path.join(args.save_dir, "training_history.png"))
    
    return engine, history


def multi_gpu_training(args):
    """Run multi-GPU distributed training."""
    print("Starting multi-GPU distributed training...")
    
    # Get datasets
    train_dataset, val_dataset = get_cifar10_datasets(args.data_dir)
    
    # Model factory function
    def create_model():
        return ResNetModel(num_classes=10)
    
    # Training configuration
    config = {
        'model_fn': create_model,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'optimizer_fn': lambda params: torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=0.9,
            weight_decay=1e-4
        ),
        'criterion_fn': lambda: nn.CrossEntropyLoss(),
        'scheduler_fn': lambda opt: torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=[60, 120, 160],
            gamma=0.2
        ),
        'engine_kwargs': {
            'use_amp': args.use_amp,
            'use_ema': args.use_ema,
            'experiment_name': args.experiment_name,
            'save_dir': args.save_dir
        },
        'train_kwargs': {
            'num_epochs': args.epochs,
            'save_best': True,
            'save_last': True,
            'early_stopping_patience': args.patience
        }
    }
    
    # Launch distributed training
    world_size = args.world_size or torch.cuda.device_count()
    launch_distributed_training(
        train_fn=distributed_train_worker,
        world_size=world_size,
        backend=args.backend,
        **config
    )
    
    print("Multi-GPU distributed training completed!")


def main():
    parser = argparse.ArgumentParser(description='Distributed Training Example')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--use-ema', action='store_true', help='Use exponential moving average')
    
    # Distributed arguments
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--world-size', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'], 
                       help='Distributed backend')
    
    # Experiment arguments
    parser.add_argument('--experiment-name', type=str, default='cifar10_training', 
                       help='Experiment name')
    parser.add_argument('--save-dir', type=str, default='./experiments', 
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Use AMP: {args.use_amp}")
    print(f"  Use EMA: {args.use_ema}")
    print(f"  Distributed: {args.distributed}")
    if args.distributed:
        print(f"  World size: {args.world_size or torch.cuda.device_count()}")
        print(f"  Backend: {args.backend}")
    print(f"  Experiment: {args.experiment_name}")
    print(f"  Save dir: {args.save_dir}")
    print()
    
    # Run training
    if args.distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        multi_gpu_training(args)
    else:
        if args.distributed:
            print("Warning: Distributed training requested but only 1 GPU available. Using single GPU.")
        single_gpu_training(args)


if __name__ == '__main__':
    main()