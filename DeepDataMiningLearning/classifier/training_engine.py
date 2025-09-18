"""Training Engine for Unified Image Classification

Provides custom training loops with PyTorch standard practices.
Supports EMA, mixed precision, logging, and various training strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Optional imports
try:
    from timm.utils import ModelEmaV2, AverageMeter
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[INFO] timm not available for EMA. Install with: pip install timm")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[INFO] wandb not available for logging. Install with: pip install wandb")


class AverageMeterFallback:
    """Fallback AverageMeter if timm is not available."""
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


class ModelEmaFallback:
    """Fallback EMA implementation if timm is not available."""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class TrainingEngine:
    """Custom training engine with PyTorch standard practices."""
    
    def __init__(
        self,
        model,
        device: torch.device,
        use_amp: bool = True,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        gradient_clip_val: Optional[float] = None,
        log_interval: int = 100,
        save_dir: str = "./checkpoints",
        experiment_name: str = "experiment",
        use_wandb: bool = False,
        wandb_project: str = "image-classification",
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1
    ):
        """
        Initialize training engine.
        
        Args:
            model: Model to train
            device: Device to use for training
            use_amp: Whether to use automatic mixed precision
            use_ema: Whether to use exponential moving average
            ema_decay: EMA decay rate
            gradient_clip_val: Gradient clipping value
            log_interval: Logging interval in steps
            save_dir: Directory to save checkpoints
            experiment_name: Name of the experiment
            use_wandb: Whether to use wandb for logging
            wandb_project: Wandb project name
            distributed: Whether to use distributed training
            local_rank: Local rank for distributed training
            world_size: World size for distributed training
        """
        self.model = model
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_ema = use_ema
        self.gradient_clip_val = gradient_clip_val
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.distributed = distributed
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main_process = local_rank == 0
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize AMP scaler
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Initialize EMA
        if self.use_ema:
            if TIMM_AVAILABLE:
                self.ema_model = ModelEmaV2(model, decay=ema_decay)
            else:
                self.ema_model = ModelEmaFallback(model, decay=ema_decay)
        else:
            self.ema_model = None
        
        # Wrap model with DDP if distributed
        if self.distributed:
            # Convert BatchNorm to SyncBatchNorm for better distributed training
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
        
        # Initialize logging (only on main process)
        self.use_wandb = use_wandb and WANDB_AVAILABLE and self.is_main_process
        if self.use_wandb:
            wandb.init(project=wandb_project, name=experiment_name)
        
        # Training history
        self.history = defaultdict(list)
        self.global_step = 0
        self.epoch = 0
    
    @staticmethod
    def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
        """Initialize distributed training."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    @staticmethod
    def cleanup_distributed():
        """Clean up distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def _reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce tensor across all processes."""
        if not self.distributed:
            return tensor
        
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt
    
    def _should_log(self) -> bool:
        """Check if current process should log."""
        return self.is_main_process
    
    def train_epoch(
        self,
        train_loader,
        optimizer,
        criterion,
        scheduler=None,
        epoch: int = 0
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.epoch = epoch
        
        # Metrics
        if TIMM_AVAILABLE:
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            batch_time = AverageMeter()
        else:
            loss_meter = AverageMeterFallback()
            acc_meter = AverageMeterFallback()
            batch_time = AverageMeterFallback()
        
        num_steps_per_epoch = len(train_loader)
        end = time.time()
        
        # Progress bar (only on main process)
        if self.is_main_process:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            data_iter = pbar
        else:
            data_iter = train_loader
        
        for batch_idx, (data, target) in enumerate(data_iter):
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = criterion(output, target)
                
                # Backward pass with AMP
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_val is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard forward/backward pass
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                # Optimizer step
                optimizer.step()
            
            # Update EMA
            if self.ema_model is not None:
                self.ema_model.update(self.model)
            
            # Update scheduler (for step-based schedulers)
            if scheduler is not None and hasattr(scheduler, 'step_update'):
                scheduler.step_update(num_updates=self.global_step)
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            acc = pred.eq(target).float().mean()
            
            # Update metrics
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc.item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update progress bar (only on main process)
            if self.is_main_process:
                pbar.set_postfix({
                    'Loss': f'{loss_meter.avg:.4f}',
                    'Acc': f'{acc_meter.avg:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            # Logging (only on main process)
            if self.global_step % self.log_interval == 0 and self._should_log():
                # Reduce metrics across all processes for accurate logging
                if self.distributed:
                    loss_reduced = self._reduce_tensor(torch.tensor(loss.item(), device=self.device))
                    acc_reduced = self._reduce_tensor(torch.tensor(acc.item(), device=self.device))
                    self._log_metrics({
                        'train/loss': loss_reduced.item(),
                        'train/accuracy': acc_reduced.item(),
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                        'train/batch_time': batch_time.avg
                    })
                else:
                    self._log_metrics({
                        'train/loss': loss_meter.avg,
                        'train/accuracy': acc_meter.avg,
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                        'train/batch_time': batch_time.avg
                    })
            
            self.global_step += 1
        
        # Update scheduler (for epoch-based schedulers)
        if scheduler is not None and not hasattr(scheduler, 'step_update'):
            scheduler.step()
        
        return {
            'loss': loss_meter.avg,
            'accuracy': acc_meter.avg,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
    
    def validate(
        self,
        val_loader,
        criterion,
        use_ema: bool = False
    ) -> Dict[str, float]:
        """Validate the model."""
        # Choose model to evaluate
        if use_ema and self.ema_model is not None:
            if hasattr(self.ema_model, 'apply_shadow'):
                self.ema_model.apply_shadow()
            model_to_eval = self.model
        else:
            model_to_eval = self.model
        
        model_to_eval.eval()
        
        # Metrics
        if TIMM_AVAILABLE:
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            top5_meter = AverageMeter()
        else:
            loss_meter = AverageMeterFallback()
            acc_meter = AverageMeterFallback()
            top5_meter = AverageMeterFallback()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            # Progress bar (only on main process)
            if self.is_main_process:
                pbar = tqdm(val_loader, desc='Validation')
                data_iter = pbar
            else:
                data_iter = val_loader
            
            for data, target in data_iter:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        output = model_to_eval(data)
                        loss = criterion(output, target)
                else:
                    output = model_to_eval(data)
                    loss = criterion(output, target)
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                acc = pred.eq(target).float().mean()
                
                # Calculate top-5 accuracy
                _, top5_pred = output.topk(5, dim=1)
                top5_acc = top5_pred.eq(target.view(-1, 1).expand_as(top5_pred)).float().sum(dim=1).mean()
                
                # Update metrics
                loss_meter.update(loss.item(), batch_size)
                acc_meter.update(acc.item(), batch_size)
                top5_meter.update(top5_acc.item(), batch_size)
                
                # Store predictions
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Update progress bar (only on main process)
                if self.is_main_process:
                    pbar.set_postfix({
                        'Loss': f'{loss_meter.avg:.4f}',
                        'Acc': f'{acc_meter.avg:.4f}',
                        'Top5': f'{top5_meter.avg:.4f}'
                    })
        
        # Restore original model if using EMA
        if use_ema and self.ema_model is not None and hasattr(self.ema_model, 'restore'):
            self.ema_model.restore()
        
        # Reduce metrics across all processes for distributed training
        if self.distributed:
            loss_tensor = torch.tensor(loss_meter.avg, device=self.device)
            acc_tensor = torch.tensor(acc_meter.avg, device=self.device)
            top5_tensor = torch.tensor(top5_meter.avg, device=self.device)
            
            loss_reduced = self._reduce_tensor(loss_tensor)
            acc_reduced = self._reduce_tensor(acc_tensor)
            top5_reduced = self._reduce_tensor(top5_tensor)
            
            return {
                'loss': loss_reduced.item(),
                'accuracy': acc_reduced.item(),
                'top5_accuracy': top5_reduced.item(),
                'predictions': np.array(all_predictions),
                'targets': np.array(all_targets)
            }
        else:
            return {
                'loss': loss_meter.avg,
                'accuracy': acc_meter.avg,
                'top5_accuracy': top5_meter.avg,
                'predictions': np.array(all_predictions),
                'targets': np.array(all_targets)
            }
    
    def train(
        self,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler=None,
        num_epochs: int = 100,
        save_best: bool = True,
        save_last: bool = True,
        early_stopping_patience: Optional[int] = None,
        monitor_metric: str = 'accuracy'
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        if self._should_log():
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Device: {self.device}")
            print(f"AMP: {self.use_amp}")
            print(f"EMA: {self.use_ema}")
            print(f"Distributed: {self.distributed}")
            if self.distributed:
                print(f"World size: {self.world_size}, Local rank: {self.local_rank}")
        
        best_metric = 0.0 if 'acc' in monitor_metric else float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Set epoch for distributed sampler
            if self.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            if self._should_log():
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch(
                train_loader, optimizer, criterion, scheduler, epoch
            )
            
            # Validation
            val_metrics = self.validate(val_loader, criterion)
            
            # Validation with EMA (if available)
            if self.ema_model is not None:
                ema_val_metrics = self.validate(val_loader, criterion, use_ema=True)
                if self._should_log():
                    print(f"EMA Val - Loss: {ema_val_metrics['loss']:.4f}, "
                          f"Acc: {ema_val_metrics['accuracy']:.4f}, "
                          f"Top5: {ema_val_metrics['top5_accuracy']:.4f}")
            
            # Log epoch metrics
            epoch_metrics = {
                'train/epoch_loss': train_metrics['loss'],
                'train/epoch_accuracy': train_metrics['accuracy'],
                'val/epoch_loss': val_metrics['loss'],
                'val/epoch_accuracy': val_metrics['accuracy'],
                'val/epoch_top5_accuracy': val_metrics['top5_accuracy'],
                'epoch': epoch
            }
            
            if self.ema_model is not None:
                epoch_metrics.update({
                    'val/ema_epoch_loss': ema_val_metrics['loss'],
                    'val/ema_epoch_accuracy': ema_val_metrics['accuracy'],
                    'val/ema_epoch_top5_accuracy': ema_val_metrics['top5_accuracy']
                })
            
            if self._should_log():
                self._log_metrics(epoch_metrics)
            
            # Update history
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                if key not in ['predictions', 'targets']:
                    self.history[f'val_{key}'].append(value)
            
            # Check for best model
            current_metric = val_metrics[monitor_metric]
            is_best = False
            
            if 'acc' in monitor_metric:
                if current_metric > best_metric:
                    best_metric = current_metric
                    is_best = True
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                if current_metric < best_metric:
                    best_metric = current_metric
                    is_best = True
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # Save checkpoints (only on main process)
            if self._should_log():
                if save_best and is_best:
                    self.save_checkpoint(epoch, optimizer, scheduler, 'best')
                
                if save_last:
                    self.save_checkpoint(epoch, optimizer, scheduler, 'last')
            
            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                if self._should_log():
                    print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
                break
            
            if self._should_log():
                print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                      f"Top5: {val_metrics['top5_accuracy']:.4f}")
                print(f"Best {monitor_metric}: {best_metric:.4f}")
        
        if self._should_log():
            print("\nTraining completed!")
        return dict(self.history)
    
    def save_checkpoint(
        self,
        epoch: int,
        optimizer,
        scheduler=None,
        suffix: str = 'last'
    ):
        """Save training checkpoint."""
        # Get model state dict (handle DDP wrapper)
        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'history': dict(self.history),
            'global_step': self.global_step
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if self.ema_model is not None:
            if hasattr(self.ema_model, 'state_dict'):
                checkpoint['ema_state_dict'] = self.ema_model.state_dict()
            else:
                checkpoint['ema_shadow'] = self.ema_model.shadow
        
        checkpoint_path = self.save_dir / f"{self.experiment_name}_{suffix}.pth"
        torch.save(checkpoint, checkpoint_path)
        if self._should_log():
            print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        optimizer=None,
        scheduler=None,
        load_ema: bool = True
    ):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model (handle DDP wrapper)
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load EMA
        if load_ema and self.ema_model is not None:
            if 'ema_state_dict' in checkpoint:
                self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            elif 'ema_shadow' in checkpoint:
                self.ema_model.shadow = checkpoint['ema_shadow']
        
        # Load training state
        self.history = defaultdict(list, checkpoint.get('history', {}))
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        
        if self._should_log():
            print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint.get('epoch', 0)
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb and/or console."""
        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        if not self.history:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        if 'train_loss' in self.history and 'val_loss' in self.history:
            axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
            axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy plot
        if 'train_accuracy' in self.history and 'val_accuracy' in self.history:
            axes[0, 1].plot(self.history['train_accuracy'], label='Train Acc')
            axes[0, 1].plot(self.history['val_accuracy'], label='Val Acc')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate plot
        if 'train_learning_rate' in self.history:
            axes[1, 0].plot(self.history['train_learning_rate'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].grid(True)
        
        # Top-5 accuracy plot
        if 'val_top5_accuracy' in self.history:
            axes[1, 1].plot(self.history['val_top5_accuracy'], label='Val Top-5')
            axes[1, 1].set_title('Top-5 Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Top-5 Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved: {save_path}")
        
        plt.show()
    
    def save_history(self, save_path: str):
        """Save training history to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        history_json = {}
        for key, value in self.history.items():
            if isinstance(value, np.ndarray):
                history_json[key] = value.tolist()
            else:
                history_json[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"Training history saved: {save_path}")


def launch_distributed_training(
    train_fn: Callable,
    world_size: int,
    backend: str = 'nccl',
    **kwargs
):
    """Launch distributed training across multiple GPUs.
    
    Args:
        train_fn: Training function that takes (rank, world_size, **kwargs)
        world_size: Number of processes (GPUs) to use
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        **kwargs: Additional arguments passed to train_fn
    """
    if world_size <= 1:
        print("World size <= 1, running single GPU training")
        train_fn(0, 1, **kwargs)
        return
    
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to single GPU training")
        train_fn(0, 1, **kwargs)
        return
    
    if torch.cuda.device_count() < world_size:
        print(f"Only {torch.cuda.device_count()} GPUs available, using {torch.cuda.device_count()} instead of {world_size}")
        world_size = torch.cuda.device_count()
    
    print(f"Launching distributed training on {world_size} GPUs")
    mp.spawn(
        train_fn,
        args=(world_size, kwargs),
        nprocs=world_size,
        join=True
    )


def distributed_train_worker(rank: int, world_size: int, config: dict):
    """Distributed training worker function.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Training configuration dictionary
    """
    # Setup distributed training
    TrainingEngine.setup_distributed(rank, world_size)
    
    try:
        # Set device
        device = torch.device(f'cuda:{rank}')
        
        # Get model, datasets, etc. from config
        model = config['model_fn']().to(device)
        train_dataset = config['train_dataset']
        val_dataset = config['val_dataset']
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 32),
            sampler=train_sampler,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 32),
            sampler=val_sampler,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Setup optimizer and criterion
        optimizer = config['optimizer_fn'](model.parameters())
        criterion = config['criterion_fn']()
        scheduler = config.get('scheduler_fn', lambda opt: None)(optimizer)
        
        # Create training engine
        engine = TrainingEngine(
            model=model,
            device=device,
            distributed=True,
            local_rank=rank,
            world_size=world_size,
            **config.get('engine_kwargs', {})
        )
        
        # Train
        history = engine.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            **config.get('train_kwargs', {})
        )
        
        if rank == 0:
            print("Distributed training completed successfully!")
            if history:
                print(f"Final train accuracy: {history['train_accuracy'][-1]:.4f}")
                print(f"Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    
    finally:
        # Cleanup
        TrainingEngine.cleanup_distributed()


def test_training_engine():
    """Test the training engine with a simple model and dataset."""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            return self.features(x)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    
    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training engine
    engine = TrainingEngine(
        model=model,
        device=device,
        use_amp=True,
        use_ema=False,
        experiment_name="test_experiment"
    )
    
    # Train
    history = engine.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=2,
        save_best=True,
        save_last=True
    )
    
    print("Training completed successfully!")
    print(f"Final train accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # Plot training history
    engine.plot_training_history("training_history.png")
    
    return engine, history


def test_distributed_training():
    """Test distributed training with multiple GPUs."""
    import torchvision
    import torchvision.transforms as transforms
    
    # Simple model factory
    def create_model():
        class SimpleModel(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, num_classes)
                )
            
            def forward(self, x):
                return self.features(x)
        
        return SimpleModel()
    
    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Training configuration
    config = {
        'model_fn': create_model,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'batch_size': 64,
        'num_workers': 4,
        'optimizer_fn': lambda params: torch.optim.Adam(params, lr=0.001),
        'criterion_fn': lambda: nn.CrossEntropyLoss(),
        'scheduler_fn': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1),
        'engine_kwargs': {
            'use_amp': True,
            'use_ema': False,
            'experiment_name': 'distributed_test'
        },
        'train_kwargs': {
            'num_epochs': 2,
            'save_best': True,
            'save_last': True
        }
    }
    
    # Launch distributed training
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    launch_distributed_training(
        train_fn=distributed_train_worker,
        world_size=world_size,
        **config
    )


if __name__ == "__main__":
    # Test single GPU training
    print("Testing single GPU training...")
    test_training_engine()
    
    # Test distributed training if multiple GPUs available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("\nTesting distributed training...")
        test_distributed_training()
    else:
        print("\nSkipping distributed training test (requires multiple GPUs)")