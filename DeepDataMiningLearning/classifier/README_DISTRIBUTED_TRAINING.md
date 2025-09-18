# Multi-GPU Distributed Training Guide

This guide explains how to use the multi-GPU distributed training functionality added to the `TrainingEngine` class.

## Overview

The `TrainingEngine` now supports distributed training across multiple GPUs using PyTorch's `DistributedDataParallel` (DDP). This allows you to:

- Scale training across multiple GPUs on a single machine
- Achieve faster training times with proper data parallelism
- Maintain training stability with synchronized batch normalization
- Handle distributed logging and checkpointing

## Key Features

### Distributed Training Support
- **Automatic DDP Setup**: The engine automatically wraps your model with `DistributedDataParallel`
- **Synchronized BatchNorm**: Uses `SyncBatchNorm` for consistent normalization across GPUs
- **Distributed Sampling**: Handles data distribution across processes automatically
- **Gradient Synchronization**: Ensures gradients are properly averaged across all GPUs

### Logging and Monitoring
- **Main Process Logging**: Only the main process (rank 0) handles logging and progress bars
- **Distributed Metrics**: Validation metrics are properly reduced across all processes
- **Checkpoint Management**: Model state dictionaries are handled correctly for DDP models

### Utility Functions
- **Easy Launch**: `launch_distributed_training()` function for simple multi-GPU setup
- **Worker Function**: `distributed_train_worker()` handles the distributed training workflow
- **Automatic Fallback**: Falls back to single-GPU training if distributed setup fails

## Usage

### Method 1: Using the TrainingEngine Directly

```python
import torch
from training_engine import TrainingEngine
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Initialize distributed training (call this in each process)
TrainingEngine.setup_distributed(rank, world_size)

# Create model and move to GPU
model = YourModel().to(device)

# Create distributed samplers
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

# Create data loaders with samplers
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

# Create training engine with distributed=True
engine = TrainingEngine(
    model=model,
    device=device,
    distributed=True,
    local_rank=rank,
    world_size=world_size
)

# Train as usual
history = engine.train(
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    num_epochs=num_epochs
)

# Cleanup
TrainingEngine.cleanup_distributed()
```

### Method 2: Using the Launch Function (Recommended)

```python
from training_engine import launch_distributed_training, distributed_train_worker

# Define your training configuration
config = {
    'model_fn': lambda: YourModel(),
    'train_dataset': train_dataset,
    'val_dataset': val_dataset,
    'batch_size': 64,
    'num_workers': 4,
    'optimizer_fn': lambda params: torch.optim.Adam(params, lr=0.001),
    'criterion_fn': lambda: nn.CrossEntropyLoss(),
    'scheduler_fn': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=10),
    'engine_kwargs': {
        'use_amp': True,
        'use_ema': False,
        'experiment_name': 'my_experiment'
    },
    'train_kwargs': {
        'num_epochs': 100,
        'save_best': True,
        'save_last': True
    }
}

# Launch distributed training
world_size = torch.cuda.device_count()  # Use all available GPUs
launch_distributed_training(
    train_fn=distributed_train_worker,
    world_size=world_size,
    **config
)
```

### Method 3: Using the Example Script

```bash
# Single GPU training
python distributed_training_example.py --epochs 200 --batch-size 128 --lr 0.1

# Multi-GPU training
python distributed_training_example.py --distributed --epochs 200 --batch-size 128 --lr 0.1

# Multi-GPU with specific number of GPUs
python distributed_training_example.py --distributed --world-size 4 --epochs 200 --batch-size 128

# With mixed precision and EMA
python distributed_training_example.py --distributed --use-amp --use-ema --epochs 200
```

## Configuration Parameters

### TrainingEngine Parameters for Distributed Training

- `distributed` (bool): Enable distributed training mode
- `local_rank` (int): Local rank of the current process
- `world_size` (int): Total number of processes
- `backend` (str): Distributed backend ('nccl' for GPU, 'gloo' for CPU)

### Launch Function Parameters

- `train_fn`: Training function that accepts (rank, world_size, **kwargs)
- `world_size`: Number of processes/GPUs to use
- `backend`: Distributed backend (default: 'nccl')
- `**kwargs`: Additional arguments passed to the training function

## Best Practices

### Batch Size Scaling
- The batch size you specify is per-GPU batch size
- Effective batch size = batch_size × world_size
- You may need to adjust learning rate accordingly (linear scaling rule)

### Learning Rate Scaling
- Common practice: scale learning rate linearly with world size
- Example: if single-GPU LR is 0.1, use 0.1 × world_size for multi-GPU
- Some models may require different scaling strategies

### Data Loading
- Use `pin_memory=True` for faster GPU transfers
- Set `num_workers` appropriately (typically 4-8 per GPU)
- Ensure your dataset can handle multiple worker processes

### Memory Management
- Monitor GPU memory usage across all devices
- Reduce batch size if you encounter OOM errors
- Consider gradient checkpointing for very large models

## Troubleshooting

### Common Issues

1. **NCCL Initialization Errors**
   - Ensure all GPUs are visible: `export CUDA_VISIBLE_DEVICES=0,1,2,3`
   - Check NCCL version compatibility
   - Try using 'gloo' backend as fallback

2. **Hanging During Training**
   - Usually caused by uneven data distribution
   - Ensure all processes have the same number of batches
   - Use `drop_last=True` in DataLoader if needed

3. **Performance Issues**
   - Check GPU utilization across all devices
   - Ensure data loading is not the bottleneck
   - Monitor network bandwidth for multi-node setups

4. **Checkpoint Loading Issues**
   - The engine handles DDP state dict automatically
   - Ensure checkpoint was saved from the main process (rank 0)
   - Use `map_location` when loading on different hardware

### Debug Mode

Set environment variables for debugging:

```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
```

## Performance Expectations

### Scaling Efficiency
- Near-linear speedup for compute-bound workloads
- Efficiency depends on model size, batch size, and communication overhead
- Larger models typically scale better due to higher compute-to-communication ratio

### Memory Usage
- Each GPU holds a full copy of the model
- Gradients are synchronized but not stored redundantly
- Peak memory usage similar to single-GPU training

## Examples

See `distributed_training_example.py` for a complete working example with:
- CIFAR-10 dataset
- ResNet-like model
- Proper data augmentation
- Command-line argument parsing
- Both single-GPU and multi-GPU modes

## Advanced Usage

### Custom Distributed Metrics

The engine provides `_reduce_tensor()` method for custom metric reduction:

```python
# In your custom training loop
custom_metric = torch.tensor(some_value).to(device)
reduced_metric = engine._reduce_tensor(custom_metric)
```

### Mixed Precision Training

Distributed training works seamlessly with automatic mixed precision:

```python
engine = TrainingEngine(
    model=model,
    device=device,
    distributed=True,
    use_amp=True,  # Enable mixed precision
    local_rank=rank,
    world_size=world_size
)
```

### Exponential Moving Average (EMA)

EMA is supported in distributed mode:

```python
engine = TrainingEngine(
    model=model,
    device=device,
    distributed=True,
    use_ema=True,  # Enable EMA
    ema_decay=0.9999,
    local_rank=rank,
    world_size=world_size
)
```

The EMA model is automatically synchronized across processes during validation.