# Unified Image Classification System

A comprehensive, modular image classification framework that supports multiple model sources (torchvision, timm, HuggingFace) and dataset formats with both custom training loops and HuggingFace Trainer integration.

## 🚀 Features

### Model Support
- **Torchvision Models**: ResNet, VGG, DenseNet, EfficientNet, MobileNet, and more
- **TIMM Models**: 1000+ state-of-the-art models including Vision Transformers, ConvNeXt, etc.
- **HuggingFace Models**: Pre-trained vision models from the HuggingFace Hub

### Dataset Support
- **Torchvision Datasets**: CIFAR-10/100, ImageNet, and other built-in datasets
- **HuggingFace Datasets**: Access to thousands of datasets from the HuggingFace Hub
- **Custom Folder Datasets**: Load your own datasets from folder structures

### Training Options
- **Custom Training Loop**: PyTorch-standard training with advanced features
- **HuggingFace Trainer**: Leverage HuggingFace's powerful Trainer API
- **Advanced Features**: AMP, EMA, gradient clipping, early stopping, and more

### Comprehensive Tools
- **Unified Inference**: Single interface for all model types
- **Rich Visualizations**: Dataset samples, training curves, confusion matrices
- **Performance Analysis**: Model analysis, benchmarking, and comparison tools
- **Experiment Management**: Configuration saving/loading and result tracking

## 📁 Project Structure

```
classifier/
├── mytorchmodels.py          # Main application script
├── unified_dataset.py        # Unified dataset handling
├── unified_model.py          # Unified model interface
├── training_engine.py        # Custom training loop implementation
├── hf_trainer.py            # HuggingFace Trainer integration
├── inference_engine.py       # Unified inference interface
├── utils.py                 # Utility functions and visualizations
├── examples_and_tests.py    # Comprehensive examples and tests
└── README.md               # This file
```

## 🛠️ Installation & Requirements

### System Requirements

- **Python**: 3.8+ (recommended: 3.9 or 3.10)
- **CUDA**: 11.8+ (for GPU acceleration)
- **Memory**: 8GB+ RAM (16GB+ recommended for large models)
- **Storage**: 10GB+ free space (for models and datasets)

### Core Dependencies

```bash
# Deep Learning Framework
torch>=2.0.0                  # PyTorch deep learning framework
torchvision>=0.15.0           # Computer vision models and transforms
torchaudio>=2.0.0             # Audio processing (for completeness)

# Model Libraries
timm>=0.9.0                   # PyTorch Image Models (1000+ pretrained models)
transformers>=4.30.0          # HuggingFace Transformers library
datasets>=2.12.0              # HuggingFace Datasets library
accelerate>=0.20.0            # HuggingFace training acceleration

# Data Processing & Visualization
numpy>=1.21.0                 # Numerical computing
Pillow>=9.0.0                 # Image processing library
matplotlib>=3.5.0             # Plotting and visualization
seaborn>=0.11.0               # Statistical data visualization
opencv-python>=4.7.0         # Computer vision operations

# Machine Learning & Metrics
scikit-learn>=1.1.0           # Machine learning utilities and metrics
scipy>=1.8.0                  # Scientific computing

# Progress & Utilities
tqdm>=4.64.0                  # Progress bars
rich>=12.0.0                  # Rich text and beautiful formatting
```

### Optional Dependencies

```bash
# Experiment Tracking & Monitoring
wandb>=0.15.0                 # Weights & Biases experiment tracking
tensorboard>=2.12.0           # TensorBoard logging
mlflow>=2.3.0                 # MLflow experiment tracking

# Performance & Optimization
onnx>=1.14.0                  # ONNX model format
onnxruntime-gpu>=1.15.0       # ONNX Runtime with GPU support
tensorrt>=8.6.0               # NVIDIA TensorRT for inference optimization
pycuda>=2022.2                # Python CUDA bindings

# Development & Testing
pytest>=7.0.0                 # Testing framework
black>=23.0.0                 # Code formatting
flake8>=5.0.0                 # Code linting
jupyter>=1.0.0                # Jupyter notebooks
ipywidgets>=8.0.0             # Interactive widgets for notebooks

# Additional Data Sources
webdataset>=0.2.0             # WebDataset format support
albumentations>=1.3.0         # Advanced image augmentations
imgaug>=0.4.0                 # Image augmentation library
```

### Installation Methods

#### Method 1: Conda Environment (Recommended)
```bash
# Create conda environment
conda create -n classifier python=3.10
conda activate classifier

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install timm transformers datasets accelerate
pip install matplotlib seaborn scikit-learn tqdm rich Pillow opencv-python
pip install wandb tensorboard  # Optional: experiment tracking
```

#### Method 2: Virtual Environment
```bash
# Create virtual environment
python -m venv classifier_env
source classifier_env/bin/activate  # On Windows: classifier_env\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### Method 3: Docker (Production)
```bash
# Build Docker image
docker build -t classifier:latest .

# Run container with GPU support
docker run --gpus all -v $(pwd):/workspace classifier:latest
```

### Quick Setup

```bash
# Clone repository
git clone https://github.com/lkk688/DeepDataMiningLearning.git
cd DeepDataMiningLearning/classifier

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, timm, transformers; print('Installation successful!')"
```

### Hardware Recommendations

#### For Development & Small Experiments
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **GPU**: GTX 1060 6GB or better
- **Storage**: SSD recommended

#### For Production & Large Models
- **CPU**: 8+ cores (Intel Xeon or AMD EPYC)
- **RAM**: 32GB+
- **GPU**: RTX 3080/4080, A100, or V100
- **Storage**: NVMe SSD with 100GB+ free space

## 🚀 Quick Start

### Basic Usage

```bash
# Train ResNet-18 on CIFAR-10 with custom training loop
python mytorchmodels.py \
    --model resnet18 \
    --model_source torchvision \
    --dataset cifar10 \
    --dataset_source torchvision \
    --epochs 10 \
    --batch_size 32

# Train on local flower_photos dataset
# Define training and test data directories 
# data_dir = './flower_photos/' 
# train_dir = os.path.join(data_dir, 'train/') 
# test_dir = os.path.join(data_dir, 'test/') 
# 
# Classes are folders in each directory with these names 
# classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
python mytorchmodels.py \
    --model resnet18 \
    --model_source torchvision \
    --dataset flower_photos \
    --dataset_source folder \
    --data_dir ./flower_photos \
    --epochs 10 \
    --batch_size 32

# Train Vision Transformer with HuggingFace Trainer
python mytorchmodels.py \
    --model google/vit-base-patch16-224 \
    --model_source huggingface \
    --dataset cifar10 \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 5

# Train TIMM model on custom dataset
python mytorchmodels.py \
    --model efficientnet_b0 \
    --model_source timm \
    --dataset my_dataset \
    --dataset_source folder \
    --data_dir /path/to/your/dataset \
    --epochs 20
```

### Advanced Features

```bash
# Training with advanced features
python mytorchmodels.py \
    --model resnet50 \
    --model_source torchvision \
    --dataset cifar100 \
    --use_amp \
    --use_ema \
    --early_stopping 5 \
    --use_wandb \
    --experiment_name "resnet50_cifar100"

# Evaluation only
python mytorchmodels.py \
    --evaluate_only \
    --load_model ./experiments/my_experiment/final_model.pth

# Dataset visualization only
python mytorchmodels.py \
    --visualize_only \
    --dataset cifar10 \
    --dataset_source torchvision
```

### Inference Engine with Multiple Backends

The inference engine supports multiple backends for optimized inference:

#### Installation

```bash
# Core dependencies (required)
pip install torch torchvision pillow cairosvg

# ONNX Runtime backend (recommended)
pip install onnxruntime onnx

# TensorRT backend (NVIDIA GPUs only)
pip install tensorrt pycuda

# GradCAM visualization (optional)
pip install grad-cam
```

#### Basic Inference

```bash
# Basic inference test
python inference_engine.py --test basic

# Test with sample SVG images
python inference_engine.py --test samples

# Test all backends
python inference_engine.py --test backends

# Benchmark all backends
python inference_engine.py --test benchmark_backends

# Run all tests
python inference_engine.py --test all
```

#### Backend-Specific Usage

```bash
# PyTorch backend (default)
python inference_engine.py --backend pytorch --model resnet18

# ONNX Runtime backend (faster CPU inference)
python inference_engine.py --backend onnxruntime --model resnet18

# TensorRT backend (fastest GPU inference)
python inference_engine.py --backend tensorrt --model resnet18

# Compare all backends with benchmarking
python inference_engine.py --test benchmark_backends --model resnet50
```

#### Advanced Features

```bash
# Inference with GradCAM visualization
python inference_engine.py --gradcam --target_layer layer4 --image path/to/image.jpg

# Batch inference with different batch sizes
python inference_engine.py --batch_size 8 --backend onnxruntime

# Custom model inference
python inference_engine.py --model_path ./my_model.pth --backend pytorch

# Performance profiling
python inference_engine.py --test benchmark_backends --batch_sizes 1,4,8,16
```

## 📊 Examples and Testing

Run comprehensive examples and tests:

```bash
python examples_and_tests.py
```

This will test:
- All dataset sources (torchvision, HuggingFace, folder)
- All model sources (torchvision, timm, HuggingFace)
- Both training methods (custom loop, HuggingFace Trainer)
- Inference capabilities
- Visualization tools
- Configuration management
- Complete integration test

### Inference Engine Testing

Test the inference engine with different backends:

```bash
# Quick functionality test
python inference_engine.py --test basic

# Test sample image processing (SVG support)
python inference_engine.py --test samples

# Test all available backends
python inference_engine.py --test backends

# Benchmark performance across backends
python inference_engine.py --test benchmark_backends

# Comprehensive test suite
python inference_engine.py --test all
```

### Backend Performance Comparison

Example benchmark results on different hardware:

```bash
# CPU Performance (Intel i7-12700K)
python inference_engine.py --test benchmark_backends --model resnet18
# PyTorch: ~15ms per image
# ONNX Runtime: ~8ms per image (1.9x faster)

# GPU Performance (RTX 4080)
python inference_engine.py --test benchmark_backends --model resnet50
# PyTorch: ~2.1ms per image
# ONNX Runtime: ~1.8ms per image
# TensorRT: ~0.9ms per image (2.3x faster)
```

### Real-world Usage Examples

```bash
# Production inference with ONNX Runtime
python inference_engine.py \
    --backend onnxruntime \
    --model resnet50 \
    --batch_size 16 \
    --image_dir ./production_images/

# High-performance GPU inference with TensorRT
python inference_engine.py \
    --backend tensorrt \
    --model efficientnet_b0 \
    --batch_size 32 \
    --optimize_for_inference

# Explainable AI with GradCAM
python inference_engine.py \
    --gradcam \
    --target_layer layer4.1.conv2 \
    --image ./sample_image.jpg \
    --save_cam ./gradcam_output.jpg
```

## 🔧 Configuration Options

### Model Arguments
- `--model`: Model name (e.g., `resnet18`, `efficientnet_b0`, `google/vit-base-patch16-224`)
- `--model_source`: Model source (`torchvision`, `timm`, `huggingface`)
- `--pretrained`: Use pretrained weights (default: True)

### Dataset Arguments
- `--dataset`: Dataset name or path
- `--dataset_source`: Dataset source (`torchvision`, `huggingface`, `folder`)
- `--data_dir`: Data directory for folder datasets
- `--image_size`: Input image size (default: 224)
- `--val_split`: Validation split ratio (default: 0.2)
- `--no_augment`: Disable data augmentation

### Training Arguments
- `--trainer`: Training method (`custom`, `huggingface`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--early_stopping`: Early stopping patience
- `--use_amp`: Use automatic mixed precision
- `--use_ema`: Use exponential moving average
- `--gradient_clip`: Gradient clipping value

### Logging and Saving
- `--experiment_name`: Experiment name (default: 'image_classification')
- `--save_dir`: Directory to save results (default: './experiments')
- `--use_wandb`: Use Weights & Biases logging
- `--wandb_project`: Wandb project name

### Inference Engine Arguments
- `--backend`: Inference backend (`pytorch`, `onnxruntime`, `tensorrt`)
- `--test`: Test mode (`basic`, `samples`, `backends`, `benchmark_backends`, `all`)
- `--model_path`: Path to custom model file
- `--image`: Single image path for inference
- `--image_dir`: Directory containing images for batch inference
- `--batch_sizes`: Comma-separated batch sizes for benchmarking (e.g., `1,4,8,16`)
- `--gradcam`: Enable GradCAM visualization
- `--target_layer`: Target layer for GradCAM (e.g., `layer4.1.conv2`)
- `--save_cam`: Path to save GradCAM visualization
- `--optimize_for_inference`: Apply inference optimizations (TensorRT only)
- `--precision`: Inference precision (`fp32`, `fp16`, `int8`) for TensorRT
- `--workspace_size`: TensorRT workspace size in MB (default: 1024)

## 🏗️ Technical Architecture & Design

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Unified Image Classification System                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │   Data Layer    │    │   Model Layer   │    │ Training Layer  │             │
│  │                 │    │                 │    │                 │             │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │             │
│  │ │Torchvision  │ │    │ │Torchvision  │ │    │ │Custom Loop  │ │             │
│  │ │Datasets     │ │    │ │Models       │ │    │ │(PyTorch)    │ │             │
│  │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │             │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │             │
│  │ │HuggingFace  │ │◄──►│ │TIMM Models  │ │◄──►│ │HuggingFace  │ │             │
│  │ │Datasets     │ │    │ │(1000+)      │ │    │ │Trainer      │ │             │
│  │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │             │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │                 │             │
│  │ │Folder       │ │    │ │HuggingFace  │ │    │                 │             │
│  │ │Structure    │ │    │ │Models       │ │    │                 │             │
│  │ └─────────────┘ │    │ └─────────────┘ │    │                 │             │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘             │
│           │                       │                       │                     │
│           └───────────────────────┼───────────────────────┘                     │
│                                   │                                             │
│  ┌─────────────────────────────────┼─────────────────────────────────┐           │
│  │                    Unified Interface Layer                        │           │
│  │                                 │                                 │           │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐│           │
│  │  │UnifiedDataset   │    │UnifiedModel     │    │InferenceEngine  ││           │
│  │  │- Auto preprocessing│ │- Model adaptation│   │- Batch prediction││          │
│  │  │- Augmentation   │    │- Consistent API │    │- Performance    ││           │
│  │  │- Train/Val split│    │- Multi-source   │    │- Benchmarking   ││           │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘│           │
│  └─────────────────────────────────────────────────────────────────────┘           │
│                                   │                                             │
│  ┌─────────────────────────────────┼─────────────────────────────────┐           │
│  │                    Utilities & Analysis Layer                     │           │
│  │                                 │                                 │           │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐│           │
│  │  │Visualization    │    │Model Analysis   │    │Experiment Mgmt  ││           │
│  │  │- Dataset preview│    │- Performance    │    │- Config saving  ││           │
│  │  │- Training curves│    │- Comparison     │    │- Result tracking││           │
│  │  │- Confusion matrix│   │- Benchmarking   │    │- Reproducibility││           │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘│           │
│  └─────────────────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Core Components Deep Dive

#### 1. UnifiedImageDataset (`unified_dataset.py`)
**Purpose**: Provides a consistent interface across different dataset sources

**Key Features**:
- **Multi-source Support**: Torchvision, HuggingFace, and folder-based datasets
- **Automatic Preprocessing**: Model-specific image transformations
- **Smart Augmentation**: Context-aware data augmentation strategies
- **Flexible Splitting**: Configurable train/validation splits

**Technical Implementation**:
```python
class UnifiedImageDataset:
    def __init__(self, dataset_name, source, **kwargs):
        self.source = source
        self.transforms = self._get_transforms(**kwargs)
        self.dataset = self._load_dataset(dataset_name, **kwargs)
    
    def _get_transforms(self, image_size=224, augment=True):
        # Automatic transform selection based on model requirements
        return create_transform(
            input_size=image_size,
            is_training=augment,
            auto_augment='rand-m9-mstd0.5' if augment else None
        )
```

#### 2. UnifiedImageClassifier (`unified_model.py`)
**Purpose**: Unified interface for different model architectures and sources

**Key Features**:
- **Multi-source Models**: Torchvision, TIMM, HuggingFace integration
- **Automatic Adaptation**: Dynamic classifier head adjustment
- **Consistent API**: Same interface regardless of model source
- **Preprocessing Pipeline**: Model-specific preprocessing

**Technical Implementation**:
```python
class UnifiedImageClassifier:
    def __init__(self, model_name, source, num_classes, **kwargs):
        self.model = self._create_model(model_name, source, **kwargs)
        self.model = self._adapt_classifier(self.model, num_classes)
        self.preprocessor = self._get_preprocessor(source, model_name)
    
    def _adapt_classifier(self, model, num_classes):
        # Dynamic classifier adaptation based on model architecture
        if hasattr(model, 'classifier'):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif hasattr(model, 'head'):
            model.head = nn.Linear(model.head.in_features, num_classes)
```

#### 3. TrainingEngine (`training_engine.py`)
**Purpose**: Advanced PyTorch training loop with modern best practices

**Key Features**:
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) support
- **Model Averaging**: Exponential Moving Average (EMA) for better convergence
- **Gradient Management**: Gradient clipping and accumulation
- **Advanced Scheduling**: Cosine annealing with warmup
- **Comprehensive Logging**: Metrics tracking and visualization

**Technical Implementation**:
```python
class TrainingEngine:
    def __init__(self, model, optimizer, scheduler, **kwargs):
        self.scaler = GradScaler() if kwargs.get('use_amp') else None
        self.ema = ModelEmaV2(model) if kwargs.get('use_ema') else None
        self.gradient_clip = kwargs.get('gradient_clip', 0.0)
    
    def train_epoch(self, dataloader):
        for batch_idx, (data, target) in enumerate(dataloader):
            with autocast(enabled=self.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
```

#### 4. InferenceEngine (`inference_engine.py`)
**Purpose**: High-performance inference with optimization support

**Key Features**:
- **Batch Processing**: Efficient batch inference
- **Performance Optimization**: TensorRT and ONNX Runtime support
- **Benchmarking**: Latency and throughput analysis
- **Multi-format Support**: PyTorch, ONNX, TensorRT models

### Data Flow Architecture

```
Input Image(s) → Preprocessing → Model Inference → Post-processing → Results
     │              │               │                │              │
     │              │               │                │              │
┌────▼────┐    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
│ Image   │    │Transform│    │ Forward │    │Softmax/ │    │Class    │
│Loading  │    │Pipeline │    │ Pass    │    │Argmax   │    │Prediction│
│         │    │         │    │         │    │         │    │         │
│• PIL    │    │• Resize │    │• CNN/   │    │• Prob   │    │• Label  │
│• OpenCV │    │• Norm   │    │  ViT    │    │• Top-K  │    │• Conf   │
│• Tensor │    │• Augment│    │• Batch  │    │• Metrics│    │• Metrics│
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### Memory Management & Optimization

#### Memory Optimization Strategies
1. **Gradient Checkpointing**: Trade compute for memory
2. **Mixed Precision**: FP16 training reduces memory by ~50%
3. **Batch Size Scaling**: Dynamic batch size based on available memory
4. **Model Sharding**: Large model distribution across GPUs

#### Performance Optimization
1. **Compiled Models**: `torch.compile()` for 2x speedup
2. **TensorRT Integration**: Up to 10x inference speedup
3. **ONNX Runtime**: Cross-platform optimized inference
4. **Quantization**: INT8 quantization for deployment

### Extensibility Framework

#### Adding New Model Sources
```python
# Example: Adding a new model source
class CustomModelSource:
    def create_model(self, model_name, **kwargs):
        # Custom model creation logic
        pass
    
    def get_preprocessor(self, model_name):
        # Custom preprocessing pipeline
        pass

# Register new source
MODEL_SOURCES['custom'] = CustomModelSource()
```

#### Adding New Dataset Sources
```python
# Example: Adding a new dataset source
class CustomDatasetSource:
    def load_dataset(self, dataset_name, **kwargs):
        # Custom dataset loading logic
        pass
    
    def get_classes(self, dataset):
        # Extract class information
        pass

# Register new source
DATASET_SOURCES['custom'] = CustomDatasetSource()
```

## 📈 Supported Models

### Torchvision Models
- ResNet (18, 34, 50, 101, 152)
- VGG (11, 13, 16, 19)
- DenseNet (121, 161, 169, 201)
- EfficientNet (B0-B7)
- MobileNet (V2, V3)
- And many more...

### TIMM Models
- Vision Transformers (ViT, DeiT, Swin)
- ConvNeXt
- EfficientNet variants
- RegNet
- ResNeXt
- 1000+ models available

### HuggingFace Models
- Vision Transformers
- CLIP models
- DeiT models
- BEiT models
- Any vision model from HuggingFace Hub

## 📊 Supported Datasets

### Torchvision Datasets
- CIFAR-10/100
- ImageNet
- MNIST
- Fashion-MNIST
- STL-10
- And more...

### HuggingFace Datasets
- CIFAR-10/100
- ImageNet-1k
- Food-101
- Oxford Pets
- Thousands of datasets available

### Custom Datasets
- Folder structure: `dataset/class1/`, `dataset/class2/`, etc.
- Automatic class detection
- Support for various image formats

## 🎯 Use Cases

### Research and Experimentation
- Quick prototyping with different model architectures
- Comparing performance across model families
- Ablation studies with different training configurations

### Production Deployment
- Unified inference interface for model serving
- Performance benchmarking and optimization
- Model analysis and comparison tools

### Educational Purposes
- Learning different model architectures
- Understanding training best practices
- Exploring various datasets and preprocessing techniques

## 🔍 Advanced Features

### Training with Advanced Optimizers and Schedulers
```bash
# Train with LAMB optimizer and cosine learning rate scheduler
python mytorchmodels.py --model resnet50 --dataset cifar10 --epochs 50 --batch_size 64 --lr 0.001

# Available optimizers from timm: sgd, adam, adamw, adamp, adabelief, lamb
# The get_optimizer() function uses timm.optim.create_optimizer_v2() for advanced optimization

# Cosine learning rate scheduler with warmup (automatically configured)
# Uses timm.scheduler.CosineLRScheduler with warmup and cycle support
```

### Data Augmentation Features
```bash
# Enable automatic augmentation (RandAugment)
python mytorchmodels.py --model resnet50 --dataset cifar10
# Default uses timm's auto_augment="rand-m9-mstd0.5" for training

# Disable augmentation for comparison
python mytorchmodels.py --model resnet50 --dataset cifar10 --no_augment

# Preview augmentation effects
python examples_and_tests.py  # Includes augmentation preview functionality
```

### Advanced Training Features
```bash
# Use Automatic Mixed Precision (AMP) for faster training
python mytorchmodels.py --model resnet50 --use_amp --epochs 50

# Use Exponential Moving Average (EMA) for better model performance
python mytorchmodels.py --model resnet50 --use_ema --epochs 50

# Apply gradient clipping for stable training
python mytorchmodels.py --model resnet50 --gradient_clip 1.0 --epochs 50

# Combine all advanced features
python mytorchmodels.py --model efficientnet_b0 --model_source timm --use_amp --use_ema --gradient_clip 1.0 --epochs 100
```

### Model Sources and Advanced Models
```bash
# Train with timm models (1000+ pre-trained models)
python mytorchmodels.py --model_source timm --model efficientnet_b0 --dataset imagenet
python mytorchmodels.py --model_source timm --model vit_base_patch16_224 --dataset cifar10

# Train with Hugging Face models
python mytorchmodels.py --model_source huggingface --model microsoft/resnet-50

# Use different image sizes for different models
python mytorchmodels.py --model_source timm --model vit_large_patch16_224 --image_size 224
```

### Weights & Biases Integration
```bash
python mytorchmodels.py --use_wandb --wandb_project my_project
```

### Early Stopping
```bash
python mytorchmodels.py --early_stopping 5
```

### Dataset Visualization and Analysis
```bash
# Visualize dataset samples and class distribution
python mytorchmodels.py --visualize_only --dataset cifar10

# Create augmentation preview (shows original vs augmented images)
# Available in utils.py: create_data_augmentation_preview()
```

### Additional Advanced Features Available in Code

#### Timm Data Augmentation Options
The codebase supports timm's advanced augmentation through `mydataset.py`:
- **RandAugment**: `auto_augment="rand-m9-mstd0.5"` (default when augmentation enabled)
- **AutoAugment**: Can be configured in `create_transform()`
- **TrivialAugment**: Available through timm's transform factory

#### Advanced Optimizers (via timm)
Available through `get_optimizer()` function:
- **LAMB**: Large batch optimization
- **AdamP**: Adaptive gradient clipping
- **AdaBelief**: Adapting stepsizes by belief in gradient direction
- **AdamW**: Adam with decoupled weight decay

#### Learning Rate Schedulers (via timm)
Available through `get_scheduler()` function:
- **CosineLRScheduler**: Cosine annealing with warmup
- **Warmup**: Configurable warmup periods
- **Cycle Support**: Multiple training cycles

#### Model Architecture Features
- **EMA (Exponential Moving Average)**: Better model stability
- **AMP (Automatic Mixed Precision)**: Faster training with mixed precision
- **Gradient Clipping**: Prevents gradient explosion
- **Multiple Model Sources**: torchvision, timm (1000+ models), huggingface

#### Data Loading Enhancements
- **Unified Dataset Interface**: Consistent API across different data sources
- **Automatic Preprocessing**: Model-specific preprocessing pipelines
- **Flexible Data Sources**: torchvision datasets, folder structure, huggingface datasets

## 🤗 Advanced HuggingFace Integration Examples

### HuggingFace Dataset Training Examples

#### 1. Training on Popular HuggingFace Vision Datasets

```bash
# Train Vision Transformer on CIFAR-10 from HuggingFace Hub
python mytorchmodels.py \
    --model google/vit-base-patch16-224 \
    --model_source huggingface \
    --dataset cifar10 \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 10 \
    --batch_size 32 \
    --lr 2e-5 \
    --use_wandb \
    --experiment_name "vit_cifar10_hf"

# Train on Food-101 dataset with DeiT model
python mytorchmodels.py \
    --model facebook/deit-base-distilled-patch16-224 \
    --model_source huggingface \
    --dataset food101 \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 20 \
    --batch_size 16 \
    --lr 1e-4 \
    --use_amp \
    --early_stopping 5

# Train on Oxford-IIIT Pet Dataset
python mytorchmodels.py \
    --model microsoft/resnet-50 \
    --model_source huggingface \
    --dataset oxford-iiit-pet \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 15 \
    --batch_size 32 \
    --lr 3e-4 \
    --use_ema
```

#### 2. Custom HuggingFace Dataset Integration

```bash
# Train on custom dataset uploaded to HuggingFace Hub
python mytorchmodels.py \
    --model google/vit-base-patch16-224 \
    --model_source huggingface \
    --dataset username/my-custom-dataset \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 25 \
    --batch_size 24 \
    --lr 2e-5 \
    --gradient_clip 1.0 \
    --use_wandb \
    --wandb_project "custom_dataset_training"

# Train with specific dataset configuration/split
python mytorchmodels.py \
    --model facebook/convnext-base-224 \
    --model_source huggingface \
    --dataset imagenet-1k \
    --dataset_source huggingface \
    --dataset_config "default" \
    --dataset_split "train[:10%]" \
    --trainer huggingface \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4
```

### Advanced HuggingFace Model Training

#### 1. Vision Transformer Variants

```bash
# Train different ViT variants
# Base ViT
python mytorchmodels.py \
    --model google/vit-base-patch16-224 \
    --model_source huggingface \
    --dataset cifar100 \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 50 \
    --batch_size 32 \
    --lr 3e-4 \
    --use_amp \
    --use_ema

# Large ViT for better performance
python mytorchmodels.py \
    --model google/vit-large-patch16-224 \
    --model_source huggingface \
    --dataset imagenet-1k \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --gradient_accumulation_steps 4 \
    --use_amp

# DeiT with distillation
python mytorchmodels.py \
    --model facebook/deit-base-distilled-patch16-224 \
    --model_source huggingface \
    --dataset beans \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 20 \
    --batch_size 32 \
    --lr 5e-5
```

#### 2. ConvNeXt and Modern CNN Architectures

```bash
# ConvNeXt training
python mytorchmodels.py \
    --model facebook/convnext-base-224 \
    --model_source huggingface \
    --dataset cifar10 \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 30 \
    --batch_size 32 \
    --lr 4e-3 \
    --weight_decay 0.05 \
    --use_amp

# Swin Transformer
python mytorchmodels.py \
    --model microsoft/swin-base-patch4-window7-224 \
    --model_source huggingface \
    --dataset imagenet-1k \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 90 \
    --batch_size 16 \
    --lr 1e-3 \
    --use_amp \
    --early_stopping 10
```

#### 3. Multi-Modal and Specialized Models

```bash
# CLIP model fine-tuning for classification
python mytorchmodels.py \
    --model openai/clip-vit-base-patch32 \
    --model_source huggingface \
    --dataset cifar10 \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 15 \
    --batch_size 64 \
    --lr 1e-5 \
    --freeze_backbone \
    --use_amp

# BEiT (BERT pre-training for images)
python mytorchmodels.py \
    --model microsoft/beit-base-patch16-224 \
    --model_source huggingface \
    --dataset food101 \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 25 \
    --batch_size 24 \
    --lr 2e-4
```

### Advanced Training Configurations

#### 1. Hyperparameter Optimization with HuggingFace

```python
# Example configuration for hyperparameter search
# Save as hf_training_config.py

from transformers import TrainingArguments
import optuna

def create_training_args(trial=None):
    if trial:
        # Optuna hyperparameter optimization
        lr = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.3)
    else:
        # Default values
        lr = 2e-5
        batch_size = 32
        weight_decay = 0.01
    
    return TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=True,  # Mixed precision
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

# Run hyperparameter optimization
python mytorchmodels.py \
    --model google/vit-base-patch16-224 \
    --model_source huggingface \
    --dataset cifar10 \
    --dataset_source huggingface \
    --trainer huggingface \
    --hyperparameter_search \
    --n_trials 20
```

#### 2. Advanced Data Augmentation with HuggingFace

```bash
# Custom augmentation pipeline
python mytorchmodels.py \
    --model facebook/deit-base-patch16-224 \
    --model_source huggingface \
    --dataset cifar100 \
    --dataset_source huggingface \
    --trainer huggingface \
    --augmentation_strategy "advanced" \
    --mixup_alpha 0.2 \
    --cutmix_alpha 1.0 \
    --random_erase_prob 0.25 \
    --epochs 100 \
    --batch_size 32
```

#### 3. Multi-GPU and Distributed Training

```bash
# Single-node multi-GPU training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    mytorchmodels.py \
    --model google/vit-large-patch16-224 \
    --model_source huggingface \
    --dataset imagenet-1k \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 90 \
    --batch_size 16 \
    --lr 1e-3 \
    --use_amp \
    --gradient_accumulation_steps 2

# Multi-node distributed training
python -m torch.distributed.launch \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=12345 \
    --nproc_per_node=8 \
    mytorchmodels.py \
    --model facebook/convnext-large-224 \
    --model_source huggingface \
    --dataset imagenet-1k \
    --dataset_source huggingface \
    --trainer huggingface \
    --epochs 300 \
    --batch_size 8 \
    --lr 4e-3
```

### Custom Dataset Creation for HuggingFace

#### 1. Creating and Uploading Custom Datasets

```python
# create_hf_dataset.py - Script to create HuggingFace dataset
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image
from PIL import Image as PILImage
import os

def create_custom_dataset(data_dir):
    """Create HuggingFace dataset from folder structure"""
    
    # Define features
    features = Features({
        'image': Image(),
        'label': ClassLabel(names=sorted(os.listdir(data_dir)))
    })
    
    # Collect data
    data = {'image': [], 'label': []}
    for label_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            data['image'].append(img_path)
            data['label'].append(label_idx)
    
    # Create dataset
    dataset = Dataset.from_dict(data, features=features)
    
    # Split into train/validation
    dataset = dataset.train_test_split(test_size=0.2, stratify_by_column='label')
    
    return DatasetDict({
        'train': dataset['train'],
        'validation': dataset['test']
    })

# Usage
dataset = create_custom_dataset('./my_custom_data')
dataset.push_to_hub("username/my-custom-dataset")
```

#### 2. Advanced Dataset Preprocessing

```python
# advanced_preprocessing.py
from transformers import AutoImageProcessor
from datasets import load_dataset

def preprocess_dataset(dataset_name, model_name):
    """Advanced preprocessing for HuggingFace datasets"""
    
    # Load dataset and processor
    dataset = load_dataset(dataset_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    def transform(examples):
        # Apply model-specific preprocessing
        examples['pixel_values'] = processor(
            examples['image'], 
            return_tensors='pt'
        )['pixel_values']
        return examples
    
    # Apply transformations
    dataset = dataset.with_transform(transform)
    return dataset

# Usage in training script
python mytorchmodels.py \
    --model google/vit-base-patch16-224 \
    --model_source huggingface \
    --dataset username/preprocessed-dataset \
    --dataset_source huggingface \
    --trainer huggingface \
    --custom_preprocessing
```

### Production Deployment with HuggingFace

#### 1. Model Export and Optimization

```bash
# Export trained model to HuggingFace Hub
python mytorchmodels.py \
    --model google/vit-base-patch16-224 \
    --model_source huggingface \
    --load_model ./experiments/my_experiment/final_model.pth \
    --export_to_hub \
    --hub_model_name "username/my-finetuned-vit" \
    --hub_private False

# Convert to ONNX for deployment
python mytorchmodels.py \
    --model username/my-finetuned-vit \
    --model_source huggingface \
    --export_onnx \
    --onnx_path ./models/my_model.onnx \
    --optimize_for_inference
```

#### 2. Inference API Integration

```python
# hf_inference_api.py - Production inference with HuggingFace
from transformers import pipeline
import requests
from PIL import Image

# Local inference
classifier = pipeline(
    "image-classification",
    model="username/my-finetuned-vit",
    device=0  # GPU
)

# Batch inference
images = [Image.open(f"image_{i}.jpg") for i in range(10)]
results = classifier(images, batch_size=8)

# HuggingFace Inference API
API_URL = "https://api-inference.huggingface.co/models/username/my-finetuned-vit"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_api(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

## 📝 Example Workflows

### 1. Quick Experiment
```bash
# Train a small model quickly
python mytorchmodels.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 5 \
    --batch_size 64
```

### 2. Production Training
```bash
# Full training with all features
python mytorchmodels.py \
    --model efficientnet_b3 \
    --model_source timm \
    --dataset my_custom_dataset \
    --dataset_source folder \
    --data_dir /path/to/data \
    --epochs 100 \
    --batch_size 32 \
    --use_amp \
    --use_ema \
    --early_stopping 10 \
    --use_wandb \
    --experiment_name "production_model"
```

### 3. Model Comparison
```bash
# Compare different models
for model in resnet18 resnet50 efficientnet_b0; do
    python mytorchmodels.py \
        --model $model \
        --dataset cifar100 \
        --epochs 20 \
        --experiment_name "comparison_${model}"
done
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 16`
   - Use gradient accumulation in HuggingFace Trainer
   - Enable AMP: `--use_amp`

2. **Model Not Found**
   - Check model name spelling
   - Verify model source is correct
   - For HuggingFace models, ensure they support image classification

3. **Dataset Loading Issues**
   - Verify dataset path exists
   - Check folder structure for custom datasets
   - Ensure proper permissions

4. **Import Errors**
   - Install missing dependencies
   - Check Python environment
   - Verify package versions

### Performance Tips

1. **Speed Up Training**
   - Use AMP: `--use_amp`
   - Increase batch size if memory allows
   - Use more workers: `--num_workers 8`
   - Use faster models for prototyping

2. **Improve Accuracy**
   - Use pretrained models: `--pretrained`
   - Enable data augmentation (default)
   - Try different learning rates
   - Use larger models if computational budget allows

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Add support for new model sources
2. Implement additional dataset formats
3. Add new visualization tools
4. Improve documentation
5. Report bugs and suggest features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Torchvision](https://pytorch.org/vision/) for computer vision models and datasets
- [TIMM](https://github.com/rwightman/pytorch-image-models) for state-of-the-art models
- [HuggingFace](https://huggingface.co/) for transformers and datasets
- [Weights & Biases](https://wandb.ai/) for experiment tracking

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TIMM Documentation](https://rwightman.github.io/pytorch-image-models/)
- [HuggingFace Documentation](https://huggingface.co/docs)
- [Computer Vision Best Practices](https://github.com/microsoft/computervision-recipes)

---

**Happy Training! 🚀**