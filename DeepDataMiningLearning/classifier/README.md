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

## 🛠️ Installation

### Requirements

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install transformers datasets
pip install timm
pip install matplotlib seaborn
pip install scikit-learn
pip install tqdm
pip install Pillow

# Optional dependencies
pip install wandb              # For experiment tracking
pip install accelerate         # For HuggingFace training
```

### Quick Setup

```bash
https://github.com/lkk688/DeepDataMiningLearning.git
cd DeepDataMiningLearning/classifier
pip install -r requirements.txt  # If you create one
```

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

## 🏗️ Architecture Overview

### Core Components

1. **UnifiedImageDataset** (`unified_dataset.py`)
   - Handles multiple dataset sources with consistent interface
   - Automatic preprocessing and augmentation
   - Support for train/validation splits

2. **UnifiedImageClassifier** (`unified_model.py`)
   - Unified interface for different model sources
   - Automatic model adaptation for classification tasks
   - Consistent preprocessing pipelines

3. **TrainingEngine** (`training_engine.py`)
   - Custom PyTorch training loop with best practices
   - Advanced features: AMP, EMA, gradient clipping
   - Comprehensive logging and checkpointing

4. **HuggingFaceTrainer** (`hf_trainer.py`)
   - Integration with HuggingFace Trainer API
   - Automatic dataset and model wrapping
   - Advanced training features and callbacks

5. **InferenceEngine** (`inference_engine.py`)
   - Unified inference interface for all model types
   - Batch and single image prediction
   - Performance benchmarking and analysis

6. **Utils** (`utils.py`)
   - Comprehensive visualization tools
   - Model analysis and comparison
   - Experiment configuration management

### Design Principles

- **Modularity**: Each component is self-contained and reusable
- **Extensibility**: Easy to add new model sources or dataset types
- **Consistency**: Unified interfaces across different backends
- **Best Practices**: Follows PyTorch and ML best practices
- **Flexibility**: Supports various training and inference scenarios

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

### Automatic Mixed Precision (AMP)
```bash
python mytorchmodels.py --use_amp
```

### Exponential Moving Average (EMA)
```bash
python mytorchmodels.py --use_ema
```

### Weights & Biases Integration
```bash
python mytorchmodels.py --use_wandb --wandb_project my_project
```

### Early Stopping
```bash
python mytorchmodels.py --early_stopping 5
```

### Gradient Clipping
```bash
python mytorchmodels.py --gradient_clip 1.0
```

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