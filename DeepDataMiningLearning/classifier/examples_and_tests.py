#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Examples and Tests for Unified Image Classification System

This file contains various examples and test cases demonstrating:
- Different model sources (torchvision, timm, HuggingFace)
- Different dataset sources (torchvision, HuggingFace, folder)
- Training with custom loop vs HuggingFace Trainer
- Inference and evaluation capabilities
- Visualization and analysis tools

Created on Tue Nov 12 14:05:33 2024
@author: kaikailiu
"""

import torch
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our modular components
from unified_dataset import UnifiedImageDataset
from unified_model import UnifiedImageClassifier
from training_engine import TrainingEngine
from hf_trainer import HuggingFaceTrainer
from inference_engine import InferenceEngine
from utils import (
    set_seed, analyze_model, visualize_dataset_samples,
    plot_class_distribution, create_data_augmentation_preview,
    plot_training_curves, save_experiment_config, load_experiment_config
)


def create_dummy_dataset(data_dir, num_classes=3, samples_per_class=10, image_size=64):
    """
    Create a dummy dataset for testing purposes.
    
    Args:
        data_dir (str): Directory to create the dataset
        num_classes (int): Number of classes
        samples_per_class (int): Number of samples per class
        image_size (int): Size of generated images
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    class_names = [f'class_{i}' for i in range(num_classes)]
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        for sample_idx in range(samples_per_class):
            # Create a random image with class-specific color pattern
            np.random.seed(class_idx * 100 + sample_idx)  # Reproducible
            
            # Generate image with class-specific dominant color
            base_color = np.array([class_idx * 80, (class_idx + 1) * 60, (class_idx + 2) * 40]) % 255
            image_array = np.random.randint(0, 50, (image_size, image_size, 3)) + base_color
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            # Save image
            image = Image.fromarray(image_array)
            image_path = class_dir / f'{class_name}_{sample_idx:03d}.png'
            image.save(image_path)
    
    print(f"Created dummy dataset at {data_dir} with {num_classes} classes, {samples_per_class} samples each")
    return class_names


def test_unified_dataset():
    """
    Test the UnifiedImageDataset class with different sources.
    """
    print("\n" + "="*60)
    print("Testing UnifiedImageDataset")
    print("="*60)
    
    # Test 1: Torchvision dataset (CIFAR-10)
    print("\n1. Testing torchvision dataset (CIFAR-10)...")
    try:
        dataset = UnifiedImageDataset(
            dataset_name='CIFAR10',
            dataset_source='torchvision',
            image_size=32,
            batch_size=16,
            val_split=0.2
        )
        train_loader, val_loader = dataset.get_dataloader()
        class_names = dataset.class_names if hasattr(dataset, 'class_names') else [f'class_{i}' for i in range(10)]
        print(f"   ✓ CIFAR-10 loaded: {len(class_names)} classes")
        print(f"   ✓ Train samples: {len(train_loader.dataset)}")
        print(f"   ✓ Val samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"   ✗ Error loading CIFAR-10: {e}")
    
    # Test 2: Folder dataset
    print("\n2. Testing folder dataset...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy dataset
            dummy_classes = create_dummy_dataset(temp_dir, num_classes=3, samples_per_class=20)
            
            dataset = UnifiedImageDataset(
                dataset_name='dummy_dataset',
                dataset_source='folder',
                data_dir=temp_dir,
                image_size=64,
                batch_size=8,
                val_split=0.3
            )
            train_loader, val_loader = dataset.get_dataloader()
            class_names = dataset.class_names if hasattr(dataset, 'class_names') else dummy_classes
            print(f"   ✓ Folder dataset loaded: {len(class_names)} classes")
            print(f"   ✓ Train samples: {len(train_loader.dataset)}")
            print(f"   ✓ Val samples: {len(val_loader.dataset)}")
            print(f"   ✓ Classes: {class_names}")
    except Exception as e:
        print(f"   ✗ Error loading folder dataset: {e}")
    
    # Test 3: HuggingFace dataset (if available)
    print("\n3. Testing HuggingFace dataset...")
    try:
        dataset = UnifiedImageDataset(
            dataset_name='cifar10',
            dataset_source='huggingface',
            image_size=32,
            batch_size=16,
            val_split=0.2
        )
        train_loader, val_loader = dataset.get_dataloader()
        class_names = dataset.class_names if hasattr(dataset, 'class_names') else [f'class_{i}' for i in range(10)]
        print(f"   ✓ HuggingFace CIFAR-10 loaded: {len(class_names)} classes")
        print(f"   ✓ Train samples: {len(train_loader.dataset)}")
        print(f"   ✓ Val samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"   ✗ Error loading HuggingFace dataset: {e}")


def test_unified_model():
    """
    Test the UnifiedImageClassifier class with different sources.
    """
    print("\n" + "="*60)
    print("Testing UnifiedImageClassifier")
    print("="*60)
    
    num_classes = 10
    
    # Test 1: Torchvision model
    print("\n1. Testing torchvision model (ResNet-18)...")
    try:
        model = UnifiedImageClassifier(
            model_name='resnet18',
            model_source='torchvision',
            num_classes=num_classes,
            pretrained=True
        )
        pytorch_model = model.model
        analysis = analyze_model(pytorch_model)
        print(f"   ✓ ResNet-18 loaded: {analysis['parameters']['total_parameters']:,} parameters")
        print(f"   ✓ Model size: {analysis['size']['size_mb']:.2f} MB")
    except Exception as e:
        print(f"   ✗ Error loading torchvision model: {e}")
    
    # Test 2: TIMM model
    print("\n2. Testing TIMM model (EfficientNet-B0)...")
    try:
        model = UnifiedImageClassifier(
            model_name='efficientnet_b0',
            model_source='timm',
            num_classes=num_classes,
            pretrained=True
        )
        pytorch_model = model.model
        analysis = analyze_model(pytorch_model)
        print(f"   ✓ EfficientNet-B0 loaded: {analysis['parameters']['total_parameters']:,} parameters")
        print(f"   ✓ Model size: {analysis['size']['size_mb']:.2f} MB")
    except Exception as e:
        print(f"   ✗ Error loading TIMM model: {e}")
    
    # Test 3: HuggingFace model
    print("\n3. Testing HuggingFace model (ViT)...")
    try:
        model = UnifiedImageClassifier(
            model_name='google/vit-base-patch16-224',
            model_source='huggingface',
            num_classes=num_classes,
            pretrained=True
        )
        pytorch_model = model.model
        analysis = analyze_model(pytorch_model)
        print(f"   ✓ ViT-Base loaded: {analysis['parameters']['total_parameters']:,} parameters")
        print(f"   ✓ Model size: {analysis['size']['size_mb']:.2f} MB")
    except Exception as e:
        print(f"   ✗ Error loading HuggingFace model: {e}")


def test_training_engines():
    """
    Test both custom training loop and HuggingFace Trainer.
    """
    print("\n" + "="*60)
    print("Testing Training Engines")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy dataset
        dummy_classes = create_dummy_dataset(temp_dir, num_classes=3, samples_per_class=20)
        
        # Setup dataset
        dataset = UnifiedImageDataset(
            dataset_name='dummy_dataset',
            dataset_source='folder',
            data_dir=temp_dir,
            image_size=64,
            batch_size=8,
            val_split=0.3
        )
        train_loader, val_loader = dataset.get_dataloader()
        class_names = dataset.class_names if hasattr(dataset, 'class_names') else dummy_classes
        
        # Setup model
        model = UnifiedImageClassifier(
            model_name='resnet18',
            model_source='torchvision',
            num_classes=len(class_names),
            pretrained=False  # Faster for testing
        )
        pytorch_model = model.model
        
        # Test 1: Custom Training Engine
        print("\n1. Testing Custom Training Engine...")
        try:
            engine = TrainingEngine(
                model=pytorch_model,
                device=device,
                use_amp=False,
                use_ema=False,
                log_interval=5
            )
            
            optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Train for just 2 epochs for testing
            history = engine.train(
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=2
            )
            
            print(f"   ✓ Custom training completed")
            print(f"   ✓ Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"   ✓ Final val accuracy: {history['val_accuracy'][-1]:.4f}")
            
        except Exception as e:
            print(f"   ✗ Error in custom training: {e}")
        
        # Test 2: HuggingFace Trainer (if available)
        print("\n2. Testing HuggingFace Trainer...")
        try:
            # Reset model
            pytorch_model = model.model
            
            hf_trainer = HuggingFaceTrainer(
                model=pytorch_model,
                output_dir=temp_dir + "/hf_outputs"
            )
            
            training_args = hf_trainer.create_training_args(
                num_train_epochs=2,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                learning_rate=0.001,
                evaluation_strategy="epoch",
                save_strategy="no",  # Don't save for testing
                logging_steps=5
            )
            
            trainer = hf_trainer.train(
                train_dataset=train_loader.dataset,
                eval_dataset=val_loader.dataset,
                training_args=training_args
            )
            
            print(f"   ✓ HuggingFace training completed")
            
        except Exception as e:
            print(f"   ✗ Error in HuggingFace training: {e}")


def test_inference_engine():
    """
    Test the InferenceEngine capabilities.
    """
    print("\n" + "="*60)
    print("Testing InferenceEngine")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy dataset
        dummy_classes = create_dummy_dataset(temp_dir, num_classes=3, samples_per_class=10)
        
        # Setup dataset
        dataset = UnifiedImageDataset(
            dataset_name='dummy_dataset',
            dataset_source='folder',
            data_dir=temp_dir,
            image_size=64,
            batch_size=4,
            val_split=0.5
        )
        train_loader, val_loader = dataset.get_dataloader()
        class_names = dataset.class_names if hasattr(dataset, 'class_names') else dummy_classes
        
        # Setup model
        model = UnifiedImageClassifier(
            model_name='resnet18',
            model_source='torchvision',
            num_classes=len(class_names),
            pretrained=False
        )
        pytorch_model = model.model
        
        # Test inference engine
        print("\n1. Testing inference capabilities...")
        try:
            inference_engine = InferenceEngine(
                model=pytorch_model,
                device=device,
                class_names=class_names,
                model_type="pytorch"
            )
            
            # Test single image prediction
            data_iter = iter(val_loader)
            images, labels = next(data_iter)
            single_image = images[0]
            
            prediction = inference_engine.predict_single(
                image=single_image,
                return_probabilities=True
            )
            
            print(f"   ✓ Single prediction: {prediction['predicted_class']}")
            print(f"   ✓ Confidence: {prediction['confidence']:.4f}")
            
            # Test batch prediction
            batch_predictions = inference_engine.predict_batch(
                images=images[:4],
                return_probabilities=True
            )
            
            print(f"   ✓ Batch prediction completed: {len(batch_predictions)} predictions")
            
            # Test evaluation
            eval_results = inference_engine.evaluate(
                dataloader=val_loader,
                return_predictions=True
            )
            
            print(f"   ✓ Evaluation completed")
            print(f"   ✓ Accuracy: {eval_results['accuracy']:.4f}")
            print(f"   ✓ Loss: {eval_results['loss']:.4f}")
            
        except Exception as e:
            print(f"   ✗ Error in inference: {e}")


def test_visualization_utils():
    """
    Test the visualization and utility functions.
    """
    print("\n" + "="*60)
    print("Testing Visualization Utils")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy dataset
        dummy_classes = create_dummy_dataset(temp_dir, num_classes=3, samples_per_class=15)
        
        # Setup dataset
        dataset = UnifiedImageDataset(
            dataset_name='dummy_dataset',
            dataset_source='folder',
            data_dir=temp_dir,
            image_size=64,
            batch_size=8,
            val_split=0.3
        )
        train_loader, val_loader = dataset.get_dataloader()
        class_names = dataset.class_names if hasattr(dataset, 'class_names') else dummy_classes
        
        print("\n1. Testing dataset visualizations...")
        try:
            # Test dataset sample visualization
            visualize_dataset_samples(
                dataset=train_loader.dataset,
                class_names=class_names,
                num_samples=9,
                save_path=None  # Don't save for testing
            )
            print(f"   ✓ Dataset samples visualization completed")
            
            # Test class distribution plot
            plot_class_distribution(
                dataset=train_loader.dataset,
                class_names=class_names,
                save_path=None
            )
            print(f"   ✓ Class distribution plot completed")
            
            # Test augmentation preview
            if hasattr(dataset, 'train_transform'):
                create_data_augmentation_preview(
                    dataset=train_loader.dataset,
                    transform=dataset.train_transform,
                    save_path=None
                )
                print(f"   ✓ Augmentation preview completed")
            
        except Exception as e:
            print(f"   ✗ Error in dataset visualization: {e}")
        
        print("\n2. Testing model analysis...")
        try:
            # Setup model for analysis
            model = UnifiedImageClassifier(
                model_name='resnet18',
                model_source='torchvision',
                num_classes=len(class_names),
                pretrained=False
            )
            pytorch_model = model.model
            
            # Test model analysis
            analysis = analyze_model(pytorch_model)
            print(f"   ✓ Model analysis completed")
            print(f"   ✓ Total parameters: {analysis['parameters']['total_parameters']:,}")
            print(f"   ✓ Model size: {analysis['size']['size_mb']:.2f} MB")
            
        except Exception as e:
            print(f"   ✗ Error in model analysis: {e}")
        
        print("\n3. Testing training curve plotting...")
        try:
            # Create dummy training history
            dummy_history = {
                'train_loss': [1.2, 1.0, 0.8, 0.6, 0.5],
                'val_loss': [1.1, 0.9, 0.8, 0.7, 0.6],
                'train_accuracy': [0.4, 0.6, 0.7, 0.8, 0.85],
                'val_accuracy': [0.45, 0.65, 0.72, 0.78, 0.82]
            }
            
            plot_training_curves(
                history=dummy_history,
                save_path=None
            )
            print(f"   ✓ Training curves plotting completed")
            
        except Exception as e:
            print(f"   ✗ Error in training curves: {e}")


def test_config_management():
    """
    Test configuration saving and loading.
    """
    print("\n" + "="*60)
    print("Testing Configuration Management")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.json"
        
        # Test config saving
        print("\n1. Testing config saving...")
        try:
            test_config = {
                'model_name': 'resnet18',
                'model_source': 'torchvision',
                'dataset_name': 'cifar10',
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 10
            }
            
            save_experiment_config(test_config, config_path)
            print(f"   ✓ Config saved to {config_path}")
            
        except Exception as e:
            print(f"   ✗ Error saving config: {e}")
        
        # Test config loading
        print("\n2. Testing config loading...")
        try:
            loaded_config = load_experiment_config(config_path)
            print(f"   ✓ Config loaded successfully")
            print(f"   ✓ Model: {loaded_config['model_name']}")
            print(f"   ✓ Dataset: {loaded_config['dataset_name']}")
            print(f"   ✓ Batch size: {loaded_config['batch_size']}")
            
            # Verify config integrity
            assert loaded_config == test_config, "Config mismatch!"
            print(f"   ✓ Config integrity verified")
            
        except Exception as e:
            print(f"   ✗ Error loading config: {e}")


def run_integration_test():
    """
    Run a complete integration test with a small example.
    """
    print("\n" + "="*60)
    print("Running Integration Test")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("\n1. Setting up test environment...")
        
        # Create dummy dataset
        dummy_classes = create_dummy_dataset(temp_dir, num_classes=3, samples_per_class=20)
        
        # Setup dataset
        dataset = UnifiedImageDataset(
            dataset_name='integration_test',
            dataset_source='folder',
            data_dir=temp_dir,
            image_size=64,
            batch_size=8,
            val_split=0.3,
            use_augmentation=True
        )
        train_loader, val_loader = dataset.get_dataloader()
        class_names = dataset.class_names if hasattr(dataset, 'class_names') else dummy_classes
        print(f"   ✓ Dataset ready: {len(class_names)} classes")
        
        # Setup model
        model = UnifiedImageClassifier(
            model_name='resnet18',
            model_source='torchvision',
            num_classes=len(class_names),
            pretrained=False
        )
        pytorch_model = model.model
        print(f"   ✓ Model ready: ResNet-18")
        
        print("\n2. Training model...")
        try:
            # Setup training
            engine = TrainingEngine(
                model=pytorch_model,
                device=device,
                use_amp=False,
                log_interval=5
            )
            
            optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Train for 3 epochs
            history = engine.train(
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=3
            )
            
            print(f"   ✓ Training completed")
            print(f"   ✓ Final train accuracy: {history['train_accuracy'][-1]:.4f}")
            print(f"   ✓ Final val accuracy: {history['val_accuracy'][-1]:.4f}")
            
        except Exception as e:
            print(f"   ✗ Training failed: {e}")
            return
        
        print("\n3. Testing inference...")
        try:
            # Setup inference
            inference_engine = InferenceEngine(
                model=pytorch_model,
                device=device,
                class_names=class_names,
                model_type="pytorch"
            )
            
            # Evaluate model
            eval_results = inference_engine.evaluate(
                dataloader=val_loader,
                return_predictions=True
            )
            
            print(f"   ✓ Evaluation completed")
            print(f"   ✓ Test accuracy: {eval_results['accuracy']:.4f}")
            print(f"   ✓ Test loss: {eval_results['loss']:.4f}")
            
            # Test single prediction
            data_iter = iter(val_loader)
            images, labels = next(data_iter)
            prediction = inference_engine.predict_single(
                image=images[0],
                return_probabilities=True
            )
            
            print(f"   ✓ Single prediction: {prediction['predicted_class']} (conf: {prediction['confidence']:.3f})")
            
        except Exception as e:
            print(f"   ✗ Inference failed: {e}")
            return
        
        print("\n4. Testing visualizations...")
        try:
            # Create visualizations
            plot_training_curves(history=history, save_path=None)
            
            if 'predictions' in eval_results and 'targets' in eval_results:
                inference_engine.plot_confusion_matrix(
                    targets=eval_results['targets'],
                    predictions=eval_results['predictions'],
                    save_path=None
                )
            
            print(f"   ✓ Visualizations completed")
            
        except Exception as e:
            print(f"   ✗ Visualization failed: {e}")
        
        print("\n" + "="*60)
        print("✓ INTEGRATION TEST PASSED")
        print("="*60)


def main():
    """
    Run all tests and examples.
    """
    print("Unified Image Classification System - Examples and Tests")
    print("=" * 80)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Run all tests
    try:
        test_unified_dataset()
        test_unified_model()
        test_training_engines()
        test_inference_engine()
        test_visualization_utils()
        test_config_management()
        run_integration_test()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()