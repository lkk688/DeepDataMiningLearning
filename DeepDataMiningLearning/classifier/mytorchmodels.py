#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Image Classification System

A comprehensive image classification framework supporting:
- torchvision, timm, and HuggingFace models
- Multiple dataset formats and sources
- Custom training loops and HuggingFace Trainer integration
- Advanced inference and evaluation capabilities
- Comprehensive visualization and analysis tools
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import get_model, get_model_weights, list_models
import argparse
import os
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[INFO] timm not available. Install with: pip install timm")

# Import our modular components
from unified_dataset import UnifiedImageDataset
from unified_model import UnifiedImageClassifier
from training_engine import TrainingEngine
from hf_trainer import HuggingFaceTrainer
from inference_engine import InferenceEngine
from utils import (
    set_seed, analyze_model, visualize_dataset_samples,
    plot_class_distribution, create_data_augmentation_preview,
    plot_training_curves, save_experiment_config
)

# Utility classes and functions
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

def get_labelfn(model):
    """Get label function for model."""
    return lambda x: f'class_{x}'


def create_torchclassifiermodel(
    model_name,
    numclasses=None,
    model_type="torchvision",
    torchhublink=None,
    freezeparameters=True,
    pretrained=True,
    dropoutp=0.2,
):
    pretrained_model = None
    preprocess = None
    imagenet_classes = None
    # different model_type: 'torchvision', 'torchhub'
    if model_type == "torchvision":
        model_names = list_models(module=torchvision.models)
        if model_name in model_names:
            # Step 1: Initialize model with the best available weights
            weights_enum = get_model_weights(model_name)
            weights = weights_enum.IMAGENET1K_V1
            # print([weight for weight in weights_enum])
            # weights = get_weight("ResNet50_Weights.IMAGENET1K_V2")#ResNet50_Weights.DEFAULT
            if pretrained == True:
                pretrained_model = get_model(
                    model_name, weights=weights
                )  # weights="DEFAULT"
                # pretrained_model=get_model(model_name, weights="DEFAULT")
            else:
                pretrained_model = get_model(model_name, weights=None)
            # print(pretrained_model)
            # Freeze the base parameters
            if freezeparameters == True:
                print("Freeze parameters")
                for parameter in pretrained_model.parameters():
                    parameter.requires_grad = False
            # Step 2: Initialize the inference transforms
            preprocess = weights.transforms()  # preprocess.crop_size
            imagenet_classes = weights.meta["categories"]
            # Step 3: Apply inference preprocessing transforms
            # batch = preprocess(img).unsqueeze(0)
            if numclasses is not None and len(imagenet_classes) != numclasses:
                pretrained_model = modify_classifier(
                    pretrained_model=pretrained_model,
                    numclasses=numclasses,
                    dropoutp=dropoutp,
                )
            else:
                numclasses = len(imagenet_classes)
            return pretrained_model
        else:
            print("Model name not exist.")
    elif model_type == "torchhub" and torchhublink is not None:
        #'deit_base_patch16_224'
        # pretrained_model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)
        pretrained_model = torch.hub.load(
            torchhublink, model_name, pretrained=pretrained
        )
    elif model_type == "timm":
        if not TIMM_AVAILABLE:
            raise ImportError("timm not available. Install with: pip install timm")
        # if model_name in model_names:
        pretrained_model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=numclasses
        )
        data_cfg = timm.data.resolve_data_config(pretrained_model.pretrained_cfg)
        preprocess = timm.data.create_transform(**data_cfg)

    totalparameters= sum([m.numel() for m in pretrained_model.parameters()])
    print(
        f"Model {model_name} created, param count: {totalparameters/10e6}M"
    )
    num_classes = getattr(pretrained_model, "num_classes", None)
    return pretrained_model, preprocess, num_classes, imagenet_classes


def modify_classifier(pretrained_model, numclasses, dropoutp=0.3, classifiername=None):
    # display model architecture
    lastmoduleinlist = list(pretrained_model.named_children())[-1]
    # print("lastmoduleinlist len:",len(lastmoduleinlist))
    lastmodulename = lastmoduleinlist[0]
    print("lastmodulename:", lastmodulename)
    lastlayer = lastmoduleinlist[-1]
    newclassifier = lastlayer #default
    if isinstance(lastlayer, nn.Linear):
        print("Linear layer")
        newclassifier = nn.Linear(
            in_features=lastlayer.in_features, out_features=numclasses
        )
    elif isinstance(lastlayer, nn.Sequential):
        print("Sequential layer")
        lastlayerlist = list(lastlayer)  # [-1] #last layer
        # print("lastlayerlist type:",type(lastlayerlist))
        if isinstance(lastlayerlist, list):
            # print("your object is a list !")
            lastlayer = lastlayerlist[-1]
            newclassifier = torch.nn.Sequential(
                torch.nn.Dropout(p=dropoutp, inplace=True),
                torch.nn.Linear(
                    in_features=lastlayer.in_features,
                    out_features=numclasses,  # same number of output units as our number of classes
                    bias=True,
                ),
            )
        else:
            print("Error: Sequential layer is not list:", lastlayer)
            # newclassifier = nn.Linear(in_features=lastlayer.in_features, out_features=classnum)
    if lastmodulename == "heads":
        pretrained_model.heads = newclassifier  # .to(device)
    elif lastmodulename == "classifier":
        pretrained_model.classifier = newclassifier  # .to(device)
    elif lastmodulename == "fc":
        pretrained_model.fc = newclassifier  # .to(device)
    elif classifiername is not None:
        lastlayer = newclassifier  # not tested!!
    else:
        print("Please check the last module name of the model.")

    return pretrained_model


def get_optimizer(model, opt="lamb", lr=0.01, weight_decay=0.01, momentum=0):
    # optimizer = timm.optim.AdamP(model.parameters(), lr=0.01)
    # opt: name of optimizer to create: sgd, adam, adamw, adamp, adabelief, lamp
    # lr: initial learning rate
    # weight_decay: weight decay to apply in optimizer
    # momentum:  momentum for momentum based optimizers
    if not TIMM_AVAILABLE:
        raise ImportError("timm not available. Install with: pip install timm")
    optimizer = timm.optim.create_optimizer_v2(
        model, opt=opt, lr=lr, weight_decay=weight_decay, momentum=momentum
    )
    return optimizer


def get_scheduler(optimizer, num_epochs, num_repeat=2, warmup_t=3):
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
    #                                                                 T_0=num_epoch_repeat*num_steps_per_epoch,
    #                                                                 T_mult=1,
    #                                                                 eta_min=1e-6,
    #                                                                 last_epoch=-1)
    if not TIMM_AVAILABLE:
        raise ImportError("timm not available. Install with: pip install timm")
    num_epoch_repeat = num_epochs // num_repeat
    scheduler = timm.scheduler.CosineLRScheduler(
        optimizer,
        t_initial=num_epoch_repeat,
        lr_min=1e-5,
        warmup_lr_init=0.01,
        warmup_t=warmup_t,
        cycle_limit=num_epoch_repeat + 1,
    )
    return scheduler


def train_step(
    model, dataloader, loss_fn, optimizer, device, epoch, scheduler, ema_model
):
    """Trains a PyTorch model for a single epoch.
    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    num_steps_per_epoch = len(dataloader)
    num_updates = epoch * num_steps_per_epoch

    lrs = []

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # 6. Update EMA model parameters
        if ema_model is not None:
            ema_model.update(model)

        # new added for timm
        if scheduler is not None:
            num_updates += 1
            scheduler.step_update(num_updates=num_updates)
            lrs.append(optimizer.param_groups[0]["lr"])

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc, lrs, ema_model


def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    loss_fn,
    epochs,
    device,
    scheduler,
    use_ema=False,
):
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary

    # Loop through training and testing steps for a number of epochs

    # Make sure model on target device
    model.to(device)

    if use_ema:
        if not TIMM_AVAILABLE:
            raise ImportError("timm not available. Install with: pip install timm")
        ema_model = timm.utils.ModelEmaV2(model, decay=0.9)
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "ematest_loss": [],
            "ematest_acc": [],
            "all_lrs": [],
        }
    else:
        ema_model = None
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "all_lrs": [],
        }

    all_lrs = []

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, lrs, ema_model = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scheduler=scheduler,
            ema_model=ema_model,
        )

        if scheduler is not None:
            all_lrs.extend(lrs)
            scheduler.step(epoch + 1)

        test_loss, test_acc, _, _, _ = inference(
            model=model, dataloader=test_dataloader, device=device, loss_fn=loss_fn, use_probs=False, top_k=None, to_label=None
        )
        if ema_model is not None:
            ematest_loss, ematest_acc, _, _, _ = inference(
                model=ema_model,
                dataloader=test_dataloader,
                device=device,
                loss_fn=loss_fn,
                use_probs=False, top_k=None, to_label=None
            )

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        if ema_model is not None:
            print( f"test_loss: {ematest_loss:.4f}, test_acc: {ematest_acc:.4f}"
            )
            

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        if ema_model is not None:
            results["ematest_loss"].append(ematest_loss)
            results["ematest_acc"].append(ematest_acc)
        results["all_lrs"].append(all_lrs)

    # Return the filled results at the end of the epochs
    return results, ema_model


def inference(
    model, dataloader, device, loss_fn=None, use_probs=True, top_k=5, to_label=None
):
    # Put model in eval mode
    model.eval()
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    num_classes = getattr(model, "num_classes", 1000)

    if top_k is not None:
        top_k = min(top_k, num_classes)
    all_indices = []
    all_labels = []
    all_outputs = []
    to_label = get_labelfn(model)
    batch_time = AverageMeter()
    end = time.time()
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            if y is not None:
                # Send data to target device
                X, y = X.to(device), y.to(device)
            else:
                X = X.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            if loss_fn is not None:
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            if y is not None:
                test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

            if use_probs:
                output = test_pred_logits.softmax(-1)
            else:
                output = test_pred_logits

            if top_k:
                output, indices = output.topk(top_k)
                np_indices = indices.cpu().numpy()
                all_indices.append(np_indices)
                if to_label is not None:
                    np_labels = to_label(np_indices)
                    all_labels.append(np_labels)

            all_outputs.append(output.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    all_indices = np.concatenate(all_indices, axis=0) if all_indices else None
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    all_outputs = np.concatenate(all_outputs, axis=0).astype(np.float32)
    return test_loss, test_acc, all_indices, all_labels, all_outputs

def setup_datasets(args):
    """Setup datasets based on arguments."""
    dataset = UnifiedImageDataset(
        data_source=args.dataset_source,
        dataset_name=args.dataset,
        data_path=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        augment=not args.no_augment,
        num_workers=args.num_workers
    )
    train_loader, val_loader = dataset.get_dataloader()
    class_names = dataset.get_class_names()
    return train_loader, val_loader, class_names, dataset

def setup_model(args, num_classes):
    """Setup model based on arguments."""
    unified_model = UnifiedImageClassifier(
        model_source=args.model_source,
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained
    )
    model = unified_model.model
    return model, unified_model

def visualize_dataset(args, dataset, class_names, save_dir):
    """Create dataset visualizations."""
    print("Creating dataset visualizations...")
    # Add visualization code here
    pass

def train_with_custom_loop(args, model, train_loader, val_loader, class_names, save_dir):
    """Train model using custom training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    engine = TrainingEngine(
        model=model,
        device=device,
        use_amp=args.use_amp,
        use_ema=args.use_ema,
        gradient_clip_val=args.gradient_clip,
        save_dir=str(save_dir),
        experiment_name="image_classification"
    )
    
    # Create optimizer and criterion
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    history = engine.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=args.epochs
    )
    return engine, history

def train_with_hf_trainer(args, model, train_loader, val_loader, class_names, save_dir):
    """Train model using HuggingFace Trainer."""
    hf_trainer = HuggingFaceTrainer(
        model=model,
        output_dir=str(save_dir)
    )
    trainer = hf_trainer.train(
         train_dataset=train_loader.dataset,
         eval_dataset=val_loader.dataset,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size
    )
    return hf_trainer, trainer

def evaluate_model(args, model, val_loader, class_names, save_dir):
    """Evaluate model performance."""
    inference_engine = InferenceEngine(
        model=model,
        class_names=class_names
    )
    results = inference_engine.evaluate(val_loader)
    return results, inference_engine

def create_experiment_config(args):
    """Create experiment configuration."""
    config = {
        'model': args.model,
        'model_source': args.model_source,
        'dataset': args.dataset,
        'dataset_source': args.dataset_source,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'image_size': args.image_size,
        'pretrained': args.pretrained,
        'trainer': args.trainer
    }
    return config

def main():
    parser = argparse.ArgumentParser(description='Unified Image Classification System')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model name (e.g., resnet18, vit-base-patch16-224, microsoft/resnet-50)')
    parser.add_argument('--model_source', type=str, default='torchvision',
                        choices=['torchvision', 'timm', 'huggingface'],
                        help='Model source')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset name or path')
    parser.add_argument('--dataset_source', type=str, default='torchvision',
                        choices=['torchvision', 'huggingface', 'folder'],
                        help='Dataset source')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (for folder datasets)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    
    # Training arguments
    parser.add_argument('--trainer', type=str, default='custom',
                        choices=['custom', 'huggingface'],
                        help='Training method')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--early_stopping', type=int, default=None,
                        help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use exponential moving average')
    parser.add_argument('--gradient_clip', type=float, default=None,
                        help='Gradient clipping value')
    
    # Logging and saving arguments
    parser.add_argument('--experiment_name', type=str, default='image_classification',
                        help='Experiment name')
    parser.add_argument('--save_dir', type=str, default='./experiments',
                        help='Directory to save results')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='unified-image-classification',
                        help='Wandb project name')
    
    # Execution arguments
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only evaluate (requires saved model)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load saved model')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Only create visualizations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    config = create_experiment_config(args)
    save_experiment_config(config, save_dir / "config.json")
    
    print(f"Unified Image Classification System")
    print(f"Experiment: {args.experiment_name}")
    print(f"Save directory: {save_dir}")
    print("-" * 60)
    
    # Setup datasets
    train_loader, val_loader, class_names, dataset = setup_datasets(args)
    
    # Create dataset visualizations if requested
    if args.visualize_only:
        visualize_dataset(args, dataset, class_names, save_dir)
        print("Dataset visualizations completed!")
        return
    
    # Setup model
    model, unified_model = setup_model(args, len(class_names))
    
    # Load model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    
    if args.evaluate_only:
        # Only evaluate
        eval_results, inference_engine = evaluate_model(
            args, model, val_loader, class_names, save_dir
        )
        print("Evaluation completed!")
        return
    
    # Create dataset visualizations
    visualize_dataset(args, dataset, class_names, save_dir)
    
    # Training
    if args.trainer == 'custom':
        # Custom training loop
        engine, history = train_with_custom_loop(
            args, model, train_loader, val_loader, class_names, save_dir
        )
        
        # Plot training curves
        plot_training_curves(
            history=history,
            save_path=str(save_dir / "training_curves.png")
        )
        
    elif args.trainer == 'huggingface':
        # HuggingFace Trainer
        hf_trainer, trainer = train_with_hf_trainer(
            args, model, train_loader, val_loader, 
            class_names, save_dir
        )
    
    # Evaluation
    eval_results, inference_engine = evaluate_model(
        args, model, val_loader, class_names, save_dir
    )
    
    # Save final model
    final_model_path = save_dir / "final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'class_names': class_names,
        'eval_results': eval_results
    }, final_model_path)
    
    print(f"\nTraining and evaluation completed!")
    print(f"Results saved to: {save_dir}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Final validation accuracy: {eval_results['accuracy']:.4f}")


if __name__ == '__main__':
    main()

    