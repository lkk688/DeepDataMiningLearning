"""HuggingFace Trainer Integration for Unified Image Classification

Provides integration with HuggingFace's Trainer API for standardized training.
Supports custom metrics, callbacks, and model evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json

# HuggingFace imports
try:
    from transformers import (
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
        TrainerCallback,
        TrainerState,
        TrainerControl
    )
    from transformers.trainer_utils import EvalPrediction
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[ERROR] transformers not available. Install with: pip install transformers")

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[INFO] sklearn not available for advanced metrics. Install with: pip install scikit-learn")


class ImageClassificationDataset(torch.utils.data.Dataset):
    """Dataset wrapper for HuggingFace Trainer."""
    
    def __init__(self, dataset):
        """
        Initialize dataset wrapper.
        
        Args:
            dataset: PyTorch dataset with __getitem__ returning (image, label)
        """
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {
            'pixel_values': image,
            'labels': label
        }


class ModelWrapper(nn.Module):
    """Model wrapper for HuggingFace Trainer compatibility."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixel_values, labels=None):
        logits = self.model(pixel_values)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }


class MetricsCallback(TrainerCallback):
    """Custom callback for logging additional metrics."""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Called after evaluation."""
        if logs:
            # Store metrics
            self.metrics_history.append({
                'step': state.global_step,
                'epoch': state.epoch,
                **logs
            })
            
            # Save metrics to file
            metrics_file = self.log_dir / "metrics_history.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when logging."""
        if logs and 'eval_loss' in logs:
            print(f"Step {state.global_step}: {logs}")


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Basic accuracy
    accuracy = accuracy_score(labels, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'eval_samples': len(labels)
    }
    
    # Advanced metrics if sklearn is available
    if SKLEARN_AVAILABLE:
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Per-class accuracy
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                class_acc = accuracy_score(labels[mask], predictions[mask])
                metrics[f'accuracy_class_{label}'] = class_acc
    
    return metrics


class HuggingFaceTrainer:
    """HuggingFace Trainer wrapper for image classification."""
    
    def __init__(
        self,
        model,
        output_dir: str = "./hf_outputs",
        experiment_name: str = "image_classification",
        use_wandb: bool = False,
        wandb_project: str = "image-classification-hf"
    ):
        """
        Initialize HuggingFace trainer wrapper.
        
        Args:
            model: PyTorch model for image classification
            output_dir: Directory for outputs and checkpoints
            experiment_name: Name of the experiment
            use_wandb: Whether to use wandb for logging
            wandb_project: Wandb project name
        """
        if not HF_AVAILABLE:
            raise ImportError("transformers is required for HuggingFace trainer")
        
        self.model = ModelWrapper(model)
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name=self.experiment_name,
                dir=str(self.output_dir)
            )
    
    def create_training_args(
        self,
        num_train_epochs: int = 10,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 32,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 1000,
        evaluation_strategy: str = "steps",
        save_strategy: str = "steps",
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "accuracy",
        greater_is_better: bool = True,
        early_stopping_patience: Optional[int] = None,
        fp16: bool = False,
        dataloader_num_workers: int = 4,
        remove_unused_columns: bool = False,
        **kwargs
    ) -> TrainingArguments:
        """
        Create training arguments for HuggingFace Trainer.
        
        Args:
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Training batch size per device
            per_device_eval_batch_size: Evaluation batch size per device
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            logging_steps: Logging frequency
            eval_steps: Evaluation frequency
            save_steps: Checkpoint saving frequency
            evaluation_strategy: When to evaluate ("steps" or "epoch")
            save_strategy: When to save ("steps" or "epoch")
            load_best_model_at_end: Whether to load best model at end
            metric_for_best_model: Metric to use for best model selection
            greater_is_better: Whether higher metric values are better
            early_stopping_patience: Early stopping patience (None to disable)
            fp16: Whether to use mixed precision training
            dataloader_num_workers: Number of dataloader workers
            remove_unused_columns: Whether to remove unused columns
            **kwargs: Additional arguments for TrainingArguments
        
        Returns:
            TrainingArguments object
        """
        # Set up reporting
        report_to = []
        if self.use_wandb:
            report_to.append("wandb")
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            eval_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            fp16=fp16,
            dataloader_num_workers=dataloader_num_workers,
            remove_unused_columns=remove_unused_columns,
            report_to=report_to,
            run_name=self.experiment_name,
            **kwargs
        )
        
        return training_args
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        training_args=None,
        **trainer_kwargs
    ):
        """
        Train the model using HuggingFace Trainer.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            training_args: TrainingArguments (will create default if None)
            **trainer_kwargs: Additional arguments for Trainer
        
        Returns:
            Trainer object after training
        """
        # Wrap datasets
        train_dataset_wrapped = ImageClassificationDataset(train_dataset)
        eval_dataset_wrapped = None
        if eval_dataset is not None:
            eval_dataset_wrapped = ImageClassificationDataset(eval_dataset)
        
        # Create default training arguments if not provided
        if training_args is None:
            training_args = self.create_training_args()
        
        # Set up callbacks
        callbacks = [MetricsCallback(log_dir=self.output_dir / "metrics")]
        
        # Add early stopping if specified
        if hasattr(training_args, 'early_stopping_patience') and training_args.early_stopping_patience:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience
            ))
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset_wrapped,
            eval_dataset=eval_dataset_wrapped,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **trainer_kwargs
        )
        
        print(f"Starting HuggingFace training...")
        print(f"Output directory: {self.output_dir}")
        print(f"Training samples: {len(train_dataset_wrapped)}")
        if eval_dataset_wrapped:
            print(f"Evaluation samples: {len(eval_dataset_wrapped)}")
        
        # Train the model
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        
        print("Training completed!")
        return trainer
    
    def evaluate(
        self,
        eval_dataset,
        trainer=None,
        training_args=None
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_dataset: Evaluation dataset
            trainer: Existing trainer (will create new if None)
            training_args: TrainingArguments (for new trainer)
        
        Returns:
            Evaluation metrics
        """
        if trainer is None:
            # Create trainer for evaluation
            eval_dataset_wrapped = ImageClassificationDataset(eval_dataset)
            
            if training_args is None:
                training_args = self.create_training_args()
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                eval_dataset=eval_dataset_wrapped,
                compute_metrics=compute_metrics
            )
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        print("Evaluation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        return eval_results
    
    def predict(
        self,
        test_dataset,
        trainer=None,
        training_args=None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Make predictions on test dataset.
        
        Args:
            test_dataset: Test dataset
            trainer: Existing trainer (will create new if None)
            training_args: TrainingArguments (for new trainer)
        
        Returns:
            Tuple of (predictions, labels, metrics)
        """
        if trainer is None:
            # Create trainer for prediction
            test_dataset_wrapped = ImageClassificationDataset(test_dataset)
            
            if training_args is None:
                training_args = self.create_training_args()
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                compute_metrics=compute_metrics
            )
        
        # Make predictions
        predictions = trainer.predict(ImageClassificationDataset(test_dataset))
        
        # Extract results
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        metrics = predictions.metrics
        
        print("Prediction Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return pred_labels, true_labels, metrics
    
    def save_model(self, save_path: str, trainer=None):
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model
            trainer: Trainer object (will use self.model if None)
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if trainer is not None:
            trainer.save_model(str(save_path))
        else:
            # Save the underlying PyTorch model
            torch.save({
                'model_state_dict': self.model.model.state_dict(),
                'model_config': getattr(self.model.model, 'config', None)
            }, save_path / "pytorch_model.bin")
        
        print(f"Model saved to: {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load a trained model.
        
        Args:
            load_path: Path to load the model from
        """
        load_path = Path(load_path)
        
        # Try to load HuggingFace format first
        model_file = load_path / "pytorch_model.bin"
        if model_file.exists():
            checkpoint = torch.load(model_file, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Model loaded from: {load_path}")
        else:
            raise FileNotFoundError(f"No model found at {load_path}")


def test_hf_trainer():
    """Test function for HuggingFace trainer."""
    if not HF_AVAILABLE:
        print("HuggingFace transformers not available, skipping test")
        return
    
    print("Testing HuggingFace Trainer...")
    
    # Create dummy model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 10)
    )
    
    # Create dummy dataset
    dummy_data = torch.randn(100, 3, 32, 32)
    dummy_targets = torch.randint(0, 10, (100,))
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
    
    # Split dataset
    train_size = int(0.8 * len(dummy_dataset))
    val_size = len(dummy_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dummy_dataset, [train_size, val_size]
    )
    
    # Initialize HF trainer
    hf_trainer = HuggingFaceTrainer(
        model=model,
        experiment_name="test_hf_experiment"
    )
    
    # Create training arguments
    training_args = hf_trainer.create_training_args(
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10
    )
    
    # Train
    trainer = hf_trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_args=training_args
    )
    
    # Evaluate
    eval_results = hf_trainer.evaluate(val_dataset, trainer)
    
    # Make predictions
    pred_labels, true_labels, metrics = hf_trainer.predict(val_dataset, trainer)
    
    print(f"Predictions shape: {pred_labels.shape}")
    print(f"True labels shape: {true_labels.shape}")
    print(f"Test metrics: {metrics}")
    
    print("HuggingFace Trainer testing completed!")


if __name__ == "__main__":
    test_hf_trainer()