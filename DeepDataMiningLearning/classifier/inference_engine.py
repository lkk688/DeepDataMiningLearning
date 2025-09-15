"""Inference Engine for Unified Image Classification

Provides unified inference interface for torchvision, timm, and HuggingFace models.
Supports batch inference, visualization, and performance analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

# Optional imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[INFO] opencv-python not available. Install with: pip install opencv-python")

try:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, 
        confusion_matrix, classification_report
    )
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[INFO] sklearn/seaborn not available for advanced metrics")

try:
    import grad_cam
    from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("[INFO] pytorch-grad-cam not available. Install with: pip install grad-cam")


class InferenceEngine:
    """Unified inference engine for image classification models."""
    
    def __init__(
        self,
        model,
        device: torch.device = None,
        class_names: Optional[List[str]] = None,
        model_type: str = "pytorch",
        use_amp: bool = True
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Trained model (PyTorch, timm, or HuggingFace)
            device: Device to run inference on
            class_names: List of class names for predictions
            model_type: Type of model ("pytorch", "timm", "huggingface")
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.model_type = model_type.lower()
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize performance tracking
        self.inference_times = []
        self.batch_sizes = []
        
        print(f"Inference engine initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model type: {self.model_type}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Classes: {len(class_names) if class_names else 'Unknown'}")
    
    def predict_single(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        transform=None,
        return_probabilities: bool = False,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Predict on a single image.
        
        Args:
            image: Input image (tensor, numpy array, or PIL Image)
            transform: Transform to apply to image
            return_probabilities: Whether to return class probabilities
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        if isinstance(image, (np.ndarray, Image.Image)):
            if transform is None:
                raise ValueError("Transform is required for numpy/PIL images")
            image = transform(image)
        
        # Ensure tensor is on correct device and has batch dimension
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            image = image.to(self.device)
        
        # Inference
        start_time = time.time()
        
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self._forward_pass(image)
            else:
                outputs = self._forward_pass(image)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.batch_sizes.append(1)
        
        # Process outputs
        if isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Get probabilities
        probabilities = F.softmax(logits, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()
        
        # Prepare results
        results = {
            'predicted_class': int(top_indices[0]),
            'confidence': float(top_probs[0]),
            'inference_time': inference_time,
            'top_k_predictions': []
        }
        
        # Add class names if available
        if self.class_names:
            results['predicted_class_name'] = self.class_names[top_indices[0]]
        
        # Add top-k results
        for i in range(len(top_indices)):
            pred_info = {
                'class_id': int(top_indices[i]),
                'confidence': float(top_probs[i])
            }
            if self.class_names:
                pred_info['class_name'] = self.class_names[top_indices[i]]
            results['top_k_predictions'].append(pred_info)
        
        # Add full probabilities if requested
        if return_probabilities:
            results['probabilities'] = probabilities.squeeze().cpu().numpy().tolist()
        
        return results
    
    def predict_batch(
        self,
        images: Union[torch.Tensor, List],
        batch_size: int = 32,
        return_probabilities: bool = False,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict on a batch of images.
        
        Args:
            images: Batch of images (tensor or list of tensors)
            batch_size: Batch size for processing
            return_probabilities: Whether to return class probabilities
            show_progress: Whether to show progress bar
        
        Returns:
            List of prediction results
        """
        if isinstance(images, list):
            # Convert list to tensor
            images = torch.stack(images)
        
        images = images.to(self.device)
        num_images = images.size(0)
        
        results = []
        
        # Process in batches
        if show_progress:
            from tqdm.auto import tqdm
            pbar = tqdm(range(0, num_images, batch_size), desc="Inference")
        else:
            pbar = range(0, num_images, batch_size)
        
        for i in pbar:
            end_idx = min(i + batch_size, num_images)
            batch_images = images[i:end_idx]
            
            start_time = time.time()
            
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_pass(batch_images)
                else:
                    outputs = self._forward_pass(batch_images)
            
            inference_time = time.time() - start_time
            current_batch_size = batch_images.size(0)
            
            self.inference_times.append(inference_time)
            self.batch_sizes.append(current_batch_size)
            
            # Process outputs
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                logits = outputs
            
            # Get probabilities
            probabilities = F.softmax(logits, dim=1)
            
            # Get predictions for each image in batch
            for j in range(current_batch_size):
                img_probs = probabilities[j]
                top_prob, top_idx = torch.max(img_probs, dim=0)
                
                result = {
                    'predicted_class': int(top_idx.item()),
                    'confidence': float(top_prob.item()),
                    'inference_time': inference_time / current_batch_size  # Average per image
                }
                
                if self.class_names:
                    result['predicted_class_name'] = self.class_names[top_idx.item()]
                
                if return_probabilities:
                    result['probabilities'] = img_probs.cpu().numpy().tolist()
                
                results.append(result)
        
        return results
    
    def evaluate(
        self,
        dataloader,
        return_predictions: bool = False,
        save_results: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            return_predictions: Whether to return all predictions
            save_results: Path to save evaluation results
        
        Returns:
            Evaluation metrics and optionally predictions
        """
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        total_time = 0
        total_samples = 0
        
        print("Starting evaluation...")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                batch_size = images.size(0)
                start_time = time.time()
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_pass(images)
                else:
                    outputs = self._forward_pass(images)
                
                batch_time = time.time() - start_time
                total_time += batch_time
                total_samples += batch_size
                
                # Process outputs
                if isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Get predictions and confidences
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                
                if (batch_idx + 1) % 100 == 0:
                    print(f"Processed {batch_idx + 1} batches...")
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_confidences = np.array(all_confidences)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        avg_confidence = np.mean(all_confidences)
        avg_inference_time = total_time / total_samples
        
        results = {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'average_inference_time': avg_inference_time,
            'total_samples': total_samples,
            'total_time': total_time
        }
        
        # Advanced metrics if sklearn is available
        if SKLEARN_AVAILABLE:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average='weighted', zero_division=0
            )
            
            results.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            # Per-class metrics
            if self.class_names:
                class_report = classification_report(
                    all_targets, all_predictions,
                    target_names=self.class_names,
                    output_dict=True,
                    zero_division=0
                )
                results['per_class_metrics'] = class_report
        
        # Add predictions if requested
        if return_predictions:
            results['predictions'] = all_predictions.tolist()
            results['targets'] = all_targets.tolist()
            results['confidences'] = all_confidences.tolist()
        
        # Save results if requested
        if save_results:
            self.save_evaluation_results(results, save_results)
        
        # Print summary
        print(f"\nEvaluation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Average Confidence: {avg_confidence:.4f}")
        print(f"  Average Inference Time: {avg_inference_time*1000:.2f} ms")
        print(f"  Total Samples: {total_samples}")
        
        if SKLEARN_AVAILABLE:
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
        
        return results
    
    def _forward_pass(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        if self.model_type == "huggingface":
            # HuggingFace models expect pixel_values
            outputs = self.model(pixel_values=images)
            return outputs.logits if hasattr(outputs, 'logits') else outputs
        else:
            # Standard PyTorch/timm models
            return self.model(images)
    
    def visualize_predictions(
        self,
        images: torch.Tensor,
        predictions: List[Dict],
        num_images: int = 8,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ):
        """
        Visualize predictions on images.
        
        Args:
            images: Batch of images
            predictions: List of prediction results
            num_images: Number of images to visualize
            figsize: Figure size
            save_path: Path to save visualization
        """
        num_images = min(num_images, len(images), len(predictions))
        
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()
        
        for i in range(num_images):
            # Convert tensor to numpy for visualization
            img = images[i].cpu().numpy()
            if img.shape[0] == 3:  # CHW to HWC
                img = np.transpose(img, (1, 2, 0))
            
            # Normalize for display
            img = (img - img.min()) / (img.max() - img.min())
            
            # Plot image
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Add prediction info
            pred = predictions[i]
            title = f"Class: {pred['predicted_class']}\n"
            if 'predicted_class_name' in pred:
                title = f"Class: {pred['predicted_class_name']}\n"
            title += f"Conf: {pred['confidence']:.3f}"
            
            axes[i].set_title(title, fontsize=10)
        
        # Hide unused subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        normalize: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            targets: True labels
            predictions: Predicted labels
            normalize: Whether to normalize the matrix
            figsize: Figure size
            save_path: Path to save plot
        """
        if not SKLEARN_AVAILABLE:
            print("sklearn not available for confusion matrix")
            return
        
        # Compute confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names if self.class_names else range(cm.shape[1]),
            yticklabels=self.class_names if self.class_names else range(cm.shape[0])
        )
        
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: {save_path}")
        
        plt.show()
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get inference performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {"message": "No inference data available"}
        
        times = np.array(self.inference_times)
        batch_sizes = np.array(self.batch_sizes)
        
        # Calculate per-sample times
        per_sample_times = times / batch_sizes
        
        stats = {
            'total_inferences': len(self.inference_times),
            'total_samples': int(np.sum(batch_sizes)),
            'avg_batch_time': float(np.mean(times)),
            'avg_per_sample_time': float(np.mean(per_sample_times)),
            'min_per_sample_time': float(np.min(per_sample_times)),
            'max_per_sample_time': float(np.max(per_sample_times)),
            'std_per_sample_time': float(np.std(per_sample_times)),
            'throughput_samples_per_sec': float(np.sum(batch_sizes) / np.sum(times))
        }
        
        return stats
    
    def save_evaluation_results(self, results: Dict, save_path: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save results
        """
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Evaluation results saved: {save_path}")
    
    def reset_performance_tracking(self):
        """Reset performance tracking statistics."""
        self.inference_times = []
        self.batch_sizes = []
        print("Performance tracking reset")


def test_inference_engine():
    """Test function for InferenceEngine."""
    print("Testing InferenceEngine...")
    
    # Create dummy model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 10)
    )
    
    # Create dummy class names
    class_names = [f"class_{i}" for i in range(10)]
    
    # Initialize inference engine
    engine = InferenceEngine(
        model=model,
        class_names=class_names,
        model_type="pytorch"
    )
    
    # Test single prediction
    dummy_image = torch.randn(3, 32, 32)
    result = engine.predict_single(dummy_image, return_probabilities=True)
    print(f"Single prediction result: {result}")
    
    # Test batch prediction
    dummy_batch = torch.randn(20, 3, 32, 32)
    batch_results = engine.predict_batch(dummy_batch, batch_size=8)
    print(f"Batch prediction results: {len(batch_results)} predictions")
    
    # Test evaluation
    dummy_dataset = torch.utils.data.TensorDataset(
        torch.randn(50, 3, 32, 32),
        torch.randint(0, 10, (50,))
    )
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=16)
    
    eval_results = engine.evaluate(dummy_loader)
    print(f"Evaluation results: {eval_results}")
    
    # Get performance stats
    perf_stats = engine.get_performance_stats()
    print(f"Performance stats: {perf_stats}")
    
    print("InferenceEngine testing completed!")


if __name__ == "__main__":
    test_inference_engine()