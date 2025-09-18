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

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("[INFO] TensorRT not available. Install with: pip install tensorrt pycuda")

# ONNX Runtime imports
try:
    import onnxruntime as ort
    import onnx
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("[INFO] ONNX Runtime not available. Install with: pip install onnxruntime-gpu onnx")


class InferenceEngine:
    """Unified inference engine for image classification models."""
    
    def __init__(
        self,
        model,
        device: torch.device = None,
        class_names: Optional[List[str]] = None,
        model_type: str = "pytorch",
        use_amp: bool = True,
        backend: str = "pytorch",
        tensorrt_engine_path: Optional[str] = None,
        onnx_model_path: Optional[str] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Trained model (PyTorch, timm, or HuggingFace)
            device: Device to run inference on
            class_names: List of class names for predictions
            model_type: Type of model ("pytorch", "timm", "huggingface")
            use_amp: Whether to use automatic mixed precision
            backend: Inference backend ("pytorch", "tensorrt", "onnxruntime")
            tensorrt_engine_path: Path to TensorRT engine file
            onnx_model_path: Path to ONNX model file
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.model_type = model_type.lower()
        self.use_amp = use_amp and torch.cuda.is_available()
        self.backend = backend.lower()
        
        # Initialize backend-specific components
        self.tensorrt_engine = None
        self.tensorrt_context = None
        self.onnx_session = None
        
        if self.backend == "tensorrt" and TENSORRT_AVAILABLE:
            self._initialize_tensorrt(tensorrt_engine_path)
        elif self.backend == "onnxruntime" and ONNXRUNTIME_AVAILABLE:
            self._initialize_onnxruntime(onnx_model_path)
        else:
            # Default PyTorch backend
            self.backend = "pytorch"
            self.model.to(self.device)
            self.model.eval()
        
        # Initialize performance tracking
        self.inference_times = []
        self.batch_sizes = []
        
        print(f"Inference engine initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model type: {self.model_type}")
        print(f"  Backend: {self.backend}")
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
        """Forward pass through the model using the specified backend."""
        if self.backend == "tensorrt":
            return self._tensorrt_inference(images)
        elif self.backend == "onnxruntime":
            return self._onnxruntime_inference(images)
        else:
            # Default PyTorch backend
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
    
    def _initialize_tensorrt(self, engine_path: str):
        """Initialize TensorRT engine for inference."""
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install with: pip install tensorrt pycuda")
        
        if not engine_path or not Path(engine_path).exists():
            raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")
        
        # Load TensorRT engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        # Create TensorRT runtime and engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        self.tensorrt_engine = runtime.deserialize_cuda_engine(engine_data)
        self.tensorrt_context = self.tensorrt_engine.create_execution_context()
        
        # Allocate GPU memory for inputs and outputs
        self.tensorrt_inputs = []
        self.tensorrt_outputs = []
        self.tensorrt_bindings = []
        
        for i in range(self.tensorrt_engine.num_bindings):
            binding_name = self.tensorrt_engine.get_binding_name(i)
            size = trt.volume(self.tensorrt_engine.get_binding_shape(i))
            dtype = trt.nptype(self.tensorrt_engine.get_binding_dtype(i))
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.tensorrt_bindings.append(int(device_mem))
            
            if self.tensorrt_engine.binding_is_input(i):
                self.tensorrt_inputs.append({'host': host_mem, 'device': device_mem, 'shape': self.tensorrt_engine.get_binding_shape(i)})
            else:
                self.tensorrt_outputs.append({'host': host_mem, 'device': device_mem, 'shape': self.tensorrt_engine.get_binding_shape(i)})
        
        print(f"TensorRT engine initialized: {engine_path}")
    
    def _initialize_onnxruntime(self, model_path: str):
        """Initialize ONNX Runtime session for inference."""
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime-gpu onnx")
        
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")
        
        # Try different provider configurations with fallbacks
        provider_configs = []
        
        # First try: GPU providers if CUDA is available
        if torch.cuda.is_available():
            provider_configs.append([
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_max_workspace_size': 2147483648,  # 2GB
                    'trt_fp16_enable': True,
                }),
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ])
            
            # Second try: CUDA only (without TensorRT)
            provider_configs.append([
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                }),
                'CPUExecutionProvider'
            ])
        
        # Final fallback: CPU only
        provider_configs.append(['CPUExecutionProvider'])
        
        # Try each provider configuration until one works
        last_error = None
        for i, providers in enumerate(provider_configs):
            try:
                print(f"Attempting ONNX Runtime initialization with providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
                self.onnx_session = ort.InferenceSession(model_path, providers=providers)
                
                # Get input/output info
                self.onnx_input_name = self.onnx_session.get_inputs()[0].name
                self.onnx_output_name = self.onnx_session.get_outputs()[0].name
                
                print(f"ONNX Runtime session initialized successfully: {model_path}")
                print(f"Active providers: {self.onnx_session.get_providers()}")
                return
                
            except Exception as e:
                last_error = e
                print(f"Provider configuration {i+1} failed: {str(e)}")
                if i < len(provider_configs) - 1:
                    print("Trying next provider configuration...")
                continue
        
        # If all configurations failed, raise the last error
        raise RuntimeError(f"Failed to initialize ONNX Runtime with any provider configuration. Last error: {last_error}")
    
    def _tensorrt_inference(self, images: torch.Tensor) -> torch.Tensor:
        """Run inference using TensorRT engine."""
        if self.tensorrt_engine is None:
            raise RuntimeError("TensorRT engine not initialized")
        
        # Convert PyTorch tensor to numpy
        input_data = images.cpu().numpy().astype(np.float32)
        
        # Copy input data to GPU
        np.copyto(self.tensorrt_inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod(self.tensorrt_inputs[0]['device'], self.tensorrt_inputs[0]['host'])
        
        # Run inference
        self.tensorrt_context.execute_v2(bindings=self.tensorrt_bindings)
        
        # Copy output data from GPU
        cuda.memcpy_dtoh(self.tensorrt_outputs[0]['host'], self.tensorrt_outputs[0]['device'])
        
        # Convert output to PyTorch tensor
        output_shape = self.tensorrt_outputs[0]['shape']
        output_data = self.tensorrt_outputs[0]['host'].reshape(output_shape)
        
        return torch.from_numpy(output_data).to(self.device)
    
    def _onnxruntime_inference(self, images: torch.Tensor) -> torch.Tensor:
        """Run inference using ONNX Runtime."""
        if self.onnx_session is None:
            raise RuntimeError("ONNX Runtime session not initialized")
        
        # Convert PyTorch tensor to numpy
        input_data = images.cpu().numpy().astype(np.float32)
        
        # Run inference
        outputs = self.onnx_session.run([self.onnx_output_name], {self.onnx_input_name: input_data})
        
        # Convert output to PyTorch tensor
        return torch.from_numpy(outputs[0]).to(self.device)
    
    @staticmethod
    def convert_to_tensorrt(
        pytorch_model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        engine_path: str,
        fp16: bool = True,
        max_batch_size: int = 32
    ):
        """
        Convert PyTorch model to TensorRT engine.
        
        Args:
            pytorch_model: PyTorch model to convert
            input_shape: Input tensor shape (C, H, W)
            engine_path: Path to save TensorRT engine
            fp16: Whether to use FP16 precision
            max_batch_size: Maximum batch size
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install with: pip install tensorrt pycuda")
        
        try:
            import torch2trt
            
            # Create dummy input
            dummy_input = torch.randn(1, *input_shape).cuda()
            
            # Convert to TensorRT
            model_trt = torch2trt.torch2trt(
                pytorch_model.cuda().eval(),
                [dummy_input],
                fp16_mode=fp16,
                max_batch_size=max_batch_size
            )
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(model_trt.engine.serialize())
            
            print(f"TensorRT engine saved: {engine_path}")
            
        except ImportError:
            print("torch2trt not available. Install with: pip install torch2trt")
            print("Alternative: Use TensorRT Python API for conversion")
    
    @staticmethod
    def convert_to_onnx(
        pytorch_model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        onnx_path: str,
        opset_version: int = 11
    ):
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            pytorch_model: PyTorch model to convert
            input_shape: Input tensor shape (C, H, W)
            onnx_path: Path to save ONNX model
            opset_version: ONNX opset version
        """
        # Move model to CPU for ONNX export
        pytorch_model = pytorch_model.cpu()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Try to use the new torch.export-based ONNX exporter first
        try:
            # Check if torch.export is available (PyTorch 2.1+)
            import torch.export as torch_export
            
            # Export to ONNX using the new dynamo-based exporter
            torch.onnx.export(
                pytorch_model.eval(),
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                dynamo=True  # Use the new torch.export-based exporter
            )
            print(f"ONNX model saved using new torch.export-based exporter: {onnx_path}")
            
        except (ImportError, AttributeError, RuntimeError) as e:
            # Fallback to legacy TorchScript-based exporter
            print(f"New torch.export-based exporter not available or failed ({str(e)}), using legacy exporter")
            torch.onnx.export(
                pytorch_model.eval(),
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            print(f"ONNX model saved using legacy TorchScript-based exporter: {onnx_path}")
        
        # Verify ONNX model
        if ONNXRUNTIME_AVAILABLE:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification passed")


def test_backends():
    """Test different inference backends (PyTorch, TensorRT, ONNX Runtime)."""
    print("\n" + "="*60)
    print("TESTING INFERENCE BACKENDS")
    print("="*60)
    
    # Test model and data
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    class_names = [f'class_{i}' for i in range(1000)]  # ImageNet classes
    
    # Create sample input
    sample_input = torch.randn(1, 3, 224, 224)
    
    print("\n1. Testing PyTorch Backend:")
    try:
        engine_pytorch = InferenceEngine(
            model=model,
            class_names=class_names,
            backend='pytorch'
        )
        
        result_pytorch = engine_pytorch.predict_single(sample_input)
        print(f"   ✓ PyTorch inference successful")
        print(f"   Top prediction: {result_pytorch['predicted_class']} ({result_pytorch['confidence']:.3f})")
        
    except Exception as e:
        print(f"   ✗ PyTorch backend failed: {e}")
    
    print("\n2. Testing ONNX Runtime Backend:")
    try:
        # Convert model to ONNX
        onnx_path = "temp_model.onnx"
        InferenceEngine.convert_to_onnx(model, (3, 224, 224), onnx_path)
        
        engine_onnx = InferenceEngine(
            model=None,
            class_names=class_names,
            backend='onnxruntime',
            onnx_model_path=onnx_path
        )
        
        result_onnx = engine_onnx.predict_single(sample_input)
        print(f"   ✓ ONNX Runtime inference successful")
        print(f"   Top prediction: {result_onnx['predicted_class']} ({result_onnx['confidence']:.3f})")
        
        # Clean up
        Path(onnx_path).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"   ✗ ONNX Runtime backend failed: {e}")
        Path("temp_model.onnx").unlink(missing_ok=True)
    
    print("\n3. Testing TensorRT Backend:")
    try:
        if TENSORRT_AVAILABLE:
            # Convert model to TensorRT
            engine_path = "temp_model.trt"
            InferenceEngine.convert_to_tensorrt(model, (3, 224, 224), engine_path)
            
            engine_trt = InferenceEngine(
                model=None,
                class_names=class_names,
                backend='tensorrt',
                tensorrt_engine_path=engine_path
            )
            
            result_trt = engine_trt.predict_single(sample_input)
            print(f"   ✓ TensorRT inference successful")
            print(f"   Top prediction: {result_trt['predicted_class']} ({result_trt['confidence']:.3f})")
            
            # Clean up
            Path(engine_path).unlink(missing_ok=True)
        else:
            print(f"   ⚠ TensorRT not available - skipping test")
            
    except Exception as e:
        print(f"   ✗ TensorRT backend failed: {e}")
        Path("temp_model.trt").unlink(missing_ok=True)
    
    print("\n" + "="*60)
    print("BACKEND TESTING COMPLETED")
    print("="*60)

def benchmark_backends():
    """Benchmark performance across different backends."""
    print("\n" + "="*60)
    print("BENCHMARKING INFERENCE BACKENDS")
    print("="*60)
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    class_names = [f'class_{i}' for i in range(1000)]
    
    # Test configurations
    batch_sizes = [1, 4, 8, 16]
    num_iterations = 50
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBatch Size: {batch_size}")
        sample_input = torch.randn(batch_size, 3, 224, 224)
        
        # PyTorch benchmark
        try:
            engine_pytorch = InferenceEngine(model=model, class_names=class_names, backend='pytorch')
            
            # Warmup
            for _ in range(5):
                _ = engine_pytorch._forward_pass(sample_input)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                _ = engine_pytorch._forward_pass(sample_input)
            pytorch_time = (time.time() - start_time) / num_iterations
            
            results[f'pytorch_bs{batch_size}'] = pytorch_time
            print(f"   PyTorch: {pytorch_time*1000:.2f} ms/batch ({batch_size/pytorch_time:.1f} imgs/sec)")
            
        except Exception as e:
            print(f"   PyTorch failed: {e}")
        
        # ONNX Runtime benchmark
        try:
            onnx_path = f"temp_model_bs{batch_size}.onnx"
            InferenceEngine.convert_to_onnx(model, (3, 224, 224), onnx_path)
            
            engine_onnx = InferenceEngine(
                model=None, 
                class_names=class_names, 
                backend='onnxruntime',
                onnx_model_path=onnx_path
            )
            
            # Warmup
            for _ in range(5):
                _ = engine_onnx._forward_pass(sample_input)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                _ = engine_onnx._forward_pass(sample_input)
            onnx_time = (time.time() - start_time) / num_iterations
            
            results[f'onnx_bs{batch_size}'] = onnx_time
            print(f"   ONNX Runtime: {onnx_time*1000:.2f} ms/batch ({batch_size/onnx_time:.1f} imgs/sec)")
            
            Path(onnx_path).unlink(missing_ok=True)
            
        except Exception as e:
            print(f"   ONNX Runtime failed: {e}")
            Path(f"temp_model_bs{batch_size}.onnx").unlink(missing_ok=True)
    
    # Print summary
    print(f"\n{'Backend':<15} {'Batch Size':<12} {'Time (ms)':<12} {'Throughput (imgs/sec)':<20}")
    print("-" * 60)
    
    for key, time_val in results.items():
        backend, bs_info = key.split('_')
        batch_size = int(bs_info[2:])
        throughput = batch_size / time_val
        print(f"{backend:<15} {batch_size:<12} {time_val*1000:<12.2f} {throughput:<20.1f}")
    
    print("\n" + "="*60)
    print("BACKEND BENCHMARKING COMPLETED")
    print("="*60)

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


def test_sample_images(model_name: str = "resnet18", custom_image_path: str = None, backend: str = "pytorch"):
    """Test inference on sample images or a custom image.
    
    Args:
        model_name: Name of the model to use for testing
        custom_image_path: Path to a custom image file (e.g., 'sampledata/bus.jpg')
    """
    print("\n" + "="*60)
    print("TESTING SAMPLE IMAGES")
    print("="*60)
    
    try:
        import torchvision.transforms as transforms
        from torchvision.models import resnet18
        import cairosvg
        from io import BytesIO
        import json
        
        # Load a pre-trained model
        print(f"Loading pre-trained {model_name}...")
        model = resnet18(pretrained=True)
        
        # Full ImageNet class names (1000 classes)
        imagenet_classes = []
        try:
            # Try to load ImageNet class names from a standard source
            import urllib.request
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            with urllib.request.urlopen(url) as response:
                imagenet_classes = [line.decode('utf-8').strip() for line in response.readlines()]
        except:
            # Fallback to generic class names if download fails
            imagenet_classes = [f"class_{i}" for i in range(1000)]
        
        print(f"Loaded {len(imagenet_classes)} class names")
        
        # Create transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Handle backend-specific initialization
        onnx_model_path = None
        if backend == "onnxruntime" and ONNXRUNTIME_AVAILABLE:
            # Convert PyTorch model to ONNX for ONNX Runtime backend
            onnx_model_path = f"temp_{model_name}.onnx"
            print(f"Converting {model_name} to ONNX format...")
            InferenceEngine.convert_to_onnx(model, (3, 224, 224), onnx_model_path)
        
        # Initialize inference engine
        engine = InferenceEngine(
            model=model,
            class_names=imagenet_classes,
            model_type="pytorch",
            backend=backend,
            onnx_model_path=onnx_model_path
        )
        
        # Test custom image if provided
        if custom_image_path:
            print(f"\nTesting custom image: {custom_image_path}")
            
            try:
                custom_path = Path(custom_image_path)
                if not custom_path.exists():
                    print(f"  Error: Custom image file not found: {custom_image_path}")
                else:
                    print(f"\n--- Testing {custom_path.name} ---")
                    
                    # Load and process the custom image
                    if custom_path.suffix.lower() == '.svg':
                        # Convert SVG to PNG in memory
                        png_data = cairosvg.svg2png(url=str(custom_path))
                        image = Image.open(BytesIO(png_data)).convert('RGB')
                    else:
                        # Load regular image formats (jpg, png, etc.)
                        image = Image.open(custom_path).convert('RGB')
                    
                    # Run inference
                    result = engine.predict_single(
                        image, 
                        transform=transform, 
                        return_probabilities=True,
                        top_k=3
                    )
                    
                    print(f"  Predicted class: {result.get('predicted_class_name', 'Unknown')}")
                    print(f"  Confidence: {result['confidence']:.4f}")
                    print(f"  Inference time: {result['inference_time']:.4f}s")
                    print("  Top 3 predictions:")
                    for i, pred in enumerate(result['top_k_predictions'][:3]):
                        print(f"    {i+1}. {pred.get('class_name', f'Class {pred['class_id']}')} "
                              f"({pred['confidence']:.4f})")
                    
            except Exception as e:
                print(f"  Error processing custom image {custom_image_path}: {e}")
        
        # Test sample images from directory (only if no custom image provided)
        if not custom_image_path:
            sample_dir = Path(__file__).parent / "sample_images"
            if sample_dir.exists():
                print(f"\nTesting sample images from: {sample_dir}")
                
                for image_path in sample_dir.glob("*.svg"):
                    print(f"\n--- Testing {image_path.name} ---")
                    
                    try:
                        # Convert SVG to PNG in memory
                        png_data = cairosvg.svg2png(url=str(image_path))
                        image = Image.open(BytesIO(png_data)).convert('RGB')
                        
                        # Run inference
                        result = engine.predict_single(
                            image, 
                            transform=transform, 
                            return_probabilities=True,
                            top_k=3
                        )
                        
                        print(f"  Predicted class: {result.get('predicted_class_name', 'Unknown')}")
                        print(f"  Confidence: {result['confidence']:.4f}")
                        print(f"  Inference time: {result['inference_time']:.4f}s")
                        print("  Top 3 predictions:")
                        for i, pred in enumerate(result['top_k_predictions'][:3]):
                            print(f"    {i+1}. {pred.get('class_name', f'Class {pred['class_id']}')} "
                                  f"({pred['confidence']:.4f})")
                        
                    except Exception as e:
                        print(f"  Error processing {image_path.name}: {e}")
            else:
                print(f"Sample images directory not found: {sample_dir}")
                print("To test with a custom image, use: --custom-image path/to/your/image.jpg")
            
    except ImportError as e:
        print(f"Missing dependencies for sample image testing: {e}")
        print("Install with: pip install torchvision cairosvg pillow")
    except Exception as e:
        print(f"Error in sample image testing: {e}")
    finally:
        # Clean up temporary ONNX files
        if backend == "onnxruntime" and onnx_model_path:
            try:
                Path(onnx_model_path).unlink(missing_ok=True)
                print(f"Cleaned up temporary ONNX file: {onnx_model_path}")
            except:
                pass


def benchmark_inference():
    """Benchmark inference performance with different configurations."""
    print("\n" + "="*60)
    print("INFERENCE PERFORMANCE BENCHMARK")
    print("="*60)
    
    try:
        import torchvision.transforms as transforms
        from torchvision.models import resnet18, resnet50
        
        # Test configurations
        configs = [
            {"model_name": "resnet18", "batch_sizes": [1, 4, 8, 16]},
            {"model_name": "resnet50", "batch_sizes": [1, 4, 8, 16]},
        ]
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        for config in configs:
            print(f"\n--- Benchmarking {config['model_name']} ---")
            
            # Load model
            if config['model_name'] == "resnet18":
                model = resnet18(pretrained=True)
            else:
                model = resnet50(pretrained=True)
            
            # Initialize engine
            engine = InferenceEngine(
                model=model,
                model_type="pytorch",
                use_amp=True
            )
            
            for batch_size in config['batch_sizes']:
                print(f"\n  Batch size: {batch_size}")
                
                # Create dummy batch
                dummy_batch = torch.randn(batch_size, 3, 224, 224)
                
                # Warmup
                for _ in range(5):
                    _ = engine.predict_batch(dummy_batch, show_progress=False)
                
                # Reset tracking
                engine.reset_performance_tracking()
                
                # Benchmark
                start_time = time.time()
                for _ in range(10):
                    _ = engine.predict_batch(dummy_batch, show_progress=False)
                total_time = time.time() - start_time
                
                # Get stats
                stats = engine.get_performance_stats()
                
                print(f"    Average inference time: {stats['avg_per_sample_time']:.4f}s")
                print(f"    Throughput: {batch_size * 10 / total_time:.2f} images/sec")
                print(f"    Total time (10 batches): {total_time:.4f}s")
                
    except ImportError as e:
        print(f"Missing dependencies for benchmarking: {e}")
        print("Install with: pip install torchvision")
    except Exception as e:
        print(f"Error in benchmarking: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference Engine Testing")
    parser.add_argument("--test", choices=["basic", "samples", "benchmark", "backends", "benchmark_backends", "all"], 
                       default="all", help="Type of test to run")
    parser.add_argument("--model", default="resnet18", 
                       help="Model to use for testing")
    parser.add_argument("--custom-image", type=str, 
                       help="Path to a custom image file for testing (e.g., sampledata/bus.jpg)")
    parser.add_argument('--backend', type=str, choices=['pytorch', 'tensorrt', 'onnxruntime'], 
                       default='pytorch', help='Inference backend to use (default: pytorch)')
    
    args = parser.parse_args()
    
    if args.test in ["basic", "all"]:
        test_inference_engine()
    
    if args.test in ["samples", "all"]:
        test_sample_images(model_name=args.model, custom_image_path=args.custom_image, backend=args.backend)
    
    if args.test in ["benchmark", "all"]:
        benchmark_inference()
    
    if args.test in ["backends", "all"]:
        test_backends()
    
    if args.test in ["benchmark_backends", "all"]:
        benchmark_backends()