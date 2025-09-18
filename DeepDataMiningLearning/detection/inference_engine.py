"""
Comprehensive Object Detection Inference Engine

This module provides a unified interface for object detection inference with support for:
- Multiple model types: YOLO, HuggingFace detection models, TorchVision models
- Multiple backends: PyTorch, TensorRT, ONNX Runtime (CUDA and TensorRT execution providers)
- Batch processing, video processing, and benchmarking capabilities

"""

import torch
import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import argparse

# Core imports
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torchvision.transforms as transforms

# Model imports
from DeepDataMiningLearning.detection.models import (
    create_detectionmodel, 
    get_torchvision_detection_models, 
    load_trained_model
)

# Optional backend imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logging.warning("TensorRT not available. Install tensorrt for TensorRT backend support.")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime not available. Install onnxruntime-gpu for ONNX backend support.")

try:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("HuggingFace transformers not available. Install transformers for HF model support.")


class BackendType(Enum):
    """Supported inference backends"""
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    ONNX_CPU = "onnx_cpu"
    ONNX_CUDA = "onnx_cuda"
    ONNX_TENSORRT = "onnx_tensorrt"


class ModelType(Enum):
    """Supported model types"""
    YOLO = "yolo"
    TORCHVISION = "torchvision"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class DetectionResult:
    """Detection result container"""
    boxes: np.ndarray  # [N, 4] in xyxy format
    scores: np.ndarray  # [N]
    labels: np.ndarray  # [N]
    class_names: List[str]  # [N]
    inference_time: float
    preprocessing_time: float
    postprocessing_time: float


@dataclass
class InferenceConfig:
    """Configuration for inference engine"""
    model_path: str
    model_type: ModelType
    backend: BackendType
    device: str = "cuda:0"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    max_detections: int = 100
    input_size: Tuple[int, int] = (640, 640)
    fp16: bool = False
    batch_size: int = 1
    warmup_runs: int = 3


class DetectionInferenceEngine:
    """
    Unified Object Detection Inference Engine
    
    Supports multiple model types and inference backends with comprehensive
    benchmarking and testing capabilities.
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize the detection inference engine
        
        Args:
            config: InferenceConfig object with model and inference settings
        """
        self.config = config
        self.model = None
        self.preprocessor = None
        self.class_names = []
        self.device = torch.device(config.device)
        
        # Backend-specific attributes
        self.trt_context = None
        self.trt_engine = None
        self.onnx_session = None
        
        # Performance tracking
        self.inference_times = []
        self.preprocessing_times = []
        self.postprocessing_times = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and backend
        self._load_model()
        self._setup_backend()
        
    def _load_model(self):
        """Load the specified model type"""
        self.logger.info(f"Loading {self.config.model_type.value} model from {self.config.model_path}")
        
        if self.config.model_type == ModelType.YOLO:
            self._load_yolo_model()
        elif self.config.model_type == ModelType.TORCHVISION:
            self._load_torchvision_model()
        elif self.config.model_type == ModelType.HUGGINGFACE:
            self._load_huggingface_model()
        elif self.config.model_type == ModelType.CUSTOM:
            self._load_custom_model()
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
    def _load_yolo_model(self):
        """Load YOLO model"""
        try:
            # Extract model name and scale from path
            model_name = "yolov8"  # Default
            scale = "n"  # Default
            
            if "yolov8" in self.config.model_path.lower():
                model_name = "yolov8"
                if "yolov8n" in self.config.model_path.lower():
                    scale = "n"
                elif "yolov8s" in self.config.model_path.lower():
                    scale = "s"
                elif "yolov8m" in self.config.model_path.lower():
                    scale = "m"
                elif "yolov8l" in self.config.model_path.lower():
                    scale = "l"
                elif "yolov8x" in self.config.model_path.lower():
                    scale = "x"
            
            self.model, self.preprocessor, self.class_names = create_detectionmodel(
                modelname=model_name,
                num_classes=80,  # COCO classes
                trainable_layers=0,
                ckpt_file=self.config.model_path,
                fp16=self.config.fp16,
                device=self.config.device,
                scale=scale
            )
            
            self.model.eval()
            self.logger.info(f"Successfully loaded YOLO model: {model_name}{scale}")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
            
    def _load_torchvision_model(self):
        """Load TorchVision model"""
        try:
            # Extract model name from path or use default
            model_name = "fasterrcnn_resnet50_fpn_v2"
            
            if Path(self.config.model_path).exists():
                # Load trained model
                num_classes = 91  # COCO + background
                self.model, self.preprocessor = load_trained_model(
                    model_name, num_classes, self.config.model_path
                )
            else:
                # Load pretrained model
                self.model, self.preprocessor, weights, self.class_names = get_torchvision_detection_models(
                    model_name
                )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Successfully loaded TorchVision model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load TorchVision model: {e}")
            raise
            
    def _load_huggingface_model(self):
        """Load HuggingFace model"""
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers not available")
            
        try:
            # Support for popular HF detection models
            model_name = self.config.model_path
            
            self.preprocessor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Get class names from model config
            if hasattr(self.model.config, 'id2label'):
                self.class_names = list(self.model.config.id2label.values())
            else:
                self.class_names = [f"class_{i}" for i in range(91)]  # Default COCO
                
            self.logger.info(f"Successfully loaded HuggingFace model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace model: {e}")
            raise
            
    def _load_custom_model(self):
        """Load custom model"""
        # Placeholder for custom model loading logic
        # Users can extend this method for their specific models
        raise NotImplementedError("Custom model loading not implemented")
        
    def _setup_backend(self):
        """Setup the specified inference backend"""
        self.logger.info(f"Setting up {self.config.backend.value} backend")
        
        if self.config.backend == BackendType.PYTORCH:
            self._setup_pytorch_backend()
        elif self.config.backend == BackendType.TENSORRT:
            self._setup_tensorrt_backend()
        elif self.config.backend in [BackendType.ONNX_CPU, BackendType.ONNX_CUDA, BackendType.ONNX_TENSORRT]:
            self._setup_onnx_backend()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
            
    def _setup_pytorch_backend(self):
        """Setup PyTorch backend"""
        if self.config.fp16 and self.device.type == 'cuda':
            self.model = self.model.half()
        self.logger.info("PyTorch backend ready")
        
    def _setup_tensorrt_backend(self):
        """Setup TensorRT backend"""
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not available")
            
        self._load_tensorrt_backend()
        
    def _load_tensorrt_backend(self) -> None:
        """Load TensorRT backend"""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT and PyCUDA are not available. Please install tensorrt and pycuda packages.")
        
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            from cuda import cudart
        except ImportError:
            raise ImportError("TensorRT and PyCUDA are required for TensorRT backend")
        
        self.logger.info("Loading TensorRT backend...")
        
        # Check if model path is a TensorRT engine file
        if self.config.model_path.endswith('.engine') or self.config.model_path.endswith('.trt'):
            engine_path = self.config.model_path
        else:
            # Convert PyTorch/ONNX model to TensorRT engine
            engine_path = self._convert_to_tensorrt()
        
        # Load TensorRT engine
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.trt_logger)
        self.trt_engine = runtime.deserialize_cuda_engine(engine_data)
        self.trt_context = self.trt_engine.create_execution_context()
        
        # Get input/output bindings
        self.trt_bindings = []
        self.trt_inputs = []
        self.trt_outputs = []
        
        for i in range(self.trt_engine.num_bindings):
            binding_name = self.trt_engine.get_binding_name(i)
            binding_shape = self.trt_engine.get_binding_shape(i)
            binding_dtype = trt.nptype(self.trt_engine.get_binding_dtype(i))
            
            if self.trt_engine.binding_is_input(i):
                self.trt_inputs.append({
                    'name': binding_name,
                    'shape': binding_shape,
                    'dtype': binding_dtype,
                    'index': i
                })
            else:
                self.trt_outputs.append({
                    'name': binding_name,
                    'shape': binding_shape,
                    'dtype': binding_dtype,
                    'index': i
                })
        
        # Allocate GPU memory
        self._allocate_trt_buffers()
        
        self.logger.info(f"TensorRT engine loaded successfully")
        self.logger.info(f"Input bindings: {[inp['name'] for inp in self.trt_inputs]}")
        self.logger.info(f"Output bindings: {[out['name'] for out in self.trt_outputs]}")
    
    def _convert_to_tensorrt(self) -> str:
        """Convert model to TensorRT engine"""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available. Please install tensorrt package.")
        
        import tensorrt as trt
        import os
        
        engine_path = self.config.model_path.replace('.pt', '.engine').replace('.onnx', '.engine')
        
        if os.path.exists(engine_path):
            self.logger.info(f"Using existing TensorRT engine: {engine_path}")
            return engine_path
        
        self.logger.info(f"Converting model to TensorRT engine: {engine_path}")
        
        # Create TensorRT builder
        builder = trt.Builder(self.trt_logger)
        config = builder.create_builder_config()
        
        # Set memory pool size (8GB)
        config.max_workspace_size = 8 << 30
        
        # Enable FP16 if requested
        if self.config.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        # Create network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        if self.config.model_path.endswith('.onnx'):
            # Parse ONNX model
            parser = trt.OnnxParser(network, self.trt_logger)
            with open(self.config.model_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        self.logger.error(parser.get_error(error))
                    raise RuntimeError("Failed to parse ONNX model")
        else:
            # For PyTorch models, first convert to ONNX
            onnx_path = self._convert_pytorch_to_onnx()
            parser = trt.OnnxParser(network, self.trt_logger)
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        self.logger.error(parser.get_error(error))
                    raise RuntimeError("Failed to parse ONNX model")
        
        # Set optimization profiles for dynamic shapes
        profile = builder.create_optimization_profile()
        
        # Configure input shapes (assuming batch size can vary)
        input_name = network.get_input(0).name
        min_shape = (1, 3, *self.config.input_size)
        opt_shape = (self.config.batch_size, 3, *self.config.input_size)
        max_shape = (self.config.batch_size * 2, 3, *self.config.input_size)
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # Build engine
        self.logger.info("Building TensorRT engine... This may take several minutes.")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        self.logger.info(f"TensorRT engine saved to: {engine_path}")
        return engine_path
    
    def _convert_pytorch_to_onnx(self) -> str:
        """Convert PyTorch model to ONNX"""
        import os
        
        onnx_path = self.config.model_path.replace('.pt', '.onnx')
        
        if os.path.exists(onnx_path):
            self.logger.info(f"Using existing ONNX model: {onnx_path}")
            return onnx_path
        
        self.logger.info(f"Converting PyTorch model to ONNX: {onnx_path}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *self.config.input_size).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        self.logger.info(f"ONNX model saved to: {onnx_path}")
        return onnx_path
    
    def _allocate_trt_buffers(self) -> None:
        """Allocate GPU memory for TensorRT inference"""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT and PyCUDA are not available. Please install tensorrt and pycuda packages.")
        
        import pycuda.driver as cuda
        
        self.trt_input_buffers = []
        self.trt_output_buffers = []
        self.trt_bindings = [None] * self.trt_engine.num_bindings
        
        # Allocate input buffers
        for inp in self.trt_inputs:
            # Calculate size for maximum batch size
            max_batch_size = self.config.batch_size * 2
            shape = list(inp['shape'])
            if shape[0] == -1:  # Dynamic batch dimension
                shape[0] = max_batch_size
            
            size = np.prod(shape) * np.dtype(inp['dtype']).itemsize
            host_mem = cuda.pagelocked_empty(shape, inp['dtype'])
            device_mem = cuda.mem_alloc(size)
            
            self.trt_input_buffers.append({
                'host': host_mem,
                'device': device_mem,
                'shape': shape,
                'dtype': inp['dtype']
            })
            self.trt_bindings[inp['index']] = int(device_mem)
        
        # Allocate output buffers
        for out in self.trt_outputs:
            # Calculate size for maximum batch size
            max_batch_size = self.config.batch_size * 2
            shape = list(out['shape'])
            if shape[0] == -1:  # Dynamic batch dimension
                shape[0] = max_batch_size
            
            size = np.prod(shape) * np.dtype(out['dtype']).itemsize
            host_mem = cuda.pagelocked_empty(shape, out['dtype'])
            device_mem = cuda.mem_alloc(size)
            
            self.trt_output_buffers.append({
                'host': host_mem,
                'device': device_mem,
                'shape': shape,
                'dtype': out['dtype']
            })
            self.trt_bindings[out['index']] = int(device_mem)
        
        # Create CUDA stream
        self.trt_stream = cuda.Stream()
    
    def _run_onnx_inference(self, input_tensor: torch.Tensor) -> Any:
        """Run ONNX Runtime inference"""
        # Convert input tensor to numpy
        input_np = input_tensor.cpu().numpy()
        
        # Prepare input dictionary
        input_name = self.onnx_inputs[0].name
        inputs = {input_name: input_np}
        
        # Run inference
        outputs = self.onnx_session.run(None, inputs)
        
        # Convert outputs back to tensors
        if len(outputs) == 1:
            return torch.from_numpy(outputs[0])
        else:
            return [torch.from_numpy(output) for output in outputs]
        
    def _setup_onnx_backend(self):
        """Setup ONNX Runtime backend"""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
            
        self._load_onnx_backend()
    
    def _load_onnx_backend(self) -> None:
        """Load ONNX Runtime backend"""
        import onnxruntime as ort
        
        self.logger.info(f"Loading ONNX Runtime backend: {self.config.backend}")
        
        # Check if model path is an ONNX file
        if self.config.model_path.endswith('.onnx'):
            onnx_path = self.config.model_path
        else:
            # Convert PyTorch model to ONNX
            onnx_path = self._convert_pytorch_to_onnx()
        
        # Configure execution providers based on backend type
        providers = self._get_onnx_providers()
        
        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable parallel execution
        sess_options.intra_op_num_threads = 0  # Use all available cores
        sess_options.inter_op_num_threads = 0
        
        # Enable memory pattern optimization
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        # Set execution mode
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        try:
            self.onnx_session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=providers
            )
        except Exception as e:
            self.logger.error(f"Failed to create ONNX Runtime session: {e}")
            # Fallback to CPU provider
            self.logger.warning("Falling back to CPU provider")
            self.onnx_session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
        
        # Get input/output information
        self.onnx_inputs = self.onnx_session.get_inputs()
        self.onnx_outputs = self.onnx_session.get_outputs()
        
        # Log session information
        self.logger.info(f"ONNX Runtime session created successfully")
        self.logger.info(f"Available providers: {ort.get_available_providers()}")
        self.logger.info(f"Session providers: {self.onnx_session.get_providers()}")
        self.logger.info(f"Input names: {[inp.name for inp in self.onnx_inputs]}")
        self.logger.info(f"Output names: {[out.name for out in self.onnx_outputs]}")
        
        # Validate input shapes
        for inp in self.onnx_inputs:
            self.logger.info(f"Input '{inp.name}': shape={inp.shape}, type={inp.type}")
        
        for out in self.onnx_outputs:
            self.logger.info(f"Output '{out.name}': shape={out.shape}, type={out.type}")
    
    def _get_onnx_providers(self) -> List[str]:
        """Get ONNX Runtime execution providers based on backend type"""
        import onnxruntime as ort
        
        available_providers = ort.get_available_providers()
        
        if self.config.backend == BackendType.ONNX_CPU:
            return ['CPUExecutionProvider']
        
        elif self.config.backend == BackendType.ONNX_CUDA:
            providers = []
            if 'CUDAExecutionProvider' in available_providers:
                # Configure CUDA provider options
                cuda_options = {
                    'device_id': int(self.config.device.split(':')[-1]) if ':' in self.config.device else 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
                providers.append(('CUDAExecutionProvider', cuda_options))
            else:
                self.logger.warning("CUDA provider not available, falling back to CPU")
            
            providers.append('CPUExecutionProvider')
            return providers
        
        elif self.config.backend == BackendType.ONNX_TENSORRT:
            providers = []
            if 'TensorrtExecutionProvider' in available_providers:
                # Configure TensorRT provider options
                trt_options = {
                    'device_id': int(self.config.device.split(':')[-1]) if ':' in self.config.device else 0,
                    'trt_max_workspace_size': 8 * 1024 * 1024 * 1024,  # 8GB
                    'trt_fp16_enable': self.config.fp16,
                    'trt_int8_enable': False,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': './trt_cache',
                    'trt_dump_subgraphs': False,
                }
                providers.append(('TensorrtExecutionProvider', trt_options))
            else:
                self.logger.warning("TensorRT provider not available")
            
            # Fallback providers
            if 'CUDAExecutionProvider' in available_providers:
                cuda_options = {
                    'device_id': int(self.config.device.split(':')[-1]) if ':' in self.config.device else 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB
                }
                providers.append(('CUDAExecutionProvider', cuda_options))
            
            providers.append('CPUExecutionProvider')
            return providers
        
        else:
            return ['CPUExecutionProvider']
            
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess input image for inference
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed tensor
        """
        start_time = time.time()
        
        # Load image if path is provided
        if isinstance(image, str):
            if self.config.model_type == ModelType.YOLO:
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = read_image(image)
        elif isinstance(image, np.ndarray):
            img = image
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
            
        # Apply model-specific preprocessing
        if self.config.model_type == ModelType.YOLO:
            if isinstance(img, np.ndarray):
                processed = self.preprocessor([img])
            else:
                processed = self.preprocessor(img)
        elif self.config.model_type == ModelType.HUGGINGFACE:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            processed = self.preprocessor(img, return_tensors="pt")
            processed = processed['pixel_values']
        else:
            # TorchVision models
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            processed = [self.preprocessor(img)]
            processed = torch.stack(processed)
            
        # Move to device
        if isinstance(processed, torch.Tensor):
            processed = processed.to(self.device)
        elif isinstance(processed, list):
            processed = [p.to(self.device) for p in processed]
            
        preprocessing_time = time.time() - start_time
        self.preprocessing_times.append(preprocessing_time)
        
        return processed
        
    def run_inference(self, preprocessed_input: torch.Tensor) -> Any:
        """
        Run inference on preprocessed input
        
        Args:
            preprocessed_input: Preprocessed tensor
            
        Returns:
            Raw model output
        """
        start_time = time.time()
        
        with torch.no_grad():
            if self.config.backend == BackendType.PYTORCH:
                if self.config.model_type == ModelType.HUGGINGFACE:
                    outputs = self.model(preprocessed_input)
                else:
                    outputs = self.model(preprocessed_input)
            elif self.config.backend in [BackendType.ONNX_CPU, BackendType.ONNX_CUDA, BackendType.ONNX_TENSORRT]:
                # ONNX Runtime inference
                outputs = self._run_onnx_inference(preprocessed_input)
            elif self.config.backend == BackendType.TENSORRT:
                # TensorRT inference
                outputs = self._run_tensorrt_inference(preprocessed_input)
            else:
                raise ValueError(f"Unsupported backend: {self.config.backend}")
                
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return outputs
        
    def postprocess_outputs(self, outputs: Any, original_image_shape: Tuple[int, int]) -> DetectionResult:
        """
        Postprocess model outputs to detection results
        
        Args:
            outputs: Raw model outputs
            original_image_shape: Original image shape (H, W)
            
        Returns:
            DetectionResult object
        """
        start_time = time.time()
        
        if self.config.model_type == ModelType.YOLO:
            # YOLO postprocessing
            detections = self.preprocessor.postprocess(
                outputs, 
                self.config.input_size, 
                [original_image_shape]
            )
            detection = detections[0]
            
            # Filter by confidence
            mask = detection['scores'] >= self.config.confidence_threshold
            boxes = detection['boxes'][mask].cpu().numpy()
            scores = detection['scores'][mask].cpu().numpy()
            labels = detection['labels'][mask].cpu().numpy()
            
        elif self.config.model_type == ModelType.HUGGINGFACE:
            # HuggingFace postprocessing
            target_sizes = torch.tensor([original_image_shape])
            results = self.preprocessor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.config.confidence_threshold
            )[0]
            
            boxes = results['boxes'].cpu().numpy()
            scores = results['scores'].cpu().numpy()
            labels = results['labels'].cpu().numpy()
            
        else:
            # TorchVision postprocessing
            detection = outputs[0]
            mask = detection['scores'] >= self.config.confidence_threshold
            boxes = detection['boxes'][mask].cpu().numpy()
            scores = detection['scores'][mask].cpu().numpy()
            labels = detection['labels'][mask].cpu().numpy()
            
        # Apply NMS if needed
        if len(boxes) > 0:
            keep_indices = self._apply_nms(boxes, scores)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
            
        # Limit detections
        if len(boxes) > self.config.max_detections:
            top_indices = np.argsort(scores)[::-1][:self.config.max_detections]
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            labels = labels[top_indices]
            
        # Get class names
        class_names = [self.class_names[int(label)] if int(label) < len(self.class_names) 
                      else f"class_{int(label)}" for label in labels]
        
        postprocessing_time = time.time() - start_time
        self.postprocessing_times.append(postprocessing_time)
        
        return DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_names=class_names,
            inference_time=self.inference_times[-1] if self.inference_times else 0.0,
            preprocessing_time=self.preprocessing_times[-1] if self.preprocessing_times else 0.0,
            postprocessing_time=postprocessing_time
        )
        
    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Apply Non-Maximum Suppression"""
        try:
            import torchvision.ops as ops
            boxes_tensor = torch.from_numpy(boxes)
            scores_tensor = torch.from_numpy(scores)
            keep = ops.nms(boxes_tensor, scores_tensor, self.config.nms_threshold)
            return keep.numpy()
        except:
            # Fallback to simple NMS or return all
            return np.arange(len(boxes))
            
    def _run_tensorrt_inference(self, input_tensor: torch.Tensor) -> Any:
        """Run TensorRT inference"""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT and PyCUDA are not available. Please install tensorrt and pycuda packages.")
        
        import pycuda.driver as cuda
        
        # Convert input tensor to numpy and copy to host buffer
        input_np = input_tensor.cpu().numpy()
        batch_size = input_np.shape[0]
        
        # Set dynamic shapes if needed
        if self.trt_engine.has_implicit_batch_dimension:
            self.trt_context.set_binding_shape(0, input_np.shape)
        
        # Copy input data to host buffer
        input_buffer = self.trt_input_buffers[0]
        np.copyto(input_buffer['host'][:batch_size], input_np)
        
        # Transfer input data to GPU
        cuda.memcpy_htod_async(input_buffer['device'], input_buffer['host'], self.trt_stream)
        
        # Run inference
        self.trt_context.execute_async_v2(bindings=self.trt_bindings, stream_handle=self.trt_stream.handle)
        
        # Transfer output data back to CPU
        outputs = []
        for i, output_buffer in enumerate(self.trt_output_buffers):
            cuda.memcpy_dtoh_async(output_buffer['host'], output_buffer['device'], self.trt_stream)
            self.trt_stream.synchronize()
            
            # Get actual output shape based on batch size
            output_shape = list(output_buffer['shape'])
            if output_shape[0] == -1 or output_shape[0] > batch_size:
                output_shape[0] = batch_size
            
            output_data = output_buffer['host'][:np.prod(output_shape)].reshape(output_shape)
            outputs.append(torch.from_numpy(output_data.copy()))
        
        # Return outputs in format expected by postprocessing
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
        
    def detect(self, image: Union[str, np.ndarray, Image.Image]) -> DetectionResult:
        """
        Run complete detection pipeline on a single image
        
        Args:
            image: Input image
            
        Returns:
            DetectionResult object
        """
        # Get original image shape
        if isinstance(image, str):
            img = cv2.imread(image)
            original_shape = img.shape[:2]
        elif isinstance(image, np.ndarray):
            original_shape = image.shape[:2]
        elif isinstance(image, Image.Image):
            original_shape = (image.height, image.width)
        else:
            original_shape = (640, 640)  # Default
            
        # Run pipeline
        preprocessed = self.preprocess_image(image)
        outputs = self.run_inference(preprocessed)
        result = self.postprocess_outputs(outputs, original_shape)
        
        return result
        
    def detect_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[DetectionResult]:
        """
        Run detection on a batch of images
        
        Args:
            images: List of input images
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        for image in images:
            result = self.detect(image)
            results.append(result)
        return results
        
    def benchmark(self, image: Union[str, np.ndarray, Image.Image], num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark the inference engine
        
        Args:
            image: Test image
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark statistics
        """
        self.logger.info(f"Running benchmark with {num_runs} iterations...")
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            _ = self.detect(image)
            
        # Clear previous times
        self.inference_times.clear()
        self.preprocessing_times.clear()
        self.postprocessing_times.clear()
        
        # Benchmark runs
        total_start = time.time()
        for _ in range(num_runs):
            _ = self.detect(image)
        total_time = time.time() - total_start
        
        # Calculate statistics
        stats = {
            'total_time': total_time,
            'avg_total_time': total_time / num_runs,
            'fps': num_runs / total_time,
            'avg_preprocessing_time': np.mean(self.preprocessing_times),
            'avg_inference_time': np.mean(self.inference_times),
            'avg_postprocessing_time': np.mean(self.postprocessing_times),
            'std_preprocessing_time': np.std(self.preprocessing_times),
            'std_inference_time': np.std(self.inference_times),
            'std_postprocessing_time': np.std(self.postprocessing_times),
        }
        
        return stats
    
    def test_accuracy(self, test_images: List[str], ground_truth: List[Dict], 
                     iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Test model accuracy on a dataset
        
        Args:
            test_images: List of test image paths
            ground_truth: List of ground truth annotations
            iou_threshold: IoU threshold for matching predictions
            
        Returns:
            Accuracy metrics (mAP, precision, recall)
        """
        self.logger.info(f"Testing accuracy on {len(test_images)} images...")
        
        all_predictions = []
        all_ground_truths = []
        
        for img_path, gt in zip(test_images, ground_truth):
            result = self.detect(img_path)
            
            # Convert predictions to evaluation format
            predictions = {
                'boxes': result.boxes,
                'scores': result.scores,
                'labels': result.labels
            }
            all_predictions.append(predictions)
            all_ground_truths.append(gt)
        
        # Calculate metrics (simplified implementation)
        metrics = self._calculate_metrics(all_predictions, all_ground_truths, iou_threshold)
        return metrics
    
    def _calculate_metrics(self, predictions: List[Dict], ground_truths: List[Dict], 
                          iou_threshold: float) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        # Simplified metric calculation
        # In practice, you would use libraries like pycocotools
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred['boxes']
            gt_boxes = gt.get('boxes', [])
            
            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue
            elif len(pred_boxes) == 0:
                total_fn += len(gt_boxes)
                continue
            elif len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue
            
            # Calculate IoU matrix and match predictions
            matched_gt = set()
            for i, pred_box in enumerate(pred_boxes):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt_box in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    
                    iou = self._calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_threshold:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1
            
            total_fn += len(gt_boxes) - len(matched_gt)
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        }
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two boxes"""
        # Convert to [x1, y1, x2, y2] format if needed
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def profile_model(self, image: Union[str, np.ndarray, Image.Image], 
                     detailed: bool = False) -> Dict[str, Any]:
        """
        Profile model performance and memory usage
        
        Args:
            image: Test image
            detailed: Whether to include detailed profiling
            
        Returns:
            Profiling results
        """
        import psutil
        import gc
        
        self.logger.info("Profiling model performance...")
        
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Run inference with profiling
        if detailed and hasattr(torch, 'profiler'):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                result = self.detect(image)
            
            # Get profiler results
            profiler_output = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        else:
            result = self.detect(image)
            profiler_output = None
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        gpu_memory_usage = 0
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_usage = gpu_memory_after - gpu_memory_before
        
        profile_results = {
            'inference_time': result.inference_time,
            'preprocessing_time': result.preprocessing_time,
            'postprocessing_time': result.postprocessing_time,
            'total_time': result.inference_time + result.preprocessing_time + result.postprocessing_time,
            'memory_usage_mb': memory_usage,
            'gpu_memory_usage_mb': gpu_memory_usage,
            'num_detections': len(result.boxes),
            'model_type': self.config.model_type.value,
            'backend': self.config.backend.value,
            'device': self.config.device,
            'input_size': self.config.input_size,
            'fp16': self.config.fp16
        }
        
        if profiler_output:
            profile_results['detailed_profile'] = profiler_output
        
        return profile_results
    
    def detect_video(self, video_path: str, output_path: Optional[str] = None, 
                    skip_frames: int = 0, max_frames: Optional[int] = None,
                    show_progress: bool = True) -> Dict[str, Any]:
        """
        Detect objects in video with advanced processing options
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            skip_frames: Number of frames to skip between detections
            max_frames: Maximum number of frames to process
            show_progress: Whether to show progress bar
            
        Returns:
            Video processing statistics
        """
        import cv2
        from tqdm import tqdm
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing statistics
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'skipped_frames': 0,
            'total_detections': 0,
            'processing_times': [],
            'detection_counts': [],
            'fps_values': []
        }
        
        frame_idx = 0
        processed_count = 0
        
        # Progress bar
        pbar = tqdm(total=min(total_frames, max_frames or total_frames), 
                   desc="Processing video", disable=not show_progress)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check frame limits
                if max_frames and processed_count >= max_frames:
                    break
                
                # Skip frames if specified
                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    stats['skipped_frames'] += 1
                    frame_idx += 1
                    pbar.update(1)
                    continue
                
                # Process frame
                start_time = time.time()
                result = self.detect(frame)
                process_time = time.time() - start_time
                
                # Update statistics
                stats['processed_frames'] += 1
                stats['total_detections'] += len(result.boxes)
                stats['processing_times'].append(process_time)
                stats['detection_counts'].append(len(result.boxes))
                stats['fps_values'].append(1.0 / process_time if process_time > 0 else 0)
                
                # Visualize and save
                if writer or output_path:
                    vis_img = self.visualize_results(frame, result)
                    vis_frame = np.array(vis_img)
                    
                    # Add performance info
                    fps_text = f"FPS: {1/process_time:.1f} | Detections: {len(result.boxes)}"
                    cv2.putText(vis_frame, fps_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if writer:
                        writer.write(vis_frame)
                
                processed_count += 1
                frame_idx += 1
                pbar.update(1)
                
                # Update progress description
                if show_progress and processed_count % 30 == 0:
                    avg_fps = np.mean(stats['fps_values'][-30:])
                    pbar.set_description(f"Processing video (avg FPS: {avg_fps:.1f})")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            pbar.close()
        
        # Calculate final statistics
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['avg_fps'] = np.mean(stats['fps_values'])
            stats['avg_detections_per_frame'] = np.mean(stats['detection_counts'])
            stats['total_processing_time'] = sum(stats['processing_times'])
        
        return stats
    
    def detect_webcam(self, camera_id: int = 0, output_path: Optional[str] = None,
                     max_duration: Optional[float] = None) -> None:
        """
        Real-time detection from webcam
        
        Args:
            camera_id: Camera device ID
            output_path: Path to save output video (optional)
            max_duration: Maximum recording duration in seconds
        """
        import cv2
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        self.logger.info(f"Camera: {width}x{height}, {fps} FPS")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check duration limit
                if max_duration and (time.time() - start_time) > max_duration:
                    break
                
                # Run detection
                result = self.detect(frame)
                
                # Visualize results
                vis_img = self.visualize_results(frame, result)
                vis_frame = np.array(vis_img)
                
                # Add info overlay
                fps_text = f"FPS: {1/result.inference_time:.1f} | Objects: {len(result.boxes)}"
                cv2.putText(vis_frame, fps_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Real-time Detection', vis_frame)
                
                # Save frame
                if writer:
                    writer.write(vis_frame)
                
                frame_count += 1
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
        total_time = time.time() - start_time
        self.logger.info(f"Webcam session: {frame_count} frames in {total_time:.1f}s")
        
    def visualize_results(self, image: Union[str, np.ndarray, Image.Image], 
                         result: DetectionResult, save_path: Optional[str] = None) -> Image.Image:
        """
        Visualize detection results on image
        
        Args:
            image: Original image
            result: Detection results
            save_path: Optional path to save visualization
            
        Returns:
            PIL Image with visualizations
        """
        # Load image if needed
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = np.array(image)
            
        # Draw bounding boxes
        for i, (box, score, class_name) in enumerate(zip(result.boxes, result.scores, result.class_names)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        
        # Save if requested
        if save_path:
            pil_img.save(save_path)
            
        return pil_img


def create_inference_engine(model_path: str, model_type: str, backend: str = "pytorch", **kwargs) -> DetectionInferenceEngine:
    """
    Factory function to create inference engine
    
    Args:
        model_path: Path to model file
        model_type: Type of model (yolo, torchvision, huggingface, custom)
        backend: Inference backend (pytorch, tensorrt, onnx_cpu, onnx_cuda, onnx_tensorrt)
        **kwargs: Additional configuration parameters
        
    Returns:
        DetectionInferenceEngine instance
    """
    config = InferenceConfig(
        model_path=model_path,
        model_type=ModelType(model_type.lower()),
        backend=BackendType(backend.lower()),
        **kwargs
    )
    
    return DetectionInferenceEngine(config)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Object Detection Inference Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # YOLO inference with PyTorch backend
  python inference_engine.py --model-path yolov8n.pt --model-type yolo --input image.jpg
  
  # HuggingFace DETR with ONNX CUDA backend
  python inference_engine.py --model-path facebook/detr-resnet-50 --model-type huggingface --backend onnx_cuda --input image.jpg
  
  # Batch processing with TensorRT
  python inference_engine.py --model-path model.pt --model-type yolo --backend tensorrt --input-dir images/ --output-dir results/
  
  # Video processing
  python inference_engine.py --model-path model.pt --model-type yolo --input video.mp4 --output video_output.mp4
  
  # Benchmarking
  python inference_engine.py --model-path model.pt --model-type yolo --benchmark --benchmark-runs 100 --input image.jpg
        """
    )
    
    # Model configuration
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model file or HuggingFace model name')
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['yolo', 'torchvision', 'huggingface', 'custom'],
                       help='Type of model to use')
    parser.add_argument('--backend', type=str, default='pytorch',
                       choices=['pytorch', 'tensorrt', 'onnx_cpu', 'onnx_cuda', 'onnx_tensorrt'],
                       help='Inference backend to use')
    
    # Input/Output
    parser.add_argument('--input', type=str,
                       help='Input image or video file')
    parser.add_argument('--input-dir', type=str,
                       help='Input directory for batch processing')
    parser.add_argument('--output', type=str,
                       help='Output file path')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for batch processing')
    
    # Inference parameters
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for inference')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--nms-threshold', type=float, default=0.45,
                       help='NMS threshold')
    parser.add_argument('--max-detections', type=int, default=100,
                       help='Maximum number of detections per image')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640],
                       help='Input size (height width)')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 precision')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for processing')
    
    # Benchmarking
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmarking')
    parser.add_argument('--benchmark-runs', type=int, default=100,
                       help='Number of benchmark runs')
    parser.add_argument('--warmup-runs', type=int, default=3,
                       help='Number of warmup runs')
    
    # Visualization
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip visualization')
    parser.add_argument('--save-visualization', action='store_true',
                       help='Save visualization results')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--log-file', type=str,
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if args.log_file:
        logging.basicConfig(level=log_level, format=log_format, 
                          handlers=[
                              logging.FileHandler(args.log_file),
                              logging.StreamHandler()
                          ])
    else:
        logging.basicConfig(level=log_level, format=log_format)
    
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not args.input and not args.input_dir and not args.benchmark:
        parser.error("Must specify --input, --input-dir, or --benchmark")
    
    # Create inference configuration
    config = InferenceConfig(
        model_path=args.model_path,
        model_type=ModelType(args.model_type),
        backend=BackendType(args.backend),
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections,
        input_size=tuple(args.input_size),
        fp16=args.fp16,
        batch_size=args.batch_size,
        warmup_runs=args.warmup_runs
    )
    
    # Create inference engine
    try:
        engine = DetectionInferenceEngine(config)
        logger.info("Inference engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        return 1
    
    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run benchmarking if requested
    if args.benchmark:
        if not args.input:
            logger.error("Benchmark requires --input image")
            return 1
            
        logger.info("Running benchmark...")
        stats = engine.benchmark(args.input, args.benchmark_runs)
        
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        print(f"Model: {args.model_path}")
        print(f"Backend: {args.backend}")
        print(f"Device: {args.device}")
        print(f"Runs: {args.benchmark_runs}")
        print("-"*50)
        print(f"Average FPS: {stats['fps']:.2f}")
        print(f"Average Total Time: {stats['avg_total_time']*1000:.2f} ms")
        print(f"Average Preprocessing: {stats['avg_preprocessing_time']*1000:.2f} ms")
        print(f"Average Inference: {stats['avg_inference_time']*1000:.2f} ms")
        print(f"Average Postprocessing: {stats['avg_postprocessing_time']*1000:.2f} ms")
        print("-"*50)
        print(f"Std Preprocessing: {stats['std_preprocessing_time']*1000:.2f} ms")
        print(f"Std Inference: {stats['std_inference_time']*1000:.2f} ms")
        print(f"Std Postprocessing: {stats['std_postprocessing_time']*1000:.2f} ms")
        print("="*50)
        
        return 0
    
    # Process single image/video
    if args.input:
        input_path = Path(args.input)
        
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input}")
            return 1
        
        # Check if it's a video file
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        if input_path.suffix.lower() in video_extensions:
            logger.info("Processing video...")
            process_video(engine, args.input, args.output, args)
        else:
            logger.info("Processing single image...")
            result = engine.detect(args.input)
            
            # Print results
            print(f"\nDetected {len(result.boxes)} objects:")
            for i, (box, score, class_name) in enumerate(zip(result.boxes, result.scores, result.class_names)):
                print(f"  {i+1}. {class_name}: {score:.3f} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
            
            print(f"\nTiming:")
            print(f"  Preprocessing: {result.preprocessing_time*1000:.2f} ms")
            print(f"  Inference: {result.inference_time*1000:.2f} ms")
            print(f"  Postprocessing: {result.postprocessing_time*1000:.2f} ms")
            print(f"  Total: {(result.preprocessing_time + result.inference_time + result.postprocessing_time)*1000:.2f} ms")
            
            # Visualize results
            if not args.no_visualization:
                output_path = args.output or f"{input_path.stem}_result{input_path.suffix}"
                vis_img = engine.visualize_results(args.input, result, output_path)
                logger.info(f"Visualization saved to: {output_path}")
    
    # Process batch of images
    if args.input_dir:
        input_dir = Path(args.input_dir)
        
        if not input_dir.exists():
            logger.error(f"Input directory not found: {args.input_dir}")
            return 1
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.error(f"No image files found in: {args.input_dir}")
            return 1
        
        logger.info(f"Processing {len(image_files)} images...")
        
        total_time = 0
        total_detections = 0
        
        for i, image_file in enumerate(image_files):
            start_time = time.time()
            result = engine.detect(str(image_file))
            process_time = time.time() - start_time
            
            total_time += process_time
            total_detections += len(result.boxes)
            
            # Save visualization if requested
            if args.save_visualization or not args.no_visualization:
                output_path = Path(args.output_dir) / f"{image_file.stem}_result{image_file.suffix}"
                engine.visualize_results(str(image_file), result, str(output_path))
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_files)} images")
        
        # Print batch statistics
        avg_time = total_time / len(image_files)
        avg_detections = total_detections / len(image_files)
        
        print(f"\nBatch Processing Results:")
        print(f"  Images processed: {len(image_files)}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Average time per image: {avg_time*1000:.2f} ms")
        print(f"  Average FPS: {1/avg_time:.2f}")
        print(f"  Total detections: {total_detections}")
        print(f"  Average detections per image: {avg_detections:.1f}")
    
    return 0


def process_video(engine: DetectionInferenceEngine, input_path: str, output_path: Optional[str], args) -> None:
    """Process video file"""
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger = logging.getLogger(__name__)
    logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    total_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            start_time = time.time()
            result = engine.detect(frame)
            process_time = time.time() - start_time
            
            total_time += process_time
            total_detections += len(result.boxes)
            frame_count += 1
            
            # Draw results on frame
            if not args.no_visualization:
                # Convert BGR to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                vis_img = engine.visualize_results(frame_rgb, result)
                vis_frame = cv2.cvtColor(np.array(vis_img), cv2.COLOR_RGB2BGR)
                
                # Add FPS text
                fps_text = f"FPS: {1/process_time:.1f}"
                cv2.putText(vis_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if writer:
                    writer.write(vis_frame)
                
                # Display frame (optional)
                if not output_path:  # Only display if not saving
                    cv2.imshow('Detection Results', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    # Print video processing statistics
    avg_time = total_time / frame_count if frame_count > 0 else 0
    avg_fps = 1 / avg_time if avg_time > 0 else 0
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    
    print(f"\nVideo Processing Results:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average time per frame: {avg_time*1000:.2f} ms")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per frame: {avg_detections:.1f}")


if __name__ == "__main__":
    exit(main())