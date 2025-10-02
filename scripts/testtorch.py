import torch
print(f"PyTorch version: {torch.__version__}")

# Check CUDA availability with error handling
cuda_available = False
try:
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
except Exception as e:
    print(f"CUDA availability check failed: {e}")

# setting device on GPU if available, else CPU
device = torch.device('cuda' if cuda_available else 'cpu')
print(f'Using device: {device}')

# Additional Info when using cuda
if device.type == 'cuda' and cuda_available:
    try:
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device properties: {torch.cuda.get_device_properties(0)}")
        print(f"Utilization: {torch.cuda.utilization(0)}")
        print('Memory Usage:')
        print(f'Memory usage: {round(torch.cuda.memory_usage(0)/1024**3,1)} GB')
        print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
        print(f'Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB')
    except Exception as e:
        print(f"Error getting CUDA device info: {e}")

# Safe CUDA device count check
try:
    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")
except Exception as e:
    print(f"Error getting device count: {e}")
    device_count = 0

# Safe current device check
if cuda_available and device_count > 0:
    try:
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device}")
        print(f"CUDA device object: {torch.cuda.device(0)}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Error accessing CUDA device: {e}")

# Safe tensor creation test
try:
    if cuda_available and device_count > 0:
        test_tensor = torch.rand(2,3).cuda()
        print(f"✅ CUDA tensor creation successful: {test_tensor}")
    else:
        test_tensor = torch.rand(2,3)
        print(f"CPU tensor creation: {test_tensor}")
except Exception as e:
    print(f"❌ Tensor creation failed: {e}")
    # Fallback to CPU
    test_tensor = torch.rand(2,3)
    print(f"Fallback CPU tensor: {test_tensor}")

# TensorRT check with error handling
try:
    import tensorrt
    print(f"✅ TensorRT version: {tensorrt.__version__}")
    # Test TensorRT builder creation
    try:
        logger = tensorrt.Logger(tensorrt.Logger.WARNING)
        builder = tensorrt.Builder(logger)
        print("✅ TensorRT Builder created successfully")
    except Exception as e:
        print(f"❌ TensorRT Builder creation failed: {e}")
except ImportError:
    print("❌ TensorRT not installed")
except Exception as e:
    print(f"❌ TensorRT import failed: {e}")

print("\n=== Summary ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {cuda_available}")
print(f"Device: {device}")
if cuda_available and device_count > 0:
    print("✅ GPU acceleration available")
else:
    print("⚠️  Using CPU only - GPU acceleration not available")