# YOLO Model Configurations
# This file contains all YOLO model configurations as Python dictionaries
# to replace the YAML configuration files

# Default configuration settings
DEFAULT_CFG = {
    'task': 'detect',
    'mode': 'train',
    'epochs': 100,
    'patience': 50,
    'batch': 16,
    'imgsz': 640,
    'save': True,
    'save_period': -1,
    'cache': False,
    'device': None,
    'workers': 8,
    'project': None,
    'name': None,
    'exist_ok': False,
    'pretrained': True,
    'optimizer': 'auto',
    'verbose': True,
    'seed': 0,
    'deterministic': True,
    'single_cls': False,
    'rect': False,
    'cos_lr': False,
    'close_mosaic': 10,
    'resume': False,
    'amp': True,
    'fraction': 1.0,
    'profile': False,
    'freeze': None,
    'overlap_mask': True,
    'mask_ratio': 4,
    'dropout': 0.0,
    'val': True,
    'split': 'val',
    'save_json': False,
    'save_hybrid': False,
    'conf': None,
    'iou': 0.7,
    'max_det': 300,
    'half': False,
    'dnn': False,
    'plots': True,
    'agnostic_nms': False,
    'classes': None,
    'names': {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
        27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
        61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
        67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
        72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
        77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }
}

# YOLOv8 Model Configuration
YOLOV8_CFG = {
    'model_name': 'yolov8',
    'nc': 80,  # number of classes
    'scales': {
        # [depth, width, max_channels]
        'n': [0.33, 0.25, 1024],  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
        's': [0.33, 0.50, 1024],  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
        'm': [0.67, 0.75, 768],   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
        'l': [1.00, 1.00, 512],   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
        'x': [1.00, 1.25, 512]    # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
    },
    'backbone': [
        # [from, repeats, module, args]
        [-1, 1, 'Conv', [64, 3, 2]],      # 0-P1/2
        [-1, 1, 'Conv', [128, 3, 2]],     # 1-P2/4
        [-1, 3, 'C2f', [128, True]],
        [-1, 1, 'Conv', [256, 3, 2]],     # 3-P3/8
        [-1, 6, 'C2f', [256, True]],
        [-1, 1, 'Conv', [512, 3, 2]],     # 5-P4/16
        [-1, 6, 'C2f', [512, True]],
        [-1, 1, 'Conv', [1024, 3, 2]],    # 7-P5/32
        [-1, 3, 'C2f', [1024, True]],
        [-1, 1, 'SPPF', [1024, 5]]        # 9
    ],
    'head': [
        [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 6], 1, 'Concat', [1]],      # cat backbone P4
        [-1, 3, 'C2f', [512]],            # 12
        
        [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 4], 1, 'Concat', [1]],      # cat backbone P3
        [-1, 3, 'C2f', [256]],            # 15 (P3/8-small)
        
        [-1, 1, 'Conv', [256, 3, 2]],
        [[-1, 12], 1, 'Concat', [1]],     # cat head P4
        [-1, 3, 'C2f', [512]],            # 18 (P4/16-medium)
        
        [-1, 1, 'Conv', [512, 3, 2]],
        [[-1, 9], 1, 'Concat', [1]],      # cat head P5
        [-1, 3, 'C2f', [1024]],           # 21 (P5/32-large)
        
        [[15, 18, 21], 1, 'Detect', ['nc']]  # Detect(P3, P4, P5)
    ]
}

# YOLOv5s Model Configuration
YOLOV5S_CFG = {
    'model_name': 'yolov5s',
    'nc': 80,  # number of classes
    'depth_multiple': 0.33,  # model depth multiple
    'width_multiple': 0.50,  # layer channel multiple
    'anchors': [
        [10, 13, 16, 30, 33, 23],      # P3/8
        [30, 61, 62, 45, 59, 119],     # P4/16
        [116, 90, 156, 198, 373, 326]  # P5/32
    ],
    'backbone': [
        # [from, number, module, args]
        [-1, 1, 'Conv', [64, 6, 2, 2]],   # 0-P1/2
        [-1, 1, 'Conv', [128, 3, 2]],     # 1-P2/4
        [-1, 3, 'C3', [128]],
        [-1, 1, 'Conv', [256, 3, 2]],     # 3-P3/8
        [-1, 6, 'C3', [256]],
        [-1, 1, 'Conv', [512, 3, 2]],     # 5-P4/16
        [-1, 9, 'C3', [512]],
        [-1, 1, 'Conv', [1024, 3, 2]],    # 7-P5/32
        [-1, 3, 'C3', [1024]],
        [-1, 1, 'SPPF', [1024, 5]]        # 9
    ],
    'head': [
        [-1, 1, 'Conv', [512, 1, 1]],
        [-1, 1, 'nn.Upsample', [None, 2, "nearest"]],
        [[-1, 6], 1, 'Concat', [1]],      # cat backbone P4
        [-1, 3, 'C3', [512, False]],      # 13
        
        [-1, 1, 'Conv', [256, 1, 1]],
        [-1, 1, 'nn.Upsample', [None, 2, "nearest"]],
        [[-1, 4], 1, 'Concat', [1]],      # cat backbone P3
        [-1, 3, 'C3', [256, False]],      # 17 (P3/8-small)
        
        [-1, 1, 'Conv', [256, 3, 2]],
        [[-1, 14], 1, 'Concat', [1]],     # cat head P4
        [-1, 3, 'C3', [512, False]],      # 20 (P4/16-medium)
        
        [-1, 1, 'Conv', [512, 3, 2]],
        [[-1, 10], 1, 'Concat', [1]],     # cat head P5
        [-1, 3, 'C3', [1024, False]],     # 23 (P5/32-large)
        
        [[17, 20, 23], 1, 'Detect', ['nc', 'anchors']]  # Detect(P3, P4, P5)
    ]
}

# YOLOv5l Model Configuration (similar to v5s but with different scaling)
YOLOV5L_CFG = {
    'model_name': 'yolov5l',
    'nc': 80,  # number of classes
    'depth_multiple': 1.0,   # model depth multiple
    'width_multiple': 1.0,   # layer channel multiple
    'anchors': [
        [10, 13, 16, 30, 33, 23],      # P3/8
        [30, 61, 62, 45, 59, 119],     # P4/16
        [116, 90, 156, 198, 373, 326]  # P5/32
    ],
    'backbone': [
        # [from, number, module, args]
        [-1, 1, 'Conv', [64, 6, 2, 2]],   # 0-P1/2
        [-1, 1, 'Conv', [128, 3, 2]],     # 1-P2/4
        [-1, 3, 'C3', [128]],
        [-1, 1, 'Conv', [256, 3, 2]],     # 3-P3/8
        [-1, 9, 'C3', [256]],
        [-1, 1, 'Conv', [512, 3, 2]],     # 5-P4/16
        [-1, 9, 'C3', [512]],
        [-1, 1, 'Conv', [1024, 3, 2]],    # 7-P5/32
        [-1, 3, 'C3', [1024]],
        [-1, 1, 'SPPF', [1024, 5]]        # 9
    ],
    'head': [
        [-1, 1, 'Conv', [512, 1, 1]],
        [-1, 1, 'nn.Upsample', [None, 2, "nearest"]],
        [[-1, 6], 1, 'Concat', [1]],      # cat backbone P4
        [-1, 3, 'C3', [512, False]],      # 13
        
        [-1, 1, 'Conv', [256, 1, 1]],
        [-1, 1, 'nn.Upsample', [None, 2, "nearest"]],
        [[-1, 4], 1, 'Concat', [1]],      # cat backbone P3
        [-1, 3, 'C3', [256, False]],      # 17 (P3/8-small)
        
        [-1, 1, 'Conv', [256, 3, 2]],
        [[-1, 14], 1, 'Concat', [1]],     # cat head P4
        [-1, 3, 'C3', [512, False]],      # 20 (P4/16-medium)
        
        [-1, 1, 'Conv', [512, 3, 2]],
        [[-1, 10], 1, 'Concat', [1]],     # cat head P5
        [-1, 3, 'C3', [1024, False]],     # 23 (P5/32-large)
        
        [[17, 20, 23], 1, 'Detect', ['nc', 'anchors']]  # Detect(P3, P4, P5)
    ]
}

# YOLOv11 Model Configuration with C3K2 and C2PSA blocks
YOLOV11_CFG = {
    'model_name': 'yolov11',
    'nc': 80,  # number of classes
    'depth_multiple': 0.50,  # model depth multiple
    'width_multiple': 0.25,  # layer channel multiple
    'max_channels': 1024,
    'scales': {
        'n': [0.50, 0.25, 1024],   # YOLOv11n summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
        's': [0.50, 0.50, 1024],   # YOLOv11s summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
        'm': [0.50, 0.75, 576],    # YOLOv11m summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
        'l': [1.00, 1.00, 512],    # YOLOv11l summary: 631 layers, 253717504 parameters, 25371488 gradients, 86.9 GFLOPs
        'x': [1.00, 1.25, 512]     # YOLOv11x summary: 631 layers, 56966176 parameters, 56966160 gradients, 194.9 GFLOPs
    },
    'backbone': [
        # [from, repeats, module, args]
        [-1, 1, 'Conv', [64, 3, 2]],      # 0-P1/2
        [-1, 1, 'Conv', [128, 3, 2]],     # 1-P2/4
        [-1, 2, 'C3K2', [256, False, 0.25]],  # 2
        [-1, 1, 'Conv', [256, 3, 2]],     # 3-P3/8
        [-1, 2, 'C3K2', [512, False, 0.25]],  # 4
        [-1, 1, 'Conv', [512, 3, 2]],     # 5-P4/16
        [-1, 2, 'C3K2', [512, True]],     # 6
        [-1, 1, 'Conv', [1024, 3, 2]],    # 7-P5/32
        [-1, 2, 'C3K2', [1024, True]],    # 8
        [-1, 1, 'SPPF', [1024, 5]]        # 9
    ],
    'head': [
        [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 6], 1, 'Concat', [1]],      # cat backbone P4
        [-1, 2, 'C3K2', [512, False]],    # 12
        
        [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 4], 1, 'Concat', [1]],      # cat backbone P3
        [-1, 2, 'C3K2', [256, False]],    # 15 (P3/8-small)
        
        [-1, 1, 'Conv', [256, 3, 2]],
        [[-1, 12], 1, 'Concat', [1]],     # cat head P4
        [-1, 2, 'C3K2', [512, False]],    # 18 (P4/16-medium)
        
        [-1, 1, 'Conv', [512, 3, 2]],
        [[-1, 9], 1, 'Concat', [1]],      # cat head P5
        [-1, 2, 'C3K2', [1024, False]],   # 21 (P5/32-large)
        
        [[15, 18, 21], 1, 'Detect', ['nc']]  # Detect(P3, P4, P5)
    ]
}

# YOLOv12 Model Configuration with Area Attention (A2) and R-ELAN
YOLOV12_CFG = {
    'model_name': 'yolov12',
    'nc': 80,  # number of classes
    'depth_multiple': 0.50,  # model depth multiple
    'width_multiple': 0.25,  # layer channel multiple
    'max_channels': 1024,
    'attention_enabled': True,
    'flash_attention': True,
    'mlp_ratio': 1.2,  # Optimized MLP ratio for YOLOv12
    'scales': {
        'n': [0.50, 0.25, 1024],   # YOLOv12n
        's': [0.50, 0.50, 1024],   # YOLOv12s
        'm': [0.50, 0.75, 576],    # YOLOv12m
        'l': [1.00, 1.00, 512],    # YOLOv12l
        'x': [1.00, 1.25, 512]     # YOLOv12x
    },
    'backbone': [
        # [from, repeats, module, args]
        [-1, 1, 'Conv', [64, 3, 2]],      # 0-P1/2
        [-1, 1, 'Conv', [128, 3, 2]],     # 1-P2/4
        [-1, 2, 'R-ELAN', [256, False, 0.25]],  # 2 - Residual ELAN
        [-1, 1, 'Conv', [256, 3, 2]],     # 3-P3/8
        [-1, 2, 'R-ELAN', [512, False, 0.25]],  # 4
        [-1, 1, 'Conv', [512, 3, 2]],     # 5-P4/16
        [-1, 2, 'R-ELAN', [512, True]],   # 6
        [-1, 1, 'Conv', [1024, 3, 2]],    # 7-P5/32
        [-1, 2, 'R-ELAN', [1024, True]],  # 8
        [-1, 1, 'A2', [1024, 5]]          # 9 - Area Attention
    ],
    'head': [
        [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 6], 1, 'Concat', [1]],      # cat backbone P4
        [-1, 2, 'R-ELAN', [512, False]],  # 12
        
        [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 4], 1, 'Concat', [1]],      # cat backbone P3
        [-1, 2, 'R-ELAN', [256, False]],  # 15 (P3/8-small)
        
        [-1, 1, 'Conv', [256, 3, 2]],
        [[-1, 12], 1, 'Concat', [1]],     # cat head P4
        [-1, 2, 'R-ELAN', [512, False]],  # 18 (P4/16-medium)
        
        [-1, 1, 'Conv', [512, 3, 2]],
        [[-1, 9], 1, 'Concat', [1]],      # cat head P5
        [-1, 2, 'R-ELAN', [1024, False]], # 21 (P5/32-large)
        
        [[15, 18, 21], 1, 'Detect', ['nc']]  # Detect(P3, P4, P5)
    ]
}

# Model configuration mapping
MODEL_CONFIGS = {
    'yolov8': YOLOV8_CFG,
    'yolov11': YOLOV11_CFG,
    'yolov12': YOLOV12_CFG,
    'yolov5s': YOLOV5S_CFG,
    'yolov5l': YOLOV5L_CFG,
    'default': DEFAULT_CFG
}

def get_model_config(model_name):
    """
    Get model configuration by name
    
    Args:
        model_name (str): Name of the model (e.g., 'yolov8', 'yolov5s', 'yolov5l')
    
    Returns:
        dict: Model configuration dictionary
    """
    return MODEL_CONFIGS.get(model_name, None)

def get_default_config():
    """
    Get default configuration
    
    Returns:
        dict: Default configuration dictionary
    """
    return DEFAULT_CFG.copy()