"""
Hyperparameters configuration for YOLO models.
This module provides default hyperparameters compatible with Ultralytics YOLO models.
"""

class SimpleNamespace:
    """Simple namespace class to hold hyperparameters."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Default hyperparameters for YOLOv8 training
DEFAULT_HYPERPARAMETERS = SimpleNamespace(
    # Loss gains
    box=7.5,      # box loss gain
    cls=0.5,      # cls loss gain  
    dfl=1.5,      # dfl loss gain
    
    # Training hyperparameters
    lr0=0.01,     # initial learning rate
    lrf=0.01,     # final learning rate factor
    momentum=0.937,  # SGD momentum
    weight_decay=0.0005,  # optimizer weight decay
    warmup_epochs=3.0,   # warmup epochs
    warmup_momentum=0.8, # warmup initial momentum
    warmup_bias_lr=0.1,  # warmup initial bias lr
    
    # Data augmentation
    hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s=0.7,    # image HSV-Saturation augmentation (fraction)
    hsv_v=0.4,    # image HSV-Value augmentation (fraction)
    degrees=0.0,  # image rotation (+/- deg)
    translate=0.1, # image translation (+/- fraction)
    scale=0.5,    # image scale (+/- gain)
    shear=0.0,    # image shear (+/- deg)
    perspective=0.0, # image perspective (+/- fraction), range 0-0.001
    flipud=0.0,   # image flip up-down (probability)
    fliplr=0.5,   # image flip left-right (probability)
    mosaic=1.0,   # image mosaic (probability)
    mixup=0.0,    # image mixup (probability)
    copy_paste=0.0, # segment copy-paste (probability)
    
    # Other training parameters
    anchor_t=4.0, # anchor-multiple threshold
    fl_gamma=0.0, # focal loss gamma (efficientDet default gamma=1.5)
    label_smoothing=0.0, # label smoothing epsilon
    nbs=64,       # nominal batch size
    overlap_mask=True, # masks should overlap during training (segment train only)
    mask_ratio=4, # mask downsample ratio (segment train only)
    dropout=0.0,  # use dropout regularization (classify train only)
    val=True,     # validate/test during training
)

def get_hyperparameters():
    """Get default hyperparameters for YOLO training."""
    return DEFAULT_HYPERPARAMETERS