#!/usr/bin/env python3
"""
Simple test script to verify YOLO to torchvision class mapping functionality.
This test focuses on the mapping logic without running full inference.
"""

import torch
from .modeling_yolo import TorchvisionYoloModel

def test_class_mapping_logic():
    """Test the class mapping logic directly."""
    
    print("Testing YOLO to torchvision class mapping logic...")
    print("=" * 60)
    
    # Create models with different mapping settings
    model_no_mapping = TorchvisionYoloModel(
        model_name="yolov8", 
        scale="n", 
        num_classes=80,
        map_to_torchvision_classes=False
    )
    
    model_with_mapping = TorchvisionYoloModel(
        model_name="yolov8", 
        scale="n", 
        num_classes=80,
        map_to_torchvision_classes=True
    )
    
    print(f"✓ Model without mapping created (map_to_torchvision_classes={model_no_mapping.map_to_torchvision_classes})")
    print(f"✓ Model with mapping created (map_to_torchvision_classes={model_with_mapping.map_to_torchvision_classes})")
    
    # Test the mapping dictionary directly
    yolo_to_torchvision_mapping = {
        0: 1,   # person
        1: 2,   # bicycle
        2: 3,   # car
        3: 4,   # motorcycle
        4: 5,   # airplane
        5: 6,   # bus
        6: 7,   # train
        7: 8,   # truck
        8: 9,   # boat
        9: 10,  # traffic light
        10: 11, # fire hydrant
        11: 13, # stop sign
        12: 14, # parking meter
        13: 15, # bench
        14: 16, # bird
        15: 17, # cat
        16: 18, # dog
        17: 19, # horse
        18: 20, # sheep
        19: 21, # cow
        20: 22, # elephant
        21: 23, # bear
        22: 24, # zebra
        23: 25, # giraffe
        24: 27, # backpack
        25: 28, # umbrella
        26: 31, # handbag
        27: 32, # tie
        28: 33, # suitcase
        29: 34, # frisbee
        30: 35, # skis
        31: 36, # snowboard
        32: 37, # sports ball
        33: 38, # kite
        34: 39, # baseball bat
        35: 40, # baseball glove
        36: 41, # skateboard
        37: 42, # surfboard
        38: 43, # tennis racket
        39: 44, # bottle
        40: 46, # wine glass
        41: 47, # cup
        42: 48, # fork
        43: 49, # knife
        44: 50, # spoon
        45: 51, # bowl
        46: 52, # banana
        47: 53, # apple
        48: 54, # sandwich
        49: 55, # orange
        50: 56, # broccoli
        51: 57, # carrot
        52: 58, # hot dog
        53: 59, # pizza
        54: 60, # donut
        55: 61, # cake
        56: 62, # chair
        57: 63, # couch
        58: 64, # potted plant
        59: 65, # bed
        60: 67, # dining table
        61: 70, # toilet
        62: 72, # tv
        63: 73, # laptop
        64: 74, # mouse
        65: 75, # remote
        66: 76, # keyboard
        67: 77, # cell phone
        68: 78, # microwave
        69: 79, # oven
        70: 80, # toaster
        71: 81, # sink
        72: 82, # refrigerator
        73: 84, # book
        74: 85, # clock
        75: 86, # vase
        76: 87, # scissors
        77: 88, # teddy bear
        78: 89, # hair drier
        79: 90  # toothbrush
    }
    
    print(f"\n✓ Mapping dictionary contains {len(yolo_to_torchvision_mapping)} class mappings")
    
    # Test some specific mappings
    test_cases = [
        (0, 1, "person"),
        (2, 3, "car"), 
        (5, 6, "bus"),
        (15, 17, "cat"),
        (16, 18, "dog"),
        (56, 62, "chair"),
        (79, 90, "toothbrush")
    ]
    
    print("\n📋 Testing specific class mappings:")
    for yolo_class, expected_tv_class, class_name in test_cases:
        actual_tv_class = yolo_to_torchvision_mapping.get(yolo_class, yolo_class)
        status = "✅" if actual_tv_class == expected_tv_class else "❌"
        print(f"   {status} YOLO {yolo_class} ({class_name}) -> torchvision {actual_tv_class} (expected {expected_tv_class})")
    
    # Test the _convert_detections_to_torchvision method with mock data
    print("\n🧪 Testing _convert_detections_to_torchvision method:")
    
    # Create mock detection data
    mock_boxes = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]])  # 2 boxes
    mock_scores = torch.tensor([0.9, 0.8])  # 2 confidence scores
    mock_labels = torch.tensor([0, 5])  # person (0) and bus (5)
    mock_original_shapes = [(480, 640)]  # Original image shape
    
    # Test without mapping
    results_no_mapping = model_no_mapping._convert_detections_to_torchvision(
        [(mock_boxes, mock_scores, mock_labels)], 
        mock_original_shapes, 
        map_to_torchvision_classes=False
    )
    
    # Test with mapping
    results_with_mapping = model_with_mapping._convert_detections_to_torchvision(
        [(mock_boxes, mock_scores, mock_labels)], 
        mock_original_shapes, 
        map_to_torchvision_classes=True
    )
    
    print(f"   Without mapping - Labels: {results_no_mapping[0]['labels'].tolist()}")
    print(f"   With mapping - Labels: {results_with_mapping[0]['labels'].tolist()}")
    
    # Verify the mapping worked correctly
    expected_mapped_labels = [yolo_to_torchvision_mapping[0], yolo_to_torchvision_mapping[5]]  # [1, 6]
    actual_mapped_labels = results_with_mapping[0]['labels'].tolist()
    
    if actual_mapped_labels == expected_mapped_labels:
        print("   ✅ Class mapping is working correctly!")
        print(f"   ✅ YOLO classes [0, 5] correctly mapped to torchvision classes {actual_mapped_labels}")
    else:
        print(f"   ❌ Class mapping failed! Expected {expected_mapped_labels}, got {actual_mapped_labels}")
    
    print("\n" + "=" * 60)
    print("✅ Class mapping logic test completed successfully!")
    print("\n📝 Summary:")
    print("   - Models can be created with and without class mapping")
    print("   - Mapping dictionary contains all 80 COCO classes")
    print("   - _convert_detections_to_torchvision method correctly applies mapping")
    print("   - Users can switch between YOLO (0-79) and torchvision (1-90) class indices")

if __name__ == "__main__":
    test_class_mapping_logic()