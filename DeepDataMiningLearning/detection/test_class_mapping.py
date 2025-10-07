#!/usr/bin/env python3
"""
Test script to verify YOLO to torchvision class mapping functionality.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import the TorchvisionYoloModel
from .modeling_yolo import TorchvisionYoloModel

def test_class_mapping():
    """Test the class mapping functionality."""
    
    # Load a test image
    image_path = "../../sampledata/bus.jpg"
    image = Image.open(image_path).convert('RGB')
    
    # Convert PIL image to tensor for the model
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    
    print("Testing YOLO to torchvision class mapping...")
    print("=" * 50)
    
    # Test 1: Without class mapping (default YOLO classes 0-79)
    print("\n1. Testing WITHOUT class mapping (YOLO classes 0-79):")
    model_no_mapping = TorchvisionYoloModel(
        model_name="yolov8", 
        scale="n", 
        num_classes=80,
        map_to_torchvision_classes=False
    )
    
    # Run inference
    results_no_mapping = model_no_mapping([image_tensor])
    
    print(f"   Number of detections: {len(results_no_mapping[0]['boxes'])}")
    if len(results_no_mapping[0]['boxes']) > 0:
        print(f"   Class labels (YOLO): {results_no_mapping[0]['labels'].tolist()}")
        print(f"   Confidence scores: {results_no_mapping[0]['scores'].tolist()}")
    
    # Test 2: With class mapping (torchvision classes 1-90)
    print("\n2. Testing WITH class mapping (torchvision classes 1-90):")
    model_with_mapping = TorchvisionYoloModel(
        model_name="yolov8", 
        scale="n", 
        num_classes=80,
        map_to_torchvision_classes=True
    )
    
    # Run inference
    results_with_mapping = model_with_mapping([image_tensor])
    
    print(f"   Number of detections: {len(results_with_mapping[0]['boxes'])}")
    if len(results_with_mapping[0]['boxes']) > 0:
        print(f"   Class labels (torchvision): {results_with_mapping[0]['labels'].tolist()}")
        print(f"   Confidence scores: {results_with_mapping[0]['scores'].tolist()}")
    
    # Compare results
    print("\n3. Comparison:")
    if len(results_no_mapping[0]['boxes']) > 0 and len(results_with_mapping[0]['boxes']) > 0:
        yolo_labels = results_no_mapping[0]['labels'].tolist()
        torchvision_labels = results_with_mapping[0]['labels'].tolist()
        
        print("   YOLO -> Torchvision mapping:")
        for i, (yolo_label, tv_label) in enumerate(zip(yolo_labels, torchvision_labels)):
            print(f"     Detection {i+1}: {yolo_label} -> {tv_label}")
        
        # Verify mapping is correct
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
        
        mapping_correct = True
        for yolo_label, tv_label in zip(yolo_labels, torchvision_labels):
            expected_tv_label = yolo_to_torchvision_mapping.get(yolo_label, yolo_label)
            if tv_label != expected_tv_label:
                print(f"   ❌ Mapping error: YOLO {yolo_label} -> {tv_label}, expected {expected_tv_label}")
                mapping_correct = False
        
        if mapping_correct:
            print("   ✅ All class mappings are correct!")
        
    print("\n" + "=" * 50)
    print("Class mapping test completed!")

if __name__ == "__main__":
    test_class_mapping()