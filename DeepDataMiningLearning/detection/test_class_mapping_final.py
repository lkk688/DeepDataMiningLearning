#!/usr/bin/env python3
"""
Final test script to verify YOLO to torchvision class mapping functionality.
This test creates proper mock UltralyticsResult objects.
"""

import torch
from .modeling_yolo import TorchvisionYoloModel

class MockBoxes:
    """Mock boxes object to simulate UltralyticsResult.boxes"""
    def __init__(self, boxes, scores, labels):
        # Create detection data in format [x1, y1, x2, y2, conf, class]
        self.data = torch.cat([
            boxes,  # [N, 4] - bounding boxes
            scores.unsqueeze(1),  # [N, 1] - confidence scores
            labels.unsqueeze(1).float()  # [N, 1] - class labels
        ], dim=1)  # Result: [N, 6]

class MockUltralyticsResult:
    """Mock UltralyticsResult object"""
    def __init__(self, boxes, scores, labels):
        self.boxes = MockBoxes(boxes, scores, labels)

def test_class_mapping_with_mock_data():
    """Test the class mapping functionality with proper mock data."""
    
    print("Testing YOLO to torchvision class mapping with mock UltralyticsResult...")
    print("=" * 70)
    
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
    
    # Create mock detection data
    mock_boxes = torch.tensor([
        [100.0, 100.0, 200.0, 200.0],  # person detection
        [300.0, 300.0, 400.0, 400.0],  # bus detection
        [50.0, 50.0, 150.0, 150.0]     # car detection
    ])
    mock_scores = torch.tensor([0.9, 0.8, 0.85])
    mock_labels = torch.tensor([0, 5, 2])  # person (0), bus (5), car (2)
    mock_original_shapes = [(480, 640)]  # Original image shape
    
    # Create mock UltralyticsResult
    mock_result = MockUltralyticsResult(mock_boxes, mock_scores, mock_labels)
    mock_detections = [mock_result]
    
    print(f"\n📋 Mock detection data:")
    print(f"   Boxes: {mock_boxes.tolist()}")
    print(f"   Scores: {mock_scores.tolist()}")
    print(f"   YOLO Labels: {mock_labels.tolist()} (person, bus, car)")
    
    # Test without mapping
    print(f"\n🧪 Testing WITHOUT class mapping:")
    results_no_mapping = model_no_mapping._convert_detections_to_torchvision(
        mock_detections, 
        mock_original_shapes, 
        map_to_torchvision_classes=False
    )
    
    print(f"   Labels (YOLO): {results_no_mapping[0]['labels'].tolist()}")
    print(f"   Scores: {results_no_mapping[0]['scores'].tolist()}")
    print(f"   Boxes: {results_no_mapping[0]['boxes'].tolist()}")
    
    # Test with mapping
    print(f"\n🧪 Testing WITH class mapping:")
    results_with_mapping = model_with_mapping._convert_detections_to_torchvision(
        mock_detections, 
        mock_original_shapes, 
        map_to_torchvision_classes=True
    )
    
    print(f"   Labels (torchvision): {results_with_mapping[0]['labels'].tolist()}")
    print(f"   Scores: {results_with_mapping[0]['scores'].tolist()}")
    print(f"   Boxes: {results_with_mapping[0]['boxes'].tolist()}")
    
    # Verify the mapping
    print(f"\n📊 Mapping verification:")
    yolo_labels = results_no_mapping[0]['labels'].tolist()
    torchvision_labels = results_with_mapping[0]['labels'].tolist()
    
    # Expected mappings: 0->1 (person), 5->6 (bus), 2->3 (car)
    expected_mappings = {0: 1, 5: 6, 2: 3}
    
    mapping_correct = True
    for i, (yolo_label, tv_label) in enumerate(zip(yolo_labels, torchvision_labels)):
        expected_tv_label = expected_mappings.get(yolo_label, yolo_label)
        status = "✅" if tv_label == expected_tv_label else "❌"
        class_names = ["person", "bus", "car"]
        print(f"   {status} Detection {i+1} ({class_names[i]}): YOLO {yolo_label} -> torchvision {tv_label} (expected {expected_tv_label})")
        if tv_label != expected_tv_label:
            mapping_correct = False
    
    # Final verification
    print(f"\n🎯 Final Results:")
    if mapping_correct and len(yolo_labels) == 3 and len(torchvision_labels) == 3:
        print("   ✅ All class mappings are working correctly!")
        print("   ✅ Both modes (with/without mapping) produce expected results")
        print("   ✅ Users can successfully switch between YOLO and torchvision class indices")
    else:
        print("   ❌ Some mappings failed or unexpected results occurred")
    
    print("\n" + "=" * 70)
    print("🎉 Class mapping functionality test completed!")
    
    return mapping_correct

if __name__ == "__main__":
    success = test_class_mapping_with_mock_data()
    if success:
        print("\n✅ ALL TESTS PASSED - Class mapping feature is working correctly!")
    else:
        print("\n❌ SOME TESTS FAILED - Please check the implementation!")