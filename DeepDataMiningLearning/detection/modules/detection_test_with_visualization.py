#!/usr/bin/env python3
"""
Object Detection Test with Visualization
Compare our custom YOLOv8 model with official Ultralytics model on real images.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import time
from ultralytics import YOLO
from output_matcher import UltralyticsCompatibleModel

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def load_and_preprocess_image(image_path, target_size=(640, 640)):
    """Load and preprocess image for YOLO inference."""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image_rgb.shape[:2]  # H, W
    
    # Resize image while maintaining aspect ratio
    h, w = image_rgb.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image_rgb, (new_w, new_h))
    
    # Pad to target size
    padded = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Convert to tensor and normalize
    tensor = torch.from_numpy(padded).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # BHWC -> BCHW
    
    return tensor, image_rgb, original_shape, scale, (x_offset, y_offset)

def postprocess_detections(raw_outputs, original_shape, scale, offset, conf_threshold=0.5):
    """Post-process raw model outputs to get detection boxes."""
    
    if not isinstance(raw_outputs, (list, tuple)):
        raw_outputs = [raw_outputs]
    
    all_detections = []
    
    # Process each scale output
    for scale_idx, output in enumerate(raw_outputs):
        if output.dim() == 4:  # Feature map format [B, C, H, W]
            B, C, H, W = output.shape
            
            # Calculate stride for this scale
            stride = 8 * (2 ** scale_idx)  # 8, 16, 32 for the three scales
            
            # Create grid
            grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1).float().to(output.device)
            grid = grid.unsqueeze(0).unsqueeze(3)  # [1, H, W, 1, 2]
            
            # Reshape output to [B, H, W, anchors_per_cell, C]
            # For YOLOv8, typically 1 anchor per cell, so C should be 84 (4+1+79) or 144
            if C == 144:  # Our model format
                # Assume format: [4 box + 1 conf + 79 classes] * some_factor
                # Let's try to extract the first 84 channels
                output_reshaped = output[:, :84, :, :].permute(0, 2, 3, 1).contiguous()  # [B, H, W, 84]
                output_reshaped = output_reshaped.view(B, H, W, 1, 84)  # [B, H, W, 1, 84]
            elif C == 84:  # Standard format
                output_reshaped = output.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 84]
                output_reshaped = output_reshaped.view(B, H, W, 1, 84)  # [B, H, W, 1, 84]
            else:
                continue  # Skip this output if format is unexpected
            
            # Extract predictions
            box_preds = output_reshaped[..., :4]  # [B, H, W, 1, 4] - x, y, w, h
            conf_preds = output_reshaped[..., 4:5]  # [B, H, W, 1, 1] - confidence
            class_preds = output_reshaped[..., 5:]  # [B, H, W, 1, 79] - class probabilities
            
            # Apply sigmoid to confidence and class predictions
            conf_preds = torch.sigmoid(conf_preds)
            class_preds = torch.sigmoid(class_preds)
            
            # Convert box predictions to absolute coordinates
            # YOLOv8 uses a different box format
            box_preds[..., :2] = (box_preds[..., :2] * 2 - 0.5 + grid) * stride
            box_preds[..., 2:] = (box_preds[..., 2:] * 2) ** 2 * stride
            
            # Convert to corner format
            x_center, y_center = box_preds[..., 0], box_preds[..., 1]
            width, height = box_preds[..., 2], box_preds[..., 3]
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Get class predictions
            class_conf, class_pred = torch.max(class_preds, dim=-1)
            final_conf = conf_preds.squeeze(-1) * class_conf
            
            # Apply confidence threshold
            mask = final_conf > conf_threshold
            
            if mask.any():
                # Get valid predictions
                valid_x1 = x1[mask]
                valid_y1 = y1[mask]
                valid_x2 = x2[mask]
                valid_y2 = y2[mask]
                valid_conf = final_conf[mask]
                valid_class = class_pred[mask]
                
                # Scale back to original image coordinates
                x_offset, y_offset = offset
                
                # Adjust for padding offset and scale
                valid_x1 = (valid_x1 - x_offset) / scale
                valid_y1 = (valid_y1 - y_offset) / scale
                valid_x2 = (valid_x2 - x_offset) / scale
                valid_y2 = (valid_y2 - y_offset) / scale
                
                # Clip to image bounds
                valid_x1 = torch.clamp(valid_x1, 0, original_shape[1])
                valid_y1 = torch.clamp(valid_y1, 0, original_shape[0])
                valid_x2 = torch.clamp(valid_x2, 0, original_shape[1])
                valid_y2 = torch.clamp(valid_y2, 0, original_shape[0])
                
                # Filter out invalid boxes (too small or negative area)
                valid_width = valid_x2 - valid_x1
                valid_height = valid_y2 - valid_y1
                area_mask = (valid_width > 10) & (valid_height > 10)  # Minimum size filter
                
                if area_mask.any():
                    # Create detection results
                    for i in range(len(valid_x1)):
                        if area_mask[i] and valid_conf[i] > conf_threshold:
                            class_id = valid_class[i].item()
                            if class_id < len(COCO_CLASSES):
                                all_detections.append({
                                    'bbox': [valid_x1[i].item(), valid_y1[i].item(), 
                                           valid_x2[i].item(), valid_y2[i].item()],
                                    'confidence': valid_conf[i].item(),
                                    'class_id': class_id,
                                    'class_name': COCO_CLASSES[class_id]
                                })
    
    # Apply Non-Maximum Suppression
    if all_detections:
        all_detections = apply_nms(all_detections, iou_threshold=0.5)
    
    return all_detections

def apply_nms(detections, iou_threshold=0.45):
    """Apply Non-Maximum Suppression to remove duplicate detections."""
    
    if not detections:
        return detections
    
    # Convert to tensors for easier processing
    boxes = torch.tensor([det['bbox'] for det in detections])
    scores = torch.tensor([det['confidence'] for det in detections])
    
    # Apply NMS
    keep_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    
    # Return filtered detections
    filtered_detections = [detections[i] for i in keep_indices.tolist()]
    
    return filtered_detections

def visualize_detections(image, detections, title="Detections", save_path=None):
    """Visualize detection results with bounding boxes."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Colors for different classes
    colors = plt.cm.Set3(np.linspace(0, 1, len(COCO_CLASSES)))
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_name = det['class_name']
        class_id = det['class_id']
        
        # Draw bounding box
        width = x2 - x1
        height = y2 - y1
        
        color = colors[class_id % len(colors)]
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label = f'{class_name}: {conf:.2f}'
        ax.text(x1, y1-5, label, fontsize=10, color=color, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    return fig

def test_detection_on_image(image_path, conf_threshold=0.5):
    """Test object detection on a single image with both models."""
    
    print(f"\n{'='*80}")
    print(f"TESTING OBJECT DETECTION ON: {os.path.basename(image_path)}")
    print(f"{'='*80}")
    
    # Load and preprocess image
    print("Loading and preprocessing image...")
    tensor_input, original_image, original_shape, scale, offset = load_and_preprocess_image(image_path)
    print(f"Original image shape: {original_shape}")
    print(f"Preprocessed tensor shape: {tensor_input.shape}")
    
    # Load models
    print("\nLoading models...")
    
    # Official model
    print("Loading official Ultralytics model...")
    official_model = YOLO('yolov8n.pt')
    
    # Our custom model
    print("Loading our custom model...")
    yaml_path = '/Developer/DeepDataMiningLearning/DeepDataMiningLearning/detection/modules/ultralytics_yolov8.yaml'
    our_model = UltralyticsCompatibleModel(yaml_path, scale='n', ch=3, load_official_weights=True)
    
    # Run inference
    print("\nRunning inference...")
    
    # Official model inference
    print("Official model inference...")
    start_time = time.time()
    with torch.no_grad():
        official_results = official_model(tensor_input, verbose=False)
    official_time = time.time() - start_time
    
    # Our model inference
    print("Our model inference...")
    start_time = time.time()
    with torch.no_grad():
        our_raw_outputs = our_model(tensor_input)
    our_time = time.time() - start_time
    
    print(f"Official model inference time: {official_time:.4f}s")
    print(f"Our model inference time: {our_time:.4f}s")
    
    # Process results
    print("\nProcessing results...")
    
    # Official model results
    official_detections = []
    if len(official_results) > 0:
        result = official_results[0]
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                if confs[i] > conf_threshold:
                    official_detections.append({
                        'bbox': boxes[i].tolist(),
                        'confidence': float(confs[i]),
                        'class_id': int(classes[i]),
                        'class_name': COCO_CLASSES[classes[i]] if classes[i] < len(COCO_CLASSES) else f'class_{classes[i]}'
                    })
    
    # Our model results
    our_detections = postprocess_detections(our_raw_outputs, original_shape, scale, offset, conf_threshold)
    
    print(f"Official model detections: {len(official_detections)}")
    print(f"Our model detections: {len(our_detections)}")
    
    # Print detection details
    if official_detections:
        print("\nOfficial model detections:")
        for i, det in enumerate(official_detections):
            print(f"  {i+1}: {det['class_name']} ({det['confidence']:.3f})")
    
    if our_detections:
        print("\nOur model detections:")
        for i, det in enumerate(our_detections):
            print(f"  {i+1}: {det['class_name']} ({det['confidence']:.3f})")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Create output directory
    output_dir = '/Developer/DeepDataMiningLearning/DeepDataMiningLearning/detection/modules/detection_results'
    os.makedirs(output_dir, exist_ok=True)
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Visualize official model results
    fig1 = visualize_detections(
        original_image, 
        official_detections, 
        f"Official Ultralytics YOLOv8n - {len(official_detections)} detections",
        os.path.join(output_dir, f"{image_name}_official.png")
    )
    
    # Visualize our model results
    fig2 = visualize_detections(
        original_image, 
        our_detections, 
        f"Our Custom YOLOv8n - {len(our_detections)} detections",
        os.path.join(output_dir, f"{image_name}_custom.png")
    )
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Official results
    ax1.imshow(original_image)
    ax1.set_title(f"Official Model ({len(official_detections)} detections)", fontsize=14, fontweight='bold')
    colors = plt.cm.Set3(np.linspace(0, 1, len(COCO_CLASSES)))
    
    for det in official_detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_name = det['class_name']
        class_id = det['class_id']
        
        width = x2 - x1
        height = y2 - y1
        color = colors[class_id % len(colors)]
        
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        
        label = f'{class_name}: {conf:.2f}'
        ax1.text(x1, y1-5, label, fontsize=9, color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax1.axis('off')
    
    # Our results
    ax2.imshow(original_image)
    ax2.set_title(f"Our Model ({len(our_detections)} detections)", fontsize=14, fontweight='bold')
    
    for det in our_detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_name = det['class_name']
        class_id = det['class_id']
        
        width = x2 - x1
        height = y2 - y1
        color = colors[class_id % len(colors)]
        
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        
        label = f'{class_name}: {conf:.2f}'
        ax2.text(x1, y1-5, label, fontsize=9, color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax2.axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f"{image_name}_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Comparison saved to: {comparison_path}")
    
    plt.show()
    
    return {
        'official_detections': official_detections,
        'our_detections': our_detections,
        'official_time': official_time,
        'our_time': our_time,
        'image_shape': original_shape
    }

def main():
    """Main function to test detection on multiple sample images."""
    
    print("🎯 YOLO Object Detection Test with Visualization")
    print("=" * 80)
    
    # Sample images to test
    sample_images = [
        '/Developer/DeepDataMiningLearning/sampledata/bus.jpg',
        '/Developer/DeepDataMiningLearning/sampledata/sjsupeople.jpg',
        '/Developer/DeepDataMiningLearning/sampledata/sjsuimag1.jpg'
    ]
    
    results = {}
    
    for image_path in sample_images:
        if os.path.exists(image_path):
            try:
                result = test_detection_on_image(image_path, conf_threshold=0.5)
                results[os.path.basename(image_path)] = result
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  Image not found: {image_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("DETECTION TEST SUMMARY")
    print(f"{'='*80}")
    
    total_official_detections = 0
    total_our_detections = 0
    total_official_time = 0
    total_our_time = 0
    
    for image_name, result in results.items():
        official_count = len(result['official_detections'])
        our_count = len(result['our_detections'])
        
        total_official_detections += official_count
        total_our_detections += our_count
        total_official_time += result['official_time']
        total_our_time += result['our_time']
        
        print(f"{image_name}:")
        print(f"  Official: {official_count} detections ({result['official_time']:.4f}s)")
        print(f"  Our model: {our_count} detections ({result['our_time']:.4f}s)")
        print(f"  Shape: {result['image_shape']}")
    
    if results:
        avg_official_time = total_official_time / len(results)
        avg_our_time = total_our_time / len(results)
        
        print(f"\nOverall Statistics:")
        print(f"  Total images processed: {len(results)}")
        print(f"  Total official detections: {total_official_detections}")
        print(f"  Total our detections: {total_our_detections}")
        print(f"  Average official inference time: {avg_official_time:.4f}s")
        print(f"  Average our inference time: {avg_our_time:.4f}s")
        print(f"  Speed ratio: {avg_our_time/avg_official_time:.2f}x")
        
        print(f"\n✅ Detection test completed successfully!")
        print(f"📁 Results saved in: /Developer/DeepDataMiningLearning/DeepDataMiningLearning/detection/modules/detection_results/")
    
    return results

if __name__ == "__main__":
    main()