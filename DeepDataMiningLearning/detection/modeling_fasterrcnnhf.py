import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from transformers import PreTrainedModel, PretrainedConfig
from huggingface_hub import HfApi, create_repo, notebook_login
import os
import onnx #pip install onnx onnxruntime-gpu
import onnxruntime
from PIL import Image
import numpy as np
from typing import List, Dict


COCO_INSTANCE_CATEGORY_NAMES = [
                '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]

# ======================
# 1. Custom Config Class
# ======================
class FasterRCNNConfig(PretrainedConfig):
    """
    Configuration class for FasterRCNN model.
    """
    model_type = "fasterrcnn"
    
    def __init__(
        self,
        num_classes=91,  # COCO has 91 classes (including background)
        backbone_type="resnet50_fpn_v2",
        min_size=800,
        max_size=1333,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        use_pretrained_model=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.backbone_type = backbone_type
        self.min_size = min_size
        self.max_size = max_size
        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.box_detections_per_img = box_detections_per_img
        self.use_pretrained_model = use_pretrained_model
        
        # Create id2label and label2id mappings for COCO classes
        if num_classes == 91:  # COCO
            self.id2label = {i: COCO_INSTANCE_CATEGORY_NAMES[i] for i in range(len(COCO_INSTANCE_CATEGORY_NAMES))}
            self.label2id = {v: k for k, v in self.id2label.items()}
        else:
            # Generic class names
            self.id2label = {i: f"class_{i}" for i in range(num_classes)}
            self.label2id = {v: k for k, v in self.id2label.items()}
            
# ======================
# 2. Custom Model Class
# ======================
class FasterRCNNModel(PreTrainedModel):
    config_class = FasterRCNNConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Check if we should use a pretrained model directly
        if hasattr(config, 'use_pretrained_model') and config.use_pretrained_model:
            # Use pretrained model directly without modifying num_classes
            if hasattr(config, 'backbone_type') and config.backbone_type == 'resnet50_fpn_v2':
                self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                    weights="DEFAULT",
                    min_size=config.min_size,
                    max_size=config.max_size
                )
            else:
                # Default to MobileNetV2 if not specified
                self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                    weights="DEFAULT",
                    min_size=config.min_size,
                    max_size=config.max_size
                )
            
            # Update config with the actual number of classes from the pretrained model
            config.num_classes = self.model.roi_heads.box_predictor.cls_score.out_features
            config.id2label = {i: f"class_{i}" for i in range(config.num_classes)}
            config.label2id = {f"class_{i}": i for i in range(config.num_classes)}
        else:
            # Create custom model architecture with specified num_classes
            if hasattr(config, 'backbone_type') and config.backbone_type == 'resnet50_fpn_v2':
                # Use ResNet50 FPN v2 backbone
                backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                    'resnet50', weights="DEFAULT", trainable_layers=3
                )
                backbone.out_channels = 256  # ResNet FPN has 256 output channels
            else:
                # Default to MobileNetV2 backbone
                backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
                backbone.out_channels = 1280
            
            anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),)
            )
            
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=7,
                sampling_ratio=2
            )
            
            self.model = FasterRCNN(
                backbone,
                num_classes=config.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                min_size=config.min_size,
                max_size=config.max_size
            )
    
    #Processes the model outputs to make them compatible with the Hugging Face pipeline format
        #Processes the model outputs to make them compatible with the Hugging Face pipeline format
        #Processes the model outputs to make them compatible with the Hugging Face pipeline format
    def forward(self, images=None, targets=None, pixel_values=None, **kwargs):
        # Handle both 'images' and 'pixel_values' input formats
        if pixel_values is not None:
            images = pixel_values #list of tensor
        
        # If no images/pixel_values provided, check for 'inputs' in kwargs
        if images is None and 'inputs' in kwargs:
            images = kwargs['inputs']
            
        # Ensure we have images to process
        if images is None:
            raise ValueError("No images or pixel_values provided to model")
        
        # If targets are provided, it's training mode
        if targets is not None:
            # Ensure targets is properly formatted for torchvision models
            if isinstance(targets, dict) and all(k in targets for k in ['boxes', 'labels']):
                # Convert from transformers format to torchvision format
                formatted_targets = []
                for i in range(len(images)):
                    target_dict = {
                        'boxes': targets['boxes'][i] if 'boxes' in targets else targets.get('bbox', [])[i],
                        'labels': targets['labels'][i] if 'labels' in targets else targets.get('class_labels', [])[i]
                    }
                    # Add optional fields if present
                    if 'masks' in targets:
                        target_dict['masks'] = targets['masks'][i]
                    if 'area' in targets:
                        target_dict['area'] = targets['area'][i]
                    if 'iscrowd' in targets:
                        target_dict['iscrowd'] = targets['iscrowd'][i]
                    
                    formatted_targets.append(target_dict)
                
                return self.model(images, formatted_targets)
            else:
                # Assume targets is already in the correct format
                return self.model(images, targets)
        
        # Run the model for inference
        
        outputs = self.model(images)
        
        # For inference, return in a format compatible with HF pipelines
        if isinstance(outputs, list):
            # Convert torchvision format to transformers format
            batch_size = len(outputs)
            
            # Debug the raw outputs
            print(f"Raw torchvision outputs: {[list(out.keys()) for out in outputs]}")
            
            # Convert torchvision format to transformers format
            pred_boxes = []
            pred_scores = []
            pred_labels = []
            
            for i in range(batch_size):
                pred_boxes.append(outputs[i]['boxes'])
                pred_scores.append(outputs[i]['scores'])
                pred_labels.append(outputs[i]['labels'])
            
            # Stack if possible, otherwise keep as list
            if batch_size > 0:
                try:
                    pred_boxes = torch.stack(pred_boxes)
                    pred_scores = torch.stack(pred_scores)
                    pred_labels = torch.stack(pred_labels)
                except Exception as e:
                    print(f"Warning: Could not stack outputs: {e}")
                    # If stacking fails, use the first item directly
                    if len(pred_boxes) > 0:
                        pred_boxes = pred_boxes[0]
                        pred_scores = pred_scores[0]
                        pred_labels = pred_labels[0]
            
            # Create a class that supports both dictionary and attribute access
            class DictWithAttributes(dict):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.__dict__ = self
            
            # Print debug info
            print(f"Model output: boxes shape={pred_boxes.shape if hasattr(pred_boxes, 'shape') else 'list'}, "
                  f"scores shape={pred_scores.shape if hasattr(pred_scores, 'shape') else 'list'}")
            
            # Return as an object that supports both dictionary and attribute access
            # IMPORTANT: Include both original scores and labels directly, not just logits
            # return DictWithAttributes({
            #     "boxes": pred_boxes,
            #     "scores": pred_scores,
            #     "labels": pred_labels,
            #     "pred_boxes": pred_boxes,
            #     "pred_scores": pred_scores,
            #     "pred_labels": pred_labels
            # })
            # Return as an object that supports both dictionary and attribute access
            return DictWithAttributes({
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels
            })
        
        return outputs

    
# ======================
# 3. Preprocessing Utilities
# ======================
class FasterRCNNProcessor:
    def __init__(self, config=None, device='cuda'):
        self.config = config
        self.size_divisibility = 32
        self.device = device
    
        
        # Create torchvision transform
        # self.transform = torchvision.transforms.Compose([
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        # ])
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        
    def __call__(self, images: List[Image.Image]):
        # Use torchvision's preprocessing
        processed_images = []
        for image in images:
            # Handle different image modes
            # if image.mode != "RGB":
            #     image = image.convert("RGB")
            
            # Apply torchvision transform
            # 
            # img_tensor = transform(image).to(device) #[3, 1080, 810]
            image_tensor = self.transform(image).to(self.device)
            #img_tensor = transform(image).to(self.device) #[3, 1080, 810]
            processed_images.append(image_tensor)
        
        #FasterRCNN need list of image tensor as input
        return {"pixel_values": processed_images}
            # # Stack tensors if more than one image
            # if len(processed_images) > 1:
            #     return {"pixel_values": torch.stack(processed_images)}
            # else:
            #     return {"pixel_values": processed_images[0].unsqueeze(0)}
        
        
    def post_process_object_detection(self, outputs, threshold=0.5, target_sizes=None):
        """
        Use torchvision-style post-processing for object detection
        """
        # Extract predictions directly from model outputs
        if isinstance(outputs, dict):
            # Try to get direct outputs from our model format
            pred_boxes = outputs.get("boxes", outputs.get("pred_boxes", None))
            pred_scores = outputs.get("scores", outputs.get("pred_scores", None))
            pred_labels = outputs.get("labels", outputs.get("pred_labels", None))
        else:
            # Extract from object attributes
            pred_boxes = getattr(outputs, "boxes", getattr(outputs, "pred_boxes", None))
            pred_scores = getattr(outputs, "scores", getattr(outputs, "pred_scores", None))
            pred_labels = getattr(outputs, "labels", getattr(outputs, "pred_labels", None))
        
        if pred_boxes is None or pred_scores is None or pred_labels is None:
            raise ValueError("Model outputs don't contain required detection information")
        
        # Process each image in the batch
        results = []
        
        # Handle case where predictions are for a single image (not batched)
        if len(pred_boxes.shape) == 2:  # [num_detections, 4]
            pred_boxes = pred_boxes.unsqueeze(0)
            pred_scores = pred_scores.unsqueeze(0)
            pred_labels = pred_labels.unsqueeze(0)
        
        # Debug information
        print(f"Processing detection results: boxes shape={pred_boxes.shape}, scores shape={pred_scores.shape}")
        
        for i, (boxes, scores, labels) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            # Filter by threshold
            keep = scores > threshold
            filtered_boxes = boxes[keep].cpu()
            filtered_scores = scores[keep].cpu()
            filtered_labels = labels[keep].cpu()
            
            # Debug information
            print(f"Image {i}: Found {len(filtered_boxes)} detections above threshold {threshold}")
            
            # Skip if no detections
            if len(filtered_boxes) == 0:
                results.append([])
                continue
            
            # Convert to expected format
            image_predictions = []
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                # Get box coordinates
                x1, y1, x2, y2 = box.tolist()
                width = x2 - x1
                height = y2 - y1
                
                # Get label as string
                label_idx = int(label.item())
                if hasattr(self, 'config') and hasattr(self.config, 'id2label'):
                    label_str = self.config.id2label.get(label_idx, f"class_{label_idx}")
                else:
                    # Fallback to COCO labels if using a pretrained model
                    if 0 <= label_idx < len(COCO_INSTANCE_CATEGORY_NAMES):
                        label_str = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
                    else:
                        label_str = f"class_{label_idx}"
                
                # Create prediction entry
                image_predictions.append({
                    "score": float(score),
                    "label": label_str,
                    "box": {
                        "xmin": float(x1),
                        "ymin": float(y1),
                        "xmax": float(x2),
                        "ymax": float(y2),
                        "width": float(width),
                        "height": float(height)
                    }
                })
                # Debug information
                print(f"  Detection: {label_str} (score={float(score):.3f}) at box={x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}")
            
            # Sort by score (highest first)
            image_predictions.sort(key=lambda x: x["score"], reverse=True)
            results.append(image_predictions)
        
        return results
    
class FasterRCNNProcessor_hf:
    def __init__(self, config):
        self.config = config
        self.size_divisibility = 32
        
        # Define normalization parameters (ImageNet mean and std)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Use Hugging Face's AutoImageProcessor
        from transformers import AutoImageProcessor
        
        # Try to use a compatible image processor from Hugging Face
        try:
            # Use DETR image processor as it's compatible with object detection models
            self.image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.use_hf_processor = True
            print("Using Hugging Face image processor")
        except Exception as e:
            print(f"Could not load Hugging Face image processor: {e}")
            print("Falling back to custom implementation")
            self.use_hf_processor = False
        
    def __call__(self, images: List[Image.Image]):
        if self.use_hf_processor:
            # Use the Hugging Face image processor
            return self.image_processor(images=images, return_tensors="pt")
        else:
            # Fallback to custom implementation
            # Convert images to tensors and normalize
            processed_images = []
            for image in images:
                # Handle different image modes
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Convert to numpy array and normalize
                image_np = np.array(image).astype(np.float32) / 255.0
                
                # Apply normalization
                for i in range(3):
                    image_np[:, :, i] = (image_np[:, :, i] - self.mean[i]) / self.std[i]
                
                # Convert to tensor with correct channel order
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
                processed_images.append(image_tensor)
            
            # Create image tensors with padding
            images_tensor = self.batch_images(processed_images)
            return {"pixel_values": images_tensor}
    
    def batch_images(self, images):
        """
        Batch images with padding to ensure all images have the same dimensions.
        """
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        
        # Make sure dimensions are divisible by size_divisibility
        stride = self.size_divisibility
        max_size = list(max_size)
        max_size[1] = (max_size[1] + (stride - 1)) // stride * stride
        max_size[2] = (max_size[2] + (stride - 1)) // stride * stride
        max_size = tuple(max_size)
        
        # Create batch tensor
        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            
        return batched_imgs
    
    def prepare_inputs_for_generation(self, pixel_values, **kwargs):
        """
        Prepare inputs for the model's forward pass.
        Required for compatibility with some HF features.
        """
        return {
            "pixel_values": pixel_values,
            **kwargs
        }
        
    def post_process_object_detection(self, outputs, threshold=0.5, target_sizes=None):
        """
        Custom post-processing for object detection that works with dictionary outputs.
        
        Args:
            outputs: Model outputs (can be dict or object with attributes)
            threshold: Score threshold for detections
            target_sizes: Original image sizes for rescaling boxes
            
        Returns:
            List of dictionaries with detection results
        """
        # Handle both dictionary and object with attributes
        if isinstance(outputs, dict):
            # First try to get direct scores and labels (preferred)
            pred_scores = outputs.get("scores", outputs.get("pred_scores", None))
            pred_labels = outputs.get("labels", outputs.get("pred_labels", None))
            pred_boxes = outputs.get("boxes", outputs.get("pred_boxes", None))
            
            # If direct scores not available, fall back to logits
            if pred_scores is None or pred_labels is None:
                pred_logits = outputs.get("pred_logits", outputs.get("logits", None))
                if pred_logits is not None:
                    # Extract scores and labels from logits
                    if pred_logits.shape[-1] > 1:  # Multi-class case
                        pred_scores, pred_labels = pred_logits.softmax(-1).max(-1)
                    else:  # Binary case
                        pred_scores = pred_logits.sigmoid().squeeze(-1)
                        pred_labels = torch.ones_like(pred_scores, dtype=torch.long)
        else:
            # Extract predictions from object with attributes
            pred_scores = getattr(outputs, "scores", getattr(outputs, "pred_scores", None))
            pred_labels = getattr(outputs, "labels", getattr(outputs, "pred_labels", None))
            pred_boxes = getattr(outputs, "boxes", getattr(outputs, "pred_boxes", None))
            
            # Fall back to logits if needed
            if pred_scores is None or pred_labels is None:
                pred_logits = getattr(outputs, "pred_logits", getattr(outputs, "logits", None))
                if pred_logits is not None:
                    # Extract scores and labels from logits
                    if pred_logits.shape[-1] > 1:  # Multi-class case
                        pred_scores, pred_labels = pred_logits.softmax(-1).max(-1)
                    else:  # Binary case
                        pred_scores = pred_logits.sigmoid().squeeze(-1)
                        pred_labels = torch.ones_like(pred_scores, dtype=torch.long)
        
        if pred_boxes is None or pred_scores is None or pred_labels is None:
            raise ValueError("Model outputs don't contain required detection information")
        
        # Process each image in the batch
        results = []
        
        # Handle case where predictions are for a single image (not batched)
        if len(pred_boxes.shape) == 2:  # [num_detections, 4]
            pred_boxes = pred_boxes.unsqueeze(0)
            pred_scores = pred_scores.unsqueeze(0)
            pred_labels = pred_labels.unsqueeze(0)
        
        for i, (boxes, scores, labels) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            # Filter by threshold
            keep = scores > threshold
            filtered_boxes = boxes[keep].cpu()
            filtered_scores = scores[keep].cpu()
            filtered_labels = labels[keep].cpu()
            
            # Debug information
            print(f"Image {i}: Found {len(filtered_boxes)} detections above threshold {threshold}")
            
            # Skip if no detections
            if len(filtered_boxes) == 0:
                results.append([])
                continue
            
            # Rescale boxes if target sizes provided
            if target_sizes is not None:
                try:
                    img_h, img_w = target_sizes[i]
                    
                    # Try to use image processor's size if available
                    if hasattr(self, 'image_processor') and self.use_hf_processor:
                        try:
                            # Get normalization size from image processor
                            if hasattr(self.image_processor, 'size'):
                                if isinstance(self.image_processor.size, dict):
                                    proc_h = self.image_processor.size.get("height", self.image_processor.size.get("shortest_edge", img_h))
                                    proc_w = self.image_processor.size.get("width", self.image_processor.size.get("shortest_edge", img_w))
                                else:
                                    proc_h = proc_w = self.image_processor.size
                                
                                scale_x = img_w / proc_w
                                scale_y = img_h / proc_h
                            else:
                                # Fallback to assuming no scaling needed
                                scale_x = scale_y = 1.0
                        except Exception as e:
                            print(f"Warning: Error using image processor size: {e}")
                            scale_x = scale_y = 1.0
                    else:
                        # No image processor, assume model output is already in image coordinates
                        scale_x = scale_y = 1.0
                    
                    # Apply scaling
                    filtered_boxes[:, 0::2] *= scale_x
                    filtered_boxes[:, 1::2] *= scale_y
                    
                    # Ensure boxes are within image boundaries
                    filtered_boxes[:, 0::2] = torch.clamp(filtered_boxes[:, 0::2], min=0, max=img_w)
                    filtered_boxes[:, 1::2] = torch.clamp(filtered_boxes[:, 1::2], min=0, max=img_h)
                    
                except Exception as e:
                    print(f"Warning: Could not rescale boxes: {e}")
            
            # Convert to expected format
            image_predictions = []
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                # Get box coordinates
                if len(box) == 4:  # Standard box format [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box.tolist()
                    
                    # Handle case where x1 > x2 or y1 > y2
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                        
                    width = x2 - x1
                    height = y2 - y1
                else:
                    # Alternative box format [x, y, w, h]
                    x, y, width, height = box.tolist()
                    x1, y1 = x, y
                    x2, y2 = x + width, y + height
                
                # Get label as string
                label_idx = int(label.item())
                if hasattr(self, 'config') and hasattr(self.config, 'id2label'):
                    label_str = self.config.id2label.get(label_idx, f"class_{label_idx}")
                else:
                    # Fallback to COCO labels if using a pretrained model
                    if 0 <= label_idx < len(COCO_INSTANCE_CATEGORY_NAMES):
                        label_str = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
                    else:
                        label_str = f"class_{label_idx}"
                
                # Create prediction entry
                image_predictions.append({
                    "score": float(score),
                    "label": label_str,
                    "box": {
                        "xmin": float(x1),
                        "ymin": float(y1),
                        "xmax": float(x2),
                        "ymax": float(y2),
                        "width": float(width),
                        "height": float(height)
                    }
                })
                # Debug information
                print(f"  Detection: {label_str} (score={float(score):.3f}) at box={x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}")
            
            # Sort by score (highest first)
            image_predictions.sort(key=lambda x: x["score"], reverse=True)
            results.append(image_predictions)
        
        return results

from typing import Union, Dict, List, Any
def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Moves data to the specified device. Handles:
    - Single tensors
    - Lists/tuples of tensors
    - Dictionaries with tensor values
    - Nested structures of the above
    
    Args:
        data: Input data (tensor, list, dict, or nested structure)
        device: Target device (e.g., 'cuda:0' or torch.device('cpu'))
    
    Returns:
        Data moved to the target device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif hasattr(data, 'to'):  # Other objects with .to() method (e.g., nn.Module)
        return data.to(device)
    else:
        # Return data as is if not a tensor and doesn't have .to()
        return data
    
def inference_image(model_path, image_path, model_type="huggingface", output_path="output/", threshold=0.5, checkpoint_path=None):
    """
    Run inference on a single image and save the result with bounding boxes.
    
    Args:
        model_path (str): Path to the model or model name on Hugging Face Hub
        image_path (str or PIL.Image): Path to the input image or PIL Image object
        model_type (str): Type of model to use - "huggingface", "local", or "torchvision"
        output_path (str, optional): Path to save the output image with bounding boxes
        threshold (float): Confidence threshold for detections
        checkpoint_path (str, optional): Path to model checkpoint to load
        
    Returns:
        tuple: (detections, output_image_path)
    """
    import cv2
    import numpy as np
    import os
    from PIL import Image
    import torch
    import torchvision.transforms as T
    
    # Set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
        image_name = os.path.basename(image_path)
    else:
        image = image_path
        image_name = "detection_result.jpg"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"detected_{image_name}" if isinstance(image_path, str) else "detection_result.jpg")
    
    # Different model loading and inference based on model_type
    if model_type == "torchvision":
        # Use torchvision's default inference process for pretrained models
        print("Using torchvision's default inference process")
        
        # Load a pretrained model directly from torchvision
        if model_path == "fasterrcnn_resnet50_fpn_v2":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        elif model_path == "fasterrcnn_resnet50_fpn":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        elif model_path == "fasterrcnn_mobilenet_v3_large_fpn":
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        elif model_path == "fasterrcnn_mobilenet_v3_large_320_fpn":
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
        else:
            raise ValueError(f"Unknown torchvision model: {model_path}")
        
        model.to(device)
        model.eval()
        
        # Standard torchvision transform
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(image).to(device)
        
        # Run inference with torchvision model
        with torch.no_grad():
            prediction = model([img_tensor])
        
        # Process torchvision output format
        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        
        # Filter by threshold
        keep = scores > threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # Create detections in our standard format
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            detections.append({
                'score': float(score),
                'label': COCO_INSTANCE_CATEGORY_NAMES[label],
                'box': {
                    'xmin': float(x1),
                    'ymin': float(y1),
                    'xmax': float(x2),
                    'ymax': float(y2)
                }
            })
    
    elif model_type == "huggingface":
        # Register architecture if needed
        register_fasterrcnn_architecture()
        # Load model
        from transformers import AutoModelForObjectDetection, AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForObjectDetection.from_pretrained(model_path).to(device)
        print(f"Loaded model from {model_path}")
        
        # Create processor
        processor = FasterRCNNProcessor(model.config)
        
        # Preprocess image
        inputs = processor([image])
        
        # Move inputs to the same device as model
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process outputs
        original_size = image.size[::-1]  # (height, width)
        detections = processor.post_process_object_detection(
            outputs, 
            threshold=threshold,
            target_sizes=[original_size]
        )[0]
    
    elif model_type=="local":  # "local" model
        print("Creating a new model with default configuration")
        # Use a pretrained model with COCO classes
        config = FasterRCNNConfig(num_classes=91, use_pretrained_model=True)
        model = FasterRCNNModel(config).to(device)
        #model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        
        # Print model configuration
        print(f"Model configuration: num_classes={config.num_classes}, backbone={config.backbone_type}")
    
        # Load checkpoint if provided
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        # Set model to evaluation mode
        model.eval()
        
        # Create processor
        processor = FasterRCNNProcessor(model.config, device=device)
        # Preprocess image
        inputs = processor([image])
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"Model output keys: {outputs.keys() if isinstance(outputs, dict) else 'not a dict'}")
        
        # Post-process outputs
        original_size = image.size[::-1]  # (height, width)
        print(f"Original image size: {original_size}")
        
        detections = processor.post_process_object_detection(
            outputs, 
            threshold=threshold,  # Use the provided threshold (default 0.5)
            target_sizes=[original_size]
        )[0]
        
        # Map class indices to COCO class names if using pretrained model
        if config.use_pretrained_model:
            # Update detection labels with COCO class names
            for detection in detections:
                class_idx = int(detection['label'].split('_')[-1])
                if 0 <= class_idx < len(COCO_INSTANCE_CATEGORY_NAMES):
                    detection['label'] = COCO_INSTANCE_CATEGORY_NAMES[class_idx]
    
    else:
        print("Not supported")
        return None
               
    # Convert PIL image to OpenCV format for visualization
    img_cv = np.array(image)
    img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
    
    # Draw bounding boxes
    for detection in detections:
        # Get box coordinates
        box = detection['box']
        x1, y1 = int(box['xmin']), int(box['ymin'])
        x2, y2 = int(box['xmax']), int(box['ymax'])
        
        # Get label and score
        label = detection['label']
        score = detection['score']
        
        # Generate color based on label hash
        label_hash = hash(label) % 255
        color = (label_hash, 255 - label_hash, (label_hash + 125) % 255)
        
        # Draw rectangle
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label_text = f"{label}: {score:.2f}"
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_cv, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img_cv, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save image
    cv2.imwrite(output_file, img_cv)
    print(f"Detection result saved to {output_file}")
    
    # Print detection results
    print(f"Found {len(detections)} objects:")
    for i, detection in enumerate(detections):
        print(f"  {i+1}. {detection['label']} (confidence: {detection['score']:.2f})")
    
    return detections, output_file

# ======================
# 4. ONNX Conversion
# ======================
def convert_to_onnx(model, output_path, dummy_input):
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["pixel_values"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "boxes": {0: "num_detections"},
            "scores": {0: "num_detections"},
            "labels": {0: "num_detections"}
        },
        opset_version=12
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

# ======================
# 5. Inference Pipeline
# ======================
class FasterRCNNPipeline:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        
    def __call__(self, images, threshold=0.5):
        # Preprocess
        inputs = self.processor(images)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        results = []
        for output in outputs:
            keep = output['scores'] > threshold
            results.append({
                "boxes": output['boxes'][keep].cpu().numpy(),
                "scores": output['scores'][keep].cpu().numpy(),
                "labels": output['labels'][keep].cpu().numpy()
            })
        
        return results

# ======================
# 6. ONNX Inference Class
# ======================
class FasterRCNNOnnxInference:
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        
    def predict(self, pixel_values):
        outputs = self.session.run(None, {self.input_name: pixel_values.numpy()})
        return {
            "boxes": outputs[0],
            "scores": outputs[1],
            "labels": outputs[2]
        }

# ======================
# 7. Save to Hugging Face
# ======================
def save_to_hub(model, processor, repo_name):
    # Create directory
    os.makedirs(repo_name, exist_ok=True)
    
    # Save model and config
    model.save_pretrained(repo_name)
    
    # Save image processor
    if hasattr(processor, 'image_processor') and processor.use_hf_processor:
        # Save the Hugging Face image processor
        processor.image_processor.save_pretrained(repo_name)
    else:
        # Save custom processor
        torch.save(processor, os.path.join(repo_name, "processor.bin"))
    
    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 3, 800, 800)
    
    # Convert to ONNX
    convert_to_onnx(
        model.model, 
        os.path.join(repo_name, "model.onnx"), 
        dummy_input
    )
    
    # Create inference script
    with open(os.path.join(repo_name, "inference.py"), "w") as f:
        f.write("""from PIL import Image
import torch
import numpy as np
import onnxruntime
import os

class FasterRCNNInference:
    def __init__(self, model_path):
        # Load ONNX model
        self.ort_session = onnxruntime.InferenceSession(f"{model_path}/model.onnx")
        
        # Try to load Hugging Face image processor
        try:
            from transformers import AutoImageProcessor
            self.image_processor = AutoImageProcessor.from_pretrained(model_path)
            self.use_hf_processor = True
            print("Using Hugging Face image processor")
        except Exception as e:
            print(f"Could not load Hugging Face image processor: {e}")
            print("Falling back to custom processor")
            self.processor = torch.load(f"{model_path}/processor.bin")
            self.use_hf_processor = False
        
    def predict(self, images, threshold=0.5):
        # Preprocess
        if self.use_hf_processor:
            inputs = self.image_processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].numpy()
        else:
            inputs = self.processor(images)
            pixel_values = inputs["pixel_values"].numpy()
        
        # Run inference
        outputs = self.ort_session.run(None, {"pixel_values": pixel_values})
        
        # Post-process
        results = []
        for boxes, scores, labels in zip(outputs[0], outputs[1], outputs[2]):
            keep = scores > threshold
            results.append({
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep]
            })
        
        return results
""")
    
    # Create README
    with open(os.path.join(repo_name, "README.md"), "w") as f:
        f.write(f"""---
tags:
- object-detection
- vision
- fasterrcnn
- onnx
license: apache-2.0
---

# Custom Faster R-CNN Model

## Usage

### PyTorch
```python
from inference import FasterRCNNInference
from PIL import Image

model = FasterRCNNInference(".")
image = Image.open("your_image.jpg")
results = model.predict([image])""")


def upload2hf():
    # Initialize config and model
    config = FasterRCNNConfig(num_classes=80)
    model = FasterRCNNModel(config)
    
    # Initialize processor
    processor = FasterRCNNProcessor(config)

    # Save to hub
    save_to_hub(model, processor, "myfasterrcnn-detector")

    # Login and upload
    #terminal: huggingface-cli login
    api = HfApi()
    create_repo("lkk688/myfasterrcnn-detector", exist_ok=True)
    api.upload_folder(
        folder_path="myfasterrcnn-detector",
        repo_id="lkk688/myfasterrcnn-detector",
        repo_type="model"
    )

# ======================
# 8. Register Custom Architecture
# ======================
def register_fasterrcnn_architecture():
    """
    Register the FasterRCNN model architecture with the Hugging Face transformers library
    for full integration with the transformers ecosystem.
    """
    from transformers import AutoConfig, AutoModel, AutoModelForObjectDetection
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_FOR_OBJECT_DETECTION_MAPPING
    
    # Register the config
    CONFIG_MAPPING.register("fasterrcnn", FasterRCNNConfig)
    
    # Register the model architecture
    MODEL_MAPPING.register(FasterRCNNConfig, FasterRCNNModel)
    MODEL_FOR_OBJECT_DETECTION_MAPPING.register(FasterRCNNConfig, FasterRCNNModel)
    
    print("FasterRCNN architecture registered successfully with Hugging Face transformers")
    
def hf_pipeline_inference(model_id="lkk688/myfasterrcnn-detector", image_path="sampledata/bus.jpg", 
                          threshold=0.5, device=None, visualize=True, output_path=None):
    """
    Run inference using the Hugging Face pipeline API.
    
    Args:
        model_id (str): Hugging Face model ID or local path
        image_path (str): Path to the input image or directory
        threshold (float): Confidence threshold for detections
        device (int or str): Device to run inference on (-1 for CPU, 0+ for GPU)
        visualize (bool): Whether to visualize and save the results
        output_path (str): Path to save the visualization
        
    Returns:
        list: Detection results
    """
    from transformers import pipeline, AutoImageProcessor, AutoModelForObjectDetection
    from PIL import Image
    import cv2
    import numpy as np
    import os
    
    # Register architecture if needed
    register_fasterrcnn_architecture()
    
    # Set device
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    
    try:
        # Try to load the model and image processor directly
        model = AutoModelForObjectDetection.from_pretrained(model_id)
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        
        # Create pipeline with explicit image processor
        object_detector = pipeline(
            "object-detection", 
            model=model,
            image_processor=image_processor,
            device=device,
            threshold=threshold
        )
        print("Successfully created pipeline with Hugging Face image processor")
    except Exception as e:
        print(f"Error loading model or image processor: {e}")
        print("Trying alternative approach...")
        
        try:
            # Try creating pipeline directly
            object_detector = pipeline(
                "object-detection", 
                model=model_id,
                device=device,
                threshold=threshold
            )
            print("Successfully created pipeline without explicit image processor")
        except Exception as e2:
            print(f"Error creating pipeline: {e2}")
            print("Using custom inference implementation")
            
            # Load model
            model = AutoModelForObjectDetection.from_pretrained(model_id)
            model.to(f"cuda:{device}" if device >= 0 else "cpu")
            
            # Use our custom processor instead of DETR
            image_processor = FasterRCNNProcessor(model.config)
            
            # Create custom detector function
            def object_detector(image):
                # Preprocess
                inputs = image_processor(images=[image], return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(f"cuda:{device}" if device >= 0 else "cpu")
                
                # Get original image size for rescaling boxes
                original_size = image.size[::-1]  # (height, width)
                
                # Run inference
                with torch.no_grad():
                    outputs = model(pixel_values)
                
                # Use our custom post-processing
                detections = image_processor.post_process_object_detection(
                    outputs, 
                    threshold=threshold,
                    target_sizes=[original_size]
                )[0]  # Get first image's results
                
                # Convert box format if needed
                for detection in detections:
                    if "xmax" in detection["box"] and "width" not in detection["box"]:
                        box = detection["box"]
                        box["width"] = box["xmax"] - box["xmin"]
                        box["height"] = box["ymax"] - box["ymin"]
                
                return detections
    
    # Check if input is a directory or file
    if os.path.isdir(image_path):
        # Process all images in directory
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        results = []
        
        for img_file in image_files:
            try:
                # Load image
                image = Image.open(img_file).convert("RGB")
                
                # Run inference
                detections = object_detector(image)
                
                # Visualize if requested
                if visualize:
                    visualized_img = visualize_detections(img_file, detections, output_path)
                
                results.append({
                    "file": img_file,
                    "detections": detections
                })
                
                # Print results
                print(f"\nResults for {os.path.basename(img_file)}:")
                for detection in detections:
                    print(f"Detected {detection['label']} with confidence {round(detection['score'], 3)} "
                          f"at location {detection['box']}")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        return results
    else:
        # Process single image
        image = Image.open(image_path).convert("RGB")
        
        # Run inference
        detections = object_detector(image)
        
        # Visualize if requested
        if visualize:
            visualized_img = visualize_detections(image_path, detections, output_path)
        
        # Print results
        print(f"\nResults for {os.path.basename(image_path)}:")
        for detection in detections:
            print(f"Detected {detection['label']} with confidence {round(detection['score'], 3)} "
                  f"at location {detection['box']}")
        
        return detections

def visualize_detections(image_path, detections, output_path=None):
    """
    Visualize detection results on an image.
    
    Args:
        image_path (str): Path to the input image
        detections (list): Detection results from pipeline
        output_path (str, optional): Path to save the visualization
        
    Returns:
        np.ndarray: Visualization image with bounding boxes
    """
    import cv2
    import numpy as np
    import os
    from PIL import Image
    
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        # Convert PIL image to OpenCV format
        img = np.array(image_path)
        img = img[:, :, ::-1].copy()  # RGB to BGR
    
    # Draw boxes
    for detection in detections:
        # Get box coordinates
        box = detection['box']
        x1, y1, width, height = int(box['xmin']), int(box['ymin']), int(box['width']), int(box['height'])
        x2, y2 = x1 + width, y1 + height
        
        # Get label and score
        label = detection['label']
        score = detection['score']
        
        # Generate color based on label hash
        label_hash = hash(label) % 255
        color = (label_hash, 255 - label_hash, (label_hash + 125) % 255)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label_text = f"{label}: {score:.2f}"
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save if output path is provided
    if output_path:
        # If output_path is a directory, create filename
        if os.path.isdir(output_path):
            if isinstance(image_path, str):
                base_filename = os.path.basename(image_path)
                output_file = os.path.join(output_path, f"{os.path.splitext(base_filename)[0]}_detected{os.path.splitext(base_filename)[1]}")
            else:
                output_file = os.path.join(output_path, "detection_result.jpg")
        else:
            output_file = output_path
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save image
        cv2.imwrite(output_file, img)
        print(f"Visualization saved to {output_file}")
    
    return img

# Create inference class directly
class FasterRCNNInference_local:
    def __init__(self, model_path, device="cuda"):
        register_fasterrcnn_architecture()
        # Try to load Hugging Face image processor
        from transformers import pipeline, AutoImageProcessor, AutoModelForObjectDetection
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForObjectDetection.from_pretrained(model_path).to(device)
        self.use_hf_processor = True
        self.device = device
        print("Using Hugging Face image processor")
    
    
    def process_image_file(self, image_path, threshold=0.5, visualize=True, output_dir=None):
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        results = self.predict([image], threshold)[0]
        
        if visualize:
            visualized_img = visualize_detections(image_path, 
                                                    [{"score": float(s), 
                                                    "label": f"class_{int(l)}", 
                                                    "box": {"xmin": float(b[0]), 
                                                            "ymin": float(b[1]), 
                                                            "width": float(b[2]-b[0]), 
                                                            "height": float(b[3]-b[1])}} 
                                                    for b, s, l in zip(results["boxes"], 
                                                                        results["scores"], 
                                                                        results["labels"])], 
                                                    output_dir)
        return results
    
    def process_directory(self, directory, threshold=0.5, visualize=True, output_dir=None):
        import glob
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
        
        results = []
        for image_path in image_files:
            try:
                result = self.process_image_file(image_path, threshold, visualize, output_dir)
                results.append({"file": image_path, "detections": result})
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        return results
    
    def predict(self, image, threshold=0.5):
        from PIL import Image
        # Preprocess
        if self.use_hf_processor:
            inputs = self.image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].numpy() #(1, 3, 1066, 800)
        else:
            inputs = self.processor(image)
            pixel_values = inputs["pixel_values"].numpy()
        
        inputs = inputs.to(self.device)
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        #outputs is a dict
        
        # Convert outputs to COCO API format
        target_sizes = torch.tensor([image.size[::-1]])  # [height, width] [[1080,  810]])
        results = self.image_processor.post_process_object_detection(
            outputs, 
            threshold=0.5, 
            target_sizes=target_sizes
        )[0]

        # Print results
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {model.config.id2label[label.item()]} "
                f"with confidence {round(score.item(), 3)} at location {box}"
            )
    
        return results
                        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FasterRCNN Model Operations")
    parser.add_argument("--action", type=str, default="test", 
                        choices=["test", "upload", "inference", "pipeline"],
                        help="Action to perform: upload to HF, run ONNX inference, or run HF pipeline")
    parser.add_argument("--model_id", type=str, default="lkk688/myfasterrcnn-detector",
                        help="Hugging Face model ID or local path")
    parser.add_argument("--image", type=str, default="sampledata/bus.jpg",
                        help="Path to input image or directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output directory or file")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Detection confidence threshold")
    parser.add_argument("--device", type=int, default=None,
                        help="Device to run inference on (-1 for CPU, 0+ for GPU)")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Visualize detection results")
    parser.add_argument("--use_onnx", action="store_true", default=True,
                        help="Use ONNX runtime for inference (faster)")
    
    args = parser.parse_args()
    
    if args.action == "upload":
        upload2hf()
        print("Model uploaded successfully to Hugging Face Hub")
    elif args.action == "test":
        # Example usage
        detections, output_path = inference_image(
            model_path="fasterrcnn_resnet50_fpn_v2",  # HF Hub model or local path
            image_path=args.image,
            model_type="local", #"local", "torchvision",  # or "huggingface"
            output_path="output",  # Optional
            threshold=0.2,
            checkpoint_path=None  # Optional: path to checkpoint
        )
    elif args.action == "inference":
        #use local FasterRCNNInference
        model = FasterRCNNInference_local(args.model_id)
        image = Image.open(args.image)
        result = model.predict(image=image)
    
    # Handle pipeline action
    if args.action == "pipeline":
        print(f"Running pipeline inference with model: {args.model_id}")
        # Use the HF pipeline
        hf_pipeline_inference(
            model_id=args.model_id,
            image_path=args.image,
            threshold=args.threshold,
            device=args.device,
            visualize=args.visualize,
            output_path=args.output
        )