import torch
import numpy as np
import cv2
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects, binary_dilation
from typing import List, Dict, Tuple, Any, Optional

#pip install git+https://github.com/facebookresearch/segment-anything.git
from segment_anything import sam_model_registry, SamPredictor

#pip install supervision
import supervision as sv #https://supervision.roboflow.com/latest/

#pip install git+https://github.com/IDEA-Research/GroundingDINO.git
from groundingdino.util.inference import load_model, load_image, predict, annotate
#from groundingdino.util.utils import clean_state_dict
#from groundingdino.config import GroundingDINOConfig

# Step 1: Set up HF models for common objects
def setup_hf_world(use_owlv2=False):
    if use_owlv2 == False:
        model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlvit-base-patch32", torch_dtype=torch.float16).to("cuda")
        processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
    else:
        model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", torch_dtype=torch.float16).to("cuda")
        processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
    return model, processor

# Function to detect common objects using HF models
def detect_common_objects(image, model, processor, text_labels, device="cuda", grounding=True):
    #text_labels = [["person", "car", "bicycle", "motorcycle", "truck", "bus", "traffic light", "stop sign"]]
    #text_labels = [["a photo of a cat", "a photo of a dog"]]
    inputs = processor(text=text_labels, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.tensor([[image.height, image.width]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)

    if grounding == True:
        results = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, box_threshold=0.4, text_threshold=0.3
        )
    else:
        results = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.2, text_labels=text_labels
        )
    # Retrieve predictions for the first image for the corresponding text queries
    result = results[0]
    boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
    for box, score, text_label in zip(boxes, scores, text_labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")
        
    # boxes = []
    # labels = []
    # scores = []
    
    # for box, label, score in zip(results["boxes"], results["text_labels"], results["scores"]):
    #     box = [round(x, 2) for x in box.tolist()]
    #     boxes.append(box)
    #     labels.append(label)
    #     scores.append(score.item())
    #     print(f"Detected {label} with confidence {round(score.item(), 3)} at location {box}")
    
    return boxes, text_labels, scores

# Step 2: Set up Grounding DINO for specific anomalies
#https://huggingface.co/docs/transformers/en/model_doc/grounding-dino#transformers.GroundingDinoForObjectDetection
def setup_grounding_dino():
    # config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    # checkpoint_path = "weights/groundingdino_swint_ogc.pth"
    # config = GroundingDINOConfig.fromfile(config_path)
    # model = load_model(config, checkpoint_path)
    # model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    # return model
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    device = "cuda"
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return model, processor

# Function to detect anomalies using Grounding DINO
#ref: https://github.com/IDEA-Research/GroundingDINO
#https://github.com/IDEA-Research/Grounded-SAM-2/tree/main
def detect_anomalies(image_path, model):
    image_source, image = load_image(image_path)
    
    # Define text prompts for anomalies
    TEXT_PROMPT = "dumped trash . blocked sidewalk . illegal parking . abandoned object . damaged infrastructure"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("annotated_image.jpg", annotated_frame)
    
    return boxes, phrases, logits

# Step 3: Set up SAM for refining detections
def setup_sam():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def refine_with_sam(
    image: np.ndarray,
    boxes_list: List[np.ndarray],
    class_labels_list: List[List[str]],
    scores_list: List[np.ndarray],
    sam_checkpoint: str = "sam_vit_h_4b8939.pth",
    model_type: str = "vit_h",
    device: str = "cuda",
    iou_threshold: float = 0.5,
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    box_extension_factor: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Refine detection results from multiple models using SAM.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        boxes_list: List of bounding boxes arrays from different models, each with shape (n, 4)
                   in format [x1, y1, x2, y2]
        class_labels_list: List of class labels corresponding to each boxes array
        scores_list: List of confidence scores corresponding to each boxes array
        sam_checkpoint: Path to SAM model checkpoint
        model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
        device: Device to run the model on ('cuda' or 'cpu')
        iou_threshold: Threshold for NMS to remove duplicate detections
        score_threshold: Minimum confidence score to keep a detection
        mask_threshold: Threshold to binarize SAM masks
        box_extension_factor: Factor to extend boxes before SAM processing for better context
        
    Returns:
        Tuple containing:
        - refined_boxes: Combined and refined bounding boxes
        - refined_masks: Binary masks for each detection
        - refined_labels: Class labels for each refined detection
        - refined_scores: Confidence scores for each refined detection
    """
    # Initialize SAM model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # Set image in SAM predictor
    predictor.set_image(image)
    
    # Combine all detections
    all_boxes = []
    all_labels = []
    all_scores = []
    
    for boxes, labels, scores in zip(boxes_list, class_labels_list, scores_list):
        if len(boxes) == 0:
            continue
            
        # Filter by score threshold
        valid_indices = scores >= score_threshold
        filtered_boxes = boxes[valid_indices]
        filtered_labels = [labels[i] for i in range(len(labels)) if valid_indices[i]]
        filtered_scores = scores[valid_indices]
        
        all_boxes.append(filtered_boxes)
        all_labels.extend(filtered_labels)
        all_scores.append(filtered_scores)
    
    if not all_boxes:
        return np.array([]), np.array([]), [], np.array([])
    
    all_boxes = np.vstack(all_boxes)
    all_scores = np.concatenate(all_scores)
    
    # Apply NMS to remove duplicate detections
    refined_indices = apply_nms(all_boxes, all_scores, iou_threshold)
    refined_boxes = all_boxes[refined_indices]
    refined_labels = [all_labels[i] for i in refined_indices]
    refined_scores = all_scores[refined_indices]
    
    # Extend boxes slightly for better SAM results
    extended_boxes = extend_boxes(refined_boxes, image.shape[1], image.shape[0], box_extension_factor)
    
    # Convert boxes to format expected by SAM
    sam_boxes = torch.tensor(extended_boxes, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(sam_boxes, (image.shape[0], image.shape[1]))
    
    # Generate masks
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    
    # Convert masks to numpy and refine bounding boxes
    refined_masks = masks.cpu().numpy()
    refined_boxes_from_masks = []
    final_masks = []
    
    for i, mask in enumerate(refined_masks):
        # Threshold mask
        binary_mask = (mask[0] > mask_threshold).astype(np.uint8)
        final_masks.append(binary_mask)
        
        # Find contours to get refined box
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (main object)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            refined_boxes_from_masks.append([x, y, x + w, y + h])
        else:
            # If no contour found, use the original box
            refined_boxes_from_masks.append(refined_boxes[i])
    
    refined_boxes_from_masks = np.array(refined_boxes_from_masks)
    final_masks = np.array(final_masks)
    
    return refined_boxes_from_masks, final_masks, refined_labels, refined_scores

def apply_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        boxes: Array of bounding boxes in format [x1, y1, x2, y2]
        scores: Array of confidence scores
        iou_threshold: IoU threshold for considering boxes as duplicates
        
    Returns:
        List of indices of boxes to keep
    """
    # Sort boxes by score
    sorted_indices = np.argsort(scores)[::-1]
    keep_indices = []
    
    while sorted_indices.size > 0:
        # Pick the box with highest score
        current_index = sorted_indices[0]
        keep_indices.append(current_index)
        
        if sorted_indices.size == 1:
            break
            
        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[current_index], boxes[sorted_indices[1:]])
        
        # Remove boxes with IoU over threshold
        remaining_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[1:][remaining_indices]
    
    return keep_indices

def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between a single box and an array of boxes.
    
    Args:
        box: Single bounding box in format [x1, y1, x2, y2]
        boxes: Array of bounding boxes in format [x1, y1, x2, y2]
        
    Returns:
        Array of IoU values
    """
    # Calculate intersection area
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate areas of both boxes
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Calculate IoU
    iou = intersection_area / (box_area + boxes_area - intersection_area)
    
    return iou

def extend_boxes(boxes: np.ndarray, img_width: int, img_height: int, extension_factor: float) -> np.ndarray:
    """
    Extend boxes by a factor while keeping them within image boundaries.
    
    Args:
        boxes: Array of bounding boxes in format [x1, y1, x2, y2]
        img_width: Width of the image
        img_height: Height of the image
        extension_factor: Factor by which to extend boxes
        
    Returns:
        Extended boxes
    """
    extended_boxes = boxes.copy()
    
    # Calculate width and height of each box
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    
    # Calculate extension amount
    x_extension = widths * extension_factor / 2
    y_extension = heights * extension_factor / 2
    
    # Extend boxes
    extended_boxes[:, 0] = np.maximum(0, boxes[:, 0] - x_extension)
    extended_boxes[:, 1] = np.maximum(0, boxes[:, 1] - y_extension)
    extended_boxes[:, 2] = np.minimum(img_width, boxes[:, 2] + x_extension)
    extended_boxes[:, 3] = np.minimum(img_height, boxes[:, 3] + y_extension)
    
    return extended_boxes

def visualize_results(
    image: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
    labels: List[str],
    scores: np.ndarray,
    output_path: str = "refined_detection.jpg",
    alpha: float = 0.5,
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Visualize detection results with boxes and segmentation masks.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        boxes: Array of bounding boxes in format [x1, y1, x2, y2]
        masks: Array of binary masks
        labels: List of class labels for each detection
        scores: Array of confidence scores
        output_path: Path to save the visualization
        alpha: Transparency of masks
        color_map: Dictionary mapping class labels to RGB colors
        
    Returns:
        Visualization image with boxes and masks
    """
    # If no color map is provided, create one
    if color_map is None:
        # Define some default colors (R, G, B)
        default_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
        ]
        
        # Create a color map for each unique label
        unique_labels = set(labels)
        color_map = {}
        
        for i, label in enumerate(unique_labels):
            color_map[label] = default_colors[i % len(default_colors)]
    
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Create an overlay for the masks
    mask_overlay = np.zeros_like(vis_image)
    
    # Draw each detection
    for i in range(len(boxes)):
        # Get box coordinates
        x1, y1, x2, y2 = boxes[i].astype(int)
        
        # Get class label and color
        label = labels[i]
        color = color_map[label]
        
        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Add label with score
        text = f"{label}: {scores[i]:.2f}"
        cv2.putText(vis_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Apply mask if available
        if i < len(masks):
            mask = masks[i]
            mask_color = (*color, 255)  # Add alpha channel
            
            # Color the mask area in the overlay
            colored_mask = np.zeros_like(vis_image)
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]
            
            # Add the colored mask to the overlay
            mask_overlay = cv2.addWeighted(mask_overlay, 1, colored_mask, 1, 0)
    
    # Combine the original image with the mask overlay
    vis_image = cv2.addWeighted(vis_image, 1, mask_overlay, alpha, 0)
    
    # Save the visualization
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image

# Example usage in the main pipeline
def integrate_sam_refinement(video_path, output_path, sample_rate=30):
    # Set up models (Assuming the models are already initialized)
    hf_model, hf_processor = setup_hf_world()
    dino_model, dino_processor = setup_grounding_dino()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    annotations = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every n-th frame
        if frame_count % sample_rate == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            text_labels = [["person", "car", "bicycle", "motorcycle", "truck", "bus", "traffic light", "stop sign"]]
            #text_labels = [["a photo of a cat", "a photo of a dog"]]
            # Detect common objects with YOLO-World
            common_boxes, common_labels, common_scores = detect_common_objects(
                pil_image, hf_model, hf_processor, text_labels
            )
            
            # Save frame temporarily for Grounding DINO
            #temp_path = f"{output_path}/temp_frame_{frame_count}.jpg"
            #cv2.imwrite(temp_path, frame)
            
            # Detect anomalies with Grounding DINO
            #anomaly_boxes, anomaly_labels, anomaly_scores = detect_anomalies(temp_path, dino_model)
            anomaly_boxes, anomaly_labels, anomaly_scores = detect_common_objects(
                pil_image, dino_model, dino_processor, text_labels
            )
            
            # Prepare lists for SAM refinement
            boxes_list = [
                np.array(common_boxes),
                np.array(anomaly_boxes)
            ]
            labels_list = [common_labels, anomaly_labels]
            scores_list = [
                np.array(common_scores),
                np.array(anomaly_scores)
            ]
            
            # Refine detections with SAM
            refined_boxes, refined_masks, refined_labels, refined_scores = refine_with_sam(
                frame_rgb, boxes_list, labels_list, scores_list
            )
            
            # Visualize results
            vis_image = visualize_results(
                frame_rgb, refined_boxes, refined_masks, refined_labels, refined_scores,
                output_path=f"{output_path}/refined_frame_{frame_count}.jpg"
            )
            
            # Save annotations in COCO format
            for i, (box, label, score, mask) in enumerate(zip(refined_boxes, refined_labels, refined_scores, refined_masks)):
                x1, y1, x2, y2 = box
                
                # Save mask as separate file
                mask_path = f"{output_path}/mask_{frame_count}_{i}.png"
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                
                annotations.append({
                    "image_id": frame_count // sample_rate,
                    "frame_path": f"{output_path}/frame_{frame_count}.jpg",
                    "category": label,
                    "bbox": [x1, y1, x2-x1, y2-y1],  # [x, y, width, height]
                    "score": float(score),
                    "mask_path": mask_path,
                    "segmentation": encode_rle(mask)  # RLE encoding for COCO format
                })
            
            # Save the original frame
            cv2.imwrite(f"{output_path}/frame_{frame_count}.jpg", frame)
            
            # Clean up temp file
            # import os
            # if os.path.exists(temp_path):
            #     os.remove(temp_path)
                
        frame_count += 1
    
    cap.release()
    
    # Save annotations to JSON
    import json
    with open(f"{output_path}/annotations.json", "w") as f:
        json.dump(annotations, f)
    
    return annotations

def encode_rle(mask):
    """
    Encode binary mask to Run-Length Encoding format for COCO.
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()

def setup_depth_model(model_name="Intel/dpt-large"):
    """
    Set up a Hugging Face depth estimation model.
    
    Args:
        model_name: Name of the depth estimation model
        
    Returns:
        feature_extractor: DPT feature extractor
        model: DPT depth estimation model
    """
    feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
    model = DPTForDepthEstimation.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return feature_extractor, model

def predict_depth(image, feature_extractor, model):
    """
    Predict depth map from an image.
    
    Args:
        image: PIL Image or numpy array
        feature_extractor: DPT feature extractor
        model: DPT depth estimation model
        
    Returns:
        depth_map: Normalized depth map as numpy array
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if image.shape[2] == 3:  # Check if RGB
            image = Image.fromarray(image)
        else:  # BGR to RGB conversion
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Prepare image for model
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Convert to numpy and normalize
    depth_map = predicted_depth.squeeze().cpu().numpy()
    
    # Normalize depth map to 0-1 range for visualization
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    return depth_norm

def depth_based_refinement(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: List[str],
    scores: np.ndarray,
    depth_map: np.ndarray,
    depth_threshold: float = 0.2,
    min_region_size: int = 100,
    expansion_ratio: float = 0.1
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Refine object detection bounding boxes using depth information.
    
    Args:
        image: RGB image as numpy array
        boxes: Array of bounding boxes in format [x1, y1, x2, y2]
        labels: List of class labels
        scores: Array of confidence scores
        depth_map: Depth map as numpy array (normalized 0-1)
        depth_threshold: Threshold for depth discontinuity
        min_region_size: Minimum size of a valid region in pixels
        expansion_ratio: How much to expand the initial box for depth analysis
        
    Returns:
        refined_boxes: Improved bounding boxes
        refined_labels: Corresponding labels
        refined_scores: Corresponding confidence scores
    """
    image_height, image_width = image.shape[:2]
    refined_boxes = []
    refined_labels = []
    refined_scores = []
    
    # Process each detection
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = map(int, box)
        
        # Expand box slightly for better depth analysis
        width, height = x2 - x1, y2 - y1
        exp_x = int(width * expansion_ratio)
        exp_y = int(height * expansion_ratio)
        
        # Ensure expanded box stays within image boundaries
        e_x1 = max(0, x1 - exp_x)
        e_y1 = max(0, y1 - exp_y)
        e_x2 = min(image_width - 1, x2 + exp_x)
        e_y2 = min(image_height - 1, y2 + exp_y)
        
        # Extract region of interest from depth map
        depth_roi = depth_map[e_y1:e_y2, e_x1:e_x2]
        
        # Skip if ROI is empty
        if depth_roi.size == 0:
            refined_boxes.append(box)
            refined_labels.append(label)
            refined_scores.append(score)
            continue
        
        # Calculate depth gradients to find object boundaries
        grad_x = cv2.Sobel(depth_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_roi, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold gradient magnitude to get object boundaries
        boundaries = grad_magnitude > depth_threshold
        
        # Create foreground mask based on depth consistency
        # Get median depth of the central region as reference
        center_x, center_y = (x1 + x2) // 2 - e_x1, (y1 + y2) // 2 - e_y1
        center_size = min(width, height) // 4
        center_x1 = max(0, center_x - center_size)
        center_y1 = max(0, center_y - center_size)
        center_x2 = min(depth_roi.shape[1] - 1, center_x + center_size)
        center_y2 = min(depth_roi.shape[0] - 1, center_y + center_size)
        
        if center_x2 > center_x1 and center_y2 > center_y1:
            center_depth = np.median(depth_roi[center_y1:center_y2, center_x1:center_x2])
            
            # Create mask for pixels with similar depth to the center
            depth_mask = np.abs(depth_roi - center_depth) < depth_threshold
            
            # Combine with boundary information
            mask = depth_mask & ~boundaries
            
            # Dilate to connect nearby regions
            mask = binary_dilation(mask, iterations=2)
            
            # Label connected components
            labeled_mask, num_features = label(mask)
            
            # Find the largest connected component
            if num_features > 0:
                # Get the label of the center pixel
                center_label = labeled_mask[center_y, center_x]
                
                if center_label > 0:
                    # Extract object mask for the center label
                    object_mask = (labeled_mask == center_label)
                    
                    # Find object bounds
                    obj_y_indices, obj_x_indices = np.where(object_mask)
                    
                    if len(obj_y_indices) > min_region_size:
                        # Get bounding box from mask
                        mask_x1 = np.min(obj_x_indices) + e_x1
                        mask_y1 = np.min(obj_y_indices) + e_y1
                        mask_x2 = np.max(obj_x_indices) + e_x1
                        mask_y2 = np.max(obj_y_indices) + e_y1
                        
                        # Use the refined box
                        refined_box = [float(mask_x1), float(mask_y1), float(mask_x2), float(mask_y2)]
                        refined_boxes.append(refined_box)
                        refined_labels.append(label)
                        refined_scores.append(score)
                        continue
        
        # Fallback to original box if depth refinement fails
        refined_boxes.append(box)
        refined_labels.append(label)
        refined_scores.append(score)
    
    return np.array(refined_boxes), refined_labels, np.array(refined_scores)

def visualize_depth_refinement(
    image: np.ndarray,
    depth_map: np.ndarray,
    original_boxes: np.ndarray,
    refined_boxes: np.ndarray,
    labels: List[str],
    output_path: str = "depth_refined_detection.jpg"
) -> np.ndarray:
    """
    Visualize original and depth-refined bounding boxes.
    
    Args:
        image: RGB image
        depth_map: Normalized depth map
        original_boxes: Original detection boxes
        refined_boxes: Refined detection boxes
        labels: Class labels
        output_path: Path to save visualization
        
    Returns:
        Visualization image
    """
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original image with detection boxes
    axes[0].imshow(image)
    axes[0].set_title("Original Detection")
    for box in original_boxes:
        x1, y1, x2, y2 = map(int, box)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                             fill=False, edgecolor='red', linewidth=2)
        axes[0].add_patch(rect)
    
    # Plot depth map
    axes[1].imshow(depth_map, cmap='plasma')
    axes[1].set_title("Depth Map")
    
    # Plot image with refined boxes
    axes[2].imshow(image)
    axes[2].set_title("Depth-Refined Detection")
    for i, box in enumerate(refined_boxes):
        x1, y1, x2, y2 = map(int, box)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                             fill=False, edgecolor='green', linewidth=2)
        axes[2].add_patch(rect)
        axes[2].text(x1, y1-5, labels[i], color='white', 
                    backgroundcolor='green', fontsize=8)
    
    # Remove axis ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Save and show
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Create a composite image for return
    vis_img = cv2.imread(output_path)
    return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

def integrate_depth_refinement(video_path, output_path, sample_rate=30):
    """
    Complete pipeline integrating depth estimation and object detection.
    
    Args:
        video_path: Path to input video
        output_path: Path to save outputs
        sample_rate: Process every nth frame
    """
    # Set up models
    print("Setting up models...")
    # Object detection models
    yolo_model, yolo_processor = setup_yolo_world()
    dino_model = setup_grounding_dino()
    
    # Depth estimation model
    depth_extractor, depth_model = setup_depth_model()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    annotations = []
    
    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every n-th frame
        if frame_count % sample_rate == 0:
            print(f"Processing frame {frame_count}...")
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Step 1: Detect objects with YOLO-World
            common_boxes, common_labels, common_scores = detect_common_objects(
                pil_image, yolo_model, yolo_processor
            )
            common_boxes = np.array(common_boxes)
            common_scores = np.array(common_scores)
            
            # Step 2: Detect anomalies with Grounding DINO
            temp_path = f"{output_path}/temp_frame_{frame_count}.jpg"
            cv2.imwrite(temp_path, frame)
            anomaly_boxes, anomaly_labels, anomaly_scores = detect_anomalies(temp_path, dino_model)
            
            # Step 3: Estimate depth
            depth_map = predict_depth(pil_image, depth_extractor, depth_model)
            
            # Visualize depth map
            plt.imsave(f"{output_path}/depth_map_{frame_count}.jpg", 
                     depth_map, cmap='plasma')
            
            # Step 4: Refine common object detections with depth
            refined_common_boxes, refined_common_labels, refined_common_scores = depth_based_refinement(
                frame_rgb, common_boxes, common_labels, common_scores, depth_map
            )
            
            # Step 5: Refine anomaly detections with depth
            refined_anomaly_boxes, refined_anomaly_labels, refined_anomaly_scores = depth_based_refinement(
                frame_rgb, anomaly_boxes, anomaly_labels, anomaly_scores.tolist(), depth_map
            )
            
            # Step 6: Visualize original vs refined detections
            visualize_depth_refinement(
                frame_rgb, depth_map, common_boxes, refined_common_boxes, 
                common_labels, f"{output_path}/depth_refined_{frame_count}.jpg"
            )
            
            # Step 7: Combine all refined detections
            all_boxes = np.vstack([refined_common_boxes, refined_anomaly_boxes])
            all_labels = refined_common_labels + refined_anomaly_labels
            all_scores = np.concatenate([refined_common_scores, refined_anomaly_scores])
            
            # Step 8: Further refine with SAM if needed
            # This is optional - uncomment if you have SAM set up
            # sam_predictor = setup_sam()
            # final_boxes, final_masks, final_labels, final_scores = refine_with_sam(
            #     frame_rgb, [all_boxes], [all_labels], [all_scores]
            # )
            
            # For now, use the depth-refined results directly
            final_boxes = all_boxes
            final_labels = all_labels
            final_scores = all_scores
            
            # Step 9: Save annotations
            for i, (box, label, score) in enumerate(zip(final_boxes, final_labels, final_scores)):
                x1, y1, x2, y2 = box
                annotations.append({
                    "image_id": frame_count // sample_rate,
                    "frame_path": f"{output_path}/frame_{frame_count}.jpg",
                    "category": label,
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "score": float(score),
                    "depth_refined": True
                })
            
            # Save the frame
            cv2.imwrite(f"{output_path}/frame_{frame_count}.jpg", frame)
            
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        frame_count += 1
    
    cap.release()
    
    # Save annotations to JSON
    import json
    with open(f"{output_path}/annotations.json", "w") as f:
        json.dump(annotations, f)
    
    print(f"Processing complete. Generated {len(annotations)} annotations.")
    return annotations

def combine_depth_and_sam_refinement(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: List[str],
    scores: np.ndarray,
    depth_map: np.ndarray,
    sam_predictor: Any,
    depth_threshold: float = 0.2,
    min_region_size: int = 100
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Two-stage refinement using both depth information and SAM.
    
    Args:
        image: RGB image
        boxes: Detection boxes
        labels: Class labels
        scores: Confidence scores
        depth_map: Depth map
        sam_predictor: SAM predictor instance
        depth_threshold: Threshold for depth discontinuity
        min_region_size: Minimum region size
        
    Returns:
        final_boxes: Final refined boxes
        final_masks: Segmentation masks
        final_labels: Class labels
        final_scores: Confidence scores
    """
    # Stage 1: Refine with depth
    depth_boxes, depth_labels, depth_scores = depth_based_refinement(
        image, boxes, labels, scores, depth_map, depth_threshold, min_region_size
    )
    
    # Stage 2: Refine with SAM
    sam_predictor.set_image(image)
    
    # Convert boxes to format expected by SAM
    sam_boxes = torch.tensor(depth_boxes, device=sam_predictor.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        sam_boxes, (image.shape[0], image.shape[1])
    )
    
    # Generate masks
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    
    # Convert masks to numpy
    masks_np = masks.cpu().numpy()
    
    # Refine boxes based on masks
    final_boxes = []
    final_masks = []
    
    for i, mask in enumerate(masks_np):
        binary_mask = (mask[0] > 0.5).astype(np.uint8)
        final_masks.append(binary_mask)
        
        # Find contours to get refined box
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            final_boxes.append([x, y, x + w, y + h])
        else:
            # Fallback to depth box
            final_boxes.append(depth_boxes[i])
    
    return np.array(final_boxes), np.array(final_masks), depth_labels, depth_scores

# Main pipeline to process video and generate annotations
def video_pipeline1(video_path, output_path, sample_rate=30):
    # Set up models
    yolo_model, yolo_processor = setup_hf_world()
    dino_model = setup_grounding_dino()
    sam_predictor = setup_sam()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    annotations = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every n-th frame
        if frame_count % sample_rate == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Detect common objects
            common_boxes, common_labels, common_scores = detect_common_objects(
                pil_image, yolo_model, yolo_processor
            )
            
            # Save frame temporarily for Grounding DINO
            temp_path = f"{output_path}/temp_frame_{frame_count}.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Detect anomalies
            anomaly_boxes, anomaly_labels, anomaly_scores = detect_anomalies(temp_path, dino_model)
            
            # Combine detections
            all_boxes = common_boxes + anomaly_boxes.tolist()
            all_labels = common_labels + anomaly_labels
            all_scores = common_scores + anomaly_scores.tolist()
            
            # Optional: Refine with SAM
            if len(all_boxes) > 0:
                masks = refine_with_sam(pil_image, sam_predictor, all_boxes)
                
                # Convert masks to refined boxes if needed
                # ... (implementation depends on specific needs)
            
            # Save annotations in COCO format
            for box, label, score in zip(all_boxes, all_labels, all_scores):
                x1, y1, x2, y2 = box
                annotations.append({
                    "image_id": frame_count // sample_rate,
                    "frame_path": f"{output_path}/frame_{frame_count}.jpg",
                    "category": label,
                    "bbox": [x1, y1, x2-x1, y2-y1],  # [x, y, width, height]
                    "score": score
                })
            
            # Save the frame
            cv2.imwrite(f"{output_path}/frame_{frame_count}.jpg", frame)
            
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        frame_count += 1
    
    cap.release()
    
    # Save annotations to JSON
    import json
    with open(f"{output_path}/annotations.json", "w") as f:
        json.dump(annotations, f)
    
    return annotations

import requests
def test_zeroshot_objectdetection():
    #text_labels = [["person", "car", "bicycle", "motorcycle", "truck", "bus", "traffic light", "stop sign"]]
    text_labels = [["a photo of a cat", "a photo of a dog"]]
    
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    
    hf_model, hf_processor = setup_hf_world()
    dino_model, dino_processor = setup_grounding_dino()
    
    common_boxes, common_labels, common_scores = detect_common_objects(
        image, hf_model, hf_processor, text_labels, grounding = False
    )
    
    text_labels = [["a cat", "a remote control"]]
    common_boxes, common_labels, common_scores = detect_common_objects(
        image, dino_model, dino_processor, text_labels, grounding = True
    )

def main():
    """
    Main function to run the entire pipeline.
    """
    video_path = "data/SJSU_Sample_Video.mp4"
    output_path = "output/video_pipeline1"
    
    # Create output directory
    import os
    os.makedirs(output_path, exist_ok=True)
    
    test_zeroshot_objectdetection()
    
    # Run the pipeline
    #annotations = video_pipeline1(video_path, output_path)
    annotations =integrate_sam_refinement(video_path, output_path, sample_rate=5)
    
    print(f"Total annotations generated: {len(annotations)}")
    
    # Optionally convert annotations to YOLOv5/YOLOv8 format for training
    # convert_to_yolo_format(annotations, output_path)

if __name__ == "__main__":
    main()