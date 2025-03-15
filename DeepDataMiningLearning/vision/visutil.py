
from typing import List, Dict, Tuple, Any, Optional
import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
import bisect
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import torch
from matplotlib.colors import hsv_to_rgb

def _process_panoptic_segment(segment_info, panoptic_seg_np, image_size, id2label=None):
    """Process a single panoptic segment and return its information."""
    segment_id = segment_info["id"] #1
    label_id = segment_info["label_id"] #2
    score = segment_info.get("score", 1.0)
    was_fused = segment_info.get("was_fused", False)
    
    # Get label
    if id2label is not None and label_id in id2label:
        label = id2label[label_id] #'car'
    else:
        label = f"Class {label_id}"
    
    # Get binary mask and calculate area
    binary_mask = (panoptic_seg_np == segment_id) #(450, 800) True/False
    segment_area = np.sum(binary_mask)
    
    # Calculate bounding box
    y_indices, x_indices = np.where(binary_mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
        
    bbox = [
        int(np.min(x_indices)), 
        int(np.min(y_indices)), 
        int(np.max(x_indices)), 
        int(np.max(y_indices))
    ]
    
    # Determine if this is a "thing" or "stuff"
    is_thing = segment_area <= (image_size[0] * image_size[1] * 0.4)
    
    return {
        "id": segment_id,
        "label_id": label_id,
        "label": label,
        "score": float(score) if isinstance(score, (int, float, np.number)) else None,
        "area": int(segment_area),
        "bbox": bbox,
        "is_thing": is_thing,
        "was_fused": was_fused,
        "binary_mask": binary_mask
    }

def _draw_panoptic_segment(draw, segment_data, color, font, alpha=0.5):
    """Draw a single panoptic segment on the image."""
    x_min, y_min, x_max, y_max = segment_data["bbox"]
    
    # Draw bounding box for "thing" instances
    if segment_data["is_thing"]:
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
        
        # Draw label
        label_text = segment_data["label"]
        if segment_data["score"] is not None:
            label_text += f" {segment_data['score']:.2f}"
        draw.text((x_min, y_min - 12), label_text, fill=color, font=font)

def _create_panoptic_overlay(image_size, segments_data, colors):
    """Create the panoptic segmentation overlay."""
    overlay = np.zeros((*image_size, 3), dtype=np.uint8) #(450, 800, 3)
    
    for segment, color in zip(segments_data, colors):
        if segment is not None:
            mask = segment["binary_mask"]
            overlay[mask] = color
            
    return Image.fromarray(overlay)

def visualize_results(
    image,
    boxes=None, #boxes (np.ndarray or torch.Tensor, optional): Bounding boxes in format [x1, y1, x2, y2]
    labels=None,
    scores=None,
    semantic_seg=None,
    instance_seg=None,
    panoptic_seg=None,
    draw_boxes=True, #for panoptic 
    draw_masks=True,
    depth_map=None,
    class_names=None,
    colors=None,
    output_path=None,
    alpha=0.5,
    box_thickness=2,
    text_size=12,
    depth_cmap='plasma',
    show_legend=False,
    label_segments=False,  # New parameter to enable segment labeling
    label_font_size=10     # New parameter for segment label font size
):
    """
    Visualize detection and segmentation results on an image.
    
    Args:
        image (PIL.Image or np.ndarray): The original image
        boxes (np.ndarray or torch.Tensor, optional): Bounding boxes in format [x1, y1, x2, y2]
        labels (np.ndarray or torch.Tensor, optional): Class labels for each box
        scores (np.ndarray or torch.Tensor, optional): Confidence scores for each box
        semantic_seg (np.ndarray or torch.Tensor, optional): Semantic segmentation map
        instance_seg (np.ndarray or torch.Tensor, optional): Instance segmentation map
        panoptic_seg (dict, optional): Dict with 'segments_info' and 'panoptic_seg' keys
        depth_map (np.ndarray or torch.Tensor, optional): Depth map
        class_names (list, optional): List of class names
        colors (dict, optional): Dict mapping class IDs to RGB tuples
        output_path (str, optional): Path to save the visualization
        alpha (float, optional): Transparency of segmentation overlay
        box_thickness (int, optional): Thickness of bounding box lines
        text_size (int, optional): Size of text for labels
        depth_cmap (str, optional): Matplotlib colormap for depth visualization
        show_legend (bool, optional): Whether to show a legend for classes
        label_segments (bool, optional): Whether to add labels to segment centroids
        label_font_size (int, optional): Font size for segment labels
        
    Returns:
        PIL.Image: The visualization result
    """
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            image = image[0]  # Take first image if batch
        image = image.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray((image * 255).astype('uint8'))
    
    # Create a copy for drawing
    result_img = image.copy()
    
    # Convert tensors to numpy if needed
    if boxes is not None and isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if semantic_seg is not None and isinstance(semantic_seg, torch.Tensor):
        semantic_seg = semantic_seg.cpu().numpy()
    if instance_seg is not None and isinstance(instance_seg, torch.Tensor):
        instance_seg = instance_seg.cpu().numpy()
    if depth_map is not None and isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if panoptic_seg is not None and 'panoptic_seg' in panoptic_seg and isinstance(panoptic_seg['panoptic_seg'], torch.Tensor):
        panoptic_seg['panoptic_seg'] = panoptic_seg['panoptic_seg'].cpu().numpy()
    
    # Generate colors if not provided
    if colors is None and (labels is not None or semantic_seg is not None or instance_seg is not None or panoptic_seg is not None):
        max_classes = 0
        if labels is not None:
            max_classes = max(max_classes, len(labels) + 1)
        if semantic_seg is not None:
            max_classes = max(max_classes, np.max(semantic_seg) + 1)
        if panoptic_seg is not None and 'segments_info' in panoptic_seg:
            #max_classes = max(max_classes, max([s['category_id'] for s in panoptic_seg['segments_info']]) + 1)
            max_classes = max(max_classes, max([s['label_id'] for s in panoptic_seg['segments_info']]) + 1)
        
        # Ensure we have at least 10 colors even if max_classes is smaller
        max_classes = max(10, max_classes)
        
        # colors = {}
        # for i in range(max_classes):
        #     colors[i] = (
        #         int((i * 37 + 142) % 255),
        #         int((i * 91 + 89) % 255),
        #         int((i * 173 + 127) % 255)
        #     )
        # Generate random colors for segmentation masks
        def get_random_colors(n):
            colors = []
            for i in range(n):
                # Use HSV color space for better visual distinction
                hue = i / n
                saturation = 0.9
                value = 0.9
                rgb = hsv_to_rgb((hue, saturation, value))
                colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
            return colors
        
        colors = get_random_colors(max_classes)
    
    # Handle different segmentation types
    segmentation_overlay = None
    segment_centroids = []  # Store centroids for labeling
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", text_size)
        segment_font = ImageFont.truetype("arial.ttf", label_font_size)
    except IOError:
        font = ImageFont.load_default()
        segment_font = ImageFont.load_default()
    
    # Semantic Segmentation
    if semantic_seg is not None:
        # Create a colored segmentation map
        seg_colored = np.zeros((semantic_seg.shape[0], semantic_seg.shape[1], 3), dtype=np.uint8)
        
        # Store class centroids for labeling
        if label_segments:
            for class_id in np.unique(semantic_seg):
                if class_id == 0 and len(np.unique(semantic_seg)) > 1:  # Skip background if it's 0
                    continue
                
                mask = semantic_seg == class_id
                # Calculate centroid
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0:
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    
                    # Get class name
                    if class_names is not None and class_id < len(class_names):
                        label_text = class_names[class_id]
                    else:
                        label_text = f"Class {class_id}"
                    
                    segment_centroids.append({
                        'x': center_x,
                        'y': center_y,
                        'label': label_text,
                        'color': colors.get(class_id, (255, 255, 255))
                    })
                
                # Color the mask
                color = colors.get(class_id, (255, 255, 255))  # Default to white if no color
                seg_colored[mask] = color
        else:
            # Just color the segments without calculating centroids
            for class_id in np.unique(semantic_seg):
                if class_id == 0 and len(np.unique(semantic_seg)) > 1:  # Skip background if it's 0
                    continue
                mask = semantic_seg == class_id
                color = colors.get(class_id, (255, 255, 255))  # Default to white if no color
                seg_colored[mask] = color
        
        segmentation_overlay = Image.fromarray(seg_colored)
    
    # Instance Segmentation
    if instance_seg is not None:
        # Create a colored instance map
        instance_colored = np.zeros((instance_seg.shape[0], instance_seg.shape[1], 3), dtype=np.uint8)
        
        # Process each instance
        for instance_id in np.unique(instance_seg):
            if instance_id == 0:  # Skip background
                continue
                
            mask = instance_seg == instance_id
            
            # If labels are provided, use the class color, otherwise random color based on instance_id
            class_id = labels[instance_id - 1] if labels is not None and instance_id <= len(labels) else instance_id
            color = colors.get(class_id, colors.get(instance_id % len(colors), (255, 255, 255)))
            
            # Color the mask
            instance_colored[mask] = color
            
            # Store instance centroid for labeling
            if label_segments:
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0:
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    
                    # Get instance label
                    if class_names is not None and class_id < len(class_names):
                        label_text = f"{class_names[class_id]} {instance_id}"
                    else:
                        label_text = f"Instance {instance_id}"
                    
                    segment_centroids.append({
                        'x': center_x,
                        'y': center_y,
                        'label': label_text,
                        'color': color
                    })
        
        # If we already have semantic seg, blend them
        if segmentation_overlay is not None:
            instance_overlay = Image.fromarray(instance_colored)
            # Blend semantic and instance with equal weight
            segmentation_overlay = Image.blend(segmentation_overlay, instance_overlay, 0.5)
        else:
            segmentation_overlay = Image.fromarray(instance_colored)
    
    # Panoptic Segmentation
    # if panoptic_seg is not None and 'panoptic_seg' in panoptic_seg and 'segments_info' in panoptic_seg:
    #     panoptic_colored = np.zeros((panoptic_seg['panoptic_seg'].shape[0], panoptic_seg['panoptic_seg'].shape[1], 3), dtype=np.uint8)
        
    #     for segment in panoptic_seg['segments_info']:
    #         segment_id = segment['id']
    #         category_id = segment['category_id']
    #         mask = panoptic_seg['panoptic_seg'] == segment_id
            
    #         # Get color for this category
    #         color = colors.get(category_id, (255, 255, 255))
            
    #         # Color the mask
    #         panoptic_colored[mask] = color
            
    #         # Store segment centroid for labeling
    #         if label_segments:
    #             y_indices, x_indices = np.where(mask)
    #             if len(y_indices) > 0:
    #                 center_x = int(np.mean(x_indices))
    #                 center_y = int(np.mean(y_indices))
                    
    #                 # Get category name
    #                 if class_names is not None and category_id < len(class_names):
    #                     label_text = class_names[category_id]
    #                 else:
    #                     label_text = f"Category {category_id}"
                    
    #                 # Add instance ID for things (not stuff)
    #                 if 'isthing' in segment and segment['isthing']:
    #                     label_text = f"{label_text} {segment_id}"
                    
    #                 segment_centroids.append({
    #                     'x': center_x,
    #                     'y': center_y,
    #                     'label': label_text,
    #                     'color': color
    #                 })
        
    #     panoptic_overlay = Image.fromarray(panoptic_colored)
        
    #     # If we already have other segmentation, blend them
    #     if segmentation_overlay is not None:
    #         segmentation_overlay = Image.blend(segmentation_overlay, panoptic_overlay, 0.5)
    #     else:
    #         segmentation_overlay = panoptic_overlay
    # Process panoptic segmentation
        # Process panoptic segmentation
    if panoptic_seg is not None and 'panoptic_seg' in panoptic_seg and 'segments_info' in panoptic_seg:
        # Process all segments
        panoptic_seg_np = panoptic_seg['panoptic_seg'] #(450, 800)
        segments_info = panoptic_seg['segments_info'] #list
        id2label = {i: label for i, label in enumerate(labels)} if labels is not None else None
        segments_data = [
            _process_panoptic_segment(
                segment_info, 
                panoptic_seg_np, 
                (result_img.height, result_img.width),
                id2label
            )
            for segment_info in segments_info
        ] #get bbox
        
        # Create overlay if masks should be drawn
        if draw_masks:
            overlay = _create_panoptic_overlay(
                (result_img.height, result_img.width),
                segments_data,
                colors
            )
            # Convert overlay to PIL Image and store as segmentation_overlay
            if segmentation_overlay is not None:
                # Blend with existing overlay
                panoptic_overlay = overlay
                segmentation_overlay = Image.blend(segmentation_overlay, panoptic_overlay, 0.5)
            else:
                segmentation_overlay = overlay
        #segmentation_overlay.save("output/testseg.png")
        # Draw bounding boxes and labels if requested
        if draw_boxes:
            draw = ImageDraw.Draw(result_img)
            for segment, color in zip(segments_data, colors):
                if segment is not None:
                    _draw_panoptic_segment(draw, segment, color, font, alpha)


    # Apply segmentation overlay
    if segmentation_overlay is not None:
        # Resize if needed
        if segmentation_overlay.size != result_img.size:
            segmentation_overlay = segmentation_overlay.resize(result_img.size, Image.NEAREST)
        
        # Blend with original image
        result_img = Image.blend(result_img, segmentation_overlay, alpha)
        
        # Add segment labels at centroids if requested
        if label_segments and segment_centroids:
            draw = ImageDraw.Draw(result_img)
            
            for centroid in segment_centroids:
                # Draw text with outline for better visibility
                # First draw black outline
                for offset_x, offset_y in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    draw.text(
                        (centroid['x'] + offset_x, centroid['y'] + offset_y),
                        centroid['label'],
                        font=segment_font,
                        fill=(0, 0, 0)
                    )
                
                # Then draw text in white or contrasting color
                # Choose white or black text based on background color brightness
                r, g, b = centroid['color']
                brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                text_color = (0, 0, 0) if brightness > 0.5 else (255, 255, 255)
                
                draw.text(
                    (centroid['x'], centroid['y']),
                    centroid['label'],
                    font=segment_font,
                    fill=text_color
                )
    
    # Depth map visualization
    if depth_map is not None:
        # Normalize the depth map
        depth_norm = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-8)
        
        # Apply colormap
        depth_colored = (cm.get_cmap(depth_cmap)(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        depth_overlay = Image.fromarray(depth_colored)
        
        # Resize if needed
        if depth_overlay.size != result_img.size:
            depth_overlay = depth_overlay.resize(result_img.size, Image.NEAREST)
        
        # Create a composite image with depth map
        if segmentation_overlay is not None:
            # We already have segmentation, create a separate depth visualization
            # Create a new figure for side-by-side visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(np.array(result_img))
            ax1.set_title("Detection & Segmentation")
            ax1.axis('off')
            
            ax2.imshow(np.array(depth_overlay))
            ax2.set_title("Depth Map")
            ax2.axis('off')
            
            plt.tight_layout()
            
            if output_path:
                # Save the figure
                directory = os.path.dirname(output_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                    
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            
            # Return the first visualization as the primary result
            # We'll continue with result_img unchanged
        else:
            # No segmentation, we can blend depth with the original
            result_img = Image.blend(result_img, depth_overlay, alpha)
    
    # Draw bounding boxes and labels
    if boxes is not None:
        draw = ImageDraw.Draw(result_img)
        
        for i in range(len(boxes)):
            # Get box coordinates
            box = boxes[i]
            x1, y1, x2, y2 = box.astype(int) if isinstance(box, np.ndarray) else map(int, box)
            
            # Get class info
            class_id = int(labels[i]) if labels is not None else 0
            score = scores[i] if scores is not None else None
            color = colors.get(class_id, (255, 0, 0))
            
            # Prepare label text
            if class_names is not None and class_id < len(class_names):
                label_text = class_names[class_id]
            else:
                label_text = f"Class {class_id}"
                
            if score is not None:
                label_text = f"{label_text}: {score:.2f}"
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)
            
            # Draw label background
            text_width, text_height = draw.textbbox((0, 0), label_text, font=font)[2:]
            draw.rectangle([x1, y1, x1 + text_width, y1 + text_height], fill=color)
            
            # Draw label text
            draw.text((x1, y1), label_text, fill=(255, 255, 255), font=font)
    
    # Add legend if requested
    if show_legend and class_names is not None and (labels is not None or semantic_seg is not None or panoptic_seg is not None):
        legend_width = 200
        legend_height = len(class_names) * 25
        legend_img = Image.new('RGB', (legend_width, legend_height), (255, 255, 255))
        legend_draw = ImageDraw.Draw(legend_img)
        
        for i, name in enumerate(class_names):
            if i >= len(class_names):
                break
                
            color = colors.get(i, (255, 0, 0))
            y_pos = i * 25 + 5
            legend_draw.rectangle([5, y_pos, 20, y_pos + 15], fill=color)
            legend_draw.text((30, y_pos), name, fill=(0, 0, 0), font=font)
        
        # Create a composite image with the legend
        composite_width = result_img.width + legend_img.width
        composite_height = max(result_img.height, legend_img.height)
        composite = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))
        composite.paste(result_img, (0, 0))
        composite.paste(legend_img, (result_img.width, 0))
        result_img = composite
    
    # Save the result if output path is provided
    if output_path and depth_map is None:  # If depth map is present, we've already saved
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        result_img.save(output_path)
    
    return result_img

def draw_segmentation(pred_seg):
    pred_seg_np = pred_seg.cpu().numpy().astype(np.uint8) #ensure correct data type for image creation
    # Create a PIL image from the segmentation mask
    seg_image = Image.fromarray(pred_seg_np)
    # Optionally, you can colorize the segmentation mask for better visualization
    if len(torch.unique(pred_seg)) <= 20: #if there is a small number of classes, colorize.
        num_classes = len(torch.unique(pred_seg))
        palette = []
        for i in range(num_classes):
            # Generate a color for each class (simple RGB values)
            r = (i * 37) % 255
            g = (i * 91) % 255
            b = (i * 173) % 255
            palette.extend([r, g, b])
        seg_image.putpalette(palette)
        seg_image = seg_image.convert("RGB") #convert to RGB to be saved as png.
    
    # Save the segmentation result
    seg_image.save("data/segmentation_testresult.jpg")
        
#bounding box visualization functions
def visualize_bboxsegresults(
    image: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
    labels: List[str],
    scores: np.ndarray,
    output_path: str = "object_detection.jpg",
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

def visualize_bbox(image, results, id2label):
    pilimage=Image.fromarray(image)#numpy HWC
    draw = ImageDraw.Draw(pilimage)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1) #[xmin, ymin, xmax, ymax]
        draw.text((x, y), id2label[str(int(label.item()))], fill="white")
    pilimage.save("output/ImageDraw.png")
    return pilimage

#ref: https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/util/inference.py
def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame
