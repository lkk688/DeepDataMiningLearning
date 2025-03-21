import cv2
import os
import datetime
import subprocess
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from fractions import Fraction
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import glob
from transformers import AutoProcessor, AutoImageProcessor, AutoModelForUniversalSegmentation, AutoModelForZeroShotObjectDetection

from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#draw_panoptic_segmentation(**panoptic_segmentation)
def draw_panoptic_segmentation(segmentation, segments_info, id2label):
    segmentation = segmentation.cpu()
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    fig, ax = plt.subplots()
    ax.imshow(segmentation) #[450, 800]
    instances_counter = defaultdict(int)
    handles = []
    # for each segment, draw its legend
    for segment in segments_info:
        segment_id = segment['id'] #1
        segment_label_id = segment['label_id'] #
        segment_label = id2label[segment_label_id] #'car'
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))
        
    ax.legend(handles=handles)
    plt.savefig('output/panoptic_test1.png')

def draw_semantic_segmentation(segmentation, id2label):
    segmentation = segmentation.cpu()
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    # get all the unique numbers
    labels_ids = torch.unique(segmentation).tolist()
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    handles = []
    for label_id in labels_ids:
        label = model.config.id2label[label_id]
        color = viridis(label_id)
        handles.append(mpatches.Patch(color=color, label=label))
    ax.legend(handles=handles)
    plt.savefig('output/semantic_test1.png')

def extract_metadata(video_path):
    """Extract metadata including GPS information from video file using ffprobe."""
    cmd = [
        'ffprobe', 
        '-v', 'quiet', 
        '-print_format', 'json', 
        '-show_format', 
        '-show_streams', 
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        metadata = json.loads(result.stdout)
        
        # Extract GPS data if available
        gps_info = {}
        creation_time = None
        
        # Look for creation time and GPS data in format tags
        if 'format' in metadata and 'tags' in metadata['format']:
            tags = metadata['format']['tags']
            
            # Extract creation time
            time_fields = ['creation_time', 'date', 'com.apple.quicktime.creationdate']
            for field in time_fields:
                if field in tags:
                    creation_time = tags[field]
                    break
            
            # Common GPS metadata fields
            gps_fields = [
                'location', 'location-eng', 'GPS', 
                'GPSLatitude', 'GPSLongitude', 'GPSAltitude',
                'com.apple.quicktime.location.ISO6709'
            ]
            
            for field in gps_fields:
                if field in tags:
                    gps_info[field] = tags[field]
        
        # Also check stream metadata for creation time if not found
        if creation_time is None and 'streams' in metadata:
            for stream in metadata['streams']:
                if 'tags' in stream and 'creation_time' in stream['tags']:
                    creation_time = stream['tags']['creation_time']
                    break
        
        # If no creation time found, use file modification time
        if creation_time is None:
            file_mtime = os.path.getmtime(video_path)
            creation_time = datetime.datetime.fromtimestamp(file_mtime).isoformat()
        
        return metadata, gps_info, creation_time
    
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None, {}, None

def resize_with_aspect_ratio(image, target_size):
    """
    Resize image maintaining aspect ratio.
    
    Parameters:
    - image: PIL Image or numpy array
    - target_size: Tuple of (width, height) representing the maximum dimensions
    
    Returns:
    - Resized PIL Image
    """
    if isinstance(image, np.ndarray):
        # Convert OpenCV image (numpy array) to PIL
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Get original dimensions
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    # Calculate aspect ratios
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height
    
    # Determine new dimensions maintaining aspect ratio
    if original_aspect > target_aspect:
        # Width constrained
        new_width = target_width
        new_height = int(target_width / original_aspect)
    else:
        # Height constrained
        new_height = target_height
        new_width = int(target_height * original_aspect)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_image

def extract_key_frames(video_path, output_dir, target_size=(640, 480), extraction_method="scene_change"):
    """
    Extract key frames from a video file and save them with timestamp names.
    
    Parameters:
    - video_path: Path to the input video file
    - output_dir: Directory to save extracted frames
    - target_size: Tuple of (width, height) maximum dimensions for resizing
    - extraction_method: Method to extract frames ('scene_change', 'interval', or 'both')
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get video metadata
    metadata, gps_info, creation_time = extract_metadata(video_path)
    
    # Save metadata to a JSON file
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            'video_metadata': metadata, 
            'gps_info': gps_info, 
            'creation_time': creation_time
        }, f, indent=4)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate video creation time based on metadata or file timestamp
    video_creation_datetime = None
    if creation_time:
        try:
            # Try different time formats
            for time_format in [
                "%Y-%m-%dT%H:%M:%S.%fZ", 
                "%Y-%m-%dT%H:%M:%SZ", 
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"
            ]:
                try:
                    video_creation_datetime = datetime.datetime.strptime(creation_time, time_format)
                    break
                except ValueError:
                    continue
        except:
            pass  # Use None if parsing fails
    
    print(f"Video Information:")
    print(f"- Frame Rate: {fps} fps")
    print(f"- Frame Count: {frame_count}")
    print(f"- Resolution: {original_width}x{original_height}")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"- Creation Time: {creation_time}")
    print(f"- GPS Info: {gps_info}")
    
    # Initialize variables
    prev_frame = None
    frame_idx = 0
    saved_count = 0
    
    # Parameters for scene change detection
    min_scene_change_threshold = 30.0  # Minimum threshold for scene change
    frame_interval = int(fps) * 1  # Save a frame every second as fallback
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale for scene change detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        should_save = False
        reason = ""
        
        # Method 1: Detect scene changes
        if extraction_method in ["scene_change", "both"]:
            if prev_frame is not None:
                # Calculate mean absolute difference between current and previous frame
                diff = cv2.absdiff(gray, prev_frame)
                mean_diff = np.mean(diff)
                
                if mean_diff > min_scene_change_threshold:
                    should_save = True
                    reason = f"scene_change (diff={mean_diff:.2f})"
        
        # Method 2: Save frames at regular intervals
        if extraction_method in ["interval", "both"]:
            if frame_idx % frame_interval == 0:
                should_save = True
                reason = "interval"
        
        # Save the frame if needed
        if should_save:
            # Calculate timestamp in the video
            timestamp_seconds = frame_idx / fps
            timestamp = str(datetime.timedelta(seconds=int(timestamp_seconds)))
            milliseconds = int((timestamp_seconds - int(timestamp_seconds)) * 1000)
            timestamp = f"{timestamp}.{milliseconds:03d}"
            
            # Calculate frame creation time if video creation time is available
            frame_creation_time = None
            if video_creation_datetime:
                frame_creation_time = (video_creation_datetime + 
                                      datetime.timedelta(seconds=timestamp_seconds)).isoformat()
            
            # Resize the frame maintaining aspect ratio
            pil_img = resize_with_aspect_ratio(frame, target_size)
            new_width, new_height = pil_img.size
            
            # Save the frame
            filename = f"frame_{timestamp.replace(':', '-')}_{reason}.jpg"
            output_path = os.path.join(output_dir, filename)
            pil_img.save(output_path, quality=95)
            
            # Save frame metadata
            frame_meta = {
                "frame_index": frame_idx,
                "timestamp_seconds": timestamp_seconds,
                "timestamp": timestamp,
                "extraction_reason": reason,
                "original_dimensions": {
                    "width": original_width,
                    "height": original_height
                },
                "resized_dimensions": {
                    "width": new_width,
                    "height": new_height
                },
                "video_creation_time": creation_time,
                "frame_creation_time": frame_creation_time,
                "extraction_time": datetime.datetime.now().isoformat()
            }
            
            # Add GPS data to frame metadata
            if gps_info:
                frame_meta["gps_info"] = gps_info
            
            # Save frame metadata
            frame_meta_file = os.path.join(output_dir, f"{filename.replace('.jpg', '.json')}")
            with open(frame_meta_file, 'w') as f:
                json.dump(frame_meta, f, indent=4)
            
            saved_count += 1
            print(f"Saved frame {saved_count}: {filename} ({reason})")
        
        # Update variables for next iteration
        prev_frame = gray.copy()
        frame_idx += 1
    
    # Release resources
    cap.release()
    print(f"Extraction complete. Saved {saved_count} key frames to {output_dir}")

def perform_panoptic_segmentation(frames_dir, output_dir, model_name="facebook/mask2former-swin-large-coco-panoptic"):
    """
    Perform panoptic segmentation on extracted frames and save visualizations.
    
    Parameters:
    - frames_dir: Directory containing extracted frames
    - output_dir: Directory to save segmentation results
    - model_name: Name or path of the segmentation model to use
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create subdirectories for different visualization types
    bbox_dir = os.path.join(output_dir, "bboxes")
    mask_dir = os.path.join(output_dir, "masks")
    combined_dir = os.path.join(output_dir, "combined")
    
    for directory in [bbox_dir, mask_dir, combined_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Load model and processor
    print(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForUniversalSegmentation.from_pretrained(model_name).to(device)
    
    # Get all image files in the input directory
    image_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    
    print(f"Found {len(image_files)} images to process")
    
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
    
    for img_path in image_files:
        base_filename = os.path.basename(img_path)
        base_name = os.path.splitext(base_filename)[0]
        print(f"Processing {base_filename}...")
        
        # Load image
        image = Image.open(img_path)
        input_image = image.copy()
        
        # Load associated metadata
        json_path = os.path.join(frames_dir, base_name + ".json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                frame_meta = json.load(f)
        else:
            frame_meta = {}
        
        # Preprocess the image
        inputs = processor(images=input_image, return_tensors="pt").to(device)
        #pixel_values: [1, 3, 384, 384], pixel_mask: [1, 384, 384]
        
        # Generate predictions
        with torch.no_grad():
            outputs = model(**inputs)
        #class_queries_logits [1, 200, 134],  masks_queries_logits [1, 200, 96, 96]
        
        # Post-process results
        result = processor.post_process_panoptic_segmentation(
            outputs, 
            target_sizes=[input_image.size[::-1]]
        )[0]
        
        # Extract segmentation mask and metadata
        panoptic_seg = result["segmentation"] #[450, 800]
        segments_info = result["segments_info"] #list of 15,
        print(segments_info[0].keys()) #['id', 'label_id', 'was_fused', 'score']
        
        # Convert panoptic segmentation to numpy array for easier manipulation
        panoptic_seg_np = panoptic_seg.cpu().numpy() #(450, 800) all 12
        
        # Get unique segments and class information
        segments = []
        random_colors = get_random_colors(len(segments_info))
        
        # Create visualization images
        bbox_image = input_image.copy()
        mask_image = Image.new("RGB", input_image.size, (0, 0, 0))
        combined_image = input_image.copy()
        
        draw_bbox = ImageDraw.Draw(bbox_image)
        draw_combined = ImageDraw.Draw(combined_image)
        
        # Try to load font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Draw each segment
        for i, segment_info in enumerate(segments_info):
            segment_id = segment_info["id"]
            label_id = segment_info["label_id"]#["category_id"] 2
            score = segment_info.get("score", 1.0)
            was_fused = segment_info.get("was_fused", False)
            
            # Get label from processor's id2label mapping
            if hasattr(processor, 'id2label') and label_id in processor.id2label:
                label = processor.id2label[label_id]
            else:
                label = f"Class {label_id}"
            
            # Get binary mask for this segment and calculate area
            binary_mask = (panoptic_seg_np == segment_id) #True/False (450, 800)
            segment_area = np.sum(binary_mask)
            
            # Calculate bounding box
            y_indices, x_indices = np.where(binary_mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
            else:
                # Skip segments with no visible pixels
                continue
            
            # Determine if this is a "thing" (object) or "stuff" (background)
            # This is a heuristic based on common datasets like COCO
            # Objects typically have smaller areas and higher scores
            is_thing = True
            if segment_area > (input_image.width * input_image.height * 0.4):
                # Large segments are likely background "stuff"
                is_thing = False
            
            # Store segment information
            segment_data = {
                "id": segment_id,
                "label_id": label_id,
                "label": label,
                "score": float(score) if isinstance(score, (int, float, np.number)) else None,
                "area": int(segment_area),
                "bbox": bbox,
                "is_thing": is_thing,
                "was_fused": was_fused
            }
            segments.append(segment_data)
            
            # Get color for this segment
            color = random_colors[i]
            
            # Draw bounding box
            draw_bbox.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
            if is_thing:
                label_text = f"{label}"
                if score is not None and isinstance(score, (int, float, np.number)):
                    label_text += f" {score:.2f}"
                draw_bbox.text((x_min, y_min - 12), label_text, fill=color, font=font)
            
            # Draw segmentation mask
            mask_data = np.zeros((input_image.height, input_image.width, 3), dtype=np.uint8)
            mask_data[binary_mask] = color #(450, 800, 3)
            mask_img = Image.fromarray(mask_data)
            mask_image = Image.alpha_composite(
                mask_image.convert('RGBA'), 
                Image.blend(Image.new('RGBA', mask_image.size, (0, 0, 0, 0)), 
                           mask_img.convert('RGBA'), 
                           alpha=1)
            ).convert('RGB')
            
            # Draw combined visualization (transparent mask over image)
            overlay = Image.new('RGBA', input_image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            mask_color = color + (64,)  # Add alpha value
            
            # Create mask from binary array
            for y in range(binary_mask.shape[0]):
                for x in range(binary_mask.shape[1]):
                    if binary_mask[y, x]:
                        overlay_draw.point((x, y), fill=mask_color)
            
            # Composite the overlay onto the combined image
            combined_image = Image.alpha_composite(
                combined_image.convert('RGBA'),
                overlay
            ).convert('RGB')
            
            # Draw bounding box on combined image
            draw_combined = ImageDraw.Draw(combined_image)
            draw_combined.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
            if is_thing:
                label_text = f"{label}"
                if score is not None and isinstance(score, (int, float, np.number)):
                    label_text += f" {score:.2f}"
                draw_combined.text((x_min, y_min - 12), label_text, fill=color, font=font)
        
        # Save segmentation results
        bbox_image.save(os.path.join(bbox_dir, f"{base_name}_bbox.jpg"))
        mask_image.save(os.path.join(mask_dir, f"{base_name}_mask.jpg"))
        combined_image.save(os.path.join(combined_dir, f"{base_name}_combined.jpg"))
        
        # Save segmentation metadata
        seg_meta = {
            "original_frame": base_filename,
            "segments": segments,
            "frame_metadata": frame_meta,
            "model": model_name,
            "segmentation_time": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, f"{base_name}_segmentation.json"), 'w') as f:
            json.dump(seg_meta, f, indent=4)
        
        print(f"Saved segmentation results for {base_filename}")
    
    print(f"Segmentation complete. Processed {len(image_files)} images.")


def perform_panoptic_segmentation2(frames_dir, output_dir, model_name="facebook/mask2former-swin-large-coco-panoptic", task='panoptic'):
    """
    Perform panoptic segmentation on extracted frames and save visualizations.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load model and processor
    print(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    #processor = AutoImageProcessor.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForUniversalSegmentation.from_pretrained(model_name).to(device)
    
    # Print the id2label mapping
    id2label = None
    labels = None
    if hasattr(processor, 'id2label'):
        print("ID to Label Mapping:")
        id2label = processor.id2label
    elif hasattr(model.config, 'id2label'):
        id2label=model.config.id2label
    else:
        print("No id2label mapping found in processor")
    if id2label is not None:
        labels = list(id2label.values())
        for id, label in id2label.items():
            print(f"  {id}: {label}")
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"Found {len(image_files)} images to process")
    
    for img_path in image_files:
        base_filename = os.path.basename(img_path)
        base_name = os.path.splitext(base_filename)[0]
        print(f"Processing {base_filename}...")
        
        # Load image and metadata
        image = Image.open(img_path)
        input_image = np.array(image)  # Convert to numpy array for visualization
        
        json_path = os.path.join(frames_dir, base_name + ".json")
        frame_meta = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                frame_meta = json.load(f)
        
        # Preprocess and generate predictions
        if "oneformer" in model_name:
            inputs = processor(images=image, task_inputs=[task], return_tensors="pt").to(device)
            #decode the task inputs back to text:
            checktask=processor.tokenizer.batch_decode(inputs.task_inputs) #task_inputs: [1, 77]
            print("Decode task is: ", checktask)
        else:
            inputs = processor(images=image, return_tensors="pt").to(device)
        
        for k,v in inputs.items():
            print(k,v.shape) 
        #pixel_values:[1, 3, 749, 1333], 
        # pixel_mask: [1, 749, 1333], 
        # task_inputs: [1, 77]
  
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process results
        target_size=[image.size[::-1]]
        if task == "panoptic":
            result = processor.post_process_panoptic_segmentation(
                outputs, 
                target_sizes=target_size
            )[0]
            print(result.keys()) #['segmentation', 'segments_info']
        elif task == "semantic":
            result = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=target_size
            )[0]
            #[450, 800] tensor
        elif task == "instance":
            result = processor.post_process_instance_segmentation(
                outputs,
                target_sizes=target_size
            )[0]
        else:
            raise ValueError(f"Unknown task: {task}")
        
        draw_panoptic_segmentation(result["segmentation"], result["segments_info"], id2label)
        # Prepare visualization inputs
        panoptic_result = {
            "panoptic_seg": result["segmentation"].cpu().numpy(), #(450, 800)
            "segments_info": result["segments_info"] #list of dict
        }
        
        # Use visualize_results for different visualization types
        from visutil import visualize_results
        
        # Create output paths
        bbox_path = os.path.join(output_dir, f"{base_name}_bbox.jpg")
        mask_path = os.path.join(output_dir, f"{base_name}_mask.jpg")
        combined_path = os.path.join(output_dir, f"{base_name}_combined.jpg")
        
        # Visualize bounding boxes only
        bbox_img = visualize_results(
            image=input_image.copy(),  # Use a copy to prevent modifications
            panoptic_seg=panoptic_result,
            draw_boxes=True,
            draw_masks=False,
            labels=labels,
            alpha=0.5,
            output_path=bbox_path
        )
        
        # Visualize masks only
        mask_img = visualize_results(
            image=input_image.copy(),
            panoptic_seg=panoptic_result,
            draw_boxes=False,
            draw_masks=True,
            labels=labels,
            alpha=0.7,  # Increased alpha for better mask visibility
            output_path=mask_path
        )
        
        # Visualize combined (boxes + masks)
        combined_img = visualize_results(
            image=input_image.copy(),
            panoptic_seg=panoptic_result,
            draw_boxes=True,
            draw_masks=True,
            labels=labels,
            alpha=0.5,
            output_path=combined_path
        )
        
        # Save segmentation metadata
        segments = []
        for segment_info in result["segments_info"]:
            segment_data = {
                "id": segment_info["id"],
                "label_id": segment_info["label_id"],
                "label": id2label[segment_info["label_id"]],
                "score": float(segment_info.get("score", 1.0)),
                "was_fused": segment_info.get("was_fused", False)
            }
            segments.append(segment_data)
        
        seg_meta = {
            "original_frame": base_filename,
            "segments": segments,
            "frame_metadata": frame_meta,
            "model": model_name,
            "segmentation_time": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, f"{base_name}_segmentation.json"), 'w') as f:
            json.dump(seg_meta, f, indent=4)
        
        print(f"Saved segmentation results for {base_filename}")
    
    print(f"Segmentation complete. Processed {len(image_files)} images.")

def perform_grounding_segmentation(frames_dir, output_dir, text_prompt="person, car, bicycle, motorcycle, truck, traffic light", 
                                  box_threshold=0.35, text_threshold=0.25):
    """
    Perform object detection using GroundingDINO and segmentation using Segment Anything Model (SAM).
    
    Parameters:
    - frames_dir: Directory containing extracted frames
    - output_dir: Directory to save segmentation results
    - text_prompt: Text prompt describing objects to detect (comma-separated)
    - box_threshold: Confidence threshold for bounding box predictions
    - text_threshold: Confidence threshold for text-to-box associations
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create subdirectories for different visualization types
    bbox_dir = os.path.join(output_dir, "bboxes")
    mask_dir = os.path.join(output_dir, "masks")
    combined_dir = os.path.join(output_dir, "combined")
    
    for directory in [bbox_dir, mask_dir, combined_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Load GroundingDINO model
    print("Loading GroundingDINO model...")
    from transformers import AutoProcessor, AutoModelForObjectDetection
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load GroundingDINO model and processor
    grounding_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    
    # Load SAM model
    print("Loading Segment Anything Model...")
    from transformers import SamProcessor, SamModel
    
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"Found {len(image_files)} images to process")
    
    # Generate random colors for visualization
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
    
    # Process each image
    for img_path in image_files:
        base_filename = os.path.basename(img_path)
        base_name = os.path.splitext(base_filename)[0]
        print(f"Processing {base_filename}...")
        
        # Load image and metadata
        image = Image.open(img_path)
        input_image = np.array(image)  # Convert to numpy array for visualization
        
        json_path = os.path.join(frames_dir, base_name + ".json")
        frame_meta = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                frame_meta = json.load(f)
        
        # Step 1: Object detection with GroundingDINO
        #https://huggingface.co/IDEA-Research/grounding-dino-base
        
        grounding_inputs = grounding_processor(text=text_prompt, images=image, return_tensors="pt").to(device)
        print(grounding_inputs.keys()) #['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask']
        #pixel_values: [1, 3, 750, 1333]
        with torch.no_grad():
            grounding_outputs = grounding_model(**grounding_inputs)
        
        # Convert outputs to COCO format
        target_sizes = torch.tensor([image.size[::-1]])
        #Converts the raw output of [`GroundingDinoForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        #bottom_right_x, bottom_right_y) format and get the associated text label.
        results = grounding_processor.post_process_grounded_object_detection(
            outputs=grounding_outputs,
            input_ids= grounding_inputs.input_ids,
            text_threshold=0.3,
            threshold=box_threshold
        )[0]
        print(results.keys()) #['scores', 'boxes', 'text_labels', 'labels']
        
        # Create visualization images
        bbox_image = image.copy()
        mask_image = Image.new("RGB", image.size, (0, 0, 0))
        combined_image = image.copy()
        
        draw_bbox = ImageDraw.Draw(bbox_image)
        
        # Try to load font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Store detected objects
        segments = []
        
        # Generate random colors for each detected box
        random_colors = get_random_colors(len(results["boxes"]))
        
        # Process each detected box
        for i, (box, label, score) in enumerate(zip(results["boxes"], results["labels"], results["scores"])):
            if score >= text_threshold:
                # Convert box from center format to corner format
                box = box.cpu().numpy()
                x_min, y_min, x_max, y_max = box
                
                # Draw bounding box
                color = random_colors[i]
                draw_bbox.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
                label_text = f"{label}: {score:.2f}"
                draw_bbox.text((x_min, y_min - 12), label_text, fill=color, font=font)
                
                # Step 2: Generate segmentation mask with SAM
                # Convert box to SAM format [x1, y1, x2, y2]
                #box_for_sam = torch.tensor([[x_min, y_min, x_max, y_max]]).to(device)
                box_for_sam = torch.tensor([[[x_min, y_min, x_max, y_max]]])
                
                # Prepare SAM inputs
                sam_inputs = sam_processor(
                    image, 
                    input_boxes=box_for_sam, 
                    return_tensors="pt"
                ).to(device)
                
                # Generate mask
                with torch.no_grad():
                    sam_outputs = sam_model(**sam_inputs)
                
                # Get mask predictions
                masks = sam_processor.image_processor.post_process_masks(
                    sam_outputs.pred_masks.cpu(),
                    sam_inputs["original_sizes"].cpu(),
                    sam_inputs["reshaped_input_sizes"].cpu()
                )
                
                # Get the best mask (highest IoU with the box)
                # Fix: mask shape is [C, H, W] not [H, W]
                mask = masks[0][0].numpy()  # Shape: [C, H, W]
                
                # Convert mask to binary [H, W] format if it's in [C, H, W] format
                if len(mask.shape) == 3:
                    # Take the first channel or average across channels
                    mask = mask[0]  # Now shape is [H, W]
                
                # Ensure mask has the correct dimensions
                if mask.shape[0] != image.height or mask.shape[1] != image.width:
                    print(f"Warning: Mask shape {mask.shape} doesn't match image size {image.size}")
                    # Resize mask to match image dimensions if needed
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_img = mask_img.resize(image.size)
                    mask = np.array(mask_img) > 0
                    
                # Calculate mask area
                mask_area = np.sum(mask)
                
                # Store segment information
                segment_data = {
                    "id": i,
                    "label": label,
                    "score": float(score),
                    "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                    "area": int(mask_area),
                    "is_thing": True  # All detected objects are "things"
                }
                segments.append(segment_data)
                
                # Draw mask on mask image
                mask_data = np.zeros((image.height, image.width, 3), dtype=np.uint8)
                mask_data[mask > 0] = color
                mask_img = Image.fromarray(mask_data)
                mask_image = Image.alpha_composite(
                    mask_image.convert('RGBA'), 
                    Image.blend(Image.new('RGBA', mask_image.size, (0, 0, 0, 0)), 
                               mask_img.convert('RGBA'), 
                               alpha=1)
                ).convert('RGB')
                
                # Draw combined visualization (transparent mask over image)
                overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                mask_color = color + (64,)  # Add alpha value
                
                # Create mask from binary array - more efficient approach
                mask_indices = np.where(mask > 0)
                for y, x in zip(mask_indices[0], mask_indices[1]):
                    if 0 <= y < image.height and 0 <= x < image.width:
                        overlay_draw.point((x, y), fill=mask_color)
                
                # Composite the overlay onto the combined image
                combined_image = Image.alpha_composite(
                    combined_image.convert('RGBA'),
                    overlay
                ).convert('RGB')
                
                # Draw bounding box on combined image with thicker lines for visibility
                draw_combined = ImageDraw.Draw(combined_image)
                draw_combined.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
                draw_combined.text((x_min, y_min - 12), label_text, fill=color, font=font)
        
        # Save visualization results
        bbox_image.save(os.path.join(bbox_dir, f"{base_name}_bbox.jpg"))
        mask_image.save(os.path.join(mask_dir, f"{base_name}_mask.jpg"))
        combined_image.save(os.path.join(combined_dir, f"{base_name}_combined.jpg"))
        
        # Save segmentation metadata
        seg_meta = {
            "original_frame": base_filename,
            "segments": segments,
            "frame_metadata": frame_meta,
            "model": "GroundingDINO + SAM",
            "text_prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "segmentation_time": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, f"{base_name}_segmentation.json"), 'w') as f:
            json.dump(seg_meta, f, indent=4)
        
        print(f"Saved segmentation results for {base_filename}")
    
    print(f"Segmentation complete. Processed {len(image_files)} images.")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract key frames and perform panoptic segmentation')
    parser.add_argument('--video_path', type=str, default='data/SJSU_Sample_Video.mp4', help='Path to the input video file')
    parser.add_argument('--frames_dir', type=str, default='output/extracted_frames_frames_20250313_173155', help='Directory containing already extracted frames (skip video extraction)')
    parser.add_argument('--output_dir', type=str, default='output/output_frames', help='Directory to save extracted frames')
    parser.add_argument('--max_width', type=int, default=800, help='Maximum width to resize frames to')
    parser.add_argument('--max_height', type=int, default=800, help='Maximum height to resize frames to')
    parser.add_argument('--method', type=str, default='both', choices=['scene_change', 'interval', 'both'], 
                        help='Method to extract frames: scene_change, interval, or both')
    parser.add_argument('--segmentation_model', type=str, default="shi-labs/oneformer_coco_swin_large",
                        help='HuggingFace model to use for panoptic segmentation, facebook/mask2former-swin-large-cityscapes-panoptic, facebook/mask2former-swin-large-coco-panoptic')
    parser.add_argument('--segmentation_type', type=str, default='universal', 
                    choices=['universal', 'grounding'], 
                    help='Type of segmentation to perform')
    parser.add_argument('--task', type=str, default='semantic', 
                    choices=['semantic', 'instance', 'panoptic'], 
                    help='Type of tasks to perform')#task = "semantic" #task_inputs=["panoptic"], "instance" semantic
    parser.add_argument('--text_prompt', type=str, 
                    default='person, car, bicycle, motorcycle, truck, traffic light', 
                    help='Text prompt for GroundingDINO (comma-separated objects)')
    parser.add_argument('--skip_extraction', default=True, help='Skip video extraction and use existing frames') #action='store_true'
    parser.add_argument('--skip_segmentation', action='store_true', help='Skip segmentation and only extract frames')
    
    args = parser.parse_args()
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process video if provided and not skipped
    if args.video_path and not args.skip_extraction:
        frames_dir = f"{args.output_dir}_frames_{timestamp}"
        extract_key_frames(
            args.video_path, 
            frames_dir, 
            target_size=(args.max_width, args.max_height),
            extraction_method=args.method
        )
    else:
        frames_dir = args.frames_dir
    
    # Perform segmentation if not skipped
    if not args.skip_segmentation:
        seg_output_dir = f"{args.output_dir}_segmentation_{timestamp}"
        if args.segmentation_type == "universal":
            perform_panoptic_segmentation2(
                frames_dir,
                seg_output_dir,
                model_name=args.segmentation_model,
                task=args.task
            )
        else:
            perform_grounding_segmentation(frames_dir, seg_output_dir, text_prompt="person, car, bicycle, motorcycle, truck, traffic light", 
                                  box_threshold=0.35, text_threshold=0.25)