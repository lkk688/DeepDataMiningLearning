import cv2
import os
import datetime
import subprocess
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from fractions import Fraction
import time
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import glob

def create_video_from_images(images_dir, output_path, pattern="*_combined.jpg", fps=10, output_type="video"):
    """
    Combines images with matching pattern into a video or GIF.
    
    Parameters:
    - images_dir: Directory containing the images
    - output_path: Path to save the output video or GIF
    - pattern: Glob pattern to match image files (default: "*_combined.jpg")
    - fps: Frames per second for the output video (default: 10)
    - output_type: "video" or "gif" (default: "video")
    
    Returns:
    - Path to the created video/GIF file
    """
    # Get all matching image files
    image_files = sorted(glob.glob(os.path.join(images_dir, pattern)))
    
    if not image_files:
        print(f"No images found matching pattern '{pattern}' in {images_dir}")
        return None
    
    print(f"Found {len(image_files)} images to process")
    
    # Extract timestamps from filenames and sort by timestamp
    def extract_timestamp(filename):
        # Extract the timestamp part from the filename
        # Assuming format like "frame_00-00-00.000_reason_suffix.jpg"
        base_name = os.path.basename(filename)
        # Extract the timestamp part between "frame_" and "_"
        match = base_name.split("frame_")[1].split("_")[0] if "frame_" in base_name else base_name
        return match
    
    # Sort files by timestamp
    image_files.sort(key=extract_timestamp)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if output_type.lower() == "gif":
        # Create GIF
        print(f"Creating GIF from {len(image_files)} images...")
        images = []
        for img_path in image_files:
            img = Image.open(img_path)
            images.append(img)
        
        # Calculate duration for each frame (in milliseconds)
        duration = int(1000 / fps)
        
        # Save as GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=duration,
            loop=0
        )
        print(f"GIF created successfully: {output_path}")
    
    else:  # Default to video
        # Get dimensions from first image
        first_img = cv2.imread(image_files[0])
        height, width, _ = first_img.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for AVI
        video_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Creating video from {len(image_files)} images...")
        for img_path in image_files:
            frame = cv2.imread(img_path)
            video_out.write(frame)
        
        # Release resources
        video_out.release()
        print(f"Video created successfully: {output_path}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract key frames and perform panoptic segmentation')
    parser.add_argument('--folder_path', type=str, default='output/extracted_frames_segmentation_20250314_110219', help='Path to the input folder file')
    parser.add_argument('--output_path', type=str, default='output/output.mp4', help='Path to the output video file')
    parser.add_argument('--outputformat', type=str, default='mp4', choices=['gif', 'mp4'], 
                        help='Output format of the video')

    args = parser.parse_args()

    create_video_from_images(
        images_dir=args.folder_path,
        output_path=args.output_path,
        pattern="*_combined.jpg",
        fps=10,
        output_type="video"
    )