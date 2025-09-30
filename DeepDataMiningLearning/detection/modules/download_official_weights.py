#!/usr/bin/env python3
"""
Download official YOLOv8 weights for compatibility testing.
"""

import os
import urllib.request
from pathlib import Path

def download_file(url, filename):
    """Download file from URL."""
    print(f"📥 Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded {filename} successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def main():
    """Download official YOLOv8 weights."""
    print("🔽 Downloading Official YOLOv8 Weights")
    print("="*50)
    
    # Official YOLOv8 weight URLs
    weights = {
        'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
    }
    
    # Create weights directory if it doesn't exist
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    
    success_count = 0
    for filename, url in weights.items():
        filepath = weights_dir / filename
        
        # Skip if file already exists
        if filepath.exists():
            print(f"✓ {filename} already exists, skipping...")
            success_count += 1
            continue
        
        if download_file(url, str(filepath)):
            success_count += 1
    
    print(f"\n📊 Summary: {success_count}/{len(weights)} weights downloaded successfully")
    
    if success_count > 0:
        print("\n🎯 Ready to run compatibility analysis!")
        print("Run: python debug_weight_compatibility.py")

if __name__ == "__main__":
    main()