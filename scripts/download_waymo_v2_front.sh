#!/bin/bash
# Download Waymo Open Dataset v2 - only FRONT camera (2D detection subset)

# ======== CONFIG ========
SAVE_DIR=/data/Datasets/WaymoV2_FRONT
BUCKET=gs://waymo_open_dataset_v_2_0_1/training
#waymo_open_dataset_v_2_0_1/training/camera_image
# ========================

# Create local folders
mkdir -p $SAVE_DIR/camera_image
mkdir -p $SAVE_DIR/camera_box
mkdir -p $SAVE_DIR/camera_calibration

echo "✅ Downloading Waymo V2 FRONT camera data into $SAVE_DIR"

# Download FRONT camera images
echo "➡️ Downloading camera_image (FRONT) ..."
gsutil -m cp $BUCKET/camera_image/camera_FRONT.parquet $SAVE_DIR/camera_image/

# Download FRONT camera boxes
echo "➡️ Downloading camera_box (FRONT) ..."
gsutil -m cp $BUCKET/camera_box/camera_FRONT.parquet $SAVE_DIR/camera_box/

# Download FRONT camera calibration
echo "➡️ Downloading camera_calibration (FRONT) ..."
gsutil -m cp $BUCKET/camera_calibration/camera_FRONT.parquet $SAVE_DIR/camera_calibration/

echo "🎉 Done! FRONT camera files saved in: $SAVE_DIR"