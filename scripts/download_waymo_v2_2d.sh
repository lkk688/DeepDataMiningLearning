#!/bin/bash
# Download full Waymo v2.1 2D detection dataset (all parquet files in target folders)

# ======== CONFIG ========
SAVE_DIR=/data/Datasets/WaymoV2_1_2D
BUCKET=gs://waymo_open_dataset_v_2_0_1
SPLIT=training   # can be training / validation / testing
# ========================

# Create local folders
mkdir -p $SAVE_DIR/$SPLIT/camera_image
mkdir -p $SAVE_DIR/$SPLIT/camera_box
mkdir -p $SAVE_DIR/$SPLIT/camera_calibration

echo "✅ Downloading Waymo v2.1 2D detection data ($SPLIT split) into $SAVE_DIR"

# Download full camera_image folder
echo "➡️ Downloading camera_image ..."
#gsutil -m cp -r $BUCKET/$SPLIT/camera_image/* $SAVE_DIR/$SPLIT/camera_image/
#gsutil -m cp -c -r $BUCKET/$SPLIT/camera_image/* $SAVE_DIR/$SPLIT/camera_image/

# 列出 camera_image 下的文件，取前 10 个
gsutil ls $BUCKET/$SPLIT/camera_image/ | head -n 10 > filelist.txt

# 批量下载这 10 个文件
gsutil -m cp -c -I $SAVE_DIR/$SPLIT/camera_image/ < filelist.txt

# # Download full camera_box folder
# echo "➡️ Downloading camera_box ..."
# gsutil -m cp -r $BUCKET/$SPLIT/camera_box/* $SAVE_DIR/$SPLIT/camera_box/

# # Download full camera_calibration folder
# echo "➡️ Downloading camera_calibration ..."
# gsutil -m cp -r $BUCKET/$SPLIT/camera_calibration/* $SAVE_DIR/$SPLIT/camera_calibration/

# echo "🎉 Done! Waymo v2.1 2D detection data for $SPLIT saved to: $SAVE_DIR"