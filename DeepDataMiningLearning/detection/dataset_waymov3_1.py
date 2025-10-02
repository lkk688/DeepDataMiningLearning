import os
import cv2
import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import pyarrow as pa

import json
import shutil
from pathlib import Path
from PIL import Image

import cv2
import json
from pathlib import Path

# ---------- helpers ----------

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import cv2

def _parse_intrinsic(intrinsic_raw):
    """
    Parse intrinsic calibration from Waymo v2.1 parquet field.
    Expect 9 floats → reshape (3,3).
    """
    if intrinsic_raw is None:
        return np.eye(3, dtype=np.float32)

    arr = np.array(intrinsic_raw, dtype=np.float32).ravel()
    if arr.size == 9:
        return arr.reshape(3, 3)
    else:
        print(f"⚠️ Unexpected intrinsic size {arr.size}, using identity.")
        return np.eye(3, dtype=np.float32)


def _parse_extrinsic(extrinsic_raw):
    """
    Parse extrinsic calibration from Waymo v2.1 parquet field.
    Expect 16 floats → reshape (4,4).
    """
    if extrinsic_raw is None:
        return np.eye(4, dtype=np.float32)

    arr = np.array(extrinsic_raw, dtype=np.float32).ravel()
    if arr.size == 16:
        return arr.reshape(4, 4)
    else:
        print(f"⚠️ Unexpected extrinsic size {arr.size}, using identity.")
        return np.eye(4, dtype=np.float32)

def _guess_image_column_name(pf: pq.ParquetFile) -> str:
    """
    Heuristically find the image-bytes column in Waymo v2.1 camera_image parquet.
    """
    candidates = []
    for field in pf.schema_arrow:   # ✅ use schema_arrow instead of schema
        t = field.type
        if t.id in (pa.binary().id, pa.large_binary().id):
            name = field.name.lower()
            if any(k in name for k in ["image", "jpeg", "jpg", "png", "encoded"]):
                candidates.append(field.name)

    if candidates:
        candidates.sort(key=lambda n: (len(n), n))
        return candidates[0]

    # fallback: any binary column
    for field in pf.schema_arrow:
        t = field.type
        if t.id in (pa.binary().id, pa.large_binary().id):
            return field.name

    raise KeyError(
        f"❌ Could not find an image-bytes column. Schema: {[f'{f.name}:{f.type}' for f in pf.schema_arrow]}"
    )


def load_image_from_parquet(parquet_path: str, row_idx: int = 0) -> np.ndarray:
    """
    Load one image from a Waymo v2.1 camera_image parquet file.

    Args:
        parquet_path (str): Path to .parquet file
        row_idx (int): Row index (frame index) to read

    Returns:
        np.ndarray: Decoded RGB image (H, W, 3), dtype=uint8
    """
    pf = pq.ParquetFile(parquet_path)

    # Find the correct column dynamically
    image_col = _guess_image_column_name(pf)

    # Read just that row and column (efficient)
    table = pf.read_row_groups([0], columns=[image_col]).to_pandas()

    if row_idx >= len(table):
        raise IndexError(f"Row {row_idx} out of range for parquet file with {len(table)} rows.")

    img_bytes = table[image_col].iloc[row_idx]
    if not isinstance(img_bytes, (bytes, bytearray)):
        raise ValueError(f"Row {row_idx} does not contain valid image bytes.")

    # Decode compressed image bytes (JPEG/PNG) → BGR
    img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"cv2.imdecode() failed for row {row_idx} in {parquet_path}")

    # Convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def _read_first_image_bytes(pf: pq.ParquetFile, image_col: str) -> bytes:
    """
    Iterate row-groups and return the first non-null image bytes from `image_col`.
    Avoids loading the whole table.
    """
    num_rgs = pf.num_row_groups
    for rg in range(num_rgs):
        tbl = pf.read_row_group(rg, columns=[image_col])
        # Convert a single column to Python list of bytes/None efficiently
        col = tbl.column(image_col).to_pylist()
        for v in col:
            if v:  # non-empty bytes
                return v
    raise ValueError(f"No non-empty data found in column '{image_col}' across {num_rgs} row-groups.")

def _decode_image_rgb(img_bytes: bytes) -> np.ndarray:
    """Decode compressed image bytes to RGB ndarray (H,W,3)"""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img_bgr is None:
        raise ValueError("cv2.imdecode() returned None; bytes may be corrupted or unsupported.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ---------- drop-in replacement inside your Dataset.__getitem__ ----------

def _load_rgb_from_camera_image_parquet(parquet_path: str) -> np.ndarray:
    """
    Robust loader: find the correct image column dynamically and decode first frame.
    If you want a specific frame index, extend this to also track row index.
    """
    pf = pq.ParquetFile(parquet_path)
    image_col = _guess_image_column_name(pf)
    img_bytes = _read_first_image_bytes(pf, image_col)
    img_rgb = _decode_image_rgb(img_bytes)
    return img_rgb

def coco_to_video(coco_json, image_dir, save_path="output.mp4", fps=5, max_frames=None):
    """
    Render COCO images with 2D boxes into a video.
    
    Args:
        coco_json (str): Path to annotations.json
        image_dir (str): Path to images/ folder
        save_path (str): Output video file (.mp4)
        fps (int): Frames per second
        max_frames (int or None): Limit frames for quick test
    """
    image_dir = Path(image_dir)
    
    # Load COCO annotations
    with open(coco_json, "r") as f:
        coco = json.load(f)

    images = sorted(coco["images"], key=lambda x: x["id"])
    anns = coco["annotations"]
    cats = {c["id"]: c["name"] for c in coco["categories"]}

    # Build mapping image_id -> annotations
    ann_map = {}
    for ann in anns:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    if max_frames:
        images = images[:max_frames]

    # Get video size from first image
    first_img_path = image_dir / Path(images[0]["file_name"]).name
    frame0 = cv2.imread(str(first_img_path))
    h, w = frame0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    for i, img_info in enumerate(images):
        img_path = image_dir / Path(img_info["file_name"]).name
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # Draw boxes
        for ann in ann_map.get(img_info["id"], []):
            x, y, bw, bh = ann["bbox"]
            cat_id = ann["category_id"]
            cat_name = cats.get(cat_id, str(cat_id))

            # Rectangle
            cv2.rectangle(frame, (int(x), int(y)), (int(x+bw), int(y+bh)), (0, 0, 255), 2)
            # Label
            cv2.putText(frame, cat_name, (int(x), max(0, int(y-5))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        video.write(frame)

        if (i+1) % 20 == 0:
            print(f"Processed {i+1}/{len(images)} frames")

    video.release()
    print(f"✅ Video saved: {save_path}")

def export_to_coco(dataset, output_dir, max_images=None, step=1):
    """
    Export Waymo2DDataset into COCO format with original IDs in filenames.

    Args:
        dataset: Waymo2DDataset or WaymoPartial2DDataset instance
                 (must return target dict with 'segment', 'timestamp', 'camera_id')
        output_dir (str or Path): Output directory
        max_images (int or None): Limit number of images to export
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {"description": "Waymo v2.1 subset (2D boxes, COCO format)"},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "Vehicle"},
            {"id": 2, "name": "Pedestrian"},
            {"id": 3, "name": "Cyclist"},
            {"id": 4, "name": "Sign"},
        ],
        "images": [],
        "annotations": []
    }

    ann_id = 1
    total_images = len(dataset)
    if max_images:
        total_images = min(total_images, max_images)
    #subsample based on step
    selected_indices = list(range(0, total_images, step))
    num_images = len(selected_indices)
    #num_images = len(dataset) if max_images is None else min(len(dataset), max_images)
    print(f"📦 Exporting {num_images} images (step={step}) to {output_dir} in COCO format...")

    #for new_id in range(num_images):
    for new_id, i in enumerate(tqdm(selected_indices, desc="Export to COCO",  unit="img")):
        #img_tensor, target = dataset[new_id]
        img_tensor, target = dataset[i]

        # Convert tensor back to numpy (CHW -> HWC)
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)

        # === Use original Waymo metadata for filename ===
        seg_name = str(target.get("segment", "seg"))
        timestamp = str(target.get("timestamp", "ts"))
        camera_id = str(target.get("camera_id", "cam"))

        # Construct filename: segment_camera_timestamp.jpg
        file_name = f"{seg_name}_{camera_id}_{timestamp}.jpg"
        img_path = images_dir / file_name
        Image.fromarray(img).save(img_path)

        h, w = img.shape[:2]

        # Add COCO image entry
        coco["images"].append({
            "id": new_id,  # keep COCO id sequential
            "file_name": f"images/{file_name}",
            "width": w,
            "height": h
        })

        # Add annotations
        boxes = target["boxes"].cpu().numpy()
        labels = target["labels"].cpu().numpy()

        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            w_box = xmax - xmin
            h_box = ymax - ymin
            area = w_box * h_box

            coco["annotations"].append({
                "id": ann_id,
                "image_id": new_id,  # reference COCO image id
                "category_id": int(label),
                "bbox": [float(xmin), float(ymin), float(w_box), float(h_box)],
                "area": float(area),
                "iscrowd": 0
            })
            ann_id += 1

    # Save COCO annotations JSON
    ann_path = output_dir / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f)

    print(f"✅ Export complete: {num_images} images, {len(coco['annotations'])} annotations")
    print(f"📂 Images saved in {images_dir}")
    print(f"📄 Annotations saved in {ann_path}")

WAYMO_CLASSES = {
    0: "Unknown",
    1: "Vehicle",
    2: "Pedestrian",
    3: "Cyclist",
    4: "Sign"
}

def visualize_sample(img_tensor, target, save_path=None):
    """
    Visualize a dataset sample with projected 2D bounding boxes + category names.
    
    Args:
        img_tensor (torch.Tensor): Image tensor [3,H,W]
        target (dict): Dictionary with "boxes" [N,4] and "labels"
        save_path (str or None): If set, save visualization to this path
    """
    # Convert tensor to numpy HWC
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    boxes = target["boxes"].cpu().numpy()
    labels = target["labels"].cpu().numpy()

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        # 显示类别名字，如果没有映射就显示 ID
        class_name = WAYMO_CLASSES.get(int(label), f"cls {label}")
        ax.text(
            xmin, max(0, ymin - 5), class_name,
            color='yellow', fontsize=10, weight='bold',
            bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none")
        )

    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"✅ Saved visualization: {save_path}")
    else:
        plt.show()

    plt.close(fig)

def find_box_columns(df):
    """
    Find column names for 3D box fields in a Waymo v2.1 camera_box parquet DataFrame.
    Returns dict with keys: center_x, center_y, center_z, length, width, height, heading, type
    """
    colmap = {}
    for c in df.columns:
        name = c.lower()
        if "center" in name and ("x" in name):
            colmap["center_x"] = c
        elif "center" in name and ("y" in name):
            colmap["center_y"] = c
        elif "center" in name and ("z" in name):
            colmap["center_z"] = c
        elif "length" in name:
            colmap["length"] = c
        elif "width" in name:
            colmap["width"] = c
        elif "height" in name:
            colmap["height"] = c
        elif "heading" in name or "yaw" in name:
            colmap["heading"] = c
        elif "type" in name and "name" not in name:
            colmap["type"] = c
    return colmap

def extract_2d_boxes(box_df):
    """
    Extract 2D boxes from Waymo v2.1 camera_box parquet DataFrame.
    Returns list of boxes [xmin, ymin, xmax, ymax] and labels.
    """
    boxes, labels = [], []

    # column names from schema
    cx_col = "[CameraBoxComponent].box.center.x"
    cy_col = "[CameraBoxComponent].box.center.y"
    w_col  = "[CameraBoxComponent].box.size.x"
    h_col  = "[CameraBoxComponent].box.size.y"
    type_col = "[CameraBoxComponent].type"

    for _, row in box_df.iterrows():
        cx, cy = row[cx_col], row[cy_col]
        bw, bh = row[w_col], row[h_col]

        xmin = cx - bw / 2
        ymin = cy - bh / 2
        xmax = cx + bw / 2
        ymax = cy + bh / 2

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(int(row[type_col]))

    return boxes, labels

def project_3d_box_to_2d(box_3d, intrinsic, extrinsic, img_w, img_h):
    """
    Project a 3D bounding box into the 2D image plane.
    Returns [xmin, ymin, xmax, ymax] or None if invalid.
    """
    cx, cy, cz = box_3d["center"]
    length, width, height = box_3d["size"]
    heading = box_3d["heading"]

    # Define 8 corners
    x_corners = [ length/2,  length/2, -length/2, -length/2,  length/2,  length/2, -length/2, -length/2]
    y_corners = [ width/2, -width/2, -width/2,  width/2,  width/2, -width/2, -width/2,  width/2]
    z_corners = [0, 0, 0, 0, -height, -height, -height, -height]

    corners = np.vstack([x_corners, y_corners, z_corners])  # (3,8)

    # Rotation around z-axis
    R = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading),  np.cos(heading), 0],
        [0, 0, 1]
    ])
    rotated = R @ corners

    # Translate to center
    translated = rotated + np.array([[cx], [cy], [cz]])

    # Homogeneous coords
    ones = np.ones((1, translated.shape[1]))
    pts_vehicle = np.vstack([translated, ones])

    # Vehicle → camera
    pts_camera = extrinsic @ pts_vehicle

    # Skip if all behind camera
    if np.all(pts_camera[2, :] <= 0):
        return None

    # Project into image
    pts_norm = pts_camera[:3, :] / pts_camera[2, :]
    pts_img = intrinsic @ pts_norm

    u = pts_img[0, :]
    v = pts_img[1, :]

    xmin, xmax = np.min(u), np.max(u)
    ymin, ymax = np.min(v), np.max(v)

    # Skip if completely outside
    if xmax < 0 or ymax < 0 or xmin > img_w or ymin > img_h:
        return None

    # Clip to image boundaries
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)

    if xmax <= xmin or ymax <= ymin:
        return None

    return [xmin, ymin, xmax, ymax]


class WaymoPartial2DDataset(Dataset):
    """
    Waymo v2.1 2D Detection Dataset (partial download supported).
    Only uses files that exist in all: camera_image / camera_box / camera_calibration
    """

    def __init__(self, root_dir, split="training", max_files=None, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.img_dir = os.path.join(root_dir, split, "camera_image")
        self.box_dir = os.path.join(root_dir, split, "camera_box")
        self.calib_dir = os.path.join(root_dir, split, "camera_calibration")

        # Use only image files that exist locally
        all_img_files = sorted(os.listdir(self.img_dir))
        if max_files:
            all_img_files = all_img_files[:max_files]

        # Keep only files that also exist in box + calibration
        self.valid_files = []
        for f in all_img_files:
            if os.path.exists(os.path.join(self.box_dir, f)) and \
               os.path.exists(os.path.join(self.calib_dir, f)):
                self.valid_files.append(f)

        print(f"✅ Found {len(self.valid_files)} usable segments with images+box+calibration")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        fname = self.valid_files[idx]

        # --- Load image (robust column guess) ---
        img_path = os.path.join(self.img_dir, fname)
        img_rgb = _load_rgb_from_camera_image_parquet(img_path)
        h, w, _ = img_rgb.shape

        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0

        # --- Load 2D boxes directly ---
        box_pf = pq.ParquetFile(os.path.join(self.box_dir, fname))
        box_df = box_pf.read_row_group(0).to_pandas()

        # Column names in Waymo v2.1 camera_box
        cx_col = "[CameraBoxComponent].box.center.x"
        cy_col = "[CameraBoxComponent].box.center.y"
        w_col  = "[CameraBoxComponent].box.size.x"
        h_col  = "[CameraBoxComponent].box.size.y"
        type_col = "[CameraBoxComponent].type"

        boxes, labels = [], []
        for _, row in box_df.iterrows():
            cx, cy = float(row[cx_col]), float(row[cy_col])
            bw, bh = float(row[w_col]), float(row[h_col])

            xmin = cx - bw / 2
            ymin = cy - bh / 2
            xmax = cx + bw / 2
            ymax = cy + bh / 2

            # clip to image boundaries
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(row[type_col]))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64)
        }

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, target

import os
import cv2
import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset

def _decode_image_rgb(img_bytes: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes into RGB image."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode failed")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

import os
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

# ============================
# Waymo v2.1 column name constants (keys + 2D boxes)
# ============================
# In v2.1, both camera_image and camera_box shards contain the following keys
KEY_SEG = "key.segment_context_name"      # segment identifier (string), the "sequence" name
KEY_TS  = "key.frame_timestamp_micros"    # frame timestamp (int64, microseconds)
KEY_CAM = "key.camera_name"               # camera id (int8), not a human-readable channel string

# In v2.1 camera_box shards, 2D bounding boxes are already provided in pixel space as:
#   - center.x / center.y   (box center in pixels)
#   - size.x / size.y       (box width/height in pixels)
# The column names are fully qualified with the component name.
BOX_CX   = "[CameraBoxComponent].box.center.x"
BOX_CY   = "[CameraBoxComponent].box.center.y"
BOX_W    = "[CameraBoxComponent].box.size.x"
BOX_H    = "[CameraBoxComponent].box.size.y"
BOX_TYPE = "[CameraBoxComponent].type"    # category id (int8)


# ============================
# Utility helpers
# ============================

def _decode_image_rgb(img_bytes: bytes) -> np.ndarray:
    """
    Decode compressed image bytes (JPEG/PNG) into an RGB numpy array (H, W, 3).

    OpenCV's imdecode returns BGR, so we convert to RGB for consistency with PyTorch/Matplotlib.
    """
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode failed: image bytes may be corrupted or unsupported.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _guess_image_column_name_from_schema(pf: pq.ParquetFile) -> str:
    """
    Heuristically find the image-bytes column in a Waymo v2.1 camera_image parquet shard.

    Why we need this:
    - Column name for encoded images is not guaranteed to be a fixed string like 'image'.
    - We therefore scan the Arrow schema and pick a binary/large_binary column
      whose name looks image-related. If none explicitly matches, fall back to the first binary column.
    """
    # NOTE: pf.schema_arrow gives Arrow Schema (SchemaField) with a `.type`,
    # while pf.schema gives Parquet ColumnSchema (no `.type` attribute).
    candidates = []
    for field in pf.schema_arrow:
        t = field.type
        is_bytes = (t.id in (pa.binary().id, pa.large_binary().id))
        if is_bytes:
            lname = field.name.lower()
            if any(k in lname for k in ["image", "jpeg", "jpg", "png", "encoded"]):
                candidates.append(field.name)

    if candidates:
        # Prefer the shortest/most "image-like" one deterministically
        candidates.sort(key=lambda n: (len(n), n))
        return candidates[0]

    # Fallback: any binary/large_binary column
    for field in pf.schema_arrow:
        t = field.type
        if t.id in (pa.binary().id, pa.large_binary().id):
            return field.name

    raise KeyError(
        f"Could not find an image-bytes column. Schema: {[f'{f.name}:{f.type}' for f in pf.schema_arrow]}"
    )


def _collect_time_order_index(pf: pq.ParquetFile) -> List[Tuple[int, int, int, str, int]]:
    """
    Build a time-ordered index for a camera_image parquet shard.

    Returns a list of tuples:
        (row_group_id, row_index_in_group, timestamp, segment_name, camera_id)

    Why we do this:
    - Each camera_image parquet shard contains MANY frames (rows).
    - We want to iterate frames in chronological order by timestamp, not by raw row order.
    - This helper reads ONLY the small key columns and constructs a per-row mapping
      to (row_group, in-group-row-index), then sorts by timestamp.

    Note:
    - Waymo v2.1 shards typically have a single row group, but we implement the general logic.
    """
    items: List[Tuple[int, int, int, str, int]] = []
    num_rgs = pf.num_row_groups

    required_cols = [KEY_SEG, KEY_TS, KEY_CAM]
    # Iterate all row groups and collect the tiny key DataFrames
    offset = 0
    for rg in range(num_rgs):
        tbl = pf.read_row_group(rg, columns=required_cols)
        df = tbl.to_pandas()

        # enumerate rows in this row group
        for i in range(len(df)):
            seg = df.iloc[i][KEY_SEG]
            ts  = int(df.iloc[i][KEY_TS])
            cam = int(df.iloc[i][KEY_CAM])
            items.append((rg, i, ts, seg, cam))

        offset += len(df)

    # sort by timestamp ascending (chronological order)
    items.sort(key=lambda t: t[2])
    return items


# ============================
# The Dataset (frame-level, time-ordered)
# ============================

class Waymo2DDataset(Dataset):
    """
    Waymo v2.1 → 2D Detection Dataset (frame-level, time-ordered).

    What "v2.1 shards" look like:
    - The dataset is sharded into many parquet files per split (training/validation/testing).
    - camera_image/*.parquet: contains MANY frames (rows). Each row = one frame for some camera at some timestamp.
      Important keys to align with labels:
        * key.segment_context_name (string)   : the sequence/segment id
        * key.frame_timestamp_micros (int64)  : frame time in microseconds (use to order frames)
        * key.camera_name (int8)              : camera id (not a channel string)
      It also contains ONE binary column with the compressed image (jpeg/png) bytes.

    - camera_box/*.parquet: contains 2D boxes (NOT 3D) for all frames in the same shard.
      Each row corresponds to ONE object in ONE frame and includes the same keys above,
      plus pixel-space box center/size:
        * [CameraBoxComponent].box.center.x : pixel center x
        * [CameraBoxComponent].box.center.y : pixel center y
        * [CameraBoxComponent].box.size.x   : pixel width
        * [CameraBoxComponent].box.size.y   : pixel height
        * [CameraBoxComponent].type         : category id

    This Dataset:
    - Expands a frame index at __init__ by:
        * listing camera_image shards on disk
        * discarding shards that don't have a matching camera_box shard (same filename)
        * scanning each image shard to build a time-ordered list of frames:
          (filename, row_group, in_group_row_index, timestamp, segment, cam_id, and image_column_name)
      Then __len__ is the number of frames, and __getitem__ returns a single (image, target) pair:
        image: torch.Tensor [3, H, W], float32 in [0,1]
        target: dict with
            "boxes": Tensor [N, 4]  in [xmin, ymin, xmax, ymax]
            "labels": Tensor [N]    category ids (int64)
            "image_id": Tensor [1]  the global index (int64)
    """

    def __init__(self, root_dir: str, split: str = "training",
                 max_frames: int | None = None, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Directories for the three important components
        # We only need images and boxes for 2D detection; calibration is NOT needed in v2.1 (boxes already 2D).
        self.img_dir = os.path.join(root_dir, split, "camera_image")
        self.box_dir = os.path.join(root_dir, split, "camera_box")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"camera_image folder not found: {self.img_dir}")
        if not os.path.isdir(self.box_dir):
            raise FileNotFoundError(f"camera_box folder not found: {self.box_dir}")

        # List all image shards; keep only those that also exist in camera_box (same shard filename).
        all_img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".parquet") or f.endswith(".parquet_.gstmp")])
        # Drop unfinished gstmp by default; if you want resume you can keep them.
        img_files = [f for f in all_img_files if f.endswith(".parquet")]
        valid_files = [f for f in img_files if os.path.exists(os.path.join(self.box_dir, f))]

        if not valid_files:
            raise RuntimeError("No matching shards found between camera_image/ and camera_box/.")

        # Build a frame-level index across shards, ordered by timestamp within each shard;
        # also store per-file image column name for faster access in __getitem__.
        self.frame_index: List[Tuple[str, int, int, int, str, int]] = []  # (fname, rg, rg_row, ts, seg, cam)
        self.image_col_by_file: Dict[str, str] = {}

        total = 0
        for fname in valid_files:
            img_path = os.path.join(self.img_dir, fname)
            pf = pq.ParquetFile(img_path)

            # Find the encoded image column name once per file.
            img_col = _guess_image_column_name_from_schema(pf)
            self.image_col_by_file[fname] = img_col

            # Build time-ordered (row_group, in_group_row, ts, seg, cam) for this shard.
            time_index = _collect_time_order_index(pf)
            for (rg, rg_row, ts, seg, cam) in time_index:
                self.frame_index.append((fname, rg, rg_row, ts, seg, cam))
                total += 1
                if max_frames is not None and total >= max_frames:
                    break
            if max_frames is not None and total >= max_frames:
                break

        print(f"✅ Waymo2DDataset initialized with {len(self.frame_index)} frames (time-ordered)")

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Load ONE frame (image + 2D boxes) by global frame index.

        Steps:
        1) Read the image row from camera_image shard using (row_group, in_group_row).
        2) Get the alignment keys (segment/timestamp/camera) from the same row.
        3) Read the matching camera_box shard and filter rows where keys match this frame.
        4) Convert Waymo 2D center/size → [xmin, ymin, xmax, ymax], clip to image bounds.
        """
        fname, rg, rg_row, ts, seg, cam = self.frame_index[idx]

        # ---------- 1) Load the image row ----------
        img_path = os.path.join(self.img_dir, fname)
        img_pf = pq.ParquetFile(img_path)
        img_col = self.image_col_by_file[fname]

        # Read ONLY the required columns for this row-group to keep memory lower.
        # We also re-read the keys here to be robust against any discrepancy.
        tbl = img_pf.read_row_group(rg, columns=[img_col, KEY_SEG, KEY_TS, KEY_CAM])
        df = tbl.to_pandas()

        # Fetch the encoded image bytes and keys for the *specific* row within the row-group
        row = df.iloc[rg_row]
        img_bytes = row[img_col]
        img_rgb = _decode_image_rgb(img_bytes)
        h, w, _ = img_rgb.shape

        # (Optional) trust the keys we stored at indexing time; or use from row for safety
        seg_name  = row[KEY_SEG]
        timestamp = int(row[KEY_TS])
        cam_id    = int(row[KEY_CAM])

        # Convert to CHW float tensor in [0,1]
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0

        # ---------- 2) Load boxes for THIS frame ----------
        box_path = os.path.join(self.box_dir, fname)
        box_pf = pq.ParquetFile(box_path)

        # Read only the needed columns; the shard can be large, but we filter by keys immediately.
        # NOTE: Waymo v2.1 shards typically have 1 row-group; if not, you may iterate all row-groups here.
        box_tbl = box_pf.read_row_group(0, columns=[KEY_SEG, KEY_TS, KEY_CAM, BOX_CX, BOX_CY, BOX_W, BOX_H, BOX_TYPE])
        box_df = box_tbl.to_pandas()

        # Filter annotations that belong to THIS frame (same segment, same timestamp, same camera)
        # This reduces thousands of rows to the N objects present in the current image.
        frame_boxes = box_df[
            (box_df[KEY_SEG] == seg_name) &
            (box_df[KEY_TS] == timestamp) &
            (box_df[KEY_CAM] == cam_id)
        ]

        boxes: List[List[float]] = []
        labels: List[int] = []

        # ---------- 3) Convert center/size → [xmin, ymin, xmax, ymax] ----------
        # All values are already in pixel space (no projection needed).
        for _, b in frame_boxes.iterrows():
            cx = float(b[BOX_CX])
            cy = float(b[BOX_CY])
            bw = float(b[BOX_W])
            bh = float(b[BOX_H])

            xmin = cx - bw / 2.0
            ymin = cy - bh / 2.0
            xmax = cx + bw / 2.0
            ymax = cy + bh / 2.0

            # Clip to image bounds to avoid negative or > width/height coordinates
            xmin = max(0.0, xmin)
            ymin = max(0.0, ymin)
            xmax = min(float(w), xmax)
            ymax = min(float(h), ymax)

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(b[BOX_TYPE]))

        target: Dict[str, Any] = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),  dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "segment": seg_name,
            "timestamp": timestamp,
            "camera_id": cam_id,
            # (Optional) you can add extra metadata for debugging/exports:
            # "segment": seg_name, "timestamp": timestamp, "camera_id": cam_id
        }

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, target
    
def main():
    basefolder = r"E:\Shared\Dataset\waymodata"
    SPLIT = "training"
    img = load_image_from_parquet(
        os.path.join(basefolder, SPLIT, "camera_image/1005081002024129653_5313_150_5333_150.parquet"),
        row_idx=0
    )
    import pyarrow.parquet as pq
    box_pf = pq.ParquetFile(os.path.join(basefolder, SPLIT, "camera_box/15832924468527961_1564_160_1584_160.parquet"))
    print(box_pf.schema_arrow)


    print("Image shape:", img.shape)  # e.g., (1280, 1920, 3)

    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.show()

    # dataset = WaymoPartial2DDataset("/data/Datasets/WaymoV2_1_2D", split="training", max_files=5)

    # img, target = dataset[0]
    # print("Image:", img.shape)   # [3,H,W]
    # print("Boxes:", target["boxes"])
    # print("Labels:", target["labels"])
    max_frames = None #100000
    dataset = Waymo2DDataset(basefolder, split=SPLIT, max_frames=max_frames)

    img, target = dataset[0]
    print("Image:", img.shape)        # [3,H,W] [3, 1280, 1920]
    print("Boxes:", target["boxes"])  # [[xmin,ymin,xmax,ymax], ...] 5,4
    print("Labels:", target["labels"]) #[1, 1, 1, 1, 1]

    # Visualize and save to file
    visualize_sample(img, target, save_path="sample_0_vis.png")

    output_foldername="waymo_subset_coco_step10"
    export_to_coco(dataset, output_dir=os.path.join(basefolder, output_foldername), max_images=max_frames, step=10)

    coco_to_video(
        coco_json=os.path.join(basefolder, output_foldername, "annotations.json"), #"/data/Datasets/waymo_subset_coco/annotations.json",
        image_dir=os.path.join(basefolder, output_foldername, "images"), #"/data/Datasets/waymo_subset_coco/images",
        save_path=os.path.join(basefolder, output_foldername, "waymo_subset.mp4"),
        fps=5,
        max_frames=max_frames   # 可选，只导出前100帧
    )

if __name__ == "__main__":
    main()