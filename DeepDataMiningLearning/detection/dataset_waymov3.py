import os
import cv2
import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def export_to_coco(dataset, output_dir, max_images=None):
    """
    Export WaymoPartial2DDataset into COCO format.
    
    Args:
        dataset: WaymoPartial2DDataset instance
        output_dir (str): Directory to save images + annotations.json
        max_images (int or None): Limit number of images to export
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {"description": "Waymo v2.1 subset (projected 2D boxes)"},
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
    num_images = len(dataset) if max_images is None else min(len(dataset), max_images)

    print(f"📦 Exporting {num_images} images to {output_dir} in COCO format...")

    for i in range(num_images):
        img_tensor, target = dataset[i]

        # Convert tensor back to numpy
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)

        # Save image
        file_name = f"{i:06d}.jpg"
        img_path = images_dir / file_name
        Image.fromarray(img).save(img_path)

        h, w = img.shape[:2]

        # Add COCO image entry
        coco["images"].append({
            "id": i,
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
                "image_id": i,
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

def main():
    img = load_image_from_parquet(
        "/data/Datasets/WaymoV2_1_2D/training/camera_image/1005081002024129653_5313_150_5333_150.parquet",
        row_idx=0
    )
    import pyarrow.parquet as pq
    box_pf = pq.ParquetFile("/data/Datasets/WaymoV2_1_2D/training/camera_box/15832924468527961_1564_160_1584_160.parquet")
    print(box_pf.schema_arrow)


    print("Image shape:", img.shape)  # e.g., (1280, 1920, 3)

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

    dataset = WaymoPartial2DDataset("/data/Datasets/WaymoV2_1_2D", split="training", max_files=5)

    img, target = dataset[0]
    print("Image:", img.shape)   # [3,H,W]
    print("Boxes:", target["boxes"])
    print("Labels:", target["labels"])

    # Visualize and save to file
    visualize_sample(img, target, save_path="sample_0_vis.png")

    export_to_coco(dataset, output_dir="/data/Datasets/waymo_subset_coco", max_images=10)

    coco_to_video(
        coco_json="/data/Datasets/waymo_subset_coco/annotations.json",
        image_dir="/data/Datasets/waymo_subset_coco/images",
        save_path="/data/Datasets/waymo_subset.mp4",
        fps=5,
        max_frames=100   # 可选，只导出前100帧
    )

if __name__ == "__main__":
    main()