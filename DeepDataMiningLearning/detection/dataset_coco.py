import cv2
import os
from pathlib import Path
from pycocotools.coco import COCO

def coco_to_video(image_root, ann_file, save_path="coco_output.mp4", fps=5, max_frames=None):
    """
    Load a COCO dataset and export annotated images to a video.

    Args:
        image_root (str): Root folder containing images (relative paths from JSON).
        ann_file (str): Path to COCO annotations.json.
        save_path (str): Path to save the video file (.mp4).
        fps (int): Frames per second of output video.
        max_frames (int or None): Limit number of frames (for testing).
    """
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    if max_frames:
        img_ids = img_ids[:max_frames]

    print(f"📦 Found {len(img_ids)} images in {ann_file}")

    # Pre-fetch categories
    cats = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

    # Get size from first frame
    first_img = coco.loadImgs(img_ids[0])[0]
    first_path = os.path.join(image_root, first_img['file_name'])
    first_frame = cv2.imread(first_path)
    if first_frame is None:
        raise FileNotFoundError(f"Could not read {first_path}")
    h, w = first_frame.shape[:2]

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    for i, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_root, img_info['file_name'])
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"⚠️ Skipping missing image: {img_path}")
            continue

        # Get annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            x, y, w, h = [int(v) for v in ann['bbox']]
            cat_name = cats.get(ann['category_id'], str(ann['category_id']))

            # Draw bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, cat_name, (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        video.write(frame)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(img_ids)} frames")

    video.release()
    print(f"✅ Video saved to {save_path}")

def main():

    coco_to_video(
            image_root="/data/Datasets/nuscenes_subset_coco_step10",
            ann_file="/data/Datasets/nuscenes_subset_coco_step10/annotations.json",
            save_path="/data/Datasets/nuscenes_subset_coco_step10/nuscenes_subset.mp4",
            fps=5,
            max_frames=1000
        )

if __name__ == "__main__":
    main()