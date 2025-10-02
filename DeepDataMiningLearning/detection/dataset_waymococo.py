#ref: https://github.com/lkk688/WaymoObjectDetection/blob/master/MyDetector/torchvision_waymococo_train.py
import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
from glob import glob
import os
import math
import itertools
import torch.utils.data as data
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import cv2
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import shutil

class WaymoCOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.coco = COCO(annotation)
        self.is_train = train
        self.ids = list(sorted(self.coco.imgs.keys()))#id string list

        #
        dataset=self.coco.dataset #'images': image filename (images/xxx.jpg) with image_id (0000001)
        imgToAnns=self.coco.imgToAnns #image_id to list of annotations
        catToImgs =self.coco.catToImgs #three classes, 1,2,4
        cats=self.coco.cats
        self.numclass = 5 #len(catToImgs) + 1 #three classes + background
        #num_classes=5 # ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
        #previous_num_classes = 4 #Unknown:0, Vehicles: 1, Pedestrians: 2, Cyclists: 3, Signs (removed)
        #Real data only has 
        self.INSTANCE_CATEGORY_NAMES = ['__background__','Vehicles', 'Pedestrians', 'Cyclists', 'Signs']
        #self.INSTANCE2id = {'__background__':0, 'Vehicles': 1, 'Pedestrians': 2, 'Cyclists': 4} #background is 0
        #self.id2INSTANCE = {v: k for k, v in self.INSTANCE2id.items()}
        #In annotation, class is 1,2,4

    
    def _get_target(self, id):
        'Get annotations for sample'

        # List: get annotation id from coco
        ann_ids = self.coco.getAnnIds(imgIds=id)
        # Dictionary: target coco_annotation file for an image
        #ref: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
        annotations = self.coco.loadAnns(ann_ids)

        boxes, categories = [], []
        for ann in annotations:
            if ann['bbox'][2] < 1 and ann['bbox'][3] < 1:
                continue
            boxes.append(ann['bbox'])
            cat = ann['category_id']
            if 'categories' in self.coco.dataset:
                cat = self.categories_inv[cat]
            categories.append(cat)

        if boxes:
            target = (torch.FloatTensor(boxes),
                      torch.FloatTensor(categories).unsqueeze(1))
        else:
            target = (torch.ones([1, 4]), torch.ones([1, 1]) * -1)

        return target


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        imginfo=self.coco.imgs[img_id]
        path = imginfo['file_name']
        #print(f'index: {index}, img_id:{img_id}, info: {imginfo}')

        # path for input image
        #loadedimglist=coco.loadImgs(img_id)
        # print(loadedimglist)
        #path = coco.loadImgs(img_id)[0]['file_name']
        #print("image path:", path)
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        #img = Image.open(os.path.join(self.root, path)).convert('RGB')


        # List: get annotation id from coco
        #ann_ids = coco.getAnnIds(imgIds=img_id)
        annolist=[self.coco.imgToAnns[img_id]]
        anns = list(itertools.chain.from_iterable(annolist))
        ann_ids = [ann['id'] for ann in anns]
        # Dictionary: target coco_annotation file for an image
        #ref: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
        targets  = coco.loadAnns(ann_ids)
        #targets=self.anns[ann_ids]
        #print("targets:", targets)
        
        #image_id = targets["image_id"].item()

        # number of objects in the image
        num_objs = len(targets)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        target = {}
        target_bbox = []
        target_labels = []
        target_areas = []
        target_crowds = []
        for i in range(num_objs):
            xmin = targets[i]['bbox'][0]
            ymin = targets[i]['bbox'][1]
            width=targets[i]['bbox'][2]
            xmax = xmin + width
            height = targets[i]['bbox'][3]
            ymax = ymin + height
            if xmin<=xmax and ymin<=ymax and xmin>=0 and ymin>=0 and width>1 and height>1:
                target_bbox.append([xmin, ymin, xmax, ymax])
                target_labels.append(targets[i]['category_id'])
                #target_crowds.append(targets[i]['iscrowd'])
                target_areas.append(targets[i]['area'])
        num_objs=len(target_bbox)
        #print("target_bbox len:", num_objs)
        if num_objs>0:
            #print("target_labels:", target_labels)
            target['boxes'] = torch.as_tensor(target_bbox, dtype=torch.float32)
            # Labels int value for class
            target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)
            #target['image_id'] = torch.tensor([int(img_id)])
            #target['image_id'] = torch.tensor(int(img_id))
            target['image_id'] = int(img_id)
            #torch.tensor([int(frameitem.context.name.split("_")[-2] + str(index))])
            target["area"] = torch.as_tensor(np.array(target_areas), dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#torch.zeros((len(target['boxes'])), dtype=torch.int64)
        else:
            #negative example, ref: https://github.com/pytorch/vision/issues/2144
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)#not empty
            target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)#empty
            #target['image_id'] = torch.tensor([int(img_id)])
            #target['image_id'] = torch.tensor(int(img_id))
            target['image_id'] = int(img_id)
            target["area"] = torch.as_tensor(np.array(target_areas), dtype=torch.float32)#empty
            target["iscrowd"] = torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#empty

        # if self.transforms is not None:
        #     img = self.transforms(img)
        if self.transform:
            img, target = self.transform(img, target)
        #print("target:", target)
        return img, target

    def __len__(self):
        return len(self.ids)

import DeepDataMiningLearning.detection.transforms as T
def get_transformsimple(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ToDtype(torch.float, scale=True))
    # if train:
    #     transforms.append(RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def validate_annotations(coco, root):
    """Validate bounding boxes and print annotation issues."""
    print("=== Validating annotations ===")
    invalid_bboxes = 0
    for ann in coco.dataset['annotations']:
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0:
            invalid_bboxes += 1
    print(f"Total annotations: {len(coco.dataset['annotations'])}")
    print(f"Invalid bounding boxes (w<=0 or h<=0): {invalid_bboxes}")


def dataset_summary(coco):
    """Summarize dataset by folder, categories, and image counts."""
    print("\n=== Dataset Summary ===")
    folder_counts = defaultdict(int)
    for img in coco.dataset['images']:
        folder = Path(img['file_name']).parent.name
        folder_counts[folder] += 1

    print(f"Total Folders: {len(folder_counts)}")
    for folder, count in folder_counts.items():
        print(f"  {folder}: {count} images")

    print(f"\nCategories ({len(coco.cats)}):")
    for cid, cat in coco.cats.items():
        print(f"  {cid}: {cat['name']}")

    print(f"\nTotal Images: {len(coco.dataset['images'])}")
    print(f"Total Annotations: {len(coco.dataset['annotations'])}")

def visualize_sample(dataset, idx=0, save_dir="vis_output"):
    """Visualize a dataset sample with bounding boxes and save to file."""
    img, target = dataset[idx]

    # Convert to numpy RGB for matplotlib
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    elif isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))  # Force RGB
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)  # RGB is correct for matplotlib

    boxes = target["boxes"].cpu().numpy()
    labels = target["labels"].cpu().numpy()
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        cat_name = dataset.INSTANCE_CATEGORY_NAMES[label] \
            if label < len(dataset.INSTANCE_CATEGORY_NAMES) else str(label)
        ax.text(xmin, ymin-5, cat_name, color='yellow', fontsize=10, weight='bold')

    plt.axis("off")

    # Save visualization
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"sample_{idx}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"✅ Saved visualization to {save_path}")



def export_folder_to_video(dataset, folder_name, save_path="output_video.mp4", fps=5):
    """
    Export all images in a given folder (with bounding boxes) into a video.
    
    Args:
        dataset: WaymoCOCODataset
        folder_name (str): folder to filter (e.g., "segment-1234")
        save_path (str): output video file path
        fps (int): frames per second for video
    """
    # Find all image IDs belonging to this folder
    img_ids = []
    for img_id, imginfo in dataset.coco.imgs.items():
        if Path(imginfo['file_name']).parent.name == folder_name:
            img_ids.append(img_id)

    if len(img_ids) == 0:
        print(f"No images found in folder: {folder_name}")
        return

    print(f"Found {len(img_ids)} images in folder {folder_name}. Generating video...")

    # Collect frames
    frames = []
    for img_id in sorted(img_ids):
        imginfo = dataset.coco.imgs[img_id]
        img_path = os.path.join(dataset.root, imginfo['file_name'])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        # Get annotations
        anns = dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=img_id))
        boxes = [ann["bbox"] for ann in anns]
        labels = [ann["category_id"] for ann in anns]

        # Draw annotations
        for box, label in zip(boxes, labels):
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cat_name = dataset.INSTANCE_CATEGORY_NAMES[label] \
                if label < len(dataset.INSTANCE_CATEGORY_NAMES) else str(label)
            cv2.putText(img, cat_name, (x, max(0, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        frames.append(img)

    # Set video size (from first frame)
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    # for frame in frames:
    #     #video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #     video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    for frame in frames:
        # Convert from RGB (PIL) → BGR (OpenCV)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)

    video.release()
    print(f"✅ Video saved to {save_path}")

from pathlib import Path
def check_image_color(img_path):
    # Load with PIL (always RGB)
    pil_img = Image.open(img_path).convert("RGB")
    pil_arr = np.array(pil_img)

    # Load with OpenCV (always BGR)
    cv_img = cv2.imread(img_path)  # BGR
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # Show side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(pil_arr)
    axs[0].set_title("PIL (RGB)")
    axs[1].imshow(cv_img_rgb)
    axs[1].set_title("OpenCV (BGR→RGB)")
    plt.show()

def subsample_dataset(dataset_root, ann_file, output_root, step=8, resize_scale=None):
    """
    Subsample images in each folder (keep 1 out of N), with optional downsampling.
    Copies images and updates COCO annotations accordingly.

    Args:
        dataset_root (str): Root of original dataset (with images + annotations).
        ann_file (str): Path to original COCO annotation json.
        output_root (str): New dataset root to save subsampled dataset.
        step (int): Keep 1 image every 'step' (e.g., 8 means keep every 8th image).
        resize_scale (float or None): If set, resize images by this factor (e.g., 0.5 = half size).
    """
    os.makedirs(output_root, exist_ok=True)

    # Load original COCO annotations
    with open(ann_file, "r") as f:
        coco = json.load(f)

    # Build folder → images mapping
    folder_to_imgs = {}
    for img in coco["images"]:
        folder = Path(img["file_name"]).parent.as_posix()
        folder_to_imgs.setdefault(folder, []).append(img)

    # New dataset structure
    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": [],
        "annotations": []
    }

    kept_img_ids = set()

    # Pick every Nth image in each folder
    for folder, imgs in folder_to_imgs.items():
        imgs_sorted = sorted(imgs, key=lambda x: x["file_name"])
        selected = imgs_sorted[::step]  # keep every Nth

        for img in tqdm(selected, desc=f"Processing {folder}"):
            kept_img_ids.add(img["id"])

            src_path = Path(dataset_root) / img["file_name"]
            dst_path = Path(output_root) / img["file_name"]
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if resize_scale is None:
                # Just copy
                shutil.copy2(src_path, dst_path)
                new_img = img.copy()
            else:
                # Load → resize → save
                im = cv2.imread(str(src_path))  # BGR
                if im is None:
                    print(f"Warning: cannot read {src_path}")
                    continue
                new_w = int(im.shape[1] * resize_scale)
                new_h = int(im.shape[0] * resize_scale)
                im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst_path), im_resized)

                # Update image metadata
                new_img = img.copy()
                new_img["width"] = new_w
                new_img["height"] = new_h

            new_coco["images"].append(new_img)

    # Keep only annotations belonging to selected images
    for ann in coco["annotations"]:
        if ann["image_id"] in kept_img_ids:
            new_ann = ann.copy()
            if resize_scale is not None:
                # Scale bbox and area
                x, y, w, h = new_ann["bbox"]
                new_ann["bbox"] = [x * resize_scale, y * resize_scale,
                                   w * resize_scale, h * resize_scale]
                new_ann["area"] = new_ann["area"] * (resize_scale ** 2)
            new_coco["annotations"].append(new_ann)

    # Save new annotation file
    suffix = "_subsampled"
    if resize_scale is not None:
        suffix += f"_resized{resize_scale}"
    out_json = Path(output_root) / Path(ann_file).name.replace(".json", f"{suffix}.json")

    with open(out_json, "w") as f:
        json.dump(new_coco, f)

    print(f"✅ Subsampled dataset saved to: {output_root}")
    print(f"✅ New annotation file: {out_json}")
    print(f"✅ Kept {len(new_coco['images'])} images (out of {len(coco['images'])})")


if __name__ == "__main__":
    #check_image_color("/data/Datasets/WaymoCOCO/Training/training_0031/9529958888589376527_640_000_660_000_1557956305498634_FRONT.jpg")
    subsample_dataset(
        dataset_root="/data/Datasets/WaymoCOCO/Training",
        ann_file="/data/Datasets/WaymoCOCO/Training/annotations.json",
        output_root="/data/Datasets/WaymoCOCO/Training_subsampled",
        step=8
    )

    # path to your own data and coco file
    data_root = '/data/Datasets/WaymoCOCO/Training'#'/data/cmpe249-fa23/WaymoCOCO/'

    #jsonfile = 'annotations_train684step8allobject.json' #4 classes, 0,1,2,4 85008
    jsonfile = 'annotations.json' #4 classes, 0,1,2,4 199935
    #jsonfile = 'annotations_trainallnew.json' #4 classes, 0,1,2,4 677530
    #jsonfile = '3classsub_annotations_trainall.json' #4 classes, 0,1,2,4 20820
    ann_file = os.path.join(data_root, jsonfile)#'annotations_train20new.json'
    # create own Dataset
    mywaymodataset = WaymoCOCODataset(root=data_root,  
                          annotation=ann_file
                          )
    length = len(mywaymodataset)
    print("Dataset",len(mywaymodataset))#85008
    img, target = mywaymodataset[0]
    print(target.keys())

    print("Dataset length:", len(mywaymodataset))
    validate_annotations(mywaymodataset.coco, data_root)
    dataset_summary(mywaymodataset.coco)

    # Visualize a few samples
    for i in [0, 100, 150, 155, 160, 170, 175]:  # pick some indices
        visualize_sample(mywaymodataset, idx=i, save_dir='output')
    
    # Export all annotated images in one folder into a video
    export_folder_to_video(mywaymodataset, folder_name="training_0031", save_path="output/training_0031.mp4", fps=5)