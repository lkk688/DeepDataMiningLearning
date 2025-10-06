import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from DeepDataMiningLearning.detection.coco_utils import get_coco
import os
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
import csv
from DeepDataMiningLearning.detection.dataset_kitti import KittiDataset
from DeepDataMiningLearning.detection.dataset_waymococo import WaymoCOCODataset
from collections import defaultdict
#pip install pycocotools
import torch
import DeepDataMiningLearning.detection.transforms as reference_transforms
from torchvision.transforms import functional as F
import random

WrapNewDict = False

# Custom Resize transform for detection
class Resize:
    def __init__(self, size, antialias=True):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.antialias = antialias
    
    def __call__(self, image, target):
        # Resize image
        image = F.resize(image, self.size, antialias=self.antialias)
        
        # If target contains boxes, we need to resize them too
        if target is not None and "boxes" in target:
            # Get original and new dimensions
            _, orig_height, orig_width = F.get_dimensions(image)
            new_height, new_width = self.size
            
            # Calculate scale factors
            scale_x = new_width / orig_width
            scale_y = new_height / orig_height
            
            # Scale boxes
            boxes = target["boxes"]
            boxes[:, 0] *= scale_x  # x1
            boxes[:, 1] *= scale_y  # y1
            boxes[:, 2] *= scale_x  # x2
            boxes[:, 3] *= scale_y  # y2
            target["boxes"] = boxes
        
        return image, target

#download coco
#wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip
#wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Create different colors for each class.
COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))

def get_modules(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2
        import torchvision.tv_tensors

        return torchvision.transforms.v2, torchvision.tv_tensors
    else:
        return reference_transforms, None


class DetectionPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter.
    def __init__(
        self,
        *,
        data_augmentation,
        hflip_prob=0.5,
        mean=(123.0, 117.0, 104.0),
        backend="pil",
        use_v2=False,
        img_size=640,
    ):

        T, tv_tensors = get_modules(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tv_tensor":
            transforms.append(T.ToImage())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        # Add resize transform at the beginning to ensure consistent image size
        transforms.append(Resize(size=(img_size, img_size), antialias=True))

        if data_augmentation == "hflip":
            transforms += [T.RandomHorizontalFlip(p=hflip_prob)]
        elif data_augmentation == "lsj":
            transforms += [
                T.ScaleJitter(target_size=(1024, 1024), antialias=True),
                # TODO: FixedSizeCrop below doesn't work on tensors!
                reference_transforms.FixedSizeCrop(size=(1024, 1024), fill=mean),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "multiscale":
            transforms += [
                T.RandomShortestSize(min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssd":
            fill = defaultdict(lambda: mean, {tv_tensors.Mask: 0}) if use_v2 else list(mean)
            transforms += [
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=fill),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssdlite":
            transforms += [
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2.
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]

        transforms += [T.ToDtype(torch.float, scale=True)]

        if use_v2:
            transforms += [
                T.ConvertBoundingBoxFormat(tv_tensors.BoundingBoxFormat.XYXY),
                T.SanitizeBoundingBoxes(),
                T.ToPureTensor(),
            ]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self, backend="pil", use_v2=False, img_size=640):
        T, _ = get_modules(use_v2)
        transforms = []
        backend = backend.lower()
        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2?
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]
        elif backend == "tensor":
            transforms += [T.PILToTensor()]
        elif backend == "tv_tensor":
            transforms += [T.ToImage()]
        else:
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        # Add resize transform to ensure consistent image size
        transforms.append(Resize(size=(img_size, img_size), antialias=True))

        transforms += [T.ToDtype(torch.float, scale=True)]

        if use_v2:
            transforms += [T.ToPureTensor()]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_dataset(datasetname, is_train, is_val, args, output_format="torch", img_size=None):
    # Use img_size from args if not explicitly provided
    # if img_size is None:
    #     img_size = getattr(args, 'img_size', 640)  # Default to 640 if not found
    
    
    if datasetname.lower() == 'coco':
        ds, num_classes = get_cocodataset(is_train, is_val, args, output_format, img_size)
    elif datasetname.lower() == 'kitti':
        ds, num_classes = get_kittidataset(is_train, is_val, args, output_format, img_size)
    elif datasetname.lower() == 'kitti_yolo':
        ds, num_classes = get_kittiyolodataset(is_train, is_val, args, output_format, img_size)
    elif datasetname.lower() == 'waymococo':
        ds, num_classes = get_waymococodataset(is_train, is_val, args.data_path, args.annotationfile, output_format, img_size)
    elif datasetname.lower() == 'yolo':
        ds, num_classes = get_yolodataset(is_train, is_val, args, output_format, img_size)
    return ds, num_classes

def get_transform(is_train, args, img_size=640):
    if is_train:
        return DetectionPresetTrain(
            data_augmentation=args.data_augmentation, backend=args.backend, use_v2=args.use_v2, img_size=img_size
        )
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return DetectionPresetEval(backend=args.backend, use_v2=args.use_v2, img_size=img_size)  

def get_transform_new(is_train, args, img_size=640):
    """
    Unified transform function that keeps image and annotations synchronized.
    Works for both training and evaluation, even when pretrained weights are used.
    """

    # 1️⃣ Training mode → DetectionPresetTrain (resizes & augments image + boxes)
    if is_train:
        return DetectionPresetTrain(
            data_augmentation=args.data_augmentation,
            backend=args.backend,
            use_v2=args.use_v2,
            img_size=img_size
        )

    # 2️⃣ Evaluation mode (test_only=True with weights)
    elif args.weights and args.test_only:
        # Load model-defined preprocessing (resize + normalization)
        weights = torchvision.models.get_weight(args.weights)
        weight_trans = weights.transforms()  # image-only transform

        # Wrap into a transform that also rescales boxes/masks
        def transform_with_boxes(img, target):
            orig_w, orig_h = img.size
            img = weight_trans(img)
            new_w, new_h = img.shape[-1], img.shape[-2]

            # If boxes exist → resize them to match new image size
            if "boxes" in target:
                boxes = target["boxes"]
                boxes = resize_boxes(boxes, (orig_h, orig_w), (new_h, new_w))
                target["boxes"] = boxes

            # Resize masks if present
            if "masks" in target:
                target["masks"] = torchvision.transforms.functional.resize(
                    target["masks"], (new_h, new_w)
                )

            return img, target

        return transform_with_boxes

    # 3️⃣ Default evaluation (no pretrained weights)
    else:
        return DetectionPresetEval(
            backend=args.backend,
            use_v2=args.use_v2,
            img_size=img_size
        )

def resize_boxes(boxes, original_size, new_size):
    ratios = [float(new_size[0]) / original_size[0],
              float(new_size[1]) / original_size[1]]
    ratio_height, ratio_width = ratios
    boxes = boxes * torch.tensor([ratio_width, ratio_height, ratio_width, ratio_height])
    return boxes

def make_train_transform_with_resize(
    img_size=640,
    resize_prob=1.0,
    hflip_prob=0.5,
    color_jitter=True,
    normalize=True
):
    """
    Returns a training transform that resizes image + annotations together,
    applies random horizontal flip and optional color jitter.

    Args:
        img_size (int): target shorter side for resizing (aspect ratio preserved)
        resize_prob (float): probability of resizing (1.0 = always resize)
        hflip_prob (float): probability of horizontal flip
        color_jitter (bool): apply random brightness/contrast/saturation
        normalize (bool): apply ImageNet mean/std normalization
    Returns:
        transform: callable (img, target) -> (img, target)
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def transform(img, target):
        orig_w, orig_h = img.size if not torch.is_tensor(img) else (img.shape[-1], img.shape[-2])

        # ------------------------
        # 1. Resize (aspect-ratio preserving)
        # ------------------------
        if img_size is not None and random.random() < resize_prob:
            img = F.resize(img, [img_size])
            new_w, new_h = img.size if not torch.is_tensor(img) else (img.shape[-1], img.shape[-2])

            if "boxes" in target:
                boxes = target["boxes"]
                if not torch.is_tensor(boxes):
                    boxes = torch.as_tensor(boxes, dtype=torch.float32)
                boxes = resize_boxes(boxes, (orig_h, orig_w), (new_h, new_w))
                target["boxes"] = boxes

            #if "masks" in target:
                #target["masks"] = F.resize(target["masks"], [new_h, new_w])

            if "masks" in target and target["masks"].numel() > 0:
                target["masks"] = F.resize(target["masks"], [new_h, new_w])
        # ------------------------
        # 2. Random horizontal flip
        # ------------------------
        if random.random() < hflip_prob:
            img = F.hflip(img)
            w, h = img.size
            if "boxes" in target:
                boxes = target["boxes"]
                if not torch.is_tensor(boxes):
                    boxes = torch.as_tensor(boxes, dtype=torch.float32)

                # Skip if no boxes
                if boxes.numel() == 0: #empty box
                    target["boxes"] = boxes.reshape(0, 4)
                else:
                    # Ensure 2-D shape [N,4]
                    if boxes.ndim == 1:
                        boxes = boxes.unsqueeze(0)

                    # Horizontal flip
                    boxes = boxes.clone()
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                    target["boxes"] = boxes
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)

        # ------------------------
        # 3. Color jitter (brightness/contrast/saturation)
        # ------------------------
        if color_jitter:
            img = F.adjust_brightness(img, 1.0 + (random.random() - 0.5) * 0.2)
            img = F.adjust_contrast(img, 1.0 + (random.random() - 0.5) * 0.2)
            img = F.adjust_saturation(img, 1.0 + (random.random() - 0.5) * 0.2)

        # ------------------------
        # 4. Convert to tensor
        # ------------------------
        img = F.to_tensor(img)

        # ------------------------
        # 5. Normalize (optional)
        # ------------------------
        if normalize:
            img = F.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        return img, target

    return transform

def make_eval_transform_with_resize(img_size=None):
    """
    Returns a transform that keeps image and annotations aligned.
    - If img_size is given (e.g. 640): resize image + boxes/masks with aspect ratio preserved.
    - If img_size is None: no resizing is performed.
    
    Args:
        img_size (int or None): target shorter side length. If None, skip resizing.
    Returns:
        transform: callable (img, target) -> (img, target)
    """
    def transform(img, target):
        # Original size (W, H)
        orig_w, orig_h = img.size if not torch.is_tensor(img) else (img.shape[-1], img.shape[-2])

        # --------------------------
        # Case 1: resize with aspect ratio
        # --------------------------
        if img_size is not None:
            # Resize image while preserving aspect ratio
            img = F.resize(img, [img_size])

            # New size (W, H)
            new_w, new_h = img.size if not torch.is_tensor(img) else (img.shape[-1], img.shape[-2])

            # Scale bounding boxes
            if "boxes" in target:
                boxes = target["boxes"]
                if not torch.is_tensor(boxes):
                    boxes = torch.as_tensor(boxes, dtype=torch.float32)
                boxes = resize_boxes(boxes, (orig_h, orig_w), (new_h, new_w))
                target["boxes"] = boxes

            # Resize masks if present
            #if "masks" in target:
                #target["masks"] = F.resize(target["masks"], [new_h, new_w])
            if "masks" in target and target["masks"].numel() > 0:
                target["masks"] = F.resize(target["masks"], [new_h, new_w])
        # --------------------------
        # Case 2: skip resizing entirely
        # --------------------------
        else:
            # No resizing: just ensure correct tensor type
            if "boxes" in target and not torch.is_tensor(target["boxes"]):
                target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)

        # Convert to tensor before returning
        img = F.to_tensor(img)
        return img, target

    return transform

def get_cocodataset(is_train, is_val, args, output_format="torch", img_size=640):
    image_set = "train" if is_train else "val"
    num_classes, mode = {"coco": (91, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    with_masks = False #"mask" in args.model
    # def transform(img, target):
    #     img = F.to_tensor(img)
    #     return img, target
    ds = get_coco(
        root=args.data_path,
        image_set=image_set,
        transforms=make_train_transform_with_resize(img_size) if is_train else make_eval_transform_with_resize(img_size), #, #get_transform_new(is_train, args, img_size),
        mode=mode,
        use_v2=args.use_v2,
        with_masks=with_masks,
    )
    print(f"✅ COCO dataset loaded: {len(ds)} samples, {num_classes} classes, format: {output_format}, img_size: {img_size}")
    return ds, num_classes

   
def get_kittidataset(is_train, is_val, args, output_format="torch", img_size=640):
    """
    Get KITTI dataset with enhanced features from the updated KittiDataset class
    """
    rootPath = args.data_path
    
    # Enhanced KITTI dataset initialization with validation and improved features
    if is_val == True:
        transformfunc = get_transform(False, args, img_size)
        dataset = KittiDataset(
            root=rootPath, 
            train=True, 
            split='val', 
            transform=transformfunc,
            output_format=output_format,  # Use configurable output format
            validate_dataset=True   # Enable dataset validation
        )
    else:
        transformfunc = get_transform(True, args, img_size)  # Add augmentation
        dataset = KittiDataset(
            root=rootPath, 
            train=is_train, 
            split='train', 
            transform=transformfunc,
            output_format=output_format,  # Use configurable output format
            validate_dataset=True   # Enable dataset validation
        )
    
    num_classes = dataset.numclass
    print(f"✅ KITTI dataset loaded: {len(dataset)} samples, {num_classes} classes, format: {output_format}, img_size: {img_size}")
    return dataset, num_classes

def get_kittiyolodataset(is_train, is_val, args, output_format="yolo", img_size=640):
    """
    Get KITTI dataset with YOLO format output using enhanced KittiDataset
    """
    rootPath = args.data_path
    
    # Enhanced KITTI dataset with YOLO format output
    if is_val == True:
        # No augmentation for validation
        transformfunc = get_transform(False, args, img_size)
        dataset = KittiDataset(
            root=rootPath, 
            train=True, 
            split='val', 
            transform=transformfunc,  # Use transform function with img_size
            output_format=output_format,  # Use configurable output format (default: yolo)
            validate_dataset=True   # Enable dataset validation
        )
    else:
        # Training - transforms handled internally by dataset
        transformfunc = get_transform(True, args, img_size)  # Add augmentation
        dataset = KittiDataset(
            root=rootPath, 
            train=is_train, 
            split='train', 
            transform=transformfunc,  # Use transform function with img_size
            output_format=output_format,  # Use configurable output format (default: yolo)
            validate_dataset=True   # Enable dataset validation
        )
    
    num_classes = dataset.numclass
    print(f"✅ KITTI YOLO dataset loaded: {len(dataset)} samples, {num_classes} classes, format: {output_format}, img_size: {img_size}")
    return dataset, num_classes

import os
import json
import random
from pycocotools.coco import COCO
from torch.utils.data import Subset
import torch

def _split_coco_json(input_json, output_dir, val_ratio=0.2, seed=42):
    """
    Split a single COCO JSON file into train/val JSON files.

    Args:
        input_json (str): Path to full COCO-style annotation file.
        output_dir (str): Directory to save split files.
        val_ratio (float): Fraction of images to use for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        (str, str): paths to train_json, val_json
    """
    random.seed(seed)
    coco = COCO(input_json)
    img_ids = list(coco.imgs.keys())
    random.shuffle(img_ids)

    split_idx = int(len(img_ids) * (1 - val_ratio))
    train_ids, val_ids = img_ids[:split_idx], img_ids[split_idx:]

    def build_subset(ids):
        """Return a valid COCO subset dictionary."""
        subset = {
            "info": coco.dataset.get("info", {}),
            "licenses": coco.dataset.get("licenses", []),
            "categories": coco.dataset["categories"],
            "images": [coco.imgs[i] for i in ids],
            "annotations": [a for a in coco.dataset["annotations"] if a["image_id"] in ids],
        }
        return subset

    os.makedirs(output_dir, exist_ok=True)
    train_json = os.path.join(output_dir, "annotations_train_split.json")
    val_json   = os.path.join(output_dir, "annotations_val_split.json")

    with open(train_json, "w") as f:
        json.dump(build_subset(train_ids), f)
    with open(val_json, "w") as f:
        json.dump(build_subset(val_ids), f)

    print(f"✅ COCO split generated: {len(train_ids)} train / {len(val_ids)} val")
    print(f"   Train annotations: {train_json}")
    print(f"   Val annotations:   {val_json}")

    return train_json, val_json


def get_waymococodataset(is_train, is_val, data_path, annotationfile, output_format="torch", img_size=640, val_ratio=0.1):
    """
    Automatically detect or generate train/val splits for Waymo COCO dataset.

    Args:
        is_train (bool): whether to return the training dataset.
        is_val (bool):   whether to return the validation dataset.
        args:            must contain args.data_path (root dir) and args.annotationfile.
        output_format (str): description of output format, not used.
        img_size (int):  target resize for transforms.
        val_ratio (float): fraction of images to use for validation if splitting.
    """
    rootPath = data_path
    annotation = annotationfile

    # Define expected split paths inside rootPath
    train_split_path = os.path.join(rootPath, "annotations_train_split.json")
    val_split_path   = os.path.join(rootPath, "annotations_val_split.json")

    # --- Step 1: Detect if split files already exist ---
    if not (os.path.exists(train_split_path) and os.path.exists(val_split_path)):
        print("[INFO] Train/val split files not found. Generating new split...")
        _split_coco_json(annotation, rootPath, val_ratio=val_ratio)
    else:
        print("[INFO] Found existing train/val split files in rootPath. Reusing them.")

    # --- Step 2: Choose which split to load ---
    if is_val:
        ann_path = val_split_path
        transform = make_eval_transform_with_resize(img_size)
        split_name = "validation"
    else:
        ann_path = train_split_path
        transform = make_train_transform_with_resize(img_size)
        split_name = "training"

    # --- Step 3: Load dataset ---
    dataset = WaymoCOCODataset(rootPath, ann_path, train=is_train, transform=transform)
    num_classes = dataset.numclass
    print(f"✅ Waymo COCO {split_name} set loaded: {len(dataset)} samples, {num_classes} classes, img_size={img_size}")

    return dataset, num_classes

# def get_waymococodataset(is_train, is_val, args, output_format="torch", img_size=640):
#     rootPath=args.data_path
#     annotation=args.annotationfile
#     if is_val == True:
#         #annotation = os.path.join(rootPath, 'annotations_val20new.json') #'annotations_val50new.json'
#         #transformfunc=get_transform_new(False, args, img_size)
#         dataset = WaymoCOCODataset(rootPath, annotation, train=True, transform=make_eval_transform_with_resize(img_size)) #No augumentation
#     else: #Training
#         #annotation = os.path.join(rootPath, 'annotations_train200new.json') 
#         #transformfunc=get_transform_new(True, args, img_size) #add augumentation
#         dataset = WaymoCOCODataset(rootPath, annotation, train=is_train, transform=make_train_transform_with_resize(img_size)) #add augumentation
    
#     num_classes = dataset.numclass
#     print(f"✅ Waymo COCO dataset loaded: {len(dataset)} samples, {num_classes} classes, format: {output_format}, img_size: {img_size}")
#     return dataset, num_classes
    #mykitti = datasets.Kitti(root=rootPath, train= True, transform = get_transform(is_train, args), target_transform = None, download = False)

import yaml
from DeepDataMiningLearning.detection.dataset_yolo import YOLODataset
def get_yolodataset(is_train, is_val, args, output_format="yolo", img_size=640):
    rootPath=args.data_path #'/data/cmpe249-fa23/coco/'
    #annotation=args.annotationfile #'/data/cmpe249-fa23/coco/train2017.txt'
    dataset_cfgfile = os.path.join(os.path.dirname(__file__), 'dataset.yaml')
    with open(dataset_cfgfile, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        data['yaml_file'] = str(dataset_cfgfile)
        data['kpt_shape'] = [17, 3] #for keypoint
    pathsplits=args.data_path.split('/') 
    pathsplits=[x for x in pathsplits if x] #filter out empty string
    if is_val == True:
        if pathsplits[-1].endswith('coco'):
            annotation = os.path.join(rootPath, 'val2017.txt')
        else:
            annotation = os.path.join(rootPath, 'images_val.txt')
        yolodataset = YOLODataset(root=rootPath, annotation=annotation, train=False, transform=None, data=data, imgsz=img_size, classes=None,use_segments=False,use_keypoints=False)
    else: #training
        if pathsplits[-1].endswith('coco'):
            annotation = os.path.join(rootPath, 'train2017.txt')
        else:
            annotation = os.path.join(rootPath, 'images_train.txt')
        yolodataset = YOLODataset(root=rootPath, annotation=annotation, train=is_train, transform=None, data=data, imgsz=img_size, classes=None,use_segments=False,use_keypoints=False)
    num_classes = yolodataset.numclass
    print(f"✅ YOLO dataset loaded: {len(yolodataset)} samples, {num_classes} classes, format: {output_format}, img_size: {img_size}")
    return yolodataset, num_classes

from DeepDataMiningLearning.detection import utils
class args:
    data_path = '/data/cmpe249-fa23/coco/' #'/data/cmpe249-fa23/WaymoCOCO/' #'/data/cmpe249-fa23/coco/'
    annotationfile = '/data/cmpe249-fa23/coco/train2017.txt'
    weights = None
    test_only = True
    backend = 'PIL' #tensor
    use_v2 = False

if __name__ == "__main__":
    is_train =False
    is_val =True
    datasetname='yolo' #'waymococo' #'yolo'
    dataset, num_classes=get_dataset(datasetname, is_train, is_val, args, output_format="yolo")
    print("train set len:", len(dataset))
    test_sampler = torch.utils.data.SequentialSampler(dataset)
    train_collate_fn = utils.mycollate_fn
    data_loader_test = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=test_sampler, num_workers=1, collate_fn=train_collate_fn
    )
    for batch in data_loader_test:
        print(batch.keys()) #['img', 'bboxes', 'cls', 'batch_idx']
        break

    print(batch.keys())